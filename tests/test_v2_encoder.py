from copy import deepcopy

import torch

from kaggrl.v2_encoder import RecurrentCore, StateEncoder
from kaggrl.v2_tensorize import collate_transitions


def _tiles():
    return [[None for _ in range(10)] for _ in range(10)]


def _state(hands=2):
    hand_pos = [[3 + i, 4] for i in range(hands)]
    own_grid = _tiles(); rival_grid = _tiles()
    own = {"money": 5000, "farmer": [4, 4], "hands": deepcopy(hand_pos), "hires_today": hands,
           "unlocked_quadrants": ["NW", "NE"], "tiles": own_grid}
    rival = {"money": 4500, "farmer": [8, 8], "hands": [[7, 8]], "hires_today": 1,
             "unlocked_quadrants": ["NW"], "tiles": rival_grid}
    inventories = [{"WHEAT": 2}] + [{"WHEAT": i + 1} for i in range(hands)]
    units = [{"kind": "farmer", "index": 0, "position": [4, 4], "inventory": inventories[0]}]
    units += [{"kind": "hand", "index": i, "position": hand_pos[i], "inventory": inventories[i + 1]}
              for i in range(hands)]
    return {"player": 0, "step": 100, "day": 4, "hour": 4, "own": own, "rival": rival,
            "private": {"shed": {}, "seeds": {}, "inventories": inventories},
            "own_grid": own_grid, "rival_grid": rival_grid, "own_units": units,
            "rival_units": [{"kind": "farmer", "index": 0, "position": [8, 8]},
                            {"kind": "hand", "index": 0, "position": [7, 8]}],
            "market": {"inventory": {"WHEAT": 10000}, "prices": {"WHEAT": 25}},
            "town": {"unlocked_shops": []}, "town_shops": []}


def _row(state):
    hand_count = len(state["own_units"]) - 1
    return {"state": state, "previous_effect": {},
            "canonical_action": {
                "farmer": {"op": "PASS", "item": None, "quantity": None, "raw": ["PASS"]},
                "hands": [{"op": "PASS", "item": None, "quantity": None, "raw": ["PASS"]}
                          for _ in range(hand_count)],
                "market": [{"kind": "STOP_QUEUE", "op": None, "item": None,
                            "quantity": None, "raw": []}],
            }, "effects": {}}


def test_encoder_is_sensitive_to_spatial_geometry_not_only_aggregate_counts():
    torch.manual_seed(0)
    north = _state(2); south = deepcopy(north)
    north["own_grid"][2][4] = {"kind": "PLANT", "crop": "WHEAT", "yield_units": 3}
    south["own_grid"][6][4] = {"kind": "PLANT", "crop": "WHEAT", "yield_units": 3}
    north["own"]["tiles"] = deepcopy(north["own_grid"])
    south["own"]["tiles"] = deepcopy(south["own_grid"])
    batch = collate_transitions([_row(north), _row(south)])
    model = StateEncoder().eval()
    out = model(batch)
    assert out.fused.shape == (2, 256)
    assert not torch.allclose(out.fused[0], out.fused[1], atol=1e-7, rtol=0)


def test_padding_from_larger_batch_does_not_change_real_unit_context():
    torch.manual_seed(1)
    model = StateEncoder().eval()
    small = collate_transitions([_row(_state(2))])
    mixed = collate_transitions([_row(_state(2)), _row(_state(22))])
    with torch.no_grad():
        a = model(small)
        b = model(mixed)
    assert torch.allclose(a.own_unit_ctx[0, :3], b.own_unit_ctx[0, :3], atol=2e-6, rtol=2e-6)
    assert torch.allclose(a.tile_ctx[0], b.tile_ctx[0], atol=2e-6, rtol=2e-6)
    assert torch.allclose(a.fused[0], b.fused[0], atol=2e-6, rtol=2e-6)


def test_reordering_real_hands_reorders_context_but_preserves_global_state():
    torch.manual_seed(2)
    model = StateEncoder().eval()
    first = _state(2)
    second = deepcopy(first)
    second["own_units"][1], second["own_units"][2] = second["own_units"][2], second["own_units"][1]
    with torch.no_grad():
        a = model(collate_transitions([_row(first)]))
        b = model(collate_transitions([_row(second)]))
    assert torch.allclose(a.own_unit_ctx[0, 0], b.own_unit_ctx[0, 0], atol=1e-6, rtol=1e-6)
    assert torch.allclose(a.own_unit_ctx[0, 1], b.own_unit_ctx[0, 2], atol=1e-6, rtol=1e-6)
    assert torch.allclose(a.own_unit_ctx[0, 2], b.own_unit_ctx[0, 1], atol=1e-6, rtol=1e-6)
    assert torch.allclose(a.fused[0], b.fused[0], atol=1e-6, rtol=1e-6)


def test_recurrent_core_reset_removes_prior_episode_history():
    torch.manual_seed(3)
    core = RecurrentCore().eval()
    first_obs = torch.randn(2, 256)
    unrelated = torch.randn(2, 256)
    with torch.no_grad():
        clean = core.step(first_obs, core.zero_state(2, first_obs.device, first_obs.dtype))
        prior = core.step(unrelated, core.zero_state(2, first_obs.device, first_obs.dtype))
        assert not torch.allclose(clean[0], core.step(first_obs, (prior[0], prior[1]))[0])
        reset = core.step(first_obs, core.zero_state(2, first_obs.device, first_obs.dtype))
    for expected, actual in zip(clean, reset):
        assert torch.equal(expected, actual)
    assert clean[0].shape == (2, 256) and clean[2].shape == (2, 128)


def test_encoder_plus_recurrent_core_stays_within_stage1_parameter_budget():
    encoder = StateEncoder()
    core = RecurrentCore()
    params = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in core.parameters())
    assert 500_000 < params < 1_300_000


def test_previous_requested_action_changes_unit_and_global_representation():
    torch.manual_seed(4)
    base = _state(2)
    pass_row = _row(deepcopy(base))
    east_row = _row(deepcopy(base))
    pass_row["previous_action"] = {
        "farmer": {"op": "PASS", "item": None, "quantity": None, "raw": ["PASS"]},
        "hands": [
            {"op": "PASS", "item": None, "quantity": None, "raw": ["PASS"]},
            {"op": "PASS", "item": None, "quantity": None, "raw": ["PASS"]},
        ],
        "market": [{"kind": "STOP_QUEUE", "op": None, "item": None, "quantity": None, "raw": []}],
    }
    east_row["previous_action"] = deepcopy(pass_row["previous_action"])
    east_row["previous_action"]["farmer"] = {
        "op": "EAST", "item": None, "quantity": None, "raw": ["EAST"]
    }
    east_row["previous_action"]["market"] = [
        {"kind": "ORDER", "op": "HIRE", "item": None, "quantity": None, "raw": ["HIRE"]},
        {"kind": "STOP_QUEUE", "op": None, "item": None, "quantity": None, "raw": []},
    ]
    model = StateEncoder().eval()
    with torch.no_grad():
        a = model(collate_transitions([pass_row]))
        b = model(collate_transitions([east_row]))
    assert not torch.allclose(a.own_unit_ctx[0, 0], b.own_unit_ctx[0, 0], atol=1e-7, rtol=0)
    assert not torch.allclose(a.fused[0], b.fused[0], atol=1e-7, rtol=0)


def test_commodity_entity_channel_changes_fused_representation():
    torch.manual_seed(5)
    batch = collate_transitions([_row(_state(2))])
    model = StateEncoder().eval()
    with torch.no_grad():
        baseline = model(batch).fused.clone()
        batch.commodities = batch.commodities.clone()
        batch.commodities[..., -1] += 3.0
        changed = model(batch).fused
    assert not torch.allclose(baseline, changed, atol=1e-7, rtol=0)


def test_previous_observed_unit_effect_changes_unit_context():
    torch.manual_seed(6)
    clean = _row(_state(1))
    effected = _row(deepcopy(clean["state"]))
    effected["previous_effect"] = {
        "action_evidence": [{"actor": "farmer", "op": "PICKUP", "status": "confirmed",
                             "observed": {"inventory_delta": {"WHEAT": 4}}}],
        "unit_position_delta": {"farmer": [1, 0]},
    }
    model = StateEncoder().eval()
    with torch.no_grad():
        a = model(collate_transitions([clean]))
        b = model(collate_transitions([effected]))
    assert not torch.allclose(a.own_unit_ctx[0, 0], b.own_unit_ctx[0, 0], atol=1e-7, rtol=0)
