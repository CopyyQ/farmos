import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from kaggrl.v2_tensorize import (
    ECONOMY_FEATURE_INDEX,
    UNIT_FEATURE_INDEX,
    Stage0AcceptanceError,
    collate_transitions,
    signed_log1p,
    tensorize_state,
    verify_stage0_acceptance,
)


def _tiles():
    return [[None for _ in range(10)] for _ in range(10)]


def _state(hands=0, *, money=200000, inventory=1000, step=100):
    own_hands = [[(i + 5) % 10, (i + 3) % 10] for i in range(hands)]
    inventories = [{"WHEAT": inventory}] + [{"WHEAT": inventory + i + 1} for i in range(hands)]
    own = {"money": money, "farmer": [4, 4], "hands": own_hands, "hires_today": hands,
           "unlocked_quadrants": ["NW", "NE"], "tiles": _tiles()}
    rival = {"money": 12345, "farmer": [8, 8], "hands": [[7, 8]], "hires_today": 1,
             "unlocked_quadrants": ["NW"], "tiles": _tiles()}
    own_units = [{"kind": "farmer", "index": 0, "position": [4, 4], "inventory": inventories[0]}]
    own_units += [{"kind": "hand", "index": i, "position": own_hands[i], "inventory": inventories[i + 1]}
                 for i in range(hands)]
    rival_units = [
        {"kind": "farmer", "index": 0, "position": [8, 8]},
        {"kind": "hand", "index": 0, "position": [7, 8]},
    ]
    return {
        "player": 0, "step": step, "day": step // 24, "hour": step % 24,
        "own": own, "rival": rival,
        "private": {"shed": {"WHEAT": inventory}, "seeds": {"WHEAT": inventory},
                    "inventories": inventories},
        "own_grid": deepcopy(own["tiles"]), "rival_grid": deepcopy(rival["tiles"]),
        "own_units": own_units, "rival_units": rival_units,
        "market": {"inventory": {"WHEAT": 10000}, "prices": {"WHEAT": 25}},
        "town": {"unlocked_shops": ["BAKERY", "BAKERY"]},
        "town_shops": ["BAKERY", "BAKERY"],
    }


def _row(hands, **kwargs):
    return {
        "state": _state(hands, **kwargs),
        "previous_effect": {},
        "canonical_action": {"farmer": {"op": "PASS", "item": None, "quantity": None, "raw": ["PASS"]},
                             "hands": [{"op": "PASS", "item": None, "quantity": None, "raw": ["PASS"]}
                                       for _ in range(hands)],
                             "market": [{"kind": "STOP_QUEUE", "op": None,
                                         "item": None, "quantity": None, "raw": []}]},
        "effects": {}, "terminal_result": 0, "final_margin": 0,
    }


def test_dynamic_padding_uses_batch_max_without_hand_truncation():
    batch = collate_transitions([_row(5), _row(22)])
    assert batch.own_units.shape[1] == 23
    assert batch.own_unit_mask[0].sum().item() == 6
    assert batch.own_unit_mask[1].sum().item() == 23
    assert batch.rival_unit_mask.sum(dim=1).tolist() == [2, 2]


def test_padding_does_not_change_real_unit_features():
    small = collate_transitions([_row(5)])
    mixed = collate_transitions([_row(5), _row(22)])
    assert torch.equal(small.own_units[0, :6], mixed.own_units[0, :6])
    assert torch.equal(small.own_unit_mask[0, :6], mixed.own_unit_mask[0, :6])


def test_large_numeric_values_remain_finite_and_distinguishable():
    a = tensorize_state(_state(1, money=200000, inventory=1000), {})
    b = tensorize_state(_state(1, money=400000, inventory=100000), {})
    money_idx = ECONOMY_FEATURE_INDEX["own_money"]
    wheat_idx = UNIT_FEATURE_INDEX["inventory:WHEAT"]
    assert torch.isfinite(a.economy).all() and torch.isfinite(b.economy).all()
    assert a.economy[money_idx] != b.economy[money_idx]
    assert a.own_units[0, wheat_idx] != b.own_units[0, wheat_idx]
    assert signed_log1p(100000) > signed_log1p(1000) > 0
    assert signed_log1p(-100000) < signed_log1p(-1000) < 0


def test_actor_index_features_support_large_dynamic_indices():
    state = tensorize_state(_state(32), {})
    idx = UNIT_FEATURE_INDEX["actor_index_log"]
    sin_idx = UNIT_FEATURE_INDEX["actor_index_sin"]
    assert state.own_units.shape[0] == 33
    assert math.isfinite(float(state.own_units[-1, idx]))
    assert state.own_units[1, idx] != state.own_units[-1, idx]
    assert state.own_units[1, sin_idx] != state.own_units[-1, sin_idx]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _accepted_stage0_root(tmp_path):
    live = tmp_path / "live_v2"
    manifests = live / "manifests"
    manifests.mkdir(parents=True)
    (live / "transitions.parquet").write_bytes(b"trusted-dataset")
    (manifests / "live_top10_snapshot.json").write_bytes(b"snapshot")
    (manifests / "trusted_corpus_manifest.json").write_bytes(b"corpus")
    from kaggrl.v2_tensorize import stage0_code_hashes
    artifacts = {
        "dataset_sha256": _sha(live / "transitions.parquet"),
        "selection_snapshot_sha256": _sha(manifests / "live_top10_snapshot.json"),
        "trusted_corpus_manifest_sha256": _sha(manifests / "trusted_corpus_manifest.json"),
        **stage0_code_hashes(),
    }
    marker = {"accepted": True, "counters": {"x": 0}, "artifacts": artifacts}
    (manifests / "STAGE0_ACCEPTED.json").write_text(json.dumps(marker), encoding="utf-8")
    return live


def test_stage1_gate_accepts_bound_stage0_marker_and_rejects_tamper(tmp_path):
    live = _accepted_stage0_root(tmp_path)
    marker = verify_stage0_acceptance(live)
    assert marker["accepted"] is True
    (live / "transitions.parquet").write_bytes(b"tampered")
    with pytest.raises(Stage0AcceptanceError):
        verify_stage0_acceptance(live)


def test_action_sequence_masks_follow_dynamic_hands_and_market_slots():
    first = _row(5)
    second = _row(22)
    second["canonical_action"]["market"] = [
        {"kind": "ORDER", "op": "HIRE", "item": None, "quantity": None, "raw": ["HIRE"]},
        {"kind": "ORDER", "op": "SELL", "item": "WHEAT", "quantity": 3,
         "raw": ["SELL", "WHEAT", 3]},
        {"kind": "STOP_QUEUE", "op": None, "item": None, "quantity": None, "raw": []},
    ]
    batch = collate_transitions([first, second])
    assert batch.hand_action_mask.shape == (2, 22)
    assert batch.hand_action_mask.sum(dim=1).tolist() == [5, 22]
    assert batch.market_slot_mask.shape == (2, 3)
    assert batch.market_slot_mask.sum(dim=1).tolist() == [1, 3]


def test_batch_retains_structured_states_for_mechanics_ledger():
    row = _row(3, step=321)
    batch = collate_transitions([row])
    assert len(batch.structured_states) == 1
    assert batch.structured_states[0]["step"] == 321
    assert len(batch.structured_states[0]["own_units"]) == 4


def test_contiguous_rows_derive_previous_action_and_effect_context():
    first = _row(1)
    second = _row(1)
    first.update({"episode_id": 77, "seat": 0, "step": 0})
    second.update({"episode_id": 77, "seat": 0, "step": 1})
    first["canonical_action"]["farmer"] = {
        "op": "EAST", "item": None, "quantity": None, "raw": ["EAST"]
    }
    first["effects"] = {"money_delta": 7}
    first.pop("previous_effect", None)
    second.pop("previous_effect", None)
    batch = collate_transitions([first, second])
    assert batch.previous_actions[1] == first["canonical_action"]
    assert batch.previous_effect[1].abs().sum().item() > 0
    assert batch.previous_action_global[1].abs().sum().item() >= 0


def test_noninitial_row_fails_closed_without_predecessor_context():
    row = _row(0)
    row.update({"episode_id": 88, "seat": 1, "step": 9})
    row.pop("previous_effect", None)
    with pytest.raises(ValueError, match="previous transition context"):
        collate_transitions([row])


def test_previous_unit_command_is_tensorized_per_current_actor():
    from kaggrl.v2_tensorize import PREV_UNIT_ACTION_FEATURE_INDEX

    previous = {
        "farmer": {"op": "EAST", "item": None, "quantity": None, "raw": ["EAST"]},
        "hands": [{"op": "PICKUP", "item": "WHEAT", "quantity": 1000,
                   "raw": ["PICKUP", "WHEAT", 1000]}],
        "market": [],
    }
    state = tensorize_state(_state(1), {}, previous)
    east = PREV_UNIT_ACTION_FEATURE_INDEX["op:EAST"]
    pickup = PREV_UNIT_ACTION_FEATURE_INDEX["op:PICKUP"]
    qty = PREV_UNIT_ACTION_FEATURE_INDEX["quantity_log"]
    assert state.previous_unit_actions[0, east].item() == 1.0
    assert state.previous_unit_actions[1, pickup].item() == 1.0
    assert state.previous_unit_actions[1, qty].item() == pytest.approx(signed_log1p(1000))


def test_day_reset_does_not_reuse_old_hand_command_identity():
    from kaggrl.v2_tensorize import PREV_UNIT_ACTION_FEATURE_INDEX

    previous = {"farmer": _unit("PASS"), "hands": [_unit("EAST")], "market": []}
    state = tensorize_state(_state(1), {"day_reset": True}, previous)
    none_idx = PREV_UNIT_ACTION_FEATURE_INDEX["none"]
    assert state.previous_unit_actions[1, none_idx].item() == 1.0


def _unit(op="PASS", item=None, quantity=None):
    return {"op": op, "item": item, "quantity": quantity, "raw": [op]}


def test_model_inputs_exclude_current_and_future_training_targets():
    batch = collate_transitions([_row(2)])
    inputs = batch.model_inputs()
    forbidden = {
        "canonical_actions", "effects_targets", "auxiliary_targets",
        "market_slot_mask", "terminal_money", "terminal_margin",
        "future_resource", "unit_task", "opponent_effect",
    }
    assert forbidden.isdisjoint(inputs)
    assert {"own_grid", "own_units", "previous_unit_actions",
            "previous_action_global", "previous_effect"}.issubset(inputs)


def test_commodity_features_are_entity_wise_not_only_flat_economy():
    from kaggrl.v2_tensorize import (
        COMMODITY_FEATURE_INDEX, COMMODITY_NAMES, COMMODITY_TO_INDEX,
    )
    state = tensorize_state(_state(1, inventory=4321), {})
    assert state.commodities.shape[0] == len(COMMODITY_NAMES)
    wheat = state.commodities[COMMODITY_TO_INDEX["WHEAT"]]
    assert wheat[COMMODITY_FEATURE_INDEX["item:WHEAT"]].item() == 1.0
    assert wheat[COMMODITY_FEATURE_INDEX["shed_quantity"]].item() > 0
    assert wheat[COMMODITY_FEATURE_INDEX["seed_quantity"]].item() > 0
    assert wheat[COMMODITY_FEATURE_INDEX["market_inventory"]].item() > 0


def test_economy_contains_hard_obligations_and_capacity_features():
    state = tensorize_state(_state(2, inventory=10), {})
    assert state.economy[ECONOMY_FEATURE_INDEX["next_hire_cost"]].item() == pytest.approx(signed_log1p(2))
    assert state.economy[ECONOMY_FEATURE_INDEX["next_land_cost"]].item() == pytest.approx(signed_log1p(2000))
    assert state.economy[ECONOMY_FEATURE_INDEX["shed_capacity"]].item() == pytest.approx(signed_log1p(100))
    assert state.economy[ECONOMY_FEATURE_INDEX["shed_free_space"]].item() == pytest.approx(signed_log1p(90))
    assert state.economy[ECONOMY_FEATURE_INDEX["market_slots_remaining"]].item() == pytest.approx(signed_log1p(10))


def test_unit_features_include_depot_distance_without_search_router():
    state = tensorize_state(_state(1), {})
    depot = UNIT_FEATURE_INDEX["depot_distance"]
    assert state.own_units[0, depot].item() == 0.0
    assert state.own_units[1, depot].item() > 0.0


def test_previous_observed_effect_is_encoded_per_current_unit():
    from kaggrl.v2_tensorize import PREV_UNIT_EFFECT_FEATURE_INDEX

    effect = {
        "action_evidence": [
            {"actor": "farmer", "op": "PICKUP", "status": "confirmed",
             "observed": {"inventory_delta": {"WHEAT": 6}}},
            {"actor": "hand:0", "op": "WATER", "status": "failed", "observed": {}},
        ],
        "unit_position_delta": {"farmer": [1, 0], "hand:0": [0, 0]},
    }
    state = tensorize_state(_state(1), effect, {})
    confirmed = PREV_UNIT_EFFECT_FEATURE_INDEX["status:confirmed"]
    failed = PREV_UNIT_EFFECT_FEATURE_INDEX["status:failed"]
    wheat = PREV_UNIT_EFFECT_FEATURE_INDEX["inventory_delta:WHEAT"]
    dx = PREV_UNIT_EFFECT_FEATURE_INDEX["dx"]
    assert state.previous_unit_effects[0, confirmed].item() == 1.0
    assert state.previous_unit_effects[0, wheat].item() == pytest.approx(signed_log1p(6))
    assert state.previous_unit_effects[0, dx].item() == pytest.approx(1.0)
    assert state.previous_unit_effects[1, failed].item() == 1.0


def test_effective_supervision_turns_failed_sell_into_nop_slot():
    row = _row(0, inventory=3)
    row["canonical_action"]["farmer"] = {
        "op": "PICKUP", "item": "WHEAT", "quantity": 3,
        "raw": ["PICKUP", "WHEAT", 3],
    }
    row["canonical_action"]["market"] = [
        {"kind": "ORDER", "op": "SELL", "item": "WHEAT", "quantity": 1,
         "raw": ["SELL", "WHEAT", 1]},
        {"kind": "STOP_QUEUE", "op": None, "item": None, "quantity": None, "raw": []},
    ]
    batch = collate_transitions([row])
    assert batch.canonical_actions[0]["market"][0]["kind"] == "NOP_SLOT"


def test_effective_supervision_clamps_sell_to_known_shed_inventory():
    row = _row(0, inventory=10)
    row["canonical_action"]["market"] = [
        {"kind": "ORDER", "op": "SELL", "item": "WHEAT", "quantity": 1000,
         "raw": ["SELL", "WHEAT", 1000]},
        {"kind": "STOP_QUEUE", "op": None, "item": None, "quantity": None, "raw": []},
    ]
    batch = collate_transitions([row])
    sell = batch.canonical_actions[0]["market"][0]
    assert sell["op"] == "SELL"
    assert sell["quantity"] == 10
    assert sell["raw"] == ["SELL", "WHEAT", 10]


def test_effective_supervision_sees_unit_deposit_before_market_sell():
    row = _row(0, inventory=0)
    row["state"]["private"]["inventories"][0] = {"WHEAT": 1}
    row["state"]["own_units"][0]["inventory"] = {"WHEAT": 1}
    row["canonical_action"]["farmer"] = {
        "op": "PLACE", "item": "WHEAT", "quantity": 1,
        "raw": ["PLACE", "WHEAT", 1],
    }
    row["canonical_action"]["market"] = [
        {"kind": "ORDER", "op": "SELL", "item": "WHEAT", "quantity": 1,
         "raw": ["SELL", "WHEAT", 1]},
        {"kind": "STOP_QUEUE", "op": None, "item": None, "quantity": None, "raw": []},
    ]
    batch = collate_transitions([row])
    sell = batch.canonical_actions[0]["market"][0]
    assert sell["op"] == "SELL"
    assert sell["quantity"] == 1


def test_effective_supervision_allows_same_turn_buy_then_sell():
    row = _row(0, inventory=0)
    row["canonical_action"]["market"] = [
        {"kind": "ORDER", "op": "BUY_PRODUCT", "item": "WHEAT", "quantity": 60,
         "raw": ["BUY_PRODUCT", "WHEAT", 60]},
        {"kind": "ORDER", "op": "SELL", "item": "WHEAT", "quantity": 1000,
         "raw": ["SELL", "WHEAT", 1000]},
        {"kind": "STOP_QUEUE", "op": None, "item": None, "quantity": None, "raw": []},
    ]
    batch = collate_transitions([row])
    assert batch.canonical_actions[0]["market"][0]["op"] == "BUY_PRODUCT"
    sell = batch.canonical_actions[0]["market"][1]
    assert sell["kind"] == "ORDER"
    assert sell["op"] == "SELL"
    assert sell["quantity"] == 60
