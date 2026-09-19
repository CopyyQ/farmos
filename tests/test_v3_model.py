from dataclasses import asdict

import numpy as np
import torch

from kaggrl.v2_observation import normalize_observation
from kaggrl.v2_tensorize import collate_transitions
from kaggrl.v3_model import TemporalIntentPolicy


def _unit(op="PASS"):
    return {"op": op, "item": None, "quantity": None, "raw": [op]}


def _stop():
    return {
        "kind": "STOP_QUEUE", "op": None,
        "item": None, "quantity": None, "raw": [],
    }


def _structured(hands: int, step: int = 7):
    tiles = [[None for _ in range(10)] for _ in range(10)]
    hand_positions = [[i % 10, (i // 10) % 10] for i in range(hands)]
    obs = {
        "player": 0, "step": step, "day": step // 24, "hour": step % 24,
        "farms": [
            {"money": 3000, "farmer": [4, 4], "hands": hand_positions,
             "hires_today": hands, "unlocked_quadrants": ["NW"], "tiles": tiles},
            {"money": 2500, "farmer": [8, 8], "hands": [],
             "hires_today": 0, "unlocked_quadrants": ["NW"], "tiles": tiles},
        ],
        "private": {
            "shed": {"WHEAT": 20}, "seeds": {"WHEAT": 8},
            "inventories": [{"WHEAT": 2}] + [{"WHEAT": 1}] * hands,
        },
        "market": {"inventory": {"WHEAT": 10000}, "prices": {"WHEAT": 25}},
        "town": {"unlocked_shops": []},
    }
    return asdict(normalize_observation(obs))


def _row(hands: int, step: int = 7):
    action = {
        "farmer": _unit("EAST"),
        "hands": [_unit() for _ in range(hands)],
        "market": [_stop()],
    }
    return {
        "state": _structured(hands, step),
        "previous_action": {}, "previous_effect": {},
        "canonical_action": action, "effects": {},
        "terminal_result": 0, "final_margin": 0,
    }

def test_same_episode_continuation_differs_from_forced_reset():
    torch.manual_seed(21)
    model = TemporalIntentPolicy().eval()
    batch = collate_transitions([_row(2)])
    first = model.sample_step(
        batch, None, np.random.default_rng(1), deterministic=True,
    )
    carried = model.sample_step(
        batch, first.temporal_state, np.random.default_rng(2), deterministic=True,
    )
    reset = model.sample_step(
        batch, None, np.random.default_rng(2), deterministic=True,
    )
    assert not torch.allclose(carried.fused_temporal, reset.fused_temporal)


def test_dynamic_hands_0_17_32_64_keep_real_outputs():
    torch.manual_seed(22)
    model = TemporalIntentPolicy().eval()
    for hands in (0, 17, 32, 64):
        batch = collate_transitions([_row(hands)])
        out = model.sample_step(
            batch, None, np.random.default_rng(3), deterministic=True,
        )
        assert len(out.rows[0].hands) == hands

def test_teacher_and_sample_paths_share_same_temporal_transition():
    torch.manual_seed(23)
    model = TemporalIntentPolicy().eval()
    batch = collate_transitions([_row(2)])
    sampled = model.sample_step(
        batch, None, np.random.default_rng(4), deterministic=True,
    )
    taught = model.teacher_step(batch, batch.canonical_actions, None)
    assert torch.allclose(sampled.fused_temporal, taught.fused_temporal)
    assert torch.allclose(sampled.intent, taught.intent)
    assert torch.allclose(
        sampled.temporal_state.memory, taught.temporal_state.memory,
    )


def test_v3_torch_trace_exposes_raw_and_masked_farmer_decision():
    torch.manual_seed(24)
    model = TemporalIntentPolicy().eval()
    batch = collate_transitions([_row(2)])
    traced = model.trace_sample_step(
        batch, None, np.random.default_rng(5), deterministic=True,
    )
    farmer = traced["decisions"][0]
    assert farmer["actor"] == "farmer"
    assert len(farmer["raw_logits"]) == len(farmer["legal_mask"])
    assert len(farmer["masked_logits"]) == len(farmer["raw_logits"])
    assert farmer["chosen_op"] in farmer["ops"]
    assert farmer["decision_reason"] == "model_argmax"


def test_strategy_slot_conditions_intent_without_changing_temporal_state():
    torch.manual_seed(25)
    model = TemporalIntentPolicy(strategy_count=3).eval()
    with torch.no_grad():
        model.strategy_embedding.weight.zero_()
        model.strategy_embedding.weight[1, 0] = 1.0
    batch = collate_transitions([_row(0, step=0)])
    slot0 = torch.tensor([0], dtype=torch.long)
    slot1 = torch.tensor([1], dtype=torch.long)
    out0 = model.sample_step(
        batch, None, np.random.default_rng(6), deterministic=True,
        strategy_slots=slot0,
    )
    out1 = model.sample_step(
        batch, None, np.random.default_rng(6), deterministic=True,
        strategy_slots=slot1,
    )
    assert torch.allclose(out0.fused_temporal, out1.fused_temporal)
    assert torch.allclose(
        out0.temporal_state.memory, out1.temporal_state.memory,
    )
    assert not torch.allclose(out0.intent, out1.intent)


def test_strategy_conditioning_rejects_missing_slot_when_enabled():
    model = TemporalIntentPolicy(strategy_count=2).eval()
    batch = collate_transitions([_row(0, step=0)])
    try:
        model.sample_step(
            batch, None, np.random.default_rng(7), deterministic=True,
        )
    except ValueError as exc:
        assert "strategy" in str(exc).lower()
    else:
        raise AssertionError("strategy-enabled policy must require a slot")


def test_v3_teacher_step_supports_model_previous_semantic_conditioning():
    torch.manual_seed(26)
    model = TemporalIntentPolicy().eval()
    batch = collate_transitions([_row(1)])
    with torch.no_grad():
        model.unit_op_head.weight.zero_()
        model.unit_op_head.bias.zero_()
        model.unit_op_head.bias[model.unit_op_to_id["PASS"]] = 9.0
        out = model.teacher_step(
            batch,
            batch.canonical_actions,
            None,
            teacher_mix_probability=0.0,
            conditioning_rng=np.random.default_rng(9),
        )
    assert out.rows[0].farmer.chosen_action["op"] == "EAST"
    assert out.rows[0].trace[1].previous_semantic == "U:PASS"


def test_v3_teacher_mix_zero_is_deterministic_with_same_conditioning_seed():
    torch.manual_seed(26)
    model = TemporalIntentPolicy().eval()
    batch = collate_transitions([_row(2)])
    with torch.no_grad():
        model.unit_op_head.weight.zero_()
        model.unit_op_head.bias.zero_()
        model.unit_op_head.bias[model.unit_op_to_id["PASS"]] = 8.0
        a = model.teacher_step(
            batch, batch.canonical_actions, None,
            teacher_mix_probability=0.0,
            conditioning_rng=np.random.default_rng(1234),
        )
        b = model.teacher_step(
            batch, batch.canonical_actions, None,
            teacher_mix_probability=0.0,
            conditioning_rng=np.random.default_rng(1234),
        )
    assert a.rows[0].trace == b.rows[0].trace
    assert torch.equal(a.rows[0].hands[0].op_logits, b.rows[0].hands[0].op_logits)
    assert a.rows[0].trace[1].previous_semantic == "U:PASS"
    # teacher_mix=0 must advance the conditioning ledger with the sampled
    # PASS action, not with the teacher EAST target.
    assert a.rows[0].trace[1].farmer_position_before == (4, 4)


def test_v3_teacher_mix_one_preserves_existing_teacher_path():
    torch.manual_seed(27)
    model = TemporalIntentPolicy().eval()
    batch = collate_transitions([_row(1)])
    with torch.no_grad():
        baseline = model.teacher_step(batch, batch.canonical_actions, None)
        mixed = model.teacher_step(
            batch, batch.canonical_actions, None,
            teacher_mix_probability=1.0,
            conditioning_rng=np.random.default_rng(55),
        )
    assert baseline.rows[0].trace == mixed.rows[0].trace
    assert torch.equal(
        baseline.rows[0].market[0].op_logits,
        mixed.rows[0].market[0].op_logits,
    )
