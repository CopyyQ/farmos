from dataclasses import asdict

import torch

from kaggrl.constants import ITEM_TO_ID
from kaggrl.v2_model import RecurrentIntentPolicy
from kaggrl.v2_observation import normalize_observation
from kaggrl.v2_quantity import encode_quantity
from kaggrl.v2_tensorize import collate_transitions


def _structured(hands: int, money: int = 3000):
    own_tiles = [[None for _ in range(10)] for _ in range(10)]
    rival_tiles = [[None for _ in range(10)] for _ in range(10)]
    hand_pos = [[i % 10, (i // 10) % 10] for i in range(hands)]
    obs = {
        "player": 0, "step": 123, "day": 5, "hour": 3,
        "farms": [
            {"money": money, "farmer": [4, 4], "hands": hand_pos,
             "hires_today": 0, "unlocked_quadrants": ["NW"], "tiles": own_tiles},
            {"money": 2500, "farmer": [7, 7], "hands": [],
             "hires_today": 0, "unlocked_quadrants": ["NW"], "tiles": rival_tiles},
        ],
        "private": {"shed": {"WHEAT": 5000}, "seeds": {"WHEAT": 50},
                    "inventories": [{} for _ in range(hands + 1)]},
        "market": {"inventory": {"WHEAT": 10000}, "prices": {"WHEAT": 25}},
        "town": {"unlocked_shops": []},
    }
    return asdict(normalize_observation(obs))


def _unit(op="PASS", item=None, quantity=None):
    return {"op": op, "item": item, "quantity": quantity, "raw": [op]}


def _stop():
    return {"kind": "STOP_QUEUE", "op": None, "item": None, "quantity": None, "raw": []}


def _order(op, item=None, quantity=None):
    raw = [op] if item is None else [op, item, quantity]
    return {"kind": "ORDER", "op": op, "item": item, "quantity": quantity, "raw": raw}


def _row(hands: int, farmer=None, hand_actions=None, market=None):
    farmer = farmer or _unit()
    hand_actions = hand_actions or [_unit() for _ in range(hands)]
    market = market or [_stop()]
    return {
        "state": _structured(hands),
        "previous_effect": {},
        "canonical_action": {"farmer": farmer, "hands": hand_actions, "market": market},
        "effects": {}, "terminal_result": 0, "final_margin": 0,
    }


def test_dynamic_decoder_emits_exact_current_hand_count():
    rows = [_row(0), _row(17), _row(32)]
    batch = collate_transitions(rows)
    model = RecurrentIntentPolicy().eval()
    with torch.no_grad():
        output = model.forward_sequence(batch, teacher_actions=batch.canonical_actions)
    assert [len(row.hands) for row in output.rows] == [0, 17, 32]
    assert all(1 <= len(row.market) <= 10 for row in output.rows)


def test_teacher_forcing_changes_next_decision_conditioning_and_ledger():
    row = _row(1, farmer=_unit("EAST"))
    batch = collate_transitions([row])
    model = RecurrentIntentPolicy().eval()
    with torch.no_grad():
        model.unit_op_head.weight.zero_()
        model.unit_op_head.bias.zero_()
        model.unit_op_head.bias[model.unit_op_to_id["PASS"]] = 9.0
        teacher = model.forward_sequence(batch, teacher_actions=batch.canonical_actions)
        free = model.forward_sequence(batch, teacher_actions=None)
    teacher_hand = teacher.rows[0].trace[1]
    free_hand = free.rows[0].trace[1]
    assert teacher_hand.previous_semantic == "U:EAST"
    assert free_hand.previous_semantic == "U:PASS"
    assert teacher_hand.farmer_position_before == (5, 4)
    assert free_hand.farmer_position_before == (4, 4)


def test_teacher_market_hire_updates_ledger_before_next_slot():
    row = _row(0, market=[_order("HIRE"), _stop()])
    batch = collate_transitions([row])
    model = RecurrentIntentPolicy().eval()
    with torch.no_grad():
        output = model.forward_sequence(batch, teacher_actions=batch.canonical_actions)
    market_traces = [t for t in output.rows[0].trace if t.domain == "market"]
    assert market_traces[0].chosen_semantic == "M:HIRE"
    assert market_traces[1].hires_today_before == 1
    assert market_traces[1].previous_semantic == "M:HIRE"


def test_teacher_atomic_plant_oversubscription_blocks_all_plant_legality():
    row = _row(
        1,
        farmer=_unit("PLANT", "CARROT"),
        hand_actions=[_unit("PLANT", "CARROT")],
    )
    row["state"]["private"]["seeds"] = {"CARROT": 1}
    batch = collate_transitions([row])
    model = RecurrentIntentPolicy().eval()
    with torch.no_grad():
        output = model.forward_sequence(
            batch, teacher_actions=batch.canonical_actions,
        )
    carrot_id = ITEM_TO_ID["CARROT"]
    farmer = output.rows[0].farmer
    hand = output.rows[0].hands[0]
    assert farmer.chosen_action["op"] == "PLANT"
    assert hand.chosen_action["op"] == "PLANT"
    assert farmer.legal_item_mask[carrot_id].item() is False
    assert hand.legal_item_mask[carrot_id].item() is False


def test_teacher_quantity_uses_exact_decimal_token_stream():
    row = _row(0, farmer=_unit("PICKUP", "WHEAT", 1000))
    batch = collate_transitions([row])
    model = RecurrentIntentPolicy().eval()
    with torch.no_grad():
        output = model.forward_sequence(batch, teacher_actions=batch.canonical_actions)
    decision = output.rows[0].farmer
    assert decision.quantity_tokens == encode_quantity(1000)
    assert decision.quantity_logits is not None
    assert decision.quantity_logits.shape[0] == len(encode_quantity(1000))


def test_auxiliary_heads_and_joint_decoder_backpropagate_finite_gradients():
    batch = collate_transitions([_row(2), _row(5)])
    model = RecurrentIntentPolicy()
    output = model.forward_sequence(batch, teacher_actions=batch.canonical_actions)
    loss = output.aux.effect.square().mean()
    loss = loss + output.aux.future_resource.square().mean()
    loss = loss + output.aux.unit_task.square().mean()
    loss = loss + output.aux.terminal_money.square().mean()
    for row in output.rows:
        loss = loss + row.farmer.op_logits.square().mean()
        for decision in row.hands:
            loss = loss + decision.op_logits.square().mean()
        for decision in row.market:
            loss = loss + decision.op_logits.square().mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_model_parameter_budget_remains_below_two_point_five_million():
    model = RecurrentIntentPolicy()
    params = sum(p.numel() for p in model.parameters())
    assert params < 2_500_000


def test_teacher_mix_one_preserves_teacher_previous_semantic_exactly():
    row = _row(1, farmer=_unit("EAST"))
    batch = collate_transitions([row])
    model = RecurrentIntentPolicy().eval()
    with torch.no_grad():
        baseline = model.forward_sequence(
            batch, teacher_actions=batch.canonical_actions,
        )
        mixed = model.forward_sequence(
            batch, teacher_actions=batch.canonical_actions,
            teacher_mix_probability=1.0,
        )
    assert mixed.rows[0].trace == baseline.rows[0].trace
    assert torch.equal(
        mixed.rows[0].hands[0].op_logits,
        baseline.rows[0].hands[0].op_logits,
    )


def test_teacher_mix_zero_uses_model_semantic_and_model_conditioning_ledger():
    import numpy as np

    row = _row(1, farmer=_unit("EAST"))
    batch = collate_transitions([row])
    model = RecurrentIntentPolicy().eval()
    with torch.no_grad():
        model.unit_op_head.weight.zero_()
        model.unit_op_head.bias.zero_()
        model.unit_op_head.bias[model.unit_op_to_id["PASS"]] = 9.0
        mixed = model.forward_sequence(
            batch,
            teacher_actions=batch.canonical_actions,
            teacher_mix_probability=0.0,
            conditioning_rng=np.random.default_rng(123),
        )
    farmer = mixed.rows[0].farmer
    hand_trace = mixed.rows[0].trace[1]
    assert farmer.chosen_action["op"] == "EAST"
    assert hand_trace.previous_semantic == "U:PASS"
    assert hand_trace.farmer_position_before == (4, 4)
