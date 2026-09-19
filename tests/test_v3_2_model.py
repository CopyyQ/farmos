from dataclasses import asdict

import numpy as np
import torch

from kaggrl.constants import ITEM_TO_ID
from kaggrl.v2_observation import normalize_observation
from kaggrl.v2_quantity import DIGIT_OFFSET, END_ID
from kaggrl.v2_tensorize import collate_transitions
from kaggrl.v3_2_model import ACTIVE_MARKET_OPS, TemporalIntentPolicyV32


def _unit(op="PASS", item=None, quantity=None):
    return {"op": op, "item": item, "quantity": quantity, "raw": [op]}


def _stop():
    return {
        "kind": "STOP_QUEUE", "op": None,
        "item": None, "quantity": None, "raw": [],
    }


def _nop():
    return {
        "kind": "NOP_SLOT", "op": None,
        "item": None, "quantity": None, "raw": [],
    }


def _structured(*, money=3000, hands=0, land_exhausted=False):
    tiles = [[None for _ in range(10)] for _ in range(10)]
    hand_positions = [[4, 4] for _ in range(hands)]
    quadrants = ["NW", "NE", "SW", "SE"] if land_exhausted else ["NW"]
    obs = {
        "player": 0, "step": 0, "day": 0, "hour": 0,
        "farms": [
            {
                "money": money, "farmer": [4, 4], "hands": hand_positions,
                "hires_today": 0, "unlocked_quadrants": quadrants,
                "tiles": tiles,
            },
            {
                "money": 3000, "farmer": [7, 7], "hands": [],
                "hires_today": 0, "unlocked_quadrants": ["NW"],
                "tiles": [[None for _ in range(10)] for _ in range(10)],
            },
        ],
        "private": {
            "shed": {},
            "seeds": {
                "WHEAT": 0, "CARROT": 0, "TOMATO": 0,
                "STRAWBERRY": 0, "MELON": 0,
            },
            "inventories": [{} for _ in range(hands + 1)],
        },
        "market": {
            "inventory": {
                "WHEAT": 100, "CARROT": 100, "TOMATO": 100,
                "STRAWBERRY": 100, "MELON": 100, "EGG": 100,
                "MILK": 100, "WOOL": 100, "FERTILIZER": 100,
            },
            "prices": {
                "WHEAT": 10, "CARROT": 20, "TOMATO": 50,
                "STRAWBERRY": 100, "MELON": 80, "EGG": 10,
                "MILK": 10, "WOOL": 10, "FERTILIZER": 10,
            },
        },
        "town": {"unlocked_shops": []},
    }
    return asdict(normalize_observation(obs))


def _row(*, money=3000, hands=0, land_exhausted=False, market=None):
    action = {
        "farmer": _unit(),
        "hands": [_unit() for _ in range(hands)],
        "market": market or [_stop()],
    }
    return {
        "state": _structured(
            money=money, hands=hands, land_exhausted=land_exhausted,
        ),
        "previous_action": {},
        "previous_effect": {},
        "canonical_action": action,
        "effects": {},
        "terminal_result": 0,
        "final_margin": 0,
    }


def _force_hierarchical_market(model, *, stop_bias, continue_bias, active_op):
    with torch.no_grad():
        model.market_continue_head.weight.zero_()
        model.market_continue_head.bias[:] = torch.tensor(
            [float(stop_bias), float(continue_bias)]
        )
        model.market_active_op_head.weight.zero_()
        model.market_active_op_head.bias.zero_()
        model.market_active_op_head.bias[
            model.market_active_op_to_id[active_op]
        ] = 20.0


def test_v32_stop_head_terminates_before_active_operation():
    model = TemporalIntentPolicyV32(strategy_count=2).eval()
    _force_hierarchical_market(
        model, stop_bias=20.0, continue_bias=-20.0, active_op="HIRE",
    )
    batch = collate_transitions([_row()])
    output = model.sample_step(
        batch, None, np.random.default_rng(1), deterministic=True,
        strategy_slots=torch.tensor([0]),
    )
    assert len(output.rows[0].market) == 1
    assert output.rows[0].market[0].chosen_action["kind"] == "STOP_QUEUE"
    assert output.rows[0].market[0].continue_logits is not None


def test_v32_continue_chooses_only_active_market_vocabulary():
    model = TemporalIntentPolicyV32(strategy_count=2).eval()
    _force_hierarchical_market(
        model, stop_bias=-20.0, continue_bias=20.0, active_op="HIRE",
    )
    batch = collate_transitions([_row()])
    output = model.sample_step(
        batch, None, np.random.default_rng(2), deterministic=True,
        strategy_slots=torch.tensor([0]),
    )
    first = output.rows[0].market[0]
    assert first.chosen_action["op"] == "HIRE"
    assert first.chosen_action["kind"] == "ORDER"
    assert first.chosen_action["kind"] != "NOP_SLOT"
    assert first.op_logits.numel() == len(ACTIVE_MARKET_OPS)
    assert first.legal_continue_mask.tolist() == [True, True]


def test_v32_continue_falls_back_to_nop_when_no_economic_market_op_is_legal():
    model = TemporalIntentPolicyV32(strategy_count=2).eval()
    _force_hierarchical_market(
        model, stop_bias=-20.0, continue_bias=20.0, active_op="HIRE",
    )
    batch = collate_transitions([
        _row(money=0, land_exhausted=True),
    ])
    output = model.sample_step(
        batch, None, np.random.default_rng(3), deterministic=True,
        strategy_slots=torch.tensor([0]),
    )
    decision = output.rows[0].market[0]
    assert decision.legal_continue_mask.tolist() == [True, True]
    assert decision.legal_op_mask.tolist()[
        model.market_active_op_to_id["NOP_SLOT"]
    ] is True
    assert decision.chosen_action["kind"] == "NOP_SLOT"


def test_v32_teacher_nop_keeps_queue_open_and_exposes_hierarchy():
    model = TemporalIntentPolicyV32(strategy_count=2).eval()
    batch = collate_transitions([_row(market=[_nop(), _stop()])])
    output = model.teacher_step(
        batch, batch.canonical_actions, None,
        strategy_slots=torch.tensor([0]),
    )
    assert len(output.rows[0].market) == 2
    assert output.rows[0].market[0].chosen_action["kind"] == "NOP_SLOT"
    assert output.rows[0].market[1].chosen_action["kind"] == "STOP_QUEUE"
    assert output.rows[0].market[0].continue_logits is not None


def test_v32_strategy_slot_conditions_temporal_state_and_intent():
    torch.manual_seed(91)
    model = TemporalIntentPolicyV32(strategy_count=2).eval()
    with torch.no_grad():
        model.strategy_embedding.weight.zero_()
        model.strategy_embedding.weight[1, 0] = 1.0
    batch = collate_transitions([_row()])
    out0 = model.sample_step(
        batch, None, np.random.default_rng(4), deterministic=True,
        strategy_slots=torch.tensor([0]),
    )
    out1 = model.sample_step(
        batch, None, np.random.default_rng(4), deterministic=True,
        strategy_slots=torch.tensor([1]),
    )
    assert out0.temporal_state.memory.shape == out1.temporal_state.memory.shape
    assert torch.isfinite(out0.temporal_state.memory).all()
    assert torch.isfinite(out1.temporal_state.memory).all()
    assert not torch.allclose(
        out0.temporal_state.memory, out1.temporal_state.memory,
    )
    assert not torch.allclose(out0.temporal_state.h, out1.temporal_state.h)
    assert not torch.allclose(out0.intent, out1.intent)


def test_v32_trace_labels_continue_and_active_heads_separately():
    model = TemporalIntentPolicyV32(strategy_count=2).eval()
    _force_hierarchical_market(
        model, stop_bias=-20.0, continue_bias=20.0, active_op="HIRE",
    )
    batch = collate_transitions([_row()])
    traced = model.trace_sample_step(
        batch, None, np.random.default_rng(5), deterministic=True,
        strategy_slots=torch.tensor([0]),
    )
    decisions = traced["decisions"]
    continue_rows = [
        d for d in decisions
        if d["actor"] == "market:0:continue"
    ]
    active_rows = [
        d for d in decisions
        if d["actor"] == "market:0:active"
    ]
    assert len(continue_rows) == 1
    assert continue_rows[0]["ops"] == ["STOP", "CONTINUE"]
    assert continue_rows[0]["chosen_op"] == "CONTINUE"
    assert len(active_rows) == 1
    assert active_rows[0]["ops"] == list(ACTIVE_MARKET_OPS)
    assert active_rows[0]["chosen_op"] == "HIRE"


def test_v32_sell_quantity_never_exceeds_remaining_shed_inventory():
    model = TemporalIntentPolicyV32(strategy_count=2).eval()
    _force_hierarchical_market(
        model, stop_bias=-20.0, continue_bias=20.0, active_op="SELL",
    )
    with torch.no_grad():
        model.unit_op_head.weight.zero_()
        model.unit_op_head.bias.fill_(-20.0)
        model.unit_op_head.bias[model.unit_op_to_id["PASS"]] = 20.0
        model.item_head.weight.zero_()
        model.item_head.bias.fill_(-20.0)
        model.item_head.bias[ITEM_TO_ID["WHEAT"]] = 20.0
        model.quantity_decoder.output.weight.zero_()
        model.quantity_decoder.output.bias.fill_(-20.0)
        model.quantity_decoder.output.bias[DIGIT_OFFSET + 9] = 30.0
        model.quantity_decoder.output.bias[END_ID] = 10.0

    row = _row()
    row["state"]["private"]["shed"]["WHEAT"] = 10
    batch = collate_transitions([row])
    output = model.sample_step(
        batch, None, np.random.default_rng(123), deterministic=True,
        strategy_slots=torch.tensor([0]),
    )
    sells = [
        decision.chosen_action
        for decision in output.rows[0].market
        if decision.chosen_action.get("op") == "SELL"
    ]
    assert len(sells) >= 2
    assert sells[0]["item"] == "WHEAT"
    assert 1 <= int(sells[0]["quantity"]) <= 10
    assert 1 <= int(sells[1]["quantity"]) <= 10 - int(sells[0]["quantity"])
    assert sum(int(action["quantity"]) for action in sells) <= 10


def test_v32_continue_head_can_emit_nop_slot_without_fake_order():
    model = TemporalIntentPolicyV32(strategy_count=2).eval()
    _force_hierarchical_market(
        model, stop_bias=-20.0, continue_bias=20.0, active_op="NOP_SLOT",
    )
    batch = collate_transitions([_row()])
    output = model.sample_step(
        batch, None, np.random.default_rng(77), deterministic=True,
        strategy_slots=torch.tensor([0]),
    )
    first = output.rows[0].market[0]
    assert first.chosen_action == {
        "kind": "NOP_SLOT", "op": None,
        "item": None, "quantity": None, "raw": [],
    }
    assert first.op_logits.numel() == len(ACTIVE_MARKET_OPS)


def _force_pass_and_large_quantity(model, *, item):
    with torch.no_grad():
        model.unit_op_head.weight.zero_()
        model.unit_op_head.bias.fill_(-20.0)
        model.unit_op_head.bias[model.unit_op_to_id["PASS"]] = 20.0
        model.item_head.weight.zero_()
        model.item_head.bias.fill_(-20.0)
        model.item_head.bias[ITEM_TO_ID[item]] = 20.0
        model.quantity_decoder.output.weight.zero_()
        model.quantity_decoder.output.bias.fill_(-20.0)
        model.quantity_decoder.output.bias[DIGIT_OFFSET + 9] = 30.0
        model.quantity_decoder.output.bias[END_ID] = 10.0


def test_v32_buy_product_quantity_never_exceeds_remaining_shed_capacity():
    model = TemporalIntentPolicyV32(strategy_count=2).eval()
    _force_hierarchical_market(
        model, stop_bias=-20.0, continue_bias=20.0, active_op="BUY_PRODUCT",
    )
    _force_pass_and_large_quantity(model, item="WHEAT")
    row = _row(money=3000)
    row["state"]["private"]["shed"]["FERTILIZER"] = 98
    batch = collate_transitions([row])
    output = model.sample_step(
        batch, None, np.random.default_rng(321), deterministic=True,
        strategy_slots=torch.tensor([0]),
    )
    buys = [
        d.chosen_action for d in output.rows[0].market
        if d.chosen_action.get("op") == "BUY_PRODUCT"
    ]
    assert buys
    assert all(int(a["quantity"]) >= 1 for a in buys)
    assert sum(int(a["quantity"]) for a in buys) <= 2


def test_v32_buy_seed_quantity_respects_exact_cash_before_market_uncertainty():
    model = TemporalIntentPolicyV32(strategy_count=2).eval()
    _force_hierarchical_market(
        model, stop_bias=-20.0, continue_bias=20.0, active_op="BUY_SEED",
    )
    _force_pass_and_large_quantity(model, item="WHEAT")
    row = _row(money=25)
    batch = collate_transitions([row])
    output = model.sample_step(
        batch, None, np.random.default_rng(322), deterministic=True,
        strategy_slots=torch.tensor([0]),
    )
    buys = [
        d.chosen_action for d in output.rows[0].market
        if d.chosen_action.get("op") == "BUY_SEED"
        and d.chosen_action.get("item") == "WHEAT"
    ]
    assert buys
    assert sum(int(a["quantity"]) for a in buys) <= 2


def test_v32_teacher_buy_seed_can_exceed_safe_lower_bound_after_uncertain_sale():
    model = TemporalIntentPolicyV32(strategy_count=2).eval()
    market = [
        {
            "kind": "ORDER", "op": "SELL", "item": "WHEAT",
            "quantity": 10, "raw": ["SELL", "WHEAT", 10],
            "_executed": True,
        },
        {
            "kind": "ORDER", "op": "BUY_SEED", "item": "WHEAT",
            "quantity": 2, "raw": ["BUY_SEED", "WHEAT", 2],
            "_executed": True,
        },
        _stop(),
    ]
    row = _row(money=0, market=market)
    row["state"]["private"]["shed"]["WHEAT"] = 10
    batch = collate_transitions([row])
    output = model.teacher_step(
        batch, batch.canonical_actions, None,
        strategy_slots=torch.tensor([0]),
    )
    decisions = output.rows[0].market
    assert decisions[0].chosen_action["op"] == "SELL"
    assert decisions[1].chosen_action["op"] == "BUY_SEED"
    assert decisions[1].chosen_action["quantity"] == 2
    # Runtime generation remains conservative: only one WHEAT seed is
    # guaranteed affordable from floor-price sale proceeds.
    assert decisions[1].quantity_max_value == 1


def test_v32_opening_active_residual_directly_separates_strategy_slots():
    model = TemporalIntentPolicyV32(strategy_count=2).eval()
    hire_id = model.market_active_op_to_id["HIRE"]
    animal_id = model.market_active_op_to_id["BUY_ANIMAL"]
    with torch.no_grad():
        model.strategy_embedding.weight.zero_()
        model.strategy_embedding.weight[0, hire_id] = 20.0
        model.strategy_embedding.weight[1, animal_id] = 20.0
        model.market_continue_head.weight.zero_()
        model.market_continue_head.bias[:] = torch.tensor([-20.0, 20.0])
        model.market_active_op_head.weight.zero_()
        model.market_active_op_head.bias.zero_()

    batch = collate_transitions([_row()])
    out_hire = model.sample_step(
        batch, None, np.random.default_rng(401), deterministic=True,
        strategy_slots=torch.tensor([0]),
    )
    out_animal = model.sample_step(
        batch, None, np.random.default_rng(401), deterministic=True,
        strategy_slots=torch.tensor([1]),
    )
    first_hire = out_hire.rows[0].market[0]
    first_animal = out_animal.rows[0].market[0]
    assert first_hire.chosen_action["op"] == "HIRE"
    assert first_animal.chosen_action["op"] == "BUY_ANIMAL"
    assert first_hire.op_logits[hire_id] > first_hire.op_logits[animal_id]
    assert first_animal.op_logits[animal_id] > first_animal.op_logits[hire_id]


def test_v32_opening_active_residual_is_disabled_after_step_zero():
    model = TemporalIntentPolicyV32(strategy_count=2).eval()
    with torch.no_grad():
        model.strategy_embedding.weight.zero_()
        model.strategy_embedding.weight[0, model.market_active_op_to_id["HIRE"]] = 20.0
        model.market_continue_head.weight.zero_()
        model.market_continue_head.bias[:] = torch.tensor([-20.0, 20.0])
        model.market_active_op_head.weight.zero_()
        model.market_active_op_head.bias.zero_()

    row = _row()
    row["state"]["step"] = 1
    batch = collate_transitions([row])
    output = model.sample_step(
        batch, None, np.random.default_rng(402), deterministic=True,
        strategy_slots=torch.tensor([0]),
    )
    first = output.rows[0].market[0]
    assert torch.allclose(first.op_logits, torch.zeros_like(first.op_logits))


def test_v32_teacher_buy_animal_accepts_uncertain_sale_proceeds():
    model = TemporalIntentPolicyV32(strategy_count=2).eval()
    sell = {
        "kind": "ORDER", "op": "SELL", "item": "WOOL",
        "quantity": 12, "raw": ["SELL", "WOOL", 12], "_executed": True,
    }
    buy = {
        "kind": "ORDER", "op": "BUY_ANIMAL", "item": "COW",
        "quantity": 2, "raw": ["BUY_ANIMAL", "COW", 2], "_executed": True,
    }
    row = _row(money=743, market=[sell, buy, _stop()])
    row["state"]["private"]["shed"]["WOOL"] = 12
    batch = collate_transitions([row])
    output = model.teacher_step(
        batch, batch.canonical_actions, None,
        strategy_slots=torch.tensor([0]),
    )
    assert output.rows[0].market[1].chosen_action["quantity"] == 2
    assert output.rows[0].market[1].quantity_max_value == 1
