import torch

from kaggrl.v2_metrics import (
    economic_activity_summary,
    joint_step_exact,
    quantity_metrics,
    semantic_domain_metrics,
)
from kaggrl.v2_quantity import encode_quantity


def _unit(op="PASS", item=None, quantity=None):
    return {"op": op, "item": item, "quantity": quantity, "raw": [op]}


def _slot(op="STOP_QUEUE", item=None, quantity=None):
    if op in {"STOP_QUEUE", "NOP_SLOT"}:
        return {"kind": op, "op": None, "item": None, "quantity": None, "raw": []}
    return {"kind": "ORDER", "op": op, "item": item, "quantity": quantity,
            "raw": [op, item, quantity]}


def _joint(hands=20):
    return {
        "farmer": _unit("EAST"),
        "hands": [_unit("PASS") for _ in range(hands)],
        "market": [_slot("HIRE"), _slot("SELL", "WHEAT", 3), _slot()],
    }


def test_one_wrong_hand_only_modestly_changes_hand_mean_but_breaks_joint_exact():
    target = _joint(20)
    pred = _joint(20)
    pred["hands"][7] = _unit("NORTH")
    metrics = semantic_domain_metrics([pred], [target], masks=None)
    assert metrics["farmer_semantic_exact"] == 1.0
    assert metrics["mean_hand_semantic_exact"] == 19 / 20
    assert metrics["all_hands_exact"] == 0.0
    assert metrics["market_sequence_exact"] == 1.0
    assert metrics["full_joint_step_exact"] == 0.0
    exact = joint_step_exact([pred], [target], masks=None)
    assert exact.dtype == torch.bool and exact.tolist() == [False]


def test_padded_hands_and_market_slots_do_not_change_metrics():
    target = _joint(2); pred = _joint(2)
    baseline = semantic_domain_metrics([pred], [target], masks=None)
    target["hands"] += [_unit("WEST") for _ in range(5)]
    pred["hands"] += [_unit("EAST") for _ in range(5)]
    target["market"] += [_slot("NOP_SLOT") for _ in range(4)]
    pred["market"] += [_slot("BUY_LAND") for _ in range(4)]
    masks = {
        "hands": torch.tensor([[True, True, False, False, False, False, False]]),
        "market": torch.tensor([[True, True, True, False, False, False, False]]),
    }
    padded = semantic_domain_metrics([pred], [target], masks=masks)
    for key in baseline:
        assert padded[key] == baseline[key], key


def test_quantity_metrics_compare_decoded_integer_and_digit_stream():
    values = [0, 1, 101, 1000, 123456]
    targets = [encode_quantity(value) for value in values]
    perfect = quantity_metrics(targets, targets, active=[True] * len(values))
    assert perfect["quantity_integer_exact"] == 1.0
    assert perfect["digit_token_accuracy"] == 1.0
    assert perfect["mean_digit_edit_distance"] == 0.0

    pred = list(targets)
    pred[-1] = encode_quantity(123455)
    changed = quantity_metrics(pred, targets, active=[True] * len(values))
    assert changed["quantity_integer_exact"] == 4 / 5
    assert changed["digit_token_accuracy"] < 1.0
    assert changed["mean_digit_edit_distance"] > 0.0


def test_stop_queue_and_nop_slot_are_not_semantically_equal():
    target = _joint(0); pred = _joint(0)
    pred["market"][-1] = _slot("NOP_SLOT")
    metrics = semantic_domain_metrics([pred], [target], masks=None)
    assert metrics["market_sequence_exact"] == 0.0
    assert metrics["stop_nop_confusions"] == 1


def test_economic_activity_summary_counts_observed_confirmed_effects():
    transitions = [{
        "effects": {
            "money_delta": 30, "shed_delta": {"WHEAT": -3},
            "hand_count_delta": 1,
            "action_evidence": [
                {"actor": "market:0", "op": "HIRE", "status": "confirmed", "observed": {}},
                {"actor": "farmer", "op": "PLANT", "status": "confirmed", "observed": {}},
                {"actor": "hand:0", "op": "HARVEST", "status": "confirmed",
                 "observed": {"inventory_delta": {"WHEAT": 4}}},
                {"actor": "market:1", "op": "SELL", "status": "confirmed",
                 "observed": {"item": "WHEAT"}},
            ],
        }
    }]
    summary = economic_activity_summary(transitions)
    assert summary["confirmed_hires"] == 1
    assert summary["confirmed_plants"] == 1
    assert summary["confirmed_harvests"] == 1
    assert summary["harvest_units_gained"] == 4
    assert summary["confirmed_sales"] == 1


def test_active_market_metrics_expose_zero_buy_and_sell_recall():
    target = {
        "farmer": _unit("PASS"),
        "hands": [],
        "market": [
            _slot("BUY_SEED", "WHEAT", 1),
            _slot("SELL", "CARROT", 3),
            _slot("HIRE"),
            _slot("STOP_QUEUE"),
        ],
    }
    pred = {
        "farmer": _unit("PASS"),
        "hands": [],
        "market": [
            _slot("HIRE"),
            _slot("HIRE"),
            _slot("HIRE"),
            _slot("STOP_QUEUE"),
        ],
    }
    metrics = semantic_domain_metrics([pred], [target])
    assert metrics["market_continue_accuracy"] == 1.0
    assert metrics["market_active_target_count"] == 3
    assert metrics["market_buy_target_count"] == 1
    assert metrics["market_sell_target_count"] == 1
    assert metrics["market_active_op_accuracy"] == 1 / 3
    assert metrics["market_active_semantic_exact"] == 1 / 3
    assert metrics["market_buy_op_recall"] == 0.0
    assert metrics["market_sell_op_recall"] == 0.0
    assert metrics["market_hire_op_recall"] == 1.0


def test_active_market_metrics_ignore_nop_for_active_op_scoring():
    target = {
        "farmer": _unit("PASS"),
        "hands": [],
        "market": [_slot("NOP_SLOT"), _slot("BUY_PRODUCT", "WHEAT", 2), _slot()],
    }
    pred = {
        "farmer": _unit("PASS"),
        "hands": [],
        "market": [_slot("HIRE"), _slot("BUY_PRODUCT", "WHEAT", 2), _slot()],
    }
    metrics = semantic_domain_metrics([pred], [target])
    assert metrics["market_active_target_count"] == 1
    assert metrics["market_active_op_accuracy"] == 1.0
    assert metrics["market_active_semantic_exact"] == 1.0
    assert metrics["market_continue_accuracy"] == 1.0
