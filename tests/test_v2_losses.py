from types import SimpleNamespace

import pytest
import torch

from kaggrl.constants import ITEM_TO_ID, UNIT_OPS
from kaggrl.v2_ledger import MARKET_OPS
from kaggrl.v2_losses import (
    DEFAULT_LOSS_WEIGHTS, _market_phase_active_weight, _market_phase_continue_weight,
    action_loss, total_pretrain_loss,
)
from kaggrl.v2_model import DecisionOutput, RowPolicyOutput
from kaggrl.v2_quantity import VOCAB_SIZE, encode_quantity


def _decision(op_count, action, *, item_logits=None, quantity_logits=None):
    return DecisionOutput(
        op_logits=torch.zeros(op_count, requires_grad=True),
        item_logits=(torch.zeros(max(ITEM_TO_ID.values()) + 1, requires_grad=True)
                     if item_logits is None else item_logits),
        quantity_logits=quantity_logits,
        quantity_tokens=None,
        chosen_action=action,
        legal_op_mask=torch.ones(op_count, dtype=torch.bool),
        legal_item_mask=None,
    )


def _unit(op="PASS", item=None, quantity=None):
    return {"op": op, "item": item, "quantity": quantity, "raw": [op]}


def _market(op="STOP_QUEUE", item=None, quantity=None):
    if op in {"STOP_QUEUE", "NOP_SLOT"}:
        return {"kind": op, "op": None, "item": None, "quantity": None, "raw": []}
    return {"kind": "ORDER", "op": op, "item": item, "quantity": quantity, "raw": [op]}


def _row(hand_count):
    farmer = _decision(len(UNIT_OPS), _unit())
    hands = tuple(_decision(len(UNIT_OPS), _unit()) for _ in range(hand_count))
    market = (_decision(len(MARKET_OPS), _market()),)
    return RowPolicyOutput(farmer=farmer, hands=hands, market=market, trace=())


def _target(hand_count):
    return {
        "farmer": _unit(),
        "hands": [_unit() for _ in range(hand_count)],
        "market": [_market()],
    }


def test_hands_domain_is_mean_not_scaled_by_real_hand_count():
    one = action_loss(SimpleNamespace(rows=(_row(1),)), (_target(1),), None, torch.ones(1))
    many = action_loss(SimpleNamespace(rows=(_row(20),)), (_target(20),), None, torch.ones(1))
    assert torch.allclose(one.hands, many.hands)
    assert torch.allclose(one.total, many.total)


def test_market_domain_weights_each_decision_equally_across_rows():
    stop = _decision(len(MARKET_OPS), _market())
    stop_logits = torch.full((len(MARKET_OPS),), -4.0)
    stop_logits[MARKET_OPS.index("HIRE")] = 4.0
    stop.op_logits = stop_logits.requires_grad_(True)

    def hire_decision():
        decision = _decision(len(MARKET_OPS), _market("HIRE"))
        logits = torch.full((len(MARKET_OPS),), -4.0)
        logits[MARKET_OPS.index("HIRE")] = 4.0
        decision.op_logits = logits.requires_grad_(True)
        return decision

    stop_row = RowPolicyOutput(
        farmer=_decision(len(UNIT_OPS), _unit()),
        hands=(), market=(stop,), trace=(),
    )
    hire_row = RowPolicyOutput(
        farmer=_decision(len(UNIT_OPS), _unit()),
        hands=(), market=tuple(hire_decision() for _ in range(3)), trace=(),
    )
    stop_target = {"farmer": _unit(), "hands": [], "market": [_market()]}
    hire_target = {
        "farmer": _unit(), "hands": [],
        "market": [_market("HIRE") for _ in range(3)],
    }

    stop_loss = action_loss(SimpleNamespace(rows=(stop_row,)), (stop_target,)).market
    hire_loss = action_loss(SimpleNamespace(rows=(hire_row,)), (hire_target,)).market
    mixed = action_loss(
        SimpleNamespace(rows=(stop_row, hire_row)),
        (stop_target, hire_target),
    ).market
    expected = (stop_loss + 3.0 * hire_loss) / 4.0
    assert torch.allclose(mixed, expected)
    assert not torch.allclose(mixed, (stop_loss + hire_loss) / 2.0)


def test_pass_ignores_item_and_quantity_logits():
    target = ({"farmer": _unit("PASS"), "hands": [], "market": [_market()]},)
    a = _row(0)
    b = _row(0)
    a.farmer.item_logits = torch.randn_like(a.farmer.item_logits) * 100
    b.farmer.item_logits = torch.randn_like(b.farmer.item_logits) * -100
    a.farmer.quantity_logits = torch.randn(7, VOCAB_SIZE, requires_grad=True)
    b.farmer.quantity_logits = torch.randn(2, VOCAB_SIZE, requires_grad=True)
    la = action_loss(SimpleNamespace(rows=(a,)), target, None, torch.ones(1))
    lb = action_loss(SimpleNamespace(rows=(b,)), target, None, torch.ones(1))
    assert torch.allclose(la.farmer, lb.farmer)


def test_plant_trains_item_but_not_quantity():
    item_logits = torch.full((max(ITEM_TO_ID.values()) + 1,), -5.0, requires_grad=True)
    item_logits = item_logits.clone(); item_logits[ITEM_TO_ID["WHEAT"]] = 5.0
    farmer = _decision(len(UNIT_OPS), _unit("PLANT", "WHEAT"), item_logits=item_logits,
                       quantity_logits=torch.randn(4, VOCAB_SIZE, requires_grad=True))
    row = RowPolicyOutput(farmer=farmer, hands=(), market=(_decision(len(MARKET_OPS), _market()),), trace=())
    target = ({"farmer": _unit("PLANT", "WHEAT"), "hands": [], "market": [_market()]},)
    good = action_loss(SimpleNamespace(rows=(row,)), target, None, torch.ones(1)).farmer
    bad_logits = item_logits.detach().clone().requires_grad_(True)
    bad_logits = bad_logits.clone(); bad_logits[ITEM_TO_ID["WHEAT"]] = -10.0
    row.farmer.item_logits = bad_logits
    bad = action_loss(SimpleNamespace(rows=(row,)), target, None, torch.ones(1)).farmer
    assert good < bad


def test_pickup_quantity_loss_stops_at_end_and_ignores_padded_digits():
    tokens = encode_quantity(1000)
    logits = torch.zeros(len(tokens) + 5, VOCAB_SIZE, requires_grad=True)
    farmer = _decision(len(UNIT_OPS), _unit("PICKUP", "WHEAT", 1000), quantity_logits=logits)
    row = RowPolicyOutput(farmer=farmer, hands=(), market=(_decision(len(MARKET_OPS), _market()),), trace=())
    target = ({"farmer": _unit("PICKUP", "WHEAT", 1000), "hands": [], "market": [_market()]},)
    loss = action_loss(SimpleNamespace(rows=(row,)), target, None, torch.ones(1)).farmer
    loss.backward()
    assert logits.grad is not None
    assert logits.grad[len(tokens):].abs().sum().item() == 0.0


def test_hire_market_loss_ignores_item_and_quantity_arguments():
    hire = _market("HIRE")
    decision = _decision(len(MARKET_OPS), hire, quantity_logits=torch.randn(6, VOCAB_SIZE, requires_grad=True))
    row = RowPolicyOutput(farmer=_decision(len(UNIT_OPS), _unit()), hands=(), market=(decision,), trace=())
    target = ({"farmer": _unit(), "hands": [], "market": [hire]},)
    first = action_loss(SimpleNamespace(rows=(row,)), target, None, torch.ones(1)).market
    row.market[0].item_logits = torch.randn_like(decision.item_logits) * 999
    row.market[0].quantity_logits = torch.randn(2, VOCAB_SIZE, requires_grad=True)
    second = action_loss(SimpleNamespace(rows=(row,)), target, None, torch.ones(1)).market
    assert torch.allclose(first, second)


def test_total_pretrain_loss_uses_frozen_initial_coefficients():
    row = _row(1)
    outputs = SimpleNamespace(
        rows=(row,),
        aux=SimpleNamespace(
            effect=torch.ones(1, 3), future_resource=torch.ones(1, 4),
            unit_task=torch.ones(1, 2, 5), opponent_effect=torch.ones(1, 6),
            terminal_money=torch.ones(1), terminal_margin=torch.ones(1),
        ),
    )
    batch = SimpleNamespace(
        canonical_actions=(_target(1),),
        own_unit_mask=torch.tensor([[True, True]]),
        auxiliary_targets={
            "effect": torch.zeros(1, 3), "future_resource": torch.zeros(1, 4),
            "unit_task": torch.zeros(1, 2, 5), "opponent_effect": torch.zeros(1, 6),
            "terminal_money": torch.zeros(1), "terminal_margin": torch.zeros(1),
        },
    )
    result = total_pretrain_loss(outputs, batch)
    assert DEFAULT_LOSS_WEIGHTS == {
        "action": 1.0, "effect": 0.25, "future_resource": 0.15,
        "unit_task": 0.15, "opponent_effect": 0.10, "value": 0.05,
    }
    expected = sum(DEFAULT_LOSS_WEIGHTS[name] * result[name] for name in DEFAULT_LOSS_WEIGHTS)
    assert torch.allclose(result["total"], expected)


def test_zero_hand_rows_do_not_dilute_hands_domain_mean():
    single = action_loss(
        SimpleNamespace(rows=(_row(1),)),
        (_target(1),), None, torch.ones(1),
    )
    mixed = action_loss(
        SimpleNamespace(rows=(_row(0), _row(1))),
        (_target(0), _target(1)), None, torch.ones(2),
    )
    assert torch.allclose(single.hands, mixed.hands)


def test_v3_behavior_family_mapping_covers_unit_and_market_operations():
    from kaggrl.v3_behavior import behavior_family

    assert behavior_family(_unit("PASS"), "unit") == "WAIT"
    assert behavior_family(_unit("NORTH"), "unit") == "MOVEMENT"
    assert behavior_family(_unit("PICKUP"), "unit") == "ACQUISITION"
    assert behavior_family(_unit("PLANT"), "unit") == "PRODUCTION"
    assert behavior_family(_unit("WATER"), "unit") == "MAINTENANCE"
    assert behavior_family(_unit("HARVEST"), "unit") == "HARVEST"
    assert behavior_family(_unit("DROP"), "unit") == "DEPOSIT"
    assert behavior_family(_market("BUY_SEED"), "market") == "ACQUISITION"
    assert behavior_family(_market("SELL"), "market") == "SALE"
    assert behavior_family(_market("HIRE"), "market") == "HIRE"
    assert behavior_family(_market("BUY_LAND"), "market") == "EXPANSION"
    assert behavior_family(_market("STOP_QUEUE"), "market") == "WAIT"


def test_v3_family_weights_upweight_rare_families_without_exceeding_cap():
    from kaggrl.v3_behavior import compute_family_weights

    weights = compute_family_weights(
        {"WAIT": 1000, "ACQUISITION": 100, "SALE": 10}, cap=5.0,
    )
    assert weights["WAIT"] == 1.0
    assert weights["ACQUISITION"] > weights["WAIT"]
    assert weights["SALE"] == 5.0
    assert max(weights.values()) <= 5.0


def test_family_weight_scales_unit_operation_loss_only():
    row = _row(0)
    target = ({
        "farmer": _unit("NORTH"), "hands": [], "market": [_market()],
    },)
    base = action_loss(
        SimpleNamespace(rows=(row,)), target, None, torch.ones(1),
    ).farmer
    weighted = action_loss(
        SimpleNamespace(rows=(row,)), target, None, torch.ones(1),
        family_weights={"MOVEMENT": 3.0, "WAIT": 1.0},
    ).farmer
    assert torch.allclose(weighted, base * 3.0)


def test_family_weight_does_not_scale_market_item_or_quantity_terms():
    quantity_logits = torch.zeros(
        len(encode_quantity(1)), VOCAB_SIZE, requires_grad=True,
    )
    buy = _market("BUY_SEED", item="WHEAT", quantity=1)
    decision = _decision(
        len(MARKET_OPS), buy, quantity_logits=quantity_logits,
    )
    row = RowPolicyOutput(
        farmer=_decision(len(UNIT_OPS), _unit()),
        hands=(), market=(decision,), trace=(),
    )
    target = ({"farmer": _unit(), "hands": [], "market": [buy]},)
    base = action_loss(SimpleNamespace(rows=(row,)), target).market
    weighted = action_loss(
        SimpleNamespace(rows=(row,)), target,
        family_weights={"ACQUISITION": 3.0, "WAIT": 1.0},
    ).market
    op_ce = torch.log(torch.tensor(float(len(MARKET_OPS))))
    assert torch.allclose(weighted - base, 2.0 * op_ce)


def test_domain_family_weights_keep_market_distribution_separate_from_units():
    from kaggrl.v3_behavior import compute_domain_family_weights

    weights = compute_domain_family_weights({
        "unit": {"WAIT": 10, "ACQUISITION": 1000, "MOVEMENT": 2000},
        "market": {"WAIT": 1000, "HIRE": 400, "ACQUISITION": 200},
    }, cap=3.0)
    assert weights["market"]["WAIT"] == 1.0
    assert weights["market"]["HIRE"] == 2.5
    assert weights["market"]["ACQUISITION"] == 3.0
    assert weights["unit"]["ACQUISITION"] == 2.0
    assert weights["market"]["ACQUISITION"] > weights["market"]["HIRE"]


def test_nested_family_weights_apply_by_decision_domain():
    row = _row(0)
    target = ({
        "farmer": _unit("NORTH"),
        "hands": [],
        "market": [_market("HIRE")],
    },)
    weights = {
        "unit": {"MOVEMENT": 2.0},
        "market": {"HIRE": 3.0},
    }
    loss = action_loss(
        SimpleNamespace(rows=(row,)), target,
        family_weights=weights,
    )
    unit_ce = torch.log(torch.tensor(float(len(UNIT_OPS))))
    market_ce = torch.log(torch.tensor(float(len(MARKET_OPS))))
    assert torch.allclose(loss.farmer, 2.0 * unit_ce)
    assert torch.allclose(loss.market, 3.0 * market_ce)


def test_v3_family_weights_never_upweight_wait_when_other_family_is_more_common():
    from kaggrl.v3_behavior import compute_family_weights

    weights = compute_family_weights(
        {"MOVEMENT": 1000, "WAIT": 100, "ACQUISITION": 50}, cap=3.0,
    )
    assert weights["WAIT"] == 1.0
    assert weights["MOVEMENT"] == 1.0
    assert weights["ACQUISITION"] == 3.0


def _target_op_grad_for_farmer(target_action):
    row = _row(0)
    row.farmer.op_logits = torch.zeros(len(UNIT_OPS), requires_grad=True)
    row.farmer.item_logits = torch.zeros(max(ITEM_TO_ID.values()) + 1, requires_grad=True)
    if target_action["op"] in {"PICKUP", "PLACE"}:
        row.farmer.quantity_logits = torch.zeros(
            len(encode_quantity(target_action["quantity"])), VOCAB_SIZE,
            requires_grad=True,
        )
    target = ({
        "farmer": target_action,
        "hands": [],
        "market": [_market()],
    },)
    loss = action_loss(SimpleNamespace(rows=(row,)), target).farmer
    loss.backward()
    target_id = UNIT_OPS.index(target_action["op"])
    return float(abs(row.farmer.op_logits.grad[target_id].item()))


def _target_op_grad_for_market(target_action):
    row = _row(0)
    decision = row.market[0]
    decision.op_logits = torch.zeros(len(MARKET_OPS), requires_grad=True)
    decision.item_logits = torch.zeros(max(ITEM_TO_ID.values()) + 1, requires_grad=True)
    if target_action.get("quantity") is not None:
        decision.quantity_logits = torch.zeros(
            len(encode_quantity(target_action["quantity"])), VOCAB_SIZE,
            requires_grad=True,
        )
    target = ({
        "farmer": _unit(),
        "hands": [],
        "market": [target_action],
    },)
    loss = action_loss(SimpleNamespace(rows=(row,)), target).market
    loss.backward()
    op = target_action["kind"] if target_action.get("kind") in {"STOP_QUEUE", "NOP_SLOT"} else target_action["op"]
    target_id = MARKET_OPS.index(op)
    return float(abs(decision.op_logits.grad[target_id].item()))


def test_unit_op_gradient_is_not_divided_by_item_or_quantity_terms():
    pass_grad = _target_op_grad_for_farmer(_unit("PASS"))
    plant_grad = _target_op_grad_for_farmer(_unit("PLANT", "WHEAT"))
    pickup_grad = _target_op_grad_for_farmer(_unit("PICKUP", "WHEAT", 3))
    assert plant_grad == pytest.approx(pass_grad, rel=1e-6, abs=1e-6)
    assert pickup_grad == pytest.approx(pass_grad, rel=1e-6, abs=1e-6)


def test_market_buy_op_gradient_is_not_divided_by_item_and_quantity_terms():
    hire_grad = _target_op_grad_for_market(_market("HIRE"))
    buy_grad = _target_op_grad_for_market(_market("BUY_SEED", "WHEAT", 1))
    assert buy_grad == pytest.approx(hire_grad, rel=1e-6, abs=1e-6)


def test_hierarchical_market_stop_trains_only_termination():
    row = _row(0)
    decision = row.market[0]
    decision.continue_logits = torch.zeros(2, requires_grad=True)
    decision.op_logits = (torch.randn(len(MARKET_OPS)) * 100.0).requires_grad_()
    target = ({
        "farmer": _unit(),
        "hands": [],
        "market": [_market("STOP_QUEUE")],
    },)
    loss = action_loss(SimpleNamespace(rows=(row,)), target).market
    assert torch.allclose(loss, torch.log(torch.tensor(2.0)))
    loss.backward()
    assert decision.continue_logits.grad is not None
    assert decision.op_logits.grad is None


def test_hierarchical_market_nop_trains_continue_and_nop_head():
    row = _row(0)
    decision = row.market[0]
    decision.continue_logits = torch.zeros(2, requires_grad=True)
    decision.op_logits = torch.zeros(len(MARKET_OPS), requires_grad=True)
    target = ({
        "farmer": _unit(),
        "hands": [],
        "market": [_market("NOP_SLOT")],
    },)
    loss = action_loss(SimpleNamespace(rows=(row,)), target).market
    expected = torch.log(torch.tensor(2.0)) + torch.log(
        torch.tensor(float(len(MARKET_OPS)))
    )
    assert torch.allclose(loss, expected)
    loss.backward()
    assert decision.continue_logits.grad is not None
    assert decision.op_logits.grad is not None
    assert decision.op_logits.grad[MARKET_OPS.index("NOP_SLOT")].abs().item() > 0


def test_v32_queue_survival_weights_early_continue_more_than_late_continue():
    def v32_market(op):
        decision = _decision(len(MARKET_OPS), _market(op))
        decision.continue_logits = torch.tensor(
            [4.0, -4.0], requires_grad=True,
        )
        if op == "HIRE":
            logits = torch.full((len(MARKET_OPS),), -4.0)
            logits[MARKET_OPS.index("HIRE")] = 4.0
            decision.op_logits = logits.requires_grad_(True)
        return decision

    early = v32_market("HIRE")
    late = v32_market("HIRE")
    stop = v32_market("STOP_QUEUE")
    row = RowPolicyOutput(
        farmer=_decision(len(UNIT_OPS), _unit()),
        hands=(),
        market=(early, late, stop),
        trace=(),
    )
    target = ({
        "farmer": _unit(),
        "hands": [],
        "market": [_market("HIRE"), _market("HIRE"), _market("STOP_QUEUE")],
    },)
    loss = action_loss(SimpleNamespace(rows=(row,)), target).market
    loss.backward()

    early_grad = abs(float(early.continue_logits.grad[1]))
    late_grad = abs(float(late.continue_logits.grad[1]))
    assert early_grad > late_grad
    assert early_grad / late_grad == pytest.approx(1.35, rel=1e-4)


def test_early_game_continue_weight_decays_after_step_three():
    assert _market_phase_continue_weight(0) == 4.0
    assert _market_phase_continue_weight(1) == 2.0
    assert _market_phase_continue_weight(2) == 1.0
    assert _market_phase_continue_weight(3) == 1.0
    assert _market_phase_continue_weight(4) == 1.0
    assert _market_phase_continue_weight(400) == 1.0


def test_early_game_row_weight_scales_only_continue_gradient():
    def make_row():
        decision = _decision(len(MARKET_OPS), _market("HIRE"))
        decision.continue_logits = torch.tensor([4.0, -4.0], requires_grad=True)
        op_logits = torch.full((len(MARKET_OPS),), -4.0)
        op_logits[MARKET_OPS.index("HIRE")] = 4.0
        decision.op_logits = op_logits.requires_grad_(True)
        row = RowPolicyOutput(
            farmer=_decision(len(UNIT_OPS), _unit()),
            hands=(), market=(decision,), trace=(),
        )
        target = ({
            "farmer": _unit(), "hands": [], "market": [_market("HIRE")],
        },)
        return row, decision, target

    normal_row, normal_decision, target = make_row()
    normal = action_loss(
        SimpleNamespace(rows=(normal_row,)), target,
        market_continue_row_weight=[1.0],
    ).market
    normal.backward()
    normal_continue = abs(float(normal_decision.continue_logits.grad[1]))
    normal_active = abs(float(
        normal_decision.op_logits.grad[MARKET_OPS.index("HIRE")]
    ))

    early_row, early_decision, target = make_row()
    early = action_loss(
        SimpleNamespace(rows=(early_row,)), target,
        market_continue_row_weight=[4.0],
    ).market
    early.backward()
    early_continue = abs(float(early_decision.continue_logits.grad[1]))
    early_active = abs(float(
        early_decision.op_logits.grad[MARKET_OPS.index("HIRE")]
    ))

    assert early_continue / normal_continue == pytest.approx(4.0, rel=1e-5)
    assert early_active == pytest.approx(normal_active, rel=1e-6, abs=1e-6)


def test_early_game_row_weight_does_not_scale_later_market_slots():
    def v32_market(op):
        decision = _decision(len(MARKET_OPS), _market(op))
        decision.continue_logits = torch.tensor([4.0, -4.0], requires_grad=True)
        logits = torch.full((len(MARKET_OPS),), -4.0)
        if op != "STOP_QUEUE":
            logits[MARKET_OPS.index(op)] = 4.0
        decision.op_logits = logits.requires_grad_(True)
        return decision

    first = v32_market("HIRE")
    second = v32_market("HIRE")
    stop = v32_market("STOP_QUEUE")
    row = RowPolicyOutput(
        farmer=_decision(len(UNIT_OPS), _unit()),
        hands=(), market=(first, second, stop), trace=(),
    )
    target = ({
        "farmer": _unit(), "hands": [],
        "market": [_market("HIRE"), _market("HIRE"), _market("STOP_QUEUE")],
    },)
    loss = action_loss(
        SimpleNamespace(rows=(row,)), target,
        market_continue_row_weight=[4.0],
    ).market
    loss.backward()

    first_grad = abs(float(first.continue_logits.grad[1]))
    second_grad = abs(float(second.continue_logits.grad[1]))
    # Queue-survival is 1.35x on slot 0 vs slot 1, then the phase multiplier
    # applies only to slot 0.
    assert first_grad / second_grad == pytest.approx(4.0 * 1.35, rel=1e-4)


def test_market_active_op_weight_overrides_family_weight_without_scaling_continue():
    def make():
        decision = _decision(len(MARKET_OPS), _market("HIRE"))
        decision.continue_logits = torch.zeros(2, requires_grad=True)
        decision.op_logits = torch.zeros(len(MARKET_OPS), requires_grad=True)
        row = RowPolicyOutput(
            farmer=_decision(len(UNIT_OPS), _unit()),
            hands=(), market=(decision,), trace=(),
        )
        target = ({
            "farmer": _unit(), "hands": [], "market": [_market("HIRE")],
        },)
        return row, decision, target

    base_row, base_decision, target = make()
    base = action_loss(
        SimpleNamespace(rows=(base_row,)), target,
        family_weights={"market": {"HIRE": 7.0}},
        market_active_op_weights={"HIRE": 1.0},
    ).market
    base.backward()
    base_op = abs(float(base_decision.op_logits.grad[MARKET_OPS.index("HIRE")]))
    base_continue = abs(float(base_decision.continue_logits.grad[1]))

    weighted_row, weighted_decision, target = make()
    weighted = action_loss(
        SimpleNamespace(rows=(weighted_row,)), target,
        family_weights={"market": {"HIRE": 7.0}},
        market_active_op_weights={"HIRE": 4.0},
    ).market
    weighted.backward()
    weighted_op = abs(float(weighted_decision.op_logits.grad[MARKET_OPS.index("HIRE")]))
    weighted_continue = abs(float(weighted_decision.continue_logits.grad[1]))

    assert weighted_op / base_op == pytest.approx(4.0, rel=1e-6)
    assert weighted_continue == pytest.approx(base_continue, rel=1e-6, abs=1e-6)


def test_opening_active_weight_is_step0_only():
    assert _market_phase_active_weight(0) == 32.0
    assert _market_phase_active_weight(1) == 1.0
    assert _market_phase_active_weight(100) == 1.0


def test_opening_active_row_weight_scales_active_op_but_not_continue():
    row = _row(0)
    decision = row.market[0]
    decision.continue_logits = torch.zeros(2, requires_grad=True)
    decision.op_logits = torch.zeros(len(MARKET_OPS), requires_grad=True)
    target = ({
        "farmer": _unit(), "hands": [],
        "market": [_market("HIRE")],
    },)
    base = action_loss(
        SimpleNamespace(rows=(row,)), target,
        market_active_row_weight=[1.0],
    ).market
    base.backward()
    base_active = abs(float(decision.op_logits.grad[MARKET_OPS.index("HIRE")]))
    base_continue = abs(float(decision.continue_logits.grad[1]))

    row2 = _row(0)
    d2 = row2.market[0]
    d2.continue_logits = torch.zeros(2, requires_grad=True)
    d2.op_logits = torch.zeros(len(MARKET_OPS), requires_grad=True)
    weighted = action_loss(
        SimpleNamespace(rows=(row2,)), target,
        market_active_row_weight=[32.0],
    ).market
    weighted.backward()
    weighted_active = abs(float(d2.op_logits.grad[MARKET_OPS.index("HIRE")]))
    weighted_continue = abs(float(d2.continue_logits.grad[1]))

    assert weighted_active / base_active == pytest.approx(32.0, rel=1e-5)
    assert weighted_continue == pytest.approx(base_continue, rel=1e-6, abs=1e-6)
