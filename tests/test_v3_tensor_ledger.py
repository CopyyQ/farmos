import torch
from dataclasses import fields

from kaggrl.constants import ITEM_TO_ID, PRODUCTS, UNIT_OPS
from kaggrl.v2_tensorize import collate_transitions
from kaggrl.v2_ledger import MARKET_OPS, ShadowLedger
from kaggrl.v3_tensor_ledger import (
    ITEM_CLASSES,
    MARKET_OP_TO_ID,
    UNIT_OP_TO_ID,
    TensorLedger,
)


def _tiles():
    return [[None if x < 5 and y < 5 else "LOCKED" for x in range(10)] for y in range(10)]


def _state(*, money=3000, hires=0, shed=None, inventory=None, seeds=None, hands=0):
    grid = _tiles()
    inv = dict(inventory or {})
    hand_positions = [[3 - i, 4] for i in range(hands)]
    own = {
        "money": money,
        "farmer": [4, 4],
        "hands": hand_positions,
        "hires_today": hires,
        "unlocked_quadrants": ["NW"],
        "tiles": grid,
    }
    prices = {name: 25 for name in PRODUCTS}
    prices.update({"WHEAT": 10, "FERTILIZER": 20})
    return {
        "player": 0,
        "day": 4,
        "hour": 4,
        "own": own,
        "rival": {
            "money": 3000,
            "farmer": [8, 8],
            "hands": [],
            "hires_today": 0,
            "unlocked_quadrants": ["NW"],
            "tiles": _tiles(),
        },
        "private": {
            "shed": dict(shed or {}),
            "seeds": dict(seeds or {}),
            "inventories": [inv] + [{} for _ in range(hands)],
        },
        "own_grid": grid,
        "rival_grid": _tiles(),
        "own_units": tuple(
            [{"kind": "farmer", "index": 0, "position": [4, 4], "inventory": inv}]
            + [
                {"kind": "hand", "index": i, "position": pos, "inventory": {}}
                for i, pos in enumerate(hand_positions)
            ]
        ),
        "rival_units": ({"kind": "farmer", "index": 0, "position": [8, 8]},),
        "market": {
            "inventory": {item: 10000 for item in PRODUCTS},
            "prices": prices,
        },
        "town": {"unlocked_shops": []},
        "town_shops": (),
    }


def _tensor_market_action(op, item=None, quantity=None, *, known=False):
    op_id = MARKET_OP_TO_ID[op]
    item_id = ITEM_TO_ID.get(item, 0) if item is not None else 0
    qty = -1 if quantity is None else int(quantity)
    return (
        torch.tensor([op_id]),
        torch.tensor([item_id]),
        torch.tensor([qty]),
        torch.tensor([known]),
    )


def _tensor_unit_action(op, item=None, quantity=None):
    op_id = UNIT_OP_TO_ID[op]
    item_id = ITEM_TO_ID.get(item, 0) if item is not None else 0
    qty = -1 if quantity is None else int(quantity)
    return torch.tensor([op_id]), torch.tensor([item_id]), torch.tensor([qty])


def _shadow_market_op(mask, op):
    return bool(mask.ops.get(op, False))


def _assert_market_legal_parity(shadow, tensor, slot):
    expected = shadow.legal_market_mask(slot, {})
    got = tensor.legal_market(slot)
    for op in MARKET_OPS:
        assert bool(got.op_mask[0, MARKET_OP_TO_ID[op]].item()) == _shadow_market_op(expected, op), op
    for op in ("BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"):
        op_id = MARKET_OP_TO_ID[op]
        choices = expected.items.get(op, {})
        qmax = (expected.metadata.get("market_quantity_max_by_op_item") or {}).get(op, {})
        for name, item_id in ITEM_TO_ID.items():
            assert bool(got.item_mask[0, op_id, item_id].item()) == bool(choices.get(name, False)), (op, name)
            expected_max = int(qmax[name]) if name in qmax else -1
            assert int(got.quantity_max[0, op_id, item_id].item()) == expected_max, (op, name)


def _assert_core_state_parity(shadow, tensor):
    assert int(tensor.cash[0].item()) == shadow.cash_lower_bound
    assert bool(tensor.cash_uncertain[0].item()) == bool(shadow.cash_uncertain)
    assert int(tensor.hires_today[0].item()) == shadow.hires_today
    assert bool(tensor.hire_count_uncertain[0].item()) == bool(shadow.hire_count_uncertain)
    assert int(tensor.land_count[0].item()) == len(shadow.unlocked_quadrants)
    assert bool(tensor.land_count_uncertain[0].item()) == bool(shadow.land_count_uncertain)
    assert int(tensor.shed_reserved[0].item()) == shadow.shed_reserved
    assert bool(tensor.shed_uncertain[0].item()) == bool(shadow.shed_uncertain)
    assert int(tensor.market_slots_used[0].item()) == shadow.market_slots_used
    assert bool(tensor.market_stopped[0].item()) == bool(shadow.market_stopped)
    for name, item_id in ITEM_TO_ID.items():
        assert int(tensor.shed[0, item_id].item()) == int(shadow.shed.get(name, 0)), name
        assert int(tensor.seeds[0, item_id].item()) == int(shadow.seeds.get(name, 0)), name



def test_tensor_ledger_from_batch_matches_from_states():
    state = _state(
        money=2345,
        hires=3,
        shed={"WHEAT": 7, "MILK": 2},
        inventory={"FERTILIZER": 2, "WHEAT": 5},
        seeds={"CARROT": 4},
        hands=2,
    )
    state["step"] = 100
    state["day"] = 4
    state["own_grid"][1][1] = {
        "kind": "WEED",
        "fertilizer_available": True,
    }
    state["own_grid"][2][2] = {
        "kind": "PLANT",
        "crop": "CARROT",
        "watered_today": True,
        "yield_units": 3,
        "planted_day": 2,
        "fertilized_until_day": 6,
    }
    state["own_grid"][3][3] = {
        "kind": "PASTURE",
        "animal": "COW",
        "fed_today": True,
        "cared_today": True,
        "yield_units": 2,
    }
    action = {
        "farmer": {
            "op": "PASS", "item": None, "quantity": None, "raw": ["PASS"],
        },
        "hands": [
            {"op": "PASS", "item": None, "quantity": None, "raw": ["PASS"]},
            {"op": "PASS", "item": None, "quantity": None, "raw": ["PASS"]},
        ],
        "market": [
            {
                "kind": "STOP_QUEUE",
                "op": None,
                "item": None,
                "quantity": None,
                "raw": [],
            }
        ],
    }
    batch = collate_transitions([
        {
            "episode_id": 1,
            "seat": 0,
            "step": 100,
            "state": state,
            "canonical_action": action,
            "previous_action": {},
            "previous_effect": {},
            "effects": {},
        }
    ])
    reference = TensorLedger.from_states(batch.structured_states)
    rebuilt = TensorLedger.from_batch(
        batch, structured_states=batch.structured_states,
    )

    for field in fields(reference):
        left = getattr(reference, field.name)
        right = getattr(rebuilt, field.name)
        if torch.is_tensor(left):
            assert torch.equal(left, right), field.name
        else:
            assert left == right, field.name

    for actor_index in range(reference.max_units):
        ref_legal = reference.legal_unit(actor_index)
        got_legal = rebuilt.legal_unit(actor_index)
        assert torch.equal(ref_legal.op_mask, got_legal.op_mask)
        assert torch.equal(ref_legal.item_mask, got_legal.item_mask)
        assert torch.equal(ref_legal.quantity_max, got_legal.quantity_max)

    for slot in range(10):
        ref_market = reference.legal_market(slot)
        got_market = rebuilt.legal_market(slot)
        assert torch.equal(ref_market.op_mask, got_market.op_mask)
        assert torch.equal(ref_market.item_mask, got_market.item_mask)
        assert torch.equal(ref_market.quantity_max, got_market.quantity_max)


def test_tensor_market_legal_matches_shadow_across_resource_states():
    states = [
        _state(money=3),
        _state(money=3000, shed={"WHEAT": 5}),
        _state(money=200, shed={"WHEAT": 99}),
    ]
    for state in states:
        shadow = ShadowLedger.from_state(state)
        tensor = TensorLedger.from_states([state])
        _assert_market_legal_parity(shadow, tensor, 0)


def test_tensor_market_sequence_matches_shadow_state_and_masks():
    state = _state(money=3000, shed={"WHEAT": 5})
    shadow = ShadowLedger.from_state(state)
    tensor = TensorLedger.from_states([state])
    actions = [
        ("HIRE", None, None),
        ("BUY_SEED", "CARROT", 3),
        ("SELL", "WHEAT", 2),
        ("BUY_ANIMAL", "COW", 1),
        ("BUY_PRODUCT", "WHEAT", 2),
        ("NOP_SLOT", None, None),
        ("STOP_QUEUE", None, None),
    ]
    for slot, (op, item, quantity) in enumerate(actions):
        _assert_market_legal_parity(shadow, tensor, slot)
        if op in {"STOP_QUEUE", "NOP_SLOT"}:
            shadow_action = {"kind": op}
        else:
            shadow_action = {"kind": "ORDER", "op": op, "item": item, "quantity": quantity}
        shadow.apply_market(shadow_action)
        op_t, item_t, qty_t, known_t = _tensor_market_action(op, item, quantity)
        tensor.apply_market(op_t, item_t, qty_t, known_executed=known_t)
        _assert_core_state_parity(shadow, tensor)


def test_tensor_known_executed_market_updates_match_shadow():
    state = _state(money=3000, shed={"WHEAT": 5})
    for op, item, quantity in [
        ("HIRE", None, None),
        ("BUY_LAND", None, None),
        ("BUY_SEED", "WHEAT", 2),
        ("BUY_ANIMAL", "GOOSE", 1),
        ("BUY_PRODUCT", "FERTILIZER", 2),
        ("SELL", "WHEAT", 2),
    ]:
        shadow = ShadowLedger.from_state(state)
        tensor = TensorLedger.from_states([state])
        if op in {"STOP_QUEUE", "NOP_SLOT"}:
            action = {"kind": op, "_executed": True}
        else:
            action = {
                "kind": "ORDER",
                "op": op,
                "item": item,
                "quantity": quantity,
                "_executed": True,
            }
        shadow.apply_market(action)
        op_t, item_t, qty_t, known_t = _tensor_market_action(
            op, item, quantity, known=True
        )
        tensor.apply_market(op_t, item_t, qty_t, known_executed=known_t)
        _assert_core_state_parity(shadow, tensor)


def test_tensor_unit_pickup_and_drop_match_shadow():
    state = _state(
        shed={"WHEAT": 9, "CARROT": 90},
        inventory={"MILK": 3},
    )
    shadow = ShadowLedger.from_state(state)
    tensor = TensorLedger.from_states([state])

    for op, item, quantity in [
        ("PICKUP", "WHEAT", 4),
        ("DROP", None, None),
    ]:
        shadow.apply_unit(
            "farmer",
            {"op": op, "item": item, "quantity": quantity},
        )
        op_t, item_t, qty_t = _tensor_unit_action(op, item, quantity)
        tensor.apply_unit(0, op_t, item_t, qty_t)

    for name, item_id in ITEM_TO_ID.items():
        assert int(tensor.shed[0, item_id].item()) == int(shadow.shed.get(name, 0))
        assert int(tensor.inventory[0, 0, item_id].item()) == int(
            shadow.unit_inventories["farmer"].get(name, 0)
        )


def test_tensor_ledger_vector_matches_shadow_reference():
    state = _state(money=1234, hires=3, shed={"WHEAT": 7}, seeds={"CARROT": 2})
    shadow = ShadowLedger.from_state(state)
    tensor = TensorLedger.from_states([state])
    ref = torch.zeros(1, dtype=torch.float32)
    from kaggrl.v2_model import RecurrentIntentPolicy

    shadow_vec = RecurrentIntentPolicy._ledger_vector(shadow, ref)
    tensor_vec = tensor.ledger_vector(ref)[0]
    assert torch.allclose(tensor_vec, shadow_vec, atol=1e-6, rtol=0.0)
    wheat = ITEM_TO_ID["WHEAT"]
    assert int(tensor.shed[0, wheat]) == shadow.shed.get("WHEAT", 0)
    assert int(tensor.inventory[0, 0, wheat]) == (
        shadow.unit_inventories["farmer"].get("WHEAT", 0)
    )
    assert tensor.positions[0, 0].tolist() == shadow.unit_positions["farmer"]


def test_tensor_ledger_market_updates_match_shadow():
    state = _state(
        money=1000, shed={"WHEAT": 10}, seeds={"WHEAT": 1}
    )
    shadow = ShadowLedger.from_state(state)
    tensor = TensorLedger.from_states([state])
    commands = (
        {"kind": "ORDER", "op": "SELL", "item": "WHEAT", "quantity": 5},
        {"kind": "ORDER", "op": "HIRE", "item": None, "quantity": None},
        {"kind": "ORDER", "op": "BUY_SEED", "item": "WHEAT", "quantity": 2},
        {"kind": "ORDER", "op": "BUY_ANIMAL", "item": "COW", "quantity": 1},
        {"kind": "STOP_QUEUE", "op": None, "item": None, "quantity": None},
    )
    for command in commands:
        shadow.apply_market(command)
        kind = str(command.get("kind", "ORDER"))
        name = kind if kind in {"STOP_QUEUE", "NOP_SLOT"} else command["op"]
        op, item, qty, known = _tensor_market_action(
            name,
            item=command.get("item"),
            quantity=command.get("quantity"),
        )
        tensor.apply_market(op, item, qty, known_executed=known)
    assert int(tensor.cash[0]) == shadow.cash_lower_bound
    assert int(tensor.hires_today[0]) == shadow.hires_today
    assert int(tensor.market_slots_used[0]) == shadow.market_slots_used
    assert bool(tensor.market_stopped[0]) == shadow.market_stopped
    for name, item_id in ITEM_TO_ID.items():
        assert int(tensor.shed[0, item_id]) == shadow.shed.get(name, 0)
    for name in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"):
        item_id = ITEM_TO_ID[name]
        assert int(tensor.seeds[0, item_id]) == shadow.seeds.get(name, 0)


def test_tensor_ledger_vector_matches_shadow_vector_contract():
    from kaggrl.v2_model import RecurrentIntentPolicy

    state = _state(
        money=1234, hires=3,
        shed={"WHEAT": 4, "MILK": 2}, seeds={"WHEAT": 2},
    )
    shadow = ShadowLedger.from_state(state)
    tensor = TensorLedger.from_states([state])
    ref = torch.zeros(1, dtype=torch.float32)
    expected = RecurrentIntentPolicy._ledger_vector(
        shadow, ref
    )
    got = tensor.ledger_vector(ref)[0]
    assert torch.allclose(got, expected, atol=1e-6, rtol=0)
