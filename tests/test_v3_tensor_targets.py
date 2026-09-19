import torch

from kaggrl.constants import ITEM_TO_ID, UNIT_OPS
from kaggrl.v2_ledger import MARKET_OPS
from kaggrl.v2_quantity import DIGIT_OFFSET, END_ID, OMIT_ID
from kaggrl.v3_tensor_targets import (
    MARKET_OP_TO_ID,
    UNIT_OP_TO_ID,
    TensorActionTargets,
)


def test_tensor_action_targets_encode_units_market_and_execution():
    actions = [{
        "farmer": {
            "op": "PICKUP",
            "item": "WHEAT",
            "quantity": 12,
        },
        "hands": [{
            "op": "PASS",
            "item": None,
            "quantity": None,
        }],
        "market": [
            {
                "kind": "ORDER",
                "op": "BUY_ANIMAL",
                "item": "COW",
                "quantity": 2,
                "_executed": True,
            },
            {
                "kind": "STOP_QUEUE",
                "op": None,
                "item": None,
                "quantity": None,
            },
        ],
    }]
    targets = TensorActionTargets.from_actions(actions)
    assert targets.unit_op.shape == (1, 2)
    assert targets.unit_op[0, 0].item() == UNIT_OP_TO_ID["PICKUP"]
    assert targets.unit_item[0, 0].item() == ITEM_TO_ID["WHEAT"]
    assert targets.unit_quantity[0, 0].item() == 12
    assert targets.unit_mask.tolist() == [[True, True]]
    expected = [DIGIT_OFFSET + 1, DIGIT_OFFSET + 2, END_ID]
    assert targets.unit_quantity_tokens[0, 0, :3].tolist() == expected
    assert targets.unit_quantity_token_mask[0, 0, :3].tolist() == [True, True, True]

    assert targets.market_op[0, 0].item() == MARKET_OP_TO_ID["BUY_ANIMAL"]
    assert targets.market_item[0, 0].item() == ITEM_TO_ID["COW"]
    assert targets.market_quantity[0, 0].item() == 2
    assert targets.market_executed[0, 0].item() is True
    assert targets.market_op[0, 1].item() == MARKET_OP_TO_ID["STOP_QUEUE"]
    assert targets.market_mask[0, :2].tolist() == [True, True]


def test_tensor_action_targets_pads_variable_unit_width():
    actions = [
        {
            "farmer": {"op": "PASS"},
            "hands": [],
            "market": [],
        },
        {
            "farmer": {"op": "PASS"},
            "hands": [
                {"op": "EAST"},
                {"op": "WEST"},
            ],
            "market": [{"kind": "NOP_SLOT"}],
        },
    ]
    targets = TensorActionTargets.from_actions(actions)
    assert targets.unit_op.shape == (2, 3)
    assert targets.unit_mask.tolist() == [
        [True, False, False],
        [True, True, True],
    ]
    assert targets.market_mask[0, 0].item() is True
    assert targets.market_op[0, 0].item() == MARKET_OP_TO_ID["STOP_QUEUE"]
    assert targets.market_op[1, 0].item() == MARKET_OP_TO_ID["NOP_SLOT"]


def test_atomic_plant_blocked_is_computed_as_tensor_batch():
    actions = [
        {
            "farmer": {"op": "PLANT", "item": "CARROT"},
            "hands": [{"op": "PLANT", "item": "CARROT"}],
            "market": [],
        },
        {
            "farmer": {"op": "PLANT", "item": "WHEAT"},
            "hands": [],
            "market": [],
        },
    ]
    targets = TensorActionTargets.from_actions(actions)
    seeds = torch.zeros((2, max(ITEM_TO_ID.values()) + 1), dtype=torch.long)
    seeds[0, ITEM_TO_ID["CARROT"]] = 1
    seeds[1, ITEM_TO_ID["WHEAT"]] = 1
    blocked = targets.atomic_plant_blocked(seeds)
    assert blocked[0, ITEM_TO_ID["CARROT"]].item() is True
    assert blocked[1, ITEM_TO_ID["WHEAT"]].item() is False


def test_tensor_targets_use_frozen_vocabularies():
    assert set(UNIT_OP_TO_ID) == set(UNIT_OPS)
    assert set(MARKET_OP_TO_ID) == set(MARKET_OPS)


def test_omitted_quantity_uses_omit_token():
    targets = TensorActionTargets.from_actions([{
        "farmer": {"op": "PASS", "quantity": None},
        "hands": [],
        "market": [{"kind": "STOP_QUEUE"}],
    }])
    assert targets.unit_quantity[0, 0].item() == -1
    assert targets.unit_quantity_tokens[0, 0, 0].item() == OMIT_ID
    assert targets.unit_quantity_token_mask[0, 0, 0].item() is True
