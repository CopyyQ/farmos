from copy import deepcopy

import pytest

from kaggrl.v2_ledger import ShadowLedger


def _tiles():
    return [[None for _ in range(10)] for _ in range(10)]


def _state(*, money=3000, hands=0, hires_today=0, shed=None, seeds=None, farmer=(4, 4)):
    grid = _tiles()
    hand_pos = [[(i + 5) % 10, 4 + (i // 5)] for i in range(hands)]
    inventories = [{} for _ in range(hands + 1)]
    own = {"money": money, "farmer": list(farmer), "hands": hand_pos,
           "hires_today": hires_today, "unlocked_quadrants": ["NW"], "tiles": grid}
    rival = {"money": 3000, "farmer": [8, 8], "hands": [], "hires_today": 0,
             "unlocked_quadrants": ["NW"], "tiles": _tiles()}
    units = [{"kind": "farmer", "index": 0, "position": list(farmer), "inventory": inventories[0]}]
    units += [{"kind": "hand", "index": i, "position": hand_pos[i], "inventory": inventories[i + 1]}
              for i in range(hands)]
    return {"player": 0, "step": 100, "day": 4, "hour": 4, "own": own, "rival": rival,
            "private": {"shed": dict(shed or {}), "seeds": dict(seeds or {}),
                        "inventories": inventories},
            "own_grid": grid, "rival_grid": rival["tiles"], "own_units": units,
            "rival_units": [{"kind": "farmer", "index": 0, "position": [8, 8]}],
            "market": {"inventory": {"WHEAT": 10000, "MILK": 10000},
                       "prices": {"WHEAT": 25, "MILK": 160}},
            "town": {"unlocked_shops": []}, "town_shops": []}


def test_unit_mask_uses_engine_legality_not_strategy():
    state = _state(seeds={"WHEAT": 1}, farmer=(0, 0))
    state["own_grid"][0][1] = "LOCKED"
    ledger = ShadowLedger.from_state(state)
    mask = ledger.legal_unit_mask("farmer", {})
    assert not mask.allows("NORTH") and not mask.allows("WEST")
    assert mask.allows("EAST")  # movement onto LOCKED is legal in engine
    assert mask.allows("SOUTH")
    assert mask.allows("PLANT", "WHEAT")
    assert not mask.allows("PLANT", "COW")


def test_dynamic_hand_lookup_has_no_fixed_cap():
    ledger = ShadowLedger.from_state(_state(hands=25))
    assert ledger.legal_unit_mask("hand:24", {}).allows("PASS")
    with pytest.raises(KeyError):
        ledger.legal_unit_mask("hand:25", {})


def test_atomic_plant_demand_never_exceeds_available_seed():
    state = _state(hands=2, seeds={"WHEAT": 2})
    state["own_units"][0]["position"] = [0, 0]
    state["own_units"][1]["position"] = [1, 0]
    state["own_units"][2]["position"] = [2, 0]
    state["own"]["farmer"] = [0, 0]
    state["own"]["hands"] = [[1, 0], [2, 0]]
    ledger = ShadowLedger.from_state(state)
    assert ledger.legal_unit_mask("farmer", {}).allows("PLANT", "WHEAT")
    ledger.apply_unit("farmer", ["PLANT", "WHEAT"])
    assert ledger.legal_unit_mask("hand:0", {}).allows("PLANT", "WHEAT")
    ledger.apply_unit("hand:0", ["PLANT", "WHEAT"])
    assert not ledger.legal_unit_mask("hand:1", {}).allows("PLANT", "WHEAT")


def test_hire_fibonacci_and_land_exhaustion_are_hard_rules():
    ledger = ShadowLedger.from_state(_state(money=10000, hires_today=16))
    assert ledger.next_hire_cost == 1597
    assert ledger.legal_market_mask(0, []).allows("HIRE")
    ledger.apply_market(["HIRE"])
    assert ledger.hires_today == 17
    assert ledger.next_hire_cost == 2584

    full = _state(money=10000)
    full["own"]["unlocked_quadrants"] = ["NW", "NE", "SW", "SE"]
    full_ledger = ShadowLedger.from_state(full)
    assert not full_ledger.legal_market_mask(0, []).allows("BUY_LAND")


def test_market_slot_budget_preserves_stop_and_nop_semantics():
    ledger = ShadowLedger.from_state(_state())
    before_limit = ledger.legal_market_mask(9, [])
    assert before_limit.allows("STOP_QUEUE") and before_limit.allows("NOP_SLOT")
    at_limit = ledger.legal_market_mask(10, [])
    assert at_limit.allows("STOP_QUEUE")
    assert not at_limit.allows("NOP_SLOT")
    assert not at_limit.allows("HIRE")


def test_guaranteed_prior_sale_credit_can_fund_fixed_cost_later_slot():
    state = _state(money=0, hires_today=5, shed={"WHEAT": 10})  # next hire costs 8
    ledger = ShadowLedger.from_state(state)
    assert not ledger.legal_market_mask(0, []).allows("HIRE")
    ledger.apply_market(["SELL", "WHEAT", 1000])
    assert ledger.cash_lower_bound >= 10
    assert ledger.legal_market_mask(1, []).allows("HIRE")


def test_shed_capacity_is_hard_but_opponent_market_price_is_uncertain():
    full_shed = {"WHEAT": 100}
    ledger = ShadowLedger.from_state(_state(money=10000, shed=full_shed))
    mask = ledger.legal_market_mask(0, [])
    assert not mask.allows("BUY_PRODUCT", "WHEAT")
    assert not mask.allows("BUY_ANIMAL", "COW")
    assert mask.allows("SELL", "WHEAT")

    open_ledger = ShadowLedger.from_state(_state(money=100, shed={"WHEAT": 1}))
    open_mask = open_ledger.legal_market_mask(0, [])
    assert open_mask.allows("BUY_PRODUCT", "WHEAT")
    assert open_mask.is_uncertain("BUY_PRODUCT")
    assert open_mask.allows("SELL", "WHEAT")
    assert open_mask.is_uncertain("SELL")


def test_ledger_does_not_encode_roi_preferences():
    state = _state(money=1, shed={"MILK": 1}, seeds={"MELON": 1})
    state["private"]["inventories"][0] = {"WHEAT": 1}
    state["own_units"][0]["inventory"] = {"WHEAT": 1}
    state["own_grid"][4][4] = {"kind": "PASTURE", "animal": "COW", "fed_today": False,
                                "cared_today": False, "fertilizer_available": False, "yield_units": 0}
    ledger = ShadowLedger.from_state(state)
    unit = ledger.legal_unit_mask("farmer", {})
    assert unit.allows("FEED")  # regardless of ROI
    market = ledger.legal_market_mask(0, [])
    assert market.allows("SELL", "MILK")  # even if sale price is unattractive

    plant_state = _state(money=1, seeds={"MELON": 1}, farmer=(0, 0))
    assert ShadowLedger.from_state(plant_state).legal_unit_mask("farmer", {}).allows("PLANT", "MELON")


def test_large_quantities_are_not_clamped_by_ledger():
    state = _state(money=0, shed={"WHEAT": 7})
    ledger = ShadowLedger.from_state(state)
    ledger.apply_market(["SELL", "WHEAT", 1000])
    assert ledger.shed.get("WHEAT", 0) == 0
    assert ledger.cash_lower_bound == 7  # guaranteed floor-price proceeds

    pickup = _state(shed={"WHEAT": 9})
    pickup_ledger = ShadowLedger.from_state(pickup)
    pickup_ledger.apply_unit("farmer", ["PICKUP", "WHEAT", 1000])
    assert pickup_ledger.unit_inventories["farmer"]["WHEAT"] == 9
    assert pickup_ledger.shed.get("WHEAT", 0) == 0


def test_unit_zero_quantity_is_noop_not_one():
    state = _state(shed={"WHEAT": 9})
    ledger = ShadowLedger.from_state(state)
    ledger.apply_unit("farmer", ["PICKUP", "WHEAT", 0])
    assert ledger.shed["WHEAT"] == 9
    assert ledger.unit_inventories["farmer"].get("WHEAT", 0) == 0

    state2 = _state(shed={"WHEAT": 0})
    state2["private"]["inventories"][0] = {"WHEAT": 3}
    state2["own_units"][0]["inventory"] = {"WHEAT": 3}
    ledger2 = ShadowLedger.from_state(state2)
    ledger2.apply_unit("farmer", ["PLACE", "WHEAT", 0])
    assert ledger2.shed.get("WHEAT", 0) == 0
    assert ledger2.unit_inventories["farmer"]["WHEAT"] == 3


def test_sell_quantity_bound_tracks_remaining_known_inventory():
    ledger = ShadowLedger.from_state(_state(money=0, shed={"WHEAT": 10}))
    first = ledger.legal_market_mask(0, [])
    assert first.allows("SELL", "WHEAT")
    assert first.metadata["sell_max_by_item"]["WHEAT"] == 10

    ledger.apply_market(["SELL", "WHEAT", 7])
    second = ledger.legal_market_mask(1, [])
    assert second.allows("SELL", "WHEAT")
    assert second.metadata["sell_max_by_item"]["WHEAT"] == 3

    ledger.apply_market(["SELL", "WHEAT", 3])
    third = ledger.legal_market_mask(2, [])
    assert not third.allows("SELL", "WHEAT")
    assert "WHEAT" not in third.metadata["sell_max_by_item"]


def test_sell_does_not_use_inventory_that_only_might_exist():
    ledger = ShadowLedger.from_state(_state(money=100, shed={"WHEAT": 0}))
    ledger.shed_uncertain = True
    ledger.shed_uncertain_items.add("WHEAT")
    mask = ledger.legal_market_mask(1, [])
    assert not mask.allows("SELL", "WHEAT")
    assert "WHEAT" not in mask.metadata["sell_max_by_item"]


def test_effective_buy_product_updates_known_inventory_for_following_sell():
    ledger = ShadowLedger.from_state(_state(money=3000, shed={"WHEAT": 0}))
    ledger.apply_market({
        "kind": "ORDER", "op": "BUY_PRODUCT", "item": "WHEAT",
        "quantity": 13, "_executed": True,
    })
    assert ledger.shed["WHEAT"] == 13
    mask = ledger.legal_market_mask(1, [])
    assert mask.allows("SELL", "WHEAT")
    assert mask.metadata["sell_max_by_item"]["WHEAT"] == 13
    ledger.apply_market({
        "kind": "ORDER", "op": "SELL", "item": "WHEAT",
        "quantity": 13, "_executed": True,
    })
    assert ledger.shed["WHEAT"] == 0
    assert not ledger.legal_market_mask(2, []).allows("SELL", "WHEAT")


def test_requested_buy_product_does_not_create_guaranteed_sell_inventory():
    ledger = ShadowLedger.from_state(_state(money=3000, shed={"WHEAT": 0}))
    ledger.apply_market(["BUY_PRODUCT", "WHEAT", 13])
    assert ledger.shed.get("WHEAT", 0) == 0
    assert not ledger.legal_market_mask(1, []).allows("SELL", "WHEAT")


def test_effective_sell_rejects_impossible_sidecar_quantity():
    ledger = ShadowLedger.from_state(_state(money=0, shed={"WHEAT": 10}))
    with pytest.raises(RuntimeError, match="executed SELL exceeds"):
        ledger.apply_market({
            "kind": "ORDER", "op": "SELL", "item": "WHEAT",
            "quantity": 11, "_executed": True,
        })


def test_unit_quantity_metadata_tracks_pickup_and_place_limits():
    state = _state(shed={"WHEAT": 7}, farmer=(4, 4))
    state["private"]["inventories"][0] = {"WHEAT": 5}
    state["own_units"][0]["inventory"] = {"WHEAT": 5}
    ledger = ShadowLedger.from_state(state)
    mask = ledger.legal_unit_mask("farmer", {})
    bounds = mask.metadata["unit_quantity_max_by_op_item"]
    assert bounds["PICKUP"]["WHEAT"] == 7
    assert bounds["PLACE"]["WHEAT"] == 5


def test_unknown_buy_reserves_capacity_without_creating_sellable_inventory():
    ledger = ShadowLedger.from_state(
        _state(money=3000, shed={"FERTILIZER": 90})
    )
    first = ledger.legal_market_mask(0, [])
    qmax = first.metadata["market_quantity_max_by_op_item"]
    assert qmax["BUY_PRODUCT"]["WHEAT"] == 10

    ledger.apply_market(["BUY_PRODUCT", "WHEAT", 7])
    assert ledger.shed.get("WHEAT", 0) == 0
    assert ledger.shed_reserved == 7

    second = ledger.legal_market_mask(1, [])
    qmax2 = second.metadata["market_quantity_max_by_op_item"]
    assert ledger.cash_lower_bound == 0
    assert not second.allows("BUY_PRODUCT", "WHEAT")
    assert "WHEAT" not in qmax2["BUY_PRODUCT"]
    assert not second.allows("SELL", "WHEAT")


def test_fixed_cost_market_quantity_bounds_use_money_and_capacity():
    ledger = ShadowLedger.from_state(
        _state(money=1000, shed={"FERTILIZER": 98})
    )
    mask = ledger.legal_market_mask(0, [])
    qmax = mask.metadata["market_quantity_max_by_op_item"]
    assert qmax["BUY_SEED"]["WHEAT"] == 100
    assert qmax["BUY_ANIMAL"]["COW"] == 2
    assert qmax["BUY_PRODUCT"]["WHEAT"] == 2

def test_variable_price_buy_blocks_unfunded_later_market_spend():
    ledger = ShadowLedger.from_state(_state(money=3000, shed={}))
    first = ledger.legal_market_mask(0, [])
    assert first.allows("BUY_PRODUCT", "WHEAT")
    ledger.apply_market(["BUY_PRODUCT", "WHEAT", 1])

    second = ledger.legal_market_mask(1, [])
    assert ledger.cash_uncertain is True
    assert ledger.cash_lower_bound == 0
    assert not second.allows("BUY_PRODUCT", "WHEAT")
    assert not second.allows("BUY_ANIMAL", "COW")
    assert not second.allows("BUY_SEED", "WHEAT")
    assert not second.allows("HIRE")
    assert second.allows("STOP_QUEUE")


def test_buy_product_requires_current_quote_affordability():
    ledger = ShadowLedger.from_state(_state(money=24, shed={}))
    mask = ledger.legal_market_mask(0, [])
    assert not mask.allows("BUY_PRODUCT", "WHEAT")
