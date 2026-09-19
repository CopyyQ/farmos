from copy import deepcopy

from kaggle_environments.envs.kaggriculture import kaggriculture as kg

from kaggrl.v2_ledger import ShadowLedger


def _tiles():
    return [[None if x < 5 and y < 5 else "LOCKED" for x in range(10)] for y in range(10)]


def _state(*, money=3000, hires=0, shed=None, inventory=None, seeds=None, hands=0):
    grid = _tiles()
    inv = dict(inventory or {})
    hand_positions = [[3 - i, 4] for i in range(hands)]
    own = {"money": money, "farmer": [4, 4], "hands": hand_positions, "hires_today": hires,
           "unlocked_quadrants": ["NW"], "tiles": grid}
    return {"player": 0, "day": 4, "hour": 4, "own": own,
            "rival": {"money": 3000, "farmer": [8, 8], "hands": [], "hires_today": 0,
                      "unlocked_quadrants": ["NW"], "tiles": _tiles()},
            "private": {
                "shed": dict(shed or {}),
                "seeds": dict(seeds or {}),
                "inventories": [inv] + [{} for _ in range(hands)],
            },
            "own_grid": grid, "rival_grid": _tiles(),
            "own_units": (
                [{"kind": "farmer", "index": 0, "position": [4, 4], "inventory": inv}]
                + [
                    {"kind": "hand", "index": i, "position": pos, "inventory": {}}
                    for i, pos in enumerate(hand_positions)
                ]
            ),
            "rival_units": [{"kind": "farmer", "index": 0, "position": [8, 8]}],
            "market": {"inventory": {item: 10000 for item in kg.PRODUCTS},
                       "prices": {item: kg.MARKET_PARAMS[item]["base"] for item in kg.PRODUCTS}},
            "town": {"unlocked_shops": []}, "town_shops": []}


def test_pickup_large_quantity_matches_engine_effect():
    state = _state(shed={"WHEAT": 9})
    farm = deepcopy(state["own"]); private = deepcopy(state["private"])
    kg._apply_unit_action(farm, private, 0, ["PICKUP", "WHEAT", 1000], 10, 4, 24, 100)
    ledger = ShadowLedger.from_state(state)
    ledger.apply_unit("farmer", ["PICKUP", "WHEAT", 1000])
    assert ledger.shed.get("WHEAT", 0) == private["shed"].get("WHEAT", 0)
    assert ledger.unit_inventories["farmer"] == private["inventories"][0]


def test_drop_overflow_matches_engine_discard_semantics():
    state = _state(shed={"WHEAT": 99}, inventory={"MILK": 3})
    farm = deepcopy(state["own"]); private = deepcopy(state["private"])
    kg._apply_unit_action(farm, private, 0, ["DROP"], 10, 4, 24, 100)
    ledger = ShadowLedger.from_state(state)
    ledger.apply_unit("farmer", ["DROP"])
    assert sum(ledger.shed.values()) == sum(private["shed"].values()) == 100
    assert ledger.unit_inventories["farmer"] == private["inventories"][0] == {}


def test_hire_fibonacci_effect_matches_engine_when_affordability_is_exact():
    state = _state(money=10000, hires=16)
    farm = deepcopy(state["own"]); private = deepcopy(state["private"])
    kg._do_hire(farm, private, 10, 1)
    ledger = ShadowLedger.from_state(state)
    ledger.apply_market(["HIRE"])
    assert ledger.cash_lower_bound == farm["money"]
    assert ledger.hires_today == farm["hires_today"] == 17
    assert len(farm["hands"]) == 1


def test_buy_land_effect_matches_engine_when_affordability_is_exact():
    state = _state(money=3000)
    farm = deepcopy(state["own"])
    kg._do_buy_land(farm, 10)
    ledger = ShadowLedger.from_state(state)
    ledger.apply_market(["BUY_LAND"])
    assert ledger.cash_lower_bound == farm["money"] == 2000
    assert ledger.unlocked_quadrants == farm["unlocked_quadrants"] == ["NW", "NE"]


def test_nine_sequential_hires_match_engine_and_cost_exactly_88():
    state = _state(money=3000, hires=0)
    farm = deepcopy(state["own"])
    private = deepcopy(state["private"])
    ledger = ShadowLedger.from_state(state)
    for slot in range(9):
        assert ledger.legal_market_mask(slot, {}).allows("HIRE")
        kg._do_hire(farm, private, 10, 1)
        ledger.apply_market(["HIRE"])
    assert farm["money"] == ledger.cash_lower_bound == 2912
    assert farm["hires_today"] == ledger.hires_today == 9
    assert len(farm["hands"]) == 9
    assert ledger.next_hire_cost == 55


def test_cumulative_hire_cost_masks_next_unaffordable_slot():
    ledger = ShadowLedger.from_state(_state(money=3, hires=0))
    assert ledger.legal_market_mask(0, {}).allows("HIRE")
    ledger.apply_market(["HIRE"])
    assert ledger.legal_market_mask(1, {}).allows("HIRE")
    ledger.apply_market(["HIRE"])
    assert ledger.cash_lower_bound == 1
    assert ledger.next_hire_cost == 2
    assert not ledger.legal_market_mask(2, {}).allows("HIRE")


def test_atomic_plant_block_matches_engine_all_or_nothing_semantics():
    state = _state(seeds={"CARROT": 1}, hands=1)
    actions = [["PLANT", "CARROT"], ["PLANT", "CARROT"]]

    farm = deepcopy(state["own"])
    private = deepcopy(state["private"])
    demand = {"CARROT": 2}
    blocked = {
        crop for crop, n in demand.items()
        if n > int(private["seeds"].get(crop, 0))
    }
    for unit_index, action in enumerate(actions):
        allowed = ["PASS"] if action[1] in blocked else action
        kg._apply_unit_action(
            farm, private, unit_index, allowed, 10, 4, 24, 100,
        )

    ledger = ShadowLedger.from_state(state)
    ledger.set_atomic_plant_blocked(blocked)
    ledger.apply_unit("farmer", actions[0])
    ledger.apply_unit("hand:0", actions[1])

    assert private["seeds"]["CARROT"] == 1
    assert ledger.seeds["CARROT"] == 1
    assert farm["tiles"][4][4] is None
    assert farm["tiles"][4][3] is None
    assert ledger.grid[4][4] is None
    assert ledger.grid[4][3] is None
