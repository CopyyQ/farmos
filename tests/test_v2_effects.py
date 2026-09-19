from copy import deepcopy

from kaggrl.v2_effects import derive_effects


def _tiles():
    return [[None for _ in range(10)] for _ in range(10)]


def _obs(*, day=3, hour=8, money=1000, hands=None, shed=None, seeds=None):
    hands = [[5, 4]] if hands is None else hands
    inventories = [{}, *[{} for _ in hands]]
    return {
        "player": 0, "day": day, "hour": hour,
        "farms": [
            {"money": money, "farmer": [4, 4], "hands": deepcopy(hands),
             "hires_today": len(hands), "unlocked_quadrants": ["NW"], "tiles": _tiles()},
            {"money": 900, "farmer": [8, 8], "hands": [],
             "hires_today": 0, "unlocked_quadrants": ["NW"], "tiles": _tiles()},
        ],
        "private": {"shed": dict(shed or {}), "seeds": dict(seeds or {}), "inventories": inventories},
        "market": {"inventory": {"WHEAT": 10000, "MILK": 10000}, "prices": {"WHEAT": 25, "MILK": 160}},
        "town": {"unlocked_shops": []},
    }


def _status(effects, actor, op):
    rows = [e for e in effects.action_evidence if e.actor == actor and e.op == op]
    assert len(rows) == 1
    return rows[0].status


def test_hire_success_and_failed_hire_are_separated_by_observed_effect():
    before = _obs(money=1000, hands=[])
    after = _obs(money=999, hands=[[4, 5]])
    ok = derive_effects(before, {"farmer": ["PASS"], "hands": [], "market": [["HIRE"]]}, after)
    assert ok.hand_count_delta == 1
    assert ok.money_delta == -1
    assert _status(ok, "market:0", "HIRE") == "confirmed"

    failed = derive_effects(before, {"farmer": ["PASS"], "hands": [], "market": [["HIRE"]]}, deepcopy(before))
    assert _status(failed, "market:0", "HIRE") == "failed"


def test_buy_seed_requires_observed_seed_increase_to_confirm():
    before = _obs(money=1000, seeds={"TOMATO": 2})
    after = _obs(money=950, seeds={"TOMATO": 3})
    action = {"farmer": ["PASS"], "hands": [["PASS"]], "market": [["BUY_SEED", "TOMATO", 1]]}
    effects = derive_effects(before, action, after)
    assert effects.seed_delta["TOMATO"] == 1
    assert _status(effects, "market:0", "BUY_SEED") == "confirmed"


def test_buy_without_supporting_delta_is_not_confirmed():
    before = _obs(money=10, seeds={"MELON": 0})
    action = {"farmer": ["PASS"], "hands": [["PASS"]], "market": [["BUY_SEED", "MELON", 1]]}
    effects = derive_effects(before, action, deepcopy(before))
    assert _status(effects, "market:0", "BUY_SEED") == "failed"


def test_move_and_harvest_use_actor_specific_observed_changes():
    before = _obs()
    moved = deepcopy(before)
    moved["farms"][0]["farmer"] = [5, 4]
    move_effects = derive_effects(before, {"farmer": ["EAST"], "hands": [["PASS"]], "market": []}, moved)
    assert move_effects.unit_position_delta["farmer"] == [1, 0]
    assert _status(move_effects, "farmer", "EAST") == "confirmed"

    harvest_before = _obs()
    harvest_after = deepcopy(harvest_before)
    harvest_after["private"]["inventories"][0]["MILK"] = 3
    harvest = derive_effects(harvest_before, {"farmer": ["HARVEST"], "hands": [["PASS"]], "market": []}, harvest_after)
    assert _status(harvest, "farmer", "HARVEST") == "confirmed"


def test_place_animal_confirmation_comes_from_next_public_tile():
    before = _obs(shed={"COW": 1})
    before["farms"][0]["tiles"][4][4] = {"kind": "PASTURE"}
    before["private"]["inventories"][0] = {"COW": 1}
    after = deepcopy(before)
    after["private"]["inventories"][0] = {}
    after["farms"][0]["tiles"][4][4] = {"kind": "PASTURE", "animal": "COW", "placed_day": 3}
    effects = derive_effects(before, {"farmer": ["PLACE", "COW"], "hands": [["PASS"]], "market": []}, after)
    assert _status(effects, "farmer", "PLACE") == "confirmed"


def test_resource_market_and_day_reset_deltas_are_exact():
    before = _obs(day=3, hour=23, shed={"MILK": 5}, seeds={"WHEAT": 4}, hands=[[5, 4], [4, 5]])
    after = _obs(day=4, hour=0, money=1200, shed={"MILK": 8}, seeds={"WHEAT": 2}, hands=[])
    after["market"]["inventory"]["MILK"] = 10007
    after["market"]["prices"]["MILK"] = 150
    effects = derive_effects(before, {"farmer": ["PASS"], "hands": [["PASS"], ["PASS"]], "market": []}, after)
    assert effects.day_changed is True
    assert effects.day_reset is True
    assert effects.hand_count_delta == -2
    assert effects.shed_delta["MILK"] == 3
    assert effects.seed_delta["WHEAT"] == -2
    assert effects.market_inventory_delta["MILK"] == 7
    assert effects.market_price_delta["MILK"] == -10


def test_opponent_public_changes_are_inferred_not_confirmed():
    before = _obs()
    after = deepcopy(before)
    after["farms"][1]["money"] += 500
    after["farms"][1]["farmer"] = [7, 8]
    effects = derive_effects(before, {"farmer": ["PASS"], "hands": [["PASS"]], "market": []}, after)
    assert effects.opponent_public["confidence"] == "inferred"
    assert effects.opponent_public["money_delta"] == 500
    assert effects.opponent_public["farmer_position_delta"] == [-1, 0]


def test_buy_product_is_not_confirmed_when_unit_deposit_confounds_shed_delta():
    before = _obs(money=1000, shed={"WHEAT": 0})
    before["private"]["inventories"][0] = {"WHEAT": 2}
    after = deepcopy(before)
    after["private"]["inventories"][0] = {}
    after["private"]["shed"]["WHEAT"] = 2
    action = {
        "farmer": ["DROP"],
        "hands": [["PASS"]],
        "market": [["BUY_PRODUCT", "WHEAT", 1]],
    }
    effects = derive_effects(before, action, after)
    assert _status(effects, "farmer", "DROP") == "confirmed"
    assert _status(effects, "market:0", "BUY_PRODUCT") == "unconfirmed"


def test_derive_effects_does_not_deep_normalize_full_observations(monkeypatch):
    import kaggrl.v2_effects as effects_module

    def forbidden(*args, **kwargs):
        raise AssertionError("derive_effects must operate on raw observation deltas")

    monkeypatch.setattr(effects_module, "normalize_observation", forbidden, raising=False)
    before = _obs(money=100)
    after = deepcopy(before)
    after["farms"][0]["money"] = 95
    result = effects_module.derive_effects(before, {"farmer": ["PASS"], "hands": [], "market": []}, after)
    assert result.money_delta == -5


def test_partial_multi_hire_is_unconfirmed_per_slot():
    before = _obs(money=1000, hands=[])
    after = _obs(money=999, hands=[[4, 5]])
    action = {"farmer": ["PASS"], "hands": [], "market": [["HIRE"], ["HIRE"], ["HIRE"]]}
    effects = derive_effects(before, action, after)
    assert [_status(effects, f"market:{i}", "HIRE") for i in range(3)] == ["unconfirmed"] * 3


def test_buy_and_sell_same_product_are_unconfirmed_when_net_shed_delta_is_ambiguous():
    before = _obs(money=1000, shed={"WHEAT": 10})
    after = deepcopy(before)
    action = {
        "farmer": ["PASS"], "hands": [["PASS"]],
        "market": [["SELL", "WHEAT", 1], ["BUY_PRODUCT", "WHEAT", 1]],
    }
    effects = derive_effects(before, action, after)
    assert _status(effects, "market:0", "SELL") == "unconfirmed"
    assert _status(effects, "market:1", "BUY_PRODUCT") == "unconfirmed"


def test_day_end_drop_is_unconfirmed_because_automatic_inventory_drop_confounds_it():
    before = _obs(day=3, hour=23, shed={"MILK": 0})
    before["private"]["inventories"][0] = {"MILK": 2}
    after = _obs(day=4, hour=0, shed={"MILK": 2}, hands=[])
    effects = derive_effects(before, {"farmer": ["DROP"], "hands": [["PASS"]], "market": []}, after)
    assert _status(effects, "farmer", "DROP") == "unconfirmed"


def test_invalid_buy_product_item_is_failed_even_if_shed_increases_for_other_reason():
    before = _obs(money=1000, shed={"CARROT": 0})
    before["private"]["inventories"][0] = {"CARROT": 2}
    after = deepcopy(before)
    after["private"]["inventories"][0] = {}
    after["private"]["shed"]["CARROT"] = 2
    action = {
        "farmer": ["DROP"], "hands": [["PASS"]],
        "market": [["BUY_PRODUCT", "CARROT", 1]],
    }
    effects = derive_effects(before, action, after)
    assert _status(effects, "market:0", "BUY_PRODUCT") == "failed"


def test_all_multi_hires_confirm_when_observed_hand_growth_matches_all_slots():
    before = _obs(money=1000, hands=[])
    after = _obs(money=996, hands=[[4, 5], [5, 4], [4, 4]])
    action = {"farmer": ["PASS"], "hands": [], "market": [["HIRE"], ["HIRE"], ["HIRE"]]}
    effects = derive_effects(before, action, after)
    assert [_status(effects, f"market:{i}", "HIRE") for i in range(3)] == ["confirmed"] * 3


def test_buy_seed_is_unconfirmed_when_same_turn_plant_confounds_seed_delta():
    before = _obs(money=1000, seeds={"TOMATO": 2})
    after = deepcopy(before)
    after["private"]["seeds"]["TOMATO"] = 2
    action = {
        "farmer": ["PLANT", "TOMATO"], "hands": [["PASS"]],
        "market": [["BUY_SEED", "TOMATO", 1]],
    }
    effects = derive_effects(before, action, after)
    assert _status(effects, "market:0", "BUY_SEED") == "unconfirmed"
