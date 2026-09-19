from copy import deepcopy

from kaggrl.v2_effect_tracker import EffectTracker


def _tiles():
    return [[None for _ in range(10)] for _ in range(10)]


def _obs(*, step=8, day=0, hour=8, money=1000, hands=None, shed=None, seeds=None):
    hands = [[5, 4]] if hands is None else deepcopy(hands)
    inventories = [{}, *[{} for _ in hands]]
    return {
        "player": 0, "step": step, "day": day, "hour": hour,
        "farms": [
            {"money": money, "farmer": [4, 4], "hands": hands,
             "hires_today": len(hands), "unlocked_quadrants": ["NW"], "tiles": _tiles()},
            {"money": 900, "farmer": [8, 8], "hands": [], "hires_today": 0,
             "unlocked_quadrants": ["NW"], "tiles": _tiles()},
        ],
        "private": {"shed": dict(shed or {}), "seeds": dict(seeds or {}),
                    "inventories": inventories},
        "market": {"inventory": {"WHEAT": 10000}, "prices": {"WHEAT": 25}},
        "town": {"unlocked_shops": []},
    }

def test_tracker_confirms_movement_hire_and_buy_only_from_observed_deltas():
    tracker = EffectTracker()
    before = _obs(hands=[])
    moved = deepcopy(before); moved["farms"][0]["farmer"] = [5, 4]
    movement = tracker.observe(before, {"farmer": ["EAST"], "hands": [], "market": []}, moved)
    assert "movement" in movement.effective_families

    hired = _obs(money=999, hands=[[4, 5]])
    hire = tracker.observe(before, {"farmer": ["PASS"], "hands": [], "market": [["HIRE"]]}, hired)
    assert {"acquisition", "hire"}.issubset(hire.effective_families)
    failed = tracker.observe(before, {"farmer": ["PASS"], "hands": [], "market": [["HIRE"]]}, deepcopy(before))
    assert "hire" not in failed.effective_families and failed.failed_count == 1

    buy_before = _obs(seeds={"WHEAT": 1})
    buy_after = _obs(money=975, seeds={"WHEAT": 2})
    buy = tracker.observe(buy_before, {"farmer": ["PASS"], "hands": [["PASS"]],
                                      "market": [["BUY_SEED", "WHEAT", 1]]}, buy_after)
    assert {"acquisition", "purchase"}.issubset(buy.effective_families)

def test_tracker_confirms_plant_service_harvest_deposit_and_sale_from_state_changes():
    tracker = EffectTracker()
    plant_before = _obs(seeds={"WHEAT": 2})
    plant_after = deepcopy(plant_before)
    plant_after["private"]["seeds"]["WHEAT"] = 1
    plant_after["farms"][0]["tiles"][4][4] = {
        "kind": "PLANT", "crop": "WHEAT", "planted_day": 0,
        "watered_today": False,
    }
    plant = tracker.observe(plant_before, {"farmer": ["PLANT", "WHEAT"],
        "hands": [["PASS"]], "market": []}, plant_after)
    assert "production" in plant.effective_families

    water_before = deepcopy(plant_after)
    water_after = deepcopy(water_before)
    water_after["farms"][0]["tiles"][4][4]["watered_today"] = True
    water = tracker.observe(water_before, {"farmer": ["WATER"],
        "hands": [["PASS"]], "market": []}, water_after)
    assert "service" in water.effective_families

    harvest_before = _obs()
    harvest_after = deepcopy(harvest_before)
    harvest_after["private"]["inventories"][0]["WHEAT"] = 3
    harvest = tracker.observe(harvest_before, {"farmer": ["HARVEST"],
        "hands": [["PASS"]], "market": []}, harvest_after)
    assert "harvest" in harvest.effective_families

    deposit_before = _obs(shed={"WHEAT": 0})
    deposit_before["private"]["inventories"][0] = {"WHEAT": 2}
    deposit_after = deepcopy(deposit_before)
    deposit_after["private"]["inventories"][0] = {"WHEAT": 1}
    deposit_after["private"]["shed"]["WHEAT"] = 1
    deposit = tracker.observe(deposit_before, {"farmer": ["PLACE", "WHEAT", 1],
        "hands": [["PASS"]], "market": []}, deposit_after)
    assert "deposit" in deposit.effective_families

    sell_before = _obs(money=1000, shed={"WHEAT": 2})
    sell_after = _obs(money=1025, shed={"WHEAT": 1})
    sell_after["market"]["inventory"]["WHEAT"] = 10001
    sale = tracker.observe(sell_before, {"farmer": ["PASS"], "hands": [["PASS"]],
        "market": [["SELL", "WHEAT", 1]]}, sell_after)
    assert "sale" in sale.effective_families
    assert sale.transition.money_delta == 25


def test_tracker_records_dawn_hand_expiration_without_claiming_effective_action():
    tracker = EffectTracker()
    before = _obs(step=23, day=0, hour=23, hands=[[5, 4], [4, 5]])
    after = _obs(step=24, day=1, hour=0, hands=[])
    record = tracker.observe(before, {"farmer": ["PASS"],
        "hands": [["PASS"], ["PASS"]], "market": []}, after)
    assert record.transition.day_reset is True
    assert record.dawn_hand_expiration == 2
    assert record.effective is False
