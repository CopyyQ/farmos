import pytest

from kaggle_environments.envs.kaggriculture import kaggriculture as kg

from kaggrl.constants import PRODUCTS
from kaggrl.v4_market_race import (
    MarketRaceTracker,
    front_run_future_sales,
    market_price,
    optimize_sell_order,
    simulate_lockstep_sell_margin,
    suppress_due_sales,
)


def _obs():
    return {
        "player": 0,
        "market": {
            "inventory": {item: 10000 for item in PRODUCTS},
            "prices": {
                item: kg.market_price(item, 10000)
                for item in PRODUCTS
            },
        },
        "farms": [
            {"tiles": []},
            {"tiles": []},
        ],
    }


@pytest.mark.parametrize("inventory", [9400, 9999, 10000, 10050, 10400])
def test_market_price_matches_reference_engine(inventory):
    for item in PRODUCTS:
        assert market_price(item, inventory) == kg.market_price(
            item, inventory
        )


def test_lockstep_same_slot_uses_precommit_quote_for_both_players():
    obs = _obs()
    ours = [["SELL", "WOOL", 3]]
    theirs = [["SELL", "WOOL", 3]]
    own, rival, margin = simulate_lockstep_sell_margin(
        obs, ours, theirs
    )
    assert own == rival
    assert margin == 0.0


def test_sell_order_moves_race_sensitive_item_earlier():
    obs = _obs()
    obs["farms"][1]["tiles"] = [[{
        "animal": "SHEEP", "yield_units": 20,
    }]]
    tracker = MarketRaceTracker()
    tracker.pressure["WOOL"] = 20.0
    tracker.pressure["MILK"] = 1.0
    action = {
        "farmer": ["PASS"],
        "hands": [],
        "market": [
            ["SELL", "MILK", 5],
            ["SELL", "WOOL", 5],
        ],
    }

    optimized, meta = optimize_sell_order(
        obs, action, tracker, min_expected_margin_gain=0.0
    )

    assert meta.applied is True
    assert optimized["market"][0][1] == "WOOL"
    assert optimized["market"][1][1] == "MILK"
    assert meta.optimized_score > meta.baseline_score


def test_sell_order_never_crosses_non_sell_dependency_slots():
    obs = _obs()
    tracker = MarketRaceTracker()
    tracker.pressure["WOOL"] = 30.0
    action = {
        "farmer": ["PASS"],
        "hands": [],
        "market": [
            ["BUY_SEED", "WHEAT", 2],
            ["SELL", "MILK", 5],
            ["HIRE"],
            ["SELL", "WOOL", 5],
        ],
    }

    optimized, meta = optimize_sell_order(
        obs, action, tracker, min_expected_margin_gain=0.0
    )

    assert optimized["market"] == action["market"]
    assert meta.applied is False


def test_sell_order_reorders_only_inside_contiguous_sell_block():
    obs = _obs()
    obs["farms"][1]["tiles"] = [[{
        "animal": "SHEEP", "yield_units": 20,
    }]]
    tracker = MarketRaceTracker()
    tracker.pressure["WOOL"] = 30.0
    action = {
        "farmer": ["PASS"],
        "hands": [],
        "market": [
            ["SELL", "MILK", 5],
            ["SELL", "WOOL", 5],
            ["HIRE"],
        ],
    }
    optimized, meta = optimize_sell_order(
        obs, action, tracker, min_expected_margin_gain=0.0
    )
    assert optimized["market"][2] == ["HIRE"]
    assert optimized["market"][0][1] == "WOOL"
    assert optimized["market"][1][1] == "MILK"
    assert meta.applied is True


def test_tracker_infers_rival_sale_after_subtracting_our_previous_flow():
    tracker = MarketRaceTracker(decay=0.0)
    before = _obs()
    tracker.observe(before)
    tracker.record_action({
        "market": [["SELL", "MILK", 4]],
    })

    after = _obs()
    after["market"]["inventory"]["MILK"] = 10010
    tracker.observe(after)

    # Inventory rose by 10, four units were ours -> six inferred rival sales.
    assert tracker.pressure["MILK"] == pytest.approx(6.0)


class _FutureSaleMacro:
    def action_for_route(self, obs, route_id, configuration=None):
        if int(obs["step"]) == 201:
            return {
                "farmer": ["PASS"],
                "hands": [],
                "market": [["SELL", "WOOL", 4]],
            }
        return {"farmer": ["PASS"], "hands": [], "market": []}


def _front_run_obs(price):
    obs = _obs()
    obs.update({
        "step": 200,
        "day": 8,
        "hour": 8,
        "private": {"shed": {"WOOL": 5}},
    })
    obs["market"]["prices"]["WOOL"] = int(price)
    return obs


def test_front_run_pulls_only_future_sale_when_price_above_base():
    action = {"farmer": ["PASS"], "hands": [], "market": []}
    pulled, debts = front_run_future_sales(
        _front_run_obs(220),
        action,
        _FutureSaleMacro(),
        9,
        horizon=1,
    )
    assert pulled["market"] == [["SELL", "WOOL", 4]]
    assert debts == {201: {"WOOL": 4}}

    due = _FutureSaleMacro().action_for_route(
        {"step": 201}, 9
    )
    suppressed = suppress_due_sales(due, debts[201])
    assert suppressed["market"] == []


def test_front_run_is_blocked_when_book_is_at_or_below_base():
    action = {"farmer": ["PASS"], "hands": [], "market": []}
    pulled, debts = front_run_future_sales(
        _front_run_obs(200),
        action,
        _FutureSaleMacro(),
        9,
        horizon=1,
    )
    assert pulled == action
    assert debts == {}


def test_front_run_respects_existing_reservation_debt():
    action = {"farmer": ["PASS"], "hands": [], "market": []}
    pulled, debts = front_run_future_sales(
        _front_run_obs(220),
        action,
        _FutureSaleMacro(),
        9,
        horizon=1,
        existing_debts={201: {"WOOL": 4}},
    )
    assert pulled == action
    assert debts == {}
