from kaggrl.v4_options import (
    StepStrategyContext,
    V4Option,
    compile_market_mode,
    safe_sales_queue,
    validate_option,
)


def _obs(step=144):
    day, hour = divmod(step, 24)
    return {
        "player": 0,
        "step": step,
        "day": day,
        "hour": hour,
        "private": {
            "shed": {"WHEAT": 4, "MILK": 3, "FERTILIZER": 0},
        },
        "town": {"unlocked_shops": []},
    }


def test_strategy_context_is_absolute_step_aware():
    context = StepStrategyContext.from_observation(
        _obs(680),
        {"episodeSteps": 720, "turnsPerDay": 24},
        base_route_id=2,
        route_ids=(0, 2, 9, 100),
    )
    assert context.step == 680
    assert context.day == 28
    assert context.hour == 8
    assert context.remaining_steps == 39
    assert context.phase_name == "liquidation"
    assert context.base_route_id == 2


def test_invalid_route_falls_back_to_current_base_route():
    context = StepStrategyContext.from_observation(
        _obs(),
        None,
        base_route_id=100,
        route_ids=(0, 9, 100),
    )
    option = validate_option(
        V4Option(route_id=999, market_mode="KEEP_ROUTE"),
        context,
    )
    assert option.route_id == 100


def test_no_spend_keeps_only_sell_orders_bounded_by_known_shed():
    obs = _obs()
    action = {
        "farmer": ["PASS"],
        "hands": [],
        "market": [
            ["SELL", "WHEAT", 10],
            ["HIRE"],
            ["BUY_SEED", "WHEAT", 5],
            ["SELL", "MILK", 1],
        ],
    }
    out = compile_market_mode(obs, action, "NO_SPEND")
    assert out["market"] == [
        ["SELL", "WHEAT", 4],
        ["SELL", "MILK", 1],
    ]


def test_liquidate_shed_sells_remaining_known_inventory_without_oversell():
    queue = safe_sales_queue(
        _obs(),
        [["SELL", "WHEAT", 2], ["BUY_PRODUCT", "WHEAT", 99]],
        liquidate_all=True,
    )
    assert queue == [
        ["SELL", "WHEAT", 2],
        ["SELL", "WHEAT", 2],
        ["SELL", "MILK", 3],
    ]
    totals = {}
    for _, item, quantity in queue:
        totals[item] = totals.get(item, 0) + quantity
    assert totals == {"WHEAT": 4, "MILK": 3}


def test_keep_route_is_bit_exact_copy():
    action = {
        "farmer": ["NORTH"],
        "hands": [["PASS"]],
        "market": [["HIRE"], ["SELL", "WHEAT", 2]],
    }
    out = compile_market_mode(_obs(), action, "KEEP_ROUTE")
    assert out == action
    assert out is not action
