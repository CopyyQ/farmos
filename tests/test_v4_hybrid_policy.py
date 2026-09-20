from kaggrl.macro_policy import MacroPolicy
from kaggrl.residual_actions import MarketEdit, ResidualAction
from kaggrl.v4_hybrid_policy import FarmOSV4HybridPolicy
from kaggrl.v4_options import V4Option


def test_macro_reconstructs_missing_step_from_day_hour():
    routes = {
        0: [
            {"farmer": ["PASS"], "hands": [], "market": [["SELL", "WHEAT", 1]]}
            for _ in range(719)
        ],
        2: [
            {"farmer": ["PASS"], "hands": [], "market": []}
            for _ in range(719)
        ],
        100: [
            {"farmer": ["PASS"], "hands": [], "market": []}
            for _ in range(719)
        ],
    }
    policy = MacroPolicy(routes, {}, {})
    obs = {"day": 1, "hour": 3, "town": {"unlocked_shops": []}}
    assert MacroPolicy._step(obs) == 27
    assert policy.act(obs)["market"] == [["SELL", "WHEAT", 1]]


def test_v4_stage0_is_macro_playable_base():
    policy = FarmOSV4HybridPolicy()
    obs = {
        "player": 1,
        "day": 0,
        "hour": 0,
        "town": {"unlocked_shops": []},
    }
    action = policy.act(obs)
    assert set(action) == {"farmer", "hands", "market"}
    assert isinstance(action["market"], list)


def test_v4_residual_is_confidence_gated():
    baseline = FarmOSV4HybridPolicy()
    obs = {
        "player": 0,
        "step": 1,
        "day": 0,
        "hour": 1,
        "town": {"unlocked_shops": []},
    }
    expected = baseline.act(obs)

    residual = ResidualAction(
        market0=MarketEdit("DROP"),
        market1=MarketEdit("KEEP"),
    )
    low = FarmOSV4HybridPolicy(
        residual=lambda obs, cfg, base: (residual, 0.2),
        min_confidence=0.8,
    )
    assert low.act(obs) == expected


def test_v4_residual_exception_falls_back_to_macro():
    obs = {
        "player": 0,
        "step": 2,
        "day": 0,
        "hour": 2,
        "town": {"unlocked_shops": []},
    }
    base = FarmOSV4HybridPolicy().act(obs)

    def explode(obs, config, action):
        raise RuntimeError("bad residual")

    policy = FarmOSV4HybridPolicy(residual=explode)
    assert policy.act(obs) == base


def test_v4_residual_receives_canonical_step_when_raw_step_is_missing():
    seen = {}

    def residual(obs, config, base):
        seen.update(step=obs["step"], day=obs["day"], hour=obs["hour"])
        return None

    policy = FarmOSV4HybridPolicy(residual=residual)
    policy.act({
        "player": 1,
        "day": 1,
        "hour": 3,
        "town": {"unlocked_shops": []},
    })
    assert seen == {"step": 27, "day": 1, "hour": 3}


def test_v4_option_can_switch_route_on_the_current_step():
    obs = {
        "player": 0,
        "step": 144,
        "day": 6,
        "hour": 0,
        "private": {"shed": {}},
        "town": {"unlocked_shops": ["YARN_STORE", "BAKERY"]},
    }
    baseline = FarmOSV4HybridPolicy()
    base_action = baseline.act(obs)

    policy = FarmOSV4HybridPolicy(
        option_policy=lambda obs, cfg, ctx: (
            V4Option(route_id=119, market_mode="KEEP_ROUTE"),
            0.99,
        )
    )
    action = policy.act(obs)
    expected = policy.base.action_for_route(obs, 119)

    assert action == expected
    assert action != base_action
    assert policy.last_strategy["step"] == 144
    assert policy.last_strategy["base_route_id"] == 9
    assert policy.last_strategy["selected_route_id"] == 119
    assert policy.last_strategy["option_applied"] is True


def test_v4_option_is_re_evaluated_every_step():
    seen = []

    def choose(obs, cfg, ctx):
        seen.append((ctx.step, ctx.remaining_steps, ctx.phase_name))
        route = 119 if ctx.step % 2 == 0 else 9
        return V4Option(route_id=route), 1.0

    policy = FarmOSV4HybridPolicy(option_policy=choose)
    for step in (144, 145, 146):
        day, hour = divmod(step, 24)
        policy.act({
            "player": 0,
            "step": step,
            "day": day,
            "hour": hour,
            "private": {"shed": {}},
            "town": {"unlocked_shops": ["YARN_STORE", "BAKERY"]},
        })
        assert policy.last_strategy["selected_route_id"] == (
            119 if step % 2 == 0 else 9
        )

    assert [row[0] for row in seen] == [144, 145, 146]
    assert [row[1] for row in seen] == [575, 574, 573]


def test_v4_low_confidence_option_keeps_exact_macro_baseline():
    obs = {
        "player": 0,
        "step": 144,
        "day": 6,
        "hour": 0,
        "private": {"shed": {}},
        "town": {"unlocked_shops": ["YARN_STORE", "BAKERY"]},
    }
    expected = FarmOSV4HybridPolicy().act(obs)
    policy = FarmOSV4HybridPolicy(
        option_policy=lambda obs, cfg, ctx: (
            V4Option(route_id=119),
            0.10,
        ),
        min_option_confidence=0.65,
    )
    assert policy.act(obs) == expected
    assert policy.last_strategy["option_applied"] is False
    assert policy.last_strategy["selected_route_id"] == 9


def test_v4_option_context_uses_reconstructed_seat1_step():
    seen = {}

    def choose(obs, cfg, ctx):
        seen.update(
            step=ctx.step,
            day=ctx.day,
            hour=ctx.hour,
            remaining=ctx.remaining_steps,
        )
        return None

    policy = FarmOSV4HybridPolicy(option_policy=choose)
    policy.act({
        "player": 1,
        "day": 28,
        "hour": 8,
        "private": {"shed": {}},
        "town": {"unlocked_shops": []},
    })
    assert seen == {
        "step": 680,
        "day": 28,
        "hour": 8,
        "remaining": 39,
    }


def test_v4_liquidation_mode_never_sells_more_than_known_shed():
    obs = {
        "player": 0,
        "step": 680,
        "day": 28,
        "hour": 8,
        "private": {"shed": {"WHEAT": 3, "MILK": 2}},
        "town": {"unlocked_shops": []},
    }
    policy = FarmOSV4HybridPolicy(
        option_policy=lambda obs, cfg, ctx: (
            V4Option(route_id=2, market_mode="LIQUIDATE_SHED"),
            1.0,
        )
    )
    action = policy.act(obs)
    assert all(order[0] == "SELL" for order in action["market"])
    sold = {}
    for _, item, quantity in action["market"]:
        sold[item] = sold.get(item, 0) + quantity
    assert sold.get("WHEAT", 0) <= 3
    assert sold.get("MILK", 0) <= 2


def test_macro_compatible_routes_follow_step_and_shop_boundaries():
    policy = FarmOSV4HybridPolicy()
    shops = ["YARN_STORE", "BAKERY"]

    before = {
        "step": 143, "day": 5, "hour": 23,
        "town": {"unlocked_shops": shops},
    }
    reveal = {
        "step": 144, "day": 6, "hour": 0,
        "town": {"unlocked_shops": shops},
    }
    liquidation = {
        "step": 648, "day": 27, "hour": 0,
        "town": {"unlocked_shops": shops},
    }

    assert policy.base.compatible_route_ids(before) == (0,)
    policy.base.reset()
    assert policy.base.compatible_route_ids(reveal) == (9, 119)
    policy.base.reset()
    assert policy.base.compatible_route_ids(liquidation) == (9, 119, 2)
