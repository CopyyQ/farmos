from kaggrl.macro_policy import MacroPolicy
from kaggrl.residual_actions import MarketEdit, ResidualAction
from kaggrl.v4_hybrid_policy import FarmOSV4HybridPolicy


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
