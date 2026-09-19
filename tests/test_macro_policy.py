from kaggrl.macro_policy import MacroPolicy


def _obs(step, shops):
    return {"step": step, "town": {"unlocked_shops": shops}}


def _routes():
    base = {"farmer": ["PASS"], "hands": [], "market": []}
    routes = {}
    for rid in (0, 1, 2, 100):
        routes[rid] = [dict(base, market=[["SELL", "WHEAT", rid + 1]]) for _ in range(720)]
    return routes


def test_macro_route_switches_on_day6_and_day27():
    p = MacroPolicy(_routes(), {("BRUNCH_SPOT", "ICE_CREAM_SHOP"): 1}, {})
    assert p.route_id(_obs(143, [])) == 0
    p.act(_obs(144, ["BRUNCH_SPOT", "ICE_CREAM_SHOP"]))
    assert p.route_id(_obs(144, ["BRUNCH_SPOT", "ICE_CREAM_SHOP"])) == 1
    p.act(_obs(648, ["BRUNCH_SPOT", "ICE_CREAM_SHOP"]))
    assert p.route_id(_obs(648, ["BRUNCH_SPOT", "ICE_CREAM_SHOP"])) == 2


def test_macro_returns_deep_copy():
    p = MacroPolicy(_routes(), {}, {})
    a = p.act(_obs(0, []))
    a["market"][0][2] = 999
    b = p.act(_obs(1, []))
    assert b["market"][0][2] == 1


def test_v45_macro_data_has_all_route_lengths():
    from kaggrl.v45_macro_data import load_v45_macro_data
    routes, new_map, old_map = load_v45_macro_data()
    assert {0, 2, 100}.issubset(routes)
    assert all(len(route) == 719 for route in routes.values())
    assert len(new_map) == 64 and len(old_map) == 64


def test_macro_applies_v45_opening_arm():
    routes = _routes()
    routes[0][0]["market"] = [["BUY_PRODUCT", "WHEAT", 5], ["BUY_PRODUCT", "WHEAT", 10], ["SELL", "WHEAT", 60]]
    p = MacroPolicy(routes, {}, {})
    assert p.act(_obs(0, []))["market"] == [["BUY_PRODUCT", "WHEAT", 70], ["SELL", "WHEAT", 70]]


def test_macro_applies_sales_first_from_step_144():
    routes = _routes()
    routes[0][144]["market"] = [["BUY_PRODUCT", "WHEAT", 3], ["HIRE"], ["SELL", "FERTILIZER", 2], ["SELL", "WHEAT", 1], ["BUY_PRODUCT", "FERTILIZER", 1]]
    p = MacroPolicy(routes, {}, {})
    out = p.act(_obs(144, ["YARN_STORE"]))["market"]
    assert out == [["SELL", "FERTILIZER", 2], ["BUY_PRODUCT", "WHEAT", 3], ["SELL", "WHEAT", 1], ["HIRE"], ["BUY_PRODUCT", "FERTILIZER", 1]]
