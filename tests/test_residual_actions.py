from kaggrl.residual_actions import KEEP, MarketEdit, ResidualAction, apply_residual, valid_market_order


def _base():
    return {
        "farmer": ["PASS"],
        "hands": [["NORTH"]],
        "market": [["SELL", "MILK", 3], ["SELL", "WOOL", 2], ["BUY_SEED", "WHEAT", 1]],
    }


def test_keep_is_identity_and_preserves_non_market_actions():
    base = _base()
    out = apply_residual(base, ResidualAction(KEEP, KEEP, None))
    assert out == base and out is not base


def test_replace_second_effective_order_preserves_tail():
    base = _base()
    r = ResidualAction(KEEP, MarketEdit("REPLACE", ["SELL", "WOOL", 1]), None)
    out = apply_residual(base, r)
    assert out["market"] == [["SELL", "MILK", 3], ["SELL", "WOOL", 1], ["BUY_SEED", "WHEAT", 1]]
    assert out["farmer"] == base["farmer"] and out["hands"] == base["hands"]


def test_drop_first_effective_order_shifts_tail_forward():
    out = apply_residual(_base(), ResidualAction(MarketEdit("DROP"), KEEP, None))
    assert out["market"] == [["SELL", "WOOL", 2], ["BUY_SEED", "WHEAT", 1]]


def test_market_validation_is_verb_specific():
    assert valid_market_order(["HIRE"])
    assert valid_market_order(["BUY_LAND"])
    assert valid_market_order(["BUY_ANIMAL", "COW", 2])
    assert valid_market_order(["BUY_SEED", "WHEAT", 3])
    assert not valid_market_order(["SELL", "MILK", 0])
    assert not valid_market_order(["BUY_ANIMAL", "WHEAT", 1])


def test_market_pass_placeholder_is_preserved_as_queue_slot():
    assert valid_market_order(["PASS"])
    base = _base()
    out = apply_residual(base, ResidualAction(MarketEdit("REPLACE", ["PASS"]), KEEP, None))
    assert out["market"][0] == ["PASS"]
    assert out["market"][1] == ["SELL", "WOOL", 2]
