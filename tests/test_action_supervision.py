import numpy as np


def test_supervision_mask_ignores_unused_arguments_and_trailing_market_slots():
    from kaggrl.actions import ActionCodec
    codec = ActionCodec(max_hands=2, max_market_orders=4)
    action = {
        "farmer": ["PLANT", "WHEAT"],
        "hands": [["PICKUP", "WHEAT", 2]],
        "market": [["BUY_SEED", "MELON", 2], ["HIRE"]],
    }
    mask = codec.supervision_mask(action, hand_count=1)
    expected = np.zeros(codec.width, dtype=np.float32)
    expected[0:2] = 1
    expected[3:6] = 1
    market_start = codec.unit_slots * 3
    expected[market_start:market_start + 3] = 1
    expected[market_start + 3] = 1
    expected[market_start + 6] = 1
    np.testing.assert_array_equal(mask, expected)


def test_decode_stops_market_at_first_none_token():
    from kaggrl.actions import ActionCodec
    codec = ActionCodec(max_hands=0, max_market_orders=2)
    tokens = np.zeros(codec.width, dtype=np.int64)
    market_start = codec.unit_slots * 3
    tokens[market_start + 3:market_start + 6] = [4, 1, 5]
    obs = {"player": 0, "farms": [{"hands": []}]}
    assert codec.decode(tokens, obs)["market"] == []
