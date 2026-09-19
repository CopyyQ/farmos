from kaggle_environments import make


def _obs():
    env = make("kaggriculture", configuration={"seed": 1, "episodeSteps": 720}, debug=False)
    return env.state[0].observation


def test_observation_encoder_is_fixed_and_finite():
    from kaggrl.observation import ObservationEncoder
    import numpy as np
    enc = ObservationEncoder()
    x = enc.encode(_obs())
    assert x.shape == (enc.size,)
    assert x.dtype == np.float32
    assert np.isfinite(x).all()


def test_action_codec_round_trip_simple_action():
    from kaggrl.actions import ActionCodec
    codec = ActionCodec(max_hands=16, max_market_orders=10)
    action = {"farmer": ["PLANT", "WHEAT"], "hands": [], "market": [["BUY_SEED", "MELON", 2], ["HIRE"]]}
    encoded = codec.encode(action, hand_count=0)
    decoded = codec.decode(encoded, _obs())
    assert decoded["farmer"] == ["PLANT", "WHEAT"]
    assert decoded["hands"] == []
    assert decoded["market"][0] == ["BUY_SEED", "MELON", 2]
    assert decoded["market"][1] == ["HIRE"]