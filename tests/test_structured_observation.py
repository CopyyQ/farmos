from copy import deepcopy
import numpy as np
from kaggle_environments import make


def _obs():
    env = make("kaggriculture", configuration={"seed": 7, "episodeSteps": 720}, debug=False)
    return env.state[0].observation


def test_encoder_reacts_to_time_private_market_and_tiles():
    from kaggrl.observation import ObservationEncoder
    enc = ObservationEncoder()
    base = _obs()
    assert enc.size == 1024
    x0 = enc.encode(base)
    assert x0.shape == (enc.size,)
    assert np.isfinite(x0).all()
    variants = []
    timed = deepcopy(base); timed["step"] = 719; timed["day"] = 29; timed["hour"] = 23; variants.append(timed)
    private = deepcopy(base); private["private"]["shed"]["WHEAT"] = 25; variants.append(private)
    market = deepcopy(base); market["market"]["prices"]["MILK"] = 999; variants.append(market)
    tile = deepcopy(base); tile["farms"][0]["tiles"][0][0] = {"kind":"WEED"}; variants.append(tile)
    for obs in variants:
        assert not np.array_equal(x0, enc.encode(obs))


def test_encoder_ignores_fake_hidden_seed_field():
    from kaggrl.observation import ObservationEncoder
    enc = ObservationEncoder()
    obs = _obs()
    x0 = enc.encode(obs)
    other = deepcopy(obs); other["seed"] = 999999
    np.testing.assert_array_equal(x0, enc.encode(other))


def test_encoder_is_canonical_across_player_seats():
    from kaggrl.observation import ObservationEncoder
    enc = ObservationEncoder()
    obs0 = _obs()
    obs1 = deepcopy(obs0)
    obs1["farms"] = [deepcopy(obs0["farms"][1]), deepcopy(obs0["farms"][0])]
    obs1["player"] = 1
    # private belongs to the acting player in both observations.
    np.testing.assert_allclose(enc.encode(obs0), enc.encode(obs1), atol=0.0, rtol=0.0)
