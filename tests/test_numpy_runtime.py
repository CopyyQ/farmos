import numpy as np


def test_numpy_lstm_policy_shapes_and_reset():
    from kaggrl.numpy_runtime import NumpyLSTMPolicy, make_random_weights
    weights = make_random_weights(input_dim=64, hidden_dim=16, token_width=12, seed=7)
    policy = NumpyLSTMPolicy(weights)
    state = policy.initial_state()
    x = np.linspace(-1, 1, 64, dtype=np.float32)
    tokens1, state1 = policy.act(x, state)
    tokens2, _ = policy.act(x, policy.initial_state())
    assert tokens1.shape == (12,)
    assert tokens1.dtype == np.int16
    assert np.array_equal(tokens1, tokens2)
    assert state1[0].shape == (16,)
    assert np.isfinite(state1[0]).all() and np.isfinite(state1[1]).all()
