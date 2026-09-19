from __future__ import annotations
import numpy as np

VOCAB_SIZE = 128

def _sigmoid(x):
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))

def make_random_weights(input_dim: int, hidden_dim: int, token_width: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    scale = 0.03
    return {
        "input_dim": np.array(input_dim, dtype=np.int32),
        "hidden_dim": np.array(hidden_dim, dtype=np.int32),
        "token_width": np.array(token_width, dtype=np.int32),
        "W_x": (rng.standard_normal((input_dim, hidden_dim)) * scale).astype(np.float32),
        "b_x": np.zeros(hidden_dim, np.float32),
        "W_lstm": (rng.standard_normal((hidden_dim * 2, hidden_dim * 4)) * scale).astype(np.float32),
        "b_lstm": np.zeros(hidden_dim * 4, np.float32),
        "W_out": (rng.standard_normal((hidden_dim, token_width * VOCAB_SIZE)) * scale).astype(np.float32),
        "b_out": np.zeros(token_width * VOCAB_SIZE, np.float32),
    }
class NumpyLSTMPolicy:
    def __init__(self, weights):
        self.w = weights
        self.input_dim = int(weights["input_dim"])
        self.hidden_dim = int(weights["hidden_dim"])
        self.token_width = int(weights["token_width"])

    def initial_state(self):
        z = np.zeros(self.hidden_dim, dtype=np.float32)
        return z.copy(), z.copy()

    def act(self, observation, state):
        x = np.asarray(observation, dtype=np.float32).reshape(self.input_dim)
        h, c = state
        e = np.tanh(x @ self.w["W_x"] + self.w["b_x"])
        gates = np.concatenate([e, h]) @ self.w["W_lstm"] + self.w["b_lstm"]
        i, f, g, o = np.split(gates, 4)
        i, f, o = _sigmoid(i), _sigmoid(f), _sigmoid(o)
        g = np.tanh(g)
        c2 = f * c + i * g
        h2 = o * np.tanh(c2)
        logits = (h2 @ self.w["W_out"] + self.w["b_out"]).reshape(self.token_width, VOCAB_SIZE)
        tokens = np.argmax(logits, axis=1).astype(np.int16)
        return tokens, (h2.astype(np.float32), c2.astype(np.float32))

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=False)
        return cls({k: data[k] for k in data.files})
