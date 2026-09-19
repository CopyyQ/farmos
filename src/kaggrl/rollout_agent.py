from __future__ import annotations

import numpy as np

from .actions import ActionCodec
from .observation import ObservationEncoder
from .structured_numpy_runtime import StructuredNumpyPolicy


class NumpyRolloutAgent:
    def __init__(self, model_path, seed=0, deterministic=False, memory_horizon=48):
        self.model_path = str(model_path)
        self.policy = StructuredNumpyPolicy.load(model_path)
        self.encoder = ObservationEncoder(size=self.policy.input_dim, max_hands=self.policy.max_hands)
        self.codec = ActionCodec(self.policy.max_hands, self.policy.max_market_orders)
        self.base_seed = int(seed)
        self.deterministic = bool(deterministic)
        self.memory_horizon = int(memory_horizon) if memory_horizon else 0
        self.state = self.policy.initial_state()
        self.last_step = -1
        self.records = []
        self.rng = np.random.default_rng(self.base_seed)
        self.state_reset_count = 0

    def reset(self, player=0):
        self.state = self.policy.initial_state()
        self.last_step = -1
        self.records = []
        self.rng = np.random.default_rng(self.base_seed + 100003 * int(player))
        self.state_reset_count = 1

    def reset_memory(self):
        self.state = self.policy.initial_state()
        self.state_reset_count += 1
    def __call__(self, obs, configuration=None):
        player = int(obs["player"])
        step = int(obs["step"])
        if step == 0 or step <= self.last_step:
            self.reset(player)
        elif self.memory_horizon and step % self.memory_horizon == 0:
            self.reset_memory()
        encoded = self.encoder.encode(obs)
        hand_count = len(obs["farms"][player]["hands"])
        tokens, mask, logp, value, state2 = self.policy.sample_step(
            encoded, self.state, hand_count, self.rng, self.deterministic
        )
        action = self.codec.decode(tokens, obs)
        self.state = state2
        self.last_step = step
        self.records.append({
            "step": step,
            "obs": encoded,
            "action": tokens,
            "mask": mask,
            "logp": logp,
            "value": value,
        })
        return action