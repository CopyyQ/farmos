from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from kaggrl.clock import resolve_clock
from kaggrl.v2_effect_tracker import EffectRecord, EffectTracker
from kaggrl.v2_numpy_runtime import V2NumpyPolicy
from kaggrl.v2_observation import normalize_observation


def _plain(value):
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_plain(item) for item in value)
    return value


class V2NumpyRolloutAgent:
    def __init__(self, model_path, seed: int = 20260917, deterministic: bool = False):
        self.model_path = Path(model_path)
        self.policy = V2NumpyPolicy.load(self.model_path)
        self.seed = int(seed)
        self.deterministic = bool(deterministic)
        self.effect_tracker = EffectTracker()
        self.episode_index = -1
        self.reset_count = 0
        self.recurrent_state = None
        self.previous_step: int | None = None
        self.previous_obs: dict[str, Any] | None = None
        self.pending_requested_action: dict[str, Any] | None = None
        self.last_effect: EffectRecord | None = None
        self.last_metadata: dict[str, Any] = {}
        self.telemetry: list[dict[str, Any]] = []
        self._rng = np.random.default_rng(self.seed)

    def _reset_episode(self) -> None:
        self.episode_index += 1
        self.reset_count += 1
        self.recurrent_state = None
        self.previous_step = None
        self.previous_obs = None
        self.pending_requested_action = None
        self.last_effect = None
        self._rng = np.random.default_rng(self.seed + self.episode_index)

    def _needs_reset(self, step: int) -> bool:
        if self.previous_step is None:
            return True
        if step == 0 and self.previous_step != 0:
            return True
        return step < self.previous_step

    def _previous_effect(self, observation: dict[str, Any]):
        if self.previous_obs is None or self.pending_requested_action is None:
            self.last_effect = None
            return {}
        record = self.effect_tracker.observe(
            self.previous_obs, self.pending_requested_action, observation,
        )
        self.last_effect = record
        return record.to_model_effect()

    def act(self, observation: dict[str, Any], configuration=None):
        observation = _plain(observation)
        clock = resolve_clock(observation, configuration)
        step = clock.step
        observation["step"] = step
        observation["day"] = clock.day
        observation["hour"] = clock.hour
        reset = self._needs_reset(step)
        if reset:
            self._reset_episode()
            previous_effect = {}
        else:
            previous_effect = self._previous_effect(observation)

        structured = asdict(normalize_observation(observation))
        output = self.policy.step(
            structured, previous_effect, None, self.recurrent_state,
            self._rng, deterministic=self.deterministic,
        )
        self.recurrent_state = output.recurrent_state
        self.previous_obs = deepcopy(observation)
        self.previous_step = step
        self.pending_requested_action = deepcopy(output.engine_action)
        self.last_metadata = {
            "step": step, "day": clock.day, "hour": clock.hour,
            "remaining_steps": clock.remaining_steps,
            "episode_index": self.episode_index, "reset": reset,
            "previous_effect": deepcopy(previous_effect),
            "canonical_action": deepcopy(output.canonical_action),
            "engine_action": deepcopy(output.engine_action),
            "logp": float(output.logp),
            "terminal_money": float(output.terminal_money),
            "terminal_margin": float(output.terminal_margin),
            "quantity_token_count": int(output.quantity_token_count),
            "effective_families": ([] if self.last_effect is None
                                   else list(self.last_effect.effective_families)),
        }
        self.telemetry.append(deepcopy(self.last_metadata))
        return output.engine_action

    def __call__(self, observation: dict[str, Any], configuration=None):
        return self.act(observation, configuration)
