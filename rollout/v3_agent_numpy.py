from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from kaggrl.v2_effect_tracker import EffectRecord, EffectTracker
from kaggrl.v2_observation import normalize_observation
from kaggrl.v3_numpy_runtime import V3NumpyPolicy


def _plain(value):
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_plain(item) for item in value)
    return value


class V3NumpyRolloutAgent:
    POLICY_CLASS = V3NumpyPolicy

    def __init__(self, model_path, seed: int = 20260917, deterministic: bool = False,
                 capture_decision_trace: bool = False,
                 strategy_slot: int | None = None):
        self.model_path = Path(model_path)
        self.policy = self.POLICY_CLASS.load(self.model_path)
        if self.policy.strategy_count:
            resolved_slot = (
                self.policy.default_strategy_slot
                if strategy_slot is None else int(strategy_slot)
            )
            if not 0 <= resolved_slot < self.policy.strategy_count:
                raise ValueError("strategy slot is out of range")
            self.strategy_slot: int | None = resolved_slot
        else:
            if strategy_slot is not None:
                raise ValueError("base V3 policy does not accept a strategy slot")
            self.strategy_slot = None
        self.seed = int(seed)
        self.deterministic = bool(deterministic)
        self.effect_tracker = EffectTracker()
        self.episode_index = -1
        self.reset_count = 0
        self.recurrent_state = None
        self.previous_step: int | None = None
        self.previous_player: int | None = None
        self.previous_obs: dict[str, Any] | None = None
        self.pending_requested_action: dict[str, Any] | None = None
        self.last_effect: EffectRecord | None = None
        self.last_metadata: dict[str, Any] = {}
        self.telemetry: list[dict[str, Any]] = []
        self.capture_decision_trace = bool(capture_decision_trace)
        self.diagnostic_fixtures: list[dict[str, Any]] = []
        self._rng = np.random.default_rng(self.seed)

    def _reset_episode(self) -> None:
        self.episode_index += 1
        self.reset_count += 1
        self.recurrent_state = None
        self.previous_step = None
        self.previous_player = None
        self.previous_obs = None
        self.pending_requested_action = None
        self.last_effect = None
        self._rng = np.random.default_rng(self.seed + self.episode_index)

    def _needs_reset(self, step: int, player: int) -> bool:
        if self.previous_step is None:
            return True
        if self.previous_player is not None and player != self.previous_player:
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
        del configuration
        observation = _plain(observation)
        step = int(observation.get("step", 0))
        player = int(observation.get("player", 0))
        reset = self._needs_reset(step, player)
        if reset:
            self._reset_episode()
            previous_effect = {}
        else:
            previous_effect = self._previous_effect(observation)

        structured = asdict(normalize_observation(observation))
        state_before = self.recurrent_state
        previous_action_before = (
            {} if state_before is None else deepcopy(state_before.previous_action)
        )
        decision_trace = None
        if self.capture_decision_trace:
            traced = self.policy.trace_step(
                structured, previous_effect, None, state_before,
                self._rng, deterministic=self.deterministic,
                strategy_slot=self.strategy_slot,
            )
            output = traced["output"]
            decision_trace = deepcopy(traced["decisions"])
        else:
            output = self.policy.step(
                structured, previous_effect, None, state_before,
                self._rng, deterministic=self.deterministic,
                strategy_slot=self.strategy_slot,
            )
        self.recurrent_state = output.recurrent_state
        self.previous_obs = deepcopy(observation)
        self.previous_step = step
        self.previous_player = player
        self.pending_requested_action = deepcopy(output.engine_action)
        self.last_metadata = {
            "step": step,
            "player": player,
            "episode_index": self.episode_index,
            "reset": reset,
            "strategy_slot": self.strategy_slot,
            "previous_effect": deepcopy(previous_effect),
            "canonical_action": deepcopy(output.canonical_action),
            "engine_action": deepcopy(output.engine_action),
            "logp": float(output.logp),
            "terminal_money": float(output.terminal_money),
            "terminal_margin": float(output.terminal_margin),
            "quantity_token_count": int(output.quantity_token_count),
            "temporal_valid_length": int(output.recurrent_state.valid_length),
            "temporal_write_pos": int(output.recurrent_state.write_pos),
            "effective_families": ([] if self.last_effect is None
                                   else list(self.last_effect.effective_families)),
        }
        if decision_trace is not None:
            self.last_metadata["decision_trace"] = deepcopy(decision_trace)
            state_plain = None
            if state_before is not None:
                state_plain = {
                    "h": state_before.h.tolist(), "c": state_before.c.tolist(),
                    "memory": state_before.memory.tolist(),
                    "valid_length": int(state_before.valid_length),
                    "write_pos": int(state_before.write_pos),
                    "previous_action": deepcopy(state_before.previous_action),
                }
            self.diagnostic_fixtures.append({
                "step": step, "player": player, "reset": reset,
                "strategy_slot": self.strategy_slot,
                "observation": deepcopy(observation),
                "structured_state": deepcopy(structured),
                "previous_effect": deepcopy(previous_effect),
                "previous_action": previous_action_before,
                "recurrent_state_before": state_plain,
                "decisions": deepcopy(decision_trace),
                "canonical_action": deepcopy(output.canonical_action),
                "engine_action": deepcopy(output.engine_action),
            })
        self.telemetry.append(deepcopy(self.last_metadata))
        return output.engine_action

    def __call__(self, observation: dict[str, Any], configuration=None):
        return self.act(observation, configuration)
