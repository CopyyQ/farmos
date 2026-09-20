from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from kaggrl.clock import resolve_clock
from kaggrl.observation import ObservationEncoder
from kaggrl.v4_option_numpy_runtime import V4OptionNumpyPolicy
from kaggrl.v4_options import StepStrategyContext, V4Option


class V4NumpyOptionAdapter:
    """Torch-free recurrent V4 strategic option callback."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        max_clock_error_turns: float = 4.0,
        require_phase_match: bool = True,
        route_gate_threshold: float | None = None,
        min_route_probability: float = 0.55,
        min_market_probability: float = 0.65,
        allowed_market_modes=("KEEP_ROUTE",),
        allow_route_switch: bool = True,
        route_switch_steps=(144,),
        liquidation_max_remaining_steps: int | None = None,
        telemetry_limit: int = 2048,
    ):
        self.policy = V4OptionNumpyPolicy.load(model_path)
        self.encoder = ObservationEncoder(
            size=self.policy.input_dim,
            clock_schema="v4",
        )
        self.max_clock_error_turns = float(max_clock_error_turns)
        self.require_phase_match = bool(require_phase_match)
        self.route_gate_threshold = float(
            self.policy.route_gate_threshold
            if route_gate_threshold is None
            else route_gate_threshold
        )
        self.min_route_probability = float(min_route_probability)
        self.min_market_probability = float(min_market_probability)
        self.route_to_class = {
            route_id: index
            for index, route_id in enumerate(self.policy.route_ids)
        }
        self.allowed_market_modes = frozenset(
            self.policy.market_modes
            if allowed_market_modes is None
            else [str(value) for value in allowed_market_modes]
        )
        unknown_modes = self.allowed_market_modes - set(
            self.policy.market_modes
        )
        if unknown_modes:
            raise ValueError(
                f"unknown allowed V4 market modes: {sorted(unknown_modes)}"
            )
        self.allow_route_switch = bool(allow_route_switch)
        self.route_switch_steps = frozenset(
            int(value) for value in route_switch_steps
        )
        self.liquidation_max_remaining_steps = (
            None
            if liquidation_max_remaining_steps is None
            else max(0, int(liquidation_max_remaining_steps))
        )
        self.telemetry_limit = int(telemetry_limit)
        self.reset()

    def reset(self):
        self.state = None
        self.previous_step: int | None = None
        self.previous_player: int | None = None
        self.last_metadata: dict[str, Any] | None = None
        self.telemetry: list[dict[str, Any]] = []

    def _needs_reset(self, step: int, player: int) -> bool:
        if self.previous_step is None:
            return True
        if self.previous_player is not None and player != self.previous_player:
            return True
        return step == 0 or step <= self.previous_step

    def _compatible_route_choice(self, logits, context):
        allowed = [
            (route_id, self.route_to_class[route_id])
            for route_id in context.route_ids
            if route_id in self.route_to_class
        ]
        if not allowed:
            return int(context.base_route_id), 1.0
        values = np.asarray(
            [logits[index] for _, index in allowed],
            dtype=np.float64,
        )
        values -= float(np.max(values))
        probabilities = np.exp(values)
        probabilities /= max(
            np.finfo(np.float64).tiny,
            float(probabilities.sum()),
        )
        local = int(np.argmax(probabilities))
        return int(allowed[local][0]), float(probabilities[local])

    def __call__(
        self,
        observation: dict[str, Any],
        configuration,
        context: StepStrategyContext,
    ):
        step = int(context.step)
        player = int(observation.get("player", 0))
        reset = self._needs_reset(step, player)
        if reset:
            self.state = None

        features = self.encoder.encode(observation, configuration)
        clock_context = resolve_clock(
            observation, configuration
        ).features()
        output = self.policy.step(
            features, clock_context, self.state
        )
        self.state = output.state
        self.previous_step = step
        self.previous_player = player

        last_step = max(1, int(context.step + context.remaining_steps))
        actual_step_norm = step / last_step
        actual_remaining_norm = context.remaining_steps / last_step
        step_error = abs(
            output.predicted_step_norm - actual_step_norm
        ) * last_step
        remaining_error = abs(
            output.predicted_remaining_norm - actual_remaining_norm
        ) * last_step
        clock_error = max(step_error, remaining_error)
        phase_match = output.phase_id == int(context.phase_index)
        clock_ok = clock_error <= self.max_clock_error_turns
        phase_ok = phase_match or not self.require_phase_match
        accepted = bool(clock_ok and phase_ok)

        selected_route, route_confidence = (
            self._compatible_route_choice(
                output.route_logits, context
            )
        )
        route_switch_allowed = bool(
            self.allow_route_switch
            and step in self.route_switch_steps
            and selected_route != int(context.base_route_id)
            and output.route_gate_probability
            >= self.route_gate_threshold
            and route_confidence >= self.min_route_probability
        )
        market_mode = (
            str(output.market_mode)
            if (
                output.market_confidence >= self.min_market_probability
                and str(output.market_mode) in self.allowed_market_modes
            )
            else "KEEP_ROUTE"
        )
        if (
            market_mode == "LIQUIDATE_SHED"
            and self.liquidation_max_remaining_steps is not None
            and context.remaining_steps
            > self.liquidation_max_remaining_steps
        ):
            market_mode = "KEEP_ROUTE"
        changed = bool(
            route_switch_allowed or market_mode != "KEEP_ROUTE"
        )

        metadata = {
            "step": step,
            "remaining_steps": int(context.remaining_steps),
            "phase": str(context.phase_name),
            "player": player,
            "reset": reset,
            "base_route_id": int(context.base_route_id),
            "compatible_routes": list(context.route_ids),
            "candidate_route_id": int(selected_route),
            "route_id": (
                int(selected_route)
                if route_switch_allowed
                else None
            ),
            "market_mode": market_mode,
            "confidence": 1.0 if changed else 0.0,
            "route_confidence": float(route_confidence),
            "route_gate": float(output.route_gate_probability),
            "route_gate_probability": float(
                output.route_gate_probability
            ),
            "route_switch_allowed": route_switch_allowed,
            "market_confidence": float(output.market_confidence),
            "predicted_phase_id": int(output.phase_id),
            "actual_phase_id": int(context.phase_index),
            "phase_match": bool(phase_match),
            "predicted_step_norm": float(output.predicted_step_norm),
            "predicted_remaining_norm": float(
                output.predicted_remaining_norm
            ),
            "clock_error_turns": float(clock_error),
            "clock_ok": bool(clock_ok),
            "accepted": accepted,
            "changed": changed,
        }
        self.last_metadata = metadata
        self.telemetry.append(deepcopy(metadata))
        if len(self.telemetry) > self.telemetry_limit:
            del self.telemetry[:-self.telemetry_limit]

        if not accepted or not changed:
            return None, 0.0
        return (
            V4Option(
                route_id=(
                    int(selected_route)
                    if route_switch_allowed
                    else None
                ),
                market_mode=market_mode,
            ),
            1.0,
        )
