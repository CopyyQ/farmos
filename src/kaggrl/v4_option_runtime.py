from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .clock import CLOCK_FEATURES, resolve_clock
from .observation import ObservationEncoder
from .v4_option_model import V4OptionPolicy
from .v4_options import MARKET_MODES, StepStrategyContext, V4Option


class TorchV4OptionRuntime:
    """Closed-loop strategic runtime used for validation before NumPy export."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str = "cpu",
        min_route_probability: float = 0.55,
        min_market_probability: float = 0.65,
        allowed_market_modes=("KEEP_ROUTE",),
        allow_route_switch: bool = True,
        route_switch_steps=(144,),
        liquidation_max_remaining_steps: int | None = None,
    ):
        import torch

        self.torch = torch
        self.device = torch.device(device)
        payload = torch.load(
            str(checkpoint),
            map_location=self.device,
            weights_only=False,
        )
        if payload.get("architecture_version") != (
            V4OptionPolicy.ARCHITECTURE_VERSION
        ):
            raise RuntimeError("unsupported V4 option checkpoint")
        if payload.get("observation_schema") != (
            "macro_semantic_v4_clock_v2"
        ):
            raise RuntimeError("V4 option checkpoint observation mismatch")
        if payload.get("clock_features") != list(CLOCK_FEATURES):
            raise RuntimeError("V4 option checkpoint clock schema mismatch")

        self.route_ids = tuple(int(x) for x in payload["route_ids"])
        self.route_to_class = {
            route_id: index
            for index, route_id in enumerate(self.route_ids)
        }
        self.market_modes = tuple(payload["market_modes"])
        if self.market_modes != tuple(MARKET_MODES):
            raise RuntimeError("V4 option market-mode schema mismatch")

        self.model = V4OptionPolicy(
            int(payload["input_dim"]),
            route_count=len(self.route_ids),
            market_mode_count=len(self.market_modes),
            hidden_dim=int(payload["hidden_dim"]),
            clock_dim=int(payload["clock_dim"]),
        ).to(self.device)
        self.model.load_state_dict(payload["model_state"], strict=True)
        self.model.eval()

        self.encoder = ObservationEncoder(clock_schema="v4")
        self.route_gate_threshold = float(
            payload.get("route_gate_threshold", 0.5)
        )
        self.min_route_probability = float(min_route_probability)
        self.min_market_probability = float(min_market_probability)
        self.allowed_market_modes = frozenset(
            MARKET_MODES if allowed_market_modes is None
            else [str(value) for value in allowed_market_modes]
        )
        unknown_modes = self.allowed_market_modes - set(MARKET_MODES)
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
        self.state = None
        self.last_step: int | None = None
        self.last_decision: dict[str, Any] | None = None
        self.telemetry: list[dict[str, Any]] = []

    def reset(self):
        self.state = None
        self.last_step = None
        self.last_decision = None
        self.telemetry = []

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        values = values - np.max(values)
        exp = np.exp(values)
        return exp / max(np.finfo(np.float64).tiny, float(exp.sum()))

    def __call__(
        self,
        observation: dict[str, Any],
        configuration: Any,
        context: StepStrategyContext,
    ):
        torch = self.torch
        clock = resolve_clock(observation, configuration)
        if clock.step == 0 and self.last_step not in {None, 0}:
            self.reset()
        if self.last_step is not None and clock.step not in {
            self.last_step,
            self.last_step + 1,
        }:
            # Never carry recurrent state across an unknown temporal jump.
            self.state = None
        self.last_step = clock.step

        encoded = self.encoder.encode(
            observation, configuration
        ).astype(np.float32, copy=False)
        clock_context = np.asarray(
            clock.features(), dtype=np.float32
        )
        if clock_context.size != len(CLOCK_FEATURES):
            raise RuntimeError("runtime clock width mismatch")

        obs_tensor = torch.from_numpy(encoded).view(1, 1, -1).to(
            self.device
        )
        clock_tensor = torch.from_numpy(clock_context).view(1, 1, -1).to(
            self.device
        )
        with torch.inference_mode():
            output, self.state = self.model.forward_sequence(
                obs_tensor,
                clock_tensor,
                self.state,
            )
            self.state = tuple(value.detach() for value in self.state)

        raw_route = output["route"][0, 0].detach().float().cpu().numpy()
        allowed_classes = [
            self.route_to_class[route_id]
            for route_id in context.route_ids
            if route_id in self.route_to_class
        ]
        route_id = None
        route_probability = 0.0
        if allowed_classes:
            allowed_logits = raw_route[allowed_classes]
            probabilities = self._softmax(allowed_logits)
            local_index = int(np.argmax(probabilities))
            route_probability = float(probabilities[local_index])
            selected_class = allowed_classes[local_index]
            candidate_route = self.route_ids[selected_class]
        else:
            candidate_route = context.base_route_id

        route_gate = float(
            output["route_gate"][0, 0].detach().float().cpu().item()
        )
        if (
            self.allow_route_switch
            and clock.step in self.route_switch_steps
            and int(candidate_route) != int(context.base_route_id)
            and route_gate >= self.route_gate_threshold
            and route_probability >= self.min_route_probability
        ):
            route_id = int(candidate_route)

        raw_market = (
            output["market"][0, 0].detach().float().cpu().numpy()
        )
        market_probabilities = self._softmax(raw_market)
        market_index = int(np.argmax(market_probabilities))
        market_probability = float(market_probabilities[market_index])
        market_mode = "KEEP_ROUTE"
        predicted_market_mode = self.market_modes[market_index]
        if (
            market_probability >= self.min_market_probability
            and predicted_market_mode in self.allowed_market_modes
        ):
            market_mode = predicted_market_mode
        if (
            market_mode == "LIQUIDATE_SHED"
            and self.liquidation_max_remaining_steps is not None
            and clock.remaining_steps
            > self.liquidation_max_remaining_steps
        ):
            market_mode = "KEEP_ROUTE"

        changed = (
            route_id is not None
            or market_mode != "KEEP_ROUTE"
        )
        self.last_decision = {
            "step": clock.step,
            "remaining_steps": clock.remaining_steps,
            "phase": context.phase_name,
            "base_route_id": int(context.base_route_id),
            "compatible_routes": list(context.route_ids),
            "route_id": route_id,
            "route_gate": route_gate,
            "route_probability": route_probability,
            "market_mode": market_mode,
            "market_probability": market_probability,
            "changed": bool(changed),
        }
        self.telemetry.append(dict(self.last_decision))
        if not changed:
            return None
        # Thresholding is internal; V4HybridPolicy should not gate this again.
        return V4Option(
            route_id=route_id,
            market_mode=market_mode,
        ), 1.0
