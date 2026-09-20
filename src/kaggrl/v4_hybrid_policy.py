from __future__ import annotations

from typing import Any, Callable

from .clock import resolve_clock
from .macro_policy import MacroPolicy
from .residual_actions import ResidualAction, apply_residual
from .v45_macro_data import load_v45_macro_data


ResidualFn = Callable[[dict[str, Any], Any, dict[str, Any]], Any]


class FarmOSV4HybridPolicy:
    """Macro-first FarmOS policy with an optional learned residual."""

    def __init__(
        self,
        residual: ResidualFn | None = None,
        *,
        min_confidence: float = 0.80,
    ):
        routes, new_routes, old_routes = load_v45_macro_data()
        self.base = MacroPolicy(routes, new_routes, old_routes)
        self.residual = residual
        self.min_confidence = float(min_confidence)
        self._last_step: int | None = None

    def reset(self) -> None:
        self.base.reset()
        self._last_step = None

    @staticmethod
    def _unpack_residual(value: Any) -> tuple[ResidualAction | None, float]:
        if value is None:
            return None, 0.0
        if isinstance(value, ResidualAction):
            return value, 1.0
        if isinstance(value, tuple) and len(value) == 2:
            residual, confidence = value
            if residual is None or isinstance(residual, ResidualAction):
                return residual, float(confidence)
        raise TypeError(
            "residual callback must return ResidualAction, "
            "(ResidualAction, confidence), or None"
        )

    def act(self, observation, configuration=None) -> dict[str, Any]:
        obs = dict(observation)
        clock = resolve_clock(obs, configuration)
        step = clock.step
        # Seat 1 does not expose observation["step"] in the current engine.
        # Canonicalize the clock at the V4 boundary so every residual sees it.
        obs["step"] = step
        obs["day"] = clock.day
        obs["hour"] = clock.hour
        if step == 0 and self._last_step not in {None, 0}:
            self.base.reset()
        self._last_step = step

        base_action = self.base.act(obs, configuration)
        if self.residual is None:
            return base_action

        try:
            residual, confidence = self._unpack_residual(
                self.residual(obs, configuration, base_action)
            )
        except Exception:
            return base_action
        if residual is None or confidence < self.min_confidence:
            return base_action
        return apply_residual(base_action, residual)

    __call__ = act
