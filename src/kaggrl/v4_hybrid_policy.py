from __future__ import annotations

from typing import Any, Callable

from .clock import resolve_clock
from .macro_policy import MacroPolicy
from .residual_actions import ResidualAction, apply_residual
from .v4_market_race import (
    MarketRaceTracker,
    front_run_future_sales,
    optimize_sell_order,
    suppress_due_sales,
)
from .v4_options import (
    StepStrategyContext,
    V4Option,
    compile_market_mode,
    validate_option,
)
from .v45_macro_data import load_v45_macro_data


ResidualFn = Callable[[dict[str, Any], Any, dict[str, Any]], Any]
OptionFn = Callable[[dict[str, Any], Any, StepStrategyContext], Any]


class FarmOSV4HybridPolicy:
    """Macro-first FarmOS policy with an optional learned residual."""

    def __init__(
        self,
        residual: ResidualFn | None = None,
        *,
        option_policy: OptionFn | None = None,
        min_confidence: float = 0.80,
        min_option_confidence: float = 0.65,
        enable_market_race_ordering: bool = True,
        market_race_min_gain: float = 1.0,
    ):
        routes, new_routes, old_routes = load_v45_macro_data()
        self.base = MacroPolicy(routes, new_routes, old_routes)
        self.residual = residual
        self.option_policy = option_policy
        self.min_confidence = float(min_confidence)
        self.min_option_confidence = float(min_option_confidence)
        self.route_ids = tuple(sorted(int(key) for key in routes))
        self.enable_market_race_ordering = bool(
            enable_market_race_ordering
        )
        self.market_race_min_gain = float(market_race_min_gain)
        self.market_race = MarketRaceTracker()
        self.front_run_debts: dict[
            tuple[int, int], dict[str, int]
        ] = {}
        self._last_step: int | None = None
        self.last_strategy: dict[str, Any] | None = None

    def reset(self) -> None:
        self.base.reset()
        self.market_race.reset()
        self.front_run_debts = {}
        self._last_step = None
        self.last_strategy = None

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

    @staticmethod
    def _unpack_option(value: Any) -> tuple[V4Option | None, float]:
        if value is None:
            return None, 0.0
        if isinstance(value, V4Option):
            return value, 1.0
        if isinstance(value, tuple) and len(value) == 2:
            option, confidence = value
            if option is None or isinstance(option, V4Option):
                return option, float(confidence)
        raise TypeError(
            "option callback must return V4Option, "
            "(V4Option, confidence), or None"
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
            self.market_race.reset()
            self.front_run_debts = {}
        self._last_step = step
        self.market_race.observe(obs)

        base_route_id = self.base.route_id(obs, configuration)
        base_action = self.base.action_for_route(
            obs, base_route_id, configuration,
        )
        compatible_routes = self.base.compatible_route_ids(
            obs, configuration,
        )
        context = StepStrategyContext.from_observation(
            obs,
            configuration,
            base_route_id=base_route_id,
            route_ids=compatible_routes,
        )

        selected_route = base_route_id
        market_mode = "KEEP_ROUTE"
        option_confidence = 0.0
        option_applied = False
        action = base_action

        if self.option_policy is not None:
            try:
                option, option_confidence = self._unpack_option(
                    self.option_policy(obs, configuration, context)
                )
                if (
                    option is not None
                    and option_confidence >= self.min_option_confidence
                ):
                    option = validate_option(option, context)
                    selected_route = option.resolved_route(context)
                    market_mode = option.market_mode
                    action = self.base.action_for_route(
                        obs, selected_route, configuration,
                    )
                    action = compile_market_mode(
                        obs, action, option.market_mode,
                    )
                    option_applied = True
            except Exception:
                action = base_action
                selected_route = base_route_id
                market_mode = "KEEP_ROUTE"
                option_confidence = 0.0
                option_applied = False

        # A sale pulled forward earlier is removed from its original due step.
        for debt_key in [
            key for key in self.front_run_debts
            if key[1] < step
        ]:
            self.front_run_debts.pop(debt_key, None)
        due_debt = self.front_run_debts.pop(
            (int(selected_route), int(step)), None
        )
        action = suppress_due_sales(action, due_debt)

        front_run_metadata = None
        if market_mode in {"FRONT_RUN_1", "FRONT_RUN_9"}:
            horizon = 1 if market_mode == "FRONT_RUN_1" else 9
            existing_route_debts = {
                int(due_step): dict(debt)
                for (route_id, due_step), debt
                in self.front_run_debts.items()
                if int(route_id) == int(selected_route)
            }
            action, new_debts = front_run_future_sales(
                obs,
                action,
                self.base,
                int(selected_route),
                configuration,
                horizon=horizon,
                existing_debts=existing_route_debts,
            )
            for due_step, debt in new_debts.items():
                key = (int(selected_route), int(due_step))
                merged = self.front_run_debts.setdefault(key, {})
                for item, quantity in debt.items():
                    merged[item] = merged.get(item, 0) + int(quantity)
            front_run_metadata = {
                "horizon": horizon,
                "due_steps": sorted(int(x) for x in new_debts),
                "pulled_units": sum(
                    int(quantity)
                    for debt in new_debts.values()
                    for quantity in debt.values()
                ),
            }

        self.last_strategy = {
            "step": step,
            "day": clock.day,
            "hour": clock.hour,
            "remaining_steps": clock.remaining_steps,
            "phase": context.phase_name,
            "base_route_id": int(base_route_id),
            "selected_route_id": int(selected_route),
            "market_mode": market_mode,
            "option_confidence": float(option_confidence),
            "option_applied": bool(option_applied),
            "front_run": front_run_metadata,
        }

        if self.residual is not None:
            try:
                residual, confidence = self._unpack_residual(
                    self.residual(obs, configuration, action)
                )
                if residual is not None and confidence >= self.min_confidence:
                    action = apply_residual(action, residual)
            except Exception:
                pass

        race_metadata = None
        if self.enable_market_race_ordering:
            try:
                action, race = optimize_sell_order(
                    obs,
                    action,
                    self.market_race,
                    min_expected_margin_gain=self.market_race_min_gain,
                )
                race_metadata = {
                    "applied": bool(race.applied),
                    "expected_margin_gain": float(
                        race.expected_margin_gain
                    ),
                    "baseline_score": float(race.baseline_score),
                    "optimized_score": float(race.optimized_score),
                    "opponent_orders": race.opponent_orders,
                    "sell_positions": race.sell_positions,
                }
            except Exception:
                race_metadata = {
                    "applied": False,
                    "error": True,
                }

        self.market_race.record_action(action)
        if self.last_strategy is not None:
            self.last_strategy["market_race"] = race_metadata
        return action

    __call__ = act
