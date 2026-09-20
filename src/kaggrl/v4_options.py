from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Literal

from .clock import PHASE_NAMES, GameClock, resolve_clock
from .constants import PRODUCTS

MarketMode = Literal["KEEP_ROUTE", "NO_SPEND", "LIQUIDATE_SHED"]
MARKET_MODES: tuple[MarketMode, ...] = (
    "KEEP_ROUTE",
    "NO_SPEND",
    "LIQUIDATE_SHED",
)
# WHEAT and FERTILIZER are operational inputs. A safe liquidation must not
# remove them from the macro policy's future execution state.
SAFE_LIQUIDATION_PRODUCTS = tuple(
    item for item in PRODUCTS if item not in {"WHEAT", "FERTILIZER"}
)


@dataclass(frozen=True)
class StepStrategyContext:
    step: int
    day: int
    hour: int
    remaining_steps: int
    phase_index: int
    phase_name: str
    base_route_id: int
    route_ids: tuple[int, ...]
    unlocked_shops: tuple[str, ...]

    @classmethod
    def from_observation(
        cls,
        observation: dict[str, Any],
        configuration: Any,
        *,
        base_route_id: int,
        route_ids: tuple[int, ...],
    ) -> "StepStrategyContext":
        clock = resolve_clock(observation, configuration)
        shops = tuple(
            (observation.get("town", {}).get("unlocked_shops", []) or [])
        )
        return cls(
            step=clock.step,
            day=clock.day,
            hour=clock.hour,
            remaining_steps=clock.remaining_steps,
            phase_index=clock.phase_index,
            phase_name=PHASE_NAMES[clock.phase_index],
            base_route_id=int(base_route_id),
            route_ids=tuple(int(value) for value in route_ids),
            unlocked_shops=shops,
        )


@dataclass(frozen=True)
class V4Option:
    """One-step strategic decision.

    The policy re-evaluates this object every environment step.  It therefore
    carries absolute route/economic intent, not a long blind macro commitment.
    """

    route_id: int | None = None
    market_mode: MarketMode = "KEEP_ROUTE"

    def resolved_route(self, context: StepStrategyContext) -> int:
        route = context.base_route_id if self.route_id is None else int(self.route_id)
        if route not in context.route_ids:
            return context.base_route_id
        return route


def _private_shed(observation: dict[str, Any]) -> dict[str, int]:
    private = observation.get("private") or {}
    shed = private.get("shed") or {}
    if not isinstance(shed, dict):
        return {}
    return {
        str(item): max(0, int(amount or 0))
        for item, amount in shed.items()
        if str(item) in PRODUCTS and int(amount or 0) > 0
    }


def _sell_parts(order: Any) -> tuple[str, int] | None:
    if not isinstance(order, (list, tuple)) or len(order) < 3:
        return None
    if str(order[0]) != "SELL":
        return None
    item = str(order[1])
    if item not in PRODUCTS:
        return None
    try:
        quantity = int(order[2])
    except (TypeError, ValueError):
        return None
    if quantity <= 0:
        return None
    return item, quantity


def safe_sales_queue(
    observation: dict[str, Any],
    preferred_market: list[Any] | tuple[Any, ...],
    *,
    liquidate_all: bool,
    max_slots: int = 10,
) -> list[list[Any]]:
    """Build SELL-only orders bounded by inventory known in the private shed."""

    remaining = _private_shed(observation)
    out: list[list[Any]] = []

    for raw in preferred_market or []:
        parsed = _sell_parts(raw)
        if parsed is None:
            continue
        item, requested = parsed
        available = remaining.get(item, 0)
        quantity = min(requested, available)
        if quantity <= 0:
            continue
        out.append(["SELL", item, quantity])
        remaining[item] = available - quantity
        if len(out) >= max_slots:
            return out

    if liquidate_all:
        for item in PRODUCTS:
            quantity = remaining.get(item, 0)
            if quantity <= 0:
                continue
            out.append(["SELL", item, quantity])
            if len(out) >= max_slots:
                break
    return out[:max_slots]


def append_safe_liquidation_sales(
    observation: dict[str, Any],
    base_market: list[Any] | tuple[Any, ...],
    *,
    max_slots: int = 10,
) -> list[list[Any]]:
    """Preserve base orders and append only non-operational inventory sales."""

    out = [list(order) for order in (base_market or []) if order]
    if len(out) >= max_slots:
        return out[:max_slots]

    remaining = _private_shed(observation)
    # Reserve stock already requested by the base route so appended orders
    # cannot double-sell the same known inventory.
    for order in out:
        parsed = _sell_parts(order)
        if parsed is None:
            continue
        item, requested = parsed
        remaining[item] = max(
            0, remaining.get(item, 0) - requested
        )

    for item in SAFE_LIQUIDATION_PRODUCTS:
        quantity = remaining.get(item, 0)
        if quantity <= 0:
            continue
        out.append(["SELL", item, quantity])
        if len(out) >= max_slots:
            break
    return out[:max_slots]


def compile_market_mode(
    observation: dict[str, Any],
    action: dict[str, Any],
    mode: MarketMode,
) -> dict[str, Any]:
    out = copy.deepcopy(action)
    if mode == "KEEP_ROUTE":
        return out
    preferred = list(out.get("market") or [])
    if mode == "LIQUIDATE_SHED":
        out["market"] = append_safe_liquidation_sales(
            observation, preferred,
        )
        return out
    # NO_SPEND is retained only as a diagnostic/learned label. It is blocked
    # by safe runtimes by default because deleting scheduled purchases can
    # invalidate the macro trajectory.
    out["market"] = safe_sales_queue(
        observation,
        preferred,
        liquidate_all=False,
    )
    return out


def validate_option(
    option: V4Option,
    context: StepStrategyContext,
) -> V4Option:
    route_id = option.resolved_route(context)
    market_mode = (
        option.market_mode
        if option.market_mode in MARKET_MODES
        else "KEEP_ROUTE"
    )
    return V4Option(route_id=route_id, market_mode=market_mode)
