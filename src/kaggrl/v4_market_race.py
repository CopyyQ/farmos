from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any

from .constants import PRODUCTS

MARKET_I0 = 10000
PRICE_FLOOR = 1
MARKET_PARAMS = {
    "WHEAT":      {"base": 25, "I0": MARKET_I0, "T": 400, "below_func": "sqrt", "below_target": 0.80, "above_func": "log", "above_target": 0.20},
    "CARROT":     {"base": 35, "I0": MARKET_I0, "T": 450, "below_func": "hinge", "below_target": 1.00, "above_func": "sqrt", "above_target": 0.70},
    "TOMATO":     {"base": 60, "I0": MARKET_I0, "T": 200, "below_func": "hinge", "below_target": 0.40, "above_func": "sqrt", "above_target": 0.60},
    "STRAWBERRY": {"base": 120, "I0": MARKET_I0, "T": 100, "below_func": "sqrt", "below_target": 0.70, "above_func": "linear", "above_target": 1.60},
    "MELON":      {"base": 250, "I0": MARKET_I0, "T": 300, "below_func": "log", "below_target": 0.20, "above_func": "sq", "above_target": 3.60},
    "EGG":        {"base": 50, "I0": MARKET_I0, "T": 332, "below_func": "hinge", "below_target": 0.40, "above_func": "log", "above_target": 0.20},
    "MILK":       {"base": 160, "I0": MARKET_I0, "T": 122, "below_func": "sqrt", "below_target": 0.60, "above_func": "linear", "above_target": 1.60},
    "WOOL":       {"base": 200, "I0": MARKET_I0, "T": 105, "below_func": "log", "below_target": 0.20, "above_func": "sq", "above_target": 3.20},
    "FERTILIZER": {"base": 100, "I0": MARKET_I0, "T": 200, "below_func": "linear", "below_target": 0.40, "above_func": "linear", "above_target": 0.40},
}
HINGE_GAIN = 8.0
ANIMAL_PRODUCT = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}


def _shape(func: str, x: float, t: float | None = None) -> float:
    x = max(0.0, float(x))
    if func == "linear":
        return x
    if func == "sq":
        return x * x
    if func == "sqrt":
        return math.sqrt(x)
    if func == "log":
        return math.log(1.0 + x)
    if func == "log10":
        return math.log10(1.0 + x)
    if func == "hinge":
        if not t or t <= 0:
            return x
        u = x / t
        return u + HINGE_GAIN * max(0.0, u - 1.0) ** 2
    return x


def market_price(
    item: str,
    inventory: int,
    params: dict[str, Any] | None = None,
) -> int:
    """Bit-compatible with the Kaggriculture reference pricing formula."""

    p = (params or MARKET_PARAMS)[item]
    base = float(p["base"])
    i0 = int(p["I0"])
    t = float(p["T"])
    if inventory < i0:
        func = str(p["below_func"])
        amp = float(p["below_target"]) * base / _shape(func, t, t)
        price = base + amp * _shape(func, i0 - inventory, t)
    else:
        func = str(p["above_func"])
        amp = float(p["above_target"]) * base / _shape(func, t, t)
        price = base - amp * _shape(func, inventory - i0, t)
    return max(PRICE_FLOOR, int(round(price)))


def _sell(order: Any) -> tuple[str, int] | None:
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


def _own_market_net(action: dict[str, Any]) -> dict[str, int]:
    net = {item: 0 for item in PRODUCTS}
    for order in action.get("market") or []:
        parsed = _sell(order)
        if parsed is not None:
            item, quantity = parsed
            net[item] += quantity
            continue
        if (
            isinstance(order, (list, tuple))
            and len(order) >= 3
            and str(order[0]) == "BUY_PRODUCT"
            and str(order[1]) in net
        ):
            try:
                net[str(order[1])] -= max(0, int(order[2]))
            except (TypeError, ValueError):
                pass
    return net


def _rival_production_prior(observation: dict[str, Any]) -> dict[str, float]:
    prior = {item: 0.0 for item in PRODUCTS}
    farms = list(observation.get("farms") or [])
    if len(farms) < 2:
        return prior
    player = max(0, min(int(observation.get("player", 0) or 0), len(farms) - 1))
    rival = farms[1 - player]
    tiles = rival.get("tiles", []) if isinstance(rival, dict) else []
    for row in tiles or []:
        for tile in row or []:
            if not isinstance(tile, dict):
                continue
            crop = tile.get("crop")
            animal = tile.get("animal")
            yield_units = max(0.0, float(tile.get("yield_units", 0) or 0))
            if crop in prior:
                prior[crop] += 0.25 + 0.10 * yield_units
            product = ANIMAL_PRODUCT.get(str(animal))
            if product in prior:
                prior[product] += 0.35 + 0.15 * yield_units
    return prior


@dataclass
class MarketRaceDecision:
    applied: bool
    expected_margin_gain: float
    baseline_score: float
    optimized_score: float
    opponent_orders: list[list[Any]]
    sell_positions: list[int]


class MarketRaceTracker:
    """Estimate which products the rival is likely to dump next.

    Positive market-inventory residuals, after subtracting our previous
    requested market flow, are treated as conservative evidence of rival sales.
    End-of-day town consumption is negative, so it cannot create false positive
    sale pressure under this estimator.
    """

    def __init__(self, *, decay: float = 0.75):
        self.decay = float(decay)
        self.pressure = {item: 0.0 for item in PRODUCTS}
        self.previous_inventory: dict[str, int] | None = None
        self.previous_own_net = {item: 0 for item in PRODUCTS}

    def reset(self) -> None:
        self.pressure = {item: 0.0 for item in PRODUCTS}
        self.previous_inventory = None
        self.previous_own_net = {item: 0 for item in PRODUCTS}

    def observe(self, observation: dict[str, Any]) -> None:
        market = observation.get("market") or {}
        inventory = market.get("inventory") or {}
        current = {
            item: int(inventory.get(item, MARKET_I0) or MARKET_I0)
            for item in PRODUCTS
        }
        if self.previous_inventory is not None:
            for item in PRODUCTS:
                observed_delta = current[item] - self.previous_inventory[item]
                residual = observed_delta - int(self.previous_own_net.get(item, 0))
                inferred_rival_sale = max(0.0, float(residual))
                self.pressure[item] = (
                    self.decay * self.pressure[item]
                    + inferred_rival_sale
                )
        self.previous_inventory = current
        self.previous_own_net = {item: 0 for item in PRODUCTS}

    def record_action(self, action: dict[str, Any]) -> None:
        self.previous_own_net = _own_market_net(action)

    def predicted_opponent_orders(
        self,
        observation: dict[str, Any],
        *,
        max_orders: int = 5,
        max_quantity: int = 80,
        min_pressure: float = 0.75,
    ) -> list[list[Any]]:
        prior = _rival_production_prior(observation)
        scored = []
        for item in PRODUCTS:
            score = float(self.pressure[item]) + float(prior[item])
            if score < min_pressure:
                continue
            quantity = max(1, min(max_quantity, int(round(score))))
            scored.append((score, item, quantity))
        scored.sort(key=lambda row: (-row[0], row[1]))
        return [
            ["SELL", item, quantity]
            for _, item, quantity in scored[:max_orders]
        ]


def simulate_lockstep_sell_margin(
    observation: dict[str, Any],
    our_market: list[Any] | tuple[Any, ...],
    opponent_market: list[Any] | tuple[Any, ...],
) -> tuple[float, float, float]:
    """Simulate SELL-only price interaction using reference lockstep quoting."""

    market = observation.get("market") or {}
    inventory = {
        item: int((market.get("inventory") or {}).get(item, MARKET_I0) or MARKET_I0)
        for item in PRODUCTS
    }
    params = market.get("params")
    our_revenue = 0.0
    opponent_revenue = 0.0
    slots = max(len(our_market or []), len(opponent_market or []))
    for slot in range(slots):
        ours = _sell(our_market[slot]) if slot < len(our_market or []) else None
        theirs = (
            _sell(opponent_market[slot])
            if slot < len(opponent_market or [])
            else None
        )
        own_item, own_left = ours if ours is not None else (None, 0)
        opp_item, opp_left = theirs if theirs is not None else (None, 0)
        while own_left > 0 or opp_left > 0:
            own_quote = (
                market_price(own_item, inventory[own_item], params)
                if own_left > 0 else None
            )
            opp_quote = (
                market_price(opp_item, inventory[opp_item], params)
                if opp_left > 0 else None
            )
            if own_left > 0:
                our_revenue += float(own_quote)
                inventory[own_item] += 1
                own_left -= 1
            if opp_left > 0:
                opponent_revenue += float(opp_quote)
                inventory[opp_item] += 1
                opp_left -= 1
    return our_revenue, opponent_revenue, our_revenue - opponent_revenue


def quote_exposure_priority(
    observation: dict[str, Any],
    order: Any,
) -> float:
    parsed = _sell(order)
    if parsed is None:
        return 0.0
    item, quantity = parsed
    market = observation.get("market") or {}
    inventory = int(
        (market.get("inventory") or {}).get(item, MARKET_I0)
        or MARKET_I0
    )
    params = market.get("params")
    farms = list(observation.get("farms") or [])
    standing = 0
    if len(farms) >= 2:
        player = max(
            0, min(int(observation.get("player", 0) or 0), len(farms) - 1)
        )
        rival = farms[1 - player]
        crop_item = item if item in {
            "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"
        } else None
        animal = {
            "EGG": "GOOSE", "MILK": "COW", "WOOL": "SHEEP"
        }.get(item)
        for row in (rival.get("tiles", []) if isinstance(rival, dict) else []):
            for tile in row or []:
                if not isinstance(tile, dict):
                    continue
                if (
                    (crop_item is not None and tile.get("crop") == crop_item)
                    or (animal is not None and tile.get("animal") == animal)
                ):
                    standing += max(0, int(tile.get("yield_units", 0) or 0))
    batch = min(24, max(8, standing))
    now = sum(
        market_price(item, inventory + offset, params)
        for offset in range(quantity)
    )
    later = sum(
        market_price(item, inventory + batch + offset, params)
        for offset in range(quantity)
    )
    return float(now - later)


def optimize_sell_order(
    observation: dict[str, Any],
    action: dict[str, Any],
    tracker: MarketRaceTracker,
    *,
    min_expected_margin_gain: float = 1.0,
    max_exact_sells: int = 5,
) -> tuple[dict[str, Any], MarketRaceDecision]:
    """Reorder only SELL slots to maximize expected lockstep margin.

    Non-SELL orders never move. This preserves macro dependencies while letting
    vulnerable products occupy earlier sale slots when the rival is expected to
    dump the same market.
    """

    out = {
        **action,
        "market": [list(order) for order in (action.get("market") or [])],
    }
    market = out["market"]
    sell_positions = [
        index for index, order in enumerate(market)
        if _sell(order) is not None
    ]
    opponent = tracker.predicted_opponent_orders(observation)

    # Only reorder contiguous SELL blocks. Never move a SELL across BUY/HIRE/
    # BUY_LAND because those separators are part of the macro's economic
    # dependency ordering.
    blocks: list[tuple[int, int]] = []
    index = 0
    while index < len(market):
        if _sell(market[index]) is None:
            index += 1
            continue
        end = index + 1
        while end < len(market) and _sell(market[end]) is not None:
            end += 1
        if 2 <= end - index <= 6:
            blocks.append((index, end))
        index = end

    if not blocks:
        return out, MarketRaceDecision(
            applied=False,
            expected_margin_gain=0.0,
            baseline_score=0.0,
            optimized_score=0.0,
            opponent_orders=opponent,
            sell_positions=sell_positions,
        )

    # General case from the stronger agent: rank each distinct sale by how much
    # revenue it would lose if a small rival batch hit that product first.
    # Earlier positions receive larger weights because they are more valuable
    # in a slot-by-slot market race.
    best_market = [list(order) for order in market]
    baseline_score = 0.0
    best_score = 0.0
    for start, end in blocks:
        block = best_market[start:end]
        if len({str(order[1]) for order in block}) != len(block):
            continue
        priorities = [
            quote_exposure_priority(observation, order)
            for order in block
        ]
        width = len(block)
        baseline_score += sum(
            (width - offset) * priority
            for offset, priority in enumerate(priorities)
        )
        ranked = sorted(
            zip(block, priorities),
            key=lambda row: row[1],
            reverse=True,
        )
        ranked_block = [list(order) for order, _ in ranked]
        ranked_priorities = [priority for _, priority in ranked]
        best_score += sum(
            (width - offset) * priority
            for offset, priority in enumerate(ranked_priorities)
        )
        best_market[start:end] = ranked_block

    gain = best_score - float(baseline_score)
    applied = bool(
        gain >= float(min_expected_margin_gain)
        and best_market != market
    )
    if applied:
        out["market"] = best_market
    return out, MarketRaceDecision(
        applied=applied,
        expected_margin_gain=float(max(0.0, gain)),
        baseline_score=float(baseline_score),
        optimized_score=float(best_score if applied else baseline_score),
        opponent_orders=opponent,
        sell_positions=sell_positions,
    )


def suppress_due_sales(
    action: dict[str, Any],
    debt: dict[str, int] | None,
) -> dict[str, Any]:
    if not debt:
        return {**action, "market": [list(x) for x in action.get("market") or []]}
    remaining = {str(k): max(0, int(v)) for k, v in debt.items()}
    out_market: list[list[Any]] = []
    for raw in action.get("market") or []:
        order = list(raw)
        parsed = _sell(order)
        if parsed is None:
            out_market.append(order)
            continue
        item, quantity = parsed
        cut = min(quantity, remaining.get(item, 0))
        remaining[item] = max(0, remaining.get(item, 0) - cut)
        quantity -= cut
        if quantity > 0:
            order[2] = quantity
            out_market.append(order)
    return {**action, "market": out_market}


def front_run_future_sales(
    observation: dict[str, Any],
    action: dict[str, Any],
    macro,
    route_id: int,
    configuration=None,
    *,
    horizon: int,
    max_orders: int = 10,
    existing_debts: dict[int, dict[str, int]] | None = None,
) -> tuple[dict[str, Any], dict[int, dict[str, int]]]:
    from .clock import resolve_clock

    clock = resolve_clock(observation, configuration)
    step = int(clock.step)
    if step < 192 or step >= 696 or horizon <= 0:
        return action, {}
    market_obs = observation.get("market") or {}
    prices = market_obs.get("prices") or {}
    shed = (observation.get("private") or {}).get("shed") or {}
    available = {
        item: max(0, int(shed.get(item, 0) or 0))
        for item in PRODUCTS
    }
    out = {**action, "market": [list(x) for x in action.get("market") or []]}
    blocked = set()
    for raw in out["market"]:
        parsed = _sell(raw)
        if parsed is not None:
            item, quantity = parsed
            available[item] = max(0, available.get(item, 0) - quantity)
            blocked.add(item)
        elif (
            isinstance(raw, (list, tuple))
            and len(raw) >= 2
            and str(raw[0]) == "BUY_PRODUCT"
        ):
            blocked.add(str(raw[1]))

    commands = [out.get("farmer") or ["PASS"], *(out.get("hands") or [])]
    for command in commands:
        if (
            isinstance(command, (list, tuple))
            and len(command) >= 2
            and str(command[0]) == "PICKUP"
        ):
            blocked.add(str(command[1]))
    debts: dict[int, dict[str, int]] = {}
    end = min(695, step + int(horizon))
    for item in PRODUCTS:
        if (
            item in blocked
            or available.get(item, 0) <= 0
            or int(prices.get(item, 0) or 0)
            <= int(MARKET_PARAMS[item]["base"])
        ):
            continue
        remaining = int(available[item])
        reservations: list[tuple[int, int]] = []
        for due_step in range(step + 1, end + 1):
            day, hour = divmod(due_step, 24)
            future_obs = dict(observation)
            future_obs["step"] = due_step
            future_obs["day"] = day
            future_obs["hour"] = hour
            future = macro.action_for_route(
                future_obs, int(route_id), configuration
            )
            future_commands = [
                future.get("farmer") or ["PASS"],
                *(future.get("hands") or []),
            ]
            if any(
                isinstance(command, (list, tuple))
                and len(command) >= 2
                and list(command[:2]) == ["PICKUP", item]
                for command in future_commands
            ):
                break
            if any(
                isinstance(order, (list, tuple))
                and len(order) >= 2
                and list(order[:2]) == ["BUY_PRODUCT", item]
                for order in (future.get("market") or [])
            ):
                break
            planned = sum(
                max(0, int(order[2]))
                for order in (future.get("market") or [])
                if _sell(order) is not None and str(order[1]) == item
            )
            already_reserved = int(
                ((existing_debts or {}).get(due_step, {}) or {}).get(
                    item, 0
                )
            )
            planned = max(0, planned - already_reserved)
            amount = min(remaining, planned)
            if amount > 0:
                reservations.append((due_step, amount))
                remaining -= amount
            if remaining <= 0:
                break

        quantity = sum(amount for _, amount in reservations)
        if quantity <= 0 or len(out["market"]) >= max_orders:
            continue
        out["market"].append(["SELL", item, quantity])
        for due_step, amount in reservations:
            due = debts.setdefault(int(due_step), {})
            due[item] = due.get(item, 0) + int(amount)

    return out, debts
