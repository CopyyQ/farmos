from __future__ import annotations

import math


def signed_log1p(value: float | int) -> float:
    x = float(value or 0.0)
    return math.copysign(math.log1p(abs(x)), x) if x else 0.0


def economic_market_features_from_shadow_ledger(ledger) -> tuple[float, ...]:
    cash = int(getattr(ledger, "cash_lower_bound", 0) or 0)
    next_hire = int(getattr(ledger, "next_hire_cost", 0) or 0)
    next_land_value = getattr(ledger, "next_land_cost", None)
    next_land = int(next_land_value or 0)
    hires_today = int(getattr(ledger, "hires_today", 0) or 0)
    unit_positions = getattr(ledger, "unit_positions", {}) or {}
    land_count = len(getattr(ledger, "unlocked_quadrants", []) or [])
    shed = getattr(ledger, "shed", {}) or {}
    unit_inventory = getattr(ledger, "unit_inventories", {}) or {}
    prices = getattr(ledger, "market_prices", {}) or {}

    inventory_value = 0
    shed_units = 0
    for item, amount in shed.items():
        q = max(0, int(amount or 0))
        shed_units += q
        inventory_value += q * max(0, int(prices.get(item, 0) or 0))
    for inventory in unit_inventory.values():
        for item, amount in (inventory or {}).items():
            q = max(0, int(amount or 0))
            inventory_value += q * max(0, int(prices.get(item, 0) or 0))

    market_slots_used = int(getattr(ledger, "market_slots_used", 0) or 0)
    step = int(getattr(ledger, "step", 0) or 0)

    cash_log = signed_log1p(cash)
    hire_log = signed_log1p(next_hire)
    land_log = signed_log1p(next_land)

    return (
        cash_log,
        hire_log,
        land_log,
        cash_log - hire_log,
        cash_log - land_log if next_land > 0 else 0.0,
        signed_log1p(hires_today),
        signed_log1p(len(unit_positions)),
        signed_log1p(land_count),
        signed_log1p(inventory_value),
        signed_log1p(shed_units),
        signed_log1p(max(0, 10 - market_slots_used)),
        signed_log1p(step),
    )
