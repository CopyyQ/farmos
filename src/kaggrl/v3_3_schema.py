from __future__ import annotations

from .v3_2_schema import (
    ACTIVE_MARKET_OPS,
    CONTINUE_ID,
    STOP_ID,
    STRATEGY_CORE_SCALE,
    OPENING_ACTIVE_INTENT_SCALE,
)

ARCHITECTURE_VERSION = "rl_v3_3_dagger_economic_market"
FORMAT_VERSION = 5

ECONOMIC_MARKET_SCALE = 1.0
SHORT_ECONOMIC_HORIZONS = (1, 4, 24, 48)
SHORT_ECONOMIC_FEATURES = tuple(
    f"{name}_h{horizon}"
    for horizon in SHORT_ECONOMIC_HORIZONS
    for name in ("money_delta", "net_worth_delta")
)
SHORT_ECONOMIC_DIM = len(SHORT_ECONOMIC_FEATURES)

ECONOMIC_MARKET_FEATURES = (
    "cash",
    "next_hire_cost",
    "next_land_cost",
    "hire_affordability",
    "land_affordability",
    "hires_today",
    "unit_count",
    "land_count",
    "inventory_value",
    "shed_units",
    "market_slots_remaining",
    "step",
)
ECONOMIC_MARKET_DIM = len(ECONOMIC_MARKET_FEATURES)
