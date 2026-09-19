from __future__ import annotations

from typing import Any, Mapping

MOVEMENT_OPS = {"NORTH", "SOUTH", "EAST", "WEST"}
ACQUISITION_UNIT_OPS = {"PICKUP", "COLLECT_FERTILIZER"}
PRODUCTION_UNIT_OPS = {"PLACE", "PLANT", "BUILD_COOP", "BUILD_PASTURE", "DIG"}
MAINTENANCE_UNIT_OPS = {"WATER", "FERTILIZE", "FEED", "CARE"}

BEHAVIOR_FAMILIES = (
    "WAIT", "MOVEMENT", "ACQUISITION", "PRODUCTION", "MAINTENANCE",
    "HARVEST", "DEPOSIT", "SALE", "HIRE", "EXPANSION",
)


def _market_op(action: Mapping[str, Any]) -> str:
    kind = str(action.get("kind", "ORDER"))
    if kind in {"STOP_QUEUE", "NOP_SLOT"}:
        return kind
    return str(action.get("op", "NOP_SLOT"))


def behavior_family(action: Mapping[str, Any], domain: str) -> str:
    domain = str(domain)
    if domain == "market":
        op = _market_op(action)
        if op in {"STOP_QUEUE", "NOP_SLOT"}:
            return "WAIT"
        if op in {"BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL"}:
            return "ACQUISITION"
        if op == "SELL":
            return "SALE"
        if op == "HIRE":
            return "HIRE"
        if op == "BUY_LAND":
            return "EXPANSION"
        raise ValueError(f"unknown market operation: {op}")

    if domain != "unit":
        raise ValueError(f"unknown behavior domain: {domain}")
    op = str(action.get("op", "PASS"))
    if op == "PASS":
        return "WAIT"
    if op in MOVEMENT_OPS:
        return "MOVEMENT"
    if op in ACQUISITION_UNIT_OPS:
        return "ACQUISITION"
    if op in PRODUCTION_UNIT_OPS:
        return "PRODUCTION"
    if op in MAINTENANCE_UNIT_OPS:
        return "MAINTENANCE"
    if op == "HARVEST":
        return "HARVEST"
    if op == "DROP":
        return "DEPOSIT"
    raise ValueError(f"unknown unit operation: {op}")


def compute_family_weights(
    counts: Mapping[str, int],
    cap: float,
    normalize: bool = True,
) -> dict[str, float]:
    del normalize
    if float(cap) < 1.0:
        raise ValueError("family weight cap must be >= 1")
    active = {str(name): int(count) for name, count in counts.items() if int(count) > 0}
    if not active:
        raise ValueError("family counts must contain a positive count")
    maximum = max(active.values())
    weights = {
        name: min(float(cap), float(maximum) / float(count))
        for name, count in sorted(active.items())
    }
    if "WAIT" in weights:
        weights["WAIT"] = 1.0
    return weights


def compute_domain_family_weights(
    counts_by_domain: Mapping[str, Mapping[str, int]],
    cap: float,
) -> dict[str, dict[str, float]]:
    return {
        str(domain): compute_family_weights(counts, cap)
        for domain, counts in sorted(counts_by_domain.items())
    }
