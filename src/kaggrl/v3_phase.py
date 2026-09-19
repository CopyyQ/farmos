from __future__ import annotations

from enum import Enum
from typing import Any, Mapping

from .v3_behavior import behavior_family


class PhaseLabel(str, Enum):
    ACQUIRE = "ACQUIRE"
    MOVE_TO_TARGET = "MOVE_TO_TARGET"
    PRODUCE = "PRODUCE"
    MAINTAIN = "MAINTAIN"
    HARVEST = "HARVEST"
    DEPOSIT = "DEPOSIT"
    SELL = "SELL"
    HIRE = "HIRE"
    EXPAND = "EXPAND"
    WAIT = "WAIT"
    RECOVER = "RECOVER"
    UNKNOWN = "UNKNOWN"


FAMILY_TO_PHASE = {
    "WAIT": PhaseLabel.WAIT,
    "MOVEMENT": PhaseLabel.MOVE_TO_TARGET,
    "ACQUISITION": PhaseLabel.ACQUIRE,
    "PRODUCTION": PhaseLabel.PRODUCE,
    "MAINTENANCE": PhaseLabel.MAINTAIN,
    "HARVEST": PhaseLabel.HARVEST,
    "DEPOSIT": PhaseLabel.DEPOSIT,
    "SALE": PhaseLabel.SELL,
    "HIRE": PhaseLabel.HIRE,
    "EXPANSION": PhaseLabel.EXPAND,
}

PLAN_PHASES = (
    PhaseLabel.ACQUIRE,
    PhaseLabel.MOVE_TO_TARGET,
    PhaseLabel.PRODUCE,
    PhaseLabel.MAINTAIN,
    PhaseLabel.HARVEST,
    PhaseLabel.DEPOSIT,
    PhaseLabel.SELL,
    PhaseLabel.HIRE,
    PhaseLabel.EXPAND,
    PhaseLabel.RECOVER,
)

PHASE_ORDER = {
    PhaseLabel.ACQUIRE: 0,
    PhaseLabel.MOVE_TO_TARGET: 1,
    PhaseLabel.PRODUCE: 2,
    PhaseLabel.MAINTAIN: 3,
    PhaseLabel.HARVEST: 4,
    PhaseLabel.DEPOSIT: 5,
    PhaseLabel.SELL: 6,
    PhaseLabel.HIRE: 7,
    PhaseLabel.EXPAND: 8,
    PhaseLabel.WAIT: 9,
    PhaseLabel.RECOVER: 10,
    PhaseLabel.UNKNOWN: 11,
}

INTENT_LABELS = tuple(
    phase
    for phase in sorted(PHASE_ORDER, key=PHASE_ORDER.__getitem__)
    if phase is not PhaseLabel.UNKNOWN
)
INTENT_TO_INDEX = {phase: index for index, phase in enumerate(INTENT_LABELS)}


def _has_recovery_evidence(effect: Mapping[str, Any]) -> bool:
    return bool(
        effect.get("recovery")
        or effect.get("recovered")
        or effect.get("recovered_from_invalid")
    )


def derive_phase_candidates(
    action: Mapping[str, Any],
    effect: Mapping[str, Any],
) -> tuple[PhaseLabel, ...]:
    phases: set[PhaseLabel] = set()
    farmer = action.get("farmer") or {"op": "PASS"}
    phases.add(FAMILY_TO_PHASE[behavior_family(farmer, "unit")])
    for command in action.get("hands") or []:
        phases.add(FAMILY_TO_PHASE[behavior_family(command, "unit")])
    for slot in action.get("market") or []:
        phases.add(FAMILY_TO_PHASE[behavior_family(slot, "market")])
    if _has_recovery_evidence(effect):
        phases.add(PhaseLabel.RECOVER)

    non_wait = {phase for phase in phases if phase is not PhaseLabel.WAIT}
    selected = non_wait if non_wait else {PhaseLabel.WAIT}
    return tuple(sorted(selected, key=PHASE_ORDER.__getitem__))


def derive_plan_set(
    state: Mapping[str, Any],
    action: Mapping[str, Any],
    effect: Mapping[str, Any],
) -> frozenset[PhaseLabel]:
    del state
    candidates = derive_phase_candidates(action, effect)
    return frozenset(
        phase for phase in candidates
        if phase not in {PhaseLabel.WAIT, PhaseLabel.UNKNOWN}
    )


def intent_mask(
    candidates: tuple[PhaseLabel, ...] | list[PhaseLabel],
) -> tuple[bool, ...]:
    active = set(candidates)
    if PhaseLabel.UNKNOWN in active:
        raise ValueError("UNKNOWN is not a trainable concurrent intent")
    return tuple(phase in active for phase in INTENT_LABELS)


def derive_phase(
    state: Mapping[str, Any],
    action: Mapping[str, Any],
    effect: Mapping[str, Any],
) -> PhaseLabel:
    del state
    candidates = derive_phase_candidates(action, effect)
    if len(candidates) == 1:
        return candidates[0]
    return PhaseLabel.UNKNOWN


def derive_transition_target(
    current_phase: PhaseLabel | str,
    next_phase: PhaseLabel | str,
) -> bool | None:
    current = PhaseLabel(current_phase)
    following = PhaseLabel(next_phase)
    if PhaseLabel.UNKNOWN in {current, following}:
        return None
    return current != following
