from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from kaggrl.v3_phase import (
    INTENT_LABELS,
    PHASE_ORDER,
    PLAN_PHASES,
    PhaseLabel,
    intent_mask,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_intents(raw) -> tuple[PhaseLabel, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("intent list must be non-empty")
    values = tuple(PhaseLabel(str(item)) for item in raw)
    if PhaseLabel.UNKNOWN in values:
        raise ValueError("UNKNOWN is not allowed in concurrent intent labels")
    if len(set(values)) != len(values):
        raise ValueError("intent list contains duplicates")
    ordered = tuple(sorted(values, key=PHASE_ORDER.__getitem__))
    if values != ordered:
        raise ValueError("intent list is not in frozen order")
    if PhaseLabel.WAIT in values and len(values) != 1:
        raise ValueError("WAIT cannot coexist with non-WAIT intents")
    return values


def _parse_mask(raw, intents: tuple[PhaseLabel, ...]) -> tuple[bool, ...]:
    if not isinstance(raw, list) or len(raw) != len(INTENT_LABELS):
        raise ValueError("intent mask shape mismatch")
    mask = tuple(bool(value) for value in raw)
    if mask != intent_mask(intents):
        raise ValueError("intent mask does not match intent list")
    return mask


def audit_phase_sidecar(path: str | Path) -> dict:
    path = Path(path)
    rows = pq.read_table(path).to_pylist()
    intent_counts = Counter()
    cardinality_counts = Counter()
    malformed = 0

    for row in rows:
        try:
            current = _parse_intents(row.get("intents"))
            current_mask = _parse_mask(row.get("intent_mask"), current)
            cardinality_counts[len(current)] += 1
            for item in current:
                intent_counts[item.value] += 1

            expected_single = current[0].value if len(current) == 1 else None
            if row.get("single_phase") != expected_single:
                raise ValueError("single_phase diagnostic mismatch")

            raw_next = row.get("next_intents")
            raw_next_mask = row.get("next_intent_mask")
            raw_transition = row.get("transition_count")
            if raw_next is None:
                if raw_next_mask is not None or raw_transition is not None:
                    raise ValueError("terminal row has next-intent payload")
                continue

            following = _parse_intents(raw_next)
            following_mask = _parse_mask(raw_next_mask, following)
            expected_transition = sum(
                bool(a) != bool(b)
                for a, b in zip(current_mask, following_mask)
            )
            if raw_transition is None or int(raw_transition) != expected_transition:
                raise ValueError("transition_count mismatch")
        except (KeyError, TypeError, ValueError):
            malformed += 1

    total = len(rows)
    return {
        "rows": total,
        "intent_counts": {
            phase.value: int(intent_counts.get(phase.value, 0))
            for phase in (*INTENT_LABELS, PhaseLabel.UNKNOWN)
        },
        "cardinality_counts": {
            str(key): int(value)
            for key, value in sorted(cardinality_counts.items())
        },
        "multi_intent_fraction": (
            float(sum(v for k, v in cardinality_counts.items() if k > 1)) / total
            if total else 0.0
        ),
        "malformed_rows": int(malformed),
        "sha256": _sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    print(json.dumps(
        audit_phase_sidecar(args.path), indent=2, sort_keys=True,
    ))


if __name__ == "__main__":
    main()


def _parse_plan_list(raw) -> tuple[PhaseLabel, ...]:
    if not isinstance(raw, list):
        raise ValueError("plan list must be a list")
    values = tuple(PhaseLabel(str(item)) for item in raw)
    if any(item not in PLAN_PHASES for item in values):
        raise ValueError("plan list contains non-plan label")
    if len(set(values)) != len(values):
        raise ValueError("plan list contains duplicates")
    ordered = tuple(sorted(values, key=PHASE_ORDER.__getitem__))
    if values != ordered:
        raise ValueError("plan list is not in frozen order")
    return values


def _parse_plan_mask(raw, plans: tuple[PhaseLabel, ...]) -> tuple[bool, ...]:
    if not isinstance(raw, list) or len(raw) != len(PLAN_PHASES):
        raise ValueError("plan mask shape mismatch")
    mask = tuple(bool(value) for value in raw)
    expected = tuple(phase in set(plans) for phase in PLAN_PHASES)
    if mask != expected:
        raise ValueError("plan mask does not match plan list")
    return mask


def audit_plan_sidecar(path: str | Path) -> dict:
    path = Path(path)
    rows = pq.read_table(path).to_pylist()
    plan_counts = Counter()
    cardinality_counts = Counter()
    malformed = 0

    for row in rows:
        try:
            current = _parse_plan_list(row.get("active_plans"))
            current_mask = _parse_plan_mask(row.get("plan_mask"), current)
            cardinality_counts[len(current)] += 1
            for item in current:
                plan_counts[item.value] += 1

            has_next = bool(row.get("has_next"))
            following = _parse_plan_list(row.get("next_active_plans"))
            following_mask = _parse_plan_mask(
                row.get("next_plan_mask"), following,
            )
            transition = row.get("transition_count")
            if not has_next:
                if following or transition is not None:
                    raise ValueError("terminal row has next-plan payload")
                continue
            expected_transition = sum(
                bool(a) != bool(b)
                for a, b in zip(current_mask, following_mask)
            )
            if transition is None or int(transition) != expected_transition:
                raise ValueError("transition_count mismatch")
        except (KeyError, TypeError, ValueError):
            malformed += 1

    total = len(rows)
    multi = sum(v for k, v in cardinality_counts.items() if k > 1)
    return {
        "rows": total,
        "plan_counts": {
            phase.value: int(plan_counts.get(phase.value, 0))
            for phase in PLAN_PHASES
        },
        "cardinality_counts": {
            str(key): int(value)
            for key, value in sorted(cardinality_counts.items())
        },
        "multi_intent_rows": int(multi),
        "multi_intent_fraction": (
            float(multi) / float(total) if total else 0.0
        ),
        "malformed_rows": int(malformed),
        "sha256": _sha256(path),
    }
