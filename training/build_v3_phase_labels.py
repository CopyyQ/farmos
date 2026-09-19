from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from kaggrl.v3_phase import (
    INTENT_LABELS,
    PHASE_ORDER,
    PLAN_PHASES,
    derive_phase_candidates,
    derive_plan_set,
    intent_mask,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _transition_count(current: tuple[bool, ...], following: tuple[bool, ...]) -> int:
    if len(current) != len(following):
        raise ValueError("intent mask length mismatch")
    return sum(bool(a) != bool(b) for a, b in zip(current, following))


def build_phase_sidecar(
    dataset_path: str | Path,
    *,
    split: str = "train",
    out_path: str | Path,
) -> dict:
    dataset_path = Path(dataset_path)
    out_path = Path(out_path)
    source = pq.read_table(
        dataset_path,
        filters=[("split", "=", str(split)), ("role", "=", "active_best")],
        columns=[
            "episode_id", "seat", "step",
            "canonical_action_json", "effects_json",
        ],
    ).to_pylist()
    grouped: dict[tuple[int, int], list[dict]] = {}
    for row in source:
        key = (int(row["episode_id"]), int(row["seat"]))
        grouped.setdefault(key, []).append(row)

    records: list[dict] = []
    intent_counts = Counter()
    cardinality_counts = Counter()

    for (episode_id, seat), episode_rows in sorted(grouped.items()):
        ordered = sorted(episode_rows, key=lambda row: int(row["step"]))
        candidates = [
            derive_phase_candidates(
                json.loads(row["canonical_action_json"]),
                json.loads(row["effects_json"]),
            )
            for row in ordered
        ]
        masks = [intent_mask(value) for value in candidates]
        for index, (row, active, mask) in enumerate(
            zip(ordered, candidates, masks)
        ):
            next_active = candidates[index + 1] if index + 1 < len(candidates) else None
            next_mask = masks[index + 1] if index + 1 < len(masks) else None
            for item in active:
                intent_counts[item.value] += 1
            cardinality_counts[len(active)] += 1
            records.append({
                "episode_id": episode_id,
                "seat": seat,
                "step": int(row["step"]),
                "intents": [item.value for item in active],
                "intent_mask": list(mask),
                "next_intents": (
                    None if next_active is None
                    else [item.value for item in next_active]
                ),
                "next_intent_mask": None if next_mask is None else list(next_mask),
                "single_phase": active[0].value if len(active) == 1 else None,
                "transition_count": (
                    None if next_mask is None
                    else _transition_count(mask, next_mask)
                ),
            })

    if not records:
        raise ValueError("intent sidecar would be empty")
    table = pa.Table.from_pylist(records)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp = out_path.with_suffix(out_path.suffix + ".tmp")
    pq.write_table(table, temp, compression="zstd")
    temp.replace(out_path)
    total = len(records)
    return {
        "rows": total,
        "intent_labels": [phase.value for phase in INTENT_LABELS],
        "intent_counts": {
            phase.value: int(intent_counts.get(phase.value, 0))
            for phase in INTENT_LABELS
        },
        "cardinality_counts": {
            str(key): int(value)
            for key, value in sorted(cardinality_counts.items())
        },
        "multi_intent_fraction": float(
            sum(value for key, value in cardinality_counts.items() if key > 1)
        ) / float(total),
        "sha256": _sha256(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(build_phase_sidecar(
        args.dataset, split=args.split, out_path=args.out,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


def _plan_mask(plans) -> tuple[bool, ...]:
    active = set(plans)
    return tuple(phase in active for phase in PLAN_PHASES)


def build_plan_sidecar(
    dataset_path: str | Path,
    *,
    split: str = "train",
    out_path: str | Path,
) -> dict:
    dataset_path = Path(dataset_path)
    out_path = Path(out_path)
    source = pq.read_table(
        dataset_path,
        filters=[("split", "=", str(split)), ("role", "=", "active_best")],
        columns=[
            "episode_id", "seat", "step",
            "canonical_action_json", "effects_json",
        ],
    ).to_pylist()
    grouped: dict[tuple[int, int], list[dict]] = {}
    for row in source:
        key = (int(row["episode_id"]), int(row["seat"]))
        grouped.setdefault(key, []).append(row)

    records: list[dict] = []
    plan_counts = Counter()
    cardinality_counts = Counter()

    for (episode_id, seat), episode_rows in sorted(grouped.items()):
        ordered = sorted(episode_rows, key=lambda row: int(row["step"]))
        plan_sets = [
            derive_plan_set(
                {},
                json.loads(row["canonical_action_json"]),
                json.loads(row["effects_json"]),
            )
            for row in ordered
        ]
        for index, (row, active_set) in enumerate(zip(ordered, plan_sets)):
            active = tuple(sorted(active_set, key=PHASE_ORDER.__getitem__))
            has_next = index + 1 < len(plan_sets)
            next_active = (
                tuple(sorted(plan_sets[index + 1], key=PHASE_ORDER.__getitem__))
                if has_next else ()
            )
            current_mask = _plan_mask(active)
            next_mask = _plan_mask(next_active)
            for item in active:
                plan_counts[item.value] += 1
            cardinality_counts[len(active)] += 1
            records.append({
                "episode_id": episode_id,
                "seat": seat,
                "step": int(row["step"]),
                "active_plans": [item.value for item in active],
                "plan_mask": list(current_mask),
                "has_next": bool(has_next),
                "next_active_plans": [item.value for item in next_active],
                "next_plan_mask": list(next_mask),
                "transition_count": (
                    sum(bool(a) != bool(b) for a, b in zip(current_mask, next_mask))
                    if has_next else None
                ),
            })

    if not records:
        raise ValueError("plan sidecar would be empty")
    table = pa.Table.from_pylist(records)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp = out_path.with_suffix(out_path.suffix + ".tmp")
    pq.write_table(table, temp, compression="zstd")
    temp.replace(out_path)
    total = len(records)
    multi = sum(value for key, value in cardinality_counts.items() if key > 1)
    return {
        "rows": total,
        "plan_labels": [phase.value for phase in PLAN_PHASES],
        "plan_counts": {
            phase.value: int(plan_counts.get(phase.value, 0))
            for phase in PLAN_PHASES
        },
        "cardinality_counts": {
            str(key): int(value)
            for key, value in sorted(cardinality_counts.items())
        },
        "multi_intent_rows": int(multi),
        "multi_intent_fraction": float(multi) / float(total),
        "sha256": _sha256(out_path),
    }
