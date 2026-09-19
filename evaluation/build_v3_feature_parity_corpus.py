from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq

from evaluation.eval_v3_parity import (
    capture_feature_snapshot,
    capture_observation_feature_snapshot,
)
from kaggrl.v2_tensorize import _row_state


def collect_expert_snapshots(
    dataset_path: str | Path,
    *,
    split: str = "train",
    steps: set[int] | None = None,
    max_rows: int = 1000,
) -> list[dict[str, Any]]:
    wanted = None if steps is None else {int(step) for step in steps}
    table = pq.read_table(
        Path(dataset_path),
        filters=[("split", "=", str(split)), ("role", "=", "active_best")],
        columns=[
            "episode_id", "seat", "step", "state_zlib",
            "canonical_action_json", "effects_json",
        ],
    )
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in table.to_pylist():
        grouped[(int(row["episode_id"]), int(row["seat"]))].append(row)

    out: list[dict[str, Any]] = []
    for (episode_id, seat), rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: int(row["step"]))
        previous_action: dict[str, Any] = {}
        previous_effect: dict[str, Any] = {}
        for row in ordered:
            step = int(row["step"])
            if wanted is None or step in wanted:
                out.append({
                    "source": "expert",
                    "episode_id": episode_id,
                    "seat": seat,
                    "step": step,
                    "previous_action_present": bool(previous_action),
                    "previous_effect_present": bool(previous_effect),
                    "feature": capture_feature_snapshot(
                        _row_state(row), previous_action, previous_effect,
                    ),
                })
                if len(out) >= int(max_rows):
                    return out
            previous_action = json.loads(row["canonical_action_json"])
            previous_effect = json.loads(row["effects_json"])
    return out


def collect_live_snapshots(
    model_path: str | Path,
    seeds: Iterable[int],
    *,
    steps: set[int],
    opponent: str = "starter",
    episode_steps: int | None = None,
) -> list[dict[str, Any]]:
    from kaggle_environments import make
    from evaluation.eval_v2_closed_loop import _resolve_named_agent
    from rollout.v3_agent_numpy import V3NumpyRolloutAgent

    wanted = {int(step) for step in steps}
    if not wanted:
        raise ValueError("live snapshot steps must not be empty")
    limit = int(episode_steps or (max(wanted) + 2))
    out: list[dict[str, Any]] = []
    for seed in [int(value) for value in seeds]:
        for seat in (0, 1):
            learner = V3NumpyRolloutAgent(
                model_path,
                seed=seed + 101,
                deterministic=True,
                capture_decision_trace=True,
            )
            rival, _ = _resolve_named_agent(opponent)
            env = make(
                "kaggriculture",
                configuration={"seed": seed, "episodeSteps": limit},
                debug=False,
            )
            agents = [learner, rival] if seat == 0 else [rival, learner]
            env.run(agents)
            for fixture in learner.diagnostic_fixtures:
                step = int(fixture["step"])
                if step not in wanted:
                    continue
                direct = capture_observation_feature_snapshot(
                    fixture["observation"],
                    fixture.get("previous_action") or {},
                    fixture.get("previous_effect") or {},
                )
                structured = capture_feature_snapshot(
                    fixture["structured_state"],
                    fixture.get("previous_action") or {},
                    fixture.get("previous_effect") or {},
                )
                out.append({
                    "source": "live",
                    "seed": seed,
                    "seat": seat,
                    "step": step,
                    "raw_structured_exact": (
                        direct["feature_sha256"] == structured["feature_sha256"]
                    ),
                    "feature": structured,
                })
    return out


def summarize_feature_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty feature snapshot set")
    names = sorted(rows[0]["feature"]["stats"])
    summary: dict[str, Any] = {"rows": len(rows), "features": {}}
    for name in names:
        values = [float(row["feature"]["stats"][name]["mean"]) for row in rows]
        summary["features"][name] = {
            "mean_of_means": sum(values) / len(values),
            "min_mean": min(values),
            "max_mean": max(values),
            "shape": rows[0]["feature"]["shapes"][name],
            "dtype": rows[0]["feature"]["dtypes"][name],
        }
    return summary


_DYNAMIC_UNIT_FEATURES = {
    "own_units",
    "previous_unit_actions",
    "previous_unit_effects",
}


def _feature_schema_compatible(
    name: str,
    left_shape: list[int],
    right_shape: list[int],
    left_dtype: str,
    right_dtype: str,
) -> bool:
    if left_dtype != right_dtype:
        return False
    if name in _DYNAMIC_UNIT_FEATURES:
        return (
            len(left_shape) == len(right_shape)
            and list(left_shape[1:]) == list(right_shape[1:])
        )
    return list(left_shape) == list(right_shape)


def build_distribution_comparison(
    expert_rows: list[dict[str, Any]],
    live_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    expert = summarize_feature_distribution(expert_rows)
    live = summarize_feature_distribution(live_rows)
    if set(expert["features"]) != set(live["features"]):
        return {"passed": False, "reason": "feature_names_mismatch"}
    mismatched_schema = []
    drift = {}
    for name in expert["features"]:
        left = expert["features"][name]
        right = live["features"][name]
        if not _feature_schema_compatible(
            name,
            left["shape"],
            right["shape"],
            left["dtype"],
            right["dtype"],
        ):
            mismatched_schema.append(name)
        drift[name] = {
            "expert_mean": left["mean_of_means"],
            "live_mean": right["mean_of_means"],
            "abs_mean_gap": abs(left["mean_of_means"] - right["mean_of_means"]),
        }
    raw_exact = all(
        bool(row.get("raw_structured_exact", True)) for row in live_rows
    )
    return {
        "passed": not mismatched_schema and raw_exact,
        "expert_rows": expert["rows"],
        "live_rows": live["rows"],
        "raw_structured_exact": raw_exact,
        "schema_mismatches": mismatched_schema,
        "feature_mean_drift": drift,
    }
