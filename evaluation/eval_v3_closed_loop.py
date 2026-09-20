from __future__ import annotations

import json
import math
import os
import statistics
import subprocess
import sys
from collections import Counter

import numpy as np
from pathlib import Path
from typing import Any, Iterable

from kaggle_environments import make

from kaggrl.constants import UNIT_OPS
from kaggrl.v2_ledger import MARKET_OPS
from rollout.v3_agent_numpy import V3NumpyRolloutAgent
from evaluation.eval_v2_closed_loop import (
    GameSpec,
    _callback_errors,
    _effect_telemetry,
    _final_money,
    _money_trajectory,
    _own_farm,
    _resolve_named_agent,
    _schema_errors,
    _sha256,
    artifact_name,
    build_game_specs,
    summarize_closed_loop,
)

DEFAULT_GATE_THRESHOLDS = {
    "max_effectless_streak": 600,
    "farmer_op_gap_max": 0.25,
    "hands_op_gap_max": 0.25,
    "market_op_gap_max": 0.25,
    "farmer_pass_fraction_max": 0.80,
    "hands_pass_fraction_max": 0.80,
    "stop_queue_fraction_max": 0.95,
    "stop_queue_expert_gap": 0.20,
}


def build_v3_game_specs(seeds: Iterable[int], opponents: Iterable[str]) -> list[GameSpec]:
    return build_game_specs(seeds, opponents)


def _agent_class_for_model(model_path: Path):
    path = Path(model_path)
    with np.load(path, allow_pickle=False) as archive:
        if "format_version" not in archive.files:
            raise ValueError("NumPy policy archive has no format_version")
        format_version = int(archive["format_version"])
    if format_version == 3:
        return V3NumpyRolloutAgent
    if format_version == 4:
        from rollout.v3_2_agent_numpy import V32NumpyRolloutAgent
        return V32NumpyRolloutAgent
    if format_version == 5:
        from rollout.v3_3_agent_numpy import V33NumpyRolloutAgent
        return V33NumpyRolloutAgent
    raise ValueError(f"unsupported V3 NumPy format version: {format_version}")


def _market_op(slot: dict[str, Any]) -> str:
    kind = str(slot.get("kind", "ORDER"))
    return kind if kind in {"STOP_QUEUE", "NOP_SLOT"} else str(slot.get("op", "NOP_SLOT"))


def _action_histograms(telemetry):
    hist = {
        "farmer": Counter({op: 0 for op in UNIT_OPS}),
        "hands": Counter({op: 0 for op in UNIT_OPS}),
        "market": Counter({op: 0 for op in MARKET_OPS}),
    }
    for row in telemetry:
        action = row.get("canonical_action") or {}
        hist["farmer"][str((action.get("farmer") or {}).get("op", "PASS"))] += 1
        for command in action.get("hands") or []:
            hist["hands"][str(command.get("op", "PASS"))] += 1
        for slot in action.get("market") or []:
            hist["market"][_market_op(slot)] += 1
    return {
        domain: {key: int(value) for key, value in counter.items()}
        for domain, counter in hist.items()
    }


def _family_ok(families: dict[str, Any], *names: str) -> bool:
    return any(int(families.get(name, 0) or 0) > 0 for name in names)


def evaluate_short_smoke(records, parity_report) -> dict[str, Any]:
    failures = []
    if not bool((parity_report or {}).get("passed", False)):
        failures.append("parity_mismatch")
    seats = set()
    for row in list(records or []):
        seat = int(row.get("learner_seat", -1)); seats.add(seat)
        prefix = f"seat{seat}"
        if row.get("schema_valid") is not True:
            failures.append(f"{prefix}_invalid_schema")
        if row.get("timeout") is not False:
            failures.append(f"{prefix}_timeout")
        if row.get("torch_import_free") is not True:
            failures.append(f"{prefix}_torch_import")
        hist = row.get("action_histograms") or {}
        farmer = hist.get("farmer") or {}
        hands = hist.get("hands") or {}
        farmer_nonpass = sum(int(v) for k, v in farmer.items() if k != "PASS")
        hand_total = sum(int(v) for v in hands.values())
        hand_nonpass = sum(int(v) for k, v in hands.items() if k != "PASS")
        if farmer_nonpass <= 0:
            failures.append(f"{prefix}_farmer_all_pass")
        if hand_total > 0 and hand_nonpass <= 0:
            failures.append(f"{prefix}_hands_all_pass")
        families = row.get("effective_family_counts") or {}
        if not _family_ok(families, "movement"):
            failures.append(f"{prefix}_missing_movement")
        if not _family_ok(families, "acquisition"):
            failures.append(f"{prefix}_missing_acquisition")
    if len(records or []) > 1 and seats != {0, 1}:
        failures.append("missing_symmetric_learner_seats")
    failures = sorted(set(failures))
    return {"passed": not failures, "failures": failures, "learner_seats": sorted(seats)}


def evaluate_practical_gate(matrix, offline_report, thresholds=None) -> dict[str, Any]:
    limits = dict(DEFAULT_GATE_THRESHOLDS)
    if thresholds:
        limits.update(thresholds)
    records = list((matrix or {}).get("records") or [])
    failures: list[str] = []
    seats = {int(row.get("learner_seat", -1)) for row in records}
    if seats != {0, 1}:
        failures.append("missing_symmetric_learner_seats")
    for row in records:
        seat = int(row.get("learner_seat", -1))
        prefix = f"seat{seat}"
        if row.get("steps", 720) != 720 or row.get("statuses") != ["DONE", "DONE"]:
            failures.append(f"{prefix}_not_full_done")
        if row.get("finite") is not True:
            failures.append(f"{prefix}_non_finite")
        if row.get("schema_valid") is not True:
            failures.append(f"{prefix}_invalid_schema")
        if row.get("timeout") is not False:
            failures.append(f"{prefix}_timeout")
        if row.get("torch_import_free") is not True:
            failures.append(f"{prefix}_torch_import")
        if int(row.get("longest_effectless_streak", 10**9)) >= int(limits["max_effectless_streak"]):
            failures.append(f"{prefix}_effectless_streak")
        families = row.get("effective_family_counts") or {}
        if not _family_ok(families, "movement"):
            failures.append(f"{prefix}_missing_movement")
        if not _family_ok(families, "acquisition"):
            failures.append(f"{prefix}_missing_acquisition")
        if not _family_ok(families, "production", "service"):
            failures.append(f"{prefix}_missing_production_service")
        # Kaggriculture automatically drops all farmer/hand inventory
        # into the shed at end-of-day. Explicit DROP is therefore optional
        # and must not be a practical-playability gate.
        if not _family_ok(families, "sale"):
            failures.append(f"{prefix}_missing_sale")

    if offline_report.get("finite") is not True:
        failures.append("offline_non_finite")
    gaps = offline_report.get("gaps") or {}
    for key, threshold_key in (
        ("farmer_op", "farmer_op_gap_max"),
        ("hands_op", "hands_op_gap_max"),
        ("market_op", "market_op_gap_max"),
    ):
        if float(gaps.get(key, float("inf"))) > float(limits[threshold_key]):
            failures.append(f"offline_{key}_gap")
    free = offline_report.get("free_history") or {}
    fractions = free.get("op_fractions") or {}
    if fractions:
        if float(fractions.get("farmer_pass", 0.0)) > float(limits["farmer_pass_fraction_max"]):
            failures.append("offline_farmer_pass_collapse")
        if float(fractions.get("hands_pass", 0.0)) > float(limits["hands_pass_fraction_max"]):
            failures.append("offline_hands_pass_collapse")
        stop = float(fractions.get("market_stop_queue", 0.0))
        expert_stop = float(fractions.get("expert_market_stop_queue", 0.0))
        if (stop > float(limits["stop_queue_fraction_max"]) and
                stop - expert_stop > float(limits["stop_queue_expert_gap"])):
            failures.append("offline_stop_queue_collapse")
    failures = sorted(set(failures))
    return {
        "passed": not failures,
        "failures": failures,
        "thresholds": limits,
        "records_checked": len(records),
        "learner_seats": sorted(seats),
    }


def _torch_free_preflight(model_path: Path) -> bool:
    root = Path(__file__).resolve().parents[1]
    code = r'''
import builtins, sys
_real_import = builtins.__import__
def _guard(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise RuntimeError("torch import blocked")
    return _real_import(name, *args, **kwargs)
builtins.__import__ = _guard
import numpy as np
with np.load(sys.argv[1], allow_pickle=False) as archive:
    format_version = int(archive["format_version"])
if format_version == 5:
    from rollout.v3_3_agent_numpy import V33NumpyRolloutAgent
    agent_cls = V33NumpyRolloutAgent
elif format_version == 4:
    from rollout.v3_2_agent_numpy import V32NumpyRolloutAgent
    agent_cls = V32NumpyRolloutAgent
elif format_version == 3:
    from rollout.v3_agent_numpy import V3NumpyRolloutAgent
    agent_cls = V3NumpyRolloutAgent
else:
    raise RuntimeError(f"unsupported numpy format: {format_version}")
agent = agent_cls(sys.argv[1], seed=1, deterministic=True)
tiles = [[None for _ in range(10)] for _ in range(10)]
obs = {"player":0,"step":0,"day":0,"hour":0,"farms":[
{"money":3000,"farmer":[4,4],"hands":[],"hires_today":0,"unlocked_quadrants":["NW"],"tiles":tiles},
{"money":3000,"farmer":[8,8],"hands":[],"hires_today":0,"unlocked_quadrants":["NW"],"tiles":tiles}],
"private":{"shed":{},"seeds":{},"inventories":[{}]},
"market":{"inventory":{"WHEAT":10000},"prices":{"WHEAT":25}},"town":{"unlocked_shops":[]}}
action = agent(obs)
assert set(action) == {"farmer","hands","market"}
'''
    env = dict(os.environ)
    pythonpath = [str(root / "src"), str(root)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    result = subprocess.run(
        [sys.executable, "-c", code, str(model_path)],
        cwd=str(root), env=env, capture_output=True, text=True, timeout=30,
    )
    return result.returncode == 0


def _run_single_v3_game(model_path: Path, spec: GameSpec, out_path: Path,
                        torch_import_free: bool, deterministic: bool = True):
    if out_path.exists():
        raise FileExistsError(out_path)
    agent_class = _agent_class_for_model(model_path)
    learner = agent_class(
        model_path, seed=spec.seed + 101, deterministic=deterministic,
    )
    opponent, opponent_name = _resolve_named_agent(spec.opponent)
    env = make(
        "kaggriculture",
        configuration={"seed": spec.seed, "episodeSteps": 720},
        debug=False,
    )
    agents = [learner, opponent] if spec.learner_seat == 0 else [opponent, learner]
    env.run(agents)
    if not env.steps:
        raise RuntimeError("Kaggriculture produced no engine steps")
    callback_errors = _callback_errors(env)
    schema_errors = _schema_errors(env, spec.learner_seat)
    final = env.steps[-1]
    statuses = [str(value.status) for value in final]
    timeout = any("TIMEOUT" in status.upper() for status in statuses) or any(
        "timeout" in message.lower() for message in callback_errors
    )
    schema_valid = not callback_errors and not schema_errors
    rewards = [float(value.reward) for value in final]
    money_values, day_snapshots, max_hands = _money_trajectory(
        env, spec.learner_seat,
    )
    family_counts, effect_rows, longest = _effect_telemetry(
        env, spec.learner_seat,
    )
    own_money, rival_money = _final_money(env, spec.learner_seat)
    first_obs = env.steps[0][spec.learner_seat].observation
    last_obs = env.steps[-1][spec.learner_seat].observation
    initial_land = len(
        _own_farm(first_obs, spec.learner_seat).get("unlocked_quadrants", []) or []
    )
    final_land = len(
        _own_farm(last_obs, spec.learner_seat).get("unlocked_quadrants", []) or []
    )
    finite = all(
        math.isfinite(float(row[key]))
        for row in learner.telemetry
        for key in ("logp", "terminal_money", "terminal_margin")
    )
    histograms = _action_histograms(learner.telemetry)
    max_temporal_length = max(
        (int(row["temporal_valid_length"]) for row in learner.telemetry),
        default=0,
    )
    record = {
        "label": "candidate_v3",
        "seed": int(spec.seed),
        "learner_seat": int(spec.learner_seat),
        "opponent": opponent_name,
        "steps": len(env.steps),
        "decisions": len(learner.telemetry),
        "statuses": statuses,
        "rewards": rewards,
        "finite": bool(finite),
        "schema_valid": bool(schema_valid),
        "timeout": bool(timeout),
        "torch_import_free": bool(torch_import_free),
        "callback_error_count": len(callback_errors),
        "callback_errors": callback_errors,
        "schema_errors": schema_errors,
        "final_money": own_money,
        "rival_final_money": rival_money,
        "margin": own_money - rival_money,
        "min_money": min(money_values),
        "median_money": float(statistics.median(money_values)),
        "max_money": max(money_values),
        "max_hands": max_hands,
        "land_unlocks": max(0, final_land - initial_land),
        "effective_family_counts": family_counts,
        "longest_effectless_streak": int(longest),
        "day_snapshots": day_snapshots,
        "effect_rows": effect_rows,
        "action_histograms": histograms,
        "reset_count": int(learner.reset_count),
        "max_temporal_valid_length": int(max_temporal_length),
        "model_sha256": _sha256(model_path),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    record["artifact"] = str(out_path)
    record["artifact_sha256"] = _sha256(out_path)
    return record


def summarize_v3_closed_loop(records):
    games = list(records)
    base = summarize_closed_loop(games)
    hist = {
        "farmer": Counter({op: 0 for op in UNIT_OPS}),
        "hands": Counter({op: 0 for op in UNIT_OPS}),
        "market": Counter({op: 0 for op in MARKET_OPS}),
    }
    for game in games:
        for domain, values in (game.get("action_histograms") or {}).items():
            hist[domain].update(values)
    base.update({
        "all_torch_import_free": all(game.get("torch_import_free") is True for game in games),
        "all_full_720": all(int(game.get("steps", 0)) == 720 for game in games),
        "learner_seats": sorted({int(game.get("learner_seat", -1)) for game in games}),
        "action_histograms": {
            domain: {key: int(value) for key, value in counter.items()}
            for domain, counter in hist.items()
        },
    })
    return base


def run_v3_game_matrix(model_path, specs: Iterable[GameSpec], output_dir):
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "candidate_v3_summary.json"
    if summary_path.exists():
        raise FileExistsError(summary_path)
    torch_free = _torch_free_preflight(model_path)
    if not torch_free:
        raise RuntimeError("v3 NumPy runtime failed Torch-free subprocess preflight")
    records = []
    for spec in sorted(specs, key=lambda row: (row.seed, row.opponent, row.learner_seat)):
        path = output / artifact_name(spec, "candidate_v3")
        records.append(_run_single_v3_game(
            model_path, spec, path, torch_import_free=torch_free,
            deterministic=True,
        ))
    result = {
        "kind": "candidate_v3",
        "model_sha256": _sha256(model_path),
        "torch_import_free": bool(torch_free),
        "records": records,
        "summary": summarize_v3_closed_loop(records),
    }
    summary_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return result
