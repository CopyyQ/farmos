from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from kaggle_environments import make

from kaggrl.v2_action_schema import audit_replay_actions
from kaggrl.v2_effect_tracker import EffectTracker
from rollout.v2_agent_numpy import V2NumpyRolloutAgent

ROOT = Path(__file__).resolve().parents[1]
V17 = ROOT.parent / "submission_ready_20260916_v17_27w_fixed" / "main.py"
V45 = ROOT.parent / "source_public" / "extracted_v45" / "main.py"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

@dataclass(frozen=True, order=True)
class GameSpec:
    seed: int
    learner_seat: int
    opponent: str

    def __post_init__(self):
        if int(self.learner_seat) not in (0, 1):
            raise ValueError("learner_seat must be 0 or 1")
        if not str(self.opponent):
            raise ValueError("opponent must not be empty")


def build_game_specs(seeds: Iterable[int], opponents: Iterable[str]) -> list[GameSpec]:
    specs = [
        GameSpec(int(seed), seat, str(opponent))
        for seed in sorted({int(x) for x in seeds})
        for opponent in sorted({str(x) for x in opponents})
        for seat in (0, 1)
    ]
    return sorted(specs, key=lambda row: (row.seed, row.opponent, row.learner_seat))


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))


def artifact_name(spec: GameSpec, label: str) -> str:
    return f"{_safe_name(label)}_vs_{_safe_name(spec.opponent)}_seed{spec.seed}_seat{spec.learner_seat}.json"

def _callback_errors(env) -> list[str]:
    raw = getattr(env, "logs", None)
    if not raw:
        return []
    stack = [raw]
    errors: list[str] = []
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, (list, tuple)):
            stack.extend(value)
        elif value:
            text = str(value)
            lowered = text.lower()
            if any(marker in lowered for marker in ("traceback", "exception", "error", "invalid action", "timeout")):
                errors.append(text)
    return errors

def _schema_errors(env, seat: int) -> list[dict[str, Any]]:
    replay = {"steps": []}
    for step in env.steps:
        replay["steps"].append([
            {"action": agent.action, "observation": agent.observation}
            for agent in step
        ])
    return [
        {"step": item.step, "seat": item.seat, "reason": item.reason}
        for item in audit_replay_actions(replay, seat)
    ]

def summarize_closed_loop(game_records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    games = list(game_records)
    if not games:
        raise ValueError("closed-loop summary requires at least one game")
    family_counts = Counter()
    for game in games:
        family_counts.update(game.get("effective_family_counts") or {})
    final_money = [float(game["final_money"]) for game in games]
    margins = [float(game["margin"]) for game in games]
    return {
        "games": len(games),
        "median_final_money": float(statistics.median(final_money)),
        "mean_final_money": float(statistics.fmean(final_money)),
        "median_margin": float(statistics.median(margins)),
        "effective_family_counts": dict(sorted(family_counts.items())),
        "land_unlocks": sum(int(game.get("land_unlocks", 0)) for game in games),
        "max_hands": max(int(game.get("max_hands", 0)) for game in games),
        "max_effectless_streak": max(int(game.get("longest_effectless_streak", 0)) for game in games),
        "all_done": all(game.get("statuses") == ["DONE", "DONE"] for game in games),
        "all_finite": all(game.get("finite") is True for game in games),
        "all_schema_valid": all(game.get("schema_valid") is True for game in games),
        "all_no_timeout": all(game.get("timeout") is False for game in games),
        "seeds": sorted({int(game["seed"]) for game in games}),
        "opponents": sorted({str(game["opponent"]) for game in games}),
        "learner_seats": sorted({int(game["learner_seat"]) for game in games}),
    }

def _resolve_named_agent(value: str):
    if value == "starter":
        return "starter", "starter"
    if value == "v17":
        if not V17.is_file(): raise FileNotFoundError(V17)
        return str(V17), "v17"
    if value == "v45":
        if not V45.is_file(): raise FileNotFoundError(V45)
        return str(V45), "v45"
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path), path.stem


def _own_farm(observation, seat: int):
    player = int(observation.get("player", seat))
    farms = observation.get("farms") or []
    return farms[player]


def _money_trajectory(env, seat: int):
    values = []
    snapshots = {}
    max_hands = 0
    for step_index, step in enumerate(env.steps):
        obs = step[seat].observation
        farm = _own_farm(obs, seat)
        money = int(farm.get("money", 0) or 0)
        hands = len(farm.get("hands", []) or [])
        lands = len(farm.get("unlocked_quadrants", []) or [])
        day = int(obs.get("day", step_index // 24))
        values.append(money); max_hands = max(max_hands, hands)
        snapshots[str(day)] = {"step": step_index, "money": money,
                               "hands": hands, "land_count": lands}
    return values, snapshots, max_hands

def _effect_telemetry(env, seat: int):
    tracker = EffectTracker()
    family_counts = Counter()
    rows = []
    streak = longest = 0
    for index in range(len(env.steps) - 1):
        before = env.steps[index][seat].observation
        following = env.steps[index + 1][seat]
        action = following.action or {}
        if not isinstance(action, dict):
            action = dict(action)
        record = tracker.observe(before, action, following.observation)
        family_counts.update(record.effective_families)
        if record.effective:
            streak = 0
        else:
            streak += 1; longest = max(longest, streak)
        rows.append({
            "step": index, "effective": record.effective,
            "effective_families": list(record.effective_families),
            "confirmed_count": record.confirmed_count,
            "failed_count": record.failed_count,
            "unconfirmed_count": record.unconfirmed_count,
            "dawn_hand_expiration": record.dawn_hand_expiration,
            "money_delta": int(record.transition.money_delta),
            "hand_count_delta": int(record.transition.hand_count_delta),
        })
    return dict(sorted(family_counts.items())), rows, longest


def _final_money(env, seat: int):
    obs = env.steps[-1][seat].observation
    player = int(obs.get("player", seat))
    farms = obs.get("farms") or []
    own = int(farms[player].get("money", 0) or 0)
    rival = int(farms[1 - player].get("money", 0) or 0)
    return own, rival

def _run_single_game(learner_source, spec: GameSpec, out_path: Path, label: str,
                     *, v2_candidate: bool, deterministic: bool = True):
    if v2_candidate:
        model_path = Path(learner_source)
        learner = V2NumpyRolloutAgent(
            model_path, seed=spec.seed + 101, deterministic=deterministic,
        )
        learner_name = label
        learner_sha = _sha256(model_path)
    else:
        learner, learner_name = _resolve_named_agent(str(learner_source))
        learner_sha = _sha256(Path(learner)) if learner != "starter" else None
    opponent, opponent_name = _resolve_named_agent(spec.opponent)
    env = make("kaggriculture", configuration={"seed": spec.seed, "episodeSteps": 720}, debug=False)
    agents = [learner, opponent] if spec.learner_seat == 0 else [opponent, learner]
    env.run(agents)
    callback_errors = _callback_errors(env)
    schema_errors = _schema_errors(env, spec.learner_seat)
    if len(env.steps) != 720:
        raise RuntimeError(f"expected 720 engine steps, got {len(env.steps)}")
    final = env.steps[-1]
    statuses = [str(value.status) for value in final]
    timeout = any("TIMEOUT" in status.upper() for status in statuses) or any(
        "timeout" in message.lower() for message in callback_errors
    )
    schema_valid = not callback_errors and not schema_errors
    if statuses != ["DONE", "DONE"]:
        raise RuntimeError(f"non-DONE final status: {statuses}")
    rewards = [float(value.reward) for value in final]
    if v2_candidate and len(learner.telemetry) != 719:
        raise RuntimeError(f"expected 719 learner decisions, got {len(learner.telemetry)}")
    money_values, day_snapshots, max_hands = _money_trajectory(env, spec.learner_seat)
    family_counts, effect_rows, longest = _effect_telemetry(env, spec.learner_seat)
    own_money, rival_money = _final_money(env, spec.learner_seat)
    first_obs = env.steps[0][spec.learner_seat].observation
    last_obs = env.steps[-1][spec.learner_seat].observation
    initial_land = len(_own_farm(first_obs, spec.learner_seat).get("unlocked_quadrants", []) or [])
    final_land = len(_own_farm(last_obs, spec.learner_seat).get("unlocked_quadrants", []) or [])
    finite = True
    if v2_candidate:
        finite = all(
            math.isfinite(float(row[key]))
            for row in learner.telemetry
            for key in ("logp", "terminal_money", "terminal_margin")
        )
    record = {
        "label": label, "seed": spec.seed, "learner_seat": spec.learner_seat,
        "opponent": opponent_name, "learner": learner_name,
        "steps": len(env.steps), "decisions": 719,
        "statuses": statuses, "rewards": rewards, "finite": bool(finite),
        "schema_valid": bool(schema_valid), "timeout": bool(timeout),
        "callback_error_count": len(callback_errors),
        "callback_errors": callback_errors, "schema_errors": schema_errors,
        "final_money": own_money, "rival_final_money": rival_money,
        "margin": own_money - rival_money,
        "min_money": min(money_values), "median_money": float(statistics.median(money_values)),
        "max_money": max(money_values), "max_hands": max_hands,
        "land_unlocks": max(0, final_land - initial_land),
        "effective_family_counts": family_counts,
        "longest_effectless_streak": longest,
        "day_snapshots": day_snapshots,
        "effect_rows": effect_rows,
        "model_sha256": learner_sha,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    record["artifact"] = str(out_path)
    record["artifact_sha256"] = _sha256(out_path)
    return record

def run_game_matrix(model_path, specs: Iterable[GameSpec], output_dir) -> dict[str, Any]:
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    records = []
    for spec in sorted(specs, key=lambda row: (row.seed, row.opponent, row.learner_seat)):
        path = output / artifact_name(spec, "candidate")
        records.append(_run_single_game(
            model_path, spec, path, "candidate", v2_candidate=True, deterministic=True,
        ))
    summary = summarize_closed_loop(records)
    result = {"kind": "candidate", "model_sha256": _sha256(Path(model_path)),
              "records": records, "summary": summary}
    (output / "candidate_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return result


def run_control_matrix(specs: Iterable[GameSpec], output_dir, control: str = "v17") -> dict[str, Any]:
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    records = []
    for spec in sorted(specs, key=lambda row: (row.seed, row.opponent, row.learner_seat)):
        path = output / artifact_name(spec, f"control_{control}")
        records.append(_run_single_game(
            control, spec, path, f"control_{control}", v2_candidate=False,
        ))
    summary = summarize_closed_loop(records)
    result = {"kind": "control", "control": control, "records": records, "summary": summary}
    (output / f"control_{_safe_name(control)}_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return result
