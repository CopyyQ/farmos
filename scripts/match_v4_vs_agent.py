from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from kaggle_environments import make
from kaggrl.clock import resolve_clock
from rollout.v4_hybrid_agent import V4HybridRolloutAgent


def _parse_seeds(text: str) -> list[int]:
    seeds = sorted({int(x.strip()) for x in text.split(",") if x.strip()})
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def _resolve_opponent(path: str | Path) -> Path:
    value = Path(path).resolve()
    if value.is_dir():
        value = value / "main.py"
    if not value.is_file():
        raise FileNotFoundError(value)
    return value


def _audit_clock(env) -> dict:
    seats = []
    for seat in (0, 1):
        missing_raw_step = 0
        mismatches = []
        for index, frame in enumerate(env.steps):
            obs = dict(frame[seat].observation)
            missing_raw_step += int(obs.get("step") is None)
            clock = resolve_clock(obs, env.configuration)
            if clock.step != index:
                mismatches.append({
                    "frame": index,
                    "resolved_step": clock.step,
                    "day": clock.day,
                    "hour": clock.hour,
                })
        seats.append({
            "seat": seat,
            "frames": len(env.steps),
            "missing_raw_step": missing_raw_step,
            "resolved_first": resolve_clock(
                dict(env.steps[0][seat].observation), env.configuration
            ).step,
            "resolved_last": resolve_clock(
                dict(env.steps[-1][seat].observation), env.configuration
            ).step,
            "mismatch_count": len(mismatches),
            "mismatch_examples": mismatches[:5],
        })
    return {
        "ok": all(row["mismatch_count"] == 0 for row in seats),
        "seats": seats,
    }


def _strategy_audit(runtime) -> dict:
    if runtime is None:
        return {
            "enabled": False,
            "steps": 0,
            "step_sequence_ok": True,
        }
    rows = list(runtime.telemetry)
    steps = [int(row["step"]) for row in rows]
    phase_counts = {}
    phase_changed = {}
    route_counts = {}
    market_counts = {}
    for row in rows:
        phase = str(row["phase"])
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
        if row.get("changed"):
            phase_changed[phase] = phase_changed.get(phase, 0) + 1
        route = row.get("route_id")
        if route is not None:
            key = str(int(route))
            route_counts[key] = route_counts.get(key, 0) + 1
        mode = str(row.get("market_mode", "KEEP_ROUTE"))
        market_counts[mode] = market_counts.get(mode, 0) + 1
    return {
        "enabled": True,
        "steps": len(rows),
        "first_step": steps[0] if steps else None,
        "last_step": steps[-1] if steps else None,
        "step_sequence_ok": (
            steps == list(range(len(steps)))
        ),
        "changed_steps": sum(bool(row.get("changed")) for row in rows),
        "route_override_steps": sum(
            row.get("route_id") is not None for row in rows
        ),
        "phase_counts": phase_counts,
        "changed_by_phase": phase_changed,
        "route_counts": route_counts,
        "market_mode_counts": market_counts,
        "mean_route_gate": (
            statistics.fmean(float(row["route_gate"]) for row in rows)
            if rows else 0.0
        ),
    }


def match_v4_vs_agent(
    opponent: str | Path,
    *,
    opponent_name: str = "opponent",
    seeds: list[int] | None = None,
    output_dir: str | Path | None = None,
    option_checkpoint: str | Path | None = None,
    min_route_probability: float = 0.55,
    min_market_probability: float = 0.65,
    allowed_market_modes=None,
    allow_route_switch: bool = True,
    liquidation_max_remaining_steps: int | None = None,
    enable_market_race_ordering: bool = True,
) -> dict:
    opponent_path = _resolve_opponent(opponent)
    seeds = list(seeds or [20289000, 20289001, 20289002])
    records = []
    checkpoint_path = None
    runtime_class = None
    runtime_backend = None
    if option_checkpoint is not None:
        checkpoint_path = Path(option_checkpoint).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        if checkpoint_path.suffix.lower() == ".npz":
            from rollout.v4_option_numpy_adapter import V4NumpyOptionAdapter
            runtime_class = V4NumpyOptionAdapter
            runtime_backend = "numpy"
        else:
            from kaggrl.v4_option_runtime import TorchV4OptionRuntime
            runtime_class = TorchV4OptionRuntime
            runtime_backend = "torch"

    for seed in seeds:
        for seat in (0, 1):
            if runtime_class is None:
                runtime = None
            else:
                common = dict(
                    min_route_probability=min_route_probability,
                    min_market_probability=min_market_probability,
                    allowed_market_modes=allowed_market_modes,
                    allow_route_switch=allow_route_switch,
                    liquidation_max_remaining_steps=(
                        liquidation_max_remaining_steps
                    ),
                )
                runtime = (
                    runtime_class(
                        checkpoint_path,
                        device="cpu",
                        **common,
                    )
                    if runtime_backend == "torch"
                    else runtime_class(checkpoint_path, **common)
                )
            learner = V4HybridRolloutAgent(
                option_policy=runtime,
                min_option_confidence=0.0,
                enable_market_race_ordering=enable_market_race_ordering,
            )
            agents = (
                [learner, str(opponent_path)]
                if seat == 0
                else [str(opponent_path), learner]
            )
            env = make(
                "kaggriculture",
                configuration={"seed": int(seed), "episodeSteps": 720},
                debug=False,
            )
            env.run(agents)
            clock_audit = _audit_clock(env)
            final = env.steps[-1]
            statuses = [str(agent.status) for agent in final]
            observation = final[0].observation
            farms = observation.farms
            own = int(farms[seat].money)
            rival = int(farms[1 - seat].money)
            records.append({
                "seed": int(seed),
                "seat": int(seat),
                "money": own,
                "opponent_money": rival,
                "margin": own - rival,
                "statuses": statuses,
                "clock_audit": clock_audit,
                "strategy_audit": _strategy_audit(runtime),
            })

    margins = [row["margin"] for row in records]
    summary = {
        "games": len(records),
        "wins": sum(value > 0 for value in margins),
        "losses": sum(value < 0 for value in margins),
        "ties": sum(value == 0 for value in margins),
        "win_rate": sum(value > 0 for value in margins) / max(1, len(records)),
        "mean_money": statistics.fmean(row["money"] for row in records),
        "mean_opponent_money": statistics.fmean(
            row["opponent_money"] for row in records
        ),
        "mean_margin": statistics.fmean(margins),
        "runtime_ok": all(
            row["statuses"] == ["DONE", "DONE"] for row in records
        ),
        "clock_ok": all(row["clock_audit"]["ok"] for row in records),
        "strategy_enabled": checkpoint_path is not None,
        "strategy_steps_ok": all(
            (
                not row["strategy_audit"]["enabled"]
                or (
                    row["strategy_audit"]["step_sequence_ok"]
                    and row["strategy_audit"]["steps"] == 719
                )
            )
            for row in records
        ),
    }

    if output_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output = ROOT / "runs" / f"v4_macro_vs_{opponent_name}_{stamp}"
    else:
        output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "match_report.json"
    report_path.write_text(
        json.dumps(
            {
                "opponent": opponent_name,
                "opponent_path": str(opponent_path),
                "option_checkpoint": (
                    None if checkpoint_path is None
                    else str(checkpoint_path)
                ),
                "market_race_ordering": bool(
                    enable_market_race_ordering
                ),
                "seeds": seeds,
                "summary": summary,
                "games": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"=== FARMOS V4 MACRO vs {opponent_name.upper()} ===")
    for row in records:
        result = (
            "WIN" if row["margin"] > 0
            else ("LOSS" if row["margin"] < 0 else "TIE")
        )
        missing = [
            item["missing_raw_step"]
            for item in row["clock_audit"]["seats"]
        ]
        print(
            f"seed={row['seed']} seat={row['seat']} {result} "
            f"money={row['money']} opponent={row['opponent_money']} "
            f"margin={row['margin']:+d} "
            f"clock_ok={row['clock_audit']['ok']} "
            f"raw_step_missing={missing} "
            f"strategy_changed={row['strategy_audit'].get('changed_steps', 0)}"
        )
    print(
        f"TOTAL games={summary['games']} wins={summary['wins']} "
        f"losses={summary['losses']} ties={summary['ties']} "
        f"win_rate={summary['win_rate']:.1%} "
        f"mean_money={summary['mean_money']:.1f} "
        f"mean_margin={summary['mean_margin']:+.1f} "
        f"runtime_ok={summary['runtime_ok']} "
        f"clock_ok={summary['clock_ok']} "
        f"strategy_steps_ok={summary['strategy_steps_ok']}"
    )
    print(f"REPORT={report_path}")
    print(
        "FARMOS_V4_MATCH="
        + json.dumps(
            {
                "opponent": opponent_name,
                "report": str(report_path),
                **summary,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return {"summary": summary, "games": records, "report": str(report_path)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark macro-first FarmOS V4 against an agent."
    )
    parser.add_argument("--opponent", required=True)
    parser.add_argument("--opponent-name", default="opponent")
    parser.add_argument(
        "--seeds",
        default="20289000,20289001,20289002,20289003,20289004",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--option-checkpoint", default=None)
    parser.add_argument("--min-route-probability", type=float, default=0.55)
    parser.add_argument("--min-market-probability", type=float, default=0.65)
    parser.add_argument(
        "--allowed-market-modes",
        default="KEEP_ROUTE",
    )
    parser.add_argument("--disable-route-switch", action="store_true")
    parser.add_argument(
        "--disable-market-race-ordering",
        action="store_true",
    )
    parser.add_argument(
        "--liquidation-max-remaining-steps",
        type=int,
        default=None,
    )
    args = parser.parse_args()
    match_v4_vs_agent(
        args.opponent,
        opponent_name=args.opponent_name,
        seeds=_parse_seeds(args.seeds),
        output_dir=args.output_dir,
        option_checkpoint=args.option_checkpoint,
        min_route_probability=args.min_route_probability,
        min_market_probability=args.min_market_probability,
        allowed_market_modes=tuple(
            value.strip()
            for value in args.allowed_market_modes.split(",")
            if value.strip()
        ),
        allow_route_switch=not args.disable_route_switch,
        liquidation_max_remaining_steps=(
            args.liquidation_max_remaining_steps
        ),
        enable_market_race_ordering=(
            not args.disable_market_race_ordering
        ),
    )


if __name__ == "__main__":
    main()
