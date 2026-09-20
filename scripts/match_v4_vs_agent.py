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


def match_v4_vs_agent(
    opponent: str | Path,
    *,
    opponent_name: str = "opponent",
    seeds: list[int] | None = None,
    output_dir: str | Path | None = None,
) -> dict:
    opponent_path = _resolve_opponent(opponent)
    seeds = list(seeds or [20289000, 20289001, 20289002])
    records = []

    for seed in seeds:
        for seat in (0, 1):
            learner = V4HybridRolloutAgent()
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
        print(
            f"seed={row['seed']} seat={row['seat']} {result} "
            f"money={row['money']} opponent={row['opponent_money']} "
            f"margin={row['margin']:+d}"
        )
    print(
        f"TOTAL games={summary['games']} wins={summary['wins']} "
        f"losses={summary['losses']} ties={summary['ties']} "
        f"win_rate={summary['win_rate']:.1%} "
        f"mean_money={summary['mean_money']:.1f} "
        f"mean_margin={summary['mean_margin']:+.1f} "
        f"runtime_ok={summary['runtime_ok']}"
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
    args = parser.parse_args()
    match_v4_vs_agent(
        args.opponent,
        opponent_name=args.opponent_name,
        seeds=_parse_seeds(args.seeds),
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
