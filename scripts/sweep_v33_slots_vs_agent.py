from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.match_v33_vs_agent import (
    _load_checkpoint,
    _parse_seeds,
    match_checkpoint_vs_agent,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep all strategy slots of a V3.2/V3.3 checkpoint against one agent."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--opponent", required=True)
    parser.add_argument("--opponent-name", default=None)
    parser.add_argument(
        "--seeds",
        default="20271005",
        help="comma-separated unseen seeds; every slot plays both seats",
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    opponent = Path(args.opponent).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not opponent.is_file():
        raise FileNotFoundError(opponent)

    payload, _, slot_to_team, _ = _load_checkpoint(checkpoint)
    seeds = _parse_seeds(args.seeds)
    opponent_name = args.opponent_name or opponent.parent.name or opponent.stem

    root = (
        Path(args.output_dir)
        if args.output_dir
        else checkpoint.parent / f"sweep_vs_{opponent_name}_epoch{payload.get('epoch', 'x')}"
    )
    root.mkdir(parents=True, exist_ok=False)

    rows = []
    for slot, team_id in enumerate(slot_to_team):
        report = match_checkpoint_vs_agent(
            checkpoint,
            opponent,
            opponent_name=opponent_name,
            slot=slot,
            seeds=seeds,
            output_dir=root / f"slot_{slot:02d}",
        )
        summary = report["summary"]
        rows.append({
            "slot": int(slot),
            "team_id": int(team_id),
            "games": int(summary["games"]),
            "wins": int(summary["wins"]),
            "losses": int(summary["losses"]),
            "ties": int(summary["ties"]),
            "win_rate": float(summary["win_rate"]),
            "mean_margin": float(summary["mean_margin"]),
            "median_margin": float(summary["median_margin"]),
            "mean_final_money": float(summary["mean_final_money"]),
            "mean_opponent_money": float(summary["mean_opponent_money"]),
            "runtime_ok": bool(summary["runtime_ok"]),
        })

    # Screen ordering is purely diagnostic: prefer actual wins, then money/margin.
    ordered = sorted(
        rows,
        key=lambda row: (
            row["wins"],
            row["mean_final_money"],
            row["mean_margin"],
        ),
        reverse=True,
    )
    result = {
        "checkpoint": str(checkpoint),
        "opponent": str(opponent),
        "opponent_name": opponent_name,
        "seeds": seeds,
        "slots": rows,
        "screen_order": [int(row["slot"]) for row in ordered],
        "best_screen_slot": int(ordered[0]["slot"]) if ordered else None,
        "best_screen_team_id": int(ordered[0]["team_id"]) if ordered else None,
        "all_runtime_ok": all(row["runtime_ok"] for row in rows),
        "all_near_starting_cash": all(
            2500.0 <= row["mean_final_money"] <= 3500.0 for row in rows
        ),
        "mean_slot_money": (
            float(statistics.fmean(row["mean_final_money"] for row in rows))
            if rows else 0.0
        ),
    }
    report_path = root / "slot_sweep_report.json"
    report_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("\n=== FARMOS STRATEGY SLOT SWEEP ===")
    for row in ordered:
        print(
            f"slot={row['slot']:02d} team={row['team_id']} "
            f"wins={row['wins']}/{row['games']} "
            f"money={row['mean_final_money']:.1f} "
            f"opp={row['mean_opponent_money']:.1f} "
            f"margin={row['mean_margin']:+.1f} "
            f"runtime_ok={row['runtime_ok']}"
        )
    print(
        "SLOT_SWEEP="
        + json.dumps({
            "report": str(report_path),
            "best_screen_slot": result["best_screen_slot"],
            "best_screen_team_id": result["best_screen_team_id"],
            "all_runtime_ok": result["all_runtime_ok"],
            "all_near_starting_cash": result["all_near_starting_cash"],
            "mean_slot_money": result["mean_slot_money"],
        }, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
