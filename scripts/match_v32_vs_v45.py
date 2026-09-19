from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import torch
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from kaggrl.v3_2_export import export_v3_2_numpy
from kaggrl.v3_2_model import TemporalIntentPolicyV32
from kaggrl.v3_2_schema import ARCHITECTURE_VERSION as V32_ARCH
from kaggrl.v3_3_export import export_v3_3_numpy
from kaggrl.v3_3_model import TemporalIntentPolicyV33
from kaggrl.v3_3_schema import ARCHITECTURE_VERSION as V33_ARCH

V45 = ROOT.parent / "source_public" / "extracted_v45" / "main.py"


def _load_match_checkpoint(checkpoint: Path):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    architecture = str(payload.get("architecture_version", ""))
    manifest = payload.get("strategy_manifest") or {}
    slot_to_team = tuple(
        int(value) for value in manifest.get("slot_to_team") or ()
    )
    if not slot_to_team:
        raise RuntimeError("checkpoint has no strategy_manifest.slot_to_team")

    if architecture == V32_ARCH:
        model = TemporalIntentPolicyV32(
            strategy_count=len(slot_to_team)
        ).eval()
        exporter = export_v3_2_numpy
    elif architecture == V33_ARCH:
        model = TemporalIntentPolicyV33(
            strategy_count=len(slot_to_team)
        ).eval()
        exporter = export_v3_3_numpy
    else:
        raise RuntimeError(
            "v45 match requires a V3.2/V3.3 checkpoint, "
            f"got {architecture!r}"
        )
    model.load_state_dict(payload["model_state"], strict=True)
    return payload, model, slot_to_team, exporter


def _parse_seeds(text: str) -> list[int]:
    seeds = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            seeds.append(int(item))
    if not seeds:
        raise ValueError("at least one seed is required")
    return sorted(set(seeds))


def _summarize(records: list[dict]) -> dict:
    wins = sum(float(row["margin"]) > 0 for row in records)
    ties = sum(float(row["margin"]) == 0 for row in records)
    losses = sum(float(row["margin"]) < 0 for row in records)
    margins = [float(row["margin"]) for row in records]
    money = [float(row["final_money"]) for row in records]
    rival_money = [float(row["rival_final_money"]) for row in records]

    by_seat = {}
    grouped = defaultdict(list)
    for row in records:
        grouped[int(row["learner_seat"])].append(row)
    for seat, rows in sorted(grouped.items()):
        seat_margins = [float(row["margin"]) for row in rows]
        seat_wins = sum(value > 0 for value in seat_margins)
        by_seat[str(seat)] = {
            "games": len(rows),
            "wins": int(seat_wins),
            "losses": int(sum(value < 0 for value in seat_margins)),
            "ties": int(sum(value == 0 for value in seat_margins)),
            "win_rate": float(seat_wins / max(len(rows), 1)),
            "mean_margin": float(statistics.fmean(seat_margins)),
            "median_margin": float(statistics.median(seat_margins)),
            "mean_final_money": float(statistics.fmean(
                float(row["final_money"]) for row in rows
            )),
        }

    runtime_ok = all(
        row.get("statuses") == ["DONE", "DONE"]
        and row.get("finite") is True
        and row.get("schema_valid") is True
        and row.get("timeout") is False
        and row.get("torch_import_free") is True
        for row in records
    )

    return {
        "games": len(records),
        "wins": int(wins),
        "losses": int(losses),
        "ties": int(ties),
        "win_rate": float(wins / max(len(records), 1)),
        "runtime_ok": bool(runtime_ok),
        "mean_margin": float(statistics.fmean(margins)),
        "median_margin": float(statistics.median(margins)),
        "min_margin": float(min(margins)),
        "max_margin": float(max(margins)),
        "mean_final_money": float(statistics.fmean(money)),
        "mean_v45_money": float(statistics.fmean(rival_money)),
        "by_seat": by_seat,
    }


def match_v32_vs_v45(
    checkpoint: str | Path,
    *,
    slot: int = 0,
    seeds: list[int] | None = None,
    output_dir: str | Path | None = None,
) -> dict:
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not V45.is_file():
        raise FileNotFoundError(
            f"v45 agent not found: {V45}\n"
            "Expected public v45 source at ../source_public/extracted_v45/main.py"
        )

    seeds = list(seeds or [20260919])
    payload, model, slot_to_team, exporter = _load_match_checkpoint(
        checkpoint
    )
    slot = int(slot)
    if not 0 <= slot < len(slot_to_team):
        raise ValueError(
            f"strategy slot {slot} outside [0, {len(slot_to_team) - 1}]"
        )

    if output_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output = checkpoint.parent / (
            f"vs_v45_epoch{payload.get('epoch', 'x')}_slot{slot}_{stamp}"
        )
    else:
        output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)

    from evaluation.eval_v3_closed_loop import (
        build_v3_game_specs,
        run_v3_game_matrix,
    )

    policy_path = output / f"slot_{slot:02d}_policy.npz"
    exporter(
        model,
        policy_path,
        default_strategy_slot=slot,
    )

    specs = build_v3_game_specs(seeds, ["v45"])
    matrix = run_v3_game_matrix(
        policy_path,
        specs,
        output / "games",
    )
    records = list(matrix.get("records") or [])
    summary = _summarize(records)

    compact_games = [
        {
            "seed": int(row["seed"]),
            "seat": int(row["learner_seat"]),
            "final_money": int(row["final_money"]),
            "v45_money": int(row["rival_final_money"]),
            "margin": int(row["margin"]),
            "won": bool(float(row["margin"]) > 0),
            "steps": int(row["steps"]),
            "statuses": row["statuses"],
            "schema_valid": bool(row["schema_valid"]),
            "timeout": bool(row["timeout"]),
            "land_unlocks": int(row.get("land_unlocks", 0)),
            "max_hands": int(row.get("max_hands", 0)),
            "longest_effectless_streak": int(
                row.get("longest_effectless_streak", 0)
            ),
        }
        for row in records
    ]

    report = {
        "checkpoint": str(checkpoint),
        "epoch": int(payload.get("epoch", -1)),
        "train_steps": int(payload.get("train_steps", -1)),
        "slot": slot,
        "team_id": int(slot_to_team[slot]),
        "opponent": "v45",
        "v45_path": str(V45),
        "seeds": seeds,
        "summary": summary,
        "games": compact_games,
        "matrix_summary": matrix.get("summary"),
    }
    report_path = output / "v45_match_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("\n=== FARMOS V3.2 vs V45 ===")
    for game in compact_games:
        result = (
            "WIN" if game["margin"] > 0
            else "LOSS" if game["margin"] < 0
            else "TIE"
        )
        print(
            f"seed={game['seed']} seat={game['seat']} {result} "
            f"money={game['final_money']} v45={game['v45_money']} "
            f"margin={game['margin']:+d}"
        )
    print(
        f"TOTAL games={summary['games']} wins={summary['wins']} "
        f"losses={summary['losses']} ties={summary['ties']} "
        f"win_rate={summary['win_rate']:.1%} "
        f"mean_margin={summary['mean_margin']:+.1f} "
        f"runtime_ok={summary['runtime_ok']}"
    )
    print(f"REPORT={report_path}")
    print(
        "FARMOS_V45_MATCH="
        + json.dumps({
            "report": str(report_path),
            "games": summary["games"],
            "wins": summary["wins"],
            "losses": summary["losses"],
            "ties": summary["ties"],
            "win_rate": summary["win_rate"],
            "mean_margin": summary["mean_margin"],
            "runtime_ok": summary["runtime_ok"],
        }, sort_keys=True),
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a V3.2/V3.3 checkpoint only against public v45."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--slot", type=int, default=0)
    parser.add_argument(
        "--seeds",
        default="20260919",
        help="comma-separated seeds, e.g. 20260919,20260920,20260921",
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    match_v32_vs_v45(
        args.checkpoint,
        slot=args.slot,
        seeds=_parse_seeds(args.seeds),
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
