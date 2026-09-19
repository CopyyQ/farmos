from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from kaggrl.v3_2_export import export_v3_2_numpy
from kaggrl.v3_2_model import TemporalIntentPolicyV32
from kaggrl.v3_2_schema import ARCHITECTURE_VERSION as V32_ARCH
from kaggrl.v3_3_export import export_v3_3_numpy
from kaggrl.v3_3_model import TemporalIntentPolicyV33
from kaggrl.v3_3_schema import ARCHITECTURE_VERSION as V33_ARCH


def _load_checkpoint(checkpoint: Path):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    architecture = str(payload.get("architecture_version", ""))
    manifest = payload.get("strategy_manifest") or {}
    slot_to_team = tuple(int(v) for v in manifest.get("slot_to_team") or ())
    if not slot_to_team:
        raise RuntimeError("checkpoint has no strategy_manifest.slot_to_team")
    if architecture == V32_ARCH:
        model = TemporalIntentPolicyV32(strategy_count=len(slot_to_team)).eval()
        exporter = export_v3_2_numpy
    elif architecture == V33_ARCH:
        model = TemporalIntentPolicyV33(strategy_count=len(slot_to_team)).eval()
        exporter = export_v3_3_numpy
    else:
        raise RuntimeError(
            f"match requires a V3.2/V3.3 checkpoint, got {architecture!r}"
        )
    model.load_state_dict(payload["model_state"], strict=True)
    return payload, model, slot_to_team, exporter


def _parse_seeds(text: str) -> list[int]:
    seeds = sorted({int(x.strip()) for x in str(text).split(",") if x.strip()})
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)


def _summarize(records: list[dict]) -> dict:
    margins = [float(r["margin"]) for r in records]
    wins = sum(v > 0 for v in margins)
    losses = sum(v < 0 for v in margins)
    ties = sum(v == 0 for v in margins)
    by_seat = {}
    grouped = defaultdict(list)
    for row in records:
        grouped[int(row["learner_seat"])].append(row)
    for seat, rows in sorted(grouped.items()):
        sm = [float(r["margin"]) for r in rows]
        by_seat[str(seat)] = {
            "games": len(rows),
            "wins": int(sum(v > 0 for v in sm)),
            "losses": int(sum(v < 0 for v in sm)),
            "ties": int(sum(v == 0 for v in sm)),
            "win_rate": float(sum(v > 0 for v in sm) / max(1, len(sm))),
            "mean_margin": float(statistics.fmean(sm)),
            "median_margin": float(statistics.median(sm)),
            "mean_final_money": float(statistics.fmean(
                float(r["final_money"]) for r in rows
            )),
            "mean_opponent_money": float(statistics.fmean(
                float(r["rival_final_money"]) for r in rows
            )),
        }
    runtime_ok = all(
        r.get("statuses") == ["DONE", "DONE"]
        and r.get("finite") is True
        and r.get("schema_valid") is True
        and r.get("timeout") is False
        and r.get("torch_import_free") is True
        for r in records
    )
    return {
        "games": len(records),
        "wins": int(wins),
        "losses": int(losses),
        "ties": int(ties),
        "win_rate": float(wins / max(1, len(records))),
        "runtime_ok": bool(runtime_ok),
        "mean_margin": float(statistics.fmean(margins)),
        "median_margin": float(statistics.median(margins)),
        "min_margin": float(min(margins)),
        "max_margin": float(max(margins)),
        "mean_final_money": float(statistics.fmean(
            float(r["final_money"]) for r in records
        )),
        "mean_opponent_money": float(statistics.fmean(
            float(r["rival_final_money"]) for r in records
        )),
        "by_seat": by_seat,
    }


def match_checkpoint_vs_agent(
    checkpoint: str | Path,
    opponent: str | Path,
    *,
    opponent_name: str | None = None,
    slot: int = 0,
    seeds: list[int] | None = None,
    output_dir: str | Path | None = None,
) -> dict:
    checkpoint = Path(checkpoint).resolve()
    opponent = Path(opponent).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not opponent.is_file():
        raise FileNotFoundError(opponent)

    opponent_name = opponent_name or opponent.parent.name or opponent.stem
    opponent_name = _safe_name(opponent_name)
    seeds = list(seeds or [20271000, 20271001, 20271002, 20271003, 20271004])

    payload, model, slot_to_team, exporter = _load_checkpoint(checkpoint)
    slot = int(slot)
    if not 0 <= slot < len(slot_to_team):
        raise ValueError(
            f"strategy slot {slot} outside [0, {len(slot_to_team) - 1}]"
        )

    if output_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output = checkpoint.parent / (
            f"vs_{opponent_name}_epoch{payload.get('epoch', 'x')}"
            f"_slot{slot}_{stamp}"
        )
    else:
        output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)

    from evaluation.eval_v3_closed_loop import (
        build_v3_game_specs,
        run_v3_game_matrix,
    )

    policy_path = output / f"slot_{slot:02d}_policy.npz"
    exporter(model, policy_path, default_strategy_slot=slot)

    specs = build_v3_game_specs(seeds, [str(opponent)])
    matrix = run_v3_game_matrix(policy_path, specs, output / "games")
    records = list(matrix.get("records") or [])
    summary = _summarize(records)

    compact_games = [{
        "seed": int(row["seed"]),
        "seat": int(row["learner_seat"]),
        "final_money": int(row["final_money"]),
        "opponent_money": int(row["rival_final_money"]),
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
        "effective_family_counts": row.get("effective_family_counts") or {},
        "action_histograms": row.get("action_histograms") or {},
    } for row in records]

    report = {
        "checkpoint": str(checkpoint),
        "epoch": int(payload.get("epoch", -1)),
        "train_steps": int(payload.get("train_steps", -1)),
        "slot": slot,
        "team_id": int(slot_to_team[slot]),
        "opponent": opponent_name,
        "opponent_path": str(opponent),
        "seeds": seeds,
        "summary": summary,
        "games": compact_games,
        "matrix_summary": matrix.get("summary"),
    }
    report_path = output / f"{opponent_name}_match_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"\n=== FARMOS vs {opponent_name.upper()} ===")
    for game in compact_games:
        result = "WIN" if game["margin"] > 0 else (
            "LOSS" if game["margin"] < 0 else "TIE"
        )
        print(
            f"seed={game['seed']} seat={game['seat']} {result} "
            f"money={game['final_money']} "
            f"opponent={game['opponent_money']} "
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
        "FARMOS_AGENT_MATCH="
        + json.dumps({
            "opponent": opponent_name,
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
        description="Run a V3.2/V3.3 checkpoint against an arbitrary agent."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--opponent", required=True)
    parser.add_argument("--opponent-name", default=None)
    parser.add_argument("--slot", type=int, default=0)
    parser.add_argument(
        "--seeds",
        default="20271000,20271001,20271002,20271003,20271004",
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    match_checkpoint_vs_agent(
        args.checkpoint,
        args.opponent,
        opponent_name=args.opponent_name,
        slot=args.slot,
        seeds=_parse_seeds(args.seeds),
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
