from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _opponent_tag(report: dict) -> str:
    return (
        str(report.get("opponent", ""))
        + " "
        + str(report.get("opponent_path", ""))
    ).lower()


def _games(report: dict) -> dict[tuple[int, int], dict]:
    out = {}
    for row in report.get("games", []):
        key = (int(row["seed"]), int(row["seat"]))
        if key in out:
            raise ValueError(f"duplicate game key: {key}")
        out[key] = row
    if not out:
        raise ValueError("match report has no games")
    return out


def compare_reports(
    baseline_report: str | Path,
    candidate_report: str | Path,
    *,
    min_mean_improvement: float = 0.0,
    min_median_paired_improvement: float = 0.0,
    max_single_game_regression: float = 5000.0,
) -> dict:
    baseline = _load(baseline_report)
    candidate = _load(candidate_report)
    base_path = baseline.get("opponent_path")
    cand_path = candidate.get("opponent_path")
    base_name = baseline.get("opponent")
    cand_name = candidate.get("opponent")
    if base_path is not None or cand_path is not None:
        if base_path != cand_path:
            raise ValueError("baseline/candidate opponent path differs")
    elif base_name is not None or cand_name is not None:
        if base_name != cand_name:
            raise ValueError("baseline/candidate opponent differs")
    opponent_name = (
        str(base_name)
        if base_name is not None
        else str(base_path or "unspecified")
    )
    base_games = _games(baseline)
    cand_games = _games(candidate)
    if set(base_games) != set(cand_games):
        raise ValueError("baseline/candidate seeds and seats differ")

    keys = sorted(base_games)
    base_margin = [float(base_games[key]["margin"]) for key in keys]
    cand_margin = [float(cand_games[key]["margin"]) for key in keys]
    paired = [c - b for b, c in zip(base_margin, cand_margin)]


    candidate_done = all(
        row.get("statuses") == ["DONE", "DONE"]
        for row in cand_games.values()
    )
    base_win_rate = sum(value > 0 for value in base_margin) / len(base_margin)
    cand_win_rate = sum(value > 0 for value in cand_margin) / len(cand_margin)
    mean_improvement = statistics.fmean(paired)
    median_paired_improvement = statistics.median(paired)

    checks = {
        "candidate_done_100pct": candidate_done,
        "runtime_ok": bool(candidate.get("summary", {}).get("runtime_ok")),
        "clock_ok": bool(candidate.get("summary", {}).get("clock_ok")),
        "strategy_steps_ok": bool(
            candidate.get("summary", {}).get("strategy_steps_ok")
        ),
        "mean_margin_improves": (
            mean_improvement >= float(min_mean_improvement)
        ),
        "median_paired_margin_improves": (
            median_paired_improvement
            >= float(min_median_paired_improvement)
        ),
        "win_rate_not_worse": cand_win_rate >= base_win_rate,
        "no_single_game_collapse": (
            min(paired) >= -float(max_single_game_regression)
        ),
    }
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "opponent": opponent_name,
        "opponent_path": base_path,
        "games": len(keys),
        "baseline_mean_margin": statistics.fmean(base_margin),
        "candidate_mean_margin": statistics.fmean(cand_margin),
        "mean_margin_improvement": mean_improvement,
        "median_paired_improvement": median_paired_improvement,
        "worst_paired_improvement": min(paired),
        "best_paired_improvement": max(paired),
        "baseline_win_rate": base_win_rate,
        "candidate_win_rate": cand_win_rate,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-report", required=True)
    parser.add_argument("--candidate-report", required=True)
    parser.add_argument("--min-mean-improvement", type=float, default=0.0)
    parser.add_argument(
        "--min-median-paired-improvement", type=float, default=0.0
    )
    parser.add_argument(
        "--max-single-game-regression", type=float, default=5000.0
    )
    args = parser.parse_args()
    result = compare_reports(
        args.baseline_report,
        args.candidate_report,
        min_mean_improvement=args.min_mean_improvement,
        min_median_paired_improvement=(
            args.min_median_paired_improvement
        ),
        max_single_game_regression=args.max_single_game_regression,
    )
    print("FARMOS_V4_MARGIN_PROMOTION=" + json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 2)


if __name__ == "__main__":
    main()
