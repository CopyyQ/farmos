from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SAFE_MARKET_MODES = (
    "KEEP_ROUTE",
    "LIQUIDATE_SHED",
    "HOLD_SALES",
    "FRONT_RUN_1",
    "FRONT_RUN_9",
)


def _run(args, *, env=None):
    command = [str(value) for value in args]
    print("FARMOS_CMD=" + json.dumps(command), flush=True)
    subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=True,
    )


def _python() -> str:
    return sys.executable


def _select_base_checkpoint(base_dir: pathlib.Path) -> pathlib.Path:
    best = base_dir / "bc_best.pt"
    if best.is_file():
        return best

    last = base_dir / "bc_last.pt"
    if not last.is_file():
        raise RuntimeError(
            "base V4.1 checkpoint is missing; "
            f"inspect {base_dir / 'summary.json'}"
        )

    import torch
    sys.path.insert(0, str(ROOT))
    from training.train_v4_options import bootstrap_gate

    payload = torch.load(last, map_location="cpu", weights_only=False)
    metrics = payload.get("validation_metrics", {})
    gate = bootstrap_gate(metrics)
    print(
        "FARMOS_V4_BASE_BOOTSTRAP="
        + json.dumps(
            {
                "checkpoint": str(last),
                "bootstrap_passed": bool(gate["passed"]),
                "failed_checks": [
                    key
                    for key, value in gate["checks"].items()
                    if not value
                ],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not gate["passed"]:
        raise RuntimeError(
            "base V4.1 did not pass bootstrap BC/safety gate; "
            f"inspect {base_dir / 'summary.json'}"
        )
    return last


def main():
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end FarmOS V4.1 margin/Q training, benchmark, "
            "promotion and submission packaging."
        )
    )
    parser.add_argument("--opponent", required=True)
    parser.add_argument("--opponent-name", default="target")
    parser.add_argument(
        "--transitions",
        type=pathlib.Path,
        default=ROOT / "data/top_tier/live_v2/transitions.parquet",
    )
    parser.add_argument(
        "--work-dir",
        type=pathlib.Path,
        default=ROOT / "runs/v4_margin_pipeline",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base-epochs", type=int, default=8)
    parser.add_argument("--base-batch", type=int, default=128)
    parser.add_argument("--base-lr", type=float, default=3e-4)
    parser.add_argument("--q-epochs", type=int, default=100)
    parser.add_argument("--q-batch", type=int, default=64)
    parser.add_argument("--q-lr", type=float, default=1e-3)
    parser.add_argument("--q-horizon", type=int, default=8)
    parser.add_argument(
        "--q-steps",
        default="144,360,648,696",
    )
    parser.add_argument(
        "--q-train-seeds",
        default="20310000,20310001",
    )
    parser.add_argument("--q-val-seeds", default="20310100")
    parser.add_argument("--q-test-seeds", default="20310200")
    parser.add_argument(
        "--benchmark-seeds",
        default="20320000,20320001,20320002,20320003,20320004",
    )
    parser.add_argument(
        "--min-mean-improvement", type=float, default=0.0
    )
    parser.add_argument(
        "--min-median-improvement", type=float, default=0.0
    )
    parser.add_argument(
        "--max-single-game-regression", type=float, default=5000.0
    )
    parser.add_argument(
        "--min-route-advantage", type=float, default=500.0
    )
    parser.add_argument(
        "--min-market-advantage", type=float, default=500.0
    )
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    opponent = pathlib.Path(args.opponent).resolve()
    if opponent.is_dir():
        opponent = opponent / "main.py"
    if not opponent.is_file():
        raise FileNotFoundError(opponent)
    transitions = args.transitions.resolve()
    if not transitions.is_file():
        raise FileNotFoundError(transitions)

    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    dataset = work / "v4_options_margin.parquet"
    manifest = work / "v4_options_margin.json"
    base_dir = work / "base_train"
    cf_dataset = work / "counterfactual_q.parquet"
    q_dir = work / "q_train"
    policy_npz = work / "policy_q.npz"
    baseline_dir = work / "benchmark_baseline"
    candidate_dir = work / "benchmark_candidate"
    submission = work / "submission.tar.gz"

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([
        str(ROOT / "src"),
        str(ROOT),
        env.get("PYTHONPATH", ""),
    ]).rstrip(os.pathsep)

    if args.rebuild or not dataset.is_file() or not manifest.is_file():
        _run([
            _python(),
            "training/build_v4_option_dataset.py",
            "--source", transitions,
            "--output", dataset,
            "--manifest", manifest,
            "--horizon", "8",
            "--sequence-len", "32",
        ], env=env)

    if args.rebuild and base_dir.exists():
        import shutil
        shutil.rmtree(base_dir)
    if not base_dir.exists():
        train_args = [
            _python(),
            "training/train_v4_options.py",
            "--dataset", dataset,
            "--manifest", manifest,
            "--output-dir", base_dir,
            "--sequence-len", "32",
            "--batch-sequences", str(args.base_batch),
            "--epochs", str(args.base_epochs),
            "--learning-rate", str(args.base_lr),
            "--hidden-dim", "192",
            "--device", args.device,
        ]
        if args.device == "cpu":
            train_args.append("--no-amp")
        _run(train_args, env=env)

    base_checkpoint = _select_base_checkpoint(base_dir)

    if args.rebuild:
        for path in (cf_dataset, cf_dataset.with_suffix(".json")):
            if path.exists():
                path.unlink()
    if not cf_dataset.is_file():
        _run([
            _python(),
            "scripts/build_v4_counterfactual_q.py",
            "--opponent", opponent,
            "--output", cf_dataset,
            "--train-seeds", args.q_train_seeds,
            "--val-seeds", args.q_val_seeds,
            "--test-seeds", args.q_test_seeds,
            "--steps", args.q_steps,
            "--sequence-len", "32",
            "--option-horizon", str(args.q_horizon),
        ], env=env)

    if args.rebuild and q_dir.exists():
        import shutil
        shutil.rmtree(q_dir)
    if not q_dir.exists():
        _run([
            _python(),
            "training/train_v4_counterfactual_q.py",
            "--dataset", cf_dataset,
            "--base-checkpoint", base_checkpoint,
            "--output-dir", q_dir,
            "--epochs", str(args.q_epochs),
            "--batch-size", str(args.q_batch),
            "--learning-rate", str(args.q_lr),
            "--device", args.device,
        ], env=env)

    q_checkpoint = q_dir / "q_best.pt"
    if not q_checkpoint.is_file():
        raise RuntimeError("counterfactual Q checkpoint was not created")

    _run([
        _python(),
        "scripts/export_v4_option_numpy.py",
        "--checkpoint", q_checkpoint,
        "--output", policy_npz,
    ], env=env)

    import shutil
    for folder in (baseline_dir, candidate_dir):
        if folder.exists():
            shutil.rmtree(folder)

    _run([
        _python(),
        "scripts/match_v4_vs_agent.py",
        "--opponent", opponent,
        "--opponent-name", args.opponent_name,
        "--seeds", args.benchmark_seeds,
        "--output-dir", baseline_dir,
    ], env=env)

    _run([
        _python(),
        "scripts/match_v4_vs_agent.py",
        "--opponent", opponent,
        "--opponent-name", args.opponent_name,
        "--seeds", args.benchmark_seeds,
        "--output-dir", candidate_dir,
        "--option-checkpoint", policy_npz,
        "--allowed-market-modes", ",".join(SAFE_MARKET_MODES),
    ], env=env)

    _run([
        _python(),
        "scripts/promote_v4_margin.py",
        "--baseline-report", baseline_dir / "match_report.json",
        "--candidate-report", candidate_dir / "match_report.json",
        "--min-mean-improvement", str(args.min_mean_improvement),
        "--min-median-paired-improvement",
        str(args.min_median_improvement),
        "--max-single-game-regression",
        str(args.max_single_game_regression),
    ], env=env)

    _run([
        _python(),
        "scripts/build_v4_submission.py",
        "--policy", policy_npz,
        "--output", submission,
        "--min-route-advantage", str(args.min_route_advantage),
        "--min-market-advantage", str(args.min_market_advantage),
    ], env=env)

    result = {
        "opponent": str(opponent),
        "base_checkpoint": str(base_checkpoint),
        "counterfactual_dataset": str(cf_dataset),
        "q_checkpoint": str(q_checkpoint),
        "policy_npz": str(policy_npz),
        "baseline_report": str(baseline_dir / "match_report.json"),
        "candidate_report": str(candidate_dir / "match_report.json"),
        "submission": str(submission),
    }
    (work / "pipeline_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "FARMOS_V4_PIPELINE="
        + json.dumps(result, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
