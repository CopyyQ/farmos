from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kaggrl.top10_evaluation import summarize_metrics

CHECKPOINTS = ROOT / "checkpoints" / "top10_moe_bc"
DATASET_SUMMARY = ROOT / "data" / "top_tier" / "live" / "manifests" / "full_action_bc_summary.json"
OUT = CHECKPOINTS / "summary.json"


def main():
    metrics_files = sorted(CHECKPOINTS.glob("*.metrics.json"))
    summary = summarize_metrics(metrics_files)
    expected = set(json.loads(DATASET_SUMMARY.read_text())["teams"])
    actual = set(summary["teams"])
    if actual != expected:
        raise RuntimeError(f"expert metrics mismatch: missing={sorted(expected-actual)} extra={sorted(actual-expected)}")
    OUT.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps({
        "team_count": summary["team_count"],
        "gate_pass_count": summary["gate_pass_count"],
        "best_validation_expert": summary["best_validation_expert"],
        "aggregate": summary["aggregate"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
