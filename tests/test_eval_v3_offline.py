import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
from test_train_v2_bc import _fixture, _sha

from kaggrl.v3_model import TemporalIntentPolicy
from evaluation.eval_v3_offline import (
    classify_primary_failure,
    evaluate_v3_offline,
    write_v3_offline_report,
)


def _checkpoint(path, dataset):
    torch.manual_seed(77)
    model = TemporalIntentPolicy().eval()
    torch.save({
        "architecture_version": "rl_v3_temporal_attention",
        "format_version": 3,
        "model_state": model.state_dict(),
        "dataset_sha256": _sha(dataset),
        "config": {"dataset_path": str(dataset), "seed": 20260917},
    }, path)
    return path


def test_v3_offline_report_runs_all_modes_with_matching_rows(tmp_path):
    dataset, _, _, _ = _fixture(tmp_path)
    checkpoint = _checkpoint(tmp_path / "v3.pt", dataset)
    report = evaluate_v3_offline(checkpoint, split="val")
    assert report["architecture_version"] == "rl_v3_temporal_attention"
    assert report["split_definition"] == {"split": "val", "roles": ["active_best"]}
    assert report["expert_history"]["rows"] == report["free_history"]["rows"]
    assert report["expert_history"]["rows"] == report["effect_only"]["rows"]
    assert report["expert_history"]["episodes"] == report["free_history"]["episodes"]
    assert set(report["gaps"]) == {"farmer_op", "hands_op", "market_op"}
    assert report["finite"] is True


def test_v3_offline_report_includes_explicit_collapse_fractions(tmp_path):
    dataset, _, _, _ = _fixture(tmp_path)
    report = evaluate_v3_offline(_checkpoint(tmp_path / "v3.pt", dataset), "val")
    for mode in ("expert_history", "free_history"):
        fractions = report[mode]["op_fractions"]
        assert set(fractions) >= {
            "farmer_pass", "hands_pass", "market_stop_queue", "market_nop_slot",
            "expert_farmer_pass", "expert_hands_pass", "expert_market_stop_queue",
        }
        assert report[mode]["attention"]["finite"] is True


def test_v3_offline_writer_is_immutable(tmp_path):
    dataset, _, _, _ = _fixture(tmp_path)
    checkpoint = _checkpoint(tmp_path / "v3.pt", dataset)
    output = tmp_path / "report.json"
    write_v3_offline_report(checkpoint, "val", output)
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["finite"] is True
    with pytest.raises(FileExistsError):
        write_v3_offline_report(checkpoint, "val", output)


def test_failure_classifier_uses_fail_closed_priority_order():
    assert classify_primary_failure({"runtime_ok": False}, {}) == "runtime_or_parity"
    assert classify_primary_failure(
        {"runtime_ok": True, "legality_ok": False}, {}
    ) == "legality_over_mask"
    assert classify_primary_failure(
        {"runtime_ok": True, "legality_ok": True, "action_logits_collapsed": True}, {}
    ) == "action_logit_collapse"
    assert classify_primary_failure(
        {"runtime_ok": True, "legality_ok": True, "temporal_memory_ok": False}, {}
    ) == "temporal_memory_collapse"
    assert classify_primary_failure(
        {"runtime_ok": True, "legality_ok": True, "teacher_free_gap_ok": False}, {}
    ) == "teacher_free_distribution_shift"
    assert classify_primary_failure(
        {"runtime_ok": True, "legality_ok": True, "teacher_free_gap_ok": True},
        {"economic_chain_ok": False},
    ) == "insufficient_recovery_semantics"
