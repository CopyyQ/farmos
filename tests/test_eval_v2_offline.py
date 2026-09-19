import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pyarrow.parquet as pq
import torch

from evaluation.eval_v2_offline import evaluate_checkpoint, write_offline_report
from kaggrl.v2_dataset import build_arrow_table
from kaggrl.v2_model import RecurrentIntentPolicy
from kaggrl.v2_observation import normalize_observation


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unit():
    return {"op": "PASS", "item": None, "quantity": None, "raw": ["PASS"]}


def _stop():
    return {"kind": "STOP_QUEUE", "op": None, "item": None, "quantity": None, "raw": []}

def _state(step: int, hands: int):
    tiles = [[None for _ in range(10)] for _ in range(10)]
    obs = {
        "player": 0, "step": step, "day": step // 24, "hour": step % 24,
        "farms": [
            {"money": 3000, "farmer": [4, 4], "hands": [[5, 4]] * hands,
             "hires_today": hands, "unlocked_quadrants": ["NW"], "tiles": tiles},
            {"money": 2500, "farmer": [8, 8], "hands": [], "hires_today": 0,
             "unlocked_quadrants": ["NW"], "tiles": tiles},
        ],
        "private": {"shed": {}, "seeds": {}, "inventories": [{}] * (hands + 1)},
        "market": {"inventory": {"WHEAT": 10000}, "prices": {"WHEAT": 25}},
        "town": {"unlocked_shops": []},
    }
    return asdict(normalize_observation(obs))


def _action(hands: int):
    return {"farmer": _unit(), "hands": [_unit() for _ in range(hands)], "market": [_stop()]}

def _row(step: int, hands: int):
    effects = {
        "money_delta": 0, "hand_count_delta": 0,
        "shed_delta": {}, "seed_delta": {},
        "market_inventory_delta": {}, "market_price_delta": {},
        "unit_position_delta": {}, "day_changed": False, "day_reset": False,
        "opponent_public": {"money_delta": 0, "hand_count_delta": 0,
                            "farmer_position_delta": [0, 0], "grid_changed": False,
                            "confidence": ""},
        "action_evidence": [],
    }
    return {
        "episode_id": 77, "seat": 0, "step": step,
        "create_time": "2026-09-17T00:00:00+00:00",
        "team_id": 7, "team_name": "toy", "submission_id": 70,
        "submission_date": "2026-09-16T00:00:00+00:00", "rank": 1,
        "role": "active_best", "split": "test",
        "replay_file_sha256": "a" * 64, "replay_json_sha256": "b" * 64,
        "state": _state(step, hands), "next_state": _state(step + 1, hands),
        "raw_action": {}, "canonical_action": _action(hands), "effects": effects,
        "final_own_money": 3000, "final_rival_money": 2500,
        "final_margin": 500, "terminal_result": 1,
    }

def _fixture(tmp_path: Path):
    rows = [_row(0, 0), _row(1, 2), _row(2, 0), _row(3, 2)]
    dataset = tmp_path / "transitions.parquet"
    pq.write_table(build_arrow_table(rows), dataset)
    torch.manual_seed(123)
    model = RecurrentIntentPolicy()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    checkpoint = tmp_path / "bc_best.pt"
    torch.save({
        "format_version": 1,
        "model_state": model.state_dict(),
        "dataset_sha256": _sha(dataset),
        "config": {"dataset_path": str(dataset), "sequence_len": 2},
    }, checkpoint)
    return dataset, checkpoint


def _assert_perfect_semantics(report):
    metrics = report["semantic"]
    assert metrics["farmer_semantic_exact"] == 1.0
    assert metrics["mean_hand_semantic_exact"] == 1.0
    assert metrics["market_sequence_exact"] == 1.0
    assert metrics["full_joint_step_exact"] == 1.0

def test_offline_teacher_and_free_running_are_perfect_for_zero_policy_pass_stop_dataset(tmp_path):
    dataset, checkpoint = _fixture(tmp_path)
    teacher = evaluate_checkpoint(checkpoint, "test", "teacher_forced")
    free = evaluate_checkpoint(checkpoint, "test", "free_running")
    assert teacher["rows"] == 4 and free["rows"] == 4
    assert teacher["dataset_sha256"] == _sha(dataset)
    assert free["checkpoint_sha256"] == _sha(checkpoint)
    _assert_perfect_semantics(teacher)
    _assert_perfect_semantics(free)
    assert teacher["semantic"]["stop_nop_confusions"] == 0
    assert free["semantic"]["stop_nop_confusions"] == 0


def test_offline_effect_only_zero_head_matches_zero_effect_targets(tmp_path):
    _, checkpoint = _fixture(tmp_path)
    report = evaluate_checkpoint(checkpoint, "test", "effect_only")
    assert report["rows"] == 4
    assert report["effect"]["effect_mae"] == 0.0
    assert report["effect"]["opponent_effect_mae"] == 0.0
    assert report["effect"]["future_resource_mae"] == 0.0
    assert report["finite"] is True

def test_offline_report_emits_required_slices_and_action_audits(tmp_path):
    _, checkpoint = _fixture(tmp_path)
    report = evaluate_checkpoint(checkpoint, "test", "free_running")
    assert report["slices"]["team"]["7:toy"]["rows"] == 4
    assert report["slices"]["rank_band"]["top3"]["rows"] == 4
    assert report["slices"]["day_band"]["0-5"]["rows"] == 4
    assert report["slices"]["hand_count_band"]["0"]["rows"] == 2
    assert report["slices"]["hand_count_band"]["1-4"]["rows"] == 2
    assert report["slices"]["market_order_count_band"]["0"]["rows"] == 4
    audit = report["audits"]
    assert audit["quantity"]["active_count"] == 0
    assert audit["quantity"]["integer_exact"] == 1.0
    assert audit["market_order_position_accuracy"] == 1.0
    assert audit["stop_nop_confusions"] == 0
    assert audit["top_mismatches"] == []


def test_write_offline_report_binds_checkpoint_dataset_and_evaluator(tmp_path):
    dataset, checkpoint = _fixture(tmp_path)
    output = tmp_path / "offline_eval.json"
    report = write_offline_report(checkpoint, "test", output, mode="free_running")
    assert output.is_file()
    saved = json.loads(output.read_text())
    assert saved == report
    assert saved["dataset_sha256"] == _sha(dataset)
    assert saved["checkpoint_sha256"] == _sha(checkpoint)
    assert saved["evaluator_version"] == 1
    assert saved["split_definition"] == {"split": "test", "roles": ["active_best"]}


def test_offline_audit_explicitly_reports_queue_class_counts_and_schema_validity(tmp_path):
    _, checkpoint = _fixture(tmp_path)
    report = evaluate_checkpoint(checkpoint, "test", "free_running")
    audit = report["audits"]
    assert audit["schema_valid"] is True
    assert audit["queue_class_counts"] == {"STOP_QUEUE": 4, "NOP_SLOT": 0}
