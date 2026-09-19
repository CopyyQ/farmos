import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from test_v2_rollout_agent import _obs

from evaluation.eval_v3_parity import compare_v3_fixture
from kaggrl.v3_export import export_v3_numpy
from kaggrl.v3_model import TemporalIntentPolicy
from rollout.v3_agent_numpy import V3NumpyRolloutAgent


def test_live_fixture_matches_torch_logits_masks_and_actions(tmp_path):
    torch.manual_seed(61)
    model = TemporalIntentPolicy().eval()
    checkpoint = tmp_path / "model.pt"
    torch.save({
        "architecture_version": model.ARCHITECTURE_VERSION,
        "model_state": model.state_dict(),
    }, checkpoint)
    npz = tmp_path / "policy.npz"
    export_v3_numpy(model, npz)
    agent = V3NumpyRolloutAgent(
        npz, seed=31, deterministic=True, capture_decision_trace=True,
    )
    agent(_obs(0, hands=1))
    report = compare_v3_fixture(checkpoint, npz, agent.diagnostic_fixtures[0])
    assert report["passed"] is True
    assert report["mask_equal"] is True
    assert report["action_equal"] is True
    assert report["max_raw_logit_abs_error"] < 5e-4


def test_feature_snapshot_is_stable_for_same_live_state(tmp_path):
    from evaluation.eval_v3_parity import capture_feature_snapshot

    torch.manual_seed(62)
    model = TemporalIntentPolicy().eval()
    npz = tmp_path / "policy.npz"
    export_v3_numpy(model, npz)
    agent = V3NumpyRolloutAgent(
        npz, seed=32, deterministic=True, capture_decision_trace=True,
    )
    agent(_obs(0, hands=1))
    fixture = agent.diagnostic_fixtures[0]
    left = capture_feature_snapshot(
        fixture["structured_state"], fixture.get("previous_action") or {},
        fixture.get("previous_effect") or {},
    )
    right = capture_feature_snapshot(
        fixture["structured_state"], fixture.get("previous_action") or {},
        fixture.get("previous_effect") or {},
    )
    assert left["feature_sha256"] == right["feature_sha256"]
    assert left["shapes"] == right["shapes"]


def test_feature_snapshot_comparison_identifies_economy_mismatch(tmp_path):
    import copy
    from evaluation.eval_v3_parity import (
        capture_feature_snapshot, compare_feature_snapshots,
    )

    torch.manual_seed(63)
    model = TemporalIntentPolicy().eval()
    npz = tmp_path / "policy.npz"
    export_v3_numpy(model, npz)
    agent = V3NumpyRolloutAgent(
        npz, seed=33, deterministic=True, capture_decision_trace=True,
    )
    agent(_obs(0, hands=1))
    fixture = agent.diagnostic_fixtures[0]
    changed = copy.deepcopy(fixture["structured_state"])
    changed["step"] = int(changed.get("step", 0)) + 1
    left = capture_feature_snapshot(
        fixture["structured_state"], fixture.get("previous_action") or {},
        fixture.get("previous_effect") or {},
    )
    right = capture_feature_snapshot(
        changed, fixture.get("previous_action") or {},
        fixture.get("previous_effect") or {},
    )
    report = compare_feature_snapshots(left, right)
    assert report["passed"] is False
    assert report["mismatched_families"] == ["economy"]


def test_raw_observation_and_normalized_training_state_have_identical_features():
    from dataclasses import asdict
    from kaggrl.v2_observation import normalize_observation
    from evaluation.eval_v3_parity import (
        capture_feature_snapshot, capture_observation_feature_snapshot,
        compare_feature_snapshots,
    )

    raw = _obs(0, hands=2)
    structured = asdict(normalize_observation(raw))
    live = capture_observation_feature_snapshot(raw, {}, {})
    training = capture_feature_snapshot(structured, {}, {})
    assert compare_feature_snapshots(live, training)["passed"] is True
    assert live["feature_sha256"] == training["feature_sha256"]


def test_expert_feature_snapshot_collection_uses_prepared_training_context(tmp_path):
    from test_v2_training_data import _dataset, _row
    from evaluation.build_v3_feature_parity_corpus import collect_expert_snapshots

    path = _dataset(
        tmp_path / "transitions.parquet",
        [_row(71, step) for step in range(3)],
    )
    rows = collect_expert_snapshots(
        path, split="train", steps={0, 2}, max_rows=10,
    )
    assert [row["step"] for row in rows] == [0, 2]
    assert all(row["source"] == "expert" for row in rows)
    assert all(len(row["feature"]["feature_sha256"]) == 64 for row in rows)
    assert rows[1]["previous_action_present"] is True


def test_distribution_comparison_allows_dynamic_unit_axis_but_not_feature_width():
    from evaluation.build_v3_feature_parity_corpus import build_distribution_comparison

    def row(units, width=4):
        feature = {
            "feature_sha256": "a" * 64,
            "stats": {
                "own_units": {"mean": 0.1, "min": 0.0, "max": 1.0, "finite": True},
                "economy": {"mean": 0.2, "min": 0.0, "max": 1.0, "finite": True},
            },
            "shapes": {"own_units": [units, width], "economy": [8]},
            "dtypes": {"own_units": "float32", "economy": "float32"},
        }
        return {"feature": feature, "raw_structured_exact": True}

    assert build_distribution_comparison([row(5)], [row(1)])["passed"] is True
    bad = build_distribution_comparison([row(5)], [row(1, width=7)])
    assert bad["passed"] is False
    assert bad["schema_mismatches"] == ["own_units"]
