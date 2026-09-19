from copy import deepcopy
from dataclasses import asdict
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_v2_rollout_agent import _obs

from kaggrl.v2_observation import normalize_observation
from training.build_v3_recovery_dataset import (
    canonicalize_teacher_action,
    validate_recovery_row,
)


def _row(raw_action):
    obs = _obs(0, hands=1)
    return {
        "episode_id": 9001, "seat": 0, "step": 0,
        "state": asdict(normalize_observation(obs)),
        "canonical_action": canonicalize_teacher_action(raw_action, obs),
        "previous_action": {}, "previous_effect": {}, "effects": {},
        "teacher_id": "starter", "teacher_version": "kaggle-env-1.32.7",
        "supervision_kind": "smoke_only",
    }


def test_recovery_row_requires_teacher_provenance():
    row = _row({"farmer": ["NORTH"], "hands": [["PASS"]], "market": []})
    del row["teacher_version"]
    with pytest.raises(ValueError, match="teacher_version"):
        validate_recovery_row(row)


def test_recovery_row_rejects_illegal_teacher_action():
    row = _row({"farmer": ["TELEPORT"], "hands": [["PASS"]], "market": []})
    with pytest.raises(ValueError, match="illegal farmer op"):
        validate_recovery_row(row)


def test_collect_starter_recovery_smoke(tmp_path):
    import torch
    from kaggrl.v3_export import export_v3_numpy
    from kaggrl.v3_model import TemporalIntentPolicy
    from training.build_v3_recovery_dataset import (
        collect_starter_recovery,
        read_recovery_rows,
    )

    torch.manual_seed(73)
    model_path = tmp_path / "policy.npz"
    export_v3_numpy(TemporalIntentPolicy().eval(), model_path)
    output = tmp_path / "recovery.jsonl"
    collect_starter_recovery(
        model_path, output, seeds=[777], episode_steps=6,
    )
    rows = read_recovery_rows(output)
    assert rows
    assert {int(row["seat"]) for row in rows} == {0, 1}
    assert all(row["teacher_id"] == "starter" for row in rows)
    assert all(row["supervision_kind"] == "smoke_only" for row in rows)


def test_build_expert_early_recovery_uses_trusted_actions(tmp_path):
    from test_train_v2_bc import _fixture
    from training.build_v3_recovery_dataset import (
        build_expert_early_recovery,
        read_recovery_rows,
    )
    dataset, _, _, _ = _fixture(tmp_path)
    output = tmp_path / "early_expert.jsonl"
    build_expert_early_recovery(dataset, output, max_step=1)
    rows = read_recovery_rows(output)
    assert rows
    assert max(int(row["step"]) for row in rows) <= 1
    assert all(row["supervision_kind"] == "expert" for row in rows)
    assert all(row["teacher_id"].startswith("top10_active_best:") for row in rows)
