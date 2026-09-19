import json
from pathlib import Path

import pytest
import torch

from kaggrl.v2_model import RecurrentIntentPolicy
from kaggrl.v3_model import TemporalIntentPolicy
from training.migrate_v2_to_v3 import (
    initialize_v3,
    write_v3_initialization_artifact,
)


def _v2_checkpoint(path: Path, *, tamper_shape: bool = False) -> Path:
    torch.manual_seed(91)
    model = RecurrentIntentPolicy()
    state = model.state_dict()
    if tamper_shape:
        state = dict(state)
        key = "encoder.tile_encoder.net.0.weight"
        state[key] = state[key][:-1].clone()
    torch.save({"model_state": state, "dataset_sha256": "d" * 64}, path)
    return path


def test_clean_initialization_is_deterministic():
    left, left_manifest = initialize_v3("clean", seed=20260917)
    right, right_manifest = initialize_v3("clean", seed=20260917)
    for key in left.state_dict():
        assert torch.equal(left.state_dict()[key], right.state_dict()[key]), key
    assert left_manifest["mode"] == "clean"
    assert left_manifest["v2_checkpoint_sha256"] is None
    assert left_manifest["loaded_parameters"] == []
    assert left_manifest["result_model_sha256"] == right_manifest["result_model_sha256"]


def test_partial_warm_start_loads_only_explicit_retained_components(tmp_path):
    checkpoint = _v2_checkpoint(tmp_path / "v2.pt")
    clean, _ = initialize_v3("clean", seed=20260917)
    warm, manifest = initialize_v3(
        "partial_v2_warm_start", seed=20260917, v2_checkpoint=checkpoint,
    )
    source = torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state"]
    assert torch.equal(warm.state_dict()["encoder.tile_encoder.net.0.weight"],
                       source["encoder.tile_encoder.net.0.weight"])
    assert torch.equal(warm.state_dict()["unit_op_head.weight"], source["unit_op_head.weight"])
    assert torch.equal(warm.state_dict()["core.lstm.weight_ih"],
                       clean.state_dict()["core.lstm.weight_ih"])
    assert torch.equal(warm.state_dict()["core.intent.0.weight"],
                       clean.state_dict()["core.intent.0.weight"])
    assert "core.lstm.weight_ih" not in manifest["loaded_parameters"]
    assert "core.intent.0.weight" not in manifest["loaded_parameters"]
    reasons = {row["name"]: row["reason"] for row in manifest["skipped_parameters"]}
    assert reasons["core.lstm.weight_ih"] == "temporal_parameter_excluded"
    assert reasons["core.intent.0.weight"] == "temporal_parameter_excluded"
    assert manifest["temporal_config"]["window"] == 32


def test_partial_warm_start_rejects_shape_mismatch(tmp_path):
    checkpoint = _v2_checkpoint(tmp_path / "bad.pt", tamper_shape=True)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        initialize_v3(
            "partial_v2_warm_start", seed=20260917, v2_checkpoint=checkpoint,
        )


def test_initialization_artifact_is_immutable_and_bound_to_manifest(tmp_path):
    model, manifest = initialize_v3("clean", seed=20260917)
    path = tmp_path / "v3_clean.pt"
    write_v3_initialization_artifact(model, manifest, path)
    sidecar = path.with_suffix(path.suffix + ".migration.json")
    assert path.is_file() and sidecar.is_file()
    saved = torch.load(path, map_location="cpu", weights_only=False)
    side = json.loads(sidecar.read_text(encoding="utf-8"))
    assert saved["architecture_version"] == "rl_v3_temporal_attention"
    assert saved["migration_manifest"]["result_model_sha256"] == side["result_model_sha256"]
    with pytest.raises(FileExistsError):
        write_v3_initialization_artifact(model, manifest, path)
