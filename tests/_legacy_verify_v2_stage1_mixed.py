import hashlib
import json
from pathlib import Path

import pytest

from evaluation.verify_v2_stage1 import (
    Stage1AcceptanceError,
    verify_stage1,
    write_stage1_marker,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _fixture(tmp_path: Path):
    root = tmp_path
    stage_dir = root / "checkpoints" / "rl_v2_stage1"
    stage_dir.mkdir(parents=True)
    stage0 = _write(root / "data/top_tier/live_v2/manifests/STAGE0_ACCEPTED.json", {
        "accepted": True, "counters": {"errors": 0},
        "artifacts": {"dataset_sha256": "d" * 64},
    })
    out = root / "checkpoints" / "rl_v2_stage1"; out.mkdir(parents=True)
    archive = out / "policy_init.npz"; archive.write_bytes(b"canonical-stage1-archive")
    parity = _write(out / "parity_report.json", {
        "passed": True, "action_mismatches": 0, "tolerance": 3e-5,
        "archive_sha256": _sha(archive), "feature_schema_sha256": "f" * 64,
        "rows": [{"hands": 0, "action_equal": True},
                 {"hands": 17, "action_equal": True},
                 {"hands": 32, "action_equal": True}],
    })
    build = _write(out / "build_manifest.json", {
        "parameter_count": 1_787_665,
        "archive": "checkpoints/rl_v2_stage1/policy_init.npz",
        "archive_size_bytes": archive.stat().st_size,
        "archive_sha256": _sha(archive),
        "model_parameter_sha256": "m" * 64,
        "exported_parameter_sha256": "e" * 64,
        "feature_schema_sha256": "f" * 64,
        "stage0_marker_sha256": _sha(stage0),
        "parity_report_sha256": _sha(parity), "parity_passed": True,
        "latency_target_ms": 200.0, "representative_p99_ms": 20.0,
        "latency_target_passed": True,
        "latency": [{"hands": 0, "p99_ms": 5.0, "worst_ms": 7.0},
                    {"hands": 32, "p99_ms": 20.0, "worst_ms": 25.0}],
        "stress_latency": [{"hands": 64, "p99_ms": 80.0, "worst_ms": 90.0},
                           {"hands": 128, "p99_ms": 150.0, "worst_ms": 160.0}],
        "torch_free_import": {"passed": True, "returncode": 0, "stdout": "torch-free-ok"},
        "source_sha256": {name: _sha(path) for name, path in sources.items()},
    })
    return root, stage0, build, parity, archive, sources


def test_stage1_accepts_canonical_bound_evidence(tmp_path):
    _, stage0, build, parity, archive, _ = _fixture(tmp_path)
    report = verify_stage1(stage0, build, parity)
    assert report["accepted"] is True and report["errors"] == []
    assert report["artifacts"]["stage0_marker_sha256"] == _sha(stage0)
    assert report["artifacts"]["build_manifest_sha256"] == _sha(build)
    assert report["artifacts"]["parity_report_sha256"] == _sha(parity)
    assert report["artifacts"]["archive_sha256"] == _sha(archive)
    assert report["artifacts"]["exported_parameter_sha256"] == "e" * 64
    assert report["gates"]["stress_hands"] == 128
    stage0 = _write_json(
        root / "data/top_tier/live_v2/manifests/STAGE0_ACCEPTED.json",
        {"accepted": True, "counters": {"errors": 0},
         "artifacts": {"dataset_sha256": "d" * 64}},
    )
    archive = stage_dir / "policy_init.npz"
    archive.write_bytes(b"trusted-stage1-archive")
    parity = _write_json(stage_dir / "parity_report.json", {
        "passed": True, "action_mismatches": 0, "tolerance": 3e-5,
        "archive_sha256": _sha(archive), "feature_schema_sha256": "f" * 64,
        "max_fused_abs": 1e-6, "max_h_abs": 1e-6, "max_c_abs": 1e-6,
        "max_intent_abs": 1e-6, "max_terminal_money_abs": 1e-6,
        "max_terminal_margin_abs": 1e-6,
        "rows": [{"hands": n, "action_equal": True} for n in (0, 17, 32)],
    })
    sources = {
        "v2_tensorize.py": root / "src/kaggrl/v2_tensorize.py",
        "v2_encoder.py": root / "src/kaggrl/v2_encoder.py",
        "v2_ledger.py": root / "src/kaggrl/v2_ledger.py",
        "v2_quantity.py": root / "src/kaggrl/v2_quantity.py",
        "v2_model.py": root / "src/kaggrl/v2_model.py",
        "v2_export.py": root / "src/kaggrl/v2_export.py",
        "v2_numpy_runtime.py": root / "src/kaggrl/v2_numpy_runtime.py",
        "benchmark_v2_stage1.py": root / "evaluation/benchmark_v2_stage1.py",
    }


def test_stage1_rejects_failed_parity_or_missing_metadata(tmp_path):
    _, stage0, build, parity, *_ = _fixture(tmp_path)
    pdata = json.loads(parity.read_text()); pdata["passed"] = False
    _write(parity, pdata)
    bdata = json.loads(build.read_text()); bdata["parity_report_sha256"] = _sha(parity)
    _write(build, bdata)
    report = verify_stage1(stage0, build, parity)
    assert report["accepted"] is False
    assert any("parity" in item.lower() for item in report["errors"])

    _, stage0, build, parity, *_ = _fixture(tmp_path / "missing")
    bdata = json.loads(build.read_text()); bdata.pop("parameter_count")
    _write(build, bdata)
    report = verify_stage1(stage0, build, parity)
    assert report["accepted"] is False
    assert any("parameter_count" in item for item in report["errors"])


def test_stage1_rejects_stage0_and_feature_schema_drift(tmp_path):
    _, stage0, build, parity, *_ = _fixture(tmp_path)
    bdata = json.loads(build.read_text()); bdata["stage0_marker_sha256"] = "0" * 64
    _write(build, bdata)
    report = verify_stage1(stage0, build, parity)
    assert report["accepted"] is False
    assert any("Stage 0" in item for item in report["errors"])
    _, stage0, build, parity, *_ = _fixture(tmp_path / "schema")
    pdata = json.loads(parity.read_text()); pdata["feature_schema_sha256"] = "z" * 64
    _write(parity, pdata)
    bdata = json.loads(build.read_text()); bdata["parity_report_sha256"] = _sha(parity)
    _write(build, bdata)
    report = verify_stage1(stage0, build, parity)
    assert report["accepted"] is False
    assert any("feature schema" in item.lower() for item in report["errors"])


def test_stage1_rejects_torch_import_or_source_hash_drift(tmp_path):
    _, stage0, build, parity, _, sources = _fixture(tmp_path)
    runtime = sources["v2_numpy_runtime.py"]
    runtime.write_text("import torch\n", encoding="utf-8")
    report = verify_stage1(stage0, build, parity)
    assert report["accepted"] is False
    assert any("source SHA" in item or "Torch" in item for item in report["errors"])


def test_stage1_rejects_archive_or_parity_sha_drift(tmp_path):
    _, stage0, build, parity, archive, _ = _fixture(tmp_path)
    archive.write_bytes(b"tampered")
    report = verify_stage1(stage0, build, parity)
    assert report["accepted"] is False
    assert any("archive SHA" in item for item in report["errors"])

    _, stage0, build, parity, *_ = _fixture(tmp_path / "parity")
    parity.write_text(parity.read_text() + "\n", encoding="utf-8")
    report = verify_stage1(stage0, build, parity)
    assert report["accepted"] is False
    assert any("parity report SHA" in item for item in report["errors"])
    for name, path in sources.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        body = "import numpy as np\n" if name == "v2_numpy_runtime.py" else "# source\n"
        path.write_text(body, encoding="utf-8")
    manifest = {
        "parameter_count": 1_800_000,
        "archive": "checkpoints/rl_v2_stage1/policy_init.npz",
        "archive_size_bytes": archive.stat().st_size,
        "archive_sha256": _sha(archive),
        "model_parameter_sha256": "m" * 64,
        "exported_parameter_sha256": "e" * 64,
        "feature_schema_sha256": "f" * 64,
        "stage0_marker_sha256": _sha(stage0),
        "latency": [{"hands": n, "p99_ms": 10.0, "worst_ms": 20.0}
                    for n in (0, 8, 16, 32)],
        "stress_latency": [{"hands": 64, "p99_ms": 40.0, "worst_ms": 50.0},
                           {"hands": 128, "p99_ms": 90.0, "worst_ms": 100.0}],
        "representative_p99_ms": 10.0, "latency_target_ms": 200.0,
        "latency_target_passed": True,
        "torch_free_import": {"passed": True, "returncode": 0, "stdout": "torch-free-ok"},
        "parity_report_sha256": _sha(parity), "parity_passed": True,
        "source_sha256": {name: _sha(path) for name, path in sources.items()},
    }
    build = _write_json(stage_dir / "build_manifest.json", manifest)
    return root, stage0, build, parity, archive, sources


def test_stage1_rejects_missing_stress_latency_or_torch_free_gate(tmp_path):
    _, stage0, build, parity, *_ = _fixture(tmp_path)
    bdata = json.loads(build.read_text()); bdata["stress_latency"] = [
        {"hands": 32, "p99_ms": 20.0, "worst_ms": 25.0}
    ]
    _write(build, bdata)
    report = verify_stage1(stage0, build, parity)
    assert report["accepted"] is False
    assert any("stress" in item.lower() for item in report["errors"])

    _, stage0, build, parity, *_ = _fixture(tmp_path / "torchfree")
    bdata = json.loads(build.read_text()); bdata["torch_free_import"]["passed"] = False
    _write(build, bdata)
    report = verify_stage1(stage0, build, parity)
    assert report["accepted"] is False
    assert any("Torch-free" in item for item in report["errors"])


def test_write_marker_refuses_failed_report(tmp_path):
    _, stage0, build, parity, archive, _ = _fixture(tmp_path)
    archive.write_bytes(b"tampered")
    output = tmp_path / "STAGE1_ACCEPTED.json"
    with pytest.raises(Stage1AcceptanceError):
        write_stage1_marker(stage0, build, parity, output)
    assert not output.exists()


def test_verify_stage1_accepts_current_evidence_schema(tmp_path):
    root, stage0, build, parity, archive, _ = _fixture(tmp_path)
    result = verify_stage1(stage0, build, parity)
    assert result["accepted"] is True
    assert result["artifacts"]["stage0_marker_sha256"] == _sha(stage0)
    assert result["artifacts"]["build_manifest_sha256"] == _sha(build)
    assert result["artifacts"]["parity_report_sha256"] == _sha(parity)
    assert result["artifacts"]["archive_sha256"] == _sha(archive)
    assert result["artifacts"]["exported_parameter_sha256"] == "e" * 64
    assert result["gates"]["stress_hands"] == 128


def test_verify_stage1_rejects_stage0_hash_mismatch(tmp_path):
    _, stage0, build, parity, *_ = _fixture(tmp_path)
    data = json.loads(build.read_text()); data["stage0_marker_sha256"] = "0" * 64
    _write_json(build, data)
    with pytest.raises(Stage1AcceptanceError, match="Stage 0"):
        verify_stage1(stage0, build, parity)


def test_verify_stage1_rejects_missing_128_hand_stress(tmp_path):
    _, stage0, build, parity, *_ = _fixture(tmp_path)
    data = json.loads(build.read_text())
    data["stress_latency"] = [{"hands": 64, "p99_ms": 40.0, "worst_ms": 50.0}]
    _write_json(build, data)
    with pytest.raises(Stage1AcceptanceError, match="128-hand"):
        verify_stage1(stage0, build, parity)


def test_verify_stage1_rejects_failed_parity_or_torch_free(tmp_path):
    _, stage0, build, parity, *_ = _fixture(tmp_path)
    pdata = json.loads(parity.read_text()); pdata["passed"] = False
    _write_json(parity, pdata)
    bdata = json.loads(build.read_text()); bdata["parity_report_sha256"] = _sha(parity)
    _write_json(build, bdata)
    with pytest.raises(Stage1AcceptanceError, match="parity"):
        verify_stage1(stage0, build, parity)

    _, stage0, build, parity, *_ = _fixture(tmp_path / "torch")
    bdata = json.loads(build.read_text()); bdata["torch_free_import"]["passed"] = False
    _write_json(build, bdata)
    with pytest.raises(Stage1AcceptanceError, match="Torch-free"):
        verify_stage1(stage0, build, parity)


def test_verify_stage1_rejects_missing_model_metadata(tmp_path):
    _, stage0, build, parity, *_ = _fixture(tmp_path)
    data = json.loads(build.read_text()); data.pop("parameter_count")
    _write_json(build, data)
    with pytest.raises(Stage1AcceptanceError, match="parameter_count"):
        verify_stage1(stage0, build, parity)


def test_verify_stage1_rejects_archive_or_source_drift(tmp_path):
    _, stage0, build, parity, archive, sources = _fixture(tmp_path)
    archive.write_bytes(b"tampered")
    with pytest.raises(Stage1AcceptanceError, match="archive"):
        verify_stage1(stage0, build, parity)

    _, stage0, build, parity, _, sources = _fixture(tmp_path / "source")
    sources["v2_numpy_runtime.py"].write_text("import torch\n", encoding="utf-8")
    with pytest.raises(Stage1AcceptanceError, match="source SHA|Torch"):
        verify_stage1(stage0, build, parity)


def test_write_marker_removes_stale_marker_before_failed_verification(tmp_path):
    root, stage0, build, parity, *_ = _fixture(tmp_path)
    marker = root / "checkpoints/rl_v2_stage1/STAGE1_ACCEPTED.json"
    write_stage1_marker(stage0, build, parity, marker)
    assert marker.exists()
    data = json.loads(stage0.read_text()); data["artifacts"]["dataset_sha256"] = "changed"
    _write_json(stage0, data)
    with pytest.raises(Stage1AcceptanceError):
        write_stage1_marker(stage0, build, parity, marker)
    assert not marker.exists()
