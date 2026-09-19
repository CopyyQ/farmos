from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class Stage1AcceptanceError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load(path: str | Path, label: str) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path)
    if not resolved.is_file():
        raise Stage1AcceptanceError(f"missing {label}: {resolved}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Stage1AcceptanceError(f"invalid {label}: {resolved}") from exc
    if not isinstance(value, dict):
        raise Stage1AcceptanceError(f"{label} must contain a JSON object")
    return resolved, value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage1AcceptanceError(message)


def _validate_stage0(stage0: dict[str, Any]) -> None:
    _require(stage0.get("accepted") is True, "Stage 0 marker is not accepted")
    counters = stage0.get("counters") or {}
    _require(isinstance(counters, dict), "Stage 0 counters are invalid")
    _require(all(int(value) == 0 for value in counters.values()),
             "Stage 0 marker contains nonzero counters")


def _project_root(build_path: Path) -> Path:
    parents = build_path.parents
    if len(parents) >= 3 and parents[0].name == "rl_v2_stage1" and parents[1].name == "checkpoints":
        return parents[2]
    return build_path.parent


def _resolve_archive(build_path: Path, value: str) -> Path:
    archive = Path(value)
    if archive.is_absolute():
        return archive
    root = _project_root(build_path)
    candidates = (root / archive, build_path.parent / archive)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _source_path(root: Path, name: str) -> Path:
    if name == "benchmark_v2_stage1.py":
        return root / "evaluation" / name
    return root / "src" / "kaggrl" / name


def _imports_torch(path: Path) -> bool:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError) as exc:
        raise Stage1AcceptanceError(f"cannot inspect runtime source: {path}") from exc
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(alias.name == "torch" or alias.name.startswith("torch.") for alias in node.names):
            return True
        if isinstance(node, ast.ImportFrom) and node.module and (node.module == "torch" or node.module.startswith("torch.")):
            return True
    return False


def _validate_parity(parity: dict[str, Any]) -> None:
    _require(parity.get("passed") is True, "Torch/NumPy parity gate failed")
    _require(int(parity.get("action_mismatches", 1)) == 0,
             "Torch/NumPy parity contains action mismatches")
    rows = parity.get("rows") or []
    counts = {int(row.get("hands", -1)) for row in rows if isinstance(row, dict)}
    _require({0, 17, 32}.issubset(counts), "dynamic hand parity coverage is incomplete")
    _require(all(bool(row.get("action_equal", False)) for row in rows),
             "Torch/NumPy parity contains unequal actions")
    tolerance = float(parity.get("tolerance", 0.0) or 0.0)
    _require(0.0 < tolerance <= 3e-5, "parity tolerance is invalid")
    error_keys = (
        "max_fused_abs", "max_h_abs", "max_c_abs", "max_intent_abs",
        "max_terminal_money_abs", "max_terminal_margin_abs",
    )
    for key in error_keys:
        if key in parity:
            _require(float(parity[key]) <= tolerance, f"parity error exceeds tolerance: {key}")


def _validate_build(build: dict[str, Any]) -> tuple[int, int]:
    required = (
        "parameter_count", "archive", "archive_size_bytes", "archive_sha256",
        "model_parameter_sha256", "exported_parameter_sha256", "feature_schema_sha256",
        "stage0_marker_sha256", "parity_report_sha256", "parity_passed",
        "latency_target_ms", "representative_p99_ms", "latency_target_passed",
        "stress_latency", "torch_free_import", "source_sha256",
    )
    for key in required:
        _require(key in build, f"build manifest missing {key}")
    params = int(build["parameter_count"])
    _require(0 < params < 2_500_000,
             "parameter_count is outside the Stage 1 model budget")
    _require(int(build["archive_size_bytes"]) > 0, "archive_size_bytes must be positive")
    target_ms = float(build["latency_target_ms"])
    p99_ms = float(build["representative_p99_ms"])
    _require(build.get("latency_target_passed") is True and p99_ms < target_ms <= 200.0,
             "latency p99 gate failed")
    representative = build.get("latency") or []
    _require(bool(representative), "representative latency evidence is missing")
    _require(max(float(row.get("worst_ms", 1e9)) for row in representative) < 500.0,
             "latency worst-case gate failed")
    stress = [row for row in (build.get("stress_latency") or []) if isinstance(row, dict)]
    stress_counts = {int(row.get("hands", -1)) for row in stress}
    _require(64 in stress_counts and 128 in stress_counts,
             "dynamic stress must include both 64-hand and 128-hand cases")
    _require(max(float(row.get("worst_ms", 1e9)) for row in stress) < 500.0,
             "stress latency worst-case gate failed")
    torch_free = build.get("torch_free_import") or {}
    _require(torch_free.get("passed") is True and int(torch_free.get("returncode", 1)) == 0,
             "Torch-free import gate failed")
    _require(build.get("parity_passed") is True, "build manifest parity gate failed")
    sources = build.get("source_sha256")
    _require(isinstance(sources, dict) and sources, "source_sha256 metadata is invalid")
    for key in ("v2_export.py", "v2_numpy_runtime.py", "v2_model.py"):
        _require(bool(sources.get(key)), f"source_sha256 missing {key}")
    return params, max(stress_counts)


def verify_stage1(stage0_marker: str | Path, build_manifest: str | Path,
                  parity_report: str | Path) -> dict[str, Any]:
    stage0_path, stage0 = _load(stage0_marker, "Stage 0 marker")
    build_path, build = _load(build_manifest, "build manifest")
    parity_path, parity = _load(parity_report, "parity report")
    _validate_stage0(stage0)
    params, stress_hands = _validate_build(build)
    _validate_parity(parity)

    stage0_sha = _sha256(stage0_path)
    parity_sha = _sha256(parity_path)
    _require(build.get("stage0_marker_sha256") == stage0_sha,
             "Stage 0 hash does not match build evidence")
    _require(build.get("parity_report_sha256") == parity_sha,
             "parity report SHA does not match build evidence")
    _require(parity.get("archive_sha256") == build.get("archive_sha256"),
             "archive hash mismatch between parity and build evidence")
    _require(parity.get("feature_schema_sha256") == build.get("feature_schema_sha256"),
             "feature schema hash mismatch between parity and build evidence")

    root = _project_root(build_path)
    archive = _resolve_archive(build_path, str(build["archive"]))
    _require(archive.is_file(), f"missing Stage 1 archive: {archive}")
    _require(archive.stat().st_size == int(build["archive_size_bytes"]),
             "archive size does not match build evidence")
    _require(_sha256(archive) == build["archive_sha256"],
             "archive SHA does not match build evidence")

    sources = build["source_sha256"]
    for name, expected in sources.items():
        source_path = _source_path(root, str(name))
        _require(source_path.is_file(), f"missing bound source file: {source_path}")
        _require(_sha256(source_path) == expected,
                 f"source SHA mismatch: {name}")
    runtime_source = _source_path(root, "v2_numpy_runtime.py")
    _require(not _imports_torch(runtime_source),
             "Torch import detected in NumPy runtime source")

    return {
        "accepted": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "errors": [],
        "artifacts": {
            "stage0_marker_sha256": stage0_sha,
            "build_manifest_sha256": _sha256(build_path),
            "parity_report_sha256": parity_sha,
            "archive_sha256": build["archive_sha256"],
            "model_parameter_sha256": build["model_parameter_sha256"],
            "exported_parameter_sha256": build["exported_parameter_sha256"],
            "feature_schema_sha256": build["feature_schema_sha256"],
            "source_sha256": dict(sources),
        },
        "gates": {
            "torch_numpy_parity": True,
            "torch_free_import_graph": True,
            "latency": True,
            "stress_hands": stress_hands,
            "parameter_count": params,
            "representative_p99_ms": float(build["representative_p99_ms"]),
        },
    }


def write_stage1_marker(stage0_marker: str | Path, build_manifest: str | Path,
                        parity_report: str | Path, output_path: str | Path) -> dict[str, Any]:
    output = Path(output_path)
    if output.exists():
        output.unlink()
    marker = verify_stage1(stage0_marker, build_manifest, parity_report)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    return marker


def main(argv=None) -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Verify and accept RL v2 Stage 1 evidence")
    parser.add_argument("--stage0", default=root / "data/top_tier/live_v2/manifests/STAGE0_ACCEPTED.json")
    parser.add_argument("--build", default=root / "checkpoints/rl_v2_stage1/build_manifest.json")
    parser.add_argument("--parity", default=root / "checkpoints/rl_v2_stage1/parity_report.json")
    parser.add_argument("--output", default=root / "checkpoints/rl_v2_stage1/STAGE1_ACCEPTED.json")
    args = parser.parse_args(argv)
    try:
        marker = write_stage1_marker(args.stage0, args.build, args.parity, args.output)
    except Stage1AcceptanceError as exc:
        print(f"Stage 1 verification failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(marker, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
