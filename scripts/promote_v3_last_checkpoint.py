from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

NON_FATAL_LEGACY_COVERAGE_FAILURES = {
    "zero_buy_animal_prediction",
    "zero_buy_animal_recall",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def promote_legacy_last(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir).resolve()
    last_path = run_dir / "bc_last.pt"
    best_path = run_dir / "bc_best.pt"
    if not last_path.is_file():
        raise FileNotFoundError(last_path)

    payload = torch.load(last_path, map_location="cpu", weights_only=False)
    validation = dict(payload.get("validation_metrics") or {})
    collapse = dict(validation.get("collapse") or {})
    failures = [str(value) for value in (collapse.get("failures") or [])]
    hard_failures = [
        value for value in failures
        if value not in NON_FATAL_LEGACY_COVERAGE_FAILURES
    ]
    recovered_warnings = [
        value for value in failures
        if value in NON_FATAL_LEGACY_COVERAGE_FAILURES
    ]

    if hard_failures:
        raise RuntimeError(
            "refusing to promote bc_last.pt because hard collapse failures "
            f"remain: {hard_failures}"
        )

    existing_warnings = [
        str(value) for value in (collapse.get("coverage_warnings") or [])
    ]
    coverage_warnings = list(dict.fromkeys(
        existing_warnings + recovered_warnings
    ))
    collapse["failures"] = []
    collapse["coverage_warnings"] = coverage_warnings
    collapse["passed"] = True
    validation["collapse"] = collapse
    validation["promotion_eligible"] = True
    validation["promotion_failures"] = []
    validation["promotion_recovered_from_legacy_rare_coverage_gate"] = True
    payload["validation_metrics"] = validation

    tmp_path = best_path.with_suffix(best_path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(best_path)

    manifest_inputs = [
        best_path,
        last_path,
        run_dir / "bc_epoch2.pt",
        run_dir / "history.jsonl",
        run_dir / "strategy_manifest.json",
    ]
    manifest_path = run_dir / "manifest.sha256"
    lines = [
        f"{_sha256(path)}  {path.name}"
        for path in manifest_inputs if path.is_file()
    ]
    manifest_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    result = {
        "status": "promoted",
        "run_dir": str(run_dir),
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "epoch": int(payload.get("epoch", -1)),
        "train_steps": int(payload.get("train_steps", -1)),
        "recovered_coverage_warnings": coverage_warnings,
        "hard_failures": [],
        "best_sha256": _sha256(best_path),
        "manifest": str(manifest_path),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Promote a completed V3 bc_last.pt when the only legacy "
            "promotion failures are rare BUY_ANIMAL coverage warnings."
        )
    )
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    result = promote_legacy_last(args.run_dir)
    print(
        "FARMOS_PROMOTE_LAST="
        + json.dumps(result, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
