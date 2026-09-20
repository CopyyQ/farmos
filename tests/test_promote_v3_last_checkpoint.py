from pathlib import Path

import pytest
import torch

from scripts.promote_v3_last_checkpoint import promote_legacy_last


def _write_checkpoint(path: Path, failures: list[str]):
    torch.save(
        {
            "epoch": 6,
            "train_steps": 620,
            "validation_metrics": {
                "collapse": {
                    "passed": False,
                    "failures": failures,
                },
                "promotion_eligible": False,
                "promotion_failures": failures,
            },
        },
        path,
    )


def test_promote_legacy_last_accepts_only_rare_buy_animal_failures(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_checkpoint(
        run_dir / "bc_last.pt",
        ["zero_buy_animal_prediction", "zero_buy_animal_recall"],
    )
    (run_dir / "history.jsonl").write_text("{}\n", encoding="utf-8")

    result = promote_legacy_last(run_dir)

    assert result["status"] == "promoted"
    best = torch.load(
        run_dir / "bc_best.pt", map_location="cpu", weights_only=False
    )
    validation = best["validation_metrics"]
    assert validation["promotion_eligible"] is True
    assert validation["promotion_failures"] == []
    assert validation["collapse"]["passed"] is True
    assert validation["collapse"]["failures"] == []
    assert validation["collapse"]["coverage_warnings"] == [
        "zero_buy_animal_prediction",
        "zero_buy_animal_recall",
    ]
    assert (run_dir / "manifest.sha256").is_file()


def test_promote_legacy_last_rejects_real_collapse_failure(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_checkpoint(
        run_dir / "bc_last.pt",
        ["farmer_pass_collapse", "zero_buy_animal_prediction"],
    )

    with pytest.raises(RuntimeError, match="hard collapse failures"):
        promote_legacy_last(run_dir)

    assert not (run_dir / "bc_best.pt").exists()
