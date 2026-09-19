from dataclasses import replace
from pathlib import Path

import pytest

from training.train_v3_bc import BCV3Config, _initial_state_collapse_report


def _base():
    return BCV3Config(
        dataset_path=Path("dataset"),
        stage0_marker=Path("stage0"),
        stage1_marker=Path("stage1"),
        output_dir=Path("out"),
    )


def test_fast_validation_requires_last_epoch_selection():
    with pytest.raises(ValueError, match="last_epoch"):
        replace(
            _base(),
            validation_profile="fast",
            selection_mode="offline_legacy",
        ).validate()
    replace(
        _base(),
        validation_profile="fast",
        selection_mode="last_epoch",
    ).validate()
def test_initial_state_collapse_report_only_checks_opening_gate():
    config = replace(
        _base(),
        min_initial_market_continue_accuracy=0.95,
        min_initial_market_active_accuracy=0.80,
    )
    report = _initial_state_collapse_report(
        config,
        {
            "rows": 10,
            "market_continue_accuracy": 1.0,
            "market_active_op_accuracy": 0.9,
            "active_target_count": 10,
        },
    )
    assert report["passed"] is True
    assert report["scope"] == "initial_state_only"

    failed = _initial_state_collapse_report(
        config,
        {
            "rows": 10,
            "market_continue_accuracy": 0.5,
            "market_active_op_accuracy": 0.2,
            "active_target_count": 10,
        },
    )
    assert set(failed["failures"]) == {
        "initial_market_stop_collapse",
        "initial_market_active_collapse",
    }
