import numpy as np
import pytest

from evaluation.eval_v3_closed_loop import (
    build_v3_game_specs,
    evaluate_practical_gate,
)


def _game(seat, families, *, streak=20, schema=True, timeout=False,
          finite=True, torch_free=True):
    return {
        "seed": 99, "learner_seat": seat, "opponent": "starter",
        "statuses": ["DONE", "DONE"], "finite": finite,
        "schema_valid": schema, "timeout": timeout,
        "torch_import_free": torch_free,
        "longest_effectless_streak": streak,
        "effective_family_counts": dict(families),
        "final_money": 1000, "margin": 0, "max_hands": 2,
        "land_unlocks": 0,
    }


def _offline(gap=0.10):
    return {
        "finite": True,
        "gaps": {"farmer_op": gap, "hands_op": gap, "market_op": gap},
    }


def _matrix(families0, families1=None, **kwargs):
    families1 = families0 if families1 is None else families1
    return {"records": [
        _game(0, families0, **kwargs),
        _game(1, families1, **kwargs),
    ]}


def test_v3_game_specs_are_symmetric_and_deterministic():
    specs = build_v3_game_specs([8, 7], ["starter"])
    assert [(s.seed, s.learner_seat, s.opponent) for s in specs] == [
        (7, 0, "starter"), (7, 1, "starter"),
        (8, 0, "starter"), (8, 1, "starter"),
    ]


def test_evaluator_selects_v32_agent_for_format4(tmp_path):
    from evaluation.eval_v3_closed_loop import _agent_class_for_model
    from rollout.v3_2_agent_numpy import V32NumpyRolloutAgent
    from rollout.v3_agent_numpy import V3NumpyRolloutAgent

    v3 = tmp_path / "v3.npz"
    v32 = tmp_path / "v32.npz"
    np.savez(v3, format_version=np.asarray(3, dtype=np.int32))
    np.savez(v32, format_version=np.asarray(4, dtype=np.int32))

    assert _agent_class_for_model(v3) is V3NumpyRolloutAgent
    assert _agent_class_for_model(v32) is V32NumpyRolloutAgent


@pytest.mark.parametrize("families", [
    {"movement": 10},
    {"movement": 2, "hire": 1, "acquisition": 1},
    {"movement": 2, "acquisition": 1, "production": 1, "deposit": 1},
])
def test_practical_gate_rejects_incomplete_economic_chains(families):
    result = evaluate_practical_gate(_matrix(families), _offline())
    assert result["passed"] is False
    assert result["failures"]


@pytest.mark.parametrize("kwargs", [
    {"timeout": True},
    {"schema": False},
    {"finite": False},
    {"torch_free": False},
    {"streak": 600},
])
def test_practical_gate_rejects_runtime_integrity_failures(kwargs):
    full = {"movement": 1, "acquisition": 1, "production": 1,
            "deposit": 1, "sale": 1}
    result = evaluate_practical_gate(_matrix(full, **kwargs), _offline())
    assert result["passed"] is False


def test_practical_gate_requires_each_seat_to_complete_chain():
    full = {"movement": 1, "acquisition": 1, "service": 1,
            "deposit": 1, "sale": 1}
    broken = {"movement": 1, "acquisition": 1, "service": 1,
              "deposit": 1}
    result = evaluate_practical_gate(_matrix(full, broken), _offline())
    assert result["passed"] is False
    assert any("seat1" in item for item in result["failures"])


def test_practical_gate_does_not_require_explicit_deposit():
    # Advanced Kaggriculture auto-drops unit inventories into the shed at
    # end-of-day, so a policy can be economically complete without DROP.
    full = {"movement": 2, "acquisition": 1, "production": 1,
            "service": 1, "sale": 1}
    result = evaluate_practical_gate(_matrix(full), _offline())
    assert result["passed"] is True
    assert result["failures"] == []


def test_practical_gate_rejects_large_teacher_free_gap():
    full = {"movement": 1, "acquisition": 1, "production": 1,
            "deposit": 1, "sale": 1}
    result = evaluate_practical_gate(_matrix(full), _offline(gap=0.30))
    assert result["passed"] is False
    assert "offline_farmer_op_gap" in result["failures"]


def test_torch_free_preflight_runs_real_v3_step_in_fresh_process(tmp_path):
    import torch
    from kaggrl.v3_export import export_v3_numpy
    from kaggrl.v3_model import TemporalIntentPolicy
    from evaluation.eval_v3_closed_loop import _torch_free_preflight

    torch.manual_seed(81)
    model_path = tmp_path / "policy.npz"
    export_v3_numpy(TemporalIntentPolicy().eval(), model_path)
    assert _torch_free_preflight(model_path) is True


def test_short_smoke_rejects_pass_collapse_and_parity_failure():
    from evaluation.eval_v3_closed_loop import evaluate_short_smoke

    record = _game(0, {"acquisition": 1})
    record["action_histograms"] = {
        "farmer": {"PASS": 50, "NORTH": 0},
        "hands": {"PASS": 20, "NORTH": 0},
        "market": {"STOP_QUEUE": 49, "HIRE": 1},
    }
    result = evaluate_short_smoke([record], {"passed": False})
    assert result["passed"] is False
    assert "parity_mismatch" in result["failures"]
    assert "seat0_farmer_all_pass" in result["failures"]
    assert "seat0_missing_movement" in result["failures"]


def test_short_smoke_accepts_minimum_active_behavior():
    from evaluation.eval_v3_closed_loop import evaluate_short_smoke

    records = []
    for seat in (0, 1):
        record = _game(seat, {"movement": 2, "acquisition": 1})
        record["action_histograms"] = {
            "farmer": {"PASS": 40, "NORTH": 2},
            "hands": {"PASS": 18, "NORTH": 1},
            "market": {"STOP_QUEUE": 40, "HIRE": 2},
        }
        records.append(record)
    result = evaluate_short_smoke(records, {"passed": True})
    assert result["passed"] is True


def test_short_smoke_rejects_hire_only_without_real_acquisition():
    from evaluation.eval_v3_closed_loop import evaluate_short_smoke

    record = _game(0, {"movement": 2, "hire": 4})
    record["action_histograms"] = {
        "farmer": {"PASS": 40, "NORTH": 2},
        "hands": {"PASS": 18, "NORTH": 1},
        "market": {"STOP_QUEUE": 40, "HIRE": 4},
    }
    result = evaluate_short_smoke([record], {"passed": True})
    assert result["passed"] is False
    assert "seat0_missing_acquisition" in result["failures"]


def test_practical_gate_rejects_hire_as_substitute_for_acquisition():
    families = {
        "movement": 2,
        "hire": 4,
        "production": 1,
        "service": 1,
        "deposit": 1,
        "sale": 1,
    }
    result = evaluate_practical_gate(_matrix(families), _offline())
    assert result["passed"] is False
    assert "seat0_missing_acquisition" in result["failures"]
    assert "seat1_missing_acquisition" in result["failures"]
