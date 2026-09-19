from kaggrl.v3_phase import PhaseLabel, derive_phase, derive_transition_target


def _unit(op="PASS"):
    return {"op": op, "item": None, "quantity": None, "raw": [op]}


def _market(op="STOP_QUEUE"):
    if op in {"STOP_QUEUE", "NOP_SLOT"}:
        return {"kind": op, "op": None, "item": None, "quantity": None, "raw": []}
    return {"kind": "ORDER", "op": op, "item": None, "quantity": 1, "raw": [op]}


def _joint(farmer="PASS", hands=(), market=("STOP_QUEUE",)):
    return {
        "farmer": _unit(farmer),
        "hands": [_unit(op) for op in hands],
        "market": [_market(op) for op in market],
    }


def test_phase_derivation_covers_frozen_vocabulary():
    fixtures = [
        (_joint(market=("BUY_SEED", "STOP_QUEUE")), {}, PhaseLabel.ACQUIRE),
        (_joint(farmer="NORTH"), {}, PhaseLabel.MOVE_TO_TARGET),
        (_joint(farmer="PLANT"), {}, PhaseLabel.PRODUCE),
        (_joint(farmer="WATER"), {}, PhaseLabel.MAINTAIN),
        (_joint(farmer="HARVEST"), {}, PhaseLabel.HARVEST),
        (_joint(farmer="DROP"), {}, PhaseLabel.DEPOSIT),
        (_joint(market=("SELL", "STOP_QUEUE")), {}, PhaseLabel.SELL),
        (_joint(market=("HIRE", "STOP_QUEUE")), {}, PhaseLabel.HIRE),
        (_joint(market=("BUY_LAND", "STOP_QUEUE")), {}, PhaseLabel.EXPAND),
        (_joint(), {}, PhaseLabel.WAIT),
        (_joint(), {"recovery": True}, PhaseLabel.RECOVER),
    ]
    for action, effect, expected in fixtures:
        assert derive_phase({}, action, effect) == expected


def test_phase_derivation_marks_multi_phase_joint_action_unknown():
    action = _joint(farmer="NORTH", market=("BUY_SEED", "STOP_QUEUE"))
    assert derive_phase({}, action, {}) == PhaseLabel.UNKNOWN


def test_phase_derivation_does_not_use_future_or_final_outcome_fields():
    action = _joint(farmer="PLANT")
    a = derive_phase({"final_margin": -999999}, action, {"terminal_result": -1})
    b = derive_phase({"final_margin": 999999}, action, {"terminal_result": 1})
    assert a == b == PhaseLabel.PRODUCE


def test_phase_transition_target_masks_unknown():
    assert derive_transition_target(PhaseLabel.ACQUIRE, PhaseLabel.ACQUIRE) is False
    assert derive_transition_target(PhaseLabel.ACQUIRE, PhaseLabel.MOVE_TO_TARGET) is True
    assert derive_transition_target(PhaseLabel.UNKNOWN, PhaseLabel.WAIT) is None
    assert derive_transition_target(PhaseLabel.WAIT, PhaseLabel.UNKNOWN) is None


def test_phase_sidecar_builder_and_audit_are_deterministic(tmp_path):
    from test_v2_training_data import _dataset, _row
    from training.build_v3_phase_labels import build_phase_sidecar
    from evaluation.audit_v3_phase_labels import audit_phase_sidecar
    import pyarrow.parquet as pq

    rows = [_row(91, step) for step in range(3)]
    rows[0]["canonical_action"] = {
        "farmer": _unit(), "hands": [_unit()],
        "market": [_market("BUY_SEED"), _market()],
    }
    rows[1]["canonical_action"] = {
        "farmer": _unit("NORTH"), "hands": [_unit()], "market": [_market()],
    }
    rows[2]["canonical_action"] = {
        "farmer": _unit(), "hands": [_unit()], "market": [_market()],
    }
    dataset = _dataset(tmp_path / "transitions.parquet", rows)
    out = tmp_path / "phase_labels.parquet"

    first = build_phase_sidecar(dataset, split="train", out_path=out)
    second = build_phase_sidecar(dataset, split="train", out_path=out)
    assert first["sha256"] == second["sha256"]
    table = pq.read_table(out).to_pylist()
    assert [row["intents"] for row in table] == [
        ["ACQUIRE"], ["MOVE_TO_TARGET"], ["WAIT"],
    ]
    assert [row["single_phase"] for row in table] == [
        "ACQUIRE", "MOVE_TO_TARGET", "WAIT",
    ]
    assert [row["transition_count"] for row in table] == [2, 2, None]

    audit = audit_phase_sidecar(out)
    assert audit["malformed_rows"] == 0
    assert audit["rows"] == 3
    assert audit["intent_counts"]["UNKNOWN"] == 0
    assert audit["cardinality_counts"] == {"1": 3}
    assert len(audit["sha256"]) == 64


def test_phase_candidates_preserve_concurrent_joint_intents():
    from kaggrl.v3_phase import derive_phase_candidates

    action = _joint(
        farmer="NORTH",
        hands=("WATER", "HARVEST"),
        market=("BUY_SEED", "SELL", "STOP_QUEUE"),
    )
    assert derive_phase_candidates(action, {}) == (
        PhaseLabel.ACQUIRE,
        PhaseLabel.MOVE_TO_TARGET,
        PhaseLabel.MAINTAIN,
        PhaseLabel.HARVEST,
        PhaseLabel.SELL,
    )


def test_concurrent_plan_set_preserves_simultaneous_worker_and_market_intents():
    from kaggrl.v3_phase import derive_plan_set

    action = _joint(
        farmer="NORTH",
        hands=("WATER", "HARVEST"),
        market=("BUY_SEED", "SELL", "STOP_QUEUE"),
    )
    plans = derive_plan_set({}, action, {})
    assert {phase.value for phase in plans} == {
        "ACQUIRE", "MOVE_TO_TARGET", "MAINTAIN", "HARVEST", "SELL",
    }


def test_wait_is_empty_plan_set_and_recovery_is_additive():
    from kaggrl.v3_phase import derive_plan_set

    assert derive_plan_set({}, _joint(), {}) == frozenset()
    plans = derive_plan_set(
        {}, _joint(farmer="NORTH"), {"recovery": True},
    )
    assert {phase.value for phase in plans} == {"MOVE_TO_TARGET", "RECOVER"}


def test_multi_intent_sidecar_retains_concurrent_targets(tmp_path):
    from test_v2_training_data import _dataset, _row
    from training.build_v3_phase_labels import build_plan_sidecar
    from evaluation.audit_v3_phase_labels import audit_plan_sidecar
    import pyarrow.parquet as pq

    rows = [_row(101, step) for step in range(2)]
    rows[0]["canonical_action"] = {
        "farmer": _unit("NORTH"),
        "hands": [_unit("WATER")],
        "market": [_market("BUY_SEED"), _market()],
    }
    rows[1]["canonical_action"] = {
        "farmer": _unit("HARVEST"),
        "hands": [_unit("DROP")],
        "market": [_market("SELL"), _market()],
    }
    dataset = _dataset(tmp_path / "transitions.parquet", rows)
    out = tmp_path / "plan_labels.parquet"
    result = build_plan_sidecar(dataset, split="train", out_path=out)
    table = pq.read_table(out).to_pylist()
    assert table[0]["active_plans"] == [
        "ACQUIRE", "MOVE_TO_TARGET", "MAINTAIN",
    ]
    assert table[0]["next_active_plans"] == [
        "HARVEST", "DEPOSIT", "SELL",
    ]
    assert table[1]["next_active_plans"] == []
    audit = audit_plan_sidecar(out)
    assert audit["malformed_rows"] == 0
    assert audit["rows"] == 2
    assert audit["multi_intent_rows"] == 2
    assert result["sha256"] == audit["sha256"]
