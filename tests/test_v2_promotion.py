import json
from pathlib import Path

from kaggrl.v2_promotion import evaluate_ppo_entry, write_promotion_evidence


def _record(money=1000, *, families=None, finite=True, statuses=None,
            schema_valid=True, timeout=False):
    return {
        "final_money": money,
        "margin": 0,
        "finite": finite,
        "statuses": ["DONE", "DONE"] if statuses is None else statuses,
        "schema_valid": schema_valid,
        "timeout": timeout,
        "effective_family_counts": dict(families or {}),
        "artifact_sha256": "a" * 64,
    }


def _candidate(money=(1000, 1100), families=None):
    families = families or {
        "movement": 2, "acquisition": 1, "production": 1,
        "deposit": 1, "sale": 1,
    }
    records = [_record(value, families=families) for value in money]
    return {"model_sha256": "c" * 64, "records": records,
            "summary": {"median_final_money": sum(money) / len(money)}}


def _control(money=(1200, 1000)):
    return {"control": "v17", "records": [_record(value) for value in money],
            "summary": {"median_final_money": sum(money) / len(money)}}


def _offline():
    return {
        "finite": True,
        "checkpoint_sha256": "b" * 64,
        "dataset_sha256": "d" * 64,
        "semantic": {"farmer_semantic_exact": .5},
        "audits": {
            "stop_nop_confusions": 0,
            "market_order_position_accuracy": .5,
            "queue_class_counts": {"STOP_QUEUE": 10, "NOP_SLOT": 4},
            "schema_valid": True,
        },
    }


def test_ppo_entry_accepts_only_when_all_mandatory_gates_pass():
    decision = evaluate_ppo_entry(_candidate(), _control(), _offline())
    assert decision.accepted is True
    assert decision.reasons == ()
    assert decision.measured["candidate_median_final_money"] == 1050.0
    assert decision.measured["control_median_final_money"] == 1100.0
    assert decision.measured["money_ratio"] > .60


def test_ppo_entry_rejects_failures_zero_economy_and_missing_family():
    candidate = _candidate()
    candidate["records"][0]["finite"] = False
    candidate["records"][1]["timeout"] = True
    decision = evaluate_ppo_entry(candidate, _control(), _offline())
    assert decision.accepted is False
    assert any("finite" in reason.lower() for reason in decision.reasons)
    assert any("timeout" in reason.lower() for reason in decision.reasons)

    zero = _candidate(money=(3000, 3000), families={"movement": 4})
    decision = evaluate_ppo_entry(zero, _control(), _offline())
    assert decision.accepted is False
    assert any("zero-economy" in reason.lower() for reason in decision.reasons)
    assert any("acquisition" in reason.lower() for reason in decision.reasons)
    assert any("deposit" in reason.lower() for reason in decision.reasons)
    assert any("sale" in reason.lower() for reason in decision.reasons)


def test_ppo_entry_rejects_money_ratio_and_invalid_nonpositive_control():
    decision = evaluate_ppo_entry(_candidate(money=(500, 600)), _control((1000, 1000)), _offline())
    assert decision.accepted is False
    assert any("0.60" in reason for reason in decision.reasons)

    decision = evaluate_ppo_entry(_candidate(), _control((0, -10)), _offline())
    assert decision.accepted is False
    assert decision.gates["matched_control_valid"] is False
    assert any("non-positive" in reason.lower() for reason in decision.reasons)


def test_ppo_entry_requires_offline_finite_schema_and_noncollapsed_queue_classes():
    offline = _offline(); offline["finite"] = False
    decision = evaluate_ppo_entry(_candidate(), _control(), offline)
    assert decision.accepted is False
    assert any("offline" in reason.lower() and "finite" in reason.lower() for reason in decision.reasons)

    offline = _offline(); offline["audits"]["schema_valid"] = False
    decision = evaluate_ppo_entry(_candidate(), _control(), offline)
    assert decision.accepted is False
    assert any("schema" in reason.lower() for reason in decision.reasons)

    offline = _offline(); offline["audits"]["queue_class_counts"] = {"STOP_QUEUE": 14, "NOP_SLOT": 0}
    decision = evaluate_ppo_entry(_candidate(), _control(), offline)
    assert decision.accepted is False
    assert any("queue" in reason.lower() and "collapse" in reason.lower() for reason in decision.reasons)


def test_write_promotion_evidence_uses_accepted_or_rejected_name_and_is_immutable(tmp_path):
    accepted = evaluate_ppo_entry(_candidate(), _control(), _offline())
    path = write_promotion_evidence(accepted, tmp_path)
    assert path.name == "PPO_ENTRY_ACCEPTED.json"
    payload = json.loads(path.read_text())
    assert payload["accepted"] is True
    assert payload["thresholds"]["minimum_money_ratio"] == .60
    try:
        write_promotion_evidence(accepted, tmp_path)
    except FileExistsError:
        pass
    else:
        raise AssertionError("promotion evidence must be immutable")


def test_ppo_entry_is_fail_closed_when_game_integrity_fields_are_missing():
    candidate = _candidate()
    candidate["records"][0].pop("schema_valid")
    candidate["records"][1].pop("timeout")
    decision = evaluate_ppo_entry(candidate, _control(), _offline())
    assert decision.accepted is False
    assert any("schema" in reason.lower() for reason in decision.reasons)
    assert any("timeout" in reason.lower() for reason in decision.reasons)
