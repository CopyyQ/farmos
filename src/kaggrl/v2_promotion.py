from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MINIMUM_MONEY_RATIO = 0.60
REQUIRED_FAMILIES = ("movement", "acquisition", "deposit", "sale")
PRODUCTION_FAMILIES = ("production", "service")
ECONOMIC_FAMILIES = {
    "acquisition", "hire", "purchase", "land_unlock", "production",
    "service", "harvest", "deposit", "sale",
}


@dataclass(frozen=True)
class PromotionDecision:
    accepted: bool
    reasons: tuple[str, ...]
    gates: dict[str, bool]
    measured: dict[str, Any]
    thresholds: dict[str, Any]
    evidence: dict[str, Any]


def _json_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _family_totals(records: list[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for record in records:
        for family, count in (record.get("effective_family_counts") or {}).items():
            totals[str(family)] = totals.get(str(family), 0) + int(count)
    return totals


def _median_money(records: list[dict[str, Any]]) -> float:
    if not records:
        raise ValueError("promotion evidence has no game records")
    return float(statistics.median(float(row["final_money"]) for row in records))


def _game_failures(records: list[dict[str, Any]]) -> list[str]:
    reasons = []
    for index, row in enumerate(records):
        prefix = f"game[{index}]"
        if row.get("statuses") != ["DONE", "DONE"]:
            reasons.append(f"{prefix} did not finish DONE/DONE")
        if row.get("finite") is not True:
            reasons.append(f"{prefix} finite telemetry gate failed")
        if row.get("schema_valid") is not True:
            reasons.append(f"{prefix} schema validation failed or missing")
        if row.get("timeout") is not False:
            reasons.append(f"{prefix} timeout failure or missing timeout evidence")
        families = row.get("effective_family_counts") or {}
        economy = sum(int(families.get(name, 0)) for name in ECONOMIC_FAMILIES)
        if economy <= 0:
            reasons.append(f"{prefix} is a pathological zero-economy episode")
    return reasons


def _offline_failures(report: dict[str, Any]) -> list[str]:
    reasons = []
    if report.get("finite") is not True:
        reasons.append("offline report is not finite")
    audits = report.get("audits") or {}
    if audits.get("schema_valid") is not True:
        reasons.append("offline free-running schema validation failed or missing")
    counts = audits.get("queue_class_counts") or {}
    stop = int(counts.get("STOP_QUEUE", 0) or 0)
    nop = int(counts.get("NOP_SLOT", 0) or 0)
    if stop <= 0 or nop <= 0:
        reasons.append("offline queue STOP/NOP collapse detected")
    return reasons


def evaluate_ppo_entry(candidate_summary, control_summary, offline_report) -> PromotionDecision:
    candidate_records = list(candidate_summary.get("records") or [])
    control_records = list(control_summary.get("records") or [])
    reasons = _game_failures(candidate_records)
    if not control_records:
        reasons.append("matched V17 control matrix is missing")

    family_totals = _family_totals(candidate_records)
    for family in REQUIRED_FAMILIES:
        if int(family_totals.get(family, 0)) <= 0:
            reasons.append(f"required effective-action family missing: {family}")
    if not any(int(family_totals.get(name, 0)) > 0 for name in PRODUCTION_FAMILIES):
        reasons.append("required effective-action family missing: production/service")

    candidate_median = _median_money(candidate_records) if candidate_records else float("nan")
    control_median = _median_money(control_records) if control_records else float("nan")
    matched_control_valid = bool(control_records and control_median > 0)
    money_ratio = (candidate_median / control_median) if matched_control_valid else None
    if not matched_control_valid:
        reasons.append("matched V17 control median is non-positive; gate is invalid")
    elif candidate_median < MINIMUM_MONEY_RATIO * control_median:
        reasons.append("candidate median final money is below 0.60 * matched V17 control median")

    reasons.extend(_offline_failures(offline_report))
    gates = {
        "game_integrity": not _game_failures(candidate_records),
        "required_families": all(int(family_totals.get(name, 0)) > 0 for name in REQUIRED_FAMILIES)
                             and any(int(family_totals.get(name, 0)) > 0 for name in PRODUCTION_FAMILIES),
        "matched_control_valid": matched_control_valid,
        "money_ratio": bool(matched_control_valid and money_ratio is not None
                            and money_ratio >= MINIMUM_MONEY_RATIO),
        "offline_sanity": not _offline_failures(offline_report),
    }
    measured = {
        "candidate_median_final_money": candidate_median,
        "control_median_final_money": control_median,
        "money_ratio": money_ratio,
        "effective_family_counts": family_totals,
        "candidate_games": len(candidate_records),
        "control_games": len(control_records),
    }
    evidence = {
        "candidate_summary_sha256": _json_sha(candidate_summary),
        "control_summary_sha256": _json_sha(control_summary),
        "offline_report_sha256": _json_sha(offline_report),
        "candidate_checkpoint_sha256": candidate_summary.get("model_sha256"),
        "offline_checkpoint_sha256": offline_report.get("checkpoint_sha256"),
        "dataset_sha256": offline_report.get("dataset_sha256"),
        "candidate_artifact_sha256": [row.get("artifact_sha256") for row in candidate_records],
        "control_artifact_sha256": [row.get("artifact_sha256") for row in control_records],
    }
    for key in ("stage0_marker_sha256", "stage1_marker_sha256",
                "pretrain_config_sha256", "bc_config_sha256", "evaluator_sha256"):
        value = candidate_summary.get(key) or offline_report.get(key)
        if value is not None:
            evidence[key] = value
    return PromotionDecision(
        accepted=not reasons,
        reasons=tuple(reasons), gates=gates, measured=measured,
        thresholds={"minimum_money_ratio": MINIMUM_MONEY_RATIO,
                    "required_families": list(REQUIRED_FAMILIES),
                    "production_or_service": list(PRODUCTION_FAMILIES)},
        evidence=evidence,
    )


def write_promotion_evidence(decision: PromotionDecision, path) -> Path:
    base = Path(path)
    if base.suffix.lower() == ".json":
        output = base
    else:
        output = base / ("PPO_ENTRY_ACCEPTED.json" if decision.accepted
                         else "PPO_ENTRY_REJECTED.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"promotion evidence is immutable: {output}")
    payload = asdict(decision)
    payload["reasons"] = list(decision.reasons)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output
