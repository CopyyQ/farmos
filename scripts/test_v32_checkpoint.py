from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from evaluation.audit_v3_2_pre_t4 import _clean_action, _step0_rows, _torch_action
from kaggrl.v2_tensorize import collate_transitions
from kaggrl.v3_2_export import export_v3_2_numpy
from kaggrl.v3_2_model import TemporalIntentPolicyV32
from kaggrl.v3_2_numpy_runtime import V32NumpyPolicy
from kaggrl.v3_2_schema import ARCHITECTURE_VERSION


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_dataset(checkpoint: Path, payload: dict) -> Path:
    configured = Path(str((payload.get("config") or {}).get("dataset_path", "")))
    candidates = []
    if configured:
        candidates.append(configured)
    candidates.extend([
        ROOT / "data/top_tier/live_v2/transitions.parquet",
        checkpoint.parents[2] / "data/top_tier/live_v2/transitions.parquet"
        if len(checkpoint.parents) >= 3 else ROOT / "__missing__",
    ])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "could not resolve transitions.parquet; checked: "
        + ", ".join(str(path) for path in candidates)
    )


def _load_v32_checkpoint(checkpoint: Path):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("architecture_version") != ARCHITECTURE_VERSION:
        raise RuntimeError(
            "checkpoint is not V3.2: "
            f"{payload.get('architecture_version')!r}"
        )
    manifest = payload.get("strategy_manifest") or {}
    slot_to_team = tuple(int(value) for value in manifest.get("slot_to_team") or ())
    if not slot_to_team:
        raise RuntimeError("checkpoint has no strategy_manifest.slot_to_team")
    model = TemporalIntentPolicyV32(strategy_count=len(slot_to_team)).eval()
    model.load_state_dict(payload["model_state"], strict=True)
    return payload, model, slot_to_team


def _parity_for_slot(
    model: TemporalIntentPolicyV32,
    runtime: V32NumpyPolicy,
    row: dict,
    slot: int,
    seed: int,
) -> dict:
    batch = collate_transitions([row])
    strategy_slots = torch.tensor([int(slot)], dtype=torch.long)
    with torch.no_grad():
        torch_trace = model.trace_sample_step(
            batch,
            None,
            np.random.default_rng(int(seed)),
            deterministic=True,
            strategy_slots=strategy_slots,
        )
    numpy_trace = runtime.trace_step(
        batch.structured_states[0],
        {},
        batch.previous_actions[0],
        None,
        np.random.default_rng(int(seed)),
        deterministic=True,
        strategy_slot=int(slot),
    )

    torch_action = _clean_action(_torch_action(torch_trace["output"].rows[0]))
    numpy_action = _clean_action(numpy_trace["canonical_action"])
    left = torch_trace["decisions"]
    right = numpy_trace["decisions"]
    actor_match = (
        [item["actor"] for item in left]
        == [item["actor"] for item in right]
    )
    op_vocab_match = (
        len(left) == len(right)
        and all(a["ops"] == b["ops"] for a, b in zip(left, right))
    )
    mask_match = (
        len(left) == len(right)
        and all(a["legal_mask"] == b["legal_mask"] for a, b in zip(left, right))
    )
    chosen_match = (
        len(left) == len(right)
        and all(a["chosen_op"] == b["chosen_op"] for a, b in zip(left, right))
    )
    max_logit_error = 0.0
    if len(left) != len(right):
        max_logit_error = float("inf")
    else:
        for a, b in zip(left, right):
            lval = np.asarray(a["raw_logits"], dtype=np.float32)
            rval = np.asarray(b["raw_logits"], dtype=np.float32)
            if lval.shape != rval.shape:
                max_logit_error = float("inf")
                break
            if lval.size:
                max_logit_error = max(
                    max_logit_error,
                    float(np.max(np.abs(lval - rval))),
                )

    action_match = torch_action == numpy_action
    passed = bool(
        action_match
        and actor_match
        and op_vocab_match
        and mask_match
        and chosen_match
        and max_logit_error <= 3e-5
    )
    return {
        "passed": passed,
        "action_match": bool(action_match),
        "actor_match": bool(actor_match),
        "op_vocab_match": bool(op_vocab_match),
        "mask_match": bool(mask_match),
        "chosen_match": bool(chosen_match),
        "max_logit_error": float(max_logit_error),
        "torch_action": torch_action,
        "numpy_action": numpy_action,
    }


def _family_ok(families: dict, *names: str) -> bool:
    return any(int(families.get(name, 0) or 0) > 0 for name in names)


def _game_gate(records: list[dict]) -> dict:
    technical_failures = []
    practical_failures = []
    for row in records:
        seat = int(row.get("learner_seat", -1))
        prefix = f"seat{seat}"
        if row.get("statuses") != ["DONE", "DONE"]:
            technical_failures.append(f"{prefix}_not_full_done")
        if row.get("finite") is not True:
            technical_failures.append(f"{prefix}_non_finite")
        if row.get("schema_valid") is not True:
            technical_failures.append(f"{prefix}_invalid_schema")
        if row.get("timeout") is not False:
            technical_failures.append(f"{prefix}_timeout")
        if row.get("torch_import_free") is not True:
            technical_failures.append(f"{prefix}_torch_import")

        hist = row.get("action_histograms") or {}
        farmer = hist.get("farmer") or {}
        hands = hist.get("hands") or {}
        farmer_nonpass = sum(
            int(value) for key, value in farmer.items() if key != "PASS"
        )
        hand_total = sum(int(value) for value in hands.values())
        hand_nonpass = sum(
            int(value) for key, value in hands.items() if key != "PASS"
        )
        if farmer_nonpass <= 0:
            practical_failures.append(f"{prefix}_farmer_all_pass")
        if hand_total > 0 and hand_nonpass <= 0:
            practical_failures.append(f"{prefix}_hands_all_pass")

        families = row.get("effective_family_counts") or {}
        if not _family_ok(families, "movement"):
            practical_failures.append(f"{prefix}_missing_movement")
        if not _family_ok(families, "acquisition"):
            practical_failures.append(f"{prefix}_missing_acquisition")
        if not _family_ok(families, "production", "service"):
            practical_failures.append(f"{prefix}_missing_production_service")
        if not _family_ok(families, "deposit"):
            practical_failures.append(f"{prefix}_missing_deposit")
        if not _family_ok(families, "sale"):
            practical_failures.append(f"{prefix}_missing_sale")
        if int(row.get("longest_effectless_streak", 10**9)) >= 600:
            practical_failures.append(f"{prefix}_effectless_streak")

    seats = {int(row.get("learner_seat", -1)) for row in records}
    if seats != {0, 1}:
        technical_failures.append("missing_symmetric_learner_seats")
    technical_failures = sorted(set(technical_failures))
    practical_failures = sorted(set(practical_failures))
    return {
        "technical_passed": not technical_failures,
        "technical_failures": technical_failures,
        "practical_passed": not technical_failures and not practical_failures,
        "practical_failures": practical_failures,
    }


def _validation_gate(payload: dict) -> dict:
    config = payload.get("config") or {}
    validation = payload.get("validation_metrics") or {}
    initial = validation.get("initial_state") or {}
    collapse = validation.get("collapse") or {}
    continue_acc = initial.get("market_continue_accuracy")
    active_acc = initial.get("market_active_op_accuracy")
    min_continue = float(config.get("min_initial_market_continue_accuracy", 0.95))
    min_active = float(config.get("min_initial_market_active_accuracy", 0.80))
    failures = []
    if continue_acc is None or float(continue_acc) < min_continue:
        failures.append("market_continue_accuracy")
    if active_acc is None or float(active_acc) < min_active:
        failures.append("market_active_op_accuracy")
    if collapse and collapse.get("passed") is False:
        failures.extend(
            f"collapse:{item}" for item in (collapse.get("failures") or [])
        )
    return {
        "passed": not failures,
        "failures": sorted(set(failures)),
        "market_continue_accuracy": continue_acc,
        "market_active_op_accuracy": active_acc,
        "min_market_continue_accuracy": min_continue,
        "min_market_active_accuracy": min_active,
        "promotion_eligible": validation.get("promotion_eligible"),
        "teacher_forced_loss": validation.get("teacher_forced_loss"),
    }


def test_v32_checkpoint(
    checkpoint: str | Path,
    *,
    slot: int | str = 0,
    opponent: str = "starter",
    seed: int = 20260919,
    output_dir: str | Path | None = None,
) -> dict:
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    payload, model, slot_to_team = _load_v32_checkpoint(checkpoint)
    dataset = _resolve_dataset(checkpoint, payload)
    step0_rows = _step0_rows(dataset)
    if slot == "all":
        slots = list(range(len(slot_to_team)))
    else:
        slots = [int(slot)]
    for current in slots:
        if not 0 <= int(current) < len(slot_to_team):
            raise ValueError(
                f"strategy slot {current} outside [0, {len(slot_to_team) - 1}]"
            )

    if output_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output = checkpoint.parent / f"test_epoch{payload.get('epoch', 'x')}_{stamp}"
    else:
        output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)

    from evaluation.eval_v3_closed_loop import (
        build_v3_game_specs,
        run_v3_game_matrix,
    )

    validation_gate = _validation_gate(payload)
    slot_reports = []
    for current in slots:
        team_id = int(slot_to_team[current])
        if team_id not in step0_rows:
            raise RuntimeError(
                f"validation split has no step-0 row for strategy team {team_id}"
            )
        export_path = output / f"slot_{current:02d}_policy.npz"
        export_v3_2_numpy(
            model,
            export_path,
            default_strategy_slot=current,
        )
        runtime = V32NumpyPolicy.load(export_path)
        parity = _parity_for_slot(
            model,
            runtime,
            step0_rows[team_id],
            current,
            int(seed) + current,
        )

        game_dir = output / f"slot_{current:02d}_games"
        specs = build_v3_game_specs([int(seed)], [str(opponent)])
        matrix = run_v3_game_matrix(export_path, specs, game_dir)
        game_gate = _game_gate(list(matrix.get("records") or []))
        slot_reports.append({
            "slot": int(current),
            "team_id": team_id,
            "export": str(export_path),
            "export_sha256": _sha256(export_path),
            "parity": parity,
            "game_gate": game_gate,
            "game_summary": matrix.get("summary"),
            "records": matrix.get("records"),
            "can_enter_match": bool(
                parity["passed"] and game_gate["technical_passed"]
            ),
            "practical_ready": bool(
                parity["passed"]
                and validation_gate["passed"]
                and game_gate["practical_passed"]
            ),
        })

    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "architecture_version": payload.get("architecture_version"),
        "epoch": int(payload.get("epoch", -1)),
        "train_steps": int(payload.get("train_steps", -1)),
        "strategy_count": len(slot_to_team),
        "tested_slots": slots,
        "opponent": str(opponent),
        "seed": int(seed),
        "validation_gate": validation_gate,
        "last_train_metrics": payload.get("last_train_metrics"),
        "slots": slot_reports,
        "can_enter_match": bool(
            slot_reports and all(row["can_enter_match"] for row in slot_reports)
        ),
        "practical_ready": bool(
            slot_reports and all(row["practical_ready"] for row in slot_reports)
        ),
    }
    report_path = output / "checkpoint_test_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "FARMOS_CHECKPOINT_TEST="
        + json.dumps({
            "report": str(report_path),
            "epoch": report["epoch"],
            "train_steps": report["train_steps"],
            "tested_slots": report["tested_slots"],
            "validation_passed": validation_gate["passed"],
            "can_enter_match": report["can_enter_match"],
            "practical_ready": report["practical_ready"],
            "slot_results": [
                {
                    "slot": row["slot"],
                    "team_id": row["team_id"],
                    "parity": row["parity"]["passed"],
                    "technical": row["game_gate"]["technical_passed"],
                    "practical": row["game_gate"]["practical_passed"],
                    "can_enter_match": row["can_enter_match"],
                    "practical_ready": row["practical_ready"],
                }
                for row in slot_reports
            ],
        }, sort_keys=True),
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export and live-test a V3.2 BC checkpoint."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--slot",
        default="0",
        help="strategy slot integer, or 'all'",
    )
    parser.add_argument("--opponent", default="starter")
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    slot = args.slot if args.slot == "all" else int(args.slot)
    report = test_v32_checkpoint(
        args.checkpoint,
        slot=slot,
        opponent=args.opponent,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print(json.dumps({
        "can_enter_match": report["can_enter_match"],
        "practical_ready": report["practical_ready"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
