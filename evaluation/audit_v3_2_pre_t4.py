from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from kaggrl.v2_export import model_parameter_sha256
from kaggrl.v2_tensorize import collate_transitions
from kaggrl.v3_2_export import export_v3_2_numpy
from kaggrl.v3_2_model import TemporalIntentPolicyV32
from kaggrl.v3_2_numpy_runtime import V32NumpyPolicy
from kaggrl.v3_2_schema import ARCHITECTURE_VERSION
from kaggrl.v3_strategy import build_strategy_manifest


EXPECTED_BASE_ARCH = "rl_v3_temporal_attention"
EXPECTED_NEW_KEYS = {
    "market_active_op_head.bias",
    "market_active_op_head.weight",
    "market_continue_head.bias",
    "market_continue_head.weight",
    "strategy_embedding.weight",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_action(action):
    def clean(command):
        return {
            key: value
            for key, value in command.items()
            if key not in {"raw", "_mask_fields"}
        }

    return {
        "farmer": clean(action["farmer"]),
        "hands": [clean(value) for value in action.get("hands") or []],
        "market": [clean(value) for value in action.get("market") or []],
    }


def _torch_action(row):
    return {
        "farmer": row.farmer.chosen_action,
        "hands": [decision.chosen_action for decision in row.hands],
        "market": [decision.chosen_action for decision in row.market],
    }


def _step0_rows(dataset_path: Path):
    table = pq.read_table(
        dataset_path,
        filters=[
            ("split", "=", "val"),
            ("role", "=", "active_best"),
            ("step", "=", 0),
        ],
    )
    rows = {}
    for row in table.to_pylist():
        rows.setdefault(int(row["team_id"]), row)
    return rows


def run(dataset_path: Path, source_checkpoint: Path, output_dir: Path, seed: int):
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    if source.get("architecture_version") != EXPECTED_BASE_ARCH:
        raise RuntimeError("pre-T4 migration must start from accepted base V3 checkpoint")

    train_teams = pq.read_table(
        dataset_path,
        filters=[("split", "=", "train"), ("role", "=", "active_best")],
        columns=["team_id"],
    ).column("team_id").to_pylist()
    manifest = build_strategy_manifest(int(value) for value in train_teams)
    validation_rows = _step0_rows(dataset_path)
    if set(validation_rows) != set(manifest.slot_to_team):
        raise RuntimeError("validation step-0 strategy coverage mismatch")

    torch.manual_seed(int(seed))
    model = TemporalIntentPolicyV32(strategy_count=manifest.size).eval()
    incompatible = model.load_state_dict(source["model_state"], strict=False)
    if set(incompatible.missing_keys) != EXPECTED_NEW_KEYS:
        raise RuntimeError(
            f"unexpected migration missing keys: {sorted(incompatible.missing_keys)}"
        )
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"unexpected migration keys: {sorted(incompatible.unexpected_keys)}"
        )

    init_path = output_dir / "v32_init_from_epoch1.pt"
    init_payload = {
        "format_version": 4,
        "architecture_version": ARCHITECTURE_VERSION,
        "model_state": model.state_dict(),
        "dataset_sha256": source.get("dataset_sha256"),
        "strategy_manifest": {
            "slot_to_team": list(manifest.slot_to_team),
            "sha256": manifest.sha256,
        },
        "strategy_manifest_sha256": manifest.sha256,
        "migration_new_parameter_keys": sorted(EXPECTED_NEW_KEYS),
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": _sha256(source_checkpoint),
        "migration_seed": int(seed),
    }
    torch.save(init_payload, init_path)

    slot_reports = []
    for slot, team_id in enumerate(manifest.slot_to_team):
        export_path = output_dir / f"slot_{slot:02d}_policy.npz"
        export_v3_2_numpy(
            model,
            export_path,
            default_strategy_slot=slot,
        )
        runtime = V32NumpyPolicy.load(export_path)
        batch = collate_transitions([validation_rows[int(team_id)]])
        strategy_slots = torch.tensor([slot], dtype=torch.long)

        with torch.no_grad():
            torch_trace = model.trace_sample_step(
                batch,
                None,
                np.random.default_rng(7000 + slot),
                deterministic=True,
                strategy_slots=strategy_slots,
            )
        numpy_trace = runtime.trace_step(
            batch.structured_states[0],
            {},
            batch.previous_actions[0],
            None,
            np.random.default_rng(7000 + slot),
            deterministic=True,
            strategy_slot=slot,
        )

        torch_action = _clean_action(_torch_action(torch_trace["output"].rows[0]))
        numpy_action = _clean_action(numpy_trace["canonical_action"])
        action_match = torch_action == numpy_action

        torch_decisions = torch_trace["decisions"]
        numpy_decisions = numpy_trace["decisions"]
        actor_match = (
            [row["actor"] for row in torch_decisions]
            == [row["actor"] for row in numpy_decisions]
        )
        op_vocab_match = all(
            left["ops"] == right["ops"]
            for left, right in zip(torch_decisions, numpy_decisions)
        ) and len(torch_decisions) == len(numpy_decisions)
        mask_match = all(
            left["legal_mask"] == right["legal_mask"]
            for left, right in zip(torch_decisions, numpy_decisions)
        ) and len(torch_decisions) == len(numpy_decisions)
        chosen_match = all(
            left["chosen_op"] == right["chosen_op"]
            for left, right in zip(torch_decisions, numpy_decisions)
        ) and len(torch_decisions) == len(numpy_decisions)
        max_logit_error = 0.0
        for left, right in zip(torch_decisions, numpy_decisions):
            lval = np.asarray(left["raw_logits"], dtype=np.float32)
            rval = np.asarray(right["raw_logits"], dtype=np.float32)
            if lval.shape != rval.shape:
                max_logit_error = float("inf")
                break
            if lval.size:
                max_logit_error = max(
                    max_logit_error,
                    float(np.max(np.abs(lval - rval))),
                )

        slot_reports.append({
            "slot": int(slot),
            "team_id": int(team_id),
            "export": str(export_path),
            "export_sha256": _sha256(export_path),
            "action_match": bool(action_match),
            "actor_match": bool(actor_match),
            "op_vocab_match": bool(op_vocab_match),
            "mask_match": bool(mask_match),
            "chosen_match": bool(chosen_match),
            "max_logit_error": float(max_logit_error),
            "passed": bool(
                action_match
                and actor_match
                and op_vocab_match
                and mask_match
                and chosen_match
                and max_logit_error <= 3e-5
            ),
        })

    source_files = [
        Path("src/kaggrl/v2_losses.py"),
        Path("src/kaggrl/v2_ledger.py"),
        Path("src/kaggrl/v2_model.py"),
        Path("src/kaggrl/v2_metrics.py"),
        Path("src/kaggrl/v3_2_schema.py"),
        Path("src/kaggrl/v3_2_model.py"),
        Path("src/kaggrl/v3_2_export.py"),
        Path("src/kaggrl/v3_2_numpy_runtime.py"),
        Path("training/train_v3_bc.py"),
        Path("evaluation/eval_v3_closed_loop.py"),
        Path("rollout/v3_agent_numpy.py"),
        Path("rollout/v3_2_agent_numpy.py"),
    ]
    report = {
        "kind": "rl_v3_2_local_pre_t4",
        "architecture_version": ARCHITECTURE_VERSION,
        "migration_seed": int(seed),
        "dataset": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": _sha256(source_checkpoint),
        "init_checkpoint": str(init_path),
        "init_checkpoint_sha256": _sha256(init_path),
        "model_parameter_sha256": model_parameter_sha256(model),
        "strategy_manifest": {
            "slot_to_team": list(manifest.slot_to_team),
            "sha256": manifest.sha256,
        },
        "migration_new_parameter_keys": sorted(EXPECTED_NEW_KEYS),
        "source_hashes": {
            str(path): _sha256(path) for path in source_files
        },
        "slot_reports": slot_reports,
        "all_slots_passed": all(row["passed"] for row in slot_reports),
    }
    report_path = output_dir / "pre_t4_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args()
    path = run(
        args.dataset,
        args.source_checkpoint,
        args.output_dir,
        args.seed,
    )
    print(path)


if __name__ == "__main__":
    main()
