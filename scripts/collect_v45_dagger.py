from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from kaggrl.v3_2_export import export_v3_2_numpy
from kaggrl.v3_2_model import TemporalIntentPolicyV32
from kaggrl.v3_2_schema import ARCHITECTURE_VERSION as V32_ARCH
from kaggrl.v3_3_export import export_v3_3_numpy
from kaggrl.v3_3_model import TemporalIntentPolicyV33
from kaggrl.v3_3_schema import ARCHITECTURE_VERSION as V33_ARCH
from training.build_v3_recovery_dataset import collect_v45_recovery


def _parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    return sorted(set(seeds))


def _export_checkpoint(checkpoint: Path, policy_path: Path, slot: int):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    architecture = str(payload.get("architecture_version", ""))
    manifest = payload.get("strategy_manifest") or {}
    slot_to_team = tuple(int(value) for value in manifest.get("slot_to_team") or ())
    if not slot_to_team:
        raise RuntimeError("checkpoint has no strategy manifest")
    if not 0 <= int(slot) < len(slot_to_team):
        raise ValueError(f"slot {slot} outside [0, {len(slot_to_team) - 1}]")

    if architecture == V32_ARCH:
        model = TemporalIntentPolicyV32(strategy_count=len(slot_to_team)).eval()
        model.load_state_dict(payload["model_state"], strict=True)
        export_v3_2_numpy(
            model, policy_path, default_strategy_slot=int(slot),
        )
    elif architecture == V33_ARCH:
        model = TemporalIntentPolicyV33(strategy_count=len(slot_to_team)).eval()
        model.load_state_dict(payload["model_state"], strict=True)
        export_v3_3_numpy(
            model, policy_path, default_strategy_slot=int(slot),
        )
    else:
        raise RuntimeError(
            f"DAgger collector requires V3.2/V3.3 checkpoint, got {architecture!r}"
        )
    return payload, slot_to_team


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect FarmOS-v45 DAgger states labeled by public v45."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--v45",
        default="/content/source_public/extracted_v45/main.py",
    )
    parser.add_argument(
        "--output",
        default="/content/farmos/data/recovery/v45_dagger.jsonl",
    )
    parser.add_argument("--slot", type=int, default=0)
    parser.add_argument(
        "--seeds",
        default="20260919,20260920,20260921",
    )
    parser.add_argument("--episode-steps", type=int, default=720)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    v45 = Path(args.v45).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    policy_path = output.with_suffix(".policy.npz")
    if policy_path.exists():
        policy_path.unlink()

    payload, slot_to_team = _export_checkpoint(
        checkpoint, policy_path, int(args.slot),
    )
    result = collect_v45_recovery(
        policy_path,
        v45,
        output,
        _parse_seeds(args.seeds),
        episode_steps=int(args.episode_steps),
        strategy_slot=int(args.slot),
    )
    print(
        "FARMOS_V45_DAGGER="
        + json.dumps({
            "checkpoint": str(checkpoint),
            "checkpoint_epoch": int(payload.get("epoch", -1)),
            "slot": int(args.slot),
            "team_id": int(slot_to_team[int(args.slot)]),
            "v45": str(v45),
            "recovery_dataset": str(result),
            "metadata": str(result) + ".meta.json",
            "policy": str(policy_path),
        }, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
