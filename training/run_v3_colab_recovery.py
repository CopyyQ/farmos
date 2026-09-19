from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import subprocess
import time

import torch

from kaggrl.v2_device import resolve_training_device
from kaggrl.v2_training_data import V2EpisodeDataset
from kaggrl.v3_model import TemporalIntentPolicy
from kaggrl.v3_2_model import TemporalIntentPolicyV32
from kaggrl.v3_2_schema import ARCHITECTURE_VERSION as V32_ARCHITECTURE_VERSION
from training.train_v3_bc import (
    BCV3Config, _episode_chunks, _episode_groups, _sha256, _teacher_chunk_cached, run_v3_bc,
)


@dataclass(frozen=True)
class T4MemoryPolicy:
    target_reserved_min_gib: float = 10.0
    target_reserved_max_gib: float = 13.0
    hard_reserved_max_gib: float = 14.0
    candidate_batches: tuple[int, ...] = (4, 8, 16, 32, 48, 64, 96, 128, 192, 256)


def choose_vram_candidate(samples, policy: T4MemoryPolicy) -> int:
    safe = [(int(batch), float(gib)) for batch, gib in samples if float(gib) <= policy.hard_reserved_max_gib]
    if not safe:
        raise RuntimeError("no VRAM probe stayed below the hard safety cap")
    target = [(batch, gib) for batch, gib in safe if policy.target_reserved_min_gib <= gib <= policy.target_reserved_max_gib]
    if target:
        return max(target, key=lambda item: (item[1], item[0]))[0]
    below = [(batch, gib) for batch, gib in safe if gib < policy.target_reserved_min_gib]
    if below:
        return max(below, key=lambda item: (item[1], item[0]))[0]
    return min(safe, key=lambda item: (item[1], item[0]))[0]


def assert_t4() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Colab recovery runner")
    device = resolve_training_device("cuda")
    if device.type != "cuda":
        raise RuntimeError("training device did not resolve to CUDA")
    name = torch.cuda.get_device_name(0)
    if "T4" not in name.upper():
        raise RuntimeError(f"expected Tesla T4, got: {name}")
    props = torch.cuda.get_device_properties(0)
    return {
        "device_name": name,
        "total_vram_gib": float(props.total_memory / (1024 ** 3)),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def _probe_model_from_payload(payload, device: torch.device):
    architecture = str(payload.get("architecture_version", ""))
    strategy = payload.get("strategy_manifest") or {}
    slot_to_team = list(strategy.get("slot_to_team") or [])
    if architecture == V32_ARCHITECTURE_VERSION:
        if not slot_to_team:
            raise RuntimeError("V3.2 probe checkpoint is missing strategy manifest")
        model = TemporalIntentPolicyV32(strategy_count=len(slot_to_team))
    elif architecture == "rl_v3_1_strategy_temporal_attention":
        if not slot_to_team:
            raise RuntimeError("V3.1 probe checkpoint is missing strategy manifest")
        model = TemporalIntentPolicy(strategy_count=len(slot_to_team))
    elif architecture == "rl_v3_temporal_attention":
        model = TemporalIntentPolicy()
    else:
        raise RuntimeError(
            f"unsupported probe checkpoint architecture: {architecture}"
        )
    model.load_state_dict(payload["model_state"], strict=True)
    return model.to(device).train()


def _probe_memory(model, dataset, batch_sequences: int, sequence_len: int,
                  seed: int, device: torch.device) -> dict:
    torch.manual_seed(seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model.zero_grad(set_to_none=True)
    group = next(_episode_groups(dataset, batch_sequences, seed, 0))
    chunks_by_slot = [_episode_chunks(episode, sequence_len) for episode in group]
    active = [(slot, chunks[0]) for slot, chunks in enumerate(chunks_by_slot) if chunks]
    states = [None] * len(group)
    started = time.perf_counter()
    losses, _ = _teacher_chunk_cached(model, active, states, device, recurrent_stats=None)
    forward_seconds = time.perf_counter() - started
    losses["total"].backward()
    torch.cuda.synchronize(device)
    total_seconds = time.perf_counter() - started
    result = {
        "batch_sequences": int(batch_sequences),
        "actual_sequences": len(group),
        "max_allocated_gib": float(torch.cuda.max_memory_allocated(device) / (1024 ** 3)),
        "max_reserved_gib": float(torch.cuda.max_memory_reserved(device) / (1024 ** 3)),
        "forward_seconds": float(forward_seconds),
        "total_seconds": float(total_seconds),
        "rows": int(sum(len(chunk.rows) for _, chunk in active)),
    }
    model.zero_grad(set_to_none=True)
    del losses
    torch.cuda.empty_cache()
    return result


def autotune_batch_sequences(init_checkpoint: Path, dataset_path: Path,
                             sequence_len: int, seed: int,
                             policy: T4MemoryPolicy) -> tuple[int, list[dict]]:
    device = torch.device("cuda")
    payload = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
    model = _probe_model_from_payload(payload, device)
    dataset = V2EpisodeDataset(dataset_path, "train", {"active_best"})
    probes = []
    successful = []
    for candidate in policy.candidate_batches:
        try:
            row = _probe_memory(model, dataset, candidate, sequence_len, seed, device)
            probes.append(row)
            if row["actual_sequences"] != candidate:
                break
            reserved = float(row["max_reserved_gib"])
            if reserved > policy.hard_reserved_max_gib:
                break
            successful.append((candidate, reserved))
            print(json.dumps({"vram_probe": row}, sort_keys=True), flush=True)
            if policy.target_reserved_min_gib <= reserved <= policy.target_reserved_max_gib:
                break
        except torch.OutOfMemoryError:
            probes.append({"batch_sequences": candidate, "oom": True})
            torch.cuda.empty_cache()
            break
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            probes.append({"batch_sequences": candidate, "oom": True})
            torch.cuda.empty_cache()
            break
    del model
    torch.cuda.empty_cache()
    return choose_vram_candidate(successful, policy), probes


def _nvidia_smi_snapshot() -> dict:
    command = [
        "nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        line = subprocess.check_output(command, text=True, timeout=5).strip().splitlines()[0]
        util, used, total, temp = [part.strip() for part in line.split(",")]
        return {
            "utilization_percent": float(util), "memory_used_mib": float(used),
            "memory_total_mib": float(total), "temperature_c": float(temp),
        }
    except Exception as exc:
        return {"error": str(exc)}


def _write_remote_manifest(output_dir: Path) -> Path:
    files = sorted(path for path in output_dir.rglob("*") if path.is_file() and path.name != "remote_manifest.sha256")
    lines = [f"{_sha256(path)}  {path.relative_to(output_dir).as_posix()}" for path in files]
    manifest = output_dir / "remote_manifest.sha256"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def run_colab_recovery(args) -> Path:
    gpu = assert_t4()
    policy = T4MemoryPolicy(
        target_reserved_min_gib=args.target_vram_min_gib,
        target_reserved_max_gib=args.target_vram_max_gib,
        hard_reserved_max_gib=args.hard_vram_max_gib,
    )
    probes = []
    batch_sequences = int(args.batch_sequences)
    if args.autotune_vram:
        batch_sequences, probes = autotune_batch_sequences(
            Path(args.init_checkpoint), Path(args.dataset),
            int(args.sequence_len), int(args.seed), policy,
        )
    probe_dir = Path(args.output_dir)
    probe_dir.mkdir(parents=True, exist_ok=True)
    (probe_dir / "vram_autotune.json").write_text(
        json.dumps({"chosen_batch_sequences": batch_sequences, "probes": probes}, indent=2, sort_keys=True) + "\n"
    )
    torch.manual_seed(int(args.seed))
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    config = BCV3Config(
        dataset_path=Path(args.dataset), stage0_marker=Path(args.stage0_marker),
        stage1_marker=Path(args.stage1_marker), output_dir=Path(args.output_dir),
        seed=int(args.seed), sequence_len=int(args.sequence_len),
        batch_sequences=batch_sequences, learning_rate=float(args.learning_rate),
        epochs=int(args.epochs), max_train_steps=int(args.max_train_steps),
        max_val_chunks=int(args.max_val_chunks), device="cuda",
        recovery_dataset_path=(
            None if args.recovery_dataset is None else Path(args.recovery_dataset)
        ),
        recovery_every=int(args.recovery_every),
        family_weight_cap=args.family_weight_cap,
        strategy_conditioning=bool(args.strategy_conditioning),
        model_architecture=str(args.model_architecture),
    )
    best = run_v3_bc(config, Path(args.init_checkpoint))
    torch.cuda.synchronize()
    peak_alloc = float(torch.cuda.max_memory_allocated() / (1024 ** 3))
    peak_reserved = float(torch.cuda.max_memory_reserved() / (1024 ** 3))
    gpu.update({
        "memory_policy": asdict(policy), "vram_probes": probes,
        "chosen_batch_sequences": batch_sequences,
        "peak_allocated_gib": peak_alloc, "peak_reserved_gib": peak_reserved,
        "peak_free_gib": max(0.0, gpu["total_vram_gib"] - peak_reserved),
        "vram_target_met": bool(peak_reserved >= policy.target_reserved_min_gib),
        "nvidia_smi_after": _nvidia_smi_snapshot(),
    })
    output_dir = Path(args.output_dir)
    (output_dir / "gpu_run_meta.json").write_text(json.dumps(gpu, indent=2, sort_keys=True) + "\n")
    _write_remote_manifest(output_dir)
    return best


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--stage0-marker", required=True)
    parser.add_argument("--stage1-marker", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--recovery-dataset")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--sequence-len", type=int, default=32)
    parser.add_argument("--batch-sequences", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-train-steps", type=int, required=True)
    parser.add_argument("--max-val-chunks", type=int, default=8)
    parser.add_argument("--recovery-every", type=int, default=4)
    parser.add_argument("--family-weight-cap", type=float)
    parser.add_argument("--strategy-conditioning", action="store_true")
    parser.add_argument(
        "--model-architecture",
        choices=("rl_v3_temporal_attention", V32_ARCHITECTURE_VERSION),
        default="rl_v3_temporal_attention",
    )
    parser.add_argument("--autotune-vram", action="store_true")
    parser.add_argument("--target-vram-min-gib", type=float, default=10.0)
    parser.add_argument("--target-vram-max-gib", type=float, default=13.0)
    parser.add_argument("--hard-vram-max-gib", type=float, default=14.0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    best = run_colab_recovery(args)
    print(json.dumps({"best_checkpoint": str(best)}, sort_keys=True))


if __name__ == "__main__":
    main()
