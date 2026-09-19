from __future__ import annotations

import hashlib
import json
import math
import random
import time
from itertools import islice
from collections import Counter
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kaggrl.constants import UNIT_OPS
from kaggrl.v2_device import move_step_batch, resolve_training_device
from kaggrl.v2_ledger import MARKET_OPS
from kaggrl.v2_losses import action_loss, total_pretrain_loss
from kaggrl.v2_metrics import semantic_domain_metrics
from kaggrl.v3_behavior import behavior_family, compute_domain_family_weights
from kaggrl.v2_training_data import (
    SequenceChunk,
    V2EpisodeDataset,
    _attach_auxiliary_targets,
    collate_v2_sequences,
    verify_effective_action_sidecar,
    verify_training_acceptance,
)
from kaggrl.v2_tensorize import V2Batch as StepBatch, collate_transitions
from kaggrl.v3_model import TemporalIntentPolicy
from kaggrl.v3_2_model import TemporalIntentPolicyV32
from kaggrl.v3_2_schema import (
    ACTIVE_MARKET_OPS,
    ARCHITECTURE_VERSION as V32_ARCHITECTURE_VERSION,
    CONTINUE_ID,
    STOP_ID,
    STRATEGY_CORE_SCALE,
)
from kaggrl.v3_3_export import export_v3_3_numpy
from kaggrl.v3_3_model import TemporalIntentPolicyV33
from kaggrl.v3_3_schema import (
    ARCHITECTURE_VERSION as V33_ARCHITECTURE_VERSION,
)
from kaggrl.v3_strategy import (
    StrategyManifest, build_strategy_manifest, strategy_slot_for_team,
)
from kaggrl.v3_tensor_decoder import teacher_step_tensor_mixed
from kaggrl.v3_tensor_ledger import TensorLedger
from kaggrl.v3_tensor_losses import (
    tensor_total_pretrain_loss,
    tensor_total_pretrain_loss_sequence,
)
from kaggrl.v3_tensor_targets import TensorActionTargets
from training.build_v3_recovery_dataset import (
    RecoveryCollectionEmptyError,
    collect_v45_recovery,
    read_recovery_rows,
    write_recovery_rows,
)
from training.train_v3_pretrain import (
    _detach_states,
    _episode_chunks,
    _episode_groups,
    _float_dict,
    _mean_float_dict,
    _mean_tensor_dict,
    _pack_states,
    _sha256,
    _store_state,
)

ARCHITECTURE_VERSION = "rl_v3_temporal_attention"
STRATEGY_ARCHITECTURE_VERSION = "rl_v3_1_strategy_temporal_attention"
FORMAT_VERSION = 3
SELECTION_WEIGHTS = {
    "farmer_semantic_exact": 0.30,
    "mean_hand_semantic_exact": 0.30,
    "market_sequence_exact": 0.30,
    "full_joint_step_exact": 0.10,
}
GPU_BATCH_CACHE_FORMAT_VERSION = 2

V32_SELECTION_WEIGHTS = {
    "farmer_semantic_exact": 0.25,
    "mean_hand_semantic_exact": 0.25,
    "market_active_semantic_exact": 0.25,
    "market_continue_accuracy": 0.15,
    "full_joint_step_exact": 0.10,
}


@dataclass(frozen=True)
class BCV3Config:
    dataset_path: Path
    stage0_marker: Path
    stage1_marker: Path
    output_dir: Path
    seed: int = 20260917
    sequence_len: int = 32
    batch_sequences: int = 4
    learning_rate: float = 2e-4
    init_mode: str = "checkpoint"
    lr_warmup_steps: int = 0
    lr_min_ratio: float = 1.0
    fused_optimizer: bool = True
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    epochs: int = 2
    torch_num_threads: int = 2
    device: str = "auto"
    max_train_steps: int | None = None
    max_train_steps_per_epoch: int | None = None
    max_val_chunks: int | None = None
    max_farmer_op_gap: float = 0.25
    max_hand_op_gap: float = 0.25
    max_market_op_gap: float = 0.25
    max_farmer_pass_fraction: float = 0.80
    max_hand_pass_fraction: float = 0.80
    max_stop_queue_fraction: float = 0.95
    stop_queue_expert_gap: float = 0.20
    min_initial_market_continue_accuracy: float = 0.95
    min_initial_market_active_accuracy: float = 0.80
    early_stop_patience: int = 3
    recovery_dataset_path: Path | None = None
    recovery_every: int = 4
    online_dagger: bool = False
    dagger_teacher_path: Path | None = None
    dagger_start_epoch: int = 1
    dagger_seeds_per_round: int = 1
    dagger_seed_base: int = 20270000
    dagger_episode_steps: int = 720
    dagger_strategy_slot: int = 0
    family_weight_cap: float | None = None
    market_active_op_weight_cap: float = 8.0
    market_active_op_weight_power: float = 0.75
    teacher_mix_schedule: tuple[float, ...] = (1.0,)
    opening_replay_steps: int = 0
    opening_replay_learning_rate: float = 1e-2
    require_collapse_for_best: bool = False
    strategy_conditioning: bool = False
    model_architecture: str = ARCHITECTURE_VERSION
    progress_every: int = 0
    use_amp: bool = False
    amp_init_scale: float = 1024.0
    amp_growth_interval: int = 2000
    validation_profile: str = "full"
    selection_mode: str = "offline_legacy"
    clear_cuda_cache: bool = True
    gpu_tensor_training: bool = False
    gpu_batch_cache: bool = False
    gpu_batch_cache_disk: bool = False
    gpu_batch_cache_reserve_gb: float = 3.0
    gpu_batch_cache_pin_cpu: bool = True

    def validate(self) -> None:
        if self.sequence_len <= 0 or self.batch_sequences <= 0:
            raise ValueError("sequence_len and batch_sequences must be positive")
        if self.epochs <= 0 or self.learning_rate <= 0:
            raise ValueError("invalid BC optimization configuration")
        if self.init_mode not in {"checkpoint", "scratch"}:
            raise ValueError("init_mode must be 'checkpoint' or 'scratch'")
        if int(self.lr_warmup_steps) < 0:
            raise ValueError("lr_warmup_steps must be non-negative")
        if not 0.0 < float(self.lr_min_ratio) <= 1.0:
            raise ValueError("lr_min_ratio must be in (0, 1]")
        if self.gradient_clip <= 0 or self.torch_num_threads < 1:
            raise ValueError("invalid BC runtime configuration")
        if (
            self.max_train_steps_per_epoch is not None
            and int(self.max_train_steps_per_epoch) <= 0
        ):
            raise ValueError("max_train_steps_per_epoch must be positive when set")
        if int(self.progress_every) < 0:
            raise ValueError("progress_every must be non-negative")
        if float(self.amp_init_scale) <= 0.0:
            raise ValueError("amp_init_scale must be positive")
        if int(self.amp_growth_interval) <= 0:
            raise ValueError("amp_growth_interval must be positive")
        if self.validation_profile not in {"full", "fast"}:
            raise ValueError("validation_profile must be 'full' or 'fast'")
        if self.selection_mode not in {"offline_legacy", "last_epoch"}:
            raise ValueError("selection_mode must be 'offline_legacy' or 'last_epoch'")
        if self.validation_profile == "fast" and self.selection_mode != "last_epoch":
            raise ValueError("fast validation requires selection_mode='last_epoch'")
        if self.model_architecture not in {
            ARCHITECTURE_VERSION,
            V32_ARCHITECTURE_VERSION,
            V33_ARCHITECTURE_VERSION,
        }:
            raise ValueError("unsupported V3 model_architecture")
        if self.gpu_tensor_training and self.model_architecture not in {
            V32_ARCHITECTURE_VERSION,
            V33_ARCHITECTURE_VERSION,
        }:
            raise ValueError("gpu_tensor_training currently requires V3.2/V3.3")
        if self.gpu_batch_cache and not self.gpu_tensor_training:
            raise ValueError("gpu_batch_cache requires gpu_tensor_training")
        if float(self.gpu_batch_cache_reserve_gb) < 0.5:
            raise ValueError("gpu_batch_cache_reserve_gb must be >= 0.5")
        if self.model_architecture in {
            V32_ARCHITECTURE_VERSION,
            V33_ARCHITECTURE_VERSION,
        }:
            if not self.strategy_conditioning:
                raise ValueError("V3.2/V3.3 requires strategy_conditioning")
        if self.init_mode == "scratch" and self.model_architecture != V33_ARCHITECTURE_VERSION:
            raise ValueError("scratch init is currently supported only for V3.3")
        if (
            self.model_architecture == V32_ARCHITECTURE_VERSION
            and self.recovery_dataset_path is not None
        ):
            raise ValueError("V3.2 R0 does not allow recovery supervision")
        if self.online_dagger:
            if self.model_architecture != V33_ARCHITECTURE_VERSION:
                raise ValueError("online_dagger requires V3.3")
            if self.dagger_teacher_path is None:
                raise ValueError("online_dagger requires dagger_teacher_path")
            if int(self.dagger_start_epoch) < 1:
                raise ValueError("dagger_start_epoch must be >= 1")
            if int(self.dagger_seeds_per_round) < 1:
                raise ValueError("dagger_seeds_per_round must be >= 1")
            if int(self.dagger_episode_steps) < 2:
                raise ValueError("dagger_episode_steps must be >= 2")
            if int(self.dagger_strategy_slot) < 0:
                raise ValueError("dagger_strategy_slot must be non-negative")
        if (self.recovery_dataset_path is not None or self.online_dagger) and self.recovery_every < 3:
            raise ValueError("recovery_every must preserve an expert-majority schedule")
        if self.family_weight_cap is not None and float(self.family_weight_cap) < 1.0:
            raise ValueError("family_weight_cap must be >= 1 when enabled")
        if float(self.market_active_op_weight_cap) < 1.0:
            raise ValueError("market_active_op_weight_cap must be >= 1")
        if not 0.0 <= float(self.market_active_op_weight_power) <= 1.0:
            raise ValueError("market_active_op_weight_power must be in [0, 1]")
        if not self.teacher_mix_schedule:
            raise ValueError("teacher_mix_schedule must not be empty")
        if any(not 0.0 <= float(value) <= 1.0 for value in self.teacher_mix_schedule):
            raise ValueError("teacher_mix_schedule values must be in [0, 1]")
        if int(self.opening_replay_steps) < 0:
            raise ValueError("opening_replay_steps must be non-negative")
        if float(self.opening_replay_learning_rate) <= 0.0:
            raise ValueError("opening_replay_learning_rate must be positive")
        thresholds = (
            self.max_farmer_op_gap, self.max_hand_op_gap,
            self.max_market_op_gap, self.max_farmer_pass_fraction,
            self.max_hand_pass_fraction, self.max_stop_queue_fraction,
            self.stop_queue_expert_gap,
        )
        if not 0.0 <= float(self.min_initial_market_continue_accuracy) <= 1.0:
            raise ValueError("min_initial_market_continue_accuracy must be in [0, 1]")
        if not 0.0 <= float(self.min_initial_market_active_accuracy) <= 1.0:
            raise ValueError("min_initial_market_active_accuracy must be in [0, 1]")
        if any(value < 0.0 for value in thresholds):
            raise ValueError("collapse thresholds must be non-negative")


def _teacher_mix_for_epoch(schedule, epoch: int) -> float:
    values = tuple(float(value) for value in schedule)
    if not values:
        raise ValueError("teacher_mix_schedule must not be empty")
    index = min(max(0, int(epoch) - 1), len(values) - 1)
    return float(values[index])


def _initialize_v32_migration(
    model: TemporalIntentPolicyV32,
    init_architecture: str,
    market_continue_counts: dict[str, int] | None = None,
) -> None:
    if init_architecture == V32_ARCHITECTURE_VERSION:
        return
    del market_continue_counts
    legacy_weight = model.market_op_head.weight
    legacy_bias = model.market_op_head.bias
    active_indices = [MARKET_OPS.index(op) for op in ACTIVE_MARKET_OPS]
    with torch.no_grad():
        model.market_active_op_head.weight.copy_(
            legacy_weight[active_indices]
        )
        model.market_active_op_head.bias.copy_(
            legacy_bias[active_indices]
        )
        # Keep the seeded constructor initialization of market_continue_head.
        # A constant prior made every state CONTINUE; full STOP-vs-rest legacy
        # transfer made every state STOP. The reproducible random projection
        # provides state variation from step zero and is then trained directly
        # by the balanced decision-level STOP/CONTINUE objective.
        if (
            init_architecture == ARCHITECTURE_VERSION
            and model.strategy_embedding is not None
        ):
            model.strategy_embedding.weight.zero_()


def _scheduled_learning_rate(
    config: BCV3Config,
    *,
    epoch: int,
    train_steps: int,
) -> float:
    base = float(config.learning_rate)
    warmup = int(config.lr_warmup_steps)
    warmup_scale = (
        1.0
        if warmup <= 0
        else min(1.0, float(train_steps + 1) / float(warmup))
    )
    if int(config.epochs) <= 1:
        decay_scale = 1.0
    else:
        progress = float(max(0, int(epoch) - 1)) / float(
            max(1, int(config.epochs) - 1)
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        minimum = float(config.lr_min_ratio)
        decay_scale = minimum + (1.0 - minimum) * cosine
    return base * warmup_scale * decay_scale


def _set_optimizer_lr(optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(learning_rate)


def _build_adamw(model, config: BCV3Config, device):
    kwargs = {
        "lr": float(config.learning_rate),
        "weight_decay": float(config.weight_decay),
    }
    if (
        bool(config.fused_optimizer)
        and getattr(device, "type", str(device).split(":")[0]) == "cuda"
    ):
        try:
            return torch.optim.AdamW(model.parameters(), fused=True, **kwargs), True
        except (TypeError, RuntimeError):
            pass
    return torch.optim.AdamW(model.parameters(), **kwargs), False


def _config_dict(config: BCV3Config) -> dict[str, Any]:
    value = asdict(config)
    for key in ("dataset_path", "stage0_marker", "stage1_marker", "output_dir"):
        value[key] = str(value[key])
    if value.get("recovery_dataset_path") is not None:
        value["recovery_dataset_path"] = str(value["recovery_dataset_path"])
    if value.get("dagger_teacher_path") is not None:
        value["dagger_teacher_path"] = str(value["dagger_teacher_path"])
    value["selection_weights"] = dict(
        V32_SELECTION_WEIGHTS
        if config.model_architecture in {
            V32_ARCHITECTURE_VERSION,
            V33_ARCHITECTURE_VERSION,
        }
        else SELECTION_WEIGHTS
    )
    return value


def _build_training_strategy_manifest(dataset: V2EpisodeDataset) -> StrategyManifest:
    return build_strategy_manifest(episode.team_id for episode in dataset)


def _validate_strategy_coverage(
    manifest: StrategyManifest, dataset: V2EpisodeDataset,
) -> None:
    unseen = sorted({
        int(episode.team_id) for episode in dataset
        if int(episode.team_id) not in manifest.team_to_slot
    })
    if unseen:
        raise RuntimeError(f"unseen strategy team ids outside training split: {unseen}")


def _strategy_slots_for_rows(rows, manifest: StrategyManifest, device) -> torch.Tensor:
    slots = []
    for row in rows:
        if row.get("strategy_slot") is not None:
            slot = int(row["strategy_slot"])
            if not 0 <= slot < manifest.size:
                raise ValueError(
                    f"strategy_slot {slot} outside strategy manifest"
                )
            slots.append(slot)
        else:
            slots.append(
                strategy_slot_for_team(int(row["team_id"]), manifest)
            )
    return torch.tensor(slots, dtype=torch.long, device=device)


def _opening_training_rows(dataset: V2EpisodeDataset) -> list[dict[str, Any]]:
    rows = []
    for episode in dataset:
        if not episode.rows:
            continue
        row = episode.rows[0]
        if int(row.get("step", 0) or 0) != 0:
            raise RuntimeError("training episode does not start at step 0")
        rows.append(row)
    return rows


def _opening_strategy_replay(
    model: TemporalIntentPolicyV32,
    rows: list[dict[str, Any]],
    manifest: StrategyManifest,
    device,
    *,
    steps: int,
    learning_rate: float,
) -> dict[str, float]:
    if steps <= 0:
        return {"opening_replay_steps": 0.0}
    if not rows:
        raise RuntimeError("opening replay requires step-0 rows")
    batch = collate_transitions(rows)
    move_step_batch(batch, device)
    strategy_slots = _strategy_slots_for_rows(rows, manifest, device)
    target_ids = []
    for action in batch.canonical_actions:
        market = list(action.get("market") or [])
        op = _market_op(market[0]) if market else "STOP_QUEUE"
        if op not in model.market_active_op_to_id:
            raise RuntimeError(f"opening target is not an active market op: {op}")
        target_ids.append(model.market_active_op_to_id[op])
    targets = torch.tensor(target_ids, dtype=torch.long, device=device)
    parameters = tuple(model.opening_strategy_head.parameters())
    optimizer = torch.optim.Adam(parameters, lr=float(learning_rate))
    first_loss = None
    first_acc = None
    last_loss = None
    last_acc = None
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        output = model.teacher_step(
            batch, batch.canonical_actions, None,
            strategy_slots=strategy_slots,
            teacher_mix_probability=1.0,
        )
        logits = torch.stack([row.market[0].op_logits for row in output.rows])
        loss = torch.nn.functional.cross_entropy(logits, targets)
        accuracy = (logits.argmax(dim=-1) == targets).float().mean()
        if first_loss is None:
            first_loss = float(loss.detach().cpu())
            first_acc = float(accuracy.detach().cpu())
        gradients = torch.autograd.grad(loss, parameters)
        for parameter, gradient in zip(parameters, gradients):
            parameter.grad = gradient
        optimizer.step()
        last_loss = float(loss.detach().cpu())
        last_acc = float(accuracy.detach().cpu())
    with torch.no_grad():
        output = model.teacher_step(
            batch, batch.canonical_actions, None,
            strategy_slots=strategy_slots,
            teacher_mix_probability=1.0,
        )
        logits = torch.stack([row.market[0].op_logits for row in output.rows])
        final_loss = torch.nn.functional.cross_entropy(logits, targets)
        final_acc = (logits.argmax(dim=-1) == targets).float().mean()
    return {
        "opening_replay_steps": float(steps),
        "opening_loss_before": float(first_loss),
        "opening_accuracy_before": float(first_acc),
        "opening_loss_last_update": float(last_loss),
        "opening_accuracy_last_update": float(last_acc),
        "opening_loss_after": float(final_loss.detach().cpu()),
        "opening_accuracy_after": float(final_acc.detach().cpu()),
    }


def _canonical(row_output):
    return {
        "farmer": row_output.farmer.chosen_action,
        "hands": [decision.chosen_action for decision in row_output.hands],
        "market": [decision.chosen_action for decision in row_output.market],
    }

def _market_op(slot):
    kind = str(slot.get("kind", "ORDER"))
    return kind if kind in {"STOP_QUEUE", "NOP_SLOT"} else str(slot.get("op", "NOP_SLOT"))


def _update_histograms(histograms, action):
    farmer = str((action.get("farmer") or {}).get("op", "PASS"))
    histograms["farmer"][farmer] += 1
    for command in action.get("hands") or []:
        histograms["hands"][str(command.get("op", "PASS"))] += 1
    for slot in action.get("market") or []:
        histograms["market"][_market_op(slot)] += 1


def _plain_histograms(histograms):
    vocab = {
        "farmer": UNIT_OPS,
        "hands": UNIT_OPS,
        "market": MARKET_OPS,
    }
    return {
        domain: {op: int(counter.get(op, 0)) for op in vocab[domain]}
        for domain, counter in histograms.items()
    }


def _training_family_counts(dataset: V2EpisodeDataset) -> dict[str, dict[str, int]]:
    counts = {"unit": Counter(), "market": Counter()}
    for episode in dataset:
        for row in episode.rows:
            action = row["canonical_action"]
            counts["unit"][behavior_family(action["farmer"], "unit")] += 1
            for command in action.get("hands") or []:
                counts["unit"][behavior_family(command, "unit")] += 1
            for slot in action.get("market") or []:
                counts["market"][behavior_family(slot, "market")] += 1
    return {
        domain: {name: int(value) for name, value in sorted(counter.items())}
        for domain, counter in counts.items()
    }


def _training_market_continue_counts(
    dataset: V2EpisodeDataset,
) -> dict[str, int]:
    counts = {"STOP": 0, "CONTINUE": 0}
    for episode in dataset:
        for row in episode.rows:
            action = row["canonical_action"]
            for slot in action.get("market") or []:
                if _market_op(slot) == "STOP_QUEUE":
                    counts["STOP"] += 1
                else:
                    counts["CONTINUE"] += 1
    return counts


def _training_market_active_counts(
    dataset: V2EpisodeDataset,
) -> dict[str, int]:
    counts = Counter()
    for episode in dataset:
        for row in episode.rows:
            for slot in row["canonical_action"].get("market") or []:
                op = _market_op(slot)
                if op != "STOP_QUEUE":
                    counts[op] += 1
    return {
        op: int(counts.get(op, 0))
        for op in ACTIVE_MARKET_OPS
    }


def _balanced_market_active_op_weights(
    counts: dict[str, int], *, cap: float, power: float,
) -> dict[str, float]:
    positive = [int(value) for value in counts.values() if int(value) > 0]
    if not positive:
        return {op: 1.0 for op in ACTIVE_MARKET_OPS}
    maximum = max(positive)
    result = {}
    for op in ACTIVE_MARKET_OPS:
        count = int(counts.get(op, 0))
        if count <= 0:
            result[op] = float(cap)
        else:
            result[op] = min(
                float(cap),
                (float(maximum) / float(count)) ** float(power),
            )
    return result


def _teacher_chunk(
    model, active_chunks, states, device, recurrent_stats=None,
    strategy_manifest: StrategyManifest | None = None,
    family_weights: dict[str, float] | None = None,
    market_active_op_weights: dict[str, float] | None = None,
):
    step_losses = []
    max_len = max(len(chunk.rows) for _, chunk in active_chunks)
    for time_index in range(max_len):
        active = [
            (slot, chunk) for slot, chunk in active_chunks
            if time_index < len(chunk.rows)
        ]
        slots = [slot for slot, _ in active]
        one_steps = [
            SequenceChunk(
                chunk.episode_id, chunk.seat, (chunk.rows[time_index],),
                chunk.episode_start and time_index == 0,
                chunk.episode_end and time_index == len(chunk.rows) - 1,
            )
            for _, chunk in active
        ]
        batch = collate_v2_sequences(one_steps, 1)
        move_step_batch(batch.flat, device)
        packed = _pack_states(states, slots)
        strategy_slots = (
            _strategy_slots_for_rows(
                [chunk.rows[time_index] for _, chunk in active],
                strategy_manifest, device,
            ) if strategy_manifest is not None else None
        )
        output = model.teacher_step(
            batch.flat, batch.flat.canonical_actions, packed,
            strategy_slots=strategy_slots,
        )
        step_losses.append(total_pretrain_loss(
            output, batch.flat, family_weights=family_weights,
            market_active_op_weights=market_active_op_weights,
        ))
        _store_state(states, slots, output.temporal_state)
        if recurrent_stats is not None:
            recurrent_stats["temporal_steps"] += len(slots)
    return _mean_tensor_dict(step_losses), states


def _slice_step_batch(flat, row_indices):
    host = [int(value) for value in row_indices]
    index = torch.tensor(host, dtype=torch.long, device=flat.own_grid.device)
    kwargs = {}
    for field in fields(StepBatch):
        value = getattr(flat, field.name)
        if torch.is_tensor(value):
            kwargs[field.name] = value.index_select(0, index)
        elif isinstance(value, tuple):
            kwargs[field.name] = tuple(value[i] for i in host)
        else:
            raise TypeError(f"unsupported StepBatch field {field.name}: {type(value)!r}")
    result = StepBatch(**kwargs)
    auxiliary = getattr(flat, "auxiliary_targets", None)
    if isinstance(auxiliary, dict):
        result.auxiliary_targets = {
            key: value.index_select(0, index) if torch.is_tensor(value) else value
            for key, value in auxiliary.items()
        }
    sample_weight = getattr(flat, "sample_weight", None)
    if torch.is_tensor(sample_weight):
        result.sample_weight = sample_weight.index_select(0, index)
    return result


def _teacher_chunk_tensor_sequence(
    model,
    active_chunks,
    states,
    device,
    sequence,
    tensor_targets,
    tensor_ledger,
    tensor_generator,
    *,
    strategy_manifest=None,
    family_weights=None,
    market_active_op_weights=None,
    teacher_mix_probability=1.0,
    recurrent_stats=None,
):
    chunks = [chunk for _, chunk in active_chunks]
    lengths = [len(chunk.rows) for chunk in chunks]
    if not lengths or len(set(lengths)) != 1:
        raise ValueError("tensor sequence path requires equal chunk lengths")
    batch_size = len(chunks)
    steps = lengths[0]
    if steps <= 0 or steps > int(model.core.window):
        raise ValueError("tensor sequence path requires 1 <= T <= attention window")
    if int(sequence.flat.own_grid.shape[0]) != batch_size * steps:
        raise RuntimeError("flat sequence row order/size mismatch")

    slots = [slot for slot, _ in active_chunks]
    packed = _pack_states(states, slots)
    flat_rows = [
        row
        for chunk in chunks
        for row in chunk.rows
    ]
    flat_strategy_slots = (
        _strategy_slots_for_rows(
            flat_rows,
            strategy_manifest,
            device,
        )
        if strategy_manifest is not None
        else None
    )

    encoded = model.encoder(sequence.flat)
    strategy_context = None
    core_input = encoded.fused
    if flat_strategy_slots is not None:
        if getattr(model, "strategy_embedding", None) is None:
            raise RuntimeError("strategy-conditioned fast path needs an embedding")
        strategy_context = model.strategy_embedding(
            flat_strategy_slots
        ).to(encoded.fused)
        core_context = torch.cat(
            [strategy_context, strategy_context],
            dim=-1,
        )
        if core_context.shape != encoded.fused.shape:
            raise RuntimeError("strategy core context shape mismatch")
        core_input = (
            encoded.fused
            + float(STRATEGY_CORE_SCALE) * core_context
        )

    def bt(value):
        return value.reshape(
            batch_size,
            steps,
            *value.shape[1:],
        )

    fused_seq, base_intent_seq, next_state, diagnostics = model.core.sequence(
        bt(core_input),
        bt(sequence.flat.previous_action_global),
        bt(sequence.flat.previous_effect),
        bt(sequence.flat.economy),
        packed,
    )
    fused_flat = fused_seq.reshape(batch_size * steps, -1)
    base_intent_flat = base_intent_seq.reshape(batch_size * steps, -1)
    intent_flat = (
        base_intent_flat
        if strategy_context is None
        else base_intent_flat + strategy_context
    )
    output = teacher_step_tensor_mixed(
        model,
        sequence.flat,
        tensor_targets,
        tensor_ledger,
        packed,
        strategy_slots=flat_strategy_slots,
        teacher_mix_probability=teacher_mix_probability,
        generator=tensor_generator,
        precomputed={
            "encoded": encoded,
            "fused_temporal": fused_flat,
            "intent": intent_flat,
            "temporal_state": next_state,
            "temporal_diagnostics": diagnostics,
            "strategy_context": strategy_context,
        },
    )
    losses = tensor_total_pretrain_loss_sequence(
        output,
        sequence.flat,
        tensor_targets,
        step=tensor_ledger.step,
        batch_size=batch_size,
        steps=steps,
        family_weights=family_weights,
        market_active_op_weights=market_active_op_weights,
    )
    _store_state(states, slots, next_state)
    if recurrent_stats is not None:
        recurrent_stats["temporal_steps"] += batch_size * steps
        recurrent_stats["tensor_sequence_chunks"] = (
            int(recurrent_stats.get("tensor_sequence_chunks", 0)) + 1
        )
    return losses, states


@dataclass
class _CachedTensorSequence:
    slots: tuple[int, ...]
    steps: int
    flat: StepBatch
    targets: TensorActionTargets
    ledger: TensorLedger
    strategy_slots: torch.Tensor | None
    resident_on_device: bool
    tensor_bytes: int


@dataclass
class _PreparedTrainingUpdate:
    active: tuple[tuple[int, SequenceChunk], ...]
    cached: _CachedTensorSequence | None


@dataclass
class _PreparedTrainingGroup:
    slot_count: int
    updates: tuple[_PreparedTrainingUpdate, ...]


def _tensor_bytes(value) -> int:
    if torch.is_tensor(value):
        return int(value.numel()) * int(value.element_size())
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)
    if isinstance(value, StepBatch):
        total = sum(
            _tensor_bytes(getattr(value, field.name))
            for field in fields(value)
        )
        total += _tensor_bytes(getattr(value, "auxiliary_targets", None))
        total += _tensor_bytes(getattr(value, "sample_weight", None))
        return total
    if hasattr(value, "__dataclass_fields__"):
        return sum(_tensor_bytes(getattr(value, field.name)) for field in fields(value))
    return 0


def _step_batch_to_device(
    batch: StepBatch,
    device: torch.device | str,
    *,
    non_blocking: bool = False,
) -> StepBatch:
    device = torch.device(device)
    payload = {}
    for field in fields(batch):
        value = getattr(batch, field.name)
        payload[field.name] = (
            value.to(device, non_blocking=non_blocking)
            if torch.is_tensor(value)
            else value
        )
    result = StepBatch(**payload)
    auxiliary = getattr(batch, "auxiliary_targets", None)
    if isinstance(auxiliary, dict):
        result.auxiliary_targets = {
            key: (
                value.to(device, non_blocking=non_blocking)
                if torch.is_tensor(value)
                else value
            )
            for key, value in auxiliary.items()
        }
    sample_weight = getattr(batch, "sample_weight", None)
    if torch.is_tensor(sample_weight):
        result.sample_weight = sample_weight.to(
            device, non_blocking=non_blocking,
        )
    return result


def _dataclass_to_device(value, device, *, non_blocking: bool = False):
    device = torch.device(device)
    payload = {}
    for field in fields(value):
        item = getattr(value, field.name)
        payload[field.name] = (
            item.to(device, non_blocking=non_blocking)
            if torch.is_tensor(item)
            else item
        )
    return type(value)(**payload)


def _pin_step_batch(batch: StepBatch) -> StepBatch:
    for field in fields(batch):
        value = getattr(batch, field.name)
        if torch.is_tensor(value) and value.device.type == "cpu":
            setattr(batch, field.name, value.pin_memory())
    auxiliary = getattr(batch, "auxiliary_targets", None)
    if isinstance(auxiliary, dict):
        batch.auxiliary_targets = {
            key: (
                value.pin_memory()
                if torch.is_tensor(value) and value.device.type == "cpu"
                else value
            )
            for key, value in auxiliary.items()
        }
    sample_weight = getattr(batch, "sample_weight", None)
    if torch.is_tensor(sample_weight) and sample_weight.device.type == "cpu":
        batch.sample_weight = sample_weight.pin_memory()
    return batch


def _pin_tensor_dataclass(value):
    for field in fields(value):
        item = getattr(value, field.name)
        if torch.is_tensor(item) and item.device.type == "cpu":
            setattr(value, field.name, item.pin_memory())
    return value


def _strip_step_batch_python(batch: StepBatch) -> StepBatch:
    batch.structured_states = ()
    batch.canonical_actions = ()
    batch.previous_actions = ()
    batch.effects_targets = ()
    return batch


def _materialize_cached_sequence(
    cached: _CachedTensorSequence,
    device: torch.device | str,
):
    device = torch.device(device)
    if cached.resident_on_device:
        return (
            cached.flat,
            cached.targets,
            cached.ledger,
            cached.strategy_slots,
        )
    non_blocking = bool(device.type == "cuda")
    flat = _step_batch_to_device(
        cached.flat, device, non_blocking=non_blocking,
    )
    targets = _dataclass_to_device(
        cached.targets, device, non_blocking=non_blocking,
    )
    ledger = _dataclass_to_device(
        cached.ledger, device, non_blocking=non_blocking,
    )
    strategy_slots = cached.strategy_slots
    if torch.is_tensor(strategy_slots):
        strategy_slots = strategy_slots.to(
            device, non_blocking=non_blocking,
        )
    return flat, targets, ledger, strategy_slots


def _teacher_cached_tensor_sequence(
    model,
    cached: _CachedTensorSequence,
    states,
    device,
    *,
    family_weights=None,
    market_active_op_weights=None,
    teacher_mix_probability=1.0,
    conditioning_rng=None,
    recurrent_stats=None,
):
    batch_size = len(cached.slots)
    steps = int(cached.steps)
    slots = list(cached.slots)
    packed = _pack_states(states, slots)
    flat, tensor_targets, tensor_ledger, strategy_slots = (
        _materialize_cached_sequence(cached, device)
    )
    seed = 0
    if conditioning_rng is not None and hasattr(conditioning_rng, "integers"):
        seed = int(conditioning_rng.integers(0, 2**31 - 1))
    generator_device = (
        device.type if isinstance(device, torch.device)
        else torch.device(device).type
    )
    tensor_generator = torch.Generator(device=generator_device)
    tensor_generator.manual_seed(seed)

    encoded = model.encoder(flat)
    strategy_context = None
    core_input = encoded.fused
    if strategy_slots is not None:
        if getattr(model, "strategy_embedding", None) is None:
            raise RuntimeError(
                "strategy-conditioned cached path needs an embedding"
            )
        strategy_context = model.strategy_embedding(
            strategy_slots
        ).to(encoded.fused)
        core_context = torch.cat(
            [strategy_context, strategy_context], dim=-1,
        )
        if core_context.shape != encoded.fused.shape:
            raise RuntimeError("strategy core context shape mismatch")
        core_input = (
            encoded.fused
            + float(STRATEGY_CORE_SCALE) * core_context
        )

    def bt(value):
        return value.reshape(
            batch_size, steps, *value.shape[1:],
        )

    fused_seq, base_intent_seq, next_state, diagnostics = model.core.sequence(
        bt(core_input),
        bt(flat.previous_action_global),
        bt(flat.previous_effect),
        bt(flat.economy),
        packed,
    )
    fused_flat = fused_seq.reshape(batch_size * steps, -1)
    base_intent_flat = base_intent_seq.reshape(batch_size * steps, -1)
    intent_flat = (
        base_intent_flat
        if strategy_context is None
        else base_intent_flat + strategy_context
    )
    output = teacher_step_tensor_mixed(
        model,
        flat,
        tensor_targets,
        tensor_ledger,
        packed,
        strategy_slots=strategy_slots,
        teacher_mix_probability=teacher_mix_probability,
        generator=tensor_generator,
        precomputed={
            "encoded": encoded,
            "fused_temporal": fused_flat,
            "intent": intent_flat,
            "temporal_state": next_state,
            "temporal_diagnostics": diagnostics,
            "strategy_context": strategy_context,
        },
    )
    losses = tensor_total_pretrain_loss_sequence(
        output,
        flat,
        tensor_targets,
        step=tensor_ledger.step,
        batch_size=batch_size,
        steps=steps,
        family_weights=family_weights,
        market_active_op_weights=market_active_op_weights,
    )
    _store_state(states, slots, next_state)
    if recurrent_stats is not None:
        recurrent_stats["temporal_steps"] += batch_size * steps
        recurrent_stats["tensor_sequence_chunks"] = (
            int(recurrent_stats.get("tensor_sequence_chunks", 0)) + 1
        )
        recurrent_stats["cached_tensor_sequence_chunks"] = (
            int(recurrent_stats.get("cached_tensor_sequence_chunks", 0)) + 1
        )
    return losses, states


def _teacher_chunk_cached(
    model, active_chunks, states, device, recurrent_stats=None,
    strategy_manifest: StrategyManifest | None = None,
    family_weights: dict[str, float] | None = None,
    market_active_op_weights: dict[str, float] | None = None,
    teacher_mix_probability: float = 1.0,
    conditioning_rng=None,
    gpu_tensor_training: bool = False,
):
    if isinstance(active_chunks, _CachedTensorSequence):
        return _teacher_cached_tensor_sequence(
            model,
            active_chunks,
            states,
            device,
            family_weights=family_weights,
            market_active_op_weights=market_active_op_weights,
            teacher_mix_probability=teacher_mix_probability,
            conditioning_rng=conditioning_rng,
            recurrent_stats=recurrent_stats,
        )
    chunks = [chunk for _, chunk in active_chunks]
    max_len = max(len(chunk.rows) for chunk in chunks)
    sequence = collate_v2_sequences(chunks, max_len)
    tensor_targets = None
    tensor_ledger = None
    tensor_generator = None
    if gpu_tensor_training:
        tensor_targets = TensorActionTargets.from_actions(
            sequence.flat.canonical_actions,
            max_units=sequence.flat.own_units.shape[1],
        ).to(device)
        tensor_ledger = TensorLedger.from_batch(
            sequence.flat,
            structured_states=sequence.flat.structured_states,
            device=device,
        )
        seed = 0
        if conditioning_rng is not None and hasattr(conditioning_rng, "integers"):
            seed = int(conditioning_rng.integers(0, 2**31 - 1))
        generator_device = (
            device.type if isinstance(device, torch.device) else torch.device(device).type
        )
        tensor_generator = torch.Generator(device=generator_device)
        tensor_generator.manual_seed(seed)
    move_step_batch(sequence.flat, device)
    equal_lengths = len({len(chunk.rows) for chunk in chunks}) == 1
    if (
        gpu_tensor_training
        and equal_lengths
        and max_len <= int(getattr(model.core, "window", 0))
    ):
        return _teacher_chunk_tensor_sequence(
            model,
            active_chunks,
            states,
            device,
            sequence,
            tensor_targets,
            tensor_ledger,
            tensor_generator,
            strategy_manifest=strategy_manifest,
            family_weights=family_weights,
            market_active_op_weights=market_active_op_weights,
            teacher_mix_probability=teacher_mix_probability,
            recurrent_stats=recurrent_stats,
        )
    step_losses = []
    for time_index in range(max_len):
        active_positions = [
            i for i, chunk in enumerate(chunks) if time_index < len(chunk.rows)
        ]
        slots = [active_chunks[i][0] for i in active_positions]
        row_indices = [
            int(sequence.row_slots[i, time_index].item()) for i in active_positions
        ]
        batch = _slice_step_batch(sequence.flat, row_indices)
        packed = _pack_states(states, slots)
        strategy_slots = (
            _strategy_slots_for_rows(
                [chunks[i].rows[time_index] for i in active_positions],
                strategy_manifest, device,
            ) if strategy_manifest is not None else None
        )
        if gpu_tensor_training:
            index = torch.tensor(
                row_indices, dtype=torch.long, device=device,
            )
            step_targets = tensor_targets.index_select(index)
            step_ledger = tensor_ledger.index_select(index)
            output = teacher_step_tensor_mixed(
                model,
                batch,
                step_targets,
                step_ledger,
                packed,
                strategy_slots=strategy_slots,
                teacher_mix_probability=teacher_mix_probability,
                generator=tensor_generator,
            )
            step_losses.append(tensor_total_pretrain_loss(
                output,
                batch,
                step_targets,
                step=step_ledger.step,
                family_weights=family_weights,
                market_active_op_weights=market_active_op_weights,
            ))
        else:
            output = model.teacher_step(
                batch, batch.canonical_actions, packed,
                strategy_slots=strategy_slots,
                teacher_mix_probability=teacher_mix_probability,
                conditioning_rng=conditioning_rng,
            )
            step_losses.append(total_pretrain_loss(
                output, batch, family_weights=family_weights,
                market_active_op_weights=market_active_op_weights,
            ))
        _store_state(states, slots, output.temporal_state)
        if recurrent_stats is not None:
            recurrent_stats["temporal_steps"] += len(slots)
    return _mean_tensor_dict(step_losses), states



def _gpu_batch_disk_cache_identity(
    config: BCV3Config,
    *,
    dataset_sha: str,
    effective_action_sha: str | None,
    strategy_manifest: StrategyManifest | None,
) -> tuple[Path, dict[str, Any]]:
    identity = {
        "format_version": int(GPU_BATCH_CACHE_FORMAT_VERSION),
        "dataset_sha256": str(dataset_sha),
        "effective_action_sha256": effective_action_sha,
        "sequence_len": int(config.sequence_len),
        "batch_sequences": int(config.batch_sequences),
        "seed": int(config.seed),
        "model_architecture": str(config.model_architecture),
        "strategy_manifest_sha256": (
            None if strategy_manifest is None else strategy_manifest.sha256
        ),
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:24]
    root = (
        Path(config.dataset_path).resolve().parent
        / ".farmos_tensor_cache"
        / digest
    )
    return root, identity


def _gpu_batch_disk_cache_specs(dataset, config: BCV3Config):
    groups = []
    for episodes in _episode_groups(
        dataset,
        config.batch_sequences,
        config.seed,
        0,
    ):
        chunks_by_slot = [
            _episode_chunks(ep, config.sequence_len)
            for ep in episodes
        ]
        max_chunks = max(len(chunks) for chunks in chunks_by_slot)
        updates = []
        for chunk_index in range(max_chunks):
            active = tuple(
                (slot, chunks[chunk_index])
                for slot, chunks in enumerate(chunks_by_slot)
                if chunk_index < len(chunks)
            )
            updates.append(active)
        groups.append((len(chunks_by_slot), tuple(updates)))
    return tuple(groups)


def _place_cached_tensors(
    flat,
    targets,
    ledger,
    strategy_slots,
    *,
    device: torch.device,
    reserve_bytes: int,
    tensor_bytes: int,
    pin_cpu: bool,
):
    resident_on_device = False
    if device.type == "cuda" and torch.cuda.is_available():
        free_bytes, _ = torch.cuda.mem_get_info(device)
        if tensor_bytes <= max(0, int(free_bytes) - reserve_bytes):
            try:
                flat = _step_batch_to_device(flat, device)
                targets = _dataclass_to_device(targets, device)
                ledger = _dataclass_to_device(ledger, device)
                if torch.is_tensor(strategy_slots):
                    strategy_slots = strategy_slots.to(device)
                resident_on_device = True
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                resident_on_device = False

    if (
        not resident_on_device
        and pin_cpu
        and device.type == "cuda"
        and torch.cuda.is_available()
    ):
        try:
            flat = _pin_step_batch(flat)
            targets = _pin_tensor_dataclass(targets)
            ledger = _pin_tensor_dataclass(ledger)
            if (
                torch.is_tensor(strategy_slots)
                and strategy_slots.device.type == "cpu"
            ):
                strategy_slots = strategy_slots.pin_memory()
        except RuntimeError:
            pass
    return flat, targets, ledger, strategy_slots, resident_on_device


def _try_load_gpu_batch_disk_cache(
    dataset,
    config: BCV3Config,
    device,
    strategy_manifest: StrategyManifest | None,
    *,
    dataset_sha: str | None,
    effective_action_sha: str | None,
):
    if (
        not bool(config.gpu_batch_cache_disk)
        or not dataset_sha
    ):
        return None

    cache_root, identity = _gpu_batch_disk_cache_identity(
        config,
        dataset_sha=dataset_sha,
        effective_action_sha=effective_action_sha,
        strategy_manifest=strategy_manifest,
    )
    manifest_path = cache_root / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        manifest.get("complete") is not True
        or manifest.get("identity") != identity
    ):
        return None

    specs = _gpu_batch_disk_cache_specs(dataset, config)
    file_names = list(manifest.get("files") or [])
    files = [cache_root / str(name) for name in file_names]
    if not files or any(not path.is_file() for path in files):
        return None

    started = time.perf_counter()
    device = torch.device(device)
    reserve_bytes = int(
        float(config.gpu_batch_cache_reserve_gb) * (1024 ** 3)
    )
    stats = {
        "enabled": True,
        "disk_cache_hit": True,
        "disk_cache_dir": str(cache_root),
        "cached_updates": 0,
        "gpu_resident_updates": 0,
        "cpu_resident_updates": 0,
        "fallback_updates": 0,
        "tensor_bytes": 0,
        "gpu_tensor_bytes": 0,
        "cpu_tensor_bytes": 0,
        "built_updates": 0,
        "collate_seconds": 0.0,
        "target_seconds": 0.0,
        "ledger_seconds": 0.0,
        "transfer_seconds": 0.0,
        "build_seconds": 0.0,
        "disk_load_seconds": 0.0,
    }
    groups = []
    file_cursor = 0
    for group_index, (slot_count, updates_spec) in enumerate(specs):
        updates = []
        for update_index, active in enumerate(updates_spec):
            chunks = [chunk for _, chunk in active]
            lengths = [len(chunk.rows) for chunk in chunks]
            cacheable = (
                bool(lengths)
                and len(set(lengths)) == 1
                and lengths[0] <= int(manifest.get("core_window", 10**9))
            )
            cached = None
            if cacheable:
                if file_cursor >= len(files):
                    return None
                payload = torch.load(
                    files[file_cursor],
                    map_location="cpu",
                    weights_only=False,
                )
                file_cursor += 1
                slots = tuple(int(value) for value in payload["slots"])
                if slots != tuple(slot for slot, _ in active):
                    return None
                tensor_bytes = int(payload["tensor_bytes"])
                transfer_started = time.perf_counter()
                (
                    flat,
                    targets,
                    ledger,
                    strategy_slots,
                    resident_on_device,
                ) = _place_cached_tensors(
                    payload["flat"],
                    payload["targets"],
                    payload["ledger"],
                    payload.get("strategy_slots"),
                    device=device,
                    reserve_bytes=reserve_bytes,
                    tensor_bytes=tensor_bytes,
                    pin_cpu=bool(config.gpu_batch_cache_pin_cpu),
                )
                stats["transfer_seconds"] += float(
                    time.perf_counter() - transfer_started
                )
                cached = _CachedTensorSequence(
                    slots=slots,
                    steps=int(payload["steps"]),
                    flat=flat,
                    targets=targets,
                    ledger=ledger,
                    strategy_slots=strategy_slots,
                    resident_on_device=resident_on_device,
                    tensor_bytes=tensor_bytes,
                )
                stats["cached_updates"] += 1
                stats["tensor_bytes"] += tensor_bytes
                if resident_on_device:
                    stats["gpu_resident_updates"] += 1
                    stats["gpu_tensor_bytes"] += tensor_bytes
                else:
                    stats["cpu_resident_updates"] += 1
                    stats["cpu_tensor_bytes"] += tensor_bytes
            else:
                stats["fallback_updates"] += 1
            stats["built_updates"] += 1
            updates.append(
                _PreparedTrainingUpdate(active=active, cached=cached)
            )
        groups.append(
            _PreparedTrainingGroup(
                slot_count=int(slot_count),
                updates=tuple(updates),
            )
        )

    if file_cursor != len(files):
        return None
    stats["disk_load_seconds"] = float(time.perf_counter() - started)
    stats["build_seconds"] = stats["disk_load_seconds"]
    payload = {
        **stats,
        "tensor_gb": float(stats["tensor_bytes"]) / (1024 ** 3),
        "gpu_tensor_gb": float(stats["gpu_tensor_bytes"]) / (1024 ** 3),
        "cpu_tensor_gb": float(stats["cpu_tensor_bytes"]) / (1024 ** 3),
        "reserve_gb": float(config.gpu_batch_cache_reserve_gb),
        "groups": len(groups),
    }
    print(
        "FARMOS_GPU_BATCH_CACHE="
        + json.dumps(payload, sort_keys=True),
        flush=True,
    )
    return tuple(groups), stats


def _prepare_gpu_batch_cache(
    model,
    dataset,
    config,
    device,
    strategy_manifest: StrategyManifest | None,
    *,
    dataset_sha: str | None = None,
    effective_action_sha: str | None = None,
):
    if not bool(config.gpu_batch_cache):
        return None, {
            "enabled": False,
            "cached_updates": 0,
            "gpu_resident_updates": 0,
            "cpu_resident_updates": 0,
            "fallback_updates": 0,
            "tensor_bytes": 0,
            "gpu_tensor_bytes": 0,
            "cpu_tensor_bytes": 0,
            "built_updates": 0,
            "collate_seconds": 0.0,
            "target_seconds": 0.0,
            "ledger_seconds": 0.0,
            "transfer_seconds": 0.0,
            "build_seconds": 0.0,
        }

    disk_loaded = _try_load_gpu_batch_disk_cache(
        dataset,
        config,
        device,
        strategy_manifest,
        dataset_sha=dataset_sha,
        effective_action_sha=effective_action_sha,
    )
    if disk_loaded is not None:
        return disk_loaded

    started = time.perf_counter()
    device = torch.device(device)
    reserve_bytes = int(
        float(config.gpu_batch_cache_reserve_gb) * (1024 ** 3)
    )
    groups = []
    stats = {
        "enabled": True,
        "cached_updates": 0,
        "gpu_resident_updates": 0,
        "cpu_resident_updates": 0,
        "fallback_updates": 0,
        "tensor_bytes": 0,
        "gpu_tensor_bytes": 0,
        "cpu_tensor_bytes": 0,
        "built_updates": 0,
        "collate_seconds": 0.0,
        "target_seconds": 0.0,
        "ledger_seconds": 0.0,
        "transfer_seconds": 0.0,
        "build_seconds": 0.0,
        "disk_cache_hit": False,
        "disk_write_seconds": 0.0,
    }
    disk_cache_root = None
    disk_cache_identity = None
    disk_cache_files: list[str] = []
    if bool(config.gpu_batch_cache_disk) and dataset_sha:
        disk_cache_root, disk_cache_identity = (
            _gpu_batch_disk_cache_identity(
                config,
                dataset_sha=dataset_sha,
                effective_action_sha=effective_action_sha,
                strategy_manifest=strategy_manifest,
            )
        )
        disk_cache_root.mkdir(parents=True, exist_ok=True)
        stats["disk_cache_dir"] = str(disk_cache_root)

    for episodes in _episode_groups(
        dataset,
        config.batch_sequences,
        config.seed,
        0,
    ):
        group_index = len(groups)
        chunks_by_slot = [
            _episode_chunks(ep, config.sequence_len)
            for ep in episodes
        ]
        max_chunks = max(len(chunks) for chunks in chunks_by_slot)
        updates = []
        for chunk_index in range(max_chunks):
            active = tuple(
                (slot, chunks[chunk_index])
                for slot, chunks in enumerate(chunks_by_slot)
                if chunk_index < len(chunks)
            )
            chunks = [chunk for _, chunk in active]
            lengths = [len(chunk.rows) for chunk in chunks]
            cacheable = (
                bool(lengths)
                and len(set(lengths)) == 1
                and lengths[0] <= int(getattr(model.core, "window", 0))
            )
            cached = None
            if cacheable:
                steps = int(lengths[0])
                stage_started = time.perf_counter()
                sequence = collate_v2_sequences(chunks, steps)
                stats["collate_seconds"] += float(
                    time.perf_counter() - stage_started
                )
                flat_rows = [
                    row
                    for chunk in chunks
                    for row in chunk.rows
                ]
                stage_started = time.perf_counter()
                targets = TensorActionTargets.from_actions(
                    sequence.flat.canonical_actions,
                    max_units=sequence.flat.own_units.shape[1],
                )
                stats["target_seconds"] += float(
                    time.perf_counter() - stage_started
                )
                stage_started = time.perf_counter()
                ledger = TensorLedger.from_batch(
                    sequence.flat,
                    structured_states=sequence.flat.structured_states,
                    device="cpu",
                )
                stats["ledger_seconds"] += float(
                    time.perf_counter() - stage_started
                )
                strategy_slots = (
                    _strategy_slots_for_rows(
                        flat_rows,
                        strategy_manifest,
                        torch.device("cpu"),
                    )
                    if strategy_manifest is not None
                    else None
                )
                flat = _strip_step_batch_python(sequence.flat)
                tensor_bytes = (
                    _tensor_bytes(flat)
                    + _tensor_bytes(targets)
                    + _tensor_bytes(ledger)
                    + _tensor_bytes(strategy_slots)
                )
                if disk_cache_root is not None:
                    file_name = (
                        f"group_{group_index:03d}_"
                        f"update_{chunk_index:03d}.pt"
                    )
                    cache_path = disk_cache_root / file_name
                    temp_path = cache_path.with_suffix(".tmp")
                    disk_started = time.perf_counter()
                    torch.save({
                        "slots": tuple(slot for slot, _ in active),
                        "steps": int(steps),
                        "flat": flat,
                        "targets": targets,
                        "ledger": ledger,
                        "strategy_slots": strategy_slots,
                        "tensor_bytes": int(tensor_bytes),
                    }, temp_path)
                    temp_path.replace(cache_path)
                    stats["disk_write_seconds"] += float(
                        time.perf_counter() - disk_started
                    )
                    disk_cache_files.append(file_name)

                resident_on_device = False
                transfer_started = time.perf_counter()

                if device.type == "cuda" and torch.cuda.is_available():
                    free_bytes, _ = torch.cuda.mem_get_info(device)
                    if tensor_bytes <= max(0, int(free_bytes) - reserve_bytes):
                        try:
                            device_flat = _step_batch_to_device(flat, device)
                            device_targets = _dataclass_to_device(
                                targets, device,
                            )
                            device_ledger = _dataclass_to_device(
                                ledger, device,
                            )
                            device_strategy_slots = strategy_slots
                            if torch.is_tensor(strategy_slots):
                                device_strategy_slots = strategy_slots.to(device)
                            flat = device_flat
                            targets = device_targets
                            ledger = device_ledger
                            strategy_slots = device_strategy_slots
                            resident_on_device = True
                        except torch.cuda.OutOfMemoryError:
                            torch.cuda.empty_cache()
                            resident_on_device = False

                if (
                    not resident_on_device
                    and bool(config.gpu_batch_cache_pin_cpu)
                    and device.type == "cuda"
                    and torch.cuda.is_available()
                ):
                    try:
                        flat = _pin_step_batch(flat)
                        targets = _pin_tensor_dataclass(targets)
                        ledger = _pin_tensor_dataclass(ledger)
                        if (
                            torch.is_tensor(strategy_slots)
                            and strategy_slots.device.type == "cpu"
                        ):
                            strategy_slots = strategy_slots.pin_memory()
                    except RuntimeError:
                        pass
                stats["transfer_seconds"] += float(
                    time.perf_counter() - transfer_started
                )

                cached = _CachedTensorSequence(
                    slots=tuple(slot for slot, _ in active),
                    steps=steps,
                    flat=flat,
                    targets=targets,
                    ledger=ledger,
                    strategy_slots=strategy_slots,
                    resident_on_device=resident_on_device,
                    tensor_bytes=tensor_bytes,
                )
                stats["cached_updates"] += 1
                stats["tensor_bytes"] += int(tensor_bytes)
                if resident_on_device:
                    stats["gpu_resident_updates"] += 1
                    stats["gpu_tensor_bytes"] += int(tensor_bytes)
                else:
                    stats["cpu_resident_updates"] += 1
                    stats["cpu_tensor_bytes"] += int(tensor_bytes)
            else:
                stats["fallback_updates"] += 1
            updates.append(_PreparedTrainingUpdate(
                active=active,
                cached=cached,
            ))
            stats["built_updates"] += 1
            cache_progress_every = max(
                1, int(getattr(config, "progress_every", 5) or 5)
            )
            if (
                stats["built_updates"] % cache_progress_every == 0
                or chunk_index + 1 == max_chunks
            ):
                print(
                    "FARMOS_GPU_BATCH_CACHE_PROGRESS="
                    + json.dumps({
                        "built_updates": int(stats["built_updates"]),
                        "group": int(len(groups) + 1),
                        "group_update": int(chunk_index + 1),
                        "group_updates": int(max_chunks),
                        "active_sequences": int(len(active)),
                        "rows": int(sum(lengths)),
                        "cached": bool(cached is not None),
                        "gpu_resident_updates": int(
                            stats["gpu_resident_updates"]
                        ),
                        "cpu_resident_updates": int(
                            stats["cpu_resident_updates"]
                        ),
                        "collate_seconds": float(
                            stats["collate_seconds"]
                        ),
                        "ledger_seconds": float(
                            stats["ledger_seconds"]
                        ),
                        "elapsed_seconds": float(
                            time.perf_counter() - started
                        ),
                    }, sort_keys=True),
                    flush=True,
                )
        groups.append(_PreparedTrainingGroup(
            slot_count=len(chunks_by_slot),
            updates=tuple(updates),
        ))

    stats["build_seconds"] = float(time.perf_counter() - started)
    if disk_cache_root is not None and disk_cache_identity is not None:
        manifest_payload = {
            "complete": True,
            "identity": disk_cache_identity,
            "core_window": int(getattr(model.core, "window", 0)),
            "files": list(disk_cache_files),
            "tensor_bytes": int(stats["tensor_bytes"]),
            "cached_updates": int(stats["cached_updates"]),
            "fallback_updates": int(stats["fallback_updates"]),
        }
        manifest_path = disk_cache_root / "manifest.json"
        temp_manifest = disk_cache_root / "manifest.tmp"
        temp_manifest.write_text(
            json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temp_manifest.replace(manifest_path)
    payload = {
        **stats,
        "tensor_gb": float(stats["tensor_bytes"]) / (1024 ** 3),
        "gpu_tensor_gb": float(stats["gpu_tensor_bytes"]) / (1024 ** 3),
        "cpu_tensor_gb": float(stats["cpu_tensor_bytes"]) / (1024 ** 3),
        "reserve_gb": float(config.gpu_batch_cache_reserve_gb),
        "groups": len(groups),
    }
    print(
        "FARMOS_GPU_BATCH_CACHE="
        + json.dumps(payload, sort_keys=True),
        flush=True,
    )
    return tuple(groups), stats


def _recovery_chunks(rows, sequence_len):
    grouped = {}
    for row in rows:
        key = (int(row["episode_id"]), int(row["seat"]))
        grouped.setdefault(key, []).append(dict(row))
    chunks = []
    for key in sorted(grouped):
        ordered = sorted(grouped[key], key=lambda item: int(item["step"]))
        for start in range(0, len(ordered), sequence_len):
            part = tuple(ordered[start:start + sequence_len])
            chunks.append(SequenceChunk(
                key[0], key[1], part, start == 0, start + sequence_len >= len(ordered),
            ))
    if not chunks:
        raise ValueError("recovery dataset produced no chunks")
    return chunks


def _recovery_step_loss(
    output, targets, *, supervision_kind: str, family_weights=None,
    market_active_op_weights=None,
):
    domain = action_loss(
        output,
        targets,
        family_weights=family_weights,
        market_active_op_weights=market_active_op_weights,
    )
    kind = str(supervision_kind)
    if kind == "smoke_only":
        return domain.market
    if kind in {"expert", "accepted_policy"}:
        return domain.total
    raise ValueError(f"unknown recovery supervision_kind: {kind}")


def _recovery_update(
    model, optimizer, chunk, state, config, device, family_weights=None,
    market_active_op_weights=None,
    strategy_manifest: StrategyManifest | None = None,
):
    kinds = {
        str(row.get("supervision_kind", ""))
        for row in chunk.rows
    }
    use_tensor_recovery = bool(
        getattr(config, "gpu_tensor_training", False)
        and kinds
        and kinds.issubset({"expert", "accepted_policy"})
        and len(chunk.rows) <= int(getattr(model.core, "window", 0))
    )

    if use_tensor_recovery:
        prepared_rows = [dict(row) for row in chunk.rows]
        _attach_auxiliary_targets(prepared_rows)
        prepared_chunk = SequenceChunk(
            episode_id=int(chunk.episode_id),
            seat=int(chunk.seat),
            rows=tuple(prepared_rows),
            episode_start=bool(chunk.episode_start),
            episode_end=bool(chunk.episode_end),
        )
        states = [state]
        tensor_losses, states = _teacher_chunk_cached(
            model,
            [(0, prepared_chunk)],
            states,
            device,
            recurrent_stats=None,
            strategy_manifest=strategy_manifest,
            family_weights=family_weights,
            market_active_op_weights=market_active_op_weights,
            teacher_mix_probability=1.0,
            conditioning_rng=np.random.default_rng(
                int(chunk.episode_id) * 1009 + int(chunk.start_step)
            ),
            gpu_tensor_training=True,
        )
        total = tensor_losses["action"]
        state = states[0]
    else:
        losses = []
        for row in chunk.rows:
            batch = collate_transitions([row])
            move_step_batch(batch, device)
            strategy_slots = None
            if getattr(model, "strategy_embedding", None) is not None:
                if row.get("strategy_slot") is None:
                    raise ValueError(
                        "strategy-conditioned recovery row requires strategy_slot"
                    )
                slot = int(row["strategy_slot"])
                strategy_count = int(
                    getattr(model, "strategy_count", 0) or 0
                )
                if not 0 <= slot < strategy_count:
                    raise ValueError(
                        f"recovery strategy_slot {slot} outside "
                        f"[0, {strategy_count - 1}]"
                    )
                strategy_slots = torch.tensor(
                    [slot], dtype=torch.long, device=device,
                )
            output = model.teacher_step(
                batch,
                batch.canonical_actions,
                state,
                strategy_slots=strategy_slots,
            )
            losses.append(_recovery_step_loss(
                output,
                batch.canonical_actions,
                supervision_kind=row.get("supervision_kind", ""),
                family_weights=family_weights,
                market_active_op_weights=market_active_op_weights,
            ))
            state = output.temporal_state
        total = torch.stack(losses).mean()

    if not torch.isfinite(total):
        raise RuntimeError("non-finite v3 recovery action loss")
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), config.gradient_clip,
    )
    if not torch.isfinite(torch.as_tensor(norm)):
        raise RuntimeError("non-finite v3 recovery gradient norm")
    optimizer.step()
    return (
        float(total.detach().cpu().item()),
        model.core.detach_state(state),
    )


def _finish_optimizer_step(
    model, optimizer, config, *, amp_enabled: bool, scaler=None,
):
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), config.gradient_clip,
    )
    norm_tensor = torch.as_tensor(norm)
    if amp_enabled:
        if scaler is None:
            raise RuntimeError("AMP training requires a GradScaler")
        # There are only a few dozen optimizer updates per epoch, so a
        # per-update scale/norm check is cheap and prevents silent corruption.
        scale_before = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale())
        norm_value = float(norm_tensor.detach().float().cpu())
        finite = math.isfinite(norm_value) and scale_after >= scale_before
        return norm_value, finite, scale_before, scale_after

    finite = bool(torch.isfinite(norm_tensor).item())
    if not finite:
        raise RuntimeError("non-finite v3 BC gradient norm")
    optimizer.step()
    return float(norm_tensor.detach().cpu()), True, None, None


def _train_epoch(
    model, optimizer, dataset, config, epoch,
    train_steps, recurrent_stats, device, recovery_chunks=None,
    recovery_cursor=0, recovery_states=None,
    strategy_manifest: StrategyManifest | None = None,
    family_weights: dict[str, float] | None = None,
    market_active_op_weights: dict[str, float] | None = None,
    scaler=None,
    prepared_training_groups=None,
):
    model.train()
    teacher_mix_probability = _teacher_mix_for_epoch(
        config.teacher_mix_schedule, epoch,
    )
    conditioning_rng = np.random.default_rng(
        int(config.seed) + 3000001 * int(epoch),
    )
    metric_sums: dict[str, torch.Tensor] = {}
    metric_count = 0
    recovery_losses = []
    loss_ema = None
    stop = False
    epoch_start_steps = int(train_steps)
    epoch_start_expert_updates = int(recurrent_stats["expert_updates"])
    epoch_start_recovery_updates = int(recurrent_stats["recovery_updates"])
    epoch_start_temporal = int(recurrent_stats["temporal_steps"])
    epoch_started = time.perf_counter()
    amp_device_type = getattr(device, "type", str(device).split(":")[0])
    amp_enabled = bool(config.use_amp and amp_device_type == "cuda")
    last_logged_amp_scale = None
    if (
        amp_enabled
        and scaler is not None
        and int(config.progress_every) > 0
    ):
        last_logged_amp_scale = float(scaler.get_scale())
    if prepared_training_groups is None:
        prepared_groups = []
        for episodes in _episode_groups(
            dataset, config.batch_sequences, config.seed, epoch,
        ):
            prepared_groups.append([
                _episode_chunks(ep, config.sequence_len) for ep in episodes
            ])
        epoch_expected_updates = sum(
            max(len(chunks) for chunks in group)
            for group in prepared_groups
        )
    else:
        prepared_groups = list(prepared_training_groups)
        random.Random(
            int(config.seed) + 1000003 * int(epoch)
        ).shuffle(prepared_groups)
        epoch_expected_updates = sum(
            len(group.updates) for group in prepared_groups
        )
    if config.max_train_steps_per_epoch is not None:
        epoch_expected_updates = min(
            epoch_expected_updates, int(config.max_train_steps_per_epoch),
        )
    if amp_device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    def epoch_step_cap_reached() -> bool:
        cap = config.max_train_steps_per_epoch
        return cap is not None and (int(train_steps) - epoch_start_steps) >= int(cap)
    recovery_chunks = list(recovery_chunks or [])
    recovery_states = {} if recovery_states is None else recovery_states

    def run_recovery_if_due():
        nonlocal train_steps, recovery_cursor
        if not recovery_chunks:
            return False
        if (train_steps + 1) % int(config.recovery_every) != 0:
            return False
        chunk = recovery_chunks[recovery_cursor % len(recovery_chunks)]
        recovery_cursor += 1
        key = (int(chunk.episode_id), int(chunk.seat))
        state = None if chunk.episode_start else recovery_states.get(key)
        _set_optimizer_lr(
            optimizer,
            _scheduled_learning_rate(
                config, epoch=epoch, train_steps=train_steps,
            ),
        )
        loss, state = _recovery_update(
            model, optimizer, chunk, state, config, device,
            family_weights=family_weights,
            market_active_op_weights=market_active_op_weights,
            strategy_manifest=strategy_manifest,
        )
        recovery_states[key] = None if chunk.episode_end else state
        recurrent_stats["recovery_updates"] += 1
        recurrent_stats["recovery_temporal_steps"] += len(chunk.rows)
        train_steps += 1
        recovery_losses.append(float(loss))
        return True

    def iter_group_updates(group):
        if isinstance(group, _PreparedTrainingGroup):
            for update in group.updates:
                yield list(update.active), update.cached
            return
        max_chunks = max(len(chunks) for chunks in group)
        for chunk_index in range(max_chunks):
            active = [
                (slot, chunks[chunk_index])
                for slot, chunks in enumerate(group)
                if chunk_index < len(chunks)
            ]
            yield active, None

    for group in prepared_groups:
        if isinstance(group, _PreparedTrainingGroup):
            states = [None] * int(group.slot_count)
        else:
            states = [None] * len(group)
        for active, cached_update in iter_group_updates(group):
            if (
                (config.max_train_steps is not None and train_steps >= config.max_train_steps)
                or epoch_step_cap_reached()
            ):
                stop = True; break
            if run_recovery_if_due():
                if (
                    (config.max_train_steps is not None and train_steps >= config.max_train_steps)
                    or epoch_step_cap_reached()
                ):
                    stop = True; break
            teacher_input = (
                cached_update if cached_update is not None else active
            )
            for slot, _ in active:
                recurrent_stats["state_resets" if states[slot] is None else "state_carries"] += 1
            current_lr = _scheduled_learning_rate(
                config, epoch=epoch, train_steps=train_steps,
            )
            _set_optimizer_lr(optimizer, current_lr)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=amp_device_type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                losses, states = _teacher_chunk_cached(
                    model, teacher_input, states, device, recurrent_stats,
                    strategy_manifest=strategy_manifest,
                    family_weights=family_weights,
                    market_active_op_weights=market_active_op_weights,
                    teacher_mix_probability=teacher_mix_probability,
                    conditioning_rng=conditioning_rng,
                    gpu_tensor_training=bool(config.gpu_tensor_training),
                )
            loss_snapshot = {
                key: float(value.detach().float().cpu())
                for key, value in losses.items()
            }
            nonfinite_losses = sorted(
                key
                for key, value in loss_snapshot.items()
                if not math.isfinite(value)
            )
            if nonfinite_losses:
                recurrent_stats["nonfinite_loss_skips"] += 1
                skip_count = int(
                    recurrent_stats["nonfinite_loss_skips"]
                )
                optimizer.zero_grad(set_to_none=True)
                for slot, _ in active:
                    states[slot] = None
                scale_before = None
                scale_after = None
                if amp_enabled and scaler is not None:
                    scale_before = float(scaler.get_scale())
                    scale_after = scale_before
                print(
                    "V3_BC_NONFINITE_LOSS="
                    + json.dumps({
                        "epoch": int(epoch),
                        "train_steps": int(train_steps),
                        "components": nonfinite_losses,
                        "losses": {
                            key: (
                                value
                                if math.isfinite(value)
                                else str(value)
                            )
                            for key, value in loss_snapshot.items()
                        },
                        "amp_scale_before": scale_before,
                        "amp_scale_after": scale_after,
                        "skip_count": skip_count,
                    }, sort_keys=True),
                    flush=True,
                )
                if skip_count > 3:
                    raise RuntimeError(
                        "repeated non-finite v3 BC loss: "
                        + ",".join(nonfinite_losses)
                    )
                continue
            if amp_enabled:
                if scaler is None:
                    raise RuntimeError("AMP training requires a GradScaler")
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
            else:
                losses["total"].backward()
            norm, gradient_finite, scale_before, scale_after = _finish_optimizer_step(
                model,
                optimizer,
                config,
                amp_enabled=amp_enabled,
                scaler=scaler,
            )
            states = _detach_states(model, states)
            if amp_enabled and not gradient_finite:
                recurrent_stats["amp_overflow_skips"] += 1
                optimizer.zero_grad(set_to_none=True)
                print(
                    "V3_BC_AMP_OVERFLOW=" + json.dumps({
                        "epoch": int(epoch),
                        "gradient_norm": float(norm),
                        "scale_before": float(scale_before),
                        "scale_after": float(scale_after),
                        "overflow_skips": int(recurrent_stats["amp_overflow_skips"]),
                    }, sort_keys=True),
                    flush=True,
                )
                continue
            detached_losses = {
                key: value.detach() for key, value in losses.items()
            }
            for key, value in detached_losses.items():
                if key not in metric_sums:
                    metric_sums[key] = torch.zeros_like(value)
                metric_sums[key] = metric_sums[key] + value
            metric_count += 1
            current_total = detached_losses["total"]
            loss_ema = (
                current_total
                if loss_ema is None
                else 0.90 * loss_ema + 0.10 * current_total
            )
            recurrent_stats["expert_updates"] += 1
            train_steps += 1
            if (
                int(config.progress_every) > 0
                and train_steps % int(config.progress_every) == 0
            ):
                metric_row = _float_dict(detached_losses)
                loss_ema_value = float(loss_ema.detach().cpu())
                current_amp_scale = None
                if amp_enabled and scaler is not None:
                    current_amp_scale = float(scaler.get_scale())
                    last_logged_amp_scale = current_amp_scale
                elapsed = max(time.perf_counter() - epoch_started, 1e-9)
                epoch_step = int(train_steps) - epoch_start_steps
                epoch_expert_step = (
                    int(recurrent_stats["expert_updates"])
                    - epoch_start_expert_updates
                )
                temporal_delta = int(recurrent_stats["temporal_steps"]) - epoch_start_temporal
                rate = float(temporal_delta) / elapsed
                eta = (
                    max(0, epoch_expected_updates - epoch_expert_step)
                    * elapsed / max(epoch_expert_step, 1)
                )
                progress = {
                    "epoch": int(epoch),
                    "epoch_step": int(epoch_step),
                    "epoch_expert_step": int(epoch_expert_step),
                    "epoch_expert_steps": int(epoch_expected_updates),
                    "epoch_recovery_steps": int(
                        int(recurrent_stats["recovery_updates"])
                        - epoch_start_recovery_updates
                    ),
                    "train_steps": int(train_steps),
                    "active_sequences": int(len(active)),
                    "temporal_steps": int(recurrent_stats["temporal_steps"]),
                    "temporal_steps_per_sec": float(rate),
                    "eta_seconds": float(eta),
                    "loss_total": float(metric_row["total"]),
                    "loss_ema": float(loss_ema_value),
                    "loss_action": float(metric_row.get("action", 0.0)),
                    "loss_market": float(metric_row.get("market", 0.0)),
                    "loss_economic": (
                        None
                        if "economic" not in metric_row
                        else float(metric_row["economic"])
                    ),
                    "learning_rate": float(current_lr),
                    "amp": bool(amp_enabled),
                    "amp_scale": current_amp_scale,
                    "amp_overflow_skips": int(recurrent_stats["amp_overflow_skips"]),
                    "nonfinite_loss_skips": int(
                        recurrent_stats["nonfinite_loss_skips"]
                    ),
                    "gpu_tensor_training": bool(config.gpu_tensor_training),
                    "tensor_sequence_chunks": int(
                        recurrent_stats.get("tensor_sequence_chunks", 0)
                    ),
                    "cached_tensor_sequence_chunks": int(
                        recurrent_stats.get("cached_tensor_sequence_chunks", 0)
                    ),
                    "gpu_batch_cache": bool(config.gpu_batch_cache),
                }
                if amp_device_type == "cuda" and torch.cuda.is_available():
                    progress["gpu_memory_mb"] = float(
                        torch.cuda.memory_allocated(device) / (1024 ** 2)
                    )
                    progress["gpu_peak_memory_mb"] = float(
                        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                    )
                print(
                    "V3_BC_PROGRESS=" + json.dumps(progress, sort_keys=True),
                    flush=True,
                )
        if stop:
            break
    if metric_count <= 0 and not recovery_losses:
        raise RuntimeError("v3 BC epoch executed no optimizer steps")
    summary = (
        {
            key: float((value / float(metric_count)).detach().cpu())
            for key, value in metric_sums.items()
        }
        if metric_count > 0
        else {}
    )
    elapsed = max(time.perf_counter() - epoch_started, 1e-9)
    temporal_delta = int(recurrent_stats["temporal_steps"]) - epoch_start_temporal
    summary["teacher_mix_probability"] = float(teacher_mix_probability)
    summary["amp_enabled"] = bool(amp_enabled)
    summary["epoch_elapsed_seconds"] = float(elapsed)
    summary["temporal_steps_per_sec"] = float(temporal_delta) / elapsed
    epoch_expert_updates = (
        int(recurrent_stats["expert_updates"]) - epoch_start_expert_updates
    )
    epoch_recovery_updates = (
        int(recurrent_stats["recovery_updates"]) - epoch_start_recovery_updates
    )
    summary["expert_updates"] = int(epoch_expert_updates)
    summary["recovery_updates"] = int(epoch_recovery_updates)
    summary["optimizer_steps_per_sec"] = float(
        int(train_steps) - epoch_start_steps
    ) / elapsed
    summary["expert_updates_per_sec"] = float(
        epoch_expert_updates
    ) / elapsed
    if recovery_losses:
        summary["recovery_action"] = float(sum(recovery_losses) / len(recovery_losses))
    return summary, train_steps, recovery_cursor, recovery_states

def _validation_chunks(dataset, config):
    iterator = dataset.iter_chunks(config.sequence_len)
    if config.max_val_chunks is None:
        chunks = list(iterator)
    else:
        chunks = list(islice(iterator, int(config.max_val_chunks)))
    if not chunks:
        raise RuntimeError("validation split produced no chunks")
    grouped = {}
    for chunk in chunks:
        grouped.setdefault((chunk.episode_id, chunk.seat), []).append(chunk)
    return [grouped[key] for key in sorted(grouped)]

def _attention_accumulator():
    return {
        "entropy_sum": 0.0, "mean_age_sum": 0.0,
        "count": 0, "mass": {1: 0.0, 4: 0.0, 8: 0.0, 16: 0.0, 32: 0.0},
    }


def _add_attention(acc, output):
    diag = output.temporal_diagnostics
    entropy = diag.attention_entropy.detach().cpu().numpy()
    mean_age = diag.mean_attended_age.detach().cpu().numpy()
    weights = diag.attention_weights.detach().cpu().numpy()
    lengths = output.temporal_state.valid_length.detach().cpu().numpy()
    for row in range(weights.shape[0]):
        length = int(lengths[row])
        acc["entropy_sum"] += float(entropy[row].mean())
        acc["mean_age_sum"] += float(mean_age[row].mean())
        acc["count"] += 1
        for width in acc["mass"]:
            start = max(0, length - width)
            acc["mass"][width] += float(
                weights[row, :, start:length].sum(axis=-1).mean()
            )


def _finish_attention(acc):
    count = max(1, int(acc["count"]))
    result = {
        "mean_entropy": acc["entropy_sum"] / count,
        "mean_attended_age": acc["mean_age_sum"] / count,
        "mass_recent": {
            str(width): value / count for width, value in acc["mass"].items()
        },
    }
    numeric = [
        result["mean_entropy"], result["mean_attended_age"],
        *result["mass_recent"].values(),
    ]
    result["finite"] = bool(all(np.isfinite(value) for value in numeric))
    return result

def _history_validation(
    model, episodes, mode, seed, device,
    strategy_manifest: StrategyManifest | None = None,
):
    if mode not in {"expert_history", "free_history"}:
        raise ValueError("unknown history validation mode")
    predictions, targets = [], []
    predicted_hist = {
        "farmer": Counter(), "hands": Counter(), "market": Counter(),
    }
    target_hist = {
        "farmer": Counter(), "hands": Counter(), "market": Counter(),
    }
    attention = _attention_accumulator()
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        for episode_chunks in episodes:
            state = None
            generated_previous = {}
            for chunk in episode_chunks:
                for source_row in chunk.rows:
                    row = dict(source_row)
                    if mode == "free_history":
                        row["previous_action"] = generated_previous
                    one = SequenceChunk(
                        chunk.episode_id, chunk.seat, (row,), False, False,
                    )
                    batch = collate_v2_sequences([one], 1)
                    move_step_batch(batch.flat, device)
                    strategy_slots = (
                        _strategy_slots_for_rows([row], strategy_manifest, device)
                        if strategy_manifest is not None else None
                    )
                    output = model.sample_step(
                        batch.flat, state, rng, deterministic=True,
                        strategy_slots=strategy_slots,
                    )
                    state = model.core.detach_state(output.temporal_state)
                    predicted = _canonical(output.rows[0])
                    target = batch.flat.canonical_actions[0]
                    predictions.append(predicted)
                    targets.append(target)
                    _update_histograms(predicted_hist, predicted)
                    _update_histograms(target_hist, target)
                    _add_attention(attention, output)
                    if mode == "free_history":
                        generated_previous = predicted
    semantic = semantic_domain_metrics(predictions, targets, masks=None)
    return {
        "state_source": "expert_replay",
        "previous_action_source": (
            "model_generated" if mode == "free_history" else "expert_replay"
        ),
        "is_true_closed_loop": False,
        "semantic": semantic,
        "op_histograms": _plain_histograms(predicted_hist),
        "target_op_histograms": _plain_histograms(target_hist),
        "attention": _finish_attention(attention),
    }

def _initial_state_validation(
    model, dataset, device,
    strategy_manifest: StrategyManifest | None = None,
    seed: int = 0,
):
    rows = []
    seen_teams = set()
    for episode in dataset:
        for row in episode.rows:
            if int(row.get("step", -1)) != 0:
                continue
            team_id = int(row.get("team_id", -1))
            if team_id in seen_teams:
                continue
            seen_teams.add(team_id)
            rows.append(row)
            break
    if not rows:
        return {
            "rows": 0,
            "target_continue_count": 0,
            "predicted_continue_count": 0,
            "market_continue_accuracy": 0.0,
            "market_active_op_accuracy": 0.0,
            "active_target_count": 0,
            "per_team": [],
        }

    rng = np.random.default_rng(int(seed))
    correct = 0
    active_correct = 0
    active_total = 0
    target_continue_count = 0
    predicted_continue_count = 0
    per_team = []
    with torch.no_grad():
        for row in rows:
            batch = collate_transitions([row])
            move_step_batch(batch, device)
            strategy_slots = (
                _strategy_slots_for_rows([row], strategy_manifest, device)
                if strategy_manifest is not None else None
            )
            output = model.sample_step(
                batch, None, rng, deterministic=True,
                strategy_slots=strategy_slots,
            )
            predicted = _canonical(output.rows[0])
            target = batch.canonical_actions[0]

            predicted_market = list(predicted.get("market") or [])
            target_market = list(target.get("market") or [])
            predicted_first = (
                predicted_market[0] if predicted_market
                else {"kind": "STOP_QUEUE", "op": None}
            )
            target_first = (
                target_market[0] if target_market
                else {"kind": "STOP_QUEUE", "op": None}
            )
            predicted_op = _market_op(predicted_first)
            target_op = _market_op(target_first)
            predicted_continue = predicted_op != "STOP_QUEUE"
            target_continue = target_op != "STOP_QUEUE"
            predicted_continue_count += int(predicted_continue)
            target_continue_count += int(target_continue)
            correct += int(predicted_continue == target_continue)
            if target_continue:
                active_total += 1
                active_correct += int(predicted_op == target_op)
            per_team.append({
                "team_id": int(row.get("team_id", -1)),
                "predicted_continue": bool(predicted_continue),
                "target_continue": bool(target_continue),
                "predicted_op": predicted_op,
                "target_op": target_op,
                "active_op_correct": bool(
                    target_continue and predicted_op == target_op
                ),
            })

    return {
        "rows": len(rows),
        "target_continue_count": target_continue_count,
        "predicted_continue_count": predicted_continue_count,
        "market_continue_accuracy": float(correct) / float(len(rows)),
        "market_active_op_accuracy": (
            float(active_correct) / float(active_total)
            if active_total else 1.0
        ),
        "active_target_count": int(active_total),
        "per_team": per_team,
    }


def _teacher_loss_validation(
    model, episodes, batch_sequences, device,
    strategy_manifest: StrategyManifest | None = None,
    market_active_op_weights: dict[str, float] | None = None,
):
    metrics = []
    with torch.no_grad():
        for start in range(0, len(episodes), batch_sequences):
            group = episodes[start:start + batch_sequences]
            states = [None] * len(group)
            max_chunks = max(len(chunks) for chunks in group)
            for chunk_index in range(max_chunks):
                active = [
                    (slot, chunks[chunk_index])
                    for slot, chunks in enumerate(group)
                    if chunk_index < len(chunks)
                ]
                losses, states = _teacher_chunk_cached(
                    model, active, states, device, recurrent_stats=None,
                    strategy_manifest=strategy_manifest,
                    market_active_op_weights=market_active_op_weights,
                )
                states = _detach_states(model, states)
                metrics.append(_float_dict(losses))
    return _mean_float_dict(metrics)


def _selection_score(
    semantic, model_architecture: str = ARCHITECTURE_VERSION,
):
    weights = (
        V32_SELECTION_WEIGHTS
        if model_architecture in {
            V32_ARCHITECTURE_VERSION,
            V33_ARCHITECTURE_VERSION,
        }
        else SELECTION_WEIGHTS
    )
    return float(sum(
        weights[key] * semantic[key]
        for key in weights
    ))


def _history_gaps(expert, free):
    left, right = expert["semantic"], free["semantic"]
    return {
        "farmer_op": float(left["farmer_op_accuracy"] - right["farmer_op_accuracy"]),
        "hands_op": float(left["hands_op_accuracy"] - right["hands_op_accuracy"]),
        "market_op": float(left["market_op_accuracy"] - right["market_op_accuracy"]),
    }


def _fraction(histogram, op):
    total = sum(int(value) for value in histogram.values())
    return float(histogram.get(op, 0)) / max(total, 1)

def _collapse_report(config, expert, free, gaps, initial_state=None):
    hist = free["op_histograms"]
    targets = free["target_op_histograms"]
    farmer_pass = _fraction(hist["farmer"], "PASS")
    hand_pass = _fraction(hist["hands"], "PASS")
    stop_queue = _fraction(hist["market"], "STOP_QUEUE")
    expert_stop = _fraction(targets["market"], "STOP_QUEUE")
    failures = []
    if gaps["farmer_op"] > config.max_farmer_op_gap:
        failures.append("farmer_history_gap")
    if gaps["hands_op"] > config.max_hand_op_gap:
        failures.append("hands_history_gap")
    if gaps["market_op"] > config.max_market_op_gap:
        failures.append("market_history_gap")
    if farmer_pass > config.max_farmer_pass_fraction:
        failures.append("farmer_pass_collapse")
    if hand_pass > config.max_hand_pass_fraction:
        failures.append("hand_pass_collapse")
    if (
        stop_queue > config.max_stop_queue_fraction
        and stop_queue - expert_stop > config.stop_queue_expert_gap
    ):
        failures.append("stop_queue_collapse")
    if config.model_architecture in {
        V32_ARCHITECTURE_VERSION,
        V33_ARCHITECTURE_VERSION,
    }:
        semantic = free.get("semantic") or {}
        market_hist = hist.get("market") or {}
        buy_predictions = sum(
            int(market_hist.get(op, 0) or 0)
            for op in ("BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL")
        )
        sell_predictions = int(market_hist.get("SELL", 0) or 0)
        stop_predictions = int(market_hist.get("STOP_QUEUE", 0) or 0)
        nop_predictions = int(market_hist.get("NOP_SLOT", 0) or 0)
        stop_targets = int((targets.get("market") or {}).get("STOP_QUEUE", 0) or 0)
        nop_targets = int((targets.get("market") or {}).get("NOP_SLOT", 0) or 0)
        buy_targets = int(semantic.get("market_buy_target_count", 0) or 0)
        buy_animal_targets = int(
            semantic.get("market_buy_animal_target_count", 0) or 0
        )
        sell_targets = int(semantic.get("market_sell_target_count", 0) or 0)
        if buy_targets > 0 and buy_predictions == 0:
            failures.append("zero_buy_prediction")
        if (
            buy_animal_targets > 0
            and int(market_hist.get("BUY_ANIMAL", 0) or 0) == 0
        ):
            failures.append("zero_buy_animal_prediction")
        if sell_targets > 0 and sell_predictions == 0:
            failures.append("zero_sell_prediction")
        if stop_targets > 0 and stop_predictions == 0:
            failures.append("zero_stop_prediction")
        if nop_targets > 0 and nop_predictions == 0:
            failures.append("zero_nop_prediction")
        if buy_targets > 0 and float(semantic.get("market_buy_op_recall", 0.0) or 0.0) <= 0.0:
            failures.append("zero_buy_recall")
        if (
            buy_animal_targets > 0
            and float(semantic.get("market_buy_animal_op_recall", 0.0) or 0.0) <= 0.0
        ):
            failures.append("zero_buy_animal_recall")
        if sell_targets > 0 and float(semantic.get("market_sell_op_recall", 0.0) or 0.0) <= 0.0:
            failures.append("zero_sell_recall")
        initial_state = initial_state or {}
        if int(initial_state.get("rows", 0) or 0) > 0:
            initial_accuracy = float(
                initial_state.get("market_continue_accuracy", 0.0) or 0.0
            )
            if initial_accuracy < float(config.min_initial_market_continue_accuracy):
                failures.append("initial_market_stop_collapse")
            initial_active_accuracy = float(
                initial_state.get("market_active_op_accuracy", 0.0) or 0.0
            )
            if (
                int(initial_state.get("active_target_count", 0) or 0) > 0
                and initial_active_accuracy
                < float(config.min_initial_market_active_accuracy)
            ):
                failures.append("initial_market_active_collapse")
    if not expert["attention"]["finite"] or not free["attention"]["finite"]:
        failures.append("non_finite_attention")
    return {
        "passed": not failures,
        "failures": failures,
        "farmer_pass_fraction": farmer_pass,
        "hand_pass_fraction": hand_pass,
        "stop_queue_fraction": stop_queue,
        "expert_stop_queue_fraction": expert_stop,
        "initial_market_continue_accuracy": (
            None if not initial_state
            else float(initial_state.get("market_continue_accuracy", 0.0) or 0.0)
        ),
        "initial_market_active_accuracy": (
            None if not initial_state
            else float(initial_state.get("market_active_op_accuracy", 0.0) or 0.0)
        ),
    }


def _initial_state_collapse_report(config, initial_state):
    initial_state = initial_state or {}
    failures = []
    if int(initial_state.get("rows", 0) or 0) > 0:
        if float(initial_state.get("market_continue_accuracy", 0.0) or 0.0) < float(
            config.min_initial_market_continue_accuracy
        ):
            failures.append("initial_market_stop_collapse")
        if (
            int(initial_state.get("active_target_count", 0) or 0) > 0
            and float(initial_state.get("market_active_op_accuracy", 0.0) or 0.0)
            < float(config.min_initial_market_active_accuracy)
        ):
            failures.append("initial_market_active_collapse")
    return {
        "scope": "initial_state_only",
        "passed": not failures,
        "failures": failures,
        "initial_market_continue_accuracy": initial_state.get(
            "market_continue_accuracy"
        ),
        "initial_market_active_accuracy": initial_state.get(
            "market_active_op_accuracy"
        ),
    }


def _checkpoint_payload(
    model, optimizer, config, dataset_sha, init_sha, epoch,
    train_steps, train_metrics, validation, recurrent_stats, device,
    recovery_dataset_sha=None, strategy_manifest: StrategyManifest | None = None,
    family_counts: dict[str, int] | None = None,
    family_weights: dict[str, float] | None = None,
    market_continue_counts: dict[str, int] | None = None,
    market_active_op_counts: dict[str, int] | None = None,
    market_active_op_weights: dict[str, float] | None = None,
    migration_new_parameter_keys: tuple[str, ...] = (),
    effective_action_info: dict[str, Any] | None = None,
):
    strategy_payload = None
    if strategy_manifest is not None:
        strategy_payload = {
            "slot_to_team": list(strategy_manifest.slot_to_team),
            "sha256": strategy_manifest.sha256,
        }
    return {
        "format_version": FORMAT_VERSION,
        "architecture_version": (
            V33_ARCHITECTURE_VERSION
            if config.model_architecture == V33_ARCHITECTURE_VERSION
            else (
                V32_ARCHITECTURE_VERSION
                if config.model_architecture == V32_ARCHITECTURE_VERSION
                else (
                    STRATEGY_ARCHITECTURE_VERSION
                    if strategy_manifest is not None
                    else ARCHITECTURE_VERSION
                )
            )
        ),
        "migration_new_parameter_keys": list(migration_new_parameter_keys),
        "strategy_core_scale": (
            float(STRATEGY_CORE_SCALE)
            if config.model_architecture in {
                V32_ARCHITECTURE_VERSION,
                V33_ARCHITECTURE_VERSION,
            }
            else None
        ),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "dataset_sha256": dataset_sha,
        "effective_action_sha256": (
            None if effective_action_info is None
            else effective_action_info.get("sha256")
        ),
        "effective_action_engine_version": (
            None if effective_action_info is None
            else effective_action_info.get("engine_module_version")
        ),
        "effective_action_engine_source_sha256": (
            None if effective_action_info is None
            else effective_action_info.get("engine_source_sha256")
        ),
        "effective_action_verified_non_eod_transitions": (
            None if effective_action_info is None
            else effective_action_info.get("verified_non_eod_transitions")
        ),
        "effective_action_verified_replay_files": (
            None if effective_action_info is None
            else effective_action_info.get("verified_replay_files")
        ),
        "effective_action_builder_code_sha256": (
            None if effective_action_info is None
            else effective_action_info.get("builder_code_sha256")
        ),
        "effective_action_corpus_manifest_sha256": (
            None if effective_action_info is None
            else effective_action_info.get("source_corpus_manifest_sha256")
        ),
        "recovery_dataset_sha256": recovery_dataset_sha,
        "family_counts": None if family_counts is None else dict(family_counts),
        "family_weights": None if family_weights is None else dict(family_weights),
        "market_continue_counts": (
            None if market_continue_counts is None
            else dict(market_continue_counts)
        ),
        "market_active_op_counts": (
            None if market_active_op_counts is None
            else dict(market_active_op_counts)
        ),
        "market_active_op_weights": (
            None if market_active_op_weights is None
            else dict(market_active_op_weights)
        ),
        "strategy_manifest": strategy_payload,
        "strategy_manifest_sha256": (
            None if strategy_manifest is None else strategy_manifest.sha256
        ),
        "init_checkpoint_sha256": init_sha,
        "config": _config_dict(config),
        "epoch": int(epoch),
        "train_steps": int(train_steps),
        "last_train_metrics": dict(train_metrics),
        "validation_metrics": validation,
        "recurrent_stats": dict(recurrent_stats),
        "resolved_device": str(device),
    }


def _write_manifest(output_dir: Path, paths: list[Path]):
    lines = [
        f"{_sha256(path)}  {path.name}"
        for path in paths if path.is_file()
    ]
    (output_dir / "manifest.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf-8",
    )


def _collect_online_dagger_round(
    model,
    config: BCV3Config,
    *,
    epoch: int,
    output_dir: Path,
    strategy_manifest: StrategyManifest,
    recovery_rows_all: list[dict[str, Any]],
):
    if (
        not bool(config.online_dagger)
        or int(epoch) < int(config.dagger_start_epoch)
        or int(epoch) >= int(config.epochs)
    ):
        return recovery_rows_all, None, None

    teacher_path = Path(config.dagger_teacher_path)
    if not teacher_path.is_file():
        raise FileNotFoundError(
            f"DAgger teacher not found: {teacher_path}"
        )
    slot = int(config.dagger_strategy_slot)
    if not 0 <= slot < strategy_manifest.size:
        raise ValueError(
            f"dagger_strategy_slot {slot} outside strategy manifest"
        )

    round_index = int(epoch) - int(config.dagger_start_epoch)
    seed_start = int(config.dagger_seed_base) + (
        round_index * int(config.dagger_seeds_per_round)
    )
    seeds = list(range(
        seed_start,
        seed_start + int(config.dagger_seeds_per_round),
    ))
    dagger_dir = output_dir / "dagger"
    dagger_dir.mkdir(parents=True, exist_ok=True)
    policy_path = dagger_dir / f"epoch_{epoch:03d}_policy.npz"
    round_path = dagger_dir / f"epoch_{epoch:03d}_v45.jsonl"

    export_v3_3_numpy(
        model,
        policy_path,
        default_strategy_slot=slot,
    )
    try:
        collect_v45_recovery(
            policy_path,
            teacher_path,
            round_path,
            seeds,
            episode_steps=int(config.dagger_episode_steps),
            strategy_slot=slot,
        )
    except RecoveryCollectionEmptyError as error:
        metadata = {
            "epoch": int(epoch),
            "seeds": seeds,
            "games": len(seeds) * 2,
            "new_rows": 0,
            "cumulative_rows": len(recovery_rows_all),
            "policy_path": str(policy_path),
            "round_path": None,
            "cumulative_path": None,
            "cumulative_sha256": None,
            "skipped": True,
            "reason": str(error),
            "error_report": (
                None
                if error.report_path is None
                else str(error.report_path)
            ),
        }
        print(
            "FARMOS_DAGGER_ROUND_SKIPPED="
            + json.dumps(metadata, sort_keys=True),
            flush=True,
        )
        return recovery_rows_all, None, metadata

    round_rows = read_recovery_rows(round_path)
    combined_rows = list(recovery_rows_all)
    combined_rows.extend(round_rows)

    cumulative_path = (
        dagger_dir / f"recovery_cumulative_epoch_{epoch:03d}.jsonl"
    )
    write_recovery_rows(combined_rows, cumulative_path)
    cumulative_sha = _sha256(cumulative_path)
    metadata = {
        "epoch": int(epoch),
        "seeds": seeds,
        "games": len(seeds) * 2,
        "new_rows": len(round_rows),
        "cumulative_rows": len(combined_rows),
        "policy_path": str(policy_path),
        "round_path": str(round_path),
        "cumulative_path": str(cumulative_path),
        "cumulative_sha256": cumulative_sha,
    }
    print(
        "FARMOS_DAGGER_ROUND="
        + json.dumps(metadata, sort_keys=True),
        flush=True,
    )
    return combined_rows, cumulative_path, metadata


def run_v3_bc(
    config: BCV3Config,
    init_checkpoint: Path | None = None,
) -> Path:
    config.validate()
    output_dir = Path(config.output_dir)
    blocking_outputs = (
        output_dir / "bc_best.pt",
        output_dir / "bc_last.pt",
        output_dir / "history.jsonl",
        output_dir / "strategy_manifest.json",
    )
    if any(path.exists() for path in blocking_outputs):
        raise FileExistsError(
            f"v3 BC output exists: {output_dir}. "
            "Choose a fresh output directory; existing artifacts are preserved."
        )

    acceptance = verify_training_acceptance(
        config.stage0_marker, config.stage1_marker,
    )
    dataset_sha = _sha256(config.dataset_path)
    if dataset_sha != acceptance["dataset_sha256"]:
        raise RuntimeError("v3 BC dataset SHA does not match accepted corpus")

    scratch_init = config.init_mode == "scratch"
    if scratch_init:
        if init_checkpoint is not None:
            raise ValueError(
                "scratch training must not receive an init checkpoint"
            )
        initial: dict[str, Any] = {}
        init_architecture = "scratch"
    else:
        if init_checkpoint is None:
            raise ValueError("checkpoint init requires init_checkpoint")
        init_checkpoint = Path(init_checkpoint)
        initial = torch.load(
            init_checkpoint, map_location="cpu", weights_only=False,
        )
        init_architecture = initial.get("architecture_version")
        if config.model_architecture == V33_ARCHITECTURE_VERSION:
            allowed_init = {
                V32_ARCHITECTURE_VERSION,
                V33_ARCHITECTURE_VERSION,
            }
        elif config.model_architecture == V32_ARCHITECTURE_VERSION:
            allowed_init = {
                ARCHITECTURE_VERSION,
                STRATEGY_ARCHITECTURE_VERSION,
                V32_ARCHITECTURE_VERSION,
            }
        else:
            allowed_init = (
                {ARCHITECTURE_VERSION, STRATEGY_ARCHITECTURE_VERSION}
                if config.strategy_conditioning else {ARCHITECTURE_VERSION}
            )
        if init_architecture not in allowed_init:
            raise RuntimeError("v3 BC initialization architecture mismatch")
        if initial.get("dataset_sha256") not in {None, dataset_sha}:
            raise RuntimeError("v3 BC initialization dataset SHA mismatch")
    torch.set_num_threads(config.torch_num_threads)
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    device = resolve_training_device(config.device)
    effective_action_info = None
    effective_action_path = None
    if config.model_architecture in {
        V32_ARCHITECTURE_VERSION,
        V33_ARCHITECTURE_VERSION,
    }:
        effective_action_info = verify_effective_action_sidecar(
            config.dataset_path,
        )
        effective_action_path = Path(effective_action_info["path"])
        if initial.get("effective_action_sha256") not in {
            None, effective_action_info["sha256"],
        }:
            raise RuntimeError(
                "v3.2 initialization effective-action SHA mismatch"
            )
    train_data = V2EpisodeDataset(
        config.dataset_path, "train", {"active_best"},
        effective_action_path=effective_action_path,
        require_effective_actions=(
            config.model_architecture in {
                V32_ARCHITECTURE_VERSION,
                V33_ARCHITECTURE_VERSION,
            }
        ),
    )
    val_data = V2EpisodeDataset(
        config.dataset_path, "val", {"active_best"},
        effective_action_path=effective_action_path,
        require_effective_actions=(
            config.model_architecture in {
                V32_ARCHITECTURE_VERSION,
                V33_ARCHITECTURE_VERSION,
            }
        ),
    )
    market_continue_counts = None
    market_active_op_counts = None
    market_active_op_weights = None
    family_counts = None
    family_weights = None
    if config.family_weight_cap is not None:
        family_counts = _training_family_counts(train_data)
        family_weights = compute_domain_family_weights(
            family_counts, float(config.family_weight_cap),
        )
    if config.model_architecture in {
        V32_ARCHITECTURE_VERSION,
        V33_ARCHITECTURE_VERSION,
    }:
        market_continue_counts = _training_market_continue_counts(train_data)
        market_active_op_counts = _training_market_active_counts(train_data)
        market_active_op_weights = _balanced_market_active_op_weights(
            market_active_op_counts,
            cap=float(config.market_active_op_weight_cap),
            power=float(config.market_active_op_weight_power),
        )
    strategy_manifest = None
    if config.strategy_conditioning:
        strategy_manifest = _build_training_strategy_manifest(train_data)
        _validate_strategy_coverage(strategy_manifest, val_data)
    migration_new_parameter_keys: tuple[str, ...] = ()
    if config.model_architecture == V33_ARCHITECTURE_VERSION:
        if strategy_manifest is None:
            raise RuntimeError("V3.3 requires a strategy manifest")
        model = TemporalIntentPolicyV33(
            strategy_count=strategy_manifest.size,
        )
        if scratch_init:
            # True scratch run: keep constructor initialization and do not
            # import any weights from V3.2/V3.3 checkpoints.
            migration_new_parameter_keys = ()
        else:
            if (
                initial.get("strategy_manifest_sha256")
                != strategy_manifest.sha256
            ):
                raise RuntimeError(
                    "strategy manifest mismatch in V3.3 initialization checkpoint"
                )
            if init_architecture == V33_ARCHITECTURE_VERSION:
                model.load_state_dict(initial["model_state"], strict=True)
            else:
                incompatible = model.load_state_dict(
                    initial["model_state"], strict=False,
                )
                expected_missing = {
                    "economic_continue_head.bias",
                    "economic_continue_head.weight",
                    "economic_active_head.bias",
                    "economic_active_head.weight",
                    "short_economic_head.bias",
                    "short_economic_head.weight",
                }
                if (
                    set(incompatible.missing_keys) != expected_missing
                    or incompatible.unexpected_keys
                ):
                    raise RuntimeError(
                        "unexpected state mismatch while migrating V3.2 to V3.3: "
                        f"missing={sorted(incompatible.missing_keys)} "
                        f"unexpected={sorted(incompatible.unexpected_keys)}"
                    )
                migration_new_parameter_keys = tuple(
                    sorted(incompatible.missing_keys)
                )
    elif config.model_architecture == V32_ARCHITECTURE_VERSION:
        if strategy_manifest is None:
            raise RuntimeError("V3.2 requires a strategy manifest")
        model = TemporalIntentPolicyV32(
            strategy_count=strategy_manifest.size,
        )
        if init_architecture == V32_ARCHITECTURE_VERSION:
            if initial.get("strategy_manifest_sha256") != strategy_manifest.sha256:
                raise RuntimeError("strategy manifest mismatch in V3.2 initialization checkpoint")
            incompatible = model.load_state_dict(
                initial["model_state"], strict=False,
            )
            allowed_missing = {
                "opening_strategy_head.bias",
                "opening_strategy_head.weight",
            }
            if (
                not set(incompatible.missing_keys).issubset(allowed_missing)
                or incompatible.unexpected_keys
            ):
                raise RuntimeError(
                    "unexpected state mismatch while loading V3.2 checkpoint: "
                    f"missing={sorted(incompatible.missing_keys)} "
                    f"unexpected={sorted(incompatible.unexpected_keys)}"
                )
            migration_new_parameter_keys = tuple(
                sorted(incompatible.missing_keys)
            )
        else:
            if (
                init_architecture == STRATEGY_ARCHITECTURE_VERSION
                and initial.get("strategy_manifest_sha256") != strategy_manifest.sha256
            ):
                raise RuntimeError("strategy manifest mismatch in V3.1 initialization checkpoint")
            incompatible = model.load_state_dict(
                initial["model_state"], strict=False,
            )
            expected_missing = {
                "market_active_op_head.bias",
                "market_active_op_head.weight",
                "market_continue_head.bias",
                "market_continue_head.weight",
                "opening_strategy_head.bias",
                "opening_strategy_head.weight",
            }
            if init_architecture == ARCHITECTURE_VERSION:
                expected_missing.add("strategy_embedding.weight")
            if (
                set(incompatible.missing_keys) != expected_missing
                or incompatible.unexpected_keys
            ):
                raise RuntimeError(
                    "unexpected state mismatch while migrating checkpoint to V3.2: "
                    f"missing={sorted(incompatible.missing_keys)} "
                    f"unexpected={sorted(incompatible.unexpected_keys)}"
                )
            migration_new_parameter_keys = tuple(
                sorted(incompatible.missing_keys)
            )
            _initialize_v32_migration(
                model, init_architecture, market_continue_counts,
            )
    else:
        model = TemporalIntentPolicy(
            strategy_count=(
                0 if strategy_manifest is None
                else strategy_manifest.size
            ),
        )
        if (
            strategy_manifest is None
            or init_architecture == STRATEGY_ARCHITECTURE_VERSION
        ):
            if strategy_manifest is not None:
                if (
                    initial.get("strategy_manifest_sha256")
                    != strategy_manifest.sha256
                ):
                    raise RuntimeError(
                        "strategy manifest mismatch in initialization checkpoint"
                    )
            model.load_state_dict(initial["model_state"], strict=True)
        else:
            incompatible = model.load_state_dict(
                initial["model_state"], strict=False,
            )
            if (
                set(incompatible.missing_keys)
                != {"strategy_embedding.weight"}
                or incompatible.unexpected_keys
            ):
                raise RuntimeError(
                    "unexpected state mismatch while migrating V3 checkpoint to strategy model"
                )
            migration_new_parameter_keys = tuple(
                sorted(incompatible.missing_keys)
            )
    model.to(device)
    optimizer, optimizer_fused = _build_adamw(model, config, device)
    amp_enabled = bool(
        config.use_amp
        and getattr(device, "type", str(device).split(":")[0]) == "cuda"
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
        init_scale=float(config.amp_init_scale),
        growth_interval=int(config.amp_growth_interval),
    )
    opening_rows = (
        _opening_training_rows(train_data)
        if (
            config.model_architecture in {
                V32_ARCHITECTURE_VERSION,
                V33_ARCHITECTURE_VERSION,
            }
            and int(config.opening_replay_steps) > 0
        )
        else []
    )
    val_episodes = _validation_chunks(val_data, config)
    recovery_dataset_sha = None
    recovery_chunks = []
    recovery_rows_all: list[dict[str, Any]] = []
    if config.recovery_dataset_path is not None:
        recovery_path = Path(config.recovery_dataset_path)
        recovery_dataset_sha = _sha256(recovery_path)
        recovery_rows = read_recovery_rows(recovery_path)
        recovery_rows_all.extend(recovery_rows)
        if strategy_manifest is not None:
            for row in recovery_rows:
                if row.get("strategy_slot") is None:
                    raise ValueError(
                        "strategy-conditioned recovery requires strategy_slot"
                    )
                slot = int(row["strategy_slot"])
                if not 0 <= slot < strategy_manifest.size:
                    raise ValueError(
                        f"recovery strategy_slot {slot} outside manifest"
                    )
        recovery_chunks = _recovery_chunks(
            recovery_rows, config.sequence_len,
        )

    prepared_training_groups, gpu_batch_cache_stats = (
        _prepare_gpu_batch_cache(
            model,
            train_data,
            config,
            device,
            strategy_manifest,
            dataset_sha=dataset_sha,
            effective_action_sha=(
                None
                if effective_action_info is None
                else str(effective_action_info.get("sha256"))
            ),
        )
    )
    cached_updates_per_epoch = None
    cached_max_active_sequences = None
    if prepared_training_groups is not None:
        cached_updates_per_epoch = sum(
            len(group.updates) for group in prepared_training_groups
        )
        cached_max_active_sequences = max(
            len(update.active)
            for group in prepared_training_groups
            for update in group.updates
        )
    train_plan = {
        "train_episodes": int(len(train_data)),
        "train_rows": int(sum(len(episode.rows) for episode in train_data)),
        "sequence_len": int(config.sequence_len),
        "configured_batch_sequences": int(config.batch_sequences),
        "expert_updates_per_epoch": cached_updates_per_epoch,
        "max_active_sequences": cached_max_active_sequences,
        "epochs": int(config.epochs),
        "teacher_mix_schedule": [
            float(value) for value in config.teacher_mix_schedule
        ],
        "init_mode": str(config.init_mode),
        "online_dagger": bool(config.online_dagger),
        "recovery_every": int(config.recovery_every),
        "optimizer_fused": bool(optimizer_fused),
        "market_continue_counts": market_continue_counts,
        "market_active_op_counts": market_active_op_counts,
        "market_active_op_weights": market_active_op_weights,
    }
    print(
        "FARMOS_TRAIN_PLAN=" + json.dumps(train_plan, sort_keys=True),
        flush=True,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "bc_best.pt"
    last_path = output_dir / "bc_last.pt"
    epoch2_path = output_dir / "bc_epoch2.pt"
    history_path = output_dir / "history.jsonl"
    strategy_manifest_path = output_dir / "strategy_manifest.json"
    if (best_path.exists() or last_path.exists() or history_path.exists()
            or strategy_manifest_path.exists()):
        raise FileExistsError(f"v3 BC output exists: {output_dir}")
    if strategy_manifest is not None:
        strategy_manifest_path.write_text(
            json.dumps({
                "slot_to_team": list(strategy_manifest.slot_to_team),
                "sha256": strategy_manifest.sha256,
            }, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    history_path.write_text("", encoding="utf-8")

    recurrent_stats = {
        "temporal_steps": 0,
        "state_resets": 0,
        "state_carries": 0,
        "expert_updates": 0,
        "recovery_updates": 0,
        "recovery_temporal_steps": 0,
        "amp_overflow_skips": 0,
        "nonfinite_loss_skips": 0,
        "optimizer_fused": bool(optimizer_fused),
        "scratch_init": bool(scratch_init),
        "dagger_rounds": 0,
        "dagger_rows": 0,
        "tensor_sequence_chunks": 0,
        "cached_tensor_sequence_chunks": 0,
        "gpu_batch_cache_enabled": bool(gpu_batch_cache_stats["enabled"]),
        "gpu_batch_cache_updates": int(gpu_batch_cache_stats["cached_updates"]),
        "gpu_batch_cache_gpu_updates": int(
            gpu_batch_cache_stats["gpu_resident_updates"]
        ),
        "gpu_batch_cache_cpu_updates": int(
            gpu_batch_cache_stats["cpu_resident_updates"]
        ),
        "gpu_batch_cache_fallback_updates": int(
            gpu_batch_cache_stats["fallback_updates"]
        ),
        "gpu_batch_cache_gpu_bytes": int(
            gpu_batch_cache_stats["gpu_tensor_bytes"]
        ),
        "gpu_batch_cache_cpu_bytes": int(
            gpu_batch_cache_stats["cpu_tensor_bytes"]
        ),
    }
    recovery_cursor = 0
    recovery_states = {}
    best_score = float("-inf")
    best_teacher_loss = float("inf")
    best_epoch = 0
    train_steps = 0
    init_sha = None if scratch_init else _sha256(Path(init_checkpoint))

    for epoch in range(1, config.epochs + 1):
        train_metrics, train_steps, recovery_cursor, recovery_states = _train_epoch(
            model, optimizer, train_data, config, epoch,
            train_steps, recurrent_stats, device,
            recovery_chunks=recovery_chunks,
            recovery_cursor=recovery_cursor,
            recovery_states=recovery_states,
            strategy_manifest=strategy_manifest,
            family_weights=family_weights,
            market_active_op_weights=market_active_op_weights,
            scaler=scaler,
            prepared_training_groups=prepared_training_groups,
        )
        if opening_rows:
            opening_metrics = _opening_strategy_replay(
                model,
                opening_rows,
                strategy_manifest,
                device,
                steps=int(config.opening_replay_steps),
                learning_rate=float(config.opening_replay_learning_rate),
            )
            train_metrics = {**train_metrics, **opening_metrics}
        if (
            bool(config.clear_cuda_cache)
            and getattr(device, "type", str(device).split(":")[0]) == "cuda"
            and torch.cuda.is_available()
        ):
            torch.cuda.empty_cache()
        model.eval()
        teacher_loss = _teacher_loss_validation(
            model, val_episodes, config.batch_sequences, device,
            strategy_manifest=strategy_manifest,
            market_active_op_weights=market_active_op_weights,
        )
        initial_state = (
            _initial_state_validation(
                model, val_data, device,
                strategy_manifest=strategy_manifest,
                seed=config.seed + 3000 * epoch,
            )
            if config.model_architecture in {
                V32_ARCHITECTURE_VERSION,
                V33_ARCHITECTURE_VERSION,
            }
            else None
        )
        if config.validation_profile == "full":
            expert = _history_validation(
                model, val_episodes, "expert_history",
                config.seed + 1000 * epoch, device,
                strategy_manifest=strategy_manifest,
            )
            free = _history_validation(
                model, val_episodes, "free_history",
                config.seed + 2000 * epoch, device,
                strategy_manifest=strategy_manifest,
            )
            gaps = _history_gaps(expert, free)
            collapse = _collapse_report(
                config, expert, free, gaps, initial_state=initial_state,
            )
            selection_score = _selection_score(
                free["semantic"],
                model_architecture=config.model_architecture,
            )
        else:
            expert = None
            free = None
            gaps = None
            collapse = _initial_state_collapse_report(config, initial_state)
            selection_score = None
        validation = {
            "validation_profile": str(config.validation_profile),
            "selection_mode": str(config.selection_mode),
            "teacher_forced_loss": teacher_loss,
            "initial_state": initial_state,
            "expert_history": expert,
            "free_history": free,
            "history_gaps": gaps,
            "collapse": collapse,
            "selection_score": selection_score,
            "pseudo_closed_loop_used_for_selection": bool(
                config.selection_mode == "offline_legacy"
            ),
        }
        promotion_eligible = (
            not bool(config.require_collapse_for_best)
            or bool(collapse.get("passed", False))
        )
        validation["promotion_eligible"] = bool(promotion_eligible)
        validation["promotion_failures"] = (
            [] if promotion_eligible else list(collapse.get("failures") or [])
        )
        payload = _checkpoint_payload(
            model, optimizer, config, dataset_sha, init_sha, epoch,
            train_steps, train_metrics, validation, recurrent_stats, device,
            recovery_dataset_sha=recovery_dataset_sha,
            strategy_manifest=strategy_manifest,
            family_counts=family_counts,
            family_weights=family_weights,
            market_continue_counts=market_continue_counts,
            market_active_op_counts=market_active_op_counts,
            market_active_op_weights=market_active_op_weights,
            migration_new_parameter_keys=migration_new_parameter_keys,
            effective_action_info=effective_action_info,
        )
        torch.save(payload, last_path)
        if epoch == 2:
            torch.save(payload, epoch2_path)
        if config.selection_mode == "last_epoch":
            score = float(epoch)
        else:
            score = float(validation["selection_score"])
        aux = float(teacher_loss["total"])
        if (
            promotion_eligible
            and (
                score > best_score
                or (score == best_score and aux < best_teacher_loss)
            )
        ):
            best_score = score
            best_teacher_loss = aux
            best_epoch = epoch
            torch.save(payload, best_path)

        dagger_metadata = None
        if config.online_dagger:
            if strategy_manifest is None:
                raise RuntimeError("online DAgger requires strategy manifest")
            recovery_rows_all, cumulative_path, dagger_metadata = (
                _collect_online_dagger_round(
                    model,
                    config,
                    epoch=epoch,
                    output_dir=output_dir,
                    strategy_manifest=strategy_manifest,
                    recovery_rows_all=recovery_rows_all,
                )
            )
            if cumulative_path is not None:
                recovery_dataset_sha = _sha256(cumulative_path)
                recovery_chunks = _recovery_chunks(
                    recovery_rows_all, config.sequence_len,
                )
                recovery_cursor = 0
                recovery_states = {}
                recurrent_stats["dagger_rounds"] += 1
                recurrent_stats["dagger_rows"] = len(recovery_rows_all)

        history_row = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation,
            "dagger": dagger_metadata,
            "best_epoch": int(best_epoch),
            "best_selection_score": (
                None
                if best_epoch == 0 or config.selection_mode == "last_epoch"
                else best_score
            ),
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(history_row, sort_keys=True) + "\n")
        if config.max_train_steps is not None and train_steps >= config.max_train_steps:
            break
        if epoch - best_epoch >= config.early_stop_patience:
            break

    if not last_path.is_file():
        raise RuntimeError("v3 BC did not produce bc_last.pt")
    if not best_path.is_file():
        if bool(config.require_collapse_for_best):
            raise RuntimeError(
                "no checkpoint passed the configured collapse promotion gate"
            )
        raise RuntimeError("v3 BC did not produce bc_best.pt")
    _write_manifest(
        output_dir,
        [best_path, last_path, epoch2_path, history_path, strategy_manifest_path],
    )
    return best_path
