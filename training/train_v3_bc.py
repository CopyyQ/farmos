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
from kaggrl.v3_strategy import (
    StrategyManifest, build_strategy_manifest, strategy_slot_for_team,
)
from kaggrl.v3_tensor_decoder import teacher_step_tensor_mixed
from kaggrl.v3_tensor_ledger import TensorLedger
from kaggrl.v3_tensor_losses import tensor_total_pretrain_loss
from kaggrl.v3_tensor_targets import TensorActionTargets
from training.build_v3_recovery_dataset import read_recovery_rows
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

    def validate(self) -> None:
        if self.sequence_len <= 0 or self.batch_sequences <= 0:
            raise ValueError("sequence_len and batch_sequences must be positive")
        if self.epochs <= 0 or self.learning_rate <= 0:
            raise ValueError("invalid BC optimization configuration")
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
        if self.model_architecture not in {ARCHITECTURE_VERSION, V32_ARCHITECTURE_VERSION}:
            raise ValueError("unsupported V3 model_architecture")
        if self.gpu_tensor_training and self.model_architecture != V32_ARCHITECTURE_VERSION:
            raise ValueError("gpu_tensor_training currently requires V3.2")
        if self.model_architecture == V32_ARCHITECTURE_VERSION:
            if not self.strategy_conditioning:
                raise ValueError("V3.2 requires strategy_conditioning")
            if self.recovery_dataset_path is not None:
                raise ValueError("V3.2 R0 does not allow recovery supervision")
        if self.recovery_dataset_path is not None and self.recovery_every < 3:
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


def _config_dict(config: BCV3Config) -> dict[str, Any]:
    value = asdict(config)
    for key in ("dataset_path", "stage0_marker", "stage1_marker", "output_dir"):
        value[key] = str(value[key])
    if value.get("recovery_dataset_path") is not None:
        value["recovery_dataset_path"] = str(value["recovery_dataset_path"])
    value["selection_weights"] = dict(
        V32_SELECTION_WEIGHTS
        if config.model_architecture == V32_ARCHITECTURE_VERSION
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
    return torch.tensor(
        [strategy_slot_for_team(int(row["team_id"]), manifest) for row in rows],
        dtype=torch.long, device=device,
    )


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

def _teacher_chunk_cached(
    model, active_chunks, states, device, recurrent_stats=None,
    strategy_manifest: StrategyManifest | None = None,
    family_weights: dict[str, float] | None = None,
    market_active_op_weights: dict[str, float] | None = None,
    teacher_mix_probability: float = 1.0,
    conditioning_rng=None,
    gpu_tensor_training: bool = False,
):
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
        tensor_ledger = TensorLedger.from_states(
            sequence.flat.structured_states,
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
):
    domain = action_loss(
        output, targets, family_weights=family_weights,
    )
    kind = str(supervision_kind)
    if kind == "smoke_only":
        return domain.market
    if kind in {"expert", "accepted_policy"}:
        return domain.total
    raise ValueError(f"unknown recovery supervision_kind: {kind}")


def _recovery_update(
    model, optimizer, chunk, state, config, device, family_weights=None,
):
    losses = []
    for row in chunk.rows:
        batch = collate_transitions([row])
        move_step_batch(batch, device)
        output = model.teacher_step(batch, batch.canonical_actions, state)
        losses.append(_recovery_step_loss(
            output,
            batch.canonical_actions,
            supervision_kind=row.get("supervision_kind", ""),
            family_weights=family_weights,
        ))
        state = output.temporal_state
    total = torch.stack(losses).mean()
    if not torch.isfinite(total):
        raise RuntimeError("non-finite v3 recovery action loss")
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
    if not torch.isfinite(torch.as_tensor(norm)):
        raise RuntimeError("non-finite v3 recovery gradient norm")
    optimizer.step()
    return float(total.detach().cpu().item()), model.core.detach_state(state)


def _finish_optimizer_step(
    model, optimizer, config, *, amp_enabled: bool, scaler=None,
):
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), config.gradient_clip,
    )
    norm_tensor = torch.as_tensor(norm)
    finite = bool(torch.isfinite(norm_tensor).item())
    if amp_enabled:
        if scaler is None:
            raise RuntimeError("AMP training requires a GradScaler")
        scale_before = float(scaler.get_scale())
        # GradScaler.step() inspects the overflow state captured by unscale_().
        # When overflow is present it intentionally skips optimizer.step().
        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale())
        return float(norm_tensor.detach().cpu()), finite, scale_before, scale_after
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
):
    model.train()
    teacher_mix_probability = _teacher_mix_for_epoch(
        config.teacher_mix_schedule, epoch,
    )
    conditioning_rng = np.random.default_rng(
        int(config.seed) + 3000001 * int(epoch),
    )
    metrics = []
    recovery_losses = []
    stop = False
    epoch_start_steps = int(train_steps)
    epoch_start_temporal = int(recurrent_stats["temporal_steps"])
    epoch_started = time.perf_counter()
    amp_device_type = getattr(device, "type", str(device).split(":")[0])
    amp_enabled = bool(config.use_amp and amp_device_type == "cuda")
    prepared_groups = []
    for episodes in _episode_groups(
        dataset, config.batch_sequences, config.seed, epoch,
    ):
        prepared_groups.append([
            _episode_chunks(ep, config.sequence_len) for ep in episodes
        ])
    epoch_expected_updates = sum(
        max(len(chunks) for chunks in group) for group in prepared_groups
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
        loss, state = _recovery_update(
            model, optimizer, chunk, state, config, device,
            family_weights=family_weights,
        )
        recovery_states[key] = None if chunk.episode_end else state
        recurrent_stats["recovery_updates"] += 1
        recurrent_stats["recovery_temporal_steps"] += len(chunk.rows)
        train_steps += 1
        recovery_losses.append(float(loss))
        return True

    for chunks_by_slot in prepared_groups:
        states = [None] * len(chunks_by_slot)
        max_chunks = max(len(chunks) for chunks in chunks_by_slot)
        for chunk_index in range(max_chunks):
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
            active = [
                (slot, chunks[chunk_index]) for slot, chunks in enumerate(chunks_by_slot)
                if chunk_index < len(chunks)
            ]
            for slot, _ in active:
                recurrent_stats["state_resets" if states[slot] is None else "state_carries"] += 1
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=amp_device_type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                losses, states = _teacher_chunk_cached(
                    model, active, states, device, recurrent_stats,
                    strategy_manifest=strategy_manifest,
                    family_weights=family_weights,
                    market_active_op_weights=market_active_op_weights,
                    teacher_mix_probability=teacher_mix_probability,
                    conditioning_rng=conditioning_rng,
                    gpu_tensor_training=bool(config.gpu_tensor_training),
                )
            if not torch.isfinite(losses["total"]):
                raise RuntimeError("non-finite v3 BC loss")
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
            metric_row = _float_dict(losses)
            metrics.append(metric_row)
            recurrent_stats["expert_updates"] += 1
            train_steps += 1
            if (
                int(config.progress_every) > 0
                and train_steps % int(config.progress_every) == 0
            ):
                elapsed = max(time.perf_counter() - epoch_started, 1e-9)
                epoch_step = int(train_steps) - epoch_start_steps
                temporal_delta = int(recurrent_stats["temporal_steps"]) - epoch_start_temporal
                rate = float(temporal_delta) / elapsed
                eta = (
                    max(0, epoch_expected_updates - epoch_step)
                    * elapsed / max(epoch_step, 1)
                )
                progress = {
                    "epoch": int(epoch),
                    "epoch_step": int(epoch_step),
                    "epoch_steps": int(epoch_expected_updates),
                    "train_steps": int(train_steps),
                    "active_sequences": int(len(active)),
                    "temporal_steps": int(recurrent_stats["temporal_steps"]),
                    "temporal_steps_per_sec": float(rate),
                    "eta_seconds": float(eta),
                    "loss_total": float(metric_row["total"]),
                    "amp": bool(amp_enabled),
                    "amp_scale": (
                        float(scaler.get_scale()) if amp_enabled and scaler is not None
                        else None
                    ),
                    "amp_overflow_skips": int(recurrent_stats["amp_overflow_skips"]),
                    "gpu_tensor_training": bool(config.gpu_tensor_training),
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
    if not metrics and not recovery_losses:
        raise RuntimeError("v3 BC epoch executed no optimizer steps")
    summary = _mean_float_dict(metrics) if metrics else {}
    elapsed = max(time.perf_counter() - epoch_started, 1e-9)
    temporal_delta = int(recurrent_stats["temporal_steps"]) - epoch_start_temporal
    summary["teacher_mix_probability"] = float(teacher_mix_probability)
    summary["amp_enabled"] = bool(amp_enabled)
    summary["epoch_elapsed_seconds"] = float(elapsed)
    summary["temporal_steps_per_sec"] = float(temporal_delta) / elapsed
    summary["optimizer_steps_per_sec"] = float(
        int(train_steps) - epoch_start_steps
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
        if model_architecture == V32_ARCHITECTURE_VERSION
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
    if config.model_architecture == V32_ARCHITECTURE_VERSION:
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
            V32_ARCHITECTURE_VERSION
            if config.model_architecture == V32_ARCHITECTURE_VERSION
            else (
                STRATEGY_ARCHITECTURE_VERSION if strategy_manifest is not None
                else ARCHITECTURE_VERSION
            )
        ),
        "migration_new_parameter_keys": list(migration_new_parameter_keys),
        "strategy_core_scale": (
            float(STRATEGY_CORE_SCALE)
            if config.model_architecture == V32_ARCHITECTURE_VERSION
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


def run_v3_bc(config: BCV3Config, init_checkpoint: Path) -> Path:
    config.validate()
    acceptance = verify_training_acceptance(
        config.stage0_marker, config.stage1_marker,
    )
    dataset_sha = _sha256(config.dataset_path)
    if dataset_sha != acceptance["dataset_sha256"]:
        raise RuntimeError("v3 BC dataset SHA does not match accepted corpus")
    init_checkpoint = Path(init_checkpoint)
    initial = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
    init_architecture = initial.get("architecture_version")
    if config.model_architecture == V32_ARCHITECTURE_VERSION:
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
    if config.model_architecture == V32_ARCHITECTURE_VERSION:
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
            config.model_architecture == V32_ARCHITECTURE_VERSION
        ),
    )
    val_data = V2EpisodeDataset(
        config.dataset_path, "val", {"active_best"},
        effective_action_path=effective_action_path,
        require_effective_actions=(
            config.model_architecture == V32_ARCHITECTURE_VERSION
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
    if config.model_architecture == V32_ARCHITECTURE_VERSION:
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
        if config.recovery_dataset_path is not None:
            raise RuntimeError(
                "strategy-conditioned recovery requires strategy-tagged recovery rows"
            )
    migration_new_parameter_keys: tuple[str, ...] = ()
    if config.model_architecture == V32_ARCHITECTURE_VERSION:
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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
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
            config.model_architecture == V32_ARCHITECTURE_VERSION
            and int(config.opening_replay_steps) > 0
        )
        else []
    )
    val_episodes = _validation_chunks(val_data, config)
    recovery_dataset_sha = None
    recovery_chunks = []
    if config.recovery_dataset_path is not None:
        recovery_path = Path(config.recovery_dataset_path)
        recovery_dataset_sha = _sha256(recovery_path)
        recovery_chunks = _recovery_chunks(
            read_recovery_rows(recovery_path), config.sequence_len,
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
    }
    recovery_cursor = 0
    recovery_states = {}
    best_score = float("-inf")
    best_teacher_loss = float("inf")
    best_epoch = 0
    train_steps = 0
    init_sha = _sha256(init_checkpoint)

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
            if config.model_architecture == V32_ARCHITECTURE_VERSION
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
        history_row = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation,
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
