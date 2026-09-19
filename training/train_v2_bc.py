from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kaggrl.v2_device import move_step_batch, resolve_training_device
from kaggrl.v2_losses import total_pretrain_loss
from kaggrl.v2_metrics import semantic_domain_metrics
from kaggrl.v2_model import RecurrentIntentPolicy
from kaggrl.v2_training_data import (
    EpisodeSequence,
    SequenceChunk,
    V2EpisodeDataset,
    balanced_episode_order,
    collate_v2_sequences,
    verify_training_acceptance,
)

FORMAT_VERSION = 1
SELECTION_WEIGHTS = {
    "farmer_semantic_exact": 0.30,
    "mean_hand_semantic_exact": 0.30,
    "market_sequence_exact": 0.30,
    "full_joint_step_exact": 0.10,
}

@dataclass(frozen=True)
class BCConfig:
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
    epochs: int = 12
    torch_num_threads: int = 2
    device: str = "auto"
    max_train_steps: int | None = None
    max_val_chunks: int | None = None
    collapse_threshold: float = 0.05
    collapse_consecutive: int = 2
    max_freeze_epochs: int = 2
    early_stop_patience: int = 3

    def validate(self) -> None:
        if self.sequence_len <= 0 or self.batch_sequences <= 0:
            raise ValueError("sequence_len and batch_sequences must be positive")
        if self.epochs <= 0 or self.learning_rate <= 0 or self.gradient_clip <= 0:
            raise ValueError("invalid BC optimization configuration")
        if self.torch_num_threads < 1:
            raise ValueError("torch_num_threads must be positive")
        if self.collapse_consecutive < 1 or self.max_freeze_epochs < 0:
            raise ValueError("invalid collapse configuration")

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_collapse(history: list[dict[str, float]], threshold: float = .05,
                    consecutive: int = 2) -> set[str]:
    if len(history) < consecutive + 1:
        return set()
    window = history[-(consecutive + 1):]
    keys = {"farmer": "farmer_semantic_exact", "market": "market_sequence_exact"}
    other_keys = ("farmer_semantic_exact", "mean_hand_semantic_exact", "market_sequence_exact")
    collapsed = set()
    for domain, key in keys.items():
        drops = [window[i + 1][key] - window[i][key] for i in range(consecutive)]
        if not all(delta < -float(threshold) for delta in drops):
            continue
        improved = any(window[-1][other] > window[0][other]
                       for other in other_keys if other != key)
        if improved:
            collapsed.add(domain)
    return collapsed


def _set_decoder_freeze(model: RecurrentIntentPolicy, freeze_until: dict[str, int], epoch: int) -> dict[str, bool]:
    state = {
        "farmer": int(epoch) <= int(freeze_until.get("farmer", 0)),
        "market": int(epoch) <= int(freeze_until.get("market", 0)),
    }
    for parameter in model.unit_op_head.parameters():
        parameter.requires_grad_(not state["farmer"])
    for parameter in model.market_op_head.parameters():
        parameter.requires_grad_(not state["market"])
    return state


def _episode_chunks(episode: EpisodeSequence, length: int) -> list[SequenceChunk]:
    out = []
    for start in range(0, len(episode.rows), length):
        rows = episode.rows[start:start + length]
        out.append(SequenceChunk(episode.episode_id, episode.seat, tuple(rows),
                                 start == 0, start + len(rows) == len(episode.rows)))
    return out

def _pack_states(states, slots):
    chosen = [states[slot] for slot in slots]
    if all(value is None for value in chosen):
        return None
    if any(value is None for value in chosen):
        raise RuntimeError("mixed initialized/uninitialized recurrent state")
    h = torch.stack([value[0] for value in chosen], dim=0)
    c = torch.stack([value[1] for value in chosen], dim=0)
    return h, c


def _detach_states(states):
    return [None if value is None else (value[0].detach(), value[1].detach()) for value in states]


def _canonical(row_output):
    return {
        "farmer": row_output.farmer.chosen_action,
        "hands": [decision.chosen_action for decision in row_output.hands],
        "market": [decision.chosen_action for decision in row_output.market],
    }


def _mean_tensor_dict(rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not rows:
        raise RuntimeError("cannot average empty BC loss list")
    keys = rows[0].keys()
    return {key: torch.stack([row[key] for row in rows]).mean() for key in keys}


def _float_dict(values: dict[str, torch.Tensor]) -> dict[str, float]:
    return {key: float(value.detach().cpu()) for key, value in values.items()}

def _teacher_forced_chunk(model, active_chunks, states, device):
    step_losses = []
    max_len = max(len(chunk.rows) for _, chunk in active_chunks)
    for time_index in range(max_len):
        active = [(slot, chunk) for slot, chunk in active_chunks if time_index < len(chunk.rows)]
        slots = [slot for slot, _ in active]
        one_steps = [
            SequenceChunk(chunk.episode_id, chunk.seat, (chunk.rows[time_index],),
                          chunk.episode_start and time_index == 0,
                          chunk.episode_end and time_index == len(chunk.rows) - 1)
            for _, chunk in active
        ]
        batch = collate_v2_sequences(one_steps, 1)
        move_step_batch(batch.flat, device)
        state = _pack_states(states, slots)
        output = model.forward_sequence(
            batch.flat, teacher_actions=batch.flat.canonical_actions, state=state,
        )
        step_losses.append(total_pretrain_loss(output, batch.flat))
        h, c = output.recurrent_state
        for row_index, slot in enumerate(slots):
            states[slot] = (h[row_index], c[row_index])
    return _mean_tensor_dict(step_losses), states


def _episode_groups(dataset: V2EpisodeDataset, batch_sequences: int, seed: int, epoch: int):
    episodes = balanced_episode_order(dataset, seed, epoch)
    for start in range(0, len(episodes), batch_sequences):
        yield episodes[start:start + batch_sequences]


def _config_dict(config: BCConfig) -> dict[str, Any]:
    value = asdict(config)
    for key in ("dataset_path", "stage0_marker", "stage1_marker", "output_dir"):
        value[key] = str(value[key])
    value["selection_weights"] = dict(SELECTION_WEIGHTS)
    return value

def _train_epoch(model, optimizer, dataset, config, epoch, train_steps, recurrent_stats, device):
    model.train()
    metrics = []
    for episodes in _episode_groups(dataset, config.batch_sequences, config.seed, epoch):
        chunks_by_slot = [_episode_chunks(episode, config.sequence_len) for episode in episodes]
        states = [None] * len(episodes)
        max_chunks = max(len(chunks) for chunks in chunks_by_slot)
        for chunk_index in range(max_chunks):
            if config.max_train_steps is not None and train_steps >= config.max_train_steps:
                break
            active = [(slot, chunks[chunk_index]) for slot, chunks in enumerate(chunks_by_slot)
                      if chunk_index < len(chunks)]
            for slot, _ in active:
                if states[slot] is None:
                    recurrent_stats["state_resets"] += 1
                else:
                    recurrent_stats["state_carries"] += 1
            optimizer.zero_grad(set_to_none=True)
            losses, states = _teacher_forced_chunk(model, active, states, device)
            if not torch.isfinite(losses["total"]):
                raise RuntimeError("non-finite BC loss")
            losses["total"].backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            if not torch.isfinite(torch.as_tensor(norm)):
                raise RuntimeError("non-finite BC gradient norm")
            optimizer.step()
            states = _detach_states(states)
            metrics.append(_float_dict(losses))
            train_steps += 1
        if config.max_train_steps is not None and train_steps >= config.max_train_steps:
            break
    if not metrics:
        raise RuntimeError("BC epoch executed no optimizer steps")
    keys = metrics[0].keys()
    mean = {key: sum(row[key] for row in metrics) / len(metrics) for key in keys}
    return mean, train_steps

def _validation_chunks(dataset, config):
    chunks = list(dataset.iter_chunks(config.sequence_len))
    if config.max_val_chunks is not None:
        chunks = chunks[: config.max_val_chunks]
    if not chunks:
        raise RuntimeError("validation split produced no chunks")
    return chunks


def _validation_episodes(chunks):
    grouped: dict[tuple[int, int], list[SequenceChunk]] = {}
    for chunk in chunks:
        grouped.setdefault((chunk.episode_id, chunk.seat), []).append(chunk)
    return [grouped[key] for key in sorted(grouped)]

def _teacher_validation(model, chunks, batch_sequences=1, device="cpu"):
    losses = []
    episodes = _validation_episodes(chunks)
    with torch.no_grad():
        for start in range(0, len(episodes), batch_sequences):
            group = episodes[start:start + batch_sequences]
            states = [None] * len(group)
            max_chunks = max(len(value) for value in group)
            for chunk_index in range(max_chunks):
                active = [(slot, values[chunk_index]) for slot, values in enumerate(group)
                          if chunk_index < len(values)]
                values, states = _teacher_forced_chunk(model, active, states, device)
                states = _detach_states(states)
                losses.append(_float_dict(values))
    keys = losses[0].keys()
    return {key: sum(row[key] for row in losses) / len(losses) for key in keys}


def _free_running_validation(model, chunks, seed, batch_sequences=1, device="cpu"):
    predictions, targets = [], []
    rng = np.random.default_rng(seed)
    episodes = _validation_episodes(chunks)
    with torch.no_grad():
        for start in range(0, len(episodes), batch_sequences):
            group = episodes[start:start + batch_sequences]
            states = [None] * len(group)
            max_chunks = max(len(value) for value in group)
            for chunk_index in range(max_chunks):
                active_chunks = [(slot, values[chunk_index]) for slot, values in enumerate(group)
                                 if chunk_index < len(values)]
                max_len = max(len(chunk.rows) for _, chunk in active_chunks)
                for time_index in range(max_len):
                    active = [(slot, chunk) for slot, chunk in active_chunks
                              if time_index < len(chunk.rows)]
                    slots = [slot for slot, _ in active]
                    one_steps = [SequenceChunk(
                        chunk.episode_id, chunk.seat, (chunk.rows[time_index],), False, False
                    ) for _, chunk in active]
                    batch = collate_v2_sequences(one_steps, 1)
                    move_step_batch(batch.flat, device)
                    state = _pack_states(states, slots)
                    output = model.sample_step(batch.flat, state, rng, deterministic=True)
                    h, c = output.recurrent_state
                    for row_index, slot in enumerate(slots):
                        states[slot] = (h[row_index].detach(), c[row_index].detach())
                        predictions.append(_canonical(output.rows[row_index]))
                        targets.append(batch.flat.canonical_actions[row_index])
    return semantic_domain_metrics(predictions, targets, masks=None)

def _selection_score(metrics: dict[str, float]) -> float:
    return float(sum(SELECTION_WEIGHTS[key] * metrics[key] for key in SELECTION_WEIGHTS))


def _checkpoint(model, optimizer, config, dataset_sha, init_sha, epoch,
                train_steps, train_metrics, validation_metrics,
                recurrent_stats, collapse_events, resolved_device):
    return {
        "format_version": FORMAT_VERSION,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "dataset_sha256": dataset_sha,
        "init_checkpoint_sha256": init_sha,
        "config": _config_dict(config),
        "epoch": int(epoch),
        "train_steps": int(train_steps),
        "last_train_metrics": dict(train_metrics),
        "validation_metrics": dict(validation_metrics),
        "recurrent_stats": dict(recurrent_stats),
        "collapse_events": list(collapse_events),
        "resolved_device": str(resolved_device),
    }


def _write_manifest(output_dir: Path, paths: list[Path]):
    lines = [f"{_sha256(path)}  {path.name}" for path in paths if path.is_file()]
    (output_dir / "manifest.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")

def run_bc(config: BCConfig, init_checkpoint: Path) -> Path:
    config.validate()
    acceptance = verify_training_acceptance(config.stage0_marker, config.stage1_marker)
    dataset_sha = _sha256(config.dataset_path)
    if dataset_sha != acceptance["dataset_sha256"]:
        raise RuntimeError("BC dataset SHA does not match accepted Stage 0 corpus")
    init_checkpoint = Path(init_checkpoint)
    initial = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
    if initial.get("dataset_sha256") not in {None, dataset_sha}:
        raise RuntimeError("pretraining checkpoint dataset SHA mismatch")

    torch.set_num_threads(config.torch_num_threads)
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    device = resolve_training_device(config.device)
    model = RecurrentIntentPolicy()
    model.load_state_dict(initial["model_state"], strict=True)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    train_data = V2EpisodeDataset(config.dataset_path, "train", {"active_best"})
    val_data = V2EpisodeDataset(config.dataset_path, "val", {"active_best"})
    val_chunks = _validation_chunks(val_data, config)

    output_dir = Path(config.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    best_path, last_path = output_dir / "bc_best.pt", output_dir / "bc_last.pt"
    history_path = output_dir / "history.jsonl"; history_path.write_text("", encoding="utf-8")
    recurrent_stats = {"state_resets": 0, "state_carries": 0}
    validation_history: list[dict[str, float]] = []
    collapse_events: list[dict[str, Any]] = []
    freeze_until = {"farmer": 0, "market": 0}
    best_score = float("-inf")
    best_aux = float("inf")
    best_epoch = 0
    train_steps = 0
    for epoch in range(1, config.epochs + 1):
        frozen = _set_decoder_freeze(model, freeze_until, epoch)
        train_metrics, train_steps = _train_epoch(
            model, optimizer, train_data, config, epoch, train_steps, recurrent_stats, device,
        )
        model.eval()
        teacher = _teacher_validation(
            model, val_chunks, config.batch_sequences, device,
        )
        free = _free_running_validation(
            model, val_chunks, config.seed + epoch, config.batch_sequences, device,
        )
        validation = dict(free)
        validation["teacher_forced_total"] = float(teacher["total"])
        validation["selection_score"] = _selection_score(validation)
        validation_history.append(validation)
        collapsed = detect_collapse(
            validation_history, config.collapse_threshold, config.collapse_consecutive,
        )
        if collapsed:
            for domain in collapsed:
                freeze_until[domain] = max(
                    int(freeze_until.get(domain, 0)), epoch + config.max_freeze_epochs,
                )
            collapse_events.append({
                "epoch": epoch, "domains": sorted(collapsed),
                "freeze_until": dict(freeze_until),
            })

        payload = _checkpoint(
            model, optimizer, config, dataset_sha, _sha256(init_checkpoint), epoch,
            train_steps, train_metrics, validation, recurrent_stats, collapse_events, device,
        )
        torch.save(payload, last_path)
        score, aux = validation["selection_score"], validation["teacher_forced_total"]
        if score > best_score or (score == best_score and aux < best_aux):
            best_score, best_aux, best_epoch = score, aux, epoch
            torch.save(payload, best_path)
        history_row = {"epoch": epoch, "train": train_metrics,
                       "validation": validation, "collapse": sorted(collapsed),
                       "frozen": frozen, "freeze_until": dict(freeze_until)}
        with history_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(history_row, sort_keys=True) + "\n")
        if config.max_train_steps is not None and train_steps >= config.max_train_steps:
            break
        if epoch - best_epoch >= config.early_stop_patience:
            break

    if not best_path.is_file() or not last_path.is_file():
        raise RuntimeError("BC did not produce required checkpoints")
    _write_manifest(output_dir, [best_path, last_path, history_path])
    return best_path
