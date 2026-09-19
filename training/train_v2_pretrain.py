from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

from kaggrl.v2_device import move_optimizer_state, move_step_batch, resolve_training_device
from kaggrl.v2_model import RecurrentIntentPolicy
from kaggrl.v2_structure_pretrain import structure_reconstruction_loss
from kaggrl.v2_training_data import (
    SequenceChunk,
    V2EpisodeDataset,
    balanced_episode_order,
    collate_v2_sequences,
    verify_training_acceptance,
)

FORMAT_VERSION = 1
OBJECTIVE_WEIGHTS = {
    "effect": 0.25,
    "future_resource": 0.15,
    "unit_task": 0.15,
    "opponent_effect": 0.10,
    "value": 0.05,
    "structure": 1.0,
}

@dataclass(frozen=True)
class PretrainConfig:
    dataset_path: Path
    stage0_marker: Path
    stage1_marker: Path
    output_dir: Path
    seed: int = 20260917
    sequence_len: int = 32
    batch_sequences: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    mask_rate: float = 0.15
    epochs: int = 8
    early_stop_patience: int = 3
    torch_num_threads: int = 2
    device: str = "auto"
    max_train_steps: int | None = None
    max_val_chunks: int | None = None

    def validate(self) -> None:
        if self.sequence_len <= 0 or self.batch_sequences <= 0:
            raise ValueError("sequence_len and batch_sequences must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.early_stop_patience < 1:
            raise ValueError("early_stop_patience must be positive")
        if self.torch_num_threads < 1:
            raise ValueError("torch_num_threads must be positive")
        if not 0.0 <= self.mask_rate <= 1.0:
            raise ValueError("mask_rate must be in [0,1]")
        if self.learning_rate <= 0 or self.gradient_clip <= 0:
            raise ValueError("learning_rate and gradient_clip must be positive")

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _chunks(samples, sequence_len: int) -> list[SequenceChunk]:
    out: list[SequenceChunk] = []
    for episode in samples:
        total = len(episode.rows)
        for start in range(0, total, sequence_len):
            rows = episode.rows[start:start + sequence_len]
            out.append(SequenceChunk(episode.episode_id, episode.seat, tuple(rows),
                                     start == 0, start + len(rows) == total))
    return out


def _groups(values: list[SequenceChunk], size: int) -> Iterable[list[SequenceChunk]]:
    for start in range(0, len(values), size):
        yield values[start:start + size]


def _masked_unit_mse(prediction, target, mask):
    target = target.to(device=prediction.device, dtype=prediction.dtype)
    mask = mask.to(device=prediction.device, dtype=prediction.dtype)
    if prediction.shape != target.shape:
        raise ValueError("unit-task prediction/target shape mismatch")
    per_unit = (prediction - target).square().mean(dim=-1)
    per_row = (per_unit * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    active = mask.sum(dim=1) > 0
    return per_row[active].mean() if active.any() else per_unit.sum() * 0.0


def _mse(prediction, target):
    target = target.to(device=prediction.device, dtype=prediction.dtype)
    if prediction.shape != target.shape:
        raise ValueError(f"target shape mismatch: {prediction.shape} != {target.shape}")
    return F.mse_loss(prediction, target)

def _auxiliary_forward(model: RecurrentIntentPolicy, flat_batch):
    encoded = model.encoder(flat_batch)
    h, _, intent = model.core.step(encoded.fused, None)
    return model._auxiliary(encoded, h, intent)


def _losses(model: RecurrentIntentPolicy, sequence_batch, rng: random.Random, mask_rate: float):
    flat = sequence_batch.flat
    aux = _auxiliary_forward(model, flat)
    targets = flat.auxiliary_targets
    effect = _mse(aux.effect, targets["effect"])
    future_resource = _mse(aux.future_resource, targets["future_resource"])
    unit_task = _masked_unit_mse(aux.unit_task, targets["unit_task"], flat.own_unit_mask)
    opponent_effect = _mse(aux.opponent_effect, targets["opponent_effect"])
    terminal_money = _mse(aux.terminal_money, targets["terminal_money"])
    terminal_margin = _mse(aux.terminal_margin, targets["terminal_margin"])
    value = 0.5 * (terminal_money + terminal_margin)
    structure = structure_reconstruction_loss(
        model, flat, mask_rate=mask_rate, rng=rng,
    )
    values = {
        "effect": effect,
        "future_resource": future_resource,
        "unit_task": unit_task,
        "opponent_effect": opponent_effect,
        "value": value,
        "structure": structure,
    }
    values["total"] = sum(OBJECTIVE_WEIGHTS[name] * values[name] for name in OBJECTIVE_WEIGHTS)
    return values

def _float_metrics(losses: dict[str, torch.Tensor]) -> dict[str, float]:
    return {name: float(value.detach().cpu()) for name, value in losses.items()}


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise RuntimeError("metric aggregation received no rows")
    keys = rows[0].keys()
    return {key: sum(row[key] for row in rows) / len(rows) for key in keys}


def _json_config(config: PretrainConfig) -> dict:
    value = asdict(config)
    for key in ("dataset_path", "stage0_marker", "stage1_marker", "output_dir"):
        value[key] = str(value[key])
    value["objective_weights"] = dict(OBJECTIVE_WEIGHTS)
    return value


def _checkpoint_payload(model, optimizer, config, dataset_sha, epoch,
                        train_steps, train_metrics, validation_metrics,
                        resolved_device):
    return {
        "format_version": FORMAT_VERSION,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "dataset_sha256": dataset_sha,
        "config": _json_config(config),
        "epoch": int(epoch),
        "train_steps": int(train_steps),
        "last_train_metrics": dict(train_metrics),
        "validation_metrics": dict(validation_metrics),
        "resolved_device": str(resolved_device),
    }

def _train_epoch(model, optimizer, chunks, config, epoch, train_steps, device):
    model.train()
    rows: list[dict[str, float]] = []
    rng = random.Random(config.seed + 1000 * epoch + train_steps)
    for group in _groups(chunks, config.batch_sequences):
        if config.max_train_steps is not None and train_steps >= config.max_train_steps:
            break
        batch = collate_v2_sequences(group, config.sequence_len)
        move_step_batch(batch.flat, device)
        optimizer.zero_grad(set_to_none=True)
        losses = _losses(model, batch, rng, config.mask_rate)
        if not torch.isfinite(losses["total"]):
            raise RuntimeError("non-finite pretraining loss")
        losses["total"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        if not torch.isfinite(torch.as_tensor(grad_norm)):
            raise RuntimeError("non-finite gradient norm")
        optimizer.step()
        rows.append(_float_metrics(losses))
        train_steps += 1
    if not rows:
        raise RuntimeError("pretraining epoch executed no optimizer steps")
    return _mean_metrics(rows), train_steps


def _validate(model, chunks, config, epoch, device):
    model.eval()
    rows: list[dict[str, float]] = []
    rng = random.Random(config.seed + 900000 + epoch)
    groups = list(_groups(chunks, config.batch_sequences))
    if config.max_val_chunks is not None:
        groups = groups[: config.max_val_chunks]
    with torch.no_grad():
        for group in groups:
            batch = collate_v2_sequences(group, config.sequence_len)
            move_step_batch(batch.flat, device)
            losses = _losses(model, batch, rng, config.mask_rate)
            rows.append(_float_metrics(losses))
    if not rows:
        raise RuntimeError("validation split produced no chunks")
    return _mean_metrics(rows)

def _write_manifest(output_dir: Path, paths: list[Path]) -> None:
    lines = [f"{_sha256(path)}  {path.name}" for path in paths if path.is_file()]
    (output_dir / "manifest.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_history(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _verify_resume_config(saved: dict, current: dict) -> None:
    mutable = {
        "epochs", "device", "torch_num_threads",
        "dataset_path", "stage0_marker", "stage1_marker", "output_dir",
    }
    for key, value in saved.items():
        if key in mutable:
            continue
        if current.get(key) != value:
            raise RuntimeError(f"resume config mismatch: {key}")


def run_pretraining(config: PretrainConfig, resume_checkpoint: Path | None = None) -> Path:
    config.validate()
    acceptance = verify_training_acceptance(config.stage0_marker, config.stage1_marker)
    dataset_sha = _sha256(config.dataset_path)
    if dataset_sha != acceptance["dataset_sha256"]:
        raise RuntimeError("training dataset SHA does not match accepted Stage 0 corpus")

    torch.set_num_threads(config.torch_num_threads)
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    device = resolve_training_device(config.device)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "pretrain_best.pt"
    last_path = output_dir / "pretrain_last.pt"
    history_path = output_dir / "history.jsonl"

    resume = None
    if resume_checkpoint is not None:
        resume_checkpoint = Path(resume_checkpoint)
        resume = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        if resume.get("dataset_sha256") != dataset_sha:
            raise RuntimeError("resume checkpoint dataset SHA mismatch")
        if int(resume.get("format_version", -1)) != FORMAT_VERSION:
            raise RuntimeError("resume checkpoint format version mismatch")
        _verify_resume_config(resume.get("config") or {}, _json_config(config))
    elif best_path.exists() or last_path.exists() or history_path.exists():
        raise FileExistsError(f"pretraining output exists; use resume_checkpoint: {output_dir}")

    model = RecurrentIntentPolicy()
    if resume is not None:
        model.load_state_dict(resume["model_state"], strict=True)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    if resume is not None:
        optimizer.load_state_dict(resume["optimizer_state"])
        move_optimizer_state(optimizer, device)

    train_data = V2EpisodeDataset(config.dataset_path, "train", {"active_best"})
    val_data = V2EpisodeDataset(config.dataset_path, "val", {"active_best"})
    val_chunks = _chunks(val_data, config.sequence_len)

    if resume is None:
        history_path.write_text("", encoding="utf-8")
        best_score = float("inf")
        best_epoch = 0
        train_steps = 0
        start_epoch = 1
    else:
        history = _load_history(history_path)
        if not history or int(history[-1].get("epoch", -1)) != int(resume["epoch"]):
            raise RuntimeError("resume history does not match checkpoint epoch")
        if not best_path.is_file():
            raise RuntimeError("resume requires existing pretrain_best.pt")
        best_row = min(history, key=lambda row: float(row["validation"]["total"]))
        best_score = float(best_row["validation"]["total"])
        best_epoch = int(best_row["epoch"])
        train_steps = int(resume.get("train_steps", 0))
        start_epoch = int(resume["epoch"]) + 1
        if start_epoch > config.epochs:
            return best_path

    for epoch in range(start_epoch, config.epochs + 1):
        train_order = balanced_episode_order(train_data, config.seed, epoch)
        train_chunks = _chunks(train_order, config.sequence_len)
        train_metrics, train_steps = _train_epoch(
            model, optimizer, train_chunks, config, epoch, train_steps, device,
        )
        validation_metrics = _validate(model, val_chunks, config, epoch, device)
        payload = _checkpoint_payload(
            model, optimizer, config, dataset_sha, epoch,
            train_steps, train_metrics, validation_metrics, device,
        )
        torch.save(payload, last_path)
        score = float(validation_metrics["total"])
        if score < best_score:
            best_score = score
            best_epoch = epoch
            torch.save(payload, best_path)
        history_row = {
            "epoch": epoch,
            "train_steps": train_steps,
            "train": train_metrics,
            "validation": validation_metrics,
            "best_validation_total": best_score,
        }
        with history_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(history_row, sort_keys=True) + "\n")
        if config.max_train_steps is not None and train_steps >= config.max_train_steps:
            break
        if epoch - best_epoch >= config.early_stop_patience:
            break

    if not best_path.is_file() or not last_path.is_file():
        raise RuntimeError("pretraining did not produce required checkpoints")
    _write_manifest(output_dir, [best_path, last_path, history_path])
    return best_path
