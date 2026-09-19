from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from kaggrl.v2_device import (
    move_optimizer_state,
    move_step_batch,
    resolve_training_device,
)
from kaggrl.v2_training_data import (
    EpisodeSequence,
    SequenceChunk,
    V2EpisodeDataset,
    balanced_episode_order,
    collate_v2_sequences,
    verify_training_acceptance,
)
from kaggrl.v3_model import TemporalIntentPolicy
from kaggrl.v3_structure_pretrain import structure_reconstruction_loss_v3
from kaggrl.v3_temporal import TemporalState
from training.train_v2_pretrain import (
    _masked_unit_mse,
    _mse,
    _sha256,
)

ARCHITECTURE_VERSION = "rl_v3_temporal_attention"
FORMAT_VERSION = 3
OBJECTIVE_WEIGHTS = {
    "effect": 0.25,
    "future_resource": 0.15,
    "unit_task": 0.15,
    "opponent_effect": 0.10,
    "value": 0.05,
    "structure": 1.0,
}
TEMPORAL_CONFIG = {
    "hidden_dim": 256,
    "attention_dim": 128,
    "heads": 4,
    "window": 32,
    "blocks": 1,
    "dropout": 0.0,
}


@dataclass(frozen=True)
class PretrainV3Config:
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
    initialization_mode: str = "clean"
    max_train_steps: int | None = None
    max_val_chunks: int | None = None

    def validate(self) -> None:
        if self.sequence_len <= 0 or self.batch_sequences <= 0:
            raise ValueError("sequence_len and batch_sequences must be positive")
        if self.epochs <= 0 or self.early_stop_patience < 1:
            raise ValueError("invalid epoch configuration")
        if self.learning_rate <= 0 or self.gradient_clip <= 0:
            raise ValueError("invalid optimizer configuration")
        if self.torch_num_threads < 1:
            raise ValueError("torch_num_threads must be positive")
        if not 0.0 <= self.mask_rate <= 1.0:
            raise ValueError("mask_rate must be in [0,1]")
        if self.initialization_mode not in {"clean", "partial_v2_warm_start"}:
            raise ValueError("unknown initialization_mode")


def _config_dict(config: PretrainV3Config) -> dict[str, Any]:
    value = asdict(config)
    for key in ("dataset_path", "stage0_marker", "stage1_marker", "output_dir"):
        value[key] = str(value[key])
    value["objective_weights"] = dict(OBJECTIVE_WEIGHTS)
    value["temporal_config"] = dict(TEMPORAL_CONFIG)
    return value

def _config_sha(config: PretrainV3Config) -> str:
    text = json.dumps(_config_dict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _episode_chunks(episode: EpisodeSequence, length: int) -> list[SequenceChunk]:
    chunks = []
    for start in range(0, len(episode.rows), length):
        rows = episode.rows[start:start + length]
        chunks.append(SequenceChunk(
            episode.episode_id, episode.seat, tuple(rows),
            start == 0, start + len(rows) == len(episode.rows),
        ))
    return chunks


def _episode_groups(dataset, batch_sequences: int, seed: int, epoch: int):
    episodes = balanced_episode_order(dataset, seed, epoch)
    for start in range(0, len(episodes), batch_sequences):
        yield episodes[start:start + batch_sequences]


def _pack_states(states, slots) -> TemporalState | None:
    chosen = [states[slot] for slot in slots]
    if all(value is None for value in chosen):
        return None
    if any(value is None for value in chosen):
        raise RuntimeError("mixed initialized/uninitialized temporal state")
    return TemporalState(
        h=torch.cat([value.h for value in chosen], dim=0),
        c=torch.cat([value.c for value in chosen], dim=0),
        memory=torch.cat([value.memory for value in chosen], dim=0),
        valid_length=torch.cat([value.valid_length for value in chosen], dim=0),
        write_pos=torch.cat([value.write_pos for value in chosen], dim=0),
    )

def _store_state(states, slots, packed: TemporalState) -> None:
    for row_index, slot in enumerate(slots):
        states[slot] = TemporalState(
            h=packed.h[row_index:row_index + 1],
            c=packed.c[row_index:row_index + 1],
            memory=packed.memory[row_index:row_index + 1],
            valid_length=packed.valid_length[row_index:row_index + 1],
            write_pos=packed.write_pos[row_index:row_index + 1],
        )


def _detach_states(model, states):
    return [model.core.detach_state(value) for value in states]


def _mean_tensor_dict(rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not rows:
        raise RuntimeError("cannot average empty loss list")
    return {
        key: torch.stack([row[key] for row in rows]).mean()
        for key in rows[0]
    }


def _float_dict(values: dict[str, torch.Tensor]) -> dict[str, float]:
    return {key: float(value.detach().cpu()) for key, value in values.items()}


def _mean_float_dict(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise RuntimeError("cannot average empty metric list")
    return {
        key: sum(row[key] for row in rows) / len(rows)
        for key in rows[0]
    }

def _step_losses(
    model: TemporalIntentPolicy,
    flat,
    state: TemporalState | None,
    rng: random.Random,
    mask_rate: float,
):
    encoded = model.encoder(flat)
    fused, intent, next_state, _ = model.core.step(
        encoded.fused,
        flat.previous_action_global,
        flat.previous_effect,
        flat.economy,
        state,
    )
    aux = model._auxiliary(encoded, fused, intent)
    targets = flat.auxiliary_targets
    effect = _mse(aux.effect, targets["effect"])
    future_resource = _mse(aux.future_resource, targets["future_resource"])
    unit_task = _masked_unit_mse(
        aux.unit_task, targets["unit_task"], flat.own_unit_mask,
    )
    opponent_effect = _mse(aux.opponent_effect, targets["opponent_effect"])
    terminal_money = _mse(aux.terminal_money, targets["terminal_money"])
    terminal_margin = _mse(aux.terminal_margin, targets["terminal_margin"])
    value = 0.5 * (terminal_money + terminal_margin)
    structure = structure_reconstruction_loss_v3(
        model, flat, encoded, fused, intent,
        mask_rate=mask_rate, rng=rng,
    )
    values = {
        "effect": effect,
        "future_resource": future_resource,
        "unit_task": unit_task,
        "opponent_effect": opponent_effect,
        "value": value,
        "structure": structure,
    }
    values["total"] = sum(
        OBJECTIVE_WEIGHTS[name] * values[name]
        for name in OBJECTIVE_WEIGHTS
    )
    return values, next_state

def _run_chunk(
    model,
    active_chunks,
    states,
    config,
    device,
    rng,
    recurrent_stats=None,
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
                chunk.episode_id,
                chunk.seat,
                (chunk.rows[time_index],),
                chunk.episode_start and time_index == 0,
                chunk.episode_end and time_index == len(chunk.rows) - 1,
            )
            for _, chunk in active
        ]
        batch = collate_v2_sequences(one_steps, 1)
        move_step_batch(batch.flat, device)
        packed = _pack_states(states, slots)
        losses, next_state = _step_losses(
            model, batch.flat, packed, rng, config.mask_rate,
        )
        _store_state(states, slots, next_state)
        step_losses.append(losses)
        if recurrent_stats is not None:
            recurrent_stats["temporal_steps"] += len(slots)
    return _mean_tensor_dict(step_losses), states

def _train_epoch(
    model, optimizer, dataset, config, epoch,
    train_steps, recurrent_stats, device,
):
    model.train()
    metrics = []
    rng = random.Random(config.seed + 1000 * epoch + train_steps)
    stop = False
    for episodes in _episode_groups(
        dataset, config.batch_sequences, config.seed, epoch,
    ):
        chunks_by_slot = [
            _episode_chunks(episode, config.sequence_len)
            for episode in episodes
        ]
        states = [None] * len(episodes)
        max_chunks = max(len(chunks) for chunks in chunks_by_slot)
        for chunk_index in range(max_chunks):
            if config.max_train_steps is not None and train_steps >= config.max_train_steps:
                stop = True
                break
            active = [
                (slot, chunks[chunk_index])
                for slot, chunks in enumerate(chunks_by_slot)
                if chunk_index < len(chunks)
            ]
            for slot, _ in active:
                if states[slot] is None:
                    recurrent_stats["state_resets"] += 1
                else:
                    recurrent_stats["state_carries"] += 1
            optimizer.zero_grad(set_to_none=True)
            losses, states = _run_chunk(
                model, active, states, config, device, rng, recurrent_stats,
            )
            if not torch.isfinite(losses["total"]):
                raise RuntimeError("non-finite v3 pretraining loss")
            losses["total"].backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip,
            )
            if not torch.isfinite(torch.as_tensor(norm)):
                raise RuntimeError("non-finite v3 pretraining gradient norm")
            optimizer.step()
            states = _detach_states(model, states)
            metrics.append(_float_dict(losses))
            train_steps += 1
        if stop:
            break
    if not metrics:
        raise RuntimeError("v3 pretraining epoch executed no optimizer steps")
    return _mean_float_dict(metrics), train_steps

def _validation_chunks(dataset, config):
    chunks = list(dataset.iter_chunks(config.sequence_len))
    if config.max_val_chunks is not None:
        chunks = chunks[: config.max_val_chunks]
    if not chunks:
        raise RuntimeError("validation split produced no chunks")
    grouped: dict[tuple[int, int], list[SequenceChunk]] = {}
    for chunk in chunks:
        grouped.setdefault((chunk.episode_id, chunk.seat), []).append(chunk)
    return [grouped[key] for key in sorted(grouped)]


def _validate(model, dataset, config, epoch, device):
    model.eval()
    metrics = []
    rng = random.Random(config.seed + 900000 + epoch)
    episodes = _validation_chunks(dataset, config)
    with torch.no_grad():
        for start in range(0, len(episodes), config.batch_sequences):
            group = episodes[start:start + config.batch_sequences]
            states = [None] * len(group)
            max_chunks = max(len(chunks) for chunks in group)
            for chunk_index in range(max_chunks):
                active = [
                    (slot, chunks[chunk_index])
                    for slot, chunks in enumerate(group)
                    if chunk_index < len(chunks)
                ]
                losses, states = _run_chunk(
                    model, active, states, config, device, rng,
                    recurrent_stats=None,
                )
                states = _detach_states(model, states)
                metrics.append(_float_dict(losses))
    return _mean_float_dict(metrics)


def _write_manifest(output_dir: Path, paths: list[Path]) -> None:
    lines = [
        f"{_sha256(path)}  {path.name}"
        for path in paths if path.is_file()
    ]
    (output_dir / "manifest.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf-8",
    )

def _code_sha256() -> str:
    root = Path(__file__).resolve().parents[1]
    paths = [
        Path(__file__).resolve(),
        root / "src/kaggrl/v3_temporal.py",
        root / "src/kaggrl/v3_model.py",
        root / "src/kaggrl/v3_structure_pretrain.py",
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _checkpoint_payload(
    model, optimizer, config, dataset_sha, epoch, train_steps,
    train_metrics, validation_metrics, recurrent_stats, resolved_device,
):
    return {
        "format_version": FORMAT_VERSION,
        "architecture_version": ARCHITECTURE_VERSION,
        "temporal_config": dict(TEMPORAL_CONFIG),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "dataset_sha256": dataset_sha,
        "config": _config_dict(config),
        "config_sha256": _config_sha(config),
        "code_sha256": _code_sha256(),
        "initialization_mode": config.initialization_mode,
        "epoch": int(epoch),
        "train_steps": int(train_steps),
        "last_train_metrics": dict(train_metrics),
        "validation_metrics": dict(validation_metrics),
        "recurrent_stats": dict(recurrent_stats),
        "resolved_device": str(resolved_device),
    }

def _load_history(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run_v3_pretraining(
    config: PretrainV3Config,
    init_checkpoint: Path | None = None,
    resume_checkpoint: Path | None = None,
) -> Path:
    config.validate()
    acceptance = verify_training_acceptance(
        config.stage0_marker, config.stage1_marker,
    )
    dataset_sha = _sha256(config.dataset_path)
    if dataset_sha != acceptance["dataset_sha256"]:
        raise RuntimeError("v3 training dataset SHA does not match accepted corpus")

    torch.set_num_threads(config.torch_num_threads)
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    device = resolve_training_device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "pretrain_best.pt"
    last_path = output_dir / "pretrain_last.pt"
    history_path = output_dir / "history.jsonl"
    run_config_path = output_dir / "run_config.json"
    if resume_checkpoint is not None and init_checkpoint is not None:
        raise ValueError("resume_checkpoint and init_checkpoint are mutually exclusive")

    resume = None
    if resume_checkpoint is not None:
        resume = torch.load(
            Path(resume_checkpoint), map_location="cpu", weights_only=False,
        )
        if resume.get("architecture_version") != ARCHITECTURE_VERSION:
            raise RuntimeError("resume checkpoint architecture mismatch")
        if resume.get("dataset_sha256") != dataset_sha:
            raise RuntimeError("resume checkpoint dataset SHA mismatch")
    elif best_path.exists() or last_path.exists() or history_path.exists():
        raise FileExistsError(
            f"v3 pretraining output exists; use resume_checkpoint: {output_dir}"
        )

    model = TemporalIntentPolicy()
    if resume is not None:
        model.load_state_dict(resume["model_state"], strict=True)
    elif init_checkpoint is not None:
        initial = torch.load(
            Path(init_checkpoint), map_location="cpu", weights_only=False,
        )
        if initial.get("architecture_version") != ARCHITECTURE_VERSION:
            raise RuntimeError("v3 initialization checkpoint architecture mismatch")
        if initial.get("dataset_sha256") not in {None, dataset_sha}:
            raise RuntimeError("v3 initialization checkpoint dataset SHA mismatch")
        model.load_state_dict(initial["model_state"], strict=True)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    if resume is not None:
        optimizer.load_state_dict(resume["optimizer_state"])
        move_optimizer_state(optimizer, device)
    train_data = V2EpisodeDataset(
        config.dataset_path, "train", {"active_best"},
    )
    val_data = V2EpisodeDataset(
        config.dataset_path, "val", {"active_best"},
    )
    recurrent_stats = {
        "temporal_steps": 0,
        "state_resets": 0,
        "state_carries": 0,
    }

    if resume is None:
        history_path.write_text("", encoding="utf-8")
        run_config_path.write_text(
            json.dumps(_config_dict(config), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        start_epoch = 1
        train_steps = 0
        best_score = float("inf")
        best_epoch = 0
    else:
        history = _load_history(history_path)
        if not history or int(history[-1]["epoch"]) != int(resume["epoch"]):
            raise RuntimeError("resume history does not match checkpoint epoch")
        start_epoch = int(resume["epoch"]) + 1
        train_steps = int(resume.get("train_steps", 0))
        recurrent_stats.update(resume.get("recurrent_stats") or {})
        best_row = min(
            history, key=lambda row: float(row["validation"]["total"]),
        )
        best_score = float(best_row["validation"]["total"])
        best_epoch = int(best_row["epoch"])
        if start_epoch > config.epochs:
            return best_path
    for epoch in range(start_epoch, config.epochs + 1):
        train_metrics, train_steps = _train_epoch(
            model, optimizer, train_data, config, epoch,
            train_steps, recurrent_stats, device,
        )
        validation_metrics = _validate(
            model, val_data, config, epoch, device,
        )
        payload = _checkpoint_payload(
            model, optimizer, config, dataset_sha, epoch, train_steps,
            train_metrics, validation_metrics, recurrent_stats, device,
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
            "recurrent_stats": dict(recurrent_stats),
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(history_row, sort_keys=True) + "\n")
        if config.max_train_steps is not None and train_steps >= config.max_train_steps:
            break
        if epoch - best_epoch >= config.early_stop_patience:
            break

    if not best_path.is_file() or not last_path.is_file():
        raise RuntimeError("v3 pretraining did not produce required checkpoints")
    _write_manifest(
        output_dir,
        [best_path, last_path, history_path, run_config_path],
    )
    return best_path
