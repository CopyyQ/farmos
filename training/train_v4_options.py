from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kaggrl.clock import CLOCK_FEATURES, GameClock
from kaggrl.v4_option_dataset import contiguous_window_starts
from kaggrl.v4_option_model import V4OptionPolicy, v4_option_loss

OBS_DIM = 1024
ARCHITECTURE_VERSION = V4OptionPolicy.ARCHITECTURE_VERSION


@dataclass(frozen=True)
class TrainConfig:
    dataset: pathlib.Path
    manifest: pathlib.Path
    output_dir: pathlib.Path
    sequence_len: int = 32
    batch_sequences: int = 64
    epochs: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    hidden_dim: int = 192
    gradient_clip: float = 1.0
    seed: int = 20260920
    device: str = "cuda"
    use_amp: bool = True


class OptionWindowDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        features: np.ndarray,
        indices: list[np.ndarray],
        route_to_class: dict[int, int],
        *,
        episode_steps: int,
        turns_per_day: int,
    ):
        self.frame = frame
        self.features = features
        self.indices = indices
        self.route_to_class = route_to_class
        route_ids = frame["route_id"].astype(int).to_numpy()
        self.route_class = np.asarray(
            [route_to_class[int(value)] for value in route_ids],
            dtype=np.int64,
        )
        self.route_conf = frame["route_confidence"].to_numpy(np.float32)
        route_count = len(route_to_class)
        mask_bits = frame["route_mask_bits"].astype("int64").to_numpy()
        self.route_mask = np.asarray([
            [
                bool(int(bits) & (1 << route_index))
                for route_index in range(route_count)
            ]
            for bits in mask_bits
        ], dtype=np.bool_)
        label_allowed = self.route_mask[
            np.arange(len(self.route_class)), self.route_class
        ]
        if not bool(label_allowed.all()):
            raise RuntimeError("V4 route label outside stored compatibility mask")
        self.market = frame["market_mode_id"].to_numpy(np.int64)
        self.phase = frame["phase_id"].to_numpy(np.int64)
        self.step_norm = frame["step_norm"].to_numpy(np.float32)
        self.remaining_norm = frame["remaining_norm"].to_numpy(np.float32)
        steps = frame["step"].astype(int).to_numpy()
        self.clock_context = np.asarray([
            GameClock(
                step=int(step),
                day=int(step) // int(turns_per_day),
                hour=int(step) % int(turns_per_day),
                turns_per_day=int(turns_per_day),
                episode_steps=int(episode_steps),
            ).features()
            for step in steps
        ], dtype=np.float32)
        if self.clock_context.shape[1] != len(CLOCK_FEATURES):
            raise RuntimeError("V4 clock context width mismatch")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        idx = self.indices[index]
        clock = np.stack(
            [self.step_norm[idx], self.remaining_norm[idx]],
            axis=-1,
        )
        return {
            "obs": torch.from_numpy(
                self.features[idx].astype(np.float32, copy=False)
            ),
            "route": torch.from_numpy(self.route_class[idx]),
            "route_conf": torch.from_numpy(self.route_conf[idx]),
            "route_mask": torch.from_numpy(self.route_mask[idx]),
            "route_gate": torch.from_numpy(
                (self.route_conf[idx] >= 0.05).astype(np.float32)
            ),
            "market": torch.from_numpy(self.market[idx]),
            "phase": torch.from_numpy(self.phase[idx]),
            "clock_context": torch.from_numpy(self.clock_context[idx]),
            "clock_target": torch.from_numpy(clock),
        }


def _load_manifest(path: pathlib.Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "farmos_v4_option_rows_v1":
        raise RuntimeError("unsupported V4 option dataset schema")
    if manifest.get("observation_schema") != "macro_semantic_v4_clock_v2":
        raise RuntimeError("V4 option dataset has wrong observation schema")
    if int(manifest.get("observation_dim", -1)) != OBS_DIM:
        raise RuntimeError("V4 option observation width mismatch")
    if manifest.get("route_masking") != "shop_and_phase_compatible_bitset_v1":
        raise RuntimeError("V4 option route-mask schema mismatch")
    return manifest


def _decode_features(frame: pd.DataFrame) -> np.ndarray:
    out = np.empty((len(frame), OBS_DIM), dtype=np.float16)
    for index, raw in enumerate(frame["obs_f16"].tolist()):
        values = np.frombuffer(raw, dtype=np.float16)
        if values.size != OBS_DIM:
            raise RuntimeError(
                f"bad V4 option feature width row={index}: {values.size}"
            )
        out[index] = values
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite V4 option features")
    return out


def _window_indices(
    frame: pd.DataFrame,
    split: str,
    sequence_len: int,
) -> list[np.ndarray]:
    windows: list[np.ndarray] = []
    subset = frame[frame["split"].eq(split)]
    for (_, _), group in subset.groupby(
        ["episode_id", "seat"],
        sort=False,
    ):
        group = group.sort_values("step", kind="stable")
        rows = group.index.to_numpy(np.int64)
        steps = group["step"].astype(int).tolist()
        for start in contiguous_window_starts(steps, sequence_len):
            part = rows[start:start + sequence_len]
            if len(part) != sequence_len:
                raise RuntimeError("short V4 strategic window")
            windows.append(part)
    return windows


def _to_device(batch, device):
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
    }


def _market_class_weights(
    frame: pd.DataFrame,
    *,
    class_count: int,
    power: float = 1.0,
    cap: float = 8.0,
) -> torch.Tensor:
    train = frame[frame["split"].eq("train")]
    counts = np.bincount(
        train["market_mode_id"].astype(int).to_numpy(),
        minlength=int(class_count),
    ).astype(np.float64)
    if np.any(counts <= 0):
        raise RuntimeError(
            f"missing market class in train split: {counts.tolist()}"
        )
    reference = float(np.max(counts))
    weights = np.power(reference / counts, float(power))
    weights = np.minimum(weights, float(cap))
    weights = weights / max(1e-12, float(weights.mean()))
    return torch.tensor(weights, dtype=torch.float32)


def _route_gate_pos_weight(
    frame: pd.DataFrame,
    *,
    threshold: float = 0.05,
    cap: float = 8.0,
) -> torch.Tensor:
    train = frame[frame["split"].eq("train")]
    positive = int(
        (train["route_confidence"].to_numpy(np.float32)
         >= float(threshold)).sum()
    )
    negative = int(len(train) - positive)
    if positive <= 0 or negative <= 0:
        raise RuntimeError(
            f"route gate needs both classes: pos={positive} neg={negative}"
        )
    value = min(float(cap), negative / positive)
    return torch.tensor(value, dtype=torch.float32)


def _loss(
    outputs,
    batch,
    market_class_weight=None,
    route_gate_pos_weight=None,
):
    return v4_option_loss(
        outputs,
        route_target=batch["route"],
        route_confidence=batch["route_conf"],
        route_gate_target=batch["route_gate"],
        route_mask=batch["route_mask"],
        market_target=batch["market"],
        phase_target=batch["phase"],
        clock_target=batch["clock_target"],
        market_class_weight=market_class_weight,
        route_gate_pos_weight=route_gate_pos_weight,
    )


def _route_gate_stats(
    probabilities: np.ndarray,
    truth: np.ndarray,
    threshold: float,
) -> dict:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(truth, dtype=bool)
    predicted = probabilities >= float(threshold)
    tp = int(np.sum(predicted & truth))
    tn = int(np.sum((~predicted) & (~truth)))
    fp = int(np.sum(predicted & (~truth)))
    fn = int(np.sum((~predicted) & truth))
    positive_recall = tp / max(1, tp + fn)
    negative_recall = tn / max(1, tn + fp)
    precision = tp / max(1, tp + fp)
    return {
        "threshold": float(threshold),
        "balanced_acc": 0.5 * (
            positive_recall + negative_recall
        ),
        "positive_recall": float(positive_recall),
        "negative_recall": float(negative_recall),
        "precision": float(precision),
    }


def _select_route_gate_threshold(
    probabilities: np.ndarray,
    truth: np.ndarray,
) -> dict:
    candidates = [
        _route_gate_stats(
            probabilities, truth, float(threshold)
        )
        for threshold in np.linspace(0.15, 0.85, 29)
    ]
    safe = [
        row for row in candidates
        if row["positive_recall"] >= 0.40
        and row["negative_recall"] >= 0.85
        and row["precision"] >= 0.30
    ]
    pool = safe if safe else candidates
    return max(
        pool,
        key=lambda row: (
            row["balanced_acc"],
            row["precision"],
            row["positive_recall"],
            -abs(row["threshold"] - 0.5),
        ),
    )


@torch.inference_mode()
def evaluate(
    model, loader, device, *,
    market_class_weight=None,
    route_gate_pos_weight=None,
    route_gate_threshold: float | None = None,
    last_step=719,
):
    model.eval()
    total_rows = 0
    market_correct = 0
    market_class_count = int(model.market_mode_count)
    market_true_by_class = torch.zeros(
        market_class_count, dtype=torch.long
    )
    market_correct_by_class = torch.zeros(
        market_class_count, dtype=torch.long
    )
    phase_correct = 0
    route_confident = 0
    route_correct = 0
    route_gate_probabilities = []
    route_gate_truth = []
    step_error = 0.0
    remaining_error = 0.0
    losses = []
    route_clock_values = []
    market_clock_values = []

    for batch in loader:
        batch = _to_device(batch, device)
        outputs, _ = model.forward_sequence(
            batch["obs"], batch["clock_context"]
        )
        loss = _loss(
            outputs,
            batch,
            market_class_weight,
            route_gate_pos_weight,
        )
        losses.append(float(loss.total.detach().cpu()))

        rows = int(batch["route"].numel())
        total_rows += rows
        route_logits = outputs["route"].masked_fill(
            ~batch["route_mask"].bool(),
            torch.finfo(outputs["route"].dtype).min,
        )
        route_pred = route_logits.argmax(-1)
        market_pred = outputs["market"].argmax(-1)
        phase_pred = outputs["phase"].argmax(-1)
        confident = batch["route_conf"] >= 0.05
        route_confident += int(confident.sum().item())
        route_correct += int(
            ((route_pred == batch["route"]) & confident).sum().item()
        )
        route_gate_probabilities.append(
            outputs["route_gate"].detach().float().cpu().reshape(-1)
        )
        route_gate_truth.append(
            batch["route_gate"].detach().float().cpu().reshape(-1)
        )
        market_matches = market_pred == batch["market"]
        market_correct += int(market_matches.sum().item())
        true_flat = batch["market"].reshape(-1)
        pred_flat = market_pred.reshape(-1)
        market_true_by_class += torch.bincount(
            true_flat.detach().cpu(),
            minlength=market_class_count,
        )
        correct_true = true_flat[market_matches.reshape(-1)]
        market_correct_by_class += torch.bincount(
            correct_true.detach().cpu(),
            minlength=market_class_count,
        )
        phase_correct += int(
            (phase_pred == batch["phase"]).sum().item()
        )
        clock_error = (
            outputs["clock"] - batch["clock_target"]
        ).abs()
        step_error += float(clock_error[..., 0].sum().item())
        remaining_error += float(clock_error[..., 1].sum().item())
        route_clock_values.append(
            outputs["route_clock"].detach().float().cpu().reshape(
                -1, outputs["route_clock"].shape[-1]
            )
        )
        market_clock_values.append(
            outputs["market_clock"].detach().float().cpu().reshape(
                -1, outputs["market_clock"].shape[-1]
            )
        )

    route_clock = (
        torch.cat(route_clock_values, dim=0)
        if route_clock_values else torch.zeros(1, 1)
    )
    market_clock = (
        torch.cat(market_clock_values, dim=0)
        if market_clock_values else torch.zeros(1, 1)
    )
    route_clock_std = float(
        route_clock.std(dim=0, unbiased=False).mean().item()
    )
    market_clock_std = float(
        market_clock.std(dim=0, unbiased=False).mean().item()
    )
    market_recall_by_class = {
        str(index): float(
            market_correct_by_class[index].item()
            / max(1, market_true_by_class[index].item())
        )
        for index in range(market_class_count)
    }
    market_balanced_acc = float(
        np.mean(list(market_recall_by_class.values()))
    )
    market_min_recall = float(
        min(market_recall_by_class.values())
    )
    gate_prob = (
        torch.cat(route_gate_probabilities).numpy()
        if route_gate_probabilities
        else np.zeros(1, dtype=np.float32)
    )
    gate_truth = (
        torch.cat(route_gate_truth).numpy() >= 0.5
        if route_gate_truth
        else np.zeros(1, dtype=bool)
    )
    gate_metrics = (
        _select_route_gate_threshold(gate_prob, gate_truth)
        if route_gate_threshold is None
        else _route_gate_stats(
            gate_prob, gate_truth, route_gate_threshold
        )
    )

    return {
        "rows": total_rows,
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "route_confident_rows": route_confident,
        "route_acc_confident": route_correct / max(1, route_confident),
        "route_gate_threshold": gate_metrics["threshold"],
        "route_gate_balanced_acc": gate_metrics["balanced_acc"],
        "route_gate_positive_recall": gate_metrics["positive_recall"],
        "route_gate_negative_recall": gate_metrics["negative_recall"],
        "route_gate_precision": gate_metrics["precision"],
        "market_acc": market_correct / max(1, total_rows),
        "market_balanced_acc": market_balanced_acc,
        "market_min_recall": market_min_recall,
        "market_recall_by_class": market_recall_by_class,
        "phase_acc": phase_correct / max(1, total_rows),
        "step_mae_turns": (
            step_error / max(1, total_rows) * last_step
        ),
        "remaining_mae_turns": (
            remaining_error / max(1, total_rows) * last_step
        ),
        "route_clock_logit_std": route_clock_std,
        "market_clock_logit_std": market_clock_std,
    }


def promotion_gate(metrics: dict) -> dict:
    checks = {
        "phase_acc_ge_0_99": metrics["phase_acc"] >= 0.99,
        "step_mae_le_1_5": metrics["step_mae_turns"] <= 1.5,
        "remaining_mae_le_1_5": metrics["remaining_mae_turns"] <= 1.5,
        "market_acc_ge_0_45": metrics["market_acc"] >= 0.45,
        "market_balanced_acc_ge_0_60": (
            metrics.get("market_balanced_acc", 0.0) >= 0.60
        ),
        "market_min_recall_ge_0_35": (
            metrics.get("market_min_recall", 0.0) >= 0.35
        ),
        "route_acc_confident_ge_0_35": (
            metrics["route_acc_confident"] >= 0.35
        ),
        "route_gate_balanced_acc_ge_0_65": (
            metrics.get("route_gate_balanced_acc", 0.0) >= 0.65
        ),
        "route_gate_positive_recall_ge_0_40": (
            metrics.get("route_gate_positive_recall", 0.0) >= 0.40
        ),
        "route_gate_negative_recall_ge_0_85": (
            metrics.get("route_gate_negative_recall", 0.0) >= 0.85
        ),
        "route_gate_precision_ge_0_30": (
            metrics.get("route_gate_precision", 0.0) >= 0.30
        ),
        "route_clock_logit_std_ge_0_01": (
            metrics.get("route_clock_logit_std", 0.0) >= 0.01
        ),
        "market_clock_logit_std_ge_0_01": (
            metrics.get("market_clock_logit_std", 0.0) >= 0.01
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
    }


def selection_score(metrics: dict) -> float:
    return (
        1.00 * metrics["route_acc_confident"]
        + 0.40 * metrics.get(
            "market_balanced_acc", metrics["market_acc"]
        )
        + 0.20 * metrics["phase_acc"]
        + 0.20 * metrics.get("route_gate_balanced_acc", 0.0)
        - 0.02 * metrics["step_mae_turns"]
        - 0.02 * metrics["remaining_mae_turns"]
        + 0.10 * metrics.get("route_clock_logit_std", 0.0)
        + 0.10 * metrics.get("market_clock_logit_std", 0.0)
    )


def _checkpoint(
    model,
    optimizer,
    config,
    manifest,
    route_ids,
    epoch,
    metrics,
):
    return {
        "architecture_version": ARCHITECTURE_VERSION,
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "input_dim": OBS_DIM,
        "hidden_dim": int(config.hidden_dim),
        "clock_dim": len(CLOCK_FEATURES),
        "clock_features": list(CLOCK_FEATURES),
        "route_ids": list(route_ids),
        "market_modes": list(manifest["market_modes"]),
        "sequence_len": int(config.sequence_len),
        "observation_schema": manifest["observation_schema"],
        "dataset_schema": manifest["schema_version"],
        "dataset_sha256": manifest.get("output_sha256"),
        "validation_metrics": metrics,
        "route_gate_threshold": float(
            metrics.get("route_gate_threshold", 0.5)
        ),
        "promotion_gate": promotion_gate(metrics),
    }


def train(config: TrainConfig) -> dict:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    manifest = _load_manifest(config.manifest)
    if int(manifest.get("sequence_len", config.sequence_len)) != int(
        config.sequence_len
    ):
        raise RuntimeError(
            "trainer sequence_len differs from dataset continuity contract"
        )
    frame = pd.read_parquet(config.dataset)
    frame = frame.sort_values(
        ["episode_id", "seat", "step"],
        kind="stable",
    ).reset_index(drop=True)
    features = _decode_features(frame)

    route_ids = tuple(int(value) for value in manifest["route_ids"])
    route_to_class = {
        route_id: index for index, route_id in enumerate(route_ids)
    }
    unknown = sorted(
        set(frame["route_id"].astype(int).tolist()) - set(route_ids)
    )
    if unknown:
        raise RuntimeError(f"unknown V4 route labels: {unknown}")

    train_windows = _window_indices(
        frame, "train", config.sequence_len
    )
    val_windows = _window_indices(
        frame, "val", config.sequence_len
    )
    test_windows = _window_indices(
        frame, "test", config.sequence_len
    )
    if not train_windows or not val_windows:
        raise RuntimeError("V4 option dataset needs train and val windows")

    dataset_kwargs = {
        "episode_steps": int(manifest.get("episode_steps", 720)),
        "turns_per_day": int(manifest.get("turns_per_day", 24)),
    }
    train_ds = OptionWindowDataset(
        frame, features, train_windows, route_to_class, **dataset_kwargs
    )
    val_ds = OptionWindowDataset(
        frame, features, val_windows, route_to_class, **dataset_kwargs
    )
    test_ds = OptionWindowDataset(
        frame, features, test_windows, route_to_class, **dataset_kwargs
    )
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_sequences,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=config.device.startswith("cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_sequences,
        shuffle=False,
        num_workers=0,
        pin_memory=config.device.startswith("cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_sequences,
        shuffle=False,
        num_workers=0,
        pin_memory=config.device.startswith("cuda"),
    )

    requested = torch.device(config.device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        requested = torch.device("cpu")
    device = requested
    market_class_weight = _market_class_weights(
        frame,
        class_count=len(manifest["market_modes"]),
    ).to(device)
    route_gate_pos_weight = _route_gate_pos_weight(frame).to(device)

    model = V4OptionPolicy(
        OBS_DIM,
        route_count=len(route_ids),
        market_mode_count=len(manifest["market_modes"]),
        hidden_dim=config.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    amp_enabled = bool(config.use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

    config.output_dir.mkdir(parents=True, exist_ok=False)
    history_path = config.output_dir / "history.jsonl"
    best_path = config.output_dir / "bc_best.pt"
    last_path = config.output_dir / "bc_last.pt"
    best_score = -float("inf")
    best_epoch = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                outputs, _ = model.forward_sequence(
                    batch["obs"], batch["clock_context"]
                )
                loss = _loss(
                    outputs,
                    batch,
                    market_class_weight,
                    route_gate_pos_weight,
                )
            if not torch.isfinite(loss.total):
                raise RuntimeError("non-finite V4 option loss")
            scaler.scale(loss.total).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.gradient_clip,
            )
            if not torch.isfinite(torch.as_tensor(norm)):
                raise RuntimeError("non-finite V4 option gradient")
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.total.detach().cpu()))

        val = evaluate(
            model, val_loader, device,
            market_class_weight=market_class_weight,
            route_gate_pos_weight=route_gate_pos_weight,
        )
        gate = promotion_gate(val)
        score = selection_score(val)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "validation": val,
            "promotion_gate": gate,
            "selection_score": score,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)

        payload = _checkpoint(
            model, optimizer, config, manifest, route_ids, epoch, val
        )
        torch.save(payload, last_path)
        if gate["passed"] and score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save(payload, best_path)

    result = {
        "best_epoch": int(best_epoch),
        "promotable": bool(best_path.is_file()),
        "last_checkpoint": str(last_path),
        "best_checkpoint": (
            str(best_path) if best_path.is_file() else None
        ),
    }
    if best_path.is_file():
        saved = torch.load(
            best_path, map_location=device, weights_only=False
        )
        model.load_state_dict(saved["model_state"], strict=True)
        result["test"] = evaluate(
            model, test_loader, device,
            market_class_weight=market_class_weight,
            route_gate_pos_weight=route_gate_pos_weight,
            route_gate_threshold=float(
                saved.get("route_gate_threshold", 0.5)
            ),
        )
        result["test_gate"] = promotion_gate(result["test"])

    (config.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=pathlib.Path, required=True)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--sequence-len", type=int, default=32)
    parser.add_argument("--batch-sequences", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    result = train(TrainConfig(
        dataset=args.dataset,
        manifest=args.manifest,
        output_dir=args.output_dir,
        sequence_len=args.sequence_len,
        batch_sequences=args.batch_sequences,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        device=args.device,
        use_amp=not args.no_amp,
    ))
    print("FARMOS_V4_OPTION_TRAIN=" + json.dumps(
        result, sort_keys=True
    ))


if __name__ == "__main__":
    main()
