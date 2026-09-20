from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kaggrl.clock import CLOCK_FEATURES, resolve_clock
from kaggrl.v4_objective import MARGIN_SCALE, OBJECTIVE_VERSION
from kaggrl.v4_option_model import V4OptionPolicy

INPUT_DIM = 1024


class CounterfactualDataset(Dataset):
    def __init__(self, frame, route_to_class, market_modes):
        self.frame = frame.reset_index(drop=True)
        self.route_to_class = route_to_class
        self.market_modes = tuple(market_modes)
        self.sequence_len = int(self.frame["sequence_len"].iloc[0])

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        raw = np.frombuffer(row["obs_seq_f16"], dtype=np.float16)
        obs = raw.reshape(self.sequence_len, INPUT_DIM).astype(
            np.float32, copy=False
        )
        target_step = int(row["step"])
        start = target_step - self.sequence_len + 1
        clock = np.asarray([
            resolve_clock(
                {"step": step},
                {"episodeSteps": 720, "turnsPerDay": 24},
            ).features()
            for step in range(start, target_step + 1)
        ], dtype=np.float32)
        family = 0 if row["family"] == "route" else 1
        if family == 0:
            action = self.route_to_class[int(row["action_id"])]
            base = self.route_to_class[int(row["base_route_id"])]
        else:
            action = int(row["action_id"])
            base = self.market_modes.index("KEEP_ROUTE")
        return {
            "row_index": int(index),
            "obs": torch.from_numpy(obs),
            "clock": torch.from_numpy(clock),
            "family": int(family),
            "action": int(action),
            "base": int(base),
            "candidate_margin": float(row["candidate_margin"]),
            "baseline_margin": float(row["baseline_margin"]),
            "advantage": float(row["advantage"]),
            "done": float(row["candidate_done"]),
        }


def _load_model(checkpoint: pathlib.Path):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = V4OptionPolicy(
        int(payload["input_dim"]),
        route_count=len(payload["route_ids"]),
        market_mode_count=len(payload["market_modes"]),
        hidden_dim=int(payload["hidden_dim"]),
        clock_dim=int(payload["clock_dim"]),
    )
    incompatible = model.load_state_dict(
        payload["model_state"], strict=False
    )
    q_prefixes = (
        "route_value_head.", "route_value_clock_head.",
        "market_value_head.", "market_value_clock_head.",
    )
    non_q_missing = [
        name for name in incompatible.missing_keys
        if not name.startswith(q_prefixes)
    ]
    if non_q_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"base checkpoint mismatch missing={non_q_missing} "
            f"unexpected={incompatible.unexpected_keys}"
        )
    if incompatible.missing_keys:
        with torch.no_grad():
            for head in (
                model.route_value_head,
                model.market_value_head,
            ):
                head.weight.copy_(
                    model.value_head.weight.expand_as(head.weight)
                )
                head.bias.copy_(
                    model.value_head.bias.expand_as(head.bias)
                )
            model.route_value_clock_head.weight.copy_(
                model.value_clock_head.weight.expand_as(
                    model.route_value_clock_head.weight
                )
            )
            model.market_value_clock_head.weight.copy_(
                model.value_clock_head.weight.expand_as(
                    model.market_value_clock_head.weight
                )
            )
    return model, payload


def _q_parameters(model):
    prefixes = (
        "route_value_head", "route_value_clock_head",
        "market_value_head", "market_value_clock_head",
    )
    params = []
    for name, param in model.named_parameters():
        trainable = name.startswith(prefixes)
        param.requires_grad_(trainable)
        if trainable:
            params.append(param)
    return params


def _selected_q(outputs, batch):
    last_route = outputs["route_value"][:, -1]
    last_market = outputs["market_value"][:, -1]
    batch_size = last_route.shape[0]
    selected = torch.empty(
        batch_size, device=last_route.device, dtype=last_route.dtype
    )
    base = torch.empty_like(selected)
    route_mask = batch["family"].eq(0)
    market_mask = ~route_mask
    if bool(route_mask.any()):
        selected[route_mask] = last_route[route_mask].gather(
            1, batch["action"][route_mask].unsqueeze(1)
        ).squeeze(1)
        base[route_mask] = last_route[route_mask].gather(
            1, batch["base"][route_mask].unsqueeze(1)
        ).squeeze(1)
    if bool(market_mask.any()):
        selected[market_mask] = last_market[market_mask].gather(
            1, batch["action"][market_mask].unsqueeze(1)
        ).squeeze(1)
        base[market_mask] = last_market[market_mask].gather(
            1, batch["base"][market_mask].unsqueeze(1)
        ).squeeze(1)
    return selected, base


def _to_device(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


@torch.inference_mode()
def evaluate(model, loader, frame, device):
    model.eval()
    predictions = np.empty(len(frame), dtype=np.float32)
    base_predictions = np.empty(len(frame), dtype=np.float32)
    for batch in loader:
        batch = _to_device(batch, device)
        outputs, _ = model.forward_sequence(
            batch["obs"], batch["clock"]
        )
        selected, base = _selected_q(outputs, batch)
        idx = batch["row_index"].detach().cpu().numpy()
        predictions[idx] = (
            selected.detach().cpu().numpy() * MARGIN_SCALE
        )
        base_predictions[idx] = (
            base.detach().cpu().numpy() * MARGIN_SCALE
        )

    work = frame.reset_index(drop=True).copy()
    work["predicted_margin"] = predictions
    work["predicted_base_margin"] = base_predictions
    work["predicted_advantage"] = (
        work["predicted_margin"] - work["predicted_base_margin"]
    )

    candidate = work[
        ~(
            ((work.family == "route")
             & (work.action_id == work.base_route_id))
            | ((work.family == "market") & (work.action_id == 0))
        )
    ]
    if len(candidate) >= 2:
        advantage_corr = float(np.corrcoef(
            candidate["predicted_advantage"],
            candidate["advantage"],
        )[0, 1])
        if not np.isfinite(advantage_corr):
            advantage_corr = 0.0
    else:
        advantage_corr = 0.0
    nonzero = candidate[np.abs(candidate["advantage"]) >= 1.0]
    sign_acc = float(
        np.mean(
            (nonzero["predicted_advantage"] > 0)
            == (nonzero["advantage"] > 0)
        )
    ) if len(nonzero) else 0.0

    selected_improvements = []
    regrets = []
    for _, group in work.groupby(
        ["seed", "seat", "step", "family"], sort=False
    ):
        chosen = group.loc[group["predicted_margin"].idxmax()]
        best = float(group["candidate_margin"].max())
        selected_margin = float(chosen["candidate_margin"])
        baseline = float(group["baseline_margin"].iloc[0])
        selected_improvements.append(selected_margin - baseline)
        regrets.append(best - selected_margin)

    return {
        "rows": int(len(work)),
        "candidate_rows": int(len(candidate)),
        "advantage_corr": advantage_corr,
        "advantage_sign_acc": sign_acc,
        "mean_selected_improvement": float(
            np.mean(selected_improvements)
        ),
        "median_selected_improvement": float(
            np.median(selected_improvements)
        ),
        "mean_regret": float(np.mean(regrets)),
        "zero_regret_rate": float(
            np.mean(np.asarray(regrets) <= 1e-6)
        ),
    }


def train(
    dataset_path: pathlib.Path,
    base_checkpoint: pathlib.Path,
    output_dir: pathlib.Path,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    advantage_weight: float = 2.0,
    device: str = "cpu",
):
    frame = pd.read_parquet(dataset_path)
    model, payload = _load_model(base_checkpoint)
    route_ids = tuple(int(x) for x in payload["route_ids"])
    market_modes = tuple(str(x) for x in payload["market_modes"])
    route_to_class = {
        route_id: index for index, route_id in enumerate(route_ids)
    }

    train_frame = frame[frame.split.eq("train")].reset_index(drop=True)
    val_frame = frame[frame.split.eq("val")].reset_index(drop=True)
    test_frame = frame[frame.split.eq("test")].reset_index(drop=True)
    if train_frame.empty or val_frame.empty:
        raise RuntimeError("counterfactual Q needs train and val splits")

    train_ds = CounterfactualDataset(
        train_frame, route_to_class, market_modes
    )
    val_ds = CounterfactualDataset(
        val_frame, route_to_class, market_modes
    )
    test_ds = CounterfactualDataset(
        test_frame, route_to_class, market_modes
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )

    device_obj = torch.device(device)
    model = model.to(device_obj)
    optimizer = torch.optim.AdamW(
        _q_parameters(model),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    best_score = -float("inf")
    best_epoch = 0
    best_path = output_dir / "q_best.pt"
    history = output_dir / "history.jsonl"

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            batch = _to_device(batch, device_obj)
            optimizer.zero_grad(set_to_none=True)
            outputs, _ = model.forward_sequence(
                batch["obs"], batch["clock"]
            )
            selected, base = _selected_q(outputs, batch)
            target_q = batch["candidate_margin"].float() / MARGIN_SCALE
            target_adv = batch["advantage"].float() / MARGIN_SCALE
            value_loss = F.smooth_l1_loss(selected, target_q)
            advantage_loss = F.smooth_l1_loss(
                selected - base, target_adv
            )
            loss = value_loss + float(advantage_weight) * advantage_loss
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite counterfactual Q loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        metrics = evaluate(model, val_loader, val_frame, device_obj)
        score = (
            metrics["advantage_corr"]
            + 0.5 * metrics["advantage_sign_acc"]
            + 0.0001 * metrics["mean_selected_improvement"]
            - 0.00005 * metrics["mean_regret"]
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation": metrics,
            "selection_score": float(score),
        }
        with history.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        if epoch == 1 or epoch % 10 == 0 or score > best_score:
            print(json.dumps(row, sort_keys=True), flush=True)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            checkpoint = dict(payload)
            checkpoint.update({
                "model_state": model.state_dict(),
                "objective_version": OBJECTIVE_VERSION,
                "margin_scale": MARGIN_SCALE,
                "counterfactual_q_schema": "farmos_v4_counterfactual_q_v1",
                "counterfactual_q_epoch": epoch,
                "counterfactual_q_validation": metrics,
                "counterfactual_q_steps": sorted(
                    int(value) for value in frame["step"].unique().tolist()
                ),
                "counterfactual_q_families": sorted(
                    str(value) for value in frame["family"].unique().tolist()
                ),
            })
            torch.save(checkpoint, best_path)

    saved = torch.load(best_path, map_location=device_obj, weights_only=False)
    model.load_state_dict(saved["model_state"], strict=True)
    test_metrics = (
        evaluate(model, test_loader, test_frame, device_obj)
        if not test_frame.empty else None
    )
    result = {
        "best_epoch": int(best_epoch),
        "best_checkpoint": str(best_path),
        "validation": saved["counterfactual_q_validation"],
        "test": test_metrics,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "FARMOS_V4_COUNTERFACTUAL_Q_TRAIN="
        + json.dumps(result, sort_keys=True),
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=pathlib.Path, required=True)
    parser.add_argument("--base-checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--advantage-weight", type=float, default=2.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    train(
        args.dataset,
        args.base_checkpoint,
        args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        advantage_weight=args.advantage_weight,
        device=args.device,
    )


if __name__ == "__main__":
    main()
