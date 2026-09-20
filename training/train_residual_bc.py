from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kaggrl.clock import CLOCK_FEATURES
from kaggrl.residual_model import ResidualPolicy, residual_bc_loss

DATASET = ROOT / "data" / "top_tier" / "residual_bc_2026-09-12_14.parquet"
MANIFEST = ROOT / "data" / "top_tier" / "manifests" / "residual_bc_2026-09-12_14.json"
OBSERVATION_SCHEMA = "macro_semantic_v4_clock_v2"
TARGET_SCHEMA = "legacy_market2_edit_v1"
OUT = ROOT / "checkpoints" / "residual_bc_top10_2026-09-12_14.pt"
VOCAB_OUT = ROOT / "checkpoints" / "residual_bc_top10_order_vocab.json"
METRICS_OUT = ROOT / "checkpoints" / "residual_bc_top10_metrics.json"
INPUT_DIM = 1024
HIDDEN = 128
BATCH = 256
EPOCHS = 8
LR = 2e-3
SEED = 20260917
UNK = "<UNK>"


def validate_dataset_schema():
    if not MANIFEST.is_file():
        raise RuntimeError(
            "residual dataset manifest is missing; rebuild the V4 dataset"
        )
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("observation_schema") != OBSERVATION_SCHEMA:
        raise RuntimeError(
            "stale residual observation schema; rebuild with "
            "training/build_residual_dataset.py"
        )
    if manifest.get("clock_features") != list(CLOCK_FEATURES):
        raise RuntimeError(
            "residual clock feature schema mismatch; rebuild the dataset"
        )
    if manifest.get("target_schema") != TARGET_SCHEMA:
        raise RuntimeError(
            "residual target schema mismatch; this trainer is only for the "
            "legacy first-two-market-order diagnostic, not V4 Options"
        )


def build_vocab(frame):
    values = set()
    train = frame[frame.split.eq("train")]
    for column in ("order0", "order1"):
        values.update(x for x in train[column].tolist() if x)
    return [UNK] + sorted(values)


def decode_features(frame):
    x = np.empty((len(frame), INPUT_DIM), dtype=np.float16)
    for i, raw in enumerate(frame.obs_f16.tolist()):
        x[i] = np.frombuffer(raw, dtype=np.float16, count=INPUT_DIM)
    return x


def encode_orders(frame, vocab):
    ids = {value: i for i, value in enumerate(vocab)}
    out = {}
    for column in ("order0", "order1"):
        out[column] = np.asarray([ids.get(value, 0) if value else 0 for value in frame[column]], dtype=np.int64)
    return out


def balanced_indices(frame, rng):
    train = np.flatnonzero(frame.split.to_numpy() == "train")
    changed = train[frame.changed.to_numpy()[train]]
    keep = train[~frame.changed.to_numpy()[train]]
    half = BATCH // 2
    steps = max(1, int(np.ceil(max(len(changed), len(keep)) / half)))
    for _ in range(steps):
        left = rng.choice(changed, size=half, replace=len(changed) < half)
        right = rng.choice(keep, size=BATCH - half, replace=len(keep) < BATCH - half)
        idx = np.concatenate([left, right])
        rng.shuffle(idx)
        yield idx


def batch_tensors(x, frame, order_ids, idx):
    obs = torch.from_numpy(x[idx].astype(np.float32, copy=False)).unsqueeze(1)
    labels = {
        "edit0": torch.from_numpy(frame.edit0.to_numpy(dtype=np.int64)[idx]),
        "edit1": torch.from_numpy(frame.edit1.to_numpy(dtype=np.int64)[idx]),
        "order0": torch.from_numpy(order_ids["order0"][idx]),
        "order1": torch.from_numpy(order_ids["order1"][idx]),
    }
    return obs, labels


@torch.inference_mode()
def evaluate(model, x, frame, order_ids, split):
    idx = np.flatnonzero(frame.split.to_numpy() == split)
    totals = {"n": 0, "edit0": 0, "edit1": 0, "exact": 0, "changed_tp": 0, "changed_total": 0, "pred_changed": 0, "replace_known": 0, "replace_total": 0}
    baseline_exact = 0
    for start in range(0, len(idx), 512):
        part = idx[start:start + 512]
        obs, labels = batch_tensors(x, frame, order_ids, part)
        out, _ = model.forward_sequence(obs)
        pred0 = out["edit0"][:, 0].argmax(-1)
        pred1 = out["edit1"][:, 0].argmax(-1)
        po0 = out["order0"][:, 0].argmax(-1)
        po1 = out["order1"][:, 0].argmax(-1)
        e0, e1 = labels["edit0"], labels["edit1"]
        o0, o1 = labels["order0"], labels["order1"]
        exact = pred0.eq(e0) & pred1.eq(e1)
        exact &= (~e0.eq(2)) | po0.eq(o0)
        exact &= (~e1.eq(2)) | po1.eq(o1)
        true_changed = e0.ne(0) | e1.ne(0)
        pred_changed = pred0.ne(0) | pred1.ne(0)
        known0 = e0.ne(2) | o0.ne(0)
        known1 = e1.ne(2) | o1.ne(0)
        exact &= known0 & known1
        replace_mask = e0.eq(2) | e1.eq(2)
        replace_known = ((~e0.eq(2)) | o0.ne(0)) & ((~e1.eq(2)) | o1.ne(0))
        totals["n"] += len(part)
        totals["edit0"] += int(pred0.eq(e0).sum())
        totals["edit1"] += int(pred1.eq(e1).sum())
        totals["exact"] += int(exact.sum())
        totals["changed_tp"] += int((pred_changed & true_changed).sum())
        totals["changed_total"] += int(true_changed.sum())
        totals["pred_changed"] += int(pred_changed.sum())
        totals["replace_total"] += int(replace_mask.sum())
        totals["replace_known"] += int((replace_mask & replace_known).sum())
        baseline_exact += int((e0.eq(0) & e1.eq(0)).sum())
    n = max(1, totals["n"])
    return {
        "rows": totals["n"],
        "edit0_acc": totals["edit0"] / n,
        "edit1_acc": totals["edit1"] / n,
        "row_exact_acc": totals["exact"] / n,
        "keep_baseline_exact": baseline_exact / n,
        "changed_recall": totals["changed_tp"] / max(1, totals["changed_total"]),
        "pred_changed_rate": totals["pred_changed"] / n,
        "replace_vocab_coverage": totals["replace_known"] / max(1, totals["replace_total"]),
    }


def main():
    validate_dataset_schema()
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    frame = pd.read_parquet(DATASET)
    vocab = build_vocab(frame)
    x = decode_features(frame)
    order_ids = encode_orders(frame, vocab)
    model = ResidualPolicy(INPUT_DIM, HIDDEN, len(vocab))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    VOCAB_OUT.write_text(json.dumps(vocab, ensure_ascii=False))
    history = []
    best_val = -1.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for idx in balanced_indices(frame, rng):
            obs, labels = batch_tensors(x, frame, order_ids, idx)
            out, _ = model.forward_sequence(obs)
            loss, _ = residual_bc_loss(out, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        val = evaluate(model, x, frame, order_ids, "val")
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val": val}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val["row_exact_acc"] > best_val:
            best_val = val["row_exact_acc"]
            torch.save({
                "model": model.state_dict(),
                "input_dim": INPUT_DIM,
                "hidden_dim": HIDDEN,
                "order_vocab_size": len(vocab),
                "epoch": epoch,
                "observation_schema": OBSERVATION_SCHEMA,
                "clock_features": list(CLOCK_FEATURES),
                "target_schema": TARGET_SCHEMA,
            }, OUT)
    saved = torch.load(OUT, map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model"])
    model.eval()
    metrics = {
        "best_epoch": int(saved["epoch"]),
        "parameters": sum(p.numel() for p in model.parameters()),
        "order_vocab_size": len(vocab),
        "rows": len(frame),
        "split_rows": frame.groupby("split").size().to_dict(),
        "history": history,
        "val": evaluate(model, x, frame, order_ids, "val"),
        "test": evaluate(model, x, frame, order_ids, "test"),
    }
    metrics["gate"] = {
        "changed_recall_ge_0_60": metrics["val"]["changed_recall"] >= 0.60,
        "exact_gain_ge_0_10": metrics["val"]["row_exact_acc"] - metrics["val"]["keep_baseline_exact"] >= 0.10,
    }
    METRICS_OUT.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(json.dumps(metrics["gate"], sort_keys=True))
    print("checkpoint", OUT)


if __name__ == "__main__":
    main()
