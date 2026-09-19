from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kaggrl.actions import ActionCodec
from kaggrl.top10_training import make_episode_chunks, training_phase, validation_score
from kaggrl.top_tier_expert import TopTierExpertPolicy, expert_bc_loss

DATASET = ROOT / "data" / "top_tier" / "live" / "full_action_bc.parquet"
OUT_DIR = ROOT / "checkpoints" / "top10_moe_bc"
INPUT_DIM = 1024
HIDDEN = 128
SEQ_LEN = 48
BATCH_SIZE = 8
LR = 2e-3
SEED = 20260917

def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "team"


def decode_team_arrays(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    codec = ActionCodec()
    obs = np.stack([np.frombuffer(x, dtype=np.float16, count=INPUT_DIM) for x in frame.obs_f16])
    actions = np.stack([np.frombuffer(x, dtype=np.int16, count=codec.width) for x in frame.action_i16])
    masks = np.stack([np.frombuffer(x, dtype=np.uint8, count=codec.width) for x in frame.mask_u8])
    return {
        "obs": obs,
        "actions": actions,
        "masks": masks,
        "weights": frame.sample_weight.to_numpy(dtype=np.float32),
    }


def make_batch(arrays: dict[str, np.ndarray], chunks, seq_len=SEQ_LEN):
    batch = len(chunks)
    codec = ActionCodec()
    obs = np.zeros((batch, seq_len, INPUT_DIM), dtype=np.float32)
    actions = np.zeros((batch, seq_len, codec.width), dtype=np.int64)
    masks = np.zeros((batch, seq_len, codec.width), dtype=np.float32)
    weights = np.zeros((batch, seq_len), dtype=np.float32)
    for i, chunk in enumerate(chunks):
        idx = np.asarray(chunk.indices, dtype=np.int64)
        n = len(idx)
        obs[i, :n] = arrays["obs"][idx].astype(np.float32, copy=False)
        actions[i, :n] = arrays["actions"][idx]
        masks[i, :n] = arrays["masks"][idx]
        weights[i, :n] = arrays["weights"][idx]
    return (
        torch.from_numpy(obs),
        torch.from_numpy(actions),
        torch.from_numpy(masks),
        torch.from_numpy(weights),
    )


def _metric_counts(outputs, actions, masks, weights):
    real = weights > 0
    farmer_true = actions[:, :, :3]
    farmer_mask = masks[:, :, :3] > 0
    farmer_pred = torch.stack((
        outputs["farmer_op"].argmax(-1),
        outputs["farmer_item"].argmax(-1),
        outputs["farmer_qty"].argmax(-1),
    ), dim=-1)
    farmer_op_ok = farmer_pred[..., 0].eq(farmer_true[..., 0]) & real
    farmer_full_ok = ((farmer_pred.eq(farmer_true) | ~farmer_mask).all(-1)) & real
    hand_true = actions[:, :, 3:51].view(actions.shape[0], actions.shape[1], 16, 3)
    hand_mask = masks[:, :, 3:51].view(masks.shape[0], masks.shape[1], 16, 3) > 0
    hand_pred_op = outputs["hand_op"].argmax(-1)
    hand_op_active = hand_mask[..., 0] & real.unsqueeze(-1)
    hand_op_ok = hand_pred_op.eq(hand_true[..., 0]) & hand_op_active

    market_true = actions[:, :, 51:].view(actions.shape[0], actions.shape[1], 10, 3)
    market_mask = masks[:, :, 51:].view(masks.shape[0], masks.shape[1], 10, 3) > 0
    market_pred = torch.stack((
        outputs["market_op"].argmax(-1),
        outputs["market_item"].argmax(-1),
        outputs["market_qty"].argmax(-1),
    ), dim=-1)
    market_op_active = market_mask[..., 0] & real.unsqueeze(-1)
    market_op_ok = market_pred[..., 0].eq(market_true[..., 0]) & market_op_active
    market_seq_ok = ((market_pred.eq(market_true) | ~market_mask).all(-1).all(-1)) & real

    return {
        "rows": int(real.sum()),
        "farmer_op_ok": int(farmer_op_ok.sum()),
        "farmer_full_ok": int(farmer_full_ok.sum()),
        "hand_op_ok": int(hand_op_ok.sum()),
        "hand_op_n": int(hand_op_active.sum()),
        "market_op_ok": int(market_op_ok.sum()),
        "market_op_n": int(market_op_active.sum()),
        "market_seq_ok": int(market_seq_ok.sum()),
    }

@torch.inference_mode()
def evaluate_expert(model, frame, arrays, split: str) -> dict[str, float]:
    model.eval()
    chunks = make_episode_chunks(frame[frame.split.eq(split)], seq_len=SEQ_LEN)
    totals = Counter()
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = make_batch(arrays, chunks[start:start + BATCH_SIZE])
        obs, actions, masks, weights = batch
        outputs, _ = model.forward_sequence(obs, teacher_actions=actions)
        totals.update(_metric_counts(outputs, actions, masks, weights))
    rows = max(1, totals["rows"])
    return {
        "rows": int(totals["rows"]),
        "farmer_op_acc": totals["farmer_op_ok"] / rows,
        "farmer_full_acc": totals["farmer_full_ok"] / rows,
        "hand_op_acc": totals["hand_op_ok"] / max(1, totals["hand_op_n"]),
        "market_op_acc": totals["market_op_ok"] / max(1, totals["market_op_n"]),
        "market_sequence_exact": totals["market_seq_ok"] / rows,
    }


def majority_farmer_baseline(frame: pd.DataFrame) -> tuple[int, float]:
    train = frame[frame.split.eq("train")]
    train_ops = [int(np.frombuffer(x, dtype=np.int16, count=1)[0]) for x in train.action_i16]
    majority = Counter(train_ops).most_common(1)[0][0]
    val = frame[frame.split.eq("val")]
    val_ops = [int(np.frombuffer(x, dtype=np.int16, count=1)[0]) for x in val.action_i16]
    acc = sum(x == majority for x in val_ops) / max(1, len(val_ops))
    return int(majority), float(acc)

def train_team(team_name: str, frame: pd.DataFrame, epochs: int) -> dict:
    team = frame[frame.team_name.eq(team_name)].sort_values(["episode_id", "step"]).reset_index(drop=True)
    arrays = decode_team_arrays(team)
    train_chunks = make_episode_chunks(team[team.split.eq("train")], seq_len=SEQ_LEN)
    if not train_chunks:
        raise RuntimeError(f"no training chunks for {team_name}")
    seed = SEED + sum(ord(c) for c in team_name)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = TopTierExpertPolicy(INPUT_DIM, HIDDEN)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    slug = slugify(team_name)
    checkpoint = OUT_DIR / f"{slug}.pt"
    metrics_path = OUT_DIR / f"{slug}.metrics.json"
    majority_op, majority_val = majority_farmer_baseline(team)
    history = []
    best_score = -1.0
    best_epoch = 0

    for epoch in range(1, int(epochs) + 1):
        model.train()
        phase = training_phase(epoch, epochs)
        order = rng.permutation(len(train_chunks))
        losses = []
        for start in range(0, len(order), BATCH_SIZE):
            chunks = [train_chunks[i] for i in order[start:start + BATCH_SIZE]]
            obs, actions, masks, weights = make_batch(arrays, chunks)
            outputs, _ = model.forward_sequence(obs, teacher_actions=actions)
            if phase == "farmer":
                target = actions[:, :, 0]
                active = weights > 0
                per_row = F.cross_entropy(
                    outputs["farmer_op"].reshape(-1, outputs["farmer_op"].shape[-1]),
                    target.reshape(-1),
                    reduction="none",
                ).view_as(target)
                loss = (per_row * active).sum() / active.sum().clamp_min(1)
            else:
                loss, _ = expert_bc_loss(outputs, actions, masks, weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        val = evaluate_expert(model, team, arrays, "val")
        score = validation_score(val)
        row = {
            "epoch": epoch,
            "phase": phase,
            "train_loss": float(np.mean(losses)),
            "val_score": float(score),
            "val": val,
        }
        history.append(row)
        print(json.dumps({"team": team_name, **row}), flush=True)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save({
                "model": model.state_dict(),
                "team_name": team_name,
                "input_dim": INPUT_DIM,
                "hidden_dim": HIDDEN,
                "epoch": epoch,
                "val_score": score,
            }, checkpoint)

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model"])
    val = evaluate_expert(model, team, arrays, "val")
    test = evaluate_expert(model, team, arrays, "test")
    metrics = {
        "team_name": team_name,
        "best_epoch": int(best_epoch),
        "best_val_score": float(best_score),
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "split_rows": team.groupby("split").size().to_dict(),
        "split_episodes": team.groupby("split")["episode_id"].nunique().to_dict(),
        "majority_farmer_op": majority_op,
        "majority_farmer_val_acc": majority_val,
        "history": history,
        "val": val,
        "test": test,
        "gate": {
            "farmer_beats_majority_by_0_10": val["farmer_op_acc"] >= majority_val + 0.10,
            "market_op_ge_0_50": val["market_op_acc"] >= 0.50,
        },
    }
    metrics["gate"]["pass"] = all(metrics["gate"].values())
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teams", default="", help="comma-separated team names; default trains every team")
    parser.add_argument("--epochs", type=int, default=8)
    return parser.parse_args()

def main():
    args = parse_args()
    torch.set_num_threads(4)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(DATASET)
    available = sorted(frame.team_name.unique().tolist())
    if args.teams:
        requested = [x.strip() for x in args.teams.split(",") if x.strip()]
        missing = sorted(set(requested) - set(available))
        if missing:
            raise SystemExit(f"unknown teams: {missing}")
        teams = requested
    else:
        teams = available
    results = []
    for team_name in teams:
        print(json.dumps({"start_team": team_name, "epochs": args.epochs}), flush=True)
        results.append(train_team(team_name, frame, args.epochs))
    print(json.dumps({
        "trained_teams": [x["team_name"] for x in results],
        "gate_pass": sum(bool(x["gate"]["pass"]) for x in results),
        "team_count": len(results),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
