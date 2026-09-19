from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kaggrl.structured_actor_critic import StructuredActorCritic, export_numpy
from kaggrl.structured_ppo import gae, ppo_update


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_episode(path):
    d = np.load(path, allow_pickle=False)
    return {
        "obs": torch.from_numpy(d["obs_f16"].astype(np.float32)).unsqueeze(0),
        "actions": torch.from_numpy(d["action_i16"].astype(np.int64)).unsqueeze(0),
        "masks": torch.from_numpy(d["mask_u8"].astype(np.float32)).unsqueeze(0),
        "old_logp": torch.from_numpy(d["old_logp_f32"].astype(np.float32)).unsqueeze(0),
        "old_value": torch.from_numpy(d["old_value_f32"].astype(np.float32)).unsqueeze(0),
        "reward": torch.from_numpy(d["reward_f32"].astype(np.float32)),
        "done": torch.from_numpy(d["done_u8"].astype(np.float32)),
        "model_sha256": str(d["model_sha256"].item()),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", required=True)
    parser.add_argument("--rollout", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--npz-out", required=True)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--bc-coef", type=float, default=0.05)
    parser.add_argument("--updates", type=int, default=1)
    args = parser.parse_args()

    torch.set_num_threads(4)
    parent_path = Path(args.parent)
    parent_sha_before = sha256(parent_path)
    learner = StructuredActorCritic.from_bc_checkpoint(parent_path)
    anchor = copy.deepcopy(learner).eval()
    for p in anchor.parameters():
        p.requires_grad_(False)
    episode = load_episode(args.rollout)
    optimizer = torch.optim.Adam(learner.parameters(), lr=args.lr)
    returns, advantages = gae(
        episode["reward"], episode["old_value"][0], episode["done"]
    )
    history = []
    for update in range(1, args.updates + 1):
        metrics = ppo_update(
            learner, optimizer,
            episode["obs"], episode["actions"], episode["masks"],
            episode["old_logp"], returns.unsqueeze(0), advantages.unsqueeze(0),
            anchor, bc_coef=args.bc_coef,
        )
        history.append({"update": update, **metrics})
        print(json.dumps(history[-1], sort_keys=True), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": learner.state_dict(),
        "input_dim": learner.input_dim,
        "hidden_dim": learner.hidden_dim,
        "parent": str(parent_path),
        "parent_sha256": parent_sha_before,
        "rollout": str(args.rollout),
        "history": history,
    }, out)
    export_numpy(learner, args.npz_out)
    parent_sha_after = sha256(parent_path)
    if parent_sha_after != parent_sha_before:
        raise RuntimeError("frozen parent checkpoint changed during PPO smoke")
    print(json.dumps({
        "parent_sha256": parent_sha_before,
        "child_sha256": sha256(out),
        "child_npz_sha256": sha256(args.npz_out),
        "updates": args.updates,
    }, sort_keys=True))


if __name__ == "__main__":
    main()