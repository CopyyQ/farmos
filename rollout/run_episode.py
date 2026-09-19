from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kaggle_environments import make
from kaggrl.rollout import EpisodeBuffer, transition_reward
from kaggrl.rollout_agent import NumpyRolloutAgent

V17 = ROOT.parent / "submission_ready_20260916_v17_27w_fixed" / "main.py"
V45 = ROOT.parent / "source_public" / "extracted_v45" / "main.py"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def resolve_opponent(value, seed):
    if value == "v17":
        return str(V17), "v17"
    if value == "v45":
        return str(V45), "v45"
    if value == "starter":
        return "starter", "starter"
    path = Path(value)
    if path.suffix == ".npz":
        agent = NumpyRolloutAgent(path, seed=seed + 7001, deterministic=True)
        return agent, f"expert:{path.name}"
    return str(path), str(path)

def run(model_path, opponent, seat, seed, out_path, deterministic=False):
    model_path = Path(model_path)
    learner = NumpyRolloutAgent(model_path, seed=seed + 101, deterministic=deterministic)
    opponent_agent, opponent_name = resolve_opponent(opponent, seed)
    env = make("kaggriculture", configuration={"seed": int(seed), "episodeSteps": 720}, debug=False)
    agents = [learner, opponent_agent] if int(seat) == 0 else [opponent_agent, learner]
    env.run(agents)
    if len(env.steps) != 720:
        raise RuntimeError(f"expected 720 steps, got {len(env.steps)}")
    final = env.steps[-1]
    statuses = [str(s.status) for s in final]
    rewards = [float(s.reward) for s in final]
    if any(x != "DONE" for x in statuses):
        raise RuntimeError(f"non-DONE final status: {statuses}")
    if len(learner.records) != 719:
        raise RuntimeError(f"expected 719 learner decisions, got {len(learner.records)}")
    other = 1 - int(seat)
    terminal = 1.0 if rewards[seat] > rewards[other] else -1.0 if rewards[seat] < rewards[other] else 0.0
    buf = EpisodeBuffer(sha256(model_path), opponent_name, int(seat), int(seed))
    non_idle = 0
    for i, rec in enumerate(learner.records):
        step = int(rec["step"])
        if step != i:
            raise RuntimeError(f"record alignment mismatch index={i} step={step}")
        obs_now = env.steps[step][seat].observation
        obs_next = env.steps[step + 1][seat].observation
        is_last = i == len(learner.records) - 1
        reward = transition_reward(obs_now, obs_next, terminal if is_last else 0.0)
        tokens = rec["action"]
        start = 17 * 3
        unit_ops = tokens[:start:3]
        market_ops = tokens[start::3]
        non_idle += int((unit_ops != 0).any() or (market_ops != 0).any())
        buf.append(
            step=step, obs=rec["obs"], action=tokens, mask=rec["mask"],
            old_logp=rec["logp"], old_value=rec["value"], reward=reward, done=is_last,
        )
    buf.save_npz(out_path)
    return {
        "seed": int(seed), "seat": int(seat), "opponent": opponent_name,
        "steps": len(env.steps), "decisions": len(learner.records),
        "statuses": statuses, "rewards": rewards, "terminal_result": terminal,
        "non_idle_decisions": int(non_idle), "model_sha256": sha256(model_path),
        "out": str(out_path),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--opponent", required=True)
    parser.add_argument("--seat", type=int, choices=(0, 1), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()
    result = run(args.model, args.opponent, args.seat, args.seed, args.out, args.deterministic)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()