from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def potential(observation) -> float:
    farms = list(_get(observation, "farms", []) or [])
    if len(farms) < 2:
        return 0.0
    player = int(_get(observation, "player", 0) or 0)
    player = max(0, min(player, len(farms) - 1))
    rival = 1 - player if len(farms) == 2 else next(i for i in range(len(farms)) if i != player)
    own = float(_get(farms[player], "money", 0) or 0)
    other = float(_get(farms[rival], "money", 0) or 0)
    return float(np.tanh((own - other) / 20000.0))


def transition_reward(obs, next_obs, terminal_result: float = 0.0, gamma: float = 0.995) -> float:
    shaped = float(gamma) * potential(next_obs) - potential(obs)
    return float(terminal_result) + shaped

@dataclass
class EpisodeBuffer:
    model_sha256: str
    opponent: str
    seat: int
    seed: int
    rows: list[dict] = field(default_factory=list)

    def append(self, *, step, obs, action, mask, old_logp, old_value, reward, done):
        step = int(step)
        if self.rows and step <= self.rows[-1]["step"]:
            raise ValueError("episode steps must be strictly increasing")
        self.rows.append({
            "step": step,
            "obs": np.asarray(obs, dtype=np.float32).copy(),
            "action": np.asarray(action, dtype=np.int16).copy(),
            "mask": np.asarray(mask, dtype=np.uint8).copy(),
            "old_logp": float(old_logp),
            "old_value": float(old_value),
            "reward": float(reward),
            "done": bool(done),
        })

    def save_npz(self, path):
        if not self.rows:
            raise ValueError("cannot save empty episode")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            obs_f16=np.asarray([r["obs"] for r in self.rows], dtype=np.float16),
            action_i16=np.asarray([r["action"] for r in self.rows], dtype=np.int16),
            mask_u8=np.asarray([r["mask"] for r in self.rows], dtype=np.uint8),
            old_logp_f32=np.asarray([r["old_logp"] for r in self.rows], dtype=np.float32),
            old_value_f32=np.asarray([r["old_value"] for r in self.rows], dtype=np.float32),
            reward_f32=np.asarray([r["reward"] for r in self.rows], dtype=np.float32),
            done_u8=np.asarray([r["done"] for r in self.rows], dtype=np.uint8),
            step_i16=np.asarray([r["step"] for r in self.rows], dtype=np.int16),
            model_sha256=np.asarray(self.model_sha256),
            opponent=np.asarray(self.opponent),
            seat=np.asarray(self.seat, dtype=np.int8),
            seed=np.asarray(self.seed, dtype=np.int32),
        )