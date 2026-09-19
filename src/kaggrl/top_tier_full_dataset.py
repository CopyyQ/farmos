from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Iterator

import numpy as np

from .top_tier_dataset import extract_expert_samples


def compute_source_weight(rank: int, role: str, age_hours: float) -> float:
    rank_weight = 1.25 - 0.025 * (int(rank) - 1)
    role_weight = 1.15 if role == "latest_qualified" else 1.0
    recency_weight = 2.0 ** (-max(0.0, float(age_hours)) / 48.0)
    return float(rank_weight * role_weight * recency_weight)


def _time_key(value: str) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def assign_temporal_splits(records: list[dict]) -> dict[tuple[str, int], str]:
    grouped: dict[str, dict[int, str]] = defaultdict(dict)
    for row in records:
        grouped[str(row["team_name"])][int(row["episode_id"])] = str(row["create_time"])
    out: dict[tuple[str, int], str] = {}
    for team_name, episodes in grouped.items():
        ordered = sorted(episodes.items(), key=lambda kv: _time_key(kv[1]))
        n = len(ordered)
        if n >= 5:
            n_val, n_test = 2, 2
        elif n >= 3:
            n_val, n_test = 1, 1
        elif n == 2:
            n_val, n_test = 0, 1
        else:
            n_val, n_test = 0, 0
        test_start = n - n_test
        val_start = test_start - n_val
        for idx, (episode_id, _) in enumerate(ordered):
            split = "test" if idx >= test_start else "val" if idx >= val_start else "train"
            out[(team_name, int(episode_id))] = split
    return out


def _hand_count(observation: dict) -> int:
    player = int(observation.get("player", 0) or 0)
    farms = list(observation.get("farms") or [])
    if 0 <= player < len(farms):
        return len((farms[player] or {}).get("hands") or [])
    return 0

def iter_actor_examples(
    replay: dict,
    episode_meta: dict,
    source: dict,
    encoder,
    codec,
) -> Iterator[dict]:
    episode_id = int(episode_meta["id"])
    rows = extract_expert_samples(replay, episode_id, [source])
    for row in rows:
        obs = row["observation"]
        action = row["action"]
        hand_count = _hand_count(obs)
        obs_arr = np.asarray(encoder.encode(obs), dtype=np.float16)
        action_arr = np.asarray(codec.encode(action, hand_count), dtype=np.int16)
        mask_arr = np.asarray(codec.supervision_mask(action, hand_count) > 0, dtype=np.uint8)
        yield {
            "episode_id": episode_id,
            "team_name": str(source.get("team_name", "")),
            "rank": int(source.get("rank", 0) or 0),
            "role": str(source.get("role", "active_best")),
            "create_time": str(episode_meta.get("createTime", "")),
            "seat": int(row["seat"]),
            "step": int(row["step"]),
            "obs_f16": obs_arr.tobytes(),
            "action_i16": action_arr.tobytes(),
            "mask_u8": mask_arr.tobytes(),
        }
