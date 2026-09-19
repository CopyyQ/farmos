from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SequenceChunk:
    episode_id: int
    indices: list[int]
    length: int


def make_episode_chunks(frame: pd.DataFrame, seq_len: int = 48) -> list[SequenceChunk]:
    chunks: list[SequenceChunk] = []
    for episode_id, group in frame.groupby("episode_id", sort=False):
        ordered = group.sort_values("step")
        indices = ordered.index.tolist()
        for start in range(0, len(indices), int(seq_len)):
            part = indices[start:start + int(seq_len)]
            chunks.append(SequenceChunk(int(episode_id), part, len(part)))
    return chunks


def validation_score(metrics: dict[str, float]) -> float:
    return (
        0.45 * float(metrics["farmer_op_acc"])
        + 0.20 * float(metrics["hand_op_acc"])
        + 0.35 * float(metrics["market_op_acc"])
    )


def training_phase(epoch: int, total_epochs: int) -> str:
    total_epochs = max(1, int(total_epochs))
    warmup_epochs = min(4, max(1, total_epochs // 2))
    return "farmer" if int(epoch) <= warmup_epochs else "joint"
