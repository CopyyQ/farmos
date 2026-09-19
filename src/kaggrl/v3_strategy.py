from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable


@dataclass(frozen=True)
class StrategyManifest:
    team_to_slot: dict[int, int]
    slot_to_team: tuple[int, ...]
    sha256: str

    @property
    def size(self) -> int:
        return len(self.slot_to_team)


def build_strategy_manifest(team_ids: Iterable[int]) -> StrategyManifest:
    slots = tuple(sorted({int(value) for value in team_ids}))
    if not slots:
        raise ValueError("strategy manifest requires at least one team id")
    mapping = {team_id: slot for slot, team_id in enumerate(slots)}
    payload = json.dumps({"slot_to_team": list(slots)}, separators=(",", ":"), sort_keys=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return StrategyManifest(mapping, slots, digest)


def strategy_slot_for_team(team_id: int, manifest: StrategyManifest) -> int:
    key = int(team_id)
    if key not in manifest.team_to_slot:
        raise KeyError(f"unknown strategy team id: {key}")
    return int(manifest.team_to_slot[key])
