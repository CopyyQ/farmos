from __future__ import annotations

import json
from dataclasses import dataclass

from .residual_actions import KEEP, MarketEdit


@dataclass(frozen=True)
class ResidualLabel:
    market0: MarketEdit
    market1: MarketEdit
    route_override: int | None = None
    changed: bool = False
    representable: bool = True


def participant_seat(participants_json: str, team_name: str) -> int:
    participants = json.loads(participants_json)
    hits = [i for i, name in enumerate(participants) if name == team_name]
    if len(hits) != 1:
        raise ValueError(f"expected exactly one participant match, got {len(hits)}")
    return hits[0]


def _effective(action: dict) -> list[list]:
    return [list(order) for order in action.get("market", []) if order]


def derive_market_residual(base_action: dict, teacher_action: dict) -> ResidualLabel:
    base = _effective(base_action)
    teacher = _effective(teacher_action)
    edits = []
    for idx in (0, 1):
        b = base[idx] if idx < len(base) else None
        t = teacher[idx] if idx < len(teacher) else None
        if b == t:
            edits.append(KEEP)
        elif t is None:
            edits.append(MarketEdit("DROP"))
        else:
            edits.append(MarketEdit("REPLACE", t))
    changed = any(edit.kind != "KEEP" for edit in edits)
    label = ResidualLabel(edits[0], edits[1], None, changed, True)
    from .residual_actions import apply_residual
    reconstructed = _effective(apply_residual(base_action, label))[:2]
    target = teacher[:2]
    if reconstructed != target:
        label = ResidualLabel(edits[0], edits[1], None, changed, False)
    return label


def _stable_bucket(text: str, modulus: int) -> int:
    import hashlib
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulus


def episode_split(episode_id: int) -> str:
    bucket = _stable_bucket(str(int(episode_id)), 10)
    if bucket == 0:
        return "test"
    if bucket == 1:
        return "val"
    return "train"


def sample_residual_row(episode_id: int, team_name: str, step: int, changed: bool) -> bool:
    modulus = 16 if changed else 32
    key = f"{int(episode_id)}|{team_name}|{int(step)}|{int(changed)}"
    return _stable_bucket(key, modulus) == 0
