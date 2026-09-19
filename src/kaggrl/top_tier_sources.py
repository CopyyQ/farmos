from __future__ import annotations

import math
from datetime import datetime


LATEST_QUALITY_FLOOR = 0.95


def _score(row: dict) -> float:
    return float(row.get("publicScore") or 0.0)


def _date(row: dict) -> datetime:
    text = str(row.get("dateSubmitted") or "1970-01-01T00:00:00Z")
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def choose_submission_roles(leaderboard_score: str | float, submissions: list[dict]) -> dict:
    if not submissions:
        return {"active_best": None, "latest_qualified": None}
    target = float(leaderboard_score)
    active = min(submissions, key=lambda row: (abs(_score(row) - target), -_date(row).timestamp()))
    latest = max(submissions, key=_date)
    ratio = _score(latest) / max(_score(active), 1.0)
    qualified = latest if latest.get("id") != active.get("id") and ratio >= LATEST_QUALITY_FLOOR else None
    return {"active_best": active, "latest_qualified": qualified}


def sample_weight(*, rank: int, age_days: float, role: str, score_ratio: float) -> float:
    rank_factor = math.exp(-0.06 * max(int(rank) - 1, 0))
    recency_factor = 2.0 ** (-max(float(age_days), 0.0) / 3.0)
    role_factor = 1.0 if role == "active_best" else 0.85
    quality_factor = min(max(float(score_ratio), 0.5), 1.0)
    return rank_factor * recency_factor * role_factor * quality_factor


def eligible_recent_episodes(rows: list[dict], *, limit: int) -> list[dict]:
    eligible = [
        row for row in rows
        if row.get("state") == "COMPLETED" and row.get("type") == "EPISODE_TYPE_PUBLIC"
    ]
    eligible.sort(key=lambda row: str(row.get("createTime") or ""), reverse=True)
    return eligible[: max(int(limit), 0)]
