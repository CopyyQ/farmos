from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone

LATEST_QUALITY_FLOOR = 0.95
LEADERBOARD_TIME_TOLERANCE = timedelta(milliseconds=1)


class ManifestResolutionError(RuntimeError):
    pass


def _field(row, *names, default=None):
    for name in names:
        if isinstance(row, dict) and name in row:
            return row[name]
        if not isinstance(row, dict) and hasattr(row, name):
            return getattr(row, name)
    return default


def normalize_time(value) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value or "").strip()
        if not text:
            raise ManifestResolutionError("missing submission timestamp")
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def resolve_active_submission(leaderboard_row, submissions):
    team_id = int(_field(leaderboard_row, "teamId", "team_id"))
    target = normalize_time(_field(leaderboard_row, "submissionDate", "submission_date"))
    matches = []
    for row in submissions:
        row_team = _field(row, "teamId", "team_id")
        if row_team is not None and int(row_team) != team_id:
            continue
        submitted = normalize_time(_field(row, "dateSubmitted", "date_submitted"))
        if abs(submitted - target) <= LEADERBOARD_TIME_TOLERANCE:
            matches.append(row)
    if len(matches) != 1:
        raise ManifestResolutionError(
            f"team={team_id} date={target.isoformat()} matches={len(matches)}"
        )
    return matches[0]


def _score(row):
    value = _field(row, "publicScore", "public_score")
    if value in (None, ""):
        return None
    return float(value)


def select_latest_candidate(active_submission, submissions):
    active_time = normalize_time(_field(active_submission, "dateSubmitted", "date_submitted"))
    active_score = _score(active_submission)
    if active_score is None:
        return None
    threshold = LATEST_QUALITY_FLOOR * active_score
    candidates = []
    for row in submissions:
        score = _score(row)
        if score is None or score < threshold:
            continue
        submitted = normalize_time(_field(row, "dateSubmitted", "date_submitted"))
        if submitted <= active_time:
            continue
        candidates.append((submitted, row))
    if not candidates:
        return None
    candidates.sort(key=lambda pair: pair[0])
    return candidates[-1][1]


def _submission_record(row):
    if row is None:
        return None
    return {
        "id": int(_field(row, "id")),
        "dateSubmitted": normalize_time(_field(row, "dateSubmitted", "date_submitted")).isoformat(),
        "publicScore": str(_field(row, "publicScore", "public_score", default="")),
    }


def _leaderboard_record(row, rank):
    return {
        "rank": int(rank),
        "team_id": int(_field(row, "teamId", "team_id")),
        "team_name": str(_field(row, "teamName", "team_name", default="")),
        "submissionDate": normalize_time(_field(row, "submissionDate", "submission_date")).isoformat(),
        "leaderboard_score": float(_field(row, "score", default=0.0) or 0.0),
    }


def build_snapshot(api, competition: str, top_k: int = 10) -> dict:
    leaderboard = list(api.competition_leaderboard_view(competition, page_size=max(20, int(top_k))) or [])
    frozen = [_leaderboard_record(row, rank + 1) for rank, row in enumerate(leaderboard[: int(top_k)])]
    payload = json.dumps(frozen, sort_keys=True, separators=(",", ":")).encode("utf-8")
    teams = []
    for raw_row, team in zip(leaderboard[: int(top_k)], frozen):
        submissions = list(api.competition_team_submissions(team["team_id"]) or [])
        active = resolve_active_submission(raw_row, submissions)
        latest = select_latest_candidate(active, submissions)
        teams.append({
            "rank": team["rank"],
            "team_id": team["team_id"],
            "team_name": team["team_name"],
            "leaderboard_score": team["leaderboard_score"],
            "submissionDate": team["submissionDate"],
            "roles": {
                "active_best": _submission_record(active),
                "latest_candidate": _submission_record(latest),
            },
        })
    return {
        "snapshot_time_utc": datetime.now(timezone.utc).isoformat(),
        "competition": str(competition),
        "top_n": int(top_k),
        "leaderboard_sha256": hashlib.sha256(payload).hexdigest(),
        "teams": teams,
    }


def _enum_text(value) -> str:
    if value is None:
        return ""
    name = getattr(value, "name", None)
    if name:
        return str(name)
    text = str(value)
    return text.split(".")[-1]


def _agent_record(agent):
    return {
        "submissionId": int(_field(agent, "submissionId", "submission_id", default=0) or 0),
        "index": int(_field(agent, "index", default=0) or 0),
        "reward": float(_field(agent, "reward", default=0.0) or 0.0),
        "teamName": str(_field(agent, "teamName", "team_name", default="")),
        "teamId": int(_field(agent, "teamId", "team_id", default=0) or 0),
    }


def _episode_record(episode):
    create_time = _field(episode, "createTime", "create_time")
    end_time = _field(episode, "endTime", "end_time")
    return {
        "id": int(_field(episode, "id")),
        "createTime": normalize_time(create_time).isoformat(),
        "endTime": normalize_time(end_time).isoformat() if end_time else "",
        "state": _enum_text(_field(episode, "state")),
        "type": _enum_text(_field(episode, "type")),
        "agents": [_agent_record(a) for a in list(_field(episode, "agents", default=[]) or [])],
    }


def _eligible_episode_records(episodes):
    records = [_episode_record(ep) for ep in episodes]
    records = [
        rec for rec in records
        if rec["state"] == "COMPLETED" and rec["type"] == "EPISODE_TYPE_PUBLIC"
    ]
    records.sort(key=lambda rec: rec["createTime"], reverse=True)
    return records


def collect_episode_sources(api, snapshot: dict, active_limit: int = 12, candidate_limit: int = 3) -> dict:
    by_episode = {}
    for team in snapshot.get("teams", []):
        for role, limit in (("active_best", active_limit), ("latest_candidate", candidate_limit)):
            submission = (team.get("roles") or {}).get(role)
            if not submission or int(limit) <= 0:
                continue
            submission_id = int(submission["id"])
            episodes = _eligible_episode_records(api.competition_list_episodes(submission_id))[: int(limit)]
            for episode in episodes:
                matching = [a for a in episode["agents"] if int(a["submissionId"]) == submission_id]
                if len(matching) != 1:
                    raise ManifestResolutionError(
                        f"episode={episode['id']} submission={submission_id} agent_matches={len(matching)}"
                    )
                agent = matching[0]
                entry = by_episode.setdefault(episode["id"], {"episode": episode, "sources": []})
                source = {
                    "rank": int(team["rank"]),
                    "team_id": int(team["team_id"]),
                    "team_name": str(team["team_name"]),
                    "submission_id": submission_id,
                    "submission_date": str(submission["dateSubmitted"]),
                    "submission_score": float(submission["publicScore"] or 0.0),
                    "leaderboard_score": float(team["leaderboard_score"]),
                    "role": role,
                    "seat": int(agent["index"]),
                    "reward": float(agent["reward"]),
                }
                if not any(s["submission_id"] == submission_id and s["role"] == role for s in entry["sources"]):
                    entry["sources"].append(source)
    out = dict(snapshot)
    out["episodes"] = sorted(by_episode.values(), key=lambda x: x["episode"]["createTime"], reverse=True)
    return out
