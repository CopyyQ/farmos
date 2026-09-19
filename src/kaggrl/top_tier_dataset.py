from __future__ import annotations

import copy


def _agents_info(replay: dict) -> list[dict]:
    return list((replay.get("info") or {}).get("Agents") or [])


def _source_seats(replay: dict, sources: list[dict]) -> list[tuple[int, dict]]:
    agents = _agents_info(replay)
    out = []
    for source in sources:
        submission_id = int(source["submission_id"])
        team_name = str(source.get("team_name") or "")
        for seat, agent in enumerate(agents):
            agent_submission = int(agent.get("submission_id") or -1)
            agent_name = str(agent.get("Name") or agent.get("name") or "")
            if agent_submission == submission_id or (agent_submission < 0 and team_name and agent_name == team_name):
                out.append((seat, source))
                break
    return out


def normalize_observation(observation: dict, *, seat: int, step: int) -> dict:
    obs = copy.deepcopy(observation or {})
    obs["player"] = int(obs.get("player", seat))
    obs["step"] = int(step)
    obs.setdefault("day", step // 24)
    obs.setdefault("hour", step % 24)
    return obs


def extract_expert_samples(replay: dict, episode_id: int, sources: list[dict]) -> list[dict]:
    steps = list(replay.get("steps") or [])
    rows = []
    for seat, source in _source_seats(replay, sources):
        for step in range(max(0, len(steps) - 1)):
            if seat >= len(steps[step]) or seat >= len(steps[step + 1]):
                continue
            obs = normalize_observation(
                steps[step][seat].get("observation") or {}, seat=seat, step=step
            )
            action = copy.deepcopy(steps[step + 1][seat].get("action") or {})
            rows.append({
                "episode_id": int(episode_id),
                "seat": seat,
                "step": step,
                "observation": obs,
                "action": action,
                "source": copy.deepcopy(source),
            })
    return rows
