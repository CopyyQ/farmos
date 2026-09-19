from __future__ import annotations

import json
import zlib
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from typing import Any, Iterator

from kaggrl.v2_action_schema import (
    parse_raw_action,
    raw_equal,
    semantic_equivalent,
    to_engine_action,
)
from kaggrl.v2_effects import derive_effects
from kaggrl.v2_observation import normalize_observation


def _json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def encode_zlib_json(value: Any) -> bytes:
    return zlib.compress(_json_text(value).encode("utf-8"), level=6)


def decode_zlib_json(payload: bytes) -> Any:
    return json.loads(zlib.decompress(payload).decode("utf-8"))


def _time_key(value: str) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _joint_dict(joint) -> dict[str, Any]:
    def unit(command):
        return {
            "op": command.op,
            "item": command.item,
            "quantity": command.quantity,
            "raw": list(command.raw),
        }

    def slot(value):
        return {
            "kind": value.kind,
            "op": value.op,
            "item": value.item,
            "quantity": value.quantity,
            "raw": list(value.raw),
        }

    return {
        "farmer": unit(joint.farmer),
        "hands": [unit(x) for x in joint.hands],
        "market": [slot(x) for x in joint.market],
    }


def _terminal_values(final_obs: dict[str, Any]) -> tuple[int, int, int, int]:
    player = int(final_obs.get("player", 0))
    farms = final_obs.get("farms") or []
    if not isinstance(farms, list) or player not in (0, 1) or len(farms) < 2:
        raise ValueError(f"invalid terminal farms/player: player={player}")
    own = int((farms[player] or {}).get("money", 0) or 0)
    rival = int((farms[1 - player] or {}).get("money", 0) or 0)
    margin = own - rival
    result = 1 if margin > 0 else -1 if margin < 0 else 0
    return own, rival, margin, result


def iter_transitions(replay: dict, episode_meta: dict, source: dict) -> Iterator[dict[str, Any]]:
    if "seat" not in source:
        raise ValueError("source seat is required; actor mapping must be unambiguous")
    seat = int(source["seat"])
    steps = list(replay.get("steps") or [])
    if len(steps) < 2:
        return
    if any(seat < 0 or seat >= len(step) for step in steps):
        raise ValueError(f"seat {seat} missing from replay step")
    final_obs = (steps[-1][seat] or {}).get("observation") or {}
    final_own, final_rival, final_margin, terminal_result = _terminal_values(final_obs)
    first_obs = (steps[0][seat] or {}).get("observation") or {}
    state = normalize_observation(first_obs)

    for index in range(len(steps) - 1):
        current = steps[index][seat] or {}
        following = steps[index + 1][seat] or {}
        obs = current.get("observation") or {}
        next_obs = following.get("observation") or {}
        raw_action = following.get("action") or {}
        next_state = normalize_observation(next_obs)
        hand_count = len(state.own.get("hands", []) or [])
        joint = parse_raw_action(raw_action, hand_count)
        emitted = to_engine_action(joint)
        if not raw_equal(raw_action, emitted):
            raise ValueError(f"raw action round-trip failure episode={episode_meta.get('id')} seat={seat} step={index}")
        if not semantic_equivalent(raw_action, emitted, obs):
            raise ValueError(f"semantic action round-trip failure episode={episode_meta.get('id')} seat={seat} step={index}")
        effects = derive_effects(obs, raw_action, next_obs)
        yield {
            "episode_id": int(episode_meta["id"]), "seat": seat, "step": int(state.step),
            "create_time": str(episode_meta.get("createTime", "")),
            "team_id": int(source["team_id"]), "team_name": str(source["team_name"]),
            "submission_id": int(source["submission_id"]),
            "submission_date": str(source.get("submission_date", "")),
            "rank": int(source.get("rank", 0) or 0), "role": str(source.get("role", "")),
            "replay_file_sha256": str(episode_meta.get("replay_file_sha256", "")),
            "replay_json_sha256": str(episode_meta.get("replay_json_sha256", "")),
            "state": asdict(state),
            "raw_action": deepcopy(raw_action),
            "canonical_action": _joint_dict(joint),
            "next_state": asdict(next_state),
            "effects": asdict(effects),
            "final_own_money": final_own,
            "final_rival_money": final_rival,
            "final_margin": final_margin,
            "terminal_result": terminal_result,
        }
        state = next_state


def assign_temporal_splits(records: list[dict]) -> dict[tuple[int, int, int], str]:
    grouped: dict[tuple[int, int], dict[int, str]] = {}
    for row in records:
        key = (int(row["team_id"]), int(row["submission_id"]))
        grouped.setdefault(key, {})[int(row["episode_id"])] = str(row["create_time"])
    out: dict[tuple[int, int, int], str] = {}
    for (team_id, submission_id), episodes in grouped.items():
        ordered = sorted(episodes.items(), key=lambda item: _time_key(item[1]))
        n = len(ordered)
        if n >= 5:
            n_val, n_test = 2, 2
        elif n >= 3:
            n_val, n_test = 1, 1
        elif n == 2:
            n_val, n_test = 0, 1
        else:
            n_val, n_test = 0, 0
        test_start, val_start = n - n_test, n - n_test - n_val
        for idx, (episode_id, _) in enumerate(ordered):
            split = "test" if idx >= test_start else "val" if idx >= val_start else "train"
            out[(team_id, submission_id, episode_id)] = split
    return out


def build_arrow_table(rows: list[dict]):
    import pyarrow as pa

    encoded = []
    for row in rows:
        record = {
            "episode_id": int(row["episode_id"]),
            "seat": int(row["seat"]),
            "step": int(row["step"]),
            "create_time": str(row["create_time"]),
            "team_id": int(row["team_id"]),
            "team_name": str(row["team_name"]),
            "submission_id": int(row["submission_id"]),
            "submission_date": str(row.get("submission_date", "")),
            "rank": int(row.get("rank", 0)),
            "role": str(row.get("role", "")),
            "split": str(row.get("split", "")),
            "replay_file_sha256": str(row.get("replay_file_sha256", "")),
            "replay_json_sha256": str(row.get("replay_json_sha256", "")),
            "state_zlib": encode_zlib_json(row["state"]),
            "next_state_zlib": encode_zlib_json(row["next_state"]),
            "raw_action_json": _json_text(row["raw_action"]),
            "canonical_action_json": _json_text(row["canonical_action"]),
            "effects_json": _json_text(row["effects"]),
            "final_own_money": int(row["final_own_money"]),
            "final_rival_money": int(row["final_rival_money"]),
            "final_margin": int(row["final_margin"]),
            "terminal_result": int(row["terminal_result"]),
        }
        encoded.append(record)
    return pa.Table.from_pylist(encoded)
