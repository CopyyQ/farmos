from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import importlib.metadata
import importlib.util
import json
import pathlib
import sys
from collections import Counter
from types import SimpleNamespace
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_LIVE = ROOT / "data" / "top_tier" / "live_v2"


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _load_engine():
    dist = importlib.metadata.distribution("kaggle-environments")
    engine_path = pathlib.Path(dist.locate_file(
        "kaggle_environments/envs/kaggriculture/kaggriculture.py"
    ))
    spec = importlib.util.spec_from_file_location("kaggriculture_effective_ref", engine_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Kaggriculture engine: {engine_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, str(dist.version), engine_path


def _unit_dict(op: str, item=None, quantity=None, raw=None) -> dict[str, Any]:
    return {
        "op": str(op),
        "item": item,
        "quantity": quantity,
        "raw": list(raw if raw is not None else [op]),
    }


def _pass_unit() -> dict[str, Any]:
    return _unit_dict("PASS", raw=["PASS"])


def _nop_market() -> dict[str, Any]:
    return {"kind": "NOP_SLOT", "op": None, "item": None, "quantity": None, "raw": []}


def _stop_market() -> dict[str, Any]:
    return {"kind": "STOP_QUEUE", "op": None, "item": None, "quantity": None, "raw": []}


def _market_order(op: str, item=None, quantity=None) -> dict[str, Any]:
    raw = [op] if quantity is None else [op, item, int(quantity)]
    return {
        "kind": "ORDER",
        "op": str(op),
        "item": item,
        "quantity": quantity,
        "raw": raw,
        "_executed": True,
    }


def _normalize_unit_request(value) -> list:
    return list(value) if isinstance(value, (list, tuple)) and value else ["PASS"]


def _actor_snapshot(kg, farm: dict, private: dict, actor_index: int) -> dict[str, Any]:
    pos = kg._farmer_position(farm, actor_index)
    inv = kg._farmer_inventory(private, actor_index)
    tile = None
    if pos is not None:
        x, y = int(pos[0]), int(pos[1])
        if 0 <= y < len(farm["tiles"]) and 0 <= x < len(farm["tiles"][y]):
            tile = copy.deepcopy(farm["tiles"][y][x])
    return {
        "position": None if pos is None else list(pos),
        "inventory": copy.deepcopy(dict(inv)),
        "tile": tile,
        "shed": copy.deepcopy(dict(private.get("shed") or {})),
        "seeds": copy.deepcopy(dict(private.get("seeds") or {})),
    }


def _effective_unit_from_snapshots(request: list, before: dict, after: dict) -> tuple[dict[str, Any], str]:
    op = str(request[0]) if request else "PASS"
    if op == "PASS":
        return _pass_unit(), "pass"
    changed = before != after
    if not changed:
        return _pass_unit(), "failed"
    item = str(request[1]) if len(request) > 1 and request[1] is not None else None
    if op in {"PICKUP", "PLACE"} and item is not None:
        before_n = int(before["inventory"].get(item, 0) or 0)
        after_n = int(after["inventory"].get(item, 0) or 0)
        executed = max(0, after_n - before_n) if op == "PICKUP" else max(0, before_n - after_n)
        if executed <= 0:
            return _pass_unit(), "failed"
        requested = 1
        if len(request) >= 3:
            try:
                requested = max(0, int(request[2]))
            except (TypeError, ValueError):
                requested = 0
        quantity = None if len(request) < 3 and executed == 1 else executed
        raw = [op, item] if quantity is None else [op, item, int(quantity)]
        status = "executed" if executed == requested else "partial"
        return _unit_dict(op, item=item, quantity=quantity, raw=raw), status
    if op == "PLANT":
        return _unit_dict(op, item=item, raw=[op, item]), "executed"
    return _unit_dict(op, raw=[op]), "executed"


def _trace_units(kg, farms: list[dict], privates: list[dict], actions: list[dict],
                 board_size: int, day: int, turns_per_day: int, shed_capacity: int,
                 stats: Counter) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
    effective_by_player = []
    trace_by_player = []
    for player_id in range(2):
        farm = farms[player_id]
        private = privates[player_id]
        action = actions[player_id] if isinstance(actions[player_id], dict) else {}
        farmer_req = _normalize_unit_request(action.get("farmer", ["PASS"]))
        raw_hands = action.get("hands", [])
        raw_hands = raw_hands if isinstance(raw_hands, list) else []
        hand_count = len(farm.get("hands") or [])

        provided = [farmer_req] + [_normalize_unit_request(x) for x in raw_hands]
        demand: dict[str, int] = {}
        for req in provided:
            if len(req) >= 2 and str(req[0]) == "PLANT":
                crop = str(req[1])
                demand[crop] = demand.get(crop, 0) + 1
        seeds = private.get("seeds") or {}
        blocked = {crop for crop, n in demand.items() if n > int(seeds.get(crop, 0) or 0)}

        out_units = []
        out_trace = []
        requests = [farmer_req] + [
            _normalize_unit_request(raw_hands[i]) if i < len(raw_hands) else ["PASS"]
            for i in range(hand_count)
        ]
        for actor_index, req in enumerate(requests):
            op = str(req[0]) if req else "PASS"
            applied = ["PASS"] if (op == "PLANT" and len(req) >= 2 and str(req[1]) in blocked) else req
            before = _actor_snapshot(kg, farm, private, actor_index)
            kg._apply_unit_action(
                farm, private, actor_index, applied,
                board_size, day, turns_per_day, shed_capacity,
            )
            after = _actor_snapshot(kg, farm, private, actor_index)
            effective, status = _effective_unit_from_snapshots(req, before, after)
            out_units.append(effective)
            out_trace.append({
                "actor": "farmer" if actor_index == 0 else f"hand:{actor_index - 1}",
                "requested": req,
                "effective": effective,
                "status": status,
            })
            stats[("unit", op, status)] += 1
        effective_by_player.append({
            "farmer": out_units[0],
            "hands": out_units[1:],
        })
        trace_by_player.append(out_trace)
    return effective_by_player, trace_by_player


def _trace_market(kg, farms: list[dict], privates: list[dict], market: dict,
                  actions: list[dict], config: dict, stats: Counter):
    board_size = int(config.get("boardSize", 10))
    max_orders = max(1, int(config.get("maxMarketOrdersPerTurn", 10)))
    hire_mult = int(config.get("farmHandCostMult", kg.FARM_HAND_COST_MULT))
    shed_capacity = int(config.get("shedCapacity", 100))
    queues = []
    for action in actions:
        raw = action.get("market", []) if isinstance(action, dict) else []
        raw = raw if isinstance(raw, list) else []
        queues.append(list(raw[:max_orders]))

    effective = [[_nop_market() for _ in q] for q in queues]
    traces = [[] for _ in queues]
    executed = [[0 for _ in q] for q in queues]
    requested_qty = [[None for _ in q] for q in queues]
    slot_status = [["failed" for _ in q] for q in queues]

    max_len = max((len(q) for q in queues), default=0)
    for slot in range(max_len):
        order_states = []
        for player_id, q in enumerate(queues):
            state = kg._parse_order(q[slot]) if slot < len(q) else None
            order_states.append(state)
            if state is not None and "remaining" in state and slot < len(requested_qty[player_id]):
                requested_qty[player_id][slot] = int(state["remaining"])

        for player_id, state in enumerate(order_states):
            if state is None:
                continue
            op = state["type"]
            if op == "HIRE":
                before = (farms[player_id]["money"], len(farms[player_id]["hands"]))
                kg._do_hire(farms[player_id], privates[player_id], board_size, hire_mult)
                after = (farms[player_id]["money"], len(farms[player_id]["hands"]))
                ok = after != before
                if slot < len(effective[player_id]) and ok:
                    effective[player_id][slot] = _market_order("HIRE")
                if slot < len(queues[player_id]):
                    status = "executed" if ok else "failed"
                    slot_status[player_id][slot] = status
                    stats[("market", "HIRE", status)] += 1
                order_states[player_id] = None
            elif op == "BUY_LAND":
                before = (farms[player_id]["money"], tuple(farms[player_id]["unlocked_quadrants"]))
                kg._do_buy_land(farms[player_id], board_size)
                after = (farms[player_id]["money"], tuple(farms[player_id]["unlocked_quadrants"]))
                ok = after != before
                if slot < len(effective[player_id]) and ok:
                    effective[player_id][slot] = _market_order("BUY_LAND")
                if slot < len(queues[player_id]):
                    status = "executed" if ok else "failed"
                    slot_status[player_id][slot] = status
                    stats[("market", "BUY_LAND", status)] += 1
                order_states[player_id] = None

        guard = 0
        while True:
            guard += 1
            if guard >= 100000:
                raise RuntimeError("reference market loop exceeded 100k")
            quoted = [None, None]
            for player_id, state in enumerate(order_states):
                if state is None or state["remaining"] <= 0:
                    continue
                op = state["type"]
                item = state["item"]
                if op == "SELL" and item in kg.PRODUCTS:
                    quoted[player_id] = (
                        op, item,
                        kg.market_price(item, market["inventory"][item], market.get("params")),
                        state,
                    )
                elif op == "BUY_PRODUCT" and item in ("WHEAT", "FERTILIZER"):
                    quoted[player_id] = (
                        op, item,
                        kg.market_price(item, market["inventory"][item] - 1, market.get("params")),
                        state,
                    )
                elif op == "BUY_SEED" and item in kg.CROPS:
                    quoted[player_id] = (op, item, kg.CROPS[item]["seed"], state)
                elif op == "BUY_ANIMAL" and item in kg.ANIMALS:
                    quoted[player_id] = (op, item, kg.ANIMALS[item]["cost"], state)
                else:
                    order_states[player_id] = None
            if all(q is None for q in quoted):
                break
            committed_any = False
            for player_id, quote in enumerate(quoted):
                if quote is None:
                    continue
                op, item, price, state = quote
                ok = kg._commit_unit(
                    op, item, price, farms[player_id], privates[player_id],
                    market, shed_capacity,
                )
                if ok:
                    state["remaining"] -= 1
                    if slot < len(executed[player_id]):
                        executed[player_id][slot] += 1
                    committed_any = True
                else:
                    order_states[player_id] = None
            if not committed_any:
                break
        kg._refresh_prices(market)

        for player_id in range(2):
            if slot >= len(queues[player_id]):
                continue
            parsed = kg._parse_order(queues[player_id][slot])
            if parsed is None:
                slot_status[player_id][slot] = "failed"
                stats[("market", "INVALID", "failed")] += 1
                continue
            op = parsed["type"]
            if op in {"HIRE", "BUY_LAND"}:
                continue
            req = int(requested_qty[player_id][slot] or 0)
            done = int(executed[player_id][slot])
            if done > 0:
                effective[player_id][slot] = _market_order(op, parsed["item"], done)
            status = "failed" if done == 0 else "executed" if done == req else "partial"
            slot_status[player_id][slot] = status
            stats[("market", op, status)] += 1

    for player_id in range(2):
        if len(queues[player_id]) < max_orders:
            effective[player_id].append(_stop_market())
        for slot, req in enumerate(queues[player_id]):
            eff = effective[player_id][slot]
            traces[player_id].append({
                "slot": slot,
                "requested": req,
                "effective": eff,
                "status": slot_status[player_id][slot],
            })
    return effective, traces


def _core_obs(obs: dict) -> dict[str, Any]:
    return {
        "day": obs.get("day"),
        "hour": obs.get("hour"),
        "farms": obs.get("farms"),
        "private": obs.get("private"),
        "market": obs.get("market"),
        "town": obs.get("town"),
    }


def _verify_non_eod(kg, farms, privates, market, town, current_obs, next_records,
                    config, replay_info, current_step):
    state = []
    for player_id in range(2):
        observation = copy.deepcopy(current_obs[player_id])
        observation["farms"] = farms
        observation["market"] = market
        observation["town"] = town
        observation["private"] = privates[player_id]
        state.append(SimpleNamespace(observation=_Attr(observation)))
    env = SimpleNamespace(configuration=_Attr(config), info=_Attr(replay_info or {}))
    kg._town_consume(env, state, current_step)
    for farm in farms:
        kg._decay_plants(farm, current_step)
    next_step = current_step + 1
    state[0].observation.day = next_step // int(config.get("turnsPerDay", 24))
    state[0].observation.hour = next_step % int(config.get("turnsPerDay", 24))
    for player_id in range(1, 2):
        state[player_id].observation.farms = farms
        state[player_id].observation.market = market
        state[player_id].observation.town = town
        state[player_id].observation.day = state[0].observation.day
        state[player_id].observation.hour = state[0].observation.hour
    for player_id in range(2):
        got = _core_obs(_plain(state[player_id].observation))
        expected = _core_obs(next_records[player_id].get("observation") or {})
        if got != expected:
            raise RuntimeError(
                f"reference tracer mismatch step={current_step} player={player_id}"
            )


class _Attr(dict):
    __getattr__ = dict.__getitem__
    def __setattr__(self, key, value):
        self[key] = value


def _plain(value):
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain(v) for v in value]
    return value


def build(live: pathlib.Path, output: pathlib.Path, summary_path: pathlib.Path):
    kg, engine_version, engine_path = _load_engine()
    dataset_path = live / "transitions.parquet"
    replay_dir = live / "replays"
    corpus_manifest = live / "manifests/trusted_corpus_manifest.json"
    if not dataset_path.is_file() or not replay_dir.is_dir():
        raise FileNotFoundError("accepted dataset/replay corpus is incomplete")
    if not corpus_manifest.is_file():
        raise FileNotFoundError(corpus_manifest)
    trusted_corpus = json.loads(corpus_manifest.read_text(encoding="utf-8"))
    trusted_episodes = {
        int(row["episode_id"]): row
        for row in trusted_corpus.get("episodes", [])
    }

    base = pq.read_table(
        dataset_path,
        columns=["episode_id", "seat", "step", "split", "role"],
    ).to_pylist()
    expected = {
        (int(r["episode_id"]), int(r["seat"]), int(r["step"])): (
            str(r["split"]), str(r["role"])
        )
        for r in base
    }
    if len(expected) != len(base):
        raise RuntimeError("base dataset contains duplicate transition keys")

    episode_to_seats: dict[int, set[int]] = {}
    for episode_id, seat, _ in expected:
        episode_to_seats.setdefault(episode_id, set()).add(seat)

    all_player_stats = Counter()
    active_stats = Counter()
    output_rows = []
    verified = 0
    replay_versions = Counter()
    for episode_id in sorted(episode_to_seats):
        replay_path = replay_dir / f"{episode_id}.json.gz"
        if not replay_path.is_file():
            raise FileNotFoundError(replay_path)
        trusted = trusted_episodes.get(episode_id)
        if trusted is None:
            raise RuntimeError(
                f"replay episode is not present in trusted corpus manifest: {episode_id}"
            )
        replay_file_sha = _sha256(replay_path)
        if replay_file_sha != str(trusted.get("replay_file_sha256", "")):
            raise RuntimeError(
                f"trusted replay file SHA mismatch: episode={episode_id}"
            )
        with gzip.open(replay_path, "rb") as fh:
            raw_json = fh.read()
        replay_json_sha = hashlib.sha256(raw_json).hexdigest()
        if replay_json_sha != str(trusted.get("replay_json_sha256", "")):
            raise RuntimeError(
                f"trusted replay JSON SHA mismatch: episode={episode_id}"
            )
        replay = json.loads(raw_json.decode("utf-8"))
        replay_version = str(replay.get("module_version"))
        replay_versions[replay_version] += 1
        if replay_version != engine_version:
            raise RuntimeError(
                f"engine version mismatch replay={episode_id}: "
                f"{replay_version} != installed {engine_version}"
            )
        config = replay.get("configuration") or {}
        turns_per_day = max(1, int(config.get("turnsPerDay", 24)))
        steps = replay.get("steps") or []
        for replay_index in range(1, len(steps)):
            current_records = [copy.deepcopy(steps[replay_index - 1][p]) for p in range(2)]
            next_records = [steps[replay_index][p] for p in range(2)]
            current_obs = [r.get("observation") or {} for r in current_records]
            actions = [copy.deepcopy(next_records[p].get("action") or {}) for p in range(2)]
            shared = current_obs[0]
            farms = copy.deepcopy(shared.get("farms") or [])
            privates = [copy.deepcopy(current_obs[p].get("private") or {}) for p in range(2)]
            market = copy.deepcopy(shared.get("market") or {})
            town = copy.deepcopy(shared.get("town") or {})
            current_step = int(shared.get(
                "step",
                int(shared.get("day", 0)) * turns_per_day + int(shared.get("hour", 0)),
            ))
            day = current_step // turns_per_day
            board_size = int(config.get("boardSize", 10))
            shed_capacity = int(config.get("shedCapacity", 100))

            unit_effective, unit_trace = _trace_units(
                kg, farms, privates, actions, board_size, day,
                turns_per_day, shed_capacity, all_player_stats,
            )
            market_effective, market_trace = _trace_market(
                kg, farms, privates, market, actions, config, all_player_stats,
            )

            if (current_step + 1) % turns_per_day != 0:
                _verify_non_eod(
                    kg, farms, privates, market, town, current_obs, next_records,
                    config, replay.get("info") or {}, current_step,
                )
                verified += 1

            for seat in episode_to_seats[episode_id]:
                key = (episode_id, seat, current_step)
                if key not in expected:
                    continue
                split, role = expected[key]
                effective_action = {
                    "farmer": unit_effective[seat]["farmer"],
                    "hands": unit_effective[seat]["hands"],
                    "market": market_effective[seat],
                }
                trace = {
                    "units": unit_trace[seat],
                    "market": market_trace[seat],
                    "engine_module_version": engine_version,
                }
                for item in unit_trace[seat]:
                    requested = item.get("requested") or ["PASS"]
                    op = str(requested[0]) if requested else "PASS"
                    active_stats[("unit", op, str(item.get("status", "failed")))] += 1
                for item in market_trace[seat]:
                    requested = item.get("requested")
                    op = (
                        str(requested[0])
                        if isinstance(requested, list) and requested
                        else "INVALID"
                    )
                    active_stats[("market", op, str(item.get("status", "failed")))] += 1
                output_rows.append({
                    "episode_id": episode_id,
                    "seat": seat,
                    "step": current_step,
                    "split": split,
                    "role": role,
                    "effective_action_json": _json(effective_action),
                    "execution_trace_json": _json(trace),
                    "engine_module_version": engine_version,
                })

    produced = {
        (int(r["episode_id"]), int(r["seat"]), int(r["step"]))
        for r in output_rows
    }
    missing = set(expected) - produced
    extra = produced - set(expected)
    if missing or extra or len(output_rows) != len(expected):
        raise RuntimeError(
            f"sidecar key mismatch rows={len(output_rows)} expected={len(expected)} "
            f"missing={len(missing)} extra={len(extra)}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(output_rows), tmp, compression="zstd")
    tmp.replace(output)
    summary = {
        "rows": len(output_rows),
        "verified_non_eod_transitions": verified,
        "engine_module_version": engine_version,
        "engine_source_sha256": _sha256(engine_path),
        "source_dataset_sha256": _sha256(dataset_path),
        "source_corpus_manifest_sha256": _sha256(corpus_manifest),
        "effective_actions_sha256": _sha256(output),
        "builder_code_sha256": _sha256(pathlib.Path(__file__).resolve()),
        "verified_replay_files": sum(replay_versions.values()),
        "replay_module_versions": dict(sorted(replay_versions.items())),
        "execution_stats": {
            "|".join(map(str, key)): int(value)
            for key, value in sorted(active_stats.items())
        },
        "reference_all_player_execution_stats": {
            "|".join(map(str, key)): int(value)
            for key, value in sorted(all_player_stats.items())
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", type=pathlib.Path, default=DEFAULT_LIVE)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--summary", type=pathlib.Path)
    args = parser.parse_args()
    live = args.live.resolve()
    output = (args.output or (live / "effective_actions.parquet")).resolve()
    summary = (args.summary or (live / "manifests/effective_actions_summary.json")).resolve()
    build(live, output, summary)


if __name__ == "__main__":
    main()
