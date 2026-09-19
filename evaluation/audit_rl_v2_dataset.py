from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pathlib
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from kaggrl.v2_action_schema import parse_raw_action, raw_equal, to_engine_action
from kaggrl.v2_dataset import decode_zlib_json


COUNTER_NAMES = (
    "active_mapping_failures", "membership_failures", "sha_failures",
    "seat_alignment_failures", "semantic_roundtrip_failures",
    "private_leak_failures", "hand_truncations", "quantity_clamps",
    "stop_nop_conflations", "missing_transitions", "duplicate_rows",
)


@dataclass
class AuditReport:
    passed: bool
    counters: dict[str, int]
    coverage: dict[str, Any]
    rows: int


def _sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_text(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _current_build_code_hashes() -> dict[str, str]:
    return {
        "v2_dataset": _sha256_file(ROOT / "src/kaggrl/v2_dataset.py"),
        "v2_action_schema": _sha256_file(ROOT / "src/kaggrl/v2_action_schema.py"),
        "v2_observation": _sha256_file(ROOT / "src/kaggrl/v2_observation.py"),
        "v2_effects": _sha256_file(ROOT / "src/kaggrl/v2_effects.py"),
        "builder": _sha256_file(ROOT / "training/build_rl_v2_dataset.py"),
    }


def _canonical_dict(joint) -> dict[str, Any]:
    def unit(command):
        return {"op": command.op, "item": command.item, "quantity": command.quantity, "raw": list(command.raw)}

    def slot(value):
        return {
            "kind": value.kind, "op": value.op, "item": value.item,
            "quantity": value.quantity, "raw": list(value.raw),
        }

    return {
        "farmer": unit(joint.farmer),
        "hands": [unit(x) for x in joint.hands],
        "market": [slot(x) for x in joint.market],
    }


def _quantity(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _has_private_leak(state: dict[str, Any]) -> bool:
    rival = state.get("rival") or {}
    if isinstance(rival, dict) and "private" in rival:
        return True
    for unit in state.get("rival_units") or []:
        if isinstance(unit, dict) and "inventory" in unit:
            return True
    return False


def _scan_quantities(raw_action, canonical, quantity_counts: Counter) -> tuple[int, int]:
    clamps = 0
    largest = 0
    unit_pairs = [(raw_action.get("farmer"), canonical.get("farmer", {}))]
    raw_hands = raw_action.get("hands", []) if isinstance(raw_action.get("hands", []), list) else []
    can_hands = canonical.get("hands", []) or []
    unit_pairs.extend((cmd, can_hands[i] if i < len(can_hands) else {}) for i, cmd in enumerate(raw_hands))
    for raw, can in unit_pairs:
        if isinstance(raw, list) and len(raw) >= 3:
            q = _quantity(raw[2])
            if q is not None:
                quantity_counts[q] += 1; largest = max(largest, abs(q))
                if can.get("quantity") != q:
                    clamps += 1
    raw_market = raw_action.get("market", []) if isinstance(raw_action.get("market", []), list) else []
    can_market = canonical.get("market", []) or []
    for i, raw in enumerate(raw_market):
        if isinstance(raw, list) and len(raw) >= 3:
            q = _quantity(raw[2])
            if q is not None:
                quantity_counts[q] += 1; largest = max(largest, abs(q))
                if i >= len(can_market) or can_market[i].get("quantity") != q:
                    clamps += 1
    return clamps, largest


def _expected_contract(manifest, corpus, replay_dir, counters):
    active_by_team = {}
    for team in manifest.get("teams", []):
        active = (team.get("roles") or {}).get("active_best") or {}
        if active.get("id") is not None:
            active_by_team[int(team["team_id"])] = int(active["id"])
    selection_sources = {}
    for entry in manifest.get("episodes", []):
        episode = entry.get("episode") or {}
        eid = int(episode["id"])
        api_agents = episode.get("agents") or []
        for source in entry.get("sources", []):
            if source.get("role") != "active_best":
                continue
            team_id, submission_id = int(source["team_id"]), int(source["submission_id"])
            seat = int(source["seat"])
            if active_by_team.get(team_id) != submission_id:
                counters["active_mapping_failures"] += 1
            if seat < 0 or seat >= len(api_agents):
                counters["seat_alignment_failures"] += 1
                counters["membership_failures"] += 1
            else:
                agent = api_agents[seat] or {}
                if int(agent.get("index", seat)) != seat:
                    counters["seat_alignment_failures"] += 1
                if int(agent.get("submissionId", -1)) != submission_id or int(agent.get("teamId", -1)) != team_id:
                    counters["membership_failures"] += 1
                api_name = str(agent.get("teamName", ""))
                if api_name and api_name != str(source.get("team_name", "")):
                    counters["membership_failures"] += 1
            key = (team_id, submission_id, eid, seat)
            selection_sources[key] = source
    if len(active_by_team) != len({key[0] for key in selection_sources}):
        counters["active_mapping_failures"] += abs(len(active_by_team) - len({key[0] for key in selection_sources}))

    corpus_entries = {int(row["episode_id"]): row for row in corpus.get("episodes", [])}
    if corpus.get("selection_snapshot_sha256") != _sha256_file(pathlib.Path(manifest["__path__"])):
        counters["sha_failures"] += 1
    expected_rows = {}
    replay_hashes = {}
    for key in selection_sources:
        eid = key[2]
        replay_path = replay_dir / f"{eid}.json.gz"
        if not replay_path.exists():
            counters["sha_failures"] += 1
            continue
        file_sha = _sha256_file(replay_path)
        with gzip.open(replay_path, "rb") as fh:
            raw = fh.read()
        json_sha = hashlib.sha256(raw).hexdigest()
        replay = json.loads(raw.decode("utf-8"))
        info = replay.get("info") or {}
        if info.get("EpisodeId") is not None and int(info.get("EpisodeId")) != eid:
            counters["membership_failures"] += 1
        replay_agents = info.get("Agents") or []
        seat = key[3]
        if seat < 0 or seat >= len(replay_agents):
            counters["seat_alignment_failures"] += 1
        else:
            replay_name = str((replay_agents[seat] or {}).get("Name", ""))
            source_name = str(selection_sources[key].get("team_name", ""))
            if replay_name and source_name and replay_name != source_name:
                counters["membership_failures"] += 1
        replay_hashes[eid] = (file_sha, json_sha)
        expected_rows[key] = max(0, len(replay.get("steps") or []) - 1)
    for eid, hashes in replay_hashes.items():
        corpus_row = corpus_entries.get(eid)
        if corpus_row is None:
            counters["membership_failures"] += 1
            continue
        if corpus_row.get("replay_file_sha256") != hashes[0]:
            counters["sha_failures"] += 1
        if corpus_row.get("replay_json_sha256") != hashes[1]:
            counters["sha_failures"] += 1
        expected_assignment_count = sum(1 for key in selection_sources if key[2] == eid)
        if int(corpus_row.get("active_assignments", -1)) != expected_assignment_count:
            counters["membership_failures"] += 1
    extra_corpus = set(corpus_entries) - set(replay_hashes)
    counters["membership_failures"] += len(extra_corpus)
    return selection_sources, expected_rows, replay_hashes


def audit_dataset(manifest_path, parquet_path) -> AuditReport:
    import pyarrow.parquet as pq

    manifest_path = pathlib.Path(manifest_path)
    parquet_path = pathlib.Path(parquet_path)
    corpus_path = manifest_path.parent / "trusted_corpus_manifest.json"
    marker_path = manifest_path.parent / "STAGE0_ACCEPTED.json"
    if marker_path.exists():
        marker_path.unlink()
    counters = Counter({name: 0 for name in COUNTER_NAMES})
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["__path__"] = str(manifest_path)
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    current_build_code = _current_build_code_hashes()
    stored_build_code = corpus.get("build_code_sha256") or {}
    for name, expected_sha in current_build_code.items():
        if stored_build_code.get(name) != expected_sha:
            counters["sha_failures"] += 1
    replay_dir = manifest_path.parent.parent / "replays"
    selection_sources, expected_rows, replay_hashes = _expected_contract(
        manifest, corpus, replay_dir, counters
    )

    identity_to_seat = {(k[0], k[1], k[2]): k[3] for k in selection_sources}
    observed = Counter()
    seen_rows = set()
    bad_hash_assignments = set()
    quantity_counts = Counter()
    market_lengths = Counter()
    unit_ops = Counter(); market_ops = Counter(); item_pairs = Counter()
    max_hands = max_market = largest_quantity = noop_slots = row_count = 0

    parquet = pq.ParquetFile(parquet_path)
    for batch in parquet.iter_batches(batch_size=2048):
        for row in batch.to_pylist():
            row_count += 1
            key = (int(row["team_id"]), int(row["submission_id"]), int(row["episode_id"]), int(row["seat"]))
            observed[key] += 1
            if key not in selection_sources:
                counters["membership_failures"] += 1
            expected_seat = identity_to_seat.get(key[:3])
            if expected_seat is not None and expected_seat != key[3]:
                counters["seat_alignment_failures"] += 1
            unique = (key[2], key[3], int(row["step"]))
            if unique in seen_rows:
                counters["duplicate_rows"] += 1
            seen_rows.add(unique)
            hashes = replay_hashes.get(key[2])
            if hashes and (row.get("replay_file_sha256") != hashes[0] or row.get("replay_json_sha256") != hashes[1]):
                bad_hash_assignments.add(key)
            state = decode_zlib_json(row["state_zlib"])
            next_state = decode_zlib_json(row["next_state_zlib"])
            if _has_private_leak(state) or _has_private_leak(next_state):
                counters["private_leak_failures"] += 1
            raw_action = json.loads(row["raw_action_json"])
            stored = json.loads(row["canonical_action_json"])
            hand_count = len((state.get("own") or {}).get("hands") or [])
            max_hands = max(max_hands, hand_count)
            if len(stored.get("hands", []) or []) != hand_count:
                counters["hand_truncations"] += 1
            fresh = parse_raw_action(raw_action, hand_count)
            fresh_dict = _canonical_dict(fresh)
            if not raw_equal(raw_action, to_engine_action(fresh)) or _json_text(stored) != _json_text(fresh_dict):
                counters["semantic_roundtrip_failures"] += 1

            raw_market = raw_action.get("market", []) if isinstance(raw_action.get("market", []), list) else []
            market_lengths[len(raw_market)] += 1
            max_market = max(max_market, len(raw_market))
            for i, expected_slot in enumerate(fresh_dict["market"]):
                if expected_slot["kind"] in ("STOP_QUEUE", "NOP_SLOT"):
                    actual_kind = stored.get("market", [{}])[i].get("kind") if i < len(stored.get("market", [])) else None
                    if actual_kind != expected_slot["kind"]:
                        counters["stop_nop_conflations"] += 1
                if expected_slot["kind"] == "NOP_SLOT":
                    noop_slots += 1
            clamps, row_largest = _scan_quantities(raw_action, stored, quantity_counts)
            counters["quantity_clamps"] += clamps
            largest_quantity = max(largest_quantity, row_largest)
            unit_commands = [raw_action.get("farmer"), *(raw_action.get("hands", []) if isinstance(raw_action.get("hands", []), list) else [])]
            for command in unit_commands:
                if isinstance(command, list) and command:
                    op = str(command[0]); unit_ops[op] += 1
                    if len(command) > 1:
                        item_pairs[f"unit:{op}|{command[1]}"] += 1
            for order in raw_market:
                if isinstance(order, list) and order:
                    op = str(order[0]); market_ops[op] += 1
                    if len(order) > 1:
                        item_pairs[f"market:{op}|{order[1]}"] += 1

    counters["sha_failures"] += len(bad_hash_assignments)
    expected_keys = set(expected_rows)
    observed_keys = set(observed)
    counters["membership_failures"] += len(observed_keys - expected_keys)
    for key, expected in expected_rows.items():
        actual = int(observed.get(key, 0))
        if actual != expected:
            counters["missing_transitions"] += abs(expected - actual)
    coverage = {
        "max_hand_count": max_hands,
        "max_market_length": max_market,
        "market_length_distribution": {str(k): int(v) for k, v in sorted(market_lengths.items())},
        "unit_operation_counts": dict(sorted(unit_ops.items())),
        "market_operation_counts": dict(sorted(market_ops.items())),
        "operation_item_counts": dict(sorted(item_pairs.items())),
        "quantity_distribution": {str(k): int(v) for k, v in sorted(quantity_counts.items())},
        "largest_quantity": largest_quantity,
        "noop_market_slots": noop_slots,
    }
    normalized_counters = {name: int(counters[name]) for name in COUNTER_NAMES}
    passed = all(value == 0 for value in normalized_counters.values())
    report = AuditReport(passed=passed, counters=normalized_counters, coverage=coverage, rows=row_count)
    if passed:
        marker = {
            "accepted": True,
            "accepted_at_utc": datetime.now(timezone.utc).isoformat(),
            "rows": row_count,
            "counters": normalized_counters,
            "coverage": coverage,
            "artifacts": {
                "selection_snapshot_sha256": _sha256_file(manifest_path),
                "trusted_corpus_manifest_sha256": _sha256_file(corpus_path),
                "dataset_sha256": _sha256_file(parquet_path),
                "dataset_code_sha256": _sha256_file(ROOT / "src/kaggrl/v2_dataset.py"),
                "action_schema_sha256": _sha256_file(ROOT / "src/kaggrl/v2_action_schema.py"),
                "observation_code_sha256": _sha256_file(ROOT / "src/kaggrl/v2_observation.py"),
                "effects_code_sha256": _sha256_file(ROOT / "src/kaggrl/v2_effects.py"),
                "builder_code_sha256": _sha256_file(ROOT / "training/build_rl_v2_dataset.py"),
                "audit_code_sha256": _sha256_file(pathlib.Path(__file__)),
            },
        }
        marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(ROOT / "data/top_tier/live_v2/manifests/live_top10_snapshot.json"))
    parser.add_argument("--parquet", default=str(ROOT / "data/top_tier/live_v2/transitions.parquet"))
    args = parser.parse_args()
    report = audit_dataset(args.manifest, args.parquet)
    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    raise SystemExit(0 if report.passed else 2)


if __name__ == "__main__":
    main()
