import gzip
import json
from copy import deepcopy

import pyarrow as pa
import pyarrow.parquet as pq

from evaluation.audit_rl_v2_dataset import audit_dataset
from kaggrl.v2_dataset import decode_zlib_json, encode_zlib_json
from training.build_rl_v2_dataset import build_dataset


def _tiles():
    return [[None for _ in range(10)] for _ in range(10)]


def _obs(step, money=100):
    return {
        "player": 0, "step": step, "day": step // 24, "hour": step % 24,
        "farms": [
            {"money": money, "farmer": [4 + min(step, 1), 4], "hands": [], "hires_today": 0,
             "unlocked_quadrants": ["NW"], "tiles": _tiles()},
            {"money": 90, "farmer": [8, 8], "hands": [], "hires_today": 0,
             "unlocked_quadrants": ["NW"], "tiles": _tiles()},
        ],
        "private": {"shed": {}, "seeds": {}, "inventories": [{}]},
        "market": {"inventory": {"WHEAT": 10000}, "prices": {"WHEAT": 25}},
        "town": {"unlocked_shops": []},
    }


def _agent(obs, action):
    return {"observation": obs, "action": action, "reward": 0, "status": "ACTIVE"}


def _fixture(tmp_path):
    replay_dir = tmp_path / "replays"; replay_dir.mkdir()
    manifests = tmp_path / "manifests"; manifests.mkdir()
    replay = {
        "info": {"EpisodeId": 1, "Agents": [{"Name": "A"}, {"Name": "Rival"}]},
        "steps": [
            [_agent(_obs(0), {"farmer": ["PASS"], "hands": [], "market": []}), _agent({"player": 1}, {})],
            [_agent(_obs(1), {"farmer": ["EAST"], "hands": [], "market": []}), _agent({"player": 1}, {})],
            [_agent(_obs(2, 125), {"farmer": ["PASS"], "hands": [], "market": [["SELL", "WHEAT", 1000]]}), _agent({"player": 1}, {})],
        ]
    }
    raw = json.dumps(replay, sort_keys=True).encode()
    with gzip.open(replay_dir / "1.json.gz", "wb") as fh:
        fh.write(raw)
    source = {"seat": 0, "team_id": 7, "team_name": "A", "submission_id": 11,
              "submission_date": "2026-09-16T20:00:00Z", "rank": 1, "role": "active_best"}
    manifest = {
        "snapshot_time_utc": "2026-09-17T02:24:27Z", "leaderboard_sha256": "lb",
        "teams": [{"rank": 1, "team_id": 7, "team_name": "A",
                   "roles": {"active_best": {"id": 11, "dateSubmitted": source["submission_date"]}, "latest_candidate": None}}],
        "episodes": [{"episode": {
            "id": 1, "createTime": "2026-09-17T02:00:00Z",
            "agents": [
                {"index": 0, "submissionId": 11, "teamId": 7, "teamName": "A"},
                {"index": 1, "submissionId": 22, "teamId": 8, "teamName": "Rival"},
            ],
        }, "sources": [source]}],
    }
    manifest_path = manifests / "live_top10_snapshot.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    parquet_path = tmp_path / "transitions.parquet"
    corpus_path = manifests / "trusted_corpus_manifest.json"
    build_dataset(
        manifest_path, replay_dir, parquet_path,
        manifests / "transitions_summary.json", corpus_path, chunk_rows=1,
    )
    return manifest_path, parquet_path, corpus_path, manifests / "STAGE0_ACCEPTED.json"


def test_audit_writes_acceptance_marker_only_when_all_counters_are_zero(tmp_path):
    manifest_path, parquet_path, _, marker = _fixture(tmp_path)
    report = audit_dataset(manifest_path, parquet_path)
    assert report.passed is True
    assert all(value == 0 for value in report.counters.values())
    assert marker.exists()
    accepted = json.loads(marker.read_text())
    assert accepted["accepted"] is True
    assert accepted["artifacts"]["dataset_sha256"]
    for key in (
        "dataset_code_sha256", "action_schema_sha256", "observation_code_sha256",
        "effects_code_sha256", "builder_code_sha256", "audit_code_sha256",
    ):
        assert accepted["artifacts"][key]
    assert report.coverage["max_market_length"] == 1
    assert report.coverage["largest_quantity"] == 1000


def test_audit_fails_closed_on_replay_hash_mismatch(tmp_path):
    manifest_path, parquet_path, corpus_path, marker = _fixture(tmp_path)
    corpus = json.loads(corpus_path.read_text())
    corpus["episodes"][0]["replay_file_sha256"] = "tampered"
    corpus_path.write_text(json.dumps(corpus), encoding="utf-8")
    report = audit_dataset(manifest_path, parquet_path)
    assert report.passed is False
    assert report.counters["sha_failures"] > 0
    assert not marker.exists()


def test_audit_fails_closed_on_opponent_private_leak(tmp_path):
    manifest_path, parquet_path, _, marker = _fixture(tmp_path)
    table = pq.read_table(parquet_path)
    records = table.to_pylist()
    state = decode_zlib_json(records[0]["state_zlib"])
    state["rival"]["private"] = {"shed": {"MILK": 999}}
    records[0]["state_zlib"] = encode_zlib_json(state)
    pq.write_table(pa.Table.from_pylist(records), parquet_path, compression="zstd")
    report = audit_dataset(manifest_path, parquet_path)
    assert report.passed is False
    assert report.counters["private_leak_failures"] > 0
    assert not marker.exists()


def test_audit_fails_closed_when_manifest_seat_submission_disagrees_with_episode_metadata(tmp_path):
    manifest_path, parquet_path, _, marker = _fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["episodes"][0]["episode"]["agents"][0]["submissionId"] = 999
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    report = audit_dataset(manifest_path, parquet_path)
    assert report.passed is False
    assert report.counters["membership_failures"] > 0 or report.counters["seat_alignment_failures"] > 0
    assert not marker.exists()


def test_audit_fails_closed_when_dataset_build_code_provenance_is_stale(tmp_path):
    manifest_path, parquet_path, corpus_path, marker = _fixture(tmp_path)
    corpus = json.loads(corpus_path.read_text())
    corpus["build_code_sha256"]["v2_effects"] = "0" * 64
    corpus_path.write_text(json.dumps(corpus), encoding="utf-8")
    report = audit_dataset(manifest_path, parquet_path)
    assert report.passed is False
    assert report.counters["sha_failures"] > 0
    assert not marker.exists()
