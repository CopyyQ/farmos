import gzip
import json

import pytest

from training.build_rl_v2_dataset import build_dataset


def _obs(step, x):
    tiles = [[None, None], [None, None]]
    return {
        "player": 0, "step": step, "day": 0, "hour": step,
        "farms": [
            {"money": 100, "farmer": [x, 0], "hands": [], "hires_today": 0,
             "unlocked_quadrants": ["NW"], "tiles": tiles},
            {"money": 100, "farmer": [1, 1], "hands": [], "hires_today": 0,
             "unlocked_quadrants": ["NW"], "tiles": tiles},
        ],
        "private": {"shed": {}, "seeds": {}, "inventories": [{}]},
        "market": {"inventory": {"WHEAT": 10000}, "prices": {"WHEAT": 25}},
        "town": {"unlocked_shops": []},
    }


def _agent(obs, action):
    return {"observation": obs, "action": action, "reward": 0, "status": "ACTIVE"}


def _replay():
    return {"steps": [
        [_agent(_obs(0, 0), {}), _agent(_obs(0, 1), {})],
        [_agent(_obs(1, 1), {"farmer": ["EAST"], "hands": [], "market": []}), _agent(_obs(1, 1), {})],
    ]}


def _source():
    return {
        "seat": 0, "team_id": 7, "team_name": "A", "submission_id": 11,
        "submission_date": "2026-09-17T01:00:00+00:00",
        "rank": 1, "role": "active_best",
    }


def _manifest(sources=None):
    return {
        "snapshot_time_utc": "2026-09-17T02:00:00+00:00",
        "leaderboard_sha256": "leaderboard-sha",
        "episodes": [{
            "episode": {"id": 99, "createTime": "2026-09-17T01:00:00+00:00"},
            "sources": list(sources if sources is not None else [_source()]),
        }],
    }


def _paths(tmp_path):
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    manifest_path = tmp_path / "manifest.json"
    out = tmp_path / "transitions.parquet"
    summary = tmp_path / "summary.json"
    corpus = tmp_path / "corpus.json"
    return replay_dir, manifest_path, out, summary, corpus


def test_build_dataset_writes_one_trusted_transition(tmp_path):
    replay_dir, manifest_path, out, summary_path, corpus_path = _paths(tmp_path)
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
    with gzip.open(replay_dir / "99.json.gz", "wt", encoding="utf-8") as fh:
        json.dump(_replay(), fh)

    summary = build_dataset(
        manifest_path, replay_dir, out, summary_path, corpus_path, chunk_rows=1,
    )
    assert summary["rows"] == 1
    assert summary["actor_assignments"] == 1
    assert out.exists() and summary_path.exists() and corpus_path.exists()


def test_build_dataset_fails_closed_on_missing_replay(tmp_path):
    replay_dir, manifest_path, out, summary_path, corpus_path = _paths(tmp_path)
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        build_dataset(manifest_path, replay_dir, out, summary_path, corpus_path)


def test_build_dataset_rejects_duplicate_active_assignment(tmp_path):
    replay_dir, manifest_path, out, summary_path, corpus_path = _paths(tmp_path)
    manifest_path.write_text(json.dumps(_manifest([_source(), _source()])), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate active assignment"):
        build_dataset(manifest_path, replay_dir, out, summary_path, corpus_path)


def test_build_dataset_binds_exact_source_code_hashes(tmp_path):
    replay_dir, manifest_path, out, summary_path, corpus_path = _paths(tmp_path)
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
    with gzip.open(replay_dir / "99.json.gz", "wt", encoding="utf-8") as fh:
        json.dump(_replay(), fh)
    build_dataset(manifest_path, replay_dir, out, summary_path, corpus_path, chunk_rows=1)
    corpus = json.loads(corpus_path.read_text())
    code = corpus["build_code_sha256"]
    assert set(code) == {
        "v2_dataset", "v2_action_schema", "v2_observation", "v2_effects", "builder"
    }
    assert all(len(value) == 64 for value in code.values())


def test_build_dataset_freezes_code_hashes_before_reading_sources(tmp_path, monkeypatch):
    import training.build_rl_v2_dataset as builder
    events = []
    real_sources = builder._active_source_records
    monkeypatch.setattr(builder, "_build_code_hashes", lambda: events.append("hash") or {"sentinel": "x"})
    monkeypatch.setattr(builder, "_active_source_records", lambda m: events.append("sources") or real_sources(m))
    replay_dir, manifest_path, out, summary_path, corpus_path = _paths(tmp_path)
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
    with gzip.open(replay_dir / "99.json.gz", "wt", encoding="utf-8") as fh:
        json.dump(_replay(), fh)
    builder.build_dataset(manifest_path, replay_dir, out, summary_path, corpus_path, chunk_rows=1)
    assert events == ["hash", "sources"]
