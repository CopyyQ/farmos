import json

from kaggrl.v2_dataset import (
    assign_temporal_splits,
    build_arrow_table,
    decode_zlib_json,
    iter_transitions,
)


def _tiles():
    return [[None, None], [None, None]]


def _obs(step, x, money=100):
    return {
        "player": 0, "step": step, "day": step // 24, "hour": step % 24,
        "farms": [
            {"money": money, "farmer": [x, 0], "hands": [], "hires_today": 0,
             "unlocked_quadrants": ["NW"], "tiles": _tiles()},
            {"money": 100, "farmer": [1, 1], "hands": [], "hires_today": 0,
             "unlocked_quadrants": ["NW"], "tiles": _tiles()},
        ],
        "private": {"shed": {}, "seeds": {}, "inventories": [{}]},
        "market": {"inventory": {"WHEAT": 10000}, "prices": {"WHEAT": 25}},
        "town": {"unlocked_shops": []},
    }


def _agent(obs, action):
    return {"observation": obs, "action": action, "reward": 0, "status": "ACTIVE"}


def _episode_meta():
    return {
        "id": 99,
        "createTime": "2026-09-17T01:00:00+00:00",
        "replay_file_sha256": "file-sha",
        "replay_json_sha256": "json-sha",
    }


def _source():
    return {
        "seat": 0, "team_id": 7, "team_name": "A",
        "submission_id": 11, "rank": 1, "role": "active_best",
    }


def test_iter_transitions_uses_pre_action_state_and_next_recorded_action():
    replay = {"steps": [
        [_agent(_obs(0, 0), {"farmer": ["PASS"]}), _agent(_obs(0, 1), {})],
        [_agent(_obs(1, 1), {"farmer": ["EAST"], "hands": [], "market": []}), _agent(_obs(1, 1), {})],
        [_agent(_obs(2, 1, 90), {"farmer": ["PASS"], "hands": [], "market": [["BUY_SEED", "WHEAT", 1]]}), _agent(_obs(2, 1), {})],
    ]}
    rows = list(iter_transitions(replay, _episode_meta(), _source()))
    assert len(rows) == 2
    row = rows[0]
    assert row["episode_id"] == 99 and row["seat"] == 0
    assert row["submission_id"] == 11
    assert row["replay_file_sha256"] == "file-sha"
    assert row["replay_json_sha256"] == "json-sha"
    assert row["state"]["own"]["farmer"] == [0, 0]
    assert row["next_state"]["own"]["farmer"] == [1, 0]
    assert row["raw_action"]["farmer"] == ["EAST"]
    assert row["effects"]["unit_position_delta"]["farmer"] == [1, 0]


def test_temporal_split_is_per_submission_identity():
    records = []
    for submission_id in (11, 22):
        for episode_id in range(1, 7):
            records.append({
                "team_id": 7,
                "submission_id": submission_id,
                "episode_id": submission_id * 100 + episode_id,
                "create_time": f"2026-09-{10 + episode_id:02d}T00:00:00+00:00",
            })
    splits = assign_temporal_splits(records)
    for submission_id in (11, 22):
        base = submission_id * 100
        assert splits[(7, submission_id, base + 1)] == "train"
        assert splits[(7, submission_id, base + 2)] == "train"
        assert splits[(7, submission_id, base + 3)] == "val"
        assert splits[(7, submission_id, base + 4)] == "val"
        assert splits[(7, submission_id, base + 5)] == "test"
        assert splits[(7, submission_id, base + 6)] == "test"


def test_arrow_table_keeps_variable_binary_and_json_fields():
    replay = {"steps": [
        [_agent(_obs(0, 0), {}), _agent(_obs(0, 1), {})],
        [_agent(_obs(1, 1), {"farmer": ["EAST"], "hands": [], "market": []}), _agent(_obs(1, 1), {})],
    ]}
    row = list(iter_transitions(replay, _episode_meta(), _source()))[0]
    row["split"] = "train"
    table = build_arrow_table([row])
    assert table.num_rows == 1
    assert table.schema.field("state_zlib").type == __import__("pyarrow").binary()
    state = decode_zlib_json(table.column("state_zlib")[0].as_py())
    assert state["own"]["farmer"] == [0, 0]
    assert json.loads(table.column("canonical_action_json")[0].as_py())["farmer"]["op"] == "EAST"


def test_iter_transitions_normalizes_each_observation_only_once(monkeypatch):
    import kaggrl.v2_dataset as dataset_module

    real = dataset_module.normalize_observation
    calls = []

    def counted(obs):
        calls.append(int(obs.get("step", -1)))
        return real(obs)

    monkeypatch.setattr(dataset_module, "normalize_observation", counted)
    replay = {"steps": [
        [_agent(_obs(0, 0), {}), _agent(_obs(0, 1), {})],
        [_agent(_obs(1, 1), {"farmer": ["EAST"], "hands": [], "market": []}), _agent(_obs(1, 1), {})],
        [_agent(_obs(2, 1), {"farmer": ["PASS"], "hands": [], "market": []}), _agent(_obs(2, 1), {})],
    ]}
    assert len(list(iter_transitions(replay, _episode_meta(), _source()))) == 2
    assert calls == [0, 1, 2]
