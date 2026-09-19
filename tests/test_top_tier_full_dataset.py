import datetime as dt
import numpy as np

from kaggrl.top_tier_full_dataset import (
    assign_temporal_splits,
    compute_source_weight,
    iter_actor_examples,
)


class DummyEncoder:
    def encode(self, observation):
        return np.asarray([observation["step"], observation["player"]], dtype=np.float32)


class DummyCodec:
    OPS = {"PASS": 0, "EAST": 1, "WEST": 2, "NORTH": 3}

    def encode(self, action, hand_count):
        return np.asarray([self.OPS[action["farmer"][0]], hand_count], dtype=np.int16)

    def supervision_mask(self, action, hand_count):
        return np.asarray([1, 1], dtype=np.float32)

def _agent(obs, action):
    return {"observation": obs, "action": action, "reward": 0, "status": "ACTIVE"}


def _replay():
    return {
        "info": {"Agents": [{"Name": "Top Team"}, {"Name": "Rival"}]},
        "steps": [
            [_agent({"player": 0, "step": 0, "farms": [{"hands": []}]}, {"farmer": ["PASS"], "hands": [], "market": []}),
             _agent({"player": 1}, {"farmer": ["PASS"]})],
            [_agent({"player": 0, "step": 1, "farms": [{"hands": []}]}, {"farmer": ["EAST"], "hands": [], "market": []}),
             _agent({"player": 1}, {"farmer": ["WEST"]})],
            [_agent({"player": 0, "step": 2, "farms": [{"hands": []}]}, {"farmer": ["NORTH"], "hands": [], "market": []}),
             _agent({"player": 1}, {"farmer": ["WEST"]})],
        ],
    }


def test_examples_use_next_step_action():
    source = {"team_name": "Top Team", "submission_id": 7, "rank": 2, "role": "active_best"}
    meta = {"id": 100, "createTime": "2026-09-16T10:00:00Z"}
    rows = list(iter_actor_examples(_replay(), meta, source, DummyEncoder(), DummyCodec()))
    assert [r["step"] for r in rows] == [0, 1]
    assert np.frombuffer(rows[0]["action_i16"], dtype=np.int16)[0] == DummyCodec.OPS["EAST"]
    assert np.frombuffer(rows[1]["action_i16"], dtype=np.int16)[0] == DummyCodec.OPS["NORTH"]

def test_temporal_split_holds_out_newest_episodes_per_team():
    records = []
    for i in range(6):
        records.append({
            "team_name": "A",
            "episode_id": i,
            "create_time": (dt.datetime(2026, 9, 10, tzinfo=dt.timezone.utc) + dt.timedelta(hours=i)).isoformat(),
        })
    splits = assign_temporal_splits(records)
    assert [splits[("A", i)] for i in range(6)] == ["train", "train", "val", "val", "test", "test"]


def test_temporal_split_is_independent_per_team():
    records = []
    for team in ("A", "B"):
        for i in range(5):
            records.append({"team_name": team, "episode_id": i, "create_time": f"2026-09-1{i+1}T00:00:00+00:00"})
    splits = assign_temporal_splits(records)
    assert splits[("A", 4)] == "test"
    assert splits[("B", 4)] == "test"
    assert splits[("A", 0)] == "train"


def test_newer_source_gets_higher_weight():
    assert compute_source_weight(1, "active_best", 2) > compute_source_weight(1, "active_best", 50)
    assert compute_source_weight(1, "latest_qualified", 2) > compute_source_weight(1, "active_best", 2)
    assert compute_source_weight(1, "active_best", 2) > compute_source_weight(10, "active_best", 2)
