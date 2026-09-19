from kaggrl.top_tier_dataset import extract_expert_samples


def _agent(obs, action):
    return {"observation": obs, "action": action, "reward": 0, "status": "ACTIVE"}


def test_extract_uses_next_recorded_action_for_current_observation():
    replay = {
        "info": {"Agents": [
            {"submission_id": 111, "team_id": 1},
            {"submission_id": 222, "team_id": 2},
        ]},
        "steps": [
            [_agent({"player": 0, "step": 0}, {"farmer": ["PASS"]}),
             _agent({"player": 1}, {"farmer": ["PASS"]})],
            [_agent({"player": 0, "step": 1}, {"farmer": ["EAST"]}),
             _agent({"player": 1}, {"farmer": ["WEST"]})],
            [_agent({"player": 0, "step": 2}, {"farmer": ["NORTH"]}),
             _agent({"player": 1}, {"farmer": ["SOUTH"]})],
        ],
    }
    sources = [{"submission_id": 222, "rank": 3, "role": "active_best"}]
    rows = extract_expert_samples(replay, 99, sources)
    assert [r["action"]["farmer"][0] for r in rows] == ["WEST", "SOUTH"]
    assert [r["observation"]["step"] for r in rows] == [0, 1]
    assert all(r["seat"] == 1 for r in rows)


def test_extract_falls_back_to_team_name_when_replay_hides_submission_id():
    replay = {
        "info": {"Agents": [{"Name": "Rival"}, {"Name": "Top Team"}]},
        "steps": [
            [_agent({"player": 0, "step": 0}, {}), _agent({"player": 1}, {})],
            [_agent({"player": 0, "step": 1}, {"farmer": ["EAST"]}),
             _agent({"player": 1}, {"farmer": ["WEST"]})],
        ],
    }
    sources = [{"submission_id": 222, "team_name": "Top Team", "rank": 2, "role": "active_best"}]
    rows = extract_expert_samples(replay, 100, sources)
    assert len(rows) == 1
    assert rows[0]["seat"] == 1
    assert rows[0]["action"]["farmer"] == ["WEST"]
