from copy import deepcopy
import inspect
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from test_v2_rollout_agent import _obs

from kaggrl.v3_export import export_v3_numpy
from kaggrl.v3_model import TemporalIntentPolicy
from rollout.v3_agent_numpy import V3NumpyRolloutAgent


def _model(tmp_path):
    torch.manual_seed(55)
    model = TemporalIntentPolicy().eval()
    path = tmp_path / "v3_policy.npz"
    export_v3_numpy(model, path)
    return path


def _state_tuple(agent):
    state = agent.recurrent_state
    return (state.h.copy(), state.c.copy(), state.memory.copy(),
            state.valid_length, state.write_pos)


def test_second_game_first_step_matches_fresh_agent(tmp_path):
    path = _model(tmp_path)
    agent = V3NumpyRolloutAgent(path, seed=17, deterministic=True)
    agent(_obs(0, hands=1)); agent(_obs(1, hands=1))
    second_action = agent(_obs(0, hands=1))
    second_state = _state_tuple(agent)

    fresh = V3NumpyRolloutAgent(path, seed=17, deterministic=True)
    fresh_action = fresh(_obs(0, hands=1))
    fresh_state = _state_tuple(fresh)
    assert second_action == fresh_action
    for left, right in zip(second_state[:3], fresh_state[:3]):
        assert np.allclose(left, right)
    assert second_state[3:] == fresh_state[3:]
    assert agent.last_metadata["reset"] is True


def test_player_change_and_non_monotonic_step_reset_temporal_state(tmp_path):
    agent = V3NumpyRolloutAgent(_model(tmp_path), seed=19, deterministic=True)
    agent(_obs(0)); agent(_obs(1))
    before = agent.reset_count
    changed = deepcopy(_obs(2)); changed["player"] = 1
    agent(changed)
    assert agent.reset_count == before + 1
    agent(_obs(5)); before = agent.reset_count
    agent(_obs(4))
    assert agent.reset_count == before + 1
    assert agent.recurrent_state.valid_length == 1
    assert agent.last_metadata["temporal_valid_length"] == 1
    assert "temporal_write_pos" in agent.last_metadata
    assert "effective_families" in agent.last_metadata


def test_v3_rollout_tracks_effect_and_temporal_telemetry(tmp_path):
    agent = V3NumpyRolloutAgent(_model(tmp_path), seed=23, deterministic=True)
    agent(_obs(0, hands=0))
    after = deepcopy(_obs(1, hands=0))
    after["farms"][0]["money"] = 2999
    agent(after)
    assert agent.last_effect is not None
    assert agent.last_effect.transition.money_delta == -1
    assert agent.last_metadata["temporal_valid_length"] == 2
    assert 0 <= agent.last_metadata["temporal_write_pos"] < 32


def test_v3_rollout_import_graph_is_torch_free():
    import rollout.v3_agent_numpy as module
    source = inspect.getsource(module)
    assert "import torch" not in source
    assert "from torch" not in source


def test_v3_rollout_can_capture_live_decision_fixture(tmp_path):
    agent = V3NumpyRolloutAgent(
        _model(tmp_path), seed=29, deterministic=True,
        capture_decision_trace=True,
    )
    action = agent(_obs(0, hands=1))
    assert action == agent.last_metadata["engine_action"]
    assert len(agent.diagnostic_fixtures) == 1
    fixture = agent.diagnostic_fixtures[0]
    assert fixture["step"] == 0
    assert fixture["observation"]["step"] == 0
    assert fixture["structured_state"]["step"] == 0
    assert fixture["decisions"][0]["actor"] == "farmer"
    assert fixture["reset"] is True


def test_strategy_rollout_uses_fixed_override_slot(tmp_path):
    from dataclasses import asdict
    from kaggrl.v2_observation import normalize_observation
    from kaggrl.v3_numpy_runtime import V3NumpyPolicy

    torch.manual_seed(57)
    model = TemporalIntentPolicy(strategy_count=2).eval()
    path = tmp_path / "v3_strategy_policy.npz"
    export_v3_numpy(model, path, default_strategy_slot=0)
    observation = _obs(0, hands=1)
    structured = asdict(normalize_observation(observation))
    expected = V3NumpyPolicy.load(path).step(
        structured, {}, None, None, np.random.default_rng(31),
        deterministic=True, strategy_slot=1,
    )
    agent = V3NumpyRolloutAgent(
        path, seed=31, deterministic=True, strategy_slot=1,
    )
    action = agent(observation)
    assert action == expected.engine_action
    assert agent.last_metadata["strategy_slot"] == 1
    assert agent.strategy_slot == 1


def test_v3_rollout_reconstructs_missing_step_for_temporal_lifecycle(tmp_path):
    agent = V3NumpyRolloutAgent(
        _model(tmp_path), seed=37, deterministic=True,
        capture_decision_trace=True,
    )
    first = _obs(0, hands=0)
    second = _obs(1, hands=0)
    first.pop("step")
    second.pop("step")
    agent(first)
    agent(second)
    assert agent.last_metadata["step"] == 1
    assert agent.last_metadata["day"] == 0
    assert agent.last_metadata["hour"] == 1
    assert agent.last_metadata["remaining_steps"] == 718
    assert agent.last_metadata["temporal_valid_length"] == 2
    assert agent.diagnostic_fixtures[-1]["structured_state"]["step"] == 1
