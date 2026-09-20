from copy import deepcopy
import inspect

import torch

from kaggrl.v2_export import export_v2_numpy
from kaggrl.v2_model import RecurrentIntentPolicy
from rollout.v2_agent_numpy import V2NumpyRolloutAgent


def _tiles():
    return [[None for _ in range(10)] for _ in range(10)]


def _obs(step: int, hands: int = 1):
    day, hour = divmod(step, 24)
    positions = [[5, 4] for _ in range(hands)]
    return {
        "player": 0, "step": step, "day": day, "hour": hour,
        "farms": [
            {"money": 3000, "farmer": [4, 4], "hands": positions,
             "hires_today": hands, "unlocked_quadrants": ["NW"], "tiles": _tiles()},
            {"money": 2500, "farmer": [8, 8], "hands": [], "hires_today": 0,
             "unlocked_quadrants": ["NW"], "tiles": _tiles()},
        ],
        "private": {"shed": {}, "seeds": {},
                    "inventories": [{} for _ in range(hands + 1)]},
        "market": {"inventory": {"WHEAT": 10000}, "prices": {"WHEAT": 25}},
        "town": {"unlocked_shops": []},
    }


def _model(tmp_path):
    torch.manual_seed(123)
    model = RecurrentIntentPolicy().eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    path = tmp_path / "policy.npz"
    export_v2_numpy(model, path)
    return path


def test_rollout_agent_resets_only_on_new_episode_zero_and_not_at_day_boundary(tmp_path):
    agent = V2NumpyRolloutAgent(_model(tmp_path), seed=17, deterministic=True)
    first = agent(_obs(0, hands=0))
    assert first["hands"] == []
    assert agent.reset_count == 1 and agent.episode_index == 0
    agent(_obs(1, hands=1)); agent(_obs(2, hands=1))
    assert agent.last_effect is not None
    before = agent.reset_count
    agent(_obs(23, hands=1)); agent(_obs(24, hands=0))
    assert agent.reset_count == before
    action = agent(_obs(0, hands=2))
    assert agent.reset_count == before + 1
    assert agent.episode_index == 1
    assert agent.last_metadata["previous_effect"] == {}
    assert agent.last_metadata["reset"] is True
    assert len(action["hands"]) == 2
    assert agent.recurrent_state is not None


def test_rollout_agent_tracks_effect_from_previous_requested_action(tmp_path):
    agent = V2NumpyRolloutAgent(_model(tmp_path), seed=9, deterministic=True)
    before = _obs(0, hands=0)
    agent(before)
    after = deepcopy(_obs(1, hands=0))
    after["farms"][0]["money"] = 2999
    agent(after)
    assert agent.last_effect is not None
    assert agent.last_effect.transition.money_delta == -1
    assert agent.last_metadata["step"] == 1
    assert "logp" in agent.last_metadata and "terminal_money" in agent.last_metadata


def test_rollout_wrapper_import_graph_is_torch_free():
    import rollout.v2_agent_numpy as module
    source = inspect.getsource(module)
    assert "import torch" not in source
    assert "from torch" not in source


class _KaggleLikeStruct(dict):
    def __init__(self, **entries):
        super().__init__(entries)
        self.__dict__ = self


def _structify(value):
    if isinstance(value, dict):
        return _KaggleLikeStruct(**{key: _structify(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_structify(item) for item in value]
    return value


def test_rollout_agent_accepts_kaggle_struct_observation(tmp_path):
    agent = V2NumpyRolloutAgent(_model(tmp_path), seed=21, deterministic=True)
    action = agent(_structify(_obs(0, hands=1)))
    assert set(action) == {"farmer", "hands", "market"}
    assert len(agent.telemetry) == 1


def test_v2_rollout_reconstructs_missing_step_for_lifecycle(tmp_path):
    agent = V2NumpyRolloutAgent(_model(tmp_path), seed=25, deterministic=True)
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
    assert agent.reset_count == 1
