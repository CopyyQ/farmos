import numpy as np

from kaggrl.structured_actor_critic import StructuredActorCritic, export_numpy
from kaggrl.rollout_agent import NumpyRolloutAgent


def _obs(step=0, player=0):
    empty = [[None for _ in range(4)] for _ in range(4)]
    farms = [
        {"money": 3000, "farmer": [0, 0], "hands": [], "tiles": empty, "unlocked_quadrants": []},
        {"money": 3000, "farmer": [0, 0], "hands": [], "tiles": empty, "unlocked_quadrants": []},
    ]
    return {
        "player": player, "step": step, "day": step // 24, "hour": step % 24,
        "farms": farms, "private": {"shed": {}, "seeds": {}, "inventories": [{}]},
        "market": {"inventory": {}, "prices": {}}, "town": {"unlocked_shops": []},
    }


def test_numpy_rollout_agent_resets_state_and_records_semantic_sample(tmp_path):
    model = StructuredActorCritic(1024, 16)
    model_path = tmp_path / "model.npz"
    export_numpy(model, model_path)
    agent = NumpyRolloutAgent(model_path, seed=9, deterministic=False)
    action0 = agent(_obs(0))
    action1 = agent(_obs(1))
    assert set(action0) == {"farmer", "hands", "market"}
    assert set(action1) == set(action0)
    assert len(agent.records) == 2
    assert np.isfinite(agent.records[-1]["logp"])
    assert np.isfinite(agent.records[-1]["value"])
    assert agent.records[-1]["mask"].sum() > 0
    agent(_obs(0))
    assert agent.last_step == 0
    assert len(agent.records) == 1

def test_rollout_agent_resets_only_recurrent_state_at_training_horizon(tmp_path):
    model = StructuredActorCritic(1024, 16)
    model_path = tmp_path / "model.npz"
    export_numpy(model, model_path)
    agent = NumpyRolloutAgent(model_path, seed=3, deterministic=True, memory_horizon=48)
    agent(_obs(0)); agent(_obs(1))
    assert agent.state_reset_count == 1
    agent(_obs(48))
    assert agent.state_reset_count == 2
    assert len(agent.records) == 3
    assert agent.last_step == 48