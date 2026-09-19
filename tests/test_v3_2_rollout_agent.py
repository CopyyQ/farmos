import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from test_v2_rollout_agent import _obs

from kaggrl.v3_2_export import export_v3_2_numpy
from kaggrl.v3_2_model import TemporalIntentPolicyV32
from rollout.v3_2_agent_numpy import V32NumpyRolloutAgent


def test_v32_rollout_loads_v32_policy_and_uses_fixed_strategy_slot(tmp_path):
    torch.manual_seed(111)
    model = TemporalIntentPolicyV32(strategy_count=2).eval()
    path = tmp_path / "v32_policy.npz"
    export_v3_2_numpy(model, path, default_strategy_slot=0)

    agent = V32NumpyRolloutAgent(
        path, seed=31, deterministic=True, strategy_slot=1,
    )
    action = agent(_obs(0, hands=1))

    assert isinstance(action, dict)
    assert agent.policy.architecture_version == model.ARCHITECTURE_VERSION
    assert agent.strategy_slot == 1
    assert agent.last_metadata["strategy_slot"] == 1
