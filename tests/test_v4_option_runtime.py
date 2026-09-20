import torch

from kaggrl.clock import CLOCK_FEATURES
from kaggrl.v4_option_model import V4OptionPolicy
from kaggrl.v4_option_runtime import TorchV4OptionRuntime
from kaggrl.v4_options import MARKET_MODES, StepStrategyContext


def _obs(step=144):
    day, hour = divmod(step, 24)
    tiles = [[None for _ in range(10)] for _ in range(10)]
    return {
        "player": 0,
        "step": step,
        "day": day,
        "hour": hour,
        "farms": [
            {
                "money": 3000,
                "farmer": [4, 4],
                "hands": [],
                "hires_today": 0,
                "unlocked_quadrants": ["NW"],
                "tiles": tiles,
            },
            {
                "money": 2500,
                "farmer": [8, 8],
                "hands": [],
                "hires_today": 0,
                "unlocked_quadrants": ["NW"],
                "tiles": tiles,
            },
        ],
        "private": {"shed": {}, "seeds": {}, "inventories": [{}]},
        "market": {
            "inventory": {"WHEAT": 10000},
            "prices": {"WHEAT": 25},
        },
        "town": {"unlocked_shops": ["YARN_STORE", "BAKERY"]},
    }


def _checkpoint(tmp_path):
    model = V4OptionPolicy(
        1024,
        route_count=3,
        market_mode_count=len(MARKET_MODES),
        hidden_dim=16,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        # Raw logits prefer route 100, which the context below forbids.
        model.route_head.bias[:] = torch.tensor([0.0, 1.0, 100.0])
        model.route_gate_head.bias.fill_(10.0)
        model.market_head.bias[:] = torch.tensor(
            [10.0] + [0.0] * (len(MARKET_MODES) - 1)
        )
    path = tmp_path / "v4.pt"
    torch.save({
        "architecture_version": model.ARCHITECTURE_VERSION,
        "model_state": model.state_dict(),
        "input_dim": 1024,
        "hidden_dim": 16,
        "clock_dim": len(CLOCK_FEATURES),
        "clock_features": list(CLOCK_FEATURES),
        "route_ids": [9, 119, 100],
        "market_modes": list(MARKET_MODES),
        "observation_schema": "macro_semantic_v4_clock_v2",
        "route_gate_threshold": 0.5,
    }, path)
    return path


def test_runtime_masks_incompatible_route_before_argmax(tmp_path):
    runtime = TorchV4OptionRuntime(
        _checkpoint(tmp_path),
        min_route_probability=0.50,
        min_market_probability=0.90,
    )
    context = StepStrategyContext.from_observation(
        _obs(),
        None,
        base_route_id=9,
        route_ids=(9, 119),
    )
    result = runtime(_obs(), None, context)
    assert result is not None
    option, confidence = result
    assert confidence == 1.0
    assert option.route_id == 119
    assert runtime.last_decision["compatible_routes"] == [9, 119]


def test_runtime_resets_recurrent_state_on_temporal_jump(tmp_path):
    runtime = TorchV4OptionRuntime(_checkpoint(tmp_path))
    context = StepStrategyContext.from_observation(
        _obs(144), None, base_route_id=9, route_ids=(9, 119)
    )
    runtime(_obs(144), None, context)
    first_state = runtime.state
    assert first_state is not None

    jumped = _obs(200)
    jumped_context = StepStrategyContext.from_observation(
        jumped, None, base_route_id=9, route_ids=(9, 119)
    )
    runtime(jumped, None, jumped_context)
    assert runtime.last_decision["step"] == 200
    assert runtime.state is not None


def test_torch_runtime_default_does_not_apply_unproven_market_override(tmp_path):
    model = V4OptionPolicy(
        1024,
        route_count=1,
        market_mode_count=len(MARKET_MODES),
        hidden_dim=16,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.market_head.bias[:] = torch.tensor(
            [0.0, 10.0] + [0.0] * (len(MARKET_MODES) - 2)
        )
        model.phase_head.bias[1] = 10.0
    path = tmp_path / "safe_default.pt"
    torch.save({
        "architecture_version": model.ARCHITECTURE_VERSION,
        "model_state": model.state_dict(),
        "input_dim": 1024,
        "hidden_dim": 16,
        "clock_dim": len(CLOCK_FEATURES),
        "clock_features": list(CLOCK_FEATURES),
        "route_ids": [0],
        "market_modes": list(MARKET_MODES),
        "observation_schema": "macro_semantic_v4_clock_v2",
        "route_gate_threshold": 0.5,
    }, path)
    runtime = TorchV4OptionRuntime(path)
    obs = _obs(200)
    context = StepStrategyContext.from_observation(
        obs, None, base_route_id=0, route_ids=(0,)
    )
    result = runtime(obs, None, context)
    assert result is None
    assert runtime.last_decision["market_mode"] == "KEEP_ROUTE"
    assert runtime.last_decision["changed"] is False


def test_margin_q_blocks_high_confidence_route_that_loses_money(tmp_path):
    model = V4OptionPolicy(
        1024,
        route_count=3,
        market_mode_count=len(MARKET_MODES),
        hidden_dim=16,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.route_head.bias[:] = torch.tensor([0.0, 10.0, -10.0])
        model.route_gate_head.bias.fill_(10.0)
        # route_ids = [9, 119, 100], base=9; candidate=119
        model.route_value_head.bias[:] = torch.tensor([1.0, 0.5, 0.0])
        model.market_head.bias[0] = 10.0
    path = tmp_path / "margin_q_block.pt"
    torch.save({
        "architecture_version": model.ARCHITECTURE_VERSION,
        "model_state": model.state_dict(),
        "input_dim": 1024,
        "hidden_dim": 16,
        "clock_dim": len(CLOCK_FEATURES),
        "clock_features": list(CLOCK_FEATURES),
        "route_ids": [9, 119, 100],
        "market_modes": list(MARKET_MODES),
        "observation_schema": "macro_semantic_v4_clock_v2",
        "route_gate_threshold": 0.5,
        "objective_version": "terminal_margin_advantage_weighted_bc_v1",
        "margin_scale": 10000.0,
    }, path)
    runtime = TorchV4OptionRuntime(
        path,
        min_route_probability=0.5,
        route_switch_steps=(144,),
        min_route_margin_advantage=500.0,
    )
    obs = _obs(144)
    context = StepStrategyContext.from_observation(
        obs, None, base_route_id=9, route_ids=(9, 119)
    )
    assert runtime(obs, None, context) is None
    assert runtime.last_decision["route_value_ok"] is False
    assert runtime.last_decision["route_margin_advantage"] < 0.0


def test_margin_q_allows_route_only_when_predicted_margin_improves(tmp_path):
    model = V4OptionPolicy(
        1024,
        route_count=3,
        market_mode_count=len(MARKET_MODES),
        hidden_dim=16,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.route_head.bias[:] = torch.tensor([0.0, 10.0, -10.0])
        model.route_gate_head.bias.fill_(10.0)
        model.route_value_head.bias[:] = torch.tensor([0.1, 0.3, 0.0])
        model.market_head.bias[0] = 10.0
    path = tmp_path / "margin_q_allow.pt"
    torch.save({
        "architecture_version": model.ARCHITECTURE_VERSION,
        "model_state": model.state_dict(),
        "input_dim": 1024,
        "hidden_dim": 16,
        "clock_dim": len(CLOCK_FEATURES),
        "clock_features": list(CLOCK_FEATURES),
        "route_ids": [9, 119, 100],
        "market_modes": list(MARKET_MODES),
        "observation_schema": "macro_semantic_v4_clock_v2",
        "route_gate_threshold": 0.5,
        "objective_version": "terminal_margin_advantage_weighted_bc_v1",
        "margin_scale": 10000.0,
    }, path)
    runtime = TorchV4OptionRuntime(
        path,
        min_route_probability=0.5,
        route_switch_steps=(144,),
        min_route_margin_advantage=500.0,
    )
    obs = _obs(144)
    context = StepStrategyContext.from_observation(
        obs, None, base_route_id=9, route_ids=(9, 119)
    )
    option, confidence = runtime(obs, None, context)
    assert confidence == 1.0
    assert option.route_id == 119
    assert runtime.last_decision["route_value_ok"] is True
    assert runtime.last_decision["route_margin_advantage"] > 500.0


def test_counterfactual_q_market_override_only_runs_on_validated_steps(tmp_path):
    model = V4OptionPolicy(
        1024,
        route_count=1,
        market_mode_count=len(MARKET_MODES),
        hidden_dim=16,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.market_value_head.bias[:] = torch.tensor(
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        )
        model.market_head.bias[0] = 10.0
    path = tmp_path / "q_steps.pt"
    torch.save({
        "architecture_version": model.ARCHITECTURE_VERSION,
        "model_state": model.state_dict(),
        "input_dim": 1024,
        "hidden_dim": 16,
        "clock_dim": len(CLOCK_FEATURES),
        "clock_features": list(CLOCK_FEATURES),
        "route_ids": [0],
        "market_modes": list(MARKET_MODES),
        "observation_schema": "macro_semantic_v4_clock_v2",
        "route_gate_threshold": 0.5,
        "objective_version": "terminal_margin_advantage_weighted_bc_v1",
        "margin_scale": 10000.0,
        "counterfactual_q_schema": "farmos_v4_counterfactual_q_v1",
        "counterfactual_q_steps": [648],
    }, path)
    runtime = TorchV4OptionRuntime(
        path,
        allowed_market_modes=("KEEP_ROUTE", "HOLD_SALES"),
        min_market_margin_advantage=500.0,
    )

    obs = _obs(647)
    context = StepStrategyContext.from_observation(
        obs, None, base_route_id=0, route_ids=(0,)
    )
    assert runtime(obs, None, context) is None
    assert runtime.last_decision["market_mode"] == "KEEP_ROUTE"
    assert runtime.last_decision["market_q_step_ok"] is False

    obs = _obs(648)
    context = StepStrategyContext.from_observation(
        obs, None, base_route_id=0, route_ids=(0,)
    )
    option, confidence = runtime(obs, None, context)
    assert confidence == 1.0
    assert option.market_mode == "HOLD_SALES"
    assert runtime.last_decision["market_q_step_ok"] is True
