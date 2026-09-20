import inspect

import numpy as np
import torch

from kaggrl.v4_option_export import export_v4_option_numpy
from kaggrl.v4_option_model import V4OptionPolicy
from kaggrl.v4_option_numpy_runtime import V4OptionNumpyPolicy
from kaggrl.v4_options import StepStrategyContext
from rollout.v4_option_numpy_adapter import V4NumpyOptionAdapter

from test_v2_rollout_agent import _obs


def test_v4_option_numpy_runtime_matches_torch_sequence(tmp_path):
    torch.manual_seed(81)
    model = V4OptionPolicy(
        input_dim=32,
        route_count=3,
        market_mode_count=3,
        hidden_dim=24,
    ).eval()
    path = tmp_path / "v4_options.npz"
    export_v4_option_numpy(
        model,
        path,
        route_ids=(0, 9, 100),
        market_modes=("KEEP_ROUTE", "NO_SPEND", "LIQUIDATE_SHED"),
    )
    runtime = V4OptionNumpyPolicy.load(path)

    rng = np.random.default_rng(82)
    features = rng.normal(size=(1, 7, 32)).astype(np.float32)
    clock_context = rng.normal(
        size=(1, 7, model.clock_dim)
    ).astype(np.float32)
    with torch.no_grad():
        torch_out, torch_state = model.forward_sequence(
            torch.from_numpy(features),
            torch.from_numpy(clock_context),
        )

    state = None
    numpy_rows = []
    for step in range(features.shape[1]):
        output = runtime.step(
            features[0, step], clock_context[0, step], state
        )
        state = output.state
        numpy_rows.append(output)

    np.testing.assert_allclose(
        np.stack([row.route_logits for row in numpy_rows]),
        torch_out["route"][0].numpy(),
        atol=2e-5,
        rtol=2e-5,
    )
    np.testing.assert_allclose(
        np.asarray(
            [row.route_gate_probability for row in numpy_rows],
            dtype=np.float32,
        ),
        torch_out["route_gate"][0].numpy(),
        atol=2e-5,
        rtol=2e-5,
    )
    np.testing.assert_allclose(
        np.stack([row.market_logits for row in numpy_rows]),
        torch_out["market"][0].numpy(),
        atol=2e-5,
        rtol=2e-5,
    )
    np.testing.assert_allclose(
        np.stack([row.phase_logits for row in numpy_rows]),
        torch_out["phase"][0].numpy(),
        atol=2e-5,
        rtol=2e-5,
    )
    np.testing.assert_allclose(
        np.asarray([
            [row.predicted_step_norm, row.predicted_remaining_norm]
            for row in numpy_rows
        ]),
        torch_out["clock"][0].numpy(),
        atol=2e-5,
        rtol=2e-5,
    )
    np.testing.assert_allclose(
        state.h,
        torch_state[0][0, 0].numpy(),
        atol=2e-5,
        rtol=2e-5,
    )
    np.testing.assert_allclose(
        state.c,
        torch_state[1][0, 0].numpy(),
        atol=2e-5,
        rtol=2e-5,
    )


def _clock_aware_model():
    model = V4OptionPolicy(
        input_dim=1024,
        route_count=3,
        market_mode_count=3,
        hidden_dim=16,
    ).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.route_head.bias[1] = 10.0
        model.market_head.bias[0] = 10.0
        model.phase_head.bias[2] = 10.0
    return model


def test_numpy_option_adapter_accepts_only_when_internal_clock_is_consistent(
    tmp_path,
):
    path = tmp_path / "clock_aware.npz"
    export_v4_option_numpy(
        _clock_aware_model(),
        path,
        route_ids=(0, 9, 100),
        market_modes=("KEEP_ROUTE", "NO_SPEND", "LIQUIDATE_SHED"),
    )
    adapter = V4NumpyOptionAdapter(
        path,
        max_clock_error_turns=2.0,
        require_phase_match=True,
        route_switch_steps=(360,),
    )

    obs = _obs(360, hands=0)
    context = StepStrategyContext.from_observation(
        obs,
        {"episodeSteps": 720, "turnsPerDay": 24},
        base_route_id=100,
        route_ids=(0, 9, 100),
    )
    option, confidence = adapter(obs, None, context)
    assert option.route_id == 9
    assert option.market_mode == "KEEP_ROUTE"
    assert confidence > 0.99
    assert adapter.last_metadata["clock_ok"] is True
    assert adapter.last_metadata["phase_match"] is True
    assert adapter.last_metadata["accepted"] is True

    # Same network claims phase=mid, so liquidation phase must be rejected.
    late = _obs(680, hands=0)
    late_context = StepStrategyContext.from_observation(
        late,
        {"episodeSteps": 720, "turnsPerDay": 24},
        base_route_id=2,
        route_ids=(0, 9, 100),
    )
    option, confidence = adapter(late, None, late_context)
    assert option is None
    assert confidence == 0.0
    assert adapter.last_metadata["accepted"] is False
    assert adapter.last_metadata["phase_match"] is False


def test_v4_option_submission_runtime_import_graph_is_torch_free():
    import kaggrl.v4_option_numpy_runtime as runtime_module
    import rollout.v4_option_numpy_adapter as adapter_module

    source = (
        inspect.getsource(runtime_module)
        + inspect.getsource(adapter_module)
    )
    assert "import torch" not in source
    assert "from torch" not in source


def test_route_gate_blocks_ambiguous_route_switch_but_keeps_market_option(
    tmp_path,
):
    model = _clock_aware_model()
    with torch.no_grad():
        model.route_gate_head.bias.fill_(-10.0)
    path = tmp_path / "route_gate_off.npz"
    export_v4_option_numpy(
        model,
        path,
        route_ids=(0, 9, 100),
        market_modes=("KEEP_ROUTE", "NO_SPEND", "LIQUIDATE_SHED"),
    )
    adapter = V4NumpyOptionAdapter(path)
    obs = _obs(360, hands=0)
    context = StepStrategyContext.from_observation(
        obs,
        {"episodeSteps": 720, "turnsPerDay": 24},
        base_route_id=100,
        route_ids=(0, 9, 100),
    )
    option, confidence = adapter(obs, None, context)
    assert option is None
    assert confidence == 0.0
    assert adapter.last_metadata["route_switch_allowed"] is False
    assert adapter.last_metadata["changed"] is False


def test_numpy_adapter_masks_route_to_context_candidates(tmp_path):
    model = V4OptionPolicy(
        input_dim=1024,
        route_count=3,
        market_mode_count=3,
        hidden_dim=16,
    ).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        # Global argmax would be route 100, but context forbids it.
        model.route_head.bias[:] = torch.tensor([0.0, 10.0, 100.0])
        model.route_gate_head.bias.fill_(10.0)
        model.market_head.bias[0] = 10.0
        # Step 360 is phase=mid.
        model.phase_head.bias[2] = 10.0
    path = tmp_path / "route_mask.npz"
    export_v4_option_numpy(
        model,
        path,
        route_ids=(0, 9, 100),
        market_modes=("KEEP_ROUTE", "NO_SPEND", "LIQUIDATE_SHED"),
    )
    adapter = V4NumpyOptionAdapter(
        path,
        min_route_probability=0.50,
        route_switch_steps=(360,),
    )
    obs = _obs(360, hands=0)
    context = StepStrategyContext.from_observation(
        obs,
        {"episodeSteps": 720, "turnsPerDay": 24},
        base_route_id=0,
        route_ids=(0, 9),
    )
    option, confidence = adapter(obs, None, context)
    assert option is not None
    assert option.route_id == 9
    assert confidence == 1.0
    assert adapter.last_metadata["route_id"] == 9
    assert adapter.last_metadata["compatible_routes"] == [0, 9]


def test_numpy_adapter_default_blocks_unproven_no_spend_mode(tmp_path):
    model = V4OptionPolicy(
        input_dim=1024,
        route_count=1,
        market_mode_count=3,
        hidden_dim=16,
    ).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.market_head.bias[1] = 10.0  # NO_SPEND raw prediction
        model.phase_head.bias[1] = 10.0   # growth
    path = tmp_path / "safe_market_default.npz"
    export_v4_option_numpy(
        model,
        path,
        route_ids=(0,),
        market_modes=("KEEP_ROUTE", "NO_SPEND", "LIQUIDATE_SHED"),
    )
    adapter = V4NumpyOptionAdapter(path)
    obs = _obs(200, hands=0)
    context = StepStrategyContext.from_observation(
        obs,
        {"episodeSteps": 720, "turnsPerDay": 24},
        base_route_id=0,
        route_ids=(0,),
    )
    option, confidence = adapter(obs, None, context)
    assert option is None
    assert confidence == 0.0
    assert adapter.last_metadata["market_mode"] == "KEEP_ROUTE"
    assert adapter.last_metadata["changed"] is False
