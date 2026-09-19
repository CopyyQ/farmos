import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
from test_v2_numpy_runtime import (
    _row,
    _strip_raw,
    _torch_canonical,
    _torch_row_logp,
)

from kaggrl.v2_tensorize import collate_transitions
from kaggrl.v3_export import export_v3_numpy
from kaggrl.v3_model import TemporalIntentPolicy
from kaggrl.v3_numpy_runtime import V3NumpyPolicy


def test_v3_numpy_runtime_matches_torch_through_ring_rollover(tmp_path):
    torch.manual_seed(37)
    model = TemporalIntentPolicy().eval()
    path = tmp_path / "v3_policy.npz"
    export_v3_numpy(model, path)
    runtime = V3NumpyPolicy.load(path)
    tstate = None
    nstate = None
    for step in range(40):
        batch = collate_transitions([_row(2)])
        with torch.no_grad():
            expected = model.sample_step(
                batch, tstate, np.random.default_rng(17), deterministic=True,
            )
        got = runtime.step(
            batch.structured_states[0],
            {"money_delta": -20, "hand_count_delta": 1},
            batch.previous_actions[0],
            nstate,
            np.random.default_rng(17),
            deterministic=True,
        )
        state = expected.temporal_state
        assert np.allclose(got.recurrent_state.h, state.h[0].numpy(), atol=3e-5, rtol=3e-5)
        assert np.allclose(got.recurrent_state.c, state.c[0].numpy(), atol=3e-5, rtol=3e-5)
        assert np.allclose(got.recurrent_state.memory, state.memory[0].numpy(), atol=3e-5, rtol=3e-5)
        assert got.recurrent_state.valid_length == int(state.valid_length[0])
        assert got.recurrent_state.write_pos == int(state.write_pos[0])
        assert np.allclose(got.fused_temporal, expected.fused_temporal[0].numpy(), atol=3e-5, rtol=3e-5)
        assert np.allclose(got.intent, expected.intent[0].numpy(), atol=3e-5, rtol=3e-5)
        diag = expected.temporal_diagnostics
        assert np.allclose(got.attention_weights, diag.attention_weights[0].numpy(), atol=3e-5, rtol=3e-5)
        assert np.allclose(got.attention_entropy, diag.attention_entropy[0].numpy(), atol=3e-5, rtol=3e-5)
        assert np.allclose(got.mean_attended_age, diag.mean_attended_age[0].numpy(), atol=3e-5, rtol=3e-5)
        assert _strip_raw(got.canonical_action) == _strip_raw(_torch_canonical(expected.rows[0]))
        assert abs(got.logp - float(_torch_row_logp(model, expected.rows[0]))) < 5e-4
        tstate = state
        nstate = got.recurrent_state
    assert nstate.valid_length == 32
    assert nstate.write_pos == 8


def test_v3_runtime_reset_state_is_empty(tmp_path):
    torch.manual_seed(41)
    path = tmp_path / "v3_policy.npz"
    export_v3_numpy(TemporalIntentPolicy().eval(), path)
    state = V3NumpyPolicy.load(path).initial_state()
    assert state.valid_length == 0
    assert state.write_pos == 0
    assert np.count_nonzero(state.h) == 0
    assert np.count_nonzero(state.c) == 0
    assert np.count_nonzero(state.memory) == 0


def test_v3_numpy_runtime_import_graph_is_torch_free():
    import kaggrl.v3_numpy_runtime as runtime_module
    source = inspect.getsource(runtime_module)
    assert "import torch" not in source
    assert "from torch" not in source


def test_v3_numpy_loader_rejects_tampered_parameter_archive(tmp_path):
    torch.manual_seed(43)
    path = tmp_path / "v3_policy.npz"
    export_v3_numpy(TemporalIntentPolicy().eval(), path)
    data = np.load(path, allow_pickle=False)
    arrays = {key: data[key].copy() for key in data.files}
    key = next(name for name in sorted(arrays) if name.startswith("p__"))
    arrays[key].reshape(-1)[0] += np.float32(0.125)
    tampered = tmp_path / "tampered.npz"
    np.savez_compressed(tampered, **arrays)
    with pytest.raises(ValueError, match="parameter hash"):
        V3NumpyPolicy.load(tampered)


def test_v3_numpy_trace_step_exposes_raw_and_masked_unit_decision(tmp_path):
    torch.manual_seed(47)
    model = TemporalIntentPolicy().eval()
    path = tmp_path / "v3_policy.npz"
    export_v3_numpy(model, path)
    runtime = V3NumpyPolicy.load(path)
    batch = collate_transitions([_row(2)])
    traced = runtime.trace_step(
        batch.structured_states[0], {}, batch.previous_actions[0], None,
        np.random.default_rng(17), deterministic=True,
    )
    farmer = traced["decisions"][0]
    assert farmer["actor"] == "farmer"
    assert len(farmer["raw_logits"]) == len(farmer["legal_mask"])
    assert len(farmer["masked_logits"]) == len(farmer["raw_logits"])
    assert farmer["chosen_op"] in farmer["ops"]
    assert farmer["decision_reason"] == "model_argmax"


def test_v3_numpy_trace_step_returns_same_runtime_output(tmp_path):
    torch.manual_seed(48)
    path = tmp_path / "v3_policy.npz"
    export_v3_numpy(TemporalIntentPolicy().eval(), path)
    runtime = V3NumpyPolicy.load(path)
    batch = collate_transitions([_row(1)])
    traced = runtime.trace_step(
        batch.structured_states[0], {}, batch.previous_actions[0], None,
        np.random.default_rng(19), deterministic=True,
    )
    assert traced["output"].canonical_action == traced["canonical_action"]
    assert traced["output"].engine_action == traced["engine_action"]


def test_strategy_numpy_runtime_matches_torch_for_fixed_slot(tmp_path):
    torch.manual_seed(49)
    model = TemporalIntentPolicy(strategy_count=3).eval()
    path = tmp_path / "v3_1_strategy_policy.npz"
    export_v3_numpy(model, path, default_strategy_slot=2)
    runtime = V3NumpyPolicy.load(path)
    batch = collate_transitions([_row(1)])
    slot = torch.tensor([2], dtype=torch.long)
    with torch.no_grad():
        expected = model.sample_step(
            batch, None, np.random.default_rng(21), deterministic=True,
            strategy_slots=slot,
        )
    got = runtime.step(
        batch.structured_states[0],
        {"money_delta": -20, "hand_count_delta": 1},
        batch.previous_actions[0], None,
        np.random.default_rng(21), deterministic=True, strategy_slot=2,
    )
    assert runtime.strategy_count == 3
    assert runtime.default_strategy_slot == 2
    assert np.allclose(got.intent, expected.intent[0].numpy(), atol=3e-5, rtol=3e-5)
    assert _strip_raw(got.canonical_action) == _strip_raw(_torch_canonical(expected.rows[0]))
