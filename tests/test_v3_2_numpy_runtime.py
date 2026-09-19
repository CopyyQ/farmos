import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
from test_v2_numpy_runtime import _row, _strip_raw, _torch_canonical

from kaggrl.v2_tensorize import collate_transitions
from kaggrl.v3_2_export import export_v3_2_numpy
from kaggrl.v3_2_model import TemporalIntentPolicyV32
from kaggrl.v3_2_numpy_runtime import V32NumpyPolicy


def test_v32_numpy_matches_torch_for_fixed_strategy_slot(tmp_path):
    torch.manual_seed(101)
    model = TemporalIntentPolicyV32(strategy_count=3).eval()
    path = tmp_path / "v32_policy.npz"
    export_v3_2_numpy(model, path, default_strategy_slot=2)
    runtime = V32NumpyPolicy.load(path)
    batch = collate_transitions([_row(2)])
    slot = torch.tensor([2], dtype=torch.long)
    with torch.no_grad():
        expected = model.sample_step(
            batch, None, np.random.default_rng(17),
            deterministic=True, strategy_slots=slot,
        )
    got = runtime.step(
        batch.structured_states[0],
        {"money_delta": -20, "hand_count_delta": 1},
        batch.previous_actions[0],
        None,
        np.random.default_rng(17),
        deterministic=True,
        strategy_slot=2,
    )
    assert runtime.format_version == 4
    assert runtime.strategy_count == 3
    assert runtime.default_strategy_slot == 2
    assert np.allclose(
        got.intent, expected.intent[0].numpy(),
        atol=3e-5, rtol=3e-5,
    )
    assert _strip_raw(got.canonical_action) == _strip_raw(
        _torch_canonical(expected.rows[0])
    )


def test_v32_numpy_runtime_never_generates_nop_slot(tmp_path):
    torch.manual_seed(102)
    model = TemporalIntentPolicyV32(strategy_count=2).eval()
    with torch.no_grad():
        model.market_continue_head.weight.zero_()
        model.market_continue_head.bias[:] = torch.tensor([-20.0, 20.0])
        model.market_active_op_head.weight.zero_()
        model.market_active_op_head.bias.zero_()
        model.market_active_op_head.bias[
            model.market_active_op_to_id["HIRE"]
        ] = 20.0
    path = tmp_path / "v32_no_nop.npz"
    export_v3_2_numpy(model, path, default_strategy_slot=0)
    runtime = V32NumpyPolicy.load(path)
    batch = collate_transitions([_row(0)])
    got = runtime.step(
        batch.structured_states[0], {}, batch.previous_actions[0],
        None, np.random.default_rng(19), deterministic=True,
        strategy_slot=0,
    )
    assert got.canonical_action["market"]
    assert all(
        slot.get("kind") != "NOP_SLOT"
        for slot in got.canonical_action["market"]
    )


def test_v32_numpy_runtime_import_graph_is_torch_free():
    import kaggrl.v3_2_numpy_runtime as runtime_module

    source = inspect.getsource(runtime_module)
    assert "import torch" not in source
    assert "from torch" not in source


def test_v32_export_requires_default_strategy_slot(tmp_path):
    model = TemporalIntentPolicyV32(strategy_count=2).eval()
    with pytest.raises(ValueError, match="strategy slot"):
        export_v3_2_numpy(model, tmp_path / "missing_slot.npz")


def test_v32_loader_rejects_tampered_parameter_archive(tmp_path):
    torch.manual_seed(103)
    model = TemporalIntentPolicyV32(strategy_count=2).eval()
    path = tmp_path / "v32_policy.npz"
    export_v3_2_numpy(model, path, default_strategy_slot=1)
    data = np.load(path, allow_pickle=False)
    arrays = {key: data[key].copy() for key in data.files}
    key = next(name for name in sorted(arrays) if name.startswith("p__"))
    arrays[key].reshape(-1)[0] += np.float32(0.25)
    tampered = tmp_path / "tampered.npz"
    np.savez_compressed(tampered, **arrays)
    with pytest.raises(ValueError, match="parameter hash"):
        V32NumpyPolicy.load(tampered)


def test_v32_decision_trace_matches_torch_hierarchy(tmp_path):
    torch.manual_seed(104)
    model = TemporalIntentPolicyV32(strategy_count=3).eval()
    path = tmp_path / "v32_trace.npz"
    export_v3_2_numpy(model, path, default_strategy_slot=2)
    runtime = V32NumpyPolicy.load(path)
    batch = collate_transitions([_row(1)])

    with torch.no_grad():
        torch_trace = model.trace_sample_step(
            batch,
            None,
            np.random.default_rng(23),
            deterministic=True,
            strategy_slots=torch.tensor([2]),
        )
    numpy_trace = runtime.trace_step(
        batch.structured_states[0],
        {"money_delta": -20, "hand_count_delta": 1},
        batch.previous_actions[0],
        None,
        np.random.default_rng(23),
        deterministic=True,
        strategy_slot=2,
    )

    left = torch_trace["decisions"]
    right = numpy_trace["decisions"]
    assert [d["actor"] for d in left] == [d["actor"] for d in right]
    for td, nd in zip(left, right):
        assert td["ops"] == nd["ops"]
        assert td["legal_mask"] == nd["legal_mask"]
        assert td["chosen_op"] == nd["chosen_op"]
        assert np.allclose(
            np.asarray(td["raw_logits"], dtype=np.float32),
            np.asarray(nd["raw_logits"], dtype=np.float32),
            atol=3e-5,
            rtol=3e-5,
        )


def test_v32_numpy_runtime_fresh_process_does_not_import_torch():
    import os
    import subprocess

    env = dict(os.environ)
    repo_root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = os.pathsep.join([
        str(repo_root / "src"), str(repo_root),
    ])
    code = (
        "import sys; "
        "import kaggrl.v3_2_numpy_runtime; "
        "assert 'torch' not in sys.modules, sorted("
        "k for k in sys.modules if k.startswith('torch'))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
