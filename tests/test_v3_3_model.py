from __future__ import annotations

import inspect
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from test_v3_2_model import _row

from kaggrl.v2_losses import action_loss
from kaggrl.v2_tensorize import collate_transitions
from kaggrl.v3_2_model import TemporalIntentPolicyV32
from kaggrl.v3_3_export import export_v3_3_numpy
from kaggrl.v3_3_model import TemporalIntentPolicyV33
from kaggrl.v3_3_numpy_runtime import V33NumpyPolicy


def _clean_action(action):
    def clean(command):
        return {
            key: value
            for key, value in command.items()
            if key not in {"raw", "_mask_fields"}
        }
    return {
        "farmer": clean(action["farmer"]),
        "hands": [clean(value) for value in action.get("hands") or []],
        "market": [clean(value) for value in action.get("market") or []],
    }


def _torch_action(row):
    return {
        "farmer": row.farmer.chosen_action,
        "hands": [item.chosen_action for item in row.hands],
        "market": [item.chosen_action for item in row.market],
    }


def _hire():
    return {
        "kind": "ORDER",
        "op": "HIRE",
        "item": None,
        "quantity": None,
        "raw": ["HIRE"],
    }


def _stop():
    return {
        "kind": "STOP_QUEUE",
        "op": None,
        "item": None,
        "quantity": None,
        "raw": [],
    }


def test_v33_zero_init_migration_preserves_v32_policy():
    torch.manual_seed(3301)
    v32 = TemporalIntentPolicyV32(strategy_count=2).eval()
    v33 = TemporalIntentPolicyV33(strategy_count=2).eval()
    incompatible = v33.load_state_dict(v32.state_dict(), strict=False)
    assert set(incompatible.missing_keys) == {
        "economic_continue_head.weight",
        "economic_continue_head.bias",
        "economic_active_head.weight",
        "economic_active_head.bias",
        "short_economic_head.weight",
        "short_economic_head.bias",
    }
    assert not incompatible.unexpected_keys

    batch = collate_transitions([_row(money=3000, hands=1)])
    slot = torch.tensor([1], dtype=torch.long)
    with torch.no_grad():
        left = v32.sample_step(
            batch,
            None,
            np.random.default_rng(77),
            deterministic=True,
            strategy_slots=slot,
        )
        right = v33.sample_step(
            batch,
            None,
            np.random.default_rng(77),
            deterministic=True,
            strategy_slots=slot,
        )
    assert torch.allclose(left.intent, right.intent, atol=0.0, rtol=0.0)
    assert _clean_action(_torch_action(left.rows[0])) == _clean_action(
        _torch_action(right.rows[0])
    )
    for ldec, rdec in zip(left.rows[0].market, right.rows[0].market):
        assert torch.allclose(
            ldec.continue_logits, rdec.continue_logits, atol=0.0, rtol=0.0
        )
        assert torch.allclose(
            ldec.op_logits, rdec.op_logits, atol=0.0, rtol=0.0
        )


def test_v33_economic_market_heads_receive_action_gradient():
    torch.manual_seed(3302)
    model = TemporalIntentPolicyV33(strategy_count=1).train()
    batch = collate_transitions([
        _row(money=3000, market=[_hire(), _stop()]),
    ])
    output = model.teacher_step(
        batch,
        batch.canonical_actions,
        None,
        strategy_slots=torch.tensor([0], dtype=torch.long),
    )
    loss = action_loss(output, batch.canonical_actions).total
    loss.backward()

    continue_grad = model.economic_continue_head.weight.grad
    active_grad = model.economic_active_head.weight.grad
    assert continue_grad is not None
    assert active_grad is not None
    assert float(continue_grad.abs().sum()) > 0.0
    assert float(active_grad.abs().sum()) > 0.0


def test_v33_short_economic_head_receives_value_gradient():
    torch.manual_seed(3303)
    model = TemporalIntentPolicyV33(strategy_count=1).train()
    batch = collate_transitions([_row(money=3000)])
    output = model.teacher_step(
        batch,
        batch.canonical_actions,
        None,
        strategy_slots=torch.tensor([0], dtype=torch.long),
    )
    target = torch.ones_like(output.aux.short_economic)
    loss = (output.aux.short_economic - target).square().mean()
    loss.backward()
    grad = model.short_economic_head.weight.grad
    assert grad is not None
    assert float(grad.abs().sum()) > 0.0


def test_v33_strategy_recovery_update_changes_economic_heads():
    from types import SimpleNamespace
    from kaggrl.v2_training_data import SequenceChunk
    from kaggrl.v3_strategy import build_strategy_manifest
    from training.train_v3_bc import _recovery_update

    torch.manual_seed(33035)
    model = TemporalIntentPolicyV33(strategy_count=1).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    row = _row(money=3000, market=[_hire(), _stop()])
    row.update({
        "episode_id": 991,
        "seat": 0,
        "step": 0,
        "teacher_id": "v45",
        "teacher_version": "unit-test",
        "supervision_kind": "accepted_policy",
        "learner_model_sha256": "a" * 64,
        "strategy_slot": 0,
    })
    chunk = SequenceChunk(
        episode_id=991,
        seat=0,
        rows=(row,),
        episode_start=True,
        episode_end=True,
    )
    before = model.economic_active_head.weight.detach().clone()
    loss, state = _recovery_update(
        model,
        optimizer,
        chunk,
        None,
        SimpleNamespace(
            gradient_clip=1.0,
            gpu_tensor_training=True,
        ),
        torch.device("cpu"),
        market_active_op_weights={"HIRE": 2.0},
        strategy_manifest=build_strategy_manifest([1]),
    )
    assert np.isfinite(loss)
    assert state is not None
    assert not torch.equal(before, model.economic_active_head.weight)


def test_v33_numpy_matches_torch(tmp_path):
    torch.manual_seed(3304)
    model = TemporalIntentPolicyV33(strategy_count=2).eval()
    with torch.no_grad():
        model.economic_continue_head.weight.normal_(std=0.01)
        model.economic_active_head.weight.normal_(std=0.01)

    path = tmp_path / "v33_policy.npz"
    export_v3_3_numpy(model, path, default_strategy_slot=1)
    runtime = V33NumpyPolicy.load(path)
    batch = collate_transitions([_row(money=3000, hands=1)])
    slot = torch.tensor([1], dtype=torch.long)

    with torch.no_grad():
        expected = model.sample_step(
            batch,
            None,
            np.random.default_rng(88),
            deterministic=True,
            strategy_slots=slot,
        )
    got = runtime.step(
        batch.structured_states[0],
        {},
        batch.previous_actions[0],
        None,
        np.random.default_rng(88),
        deterministic=True,
        strategy_slot=1,
    )
    assert runtime.format_version == 5
    assert _clean_action(got.canonical_action) == _clean_action(
        _torch_action(expected.rows[0])
    )
    assert np.allclose(
        got.intent,
        expected.intent[0].detach().cpu().numpy(),
        atol=3e-5,
        rtol=3e-5,
    )


def test_v33_numpy_runtime_import_graph_is_torch_free():
    import kaggrl.v3_3_numpy_runtime as runtime_module

    source = inspect.getsource(runtime_module)
    assert "import torch" not in source
    assert "from torch" not in source

    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([
        str(repo_root / "src"),
        str(repo_root),
    ])
    code = (
        "import sys; "
        "import kaggrl.v3_3_numpy_runtime; "
        "assert 'torch' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
