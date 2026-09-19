import torch
import pytest

from kaggrl.v2_device import move_step_batch, resolve_training_device


class DummyBatch:
    def __init__(self):
        self.x = torch.ones(2)
        self.mask = torch.ones(2, dtype=torch.bool)
        self.structured_states = ({"money": 3},)
        self.canonical_actions = ({"farmer": {"op": "PASS"}},)
        self.auxiliary_targets = {"effect": torch.zeros(2, 3)}
        self.sample_weight = torch.ones(2)


def test_resolve_training_device_cpu_is_explicit():
    assert resolve_training_device("cpu").type == "cpu"


def test_resolve_training_device_rejects_unavailable_xpu(monkeypatch):
    monkeypatch.setattr(torch.xpu, "is_available", lambda: False)
    assert resolve_training_device("auto").type == "cpu"
    with pytest.raises(RuntimeError, match="XPU"):
        resolve_training_device("xpu")


def test_move_step_batch_moves_tensor_fields_and_auxiliary_targets_only():
    batch = DummyBatch()
    moved = move_step_batch(batch, torch.device("cpu"))
    assert moved is batch
    assert batch.x.device.type == "cpu"
    assert batch.mask.device.type == "cpu"
    assert batch.auxiliary_targets["effect"].device.type == "cpu"
    assert batch.sample_weight.device.type == "cpu"
    assert batch.structured_states == ({"money": 3},)
    assert batch.canonical_actions[0]["farmer"]["op"] == "PASS"


def test_move_optimizer_state_keeps_optimizer_tensors_on_requested_device():
    from kaggrl.v2_device import move_optimizer_state

    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    (parameter.square().sum()).backward()
    optimizer.step()
    move_optimizer_state(optimizer, torch.device("cpu"))
    tensors = [value for state in optimizer.state.values() for value in state.values()
               if torch.is_tensor(value)]
    assert tensors
    assert all(value.device.type == "cpu" for value in tensors)


def test_resolve_training_device_cuda_is_explicit(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_training_device("cuda").type == "cuda"


def test_resolve_training_device_rejects_unavailable_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_training_device("cuda")


def test_resolve_training_device_auto_prefers_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.xpu, "is_available", lambda: True)
    assert resolve_training_device("auto").type == "cuda"
