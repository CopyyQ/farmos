from __future__ import annotations

from typing import Any

import torch


def resolve_training_device(requested: str = "auto") -> torch.device:
    value = str(requested).strip().lower()
    if value not in {"auto", "cpu", "xpu", "cuda"}:
        raise ValueError(f"unsupported training device: {requested}")
    if value == "cpu":
        return torch.device("cpu")
    cuda_available = bool(torch.cuda.is_available())
    xpu_available = bool(hasattr(torch, "xpu") and torch.xpu.is_available())
    if value == "cuda":
        if not cuda_available:
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    if value == "xpu":
        if not xpu_available:
            raise RuntimeError("XPU was requested but is not available")
        return torch.device("xpu")
    if cuda_available:
        return torch.device("cuda")
    return torch.device("xpu" if xpu_available else "cpu")


def move_step_batch(batch: Any, device: torch.device | str):
    device = torch.device(device)
    for name, value in vars(batch).items():
        if torch.is_tensor(value):
            setattr(batch, name, value.to(device))
    auxiliary = getattr(batch, "auxiliary_targets", None)
    if isinstance(auxiliary, dict):
        batch.auxiliary_targets = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in auxiliary.items()
        }
    return batch


def _move_value(value: Any, device: torch.device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_value(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_value(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_value(item, device) for item in value)
    return value


def move_optimizer_state(optimizer, device: torch.device | str):
    device = torch.device(device)
    for parameter, state in list(optimizer.state.items()):
        optimizer.state[parameter] = _move_value(state, device)
    return optimizer
