from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualPolicy(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, order_vocab_size: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.order_vocab_size = order_vocab_size
        self.input_proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.edit0 = nn.Linear(hidden_dim, 3)
        self.edit1 = nn.Linear(hidden_dim, 3)
        self.order0 = nn.Linear(hidden_dim, order_vocab_size)
        self.order1 = nn.Linear(hidden_dim, order_vocab_size)
        self.value = nn.Linear(hidden_dim, 1)

    def forward_sequence(self, obs, state=None):
        z = self.input_proj(obs)
        y, state = self.lstm(z, state)
        out = {
            "edit0": self.edit0(y),
            "edit1": self.edit1(y),
            "order0": self.order0(y),
            "order1": self.order1(y),
            "value": self.value(y).squeeze(-1),
        }
        return out, state


def _flat_ce(logits, target):
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))


def residual_bc_loss(logits, labels):
    edit0 = labels["edit0"]
    edit1 = labels["edit1"]
    loss = _flat_ce(logits["edit0"], edit0) + _flat_ce(logits["edit1"], edit1)
    replace_slots = 0
    order_losses = []
    for slot in (0, 1):
        edit = labels[f"edit{slot}"].reshape(-1)
        order = labels[f"order{slot}"].reshape(-1)
        order_logits = logits[f"order{slot}"].reshape(-1, logits[f"order{slot}"].shape[-1])
        mask = edit.eq(2)
        replace_slots += int(mask.sum().item())
        if mask.any():
            order_losses.append(F.cross_entropy(order_logits[mask], order[mask]))
    if order_losses:
        loss = loss + torch.stack(order_losses).mean()
    return loss, {"replace_slots": replace_slots}
