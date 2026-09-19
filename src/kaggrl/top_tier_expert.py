from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .actions import ActionCodec
from .constants import ITEM_CLASSES, ITEM_TO_ID, MARKET_OPS, QTY_CLASSES, UNIT_OPS


class TopTierExpertPolicy(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=128, max_hands=16, max_market_orders=10):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_hands = int(max_hands)
        self.max_market_orders = int(max_market_orders)
        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.lstm = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
        self.farmer_op = nn.Linear(self.hidden_dim, len(UNIT_OPS))
        self.farmer_item = nn.Linear(self.hidden_dim, ITEM_CLASSES)
        self.farmer_qty = nn.Linear(self.hidden_dim, QTY_CLASSES)
        self.hand_op = nn.Linear(self.hidden_dim, self.max_hands * len(UNIT_OPS))
        self.hand_item = nn.Linear(self.hidden_dim, self.max_hands * ITEM_CLASSES)
        self.hand_qty = nn.Linear(self.hidden_dim, self.max_hands * QTY_CLASSES)

        self.market_op_emb = nn.Embedding(len(MARKET_OPS), 12)
        self.market_item_emb = nn.Embedding(ITEM_CLASSES, 12)
        self.market_qty_emb = nn.Embedding(QTY_CLASSES, 8)
        self.market_cell = nn.GRUCell(32, self.hidden_dim)
        self.market_op = nn.Linear(self.hidden_dim, len(MARKET_OPS))
        self.market_item = nn.Linear(self.hidden_dim, ITEM_CLASSES)
        self.market_qty = nn.Linear(self.hidden_dim, QTY_CLASSES)

    def initial_state(self, batch_size: int, device=None):
        p = next(self.parameters())
        device = device or p.device
        z = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=p.dtype)
        return z.clone(), z.clone()

    def _trunk(self, obs: torch.Tensor, state=None):
        if obs.ndim != 3 or obs.shape[-1] != self.input_dim:
            raise ValueError(f"expected [B,T,{self.input_dim}], got {tuple(obs.shape)}")
        batch, steps, _ = obs.shape
        h, c = state if state is not None else self.initial_state(batch, obs.device)
        hs = []
        for t in range(steps):
            x = torch.tanh(self.input_proj(obs[:, t]))
            h, c = self.lstm(x, (h, c))
            hs.append(h)
        return torch.stack(hs, dim=1), (h, c)

    def forward_sequence(self, obs: torch.Tensor, teacher_actions=None, state=None):
        trunk, state = self._trunk(obs, state=state)
        batch, steps, _ = trunk.shape
        out = {
            "farmer_op": self.farmer_op(trunk),
            "farmer_item": self.farmer_item(trunk),
            "farmer_qty": self.farmer_qty(trunk),
            "hand_op": self.hand_op(trunk).view(batch, steps, self.max_hands, len(UNIT_OPS)),
            "hand_item": self.hand_item(trunk).view(batch, steps, self.max_hands, ITEM_CLASSES),
            "hand_qty": self.hand_qty(trunk).view(batch, steps, self.max_hands, QTY_CLASSES),
        }

        flat_h = trunk.reshape(batch * steps, self.hidden_dim)
        market_teacher = None
        if teacher_actions is not None:
            start = (1 + self.max_hands) * 3
            market_teacher = teacher_actions[..., start:].long().view(batch * steps, self.max_market_orders, 3)
        prev = torch.zeros(batch * steps, 3, device=obs.device, dtype=torch.long)
        dec_h = flat_h
        op_logits, item_logits, qty_logits = [], [], []
        for slot in range(self.max_market_orders):
            emb = torch.cat((
                self.market_op_emb(prev[:, 0]),
                self.market_item_emb(prev[:, 1]),
                self.market_qty_emb(prev[:, 2]),
            ), dim=-1)
            dec_h = self.market_cell(emb, dec_h)
            lo, li, lq = self.market_op(dec_h), self.market_item(dec_h), self.market_qty(dec_h)
            op_logits.append(lo); item_logits.append(li); qty_logits.append(lq)
            prev = market_teacher[:, slot] if market_teacher is not None else torch.stack((lo.argmax(-1), li.argmax(-1), lq.argmax(-1)), dim=-1)
        out["market_op"] = torch.stack(op_logits, 1).view(batch, steps, self.max_market_orders, len(MARKET_OPS))
        out["market_item"] = torch.stack(item_logits, 1).view(batch, steps, self.max_market_orders, ITEM_CLASSES)
        out["market_qty"] = torch.stack(qty_logits, 1).view(batch, steps, self.max_market_orders, QTY_CLASSES)
        return out, state


def _weighted_ce(logits, target, mask, sample_weights):
    losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="none").view_as(target)
    weights = sample_weights
    while weights.ndim < target.ndim:
        weights = weights.unsqueeze(-1)
    active = mask.to(losses.dtype) * weights.to(losses.dtype)
    denom = active.sum().clamp_min(1.0)
    return (losses * active).sum() / denom


def expert_bc_loss(outputs, actions, masks, sample_weights):
    actions = actions.long()
    masks = masks.to(dtype=torch.float32)
    batch, steps, _ = actions.shape
    unit_slots = 17
    farmer = actions[:, :, :3]
    farmer_mask = masks[:, :, :3]
    hands = actions[:, :, 3:unit_slots * 3].view(batch, steps, 16, 3)
    hand_mask = masks[:, :, 3:unit_slots * 3].view(batch, steps, 16, 3)
    market = actions[:, :, unit_slots * 3:].view(batch, steps, 10, 3)
    market_mask = masks[:, :, unit_slots * 3:].view(batch, steps, 10, 3)
    farmer_loss = (
        _weighted_ce(outputs["farmer_op"], farmer[..., 0], farmer_mask[..., 0], sample_weights)
        + _weighted_ce(outputs["farmer_item"], farmer[..., 1], farmer_mask[..., 1], sample_weights)
        + _weighted_ce(outputs["farmer_qty"], farmer[..., 2], farmer_mask[..., 2], sample_weights)
    )
    hands_loss = (
        _weighted_ce(outputs["hand_op"], hands[..., 0], hand_mask[..., 0], sample_weights)
        + _weighted_ce(outputs["hand_item"], hands[..., 1], hand_mask[..., 1], sample_weights)
        + _weighted_ce(outputs["hand_qty"], hands[..., 2], hand_mask[..., 2], sample_weights)
    )
    market_loss = (
        _weighted_ce(outputs["market_op"], market[..., 0], market_mask[..., 0], sample_weights)
        + _weighted_ce(outputs["market_item"], market[..., 1], market_mask[..., 1], sample_weights)
        + _weighted_ce(outputs["market_qty"], market[..., 2], market_mask[..., 2], sample_weights)
    )
    total = 4.0 * farmer_loss + hands_loss + 2.0 * market_loss
    return total, {"farmer": farmer_loss, "hands": hands_loss, "market": market_loss}


def decode_action(outputs: dict[str, torch.Tensor], observation: dict) -> dict:
    codec = ActionCodec()
    tokens = np.zeros(codec.width, dtype=np.int16)
    tokens[0] = int(outputs["farmer_op"][0, -1].argmax(-1))
    tokens[1] = int(outputs["farmer_item"][0, -1].argmax(-1))
    tokens[2] = int(outputs["farmer_qty"][0, -1].argmax(-1))
    for i in range(16):
        base = 3 + i * 3
        tokens[base] = int(outputs["hand_op"][0, -1, i].argmax(-1))
        tokens[base + 1] = int(outputs["hand_item"][0, -1, i].argmax(-1))
        tokens[base + 2] = int(outputs["hand_qty"][0, -1, i].argmax(-1))
    start = 17 * 3
    for i in range(10):
        base = start + i * 3
        tokens[base] = int(outputs["market_op"][0, -1, i].argmax(-1))
        tokens[base + 1] = int(outputs["market_item"][0, -1, i].argmax(-1))
        tokens[base + 2] = int(outputs["market_qty"][0, -1, i].argmax(-1))
    return codec.decode(tokens, observation)
