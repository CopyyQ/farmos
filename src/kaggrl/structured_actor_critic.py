from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from torch import nn

from .actions import ActionCodec
from .constants import MARKET_OPS, UNIT_OPS
from .top_tier_expert import ITEM_CLASSES, QTY_CLASSES, TopTierExpertPolicy


class StructuredActorCritic(TopTierExpertPolicy):
    def __init__(self, input_dim=1024, hidden_dim=128, max_hands=16, max_market_orders=10):
        super().__init__(input_dim, hidden_dim, max_hands, max_market_orders)
        self.value_head = nn.Linear(self.hidden_dim, 1)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    @classmethod
    def from_bc_checkpoint(cls, path):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(
            int(saved.get("input_dim", 1024)),
            int(saved.get("hidden_dim", 128)),
        )
        missing, unexpected = model.load_state_dict(saved["model"], strict=False)
        expected = {"value_head.weight", "value_head.bias"}
        if set(missing) - expected or unexpected:
            raise RuntimeError(f"incompatible BC checkpoint missing={missing} unexpected={unexpected}")
        nn.init.zeros_(model.value_head.weight)
        nn.init.zeros_(model.value_head.bias)
        return model

    def forward_actor_critic(self, obs: torch.Tensor, teacher_actions=None, state=None):
        trunk, state = self._trunk(obs, state=state)
        batch, steps, _ = trunk.shape
        out = self._nonmarket_heads(trunk)
        self._market_heads(out, trunk, teacher_actions)
        values = self.value_head(trunk).squeeze(-1)
        return out, values, state
    def _nonmarket_heads(self, trunk):
        batch, steps, _ = trunk.shape
        return {
            "farmer_op": self.farmer_op(trunk),
            "farmer_item": self.farmer_item(trunk),
            "farmer_qty": self.farmer_qty(trunk),
            "hand_op": self.hand_op(trunk).view(batch, steps, self.max_hands, len(UNIT_OPS)),
            "hand_item": self.hand_item(trunk).view(batch, steps, self.max_hands, ITEM_CLASSES),
            "hand_qty": self.hand_qty(trunk).view(batch, steps, self.max_hands, QTY_CLASSES),
        }

    def _market_heads(self, out, trunk, teacher_actions=None):
        batch, steps, _ = trunk.shape
        flat_h = trunk.reshape(batch * steps, self.hidden_dim)
        market_teacher = None
        if teacher_actions is not None:
            start = (1 + self.max_hands) * 3
            market_teacher = teacher_actions[..., start:].long().view(
                batch * steps, self.max_market_orders, 3
            )
        prev = torch.zeros(batch * steps, 3, device=trunk.device, dtype=torch.long)
        dec_h = flat_h
        op_logits, item_logits, qty_logits = [], [], []
        for slot in range(self.max_market_orders):
            emb = torch.cat((
                self.market_op_emb(prev[:, 0]),
                self.market_item_emb(prev[:, 1]),
                self.market_qty_emb(prev[:, 2]),
            ), dim=-1)
            dec_h = self.market_cell(emb, dec_h)
            lo = self.market_op(dec_h)
            li = self.market_item(dec_h)
            lq = self.market_qty(dec_h)
            op_logits.append(lo); item_logits.append(li); qty_logits.append(lq)
            prev = market_teacher[:, slot] if market_teacher is not None else torch.stack(
                (lo.argmax(-1), li.argmax(-1), lq.argmax(-1)), dim=-1
            )
        out["market_op"] = torch.stack(op_logits, 1).view(
            batch, steps, self.max_market_orders, len(MARKET_OPS)
        )
        out["market_item"] = torch.stack(item_logits, 1).view(
            batch, steps, self.max_market_orders, ITEM_CLASSES
        )
        out["market_qty"] = torch.stack(qty_logits, 1).view(
            batch, steps, self.max_market_orders, QTY_CLASSES
        )

    def greedy_sequence(self, obs: torch.Tensor, state=None):
        out, values, state = self.forward_actor_critic(obs, state=state)
        batch, steps, _ = obs.shape
        codec = ActionCodec(self.max_hands, self.max_market_orders)
        tokens = torch.zeros(batch, steps, codec.width, device=obs.device, dtype=torch.long)
        tokens[..., 0] = out["farmer_op"].argmax(-1)
        tokens[..., 1] = out["farmer_item"].argmax(-1)
        tokens[..., 2] = out["farmer_qty"].argmax(-1)
        for i in range(self.max_hands):
            base = 3 + 3 * i
            tokens[..., base] = out["hand_op"][..., i, :].argmax(-1)
            tokens[..., base + 1] = out["hand_item"][..., i, :].argmax(-1)
            tokens[..., base + 2] = out["hand_qty"][..., i, :].argmax(-1)
        start = (1 + self.max_hands) * 3
        for i in range(self.max_market_orders):
            base = start + 3 * i
            tokens[..., base] = out["market_op"][..., i, :].argmax(-1)
            tokens[..., base + 1] = out["market_item"][..., i, :].argmax(-1)
            tokens[..., base + 2] = out["market_qty"][..., i, :].argmax(-1)
        return tokens, values, state


def export_numpy(model: StructuredActorCritic, path):
    arrays = {
        "input_dim": np.asarray(model.input_dim, dtype=np.int32),
        "hidden_dim": np.asarray(model.hidden_dim, dtype=np.int32),
        "max_hands": np.asarray(model.max_hands, dtype=np.int32),
        "max_market_orders": np.asarray(model.max_market_orders, dtype=np.int32),
    }
    for name, value in model.state_dict().items():
        arrays["p__" + name.replace(".", "__")] = value.detach().cpu().numpy().astype(np.float32)
    np.savez_compressed(Path(path), **arrays)