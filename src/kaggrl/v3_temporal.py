from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from .v2_tensorize import (
    ECONOMY_FEATURES,
    EFFECT_FEATURES,
    PREV_ACTION_GLOBAL_FEATURES,
)


@dataclass
class TemporalState:
    h: torch.Tensor
    c: torch.Tensor
    memory: torch.Tensor
    valid_length: torch.Tensor
    write_pos: torch.Tensor


@dataclass
class TemporalDiagnostics:
    attention_weights: torch.Tensor
    attention_entropy: torch.Tensor
    mean_attended_age: torch.Tensor


class TemporalCore(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        attention_dim: int = 128,
        heads: int = 4,
        window: int = 32,
    ):
        super().__init__()
        if attention_dim % heads:
            raise ValueError("attention_dim must be divisible by heads")
        self.hidden_dim = int(hidden_dim)
        self.attention_dim = int(attention_dim)
        self.heads = int(heads)
        self.window = int(window)
        self.lstm = nn.LSTMCell(256, self.hidden_dim)
        token_input = (
            self.hidden_dim + len(PREV_ACTION_GLOBAL_FEATURES)
            + len(EFFECT_FEATURES) + len(ECONOMY_FEATURES)
        )
        self.token_proj = nn.Linear(token_input, self.attention_dim)
        self.token_norm = nn.LayerNorm(self.attention_dim)
        self.q_proj = nn.Linear(self.attention_dim, self.attention_dim, bias=False)
        self.k_proj = nn.Linear(self.attention_dim, self.attention_dim, bias=False)
        self.v_proj = nn.Linear(self.attention_dim, self.attention_dim, bias=False)
        self.o_proj = nn.Linear(self.attention_dim, self.attention_dim, bias=False)
        self.attention_to_hidden = nn.Linear(self.attention_dim, self.hidden_dim)
        self.fusion_gate = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.fusion_norm = nn.LayerNorm(self.hidden_dim)
        self.intent = nn.Sequential(nn.Linear(self.hidden_dim, 128), nn.Tanh())
        self.register_buffer(
            "relative_age",
            self._sinusoidal_age(self.window, self.attention_dim),
            persistent=False,
        )

    @staticmethod
    def _sinusoidal_age(window: int, width: int) -> torch.Tensor:
        position = torch.arange(window, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, width, 2, dtype=torch.float32)
            * (-math.log(10000.0) / float(width))
        )
        values = torch.zeros(window, width, dtype=torch.float32)
        values[:, 0::2] = torch.sin(position * div)
        values[:, 1::2] = torch.cos(position * div)
        return values

    def zero_state(self, batch_size: int, device=None, dtype=None) -> TemporalState:
        param = next(self.parameters())
        device = param.device if device is None else device
        dtype = param.dtype if dtype is None else dtype
        return TemporalState(
            h=torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype),
            c=torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype),
            memory=torch.zeros(batch_size, self.window, self.attention_dim, device=device, dtype=dtype),
            valid_length=torch.zeros(batch_size, device=device, dtype=torch.long),
            write_pos=torch.zeros(batch_size, device=device, dtype=torch.long),
        )
    @staticmethod
    def detach_state(state: TemporalState | None) -> TemporalState | None:
        if state is None:
            return None
        return TemporalState(
            h=state.h.detach(), c=state.c.detach(),
            memory=state.memory.detach(),
            valid_length=state.valid_length.detach(),
            write_pos=state.write_pos.detach(),
        )

    def _ordered_with_mask(self, state: TemporalState):
        batch = state.memory.shape[0]
        positions = torch.arange(self.window, device=state.memory.device)
        start = (state.write_pos - state.valid_length) % self.window
        indices = (start.unsqueeze(1) + positions.unsqueeze(0)) % self.window
        gather_index = indices.unsqueeze(-1).expand(
            batch, self.window, self.attention_dim,
        )
        ordered = state.memory.gather(1, gather_index)
        valid = positions.unsqueeze(0) < state.valid_length.unsqueeze(1)
        ordered = ordered * valid.unsqueeze(-1).to(ordered.dtype)
        return ordered, valid

    def ordered_memory(self, state: TemporalState) -> torch.Tensor:
        return self._ordered_with_mask(state)[0]

    @staticmethod
    def _blend(old: torch.Tensor, new: torch.Tensor, active: torch.Tensor):
        mask = active
        while mask.ndim < old.ndim:
            mask = mask.unsqueeze(-1)
        return torch.where(mask, new, old)
    def _append_token(
        self,
        state: TemporalState,
        token: torch.Tensor,
        active: torch.Tensor,
    ) -> TemporalState:
        index = state.write_pos.view(-1, 1, 1).expand(
            -1, 1, self.attention_dim,
        )
        candidate_memory = state.memory.scatter(1, index, token.unsqueeze(1))
        candidate_valid = torch.clamp(state.valid_length + 1, max=self.window)
        candidate_write = (state.write_pos + 1) % self.window
        return TemporalState(
            h=state.h,
            c=state.c,
            memory=self._blend(state.memory, candidate_memory, active),
            valid_length=self._blend(state.valid_length, candidate_valid, active),
            write_pos=self._blend(state.write_pos, candidate_write, active),
        )

    def _attend(
        self,
        current_token: torch.Tensor,
        state: TemporalState,
    ) -> tuple[torch.Tensor, TemporalDiagnostics]:
        ordered, valid = self._ordered_with_mask(state)
        positions = torch.arange(self.window, device=ordered.device)
        ages = (state.valid_length.unsqueeze(1) - 1 - positions.unsqueeze(0)).clamp(
            min=0, max=self.window - 1,
        )
        age = self.relative_age.to(ordered)[ages]
        kv_input = (ordered + age) * valid.unsqueeze(-1).to(ordered.dtype)
        q = self.q_proj(current_token)
        k = self.k_proj(kv_input)
        v = self.v_proj(kv_input)
        head_dim = self.attention_dim // self.heads
        q = q.view(q.shape[0], self.heads, head_dim)
        k = k.view(k.shape[0], self.window, self.heads, head_dim).transpose(1, 2)
        v = v.view(v.shape[0], self.window, self.heads, head_dim).transpose(1, 2)
        scores = torch.einsum("bhd,bhwd->bhw", q, k) / math.sqrt(float(head_dim))
        safe_valid = valid.clone()
        empty = ~safe_valid.any(dim=1)
        if empty.any():
            safe_valid[empty, 0] = True
        # AMP on T4 produces float16 attention scores. A hard-coded -1e9
        # cannot be represented by float16 and raises before softmax. Mask with
        # the minimum finite value for the active dtype, then perform softmax
        # and diagnostics in float32 for numerical stability.
        mask_value = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~safe_valid.unsqueeze(1), mask_value)
        weights = torch.softmax(scores.float(), dim=-1)
        if empty.any():
            weights = weights.masked_fill(empty.view(-1, 1, 1), 0.0)
        context = torch.einsum("bhw,bhwd->bhd", weights.to(v.dtype), v)
        context = context.reshape(context.shape[0], self.attention_dim)
        context = self.o_proj(context)
        entropy = -(
            weights * weights.clamp_min(torch.finfo(weights.dtype).tiny).log()
        ).sum(dim=-1)
        mean_age = (
            weights * ages.unsqueeze(1).to(weights.dtype)
        ).sum(dim=-1)
        diagnostics = TemporalDiagnostics(
            attention_weights=weights,
            attention_entropy=entropy,
            mean_attended_age=mean_age,
        )
        return context, diagnostics

    def step(
        self,
        fused: torch.Tensor,
        previous_action_global: torch.Tensor,
        previous_effect: torch.Tensor,
        economy: torch.Tensor,
        state: TemporalState | None = None,
        *,
        active_mask: torch.Tensor | None = None,
    ):
        if state is None:
            state = self.zero_state(fused.shape[0], fused.device, fused.dtype)
        active = (
            torch.ones(fused.shape[0], dtype=torch.bool, device=fused.device)
            if active_mask is None else active_mask.to(device=fused.device, dtype=torch.bool)
        )
        candidate_h, candidate_c = self.lstm(fused, (state.h, state.c))
        h = self._blend(state.h, candidate_h, active)
        c = self._blend(state.c, candidate_c, active)
        token_input = torch.cat([
            h, previous_action_global, previous_effect, economy,
        ], dim=-1)
        token = self.token_norm(self.token_proj(token_input))
        working = TemporalState(
            h=h, c=c, memory=state.memory,
            valid_length=state.valid_length, write_pos=state.write_pos,
        )
        next_state = self._append_token(working, token, active)
        attention, diagnostics = self._attend(token, next_state)
        attention_hidden = self.attention_to_hidden(attention)
        gate = torch.sigmoid(self.fusion_gate(torch.cat([h, attention_hidden], dim=-1)))
        fused_temporal = self.fusion_norm(h + gate * attention_hidden)
        fused_temporal = self._blend(state.h, fused_temporal, active)
        intent = self.intent(fused_temporal)
        return fused_temporal, intent, next_state, diagnostics
