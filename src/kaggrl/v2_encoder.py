from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .v2_tensorize import (
    COMMODITY_FEATURES,
    ECONOMY_FEATURES,
    EFFECT_FEATURES,
    PREV_ACTION_GLOBAL_FEATURES,
    PREV_UNIT_ACTION_FEATURES,
    PREV_UNIT_EFFECT_FEATURES,
    TILE_FEATURES,
    UNIT_FEATURES,
    UNIT_FEATURE_INDEX,
    V2Batch,
)


@dataclass
class EncodedState:
    fused: torch.Tensor
    own_unit_ctx: torch.Tensor
    rival_unit_ctx: torch.Tensor
    tile_ctx: torch.Tensor


class TileEncoder(nn.Module):
    def __init__(self, d_model: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(len(TILE_FEATURES), d_model), nn.SiLU(),
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model),
        )

    def forward(self, tiles: torch.Tensor) -> torch.Tensor:
        return self.net(tiles)


class _GridRound(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.self_linear = nn.Linear(d_model, d_model)
        self.neighbor_linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        total = torch.zeros_like(x)
        count = torch.zeros_like(x[..., :1])
        total[:, 1:] = total[:, 1:] + x[:, :-1]
        count[:, 1:] = count[:, 1:] + 1
        total[:, :-1] = total[:, :-1] + x[:, 1:]
        count[:, :-1] = count[:, :-1] + 1
        total[:, :, 1:] = total[:, :, 1:] + x[:, :, :-1]
        count[:, :, 1:] = count[:, :, 1:] + 1
        total[:, :, :-1] = total[:, :, :-1] + x[:, :, 1:]
        count[:, :, :-1] = count[:, :, :-1] + 1
        neighbor = total / count.clamp_min(1.0)
        delta = F.silu(self.self_linear(x) + self.neighbor_linear(neighbor))
        return self.norm(x + delta)


class GridMessagePass(nn.Module):
    def __init__(self, d_model: int = 64, rounds: int = 2):
        super().__init__()
        self.rounds = nn.ModuleList([_GridRound(d_model) for _ in range(rounds)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.rounds:
            x = block(x)
        return x


class UnitInteractionEncoder(nn.Module):
    def __init__(self, d_model: int = 128, layers: int = 2, heads: int = 4, tile_dim: int = 64):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(
                len(UNIT_FEATURES) + len(PREV_UNIT_ACTION_FEATURES)
                + len(PREV_UNIT_EFFECT_FEATURES) + tile_dim + 2, d_model
            ),
            nn.LayerNorm(d_model), nn.SiLU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=2 * d_model,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)

    @staticmethod
    def _local_tiles(units: torch.Tensor, tiles: torch.Tensor) -> torch.Tensor:
        batch, count = units.shape[:2]
        if count == 0:
            return tiles.new_zeros((batch, 0, tiles.shape[-1]))
        height, width = tiles.shape[1:3]
        x = torch.round(units[..., UNIT_FEATURE_INDEX["x"]] * max(1, width - 1)).long().clamp(0, width - 1)
        y = torch.round(units[..., UNIT_FEATURE_INDEX["y"]] * max(1, height - 1)).long().clamp(0, height - 1)
        b = torch.arange(batch, device=units.device).unsqueeze(1).expand(batch, count)
        return tiles[b, y, x]

    def _tokens(self, units: torch.Tensor, mask: torch.Tensor, tiles: torch.Tensor,
                previous_actions: torch.Tensor, previous_effects: torch.Tensor,
                side: int) -> torch.Tensor:
        local = self._local_tiles(units, tiles)
        side_bits = units.new_zeros((*units.shape[:2], 2))
        side_bits[..., side] = 1.0
        tokens = self.input_proj(torch.cat([
            units, previous_actions, previous_effects, local, side_bits,
        ], dim=-1))
        return tokens * mask.unsqueeze(-1).to(tokens.dtype)

    def forward(
        self,
        own_units: torch.Tensor,
        own_mask: torch.Tensor,
        rival_units: torch.Tensor,
        rival_mask: torch.Tensor,
        own_tiles: torch.Tensor,
        rival_tiles: torch.Tensor,
        previous_unit_actions: torch.Tensor,
        previous_unit_effects: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        own_count = own_units.shape[1]
        own = self._tokens(
            own_units, own_mask, own_tiles, previous_unit_actions,
            previous_unit_effects, side=0
        )
        rival_previous = rival_units.new_zeros(
            (*rival_units.shape[:2], len(PREV_UNIT_ACTION_FEATURES))
        )
        rival_effects = rival_units.new_zeros(
            (*rival_units.shape[:2], len(PREV_UNIT_EFFECT_FEATURES))
        )
        rival = self._tokens(
            rival_units, rival_mask, rival_tiles, rival_previous,
            rival_effects, side=1
        )
        tokens = torch.cat([own, rival], dim=1)
        mask = torch.cat([own_mask, rival_mask], dim=1)
        context = self.transformer(tokens, src_key_padding_mask=~mask)
        context = context * mask.unsqueeze(-1).to(context.dtype)
        return context[:, :own_count], context[:, own_count:]


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(values.dtype)
    total = (values * weights).sum(dim=1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return total / denom


class StateEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tile_encoder = TileEncoder(64)
        self.grid_pass = GridMessagePass(64, rounds=2)
        self.unit_encoder = UnitInteractionEncoder(128, layers=2, heads=4, tile_dim=64)
        self.commodity_encoder = nn.Sequential(
            nn.Linear(len(COMMODITY_FEATURES), 64), nn.SiLU(),
            nn.Linear(64, 64), nn.LayerNorm(64),
        )
        self.economy_encoder = nn.Sequential(
            nn.Linear(len(ECONOMY_FEATURES), 128), nn.SiLU(),
            nn.Linear(128, 128), nn.LayerNorm(128),
        )
        self.previous_action_encoder = nn.Sequential(
            nn.Linear(len(PREV_ACTION_GLOBAL_FEATURES), 64), nn.SiLU(),
            nn.Linear(64, 64), nn.LayerNorm(64),
        )
        self.effect_encoder = nn.Sequential(
            nn.Linear(len(EFFECT_FEATURES), 64), nn.SiLU(),
            nn.Linear(64, 64), nn.LayerNorm(64),
        )
        self.fusion = nn.Sequential(
            nn.Linear(64 + 64 + 128 + 128 + 64 + 128 + 64 + 64, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.LayerNorm(256),
        )

    def forward(self, batch: V2Batch) -> EncodedState:
        own_tiles = self.grid_pass(self.tile_encoder(batch.own_grid))
        rival_tiles = self.grid_pass(self.tile_encoder(batch.rival_grid))
        own_ctx, rival_ctx = self.unit_encoder(
            batch.own_units, batch.own_unit_mask,
            batch.rival_units, batch.rival_unit_mask,
            own_tiles, rival_tiles, batch.previous_unit_actions,
            batch.previous_unit_effects,
        )
        own_pool = _masked_mean(own_ctx, batch.own_unit_mask)
        rival_pool = _masked_mean(rival_ctx, batch.rival_unit_mask)
        own_grid_pool = own_tiles.mean(dim=(1, 2))
        rival_grid_pool = rival_tiles.mean(dim=(1, 2))
        commodity = self.commodity_encoder(batch.commodities).mean(dim=1)
        economy = self.economy_encoder(batch.economy)
        previous_action = self.previous_action_encoder(batch.previous_action_global)
        effect = self.effect_encoder(batch.previous_effect)
        fused = self.fusion(torch.cat([
            own_grid_pool, rival_grid_pool, own_pool, rival_pool,
            commodity, economy, previous_action, effect,
        ], dim=-1))
        return EncodedState(
            fused=fused, own_unit_ctx=own_ctx, rival_unit_ctx=rival_ctx,
            tile_ctx=torch.stack([own_tiles, rival_tiles], dim=1),
        )


class RecurrentCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTMCell(256, 256)
        self.intent = nn.Sequential(nn.Linear(256, 128), nn.Tanh())

    def zero_state(self, batch_size: int, device=None, dtype=None) -> tuple[torch.Tensor, torch.Tensor]:
        param = next(self.parameters())
        device = param.device if device is None else device
        dtype = param.dtype if dtype is None else dtype
        shape = (int(batch_size), 256)
        return (
            torch.zeros(shape, device=device, dtype=dtype),
            torch.zeros(shape, device=device, dtype=dtype),
        )

    def step(
        self,
        fused: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if state is None:
            state = self.zero_state(fused.shape[0], fused.device, fused.dtype)
        h, c = self.lstm(fused, state)
        return h, c, self.intent(h)
