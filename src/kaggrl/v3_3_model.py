from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .v2_ledger import LAND_PRICES
from .v3_2_model import TemporalIntentPolicyV32
from .v3_3_economy import economic_market_features_from_shadow_ledger
from .v3_3_schema import (
    ACTIVE_MARKET_OPS,
    ARCHITECTURE_VERSION,
    ECONOMIC_MARKET_DIM,
    ECONOMIC_MARKET_SCALE,
    SHORT_ECONOMIC_DIM,
)
from .v3_tensor_ledger import HIRE_COST_TABLE


@dataclass
class AuxiliaryOutputsV33:
    effect: torch.Tensor
    future_resource: torch.Tensor
    unit_task: torch.Tensor
    opponent_effect: torch.Tensor
    terminal_money: torch.Tensor
    terminal_margin: torch.Tensor
    short_economic: torch.Tensor


def _signed_log1p_tensor(value: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    value = value.to(dtype)
    return torch.sign(value) * torch.log1p(value.abs())


class TemporalIntentPolicyV33(TemporalIntentPolicyV32):
    ARCHITECTURE_VERSION = ARCHITECTURE_VERSION

    def __init__(self, strategy_count: int = 0):
        super().__init__(strategy_count=strategy_count)
        self.economic_continue_head = nn.Linear(ECONOMIC_MARKET_DIM, 2)
        self.economic_active_head = nn.Linear(
            ECONOMIC_MARKET_DIM, len(ACTIVE_MARKET_OPS)
        )
        self.short_economic_head = nn.Linear(384, SHORT_ECONOMIC_DIM)
        self.register_buffer(
            "_v33_hire_cost_table",
            HIRE_COST_TABLE.clone(),
            persistent=False,
        )
        self.register_buffer(
            "_v33_land_prices",
            torch.tensor(LAND_PRICES, dtype=torch.long),
            persistent=False,
        )
        # V3.2 -> V3.3 migration must initially reproduce V3.2 exactly.
        for layer in (
            self.economic_continue_head,
            self.economic_active_head,
            self.short_economic_head,
        ):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def _tensor_economic_features(
        self, ledger, ref: torch.Tensor,
    ) -> torch.Tensor:
        dtype = ref.dtype
        device = ref.device
        cash = ledger.cash.to(device=device)
        hires_today = ledger.hires_today.to(device=device)
        land_count = ledger.land_count.to(device=device)
        step = ledger.step.to(device=device)

        hire_table = self._v33_hire_cost_table
        hire_index = hires_today.clamp(0, hire_table.numel() - 1)
        next_hire = hire_table[hire_index]

        land_prices = self._v33_land_prices
        land_index = (land_count - 1).clamp_min(0)
        valid_land = land_index < land_prices.numel()
        safe_land_index = land_index.clamp(0, land_prices.numel() - 1)
        next_land = torch.where(
            valid_land,
            land_prices[safe_land_index],
            torch.zeros_like(land_index),
        )

        unit_count = ledger.unit_mask.to(device=device).sum(dim=1)
        shed = ledger.shed.to(device=device)
        inventory = ledger.inventory.to(device=device).sum(dim=1)
        prices = ledger.market_prices.to(device=device)
        inventory_value = ((shed + inventory) * prices).sum(dim=1)
        shed_units = shed.sum(dim=1)
        slots_remaining = (
            10 - ledger.market_slots_used.to(device=device)
        ).clamp_min(0)

        cash_log = _signed_log1p_tensor(cash, dtype)
        hire_log = _signed_log1p_tensor(next_hire, dtype)
        land_log = _signed_log1p_tensor(next_land, dtype)

        return torch.stack(
            [
                cash_log,
                hire_log,
                land_log,
                cash_log - hire_log,
                torch.where(
                    next_land.gt(0),
                    cash_log - land_log,
                    torch.zeros_like(cash_log),
                ),
                _signed_log1p_tensor(hires_today, dtype),
                _signed_log1p_tensor(unit_count, dtype),
                _signed_log1p_tensor(land_count, dtype),
                _signed_log1p_tensor(inventory_value, dtype),
                _signed_log1p_tensor(shed_units, dtype),
                _signed_log1p_tensor(slots_remaining, dtype),
                _signed_log1p_tensor(step, dtype),
            ],
            dim=-1,
        )

    def _economic_market_residual(
        self, ledger, ref: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.is_tensor(getattr(ledger, "cash", None)):
            features = self._tensor_economic_features(ledger, ref)
        else:
            features = ref.new_tensor(
                economic_market_features_from_shadow_ledger(ledger)
            )
        # Economic residuals are numerically sensitive during scratch
        # training. Keep these tiny heads in FP32 even when the main decoder
        # runs under CUDA autocast.
        with torch.autocast(
            device_type=ref.device.type,
            enabled=False,
        ):
            features_fp32 = features.float()
            continue_residual = self.economic_continue_head(
                features_fp32
            )
            active_residual = self.economic_active_head(
                features_fp32
            )
        scale = float(ECONOMIC_MARKET_SCALE)
        return (
            scale * continue_residual,
            scale * active_residual,
        )

    def _auxiliary(self, encoded, h: torch.Tensor, intent: torch.Tensor):
        # Auxiliary regression is cheap compared with the recurrent decoder.
        # Run it in FP32 so large scratch-regression errors cannot overflow
        # FP16 and poison the shared loss.
        with torch.autocast(
            device_type=h.device.type,
            enabled=False,
        ):
            h_fp32 = h.float()
            intent_fp32 = intent.float()
            joint = torch.cat([h_fp32, intent_fp32], dim=-1)
            own_unit_ctx = encoded.own_unit_ctx.float()
            return AuxiliaryOutputsV33(
                effect=self.effect_head(joint),
                future_resource=self.future_resource_head(joint),
                unit_task=self.unit_task_head(own_unit_ctx),
                opponent_effect=self.opponent_effect_head(joint),
                terminal_money=self.terminal_money_head(joint).squeeze(-1),
                terminal_margin=self.terminal_margin_head(joint).squeeze(-1),
                short_economic=self.short_economic_head(joint),
            )
