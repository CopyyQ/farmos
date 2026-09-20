from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .clock import CLOCK_FEATURES


@dataclass(frozen=True)
class V4OptionLoss:
    total: torch.Tensor
    route: torch.Tensor
    route_gate: torch.Tensor
    market: torch.Tensor
    phase: torch.Tensor
    clock: torch.Tensor


class V4OptionPolicy(nn.Module):
    """Small recurrent strategic policy evaluated once per environment step."""

    ARCHITECTURE_VERSION = "farmos_v4_step_strategy_options_v1"

    def __init__(
        self,
        input_dim: int,
        route_count: int,
        market_mode_count: int,
        *,
        hidden_dim: int = 192,
        clock_dim: int = len(CLOCK_FEATURES),
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.route_count = int(route_count)
        self.market_mode_count = int(market_mode_count)
        self.hidden_dim = int(hidden_dim)
        self.clock_dim = int(clock_dim)
        if self.clock_dim != len(CLOCK_FEATURES):
            raise ValueError("V4 option clock schema width mismatch")

        self.input_proj = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.Tanh(),
            nn.Linear(256, self.hidden_dim),
            nn.Tanh(),
        )
        # Clock is a mandatory independent path.  This prevents the strategic
        # core from having to rediscover a few time values inside 1024 inputs.
        self.clock_proj = nn.Linear(self.clock_dim, self.hidden_dim)
        self.lstm = nn.LSTM(
            self.hidden_dim,
            self.hidden_dim,
            batch_first=True,
        )
        self.route_head = nn.Linear(self.hidden_dim, self.route_count)
        self.route_clock_head = nn.Linear(
            self.clock_dim, self.route_count, bias=False,
        )
        self.route_gate_head = nn.Linear(self.hidden_dim, 1)
        self.route_gate_clock_head = nn.Linear(
            self.clock_dim, 1, bias=False,
        )
        self.market_head = nn.Linear(
            self.hidden_dim, self.market_mode_count
        )
        self.market_clock_head = nn.Linear(
            self.clock_dim, self.market_mode_count, bias=False,
        )
        self.phase_head = nn.Linear(self.hidden_dim, 5)
        self.phase_clock_head = nn.Linear(
            self.clock_dim, 5, bias=False,
        )
        self.value_head = nn.Linear(self.hidden_dim, 1)
        self.value_clock_head = nn.Linear(
            self.clock_dim, 1, bias=False,
        )

        # Phase is deterministic clock metadata. Seed a strong direct mapping
        # instead of making the recurrent core rediscover calendar boundaries.
        phase_start = next(
            index for index, name in enumerate(CLOCK_FEATURES)
            if name.startswith("phase:")
        )
        with torch.no_grad():
            # These two start at zero intentionally. Promotion later requires
            # non-trivial clock-logit variation, proving training learned to
            # use absolute time for strategic decisions.
            self.route_clock_head.weight.zero_()
            self.route_gate_clock_head.weight.zero_()
            self.market_clock_head.weight.zero_()
            self.value_clock_head.weight.zero_()
            self.phase_clock_head.weight.zero_()
            for phase_index in range(5):
                self.phase_clock_head.weight[
                    phase_index, phase_start + phase_index
                ] = 4.0

    def forward_sequence(self, obs, clock_context, state=None):
        if obs.ndim != 3 or obs.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected [B,T,{self.input_dim}], got {tuple(obs.shape)}"
            )
        if (
            clock_context.ndim != 3
            or clock_context.shape[:2] != obs.shape[:2]
            or clock_context.shape[-1] != self.clock_dim
        ):
            raise ValueError(
                "clock_context must be [B,T,"
                f"{self.clock_dim}] aligned with obs"
            )
        z = self.input_proj(obs)
        clock_z = self.clock_proj(clock_context.to(z.dtype))
        z = torch.tanh(z + clock_z)
        y, state = self.lstm(z, state)
        route_clock = self.route_clock_head(clock_context)
        route_gate_logits = (
            self.route_gate_head(y).squeeze(-1)
            + self.route_gate_clock_head(clock_context).squeeze(-1)
        )
        market_clock = self.market_clock_head(clock_context)
        phase_clock = self.phase_clock_head(clock_context)
        value_clock = self.value_clock_head(clock_context).squeeze(-1)
        step_index = CLOCK_FEATURES.index("step_norm")
        remaining_index = CLOCK_FEATURES.index("remaining_steps_norm")
        exact_clock = torch.stack(
            [
                clock_context[..., step_index],
                clock_context[..., remaining_index],
            ],
            dim=-1,
        ).to(y.dtype)
        return {
            "route": self.route_head(y) + route_clock,
            "route_gate_logits": route_gate_logits,
            "route_gate": torch.sigmoid(route_gate_logits),
            "market": self.market_head(y) + market_clock,
            "phase": self.phase_head(y) + phase_clock,
            "clock": exact_clock,
            "route_clock": route_clock,
            "market_clock": market_clock,
            "value": self.value_head(y).squeeze(-1) + value_clock,
        }, state


def _weighted_route_ce(logits, targets, confidence, route_mask=None):
    if route_mask is not None:
        if route_mask.shape != logits.shape:
            raise ValueError(
                f"route mask/logit shape mismatch: "
                f"{tuple(route_mask.shape)} vs {tuple(logits.shape)}"
            )
        route_mask = route_mask.to(dtype=torch.bool, device=logits.device)
        target_allowed = route_mask.gather(
            -1, targets.unsqueeze(-1)
        ).squeeze(-1)
        if not bool(target_allowed.all()):
            raise ValueError("route target is outside compatibility mask")
        logits = logits.masked_fill(
            ~route_mask,
            torch.finfo(logits.dtype).min,
        )
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    flat_conf = confidence.reshape(-1).to(flat_logits.dtype).clamp(0.0, 1.0)
    per = F.cross_entropy(flat_logits, flat_targets, reduction="none")
    denom = flat_conf.sum()
    if float(denom.detach().cpu()) <= 0.0:
        return per.sum() * 0.0
    return (per * flat_conf).sum() / denom


def v4_option_loss(
    outputs,
    *,
    route_target,
    route_confidence,
    route_gate_target,
    route_mask=None,
    market_target,
    phase_target,
    clock_target,
    market_class_weight=None,
    route_gate_pos_weight=None,
    route_weight: float = 1.0,
    route_gate_weight: float = 0.50,
    market_weight: float = 1.00,
    phase_weight: float = 1.00,
    clock_weight: float = 2.00,
) -> V4OptionLoss:
    route = _weighted_route_ce(
        outputs["route"],
        route_target,
        route_confidence,
        route_mask=route_mask,
    )
    route_gate = F.binary_cross_entropy_with_logits(
        outputs["route_gate_logits"],
        route_gate_target.to(outputs["route_gate_logits"].dtype),
        pos_weight=route_gate_pos_weight,
    )
    market = F.cross_entropy(
        outputs["market"].reshape(-1, outputs["market"].shape[-1]),
        market_target.reshape(-1),
        weight=market_class_weight,
    )
    phase = F.cross_entropy(
        outputs["phase"].reshape(-1, outputs["phase"].shape[-1]),
        phase_target.reshape(-1),
    )
    clock = F.smooth_l1_loss(
        outputs["clock"],
        clock_target.to(outputs["clock"].dtype),
    )
    total = (
        float(route_weight) * route
        + float(route_gate_weight) * route_gate
        + float(market_weight) * market
        + float(phase_weight) * phase
        + float(clock_weight) * clock
    )
    return V4OptionLoss(
        total=total,
        route=route,
        route_gate=route_gate,
        market=market,
        phase=phase,
        clock=clock,
    )


@torch.inference_mode()
def option_metrics(
    outputs,
    *,
    route_target,
    route_confidence,
    route_gate_target,
    route_mask=None,
    market_target,
    phase_target,
    clock_target,
    last_step: int = 719,
    confident_threshold: float = 0.05,
):
    route_logits = outputs["route"]
    if route_mask is not None:
        route_mask = route_mask.to(
            dtype=torch.bool, device=route_logits.device
        )
        route_logits = route_logits.masked_fill(
            ~route_mask,
            torch.finfo(route_logits.dtype).min,
        )
    route_pred = route_logits.argmax(-1)
    market_pred = outputs["market"].argmax(-1)
    phase_pred = outputs["phase"].argmax(-1)
    confident = route_confidence >= float(confident_threshold)
    confident_n = int(confident.sum().item())
    route_correct = int(
        ((route_pred == route_target) & confident).sum().item()
    )
    clock_error = (
        outputs["clock"] - clock_target.to(outputs["clock"].dtype)
    ).abs()
    return {
        "route_confident_rows": confident_n,
        "route_acc_confident": (
            route_correct / max(1, confident_n)
        ),
        "route_gate_acc": float(
            (
                (outputs["route_gate"] >= 0.5)
                == (route_gate_target >= 0.5)
            ).float().mean().item()
        ),
        "market_acc": float(
            (market_pred == market_target).float().mean().item()
        ),
        "phase_acc": float(
            (phase_pred == phase_target).float().mean().item()
        ),
        "step_mae_turns": float(
            clock_error[..., 0].mean().item() * last_step
        ),
        "remaining_mae_turns": float(
            clock_error[..., 1].mean().item() * last_step
        ),
    }
