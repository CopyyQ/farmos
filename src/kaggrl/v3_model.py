from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from .v2_model import (
    AuxiliaryOutputs,
    RecurrentIntentPolicy,
    RowPolicyOutput,
)
from .v2_tensorize import V2Batch
from .v3_temporal import (
    TemporalCore,
    TemporalDiagnostics,
    TemporalState,
)


@dataclass
class PolicyOutputV3:
    rows: tuple[RowPolicyOutput, ...]
    temporal_state: TemporalState
    fused_temporal: torch.Tensor
    intent: torch.Tensor
    aux: AuxiliaryOutputs
    temporal_diagnostics: TemporalDiagnostics

    @property
    def recurrent_state(self):
        return self.temporal_state.h, self.temporal_state.c


class TemporalIntentPolicy(RecurrentIntentPolicy):
    ARCHITECTURE_VERSION = "rl_v3_temporal_attention"

    def __init__(self, strategy_count: int = 0):
        super().__init__()
        self.strategy_count = int(strategy_count)
        if self.strategy_count < 0:
            raise ValueError("strategy_count must be non-negative")
        self.strategy_embedding = (
            torch.nn.Embedding(self.strategy_count, 128)
            if self.strategy_count else None
        )
        self.core = TemporalCore(
            hidden_dim=256,
            attention_dim=128,
            heads=4,
            window=32,
        )

    def _strategy_embedding_for(
        self, ref: torch.Tensor, strategy_slots: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if self.strategy_embedding is None:
            return None
        if strategy_slots is None:
            raise ValueError("strategy slot is required when strategy conditioning is enabled")
        slots = torch.as_tensor(
            strategy_slots, dtype=torch.long, device=ref.device,
        ).reshape(-1)
        if slots.shape[0] != ref.shape[0]:
            raise ValueError("strategy slot batch mismatch")
        if bool(((slots < 0) | (slots >= self.strategy_count)).any().item()):
            raise ValueError("strategy slot is out of range")
        return self.strategy_embedding(slots).to(ref)

    def _condition_core_input(
        self, fused: torch.Tensor, strategy_slots: torch.Tensor | None,
    ) -> torch.Tensor:
        del strategy_slots
        return fused

    def _condition_intent(
        self, intent: torch.Tensor, strategy_slots: torch.Tensor | None,
    ) -> torch.Tensor:
        embedding = self._strategy_embedding_for(intent, strategy_slots)
        return intent if embedding is None else intent + embedding

    def _run_v3(
        self,
        batch: V2Batch,
        teacher_actions,
        state: TemporalState | None,
        rng,
        deterministic: bool,
        strategy_slots: torch.Tensor | None = None,
        teacher_mix_probability: float = 1.0,
        conditioning_rng=None,
    ) -> PolicyOutputV3:
        encoded = self.encoder(batch)
        core_input = self._condition_core_input(
            encoded.fused, strategy_slots,
        )
        fused_temporal, intent, next_state, diagnostics = self.core.step(
            core_input,
            batch.previous_action_global,
            batch.previous_effect,
            batch.economy,
            state,
        )
        strategy_context = self._strategy_embedding_for(intent, strategy_slots)
        intent = (
            intent if strategy_context is None
            else intent + strategy_context
        )
        if teacher_actions is not None and len(teacher_actions) != fused_temporal.shape[0]:
            raise ValueError("teacher action batch mismatch")
        rows = []
        for row_index in range(fused_temporal.shape[0]):
            teacher = teacher_actions[row_index] if teacher_actions is not None else None
            rows.append(self._decode_row(
                encoded,
                batch,
                row_index,
                fused_temporal[row_index],
                intent[row_index],
                teacher,
                rng,
                deterministic,
                teacher_mix_probability=teacher_mix_probability,
                conditioning_rng=conditioning_rng,
                strategy_context=(
                    None if strategy_context is None
                    else strategy_context[row_index]
                ),
            ))
        return PolicyOutputV3(
            rows=tuple(rows),
            temporal_state=next_state,
            fused_temporal=fused_temporal,
            intent=intent,
            aux=self._auxiliary(encoded, fused_temporal, intent),
            temporal_diagnostics=diagnostics,
        )

    def teacher_step(
        self,
        batch: V2Batch,
        teacher_actions,
        state: TemporalState | None = None,
        strategy_slots: torch.Tensor | None = None,
        teacher_mix_probability: float = 1.0,
        conditioning_rng=None,
    ) -> PolicyOutputV3:
        return self._run_v3(
            batch, teacher_actions, state,
            rng=None, deterministic=True, strategy_slots=strategy_slots,
            teacher_mix_probability=teacher_mix_probability,
            conditioning_rng=conditioning_rng,
        )
    def sample_step(
        self,
        state_tensors: V2Batch,
        temporal_state: TemporalState | None,
        rng,
        deterministic: bool = False,
        strategy_slots: torch.Tensor | None = None,
    ) -> PolicyOutputV3:
        return self._run_v3(
            state_tensors,
            teacher_actions=None,
            state=temporal_state,
            rng=rng,
            deterministic=bool(deterministic),
            strategy_slots=strategy_slots,
        )

    def trace_sample_step(
        self,
        state_tensors: V2Batch,
        temporal_state: TemporalState | None,
        rng,
        deterministic: bool = False,
    ):
        output = self.sample_step(
            state_tensors, temporal_state, rng, deterministic=deterministic,
        )
        if len(output.rows) != 1:
            raise ValueError("trace_sample_step currently requires batch size 1")
        row = output.rows[0]
        unit_ops = [self.unit_id_to_op[i] for i in range(len(self.unit_id_to_op))]
        market_ops = [self.market_id_to_op[i] for i in range(len(self.market_id_to_op))]
        decisions = []

        def add(actor, decision, ops, domain):
            raw = decision.op_logits.detach().cpu()
            mask = decision.legal_op_mask.detach().cpu()
            masked = self._masked_logits(decision.op_logits, decision.legal_op_mask).detach().cpu()
            action = decision.chosen_action
            if domain == "unit":
                chosen = str(action.get("op", "PASS"))
            else:
                kind = str(action.get("kind", "ORDER"))
                chosen = kind if kind in {"STOP_QUEUE", "NOP_SLOT"} else str(action.get("op", "NOP_SLOT"))
            decisions.append({
                "actor": actor, "ops": list(ops),
                "raw_logits": raw.tolist(), "legal_mask": mask.tolist(),
                "masked_logits": masked.tolist(),
                "legal_action_count": int(mask.sum().item()),
                "raw_top1": str(ops[int(raw.argmax().item())]),
                "post_mask_top1": str(ops[int(masked.argmax().item())]),
                "chosen_op": chosen,
                "decision_reason": "model_argmax" if deterministic else "model_sample",
            })

        add("farmer", row.farmer, unit_ops, "unit")
        for index, decision in enumerate(row.hands):
            add(f"hand:{index}", decision, unit_ops, "unit")
        for index, decision in enumerate(row.market):
            add(f"market:{index}", decision, market_ops, "market")
        return {"output": output, "decisions": decisions}

    def forward_temporal_sequence(
        self,
        step_batches: Iterable[V2Batch],
        initial_state: TemporalState | None = None,
        teacher_actions=None,
    ):
        outputs = []
        state = initial_state
        actions = None if teacher_actions is None else list(teacher_actions)
        for time_index, batch in enumerate(step_batches):
            target = batch.canonical_actions if actions is None else actions[time_index]
            output = self.teacher_step(batch, target, state)
            outputs.append(output)
            state = output.temporal_state
        return outputs, state
