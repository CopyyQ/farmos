from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .constants import ITEM_TO_ID, UNIT_OPS
from .v2_model import (
    DecisionOutput,
    MARKET_ITEM_OPS,
    MARKET_QUANTITY_OPS,
)
from .v3_model import TemporalIntentPolicy
from .v3_2_schema import (
    ACTIVE_MARKET_OPS,
    ARCHITECTURE_VERSION,
    CONTINUE_ID,
    STOP_ID,
    STRATEGY_CORE_SCALE,
    OPENING_ACTIVE_INTENT_SCALE,
)


class TemporalIntentPolicyV32(TemporalIntentPolicy):
    ARCHITECTURE_VERSION = ARCHITECTURE_VERSION

    def __init__(self, strategy_count: int = 0):
        super().__init__(strategy_count=strategy_count)
        self.market_active_op_to_id = {
            op: i for i, op in enumerate(ACTIVE_MARKET_OPS)
        }
        self.market_active_id_to_op = {
            i: op for op, i in self.market_active_op_to_id.items()
        }
        self.market_continue_head = nn.Linear(192, 2)
        self.market_active_op_head = nn.Linear(192, len(ACTIVE_MARKET_OPS))
        self.opening_strategy_head = nn.Linear(
            128, len(ACTIVE_MARKET_OPS), bias=True,
        )
        # Backward-compatible initialization: reproduce the former direct
        # strategy_context[:width] opening residual until this head is trained.
        with torch.no_grad():
            # A fresh V3.2 model must not receive an arbitrary opening bias
            # from random strategy embeddings. Older checkpoints overwrite
            # these embeddings on load, preserving their learned behavior.
            if self.strategy_embedding is not None:
                self.strategy_embedding.weight.zero_()
            self.opening_strategy_head.weight.zero_()
            self.opening_strategy_head.bias.zero_()
            width = len(ACTIVE_MARKET_OPS)
            self.opening_strategy_head.weight[
                torch.arange(width), torch.arange(width)
            ] = 1.0

    def _condition_core_input(
        self, fused: torch.Tensor, strategy_slots: torch.Tensor | None,
    ) -> torch.Tensor:
        embedding = self._strategy_embedding_for(fused, strategy_slots)
        if embedding is None:
            return fused
        core_context = torch.cat([embedding, embedding], dim=-1)
        if core_context.shape != fused.shape:
            raise RuntimeError("strategy core context shape mismatch")
        return fused + float(STRATEGY_CORE_SCALE) * core_context

    def _sample_market_action(
        self,
        *,
        decoder_hidden: torch.Tensor,
        continue_logits: torch.Tensor,
        active_logits: torch.Tensor,
        item_logits: torch.Tensor,
        continue_mask: torch.Tensor,
        active_mask: torch.Tensor,
        legal,
        rng,
        deterministic: bool,
    ) -> dict[str, Any]:
        continue_id = self._choose(
            continue_logits, continue_mask, rng, deterministic,
        )
        if continue_id == STOP_ID:
            return {
                "kind": "STOP_QUEUE", "op": None,
                "item": None, "quantity": None, "raw": [],
            }

        active_id = self._choose(
            active_logits, active_mask, rng, deterministic,
        )
        op = self.market_active_id_to_op[active_id]
        if op == "NOP_SLOT":
            return {
                "kind": "NOP_SLOT", "op": None,
                "item": None, "quantity": None, "raw": [],
            }
        chosen = {
            "kind": "ORDER", "op": op,
            "item": None, "quantity": None, "raw": [op],
        }
        if op in MARKET_ITEM_OPS:
            item_mask = self._item_mask(
                legal.items.get(op, {}), item_logits,
            )
            item_id = self._choose(
                item_logits, item_mask, rng, deterministic,
            )
            chosen["item"] = next(
                name for name, idx in ITEM_TO_ID.items()
                if idx == item_id
            )
        if op in MARKET_QUANTITY_OPS:
            max_value = self._quantity_max_from_legal(
                legal, "market", op, chosen.get("item"),
            )
            if op == "SELL" and (max_value is None or max_value < 1):
                raise RuntimeError("SELL action has no legal inventory bound")
            _, _, quantity = self._sample_quantity(
                decoder_hidden,
                positive=True,
                rng=rng,
                deterministic=deterministic,
                max_value=max_value,
            )
            chosen["quantity"] = quantity
            chosen["raw"] = [
                op, chosen.get("item"), quantity,
            ]
        return chosen

    def _market_decision(
        self,
        *,
        slot: int,
        actor_ctx: torch.Tensor,
        decoder_hidden: torch.Tensor,
        global_h: torch.Tensor,
        intent: torch.Tensor,
        previous_emb: torch.Tensor,
        previous_label: str,
        ledger,
        teacher_action: dict[str, Any] | None,
        rng,
        deterministic: bool,
        condition_on_teacher: bool = True,
        strategy_context: torch.Tensor | None = None,
        conditioning_ledger=None,
    ):
        conditioning_ledger = ledger if conditioning_ledger is None else conditioning_ledger
        legal = ledger.legal_market_mask(slot, {})
        conditioning_legal = (
            legal if conditioning_ledger is ledger
            else conditioning_ledger.legal_market_mask(slot, {})
        )
        decoder_hidden = self._decode_input(
            actor_ctx,
            previous_emb,
            decoder_hidden,
            global_h,
            intent,
            conditioning_ledger,
            0.0,
            float(max(0, 9 - slot)) / 10.0,
        )

        continue_logits = self.market_continue_head(decoder_hidden)
        active_logits = self.market_active_op_head(decoder_hidden)
        economic_hook = getattr(self, "_economic_market_residual", None)
        if callable(economic_hook):
            economic_continue, economic_active = economic_hook(
                conditioning_ledger, decoder_hidden,
            )
            continue_logits = continue_logits + economic_continue
            active_logits = active_logits + economic_active
        if slot == 0 and int(getattr(ledger, "step", 0)) == 0:
            width = len(ACTIVE_MARKET_OPS)
            if strategy_context is None or int(strategy_context.numel()) < width:
                raise RuntimeError(
                    "strategy context is required for opening active residual"
                )
            opening_residual = self.opening_strategy_head(
                strategy_context
            )
            if int(opening_residual.numel()) != width:
                raise RuntimeError("opening strategy head width mismatch")
            active_logits = (
                active_logits
                + float(OPENING_ACTIVE_INTENT_SCALE) * opening_residual
            )
        item_logits = self.item_head(decoder_hidden)

        active_allowed = [
            bool(legal.ops.get(op, False)) for op in ACTIVE_MARKET_OPS
        ]
        active_mask = torch.tensor(
            active_allowed,
            dtype=torch.bool,
            device=active_logits.device,
        )
        continue_mask = torch.tensor(
            [True, any(active_allowed)],
            dtype=torch.bool,
            device=continue_logits.device,
        )

        if teacher_action is not None:
            chosen = dict(teacher_action)
            kind = str(chosen.get("kind", "ORDER"))
            op = (
                kind
                if kind in {"STOP_QUEUE", "NOP_SLOT"}
                else str(chosen.get("op", "NOP_SLOT"))
            )
        else:
            chosen = self._sample_market_action(
                decoder_hidden=decoder_hidden,
                continue_logits=continue_logits,
                active_logits=active_logits,
                item_logits=item_logits,
                continue_mask=continue_mask,
                active_mask=active_mask,
                legal=legal,
                rng=rng,
                deterministic=deterministic,
            )
            kind = str(chosen.get("kind", "ORDER"))
            op = (
                kind
                if kind in {"STOP_QUEUE", "NOP_SLOT"}
                else str(chosen.get("op", "NOP_SLOT"))
            )

        item_mask = None
        if op in MARKET_ITEM_OPS:
            item_mask = self._item_mask(
                legal.items.get(op, {}), item_logits,
            )
            if teacher_action is None:
                # Item was already sampled by _sample_market_action.
                pass

        quantity_logits = None
        quantity_tokens = None
        quantity_max_value = None
        if op in MARKET_QUANTITY_OPS:
            quantity_max_value = self._quantity_max_from_legal(
                legal, "market", op, chosen.get("item"),
            )
            if op == "SELL" and (quantity_max_value is None or quantity_max_value < 1):
                raise RuntimeError("SELL action has no legal inventory bound")
            if teacher_action is not None:
                quantity = chosen.get("quantity")
                teacher_bound_is_exact = (
                    condition_on_teacher
                    and op != "BUY_PRODUCT"
                    and not (
                        op in {"BUY_SEED", "BUY_ANIMAL"}
                        and bool(legal.metadata.get("cash_uncertain", False))
                    )
                )
                if (
                    teacher_bound_is_exact
                    and quantity_max_value is not None
                    and int(quantity or 0) > quantity_max_value
                ):
                    raise ValueError(
                        f"teacher {op} quantity exceeds legal quantity bound"
                    )
                quantity_logits, quantity_tokens = self._teacher_quantity(
                    decoder_hidden, quantity,
                )
            else:
                # Quantity was already sampled by _sample_market_action.
                quantity_tokens = None

        semantic = self._semantic_label(chosen, "market")
        conditioning_action = chosen
        if teacher_action is not None and not condition_on_teacher:
            conditioning_active_mask = torch.tensor(
                [bool(conditioning_legal.ops.get(candidate, False))
                 for candidate in ACTIVE_MARKET_OPS],
                dtype=torch.bool,
                device=active_logits.device,
            )
            conditioning_continue_mask = torch.tensor(
                [True, bool(conditioning_active_mask.any())],
                dtype=torch.bool,
                device=continue_logits.device,
            )
            conditioning_action = self._sample_market_action(
                decoder_hidden=decoder_hidden,
                continue_logits=continue_logits,
                active_logits=active_logits,
                item_logits=item_logits,
                continue_mask=conditioning_continue_mask,
                active_mask=conditioning_active_mask,
                legal=conditioning_legal,
                rng=rng,
                deterministic=deterministic,
            )
        conditioning_semantic = self._semantic_label(
            conditioning_action, "market",
        )

        trace = self._snapshot_trace(
            conditioning_ledger,
            "market",
            slot,
            previous_label,
            semantic,
            teacher_action is not None,
        )

        if not chosen.get("_mask_fields"):
            ledger.apply_market(chosen)
        if (
            conditioning_ledger is not ledger
            and not conditioning_action.get("_mask_fields")
        ):
            # Execution proof belongs to the expert replay state. A sampled
            # conditioning ledger may already have diverged, so carrying that
            # proof across would force impossible inventory/cash transitions.
            conditioning_ledger_action = dict(conditioning_action)
            conditioning_ledger_action.pop("_executed", None)
            conditioning_ledger.apply_market(conditioning_ledger_action)

        previous_emb = self._semantic_embedding(
            conditioning_action, "market", decoder_hidden,
        )
        decision = DecisionOutput(
            op_logits=active_logits,
            item_logits=item_logits,
            quantity_logits=quantity_logits,
            quantity_tokens=quantity_tokens,
            chosen_action=chosen,
            legal_op_mask=active_mask,
            legal_item_mask=item_mask,
            continue_logits=continue_logits,
            legal_continue_mask=continue_mask,
            quantity_max_value=quantity_max_value,
        )
        return (
            decision,
            decoder_hidden,
            previous_emb,
            conditioning_semantic,
            trace,
        )

    def trace_sample_step(
        self,
        state_tensors,
        temporal_state,
        rng,
        deterministic: bool = False,
        strategy_slots: torch.Tensor | None = None,
    ):
        output = self.sample_step(
            state_tensors,
            temporal_state,
            rng,
            deterministic=deterministic,
            strategy_slots=strategy_slots,
        )
        if len(output.rows) != 1:
            raise ValueError("trace_sample_step currently requires batch size 1")

        decisions = []

        def add(actor, decision, ops, chosen):
            raw = decision.op_logits.detach().cpu()
            mask = decision.legal_op_mask.detach().cpu()
            masked = self._masked_logits(
                decision.op_logits, decision.legal_op_mask,
            ).detach().cpu()
            decisions.append({
                "actor": actor,
                "ops": list(ops),
                "raw_logits": raw.tolist(),
                "legal_mask": mask.tolist(),
                "masked_logits": masked.tolist(),
                "legal_action_count": int(mask.sum().item()),
                "raw_top1": str(ops[int(raw.argmax().item())]),
                "post_mask_top1": str(ops[int(masked.argmax().item())]),
                "chosen_op": str(chosen),
                "decision_reason": (
                    "model_argmax" if deterministic else "model_sample"
                ),
            })

        row = output.rows[0]
        add(
            "farmer",
            row.farmer,
            UNIT_OPS,
            str(row.farmer.chosen_action.get("op", "PASS")),
        )
        for index, decision in enumerate(row.hands):
            add(
                f"hand:{index}",
                decision,
                UNIT_OPS,
                str(decision.chosen_action.get("op", "PASS")),
            )

        for index, decision in enumerate(row.market):
            if (
                decision.continue_logits is None
                or decision.legal_continue_mask is None
            ):
                raise RuntimeError(
                    "V3.2 market decision is missing continuation diagnostics"
                )
            action = decision.chosen_action
            kind = str(action.get("kind", "ORDER"))
            continue_choice = (
                "STOP" if kind == "STOP_QUEUE" else "CONTINUE"
            )
            continue_raw = decision.continue_logits.detach().cpu()
            continue_mask = decision.legal_continue_mask.detach().cpu()
            continue_masked = self._masked_logits(
                decision.continue_logits,
                decision.legal_continue_mask,
            ).detach().cpu()
            continue_ops = ("STOP", "CONTINUE")
            decisions.append({
                "actor": f"market:{index}:continue",
                "ops": list(continue_ops),
                "raw_logits": continue_raw.tolist(),
                "legal_mask": continue_mask.tolist(),
                "masked_logits": continue_masked.tolist(),
                "legal_action_count": int(continue_mask.sum().item()),
                "raw_top1": str(
                    continue_ops[int(continue_raw.argmax().item())]
                ),
                "post_mask_top1": str(
                    continue_ops[int(continue_masked.argmax().item())]
                ),
                "chosen_op": continue_choice,
                "decision_reason": (
                    "model_argmax" if deterministic else "model_sample"
                ),
            })
            if kind != "STOP_QUEUE":
                add(
                    f"market:{index}:active",
                    decision,
                    ACTIVE_MARKET_OPS,
                    str(action.get("op", "NOP_SLOT")),
                )
        return {"output": output, "decisions": decisions}
