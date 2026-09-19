from __future__ import annotations

from dataclasses import dataclass

import torch

from .constants import UNIT_OPS
from .v2_model import AuxiliaryOutputs
from .v3_2_schema import ACTIVE_MARKET_OPS, OPENING_ACTIVE_INTENT_SCALE
from .v3_temporal import TemporalDiagnostics, TemporalState
from .v3_tensor_ledger import TensorLedger
from .v3_tensor_targets import (
    MARKET_OP_TO_ID,
    UNIT_OP_TO_ID,
    TensorActionTargets,
)


@dataclass
class TensorTeacherPolicyOutput:
    unit_op_logits: torch.Tensor
    unit_item_logits: torch.Tensor
    unit_quantity_logits: torch.Tensor
    unit_quantity_token_mask: torch.Tensor
    unit_legal_op_mask: torch.Tensor
    market_continue_logits: torch.Tensor
    market_active_logits: torch.Tensor
    market_item_logits: torch.Tensor
    market_quantity_logits: torch.Tensor
    market_quantity_token_mask: torch.Tensor
    market_continue_legal_mask: torch.Tensor
    market_active_legal_mask: torch.Tensor
    temporal_state: TemporalState
    fused_temporal: torch.Tensor
    intent: torch.Tensor
    aux: AuxiliaryOutputs
    temporal_diagnostics: TemporalDiagnostics


def _signed_log1p_tensor(value: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    value = value.to(dtype)
    return torch.sign(value) * torch.log1p(value.abs())


def semantic_embedding_tensor(
    model,
    op: torch.Tensor,
    item: torch.Tensor,
    quantity: torch.Tensor,
    *,
    domain: str,
    ref: torch.Tensor,
) -> torch.Tensor:
    op = op.to(device=ref.device, dtype=torch.long)
    item = item.to(device=ref.device, dtype=torch.long)
    quantity = quantity.to(device=ref.device, dtype=torch.long)
    if domain == "unit":
        op_index = 1 + op
    elif domain == "market":
        op_index = 1 + len(UNIT_OPS) + op
    else:
        raise ValueError(f"unknown action domain: {domain}")
    omitted = quantity.lt(0)
    quantity_features = torch.stack([
        omitted.to(ref.dtype),
        _signed_log1p_tensor(
            torch.where(omitted, torch.zeros_like(quantity), quantity),
            ref.dtype,
        ),
    ], dim=-1)
    op_emb = model.op_embedding(op_index)
    item_emb = model.item_embedding(item)
    return model.action_proj(torch.cat([
        op_emb.to(ref),
        item_emb.to(ref),
        quantity_features,
    ], dim=-1))


def _decode_input_tensor(
    model,
    *,
    actor_ctx: torch.Tensor,
    previous: torch.Tensor,
    hidden: torch.Tensor,
    global_h: torch.Tensor,
    intent: torch.Tensor,
    ledger: TensorLedger,
    remaining_units: torch.Tensor,
    remaining_market: torch.Tensor,
) -> torch.Tensor:
    ledger_ctx = model.ledger_proj(ledger.ledger_vector(actor_ctx))
    tail = torch.stack([
        remaining_units.to(actor_ctx),
        remaining_market.to(actor_ctx),
    ], dim=-1)
    decoder_input = torch.cat([
        actor_ctx,
        previous,
        ledger_ctx,
        global_h,
        intent,
        tail,
    ], dim=-1)
    return model.decoder_cell(decoder_input, hidden)


def _teacher_quantity_logits_tensor(
    model,
    hidden: torch.Tensor,
    tokens: torch.Tensor,
    token_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    context = model.quantity_context(hidden)
    return model.quantity_decoder.teacher_logits_tensor(
        context,
        tokens,
        token_mask,
    )


def _masked_argmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    floor = torch.finfo(logits.dtype).min
    return logits.masked_fill(~mask, floor).argmax(dim=-1)


def _sample_quantity_argmax_tensor(
    model,
    hidden: torch.Tensor,
    *,
    positive: torch.Tensor,
    max_value: torch.Tensor,
    active: torch.Tensor,
    max_steps: int | None = None,
) -> torch.Tensor:
    context = model.quantity_context(hidden)
    qhidden = model.quantity_decoder.initial_state(context)
    batch = hidden.shape[0]
    device = hidden.device
    previous = torch.full(
        (batch,), 12, dtype=torch.long, device=device,
    )
    value = torch.zeros(batch, dtype=torch.long, device=device)
    digits = torch.zeros(batch, dtype=torch.long, device=device)
    omitted = torch.zeros(batch, dtype=torch.bool, device=device)
    finished = ~active.to(device=device, dtype=torch.bool)
    positive = positive.to(device=device, dtype=torch.bool)
    max_value = max_value.to(device=device, dtype=torch.long)
    bounded = max_value.ge(0)
    vocab = model.quantity_decoder.vocab_size
    token_ids = torch.arange(vocab, device=device)

    decode_steps = (
        model.quantity_decoder.max_digits + 1
        if max_steps is None
        else max(1, min(int(max_steps), model.quantity_decoder.max_digits + 1))
    )
    for step_index in range(decode_steps):
        logits, next_hidden = model.quantity_decoder.step(qhidden, previous)
        running = ~finished
        allowed = torch.zeros(
            (batch, vocab), dtype=torch.bool, device=device,
        )
        if step_index == 0:
            allowed[:, 0] = ~positive
            allowed[:, 1:11] = True
            allowed[positive, 1] = False
        else:
            force_end = digits.ge(model.quantity_decoder.max_digits) | (
                digits.eq(1) & value.eq(0)
            )
            allowed[:, 11] = True
            allowed[:, 1:11] = ~force_end.unsqueeze(1)
            allowed[:, 11] |= force_end

        digit_mask = token_ids.ge(1) & token_ids.le(10)
        digit_values = (token_ids - 1).clamp(0, 9)
        candidate = value.unsqueeze(1) * 10 + digit_values.unsqueeze(0)
        allowed &= ~(
            bounded.unsqueeze(1)
            & digit_mask.unsqueeze(0)
            & candidate.gt(max_value.unsqueeze(1))
        )
        end_invalid = (
            ~digits.gt(0)
            | (bounded & value.gt(max_value))
        )
        allowed[:, 11] &= ~end_invalid
        allowed[:, 0] &= ~positive
        allowed[finished, 0] = True

        token = _masked_argmax(logits, allowed)
        qhidden = torch.where(
            running.unsqueeze(-1), next_hidden, qhidden,
        )
        is_omit = running & token.eq(0)
        is_end = running & token.eq(11)
        is_digit = running & token.ge(1) & token.le(10)
        digit = (token - 1).clamp(0, 9)
        value = torch.where(
            is_digit, value * 10 + digit, value,
        )
        digits = digits + is_digit.to(torch.long)
        omitted |= is_omit
        finished |= is_omit | is_end
        previous = torch.where(running, token, previous)
    return torch.where(
        omitted | digits.eq(0),
        torch.full_like(value, -1),
        value,
    )


def _teacher_choice_mask(
    active: torch.Tensor,
    probability: float,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    p = float(probability)
    if not 0.0 <= p <= 1.0:
        raise ValueError("teacher_mix_probability must be in [0, 1]")
    if p >= 1.0:
        return active.clone()
    if p <= 0.0:
        return torch.zeros_like(active)
    draws = torch.rand(
        active.shape,
        device=active.device,
        generator=generator,
    )
    return active & draws.lt(p)


def teacher_step_tensor(
    model,
    batch,
    targets: TensorActionTargets,
    initial_ledger: TensorLedger,
    state: TemporalState | None = None,
    *,
    strategy_slots: torch.Tensor | None = None,
) -> TensorTeacherPolicyOutput:
    if targets.batch_size != int(batch.own_grid.shape[0]):
        raise ValueError("tensor target batch mismatch")
    if initial_ledger.batch_size != targets.batch_size:
        raise ValueError("tensor ledger batch mismatch")
    if targets.max_units != int(batch.own_unit_mask.shape[1]):
        raise ValueError("tensor target unit width mismatch")

    encoded = model.encoder(batch)
    core_input = model._condition_core_input(
        encoded.fused, strategy_slots,
    )
    fused_temporal, intent, next_state, diagnostics = model.core.step(
        core_input,
        batch.previous_action_global,
        batch.previous_effect,
        batch.economy,
        state,
    )
    strategy_context = model._strategy_embedding_for(
        intent, strategy_slots,
    )
    if strategy_context is not None:
        intent = intent + strategy_context

    ledger = initial_ledger.clone()
    ledger.set_atomic_plant_blocked(
        targets.atomic_plant_blocked(ledger.seeds)
    )
    batch_size = targets.batch_size
    device = fused_temporal.device
    target_device = targets.to(device)
    if ledger.device != device:
        ledger = ledger.to(device)

    decoder_hidden = model.decoder_init(torch.cat([
        fused_temporal, intent,
    ], dim=-1))
    previous_emb = model.start_action.to(fused_temporal).unsqueeze(0).expand(
        batch_size, -1
    )
    unit_counts = target_device.unit_mask.sum(dim=1).clamp_min(1)

    unit_op_logits = []
    unit_item_logits = []
    unit_quantity_logits = []
    unit_legal_op_mask = []

    for actor_index in range(target_device.max_units):
        active = target_device.unit_mask[:, actor_index]
        remaining_units = (
            unit_counts - int(actor_index) - 1
        ).clamp_min(0).to(fused_temporal.dtype) / unit_counts.to(
            fused_temporal.dtype
        )
        remaining_market = torch.ones(
            batch_size,
            device=device,
            dtype=fused_temporal.dtype,
        )
        legal = ledger.legal_unit(actor_index)
        candidate_hidden = _decode_input_tensor(
            model,
            actor_ctx=encoded.own_unit_ctx[:, actor_index],
            previous=previous_emb,
            hidden=decoder_hidden,
            global_h=fused_temporal,
            intent=intent,
            ledger=ledger,
            remaining_units=remaining_units,
            remaining_market=remaining_market,
        )
        op_logits = model.unit_op_head(candidate_hidden)
        item_logits = model.item_head(candidate_hidden)
        quantity_logits, _ = _teacher_quantity_logits_tensor(
            model,
            candidate_hidden,
            target_device.unit_quantity_tokens[:, actor_index],
            target_device.unit_quantity_token_mask[:, actor_index],
        )
        unit_op_logits.append(op_logits)
        unit_item_logits.append(item_logits)
        unit_quantity_logits.append(quantity_logits)
        unit_legal_op_mask.append(legal.op_mask)

        decoder_hidden = torch.where(
            active.unsqueeze(-1),
            candidate_hidden,
            decoder_hidden,
        )
        next_previous = semantic_embedding_tensor(
            model,
            target_device.unit_op[:, actor_index],
            target_device.unit_item[:, actor_index],
            target_device.unit_quantity[:, actor_index],
            domain="unit",
            ref=candidate_hidden,
        )
        previous_emb = torch.where(
            active.unsqueeze(-1),
            next_previous,
            previous_emb,
        )
        ledger.apply_unit(
            actor_index,
            target_device.unit_op[:, actor_index],
            target_device.unit_item[:, actor_index],
            target_device.unit_quantity[:, actor_index],
            active=active,
        )

    market_continue_logits = []
    market_active_logits = []
    market_item_logits = []
    market_quantity_logits = []
    market_continue_legal_mask = []
    market_active_legal_mask = []

    for slot in range(target_device.market_op.shape[1]):
        active = target_device.market_mask[:, slot]
        legal = ledger.legal_market(slot)
        slot_ids = torch.full(
            (batch_size,), slot, dtype=torch.long, device=device
        )
        actor_ctx = model.market_slot_embedding(slot_ids)
        remaining_units = torch.zeros(
            batch_size,
            device=device,
            dtype=fused_temporal.dtype,
        )
        remaining_market = torch.full(
            (batch_size,),
            float(max(0, 9 - slot)) / 10.0,
            device=device,
            dtype=fused_temporal.dtype,
        )
        candidate_hidden = _decode_input_tensor(
            model,
            actor_ctx=actor_ctx,
            previous=previous_emb,
            hidden=decoder_hidden,
            global_h=fused_temporal,
            intent=intent,
            ledger=ledger,
            remaining_units=remaining_units,
            remaining_market=remaining_market,
        )
        continue_logits = model.market_continue_head(candidate_hidden)
        active_logits = model.market_active_op_head(candidate_hidden)
        economic_hook = getattr(model, "_economic_market_residual", None)
        if callable(economic_hook):
            economic_continue, economic_active = economic_hook(
                ledger, candidate_hidden,
            )
            continue_logits = continue_logits + economic_continue
            active_logits = active_logits + economic_active
        if slot == 0 and strategy_context is not None:
            opening_residual = model.opening_strategy_head(
                strategy_context
            )
            opening_mask = ledger.step.eq(0).to(
                active_logits.dtype
            ).unsqueeze(-1)
            active_logits = (
                active_logits
                + opening_mask
                * float(OPENING_ACTIVE_INTENT_SCALE)
                * opening_residual
            )
        item_logits = model.item_head(candidate_hidden)
        quantity_logits, _ = _teacher_quantity_logits_tensor(
            model,
            candidate_hidden,
            target_device.market_quantity_tokens[:, slot],
            target_device.market_quantity_token_mask[:, slot],
        )

        market_continue_logits.append(continue_logits)
        market_active_logits.append(active_logits)
        market_item_logits.append(item_logits)
        market_quantity_logits.append(quantity_logits)
        market_continue_legal_mask.append(
            ledger.continue_market_mask(legal)
        )
        market_active_legal_mask.append(
            ledger.active_market_mask(legal)
        )

        decoder_hidden = torch.where(
            active.unsqueeze(-1),
            candidate_hidden,
            decoder_hidden,
        )
        next_previous = semantic_embedding_tensor(
            model,
            target_device.market_op[:, slot],
            target_device.market_item[:, slot],
            target_device.market_quantity[:, slot],
            domain="market",
            ref=candidate_hidden,
        )
        previous_emb = torch.where(
            active.unsqueeze(-1),
            next_previous,
            previous_emb,
        )
        ledger.apply_market(
            target_device.market_op[:, slot],
            target_device.market_item[:, slot],
            target_device.market_quantity[:, slot],
            known_executed=target_device.market_executed[:, slot],
            active=active,
        )

    return TensorTeacherPolicyOutput(
        unit_op_logits=torch.stack(unit_op_logits, dim=1),
        unit_item_logits=torch.stack(unit_item_logits, dim=1),
        unit_quantity_logits=torch.stack(unit_quantity_logits, dim=1),
        unit_quantity_token_mask=target_device.unit_quantity_token_mask,
        unit_legal_op_mask=torch.stack(unit_legal_op_mask, dim=1),
        market_continue_logits=torch.stack(market_continue_logits, dim=1),
        market_active_logits=torch.stack(market_active_logits, dim=1),
        market_item_logits=torch.stack(market_item_logits, dim=1),
        market_quantity_logits=torch.stack(market_quantity_logits, dim=1),
        market_quantity_token_mask=target_device.market_quantity_token_mask,
        market_continue_legal_mask=torch.stack(
            market_continue_legal_mask, dim=1
        ),
        market_active_legal_mask=torch.stack(
            market_active_legal_mask, dim=1
        ),
        temporal_state=next_state,
        fused_temporal=fused_temporal,
        intent=intent,
        aux=model._auxiliary(
            encoded, fused_temporal, intent,
        ),
        temporal_diagnostics=diagnostics,
    )


def teacher_step_tensor_mixed(
    model,
    batch,
    targets: TensorActionTargets,
    initial_ledger: TensorLedger,
    state: TemporalState | None = None,
    *,
    strategy_slots: torch.Tensor | None = None,
    teacher_mix_probability: float = 1.0,
    generator: torch.Generator | None = None,
    precomputed: dict | None = None,
) -> TensorTeacherPolicyOutput:
    if float(teacher_mix_probability) >= 1.0 and precomputed is None:
        return teacher_step_tensor(
            model,
            batch,
            targets,
            initial_ledger,
            state,
            strategy_slots=strategy_slots,
        )
    if targets.batch_size != int(batch.own_grid.shape[0]):
        raise ValueError("tensor target batch mismatch")
    if initial_ledger.batch_size != targets.batch_size:
        raise ValueError("tensor ledger batch mismatch")
    if targets.max_units != int(batch.own_unit_mask.shape[1]):
        raise ValueError("tensor target unit width mismatch")

    if precomputed is None:
        encoded = model.encoder(batch)
        core_input = model._condition_core_input(
            encoded.fused, strategy_slots,
        )
        fused_temporal, intent, next_state, diagnostics = model.core.step(
            core_input,
            batch.previous_action_global,
            batch.previous_effect,
            batch.economy,
            state,
        )
        strategy_context = model._strategy_embedding_for(
            intent, strategy_slots,
        )
        if strategy_context is not None:
            intent = intent + strategy_context
    else:
        encoded = precomputed["encoded"]
        fused_temporal = precomputed["fused_temporal"]
        intent = precomputed["intent"]
        next_state = precomputed["temporal_state"]
        diagnostics = precomputed["temporal_diagnostics"]
        strategy_context = precomputed.get("strategy_context")

    device = fused_temporal.device
    target_device = targets.to(device)
    expert_ledger = initial_ledger.to(device).clone()
    shared_conditioning_ledger = float(teacher_mix_probability) >= 1.0
    conditioning_ledger = (
        expert_ledger
        if shared_conditioning_ledger
        else initial_ledger.to(device).clone()
    )
    expert_ledger.set_atomic_plant_blocked(
        target_device.atomic_plant_blocked(expert_ledger.seeds)
    )
    batch_size = targets.batch_size
    rows = torch.arange(batch_size, device=device)
    decoder_hidden = model.decoder_init(torch.cat([
        fused_temporal, intent,
    ], dim=-1))
    previous_emb = model.start_action.to(fused_temporal).unsqueeze(0).expand(
        batch_size, -1
    )
    unit_counts = target_device.unit_mask.sum(dim=1).clamp_min(1)

    unit_op_logits = []
    unit_item_logits = []
    unit_quantity_logits = []
    unit_legal_op_mask = []

    unit_item_op_ids = None
    unit_quantity_op_ids = None
    if not shared_conditioning_ledger:
        unit_item_op_ids = torch.tensor(
            [
                UNIT_OP_TO_ID["PICKUP"],
                UNIT_OP_TO_ID["PLACE"],
                UNIT_OP_TO_ID["PLANT"],
            ],
            device=device,
            dtype=torch.long,
        )
        unit_quantity_op_ids = torch.tensor(
            [
                UNIT_OP_TO_ID["PICKUP"],
                UNIT_OP_TO_ID["PLACE"],
            ],
            device=device,
            dtype=torch.long,
        )

    for actor_index in range(target_device.max_units):
        active = target_device.unit_mask[:, actor_index]
        remaining_units = (
            unit_counts - int(actor_index) - 1
        ).clamp_min(0).to(fused_temporal.dtype) / unit_counts.to(
            fused_temporal.dtype
        )
        remaining_market = torch.ones(
            batch_size, device=device, dtype=fused_temporal.dtype,
        )
        expert_legal = expert_ledger.legal_unit(actor_index)
        conditioning_legal = conditioning_ledger.legal_unit(actor_index)
        candidate_hidden = _decode_input_tensor(
            model,
            actor_ctx=encoded.own_unit_ctx[:, actor_index],
            previous=previous_emb,
            hidden=decoder_hidden,
            global_h=fused_temporal,
            intent=intent,
            ledger=conditioning_ledger,
            remaining_units=remaining_units,
            remaining_market=remaining_market,
        )
        op_logits = model.unit_op_head(candidate_hidden)
        item_logits = model.item_head(candidate_hidden)
        quantity_logits, _ = _teacher_quantity_logits_tensor(
            model,
            candidate_hidden,
            target_device.unit_quantity_tokens[:, actor_index],
            target_device.unit_quantity_token_mask[:, actor_index],
        )
        unit_op_logits.append(op_logits)
        unit_item_logits.append(item_logits)
        unit_quantity_logits.append(quantity_logits)
        unit_legal_op_mask.append(expert_legal.op_mask)

        if shared_conditioning_ledger:
            conditioning_op = target_device.unit_op[:, actor_index]
            conditioning_item = target_device.unit_item[:, actor_index]
            conditioning_quantity = target_device.unit_quantity[:, actor_index]
        else:
            sampled_op = _masked_argmax(
                op_logits, conditioning_legal.op_mask
            )
            sampled_item_mask = conditioning_legal.item_mask[
                rows, sampled_op
            ]
            sampled_item = _masked_argmax(
                item_logits, sampled_item_mask
            )
            needs_item = sampled_op.unsqueeze(1).eq(
                unit_item_op_ids.unsqueeze(0)
            ).any(dim=1)
            sampled_item = torch.where(
                needs_item, sampled_item, torch.zeros_like(sampled_item)
            )
            quantity_max = conditioning_legal.quantity_max[
                rows,
                sampled_op,
                sampled_item.clamp(
                    0, conditioning_legal.quantity_max.shape[-1] - 1
                ),
            ]
            needs_quantity = sampled_op.unsqueeze(1).eq(
                unit_quantity_op_ids.unsqueeze(0)
            ).any(dim=1) & active
            sampled_quantity = _sample_quantity_argmax_tensor(
                model,
                candidate_hidden,
                positive=torch.zeros_like(active),
                max_value=quantity_max,
                active=needs_quantity,
                max_steps=target_device.unit_quantity_tokens.shape[-1],
            )
            sampled_quantity = torch.where(
                needs_quantity,
                sampled_quantity,
                torch.full_like(sampled_quantity, -1),
            )

            use_teacher = _teacher_choice_mask(
                active,
                teacher_mix_probability,
                generator=generator,
            )
            conditioning_op = torch.where(
                use_teacher,
                target_device.unit_op[:, actor_index],
                sampled_op,
            )
            conditioning_item = torch.where(
                use_teacher,
                target_device.unit_item[:, actor_index],
                sampled_item,
            )
            conditioning_quantity = torch.where(
                use_teacher,
                target_device.unit_quantity[:, actor_index],
                sampled_quantity,
            )

        decoder_hidden = torch.where(
            active.unsqueeze(-1),
            candidate_hidden,
            decoder_hidden,
        )
        next_previous = semantic_embedding_tensor(
            model,
            conditioning_op,
            conditioning_item,
            conditioning_quantity,
            domain="unit",
            ref=candidate_hidden,
        )
        previous_emb = torch.where(
            active.unsqueeze(-1),
            next_previous,
            previous_emb,
        )
        expert_ledger.apply_unit(
            actor_index,
            target_device.unit_op[:, actor_index],
            target_device.unit_item[:, actor_index],
            target_device.unit_quantity[:, actor_index],
            active=active,
        )
        if not shared_conditioning_ledger:
            conditioning_ledger.apply_unit(
                actor_index,
                conditioning_op,
                conditioning_item,
                conditioning_quantity,
                active=active,
            )

    market_continue_logits = []
    market_active_logits = []
    market_item_logits = []
    market_quantity_logits = []
    market_continue_legal_mask = []
    market_active_legal_mask = []
    active_to_market = None
    market_item_ops = None
    if not shared_conditioning_ledger:
        active_to_market = torch.tensor(
            [MARKET_OP_TO_ID[name] for name in ACTIVE_MARKET_OPS],
            device=device,
            dtype=torch.long,
        )
        market_item_ops = torch.tensor(
            [
                MARKET_OP_TO_ID["BUY_SEED"],
                MARKET_OP_TO_ID["BUY_PRODUCT"],
                MARKET_OP_TO_ID["BUY_ANIMAL"],
                MARKET_OP_TO_ID["SELL"],
            ],
            device=device,
            dtype=torch.long,
        )

    for slot in range(target_device.market_op.shape[1]):
        active = target_device.market_mask[:, slot]
        expert_legal = expert_ledger.legal_market(slot)
        conditioning_legal = conditioning_ledger.legal_market(slot)
        slot_ids = torch.full(
            (batch_size,), slot, dtype=torch.long, device=device
        )
        actor_ctx = model.market_slot_embedding(slot_ids)
        candidate_hidden = _decode_input_tensor(
            model,
            actor_ctx=actor_ctx,
            previous=previous_emb,
            hidden=decoder_hidden,
            global_h=fused_temporal,
            intent=intent,
            ledger=conditioning_ledger,
            remaining_units=torch.zeros(
                batch_size, device=device, dtype=fused_temporal.dtype,
            ),
            remaining_market=torch.full(
                (batch_size,),
                float(max(0, 9 - slot)) / 10.0,
                device=device,
                dtype=fused_temporal.dtype,
            ),
        )
        continue_logits = model.market_continue_head(candidate_hidden)
        active_logits = model.market_active_op_head(candidate_hidden)
        economic_hook = getattr(model, "_economic_market_residual", None)
        if callable(economic_hook):
            economic_continue, economic_active = economic_hook(
                conditioning_ledger, candidate_hidden,
            )
            continue_logits = continue_logits + economic_continue
            active_logits = active_logits + economic_active
        if slot == 0 and strategy_context is not None:
            opening_residual = model.opening_strategy_head(
                strategy_context
            )
            opening_mask = expert_ledger.step.eq(0).to(
                active_logits.dtype
            ).unsqueeze(-1)
            active_logits = (
                active_logits
                + opening_mask
                * float(OPENING_ACTIVE_INTENT_SCALE)
                * opening_residual
            )
        item_logits = model.item_head(candidate_hidden)
        quantity_logits, _ = _teacher_quantity_logits_tensor(
            model,
            candidate_hidden,
            target_device.market_quantity_tokens[:, slot],
            target_device.market_quantity_token_mask[:, slot],
        )
        market_continue_logits.append(continue_logits)
        market_active_logits.append(active_logits)
        market_item_logits.append(item_logits)
        market_quantity_logits.append(quantity_logits)
        market_continue_legal_mask.append(
            expert_ledger.continue_market_mask(expert_legal)
        )
        market_active_legal_mask.append(
            expert_ledger.active_market_mask(expert_legal)
        )

        if shared_conditioning_ledger:
            conditioning_op = target_device.market_op[:, slot]
            conditioning_item = target_device.market_item[:, slot]
            conditioning_quantity = target_device.market_quantity[:, slot]
        else:
            conditioning_continue_mask = (
                conditioning_ledger.continue_market_mask(conditioning_legal)
            )
            sampled_continue = _masked_argmax(
                continue_logits, conditioning_continue_mask
            )
            conditioning_active_mask = (
                conditioning_ledger.active_market_mask(conditioning_legal)
            )
            sampled_active_id = _masked_argmax(
                active_logits, conditioning_active_mask
            )
            sampled_op = torch.where(
                sampled_continue.eq(0),
                torch.full_like(
                    sampled_active_id,
                    MARKET_OP_TO_ID["STOP_QUEUE"],
                ),
                active_to_market[sampled_active_id],
            )
            sampled_item_mask = conditioning_legal.item_mask[
                rows, sampled_op
            ]
            sampled_item = _masked_argmax(
                item_logits, sampled_item_mask
            )
            needs_item = sampled_op.unsqueeze(1).eq(
                market_item_ops.unsqueeze(0)
            ).any(dim=1)
            sampled_item = torch.where(
                needs_item, sampled_item, torch.zeros_like(sampled_item)
            )
            quantity_max = conditioning_legal.quantity_max[
                rows,
                sampled_op,
                sampled_item.clamp(
                    0, conditioning_legal.quantity_max.shape[-1] - 1
                ),
            ]
            needs_quantity = needs_item & active
            sampled_quantity = _sample_quantity_argmax_tensor(
                model,
                candidate_hidden,
                positive=torch.ones_like(active),
                max_value=quantity_max,
                active=needs_quantity,
                max_steps=target_device.market_quantity_tokens.shape[-1],
            )
            sampled_quantity = torch.where(
                needs_quantity,
                sampled_quantity,
                torch.full_like(sampled_quantity, -1),
            )

            use_teacher = _teacher_choice_mask(
                active,
                teacher_mix_probability,
                generator=generator,
            )
            conditioning_op = torch.where(
                use_teacher,
                target_device.market_op[:, slot],
                sampled_op,
            )
            conditioning_item = torch.where(
                use_teacher,
                target_device.market_item[:, slot],
                sampled_item,
            )
            conditioning_quantity = torch.where(
                use_teacher,
                target_device.market_quantity[:, slot],
                sampled_quantity,
            )

        decoder_hidden = torch.where(
            active.unsqueeze(-1),
            candidate_hidden,
            decoder_hidden,
        )
        next_previous = semantic_embedding_tensor(
            model,
            conditioning_op,
            conditioning_item,
            conditioning_quantity,
            domain="market",
            ref=candidate_hidden,
        )
        previous_emb = torch.where(
            active.unsqueeze(-1),
            next_previous,
            previous_emb,
        )
        expert_ledger.apply_market(
            target_device.market_op[:, slot],
            target_device.market_item[:, slot],
            target_device.market_quantity[:, slot],
            known_executed=target_device.market_executed[:, slot],
            active=active,
        )
        if not shared_conditioning_ledger:
            conditioning_ledger.apply_market(
                conditioning_op,
                conditioning_item,
                conditioning_quantity,
                known_executed=torch.zeros_like(active),
                active=active,
            )

    return TensorTeacherPolicyOutput(
        unit_op_logits=torch.stack(unit_op_logits, dim=1),
        unit_item_logits=torch.stack(unit_item_logits, dim=1),
        unit_quantity_logits=torch.stack(unit_quantity_logits, dim=1),
        unit_quantity_token_mask=target_device.unit_quantity_token_mask,
        unit_legal_op_mask=torch.stack(unit_legal_op_mask, dim=1),
        market_continue_logits=torch.stack(market_continue_logits, dim=1),
        market_active_logits=torch.stack(market_active_logits, dim=1),
        market_item_logits=torch.stack(market_item_logits, dim=1),
        market_quantity_logits=torch.stack(market_quantity_logits, dim=1),
        market_quantity_token_mask=target_device.market_quantity_token_mask,
        market_continue_legal_mask=torch.stack(
            market_continue_legal_mask, dim=1
        ),
        market_active_legal_mask=torch.stack(
            market_active_legal_mask, dim=1
        ),
        temporal_state=next_state,
        fused_temporal=fused_temporal,
        intent=intent,
        aux=model._auxiliary(
            encoded, fused_temporal, intent,
        ),
        temporal_diagnostics=diagnostics,
    )
