from __future__ import annotations

import random

import torch

from .v2_ledger import ShadowLedger
from .v2_structure_pretrain import (
    _decode_batched_step,
    mask_joint_action,
)


def structure_reconstruction_loss_v3(
    model,
    batch,
    encoded,
    fused_temporal,
    intent,
    mask_rate: float = 0.15,
    rng=None,
):
    rng = random.Random() if rng is None else rng
    masked_rows = [
        mask_joint_action(action, rng, mask_rate)
        for action in batch.canonical_actions
    ]
    if not any(row.masked_fields for row in masked_rows):
        return next(model.parameters()).sum() * 0.0
    decoder_hidden = model.decoder_init(
        torch.cat([fused_temporal, intent], dim=-1)
    )
    previous_emb = model.start_action.to(fused_temporal).unsqueeze(0).expand(
        fused_temporal.shape[0], -1,
    ).clone()
    conditioned = [row.conditioned_action for row in masked_rows]
    ledgers = [ShadowLedger.from_state(state) for state in batch.structured_states]
    loss_sum = fused_temporal.sum() * 0.0
    loss_count = 0
    rows = list(range(len(conditioned)))
    farmer_actions = [
        action.get("farmer") or {"op": "PASS"}
        for action in conditioned
    ]
    unit_counts = [
        1 + len(action.get("hands") or [])
        for action in conditioned
    ]
    remaining_units = [
        float(count - 1) / float(count)
        for count in unit_counts
    ]
    decoder_hidden, previous_emb, current_loss, current_count = _decode_batched_step(
        model,
        row_indices=rows,
        actions=farmer_actions,
        actor_ctx=encoded.own_unit_ctx[:, 0],
        actor_names=["farmer"] * len(rows),
        remaining_units=remaining_units,
        remaining_market=[1.0] * len(rows),
        decoder_hidden=decoder_hidden,
        previous_emb=previous_emb,
        global_h=fused_temporal,
        intent=intent,
        ledgers=ledgers,
        domain="unit",
    )
    loss_sum = loss_sum + current_loss
    loss_count += current_count
    max_hands = max(
        (len(action.get("hands") or []) for action in conditioned),
        default=0,
    )
    for hand_index in range(max_hands):
        active = [
            row for row, action in enumerate(conditioned)
            if hand_index < len(action.get("hands") or [])
        ]
        if not active:
            continue
        index = fused_temporal.new_tensor(active, dtype=torch.long)
        actions = [conditioned[row]["hands"][hand_index] for row in active]
        actor_ctx = encoded.own_unit_ctx.index_select(0, index)[:, hand_index + 1]
        remaining = [
            float(max(0, unit_counts[row] - hand_index - 2)) / float(unit_counts[row])
            for row in active
        ]
        decoder_hidden, previous_emb, current_loss, current_count = _decode_batched_step(
            model,
            row_indices=active,
            actions=actions,
            actor_ctx=actor_ctx,
            actor_names=[f"hand:{hand_index}"] * len(active),
            remaining_units=remaining,
            remaining_market=[1.0] * len(active),
            decoder_hidden=decoder_hidden,
            previous_emb=previous_emb,
            global_h=fused_temporal,
            intent=intent,
            ledgers=ledgers,
            domain="unit",
        )
        loss_sum = loss_sum + current_loss
        loss_count += current_count

    market_alive = [True] * len(conditioned)
    for slot in range(10):
        active = [
            row for row, action in enumerate(conditioned)
            if market_alive[row] and slot < len(action.get("market") or [])
        ]
        if not active:
            continue
        actions = [conditioned[row]["market"][slot] for row in active]
        slot_ids = fused_temporal.new_full(
            (len(active),), slot, dtype=torch.long,
        )
        actor_ctx = model.market_slot_embedding(slot_ids)
        decoder_hidden, previous_emb, current_loss, current_count = _decode_batched_step(
            model,
            row_indices=active,
            actions=actions,
            actor_ctx=actor_ctx,
            actor_names=["market"] * len(active),
            remaining_units=[0.0] * len(active),
            remaining_market=[float(max(0, 9 - slot)) / 10.0] * len(active),
            decoder_hidden=decoder_hidden,
            previous_emb=previous_emb,
            global_h=fused_temporal,
            intent=intent,
            ledgers=ledgers,
            domain="market",
        )
        loss_sum = loss_sum + current_loss
        loss_count += current_count
        for local_index, row in enumerate(active):
            if str(actions[local_index].get("kind", "ORDER")) == "STOP_QUEUE":
                market_alive[row] = False

    if loss_count <= 0:
        return next(model.parameters()).sum() * 0.0
    return loss_sum / float(loss_count)
