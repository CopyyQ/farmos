from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .constants import ITEM_TO_ID, UNIT_OPS
from .v2_ledger import MARKET_OPS, ShadowLedger
from .v2_tensorize import signed_log1p
from .v2_quantity import encode_quantity

UNIT_ITEM_OPS = {"PICKUP", "PLACE", "PLANT"}
UNIT_QUANTITY_OPS = {"PICKUP", "PLACE"}
MARKET_ITEM_OPS = {"BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"}
MARKET_QUANTITY_OPS = set(MARKET_ITEM_OPS)
UNIT_OP_TO_ID = {op: i for i, op in enumerate(UNIT_OPS)}
MARKET_OP_TO_ID = {op: i for i, op in enumerate(MARKET_OPS)}


@dataclass(frozen=True)
class MaskedAction:
    conditioned_action: dict[str, Any]
    target_action: dict[str, Any]
    masked_fields: tuple[tuple[str, int, str], ...]


def _draw(rng, rate: float) -> bool:
    return float(rng.random()) < float(rate)


def _mask_command(command: dict[str, Any], active_fields: tuple[str, ...], rng, rate: float,
                  domain: str, index: int, masked: list[tuple[str, int, str]]) -> dict[str, Any]:
    out = deepcopy(command)
    selected = []
    for field in active_fields:
        if _draw(rng, rate):
            selected.append(field)
            masked.append((domain, int(index), field))
    if selected:
        out["_mask_fields"] = tuple(selected)
    return out


def _unit_active_fields(command: dict[str, Any]) -> tuple[str, ...]:
    op = str(command.get("op", "PASS"))
    fields = ["op"]
    if op in UNIT_ITEM_OPS:
        fields.append("item")
    if op in UNIT_QUANTITY_OPS:
        fields.append("quantity")
    return tuple(fields)


def _market_op(command: dict[str, Any]) -> str:
    kind = str(command.get("kind", "ORDER"))
    if kind in {"STOP_QUEUE", "NOP_SLOT"}:
        return kind
    return str(command.get("op", "NOP_SLOT"))


def _market_active_fields(command: dict[str, Any]) -> tuple[str, ...]:
    op = _market_op(command)
    fields = ["op"]
    if op in MARKET_ITEM_OPS:
        fields.append("item")
    if op in MARKET_QUANTITY_OPS:
        fields.append("quantity")
    return tuple(fields)


def mask_joint_action(action: dict[str, Any], rng=None, rate: float = 0.15) -> MaskedAction:
    if not 0.0 <= float(rate) <= 1.0:
        raise ValueError("mask rate must be in [0,1]")
    rng = random.Random() if rng is None else rng
    target = deepcopy(action)
    conditioned = deepcopy(action)
    masked: list[tuple[str, int, str]] = []
    conditioned["farmer"] = _mask_command(
        conditioned.get("farmer") or {"op": "PASS"},
        _unit_active_fields(conditioned.get("farmer") or {"op": "PASS"}),
        rng, rate, "farmer", 0, masked,
    )
    hands = []
    for index, command in enumerate(conditioned.get("hands") or []):
        hands.append(_mask_command(
            command, _unit_active_fields(command), rng, rate, "hand", index, masked,
        ))
    conditioned["hands"] = hands
    market = []
    for index, command in enumerate(conditioned.get("market") or []):
        market.append(_mask_command(
            command, _market_active_fields(command), rng, rate, "market", index, masked,
        ))
    conditioned["market"] = market
    return MaskedAction(conditioned, target, tuple(masked))


def _ce(logits: torch.Tensor, target: int) -> torch.Tensor:
    return F.cross_entropy(logits.unsqueeze(0), logits.new_tensor([target], dtype=torch.long))


def _quantity_ce(logits: torch.Tensor | None, quantity: int | None) -> torch.Tensor:
    tokens = encode_quantity(quantity)
    if logits is None or logits.shape[0] < len(tokens):
        raise ValueError("missing quantity logits for masked quantity field")
    target = logits.new_tensor(tokens, dtype=torch.long)
    return F.cross_entropy(logits[:len(tokens)], target)


def _field_loss(decision, target: dict[str, Any], domain: str, field: str) -> torch.Tensor:
    if domain in {"farmer", "hand"}:
        op = str(target.get("op", "PASS"))
        if field == "op":
            return _ce(decision.op_logits, UNIT_OP_TO_ID[op])
    else:
        op = _market_op(target)
        if field == "op":
            return _ce(decision.op_logits, MARKET_OP_TO_ID[op])
    if field == "item":
        item = target.get("item")
        if item not in ITEM_TO_ID:
            raise ValueError(f"masked item target is invalid: {item}")
        return _ce(decision.item_logits, ITEM_TO_ID[item])
    if field == "quantity":
        return _quantity_ce(decision.quantity_logits, target.get("quantity"))
    raise ValueError(f"unsupported masked field: {field}")


def structure_reconstruction_loss_reference(model, batch, mask_rate: float = 0.15, rng=None) -> torch.Tensor:
    rng = random.Random() if rng is None else rng
    masked_rows = [mask_joint_action(action, rng, mask_rate) for action in batch.canonical_actions]
    if not any(row.masked_fields for row in masked_rows):
        return next(model.parameters()).sum() * 0.0
    output = model.forward_sequence(
        batch, teacher_actions=tuple(row.conditioned_action for row in masked_rows)
    )
    losses: list[torch.Tensor] = []
    for row_output, masked in zip(output.rows, masked_rows):
        target = masked.target_action
        for domain, index, field in masked.masked_fields:
            if domain == "farmer":
                decision = row_output.farmer
                command = target["farmer"]
            elif domain == "hand":
                decision = row_output.hands[index]
                command = target["hands"][index]
            else:
                decision = row_output.market[index]
                command = target["market"][index]
            losses.append(_field_loss(decision, command, domain, field))
    if not losses:
        return next(model.parameters()).sum() * 0.0
    return torch.stack(losses).mean()


def _ledger_values(ledger: ShadowLedger) -> list[float]:
    next_land = ledger.next_land_cost
    return [
        signed_log1p(ledger.cash_lower_bound), float(ledger.cash_uncertain),
        signed_log1p(ledger.hires_today), signed_log1p(ledger.next_hire_cost),
        signed_log1p(next_land or 0), 1.0 if next_land is None else 0.0,
        signed_log1p(sum(max(0, int(v)) for v in ledger.shed.values())),
        signed_log1p(ledger._shed_room()), float(ledger.shed_uncertain),
        signed_log1p(sum(max(0, int(v)) for v in ledger.plant_demand.values())),
        float(ledger.market_slots_used) / 10.0, float(ledger.market_stopped),
    ]


def _batched_action_embedding(model, actions, domain: str, ref: torch.Tensor) -> torch.Tensor:
    op_indices, item_indices, quantity_features = [], [], []
    op_masked, item_masked, quantity_masked = [], [], []
    for action in actions:
        masked = set(action.get("_mask_fields") or ())
        if domain == "unit":
            op = str(action.get("op", "PASS"))
            op_index = 1 + model.unit_op_to_id.get(op, model.unit_op_to_id["PASS"])
        else:
            op = _market_op(action)
            op_index = 1 + len(UNIT_OPS) + model.market_op_to_id.get(op, model.market_op_to_id["NOP_SLOT"])
        item = action.get("item")
        quantity = action.get("quantity")
        op_indices.append(op_index)
        item_indices.append(int(ITEM_TO_ID.get(str(item), 0)) if item is not None else 0)
        quantity_features.append([
            1.0 if quantity is None else 0.0,
            signed_log1p(quantity if quantity is not None else 0),
        ])
        op_masked.append("op" in masked)
        item_masked.append("item" in masked)
        quantity_masked.append("quantity" in masked)
    op_ids = ref.new_tensor(op_indices, dtype=torch.long)
    item_ids = ref.new_tensor(item_indices, dtype=torch.long)
    op_emb = model.op_embedding(op_ids)
    item_emb = model.item_embedding(item_ids)
    op_mask = ref.new_tensor(op_masked, dtype=torch.bool).unsqueeze(-1)
    item_mask = ref.new_tensor(item_masked, dtype=torch.bool).unsqueeze(-1)
    op_emb = torch.where(op_mask, model.mask_op_embedding.to(ref).unsqueeze(0), op_emb)
    item_emb = torch.where(item_mask, model.mask_item_embedding.to(ref).unsqueeze(0), item_emb)
    quantity = ref.new_tensor(quantity_features)
    quantity_mask = ref.new_tensor(quantity_masked, dtype=torch.bool).unsqueeze(-1)
    masked_quantity = model.mask_quantity_features.to(ref).unsqueeze(0).expand_as(quantity)
    quantity = torch.where(quantity_mask, masked_quantity, quantity)
    return model.action_proj(torch.cat([op_emb, item_emb, quantity], dim=-1))


def _masked_field_losses(model, hidden: torch.Tensor, actions, domain: str):
    total = hidden.sum() * 0.0
    count = 0
    op_rows = [i for i, action in enumerate(actions) if "op" in set(action.get("_mask_fields") or ())]
    if op_rows:
        indices = hidden.new_tensor(op_rows, dtype=torch.long)
        selected = hidden.index_select(0, indices)
        logits = model.unit_op_head(selected) if domain == "unit" else model.market_op_head(selected)
        if domain == "unit":
            targets = [UNIT_OP_TO_ID[str(actions[i].get("op", "PASS"))] for i in op_rows]
        else:
            targets = [MARKET_OP_TO_ID[_market_op(actions[i])] for i in op_rows]
        target = logits.new_tensor(targets, dtype=torch.long)
        total = total + F.cross_entropy(logits, target, reduction="sum")
        count += len(op_rows)
    item_rows = [i for i, action in enumerate(actions) if "item" in set(action.get("_mask_fields") or ())]
    if item_rows:
        indices = hidden.new_tensor(item_rows, dtype=torch.long)
        logits = model.item_head(hidden.index_select(0, indices))
        targets = [ITEM_TO_ID[actions[i]["item"]] for i in item_rows]
        total = total + F.cross_entropy(logits, logits.new_tensor(targets, dtype=torch.long), reduction="sum")
        count += len(item_rows)
    quantity_rows = [i for i, action in enumerate(actions) if "quantity" in set(action.get("_mask_fields") or ())]
    if quantity_rows:
        indices = hidden.new_tensor(quantity_rows, dtype=torch.long)
        selected = hidden.index_select(0, indices)
        contexts = model.quantity_context(selected)
        targets = [encode_quantity(actions[i].get("quantity")) for i in quantity_rows]
        logits, _ = model.quantity_decoder.teacher_logits(contexts, targets)
        for row_index, tokens in enumerate(targets):
            target = logits.new_tensor(tokens, dtype=torch.long)
            total = total + F.cross_entropy(logits[row_index, :len(tokens)], target)
            count += 1
    return total, count


def _decode_batched_step(model, *, row_indices, actions, actor_ctx, actor_names,
                         remaining_units, remaining_market, decoder_hidden,
                         previous_emb, global_h, intent, ledgers, domain):
    index = decoder_hidden.new_tensor(row_indices, dtype=torch.long)
    current_hidden = decoder_hidden.index_select(0, index)
    current_previous = previous_emb.index_select(0, index)
    ledger_tensor = actor_ctx.new_tensor([_ledger_values(ledgers[row]) for row in row_indices])
    ledger_ctx = model.ledger_proj(ledger_tensor)
    tail = actor_ctx.new_tensor(list(zip(remaining_units, remaining_market)))
    decoder_input = torch.cat([
        actor_ctx, current_previous, ledger_ctx,
        global_h.index_select(0, index), intent.index_select(0, index), tail,
    ], dim=-1)
    next_hidden = model.decoder_cell(decoder_input, current_hidden)
    loss_sum, loss_count = _masked_field_losses(model, next_hidden, actions, domain)
    next_previous = _batched_action_embedding(model, actions, domain, next_hidden)
    decoder_hidden = decoder_hidden.index_copy(0, index, next_hidden)
    previous_emb = previous_emb.index_copy(0, index, next_previous)
    for local_index, row in enumerate(row_indices):
        action = actions[local_index]
        if action.get("_mask_fields"):
            continue
        if domain == "unit":
            ledgers[row].apply_unit(actor_names[local_index], action)
        else:
            ledgers[row].apply_market(action)
    return decoder_hidden, previous_emb, loss_sum, loss_count


def structure_reconstruction_loss(model, batch, mask_rate: float = 0.15, rng=None) -> torch.Tensor:
    rng = random.Random() if rng is None else rng
    masked_rows = [mask_joint_action(action, rng, mask_rate) for action in batch.canonical_actions]
    if not any(row.masked_fields for row in masked_rows):
        return next(model.parameters()).sum() * 0.0
    encoded = model.encoder(batch)
    global_h, _, intent = model.core.step(encoded.fused, None)
    decoder_hidden = model.decoder_init(torch.cat([global_h, intent], dim=-1))
    previous_emb = model.start_action.to(global_h).unsqueeze(0).expand(global_h.shape[0], -1).clone()
    conditioned = [row.conditioned_action for row in masked_rows]
    ledgers = [ShadowLedger.from_state(state) for state in batch.structured_states]
    loss_sum = global_h.sum() * 0.0
    loss_count = 0
    rows = list(range(len(conditioned)))
    farmer_actions = [action.get("farmer") or {"op": "PASS"} for action in conditioned]
    unit_counts = [1 + len(action.get("hands") or []) for action in conditioned]
    remaining_units = [float(count - 1) / float(count) for count in unit_counts]
    decoder_hidden, previous_emb, current_loss, current_count = _decode_batched_step(
        model, row_indices=rows, actions=farmer_actions,
        actor_ctx=encoded.own_unit_ctx[:, 0], actor_names=["farmer"] * len(rows),
        remaining_units=remaining_units, remaining_market=[1.0] * len(rows),
        decoder_hidden=decoder_hidden, previous_emb=previous_emb,
        global_h=global_h, intent=intent, ledgers=ledgers, domain="unit",
    )
    loss_sum = loss_sum + current_loss
    loss_count += current_count

    max_hands = max((len(action.get("hands") or []) for action in conditioned), default=0)
    for hand_index in range(max_hands):
        active = [row for row, action in enumerate(conditioned)
                  if hand_index < len(action.get("hands") or [])]
        if not active:
            continue
        active_tensor = global_h.new_tensor(active, dtype=torch.long)
        actions = [conditioned[row]["hands"][hand_index] for row in active]
        actor_ctx = encoded.own_unit_ctx.index_select(0, active_tensor)[:, hand_index + 1]
        remaining = [
            float(max(0, unit_counts[row] - hand_index - 2)) / float(unit_counts[row])
            for row in active
        ]
        decoder_hidden, previous_emb, current_loss, current_count = _decode_batched_step(
            model, row_indices=active, actions=actions, actor_ctx=actor_ctx,
            actor_names=[f"hand:{hand_index}"] * len(active),
            remaining_units=remaining, remaining_market=[1.0] * len(active),
            decoder_hidden=decoder_hidden, previous_emb=previous_emb,
            global_h=global_h, intent=intent, ledgers=ledgers, domain="unit",
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
        slot_ids = global_h.new_full((len(active),), slot, dtype=torch.long)
        actor_ctx = model.market_slot_embedding(slot_ids)
        decoder_hidden, previous_emb, current_loss, current_count = _decode_batched_step(
            model, row_indices=active, actions=actions, actor_ctx=actor_ctx,
            actor_names=["market"] * len(active), remaining_units=[0.0] * len(active),
            remaining_market=[float(max(0, 9 - slot)) / 10.0] * len(active),
            decoder_hidden=decoder_hidden, previous_emb=previous_emb,
            global_h=global_h, intent=intent, ledgers=ledgers, domain="market",
        )
        loss_sum = loss_sum + current_loss
        loss_count += current_count
        for local_index, row in enumerate(active):
            if str(actions[local_index].get("kind", "ORDER")) == "STOP_QUEUE":
                market_alive[row] = False
    if loss_count <= 0:
        return next(model.parameters()).sum() * 0.0
    return loss_sum / float(loss_count)
