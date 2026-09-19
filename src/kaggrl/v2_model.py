from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from .constants import ITEM_TO_ID, UNIT_OPS
from .v2_encoder import RecurrentCore, StateEncoder
from .v2_ledger import MARKET_OPS, ShadowLedger
from .v2_quantity import (
    DIGIT_OFFSET,
    END_ID,
    OMIT_ID,
    START_ID,
    QuantityDecoder,
    decode_quantity,
    encode_quantity,
)
from .v2_tensorize import EFFECT_FEATURES, V2Batch, signed_log1p

ITEM_NAMES = tuple(ITEM_TO_ID.keys())
ITEM_CLASSES = max(ITEM_TO_ID.values()) + 1
UNIT_ITEM_OPS = {"PICKUP", "PLACE", "PLANT"}
UNIT_QUANTITY_OPS = {"PICKUP", "PLACE"}
MARKET_ITEM_OPS = {"BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"}
MARKET_QUANTITY_OPS = set(MARKET_ITEM_OPS)


@dataclass
class DecisionOutput:
    op_logits: torch.Tensor
    item_logits: torch.Tensor
    quantity_logits: torch.Tensor | None
    quantity_tokens: tuple[int, ...] | None
    chosen_action: dict[str, Any]
    legal_op_mask: torch.Tensor
    legal_item_mask: torch.Tensor | None
    continue_logits: torch.Tensor | None = None
    legal_continue_mask: torch.Tensor | None = None
    quantity_max_value: int | None = None


@dataclass
class DecodeTrace:
    domain: str
    index: int
    previous_semantic: str
    chosen_semantic: str
    farmer_position_before: tuple[int, int]
    hires_today_before: int
    cash_lower_bound_before: int
    market_slots_used_before: int
    teacher_forced: bool


@dataclass
class RowPolicyOutput:
    farmer: DecisionOutput
    hands: tuple[DecisionOutput, ...]
    market: tuple[DecisionOutput, ...]
    trace: tuple[DecodeTrace, ...]


@dataclass
class AuxiliaryOutputs:
    effect: torch.Tensor
    future_resource: torch.Tensor
    unit_task: torch.Tensor
    opponent_effect: torch.Tensor
    terminal_money: torch.Tensor
    terminal_margin: torch.Tensor


@dataclass
class PolicyOutput:
    rows: tuple[RowPolicyOutput, ...]
    recurrent_state: tuple[torch.Tensor, torch.Tensor]
    intent: torch.Tensor
    aux: AuxiliaryOutputs


class RecurrentIntentPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = StateEncoder()
        self.core = RecurrentCore()
        self.unit_op_to_id = {op: i for i, op in enumerate(UNIT_OPS)}
        self.market_op_to_id = {op: i for i, op in enumerate(MARKET_OPS)}
        self.unit_id_to_op = {i: op for op, i in self.unit_op_to_id.items()}
        self.market_id_to_op = {i: op for op, i in self.market_op_to_id.items()}
        self.op_embedding = nn.Embedding(1 + len(UNIT_OPS) + len(MARKET_OPS), 32)
        self.item_embedding = nn.Embedding(ITEM_CLASSES, 16)
        self.action_proj = nn.Sequential(nn.Linear(50, 64), nn.SiLU())
        self.start_action = nn.Parameter(torch.zeros(64))
        self.mask_op_embedding = nn.Parameter(torch.empty(32))
        self.mask_item_embedding = nn.Parameter(torch.empty(16))
        self.mask_quantity_features = nn.Parameter(torch.empty(2))
        nn.init.normal_(self.mask_op_embedding, std=0.02)
        nn.init.normal_(self.mask_item_embedding, std=0.02)
        nn.init.normal_(self.mask_quantity_features, std=0.02)
        self.market_slot_embedding = nn.Embedding(10, 128)
        self.decoder_init = nn.Sequential(nn.Linear(384, 192), nn.Tanh())
        self.ledger_proj = nn.Sequential(nn.Linear(12, 32), nn.SiLU())
        self.decoder_cell = nn.GRUCell(610, 192)
        self.unit_op_head = nn.Linear(192, len(UNIT_OPS))
        self.market_op_head = nn.Linear(192, len(MARKET_OPS))
        self.item_head = nn.Linear(192, ITEM_CLASSES)
        self.quantity_context = nn.Sequential(nn.Linear(192, 96), nn.Tanh())
        self.quantity_decoder = QuantityDecoder(96, hidden_dim=64, token_dim=16, max_digits=5)
        self.effect_head = nn.Linear(384, len(EFFECT_FEATURES))
        self.future_resource_head = nn.Linear(384, 32)
        self.unit_task_head = nn.Linear(128, 16)
        self.opponent_effect_head = nn.Linear(384, 16)
        self.terminal_money_head = nn.Linear(384, 1)
        self.terminal_margin_head = nn.Linear(384, 1)

    def _parts(self, action: dict[str, Any], domain: str) -> tuple[str, str | None, int | None]:
        if domain == "unit":
            return str(action.get("op", "PASS")), action.get("item"), action.get("quantity")
        kind = str(action.get("kind", "ORDER"))
        if kind in {"STOP_QUEUE", "NOP_SLOT"}:
            return kind, None, None
        return str(action.get("op", "NOP_SLOT")), action.get("item"), action.get("quantity")

    def _semantic_label(self, action: dict[str, Any], domain: str) -> str:
        op, _, _ = self._parts(action, domain)
        return ("U:" if domain == "unit" else "M:") + op

    def _semantic_embedding(self, action: dict[str, Any], domain: str, ref: torch.Tensor) -> torch.Tensor:
        op, item, quantity = self._parts(action, domain)
        masked = set(action.get("_mask_fields") or [])
        if domain == "unit":
            op_index = 1 + self.unit_op_to_id.get(op, self.unit_op_to_id["PASS"])
        else:
            op_index = 1 + len(UNIT_OPS) + self.market_op_to_id.get(op, self.market_op_to_id["NOP_SLOT"])
        item_index = int(ITEM_TO_ID.get(str(item), 0)) if item is not None else 0
        if "quantity" in masked:
            quantity_features = self.mask_quantity_features.to(ref)
        else:
            quantity_features = ref.new_tensor([
                1.0 if quantity is None else 0.0,
                signed_log1p(quantity if quantity is not None else 0),
            ])
        op_emb = (self.mask_op_embedding.to(ref) if "op" in masked else
                  self.op_embedding(ref.new_tensor(op_index, dtype=torch.long)))
        item_emb = (self.mask_item_embedding.to(ref) if "item" in masked else
                    self.item_embedding(ref.new_tensor(item_index, dtype=torch.long)))
        return self.action_proj(torch.cat([op_emb, item_emb, quantity_features], dim=-1))

    @staticmethod
    def _ledger_vector(ledger: ShadowLedger, ref: torch.Tensor) -> torch.Tensor:
        next_land = ledger.next_land_cost
        values = [
            signed_log1p(ledger.cash_lower_bound), float(ledger.cash_uncertain),
            signed_log1p(ledger.hires_today), signed_log1p(ledger.next_hire_cost),
            signed_log1p(next_land or 0), 1.0 if next_land is None else 0.0,
            signed_log1p(sum(max(0, int(v)) for v in ledger.shed.values())),
            signed_log1p(ledger._shed_room()), float(ledger.shed_uncertain),
            signed_log1p(sum(max(0, int(v)) for v in ledger.plant_demand.values())),
            float(ledger.market_slots_used) / 10.0, float(ledger.market_stopped),
        ]
        return ref.new_tensor(values)

    @staticmethod
    def _snapshot_trace(ledger: ShadowLedger, domain: str, index: int,
                        previous: str, chosen: str, teacher_forced: bool) -> DecodeTrace:
        farmer = ledger.unit_positions.get("farmer", [0, 0])
        return DecodeTrace(
            domain=domain, index=int(index), previous_semantic=previous,
            chosen_semantic=chosen,
            farmer_position_before=(int(farmer[0]), int(farmer[1])),
            hires_today_before=int(ledger.hires_today),
            cash_lower_bound_before=int(ledger.cash_lower_bound),
            market_slots_used_before=int(ledger.market_slots_used),
            teacher_forced=bool(teacher_forced),
        )

    @staticmethod
    def _masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if not bool(mask.any()):
            raise RuntimeError("decoder received an empty legal mask")
        # Keep legal masking representable under CUDA AMP/float16.
        return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)

    @staticmethod
    def _choose(logits: torch.Tensor, mask: torch.Tensor, rng, deterministic: bool) -> int:
        masked = RecurrentIntentPolicy._masked_logits(logits, mask)
        if deterministic:
            return int(masked.argmax().item())
        probs = torch.softmax(masked.detach().float(), dim=-1).cpu().numpy()
        if rng is not None and hasattr(rng, "choice"):
            return int(rng.choice(len(probs), p=probs))
        return int(torch.multinomial(torch.as_tensor(probs), 1).item())

    @staticmethod
    def _teacher_conditioning_choice(
        teacher_action: dict[str, Any] | None,
        probability: float,
        rng,
    ) -> bool:
        if teacher_action is None:
            return False
        p = float(probability)
        if not 0.0 <= p <= 1.0:
            raise ValueError("teacher_mix_probability must be in [0, 1]")
        if p >= 1.0:
            return True
        if p <= 0.0:
            return False
        if rng is None or not hasattr(rng, "random"):
            raise ValueError("conditioning_rng with random() is required for mixed teacher conditioning")
        return bool(float(rng.random()) < p)

    def _decode_input(self, actor_ctx: torch.Tensor, previous: torch.Tensor,
                      hidden: torch.Tensor, global_h: torch.Tensor,
                      intent: torch.Tensor, ledger: ShadowLedger,
                      remaining_units: float, remaining_market: float) -> torch.Tensor:
        ledger_ctx = self.ledger_proj(self._ledger_vector(ledger, actor_ctx))
        tail = actor_ctx.new_tensor([float(remaining_units), float(remaining_market)])
        decoder_input = torch.cat([
            actor_ctx, previous, ledger_ctx, global_h, intent, tail,
        ], dim=-1).unsqueeze(0)
        return self.decoder_cell(decoder_input, hidden.unsqueeze(0)).squeeze(0)

    def _teacher_quantity(self, hidden: torch.Tensor, quantity: int | None) -> tuple[torch.Tensor, tuple[int, ...]]:
        tokens = encode_quantity(quantity)
        context = self.quantity_context(hidden).unsqueeze(0)
        logits, mask = self.quantity_decoder.teacher_logits(context, [tokens])
        return logits[0, mask[0]], tokens

    def _sample_quantity(self, hidden: torch.Tensor, *, positive: bool, rng,
                         deterministic: bool, max_value: int | None = None
                         ) -> tuple[torch.Tensor, tuple[int, ...], int | None]:
        if max_value is not None:
            max_value = int(max_value)
            if positive and max_value < 1:
                raise ValueError("positive bounded quantity requires max_value >= 1")
            if max_value < 0:
                raise ValueError("quantity max_value must be nonnegative")
        context = self.quantity_context(hidden).unsqueeze(0)
        qhidden = self.quantity_decoder.initial_state(context)
        previous = torch.full((1,), START_ID, dtype=torch.long, device=hidden.device)
        tokens: list[int] = []
        logits_history: list[torch.Tensor] = []
        digits: list[int] = []
        for step_index in range(self.quantity_decoder.max_digits + 1):
            logits, qhidden = self.quantity_decoder.step(qhidden, previous)
            raw = logits[0]

            allowed_ids: set[int] = set()
            if step_index == 0:
                if not positive:
                    allowed_ids.add(OMIT_ID)
                    allowed_ids.update(DIGIT_OFFSET + digit for digit in range(10))
                else:
                    allowed_ids.update(DIGIT_OFFSET + digit for digit in range(1, 10))
            else:
                if len(digits) >= self.quantity_decoder.max_digits or (
                    len(digits) == 1 and digits[0] == 0
                ):
                    allowed_ids.add(END_ID)
                else:
                    allowed_ids.update(DIGIT_OFFSET + digit for digit in range(10))
                    allowed_ids.add(END_ID)

            if max_value is not None:
                bounded_ids: set[int] = set()
                for token_id in allowed_ids:
                    if token_id == OMIT_ID:
                        if not positive:
                            bounded_ids.add(token_id)
                        continue
                    if token_id == END_ID:
                        if digits:
                            current = int("".join(str(value) for value in digits))
                            if current <= max_value:
                                bounded_ids.add(token_id)
                        continue
                    digit = int(token_id) - DIGIT_OFFSET
                    if not digits and digit == 0 and positive:
                        continue
                    candidate_digits = [*digits, digit]
                    candidate = int("".join(str(value) for value in candidate_digits))
                    if candidate <= max_value:
                        bounded_ids.add(token_id)
                allowed_ids = bounded_ids

            if not allowed_ids:
                raise RuntimeError("quantity decoder has no legal token under current bound")
            allowed = torch.zeros_like(raw, dtype=torch.bool)
            allowed[list(sorted(allowed_ids))] = True
            token = self._choose(raw, allowed, rng, deterministic)
            logits_history.append(raw)
            tokens.append(token)
            if token == OMIT_ID or token == END_ID:
                break
            digits.append(int(token) - DIGIT_OFFSET)
            previous = previous.new_tensor([token])
        value = decode_quantity(tokens, max_digits=self.quantity_decoder.max_digits)
        if max_value is not None and value is not None and int(value) > max_value:
            raise RuntimeError("bounded quantity decoder exceeded max_value")
        return torch.stack(logits_history, dim=0), tuple(tokens), value

    def _item_mask(self, choices: dict[str, bool], ref: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros((ITEM_CLASSES,), dtype=torch.bool, device=ref.device)
        for item, allowed in choices.items():
            item_id = ITEM_TO_ID.get(str(item))
            if item_id is not None and allowed:
                mask[int(item_id)] = True
        return mask

    @staticmethod
    def _quantity_max_from_legal(legal, domain: str, op: str, item: str | None) -> int | None:
        key = (
            "unit_quantity_max_by_op_item"
            if domain == "unit"
            else "market_quantity_max_by_op_item"
        )
        by_op = (legal.metadata.get(key) or {}).get(str(op), {})
        if item is None or str(item) not in by_op:
            return None
        value = int(by_op[str(item)])
        return value if value >= 0 else None

    def _unit_conditioning_action(
        self, *, decoder_hidden: torch.Tensor, op_logits: torch.Tensor,
        item_logits: torch.Tensor, op_mask: torch.Tensor, legal, rng,
        deterministic: bool,
    ) -> dict[str, Any]:
        op = self.unit_id_to_op[self._choose(op_logits, op_mask, rng, deterministic)]
        chosen = {"op": op, "item": None, "quantity": None, "raw": [op]}
        if op in UNIT_ITEM_OPS:
            item_mask = self._item_mask(legal.items.get(op, {}), item_logits)
            item_id = self._choose(item_logits, item_mask, rng, deterministic)
            chosen["item"] = next(name for name, idx in ITEM_TO_ID.items() if idx == item_id)
        if op in UNIT_QUANTITY_OPS:
            max_value = self._quantity_max_from_legal(
                legal, "unit", op, chosen.get("item"),
            )
            _, _, quantity = self._sample_quantity(
                decoder_hidden, positive=False, rng=rng, deterministic=deterministic,
                max_value=max_value,
            )
            chosen["quantity"] = quantity
            chosen["raw"] = [op, chosen.get("item")] + ([] if quantity is None else [quantity])
        elif op == "PLANT":
            chosen["raw"] = [op, chosen.get("item")]
        return chosen

    def _market_conditioning_action(
        self, *, decoder_hidden: torch.Tensor, op_logits: torch.Tensor,
        item_logits: torch.Tensor, op_mask: torch.Tensor, legal, rng,
        deterministic: bool,
    ) -> dict[str, Any]:
        op = self.market_id_to_op[self._choose(op_logits, op_mask, rng, deterministic)]
        if op in {"STOP_QUEUE", "NOP_SLOT"}:
            return {"kind": op, "op": None, "item": None, "quantity": None, "raw": []}
        chosen = {"kind": "ORDER", "op": op, "item": None, "quantity": None, "raw": [op]}
        if op in MARKET_ITEM_OPS:
            item_mask = self._item_mask(legal.items.get(op, {}), item_logits)
            item_id = self._choose(item_logits, item_mask, rng, deterministic)
            chosen["item"] = next(name for name, idx in ITEM_TO_ID.items() if idx == item_id)
        if op in MARKET_QUANTITY_OPS:
            max_value = self._quantity_max_from_legal(
                legal, "market", op, chosen.get("item"),
            )
            if op == "SELL" and (max_value is None or max_value < 1):
                raise RuntimeError("SELL conditioning action has no legal inventory bound")
            _, _, quantity = self._sample_quantity(
                decoder_hidden,
                positive=True,
                rng=rng,
                deterministic=deterministic,
                max_value=max_value,
            )
            chosen["quantity"] = quantity
            chosen["raw"] = [op, chosen.get("item"), quantity]
        return chosen

    def _unit_decision(self, *, actor: str, actor_ctx: torch.Tensor, decoder_hidden: torch.Tensor,
                       global_h: torch.Tensor, intent: torch.Tensor, previous_emb: torch.Tensor,
                       previous_label: str, ledger: ShadowLedger, remaining_units: float,
                       index: int, teacher_action: dict[str, Any] | None, rng,
                       deterministic: bool, condition_on_teacher: bool = True,
                       conditioning_ledger: ShadowLedger | None = None):
        conditioning_ledger = ledger if conditioning_ledger is None else conditioning_ledger
        legal = ledger.legal_unit_mask(actor, {})
        conditioning_legal = (
            legal if conditioning_ledger is ledger
            else conditioning_ledger.legal_unit_mask(actor, {})
        )
        decoder_hidden = self._decode_input(
            actor_ctx, previous_emb, decoder_hidden, global_h, intent, conditioning_ledger,
            remaining_units, 1.0,
        )
        op_logits = self.unit_op_head(decoder_hidden)
        item_logits = self.item_head(decoder_hidden)
        op_mask = torch.tensor(
            [bool(legal.ops.get(op, False)) for op in UNIT_OPS],
            dtype=torch.bool, device=op_logits.device,
        )
        if teacher_action is not None:
            chosen = dict(teacher_action)
            op = str(chosen.get("op", "PASS"))
        else:
            op = self.unit_id_to_op[self._choose(op_logits, op_mask, rng, deterministic)]
            chosen = {"op": op, "item": None, "quantity": None, "raw": [op]}
        item_mask = None
        if op in UNIT_ITEM_OPS:
            item_mask = self._item_mask(legal.items.get(op, {}), item_logits)
            if teacher_action is None:
                item_id = self._choose(item_logits, item_mask, rng, deterministic)
                chosen["item"] = next(name for name, idx in ITEM_TO_ID.items() if idx == item_id)
        quantity_logits = None
        quantity_tokens = None
        quantity_max_value = None
        if op in UNIT_QUANTITY_OPS:
            quantity_max_value = self._quantity_max_from_legal(
                legal, "unit", op, chosen.get("item"),
            )
            if teacher_action is not None:
                teacher_quantity = 1 if chosen.get("quantity") is None else int(chosen.get("quantity"))
                if (
                    condition_on_teacher
                    and quantity_max_value is not None
                    and teacher_quantity > quantity_max_value
                ):
                    raise ValueError(
                        f"teacher {op} quantity exceeds legal bound: "
                        f"{teacher_quantity}>{quantity_max_value}"
                    )
                quantity_logits, quantity_tokens = self._teacher_quantity(decoder_hidden, chosen.get("quantity"))
            else:
                quantity_logits, quantity_tokens, quantity = self._sample_quantity(
                    decoder_hidden, positive=False, rng=rng, deterministic=deterministic,
                    max_value=quantity_max_value,
                )
                chosen["quantity"] = quantity
                chosen["raw"] = [op, chosen.get("item")] + ([] if quantity is None else [quantity])
        elif op == "PLANT" and teacher_action is None:
            chosen["raw"] = [op, chosen.get("item")]
        semantic = self._semantic_label(chosen, "unit")
        conditioning_action = chosen
        if teacher_action is not None and not condition_on_teacher:
            conditioning_op_mask = torch.tensor(
                [bool(conditioning_legal.ops.get(candidate, False)) for candidate in UNIT_OPS],
                dtype=torch.bool, device=op_logits.device,
            )
            conditioning_action = self._unit_conditioning_action(
                decoder_hidden=decoder_hidden, op_logits=op_logits,
                item_logits=item_logits, op_mask=conditioning_op_mask,
                legal=conditioning_legal, rng=rng, deterministic=deterministic,
            )
        conditioning_semantic = self._semantic_label(conditioning_action, "unit")
        trace = self._snapshot_trace(
            conditioning_ledger, "unit", index, previous_label, semantic,
            teacher_action is not None,
        )
        if not chosen.get("_mask_fields"):
            ledger.apply_unit(actor, chosen)
        if (
            conditioning_ledger is not ledger
            and not conditioning_action.get("_mask_fields")
        ):
            conditioning_ledger.apply_unit(actor, conditioning_action)
        previous_emb = self._semantic_embedding(
            conditioning_action, "unit", decoder_hidden,
        )
        decision = DecisionOutput(
            op_logits, item_logits, quantity_logits, quantity_tokens,
            chosen, op_mask, item_mask,
            quantity_max_value=quantity_max_value,
        )
        return decision, decoder_hidden, previous_emb, conditioning_semantic, trace

    def _market_decision(self, *, slot: int, actor_ctx: torch.Tensor, decoder_hidden: torch.Tensor,
                         global_h: torch.Tensor, intent: torch.Tensor, previous_emb: torch.Tensor,
                         previous_label: str, ledger: ShadowLedger,
                         teacher_action: dict[str, Any] | None, rng, deterministic: bool,
                         condition_on_teacher: bool = True,
                         strategy_context: torch.Tensor | None = None,
                         conditioning_ledger: ShadowLedger | None = None):
        del strategy_context
        conditioning_ledger = ledger if conditioning_ledger is None else conditioning_ledger
        legal = ledger.legal_market_mask(slot, {})
        conditioning_legal = (
            legal if conditioning_ledger is ledger
            else conditioning_ledger.legal_market_mask(slot, {})
        )
        decoder_hidden = self._decode_input(
            actor_ctx, previous_emb, decoder_hidden, global_h, intent, conditioning_ledger,
            0.0, float(max(0, 9 - slot)) / 10.0,
        )
        op_logits = self.market_op_head(decoder_hidden)
        item_logits = self.item_head(decoder_hidden)
        op_mask = torch.tensor(
            [bool(legal.ops.get(op, False)) for op in MARKET_OPS],
            dtype=torch.bool, device=op_logits.device,
        )
        if teacher_action is not None:
            chosen = dict(teacher_action)
            kind = str(chosen.get("kind", "ORDER"))
            op = kind if kind in {"STOP_QUEUE", "NOP_SLOT"} else str(chosen.get("op", "NOP_SLOT"))
        else:
            op = self.market_id_to_op[self._choose(op_logits, op_mask, rng, deterministic)]
            if op in {"STOP_QUEUE", "NOP_SLOT"}:
                chosen = {"kind": op, "op": None, "item": None, "quantity": None, "raw": []}
            else:
                chosen = {"kind": "ORDER", "op": op, "item": None, "quantity": None, "raw": [op]}
        item_mask = None
        if op in MARKET_ITEM_OPS:
            item_mask = self._item_mask(legal.items.get(op, {}), item_logits)
            if teacher_action is None:
                item_id = self._choose(item_logits, item_mask, rng, deterministic)
                chosen["item"] = next(name for name, idx in ITEM_TO_ID.items() if idx == item_id)
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
                quantity_logits, quantity_tokens, quantity = self._sample_quantity(
                    decoder_hidden,
                    positive=True,
                    rng=rng,
                    deterministic=deterministic,
                    max_value=quantity_max_value,
                )
                chosen["quantity"] = quantity
                chosen["raw"] = [op, chosen.get("item"), quantity]
        semantic = self._semantic_label(chosen, "market")
        conditioning_action = chosen
        if teacher_action is not None and not condition_on_teacher:
            conditioning_op_mask = torch.tensor(
                [bool(conditioning_legal.ops.get(candidate, False)) for candidate in MARKET_OPS],
                dtype=torch.bool, device=op_logits.device,
            )
            conditioning_action = self._market_conditioning_action(
                decoder_hidden=decoder_hidden, op_logits=op_logits,
                item_logits=item_logits, op_mask=conditioning_op_mask,
                legal=conditioning_legal, rng=rng, deterministic=deterministic,
            )
        conditioning_semantic = self._semantic_label(
            conditioning_action, "market",
        )
        trace = self._snapshot_trace(
            conditioning_ledger, "market", slot, previous_label, semantic,
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
            op_logits, item_logits, quantity_logits, quantity_tokens,
            chosen, op_mask, item_mask,
            quantity_max_value=quantity_max_value,
        )
        return decision, decoder_hidden, previous_emb, conditioning_semantic, trace

    @staticmethod
    def _teacher_atomic_plant_blocked(
        teacher_action: dict[str, Any] | None,
        ledger: ShadowLedger,
    ) -> set[str]:
        if teacher_action is None:
            return set()
        demand: dict[str, int] = {}
        commands = [
            teacher_action.get("farmer") or {},
            *list(teacher_action.get("hands") or []),
        ]
        for command in commands:
            if str(command.get("op", "PASS")) != "PLANT":
                continue
            crop = command.get("item")
            if crop is None:
                continue
            name = str(crop)
            demand[name] = demand.get(name, 0) + 1
        return {
            crop for crop, count in demand.items()
            if count > int(ledger.seeds.get(crop, 0) or 0)
        }

    def _decode_row(self, encoded, batch: V2Batch, row_index: int,
                    global_h: torch.Tensor, intent: torch.Tensor,
                    teacher_action: dict[str, Any] | None, rng, deterministic: bool,
                    teacher_mix_probability: float = 1.0,
                    conditioning_rng=None,
                    strategy_context: torch.Tensor | None = None) -> RowPolicyOutput:
        ledger = ShadowLedger.from_state(batch.structured_states[row_index])
        conditioning_ledger = (
            ledger
            if teacher_action is None or float(teacher_mix_probability) >= 1.0
            else deepcopy(ledger)
        )
        if teacher_action is not None:
            ledger.set_atomic_plant_blocked(
                self._teacher_atomic_plant_blocked(
                    teacher_action, ledger,
                )
            )
        decoder_hidden = self.decoder_init(torch.cat([global_h, intent], dim=-1))
        previous_emb = self.start_action.to(global_h).clone()
        previous_label = "START"
        trace: list[DecodeTrace] = []
        structured_state = batch.structured_states[row_index]
        if isinstance(structured_state, dict):
            own_units = structured_state.get("own_units")
            if isinstance(own_units, (list, tuple)):
                own_count = len(own_units)
            else:
                own_state = structured_state.get("own") or {}
                own_count = 1 + len(own_state.get("hands") or [])
        else:
            own_units = getattr(structured_state, "own_units", None)
            if isinstance(own_units, (list, tuple)):
                own_count = len(own_units)
            else:
                own_state = getattr(structured_state, "own", None)
                hands = getattr(own_state, "hands", None) if own_state is not None else None
                own_count = 1 + len(hands or [])
        if own_count < 1:
            raise ValueError("own unit set must include the main farmer")
        teacher_hands = list((teacher_action or {}).get("hands") or [])
        if teacher_action is not None and len(teacher_hands) != own_count - 1:
            raise ValueError("teacher hand count does not match current state")

        farmer_teacher = (teacher_action or {}).get("farmer") if teacher_action is not None else None
        farmer_condition_teacher = self._teacher_conditioning_choice(
            farmer_teacher, teacher_mix_probability, conditioning_rng,
        )
        farmer, decoder_hidden, previous_emb, previous_label, item_trace = self._unit_decision(
            actor="farmer", actor_ctx=encoded.own_unit_ctx[row_index, 0],
            decoder_hidden=decoder_hidden, global_h=global_h, intent=intent,
            previous_emb=previous_emb, previous_label=previous_label, ledger=ledger,
            remaining_units=float(max(0, own_count - 1)) / float(own_count), index=0,
            teacher_action=farmer_teacher, rng=rng, deterministic=deterministic,
            condition_on_teacher=farmer_condition_teacher,
            conditioning_ledger=conditioning_ledger,
        )
        trace.append(item_trace)
        hands: list[DecisionOutput] = []
        for hand_index in range(own_count - 1):
            teacher = teacher_hands[hand_index] if teacher_action is not None else None
            hand_condition_teacher = self._teacher_conditioning_choice(
                teacher, teacher_mix_probability, conditioning_rng,
            )
            decision, decoder_hidden, previous_emb, previous_label, item_trace = self._unit_decision(
                actor=f"hand:{hand_index}", actor_ctx=encoded.own_unit_ctx[row_index, hand_index + 1],
                decoder_hidden=decoder_hidden, global_h=global_h, intent=intent,
                previous_emb=previous_emb, previous_label=previous_label, ledger=ledger,
                remaining_units=float(max(0, own_count - hand_index - 2)) / float(own_count),
                index=hand_index + 1, teacher_action=teacher, rng=rng,
                deterministic=deterministic,
                condition_on_teacher=hand_condition_teacher,
                conditioning_ledger=conditioning_ledger,
            )
            hands.append(decision); trace.append(item_trace)

        market_teacher = list((teacher_action or {}).get("market") or []) if teacher_action is not None else None
        if teacher_action is not None and not market_teacher:
            market_teacher = [{"kind": "STOP_QUEUE", "op": None, "item": None,
                               "quantity": None, "raw": []}]
        market: list[DecisionOutput] = []
        limit = min(10, len(market_teacher)) if market_teacher is not None else 10
        for slot in range(limit):
            teacher = market_teacher[slot] if market_teacher is not None else None
            market_condition_teacher = self._teacher_conditioning_choice(
                teacher, teacher_mix_probability, conditioning_rng,
            )
            slot_ctx = self.market_slot_embedding(
                global_h.new_tensor(slot, dtype=torch.long)
            )
            decision, decoder_hidden, previous_emb, previous_label, item_trace = self._market_decision(
                slot=slot, actor_ctx=slot_ctx, decoder_hidden=decoder_hidden,
                global_h=global_h, intent=intent, previous_emb=previous_emb,
                previous_label=previous_label, ledger=ledger,
                teacher_action=teacher, rng=rng, deterministic=deterministic,
                condition_on_teacher=market_condition_teacher,
                strategy_context=strategy_context,
                conditioning_ledger=conditioning_ledger,
            )
            market.append(decision); trace.append(item_trace)
            if decision.chosen_action.get("kind") == "STOP_QUEUE":
                break
        if not market:
            raise RuntimeError("market decoder emitted no slot")
        return RowPolicyOutput(
            farmer=farmer, hands=tuple(hands), market=tuple(market), trace=tuple(trace)
        )

    def _auxiliary(self, encoded, h: torch.Tensor, intent: torch.Tensor) -> AuxiliaryOutputs:
        joint = torch.cat([h, intent], dim=-1)
        return AuxiliaryOutputs(
            effect=self.effect_head(joint),
            future_resource=self.future_resource_head(joint),
            unit_task=self.unit_task_head(encoded.own_unit_ctx),
            opponent_effect=self.opponent_effect_head(joint),
            terminal_money=self.terminal_money_head(joint).squeeze(-1),
            terminal_margin=self.terminal_margin_head(joint).squeeze(-1),
        )

    def _run(
        self, batch: V2Batch, teacher_actions, state, rng, deterministic: bool,
        teacher_mix_probability: float = 1.0, conditioning_rng=None,
    ) -> PolicyOutput:
        encoded = self.encoder(batch)
        h, c, intent = self.core.step(encoded.fused, state)
        if teacher_actions is not None and len(teacher_actions) != h.shape[0]:
            raise ValueError("teacher action batch mismatch")
        rows = []
        for row_index in range(h.shape[0]):
            teacher = teacher_actions[row_index] if teacher_actions is not None else None
            rows.append(self._decode_row(
                encoded, batch, row_index, h[row_index], intent[row_index],
                teacher, rng, deterministic,
                teacher_mix_probability=teacher_mix_probability,
                conditioning_rng=conditioning_rng,
            ))
        return PolicyOutput(
            rows=tuple(rows), recurrent_state=(h, c), intent=intent,
            aux=self._auxiliary(encoded, h, intent),
        )

    def forward_sequence(
        self,
        batch: V2Batch,
        teacher_actions=None,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
        teacher_mix_probability: float = 1.0,
        conditioning_rng=None,
    ) -> PolicyOutput:
        return self._run(
            batch, teacher_actions=teacher_actions, state=state,
            rng=None, deterministic=True,
            teacher_mix_probability=teacher_mix_probability,
            conditioning_rng=conditioning_rng,
        )

    def sample_step(
        self,
        state_tensors: V2Batch,
        recurrent_state: tuple[torch.Tensor, torch.Tensor] | None,
        rng,
        deterministic: bool = False,
    ) -> PolicyOutput:
        return self._run(
            state_tensors, teacher_actions=None, state=recurrent_state,
            rng=rng, deterministic=bool(deterministic),
        )
