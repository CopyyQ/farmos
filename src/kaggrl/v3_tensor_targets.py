from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import torch

from .constants import ITEM_TO_ID, UNIT_OPS
from .v2_ledger import MARKET_OPS
from .v2_quantity import END_ID, OMIT_ID, DIGIT_OFFSET

UNIT_OP_TO_ID = {op: i for i, op in enumerate(UNIT_OPS)}
MARKET_OP_TO_ID = {op: i for i, op in enumerate(MARKET_OPS)}
ITEM_CLASSES = max(ITEM_TO_ID.values()) + 1
MAX_MARKET_SLOTS = 10
MAX_QUANTITY_TOKENS = 6


def _market_op(action: dict[str, Any]) -> str:
    kind = str(action.get("kind", "ORDER"))
    if kind in {"STOP_QUEUE", "NOP_SLOT"}:
        return kind
    return str(action.get("op", "NOP_SLOT"))


def _quantity_tokens(value: int | None) -> tuple[int, ...]:
    if value is None:
        return (OMIT_ID,)
    number = int(value)
    if number < 0:
        raise ValueError("quantity must be non-negative")
    digits = tuple(DIGIT_OFFSET + int(ch) for ch in str(number))
    tokens = digits + (END_ID,)
    if len(tokens) > MAX_QUANTITY_TOKENS:
        raise ValueError("quantity exceeds tensor target width")
    return tokens


@dataclass
class TensorActionTargets:
    unit_op: torch.Tensor
    unit_item: torch.Tensor
    unit_quantity: torch.Tensor
    unit_mask: torch.Tensor
    unit_quantity_tokens: torch.Tensor
    unit_quantity_token_mask: torch.Tensor
    market_op: torch.Tensor
    market_item: torch.Tensor
    market_quantity: torch.Tensor
    market_mask: torch.Tensor
    market_executed: torch.Tensor
    market_quantity_tokens: torch.Tensor
    market_quantity_token_mask: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.unit_op.shape[0])

    @property
    def max_units(self) -> int:
        return int(self.unit_op.shape[1])

    def to(self, device: torch.device | str) -> "TensorActionTargets":
        device = torch.device(device)
        values = {}
        for field in fields(self):
            value = getattr(self, field.name)
            values[field.name] = value.to(device)
        return TensorActionTargets(**values)

    def index_select(self, index: torch.Tensor) -> "TensorActionTargets":
        index = index.to(device=self.unit_op.device, dtype=torch.long)
        return TensorActionTargets(**{
            field.name: getattr(self, field.name).index_select(0, index)
            for field in fields(self)
        })

    @classmethod
    def from_actions(
        cls,
        actions,
        *,
        max_units: int | None = None,
        device: torch.device | str = "cpu",
    ) -> "TensorActionTargets":
        actions = tuple(actions)
        if not actions:
            raise ValueError("tensor action targets require a non-empty batch")
        observed_units = [
            1 + len(list((action or {}).get("hands") or []))
            for action in actions
        ]
        width = max(observed_units) if max_units is None else int(max_units)
        if width < max(observed_units):
            raise ValueError("max_units is smaller than a teacher action")
        batch = len(actions)

        unit_op = torch.full(
            (batch, width), UNIT_OP_TO_ID["PASS"], dtype=torch.long
        )
        unit_item = torch.zeros((batch, width), dtype=torch.long)
        unit_quantity = torch.full((batch, width), -1, dtype=torch.long)
        unit_mask = torch.zeros((batch, width), dtype=torch.bool)
        unit_quantity_tokens = torch.zeros(
            (batch, width, MAX_QUANTITY_TOKENS), dtype=torch.long
        )
        unit_quantity_token_mask = torch.zeros(
            (batch, width, MAX_QUANTITY_TOKENS), dtype=torch.bool
        )

        market_op = torch.full(
            (batch, MAX_MARKET_SLOTS),
            MARKET_OP_TO_ID["STOP_QUEUE"],
            dtype=torch.long,
        )
        market_item = torch.zeros(
            (batch, MAX_MARKET_SLOTS), dtype=torch.long
        )
        market_quantity = torch.full(
            (batch, MAX_MARKET_SLOTS), -1, dtype=torch.long
        )
        market_mask = torch.zeros(
            (batch, MAX_MARKET_SLOTS), dtype=torch.bool
        )
        market_executed = torch.zeros(
            (batch, MAX_MARKET_SLOTS), dtype=torch.bool
        )
        market_quantity_tokens = torch.zeros(
            (batch, MAX_MARKET_SLOTS, MAX_QUANTITY_TOKENS),
            dtype=torch.long,
        )
        market_quantity_token_mask = torch.zeros(
            (batch, MAX_MARKET_SLOTS, MAX_QUANTITY_TOKENS),
            dtype=torch.bool,
        )

        for row, action in enumerate(actions):
            action = action or {}
            units = [
                action.get("farmer") or {
                    "op": "PASS",
                    "item": None,
                    "quantity": None,
                },
                *list(action.get("hands") or []),
            ]
            for index, command in enumerate(units):
                op = str((command or {}).get("op", "PASS"))
                if op not in UNIT_OP_TO_ID:
                    raise ValueError(f"unknown unit op target: {op}")
                item = (command or {}).get("item")
                quantity = (command or {}).get("quantity")
                unit_op[row, index] = UNIT_OP_TO_ID[op]
                unit_item[row, index] = ITEM_TO_ID.get(str(item), 0) if item is not None else 0
                unit_quantity[row, index] = -1 if quantity is None else int(quantity)
                unit_mask[row, index] = True
                tokens = _quantity_tokens(quantity)
                unit_quantity_tokens[row, index, : len(tokens)] = torch.tensor(
                    tokens, dtype=torch.long
                )
                unit_quantity_token_mask[row, index, : len(tokens)] = True

            market = list(action.get("market") or [])
            if not market:
                market = [{
                    "kind": "STOP_QUEUE",
                    "op": None,
                    "item": None,
                    "quantity": None,
                }]
            if len(market) > MAX_MARKET_SLOTS:
                raise ValueError("teacher market queue exceeds 10 slots")
            for slot, command in enumerate(market):
                command = command or {}
                op = _market_op(command)
                if op not in MARKET_OP_TO_ID:
                    raise ValueError(f"unknown market op target: {op}")
                item = command.get("item")
                quantity = command.get("quantity")
                market_op[row, slot] = MARKET_OP_TO_ID[op]
                market_item[row, slot] = (
                    ITEM_TO_ID.get(str(item), 0) if item is not None else 0
                )
                market_quantity[row, slot] = (
                    -1 if quantity is None else int(quantity)
                )
                market_mask[row, slot] = True
                market_executed[row, slot] = bool(
                    command.get("_executed", False)
                )
                tokens = _quantity_tokens(quantity)
                market_quantity_tokens[row, slot, : len(tokens)] = torch.tensor(
                    tokens, dtype=torch.long
                )
                market_quantity_token_mask[row, slot, : len(tokens)] = True

        return cls(
            unit_op=unit_op,
            unit_item=unit_item,
            unit_quantity=unit_quantity,
            unit_mask=unit_mask,
            unit_quantity_tokens=unit_quantity_tokens,
            unit_quantity_token_mask=unit_quantity_token_mask,
            market_op=market_op,
            market_item=market_item,
            market_quantity=market_quantity,
            market_mask=market_mask,
            market_executed=market_executed,
            market_quantity_tokens=market_quantity_tokens,
            market_quantity_token_mask=market_quantity_token_mask,
        ).to(device)

    def atomic_plant_blocked(
        self, seeds: torch.Tensor
    ) -> torch.Tensor:
        if seeds.ndim != 2 or seeds.shape[0] != self.batch_size:
            raise ValueError("seed tensor must be [batch, items]")
        demand = torch.zeros_like(seeds)
        plant_id = UNIT_OP_TO_ID["PLANT"]
        active = self.unit_mask & self.unit_op.eq(plant_id)
        item = self.unit_item.clamp(0, seeds.shape[1] - 1)
        demand.scatter_add_(
            1,
            item,
            active.to(demand.dtype),
        )
        return demand.gt(seeds)
