from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from .constants import ANIMALS, CROPS, PRODUCTS, UNIT_OPS

MARKET_ORDER_OPS = {"BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"}
MARKET_ATOMIC_OPS = {"HIRE", "BUY_LAND"}
MAX_MARKET_ORDERS = 10


@dataclass(frozen=True)
class UnitCommand:
    op: str
    item: str | None = None
    quantity: int | None = None
    raw: tuple[Any, ...] = ()


@dataclass(frozen=True)
class MarketSlot:
    kind: str
    op: str | None = None
    item: str | None = None
    quantity: int | None = None
    raw: tuple[Any, ...] = ()

    @classmethod
    def stop(cls):
        return cls("STOP_QUEUE")


@dataclass(frozen=True)
class JointAction:
    farmer: UnitCommand
    hands: tuple[UnitCommand, ...]
    market: tuple[MarketSlot, ...]
    raw_action: dict | None = None


@dataclass(frozen=True)
class RoundTripFailure:
    step: int
    seat: int
    reason: str
    raw: Any
    emitted: Any


def _as_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _to_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_unit(command) -> UnitCommand:
    raw = tuple(_as_list(command))
    values = list(raw) if raw else ["PASS"]
    op = str(values[0]) if values else "PASS"
    item = str(values[1]) if len(values) > 1 and values[1] is not None else None
    quantity = _to_int(values[2]) if len(values) > 2 else None
    return UnitCommand(op=op, item=item, quantity=quantity, raw=raw)


def _market_item_valid(op: str, item: str | None) -> bool:
    if op == "BUY_SEED":
        return item in CROPS
    if op == "BUY_PRODUCT":
        return item in {"WHEAT", "FERTILIZER"}
    if op == "BUY_ANIMAL":
        return item in ANIMALS
    if op == "SELL":
        return item in PRODUCTS
    return False


def _parse_market_slot(order) -> MarketSlot:
    raw = tuple(_as_list(order))
    if not raw:
        return MarketSlot("NOP_SLOT", raw=raw)
    op = str(raw[0])
    if op in MARKET_ATOMIC_OPS:
        return MarketSlot("ORDER", op=op, raw=raw)
    if op not in MARKET_ORDER_OPS or len(raw) < 3:
        return MarketSlot("NOP_SLOT", op=op, raw=raw)
    item = str(raw[1]) if raw[1] is not None else None
    quantity = _to_int(raw[2])
    if quantity is None or quantity <= 0 or not _market_item_valid(op, item):
        return MarketSlot("NOP_SLOT", op=op, item=item, quantity=quantity, raw=raw)
    return MarketSlot("ORDER", op=op, item=item, quantity=quantity, raw=raw)


def parse_raw_action(action, hand_count: int) -> JointAction:
    raw_action = deepcopy(action if isinstance(action, dict) else {})
    farmer = _parse_unit(raw_action.get("farmer", ["PASS"]))
    raw_hands = _as_list(raw_action.get("hands", []))
    hands = tuple(
        _parse_unit(raw_hands[i] if i < len(raw_hands) else ["PASS"])
        for i in range(max(0, int(hand_count)))
    )
    raw_market = _as_list(raw_action.get("market", []))
    market = [_parse_market_slot(order) for order in raw_market]
    if len(raw_market) < MAX_MARKET_ORDERS:
        market.append(MarketSlot.stop())
    return JointAction(farmer, hands, tuple(market), raw_action=raw_action)


def _unit_to_engine(command: UnitCommand) -> list:
    op = command.op
    if op in {"PICKUP", "PLACE"}:
        out = [op, command.item]
        if command.quantity is not None:
            out.append(int(command.quantity))
        return out
    if op == "PLANT":
        return [op, command.item]
    return [op]


def _slot_to_engine(slot: MarketSlot):
    if slot.kind == "STOP_QUEUE":
        return None
    if slot.raw:
        return list(slot.raw)
    if slot.kind == "NOP_SLOT":
        return ["SELL", "WHEAT", 0]
    if slot.op in MARKET_ATOMIC_OPS:
        return [slot.op]
    if slot.op in MARKET_ORDER_OPS:
        return [slot.op, slot.item, int(slot.quantity)]
    return ["SELL", "WHEAT", 0]


def to_engine_action(joint: JointAction) -> dict:
    if joint.raw_action is not None:
        return deepcopy(joint.raw_action)
    market = []
    for slot in joint.market[:MAX_MARKET_ORDERS]:
        order = _slot_to_engine(slot)
        if order is None:
            break
        market.append(order)
    return {
        "farmer": _unit_to_engine(joint.farmer),
        "hands": [_unit_to_engine(x) for x in joint.hands],
        "market": market,
    }


def raw_equal(a, b) -> bool:
    return a == b


def _unit_signature(command: UnitCommand):
    if command.op in {"PICKUP", "PLACE"}:
        qty = 1 if command.quantity is None else int(command.quantity)
        return (command.op, command.item, qty)
    if command.op == "PLANT":
        return (command.op, command.item)
    return (command.op,)


def _market_signature(slot: MarketSlot):
    if slot.kind == "STOP_QUEUE":
        return ("STOP_QUEUE",)
    if slot.kind == "NOP_SLOT":
        return ("NOP_SLOT",)
    if slot.op in MARKET_ATOMIC_OPS:
        return (slot.op,)
    return (slot.op, slot.item, int(slot.quantity))


def _hand_count_from_observation(observation, a, b):
    if isinstance(observation, dict):
        try:
            player = int(observation.get("player", 0))
            return len(observation["farms"][player].get("hands", []))
        except (KeyError, IndexError, TypeError, ValueError):
            pass
    ah = len(_as_list(a.get("hands", []))) if isinstance(a, dict) else 0
    bh = len(_as_list(b.get("hands", []))) if isinstance(b, dict) else 0
    return max(ah, bh)


def semantic_equivalent(raw_a, raw_b, observation) -> bool:
    hand_count = _hand_count_from_observation(observation, raw_a, raw_b)
    a = parse_raw_action(raw_a, hand_count)
    b = parse_raw_action(raw_b, hand_count)
    return (
        _unit_signature(a.farmer) == _unit_signature(b.farmer)
        and tuple(map(_unit_signature, a.hands)) == tuple(map(_unit_signature, b.hands))
        and tuple(map(_market_signature, a.market)) == tuple(map(_market_signature, b.market))
    )


def audit_replay_actions(replay: dict, source_seat: int) -> list[RoundTripFailure]:
    failures: list[RoundTripFailure] = []
    steps = list(replay.get("steps") or [])
    seat = int(source_seat)
    for step_index in range(1, len(steps)):
        if seat >= len(steps[step_index]):
            failures.append(RoundTripFailure(step_index, seat, "missing_seat", None, None))
            continue
        record = steps[step_index][seat] or {}
        raw = record.get("action") if isinstance(record, dict) else None
        raw = raw if isinstance(raw, dict) else {}
        previous = steps[step_index - 1][seat] if seat < len(steps[step_index - 1]) else {}
        observation = previous.get("observation", {}) if isinstance(previous, dict) else {}
        hand_count = _hand_count_from_observation(observation, raw, raw)
        try:
            joint = parse_raw_action(raw, hand_count)
            emitted = to_engine_action(joint)
        except Exception as exc:
            failures.append(
                RoundTripFailure(step_index, seat, f"parse_error:{type(exc).__name__}", raw, None)
            )
            continue
        if not raw_equal(raw, emitted):
            failures.append(RoundTripFailure(step_index, seat, "raw_round_trip", raw, emitted))
            continue
        if not semantic_equivalent(raw, emitted, observation):
            failures.append(RoundTripFailure(step_index, seat, "semantic_round_trip", raw, emitted))
    return failures
