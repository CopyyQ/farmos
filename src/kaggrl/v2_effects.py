from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from kaggrl.constants import ANIMALS, CROPS, PRODUCTS

MOVE_DELTAS = {
    "NORTH": (0, -1),
    "SOUTH": (0, 1),
    "EAST": (1, 0),
    "WEST": (-1, 0),
}


@dataclass(frozen=True)
class ActionEvidence:
    actor: str
    op: str
    status: str
    observed: dict[str, Any]


@dataclass
class TransitionEffects:
    money_delta: int
    hand_count_delta: int
    shed_delta: dict[str, int]
    seed_delta: dict[str, int]
    market_inventory_delta: dict[str, int]
    market_price_delta: dict[str, int]
    unit_position_delta: dict[str, list[int]]
    day_changed: bool
    day_reset: bool
    action_evidence: tuple[ActionEvidence, ...]
    opponent_public: dict[str, Any]


def _farm_view(obs: dict[str, Any], *, rival: bool = False) -> dict[str, Any]:
    player = int(obs.get("player", 0))
    index = 1 - player if rival else player
    farms = obs.get("farms") or []
    if not isinstance(farms, list) or index < 0 or index >= len(farms):
        raise ValueError(f"invalid farms/player: player={player} rival={rival}")
    farm = farms[index]
    if not isinstance(farm, dict):
        raise ValueError(f"farm[{index}] is not a mapping")
    return farm


def _private_view(obs: dict[str, Any]) -> dict[str, Any]:
    value = obs.get("private") or {}
    return value if isinstance(value, dict) else {}


def _market_view(obs: dict[str, Any]) -> dict[str, Any]:
    value = obs.get("market") or {}
    return value if isinstance(value, dict) else {}


def _dict_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, int]:
    keys = set(before) | set(after)
    out: dict[str, int] = {}
    for key in sorted(keys):
        delta = int(after.get(key, 0) or 0) - int(before.get(key, 0) or 0)
        if delta:
            out[str(key)] = delta
    return out


def _position_delta(before, after) -> list[int] | None:
    if not (isinstance(before, (list, tuple)) and isinstance(after, (list, tuple))):
        return None
    if len(before) < 2 or len(after) < 2:
        return None
    return [int(after[0]) - int(before[0]), int(after[1]) - int(before[1])]


def _inventory(obs: dict[str, Any], actor: str) -> dict[str, int]:
    private = obs.get("private") or {}
    inventories = private.get("inventories") or []
    if actor == "farmer":
        index = 0
    elif actor.startswith("hand:"):
        index = int(actor.split(":", 1)[1]) + 1
    else:
        return {}
    if 0 <= index < len(inventories) and isinstance(inventories[index], dict):
        return inventories[index]
    return {}


def _actor_position(obs: dict[str, Any], actor: str):
    player = int(obs.get("player", 0))
    farm = (obs.get("farms") or [])[player]
    if actor == "farmer":
        return farm.get("farmer")
    if actor.startswith("hand:"):
        index = int(actor.split(":", 1)[1])
        hands = farm.get("hands") or []
        if index < len(hands):
            return hands[index]
    return None


def _tile_at(obs: dict[str, Any], position):
    if not (isinstance(position, (list, tuple)) and len(position) >= 2):
        return None
    player = int(obs.get("player", 0))
    farm = (obs.get("farms") or [])[player]
    tiles = farm.get("tiles") or []
    x, y = int(position[0]), int(position[1])
    if 0 <= y < len(tiles) and 0 <= x < len(tiles[y]):
        return tiles[y][x]
    return None


def _unit_evidence(before, after, actor: str, command, day_changed: bool) -> ActionEvidence | None:
    if not isinstance(command, list) or not command:
        return None
    op = str(command[0])
    if op == "PASS":
        return None
    before_pos = _actor_position(before, actor)
    after_pos = _actor_position(after, actor)
    if op in MOVE_DELTAS:
        delta = _position_delta(before_pos, after_pos)
        expected = list(MOVE_DELTAS[op])
        status = "confirmed" if delta == expected else ("unconfirmed" if after_pos is None else "failed")
        return ActionEvidence(actor, op, status, {"position_delta": delta})
    before_inv = _inventory(before, actor)
    after_inv = _inventory(after, actor)
    inv_delta = _dict_delta(before_inv, after_inv)
    if op == "HARVEST":
        gained = sum(v for v in inv_delta.values() if v > 0)
        status = "unconfirmed" if day_changed else ("confirmed" if gained > 0 else "failed")
        return ActionEvidence(actor, op, status, {"inventory_delta": inv_delta})
    if op == "PLACE":
        item = command[1] if len(command) >= 2 else None
        after_tile = _tile_at(after, before_pos)
        animal_placed = isinstance(after_tile, dict) and after_tile.get("animal") == item
        item_removed = int(after_inv.get(item, 0) or 0) < int(before_inv.get(item, 0) or 0) if item else False
        if animal_placed:
            status = "confirmed"
        elif day_changed:
            status = "unconfirmed"
        else:
            status = "confirmed" if item_removed else "failed"
        return ActionEvidence(actor, op, status, {
            "item": item,
            "animal_placed": animal_placed,
            "inventory_delta": inv_delta,
        })
    if op == "PICKUP":
        gained = sum(v for v in inv_delta.values() if v > 0)
        status = "unconfirmed" if day_changed else ("confirmed" if gained > 0 else "failed")
        return ActionEvidence(actor, op, status, {"inventory_delta": inv_delta})
    if op == "DROP":
        lost = -sum(v for v in inv_delta.values() if v < 0)
        status = "unconfirmed" if day_changed else ("confirmed" if lost > 0 else "failed")
        return ActionEvidence(actor, op, status, {"inventory_delta": inv_delta})
    return ActionEvidence(actor, op, "unconfirmed", {})


def _unit_shed_confound(action, item: str | None, *, direction: str) -> bool:
    if not isinstance(action, dict):
        return False
    commands = [action.get("farmer"), *(action.get("hands", []) if isinstance(action.get("hands", []), list) else [])]
    for command in commands:
        if not isinstance(command, list) or not command:
            continue
        op = str(command[0])
        cmd_item = str(command[1]) if len(command) > 1 else None
        if direction in {"increase", "any"} and (op == "DROP" or (op == "PLACE" and cmd_item == item)):
            return True
        if direction in {"decrease", "any"} and op == "PICKUP" and cmd_item == item:
            return True
    return False


def _market_request_valid(order) -> bool:
    if not isinstance(order, list) or not order:
        return False
    op = str(order[0])
    if op in {"HIRE", "BUY_LAND"}:
        return True
    if op not in {"BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"} or len(order) < 3:
        return False
    try:
        quantity = int(order[2])
    except (TypeError, ValueError):
        return False
    if quantity <= 0:
        return False
    item = str(order[1]) if order[1] is not None else None
    if op == "BUY_SEED":
        return item in CROPS
    if op == "BUY_PRODUCT":
        return item in {"WHEAT", "FERTILIZER"}
    if op == "BUY_ANIMAL":
        return item in ANIMALS
    return item in PRODUCTS


def _unit_plant_confound(action, crop: str | None) -> bool:
    if not isinstance(action, dict):
        return False
    commands = [action.get("farmer"), *(action.get("hands", []) if isinstance(action.get("hands", []), list) else [])]
    return any(
        isinstance(command, list) and len(command) >= 2
        and str(command[0]) == "PLANT" and str(command[1]) == crop
        for command in commands
    )


def _requested_quantity_total(market, op: str, item: str | None) -> int:
    total = 0
    for order in market:
        if not _market_request_valid(order):
            continue
        if str(order[0]) != op:
            continue
        order_item = str(order[1]) if len(order) > 1 else None
        if order_item != item:
            continue
        if len(order) >= 3:
            total += int(order[2])
    return total


def _market_evidence(before, after, action, shed_delta, seed_delta, money_delta, day_changed):
    market = action.get("market", []) if isinstance(action, dict) else []
    if not isinstance(market, list):
        return []
    counts: dict[tuple[str, str | None], int] = {}
    for order in market:
        if isinstance(order, list) and order:
            key = (str(order[0]), str(order[1]) if len(order) > 1 else None)
            counts[key] = counts.get(key, 0) + 1
    before_farm = _farm_view(before)
    after_farm = _farm_view(after)
    hand_growth = after_farm.get("hands", []) or []
    hand_before = before_farm.get("hands", []) or []
    hires_confirmable = max(0, len(hand_growth) - len(hand_before))
    land_growth = max(0, len(after_farm.get("unlocked_quadrants", []) or []) - len(before_farm.get("unlocked_quadrants", []) or []))
    total_hires = counts.get(("HIRE", None), 0)
    total_lands = counts.get(("BUY_LAND", None), 0)
    evidence = []
    for index, order in enumerate(market):
        if not isinstance(order, list) or not order:
            continue
        op = str(order[0])
        item = str(order[1]) if len(order) > 1 else None
        actor = f"market:{index}"
        if not _market_request_valid(order):
            evidence.append(ActionEvidence(actor, op, "failed", {
                "item": item, "invalid_request": True,
            }))
            continue
        if op == "HIRE":
            if day_changed:
                status = "unconfirmed"
            elif hires_confirmable == total_hires:
                status = "confirmed"
            elif hires_confirmable == 0:
                status = "failed"
            else:
                status = "unconfirmed"
            evidence.append(ActionEvidence(actor, op, status, {
                "hand_count_delta": len(hand_growth) - len(hand_before),
                "requested_slots": total_hires,
            }))
        elif op == "BUY_LAND":
            if land_growth == total_lands:
                status = "confirmed"
            elif land_growth == 0:
                status = "failed"
            else:
                status = "unconfirmed"
            evidence.append(ActionEvidence(actor, op, status, {
                "land_count_delta": land_growth, "requested_slots": total_lands,
            }))
        elif op == "BUY_SEED":
            delta = int(seed_delta.get(item, 0) or 0)
            requested = _requested_quantity_total(market, op, item)
            confounded = _unit_plant_confound(action, item)
            if confounded:
                status = "unconfirmed"
            elif delta == requested:
                status = "confirmed"
            elif delta == 0:
                status = "failed"
            else:
                status = "unconfirmed"
            evidence.append(ActionEvidence(actor, op, status, {
                "item": item, "seed_delta": delta, "requested_total": requested,
                "confounded": confounded,
            }))
        elif op in ("BUY_PRODUCT", "BUY_ANIMAL"):
            delta = int(shed_delta.get(item, 0) or 0)
            requested = _requested_quantity_total(market, op, item)
            opposite = op == "BUY_PRODUCT" and _requested_quantity_total(market, "SELL", item) > 0
            confounded = day_changed or opposite or _unit_shed_confound(action, item, direction="any")
            if confounded:
                status = "unconfirmed"
            elif delta == requested:
                status = "confirmed"
            elif delta <= 0:
                status = "failed"
            else:
                status = "unconfirmed"
            evidence.append(ActionEvidence(actor, op, status, {
                "item": item, "shed_delta": delta, "requested_total": requested,
                "confounded": confounded,
            }))
        elif op == "SELL":
            delta = int(shed_delta.get(item, 0) or 0)
            requested = _requested_quantity_total(market, op, item)
            opposite = _requested_quantity_total(market, "BUY_PRODUCT", item) > 0
            confounded = day_changed or opposite or _unit_shed_confound(action, item, direction="any")
            sold = max(0, -delta)
            if confounded:
                status = "unconfirmed"
            elif sold == requested:
                status = "confirmed"
            elif sold == 0:
                status = "failed"
            else:
                status = "unconfirmed"
            evidence.append(ActionEvidence(actor, op, status, {
                "item": item, "shed_delta": delta, "money_delta": money_delta,
                "requested_total": requested, "confounded": confounded,
            }))
        else:
            evidence.append(ActionEvidence(actor, op, "failed", {"invalid_request": True}))
    return evidence


def _unit_position_deltas(before, after):
    result: dict[str, list[int]] = {}
    actors = ["farmer"]
    player = int(before.get("player", 0))
    hands = ((before.get("farms") or [])[player].get("hands") or [])
    actors.extend(f"hand:{i}" for i in range(len(hands)))
    for actor in actors:
        delta = _position_delta(_actor_position(before, actor), _actor_position(after, actor))
        if delta is not None:
            result[actor] = delta
    return result


def derive_effects(obs_t: dict[str, Any], action_t: dict[str, Any], obs_t1: dict[str, Any]) -> TransitionEffects:
    before_farm = _farm_view(obs_t)
    after_farm = _farm_view(obs_t1)
    before_private = _private_view(obs_t)
    after_private = _private_view(obs_t1)
    before_market = _market_view(obs_t)
    after_market = _market_view(obs_t1)
    money_delta = int(after_farm.get("money", 0) or 0) - int(before_farm.get("money", 0) or 0)
    hand_count_delta = len(after_farm.get("hands", []) or []) - len(before_farm.get("hands", []) or [])
    shed_delta = _dict_delta(before_private.get("shed", {}) or {}, after_private.get("shed", {}) or {})
    seed_delta = _dict_delta(before_private.get("seeds", {}) or {}, after_private.get("seeds", {}) or {})
    market_inventory_delta = _dict_delta(before_market.get("inventory", {}) or {}, after_market.get("inventory", {}) or {})
    market_price_delta = _dict_delta(before_market.get("prices", {}) or {}, after_market.get("prices", {}) or {})
    day_changed = int(obs_t1.get("day", 0)) != int(obs_t.get("day", 0))
    evidence: list[ActionEvidence] = []
    farmer_cmd = action_t.get("farmer") if isinstance(action_t, dict) else None
    row = _unit_evidence(obs_t, obs_t1, "farmer", farmer_cmd, day_changed)
    if row is not None:
        evidence.append(row)
    hands = action_t.get("hands", []) if isinstance(action_t, dict) else []
    if isinstance(hands, list):
        for index, command in enumerate(hands):
            row = _unit_evidence(obs_t, obs_t1, f"hand:{index}", command, day_changed)
            if row is not None:
                evidence.append(row)
    evidence.extend(_market_evidence(
        obs_t, obs_t1, action_t if isinstance(action_t, dict) else {},
        shed_delta, seed_delta, money_delta, day_changed,
    ))
    before_rival = _farm_view(obs_t, rival=True)
    after_rival = _farm_view(obs_t1, rival=True)
    rival_money_delta = int(after_rival.get("money", 0) or 0) - int(before_rival.get("money", 0) or 0)
    rival_farmer_delta = _position_delta(before_rival.get("farmer"), after_rival.get("farmer"))
    opponent_public = {
        "confidence": "inferred",
        "money_delta": rival_money_delta,
        "farmer_position_delta": rival_farmer_delta,
        "hand_count_delta": len(after_rival.get("hands", []) or []) - len(before_rival.get("hands", []) or []),
        "grid_changed": (before_rival.get("tiles") or []) != (after_rival.get("tiles") or []),
    }
    return TransitionEffects(
        money_delta=money_delta,
        hand_count_delta=hand_count_delta,
        shed_delta=shed_delta,
        seed_delta=seed_delta,
        market_inventory_delta=market_inventory_delta,
        market_price_delta=market_price_delta,
        unit_position_delta=_unit_position_deltas(obs_t, obs_t1),
        day_changed=day_changed,
        day_reset=day_changed and int(obs_t1.get("hour", 0)) == 0,
        action_evidence=tuple(evidence),
        opponent_public=deepcopy(opponent_public),
    )
