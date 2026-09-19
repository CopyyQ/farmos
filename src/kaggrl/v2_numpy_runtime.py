from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .constants import ANIMALS, CROPS, ITEM_TO_ID, PRODUCTS, UNIT_OPS
from .v2_ledger import MARKET_OPS, ShadowLedger

ITEM_NAMES = tuple(ITEM_TO_ID.keys())
ID_TO_ITEM = {int(value): key for key, value in ITEM_TO_ID.items()}
UNIT_OP_TO_ID = {op: i for i, op in enumerate(UNIT_OPS)}
MARKET_OP_TO_ID = {op: i for i, op in enumerate(MARKET_OPS)}
UNIT_ITEM_OPS = {"PICKUP", "PLACE", "PLANT"}
UNIT_QUANTITY_OPS = {"PICKUP", "PLACE"}
MARKET_ITEM_OPS = {"BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"}
MARKET_QUANTITY_OPS = set(MARKET_ITEM_OPS)
OMIT_ID, DIGIT_OFFSET, END_ID, START_ID, QUANTITY_VOCAB = 0, 1, 11, 12, 12
MAX_QUANTITY_DIGITS = 5
SHOP_NAMES = (
    "BAKERY", "PIZZA_SHOP", "BRUNCH_SPOT", "YARN_STORE",
    "ICE_CREAM_SHOP", "PET_CAFE", "SMOOTHIE_SHOP", "FARMERS_MARKET",
)
TILE_KINDS = ("EMPTY", "LOCKED", "WEED", "PLANT", "COOP", "PASTURE", "OTHER")
TILE_FEATURES = (
    "x", "y", *(f"kind:{name}" for name in TILE_KINDS),
    *(f"crop:{name}" for name in CROPS), *(f"animal:{name}" for name in ANIMALS),
    "watered_today", "fed_today", "cared_today", "fertilizer_available",
    "yield_units", "planted_day", "placed_day", "max_lifespan_step",
    "fertilized_until_day", "consecutive_unwatered", "consecutive_unfed", "pending_care_bonus",
)
TILE_INDEX = {name: i for i, name in enumerate(TILE_FEATURES)}

UNIT_FEATURES = (
    "kind:farmer", "kind:hand", "actor_index_log", "actor_index_sin", "actor_index_cos",
    "x", "y", "depot_distance", *(f"inventory:{item}" for item in ITEM_NAMES),
)
UNIT_INDEX = {name: i for i, name in enumerate(UNIT_FEATURES)}
MARKET_ACTION_NAMES = (
    "STOP_QUEUE", "NOP_SLOT", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL", "HIRE", "BUY_LAND",
)
PREV_UNIT_ACTION_FEATURES = (
    "none", *(f"op:{op}" for op in UNIT_OPS),
    "item:none", *(f"item:{item}" for item in ITEM_NAMES),
    "quantity_omitted", "quantity_log",
)
PREV_UNIT_INDEX = {name: i for i, name in enumerate(PREV_UNIT_ACTION_FEATURES)}
PREV_UNIT_EFFECT_STATUSES = ("confirmed", "failed", "unconfirmed")
PREV_UNIT_EFFECT_FEATURES = (
    "none", *(f"status:{name}" for name in PREV_UNIT_EFFECT_STATUSES),
    *(f"op:{op}" for op in UNIT_OPS), "dx", "dy", "move_l1",
    *(f"inventory_delta:{item}" for item in ITEM_NAMES),
)
PREV_UNIT_EFFECT_INDEX = {name: i for i, name in enumerate(PREV_UNIT_EFFECT_FEATURES)}
COMMODITY_NAMES = ITEM_NAMES
COMMODITY_FEATURES = (
    *(f"item:{name}" for name in COMMODITY_NAMES),
    "is_crop", "is_animal", "is_product", "shed_quantity", "seed_quantity",
    "carried_quantity", "market_inventory", "market_price",
)
COMMODITY_INDEX = {name: i for i, name in enumerate(COMMODITY_FEATURES)}
PREV_ACTION_GLOBAL_FEATURES = (
    "none", *(f"market_op_count:{op}" for op in MARKET_ACTION_NAMES),
    *(f"market_item_count:{item}" for item in ITEM_NAMES),
    "market_quantity_count", "market_quantity_log_sum",
)
PREV_GLOBAL_INDEX = {name: i for i, name in enumerate(PREV_ACTION_GLOBAL_FEATURES)}
ECONOMY_FEATURES = (
    "step_log", "day_log", "hour_sin", "hour_cos", "turns_to_day_end_log", "turns_to_game_end_log",
    "own_money", "rival_money", "own_hires_today", "rival_hires_today",
    "own_hand_count", "rival_hand_count", "own_land_count", "rival_land_count",
    "next_hire_cost", "next_land_cost", "land_exhausted",
    "shed_capacity", "shed_free_space", "market_slots_remaining",
    *(f"shed:{item}" for item in ITEM_NAMES), *(f"seed:{crop}" for crop in CROPS),
    *(f"market_inventory:{item}" for item in PRODUCTS), *(f"market_price:{item}" for item in PRODUCTS),
    *(f"shop_count:{shop}" for shop in SHOP_NAMES),
)
ECONOMY_INDEX = {name: i for i, name in enumerate(ECONOMY_FEATURES)}

EFFECT_FEATURES = (
    "money_delta", "hand_count_delta", "day_changed", "day_reset", "unit_move_l1_total",
    *(f"shed_delta:{item}" for item in ITEM_NAMES), *(f"seed_delta:{crop}" for crop in CROPS),
    *(f"market_inventory_delta:{item}" for item in PRODUCTS), *(f"market_price_delta:{item}" for item in PRODUCTS),
    "rival_money_delta", "rival_hand_count_delta",
)
EFFECT_INDEX = {name: i for i, name in enumerate(EFFECT_FEATURES)}


def _signed_log1p(value):
    x = float(value or 0.0)
    return math.copysign(math.log1p(abs(x)), x) if x else 0.0


def _mapping(value):
    if isinstance(value, dict):
        return value
    if hasattr(value, "__dict__"):
        return vars(value)
    return {}


def _field(state, name, default=None):
    if isinstance(state, dict):
        return state.get(name, default)
    return getattr(state, name, default)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _silu(x):
    return x * _sigmoid(x)


def _gelu(x):
    flat = np.asarray(x, dtype=np.float32).reshape(-1)
    erf = np.fromiter((math.erf(float(v) / math.sqrt(2.0)) for v in flat), dtype=np.float32, count=flat.size)
    return (0.5 * flat * (1.0 + erf)).reshape(np.asarray(x).shape).astype(np.float32)


def _softmax(x, axis=-1):
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def _tile_kind(tile):
    if tile is None:
        return "EMPTY"
    if isinstance(tile, str):
        return tile if tile in TILE_KINDS else "OTHER"
    if isinstance(tile, dict):
        kind = str(tile.get("kind", "OTHER"))
        return kind if kind in TILE_KINDS else "OTHER"
    return "OTHER"


def _tile_vector(tile, x, y, board_size):
    out = np.zeros(len(TILE_FEATURES), np.float32)
    denom = max(1, board_size - 1)
    out[TILE_INDEX["x"]] = float(x) / denom
    out[TILE_INDEX["y"]] = float(y) / denom
    out[TILE_INDEX[f"kind:{_tile_kind(tile)}"]] = 1.0
    data = tile if isinstance(tile, dict) else {}
    crop, animal = data.get("crop"), data.get("animal")
    if crop in CROPS:
        out[TILE_INDEX[f"crop:{crop}"]] = 1.0
    if animal in ANIMALS:
        out[TILE_INDEX[f"animal:{animal}"]] = 1.0
    for name in ("watered_today", "fed_today", "cared_today", "fertilizer_available"):
        out[TILE_INDEX[name]] = float(bool(data.get(name, False)))
    for name in ("yield_units", "planted_day", "placed_day", "max_lifespan_step",
                 "fertilized_until_day", "consecutive_unwatered", "consecutive_unfed", "pending_care_bonus"):
        out[TILE_INDEX[name]] = _signed_log1p(data.get(name, 0) or 0)
    return out


def _grid_tensor(grid):
    rows = list(grid or []) or [[None for _ in range(10)] for _ in range(10)]
    height = len(rows); width = max((len(row) for row in rows), default=0)
    if height != width or any(len(row) != width for row in rows):
        raise ValueError("farm grid must be square and non-ragged")
    out = np.zeros((height, width, len(TILE_FEATURES)), np.float32)
    for y, row in enumerate(rows):
        for x, tile in enumerate(row):
            out[y, x] = _tile_vector(tile, x, y, width)
    return out


def _unit_vector(unit, board_size, own_private):
    data = _mapping(unit)
    out = np.zeros(len(UNIT_FEATURES), np.float32)
    kind = str(data.get("kind", "hand"))
    out[UNIT_INDEX["kind:farmer"]] = float(kind == "farmer")
    out[UNIT_INDEX["kind:hand"]] = float(kind == "hand")
    actor_position = 0 if kind == "farmer" else int(data.get("index", 0)) + 1
    out[UNIT_INDEX["actor_index_log"]] = math.log1p(max(0, actor_position))
    out[UNIT_INDEX["actor_index_sin"]] = math.sin(float(actor_position))
    out[UNIT_INDEX["actor_index_cos"]] = math.cos(float(actor_position))
    position = data.get("position") or [0, 0]
    denom = max(1, board_size - 1)
    if isinstance(position, (list, tuple)) and len(position) >= 2:
        px, py = int(position[0]), int(position[1])
        out[UNIT_INDEX["x"]] = float(px) / denom
        out[UNIT_INDEX["y"]] = float(py) / denom
        half = board_size // 2
        depot_access = ((half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half))
        distance = min(abs(px - dx) + abs(py - dy) for dx, dy in depot_access)
        out[UNIT_INDEX["depot_distance"]] = float(distance) / denom
    inventory = data.get("inventory") if own_private else {}
    inventory = inventory if isinstance(inventory, dict) else {}
    for item in ITEM_NAMES:
        out[UNIT_INDEX[f"inventory:{item}"]] = _signed_log1p(inventory.get(item, 0) or 0)
    return out


def _units_tensor(units, board_size, own_private):
    values = list(units or [])
    if not values:
        return np.zeros((0, len(UNIT_FEATURES)), np.float32)
    return np.stack([_unit_vector(unit, board_size, own_private) for unit in values]).astype(np.float32)


def _commodity_tensor(state):
    private = _mapping(_field(state, "private", {})); shed = _mapping(private.get("shed") or {})
    seeds = _mapping(private.get("seeds") or {}); market = _mapping(_field(state, "market", {}))
    market_inventory = _mapping(market.get("inventory") or {}); market_prices = _mapping(market.get("prices") or {})
    carried = {name: 0 for name in COMMODITY_NAMES}
    for unit in list(_field(state, "own_units", []) or []):
        inventory = _mapping(_mapping(unit).get("inventory") or {})
        for name in COMMODITY_NAMES:
            carried[name] += int(inventory.get(name, 0) or 0)
    out = np.zeros((len(COMMODITY_NAMES), len(COMMODITY_FEATURES)), np.float32)
    for row, name in enumerate(COMMODITY_NAMES):
        out[row, COMMODITY_INDEX[f"item:{name}"]] = 1.0
        out[row, COMMODITY_INDEX["is_crop"]] = float(name in CROPS)
        out[row, COMMODITY_INDEX["is_animal"]] = float(name in ANIMALS)
        out[row, COMMODITY_INDEX["is_product"]] = float(name in PRODUCTS)
        out[row, COMMODITY_INDEX["shed_quantity"]] = _signed_log1p(shed.get(name, 0))
        out[row, COMMODITY_INDEX["seed_quantity"]] = _signed_log1p(seeds.get(name, 0))
        out[row, COMMODITY_INDEX["carried_quantity"]] = _signed_log1p(carried[name])
        out[row, COMMODITY_INDEX["market_inventory"]] = _signed_log1p(market_inventory.get(name, 0))
        out[row, COMMODITY_INDEX["market_price"]] = _signed_log1p(market_prices.get(name, 0))
    return out


def _previous_unit_action_vector(command):
    out = np.zeros(len(PREV_UNIT_ACTION_FEATURES), np.float32)
    data = _mapping(command)
    if not data:
        out[PREV_UNIT_INDEX["none"]] = 1.0
        return out
    op = str(data.get("op", "PASS"))
    if op in UNIT_OPS:
        out[PREV_UNIT_INDEX[f"op:{op}"]] = 1.0
    item = data.get("item")
    if item in ITEM_NAMES:
        out[PREV_UNIT_INDEX[f"item:{item}"]] = 1.0
    else:
        out[PREV_UNIT_INDEX["item:none"]] = 1.0
    quantity = data.get("quantity")
    if quantity is None:
        out[PREV_UNIT_INDEX["quantity_omitted"]] = 1.0
    else:
        out[PREV_UNIT_INDEX["quantity_log"]] = _signed_log1p(quantity)
    return out


def _previous_unit_actions_tensor(state, previous_action, previous_effect):
    units = list(_field(state, "own_units", []) or [])
    out = np.zeros((len(units), len(PREV_UNIT_ACTION_FEATURES)), np.float32)
    previous = _mapping(previous_action); effect = _mapping(previous_effect)
    farmer = previous.get("farmer") if previous else None
    hands = list(previous.get("hands") or []) if previous else []
    day_reset = bool(effect.get("day_reset", False))
    for slot in range(len(units)):
        if slot == 0:
            command = farmer
        elif day_reset:
            command = None
        else:
            idx = slot - 1
            command = hands[idx] if idx < len(hands) else None
        out[slot] = _previous_unit_action_vector(command)
    return out


def _previous_unit_effects_tensor(state, previous_effect):
    units = list(_field(state, "own_units", []) or [])
    out = np.zeros((len(units), len(PREV_UNIT_EFFECT_FEATURES)), np.float32)
    if len(units):
        out[:, PREV_UNIT_EFFECT_INDEX["none"]] = 1.0
    effect = _mapping(previous_effect)
    evidence_by_actor = {}
    for evidence in list(effect.get("action_evidence") or []):
        data = _mapping(evidence); actor = str(data.get("actor", ""))
        if actor: evidence_by_actor[actor] = data
    position_delta = _mapping(effect.get("unit_position_delta") or {})
    day_reset = bool(effect.get("day_reset", False))
    for slot, unit in enumerate(units):
        unit_data = _mapping(unit)
        actor = "farmer" if str(unit_data.get("kind")) == "farmer" else f"hand:{int(unit_data.get('index', slot - 1))}"
        if day_reset and actor.startswith("hand:"): continue
        evidence = evidence_by_actor.get(actor); delta = position_delta.get(actor)
        if evidence is not None or delta is not None:
            out[slot, PREV_UNIT_EFFECT_INDEX["none"]] = 0.0
        if evidence is not None:
            status = str(evidence.get("status", "unconfirmed"))
            if status in PREV_UNIT_EFFECT_STATUSES:
                out[slot, PREV_UNIT_EFFECT_INDEX[f"status:{status}"]] = 1.0
            op = str(evidence.get("op", ""))
            if op in UNIT_OPS: out[slot, PREV_UNIT_EFFECT_INDEX[f"op:{op}"]] = 1.0
            inv_delta = _mapping(_mapping(evidence.get("observed") or {}).get("inventory_delta") or {})
            for item in ITEM_NAMES:
                out[slot, PREV_UNIT_EFFECT_INDEX[f"inventory_delta:{item}"]] = _signed_log1p(inv_delta.get(item, 0))
        if isinstance(delta, (list, tuple)) and len(delta) >= 2:
            dx, dy = float(delta[0]), float(delta[1])
            out[slot, PREV_UNIT_EFFECT_INDEX["dx"]] = dx; out[slot, PREV_UNIT_EFFECT_INDEX["dy"]] = dy
            out[slot, PREV_UNIT_EFFECT_INDEX["move_l1"]] = _signed_log1p(abs(dx) + abs(dy))
    return out


def _previous_action_global_tensor(previous_action):
    out = np.zeros(len(PREV_ACTION_GLOBAL_FEATURES), np.float32)
    previous = _mapping(previous_action)
    if not previous:
        out[PREV_GLOBAL_INDEX["none"]] = 1.0
        return out
    op_counts = {name: 0 for name in MARKET_ACTION_NAMES}
    item_counts = {name: 0 for name in ITEM_NAMES}
    quantity_count = 0; quantity_log_sum = 0.0
    for slot in list(previous.get("market") or []):
        data = _mapping(slot); kind = str(data.get("kind", "ORDER"))
        op = kind if kind in {"STOP_QUEUE", "NOP_SLOT"} else str(data.get("op", "NOP_SLOT"))
        if op in op_counts: op_counts[op] += 1
        item = data.get("item")
        if item in item_counts: item_counts[item] += 1
        quantity = data.get("quantity")
        if quantity is not None:
            quantity_count += 1; quantity_log_sum += _signed_log1p(quantity)
    for op, count in op_counts.items():
        out[PREV_GLOBAL_INDEX[f"market_op_count:{op}"]] = _signed_log1p(count)
    for item, count in item_counts.items():
        out[PREV_GLOBAL_INDEX[f"market_item_count:{item}"]] = _signed_log1p(count)
    out[PREV_GLOBAL_INDEX["market_quantity_count"]] = _signed_log1p(quantity_count)
    out[PREV_GLOBAL_INDEX["market_quantity_log_sum"]] = quantity_log_sum
    return out


def _economy_tensor(state):
    out = np.zeros(len(ECONOMY_FEATURES), np.float32)
    step = int(_field(state, "step", 0) or 0); day = int(_field(state, "day", step // 24) or 0)
    hour = int(_field(state, "hour", step % 24) or 0)
    out[ECONOMY_INDEX["step_log"]] = math.log1p(max(0, step))
    out[ECONOMY_INDEX["day_log"]] = math.log1p(max(0, day))
    angle = 2.0 * math.pi * (hour % 24) / 24.0
    out[ECONOMY_INDEX["hour_sin"]] = math.sin(angle); out[ECONOMY_INDEX["hour_cos"]] = math.cos(angle)
    out[ECONOMY_INDEX["turns_to_day_end_log"]] = math.log1p(max(0, 23 - hour))
    out[ECONOMY_INDEX["turns_to_game_end_log"]] = math.log1p(max(0, 719 - step))
    own = _mapping(_field(state, "own", {})); rival = _mapping(_field(state, "rival", {}))
    private = _mapping(_field(state, "private", {}))
    out[ECONOMY_INDEX["own_money"]] = _signed_log1p(own.get("money", 0)); out[ECONOMY_INDEX["rival_money"]] = _signed_log1p(rival.get("money", 0))
    out[ECONOMY_INDEX["own_hires_today"]] = _signed_log1p(own.get("hires_today", 0)); out[ECONOMY_INDEX["rival_hires_today"]] = _signed_log1p(rival.get("hires_today", 0))
    out[ECONOMY_INDEX["own_hand_count"]] = _signed_log1p(len(own.get("hands") or [])); out[ECONOMY_INDEX["rival_hand_count"]] = _signed_log1p(len(rival.get("hands") or []))
    own_land_count = len(own.get("unlocked_quadrants") or [])
    out[ECONOMY_INDEX["own_land_count"]] = _signed_log1p(own_land_count); out[ECONOMY_INDEX["rival_land_count"]] = _signed_log1p(len(rival.get("unlocked_quadrants") or []))
    hires = int(own.get("hires_today", 0) or 0); a, b = 1, 1
    for _ in range(max(0, hires)): a, b = b, a + b
    out[ECONOMY_INDEX["next_hire_cost"]] = _signed_log1p(a)
    land_prices = (1000, 2000, 4000); land_index = max(0, own_land_count - 1)
    next_land = land_prices[land_index] if land_index < len(land_prices) else 0
    out[ECONOMY_INDEX["next_land_cost"]] = _signed_log1p(next_land)
    out[ECONOMY_INDEX["land_exhausted"]] = float(land_index >= len(land_prices))
    shed = _mapping(private.get("shed") or {}); seeds = _mapping(private.get("seeds") or {})
    capacity = int(private.get("shed_capacity", 100) or 100); used = sum(max(0, int(v or 0)) for v in shed.values())
    out[ECONOMY_INDEX["shed_capacity"]] = _signed_log1p(capacity)
    out[ECONOMY_INDEX["shed_free_space"]] = _signed_log1p(max(0, capacity - used))
    out[ECONOMY_INDEX["market_slots_remaining"]] = _signed_log1p(10)
    for item in ITEM_NAMES: out[ECONOMY_INDEX[f"shed:{item}"]] = _signed_log1p(shed.get(item, 0))
    for crop in CROPS: out[ECONOMY_INDEX[f"seed:{crop}"]] = _signed_log1p(seeds.get(crop, 0))
    market = _mapping(_field(state, "market", {})); inv = _mapping(market.get("inventory") or {}); prices = _mapping(market.get("prices") or {})
    for item in PRODUCTS:
        out[ECONOMY_INDEX[f"market_inventory:{item}"]] = _signed_log1p(inv.get(item, 0)); out[ECONOMY_INDEX[f"market_price:{item}"]] = _signed_log1p(prices.get(item, 0))
    shops = list(_field(state, "town_shops", []) or []) or list(_mapping(_field(state, "town", {})).get("unlocked_shops") or [])
    for shop in SHOP_NAMES: out[ECONOMY_INDEX[f"shop_count:{shop}"]] = _signed_log1p(shops.count(shop))
    return out


def _effect_tensor(effect):
    if isinstance(effect, np.ndarray):
        arr = np.asarray(effect, dtype=np.float32).reshape(-1)
        if arr.size != len(EFFECT_FEATURES):
            raise ValueError("previous effect vector width mismatch")
        return arr
    data = _mapping(effect); out = np.zeros(len(EFFECT_FEATURES), np.float32)
    out[EFFECT_INDEX["money_delta"]] = _signed_log1p(data.get("money_delta", 0))
    out[EFFECT_INDEX["hand_count_delta"]] = _signed_log1p(data.get("hand_count_delta", 0))
    out[EFFECT_INDEX["day_changed"]] = float(bool(data.get("day_changed", False)))
    out[EFFECT_INDEX["day_reset"]] = float(bool(data.get("day_reset", False)))
    move_total = 0.0
    for delta in _mapping(data.get("unit_position_delta") or {}).values():
        if isinstance(delta, (list, tuple)):
            move_total += sum(abs(float(v)) for v in delta[:2])
    out[EFFECT_INDEX["unit_move_l1_total"]] = _signed_log1p(move_total)
    for prefix, names in (("shed_delta", ITEM_NAMES), ("seed_delta", CROPS),
                          ("market_inventory_delta", PRODUCTS), ("market_price_delta", PRODUCTS)):
        values = _mapping(data.get(prefix) or {})
        for item in names:
            out[EFFECT_INDEX[f"{prefix}:{item}"]] = _signed_log1p(values.get(item, 0))
    opponent = _mapping(data.get("opponent_public") or {})
    out[EFFECT_INDEX["rival_money_delta"]] = _signed_log1p(opponent.get("money_delta", 0))
    out[EFFECT_INDEX["rival_hand_count_delta"]] = _signed_log1p(opponent.get("hand_count_delta", 0))
    return out


def _tensorize_state(state, previous_effect, previous_action):
    own_grid = _grid_tensor(_field(state, "own_grid", [])); rival_grid = _grid_tensor(_field(state, "rival_grid", []))
    if own_grid.shape[:2] != rival_grid.shape[:2]:
        raise ValueError("own/rival farm grid shapes differ")
    board_size = int(own_grid.shape[0])
    own_units = _units_tensor(_field(state, "own_units", []), board_size, True)
    rival_units = _units_tensor(_field(state, "rival_units", []), board_size, False)
    effect_mapping = previous_effect if isinstance(previous_effect, dict) else {}
    effect_value = previous_effect if isinstance(previous_effect, np.ndarray) else (previous_effect or {})
    return {
        "own_grid": own_grid, "rival_grid": rival_grid,
        "own_units": own_units, "rival_units": rival_units,
        "previous_unit_actions": _previous_unit_actions_tensor(state, previous_action or {}, effect_mapping),
        "previous_unit_effects": _previous_unit_effects_tensor(state, effect_mapping),
        "commodities": _commodity_tensor(state),
        "economy": _economy_tensor(state), "previous_action_global": _previous_action_global_tensor(previous_action or {}),
        "previous_effect": _effect_tensor(effect_value),
    }


def _feature_schema_sha256():
    payload = {
        "unit_ops": list(UNIT_OPS), "market_ops": list(MARKET_OPS),
        "item_to_id": dict(ITEM_TO_ID), "tile_features": list(TILE_FEATURES),
        "unit_features": list(UNIT_FEATURES), "prev_unit_action_features": list(PREV_UNIT_ACTION_FEATURES),
        "prev_unit_effect_features": list(PREV_UNIT_EFFECT_FEATURES),
        "commodity_features": list(COMMODITY_FEATURES),
        "economy_features": list(ECONOMY_FEATURES), "prev_action_global_features": list(PREV_ACTION_GLOBAL_FEATURES),
        "effect_features": list(EFFECT_FEATURES), "shop_names": list(SHOP_NAMES),
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _parameter_arrays_sha256(arrays):
    h = hashlib.sha256()
    for name in sorted(key for key in arrays if key.startswith("p__")):
        arr = np.asarray(arrays[name])
        h.update(name.encode("utf-8")); h.update(b"\0")
        h.update(str(arr.dtype).encode("ascii")); h.update(str(arr.shape).encode("ascii")); h.update(arr.tobytes())
    return h.hexdigest()


def _linear(x, w, b):
    return np.asarray(x, np.float32) @ np.asarray(w, np.float32).T + np.asarray(b, np.float32)


def _layer_norm(x, weight, bias, eps=1e-5):
    x = np.asarray(x, np.float32)
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    y = (x - mean) / np.sqrt(var + eps)
    return y * np.asarray(weight, np.float32) + np.asarray(bias, np.float32)


@dataclass
class RuntimeState:
    h: np.ndarray
    c: np.ndarray
    previous_action: dict[str, Any]


@dataclass
class RuntimeStepOutput:
    canonical_action: dict[str, Any]
    engine_action: dict[str, Any]
    logp: float
    terminal_money: float
    terminal_margin: float
    recurrent_state: RuntimeState
    quantity_token_count: int


class V2NumpyPolicy:
    FORMAT_VERSION = 1

    def __init__(self, arrays):
        if int(arrays["format_version"]) != self.FORMAT_VERSION:
            raise ValueError("unsupported v2 numpy format version")
        if str(arrays["feature_schema_sha256"]) != _feature_schema_sha256():
            raise ValueError("v2 feature schema hash mismatch")
        expected_parameters = str(arrays.get("exported_parameter_sha256", ""))
        if not expected_parameters or expected_parameters != _parameter_arrays_sha256(arrays):
            raise ValueError("v2 exported parameter hash mismatch")
        self.model_parameter_sha256 = str(arrays["model_parameter_sha256"])
        self.exported_parameter_sha256 = expected_parameters
        self.parameter_count = int(arrays["parameter_count"])
        self.w = {key[3:].replace("__", "."): np.asarray(value, np.float32)
                  for key, value in arrays.items() if key.startswith("p__")}

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=False)
        return cls({key: data[key] for key in data.files})

    def initial_state(self):
        return RuntimeState(np.zeros(256, np.float32), np.zeros(256, np.float32), {})

    def _tile_encode(self, grid):
        x = _linear(grid, self.w["encoder.tile_encoder.net.0.weight"], self.w["encoder.tile_encoder.net.0.bias"])
        x = _silu(x)
        x = _linear(x, self.w["encoder.tile_encoder.net.2.weight"], self.w["encoder.tile_encoder.net.2.bias"])
        return _layer_norm(x, self.w["encoder.tile_encoder.net.3.weight"], self.w["encoder.tile_encoder.net.3.bias"])

    def _grid_pass(self, x):
        for index in range(2):
            total = np.zeros_like(x); count = np.zeros((*x.shape[:-1], 1), np.float32)
            total[1:] += x[:-1]; count[1:] += 1
            total[:-1] += x[1:]; count[:-1] += 1
            total[:, 1:] += x[:, :-1]; count[:, 1:] += 1
            total[:, :-1] += x[:, 1:]; count[:, :-1] += 1
            neighbor = total / np.maximum(count, 1.0)
            prefix = f"encoder.grid_pass.rounds.{index}"
            delta = _silu(
                _linear(x, self.w[prefix + ".self_linear.weight"], self.w[prefix + ".self_linear.bias"])
                + _linear(neighbor, self.w[prefix + ".neighbor_linear.weight"], self.w[prefix + ".neighbor_linear.bias"])
            )
            x = _layer_norm(x + delta, self.w[prefix + ".norm.weight"], self.w[prefix + ".norm.bias"])
        return x.astype(np.float32)

    @staticmethod
    def _local_tiles(units, tiles):
        if units.shape[0] == 0:
            return np.zeros((0, tiles.shape[-1]), np.float32)
        h, w = tiles.shape[:2]
        x = np.rint(units[:, UNIT_INDEX["x"]] * max(1, w - 1)).astype(np.int64)
        y = np.rint(units[:, UNIT_INDEX["y"]] * max(1, h - 1)).astype(np.int64)
        x = np.clip(x, 0, w - 1); y = np.clip(y, 0, h - 1)
        return tiles[y, x]

    def _unit_tokens(self, units, tiles, previous_actions, previous_effects, side):
        local = self._local_tiles(units, tiles)
        side_bits = np.zeros((units.shape[0], 2), np.float32)
        if units.shape[0]: side_bits[:, side] = 1.0
        x = np.concatenate([units, previous_actions, previous_effects, local, side_bits], axis=-1)
        x = _linear(x, self.w["encoder.unit_encoder.input_proj.0.weight"], self.w["encoder.unit_encoder.input_proj.0.bias"])
        x = _layer_norm(x, self.w["encoder.unit_encoder.input_proj.1.weight"], self.w["encoder.unit_encoder.input_proj.1.bias"])
        return _silu(x)

    def _self_attention(self, x, prefix):
        in_w = self.w[prefix + ".self_attn.in_proj_weight"]
        in_b = self.w[prefix + ".self_attn.in_proj_bias"]
        qkv = _linear(x, in_w, in_b)
        q, k, v = np.split(qkv, 3, axis=-1)
        heads = 4; dim = q.shape[-1] // heads
        q = q.reshape(q.shape[0], heads, dim).transpose(1, 0, 2)
        k = k.reshape(k.shape[0], heads, dim).transpose(1, 0, 2)
        v = v.reshape(v.shape[0], heads, dim).transpose(1, 0, 2)
        scores = np.matmul(q, np.swapaxes(k, -1, -2)) / math.sqrt(float(dim))
        attn = _softmax(scores, axis=-1)
        context = np.matmul(attn, v).transpose(1, 0, 2).reshape(x.shape[0], -1)
        return _linear(
            context,
            self.w[prefix + ".self_attn.out_proj.weight"],
            self.w[prefix + ".self_attn.out_proj.bias"],
        )

    def _transformer(self, tokens):
        x = tokens
        for index in range(2):
            prefix = f"encoder.unit_encoder.transformer.layers.{index}"
            norm1 = _layer_norm(x, self.w[prefix + ".norm1.weight"], self.w[prefix + ".norm1.bias"])
            x = x + self._self_attention(norm1, prefix)
            norm2 = _layer_norm(x, self.w[prefix + ".norm2.weight"], self.w[prefix + ".norm2.bias"])
            ff = _linear(norm2, self.w[prefix + ".linear1.weight"], self.w[prefix + ".linear1.bias"])
            ff = _gelu(ff)
            ff = _linear(ff, self.w[prefix + ".linear2.weight"], self.w[prefix + ".linear2.bias"])
            x = x + ff
        return x.astype(np.float32)

    def _mlp_norm(self, x, prefix):
        x = _linear(x, self.w[prefix + ".0.weight"], self.w[prefix + ".0.bias"])
        x = _silu(x)
        x = _linear(x, self.w[prefix + ".2.weight"], self.w[prefix + ".2.bias"])
        return _layer_norm(x, self.w[prefix + ".3.weight"], self.w[prefix + ".3.bias"])

    def _fusion(self, x):
        x = _linear(x, self.w["encoder.fusion.0.weight"], self.w["encoder.fusion.0.bias"])
        x = _silu(x)
        x = _linear(x, self.w["encoder.fusion.2.weight"], self.w["encoder.fusion.2.bias"])
        return _layer_norm(x, self.w["encoder.fusion.3.weight"], self.w["encoder.fusion.3.bias"])

    def _lstm_step(self, fused, recurrent_state):
        if recurrent_state is None:
            h = np.zeros(256, np.float32); c = np.zeros(256, np.float32)
        elif isinstance(recurrent_state, RuntimeState):
            h, c = recurrent_state.h, recurrent_state.c
        else:
            h, c = recurrent_state[:2]
        gates = (
            _linear(fused, self.w["core.lstm.weight_ih"], self.w["core.lstm.bias_ih"])
            + _linear(h, self.w["core.lstm.weight_hh"], self.w["core.lstm.bias_hh"])
        )
        i, f, g, o = np.split(gates, 4)
        i = _sigmoid(i); f = _sigmoid(f); o = _sigmoid(o); g = np.tanh(g)
        c2 = f * c + i * g
        h2 = o * np.tanh(c2)
        intent = np.tanh(_linear(h2, self.w["core.intent.0.weight"], self.w["core.intent.0.bias"]))
        return h2.astype(np.float32), c2.astype(np.float32), intent.astype(np.float32)

    def debug_encode(self, structured_state, previous_effect=None, previous_action=None, recurrent_state=None):
        effect_value = {} if previous_effect is None else previous_effect
        action_value = {} if previous_action is None else previous_action
        features = _tensorize_state(structured_state, effect_value, action_value)
        own_tiles = self._grid_pass(self._tile_encode(features["own_grid"]))
        rival_tiles = self._grid_pass(self._tile_encode(features["rival_grid"]))
        own_tokens = self._unit_tokens(
            features["own_units"], own_tiles, features["previous_unit_actions"],
            features["previous_unit_effects"], side=0
        )
        rival_previous = np.zeros((features["rival_units"].shape[0], len(PREV_UNIT_ACTION_FEATURES)), np.float32)
        rival_effects = np.zeros((features["rival_units"].shape[0], len(PREV_UNIT_EFFECT_FEATURES)), np.float32)
        rival_tokens = self._unit_tokens(
            features["rival_units"], rival_tiles, rival_previous, rival_effects, side=1
        )
        tokens = np.concatenate([own_tokens, rival_tokens], axis=0)
        context = self._transformer(tokens)
        own_count = own_tokens.shape[0]
        own_ctx = context[:own_count]; rival_ctx = context[own_count:]
        own_pool = own_ctx.mean(axis=0) if own_count else np.zeros(128, np.float32)
        rival_pool = rival_ctx.mean(axis=0) if rival_ctx.shape[0] else np.zeros(128, np.float32)
        commodity = self._mlp_norm(features["commodities"], "encoder.commodity_encoder").mean(axis=0)
        economy = self._mlp_norm(features["economy"], "encoder.economy_encoder")
        previous = self._mlp_norm(features["previous_action_global"], "encoder.previous_action_encoder")
        effect = self._mlp_norm(features["previous_effect"], "encoder.effect_encoder")
        fused = self._fusion(np.concatenate([
            own_tiles.mean(axis=(0, 1)), rival_tiles.mean(axis=(0, 1)),
            own_pool, rival_pool, commodity, economy, previous, effect,
        ]))
        h, c, intent = self._lstm_step(fused, recurrent_state)
        return {"fused": fused.astype(np.float32), "own_unit_ctx": own_ctx.astype(np.float32),
                "rival_unit_ctx": rival_ctx.astype(np.float32), "own_tiles": own_tiles.astype(np.float32),
                "rival_tiles": rival_tiles.astype(np.float32), "h": h, "c": c, "intent": intent}

    @staticmethod
    def _draw(logits, allowed, rng, deterministic):
        logits = np.asarray(logits, np.float64).copy()
        mask = np.asarray(allowed, dtype=bool)
        if mask.shape != logits.shape or not mask.any():
            raise RuntimeError("empty or invalid legal mask")
        logits[~mask] = -1e9
        probs = _softmax(logits, axis=-1)
        if deterministic:
            choice = int(np.argmax(probs))
        else:
            choice = int(rng.choice(len(probs), p=probs))
        return choice, float(math.log(max(float(probs[choice]), 1e-30)))

    def _gru_cell(self, x, h, prefix):
        gi = _linear(x, self.w[prefix + ".weight_ih"], self.w[prefix + ".bias_ih"])
        gh = _linear(h, self.w[prefix + ".weight_hh"], self.w[prefix + ".bias_hh"])
        i_r, i_z, i_n = np.split(gi, 3)
        h_r, h_z, h_n = np.split(gh, 3)
        r = _sigmoid(i_r + h_r); z = _sigmoid(i_z + h_z)
        n = np.tanh(i_n + r * h_n)
        return ((1.0 - z) * n + z * h).astype(np.float32)

    @staticmethod
    def _parts(action, domain):
        if domain == "unit":
            return str(action.get("op", "PASS")), action.get("item"), action.get("quantity")
        kind = str(action.get("kind", "ORDER"))
        if kind in {"STOP_QUEUE", "NOP_SLOT"}:
            return kind, None, None
        return str(action.get("op", "NOP_SLOT")), action.get("item"), action.get("quantity")

    def _semantic_embedding(self, action, domain):
        op, item, quantity = self._parts(action, domain)
        if domain == "unit":
            op_index = 1 + UNIT_OP_TO_ID.get(op, UNIT_OP_TO_ID["PASS"])
        else:
            op_index = 1 + len(UNIT_OPS) + MARKET_OP_TO_ID.get(op, MARKET_OP_TO_ID["NOP_SLOT"])
        item_index = int(ITEM_TO_ID.get(str(item), 0)) if item is not None else 0
        quantity_features = np.asarray([
            1.0 if quantity is None else 0.0,
            _signed_log1p(quantity if quantity is not None else 0),
        ], np.float32)
        vector = np.concatenate([
            self.w["op_embedding.weight"][op_index],
            self.w["item_embedding.weight"][item_index],
            quantity_features,
        ])
        return _silu(_linear(vector, self.w["action_proj.0.weight"], self.w["action_proj.0.bias"])).astype(np.float32)

    @staticmethod
    def _ledger_vector(ledger):
        next_land = ledger.next_land_cost
        return np.asarray([
            _signed_log1p(ledger.cash_lower_bound), float(ledger.cash_uncertain),
            _signed_log1p(ledger.hires_today), _signed_log1p(ledger.next_hire_cost),
            _signed_log1p(next_land or 0), 1.0 if next_land is None else 0.0,
            _signed_log1p(sum(max(0, int(v)) for v in ledger.shed.values())),
            _signed_log1p(ledger._shed_room()), float(ledger.shed_uncertain),
            _signed_log1p(sum(max(0, int(v)) for v in ledger.plant_demand.values())),
            float(ledger.market_slots_used) / 10.0, float(ledger.market_stopped),
        ], np.float32)

    def _decoder_step(self, actor_ctx, previous, hidden, global_h, intent, ledger,
                      remaining_units, remaining_market):
        ledger_ctx = _silu(_linear(
            self._ledger_vector(ledger), self.w["ledger_proj.0.weight"], self.w["ledger_proj.0.bias"]
        ))
        x = np.concatenate([
            actor_ctx, previous, ledger_ctx, global_h, intent,
            np.asarray([remaining_units, remaining_market], np.float32),
        ])
        return self._gru_cell(x, hidden, "decoder_cell")

    @staticmethod
    def _encode_quantity(value):
        if value is None:
            return (OMIT_ID,)
        value = int(value)
        if value < 0:
            raise ValueError("runtime neural quantity must be nonnegative")
        return tuple(DIGIT_OFFSET + int(ch) for ch in str(value)) + (END_ID,)

    @staticmethod
    def _decode_quantity(tokens):
        values = [int(x) for x in tokens]
        if values == [OMIT_ID]:
            return None
        digits = []
        for token in values:
            if token == END_ID:
                if not digits:
                    raise ValueError("empty quantity")
                return int("".join(digits))
            digit = token - DIGIT_OFFSET
            if not 0 <= digit <= 9:
                raise ValueError("invalid quantity token")
            digits.append(str(digit))
        raise ValueError("unterminated quantity")

    @staticmethod
    def _quantity_mask(step_index, tokens, positive, max_value=None):
        mask = np.zeros(QUANTITY_VOCAB, dtype=bool)
        digit_tokens = [
            int(token) for token in tokens
            if int(token) not in {OMIT_ID, END_ID}
        ]
        digits = len(digit_tokens)
        if step_index == 0:
            if positive:
                mask[DIGIT_OFFSET + 1:DIGIT_OFFSET + 10] = True
            else:
                mask[OMIT_ID] = True; mask[DIGIT_OFFSET:DIGIT_OFFSET + 10] = True
        else:
            mask[DIGIT_OFFSET:DIGIT_OFFSET + 10] = True; mask[END_ID] = True
            if digits >= MAX_QUANTITY_DIGITS:
                mask[:] = False; mask[END_ID] = True
            if len(tokens) == 1 and tokens[0] == DIGIT_OFFSET:
                mask[:] = False; mask[END_ID] = True
        if max_value is not None:
            max_value = int(max_value)
            for token_id in range(QUANTITY_VOCAB):
                if not mask[token_id]:
                    continue
                if token_id == OMIT_ID:
                    mask[token_id] = bool(not positive)
                    continue
                if token_id == END_ID:
                    if not digit_tokens:
                        mask[token_id] = False
                    else:
                        current = int("".join(
                            str(token - DIGIT_OFFSET) for token in digit_tokens
                        ))
                        mask[token_id] = current <= max_value
                    continue
                digit = token_id - DIGIT_OFFSET
                if not digit_tokens and digit == 0 and positive:
                    mask[token_id] = False
                    continue
                candidate = int("".join(
                    str(value) for value in [
                        *(token - DIGIT_OFFSET for token in digit_tokens), digit,
                    ]
                ))
                mask[token_id] = candidate <= max_value
        return mask

    def _quantity_state(self, decoder_hidden):
        context = np.tanh(_linear(
            decoder_hidden, self.w["quantity_context.0.weight"], self.w["quantity_context.0.bias"]
        ))
        return np.tanh(_linear(
            context, self.w["quantity_decoder.init.weight"], self.w["quantity_decoder.init.bias"]
        )).astype(np.float32)

    def _quantity_step(self, hidden, previous_token):
        embedding = self.w["quantity_decoder.embedding.weight"][int(previous_token)]
        next_hidden = self._gru_cell(embedding, hidden, "quantity_decoder.gru")
        logits = _linear(
            next_hidden, self.w["quantity_decoder.output.weight"], self.w["quantity_decoder.output.bias"]
        )
        return logits, next_hidden

    def _quantity_evaluate(self, decoder_hidden, quantity, positive, max_value=None):
        tokens = self._encode_quantity(quantity)
        hidden = self._quantity_state(decoder_hidden)
        previous = START_ID; history = []; logp = 0.0
        consumed = []
        for step_index, target in enumerate(tokens):
            logits, hidden = self._quantity_step(hidden, previous)
            mask = self._quantity_mask(step_index, consumed, positive, max_value=max_value)
            if not mask[int(target)]:
                raise ValueError("quantity target is illegal under runtime mask")
            _, lp = self._draw(logits, mask, None, True) if False else (None, None)
            masked = np.asarray(logits, np.float64).copy(); masked[~mask] = -1e9
            probs = _softmax(masked, axis=-1)
            logp += math.log(max(float(probs[int(target)]), 1e-30))
            history.append(np.asarray(logits, np.float32)); consumed.append(int(target)); previous = int(target)
        return tokens, float(logp), history

    def _quantity_sample(self, decoder_hidden, positive, rng, deterministic, max_value=None):
        hidden = self._quantity_state(decoder_hidden)
        previous = START_ID; tokens = []; logp = 0.0; history = []
        for step_index in range(MAX_QUANTITY_DIGITS + 1):
            logits, hidden = self._quantity_step(hidden, previous)
            mask = self._quantity_mask(step_index, tokens, positive, max_value=max_value)
            token, lp = self._draw(logits, mask, rng, deterministic)
            history.append(np.asarray(logits, np.float32)); tokens.append(token); logp += lp
            if token in {OMIT_ID, END_ID}:
                break
            previous = token
        return tuple(tokens), self._decode_quantity(tokens), float(logp), history

    @staticmethod
    def _target_logp(logits, mask, target):
        logits = np.asarray(logits, np.float64).copy(); mask = np.asarray(mask, dtype=bool)
        target = int(target)
        if not mask[target]:
            raise ValueError("teacher action is illegal under runtime mask")
        logits[~mask] = -1e9
        probs = _softmax(logits, axis=-1)
        return float(math.log(max(float(probs[target]), 1e-30)))

    @staticmethod
    def _item_mask(choices):
        mask = np.zeros(max(ITEM_TO_ID.values()) + 1, dtype=bool)
        for item, allowed in choices.items():
            item_id = ITEM_TO_ID.get(str(item))
            if item_id is not None and allowed:
                mask[int(item_id)] = True
        return mask

    @staticmethod
    def _quantity_max_from_legal(legal, domain, op, item):
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

    @staticmethod
    def _unit_raw(op, item, quantity):
        if op in {"PICKUP", "PLACE"}:
            return [op, item] + ([] if quantity is None else [int(quantity)])
        if op == "PLANT":
            return [op, item]
        return [op]

    @staticmethod
    def _market_raw(op, item, quantity):
        if op in {"STOP_QUEUE", "NOP_SLOT"}:
            return []
        if op in {"HIRE", "BUY_LAND"}:
            return [op]
        return [op, item, int(quantity)]

    def _capture_op_trace(self, actor, ops, logits, mask, chosen_op,
                          deterministic, teacher_forced=False):
        sink = getattr(self, "_trace_sink", None)
        if sink is None:
            return
        raw = np.asarray(logits, np.float64).copy()
        legal = np.asarray(mask, dtype=bool).copy()
        masked = raw.copy()
        masked[~legal] = -1e9
        sink.append({
            "actor": str(actor), "ops": list(ops),
            "raw_logits": raw.tolist(), "legal_mask": legal.tolist(),
            "masked_logits": masked.tolist(),
            "legal_action_count": int(legal.sum()),
            "raw_top1": str(ops[int(np.argmax(raw))]),
            "post_mask_top1": str(ops[int(np.argmax(masked))]),
            "chosen_op": str(chosen_op),
            "decision_reason": ("teacher_forced" if teacher_forced else
                                "model_argmax" if deterministic else "model_sample"),
        })

    def _unit_decision(self, actor, actor_ctx, hidden, global_h, intent, previous,
                       ledger, remaining_units, teacher, rng, deterministic):
        legal = ledger.legal_unit_mask(actor, {})
        hidden = self._decoder_step(
            actor_ctx, previous, hidden, global_h, intent, ledger, remaining_units, 1.0
        )
        op_logits = _linear(hidden, self.w["unit_op_head.weight"], self.w["unit_op_head.bias"])
        item_logits = _linear(hidden, self.w["item_head.weight"], self.w["item_head.bias"])
        op_mask = np.asarray([bool(legal.ops.get(op, False)) for op in UNIT_OPS], dtype=bool)
        logp = 0.0; quantity_count = 0
        if teacher is None:
            op_id, lp = self._draw(op_logits, op_mask, rng, deterministic)
            logp += lp; op = UNIT_OPS[op_id]
            chosen = {"op": op, "item": None, "quantity": None, "raw": [op]}
        else:
            chosen = dict(teacher); op = str(chosen.get("op", "PASS"))
            logp += self._target_logp(op_logits, op_mask, UNIT_OP_TO_ID[op])
        self._capture_op_trace(
            actor, UNIT_OPS, op_logits, op_mask, op, deterministic,
            teacher_forced=teacher is not None,
        )
        if op in UNIT_ITEM_OPS:
            item_mask = self._item_mask(legal.items.get(op, {}))
            if teacher is None:
                item_id, lp = self._draw(item_logits, item_mask, rng, deterministic)
                logp += lp; chosen["item"] = ID_TO_ITEM[item_id]
            else:
                logp += self._target_logp(item_logits, item_mask, ITEM_TO_ID[chosen["item"]])
        if op in UNIT_QUANTITY_OPS:
            max_value = self._quantity_max_from_legal(
                legal, "unit", op, chosen.get("item"),
            )
            if teacher is None:
                tokens, quantity, lp, _ = self._quantity_sample(
                    hidden, False, rng, deterministic, max_value=max_value,
                )
                chosen["quantity"] = quantity; logp += lp
            else:
                tokens, lp, _ = self._quantity_evaluate(
                    hidden, chosen.get("quantity"), False, max_value=max_value,
                )
                logp += lp
            quantity_count += len(tokens)
        chosen["raw"] = self._unit_raw(op, chosen.get("item"), chosen.get("quantity"))
        ledger.apply_unit(actor, chosen)
        return chosen, hidden, self._semantic_embedding(chosen, "unit"), float(logp), quantity_count

    def _market_decision(self, slot, actor_ctx, hidden, global_h, intent, previous,
                         ledger, teacher, rng, deterministic,
                         strategy_context=None):
        del strategy_context
        legal = ledger.legal_market_mask(slot, {})
        hidden = self._decoder_step(
            actor_ctx, previous, hidden, global_h, intent, ledger,
            0.0, float(max(0, 9 - slot)) / 10.0,
        )
        op_logits = _linear(hidden, self.w["market_op_head.weight"], self.w["market_op_head.bias"])
        item_logits = _linear(hidden, self.w["item_head.weight"], self.w["item_head.bias"])
        op_mask = np.asarray([bool(legal.ops.get(op, False)) for op in MARKET_OPS], dtype=bool)
        logp = 0.0; quantity_count = 0
        if teacher is None:
            op_id, lp = self._draw(op_logits, op_mask, rng, deterministic)
            logp += lp; op = MARKET_OPS[op_id]
            self._capture_op_trace(
                f"market:{slot}", MARKET_OPS, op_logits, op_mask, op,
                deterministic, teacher_forced=False,
            )
            if op in {"STOP_QUEUE", "NOP_SLOT"}:
                chosen = {"kind": op, "op": None, "item": None, "quantity": None, "raw": []}
            else:
                chosen = {"kind": "ORDER", "op": op, "item": None, "quantity": None, "raw": [op]}
        else:
            chosen = dict(teacher); kind = str(chosen.get("kind", "ORDER"))
            op = kind if kind in {"STOP_QUEUE", "NOP_SLOT"} else str(chosen.get("op", "NOP_SLOT"))
            logp += self._target_logp(op_logits, op_mask, MARKET_OP_TO_ID[op])
        if op in MARKET_ITEM_OPS:
            item_mask = self._item_mask(legal.items.get(op, {}))
            if teacher is None:
                item_id, lp = self._draw(item_logits, item_mask, rng, deterministic)
                logp += lp; chosen["item"] = ID_TO_ITEM[item_id]
            else:
                logp += self._target_logp(item_logits, item_mask, ITEM_TO_ID[chosen["item"]])
            max_value = self._quantity_max_from_legal(
                legal, "market", op, chosen.get("item"),
            )
            if op == "SELL" and (max_value is None or max_value < 1):
                raise RuntimeError("SELL action has no legal inventory bound")
            if teacher is None:
                tokens, quantity, lp, _ = self._quantity_sample(
                    hidden, True, rng, deterministic, max_value=max_value,
                )
                chosen["quantity"] = quantity; logp += lp
            else:
                tokens, lp, _ = self._quantity_evaluate(
                    hidden, chosen.get("quantity"), True, max_value=max_value,
                )
                logp += lp
            quantity_count += len(tokens)
        chosen["raw"] = self._market_raw(op, chosen.get("item"), chosen.get("quantity"))
        ledger.apply_market(chosen)
        return chosen, hidden, self._semantic_embedding(chosen, "market"), float(logp), quantity_count

    @staticmethod
    def _engine_action(canonical):
        market = []
        for command in canonical["market"][:10]:
            kind = str(command.get("kind", "ORDER"))
            if kind == "STOP_QUEUE":
                break
            if kind == "NOP_SLOT":
                market.append(["SELL", "WHEAT", 0])
            else:
                market.append(list(command.get("raw") or []))
        return {
            "farmer": list(canonical["farmer"].get("raw") or ["PASS"]),
            "hands": [list(command.get("raw") or ["PASS"]) for command in canonical["hands"]],
            "market": market,
        }

    def _decode_joint(self, structured_state, previous_effect, previous_action,
                      recurrent_state, rng, deterministic, teacher_action=None):
        debug = self.debug_encode(structured_state, previous_effect, previous_action, recurrent_state)
        h, c, intent = debug["h"], debug["c"], debug["intent"]
        ledger = ShadowLedger.from_state(structured_state)
        hidden = np.tanh(_linear(
            np.concatenate([h, intent]), self.w["decoder_init.0.weight"], self.w["decoder_init.0.bias"]
        )).astype(np.float32)
        previous = self.w["start_action"].astype(np.float32).copy()
        own_ctx = debug["own_unit_ctx"]; own_count = own_ctx.shape[0]
        if own_count < 1:
            raise ValueError("own unit set must include the main farmer")
        teacher_hands = list((teacher_action or {}).get("hands") or [])
        if teacher_action is not None and len(teacher_hands) != own_count - 1:
            raise ValueError("teacher hand count does not match state")
        logp = 0.0; quantity_count = 0
        farmer_teacher = (teacher_action or {}).get("farmer") if teacher_action is not None else None
        farmer, hidden, previous, lp, count = self._unit_decision(
            "farmer", own_ctx[0], hidden, h, intent, previous, ledger,
            float(max(0, own_count - 1)) / float(own_count),
            farmer_teacher, rng, deterministic,
        )
        logp += lp; quantity_count += count
        hands = []
        for hand_index in range(own_count - 1):
            teacher = teacher_hands[hand_index] if teacher_action is not None else None
            action, hidden, previous, lp, count = self._unit_decision(
                f"hand:{hand_index}", own_ctx[hand_index + 1], hidden, h, intent,
                previous, ledger,
                float(max(0, own_count - hand_index - 2)) / float(own_count),
                teacher, rng, deterministic,
            )
            hands.append(action); logp += lp; quantity_count += count
        market_teacher = list((teacher_action or {}).get("market") or []) if teacher_action is not None else None
        if teacher_action is not None and not market_teacher:
            market_teacher = [{"kind": "STOP_QUEUE", "op": None, "item": None,
                               "quantity": None, "raw": []}]
        market = []
        limit = min(10, len(market_teacher)) if market_teacher is not None else 10
        for slot in range(limit):
            teacher = market_teacher[slot] if market_teacher is not None else None
            slot_ctx = self.w["market_slot_embedding.weight"][slot]
            action, hidden, previous, lp, count = self._market_decision(
                slot, slot_ctx, hidden, h, intent, previous, ledger,
                teacher, rng, deterministic,
            )
            market.append(action); logp += lp; quantity_count += count
            if action.get("kind") == "STOP_QUEUE":
                break
        if not market:
            raise RuntimeError("market decoder emitted no slot")
        canonical = {"farmer": farmer, "hands": hands, "market": market}
        joint = np.concatenate([h, intent])
        terminal_money = float(_linear(
            joint, self.w["terminal_money_head.weight"], self.w["terminal_money_head.bias"]
        )[0])
        terminal_margin = float(_linear(
            joint, self.w["terminal_margin_head.weight"], self.w["terminal_margin_head.bias"]
        )[0])
        state = RuntimeState(h.copy(), c.copy(), canonical)
        return RuntimeStepOutput(
            canonical_action=canonical,
            engine_action=self._engine_action(canonical),
            logp=float(logp), terminal_money=terminal_money,
            terminal_margin=terminal_margin, recurrent_state=state,
            quantity_token_count=int(quantity_count),
        )

    def step(self, structured_state, previous_effect, previous_action, recurrent_state,
             rng, deterministic=False):
        if previous_action is None and isinstance(recurrent_state, RuntimeState):
            previous_action = recurrent_state.previous_action
        previous_action = {} if previous_action is None else previous_action
        rng = np.random.default_rng() if rng is None else rng
        return self._decode_joint(
            structured_state, previous_effect or {}, previous_action,
            recurrent_state, rng, bool(deterministic), teacher_action=None,
        )

    def evaluate_action(self, structured_state, previous_effect, previous_action,
                        recurrent_state, canonical_action):
        if previous_action is None and isinstance(recurrent_state, RuntimeState):
            previous_action = recurrent_state.previous_action
        previous_action = {} if previous_action is None else previous_action
        return self._decode_joint(
            structured_state, previous_effect or {}, previous_action,
            recurrent_state, np.random.default_rng(0), True,
            teacher_action=canonical_action,
        )
