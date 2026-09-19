from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .constants import ANIMALS, CROPS, ITEM_TO_ID, PRODUCTS, UNIT_OPS
from .v2_ledger import ShadowLedger

ROOT = Path(__file__).resolve().parents[2]
ITEM_NAMES = tuple(ITEM_TO_ID.keys())
SHOP_NAMES = (
    "BAKERY", "PIZZA_SHOP", "BRUNCH_SPOT", "YARN_STORE",
    "ICE_CREAM_SHOP", "PET_CAFE", "SMOOTHIE_SHOP", "FARMERS_MARKET",
)
SHOP_PRODUCTS = {
    "BAKERY": ("EGG", "WHEAT"),
    "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
    "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
    "YARN_STORE": ("WOOL",),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE": ("CARROT",),
    "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
}
TILE_KINDS = ("EMPTY", "LOCKED", "WEED", "PLANT", "COOP", "PASTURE", "OTHER")
DEFAULT_SHED_CAPACITY = 100
DEFAULT_MARKET_SLOTS = 10
LAND_PRICES = (1000, 2000, 4000)


class Stage0AcceptanceError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stage0_code_hashes() -> dict[str, str]:
    return {
        "dataset_code_sha256": _sha256(ROOT / "src/kaggrl/v2_dataset.py"),
        "action_schema_sha256": _sha256(ROOT / "src/kaggrl/v2_action_schema.py"),
        "observation_code_sha256": _sha256(ROOT / "src/kaggrl/v2_observation.py"),
        "effects_code_sha256": _sha256(ROOT / "src/kaggrl/v2_effects.py"),
        "builder_code_sha256": _sha256(ROOT / "training/build_rl_v2_dataset.py"),
        "audit_code_sha256": _sha256(ROOT / "evaluation/audit_rl_v2_dataset.py"),
    }


def verify_stage0_acceptance(live_root: str | Path) -> dict[str, Any]:
    live = Path(live_root)
    manifests = live / "manifests"
    marker_path = manifests / "STAGE0_ACCEPTED.json"
    if not marker_path.exists():
        raise Stage0AcceptanceError(f"missing Stage 0 marker: {marker_path}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("accepted") is not True:
        raise Stage0AcceptanceError("Stage 0 marker is not accepted")
    if any(int(v) != 0 for v in (marker.get("counters") or {}).values()):
        raise Stage0AcceptanceError("Stage 0 marker contains nonzero counters")
    artifacts = marker.get("artifacts") or {}
    expected_files = {
        "dataset_sha256": live / "transitions.parquet",
        "selection_snapshot_sha256": manifests / "live_top10_snapshot.json",
        "trusted_corpus_manifest_sha256": manifests / "trusted_corpus_manifest.json",
    }
    for key, path in expected_files.items():
        if not path.exists() or artifacts.get(key) != _sha256(path):
            raise Stage0AcceptanceError(f"Stage 0 artifact mismatch: {key}")
    for key, value in stage0_code_hashes().items():
        if artifacts.get(key) != value:
            raise Stage0AcceptanceError(f"Stage 0 code mismatch: {key}")
    return marker


def signed_log1p(value: float | int) -> float:
    x = float(value or 0.0)
    return math.copysign(math.log1p(abs(x)), x) if x else 0.0


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "__dict__"):
        return vars(value)
    return {}


def _field(state: Any, name: str, default=None):
    if isinstance(state, dict):
        return state.get(name, default)
    return getattr(state, name, default)


TILE_FEATURES = (
    "x", "y",
    *(f"kind:{name}" for name in TILE_KINDS),
    *(f"crop:{name}" for name in CROPS),
    *(f"animal:{name}" for name in ANIMALS),
    "watered_today", "fed_today", "cared_today", "fertilizer_available",
    "yield_units", "planted_day", "placed_day", "max_lifespan_step",
    "fertilized_until_day", "consecutive_unwatered", "consecutive_unfed",
    "pending_care_bonus",
)
TILE_FEATURE_INDEX = {name: i for i, name in enumerate(TILE_FEATURES)}

UNIT_FEATURES = (
    "kind:farmer", "kind:hand", "actor_index_log", "actor_index_sin", "actor_index_cos",
    "x", "y", "depot_distance", *(f"inventory:{item}" for item in ITEM_NAMES),
)
UNIT_FEATURE_INDEX = {name: i for i, name in enumerate(UNIT_FEATURES)}

MARKET_ACTION_NAMES = (
    "STOP_QUEUE", "NOP_SLOT", "BUY_SEED", "BUY_PRODUCT",
    "BUY_ANIMAL", "SELL", "HIRE", "BUY_LAND",
)
PREV_UNIT_ACTION_FEATURES = (
    "none", *(f"op:{op}" for op in UNIT_OPS),
    "item:none", *(f"item:{item}" for item in ITEM_NAMES),
    "quantity_omitted", "quantity_log",
)
PREV_UNIT_ACTION_FEATURE_INDEX = {
    name: i for i, name in enumerate(PREV_UNIT_ACTION_FEATURES)
}
PREV_UNIT_EFFECT_STATUSES = ("confirmed", "failed", "unconfirmed")
PREV_UNIT_EFFECT_FEATURES = (
    "none", *(f"status:{name}" for name in PREV_UNIT_EFFECT_STATUSES),
    *(f"op:{op}" for op in UNIT_OPS), "dx", "dy", "move_l1",
    *(f"inventory_delta:{item}" for item in ITEM_NAMES),
)
PREV_UNIT_EFFECT_FEATURE_INDEX = {
    name: i for i, name in enumerate(PREV_UNIT_EFFECT_FEATURES)
}
PREV_ACTION_GLOBAL_FEATURES = (
    "none", *(f"market_op_count:{op}" for op in MARKET_ACTION_NAMES),
    *(f"market_item_count:{item}" for item in ITEM_NAMES),
    "market_quantity_count", "market_quantity_log_sum",
)
PREV_ACTION_GLOBAL_FEATURE_INDEX = {
    name: i for i, name in enumerate(PREV_ACTION_GLOBAL_FEATURES)
}

COMMODITY_NAMES = ITEM_NAMES
COMMODITY_TO_INDEX = {name: i for i, name in enumerate(COMMODITY_NAMES)}
COMMODITY_FEATURES = (
    *(f"item:{name}" for name in COMMODITY_NAMES),
    "is_crop", "is_animal", "is_product", "shed_quantity", "seed_quantity",
    "carried_quantity", "market_inventory", "market_price",
)
COMMODITY_FEATURE_INDEX = {name: i for i, name in enumerate(COMMODITY_FEATURES)}

ECONOMY_FEATURES = (
    "step_log", "day_log", "hour_sin", "hour_cos", "turns_to_day_end_log", "turns_to_game_end_log",
    "own_money", "rival_money", "own_hires_today", "rival_hires_today",
    "own_hand_count", "rival_hand_count", "own_land_count", "rival_land_count",
    "next_hire_cost", "next_land_cost", "land_exhausted",
    "shed_capacity", "shed_free_space", "market_slots_remaining",
    *(f"shed:{item}" for item in ITEM_NAMES),
    *(f"seed:{crop}" for crop in CROPS),
    *(f"market_inventory:{item}" for item in PRODUCTS),
    *(f"market_price:{item}" for item in PRODUCTS),
    *(f"shop_count:{shop}" for shop in SHOP_NAMES),
)
ECONOMY_FEATURE_INDEX = {name: i for i, name in enumerate(ECONOMY_FEATURES)}


EFFECT_FEATURES = (
    "money_delta", "hand_count_delta", "day_changed", "day_reset", "unit_move_l1_total",
    *(f"shed_delta:{item}" for item in ITEM_NAMES),
    *(f"seed_delta:{crop}" for crop in CROPS),
    *(f"market_inventory_delta:{item}" for item in PRODUCTS),
    *(f"market_price_delta:{item}" for item in PRODUCTS),
    "rival_money_delta", "rival_hand_count_delta",
)
EFFECT_FEATURE_INDEX = {name: i for i, name in enumerate(EFFECT_FEATURES)}


@dataclass
class TensorizedState:
    own_grid: torch.Tensor
    rival_grid: torch.Tensor
    own_units: torch.Tensor
    rival_units: torch.Tensor
    previous_unit_actions: torch.Tensor
    previous_unit_effects: torch.Tensor
    commodities: torch.Tensor
    economy: torch.Tensor
    previous_action_global: torch.Tensor
    previous_effect: torch.Tensor


@dataclass
class V2Batch:
    own_grid: torch.Tensor
    rival_grid: torch.Tensor
    own_units: torch.Tensor
    own_unit_mask: torch.Tensor
    rival_units: torch.Tensor
    rival_unit_mask: torch.Tensor
    previous_unit_actions: torch.Tensor
    previous_unit_effects: torch.Tensor
    commodities: torch.Tensor
    hand_action_mask: torch.Tensor
    market_slot_mask: torch.Tensor
    economy: torch.Tensor
    previous_action_global: torch.Tensor
    previous_effect: torch.Tensor
    structured_states: tuple[Any, ...]
    canonical_actions: tuple[Any, ...]
    previous_actions: tuple[Any, ...]
    effects_targets: tuple[Any, ...]

    def model_inputs(self) -> dict[str, Any]:
        return {
            "own_grid": self.own_grid,
            "rival_grid": self.rival_grid,
            "own_units": self.own_units,
            "own_unit_mask": self.own_unit_mask,
            "rival_units": self.rival_units,
            "rival_unit_mask": self.rival_unit_mask,
            "previous_unit_actions": self.previous_unit_actions,
            "previous_unit_effects": self.previous_unit_effects,
            "commodities": self.commodities,
            "economy": self.economy,
            "previous_action_global": self.previous_action_global,
            "previous_effect": self.previous_effect,
            "structured_states": self.structured_states,
        }


def _tile_kind(tile: Any) -> str:
    if tile is None:
        return "EMPTY"
    if isinstance(tile, str):
        return tile if tile in TILE_KINDS else "OTHER"
    if isinstance(tile, dict):
        value = str(tile.get("kind", "OTHER"))
        return value if value in TILE_KINDS else "OTHER"
    return "OTHER"


def _tile_vector(tile: Any, x: int, y: int, board_size: int) -> torch.Tensor:
    out = torch.zeros(len(TILE_FEATURES), dtype=torch.float32)
    denom = max(1, board_size - 1)
    out[TILE_FEATURE_INDEX["x"]] = float(x) / denom
    out[TILE_FEATURE_INDEX["y"]] = float(y) / denom
    kind = _tile_kind(tile)
    out[TILE_FEATURE_INDEX[f"kind:{kind}"]] = 1.0
    data = tile if isinstance(tile, dict) else {}
    crop = data.get("crop")
    animal = data.get("animal")
    if crop in CROPS:
        out[TILE_FEATURE_INDEX[f"crop:{crop}"]] = 1.0
    if animal in ANIMALS:
        out[TILE_FEATURE_INDEX[f"animal:{animal}"]] = 1.0
    for name in ("watered_today", "fed_today", "cared_today", "fertilizer_available"):
        out[TILE_FEATURE_INDEX[name]] = float(bool(data.get(name, False)))
    for name in (
        "yield_units", "planted_day", "placed_day", "max_lifespan_step",
        "fertilized_until_day", "consecutive_unwatered", "consecutive_unfed",
        "pending_care_bonus",
    ):
        out[TILE_FEATURE_INDEX[name]] = signed_log1p(data.get(name, 0) or 0)
    return out


def _grid_tensor(grid: Any) -> torch.Tensor:
    rows = list(grid or [])
    if not rows:
        rows = [[None for _ in range(10)] for _ in range(10)]
    height = len(rows)
    width = max((len(row) for row in rows), default=0)
    if height != width:
        raise ValueError(f"farm grid must be square, got {height}x{width}")
    out = torch.zeros((height, width, len(TILE_FEATURES)), dtype=torch.float32)
    for y, row in enumerate(rows):
        if len(row) != width:
            raise ValueError("ragged farm grid")
        for x, tile in enumerate(row):
            out[y, x] = _tile_vector(tile, x, y, width)
    return out


def _unit_vector(unit: Any, *, board_size: int, own_private: bool) -> torch.Tensor:
    data = _mapping(unit)
    out = torch.zeros(len(UNIT_FEATURES), dtype=torch.float32)
    kind = str(data.get("kind", "hand"))
    out[UNIT_FEATURE_INDEX["kind:farmer"]] = float(kind == "farmer")
    out[UNIT_FEATURE_INDEX["kind:hand"]] = float(kind == "hand")
    actor_position = 0 if kind == "farmer" else int(data.get("index", 0)) + 1
    out[UNIT_FEATURE_INDEX["actor_index_log"]] = math.log1p(max(0, actor_position))
    out[UNIT_FEATURE_INDEX["actor_index_sin"]] = math.sin(float(actor_position))
    out[UNIT_FEATURE_INDEX["actor_index_cos"]] = math.cos(float(actor_position))
    position = data.get("position") or [0, 0]
    denom = max(1, board_size - 1)
    if isinstance(position, (list, tuple)) and len(position) >= 2:
        px, py = int(position[0]), int(position[1])
        out[UNIT_FEATURE_INDEX["x"]] = float(px) / denom
        out[UNIT_FEATURE_INDEX["y"]] = float(py) / denom
        half = board_size // 2
        depot_access = ((half - 1, half - 1), (half, half - 1),
                        (half - 1, half), (half, half))
        distance = min(abs(px - dx) + abs(py - dy) for dx, dy in depot_access)
        out[UNIT_FEATURE_INDEX["depot_distance"]] = float(distance) / denom
    inventory = data.get("inventory") if own_private else {}
    inventory = inventory if isinstance(inventory, dict) else {}
    for item in ITEM_NAMES:
        out[UNIT_FEATURE_INDEX[f"inventory:{item}"]] = signed_log1p(inventory.get(item, 0) or 0)
    return out


def _units_tensor(units: Any, *, board_size: int, own_private: bool) -> torch.Tensor:
    values = list(units or [])
    if not values:
        return torch.zeros((0, len(UNIT_FEATURES)), dtype=torch.float32)
    return torch.stack([
        _unit_vector(unit, board_size=board_size, own_private=own_private)
        for unit in values
    ])


def _commodity_tensor(state: Any) -> torch.Tensor:
    private = _mapping(_field(state, "private", {}))
    shed = _mapping(private.get("shed") or {})
    seeds = _mapping(private.get("seeds") or {})
    market = _mapping(_field(state, "market", {}))
    market_inventory = _mapping(market.get("inventory") or {})
    market_prices = _mapping(market.get("prices") or {})
    carried = {name: 0 for name in COMMODITY_NAMES}
    for unit in list(_field(state, "own_units", []) or []):
        inventory = _mapping(_mapping(unit).get("inventory") or {})
        for name in COMMODITY_NAMES:
            carried[name] += int(inventory.get(name, 0) or 0)
    out = torch.zeros((len(COMMODITY_NAMES), len(COMMODITY_FEATURES)), dtype=torch.float32)
    for row, name in enumerate(COMMODITY_NAMES):
        out[row, COMMODITY_FEATURE_INDEX[f"item:{name}"]] = 1.0
        out[row, COMMODITY_FEATURE_INDEX["is_crop"]] = float(name in CROPS)
        out[row, COMMODITY_FEATURE_INDEX["is_animal"]] = float(name in ANIMALS)
        out[row, COMMODITY_FEATURE_INDEX["is_product"]] = float(name in PRODUCTS)
        out[row, COMMODITY_FEATURE_INDEX["shed_quantity"]] = signed_log1p(shed.get(name, 0))
        out[row, COMMODITY_FEATURE_INDEX["seed_quantity"]] = signed_log1p(seeds.get(name, 0))
        out[row, COMMODITY_FEATURE_INDEX["carried_quantity"]] = signed_log1p(carried[name])
        out[row, COMMODITY_FEATURE_INDEX["market_inventory"]] = signed_log1p(market_inventory.get(name, 0))
        out[row, COMMODITY_FEATURE_INDEX["market_price"]] = signed_log1p(market_prices.get(name, 0))
    return out


def _previous_unit_action_vector(command: Any) -> torch.Tensor:
    out = torch.zeros(len(PREV_UNIT_ACTION_FEATURES), dtype=torch.float32)
    data = _mapping(command)
    if not data:
        out[PREV_UNIT_ACTION_FEATURE_INDEX["none"]] = 1.0
        return out
    op = str(data.get("op", "PASS"))
    if op in UNIT_OPS:
        out[PREV_UNIT_ACTION_FEATURE_INDEX[f"op:{op}"]] = 1.0
    item = data.get("item")
    item_name = str(item) if item is not None else None
    if item_name in ITEM_NAMES:
        out[PREV_UNIT_ACTION_FEATURE_INDEX[f"item:{item_name}"]] = 1.0
    else:
        out[PREV_UNIT_ACTION_FEATURE_INDEX["item:none"]] = 1.0
    quantity = data.get("quantity")
    if quantity is None:
        out[PREV_UNIT_ACTION_FEATURE_INDEX["quantity_omitted"]] = 1.0
    else:
        out[PREV_UNIT_ACTION_FEATURE_INDEX["quantity_log"]] = signed_log1p(quantity)
    return out


def _previous_unit_actions_tensor(state: Any, previous_action: Any, previous_effect: Any) -> torch.Tensor:
    units = list(_field(state, "own_units", []) or [])
    out = torch.zeros((len(units), len(PREV_UNIT_ACTION_FEATURES)), dtype=torch.float32)
    previous = _mapping(previous_action)
    effect = _mapping(previous_effect)
    farmer = previous.get("farmer") if previous else None
    hands = list(previous.get("hands") or []) if previous else []
    day_reset = bool(effect.get("day_reset", False))
    for slot in range(len(units)):
        if slot == 0:
            command = farmer
        elif day_reset:
            command = None
        else:
            hand_index = slot - 1
            command = hands[hand_index] if hand_index < len(hands) else None
        out[slot] = _previous_unit_action_vector(command)
    return out


def _previous_unit_effects_tensor(state: Any, previous_effect: Any) -> torch.Tensor:
    units = list(_field(state, "own_units", []) or [])
    out = torch.zeros((len(units), len(PREV_UNIT_EFFECT_FEATURES)), dtype=torch.float32)
    if len(units):
        out[:, PREV_UNIT_EFFECT_FEATURE_INDEX["none"]] = 1.0
    effect = _mapping(previous_effect)
    evidence_by_actor = {}
    for evidence in list(effect.get("action_evidence") or []):
        data = _mapping(evidence)
        actor = str(data.get("actor", ""))
        if actor:
            evidence_by_actor[actor] = data
    position_delta = _mapping(effect.get("unit_position_delta") or {})
    day_reset = bool(effect.get("day_reset", False))
    for slot, unit in enumerate(units):
        unit_data = _mapping(unit)
        actor = "farmer" if str(unit_data.get("kind")) == "farmer" else f"hand:{int(unit_data.get('index', slot - 1))}"
        if day_reset and actor.startswith("hand:"):
            continue
        evidence = evidence_by_actor.get(actor)
        delta = position_delta.get(actor)
        if evidence is not None or delta is not None:
            out[slot, PREV_UNIT_EFFECT_FEATURE_INDEX["none"]] = 0.0
        if evidence is not None:
            status = str(evidence.get("status", "unconfirmed"))
            if status in PREV_UNIT_EFFECT_STATUSES:
                out[slot, PREV_UNIT_EFFECT_FEATURE_INDEX[f"status:{status}"]] = 1.0
            op = str(evidence.get("op", ""))
            if op in UNIT_OPS:
                out[slot, PREV_UNIT_EFFECT_FEATURE_INDEX[f"op:{op}"]] = 1.0
            observed = _mapping(evidence.get("observed") or {})
            inv_delta = _mapping(observed.get("inventory_delta") or {})
            for item in ITEM_NAMES:
                out[slot, PREV_UNIT_EFFECT_FEATURE_INDEX[f"inventory_delta:{item}"]] = signed_log1p(inv_delta.get(item, 0))
        if isinstance(delta, (list, tuple)) and len(delta) >= 2:
            dx, dy = float(delta[0]), float(delta[1])
            out[slot, PREV_UNIT_EFFECT_FEATURE_INDEX["dx"]] = dx
            out[slot, PREV_UNIT_EFFECT_FEATURE_INDEX["dy"]] = dy
            out[slot, PREV_UNIT_EFFECT_FEATURE_INDEX["move_l1"]] = signed_log1p(abs(dx) + abs(dy))
    return out


def _previous_action_global_tensor(previous_action: Any) -> torch.Tensor:
    out = torch.zeros(len(PREV_ACTION_GLOBAL_FEATURES), dtype=torch.float32)
    previous = _mapping(previous_action)
    if not previous:
        out[PREV_ACTION_GLOBAL_FEATURE_INDEX["none"]] = 1.0
        return out
    market = list(previous.get("market") or [])
    op_counts = {name: 0 for name in MARKET_ACTION_NAMES}
    item_counts = {name: 0 for name in ITEM_NAMES}
    quantity_count = 0
    quantity_log_sum = 0.0
    for slot in market:
        data = _mapping(slot)
        kind = str(data.get("kind", "ORDER"))
        op = kind if kind in {"STOP_QUEUE", "NOP_SLOT"} else str(data.get("op", "NOP_SLOT"))
        if op in op_counts:
            op_counts[op] += 1
        item = data.get("item")
        if item in item_counts:
            item_counts[item] += 1
        quantity = data.get("quantity")
        if quantity is not None:
            quantity_count += 1
            quantity_log_sum += signed_log1p(quantity)
    for op, count in op_counts.items():
        out[PREV_ACTION_GLOBAL_FEATURE_INDEX[f"market_op_count:{op}"]] = signed_log1p(count)
    for item, count in item_counts.items():
        out[PREV_ACTION_GLOBAL_FEATURE_INDEX[f"market_item_count:{item}"]] = signed_log1p(count)
    out[PREV_ACTION_GLOBAL_FEATURE_INDEX["market_quantity_count"]] = signed_log1p(quantity_count)
    out[PREV_ACTION_GLOBAL_FEATURE_INDEX["market_quantity_log_sum"]] = float(quantity_log_sum)
    return out


LAND_PRICES = (1000, 2000, 4000)
DEFAULT_SHED_CAPACITY = 100


def _count_land(farm: dict[str, Any]) -> int:
    return len(farm.get("unlocked_quadrants") or [])


def _next_hire_cost(hires_today: int) -> int:
    a, b = 1, 1
    for _ in range(max(0, int(hires_today))):
        a, b = b, a + b
    return a


def _economy_tensor(state: Any) -> torch.Tensor:
    out = torch.zeros(len(ECONOMY_FEATURES), dtype=torch.float32)
    step = int(_field(state, "step", 0) or 0)
    day = int(_field(state, "day", step // 24) or 0)
    hour = int(_field(state, "hour", step % 24) or 0)
    out[ECONOMY_FEATURE_INDEX["step_log"]] = math.log1p(max(0, step))
    out[ECONOMY_FEATURE_INDEX["day_log"]] = math.log1p(max(0, day))
    angle = 2.0 * math.pi * (hour % 24) / 24.0
    out[ECONOMY_FEATURE_INDEX["hour_sin"]] = math.sin(angle)
    out[ECONOMY_FEATURE_INDEX["hour_cos"]] = math.cos(angle)
    out[ECONOMY_FEATURE_INDEX["turns_to_day_end_log"]] = math.log1p(max(0, 23 - hour))
    out[ECONOMY_FEATURE_INDEX["turns_to_game_end_log"]] = math.log1p(max(0, 719 - step))
    own = _mapping(_field(state, "own", {}))
    rival = _mapping(_field(state, "rival", {}))
    private = _mapping(_field(state, "private", {}))
    out[ECONOMY_FEATURE_INDEX["own_money"]] = signed_log1p(own.get("money", 0))
    out[ECONOMY_FEATURE_INDEX["rival_money"]] = signed_log1p(rival.get("money", 0))
    out[ECONOMY_FEATURE_INDEX["own_hires_today"]] = signed_log1p(own.get("hires_today", 0))
    out[ECONOMY_FEATURE_INDEX["rival_hires_today"]] = signed_log1p(rival.get("hires_today", 0))
    out[ECONOMY_FEATURE_INDEX["own_hand_count"]] = signed_log1p(len(own.get("hands") or []))
    out[ECONOMY_FEATURE_INDEX["rival_hand_count"]] = signed_log1p(len(rival.get("hands") or []))
    own_land_count = _count_land(own)
    out[ECONOMY_FEATURE_INDEX["own_land_count"]] = signed_log1p(own_land_count)
    out[ECONOMY_FEATURE_INDEX["rival_land_count"]] = signed_log1p(_count_land(rival))
    out[ECONOMY_FEATURE_INDEX["next_hire_cost"]] = signed_log1p(
        _next_hire_cost(int(own.get("hires_today", 0) or 0))
    )
    land_index = max(0, own_land_count - 1)
    next_land_cost = LAND_PRICES[land_index] if land_index < len(LAND_PRICES) else 0
    out[ECONOMY_FEATURE_INDEX["next_land_cost"]] = signed_log1p(next_land_cost)
    out[ECONOMY_FEATURE_INDEX["land_exhausted"]] = float(land_index >= len(LAND_PRICES))
    shed = _mapping(private.get("shed") or {})
    capacity = int(private.get("shed_capacity", DEFAULT_SHED_CAPACITY) or DEFAULT_SHED_CAPACITY)
    used = sum(max(0, int(v or 0)) for v in shed.values())
    out[ECONOMY_FEATURE_INDEX["shed_capacity"]] = signed_log1p(capacity)
    out[ECONOMY_FEATURE_INDEX["shed_free_space"]] = signed_log1p(max(0, capacity - used))
    out[ECONOMY_FEATURE_INDEX["market_slots_remaining"]] = signed_log1p(10)
    seeds = _mapping(private.get("seeds") or {})
    for item in ITEM_NAMES:
        out[ECONOMY_FEATURE_INDEX[f"shed:{item}"]] = signed_log1p(shed.get(item, 0))
    for crop in CROPS:
        out[ECONOMY_FEATURE_INDEX[f"seed:{crop}"]] = signed_log1p(seeds.get(crop, 0))
    market = _mapping(_field(state, "market", {}))
    market_inventory = _mapping(market.get("inventory") or {})
    market_prices = _mapping(market.get("prices") or {})
    for item in PRODUCTS:
        out[ECONOMY_FEATURE_INDEX[f"market_inventory:{item}"]] = signed_log1p(market_inventory.get(item, 0))
        out[ECONOMY_FEATURE_INDEX[f"market_price:{item}"]] = signed_log1p(market_prices.get(item, 0))
    shops = list(_field(state, "town_shops", []) or [])
    if not shops:
        shops = list(_mapping(_field(state, "town", {})).get("unlocked_shops") or [])
    for shop in SHOP_NAMES:
        out[ECONOMY_FEATURE_INDEX[f"shop_count:{shop}"]] = signed_log1p(shops.count(shop))
    return out


def _effect_tensor(effect: Any) -> torch.Tensor:
    data = _mapping(effect)
    out = torch.zeros(len(EFFECT_FEATURES), dtype=torch.float32)
    out[EFFECT_FEATURE_INDEX["money_delta"]] = signed_log1p(data.get("money_delta", 0))
    out[EFFECT_FEATURE_INDEX["hand_count_delta"]] = signed_log1p(data.get("hand_count_delta", 0))
    out[EFFECT_FEATURE_INDEX["day_changed"]] = float(bool(data.get("day_changed", False)))
    out[EFFECT_FEATURE_INDEX["day_reset"]] = float(bool(data.get("day_reset", False)))
    position_deltas = _mapping(data.get("unit_position_delta") or {})
    move_total = 0.0
    for delta in position_deltas.values():
        if isinstance(delta, (list, tuple)):
            move_total += sum(abs(float(v)) for v in delta[:2])
    out[EFFECT_FEATURE_INDEX["unit_move_l1_total"]] = signed_log1p(move_total)
    for prefix, names in (
        ("shed_delta", ITEM_NAMES), ("seed_delta", CROPS),
        ("market_inventory_delta", PRODUCTS), ("market_price_delta", PRODUCTS),
    ):
        values = _mapping(data.get(prefix) or {})
        for item in names:
            out[EFFECT_FEATURE_INDEX[f"{prefix}:{item}"]] = signed_log1p(values.get(item, 0))
    opponent = _mapping(data.get("opponent_public") or {})
    out[EFFECT_FEATURE_INDEX["rival_money_delta"]] = signed_log1p(opponent.get("money_delta", 0))
    out[EFFECT_FEATURE_INDEX["rival_hand_count_delta"]] = signed_log1p(opponent.get("hand_count_delta", 0))
    return out


def tensorize_state(structured_obs: Any, previous_effect: Any = None,
                    previous_action: Any = None) -> TensorizedState:
    own_grid = _grid_tensor(_field(structured_obs, "own_grid", []))
    rival_grid = _grid_tensor(_field(structured_obs, "rival_grid", []))
    if own_grid.shape[:2] != rival_grid.shape[:2]:
        raise ValueError("own/rival farm grid shapes differ")
    board_size = int(own_grid.shape[0])
    own_units = _units_tensor(
        _field(structured_obs, "own_units", []), board_size=board_size, own_private=True
    )
    rival_units = _units_tensor(
        _field(structured_obs, "rival_units", []), board_size=board_size, own_private=False
    )
    previous_effect = previous_effect or {}
    previous_action = previous_action or {}
    return TensorizedState(
        own_grid=own_grid,
        rival_grid=rival_grid,
        own_units=own_units,
        rival_units=rival_units,
        previous_unit_actions=_previous_unit_actions_tensor(
            structured_obs, previous_action, previous_effect
        ),
        previous_unit_effects=_previous_unit_effects_tensor(
            structured_obs, previous_effect
        ),
        commodities=_commodity_tensor(structured_obs),
        economy=_economy_tensor(structured_obs),
        previous_action_global=_previous_action_global_tensor(previous_action),
        previous_effect=_effect_tensor(previous_effect),
    )


def _decode_json_field(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _row_state(row: dict[str, Any]) -> Any:
    if "state" in row:
        return row["state"]
    payload = row.get("state_zlib")
    if payload is None:
        raise KeyError("transition row has no state/state_zlib")
    from .v2_dataset import decode_zlib_json
    return decode_zlib_json(payload)


def _row_value(row: dict[str, Any], direct: str, encoded: str, default):
    if direct in row:
        return row[direct]
    if encoded in row and row[encoded] is not None:
        return _decode_json_field(row[encoded])
    return default


def _transition_key(row: dict[str, Any]) -> tuple[int, int, int] | None:
    try:
        return int(row["episode_id"]), int(row["seat"]), int(row["step"])
    except (KeyError, TypeError, ValueError):
        return None


def _derive_previous_context(rows: list[dict[str, Any]]) -> list[tuple[Any, Any]]:
    lookup: dict[tuple[int, int, int], dict[str, Any]] = {}
    for row in rows:
        key = _transition_key(row)
        if key is not None:
            if key in lookup:
                raise ValueError(f"duplicate transition key in batch: {key}")
            lookup[key] = row
    contexts: list[tuple[Any, Any]] = []
    for row in rows:
        explicit_action = "previous_action" in row or "previous_action_json" in row
        explicit_effect = "previous_effect" in row or "previous_effect_json" in row
        previous_action = _row_value(row, "previous_action", "previous_action_json", {}) if explicit_action else None
        previous_effect = _row_value(row, "previous_effect", "previous_effect_json", {}) if explicit_effect else None
        key = _transition_key(row)
        if previous_action is None or previous_effect is None:
            if key is None:
                derived_action, derived_effect = {}, {}
            elif key[2] == 0:
                derived_action, derived_effect = {}, {}
            else:
                predecessor = lookup.get((key[0], key[1], key[2] - 1))
                if predecessor is None:
                    raise ValueError(
                        f"missing previous transition context for episode={key[0]} seat={key[1]} step={key[2]}"
                    )
                derived_action = _row_value(predecessor, "canonical_action", "canonical_action_json", {})
                derived_effect = _row_value(predecessor, "effects", "effects_json", {})
            if previous_action is None:
                previous_action = derived_action
            if previous_effect is None:
                previous_effect = derived_effect
        contexts.append((previous_action or {}, previous_effect or {}))
    return contexts


def _pad_entities(states: list[TensorizedState], attr: str, features: int) -> tuple[torch.Tensor, torch.Tensor]:
    counts = [int(getattr(state, attr).shape[0]) for state in states]
    width = max(counts, default=0)
    values = torch.zeros((len(states), width, int(features)), dtype=torch.float32)
    mask = torch.zeros((len(states), width), dtype=torch.bool)
    for i, state in enumerate(states):
        current = getattr(state, attr)
        n = int(current.shape[0])
        if n:
            values[i, :n] = current
            mask[i, :n] = True
    return values, mask


def _pad_units(states: list[TensorizedState], attr: str) -> tuple[torch.Tensor, torch.Tensor]:
    return _pad_entities(states, attr, len(UNIT_FEATURES))


def _effective_supervision_action(structured: Any, action: Any) -> dict[str, Any]:
    result = deepcopy(action if isinstance(action, dict) else {})
    result.setdefault("farmer", {"op": "PASS", "item": None, "quantity": None, "raw": ["PASS"]})
    result.setdefault("hands", [])
    result.setdefault("market", [])
    ledger = ShadowLedger.from_state(structured)

    unit_commands = [result.get("farmer") or {}, *list(result.get("hands") or [])]
    plant_demand: dict[str, int] = {}
    for command in unit_commands:
        if str((command or {}).get("op", "PASS")) == "PLANT":
            item = (command or {}).get("item")
            if item is not None:
                name = str(item)
                plant_demand[name] = plant_demand.get(name, 0) + 1
    ledger.set_atomic_plant_blocked(
        crop for crop, count in plant_demand.items()
        if count > int(ledger.seeds.get(crop, 0) or 0)
    )

    def apply_unit_if_in_bounds(actor: str, command: Any) -> None:
        pos = ledger.unit_positions.get(actor)
        if pos is None or len(pos) < 2:
            return
        x, y = int(pos[0]), int(pos[1])
        if not (0 <= x < ledger.board_size and 0 <= y < ledger.board_size):
            return
        ledger.apply_unit(actor, command or {})

    apply_unit_if_in_bounds("farmer", result.get("farmer") or {})
    for index, command in enumerate(result.get("hands") or []):
        apply_unit_if_in_bounds(f"hand:{index}", command)

    sanitized_market: list[dict[str, Any]] = []
    for slot, source in enumerate(result.get("market") or []):
        order = deepcopy(source if isinstance(source, dict) else {})
        kind = str(order.get("kind", "ORDER"))
        if kind == "STOP_QUEUE":
            sanitized_market.append(order)
            ledger.apply_market(order)
            break
        if kind == "NOP_SLOT":
            sanitized_market.append(order)
            ledger.apply_market(order)
            continue

        op = str(order.get("op", "NOP_SLOT"))
        if op == "SELL":
            item = str(order.get("item")) if order.get("item") is not None else None
            try:
                requested = int(order.get("quantity") or 0)
            except (TypeError, ValueError):
                requested = 0
            legal = ledger.legal_market_mask(slot, {})
            bound = int(
                (legal.metadata.get("sell_max_by_item") or {}).get(str(item), 0)
            )
            if requested <= 0 or not legal.allows("SELL", item) or bound <= 0:
                order = {
                    "kind": "NOP_SLOT", "op": None,
                    "item": None, "quantity": None, "raw": [],
                }
            else:
                effective = min(requested, bound)
                order["quantity"] = effective
                order["raw"] = ["SELL", item, effective]

        sanitized_market.append(order)
        ledger.apply_market(order)

    result["market"] = sanitized_market
    return result


def collate_transitions(rows: list[dict[str, Any]]) -> V2Batch:
    if not rows:
        raise ValueError("cannot collate an empty transition list")
    states: list[TensorizedState] = []
    structured_states = []
    canonical_actions = []
    previous_actions = []
    effects_targets = []
    contexts = _derive_previous_context(rows)
    for row, (previous_action, previous_effect) in zip(rows, contexts):
        structured = _row_state(row)
        structured_states.append(structured)
        states.append(tensorize_state(structured, previous_effect, previous_action))
        source_action = _row_value(row, "canonical_action", "canonical_action_json", {})
        canonical_actions.append(_effective_supervision_action(structured, source_action))
        previous_actions.append(previous_action)
        effects_targets.append(_row_value(row, "effects", "effects_json", {}))
    own_shapes = {tuple(state.own_grid.shape) for state in states}
    rival_shapes = {tuple(state.rival_grid.shape) for state in states}
    if len(own_shapes) != 1 or len(rival_shapes) != 1:
        raise ValueError("all grid tensors in a batch must share one shape")
    own_units, own_mask = _pad_units(states, "own_units")
    rival_units, rival_mask = _pad_units(states, "rival_units")
    previous_unit_actions, previous_unit_mask = _pad_entities(
        states, "previous_unit_actions", len(PREV_UNIT_ACTION_FEATURES)
    )
    if not torch.equal(previous_unit_mask, own_mask):
        raise ValueError("previous-action unit mask does not match current own-unit mask")
    previous_unit_effects, previous_effect_mask = _pad_entities(
        states, "previous_unit_effects", len(PREV_UNIT_EFFECT_FEATURES)
    )
    if not torch.equal(previous_effect_mask, own_mask):
        raise ValueError("previous-effect unit mask does not match current own-unit mask")
    hand_counts = [len(_mapping(action).get("hands") or []) for action in canonical_actions]
    expected_hands = [max(0, int(state.own_units.shape[0]) - 1) for state in states]
    if hand_counts != expected_hands:
        raise ValueError(f"action/state hand count mismatch: actions={hand_counts} states={expected_hands}")
    batch_hand_width = max(hand_counts, default=0)
    hand_action_mask = torch.zeros((len(states), batch_hand_width), dtype=torch.bool)
    market_counts = [len(_mapping(action).get("market") or []) for action in canonical_actions]
    max_market = max(market_counts, default=0)
    market_slot_mask = torch.zeros((len(states), max_market), dtype=torch.bool)
    for i, (hands, market) in enumerate(zip(hand_counts, market_counts)):
        hand_action_mask[i, :hands] = True
        market_slot_mask[i, :market] = True
    return V2Batch(
        own_grid=torch.stack([state.own_grid for state in states]),
        rival_grid=torch.stack([state.rival_grid for state in states]),
        own_units=own_units,
        own_unit_mask=own_mask,
        rival_units=rival_units,
        rival_unit_mask=rival_mask,
        previous_unit_actions=previous_unit_actions,
        previous_unit_effects=previous_unit_effects,
        commodities=torch.stack([state.commodities for state in states]),
        hand_action_mask=hand_action_mask,
        market_slot_mask=market_slot_mask,
        economy=torch.stack([state.economy for state in states]),
        previous_action_global=torch.stack([state.previous_action_global for state in states]),
        previous_effect=torch.stack([state.previous_effect for state in states]),
        structured_states=tuple(structured_states),
        canonical_actions=tuple(canonical_actions),
        previous_actions=tuple(previous_actions),
        effects_targets=tuple(effects_targets),
    )
