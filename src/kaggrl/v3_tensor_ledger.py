from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import torch

from .constants import ANIMALS, CROPS, ITEM_TO_ID, PRODUCTS, UNIT_OPS
from .v2_ledger import (
    ANIMAL_COST,
    ANIMAL_STRUCTURE,
    CROP_FIRST_YIELD,
    CROP_ONGOING,
    LAND_PRICES,
    MARKET_OPS,
    SEED_COST,
)
from .v3_2_schema import ACTIVE_MARKET_OPS
from .v2_tensorize import (
    COMMODITY_FEATURE_INDEX,
    ECONOMY_FEATURE_INDEX,
    TILE_FEATURE_INDEX,
    TILE_KINDS,
    UNIT_FEATURE_INDEX,
)

ITEM_NAMES = tuple(ITEM_TO_ID)
ITEM_CLASSES = max(ITEM_TO_ID.values()) + 1
UNIT_OP_TO_ID = {op: i for i, op in enumerate(UNIT_OPS)}
MARKET_OP_TO_ID = {op: i for i, op in enumerate(MARKET_OPS)}
ACTIVE_MARKET_OP_TO_ID = {op: i for i, op in enumerate(ACTIVE_MARKET_OPS)}
TILE_KIND_TO_ID = {
    "EMPTY": 0, "LOCKED": 1, "WEED": 2, "PLANT": 3,
    "COOP": 4, "PASTURE": 5, "OTHER": 6,
}
CROP_IDS = torch.tensor([ITEM_TO_ID[name] for name in CROPS], dtype=torch.long)
ANIMAL_IDS = torch.tensor([ITEM_TO_ID[name] for name in ANIMALS], dtype=torch.long)
PRODUCT_IDS = torch.tensor([ITEM_TO_ID[name] for name in PRODUCTS], dtype=torch.long)
SEED_COST_BY_ITEM = torch.zeros(ITEM_CLASSES, dtype=torch.long)
ANIMAL_COST_BY_ITEM = torch.zeros(ITEM_CLASSES, dtype=torch.long)
CROP_FIRST_YIELD_BY_ITEM = torch.full((ITEM_CLASSES,), 10**9, dtype=torch.long)
CROP_ONGOING_BY_ITEM = torch.zeros(ITEM_CLASSES, dtype=torch.bool)
ANIMAL_STRUCTURE_KIND_BY_ITEM = torch.zeros(ITEM_CLASSES, dtype=torch.long)
ANIMAL_PRODUCT_BY_ITEM = torch.zeros(ITEM_CLASSES, dtype=torch.long)
for _name, _cost in SEED_COST.items():
    SEED_COST_BY_ITEM[ITEM_TO_ID[_name]] = int(_cost)
for _name, _cost in ANIMAL_COST.items():
    ANIMAL_COST_BY_ITEM[ITEM_TO_ID[_name]] = int(_cost)
for _name, _value in CROP_FIRST_YIELD.items():
    CROP_FIRST_YIELD_BY_ITEM[ITEM_TO_ID[_name]] = int(_value)
for _name, _value in CROP_ONGOING.items():
    CROP_ONGOING_BY_ITEM[ITEM_TO_ID[_name]] = bool(_value)
for _name, _kind in ANIMAL_STRUCTURE.items():
    ANIMAL_STRUCTURE_KIND_BY_ITEM[ITEM_TO_ID[_name]] = TILE_KIND_TO_ID[_kind]
for _animal, _product in {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}.items():
    ANIMAL_PRODUCT_BY_ITEM[ITEM_TO_ID[_animal]] = ITEM_TO_ID[_product]

_WHEAT = ITEM_TO_ID["WHEAT"]
_FERTILIZER = ITEM_TO_ID["FERTILIZER"]
def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "__dict__"):
        return vars(value)
    return {}


def _field(value: Any, name: str, default=None):
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _tile_kind(tile: Any) -> int:
    if tile is None:
        return TILE_KIND_TO_ID["EMPTY"]
    if isinstance(tile, str):
        return TILE_KIND_TO_ID.get(tile, TILE_KIND_TO_ID["OTHER"])
    if isinstance(tile, dict):
        return TILE_KIND_TO_ID.get(str(tile.get("kind", "OTHER")), TILE_KIND_TO_ID["OTHER"])
    return TILE_KIND_TO_ID["OTHER"]


def _signed_log1p_tensor(value: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    value = value.to(dtype)
    return torch.sign(value) * torch.log1p(value.abs())


def _inverse_signed_log1p_long(value: torch.Tensor) -> torch.Tensor:
    decoded = torch.sign(value) * torch.expm1(value.abs())
    return torch.round(decoded).to(torch.long)


def _fib_table(size: int = 64) -> torch.Tensor:
    values = []
    a, b = 1, 1
    for _ in range(size):
        values.append(a)
        a, b = b, a + b
    return torch.tensor(values, dtype=torch.long)


HIRE_COST_TABLE = _fib_table()
@dataclass
class TensorUnitLegal:
    op_mask: torch.Tensor
    item_mask: torch.Tensor
    quantity_max: torch.Tensor


@dataclass
class TensorMarketLegal:
    op_mask: torch.Tensor
    item_mask: torch.Tensor
    quantity_max: torch.Tensor


@dataclass
class TensorLedger:
    step: torch.Tensor
    day: torch.Tensor
    cash: torch.Tensor
    cash_uncertain: torch.Tensor
    hires_today: torch.Tensor
    hire_count_uncertain: torch.Tensor
    land_count: torch.Tensor
    land_count_uncertain: torch.Tensor
    shed: torch.Tensor
    shed_reserved: torch.Tensor
    shed_uncertain: torch.Tensor
    seeds: torch.Tensor
    plant_demand: torch.Tensor
    atomic_plant_blocked: torch.Tensor
    positions: torch.Tensor
    unit_mask: torch.Tensor
    inventory: torch.Tensor
    inventory_order: torch.Tensor
    grid_kind: torch.Tensor
    grid_is_dict: torch.Tensor
    grid_crop: torch.Tensor
    grid_animal: torch.Tensor
    grid_watered: torch.Tensor
    grid_fed: torch.Tensor
    grid_cared: torch.Tensor
    grid_fertilizer_available: torch.Tensor
    grid_yield: torch.Tensor
    grid_planted_day: torch.Tensor
    grid_fertilized_until_day: torch.Tensor
    market_prices: torch.Tensor
    market_slots_used: torch.Tensor
    market_stopped: torch.Tensor
    shed_capacity: int = 100

    @property
    def device(self) -> torch.device:
        return self.cash.device

    @property
    def batch_size(self) -> int:
        return int(self.cash.shape[0])

    @property
    def max_units(self) -> int:
        return int(self.positions.shape[1])

    @property
    def board_size(self) -> int:
        return int(self.grid_kind.shape[1])

    def clone(self) -> "TensorLedger":
        values = {}
        for field in fields(self):
            value = getattr(self, field.name)
            values[field.name] = value.clone() if torch.is_tensor(value) else value
        return TensorLedger(**values)
    def to(self, device: torch.device | str) -> "TensorLedger":
        device = torch.device(device)
        values = {}
        for field in fields(self):
            value = getattr(self, field.name)
            values[field.name] = value.to(device) if torch.is_tensor(value) else value
        return TensorLedger(**values)

    def index_select(self, index: torch.Tensor) -> "TensorLedger":
        index = index.to(device=self.device, dtype=torch.long)
        values = {}
        for field in fields(self):
            value = getattr(self, field.name)
            values[field.name] = (
                value.index_select(0, index) if torch.is_tensor(value) else value
            )
        return TensorLedger(**values)

    @classmethod
    def from_batch(
        cls,
        batch,
        *,
        structured_states=None,
        device: torch.device | str | None = None,
    ) -> "TensorLedger":
        """Build a ledger from tensors already produced by V2 collation.

        Python structured states are consulted only to preserve inventory key
        order, which can matter for DROP when shed capacity is tight.
        """
        source_device = batch.own_grid.device
        target_device = (
            source_device if device is None else torch.device(device)
        )
        own_grid = batch.own_grid
        own_units = batch.own_units
        unit_mask = batch.own_unit_mask.to(torch.bool)
        commodities = batch.commodities
        economy = batch.economy

        b, board, _, _ = own_grid.shape
        max_units = int(own_units.shape[1])

        def econ(name: str) -> torch.Tensor:
            return economy[:, ECONOMY_FEATURE_INDEX[name]]

        step = _inverse_signed_log1p_long(econ("step_log"))
        day = _inverse_signed_log1p_long(econ("day_log"))
        cash = _inverse_signed_log1p_long(econ("own_money"))
        hires_today = _inverse_signed_log1p_long(
            econ("own_hires_today")
        )
        land_count = _inverse_signed_log1p_long(
            econ("own_land_count")
        )

        shed = torch.zeros(
            (b, ITEM_CLASSES), dtype=torch.long, device=source_device,
        )
        seeds = torch.zeros_like(shed)
        market_prices = torch.zeros_like(shed)
        for commodity_row, name in enumerate(ITEM_NAMES):
            item_id = ITEM_TO_ID[name]
            shed[:, item_id] = _inverse_signed_log1p_long(
                commodities[
                    :, commodity_row,
                    COMMODITY_FEATURE_INDEX["shed_quantity"],
                ]
            )
            seeds[:, item_id] = _inverse_signed_log1p_long(
                commodities[
                    :, commodity_row,
                    COMMODITY_FEATURE_INDEX["seed_quantity"],
                ]
            )
            market_prices[:, item_id] = _inverse_signed_log1p_long(
                commodities[
                    :, commodity_row,
                    COMMODITY_FEATURE_INDEX["market_price"],
                ]
            )

        denom = max(1, board - 1)
        positions = torch.zeros(
            (b, max_units, 2), dtype=torch.long, device=source_device,
        )
        positions[:, :, 0] = torch.round(
            own_units[:, :, UNIT_FEATURE_INDEX["x"]] * float(denom)
        ).to(torch.long)
        positions[:, :, 1] = torch.round(
            own_units[:, :, UNIT_FEATURE_INDEX["y"]] * float(denom)
        ).to(torch.long)
        positions = torch.where(
            unit_mask.unsqueeze(-1),
            positions,
            torch.zeros_like(positions),
        )

        inventory = torch.zeros(
            (b, max_units, ITEM_CLASSES),
            dtype=torch.long,
            device=source_device,
        )
        for name in ITEM_NAMES:
            item_id = ITEM_TO_ID[name]
            inventory[:, :, item_id] = _inverse_signed_log1p_long(
                own_units[
                    :, :, UNIT_FEATURE_INDEX[f"inventory:{name}"]
                ]
            )
        inventory = torch.where(
            unit_mask.unsqueeze(-1),
            inventory,
            torch.zeros_like(inventory),
        )

        inventory_order = torch.zeros_like(inventory)
        states = (
            tuple(structured_states)
            if structured_states is not None
            else tuple(getattr(batch, "structured_states", ()) or ())
        )
        if states:
            if len(states) != b:
                raise ValueError(
                    "structured state count does not match tensor batch"
                )
            order_rows = []
            for state in states:
                units = list(_field(state, "own_units", []) or [])
                row_order = torch.zeros(
                    (max_units, ITEM_CLASSES), dtype=torch.long,
                )
                for actor_index, unit in enumerate(units[:max_units]):
                    inv = _mapping(_mapping(unit).get("inventory") or {})
                    slot = 0
                    for name, amount in inv.items():
                        item_id = ITEM_TO_ID.get(str(name))
                        if (
                            item_id is not None
                            and int(amount or 0) > 0
                            and slot < ITEM_CLASSES
                        ):
                            row_order[actor_index, slot] = int(item_id)
                            slot += 1
                order_rows.append(row_order)
            inventory_order = torch.stack(order_rows).to(source_device)
        else:
            item_ids = torch.arange(
                ITEM_CLASSES, device=source_device, dtype=torch.long,
            ).view(1, 1, -1)
            present = inventory.gt(0) & item_ids.gt(0)
            inventory_order = torch.where(
                present,
                item_ids.expand_as(inventory),
                torch.zeros_like(inventory),
            )

        kind_scores = torch.stack(
            [
                own_grid[:, :, :, TILE_FEATURE_INDEX[f"kind:{name}"]]
                for name in TILE_KINDS
            ],
            dim=-1,
        )
        grid_kind = kind_scores.argmax(dim=-1).to(torch.long)
        grid_is_dict = (
            grid_kind.ne(TILE_KIND_TO_ID["EMPTY"])
            & grid_kind.ne(TILE_KIND_TO_ID["LOCKED"])
        )

        grid_crop = torch.zeros(
            (b, board, board), dtype=torch.long, device=source_device,
        )
        for name in CROPS:
            grid_crop = torch.where(
                own_grid[
                    :, :, :, TILE_FEATURE_INDEX[f"crop:{name}"]
                ].gt(0.5),
                torch.full_like(grid_crop, ITEM_TO_ID[name]),
                grid_crop,
            )
        grid_animal = torch.zeros_like(grid_crop)
        for name in ANIMALS:
            grid_animal = torch.where(
                own_grid[
                    :, :, :, TILE_FEATURE_INDEX[f"animal:{name}"]
                ].gt(0.5),
                torch.full_like(grid_animal, ITEM_TO_ID[name]),
                grid_animal,
            )

        def tile_bool(name: str) -> torch.Tensor:
            return own_grid[
                :, :, :, TILE_FEATURE_INDEX[name]
            ].gt(0.5)

        def tile_long(name: str) -> torch.Tensor:
            return _inverse_signed_log1p_long(
                own_grid[:, :, :, TILE_FEATURE_INDEX[name]]
            )

        grid_yield = tile_long("yield_units")
        grid_planted_day = tile_long("planted_day")
        fertilized_raw = own_grid[
            :, :, :, TILE_FEATURE_INDEX["fertilized_until_day"]
        ]
        grid_fertilized_until_day = _inverse_signed_log1p_long(
            fertilized_raw
        )
        grid_fertilized_until_day = torch.where(
            grid_is_dict & fertilized_raw.ne(0),
            grid_fertilized_until_day,
            torch.full_like(grid_fertilized_until_day, -1),
        )

        zeros_b = torch.zeros(
            b, dtype=torch.bool, device=source_device,
        )
        zeros_l = torch.zeros(
            b, dtype=torch.long, device=source_device,
        )
        zeros_item_l = torch.zeros_like(shed)
        zeros_item_b = torch.zeros(
            shed.shape, dtype=torch.bool, device=source_device,
        )

        ledger = cls(
            step=step,
            day=day,
            cash=cash,
            cash_uncertain=zeros_b.clone(),
            hires_today=hires_today,
            hire_count_uncertain=zeros_b.clone(),
            land_count=land_count,
            land_count_uncertain=zeros_b.clone(),
            shed=shed,
            shed_reserved=zeros_l.clone(),
            shed_uncertain=zeros_b.clone(),
            seeds=seeds,
            plant_demand=zeros_item_l.clone(),
            atomic_plant_blocked=zeros_item_b,
            positions=positions,
            unit_mask=unit_mask,
            inventory=inventory,
            inventory_order=inventory_order,
            grid_kind=grid_kind,
            grid_is_dict=grid_is_dict,
            grid_crop=grid_crop,
            grid_animal=grid_animal,
            grid_watered=tile_bool("watered_today"),
            grid_fed=tile_bool("fed_today"),
            grid_cared=tile_bool("cared_today"),
            grid_fertilizer_available=tile_bool(
                "fertilizer_available"
            ),
            grid_yield=grid_yield,
            grid_planted_day=grid_planted_day,
            grid_fertilized_until_day=grid_fertilized_until_day,
            market_prices=market_prices,
            market_slots_used=zeros_l.clone(),
            market_stopped=zeros_b.clone(),
        )
        return ledger.to(target_device)

    @classmethod
    def from_states(cls, states, *, device: torch.device | str = "cpu") -> "TensorLedger":
        states = tuple(states)
        if not states:
            raise ValueError("TensorLedger requires at least one state")
        batch = len(states)
        units_per_row = []
        board_sizes = []
        for state in states:
            units = list(_field(state, "own_units", []) or [])
            if not units:
                own = _mapping(_field(state, "own", {}))
                units = [{"kind": "farmer", "index": 0, "position": own.get("farmer", [0, 0])}]
                units += [
                    {"kind": "hand", "index": i, "position": p}
                    for i, p in enumerate(own.get("hands") or [])
                ]
            units_per_row.append(units)
            grid = list(_field(state, "own_grid", []) or [])
            board_sizes.append(len(grid) or 10)
        if len(set(board_sizes)) != 1:
            raise ValueError("TensorLedger batch must share one board size")
        board = board_sizes[0]
        max_units = max(len(units) for units in units_per_row)
        zlong = lambda *shape: torch.zeros(shape, dtype=torch.long)
        zbool = lambda *shape: torch.zeros(shape, dtype=torch.bool)
        step = zlong(batch)
        day = zlong(batch)
        cash = zlong(batch)
        cash_uncertain = zbool(batch)
        hires_today = zlong(batch)
        hire_count_uncertain = zbool(batch)
        land_count = zlong(batch)
        land_count_uncertain = zbool(batch)
        shed = zlong(batch, ITEM_CLASSES)
        shed_reserved = zlong(batch)
        shed_uncertain = zbool(batch)
        seeds = zlong(batch, ITEM_CLASSES)
        plant_demand = zlong(batch, ITEM_CLASSES)
        atomic_plant_blocked = zbool(batch, ITEM_CLASSES)
        positions = zlong(batch, max_units, 2)
        unit_mask = zbool(batch, max_units)
        inventory = zlong(batch, max_units, ITEM_CLASSES)
        inventory_order = zlong(batch, max_units, ITEM_CLASSES)
        grid_kind = zlong(batch, board, board)
        grid_is_dict = zbool(batch, board, board)
        grid_crop = zlong(batch, board, board)
        grid_animal = zlong(batch, board, board)
        grid_watered = zbool(batch, board, board)
        grid_fed = zbool(batch, board, board)
        grid_cared = zbool(batch, board, board)
        grid_fertilizer_available = zbool(batch, board, board)
        grid_yield = zlong(batch, board, board)
        grid_planted_day = zlong(batch, board, board)
        grid_fertilized_until_day = torch.full((batch, board, board), -1, dtype=torch.long)
        market_prices = zlong(batch, ITEM_CLASSES)
        market_slots_used = zlong(batch)
        market_stopped = zbool(batch)
        for row, (state, units) in enumerate(zip(states, units_per_row)):
            own = _mapping(_field(state, "own", {}))
            private = _mapping(_field(state, "private", {}))
            step[row] = int(_field(state, "step", 0) or 0)
            day[row] = int(_field(state, "day", 0) or 0)
            cash[row] = int(own.get("money", 0) or 0)
            hires_today[row] = int(own.get("hires_today", 0) or 0)
            land_count[row] = len(own.get("unlocked_quadrants") or [])
            for name, amount in _mapping(private.get("shed") or {}).items():
                item_id = ITEM_TO_ID.get(str(name))
                if item_id is not None:
                    shed[row, item_id] = int(amount or 0)
            for name, amount in _mapping(private.get("seeds") or {}).items():
                item_id = ITEM_TO_ID.get(str(name))
                if item_id is not None:
                    seeds[row, item_id] = int(amount or 0)

            inventories = list(private.get("inventories") or [])
            for actor_index, unit in enumerate(units):
                data = _mapping(unit)
                pos = list(data.get("position") or [0, 0])
                positions[row, actor_index, 0] = int(pos[0])
                positions[row, actor_index, 1] = int(pos[1])
                unit_mask[row, actor_index] = True
                inv = data.get("inventory")
                if not isinstance(inv, dict) and actor_index < len(inventories):
                    inv = inventories[actor_index]
                inv = _mapping(inv or {})
                order_slot = 0
                for name, amount in inv.items():
                    item_id = ITEM_TO_ID.get(str(name))
                    if item_id is None:
                        continue
                    inventory[row, actor_index, item_id] = int(amount or 0)
                    if int(amount or 0) > 0 and order_slot < ITEM_CLASSES:
                        inventory_order[row, actor_index, order_slot] = item_id
                        order_slot += 1
            grid = list(_field(state, "own_grid", []) or [])
            if not grid:
                grid = [[None for _ in range(board)] for _ in range(board)]
            for y, grid_row in enumerate(grid):
                for x, tile in enumerate(grid_row):
                    grid_kind[row, y, x] = _tile_kind(tile)
                    if isinstance(tile, dict):
                        grid_is_dict[row, y, x] = True
                        crop = tile.get("crop")
                        animal = tile.get("animal")
                        if crop in ITEM_TO_ID:
                            grid_crop[row, y, x] = ITEM_TO_ID[crop]
                        if animal in ITEM_TO_ID:
                            grid_animal[row, y, x] = ITEM_TO_ID[animal]
                        grid_watered[row, y, x] = bool(tile.get("watered_today", False))
                        grid_fed[row, y, x] = bool(tile.get("fed_today", False))
                        grid_cared[row, y, x] = bool(tile.get("cared_today", False))
                        grid_fertilizer_available[row, y, x] = bool(
                            tile.get("fertilizer_available", False)
                        )
                        grid_yield[row, y, x] = int(tile.get("yield_units", 0) or 0)
                        grid_planted_day[row, y, x] = int(tile.get("planted_day", 0) or 0)
                        grid_fertilized_until_day[row, y, x] = int(
                            tile.get("fertilized_until_day", -1) or -1
                        )
            market = _mapping(_field(state, "market", {}))
            for name, price in _mapping(market.get("prices") or {}).items():
                item_id = ITEM_TO_ID.get(str(name))
                if item_id is not None:
                    market_prices[row, item_id] = int(price or 0)
        ledger = cls(
            step=step, day=day, cash=cash, cash_uncertain=cash_uncertain,
            hires_today=hires_today, hire_count_uncertain=hire_count_uncertain,
            land_count=land_count, land_count_uncertain=land_count_uncertain,
            shed=shed, shed_reserved=shed_reserved, shed_uncertain=shed_uncertain,
            seeds=seeds, plant_demand=plant_demand,
            atomic_plant_blocked=atomic_plant_blocked,
            positions=positions, unit_mask=unit_mask, inventory=inventory,
            inventory_order=inventory_order,
            grid_kind=grid_kind, grid_is_dict=grid_is_dict,
            grid_crop=grid_crop, grid_animal=grid_animal,
            grid_watered=grid_watered, grid_fed=grid_fed,
            grid_cared=grid_cared,
            grid_fertilizer_available=grid_fertilizer_available,
            grid_yield=grid_yield, grid_planted_day=grid_planted_day,
            grid_fertilized_until_day=grid_fertilized_until_day,
            market_prices=market_prices, market_slots_used=market_slots_used,
            market_stopped=market_stopped,
        )
        return ledger.to(device)

    def _constant(self, value: torch.Tensor) -> torch.Tensor:
        return value.to(self.device)

    def next_hire_cost(self) -> torch.Tensor:
        table = self._constant(HIRE_COST_TABLE)
        index = self.hires_today.clamp(min=0, max=table.numel() - 1)
        return table[index]

    def next_land_cost(self) -> tuple[torch.Tensor, torch.Tensor]:
        prices = self.cash.new_tensor(LAND_PRICES)
        extra = (self.land_count - 1).clamp_min(0)
        exhausted = extra >= prices.numel()
        cost = prices[extra.clamp(max=prices.numel() - 1)]
        return torch.where(exhausted, torch.zeros_like(cost), cost), exhausted
    def shed_room(self) -> torch.Tensor:
        known = self.shed.clamp_min(0).sum(dim=-1)
        return (int(self.shed_capacity) - known - self.shed_reserved.clamp_min(0)).clamp_min(0)

    def ledger_vector(self, ref: torch.Tensor) -> torch.Tensor:
        dtype = ref.dtype
        land_cost, exhausted = self.next_land_cost()
        values = (
            _signed_log1p_tensor(self.cash, dtype),
            self.cash_uncertain.to(dtype),
            _signed_log1p_tensor(self.hires_today, dtype),
            _signed_log1p_tensor(self.next_hire_cost(), dtype),
            _signed_log1p_tensor(land_cost, dtype),
            exhausted.to(dtype),
            _signed_log1p_tensor(self.shed.clamp_min(0).sum(dim=-1), dtype),
            _signed_log1p_tensor(self.shed_room(), dtype),
            self.shed_uncertain.to(dtype),
            _signed_log1p_tensor(self.plant_demand.clamp_min(0).sum(dim=-1), dtype),
            self.market_slots_used.to(dtype) / 10.0,
            self.market_stopped.to(dtype),
        )
        return torch.stack(values, dim=-1).to(device=ref.device)

    def _tile_values(self, actor_index: int):
        pos = self.positions[:, actor_index]
        x = pos[:, 0].clamp(0, self.board_size - 1)
        y = pos[:, 1].clamp(0, self.board_size - 1)
        rows = torch.arange(self.batch_size, device=self.device)
        return rows, x, y

    def _inventory_add(self, actor_index: int, item: torch.Tensor,
                       amount: torch.Tensor, active: torch.Tensor) -> None:
        rows = torch.arange(self.batch_size, device=self.device)
        item = item.clamp(0, ITEM_CLASSES - 1)
        old = self.inventory[rows, actor_index, item]
        add = torch.where(active, amount.clamp_min(0), torch.zeros_like(amount))
        self.inventory[rows, actor_index, item] = old + add
        new_key = active & (add > 0) & (old <= 0) & (item > 0)
        order = self.inventory_order[:, actor_index]
        empty = order.eq(0)
        first = empty.to(torch.int64).argmax(dim=1)
        has_empty = empty.any(dim=1)
        write = new_key & has_empty
        order[rows[write], first[write]] = item[write]
    def _compact_inventory_order(self, actor_index: int) -> None:
        order = self.inventory_order[:, actor_index]
        positions = torch.arange(order.shape[1], device=self.device).unsqueeze(0)
        key = positions + order.eq(0).to(torch.long) * order.shape[1]
        indices = key.argsort(dim=1)
        self.inventory_order[:, actor_index] = order.gather(1, indices)

    def _inventory_sub(self, actor_index: int, item: torch.Tensor,
                       amount: torch.Tensor, active: torch.Tensor) -> None:
        rows = torch.arange(self.batch_size, device=self.device)
        item = item.clamp(0, ITEM_CLASSES - 1)
        old = self.inventory[rows, actor_index, item]
        sub = torch.where(active, amount.clamp_min(0), torch.zeros_like(amount))
        new = (old - sub).clamp_min(0)
        self.inventory[rows, actor_index, item] = new
        remove = active & (old > 0) & (new <= 0) & (item > 0)
        order = self.inventory_order[:, actor_index]
        order[remove] = torch.where(
            order[remove].eq(item[remove].unsqueeze(1)),
            torch.zeros_like(order[remove]),
            order[remove],
        )
        self._compact_inventory_order(actor_index)

    def set_atomic_plant_blocked(self, blocked: torch.Tensor) -> None:
        if blocked.shape != self.atomic_plant_blocked.shape:
            raise ValueError("atomic plant blocked tensor shape mismatch")
        self.atomic_plant_blocked = blocked.to(
            device=self.device, dtype=torch.bool,
        )
    def legal_unit(self, actor_index: int) -> TensorUnitLegal:
        if actor_index < 0 or actor_index >= self.max_units:
            raise IndexError(actor_index)
        b = self.batch_size
        ops = torch.zeros((b, len(UNIT_OPS)), dtype=torch.bool, device=self.device)
        items = torch.zeros(
            (b, len(UNIT_OPS), ITEM_CLASSES), dtype=torch.bool, device=self.device
        )
        qmax = torch.full(
            (b, len(UNIT_OPS), ITEM_CLASSES), -1, dtype=torch.long, device=self.device
        )
        active = self.unit_mask[:, actor_index]
        rows, x, y = self._tile_values(actor_index)
        kind = self.grid_kind[rows, y, x]
        is_dict = self.grid_is_dict[rows, y, x]
        crop = self.grid_crop[rows, y, x]
        animal = self.grid_animal[rows, y, x]
        inv = self.inventory[:, actor_index]
        half = self.board_size // 2
        adjacent = active & (x >= half - 1) & (x <= half) & (y >= half - 1) & (y <= half)

        ops[:, UNIT_OP_TO_ID["PASS"]] = active
        ops[:, UNIT_OP_TO_ID["NORTH"]] = active & (y > 0)
        ops[:, UNIT_OP_TO_ID["SOUTH"]] = active & (y + 1 < self.board_size)
        ops[:, UNIT_OP_TO_ID["EAST"]] = active & (x + 1 < self.board_size)
        ops[:, UNIT_OP_TO_ID["WEST"]] = active & (x > 0)
        pickup = adjacent.unsqueeze(1) & self.shed.gt(0)
        pickup[:, 0] = False
        items[:, UNIT_OP_TO_ID["PICKUP"]] = pickup
        qmax[:, UNIT_OP_TO_ID["PICKUP"]] = torch.where(
            pickup, self.shed.clamp_min(0), qmax[:, UNIT_OP_TO_ID["PICKUP"]]
        )
        ops[:, UNIT_OP_TO_ID["PICKUP"]] = pickup.any(dim=1)
        ops[:, UNIT_OP_TO_ID["DROP"]] = adjacent & inv.gt(0).any(dim=1)

        room = self.shed_room()
        item_ids = torch.arange(ITEM_CLASSES, device=self.device).unsqueeze(0).expand(b, -1)
        structure_kind = self._constant(ANIMAL_STRUCTURE_KIND_BY_ITEM)[item_ids]
        carried = inv.gt(0)
        animal_here = (
            active.unsqueeze(1)
            & item_ids.gt(0)
            & structure_kind.gt(0)
            & kind.unsqueeze(1).eq(structure_kind)
            & animal.unsqueeze(1).eq(0)
            & is_dict.unsqueeze(1)
        )
        to_shed = adjacent.unsqueeze(1) & room.unsqueeze(1).gt(0)
        place = carried & (animal_here | to_shed)
        place[:, 0] = False
        items[:, UNIT_OP_TO_ID["PLACE"]] = place
        place_max = torch.minimum(inv.clamp_min(0), room.unsqueeze(1))
        place_max = torch.where(animal_here, torch.ones_like(place_max), place_max)
        qmax[:, UNIT_OP_TO_ID["PLACE"]] = torch.where(
            place, place_max, qmax[:, UNIT_OP_TO_ID["PLACE"]]
        )
        ops[:, UNIT_OP_TO_ID["PLACE"]] = place.any(dim=1)

        unlocked = active & kind.ne(TILE_KIND_TO_ID["LOCKED"])
        empty = unlocked & kind.eq(TILE_KIND_TO_ID["EMPTY"])
        crop_ids = self._constant(CROP_IDS)
        plant_by_item = torch.zeros((b, ITEM_CLASSES), dtype=torch.bool, device=self.device)
        available_seed = self.seeds[:, crop_ids] - self.plant_demand[:, crop_ids]
        crop_allowed = (
            empty.unsqueeze(1)
            & available_seed.gt(0)
            & ~self.atomic_plant_blocked[:, crop_ids]
        )
        plant_by_item[:, crop_ids] = crop_allowed
        items[:, UNIT_OP_TO_ID["PLANT"]] = plant_by_item
        ops[:, UNIT_OP_TO_ID["PLANT"]] = crop_allowed.any(dim=1)
        ops[:, UNIT_OP_TO_ID["BUILD_COOP"]] = empty
        ops[:, UNIT_OP_TO_ID["BUILD_PASTURE"]] = empty

        plant_tile = unlocked & is_dict & kind.eq(TILE_KIND_TO_ID["PLANT"])
        watered = self.grid_watered[rows, y, x]
        yield_units = self.grid_yield[rows, y, x]
        planted_day = self.grid_planted_day[rows, y, x]
        first_yield = self._constant(CROP_FIRST_YIELD_BY_ITEM)[crop.clamp(0, ITEM_CLASSES - 1)]
        age = self.day - planted_day
        ops[:, UNIT_OP_TO_ID["WATER"]] = plant_tile & ~watered
        ops[:, UNIT_OP_TO_ID["HARVEST"]] |= (
            plant_tile & yield_units.gt(0) & age.ge(first_yield)
        )
        ops[:, UNIT_OP_TO_ID["FERTILIZE"]] = (
            plant_tile & inv[:, _FERTILIZER].gt(0)
        )
        ops[:, UNIT_OP_TO_ID["DIG"]] = plant_tile

        animal_tile = unlocked & is_dict & animal.gt(0)
        fed = self.grid_fed[rows, y, x]
        cared = self.grid_cared[rows, y, x]
        fertilizer_available = self.grid_fertilizer_available[rows, y, x]
        ops[:, UNIT_OP_TO_ID["HARVEST"]] |= animal_tile & yield_units.gt(0)
        ops[:, UNIT_OP_TO_ID["FEED"]] = animal_tile & ~fed & inv[:, _WHEAT].gt(0)
        ops[:, UNIT_OP_TO_ID["COLLECT_FERTILIZER"]] = (
            animal_tile & fertilizer_available
        )
        ops[:, UNIT_OP_TO_ID["CARE"]] = animal_tile & ~cared

        other_dict = unlocked & is_dict & ~plant_tile & ~animal_tile
        ops[:, UNIT_OP_TO_ID["DIG"]] |= other_dict
        return TensorUnitLegal(ops, items, qmax)
    def legal_market(self, slot: int) -> TensorMarketLegal:
        b = self.batch_size
        ops = torch.zeros((b, len(MARKET_OPS)), dtype=torch.bool, device=self.device)
        items = torch.zeros(
            (b, len(MARKET_OPS), ITEM_CLASSES), dtype=torch.bool, device=self.device
        )
        qmax = torch.full(
            (b, len(MARKET_OPS), ITEM_CLASSES), -1, dtype=torch.long, device=self.device
        )
        ops[:, MARKET_OP_TO_ID["STOP_QUEUE"]] = True
        active = ~self.market_stopped & self.market_slots_used.lt(10) & (int(slot) < 10)
        ops[:, MARKET_OP_TO_ID["NOP_SLOT"]] = active

        hire_cost = self.next_hire_cost()
        ops[:, MARKET_OP_TO_ID["HIRE"]] = (
            active & ~self.hire_count_uncertain & self.cash.ge(hire_cost)
        )
        land_cost, land_exhausted = self.next_land_cost()
        ops[:, MARKET_OP_TO_ID["BUY_LAND"]] = (
            active & ~land_exhausted & ~self.land_count_uncertain
            & self.cash.ge(land_cost)
        )

        seed_ids = self._constant(CROP_IDS)
        seed_cost = self._constant(SEED_COST_BY_ITEM)[seed_ids]
        seed_allowed = active.unsqueeze(1) & self.cash.unsqueeze(1).ge(seed_cost.unsqueeze(0))
        items[:, MARKET_OP_TO_ID["BUY_SEED"], seed_ids] = seed_allowed
        seed_qmax = self.cash.unsqueeze(1).clamp_min(0) // seed_cost.unsqueeze(0)
        qmax[:, MARKET_OP_TO_ID["BUY_SEED"], seed_ids] = torch.where(
            seed_allowed, seed_qmax, torch.full_like(seed_qmax, -1)
        )
        ops[:, MARKET_OP_TO_ID["BUY_SEED"]] = seed_allowed.any(dim=1)
        room = self.shed_room()
        animal_ids = self._constant(ANIMAL_IDS)
        animal_cost = self._constant(ANIMAL_COST_BY_ITEM)[animal_ids]
        animal_allowed = (
            active.unsqueeze(1)
            & self.cash.unsqueeze(1).ge(animal_cost.unsqueeze(0))
            & room.unsqueeze(1).gt(0)
        )
        items[:, MARKET_OP_TO_ID["BUY_ANIMAL"], animal_ids] = animal_allowed
        animal_qmax = torch.minimum(
            room.unsqueeze(1).expand(-1, len(ANIMALS)),
            self.cash.unsqueeze(1).clamp_min(0) // animal_cost.unsqueeze(0),
        )
        qmax[:, MARKET_OP_TO_ID["BUY_ANIMAL"], animal_ids] = torch.where(
            animal_allowed, animal_qmax, torch.full_like(animal_qmax, -1)
        )
        ops[:, MARKET_OP_TO_ID["BUY_ANIMAL"]] = animal_allowed.any(dim=1)

        product_ids = self.cash.new_tensor([_WHEAT, _FERTILIZER])
        quotes = self.market_prices[:, product_ids].clamp_min(1)
        product_allowed = (
            active.unsqueeze(1)
            & self.cash.unsqueeze(1).ge(quotes)
            & room.unsqueeze(1).gt(0)
        )
        items[:, MARKET_OP_TO_ID["BUY_PRODUCT"], product_ids] = product_allowed
        product_qmax = torch.minimum(
            room.unsqueeze(1).expand_as(quotes),
            self.cash.unsqueeze(1).clamp_min(0) // quotes,
        )
        qmax[:, MARKET_OP_TO_ID["BUY_PRODUCT"], product_ids] = torch.where(
            product_allowed, product_qmax, torch.full_like(product_qmax, -1)
        )
        ops[:, MARKET_OP_TO_ID["BUY_PRODUCT"]] = product_allowed.any(dim=1)
        product_ids = self._constant(PRODUCT_IDS)
        available = self.shed[:, product_ids].clamp_min(0)
        sell_allowed = active.unsqueeze(1) & available.gt(0)
        items[:, MARKET_OP_TO_ID["SELL"], product_ids] = sell_allowed
        qmax[:, MARKET_OP_TO_ID["SELL"], product_ids] = torch.where(
            sell_allowed, available, torch.full_like(available, -1)
        )
        ops[:, MARKET_OP_TO_ID["SELL"]] = sell_allowed.any(dim=1)
        return TensorMarketLegal(ops, items, qmax)

    def _set_tile_empty(self, rows: torch.Tensor, y: torch.Tensor, x: torch.Tensor) -> None:
        self.grid_kind[rows, y, x] = TILE_KIND_TO_ID["EMPTY"]
        self.grid_is_dict[rows, y, x] = False
        self.grid_crop[rows, y, x] = 0
        self.grid_animal[rows, y, x] = 0
        self.grid_watered[rows, y, x] = False
        self.grid_fed[rows, y, x] = False
        self.grid_cared[rows, y, x] = False
        self.grid_fertilizer_available[rows, y, x] = False
        self.grid_yield[rows, y, x] = 0
        self.grid_planted_day[rows, y, x] = 0
        self.grid_fertilized_until_day[rows, y, x] = -1

    def apply_unit(self, actor_index: int, op: torch.Tensor, item: torch.Tensor,
                   quantity: torch.Tensor, *, active: torch.Tensor | None = None) -> None:
        legal = self.legal_unit(actor_index)
        rows = torch.arange(self.batch_size, device=self.device)
        active = (
            self.unit_mask[:, actor_index].clone()
            if active is None else active.to(self.device, dtype=torch.bool)
        )
        op = op.to(self.device, dtype=torch.long)
        item = item.to(self.device, dtype=torch.long).clamp(0, ITEM_CLASSES - 1)
        quantity = quantity.to(self.device, dtype=torch.long)
        op_ok = legal.op_mask.gather(1, op.unsqueeze(1)).squeeze(1)
        item_ops = self.cash.new_tensor([
            UNIT_OP_TO_ID["PICKUP"], UNIT_OP_TO_ID["PLACE"], UNIT_OP_TO_ID["PLANT"]
        ])
        needs_item = op.unsqueeze(1).eq(item_ops.unsqueeze(0)).any(dim=1)
        item_ok = legal.item_mask[
            rows, op.clamp(0, len(UNIT_OPS) - 1), item
        ]
        active = active & op_ok & (~needs_item | item_ok)
        _, x, y = self._tile_values(actor_index)

        moves = {
            "NORTH": (0, -1), "SOUTH": (0, 1),
            "EAST": (1, 0), "WEST": (-1, 0),
        }
        for name, (dx, dy) in moves.items():
            mask = active & op.eq(UNIT_OP_TO_ID[name])
            self.positions[:, actor_index, 0] += mask.to(torch.long) * int(dx)
            self.positions[:, actor_index, 1] += mask.to(torch.long) * int(dy)

        drop = active & op.eq(UNIT_OP_TO_ID["DROP"])
        order = self.inventory_order[:, actor_index]
        for slot in range(order.shape[1]):
            current_item = order[:, slot]
            valid = drop & current_item.gt(0)
            amount = self.inventory[
                rows, actor_index, current_item.clamp(0, ITEM_CLASSES - 1)
            ]
            room = self.shed_room()
            take = torch.minimum(amount.clamp_min(0), room)
            take = torch.where(valid, take, torch.zeros_like(take))
            old_shed = self.shed[
                rows, current_item.clamp(0, ITEM_CLASSES - 1)
            ]
            self.shed[
                rows, current_item.clamp(0, ITEM_CLASSES - 1)
            ] = old_shed + take
        self.inventory[drop, actor_index] = 0
        self.inventory_order[drop, actor_index] = 0

        pickup = active & op.eq(UNIT_OP_TO_ID["PICKUP"])
        n = torch.where(quantity.lt(0), torch.ones_like(quantity), quantity)
        n = torch.minimum(
            n.clamp_min(0), self.shed[rows, item].clamp_min(0)
        )
        n = torch.where(
            pickup & quantity.ne(0), n, torch.zeros_like(n)
        )
        self.shed[rows, item] -= n
        self._inventory_add(actor_index, item, n, pickup & n.gt(0))

        place = active & op.eq(UNIT_OP_TO_ID["PLACE"])
        kind = self.grid_kind[rows, y, x]
        is_dict = self.grid_is_dict[rows, y, x]
        structure = self._constant(ANIMAL_STRUCTURE_KIND_BY_ITEM)[item]
        animal_here = (
            place & structure.gt(0) & kind.eq(structure)
            & self.grid_animal[rows, y, x].eq(0) & is_dict
        )
        self._inventory_sub(
            actor_index, item, torch.ones_like(quantity), animal_here,
        )
        ar, ay, ax = rows[animal_here], y[animal_here], x[animal_here]
        self.grid_animal[ar, ay, ax] = item[animal_here]
        self.grid_is_dict[ar, ay, ax] = True
        self.grid_yield[ar, ay, ax] = 0
        self.grid_fed[ar, ay, ax] = False
        self.grid_cared[ar, ay, ax] = False
        self.grid_fertilizer_available[ar, ay, ax] = False

        to_shed = place & ~animal_here
        n = torch.where(quantity.lt(0), torch.ones_like(quantity), quantity)
        carried = self.inventory[rows, actor_index, item].clamp_min(0)
        n = torch.minimum(
            torch.minimum(n.clamp_min(0), carried), self.shed_room()
        )
        n = torch.where(
            to_shed & quantity.ne(0), n, torch.zeros_like(n)
        )
        self._inventory_sub(
            actor_index, item, n, to_shed & n.gt(0)
        )
        self.shed[rows, item] += n

        plant = active & op.eq(UNIT_OP_TO_ID["PLANT"])
        self.plant_demand[rows, item] += plant.to(torch.long)
        pr, py, px = rows[plant], y[plant], x[plant]
        crop_item = item[plant]
        ongoing = self._constant(CROP_ONGOING_BY_ITEM)[crop_item]
        self.grid_kind[pr, py, px] = TILE_KIND_TO_ID["PLANT"]
        self.grid_is_dict[pr, py, px] = True
        self.grid_crop[pr, py, px] = crop_item
        self.grid_animal[pr, py, px] = 0
        self.grid_planted_day[pr, py, px] = self.day[plant]
        self.grid_watered[pr, py, px] = False
        self.grid_yield[pr, py, px] = (~ongoing).to(torch.long)
        self.grid_fertilized_until_day[pr, py, px] = -1

        for name, kind_id in (
            ("BUILD_COOP", TILE_KIND_TO_ID["COOP"]),
            ("BUILD_PASTURE", TILE_KIND_TO_ID["PASTURE"]),
        ):
            mask = active & op.eq(UNIT_OP_TO_ID[name])
            mr, my, mx = rows[mask], y[mask], x[mask]
            self.grid_kind[mr, my, mx] = kind_id
            self.grid_is_dict[mr, my, mx] = True
            self.grid_crop[mr, my, mx] = 0
            self.grid_animal[mr, my, mx] = 0

        water = active & op.eq(UNIT_OP_TO_ID["WATER"])
        self.grid_watered[rows[water], y[water], x[water]] = True
        harvest = active & op.eq(UNIT_OP_TO_ID["HARVEST"])
        amount = torch.where(
            harvest,
            self.grid_yield[rows, y, x].clone(),
            torch.zeros_like(self.grid_yield[rows, y, x]),
        )
        crop_item = self.grid_crop[rows, y, x]
        animal_item = self.grid_animal[rows, y, x]
        self.grid_yield[rows[harvest], y[harvest], x[harvest]] = 0

        plant_rows = harvest & crop_item.gt(0)
        self._inventory_add(
            actor_index, crop_item, amount, plant_rows
        )
        ongoing = self._constant(CROP_ONGOING_BY_ITEM)[
            crop_item.clamp(0, ITEM_CLASSES - 1)
        ]
        clear = plant_rows & ~ongoing
        self._set_tile_empty(
            rows[clear], y[clear], x[clear],
        )

        animal_rows = harvest & animal_item.gt(0)
        product = self._constant(ANIMAL_PRODUCT_BY_ITEM)[
            animal_item.clamp(0, ITEM_CLASSES - 1)
        ]
        self._inventory_add(
            actor_index, product, amount, animal_rows
        )

        fertilize = active & op.eq(UNIT_OP_TO_ID["FERTILIZE"])
        self._inventory_sub(
            actor_index,
            torch.full_like(item, _FERTILIZER),
            torch.ones_like(quantity),
            fertilize,
        )
        fr, fy, fx = rows[fertilize], y[fertilize], x[fertilize]
        self.grid_fertilized_until_day[fr, fy, fx] = torch.maximum(
            self.grid_fertilized_until_day[fr, fy, fx],
            self.day[fertilize] + 2,
        )
        dig = active & op.eq(UNIT_OP_TO_ID["DIG"])
        self._set_tile_empty(rows[dig], y[dig], x[dig])

        feed = active & op.eq(UNIT_OP_TO_ID["FEED"])
        self._inventory_sub(
            actor_index,
            torch.full_like(item, _WHEAT),
            torch.ones_like(quantity),
            feed,
        )
        self.grid_fed[rows[feed], y[feed], x[feed]] = True

        collect = active & op.eq(UNIT_OP_TO_ID["COLLECT_FERTILIZER"])
        self.grid_fertilizer_available[
            rows[collect], y[collect], x[collect]
        ] = False
        self._inventory_add(
            actor_index,
            torch.full_like(item, _FERTILIZER),
            torch.ones_like(quantity),
            collect,
        )

        care = active & op.eq(UNIT_OP_TO_ID["CARE"])
        self.grid_cared[rows[care], y[care], x[care]] = True

    def apply_market(
        self,
        op: torch.Tensor,
        item: torch.Tensor,
        quantity: torch.Tensor,
        *,
        known_executed: torch.Tensor | None = None,
        active: torch.Tensor | None = None,
    ) -> None:
        rows = torch.arange(self.batch_size, device=self.device)
        op = op.to(self.device, dtype=torch.long)
        item = item.to(self.device, dtype=torch.long).clamp(0, ITEM_CLASSES - 1)
        quantity = quantity.to(self.device, dtype=torch.long)
        active = (
            torch.ones(
                self.batch_size, dtype=torch.bool, device=self.device
            )
            if active is None
            else active.to(self.device, dtype=torch.bool)
        )
        known = (
            torch.zeros_like(active)
            if known_executed is None
            else known_executed.to(self.device, dtype=torch.bool)
        )

        was_stopped = self.market_stopped.clone()
        stop = active & op.eq(MARKET_OP_TO_ID["STOP_QUEUE"])
        self.market_stopped |= stop
        live = (
            active
            & ~stop
            & ~was_stopped
            & self.market_slots_used.lt(10)
        )
        self.market_slots_used += live.to(torch.long)
        nop = op.eq(MARKET_OP_TO_ID["NOP_SLOT"])
        work = live & ~nop

        hire = work & op.eq(MARKET_OP_TO_ID["HIRE"])
        cost = self.next_hire_cost()
        exact = hire & known
        self.cash[exact] = (
            self.cash[exact] - cost[exact]
        ).clamp_min(0)
        self.hires_today[exact] += 1

        uncertain_ok = hire & ~known & ~self.hire_count_uncertain
        enough = uncertain_ok & self.cash.ge(cost)
        self.cash[enough] -= cost[enough]
        self.hires_today[enough] += 1
        maybe = uncertain_ok & ~enough & self.cash_uncertain
        self.cash[maybe] = 0
        self.hire_count_uncertain[maybe] = True

        land = work & op.eq(MARKET_OP_TO_ID["BUY_LAND"])
        cost, exhausted = self.next_land_cost()
        eligible = (
            land & ~exhausted & ~self.land_count_uncertain
        )
        exact = eligible & known
        self.cash[exact] = (
            self.cash[exact] - cost[exact]
        ).clamp_min(0)
        self.land_count[exact] += 1
        uncertain = eligible & ~known
        enough = uncertain & self.cash.ge(cost)
        self.cash[enough] -= cost[enough]
        self.land_count[enough] += 1
        maybe = uncertain & ~enough & self.cash_uncertain
        self.cash[maybe] = 0
        self.land_count_uncertain[maybe] = True

        valid_qty = work & quantity.gt(0)

        sell = valid_qty & op.eq(MARKET_OP_TO_ID["SELL"])
        available = self.shed[rows, item].clamp_min(0)
        sold = torch.where(
            known, quantity, torch.minimum(quantity, available)
        )
        sold = torch.where(
            sell, sold, torch.zeros_like(sold)
        )
        self.shed[rows, item] -= sold
        self.cash += sold
        self.cash_uncertain |= sell & sold.gt(0)

        buy_seed = valid_qty & op.eq(
            MARKET_OP_TO_ID["BUY_SEED"]
        )
        cost = self._constant(SEED_COST_BY_ITEM)[item]
        total = cost * quantity
        exact = buy_seed & known & cost.gt(0)
        self.cash[exact] = (
            self.cash[exact] - total[exact]
        ).clamp_min(0)
        self.seeds[rows[exact], item[exact]] += quantity[exact]

        normal = buy_seed & ~known & cost.gt(0)
        enough = normal & self.cash.ge(total)
        self.cash[enough] -= total[enough]
        self.seeds[rows[enough], item[enough]] += quantity[enough]

        partial = normal & ~enough & ~self.cash_uncertain
        units = torch.where(
            cost.gt(0),
            self.cash // cost.clamp_min(1),
            torch.zeros_like(cost),
        )
        units = torch.minimum(
            units, quantity
        ).clamp_min(0)
        self.cash[partial] -= (
            units[partial] * cost[partial]
        )
        self.seeds[
            rows[partial], item[partial]
        ] += units[partial]

        uncertain = (
            normal & ~enough & self.cash_uncertain
        )
        self.cash[uncertain] = torch.minimum(
            self.cash[uncertain],
            (cost[uncertain] - 1).clamp_min(0),
        )

        buy_animal = valid_qty & op.eq(
            MARKET_OP_TO_ID["BUY_ANIMAL"]
        )
        cost = self._constant(
            ANIMAL_COST_BY_ITEM
        )[item]
        room = self.shed_room()
        exact = (
            buy_animal & known & cost.gt(0)
        )
        self.cash[exact] = (
            self.cash[exact]
            - cost[exact] * quantity[exact]
        ).clamp_min(0)
        self.shed[
            rows[exact], item[exact]
        ] += quantity[exact]

        normal = (
            buy_animal & ~known & cost.gt(0)
        )
        certain = (
            normal
            & ~self.cash_uncertain
            & ~self.shed_uncertain
        )
        units = torch.minimum(
            torch.minimum(
                quantity,
                self.cash // cost.clamp_min(1),
            ),
            room,
        ).clamp_min(0)
        self.cash[certain] -= (
            units[certain] * cost[certain]
        )
        self.shed[
            rows[certain], item[certain]
        ] += units[certain]

        remaining = normal & ~certain
        fully = (
            remaining
            & room.ge(quantity)
            & self.cash.ge(cost * quantity)
        )
        self.cash[fully] -= (
            cost[fully] * quantity[fully]
        )
        self.shed[
            rows[fully], item[fully]
        ] += quantity[fully]

        uncertain = remaining & ~fully
        reserve = torch.minimum(quantity, room)
        self.shed_reserved[uncertain] += (
            reserve[uncertain]
        )
        self.cash[uncertain] = torch.minimum(
            self.cash[uncertain],
            (cost[uncertain] - 1).clamp_min(0),
        )
        self.cash_uncertain[uncertain] = True
        self.shed_uncertain[uncertain] = True

        buy_product = valid_qty & op.eq(
            MARKET_OP_TO_ID["BUY_PRODUCT"]
        )
        product_ok = item.eq(_WHEAT) | item.eq(
            _FERTILIZER
        )
        buy_product &= product_ok
        room = self.shed_room()
        exact = buy_product & known
        self.shed[
            rows[exact], item[exact]
        ] += quantity[exact]
        self.cash[exact] = 0
        self.cash_uncertain[exact] = True

        normal = (
            buy_product
            & ~known
            & room.gt(0)
            & self.cash.gt(0)
        )
        reserve = torch.minimum(quantity, room)
        self.shed_reserved[normal] += reserve[normal]
        self.cash[normal] = 0
        self.cash_uncertain[normal] = True
        self.shed_uncertain[normal] = True

    def active_market_mask(
        self, legal: TensorMarketLegal,
    ) -> torch.Tensor:
        ids = self.cash.new_tensor([
            MARKET_OP_TO_ID[name]
            for name in ACTIVE_MARKET_OPS
        ])
        return legal.op_mask.index_select(1, ids)

    def continue_market_mask(
        self, legal: TensorMarketLegal,
    ) -> torch.Tensor:
        active = self.active_market_mask(
            legal
        ).any(dim=1)
        return torch.stack(
            [torch.ones_like(active), active],
            dim=1,
        )
