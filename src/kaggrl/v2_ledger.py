from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from .constants import ANIMALS, CROPS, ITEM_TO_ID, PRODUCTS, UNIT_OPS

ITEM_NAMES = tuple(ITEM_TO_ID.keys())
MARKET_OPS = ("STOP_QUEUE", "NOP_SLOT", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL", "HIRE", "BUY_LAND")
MOVES = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}
ANIMAL_STRUCTURE = {"GOOSE": "COOP", "COW": "PASTURE", "SHEEP": "PASTURE"}
CROP_FIRST_YIELD = {"WHEAT": 2, "CARROT": 2, "TOMATO": 8, "STRAWBERRY": 10, "MELON": 10}
CROP_ONGOING = {"WHEAT": False, "CARROT": False, "TOMATO": True, "STRAWBERRY": True, "MELON": False}
CROP_MAX_YIELD_DAY = {"WHEAT": 4, "CARROT": 3, "TOMATO": 8, "STRAWBERRY": 10, "MELON": 12}
SEED_COST = {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}
ANIMAL_COST = {"GOOSE": 300, "COW": 400, "SHEEP": 500}
LAND_ORDER = ("NE", "SW", "SE")
LAND_PRICES = (1000, 2000, 4000)


@dataclass
class LegalMask:
    ops: dict[str, bool]
    items: dict[str, dict[str, bool]] = field(default_factory=dict)
    uncertain_ops: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)

    def allows(self, op: str, item: str | None = None) -> bool:
        if not self.ops.get(op, False):
            return False
        if item is None:
            return True
        choices = self.items.get(op)
        return bool(choices and choices.get(item, False))

    def is_uncertain(self, op: str) -> bool:
        return op in self.uncertain_ops


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


def _fib(n: int) -> int:
    a, b = 1, 1
    for _ in range(max(0, int(n))):
        a, b = b, a + b
    return a


def _parts(command: Any) -> tuple[str, str | None, int | None, str | None]:
    if isinstance(command, (list, tuple)):
        if not command:
            return "NOP_SLOT", None, None, "NOP_SLOT"
        op = str(command[0])
        item = str(command[1]) if len(command) > 1 and command[1] is not None else None
        try:
            qty = int(command[2]) if len(command) > 2 else None
        except (TypeError, ValueError):
            qty = None
        return op, item, qty, None
    data = _mapping(command)
    kind = data.get("kind")
    if kind in {"STOP_QUEUE", "NOP_SLOT"}:
        return str(kind), None, None, str(kind)
    return str(data.get("op", "PASS")), data.get("item"), data.get("quantity"), kind


class ShadowLedger:
    def __init__(self):
        self.board_size = 10
        self.step = 0
        self.day = 0
        self.shed_capacity = 100
        self.cash_lower_bound = 0
        self.cash_uncertain = False
        self.hires_today = 0
        self.hire_count_uncertain = False
        self.unlocked_quadrants: list[str] = []
        self.land_count_uncertain = False
        self.shed: dict[str, int] = {}
        # Capacity reserved by market buys whose exact execution is unknown.
        # Reserved units are NOT sellable inventory; they only reduce guaranteed room.
        self.shed_reserved = 0
        self.shed_uncertain = False
        self.shed_uncertain_items: set[str] = set()
        self.seeds: dict[str, int] = {}
        self.seed_uncertain_items: set[str] = set()
        self.plant_demand: dict[str, int] = {}
        self.atomic_plant_blocked: set[str] = set()
        self.grid: list[list[Any]] = []
        self.unit_positions: dict[str, list[int]] = {}
        self.unit_inventories: dict[str, dict[str, int]] = {}
        self.market_inventory: dict[str, int] = {}
        self.market_prices: dict[str, int] = {}
        self.market_slots_used = 0
        self.market_stopped = False

    @classmethod
    def from_state(cls, state: Any, *, shed_capacity: int = 100) -> "ShadowLedger":
        obj = cls()
        own = _mapping(_field(state, "own", {}))
        private = _mapping(_field(state, "private", {}))
        obj.grid = deepcopy(_field(state, "own_grid", own.get("tiles") or []))
        obj.board_size = len(obj.grid) or 10
        obj.step = int(_field(state, "step", 0) or 0)
        obj.day = int(_field(state, "day", 0) or 0)
        obj.shed_capacity = int(shed_capacity)
        obj.cash_lower_bound = int(own.get("money", 0) or 0)
        obj.hires_today = int(own.get("hires_today", 0) or 0)
        obj.unlocked_quadrants = list(own.get("unlocked_quadrants") or [])
        obj.shed = {str(k): int(v or 0) for k, v in _mapping(private.get("shed") or {}).items()}
        obj.seeds = {str(k): int(v or 0) for k, v in _mapping(private.get("seeds") or {}).items()}
        units = list(_field(state, "own_units", []) or [])
        if not units:
            units = [{"kind": "farmer", "index": 0, "position": own.get("farmer", [0, 0])}]
            units += [{"kind": "hand", "index": i, "position": p}
                      for i, p in enumerate(own.get("hands") or [])]
        inventories = list(private.get("inventories") or [])
        for slot, unit in enumerate(units):
            data = _mapping(unit)
            actor = "farmer" if str(data.get("kind")) == "farmer" else f"hand:{int(data.get('index', slot - 1))}"
            obj.unit_positions[actor] = list(data.get("position") or [0, 0])
            inv = data.get("inventory")
            if not isinstance(inv, dict) and slot < len(inventories):
                inv = inventories[slot]
            obj.unit_inventories[actor] = {str(k): int(v or 0) for k, v in _mapping(inv or {}).items()}
        market = _mapping(_field(state, "market", {}))
        obj.market_inventory = {str(k): int(v or 0) for k, v in _mapping(market.get("inventory") or {}).items()}
        obj.market_prices = {str(k): int(v or 0) for k, v in _mapping(market.get("prices") or {}).items()}
        return obj

    @property
    def next_hire_cost(self) -> int:
        return _fib(self.hires_today)

    @property
    def next_land_cost(self) -> int | None:
        extra = max(0, len(self.unlocked_quadrants) - 1)
        return LAND_PRICES[extra] if extra < len(LAND_PRICES) else None

    def _shed_room(self) -> int:
        known = sum(max(0, int(v)) for v in self.shed.values())
        return max(0, self.shed_capacity - known - max(0, int(self.shed_reserved)))

    def _shed_adjacent(self, pos: list[int]) -> bool:
        half = self.board_size // 2
        access = {(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)}
        return tuple(pos) in access

    def _actor(self, actor: str) -> tuple[list[int], dict[str, int]]:
        if actor not in self.unit_positions:
            raise KeyError(actor)
        return self.unit_positions[actor], self.unit_inventories[actor]

    def _tile(self, pos: list[int]):
        x, y = int(pos[0]), int(pos[1])
        return self.grid[y][x]

    def set_atomic_plant_blocked(self, crops) -> None:
        self.atomic_plant_blocked = {
            str(crop) for crop in crops if str(crop) in CROPS
        }

    def legal_unit_mask(self, actor: str, partial_joint_action: Any) -> LegalMask:
        pos, inv = self._actor(actor)
        x, y = int(pos[0]), int(pos[1])
        tile = self._tile(pos)
        ops = {op: False for op in UNIT_OPS}
        items = {"PICKUP": {}, "PLACE": {}, "PLANT": {}}
        pickup_max_by_item: dict[str, int] = {}
        place_max_by_item: dict[str, int] = {}
        ops["PASS"] = True
        for op, (dx, dy) in MOVES.items():
            ops[op] = 0 <= x + dx < self.board_size and 0 <= y + dy < self.board_size
        adjacent = self._shed_adjacent(pos)
        if adjacent:
            pickup = {item: int(self.shed.get(item, 0)) > 0 for item in ITEM_NAMES}
            items["PICKUP"] = pickup
            pickup_max_by_item = {
                item: max(0, int(self.shed.get(item, 0)))
                for item, allowed in pickup.items() if allowed
            }
            ops["PICKUP"] = any(pickup.values())
            ops["DROP"] = any(int(v) > 0 for v in inv.values())
        place_items: dict[str, bool] = {}
        room = self._shed_room()
        for item in ITEM_NAMES:
            carried_n = max(0, int(inv.get(item, 0)))
            carried = carried_n > 0
            animal_here = (
                item in ANIMALS and isinstance(tile, dict)
                and tile.get("kind") == ANIMAL_STRUCTURE[item] and "animal" not in tile
            )
            place_items[item] = carried and (animal_here or (adjacent and room > 0))
            if place_items[item]:
                place_max_by_item[item] = 1 if animal_here else min(carried_n, room)
        items["PLACE"] = place_items
        ops["PLACE"] = any(place_items.values())
        if tile == "LOCKED":
            return LegalMask(
                ops=ops, items=items,
                metadata={"unit_quantity_max_by_op_item": {
                    "PICKUP": pickup_max_by_item, "PLACE": place_max_by_item,
                }},
            )
        if tile is None:
            plant = {
                crop: (
                    crop not in self.atomic_plant_blocked
                    and self.seeds.get(crop, 0) - self.plant_demand.get(crop, 0) > 0
                )
                for crop in CROPS
            }
            items["PLANT"] = plant
            ops["PLANT"] = any(plant.values())
            ops["BUILD_COOP"] = True
            ops["BUILD_PASTURE"] = True
        elif isinstance(tile, dict):
            if tile.get("kind") == "PLANT":
                ops["WATER"] = not bool(tile.get("watered_today", False))
                age = self.day - int(tile.get("planted_day", self.day))
                crop = str(tile.get("crop", ""))
                ops["HARVEST"] = int(tile.get("yield_units", 0) or 0) > 0 and age >= CROP_FIRST_YIELD.get(crop, 10**9)
                ops["FERTILIZE"] = int(inv.get("FERTILIZER", 0)) > 0
                ops["DIG"] = True
            elif "animal" in tile:
                ops["HARVEST"] = int(tile.get("yield_units", 0) or 0) > 0
                ops["FEED"] = not bool(tile.get("fed_today", False)) and int(inv.get("WHEAT", 0)) > 0
                ops["COLLECT_FERTILIZER"] = bool(tile.get("fertilizer_available", False))
                ops["CARE"] = not bool(tile.get("cared_today", False))
            else:
                ops["DIG"] = True
        return LegalMask(
            ops=ops, items=items,
            metadata={"unit_quantity_max_by_op_item": {
                "PICKUP": pickup_max_by_item, "PLACE": place_max_by_item,
            }},
        )

    def apply_unit(self, actor: str, command: Any) -> None:
        op, item, qty, _ = _parts(command)
        mask = self.legal_unit_mask(actor, {})
        if not mask.allows(op, item if op in {"PICKUP", "PLACE", "PLANT"} else None):
            return
        pos, inv = self._actor(actor)
        x, y = int(pos[0]), int(pos[1])
        if op in MOVES:
            dx, dy = MOVES[op]
            self.unit_positions[actor] = [x + dx, y + dy]
            return
        if op == "PASS":
            return
        if op == "DROP":
            for product, amount in list(inv.items()):
                amount = max(0, int(amount))
                room = self._shed_room()
                take = min(amount, room)
                if take:
                    self.shed[product] = self.shed.get(product, 0) + take
                inv.pop(product, None)  # engine discards overflow too
            return
        if op == "PICKUP":
            if qty is not None and int(qty) <= 0:
                return
            n = int(qty) if qty is not None else 1
            n = min(n, max(0, int(self.shed.get(item, 0))))
            if n:
                self.shed[item] = self.shed.get(item, 0) - n
                inv[item] = inv.get(item, 0) + n
            return
        tile = self.grid[y][x]
        if op == "PLACE":
            if item in ANIMALS and isinstance(tile, dict) and tile.get("kind") == ANIMAL_STRUCTURE[item] and "animal" not in tile:
                inv[item] -= 1
                if inv[item] <= 0:
                    inv.pop(item, None)
                self.grid[y][x] = {
                    "kind": ANIMAL_STRUCTURE[item], "animal": item, "placed_day": self.day,
                    "yield_units": 0, "consecutive_unfed": 0, "fed_today": False,
                    "cared_today": False, "fertilizer_available": False, "pending_care_bonus": 0,
                }
                return
            if qty is not None and int(qty) <= 0:
                return
            n = int(qty) if qty is not None else 1
            n = min(n, max(0, int(inv.get(item, 0))), self._shed_room())
            if n:
                inv[item] -= n
                if inv[item] <= 0:
                    inv.pop(item, None)
                self.shed[item] = self.shed.get(item, 0) + n
            return
        if op == "PLANT":
            self.plant_demand[item] = self.plant_demand.get(item, 0) + 1
            ongoing = CROP_ONGOING[item]
            self.grid[y][x] = {
                "kind": "PLANT", "crop": item, "planted_day": self.day,
                "watered_today": False, "consecutive_unwatered": 1,
                "yield_units": 0 if ongoing else 1,
                "max_lifespan_step": -1, "fertilized_until_day": -1,
            }
            return
        if op in {"BUILD_COOP", "BUILD_PASTURE"}:
            self.grid[y][x] = {"kind": "COOP" if op == "BUILD_COOP" else "PASTURE"}
            return
        if op == "WATER":
            tile["watered_today"] = True
        elif op == "HARVEST":
            units = int(tile.get("yield_units", 0) or 0)
            tile["yield_units"] = 0
            if tile.get("kind") == "PLANT":
                crop = tile["crop"]
                inv[crop] = inv.get(crop, 0) + units
                if not CROP_ONGOING.get(crop, False):
                    self.grid[y][x] = None
            elif "animal" in tile:
                product = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}[tile["animal"]]
                inv[product] = inv.get(product, 0) + units
        elif op == "FERTILIZE":
            inv["FERTILIZER"] -= 1
            if inv["FERTILIZER"] <= 0:
                inv.pop("FERTILIZER", None)
            tile["fertilized_until_day"] = max(int(tile.get("fertilized_until_day", -1)), self.day + 2)
        elif op == "DIG":
            self.grid[y][x] = None
        elif op == "FEED":
            inv["WHEAT"] -= 1
            if inv["WHEAT"] <= 0:
                inv.pop("WHEAT", None)
            tile["fed_today"] = True
        elif op == "COLLECT_FERTILIZER":
            tile["fertilizer_available"] = False
            inv["FERTILIZER"] = inv.get("FERTILIZER", 0) + 1
        elif op == "CARE":
            tile["cared_today"] = True

    def _fixed_cost_allowed(self, cost: int) -> tuple[bool, bool]:
        # cash_lower_bound is the amount guaranteed to remain after earlier
        # market orders. Once a variable-price order makes cash uncertain,
        # never treat that uncertainty as permission to spend money that may
        # no longer exist. Only guaranteed cash may fund later fixed-cost ops.
        return self.cash_lower_bound >= int(cost), False

    def legal_market_mask(self, slot: int, partial_market: Any) -> LegalMask:
        ops = {op: False for op in MARKET_OPS}
        items = {"BUY_SEED": {}, "BUY_PRODUCT": {}, "BUY_ANIMAL": {}, "SELL": {}}
        uncertain: set[str] = set()
        ops["STOP_QUEUE"] = True
        if self.market_stopped or int(slot) >= 10:
            return LegalMask(ops=ops, items=items, uncertain_ops=uncertain)
        ops["NOP_SLOT"] = True

        if self.hire_count_uncertain:
            # The next Fibonacci hire price is not known after an uncertain
            # prior hire, so no later hire can be guaranteed legal.
            ops["HIRE"] = False
        else:
            ops["HIRE"], u = self._fixed_cost_allowed(self.next_hire_cost)
            if u:
                uncertain.add("HIRE")

        land_cost = self.next_land_cost
        if land_cost is not None:
            if self.land_count_uncertain:
                ops["BUY_LAND"] = False
            else:
                ops["BUY_LAND"], u = self._fixed_cost_allowed(land_cost)
                if u:
                    uncertain.add("BUY_LAND")
        for crop, cost in SEED_COST.items():
            allowed, u = self._fixed_cost_allowed(cost)
            items["BUY_SEED"][crop] = allowed
            if u and allowed:
                uncertain.add("BUY_SEED")
        ops["BUY_SEED"] = any(items["BUY_SEED"].values())

        room_known = self._shed_room()
        for animal, cost in ANIMAL_COST.items():
            cash_ok, u = self._fixed_cost_allowed(cost)
            room_ok = room_known > 0
            items["BUY_ANIMAL"][animal] = cash_ok and room_ok
            if items["BUY_ANIMAL"][animal] and (u or self.shed_uncertain):
                uncertain.add("BUY_ANIMAL")
        ops["BUY_ANIMAL"] = any(items["BUY_ANIMAL"].values())

        for item in ("WHEAT", "FERTILIZER"):
            # Product prices are re-quoted while both players' queues commit.
            # The observed quote is not a guarantee, but it is a necessary
            # affordability floor. Later slots use a zero cash lower bound.
            quote = max(1, int(self.market_prices.get(item, 1) or 1))
            possible_cash = self.cash_lower_bound >= quote
            room_ok = room_known > 0
            items["BUY_PRODUCT"][item] = possible_cash and room_ok
        ops["BUY_PRODUCT"] = any(items["BUY_PRODUCT"].values())
        if ops["BUY_PRODUCT"]:
            uncertain.add("BUY_PRODUCT")  # opponent may move the quote before commit

        sell_max_by_item: dict[str, int] = {}
        market_quantity_max_by_op_item: dict[str, dict[str, int]] = {
            "BUY_SEED": {}, "BUY_PRODUCT": {}, "BUY_ANIMAL": {}, "SELL": {},
        }
        for crop, cost in SEED_COST.items():
            guaranteed = max(0, int(self.cash_lower_bound)) // int(cost)
            if guaranteed > 0:
                market_quantity_max_by_op_item["BUY_SEED"][crop] = guaranteed
        for animal, cost in ANIMAL_COST.items():
            structural = min(
                max(0, int(room_known)),
                max(0, int(self.cash_lower_bound)) // int(cost),
            )
            if structural > 0:
                market_quantity_max_by_op_item["BUY_ANIMAL"][animal] = structural
        if room_known > 0 and self.cash_lower_bound > 0:
            for product in ("WHEAT", "FERTILIZER"):
                if items["BUY_PRODUCT"].get(product, False):
                    quote = max(1, int(self.market_prices.get(product, 1) or 1))
                    affordable = max(0, int(self.cash_lower_bound)) // quote
                    if affordable > 0:
                        market_quantity_max_by_op_item["BUY_PRODUCT"][product] = min(
                            int(room_known), int(affordable),
                        )
        for item in PRODUCTS:
            known_available = max(0, int(self.shed.get(item, 0)))
            # SELL must be legal by construction. Do not sell inventory that only
            # might exist after an uncertain earlier market transaction.
            items["SELL"][item] = known_available > 0
            if known_available > 0:
                sell_max_by_item[item] = known_available
                market_quantity_max_by_op_item["SELL"][item] = known_available
        ops["SELL"] = any(items["SELL"].values())
        if ops["SELL"]:
            uncertain.add("SELL")  # proceeds are quote-dependent; inventory is not
        return LegalMask(
            ops=ops, items=items, uncertain_ops=uncertain,
            metadata={
                "cash_lower_bound": self.cash_lower_bound,
                "cash_uncertain": bool(self.cash_uncertain),
                "shed_room": room_known,
                "sell_max_by_item": sell_max_by_item,
                "market_quantity_max_by_op_item": market_quantity_max_by_op_item,
            },
        )

    def _fixed_execution(self, cost: int) -> bool | None:
        cost = int(cost)
        if self.cash_lower_bound >= cost:
            self.cash_lower_bound -= cost
            return True
        if self.cash_uncertain:
            self.cash_lower_bound = 0
            return None
        return False

    def apply_market(self, order: Any) -> None:
        op, item, qty, kind = _parts(order)
        known_executed = bool(_field(order, "_executed", False))
        if op == "STOP_QUEUE" or kind == "STOP_QUEUE":
            self.market_stopped = True
            return
        if self.market_stopped or self.market_slots_used >= 10:
            return
        self.market_slots_used += 1
        if op == "NOP_SLOT" or kind == "NOP_SLOT":
            return
        if op == "HIRE":
            if known_executed:
                cost = self.next_hire_cost
                self.cash_lower_bound = max(0, self.cash_lower_bound - cost)
                self.hires_today += 1
                self.hire_count_uncertain = False
                return
            if self.hire_count_uncertain:
                return
            result = self._fixed_execution(self.next_hire_cost)
            if result is True:
                self.hires_today += 1
            elif result is None:
                self.hire_count_uncertain = True
            return
        if op == "BUY_LAND":
            cost = self.next_land_cost
            if cost is None or self.land_count_uncertain:
                return
            if known_executed:
                self.cash_lower_bound = max(0, self.cash_lower_bound - cost)
                extra = len(self.unlocked_quadrants) - 1
                self.unlocked_quadrants.append(LAND_ORDER[extra])
                self.land_count_uncertain = False
                return
            result = self._fixed_execution(cost)
            if result is True:
                extra = len(self.unlocked_quadrants) - 1
                self.unlocked_quadrants.append(LAND_ORDER[extra])
            elif result is None:
                self.land_count_uncertain = True
            return
        if qty is None or int(qty) <= 0:
            return
        n = int(qty)
        if op == "SELL":
            if item not in PRODUCTS:
                return
            available = max(0, int(self.shed.get(item, 0)))
            if known_executed and n > available:
                raise RuntimeError(
                    f"executed SELL exceeds shadow inventory: {item} {n}>{available}"
                )
            sold = n if known_executed else min(n, available)
            if sold <= 0:
                return
            self.shed[item] = self.shed.get(item, 0) - sold
            self.cash_lower_bound += sold  # market price floor is exactly 1
            self.cash_uncertain = True
            return
        if op == "BUY_SEED":
            if item not in SEED_COST:
                return
            total = SEED_COST[item] * n
            if known_executed:
                self.cash_lower_bound = max(0, self.cash_lower_bound - total)
                self.seeds[item] = self.seeds.get(item, 0) + n
                self.seed_uncertain_items.discard(item)
                return
            if self.cash_lower_bound >= total:
                self.cash_lower_bound -= total
                self.seeds[item] = self.seeds.get(item, 0) + n
            elif not self.cash_uncertain:
                units = min(n, self.cash_lower_bound // SEED_COST[item])
                self.cash_lower_bound -= units * SEED_COST[item]
                self.seeds[item] = self.seeds.get(item, 0) + units
            else:
                self.seed_uncertain_items.add(item)
                self.cash_lower_bound = min(self.cash_lower_bound, SEED_COST[item] - 1)
            return
        if op == "BUY_ANIMAL":
            if item not in ANIMAL_COST:
                return
            cost = ANIMAL_COST[item]
            room = self._shed_room()
            if known_executed:
                if not self.shed_uncertain and n > room:
                    raise RuntimeError(
                        f"executed BUY_ANIMAL exceeds shed room: {item} {n}>{room}"
                    )
                self.cash_lower_bound = max(0, self.cash_lower_bound - cost * n)
                self.shed[item] = self.shed.get(item, 0) + n
                return
            if not self.cash_uncertain and not self.shed_uncertain:
                units = min(n, self.cash_lower_bound // cost, room)
                self.cash_lower_bound -= units * cost
                self.shed[item] = self.shed.get(item, 0) + units
                return
            if room >= n and self.cash_lower_bound >= cost * n:
                self.cash_lower_bound -= cost * n
                self.shed[item] = self.shed.get(item, 0) + n
            else:
                reserve = min(n, room)
                self.shed_reserved += reserve
                self.cash_lower_bound = min(self.cash_lower_bound, max(0, cost - 1))
                self.cash_uncertain = True
                self.shed_uncertain = True
                self.shed_uncertain_items.add(item)
            return
        if op == "BUY_PRODUCT":
            if item not in {"WHEAT", "FERTILIZER"}:
                return
            room = self._shed_room()
            if known_executed:
                if not self.shed_uncertain and n > room:
                    raise RuntimeError(
                        f"executed BUY_PRODUCT exceeds shed room: {item} {n}>{room}"
                    )
                self.shed[item] = self.shed.get(item, 0) + n
                # Quantity is exact from the replay tracer, but variable market
                # prices make the remaining cash amount unknown to ShadowLedger.
                self.cash_lower_bound = 0
                self.cash_uncertain = True
                self.shed_uncertain_items.discard(item)
                return
            if room <= 0:
                return
            if self.cash_lower_bound <= 0:
                return
            self.shed_reserved += min(n, room)
            # The exact variable-price spend is unknown until the engine
            # commits both players' orders. The only guaranteed post-order
            # cash floor is zero, so later slots must not spend the old cash.
            self.cash_lower_bound = 0
            self.cash_uncertain = True
            self.shed_uncertain = True
            self.shed_uncertain_items.add(item)
            return
