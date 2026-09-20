from __future__ import annotations
import math
import numpy as np
from .clock import (
    CLOCK_FEATURES,
    LEGACY_CLOCK_FEATURES,
    V4_CLOCK_EXTRA_FEATURES,
    resolve_clock,
)
from .constants import CROPS, ANIMALS, PRODUCTS

SHOPS = (
    "BAKERY", "PIZZA_SHOP", "BRUNCH_SPOT", "YARN_STORE",
    "ICE_CREAM_SHOP", "PET_CAFE", "SMOOTHIE_SHOP", "FARMERS_MARKET",
)
SHELF_ITEMS = tuple(dict.fromkeys((*PRODUCTS, *ANIMALS)))
QUADRANTS = ("NW", "NE", "SW", "SE")


class ObservationEncoder:
    """Canonical macro-first semantic encoder for Kaggriculture observations."""
    def __init__(
        self,
        size: int = 1024,
        max_hands: int = 16,
        *,
        clock_schema: str = "legacy",
    ):
        if clock_schema not in {"legacy", "v4"}:
            raise ValueError("clock_schema must be 'legacy' or 'v4'")
        self.size = int(size)
        self.max_hands = int(max_hands)
        self.clock_schema = clock_schema

    @staticmethod
    def _get(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    @staticmethod
    def _clip(x, lo=0.0, hi=1.0):
        return float(max(lo, min(hi, float(x))))

    @classmethod
    def _qty(cls, value, scale=100.0):
        value = max(0.0, float(value or 0.0))
        return cls._clip(math.log1p(value) / math.log1p(scale))

    @classmethod
    def _money(cls, value):
        value = max(0.0, float(value or 0.0))
        return cls._clip(math.log1p(value) / math.log1p(200000.0))

    @classmethod
    def _coord(cls, value, board_size):
        if board_size <= 1:
            return 0.0
        return cls._clip(float(value) / float(board_size - 1)) * 2.0 - 1.0

    def _position(self, pos, board_size):
        pos = list(pos or [0, 0])
        x = pos[0] if len(pos) > 0 else 0
        y = pos[1] if len(pos) > 1 else 0
        return [self._coord(x, board_size), self._coord(y, board_size)]

    def _tile_features(self, tile, day):
        out = [0.0] * 21
        if tile is None:
            out[0] = 1.0
            return out
        if tile == "LOCKED":
            out[1] = 1.0
            return out
        kind = self._get(tile, "kind", "")
        kind_slot = {"WEED": 2, "PLANT": 3, "COOP": 4, "PASTURE": 5}.get(kind)
        if kind_slot is not None:
            out[kind_slot] = 1.0
        crop = self._get(tile, "crop")
        if crop in CROPS:
            out[6 + CROPS.index(crop)] = 1.0
        animal = self._get(tile, "animal")
        if animal in ANIMALS:
            out[11 + ANIMALS.index(animal)] = 1.0
        out[14] = self._clip(float(self._get(tile, "yield_units", 0) or 0) / 6.0)
        out[15] = 1.0 if self._get(tile, "watered_today", False) else 0.0
        out[16] = 1.0 if self._get(tile, "fed_today", False) else 0.0
        out[17] = 1.0 if self._get(tile, "cared_today", False) else 0.0
        out[18] = 1.0 if self._get(tile, "fertilizer_available", False) else 0.0
        stress = max(
            float(self._get(tile, "consecutive_unwatered", 0) or 0),
            float(self._get(tile, "consecutive_unfed", 0) or 0),
        )
        out[19] = self._clip(stress / 2.0)
        born = self._get(tile, "planted_day", self._get(tile, "placed_day", day))
        out[20] = self._clip((float(day) - float(born or 0)) / 30.0)
        return out

    def _farm_public(self, farm, day, board_size):
        out = [self._money(self._get(farm, "money", 0))]
        out.extend(self._position(self._get(farm, "farmer", [0, 0]), board_size))
        hands = list(self._get(farm, "hands", []) or [])[: self.max_hands]
        out.append(self._clip(len(hands) / max(1, self.max_hands)))
        for i in range(self.max_hands):
            if i < len(hands):
                out.append(1.0)
                out.extend(self._position(hands[i], board_size))
            else:
                out.extend((0.0, 0.0, 0.0))
        unlocked = set(self._get(farm, "unlocked_quadrants", []) or [])
        out.extend(1.0 if q in unlocked else 0.0 for q in QUADRANTS)
        out.append(self._clip(float(self._get(farm, "hires_today", 0) or 0) / 16.0))

        tiles = list(self._get(farm, "tiles", []) or [])
        total = float(max(1, board_size * board_size))
        kinds = {k: 0.0 for k in ("EMPTY", "LOCKED", "WEED", "PLANT", "COOP", "PASTURE")}
        crop_stats = {c: [0.0, 0.0, 0.0, 0.0] for c in CROPS}
        animal_stats = {a: [0.0] * 6 for a in ANIMALS}
        for row in tiles:
            for tile in row:
                if tile is None:
                    kinds["EMPTY"] += 1.0; continue
                if tile == "LOCKED":
                    kinds["LOCKED"] += 1.0; continue
                kind = self._get(tile, "kind", "")
                if kind in kinds: kinds[kind] += 1.0
                crop = self._get(tile, "crop")
                if crop in crop_stats:
                    s = crop_stats[crop]; s[0] += 1.0
                    s[1] += float(self._get(tile, "yield_units", 0) or 0)
                    s[2] += 1.0 if self._get(tile, "watered_today", False) else 0.0
                    s[3] += float(self._get(tile, "consecutive_unwatered", 0) or 0)
                animal = self._get(tile, "animal")
                if animal in animal_stats:
                    s = animal_stats[animal]; s[0] += 1.0
                    s[1] += float(self._get(tile, "yield_units", 0) or 0)
                    s[2] += 1.0 if self._get(tile, "fed_today", False) else 0.0
                    s[3] += 1.0 if self._get(tile, "cared_today", False) else 0.0
                    s[4] += 1.0 if self._get(tile, "fertilizer_available", False) else 0.0
                    s[5] += float(self._get(tile, "consecutive_unfed", 0) or 0)
        out.extend(kinds[k] / total for k in kinds)
        for crop in CROPS:
            count, yield_units, watered, stress = crop_stats[crop]
            out.extend((count / total, yield_units / (total * 6.0), watered / total, stress / (total * 2.0)))
        for animal in ANIMALS:
            count, yield_units, fed, cared, fert, stress = animal_stats[animal]
            out.extend((count / total, yield_units / (total * 6.0), fed / total,
                        cared / total, fert / total, stress / (total * 2.0)))
        return out

    def _private_features(self, private):
        out = []
        shed = self._get(private, "shed", {}) or {}
        seeds = self._get(private, "seeds", {}) or {}
        out.extend(self._qty(self._get(shed, item, 0), 100) for item in SHELF_ITEMS)
        out.extend(self._qty(self._get(seeds, crop, 0), 100) for crop in CROPS)
        inventories = list(self._get(private, "inventories", []) or [])
        for i in range(1 + self.max_hands):
            present = i < len(inventories)
            out.append(1.0 if present else 0.0)
            inv = inventories[i] if present else {}
            out.extend(self._qty(self._get(inv, item, 0), 20) for item in SHELF_ITEMS)
        return out

    def _tile_at(self, farm, pos):
        tiles = list(self._get(farm, "tiles", []) or [])
        pos = list(pos or [0, 0])
        if len(pos) < 2:
            return None
        x, y = int(pos[0]), int(pos[1])
        if y < 0 or y >= len(tiles):
            return None
        row = tiles[y]
        if x < 0 or x >= len(row):
            return None
        return row[x]

    def encode(self, observation, configuration=None):
        player = int(self._get(observation, "player", 0) or 0)
        farms = list(self._get(observation, "farms", []) or [])
        if not farms:
            return np.zeros(self.size, dtype=np.float32)
        player = max(0, min(player, len(farms) - 1))
        clock = resolve_clock(observation, configuration)
        day, hour = clock.day, clock.hour
        board_size = len(self._get(farms[player], "tiles", []) or []) or 10
        clock_values = dict(zip(CLOCK_FEATURES, clock.features()))
        if len(clock_values) != len(CLOCK_FEATURES):
            raise RuntimeError("clock feature schema mismatch")
        # Preserve the original first five clock positions so legacy
        # checkpoints keep the exact feature layout they were trained on.
        out = [clock_values[name] for name in LEGACY_CLOCK_FEATURES]

        market = self._get(observation, "market", {}) or {}
        inventory = self._get(market, "inventory", {}) or {}
        prices = self._get(market, "prices", {}) or {}
        for item in PRODUCTS:
            inv = float(self._get(inventory, item, 10000) or 0)
            price = float(self._get(prices, item, 0) or 0)
            out.append(self._clip((inv - 8000.0) / 4000.0, -1.0, 1.0))
            out.append(self._clip(math.log1p(max(0.0, price)) / math.log1p(1000.0)))
            out.append(1.0 if price <= 1.0 else 0.0)
        town = self._get(observation, "town", {}) or {}
        unlocked_shops = list(self._get(town, "unlocked_shops", []) or [])
        for shop in SHOPS:
            out.append(self._clip(unlocked_shops.count(shop) / 8.0))

        farm_order = [player] + [i for i in range(len(farms)) if i != player]
        for idx in farm_order[:2]:
            out.extend(self._farm_public(farms[idx], day, board_size))
        if len(farm_order) < 2:
            out.extend([0.0] * len(self._farm_public({}, day, board_size)))

        private = self._get(observation, "private", {}) or {}
        out.extend(self._private_features(private))

        own = farms[player]
        unit_positions = [self._get(own, "farmer", [0, 0])]
        unit_positions.extend(list(self._get(own, "hands", []) or [])[: self.max_hands])
        while len(unit_positions) < 1 + self.max_hands:
            unit_positions.append(None)
        for pos in unit_positions:
            if pos is None:
                out.extend([0.0] * 21)
            else:
                out.extend(self._tile_features(self._tile_at(own, pos), day))

        if self.clock_schema == "v4":
            # Use padding space for new V4 time signals instead of shifting
            # any legacy market/farm/private feature index.
            out.extend(
                clock_values[name] for name in V4_CLOCK_EXTRA_FEATURES
            )

        arr = np.zeros(self.size, dtype=np.float32)
        n = min(len(out), self.size)
        if n:
            arr[:n] = np.asarray(out[:n], dtype=np.float32)
        return arr
