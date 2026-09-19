from __future__ import annotations
import numpy as np
from .constants import UNIT_TO_ID, ID_TO_UNIT, MARKET_TO_ID, ID_TO_MARKET, ITEM_TO_ID, ID_TO_ITEM, MAX_QUANTITY


class ActionCodec:
    """Fixed-width token codec for farmer, hands, and ordered market slots."""
    def __init__(self, max_hands: int = 16, max_market_orders: int = 10):
        self.max_hands = int(max_hands)
        self.max_market_orders = int(max_market_orders)
        self.unit_slots = 1 + self.max_hands
        self.width = self.unit_slots * 3 + self.max_market_orders * 3

    def _unit_tokens(self, command):
        command = list(command or ["PASS"])
        op = UNIT_TO_ID.get(command[0], UNIT_TO_ID["PASS"])
        arg = ITEM_TO_ID.get(command[1], 0) if len(command) > 1 else 0
        qty = int(command[2]) if len(command) > 2 and isinstance(command[2], (int, float)) else 0
        return [op, arg, max(0, min(MAX_QUANTITY, qty))]

    def _market_tokens(self, order):
        order = list(order or [])
        if not order:
            return [MARKET_TO_ID["NONE"], 0, 0]
        op = MARKET_TO_ID.get(order[0], MARKET_TO_ID["NONE"])
        arg = ITEM_TO_ID.get(order[1], 0) if len(order) > 1 and isinstance(order[1], str) else 0
        qty = int(order[2]) if len(order) > 2 and isinstance(order[2], (int, float)) else 0
        return [op, arg, max(0, min(MAX_QUANTITY, qty))]

    def encode(self, action, hand_count: int):
        tokens = []
        tokens.extend(self._unit_tokens(action.get("farmer", ["PASS"])))
        hands = list(action.get("hands", []))[: self.max_hands]
        for i in range(self.max_hands):
            command = hands[i] if i < min(hand_count, len(hands)) else ["PASS"]
            tokens.extend(self._unit_tokens(command))
        market = list(action.get("market", []))[: self.max_market_orders]
        for i in range(self.max_market_orders):
            tokens.extend(self._market_tokens(market[i] if i < len(market) else []))
        return np.asarray(tokens, dtype=np.int16)

    def supervision_mask(self, action, hand_count: int):
        mask = np.zeros(self.width, dtype=np.float32)
        commands = [list(action.get("farmer", ["PASS"]))]
        hands = list(action.get("hands", []))
        for i in range(min(hand_count, self.max_hands)):
            commands.append(list(hands[i]) if i < len(hands) else ["PASS"])
        for i, command in enumerate(commands):
            j = i * 3
            mask[j] = 1.0
            op = command[0] if command else "PASS"
            if op in {"PICKUP", "PLACE", "PLANT"} and len(command) > 1:
                mask[j + 1] = 1.0
            if op in {"PICKUP", "PLACE"} and len(command) > 2:
                mask[j + 2] = 1.0
        market = list(action.get("market", []))[: self.max_market_orders]
        start = self.unit_slots * 3
        for i, order in enumerate(market):
            j = start + i * 3
            mask[j] = 1.0
            if order and order[0] in {"BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"}:
                if len(order) > 1:
                    mask[j + 1] = 1.0
                if len(order) > 2:
                    mask[j + 2] = 1.0
        if len(market) < self.max_market_orders:
            mask[start + len(market) * 3] = 1.0
        return mask

    def _decode_unit(self, tri):
        op = ID_TO_UNIT.get(int(tri[0]), "PASS")
        if op in {"PICKUP", "PLACE", "PLANT"}:
            item = ID_TO_ITEM.get(int(tri[1]))
            if item is None:
                return ["PASS"]
            if op in {"PICKUP", "PLACE"} and int(tri[2]) > 0:
                return [op, item, int(tri[2])]
            return [op, item]
        return [op]

    def _decode_market(self, tri):
        op = ID_TO_MARKET.get(int(tri[0]), "NONE")
        if op == "NONE":
            return None
        if op in {"HIRE", "BUY_LAND"}:
            return [op]
        item = ID_TO_ITEM.get(int(tri[1]))
        qty = max(1, int(tri[2]))
        if item is None:
            return None
        return [op, item, qty]

    def decode(self, encoded, observation):
        a = np.asarray(encoded).reshape(-1, 3)
        player = int(observation["player"])
        hand_count = len(observation["farms"][player]["hands"])
        farmer = self._decode_unit(a[0])
        hands = [self._decode_unit(a[1 + i]) for i in range(min(hand_count, self.max_hands))]
        start = self.unit_slots
        market = []
        for i in range(self.max_market_orders):
            order = self._decode_market(a[start + i])
            if order is None:
                break
            market.append(order)
        return {"farmer": farmer, "hands": hands, "market": market}

    def valid_shape(self, encoded) -> bool:
        return np.asarray(encoded).shape == (self.width,)
