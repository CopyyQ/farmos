from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Literal

from .constants import ANIMALS, CROPS, PRODUCTS


@dataclass(frozen=True)
class MarketEdit:
    kind: Literal["KEEP", "DROP", "REPLACE"]
    order: list | None = None


KEEP = MarketEdit("KEEP")


@dataclass(frozen=True)
class ResidualAction:
    market0: MarketEdit = KEEP
    market1: MarketEdit = KEEP
    route_override: int | None = None


def _positive_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def valid_market_order(order):
    if not isinstance(order, list) or not order:
        return False
    verb = order[0]
    if verb in ("PASS", "HIRE", "BUY_LAND"):
        return len(order) == 1
    if len(order) != 3 or not _positive_int(order[2]):
        return False
    item = order[1]
    if verb == "BUY_SEED":
        return item in CROPS
    if verb == "BUY_ANIMAL":
        return item in ANIMALS
    if verb in ("BUY_PRODUCT", "SELL"):
        return item in PRODUCTS
    return False


def apply_residual(base_action, residual: ResidualAction):
    out = copy.deepcopy(base_action)
    effective = [list(x) for x in out.get("market", []) if x]
    edits = (residual.market0, residual.market1)
    for idx in (1, 0):
        edit = edits[idx]
        if edit.kind == "DROP" and idx < len(effective):
            effective.pop(idx)
        elif edit.kind == "REPLACE" and valid_market_order(edit.order):
            if idx < len(effective):
                effective[idx] = list(edit.order)
            else:
                effective.append(list(edit.order))
    out["market"] = effective[:10]
    return out
