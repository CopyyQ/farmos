from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

from .v2_effects import ActionEvidence, TransitionEffects, derive_effects

MOVE_OPS = {"NORTH", "SOUTH", "EAST", "WEST"}
SERVICE_OPS = {"WATER", "FEED", "CARE", "FERTILIZE", "COLLECT_FERTILIZER"}
PRODUCTION_OPS = {"PLANT", "BUILD_COOP", "BUILD_PASTURE", "DIG"}
BUY_OPS = {"BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL"}


@dataclass(frozen=True)
class EffectRecord:
    requested_action: dict[str, Any]
    transition: TransitionEffects
    evidence: tuple[ActionEvidence, ...]
    effective_families: tuple[str, ...]
    dawn_hand_expiration: int

    @property
    def effective(self) -> bool:
        return any(row.status == "confirmed" for row in self.evidence)

    @property
    def confirmed_count(self) -> int:
        return sum(row.status == "confirmed" for row in self.evidence)

    @property
    def failed_count(self) -> int:
        return sum(row.status == "failed" for row in self.evidence)

    @property
    def unconfirmed_count(self) -> int:
        return sum(row.status == "unconfirmed" for row in self.evidence)

    @property
    def effective_ops(self) -> tuple[str, ...]:
        return tuple(row.op for row in self.evidence if row.status == "confirmed")

    def to_model_effect(self) -> dict[str, Any]:
        value = asdict(self.transition)
        value["action_evidence"] = [asdict(row) for row in self.evidence]
        return value


def _farm(obs: dict[str, Any]) -> dict[str, Any]:
    player = int(obs.get("player", 0))
    farms = obs.get("farms") or []
    return farms[player] if 0 <= player < len(farms) else {}


def _actor_position(obs: dict[str, Any], actor: str):
    farm = _farm(obs)
    if actor == "farmer":
        return farm.get("farmer")
    if actor.startswith("hand:"):
        index = int(actor.split(":", 1)[1])
        hands = farm.get("hands") or []
        return hands[index] if index < len(hands) else None
    return None


def _inventory(obs: dict[str, Any], actor: str) -> dict[str, Any]:
    private = obs.get("private") or {}
    values = private.get("inventories") or []
    index = 0 if actor == "farmer" else 1 + int(actor.split(":", 1)[1])
    return values[index] if 0 <= index < len(values) and isinstance(values[index], dict) else {}

def _tile_at(obs: dict[str, Any], position):
    if not (isinstance(position, (list, tuple)) and len(position) >= 2):
        return None
    tiles = _farm(obs).get("tiles") or []
    x, y = int(position[0]), int(position[1])
    if 0 <= y < len(tiles) and 0 <= x < len(tiles[y]):
        return tiles[y][x]
    return None


def _command_for_actor(action: dict[str, Any], actor: str):
    if actor == "farmer":
        return action.get("farmer")
    if actor.startswith("hand:"):
        index = int(actor.split(":", 1)[1])
        hands = action.get("hands") or []
        return hands[index] if index < len(hands) else None
    return None


def _confirmed_from_tile(before, after, actor: str, op: str, command):
    position = _actor_position(before, actor)
    before_tile, after_tile = _tile_at(before, position), _tile_at(after, position)
    if op == "PLANT":
        item = command[1] if isinstance(command, list) and len(command) > 1 else None
        ok = isinstance(after_tile, dict) and after_tile.get("crop") == item and after_tile != before_tile
        return ok, {"item": item, "tile_changed": after_tile != before_tile}
    if op == "WATER":
        return (isinstance(after_tile, dict) and bool(after_tile.get("watered_today"))
                and not bool((before_tile or {}).get("watered_today"))), {"watered_today": True}
    if op == "FEED":
        return (isinstance(after_tile, dict) and bool(after_tile.get("fed_today"))
                and not bool((before_tile or {}).get("fed_today"))), {"fed_today": True}
    if op == "CARE":
        return (isinstance(after_tile, dict) and bool(after_tile.get("cared_today"))
                and not bool((before_tile or {}).get("cared_today"))), {"cared_today": True}
    if op == "FERTILIZE":
        before_until = int((before_tile or {}).get("fertilized_until_day", -1) or -1)
        after_until = int((after_tile or {}).get("fertilized_until_day", -1) or -1)
        return after_until > before_until, {"fertilized_until_day": after_until}
    if op == "BUILD_COOP":
        ok = isinstance(after_tile, dict) and after_tile.get("kind") == "COOP" and after_tile != before_tile
        return ok, {"tile_kind": "COOP"}
    if op == "BUILD_PASTURE":
        ok = isinstance(after_tile, dict) and after_tile.get("kind") == "PASTURE" and after_tile != before_tile
        return ok, {"tile_kind": "PASTURE"}
    if op == "DIG":
        return before_tile != after_tile, {"tile_changed": before_tile != after_tile}
    return False, {}


def _supplement_evidence(before, action, after, evidence):
    result = []
    day_changed = int(after.get("day", 0)) != int(before.get("day", 0))
    for row in evidence:
        if row.status != "unconfirmed" or row.actor.startswith("market:"):
            result.append(row); continue
        op = row.op
        if op == "COLLECT_FERTILIZER":
            b = int(_inventory(before, row.actor).get("FERTILIZER", 0) or 0)
            a = int(_inventory(after, row.actor).get("FERTILIZER", 0) or 0)
            ok, observed = a > b, {"fertilizer_inventory_delta": a - b}
        elif op in SERVICE_OPS | PRODUCTION_OPS:
            ok, observed = _confirmed_from_tile(
                before, after, row.actor, op, _command_for_actor(action, row.actor),
            )
        else:
            result.append(row); continue
        status = "unconfirmed" if day_changed else ("confirmed" if ok else "failed")
        result.append(ActionEvidence(row.actor, op, status, {**row.observed, **observed}))
    return tuple(result)

def _families(evidence: tuple[ActionEvidence, ...]) -> tuple[str, ...]:
    families = set()
    for row in evidence:
        if row.status != "confirmed":
            continue
        op = row.op
        if op in MOVE_OPS:
            families.add("movement")
        elif op == "HIRE":
            families.update(("acquisition", "hire"))
        elif op in BUY_OPS:
            families.update(("acquisition", "purchase"))
        elif op == "BUY_LAND":
            families.add("land_unlock")
        elif op in PRODUCTION_OPS:
            families.add("production")
        elif op in SERVICE_OPS:
            families.add("service")
        elif op == "HARVEST":
            families.add("harvest")
        elif op == "DROP":
            families.add("deposit")
        elif op == "PLACE":
            if bool(row.observed.get("animal_placed")):
                families.add("production")
            else:
                families.add("deposit")
        elif op == "PICKUP":
            families.add("cargo_pickup")
        elif op == "SELL":
            families.add("sale")
    return tuple(sorted(families))


class EffectTracker:
    def observe(self, previous_obs: dict[str, Any], requested_action: dict[str, Any],
                current_obs: dict[str, Any]) -> EffectRecord:
        transition = derive_effects(previous_obs, requested_action, current_obs)
        evidence = _supplement_evidence(
            previous_obs, requested_action, current_obs, transition.action_evidence,
        )
        dawn_expiration = max(0, -int(transition.hand_count_delta)) if transition.day_reset else 0
        return EffectRecord(
            requested_action=deepcopy(requested_action), transition=transition,
            evidence=evidence, effective_families=_families(evidence),
            dawn_hand_expiration=dawn_expiration,
        )
