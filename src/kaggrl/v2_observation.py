from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any


@dataclass
class StructuredObservation:
    player: int
    step: int
    day: int
    hour: int
    own: dict[str, Any]
    rival: dict[str, Any]
    private: dict[str, Any]
    own_grid: list[list[Any]]
    rival_grid: list[list[Any]]
    own_units: tuple[dict[str, Any], ...]
    rival_units: tuple[dict[str, Any], ...]
    market: dict[str, Any]
    town: dict[str, Any]
    town_shops: tuple[str, ...]


def _farm(obs: dict[str, Any], player: int) -> dict[str, Any]:
    farms = obs.get("farms")
    if not isinstance(farms, list) or player < 0 or player >= len(farms):
        raise ValueError(f"invalid farms/player: player={player}")
    farm = farms[player]
    if not isinstance(farm, dict):
        raise ValueError(f"farm[{player}] is not a mapping")
    return farm


def _public_units(farm: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    units: list[dict[str, Any]] = [{
        "kind": "farmer",
        "index": 0,
        "position": deepcopy(farm.get("farmer", [0, 0])),
    }]
    for index, position in enumerate(farm.get("hands") or []):
        units.append({
            "kind": "hand",
            "index": index,
            "position": deepcopy(position),
        })
    return tuple(units)


def _own_units(farm: dict[str, Any], private: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    inventories = private.get("inventories") or []
    public = _public_units(farm)
    units: list[dict[str, Any]] = []
    for inventory_index, unit in enumerate(public):
        row = deepcopy(unit)
        row["inventory"] = deepcopy(inventories[inventory_index]) if inventory_index < len(inventories) else {}
        units.append(row)
    return tuple(units)


def public_opponent_view(obs: dict[str, Any]) -> dict[str, Any]:
    player = int(obs.get("player", 0))
    rival_player = 1 - player
    return deepcopy(_farm(obs, rival_player))


def normalize_observation(obs: dict[str, Any]) -> StructuredObservation:
    if not isinstance(obs, dict):
        raise TypeError("observation must be a dict")
    player = int(obs.get("player", 0))
    if player not in (0, 1):
        raise ValueError(f"unsupported player index: {player}")
    day = int(obs.get("day", 0))
    hour = int(obs.get("hour", 0))
    step = int(obs.get("step", day * 24 + hour))
    own = deepcopy(_farm(obs, player))
    rival = public_opponent_view(obs)
    private = deepcopy(obs.get("private") or {})
    market = deepcopy(obs.get("market") or {})
    town = deepcopy(obs.get("town") or {})
    own_grid = deepcopy(own.get("tiles") or [])
    rival_grid = deepcopy(rival.get("tiles") or [])
    return StructuredObservation(
        player=player, step=step, day=day, hour=hour,
        own=own, rival=rival, private=private,
        own_grid=own_grid, rival_grid=rival_grid,
        own_units=_own_units(own, private), rival_units=_public_units(rival),
        market=market, town=town,
        town_shops=tuple(deepcopy(town.get("unlocked_shops") or [])),
    )
