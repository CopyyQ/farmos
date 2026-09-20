from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from .clock import PHASE_NAMES, resolve_clock
from .macro_policy import MacroPolicy
from .observation import ObservationEncoder
from .v2_dataset import decode_zlib_json
from .v4_options import MARKET_MODES
from .v45_macro_data import load_v45_macro_data

SPEND_OPS = {
    "HIRE", "BUY_LAND", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL",
}


def structured_state_to_observation(state: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct a canonical raw-like observation from stored Stage-0 state."""

    own = deepcopy(state.get("own") or {})
    rival = deepcopy(state.get("rival") or {})
    return {
        "player": 0,
        "step": int(state.get("step", 0) or 0),
        "day": int(state.get("day", 0) or 0),
        "hour": int(state.get("hour", 0) or 0),
        "farms": [own, rival],
        "private": deepcopy(state.get("private") or {}),
        "market": deepcopy(state.get("market") or {}),
        "town": deepcopy(state.get("town") or {}),
    }


def _command_signature(command: Any) -> tuple:
    if not command:
        return ()
    if isinstance(command, dict):
        kind = command.get("kind")
        if kind in {"STOP_QUEUE", "NOP_SLOT"}:
            return (str(kind),)
        op = command.get("op")
        if not op:
            return ()
        return (
            str(op),
            None if command.get("item") is None else str(command.get("item")),
            None if command.get("quantity") is None
            else int(command.get("quantity")),
        )
    if isinstance(command, (list, tuple)):
        out = []
        for value in command:
            if isinstance(value, bool):
                out.append(value)
            elif isinstance(value, (int, np.integer)):
                out.append(int(value))
            elif value is None:
                out.append(None)
            else:
                out.append(str(value))
        return tuple(out)
    return (str(command),)


def action_signature(action: dict[str, Any]) -> tuple:
    farmer = _command_signature(action.get("farmer") or ["PASS"])
    hands = tuple(
        _command_signature(command)
        for command in (action.get("hands") or [])
    )
    market = tuple(
        signature
        for command in (action.get("market") or [])
        if (signature := _command_signature(command))
    )
    return farmer, hands, market


def _positional_distance(left: Sequence[tuple], right: Sequence[tuple]) -> float:
    total = 0.0
    width = max(len(left), len(right))
    for index in range(width):
        lrow = left[index] if index < len(left) else ()
        rrow = right[index] if index < len(right) else ()
        total += float(lrow != rrow)
    return total


def signature_distance(left: tuple, right: tuple) -> float:
    lf, lh, lm = left
    rf, rh, rm = right
    return (
        3.0 * float(lf != rf)
        + 1.0 * _positional_distance(lh, rh)
        + 2.0 * _positional_distance(lm, rm)
    )


def action_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    return signature_distance(action_signature(left), action_signature(right))


def build_route_signature_table(
    *,
    route_ids: Sequence[int] | None = None,
    actionable_steps: int = 719,
) -> dict[int, tuple[tuple, ...]]:
    routes, new_routes, old_routes = load_v45_macro_data()
    macro = MacroPolicy(routes, new_routes, old_routes)
    candidates = tuple(
        sorted(routes)
        if route_ids is None
        else [int(x) for x in route_ids]
    )
    table: dict[int, tuple[tuple, ...]] = {}
    for route_id in candidates:
        signatures = []
        for step in range(int(actionable_steps)):
            day, hour = divmod(step, 24)
            obs = {
                "step": step,
                "day": day,
                "hour": hour,
                "town": {"unlocked_shops": []},
            }
            signatures.append(
                action_signature(macro.action_for_route(obs, route_id))
            )
        table[int(route_id)] = tuple(signatures)
    return table


@dataclass(frozen=True)
class RouteLabel:
    route_id: int
    confidence: float
    best_score: float
    second_score: float


def route_label_for_horizon(
    observations: Sequence[dict[str, Any]],
    teacher_actions: Sequence[dict[str, Any]],
    *,
    route_ids: Sequence[int] | None = None,
    signature_table: dict[int, tuple[tuple, ...]] | None = None,
) -> RouteLabel:
    if not observations or len(observations) != len(teacher_actions):
        raise ValueError("route horizon requires aligned non-empty rows")

    if signature_table is None:
        signature_table = build_route_signature_table(route_ids=route_ids)
    candidates = tuple(
        sorted(signature_table)
        if route_ids is None
        else [int(x) for x in route_ids]
    )
    if not candidates:
        raise ValueError("no route candidates")

    teacher_signatures = [action_signature(action) for action in teacher_actions]
    steps = [int(obs.get("step", 0) or 0) for obs in observations]

    scores: list[tuple[float, int]] = []
    for route_id in candidates:
        signatures = signature_table[int(route_id)]
        score = 0.0
        for offset, (step, teacher_signature) in enumerate(
            zip(steps, teacher_signatures)
        ):
            weight = 1.0 / (1.0 + 0.15 * offset)
            predicted_signature = signatures[step]
            score += weight * signature_distance(
                predicted_signature,
                teacher_signature,
            )
        scores.append((score, int(route_id)))
    scores.sort(key=lambda row: (row[0], row[1]))
    best_score, best_route = scores[0]
    second_score = scores[1][0] if len(scores) > 1 else best_score
    confidence = (
        max(
            0.0,
            min(
                1.0,
                (second_score - best_score) / max(1.0, second_score),
            ),
        )
        if len(scores) > 1
        else 0.0
    )
    return RouteLabel(
        route_id=best_route,
        confidence=confidence,
        best_score=float(best_score),
        second_score=float(second_score),
    )


def route_labels_for_episode(
    observations: Sequence[dict[str, Any]],
    teacher_actions: Sequence[dict[str, Any]],
    *,
    horizon: int,
    route_ids: Sequence[int],
    signature_table: dict[int, tuple[tuple, ...]],
    restrict_compatible: bool = False,
) -> list[RouteLabel]:
    if len(observations) != len(teacher_actions) or not observations:
        raise ValueError("route episode requires aligned non-empty rows")
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    candidates = tuple(int(value) for value in route_ids)
    candidate_index = {
        route_id: index for index, route_id in enumerate(candidates)
    }
    routes, new_routes, old_routes = load_v45_macro_data()
    macro = MacroPolicy(routes, new_routes, old_routes)
    teacher_signatures = [
        action_signature(action) for action in teacher_actions
    ]
    steps = np.asarray(
        [int(obs.get("step", 0) or 0) for obs in observations],
        dtype=np.int64,
    )
    distance = np.empty(
        (len(observations), len(candidates)),
        dtype=np.float32,
    )
    for route_index, route_id in enumerate(candidates):
        route_signatures = signature_table[route_id]
        distance[:, route_index] = np.asarray([
            signature_distance(
                route_signatures[int(step)],
                teacher_signatures[row_index],
            )
            for row_index, step in enumerate(steps)
        ], dtype=np.float32)

    weights = 1.0 / (
        1.0 + 0.15 * np.arange(horizon, dtype=np.float32)
    )
    labels: list[RouteLabel] = []
    for index, observation in enumerate(observations):
        end = min(len(observations), index + horizon)
        width = end - index
        scores = (
            distance[index:end]
            * weights[:width, None]
        ).sum(axis=0)

        if restrict_compatible:
            compatible = tuple(
                route_id
                for route_id in macro.compatible_route_ids(observation)
                if route_id in candidate_index
            )
            if not compatible:
                compatible = (macro.route_id(observation),)
        else:
            compatible = candidates
        compatible_indices = np.asarray(
            [candidate_index[int(route_id)] for route_id in compatible],
            dtype=np.int64,
        )
        local_scores = scores[compatible_indices]
        local_order = np.argsort(local_scores, kind="stable")
        best_local = int(local_order[0])
        best_index = int(compatible_indices[best_local])
        best_score = float(scores[best_index])

        if len(local_order) == 1:
            second_score = best_score
            # A forced route is not a learned strategic choice.
            confidence = 0.0
        else:
            second_local = int(local_order[1])
            second_index = int(compatible_indices[second_local])
            second_score = float(scores[second_index])
            confidence = max(
                0.0,
                min(
                    1.0,
                    (second_score - best_score)
                    / max(1.0, second_score),
                ),
            )
        labels.append(RouteLabel(
            route_id=candidates[best_index],
            confidence=confidence,
            best_score=best_score,
            second_score=second_score,
        ))
    return labels


def market_mode_label(
    observation: dict[str, Any],
    teacher_action: dict[str, Any],
    base_action: dict[str, Any],
) -> str:
    teacher_orders = [
        command for command in (teacher_action.get("market") or [])
        if _command_signature(command)
    ]
    teacher_ops = {
        _command_signature(command)[0]
        for command in teacher_orders
        if _command_signature(command)
    }
    has_spend = bool(teacher_ops & SPEND_OPS)
    has_sell = "SELL" in teacher_ops
    if has_spend:
        return "KEEP_ROUTE"

    if has_sell:
        clock = resolve_clock(observation)
        shed = (observation.get("private") or {}).get("shed") or {}
        known_total = sum(max(0, int(v or 0)) for v in shed.values())
        sold = 0
        for command in teacher_orders:
            sig = _command_signature(command)
            if len(sig) >= 3 and sig[0] == "SELL":
                sold += max(0, int(sig[2] or 0))
        if (
            clock.phase_index == len(PHASE_NAMES) - 1
            and known_total > 0
            and sold / max(1, known_total) >= 0.50
        ):
            return "LIQUIDATE_SHED"
        return "NO_SPEND"

    base_ops = {
        _command_signature(command)[0]
        for command in (base_action.get("market") or [])
        if _command_signature(command)
    }
    return "NO_SPEND" if base_ops & SPEND_OPS else "KEEP_ROUTE"


@dataclass(frozen=True)
class V4OptionRow:
    episode_id: int
    seat: int
    step: int
    obs_f16: bytes
    route_id: int
    route_mask_bits: int
    route_confidence: float
    market_mode_id: int
    phase_id: int
    step_norm: float
    remaining_norm: float
    final_own_money: float
    final_rival_money: float
    final_margin: float
    terminal_result: int


def encode_option_rows(
    rows: Sequence[dict[str, Any]],
    *,
    horizon: int = 8,
    signature_table: dict[int, tuple[tuple, ...]] | None = None,
) -> list[V4OptionRow]:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    routes, new_routes, old_routes = load_v45_macro_data()
    route_ids = tuple(sorted(routes))
    route_bit = {
        route_id: index for index, route_id in enumerate(route_ids)
    }
    macro = MacroPolicy(routes, new_routes, old_routes)
    if signature_table is None:
        signature_table = build_route_signature_table(route_ids=route_ids)
    encoder = ObservationEncoder(clock_schema="v4")
    mode_to_id = {name: index for index, name in enumerate(MARKET_MODES)}

    decoded: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in rows:
        state = (
            decode_zlib_json(row["state_zlib"])
            if isinstance(row.get("state_zlib"), (bytes, bytearray))
            else deepcopy(row["state"])
        )
        obs = structured_state_to_observation(state)
        action = row.get("raw_action")
        if action is None:
            action = json.loads(row["raw_action_json"])
        decoded.append((obs, action))

    route_labels = route_labels_for_episode(
        [row[0] for row in decoded],
        [row[1] for row in decoded],
        horizon=horizon,
        route_ids=route_ids,
        signature_table=signature_table,
        restrict_compatible=True,
    )

    out: list[V4OptionRow] = []
    for index, row in enumerate(rows):
        obs, teacher = decoded[index]
        route_label = route_labels[index]

        base_route = macro.route_id(obs)
        base_action = macro.action_for_route(obs, base_route)
        market_mode = market_mode_label(obs, teacher, base_action)
        clock = resolve_clock(obs)
        feature = encoder.encode(obs).astype(np.float16, copy=False)
        last_step = max(1, clock.episode_steps - 1)
        compatible = macro.compatible_route_ids(obs)
        route_mask_bits = 0
        for route_id in compatible:
            bit = route_bit.get(int(route_id))
            if bit is not None:
                route_mask_bits |= 1 << bit
        label_bit = route_bit[int(route_label.route_id)]
        if not (route_mask_bits & (1 << label_bit)):
            raise RuntimeError(
                f"route label outside compatibility mask: "
                f"step={clock.step} route={route_label.route_id} "
                f"compatible={compatible}"
            )

        out.append(V4OptionRow(
            episode_id=int(row["episode_id"]),
            seat=int(row["seat"]),
            step=int(row["step"]),
            obs_f16=feature.tobytes(order="C"),
            route_id=int(route_label.route_id),
            route_mask_bits=int(route_mask_bits),
            route_confidence=float(route_label.confidence),
            market_mode_id=int(mode_to_id[market_mode]),
            phase_id=int(clock.phase_index),
            step_norm=float(clock.step / last_step),
            remaining_norm=float(clock.remaining_steps / last_step),
            final_own_money=float(row["final_own_money"]),
            final_rival_money=float(row["final_rival_money"]),
            final_margin=float(row["final_margin"]),
            terminal_result=int(row["terminal_result"]),
        ))
    return out


def contiguous_window_starts(
    steps: Sequence[int],
    sequence_len: int,
) -> tuple[int, ...]:
    if sequence_len <= 0:
        raise ValueError("sequence_len must be positive")
    values = [int(x) for x in steps]
    if len(values) < sequence_len:
        return ()
    for left, right in zip(values, values[1:]):
        if right != left + 1:
            raise ValueError(
                f"non-contiguous strategic sequence: {left}->{right}"
            )
    starts = list(range(0, len(values) - sequence_len + 1, sequence_len))
    last = len(values) - sequence_len
    if not starts or starts[-1] != last:
        starts.append(last)
    return tuple(starts)
