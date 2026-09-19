from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import pyarrow.parquet as pq
import torch

from .v2_dataset import decode_zlib_json
from .v2_ledger import LAND_PRICES, SEED_COST, ShadowLedger
from .v3_3_schema import SHORT_ECONOMIC_HORIZONS
from .v2_tensorize import (
    EFFECT_FEATURES,
    V2Batch as StepBatch,
    _effect_tensor,
    collate_transitions,
    signed_log1p,
    stage0_code_hashes,
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_training_acceptance(stage0_marker: Path, stage1_marker: Path) -> dict[str, Any]:
    stage0_marker = Path(stage0_marker)
    stage1_marker = Path(stage1_marker)
    stage0 = _load_json(stage0_marker, "Stage 0 marker")
    stage1 = _load_json(stage1_marker, "Stage 1 marker")
    _require(stage0.get("accepted") is True, "Stage 0 is not accepted")
    _require(stage1.get("accepted") is True, "Stage 1 is not accepted")
    counters = stage0.get("counters") or {}
    _require(isinstance(counters, dict) and all(int(v) == 0 for v in counters.values()),
             "Stage 0 marker contains nonzero counters")

    live_root = stage0_marker.parent.parent
    artifacts = stage0.get("artifacts") or {}
    dataset = live_root / "transitions.parquet"
    snapshot = stage0_marker.parent / "live_top10_snapshot.json"
    corpus = stage0_marker.parent / "trusted_corpus_manifest.json"
    for key, path in (
        ("dataset_sha256", dataset),
        ("selection_snapshot_sha256", snapshot),
        ("trusted_corpus_manifest_sha256", corpus),
    ):
        if key in artifacts:
            _require(path.is_file(), f"missing Stage 0 artifact: {path}")
            actual = _sha256(path)
            label = "dataset SHA" if key == "dataset_sha256" else f"Stage 0 {key}"
            _require(actual == artifacts[key], f"{label} mismatch")
    current_code = stage0_code_hashes()
    for key, actual in current_code.items():
        if key in artifacts:
            _require(actual == artifacts[key], f"Stage 0 code hash mismatch: {key}")

    stage0_sha = _sha256(stage0_marker)
    stage1_artifacts = stage1.get("artifacts") or {}
    _require(stage1_artifacts.get("stage0_marker_sha256") == stage0_sha,
             "Stage 1 does not bind the accepted Stage 0 marker")
    stage1_dir = stage1_marker.parent
    build_path = stage1_dir / "build_manifest.json"
    parity_path = stage1_dir / "parity_report.json"
    if "build_manifest_sha256" in stage1_artifacts:
        _require(build_path.is_file() and _sha256(build_path) == stage1_artifacts["build_manifest_sha256"],
                 "Stage 1 build manifest SHA mismatch")
    if "parity_report_sha256" in stage1_artifacts:
        _require(parity_path.is_file() and _sha256(parity_path) == stage1_artifacts["parity_report_sha256"],
                 "Stage 1 parity report SHA mismatch")
    if "archive_sha256" in stage1_artifacts:
        build = _load_json(build_path, "Stage 1 build manifest")
        archive_value = build.get("archive", "policy_init.npz")
        archive = Path(archive_value)
        if not archive.is_absolute():
            root = stage1_dir.parents[1] if len(stage1_dir.parents) >= 2 else stage1_dir
            candidate = root / archive
            archive = candidate if candidate.is_file() else stage1_dir / Path(archive_value).name
        _require(archive.is_file() and _sha256(archive) == stage1_artifacts["archive_sha256"],
                 "Stage 1 archive SHA mismatch")
    return {
        "accepted": True,
        "stage0_marker_sha256": stage0_sha,
        "stage1_marker_sha256": _sha256(stage1_marker),
        "dataset_sha256": artifacts.get("dataset_sha256", _sha256(dataset)),
    }


def verify_effective_action_sidecar(dataset_path: Path) -> dict[str, Any]:
    dataset_path = Path(dataset_path)
    sidecar = dataset_path.with_name("effective_actions.parquet")
    summary_path = dataset_path.parent / "manifests" / "effective_actions_summary.json"
    if not sidecar.is_file():
        raise RuntimeError(f"missing effective-action sidecar: {sidecar}")
    summary = _load_json(summary_path, "effective-action summary")
    dataset_sha = _sha256(dataset_path)
    sidecar_sha = _sha256(sidecar)
    _require(
        summary.get("source_dataset_sha256") == dataset_sha,
        "effective-action sidecar is not bound to the accepted dataset",
    )
    _require(
        summary.get("effective_actions_sha256") == sidecar_sha,
        "effective-action sidecar SHA mismatch",
    )
    row_count = int(pq.ParquetFile(sidecar).metadata.num_rows)
    _require(
        int(summary.get("rows", -1)) == row_count and row_count > 0,
        "effective-action sidecar row count mismatch",
    )
    corpus = dataset_path.parent / "manifests" / "trusted_corpus_manifest.json"
    _require(corpus.is_file(), f"missing trusted corpus manifest: {corpus}")
    _require(
        summary.get("source_corpus_manifest_sha256") == _sha256(corpus),
        "effective-action sidecar corpus-manifest SHA mismatch",
    )
    _require(
        int(summary.get("verified_replay_files", -1)) > 0,
        "effective-action sidecar has no verified replay files",
    )
    _require(
        len(str(summary.get("builder_code_sha256", ""))) == 64,
        "effective-action sidecar has invalid builder code SHA",
    )
    _require(
        int(summary.get("verified_non_eod_transitions", -1)) > 0,
        "effective-action sidecar was not engine-verified",
    )
    _require(
        bool(str(summary.get("engine_module_version", ""))),
        "effective-action sidecar has no engine version",
    )
    _require(
        len(str(summary.get("engine_source_sha256", ""))) == 64,
        "effective-action sidecar has invalid engine source SHA",
    )
    return {
        "path": sidecar,
        "sha256": sidecar_sha,
        "engine_module_version": str(summary.get("engine_module_version", "")),
        "engine_source_sha256": str(summary.get("engine_source_sha256", "")),
        "builder_code_sha256": str(summary.get("builder_code_sha256", "")),
        "source_corpus_manifest_sha256": str(
            summary.get("source_corpus_manifest_sha256", "")
        ),
        "verified_replay_files": int(summary["verified_replay_files"]),
        "verified_non_eod_transitions": int(summary["verified_non_eod_transitions"]),
    }


FUTURE_HORIZONS = (1, 4, 24, 48)
FUTURE_RESOURCE_FEATURES = (
    "wheat_demand", "fertilizer_demand", "seed_demand", "shed_pressure",
    "hire_count", "sale_volume", "production_volume", "purchase_volume",
)
UNIT_TASK_DIM = 16
OPPONENT_EFFECT_DIM = 16


def _market_op(slot: dict[str, Any]) -> str:
    kind = str(slot.get("kind", "ORDER"))
    if kind in {"STOP_QUEUE", "NOP_SLOT"}:
        return kind
    return str(slot.get("op", "NOP_SLOT"))


def _resource_row_values(row: dict[str, Any]) -> list[float]:
    values = [0.0] * len(FUTURE_RESOURCE_FEATURES)
    action = row.get("canonical_action") or {}
    units = [action.get("farmer") or {}, *(action.get("hands") or [])]
    for command in units:
        op = str(command.get("op", "PASS"))
        item = command.get("item")
        quantity = command.get("quantity")
        q = 1 if quantity is None else max(0, int(quantity))
        if op in {"FEED", "PICKUP"} and item == "WHEAT":
            values[0] += q
        if op == "FERTILIZE":
            values[1] += 1
        if op == "PLANT":
            values[2] += 1
    for slot in action.get("market") or []:
        op = _market_op(slot)
        quantity = slot.get("quantity")
        q = 1 if quantity is None else max(0, int(quantity))
        if op == "HIRE":
            values[4] += 1
        elif op == "SELL":
            values[5] += q
        elif op in {"BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL"}:
            values[7] += q
    effects = row.get("effects") or {}
    shed_delta = effects.get("shed_delta") or {}
    values[3] += sum(float(v or 0) for v in shed_delta.values())
    for evidence in effects.get("action_evidence") or []:
        if not isinstance(evidence, dict) or evidence.get("status") != "confirmed":
            continue
        if str(evidence.get("op")) == "HARVEST":
            inv = (evidence.get("observed") or {}).get("inventory_delta") or {}
            values[6] += sum(max(0, int(v or 0)) for v in inv.values())
    return values


def _resource_summary(rows: Sequence[dict[str, Any]]) -> list[float]:
    values = [0.0] * len(FUTURE_RESOURCE_FEATURES)
    for row in rows:
        current = _resource_row_values(row)
        for index, value in enumerate(current):
            values[index] += value
    return [signed_log1p(value) for value in values]


def _future_resource_targets(rows: Sequence[dict[str, Any]]) -> list[list[float]]:
    width = len(FUTURE_RESOURCE_FEATURES)
    prefix: list[list[float]] = [[0.0] * width]
    for row in rows:
        current = _resource_row_values(row)
        previous = prefix[-1]
        prefix.append([
            previous[index] + current[index] for index in range(width)
        ])

    out: list[list[float]] = []
    total_rows = len(rows)
    for index in range(total_rows):
        features: list[float] = []
        start = index + 1
        for horizon in FUTURE_HORIZONS:
            end = min(total_rows, start + horizon)
            values = [
                prefix[end][feature] - prefix[start][feature]
                for feature in range(width)
            ]
            features.extend(signed_log1p(value) for value in values)
        out.append(features)
    return out


def _opponent_effect_target(effects: dict[str, Any]) -> list[float]:
    public = effects.get("opponent_public") or {}
    money = float(public.get("money_delta", 0) or 0)
    hands = float(public.get("hand_count_delta", 0) or 0)
    delta = public.get("farmer_position_delta") or [0, 0]
    dx = float(delta[0]) if len(delta) > 0 else 0.0
    dy = float(delta[1]) if len(delta) > 1 else 0.0
    moved = float(dx != 0.0 or dy != 0.0)
    activity = float(money != 0.0 or hands != 0.0 or moved or bool(public.get("grid_changed", False)))
    return [
        signed_log1p(money), signed_log1p(hands), dx, dy,
        float(bool(public.get("grid_changed", False))),
        float(str(public.get("confidence", "")) == "inferred"),
        float(money > 0), float(money < 0), float(hands > 0), float(hands < 0),
        moved, activity, float(bool(effects.get("day_changed", False))),
        float(bool(effects.get("day_reset", False))),
        signed_log1p(abs(money)), signed_log1p(abs(hands)),
    ]


_MOVE_DELTA = {"NORTH": (0.0, -1.0), "SOUTH": (0.0, 1.0),
               "EAST": (1.0, 0.0), "WEST": (-1.0, 0.0)}
_SERVICE_OPS = {"WATER", "FEED", "CARE", "FERTILIZE", "COLLECT_FERTILIZER"}
_BUILD_OPS = {"BUILD_COOP", "BUILD_PASTURE", "DIG"}


def _unit_command(row: dict[str, Any], actor_index: int) -> dict[str, Any] | None:
    action = row.get("canonical_action") or {}
    if actor_index == 0:
        return action.get("farmer") or {"op": "PASS"}
    hands = list(action.get("hands") or [])
    index = actor_index - 1
    return hands[index] if index < len(hands) else None


def _unit_task_vector(rows: Sequence[dict[str, Any]], row_index: int,
                      actor_index: int) -> list[float]:
    features = [0.0] * UNIT_TASK_DIM
    found: tuple[int, dict[str, Any]] | None = None
    for future_index in range(row_index + 1, len(rows)):
        if actor_index > 0 and bool((rows[future_index - 1].get("effects") or {}).get("day_reset", False)):
            break
        command = _unit_command(rows[future_index], actor_index)
        if command is None:
            break
        if str(command.get("op", "PASS")) != "PASS":
            found = (future_index, command)
            break
    if found is None:
        features[1] = 1.0
        return features
    future_index, command = found
    op = str(command.get("op", "PASS"))
    features[0] = 1.0
    if op in _MOVE_DELTA:
        features[2] = 1.0
    elif op == "PICKUP":
        features[3] = 1.0
    elif op in {"PLACE", "DROP"}:
        features[4] = 1.0
    elif op == "PLANT":
        features[5] = 1.0
    elif op in _SERVICE_OPS:
        features[6] = 1.0
    elif op == "HARVEST":
        features[7] = 1.0
    elif op in _BUILD_OPS:
        features[8] = 1.0
    else:
        features[9] = 1.0
    features[10] = signed_log1p(future_index - row_index)
    dx, dy = _MOVE_DELTA.get(op, (0.0, 0.0))
    features[11], features[12] = dx, dy
    features[13] = float(op in {"PLACE", "DROP"})
    features[14] = float(op == "PICKUP")
    features[15] = float(op in {"PLANT", "HARVEST"} or op in _SERVICE_OPS or op in _BUILD_OPS)
    return features


def _unit_task_from_event(row_index: int,
                          event: tuple[int, dict[str, Any]] | None) -> list[float]:
    features = [0.0] * UNIT_TASK_DIM
    if event is None:
        features[1] = 1.0
        return features
    future_index, command = event
    op = str(command.get("op", "PASS"))
    features[0] = 1.0
    category_index = 2 if op in _MOVE_DELTA else 3 if op == "PICKUP" else 4 if op in {"PLACE", "DROP"} else 5 if op == "PLANT" else 6 if op in _SERVICE_OPS else 7 if op == "HARVEST" else 8 if op in _BUILD_OPS else 9
    features[category_index] = 1.0
    features[10] = signed_log1p(future_index - row_index)
    features[11], features[12] = _MOVE_DELTA.get(op, (0.0, 0.0))
    features[13] = float(op in {"PLACE", "DROP"})
    features[14] = float(op == "PICKUP")
    features[15] = float(op in {"PLANT", "HARVEST"} or op in _SERVICE_OPS or op in _BUILD_OPS)
    return features


def _unit_task_targets(rows: Sequence[dict[str, Any]]) -> list[list[list[float]]]:
    max_units = max((1 + len((row.get("canonical_action") or {}).get("hands") or []) for row in rows), default=1)
    next_event: list[tuple[int, dict[str, Any]] | None] = [None] * max_units
    targets: list[list[list[float]] | None] = [None] * len(rows)
    for index in range(len(rows) - 1, -1, -1):
        row = rows[index]
        unit_count = 1 + len((row.get("canonical_action") or {}).get("hands") or [])
        reset = bool((row.get("effects") or {}).get("day_reset", False))
        current_targets = []
        for actor_index in range(unit_count):
            event = None if actor_index > 0 and reset else next_event[actor_index]
            current_targets.append(_unit_task_from_event(index, event))
        targets[index] = current_targets
        for actor_index in range(max_units):
            if actor_index >= unit_count:
                next_event[actor_index] = None
                continue
            if actor_index > 0 and reset:
                next_event[actor_index] = None
            command = _unit_command(row, actor_index)
            if command is None:
                next_event[actor_index] = None
            elif str(command.get("op", "PASS")) != "PASS":
                next_event[actor_index] = (index, command)
    return [value or [] for value in targets]


def _dictish(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "__dict__"):
        return vars(value)
    return {}


def _short_economic_snapshot(row: dict[str, Any]) -> tuple[float, float]:
    state = _row_state_for_supervision(row)
    own = _dictish(
        state.get("own", {}) if isinstance(state, dict)
        else getattr(state, "own", {})
    )
    private = _dictish(
        state.get("private", {}) if isinstance(state, dict)
        else getattr(state, "private", {})
    )
    market = _dictish(
        state.get("market", {}) if isinstance(state, dict)
        else getattr(state, "market", {})
    )
    prices = _dictish(market.get("prices") or {})
    money = float(own.get("money", 0) or 0)

    inventory_value = 0.0
    shed = _dictish(private.get("shed") or {})
    for item, amount in shed.items():
        inventory_value += (
            max(0.0, float(amount or 0))
            * max(0.0, float(prices.get(item, 0) or 0))
        )

    inventories = list(private.get("inventories") or [])
    for inventory in inventories:
        for item, amount in _dictish(inventory).items():
            inventory_value += (
                max(0.0, float(amount or 0))
                * max(0.0, float(prices.get(item, 0) or 0))
            )

    seed_value = 0.0
    for crop, amount in _dictish(private.get("seeds") or {}).items():
        seed_value += (
            max(0.0, float(amount or 0))
            * max(0.0, float(SEED_COST.get(str(crop), 0) or 0))
        )
    land_count = len(own.get("unlocked_quadrants") or [])
    extra_land = max(0, land_count - 1)
    land_value = float(sum(LAND_PRICES[:extra_land]))
    return money, money + inventory_value + seed_value + land_value


def _short_economic_targets(
    rows: Sequence[dict[str, Any]],
) -> list[list[float]]:
    snapshots = [_short_economic_snapshot(row) for row in rows]
    out: list[list[float]] = []
    total = len(rows)
    for index, row in enumerate(rows):
        current_money, current_worth = snapshots[index]
        values: list[float] = []
        terminal_money = float(row.get("final_own_money", current_money) or 0)
        for horizon in SHORT_ECONOMIC_HORIZONS:
            future_index = index + int(horizon)
            if future_index < total:
                future_money, future_worth = snapshots[future_index]
            else:
                # The accepted corpus stores final cash but not a terminal
                # inventory valuation. Use cash for both terminal quantities;
                # this is conservative and avoids inventing liquidation value.
                future_money = terminal_money
                future_worth = terminal_money
            values.extend([
                signed_log1p(future_money - current_money),
                signed_log1p(future_worth - current_worth),
            ])
        out.append(values)
    return out


def _attach_auxiliary_targets(rows: list[dict[str, Any]]) -> None:
    future_targets = _future_resource_targets(rows)
    short_economic_targets = _short_economic_targets(rows)
    unit_targets = _unit_task_targets(rows)
    for index, row in enumerate(rows):
        effect = row.get("effects") or {}
        row["_effect_target"] = _effect_tensor(effect).tolist()
        row["_future_resource_target"] = future_targets[index]
        row["_short_economic_target"] = short_economic_targets[index]
        row["_opponent_effect_target"] = _opponent_effect_target(effect)
        row["_unit_task_target"] = unit_targets[index]

def _decode_json(value: Any, default):
    if value is None:
        return default
    if isinstance(value, str):
        return json.loads(value)
    return value


def _row_state_for_supervision(row: dict[str, Any]) -> Any:
    if "state" in row:
        return row["state"]
    payload = row.get("state_zlib")
    if payload is None:
        raise KeyError("transition row has no state/state_zlib")
    return decode_zlib_json(payload)


def _nop_market_slot() -> dict[str, Any]:
    return {
        "kind": "NOP_SLOT",
        "op": None,
        "item": None,
        "quantity": None,
        "raw": [],
    }


def _sanitize_executable_sell_targets(
    row: dict[str, Any], action: dict[str, Any],
) -> dict[str, Any]:
    """Convert requested SELL quantities into quantities the engine can commit.

    Kaggriculture accepts an arbitrary positive requested SELL quantity but commits
    units one-by-one until the shed is empty. Training on the raw request therefore
    teaches impossible oversells. This helper keeps the original dataset immutable
    while exposing engine-executable SELL supervision to the model.
    """
    ledger = ShadowLedger.from_state(_row_state_for_supervision(row))
    ledger.apply_unit("farmer", action.get("farmer") or {"op": "PASS"})
    for index, command in enumerate(action.get("hands") or []):
        ledger.apply_unit(f"hand:{index}", command)

    sanitized_market: list[dict[str, Any]] = []
    for slot, source in enumerate(action.get("market") or []):
        order = dict(source)
        kind = str(order.get("kind", "ORDER"))
        op = (
            kind
            if kind in {"STOP_QUEUE", "NOP_SLOT"}
            else str(order.get("op", "NOP_SLOT"))
        )
        if op == "SELL":
            item = str(order.get("item"))
            try:
                requested = int(order.get("quantity") or 0)
            except (TypeError, ValueError):
                requested = 0
            legal = ledger.legal_market_mask(slot, {})
            available = int(
                (legal.metadata.get("sell_max_by_item") or {}).get(item, 0)
            )
            executed = min(max(0, requested), max(0, available))
            if executed <= 0:
                order = _nop_market_slot()
            else:
                order["kind"] = "ORDER"
                order["op"] = "SELL"
                order["item"] = item
                order["quantity"] = executed
                order["raw"] = ["SELL", item, executed]
        sanitized_market.append(order)
        ledger.apply_market(order)

    sanitized = dict(action)
    sanitized["farmer"] = dict(action.get("farmer") or {"op": "PASS"})
    sanitized["hands"] = [dict(command) for command in action.get("hands") or []]
    sanitized["market"] = sanitized_market
    return sanitized


@dataclass(frozen=True)
class EpisodeSequence:
    episode_id: int
    seat: int
    team_id: int
    role: str
    split: str
    rows: tuple[dict[str, Any], ...]

    def __len__(self) -> int:
        return len(self.rows)


def balanced_episode_order(samples: Iterable[EpisodeSequence], seed: int, epoch: int) -> list[EpisodeSequence]:
    episodes = list(samples)
    if not episodes:
        return []
    grouped: dict[int, list[EpisodeSequence]] = {}
    for episode in episodes:
        grouped.setdefault(int(episode.team_id), []).append(episode)
    rng = random.Random(int(seed) + 1000003 * int(epoch))
    per_team = max(len(group) for group in grouped.values())
    draws: dict[int, list[EpisodeSequence]] = {}
    for team, group in sorted(grouped.items()):
        shuffled = list(group)
        rng.shuffle(shuffled)
        draws[team] = [shuffled[index % len(shuffled)] for index in range(per_team)]
    out: list[EpisodeSequence] = []
    teams = sorted(grouped)
    for index in range(per_team):
        round_teams = list(teams)
        rng.shuffle(round_teams)
        out.extend(draws[team][index] for team in round_teams)
    return out


@dataclass(frozen=True)
class SequenceChunk:
    episode_id: int
    seat: int
    rows: tuple[dict[str, Any], ...]
    episode_start: bool
    episode_end: bool

    @property
    def start_step(self) -> int:
        return int(self.rows[0]["step"])

    @property
    def end_step(self) -> int:
        return int(self.rows[-1]["step"])


class V2EpisodeDataset(Sequence[EpisodeSequence]):
    def __init__(
        self,
        dataset_path: Path,
        split: str,
        roles: set[str],
        *,
        effective_action_path: Path | None = None,
        require_effective_actions: bool = False,
    ):
        self.dataset_path = Path(dataset_path)
        self.split = str(split)
        self.roles = {str(role) for role in roles}
        if self.split not in {"train", "val", "test"}:
            raise ValueError(f"unknown split: {self.split}")
        if not self.roles:
            raise ValueError("roles must not be empty")
        if self.split == "train" and self.roles != {"active_best"}:
            raise RuntimeError("primary training split may use active_best rows only")
        if not self.dataset_path.is_file():
            raise FileNotFoundError(self.dataset_path)

        training_columns = (
            "episode_id", "seat", "step", "team_id", "role", "split",
            "state_zlib", "canonical_action_json", "effects_json",
            "final_own_money", "final_margin", "terminal_result",
        )
        available_columns = set(pq.ParquetFile(self.dataset_path).schema_arrow.names)
        selected_columns = [
            name for name in training_columns if name in available_columns
        ]
        required_columns = {
            "episode_id", "seat", "step", "team_id", "role", "split",
            "state_zlib", "canonical_action_json", "effects_json",
        }
        missing = sorted(required_columns - available_columns)
        if missing:
            raise RuntimeError(
                f"training dataset is missing required columns: {missing}"
            )
        table = pq.read_table(
            self.dataset_path,
            columns=selected_columns,
            filters=[("split", "=", self.split), ("role", "in", sorted(self.roles))],
        )
        rows = table.to_pylist()
        if any(str(row.get("role", "")) not in self.roles for row in rows):
            raise RuntimeError("dataset role filter returned an unexpected role")

        self.effective_action_path: Path | None = None
        if effective_action_path is not None:
            self.effective_action_path = Path(effective_action_path)
        elif require_effective_actions:
            self.effective_action_path = self.dataset_path.with_name(
                "effective_actions.parquet"
            )
        if self.effective_action_path is not None:
            if not self.effective_action_path.is_file():
                raise RuntimeError(
                    f"missing effective-action sidecar: {self.effective_action_path}"
                )
            effective_columns = {
                "episode_id", "seat", "step", "split", "role",
                "effective_action_json",
            }
            available_effective = set(
                pq.ParquetFile(self.effective_action_path).schema_arrow.names
            )
            missing_effective = sorted(effective_columns - available_effective)
            if missing_effective:
                raise RuntimeError(
                    "effective-action sidecar is missing required columns: "
                    f"{missing_effective}"
                )
            effective_rows = pq.read_table(
                self.effective_action_path,
                columns=sorted(effective_columns),
                filters=[
                    ("split", "=", self.split),
                    ("role", "in", sorted(self.roles)),
                ],
            ).to_pylist()
            effective_lookup: dict[tuple[int, int, int], str] = {}
            for effective_row in effective_rows:
                key = (
                    int(effective_row["episode_id"]),
                    int(effective_row["seat"]),
                    int(effective_row["step"]),
                )
                if key in effective_lookup:
                    raise RuntimeError(
                        f"duplicate effective-action transition key: {key}"
                    )
                effective_lookup[key] = str(
                    effective_row["effective_action_json"]
                )
            for row in rows:
                key = (
                    int(row["episode_id"]),
                    int(row["seat"]),
                    int(row["step"]),
                )
                if key not in effective_lookup:
                    raise RuntimeError(
                        f"missing effective action for transition: {key}"
                    )
                row["effective_action_json"] = effective_lookup[key]
            if len(effective_lookup) != len(rows):
                raise RuntimeError(
                    "effective-action sidecar/filter row count mismatch"
                )
        elif require_effective_actions:
            raise RuntimeError("effective-action supervision is required")

        grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault((int(row["episode_id"]), int(row["seat"])), []).append(row)
        episodes = []
        for key in sorted(grouped):
            episodes.append(self._prepare_episode(grouped[key]))
        self._episodes = tuple(episodes)
    @staticmethod
    def _prepare_episode(rows: list[dict[str, Any]]) -> EpisodeSequence:
        ordered = sorted(rows, key=lambda row: int(row["step"]))
        steps = [int(row["step"]) for row in ordered]
        if len(steps) != len(set(steps)):
            raise RuntimeError("duplicate step inside episode")
        if any(b != a + 1 for a, b in zip(steps, steps[1:])):
            raise RuntimeError("episode rows are not contiguous")
        prepared = []
        previous_action: Any = {}
        previous_effect: Any = {}
        for source in ordered:
            row = dict(source)
            requested_action = _decode_json(
                row.get("canonical_action_json"),
                row.get("canonical_action") or {},
            )
            if row.get("effective_action_json") is not None:
                action = _decode_json(row.get("effective_action_json"), {})
            else:
                action = _sanitize_executable_sell_targets(row, requested_action)
            effects = _decode_json(row.get("effects_json"), row.get("effects") or {})
            row["requested_canonical_action"] = requested_action
            row["canonical_action"] = action
            row["effects"] = effects
            # Parsed values are the training contract. Drop duplicate JSON blobs
            # to keep the 70k+ transition corpus from occupying RAM twice.
            row.pop("canonical_action_json", None)
            row.pop("effective_action_json", None)
            row.pop("effects_json", None)
            row["previous_action"] = previous_action
            row["previous_effect"] = previous_effect
            prepared.append(row)
            # At inference the recurrent context contains the action the
            # agent actually requested, while supervision should still target the
            # engine-effective action. Keep those concepts separate so temporal
            # training matches rollout behavior.
            previous_action = requested_action
            previous_effect = effects
        _attach_auxiliary_targets(prepared)
        first = prepared[0]
        return EpisodeSequence(
            episode_id=int(first["episode_id"]), seat=int(first["seat"]),
            team_id=int(first.get("team_id", 0)), role=str(first.get("role", "")),
            split=str(first.get("split", "")), rows=tuple(prepared),
        )

    def __len__(self) -> int:
        return len(self._episodes)

    def __getitem__(self, index):
        return self._episodes[index]

    def __iter__(self) -> Iterator[EpisodeSequence]:
        return iter(self._episodes)

    def iter_chunks(self, sequence_len: int) -> Iterator[SequenceChunk]:
        length = int(sequence_len)
        if length <= 0:
            raise ValueError("sequence_len must be positive")
        for episode in self._episodes:
            total = len(episode.rows)
            for start in range(0, total, length):
                rows = episode.rows[start:start + length]
                yield SequenceChunk(
                    episode_id=episode.episode_id,
                    seat=episode.seat,
                    rows=tuple(rows),
                    episode_start=(start == 0),
                    episode_end=(start + len(rows) == total),
                )


def _chunks(samples: Iterable[EpisodeSequence | SequenceChunk], sequence_len: int) -> list[SequenceChunk]:
    length = int(sequence_len)
    if length <= 0:
        raise ValueError("sequence_len must be positive")
    out: list[SequenceChunk] = []
    for sample in samples:
        if isinstance(sample, SequenceChunk):
            if len(sample.rows) > length:
                raise ValueError("pre-chunked sample exceeds sequence_len")
            out.append(sample)
            continue
        if not isinstance(sample, EpisodeSequence):
            raise TypeError(f"unsupported sequence sample: {type(sample)!r}")
        total = len(sample.rows)
        for start in range(0, total, length):
            rows = sample.rows[start:start + length]
            out.append(SequenceChunk(
                episode_id=sample.episode_id, seat=sample.seat, rows=tuple(rows),
                episode_start=(start == 0), episode_end=(start + len(rows) == total),
            ))
    return out


@dataclass
class V2Batch:
    flat: StepBatch
    chunks: tuple[SequenceChunk, ...]
    sequence_mask: torch.Tensor
    reset_mask: torch.Tensor
    done_mask: torch.Tensor
    row_slots: torch.Tensor
    terminal_money: torch.Tensor
    terminal_margin: torch.Tensor
    terminal_result: torch.Tensor
    auxiliary_targets: dict[str, torch.Tensor]

    def model_inputs(self) -> dict[str, Any]:
        inputs = dict(self.flat.model_inputs())
        inputs.update({
            "sequence_mask": self.sequence_mask,
            "reset_mask": self.reset_mask,
            "done_mask": self.done_mask,
            "row_slots": self.row_slots,
        })
        return inputs

    def targets(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "canonical_actions": self.flat.canonical_actions,
            "effects_targets": self.flat.effects_targets,
            "terminal_result": self.terminal_result,
        }
        result.update(self.auxiliary_targets)
        return result


def collate_v2_sequences(samples: Iterable[EpisodeSequence | SequenceChunk],
                         sequence_len: int) -> V2Batch:
    chunks = _chunks(samples, sequence_len)
    if not chunks:
        raise ValueError("cannot collate an empty sequence batch")
    length = int(sequence_len)
    flat_rows: list[dict[str, Any]] = []
    row_slots = torch.full((len(chunks), length), -1, dtype=torch.long)
    sequence_mask = torch.zeros((len(chunks), length), dtype=torch.bool)
    reset_mask = torch.zeros_like(sequence_mask)
    done_mask = torch.zeros_like(sequence_mask)
    terminal_money: list[float] = []
    terminal_margin: list[float] = []
    terminal_result: list[float] = []

    for batch_index, chunk in enumerate(chunks):
        for time_index, row in enumerate(chunk.rows):
            row_slots[batch_index, time_index] = len(flat_rows)
            sequence_mask[batch_index, time_index] = True
            flat_rows.append(dict(row))
            terminal_money.append(signed_log1p(row.get("final_own_money", 0)))
            terminal_margin.append(signed_log1p(row.get("final_margin", 0)))
            terminal_result.append(float(row.get("terminal_result", 0)))
        if chunk.episode_start and chunk.rows:
            reset_mask[batch_index, 0] = True
        if chunk.episode_end and chunk.rows:
            done_mask[batch_index, len(chunk.rows) - 1] = True

    flat = collate_transitions(flat_rows)
    effect_target = torch.tensor(
        [row["_effect_target"] for row in flat_rows], dtype=torch.float32
    )
    future_resource = torch.tensor(
        [row["_future_resource_target"] for row in flat_rows], dtype=torch.float32
    )
    short_economic = torch.tensor(
        [row["_short_economic_target"] for row in flat_rows], dtype=torch.float32
    )
    opponent_effect = torch.tensor(
        [row["_opponent_effect_target"] for row in flat_rows], dtype=torch.float32
    )
    unit_task = torch.zeros(
        (len(flat_rows), flat.own_units.shape[1], UNIT_TASK_DIM), dtype=torch.float32
    )
    for row_index, row in enumerate(flat_rows):
        current = torch.tensor(row["_unit_task_target"], dtype=torch.float32)
        if current.numel():
            unit_task[row_index, :current.shape[0]] = current
    terminal_money_tensor = torch.tensor(terminal_money, dtype=torch.float32)
    terminal_margin_tensor = torch.tensor(terminal_margin, dtype=torch.float32)
    auxiliary_targets = {
        "effect": effect_target,
        "future_resource": future_resource,
        "short_economic": short_economic,
        "unit_task": unit_task,
        "opponent_effect": opponent_effect,
        "terminal_money": terminal_money_tensor,
        "terminal_margin": terminal_margin_tensor,
    }
    flat.auxiliary_targets = auxiliary_targets
    flat.sample_weight = torch.ones(len(flat_rows), dtype=torch.float32)
    return V2Batch(
        flat=flat, chunks=tuple(chunks), sequence_mask=sequence_mask,
        reset_mask=reset_mask, done_mask=done_mask, row_slots=row_slots,
        terminal_money=terminal_money_tensor,
        terminal_margin=terminal_margin_tensor,
        terminal_result=torch.tensor(terminal_result, dtype=torch.float32),
        auxiliary_targets=auxiliary_targets,
    )
