from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Iterable


from kaggrl.constants import UNIT_OPS
from kaggrl.v2_action_schema import parse_raw_action
from kaggrl.v2_effect_tracker import EffectTracker
from kaggrl.v2_ledger import MARKET_OPS, ShadowLedger
from kaggrl.v2_observation import normalize_observation


class RecoveryCollectionEmptyError(RuntimeError):
    def __init__(self, message: str, *, report_path: Path | None = None):
        super().__init__(message)
        self.report_path = report_path


def canonicalize_teacher_action(raw_action: dict[str, Any], observation: dict[str, Any]):
    player = int(observation.get("player", 0))
    farms = observation.get("farms") or []
    hand_count = len((farms[player] or {}).get("hands", []))
    joint = parse_raw_action(raw_action, hand_count)
    return {
        "farmer": asdict(joint.farmer),
        "hands": [asdict(item) for item in joint.hands],
        "market": [asdict(item) for item in joint.market],
    }


def _market_op(slot: dict[str, Any]) -> str:
    kind = str(slot.get("kind", "ORDER"))
    return kind if kind in {"STOP_QUEUE", "NOP_SLOT"} else str(slot.get("op", "NOP_SLOT"))



def _nop_market_slot() -> dict[str, Any]:
    return {
        "kind": "NOP_SLOT",
        "op": None,
        "item": None,
        "quantity": None,
        "raw": [],
    }


def _pass_unit() -> dict[str, Any]:
    return {
        "op": "PASS",
        "item": None,
        "quantity": None,
        "raw": ["PASS"],
    }


def _project_unit_command(
    ledger: ShadowLedger,
    actor: str,
    command: dict[str, Any],
) -> dict[str, Any]:
    op = str(command.get("op", "PASS"))
    legal = ledger.legal_unit_mask(actor, {})
    if op not in UNIT_OPS or not bool(legal.ops.get(op, False)):
        return _pass_unit()

    item = command.get("item")
    if op in {"PICKUP", "PLACE", "PLANT"}:
        choices = legal.items.get(op, {})
        if item is None or not bool(choices.get(str(item), False)):
            return _pass_unit()
        item = str(item)
    else:
        item = None

    quantity = None
    if op in {"PICKUP", "PLACE"}:
        bounds = (
            legal.metadata.get("unit_quantity_max_by_op_item") or {}
        ).get(op, {})
        maximum = max(0, int(bounds.get(item, 0) or 0))
        try:
            requested = int(command.get("quantity") or 1)
        except (TypeError, ValueError):
            requested = 1
        quantity = min(max(0, requested), maximum)
        if quantity <= 0:
            return _pass_unit()

    raw = [op]
    if item is not None:
        raw.append(item)
    if quantity is not None:
        raw.append(int(quantity))
    return {
        "op": op,
        "item": item,
        "quantity": quantity,
        "raw": raw,
    }


def _project_market_slot(
    ledger: ShadowLedger,
    slot_index: int,
    slot: dict[str, Any],
) -> dict[str, Any]:
    op = _market_op(slot)
    legal = ledger.legal_market_mask(slot_index, {})
    if op == "STOP_QUEUE":
        if bool(legal.ops.get("STOP_QUEUE", False)):
            return {
                "kind": "STOP_QUEUE",
                "op": None,
                "item": None,
                "quantity": None,
                "raw": [],
            }
        return _nop_market_slot()
    if op == "NOP_SLOT":
        return _nop_market_slot()
    if op not in MARKET_OPS or not bool(legal.ops.get(op, False)):
        return _nop_market_slot()

    if op in {"HIRE", "BUY_LAND"}:
        return {
            "kind": "ORDER",
            "op": op,
            "item": None,
            "quantity": None,
            "raw": [op],
        }

    item = slot.get("item")
    choices = legal.items.get(op, {})
    if item is None or not bool(choices.get(str(item), False)):
        return _nop_market_slot()
    item = str(item)
    bounds = (
        legal.metadata.get("market_quantity_max_by_op_item") or {}
    ).get(op, {})
    maximum = max(0, int(bounds.get(item, 0) or 0))
    try:
        requested = int(slot.get("quantity") or 0)
    except (TypeError, ValueError):
        requested = 0
    quantity = min(max(0, requested), maximum)
    if quantity <= 0:
        return _nop_market_slot()
    return {
        "kind": "ORDER",
        "op": op,
        "item": item,
        "quantity": int(quantity),
        "raw": [op, item, int(quantity)],
    }



def _unit_semantics(command: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(command.get("op", "PASS")),
        command.get("item"),
        command.get("quantity"),
    )


def _market_semantics(slot: dict[str, Any]) -> tuple[Any, ...]:
    op = _market_op(slot)
    return (
        op,
        None if op in {"STOP_QUEUE", "NOP_SLOT", "HIRE", "BUY_LAND"} else slot.get("item"),
        None if op in {"STOP_QUEUE", "NOP_SLOT", "HIRE", "BUY_LAND"} else slot.get("quantity"),
    )


def project_teacher_action_to_executable(
    action: dict[str, Any],
    state: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    """Project a teacher request onto actions the engine can execute.

    Kaggriculture silently no-ops many invalid unit/market requests. DAgger
    supervision should teach the effective action at the learner-visited state,
    not a route request that is invalid only because the teacher did not visit
    that state itself.
    """
    ledger = ShadowLedger.from_state(state)
    corrections = 0

    requested_farmer = dict(action.get("farmer") or _pass_unit())
    farmer = _project_unit_command(
        ledger, "farmer", requested_farmer,
    )
    corrections += int(
        _unit_semantics(farmer) != _unit_semantics(requested_farmer)
    )
    ledger.apply_unit("farmer", farmer)

    hands = []
    for index, source in enumerate(action.get("hands") or []):
        requested = dict(source)
        actor = f"hand:{index}"
        projected = _project_unit_command(ledger, actor, requested)
        corrections += int(
            _unit_semantics(projected) != _unit_semantics(requested)
        )
        ledger.apply_unit(actor, projected)
        hands.append(projected)

    market = []
    for slot_index, source in enumerate(action.get("market") or []):
        requested = dict(source)
        projected = _project_market_slot(
            ledger, slot_index, requested,
        )
        corrections += int(
            _market_semantics(projected) != _market_semantics(requested)
        )
        market.append(projected)
        ledger.apply_market(projected)
        if _market_op(projected) == "STOP_QUEUE":
            break

    return {
        "farmer": farmer,
        "hands": hands,
        "market": market,
    }, corrections


def validate_recovery_row(row: dict[str, Any]) -> bool:
    for key in ("teacher_id", "teacher_version", "supervision_kind"):
        if not str(row.get(key, "")).strip():
            raise ValueError(f"recovery row missing {key}")
    if row["supervision_kind"] not in {
        "expert", "accepted_policy", "teacher_demo", "smoke_only",
    }:
        raise ValueError("unknown recovery supervision_kind")
    state = row.get("state")
    action = row.get("canonical_action") or {}
    if not isinstance(state, dict):
        raise ValueError("recovery row missing structured state")
    ledger = ShadowLedger.from_state(state)
    farmer = action.get("farmer") or {"op": "PASS"}
    farmer_op = str(farmer.get("op", "PASS"))
    legal = ledger.legal_unit_mask("farmer", {})
    if farmer_op not in UNIT_OPS or not bool(legal.ops.get(farmer_op, False)):
        raise ValueError(f"illegal farmer op: {farmer_op}")
    ledger.apply_unit("farmer", farmer)
    for index, command in enumerate(action.get("hands") or []):
        actor = f"hand:{index}"
        op = str(command.get("op", "PASS"))
        legal = ledger.legal_unit_mask(actor, {})
        if op not in UNIT_OPS or not bool(legal.ops.get(op, False)):
            raise ValueError(f"illegal hand op: {actor}:{op}")
        ledger.apply_unit(actor, command)
    for slot_index, slot in enumerate(action.get("market") or []):
        op = _market_op(slot)
        legal = ledger.legal_market_mask(slot_index, {})
        if op not in MARKET_OPS or not bool(legal.ops.get(op, False)):
            raise ValueError(f"illegal market op: slot{slot_index}:{op}")
        ledger.apply_market(slot)
        if op == "STOP_QUEUE":
            break
    return True


def write_recovery_rows(rows: Iterable[dict[str, Any]], output_path: Path) -> Path:
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    accepted = []
    for row in rows:
        validate_recovery_row(row)
        accepted.append(row)
    if not accepted:
        raise ValueError("recovery corpus would be empty")
    with output.open("w", encoding="utf-8") as handle:
        for row in accepted:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return output


def read_recovery_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    for row in rows:
        validate_recovery_row(row)
    return rows


def _plain(value):
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    try:
        if hasattr(value, "items"):
            return {key: _plain(item) for key, item in value.items()}
    except Exception:
        pass
    return value




def _environment_errors(env) -> list[str]:
    raw = getattr(env, "logs", None)
    if not raw:
        return []
    stack = [raw]
    errors: list[str] = []
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, (list, tuple)):
            stack.extend(value)
        elif value:
            text = str(value)
            lowered = text.lower()
            if any(
                marker in lowered
                for marker in (
                    "traceback",
                    "exception",
                    "error",
                    "invalid action",
                    "timeout",
                )
            ):
                errors.append(text)
    return errors


def _sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _RecoveryCollectingAgent:
    def __init__(
        self, candidate, teacher, seat, episode_id, teacher_version, model_sha,
        *, teacher_id="starter", supervision_kind="smoke_only",
        strategy_slot: int | None = None,
        drop_projected_labels: bool = False,
    ):
        self.candidate = candidate
        self.teacher = teacher
        self.seat = int(seat)
        self.episode_id = int(episode_id)
        self.teacher_version = str(teacher_version)
        self.model_sha = str(model_sha)
        self.teacher_id = str(teacher_id)
        self.supervision_kind = str(supervision_kind)
        self.strategy_slot = (
            None if strategy_slot is None else int(strategy_slot)
        )
        self.drop_projected_labels = bool(drop_projected_labels)
        self.rows: list[dict[str, Any]] = []
        self.label_errors: list[dict[str, Any]] = []
        self.projection_corrections = 0
        self.projected_rows = 0
        self.dropped_projected_rows = 0

    def _record_label_error(self, stage: str, observation, error: Exception) -> None:
        obs_plain = _plain(observation)
        self.label_errors.append({
            "stage": str(stage),
            "step": int(obs_plain.get("step", -1)),
            "player": int(obs_plain.get("player", self.seat)),
            "error_type": type(error).__name__,
            "error": str(error),
        })

    def __call__(self, observation, configuration=None):
        # Learner rollout is authoritative for DAgger state visitation.
        # Teacher-label failures must never prevent the learner action from
        # reaching the environment.
        learner_action = self.candidate(observation, configuration)
        fixture = self.candidate.diagnostic_fixtures[-1]
        teacher_obs = deepcopy(observation)
        try:
            try:
                teacher_raw = self.teacher(teacher_obs, configuration)
            except TypeError as two_arg_error:
                try:
                    teacher_raw = self.teacher(teacher_obs)
                except TypeError:
                    raise two_arg_error
        except Exception as error:
            self._record_label_error("teacher_call", observation, error)
            return learner_action

        obs_plain = _plain(observation)
        try:
            canonical = canonicalize_teacher_action(
                _plain(teacher_raw), obs_plain,
            )
        except Exception as error:
            self._record_label_error("canonicalize", observation, error)
            return learner_action

        structured_state = deepcopy(fixture["structured_state"])
        try:
            projected, corrections = project_teacher_action_to_executable(
                canonical,
                structured_state,
            )
        except Exception as error:
            self._record_label_error(
                "project_effective_action", observation, error,
            )
            return learner_action
        self.projection_corrections += int(corrections)
        self.projected_rows += int(corrections > 0)
        if self.drop_projected_labels and int(corrections) > 0:
            self.dropped_projected_rows += 1
            return learner_action

        row = {
            "episode_id": self.episode_id,
            "seat": self.seat,
            "step": int(fixture["step"]),
            "state": structured_state,
            "canonical_action": projected,
            "previous_action": deepcopy(fixture.get("previous_action") or {}),
            "previous_effect": deepcopy(fixture.get("previous_effect") or {}),
            "effects": {},
            "final_own_money": 0, "final_margin": 0, "terminal_result": 0,
            "teacher_id": self.teacher_id,
            "teacher_version": self.teacher_version,
            "supervision_kind": self.supervision_kind,
            "learner_model_sha256": self.model_sha,
        }
        if self.strategy_slot is not None:
            row["strategy_slot"] = int(self.strategy_slot)
        try:
            validate_recovery_row(row)
        except Exception as error:
            self._record_label_error("validate_label", observation, error)
            return learner_action
        self.rows.append(row)
        return learner_action


def collect_starter_recovery(model_path: Path, output_path: Path, seeds: Iterable[int],
                             episode_steps: int = 120) -> Path:
    from importlib.metadata import version
    from kaggle_environments import make
    from rollout.v3_agent_numpy import V3NumpyRolloutAgent

    model_path = Path(model_path)
    model_sha = _sha256(model_path)
    teacher_version = f"kaggle-environments=={version('kaggle-environments')}:starter"
    rows: list[dict[str, Any]] = []
    seed_list = [int(value) for value in seeds]
    for seed in seed_list:
        for seat in (0, 1):
            env = make(
                "kaggriculture",
                configuration={"seed": seed, "episodeSteps": int(episode_steps)},
                debug=False,
            )
            candidate = V3NumpyRolloutAgent(
                model_path, seed=seed + 101 + seat,
                deterministic=True, capture_decision_trace=True,
            )
            teacher = env.agents["starter"]
            collector = _RecoveryCollectingAgent(
                candidate, teacher, seat, seed * 10 + seat,
                teacher_version, model_sha,
            )
            agents = [collector, "starter"] if seat == 0 else ["starter", collector]
            env.run(agents)
            rows.extend(collector.rows)
    output = write_recovery_rows(rows, output_path)
    metadata = {
        "kind": "v3_learner_state_recovery",
        "teacher_id": "starter",
        "teacher_version": teacher_version,
        "supervision_kind": "smoke_only",
        "learner_model_sha256": model_sha,
        "seeds": seed_list,
        "episode_steps": int(episode_steps),
        "rows": len(rows),
    }
    output.with_suffix(output.suffix + ".meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return output


def _load_submission_callable(path: Path, module_name: str):
    import importlib.util
    import sys

    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import teacher submission: {path}")
    module = importlib.util.module_from_spec(spec)
    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    for name in ("agent", "main", "act"):
        value = getattr(module, name, None)
        if callable(value):
            return value
    raise RuntimeError(
        f"teacher submission has no callable agent/main/act: {path}"
    )


def _rollout_agent_class(model_path: Path):
    import numpy as np

    with np.load(model_path, allow_pickle=False) as archive:
        format_version = int(archive["format_version"])
    if format_version == 4:
        from rollout.v3_2_agent_numpy import V32NumpyRolloutAgent
        return V32NumpyRolloutAgent
    if format_version == 5:
        from rollout.v3_3_agent_numpy import V33NumpyRolloutAgent
        return V33NumpyRolloutAgent
    raise ValueError(
        "teacher DAgger requires V3.2/V3.3 NumPy policy, "
        f"got format {format_version}"
    )


def teacher_id_from_path(path: Path) -> str:
    parent = Path(path).resolve().parent.name.strip().lower()
    for prefix in ("extracted_", "public_", "agent_"):
        if parent.startswith(prefix):
            parent = parent[len(prefix):]
            break
    cleaned = "".join(
        char for char in parent
        if char.isalnum() or char in {"-", "_"}
    )
    return cleaned or "teacher"


def collect_teacher_recovery(
    model_path: Path,
    teacher_path: Path,
    output_path: Path,
    seeds: Iterable[int],
    *,
    episode_steps: int = 720,
    strategy_slot: int | None = None,
    teacher_id: str | None = None,
) -> Path:
    from kaggle_environments import make

    model_path = Path(model_path)
    teacher_path = Path(teacher_path)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if not teacher_path.is_file():
        raise FileNotFoundError(teacher_path)

    resolved_teacher_id = str(
        teacher_id or teacher_id_from_path(teacher_path)
    )
    model_sha = _sha256(model_path)
    teacher_sha = _sha256(teacher_path)
    teacher_version = (
        f"public_{resolved_teacher_id}:sha256:{teacher_sha}"
    )
    agent_class = _rollout_agent_class(model_path)
    seed_list = [int(value) for value in seeds]
    rows: list[dict[str, Any]] = []
    label_errors: list[dict[str, Any]] = []
    projection_corrections = 0
    projected_rows = 0
    dropped_projected_rows = 0

    for seed in seed_list:
        for seat in (0, 1):
            env = make(
                "kaggriculture",
                configuration={
                    "seed": int(seed),
                    "episodeSteps": int(episode_steps),
                },
                debug=False,
            )
            candidate = agent_class(
                model_path,
                seed=int(seed) + 101 + seat,
                deterministic=True,
                capture_decision_trace=True,
                strategy_slot=strategy_slot,
            )
            resolved_slot = getattr(candidate, "strategy_slot", strategy_slot)
            teacher = _load_submission_callable(
                teacher_path,
                f"farmos_{resolved_teacher_id}_teacher_{seed}_{seat}",
            )
            collector = _RecoveryCollectingAgent(
                candidate,
                teacher,
                seat,
                int(seed) * 10 + seat,
                teacher_version,
                model_sha,
                teacher_id=resolved_teacher_id,
                supervision_kind="accepted_policy",
                strategy_slot=resolved_slot,
                # A submission teacher is not assumed to be an oracle on
                # arbitrary learner states. If its request must be projected
                # to PASS/NOP/clip, using that projected request as a target
                # can teach closed-loop collapse.
                drop_projected_labels=True,
            )
            agents = (
                [collector, str(teacher_path)]
                if seat == 0
                else [str(teacher_path), collector]
            )
            env.run(agents)
            statuses = [str(value.status) for value in env.steps[-1]]
            env_errors = _environment_errors(env)
            projection_corrections += int(
                collector.projection_corrections
            )
            projected_rows += int(collector.projected_rows)
            dropped_projected_rows += int(
                collector.dropped_projected_rows
            )
            for item in collector.label_errors:
                label_errors.append({
                    "seed": int(seed),
                    "seat": int(seat),
                    **item,
                })
            for message in env_errors:
                label_errors.append({
                    "seed": int(seed),
                    "seat": int(seat),
                    "stage": "environment_callback",
                    "step": -1,
                    "player": int(seat),
                    "error_type": "EnvironmentLog",
                    "error": str(message),
                })
            if statuses != ["DONE", "DONE"]:
                raise RuntimeError(
                    f"{resolved_teacher_id} DAgger game did not finish: "
                    f"seed={seed} seat={seat} statuses={statuses}"
                )
            rows.extend(collector.rows)

    error_report = {
        "kind": "v3_teacher_dagger_label_errors",
        "teacher_id": resolved_teacher_id,
        "teacher_version": teacher_version,
        "teacher_sha256": teacher_sha,
        "learner_model_sha256": model_sha,
        "seeds": seed_list,
        "episode_steps": int(episode_steps),
        "games": len(seed_list) * 2,
        "accepted_rows": len(rows),
        "projected_rows": int(projected_rows),
        "dropped_projected_rows": int(dropped_projected_rows),
        "projection_corrections": int(projection_corrections),
        "label_error_count": len(label_errors),
        "label_errors": label_errors[:200],
    }
    error_path = output_path.with_suffix(output_path.suffix + ".errors.json")
    if label_errors or not rows:
        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_path.write_text(
            json.dumps(error_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    if not rows:
        first = label_errors[0] if label_errors else {}
        raise RecoveryCollectionEmptyError(
            f"{resolved_teacher_id} DAgger produced zero accepted labels"
            + (
                f"; first_error={first.get('stage')}:{first.get('error')}"
                if first else ""
            ),
            report_path=error_path,
        )

    output = write_recovery_rows(rows, output_path)
    metadata = {
        "kind": "v3_teacher_dagger_recovery",
        "teacher_id": resolved_teacher_id,
        "teacher_version": teacher_version,
        "teacher_sha256": teacher_sha,
        "supervision_kind": "accepted_policy",
        "learner_model_sha256": model_sha,
        "strategy_slot": (
            None if strategy_slot is None else int(strategy_slot)
        ),
        "seeds": seed_list,
        "episode_steps": int(episode_steps),
        "games": len(seed_list) * 2,
        "rows": len(rows),
        "projected_rows": int(projected_rows),
        "dropped_projected_rows": int(dropped_projected_rows),
        "projection_corrections": int(projection_corrections),
        "projection_acceptance_rate": float(
            len(rows) / max(len(rows) + dropped_projected_rows, 1)
        ),
        "label_error_count": len(label_errors),
        "label_error_report": (
            str(error_path) if label_errors else None
        ),
    }
    output.with_suffix(output.suffix + ".meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output




def _terminal_outcome(observation: dict[str, Any]) -> tuple[int, int, int]:
    player = int(observation.get("player", 0))
    farms = observation.get("farms") or []
    if not isinstance(farms, list) or len(farms) < 2:
        return 0, 0, 0
    own = int((farms[player] or {}).get("money", 0) or 0)
    rival = int((farms[1 - player] or {}).get("money", 0) or 0)
    margin = own - rival
    result = 1 if margin > 0 else (-1 if margin < 0 else 0)
    return own, margin, result


def collect_teacher_demonstrations(
    teacher_path: Path,
    output_path: Path,
    seeds: Iterable[int],
    *,
    episode_steps: int = 720,
    strategy_slot: int | None = None,
    teacher_id: str | None = None,
) -> Path:
    """Collect clean on-policy demonstrations from teacher self-play.

    Kaggle stores the action chosen from state t on env.steps[t + 1], so the
    rows are aligned as (observation[t], action[t + 1]). Any teacher request
    that still needs legality projection on its own trajectory is discarded.
    """
    from kaggle_environments import make

    teacher_path = Path(teacher_path)
    output_path = Path(output_path)
    if not teacher_path.is_file():
        raise FileNotFoundError(teacher_path)

    resolved_teacher_id = str(
        teacher_id or teacher_id_from_path(teacher_path)
    )
    teacher_sha = _sha256(teacher_path)
    teacher_version = (
        f"public_{resolved_teacher_id}:sha256:{teacher_sha}"
    )
    seed_list = [int(value) for value in seeds]
    rows: list[dict[str, Any]] = []
    projected_rows = 0
    dropped_projected_rows = 0
    projection_corrections = 0
    game_summaries = []

    for seed in seed_list:
        env = make(
            "kaggriculture",
            configuration={
                "seed": int(seed),
                "episodeSteps": int(episode_steps),
            },
            debug=False,
        )
        env.run([str(teacher_path), str(teacher_path)])
        statuses = [str(value.status) for value in env.steps[-1]]
        if statuses != ["DONE", "DONE"]:
            raise RuntimeError(
                f"{resolved_teacher_id} self-play did not finish: "
                f"seed={seed} statuses={statuses}"
            )

        per_game_rows = 0
        per_game_dropped = 0
        for seat in (0, 1):
            tracker = EffectTracker()
            previous_action: dict[str, Any] = {}
            previous_effect: dict[str, Any] = {}
            final_obs = _plain(env.steps[-1][seat].observation)
            final_own_money, final_margin, terminal_result = (
                _terminal_outcome(final_obs)
            )
            episode_id = int(seed) * 10 + int(seat)
            for index in range(max(0, len(env.steps) - 1)):
                state_agent = env.steps[index][seat]
                action_agent = env.steps[index + 1][seat]
                obs_plain = _plain(state_agent.observation)
                next_obs_plain = _plain(
                    env.steps[index + 1][seat].observation
                )
                raw_action = _plain(action_agent.action or {})
                canonical = canonicalize_teacher_action(
                    raw_action, obs_plain
                )
                effects = tracker.observe(
                    obs_plain,
                    raw_action,
                    next_obs_plain,
                ).to_model_effect()
                structured_state = asdict(
                    normalize_observation(obs_plain)
                )
                projected, corrections = (
                    project_teacher_action_to_executable(
                        canonical, structured_state
                    )
                )
                projection_corrections += int(corrections)
                projected_rows += int(corrections > 0)
                if int(corrections) > 0:
                    dropped_projected_rows += 1
                    per_game_dropped += 1
                    previous_action = canonical
                    previous_effect = effects
                    continue

                row = {
                    "episode_id": episode_id,
                    "seat": int(seat),
                    "step": int(obs_plain.get("step", index)),
                    "state": structured_state,
                    "canonical_action": projected,
                    "previous_action": deepcopy(previous_action),
                    "previous_effect": deepcopy(previous_effect),
                    "effects": deepcopy(effects),
                    "final_own_money": int(final_own_money),
                    "final_margin": int(final_margin),
                    "terminal_result": int(terminal_result),
                    "teacher_id": resolved_teacher_id,
                    "teacher_version": teacher_version,
                    "supervision_kind": "teacher_demo",
                }
                if strategy_slot is not None:
                    row["strategy_slot"] = int(strategy_slot)
                validate_recovery_row(row)
                rows.append(row)
                per_game_rows += 1
                previous_action = canonical
                previous_effect = effects

        game_summaries.append({
            "seed": int(seed),
            "accepted_rows": int(per_game_rows),
            "dropped_projected_rows": int(per_game_dropped),
        })

    if not rows:
        raise RecoveryCollectionEmptyError(
            f"{resolved_teacher_id} self-play produced zero clean labels"
        )

    output = write_recovery_rows(rows, output_path)
    metadata = {
        "kind": "v3_teacher_selfplay_demonstrations",
        "teacher_id": resolved_teacher_id,
        "teacher_version": teacher_version,
        "teacher_sha256": teacher_sha,
        "supervision_kind": "teacher_demo",
        "strategy_slot": (
            None if strategy_slot is None else int(strategy_slot)
        ),
        "seeds": seed_list,
        "episode_steps": int(episode_steps),
        "games": len(seed_list),
        "rows": len(rows),
        "projected_rows": int(projected_rows),
        "dropped_projected_rows": int(dropped_projected_rows),
        "projection_corrections": int(projection_corrections),
        "clean_label_rate": float(
            len(rows) / max(len(rows) + dropped_projected_rows, 1)
        ),
        "game_summaries": game_summaries,
    }
    output.with_suffix(output.suffix + ".meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def collect_v45_recovery(
    model_path: Path,
    v45_path: Path,
    output_path: Path,
    seeds: Iterable[int],
    *,
    episode_steps: int = 720,
    strategy_slot: int | None = None,
) -> Path:
    """Backward-compatible wrapper for existing v45 tooling/tests."""
    return collect_teacher_recovery(
        model_path,
        v45_path,
        output_path,
        seeds,
        episode_steps=episode_steps,
        strategy_slot=strategy_slot,
        teacher_id="v45",
    )


def build_expert_early_recovery(dataset_path: Path, output_path: Path,
                                max_step: int = 31) -> Path:
    from kaggrl.v2_dataset import decode_zlib_json
    from kaggrl.v2_training_data import V2EpisodeDataset

    dataset_path = Path(dataset_path)
    dataset_sha = _sha256(dataset_path)
    dataset = V2EpisodeDataset(dataset_path, "train", {"active_best"})
    rows: list[dict[str, Any]] = []
    for episode in dataset:
        for source in episode.rows:
            step = int(source["step"])
            if step > int(max_step):
                continue
            teacher_id = (
                f"top10_active_best:{int(source.get('team_id', episode.team_id))}:"
                f"{int(source.get('submission_id', 0))}"
            )
            row = {
                "episode_id": int(source["episode_id"]),
                "seat": int(source["seat"]), "step": step,
                "state": decode_zlib_json(source["state_zlib"]),
                "canonical_action": deepcopy(source["canonical_action"]),
                "previous_action": deepcopy(source.get("previous_action") or {}),
                "previous_effect": deepcopy(source.get("previous_effect") or {}),
                "effects": deepcopy(source.get("effects") or {}),
                "final_own_money": float(source.get("final_own_money", 0)),
                "final_margin": float(source.get("final_margin", 0)),
                "terminal_result": float(source.get("terminal_result", 0)),
                "teacher_id": teacher_id,
                "teacher_version": f"dataset_sha256:{dataset_sha}",
                "supervision_kind": "expert",
                "learner_model_sha256": "",
            }
            rows.append(row)
    return write_recovery_rows(rows, output_path)
