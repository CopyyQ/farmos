from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Iterable


from kaggrl.constants import UNIT_OPS
from kaggrl.v2_action_schema import parse_raw_action
from kaggrl.v2_ledger import MARKET_OPS, ShadowLedger
from kaggrl.v2_observation import normalize_observation


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


def validate_recovery_row(row: dict[str, Any]) -> bool:
    for key in ("teacher_id", "teacher_version", "supervision_kind"):
        if not str(row.get(key, "")).strip():
            raise ValueError(f"recovery row missing {key}")
    if row["supervision_kind"] not in {"expert", "accepted_policy", "smoke_only"}:
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
        self.rows: list[dict[str, Any]] = []

    def __call__(self, observation, configuration=None):
        teacher_obs = deepcopy(observation)
        try:
            teacher_raw = self.teacher(teacher_obs, configuration)
        except TypeError as two_arg_error:
            try:
                teacher_raw = self.teacher(teacher_obs)
            except TypeError:
                raise two_arg_error
        learner_action = self.candidate(observation, configuration)
        fixture = self.candidate.diagnostic_fixtures[-1]
        obs_plain = _plain(observation)
        canonical = canonicalize_teacher_action(_plain(teacher_raw), obs_plain)
        row = {
            "episode_id": self.episode_id,
            "seat": self.seat,
            "step": int(fixture["step"]),
            "state": deepcopy(fixture["structured_state"]),
            "canonical_action": canonical,
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
        validate_recovery_row(row)
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
        f"v45 DAgger requires V3.2/V3.3 NumPy policy, got format {format_version}"
    )


def collect_v45_recovery(
    model_path: Path,
    v45_path: Path,
    output_path: Path,
    seeds: Iterable[int],
    *,
    episode_steps: int = 720,
    strategy_slot: int | None = None,
) -> Path:
    from kaggle_environments import make

    model_path = Path(model_path)
    v45_path = Path(v45_path)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if not v45_path.is_file():
        raise FileNotFoundError(v45_path)

    model_sha = _sha256(model_path)
    teacher_sha = _sha256(v45_path)
    teacher_version = f"public_v45:sha256:{teacher_sha}"
    agent_class = _rollout_agent_class(model_path)
    seed_list = [int(value) for value in seeds]
    rows: list[dict[str, Any]] = []

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
                v45_path,
                f"farmos_v45_teacher_{seed}_{seat}",
            )
            collector = _RecoveryCollectingAgent(
                candidate,
                teacher,
                seat,
                int(seed) * 10 + seat,
                teacher_version,
                model_sha,
                teacher_id="v45",
                supervision_kind="accepted_policy",
                strategy_slot=resolved_slot,
            )
            agents = (
                [collector, str(v45_path)]
                if seat == 0
                else [str(v45_path), collector]
            )
            env.run(agents)
            statuses = [str(value.status) for value in env.steps[-1]]
            if statuses != ["DONE", "DONE"]:
                raise RuntimeError(
                    f"v45 DAgger game did not finish: seed={seed} seat={seat} "
                    f"statuses={statuses}"
                )
            rows.extend(collector.rows)

    output = write_recovery_rows(rows, output_path)
    metadata = {
        "kind": "v3_v45_dagger_recovery",
        "teacher_id": "v45",
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
    }
    output.with_suffix(output.suffix + ".meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


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
