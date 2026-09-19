import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from kaggrl.v2_dataset import build_arrow_table
from kaggrl.v2_observation import normalize_observation
from kaggrl.v2_training_data import (
    V2EpisodeDataset,
    collate_v2_sequences,
    verify_effective_action_sidecar,
    verify_training_acceptance,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _state(step: int, hands: int = 1):
    tiles = [[None for _ in range(10)] for _ in range(10)]
    hand_pos = [[5 + (i % 2), 4] for i in range(hands)]
    inventories = [{"WHEAT": 3}] + [{"WHEAT": i + 1} for i in range(hands)]
    obs = {
        "player": 0, "step": step, "day": step // 24, "hour": step % 24,
        "farms": [
            {"money": 3000, "farmer": [4, 4], "hands": hand_pos,
             "hires_today": hands, "unlocked_quadrants": ["NW"], "tiles": tiles},
            {"money": 2500, "farmer": [8, 8], "hands": [],
             "hires_today": 0, "unlocked_quadrants": ["NW"], "tiles": tiles},
        ],
        "private": {"shed": {"WHEAT": 20}, "seeds": {"WHEAT": 5},
                    "inventories": inventories},
        "market": {"inventory": {"WHEAT": 10000}, "prices": {"WHEAT": 25}},
        "town": {"unlocked_shops": []},
    }
    return asdict(normalize_observation(obs))


def _action(hands: int):
    unit = {"op": "PASS", "item": None, "quantity": None, "raw": ["PASS"]}
    return {"farmer": dict(unit), "hands": [dict(unit) for _ in range(hands)],
            "market": [{"kind": "STOP_QUEUE", "op": None,
                        "item": None, "quantity": None, "raw": []}]}


def _row(episode: int, step: int, *, role="active_best", split="train", hands=1):
    state = _state(step, hands)
    return {
        "episode_id": episode, "seat": 0, "step": step,
        "create_time": f"2026-09-17T00:{episode % 60:02d}:00+00:00",
        "team_id": 100 + episode, "team_name": f"team-{episode}",
        "submission_id": 200 + episode, "submission_date": "2026-09-16T00:00:00+00:00",
        "rank": 1, "role": role, "split": split,
        "replay_file_sha256": "a" * 64, "replay_json_sha256": "b" * 64,
        "state": state, "next_state": _state(step + 1, hands),
        "raw_action": {}, "canonical_action": _action(hands), "effects": {},
        "final_own_money": 3000, "final_rival_money": 2500,
        "final_margin": 500, "terminal_result": 1,
    }


def _dataset(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(build_arrow_table(list(rows)), path)
    return path


def _acceptance_fixture(tmp_path, dataset):
    live = tmp_path / "data/top_tier/live_v2"
    manifests = live / "manifests"; manifests.mkdir(parents=True, exist_ok=True)
    target = live / "transitions.parquet"; target.write_bytes(dataset.read_bytes())
    snapshot = manifests / "live_top10_snapshot.json"; snapshot.write_text("{}")
    corpus = manifests / "trusted_corpus_manifest.json"; corpus.write_text("{}")
    stage0 = _write_json(manifests / "STAGE0_ACCEPTED.json", {
        "accepted": True, "counters": {"errors": 0},
        "artifacts": {
            "dataset_sha256": _sha(target),
            "selection_snapshot_sha256": _sha(snapshot),
            "trusted_corpus_manifest_sha256": _sha(corpus),
        },
    })
    out = tmp_path / "checkpoints/rl_v2_stage1"; out.mkdir(parents=True)
    archive = out / "policy_init.npz"; archive.write_bytes(b"stage1")
    build = _write_json(out / "build_manifest.json", {"archive_sha256": _sha(archive)})
    parity = _write_json(out / "parity_report.json", {"passed": True})
    stage1 = _write_json(out / "STAGE1_ACCEPTED.json", {
        "accepted": True,
        "artifacts": {
            "stage0_marker_sha256": _sha(stage0),
            "build_manifest_sha256": _sha(build),
            "parity_report_sha256": _sha(parity),
            "archive_sha256": _sha(archive),
        },
    })
    return stage0, stage1, target


def test_training_acceptance_binds_stage0_dataset_and_stage1(tmp_path):
    source = _dataset(tmp_path / "source.parquet", [_row(1, 0)])
    stage0, stage1, dataset = _acceptance_fixture(tmp_path, source)
    report = verify_training_acceptance(stage0, stage1)
    assert report["accepted"] is True
    assert report["dataset_sha256"] == _sha(dataset)
    dataset.write_bytes(dataset.read_bytes() + b"tamper")
    with pytest.raises(RuntimeError, match="dataset SHA"):
        verify_training_acceptance(stage0, stage1)


def test_episode_dataset_filters_roles_and_chunks_without_crossing_episode(tmp_path):
    rows = [_row(10, step) for step in range(65)]
    rows += [_row(11, step) for step in range(3)]
    rows += [_row(12, step, role="latest_candidate") for step in range(4)]
    path = _dataset(tmp_path / "transitions.parquet", rows)
    dataset = V2EpisodeDataset(path, split="train", roles={"active_best"})
    assert len(dataset) == 2
    assert {sample.episode_id for sample in dataset} == {10, 11}
    chunks = list(dataset.iter_chunks(32))
    assert [(c.episode_id, c.start_step, c.end_step, len(c.rows)) for c in chunks] == [
        (10, 0, 31, 32), (10, 32, 63, 32), (10, 64, 64, 1),
        (11, 0, 2, 3),
    ]
    assert all(len({row["episode_id"] for row in chunk.rows}) == 1 for chunk in chunks)


def test_primary_training_rejects_non_active_best_role_request(tmp_path):
    path = _dataset(tmp_path / "transitions.parquet", [_row(1, 0, role="latest_candidate")])
    with pytest.raises(RuntimeError, match="active_best"):
        V2EpisodeDataset(path, split="train", roles={"latest_candidate"})


def test_sequence_batch_separates_model_inputs_from_training_targets(tmp_path):
    rows = [_row(20, step, hands=(step % 3)) for step in range(4)]
    path = _dataset(tmp_path / "transitions.parquet", rows)
    dataset = V2EpisodeDataset(path, split="train", roles={"active_best"})
    batch = collate_v2_sequences([dataset[0]], sequence_len=32)
    inputs = batch.model_inputs()
    targets = batch.targets()
    forbidden = {
        "canonical_actions", "effects_targets", "auxiliary_targets",
        "terminal_money", "terminal_margin", "future_resource",
        "unit_task", "opponent_effect", "teacher_next_state",
    }
    assert forbidden.isdisjoint(inputs)
    assert {"sequence_mask", "reset_mask", "done_mask", "row_slots"}.issubset(inputs)
    assert {"canonical_actions", "effects_targets", "terminal_money",
            "terminal_margin"}.issubset(targets)
    assert batch.sequence_mask.shape == (1, 32)
    assert batch.sequence_mask[0, :4].all() and not batch.sequence_mask[0, 4:].any()
    assert batch.reset_mask[0, 0].item() is True
    assert batch.done_mask[0, 3].item() is True
    assert torch.equal(batch.row_slots[0, :4], torch.arange(4))


def test_chunk_start_carries_previous_observable_context(tmp_path):
    rows = [_row(30, step) for step in range(33)]
    path = _dataset(tmp_path / "transitions.parquet", rows)
    dataset = V2EpisodeDataset(path, split="train", roles={"active_best"})
    chunks = list(dataset.iter_chunks(32))
    assert chunks[1].rows[0]["step"] == 32
    assert chunks[1].rows[0]["previous_action"] == chunks[0].rows[-1]["canonical_action"]
    assert chunks[1].rows[0]["previous_effect"] == chunks[0].rows[-1]["effects"]


def test_sequence_targets_build_all_auxiliary_training_labels(tmp_path):
    rows = [_row(40, step, hands=1) for step in range(3)]
    rows[0]["effects"] = {
        "money_delta": 25, "opponent_public": {
            "money_delta": -30, "hand_count_delta": 1,
            "farmer_position_delta": [1, 0], "grid_changed": True,
            "confidence": "inferred",
        }
    }
    rows[1]["canonical_action"]["hands"][0] = {
        "op": "PICKUP", "item": "WHEAT", "quantity": 3,
        "raw": ["PICKUP", "WHEAT", 3],
    }
    rows[2]["canonical_action"]["market"] = [
        {"kind": "ORDER", "op": "HIRE", "item": None, "quantity": None, "raw": ["HIRE"]},
        {"kind": "STOP_QUEUE", "op": None, "item": None, "quantity": None, "raw": []},
    ]
    path = _dataset(tmp_path / "transitions.parquet", rows)
    dataset = V2EpisodeDataset(path, split="train", roles={"active_best"})
    batch = collate_v2_sequences([dataset[0]], sequence_len=8)
    targets = batch.targets()
    assert targets["effect"].shape == (3, 42)
    assert targets["future_resource"].shape == (3, 32)
    assert targets["unit_task"].shape == (3, batch.flat.own_units.shape[1], 16)
    assert targets["opponent_effect"].shape == (3, 16)
    assert targets["effect"][0].abs().sum().item() > 0
    assert targets["future_resource"][0].abs().sum().item() > 0
    assert targets["unit_task"][0, 1].abs().sum().item() > 0
    assert targets["opponent_effect"][0].abs().sum().item() > 0
    assert batch.flat.auxiliary_targets["future_resource"].shape == (3, 32)


def test_balanced_episode_order_equalizes_team_contribution_and_is_reproducible():
    from kaggrl.v2_training_data import EpisodeSequence, balanced_episode_order

    def episode(eid, team):
        return EpisodeSequence(eid, 0, team, "active_best", "train", ({"step": 0},))

    samples = [episode(1, 10), episode(2, 10), episode(3, 10), episode(4, 20)]
    first = balanced_episode_order(samples, seed=20260917, epoch=1)
    second = balanced_episode_order(samples, seed=20260917, epoch=1)
    assert [x.episode_id for x in first] == [x.episode_id for x in second]
    counts = {}
    for item in first:
        counts[item.team_id] = counts.get(item.team_id, 0) + 1
    assert counts == {10: 3, 20: 3}
    assert len(first) == 6


def test_episode_dataset_sanitizes_sell_to_engine_executable_quantity(tmp_path):
    rows = [_row(50, 0, hands=1), _row(50, 1, hands=1)]
    rows[0]["state"]["private"]["shed"]["WHEAT"] = 2
    rows[0]["canonical_action"]["hands"][0] = {
        "op": "PICKUP", "item": "WHEAT", "quantity": 1,
        "raw": ["PICKUP", "WHEAT", 1],
    }
    rows[0]["canonical_action"]["market"] = [
        {
            "kind": "ORDER", "op": "SELL", "item": "WHEAT",
            "quantity": 2, "raw": ["SELL", "WHEAT", 2],
        },
        {"kind": "STOP_QUEUE", "op": None, "item": None, "quantity": None, "raw": []},
    ]
    path = _dataset(tmp_path / "transitions.parquet", rows)
    dataset = V2EpisodeDataset(path, split="train", roles={"active_best"})
    first, second = dataset[0].rows

    assert first["requested_canonical_action"]["market"][0]["quantity"] == 2
    assert first["canonical_action"]["market"][0] == {
        "kind": "ORDER", "op": "SELL", "item": "WHEAT",
        "quantity": 1, "raw": ["SELL", "WHEAT", 1],
    }
    assert second["previous_action"] == first["requested_canonical_action"]
    assert second["previous_action"] != first["canonical_action"]


def test_episode_dataset_turns_unexecutable_sell_into_nop_slot(tmp_path):
    row = _row(51, 0, hands=0)
    row["state"]["private"]["shed"]["WHEAT"] = 0
    row["canonical_action"]["market"] = [
        {
            "kind": "ORDER", "op": "SELL", "item": "WHEAT",
            "quantity": 100, "raw": ["SELL", "WHEAT", 100],
        },
        {"kind": "STOP_QUEUE", "op": None, "item": None, "quantity": None, "raw": []},
    ]
    path = _dataset(tmp_path / "transitions.parquet", [row])
    dataset = V2EpisodeDataset(path, split="train", roles={"active_best"})
    market = dataset[0].rows[0]["canonical_action"]["market"]
    assert market[0] == {
        "kind": "NOP_SLOT", "op": None, "item": None,
        "quantity": None, "raw": [],
    }
    assert market[1]["kind"] == "STOP_QUEUE"


def _effective_sidecar(path: Path, rows: list[dict]) -> Path:
    records = []
    for row in rows:
        records.append({
            "episode_id": int(row["episode_id"]),
            "seat": int(row["seat"]),
            "step": int(row["step"]),
            "split": str(row["split"]),
            "role": str(row["role"]),
            "effective_action_json": json.dumps(
                row["effective_action"], sort_keys=True, separators=(",", ":"),
            ),
        })
    pq.write_table(pa.Table.from_pylist(records), path)
    return path


def test_episode_dataset_prefers_exact_effective_action_sidecar(tmp_path):
    row = _row(60, 0, hands=0)
    row["canonical_action"]["market"] = [
        {
            "kind": "ORDER", "op": "SELL", "item": "WHEAT",
            "quantity": 1000, "raw": ["SELL", "WHEAT", 1000],
        },
        {"kind": "STOP_QUEUE", "op": None, "item": None, "quantity": None, "raw": []},
    ]
    dataset_path = _dataset(tmp_path / "transitions.parquet", [row])
    effective_action = _action(0)
    effective_action["market"] = [
        {"kind": "NOP_SLOT", "op": None, "item": None, "quantity": None, "raw": []},
        {"kind": "STOP_QUEUE", "op": None, "item": None, "quantity": None, "raw": []},
    ]
    sidecar = _effective_sidecar(tmp_path / "effective_actions.parquet", [{
        **row, "effective_action": effective_action,
    }])
    dataset = V2EpisodeDataset(
        dataset_path, split="train", roles={"active_best"},
        effective_action_path=sidecar, require_effective_actions=True,
    )
    prepared = dataset[0].rows[0]
    assert prepared["requested_canonical_action"]["market"][0]["quantity"] == 1000
    assert prepared["canonical_action"]["market"][0]["kind"] == "NOP_SLOT"


def test_episode_dataset_rejects_missing_effective_transition_key(tmp_path):
    rows = [_row(61, 0), _row(61, 1)]
    dataset_path = _dataset(tmp_path / "transitions.parquet", rows)
    effective_action = _action(1)
    sidecar = _effective_sidecar(tmp_path / "effective_actions.parquet", [{
        **rows[0], "effective_action": effective_action,
    }])
    with pytest.raises(RuntimeError, match="missing effective action"):
        V2EpisodeDataset(
            dataset_path, split="train", roles={"active_best"},
            effective_action_path=sidecar, require_effective_actions=True,
        )


def test_effective_action_sidecar_is_hash_bound_to_dataset(tmp_path):
    live = tmp_path / "data" / "top_tier" / "live_v2"
    live.mkdir(parents=True)
    row = _row(62, 0)
    dataset_path = _dataset(live / "transitions.parquet", [row])
    sidecar = _effective_sidecar(live / "effective_actions.parquet", [{
        **row, "effective_action": _action(1),
    }])
    corpus = _write_json(
        live / "manifests" / "trusted_corpus_manifest.json", {}
    )
    summary = _write_json(live / "manifests" / "effective_actions_summary.json", {
        "rows": 1,
        "verified_non_eod_transitions": 1,
        "verified_replay_files": 1,
        "engine_module_version": "1.32.7",
        "engine_source_sha256": "e" * 64,
        "builder_code_sha256": "b" * 64,
        "source_dataset_sha256": _sha(dataset_path),
        "source_corpus_manifest_sha256": _sha(corpus),
        "effective_actions_sha256": _sha(sidecar),
    })
    report = verify_effective_action_sidecar(dataset_path)
    assert report["sha256"] == _sha(sidecar)
    assert report["engine_module_version"] == "1.32.7"

    sidecar.write_bytes(sidecar.read_bytes() + b"tamper")
    with pytest.raises(RuntimeError, match="sidecar SHA"):
        verify_effective_action_sidecar(dataset_path)
