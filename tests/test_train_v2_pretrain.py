import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pyarrow.parquet as pq
import torch

from kaggrl.v2_dataset import build_arrow_table
from kaggrl.v2_model import RecurrentIntentPolicy
from kaggrl.v2_observation import normalize_observation
from training.train_v2_pretrain import PretrainConfig, run_pretraining


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _state(step: int, hands: int):
    tiles = [[None for _ in range(10)] for _ in range(10)]
    hand_pos = [[5 + (i % 2), 4] for i in range(hands)]
    inventories = [{"WHEAT": 2}] + [{"WHEAT": i + 1} for i in range(hands)]
    obs = {
        "player": 0, "step": step, "day": step // 24, "hour": step % 24,
        "farms": [
            {"money": 3000 + step, "farmer": [4, 4], "hands": hand_pos,
             "hires_today": hands, "unlocked_quadrants": ["NW"], "tiles": tiles},
            {"money": 2500, "farmer": [8, 8], "hands": [],
             "hires_today": 0, "unlocked_quadrants": ["NW"], "tiles": tiles},
        ],
        "private": {"shed": {"WHEAT": 20}, "seeds": {"WHEAT": 8},
                    "inventories": inventories},
        "market": {"inventory": {"WHEAT": 10000}, "prices": {"WHEAT": 25}},
        "town": {"unlocked_shops": []},
    }
    return asdict(normalize_observation(obs))


def _action(hands: int, step: int):
    farmer = {"op": "EAST" if step % 2 == 0 else "PASS",
              "item": None, "quantity": None, "raw": ["EAST" if step % 2 == 0 else "PASS"]}
    hand_actions = []
    for index in range(hands):
        if (step + index) % 3 == 0:
            hand_actions.append({"op": "PICKUP", "item": "WHEAT", "quantity": 2,
                                 "raw": ["PICKUP", "WHEAT", 2]})
        else:
            hand_actions.append({"op": "PASS", "item": None, "quantity": None, "raw": ["PASS"]})
    market = [{"kind": "ORDER", "op": "HIRE", "item": None,
               "quantity": None, "raw": ["HIRE"]}] if step % 3 == 0 else []
    market.append({"kind": "STOP_QUEUE", "op": None, "item": None,
                   "quantity": None, "raw": []})
    return {"farmer": farmer, "hands": hand_actions, "market": market}


def _row(episode: int, step: int, split: str, hands: int):
    action = _action(hands, step)
    effects = {
        "money_delta": 5 if step % 2 == 0 else -2,
        "hand_count_delta": 0,
        "shed_delta": {"WHEAT": -1 if step % 3 == 0 else 0},
        "opponent_public": {
            "money_delta": -3 if step % 2 == 0 else 1,
            "hand_count_delta": 0,
            "farmer_position_delta": [0, 1 if step % 2 == 0 else 0],
            "grid_changed": bool(step % 2 == 0), "confidence": "inferred",
        },
        "action_evidence": [{"actor": "farmer", "op": action["farmer"]["op"],
                             "status": "confirmed", "observed": {}}],
    }
    return {
        "episode_id": episode, "seat": 0, "step": step,
        "create_time": "2026-09-17T00:00:00+00:00",
        "team_id": 100 + episode, "team_name": f"team-{episode}",
        "submission_id": 200 + episode, "submission_date": "2026-09-16T00:00:00+00:00",
        "rank": 1, "role": "active_best", "split": split,
        "replay_file_sha256": "a" * 64, "replay_json_sha256": "b" * 64,
        "state": _state(step, hands), "next_state": _state(step + 1, hands),
        "raw_action": {}, "canonical_action": action, "effects": effects,
        "final_own_money": 3200, "final_rival_money": 2500,
        "final_margin": 700, "terminal_result": 1,
    }


def _fixture(tmp_path: Path):
    rows = [_row(1, step, "train", 0) for step in range(4)]
    rows += [_row(2, step, "train", 2) for step in range(4)]
    rows += [_row(3, step, "val", 1) for step in range(4)]
    live = tmp_path / "data/top_tier/live_v2"
    live.mkdir(parents=True)
    dataset = live / "transitions.parquet"
    pq.write_table(build_arrow_table(rows), dataset)
    manifests = live / "manifests"; manifests.mkdir()
    snapshot = manifests / "live_top10_snapshot.json"; snapshot.write_text("{}")
    corpus = manifests / "trusted_corpus_manifest.json"; corpus.write_text("{}")
    stage0 = _write_json(manifests / "STAGE0_ACCEPTED.json", {
        "accepted": True, "counters": {"errors": 0},
        "artifacts": {
            "dataset_sha256": _sha(dataset),
            "selection_snapshot_sha256": _sha(snapshot),
            "trusted_corpus_manifest_sha256": _sha(corpus),
        },
    })
    stage1_dir = tmp_path / "checkpoints/rl_v2_stage1"; stage1_dir.mkdir(parents=True)
    archive = stage1_dir / "policy_init.npz"; archive.write_bytes(b"stage1")
    build = _write_json(stage1_dir / "build_manifest.json", {"archive": "policy_init.npz"})
    parity = _write_json(stage1_dir / "parity_report.json", {"passed": True})
    stage1 = _write_json(stage1_dir / "STAGE1_ACCEPTED.json", {
        "accepted": True, "artifacts": {
            "stage0_marker_sha256": _sha(stage0),
            "build_manifest_sha256": _sha(build),
            "parity_report_sha256": _sha(parity),
            "archive_sha256": _sha(archive),
        },
    })
    return dataset, stage0, stage1


def test_pretraining_two_steps_updates_shared_encoder_and_writes_reproducible_checkpoint(tmp_path):
    dataset, stage0, stage1 = _fixture(tmp_path)
    output = tmp_path / "checkpoints/rl_v2_stage2/pretrain"
    config = PretrainConfig(
        dataset_path=dataset,
        stage0_marker=stage0,
        stage1_marker=stage1,
        output_dir=output,
        seed=20260917,
        sequence_len=4,
        batch_sequences=1,
        learning_rate=3e-4,
        weight_decay=1e-4,
        gradient_clip=1.0,
        mask_rate=0.5,
        epochs=1,
        max_train_steps=2,
        max_val_chunks=1,
    )
    torch.manual_seed(config.seed)
    initial = RecurrentIntentPolicy()
    before = initial.encoder.tile_encoder.net[0].weight.detach().clone()

    checkpoint = run_pretraining(config)
    assert checkpoint == output / "pretrain_best.pt"
    assert checkpoint.exists() and (output / "pretrain_last.pt").exists()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["format_version"] == 1
    assert payload["train_steps"] == 2
    assert payload["dataset_sha256"] == _sha(dataset)
    assert payload["config"]["seed"] == 20260917
    assert payload["optimizer_state"]
    assert payload["validation_metrics"]
    losses = payload["last_train_metrics"]
    for key in ("total", "effect", "future_resource", "unit_task",
                "opponent_effect", "value", "structure"):
        assert torch.isfinite(torch.tensor(losses[key])), key
    after = payload["model_state"]["encoder.tile_encoder.net.0.weight"]
    assert not torch.equal(before, after)
    assert not any(key.startswith("action") for key in losses)
    assert (output / "history.jsonl").exists()
    assert (output / "manifest.sha256").exists()


def test_pretrain_config_declares_production_early_stop_contract():
    fields = PretrainConfig.__dataclass_fields__
    assert fields["epochs"].default == 8
    assert fields["early_stop_patience"].default == 3


def test_pretrain_config_records_cpu_thread_count_for_reproducibility():
    fields = PretrainConfig.__dataclass_fields__
    assert fields["torch_num_threads"].default == 2


def test_pretrain_unit_task_loss_is_normalized_per_step_not_global_unit_count():
    from training.train_v2_pretrain import _masked_unit_mse

    target_small = torch.zeros((2, 1, 1))
    pred_small = torch.tensor([[[2.0]], [[0.0]]])
    mask_small = torch.tensor([[True], [True]])
    target_many = torch.zeros((2, 3, 1))
    pred_many = torch.tensor([[[2.0], [0.0], [0.0]], [[0.0], [0.0], [0.0]]])
    mask_many = torch.tensor([[True, False, False], [True, True, True]])
    assert torch.allclose(
        _masked_unit_mse(pred_small, target_small, mask_small),
        _masked_unit_mse(pred_many, target_many, mask_many),
    )


def test_pretrain_config_declares_auto_device_contract():
    fields = PretrainConfig.__dataclass_fields__
    assert fields["device"].default == "auto"


def test_pretraining_resume_continues_next_epoch_and_preserves_history(tmp_path):
    dataset, stage0, stage1 = _fixture(tmp_path)
    output = tmp_path / "checkpoints/rl_v2_stage2/pretrain_resume"
    first = PretrainConfig(
        dataset_path=dataset, stage0_marker=stage0, stage1_marker=stage1,
        output_dir=output, sequence_len=4, batch_sequences=1,
        epochs=1, max_val_chunks=1, device="cpu",
    )
    run_pretraining(first)
    first_last = torch.load(output / "pretrain_last.pt", map_location="cpu", weights_only=False)
    assert first_last["epoch"] == 1
    first_steps = first_last["train_steps"]

    second = PretrainConfig(
        dataset_path=dataset, stage0_marker=stage0, stage1_marker=stage1,
        output_dir=output, sequence_len=4, batch_sequences=1,
        epochs=2, max_val_chunks=1, device="cpu",
    )
    run_pretraining(second, resume_checkpoint=output / "pretrain_last.pt")
    resumed = torch.load(output / "pretrain_last.pt", map_location="cpu", weights_only=False)
    assert resumed["epoch"] == 2
    assert resumed["train_steps"] > first_steps
    history = [json.loads(line) for line in (output / "history.jsonl").read_text().splitlines()]
    assert [row["epoch"] for row in history] == [1, 2]
    assert resumed["resolved_device"] == "cpu"


def test_pretraining_resume_rejects_checkpoint_from_other_dataset(tmp_path):
    dataset, stage0, stage1 = _fixture(tmp_path)
    output = tmp_path / "checkpoints/rl_v2_stage2/pretrain_resume_bad"
    config = PretrainConfig(
        dataset_path=dataset, stage0_marker=stage0, stage1_marker=stage1,
        output_dir=output, sequence_len=4, batch_sequences=1,
        epochs=1, max_val_chunks=1, device="cpu",
    )
    run_pretraining(config)
    payload = torch.load(output / "pretrain_last.pt", map_location="cpu", weights_only=False)
    payload["dataset_sha256"] = "0" * 64
    bad = output / "wrong_dataset.pt"
    torch.save(payload, bad)
    with pytest.raises(RuntimeError, match="dataset SHA"):
        run_pretraining(config, resume_checkpoint=bad)


import pytest
