import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
import torch

from kaggrl.constants import ITEM_TO_ID, UNIT_OPS
from kaggrl.v2_dataset import build_arrow_table
from kaggrl.v2_ledger import MARKET_OPS
from kaggrl.v2_losses import action_loss
from kaggrl.v2_model import DecisionOutput, RecurrentIntentPolicy, RowPolicyOutput
from kaggrl.v2_observation import normalize_observation
from training.train_v2_bc import BCConfig, detect_collapse, run_bc


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path

def _unit(op="PASS", item=None, quantity=None):
    return {"op": op, "item": item, "quantity": quantity, "raw": [op]}


def _stop():
    return {"kind": "STOP_QUEUE", "op": None, "item": None, "quantity": None, "raw": []}


def _decision(op_count, action):
    return DecisionOutput(
        op_logits=torch.zeros(op_count, requires_grad=True),
        item_logits=torch.zeros(max(ITEM_TO_ID.values()) + 1, requires_grad=True),
        quantity_logits=None,
        quantity_tokens=None,
        chosen_action=action,
        legal_op_mask=torch.ones(op_count, dtype=torch.bool),
        legal_item_mask=None,
    )


def _output_row(hands):
    return RowPolicyOutput(
        farmer=_decision(len(UNIT_OPS), _unit()),
        hands=tuple(_decision(len(UNIT_OPS), _unit()) for _ in range(hands)),
        market=(_decision(len(MARKET_OPS), _stop()),), trace=(),
    )

def _target(hands):
    return {"farmer": _unit(), "hands": [_unit() for _ in range(hands)], "market": [_stop()]}


def test_bc_action_domain_balance_does_not_scale_with_hand_count():
    one = action_loss(SimpleNamespace(rows=(_output_row(2),)), (_target(2),))
    many = action_loss(SimpleNamespace(rows=(_output_row(20),)), (_target(20),))
    assert torch.allclose(one.farmer, many.farmer)
    assert torch.allclose(one.hands, many.hands)
    assert torch.allclose(one.market, many.market)
    assert torch.allclose(one.total, many.total)


def test_collapse_detector_requires_two_degrading_validations_and_other_domain_gain():
    history = [
        {"farmer_semantic_exact": .82, "mean_hand_semantic_exact": .50, "market_sequence_exact": .60},
        {"farmer_semantic_exact": .75, "mean_hand_semantic_exact": .55, "market_sequence_exact": .62},
        {"farmer_semantic_exact": .68, "mean_hand_semantic_exact": .60, "market_sequence_exact": .65},
    ]
    assert detect_collapse(history, threshold=.05, consecutive=2) == {"farmer"}
    history[-1]["farmer_semantic_exact"] = .74
    assert detect_collapse(history, threshold=.05, consecutive=2) == set()

def _state(step, hands):
    tiles = [[None for _ in range(10)] for _ in range(10)]
    obs = {
        "player": 0, "step": step, "day": step // 24, "hour": step % 24,
        "farms": [
            {"money": 3000 + step, "farmer": [4, 4], "hands": [[5, 4]] * hands,
             "hires_today": hands, "unlocked_quadrants": ["NW"], "tiles": tiles},
            {"money": 2500, "farmer": [8, 8], "hands": [], "hires_today": 0,
             "unlocked_quadrants": ["NW"], "tiles": tiles},
        ],
        "private": {"shed": {"WHEAT": 20}, "seeds": {"WHEAT": 8},
                    "inventories": [{"WHEAT": 2}] + [{"WHEAT": 1}] * hands},
        "market": {"inventory": {"WHEAT": 10000}, "prices": {"WHEAT": 25}},
        "town": {"unlocked_shops": []},
    }
    return asdict(normalize_observation(obs))


def _action(hands, step):
    farmer = _unit("EAST" if step % 2 == 0 else "PASS")
    return {"farmer": farmer, "hands": [_unit() for _ in range(hands)], "market": [_stop()]}

def _row(episode, step, split, hands):
    action = _action(hands, step)
    effects = {
        "money_delta": 1, "hand_count_delta": 0,
        "opponent_public": {"money_delta": 0, "hand_count_delta": 0,
                            "farmer_position_delta": [0, 0], "grid_changed": False,
                            "confidence": "inferred"},
        "action_evidence": [{"actor": "farmer", "op": action["farmer"]["op"],
                             "status": "confirmed", "observed": {}}],
    }
    return {
        "episode_id": episode, "seat": 0, "step": step,
        "create_time": "2026-09-17T00:00:00+00:00",
        "team_id": episode, "team_name": f"team-{episode}",
        "submission_id": 100 + episode, "submission_date": "2026-09-16T00:00:00+00:00",
        "rank": 1, "role": "active_best", "split": split,
        "replay_file_sha256": "a" * 64, "replay_json_sha256": "b" * 64,
        "state": _state(step, hands), "next_state": _state(step + 1, hands),
        "raw_action": {}, "canonical_action": action, "effects": effects,
        "final_own_money": 3200, "final_rival_money": 2500,
        "final_margin": 700, "terminal_result": 1,
    }

def _fixture(tmp_path: Path):
    rows = [_row(1, step, "train", 1) for step in range(4)]
    rows += [_row(2, step, "val", 1) for step in range(4)]
    live = tmp_path / "data/top_tier/live_v2"; live.mkdir(parents=True)
    dataset = live / "transitions.parquet"; pq.write_table(build_arrow_table(rows), dataset)
    manifests = live / "manifests"; manifests.mkdir()
    snapshot = manifests / "live_top10_snapshot.json"; snapshot.write_text("{}")
    corpus = manifests / "trusted_corpus_manifest.json"; corpus.write_text("{}")
    stage0 = _write(manifests / "STAGE0_ACCEPTED.json", {
        "accepted": True, "counters": {"errors": 0},
        "artifacts": {"dataset_sha256": _sha(dataset),
                      "selection_snapshot_sha256": _sha(snapshot),
                      "trusted_corpus_manifest_sha256": _sha(corpus)},
    })
    stage1_dir = tmp_path / "checkpoints/rl_v2_stage1"; stage1_dir.mkdir(parents=True)
    archive = stage1_dir / "policy_init.npz"; archive.write_bytes(b"stage1")
    build = _write(stage1_dir / "build_manifest.json", {"archive": "policy_init.npz"})
    parity = _write(stage1_dir / "parity_report.json", {"passed": True})
    stage1 = _write(stage1_dir / "STAGE1_ACCEPTED.json", {
        "accepted": True, "artifacts": {"stage0_marker_sha256": _sha(stage0),
        "build_manifest_sha256": _sha(build), "parity_report_sha256": _sha(parity),
        "archive_sha256": _sha(archive)}})
    torch.manual_seed(20260917)
    model = RecurrentIntentPolicy()
    init = tmp_path / "checkpoints/rl_v2_stage2/pretrain/pretrain_best.pt"
    init.parent.mkdir(parents=True)
    torch.save({"format_version": 1, "model_state": model.state_dict(),
                "dataset_sha256": _sha(dataset), "config": {"seed": 20260917}}, init)
    return dataset, stage0, stage1, init


def test_bc_smoke_carries_state_across_contiguous_chunks_and_writes_checkpoints(tmp_path):
    dataset, stage0, stage1, init = _fixture(tmp_path)
    output = tmp_path / "checkpoints/rl_v2_stage2/bc"
    config = BCConfig(
        dataset_path=dataset, stage0_marker=stage0, stage1_marker=stage1,
        output_dir=output, seed=20260917, sequence_len=2, batch_sequences=1,
        learning_rate=2e-4, epochs=1, max_train_steps=2, max_val_chunks=1,
        collapse_threshold=.05, collapse_consecutive=2, max_freeze_epochs=2,
    )
    best = run_bc(config, init)
    assert best == output / "bc_best.pt"
    payload = torch.load(best, map_location="cpu", weights_only=False)
    assert payload["train_steps"] == 2
    assert payload["recurrent_stats"]["state_resets"] == 1
    assert payload["recurrent_stats"]["state_carries"] >= 1
    assert payload["config"]["selection_weights"] == {
        "farmer_semantic_exact": .30, "mean_hand_semantic_exact": .30,
        "market_sequence_exact": .30, "full_joint_step_exact": .10,
    }
    validation = payload["validation_metrics"]
    for key in ("farmer_semantic_exact", "mean_hand_semantic_exact",
                "market_sequence_exact", "full_joint_step_exact", "selection_score"):
        assert key in validation and torch.isfinite(torch.tensor(validation[key]))
    assert (output / "bc_last.pt").exists()
    assert (output / "history.jsonl").exists()
    assert (output / "manifest.sha256").exists()


def test_bc_config_freeze_contract_is_explicit():
    fields = BCConfig.__dataclass_fields__
    assert fields["collapse_threshold"].default == .05
    assert fields["collapse_consecutive"].default == 2
    assert fields["max_freeze_epochs"].default == 2


def test_decoder_freeze_contract_freezes_only_degrading_primary_head():
    from training.train_v2_bc import _set_decoder_freeze

    model = RecurrentIntentPolicy()
    frozen = _set_decoder_freeze(model, {"farmer": 2, "market": 0}, epoch=2)
    assert frozen == {"farmer": True, "market": False}
    assert all(not p.requires_grad for p in model.unit_op_head.parameters())
    assert all(p.requires_grad for p in model.market_op_head.parameters())

    frozen = _set_decoder_freeze(model, {"farmer": 2, "market": 0}, epoch=3)
    assert frozen == {"farmer": False, "market": False}
    assert all(p.requires_grad for p in model.unit_op_head.parameters())


def test_bc_config_records_cpu_thread_count_for_reproducibility():
    fields = BCConfig.__dataclass_fields__
    assert fields["torch_num_threads"].default == 2


def test_validation_metrics_are_invariant_to_episode_batching(tmp_path):
    from kaggrl.v2_training_data import V2EpisodeDataset
    from training.train_v2_bc import _free_running_validation, _teacher_validation

    rows = [_row(31, step, "val", 1) for step in range(4)]
    rows += [_row(32, step, "val", 2) for step in range(4)]
    path = tmp_path / "validation.parquet"
    pq.write_table(build_arrow_table(rows), path)
    dataset = V2EpisodeDataset(path, "val", {"active_best"})
    chunks = list(dataset.iter_chunks(2))
    torch.manual_seed(9)
    model = RecurrentIntentPolicy().eval()

    teacher_one = _teacher_validation(model, chunks, batch_sequences=1)
    teacher_many = _teacher_validation(model, chunks, batch_sequences=4)
    for key in teacher_one:
        assert abs(teacher_one[key] - teacher_many[key]) < 1e-6

    free_one = _free_running_validation(model, chunks, 123, batch_sequences=1)
    free_many = _free_running_validation(model, chunks, 123, batch_sequences=4)
    assert free_one == free_many


def test_bc_aux_unit_task_loss_is_normalized_per_step_not_global_unit_count():
    from kaggrl.v2_losses import _masked_unit_mse

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


def test_bc_config_declares_auto_device_contract():
    fields = BCConfig.__dataclass_fields__
    assert fields["device"].default == "auto"
