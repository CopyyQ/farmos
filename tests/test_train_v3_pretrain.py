import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from test_train_v2_pretrain import _fixture

from training.train_v3_pretrain import (
    PretrainV3Config,
    run_v3_pretraining,
)


def _config(dataset, stage0, stage1, output, epochs=1):
    return PretrainV3Config(
        dataset_path=dataset,
        stage0_marker=stage0,
        stage1_marker=stage1,
        output_dir=output,
        seed=20260917,
        sequence_len=2,
        batch_sequences=1,
        learning_rate=3e-4,
        epochs=epochs,
        max_train_steps=2 if epochs == 1 else None,
        max_val_chunks=1,
        device="cpu",
    )

def test_v3_pretraining_steps_time_causally_and_carries_state(tmp_path):
    dataset, stage0, stage1 = _fixture(tmp_path)
    output = tmp_path / "checkpoints/rl_v3/pretrain"
    best = run_v3_pretraining(_config(dataset, stage0, stage1, output))
    payload = torch.load(best, map_location="cpu", weights_only=False)
    assert payload["architecture_version"] == "rl_v3_temporal_attention"
    assert payload["train_steps"] == 2
    stats = payload["recurrent_stats"]
    assert stats["temporal_steps"] == 4
    assert stats["state_resets"] == 1
    assert stats["state_carries"] >= 1
    assert payload["temporal_config"] == {
        "hidden_dim": 256, "attention_dim": 128,
        "heads": 4, "window": 32, "blocks": 1, "dropout": 0.0,
    }
    for value in payload["last_train_metrics"].values():
        assert torch.isfinite(torch.tensor(value))


def test_v3_pretraining_resume_continues_next_epoch(tmp_path):
    dataset, stage0, stage1 = _fixture(tmp_path)
    output = tmp_path / "checkpoints/rl_v3/pretrain_resume"
    first = _config(dataset, stage0, stage1, output, epochs=1)
    run_v3_pretraining(first)
    last = output / "pretrain_last.pt"
    second = _config(dataset, stage0, stage1, output, epochs=2)
    run_v3_pretraining(second, resume_checkpoint=last)
    payload = torch.load(last, map_location="cpu", weights_only=False)
    assert payload["epoch"] == 2
    history = [
        json.loads(line)
        for line in (output / "history.jsonl").read_text().splitlines()
    ]
    assert [row["epoch"] for row in history] == [1, 2]
    assert payload["resolved_device"] == "cpu"


def test_v3_stage1_never_flattens_time_before_temporal_core(tmp_path, monkeypatch):
    from kaggrl.v3_temporal import TemporalCore

    dataset, stage0, stage1 = _fixture(tmp_path)
    output = tmp_path / "checkpoints/rl_v3/pretrain_spy"
    seen_state_none = []
    original = TemporalCore.step

    def wrapped(self, fused, previous_action_global, previous_effect,
                economy, state=None, **kwargs):
        seen_state_none.append(state is None)
        return original(
            self, fused, previous_action_global, previous_effect,
            economy, state, **kwargs,
        )

    monkeypatch.setattr(TemporalCore, "step", wrapped)
    run_v3_pretraining(_config(dataset, stage0, stage1, output))
    assert seen_state_none[:4] == [True, False, False, False]
