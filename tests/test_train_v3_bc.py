import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
from test_train_v2_bc import _fixture, _output_row, _target

from kaggrl.constants import UNIT_OPS
from kaggrl.v2_ledger import MARKET_OPS
from kaggrl.v2_losses import action_loss
from kaggrl.v3_model import TemporalIntentPolicy
from training.train_v3_bc import BCV3Config, run_v3_bc


def _v3_init(path, dataset):
    from test_train_v2_bc import _sha
    torch.manual_seed(20260917)
    model = TemporalIntentPolicy()
    torch.save({
        "format_version": 3,
        "architecture_version": "rl_v3_temporal_attention",
        "model_state": model.state_dict(),
        "dataset_sha256": _sha(dataset),
    }, path)
    return path


def _config(dataset, stage0, stage1, output):
    return BCV3Config(
        dataset_path=dataset,
        stage0_marker=stage0,
        stage1_marker=stage1,
        output_dir=output,
        seed=20260917,
        sequence_len=2,
        batch_sequences=1,
        epochs=1,
        max_train_steps=2,
        max_val_chunks=1,
        device="cpu",
    )


def test_v3_bc_config_declares_closed_loop_gap_thresholds():
    fields = BCV3Config.__dataclass_fields__
    assert fields["sequence_len"].default == 32
    assert fields["max_farmer_op_gap"].default == 0.25
    assert fields["max_hand_op_gap"].default == 0.25
    assert fields["max_market_op_gap"].default == 0.25
    assert fields["max_farmer_pass_fraction"].default == 0.80
    assert fields["max_stop_queue_fraction"].default == 0.95
    assert fields["min_initial_market_continue_accuracy"].default == 0.95


def test_v3_bc_smoke_carries_full_temporal_state_and_writes_diagnostics(tmp_path):
    dataset, stage0, stage1, _ = _fixture(tmp_path)
    init = _v3_init(tmp_path / "v3_init.pt", dataset)
    output = tmp_path / "checkpoints/rl_v3/bc"
    best = run_v3_bc(_config(dataset, stage0, stage1, output), init)
    payload = torch.load(best, map_location="cpu", weights_only=False)
    assert payload["architecture_version"] == "rl_v3_temporal_attention"
    assert payload["train_steps"] == 2
    stats = payload["recurrent_stats"]
    assert stats["temporal_steps"] == 4
    assert stats["state_resets"] == 1
    assert stats["state_carries"] >= 1
    validation = payload["validation_metrics"]
    assert set(validation) >= {
        "expert_history", "free_history", "history_gaps", "collapse",
    }
    assert torch.isfinite(torch.tensor(validation["free_history"]["attention"]["mean_entropy"]))
    assert torch.isfinite(torch.tensor(validation["free_history"]["attention"]["mean_attended_age"]))
    for k in ("1", "4", "8", "16", "32"):
        assert k in validation["free_history"]["attention"]["mass_recent"]
    candidate = validation["free_history"]["op_histograms"]
    expert = validation["free_history"]["target_op_histograms"]
    assert set(candidate["farmer"]) == set(UNIT_OPS)
    assert set(candidate["hands"]) == set(UNIT_OPS)
    assert set(candidate["market"]) == set(MARKET_OPS)
    assert set(expert["farmer"]) == set(UNIT_OPS)
    assert set(expert["market"]) == set(MARKET_OPS)
    assert (output / "bc_last.pt").is_file()
    assert (output / "history.jsonl").is_file()
    assert (output / "manifest.sha256").is_file()


def test_v3_bc_reuses_domain_normalization_for_many_hands():
    one = action_loss(type("O", (), {"rows": (_output_row(1),)})(), (_target(1),))
    many = action_loss(type("O", (), {"rows": (_output_row(20),)})(), (_target(20),))
    assert torch.allclose(one.farmer, many.farmer)
    assert torch.allclose(one.hands, many.hands)
    assert torch.allclose(one.market, many.market)
    assert torch.allclose(one.total, many.total)


def test_v3_tbptt_detach_preserves_values_and_cuts_graph():
    from training.train_v3_pretrain import _detach_states

    model = TemporalIntentPolicy().eval()
    fused = torch.randn(1, 256, requires_grad=True)
    from kaggrl.v2_tensorize import PREV_ACTION_GLOBAL_FEATURES, EFFECT_FEATURES, ECONOMY_FEATURES
    previous = torch.zeros(1, len(PREV_ACTION_GLOBAL_FEATURES))
    effect = torch.zeros(1, len(EFFECT_FEATURES))
    economy = torch.zeros(1, len(ECONOMY_FEATURES))
    _, _, state, _ = model.core.step(fused, previous, effect, economy, None)
    detached = _detach_states(model, [state])[0]
    assert torch.equal(detached.h, state.h)
    assert torch.equal(detached.c, state.c)
    assert torch.equal(detached.memory, state.memory)
    assert detached.h.grad_fn is None
    assert detached.c.grad_fn is None
    assert detached.memory.grad_fn is None


def test_v3_bc_recovery_mix_is_expert_majority_and_recorded(tmp_path):
    from dataclasses import replace
    from test_train_v2_bc import _action, _state
    from training.build_v3_recovery_dataset import write_recovery_rows

    dataset, stage0, stage1, _ = _fixture(tmp_path)
    init = _v3_init(tmp_path / "v3_init_recovery.pt", dataset)
    rows = []
    for step in range(2):
        rows.append({
            "episode_id": 990, "seat": 0, "step": step,
            "state": _state(step, 1), "canonical_action": _action(1, step),
            "previous_action": {}, "previous_effect": {}, "effects": {},
            "final_own_money": 0, "final_margin": 0, "terminal_result": 0,
            "teacher_id": "starter", "teacher_version": "test-starter",
            "supervision_kind": "smoke_only", "learner_model_sha256": "a" * 64,
        })
    recovery = write_recovery_rows(rows, tmp_path / "recovery.jsonl")
    output = tmp_path / "checkpoints/rl_v3/recovery"
    config = replace(
        _config(dataset, stage0, stage1, output), sequence_len=1,
        max_train_steps=4, recovery_dataset_path=recovery, recovery_every=4,
    )
    best = run_v3_bc(config, init)
    payload = torch.load(best, map_location="cpu", weights_only=False)
    assert payload["train_steps"] == 4
    assert payload["recurrent_stats"]["expert_updates"] == 3
    assert payload["recurrent_stats"]["recovery_updates"] == 1
    assert payload["config"]["recovery_every"] == 4
    assert payload["recovery_dataset_sha256"]


def test_cached_teacher_chunk_matches_reference(tmp_path):
    from kaggrl.v2_training_data import V2EpisodeDataset
    from training.train_v3_bc import _teacher_chunk, _teacher_chunk_cached

    dataset, _, _, _ = _fixture(tmp_path)
    train = V2EpisodeDataset(dataset, "train", {"active_best"})
    chunks = list(train.iter_chunks(2))[:2]
    active = [(i, chunk) for i, chunk in enumerate(chunks)]
    torch.manual_seed(991)
    reference = TemporalIntentPolicy().eval()
    cached = TemporalIntentPolicy().eval()
    cached.load_state_dict(reference.state_dict())
    ref_losses, ref_states = _teacher_chunk(
        reference, active, [None] * len(active), torch.device("cpu"),
    )
    new_losses, new_states = _teacher_chunk_cached(
        cached, active, [None] * len(active), torch.device("cpu"),
    )
    assert set(ref_losses) == set(new_losses)
    for key in ref_losses:
        assert torch.allclose(ref_losses[key], new_losses[key], atol=1e-6, rtol=1e-5), key
    for ref_state, new_state in zip(ref_states, new_states):
        for name in ("h", "c", "memory", "valid_length"):
            assert torch.allclose(getattr(ref_state, name), getattr(new_state, name), atol=1e-6, rtol=1e-5), name


def test_strategy_slots_for_rows_follow_frozen_manifest():
    from kaggrl.v3_strategy import build_strategy_manifest
    from training.train_v3_bc import _strategy_slots_for_rows

    manifest = build_strategy_manifest([22, 11, 22])
    slots = _strategy_slots_for_rows(
        [{"team_id": 22}, {"team_id": 11}], manifest, torch.device("cpu"),
    )
    assert slots.dtype == torch.long
    assert slots.tolist() == [1, 0]


def test_strategy_conditioning_is_opt_in_and_validates_team_coverage(tmp_path):
    from kaggrl.v2_training_data import V2EpisodeDataset
    from training.train_v3_bc import (
        _build_training_strategy_manifest, _validate_strategy_coverage,
    )

    assert BCV3Config.__dataclass_fields__["strategy_conditioning"].default is False
    dataset, _, _, _ = _fixture(tmp_path)
    train = V2EpisodeDataset(dataset, "train", {"active_best"})
    val = V2EpisodeDataset(dataset, "val", {"active_best"})
    manifest = _build_training_strategy_manifest(train)
    assert manifest.slot_to_team == (1,)
    try:
        _validate_strategy_coverage(manifest, val)
    except RuntimeError as exc:
        assert "unseen strategy team" in str(exc)
    else:
        raise AssertionError("validation-only team must fail closed")


def test_cached_teacher_chunk_passes_strategy_slots(tmp_path):
    from kaggrl.v2_training_data import V2EpisodeDataset
    from kaggrl.v3_strategy import build_strategy_manifest
    from training.train_v3_bc import _teacher_chunk_cached

    dataset, _, _, _ = _fixture(tmp_path)
    train = V2EpisodeDataset(dataset, "train", {"active_best"})
    chunk = next(train.iter_chunks(2))
    manifest = build_strategy_manifest([1])
    torch.manual_seed(992)
    model = TemporalIntentPolicy(strategy_count=1).eval()
    losses, states = _teacher_chunk_cached(
        model, [(0, chunk)], [None], torch.device("cpu"),
        strategy_manifest=manifest,
    )
    assert torch.isfinite(losses["total"])
    assert states[0] is not None


def test_strategy_conditioned_bc_migrates_v3_checkpoint_and_records_manifest(tmp_path):
    import json
    import pyarrow as pa
    import pyarrow.parquet as pq
    from dataclasses import replace
    from test_train_v2_bc import _sha

    dataset, stage0, stage1, _ = _fixture(tmp_path)
    table = pq.read_table(dataset)
    team_index = table.schema.get_field_index("team_id")
    name_index = table.schema.get_field_index("team_name")
    table = table.set_column(
        team_index, "team_id", pa.array([1] * table.num_rows, type=table["team_id"].type),
    )
    table = table.set_column(
        name_index, "team_name", pa.array(["team-1"] * table.num_rows, type=table["team_name"].type),
    )
    pq.write_table(table, dataset)

    stage0_payload = json.loads(stage0.read_text())
    stage0_payload["artifacts"]["dataset_sha256"] = _sha(dataset)
    stage0.write_text(json.dumps(stage0_payload, sort_keys=True) + "\n")
    stage1_payload = json.loads(stage1.read_text())
    stage1_payload["artifacts"]["stage0_marker_sha256"] = _sha(stage0)
    stage1.write_text(json.dumps(stage1_payload, sort_keys=True) + "\n")

    init = _v3_init(tmp_path / "v3_strategy_init.pt", dataset)
    output = tmp_path / "checkpoints/rl_v3_1/strategy_smoke"
    config = replace(
        _config(dataset, stage0, stage1, output),
        max_train_steps=1, strategy_conditioning=True,
    )
    best = run_v3_bc(config, init)
    payload = torch.load(best, map_location="cpu", weights_only=False)
    assert payload["architecture_version"] == "rl_v3_1_strategy_temporal_attention"
    assert payload["strategy_manifest"]["slot_to_team"] == [1]
    assert len(payload["strategy_manifest_sha256"]) == 64
    assert payload["config"]["strategy_conditioning"] is True
    assert "strategy_embedding.weight" in payload["model_state"]
    manifest_path = output / "strategy_manifest.json"
    assert manifest_path.is_file()
    stored = json.loads(manifest_path.read_text())
    assert stored["slot_to_team"] == [1]
    assert stored["sha256"] == payload["strategy_manifest_sha256"]


def test_validation_chunks_respects_cap_lazily():
    from types import SimpleNamespace
    from kaggrl.v2_training_data import SequenceChunk
    from training.train_v3_bc import _validation_chunks

    class FakeDataset:
        def __init__(self):
            self.yielded = 0
        def iter_chunks(self, sequence_len):
            assert sequence_len == 2
            for index in range(5):
                self.yielded += 1
                yield SequenceChunk(
                    episode_id=index, seat=0, rows=({"step": index},),
                    episode_start=True, episode_end=True,
                )

    dataset = FakeDataset()
    config = SimpleNamespace(sequence_len=2, max_val_chunks=1)
    groups = _validation_chunks(dataset, config)
    assert dataset.yielded == 1
    assert len(groups) == 1


def test_v3_bc_family_balancing_is_opt_in():
    field = BCV3Config.__dataclass_fields__["family_weight_cap"]
    assert field.default is None


def test_v3_bc_records_family_counts_and_weights_when_enabled(tmp_path):
    from dataclasses import replace

    dataset, stage0, stage1, _ = _fixture(tmp_path)
    init = _v3_init(tmp_path / "v3_family_init.pt", dataset)
    output = tmp_path / "checkpoints/rl_v3_1/family_smoke"
    config = replace(
        _config(dataset, stage0, stage1, output),
        max_train_steps=1, family_weight_cap=3.0,
    )
    best = run_v3_bc(config, init)
    payload = torch.load(best, map_location="cpu", weights_only=False)
    assert payload["config"]["family_weight_cap"] == 3.0
    assert set(payload["family_counts"]) == {"unit", "market"}
    assert sum(sum(v.values()) for v in payload["family_counts"].values()) > 0
    assert payload["family_weights"]["unit"]["WAIT"] >= 1.0
    assert payload["family_weights"]["market"]["WAIT"] >= 1.0
    assert max(
        weight
        for domain in payload["family_weights"].values()
        for weight in domain.values()
    ) <= 3.0


def test_training_family_counts_are_scoped_by_unit_and_market(tmp_path):
    from kaggrl.v2_training_data import V2EpisodeDataset
    from training.train_v3_bc import _training_family_counts

    dataset, _, _, _ = _fixture(tmp_path)
    train = V2EpisodeDataset(dataset, "train", {"active_best"})
    counts = _training_family_counts(train)
    assert set(counts) == {"unit", "market"}
    assert sum(counts["unit"].values()) > 0
    assert sum(counts["market"].values()) > 0


def test_v3_bc_teacher_mix_schedule_is_opt_in_and_validated():
    field = BCV3Config.__dataclass_fields__["teacher_mix_schedule"]
    assert field.default == (1.0,)

    from dataclasses import replace
    base = BCV3Config(
        dataset_path=Path("dataset"),
        stage0_marker=Path("stage0"),
        stage1_marker=Path("stage1"),
        output_dir=Path("out"),
    )
    replace(base, teacher_mix_schedule=(1.0, 0.75, 0.5)).validate()
    with pytest.raises(ValueError, match="teacher_mix"):
        replace(base, teacher_mix_schedule=(1.0, 1.1)).validate()
    with pytest.raises(ValueError, match="teacher_mix"):
        replace(base, teacher_mix_schedule=()).validate()


def test_teacher_chunk_cached_forwards_teacher_mix_probability(tmp_path):
    import numpy as np
    from kaggrl.v2_training_data import V2EpisodeDataset
    from training.train_v3_bc import _teacher_chunk_cached

    dataset, _, _, _ = _fixture(tmp_path)
    train = V2EpisodeDataset(dataset, "train", {"active_best"})
    chunk = next(train.iter_chunks(2))

    class CapturingPolicy(TemporalIntentPolicy):
        def __init__(self):
            super().__init__()
            self.mix_calls = []

        def teacher_step(self, *args, **kwargs):
            self.mix_calls.append(float(kwargs.get("teacher_mix_probability", 1.0)))
            return super().teacher_step(*args, **kwargs)

    model = CapturingPolicy().eval()
    losses, _ = _teacher_chunk_cached(
        model,
        [(0, chunk)],
        [None],
        torch.device("cpu"),
        teacher_mix_probability=0.5,
        conditioning_rng=np.random.default_rng(123),
    )
    assert model.mix_calls
    assert set(model.mix_calls) == {0.5}
    assert torch.isfinite(losses["total"])


@pytest.mark.parametrize("mix", [1.0, 0.0])
def test_v32_gpu_tensor_chunk_matches_legacy_at_mix_endpoints(tmp_path, mix):
    import numpy as np
    from kaggrl.v2_training_data import V2EpisodeDataset
    from kaggrl.v3_2_model import TemporalIntentPolicyV32
    from kaggrl.v3_strategy import build_strategy_manifest
    from training.train_v3_bc import _teacher_chunk_cached

    dataset, _, _, _ = _fixture(tmp_path)
    train = V2EpisodeDataset(dataset, "train", {"active_best"})
    chunk = next(train.iter_chunks(2))
    active = [(0, chunk)]
    manifest = build_strategy_manifest([1])

    torch.manual_seed(771)
    legacy_model = TemporalIntentPolicyV32(strategy_count=1).eval()
    tensor_model = TemporalIntentPolicyV32(strategy_count=1).eval()
    tensor_model.load_state_dict(legacy_model.state_dict())

    legacy, legacy_states = _teacher_chunk_cached(
        legacy_model,
        active,
        [None],
        torch.device("cpu"),
        strategy_manifest=manifest,
        teacher_mix_probability=mix,
        conditioning_rng=np.random.default_rng(123),
        gpu_tensor_training=False,
    )
    tensor, tensor_states = _teacher_chunk_cached(
        tensor_model,
        active,
        [None],
        torch.device("cpu"),
        strategy_manifest=manifest,
        teacher_mix_probability=mix,
        conditioning_rng=np.random.default_rng(123),
        gpu_tensor_training=True,
    )
    assert set(legacy) == set(tensor)
    for key in legacy:
        assert torch.allclose(
            tensor[key], legacy[key], atol=1e-5, rtol=1e-5
        ), key
    assert torch.allclose(
        tensor_states[0].h,
        legacy_states[0].h,
        atol=1e-6,
        rtol=0.0,
    )


def test_v32_gpu_tensor_chunk_backpropagates(tmp_path):
    import numpy as np
    from kaggrl.v2_training_data import V2EpisodeDataset
    from kaggrl.v3_2_model import TemporalIntentPolicyV32
    from kaggrl.v3_strategy import build_strategy_manifest
    from training.train_v3_bc import _teacher_chunk_cached

    dataset, _, _, _ = _fixture(tmp_path)
    train = V2EpisodeDataset(dataset, "train", {"active_best"})
    chunk = next(train.iter_chunks(2))
    model = TemporalIntentPolicyV32(strategy_count=1).train()
    losses, _ = _teacher_chunk_cached(
        model,
        [(0, chunk)],
        [None],
        torch.device("cpu"),
        strategy_manifest=build_strategy_manifest([1]),
        teacher_mix_probability=0.5,
        conditioning_rng=np.random.default_rng(321),
        gpu_tensor_training=True,
    )
    losses["total"].backward()
    grads = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_teacher_mix_probability_uses_epoch_schedule_and_holds_last_value():
    from training.train_v3_bc import _teacher_mix_for_epoch

    assert _teacher_mix_for_epoch((1.0, 0.75, 0.5), 1) == 1.0
    assert _teacher_mix_for_epoch((1.0, 0.75, 0.5), 2) == 0.75
    assert _teacher_mix_for_epoch((1.0, 0.75, 0.5), 3) == 0.5
    assert _teacher_mix_for_epoch((1.0, 0.75, 0.5), 9) == 0.5


def test_run_v3_bc_records_active_teacher_mix_probability(tmp_path):
    from dataclasses import replace

    dataset, stage0, stage1, _ = _fixture(tmp_path)
    init = _v3_init(tmp_path / "v3_mix_init.pt", dataset)
    output = tmp_path / "checkpoints/rl_v3_1/mix_smoke"
    config = replace(
        _config(dataset, stage0, stage1, output),
        max_train_steps=1,
        teacher_mix_schedule=(0.5,),
    )
    best = run_v3_bc(config, init)
    payload = torch.load(best, map_location="cpu", weights_only=False)
    assert payload["config"]["teacher_mix_schedule"] == (0.5,)
    assert payload["last_train_metrics"]["teacher_mix_probability"] == 0.5


def test_smoke_only_recovery_step_loss_backprops_market_only_and_keeps_family_weight():
    from types import SimpleNamespace
    from test_train_v2_bc import _output_row, _unit
    from training.train_v3_bc import _recovery_step_loss

    row_output = _output_row(1)
    output = SimpleNamespace(rows=(row_output,))
    hire = {
        "kind": "ORDER", "op": "HIRE",
        "item": None, "quantity": None, "raw": ["HIRE"],
    }
    target = {
        "farmer": _unit("PASS"),
        "hands": [_unit("PASS")],
        "market": [hire],
    }
    loss = _recovery_step_loss(
        output,
        (target,),
        supervision_kind="smoke_only",
        family_weights={"market": {"HIRE": 3.0}},
    )
    from kaggrl.v2_losses import action_loss
    plain_market = action_loss(output, (target,)).market
    assert torch.allclose(loss, 3.0 * plain_market)
    loss.backward()
    assert row_output.farmer.op_logits.grad is None
    assert row_output.hands[0].op_logits.grad is None
    assert row_output.market[0].op_logits.grad is not None


def test_expert_recovery_step_loss_backprops_all_action_domains():
    from types import SimpleNamespace
    from test_train_v2_bc import _output_row, _target
    from training.train_v3_bc import _recovery_step_loss

    row_output = _output_row(1)
    output = SimpleNamespace(rows=(row_output,))
    target = _target(1)
    loss = _recovery_step_loss(
        output, (target,), supervision_kind="expert",
    )
    loss.backward()
    assert row_output.farmer.op_logits.grad is not None
    assert row_output.hands[0].op_logits.grad is not None
    assert row_output.market[0].op_logits.grad is not None


def test_v3_2_architecture_is_opt_in_and_recovery_is_blocked():
    from dataclasses import replace
    from kaggrl.v3_2_schema import ARCHITECTURE_VERSION as V32_ARCH

    field = BCV3Config.__dataclass_fields__["model_architecture"]
    assert field.default == "rl_v3_temporal_attention"
    base = BCV3Config(
        dataset_path=Path("dataset"),
        stage0_marker=Path("stage0"),
        stage1_marker=Path("stage1"),
        output_dir=Path("out"),
    )
    replace(
        base,
        model_architecture=V32_ARCH,
        strategy_conditioning=True,
    ).validate()
    with pytest.raises(ValueError, match="V3.2.*recovery"):
        replace(
            base,
            model_architecture=V32_ARCH,
            strategy_conditioning=True,
            recovery_dataset_path=Path("recovery.jsonl"),
        ).validate()


def test_v3_2_bc_migrates_v3_checkpoint_and_records_new_parameters(tmp_path):
    import json
    import pyarrow as pa
    import pyarrow.parquet as pq
    from dataclasses import replace
    from kaggrl.v3_2_schema import ARCHITECTURE_VERSION as V32_ARCH
    from test_train_v2_bc import _sha

    dataset, stage0, stage1, _ = _fixture(tmp_path)
    table = pq.read_table(dataset)
    team_index = table.schema.get_field_index("team_id")
    name_index = table.schema.get_field_index("team_name")
    table = table.set_column(
        team_index,
        "team_id",
        pa.array([1] * table.num_rows, type=table["team_id"].type),
    )
    table = table.set_column(
        name_index,
        "team_name",
        pa.array(["team-1"] * table.num_rows, type=table["team_name"].type),
    )
    pq.write_table(table, dataset)

    stage0_payload = json.loads(stage0.read_text())
    stage0_payload["artifacts"]["dataset_sha256"] = _sha(dataset)
    stage0.write_text(json.dumps(stage0_payload, sort_keys=True) + "\n")
    stage1_payload = json.loads(stage1.read_text())
    stage1_payload["artifacts"]["stage0_marker_sha256"] = _sha(stage0)
    stage1.write_text(json.dumps(stage1_payload, sort_keys=True) + "\n")

    effective_path = dataset.with_name("effective_actions.parquet")
    effective_rows = [{
        "episode_id": int(row["episode_id"]),
        "seat": int(row["seat"]),
        "step": int(row["step"]),
        "split": str(row["split"]),
        "role": str(row["role"]),
        "effective_action_json": str(row["canonical_action_json"]),
        "execution_trace_json": "{}",
        "engine_module_version": "1.32.7",
    } for row in table.to_pylist()]
    pq.write_table(pa.Table.from_pylist(effective_rows), effective_path)
    (dataset.parent / "manifests/effective_actions_summary.json").write_text(
        json.dumps({
            "rows": len(effective_rows),
            "verified_non_eod_transitions": len(effective_rows),
            "verified_replay_files": 1,
            "engine_module_version": "1.32.7",
            "engine_source_sha256": "e" * 64,
            "builder_code_sha256": "b" * 64,
            "source_dataset_sha256": _sha(dataset),
            "source_corpus_manifest_sha256": _sha(
                dataset.parent / "manifests/trusted_corpus_manifest.json"
            ),
            "effective_actions_sha256": _sha(effective_path),
        }, sort_keys=True) + "\n"
    )

    init = _v3_init(tmp_path / "v32_init.pt", dataset)
    output = tmp_path / "checkpoints/rl_v3_2/migration_smoke"
    config = replace(
        _config(dataset, stage0, stage1, output),
        model_architecture=V32_ARCH,
        strategy_conditioning=True,
        max_train_steps=1,
    )
    best = run_v3_bc(config, init)
    payload = torch.load(best, map_location="cpu", weights_only=False)
    assert payload["architecture_version"] == V32_ARCH
    assert payload["strategy_manifest"]["slot_to_team"] == [1]
    assert payload["config"]["model_architecture"] == V32_ARCH
    assert payload["effective_action_sha256"] == _sha(effective_path)
    assert payload["effective_action_engine_version"] == "1.32.7"
    assert payload["effective_action_verified_non_eod_transitions"] == len(effective_rows)
    assert set(payload["migration_new_parameter_keys"]) == {
        "market_active_op_head.bias",
        "market_active_op_head.weight",
        "market_continue_head.bias",
        "market_continue_head.weight",
        "opening_strategy_head.bias",
        "opening_strategy_head.weight",
        "strategy_embedding.weight",
    }
    assert "market_active_op_head.weight" in payload["model_state"]
    assert "market_continue_head.weight" in payload["model_state"]


def test_v3_2_selection_score_uses_active_market_and_continue_metrics():
    from kaggrl.v3_2_schema import ARCHITECTURE_VERSION as V32_ARCH
    from training.train_v3_bc import _selection_score

    semantic = {
        "farmer_semantic_exact": 0.8,
        "mean_hand_semantic_exact": 0.6,
        "market_sequence_exact": 0.99,
        "market_active_semantic_exact": 0.4,
        "market_continue_accuracy": 0.7,
        "full_joint_step_exact": 0.2,
    }
    score = _selection_score(semantic, model_architecture=V32_ARCH)
    expected = (
        0.25 * 0.8
        + 0.25 * 0.6
        + 0.25 * 0.4
        + 0.15 * 0.7
        + 0.10 * 0.2
    )
    assert score == pytest.approx(expected)


def test_v3_2_collapse_rejects_zero_buy_and_zero_sell_predictions():
    from dataclasses import replace
    from kaggrl.v3_2_schema import ARCHITECTURE_VERSION as V32_ARCH
    from training.train_v3_bc import _collapse_report

    config = replace(
        BCV3Config(
            dataset_path=Path("dataset"),
            stage0_marker=Path("stage0"),
            stage1_marker=Path("stage1"),
            output_dir=Path("out"),
        ),
        model_architecture=V32_ARCH,
        strategy_conditioning=True,
    )
    hist = {
        "farmer": {"PASS": 1, "NORTH": 9},
        "hands": {"PASS": 1, "NORTH": 9},
        "market": {
            "STOP_QUEUE": 3,
            "NOP_SLOT": 0,
            "BUY_SEED": 0,
            "BUY_PRODUCT": 0,
            "BUY_ANIMAL": 0,
            "SELL": 0,
            "HIRE": 7,
            "BUY_LAND": 0,
        },
    }
    targets = {
        "farmer": {"PASS": 1, "NORTH": 9},
        "hands": {"PASS": 1, "NORTH": 9},
        "market": {
            "STOP_QUEUE": 3,
            "NOP_SLOT": 0,
            "BUY_SEED": 2,
            "BUY_PRODUCT": 1,
            "BUY_ANIMAL": 1,
            "SELL": 2,
            "HIRE": 3,
            "BUY_LAND": 0,
        },
    }
    semantic = {
        "market_buy_target_count": 4,
        "market_sell_target_count": 2,
    }
    expert = {
        "op_histograms": targets,
        "target_op_histograms": targets,
        "semantic": semantic,
        "attention": {"finite": True},
    }
    free = {
        "op_histograms": hist,
        "target_op_histograms": targets,
        "semantic": semantic,
        "attention": {"finite": True},
    }
    report = _collapse_report(
        config,
        expert,
        free,
        {"farmer_op": 0.0, "hands_op": 0.0, "market_op": 0.0},
    )
    assert "zero_buy_prediction" in report["failures"]
    assert "zero_sell_prediction" in report["failures"]
    assert report["passed"] is False


def test_v32_migration_reuses_legacy_active_and_preserves_seeded_continue_init():
    from kaggrl.v3_2_model import TemporalIntentPolicyV32
    from kaggrl.v3_2_schema import ACTIVE_MARKET_OPS, CONTINUE_ID, STOP_ID
    from training.train_v3_bc import _initialize_v32_migration

    model = TemporalIntentPolicyV32(strategy_count=3)
    with torch.no_grad():
        legacy_weight = torch.arange(
            model.market_op_head.weight.numel(), dtype=torch.float32,
        ).reshape_as(model.market_op_head.weight) / 1000.0
        legacy_bias = torch.linspace(
            -0.4, 0.3, steps=model.market_op_head.bias.numel(),
        )
        model.market_op_head.weight.copy_(legacy_weight)
        model.market_op_head.bias.copy_(legacy_bias)
        model.market_active_op_head.weight.zero_()
        model.market_active_op_head.bias.zero_()
        initial_continue_weight = model.market_continue_head.weight.detach().clone()
        initial_continue_bias = model.market_continue_head.bias.detach().clone()
        model.strategy_embedding.weight.fill_(7.0)

    _initialize_v32_migration(
        model,
        "rl_v3_temporal_attention",
        {"STOP": 3, "CONTINUE": 7},
    )

    active_indices = [MARKET_OPS.index(op) for op in ACTIVE_MARKET_OPS]
    assert torch.allclose(
        model.market_active_op_head.weight,
        legacy_weight[active_indices],
    )
    assert torch.allclose(
        model.market_active_op_head.bias,
        legacy_bias[active_indices],
    )
    assert torch.allclose(
        model.market_continue_head.weight,
        initial_continue_weight,
    )
    assert torch.allclose(
        model.market_continue_head.bias,
        initial_continue_bias,
    )
    assert torch.count_nonzero(model.strategy_embedding.weight).item() == 0


def test_v32_market_continue_counts_match_binary_loss_targets():
    from types import SimpleNamespace
    from training.train_v3_bc import _training_market_continue_counts

    episode = SimpleNamespace(rows=({
        "canonical_action": {
            "farmer": {"op": "PASS"},
            "hands": [],
            "market": [
                {"kind": "STOP_QUEUE", "op": None},
                {"kind": "NOP_SLOT", "op": None},
                {"kind": "ORDER", "op": "HIRE"},
            ],
        },
    },))
    counts = _training_market_continue_counts((episode,))
    assert counts == {"STOP": 1, "CONTINUE": 2}


def test_v3_2_collapse_rejects_zero_stop_and_zero_aligned_sell_recall():
    from dataclasses import replace
    from kaggrl.v3_2_schema import ARCHITECTURE_VERSION as V32_ARCH
    from training.train_v3_bc import _collapse_report

    config = replace(
        BCV3Config(
            dataset_path=Path("dataset"),
            stage0_marker=Path("stage0"),
            stage1_marker=Path("stage1"),
            output_dir=Path("out"),
        ),
        model_architecture=V32_ARCH,
        strategy_conditioning=True,
    )
    hist = {
        "farmer": {"PASS": 1, "NORTH": 9},
        "hands": {"PASS": 1, "NORTH": 9},
        "market": {
            "STOP_QUEUE": 0, "NOP_SLOT": 2,
            "BUY_SEED": 2, "BUY_PRODUCT": 0, "BUY_ANIMAL": 0,
            "SELL": 4, "HIRE": 2, "BUY_LAND": 0,
        },
    }
    targets = {
        "farmer": {"PASS": 1, "NORTH": 9},
        "hands": {"PASS": 1, "NORTH": 9},
        "market": {
            "STOP_QUEUE": 3, "NOP_SLOT": 2,
            "BUY_SEED": 2, "BUY_PRODUCT": 0, "BUY_ANIMAL": 0,
            "SELL": 2, "HIRE": 2, "BUY_LAND": 0,
        },
    }
    semantic = {
        "market_buy_target_count": 2,
        "market_sell_target_count": 2,
        "market_buy_op_recall": 0.5,
        "market_sell_op_recall": 0.0,
    }
    report = _collapse_report(
        config,
        {"attention": {"finite": True}},
        {
            "op_histograms": hist,
            "target_op_histograms": targets,
            "semantic": semantic,
            "attention": {"finite": True},
        },
        {"farmer_op": 0.0, "hands_op": 0.0, "market_op": 0.0},
    )
    assert "zero_stop_prediction" in report["failures"]
    assert "zero_sell_recall" in report["failures"]
    assert report["passed"] is False


def test_v3_2_collapse_rejects_initial_market_stop_at_episode_start():
    from dataclasses import replace
    from kaggrl.v3_2_schema import ARCHITECTURE_VERSION as V32_ARCH
    from training.train_v3_bc import _collapse_report

    config = replace(
        BCV3Config(
            dataset_path=Path("dataset"),
            stage0_marker=Path("stage0"),
            stage1_marker=Path("stage1"),
            output_dir=Path("out"),
        ),
        model_architecture=V32_ARCH,
        strategy_conditioning=True,
    )
    hist = {
        "farmer": {"PASS": 1, "NORTH": 9},
        "hands": {"PASS": 1, "NORTH": 9},
        "market": {
            "STOP_QUEUE": 3, "NOP_SLOT": 1,
            "BUY_SEED": 1, "BUY_PRODUCT": 1, "BUY_ANIMAL": 1,
            "SELL": 2, "HIRE": 3, "BUY_LAND": 0,
        },
    }
    semantic = {
        "market_buy_target_count": 3,
        "market_sell_target_count": 2,
        "market_buy_op_recall": 0.5,
        "market_sell_op_recall": 0.5,
    }
    side = {
        "op_histograms": hist,
        "target_op_histograms": hist,
        "semantic": semantic,
        "attention": {"finite": True},
    }
    report = _collapse_report(
        config, side, side,
        {"farmer_op": 0.0, "hands_op": 0.0, "market_op": 0.0},
        initial_state={
            "rows": 10,
            "market_continue_accuracy": 0.0,
            "target_continue_count": 10,
            "predicted_continue_count": 0,
        },
    )
    assert "initial_market_stop_collapse" in report["failures"]
    assert report["initial_market_continue_accuracy"] == 0.0
    assert report["passed"] is False


def test_v32_collapse_rejects_zero_buy_animal_prediction():
    from dataclasses import replace
    from kaggrl.v3_2_schema import ARCHITECTURE_VERSION as V32_ARCH
    from training.train_v3_bc import _collapse_report

    config = replace(
        BCV3Config(
            dataset_path=Path("dataset"),
            stage0_marker=Path("stage0"),
            stage1_marker=Path("stage1"),
            output_dir=Path("out"),
        ),
        model_architecture=V32_ARCH,
        strategy_conditioning=True,
    )
    hist = {
        "farmer": {"PASS": 1, "NORTH": 9},
        "hands": {"PASS": 1, "NORTH": 9},
        "market": {
            "STOP_QUEUE": 3, "NOP_SLOT": 2,
            "BUY_SEED": 3, "BUY_PRODUCT": 2, "BUY_ANIMAL": 0,
            "SELL": 2, "HIRE": 2, "BUY_LAND": 0,
        },
    }
    targets = {
        "farmer": {"PASS": 1, "NORTH": 9},
        "hands": {"PASS": 1, "NORTH": 9},
        "market": {
            "STOP_QUEUE": 3, "NOP_SLOT": 2,
            "BUY_SEED": 2, "BUY_PRODUCT": 2, "BUY_ANIMAL": 2,
            "SELL": 2, "HIRE": 2, "BUY_LAND": 0,
        },
    }
    semantic = {
        "market_buy_target_count": 6,
        "market_buy_animal_target_count": 2,
        "market_sell_target_count": 2,
        "market_buy_op_recall": 0.5,
        "market_buy_animal_op_recall": 0.0,
        "market_sell_op_recall": 0.5,
    }
    report = _collapse_report(
        config,
        {"attention": {"finite": True}},
        {
            "op_histograms": hist,
            "target_op_histograms": targets,
            "semantic": semantic,
            "attention": {"finite": True},
        },
        {"farmer_op": 0.0, "hands_op": 0.0, "market_op": 0.0},
        initial_state={"rows": 10, "market_continue_accuracy": 1.0},
    )
    assert "zero_buy_animal_prediction" in report["failures"]
    assert "zero_buy_animal_recall" in report["failures"]
    assert report["passed"] is False
