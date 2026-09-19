from pathlib import Path
import tarfile

from training.build_colab_recovery_bundle import build_recovery_bundle
from training.run_v3_colab_recovery import T4MemoryPolicy, choose_vram_candidate


def test_t4_memory_policy_targets_high_vram_use():
    policy = T4MemoryPolicy()
    assert policy.target_reserved_min_gib == 10.0
    assert policy.target_reserved_max_gib == 13.0
    assert policy.hard_reserved_max_gib == 14.0
    assert choose_vram_candidate(
        [(4, 2.0), (8, 5.5), (16, 10.8), (32, 13.7)], policy
    ) == 16


def test_t4_memory_policy_uses_largest_safe_if_target_not_reached():
    policy = T4MemoryPolicy()
    assert choose_vram_candidate(
        [(4, 1.0), (8, 2.0), (16, 4.0), (32, 7.5)], policy
    ) == 32


def test_recovery_bundle_is_deterministic_and_excludes_secrets(tmp_path):
    root = tmp_path / "repo"; root.mkdir()
    for rel, data in {
        "src/kaggrl/a.py": "x=1\n", "training/train_v3_bc.py": "x=2\n",
        "training/run_v3_colab_recovery.py": "x=3\n", "data/train.parquet": "d",
        "checkpoints/init.pt": "i", "recovery/recovery.jsonl": "{}\n",
        ".env": "SECRET=1\n", "__pycache__/x.pyc": "bad",
    }.items():
        path = root / rel; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(data)
    bundle = build_recovery_bundle(root, tmp_path / "bundle.tar.gz", [
        "src", "training", "data/train.parquet", "checkpoints/init.pt", "recovery/recovery.jsonl",
    ])
    with tarfile.open(bundle, "r:gz") as archive:
        names = sorted(archive.getnames())
    assert "src/kaggrl/a.py" in names
    assert "data/train.parquet" in names
    assert ".env" not in names
    assert not any("__pycache__" in name for name in names)
    assert bundle.with_suffix(bundle.suffix + ".sha256").is_file()


def test_colab_runner_supports_balance_only_without_recovery_dataset():
    from training.run_v3_colab_recovery import _parser

    args = _parser().parse_args([
        "--dataset", "train.parquet",
        "--stage0-marker", "s0.json",
        "--stage1-marker", "s1.json",
        "--init-checkpoint", "init.pt",
        "--output-dir", "out",
        "--max-train-steps", "8",
        "--family-weight-cap", "3.0",
    ])
    assert args.recovery_dataset is None
    assert args.family_weight_cap == 3.0


def test_colab_runner_exposes_strategy_conditioning_flag():
    from training.run_v3_colab_recovery import _parser

    args = _parser().parse_args([
        "--dataset", "train.parquet",
        "--stage0-marker", "s0.json",
        "--stage1-marker", "s1.json",
        "--init-checkpoint", "init.pt",
        "--output-dir", "out",
        "--max-train-steps", "8",
        "--strategy-conditioning",
    ])
    assert args.strategy_conditioning is True


def test_colab_runner_exposes_v32_model_architecture():
    from kaggrl.v3_2_schema import ARCHITECTURE_VERSION as V32_ARCH
    from training.run_v3_colab_recovery import _parser

    args = _parser().parse_args([
        "--dataset", "train.parquet",
        "--stage0-marker", "s0.json",
        "--stage1-marker", "s1.json",
        "--init-checkpoint", "init.pt",
        "--output-dir", "out",
        "--max-train-steps", "8",
        "--strategy-conditioning",
        "--model-architecture", V32_ARCH,
    ])
    assert args.model_architecture == V32_ARCH


def test_colab_probe_factory_reconstructs_v32_strategy_model():
    import torch
    from kaggrl.v3_2_model import TemporalIntentPolicyV32
    from kaggrl.v3_2_schema import ARCHITECTURE_VERSION as V32_ARCH
    from training.run_v3_colab_recovery import _probe_model_from_payload

    source = TemporalIntentPolicyV32(strategy_count=3).eval()
    payload = {
        "architecture_version": V32_ARCH,
        "strategy_manifest": {"slot_to_team": [10, 20, 30]},
        "model_state": source.state_dict(),
    }
    restored = _probe_model_from_payload(payload, torch.device("cpu"))
    assert isinstance(restored, TemporalIntentPolicyV32)
    assert restored.strategy_count == 3
