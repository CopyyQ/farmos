from __future__ import annotations

import builtins
import importlib.util
import pathlib
import tarfile

from kaggrl.clock import CLOCK_FEATURES
from kaggrl.v4_objective import OBJECTIVE_VERSION
from kaggrl.v4_option_export import export_v4_option_numpy
from kaggrl.v4_option_model import V4OptionPolicy
from kaggrl.v4_options import MARKET_MODES
from scripts.build_v4_submission import build_submission


def _policy(tmp_path: pathlib.Path):
    model = V4OptionPolicy(
        1024,
        route_count=3,
        market_mode_count=len(MARKET_MODES),
        hidden_dim=16,
        clock_dim=len(CLOCK_FEATURES),
    ).eval()
    path = tmp_path / "policy.npz"
    export_v4_option_numpy(
        model,
        path,
        route_ids=[0, 9, 100],
        market_modes=MARKET_MODES,
        route_gate_threshold=0.5,
        objective_version=OBJECTIVE_VERSION,
        margin_scale=10000.0,
        counterfactual_q_schema="farmos_v4_counterfactual_q_v1",
        counterfactual_q_steps=[144, 360, 648],
    )
    return path


def test_submission_builder_requires_counterfactual_q(tmp_path):
    model = V4OptionPolicy(
        1024,
        route_count=1,
        market_mode_count=len(MARKET_MODES),
        hidden_dim=8,
        clock_dim=len(CLOCK_FEATURES),
    ).eval()
    path = tmp_path / "bc.npz"
    export_v4_option_numpy(
        model,
        path,
        route_ids=[0],
        market_modes=MARKET_MODES,
    )
    try:
        build_submission(path, tmp_path / "bad.tar.gz")
    except RuntimeError as exc:
        assert "not counterfactual-Q ready" in str(exc)
    else:
        raise AssertionError("BC-only package should fail closed")


def test_submission_is_self_contained_and_torch_free(tmp_path):
    output = tmp_path / "submission.tar.gz"
    result = build_submission(_policy(tmp_path), output)
    assert result["q_ready"] is True
    assert result["counterfactual_q_steps"] == [144, 360, 648]

    extract = tmp_path / "extract"
    extract.mkdir()
    with tarfile.open(output, "r:gz") as archive:
        archive.extractall(extract)

    assert (extract / "main.py").is_file()
    assert (extract / "policy.npz").is_file()
    assert (extract / "SUBMISSION.json").is_file()

    real_import = builtins.__import__

    def guard(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise RuntimeError("torch import blocked")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = guard
    try:
        spec = importlib.util.spec_from_file_location(
            "farmos_submission_main", extract / "main.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        builtins.__import__ = real_import

    assert callable(module.agent)
    assert module._AGENT is None
    instance = module._build_agent({
        "__raw_path__": str(extract / "main.py"),
    })
    option = instance.policy.option_policy
    assert option.q_rank_ready is True
    assert option.route_switch_steps == frozenset({144, 360, 648})
