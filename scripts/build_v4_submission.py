from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys
import tarfile
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from kaggrl.v4_option_numpy_runtime import V4OptionNumpyPolicy

Q_SCHEMA = "farmos_v4_counterfactual_q_v1"

FILES = (
    "rollout/v4_option_numpy_adapter.py",
    "rollout/v4_hybrid_agent.py",
    "src/kaggrl/clock.py",
    "src/kaggrl/constants.py",
    "src/kaggrl/macro_policy.py",
    "src/kaggrl/observation.py",
    "src/kaggrl/residual_actions.py",
    "src/kaggrl/v4_hybrid_policy.py",
    "src/kaggrl/v4_market_race.py",
    "src/kaggrl/v4_objective.py",
    "src/kaggrl/v4_option_numpy_runtime.py",
    "src/kaggrl/v4_options.py",
    "src/kaggrl/v45_macro_data.py",
)


def _main_source(
    *,
    q_ready: bool,
    min_route_advantage: float,
    min_market_advantage: float,
) -> str:
    allowed = (
        '("KEEP_ROUTE", "LIQUIDATE_SHED", "HOLD_SALES", '
        '"FRONT_RUN_1", "FRONT_RUN_9")'
        if q_ready
        else '("KEEP_ROUTE",)'
    )
    allow_route = "True" if q_ready else "False"
    return f'''from __future__ import annotations

from pathlib import Path

from rollout.v4_hybrid_agent import V4HybridRolloutAgent
from rollout.v4_option_numpy_adapter import V4NumpyOptionAdapter

_AGENT = None


def _raw_path(configuration):
    if configuration is None:
        return None
    try:
        value = configuration.get("__raw_path__")
    except Exception:
        value = getattr(configuration, "__raw_path__", None)
    return None if not value else str(value)


def _build_agent(configuration):
    raw_path = _raw_path(configuration)
    root = (
        Path(raw_path).resolve().parent
        if raw_path
        else Path.cwd()
    )
    option = V4NumpyOptionAdapter(
        root / "policy.npz",
        allowed_market_modes={allowed},
        allow_route_switch={allow_route},
        route_switch_steps=None,
        min_route_margin_advantage={float(min_route_advantage)!r},
        min_market_margin_advantage={float(min_market_advantage)!r},
        require_positive_predicted_margin=False,
    )
    return V4HybridRolloutAgent(
        option_policy=option,
        min_option_confidence=0.0,
        enable_market_race_ordering=True,
        market_race_min_gain=1.0,
    )


def agent(observation, configuration=None):
    global _AGENT
    if _AGENT is None:
        _AGENT = _build_agent(configuration)
    return _AGENT(observation, configuration)
'''


def build_submission(
    policy: pathlib.Path,
    output: pathlib.Path,
    *,
    min_route_advantage: float = 500.0,
    min_market_advantage: float = 500.0,
    require_q: bool = True,
) -> dict:
    policy = policy.resolve()
    if not policy.is_file():
        raise FileNotFoundError(policy)
    runtime = V4OptionNumpyPolicy.load(policy)
    q_ready = runtime.counterfactual_q_schema == Q_SCHEMA
    if require_q and not q_ready:
        raise RuntimeError(
            "submission policy is not counterfactual-Q ready; "
            f"schema={runtime.counterfactual_q_schema!r}"
        )
    if q_ready and not runtime.counterfactual_q_steps:
        raise RuntimeError(
            "counterfactual-Q policy has no validated Q steps"
        )

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    with tempfile.TemporaryDirectory(prefix="farmos_v4_submission_") as td:
        stage = pathlib.Path(td) / "submission"
        (stage / "kaggrl").mkdir(parents=True)
        (stage / "rollout").mkdir(parents=True)
        (stage / "kaggrl" / "__init__.py").write_text(
            "", encoding="utf-8"
        )
        (stage / "rollout" / "__init__.py").write_text(
            "", encoding="utf-8"
        )
        for relative in FILES:
            source = ROOT / relative
            if not source.is_file():
                raise FileNotFoundError(source)
            target_relative = (
                relative[len("src/"):]
                if relative.startswith("src/")
                else relative
            )
            target = stage / target_relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        shutil.copy2(policy, stage / "policy.npz")
        (stage / "main.py").write_text(
            _main_source(
                q_ready=q_ready,
                min_route_advantage=min_route_advantage,
                min_market_advantage=min_market_advantage,
            ),
            encoding="utf-8",
        )
        metadata = {
            "format": "farmos_v4_submission_v1",
            "q_ready": bool(q_ready),
            "counterfactual_q_schema": runtime.counterfactual_q_schema,
            "counterfactual_q_steps": list(runtime.counterfactual_q_steps),
            "objective_version": runtime.objective_version,
            "margin_scale": float(runtime.margin_scale),
            "min_route_advantage": float(min_route_advantage),
            "min_market_advantage": float(min_market_advantage),
        }
        (stage / "SUBMISSION.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with tarfile.open(output, "w:gz") as archive:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    archive.add(
                        path,
                        arcname=str(path.relative_to(stage)),
                        recursive=False,
                    )

    return {
        **metadata,
        "output": str(output),
        "bytes": int(output.stat().st_size),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument(
        "--min-route-advantage", type=float, default=500.0
    )
    parser.add_argument(
        "--min-market-advantage", type=float, default=500.0
    )
    parser.add_argument(
        "--allow-bc-only",
        action="store_true",
        help="build a macro-safe package even if Q training is absent",
    )
    args = parser.parse_args()
    result = build_submission(
        args.policy,
        args.output,
        min_route_advantage=args.min_route_advantage,
        min_market_advantage=args.min_market_advantage,
        require_q=not args.allow_bc_only,
    )
    print(
        "FARMOS_V4_SUBMISSION="
        + json.dumps(result, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
