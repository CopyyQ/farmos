from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kaggrl.clock import CLOCK_FEATURES
from kaggrl.v4_objective import OBJECTIVE_VERSION
from kaggrl.v4_option_export import export_v4_option_numpy
from kaggrl.v4_option_model import V4OptionPolicy


def export_checkpoint(checkpoint: pathlib.Path, output: pathlib.Path):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("architecture_version") != V4OptionPolicy.ARCHITECTURE_VERSION:
        raise RuntimeError("unsupported V4 option checkpoint")
    if payload.get("clock_features") != list(CLOCK_FEATURES):
        raise RuntimeError("V4 clock schema mismatch")

    model = V4OptionPolicy(
        int(payload["input_dim"]),
        route_count=len(payload["route_ids"]),
        market_mode_count=len(payload["market_modes"]),
        hidden_dim=int(payload["hidden_dim"]),
        clock_dim=int(payload["clock_dim"]),
    ).eval()
    incompatible = model.load_state_dict(
        payload["model_state"], strict=False
    )
    q_prefixes = (
        "route_value_head.", "route_value_clock_head.",
        "market_value_head.", "market_value_clock_head.",
    )
    missing_q = {
        name for name in incompatible.missing_keys
        if name.startswith(q_prefixes)
    }
    non_q_missing = set(incompatible.missing_keys) - missing_q
    if non_q_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "V4 export checkpoint parameter mismatch: "
            f"missing={sorted(non_q_missing)} "
            f"unexpected={sorted(incompatible.unexpected_keys)}"
        )
    if payload.get("objective_version") == OBJECTIVE_VERSION and missing_q:
        raise RuntimeError(
            f"margin-aware checkpoint missing Q heads: {sorted(missing_q)}"
        )

    export_v4_option_numpy(
        model,
        output,
        route_ids=payload["route_ids"],
        market_modes=payload["market_modes"],
        route_gate_threshold=float(
            payload.get("route_gate_threshold", 0.5)
        ),
        objective_version=str(payload.get("objective_version", "")),
        margin_scale=float(payload.get("margin_scale", 10000.0)),
        counterfactual_q_schema=str(
            payload.get("counterfactual_q_schema", "")
        ),
        counterfactual_q_steps=payload.get(
            "counterfactual_q_steps", []
        ),
    )
    result = {
        "checkpoint": str(checkpoint),
        "output": str(output),
        "epoch": int(payload.get("epoch", -1)),
        "route_gate_threshold": float(
            payload.get("route_gate_threshold", 0.5)
        ),
        "observation_schema": payload.get("observation_schema"),
        "architecture_version": payload.get("architecture_version"),
        "objective_version": payload.get("objective_version"),
        "margin_scale": float(payload.get("margin_scale", 10000.0)),
        "counterfactual_q_schema": str(
            payload.get("counterfactual_q_schema", "")
        ),
        "counterfactual_q_steps": [
            int(value)
            for value in payload.get("counterfactual_q_steps", [])
        ],
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    result = export_checkpoint(args.checkpoint, args.output)
    print("FARMOS_V4_NUMPY_EXPORT=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
