from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kaggrl.clock import CLOCK_FEATURES, resolve_clock
from kaggrl.observation import ObservationEncoder
from kaggrl.v2_dataset import decode_zlib_json
from kaggrl.v4_option_dataset import structured_state_to_observation
from kaggrl.v4_option_model import V4OptionPolicy
from kaggrl.v45_macro_data import load_v45_macro_data
from kaggrl.macro_policy import MacroPolicy


def _load_model(path: pathlib.Path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = V4OptionPolicy(
        int(payload["input_dim"]),
        route_count=len(payload["route_ids"]),
        market_mode_count=len(payload["market_modes"]),
        hidden_dim=int(payload["hidden_dim"]),
        clock_dim=int(payload["clock_dim"]),
    ).eval()
    model.load_state_dict(payload["model_state"], strict=True)
    return payload, model


def _counterfactual_observation(base, step):
    obs = copy.deepcopy(base)
    day, hour = divmod(int(step), 24)
    obs["step"] = int(step)
    obs["day"] = int(day)
    obs["hour"] = int(hour)
    return obs


def audit(
    checkpoint: pathlib.Path,
    source: pathlib.Path,
    *,
    split: str = "val",
    anchor_step: int = 350,
    steps=(50, 350, 680),
):
    payload, model = _load_model(checkpoint)
    encoder = ObservationEncoder(clock_schema="v4")
    frame = pd.read_parquet(
        source,
        columns=["split", "step", "state_zlib"],
    )
    matches = frame[
        frame["split"].eq(split)
        & frame["step"].eq(int(anchor_step))
    ]
    if matches.empty:
        raise RuntimeError("no anchor row for sensitivity audit")
    state = decode_zlib_json(matches.iloc[0]["state_zlib"])
    base = structured_state_to_observation(state)

    routes, new_routes, old_routes = load_v45_macro_data()
    macro = MacroPolicy(routes, new_routes, old_routes)
    route_ids = tuple(int(x) for x in payload["route_ids"])
    route_to_class = {
        route_id: index for index, route_id in enumerate(route_ids)
    }
    market_modes = tuple(payload["market_modes"])

    rows = []
    raw_outputs = {}
    for step in steps:
        obs = _counterfactual_observation(base, int(step))
        clock = resolve_clock(obs)
        encoded = encoder.encode(obs).astype(np.float32, copy=False)
        clock_context = np.asarray(clock.features(), dtype=np.float32)
        with torch.inference_mode():
            output, _ = model.forward_sequence(
                torch.from_numpy(encoded).view(1, 1, -1),
                torch.from_numpy(clock_context).view(1, 1, -1),
                None,
            )
        route_logits = (
            output["route"][0, 0].detach().float().numpy()
        )
        market_logits = (
            output["market"][0, 0].detach().float().numpy()
        )
        compatible = macro.compatible_route_ids(obs)
        compatible_classes = [
            route_to_class[route_id]
            for route_id in compatible
            if route_id in route_to_class
        ]
        masked = np.full_like(route_logits, -np.inf)
        if compatible_classes:
            masked[compatible_classes] = route_logits[compatible_classes]
            selected_route = route_ids[int(np.argmax(masked))]
        else:
            selected_route = macro.route_id(obs)

        market_id = int(np.argmax(market_logits))
        raw_outputs[int(step)] = {
            "route": route_logits,
            "market": market_logits,
            "route_clock": (
                output["route_clock"][0, 0].detach().float().numpy()
            ),
            "market_clock": (
                output["market_clock"][0, 0].detach().float().numpy()
            ),
        }
        rows.append({
            "step": int(step),
            "day": int(clock.day),
            "hour": int(clock.hour),
            "remaining_steps": int(clock.remaining_steps),
            "phase": int(clock.phase_index),
            "compatible_routes": list(compatible),
            "selected_route": int(selected_route),
            "market_mode": market_modes[market_id],
            "route_gate": float(
                output["route_gate"][0, 0].detach().item()
            ),
        })

    anchor = int(anchor_step)
    if anchor not in raw_outputs:
        anchor = int(steps[len(steps) // 2])
    deltas = {}
    for step in steps:
        current = raw_outputs[int(step)]
        reference = raw_outputs[anchor]
        deltas[str(int(step))] = {
            "route_logits_max_abs_vs_anchor": float(np.max(np.abs(
                current["route"] - reference["route"]
            ))),
            "market_logits_max_abs_vs_anchor": float(np.max(np.abs(
                current["market"] - reference["market"]
            ))),
            "route_clock_max_abs_vs_anchor": float(np.max(np.abs(
                current["route_clock"] - reference["route_clock"]
            ))),
            "market_clock_max_abs_vs_anchor": float(np.max(np.abs(
                current["market_clock"] - reference["market_clock"]
            ))),
        }

    non_anchor = [
        value for key, value in deltas.items()
        if int(key) != anchor
    ]
    result = {
        "checkpoint": str(checkpoint),
        "anchor_step": anchor,
        "rows": rows,
        "deltas": deltas,
        "clock_sensitive": bool(
            non_anchor
            and max(
                max(
                    row["route_clock_max_abs_vs_anchor"],
                    row["market_clock_max_abs_vs_anchor"],
                )
                for row in non_anchor
            ) > 1e-3
        ),
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--anchor-step", type=int, default=350)
    parser.add_argument("--steps", default="50,350,680")
    args = parser.parse_args()
    steps = tuple(int(x) for x in args.steps.split(",") if x.strip())
    result = audit(
        args.checkpoint,
        args.source,
        split=args.split,
        anchor_step=args.anchor_step,
        steps=steps,
    )
    print("FARMOS_V4_STEP_SENSITIVITY=" + json.dumps(
        result, sort_keys=True
    ))


if __name__ == "__main__":
    main()
