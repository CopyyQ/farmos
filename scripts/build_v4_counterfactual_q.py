from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from kaggle_environments import make
from kaggrl.clock import resolve_clock
from kaggrl.observation import ObservationEncoder
from kaggrl.v4_options import MARKET_MODES, V4Option
from rollout.v4_hybrid_agent import V4HybridRolloutAgent

FAIL_MARGIN = -100000.0
DEFAULT_Q_MARKET_MODES = (
    "KEEP_ROUTE",
    "LIQUIDATE_SHED",
    "HOLD_SALES",
    "FRONT_RUN_1",
    "FRONT_RUN_9",
)
DEFAULT_V50 = os.environ.get(
    "FARMOS_V50_AGENT",
    "/home/dmin/kaggrilture_copy/source_public/extracted_v50/main.py",
)


def _parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _resolve_agent(path: str | pathlib.Path) -> pathlib.Path:
    value = pathlib.Path(path).resolve()
    if value.is_dir():
        value = value / "main.py"
    if not value.is_file():
        raise FileNotFoundError(value)
    return value


@dataclass
class GameResult:
    margin: float
    own_money: int
    rival_money: int
    done: bool
    statuses: list[str]
    applied: bool = False


class CaptureMacroAgent:
    def __init__(self):
        self.agent = V4HybridRolloutAgent()
        self.encoder = ObservationEncoder(clock_schema="v4")
        self.features: dict[int, np.ndarray] = {}
        self.base_route: dict[int, int] = {}
        self.compatible_routes: dict[int, tuple[int, ...]] = {}

    def __call__(self, observation, configuration=None):
        obs = dict(observation)
        clock = resolve_clock(obs, configuration)
        self.features[clock.step] = self.encoder.encode(
            obs, configuration
        ).astype(np.float16, copy=False)
        action = self.agent(obs, configuration)
        strategy = self.agent.policy.last_strategy or {}
        self.base_route[clock.step] = int(
            strategy.get("base_route_id", 0)
        )
        self.compatible_routes[clock.step] = tuple(
            int(value)
            for value in self.agent.policy.base.compatible_route_ids(
                obs, configuration
            )
        )
        return action


class ForceWindowOption:
    def __init__(
        self,
        step: int,
        *,
        horizon: int,
        route_id: int | None = None,
        market_mode: str = "KEEP_ROUTE",
    ):
        self.step = int(step)
        self.horizon = max(1, int(horizon))
        self.route_id = None if route_id is None else int(route_id)
        self.market_mode = str(market_mode)
        self.applied = False
        self.applied_steps = 0

    def __call__(self, observation, configuration, context):
        current = int(context.step)
        if not self.step <= current < self.step + self.horizon:
            return None
        if (
            self.route_id is not None
            and self.route_id not in context.route_ids
        ):
            return None
        self.applied = True
        self.applied_steps += 1
        return V4Option(
            route_id=self.route_id,
            market_mode=self.market_mode,
        ), 1.0


def _run(
    opponent_path: pathlib.Path,
    seed: int,
    seat: int,
    *,
    option=None,
    capture=False,
):
    learner = CaptureMacroAgent() if capture else V4HybridRolloutAgent(
        option_policy=option,
        min_option_confidence=0.0,
    )
    agents = (
        [learner, str(opponent_path)]
        if seat == 0
        else [str(opponent_path), learner]
    )
    env = make(
        "kaggriculture",
        configuration={"seed": int(seed), "episodeSteps": 720},
        debug=False,
    )
    env.run(agents)
    final = env.steps[-1]
    statuses = [str(row.status) for row in final]
    farms = final[0].observation.farms
    own = int(farms[seat].money)
    rival = int(farms[1 - seat].money)
    done = statuses == ["DONE", "DONE"]
    result = GameResult(
        margin=float(own - rival if done else FAIL_MARGIN),
        own_money=own,
        rival_money=rival,
        done=bool(done),
        statuses=statuses,
        applied=bool(getattr(option, "applied", False)),
    )
    return result, learner


def _sequence_bytes(
    capture: CaptureMacroAgent,
    step: int,
    sequence_len: int,
) -> bytes:
    start = int(step) - int(sequence_len) + 1
    if start < 0:
        raise ValueError(
            f"step {step} is too early for sequence_len={sequence_len}"
        )
    rows = [capture.features[index] for index in range(start, step + 1)]
    array = np.stack(rows).astype(np.float16, copy=False)
    return array.tobytes(order="C")


def build(
    opponent: str | pathlib.Path,
    output: str | pathlib.Path,
    *,
    split_seeds: dict[str, list[int]],
    steps: list[int],
    sequence_len: int = 32,
    option_horizon: int = 8,
    market_modes: tuple[str, ...] = DEFAULT_Q_MARKET_MODES,
) -> dict:
    opponent_path = _resolve_agent(opponent)
    market_modes = tuple(str(value) for value in market_modes)
    unknown_modes = set(market_modes) - set(MARKET_MODES)
    if unknown_modes:
        raise ValueError(
            f"unknown counterfactual market modes: {sorted(unknown_modes)}"
        )
    if "KEEP_ROUTE" not in market_modes:
        raise ValueError("counterfactual market modes must include KEEP_ROUTE")
    output = pathlib.Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for split, seeds in split_seeds.items():
        for seed in seeds:
            for seat in (0, 1):
                baseline, capture = _run(
                    opponent_path, seed, seat, capture=True
                )
                if not baseline.done:
                    raise RuntimeError(
                        f"macro baseline failed seed={seed} seat={seat}: "
                        f"{baseline.statuses}"
                    )
                for step in steps:
                    step = int(step)
                    seq = _sequence_bytes(capture, step, sequence_len)
                    base_route = int(capture.base_route[step])
                    compatible = tuple(
                        capture.compatible_routes[step]
                    )

                    def append_row(
                        family: str,
                        action_id: int,
                        candidate_name: str,
                        result: GameResult,
                    ):
                        rows.append({
                            "split": str(split),
                            "seed": int(seed),
                            "seat": int(seat),
                            "step": step,
                            "sequence_len": int(sequence_len),
                            "obs_seq_f16": seq,
                            "family": str(family),
                            "action_id": int(action_id),
                            "base_route_id": base_route,
                            "baseline_margin": float(baseline.margin),
                            "candidate_margin": float(result.margin),
                            "advantage": float(
                                result.margin - baseline.margin
                            ),
                            "candidate_done": int(result.done),
                            "candidate_name": str(candidate_name),
                            "option_horizon": int(option_horizon),
                        })

                    # Baseline anchors make Q(base) identifiable.
                    append_row(
                        "route", base_route, f"ROUTE_{base_route}", baseline
                    )
                    append_row(
                        "market", 0, "KEEP_ROUTE", baseline
                    )

                    for route_id in compatible:
                        route_id = int(route_id)
                        if route_id == base_route:
                            continue
                        force = ForceWindowOption(
                            step,
                            horizon=option_horizon,
                            route_id=route_id,
                        )
                        candidate, _ = _run(
                            opponent_path, seed, seat, option=force
                        )
                        if force.applied:
                            append_row(
                                "route",
                                route_id,
                                f"ROUTE_{route_id}",
                                candidate,
                            )

                    for mode in market_modes:
                        if mode == "KEEP_ROUTE":
                            continue
                        mode_id = MARKET_MODES.index(mode)
                        force = ForceWindowOption(
                            step,
                            horizon=option_horizon,
                            market_mode=mode,
                        )
                        candidate, _ = _run(
                            opponent_path, seed, seat, option=force
                        )
                        if force.applied:
                            append_row(
                                "market", mode_id, mode, candidate
                            )

                print(
                    json.dumps({
                        "split": split,
                        "seed": seed,
                        "seat": seat,
                        "baseline_margin": baseline.margin,
                        "rows": len(rows),
                    }, sort_keys=True),
                    flush=True,
                )

    frame = pd.DataFrame(rows)
    frame.to_parquet(output, index=False, compression="zstd")
    summary = {
        "schema": "farmos_v4_counterfactual_q_v1",
        "opponent": str(opponent_path),
        "rows": int(len(frame)),
        "games": int(
            sum(len(values) for values in split_seeds.values()) * 2
        ),
        "steps": [int(value) for value in steps],
        "sequence_len": int(sequence_len),
        "option_horizon": int(option_horizon),
        "market_modes": list(market_modes),
        "splits": frame["split"].value_counts().to_dict(),
        "candidate_done_rate": float(frame["candidate_done"].mean()),
        "mean_advantage": float(frame["advantage"].mean()),
        "positive_advantage_rate": float(
            (frame["advantage"] > 0).mean()
        ),
        "negative_advantage_rate": float(
            (frame["advantage"] < 0).mean()
        ),
        "family_counts": frame["family"].value_counts().to_dict(),
        "output": str(output),
    }
    manifest = output.with_suffix(".json")
    manifest.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "FARMOS_V4_COUNTERFACTUAL="
        + json.dumps(summary, sort_keys=True),
        flush=True,
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opponent", default=DEFAULT_V50)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-seeds", default="20310000,20310001")
    parser.add_argument("--val-seeds", default="20310100")
    parser.add_argument("--test-seeds", default="20310200")
    parser.add_argument(
        "--steps",
        default="144,360,576,624,648,672,696,712",
    )
    parser.add_argument("--sequence-len", type=int, default=32)
    parser.add_argument("--option-horizon", type=int, default=8)
    parser.add_argument(
        "--market-modes",
        default=",".join(DEFAULT_Q_MARKET_MODES),
    )
    args = parser.parse_args()
    build(
        args.opponent,
        args.output,
        split_seeds={
            "train": _parse_ints(args.train_seeds),
            "val": _parse_ints(args.val_seeds),
            "test": _parse_ints(args.test_seeds),
        },
        steps=_parse_ints(args.steps),
        sequence_len=args.sequence_len,
        option_horizon=args.option_horizon,
        market_modes=tuple(
            value.strip()
            for value in args.market_modes.split(",")
            if value.strip()
        ),
    )


if __name__ == "__main__":
    main()
