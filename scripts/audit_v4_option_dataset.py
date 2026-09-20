from __future__ import annotations

import argparse
import json
from collections import Counter

import pandas as pd


BUCKETS = (
    ("opening", 0, 143),
    ("growth", 144, 359),
    ("mid", 360, 575),
    ("harvest", 576, 647),
    ("liquidation", 648, 718),
)


def summarize(path):
    frame = pd.read_parquet(path)
    result = {
        "rows": int(len(frame)),
        "step_min": int(frame.step.min()),
        "step_max": int(frame.step.max()),
        "unique_steps": int(frame.step.nunique()),
        "splits": {
            str(k): int(v)
            for k, v in frame.split.value_counts().sort_index().items()
        },
        "buckets": {},
    }
    for name, lo, hi in BUCKETS:
        part = frame[(frame.step >= lo) & (frame.step <= hi)]
        result["buckets"][name] = {
            "range": [lo, hi],
            "rows": int(len(part)),
            "unique_steps": int(part.step.nunique()),
            "mean_route_confidence": float(
                part.route_confidence.mean()
            ) if len(part) else None,
            "route_confident_ge_0_05": int(
                (part.route_confidence >= 0.05).sum()
            ),
            "route_confident_ge_0_20": int(
                (part.route_confidence >= 0.20).sum()
            ),
            "top_routes": [
                [int(route), int(count)]
                for route, count in part.route_id.value_counts().head(8).items()
            ],
            "market_modes": {
                str(int(k)): int(v)
                for k, v in part.market_mode_id.value_counts().sort_index().items()
            },
            "phase_ids": {
                str(int(k)): int(v)
                for k, v in part.phase_id.value_counts().sort_index().items()
            },
        }
    # Step coverage is a hard invariant across every episode/seat group.
    bad_groups = []
    for (episode_id, seat), part in frame.groupby(["episode_id", "seat"]):
        steps = part.step.astype(int).tolist()
        if steps != list(range(719)):
            bad_groups.append({
                "episode_id": int(episode_id),
                "seat": int(seat),
                "first": steps[:5],
                "last": steps[-5:],
                "count": len(steps),
            })
            if len(bad_groups) >= 10:
                break
    result["step_coverage_ok"] = not bad_groups
    result["bad_group_examples"] = bad_groups
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    args = parser.parse_args()
    result = summarize(args.dataset)
    print("FARMOS_V4_OPTION_AUDIT=" + json.dumps(result, sort_keys=True))
    if not result["step_coverage_ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
