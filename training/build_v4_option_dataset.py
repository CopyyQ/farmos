from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kaggrl.clock import DEFAULT_EPISODE_STEPS, DEFAULT_TURNS_PER_DAY
from kaggrl.v4_option_dataset import (
    build_route_signature_table,
    contiguous_window_starts,
    encode_option_rows,
)
from kaggrl.v4_options import MARKET_MODES
from kaggrl.v45_macro_data import load_v45_macro_data

SCHEMA_VERSION = "farmos_v4_option_rows_v2_margin"
OBSERVATION_SCHEMA = "macro_semantic_v4_clock_v2"
OBS_DIM = 1024

SCHEMA = pa.schema([
    ("episode_id", pa.int64()),
    ("seat", pa.int8()),
    ("step", pa.int16()),
    ("split", pa.string()),
    ("obs_f16", pa.binary()),
    ("route_id", pa.int16()),
    ("route_mask_bits", pa.int64()),
    ("route_confidence", pa.float32()),
    ("market_mode_id", pa.int8()),
    ("phase_id", pa.int8()),
    ("step_norm", pa.float32()),
    ("remaining_norm", pa.float32()),
    ("final_own_money", pa.float32()),
    ("final_rival_money", pa.float32()),
    ("final_margin", pa.float32()),
    ("terminal_result", pa.int8()),
])


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _records(rows, split):
    return [{
        "episode_id": row.episode_id,
        "seat": row.seat,
        "step": row.step,
        "split": split,
        "obs_f16": row.obs_f16,
        "route_id": row.route_id,
        "route_mask_bits": row.route_mask_bits,
        "route_confidence": row.route_confidence,
        "market_mode_id": row.market_mode_id,
        "phase_id": row.phase_id,
        "step_norm": row.step_norm,
        "remaining_norm": row.remaining_norm,
        "final_own_money": row.final_own_money,
        "final_rival_money": row.final_rival_money,
        "final_margin": row.final_margin,
        "terminal_result": row.terminal_result,
    } for row in rows]


def build_dataset(
    source: pathlib.Path,
    output: pathlib.Path,
    manifest_path: pathlib.Path,
    *,
    horizon: int = 8,
    sequence_len: int = 32,
    max_groups: int | None = None,
) -> dict:
    frame = pd.read_parquet(
        source,
        columns=[
            "episode_id", "seat", "step", "split",
            "state_zlib", "raw_action_json",
            "final_own_money", "final_rival_money",
            "final_margin", "terminal_result",
        ],
    )
    frame = frame.sort_values(
        ["episode_id", "seat", "step"],
        kind="stable",
    ).reset_index(drop=True)

    game_outcomes = frame.groupby(
        ["episode_id", "seat"], sort=False
    ).first()
    routes, _, _ = load_v45_macro_data()
    route_ids = tuple(sorted(int(key) for key in routes))
    signature_table = build_route_signature_table(route_ids=route_ids)

    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if tmp.exists():
        tmp.unlink()

    writer = pq.ParquetWriter(tmp, SCHEMA, compression="zstd")
    route_counts = Counter()
    mode_counts = Counter()
    phase_counts = Counter()
    split_counts = Counter()
    confidence_sum = 0.0
    confident_005 = 0
    confident_020 = 0
    total_rows = 0
    groups = 0
    min_step = None
    max_step = None

    try:
        for (episode_id, seat), part in frame.groupby(
            ["episode_id", "seat"],
            sort=False,
        ):
            if max_groups is not None and groups >= int(max_groups):
                break
            part = part.sort_values("step", kind="stable")
            steps = part["step"].astype(int).tolist()
            # This both validates continuity and guarantees late-tail coverage.
            contiguous_window_starts(steps, min(sequence_len, len(steps)))
            split_values = tuple(dict.fromkeys(part["split"].astype(str)))
            if len(split_values) != 1:
                raise RuntimeError(
                    f"episode/seat crosses splits: {episode_id}/{seat}"
                )
            split = split_values[0]
            rows = encode_option_rows(
                part.to_dict("records"),
                horizon=horizon,
                signature_table=signature_table,
            )
            records = _records(rows, split)
            table = pa.Table.from_pylist(records, schema=SCHEMA)
            writer.write_table(table)

            groups += 1
            total_rows += len(rows)
            split_counts[split] += len(rows)
            for row in rows:
                route_counts[int(row.route_id)] += 1
                mode_counts[int(row.market_mode_id)] += 1
                phase_counts[int(row.phase_id)] += 1
                confidence_sum += float(row.route_confidence)
                confident_005 += int(row.route_confidence >= 0.05)
                confident_020 += int(row.route_confidence >= 0.20)
                min_step = (
                    row.step if min_step is None
                    else min(min_step, row.step)
                )
                max_step = (
                    row.step if max_step is None
                    else max(max_step, row.step)
                )
    finally:
        writer.close()

    tmp.replace(output)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "observation_schema": OBSERVATION_SCHEMA,
        "observation_dim": OBS_DIM,
        "source": str(source),
        "source_sha256": _sha256(source),
        "output": str(output),
        "output_sha256": _sha256(output),
        "horizon": int(horizon),
        "sequence_len": int(sequence_len),
        "episode_steps": int(DEFAULT_EPISODE_STEPS),
        "turns_per_day": int(DEFAULT_TURNS_PER_DAY),
        "route_ids": list(route_ids),
        "market_modes": list(MARKET_MODES),
        "market_bc_modes": [
            "KEEP_ROUTE", "NO_SPEND", "LIQUIDATE_SHED",
        ],
        "market_q_only_modes": [
            "HOLD_SALES", "FRONT_RUN_1", "FRONT_RUN_9",
        ],
        "route_masking": "shop_and_phase_compatible_bitset_v1",
        "objective": "terminal_margin_advantage_weighted_bc_v1",
        "margin_scale": 10000.0,
        "source_games": int(len(game_outcomes)),
        "source_win_rate": float(
            (game_outcomes["final_margin"].astype(float) > 0.0).mean()
        ),
        "source_mean_margin": float(
            game_outcomes["final_margin"].astype(float).mean()
        ),
        "source_median_margin": float(
            game_outcomes["final_margin"].astype(float).median()
        ),
        "rows": int(total_rows),
        "groups": int(groups),
        "step_min": None if min_step is None else int(min_step),
        "step_max": None if max_step is None else int(max_step),
        "split_counts": dict(sorted(split_counts.items())),
        "route_counts": {
            str(key): int(value)
            for key, value in sorted(route_counts.items())
        },
        "market_mode_counts": {
            str(key): int(value)
            for key, value in sorted(mode_counts.items())
        },
        "phase_counts": {
            str(key): int(value)
            for key, value in sorted(phase_counts.items())
        },
        "mean_route_confidence": float(
            confidence_sum / max(1, total_rows)
        ),
        "route_confident_ge_0_05": int(confident_005),
        "route_confident_ge_0_20": int(confident_020),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=pathlib.Path,
        default=ROOT / "data/top_tier/live_v2/transitions.parquet",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=ROOT / "data/top_tier/v4_options.parquet",
    )
    parser.add_argument(
        "--manifest",
        type=pathlib.Path,
        default=ROOT / "data/top_tier/manifests/v4_options.json",
    )
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--sequence-len", type=int, default=32)
    parser.add_argument("--max-groups", type=int, default=None)
    args = parser.parse_args()
    manifest = build_dataset(
        args.source,
        args.output,
        args.manifest,
        horizon=args.horizon,
        sequence_len=args.sequence_len,
        max_groups=args.max_groups,
    )
    print("FARMOS_V4_OPTION_DATASET=" + json.dumps(
        manifest, sort_keys=True
    ))


if __name__ == "__main__":
    main()
