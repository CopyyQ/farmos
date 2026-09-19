from __future__ import annotations

import json
import pathlib
import sys
from collections import Counter, defaultdict

import numpy as np
import orjson
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kaggrl.macro_policy import MacroPolicy
from kaggrl.observation import ObservationEncoder
from kaggrl.residual_dataset import derive_market_residual, episode_split, participant_seat, sample_residual_row
from kaggrl.v45_macro_data import load_v45_macro_data

DATA = ROOT / "data" / "top_tier" / "files"
OUT = ROOT / "data" / "top_tier" / "residual_bc_2026-09-12_14.parquet"
MANIFEST = ROOT / "data" / "top_tier" / "manifests" / "residual_bc_2026-09-12_14.json"
DATES = ("2026-09-12", "2026-09-13", "2026-09-14")
EDIT_ID = {"KEEP": 0, "DROP": 1, "REPLACE": 2}
BATCH_ROWS = 1024

SCHEMA = pa.schema([
    ("episode_id", pa.int64()),
    ("team_name", pa.string()),
    ("date", pa.string()),
    ("daily_rank", pa.int8()),
    ("seat", pa.int8()),
    ("step", pa.int16()),
    ("split", pa.string()),
    ("obs_f16", pa.binary()),
    ("edit0", pa.int8()),
    ("edit1", pa.int8()),
    ("order0", pa.string()),
    ("order1", pa.string()),
    ("changed", pa.bool_()),
    ("final_margin", pa.float32()),
])


def order_key(edit):
    if edit.kind != "REPLACE" or edit.order is None:
        return ""
    return orjson.dumps(edit.order).decode("utf-8")


def flush(writer, rows):
    if not rows:
        return
    writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA))
    rows.clear()


def main():
    encoder = ObservationEncoder()
    routes, new_map, old_map = load_v45_macro_data()
    meta = pd.read_parquet(
        DATA / "episodes.parquet",
        columns=["date", "episode_id", "team_name", "participants_json", "daily_rank"],
    )
    meta = meta[meta["date"].isin(DATES)]
    by_episode = {int(eid): group for eid, group in meta.groupby("episode_id")}
    stats = Counter()
    split_rows = Counter()
    split_episodes = defaultdict(set)
    team_rows = Counter()
    order_vocab = set()
    rows = []
    if OUT.exists():
        OUT.unlink()
    writer = pq.ParquetWriter(OUT, SCHEMA, compression="zstd")
    try:
        for date in DATES:
            source = pq.ParquetFile(DATA / f"replays_{date}.parquet")
            for rg in range(source.metadata.num_row_groups):
                table = source.read_row_group(rg, columns=["episode_id", "replay_json"])
                episode_id = int(table["episode_id"][0].as_py())
                perspectives = by_episode.get(episode_id)
                if perspectives is None:
                    continue
                replay = orjson.loads(table["replay_json"][0].as_py())
                rewards = replay.get("rewards") or [0.0, 0.0]
                for meta_row in perspectives.itertuples(index=False):
                    seat = participant_seat(meta_row.participants_json, meta_row.team_name)
                    margin = float(rewards[seat]) - float(rewards[1 - seat])
                    split = episode_split(episode_id)
                    policy = MacroPolicy(routes, new_map, old_map)
                    stats["perspectives"] += 1
                    for step in range(len(replay["steps"]) - 1):
                        obs = dict(replay["steps"][step][seat]["observation"])
                        obs.setdefault("step", step)
                        teacher = replay["steps"][step + 1][seat].get("action") or {
                            "farmer": ["PASS"], "hands": [], "market": []
                        }
                        label = derive_market_residual(policy.act(obs), teacher)
                        stats["turns"] += 1
                        stats["changed"] += int(label.changed)
                        stats["representable"] += int(label.representable)
                        if not label.representable:
                            continue
                        if not sample_residual_row(episode_id, meta_row.team_name, step, label.changed):
                            continue
                        encoded = encoder.encode(obs).astype(np.float16, copy=False).tobytes()
                        order0, order1 = order_key(label.market0), order_key(label.market1)
                        if order0:
                            order_vocab.add(order0)
                        if order1:
                            order_vocab.add(order1)
                        rows.append({
                            "episode_id": episode_id,
                            "team_name": meta_row.team_name,
                            "date": meta_row.date,
                            "daily_rank": int(meta_row.daily_rank),
                            "seat": seat,
                            "step": step,
                            "split": split,
                            "obs_f16": encoded,
                            "edit0": EDIT_ID[label.market0.kind],
                            "edit1": EDIT_ID[label.market1.kind],
                            "order0": order0,
                            "order1": order1,
                            "changed": bool(label.changed),
                            "final_margin": margin,
                        })
                        stats["selected"] += 1
                        stats["selected_changed"] += int(label.changed)
                        split_rows[split] += 1
                        split_episodes[split].add(episode_id)
                        team_rows[meta_row.team_name] += 1
                        if len(rows) >= BATCH_ROWS:
                            flush(writer, rows)
                if (rg + 1) % 100 == 0:
                    print(date, rg + 1, "selected", stats["selected"], flush=True)
    finally:
        flush(writer, rows)
        writer.close()
    summary = {
        "dates": list(DATES),
        "source_unique_episodes": len(by_episode),
        "perspectives": stats["perspectives"],
        "turns": stats["turns"],
        "changed_rate": stats["changed"] / max(1, stats["turns"]),
        "representable_rate": stats["representable"] / max(1, stats["turns"]),
        "selected_rows": stats["selected"],
        "selected_changed_rate": stats["selected_changed"] / max(1, stats["selected"]),
        "split_rows": dict(split_rows),
        "split_episodes": {k: len(v) for k, v in split_episodes.items()},
        "selected_teams": len(team_rows),
        "top_team_rows": dict(team_rows.most_common(20)),
        "selected_order_vocab_size": len(order_vocab),
        "observation_dim": encoder.size,
        "observation_storage": "float16 bytes, little-endian native NumPy",
        "selection": {"changed_modulus": 16, "keep_modulus": 32},
        "split": "sha256(episode_id) mod 10: 0=test, 1=val, else=train",
    }
    MANIFEST.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
