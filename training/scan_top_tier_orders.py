from __future__ import annotations

import json
import pathlib
from collections import Counter

import pandas as pd
import pyarrow.parquet as pq

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "top_tier" / "files"
DATES = ("2026-09-12", "2026-09-13", "2026-09-14")


def main():
    meta = pd.read_parquet(DATA / "episodes.parquet", columns=["date", "episode_id", "team_name", "participants_json"])
    meta = meta[meta["date"].isin(DATES)]
    by_episode = {int(eid): group for eid, group in meta.groupby("episode_id")}
    unique_orders = set()
    verbs = Counter()
    quantities = Counter()
    max_quantity = 0
    perspectives = turns = 0
    for date in DATES:
        parquet = pq.ParquetFile(DATA / f"replays_{date}.parquet")
        for rg in range(parquet.metadata.num_row_groups):
            table = parquet.read_row_group(rg, columns=["episode_id", "replay_json"])
            eid = int(table["episode_id"][0].as_py())
            rows = by_episode.get(eid)
            if rows is None:
                continue
            replay = json.loads(table["replay_json"][0].as_py())
            participants_cache = {}
            for row in rows.itertuples(index=False):
                participants = json.loads(row.participants_json)
                seat = participants.index(row.team_name)
                perspectives += 1
                for t in range(len(replay["steps"]) - 1):
                    action = replay["steps"][t + 1][seat].get("action") or {}
                    effective = [o for o in action.get("market", []) if o]
                    turns += 1
                    for order in effective[:2]:
                        key = tuple(order)
                        unique_orders.add(key)
                        verbs[order[0]] += 1
                        if len(order) >= 3:
                            try:
                                q = int(order[2])
                            except (TypeError, ValueError):
                                continue
                            quantities[q] += 1
                            max_quantity = max(max_quantity, q)
    summary = {
        "dates": list(DATES),
        "episodes": len(by_episode),
        "perspectives": perspectives,
        "turns": turns,
        "unique_first_two_orders": len(unique_orders),
        "max_quantity": max_quantity,
        "verbs": dict(verbs),
        "quantity_p50": None,
        "quantity_p95": None,
        "quantity_p99": None,
    }
    if quantities:
        expanded = []
        for q, count in quantities.items():
            expanded.extend([q] * count)
        expanded.sort()
        for name, frac in (("quantity_p50", 0.50), ("quantity_p95", 0.95), ("quantity_p99", 0.99)):
            summary[name] = expanded[min(len(expanded) - 1, int(frac * (len(expanded) - 1)))]
    out = ROOT / "data" / "top_tier" / "manifests" / "order_scan_2026-09-12_14.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
