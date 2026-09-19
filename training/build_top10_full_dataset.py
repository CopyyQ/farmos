from __future__ import annotations

import gzip
import hashlib
import json
import pathlib
import sys
from datetime import datetime, timezone

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kaggrl.actions import ActionCodec
from kaggrl.observation import ObservationEncoder
from kaggrl.top_tier_full_dataset import (
    assign_temporal_splits,
    compute_source_weight,
    iter_actor_examples,
)

LIVE = ROOT / "data" / "top_tier" / "live"
MANIFEST = LIVE / "manifests" / "live_top10_bootstrap.json"
REPLAYS = LIVE / "replays"
OUT = LIVE / "full_action_bc.parquet"
SUMMARY = LIVE / "manifests" / "full_action_bc_summary.json"

def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_replay(episode_id: int) -> dict:
    path = REPLAYS / f"{int(episode_id)}.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def source_records(manifest: dict) -> list[dict]:
    rows = []
    for entry in manifest["episodes"]:
        episode = entry["episode"]
        for source in entry["sources"]:
            rows.append({
                "team_name": source["team_name"],
                "episode_id": int(episode["id"]),
                "create_time": episode["createTime"],
            })
    return rows

def build_frame(manifest: dict) -> tuple[pd.DataFrame, list[dict]]:
    snapshot = parse_time(manifest["snapshot_time_utc"])
    splits = assign_temporal_splits(source_records(manifest))
    encoder = ObservationEncoder(size=1024)
    codec = ActionCodec()
    rows: list[dict] = []
    unmapped: list[dict] = []
    total_assignments = sum(len(entry["sources"]) for entry in manifest["episodes"])
    done = 0
    for entry in manifest["episodes"]:
        episode = entry["episode"]
        episode_id = int(episode["id"])
        replay = load_replay(episode_id)
        expected = max(0, len(replay.get("steps") or []) - 1)
        age_hours = max(0.0, (snapshot - parse_time(episode["createTime"])).total_seconds() / 3600.0)
        for source in entry["sources"]:
            actor_rows = list(iter_actor_examples(replay, episode, source, encoder, codec))
            if len(actor_rows) != expected:
                unmapped.append({"episode_id": episode_id, "team_name": source["team_name"], "rows": len(actor_rows), "expected": expected})
                done += 1
                continue
            split = splits[(source["team_name"], episode_id)]
            base_weight = compute_source_weight(source["rank"], source["role"], age_hours)
            for row in actor_rows:
                row["split"] = split
                row["sample_weight"] = base_weight
            rows.extend(actor_rows)
            done += 1
            if done % 10 == 0 or done == total_assignments:
                print(f"assignments {done}/{total_assignments} rows={len(rows)}", flush=True)
    return pd.DataFrame(rows), unmapped

def main():
    manifest = json.loads(MANIFEST.read_text())
    frame, unmapped = build_frame(manifest)
    if unmapped:
        raise RuntimeError(f"unmapped actor assignments: {unmapped[:5]}")
    if frame.empty:
        raise RuntimeError("full-action dataset is empty")

    frame["sample_weight"] = frame.groupby("team_name")["sample_weight"].transform(lambda s: s / float(s.mean()))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT, index=False, compression="zstd")

    split_rows = frame.groupby(["team_name", "split"]).size().unstack(fill_value=0).to_dict(orient="index")
    split_episodes = frame.groupby(["team_name", "split"])["episode_id"].nunique().unstack(fill_value=0).to_dict(orient="index")
    summary = {
        "snapshot_time_utc": manifest["snapshot_time_utc"],
        "rows": int(len(frame)),
        "teams": sorted(frame.team_name.unique().tolist()),
        "team_count": int(frame.team_name.nunique()),
        "actor_assignments": int(frame[["team_name", "episode_id", "role"]].drop_duplicates().shape[0]),
        "unmapped": unmapped,
        "split_rows": split_rows,
        "split_episodes": split_episodes,
        "weight_mean_by_team": frame.groupby("team_name")["sample_weight"].mean().round(8).to_dict(),
        "artifact_sha256": {
            str(OUT.relative_to(ROOT)): sha256(OUT),
            str(MANIFEST.relative_to(ROOT)): sha256(MANIFEST),
        },
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
