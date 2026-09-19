from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pathlib
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from kaggrl.v2_dataset import assign_temporal_splits, build_arrow_table, iter_transitions


DEFAULT_LIVE = ROOT / "data" / "top_tier" / "live_v2"


def _sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: pathlib.Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _build_code_hashes() -> dict[str, str]:
    return {
        "v2_dataset": _sha256_file(ROOT / "src/kaggrl/v2_dataset.py"),
        "v2_action_schema": _sha256_file(ROOT / "src/kaggrl/v2_action_schema.py"),
        "v2_observation": _sha256_file(ROOT / "src/kaggrl/v2_observation.py"),
        "v2_effects": _sha256_file(ROOT / "src/kaggrl/v2_effects.py"),
        "builder": _sha256_file(pathlib.Path(__file__)),
    }


def _active_source_records(manifest: dict) -> list[dict]:
    records = []
    seen = set()
    for entry in manifest.get("episodes", []):
        episode = entry.get("episode") or {}
        for source in entry.get("sources", []):
            if source.get("role") != "active_best":
                continue
            key = (int(source["team_id"]), int(source["submission_id"]), int(episode["id"]))
            if key in seen:
                raise ValueError(f"duplicate active assignment: {key}")
            seen.add(key)
            records.append({
                "team_id": key[0], "submission_id": key[1], "episode_id": key[2],
                "create_time": str(episode.get("createTime", "")),
            })
    return records


def _flush_rows(writer, rows, temp_path):
    import pyarrow.parquet as pq

    if not rows:
        return writer
    table = build_arrow_table(rows)
    if writer is None:
        writer = pq.ParquetWriter(temp_path, table.schema, compression="zstd")
    writer.write_table(table)
    rows.clear()
    return writer


def build_dataset(manifest_path, replay_dir, out_path, summary_path, corpus_manifest_path, *, chunk_rows=1024):
    manifest_path = pathlib.Path(manifest_path)
    replay_dir = pathlib.Path(replay_dir)
    out_path = pathlib.Path(out_path)
    summary_path = pathlib.Path(summary_path)
    corpus_manifest_path = pathlib.Path(corpus_manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    snapshot_sha = _sha256_file(manifest_path)
    build_code_sha256 = _build_code_hashes()
    source_records = _active_source_records(manifest)
    if not source_records:
        raise ValueError("no active_best assignments in frozen snapshot")
    splits = assign_temporal_splits(source_records)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    if temp_path.exists():
        temp_path.unlink()
    writer = None
    buffer = []
    row_count = assignment_count = 0
    split_rows = Counter()
    corpus_episodes = []

    try:
        for entry in manifest.get("episodes", []):
            episode = entry.get("episode") or {}
            episode_id = int(episode["id"])
            sources = [s for s in entry.get("sources", []) if s.get("role") == "active_best"]
            if not sources:
                continue
            replay_path = replay_dir / f"{episode_id}.json.gz"
            if not replay_path.exists():
                raise FileNotFoundError(replay_path)
            file_sha = _sha256_file(replay_path)
            with gzip.open(replay_path, "rb") as fh:
                raw_json = fh.read()
            json_sha = hashlib.sha256(raw_json).hexdigest()
            replay = json.loads(raw_json.decode("utf-8"))
            episode_meta = dict(episode)
            episode_meta["replay_file_sha256"] = file_sha
            episode_meta["replay_json_sha256"] = json_sha
            expected = max(0, len(replay.get("steps") or []) - 1)

            for source in sources:
                key = (int(source["team_id"]), int(source["submission_id"]), episode_id)
                split = splits[key]
                produced = 0
                for row in iter_transitions(replay, episode_meta, source):
                    row["split"] = split
                    buffer.append(row)
                    produced += 1
                    row_count += 1
                    split_rows[split] += 1
                    if len(buffer) >= int(chunk_rows):
                        writer = _flush_rows(writer, buffer, temp_path)
                if produced != expected:
                    raise ValueError(
                        f"transition count mismatch episode={episode_id} seat={source['seat']} "
                        f"rows={produced} expected={expected}"
                    )
                assignment_count += 1
            corpus_episodes.append({
                "episode_id": episode_id,
                "replay_file_sha256": file_sha,
                "replay_json_sha256": json_sha,
                "active_assignments": len(sources),
            })

        writer = _flush_rows(writer, buffer, temp_path)
        if writer is None:
            raise ValueError("trusted transition dataset is empty")
        writer.close(); writer = None
        temp_path.replace(out_path)
    except Exception:
        if writer is not None:
            writer.close()
        if temp_path.exists():
            temp_path.unlink()
        raise

    corpus_manifest = {
        "selection_snapshot_sha256": snapshot_sha,
        "snapshot_time_utc": manifest.get("snapshot_time_utc"),
        "leaderboard_sha256": manifest.get("leaderboard_sha256"),
        "build_code_sha256": build_code_sha256,
        "episodes": corpus_episodes,
    }
    _write_json(corpus_manifest_path, corpus_manifest)
    summary = {
        "rows": row_count,
        "actor_assignments": assignment_count,
        "split_rows": dict(sorted(split_rows.items())),
        "selection_snapshot_sha256": snapshot_sha,
        "dataset_sha256": _sha256_file(out_path),
        "trusted_corpus_manifest_sha256": _sha256_file(corpus_manifest_path),
    }
    _write_json(summary_path, summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_LIVE))
    parser.add_argument("--chunk-rows", type=int, default=1024)
    args = parser.parse_args()
    live = pathlib.Path(args.root).resolve()
    summary = build_dataset(
        live / "manifests/live_top10_snapshot.json", live / "replays",
        live / "transitions.parquet", live / "manifests/transitions_summary.json",
        live / "manifests/trusted_corpus_manifest.json", chunk_rows=args.chunk_rows,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
