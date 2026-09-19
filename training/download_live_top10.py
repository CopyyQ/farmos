from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import pathlib
import sys
import tempfile
import time

from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kaggrl.live_top10_manifest import build_snapshot, collect_episode_sources

DEFAULT_ROOT = ROOT / "data" / "top_tier" / "live_v2"
MANIFEST_NAME = "live_top10_snapshot.json"


def build_live_manifest(api, root, *, competition="kaggriculture", top_k=10,
                        active_episodes=12, candidate_episodes=3):
    root = pathlib.Path(root)
    (root / "manifests").mkdir(parents=True, exist_ok=True)
    (root / "replays").mkdir(parents=True, exist_ok=True)
    snapshot = build_snapshot(api, competition, top_k=top_k)
    manifest = collect_episode_sources(
        api, snapshot, active_limit=active_episodes, candidate_limit=candidate_episodes
    )
    path = root / "manifests" / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path, manifest


def load_live_manifest(root):
    root = pathlib.Path(root)
    path = root / "manifests" / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(f"frozen live manifest not found: {path}")
    return path, json.loads(path.read_text(encoding="utf-8"))


def prepare_manifest(api, root, *, reuse_manifest=False, competition="kaggriculture",
                     top_k=10, active_episodes=12, candidate_episodes=3):
    if reuse_manifest:
        return load_live_manifest(root)
    return build_live_manifest(
        api,
        root,
        competition=competition,
        top_k=top_k,
        active_episodes=active_episodes,
        candidate_episodes=candidate_episodes,
    )


def select_episode_slice(manifest, *, offset=0, limit=0):
    episodes = list(manifest.get("episodes") or [])
    start = max(0, int(offset))
    selected = episodes[start:]
    if int(limit) > 0:
        selected = selected[: int(limit)]
    return selected


def download_one(api, root, episode_id, retries=3):
    root = pathlib.Path(root)
    target = root / "replays" / f"{int(episode_id)}.json.gz"
    if target.exists() and target.stat().st_size > 1000:
        return "skip", {"episode_id": int(episode_id), "path": str(target)}
    for attempt in range(1, int(retries) + 1):
        try:
            with tempfile.TemporaryDirectory(dir=str(root)) as td:
                cwd = os.getcwd()
                os.chdir(td)
                try:
                    api.competition_episode_replay(int(episode_id))
                finally:
                    os.chdir(cwd)
                src = pathlib.Path(td) / f"episode-{int(episode_id)}-replay.json"
                raw = src.read_bytes()
                sha = hashlib.sha256(raw).hexdigest()
                with gzip.open(target, "wb", compresslevel=1) as fh:
                    fh.write(raw)
                return "download", {
                    "episode_id": int(episode_id),
                    "sha256": sha,
                    "raw_bytes": len(raw),
                    "gzip_bytes": target.stat().st_size,
                }
        except Exception as exc:
            if attempt == int(retries):
                return "error", {"episode_id": int(episode_id), "error": repr(exc)}
            time.sleep(attempt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--competition", default="kaggriculture")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--active-episodes", type=int, default=12)
    ap.add_argument("--candidate-episodes", type=int, default=3)
    ap.add_argument("--reuse-manifest", action="store_true",
                    help="reuse the existing frozen live_top10_snapshot.json without refreshing leaderboard")
    ap.add_argument("--offset", type=int, default=0,
                    help="start index inside the frozen manifest episode order")
    ap.add_argument("--limit", type=int, default=0,
                    help="max replay files to process after offset; 0 means all remaining episodes")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    args = ap.parse_args()

    root = pathlib.Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    manifest_path, manifest = prepare_manifest(
        api,
        root,
        reuse_manifest=args.reuse_manifest,
        competition=args.competition,
        top_k=args.top_k,
        active_episodes=args.active_episodes,
        candidate_episodes=args.candidate_episodes,
    )
    episodes = select_episode_slice(manifest, offset=args.offset, limit=args.limit)
    print(json.dumps({
        "manifest": str(manifest_path),
        "snapshot_time_utc": manifest["snapshot_time_utc"],
        "leaderboard_sha256": manifest["leaderboard_sha256"],
        "teams": len(manifest["teams"]),
        "episodes_selected": len(manifest["episodes"]),
        "episodes_to_process": len(episodes),
    }, sort_keys=True), flush=True)
    downloaded = skipped = errors = 0
    downloads_log = root / "manifests" / "downloads.jsonl"
    errors_log = root / "manifests" / "errors.jsonl"
    for idx, entry in enumerate(episodes, start=1):
        episode_id = int(entry["episode"]["id"])
        status, meta = download_one(api, root, episode_id, args.retries)
        if status == "download":
            downloaded += 1
            with downloads_log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(meta, ensure_ascii=False, sort_keys=True) + "\n")
        elif status == "skip":
            skipped += 1
        else:
            errors += 1
            with errors_log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(meta, ensure_ascii=False, sort_keys=True) + "\n")
        print(f"{idx}/{len(episodes)} episode={episode_id} status={status}", flush=True)

    result = {
        "downloaded": downloaded,
        "skipped_existing": skipped,
        "errors": errors,
        "processed": len(episodes),
        "manifest": str(manifest_path),
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
