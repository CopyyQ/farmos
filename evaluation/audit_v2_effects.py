from __future__ import annotations

import argparse
import gzip
import json
import pathlib
import time

from kaggrl.v2_effects import derive_effects
from kaggrl.v2_observation import normalize_observation


def audit_effects(root: pathlib.Path) -> dict:
    manifest_path = root / "manifests" / "live_top10_snapshot.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    transitions = assignments = errors = 0
    first_error = None
    started = time.time()

    for entry in manifest["episodes"]:
        episode_id = int(entry["episode"]["id"])
        replay_path = root / "replays" / f"{episode_id}.json.gz"
        with gzip.open(replay_path, "rt", encoding="utf-8") as fh:
            replay = json.load(fh)
        for source in entry["sources"]:
            if source.get("role") != "active_best":
                continue
            assignments += 1
            seat = int(source["seat"])
            steps = replay["steps"]
            for t in range(len(steps) - 1):
                obs_t = steps[t][seat].get("observation") or {}
                action_t = steps[t + 1][seat].get("action") or {}
                obs_t1 = steps[t + 1][seat].get("observation") or {}
                try:
                    normalize_observation(obs_t)
                    derive_effects(obs_t, action_t, obs_t1)
                    transitions += 1
                except Exception as exc:
                    errors += 1
                    if first_error is None:
                        first_error = {
                            "episode_id": episode_id,
                            "seat": seat,
                            "step": t,
                            "error": repr(exc),
                        }

    return {
        "active_assignments": assignments,
        "transitions": transitions,
        "errors": errors,
        "first_error": first_error,
        "elapsed_seconds": round(time.time() - started, 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/top_tier/live_v2")
    args = ap.parse_args()
    root = pathlib.Path(args.root)
    report = audit_effects(root)
    out = root / "manifests" / "effect_audit.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, sort_keys=True), flush=True)
    if report["errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
