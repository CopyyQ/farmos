from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path

from kaggrl.macro_policy import MacroPolicy
from kaggrl.v45_macro_data import load_v45_macro_data


def effective_market(action):
    return [list(x) for x in (action or {}).get("market", []) if x]


def market_items(action, n=2):
    return tuple(tuple(order[:2]) for order in effective_market(action)[:n])


def source_seat(replay, source):
    agents = list((replay.get("info") or {}).get("Agents") or [])
    team_name = str(source.get("team_name") or "")
    submission_id = int(source.get("submission_id") or -1)
    for seat, agent in enumerate(agents):
        aid = int(agent.get("submission_id") or -1)
        name = str(agent.get("Name") or agent.get("name") or "")
        if aid == submission_id or (aid < 0 and name == team_name):
            return seat
    return None


def update_stats(stats, bucket, base, teacher):
    s = stats[bucket]
    s["n"] += 1
    s["farmer"] += base.get("farmer") == teacher.get("farmer")
    s["hands"] += base.get("hands") == teacher.get("hands")
    s["market2_items"] += market_items(base) == market_items(teacher)
    s["market2_orders"] += effective_market(base)[:2] == effective_market(teacher)[:2]


def main():
    root = Path("data/top_tier/live")
    manifest = json.loads((root / "manifests/live_top10_bootstrap.json").read_text())
    by_episode = {int(x["episode"]["id"]): x for x in manifest["episodes"]}
    routes, new_map, old_map = load_v45_macro_data()
    stats = defaultdict(lambda: defaultdict(int))
    unmapped = []

    for path in sorted((root / "replays").glob("*.json.gz")):
        eid = int(path.name.split(".")[0])
        meta = by_episode.get(eid)
        if not meta:
            continue
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            replay = json.load(fh)
        steps = list(replay.get("steps") or [])
        for source in meta["sources"]:
            seat = source_seat(replay, source)
            if seat is None:
                unmapped.append((eid, source["team_name"]))
                continue
            policy = MacroPolicy(routes, new_map, old_map)
            for step in range(max(0, len(steps) - 1)):
                obs = dict(steps[step][seat].get("observation") or {})
                obs["player"] = seat
                obs["step"] = step
                teacher = steps[step + 1][seat].get("action") or {}
                base = policy.act(obs)
                phase = "early" if step < 360 else "late"
                update_stats(stats, "all", base, teacher)
                update_stats(stats, phase, base, teacher)
                update_stats(stats, f"team:{source['team_name']}", base, teacher)

    report = {"unmapped": unmapped}
    for bucket, s in stats.items():
        n = max(s["n"], 1)
        report[bucket] = {
            "n": s["n"],
            "farmer_exact": s["farmer"] / n,
            "hands_exact": s["hands"] / n,
            "market2_items_exact": s["market2_items"] / n,
            "market2_orders_exact": s["market2_orders"] / n,
        }
    out = root / "manifests" / "macro_alignment_live.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    for bucket in ("all", "early", "late"):
        print(bucket, report.get(bucket))
    print("unmapped", len(unmapped))
    for bucket in sorted(k for k in report if k.startswith("team:")):
        print(bucket, report[bucket])


if __name__ == "__main__":
    main()
