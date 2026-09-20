from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize closed-loop action/effect behavior from a generic V3 match report."
    )
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    path = Path(args.report)
    data = json.loads(path.read_text(encoding="utf-8"))
    games = list(data.get("games") or [])
    if not games:
        raise ValueError("match report contains no games")

    actions = {
        "farmer": Counter(),
        "hands": Counter(),
        "market": Counter(),
    }
    effects = Counter()
    longest = []
    for game in games:
        for domain, values in (game.get("action_histograms") or {}).items():
            actions.setdefault(domain, Counter()).update(values or {})
        effects.update(game.get("effective_family_counts") or {})
        longest.append(int(game.get("longest_effectless_streak", 0)))

    unit_total = sum(actions["farmer"].values()) + sum(actions["hands"].values())
    unit_pass = actions["farmer"].get("PASS", 0) + actions["hands"].get("PASS", 0)
    market_total = sum(actions["market"].values())
    market_nop = actions["market"].get("NOP_SLOT", 0)
    market_stop = actions["market"].get("STOP_QUEUE", 0)

    effective_total = sum(effects.values())
    economy_families = {
        key: int(effects.get(key, 0))
        for key in (
            "acquisition",
            "production",
            "service",
            "harvest",
            "deposit",
            "sale",
            "hire",
            "purchase",
            "movement",
        )
    }

    result = {
        "games": len(games),
        "mean_final_money": float(statistics.fmean(
            float(game["final_money"]) for game in games
        )),
        "logistics_completion_ratio": float(
            (
                effects.get("deposit", 0)
                + effects.get("sale", 0)
            )
            / max(
                1,
                effects.get("harvest", 0)
                + effects.get("cargo_pickup", 0),
            )
        ),
        "mean_opponent_money": float(statistics.fmean(
            float(game["opponent_money"]) for game in games
        )),
        "unit_total": int(unit_total),
        "unit_pass": int(unit_pass),
        "unit_pass_rate": float(unit_pass / max(1, unit_total)),
        "market_total": int(market_total),
        "market_nop": int(market_nop),
        "market_stop": int(market_stop),
        "market_inactive_rate": float(
            (market_nop + market_stop) / max(1, market_total)
        ),
        "effective_total": int(effective_total),
        "effective_families": dict(sorted(effects.items())),
        "economy_families": economy_families,
        "mean_longest_effectless_streak": float(statistics.fmean(longest)),
        "action_histograms": {
            domain: dict(sorted(counter.items()))
            for domain, counter in actions.items()
        },
    }

    print("=== FARMOS MATCH BEHAVIOR ===")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(
        "FARMOS_MATCH_BEHAVIOR="
        + json.dumps({
            "unit_pass_rate": result["unit_pass_rate"],
            "market_inactive_rate": result["market_inactive_rate"],
            "effective_total": result["effective_total"],
            "economy_families": result["economy_families"],
            "mean_final_money": result["mean_final_money"],
            "logistics_completion_ratio": result[
                "logistics_completion_ratio"
            ],
        }, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
