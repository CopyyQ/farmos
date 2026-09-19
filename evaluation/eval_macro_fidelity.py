from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kaggrl.macro_policy import MacroPolicy
from kaggrl.v45_macro_data import load_v45_macro_data

CASES = [
    (ROOT / "demo_data" / "v45_vs_v17_seed201.json", 0),
    (ROOT / "demo_data" / "v45_vs_v17_seed202_seat1.json", 1),
    (ROOT / "demo_data" / "v45_vs_v17_seed203_seat0.json", 0),
    (ROOT / "demo_data" / "v45_vs_v17_seed204_seat1.json", 1),
]


def market_items(action):
    return tuple((o[0], o[1] if len(o) > 1 else None) for o in action.get("market", []) if o)


def valid_action(action):
    return (
        isinstance(action, dict)
        and isinstance(action.get("farmer"), list)
        and isinstance(action.get("hands"), list)
        and isinstance(action.get("market"), list)
        and len(action.get("market", [])) <= 10
    )


def main():
    routes, new_map, old_map = load_v45_macro_data()
    counts = {"all": 0, "farmer": 0, "market_items": 0, "early": 0, "early_farmer": 0, "early_market": 0, "invalid": 0}
    for path, seat in CASES:
        replay = json.loads(path.read_text())
        policy = MacroPolicy(routes, new_map, old_map)
        for t in range(len(replay["steps"]) - 1):
            obs = dict(replay["steps"][t][seat]["observation"])
            obs.setdefault("step", t)
            teacher = replay["steps"][t + 1][seat]["action"]
            macro = policy.act(obs)
            counts["all"] += 1
            counts["farmer"] += int(tuple(macro.get("farmer", [])) == tuple(teacher.get("farmer", [])))
            counts["market_items"] += int(market_items(macro) == market_items(teacher))
            counts["invalid"] += int(not valid_action(macro))
            if int(obs["step"]) < 360:
                counts["early"] += 1
                counts["early_farmer"] += int(tuple(macro.get("farmer", [])) == tuple(teacher.get("farmer", [])))
                counts["early_market"] += int(market_items(macro) == market_items(teacher))
    metrics = {
        "day0_14_farmer": counts["early_farmer"] / counts["early"],
        "day0_14_market_items": counts["early_market"] / counts["early"],
        "overall_farmer": counts["farmer"] / counts["all"],
        "overall_market_items": counts["market_items"] / counts["all"],
        "invalid_actions": counts["invalid"],
        "turns": counts["all"],
    }
    print(json.dumps(metrics, indent=2))
    assert metrics["day0_14_farmer"] >= 0.99
    assert metrics["day0_14_market_items"] >= 0.97
    assert metrics["overall_farmer"] >= 0.95
    assert metrics["invalid_actions"] == 0


if __name__ == "__main__":
    main()
