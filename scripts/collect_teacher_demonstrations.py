from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from training.build_v3_recovery_dataset import (
    collect_teacher_demonstrations,
    teacher_id_from_path,
)


def _parse_seeds(text: str) -> list[int]:
    values = sorted({
        int(item.strip())
        for item in str(text).split(",")
        if item.strip()
    })
    if not values:
        raise ValueError("at least one seed is required")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Collect clean on-policy recovery demonstrations from a "
            "submission teacher such as v50."
        )
    )
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--seeds",
        default="20275000,20275001",
    )
    parser.add_argument("--episode-steps", type=int, default=720)
    parser.add_argument("--strategy-slot", type=int, default=9)
    args = parser.parse_args()

    teacher = Path(args.teacher).resolve()
    output = Path(args.output).resolve()
    teacher_id = teacher_id_from_path(teacher)
    result = collect_teacher_demonstrations(
        teacher,
        output,
        _parse_seeds(args.seeds),
        episode_steps=int(args.episode_steps),
        strategy_slot=int(args.strategy_slot),
        teacher_id=teacher_id,
    )
    meta_path = result.with_suffix(result.suffix + ".meta.json")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    summary = {
        "teacher_id": metadata["teacher_id"],
        "rows": int(metadata["rows"]),
        "clean_label_rate": float(metadata["clean_label_rate"]),
        "dropped_projected_rows": int(
            metadata["dropped_projected_rows"]
        ),
        "projection_corrections": int(
            metadata["projection_corrections"]
        ),
        "strategy_slot": metadata["strategy_slot"],
        "output": str(result),
        "metadata": str(meta_path),
    }
    print(
        "FARMOS_TEACHER_DEMOS="
        + json.dumps(summary, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
