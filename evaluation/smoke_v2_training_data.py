from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from kaggrl.v2_training_data import (
    V2EpisodeDataset,
    collate_v2_sequences,
    verify_training_acceptance,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/top_tier/live_v2/transitions.parquet"
STAGE0 = ROOT / "data/top_tier/live_v2/manifests/STAGE0_ACCEPTED.json"
STAGE1 = ROOT / "checkpoints/rl_v2_stage1/STAGE1_ACCEPTED.json"
OUT = ROOT / "checkpoints/rl_v2_stage2/data_loader_smoke.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    acceptance = verify_training_acceptance(STAGE0, STAGE1)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "acceptance": acceptance,
        "dataset_sha256": _sha(DATA),
        "stage0_marker_sha256": _sha(STAGE0),
        "stage1_marker_sha256": _sha(STAGE1),
        "splits": {},
    }
    for split in ("train", "val", "test"):
        dataset = V2EpisodeDataset(DATA, split=split, roles={"active_best"})
        rows = sum(len(sample.rows) for sample in dataset)
        first = dataset[0]
        batch = collate_v2_sequences([first], sequence_len=32)
        model_input_keys = sorted(batch.model_inputs())
        report["splits"][split] = {
            "episodes": len(dataset),
            "rows": rows,
            "roles": sorted({sample.role for sample in dataset}),
            "first_episode_id": first.episode_id,
            "first_episode_rows": len(first.rows),
            "first_chunk_real_rows": int(batch.sequence_mask[0].sum().item()),
            "first_chunk_hand_width": int(batch.flat.hand_action_mask.shape[1]),
            "model_input_keys": model_input_keys,
        }
        del batch, dataset
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
