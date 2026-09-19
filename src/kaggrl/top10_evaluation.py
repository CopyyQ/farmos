from __future__ import annotations

import json
from pathlib import Path


def _mean_metric(docs: list[dict], split: str, key: str) -> float:
    values = [float(doc[split][key]) for doc in docs if split in doc and key in doc[split]]
    return sum(values) / max(1, len(values))


def summarize_metrics(metrics_files: list[Path]) -> dict:
    docs = [json.loads(Path(path).read_text()) for path in metrics_files]
    docs.sort(key=lambda doc: str(doc["team_name"]))
    if not docs:
        raise ValueError("no expert metrics provided")
    fields = ("farmer_op_acc", "hand_op_acc", "market_op_acc")
    aggregate = {
        split: {key: _mean_metric(docs, split, key) for key in fields}
        for split in ("val", "test")
    }
    best = max(docs, key=lambda doc: float(doc.get("best_val_score", -1.0)))
    return {
        "team_count": len(docs),
        "teams": [doc["team_name"] for doc in docs],
        "gate_pass_count": sum(bool(doc.get("gate", {}).get("pass")) for doc in docs),
        "best_validation_expert": best["team_name"],
        "best_validation_score": float(best.get("best_val_score", -1.0)),
        "aggregate": aggregate,
        "per_team": {doc["team_name"]: doc for doc in docs},
    }
