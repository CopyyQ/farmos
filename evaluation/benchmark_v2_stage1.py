from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from kaggrl.v2_export import export_v2_numpy, feature_schema_sha256
from kaggrl.v2_model import RecurrentIntentPolicy
from kaggrl.v2_numpy_runtime import V2NumpyPolicy
from kaggrl.v2_observation import normalize_observation
from kaggrl.v2_tensorize import collate_transitions

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "checkpoints" / "rl_v2_stage1"
STAGE0_MARKER = ROOT / "data" / "top_tier" / "live_v2" / "manifests" / "STAGE0_ACCEPTED.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _structured(hands: int) -> dict:
    own_tiles = [[None for _ in range(10)] for _ in range(10)]
    rival_tiles = [[None for _ in range(10)] for _ in range(10)]
    hand_pos = [[i % 10, (i // 10) % 10] for i in range(hands)]
    inventories = [{"FERTILIZER": 1}] + [
        {"WHEAT": (i % 50) + 1} for i in range(hands)
    ]
    obs = {
        "player": 0, "step": 101, "day": 4, "hour": 5,
        "farms": [
            {"money": 4500, "farmer": [4, 4], "hands": hand_pos,
             "hires_today": 3, "unlocked_quadrants": ["NW", "NE"], "tiles": own_tiles},
            {"money": 3900, "farmer": [8, 8], "hands": [[7, 8]],
             "hires_today": 1, "unlocked_quadrants": ["NW"], "tiles": rival_tiles},
        ],
        "private": {"shed": {"WHEAT": 70, "MILK": 12},
                    "seeds": {"WHEAT": 8, "MELON": 2}, "inventories": inventories},
        "market": {"inventory": {"WHEAT": 9990, "MILK": 10010},
                   "prices": {"WHEAT": 27, "MILK": 155}},
        "town": {"unlocked_shops": ["BAKERY", "PIZZA_SHOP"]},
    }
    from dataclasses import asdict
    return asdict(normalize_observation(obs))


def _previous_action(hands: int) -> dict:
    unit_pass = {"op": "PASS", "item": None, "quantity": None, "raw": ["PASS"]}
    return {
        "farmer": dict(unit_pass),
        "hands": [dict(unit_pass) for _ in range(hands)],
        "market": [{"kind": "STOP_QUEUE", "op": None,
                    "item": None, "quantity": None, "raw": []}],
    }


def _bench(runtime: V2NumpyPolicy, hands: int, samples: int) -> dict:
    state = _structured(hands)
    previous = _previous_action(hands)
    for _ in range(3):
        runtime.step(state, {}, previous, None, np.random.default_rng(0), deterministic=True)
    elapsed = []
    for index in range(samples):
        start = time.perf_counter_ns()
        out = runtime.step(
            state, {}, previous, None, np.random.default_rng(index), deterministic=True
        )
        elapsed.append((time.perf_counter_ns() - start) / 1e6)
        if len(out.canonical_action["hands"]) != hands:
            raise AssertionError("dynamic hand count changed during runtime benchmark")
    values = np.asarray(elapsed, dtype=np.float64)
    return {
        "hands": hands, "samples": samples, "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)), "p99_ms": float(np.percentile(values, 99)),
        "worst_ms": float(values.max()),
    }


def main() -> None:
    torch.manual_seed(20260917)
    OUT.mkdir(parents=True, exist_ok=True)
    model = RecurrentIntentPolicy().eval()
    archive = OUT / "v2_policy_stage1.npz"
    export_v2_numpy(model, archive)
    runtime = V2NumpyPolicy.load(archive)
    rows = [
        _bench(runtime, 0, 30),
        _bench(runtime, 8, 30),
        _bench(runtime, 16, 30),
        _bench(runtime, 32, 30),
        _bench(runtime, 128, 10),
    ]
    representative = [row for row in rows if row["hands"] <= 32]
    manifest = {
        "format_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "parameter_count": runtime.parameter_count,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": _sha256(archive),
        "model_parameter_sha256": runtime.model_parameter_sha256,
        "feature_schema_sha256": feature_schema_sha256(),
        "benchmarks": rows,
        "latency_gate": {
            "representative_p99_under_200ms": all(row["p99_ms"] < 200.0 for row in representative),
            "representative_worst_under_500ms": all(row["worst_ms"] < 500.0 for row in representative),
            "representative_max_p99_ms": max(row["p99_ms"] for row in representative),
            "representative_max_worst_ms": max(row["worst_ms"] for row in representative),
        },
        "stress": {"hands": 128, "completed": True},
        "source_sha256": {
            "v2_export.py": _sha256(ROOT / "src/kaggrl/v2_export.py"),
            "v2_numpy_runtime.py": _sha256(ROOT / "src/kaggrl/v2_numpy_runtime.py"),
            "v2_model.py": _sha256(ROOT / "src/kaggrl/v2_model.py"),
            "v2_tensorize.py": _sha256(ROOT / "src/kaggrl/v2_tensorize.py"),
        },
    }
    path = OUT / "build_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


# Legacy benchmark helpers above are retained for historical comparison only.
# The canonical entrypoint is the final main() below.


def _row(hands: int) -> dict:
    state = _structured(hands)
    action = {
        "farmer": {"op": "PASS", "item": None, "quantity": None, "raw": ["PASS"]},
        "hands": [
            {"op": "PASS", "item": None, "quantity": None, "raw": ["PASS"]}
            for _ in range(hands)
        ],
        "market": [{"kind": "STOP_QUEUE", "op": None, "item": None,
                    "quantity": None, "raw": []}],
    }
    return {
        "state": state, "previous_action": {}, "previous_effect": {},
        "canonical_action": action, "effects": {}, "terminal_result": 0,
        "final_margin": 0,
    }


def _canonical_from_torch(row) -> dict:
    return {
        "farmer": row.farmer.chosen_action,
        "hands": [decision.chosen_action for decision in row.hands],
        "market": [decision.chosen_action for decision in row.market],
    }


def _semantic_signature(action: dict) -> tuple:
    farmer = action["farmer"]
    hands = action["hands"]
    market = action["market"]
    def unit(x): return (x.get("op"), x.get("item"), x.get("quantity"))
    def slot(x): return (x.get("kind"), x.get("op"), x.get("item"), x.get("quantity"))
    return unit(farmer), tuple(map(unit, hands)), tuple(map(slot, market))


def _parity(model, runtime, hands_values=(0, 17, 32)) -> dict:
    rows = []
    max_fused = max_h = max_c = max_intent = 0.0
    max_money = max_margin = 0.0
    action_mismatches = 0
    for hands in hands_values:
        batch = collate_transitions([_row(hands)])
        with torch.no_grad():
            encoded = model.encoder(batch)
            h, c, intent = model.core.step(encoded.fused, None)
            policy = model.sample_step(
                batch, None, np.random.default_rng(1000 + hands), deterministic=True
            )
        debug = runtime.debug_encode(
            batch.structured_states[0], {}, batch.previous_actions[0], None
        )
        step = runtime.step(
            batch.structured_states[0], {}, batch.previous_actions[0], None,
            np.random.default_rng(1000 + hands), deterministic=True,
        )
        fused_diff = float(np.max(np.abs(debug["fused"] - encoded.fused[0].numpy())))
        h_diff = float(np.max(np.abs(debug["h"] - h[0].numpy())))
        c_diff = float(np.max(np.abs(debug["c"] - c[0].numpy())))
        intent_diff = float(np.max(np.abs(debug["intent"] - intent[0].numpy())))
        money_diff = abs(step.terminal_money - float(policy.aux.terminal_money[0]))
        margin_diff = abs(step.terminal_margin - float(policy.aux.terminal_margin[0]))
        action_equal = _semantic_signature(step.canonical_action) == _semantic_signature(
            _canonical_from_torch(policy.rows[0])
        )
        action_mismatches += int(not action_equal)
        max_fused = max(max_fused, fused_diff); max_h = max(max_h, h_diff)
        max_c = max(max_c, c_diff); max_intent = max(max_intent, intent_diff)
        max_money = max(max_money, money_diff); max_margin = max(max_margin, margin_diff)
        rows.append({"hands": hands, "action_equal": action_equal,
                     "fused_max_abs": fused_diff, "h_max_abs": h_diff,
                     "c_max_abs": c_diff, "intent_max_abs": intent_diff,
                     "terminal_money_abs": money_diff, "terminal_margin_abs": margin_diff})
    return {
        "rows": rows, "action_mismatches": action_mismatches,
        "max_fused_abs": max_fused, "max_h_abs": max_h,
        "max_c_abs": max_c, "max_intent_abs": max_intent,
        "max_terminal_money_abs": max_money,
        "max_terminal_margin_abs": max_margin,
        "tolerance": 3e-5,
        "passed": (
            action_mismatches == 0
            and max(max_fused, max_h, max_c, max_intent, max_money, max_margin) <= 3e-5
        ),
    }


def _latency(runtime, hands: int, runs: int) -> dict:
    state = _structured(hands)
    for _ in range(2):
        runtime.step(state, {}, {}, None, np.random.default_rng(0), deterministic=True)
    samples = []
    for index in range(runs):
        start = time.perf_counter_ns()
        runtime.step(
            state, {}, {}, None, np.random.default_rng(100 + index), deterministic=True
        )
        samples.append((time.perf_counter_ns() - start) / 1e6)
    values = np.asarray(samples, dtype=np.float64)
    return {
        "hands": hands, "runs": runs,
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "worst_ms": float(values.max()),
    }


def _torch_free_import() -> dict:
    code = (
        "import sys; from kaggrl.v2_numpy_runtime import V2NumpyPolicy; "
        "assert not any(k == 'torch' or k.startswith('torch.') for k in sys.modules); "
        "print('torch-free-ok')"
    )
    env = os.environ.copy(); env["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    return {"passed": result.returncode == 0 and "torch-free-ok" in result.stdout,
            "returncode": result.returncode, "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip()}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(1729)
    model = RecurrentIntentPolicy().eval()
    archive = OUT / "policy_init.npz"
    export_v2_numpy(model, archive)
    runtime = V2NumpyPolicy.load(archive)
    parity = _parity(model, runtime)
    parity.update({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "archive_sha256": _sha256(archive),
        "feature_schema_sha256": feature_schema_sha256(),
    })
    parity_path = OUT / "parity_report.json"
    parity_path.write_text(json.dumps(parity, indent=2, sort_keys=True) + "\n")

    latency = []
    for hands in (0, 8, 16, 32):
        latency.append(_latency(runtime, hands, runs=20))
    stress = [_latency(runtime, 64, runs=8), _latency(runtime, 128, runs=4)]
    representative_p99 = max(row["p99_ms"] for row in latency)
    import_report = _torch_free_import()
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "platform": platform.platform(),
        "parameter_count": runtime.parameter_count,
        "archive": str(archive.relative_to(ROOT)),
        "archive_size_bytes": archive.stat().st_size,
        "archive_sha256": _sha256(archive),
        "model_parameter_sha256": runtime.model_parameter_sha256,
        "exported_parameter_sha256": runtime.exported_parameter_sha256,
        "feature_schema_sha256": feature_schema_sha256(),
        "stage0_marker_sha256": _sha256(STAGE0_MARKER),
        "latency": latency, "stress_latency": stress,
        "representative_p99_ms": representative_p99,
        "latency_target_ms": 200.0,
        "latency_target_passed": representative_p99 < 200.0,
        "torch_free_import": import_report,
        "parity_report_sha256": _sha256(parity_path),
        "parity_passed": bool(parity["passed"]),
        "source_sha256": {
            "v2_tensorize.py": _sha256(ROOT / "src/kaggrl/v2_tensorize.py"),
            "v2_encoder.py": _sha256(ROOT / "src/kaggrl/v2_encoder.py"),
            "v2_ledger.py": _sha256(ROOT / "src/kaggrl/v2_ledger.py"),
            "v2_quantity.py": _sha256(ROOT / "src/kaggrl/v2_quantity.py"),
            "v2_model.py": _sha256(ROOT / "src/kaggrl/v2_model.py"),
            "v2_export.py": _sha256(ROOT / "src/kaggrl/v2_export.py"),
            "v2_numpy_runtime.py": _sha256(ROOT / "src/kaggrl/v2_numpy_runtime.py"),
            "benchmark_v2_stage1.py": _sha256(Path(__file__)),
        },
    }
    manifest_path = OUT / "build_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "parity_passed": parity["passed"],
        "representative_p99_ms": representative_p99,
        "latency_target_passed": manifest["latency_target_passed"],
        "torch_free": import_report["passed"],
        "archive_size_bytes": manifest["archive_size_bytes"],
        "parameter_count": manifest["parameter_count"],
        "manifest": str(manifest_path),
    }, indent=2, sort_keys=True))
    if not parity["passed"]:
        raise SystemExit("Torch/NumPy parity gate failed")
    if not import_report["passed"]:
        raise SystemExit("Torch-free import gate failed")


if __name__ == "__main__":
    main()
