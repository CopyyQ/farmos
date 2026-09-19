from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.prepare_data import prepare
from training.train_v3_bc import BCV3Config, run_v3_bc


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "teacher_mix_schedule" in payload:
        payload["teacher_mix_schedule"] = tuple(payload["teacher_mix_schedule"])
    return payload
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/t4_seq32.json")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--skip-prepare", action="store_true")
    args = parser.parse_args()

    config_path = (ROOT / args.config).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not args.skip_prepare:
        prepare(args.dataset or "ponschannel/farmos-v32-training-data")

    dataset = ROOT / "data/top_tier/live_v2/transitions.parquet"
    stage0 = ROOT / "data/top_tier/live_v2/manifests/STAGE0_ACCEPTED.json"
    stage1 = ROOT / "checkpoints/rl_v2_stage1/STAGE1_ACCEPTED.json"
    init_checkpoint = ROOT / "assets/v32_smoke_init.pt"
    run_name = args.run_name or (
        config_path.stem + "_" + time.strftime("%Y%m%d_%H%M%S")
    )
    output = ROOT / "runs" / run_name

    values = load_config(config_path)
    config = BCV3Config(
        dataset_path=dataset,
        stage0_marker=stage0,
        stage1_marker=stage1,
        output_dir=output,
        **values,
    )
    print("FARMOS_TORCH=" + torch.__version__, flush=True)
    print("FARMOS_CUDA=" + str(torch.cuda.is_available()), flush=True)
    if torch.cuda.is_available():
        print("FARMOS_GPU=" + torch.cuda.get_device_name(0), flush=True)
    print("FARMOS_CONFIG=" + json.dumps(values, sort_keys=True), flush=True)
    print("FARMOS_RUN_DIR=" + str(output), flush=True)

    started = time.perf_counter()
    best = run_v3_bc(config, init_checkpoint)
    elapsed = time.perf_counter() - started
    result = {
        "status": "ok",
        "run_name": run_name,
        "elapsed_seconds": elapsed,
        "best_checkpoint": str(best),
        "best_sha256": sha256(best),
        "last_checkpoint": str(output / "bc_last.pt"),
        "history": str(output / "history.jsonl"),
    }
    result_path = output / "run_result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("FARMOS_RESULT=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
