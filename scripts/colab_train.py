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
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--recovery-dataset", default=None)
    parser.add_argument("--dagger-teacher", default=None)
    parser.add_argument("--skip-prepare", action="store_true")
    args = parser.parse_args()

    config_path = (ROOT / args.config).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    run_name = args.run_name or (
        config_path.stem + "_" + time.strftime("%Y%m%d_%H%M%S")
    )
    output = ROOT / "runs" / run_name
    blocking_outputs = (
        output / "bc_best.pt",
        output / "bc_last.pt",
        output / "history.jsonl",
        output / "strategy_manifest.json",
    )
    if any(path.exists() for path in blocking_outputs):
        raise FileExistsError(
            f"run already exists: {output}. "
            "Choose a new --run-name; existing artifacts are preserved."
        )

    dataset = ROOT / "data/top_tier/live_v2/transitions.parquet"
    stage0 = ROOT / "data/top_tier/live_v2/manifests/STAGE0_ACCEPTED.json"
    stage1 = ROOT / "checkpoints/rl_v2_stage1/STAGE1_ACCEPTED.json"
    required_training_artifacts = (dataset, stage0, stage1)
    missing_training_artifacts = [
        path for path in required_training_artifacts if not path.is_file()
    ]
    if not args.skip_prepare or missing_training_artifacts:
        if args.skip_prepare and missing_training_artifacts:
            print(
                "FARMOS_PREPARE_FALLBACK="
                + json.dumps(
                    [str(path) for path in missing_training_artifacts],
                    sort_keys=True,
                ),
                flush=True,
            )
        prepare(args.dataset or "ponschannel/farmos-v32-training-data")
    values = load_config(config_path)
    scratch_init = str(values.get("init_mode", "checkpoint")) == "scratch"
    if scratch_init:
        if args.init_checkpoint:
            raise ValueError(
                "scratch config must not receive --init-checkpoint"
            )
        init_checkpoint = None
    else:
        init_checkpoint = (
            Path(args.init_checkpoint).resolve()
            if args.init_checkpoint
            else ROOT / "assets/v32_smoke_init.pt"
        )
        if not init_checkpoint.is_file():
            raise FileNotFoundError(init_checkpoint)
    if args.recovery_dataset:
        values["recovery_dataset_path"] = str(
            Path(args.recovery_dataset).resolve()
        )
    elif values.get("recovery_dataset_path"):
        recovery = Path(values["recovery_dataset_path"])
        if not recovery.is_absolute():
            recovery = (ROOT / recovery).resolve()
        values["recovery_dataset_path"] = str(recovery)
    if args.dagger_teacher:
        values["dagger_teacher_path"] = str(
            Path(args.dagger_teacher).resolve()
        )
    elif values.get("dagger_teacher_path"):
        teacher = Path(values["dagger_teacher_path"])
        if not teacher.is_absolute():
            teacher = (ROOT / teacher).resolve()
        values["dagger_teacher_path"] = str(teacher)
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
