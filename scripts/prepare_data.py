from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = "ponschannel/farmos-v32-training-data"
REQUIRED = (
    ROOT / "data/top_tier/live_v2/transitions.parquet",
    ROOT / "data/top_tier/live_v2/effective_actions.parquet",
    ROOT / "data/top_tier/live_v2/manifests/STAGE0_ACCEPTED.json",
    ROOT / "checkpoints/rl_v2_stage1/STAGE1_ACCEPTED.json",
    ROOT / "assets/v32_smoke_init.pt",
)


def _missing() -> list[Path]:
    return [path for path in REQUIRED if not path.is_file()]


def _kaggle_command() -> list[str]:
    executable = shutil.which("kaggle")
    if executable:
        return [executable]
    return [sys.executable, "-m", "kaggle"]
def _download(dataset: str, target: Path) -> None:
    command = _kaggle_command() + [
        "datasets", "download", "-d", dataset,
        "-p", str(target), "--force", "--unzip",
    ]
    print("FARMOS_DATA_DOWNLOAD=" + dataset, flush=True)
    subprocess.run(command, check=True)


def _install_download(download_dir: Path) -> None:
    direct_roots = ("data", "checkpoints", "assets")
    if all((download_dir / name).exists() for name in direct_roots):
        for name in direct_roots:
            shutil.copytree(
                download_dir / name,
                ROOT / name,
                dirs_exist_ok=True,
            )
        return
    archive = download_dir / "farmos_training_data.tar.gz"
    if not archive.is_file():
        candidates = list(download_dir.glob("*.tar.gz"))
        if len(candidates) != 1:
            raise RuntimeError("Kaggle training bundle has an unknown layout")
        archive = candidates[0]
    print("FARMOS_DATA_EXTRACT=" + str(archive), flush=True)
    with tarfile.open(archive, "r:gz") as handle:
        handle.extractall(ROOT, filter="data")


def prepare(dataset: str) -> None:
    missing = _missing()
    if not missing:
        print("FARMOS_DATA_READY=1", flush=True)
        return
    if not (
        Path.home().joinpath(".kaggle/kaggle.json").is_file()
        or (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))
    ):
        raise RuntimeError(
            "Kaggle credentials missing. Upload kaggle.json to ~/.kaggle/kaggle.json."
        )
    with tempfile.TemporaryDirectory(prefix="farmos-data-") as temp:
        tempdir = Path(temp)
        _download(dataset, tempdir)
        _install_download(tempdir)
    missing = _missing()
    if missing:
        names = ", ".join(str(path.relative_to(ROOT)) for path in missing)
        raise RuntimeError("training data bundle is incomplete: " + names)
    print("FARMOS_DATA_READY=1", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=os.getenv("FARMOS_KAGGLE_DATASET", DEFAULT_DATASET),
    )
    args = parser.parse_args()
    prepare(str(args.dataset))


if __name__ == "__main__":
    main()
