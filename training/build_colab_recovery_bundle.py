from __future__ import annotations

import gzip
import hashlib
from pathlib import Path
import tarfile
from typing import Iterable


_EXCLUDED_PARTS = {".git", ".env", "__pycache__", ".pytest_cache", ".mypy_cache"}
_EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _allowed(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    if any(part in _EXCLUDED_PARTS for part in rel.parts):
        return False
    return path.suffix.lower() not in _EXCLUDED_SUFFIXES


def _expand(root: Path, include_paths: Iterable[str | Path]) -> list[Path]:
    files: list[Path] = []
    for raw in include_paths:
        path = root / Path(raw)
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_file():
            if _allowed(path, root):
                files.append(path)
            continue
        files.extend(item for item in path.rglob("*") if item.is_file() and _allowed(item, root))
    return sorted(set(files), key=lambda item: item.relative_to(root).as_posix())


def build_recovery_bundle(root: Path, output_path: Path,
                          include_paths: Iterable[str | Path]) -> Path:
    root = Path(root).resolve()
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    files = _expand(root, include_paths)
    if not files:
        raise ValueError("recovery bundle would be empty")
    with output.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w") as archive:
                for path in files:
                    rel = path.relative_to(root).as_posix()
                    info = archive.gettarinfo(str(path), arcname=rel)
                    info.mtime = 0
                    info.uid = 0; info.gid = 0
                    info.uname = ""; info.gname = ""
                    with path.open("rb") as handle:
                        archive.addfile(info, handle)
    digest = _sha256(output)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{digest}  {output.name}\n", encoding="utf-8",
    )
    return output
