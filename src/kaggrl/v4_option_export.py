from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .v4_option_model import V4OptionPolicy

FORMAT_VERSION = 1


def _parameter_sha256(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(k for k in arrays if k.startswith("p__")):
        value = np.ascontiguousarray(arrays[key])
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def export_v4_option_numpy(
    model: V4OptionPolicy,
    path: str | Path,
    *,
    route_ids: list[int] | tuple[int, ...],
    market_modes: list[str] | tuple[str, ...],
    route_gate_threshold: float = 0.50,
) -> Path:
    if getattr(model, "ARCHITECTURE_VERSION", None) != (
        V4OptionPolicy.ARCHITECTURE_VERSION
    ):
        raise ValueError("wrong V4 option model architecture")
    route_ids = tuple(int(value) for value in route_ids)
    market_modes = tuple(str(value) for value in market_modes)
    if len(route_ids) != model.route_count:
        raise ValueError("route ID count does not match route head")
    if len(market_modes) != model.market_mode_count:
        raise ValueError("market mode count does not match market head")

    arrays: dict[str, np.ndarray] = {
        "format_version": np.asarray(FORMAT_VERSION, dtype=np.int32),
        "architecture_version": np.asarray(
            model.ARCHITECTURE_VERSION
        ),
        "input_dim": np.asarray(model.input_dim, dtype=np.int32),
        "hidden_dim": np.asarray(model.hidden_dim, dtype=np.int32),
        "clock_dim": np.asarray(model.clock_dim, dtype=np.int32),
        "route_ids": np.asarray(route_ids, dtype=np.int32),
        "route_gate_threshold": np.asarray(
            float(route_gate_threshold), dtype=np.float32
        ),
        "market_modes_json": np.asarray(
            json.dumps(list(market_modes))
        ),
    }
    for name, tensor in model.state_dict().items():
        arrays["p__" + name.replace(".", "__")] = (
            tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        )
    arrays["parameter_sha256"] = np.asarray(
        _parameter_sha256(arrays)
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path
