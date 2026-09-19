from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from kaggrl.v3_model import TemporalIntentPolicy

ARCHITECTURE_VERSION = "rl_v3_temporal_attention"
FORMAT_VERSION = 3

_ALLOWED_PREFIXES = (
    "encoder.",
    "op_embedding.", "item_embedding.", "action_proj.",
    "market_slot_embedding.", "decoder_init.", "ledger_proj.",
    "decoder_cell.", "unit_op_head.", "market_op_head.",
    "item_head.", "quantity_context.", "quantity_decoder.",
    "effect_head.", "future_resource_head.", "unit_task_head.",
    "opponent_effect_head.", "terminal_money_head.", "terminal_margin_head.",
)
_ALLOWED_EXACT = {
    "start_action", "mask_op_embedding", "mask_item_embedding",
    "mask_quantity_features",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _state_sha256(model: TemporalIntentPolicy) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _allowed(name: str) -> bool:
    return name in _ALLOWED_EXACT or name.startswith(_ALLOWED_PREFIXES)


def initialize_v3(
    mode: str,
    *,
    seed: int,
    v2_checkpoint: Path | None = None,
) -> tuple[TemporalIntentPolicy, dict[str, Any]]:
    if mode not in {"clean", "partial_v2_warm_start"}:
        raise ValueError("unknown v3 initialization mode")
    if mode == "partial_v2_warm_start" and v2_checkpoint is None:
        raise ValueError("partial warm start requires v2_checkpoint")
    if mode == "clean" and v2_checkpoint is not None:
        raise ValueError("clean initialization must not receive v2_checkpoint")

    torch.manual_seed(int(seed))
    model = TemporalIntentPolicy()
    loaded: list[str] = []
    skipped: list[dict[str, str]] = []
    source_sha = None
    source_dataset_sha = None
    if mode == "partial_v2_warm_start":
        source_path = Path(v2_checkpoint)
        source_sha = _sha256(source_path)
        payload = torch.load(source_path, map_location="cpu", weights_only=False)
        source = payload.get("model_state")
        if not isinstance(source, dict):
            raise RuntimeError("v2 checkpoint has no model_state")
        source_dataset_sha = payload.get("dataset_sha256")
        target = model.state_dict()
        updated = dict(target)
        for name, tensor in source.items():
            if name.startswith("core."):
                skipped.append({"name": name, "reason": "temporal_parameter_excluded"})
                continue
            if not _allowed(name):
                skipped.append({"name": name, "reason": "not_allowlisted"})
                continue
            if name not in target:
                skipped.append({"name": name, "reason": "missing_target"})
                continue
            if tuple(tensor.shape) != tuple(target[name].shape):
                raise RuntimeError(
                    f"warm-start shape mismatch for {name}: "
                    f"{tuple(tensor.shape)} != {tuple(target[name].shape)}"
                )
            updated[name] = tensor.detach().clone()
            loaded.append(name)
        model.load_state_dict(updated, strict=True)

    manifest = {
        "architecture_version": ARCHITECTURE_VERSION,
        "initialization_mode": mode,
        "mode": mode,
        "seed": int(seed),
        "source_checkpoint_sha256": source_sha,
        "v2_checkpoint_sha256": source_sha,
        "source_dataset_sha256": source_dataset_sha,
        "loaded_parameters": sorted(loaded),
        "skipped_parameters": sorted(skipped, key=lambda row: row["name"]),
        "v3_model_state_sha256": _state_sha256(model),
        "result_model_sha256": _state_sha256(model),
        "temporal_config": {
            "hidden_dim": 256, "attention_dim": 128,
            "heads": 4, "window": 32, "blocks": 1, "dropout": 0.0,
        },
    }
    return model, manifest

def write_v3_initialization(
    mode: str,
    output_path: Path,
    *,
    seed: int,
    v2_checkpoint: Path | None = None,
) -> tuple[Path, Path]:
    output = Path(output_path)
    manifest_path = output.with_suffix(".migration.json")
    if output.exists() or manifest_path.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model, manifest = initialize_v3(
        mode, seed=seed, v2_checkpoint=v2_checkpoint,
    )
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    manifest_path.write_text(manifest_text, encoding="utf-8")
    manifest_sha = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
    checkpoint = {
        "format_version": FORMAT_VERSION,
        "architecture_version": ARCHITECTURE_VERSION,
        "initialization_mode": mode,
        "seed": int(seed),
        "dataset_sha256": manifest.get("source_dataset_sha256"),
        "model_state": model.state_dict(),
        "model_state_sha256": manifest["v3_model_state_sha256"],
        "migration_manifest_sha256": manifest_sha,
        "source_checkpoint_sha256": manifest.get("source_checkpoint_sha256"),
    }
    torch.save(checkpoint, output)
    return output, manifest_path


def write_v3_initialization_artifact(
    model: TemporalIntentPolicy,
    manifest: dict[str, Any],
    output_path: Path,
) -> tuple[Path, Path]:
    output = Path(output_path)
    sidecar = Path(str(output) + ".migration.json")
    if output.exists() or sidecar.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result_sha = _state_sha256(model)
    if result_sha != manifest.get("result_model_sha256"):
        raise RuntimeError("migration manifest/model SHA mismatch")
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    sidecar.write_text(manifest_text, encoding="utf-8")
    checkpoint = {
        "format_version": FORMAT_VERSION,
        "architecture_version": ARCHITECTURE_VERSION,
        "initialization_mode": manifest.get("mode"),
        "model_state": model.state_dict(),
        "model_state_sha256": result_sha,
        "migration_manifest": dict(manifest),
        "migration_manifest_sha256": hashlib.sha256(
            manifest_text.encode("utf-8")
        ).hexdigest(),
    }
    torch.save(checkpoint, output)
    return output, sidecar
