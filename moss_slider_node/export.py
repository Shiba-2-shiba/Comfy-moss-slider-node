from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import uuid

from safetensors import safe_open
from safetensors.torch import save_file
import torch
from torch import Tensor, nn

from .lora import LoRALinear, lora_state_dict
from .checkpoint import file_sha256


def _safe_prefix(prefix: str) -> str:
    if not isinstance(prefix, str) or not prefix.strip() or prefix != prefix.strip():
        raise ValueError("prefix must be a non-empty filename stem")
    if prefix in {".", ".."} or Path(prefix).name != prefix or "/" in prefix or "\\" in prefix:
        raise ValueError("prefix must not contain a directory or traversal component")
    return prefix


def _rank_alpha(model: nn.Module) -> tuple[int, float]:
    values = {(module.rank, module.alpha) for module in model.modules() if isinstance(module, LoRALinear)}
    if not values:
        raise ValueError("model has no LoRA adapters")
    if len(values) != 1:
        raise ValueError("all LoRA adapters must use the same rank and alpha")
    return next(iter(values))


def _reserve_name(output_dir: Path, prefix: str) -> tuple[Path, Path, Path]:
    for index in range(10000):
        suffix = "" if index == 0 else f"_{index:03d}"
        lora_path = output_dir / f"{prefix}{suffix}.safetensors"
        json_path = output_dir / f"{prefix}{suffix}.json"
        lock_path = output_dir / f".{prefix}{suffix}.saving.lock"
        if lora_path.exists() or json_path.exists():
            continue
        try:
            with lock_path.open("x"):
                pass
        except FileExistsError:
            continue
        if lora_path.exists() or json_path.exists():
            lock_path.unlink(missing_ok=True)
            continue
        return lora_path, json_path, lock_path
    raise FileExistsError("could not allocate a unique LoRA output name")


def save_lora(
    model: nn.Module,
    output_dir: str | os.PathLike[str],
    prefix: str,
    manifest: Mapping[str, object],
) -> tuple[Path, Path]:
    prefix = _safe_prefix(prefix)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    rank, alpha = _rank_alpha(model)
    state = lora_state_dict(model)
    for key, value in state.items():
        if not torch.isfinite(value).all():
            raise ValueError(f"LoRA tensor must be finite: {key}")

    lora_path, json_path, lock_path = _reserve_name(directory, prefix)
    token = uuid.uuid4().hex
    lora_tmp = directory / f".{lora_path.name}.{token}.tmp"
    json_tmp = directory / f".{json_path.name}.{token}.tmp"
    committed: list[Path] = []
    try:
        metadata = {
            "architecture": "moss_image",
            "rank": str(rank),
            "alpha": str(float(alpha)),
            "ss_network_dim": str(rank),
            "ss_network_alpha": str(float(alpha)),
            "ss_base_model_version": "moss_image",
        }
        save_file(state, lora_tmp, metadata=metadata)
        manifest_data = dict(manifest)
        manifest_data.update(
            {
                "architecture": "moss_image",
                "rank": rank,
                "alpha": alpha,
                "lora_file": lora_path.name,
                "lora_sha256": file_sha256(lora_tmp),
            }
        )
        json_text = json.dumps(manifest_data, ensure_ascii=False, indent=2, sort_keys=True)
        json_tmp.write_text(json_text + "\n", encoding="utf-8")
        # The safetensors rename is the commit point. A crash before it can
        # leave an orphan JSON, but never a final-named LoRA without its manifest.
        os.replace(json_tmp, json_path)
        committed.append(json_path)
        os.replace(lora_tmp, lora_path)
        committed.append(lora_path)
        lock_path.unlink(missing_ok=True)
        return lora_path, json_path
    except BaseException:
        for path in (lora_tmp, json_tmp, lora_path, json_path, lock_path, *committed):
            path.unlink(missing_ok=True)
        raise


def read_lora(path: str | os.PathLike[str]) -> tuple[dict[str, Tensor], dict[str, str]]:
    tensors: dict[str, Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        for key in handle.keys():
            tensors[key] = handle.get_tensor(key)
    return tensors, metadata
