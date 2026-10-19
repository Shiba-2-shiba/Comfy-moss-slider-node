"""Repack shared-QKV sliders for ComfyUI's non-merging bypass loader."""

import json
import math
import re
from functools import lru_cache
from pathlib import Path

import torch

from .checkpoint import file_sha256, read_header
from .export import read_lora


_KEY = re.compile(r"lora_unet_layers_(0|[1-9]\d*)_proj_(q|k|v|o|ffn_up_gate|ffn_down)\.(lora_down\.weight|lora_up\.weight|alpha)")
_SUFFIXES = {"lora_down.weight", "lora_up.weight", "alpha"}


def pack_logical_lora(state):
    groups: dict[int, dict[str, dict[str, torch.Tensor]]] = {}
    for key, tensor in state.items():
        match = _KEY.fullmatch(key)
        if not match or not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point() or not torch.isfinite(tensor).all():
            raise ValueError(f"Invalid MOSS slider tensor: {key}")
        block, projection, suffix = match.groups()
        groups.setdefault(int(block), {}).setdefault(projection, {})[suffix] = tensor
    if not groups or set(groups) != set(range(max(groups) + 1)):
        raise ValueError("MOSS slider must contain contiguous block targets")
    packed = {}
    for block, projections in groups.items():
        if set(projections) != {"q", "k", "v", "o", "ffn_up_gate", "ffn_down"}:
            raise ValueError(f"Incomplete MOSS slider block: {block}")
        for projection, values in projections.items():
            if set(values) != _SUFFIXES:
                raise ValueError(f"Incomplete MOSS slider target: {block}/{projection}")
            down, up, alpha = (values[s] for s in ("lora_down.weight", "lora_up.weight", "alpha"))
            if (down.ndim != 2 or up.ndim != 2 or down.shape[0] < 1 or up.shape[1] != down.shape[0]
                    or alpha.numel() != 1 or float(alpha) <= 0):
                raise ValueError(f"Invalid MOSS LoRA rank/shape/alpha: {block}/{projection}")
        q, k, v = (projections[name] for name in ("q", "k", "v"))
        if any(not torch.equal(q["lora_down.weight"], part["lora_down.weight"])
               or not torch.equal(q["alpha"], part["alpha"]) for part in (k, v)):
            raise ValueError("Packed QKV requires a shared down matrix and common alpha")
        if any(part["lora_up.weight"].shape != q["lora_up.weight"].shape for part in (k, v)):
            raise ValueError("QKV output slices must have the same shape")
        targets = {"qkv": {"lora_down.weight": q["lora_down.weight"], "alpha": q["alpha"],
                           "lora_up.weight": torch.cat([part["lora_up.weight"] for part in (q, k, v)])},
                   **{name: projections[name] for name in ("o", "ffn_up_gate", "ffn_down")}}
        for name, values in targets.items():
            for suffix, tensor in values.items():
                packed[f"lora_unet_layers_{block}_proj_{name}.{suffix}"] = tensor.contiguous()
    return packed


@lru_cache(maxsize=8)
def _cached_digest(path, size, mtime_ns, header_sha256):
    return file_sha256(path)


def apply_slider(model, lora_path, strength):
    import comfy.lora
    import comfy.sd
    from comfy.ldm.moss_image.model import MOSSImage

    if not math.isfinite(strength):
        raise ValueError("Slider strength must be finite")
    native = model.model.diffusion_model
    if not isinstance(native, MOSSImage) or any(not block.fused_qkv for block in native.layers):
        raise ValueError("MossSliderApply requires a native packed MOSS MODEL")
    if model.patches or model.injections:
        raise ValueError("Apply the slider to an unpatched MOSS base MODEL")
    factory = model.cached_patcher_init
    if not factory or factory[0] is not comfy.sd.load_diffusion_model:
        raise ValueError("Cannot verify base checkpoint identity; use ComfyUI Load Diffusion Model")
    base_path = Path(factory[1][0])
    _, identity = read_header(base_path)
    digest = _cached_digest(str(base_path.resolve()), identity["size_bytes"], identity["mtime_ns"], identity["header_sha256"])
    lora_path = Path(lora_path)
    manifest_path = lora_path.with_suffix(".json")
    if not manifest_path.is_file():
        raise ValueError("MOSS slider manifest JSON is required to verify checkpoint identity")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("source"), dict):
        raise ValueError("Invalid MOSS slider manifest")
    if manifest.get("architecture") != "moss_image" or manifest["source"].get("sha256") != digest:
        raise ValueError("MOSS slider base checkpoint identity does not match the MODEL")
    if manifest.get("lora_file") != lora_path.name or manifest.get("lora_sha256") != file_sha256(lora_path):
        raise ValueError("MOSS slider artifact/manifest pair digest mismatch")
    state, metadata = read_lora(lora_path)
    if metadata.get("architecture") != "moss_image":
        raise ValueError("Not a MOSS slider artifact")
    try:
        rank, alpha = int(metadata["rank"]), float(metadata["alpha"])
        if type(manifest.get("rank")) is not int or rank != manifest["rank"] or alpha != manifest["alpha"]:
            raise ValueError("Manifest rank/alpha differs from LoRA metadata")
        if rank < 1 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("Invalid slider metadata rank/alpha")
    except (KeyError, TypeError) as error:
        raise ValueError("Missing slider metadata rank/alpha") from error
    packed = pack_logical_lora(state)
    if len(packed) != len(native.layers) * 12:
        raise ValueError("MOSS slider target count does not match the MODEL")
    for index, block in enumerate(native.layers):
        for name, module in block.proj.items():
            prefix = f"lora_unet_layers_{index}_proj_{name}"
            down, up = packed[prefix + ".lora_down.weight"], packed[prefix + ".lora_up.weight"]
            target_alpha = float(packed[prefix + ".alpha"])
            if (down.shape != (rank, module.in_features) or up.shape != (module.out_features, rank)
                    or target_alpha != float(torch.tensor(alpha, dtype=torch.float32))):
                raise ValueError(f"MOSS slider shape mismatch: {prefix}")
    mapping = comfy.lora.model_lora_keys_unet(model.model)
    loaded = comfy.lora.load_lora(packed, mapping)
    expected = {f"diffusion_model.layers.{i}.proj.{name}.weight"
                for i, block in enumerate(native.layers) for name in block.proj}
    if set(loaded) != expected or any(not isinstance(key, str) for key in loaded):
        raise ValueError("ComfyUI did not consume every packed slider target")
    if strength == 0:
        return model.clone()
    result, _ = comfy.sd.load_bypass_lora_for_models(model, None, packed, strength, 0)
    if not result.get_injections("bypass_lora"):
        raise RuntimeError("ComfyUI failed to register the MOSS slider bypass")
    return result
