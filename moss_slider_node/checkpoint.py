"""Strict metadata inspection and tensor-at-a-time loading of packed MOSS."""

import hashlib
import json
import math
import re
import struct
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn


_DTYPE_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
                "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8,
                "F8_E4M3": 1, "F8_E5M2": 1}
_TARGETS = ("qkv", "o", "ffn_up_gate", "ffn_down")


@dataclass(frozen=True)
class MossConfig:
    dim: int = 3072
    ffn_dim: int = 8192
    num_heads: int = 24
    num_layers: int = 42
    text_dim: int = 4096
    latent_channels: int = 128

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in asdict(self).values()):
            raise ValueError("MOSS dimensions must be positive integers")
        if self.dim % self.num_heads or (self.dim // self.num_heads) % 4 or self.dim // self.num_heads < 16:
            raise ValueError("MOSS head dimension must be divisible by four and at least 16")

    def linear_shapes(self):
        return {"qkv": (3 * self.dim, self.dim), "o": (self.dim, self.dim),
                "ffn_up_gate": (2 * self.ffn_dim, self.dim), "ffn_down": (self.dim, self.ffn_dim)}


def expected_parameter_shapes(config):
    """Metadata-only schema, checked against native MOSS by the integration suite."""
    dim, ffn = config.dim, config.ffn_dim
    shapes = {
        "video_embedder.weight": (dim, config.latent_channels), "video_embedder.bias": (dim,),
        "text_embedder.proj.weight": (dim, config.text_dim), "text_embedder.proj.bias": (dim,),
        "time_embedding.0.weight": (dim, 256), "time_embedding.0.bias": (dim,),
        "time_embedding.2.weight": (dim, dim), "time_embedding.2.bias": (dim,),
        "shared_block_modulation.weight": (6 * dim, dim), "shared_block_modulation.bias": (6 * dim,),
        "video_head_modulation.weight": (2 * dim, dim), "video_head_modulation.bias": (2 * dim,),
        "final_norm_video.weight": (dim,), "final_linear_video.weight": (config.latent_channels, dim),
    }
    head_dim = dim // config.num_heads
    for index in range(2):
        prefix = f"text_embedder.blocks.{index}"
        for norm in ("norm1", "norm2", "q_norm", "k_norm"):
            shapes[f"{prefix}.{norm}.weight"] = (head_dim if norm in {"q_norm", "k_norm"} else dim,)
        for name, out_features, in_features in [("qkv", 3 * dim, dim), ("proj", dim, dim),
                                               ("ffn_gate", ffn, dim), ("ffn_up", ffn, dim),
                                               ("ffn_down", dim, ffn)]:
            shapes[f"{prefix}.{name}.weight"] = (out_features, in_features)
            shapes[f"{prefix}.{name}.bias"] = (out_features,)
    for index in range(config.num_layers):
        for norm in ("norm1", "norm2", "q_norm", "k_norm"):
            shapes[f"layers.{index}.{norm}.weight"] = (head_dim if norm in {"q_norm", "k_norm"} else dim,)
        for name, shape in config.linear_shapes().items():
            shapes[f"layers.{index}.proj.{name}.weight"] = shape
    return shapes


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_header(path):
    path = Path(path)
    stat = path.stat()
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("Truncated safetensors header")
        size = struct.unpack("<Q", prefix)[0]
        if size < 2 or size > 100_000_000 or size + 8 > stat.st_size:
            raise ValueError("Invalid safetensors header size")
        raw = stream.read(size)
    try:
        header = json.loads(raw, object_pairs_hook=_unique_pairs)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Invalid safetensors header JSON") from error
    if not isinstance(header, dict):
        raise ValueError("Safetensors header must be an object")
    spans = []
    for key, item in header.items():
        if key == "__metadata__":
            if not isinstance(item, dict) or any(not isinstance(v, str) for v in item.values()):
                raise ValueError("Invalid safetensors metadata")
            continue
        if not isinstance(item, dict) or item.get("dtype") not in _DTYPE_BYTES:
            raise ValueError(f"Invalid tensor dtype: {key}")
        shape, offsets = item.get("shape"), item.get("data_offsets")
        if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError(f"Invalid tensor shape: {key}")
        if not isinstance(offsets, list) or len(offsets) != 2 or any(type(n) is not int for n in offsets):
            raise ValueError(f"Invalid tensor offsets: {key}")
        start, end = offsets
        if start < 0 or end < start or end - start != math.prod(shape) * _DTYPE_BYTES[item["dtype"]]:
            raise ValueError(f"Invalid tensor offsets/size: {key}")
        spans.append((start, end, key))
    position = 0
    for start, end, key in sorted(spans):
        if start != position:
            raise ValueError(f"Noncontiguous/overlapping tensor offsets: {key}")
        position = end
    if position != stat.st_size - size - 8:
        raise ValueError("Tensor offsets do not match payload size")
    return header, {"path": str(path.resolve()), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                    "header_bytes": size, "header_sha256": hashlib.sha256(raw).hexdigest()}


def canonical_keys(header):
    result = {}
    for key in header:
        if key == "__metadata__":
            continue
        name = key
        for prefix in ("model.diffusion_model.", "diffusion_model."):
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        if name in result:
            raise ValueError(f"Duplicate canonical checkpoint key: {name}")
        result[name] = key
    return result


def _read_config(stream, item, payload_offset):
    start, end = item["data_offsets"]
    if item["dtype"] != "U8" or len(item["shape"]) != 1 or end - start > 65536:
        raise ValueError("Invalid quantization/attention configuration tensor")
    stream.seek(payload_offset + start)
    try:
        value = json.loads(stream.read(end - start), object_pairs_hook=_unique_pairs)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Invalid quantization/attention JSON") from error
    if not isinstance(value, dict):
        raise ValueError("Quantization/attention configuration must be an object")
    return value


def _global_quantization_specs(header):
    raw = header.get("__metadata__", {}).get("_quantization_metadata")
    if raw is None:
        return None
    try:
        metadata = json.loads(raw, object_pairs_hook=_unique_pairs)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Invalid global quantization metadata JSON") from error
    if (not isinstance(metadata, dict) or metadata.get("format") != "int8_convrot"
            or not isinstance(metadata.get("layers"), dict)):
        raise ValueError("Invalid global ConvRot quantization metadata")
    layers = metadata["layers"]
    return {name: layers[key] for name, key in canonical_keys(layers).items()}


def inspect_checkpoint(path, *, config=None):
    config = config or MossConfig()
    report = {"status": "INVALID", "architecture": "moss_image", "config": asdict(config)}
    try:
        header, identity = read_header(path)
        keys = canonical_keys(header)
        report["source"] = identity
        if not {"shared_block_modulation.weight", "layers.0.proj.ffn_up_gate.weight"} <= keys.keys():
            return {**report, "status": "UNSUPPORTED", "reason": "Not a packed MOSS checkpoint"}
        layers = {int(match.group(1)) for key in keys if (match := re.match(r"layers\.(\d+)\.", key))}
        if layers != set(range(config.num_layers)):
            raise ValueError(f"MOSS block count differs from expected {config.num_layers}")
        expected = expected_parameter_shapes(config)
        global_specs = _global_quantization_specs(header)
        for name in global_specs or {}:
            shape = expected.get(name + ".weight", ())
            if len(shape) != 2:
                raise ValueError(f"Quantization metadata target is not a known Linear: {name}")
        quantized_weights = {name for name in expected if name in keys and header[keys[name]]["dtype"] == "I8"}
        required_weights = {f"layers.{i}.proj.{name}.weight"
                            for i in range(config.num_layers) for name in _TARGETS}
        if not required_weights <= quantized_weights:
            raise ValueError("All packed MOSS adapter targets must use INT8 weights")
        if any(not name.endswith(".weight") or len(expected[name]) != 2 for name in quantized_weights):
            raise ValueError("INT8 is supported only for known Linear weights")
        permitted = set(expected) | {"rope.bands"}
        for name in quantized_weights:
            permitted.update((name.removesuffix(".weight") + ".weight_scale",
                              name.removesuffix(".weight") + ".comfy_quant"))
        permitted.update(f"layers.{i}.comfy_attention.config" for i in range(config.num_layers))
        missing, extra = set(expected) - keys.keys(), keys.keys() - permitted
        if missing or extra:
            raise ValueError(f"MOSS schema keys mismatch: missing={sorted(missing)[:5]}, unexpected={sorted(extra)[:5]}")
        for name, shape in expected.items():
            item = header[keys[name]]
            dtypes = {"I8"} if name in quantized_weights else {"F32", "F16", "BF16"}
            if item["shape"] != list(shape) or item["dtype"] not in dtypes:
                raise ValueError(f"MOSS parameter shape/dtype mismatch: {name}")
        if "rope.bands" in keys:
            item = header[keys["rope.bands"]]
            if item["shape"] != [config.dim // config.num_heads // 8] or item["dtype"] != "F32":
                raise ValueError("Invalid legacy rope.bands shape/dtype")
        quantization_specs, attention_configs = {}, {}
        with Path(path).open("rb") as stream:
            for weight_name in sorted(quantized_weights):
                prefix = weight_name.removesuffix(".weight")
                shape = expected[weight_name]
                scale_name, spec_name = prefix + ".weight_scale", prefix + ".comfy_quant"
                if scale_name not in keys:
                    raise ValueError(f"Missing INT8 scale: {prefix}")
                scale = header[keys[scale_name]]
                if scale["dtype"] != "F32" or scale["shape"] != [shape[0], 1]:
                    raise ValueError(f"Invalid per-row scale shape/dtype: {prefix}")
                spec = (global_specs or {}).get(prefix)
                if spec_name in keys:
                    tensor_spec = _read_config(stream, header[keys[spec_name]], identity["header_bytes"] + 8)
                    if spec is not None and spec != tensor_spec:
                        raise ValueError(f"Conflicting tensor/global quantization config: {prefix}")
                    spec = tensor_spec
                if not isinstance(spec, dict):
                    raise ValueError(f"Missing INT8 quantization config: {prefix}")
                if (spec.get("format") != "int8_tensorwise" or spec.get("convrot") is not True
                        or spec.get("per_row", True) is not True):
                    return {**report, "status": "UNSUPPORTED", "reason": f"Unsupported ConvRot format: {prefix}"}
                if type(spec.get("convrot_groupsize")) is not int or spec["convrot_groupsize"] != 256 or shape[1] % 256:
                    return {**report, "status": "UNSUPPORTED", "reason": f"Unsupported ConvRot group: {prefix}"}
                quantization_specs[prefix] = spec
            if set(global_specs or {}) - quantization_specs.keys():
                raise ValueError("Global quantization metadata describes non-INT8 weights")
            for index in range(config.num_layers):
                name = f"layers.{index}.comfy_attention.config"
                if name not in keys:
                    if global_specs is not None:
                        continue  # Global ConvRot files use native default attention unless explicitly configured.
                    raise ValueError(f"Missing attention config: {name}")
                value = _read_config(stream, header[keys[name]], identity["header_bytes"] + 8)
                if value != {"attention": "comfy_kitchen_int8"}:
                    return {**report, "status": "UNSUPPORTED", "reason": f"Unsupported attention config: {name}"}
                attention_configs[name] = value
        return {**report, "status": "SUPPORTED", "quantized_linears": len(quantization_specs),
                "quantization_metadata": quantization_specs,
                "quantization_config_source": "global_metadata" if global_specs is not None else "tensor_configs",
                "attention_configs": len(attention_configs), "attention_metadata": attention_configs,
                "tensor_count": len(keys)}
    except (ValueError, KeyError, TypeError, OSError) as error:
        return {**report, "status": "INVALID", "reason": str(error)}


def file_sha256(path, cancel=None):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            if cancel:
                cancel()
            digest.update(chunk)
    return digest.hexdigest()


def _replace(model, name, value):
    parent, _, child = name.rpartition(".")
    setattr(model.get_submodule(parent) if parent else model, child, value)


def load_training_model(path, *, device="cpu", compute_dtype=torch.bfloat16, blocks_to_swap=0,
                        config=None, progress=None, cancel=None):
    from .model_adapter import create_native_model
    from .quantization import FrozenLinear

    config = config or MossConfig()
    if not 0 <= blocks_to_swap <= config.num_layers:
        raise ValueError("blocks_to_swap is outside the MOSS block range")
    info = inspect_checkpoint(path, config=config)
    if info["status"] != "SUPPORTED":
        raise ValueError(info["reason"])
    header, _ = read_header(path)
    keys = canonical_keys(header)
    model = create_native_model(config, device="meta", dtype=compute_dtype)
    expected = dict(model.named_parameters())
    quantized = set(info["quantization_metadata"])
    permitted = set(expected) | set(info["attention_metadata"]) | {"rope.bands"}
    for name in quantized:
        permitted.update((name + ".comfy_quant", name + ".weight_scale"))
    missing, extra = set(expected) - keys.keys(), keys.keys() - permitted
    if missing or extra:
        raise ValueError(f"MOSS checkpoint keys mismatch: missing={sorted(missing)[:5]}, unexpected={sorted(extra)[:5]}")
    for name, parameter in expected.items():
        item = header[keys[name]]
        if item["shape"] != list(parameter.shape):
            raise ValueError(f"MOSS tensor shape mismatch: {name}")
        if name.removesuffix(".weight") not in quantized and item["dtype"] not in {"BF16", "F16", "F32"}:
            raise ValueError(f"MOSS non-quantized parameter must be floating point: {name}")
    destination = torch.device(device)
    used = set()
    with safe_open(str(path), framework="pt", device="cpu") as source:
        for index, (name, _) in enumerate(expected.items()):
            if name in used:
                continue
            if cancel:
                cancel()
            if progress:
                progress(index, len(expected), f"Loading {name}")
            module_name = name.removesuffix(".weight")
            value = source.get_tensor(keys[name])
            if module_name in quantized:
                scale = source.get_tensor(keys[module_name + ".weight_scale"])
                block = re.fullmatch(r"layers\.(\d+)\.proj\.\w+", module_name)
                staged = block is not None and int(block.group(1)) >= config.num_layers - blocks_to_swap
                storage_device = torch.device("cpu") if staged else destination
                bias_name = module_name + ".bias"
                bias = source.get_tensor(keys[bias_name]) if bias_name in expected else None
                if bias is not None:
                    if not torch.isfinite(bias).all():
                        raise ValueError(f"Non-finite MOSS parameter: {bias_name}")
                    bias = bias.to(storage_device)
                    used.add(bias_name)
                layer = FrozenLinear(value.to(storage_device), bias=bias, scale=scale.to(storage_device),
                                     group_size=info["quantization_metadata"][module_name]["convrot_groupsize"])
                _replace(model, module_name, layer)
            else:
                if not torch.isfinite(value).all():
                    raise ValueError(f"Non-finite MOSS parameter: {name}")
                dtype = torch.float32 if value.dtype == torch.float32 else compute_dtype
                _replace(model, name, nn.Parameter(value.to(device=destination, dtype=dtype), requires_grad=False))
            used.add(name)
    if any(p.device.type == "meta" for p in model.parameters()) or any(b.device.type == "meta" for b in model.buffers()):
        raise ValueError("MOSS loader left meta tensors uninitialized")
    model.requires_grad_(False)
    model.train()
    for name, value in info["attention_metadata"].items():
        model.get_submodule(name.removesuffix(".config")).config = value.copy()
    return model, {**info, "linear_backend": "pytorch_local_dequant", "compute_dtype": str(compute_dtype),
                   "offload_mode": "synchronous_cpu_linear_staging", "blocks_to_swap": blocks_to_swap}
