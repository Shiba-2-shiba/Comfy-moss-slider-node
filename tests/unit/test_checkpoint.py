import json
import struct

import pytest
import torch
from safetensors.torch import save_file

from moss_slider_node.checkpoint import MossConfig, expected_parameter_shapes, inspect_checkpoint, read_header


def minimal_checkpoint(path, *, bad_scale=False, bad_format=False):
    config = MossConfig(dim=256, ffn_dim=512, num_heads=2, num_layers=2)
    tensors = {name: torch.zeros(shape) for name, shape in expected_parameter_shapes(config).items()}
    for i in range(2):
        for name, shape in {"qkv": (768, 256), "o": (256, 256),
                            "ffn_up_gate": (1024, 256), "ffn_down": (256, 512)}.items():
            prefix = f"layers.{i}.proj.{name}"
            tensors[prefix + ".weight"] = torch.zeros(shape, dtype=torch.int8)
            tensors[prefix + ".weight_scale"] = torch.ones(shape[0], 2 if bad_scale else 1)
            spec = {"format": "unsupported" if bad_format else "int8_tensorwise",
                    "convrot": True, "convrot_groupsize": 256}
            tensors[prefix + ".comfy_quant"] = torch.tensor(list(json.dumps(spec).encode()), dtype=torch.uint8)
        tensors[f"layers.{i}.comfy_attention.config"] = torch.tensor(
            list(b'{"attention":"comfy_kitchen_int8"}'), dtype=torch.uint8)
    save_file(tensors, str(path))
    return config


def test_metadata_inspection_validates_packed_blocks(tmp_path):
    path = tmp_path / "moss.safetensors"
    config = minimal_checkpoint(path)
    report = inspect_checkpoint(path, config=config)
    assert report["status"] == "SUPPORTED"
    assert report["quantized_linears"] == 8
    assert report["attention_configs"] == 2
    header, identity = read_header(path)
    assert header["layers.0.proj.qkv.weight"]["shape"] == [768, 256]
    assert len(identity["header_sha256"]) == 64


@pytest.mark.parametrize("kind,status", [("bad_scale", "INVALID"), ("bad_format", "UNSUPPORTED")])
def test_invalid_or_unsupported_quantization_is_distinguished(tmp_path, kind, status):
    path = tmp_path / "bad.safetensors"
    config = minimal_checkpoint(path, **{kind: True})
    assert inspect_checkpoint(path, config=config)["status"] == status


def test_non_moss_and_truncated_file(tmp_path):
    path = tmp_path / "other.safetensors"
    save_file({"weight": torch.zeros(2, 2)}, str(path))
    assert inspect_checkpoint(path)["status"] == "UNSUPPORTED"
    path.write_bytes(struct.pack("<Q", 500) + b"{}")
    assert inspect_checkpoint(path)["status"] == "INVALID"
    with pytest.raises(ValueError, match="header"):
        read_header(path)


@pytest.mark.parametrize("key", ["text_embedder.proj.weight", "layers.0.norm1.weight", "final_linear_video.weight"])
def test_missing_non_quantized_parameters_are_rejected(tmp_path, key):
    from safetensors.torch import load_file

    path = tmp_path / "incomplete.safetensors"
    config = minimal_checkpoint(path)
    tensors = {name: value.clone() for name, value in load_file(path).items()}
    del tensors[key]
    save_file(tensors, str(path))
    report = inspect_checkpoint(path, config=config)
    assert report["status"] == "INVALID"
    assert "missing" in report["reason"].lower()


@pytest.mark.parametrize("kind", ["shape", "dtype", "extra"])
def test_wrong_non_quantized_schema_is_rejected(tmp_path, kind):
    from safetensors.torch import load_file

    path = tmp_path / "bad_schema.safetensors"
    config = minimal_checkpoint(path)
    tensors = {name: value.clone() for name, value in load_file(path).items()}
    if kind == "shape":
        tensors["text_embedder.proj.weight"] = torch.zeros(256, 32)
    elif kind == "dtype":
        tensors["final_norm_video.weight"] = torch.ones(256, dtype=torch.int32)
    else:
        tensors["unknown.weight"] = torch.ones(3)
    save_file(tensors, str(path))
    assert inspect_checkpoint(path, config=config)["status"] == "INVALID"


def test_overlap_and_duplicate_canonical_keys_are_rejected(tmp_path):
    path = tmp_path / "bad.safetensors"
    header = {"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
              "y": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"0000")
    with pytest.raises(ValueError, match="offset"):
        read_header(path)


def metadata_checkpoint(path):
    """StarNodes-style global metadata with all eligible Linear weights quantized."""
    from safetensors.torch import load_file

    config = minimal_checkpoint(path)
    tensors = {k: v.clone() for k, v in load_file(path).items()
               if not k.endswith((".comfy_quant", ".comfy_attention.config"))}
    layers = {}
    for name, shape in expected_parameter_shapes(config).items():
        if name.endswith(".weight") and len(shape) == 2 and shape[1] % 256 == 0:
            prefix = "model.diffusion_model." + name.removesuffix(".weight")
            layers[prefix] = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}
            tensors[name] = torch.zeros(shape, dtype=torch.int8)
            tensors[name.removesuffix(".weight") + ".weight_scale"] = torch.ones(shape[0], 1)
    save_file({"model.diffusion_model." + k: v for k, v in tensors.items()}, str(path),
              metadata={"_quantization_metadata": json.dumps({"format": "int8_convrot", "layers": layers})})
    return config


def test_global_metadata_supports_additional_frozen_linears_without_attention_records(tmp_path):
    path = tmp_path / "global.safetensors"
    config = metadata_checkpoint(path)
    report = inspect_checkpoint(path, config=config)
    assert report["status"] == "SUPPORTED", report
    assert report["quantized_linears"] == 24
    assert report["attention_configs"] == 0


@pytest.mark.parametrize("kind", ["missing_spec", "unknown_module", "non_linear", "bad_json",
                                  "duplicate_canonical", "missing_scale", "bad_bias", "bad_scale"])
def test_global_metadata_rejects_incomplete_or_unsafe_schema(tmp_path, kind):
    from safetensors import safe_open
    from safetensors.torch import load_file

    path = tmp_path / "global.safetensors"
    config = metadata_checkpoint(path)
    tensors = {k: v.clone() for k, v in load_file(path).items()}
    with safe_open(str(path), framework="pt") as source:
        metadata = source.metadata()
    quant = json.loads(metadata["_quantization_metadata"])
    prefix = "model.diffusion_model."
    if kind == "missing_spec":
        del quant["layers"][prefix + "layers.0.proj.qkv"]
    elif kind == "unknown_module":
        quant["layers"][prefix + "unknown"] = next(iter(quant["layers"].values()))
    elif kind == "non_linear":
        quant["layers"][prefix + "final_norm_video"] = next(iter(quant["layers"].values()))
    elif kind == "duplicate_canonical":
        quant["layers"]["layers.0.proj.qkv"] = quant["layers"][prefix + "layers.0.proj.qkv"]
    elif kind == "missing_scale":
        del tensors[prefix + "text_embedder.proj.weight_scale"]
    elif kind == "bad_bias":
        tensors[prefix + "text_embedder.proj.bias"] = torch.zeros(config.dim, dtype=torch.int8)
    elif kind == "bad_scale":
        tensors[prefix + "text_embedder.proj.weight_scale"] = torch.ones(config.dim, 2)
    metadata["_quantization_metadata"] = "{" if kind == "bad_json" else json.dumps(quant)
    save_file(tensors, str(path), metadata=metadata)
    assert inspect_checkpoint(path, config=config)["status"] == "INVALID"


@pytest.mark.parametrize("conflict", [False, True])
def test_global_and_tensor_quantization_configs_must_agree(tmp_path, conflict):
    from safetensors import safe_open
    from safetensors.torch import load_file

    path = tmp_path / "hybrid.safetensors"
    config = metadata_checkpoint(path)
    tensors = {k: v.clone() for k, v in load_file(path).items()}
    with safe_open(str(path), framework="pt") as source:
        metadata = source.metadata()
    spec = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 64 if conflict else 256}
    tensors["model.diffusion_model.layers.0.proj.qkv.comfy_quant"] = torch.tensor(
        list(json.dumps(spec).encode()), dtype=torch.uint8)
    save_file(tensors, str(path), metadata=metadata)
    report = inspect_checkpoint(path, config=config)
    assert report["status"] == ("INVALID" if conflict else "SUPPORTED")
    if conflict:
        assert "Conflicting" in report["reason"]
