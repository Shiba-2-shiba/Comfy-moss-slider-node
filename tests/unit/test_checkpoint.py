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
