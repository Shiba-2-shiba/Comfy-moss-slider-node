import json

import pytest
import torch
from safetensors.torch import save_file

from moss_slider_node.checkpoint import MossConfig
from moss_slider_node.model_adapter import create_native_model


@pytest.fixture
def native_checkpoint(tmp_path):
    from moss_slider_node.quantization import rotate

    torch.manual_seed(761)
    config = MossConfig(dim=256, ffn_dim=512, num_heads=2, num_layers=2)
    model = create_native_model(config, device="cpu", dtype=torch.float32)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "norm" in name and parameter.ndim == 1:
                parameter.fill_(1)
            else:
                parameter.normal_(0, 0.02)
    state = {name: p.detach().clone() for name, p in model.named_parameters()}
    dense_state = {name: value.clone() for name, value in state.items()}
    spec = torch.tensor(list(json.dumps({"format": "int8_tensorwise", "convrot": True,
                                        "convrot_groupsize": 256}).encode()), dtype=torch.uint8)
    for index in range(config.num_layers):
        for name in config.linear_shapes():
            prefix = f"layers.{index}.proj.{name}"
            rotated = rotate(state[prefix + ".weight"], 256)
            scale = (rotated.abs().amax(dim=1, keepdim=True) / 127).clamp_min(1e-8)
            packed = (rotated / scale).round().clamp(-127, 127).to(torch.int8)
            state[prefix + ".weight"] = packed
            state[prefix + ".weight_scale"] = scale
            state[prefix + ".comfy_quant"] = spec.clone()
            dense_state[prefix + ".weight"] = rotate(packed.float() * scale, 256)
        state[f"layers.{index}.comfy_attention.config"] = torch.tensor(
            list(b'{"attention":"comfy_kitchen_int8"}'), dtype=torch.uint8)
    state["rope.bands"] = torch.arange(16, dtype=torch.float32)
    path = tmp_path / "tiny_moss.safetensors"
    save_file(state, str(path))
    return path, config, dense_state
