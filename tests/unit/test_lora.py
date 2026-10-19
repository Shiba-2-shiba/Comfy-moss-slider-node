from __future__ import annotations

import pytest
import torch
from torch import nn

from moss_slider_node.lora import (
    LoRALinear,
    inject_lora,
    load_lora_state_dict,
    lora_multiplier,
    lora_parameters,
    lora_state_dict,
)
from moss_slider_node.quantization import FrozenLinear


class _Projection(nn.Module):
    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.qkv = nn.Linear(width, width * 3, bias=False)
        self.o = nn.Linear(width, width, bias=False)
        self.ffn_up_gate = nn.Linear(width, width * 2, bias=False)
        self.ffn_down = nn.Linear(width * 2, width, bias=False)
        self.ignored = nn.Linear(width, width, bias=False)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = _Projection()


class _ToyModel(nn.Module):
    def __init__(self, blocks: int = 2) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Block() for _ in range(blocks)])
        self.out = nn.Linear(4, 4)


def _filled_model() -> _ToyModel:
    model = _ToyModel(blocks=1)
    inject_lora(model, rank=2, alpha=6)
    with torch.no_grad():
        for index, parameter in enumerate(lora_parameters(model), start=1):
            parameter.fill_(index / 10)
    return model


def test_non_binary_fraction_alpha_round_trips():
    model = _ToyModel(blocks=1)
    inject_lora(model, rank=2, alpha=0.1)
    with torch.no_grad():
        for p in lora_parameters(model):
            p.fill_(0.3)
    state = lora_state_dict(model)
    other = _ToyModel(blocks=1)
    inject_lora(other, rank=2, alpha=0.1)
    load_lora_state_dict(other, state)
    for a, b in zip(lora_parameters(model), lora_parameters(other)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_injects_only_packed_projection_targets_and_freezes_all_base_parameters() -> None:
    model = _ToyModel()

    names = inject_lora(model, rank=2, alpha=4)

    assert names == [
        "layers.0.proj.qkv",
        "layers.0.proj.o",
        "layers.0.proj.ffn_up_gate",
        "layers.0.proj.ffn_down",
        "layers.1.proj.qkv",
        "layers.1.proj.o",
        "layers.1.proj.ffn_up_gate",
        "layers.1.proj.ffn_down",
    ]
    assert isinstance(model.layers[0].proj.qkv, LoRALinear)
    assert model.layers[0].proj.qkv.lora_down.weight is model.layers[0].proj.qkv.lora_A
    assert model.layers[0].proj.qkv.lora_up.weight is model.layers[0].proj.qkv.lora_B
    assert isinstance(model.layers[0].proj.ignored, nn.Linear)
    assert isinstance(model.out, nn.Linear)
    assert all(not parameter.requires_grad for module in model.modules() if isinstance(module, LoRALinear) for parameter in module.base.parameters())
    assert all(parameter.dtype == torch.float32 for parameter in lora_parameters(model))
    assert all(torch.count_nonzero(module.lora_B) == 0 for module in model.modules() if isinstance(module, LoRALinear))


def test_raw_input_residual_supports_zero_positive_and_negative_multiplier() -> None:
    base = nn.Linear(3, 2, bias=True)
    layer = LoRALinear(base, rank=1, alpha=2)
    x = torch.tensor([[1.0, -2.0, 0.5]])
    with torch.no_grad():
        layer.lora_A.copy_(torch.tensor([[0.5, -1.0, 2.0]]))
        layer.lora_B.copy_(torch.tensor([[3.0], [-4.0]]))

    base_output = base(x)
    with lora_multiplier(layer, 0):
        zero_output = layer(x)
    with lora_multiplier(layer, 1):
        positive_output = layer(x)
    with lora_multiplier(layer, -1):
        negative_output = layer(x)

    torch.testing.assert_close(zero_output, base_output)
    torch.testing.assert_close(positive_output - base_output, -(negative_output - base_output))


def test_exposes_base_weight_and_bias_for_native_linear_input_activation() -> None:
    base = nn.Linear(3, 2, bias=True)
    layer = LoRALinear(base, rank=1, alpha=1)

    assert layer.weight is base.weight
    assert layer.bias is base.bias


def test_reinjection_is_rejected_without_freezing_existing_lora_parameters() -> None:
    model = _ToyModel(blocks=1)
    inject_lora(model, rank=2, alpha=2)

    with pytest.raises(ValueError, match="already"):
        inject_lora(model, rank=2, alpha=2)

    assert all(parameter.requires_grad for parameter in lora_parameters(model))


def test_b_gradient_arrives_first_and_a_gradient_arrives_after_one_update() -> None:
    layer = LoRALinear(nn.Linear(3, 2, bias=False), rank=2, alpha=2)
    optimizer = torch.optim.SGD([layer.lora_A, layer.lora_B], lr=0.1)
    x = torch.randn(4, 3)

    layer(x).square().mean().backward()
    assert torch.count_nonzero(layer.lora_B.grad) > 0
    assert torch.count_nonzero(layer.lora_A.grad) == 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    layer(x).square().mean().backward()
    assert torch.count_nonzero(layer.lora_A.grad) > 0


def test_qkv_export_copies_shared_a_slices_b_and_preserves_alpha() -> None:
    model = _filled_model()
    state = lora_state_dict(model)
    qkv = model.layers[0].proj.qkv

    for logical_index, logical_name in enumerate(("q", "k", "v")):
        prefix = f"lora_unet_layers_0_proj_{logical_name}"
        torch.testing.assert_close(state[f"{prefix}.lora_down.weight"], qkv.lora_A)
        torch.testing.assert_close(
            state[f"{prefix}.lora_up.weight"],
            qkv.lora_B.chunk(3, dim=0)[logical_index],
        )
        assert state[f"{prefix}.alpha"].item() == 6

    assert len(state) == 18


@pytest.mark.parametrize("corruption", ["missing", "nan", "independent_q_a", "wrong_alpha", "wrong_shape", "extra"])
def test_import_rejects_invalid_state_without_partial_mutation(corruption: str) -> None:
    source = _filled_model()
    state = lora_state_dict(source)
    target = _ToyModel(blocks=1)
    inject_lora(target, rank=2, alpha=6)
    before = [parameter.detach().clone() for parameter in lora_parameters(target)]

    if corruption == "missing":
        state.pop(next(iter(state)))
    elif corruption == "nan":
        key = next(key for key in state if key.endswith("lora_up.weight"))
        state[key] = state[key].clone()
        state[key].flatten()[0] = torch.nan
    elif corruption == "independent_q_a":
        key = "lora_unet_layers_0_proj_q.lora_down.weight"
        state[key] = state[key] + 1
    elif corruption == "wrong_alpha":
        state["lora_unet_layers_0_proj_k.alpha"] = torch.tensor(7.0)
    elif corruption == "wrong_shape":
        key = "lora_unet_layers_0_proj_o.lora_up.weight"
        state[key] = state[key][:-1]
    else:
        state["unexpected"] = torch.tensor(1.0)

    with pytest.raises(ValueError):
        load_lora_state_dict(target, state)

    for parameter, original in zip(lora_parameters(target), before, strict=True):
        torch.testing.assert_close(parameter, original)


def test_import_round_trip_and_multiplier_context_restores_previous_state() -> None:
    source = _filled_model()
    target = _ToyModel(blocks=1)
    inject_lora(target, rank=2, alpha=6)

    load_lora_state_dict(target, lora_state_dict(source))
    for actual, expected in zip(lora_parameters(target), lora_parameters(source), strict=True):
        torch.testing.assert_close(actual, expected)

    adapters = [module for module in target.modules() if isinstance(module, LoRALinear)]
    adapters[0].multiplier = 0.25
    with pytest.raises(RuntimeError), lora_multiplier(target, -2):
        assert all(module.multiplier == -2 for module in adapters)
        raise RuntimeError("probe")
    assert adapters[0].multiplier == 0.25
    assert all(module.multiplier == 1 for module in adapters[1:])


def test_base_weights_do_not_change_during_optimizer_steps() -> None:
    model = _ToyModel(blocks=1)
    inject_lora(model, rank=2, alpha=2)
    base_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if ".base." in name
    }
    optimizer = torch.optim.Adam(lora_parameters(model), lr=0.01)

    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        model.layers[0].proj.o(torch.randn(2, 4)).square().mean().backward()
        optimizer.step()

    for name, expected in base_before.items():
        torch.testing.assert_close(dict(model.named_parameters())[name], expected)


def test_frozen_int8_buffers_remain_unchanged_through_lora_backward() -> None:
    model = nn.Module()
    model.layers = nn.ModuleList([nn.Module()])
    model.layers[0].proj = nn.Module()
    weight = torch.randint(-8, 8, (12, 4), dtype=torch.int8)
    scale = torch.rand(12, 1, dtype=torch.float32) + 0.1
    base = FrozenLinear(weight, scale=scale, group_size=4)
    model.layers[0].proj.qkv = base
    weight_before = base.weight.clone()
    scale_before = base.scale.clone()

    assert inject_lora(model, rank=2, alpha=2) == ["layers.0.proj.qkv"]
    x = torch.randn(2, 4, requires_grad=True)
    model.layers[0].proj.qkv(x).square().mean().backward()

    assert model.layers[0].proj.qkv.lora_B.grad is not None
    assert model.layers[0].proj.qkv.lora_B.dtype == torch.float32
    assert torch.equal(base.weight, weight_before)
    assert torch.equal(base.scale, scale_before)
