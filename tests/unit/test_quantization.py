from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from moss_slider_node.quantization import FrozenLinear, hadamard, rotate, validate_quant_spec


H4 = torch.tensor(
    [
        [1.0, 1.0, 1.0, -1.0],
        [1.0, 1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0, 1.0],
    ]
) / 2.0


def independent_hadamard(size: int, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    raw_h4 = (H4 * 2.0).to(dtype=dtype)
    result = raw_h4
    while result.shape[0] < size:
        result = torch.kron(result, raw_h4)
    return result / size**0.5


def independent_rotate(x: torch.Tensor, group_size: int) -> torch.Tensor:
    matrix = independent_hadamard(group_size, dtype=x.dtype).to(x.device)
    grouped = x.reshape(-1, x.shape[-1] // group_size, group_size)
    return (grouped @ matrix).reshape_as(x)


def quant_spec(group_size: int = 256) -> dict:
    return {
        "format": "int8_tensorwise",
        "convrot": True,
        "convrot_groupsize": group_size,
    }


def test_validate_quant_spec_accepts_checkpoint_group_256() -> None:
    assert validate_quant_spec(quant_spec(), in_features=512) == 256


@pytest.mark.parametrize(
    ("spec", "in_features"),
    [
        ({}, 256),
        ({"format": "nvfp4", "convrot": True, "convrot_groupsize": 256}, 256),
        ({"format": "int8_tensorwise", "convrot": False, "convrot_groupsize": 256}, 256),
        (quant_spec(0), 256),
        (quant_spec(2), 256),
        (quant_spec(8), 256),
        (quant_spec(64), 300),
        (quant_spec(True), 256),
        (quant_spec(256), 0),
    ],
)
def test_validate_quant_spec_rejects_bad_format_group_or_width(spec: dict, in_features: int) -> None:
    with pytest.raises((TypeError, ValueError)):
        validate_quant_spec(spec, in_features)


def test_hadamard_uses_regular_h4_convention_and_is_orthonormal() -> None:
    torch.testing.assert_close(hadamard(4, "cpu", torch.float32), H4, rtol=0, atol=0)
    h256 = hadamard(256, torch.device("cpu"), torch.float64)
    torch.testing.assert_close(h256 @ h256.T, torch.eye(256, dtype=torch.float64), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(h256, h256.T, rtol=0, atol=0)


def test_hadamard_group_256_exactly_matches_comfy_kitchen() -> None:
    from comfy_kitchen.tensor.int8_utils import _build_hadamard

    expected = _build_hadamard(256, "cpu", torch.float32)

    assert torch.equal(hadamard(256, "cpu", torch.float32), expected)


def test_rotate_matches_independent_reference_for_multiple_groups_and_noncontiguous_input() -> None:
    source = torch.arange(2 * 512 * 3, dtype=torch.float64).reshape(2, 512, 3)
    x = source.transpose(1, 2)
    assert not x.is_contiguous()

    actual = rotate(x, 256)

    torch.testing.assert_close(actual, independent_rotate(x, 256), rtol=1e-12, atol=1e-12)
    assert actual.shape == x.shape


def test_int8_forward_bias_and_input_gradient_match_independent_fp32_reference() -> None:
    generator = torch.Generator().manual_seed(20261019)
    codes = torch.randint(-20, 21, (5, 512), dtype=torch.int8, generator=generator)
    scale = torch.rand(5, 1, dtype=torch.float32, generator=generator) * 0.03 + 0.001
    bias = torch.randn(5, dtype=torch.float32, generator=generator)
    layer = FrozenLinear(codes, bias, scale=scale, group_size=256)
    assert layer.in_features == 512
    assert layer.out_features == 5
    x = torch.randn(2, 3, 512, dtype=torch.float32, generator=generator, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_(True)
    dense_weight = independent_rotate(codes.float() * scale, 256)

    actual = layer(x)
    expected = F.linear(reference_x, dense_weight, bias)
    grad_output = torch.randn(actual.shape, dtype=torch.float32, generator=generator)
    actual.backward(grad_output)
    expected.backward(grad_output)

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(x.grad, reference_x.grad, rtol=1e-4, atol=1e-5)


def test_int8_input_gradient_passes_fp64_gradcheck() -> None:
    codes = torch.tensor(
        [[1, 2, -3, 4], [4, -2, 1, 3], [-2, 5, 4, -1]],
        dtype=torch.int8,
    )
    scale = torch.tensor([[0.25], [0.5], [0.125]], dtype=torch.float32)
    layer = FrozenLinear(codes, scale=scale, group_size=4)
    x = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(layer, (x,), eps=1e-6, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.bfloat16])
def test_floating_frozen_linear_matches_dense_reference(weight_dtype: torch.dtype) -> None:
    weight = torch.tensor([[1.0, -2.0, 0.5, 3.0], [-1.0, 0.25, 2.0, -0.5]], dtype=weight_dtype)
    bias = torch.tensor([0.5, -1.0], dtype=weight_dtype)
    layer = FrozenLinear(weight, bias)
    x = torch.tensor([[0.25, 1.0, -2.0, 0.5]], dtype=weight_dtype, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_(True)

    actual = layer(x)
    expected = F.linear(reference_x, weight, bias)
    actual.sum().backward()
    expected.sum().backward()

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(x.grad, reference_x.grad)


def test_multiple_frozen_layers_propagate_gradient_to_early_lora_residual() -> None:
    codes1 = torch.tensor(
        [[1, 2, -3, 4], [4, -2, 1, 3], [-2, 5, 4, -1], [3, 1, -4, 2]],
        dtype=torch.int8,
    )
    codes2 = torch.tensor([[2, -1, 3, 4], [-3, 2, 1, -2]], dtype=torch.int8)
    first = FrozenLinear(codes1, scale=torch.full((4, 1), 0.125), group_size=4)
    second = FrozenLinear(codes2, scale=torch.full((2, 1), 0.25), group_size=4)
    lora_down = nn.Linear(4, 2, bias=False, dtype=torch.float32)
    lora_up = nn.Linear(2, 4, bias=False, dtype=torch.float32)
    nn.init.normal_(lora_down.weight)
    nn.init.normal_(lora_up.weight)
    x = torch.randn(3, 4)

    output = second(first(x) + lora_up(lora_down(x)))
    output.square().mean().backward()

    assert lora_down.weight.grad is not None and torch.isfinite(lora_down.weight.grad).all()
    assert lora_up.weight.grad is not None and torch.isfinite(lora_up.weight.grad).all()
    assert lora_down.weight.grad.abs().sum() > 0
    assert lora_up.weight.grad.abs().sum() > 0


def test_frozen_storage_is_registered_as_unchanged_buffers_and_has_no_parameters() -> None:
    weight = torch.randint(-10, 11, (3, 256), dtype=torch.int8)
    scale = torch.rand(3, 1, dtype=torch.float32) + 0.01
    bias = torch.randn(3)
    layer = FrozenLinear(weight, bias, scale=scale, group_size=256)
    before = {name: tensor.detach().clone() for name, tensor in layer.named_buffers()}
    x = torch.randn(2, 256, requires_grad=True)

    layer(x).sum().backward()

    assert dict(layer.named_parameters()) == {}
    assert set(before) == {"weight", "scale", "bias"}
    for name, tensor in layer.named_buffers():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
        assert not tensor.requires_grad


def test_backward_does_not_save_a_dense_dequantized_weight() -> None:
    codes = torch.ones(8, 256, dtype=torch.int8)
    layer = FrozenLinear(codes, scale=torch.ones(8, 1, dtype=torch.float32), group_size=256)
    x = torch.randn(2, 256, requires_grad=True)
    saved: list[torch.Tensor] = []

    with torch.autograd.graph.saved_tensors_hooks(lambda tensor: saved.append(tensor) or tensor, lambda tensor: tensor):
        layer(x).square().sum().backward()

    assert any(tensor is layer.weight for tensor in saved)
    assert any(tensor is layer.scale for tensor in saved)
    assert not any(tensor.is_floating_point() and tensor.shape == layer.weight.shape for tensor in saved)


@pytest.mark.parametrize(
    ("weight", "bias", "scale", "group_size"),
    [
        (torch.ones(2, 4, dtype=torch.int8), None, None, 4),
        (torch.ones(2, 4, dtype=torch.int8), None, torch.ones(2), 4),
        (torch.ones(2, 4, dtype=torch.int8), None, torch.ones(1, 2), 4),
        (torch.ones(2, 4, dtype=torch.int8), None, torch.ones(2, 1, dtype=torch.float64), 4),
        (torch.ones(2, 4, dtype=torch.int8), None, torch.tensor([[1.0], [0.0]]), 4),
        (torch.ones(2, 4, dtype=torch.int8), None, torch.tensor([[1.0], [float("nan")]]), 4),
        (torch.ones(2, 6, dtype=torch.int8), None, torch.ones(2, 1), 4),
        (torch.ones(2, 4), None, torch.ones(2, 1), 0),
        (torch.ones(2, 4), None, None, 4),
        (torch.ones(2, 4), torch.ones(3), None, 0),
        (torch.ones(2, 4, dtype=torch.int32), None, None, 0),
        (torch.ones(2, 3, 4), None, None, 0),
    ],
)
def test_frozen_linear_rejects_bad_weight_bias_scale_or_group(
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    scale: torch.Tensor | None,
    group_size: int,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        FrozenLinear(weight, bias, scale=scale, group_size=group_size)
