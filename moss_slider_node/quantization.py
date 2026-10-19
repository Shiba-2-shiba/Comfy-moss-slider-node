"""Frozen Linear layers for floating-point and ConvRot INT8 base weights.

The INT8 path keeps the checkpoint's rotated codes and per-row scales as
buffers.  Each call stages and dequantizes only that layer's weight, while a
custom backward propagates gradients to the input without retaining the dense
weight in autograd's saved tensors.
"""

from __future__ import annotations

from functools import lru_cache

import torch
from torch import nn
from torch.nn import functional as F


_H4 = (
    (1.0, 1.0, 1.0, -1.0),
    (1.0, 1.0, -1.0, 1.0),
    (1.0, -1.0, 1.0, 1.0),
    (-1.0, 1.0, 1.0, 1.0),
)


def _validate_group_size(group_size: object, in_features: int) -> int:
    if type(group_size) is not int:
        raise TypeError("ConvRot group size must be an integer")
    if group_size < 4 or group_size & (group_size - 1):
        raise ValueError("ConvRot group size must be a power of four")
    if (group_size.bit_length() - 1) % 2:
        raise ValueError("ConvRot group size must be a power of four")
    if in_features % group_size:
        raise ValueError("ConvRot group size must divide in_features")
    return group_size


def validate_quant_spec(spec: dict, in_features: int) -> int:
    """Validate supported checkpoint metadata and return its group size."""

    if not isinstance(spec, dict):
        raise TypeError("Quantization specification must be a dictionary")
    if type(in_features) is not int:
        raise TypeError("in_features must be an integer")
    if in_features <= 0:
        raise ValueError("in_features must be positive")
    if spec.get("format") != "int8_tensorwise" or spec.get("convrot") is not True:
        raise ValueError("Only tensor-wise ConvRot INT8 weights are supported")
    if spec.get("per_row", True) is not True:
        raise ValueError("ConvRot INT8 weights require per-row scales")
    return _validate_group_size(spec.get("convrot_groupsize"), in_features)


@lru_cache(maxsize=32)
def _cached_hadamard(size: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    raw_h4 = torch.tensor(_H4, device=torch.device(device), dtype=dtype)
    matrix = raw_h4
    while matrix.shape[0] < size:
        matrix = torch.kron(matrix, raw_h4)
    return matrix / size**0.5


def hadamard(size: int, device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
    """Return the normalized regular H4 Hadamard matrix used by ConvRot."""

    _validate_group_size(size, size)
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise TypeError("Hadamard dtype must be floating point")
    normalized_device = str(torch.device(device))
    return _cached_hadamard(size, normalized_device, dtype)


def rotate(x: torch.Tensor, group_size: int) -> torch.Tensor:
    """Apply the regular Hadamard rotation independently to feature groups."""

    if not isinstance(x, torch.Tensor):
        raise TypeError("ConvRot input must be a tensor")
    if not x.is_floating_point():
        raise TypeError("ConvRot input must be floating point")
    if x.ndim == 0:
        raise ValueError("ConvRot input must have a feature dimension")
    _validate_group_size(group_size, x.shape[-1])
    matrix = hadamard(group_size, x.device, x.dtype)
    grouped = x.reshape(-1, x.shape[-1] // group_size, group_size)
    return (grouped @ matrix).reshape(x.shape)


def _materialize(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    staged = weight.to(device=device, dtype=dtype)
    if scale.numel():
        staged = staged * scale.to(device=device, dtype=dtype)
    return staged


class _FrozenLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        scale: torch.Tensor,
        bias: torch.Tensor | None,
        group_size: int,
    ) -> torch.Tensor:
        ctx.save_for_backward(weight, scale)
        ctx.group_size = group_size
        dense_weight = _materialize(weight, scale, device=x.device, dtype=x.dtype)
        linear_input = rotate(x, group_size) if group_size else x
        staged_bias = bias.to(device=x.device, dtype=x.dtype) if bias is not None else None
        return F.linear(linear_input, dense_weight, staged_bias)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if not ctx.needs_input_grad[0]:
            return None, None, None, None, None
        weight, scale = ctx.saved_tensors
        dense_weight = _materialize(
            weight,
            scale,
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        grad_input = grad_output @ dense_weight
        if ctx.group_size:
            grad_input = rotate(grad_input, ctx.group_size)
        return grad_input, None, None, None, None


class FrozenLinear(nn.Module):
    """A Linear whose base weight, bias, and optional INT8 scale are buffers."""

    in_features: int
    out_features: int
    group_size: int

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        *,
        scale: torch.Tensor | None = None,
        group_size: int = 0,
    ) -> None:
        super().__init__()
        if not isinstance(weight, torch.Tensor):
            raise TypeError("Linear weight must be a tensor")
        if weight.ndim != 2 or not all(weight.shape):
            raise ValueError("Linear weight must be a non-empty matrix")
        self.out_features, self.in_features = weight.shape

        if bias is not None:
            if not isinstance(bias, torch.Tensor):
                raise TypeError("Linear bias must be a tensor")
            if bias.shape != (self.out_features,) or not bias.is_floating_point():
                raise ValueError("Linear bias must be a floating vector with out_features elements")

        if weight.dtype == torch.int8:
            _validate_group_size(group_size, self.in_features)
            if not isinstance(scale, torch.Tensor):
                raise ValueError("INT8 weight requires a scale tensor")
            if scale.dtype != torch.float32:
                raise ValueError("INT8 scale must use float32 storage")
            if scale.shape != (self.out_features, 1):
                raise ValueError("INT8 scale must have shape (out_features, 1)")
            if not torch.isfinite(scale).all() or not torch.all(scale > 0):
                raise ValueError("INT8 scales must be finite and positive")
        else:
            if weight.dtype not in (torch.float32, torch.bfloat16):
                raise ValueError("Frozen floating weights must use float32 or bfloat16 storage")
            if type(group_size) is not int or group_size != 0:
                raise ValueError("Floating weights do not use ConvRot groups")
            if scale is not None:
                raise ValueError("Floating weights do not use quantization scales")

        self.group_size = group_size
        self.register_buffer("weight", weight.detach())
        self.register_buffer(
            "scale",
            scale.detach() if scale is not None else torch.empty(0, dtype=torch.float32, device=weight.device),
        )
        self.register_buffer("bias", bias.detach() if bias is not None else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor) or not x.is_floating_point():
            raise TypeError("FrozenLinear input must be a floating-point tensor")
        if x.ndim == 0 or x.shape[-1] != self.in_features:
            raise ValueError(f"Expected input with {self.in_features} features")
        return _FrozenLinearFunction.apply(x, self.weight, self.scale, self.bias, self.group_size)
