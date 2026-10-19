from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
import math
import re

import torch
from torch import Tensor, nn
from torch.nn import functional as F


_TARGET_RE = re.compile(
    r"^layers\.(?P<layer>\d+)\.proj\."
    r"(?P<projection>qkv|o|ffn_up_gate|ffn_down)$"
)


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Module,
        rank: int,
        alpha: float,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        if not math.isfinite(float(alpha)) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        try:
            in_features = int(base.in_features)  # type: ignore[attr-defined]
            out_features = int(base.out_features)  # type: ignore[attr-defined]
        except (AttributeError, TypeError, ValueError) as exc:
            raise TypeError("base must expose integer in_features and out_features") from exc

        for parameter in base.parameters():
            parameter.requires_grad_(False)

        if device is None:
            reference = next(base.parameters(), None)
            if reference is None:
                weight = getattr(base, "weight", None)
                reference = weight if isinstance(weight, Tensor) else next(base.buffers(), None)
            device = reference.device if reference is not None and not reference.is_meta else "cpu"

        self.base = base
        self.rank = rank
        self.alpha = float(torch.tensor(alpha, dtype=torch.float32))
        self.scaling = self.alpha / self.rank
        self.multiplier = 1.0
        self.in_features = in_features
        self.out_features = out_features
        self.lora_down = nn.Linear(in_features, rank, bias=False, dtype=torch.float32, device=device)
        self.lora_up = nn.Linear(rank, out_features, bias=False, dtype=torch.float32, device=device)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)

    @property
    def lora_A(self) -> nn.Parameter:
        return self.lora_down.weight

    @property
    def lora_B(self) -> nn.Parameter:
        return self.lora_up.weight

    @property
    def weight(self) -> Tensor:
        return self.base.weight  # type: ignore[no-any-return, attr-defined]

    @property
    def bias(self) -> Tensor | None:
        return getattr(self.base, "bias", None)

    def forward(self, x: Tensor) -> Tensor:
        base_output = self.base(x)
        if self.multiplier == 0:
            return base_output
        with torch.autocast(device_type=x.device.type, enabled=False):
            residual = F.linear(F.linear(x.to(dtype=torch.float32), self.lora_A), self.lora_B)
        return base_output + residual.to(device=base_output.device, dtype=base_output.dtype) * (
            self.scaling * self.multiplier
        )


def inject_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 8,
    device: torch.device | str | None = None,
) -> list[str]:
    if rank <= 0 or not math.isfinite(float(alpha)) or alpha <= 0:
        raise ValueError("rank and alpha must be finite and positive")
    if any(isinstance(module, LoRALinear) for module in model.modules()):
        raise ValueError("LoRA has already been injected")
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    targets = [
        (name, module)
        for name, module in model.named_modules()
        if _TARGET_RE.fullmatch(name) and not isinstance(module, LoRALinear)
    ]
    if not targets:
        raise ValueError("no supported LoRA target layers matched")
    names: list[str] = []
    for name, module in targets:
        parent_name, attribute = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        setattr(parent, attribute, LoRALinear(module, rank, alpha, device=device))
        names.append(name)
    return names


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    parameters: list[nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            parameters.extend((module.lora_A, module.lora_B))
    return parameters


def _adapters(model: nn.Module) -> Iterator[tuple[str, LoRALinear]]:
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            match = _TARGET_RE.fullmatch(name)
            if match is None:
                raise ValueError(f"LoRA adapter is outside the supported target set: {name}")
            yield name, module


@contextmanager
def lora_multiplier(model: nn.Module, multiplier: float) -> Iterator[None]:
    if not torch.isfinite(torch.tensor(float(multiplier))):
        raise ValueError("multiplier must be finite")
    adapters = [module for module in model.modules() if isinstance(module, LoRALinear)]
    previous = [module.multiplier for module in adapters]
    for module in adapters:
        module.multiplier = float(multiplier)
    try:
        yield
    finally:
        for module, value in zip(adapters, previous, strict=True):
            module.multiplier = value


def _prefix(name: str) -> str:
    return "lora_unet_" + name.replace(".", "_")


def lora_state_dict(model: nn.Module) -> dict[str, Tensor]:
    state: dict[str, Tensor] = {}
    found = False
    for name, module in _adapters(model):
        found = True
        base_prefix = _prefix(name)
        if name.endswith(".qkv"):
            if module.out_features % 3:
                raise ValueError(f"packed qkv output is not divisible by three: {name}")
            for logical_name, up in zip(("q", "k", "v"), module.lora_B.chunk(3, dim=0), strict=True):
                logical_prefix = base_prefix[:-3] + logical_name
                state[f"{logical_prefix}.lora_down.weight"] = module.lora_A.detach().cpu().clone()
                state[f"{logical_prefix}.lora_up.weight"] = up.detach().cpu().clone()
                state[f"{logical_prefix}.alpha"] = torch.tensor(module.alpha, dtype=torch.float32)
        else:
            state[f"{base_prefix}.lora_down.weight"] = module.lora_A.detach().cpu().clone()
            state[f"{base_prefix}.lora_up.weight"] = module.lora_B.detach().cpu().clone()
            state[f"{base_prefix}.alpha"] = torch.tensor(module.alpha, dtype=torch.float32)
    if not found:
        raise ValueError("model has no supported LoRA adapters")
    return state


def _require_tensor(state: Mapping[str, Tensor], key: str, shape: tuple[int, ...]) -> Tensor:
    value = state[key]
    if not isinstance(value, Tensor):
        raise ValueError(f"{key} must be a tensor")
    if tuple(value.shape) != shape:
        raise ValueError(f"{key} has shape {tuple(value.shape)}, expected {shape}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{key} must contain only finite values")
    return value


def _require_alpha(state: Mapping[str, Tensor], key: str, expected: float) -> None:
    value = _require_tensor(state, key, ())
    expected = float(torch.tensor(expected, dtype=torch.float32))
    if float(value.item()) != expected:
        raise ValueError(f"{key} is {float(value.item())}, expected {expected}")


def load_lora_state_dict(model: nn.Module, state: Mapping[str, Tensor]) -> None:
    expected_keys = set(lora_state_dict(model))
    actual_keys = set(state)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise ValueError(f"LoRA keys do not match; missing={missing}, unexpected={unexpected}")

    updates: list[tuple[LoRALinear, Tensor, Tensor]] = []
    for name, module in _adapters(model):
        base_prefix = _prefix(name)
        if name.endswith(".qkv"):
            logical_prefixes = [base_prefix[:-3] + suffix for suffix in ("q", "k", "v")]
            down_values = [
                _require_tensor(state, f"{prefix}.lora_down.weight", tuple(module.lora_A.shape))
                for prefix in logical_prefixes
            ]
            if not torch.equal(down_values[0], down_values[1]) or not torch.equal(down_values[0], down_values[2]):
                raise ValueError(f"{name} q/k/v lora_down weights must share the same A")
            rows = module.out_features // 3
            up_values = [
                _require_tensor(state, f"{prefix}.lora_up.weight", (rows, module.rank))
                for prefix in logical_prefixes
            ]
            for prefix in logical_prefixes:
                _require_alpha(state, f"{prefix}.alpha", module.alpha)
            updates.append((module, down_values[0], torch.cat(up_values, dim=0)))
        else:
            down = _require_tensor(state, f"{base_prefix}.lora_down.weight", tuple(module.lora_A.shape))
            up = _require_tensor(state, f"{base_prefix}.lora_up.weight", tuple(module.lora_B.shape))
            _require_alpha(state, f"{base_prefix}.alpha", module.alpha)
            updates.append((module, down, up))

    with torch.no_grad():
        for module, down, up in updates:
            module.lora_A.copy_(down.to(device=module.lora_A.device, dtype=torch.float32))
            module.lora_B.copy_(up.to(device=module.lora_B.device, dtype=torch.float32))
