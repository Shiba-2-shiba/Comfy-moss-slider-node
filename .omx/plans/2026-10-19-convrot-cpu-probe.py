"""CPU-only analysis fixture for MOSS ConvRot compatibility planning.

This is not product code. It compares synthetic group-256 ConvRot tensors
between comfy-kitchen and the neighboring Krea slider reference without
loading a model, checkpoint, GPU kernel, or large tensor.
"""

from __future__ import annotations

import importlib.metadata
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import comfy_kitchen as ck
from comfy_kitchen.tensor import QuantizedTensor, TensorWiseINT8Layout
from comfy_kitchen.tensor.int8_utils import (
    _build_hadamard as ck_hadamard,
    _rotate_activation as ck_rotate_activation,
    _rotate_weight as ck_rotate_weight,
)


SEED = 20261019
GROUP_SIZE = 256
OUT_FEATURES = 8
INPUT_FEATURES = (256, 512)
BATCH_SIZE = 3
RTOL = 1e-5
ATOL = 1e-5


def independent_hadamard(size: int) -> torch.Tensor:
    h4 = torch.tensor(
        [[1.0, 1.0, 1.0, -1.0],
         [1.0, 1.0, -1.0, 1.0],
         [1.0, -1.0, 1.0, 1.0],
         [-1.0, 1.0, 1.0, 1.0]],
        dtype=torch.float32,
        device="cpu",
    )
    h = h4
    while h.shape[0] < size:
        h = torch.kron(h, h4)
    return h / (size**0.5)


def independent_rotate(x: torch.Tensor, group_size: int = GROUP_SIZE) -> torch.Tensor:
    h = independent_hadamard(group_size).to(dtype=x.dtype, device=x.device)
    grouped = x.reshape(-1, x.shape[-1] // group_size, group_size)
    return (grouped @ h).reshape_as(x)


def error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | bool]:
    error = (actual - expected).abs()
    return {
        "max_abs": float(error.max()),
        "mean_abs": float(error.mean()),
        "allclose_rtol_1e-5_atol_1e-5": bool(torch.allclose(actual, expected, rtol=RTOL, atol=ATOL)),
    }


def load_krea_helpers(script_path: Path):
    workspace = script_path.resolve().parents[3]
    krea_root = workspace / "Confyui-krea2-slider-node"
    if not krea_root.is_dir():
        raise FileNotFoundError(f"Krea reference repository not found relative to probe: {krea_root}")
    sys.path.insert(0, str(krea_root))
    from krea2_slider_node.quantization import FrozenLinear, hadamard, rotate

    return FrozenLinear, hadamard, rotate


def run_probe() -> dict:
    script_path = Path(__file__)
    FrozenLinear, krea_hadamard, krea_rotate = load_krea_helpers(script_path)
    cuda_initialized_before = torch.cuda.is_initialized()
    torch.manual_seed(SEED)

    cases = []
    with ck.use_backend("eager"):
        for in_features in INPUT_FEATURES:
            qdata = torch.randint(
                -127,
                128,
                (OUT_FEATURES, in_features),
                dtype=torch.int8,
                device="cpu",
            )
            scale = torch.rand(OUT_FEATURES, 1, dtype=torch.float32, device="cpu") * 0.02 + 0.001
            x = torch.randn(BATCH_SIZE, in_features, dtype=torch.float32, device="cpu")
            grad_out = torch.randn(BATCH_SIZE, OUT_FEATURES, dtype=torch.float32, device="cpu")

            params = TensorWiseINT8Layout.Params(
                scale=scale,
                orig_dtype=torch.float32,
                orig_shape=(OUT_FEATURES, in_features),
                is_weight=True,
                convrot=True,
                convrot_groupsize=GROUP_SIZE,
            )
            qweight = QuantizedTensor(qdata, "TensorWiseINT8Layout", params)

            # Comfy Kitchen dequantizes into the original, unrotated basis.
            ck_dequant = TensorWiseINT8Layout.dequantize(qdata, params)
            dequant_reference = independent_rotate(qdata.float() * scale)

            # Comfy Kitchen eager linear rotates then dynamically quantizes the
            # activation. Its reference therefore includes activation INT8.
            ck_x = x.detach().clone().requires_grad_(True)
            ck_output = F.linear(ck_x, qweight)
            x_rotated = independent_rotate(x)
            x_scale = (x_rotated.abs().amax(dim=-1, keepdim=True).float() / 127.0).clamp_min(1e-30)
            x_qdata = (x_rotated / x_scale).round().clamp(-127, 127).to(torch.int8)
            ck_forward_reference = (x_qdata.float() @ qdata.float().T) * x_scale * scale.T
            ck_output.backward(grad_out)

            # Both implementations propagate input gradients through the frozen
            # dequantized base weight; activation quantization is not part of
            # this reference gradient.
            input_grad_reference = independent_rotate(grad_out @ (qdata.float() * scale))

            # Krea's FrozenLinear rotates floating activations and performs a
            # floating matmul with the dequantized rotated weight. Keep this
            # comparison separate from the Comfy Kitchen activation-INT8 path.
            krea_layer = FrozenLinear(qdata, scale=scale, group_size=GROUP_SIZE)
            krea_x = x.detach().clone().requires_grad_(True)
            krea_output = krea_layer(krea_x)
            krea_forward_reference = F.linear(independent_rotate(x), qdata.float() * scale)
            krea_output.backward(grad_out)

            krea_h = krea_hadamard(GROUP_SIZE, "cpu", torch.float32)
            ck_h = ck_hadamard(GROUP_SIZE, "cpu", torch.float32)
            helper_weight = torch.randn(OUT_FEATURES, in_features, dtype=torch.float32, device="cpu")
            helper_activation = torch.randn(BATCH_SIZE, in_features, dtype=torch.float32, device="cpu")
            krea_weight_rotated = krea_rotate(helper_weight, GROUP_SIZE)
            ck_weight_rotated = ck_rotate_weight(helper_weight, ck_h, GROUP_SIZE)
            krea_activation_rotated = krea_rotate(helper_activation, GROUP_SIZE)
            ck_activation_rotated = ck_rotate_activation(helper_activation, ck_h, GROUP_SIZE)

            cases.append(
                {
                    "shape": {
                        "out_features": OUT_FEATURES,
                        "in_features": in_features,
                        "batch": BATCH_SIZE,
                        "group_size": GROUP_SIZE,
                    },
                    "helper_exact": {
                        "hadamard": bool(torch.equal(krea_h, ck_h)),
                        "weight_rotation": bool(torch.equal(krea_weight_rotated, ck_weight_rotated)),
                        "activation_rotation": bool(torch.equal(krea_activation_rotated, ck_activation_rotated)),
                    },
                    "helper_error": {
                        "hadamard": error_metrics(krea_h, ck_h),
                        "weight_rotation": error_metrics(krea_weight_rotated, ck_weight_rotated),
                        "activation_rotation": error_metrics(krea_activation_rotated, ck_activation_rotated),
                    },
                    "comfy_kitchen_dequant_vs_independent_unrotated_weight": error_metrics(
                        ck_dequant, dequant_reference
                    ),
                    "comfy_kitchen_dynamic_int8_forward_vs_independent_int8_reference": error_metrics(
                        ck_output.detach(), ck_forward_reference
                    ),
                    "comfy_kitchen_input_grad_vs_frozen_weight_reference": error_metrics(
                        ck_x.grad, input_grad_reference
                    ),
                    "krea_float_activation_forward_vs_independent_float_reference": error_metrics(
                        krea_output.detach(), krea_forward_reference
                    ),
                    "krea_input_grad_vs_frozen_weight_reference": error_metrics(
                        krea_x.grad, input_grad_reference
                    ),
                }
            )

    cuda_initialized_after = torch.cuda.is_initialized()
    passed = all(
        all(case["helper_exact"].values())
        and case["comfy_kitchen_dequant_vs_independent_unrotated_weight"]["allclose_rtol_1e-5_atol_1e-5"]
        and case["comfy_kitchen_dynamic_int8_forward_vs_independent_int8_reference"]["allclose_rtol_1e-5_atol_1e-5"]
        and case["comfy_kitchen_input_grad_vs_frozen_weight_reference"]["allclose_rtol_1e-5_atol_1e-5"]
        and case["krea_float_activation_forward_vs_independent_float_reference"]["allclose_rtol_1e-5_atol_1e-5"]
        and case["krea_input_grad_vs_frozen_weight_reference"]["allclose_rtol_1e-5_atol_1e-5"]
        for case in cases
    )
    return {
        "artifact": "CPU-only synthetic analysis fixture; not product code",
        "passed": passed,
        "seed": SEED,
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "comfy_kitchen": importlib.metadata.version("comfy-kitchen"),
        },
        "backend": "comfy_kitchen eager on CPU",
        "cuda_initialized_before": cuda_initialized_before,
        "cuda_initialized_after": cuda_initialized_after,
        "activation_path_distinction": {
            "comfy_kitchen": "groupwise Hadamard rotation followed by dynamic per-row INT8 activation quantization",
            "krea": "groupwise Hadamard rotation of floating activation followed by floating matmul",
        },
        "cases": cases,
        "unverified": [
            "CUDA backend numerical equivalence",
            "Triton backend numerical equivalence",
            "full-model and real-checkpoint execution",
        ],
    }


def main() -> None:
    output_path = Path(__file__).with_suffix(".json")
    result = run_probe()
    rendered = json.dumps(result, indent=2) + "\n"
    output_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
