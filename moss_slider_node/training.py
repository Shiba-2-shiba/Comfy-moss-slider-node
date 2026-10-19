from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TrainingConfig:
    steps: int = 100
    rank: int = 8
    alpha: float = 8.0
    learning_rate: float = 1e-4
    eta: float = 1.0
    seed: int = 0
    width: int = 512
    height: int = 512
    rollout_steps: int = 8
    gradient_checkpointing: bool = True
    blocks_to_swap: int = 0
    max_grad_norm: float = 1.0

    def validate(self) -> None:
        _positive_int("steps", self.steps)
        _positive_int("rank", self.rank)
        if self.rank > 128:
            raise ValueError("rank must be at most 128")
        _positive_finite("alpha", self.alpha)
        _positive_finite("learning_rate", self.learning_rate)
        _finite("eta", self.eta)
        if self.eta < 0:
            raise ValueError("eta must be non-negative")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be an integer in 0..2**63-1")
        for name, value in (("width", self.width), ("height", self.height)):
            _positive_int(name, value)
            if value % 16:
                raise ValueError(f"{name} must be divisible by 16")
        _positive_int("rollout_steps", self.rollout_steps)
        if not isinstance(self.gradient_checkpointing, bool):
            raise TypeError("gradient_checkpointing must be a bool")
        if (
            isinstance(self.blocks_to_swap, bool)
            or not isinstance(self.blocks_to_swap, int)
            or not 0 <= self.blocks_to_swap <= 42
        ):
            raise ValueError("blocks_to_swap must be an integer in 0..42")
        _positive_finite("max_grad_norm", self.max_grad_norm)


def _positive_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _positive_finite(name: str, value: object) -> None:
    if _finite(name, value) <= 0:
        raise ValueError(f"{name} must be positive")


def make_sigma_schedule(steps: int, device: torch.device | str = "cpu") -> Tensor:
    _positive_int("steps", steps)
    u = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float32, device=device)
    return 5.0 * u / (1.0 + 4.0 * u)


def slider_teacher(base: Tensor, positive: Tensor, negative: Tensor, eta: float) -> Tensor:
    _finite("eta", eta)
    return (base + float(eta) * (positive - negative)).detach()


def train_slider(
    model: nn.Module | Any,
    conditions: Mapping[str, Tensor] | Sequence[Mapping[str, Tensor]],
    config: TrainingConfig,
    *,
    progress: Callable[[int, int, dict[str, Any]], None] | None = None,
    cancel: Callable[[], None] | None = None,
) -> dict[str, Any]:
    from .lora import lora_multiplier, lora_parameters

    config.validate()
    if not callable(getattr(model, "predict_velocity", None)):
        raise TypeError("model must provide predict_velocity(x, sigma, conditioning)")
    if int(getattr(model, "latent_channels", -1)) != 128:
        raise ValueError("model.latent_channels must be 128")
    try:
        device = torch.device(model.device)
    except (AttributeError, TypeError, RuntimeError) as exc:
        raise TypeError("model.device must be a valid torch device") from exc

    records = [conditions] if isinstance(conditions, Mapping) else conditions
    if not isinstance(records, (list, tuple)) or not records:
        raise ValueError("At least one conditioning record is required")
    # Keep inactive records on CPU; only the selected record is resident on GPU.
    prepared_records = [_validate_conditions(record, torch.device("cpu")) for record in records]
    lora_model = getattr(model, "native_model", model)
    if not isinstance(lora_model, nn.Module):
        raise TypeError("model.native_model must be a torch.nn.Module")
    parameters = list(lora_parameters(lora_model))
    if not parameters:
        raise ValueError("model has no injected LoRA parameters")
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise ValueError("LoRA parameter list contains duplicates")
    for parameter in parameters:
        if parameter.dtype != torch.float32:
            raise TypeError("LoRA parameters must use FP32")
        if not parameter.requires_grad:
            raise ValueError("LoRA parameters must require gradients")
        if not torch.isfinite(parameter).all():
            raise ValueError("LoRA parameters must be finite")

    schedule = make_sigma_schedule(config.rollout_steps, device=device)
    sample_generator = torch.Generator(device="cpu").manual_seed(config.seed)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=0.0,
        foreach=False,
    )
    report: dict[str, Any] = {
        "settings": asdict(config),
        "prediction_type": "raw_flow_velocity",
        "objective": "full_mean_squared_error",
        "teacher": "shared_base_lora_disabled",
        "student_multiplier": 1.0,
        "schedule": [float(value) for value in schedule.detach().cpu()],
        "schedule_formula": "u=linspace(1,0,N+1); sigma=5*u/(1+4*u)",
        "noise_seed": config.seed,
        "sampling": "one_seeded_cpu_generator; draw timestep then noise per step",
        "optimizer_parameter_count": sum(parameter.numel() for parameter in parameters),
        "progress_callback": "progress(completed_step, total_steps, step_details)",
        "record_count": len(prepared_records),
        "record_schedule": "round_robin",
        "steps": [],
    }

    def check_cancel() -> None:
        if cancel is not None:
            cancel()

    started = perf_counter()
    try:
        for step in range(config.steps):
            check_cancel()
            tick = perf_counter()
            record_index = step % len(prepared_records)
            prepared_conditions = {role: value.to(device=device)
                                   for role, value in prepared_records[record_index].items()}
            timestep_index = int(
                torch.randint(0, config.rollout_steps, (1,), generator=sample_generator).item()
            )
            noise = torch.randn(
                1,
                128,
                config.height // 16,
                config.width // 16,
                dtype=torch.float32,
                generator=sample_generator,
            ).to(device=device)
            with torch.no_grad(), lora_multiplier(lora_model, 0.0):
                x = noise.clone()
                for rollout_index in range(timestep_index):
                    check_cancel()
                    sigma = schedule[rollout_index].reshape(1)
                    velocity = model.predict_velocity(x, sigma, prepared_conditions["target"])
                    _validate_prediction(velocity, x, "rollout")
                    delta_sigma = schedule[rollout_index + 1] - schedule[rollout_index]
                    x = (x + delta_sigma.to(dtype=x.dtype) * velocity).detach()

                sigma = schedule[timestep_index].reshape(1)
                teacher_predictions: dict[str, Tensor] = {}
                for role in ("target", "positive", "negative"):
                    check_cancel()
                    prediction = model.predict_velocity(x, sigma, prepared_conditions[role])
                    _validate_prediction(prediction, x, f"{role} teacher")
                    teacher_predictions[role] = prediction.detach()
                teacher = slider_teacher(
                    teacher_predictions["target"],
                    teacher_predictions["positive"],
                    teacher_predictions["negative"],
                    config.eta,
                )

            optimizer.zero_grad(set_to_none=True)
            check_cancel()
            with lora_multiplier(lora_model, 1.0), torch.enable_grad():
                student = model.predict_velocity(x, sigma, prepared_conditions["target"])
                _validate_prediction(student, teacher, "student")
                loss = F.mse_loss(student.float(), teacher.float(), reduction="mean")
            if not loss.requires_grad:
                raise RuntimeError("Loss has no LoRA gradient path")
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")
            loss.backward()
            for parameter in parameters:
                if parameter.grad is None:
                    raise RuntimeError("Missing LoRA gradient")
                if not torch.isfinite(parameter.grad).all():
                    raise RuntimeError("Non-finite LoRA gradient")
            grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters,
                config.max_grad_norm,
                error_if_nonfinite=True,
            )
            if not torch.isfinite(grad_norm):
                raise RuntimeError("Non-finite LoRA gradient norm")
            optimizer.step()
            entry = {
                "step": step + 1,
                "record_index": record_index,
                "loss": float(loss.detach()),
                "grad_norm": float(grad_norm.detach()),
                "timestep_index": timestep_index,
                "seed": config.seed,
                "noise_sequence_index": step,
                "teacher_requires_grad": teacher.requires_grad,
            }
            if device.type == "cuda":
                # ROCm uses the CUDA API too. Include queued optimizer work in wall time.
                torch.cuda.synchronize(device)
            now = perf_counter()
            elapsed = now - started
            entry.update(seconds=now - tick, elapsed_seconds=elapsed,
                         eta_seconds=elapsed / (step + 1) * (config.steps - step - 1))
            report["steps"].append(entry)
            if progress is not None:
                progress(step + 1, config.steps, entry)
        report["total_seconds"] = perf_counter() - started
        return report
    finally:
        optimizer.zero_grad(set_to_none=True)


def _validate_conditions(
    conditions: Mapping[str, Tensor], device: torch.device
) -> dict[str, Tensor]:
    if not isinstance(conditions, Mapping):
        raise TypeError("conditions must be a mapping")
    required = {"target", "positive", "negative"}
    missing = required.difference(conditions)
    if missing:
        raise ValueError(f"conditions are missing roles: {sorted(missing)}")
    prepared: dict[str, Tensor] = {}
    for role in ("target", "positive", "negative"):
        value = conditions[role]
        if not isinstance(value, Tensor):
            raise TypeError(f"conditions[{role!r}] must be a tensor")
        if value.layout != torch.strided or value.ndim != 3 or value.shape[0] != 1 or value.shape[1] < 1:
            raise ValueError(f"conditions[{role!r}] must have shape [1, L, D] with L >= 1")
        if value.shape[2] < 1:
            raise ValueError(f"conditions[{role!r}] feature dimension must be positive")
        if value.device.type != "cpu":
            raise ValueError(f"conditions[{role!r}] must be a CPU tensor")
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"conditions[{role!r}] must be a finite floating-point tensor")
        prepared[role] = value.to(device=device)
    return prepared


def _validate_prediction(prediction: object, reference: Tensor, name: str) -> None:
    if not isinstance(prediction, Tensor):
        raise TypeError(f"{name} prediction must be a tensor")
    if prediction.shape != reference.shape:
        raise ValueError(
            f"{name} prediction shape {tuple(prediction.shape)} does not match {tuple(reference.shape)}"
        )
    if not torch.isfinite(prediction).all():
        raise RuntimeError(f"Non-finite {name} prediction")
