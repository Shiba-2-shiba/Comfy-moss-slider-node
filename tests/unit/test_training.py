import math

import pytest
import torch
from torch import nn

from moss_slider_node.lora import LoRALinear, inject_lora, lora_multiplier, lora_parameters
from moss_slider_node.training import (
    TrainingConfig,
    make_sigma_schedule,
    slider_teacher,
    train_slider,
)


class _Projection(nn.Module):
    def __init__(self):
        super().__init__()
        self.ffn_down = nn.Linear(128, 128, bias=False)
        with torch.no_grad():
            self.ffn_down.weight.copy_(torch.eye(128))


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = _Projection()


class _ToyNativeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Layer(), _Layer()])
        self.unrelated_trainable = nn.Parameter(torch.tensor(0.25))


class _ToyAdapter:
    device = torch.device("cpu")
    latent_channels = 128

    def __init__(self):
        self.native_model = _ToyNativeModel()
        inject_lora(self.native_model, rank=2, alpha=2.0)
        self.calls = []

    def predict_velocity(self, x, sigma, conditioning):
        del sigma
        multipliers = [
            module.multiplier
            for module in self.native_model.modules()
            if isinstance(module, LoRALinear)
        ]
        self.calls.append(
            {
                "grad": torch.is_grad_enabled(),
                "conditioning": conditioning,
                "multipliers": multipliers,
                "x": x.detach().clone(),
            }
        )
        values = x.movedim(1, -1)
        for layer in self.native_model.layers:
            values = layer.proj.ffn_down(values)
        values = values + conditioning.mean().to(values) * 0.1
        return values.movedim(-1, 1)


def _numerical_steps(report):
    return [{key: value for key, value in entry.items()
             if key not in {"seconds", "elapsed_seconds", "eta_seconds"}} for entry in report["steps"]]


def _conditions():
    return {
        "target": torch.zeros(1, 2, 5),
        "positive": torch.full((1, 3, 5), 5.0),
        "negative": torch.full((1, 1, 5), -5.0),
    }


def _toy(seed=11):
    torch.manual_seed(seed)
    return _ToyAdapter()


@torch.no_grad()
def _probe_loss(model, conditions, x):
    sigma = torch.tensor([0.5])
    with lora_multiplier(model.native_model, 0.0):
        base = model.predict_velocity(x, sigma, conditions["target"])
        positive = model.predict_velocity(x, sigma, conditions["positive"])
        negative = model.predict_velocity(x, sigma, conditions["negative"])
        teacher = slider_teacher(base, positive, negative, 1.0)
    with lora_multiplier(model.native_model, 1.0):
        student = model.predict_velocity(x, sigma, conditions["target"])
    return float((student - teacher).square().mean())


def test_shifted_sigma_schedule_uses_the_native_shift_once():
    schedule = make_sigma_schedule(6)

    assert schedule.dtype == torch.float32
    torch.testing.assert_close(
        schedule,
        torch.tensor([1.0, 25 / 26, 10 / 11, 5 / 6, 5 / 7, 1 / 2, 0.0]),
    )
    assert schedule[3].item() == pytest.approx(5 / 6)


def test_sigma_schedule_rejects_invalid_step_counts():
    for value in (0, -1, 1.5, True):
        with pytest.raises((TypeError, ValueError), match="steps"):
            make_sigma_schedule(value)


def test_slider_teacher_is_detached_and_has_expected_degenerate_cases():
    base = torch.tensor([2.0], requires_grad=True)
    positive = torch.tensor([5.0], requires_grad=True)
    negative = torch.tensor([1.0], requires_grad=True)

    teacher = slider_teacher(base, positive, negative, 0.5)

    torch.testing.assert_close(teacher, torch.tensor([4.0]))
    assert not teacher.requires_grad
    torch.testing.assert_close(slider_teacher(base, positive, negative, 0.0), base.detach())
    torch.testing.assert_close(slider_teacher(base, positive, positive, 3.0), base.detach())


def test_flow_velocity_euler_step_has_noise_minus_clean_sign():
    clean = torch.tensor([2.0, -3.0])
    noise = torch.tensor([10.0, 5.0])
    sigma = torch.tensor(5 / 6)
    sigma_next = torch.tensor(5 / 8)
    x = (1 - sigma) * clean + sigma * noise
    velocity = noise - clean

    actual = x + (sigma_next - sigma) * velocity
    expected = (1 - sigma_next) * clean + sigma_next * noise

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("steps", 0),
        ("rank", 0),
        ("rank", 129),
        ("alpha", math.inf),
        ("alpha", 0.0),
        ("learning_rate", 0.0),
        ("learning_rate", math.nan),
        ("eta", math.inf),
        ("eta", -0.01),
        ("seed", -1),
        ("seed", 2**63),
        ("width", 24),
        ("height", 0),
        ("rollout_steps", 0),
        ("gradient_checkpointing", 1),
        ("blocks_to_swap", -1),
        ("blocks_to_swap", 43),
        ("max_grad_norm", 0.0),
        ("max_grad_norm", math.inf),
    ],
)
def test_training_config_rejects_invalid_values(field, value):
    config = TrainingConfig(**{field: value})

    with pytest.raises((TypeError, ValueError), match=field):
        config.validate()


def test_training_config_accepts_documented_defaults():
    config = TrainingConfig()

    assert config == TrainingConfig(
        steps=100,
        rank=8,
        alpha=8.0,
        learning_rate=1e-4,
        eta=1.0,
        seed=0,
        width=512,
        height=512,
        rollout_steps=8,
        gradient_checkpointing=True,
        blocks_to_swap=0,
        max_grad_norm=1.0,
    )
    config.validate()


def test_training_uses_disabled_lora_for_rollout_and_teachers_and_target_for_student():
    model = _toy()
    conditions = _conditions()

    report = train_slider(
        model,
        conditions,
        TrainingConfig(steps=1, rank=2, alpha=2.0, width=16, height=16, rollout_steps=2, seed=1),
    )

    student_calls = [call for call in model.calls if call["grad"]]
    teacher_calls = [call for call in model.calls if not call["grad"]]
    assert len(student_calls) == 1
    assert student_calls[0]["conditioning"] is conditions["target"]
    assert student_calls[0]["multipliers"] == [1.0, 1.0]
    assert len(teacher_calls) >= 3
    assert all(call["multipliers"] == [0.0, 0.0] for call in teacher_calls)
    assert report["steps"][0]["teacher_requires_grad"] is False
    assert report["objective"] == "full_mean_squared_error"


def test_training_updates_only_fp32_lora_parameters_and_improves_toy_loss():
    model = _toy()
    conditions = _conditions()
    native = model.native_model
    adapter_parameters = lora_parameters(native)
    adapter_parameter_ids = {id(parameter) for parameter in adapter_parameters}
    before_adapters = [parameter.detach().clone() for parameter in adapter_parameters]
    before_base = {
        name: parameter.detach().clone()
        for name, parameter in native.named_parameters()
        if id(parameter) not in adapter_parameter_ids
    }
    probe = torch.randn(1, 128, 1, 1, generator=torch.Generator().manual_seed(73))
    loss_before = _probe_loss(model, conditions, probe)

    report = train_slider(
        model,
        conditions,
        TrainingConfig(
            steps=12,
            rank=2,
            alpha=2.0,
            learning_rate=0.02,
            width=16,
            height=16,
            rollout_steps=2,
            seed=4,
        ),
    )

    assert all(parameter.dtype == torch.float32 for parameter in adapter_parameters)
    assert any(not torch.equal(before, after) for before, after in zip(before_adapters, adapter_parameters))
    assert any(parameter.count_nonzero() for parameter in adapter_parameters[1::2])
    assert all(torch.equal(before_base[name], parameter) for name, parameter in native.named_parameters() if name in before_base)
    losses = [entry["loss"] for entry in report["steps"]]
    assert all(math.isfinite(loss) for loss in losses)
    assert _probe_loss(model, conditions, probe) < loss_before
    assert report["optimizer_parameter_count"] == sum(parameter.numel() for parameter in adapter_parameters)


def test_training_is_reproducible_for_seed_and_initial_parameters():
    config = TrainingConfig(
        steps=3,
        rank=2,
        alpha=2.0,
        learning_rate=0.01,
        width=16,
        height=16,
        rollout_steps=3,
        seed=7,
    )
    first = _toy(31)
    second = _toy(31)

    first_report = train_slider(first, _conditions(), config)
    second_report = train_slider(second, _conditions(), config)

    assert _numerical_steps(first_report) == _numerical_steps(second_report)
    for first_parameter, second_parameter in zip(
        lora_parameters(first.native_model), lora_parameters(second.native_model)
    ):
        torch.testing.assert_close(first_parameter, second_parameter, rtol=0, atol=0)


def test_training_draws_a_new_reproducible_noise_and_timestep_per_step():
    config = TrainingConfig(
        steps=4,
        rank=2,
        alpha=2.0,
        learning_rate=0.01,
        width=16,
        height=16,
        rollout_steps=4,
        seed=19,
    )
    first = _toy(41)
    second = _toy(41)

    first_report = train_slider(first, _conditions(), config)
    second_report = train_slider(second, _conditions(), config)

    first_points = [call["x"] for call in first.calls if call["grad"]]
    second_points = [call["x"] for call in second.calls if call["grad"]]
    assert len(first_points) == config.steps
    assert all(not torch.equal(first_points[0], point) for point in first_points[1:])
    for actual, expected in zip(first_points, second_points, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert [entry["noise_sequence_index"] for entry in first_report["steps"]] == list(range(config.steps))
    assert _numerical_steps(first_report) == _numerical_steps(second_report)
    assert "timestep_index" not in first_report


def test_training_reports_progress_and_checks_cancel_around_model_calls():
    model = _toy()
    progress_calls = []
    cancel_calls = 0

    def cancel():
        nonlocal cancel_calls
        cancel_calls += 1
        if cancel_calls == 4:
            raise RuntimeError("cancelled by test")

    with pytest.raises(RuntimeError, match="cancelled by test"):
        train_slider(
            model,
            _conditions(),
            TrainingConfig(steps=2, rank=2, alpha=2.0, width=16, height=16, rollout_steps=2, seed=1),
            progress=lambda step, total, details: progress_calls.append((step, total, details)),
            cancel=cancel,
        )

    assert cancel_calls == 4
    assert not progress_calls
    assert all(
        module.multiplier == 1.0
        for module in model.native_model.modules()
        if isinstance(module, LoRALinear)
    )


def test_progress_receives_one_based_step_total_and_step_details():
    calls = []

    report = train_slider(
        _toy(),
        _conditions(),
        TrainingConfig(steps=2, rank=2, alpha=2.0, width=16, height=16, rollout_steps=1),
        progress=lambda step, total, details: calls.append((step, total, details.copy())),
    )

    assert [(step, total) for step, total, _ in calls] == [(1, 2), (2, 2)]
    assert [details for _, _, details in calls] == report["steps"]
    assert report["progress_callback"] == "progress(completed_step, total_steps, step_details)"


def test_training_rejects_nonfinite_loss_before_optimizer_step():
    model = _toy()

    def nonfinite_velocity(x, sigma, conditioning):
        del sigma, conditioning
        return x * torch.tensor(float("nan"))

    model.predict_velocity = nonfinite_velocity

    with pytest.raises(RuntimeError, match="Non-finite"):
        train_slider(
            model,
            _conditions(),
            TrainingConfig(steps=1, rank=2, alpha=2.0, width=16, height=16, rollout_steps=1),
        )


def test_multiple_records_cycle_in_order_without_averaging_conditions():
    model = _toy()
    records = [_conditions(), {"target": torch.full((1, 4, 5), 10.0),
                               "positive": torch.full((1, 5, 5), 15.0),
                               "negative": torch.full((1, 2, 5), -15.0)}]
    report = train_slider(model, records, TrainingConfig(steps=3, width=16, height=16, rollout_steps=1))
    students = [call["conditioning"] for call in model.calls if call["grad"]]
    assert len(students) == 3
    assert students[0] is records[0]["target"]
    assert students[1] is records[1]["target"]
    assert students[2] is records[0]["target"]
    assert report["record_count"] == 2
    assert [step["record_index"] for step in report["steps"]] == [0, 1, 0]
    teachers = [call["conditioning"] for call in model.calls if not call["grad"]]
    assert teachers[3] is records[1]["target"]
    assert teachers[4] is records[1]["positive"]
    assert teachers[5] is records[1]["negative"]


def test_step_timing_reports_elapsed_and_remaining_training_time(monkeypatch):
    import moss_slider_node.training as training

    ticks = iter([100.0, 100.0, 102.0, 102.0, 105.0, 105.0, 109.0, 109.0])
    monkeypatch.setattr(training, "perf_counter", lambda: next(ticks))
    events = []
    report = train_slider(_toy(), _conditions(), TrainingConfig(steps=3, width=16, height=16, rollout_steps=1),
                          progress=lambda step, total, entry: events.append(entry.copy()))
    assert [entry["seconds"] for entry in report["steps"]] == [2.0, 3.0, 4.0]
    assert [entry["elapsed_seconds"] for entry in report["steps"]] == [2.0, 5.0, 9.0]
    assert [entry["eta_seconds"] for entry in report["steps"]] == [4.0, 2.5, 0.0]
    assert report["total_seconds"] == 9.0
    assert [entry["seconds"] for entry in events] == [2.0, 3.0, 4.0]
