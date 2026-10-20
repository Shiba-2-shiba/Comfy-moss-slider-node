import math

import pytest
import torch

import moss_slider_node.training as training


def test_normalization_defaults_to_neutral_and_accepts_none():
    assert training.TrainingConfig().teacher_normalization == "neutral_l2"
    training.TrainingConfig(teacher_normalization="none").validate()
    training.TrainingConfig(teacher_normalization="neutral_l2").validate()


@pytest.mark.parametrize("mode", [None, True, 1, "neutral", "mean_std", []])
def test_invalid_normalization_mode(mode):
    with pytest.raises(ValueError, match="teacher_normalization"):
        training.TrainingConfig(teacher_normalization=mode).validate()


def test_normalization_matches_reference_norm_per_sample_and_detaches():
    teacher = torch.tensor([[3., 4.], [0., 2.]], requires_grad=True)
    reference = torch.tensor([[0., 10.], [0., 1.]], requires_grad=True)
    before = teacher.detach().clone(), reference.detach().clone()
    out, stats = training.normalize_teacher_l2(teacher, reference)
    torch.testing.assert_close(out, torch.tensor([[6., 8.], [0., 1.]]))
    torch.testing.assert_close(stats["scale"], torch.tensor([2., .5]))
    assert not out.requires_grad and not stats["skipped"].any()
    assert all(not value.requires_grad and value.shape == (2,) for value in stats.values())
    torch.testing.assert_close(teacher.detach(), before[0])
    torch.testing.assert_close(reference.detach(), before[1])


@pytest.mark.parametrize("t,r", [(0., 1.), (1., 0.), (0., 0.), (1e-9, 1.), (1e-8, 1.)])
def test_normalization_skips_degenerate_norms(t, r):
    teacher, reference = torch.tensor([[t, 0.]]), torch.tensor([[r, 0.]])
    out, stats = training.normalize_teacher_l2(teacher, reference)
    torch.testing.assert_close(out, teacher)
    assert stats["scale"].item() == 1 and stats["skipped"].item()


def test_bfloat16_normalization_is_float32():
    out, _ = training.normalize_teacher_l2(torch.tensor([[3., 4.]]).bfloat16(),
                                          torch.tensor([[0., 10.]]).bfloat16())
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, torch.tensor([[6., 8.]]))


@pytest.mark.parametrize("teacher,reference", [
    (torch.ones(1, 2), torch.ones(1, 3)),
    (torch.ones(2), torch.ones(2)),
    (torch.ones(1, 2, dtype=torch.int64), torch.ones(1, 2)),
    (torch.tensor([[math.nan, 1.]]), torch.ones(1, 2)),
    (torch.ones(1, 2), torch.tensor([[math.inf, 1.]])),
])
def test_normalization_rejects_invalid_inputs(teacher, reference):
    with pytest.raises(ValueError):
        training.normalize_teacher_l2(teacher, reference)


def test_normalization_rejects_norm_overflow():
    with pytest.raises(RuntimeError, match="Non-finite"):
        training.normalize_teacher_l2(torch.full((1, 2), 1e30), torch.ones(1, 2))
