import pytest
import torch

from moss_slider_node.model_adapter import normalize_conditioning


def test_conditioning_right_padding_is_trimmed_without_mutation():
    with torch.inference_mode():
        tensor = torch.randn(1, 5, 4096)
    mask = torch.tensor([[1, 1, 1, 0, 0]])
    source = [[tensor, {"attention_mask": mask, "pooled_output": None}]]
    result = normalize_conditioning(source)
    assert result.shape == (1, 3, 4096)
    assert result.device.type == "cpu" and not result.is_inference()
    assert torch.equal(result, tensor[:, :3])
    assert source[0][0] is tensor and torch.equal(mask, torch.tensor([[1, 1, 1, 0, 0]]))


@pytest.mark.parametrize("metadata", [
    {"attention_mask": torch.tensor([[1, 0, 1]])},
    {"attention_mask": torch.tensor([[0, 0, 0]])},
    {"attention_mask": torch.tensor([[0, 1, 1]])},
    {"hooks": None}, {"area": (1, 1, 0, 0)}, {"start_percent": 0.0},
])
def test_unsupported_conditions_are_rejected(metadata):
    with pytest.raises(ValueError):
        normalize_conditioning([[torch.ones(1, 3, 4096), metadata]])


def test_only_single_batch_single_entry_finite_features_are_allowed():
    for value in [[], [[torch.ones(2, 3, 4096), {}]], [[torch.ones(1, 3, 32), {}]],
                  [[torch.full((1, 3, 4096), float("nan")), {}]],
                  [[torch.ones(1, 3, 4096), {}]] * 2]:
        with pytest.raises(ValueError):
            normalize_conditioning(value)
