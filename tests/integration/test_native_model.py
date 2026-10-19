import copy

import pytest
import torch

from moss_slider_node.checkpoint import MossConfig, expected_parameter_shapes, load_training_model
from moss_slider_node.model_adapter import MossTrainingAdapter, create_native_model, training_context


def test_metadata_schema_matches_real_native_model_without_loading_weights():
    config = MossConfig()
    native = create_native_model(config)
    assert expected_parameter_shapes(config) == {name: tuple(p.shape) for name, p in native.named_parameters()}


def test_streamed_native_moss_matches_dense_reference(native_checkpoint):
    path, config, dense_state = native_checkpoint
    model, info = load_training_model(path, config=config, compute_dtype=torch.float32)
    reference = create_native_model(config, device="cpu", dtype=torch.float32)
    reference.load_state_dict(dense_state)
    assert info["quantized_linears"] == 8
    assert not any(p.requires_grad for p in model.parameters())
    assert not any(t.device.type == "meta" for t in (*model.parameters(), *model.buffers()))
    assert model.layers[0].proj["qkv"].weight.dtype == torch.int8
    x, context = torch.randn(1, 128, 1, 2), torch.randn(1, 3, 4096)
    sigma = torch.tensor([0.4])
    with training_context(reference):
        expected = reference(x, sigma, context)
    with training_context(model):
        actual = MossTrainingAdapter(model, dtype=torch.float32).predict_velocity(x, sigma, context)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


def test_gradient_checkpointing_preserves_early_lora_gradients(native_checkpoint):
    from moss_slider_node.lora import inject_lora, lora_parameters

    path, config, _ = native_checkpoint
    model, _ = load_training_model(path, config=config, compute_dtype=torch.float32)
    inject_lora(model, rank=2, alpha=2)
    with torch.no_grad():
        for parameter in lora_parameters(model):
            parameter.normal_(0, 0.02)
    other = copy.deepcopy(model)
    x, context = torch.randn(1, 128, 1, 2), torch.randn(1, 3, 4096)
    gradients, outputs = [], []
    for current, enabled in [(model, False), (other, True)]:
        with training_context(current):
            out = MossTrainingAdapter(current, dtype=torch.float32, gradient_checkpointing=enabled).predict_velocity(
                x, torch.tensor([0.5]), context)
            out.square().mean().backward()
        outputs.append(out.detach())
        gradients.append([p.grad.clone() for p in lora_parameters(current)])
    torch.testing.assert_close(outputs[0], outputs[1])
    for a, b in zip(*gradients):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-4)
        assert torch.isfinite(a).all()
    assert gradients[0][1].abs().sum() > 0


@pytest.mark.parametrize("initial_compiler_disabled", [False, True])
def test_context_restores_flags_and_attention_after_failure(native_checkpoint, monkeypatch, initial_compiler_disabled):
    import comfy.model_management as management
    import comfy.model_prefetch as prefetch

    path, config, _ = native_checkpoint
    model, _ = load_training_model(path, config=config, compute_dtype=torch.float32)
    attention = model.layers[0].comfy_attention

    def forbidden(*args, **kwargs):
        raise AssertionError("INT8 attention must not run during training")

    attention.function = forbidden
    monkeypatch.setattr(prefetch.args, "disable_comfy_compiler", initial_compiler_disabled)
    old_flag = prefetch.args.disable_comfy_compiler
    with pytest.raises(ValueError, match="cancel"):
        with torch.inference_mode(), training_context(model):
            assert management.in_training and prefetch.args.disable_comfy_compiler
            assert attention.function is not forbidden
            raise ValueError("cancel")
    assert not management.in_training and prefetch.args.disable_comfy_compiler == old_flag
    assert attention.function is forbidden
    assert not any(block._forward_pre_hooks for block in model.text_embedder.blocks)


def test_raw_velocity_and_apply_model_denoised_contract(native_checkpoint):
    import comfy.model_base
    import comfy.supported_models

    path, config, _ = native_checkpoint
    model, _ = load_training_model(path, config=config, compute_dtype=torch.float32)
    settings = {**vars(config), "image_model": "moss_image", "fused_qkv": True}
    model_config = comfy.supported_models.MOSSImage(settings)
    model_config.set_inference_dtype(torch.float32, None)
    wrapper = comfy.model_base.MOSSImage(model_config, device="meta")
    wrapper.diffusion_model = model
    x, context = torch.randn(1, 128, 1, 2), torch.randn(1, 3, 4096)
    adapter = MossTrainingAdapter(model, dtype=torch.float32, gradient_checkpointing=False)
    with training_context(model):
        for value in [0.0, 0.25, 1.0]:
            sigma = torch.tensor([value])
            velocity = adapter.predict_velocity(x, sigma, context)
            direct = model(x, sigma, context)
            denoised = wrapper.apply_model(x, sigma, c_crossattn=context)
            torch.testing.assert_close(velocity, direct)
            torch.testing.assert_close(denoised, x - sigma * velocity)
            assert torch.isfinite(velocity).all()


def test_sigma_embedding_matches_sd_scripts_time_units():
    from comfy.ldm.flux.layers import timestep_embedding

    sigma = torch.tensor([0.0, 0.25, 0.5, 1.0])
    frequency = 10000 ** (-torch.arange(128, dtype=torch.float64) / 128)
    angles = (sigma.double() * 1000)[:, None] * frequency[None]
    expected = torch.cat([angles.cos(), angles.sin()], dim=-1).float()
    torch.testing.assert_close(timestep_embedding(sigma, 256), expected, atol=1e-4, rtol=1e-4)


def test_loading_cancel_never_returns_partial_model(native_checkpoint):
    path, config, _ = native_checkpoint
    calls = 0

    def cancel():
        nonlocal calls
        calls += 1
        if calls == 3:
            raise InterruptedError("cancel load")

    with pytest.raises(InterruptedError):
        load_training_model(path, config=config, compute_dtype=torch.float32, cancel=cancel)
    model, _ = load_training_model(path, config=config, compute_dtype=torch.float32)
    assert model.layers[1].proj["qkv"].weight.dtype == torch.int8


def test_bfloat16_compute_accepts_float32_non_quantized_checkpoint_tensors(native_checkpoint):
    from moss_slider_node.lora import inject_lora, lora_parameters

    path, config, _ = native_checkpoint
    model, _ = load_training_model(path, config=config, compute_dtype=torch.bfloat16)
    inject_lora(model, rank=2, alpha=2)
    adapter = MossTrainingAdapter(model, dtype=torch.bfloat16, gradient_checkpointing=True)
    with training_context(model):
        output = adapter.predict_velocity(torch.randn(1, 128, 1, 2), torch.tensor([0.5]), torch.randn(1, 3, 4096))
        output.float().square().mean().backward()
    assert torch.isfinite(output).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in lora_parameters(model))
