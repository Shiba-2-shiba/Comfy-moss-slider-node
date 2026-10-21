import json

import pytest
import torch

from moss_slider_node.checkpoint import file_sha256, load_training_model
from moss_slider_node.export import save_lora
from moss_slider_node.inference import apply_slider, pack_logical_lora
from moss_slider_node.lora import inject_lora, lora_parameters, lora_state_dict
from moss_slider_node.model_adapter import create_native_model, training_context


def wrap_native(model, config, path):
    import comfy.model_base
    import comfy.model_patcher
    import comfy.sd
    import comfy.supported_models

    settings = {**vars(config), "image_model": "moss_image", "fused_qkv": True}
    model_config = comfy.supported_models.MOSSImage(settings)
    model_config.set_inference_dtype(torch.float32, None)
    base = comfy.model_base.MOSSImage(model_config, device="meta")
    base.diffusion_model = model
    patcher = comfy.model_patcher.ModelPatcher(base, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))
    patcher.cached_patcher_init = (comfy.sd.load_diffusion_model, (str(path), {}))
    return patcher


def test_full_42_block_export_uses_comfy_standard_slice_mapping():
    import comfy.lora
    from moss_slider_node.checkpoint import MossConfig

    config = MossConfig()
    native = create_native_model(config, dtype=torch.float32)
    inject_lora(native, rank=1, alpha=1, device="cpu")
    state = lora_state_dict(native)
    assert len(state) == 756
    # Mapping needs the unwrapped native parameter names, so construct a second meta-only model.
    base = wrap_native(create_native_model(config, dtype=torch.float32), config, "unused-meta-only-path")
    mapping = comfy.lora.model_lora_keys_unet(base.model)
    loaded = comfy.lora.load_lora(state, mapping)
    assert len(loaded) == 252
    assert sum(isinstance(key, tuple) for key in loaded) == 126


def test_packed_bypass_matches_training_lora_and_leaves_base_unchanged(native_checkpoint, tmp_path):
    path, config, _ = native_checkpoint
    training, _ = load_training_model(path, config=config, compute_dtype=torch.float32)
    inject_lora(training, rank=2, alpha=3)
    with torch.no_grad():
        for p in lora_parameters(training):
            p.normal_(0, 0.01)
    lora, _ = save_lora(training, tmp_path, "slider", {"source": {"sha256": file_sha256(path)}})
    inference, _ = load_training_model(path, config=config, compute_dtype=torch.float32)
    patcher = wrap_native(inference, config, path)
    snapshots = {name: b.clone() for name, b in inference.named_buffers()}
    original_forward = inference.layers[0].proj["qkv"].forward
    x, c, sigma = torch.randn(1, 128, 1, 2), torch.randn(1, 3, 4096), torch.tensor([0.4])
    from moss_slider_node.lora import lora_multiplier

    for strength in [0.0, -1.0, 1.0]:
        with lora_multiplier(training, strength), training_context(training):
            expected = training(x, sigma, c)
        applied = apply_slider(patcher, lora, strength)
        try:
            applied.inject_model()
            with training_context(inference):
                actual = inference(x, sigma, c)
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
        finally:
            applied.eject_model()
        for name, b in inference.named_buffers():
            torch.testing.assert_close(b, snapshots[name], rtol=0, atol=0)
        assert inference.layers[0].proj["qkv"].forward == original_forward
        assert patcher.injections == {} and patcher.patches == {}


def test_normalized_training_artifact_roundtrips_through_existing_apply(native_checkpoint, tmp_path):
    from moss_slider_node.job import run_training_job
    from moss_slider_node.training import TrainingConfig

    path, config, _ = native_checkpoint
    conditions = {role: [[torch.randn(1, 2, 4096), {}]]
                  for role in ("target", "positive", "negative", "neutral")}
    result = run_training_job(path, conditions,
                              TrainingConfig(steps=1, rank=2, alpha=2, width=16, height=16, rollout_steps=1),
                              tmp_path, "normalized", device="cpu", model_config=config)
    manifest = json.loads(result["manifest_path"].read_text(encoding="utf-8"))
    assert manifest["training"]["teacher_normalization"]["mode"] == "neutral_l2"
    inference, _ = load_training_model(path, config=config, compute_dtype=torch.float32)
    patcher = wrap_native(inference, config, path)
    before = {name: b.clone() for name, b in inference.named_buffers()}
    x, c, sigma = torch.randn(1, 128, 1, 1), torch.randn(1, 2, 4096), torch.tensor([.4])
    with training_context(inference), torch.no_grad():
        baseline = inference(x, sigma, c)
    for strength in [0., 1.]:
        applied = apply_slider(patcher, result["lora_path"], strength)
        try:
            applied.inject_model()
            with training_context(inference), torch.no_grad():
                actual = inference(x, sigma, c)
            assert torch.isfinite(actual).all()
            if strength == 0:
                torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
        finally:
            applied.eject_model()
    assert all(torch.equal(before[name], b) for name, b in inference.named_buffers())


def test_apply_rejects_wrong_base_identity_and_independent_qkv_a(native_checkpoint, tmp_path):
    path, config, _ = native_checkpoint
    native, _ = load_training_model(path, config=config, compute_dtype=torch.float32)
    inject_lora(native, rank=2, alpha=2)
    state = lora_state_dict(native)
    state["lora_unet_layers_0_proj_q.lora_down.weight"][0, 0] += 1
    with pytest.raises(ValueError, match="shared|common"):
        pack_logical_lora(state)
    lora, manifest = save_lora(native, tmp_path, "badbase", {"source": {"sha256": "0" * 64}})
    other, _ = load_training_model(path, config=config, compute_dtype=torch.float32)
    patcher = wrap_native(other, config, path)
    with pytest.raises(ValueError, match="identity|checkpoint"):
        apply_slider(patcher, lora, 1)
    data = json.loads(manifest.read_text())
    assert data["source"]["sha256"] == "0" * 64


def test_apply_rejects_mismatched_manifest_lora_pair(native_checkpoint, tmp_path):
    path, config, _ = native_checkpoint
    native, _ = load_training_model(path, config=config, compute_dtype=torch.float32)
    inject_lora(native, rank=2, alpha=0.1)
    manifest = {"source": {"sha256": file_sha256(path)}}
    first, first_json = save_lora(native, tmp_path, "first", manifest)
    with torch.no_grad():
        lora_parameters(native)[0].add_(1)
    second, second_json = save_lora(native, tmp_path, "second", manifest)
    # A plausible filename on copied JSON must still fail the byte digest binding.
    mismatched = json.loads(first_json.read_text())
    mismatched["lora_file"] = second.name
    second_json.write_text(json.dumps(mismatched))
    inference, _ = load_training_model(path, config=config, compute_dtype=torch.float32)
    with pytest.raises(ValueError, match="digest|artifact|pair"):
        apply_slider(wrap_native(inference, config, path), second, 1)


def test_actual_comfy_quantized_ffn_keeps_bypass_with_fused_activation(native_checkpoint, tmp_path):
    import comfy.model_management as management
    import comfy.ops
    from comfy.ldm.moss_image.model import MOSSImage
    from comfy_kitchen.tensor import QuantizedTensor
    from safetensors.torch import load_file

    path, config, _ = native_checkpoint
    trained, _ = load_training_model(path, config=config, compute_dtype=torch.float32)
    inject_lora(trained, rank=2, alpha=0.1)
    with torch.no_grad():
        for parameter in lora_parameters(trained):
            parameter.normal_(0, 0.02)
    lora, _ = save_lora(trained, tmp_path, "actual_quant", {"source": {"sha256": file_sha256(path)}})
    native = MOSSImage(**vars(config), fused_qkv=True, dtype=torch.float32, device="cpu",
                       operations=comfy.ops.mixed_precision_ops({}, compute_dtype=torch.float32))
    state = load_file(path)
    state.pop("rope.bands")
    native.load_state_dict(state, strict=True)
    native.requires_grad_(False)
    down = native.layers[0].proj["ffn_down"]
    assert isinstance(down.weight, QuantizedTensor)
    codes = down.weight._qdata.clone()
    patcher = wrap_native(native, config, path)
    hidden = torch.randn(1, 3, config.ffn_dim * 2)
    h, gate = hidden.chunk(2, dim=-1)
    activated = (4 * torch.tanh(gate / 4) * torch.sigmoid(gate)) * (25 * torch.tanh(h / 25))
    assert not management.in_training
    base = down(activated)
    trained_down = trained.layers[0].proj["ffn_down"]
    delta = torch.nn.functional.linear(
        torch.nn.functional.linear(activated, trained_down.lora_A), trained_down.lora_B
    ) * trained_down.scaling
    for strength in [-1.0, 1.0]:
        applied = apply_slider(patcher, lora, strength)
        try:
            applied.inject_model()
            assert down.comfy_force_forward
            actual = comfy.ops.linear_input_act(down, hidden, "situ")
            torch.testing.assert_close(actual, base + strength * delta, atol=1e-5, rtol=1e-4)
        finally:
            applied.eject_model()
        assert not down.comfy_force_forward
        assert torch.equal(down.weight._qdata, codes)


@pytest.mark.parametrize("strengths,same_lora", [
    ((0.75, -0.5), False), ((0.75, 0.0), False), ((0.0, -0.5), False),
    ((0.75, -0.75), True),
])
def test_chained_sliders_sum_deltas_and_restore_int8_base(native_checkpoint, tmp_path, strengths, same_lora):
    import comfy.ops
    from comfy.ldm.moss_image.model import MOSSImage
    from comfy_kitchen.tensor import QuantizedTensor
    from safetensors.torch import load_file

    path, config, _ = native_checkpoint
    artifacts, trained_models = [], []
    for index, (rank, alpha) in enumerate([(2, 0.1), (3, 1.5)]):
        trained, _ = load_training_model(path, config=config, compute_dtype=torch.float32)
        inject_lora(trained, rank=rank, alpha=alpha)
        with torch.no_grad():
            for parameter in lora_parameters(trained):
                parameter.normal_(0, 0.1)
        lora, _ = save_lora(trained, tmp_path, f"chain_{index}", {"source": {"sha256": file_sha256(path)}})
        artifacts.append(lora)
        trained_models.append(trained)
    if same_lora:
        artifacts[1], trained_models[1] = artifacts[0], trained_models[0]

    native = MOSSImage(**vars(config), fused_qkv=True, dtype=torch.float32, device="cpu",
                       operations=comfy.ops.mixed_precision_ops({}, compute_dtype=torch.float32))
    state = load_file(path)
    state.pop("rope.bands")
    native.load_state_dict(state, strict=True)
    native.requires_grad_(False)
    patcher = wrap_native(native, config, path)
    down = native.layers[0].proj["ffn_down"]
    assert isinstance(down.weight, QuantizedTensor)
    codes, scale = down.weight._qdata.clone(), down.weight._params.scale.clone()
    original_forward = down.forward
    hidden = torch.randn(1, 3, config.ffn_dim * 2)
    h, gate = hidden.chunk(2, dim=-1)
    activated = (4 * torch.tanh(gate / 4) * torch.sigmoid(gate)) * (25 * torch.tanh(h / 25))
    base = down(activated)
    deltas = []
    for trained in trained_models:
        adapter = trained.layers[0].proj["ffn_down"]
        deltas.append(torch.nn.functional.linear(
            torch.nn.functional.linear(activated, adapter.lora_A), adapter.lora_B
        ) * adapter.scaling)

    first = apply_slider(patcher, artifacts[0], strengths[0])
    chained = apply_slider(first, artifacts[1], strengths[1])
    third = apply_slider(chained, artifacts[0], 0.25)
    zero = apply_slider(chained, artifacts[1], 0.0)
    # Revisit upstream outputs after downstream use and repeat to catch stale wrappers.
    cases = [(first, base + strengths[0] * deltas[0]),
             (chained, base + strengths[0] * deltas[0] + strengths[1] * deltas[1]),
             (third, base + (strengths[0] + 0.25) * deltas[0] + strengths[1] * deltas[1]),
             (zero, base + strengths[0] * deltas[0] + strengths[1] * deltas[1])]
    for applied, expected in cases + list(reversed(cases)):
        try:
            applied.inject_model()
            actual = comfy.ops.linear_input_act(down, hidden, "situ")
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
        finally:
            applied.eject_model()
        assert down.forward == original_forward
        assert not down.comfy_force_forward
        assert torch.equal(down.weight._qdata, codes)
        assert torch.equal(down.weight._params.scale, scale)
    assert patcher.injections == {} and patcher.patches == {}
