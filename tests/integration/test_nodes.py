import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture
def plugin():
    root = Path(__file__).resolve().parents[2]
    name = "moss_slider_test_extension"
    spec = importlib.util.spec_from_file_location(name, root / "__init__.py", submodule_search_locations=[str(root)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    asyncio.run(package.comfy_entrypoint())
    return package


def test_v3_registration_schema_and_non_cached_training(plugin):
    extension = asyncio.run(plugin.comfy_entrypoint())
    nodes = asyncio.run(extension.get_node_list())
    definitions = {cls.define_schema().node_id: cls for cls in nodes}
    assert set(definitions) == {"MossSliderInspect", "MossSliderTrain", "MossSliderTrainFromPrompts", "MossSliderApply"}
    for cls in nodes:
        schema = cls.define_schema()
        assert schema.category and schema.description and schema.display_name
        assert schema.inputs and schema.outputs
        # Actual V3 schema emits the frontend-compatible V1 metadata contract.
        v1 = schema.get_v1_info(cls)
        assert v1 is not None
    schema = definitions["MossSliderTrain"].define_schema()
    assert schema.not_idempotent and schema.is_output_node


def test_train_node_forwards_settings_and_returns_relative_lora_path(plugin, monkeypatch, tmp_path):
    asyncio.run(plugin.comfy_entrypoint())
    module = sys.modules[plugin.__name__ + ".nodes"]
    calls = {}

    def fake_job(checkpoint_path, conditions, config, output_dir, prefix, **kwargs):
        calls.update(checkpoint=checkpoint_path, roles=set(conditions), config=config, output_dir=output_dir)
        output = Path(output_dir) / "slider.safetensors"
        return {"lora_path": output, "manifest_path": output.with_suffix(".json"), "steps_completed": 3}

    monkeypatch.setattr(module, "run_training_job", fake_job)
    monkeypatch.setattr(module.folder_paths, "get_full_path_or_raise", lambda category, name: str(tmp_path / name))
    monkeypatch.setattr(module.folder_paths, "get_output_directory", lambda: str(tmp_path))
    output = module.MossSliderTrain.execute("moss.safetensors", [], [], [], steps=3, rank=2, alpha=4,
                                            learning_rate=0.001, eta=0.5, seed=19, width=256, height=256,
                                            rollout_steps=4, gradient_checkpointing=True, blocks_to_swap=1,
                                            output_prefix="slider")
    assert calls["roles"] == {"target", "positive", "negative"}
    assert calls["config"].rank == 2 and calls["config"].blocks_to_swap == 1
    assert output.result[0] == "moss_sliders/slider.safetensors"
    assert "steps_completed" in output.result[1]


def test_yaml_training_node_exposes_file_selection_and_reuses_training_settings(plugin):
    extension = asyncio.run(plugin.comfy_entrypoint())
    definitions = {cls.define_schema().node_id: cls for cls in asyncio.run(extension.get_node_list())}
    cls = definitions["MossSliderTrainFromPrompts"]
    schema = cls.define_schema()
    info = schema.get_v1_info(cls)
    required = info.input["required"]
    assert "clip" in required and "prompt_yaml" in required
    assert "target_conditioning" not in required
    assert "aging_slider_fullbody.yaml" in required["prompt_yaml"][1]["options"]
    assert schema.is_output_node and schema.not_idempotent
    assert required["rollout_steps"] == definitions["MossSliderTrain"].define_schema().get_v1_info(
        definitions["MossSliderTrain"]).input["required"]["rollout_steps"]


def test_yaml_node_encodes_every_record_and_exports_the_selected_snapshot(plugin, monkeypatch, tmp_path):
    import json
    import torch

    asyncio.run(plugin.comfy_entrypoint())
    module = sys.modules[plugin.__name__ + ".nodes"]
    calls = {}

    class Clip:
        def tokenize(self, text):
            return text

        def encode_from_tokens_scheduled(self, tokens):
            return [[torch.ones(1, 2, 4096), {}]]

    def fake_job(checkpoint_path, records, config, output_dir, prefix, **kwargs):
        calls.update(records=records, config=config, source=kwargs["prompt_source"])
        output = Path(output_dir) / "preset.safetensors"
        return {"lora_path": output, "manifest_path": output.with_suffix(".json"), "steps_completed": config.steps}

    monkeypatch.setattr(module, "run_training_job", fake_job)
    monkeypatch.setattr(module.folder_paths, "get_full_path_or_raise", lambda category, name: str(tmp_path / name))
    monkeypatch.setattr(module.folder_paths, "get_output_directory", lambda: str(tmp_path))
    result = module.MossSliderTrainFromPrompts.execute("moss.safetensors", Clip(), "aging_slider_fullbody.yaml",
                                                     steps=6, rank=2, alpha=2, width=256, height=256)
    assert len(calls["records"]) == 6
    assert all(set(record) == {"target", "positive", "negative", "neutral"} for record in calls["records"])
    assert calls["source"]["file"] == "aging_slider_fullbody.yaml"
    assert len(calls["source"]["records"]) == 6 and len(calls["source"]["sha256"]) == 64
    assert calls["config"].steps == 6
    assert result.result[0] == "moss_sliders/preset.safetensors"
    assert json.loads(result.result[1])["steps_completed"] == 6
    assert result.ui["text"][0] == f"LoRA: {tmp_path / 'moss_sliders' / 'preset.safetensors'}"
    assert result.ui["text"][-1] == "Prompts: aging_slider_fullbody.yaml (6 records)"


def test_normalization_schema_is_optional_and_neutral_default(plugin):
    module = sys.modules[plugin.__name__ + ".nodes"]
    manual = module.MossSliderTrain.define_schema().get_v1_info(module.MossSliderTrain).input
    yaml_cls = module.MossSliderTrainFromPrompts
    yaml_input = yaml_cls.define_schema().get_v1_info(yaml_cls).input
    for inputs in [manual, yaml_input]:
        mode = inputs["optional"]["teacher_normalization"]
        assert mode[0] == "COMBO" and mode[1]["options"] == ["none", "neutral_l2"]
        assert mode[1]["default"] == "neutral_l2"
    assert "neutral_conditioning" in manual["optional"]
    assert all("neutral_conditioning" not in group for group in yaml_input.values())


@pytest.mark.parametrize("mode", [None, "none", "neutral_l2"])
def test_manual_neutral_and_omitted_mode_are_forwarded(plugin, monkeypatch, tmp_path, mode):
    module = sys.modules[plugin.__name__ + ".nodes"]
    seen = {}

    def fake_job(checkpoint_path, records, config, output_dir, prefix, **kwargs):
        seen.update(records=records, config=config)
        return {"lora_path": Path(output_dir) / "test.safetensors", "manifest_path": Path(output_dir) / "test.json"}

    monkeypatch.setattr(module, "run_training_job", fake_job)
    monkeypatch.setattr(module.folder_paths, "get_full_path_or_raise", lambda *_: str(tmp_path / "base"))
    monkeypatch.setattr(module.folder_paths, "get_output_directory", lambda: str(tmp_path))
    settings = {} if mode is None else {"teacher_normalization": mode}
    module.MossSliderTrain.execute("base", [], [], [], neutral_conditioning=["neutral"], **settings)
    assert seen["records"]["neutral"] == ["neutral"]
    assert seen["config"].teacher_normalization == (mode or "neutral_l2")


@pytest.mark.parametrize("node_id", ["MossSliderTrain", "MossSliderTrainFromPrompts"])
def test_comfy_prompt_validation_accepts_legacy_omitted_optional_fields(plugin, monkeypatch, node_id):
    import comfy

    monkeypatch.syspath_prepend(str(Path(next(iter(comfy.__path__))).resolve().parent))
    import execution

    module = sys.modules[plugin.__name__ + ".nodes"]
    monkeypatch.setattr(module.folder_paths, "get_filename_list", lambda *_: ["base.safetensors"])
    cls = getattr(module, node_id)

    class Source:
        RETURN_TYPES = ("CONDITIONING", "CLIP")

        @classmethod
        def INPUT_TYPES(cls):
            return {}

    monkeypatch.setitem(execution.nodes.NODE_CLASS_MAPPINGS, "NeutralTestSource", Source)
    monkeypatch.setitem(execution.nodes.NODE_CLASS_MAPPINGS, node_id, cls)
    required = cls.INPUT_TYPES()["required"]
    inputs = {}
    for name, spec in required.items():
        kind, opts = spec[0], spec[1] if len(spec) > 1 else {}
        if kind in ("CONDITIONING", "CLIP"):
            inputs[name] = ["source", 0 if kind == "CONDITIONING" else 1]
        else:
            inputs[name] = opts.get("default", opts.get("options", [None])[0])
    # An old prompt contains neither teacher_normalization nor neutral_conditioning.
    prompt = {"source": {"class_type": "NeutralTestSource", "inputs": {}},
              "train": {"class_type": node_id, "inputs": inputs}}
    valid, errors, _ = asyncio.run(execution.validate_inputs("neutral-test", prompt, "train", {}))
    assert valid, errors


def test_training_progress_logs_step_time_without_mixing_load_counts(plugin, monkeypatch, tmp_path, caplog):
    import logging

    asyncio.run(plugin.comfy_entrypoint())
    module = sys.modules[plugin.__name__ + ".nodes"]
    updates = []

    class Bar:
        def __init__(self, total):
            pass

        def update_absolute(self, current, total):
            updates.append((current, total))

    def fake_job(checkpoint_path, records, config, output_dir, prefix, **kwargs):
        progress = kwargs["progress"]
        progress("prepare", 0, 1, "Unloading inference models")
        progress("hash", 0, 1, "Hashing checkpoint")
        progress("load", 0, 378, "Loading a weight")
        progress("load", 377, 378, "Loading the last weight")
        progress("train", 0, 3, "Starting training")
        progress("train", 1, 3, {"loss": 0.125, "grad_norm": 0.25, "seconds": 2.5,
                                  "elapsed_seconds": 2.5, "eta_seconds": 5.0})
        progress("save", 0, 1, "Saving LoRA and manifest")
        progress("save", 1, 1, "Saved LoRA and manifest")
        output = Path(output_dir) / "progress.safetensors"
        return {"lora_path": output, "manifest_path": output.with_suffix(".json"), "steps_completed": 1}

    monkeypatch.setattr(module, "ProgressBar", Bar)
    monkeypatch.setattr(module, "run_training_job", fake_job)
    monkeypatch.setattr(module.folder_paths, "get_full_path_or_raise", lambda category, name: str(tmp_path / name))
    monkeypatch.setattr(module.folder_paths, "get_output_directory", lambda: str(tmp_path))
    with caplog.at_level(logging.INFO):
        module.MossSliderTrain.execute("moss.safetensors", [], [], [], steps=3)
    assert updates and all(total == 3 for _, total in updates)
    assert (1, 3) in updates
    assert "step 1/3" in caplog.text
    assert "2.50s/step" in caplog.text
    assert "ETA 5.0s" in caplog.text
    assert "loss=0.125" in caplog.text
    assert "Hashing checkpoint" in caplog.text
    assert "Saved LoRA and manifest" in caplog.text


@pytest.mark.parametrize("registered_lora_root", [False, True])
def test_training_saves_pair_under_comfy_output(plugin, native_checkpoint, monkeypatch, tmp_path,
                                               registered_lora_root):
    import json
    import torch

    asyncio.run(plugin.comfy_entrypoint())
    module = sys.modules[plugin.__name__ + ".nodes"]
    checkpoint_path, model_config, _ = native_checkpoint
    output_root = tmp_path / "output"
    lora_root = tmp_path / "shared_loras"
    original_job = module.run_training_job

    def cpu_job(*args, **kwargs):
        return original_job(*args, **kwargs, device="cpu", model_config=model_config)

    monkeypatch.setattr(module, "run_training_job", cpu_job)
    monkeypatch.setattr(module.folder_paths, "get_full_path_or_raise", lambda *_: str(checkpoint_path))
    monkeypatch.setattr(module.folder_paths, "get_output_directory", lambda: str(output_root))
    monkeypatch.setattr(module.folder_paths, "get_folder_paths",
                        lambda _: [str(lora_root)] if registered_lora_root else [])
    condition = [[torch.randn(1, 2, 4096), {}]]
    result = module.MossSliderTrain.execute("tiny_moss.safetensors", condition, condition, condition,
                                           steps=1, rank=2, alpha=2, width=16, height=16, rollout_steps=1,
                                           output_prefix="output_slider")
    expected = output_root / "moss_sliders" / "output_slider.safetensors"
    assert expected.is_file()
    assert expected.with_suffix(".json").is_file()
    assert not lora_root.exists()
    assert result.result[0] == "moss_sliders/output_slider.safetensors"
    report = json.loads(result.result[1])
    assert Path(report["lora_path"]) == expected
    assert Path(report["manifest_path"]) == expected.with_suffix(".json")
    assert result.ui["text"] == [f"LoRA: {expected}", f"Manifest: {expected.with_suffix('.json')}"]
