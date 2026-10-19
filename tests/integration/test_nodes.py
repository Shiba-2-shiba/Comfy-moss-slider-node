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
    return package


def test_v3_registration_schema_and_non_cached_training(plugin):
    extension = asyncio.run(plugin.comfy_entrypoint())
    nodes = asyncio.run(extension.get_node_list())
    definitions = {cls.define_schema().node_id: cls for cls in nodes}
    assert set(definitions) == {"MossSliderInspect", "MossSliderTrain", "MossSliderApply"}
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
    monkeypatch.setattr(module.folder_paths, "get_folder_paths", lambda category: [str(tmp_path)])
    output = module.MossSliderTrain.execute("moss.safetensors", [], [], [], steps=3, rank=2, alpha=4,
                                            learning_rate=0.001, eta=0.5, seed=19, width=256, height=256,
                                            rollout_steps=4, gradient_checkpointing=True, blocks_to_swap=1,
                                            output_prefix="slider")
    assert calls["roles"] == {"target", "positive", "negative"}
    assert calls["config"].rank == 2 and calls["config"].blocks_to_swap == 1
    assert output.result[0] == "moss_sliders/slider.safetensors"
    assert "steps_completed" in output.result[1]
