from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest
import torch
from torch import nn

import moss_slider_node.export as export_module
from moss_slider_node.export import read_lora, save_lora
from moss_slider_node.lora import inject_lora, lora_parameters


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Module()])
        self.layers[0].proj = nn.Module()
        self.layers[0].proj.qkv = nn.Linear(2, 6, bias=False)
        self.layers[0].proj.o = nn.Linear(2, 2, bias=False)
        self.layers[0].proj.ffn_up_gate = nn.Linear(2, 4, bias=False)
        self.layers[0].proj.ffn_down = nn.Linear(4, 2, bias=False)


def _model() -> _Model:
    model = _Model()
    inject_lora(model, rank=1, alpha=3)
    with torch.no_grad():
        for index, parameter in enumerate(lora_parameters(model), start=1):
            parameter.fill_(index)
    return model


def test_save_read_round_trip_metadata_manifest_and_unique_collision(tmp_path: Path) -> None:
    model = _model()
    first_lora, first_json = save_lora(model, tmp_path, "slider", {"base_identity": "fixture"})
    second_lora, second_json = save_lora(model, tmp_path, "slider", {"base_identity": "fixture"})

    assert first_lora.name == "slider.safetensors"
    assert first_json.name == "slider.json"
    assert second_lora.name == "slider_001.safetensors"
    assert second_json.name == "slider_001.json"
    state, metadata = read_lora(first_lora)
    assert metadata["architecture"] == "moss_image"
    assert metadata["alpha"] == "3.0"
    assert metadata["rank"] == "1"
    assert metadata["ss_network_dim"] == "1"
    assert metadata["ss_network_alpha"] == "3.0"
    assert metadata["ss_base_model_version"] == "moss_image"
    assert state
    manifest = json.loads(first_json.read_text(encoding="utf-8"))
    assert manifest["architecture"] == "moss_image"
    assert manifest["rank"] == 1
    assert manifest["alpha"] == 3.0
    assert manifest["base_identity"] == "fixture"
    assert manifest["lora_file"] == first_lora.name
    assert manifest["lora_sha256"] == hashlib.sha256(first_lora.read_bytes()).hexdigest()
    assert not list(tmp_path.glob("*.tmp"))


def test_safetensors_publish_is_last_commit_and_failure_cleans_pair(tmp_path, monkeypatch):
    replace = export_module.os.replace
    publication = []

    def observe(source, destination):
        destination = Path(destination)
        publication.append(destination.suffix)
        if len(publication) == 2:
            assert destination.suffix == ".safetensors"
            assert destination.with_suffix(".json").is_file()
            assert not destination.exists()
            raise OSError("last commit failed")
        return replace(source, destination)

    monkeypatch.setattr(export_module.os, "replace", observe)
    with pytest.raises(OSError, match="last commit"):
        save_lora(_model(), tmp_path, "last_commit", {})
    assert publication == [".json", ".safetensors"]
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("prefix", ["../escape", "nested/name", "nested\\name", ".", "", "  "])
def test_save_rejects_path_traversal_and_invalid_prefix(tmp_path: Path, prefix: str) -> None:
    with pytest.raises(ValueError):
        save_lora(_model(), tmp_path, prefix, {})
    assert list(tmp_path.iterdir()) == []


def test_json_failure_removes_reserved_and_temporary_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_json(*args: object, **kwargs: object) -> str:
        raise TypeError("manifest is not serializable")

    monkeypatch.setattr(export_module.json, "dumps", fail_json)

    with pytest.raises(TypeError, match="not serializable"):
        save_lora(_model(), tmp_path, "broken", {"bad": object()})

    assert list(tmp_path.iterdir()) == []


def test_final_paths_are_not_visible_until_complete_files_are_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_save_file = export_module.save_file

    def assert_unpublished(*args: object, **kwargs: object) -> None:
        assert not (tmp_path / "atomic.safetensors").exists()
        assert not (tmp_path / "atomic.json").exists()
        real_save_file(*args, **kwargs)

    monkeypatch.setattr(export_module, "save_file", assert_unpublished)

    save_lora(_model(), tmp_path, "atomic", {})

    assert (tmp_path / "atomic.safetensors").is_file()
    assert (tmp_path / "atomic.json").is_file()


def test_save_rejects_non_finite_lora_without_artifacts(tmp_path: Path) -> None:
    model = _model()
    with torch.no_grad():
        lora_parameters(model)[0].flatten()[0] = torch.inf

    with pytest.raises(ValueError, match="finite"):
        save_lora(model, tmp_path, "bad", {})

    assert list(tmp_path.iterdir()) == []


def test_read_rejects_corrupt_file(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.safetensors"
    path.write_bytes(b"not a safetensors file")

    with pytest.raises(Exception):
        read_lora(path)
