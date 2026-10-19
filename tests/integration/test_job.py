import gc
import weakref

import pytest
import torch

from moss_slider_node.job import run_training_job
from moss_slider_node.training import TrainingConfig


def conditions():
    return {name: [[torch.randn(1, length, 4096), {}]]
            for name, length in [("target", 2), ("positive", 3), ("negative", 4)]}


def test_two_jobs_and_cancel_release_owned_model(native_checkpoint, tmp_path, monkeypatch):
    import comfy.model_management as management
    import moss_slider_node.job as job

    path, config, _ = native_checkpoint
    instances = []
    original = job.load_training_model

    def capture(*args, **kwargs):
        model, info = original(*args, **kwargs)
        instances.append(weakref.ref(model))
        return model, info

    calls = []
    monkeypatch.setattr(job, "load_training_model", capture)
    monkeypatch.setattr(management, "unload_all_models", lambda: calls.append("unload"))
    monkeypatch.setattr(management, "load_models_gpu", lambda *_a, **_k: pytest.fail("training must not register a ModelPatcher"))
    train_config = TrainingConfig(steps=2, rank=2, alpha=2, width=16, height=16, rollout_steps=2)
    before = list(management.current_loaded_models)
    source = conditions()
    snapshots = {name: value[0][0].clone() for name, value in source.items()}
    for _ in range(2):
        report = run_training_job(path, source, train_config, tmp_path / "out", "slider",
                                  device="cpu", model_config=config)
        assert report["steps_completed"] == 2
        assert report["lora_path"].is_file()
        assert report["manifest_path"].is_file()
        gc.collect()
        assert all(reference() is None for reference in instances)
        assert not management.in_training
        assert list(management.current_loaded_models) == before
    assert calls == ["unload", "unload"]
    for name in snapshots:
        torch.testing.assert_close(source[name][0][0], snapshots[name])

    def cancelled(phase, current, _total, _detail):
        if phase == "train" and current == 1:
            raise InterruptedError("cancel")

    try:
        run_training_job(path, source, train_config, tmp_path / "out", "cancelled",
                         device="cpu", model_config=config, progress=cancelled)
    except InterruptedError:
        pass
    else:
        pytest.fail("cancellation must propagate")
    gc.collect()
    assert all(reference() is None for reference in instances)
    assert not management.in_training
    assert not list((tmp_path / "out").glob("cancelled*"))


def test_invalid_conditioning_is_rejected_before_unloading(native_checkpoint, tmp_path, monkeypatch):
    import comfy.model_management as management

    path, config, _ = native_checkpoint
    monkeypatch.setattr(management, "unload_all_models", lambda: pytest.fail("invalid input must not unload models"))
    with pytest.raises(ValueError):
        run_training_job(path, {"target": []}, TrainingConfig(), tmp_path, "bad", device="cpu", model_config=config)


def test_multi_record_job_exports_source_snapshot_and_record_schedule(native_checkpoint, tmp_path):
    import json

    path, config, _ = native_checkpoint
    records = [conditions(), conditions()]
    source = {"file": "aging.yaml", "sha256": "a" * 64,
              "records": [{"target": "t", "positive": "p", "negative": "n"}] * 2}
    report = run_training_job(path, records, TrainingConfig(steps=3, rank=2, alpha=2, width=16,
                              height=16, rollout_steps=1), tmp_path, "multi", device="cpu",
                              model_config=config, prompt_source=source)
    manifest = json.loads(report["manifest_path"].read_text(encoding="utf-8"))
    assert report["record_count"] == 2
    assert [step["record_index"] for step in report["steps"]] == [0, 1, 0]
    assert manifest["prompt_source"] == source
