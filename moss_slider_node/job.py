"""A single training job owns all models and restores ComfyUI process state."""

import gc
import importlib.metadata
import platform
import threading
from dataclasses import asdict
from pathlib import Path

import torch

from .checkpoint import file_sha256, inspect_checkpoint, load_training_model
from .export import save_lora
from .lora import inject_lora
from .model_adapter import MossTrainingAdapter, normalize_conditioning, training_context
from .training import train_slider


_JOB_LOCK = threading.Lock()


def run_training_job(checkpoint_path, conditions, config, output_dir, prefix, *, device=None,
                     progress=None, cancel=None, model_config=None):
    import comfy.model_management as management

    if not _JOB_LOCK.acquire(blocking=False):
        raise RuntimeError("A MOSS slider training job is already running")
    model = adapter = normalized = None
    allocated = False

    def check_cancel():
        if cancel:
            cancel()

    def emit(phase, current, total, detail):
        check_cancel()
        if progress:
            progress(phase, current, total, detail)

    try:
        with torch.inference_mode(False), torch.enable_grad():
            config.validate()
            if management.in_training:
                raise RuntimeError("Another ComfyUI training context is active")
            if set(conditions) != {"target", "positive", "negative"}:
                raise ValueError("Exactly target, positive and negative conditioning roles are required")
            normalized = {name: normalize_conditioning(value) for name, value in conditions.items()}
            inspection = inspect_checkpoint(checkpoint_path, config=model_config)
            if inspection["status"] != "SUPPORTED":
                raise ValueError(inspection["reason"])
            check_cancel()
            emit("prepare", 0, 1, "Unloading inference models")
            management.unload_all_models()
            management.soft_empty_cache()
            allocated = True
            training_device = torch.device(device) if device is not None else management.get_torch_device()
            compute_dtype = torch.float32 if training_device.type == "cpu" else torch.bfloat16
            digest = file_sha256(checkpoint_path, check_cancel)
            model, model_info = load_training_model(
                checkpoint_path, device=training_device, compute_dtype=compute_dtype,
                blocks_to_swap=config.blocks_to_swap, config=model_config, cancel=check_cancel,
                progress=lambda step, total, message: emit("load", step, total, message))
            # Parameter initialization has its own seed, independent of upstream graph RNG use.
            rng_devices = []
            if training_device.type == "cuda":
                rng_devices = [training_device.index if training_device.index is not None else torch.cuda.current_device()]
            with torch.random.fork_rng(devices=rng_devices):
                torch.random.default_generator.manual_seed(config.seed)
                if rng_devices:
                    with torch.cuda.device(training_device):
                        torch.cuda.manual_seed(config.seed)
                targets = inject_lora(model, rank=config.rank, alpha=config.alpha, device=training_device)
            adapter = MossTrainingAdapter(model, device=training_device, dtype=compute_dtype,
                                          gradient_checkpointing=config.gradient_checkpointing, cancel=check_cancel)
            with training_context(model):
                report = train_slider(
                    adapter, normalized, config, cancel=check_cancel,
                    progress=lambda step, total, detail: emit("train", step, total, detail))
            check_cancel()
            versions = {name: importlib.metadata.version(name) for name in ("torch", "safetensors", "comfy-kitchen", "comfy-aimdo")}
            manifest = {"source": {**model_info["source"], "sha256": digest}, "config": asdict(config),
                        "model": model_info, "training": report, "versions": versions,
                        "platform": platform.platform(), "device": str(training_device),
                        "adapter_targets": targets, "internal_adapter_count": len(targets),
                        "qkv_export": "shared_down_matrix_and_three_output_slices",
                        "attention_backend": "pytorch", "real_checkpoint_gpu_validated": False}
            emit("save", 0, 1, "Saving LoRA and manifest")
            lora_path, manifest_path = save_lora(model, output_dir, prefix, manifest)
            return {**report, "steps_completed": len(report["steps"]),
                    "lora_path": Path(lora_path), "manifest_path": Path(manifest_path),
                    "source_sha256": digest, "internal_adapter_count": len(targets)}
    finally:
        adapter = model = normalized = None
        gc.collect()
        try:
            if allocated:
                management.soft_empty_cache()
        finally:
            _JOB_LOCK.release()
