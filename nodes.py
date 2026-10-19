"""V3 nodes for inspecting, training and applying MOSS-Image sliders."""

import json
from pathlib import Path

import folder_paths
from comfy_api.latest import ComfyExtension, io
from comfy.model_management import throw_exception_if_processing_interrupted
from comfy.utils import ProgressBar

from .moss_slider_node.checkpoint import inspect_checkpoint
from .moss_slider_node.inference import apply_slider
from .moss_slider_node.job import run_training_job
from .moss_slider_node.training import TrainingConfig


_CATEGORY = "MOSS Image/Slider LoRA"


class MossSliderInspect(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MossSliderInspect", display_name="MOSS Slider Inspect INT8", category=_CATEGORY,
            description="Inspect packed MOSS ConvRot INT8 metadata without loading the model weights.",
            inputs=[io.Combo.Input("checkpoint_name", options=folder_paths.get_filename_list("diffusion_models"))],
            outputs=[io.String.Output(display_name="status"), io.String.Output(display_name="report")],
            is_experimental=True)

    @classmethod
    def execute(cls, checkpoint_name):
        path = folder_paths.get_full_path_or_raise("diffusion_models", checkpoint_name)
        report = inspect_checkpoint(path)
        return io.NodeOutput(report["status"], json.dumps(report, ensure_ascii=False, indent=2))


class MossSliderTrain(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MossSliderTrain", display_name="MOSS Slider Train INT8", category=_CATEGORY,
            description="Train a single slider on a frozen ConvRot INT8 MOSS base. Each role requires one batch-1 "
                        "MOSS conditioning entry; scheduled, area, control and hook conditioning are unsupported. "
                        "Outputs a LoRA filename for MOSS Slider Apply and a training report.",
            inputs=[
                io.Combo.Input("checkpoint_name", options=folder_paths.get_filename_list("diffusion_models")),
                io.Conditioning.Input("target_conditioning"), io.Conditioning.Input("positive_conditioning"),
                io.Conditioning.Input("negative_conditioning"),
                io.Int.Input("steps", default=100, min=1, max=100000),
                io.Int.Input("rank", default=8, min=1, max=128),
                io.Float.Input("alpha", default=8.0, min=0.001, max=1024),
                io.Float.Input("learning_rate", default=0.0001, min=0.0000001, max=0.1, step=0.00001),
                io.Float.Input("eta", default=1.0, min=0.0, max=100),
                io.Int.Input("seed", default=0, min=0, max=0x7FFFFFFFFFFFFFFF),
                io.Int.Input("width", default=512, min=16, max=4096, step=16),
                io.Int.Input("height", default=512, min=16, max=4096, step=16),
                io.Int.Input("rollout_steps", default=8, min=1, max=100),
                io.Boolean.Input("gradient_checkpointing", default=True),
                io.Int.Input("blocks_to_swap", default=0, min=0, max=42),
                io.String.Input("output_prefix", default="moss_slider"),
            ],
            outputs=[io.String.Output(display_name="lora_name"), io.String.Output(display_name="report")],
            is_output_node=True, not_idempotent=True, is_experimental=True)

    @classmethod
    def execute(cls, checkpoint_name, target_conditioning, positive_conditioning, negative_conditioning,
                steps=100, rank=8, alpha=8.0, learning_rate=0.0001, eta=1.0, seed=0,
                width=512, height=512, rollout_steps=8, gradient_checkpointing=True,
                blocks_to_swap=0, output_prefix="moss_slider"):
        checkpoint_path = folder_paths.get_full_path_or_raise("diffusion_models", checkpoint_name)
        roots = folder_paths.get_folder_paths("loras")
        if not roots:
            raise ValueError("No ComfyUI LoRA output directory is registered")
        root = Path(roots[0])
        output_dir = root / "moss_sliders"
        config = TrainingConfig(steps=steps, rank=rank, alpha=alpha, learning_rate=learning_rate, eta=eta,
                                seed=seed, width=width, height=height, rollout_steps=rollout_steps,
                                gradient_checkpointing=gradient_checkpointing, blocks_to_swap=blocks_to_swap)
        bar = ProgressBar(steps)

        def progress(_phase, current, total, _details):
            bar.update_absolute(current, total)

        result = run_training_job(
            checkpoint_path, {"target": target_conditioning, "positive": positive_conditioning,
                              "negative": negative_conditioning}, config, output_dir, output_prefix,
            cancel=throw_exception_if_processing_interrupted, progress=progress)
        filename = result["lora_path"].relative_to(root).as_posix()
        report = {key: str(value) if isinstance(value, Path) else value for key, value in result.items()}
        return io.NodeOutput(filename, json.dumps(report, ensure_ascii=False, indent=2))


class MossSliderApply(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MossSliderApply", display_name="MOSS Slider Apply INT8", category=_CATEGORY,
            description="Apply this package's slider without merging INT8 base weights. Requires its JSON manifest "
                        "and an unpatched native Load Diffusion Model output using the same checkpoint.",
            inputs=[io.Model.Input("model"),
                    io.Combo.Input("lora_name", options=folder_paths.get_filename_list("loras")),
                    io.Float.Input("strength", default=1.0, min=-10.0, max=10.0, step=0.1)],
            outputs=[io.Model.Output(display_name="model")], is_experimental=True)

    @classmethod
    def execute(cls, model, lora_name, strength=1.0):
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        return io.NodeOutput(apply_slider(model, path, strength))


class MossSliderExtension(ComfyExtension):
    async def get_node_list(self):
        return [MossSliderInspect, MossSliderTrain, MossSliderApply]
