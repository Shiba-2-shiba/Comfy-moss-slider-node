"""V3 nodes for inspecting, training and applying MOSS-Image sliders."""

import json
import logging
from pathlib import Path

import folder_paths
from comfy_api.latest import ComfyExtension, io
from comfy.model_management import throw_exception_if_processing_interrupted
from comfy.utils import ProgressBar

from .moss_slider_node.checkpoint import inspect_checkpoint
from .moss_slider_node.inference import apply_slider
from .moss_slider_node.job import run_training_job
from .moss_slider_node.training import TrainingConfig
from .moss_slider_node.prompt_files import encode_prompt_records, list_prompt_files, load_prompt_file


LOGGER = logging.getLogger(__name__)
_CATEGORY = "MOSS Image/Slider LoRA"
_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _run_training(checkpoint_name, conditions, config, output_prefix, prompt_source=None):
    checkpoint_path = folder_paths.get_full_path_or_raise("diffusion_models", checkpoint_name)
    root = Path(folder_paths.get_output_directory())
    bar = ProgressBar(config.steps)

    def progress(phase, current, total, details):
        if phase == "train":
            bar.update_absolute(current, total)
            if current == 0:
                LOGGER.info("MOSS Slider starting training: 0/%d steps", total)
            else:
                LOGGER.info("MOSS Slider step %d/%d | %.2fs/step | elapsed %.1fs | ETA %.1fs | loss=%.6g grad=%.5g",
                            current, total, details["seconds"], details["elapsed_seconds"],
                            details["eta_seconds"], details["loss"], details["grad_norm"])
        elif phase == "load":
            if current == 0 or current == total - 1 or current % max(1, total // 10) == 0:
                LOGGER.info("MOSS Slider loading model %d/%d: %s", current, total, details)
        else:
            if phase == "prepare":
                bar.update_absolute(0, config.steps)
            LOGGER.info("MOSS Slider %s", details)

    result = run_training_job(checkpoint_path, conditions, config, root / "moss_sliders", output_prefix,
                              cancel=throw_exception_if_processing_interrupted, progress=progress,
                              prompt_source=prompt_source)
    filename = result["lora_path"].relative_to(root).as_posix()
    report = {key: str(value) if isinstance(value, Path) else value for key, value in result.items()}
    LOGGER.info("MOSS Slider saved LoRA: %s", result["lora_path"])
    LOGGER.info("MOSS Slider saved manifest: %s", result["manifest_path"])
    return io.NodeOutput(filename, json.dumps(report, ensure_ascii=False, indent=2),
                         ui={"text": [f"LoRA: {result['lora_path']}", f"Manifest: {result['manifest_path']}"]})


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
                        "Saves LoRA and manifest to ComfyUI output/moss_sliders and returns an output-relative "
                        "filename and a training report. Copy both files to a registered LoRA folder for inference.",
            inputs=[
                io.Combo.Input("checkpoint_name", options=folder_paths.get_filename_list("diffusion_models")),
                io.Conditioning.Input("target_conditioning"), io.Conditioning.Input("positive_conditioning"),
                io.Conditioning.Input("negative_conditioning"),
                io.Int.Input("steps", default=600, min=1, max=100000),
                io.Int.Input("rank", default=16, min=1, max=128),
                io.Float.Input("alpha", default=2.0, min=0.001, max=1024),
                io.Float.Input("learning_rate", default=0.0002, min=0.0000001, max=0.1, step=0.00001),
                io.Float.Input("eta", default=1.5, min=0.0, max=100),
                io.Int.Input("seed", default=0, min=0, max=0x7FFFFFFFFFFFFFFF),
                io.Int.Input("width", default=768, min=16, max=4096, step=16),
                io.Int.Input("height", default=768, min=16, max=4096, step=16),
                io.Int.Input("rollout_steps", default=16, min=1, max=100),
                io.Boolean.Input("gradient_checkpointing", default=True),
                io.Int.Input("blocks_to_swap", default=0, min=0, max=42),
                io.String.Input("output_prefix", default="moss_slider"),
                io.Combo.Input("teacher_normalization", options=["none", "neutral_l2"], default="neutral_l2",
                               optional=True, advanced=True,
                               tooltip="Match the teacher L2 norm to neutral (target when omitted). none uses the original teacher."),
                io.Conditioning.Input("neutral_conditioning", optional=True),
            ],
            outputs=[io.String.Output(display_name="lora_name"), io.String.Output(display_name="report")],
            is_output_node=True, not_idempotent=True, is_experimental=True)

    @classmethod
    def execute(cls, checkpoint_name, target_conditioning, positive_conditioning, negative_conditioning,
                steps=600, rank=16, alpha=2.0, learning_rate=0.0002, eta=1.5, seed=0,
                width=768, height=768, rollout_steps=16, gradient_checkpointing=True,
                blocks_to_swap=0, output_prefix="moss_slider", teacher_normalization="neutral_l2",
                neutral_conditioning=None):
        config = TrainingConfig(steps=steps, rank=rank, alpha=alpha, learning_rate=learning_rate, eta=eta,
                                seed=seed, width=width, height=height, rollout_steps=rollout_steps,
                                gradient_checkpointing=gradient_checkpointing, blocks_to_swap=blocks_to_swap,
                                teacher_normalization=teacher_normalization)
        conditions = {"target": target_conditioning, "positive": positive_conditioning, "negative": negative_conditioning}
        if neutral_conditioning is not None:
            conditions["neutral"] = neutral_conditioning
        return _run_training(checkpoint_name, conditions, config, output_prefix)


class MossSliderTrainFromPrompts(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        schema = MossSliderTrain.define_schema()
        schema.node_id = "MossSliderTrainFromPrompts"
        schema.display_name = "MOSS Slider Train from Prompts"
        schema.description = ("Select a YAML from this repository's prompts folder and connect a native MOSS CLIP. "
                              "All target/positive/negative records are used in order, one record per optimizer step. "
                              "Optional neutral supplies the L2 normalization reference; otherwise target is used. "
                              "Steps is the total update count. Anchor preservation loss is not supported.")
        schema.inputs = [item for item in schema.inputs if item.id not in
                         {"target_conditioning", "positive_conditioning", "negative_conditioning", "neutral_conditioning"}]
        schema.inputs[1:1] = [io.Clip.Input("clip"),
                             io.Combo.Input("prompt_yaml", options=list_prompt_files(_PROMPTS_DIR),
                                            default="deaging_slider_fullbody.yaml",
                                            tooltip="YAML files in this repository's prompts folder. All records are used.")]
        return schema

    @classmethod
    def fingerprint_inputs(cls, prompt_yaml, **kwargs):
        return load_prompt_file(_PROMPTS_DIR, prompt_yaml).sha256

    @classmethod
    def execute(cls, checkpoint_name, clip, prompt_yaml, output_prefix="moss_slider", **settings):
        config = TrainingConfig(**settings)
        config.validate()
        snapshot = load_prompt_file(_PROMPTS_DIR, prompt_yaml)
        total_roles = sum(len(record) for record in snapshot.records)
        bar = ProgressBar(total_roles)
        LOGGER.info("MOSS Slider encoding prompts: 0/%d records (%s)", len(snapshot.records), snapshot.name)

        def encoding_progress(current, total):
            bar.update_absolute(current, total)
            LOGGER.info("MOSS Slider encoding prompt roles: %d/%d", current, total)

        conditions = encode_prompt_records(clip, snapshot.records, cancel=throw_exception_if_processing_interrupted,
                                            progress=encoding_progress)
        source = {"file": snapshot.name, "sha256": snapshot.sha256, "records": snapshot.records}
        output = _run_training(checkpoint_name, conditions, config, output_prefix, prompt_source=source)
        return io.NodeOutput(*output.result, ui={"text": [*output.ui["text"],
                             f"Prompts: {snapshot.name} ({len(snapshot.records)} records)"]})


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
        return [MossSliderInspect, MossSliderTrain, MossSliderTrainFromPrompts, MossSliderApply]
