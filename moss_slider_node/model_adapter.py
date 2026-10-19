"""Native MOSS integration; ComfyUI is imported only when native execution is requested."""

from contextlib import ExitStack, contextmanager
from dataclasses import asdict

import torch
from torch.utils.checkpoint import checkpoint


def normalize_conditioning(conditioning):
    if not isinstance(conditioning, (list, tuple)) or len(conditioning) != 1:
        raise ValueError("MOSS slider requires exactly one conditioning entry per role")
    entry = conditioning[0]
    if not isinstance(entry, (list, tuple)) or len(entry) != 2:
        raise ValueError("Invalid CONDITIONING entry")
    tensor, metadata = entry
    if not isinstance(metadata, dict) or set(metadata) - {"attention_mask", "pooled_output"}:
        raise ValueError("Unsupported conditioning metadata; schedules, areas, controls and hooks are not supported")
    if (not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point() or tensor.ndim != 3
            or tensor.shape[0] != 1 or tensor.shape[2] != 4096 or not 1 <= tensor.shape[1] <= 4096):
        raise ValueError("MOSS conditioning must be a floating [1,L,4096] tensor, 1<=L<=4096")
    length = tensor.shape[1]
    mask = metadata.get("attention_mask")
    if mask is not None:
        if not isinstance(mask, torch.Tensor) or mask.shape != (1, length):
            raise ValueError("attention_mask must have shape [1,L]")
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("attention_mask must contain only zero or one")
        valid = int(mask.sum())
        if not valid or not torch.equal(mask.bool().cpu(), (torch.arange(length) < valid).unsqueeze(0)):
            raise ValueError("Only a nonempty right-padded attention_mask is supported")
        length = valid
    if not torch.isfinite(tensor).all():
        raise ValueError("MOSS conditioning contains NaN or infinity")
    with torch.inference_mode(False), torch.no_grad():
        return tensor[:, :length].detach().to(device="cpu").clone()


def create_native_model(config, *, device="meta", dtype=torch.bfloat16):
    import comfy.ops
    from comfy.ldm.moss_image.model import MOSSImage

    return MOSSImage(**asdict(config), fused_qkv=True, device=device, dtype=dtype,
                     operations=comfy.ops.manual_cast)


@contextmanager
def training_context(model):
    import comfy.model_management as management
    import comfy.model_prefetch as prefetch
    from comfy.ldm.modules.attention import ComfyAttention, attention_pytorch

    if management.in_training:
        raise RuntimeError("Another ComfyUI training context is active")
    old_compiler = prefetch.args.disable_comfy_compiler
    states = [(m, m.function, m.method) for m in model.modules() if isinstance(m, ComfyAttention)]
    try:
        with torch.inference_mode(False), torch.enable_grad():
            management.in_training = True
            prefetch.args.disable_comfy_compiler = True
            for module, _, _ in states:
                module.function = attention_pytorch
                module.method = "pytorch"
            yield
    finally:
        for module, function, method in states:
            module.function, module.method = function, method
        prefetch.args.disable_comfy_compiler = old_compiler
        management.in_training = False


class MossTrainingAdapter:
    def __init__(self, native_model, *, device="cpu", dtype=torch.bfloat16,
                 gradient_checkpointing=True, cancel=None):
        self.native_model = native_model
        self.device = torch.device(device)
        self.dtype = dtype
        self.latent_channels = native_model.latent_channels
        self.gradient_checkpointing = gradient_checkpointing
        self.cancel = cancel

    def _block_patch(self, block):
        def execute(args, _extra):
            if self.cancel:
                self.cancel()
            pe, options = args["pe"], args["transformer_options"].copy()

            def call(hidden, *mod):
                if self.cancel:
                    self.cancel()
                return block(hidden, mod, pe, options)

            if self.gradient_checkpointing and torch.is_grad_enabled():
                output = checkpoint(call, args["img"], *args["mod"], use_reentrant=False)
            else:
                output = call(args["img"], *args["mod"])
            return {"img": output}

        return execute

    def predict_velocity(self, x, sigma, conditioning):
        import comfy.model_management as management

        if not management.in_training:
            raise RuntimeError("predict_velocity requires training_context")
        if self.cancel:
            self.cancel()
        if x.ndim != 4 or x.shape[1] != self.latent_channels or x.shape[0] != 1:
            raise ValueError("MOSS latent must have shape [1,128,H,W]")
        sigma = torch.as_tensor(sigma, device=self.device, dtype=torch.float32).reshape(-1)
        if sigma.numel() != 1 or not torch.isfinite(sigma).all() or not ((sigma >= 0) & (sigma <= 1)).all():
            raise ValueError("sigma must be a finite scalar in [0,1]")
        options = {"prefetch_dynamic_vbars": False, "patches_replace": {"dit": {
            ("single_block", i): self._block_patch(block) for i, block in enumerate(self.native_model.layers)
        }}}
        with ExitStack() as hooks:
            if self.cancel:
                for block in self.native_model.text_embedder.blocks:
                    handle = block.register_forward_pre_hook(lambda _m, _a: self.cancel())
                    hooks.callback(handle.remove)
            return self.native_model(x.to(device=self.device, dtype=self.dtype), sigma,
                                     conditioning.to(device=self.device, dtype=self.dtype),
                                     transformer_options=options)
