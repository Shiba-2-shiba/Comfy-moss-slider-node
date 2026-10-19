"""ComfyUI V3 extension entrypoint; core utilities also work outside ComfyUI."""


async def comfy_entrypoint():
    from .nodes import MossSliderExtension

    return MossSliderExtension()
