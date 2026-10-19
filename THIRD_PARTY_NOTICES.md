# Third-party references

The frozen ConvRot input-gradient formulation, regular H4 Hadamard convention, and Slider LoRA workflow are informed by `Confyui-krea2-slider-node`, commit `614d2a8e7f424b047b92cd94201b61a817819eb1`, and `sd-scripts`, commit `04ca541746b7667b27eab230db067579f682895e`.

The Krea2 reference credits Musubi Tuner `4e7c7149249e7715e9168920feb4c420423abba7` and Comfy Kitchen-derived ConvRot work under Apache-2.0: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. Copyright (c) 2025 Comfy Org. The Apache-2.0 license is retained in `licenses/Apache-2.0.txt`. This project uses eager PyTorch arithmetic and does not vendor their Triton kernels.

MOSS-Image by OpenMOSS, commit `c1355e6b1986dfe8fba3c0455783a05899212342`, supplies the architecture/conditioning reference under Apache-2.0. Model weights have their own terms and are not distributed here.

The native MOSS model, ComfyUI APIs, LoRA mapping and bypass implementation are imported from the user's ComfyUI installation. Development reference: ComfyUI `8f04fd6ecf4a548937971e840af2a985529fab76`, licensed under GPL-3.0; its license is retained in `licenses/ComfyUI-GPL-3.0.txt`. No ComfyUI model source is vendored into this package.

The slider objective follows the Concept Sliders approach (Gandikota et al., ECCV 2024), adapted to MOSS raw flow velocity. No model weights or third-party datasets are bundled.
