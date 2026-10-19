import os
import sys
from pathlib import Path

import torch


torch.set_num_threads(1)


def pytest_configure():
    # The explicit path is test-only; product imports use the running ComfyUI.
    root = Path(os.environ.get("MOSS_COMFYUI_ROOT", Path(__file__).resolve().parents[2] / "ComfyUI"))
    if root.is_dir():
        sys.path.insert(0, str(root))
        import comfy.options

        comfy.options.enable_args_parsing()
        argv = sys.argv
        try:
            sys.argv = [argv[0], "--cpu", "--disable-comfy-compiler"]
            import comfy.cli_args  # noqa: F401
        finally:
            sys.argv = argv
