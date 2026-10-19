"""Inspect a packed MOSS checkpoint without loading its tensor payloads."""

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from moss_slider_node.checkpoint import file_sha256, inspect_checkpoint  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Inspect only the safetensors header and small configuration tensors (the default).",
    )
    parser.add_argument("--full-hash", action="store_true", help="Explicitly stream the full file to calculate SHA256.")
    parser.add_argument("--report", type=Path, help="Also write the JSON result to this path.")
    return parser.parse_args(argv)


def write_json(path, report):
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    args = parse_args(argv)
    checkpoint = args.checkpoint.resolve()
    report = inspect_checkpoint(checkpoint)
    report["metadata_only"] = True
    if args.full_hash:
        source = report.setdefault("source", {})
        source.setdefault("path", str(checkpoint))
        try:
            source["file_sha256"] = file_sha256(checkpoint)
        except OSError as error:
            report = {**report, "status": "INVALID", "reason": str(error), "source": source}
    if args.report:
        write_json(args.report, report)
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(output)
    return 0 if report.get("status") == "SUPPORTED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
