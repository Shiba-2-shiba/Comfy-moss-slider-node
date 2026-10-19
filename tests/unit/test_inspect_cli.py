import hashlib
import importlib.util
import json
import os
import struct
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INSPECT = PROJECT_ROOT / "tools" / "inspect_checkpoint.py"
RUNNER = PROJECT_ROOT / "tools" / "run_cpu_validation.py"


def run_inspect(*args):
    env = os.environ.copy()
    env.update({"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
    return subprocess.run(
        [sys.executable, str(INSPECT), *map(str, args)],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        check=False,
    )


def load_runner():
    spec = importlib.util.spec_from_file_location("moss_cpu_validation", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_inspect_cli_rejects_non_moss_and_writes_explicit_unicode_report(tmp_path):
    checkpoint = tmp_path / "別モデル.safetensors"
    raw_header = json.dumps(
        {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
        separators=(",", ":"),
    ).encode()
    checkpoint.write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + b"\0\0\0\0")
    report_path = tmp_path / "検証結果" / "inspection.json"

    result = run_inspect("--checkpoint", checkpoint, "--metadata-only", "--report", report_path)

    assert result.returncode == 2
    stdout_report = json.loads(result.stdout)
    assert stdout_report["status"] == "UNSUPPORTED"
    assert stdout_report["metadata_only"] is True
    assert "file_sha256" not in stdout_report["source"]
    assert json.loads(report_path.read_text(encoding="utf-8")) == stdout_report


def test_inspect_cli_truncated_file_is_invalid_and_hashes_only_when_requested(tmp_path):
    checkpoint = tmp_path / "truncated.safetensors"
    checkpoint.write_bytes(struct.pack("<Q", 500) + b"{}")

    metadata_only = run_inspect("--checkpoint", checkpoint, "--metadata-only")
    full_hash = run_inspect("--checkpoint", checkpoint, "--full-hash")

    assert metadata_only.returncode == 2
    assert json.loads(metadata_only.stdout)["status"] == "INVALID"
    assert "file_sha256" not in json.loads(metadata_only.stdout).get("source", {})
    hashed_report = json.loads(full_hash.stdout)
    assert hashed_report["status"] == "INVALID"
    assert hashed_report["source"]["file_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()


def test_validation_runner_counts_junit_results_without_treating_skips_as_passes(tmp_path):
    junit = tmp_path / "results.xml"
    junit.write_text(
        '<?xml version="1.0" encoding="utf-8"?>'
        '<testsuites tests="8" failures="2" errors="1" skipped="3" time="0.25">'
        '<testsuite name="suite" tests="8" failures="2" errors="1" skipped="3" time="0.25" />'
        '</testsuites>',
        encoding="utf-8",
    )

    counts = load_runner().read_junit_counts(junit)

    assert counts == {"tests": 8, "passed": 2, "failed": 2, "errors": 1, "skipped": 3}


def test_validation_runner_marks_missing_comfyui_as_blocked(tmp_path):
    stage = load_runner().integration_preflight(tmp_path / "missing-ComfyUI")

    assert stage["status"] == "BLOCKED"
    assert "ComfyUI" in stage["reason"]


def test_validation_runner_reports_missing_torch_as_blocked(monkeypatch):
    runner = load_runner()
    real_find_spec = runner.importlib.util.find_spec
    monkeypatch.setattr(
        runner.importlib.util,
        "find_spec",
        lambda name: None if name == "torch" else real_find_spec(name),
    )

    environment = runner.environment_report()

    assert environment["cpu_probe"]["status"] == "BLOCKED"
    assert "torch" in environment["cpu_probe"]["reason"]
