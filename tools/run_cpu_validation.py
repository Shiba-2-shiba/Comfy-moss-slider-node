"""Run the pre-GPU CPU validation stages and retain complete logs."""

import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfyui-root", type=Path, default=PROJECT_ROOT.parent / "ComfyUI")
    parser.add_argument("--report", type=Path, default=Path("test-results/cpu-validation.json"))
    return parser.parse_args(argv)


def atomic_json(path, value):
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_junit_counts(path):
    root = ET.parse(path).getroot()
    if root.tag == "testsuite":
        suites = [root]
    elif root.tag == "testsuites" and "tests" in root.attrib:
        suites = [root]
    else:
        suites = list(root.findall("./testsuite"))
    values = {
        key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    return {
        "tests": values["tests"],
        "passed": values["tests"] - values["failures"] - values["errors"] - values["skipped"],
        "failed": values["failures"],
        "errors": values["errors"],
        "skipped": values["skipped"],
    }


def integration_preflight(comfyui_root):
    root = Path(comfyui_root).resolve()
    if not (root / "comfy" / "options.py").is_file():
        return {"status": "BLOCKED", "reason": f"ComfyUI package was not found under {root}"}
    missing = [name for name in ("pytest", "torch", "safetensors") if importlib.util.find_spec(name) is None]
    if missing:
        return {"status": "BLOCKED", "reason": "Missing integration dependencies: " + ", ".join(missing)}
    return {"status": "PASS", "comfyui_root": str(root)}


def source_files():
    roots = [PROJECT_ROOT / "moss_slider_node", PROJECT_ROOT / "tests", PROJECT_ROOT / "tools"]
    files = [PROJECT_ROOT / "nodes.py", PROJECT_ROOT / "__init__.py"]
    for root in roots:
        if root.is_dir():
            files.extend(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)
    return sorted({path.resolve() for path in files if path.is_file()}, key=str)


def syntax_stage():
    started = time.perf_counter()
    checked = 0
    try:
        for path in source_files():
            source = path.read_text(encoding="utf-8")
            ast.parse(source, filename=str(path))
            compile(source, str(path), "exec", dont_inherit=True)
            checked += 1
    except (OSError, SyntaxError, UnicodeError) as error:
        return {"name": "syntax", "status": "FAIL", "checked_files": checked, "reason": str(error)}
    return {"name": "syntax", "status": "PASS", "checked_files": checked,
            "checks": ["ast.parse", "compile"], "duration_seconds": round(time.perf_counter() - started, 3)}


def run_command_stage(name, command, env, log_path, junit_path=None):
    started = time.perf_counter()
    if junit_path is not None:
        junit_path.unlink(missing_ok=True)
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result.stdout, encoding="utf-8")
    stage = {
        "name": name,
        "status": "PASS" if result.returncode == 0 else "FAIL",
        "returncode": result.returncode,
        "duration_seconds": round(time.perf_counter() - started, 3),
        "command": command,
        "log": str(log_path),
    }
    if junit_path is not None:
        if junit_path.is_file():
            stage["counts"] = read_junit_counts(junit_path)
            if stage["counts"]["passed"] == 0:
                stage["status"] = "BLOCKED"
                stage["reason"] = "No tests passed; an empty or entirely skipped suite cannot validate this stage"
        else:
            stage["status"] = "BLOCKED" if "ModuleNotFoundError" in result.stdout else "FAIL"
            stage["reason"] = "pytest did not produce JUnit XML"
        if result.returncode and ("ModuleNotFoundError" in result.stdout or "ImportError" in result.stdout):
            stage["status"] = "BLOCKED"
            stage["reason"] = "Test collection or execution was blocked by a missing import; see the retained log"
    return stage


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_report():
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    version_match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)
    if version_match is None:
        raise ValueError("Project version is missing from pyproject.toml")
    packages = {name: package_version(name) for name in ("torch", "pytest", "safetensors", "comfy-kitchen", "comfy-aimdo", "ruff", "mypy")}
    report = {
        "project_version": version_match.group(1),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
    }
    if importlib.util.find_spec("torch") is None:
        report["cpu_probe"] = {"status": "BLOCKED", "reason": "torch is not installed"}
        return report
    try:
        import torch

        report["cpu_probe"] = {
            "status": "PASS",
            "device": str(torch.ones(1, device="cpu").device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_initialized_at_start": torch.cuda.is_initialized(),
            "torch_runtime": torch.__version__,
        }
    except (ImportError, OSError, RuntimeError) as error:
        report["cpu_probe"] = {"status": "BLOCKED", "reason": f"torch CPU probe failed: {error}"}
    return report


def main(argv=None):
    args = parse_args(argv)
    report_path = args.report if args.report.is_absolute() else PROJECT_ROOT / args.report
    output_root = report_path.resolve().parent
    logs = output_root / "logs"
    junit = output_root / "junit"
    env = os.environ.copy()
    env.update({
        "MOSS_COMFYUI_ROOT": str(args.comfyui_root.resolve()),
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "CUDA_VISIBLE_DEVICES": "",
    })
    started = time.perf_counter()
    source_hashes = {str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in source_files()}
    try:
        git_result = subprocess.run(["git", "-C", str(args.comfyui_root.resolve()), "rev-parse", "HEAD"],
                                    capture_output=True, text=True, check=False)
        comfyui_commit = git_result.stdout.strip() if git_result.returncode == 0 else None
    except OSError:
        comfyui_commit = None
    report = {
        "status": "FAIL",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "comfyui_root": str(args.comfyui_root.resolve()),
        "comfyui_commit": comfyui_commit,
        "source_sha256": source_hashes,
        "environment": environment_report(),
        "stages": [],
        "gpu_validation": {"status": "NOT_RUN", "reason": "P8 GPU and full-model validation is outside this runner"},
    }
    report["stages"].append(syntax_stage())

    if importlib.util.find_spec("ruff") is None:
        report["stages"].append(
            {"name": "ruff", "status": "BLOCKED", "required": False, "reason": "ruff is not installed"}
        )
    else:
        report["stages"].append(run_command_stage(
            "ruff", [sys.executable, "-m", "ruff", "check", "."], env, logs / "ruff.log"))

    if importlib.util.find_spec("mypy") is None:
        report["stages"].append({"name": "mypy", "status": "BLOCKED", "required": False, "reason": "mypy is not installed"})
    else:
        report["stages"].append(run_command_stage(
            "mypy", [sys.executable, "-m", "mypy", "-p", "moss_slider_node"], env, logs / "mypy.log"))

    junit.mkdir(parents=True, exist_ok=True)
    unit_xml = junit / "unit.xml"
    report["stages"].append(run_command_stage(
        "unit", [sys.executable, "-m", "pytest", "tests/unit", "-m", "not gpu", "-q", f"--junitxml={unit_xml}"],
        env, logs / "unit.log", unit_xml))

    preflight = integration_preflight(args.comfyui_root)
    if preflight["status"] == "BLOCKED":
        report["stages"].append({"name": "integration", **preflight})
    else:
        integration_xml = junit / "integration.xml"
        report["stages"].append(run_command_stage(
            "integration",
            [sys.executable, "-m", "pytest", "tests/integration", "-m", "not gpu", "-q",
             f"--junitxml={integration_xml}"],
            env,
            logs / "integration.log",
            integration_xml,
        ))

    cpu_probe = report["environment"]["cpu_probe"]
    if cpu_probe["status"] == "PASS":
        import torch

        cuda_initialized = torch.cuda.is_initialized()
        cpu_probe["cuda_initialized_at_end"] = cuda_initialized
        report["stages"].append({
            "name": "cpu_boundary",
            "status": "FAIL" if cuda_initialized else "PASS",
            "cuda_initialized": cuda_initialized,
            "reason": "CUDA was initialized" if cuda_initialized else "CPU-only validation did not initialize CUDA",
        })
    else:
        report["stages"].append({"name": "cpu_boundary", "status": "BLOCKED", "reason": cpu_probe["reason"]})
    final_hashes = {str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in source_files()}
    report["stages"].append({"name": "source_integrity", "status": "PASS" if final_hashes == source_hashes else "FAIL",
                             "reason": "Source snapshot unchanged" if final_hashes == source_hashes else "Source changed during validation"})
    required_names = {"syntax", "unit", "integration", "cpu_boundary", "source_integrity"}
    required = [stage for stage in report["stages"] if stage["name"] in required_names]
    if any(stage["status"] == "FAIL" for stage in report["stages"]):
        report["status"] = "FAIL"
    elif any(stage["status"] == "BLOCKED" for stage in required):
        report["status"] = "BLOCKED"
    elif all(stage["status"] == "PASS" for stage in required):
        report["status"] = "PASS"
    else:
        report["status"] = "FAIL"
    report["duration_seconds"] = round(time.perf_counter() - started, 3)
    atomic_json(report_path, report)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    for stage in report["stages"]:
        counts = stage.get("counts")
        detail = f" {counts}" if counts else ""
        print(f"{stage['name']}: {stage['status']}{detail}")
    print(json.dumps({"status": report["status"], "report": str(report_path.resolve())}, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
