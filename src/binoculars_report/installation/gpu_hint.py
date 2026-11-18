"""Startup hint for when an NVIDIA GPU is present but torch is a CPU-only build (PyPI's Windows torch is CPU-only)."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from urllib.parse import parse_qs, urlsplit, urlunsplit

import torch

PACKAGE_NAME = "binoculars-report"
UV_RECEIPT_NAME = "uv-receipt.toml"


def gpu_hint(device: str) -> str | None:
    if device != "cpu" or torch.version.cuda is not None:
        return None
    gpu = nvidia_gpu_name()
    if gpu is None:
        return None
    return (
        f"Hint: {gpu} detected, but the installed torch {torch.__version__} is a CPU-only build.\n"
        "      Reinstall with a GPU build for much faster analysis:\n"
        f"        uv tool install --force --torch-backend auto {install_source()}"
    )


def nvidia_gpu_name() -> str | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    names = completed.stdout.strip().splitlines()
    return names[0].strip() if completed.returncode == 0 and names else None


def install_source() -> str:
    """The source this tool was installed from, as recorded by `uv tool install` in the tool environment."""
    fallback = f'"{PACKAGE_NAME} @ <source>"'
    receipt = Path(sys.prefix) / UV_RECEIPT_NAME
    try:
        requirements = tomllib.loads(receipt.read_text(encoding="utf-8"))["tool"]["requirements"]
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        return fallback
    entry = next((r for r in requirements if r.get("name") == PACKAGE_NAME), None)
    if entry is None:
        return fallback
    if "directory" in entry:
        return f'"{entry["directory"]}"'
    if "editable" in entry:
        return f'--editable "{entry["editable"]}"'
    if "git" in entry:
        return f'"{PACKAGE_NAME} @ {git_requirement_url(entry["git"])}"'
    if "url" in entry:
        return f'"{PACKAGE_NAME} @ {entry["url"]}"'
    if "path" in entry:
        return f'"{entry["path"]}"'
    return f'"{PACKAGE_NAME}{entry.get("specifier", "")}"'


def git_requirement_url(receipt_git: str) -> str:
    """Convert uv's receipt form `https://host/repo?rev=REF#COMMIT` into `git+https://host/repo@REF`."""
    url = urlsplit(receipt_git)
    query = parse_qs(url.query)
    ref = next((query[key][0] for key in ("tag", "branch", "rev") if key in query), None)
    base = urlunsplit((url.scheme, url.netloc, url.path, "", ""))
    subdirectory = f"#subdirectory={query['subdirectory'][0]}" if "subdirectory" in query else ""
    return f"git+{base}{f'@{ref}' if ref else ''}{subdirectory}"
