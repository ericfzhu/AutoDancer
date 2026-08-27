"""Reproducibility metadata and content hashing for experiment runs."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

from autodancer.live.protocol import SCHEMA_VERSION, SUPPORTED_GAME_VERSION, SUPPORTED_STEAM_BUILD


def sha256_file(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments], check=True, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _git_clean(*arguments: str) -> bool | None:
    """Return Git's quiet cleanliness result without invoking status."""
    try:
        result = subprocess.run(
            ["git", *arguments], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode not in (0, 1):
        return None
    return result.returncode == 0


def git_identity() -> dict[str, Any]:
    worktree_clean = _git_clean("diff", "--quiet")
    index_clean = _git_clean("diff", "--cached", "--quiet")
    untracked = _git("ls-files", "--others", "--exclude-standard")
    dirty = (
        None
        if worktree_clean is None or index_clean is None or untracked is None
        else not worktree_clean or not index_clean or bool(untracked)
    )
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("branch", "--show-current"),
        "dirty": dirty,
    }


def runtime_identity(device: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_logical": psutil.cpu_count(logical=True),
        "cpu_physical": psutil.cpu_count(logical=False),
        "memory_bytes": psutil.virtual_memory().total,
        "requested_device": device,
    }
    try:
        import torch

        result["torch"] = torch.__version__
        result["cuda_available"] = torch.cuda.is_available()
        result["cuda_version"] = torch.version.cuda
        result["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        result["torch"] = None
    return result


def controller_identity(
    game_dir: Path, mod_dir: Path, qualification: Path | None
) -> dict[str, Any]:
    native_candidates = sorted(mod_dir.rglob("*.dll")) if mod_dir.is_dir() else []
    lua_candidates = sorted(mod_dir.rglob("*.lua")) if mod_dir.is_dir() else []
    executable_candidates = [
        game_dir / "NecroDancer.exe",
        game_dir / "NecroDancer64.exe",
        game_dir.parent / "NecroDancer.exe",
    ]
    game_executable = next((path for path in executable_candidates if path.is_file()), None)
    qualification_summary = None
    if qualification is not None and qualification.is_file():
        try:
            report = json.loads(qualification.read_text(encoding="utf-8"))
            qualification_summary = {
                "passed": report.get("passed"),
                "completed_at": report.get("completed_at"),
                "configuration": report.get("configuration"),
                "criteria": report.get("criteria"),
            }
        except (OSError, json.JSONDecodeError):
            qualification_summary = {"passed": False, "malformed": True}
    return {
        "protocol_schema": SCHEMA_VERSION,
        "game_version": SUPPORTED_GAME_VERSION,
        "steam_build": SUPPORTED_STEAM_BUILD,
        "game_dir": str(game_dir.resolve()),
        "game_executable": None if game_executable is None else str(game_executable.resolve()),
        "game_executable_sha256": sha256_file(game_executable),
        "mod_dir": str(mod_dir.resolve()),
        "native_bridge_hashes": {
            str(path.relative_to(mod_dir)): sha256_file(path) for path in native_candidates
        },
        "lua_hashes": {
            str(path.relative_to(mod_dir)): sha256_file(path) for path in lua_candidates
        },
        "qualification_report": None if qualification is None else str(qualification.resolve()),
        "qualification_sha256": sha256_file(qualification),
        "qualification": qualification_summary,
    }


def environment_snapshot() -> dict[str, str]:
    allowed = ("CUDA_VISIBLE_DEVICES", "AUTODANCER_MLFLOW_URI", "PYTHONHASHSEED")
    return {name: os.environ[name] for name in allowed if name in os.environ}
