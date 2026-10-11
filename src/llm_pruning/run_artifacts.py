"""Exclusive run directories, input fingerprints and explicit run lifecycle metadata."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


def file_fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def git_metadata(repo_root: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        # Trust only this repo for this command; never change the user's git config.
        result = subprocess.run(
            ["git", "-c", f"safe.directory={repo_root.resolve().as_posix()}", *args],
            cwd=repo_root, capture_output=True, text=True, check=True, timeout=10,
        )
        return result.stdout.strip()

    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def environment_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for package in ("torch", "transformers", "tokenizers", "numpy"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunArtifacts:
    """Reserve a fresh directory before loading a model, and record failures too."""

    def __init__(self, output_dir: Path, repo_root: Path, config: dict[str, Any], **metadata: Any):
        self.output_dir = output_dir
        self.manifest = {
            **metadata,
            "schema_version": 1,
            "status": "running",
            "started_at": utc_now(),
            "git": git_metadata(repo_root),
            "versions": environment_versions(),
            "config": config,
            "seed": config["seed"],
        }

    def __enter__(self) -> "RunArtifacts":
        # exist_ok=False is deliberate, including for an existing empty directory.
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.write_json("config.json", self.manifest["config"])
        self.write_json("manifest.json", self.manifest)
        self.write_summary(["# Experiment run", "", "Status: running."])
        return self

    def write_json(self, filename: str, data: Any) -> None:
        (self.output_dir / filename).write_text(
            json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8",
        )

    def write_summary(self, lines: list[str]) -> None:
        if any("\n" in line or "\r" in line for line in lines) or len(lines) > 40:
            raise ValueError("summary.md must contain at most 40 single lines.")
        (self.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def complete(self, summary: list[str], **metadata: Any) -> None:
        self.write_summary(summary)
        self.manifest.update(metadata, status="complete", finished_at=utc_now())
        self.write_json("manifest.json", self.manifest)

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc is not None or self.manifest["status"] != "complete":
            self.manifest.update(status="failed", finished_at=utc_now())
            self.manifest["error"] = (
                {"type": exc_type.__name__, "message": str(exc)} if exc is not None
                else {"type": "IncompleteRun", "message": "Run exited without completing."}
            )
            self.write_json("manifest.json", self.manifest)
            self.write_summary([
                "# Experiment run", "", "Status: failed.",
                "See manifest.json for the error. Use a fresh directory when retrying.",
            ])
        return False
