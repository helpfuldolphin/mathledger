"""
Provenance & Manifest System for RFL Experiments.

Handles generation of cryptographically bound experiment manifests.
"""

import json
import os
import subprocess
import platform
import getpass
from pathlib import Path
from typing import Any, Dict, Optional

from .config import RFLConfig
from .experiment import ExperimentResult

class ManifestBuilder:
    """
    Builds and persists cryptographically bound experiment manifests.
    """

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self._git_commit = self._get_git_commit()
        self._git_branch = self._get_git_branch()
        self._user = getpass.getuser()
        self._machine = platform.node()

    def _get_git_commit(self) -> str:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"], 
                cwd=self.project_root, 
                text=True,
                stderr=subprocess.DEVNULL
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            return "unknown"

    def _get_git_branch(self) -> str:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], 
                cwd=self.project_root, 
                text=True,
                stderr=subprocess.DEVNULL
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            return "unknown"

    def build(
        self,
        run_index: int,
        config: RFLConfig,
        result: ExperimentResult,
        effective_seed: int,
        command_str: str = "python rfl/runner.py"
    ) -> Dict[str, Any]:
        """
        Construct the manifest dictionary.
        """
        manifest = {
            "manifest_version": "1.0",
            "experiment_id": config.experiment_id,
            "run_index": run_index,
            "timestamp_utc": result.end_time,
            "provenance": {
                "git_commit": self._git_commit,
                "git_branch": self._git_branch,
                "user": self._user,
                "machine": self._machine
            },
            "configuration": {
                "snapshot": config.to_dict()
            },
            "execution": {
                "command": command_str,
                "effective_seed": effective_seed,
                "python_version": platform.python_version(),
            },
            # In a future iteration, we can link to specific log files if we split them per run
            "artifacts": {
                "logs": getattr(result, "logs", []), 
                "data": [],
                "figures": getattr(result, "figures", [])
            },
            "results": result.to_dict()
        }
        return manifest

    def build_suite_manifest(
        self,
        config: RFLConfig,
        execution_summary: Dict[str, Any],
        log_paths: list[str],
        figure_paths: list[str],
        start_time: str,
        end_time: str,
        command_str: str = "python rfl/runner.py"
    ) -> Dict[str, Any]:
        """
        Construct the aggregate manifest for the entire experiment suite.
        """
        manifest = {
            "manifest_version": "1.0",
            "experiment_id": config.experiment_id,
            "timestamp_utc": end_time,
            "start_timestamp_utc": start_time,
            "provenance": {
                "git_commit": self._git_commit,
                "git_branch": self._git_branch,
                "user": self._user,
                "machine": self._machine
            },
            "configuration": {
                "config_path": f"config/rfl/{config.experiment_id}.json", # Virtual path
                "snapshot": config.to_dict()
            },
            "execution": {
                "command": command_str,
                "effective_seed": config.random_seed,
                "python_version": platform.python_version(),
                "dependencies": {} # Could be populated via pip freeze if needed
            },
            "artifacts": {
                "logs": log_paths,
                "figures": figure_paths
            },
            "results": execution_summary
        }
        return manifest

    def save(self, manifest: Dict[str, Any], output_path: Path) -> None:
        """
        Write manifest to disk.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
