"""
PHASE II Experiment Manifest Generator.

This module provides the ManifestGenerator class for creating reproducible
experiment manifests that capture configuration, provenance, and artifact
information for Phase II experiments.

RFL uplift detection is not part of manifest validation.
"""

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class GitInfo:
    """
    Git repository provenance information.

    Attributes:
        commit_sha: The current HEAD commit SHA.
        branch: The current branch name.
        is_dirty: Whether there are uncommitted changes.
        diff_sha256: SHA256 of uncommitted diff (if dirty).
    """

    commit_sha: str
    branch: str
    is_dirty: bool
    diff_sha256: Optional[str] = None


@dataclass
class SystemInfo:
    """
    System environment information.

    Attributes:
        hostname: The machine hostname.
        os: Operating system platform string.
        python_version: Python version string.
        env_hash: SHA256 hash of pip freeze output.
    """

    hostname: str
    os: str
    python_version: str
    env_hash: str


@dataclass
class LogArtifact:
    """
    Log file artifact metadata.

    Attributes:
        path: Relative or absolute path to the log file.
        sha256: SHA256 hash of the file contents.
        type: Type/category of the log (e.g., 'telemetry', 'debug').
    """

    path: str
    sha256: str
    type: str


@dataclass
class FigureArtifact:
    """
    Figure/image artifact metadata.

    Attributes:
        path: Relative or absolute path to the figure file.
        sha256: SHA256 hash of the file contents.
        description: Human-readable description of the figure.
    """

    path: str
    sha256: str
    description: str

class ManifestGenerator:
    """
    Generate experiment manifests for Phase II experiments.

    This class captures all information needed to reproduce an experiment,
    including configuration, git provenance, system information, and
    artifact references.

    RFL uplift detection is not part of manifest validation.

    Attributes:
        meta: Manifest version and type metadata.
        experiment: Experiment identification and description.
        config_path: Path to the experiment configuration file.
        mdap_seed: Master seed for deterministic execution.
        hermetic: Whether the experiment runs in hermetic isolation.
        no_network: Whether network access is disabled.
        provenance: Git and system provenance information.
        configuration: Experiment configuration details.
        artifacts: Log and figure artifact references.
    """

    def __init__(
        self,
        experiment_id: str,
        experiment_type: str,
        description: str,
        config_path: Path,
        mdap_seed: str,
        is_hermetic: bool = True,
        is_no_network: bool = True,
    ) -> None:
        """
        Initialize a new ManifestGenerator.

        Args:
            experiment_id: Unique identifier for this experiment run.
            experiment_type: Category of experiment (e.g., 'uplift_u2').
            description: Human-readable experiment description.
            config_path: Path to the YAML configuration file.
            mdap_seed: Master deterministic seed string.
            is_hermetic: Whether experiment runs without external deps.
            is_no_network: Whether network access is blocked.

        Side-effects:
            Captures current git state and system info on initialization.
        """
        self.meta: Dict[str, str] = {
            "version": "1.0.0",
            "type": "mathledger_experiment_manifest"
        }
        self.experiment: Dict[str, str] = {
            "id": experiment_id,
            "type": experiment_type,
            "timestamp_utc": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            "description": description
        }
        self.config_path = config_path
        self.mdap_seed = mdap_seed
        self.hermetic = is_hermetic
        self.no_network = is_no_network

        self.provenance: Dict[str, Any] = self._get_provenance()
        self.configuration: Dict[str, Any] = self._get_configuration()
        self.artifacts: Dict[str, List[Dict[str, str]]] = {"logs": [], "figures": []}

    def _get_git_info(self) -> Dict[str, Any]:
        """
        Retrieve git repository provenance information.

        Returns:
            Dictionary with commit_sha, branch, is_dirty, and diff_sha256.
            Returns placeholder values if git is unavailable.

        Side-effects:
            Executes git commands as subprocesses.
        """
        try:
            commit_sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            ).decode().strip()
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
            ).decode().strip()

            # Check if dirty
            status_out = subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
            ).decode().strip()
            is_dirty = bool(status_out)

            diff_sha256: Optional[str] = None
            if is_dirty:
                # Get diff hash
                diff_bytes = subprocess.check_output(
                    ["git", "diff", "HEAD"], stderr=subprocess.DEVNULL
                )
                diff_sha256 = hashlib.sha256(diff_bytes).hexdigest()

            return {
                "commit_sha": commit_sha,
                "branch": branch,
                "is_dirty": is_dirty,
                "diff_sha256": diff_sha256
            }
        except subprocess.CalledProcessError:
            return {
                "commit_sha": "unknown",
                "branch": "unknown",
                "is_dirty": False,
                "diff_sha256": None
            }
        except FileNotFoundError:
            return {
                "commit_sha": "git_not_found",
                "branch": "unknown",
                "is_dirty": False,
                "diff_sha256": None
            }

    def _get_system_info(self) -> Dict[str, str]:
        """
        Retrieve system environment information.

        Returns:
            Dictionary with hostname, os, python_version, and env_hash.

        Side-effects:
            Executes pip freeze as a subprocess.
        """
        import platform
        import socket

        env_hash = "unknown"
        try:
            # Use current sys.executable to get pip freeze of the active environment
            env_bytes = subprocess.check_output(
                [sys.executable, "-m", "pip", "freeze"], stderr=subprocess.DEVNULL
            )
            env_hash = hashlib.sha256(env_bytes).hexdigest()
        except Exception:
            pass

        return {
            "hostname": socket.gethostname(),
            "os": platform.platform(),
            "python_version": platform.python_version(),
            "env_hash": env_hash
        }

    def _get_provenance(self) -> Dict[str, Any]:
        """
        Assemble complete provenance information.

        Returns:
            Dictionary containing git and system provenance.
        """
        return {
            "git": self._get_git_info(),
            "system": self._get_system_info()
        }

    def _hash_file(self, path: Path) -> str:
        """
        Compute SHA256 hash of a file.

        Args:
            path: Path to the file to hash.

        Returns:
            Hexadecimal SHA256 digest, or "missing" if file doesn't exist.
        """
        if not path.exists():
            return "missing"
        sha256_hash = hashlib.sha256()
        with open(path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _get_configuration(self) -> Dict[str, Any]:
        """
        Build configuration section of the manifest.

        Returns:
            Dictionary with config file info, parameters, and determinism settings.
        """
        return {
            "config_file": {
                "path": str(self.config_path),
                "sha256": self._hash_file(self.config_path)
            },
            "parameters": {},
            "determinism": {
                "mdap_seed": self.mdap_seed,
                "sealing_conditions": {
                    "hermetic": self.hermetic,
                    "no_network": self.no_network
                }
            },
            "specialized": {}
        }

    def add_log(self, path: Path, log_type: str) -> None:
        """
        Add a log file artifact to the manifest.

        Args:
            path: Path to the log file (absolute or relative).
            log_type: Category of log (e.g., 'telemetry', 'debug').

        Side-effects:
            Appends to self.artifacts["logs"].
            Reads file to compute hash.
        """
        rel_path = path
        if path.is_absolute():
            # Try to make relative to cwd if possible for cleaner manifest
            try:
                rel_path = path.relative_to(Path.cwd())
            except ValueError:
                pass

        self.artifacts["logs"].append({
            "path": str(rel_path),
            "sha256": self._hash_file(path),
            "type": log_type
        })

    def add_figure(self, path: Path, description: str) -> None:
        """
        Add a figure artifact to the manifest.

        Args:
            path: Path to the figure file (absolute or relative).
            description: Human-readable description of the figure.

        Side-effects:
            Appends to self.artifacts["figures"].
            Reads file to compute hash.
        """
        rel_path = path
        if path.is_absolute():
            try:
                rel_path = path.relative_to(Path.cwd())
            except ValueError:
                pass

        self.artifacts["figures"].append({
            "path": str(rel_path),
            "sha256": self._hash_file(path),
            "description": description
        })

    def add_specialized_config(self, key: str, value: str) -> None:
        """
        Add a specialized configuration key-value pair.

        Args:
            key: Configuration key name.
            value: Configuration value.
        """
        self.configuration["specialized"][key] = value

    def save(self, output_path: Path) -> None:
        """
        Write the complete manifest to a JSON file.

        Args:
            output_path: Path where the manifest will be saved.

        Side-effects:
            Creates parent directories if needed.
            Writes JSON file to disk.
            Prints confirmation message to stdout.
        """
        full_manifest = {
            "meta": self.meta,
            "experiment": self.experiment,
            "provenance": self.provenance,
            "configuration": self.configuration,
            "artifacts": self.artifacts
        }
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(full_manifest, f, indent=2)
        print(f"Manifest saved to {output_path}")
