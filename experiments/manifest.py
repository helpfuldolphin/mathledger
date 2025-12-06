"""
PHASE II — NOT USED IN PHASE I

Manifest generation helpers for experiment reproducibility.

This module provides utilities for generating cryptographically bound
experiment manifests that capture provenance, configuration, and artifact
information for reproducibility verification.
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


class ManifestGenerator:
    """
    Generate experiment manifests with provenance and configuration tracking.

    PHASE II — NOT USED IN PHASE I

    This class captures git state, system information, configuration hashes,
    and artifact metadata to enable reproducibility verification of experiments.
    RFL uplift detection is not part of manifest validation.

    Attributes:
        meta: Manifest metadata including version and type.
        experiment: Experiment identification including id, type, and timestamp.
        config_path: Path to the configuration file.
        mdap_seed: MDAP seed for determinism tracking.
        hermetic: Whether the experiment runs in hermetic mode.
        no_network: Whether the experiment runs without network access.
        provenance: Git and system provenance information.
        configuration: Configuration snapshot including hashes.
        artifacts: Log and figure artifact tracking.
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
        Initialize a new manifest generator.

        Args:
            experiment_id: Unique identifier for the experiment.
            experiment_type: Type of experiment (e.g., "uplift_u2").
            description: Human-readable description of the experiment.
            config_path: Path to the experiment configuration file.
            mdap_seed: The MDAP seed used for deterministic execution.
            is_hermetic: Whether the experiment is hermetic (default True).
            is_no_network: Whether network access is disabled (default True).
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

        self.provenance = self._get_provenance()
        self.configuration = self._get_configuration()
        self.artifacts: Dict[str, list] = {"logs": [], "figures": []}

    def _get_git_info(self) -> Dict[str, Any]:
        """
        Retrieve git repository state information.

        Returns:
            Dictionary containing commit_sha, branch, is_dirty flag, and
            diff_sha256 hash if the working directory has uncommitted changes.
        """
        try:
            commit_sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode().strip()

            # Check if dirty
            status_out = subprocess.check_output(
                ["git", "status", "--porcelain"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            is_dirty = bool(status_out)

            diff_sha256: Optional[str] = None
            if is_dirty:
                # Get diff hash
                diff_bytes = subprocess.check_output(
                    ["git", "diff", "HEAD"],
                    stderr=subprocess.DEVNULL
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
            Dictionary containing hostname, os, python_version, and env_hash
            (SHA256 of pip freeze output).
        """
        import platform
        import socket

        env_hash = "unknown"
        try:
            # Use current sys.executable to get pip freeze of the active environment
            env_bytes = subprocess.check_output(
                [sys.executable, "-m", "pip", "freeze"],
                stderr=subprocess.DEVNULL
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
        Compile complete provenance information.

        Returns:
            Dictionary with 'git' and 'system' provenance sub-dictionaries.
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
            Hexadecimal SHA256 hash string, or "missing" if file does not exist.
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
        Build configuration snapshot with file hashes.

        Returns:
            Dictionary containing config_file info, parameters placeholder,
            determinism settings, and specialized config.
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
            path: Path to the log file.
            log_type: Type/category of the log (e.g., "experiment", "telemetry").
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
            path: Path to the figure file.
            description: Human-readable description of the figure.
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
        Add a specialized configuration parameter.

        Args:
            key: The parameter key name.
            value: The parameter value.
        """
        self.configuration["specialized"][key] = value

    def save(self, output_path: Path) -> None:
        """
        Write the manifest to a JSON file.

        Args:
            output_path: Path to write the manifest file to.
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
