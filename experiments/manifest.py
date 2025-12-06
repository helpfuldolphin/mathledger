"""
Experiment Manifest Generator
=============================

This module provides structured manifest generation for MathLedger experiments.
Manifests capture provenance, configuration, and artifact metadata for
reproducibility and audit purposes.

Module Responsibilities:
  - Capture git provenance (commit, branch, dirty state)
  - Record system environment (hostname, OS, Python version)
  - Hash configuration files and artifacts
  - Generate structured JSON manifests

Note: RFL uplift detection is not part of manifest validation.
      Manifests are purely for provenance tracking.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# --- Logging Setup ---
logger = logging.getLogger("U2Manifest")


# -----------------------------------------------------------------------------
# Data Structures
# -----------------------------------------------------------------------------

@dataclass
class GitInfo:
    """Git repository state information."""
    commit_sha: str
    branch: str
    is_dirty: bool
    diff_sha256: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "commit_sha": self.commit_sha,
            "branch": self.branch,
            "is_dirty": self.is_dirty,
            "diff_sha256": self.diff_sha256
        }


@dataclass
class SystemInfo:
    """System environment information."""
    hostname: str
    os: str
    python_version: str
    env_hash: str

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary for JSON serialization."""
        return {
            "hostname": self.hostname,
            "os": self.os,
            "python_version": self.python_version,
            "env_hash": self.env_hash
        }


@dataclass
class LogArtifact:
    """Log artifact entry."""
    path: str
    sha256: str
    log_type: str

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary for JSON serialization."""
        return {
            "path": self.path,
            "sha256": self.sha256,
            "type": self.log_type
        }


@dataclass
class FigureArtifact:
    """Figure artifact entry."""
    path: str
    sha256: str
    description: str

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary for JSON serialization."""
        return {
            "path": self.path,
            "sha256": self.sha256,
            "description": self.description
        }


class ManifestGenerator:
    """Experiment manifest generator for provenance tracking.

    Generates structured JSON manifests containing:
      - Metadata (version, type)
      - Experiment info (id, type, timestamp, description)
      - Provenance (git state, system info)
      - Configuration (paths, hashes, parameters)
      - Artifacts (logs, figures with hashes)

    Attributes:
        meta: Manifest metadata (version, type).
        experiment: Experiment identification and timestamp.
        config_path: Path to configuration file.
        mdap_seed: MDAP determinism seed.
        hermetic: Whether experiment is hermetic (no external deps).
        no_network: Whether experiment runs without network access.
        provenance: Git and system provenance information.
        configuration: Configuration details and hashes.
        artifacts: Log and figure artifact entries.
    """

    MANIFEST_VERSION = "1.0.0"
    MANIFEST_TYPE = "mathledger_experiment_manifest"

    def __init__(
        self,
        experiment_id: str,
        experiment_type: str,
        description: str,
        config_path: Path,
        mdap_seed: str,
        is_hermetic: bool = True,
        is_no_network: bool = True
    ) -> None:
        """Initialize manifest generator.

        Args:
            experiment_id: Unique experiment identifier.
            experiment_type: Type of experiment (e.g., 'u2_uplift').
            description: Human-readable experiment description.
            config_path: Path to experiment configuration file.
            mdap_seed: MDAP determinism seed string.
            is_hermetic: Whether experiment is hermetic.
            is_no_network: Whether experiment runs offline.
        """
        self.meta: Dict[str, str] = {
            "version": self.MANIFEST_VERSION,
            "type": self.MANIFEST_TYPE
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
        self.artifacts: Dict[str, List[Dict[str, str]]] = {"logs": [], "figures": []}

    def _get_git_info(self) -> Dict[str, Any]:
        """Retrieve git repository state.

        Returns:
            Dictionary containing commit SHA, branch, dirty state, and diff hash.
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

            status_out = subprocess.check_output(
                ["git", "status", "--porcelain"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            is_dirty = bool(status_out)

            diff_sha256: Optional[str] = None
            if is_dirty:
                diff_bytes = subprocess.check_output(
                    ["git", "diff", "HEAD"],
                    stderr=subprocess.DEVNULL
                )
                diff_sha256 = hashlib.sha256(diff_bytes).hexdigest()

            return GitInfo(
                commit_sha=commit_sha,
                branch=branch,
                is_dirty=is_dirty,
                diff_sha256=diff_sha256
            ).to_dict()

        except subprocess.CalledProcessError:
            logger.debug("Git command failed, returning unknown state")
            return GitInfo(
                commit_sha="unknown",
                branch="unknown",
                is_dirty=False,
                diff_sha256=None
            ).to_dict()

        except FileNotFoundError:
            logger.debug("Git not found on system")
            return GitInfo(
                commit_sha="git_not_found",
                branch="unknown",
                is_dirty=False,
                diff_sha256=None
            ).to_dict()

    def _get_system_info(self) -> Dict[str, str]:
        """Retrieve system environment information.

        Returns:
            Dictionary containing hostname, OS, Python version, and env hash.
        """
        env_hash = "unknown"
        try:
            env_bytes = subprocess.check_output(
                [sys.executable, "-m", "pip", "freeze"],
                stderr=subprocess.DEVNULL
            )
            env_hash = hashlib.sha256(env_bytes).hexdigest()
        except Exception:
            logger.debug("Failed to compute environment hash")

        return SystemInfo(
            hostname=socket.gethostname(),
            os=platform.platform(),
            python_version=platform.python_version(),
            env_hash=env_hash
        ).to_dict()

    def _get_provenance(self) -> Dict[str, Any]:
        """Build provenance information dictionary.

        Returns:
            Dictionary containing git and system provenance.
        """
        return {
            "git": self._get_git_info(),
            "system": self._get_system_info()
        }

    def _hash_file(self, path: Path) -> str:
        """Compute SHA256 hash of a file.

        Args:
            path: Path to file to hash.

        Returns:
            Hex-encoded SHA256 hash, or 'missing' if file doesn't exist.
        """
        if not path.exists():
            logger.debug(f"File not found for hashing: {path}")
            return "missing"
        sha256_hash = hashlib.sha256()
        with open(path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _get_configuration(self) -> Dict[str, Any]:
        """Build configuration dictionary.

        Returns:
            Dictionary containing config file info, parameters, and determinism settings.
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
        """Add a log artifact to the manifest.

        Args:
            path: Path to log file.
            log_type: Type/category of the log.
        """
        rel_path = self._make_relative_path(path)
        self.artifacts["logs"].append({
            "path": str(rel_path),
            "sha256": self._hash_file(path),
            "type": log_type
        })

    def add_figure(self, path: Path, description: str) -> None:
        """Add a figure artifact to the manifest.

        Args:
            path: Path to figure file.
            description: Description of the figure.
        """
        rel_path = self._make_relative_path(path)
        self.artifacts["figures"].append({
            "path": str(rel_path),
            "sha256": self._hash_file(path),
            "description": description
        })

    def _make_relative_path(self, path: Path) -> Path:
        """Convert absolute path to relative if possible.

        Args:
            path: Path to convert.

        Returns:
            Relative path if conversion succeeds, original path otherwise.
        """
        if path.is_absolute():
            try:
                return path.relative_to(Path.cwd())
            except ValueError:
                pass
        return path

    def add_specialized_config(self, key: str, value: str) -> None:
        """Add a specialized configuration parameter.

        Args:
            key: Parameter key.
            value: Parameter value.
        """
        self.configuration["specialized"][key] = value

    def save(self, output_path: Path) -> None:
        """Save manifest to JSON file.

        Args:
            output_path: Path to output file.
        """
        full_manifest = {
            "meta": self.meta,
            "experiment": self.experiment,
            "provenance": self.provenance,
            "configuration": self.configuration,
            "artifacts": self.artifacts
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(full_manifest, f, indent=2)
        logger.info(f"Manifest saved to {output_path}")
