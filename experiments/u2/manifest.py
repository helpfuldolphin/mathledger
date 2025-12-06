"""
PHASE-II — NOT USED IN PHASE I

Experiment Manifest Generation
==============================

This module provides manifest generation utilities for U2 uplift experiments.
Manifests capture the complete provenance of an experiment run including
configuration hashes, seed schedules, and output paths.

**Determinism Notes:**
    - All hash computations use SHA-256 for reproducibility.
    - Manifest structure is stable and sorted for consistent serialization.
    - Same inputs always produce the same manifest.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def compute_hash(data: str) -> str:
    """Compute the SHA-256 hash of a string.

    Args:
        data: The string to hash.

    Returns:
        The hexadecimal SHA-256 hash of the input string.

    Example:
        >>> compute_hash("test")
        '9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08'
        >>> compute_hash("test") == compute_hash("test")
        True

    **Determinism Notes:**
        - SHA-256 is deterministic (same input always produces same hash).
        - Uses UTF-8 encoding for consistent byte representation.
    """
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def compute_config_hash(config: Dict[str, Any]) -> str:
    """Compute a deterministic hash of a configuration dictionary.

    The configuration is serialized with sorted keys to ensure
    consistent ordering, then hashed using SHA-256.

    Args:
        config: The configuration dictionary to hash.

    Returns:
        The SHA-256 hash of the JSON-serialized configuration.

    Example:
        >>> config = {"a": 1, "b": 2}
        >>> compute_config_hash(config) == compute_config_hash({"b": 2, "a": 1})
        True

    **Determinism Notes:**
        - Uses sort_keys=True for stable serialization.
        - Same configuration always produces same hash.
    """
    config_str = json.dumps(config, sort_keys=True)
    return compute_hash(config_str)


def compute_series_hash(series: List[Dict[str, Any]]) -> str:
    """Compute a deterministic hash of a telemetry series.

    The series is serialized with sorted keys to ensure consistent
    ordering, then hashed using SHA-256.

    Args:
        series: The list of telemetry records to hash.

    Returns:
        The SHA-256 hash of the JSON-serialized series.

    **Determinism Notes:**
        - Uses sort_keys=True for stable serialization.
        - Same series always produces same hash.
    """
    series_str = json.dumps(series, sort_keys=True)
    return compute_hash(series_str)


def generate_manifest(
    slice_name: str,
    mode: str,
    cycles: int,
    initial_seed: int,
    slice_config: Dict[str, Any],
    prereg_hash: Optional[str],
    ht_series: List[Dict[str, Any]],
    seed_schedule: List[int],
    results_path: Path,
    manifest_path: Path,
) -> Dict[str, Any]:
    """Generate a complete experiment manifest.

    The manifest captures all provenance information needed to reproduce
    and verify an experiment run.

    Args:
        slice_name: The name of the experiment slice.
        mode: The execution mode ("baseline" or "rfl").
        cycles: The number of experiment cycles executed.
        initial_seed: The initial random seed used.
        slice_config: The configuration for the experiment slice.
        prereg_hash: The preregistration hash, if available.
        ht_series: The telemetry series (Hₜ) recorded during the run.
        seed_schedule: The deterministic seed schedule used.
        results_path: Path to the results file.
        manifest_path: Path to the manifest file.

    Returns:
        A dictionary containing the complete manifest.

    Raises:
        ValueError: If required fields are missing or invalid.

    Example:
        >>> manifest = generate_manifest(
        ...     slice_name="arithmetic_simple",
        ...     mode="baseline",
        ...     cycles=10,
        ...     initial_seed=42,
        ...     slice_config={"items": ["1+1", "2+2"]},
        ...     prereg_hash=None,
        ...     ht_series=[],
        ...     seed_schedule=[1, 2, 3],
        ...     results_path=Path("results.jsonl"),
        ...     manifest_path=Path("manifest.json"),
        ... )
        >>> manifest["slice"]
        'arithmetic_simple'

    **Determinism Notes:**
        - Manifest structure is stable across runs.
        - All hashes are computed deterministically.
    """
    if not slice_name:
        raise ValueError("slice_name cannot be empty")
    if mode not in ("baseline", "rfl"):
        raise ValueError(f"Invalid mode: {mode}. Expected 'baseline' or 'rfl'.")
    if cycles <= 0:
        raise ValueError(f"cycles must be positive, got {cycles}")

    slice_config_hash = compute_config_hash(slice_config)
    ht_series_hash = compute_series_hash(ht_series)

    manifest: Dict[str, Any] = {
        "label": "PHASE II — NOT USED IN PHASE I",
        "slice": slice_name,
        "mode": mode,
        "cycles": cycles,
        "initial_seed": initial_seed,
        "slice_config_hash": slice_config_hash,
        "prereg_hash": prereg_hash if prereg_hash else "N/A",
        "ht_series_hash": ht_series_hash,
        "deterministic_seed_schedule": seed_schedule,
        "outputs": {
            "results": str(results_path),
            "manifest": str(manifest_path),
        },
    }

    return manifest


def save_manifest(manifest: Dict[str, Any], path: Path) -> None:
    """Save a manifest to a JSON file.

    Args:
        manifest: The manifest dictionary to save.
        path: The output file path.

    Raises:
        IOError: If the file cannot be written.

    **Determinism Notes:**
        - Uses indent=2 and sorted keys for readable, stable output.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def load_manifest(path: Path) -> Dict[str, Any]:
    """Load a manifest from a JSON file.

    Args:
        path: The path to the manifest file.

    Returns:
        The loaded manifest dictionary.

    Raises:
        FileNotFoundError: If the manifest file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_manifest(manifest: Dict[str, Any]) -> List[str]:
    """Validate a manifest for required fields and structure.

    Args:
        manifest: The manifest dictionary to validate.

    Returns:
        A list of validation error messages. Empty if valid.

    Example:
        >>> errors = validate_manifest({"slice": "test"})
        >>> "Missing required field: mode" in errors
        True
    """
    errors: List[str] = []
    required_fields = [
        "label",
        "slice",
        "mode",
        "cycles",
        "initial_seed",
        "slice_config_hash",
        "ht_series_hash",
        "outputs",
    ]

    for field in required_fields:
        if field not in manifest:
            errors.append(f"Missing required field: {field}")

    if "outputs" in manifest:
        if "results" not in manifest["outputs"]:
            errors.append("Missing required output field: results")
        if "manifest" not in manifest["outputs"]:
            errors.append("Missing required output field: manifest")

    if "slice_config_hash" in manifest:
        if len(manifest["slice_config_hash"]) != 64:
            errors.append(
                f"Invalid slice_config_hash length: expected 64, got {len(manifest['slice_config_hash'])}"
            )

    return errors
