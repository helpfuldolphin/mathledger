#!/usr/bin/env python3
"""
Canonical policy migration tool.

Transforms legacy policy exports (policy.json + weights) into the
manifest-based structure described in docs/policy_archivist.md.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict

SCHEMA_VERSION = "policy_manifest@v2"
DEFAULT_SURFACE_FILE = "policy.surface.yaml"
DEFAULT_OUTPUT_ROOT = Path("artifacts/policy")
DEFAULT_WEIGHTS_NAME = "policy.weights.bin"
LEGACY_WEIGHT_CANDIDATES = ("policy.bin", "policy.pkl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate legacy policy artifacts into canonical manifest form.")
    parser.add_argument("--source", required=True, help="Path to legacy policy weights or directory.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Destination root for canonical artifacts.")
    parser.add_argument("--surface", help="Optional path to learning surface YAML; copied if provided.")
    parser.add_argument("--force", action="store_true", help="Overwrite destination if it already exists.")
    args = parser.parse_args()

    source_path = Path(args.source).resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Source path not found: {source_path}")

    weights_path = _resolve_weights(source_path)
    legacy_metadata = _load_legacy_metadata(weights_path.parent)

    policy_hash = _sha256(weights_path)
    created_at = legacy_metadata.get("created_at") or dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    destination_root = Path(args.output_root).resolve() / policy_hash

    if destination_root.exists():
        if not args.force:
            raise FileExistsError(f"Destination already exists: {destination_root}")
        shutil.rmtree(destination_root)

    destination_root.mkdir(parents=True, exist_ok=True)

    dest_weights = destination_root / DEFAULT_WEIGHTS_NAME
    shutil.copy2(weights_path, dest_weights)

    surface_source = Path(args.surface).resolve() if args.surface else weights_path.parent / DEFAULT_SURFACE_FILE
    if surface_source.exists():
        shutil.copy2(surface_source, destination_root / DEFAULT_SURFACE_FILE)

    manifest = _build_manifest(policy_hash, created_at, legacy_metadata, dest_weights.stat().st_size)
    manifest_path = destination_root / "policy.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[policy-migrate] migrated {weights_path} -> {destination_root}")
    print(f"[policy-migrate] manifest: {manifest_path}")


def _resolve_weights(source_path: Path) -> Path:
    """Locate the policy weights file."""
    if source_path.is_file():
        return source_path

    for candidate_name in (DEFAULT_WEIGHTS_NAME, *LEGACY_WEIGHT_CANDIDATES):
        candidate = source_path / candidate_name
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Unable to locate weights under {source_path}. Expected one of: "
        f"{DEFAULT_WEIGHTS_NAME}, {', '.join(LEGACY_WEIGHT_CANDIDATES)}."
    )


def _load_legacy_metadata(base_dir: Path) -> Dict[str, str]:
    """Best-effort load of policy.json for backward compatibility."""
    legacy_path = base_dir / "policy.json"
    if legacy_path.exists():
        with legacy_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data
    return {}


def _build_manifest(policy_hash: str, created_at: str, legacy: Dict[str, str], byte_size: int) -> Dict[str, object]:
    """Construct manifest dictionary following the canonical schema."""
    policy_section = {
        "hash": policy_hash,
        "version": legacy.get("version", "legacy"),
        "model_type": legacy.get("model_type", "unknown"),
        "serialization": legacy.get("serialization", "pickle@3.11"),
        "byte_size": byte_size,
        "created_at": created_at,
    }

    training_context = {
        "dataset": legacy.get("dataset", "unknown"),
        "curriculum": legacy.get("curriculum", "unknown"),
        "seed": legacy.get("seed", 0),
        "code_commit": legacy.get("code_commit", "unknown"),
        "runner": legacy.get("runner", "unknown"),
    }

    compatibility = {
        "required_features": legacy.get("required_features", ["atoms", "depth", "lean_score"]),
        "supports_symbolic_descent": legacy.get("supports_symbolic_descent", True),
        "min_runner_version": legacy.get("min_runner_version", "2025.11.20"),
        "max_runner_version": legacy.get("max_runner_version"),
    }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "policy": policy_section,
        "training_context": training_context,
        "compatibility": compatibility,
    }
    return manifest


def _sha256(path: Path) -> str:
    """Return SHA-256 hash for the given file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
