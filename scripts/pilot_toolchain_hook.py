#!/usr/bin/env python3
"""
Pilot Toolchain Provenance Hook

Generates toolchain provenance manifests for PILOT artifacts.

IMPORTANT DISCLAIMER:
    PILOT PROVENANCE ≠ EXPERIMENT PROVENANCE

    Pilot provenance binds an artifact to the toolchain state at ingestion time.
    It does NOT:
    - Claim experimental validity
    - Claim reproducibility
    - Imply parity with CAL-EXP runs
    - Constitute experiment provenance

SAVE TO REPO: YES
Rationale: Enables environment attribution for pilot artifacts without overclaim.

Usage:
    python scripts/pilot_toolchain_hook.py --artifact-id pilot-001
    python scripts/pilot_toolchain_hook.py --artifact-id pilot-001 --source path/to/artifact
    python scripts/pilot_toolchain_hook.py --artifact-id pilot-001 --output manifests/
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT))

from substrate.repro.toolchain import capture_toolchain_snapshot


# Required disclaimer - must be included verbatim
PILOT_DISCLAIMER = (
    "PILOT PROVENANCE: Binds artifact to toolchain state at ingestion. "
    "NOT experiment provenance. Does not imply experimental validity, "
    "reproducibility, or parity with CAL-EXP runs."
)


def write_deterministic_json(data: Dict[str, Any], path: Path) -> None:
    """Write JSON in deterministic normal form."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")


def hash_file(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def build_pilot_manifest(
    artifact_id: str,
    source_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Build a pilot provenance manifest.

    Args:
        artifact_id: Unique identifier for the pilot artifact.
        source_path: Optional path to the source artifact for hashing.

    Returns:
        Manifest dict in schema-compliant format.
    """
    # Capture toolchain snapshot
    try:
        snapshot = capture_toolchain_snapshot(REPO_ROOT)
        toolchain_fingerprint = snapshot.fingerprint
        toolchain_snapshot = {
            "python_version": snapshot.python.version,
            "uv_version": snapshot.python.uv_version,
            "lean_version": snapshot.lean.version,
        }
        uv_lock_hash = snapshot.python.uv_lock_hash
    except FileNotFoundError:
        # Fallback if toolchain files not found
        toolchain_fingerprint = None
        toolchain_snapshot = None
        uv_lock_hash = "unknown"

    # Build source info if path provided
    source_info = None
    if source_path and source_path.exists():
        source_info = {
            "path": str(source_path),
            "hash": hash_file(source_path),
            "size_bytes": source_path.stat().st_size,
        }

    manifest = {
        "schema_version": "1.0.0",
        "artifact_type": "pilot",
        "artifact_id": artifact_id,
        "ingestion_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "provenance_level": "pilot",
        "disclaimer": PILOT_DISCLAIMER,
        "uv_lock_hash": uv_lock_hash,
        "toolchain_fingerprint": toolchain_fingerprint,
        "toolchain_snapshot": toolchain_snapshot,
        "source": source_info,
        "_meta": {
            "generator": "scripts/pilot_toolchain_hook.py",
            "note": "Pilot provenance - environment binding only",
        },
    }

    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Generate pilot toolchain provenance manifest",
        epilog="NOTE: Pilot provenance ≠ experiment provenance. See disclaimer in output.",
    )
    parser.add_argument(
        "--artifact-id",
        required=True,
        help="Unique identifier for the pilot artifact",
    )
    parser.add_argument(
        "--source",
        type=Path,
        help="Path to source artifact (for hashing)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=REPO_ROOT / "docs" / "system_law" / "calibration" / "audits",
        help="Output directory for manifest",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output manifest as JSON to stdout",
    )
    args = parser.parse_args()

    # Build manifest
    manifest = build_pilot_manifest(args.artifact_id, args.source)

    if args.json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        # Write to file
        args.output.mkdir(parents=True, exist_ok=True)
        manifest_path = args.output / f"{args.artifact_id}_toolchain_manifest.json"
        write_deterministic_json(manifest, manifest_path)

        print(f"Pilot manifest written: {manifest_path}")
        print()
        print("=" * 70)
        print("DISCLAIMER")
        print("=" * 70)
        print(PILOT_DISCLAIMER)
        print("=" * 70)
        print()
        print("Pilot Provenance:")
        print(f"  artifact_id:           {manifest['artifact_id']}")
        print(f"  ingestion_timestamp:   {manifest['ingestion_timestamp']}")
        print(f"  uv_lock_hash:          {manifest['uv_lock_hash'][:32]}...")
        if manifest['toolchain_fingerprint']:
            print(f"  toolchain_fingerprint: {manifest['toolchain_fingerprint'][:32]}...")


if __name__ == "__main__":
    main()
