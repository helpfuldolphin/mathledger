#!/usr/bin/env python3
"""
CAL-EXP-2 Toolchain Parity Hook

Writes toolchain provenance in normal-form JSON to audits directory.
This is operationally unavoidable (must be called) but not gating (no enforcement).

Normal-form guarantees:
- Sorted keys at all levels
- Consistent indentation (2 spaces)
- Trailing newline
- Schema-compliant output

SAVE TO REPO: YES
Rationale: Ensures divergence tuning is attributable to specific toolchain state.

Usage:
    python scripts/cal_exp2_toolchain_hook.py
    python scripts/cal_exp2_toolchain_hook.py --run-dir results/cal_exp_2/p4_YYYYMMDD_HHMMSS
    python scripts/cal_exp2_toolchain_hook.py --ephemeral  # Write to results/ instead
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT))

from substrate.repro.toolchain import capture_toolchain_snapshot


def write_deterministic_json(data: Dict[str, Any], path: Path) -> None:
    """
    Write JSON in deterministic normal form.

    Guarantees:
    - Keys sorted at all levels
    - 2-space indentation
    - Trailing newline
    - UTF-8 encoding
    """
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")  # Trailing newline


def build_full_provenance_manifest(
    experiment_id: str,
    run_metadata_ref: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Build a full-provenance toolchain manifest (provenance_level=full).

    Returns manifest dict with all fields in schema order.
    """
    snapshot = capture_toolchain_snapshot(REPO_ROOT)

    manifest = {
        "schema_version": "1.0.0",
        "experiment_id": experiment_id,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "provenance_level": "full",
        "toolchain_fingerprint": snapshot.fingerprint,
        "uv_lock_hash": snapshot.python.uv_lock_hash,
        "lean_toolchain_hash": snapshot.lean.toolchain_hash,
        "lake_manifest_hash": snapshot.lean.lake_manifest_hash,
        "toolchain": {
            "schema_version": snapshot.schema_version,
            "fingerprint": snapshot.fingerprint,
            "python": {
                "version": snapshot.python.version,
                "uv_version": snapshot.python.uv_version,
                "uv_lock_hash": snapshot.python.uv_lock_hash,
            },
            "lean": {
                "version": snapshot.lean.version,
                "toolchain_hash": snapshot.lean.toolchain_hash,
                "lake_manifest_hash": snapshot.lean.lake_manifest_hash,
            },
        },
        "git_evidence": None,
        "run_metadata_ref": run_metadata_ref,
        "_meta": {
            "generator": "scripts/cal_exp2_toolchain_hook.py",
            "note": "Full provenance - all hashes captured at runtime",
        },
    }

    return manifest


def write_cal_exp2_manifest(run_dir: Path = None, canonical: bool = True) -> Path:
    """
    Write CAL-EXP-2 manifest with toolchain provenance.

    Args:
        run_dir: Optional specific run directory to reference.
        canonical: If True, write to docs/system_law/calibration/audits/ (tracked).
                   If False, write to results/cal_exp_2/ (ephemeral).

    Returns:
        Path to written manifest.
    """
    if canonical:
        output_dir = REPO_ROOT / "docs" / "system_law" / "calibration" / "audits"
    else:
        output_dir = REPO_ROOT / "results" / "cal_exp_2"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build run metadata reference if run_dir provided
    run_metadata_ref = None
    if run_dir:
        run_dir = Path(run_dir).resolve()
        if not run_dir.is_absolute():
            run_dir = REPO_ROOT / run_dir
        if run_dir.exists():
            metadata_path = run_dir / "RUN_METADATA.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    run_metadata = json.load(f)
                run_metadata_ref = {
                    "verdict": run_metadata.get("verdict"),
                    "timestamp": run_metadata.get("timestamp"),
                }

    # Build manifest
    manifest = build_full_provenance_manifest("CAL-EXP-2", run_metadata_ref)

    # Write in deterministic normal form
    filename = "cal_exp_2_toolchain_manifest.json" if canonical else "cal_exp_2_manifest.json"
    manifest_path = output_dir / filename
    write_deterministic_json(manifest, manifest_path)

    return manifest_path


def main():
    parser = argparse.ArgumentParser(
        description="Write CAL-EXP-2 manifest with toolchain provenance (normal-form JSON)"
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Specific run directory to reference",
    )
    parser.add_argument(
        "--ephemeral",
        action="store_true",
        help="Write to results/ (ephemeral) instead of audits/ (canonical)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output manifest as JSON to stdout",
    )
    args = parser.parse_args()

    manifest_path = write_cal_exp2_manifest(args.run_dir, canonical=not args.ephemeral)

    if args.json:
        with open(manifest_path) as f:
            print(f.read())
    else:
        print(f"CAL-EXP-2 manifest written: {manifest_path}")

        # Print key fields
        with open(manifest_path) as f:
            manifest = json.load(f)
        print()
        print("Toolchain Provenance (full):")
        print(f"  fingerprint:         {manifest['toolchain_fingerprint']}")
        print(f"  uv_lock_hash:        {manifest['uv_lock_hash'][:32]}...")
        print(f"  lean_toolchain_hash: {manifest['lean_toolchain_hash'][:32]}...")
        print(f"  lake_manifest_hash:  {manifest['lake_manifest_hash'][:32]}...")


if __name__ == "__main__":
    main()
