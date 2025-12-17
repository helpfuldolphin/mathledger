#!/usr/bin/env python3
"""
Smoke harness for structural calibration panel manifest binding + status extraction.

STATUS: PHASE X — P5 STRUCTURAL REGIME MONITOR

This script creates a minimal evidence pack with a structural calibration panel,
runs manifest mirroring logic, and verifies status extraction reads from manifest.

SHADOW MODE CONTRACT:
- This script is purely for verification
- It does not gate or block any operations
- All outputs are deterministic
- Exit code always 0 (never blocks)

Usage:
    python scripts/smoke_structural_cal_panel.py [--output-dir <dir>] [--output-json <path>]

Output:
    Prints signals["structure_calibration_panel"] as JSON to stdout.
    If --output-json is provided, writes machine-readable result JSON to that path.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any

# Import panel builder and signal extractor
from backend.health.lean_shadow_adapter import (
    build_structural_calibration_panel,
    extract_structural_calibration_panel_signal,
    STRUCTURAL_CALIBRATION_PANEL_SCHEMA_VERSION,
)

# Import status generator
from scripts.generate_first_light_status import generate_status


def create_minimal_evidence_pack_with_panel(
    output_dir: Path,
) -> Path:
    """
    Create minimal evidence_pack.json with structural calibration panel.

    Args:
        output_dir: Directory to write evidence_pack.json.

    Returns:
        Path to created evidence_pack.json.
    """
    # Create synthetic calibration experiment reports
    report1 = {
        "cal_id": "CAL-EXP-1",
        "structural_summary": {
            "mean_structural_error_rate": 0.1,
            "max_structural_error_rate": 0.2,
            "anomaly_bursts": [],
        },
        "governance": {
            "structure": {
                "lean_cross_check": {
                    "status": "CONSISTENT",
                },
            },
        },
    }
    
    report2 = {
        "cal_id": "CAL-EXP-2",
        "structural_summary": {
            "mean_structural_error_rate": 0.3,
            "max_structural_error_rate": 0.4,
            "anomaly_bursts": [
                {"start_index": 0, "end_index": 2, "length": 3},
            ],
        },
        "governance": {
            "structure": {
                "lean_cross_check": {
                    "status": "CONFLICT",
                },
            },
        },
    }
    
    # Build calibration panel
    panel = build_structural_calibration_panel([report1, report2])
    
    # Create evidence pack structure
    evidence = {
        "schema_version": "1.0.0",
        "timestamp": "2024-01-01T00:00:00Z",
        "governance": {
            "structure": {
                "calibration_panel": panel,
            },
        },
    }
    
    # Write evidence_pack.json
    evidence_path = output_dir / "evidence_pack.json"
    with open(evidence_path, "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2)
    
    return evidence_path


def mirror_calibration_panel_to_manifest(
    evidence_pack_dir: Path,
    manifest: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Mirror calibration panel from evidence_pack.json to manifest.

    This replicates the logic from build_first_light_evidence_pack.py
    to ensure manifest mirrors evidence governance blocks.

    Args:
        evidence_pack_dir: Directory containing evidence_pack.json.
        manifest: Existing manifest dict (will be modified).

    Returns:
        Updated manifest dict with calibration panel if present.
    """
    evidence_pack_json_path = evidence_pack_dir / "evidence_pack.json"
    if evidence_pack_json_path.exists():
        try:
            with open(evidence_pack_json_path, "r", encoding="utf-8") as f:
                evidence = json.load(f)
            governance = evidence.get("governance", {})
            structure = governance.get("structure", {})
            calibration_panel = structure.get("calibration_panel")
            if calibration_panel:
                manifest["governance"] = manifest.get("governance", {})
                if "structure" not in manifest["governance"]:
                    manifest["governance"]["structure"] = {}
                manifest["governance"]["structure"]["calibration_panel"] = calibration_panel
        except (json.JSONDecodeError, OSError, KeyError):
            # Evidence pack not available or invalid - not an error (SHADOW MODE)
            pass
    
    return manifest


def main() -> int:
    """Main entry point for smoke harness."""
    parser = argparse.ArgumentParser(
        description="Smoke harness for structural calibration panel manifest binding"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for smoke test artifacts (default: temp directory)",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary directory after completion",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Path to write machine-readable result JSON (always exit 0)",
    )
    args = parser.parse_args()
    
    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        cleanup_temp = False
    else:
        output_dir = Path(tempfile.mkdtemp(prefix="smoke_cal_panel_"))
        cleanup_temp = not args.keep_temp
    
    try:
        # Step 1: Create minimal evidence_pack.json with calibration panel
        print(f"[smoke] Creating evidence_pack.json in {output_dir}")
        evidence_path = create_minimal_evidence_pack_with_panel(output_dir)
        print(f"[smoke] Created {evidence_path}")
        
        # Step 2: Create minimal manifest.json
        print(f"[smoke] Creating manifest.json")
        manifest = {
            "schema_version": "1.0.0",
            "pack_type": "first_light_evidence",
            "mode": "SHADOW",
            "files": [],
            "file_count": 0,
        }
        
        # Step 3: Mirror calibration panel from evidence to manifest
        print(f"[smoke] Mirroring calibration panel to manifest")
        manifest_before = manifest.copy()
        manifest = mirror_calibration_panel_to_manifest(output_dir, manifest)
        
        # Check if mirroring succeeded
        manifest_mirroring_ok = (
            manifest.get("governance", {}).get("structure", {}).get("calibration_panel") is not None
        )
        
        # Write manifest.json
        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"[smoke] Created {manifest_path}")
        
        # Step 4: Run status generation
        print(f"[smoke] Running status generation")
        # Create minimal P3/P4 dirs (status generator expects these)
        p3_dir = output_dir / "p3_synthetic"
        p4_dir = output_dir / "p4_shadow"
        p3_dir.mkdir(exist_ok=True)
        p4_dir.mkdir(exist_ok=True)
        
        # Generate status
        status = generate_status(
            p3_dir=p3_dir,
            p4_dir=p4_dir,
            evidence_pack_dir=output_dir,
            pipeline="smoke_test",
        )
        
        # Step 5: Extract and print signal
        signals = status.get("signals")
        if signals is None:
            signals = {}
        panel_signal = signals.get("structure_calibration_panel")
        status_signal_present = panel_signal is not None
        
        # Count warnings
        warnings_count = len(status.get("warnings", []))
        
        if panel_signal:
            print(f"[smoke] OK: Status extraction successful")
            print(f"[smoke] Extracted signal:")
            print(json.dumps(panel_signal, indent=2))
        else:
            print(f"[smoke] WARN: Status extraction failed: signal not found")
            print(f"[smoke] Available signals: {list(signals.keys())}")
        
        # Write output JSON if requested
        if args.output_json:
            result = {
                "schema_version": "1.0.0",
                "manifest_mirroring_ok": manifest_mirroring_ok,
                "status_signal_present": status_signal_present,
                "warnings_count": warnings_count,
            }
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            print(f"[smoke] Wrote result JSON to {args.output_json}")
        
        # Always exit 0 (non-gating)
        return 0
    
    except Exception as e:
        print(f"[smoke] ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        # Exit 0 even on error (non-gating)
        return 0
    
    finally:
        if cleanup_temp and output_dir.exists():
            import shutil
            shutil.rmtree(output_dir)
            print(f"[smoke] Cleaned up temporary directory")


if __name__ == "__main__":
    raise SystemExit(main())

