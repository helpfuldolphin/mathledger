#!/usr/bin/env python3
"""
P5 Topology Auditor Report Generator

Generates a machine-readable P5 topology/bundle auditor report from existing
First Light artifacts and topology/bundle tile data.

SHADOW MODE CONTRACT:
- This script is purely observational
- It reads existing artifacts and generates auditor reports
- It does not run harnesses, tests, or modify governance decisions
- It does not execute enforcement logic

Usage:
    python scripts/generate_p5_topology_auditor_report.py \
        --p4-run-dir results/first_light/golden_run/p4/p4_20241211 \
        --evidence-pack-dir results/first_light/evidence_pack_first_light \
        --output results/first_light/evidence_pack_first_light/p5_topology_auditor_report.json

Output:
    p5_topology_auditor_report.json with:
    - scenario match (MOCK_BASELINE, HEALTHY, MISMATCH, XCOR_ANOMALY)
    - smoke validation results
    - 10-step auditor report
    - SHADOW MODE compliance markers

Deterministic Mode:
    For CI reproducibility, set P5_DETERMINISTIC_REPORTS=1 environment variable
    or pass --deterministic flag. Produces stable output with:
    - Sorted JSON keys
    - Fixed timestamp (1970-01-01T00:00:00+00:00)

    Precedence: CLI --deterministic flag overrides env var (--no-deterministic
    can explicitly disable even when env var is set).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Deterministic output support
DETERMINISTIC_TIMESTAMP = "1970-01-01T00:00:00+00:00"
DETERMINISTIC_ENV_VAR = "P5_DETERMINISTIC_REPORTS"


def _env_is_truthy(var_name: str) -> bool:
    """Check if an environment variable is set to a truthy value."""
    value = os.environ.get(var_name, "").strip().lower()
    return value in ("1", "true", "yes", "on")

from backend.health.p5_topology_reality_adapter import (
    P5_REALITY_ADAPTER_SCHEMA_VERSION,
    extract_topology_reality_metrics,
    validate_bundle_stability,
    detect_xcor_anomaly,
    run_p5_smoke_validation,
    match_p5_validation_scenario,
    build_p5_topology_reality_summary,
    generate_p5_auditor_report,
)
from backend.health.topology_bundle_adapter import (
    build_topology_bundle_console_tile,
    topology_bundle_to_governance_signal,
)

# Schema version for auditor report
P5_AUDITOR_REPORT_SCHEMA_VERSION = "1.0.0"


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_json_safe(file_path: Path) -> Optional[Dict[str, Any]]:
    """Load JSON file, returning None on error."""
    try:
        with open(file_path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def find_run_dir(base_dir: Path, prefix: str) -> Optional[Path]:
    """Find the most recent run directory with given prefix."""
    dirs = list(base_dir.glob(f"{prefix}*"))
    if not dirs:
        return None
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs[0]


def load_topology_bundle_tile_from_evidence(
    evidence_pack_dir: Path,
) -> Optional[Dict[str, Any]]:
    """
    Load topology_bundle tile from evidence pack manifest if present.

    The tile may be attached under:
    - manifest["governance"]["topology_bundle"]
    - Or in a standalone topology_bundle_tile.json file
    """
    # Try manifest first
    manifest_path = evidence_pack_dir / "manifest.json"
    if manifest_path.exists():
        manifest = load_json_safe(manifest_path)
        if manifest:
            governance = manifest.get("governance", {})
            tile = governance.get("topology_bundle")
            if tile:
                return tile

    # Try standalone file
    tile_path = evidence_pack_dir / "topology_bundle_tile.json"
    if tile_path.exists():
        return load_json_safe(tile_path)

    return None


def load_topology_bundle_from_run_dir(
    run_dir: Path,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Load topology bundle data from P4 run directory.

    Returns:
        (joint_view, consistency_result, director_panel) - any can be None
    """
    joint_view = None
    consistency_result = None
    director_panel = None

    # Try loading joint_view
    joint_view_path = run_dir / "topology_bundle_joint_view.json"
    if joint_view_path.exists():
        joint_view = load_json_safe(joint_view_path)

    # Try loading from p4_summary
    p4_summary_path = run_dir / "p4_summary.json"
    if p4_summary_path.exists():
        p4_summary = load_json_safe(p4_summary_path)
        if p4_summary:
            # Check for topology_bundle in governance section
            governance = p4_summary.get("governance", {})
            if not joint_view and "topology_bundle" in governance:
                topo_bundle = governance["topology_bundle"]
                if "joint_view" in topo_bundle:
                    joint_view = topo_bundle["joint_view"]

            # Extract consistency result
            consistency = p4_summary.get("cross_system_consistency", {})
            if consistency:
                consistency_result = consistency
            elif "consistency" in governance:
                consistency_result = governance["consistency"]

    # Try loading consistency result standalone
    consistency_path = run_dir / "cross_system_consistency.json"
    if not consistency_result and consistency_path.exists():
        consistency_result = load_json_safe(consistency_path)

    # Try loading director panel
    director_path = run_dir / "topology_bundle_director_panel.json"
    if director_path.exists():
        director_panel = load_json_safe(director_path)

    return joint_view, consistency_result, director_panel


def synthesize_joint_view_from_defaults(
    topology_mode: str = "STABLE",
    bundle_status: str = "VALID",
    alignment_status: str = "ALIGNED",
) -> Dict[str, Any]:
    """
    Synthesize a minimal joint_view from default parameters.

    Used when no topology_bundle data is available.
    """
    return {
        "schema_version": "1.0.0",
        "topology_snapshot": {
            "topology_mode": topology_mode,
            "betti_numbers": {"beta_0": 1, "beta_1": 0},
            "persistence_metrics": {"bottleneck_drift": 0.02},
            "safe_region_metrics": {"boundary_distance": 0.3},
        },
        "bundle_snapshot": {
            "bundle_status": bundle_status,
            "chain_info": {"chain_valid": bundle_status == "VALID"},
            "manifest": {"coverage": 1.0 if bundle_status == "VALID" else 0.5},
            "provenance": {"verified": bundle_status == "VALID"},
        },
        "alignment_status": {"overall_status": alignment_status},
    }


def synthesize_consistency_result(consistent: bool = True) -> Dict[str, Any]:
    """Synthesize a minimal consistency result."""
    return {
        "consistent": consistent,
        "status": "OK" if consistent else "WARN",
    }


def extract_run_id_and_slice(run_dir: Path) -> Tuple[str, str]:
    """Extract run_id and slice_name from run directory."""
    run_id = run_dir.name

    # Try to get slice from run_config
    config_path = run_dir / "run_config.json"
    slice_name = "unknown"
    if config_path.exists():
        config = load_json_safe(config_path)
        if config:
            slice_name = config.get("slice_name", config.get("slice", "unknown"))

    return run_id, slice_name


def _sort_dict_recursive(obj: Any) -> Any:
    """
    Recursively sort all dicts by keys for deterministic output.

    Handles nested dicts and lists.
    """
    if isinstance(obj, dict):
        return {k: _sort_dict_recursive(v) for k, v in sorted(obj.items())}
    elif isinstance(obj, list):
        return [_sort_dict_recursive(item) for item in obj]
    else:
        return obj


def generate_p5_topology_report(
    p4_run_dir: Optional[Path],
    evidence_pack_dir: Optional[Path],
    output_path: Path,
    force_scenario: Optional[str] = None,
    deterministic: bool = False,
) -> Dict[str, Any]:
    """
    Generate P5 topology auditor report.

    Args:
        p4_run_dir: Optional path to P4 run directory
        evidence_pack_dir: Optional path to evidence pack directory
        output_path: Path for output JSON
        force_scenario: Optional scenario to force for testing
        deterministic: If True, produce stable output (sorted keys, fixed timestamp)

    Returns:
        The generated report dict
    """
    errors: List[str] = []
    warnings: List[str] = []

    joint_view: Optional[Dict[str, Any]] = None
    consistency_result: Optional[Dict[str, Any]] = None
    director_panel: Optional[Dict[str, Any]] = None
    topology_tile: Optional[Dict[str, Any]] = None
    run_id = "unknown"
    slice_name = "unknown"

    # Strategy 1: Load from P4 run directory
    if p4_run_dir and p4_run_dir.exists():
        run_id, slice_name = extract_run_id_and_slice(p4_run_dir)
        joint_view, consistency_result, director_panel = load_topology_bundle_from_run_dir(
            p4_run_dir
        )

    # Strategy 2: Load from evidence pack
    if evidence_pack_dir and evidence_pack_dir.exists():
        # Try to load pre-built tile from evidence
        topology_tile = load_topology_bundle_tile_from_evidence(evidence_pack_dir)

        # If no tile but no joint_view either, check manifest for raw data
        if not topology_tile and not joint_view:
            manifest_path = evidence_pack_dir / "manifest.json"
            if manifest_path.exists():
                manifest = load_json_safe(manifest_path)
                if manifest:
                    evidence_data = manifest.get("evidence", {}).get("data", {})
                    if "topology_bundle_joint_view" in evidence_data:
                        joint_view = evidence_data["topology_bundle_joint_view"]

    # Strategy 3: Synthesize defaults if nothing found
    if not joint_view and not topology_tile:
        warnings.append(
            "No topology_bundle data found; synthesizing MOCK_BASELINE defaults"
        )
        # Default to MOCK_BASELINE scenario for P4 shadow mode
        joint_view = synthesize_joint_view_from_defaults(
            topology_mode="STABLE" if force_scenario == "HEALTHY" else "DRIFT",
            bundle_status="VALID" if force_scenario == "HEALTHY" else "ATTENTION",
            alignment_status="ALIGNED" if force_scenario == "HEALTHY" else "TENSION",
        )

    if not consistency_result:
        consistency_result = synthesize_consistency_result(
            consistent=(force_scenario == "HEALTHY")
        )

    # Build console tile if not already loaded
    if not topology_tile and joint_view:
        topology_tile = build_topology_bundle_console_tile(
            joint_view=joint_view,
            consistency_result=consistency_result,
            director_panel=director_panel,
        )

    if not topology_tile:
        errors.append("Failed to build or load topology_bundle tile")
        timestamp = DETERMINISTIC_TIMESTAMP if deterministic else datetime.now(timezone.utc).isoformat()
        return {
            "schema_version": P5_AUDITOR_REPORT_SCHEMA_VERSION,
            "mode": "SHADOW",
            "timestamp": timestamp,
            "success": False,
            "errors": errors,
            "warnings": warnings,
        }

    # Build governance signal
    governance_signal = None
    if joint_view:
        governance_signal = topology_bundle_to_governance_signal(
            joint_view=joint_view,
            consistency_result=consistency_result,
        )

    # Extract topology metrics
    topology_metrics = extract_topology_reality_metrics(
        topology_tile=topology_tile,
        joint_view=joint_view,
    )

    # Validate bundle stability
    bundle_validation = validate_bundle_stability(
        topology_tile=topology_tile,
        joint_view=joint_view,
        consistency_result=consistency_result,
    )

    # Detect XCOR anomalies
    xcor_detection = detect_xcor_anomaly(
        topology_tile=topology_tile,
        topology_metrics=topology_metrics,
        bundle_validation=bundle_validation,
    )

    # Run smoke validation
    smoke_result = run_p5_smoke_validation(
        topology_tile=topology_tile,
        consistency_result=consistency_result,
        joint_view=joint_view,
        scenario_override=force_scenario,
    )

    # Match scenario
    scenario_match = match_p5_validation_scenario(
        topology_tile=topology_tile,
        consistency_result=consistency_result,
    )

    # Build P5 summary
    bundle_tile = joint_view.get("bundle_snapshot", {}) if joint_view else {}
    p5_summary = build_p5_topology_reality_summary(
        topology_tile=topology_tile,
        bundle_tile=bundle_tile,
        replay_tile=None,  # Not available in this context
        telemetry_tile=None,  # Not available in this context
    )

    # Generate auditor report
    auditor_report = generate_p5_auditor_report(
        p5_summary=p5_summary,
        run_id=run_id,
        slice_name=slice_name,
    )

    # Build final output
    timestamp = DETERMINISTIC_TIMESTAMP if deterministic else datetime.now(timezone.utc).isoformat()
    report: Dict[str, Any] = {
        "schema_version": P5_AUDITOR_REPORT_SCHEMA_VERSION,
        "mode": "SHADOW",
        "timestamp": timestamp,
        "success": True,
        "run_context": {
            "run_id": run_id,
            "slice_name": slice_name,
            "p4_run_dir": str(p4_run_dir) if p4_run_dir else None,
            "evidence_pack_dir": str(evidence_pack_dir) if evidence_pack_dir else None,
        },
        # Core validation results
        "scenario_match": scenario_match,
        "smoke_validation": smoke_result,
        # Detailed analysis
        "topology_metrics": topology_metrics,
        "bundle_validation": bundle_validation,
        "xcor_detection": xcor_detection,
        # P5 summary and auditor report
        "p5_summary": p5_summary,
        "auditor_report": auditor_report,
        # Governance signal (if available)
        "governance_signal": governance_signal,
        # Input tile (for verification)
        "topology_tile": topology_tile,
        # Shadow mode invariants
        "shadow_mode_invariant_ok": smoke_result.get("shadow_mode_invariant_ok", True),
        # Diagnostics
        "errors": errors if errors else None,
        "warnings": warnings if warnings else None,
    }

    # Apply deterministic ordering if requested
    if deterministic:
        report = _sort_dict_recursive(report)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=deterministic)

    return report


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate P5 Topology Auditor Report from First Light artifacts"
    )
    parser.add_argument(
        "--p4-run-dir",
        type=str,
        help="Path to P4 run directory (e.g., results/first_light/p4/p4_20241211)",
    )
    parser.add_argument(
        "--evidence-pack-dir",
        type=str,
        help="Path to evidence pack directory (fallback for loading topology tile)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="p5_topology_auditor_report.json",
        help="Output path for report JSON (default: p5_topology_auditor_report.json)",
    )
    parser.add_argument(
        "--force-scenario",
        type=str,
        choices=["MOCK_BASELINE", "HEALTHY", "MISMATCH", "XCOR_ANOMALY"],
        help="Force a specific scenario match (for testing)",
    )
    # Deterministic mode: CLI flags override env var
    # --deterministic enables, --no-deterministic disables (even if env var set)
    det_group = parser.add_mutually_exclusive_group()
    det_group.add_argument(
        "--deterministic",
        action="store_true",
        default=None,
        help=(
            "Produce deterministic output (sorted keys, fixed timestamp). "
            f"Also enabled by {DETERMINISTIC_ENV_VAR}=1 env var."
        ),
    )
    det_group.add_argument(
        "--no-deterministic",
        action="store_true",
        default=None,
        help=f"Disable deterministic output (overrides {DETERMINISTIC_ENV_VAR} env var)",
    )

    args = parser.parse_args()

    # Resolve deterministic mode: CLI takes precedence over env var
    if args.no_deterministic:
        deterministic = False
    elif args.deterministic:
        deterministic = True
    else:
        # Fall back to env var
        deterministic = _env_is_truthy(DETERMINISTIC_ENV_VAR)

    # Validate at least one input source
    if not args.p4_run_dir and not args.evidence_pack_dir:
        print(
            "ERROR: At least one of --p4-run-dir or --evidence-pack-dir must be provided"
        )
        return 1

    p4_run_dir = Path(args.p4_run_dir) if args.p4_run_dir else None
    evidence_pack_dir = Path(args.evidence_pack_dir) if args.evidence_pack_dir else None
    output_path = Path(args.output)

    # Validate directories exist
    if p4_run_dir and not p4_run_dir.exists():
        print(f"ERROR: P4 run directory does not exist: {p4_run_dir}")
        return 1

    if evidence_pack_dir and not evidence_pack_dir.exists():
        print(f"ERROR: Evidence pack directory does not exist: {evidence_pack_dir}")
        return 1

    # Generate report
    report = generate_p5_topology_report(
        p4_run_dir=p4_run_dir,
        evidence_pack_dir=evidence_pack_dir,
        output_path=output_path,
        force_scenario=args.force_scenario,
        deterministic=deterministic,
    )

    # Print summary
    print("=" * 60)
    print("P5 Topology Auditor Report Generation")
    print("=" * 60)
    print()
    print(f"P4 Run Dir: {p4_run_dir}")
    print(f"Evidence Pack: {evidence_pack_dir}")
    print(f"Output: {output_path}")
    print()
    print("Report Summary:")
    print(f"  success: {report.get('success')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  shadow_mode_invariant_ok: {report.get('shadow_mode_invariant_ok')}")
    print()

    scenario = report.get("scenario_match", {})
    print(f"Scenario Match:")
    print(f"  scenario: {scenario.get('scenario')}")
    print(f"  confidence: {scenario.get('confidence')}")
    print()

    smoke = report.get("smoke_validation", {})
    print(f"Smoke Validation:")
    print(f"  matched_scenario: {smoke.get('matched_scenario')}")
    print(f"  validation_passed: {smoke.get('validation_passed')}")
    print()

    if report.get("warnings"):
        print("Warnings:")
        for warning in report["warnings"]:
            print(f"  - {warning}")
        print()

    if report.get("errors"):
        print("Errors:")
        for error in report["errors"]:
            print(f"  - {error}")
        print()

    print("=" * 60)
    print(f"Report written to: {output_path}")
    print("=" * 60)

    return 0 if report.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
