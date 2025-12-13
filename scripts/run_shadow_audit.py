#!/usr/bin/env python3
"""
run_shadow_audit.py v0.1 — Unified Shadow Audit Orchestrator

Orchestrates shadow log analysis and produces audit artifacts.

CANONICAL CONTRACT (v0.1):
  docs/system_law/calibration/RUN_SHADOW_AUDIT_V0_1_CONTRACT.md

FROZEN CLI:
  --input INPUT    (required) Input directory with shadow logs
  --output OUTPUT  (required) Output directory for results
  --seed SEED      (optional) Random seed for determinism
  --verbose, -v    (optional) Enable verbose output
  --dry-run        (optional) Validate without writing files

FROZEN EXIT CODES:
  0 = OK (completed successfully, including warnings)
  1 = FATAL (missing input, crash, exception)
  2 = RESERVED (unused in v0.1)

SHADOW MODE CONTRACT:
  - mode="SHADOW" in all outputs
  - schema_version="1.0.0"
  - shadow_mode_compliance.no_enforcement = true
  - No gating; all outputs are observational

Usage:
    # Basic run
    python scripts/run_shadow_audit.py \\
        --input results/first_light \\
        --output results/shadow_audit

    # With seed for reproducibility
    python scripts/run_shadow_audit.py \\
        --input results/first_light \\
        --output results/shadow_audit \\
        --seed 42

    # Dry-run validation
    python scripts/run_shadow_audit.py \\
        --input results/first_light \\
        --output results/shadow_audit \\
        --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# =============================================================================
# Constants
# =============================================================================

SCHEMA_VERSION = "1.0.0"
DEFAULT_OUTPUT_DIR = "results/shadow_audit"
DETERMINISTIC_TIMESTAMP = "1970-01-01T00:00:00+00:00"
DETERMINISTIC_DATESTAMP = "19700101_000000"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class DiscoveredInputs:
    """Container for discovered input artifacts."""
    p3_dir: Optional[Path] = None
    p4_dir: Optional[Path] = None
    p3_run_id: Optional[str] = None
    p4_run_id: Optional[str] = None
    evidence_pack_dir: Optional[Path] = None
    evidence_pack_found: bool = False


@dataclass
class StageResult:
    """Result of a single orchestration stage."""
    status: str  # "pass" | "warn" | "fail" | "skipped"
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


# =============================================================================
# Deterministic Helpers
# =============================================================================

def get_timestamp(deterministic: bool) -> str:
    """Get timestamp, fixed if deterministic mode."""
    if deterministic:
        return DETERMINISTIC_TIMESTAMP
    return datetime.now(timezone.utc).isoformat()


def generate_run_id(seed: Optional[int]) -> str:
    """
    Generate run ID per canonical format: sha_<seed>_<hash>.

    INV-04: Same seed produces identical run_id.
    When seed is provided, use hash of seed for determinism.
    When no seed, use timestamp.
    """
    if seed is not None:
        # Deterministic: hash-based suffix for reproducibility
        import hashlib
        hash_input = f"shadow_audit_seed_{seed}".encode()
        suffix = hashlib.sha256(hash_input).hexdigest()[:8]
        return f"sha_{seed}_{suffix}"
    else:
        # Non-deterministic: timestamp-based
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"sha_noseed_{timestamp}"


def _sort_dict_recursive(obj: Any) -> Any:
    """Recursively sort all dicts by keys for deterministic output."""
    if isinstance(obj, dict):
        return {k: _sort_dict_recursive(v) for k, v in sorted(obj.items())}
    elif isinstance(obj, list):
        return [_sort_dict_recursive(item) for item in obj]
    return obj


# =============================================================================
# Input Discovery Helpers
# =============================================================================

def find_run_dir(base_dir: Path, prefix: str) -> Optional[Path]:
    """Find the most recent run directory with given prefix."""
    if not base_dir.exists():
        return None
    dirs = list(base_dir.glob(f"{prefix}*"))
    if not dirs:
        return None
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs[0]


# =============================================================================
# Subprocess Helpers
# =============================================================================

def run_subprocess(
    cmd: List[str],
    cwd: Path,
    stage_name: str,
) -> tuple[bool, str, str]:
    """Run subprocess and return (success, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", f"{stage_name} timed out after 600s"
    except Exception as e:
        return False, "", f"{stage_name} failed: {e}"


# =============================================================================
# Stage: Discover Inputs (Manifest-First) - Canonical Single Input Directory
# =============================================================================

def discover_inputs(input_dir: Path) -> tuple[DiscoveredInputs, StageResult]:
    """
    Discover input artifacts using manifest-first extraction.

    CANONICAL: Single --input directory that contains shadow logs or
    P3/P4 subdirectories.

    Search order:
      1. Shadow log files directly in input_dir (*.jsonl)
      2. P3 artifacts in input_dir/p3 or input_dir containing fl_* dirs
      3. P4 artifacts in input_dir/p4 or input_dir containing p4_* dirs
      4. Evidence pack in input_dir/evidence_pack

    Returns (inputs, stage_result).
    """
    inputs = DiscoveredInputs()
    details: Dict[str, Any] = {"input_dir": str(input_dir)}

    # Look for shadow log files directly in input
    shadow_logs = list(input_dir.glob("*.jsonl")) + list(input_dir.glob("shadow_log*.json"))
    if shadow_logs:
        details["shadow_logs_found"] = len(shadow_logs)

    # Try P3 subdirectory first, then input_dir itself
    p3_candidates = [input_dir / "p3", input_dir]
    for p3_dir in p3_candidates:
        if p3_dir.exists():
            p3_run = find_run_dir(p3_dir, "fl_")
            if p3_run:
                inputs.p3_dir = p3_dir
                inputs.p3_run_id = p3_run.name
                details["p3_run_id"] = p3_run.name
                break
    if inputs.p3_run_id is None:
        details["p3_warning"] = "No fl_* directory found"

    # Try P4 subdirectory first, then input_dir itself
    p4_candidates = [input_dir / "p4", input_dir]
    for p4_dir in p4_candidates:
        if p4_dir.exists():
            p4_run = find_run_dir(p4_dir, "p4_")
            if p4_run:
                inputs.p4_dir = p4_dir
                inputs.p4_run_id = p4_run.name
                details["p4_run_id"] = p4_run.name
                break
    if inputs.p4_run_id is None:
        details["p4_warning"] = "No p4_* directory found"

    # Check evidence pack
    evidence_pack_dir = input_dir / "evidence_pack"
    if evidence_pack_dir.exists():
        manifest = evidence_pack_dir / "manifest.json"
        if manifest.exists():
            inputs.evidence_pack_dir = evidence_pack_dir
            inputs.evidence_pack_found = True
            details["evidence_pack_found"] = True
        else:
            details["evidence_pack_warning"] = "manifest.json not found in evidence_pack/"

    # Determine stage status - PASS if we found shadow logs or any P3/P4 artifacts
    has_shadow_logs = len(shadow_logs) > 0
    has_p3_or_p4 = inputs.p3_run_id is not None or inputs.p4_run_id is not None

    if not has_shadow_logs and not has_p3_or_p4:
        return inputs, StageResult(
            status="warn",
            details=details,
            error="No shadow logs or P3/P4 artifacts found",
        )

    status = "pass"
    return inputs, StageResult(status=status, details=details)


# =============================================================================
# Stage: Run P4 Harness
# =============================================================================

def run_p4_harness(
    cycles: int,
    seed: Optional[int],
    output_dir: Path,
    telemetry_adapter: str,
    project_root: Path,
) -> StageResult:
    """Run P4 harness subprocess."""
    cmd = [
        sys.executable,
        "scripts/usla_first_light_p4_harness.py",
        "--cycles", str(cycles),
        "--output-dir", str(output_dir),
        "--telemetry-adapter", telemetry_adapter,
    ]
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    success, stdout, stderr = run_subprocess(cmd, project_root, "p4_harness")

    if not success:
        return StageResult(status="warn", error=stderr or "P4 harness failed")

    # Extract summary from output
    p4_run = find_run_dir(output_dir, "p4_")
    details: Dict[str, Any] = {"output_dir": str(output_dir)}

    if p4_run:
        summary_path = p4_run / "p4_summary.json"
        if summary_path.exists():
            try:
                with open(summary_path, encoding="utf-8") as f:
                    summary = json.load(f)
                details["cycles_completed"] = summary.get("cycles_completed", cycles)
                details["twin_accuracy"] = summary.get("twin_accuracy", {}).get(
                    "success_prediction_accuracy", None
                )
                details["divergence_rate"] = summary.get("divergence_analysis", {}).get(
                    "divergence_rate", None
                )
            except (json.JSONDecodeError, OSError):
                pass

    return StageResult(status="pass", details=details)


# =============================================================================
# Stage: Build Evidence Pack
# =============================================================================

def build_evidence_pack(
    p3_dir: Path,
    p4_dir: Path,
    output_dir: Path,
    project_root: Path,
) -> StageResult:
    """Build evidence pack via subprocess."""
    cmd = [
        sys.executable,
        "scripts/build_first_light_evidence_pack.py",
        "--p3-dir", str(p3_dir),
        "--p4-dir", str(p4_dir),
        "--output-dir", str(output_dir),
    ]

    success, stdout, stderr = run_subprocess(cmd, project_root, "evidence_pack")

    if not success:
        return StageResult(status="warn", error=stderr or "Evidence pack build failed")

    # Extract manifest info
    manifest_path = output_dir / "manifest.json"
    details: Dict[str, Any] = {"output_dir": str(output_dir)}

    if manifest_path.exists():
        try:
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
            details["merkle_root"] = manifest.get("merkle_root")
            details["artifact_count"] = len(manifest.get("artifacts", []))
            advisories = manifest.get("governance_advisories", [])
            details["governance_advisories"] = len(advisories)
        except (json.JSONDecodeError, OSError):
            pass

    return StageResult(status="pass", details=details)


# =============================================================================
# Stage: Generate Status
# =============================================================================

def generate_status(
    p3_dir: Path,
    p4_dir: Path,
    evidence_pack_dir: Path,
    output_path: Path,
    project_root: Path,
) -> StageResult:
    """Generate first_light_status.json via subprocess."""
    cmd = [
        sys.executable,
        "scripts/generate_first_light_status.py",
        "--p3-dir", str(p3_dir),
        "--p4-dir", str(p4_dir),
        "--evidence-pack-dir", str(evidence_pack_dir),
        "--output", str(output_path),
    ]

    success, stdout, stderr = run_subprocess(cmd, project_root, "status")

    if not success:
        return StageResult(status="warn", error=stderr or "Status generation failed")

    details: Dict[str, Any] = {"output_path": str(output_path)}

    if output_path.exists():
        try:
            with open(output_path, encoding="utf-8") as f:
                status = json.load(f)
            details["overall_health"] = status.get("overall_health")
            details["nci_score"] = status.get("nci_health", {}).get("score")
            details["criteria_passed"] = status.get("criteria_summary", {}).get("all_passed")
        except (json.JSONDecodeError, OSError):
            pass

    return StageResult(status="pass", details=details)


# =============================================================================
# Stage: Generate Alignment View
# =============================================================================

def generate_alignment_view(
    p3_dir: Path,
    p4_dir: Path,
    evidence_pack_dir: Path,
    output_path: Path,
    p5_replay_logs: Optional[Path],
    project_root: Path,
) -> StageResult:
    """Generate first_light_alignment.json via subprocess."""
    cmd = [
        sys.executable,
        "scripts/generate_first_light_alignment_view.py",
        "--p3-dir", str(p3_dir),
        "--p4-dir", str(p4_dir),
        "--evidence-pack-dir", str(evidence_pack_dir),
        "--output", str(output_path),
    ]
    if p5_replay_logs:
        cmd.extend(["--p5-replay-logs", str(p5_replay_logs)])

    success, stdout, stderr = run_subprocess(cmd, project_root, "alignment_view")

    if not success:
        return StageResult(status="warn", error=stderr or "Alignment view generation failed")

    details: Dict[str, Any] = {"output_path": str(output_path)}

    if output_path.exists():
        try:
            with open(output_path, encoding="utf-8") as f:
                alignment = json.load(f)
            fusion = alignment.get("fusion_result", {})
            escalation = alignment.get("escalation", {})
            details["decision"] = fusion.get("decision")
            details["escalation_level"] = escalation.get("level_name")
            details["conflicts"] = len(alignment.get("conflict_detections", []))
        except (json.JSONDecodeError, OSError):
            pass

    return StageResult(status="pass", details=details)


# =============================================================================
# Run Summary Writer
# =============================================================================

def _write_summary(
    run_summary: Dict[str, Any],
    output_dir: Path,
    deterministic: bool,
) -> Path:
    """Write run_summary.json with optional deterministic formatting."""
    summary_path = output_dir / "run_summary.json"

    output = run_summary
    if deterministic:
        output = _sort_dict_recursive(run_summary)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, sort_keys=deterministic)

    return summary_path


def _write_first_light_status(
    run_summary: Dict[str, Any],
    output_dir: Path,
    deterministic: bool,
) -> Path:
    """Write first_light_status.json (MC-03 required output)."""
    status_path = output_dir / "first_light_status.json"

    # Extract relevant info for first_light_status
    status_data: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "mode": "SHADOW",
        "run_id": run_summary.get("run_id"),
        "status": run_summary.get("status", "OK"),
        "final_status": run_summary.get("final_status", "pass"),
        "generated_at": run_summary.get("finished_at"),
        "shadow_mode_compliance": run_summary.get("shadow_mode_compliance", {
            "no_enforcement": True,
            "observational_only": True,
            "schema_version": SCHEMA_VERSION,
        }),
        "summary": run_summary.get("summary", {}),
    }

    output = status_data
    if deterministic:
        output = _sort_dict_recursive(status_data)

    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, sort_keys=deterministic)

    return status_path


def compute_final_status(stages: Dict[str, StageResult], errors: List[str]) -> str:
    """Compute final status from stage results."""
    discover = stages.get("discover")
    if discover and discover.status == "fail":
        return "fail"

    has_warnings = bool(errors)
    has_warn_stages = any(s.status == "warn" for s in stages.values())

    if has_warnings or has_warn_stages:
        return "warn"
    return "pass"


# =============================================================================
# Main Orchestrator
# =============================================================================

def run_shadow_audit(args: argparse.Namespace) -> int:
    """
    Main orchestration entry point.

    CANONICAL EXIT CODES:
      0 = OK (completed successfully, including warnings)
      1 = FATAL (missing input, crash)
    """
    project_root = Path(__file__).parent.parent
    input_dir = Path(args.input)
    output_base = Path(args.output)
    use_deterministic = args.seed is not None

    # Generate run ID and output directory
    run_id = generate_run_id(args.seed)
    output_dir = output_base / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize run summary with SHADOW MODE markers
    run_summary: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "mode": "SHADOW",
        "enforcement": False,
        "started_at": get_timestamp(use_deterministic),
        "stages": {},
        "errors": [],
        "warnings": [],
        "divergences": [],
        "final_status": "pending",
        # Canonical shadow_mode_compliance block (MC-06)
        "shadow_mode_compliance": {
            "no_enforcement": True,
            "observational_only": True,
            "schema_version": SCHEMA_VERSION,
        },
    }

    stages: Dict[str, StageResult] = {}

    # -------------------------------------------------------------------------
    # STAGE 1: Discover inputs from single input directory
    # -------------------------------------------------------------------------
    if args.verbose:
        print(f"[shadow_audit] Discovering inputs from: {input_dir}")
    else:
        print(f"[shadow_audit] Discovering inputs...")

    inputs, discover_result = discover_inputs(input_dir)
    stages["discover"] = discover_result
    run_summary["stages"]["discover"] = {
        "status": discover_result.status,
        **discover_result.details,
    }
    if discover_result.error:
        run_summary["warnings"].append(f"discover: {discover_result.error}")

    if args.verbose:
        print(f"  P3 run: {inputs.p3_run_id or 'not found'}")
        print(f"  P4 run: {inputs.p4_run_id or 'not found'}")
        print(f"  Evidence pack: {'found' if inputs.evidence_pack_found else 'not found'}")

    # Resolve directories from discovered inputs
    p3_dir = inputs.p3_dir
    p4_dir = inputs.p4_dir
    evidence_pack_dir = inputs.evidence_pack_dir

    # -------------------------------------------------------------------------
    # STAGE 2: Run P4 harness (optional)
    # -------------------------------------------------------------------------
    if args.run_p4_harness:
        print(f"[shadow_audit] Running P4 harness ({args.p4_cycles} cycles)...")
        p4_result = run_p4_harness(
            cycles=args.p4_cycles,
            seed=args.seed,
            output_dir=output_dir / "p4_shadow",
            telemetry_adapter=args.telemetry_adapter,
            project_root=project_root,
        )
        stages["p4_harness"] = p4_result
        run_summary["stages"]["p4_harness"] = {
            "status": p4_result.status,
            **p4_result.details,
        }
        if p4_result.error:
            run_summary["warnings"].append(f"p4_harness: {p4_result.error}")

        # Update p4_dir to use new output
        if p4_result.status == "pass":
            p4_dir = output_dir / "p4_shadow"

        print(f"  Status: {p4_result.status}")
    else:
        run_summary["stages"]["p4_harness"] = {"status": "skipped"}

    # -------------------------------------------------------------------------
    # STAGE 3: Build evidence pack (optional)
    # -------------------------------------------------------------------------
    if args.build_evidence_pack:
        print(f"[shadow_audit] Building evidence pack...")
        if p3_dir and p4_dir:
            pack_result = build_evidence_pack(
                p3_dir=p3_dir,
                p4_dir=p4_dir,
                output_dir=output_dir / "evidence_pack",
                project_root=project_root,
            )
            stages["evidence_pack"] = pack_result
            run_summary["stages"]["evidence_pack"] = {
                "status": pack_result.status,
                **pack_result.details,
            }
            if pack_result.error:
                run_summary["warnings"].append(f"evidence_pack: {pack_result.error}")

            # Update evidence_pack_dir
            if pack_result.status == "pass":
                evidence_pack_dir = output_dir / "evidence_pack"

            print(f"  Status: {pack_result.status}")
        else:
            run_summary["stages"]["evidence_pack"] = {
                "status": "warn",
                "error": "Missing P3 or P4 directory",
            }
            run_summary["warnings"].append("evidence_pack: Missing P3 or P4 directory")
    else:
        run_summary["stages"]["evidence_pack"] = {"status": "skipped"}

    # Use existing or default evidence pack dir for status generation
    if evidence_pack_dir is None:
        evidence_pack_dir = output_dir / "evidence_pack"
        evidence_pack_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # STAGE 4: Generate status
    # -------------------------------------------------------------------------
    print(f"[shadow_audit] Generating status...")
    if p3_dir and p4_dir:
        status_result = generate_status(
            p3_dir=p3_dir,
            p4_dir=p4_dir,
            evidence_pack_dir=evidence_pack_dir,
            output_path=output_dir / "first_light_status.json",
            project_root=project_root,
        )
        stages["status"] = status_result
        run_summary["stages"]["status"] = {
            "status": status_result.status,
            **status_result.details,
        }
        if status_result.error:
            run_summary["warnings"].append(f"status: {status_result.error}")
        print(f"  Status: {status_result.status}")
    else:
        run_summary["stages"]["status"] = {
            "status": "warn",
            "error": "Missing P3 or P4 directory",
        }

    # -------------------------------------------------------------------------
    # STAGE 5: Generate alignment view (optional)
    # -------------------------------------------------------------------------
    if args.alignment_view:
        print(f"[shadow_audit] Generating alignment view...")
        if p3_dir and p4_dir:
            p5_logs = Path(args.p5_replay_logs) if args.p5_replay_logs else None
            alignment_result = generate_alignment_view(
                p3_dir=p3_dir,
                p4_dir=p4_dir,
                evidence_pack_dir=evidence_pack_dir,
                output_path=output_dir / "first_light_alignment.json",
                p5_replay_logs=p5_logs,
                project_root=project_root,
            )
            stages["alignment_view"] = alignment_result
            run_summary["stages"]["alignment_view"] = {
                "status": alignment_result.status,
                **alignment_result.details,
            }
            if alignment_result.error:
                run_summary["warnings"].append(f"alignment_view: {alignment_result.error}")
            print(f"  Status: {alignment_result.status}")
        else:
            run_summary["stages"]["alignment_view"] = {
                "status": "warn",
                "error": "Missing P3 or P4 directory",
            }
    else:
        run_summary["stages"]["alignment_view"] = {"status": "skipped"}

    # -------------------------------------------------------------------------
    # FINALIZE
    # -------------------------------------------------------------------------
    final_status = compute_final_status(stages, run_summary["errors"])
    run_summary["final_status"] = final_status
    run_summary["status"] = "WARN" if final_status == "warn" else "OK"
    run_summary["finished_at"] = get_timestamp(use_deterministic)

    # Summary statistics
    executed_stages = [s for s in stages.values() if s.status != "skipped"]
    run_summary["summary"] = {
        "total_stages": len(executed_stages),
        "passed_stages": len([s for s in stages.values() if s.status == "pass"]),
        "warned_stages": len([s for s in stages.values() if s.status == "warn"]),
        "failed_stages": len([s for s in stages.values() if s.status == "fail"]),
        "divergence_count": len(run_summary["divergences"]),
        "divergence_rate": 0.0,
    }

    summary_path = _write_summary(run_summary, output_dir, use_deterministic)

    # Also write first_light_status.json (canonical output MC-03)
    _write_first_light_status(run_summary, output_dir, use_deterministic)

    if args.verbose:
        print()
        print(f"[shadow_audit] Complete: {final_status.upper()}")
        print(f"  Output: {output_dir}")
        print(f"  Summary: {summary_path}")

    # CANONICAL: exit 0 for completed runs (including warnings)
    return 0


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    CANONICAL CLI (v0.1 - FROZEN):
      --input INPUT    (required)
      --output OUTPUT  (required)
      --seed SEED      (optional)
      --verbose, -v    (optional)
      --dry-run        (optional)
    """
    parser = argparse.ArgumentParser(
        description="SHADOW_AUDIT v0.1 — Shadow Log Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
CANONICAL CONTRACT (v0.1):
  CLI flags are FROZEN. See: docs/system_law/calibration/RUN_SHADOW_AUDIT_V0_1_CONTRACT.md

EXIT CODES:
  0 = OK (completed successfully, including warnings)
  1 = FATAL (missing input, crash, exception)
  2 = RESERVED (unused in v0.1)

SHADOW MODE:
  - mode="SHADOW" in all outputs
  - schema_version="1.0.0"
  - shadow_mode_compliance.no_enforcement = true

Examples:
  # Basic run
  python scripts/run_shadow_audit.py --input results/first_light --output results/shadow_audit

  # With seed for reproducibility
  python scripts/run_shadow_audit.py --input results/first_light --output results/shadow_audit --seed 42

  # Dry-run validation
  python scripts/run_shadow_audit.py --input results/first_light --output results/shadow_audit --dry-run
""",
    )

    # CANONICAL: --input (required)
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory containing shadow logs or P3/P4 artifacts",
    )

    # CANONICAL: --output (required)
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for results",
    )

    # CANONICAL: --seed (optional)
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for deterministic output (enables sorted keys, deterministic run_id)",
    )

    # CANONICAL: --verbose (optional)
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    # CANONICAL: --dry-run (optional)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print plan without writing files",
    )

    # Internal flags (not in canonical CLI, for backwards compatibility in orchestration)
    # These are not documented in --help and will be removed in future versions
    parser.add_argument("--run-p4-harness", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--p4-cycles", type=int, default=100, help=argparse.SUPPRESS)
    parser.add_argument("--telemetry-adapter", type=str, default="mock", help=argparse.SUPPRESS)
    parser.add_argument("--build-evidence-pack", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--alignment-view", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--p5-replay-logs", type=str, help=argparse.SUPPRESS)

    return parser.parse_args()


def main() -> int:
    """
    Main entry point.

    CANONICAL EXIT CODES:
      0 = OK (completed successfully, including warnings)
      1 = FATAL (missing input, crash, exception)
    """
    args = parse_args()
    input_dir = Path(args.input)

    # INV-07: Missing input directory → exit 1
    if not input_dir.exists():
        print(f"[shadow_audit] FATAL: Input directory does not exist: {input_dir}", file=sys.stderr)
        return 1

    if args.dry_run:
        # INV-05: --dry-run validates without writing files
        print("=" * 60)
        print("SHADOW AUDIT v0.1 — DRY RUN")
        print("=" * 60)
        print()
        print("Configuration:")
        print(f"  Input: {args.input}")
        print(f"  Output: {args.output}")
        print(f"  Seed: {args.seed}")
        print()
        print("Validation:")
        print(f"  Input exists: {input_dir.exists()}")
        shadow_logs = list(input_dir.glob("*.jsonl"))
        print(f"  Shadow logs found: {len(shadow_logs)}")
        print()
        print("Would produce artifacts:")
        run_id = generate_run_id(args.seed)
        print(f"  {args.output}/{run_id}/run_summary.json")
        print(f"  {args.output}/{run_id}/first_light_status.json")
        print()
        print("DRY RUN COMPLETE - no files written")
        return 0

    if args.verbose:
        print("=" * 60)
        print("SHADOW AUDIT v0.1")
        print("=" * 60)
        print()

    try:
        return run_shadow_audit(args)
    except Exception as e:
        print(f"[shadow_audit] FATAL ERROR: {e}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 1  # Exit 1 on exception


if __name__ == "__main__":
    sys.exit(main())
