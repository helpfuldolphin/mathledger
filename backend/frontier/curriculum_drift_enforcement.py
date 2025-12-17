# REAL-READY
"""
Curriculum Drift Enforcement Module

Provides runtime drift detection and governance signal emission (BLOCK/WARN)
for curriculum configuration changes. Integrates with CurriculumDriftSentinel
and implements the governance spec defined in docs/curriculum_drift_signals.md.

Author: MANUS-E, Curriculum Integrity Engineer
Status: OPERATIONAL
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from backend.frontier.curriculum import (
    CurriculumSystem,
    CurriculumDriftSentinel,
    CurriculumDriftError,
    load as load_curriculum
)


@dataclass
class DriftReport:
    """Structured drift report for audit trail."""
    timestamp: str
    curriculum_slug: str
    expected_fingerprint: str
    observed_fingerprint: str
    violations: List[str]
    governance_signal: str  # "BLOCK" or "WARN"
    recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


def emit_drift_signal(
    signal: str,
    violations: List[str],
    curriculum_slug: str,
    expected_fingerprint: str,
    observed_fingerprint: str,
    artifact_dir: Optional[Path] = None
) -> None:
    """
    Emit a drift governance signal (BLOCK or WARN).
    
    Args:
        signal: "BLOCK" or "WARN"
        violations: List of violation messages
        curriculum_slug: Curriculum system slug (e.g., "pl")
        expected_fingerprint: Baseline curriculum fingerprint
        observed_fingerprint: Current curriculum fingerprint
        artifact_dir: Directory to write drift_report.json (if None, uses current dir)
    
    Raises:
        CurriculumDriftError: If signal is "BLOCK"
    """
    # Create drift report
    report = DriftReport(
        timestamp=datetime.utcnow().isoformat() + "Z",
        curriculum_slug=curriculum_slug,
        expected_fingerprint=expected_fingerprint,
        observed_fingerprint=observed_fingerprint,
        violations=violations,
        governance_signal=signal,
        recommendation=(
            "Revert curriculum changes or update baseline fingerprint."
            if signal == "BLOCK"
            else "Review discrepancies and remediate upstream sources."
        )
    )
    
    # Write drift report to artifact directory
    if artifact_dir is None:
        artifact_dir = Path.cwd()
    else:
        artifact_dir = Path(artifact_dir)
    
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_path = artifact_dir / "drift_report.json"
    
    with open(report_path, 'w') as f:
        json.dump(report.to_dict(), f, indent=2, sort_keys=True)
    
    # Emit signal
    if signal == "BLOCK":
        error_msg = (
            f"Curriculum drift detected (BLOCK signal):\n"
            + "\n".join(f"  - {v}" for v in violations)
            + f"\n\nDrift report written to: {report_path}"
        )
        raise CurriculumDriftError(error_msg)
    elif signal == "WARN":
        warning_msg = (
            f"WARNING: Curriculum drift detected (WARN signal):\n"
            + "\n".join(f"  - {v}" for v in violations)
            + f"\n\nDrift report written to: {report_path}"
        )
        print(warning_msg, file=sys.stderr)
    else:
        raise ValueError(f"Invalid governance signal: {signal}. Must be 'BLOCK' or 'WARN'.")


def check_curriculum_drift(
    curriculum_slug: str,
    baseline_fingerprint: str,
    artifact_dir: Optional[Path] = None,
    signal_mode: str = "BLOCK"
) -> bool:
    """
    Check for curriculum drift and emit governance signal.
    
    Args:
        curriculum_slug: Curriculum system slug (e.g., "pl")
        baseline_fingerprint: Expected curriculum fingerprint
        artifact_dir: Directory to write drift_report.json
        signal_mode: "BLOCK" (fail-closed) or "WARN" (log-only)
    
    Returns:
        True if no drift detected, False if drift detected (WARN mode only)
    
    Raises:
        CurriculumDriftError: If drift detected in BLOCK mode
    """
    # Load current curriculum
    current_system = load_curriculum(curriculum_slug)
    
    # Create sentinel with baseline
    sentinel = CurriculumDriftSentinel(
        baseline_fingerprint=baseline_fingerprint,
        baseline_version=current_system.version,
        baseline_slice_count=len(current_system.slices)
    )
    
    # Check for violations
    violations = sentinel.check(current_system)
    
    if violations:
        # Emit governance signal
        emit_drift_signal(
            signal=signal_mode,
            violations=violations,
            curriculum_slug=curriculum_slug,
            expected_fingerprint=baseline_fingerprint,
            observed_fingerprint=current_system.fingerprint(),
            artifact_dir=artifact_dir
        )
        return False
    
    return True


def stamp_run_ledger_with_fingerprint(
    ledger_entry: Dict[str, Any],
    curriculum_slug: str
) -> Dict[str, Any]:
    """
    Stamp a RunLedgerEntry with curriculum fingerprint for provenance.
    
    Args:
        ledger_entry: Dictionary representing a RunLedgerEntry
        curriculum_slug: Curriculum system slug (e.g., "pl")
    
    Returns:
        Updated ledger_entry with curriculum_fingerprint and curriculum_slug fields
    """
    system = load_curriculum(curriculum_slug)
    
    ledger_entry["curriculum_slug"] = curriculum_slug
    ledger_entry["curriculum_fingerprint"] = system.fingerprint()
    
    return ledger_entry


# CLI interface for drift checking
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Check curriculum drift and emit governance signals"
    )
    parser.add_argument(
        "curriculum_slug",
        help="Curriculum system slug (e.g., 'pl')"
    )
    parser.add_argument(
        "baseline_fingerprint",
        help="Expected curriculum fingerprint (64-char hex SHA-256)"
    )
    parser.add_argument(
        "--artifact-dir",
        default=".",
        help="Directory to write drift_report.json (default: current directory)"
    )
    parser.add_argument(
        "--mode",
        choices=["BLOCK", "WARN"],
        default="BLOCK",
        help="Governance signal mode (default: BLOCK)"
    )
    
    args = parser.parse_args()
    
    try:
        no_drift = check_curriculum_drift(
            curriculum_slug=args.curriculum_slug,
            baseline_fingerprint=args.baseline_fingerprint,
            artifact_dir=Path(args.artifact_dir),
            signal_mode=args.mode
        )
        
        if no_drift:
            print(f"[PASS] No curriculum drift detected for '{args.curriculum_slug}'")
            sys.exit(0)
        else:
            # WARN mode: drift detected but not blocking
            print(f"[WARN] Curriculum drift detected for '{args.curriculum_slug}' (non-blocking)")
            sys.exit(1)
    
    except CurriculumDriftError as e:
        # BLOCK mode: drift detected and blocking
        print(f"[BLOCK] {e}", file=sys.stderr)
        sys.exit(2)
