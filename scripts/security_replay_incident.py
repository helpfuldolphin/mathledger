#!/usr/bin/env python3
"""
security_replay_incident.py - Replay Failure Incident Classification

PHASE II -- NOT RUN IN PHASE I

Implements the REPLAY_FAILURE incident response from U2_SECURITY_PLAYBOOK.md.
Ingests replay receipts and manifests, classifies incidents, and outputs
structured incident reports.

This tool is diagnostic only - no automatic repair.
"""

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional


class ReplayStatus(Enum):
    FULL_MATCH = "FULL_MATCH"
    PARTIAL_MATCH = "PARTIAL_MATCH"
    NO_MATCH = "NO_MATCH"


class Severity(Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class Admissibility(Enum):
    ADMISSIBLE = "admissible"
    CONDITIONAL = "conditional"
    REDUCED = "reduced"
    INADMISSIBLE = "inadmissible"


class UpliftClaimStatus(Enum):
    VALID = "valid"
    VALID_WITH_CAVEAT = "valid_with_caveat"
    WEAKENED = "weakened"
    INVALID = "invalid"


@dataclass
class ReplayIncidentReport:
    """Structured replay incident report per playbook specification."""

    incident_id: str
    incident_type: str
    run_id: str
    detected_at: str

    # Replay analysis
    replay_status: str
    cycles_total: int
    cycles_matched: int
    match_percentage: float
    divergence_point: Optional[int]
    divergence_type: Optional[str]

    # Classification per playbook matrix
    severity: str
    admissibility: str
    uplift_claim_status: str

    # Recommended actions from playbook
    recommended_actions: list

    # Artifact references
    artifacts: dict

    # Metadata
    analyzer_version: str = "1.0.0"
    playbook_reference: str = "U2_SECURITY_PLAYBOOK.md#replay-failure-incident-response"


def compute_file_hash(filepath: Path) -> str:
    """Compute SHA256 hash of file contents."""
    if not filepath.exists():
        return "FILE_NOT_FOUND"
    return hashlib.sha256(filepath.read_bytes()).hexdigest()


def generate_incident_id(run_id: str, timestamp: datetime) -> str:
    """Generate deterministic incident ID."""
    ts_str = timestamp.strftime("%Y%m%d_%H%M%S")
    return f"REPLAY_{ts_str}_{run_id[:8]}"


def classify_replay_status(cycles_matched: int, cycles_total: int, divergence_point: Optional[int]) -> ReplayStatus:
    """
    Classify replay status per playbook matrix.

    | Replay Status | Run Admissibility | Uplift Claim Status |
    |---------------|-------------------|---------------------|
    | FULL MATCH | Admissible | Valid |
    | PARTIAL MATCH (>80% cycles) | Conditionally admissible | Valid with caveat |
    | PARTIAL MATCH (50-80% cycles) | Reduced admissibility | Weakened claim |
    | PARTIAL MATCH (<50% cycles) | Inadmissible | Invalid |
    | NO MATCH (cycle 0 divergence) | Inadmissible | Invalid |
    """
    if cycles_total == 0:
        return ReplayStatus.NO_MATCH

    if divergence_point == 0:
        return ReplayStatus.NO_MATCH

    if cycles_matched == cycles_total:
        return ReplayStatus.FULL_MATCH

    return ReplayStatus.PARTIAL_MATCH


def determine_severity(divergence_point: Optional[int], cycles_total: int) -> Severity:
    """
    Determine severity per playbook triage matrix.

    | Divergence Point | Scope | Severity | Response Time |
    |------------------|-------|----------|---------------|
    | Cycle 0-5 | Total | CRITICAL | Immediate |
    | Cycle 6-N/2 | Majority | HIGH | < 1 hour |
    | Cycle > N/2 | Minority | MEDIUM | < 4 hours |
    | Final cycle only | Minimal | LOW | < 24 hours |
    """
    if divergence_point is None:
        return Severity.LOW  # No divergence

    if divergence_point <= 5:
        return Severity.CRITICAL

    half_cycles = cycles_total // 2

    if divergence_point <= half_cycles:
        return Severity.HIGH

    if divergence_point == cycles_total - 1:
        return Severity.LOW

    return Severity.MEDIUM


def determine_admissibility(replay_status: ReplayStatus, match_percentage: float) -> tuple[Admissibility, UpliftClaimStatus]:
    """
    Determine admissibility and uplift claim status per playbook matrix.
    """
    if replay_status == ReplayStatus.FULL_MATCH:
        return Admissibility.ADMISSIBLE, UpliftClaimStatus.VALID

    if replay_status == ReplayStatus.NO_MATCH:
        return Admissibility.INADMISSIBLE, UpliftClaimStatus.INVALID

    # PARTIAL_MATCH - graduated by percentage
    if match_percentage > 80:
        return Admissibility.CONDITIONAL, UpliftClaimStatus.VALID_WITH_CAVEAT
    elif match_percentage >= 50:
        return Admissibility.REDUCED, UpliftClaimStatus.WEAKENED
    else:
        return Admissibility.INADMISSIBLE, UpliftClaimStatus.INVALID


def get_recommended_actions(severity: Severity, replay_status: ReplayStatus, admissibility: Admissibility) -> list[str]:
    """
    Generate recommended actions per playbook guidance.
    """
    actions = []

    # Immediate actions based on severity
    if severity == Severity.CRITICAL:
        actions.append("IMMEDIATE: Halt any ongoing runs")
        actions.append("IMMEDIATE: Preserve all artifacts before remediation")
        actions.append("Create forensic bundle with collection script")
    elif severity == Severity.HIGH:
        actions.append("HIGH PRIORITY: Investigate within 1 hour")
        actions.append("Preserve artifacts for analysis")
    elif severity == Severity.MEDIUM:
        actions.append("Schedule investigation within 4 hours")
    else:
        actions.append("Document for review within 24 hours")

    # Status-specific actions
    if replay_status == ReplayStatus.NO_MATCH:
        actions.append("Check initialization state for cycle 0 divergence")
        actions.append("Verify seed values match between original and replay")
        actions.append("Consider SEED_DRIFT diagnosis path")
    elif replay_status == ReplayStatus.PARTIAL_MATCH:
        actions.append("Identify exact divergence point using diff analysis")
        actions.append("Determine if divergence is seed drift or substrate nondeterminism")
        actions.append("Run security_seed_drift_analysis.py for classification")

    # Admissibility-specific actions
    if admissibility == Admissibility.INADMISSIBLE:
        actions.append("Mark run as INVALIDATED in manifest")
        actions.append("Do not include in uplift claims")
        actions.append("Archive with invalidation documentation")
    elif admissibility == Admissibility.CONDITIONAL:
        actions.append("Document caveat in run manifest")
        actions.append("Include match percentage in any claims")
    elif admissibility == Admissibility.REDUCED:
        actions.append("Document reduced admissibility status")
        actions.append("Consider re-running experiment if feasible")

    return actions


def load_replay_receipt(filepath: Path) -> dict:
    """Load and validate replay receipt JSON."""
    if not filepath.exists():
        raise FileNotFoundError(f"Replay receipt not found: {filepath}")

    with open(filepath) as f:
        receipt = json.load(f)

    # Validate required fields
    required_fields = ["status", "cycles_replayed", "cycles_matched"]
    for field in required_fields:
        if field not in receipt:
            raise ValueError(f"Replay receipt missing required field: {field}")

    return receipt


def load_manifest(filepath: Path) -> dict:
    """Load manifest file (YAML or JSON)."""
    if not filepath.exists():
        raise FileNotFoundError(f"Manifest not found: {filepath}")

    content = filepath.read_text()

    # Try JSON first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try YAML
    try:
        import yaml
        return yaml.safe_load(content)
    except ImportError:
        # YAML not available, try simple key: value parsing
        result = {}
        for line in content.splitlines():
            if ':' in line and not line.strip().startswith('#'):
                key, _, value = line.partition(':')
                result[key.strip()] = value.strip()
        return result


def analyze_replay_incident(
    replay_receipt: dict,
    primary_manifest: dict,
    replay_manifest: Optional[dict] = None,
    run_id: Optional[str] = None
) -> ReplayIncidentReport:
    """
    Analyze replay receipt and classify incident per playbook.

    This is the main analysis function that implements the playbook's
    incident classification logic.
    """
    # Extract run ID
    if run_id is None:
        run_id = primary_manifest.get("run_id", replay_receipt.get("run_id", "UNKNOWN"))

    # Generate deterministic timestamp and incident ID
    timestamp = datetime.now(timezone.utc)
    incident_id = generate_incident_id(run_id, timestamp)

    # Extract replay metrics
    cycles_total = replay_receipt.get("cycles_replayed", 0)
    cycles_matched = replay_receipt.get("cycles_matched", 0)
    divergence_point = replay_receipt.get("divergence_point")
    divergence_type = replay_receipt.get("divergence_type")

    # Calculate match percentage
    match_percentage = (cycles_matched / cycles_total * 100) if cycles_total > 0 else 0.0

    # Classify per playbook
    replay_status = classify_replay_status(cycles_matched, cycles_total, divergence_point)
    severity = determine_severity(divergence_point, cycles_total)
    admissibility, uplift_claim_status = determine_admissibility(replay_status, match_percentage)

    # Get recommended actions
    recommended_actions = get_recommended_actions(severity, replay_status, admissibility)

    # Collect artifact references
    artifacts = {
        "primary_manifest": primary_manifest.get("_source_path", "unknown"),
        "replay_receipt": replay_receipt.get("_source_path", "unknown"),
        "replay_manifest": replay_manifest.get("_source_path", "unknown") if replay_manifest else None,
    }

    return ReplayIncidentReport(
        incident_id=incident_id,
        incident_type="REPLAY_FAILURE",
        run_id=run_id,
        detected_at=timestamp.isoformat(),
        replay_status=replay_status.value,
        cycles_total=cycles_total,
        cycles_matched=cycles_matched,
        match_percentage=round(match_percentage, 2),
        divergence_point=divergence_point,
        divergence_type=divergence_type,
        severity=severity.value,
        admissibility=admissibility.value,
        uplift_claim_status=uplift_claim_status.value,
        recommended_actions=recommended_actions,
        artifacts=artifacts,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Replay Failure Incident Classification Tool",
        epilog="See U2_SECURITY_PLAYBOOK.md for incident response procedures."
    )
    parser.add_argument(
        "--replay-receipt",
        type=Path,
        required=True,
        help="Path to determinism_replay_receipt.json"
    )
    parser.add_argument(
        "--primary-manifest",
        type=Path,
        required=True,
        help="Path to primary run manifest"
    )
    parser.add_argument(
        "--replay-manifest",
        type=Path,
        help="Path to replay run manifest (optional)"
    )
    parser.add_argument(
        "--run-id",
        help="Override run ID (optional)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("replay_incident_report.json"),
        help="Output path for incident report"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output"
    )

    args = parser.parse_args()

    try:
        # Load inputs
        replay_receipt = load_replay_receipt(args.replay_receipt)
        replay_receipt["_source_path"] = str(args.replay_receipt)

        primary_manifest = load_manifest(args.primary_manifest)
        primary_manifest["_source_path"] = str(args.primary_manifest)

        replay_manifest = None
        if args.replay_manifest:
            replay_manifest = load_manifest(args.replay_manifest)
            replay_manifest["_source_path"] = str(args.replay_manifest)

        # Analyze
        report = analyze_replay_incident(
            replay_receipt=replay_receipt,
            primary_manifest=primary_manifest,
            replay_manifest=replay_manifest,
            run_id=args.run_id
        )

        # Output
        report_dict = asdict(report)

        with open(args.output, 'w') as f:
            json.dump(report_dict, f, indent=2)

        if not args.quiet:
            print(f"Replay Incident Report Generated: {args.output}")
            print(f"  Incident ID: {report.incident_id}")
            print(f"  Status: {report.replay_status}")
            print(f"  Severity: {report.severity}")
            print(f"  Admissibility: {report.admissibility}")
            print(f"  Match: {report.cycles_matched}/{report.cycles_total} ({report.match_percentage}%)")
            if report.divergence_point is not None:
                print(f"  Divergence at cycle: {report.divergence_point}")

        # Exit code based on admissibility
        if report.admissibility == Admissibility.INADMISSIBLE.value:
            sys.exit(2)
        elif report.admissibility in [Admissibility.CONDITIONAL.value, Admissibility.REDUCED.value]:
            sys.exit(1)
        else:
            sys.exit(0)

    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(3)
    except ValueError as e:
        print(f"ERROR: Invalid input - {e}", file=sys.stderr)
        sys.exit(3)
    except Exception as e:
        print(f"ERROR: Unexpected error - {e}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
