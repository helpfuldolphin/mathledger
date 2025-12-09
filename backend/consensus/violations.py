"""
Consensus Violations Tracking & Reporting

Author: Manus-B (Ledger Replay Architect & Attestation Runtime Engineer)
Phase: III - Consensus Runtime Activation
Date: 2025-12-06

Purpose:
    Track, classify, and report consensus rule violations.
    
    Violations are:
    - Detected by validators
    - Classified by severity and type
    - Logged for audit trail
    - Reported for remediation

Design Principles:
    1. Comprehensive: All violations tracked
    2. Auditable: Full audit trail maintained
    3. Actionable: Clear remediation guidance
    4. Performant: Efficient storage and retrieval
"""

from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .rules import RuleViolation, RuleViolationType, RuleSeverity


# ============================================================================
# VIOLATION TRACKING
# ============================================================================

@dataclass
class ViolationRecord:
    """
    Comprehensive violation record for audit trail.
    
    Attributes:
        violation: Original RuleViolation
        detected_at: Timestamp when violation was detected
        detector: Name of validator that detected violation
        remediation_status: Current remediation status
        remediation_notes: Notes on remediation efforts
        metadata: Additional metadata
    """
    violation: RuleViolation
    detected_at: str
    detector: str
    remediation_status: str  # "pending" | "in_progress" | "resolved" | "ignored"
    remediation_notes: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "violation": self.violation.to_dict(),
            "detected_at": self.detected_at,
            "detector": self.detector,
            "remediation_status": self.remediation_status,
            "remediation_notes": self.remediation_notes,
            "metadata": self.metadata,
        }


class ViolationTracker:
    """
    Tracks consensus rule violations.
    
    Usage:
        tracker = ViolationTracker()
        tracker.record(violation, detector="BlockValidator")
        tracker.update_remediation(violation_id, "resolved", "Fixed by PR #123")
        report = tracker.generate_report()
    """
    
    def __init__(self):
        """Initialize violation tracker."""
        self.violations: List[ViolationRecord] = []
        self.violation_index: Dict[str, List[ViolationRecord]] = {}
    
    def record(
        self,
        violation: RuleViolation,
        detector: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ViolationRecord:
        """
        Record violation.
        
        Args:
            violation: RuleViolation to record
            detector: Name of validator that detected violation
            metadata: Additional metadata
        
        Returns:
            ViolationRecord
        """
        record = ViolationRecord(
            violation=violation,
            detected_at=datetime.utcnow().isoformat() + "Z",
            detector=detector,
            remediation_status="pending",
            remediation_notes="",
            metadata=metadata or {},
        )
        
        self.violations.append(record)
        
        # Index by violation type
        violation_type = violation.violation_type.value
        if violation_type not in self.violation_index:
            self.violation_index[violation_type] = []
        self.violation_index[violation_type].append(record)
        
        return record
    
    def update_remediation(
        self,
        violation_index: int,
        status: str,
        notes: str,
    ):
        """
        Update remediation status.
        
        Args:
            violation_index: Index of violation in violations list
            status: New remediation status
            notes: Remediation notes
        """
        if 0 <= violation_index < len(self.violations):
            self.violations[violation_index].remediation_status = status
            self.violations[violation_index].remediation_notes = notes
    
    def get_violations_by_type(self, violation_type: RuleViolationType) -> List[ViolationRecord]:
        """Get violations by type."""
        return self.violation_index.get(violation_type.value, [])
    
    def get_violations_by_severity(self, severity: RuleSeverity) -> List[ViolationRecord]:
        """Get violations by severity."""
        return [v for v in self.violations if v.violation.severity == severity]
    
    def get_violations_by_block(self, block_number: int) -> List[ViolationRecord]:
        """Get violations for specific block."""
        return [v for v in self.violations if v.violation.block_number == block_number]
    
    def get_pending_violations(self) -> List[ViolationRecord]:
        """Get violations with pending remediation."""
        return [v for v in self.violations if v.remediation_status == "pending"]
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate violation report.
        
        Returns:
            Dictionary with violation statistics and details
        """
        # Count by type
        by_type = {}
        for violation_type in RuleViolationType:
            count = len(self.get_violations_by_type(violation_type))
            if count > 0:
                by_type[violation_type.value] = count
        
        # Count by severity
        by_severity = {}
        for severity in RuleSeverity:
            count = len(self.get_violations_by_severity(severity))
            if count > 0:
                by_severity[severity.value] = count
        
        # Count by remediation status
        by_status = {}
        for record in self.violations:
            status = record.remediation_status
            by_status[status] = by_status.get(status, 0) + 1
        
        return {
            "total_violations": len(self.violations),
            "by_type": by_type,
            "by_severity": by_severity,
            "by_status": by_status,
            "pending_count": len(self.get_pending_violations()),
            "critical_count": len(self.get_violations_by_severity(RuleSeverity.CRITICAL)),
            "error_count": len(self.get_violations_by_severity(RuleSeverity.ERROR)),
            "warning_count": len(self.get_violations_by_severity(RuleSeverity.WARNING)),
        }


# ============================================================================
# VIOLATION CLASSIFICATION
# ============================================================================

def classify_violation_impact(violation: RuleViolation) -> str:
    """
    Classify violation impact.
    
    Args:
        violation: RuleViolation to classify
    
    Returns:
        Impact classification ("low" | "medium" | "high" | "critical")
    
    Classification Rules:
        - CRITICAL severity → critical impact
        - ERROR severity + chain-level → high impact
        - ERROR severity + block-level → medium impact
        - WARNING severity → low impact
    """
    if violation.severity == RuleSeverity.CRITICAL:
        return "critical"
    
    if violation.severity == RuleSeverity.ERROR:
        # Chain-level violations have high impact
        chain_level_types = [
            RuleViolationType.MONOTONICITY_VIOLATION,
            RuleViolationType.LINEAGE_BROKEN,
            RuleViolationType.FORK_DETECTED,
            RuleViolationType.CYCLE_DETECTED,
        ]
        if violation.violation_type in chain_level_types:
            return "high"
        else:
            return "medium"
    
    if violation.severity == RuleSeverity.WARNING:
        return "low"
    
    return "low"


def get_remediation_guidance(violation: RuleViolation) -> str:
    """
    Get remediation guidance for violation.
    
    Args:
        violation: RuleViolation
    
    Returns:
        Remediation guidance string
    """
    guidance_map = {
        RuleViolationType.INVALID_BLOCK_STRUCTURE: "Fix block structure: ensure all required fields are present and have correct types.",
        RuleViolationType.INVALID_ATTESTATION: "Recompute attestation roots: H_t = Hash(R_t || U_t). Verify hash algorithm matches hash_version.",
        RuleViolationType.INVALID_PREV_HASH: "Recompute prev_hash: Hash(predecessor.block_identity). Verify hash algorithm matches predecessor's hash_version.",
        RuleViolationType.INVALID_HASH_VERSION: "Verify hash_version in attestation_metadata. Supported: sha256-v1, dual-v1, sha3-v1.",
        RuleViolationType.MONOTONICITY_VIOLATION: "Investigate gap or duplicate in block_number sequence. Check for missing blocks or database corruption.",
        RuleViolationType.LINEAGE_BROKEN: "Verify prev_hash chain. Ensure each block references its predecessor correctly.",
        RuleViolationType.FORK_DETECTED: "Resolve fork using reorg policy (longest_chain or highest_attestation). Mark orphaned blocks.",
        RuleViolationType.CYCLE_DETECTED: "Investigate cycle in prev_hash chain. This indicates database corruption or malicious tampering.",
        RuleViolationType.ORPHAN_BLOCK: "Locate predecessor block. If missing, investigate database integrity.",
        RuleViolationType.INVALID_EPOCH_BOUNDARY: "Verify epoch boundary: start = epoch_number * epoch_size, end = (epoch_number + 1) * epoch_size.",
        RuleViolationType.INVALID_EPOCH_ROOT: "Recompute epoch root: MerkleRoot([H_0, ..., H_99]). Verify hash algorithm matches epoch hash_version.",
        RuleViolationType.INCOMPLETE_EPOCH: "Wait for epoch to complete (100 blocks) before sealing. Or reseal with partial epoch flag.",
        RuleViolationType.INVALID_HASH_TRANSITION: "Verify hash version transition follows PQ migration rules. Valid: sha256→dual, dual→sha3.",
        RuleViolationType.MISSING_DUAL_COMMITMENT: "Add composite_attestation_root_sha3 field for dual-commitment blocks.",
        RuleViolationType.PREMATURE_SHA3_ADOPTION: "Revert to dual-commitment phase before adopting pure SHA-3.",
    }
    
    return guidance_map.get(violation.violation_type, "No specific guidance available. Investigate violation context.")


# ============================================================================
# VIOLATION REPORTING
# ============================================================================

def generate_violation_summary(violations: List[RuleViolation]) -> str:
    """
    Generate human-readable violation summary.
    
    Args:
        violations: List of violations
    
    Returns:
        Summary string
    """
    if not violations:
        return "No violations detected."
    
    # Count by severity
    critical_count = sum(1 for v in violations if v.severity == RuleSeverity.CRITICAL)
    error_count = sum(1 for v in violations if v.severity == RuleSeverity.ERROR)
    warning_count = sum(1 for v in violations if v.severity == RuleSeverity.WARNING)
    
    # Count by type
    type_counts = {}
    for v in violations:
        type_name = v.violation_type.value
        type_counts[type_name] = type_counts.get(type_name, 0) + 1
    
    # Build summary
    summary = f"Total violations: {len(violations)}\n"
    summary += f"  Critical: {critical_count}\n"
    summary += f"  Errors: {error_count}\n"
    summary += f"  Warnings: {warning_count}\n"
    summary += "\nBy type:\n"
    for type_name, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        summary += f"  {type_name}: {count}\n"
    
    return summary


def generate_violation_report(
    violations: List[RuleViolation],
    include_guidance: bool = True,
) -> Dict[str, Any]:
    """
    Generate detailed violation report.
    
    Args:
        violations: List of violations
        include_guidance: Whether to include remediation guidance
    
    Returns:
        Report dictionary
    """
    report = {
        "summary": generate_violation_summary(violations),
        "total_violations": len(violations),
        "by_severity": {},
        "by_type": {},
        "violations": [],
    }
    
    # Count by severity
    for severity in RuleSeverity:
        count = sum(1 for v in violations if v.severity == severity)
        if count > 0:
            report["by_severity"][severity.value] = count
    
    # Count by type
    for violation_type in RuleViolationType:
        count = sum(1 for v in violations if v.violation_type == violation_type)
        if count > 0:
            report["by_type"][violation_type.value] = count
    
    # Add violation details
    for v in violations:
        violation_dict = v.to_dict()
        if include_guidance:
            violation_dict["remediation_guidance"] = get_remediation_guidance(v)
            violation_dict["impact"] = classify_violation_impact(v)
        report["violations"].append(violation_dict)
    
    return report


# ============================================================================
# VIOLATION EXPORT
# ============================================================================

def export_violations_csv(violations: List[RuleViolation], filepath: str):
    """
    Export violations to CSV file.
    
    Args:
        violations: List of violations
        filepath: Output CSV file path
    """
    import csv
    
    with open(filepath, 'w', newline='') as csvfile:
        fieldnames = [
            'violation_type',
            'severity',
            'block_number',
            'block_id',
            'message',
            'impact',
            'remediation_guidance',
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for v in violations:
            writer.writerow({
                'violation_type': v.violation_type.value,
                'severity': v.severity.value,
                'block_number': v.block_number or '',
                'block_id': v.block_id or '',
                'message': v.message,
                'impact': classify_violation_impact(v),
                'remediation_guidance': get_remediation_guidance(v),
            })


def export_violations_json(violations: List[RuleViolation], filepath: str):
    """
    Export violations to JSON file.
    
    Args:
        violations: List of violations
        filepath: Output JSON file path
    """
    import json
    
    report = generate_violation_report(violations, include_guidance=True)
    
    with open(filepath, 'w') as jsonfile:
        json.dump(report, jsonfile, indent=2)
