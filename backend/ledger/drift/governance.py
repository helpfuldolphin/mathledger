"""
Drift Radar → Governance Adaptor

Maps drift severity levels to governance signals for CI/CD enforcement.

Author: Manus-B (Ledger Replay Architect & PQ Migration Officer)
Phase: IV - Consensus Integration & Enforcement
Date: 2025-12-09
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime


class GovernanceSignal(Enum):
    """Governance signal types."""
    OK = "OK"              # No action required, proceed
    WARN = "WARN"          # Warning issued, proceed with caution
    BLOCK = "BLOCK"        # Block merge, manual review required
    EMERGENCY = "EMERGENCY"  # Emergency stop, rollback required


class DriftSeverity(Enum):
    """Drift severity levels (from drift radar)."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class GovernancePolicy:
    """
    Governance policy mapping drift severity to governance signals.
    
    Attributes:
        name: Policy name
        severity_thresholds: Dict mapping severity to max allowed count
        default_signal: Default signal if no thresholds exceeded
        block_on_critical: Block on any CRITICAL drift
        warn_on_high: Warn on any HIGH drift
    """
    name: str
    severity_thresholds: Dict[DriftSeverity, int]
    default_signal: GovernanceSignal = GovernanceSignal.OK
    block_on_critical: bool = True
    warn_on_high: bool = True
    
    def evaluate(self, drift_counts: Dict[DriftSeverity, int]) -> GovernanceSignal:
        """
        Evaluate governance signal based on drift counts.
        
        Args:
            drift_counts: Dict mapping severity to count
        
        Returns:
            GovernanceSignal
        """
        # Check CRITICAL threshold
        if self.block_on_critical and drift_counts.get(DriftSeverity.CRITICAL, 0) > 0:
            return GovernanceSignal.BLOCK
        
        # Check HIGH threshold
        if self.warn_on_high and drift_counts.get(DriftSeverity.HIGH, 0) > 0:
            # Upgrade to BLOCK if HIGH count exceeds threshold
            if drift_counts.get(DriftSeverity.HIGH, 0) > self.severity_thresholds.get(DriftSeverity.HIGH, 0):
                return GovernanceSignal.BLOCK
            return GovernanceSignal.WARN
        
        # Check MEDIUM threshold
        if drift_counts.get(DriftSeverity.MEDIUM, 0) > self.severity_thresholds.get(DriftSeverity.MEDIUM, 5):
            return GovernanceSignal.WARN
        
        # Check LOW threshold (informational)
        if drift_counts.get(DriftSeverity.LOW, 0) > self.severity_thresholds.get(DriftSeverity.LOW, 10):
            return GovernanceSignal.WARN
        
        return self.default_signal


# Predefined governance policies
STRICT_POLICY = GovernancePolicy(
    name="strict",
    severity_thresholds={
        DriftSeverity.CRITICAL: 0,
        DriftSeverity.HIGH: 0,
        DriftSeverity.MEDIUM: 2,
        DriftSeverity.LOW: 5,
    },
    block_on_critical=True,
    warn_on_high=True,
)

MODERATE_POLICY = GovernancePolicy(
    name="moderate",
    severity_thresholds={
        DriftSeverity.CRITICAL: 0,
        DriftSeverity.HIGH: 2,
        DriftSeverity.MEDIUM: 5,
        DriftSeverity.LOW: 10,
    },
    block_on_critical=True,
    warn_on_high=False,
)

PERMISSIVE_POLICY = GovernancePolicy(
    name="permissive",
    severity_thresholds={
        DriftSeverity.CRITICAL: 1,
        DriftSeverity.HIGH: 5,
        DriftSeverity.MEDIUM: 10,
        DriftSeverity.LOW: 20,
    },
    block_on_critical=False,
    warn_on_high=False,
)


@dataclass
class EvidencePack:
    """
    Evidence pack for governance decision.
    
    Attributes:
        signal: Governance signal (OK/WARN/BLOCK/EMERGENCY)
        drift_counts: Drift counts by severity
        drift_signals: List of drift signals
        policy: Governance policy used
        timestamp: Timestamp of evaluation
        metadata: Additional metadata
    """
    signal: GovernanceSignal
    drift_counts: Dict[DriftSeverity, int]
    drift_signals: List[Dict[str, Any]]
    policy: str
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "signal": self.signal.value,
            "drift_counts": {k.value: v for k, v in self.drift_counts.items()},
            "drift_signals": self.drift_signals,
            "policy": self.policy,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }
    
    def to_console_output(self) -> str:
        """Format for console output."""
        lines = []
        lines.append("=" * 60)
        lines.append("GOVERNANCE EVIDENCE PACK")
        lines.append("=" * 60)
        lines.append(f"Signal: {self.signal.value}")
        lines.append(f"Policy: {self.policy}")
        lines.append(f"Timestamp: {self.timestamp}")
        lines.append("")
        lines.append("Drift Counts:")
        for severity, count in self.drift_counts.items():
            lines.append(f"  {severity.value}: {count}")
        lines.append("")
        lines.append(f"Total Drift Signals: {len(self.drift_signals)}")
        if self.drift_signals:
            lines.append("")
            lines.append("Top 5 Drift Signals:")
            for i, signal in enumerate(self.drift_signals[:5], 1):
                lines.append(f"  {i}. [{signal['severity']}] {signal['type']}: {signal['message']}")
        lines.append("=" * 60)
        return "\n".join(lines)


class GovernanceAdaptor:
    """
    Drift Radar → Governance Adaptor.
    
    Maps drift severity to governance signals for CI/CD enforcement.
    """
    
    def __init__(self, policy: GovernancePolicy = STRICT_POLICY):
        """
        Initialize governance adaptor.
        
        Args:
            policy: Governance policy to use
        """
        self.policy = policy
    
    def evaluate_drift_signals(
        self,
        drift_signals: List[Dict[str, Any]],
    ) -> EvidencePack:
        """
        Evaluate drift signals and produce governance evidence pack.
        
        Args:
            drift_signals: List of drift signals from drift radar
        
        Returns:
            EvidencePack with governance signal
        """
        # Count drift signals by severity
        drift_counts = {
            DriftSeverity.LOW: 0,
            DriftSeverity.MEDIUM: 0,
            DriftSeverity.HIGH: 0,
            DriftSeverity.CRITICAL: 0,
        }
        
        for signal in drift_signals:
            severity_str = signal.get("severity", "LOW")
            try:
                severity = DriftSeverity[severity_str]
                drift_counts[severity] += 1
            except KeyError:
                # Unknown severity, treat as LOW
                drift_counts[DriftSeverity.LOW] += 1
        
        # Evaluate governance signal
        governance_signal = self.policy.evaluate(drift_counts)
        
        # Create evidence pack
        evidence_pack = EvidencePack(
            signal=governance_signal,
            drift_counts=drift_counts,
            drift_signals=drift_signals,
            policy=self.policy.name,
            timestamp=datetime.utcnow().isoformat(),
            metadata={
                "total_signals": len(drift_signals),
                "policy_thresholds": {k.value: v for k, v in self.policy.severity_thresholds.items()},
            },
        )
        
        return evidence_pack
    
    def should_block_merge(self, evidence_pack: EvidencePack) -> bool:
        """
        Determine if merge should be blocked.
        
        Args:
            evidence_pack: Evidence pack
        
        Returns:
            True if merge should be blocked
        """
        return evidence_pack.signal in [GovernanceSignal.BLOCK, GovernanceSignal.EMERGENCY]
    
    def should_warn(self, evidence_pack: EvidencePack) -> bool:
        """
        Determine if warning should be issued.
        
        Args:
            evidence_pack: Evidence pack
        
        Returns:
            True if warning should be issued
        """
        return evidence_pack.signal == GovernanceSignal.WARN
    
    def get_remediation_guidance(self, evidence_pack: EvidencePack) -> List[str]:
        """
        Get remediation guidance based on evidence pack.
        
        Args:
            evidence_pack: Evidence pack
        
        Returns:
            List of remediation steps
        """
        guidance = []
        
        if evidence_pack.signal == GovernanceSignal.BLOCK:
            guidance.append("MERGE BLOCKED - Manual review required")
            guidance.append("")
            guidance.append("Remediation steps:")
            
            # CRITICAL drift
            if evidence_pack.drift_counts.get(DriftSeverity.CRITICAL, 0) > 0:
                guidance.append("1. Investigate CRITICAL drift signals:")
                for signal in evidence_pack.drift_signals:
                    if signal.get("severity") == "CRITICAL":
                        guidance.append(f"   - {signal['type']}: {signal['message']}")
                guidance.append("   Action: Revert changes or fix root cause")
            
            # HIGH drift
            if evidence_pack.drift_counts.get(DriftSeverity.HIGH, 0) > 0:
                guidance.append("2. Investigate HIGH drift signals:")
                for signal in evidence_pack.drift_signals:
                    if signal.get("severity") == "HIGH":
                        guidance.append(f"   - {signal['type']}: {signal['message']}")
                guidance.append("   Action: Review and document intentional changes")
        
        elif evidence_pack.signal == GovernanceSignal.WARN:
            guidance.append("WARNING - Proceed with caution")
            guidance.append("")
            guidance.append("Review:")
            
            # MEDIUM drift
            if evidence_pack.drift_counts.get(DriftSeverity.MEDIUM, 0) > 0:
                guidance.append("1. MEDIUM drift signals:")
                for signal in evidence_pack.drift_signals:
                    if signal.get("severity") == "MEDIUM":
                        guidance.append(f"   - {signal['type']}: {signal['message']}")
        
        elif evidence_pack.signal == GovernanceSignal.OK:
            guidance.append("OK - No action required")
        
        return guidance


def create_governance_adaptor(policy_name: str = "strict") -> GovernanceAdaptor:
    """
    Create governance adaptor with specified policy.
    
    Args:
        policy_name: Policy name ("strict" | "moderate" | "permissive")
    
    Returns:
        GovernanceAdaptor
    """
    policies = {
        "strict": STRICT_POLICY,
        "moderate": MODERATE_POLICY,
        "permissive": PERMISSIVE_POLICY,
    }
    
    policy = policies.get(policy_name, STRICT_POLICY)
    return GovernanceAdaptor(policy)


# Example usage
if __name__ == "__main__":
    # Create adaptor with strict policy
    adaptor = create_governance_adaptor("strict")
    
    # Example drift signals
    drift_signals = [
        {
            "type": "SCHEMA_DRIFT",
            "severity": "CRITICAL",
            "message": "canonical_proofs schema changed without migration",
            "block_number": 12345,
        },
        {
            "type": "HASH_DELTA_DRIFT",
            "severity": "HIGH",
            "message": "Hash computation changed",
            "block_number": 12346,
        },
        {
            "type": "METADATA_DRIFT",
            "severity": "MEDIUM",
            "message": "attestation_metadata field added",
            "block_number": 12347,
        },
    ]
    
    # Evaluate
    evidence_pack = adaptor.evaluate_drift_signals(drift_signals)
    
    # Print evidence pack
    print(evidence_pack.to_console_output())
    print()
    
    # Print remediation guidance
    guidance = adaptor.get_remediation_guidance(evidence_pack)
    for line in guidance:
        print(line)
