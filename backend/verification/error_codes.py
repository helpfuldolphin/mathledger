"""
Verifier Error Codes — Stable Error Taxonomy for Phase II

This module defines a stable, comprehensive taxonomy of verifier outcomes
for the imperfect verifier regime. All verifier calls must map to one of
these error codes for deterministic, reproducible behavior.

Design Principles:
- Every verifier outcome has a stable error code
- Error codes are enum-based for type safety
- Outcomes include full telemetry metadata
- No silent failures or ambiguous states

Author: Manus-C (Telemetry Architect)
Date: 2025-12-06
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class VerifierErrorCode(Enum):
    """Stable error codes for all verifier outcomes.
    
    These codes form the canonical taxonomy for Phase II verifier behavior.
    Every verifier call must produce exactly one of these codes.
    """
    
    # ==================== Success States ====================
    
    VERIFIED = "VERIFIED"
    """Proof successfully verified by the verifier."""
    
    # ==================== Genuine Failures ====================
    
    PROOF_INVALID = "PROOF_INVALID"
    """Proof is genuinely invalid (verifier correctly rejected)."""
    
    PROOF_INCOMPLETE = "PROOF_INCOMPLETE"
    """Proof is incomplete or malformed."""
    
    # ==================== Verifier Imperfections (Noise) ====================
    
    VERIFIER_TIMEOUT = "VERIFIER_TIMEOUT"
    """Verifier exceeded time budget (may be noise-injected)."""
    
    VERIFIER_SPURIOUS_FAIL = "VERIFIER_SPURIOUS_FAIL"
    """Verifier incorrectly rejected a valid proof (false negative)."""
    
    VERIFIER_SPURIOUS_PASS = "VERIFIER_SPURIOUS_PASS"
    """Verifier incorrectly accepted an invalid proof (false positive)."""
    
    VERIFIER_INTERNAL_ERROR = "VERIFIER_INTERNAL_ERROR"
    """Verifier encountered an internal error (crash, exception)."""
    
    # ==================== Resource Constraints ====================
    
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    """Verification budget exhausted (cycle budget, candidate limit)."""
    
    MEMORY_LIMIT_EXCEEDED = "MEMORY_LIMIT_EXCEEDED"
    """Verifier exceeded memory limit."""
    
    # ==================== Abstention States ====================
    
    ABSTENTION_MOCK_MODE = "ABSTENTION_MOCK_MODE"
    """Verifier in mock mode, abstaining from real verification."""
    
    ABSTENTION_CONTROLLED_ONLY = "ABSTENTION_CONTROLLED_ONLY"
    """Verifier only handles controlled statements, abstaining on others."""
    
    def is_success(self) -> bool:
        """Check if this error code represents successful verification."""
        return self in {
            VerifierErrorCode.VERIFIED,
            VerifierErrorCode.VERIFIER_SPURIOUS_PASS,  # Success, but noisy
        }
    
    def is_failure(self) -> bool:
        """Check if this error code represents verification failure."""
        return self in {
            VerifierErrorCode.PROOF_INVALID,
            VerifierErrorCode.PROOF_INCOMPLETE,
            VerifierErrorCode.VERIFIER_SPURIOUS_FAIL,  # Failure, but noisy
        }
    
    def is_timeout(self) -> bool:
        """Check if this error code represents a timeout."""
        return self == VerifierErrorCode.VERIFIER_TIMEOUT
    
    def is_noise_injected(self) -> bool:
        """Check if this error code represents noise-injected behavior."""
        return self in {
            VerifierErrorCode.VERIFIER_TIMEOUT,
            VerifierErrorCode.VERIFIER_SPURIOUS_FAIL,
            VerifierErrorCode.VERIFIER_SPURIOUS_PASS,
        }
    
    def is_abstention(self) -> bool:
        """Check if this error code represents abstention."""
        return self in {
            VerifierErrorCode.ABSTENTION_MOCK_MODE,
            VerifierErrorCode.ABSTENTION_CONTROLLED_ONLY,
        }
    
    def is_resource_constraint(self) -> bool:
        """Check if this error code represents resource constraint."""
        return self in {
            VerifierErrorCode.BUDGET_EXHAUSTED,
            VerifierErrorCode.MEMORY_LIMIT_EXCEEDED,
        }


class VerifierTier(Enum):
    """Verifier tier for mixed-verifier routing.
    
    Different tiers trade off speed vs. accuracy:
    - FAST_NOISY: High noise, low latency
    - BALANCED: Medium noise, medium latency
    - SLOW_PRECISE: Low noise, high latency
    """
    
    FAST_NOISY = "fast_noisy"
    """Fast verifier with high noise rates."""
    
    BALANCED = "balanced"
    """Balanced verifier with moderate noise."""
    
    SLOW_PRECISE = "slow_precise"
    """Slow, precise verifier with low noise."""
    
    MOCK = "mock"
    """Mock verifier (no real verification)."""


@dataclass(frozen=True)
class VerifierOutcome:
    """Complete outcome of a verifier call with full telemetry.
    
    This dataclass captures everything needed for:
    - RFL feedback generation
    - Telemetry and observability
    - Debugging and reproducibility
    - Post-hoc analysis
    
    Invariants:
    - Every verifier call produces exactly one VerifierOutcome
    - All fields are immutable (frozen dataclass)
    - Metadata dict can contain arbitrary telemetry
    """
    
    error_code: VerifierErrorCode
    """Stable error code for this outcome."""
    
    success: bool
    """Whether verification succeeded (may differ from error_code for noise)."""
    
    duration_ms: float
    """Duration of verification in milliseconds."""
    
    tier: VerifierTier
    """Verifier tier used for this call."""
    
    noise_injected: bool
    """Whether noise was injected into this outcome."""
    
    noise_type: Optional[str]
    """Type of noise injected (timeout, spurious_fail, spurious_pass)."""
    
    attempt_count: int
    """Number of attempts (for escalation tracking)."""
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional telemetry metadata."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "error_code": self.error_code.value,
            "success": self.success,
            "duration_ms": self.duration_ms,
            "tier": self.tier.value,
            "noise_injected": self.noise_injected,
            "noise_type": self.noise_type,
            "attempt_count": self.attempt_count,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VerifierOutcome":
        """Reconstruct from dictionary."""
        return cls(
            error_code=VerifierErrorCode(data["error_code"]),
            success=data["success"],
            duration_ms=data["duration_ms"],
            tier=VerifierTier(data["tier"]),
            noise_injected=data["noise_injected"],
            noise_type=data.get("noise_type"),
            attempt_count=data["attempt_count"],
            metadata=data.get("metadata", {}),
        )
    
    def to_rfl_feedback(self) -> Optional[str]:
        """Convert outcome to RFL feedback signal.
        
        Returns:
            "positive" for successful verification
            "negative" for failed verification
            None for abstention/timeout (no feedback)
        """
        if self.is_abstention() or self.is_timeout():
            return None
        
        if self.success:
            return "positive"
        else:
            return "negative"
    
    def is_abstention(self) -> bool:
        """Check if this outcome represents abstention."""
        return self.error_code.is_abstention()
    
    def is_timeout(self) -> bool:
        """Check if this outcome represents a timeout."""
        return self.error_code.is_timeout()
    
    def should_escalate(self) -> bool:
        """Check if this outcome should trigger tier escalation.
        
        Escalation is triggered by:
        - Timeouts
        - Spurious failures
        - Internal errors
        """
        return self.error_code in {
            VerifierErrorCode.VERIFIER_TIMEOUT,
            VerifierErrorCode.VERIFIER_SPURIOUS_FAIL,
            VerifierErrorCode.VERIFIER_INTERNAL_ERROR,
        }


# ==================== Outcome Constructors ====================

def verified_outcome(
    duration_ms: float,
    tier: VerifierTier,
    attempt_count: int = 1,
    metadata: Optional[Dict[str, Any]] = None,
) -> VerifierOutcome:
    """Construct a successful verification outcome."""
    return VerifierOutcome(
        error_code=VerifierErrorCode.VERIFIED,
        success=True,
        duration_ms=duration_ms,
        tier=tier,
        noise_injected=False,
        noise_type=None,
        attempt_count=attempt_count,
        metadata=metadata or {},
    )


def proof_invalid_outcome(
    duration_ms: float,
    tier: VerifierTier,
    attempt_count: int = 1,
    metadata: Optional[Dict[str, Any]] = None,
) -> VerifierOutcome:
    """Construct a proof invalid outcome."""
    return VerifierOutcome(
        error_code=VerifierErrorCode.PROOF_INVALID,
        success=False,
        duration_ms=duration_ms,
        tier=tier,
        noise_injected=False,
        noise_type=None,
        attempt_count=attempt_count,
        metadata=metadata or {},
    )


def timeout_outcome(
    duration_ms: float,
    tier: VerifierTier,
    attempt_count: int = 1,
    noise_injected: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> VerifierOutcome:
    """Construct a timeout outcome."""
    return VerifierOutcome(
        error_code=VerifierErrorCode.VERIFIER_TIMEOUT,
        success=False,
        duration_ms=duration_ms,
        tier=tier,
        noise_injected=noise_injected,
        noise_type="timeout" if noise_injected else None,
        attempt_count=attempt_count,
        metadata=metadata or {},
    )


def spurious_fail_outcome(
    duration_ms: float,
    tier: VerifierTier,
    attempt_count: int = 1,
    metadata: Optional[Dict[str, Any]] = None,
) -> VerifierOutcome:
    """Construct a spurious failure outcome (false negative)."""
    return VerifierOutcome(
        error_code=VerifierErrorCode.VERIFIER_SPURIOUS_FAIL,
        success=False,
        duration_ms=duration_ms,
        tier=tier,
        noise_injected=True,
        noise_type="spurious_fail",
        attempt_count=attempt_count,
        metadata=metadata or {},
    )


def spurious_pass_outcome(
    duration_ms: float,
    tier: VerifierTier,
    attempt_count: int = 1,
    metadata: Optional[Dict[str, Any]] = None,
) -> VerifierOutcome:
    """Construct a spurious pass outcome (false positive)."""
    return VerifierOutcome(
        error_code=VerifierErrorCode.VERIFIER_SPURIOUS_PASS,
        success=True,
        duration_ms=duration_ms,
        tier=tier,
        noise_injected=True,
        noise_type="spurious_pass",
        attempt_count=attempt_count,
        metadata=metadata or {},
    )


def abstention_outcome(
    tier: VerifierTier,
    reason: str = "mock_mode",
    metadata: Optional[Dict[str, Any]] = None,
) -> VerifierOutcome:
    """Construct an abstention outcome."""
    error_code = (
        VerifierErrorCode.ABSTENTION_MOCK_MODE
        if reason == "mock_mode"
        else VerifierErrorCode.ABSTENTION_CONTROLLED_ONLY
    )
    return VerifierOutcome(
        error_code=error_code,
        success=False,
        duration_ms=0.0,
        tier=tier,
        noise_injected=False,
        noise_type=None,
        attempt_count=1,
        metadata=metadata or {},
    )
