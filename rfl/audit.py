"""
RFL Audit Trail Module
======================

Provides deterministic audit logging for RFL policy updates per the RFL Law.
Every transformation from H_t to ledger entry is captured with full provenance.

Usage:
    from rfl.audit import RFLAuditLog, AuditEntry

    audit = RFLAuditLog()
    entry = audit.record_transformation(attestation, result, config)

    # Verify determinism
    assert audit.verify_entry(entry, attestation, config)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from substrate.repro.determinism import deterministic_timestamp


@dataclass(frozen=True)
class SymbolicDescentGradient:
    """
    Captures the symbolic descent computation per RFL Law.

    The gradient represents the direction and magnitude of policy adjustment
    based on the abstention profile relative to tolerance.
    """
    abstention_rate: float           # α_rate: raw abstention rate [0,1]
    abstention_mass: float           # α_mass: raw abstention count
    tolerance: float                 # τ: configured tolerance
    attempt_mass: float              # Total attempts

    # Computed values (deterministic from above)
    mass_delta: float                # Δα = α_mass - (τ × attempt_mass)
    rate_delta: float                # α_rate - τ
    symbolic_descent: float          # ∇_sym = -(α_rate - τ)
    policy_reward: float             # r = max(0, 1 - α_rate)

    @classmethod
    def compute(
        cls,
        abstention_rate: float,
        abstention_mass: float,
        tolerance: float,
        attempt_mass: float,
    ) -> "SymbolicDescentGradient":
        """
        Compute the symbolic descent gradient from abstention profile.

        This is the core RFL Law transformation:
            ∇_sym = -(α_rate - τ)

        Args:
            abstention_rate: Fraction of attempts that abstained [0,1]
            abstention_mass: Absolute count of abstentions
            tolerance: Configured abstention tolerance threshold
            attempt_mass: Total number of attempts

        Returns:
            SymbolicDescentGradient with all computed values
        """
        mass_delta = abstention_mass - (tolerance * attempt_mass)
        rate_delta = abstention_rate - tolerance
        symbolic_descent = -rate_delta  # Negative rate delta = positive descent
        policy_reward = max(0.0, 1.0 - max(abstention_rate, 0.0))

        return cls(
            abstention_rate=abstention_rate,
            abstention_mass=abstention_mass,
            tolerance=tolerance,
            attempt_mass=attempt_mass,
            mass_delta=mass_delta,
            rate_delta=rate_delta,
            symbolic_descent=symbolic_descent,
            policy_reward=policy_reward,
        )

    def triggers_update(self, epsilon: float = 1e-9) -> bool:
        """
        Determine if this gradient triggers a policy update.

        A policy update is triggered when either:
        - |Δα| > epsilon (mass differs from expected)
        - |α_rate - τ| > epsilon (rate differs from tolerance)
        """
        return abs(self.mass_delta) > epsilon or abs(self.rate_delta) > epsilon

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "abstention_rate": self.abstention_rate,
            "abstention_mass": self.abstention_mass,
            "tolerance": self.tolerance,
            "attempt_mass": self.attempt_mass,
            "mass_delta": self.mass_delta,
            "rate_delta": self.rate_delta,
            "symbolic_descent": self.symbolic_descent,
            "policy_reward": self.policy_reward,
            "triggers_update": self.triggers_update(),
        }


@dataclass(frozen=True)
class StepIdComputation:
    """
    Captures the deterministic step_id computation per RFL Law.

    step_id = SHA256(experiment_id | slice_name | policy_id | H_t)
    """
    experiment_id: str
    slice_name: str
    policy_id: str
    composite_root: str  # H_t

    # Computed value
    step_id: str

    @classmethod
    def compute(
        cls,
        experiment_id: str,
        slice_name: str,
        policy_id: str,
        composite_root: str,
    ) -> "StepIdComputation":
        """
        Compute the deterministic step_id from input parameters.

        This implements the RFL Law formula:
            step_id = SHA256(experiment_id | slice_name | policy_id | H_t)
        """
        step_material = f"{experiment_id}|{slice_name}|{policy_id}|{composite_root}"
        step_id = hashlib.sha256(step_material.encode("utf-8")).hexdigest()

        return cls(
            experiment_id=experiment_id,
            slice_name=slice_name,
            policy_id=policy_id,
            composite_root=composite_root,
            step_id=step_id,
        )

    def verify(self) -> bool:
        """Verify that step_id matches the deterministic computation."""
        expected = hashlib.sha256(
            f"{self.experiment_id}|{self.slice_name}|{self.policy_id}|{self.composite_root}".encode()
        ).hexdigest()
        return self.step_id == expected

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "experiment_id": self.experiment_id,
            "slice_name": self.slice_name,
            "policy_id": self.policy_id,
            "composite_root": self.composite_root,
            "step_id": self.step_id,
            "verified": self.verify(),
        }


@dataclass
class AuditEntry:
    """
    Complete audit trail entry for an RFL transformation.

    This captures the full provenance from H_t to ledger entry,
    enabling deterministic verification and auditing.
    """
    # Source attestation (inputs)
    source_h_t: str                      # H_t composite root
    source_r_t: str                      # R_t reasoning root
    source_u_t: str                      # U_t UI root
    source_slice_id: str                 # Input slice identifier
    source_policy_id: str                # Input policy identifier

    # Computations (deterministic transformations)
    step_id_computation: StepIdComputation
    gradient: SymbolicDescentGradient

    # Result (outputs)
    result_step_id: str
    result_policy_update_applied: bool
    result_abstention_mass_delta: float

    # Metadata
    audit_timestamp: str                 # Deterministic timestamp
    audit_version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "source": {
                "h_t": self.source_h_t,
                "r_t": self.source_r_t,
                "u_t": self.source_u_t,
                "slice_id": self.source_slice_id,
                "policy_id": self.source_policy_id,
            },
            "computations": {
                "step_id": self.step_id_computation.to_dict(),
                "gradient": self.gradient.to_dict(),
            },
            "result": {
                "step_id": self.result_step_id,
                "policy_update_applied": self.result_policy_update_applied,
                "abstention_mass_delta": self.result_abstention_mass_delta,
            },
            "metadata": {
                "timestamp": self.audit_timestamp,
                "version": self.audit_version,
            }
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def verify_determinism(self) -> Tuple[bool, List[str]]:
        """
        Verify that all computed values are deterministic.

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        # Verify step_id computation
        if not self.step_id_computation.verify():
            errors.append("step_id computation mismatch")

        if self.step_id_computation.step_id != self.result_step_id:
            errors.append("step_id does not match result")

        # Verify gradient triggers_update matches result
        if self.gradient.triggers_update() != self.result_policy_update_applied:
            errors.append("policy_update_applied does not match gradient.triggers_update()")

        # Verify mass_delta matches result
        if abs(self.gradient.mass_delta - self.result_abstention_mass_delta) > 1e-9:
            errors.append("abstention_mass_delta does not match gradient.mass_delta")

        return len(errors) == 0, errors


class RFLAuditLog:
    """
    Audit log for RFL transformations.

    Maintains a complete, verifiable record of all H_t → ledger entry
    transformations for determinism verification and auditing.
    """

    def __init__(self, seed: int = 0):
        """
        Initialize audit log.

        Args:
            seed: Deterministic seed for timestamps
        """
        self.entries: List[AuditEntry] = []
        self._seed = seed
        self._entry_count = 0

    def record_transformation(
        self,
        attestation: Any,  # AttestedRunContext
        result: Any,       # RflResult
        config: Any,       # RFLConfig
        resolved_slice_name: str,
    ) -> AuditEntry:
        """
        Record an RFL transformation for auditing.

        Args:
            attestation: The input AttestedRunContext
            result: The output RflResult
            config: The RFLConfig used
            resolved_slice_name: The resolved curriculum slice name

        Returns:
            AuditEntry capturing the full transformation
        """
        # Compute gradient
        attempt_mass = float(
            attestation.metadata.get("attempt_mass", max(attestation.abstention_mass, 1.0))
        )
        gradient = SymbolicDescentGradient.compute(
            abstention_rate=attestation.abstention_rate,
            abstention_mass=attestation.abstention_mass,
            tolerance=config.abstention_tolerance,
            attempt_mass=attempt_mass,
        )

        # Compute step_id
        policy_id = attestation.policy_id or "default"
        step_id_comp = StepIdComputation.compute(
            experiment_id=config.experiment_id,
            slice_name=resolved_slice_name,
            policy_id=policy_id,
            composite_root=attestation.composite_root,
        )

        # Create audit entry
        entry = AuditEntry(
            source_h_t=attestation.composite_root,
            source_r_t=attestation.reasoning_root,
            source_u_t=attestation.ui_root,
            source_slice_id=attestation.slice_id,
            source_policy_id=policy_id,
            step_id_computation=step_id_comp,
            gradient=gradient,
            result_step_id=result.step_id,
            result_policy_update_applied=result.policy_update_applied,
            result_abstention_mass_delta=result.abstention_mass_delta,
            audit_timestamp=deterministic_timestamp(self._seed + self._entry_count).isoformat() + "Z",
        )

        self.entries.append(entry)
        self._entry_count += 1

        return entry

    def verify_all(self) -> Tuple[bool, List[Tuple[int, List[str]]]]:
        """
        Verify all entries in the audit log.

        Returns:
            Tuple of (all_valid, list of (index, errors) for invalid entries)
        """
        invalid = []
        for i, entry in enumerate(self.entries):
            is_valid, errors = entry.verify_determinism()
            if not is_valid:
                invalid.append((i, errors))
        return len(invalid) == 0, invalid

    def export(self, path: Path) -> None:
        """Export audit log to JSON file."""
        data = {
            "version": "1.0.0",
            "entry_count": len(self.entries),
            "entries": [e.to_dict() for e in self.entries],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "RFLAuditLog":
        """Load audit log from JSON file."""
        data = json.loads(path.read_text(encoding="utf-8"))
        log = cls()
        # Note: Full deserialization requires reconstructing dataclasses
        # For now, just load the raw data
        log._raw_data = data
        return log


def verify_rfl_transformation(
    h_t: str,
    r_t: str,
    u_t: str,
    abstention_rate: float,
    abstention_mass: float,
    attempt_mass: float,
    experiment_id: str,
    slice_name: str,
    policy_id: str,
    tolerance: float,
    expected_step_id: str,
    expected_symbolic_descent: float,
    expected_policy_reward: float,
    epsilon: float = 1e-9,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Verify an RFL transformation per the RFL Law.

    This function re-computes all values from inputs and compares
    them against expected outputs, providing a complete audit.

    Args:
        h_t: Composite attestation root
        r_t: Reasoning Merkle root
        u_t: UI Merkle root
        abstention_rate: Input abstention rate
        abstention_mass: Input abstention mass
        attempt_mass: Total attempts
        experiment_id: RFL experiment ID
        slice_name: Resolved slice name
        policy_id: Policy identifier
        tolerance: Abstention tolerance
        expected_step_id: Expected output step_id
        expected_symbolic_descent: Expected output ∇_sym
        expected_policy_reward: Expected output reward
        epsilon: Comparison tolerance for floats

    Returns:
        Tuple of (is_valid, audit_report dict)
    """
    # Recompute step_id
    step_material = f"{experiment_id}|{slice_name}|{policy_id}|{h_t}"
    computed_step_id = hashlib.sha256(step_material.encode()).hexdigest()

    # Recompute gradient
    computed_rate_delta = abstention_rate - tolerance
    computed_symbolic_descent = -computed_rate_delta
    computed_policy_reward = max(0.0, 1.0 - max(abstention_rate, 0.0))

    # Build audit report
    report = {
        "inputs": {
            "h_t": h_t,
            "r_t": r_t,
            "u_t": u_t,
            "abstention_rate": abstention_rate,
            "abstention_mass": abstention_mass,
            "attempt_mass": attempt_mass,
            "experiment_id": experiment_id,
            "slice_name": slice_name,
            "policy_id": policy_id,
            "tolerance": tolerance,
        },
        "computed": {
            "step_id": computed_step_id,
            "symbolic_descent": computed_symbolic_descent,
            "policy_reward": computed_policy_reward,
        },
        "expected": {
            "step_id": expected_step_id,
            "symbolic_descent": expected_symbolic_descent,
            "policy_reward": expected_policy_reward,
        },
        "checks": {
            "step_id_match": computed_step_id == expected_step_id,
            "symbolic_descent_match": abs(computed_symbolic_descent - expected_symbolic_descent) <= epsilon,
            "policy_reward_match": abs(computed_policy_reward - expected_policy_reward) <= epsilon,
        }
    }

    is_valid = all(report["checks"].values())
    report["is_valid"] = is_valid

    return is_valid, report


__all__ = [
    "SymbolicDescentGradient",
    "StepIdComputation",
    "AuditEntry",
    "RFLAuditLog",
    "verify_rfl_transformation",
]
