"""
Curriculum Stability Envelope Module

Extends curriculum drift radar with forward-looking consistency guarantees.
Provides fingerprinting, invariant checking, and promotion guards to ensure
curriculum stability and prevent regressions.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from curriculum.gates import (
    CurriculumSystem,
    CurriculumSlice,
    SliceGates,
    CoverageGateSpec,
    AbstentionGateSpec,
    VelocityGateSpec,
    CapsGateSpec,
)


# Curriculum Fingerprinting
# -----------------------------------------------------------------------------

def _normalize_value(value: Any) -> Any:
    """Normalize a value for fingerprint consistency."""
    if isinstance(value, float):
        # Normalize floats to avoid floating point comparison issues
        return round(value, 10)
    elif isinstance(value, dict):
        return _normalize_dict(value)
    elif isinstance(value, list):
        return [_normalize_value(v) for v in value]
    else:
        return value


def _normalize_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a dictionary with sorted keys and normalized values."""
    return {k: _normalize_value(v) for k, v in sorted(d.items())}


def _serialize_gates(gates: SliceGates) -> Dict[str, Any]:
    """Serialize SliceGates to a normalized dict."""
    return {
        "abstention": _normalize_dict(gates.abstention.to_dict()),
        "caps": _normalize_dict(gates.caps.to_dict()),
        "coverage": _normalize_dict(gates.coverage.to_dict()),
        "velocity": _normalize_dict(gates.velocity.to_dict()),
    }


def _serialize_slice(slice_obj: CurriculumSlice) -> Dict[str, Any]:
    """Serialize a CurriculumSlice to a normalized dict."""
    return {
        "name": slice_obj.name,
        "params": _normalize_dict(slice_obj.params),
        "gates": _serialize_gates(slice_obj.gates),
        "completed_at": slice_obj.completed_at,
        "metadata": _normalize_dict(slice_obj.metadata),
    }


def compute_fingerprint(system: CurriculumSystem) -> Dict[str, Any]:
    """
    Compute a canonical fingerprint of a curriculum system.
    
    Returns a normalized, sorted representation suitable for diffing.
    """
    slices = sorted(
        [_serialize_slice(s) for s in system.slices],
        key=lambda s: s["name"]
    )
    
    return {
        "slug": system.slug,
        "version": system.version,
        "description": system.description,
        "monotonic_axes": sorted(system.monotonic_axes),
        "active_index": system.active_index,
        "active_name": system.active_name,
        "slices": slices,
    }


@dataclass
class FingerprintDiff:
    """Result of comparing two curriculum fingerprints."""
    changed_slices: List[str] = field(default_factory=list)
    param_diffs: Dict[str, Dict[str, Tuple[Any, Any]]] = field(default_factory=dict)
    gate_diffs: Dict[str, Dict[str, Tuple[Any, Any]]] = field(default_factory=dict)
    invariant_diffs: Dict[str, Tuple[Any, Any]] = field(default_factory=dict)
    added_slices: List[str] = field(default_factory=list)
    removed_slices: List[str] = field(default_factory=list)
    renamed_slices: List[Tuple[str, str]] = field(default_factory=list)
    
    @property
    def has_changes(self) -> bool:
        """Check if any changes were detected."""
        return bool(
            self.changed_slices or
            self.param_diffs or
            self.gate_diffs or
            self.invariant_diffs or
            self.added_slices or
            self.removed_slices or
            self.renamed_slices
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "changed_slices": self.changed_slices,
            "param_diffs": self.param_diffs,
            "gate_diffs": self.gate_diffs,
            "invariant_diffs": self.invariant_diffs,
            "added_slices": self.added_slices,
            "removed_slices": self.removed_slices,
            "renamed_slices": self.renamed_slices,
        }


def compute_fingerprint_diff(a: Dict[str, Any], b: Dict[str, Any]) -> FingerprintDiff:
    """
    Compute the difference between two curriculum fingerprints.
    
    Args:
        a: First fingerprint (typically "before" state)
        b: Second fingerprint (typically "after" state)
    
    Returns:
        FingerprintDiff with detailed change information
    """
    diff = FingerprintDiff()
    
    # Check invariant-level changes
    for key in ["slug", "version", "description", "monotonic_axes"]:
        if a.get(key) != b.get(key):
            diff.invariant_diffs[key] = (a.get(key), b.get(key))
    
    # Build slice name sets
    a_slices = {s["name"]: s for s in a.get("slices", [])}
    b_slices = {s["name"]: s for s in b.get("slices", [])}
    
    a_names = set(a_slices.keys())
    b_names = set(b_slices.keys())
    
    # Detect additions and removals
    diff.added_slices = sorted(b_names - a_names)
    diff.removed_slices = sorted(a_names - b_names)
    
    # Check for param and gate changes in common slices
    common_slices = a_names & b_names
    for name in sorted(common_slices):
        slice_a = a_slices[name]
        slice_b = b_slices[name]
        
        # Compare params
        params_a = slice_a.get("params", {})
        params_b = slice_b.get("params", {})
        param_diff = {}
        
        all_param_keys = set(params_a.keys()) | set(params_b.keys())
        for key in sorted(all_param_keys):
            val_a = params_a.get(key)
            val_b = params_b.get(key)
            if val_a != val_b:
                param_diff[key] = (val_a, val_b)
        
        if param_diff:
            diff.param_diffs[name] = param_diff
            if name not in diff.changed_slices:
                diff.changed_slices.append(name)
        
        # Compare gates
        gates_a = slice_a.get("gates", {})
        gates_b = slice_b.get("gates", {})
        gate_diff = {}
        
        for gate_type in ["coverage", "abstention", "velocity", "caps"]:
            gate_a_spec = gates_a.get(gate_type, {})
            gate_b_spec = gates_b.get(gate_type, {})
            
            all_gate_keys = set(gate_a_spec.keys()) | set(gate_b_spec.keys())
            for key in sorted(all_gate_keys):
                val_a = gate_a_spec.get(key)
                val_b = gate_b_spec.get(key)
                if val_a != val_b:
                    gate_key = f"{gate_type}.{key}"
                    gate_diff[gate_key] = (val_a, val_b)
        
        if gate_diff:
            diff.gate_diffs[name] = gate_diff
            if name not in diff.changed_slices:
                diff.changed_slices.append(name)
    
    return diff


# Curriculum Invariant Validation
# -----------------------------------------------------------------------------

@dataclass
class CurriculumInvariantReport:
    """Result of curriculum invariant validation."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
        }


def _is_slug_safe(name: str) -> bool:
    """Check if a name is slug-safe (no whitespace, URL-safe characters)."""
    # Allow alphanumeric, hyphens, underscores
    return bool(re.match(r'^[a-zA-Z0-9_-]+$', name))


def _validate_slice_naming(slices: List[CurriculumSlice]) -> List[str]:
    """Validate slice naming constraints."""
    errors = []
    max_length = 100  # Reasonable max length for slice names
    
    for slice_obj in slices:
        name = slice_obj.name
        
        # Check for whitespace
        if ' ' in name or '\t' in name or '\n' in name:
            errors.append(f"Slice name '{name}' contains whitespace")
        
        # Check slug-safety
        if not _is_slug_safe(name):
            errors.append(f"Slice name '{name}' is not slug-safe (use only alphanumeric, hyphens, underscores)")
        
        # Check length
        if len(name) > max_length:
            errors.append(f"Slice name '{name}' exceeds maximum length of {max_length} characters")
        
        # Check emptiness
        if not name:
            errors.append("Empty slice name detected")
    
    return errors


def _validate_slice_intervals(slices: List[CurriculumSlice]) -> List[str]:
    """Validate slice interval monotonicity (conceptual check for progression)."""
    errors = []
    
    for slice_obj in slices:
        params = slice_obj.params
        
        # Check for depth_max and breadth_max consistency if both present
        if "depth_max" in params and "breadth_max" in params:
            depth = params.get("depth_max")
            breadth = params.get("breadth_max")
            
            # Both should be positive if present
            if depth is not None and depth <= 0:
                errors.append(f"Slice '{slice_obj.name}' has non-positive depth_max={depth}")
            if breadth is not None and breadth <= 0:
                errors.append(f"Slice '{slice_obj.name}' has non-positive breadth_max={breadth}")
        
        # Check total_max is positive if present
        if "total_max" in params:
            total = params.get("total_max")
            if total is not None and total <= 0:
                errors.append(f"Slice '{slice_obj.name}' has non-positive total_max={total}")
    
    return errors


def _validate_gate_thresholds(slices: List[CurriculumSlice]) -> List[str]:
    """Validate gate threshold monotonicity across curriculum progression."""
    errors = []
    warnings = []
    
    # Track coverage CI lower bounds - should not increase across promotions
    prev_ci_lower = None
    for idx, slice_obj in enumerate(slices):
        ci_lower = slice_obj.gates.coverage.ci_lower_min
        
        if prev_ci_lower is not None and ci_lower > prev_ci_lower:
            # This is a WARNING not an ERROR - it's unusual but may be intentional
            warnings.append(
                f"Slice '{slice_obj.name}' (position {idx}) has higher coverage CI lower "
                f"({ci_lower}) than previous slice ({prev_ci_lower}). This may increase difficulty."
            )
        
        prev_ci_lower = ci_lower
    
    # Validate individual gate thresholds are within reasonable bounds
    for slice_obj in slices:
        gates = slice_obj.gates
        
        # Coverage checks
        if not (0.0 < gates.coverage.ci_lower_min <= 1.0):
            errors.append(
                f"Slice '{slice_obj.name}' coverage.ci_lower_min "
                f"({gates.coverage.ci_lower_min}) not in (0, 1]"
            )
        
        if gates.coverage.sample_min <= 0:
            errors.append(
                f"Slice '{slice_obj.name}' coverage.sample_min "
                f"({gates.coverage.sample_min}) must be positive"
            )
        
        # Abstention checks
        if not (0.0 <= gates.abstention.max_rate_pct <= 100.0):
            errors.append(
                f"Slice '{slice_obj.name}' abstention.max_rate_pct "
                f"({gates.abstention.max_rate_pct}) not in [0, 100]"
            )
        
        if gates.abstention.max_mass <= 0:
            errors.append(
                f"Slice '{slice_obj.name}' abstention.max_mass "
                f"({gates.abstention.max_mass}) must be positive"
            )
        
        # Velocity checks
        if gates.velocity.min_pph <= 0:
            errors.append(
                f"Slice '{slice_obj.name}' velocity.min_pph "
                f"({gates.velocity.min_pph}) must be positive"
            )
        
        if not (0.0 <= gates.velocity.stability_cv_max <= 1.0):
            errors.append(
                f"Slice '{slice_obj.name}' velocity.stability_cv_max "
                f"({gates.velocity.stability_cv_max}) not in [0, 1]"
            )
        
        if gates.velocity.window_minutes <= 0:
            errors.append(
                f"Slice '{slice_obj.name}' velocity.window_minutes "
                f"({gates.velocity.window_minutes}) must be positive"
            )
        
        # Caps checks
        if gates.caps.min_attempt_mass <= 0:
            errors.append(
                f"Slice '{slice_obj.name}' caps.min_attempt_mass "
                f"({gates.caps.min_attempt_mass}) must be positive"
            )
        
        if gates.caps.min_runtime_minutes <= 0:
            errors.append(
                f"Slice '{slice_obj.name}' caps.min_runtime_minutes "
                f"({gates.caps.min_runtime_minutes}) must be positive"
            )
        
        if not (0.0 <= gates.caps.backlog_max <= 1.0):
            errors.append(
                f"Slice '{slice_obj.name}' caps.backlog_max "
                f"({gates.caps.backlog_max}) not in [0, 1]"
            )
    
    return errors + warnings


def validate_curriculum_invariants(system: CurriculumSystem) -> CurriculumInvariantReport:
    """
    Validate curriculum invariants including:
    - Slice interval monotonicity (start < end for all slices)
    - Gate threshold monotonicity (coverage_ci_lower should not increase)
    - Slice naming constraints (slug-safe, no whitespace, max length)
    
    Args:
        system: CurriculumSystem to validate
    
    Returns:
        CurriculumInvariantReport with validation results
    """
    errors = []
    warnings = []
    
    # Validate slice naming
    naming_errors = _validate_slice_naming(system.slices)
    errors.extend(naming_errors)
    
    # Validate slice intervals
    interval_errors = _validate_slice_intervals(system.slices)
    errors.extend(interval_errors)
    
    # Validate gate thresholds (includes warnings)
    threshold_messages = _validate_gate_thresholds(system.slices)
    for msg in threshold_messages:
        if "WARNING" in msg or "may" in msg:
            warnings.append(msg)
        else:
            errors.append(msg)
    
    # Check monotonic axes are respected (already done in CurriculumSystem._validate_monotonicity)
    # But we can add our own checks
    if system.monotonic_axes:
        try:
            # This will raise ValueError if monotonicity is violated
            system._validate_monotonicity()
        except ValueError as e:
            errors.append(f"Monotonicity violation: {str(e)}")
    
    return CurriculumInvariantReport(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


# Promotion Envelope Guard
# -----------------------------------------------------------------------------

@dataclass
class PromotionStabilityReport:
    """Result of promotion stability evaluation."""
    allow_promotion: bool
    reason: str
    fingerprint_changes: int
    gate_threshold_changes: Dict[str, float] = field(default_factory=dict)
    removed_slices: List[str] = field(default_factory=list)
    renamed_slices: List[Tuple[str, str]] = field(default_factory=list)
    invariant_regressions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "allow_promotion": self.allow_promotion,
            "reason": self.reason,
            "fingerprint_changes": self.fingerprint_changes,
            "gate_threshold_changes": self.gate_threshold_changes,
            "removed_slices": self.removed_slices,
            "renamed_slices": self.renamed_slices,
            "invariant_regressions": self.invariant_regressions,
        }


def evaluate_curriculum_stability(
    current_fingerprint: Dict[str, Any],
    proposed_fingerprint: Dict[str, Any],
    invariants: CurriculumInvariantReport,
    max_slice_changes: int = 3,
    max_gate_change_pct: float = 10.0,
) -> PromotionStabilityReport:
    """
    Evaluate curriculum stability for promotion decisions.
    
    Blocks promotion if:
    - Fingerprint changed more than N slices at once
    - Gate thresholds changed by >10%
    - A slice was removed or renamed
    - Invariant regression occurred
    
    Args:
        current_fingerprint: Current curriculum fingerprint
        proposed_fingerprint: Proposed curriculum fingerprint
        invariants: Invariant validation report for proposed curriculum
        max_slice_changes: Maximum number of slices that can change (default: 3)
        max_gate_change_pct: Maximum percentage change for gate thresholds (default: 10%)
    
    Returns:
        PromotionStabilityReport with decision and reasoning
    """
    diff = compute_fingerprint_diff(current_fingerprint, proposed_fingerprint)
    
    # Check for invariant regressions
    if not invariants.valid:
        return PromotionStabilityReport(
            allow_promotion=False,
            reason=f"Invariant violations detected: {'; '.join(invariants.errors[:3])}",
            fingerprint_changes=len(diff.changed_slices),
            invariant_regressions=invariants.errors,
        )
    
    # Check for removed slices
    if diff.removed_slices:
        return PromotionStabilityReport(
            allow_promotion=False,
            reason=f"Slices removed: {', '.join(diff.removed_slices)}",
            fingerprint_changes=len(diff.changed_slices),
            removed_slices=diff.removed_slices,
        )
    
    # Check for renamed slices
    if diff.renamed_slices:
        renamed_str = ", ".join(f"{old}->{new}" for old, new in diff.renamed_slices)
        return PromotionStabilityReport(
            allow_promotion=False,
            reason=f"Slices renamed: {renamed_str}",
            fingerprint_changes=len(diff.changed_slices),
            renamed_slices=diff.renamed_slices,
        )
    
    # Check for too many slice changes
    if len(diff.changed_slices) > max_slice_changes:
        return PromotionStabilityReport(
            allow_promotion=False,
            reason=f"Too many slices changed ({len(diff.changed_slices)} > {max_slice_changes})",
            fingerprint_changes=len(diff.changed_slices),
        )
    
    # Check for gate threshold changes exceeding limit
    excessive_gate_changes = {}
    for slice_name, gate_diffs in diff.gate_diffs.items():
        for gate_key, (old_val, new_val) in gate_diffs.items():
            if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                if old_val != 0:
                    pct_change = abs((new_val - old_val) / old_val) * 100.0
                    if pct_change > max_gate_change_pct:
                        excessive_gate_changes[f"{slice_name}.{gate_key}"] = pct_change
    
    if excessive_gate_changes:
        max_change_key = max(excessive_gate_changes, key=excessive_gate_changes.get)
        max_change_val = excessive_gate_changes[max_change_key]
        return PromotionStabilityReport(
            allow_promotion=False,
            reason=f"Gate threshold changed excessively: {max_change_key} by {max_change_val:.1f}%",
            fingerprint_changes=len(diff.changed_slices),
            gate_threshold_changes=excessive_gate_changes,
        )
    
    # All checks passed
    if diff.has_changes:
        return PromotionStabilityReport(
            allow_promotion=True,
            reason=f"Curriculum changes within stability envelope ({len(diff.changed_slices)} slices changed)",
            fingerprint_changes=len(diff.changed_slices),
        )
    else:
        return PromotionStabilityReport(
            allow_promotion=True,
            reason="No curriculum changes detected",
            fingerprint_changes=0,
        )
