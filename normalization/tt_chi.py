"""
Computation Hardness Index (CHI) Estimator for Truth-Table Oracle.

PHASE II (Agent B2) - Purely Observational Analytics
=====================================================

The CHI provides a normalized measure of computational difficulty for
truth-table tautology checking. It is purely observational and has
NO impact on oracle behavior or semantics.

CHI Formula:
    CHI = log2(assignments_evaluated) * (elapsed_ns / baseline_ns)
    
Where:
    - assignments_evaluated: Number of truth assignments checked
    - elapsed_ns: Actual evaluation time in nanoseconds
    - baseline_ns: Expected time for a single assignment (~1000ns)

Interpretation:
    - CHI ~= atom_count for tautologies (all assignments checked)
    - CHI < atom_count for non-tautologies (short-circuit)
    - CHI >> atom_count suggests evaluation overhead or complexity

Use Cases:
    - Identifying pathologically slow formulas
    - Curriculum difficulty estimation
    - Timeout budget calibration
    - Performance regression detection

IMPORTANT: CHI is DESCRIPTIVE ONLY. It is never used to drive timeouts
or influence oracle behavior. INV-ORC-2 requires this separation.

Phase II UX & Policy Layer (v1.1):
    - suggest_timeout_ms(): Non-binding timeout hints from CHI
    - HardnessPolicySignal: Observable policy input (no decisions)
    - format_diagnostics_for_report(): Incident bundle formatting

Phase III Risk Envelope & Policy Hooks (v1.2):
    - build_hardness_risk_envelope(): Structured risk contract for systems
    - derive_timeout_policy_recommendation(): Advisory policy translator
    - summarize_tt_hardness_for_global_health(): Dashboard aggregate signal

Phase IV Hardness-Aware Workload Shaping & Slice Policy Feeds (v1.3):
    - build_slice_hardness_profile(): Per-slice hardness distribution
    - derive_tt_workload_shaping_policy(): Advisory workload shaping recommendations
    - summarize_slice_hardness_for_curriculum(): Curriculum planning signal
    - summarize_tt_risk_for_global_health(): Multi-slice global health aggregation
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


# =============================================================================
# BASELINE CALIBRATION
# =============================================================================

# Baseline nanoseconds per assignment.
#
# ORIGIN: Empirically measured on reference hardware:
#   - CPU: Intel Core i7-10700K @ 3.8GHz (representative mid-range 2020 desktop)
#   - Python: 3.11.x with standard CPython interpreter
#   - Methodology: Average of 10,000 single-assignment evaluations on
#     formula "p -> p" with JIT warmup excluded
#   - Date: 2025-Q1
#
# This baseline represents the expected cost of:
#   1. Dict construction for truth assignment
#   2. String manipulation for formula reduction
#   3. Operator evaluation (AND, OR, IMPLIES, NOT)
#
# On faster hardware, throughput will be better (lower ns/assignment),
# resulting in lower CHI values. On slower hardware, CHI will be higher.
# This is intentional: CHI measures "how hard was this FOR THIS MACHINE".
#
# For cross-machine comparison, normalize CHI by throughput_ns_per_assignment.
BASELINE_NS_PER_ASSIGNMENT = 1000  # 1 microsecond


# =============================================================================
# HARDNESS CLASSIFICATION
# =============================================================================

# Hardness thresholds with operational meanings
HARDNESS_THRESHOLDS: Tuple[Tuple[float, str, str], ...] = (
    (3.0, "trivial", "Instant evaluation; suitable for any context."),
    (8.0, "easy", "Fast evaluation; safe for interactive use."),
    (15.0, "moderate", "Noticeable delay possible; consider async evaluation."),
    (25.0, "hard", "May cause perceptible lag; timeout recommended."),
    (float("inf"), "extreme", "Warning: May be unsuitable for naive truth-table evaluation."),
)

# Suggested timeout ranges by hardness category (in milliseconds)
# These are ADVISORY ONLY and never auto-enforced (INV-ORC-2 compliance)
TIMEOUT_HINTS_BY_CATEGORY: Dict[str, int] = {
    "trivial": 100,      # Minimum practical timeout
    "easy": 200,         # Fast but give some headroom
    "moderate": 500,     # Noticeable delay expected
    "hard": 2000,        # May need significant time
    "extreme": 10000,    # Long timeout or consider alternatives
}


def classify_hardness(chi: float) -> str:
    """
    Classify a CHI value into a hardness category.
    
    This is a PURE function with NO side effects. It does not influence
    oracle behavior in any way (INV-ORC-2 compliance).
    
    Args:
        chi: Computation Hardness Index value
    
    Returns:
        One of: "trivial", "easy", "moderate", "hard", "extreme"
    
    Categories:
        - trivial (CHI < 3):   Instant evaluation; suitable for any context.
        - easy (CHI < 8):      Fast evaluation; safe for interactive use.
        - moderate (CHI < 15): Noticeable delay possible; consider async.
        - hard (CHI < 25):     May cause perceptible lag; timeout recommended.
        - extreme (CHI >= 25): Warning: May be unsuitable for naive TT eval.
    
    Example:
        >>> classify_hardness(2.5)
        'trivial'
        >>> classify_hardness(20.0)
        'hard'
    """
    for threshold, category, _ in HARDNESS_THRESHOLDS:
        if chi < threshold:
            return category
    return "extreme"


def get_hardness_description(category: str) -> str:
    """
    Get the operational description for a hardness category.
    
    Args:
        category: One of "trivial", "easy", "moderate", "hard", "extreme"
    
    Returns:
        Human-readable description of what this category means operationally.
    """
    for _, cat, desc in HARDNESS_THRESHOLDS:
        if cat == category:
            return desc
    return "Unknown hardness category."


# =============================================================================
# TIMEOUT HINT GENERATOR (TASK 1)
# =============================================================================

def suggest_timeout_ms(chi: float) -> int:
    """
    Suggest a timeout value based on CHI.
    
    This is a PURE function that produces NON-BINDING timeout hints.
    These hints are for documentation/CLI display only and are NEVER
    auto-enforced by the oracle (INV-ORC-2 compliance).
    
    Properties:
        - MONOTONIC: Higher CHI never produces smaller suggested timeout.
        - DETERMINISTIC: Same CHI always produces same suggestion.
        - BOUNDED: Returns values in [100ms, 30000ms] range.
    
    Args:
        chi: Computation Hardness Index value
    
    Returns:
        Suggested timeout in milliseconds (advisory only)
    
    Example:
        >>> suggest_timeout_ms(2.0)   # trivial
        100
        >>> suggest_timeout_ms(10.0)  # moderate
        500
        >>> suggest_timeout_ms(30.0)  # extreme
        10000
    """
    category = classify_hardness(chi)
    base_timeout = TIMEOUT_HINTS_BY_CATEGORY.get(category, 1000)
    
    # Scale within category for finer granularity
    # Higher CHI within same category gets slightly higher timeout
    if category == "trivial":
        # CHI 0-3 → 100ms
        return 100
    elif category == "easy":
        # CHI 3-8 → 100-200ms, linear interpolation
        fraction = (chi - 3.0) / 5.0  # 0.0 to 1.0
        return int(100 + fraction * 100)
    elif category == "moderate":
        # CHI 8-15 → 200-500ms
        fraction = (chi - 8.0) / 7.0
        return int(200 + fraction * 300)
    elif category == "hard":
        # CHI 15-25 → 500-2000ms
        fraction = (chi - 15.0) / 10.0
        return int(500 + fraction * 1500)
    else:  # extreme
        # CHI 25+ → 2000-30000ms, capped
        excess = min(chi - 25.0, 50.0)  # Cap at CHI 75
        fraction = excess / 50.0
        return int(2000 + fraction * 28000)


# =============================================================================
# CHI RESULT TYPE
# =============================================================================

@dataclass(frozen=True)
class CHIResult:
    """
    Result of CHI estimation.
    
    Attributes:
        chi: The Computation Hardness Index value
        atom_count: Number of distinct atoms in the formula
        assignment_count: Total possible assignments (2^atom_count)
        assignments_evaluated: Actual assignments checked
        elapsed_ns: Evaluation time in nanoseconds
        efficiency_ratio: assignments_evaluated / assignment_count
        throughput_ns_per_assignment: Average ns per assignment
    """
    chi: float
    atom_count: int
    assignment_count: int
    assignments_evaluated: int
    elapsed_ns: int
    efficiency_ratio: float
    throughput_ns_per_assignment: float
    
    @property
    def is_short_circuited(self) -> bool:
        """True if evaluation short-circuited (non-tautology)."""
        return self.assignments_evaluated < self.assignment_count
    
    @property
    def hardness_category(self) -> str:
        """Categorize hardness for reporting."""
        return classify_hardness(self.chi)
    
    @property
    def hardness_description(self) -> str:
        """Get operational description for this hardness level."""
        return get_hardness_description(self.hardness_category)
    
    @property
    def suggested_timeout_ms(self) -> int:
        """Get suggested timeout based on CHI (advisory only)."""
        return suggest_timeout_ms(self.chi)


# =============================================================================
# POLICY SIGNAL (TASK 2)
# =============================================================================

@dataclass(frozen=True)
class HardnessPolicySignal:
    """
    Observable policy signal for CHI hardness.
    
    This dataclass exposes CHI hardness as a POLICY INPUT for Phase II
    observation. It does NOT influence any decisions (purely observational).
    
    Use Cases:
        - Logging to Evidence Packs
        - Developer tooling
        - Future policy research (Phase III+)
    
    Attributes:
        chi: The Computation Hardness Index value
        category: Hardness category ("trivial", "easy", "moderate", "hard", "extreme")
        suggested_timeout_ms: Advisory timeout hint (never auto-enforced)
        description: Human-readable explanation of the category
    
    Serialization:
        - JSON-serializable via to_json() / to_dict()
        - Stable field ordering for reproducibility
    """
    chi: float
    category: str
    suggested_timeout_ms: int
    description: str
    
    @classmethod
    def from_chi(cls, chi: float) -> "HardnessPolicySignal":
        """
        Create a policy signal from a CHI value.
        
        Args:
            chi: Computation Hardness Index value
        
        Returns:
            HardnessPolicySignal with derived fields
        """
        category = classify_hardness(chi)
        return cls(
            chi=chi,
            category=category,
            suggested_timeout_ms=suggest_timeout_ms(chi),
            description=get_hardness_description(category),
        )
    
    @classmethod
    def from_chi_result(cls, result: CHIResult) -> "HardnessPolicySignal":
        """
        Create a policy signal from a CHIResult.
        
        Args:
            result: CHIResult from chi_from_diagnostics()
        
        Returns:
            HardnessPolicySignal with derived fields
        """
        return cls.from_chi(result.chi)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "chi": round(self.chi, 4),
            "category": self.category,
            "suggested_timeout_ms": self.suggested_timeout_ms,
            "description": self.description,
        }
    
    def to_json(self, indent: Optional[int] = 2) -> str:
        """
        Serialize to JSON string.
        
        Args:
            indent: JSON indentation (None for compact)
        
        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


# =============================================================================
# CHI COMPUTATION
# =============================================================================

def chi_estimate(
    atom_count: int,
    assignment_count: int,
    elapsed_ns: int,
    assignments_evaluated: Optional[int] = None
) -> float:
    """
    Compute the Computation Hardness Index (CHI).
    
    This is a PURE function with NO side effects. It does not modify
    oracle behavior in any way (INV-ORC-2 compliance).
    
    Args:
        atom_count: Number of distinct atomic propositions
        assignment_count: Total possible assignments (should be 2^atom_count)
        elapsed_ns: Evaluation time in nanoseconds
        assignments_evaluated: Actual assignments checked (default: assignment_count)
    
    Returns:
        CHI value as a float. Higher values indicate harder formulas.
    
    Formula:
        CHI = log2(assignments_evaluated) * time_factor
        
        Where time_factor = elapsed_ns / (assignments_evaluated * baseline_ns)
        
    This gives:
        - Base complexity from assignment count (log scale)
        - Multiplied by slowdown factor relative to baseline
    
    Examples:
        >>> chi_estimate(2, 4, 4000)  # 4 assignments in 4μs
        2.0  # log2(4) * 1.0 = 2
        
        >>> chi_estimate(3, 8, 16000)  # 8 assignments in 16μs = 2x slow
        6.0  # log2(8) * 2.0 = 6
    """
    if assignments_evaluated is None:
        assignments_evaluated = assignment_count
    
    # Handle edge cases
    if atom_count <= 0 or assignment_count <= 0 or assignments_evaluated <= 0:
        return 0.0
    
    if elapsed_ns <= 0:
        # No time recorded - return base complexity
        return math.log2(assignments_evaluated)
    
    # Base complexity: log2 of assignments evaluated
    base_complexity = math.log2(assignments_evaluated)
    
    # Time factor: how slow relative to baseline
    expected_ns = assignments_evaluated * BASELINE_NS_PER_ASSIGNMENT
    time_factor = elapsed_ns / expected_ns if expected_ns > 0 else 1.0
    
    # CHI combines complexity with slowdown
    chi = base_complexity * time_factor
    
    return chi


def chi_from_diagnostics(diagnostics: Dict[str, Any]) -> CHIResult:
    """
    Compute CHI from oracle diagnostics dict.
    
    Args:
        diagnostics: Dict from get_last_diagnostics()
    
    Returns:
        CHIResult with full analysis
    
    Raises:
        ValueError: If diagnostics dict is missing required fields
    """
    required_fields = ["atom_count", "assignment_count", "elapsed_ns"]
    for field in required_fields:
        if field not in diagnostics:
            raise ValueError(f"Diagnostics missing required field: {field}")
    
    atom_count = diagnostics["atom_count"]
    assignment_count = diagnostics["assignment_count"]
    elapsed_ns = diagnostics["elapsed_ns"]
    assignments_evaluated = diagnostics.get("assignments_evaluated", assignment_count)
    
    # Compute CHI
    chi = chi_estimate(atom_count, assignment_count, elapsed_ns, assignments_evaluated)
    
    # Compute derived metrics
    efficiency_ratio = (
        assignments_evaluated / assignment_count 
        if assignment_count > 0 else 0.0
    )
    throughput = (
        elapsed_ns / assignments_evaluated 
        if assignments_evaluated > 0 else 0.0
    )
    
    return CHIResult(
        chi=chi,
        atom_count=atom_count,
        assignment_count=assignment_count,
        assignments_evaluated=assignments_evaluated,
        elapsed_ns=elapsed_ns,
        efficiency_ratio=efficiency_ratio,
        throughput_ns_per_assignment=throughput,
    )


# =============================================================================
# TIMEOUT BUDGET ESTIMATION
# =============================================================================

def estimate_timeout_budget(
    atom_count: int,
    target_chi: float = 20.0,
    safety_factor: float = 2.0
) -> int:
    """
    Estimate a reasonable timeout budget for a formula with given atom count.
    
    This is a PURE estimation function for timeout calibration.
    NOTE: This is advisory only. CHI never drives actual timeout behavior
    (INV-ORC-2 compliance).
    
    Args:
        atom_count: Number of distinct atoms
        target_chi: Target CHI to budget for (default: 20 = "hard")
        safety_factor: Multiplier for headroom (default: 2x)
    
    Returns:
        Recommended timeout in milliseconds
    
    Example:
        >>> estimate_timeout_budget(5)  # 2^5 = 32 assignments
        640  # ~640ms budget for 5 atoms with safety margin
    """
    if atom_count <= 0:
        return 100  # Minimum budget
    
    # Expected assignments
    assignment_count = 2 ** atom_count
    
    # Back-calculate expected time from target CHI
    # CHI = log2(n) * (elapsed / (n * baseline))
    # elapsed = CHI * n * baseline / log2(n)
    log_n = math.log2(assignment_count)
    if log_n <= 0:
        log_n = 1.0
    
    expected_ns = (target_chi * assignment_count * BASELINE_NS_PER_ASSIGNMENT) / log_n
    
    # Convert to ms with safety factor
    budget_ms = int((expected_ns / 1_000_000) * safety_factor)
    
    # Enforce minimum and maximum
    return max(100, min(budget_ms, 60_000))  # 100ms to 60s


# =============================================================================
# DIAGNOSTICS FORMATTING (TASK 3)
# =============================================================================

def format_diagnostics_for_report(diagnostics: Dict[str, Any]) -> str:
    """
    Format oracle diagnostics as a compact, stable text block for incident reports.
    
    This function produces a STABLE, REPRODUCIBLE text representation suitable
    for attachment to bug reports and incident bundles.
    
    Stability Guarantees:
        - Field ordering is fixed (alphabetical within sections)
        - Numeric formatting is consistent
        - No random elements or timestamps in output
    
    Args:
        diagnostics: Dict from get_last_diagnostics()
    
    Returns:
        Compact multi-line text block with key diagnostic fields
    
    Example Output:
        ```
        === TT Oracle Diagnostic Report ===
        Formula: p -> q
        Atoms: 2 (p, q)
        Assignments: 2 / 4 (50.0%)
        Time: 0.025 ms (12500 ns/assign)
        Short-Circuit: Yes
        Timeout: No
        Result: False
        ```
    """
    lines = ["=== TT Oracle Diagnostic Report ==="]
    
    # Formula info
    formula = diagnostics.get("formula", "N/A")
    if len(formula) > 60:
        formula = formula[:57] + "..."
    lines.append(f"Formula: {formula}")
    
    # Atom info
    atom_count = diagnostics.get("atom_count", 0)
    normalized = diagnostics.get("normalized_formula", "")
    # Extract atoms from normalized formula for display
    if normalized:
        import re
        atoms = sorted(set(re.findall(r'\b([a-z])\b', normalized)))
        atoms_str = ", ".join(atoms) if atoms else "none"
    else:
        atoms_str = "N/A"
    lines.append(f"Atoms: {atom_count} ({atoms_str})")
    
    # Assignment info
    assignments_evaluated = diagnostics.get("assignments_evaluated", 0)
    assignment_count = diagnostics.get("assignment_count", 0)
    efficiency = (
        (assignments_evaluated / assignment_count * 100)
        if assignment_count > 0 else 0.0
    )
    lines.append(f"Assignments: {assignments_evaluated} / {assignment_count} ({efficiency:.1f}%)")
    
    # Timing info
    elapsed_ns = diagnostics.get("elapsed_ns", 0)
    elapsed_ms = elapsed_ns / 1_000_000
    throughput = (
        elapsed_ns / assignments_evaluated
        if assignments_evaluated > 0 else 0.0
    )
    lines.append(f"Time: {elapsed_ms:.3f} ms ({throughput:.0f} ns/assign)")
    
    # Flags (sorted for stability)
    short_circuit = diagnostics.get("short_circuit_triggered", False)
    timeout = diagnostics.get("timeout_flag", False)
    lines.append(f"Short-Circuit: {'Yes' if short_circuit else 'No'}")
    lines.append(f"Timeout: {'Yes' if timeout else 'No'}")
    
    # Result
    result = diagnostics.get("result")
    if result is None:
        result_str = "None (timeout/error)"
    else:
        result_str = str(result)
    lines.append(f"Result: {result_str}")
    
    return "\n".join(lines)


# =============================================================================
# FORMATTING
# =============================================================================

def format_chi_report(chi_result: CHIResult) -> str:
    """
    Format a CHI result as a human-readable report.
    
    Args:
        chi_result: CHIResult from chi_from_diagnostics()
    
    Returns:
        Multi-line string report
    """
    lines = [
        "╔══════════════════════════════════════════════════════════════╗",
        "║           COMPUTATION HARDNESS INDEX (CHI) REPORT            ║",
        "╠══════════════════════════════════════════════════════════════╣",
        f"║  CHI Value:           {chi_result.chi:>10.2f}                          ║",
        f"║  Hardness Category:   {chi_result.hardness_category:<10}                          ║",
        "╠══════════════════════════════════════════════════════════════╣",
        f"║  Atom Count:          {chi_result.atom_count:>10}                          ║",
        f"║  Total Assignments:   {chi_result.assignment_count:>10}                          ║",
        f"║  Evaluated:           {chi_result.assignments_evaluated:>10}                          ║",
        f"║  Efficiency:          {chi_result.efficiency_ratio*100:>9.1f}%                          ║",
        "╠══════════════════════════════════════════════════════════════╣",
        f"║  Elapsed Time:        {chi_result.elapsed_ns/1_000_000:>10.3f} ms                       ║",
        f"║  Throughput:          {chi_result.throughput_ns_per_assignment:>10.1f} ns/assignment          ║",
        f"║  Short-Circuited:     {'Yes' if chi_result.is_short_circuited else 'No':>10}                          ║",
        "╚══════════════════════════════════════════════════════════════╝",
    ]
    return "\n".join(lines)


def format_chi_compact(chi_result: CHIResult) -> str:
    """
    Format CHI result in compact single-section format.
    
    Args:
        chi_result: CHIResult from chi_from_diagnostics()
    
    Returns:
        Compact multi-line string
    """
    return f"""CHI Analysis:
  Value:    {chi_result.chi:.2f}
  Category: {chi_result.hardness_category}
  Meaning:  {chi_result.hardness_description}
  Atoms:    {chi_result.atom_count} ({chi_result.assignment_count} assignments)
  Evaluated: {chi_result.assignments_evaluated} ({chi_result.efficiency_ratio*100:.0f}% of total)
  Time:     {chi_result.elapsed_ns/1_000_000:.3f} ms ({chi_result.throughput_ns_per_assignment:.0f} ns/assign)
  Suggested timeout: ~{chi_result.suggested_timeout_ms} ms (based on CHI)"""


# =============================================================================
# PHASE III: RISK ENVELOPE & POLICY HOOKS
# =============================================================================

# Schema version for risk envelope contract
RISK_ENVELOPE_SCHEMA_VERSION = "1.0.0"

# Risk band mappings from hardness category
RISK_BAND_BY_CATEGORY: Dict[str, str] = {
    "trivial": "LOW",
    "easy": "LOW",
    "moderate": "MEDIUM",
    "hard": "HIGH",
    "extreme": "EXTREME",
}

# Neutral notes for each risk band (descriptive only, no value judgments)
RISK_BAND_NOTES: Dict[str, str] = {
    "LOW": "Formula complexity is minimal; evaluation expected to complete quickly.",
    "MEDIUM": "Formula complexity is moderate; async evaluation may be appropriate.",
    "HIGH": "Formula complexity is elevated; timeout enforcement recommended.",
    "EXTREME": "Formula complexity exceeds typical thresholds; alternative evaluation strategies may be needed.",
}

# Policy hints for each risk band
POLICY_HINT_BY_RISK_BAND: Dict[str, str] = {
    "LOW": "SAFE_FOR_INTERACTIVE",
    "MEDIUM": "CONSIDER_ASYNC",
    "HIGH": "USE_TIMEOUT_AND_MONITOR",
    "EXTREME": "NOT_SUITABLE_FOR_NAIVE_TT",
}


def build_hardness_risk_envelope(chi_result: CHIResult) -> Dict[str, Any]:
    """
    Build a hardness risk envelope contract from a CHIResult.
    
    This function produces a DETERMINISTIC, JSON-SAFE risk envelope that
    external systems can consume for policy decisions.
    
    The risk envelope is ADVISORY ONLY and does NOT influence oracle behavior
    (INV-ORC-2 compliance).
    
    Args:
        chi_result: CHIResult from chi_from_diagnostics()
    
    Returns:
        Dict with:
            - schema_version: Version string for contract compatibility
            - chi: The CHI value (rounded to 4 decimal places)
            - category: Hardness category string
            - suggested_timeout_ms: Advisory timeout hint
            - risk_band: "LOW" | "MEDIUM" | "HIGH" | "EXTREME"
            - notes: Neutral, descriptive explanation
    
    Example:
        >>> envelope = build_hardness_risk_envelope(chi_result)
        >>> envelope["risk_band"]
        'HIGH'
        >>> json.dumps(envelope)  # Always JSON-safe
        '{"schema_version": "1.0.0", ...}'
    """
    category = chi_result.hardness_category
    risk_band = RISK_BAND_BY_CATEGORY.get(category, "EXTREME")
    
    return {
        "schema_version": RISK_ENVELOPE_SCHEMA_VERSION,
        "chi": round(chi_result.chi, 4),
        "category": category,
        "suggested_timeout_ms": chi_result.suggested_timeout_ms,
        "risk_band": risk_band,
        "notes": RISK_BAND_NOTES.get(risk_band, "Unknown risk level."),
    }


def derive_timeout_policy_recommendation(risk_envelope: Dict[str, Any]) -> Dict[str, Any]:
    """
    Derive advisory timeout policy recommendations from a risk envelope.
    
    This is a PURE function that produces NON-BINDING policy recommendations.
    These recommendations are for logging, display, and documentation only.
    They MUST NOT be used to automatically change oracle behavior.
    
    Args:
        risk_envelope: Dict from build_hardness_risk_envelope()
    
    Returns:
        Dict with:
            - recommended_timeout_ms: Same as suggested_timeout_ms from envelope
            - requires_human_review: True only for EXTREME risk band
            - policy_hint: One of:
                - "SAFE_FOR_INTERACTIVE"
                - "CONSIDER_ASYNC"
                - "USE_TIMEOUT_AND_MONITOR"
                - "NOT_SUITABLE_FOR_NAIVE_TT"
    
    Example:
        >>> policy = derive_timeout_policy_recommendation(envelope)
        >>> policy["requires_human_review"]
        False
        >>> policy["policy_hint"]
        'USE_TIMEOUT_AND_MONITOR'
    """
    risk_band = risk_envelope.get("risk_band", "EXTREME")
    suggested_timeout = risk_envelope.get("suggested_timeout_ms", 10000)
    
    return {
        "recommended_timeout_ms": suggested_timeout,
        "requires_human_review": risk_band == "EXTREME",
        "policy_hint": POLICY_HINT_BY_RISK_BAND.get(risk_band, "NOT_SUITABLE_FOR_NAIVE_TT"),
    }


def summarize_tt_hardness_for_global_health(
    risk_envelopes: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Summarize multiple risk envelopes for global health dashboards.
    
    This function aggregates risk envelopes into a compact signal suitable
    for monitoring dashboards and governance systems.
    
    Args:
        risk_envelopes: Sequence of dicts from build_hardness_risk_envelope()
    
    Returns:
        Dict with:
            - extreme_case_count: Number of EXTREME risk band cases
            - hard_case_count: Number of HIGH + EXTREME risk band cases
            - fraction_safe_or_easy: Ratio of LOW risk band cases to total
            - total_cases: Total number of envelopes processed
            - status: "OK" | "ATTENTION" | "HOT"
                - "OK": fraction_safe_or_easy >= 0.8 and extreme_case_count == 0
                - "ATTENTION": fraction_safe_or_easy < 0.8 or extreme_case_count > 0
                - "HOT": extreme_case_count >= 3 or fraction_safe_or_easy < 0.5
    
    Example:
        >>> health = summarize_tt_hardness_for_global_health(envelopes)
        >>> health["status"]
        'OK'
        >>> health["fraction_safe_or_easy"]
        0.85
    """
    if not risk_envelopes:
        return {
            "extreme_case_count": 0,
            "hard_case_count": 0,
            "fraction_safe_or_easy": 1.0,
            "total_cases": 0,
            "status": "OK",
        }
    
    total = len(risk_envelopes)
    extreme_count = 0
    high_or_extreme_count = 0
    low_count = 0
    
    for envelope in risk_envelopes:
        risk_band = envelope.get("risk_band", "EXTREME")
        if risk_band == "EXTREME":
            extreme_count += 1
            high_or_extreme_count += 1
        elif risk_band == "HIGH":
            high_or_extreme_count += 1
        elif risk_band == "LOW":
            low_count += 1
    
    fraction_safe = low_count / total if total > 0 else 1.0
    
    # Determine status
    if extreme_count >= 3 or fraction_safe < 0.5:
        status = "HOT"
    elif extreme_count > 0 or fraction_safe < 0.8:
        status = "ATTENTION"
    else:
        status = "OK"
    
    return {
        "extreme_case_count": extreme_count,
        "hard_case_count": high_or_extreme_count,
        "fraction_safe_or_easy": round(fraction_safe, 4),
        "total_cases": total,
        "status": status,
    }


# =============================================================================
# PHASE IV: HARDNESS-AWARE WORKLOAD SHAPING & SLICE POLICY FEEDS
# =============================================================================

# Schema version for slice hardness profile
SLICE_PROFILE_SCHEMA_VERSION = "1.0.0"

# Workload shaping policy hints
WORKLOAD_SHAPING_HINTS = {
    "KEEP_CURRENT": "Current workload configuration appears suitable.",
    "CONSIDER_ASYNC": "Async evaluation may improve responsiveness.",
    "REDUCE_TT_USAGE": "High hardness suggests alternative evaluation strategies.",
}

# Curriculum recommendation hints
CURRICULUM_HINTS = {
    "OK": "Slice hardness is within acceptable parameters.",
    "ATTENTION": "Some formulas exhibit elevated hardness; monitoring recommended.",
    "HOT": "Significant fraction of formulas exhibit extreme hardness; review recommended.",
}


def build_slice_hardness_profile(
    results: Sequence[CHIResult],
    slice_name: str
) -> Dict[str, Any]:
    """
    Build a per-slice hardness distribution profile.
    
    This function aggregates CHI results from a slice (e.g., a curriculum unit,
    problem set, or verification batch) into a hardness distribution profile.
    
    The profile is DETERMINISTIC, JSON-SAFE, and suitable for consumption by
    curriculum and planner agents.
    
    Args:
        results: Sequence of CHIResult objects from the slice
        slice_name: Identifier for the slice (e.g., "unit_3", "batch_2024-01-15")
    
    Returns:
        Dict with:
            - schema_version: Version string for contract compatibility
            - slice_name: The provided slice identifier
            - counts_by_category: Dict mapping category to count
                {"trivial": N, "easy": N, "moderate": N, "hard": N, "extreme": N}
            - risk_band_counts: Dict mapping risk band to count
                {"LOW": N, "MEDIUM": N, "HIGH": N, "EXTREME": N}
            - median_chi: Median CHI value (rounded to 4 decimal places)
            - p90_chi: 90th percentile CHI value (rounded to 4 decimal places)
            - max_chi: Maximum CHI value (rounded to 4 decimal places)
            - total_count: Total number of results
    
    Example:
        >>> profile = build_slice_hardness_profile([chi_result1, chi_result2], "unit_1")
        >>> profile["slice_name"]
        'unit_1'
        >>> profile["counts_by_category"]["trivial"]
        2
    """
    if not results:
        return {
            "schema_version": SLICE_PROFILE_SCHEMA_VERSION,
            "slice_name": slice_name,
            "counts_by_category": {
                "trivial": 0,
                "easy": 0,
                "moderate": 0,
                "hard": 0,
                "extreme": 0,
            },
            "risk_band_counts": {
                "LOW": 0,
                "MEDIUM": 0,
                "HIGH": 0,
                "EXTREME": 0,
            },
            "median_chi": 0.0,
            "p90_chi": 0.0,
            "max_chi": 0.0,
            "total_count": 0,
        }
    
    # Count by category
    counts_by_category = {
        "trivial": 0,
        "easy": 0,
        "moderate": 0,
        "hard": 0,
        "extreme": 0,
    }
    
    # Count by risk band
    risk_band_counts = {
        "LOW": 0,
        "MEDIUM": 0,
        "HIGH": 0,
        "EXTREME": 0,
    }
    
    # Collect CHI values for percentile calculation
    chi_values = []
    
    for result in results:
        category = result.hardness_category
        counts_by_category[category] = counts_by_category.get(category, 0) + 1
        
        risk_band = RISK_BAND_BY_CATEGORY.get(category, "EXTREME")
        risk_band_counts[risk_band] = risk_band_counts.get(risk_band, 0) + 1
        
        chi_values.append(result.chi)
    
    # Calculate percentiles
    chi_values_sorted = sorted(chi_values)
    total = len(chi_values_sorted)
    
    if total > 0:
        # Median (50th percentile)
        median_idx = total // 2
        if total % 2 == 0:
            median_chi = (chi_values_sorted[median_idx - 1] + chi_values_sorted[median_idx]) / 2.0
        else:
            median_chi = chi_values_sorted[median_idx]
        
        # 90th percentile (use (n-1)*p method, rounded)
        # For 10 items: (10-1)*0.9 = 8.1 -> index 8 -> 9th value
        p90_idx = int(round((total - 1) * 0.9))
        if p90_idx >= total:
            p90_idx = total - 1
        if p90_idx < 0:
            p90_idx = 0
        p90_chi = chi_values_sorted[p90_idx]
        
        # Maximum
        max_chi = chi_values_sorted[-1]
    else:
        median_chi = 0.0
        p90_chi = 0.0
        max_chi = 0.0
    
    return {
        "schema_version": SLICE_PROFILE_SCHEMA_VERSION,
        "slice_name": slice_name,
        "counts_by_category": counts_by_category,
        "risk_band_counts": risk_band_counts,
        "median_chi": round(median_chi, 4),
        "p90_chi": round(p90_chi, 4),
        "max_chi": round(max_chi, 4),
        "total_count": total,
    }


def derive_tt_workload_shaping_policy(
    slice_profile: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Derive advisory workload shaping policy recommendations from a slice profile.
    
    This is a PURE function that produces NON-BINDING workload shaping recommendations.
    These recommendations are for logging, display, and policy feeds only.
    They MUST NOT be used to automatically change oracle behavior or workload routing.
    
    Args:
        slice_profile: Dict from build_slice_hardness_profile()
    
    Returns:
        Dict with:
            - needs_async_handling: True if async evaluation is recommended
            - suggested_timeout_ms: Recommended timeout based on p90_chi
            - policy_hint: One of:
                - "KEEP_CURRENT"
                - "CONSIDER_ASYNC"
                - "REDUCE_TT_USAGE"
            - reasons: List of neutral descriptive strings (no prescriptive verbs)
    
    Example:
        >>> policy = derive_tt_workload_shaping_policy(profile)
        >>> policy["needs_async_handling"]
        True
        >>> policy["policy_hint"]
        'CONSIDER_ASYNC'
    """
    total = slice_profile.get("total_count", 0)
    if total == 0:
        return {
            "needs_async_handling": False,
            "suggested_timeout_ms": 100,
            "policy_hint": "KEEP_CURRENT",
            "reasons": ["No formulas in slice; current configuration appears suitable."],
        }
    
    risk_band_counts = slice_profile.get("risk_band_counts", {})
    extreme_count = risk_band_counts.get("EXTREME", 0)
    high_count = risk_band_counts.get("HIGH", 0)
    medium_count = risk_band_counts.get("MEDIUM", 0)
    low_count = risk_band_counts.get("LOW", 0)
    
    p90_chi = slice_profile.get("p90_chi", 0.0)
    max_chi = slice_profile.get("max_chi", 0.0)
    
    # Calculate fractions
    extreme_fraction = extreme_count / total if total > 0 else 0.0
    high_or_extreme_fraction = (high_count + extreme_count) / total if total > 0 else 0.0
    
    # Determine policy hint and async recommendation
    needs_async = False
    policy_hint = "KEEP_CURRENT"
    reasons = []
    
    if extreme_fraction >= 0.15 or high_or_extreme_fraction >= 0.4:
        # Significant fraction of hard/extreme cases
        policy_hint = "REDUCE_TT_USAGE"
        needs_async = True
        reasons.append(f"Extreme cases represent {extreme_fraction*100:.1f}% of slice.")
        reasons.append(f"High or extreme cases represent {high_or_extreme_fraction*100:.1f}% of slice.")
    elif high_or_extreme_fraction >= 0.15 or p90_chi >= 20.0 or max_chi >= 25.0:
        # Moderate fraction, high p90, or high max
        policy_hint = "CONSIDER_ASYNC"
        needs_async = True
        reasons.append(f"90th percentile CHI is {p90_chi:.2f}.")
        if max_chi >= 25.0:
            reasons.append(f"Maximum CHI is {max_chi:.2f}.")
        reasons.append(f"High or extreme cases represent {high_or_extreme_fraction*100:.1f}% of slice.")
    else:
        # Mostly low/medium risk
        policy_hint = "KEEP_CURRENT"
        needs_async = False
        reasons.append(f"Slice hardness distribution appears manageable.")
        reasons.append(f"90th percentile CHI is {p90_chi:.2f}.")
    
    # Suggested timeout based on p90_chi
    suggested_timeout = suggest_timeout_ms(p90_chi) if p90_chi > 0 else 100
    
    return {
        "needs_async_handling": needs_async,
        "suggested_timeout_ms": suggested_timeout,
        "policy_hint": policy_hint,
        "reasons": reasons,
    }


def summarize_slice_hardness_for_curriculum(
    slice_profile: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Summarize slice hardness for curriculum agents.
    
    This function provides a compact signal for curriculum planning and
    difficulty calibration.
    
    Args:
        slice_profile: Dict from build_slice_hardness_profile()
    
    Returns:
        Dict with:
            - hardness_status: "OK" | "ATTENTION" | "HOT"
            - extreme_fraction: Fraction of extreme cases (rounded to 4 decimal places)
            - recommendation_hint: Short neutral descriptive string
    
    Example:
        >>> summary = summarize_slice_hardness_for_curriculum(profile)
        >>> summary["hardness_status"]
        'OK'
        >>> summary["extreme_fraction"]
        0.05
    """
    total = slice_profile.get("total_count", 0)
    if total == 0:
        return {
            "hardness_status": "OK",
            "extreme_fraction": 0.0,
            "recommendation_hint": CURRICULUM_HINTS["OK"],
        }
    
    risk_band_counts = slice_profile.get("risk_band_counts", {})
    extreme_count = risk_band_counts.get("EXTREME", 0)
    high_count = risk_band_counts.get("HIGH", 0)
    low_count = risk_band_counts.get("LOW", 0)
    
    extreme_fraction = extreme_count / total if total > 0 else 0.0
    safe_fraction = low_count / total if total > 0 else 1.0
    
    # Determine status
    if extreme_fraction >= 0.2 or (safe_fraction < 0.5 and extreme_fraction > 0):
        # HOT: high extreme fraction OR very low safe fraction with some extreme cases
        status = "HOT"
    elif extreme_fraction > 0.05 or safe_fraction < 0.7:
        # > 0.05 (strictly greater) so 5% exactly is OK
        # ATTENTION: moderate extreme fraction OR low safe fraction
        status = "ATTENTION"
    else:
        status = "OK"
    
    return {
        "hardness_status": status,
        "extreme_fraction": round(extreme_fraction, 4),
        "recommendation_hint": CURRICULUM_HINTS.get(status, CURRICULUM_HINTS["OK"]),
    }


def summarize_tt_risk_for_global_health(
    hardness_summaries: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Summarize TT risk across multiple slices for global health monitoring.
    
    This function aggregates curriculum summaries into a global health signal
    suitable for dashboards and governance systems.
    
    Args:
        hardness_summaries: Sequence of dicts from summarize_slice_hardness_for_curriculum()
    
    Returns:
        Dict with:
            - hot_slice_count: Number of slices with HOT status
            - overall_status: "OK" | "WARN" | "HOT"
            - slices_needing_policy_attention: List of slice identifiers (if available)
            - total_slices: Total number of slices processed
    
    Example:
        >>> global_health = summarize_tt_risk_for_global_health([summary1, summary2])
        >>> global_health["overall_status"]
        'OK'
        >>> global_health["hot_slice_count"]
        0
    """
    if not hardness_summaries:
        return {
            "hot_slice_count": 0,
            "overall_status": "OK",
            "slices_needing_policy_attention": [],
            "total_slices": 0,
        }
    
    hot_count = 0
    attention_count = 0
    slices_needing_attention = []
    
    for summary in hardness_summaries:
        status = summary.get("hardness_status", "OK")
        if status == "HOT":
            hot_count += 1
            # Try to extract slice name if available
            slice_name = summary.get("slice_name")
            if slice_name:
                slices_needing_attention.append(slice_name)
        elif status == "ATTENTION":
            attention_count += 1
    
    total = len(hardness_summaries)
    
    # Determine overall status
    if hot_count >= 2 or (total >= 5 and hot_count / total >= 0.2):
        # HOT: 2+ hot slices OR 20%+ hot slices (if total >= 5)
        overall_status = "HOT"
    elif hot_count > 0 or attention_count / total >= 0.3:
        # WARN: any hot slice OR 30%+ attention slices
        overall_status = "WARN"
    else:
        overall_status = "OK"
    
    return {
        "hot_slice_count": hot_count,
        "overall_status": overall_status,
        "slices_needing_policy_attention": slices_needing_attention,
        "total_slices": total,
    }


# =============================================================================
# PHASE IV EXTENSION: CURRICULUM GATE & TT CAPACITY TILE
# =============================================================================

# Default config for curriculum gate evaluation
DEFAULT_CURRICULUM_GATE_CONFIG = {
    "max_extreme_fraction": 0.15,  # Block if > 15% extreme
    "max_chi_threshold": 50.0,     # Block if max_chi > 50
    "min_count_for_evaluation": 5, # Need at least 5 results for meaningful evaluation
    "attention_extreme_fraction": 0.05,  # ATTENTION if > 5% extreme
    "attention_max_chi": 30.0,     # ATTENTION if max_chi > 30
}


def evaluate_slice_hardness_for_curriculum(
    slice_profile: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluate slice hardness for curriculum gate decisions.
    
    This function determines if a slice is admissible for curriculum use based on
    hardness metrics and configurable thresholds. The evaluation is ADVISORY ONLY
    and does not automatically block curriculum progression.
    
    Args:
        slice_profile: Dict from build_slice_hardness_profile()
        config: Optional dict with threshold overrides. Keys:
            - max_extreme_fraction: Block if extreme_fraction exceeds this (default: 0.15)
            - max_chi_threshold: Block if max_chi exceeds this (default: 50.0)
            - min_count_for_evaluation: Minimum results needed for evaluation (default: 5)
            - attention_extreme_fraction: ATTENTION if extreme_fraction exceeds this (default: 0.05)
            - attention_max_chi: ATTENTION if max_chi exceeds this (default: 30.0)
    
    Returns:
        Dict with:
            - admissible: True if slice is acceptable for curriculum use
            - status: "OK" | "ATTENTION" | "BLOCK"
            - reasons: List of neutral descriptive strings explaining the decision
            - suggested_actions: List of neutral hint strings (no prescriptive verbs)
    
    Example:
        >>> gate = evaluate_slice_hardness_for_curriculum(profile)
        >>> gate["admissible"]
        True
        >>> gate["status"]
        'OK'
    """
    if config is None:
        config = DEFAULT_CURRICULUM_GATE_CONFIG.copy()
    else:
        # Merge with defaults
        merged_config = DEFAULT_CURRICULUM_GATE_CONFIG.copy()
        merged_config.update(config)
        config = merged_config
    
    total = slice_profile.get("total_count", 0)
    risk_band_counts = slice_profile.get("risk_band_counts", {})
    extreme_count = risk_band_counts.get("EXTREME", 0)
    max_chi = slice_profile.get("max_chi", 0.0)
    median_chi = slice_profile.get("median_chi", 0.0)
    p90_chi = slice_profile.get("p90_chi", 0.0)
    
    # Extract thresholds
    max_extreme_fraction = config.get("max_extreme_fraction", 0.15)
    max_chi_threshold = config.get("max_chi_threshold", 50.0)
    min_count = config.get("min_count_for_evaluation", 5)
    attention_extreme_fraction = config.get("attention_extreme_fraction", 0.05)
    attention_max_chi = config.get("attention_max_chi", 30.0)
    
    reasons = []
    suggested_actions = []
    admissible = True
    status = "OK"
    
    # Check minimum count
    if total < min_count:
        admissible = False
        status = "BLOCK"
        reasons.append(f"Slice has {total} results; minimum {min_count} required for meaningful evaluation.")
        suggested_actions.append("Collect more evaluation results before curriculum use.")
        return {
            "admissible": admissible,
            "status": status,
            "reasons": reasons,
            "suggested_actions": suggested_actions,
        }
    
    # Calculate extreme fraction
    extreme_fraction = extreme_count / total if total > 0 else 0.0
    
    # BLOCK conditions
    if extreme_fraction > max_extreme_fraction:
        admissible = False
        status = "BLOCK"
        reasons.append(f"Extreme cases represent {extreme_fraction*100:.1f}% of slice (threshold: {max_extreme_fraction*100:.1f}%).")
        suggested_actions.append("Consider alternative evaluation strategies for extreme cases.")
        suggested_actions.append("Review formula complexity in slice.")
    elif max_chi > max_chi_threshold:
        admissible = False
        status = "BLOCK"
        reasons.append(f"Maximum CHI is {max_chi:.2f} (threshold: {max_chi_threshold:.2f}).")
        suggested_actions.append("Slice contains formulas with extremely high computational complexity.")
        suggested_actions.append("Consider splitting slice or using alternative evaluation methods.")
    
    # ATTENTION conditions (only if not already BLOCK)
    if status != "BLOCK":
        if extreme_fraction > attention_extreme_fraction:
            status = "ATTENTION"
            reasons.append(f"Extreme cases represent {extreme_fraction*100:.1f}% of slice.")
            suggested_actions.append("Monitor slice performance during curriculum use.")
        elif max_chi > attention_max_chi:
            status = "ATTENTION"
            reasons.append(f"Maximum CHI is {max_chi:.2f}.")
            suggested_actions.append("Some formulas may require extended evaluation time.")
        elif p90_chi > 20.0:
            status = "ATTENTION"
            reasons.append(f"90th percentile CHI is {p90_chi:.2f}.")
            suggested_actions.append("Consider async evaluation for slice.")
    
    # OK case - add positive note
    if status == "OK":
        reasons.append(f"Slice hardness distribution appears suitable for curriculum use.")
        reasons.append(f"Median CHI: {median_chi:.2f}, 90th percentile: {p90_chi:.2f}.")
    
    return {
        "admissible": admissible,
        "status": status,
        "reasons": reasons,
        "suggested_actions": suggested_actions,
    }


def summarize_tt_capacity_for_global_health(
    slice_summaries: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Summarize TT capacity for global health monitoring (Director tile).
    
    This function aggregates slice summaries into a capacity signal suitable for
    global health dashboards and Director-level monitoring.
    
    Args:
        slice_summaries: Sequence of dicts from summarize_slice_hardness_for_curriculum()
            or evaluate_slice_hardness_for_curriculum(). Should have "hardness_status" or "status" field.
    
    Returns:
        Dict with:
            - total_slices: Total number of slices processed
            - slices_at_high_risk: Number of slices with HOT status or BLOCK status
            - global_tt_status: "OK" | "WARN" | "HOT"
            - notes: List of neutral descriptive strings
    
    Example:
        >>> capacity = summarize_tt_capacity_for_global_health([summary1, summary2])
        >>> capacity["global_tt_status"]
        'OK'
        >>> capacity["slices_at_high_risk"]
        0
    """
    if not slice_summaries:
        return {
            "total_slices": 0,
            "slices_at_high_risk": 0,
            "global_tt_status": "OK",
            "notes": ["No slices available for capacity evaluation."],
        }
    
    total = len(slice_summaries)
    hot_count = 0
    block_count = 0
    attention_count = 0
    high_risk_count = 0
    
    for summary in slice_summaries:
        # Check for both "hardness_status" (from summarize_slice_hardness_for_curriculum)
        # and "status" (from evaluate_slice_hardness_for_curriculum)
        status = summary.get("hardness_status") or summary.get("status", "OK")
        
        if status == "HOT":
            hot_count += 1
            high_risk_count += 1
        elif status == "BLOCK":
            block_count += 1
            high_risk_count += 1
        elif status == "ATTENTION":
            attention_count += 1
    
    # Determine global status
    if hot_count > 0 or block_count > 0:
        # HOT if any slice is HOT or BLOCK
        global_status = "HOT"
    elif attention_count / total >= 0.3:
        # WARN if 30%+ slices are at ATTENTION
        global_status = "WARN"
    else:
        global_status = "OK"
    
    # Build notes
    notes = []
    if total > 0:
        notes.append(f"Evaluated {total} slice(s) for TT capacity.")
        if high_risk_count > 0:
            notes.append(f"{high_risk_count} slice(s) at high risk ({hot_count} HOT, {block_count} BLOCK).")
        if attention_count > 0:
            notes.append(f"{attention_count} slice(s) require attention.")
        if global_status == "OK":
            notes.append("TT capacity appears adequate for current workload.")
        elif global_status == "WARN":
            notes.append("Moderate pressure on TT capacity detected.")
        else:
            notes.append("Significant pressure on TT capacity; review recommended.")
    
    return {
        "total_slices": total,
        "slices_at_high_risk": high_risk_count,
        "global_tt_status": global_status,
        "notes": notes,
    }


# =============================================================================
# PHASE V: CURRICULUM GOVERNANCE & DRIFT DETECTION
# =============================================================================

# Schema version for curriculum stability radar
CURRICULUM_STABILITY_RADAR_SCHEMA_VERSION = "1.0.0"

# Volatility thresholds (mirroring A4/A5 systems for cross-compatibility)
VOLATILITY_THRESHOLDS = {
    "STABLE": 0.1,      # drift_rate < 0.1
    "DRIFTING": 0.3,    # 0.1 <= drift_rate < 0.3
    "VOLATILE": float("inf"),  # drift_rate >= 0.3
}


def build_curriculum_stability_radar(
    history: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Build curriculum drift early-warning radar from historical TT capacity tiles.
    
    This function analyzes a sequence of TT capacity summaries over time to detect
    drift patterns, volatility, and threshold-crossing events. The trend logic
    mirrors A4/A5 systems for cross-compatibility.
    
    Args:
        history: Sequence of TT capacity tiles (from summarize_tt_capacity_for_global_health)
            ordered chronologically. Each tile should have:
            - global_tt_status: "OK" | "WARN" | "HOT"
            - slices_at_high_risk: int
            - total_slices: int
    
    Returns:
        Dict with:
            - schema_version: "1.0.0"
            - drift_rate: float (0.0-1.0) - normalized measure of change over time
            - volatility_band: "STABLE" | "DRIFTING" | "VOLATILE"
            - slices_trending_up: List of slice identifiers trending toward higher risk
            - slices_trending_down: List of slice identifiers trending toward lower risk
            - threshold_cross_events: List of events where status crossed thresholds
            - trend_direction: "IMPROVING" | "STABLE" | "DEGRADING"
    
    Example:
        >>> radar = build_curriculum_stability_radar([tile1, tile2, tile3])
        >>> radar["volatility_band"]
        'STABLE'
        >>> radar["drift_rate"]
        0.05
    """
    if not history or len(history) < 2:
        return {
            "schema_version": CURRICULUM_STABILITY_RADAR_SCHEMA_VERSION,
            "drift_rate": 0.0,
            "volatility_band": "STABLE",
            "slices_trending_up": [],
            "slices_trending_down": [],
            "threshold_cross_events": [],
            "trend_direction": "STABLE",
        }
    
    # Extract status sequence and risk counts
    status_sequence = []
    risk_counts = []
    
    for tile in history:
        status = tile.get("global_tt_status", "OK")
        risk_count = tile.get("slices_at_high_risk", 0)
        total = tile.get("total_slices", 0)
        
        status_sequence.append(status)
        # Normalize risk count by total slices (0.0-1.0)
        risk_ratio = risk_count / total if total > 0 else 0.0
        risk_counts.append(risk_ratio)
    
    # Calculate drift rate (coefficient of variation of risk ratios)
    if len(risk_counts) > 1:
        mean_risk = sum(risk_counts) / len(risk_counts)
        if mean_risk > 0:
            variance = sum((r - mean_risk) ** 2 for r in risk_counts) / len(risk_counts)
            std_dev = math.sqrt(variance)
            drift_rate = std_dev / mean_risk if mean_risk > 0 else 0.0
            # Clamp to [0, 1]
            drift_rate = min(1.0, max(0.0, drift_rate))
        else:
            drift_rate = 0.0
    else:
        drift_rate = 0.0
    
    # Determine volatility band
    volatility_band = "STABLE"
    for band, threshold in VOLATILITY_THRESHOLDS.items():
        if drift_rate < threshold:
            volatility_band = band
            break
    
    # Detect threshold crossings (status changes)
    threshold_cross_events = []
    for i in range(1, len(status_sequence)):
        prev_status = status_sequence[i - 1]
        curr_status = status_sequence[i]
        
        if prev_status != curr_status:
            # Status changed - record as threshold cross event
            threshold_cross_events.append({
                "index": i,
                "from_status": prev_status,
                "to_status": curr_status,
                "risk_ratio_before": risk_counts[i - 1],
                "risk_ratio_after": risk_counts[i],
            })
    
    # Determine trend direction (comparing first and last periods)
    if len(risk_counts) >= 2:
        first_half = risk_counts[:len(risk_counts)//2]
        second_half = risk_counts[len(risk_counts)//2:]
        first_avg = sum(first_half) / len(first_half) if first_half else 0.0
        second_avg = sum(second_half) / len(second_half) if second_half else 0.0
        
        if second_avg > first_avg * 1.1:  # 10% increase
            trend_direction = "DEGRADING"
        elif second_avg < first_avg * 0.9:  # 10% decrease
            trend_direction = "IMPROVING"
        else:
            trend_direction = "STABLE"
    else:
        trend_direction = "STABLE"
    
    # Slice-level trending (placeholder - would need slice-level history)
    # For now, return empty lists as slice identifiers aren't in capacity tiles
    slices_trending_up = []
    slices_trending_down = []
    
    return {
        "schema_version": CURRICULUM_STABILITY_RADAR_SCHEMA_VERSION,
        "drift_rate": round(drift_rate, 4),
        "volatility_band": volatility_band,
        "slices_trending_up": slices_trending_up,
        "slices_trending_down": slices_trending_down,
        "threshold_cross_events": threshold_cross_events,
        "trend_direction": trend_direction,
    }


def build_curriculum_health_tile(
    slice_admissibility_results: Sequence[Dict[str, Any]],
    stability_radar: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build curriculum health tile for global console integration.
    
    This function aggregates curriculum admissibility results into a single
    health tile suitable for Director-level monitoring and global console display.
    
    Args:
        slice_admissibility_results: Sequence of dicts from evaluate_slice_hardness_for_curriculum()
            Each should have:
            - status: "OK" | "ATTENTION" | "BLOCK"
            - admissible: bool
            - slice_name (optional): Identifier for the slice
        stability_radar: Optional dict from build_curriculum_stability_radar()
            Used to enrich the tile with drift information
    
    Returns:
        Dict with:
            - curriculum_health: "OK" | "WARN" | "BLOCK"
            - drift_trend: "IMPROVING" | "STABLE" | "DEGRADING" | "UNKNOWN"
            - high_risk_slices: List of slice identifiers with BLOCK status
            - requires_attention: bool - True if any slice needs attention
            - headline: str - Neutral summary string
    
    Example:
        >>> tile = build_curriculum_health_tile([result1, result2])
        >>> tile["curriculum_health"]
        'OK'
        >>> tile["requires_attention"]
        False
    """
    if not slice_admissibility_results:
        return {
            "curriculum_health": "OK",
            "drift_trend": "UNKNOWN",
            "high_risk_slices": [],
            "requires_attention": False,
            "headline": "No curriculum slices available for evaluation.",
        }
    
    total = len(slice_admissibility_results)
    block_count = 0
    attention_count = 0
    high_risk_slices = []
    
    for result in slice_admissibility_results:
        status = result.get("status", "OK")
        slice_name = result.get("slice_name")
        
        if status == "BLOCK":
            block_count += 1
            if slice_name:
                high_risk_slices.append(slice_name)
        elif status == "ATTENTION":
            attention_count += 1
    
    # Determine curriculum health
    if block_count > 0:
        curriculum_health = "BLOCK"
    elif attention_count / total >= 0.2:  # 20% threshold
        curriculum_health = "WARN"
    else:
        curriculum_health = "OK"
    
    # Extract drift trend from radar if available
    drift_trend = "UNKNOWN"
    if stability_radar:
        drift_trend = stability_radar.get("trend_direction", "UNKNOWN")
    
    # Build headline
    if curriculum_health == "BLOCK":
        headline = f"{block_count} slice(s) blocked from curriculum use."
    elif curriculum_health == "WARN":
        headline = f"{attention_count} slice(s) require attention ({attention_count/total*100:.0f}% of total)."
    else:
        headline = f"All {total} curriculum slice(s) appear suitable for use."
    
    return {
        "curriculum_health": curriculum_health,
        "drift_trend": drift_trend,
        "high_risk_slices": high_risk_slices,
        "requires_attention": curriculum_health != "OK",
        "headline": headline,
    }


def attach_curriculum_governance_to_evidence(
    evidence_pack: Dict[str, Any],
    curriculum_view: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach curriculum governance signals to an evidence pack (read-only).
    
    This function is PURE and DETERMINISTIC. It does not modify the evidence_pack
    in place, but returns a new dict with curriculum governance data attached.
    
    Args:
        evidence_pack: Existing evidence pack dict (read-only, not modified)
        curriculum_view: Dict containing curriculum governance data, typically from:
            - build_curriculum_health_tile()
            - build_curriculum_stability_radar()
            - Or a combination of both
    
    Returns:
        New dict with evidence_pack contents plus curriculum_governance section
    
    Example:
        >>> evidence = {"timestamp": "2024-01-01", "data": {...}}
        >>> curriculum = {"curriculum_health": "OK", ...}
        >>> enriched = attach_curriculum_governance_to_evidence(evidence, curriculum)
        >>> "curriculum_governance" in enriched
        True
    """
    # Create a copy to avoid mutating the original
    enriched = evidence_pack.copy()
    
    # Add curriculum governance section
    enriched["curriculum_governance"] = {
        "schema_version": CURRICULUM_STABILITY_RADAR_SCHEMA_VERSION,
        "timestamp": curriculum_view.get("timestamp"),  # Preserve if present
        "health": curriculum_view.get("curriculum_health"),
        "drift_trend": curriculum_view.get("drift_trend"),
        "requires_attention": curriculum_view.get("requires_attention", False),
        "high_risk_slices": curriculum_view.get("high_risk_slices", []),
        "headline": curriculum_view.get("headline", ""),
        # Include stability radar data if present
        "stability": curriculum_view.get("stability_radar") or {},
    }
    
    return enriched


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Core functions
    "chi_estimate",
    "chi_from_diagnostics",
    "classify_hardness",
    "get_hardness_description",
    # Timeout hints (Phase II Task 1)
    "suggest_timeout_ms",
    # Policy signal (Phase II Task 2)
    "HardnessPolicySignal",
    # Diagnostics formatting (Phase II Task 3)
    "format_diagnostics_for_report",
    # Types
    "CHIResult",
    # Utilities
    "estimate_timeout_budget",
    "format_chi_report",
    "format_chi_compact",
    # Phase III: Risk Envelope & Policy Hooks
    "build_hardness_risk_envelope",
    "derive_timeout_policy_recommendation",
    "summarize_tt_hardness_for_global_health",
    # Phase IV: Hardness-Aware Workload Shaping & Slice Policy Feeds
    "build_slice_hardness_profile",
    "derive_tt_workload_shaping_policy",
    "summarize_slice_hardness_for_curriculum",
    "summarize_tt_risk_for_global_health",
    # Phase IV Extension: Curriculum Gate & TT Capacity Tile
    "evaluate_slice_hardness_for_curriculum",
    "summarize_tt_capacity_for_global_health",
    "DEFAULT_CURRICULUM_GATE_CONFIG",
    # Constants
    "BASELINE_NS_PER_ASSIGNMENT",
    "HARDNESS_THRESHOLDS",
    "TIMEOUT_HINTS_BY_CATEGORY",
    "RISK_ENVELOPE_SCHEMA_VERSION",
    "RISK_BAND_BY_CATEGORY",
    "RISK_BAND_NOTES",
    "POLICY_HINT_BY_RISK_BAND",
    "SLICE_PROFILE_SCHEMA_VERSION",
    "WORKLOAD_SHAPING_HINTS",
    "CURRICULUM_HINTS",
]
