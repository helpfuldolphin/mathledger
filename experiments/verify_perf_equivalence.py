"""
Performance Equivalence and Ratchet Verification
=================================================

PERF ONLY — NO BEHAVIOR CHANGE

This script provides comprehensive performance verification:

1. **Behavioral Equivalence**: Verify that performance optimizations do not
   alter the logical behavior of the CycleRunner (H_t roots, verified counts).

2. **Performance Ratchet**: Verify that optimized code meets a minimum
   improvement threshold over baseline, preventing performance regressions.

3. **SLO Enforcement**: Compare against a checked-in baseline registry
   (config/perf_baseline.json) to enforce long-term performance contracts.

4. **Component-Level Breakdown**: Attribute performance changes to logical
   components (scoring, derivation, verification, etc.) for explainability.

5. **Narrative Generation**: Produce human-readable summaries explaining
   what changed in plain English.

================================================================================
CI INTEGRATION — GitHub Actions Example
================================================================================

```yaml
- name: Verify perf SLO
  run: |
    uv run python experiments/verify_perf_equivalence.py \\
      --baseline results/perf/baseline.json \\
      --optimized results/perf/optimized.json \\
      --slo config/perf_baseline.json \\
      --output-summary results/perf/summary.md
```

================================================================================
SLO Status Levels
================================================================================

| Status | Condition                           | CI Action |
|--------|-------------------------------------|-----------|
| OK     | regression_pct ≤ warn_threshold     | Pass      |
| WARN   | warn < regression_pct ≤ max         | Pass      |
| BLOCK  | regression_pct > block_threshold    | Fail      |

================================================================================
Component-Level Breakdown
================================================================================

If the input JSON contains a "components" field, the script will:
1. Compute per-component avg_ms_baseline vs avg_ms_optimized
2. Calculate regression/improvement percent per component
3. Generate a component table in the Markdown summary

Example input JSON with components:
```json
{
  "avg_time_per_cycle_ms": 250.0,
  "components": {
    "scoring": {"avg_ms": 80.0, "calls": 1000},
    "derivation": {"avg_ms": 120.0, "calls": 500},
    "verification": {"avg_ms": 50.0, "calls": 200}
  }
}
```

================================================================================
SLO Configuration Validation
================================================================================

The SLO config (config/perf_baseline.json) is validated at startup:
- 0 <= warn_regression_pct <= max_regression_pct <= block_regression_pct
- jitter_allowance_pct >= 0
- min_cycles_for_validity > 0

Invalid configurations will cause immediate failure with a clear error message.

================================================================================
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Remove experiments dir from path if present (avoid substrate.py shadowing)
experiments_dir = str(Path(__file__).resolve().parent)
if experiments_dir in sys.path:
    sys.path.remove(experiments_dir)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class SLOConfigError(Exception):
    """Raised when SLO configuration is invalid."""
    pass


# ---------------------------------------------------------------------------
# Status Enum
# ---------------------------------------------------------------------------

class SLOStatus(Enum):
    """SLO enforcement status levels for individual components."""
    OK = "OK"
    WARN = "WARN"
    BREACH = "BREACH"  # Renamed from BLOCK for component-level (BLOCK used for gate)
    BLOCK = "BLOCK"    # Keep for backwards compatibility
    
    @property
    def emoji(self) -> str:
        return {"OK": "✅", "WARN": "⚠️", "BREACH": "🚨", "BLOCK": "❌"}[self.value]
    
    @property
    def ci_pass(self) -> bool:
        return self in (SLOStatus.OK, SLOStatus.WARN)


class GateStatus(Enum):
    """Performance gate status levels for CI integration."""
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    
    @property
    def emoji(self) -> str:
        return {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}[self.value]
    
    @property
    def exit_code(self) -> int:
        """Return appropriate exit code for CI."""
        return {"PASS": 0, "WARN": 0, "FAIL": 1}[self.value]


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class CycleFingerprint:
    """Fingerprint of a single cycle for equivalence checking."""
    cycle_index: int
    h_t: str  # Composite attestation root
    r_t: str  # Reasoning root
    u_t: str  # UI root
    n_verified: int
    n_abstained: int
    n_candidates: int
    status: str
    candidate_hash: str


@dataclass
class EquivalenceReport:
    """Report comparing two sets of cycle fingerprints."""
    matched: bool
    cycles_compared: int
    mismatches: List[Dict[str, Any]]
    seed: int
    slice_name: str
    mode: str


@dataclass
class PerfRatchetResult:
    """Result of performance ratchet check."""
    passed: bool
    baseline_avg_ms: float
    optimized_avg_ms: float
    improvement_pct: float
    required_improvement_pct: float
    message: str


@dataclass
class ComponentMetrics:
    """Performance metrics for a single component."""
    name: str
    baseline_avg_ms: float
    optimized_avg_ms: float
    delta_pct: float
    status: SLOStatus
    
    @classmethod
    def from_data(
        cls,
        name: str,
        baseline_data: Optional[Dict[str, Any]],
        optimized_data: Optional[Dict[str, Any]],
        warn_threshold: float = 5.0,
        block_threshold: float = 25.0,
    ) -> "ComponentMetrics":
        """Create ComponentMetrics from raw component data."""
        baseline_avg = baseline_data.get("avg_ms", 0.0) if baseline_data else 0.0
        optimized_avg = optimized_data.get("avg_ms", 0.0) if optimized_data else 0.0
        
        if baseline_avg > 0:
            # Positive delta = regression (slower), negative = improvement (faster)
            delta_pct = ((optimized_avg - baseline_avg) / baseline_avg) * 100
        else:
            delta_pct = 0.0
        
        # Determine status based on regression
        if delta_pct <= warn_threshold:
            status = SLOStatus.OK
        elif delta_pct <= block_threshold:
            status = SLOStatus.WARN
        else:
            status = SLOStatus.BLOCK
        
        return cls(
            name=name,
            baseline_avg_ms=baseline_avg,
            optimized_avg_ms=optimized_avg,
            delta_pct=delta_pct,
            status=status,
        )


@dataclass
class SLOBaseline:
    """Performance SLO baseline from config/perf_baseline.json."""
    reference_avg_ms: float
    max_regression_pct: float
    warn_regression_pct: float
    block_regression_pct: float
    jitter_allowance_pct: float
    min_cycles_for_validity: int
    slice_name: str
    
    @classmethod
    def from_json(cls, path: Path) -> "SLOBaseline":
        """Load SLO baseline from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        baseline = data.get("baseline", {})
        slo = data.get("slo", {})
        tolerance = data.get("tolerance", {})
        
        instance = cls(
            reference_avg_ms=baseline.get("reference_avg_ms", 250.0),
            max_regression_pct=slo.get("max_regression_pct", 10.0),
            warn_regression_pct=slo.get("warn_regression_pct", 5.0),
            block_regression_pct=slo.get("block_regression_pct", 25.0),
            jitter_allowance_pct=tolerance.get("jitter_allowance_pct", 3.0),
            min_cycles_for_validity=tolerance.get("min_cycles_for_validity", 20),
            slice_name=baseline.get("slice_name", "slice_medium"),
        )
        
        # Validate configuration
        validate_slo_config(instance)
        
        return instance


@dataclass
class SLOResult:
    """Result of SLO enforcement check."""
    status: SLOStatus
    current_avg_ms: float
    reference_avg_ms: float
    baseline_avg_ms: float
    optimized_avg_ms: float
    regression_pct: float
    improvement_pct: float
    message: str
    components: List[ComponentMetrics] = field(default_factory=list)
    narrative: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def ci_pass(self) -> bool:
        return self.status.ci_pass


@dataclass
class ComponentSLOResult:
    """Result of evaluating a single component against its SLO."""
    name: str
    baseline_avg_ms: float
    optimized_avg_ms: float
    delta_pct: float
    status: SLOStatus
    warn_threshold: float
    breach_threshold: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "baseline_avg_ms": self.baseline_avg_ms,
            "optimized_avg_ms": self.optimized_avg_ms,
            "delta_pct": self.delta_pct,
            "status": self.status.value,
            "warn_threshold": self.warn_threshold,
            "breach_threshold": self.breach_threshold,
        }


@dataclass
class ComponentSLOEvaluation:
    """Aggregate result of evaluating all components against SLOs."""
    components: List[ComponentSLOResult]
    any_breach: bool
    worst_offender: Optional[str]
    worst_delta_pct: float
    total_components: int
    breached_count: int
    warned_count: int
    ok_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "components": [c.to_dict() for c in self.components],
            "any_breach": self.any_breach,
            "worst_offender": self.worst_offender,
            "worst_delta_pct": self.worst_delta_pct,
            "total_components": self.total_components,
            "breached_count": self.breached_count,
            "warned_count": self.warned_count,
            "ok_count": self.ok_count,
        }


@dataclass
class PerfGateResult:
    """Result of the performance gate evaluation."""
    gate_status: GateStatus
    component_breaches: List[str]
    component_warnings: List[str]
    short_summary: str
    overall_delta_pct: float
    component_eval: Optional[ComponentSLOEvaluation] = None
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate_status": self.gate_status.value,
            "component_breaches": self.component_breaches,
            "component_warnings": self.component_warnings,
            "short_summary": self.short_summary,
            "overall_delta_pct": self.overall_delta_pct,
            "component_eval": self.component_eval.to_dict() if self.component_eval else None,
            "details": self.details,
        }


@dataclass
class GlobalHealthPerfSummary:
    """Performance summary for global health monitoring."""
    perf_ok: bool
    status: str  # OK|WARN|BLOCK
    components_regressed: List[str]
    overall_delta_pct: float
    worst_component: Optional[str]
    worst_delta_pct: float
    message: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "perf_ok": self.perf_ok,
            "status": self.status,
            "components_regressed": self.components_regressed,
            "overall_delta_pct": self.overall_delta_pct,
            "worst_component": self.worst_component,
            "worst_delta_pct": self.worst_delta_pct,
            "message": self.message,
        }


# ---------------------------------------------------------------------------
# SLO Configuration Validation
# ---------------------------------------------------------------------------

def validate_slo_config(slo: SLOBaseline) -> None:
    """
    Validate SLO configuration for consistency.
    
    Raises:
        SLOConfigError: If configuration is invalid.
    
    Validation rules:
        1. 0 <= warn_regression_pct <= max_regression_pct <= block_regression_pct
        2. jitter_allowance_pct >= 0
        3. min_cycles_for_validity > 0
        4. reference_avg_ms > 0
    """
    errors = []
    
    # Check threshold ordering
    if slo.warn_regression_pct < 0:
        errors.append(f"warn_regression_pct must be >= 0, got {slo.warn_regression_pct}")
    
    if slo.max_regression_pct < slo.warn_regression_pct:
        errors.append(
            f"max_regression_pct ({slo.max_regression_pct}) must be >= "
            f"warn_regression_pct ({slo.warn_regression_pct})"
        )
    
    if slo.block_regression_pct < slo.max_regression_pct:
        errors.append(
            f"block_regression_pct ({slo.block_regression_pct}) must be >= "
            f"max_regression_pct ({slo.max_regression_pct})"
        )
    
    # Check jitter allowance
    if slo.jitter_allowance_pct < 0:
        errors.append(f"jitter_allowance_pct must be >= 0, got {slo.jitter_allowance_pct}")
    
    # Check min cycles
    if slo.min_cycles_for_validity <= 0:
        errors.append(f"min_cycles_for_validity must be > 0, got {slo.min_cycles_for_validity}")
    
    # Check reference avg
    if slo.reference_avg_ms <= 0:
        errors.append(f"reference_avg_ms must be > 0, got {slo.reference_avg_ms}")
    
    if errors:
        raise SLOConfigError(
            "Invalid SLO configuration:\n  - " + "\n  - ".join(errors)
        )


# ---------------------------------------------------------------------------
# Component-Level Analysis
# ---------------------------------------------------------------------------

def extract_components(
    baseline_data: Dict[str, Any],
    optimized_data: Dict[str, Any],
    slo: Optional[SLOBaseline] = None,
) -> List[ComponentMetrics]:
    """
    Extract and compare component-level metrics.
    
    If either input lacks a 'components' field, returns empty list.
    """
    baseline_components = baseline_data.get("components", {})
    optimized_components = optimized_data.get("components", {})
    
    if not baseline_components and not optimized_components:
        return []
    
    # Get union of component names
    all_names = set(baseline_components.keys()) | set(optimized_components.keys())
    
    # Default thresholds
    warn_threshold = slo.warn_regression_pct if slo else 5.0
    block_threshold = slo.block_regression_pct if slo else 25.0
    
    components = []
    for name in sorted(all_names):
        baseline_comp = baseline_components.get(name)
        optimized_comp = optimized_components.get(name)
        
        metrics = ComponentMetrics.from_data(
            name=name,
            baseline_data=baseline_comp,
            optimized_data=optimized_comp,
            warn_threshold=warn_threshold,
            block_threshold=block_threshold,
        )
        components.append(metrics)
    
    # Sort by delta_pct (most regression first)
    components.sort(key=lambda c: c.delta_pct, reverse=True)
    
    return components


# ---------------------------------------------------------------------------
# Narrative Generation
# ---------------------------------------------------------------------------

def generate_narrative(
    overall_improvement_pct: float,
    components: List[ComponentMetrics],
) -> str:
    """
    Generate a human-readable narrative explaining performance changes.
    
    Produces 1-3 factual sentences identifying top changes.
    
    Args:
        overall_improvement_pct: Overall improvement percentage (positive = faster)
        components: List of ComponentMetrics sorted by delta_pct
        
    Returns:
        Human-readable narrative string
    """
    sentences = []
    
    # Overall summary
    if abs(overall_improvement_pct) < 1.0:
        sentences.append("Overall performance is stable (within ±1%).")
    elif overall_improvement_pct > 0:
        sentences.append(f"Overall performance improved by {overall_improvement_pct:.1f}%.")
    else:
        sentences.append(f"Overall performance regressed by {abs(overall_improvement_pct):.1f}%.")
    
    if not components:
        return " ".join(sentences)
    
    # Find top improvements and regressions
    improvements = [c for c in components if c.delta_pct < -1.0]  # More than 1% faster
    regressions = [c for c in components if c.delta_pct > 1.0]    # More than 1% slower
    
    # Sort by magnitude
    improvements.sort(key=lambda c: c.delta_pct)  # Most negative first
    regressions.sort(key=lambda c: c.delta_pct, reverse=True)  # Most positive first
    
    # Report top improvement
    if improvements:
        top = improvements[0]
        sentences.append(f"Largest improvement: {top.name} ({top.delta_pct:+.1f}%).")
    
    # Report top regression (if any)
    if regressions:
        top = regressions[0]
        if top.status == SLOStatus.WARN:
            sentences.append(f"Notable regression in {top.name} ({top.delta_pct:+.1f}%).")
        elif top.status == SLOStatus.BLOCK:
            sentences.append(f"Significant regression in {top.name} ({top.delta_pct:+.1f}%) requires attention.")
        else:
            sentences.append(f"Minor regression in {top.name} ({top.delta_pct:+.1f}%).")
    elif not improvements:
        sentences.append("No significant component-level changes detected.")
    
    return " ".join(sentences)


def generate_narrative_for_synthetic(
    baseline_avg: float,
    optimized_avg: float,
    components: List[ComponentMetrics],
) -> str:
    """
    Generate narrative for synthetic/test cases.
    
    Handles edge cases: all improve, mixed, all regress.
    """
    if baseline_avg <= 0:
        return "Unable to compute improvement: baseline is zero."
    
    improvement_pct = ((baseline_avg - optimized_avg) / baseline_avg) * 100
    return generate_narrative(improvement_pct, components)


# ---------------------------------------------------------------------------
# Component SLO Engine (Task 1)
# ---------------------------------------------------------------------------

def load_component_slo_thresholds(slo_path: Path) -> Dict[str, Dict[str, float]]:
    """
    Load per-component SLO thresholds from config file.
    
    Returns dict mapping component_name -> {warn_regression_pct, block_regression_pct}
    """
    with open(slo_path, 'r') as f:
        data = json.load(f)
    
    component_slos = data.get("component_slos", {})
    
    # Get global defaults
    slo = data.get("slo", {})
    default_warn = slo.get("warn_regression_pct", 5.0)
    default_block = slo.get("block_regression_pct", 25.0)
    
    thresholds = {}
    for name, config in component_slos.items():
        if name.startswith("_"):  # Skip comments like "_comment"
            continue
        thresholds[name] = {
            "warn_regression_pct": config.get("warn_regression_pct", default_warn),
            "block_regression_pct": config.get("block_regression_pct", default_block),
        }
    
    # Store defaults for components not explicitly configured
    thresholds["_default"] = {
        "warn_regression_pct": default_warn,
        "block_regression_pct": default_block,
    }
    
    return thresholds


def evaluate_component_slos(
    baseline_data: Dict[str, Any],
    optimized_data: Dict[str, Any],
    slo_config: Dict[str, Dict[str, float]],
) -> ComponentSLOEvaluation:
    """
    Evaluate each component against its SLO thresholds.
    
    Args:
        baseline_data: Baseline benchmark JSON data with 'components' field
        optimized_data: Optimized benchmark JSON data with 'components' field
        slo_config: Per-component SLO thresholds from load_component_slo_thresholds()
        
    Returns:
        ComponentSLOEvaluation with per-component status and aggregate metrics
    """
    baseline_components = baseline_data.get("components", {})
    optimized_components = optimized_data.get("components", {})
    
    # Get default thresholds
    defaults = slo_config.get("_default", {
        "warn_regression_pct": 5.0,
        "block_regression_pct": 25.0,
    })
    
    # Get union of component names (excluding internal keys like _default)
    all_names = set(baseline_components.keys()) | set(optimized_components.keys())
    
    results = []
    worst_offender = None
    worst_delta = float('-inf')
    
    for name in sorted(all_names):
        baseline_comp = baseline_components.get(name, {})
        optimized_comp = optimized_components.get(name, {})
        
        baseline_avg = baseline_comp.get("avg_ms", 0.0)
        optimized_avg = optimized_comp.get("avg_ms", 0.0)
        
        # Calculate delta (positive = regression, negative = improvement)
        if baseline_avg > 0:
            delta_pct = ((optimized_avg - baseline_avg) / baseline_avg) * 100
        else:
            delta_pct = 0.0
        
        # Get component-specific thresholds or defaults
        comp_thresholds = slo_config.get(name, defaults)
        warn_threshold = comp_thresholds.get("warn_regression_pct", defaults["warn_regression_pct"])
        breach_threshold = comp_thresholds.get("block_regression_pct", defaults["block_regression_pct"])
        
        # Determine status
        if delta_pct <= warn_threshold:
            status = SLOStatus.OK
        elif delta_pct <= breach_threshold:
            status = SLOStatus.WARN
        else:
            status = SLOStatus.BREACH
        
        result = ComponentSLOResult(
            name=name,
            baseline_avg_ms=baseline_avg,
            optimized_avg_ms=optimized_avg,
            delta_pct=delta_pct,
            status=status,
            warn_threshold=warn_threshold,
            breach_threshold=breach_threshold,
        )
        results.append(result)
        
        # Track worst offender
        if delta_pct > worst_delta:
            worst_delta = delta_pct
            worst_offender = name
    
    # Sort by delta_pct (most regression first)
    results.sort(key=lambda r: r.delta_pct, reverse=True)
    
    # Aggregate counts
    breached = [r for r in results if r.status == SLOStatus.BREACH]
    warned = [r for r in results if r.status == SLOStatus.WARN]
    ok = [r for r in results if r.status == SLOStatus.OK]
    
    return ComponentSLOEvaluation(
        components=results,
        any_breach=len(breached) > 0,
        worst_offender=worst_offender,
        worst_delta_pct=worst_delta if worst_delta != float('-inf') else 0.0,
        total_components=len(results),
        breached_count=len(breached),
        warned_count=len(warned),
        ok_count=len(ok),
    )


# ---------------------------------------------------------------------------
# Performance Gate Helper (Task 2)
# ---------------------------------------------------------------------------

def evaluate_perf_gate(
    baseline_path: Path,
    optimized_path: Path,
    slo_path: Path,
) -> PerfGateResult:
    """
    Evaluate performance gate for CI integration.
    
    Returns a gate status (PASS|WARN|FAIL) based on:
    - Overall performance regression
    - Component-level SLO breaches
    
    Args:
        baseline_path: Path to baseline benchmark JSON
        optimized_path: Path to optimized benchmark JSON
        slo_path: Path to SLO config JSON
        
    Returns:
        PerfGateResult with gate_status, component_breaches, and short_summary
    """
    # Load data
    if not baseline_path.exists():
        return PerfGateResult(
            gate_status=GateStatus.FAIL,
            component_breaches=[],
            component_warnings=[],
            short_summary=f"Baseline file not found: {baseline_path}",
            overall_delta_pct=0.0,
            details={"error": "baseline_not_found"},
        )
    
    if not optimized_path.exists():
        return PerfGateResult(
            gate_status=GateStatus.FAIL,
            component_breaches=[],
            component_warnings=[],
            short_summary=f"Optimized file not found: {optimized_path}",
            overall_delta_pct=0.0,
            details={"error": "optimized_not_found"},
        )
    
    with open(baseline_path, 'r') as f:
        baseline_data = json.load(f)
    
    with open(optimized_path, 'r') as f:
        optimized_data = json.load(f)
    
    # Load SLO thresholds
    slo_config = load_component_slo_thresholds(slo_path)
    
    # Load global SLO for overall threshold
    slo = SLOBaseline.from_json(slo_path)
    
    # Calculate overall delta
    baseline_avg = baseline_data.get("avg_time_per_cycle_ms", baseline_data.get("avg_cycle_time_ms", 0))
    optimized_avg = optimized_data.get("avg_time_per_cycle_ms", optimized_data.get("avg_cycle_time_ms", 0))
    
    if baseline_avg > 0:
        overall_delta_pct = ((optimized_avg - baseline_avg) / baseline_avg) * 100
    else:
        overall_delta_pct = 0.0
    
    # Evaluate component SLOs
    comp_eval = evaluate_component_slos(baseline_data, optimized_data, slo_config)
    
    # Collect breaches and warnings
    component_breaches = [r.name for r in comp_eval.components if r.status == SLOStatus.BREACH]
    component_warnings = [r.name for r in comp_eval.components if r.status == SLOStatus.WARN]
    
    # Determine gate status
    if comp_eval.any_breach:
        gate_status = GateStatus.FAIL
    elif overall_delta_pct > slo.block_regression_pct:
        gate_status = GateStatus.FAIL
    elif comp_eval.warned_count > 0 or overall_delta_pct > slo.warn_regression_pct:
        gate_status = GateStatus.WARN
    else:
        gate_status = GateStatus.PASS
    
    # Generate short summary (1-2 neutral sentences)
    short_summary = _generate_gate_summary(
        gate_status=gate_status,
        overall_delta_pct=overall_delta_pct,
        comp_eval=comp_eval,
    )
    
    return PerfGateResult(
        gate_status=gate_status,
        component_breaches=component_breaches,
        component_warnings=component_warnings,
        short_summary=short_summary,
        overall_delta_pct=overall_delta_pct,
        component_eval=comp_eval,
        details={
            "baseline_avg_ms": baseline_avg,
            "optimized_avg_ms": optimized_avg,
            "global_warn_threshold": slo.warn_regression_pct,
            "global_block_threshold": slo.block_regression_pct,
        },
    )


def _generate_gate_summary(
    gate_status: GateStatus,
    overall_delta_pct: float,
    comp_eval: ComponentSLOEvaluation,
) -> str:
    """Generate 1-2 neutral sentences summarizing the gate result."""
    sentences = []
    
    # Overall performance
    if abs(overall_delta_pct) < 1.0:
        sentences.append("Overall performance is stable.")
    elif overall_delta_pct < 0:
        sentences.append(f"Overall performance improved by {abs(overall_delta_pct):.1f}%.")
    else:
        sentences.append(f"Overall performance regressed by {overall_delta_pct:.1f}%.")
    
    # Component summary
    if comp_eval.any_breach:
        breach_names = ", ".join(r.name for r in comp_eval.components if r.status == SLOStatus.BREACH)
        sentences.append(f"SLO breach in: {breach_names}.")
    elif comp_eval.warned_count > 0:
        warn_names = ", ".join(r.name for r in comp_eval.components if r.status == SLOStatus.WARN)
        sentences.append(f"Warnings in: {warn_names}.")
    elif comp_eval.ok_count == comp_eval.total_components and comp_eval.total_components > 0:
        sentences.append("All components within SLO bounds.")
    
    return " ".join(sentences)


# ---------------------------------------------------------------------------
# Global Health Performance Signal (Task 3)
# ---------------------------------------------------------------------------

def summarize_perf_for_global_health(gate_result: PerfGateResult) -> GlobalHealthPerfSummary:
    """
    Summarize performance gate result for global health monitoring.
    
    Produces a simplified view suitable for dashboards and health checks.
    
    Args:
        gate_result: Result from evaluate_perf_gate()
        
    Returns:
        GlobalHealthPerfSummary with perf_ok, status, and regressed components
    """
    # Map gate status to health status
    status_map = {
        GateStatus.PASS: "OK",
        GateStatus.WARN: "WARN",
        GateStatus.FAIL: "BLOCK",
    }
    status = status_map[gate_result.gate_status]
    
    # Determine perf_ok
    perf_ok = gate_result.gate_status in (GateStatus.PASS, GateStatus.WARN)
    
    # Collect regressed components (those with positive delta above threshold)
    components_regressed = gate_result.component_breaches + gate_result.component_warnings
    
    # Get worst component info
    worst_component = None
    worst_delta_pct = 0.0
    if gate_result.component_eval:
        worst_component = gate_result.component_eval.worst_offender
        worst_delta_pct = gate_result.component_eval.worst_delta_pct
    
    # Generate message
    if gate_result.gate_status == GateStatus.PASS:
        message = "Performance is healthy."
    elif gate_result.gate_status == GateStatus.WARN:
        message = f"Performance degradation detected in {len(components_regressed)} component(s)."
    else:
        message = f"Performance SLO breached in {len(gate_result.component_breaches)} component(s)."
    
    return GlobalHealthPerfSummary(
        perf_ok=perf_ok,
        status=status,
        components_regressed=components_regressed,
        overall_delta_pct=gate_result.overall_delta_pct,
        worst_component=worst_component,
        worst_delta_pct=worst_delta_pct,
        message=message,
    )


# ---------------------------------------------------------------------------
# Phase IV: Performance Trend Analytics & Release Readiness Gate
# ---------------------------------------------------------------------------

def build_perf_trend_ledger(
    gate_results: List[PerfGateResult],
    run_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Build a performance trend ledger from a sequence of gate results.
    
    Tracks performance over time to identify patterns and repeated breaches.
    
    Args:
        gate_results: Sequence of PerfGateResult objects (chronological order)
        run_ids: Optional list of run identifiers (defaults to "run_0", "run_1", ...)
        
    Returns:
        Dict with:
            - schema_version: "1.0"
            - runs: List of run summaries
            - components_with_repeated_breaches: List of component names
            - release_risk_level: "LOW" | "MEDIUM" | "HIGH"
    """
    if not gate_results:
        return {
            "schema_version": "1.0",
            "runs": [],
            "components_with_repeated_breaches": [],
            "release_risk_level": "LOW",
        }
    
    # Generate run IDs if not provided
    if run_ids is None:
        run_ids = [f"run_{i}" for i in range(len(gate_results))]
    
    if len(run_ids) != len(gate_results):
        raise ValueError(f"run_ids length ({len(run_ids)}) must match gate_results length ({len(gate_results)})")
    
    # Build run summaries
    runs = []
    component_breach_history: Dict[str, List[bool]] = {}  # component_name -> [breached_in_run_0, breached_in_run_1, ...]
    
    for i, (run_id, gate_result) in enumerate(zip(run_ids, gate_results)):
        # Extract worst component info
        worst_component = None
        worst_delta_pct = 0.0
        
        if gate_result.component_eval:
            worst_component = gate_result.component_eval.worst_offender
            worst_delta_pct = gate_result.component_eval.worst_delta_pct
        
        # Map gate status to trend status
        status_map = {
            GateStatus.PASS: "OK",
            GateStatus.WARN: "WARN",
            GateStatus.FAIL: "BLOCK",
        }
        status = status_map[gate_result.gate_status]
        
        runs.append({
            "run_id": run_id,
            "status": status,
            "worst_component": worst_component,
            "worst_delta_pct": worst_delta_pct,
            "gate_status": gate_result.gate_status.value,
            "component_breaches": gate_result.component_breaches,
            "component_warnings": gate_result.component_warnings,
            "overall_delta_pct": gate_result.overall_delta_pct,
        })
        
        # Track breach history per component
        all_components = set(gate_result.component_breaches) | set(gate_result.component_warnings)
        if gate_result.component_eval:
            all_components.update(c.name for c in gate_result.component_eval.components)
        
        for comp_name in all_components:
            if comp_name not in component_breach_history:
                component_breach_history[comp_name] = []
            # Pad history if needed
            while len(component_breach_history[comp_name]) < i:
                component_breach_history[comp_name].append(False)
            
            # Mark as breached if in breaches list
            breached = comp_name in gate_result.component_breaches
            component_breach_history[comp_name].append(breached)
    
    # Identify components with repeated breaches
    # A component has "repeated breaches" if it breaches in >= 2 of the last 3 runs
    # (or >= 50% of runs if fewer than 3 runs)
    components_with_repeated_breaches = []
    
    for comp_name, breach_history in component_breach_history.items():
        if len(breach_history) < 2:
            continue  # Need at least 2 runs to have "repeated" breaches
        
        # Check last 3 runs (or all runs if fewer than 3)
        recent_runs = breach_history[-3:] if len(breach_history) >= 3 else breach_history
        breach_count = sum(recent_runs)
        
        # Repeated if breaches in >= 2 of recent runs, or >= 50% of all runs
        if breach_count >= 2 or (breach_count >= len(breach_history) / 2 and len(breach_history) >= 2):
            components_with_repeated_breaches.append(comp_name)
    
    # Determine release risk level
    total_runs = len(gate_results)
    fail_count = sum(1 for r in gate_results if r.gate_status == GateStatus.FAIL)
    warn_count = sum(1 for r in gate_results if r.gate_status == GateStatus.WARN)
    
    if len(components_with_repeated_breaches) > 0 or fail_count >= total_runs * 0.5:
        release_risk_level = "HIGH"
    elif fail_count > 0 or warn_count >= total_runs * 0.5 or len(components_with_repeated_breaches) > 0:
        release_risk_level = "MEDIUM"
    else:
        release_risk_level = "LOW"
    
    return {
        "schema_version": "1.0",
        "runs": runs,
        "components_with_repeated_breaches": sorted(components_with_repeated_breaches),
        "release_risk_level": release_risk_level,
        "total_runs": total_runs,
        "fail_count": fail_count,
        "warn_count": warn_count,
        "pass_count": total_runs - fail_count - warn_count,
    }


def evaluate_release_readiness(
    trend_ledger: Dict[str, Any],
    recent_runs_threshold: int = 3,
) -> Dict[str, Any]:
    """
    Evaluate release readiness based on performance trend ledger.
    
    Blocks release if components have repeated breaches across recent runs.
    
    Args:
        trend_ledger: Result from build_perf_trend_ledger()
        recent_runs_threshold: Number of recent runs to consider (default: 3)
        
    Returns:
        Dict with:
            - release_ok: bool
            - blocking_components: List of component names
            - status: "OK" | "WARN" | "BLOCK"
            - rationale: Human-readable explanation
    """
    if not trend_ledger.get("runs"):
        return {
            "release_ok": True,
            "blocking_components": [],
            "status": "OK",
            "rationale": "No performance data available.",
        }
    
    runs = trend_ledger["runs"]
    components_with_repeated_breaches = trend_ledger.get("components_with_repeated_breaches", [])
    
    # Check recent runs
    recent_runs = runs[-recent_runs_threshold:] if len(runs) >= recent_runs_threshold else runs
    
    # Count recent failures
    recent_fail_count = sum(1 for r in recent_runs if r["status"] == "BLOCK")
    recent_warn_count = sum(1 for r in recent_runs if r["status"] == "WARN")
    
    # Determine status
    if len(components_with_repeated_breaches) > 0:
        # BLOCK if any component has repeated breaches
        status = "BLOCK"
        release_ok = False
        blocking_components = components_with_repeated_breaches
        rationale = f"Release blocked: {len(blocking_components)} component(s) with repeated SLO breaches."
    elif recent_fail_count >= 2:
        # BLOCK if 2+ recent failures
        status = "BLOCK"
        release_ok = False
        blocking_components = []
        rationale = f"Release blocked: {recent_fail_count} recent gate failures."
    elif recent_fail_count > 0 or recent_warn_count >= len(recent_runs) * 0.5:
        # WARN if occasional failures or many warnings
        status = "WARN"
        release_ok = True  # WARN doesn't block, but flags concern
        blocking_components = []
        rationale = f"Release warning: {recent_fail_count} recent failure(s), {recent_warn_count} warning(s)."
    else:
        # OK otherwise
        status = "OK"
        release_ok = True
        blocking_components = []
        rationale = "Release ready: performance within acceptable bounds."
    
    return {
        "release_ok": release_ok,
        "blocking_components": blocking_components,
        "status": status,
        "rationale": rationale,
        "recent_runs_analyzed": len(recent_runs),
        "recent_fail_count": recent_fail_count,
        "recent_warn_count": recent_warn_count,
    }


def build_perf_director_panel(
    trend_ledger: Dict[str, Any],
    readiness: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a director console performance panel for dashboard embedding.
    
    Provides a simplified, deterministic view suitable for executive dashboards.
    
    Args:
        trend_ledger: Result from build_perf_trend_ledger()
        readiness: Result from evaluate_release_readiness()
        
    Returns:
        Dict with:
            - status_light: "GREEN" | "YELLOW" | "RED"
            - headline: Short neutral string
            - primary_concerns: List of {component, recent_delta_pct}
    """
    if not trend_ledger.get("runs"):
        return {
            "status_light": "GREEN",
            "headline": "No performance data available.",
            "primary_concerns": [],
        }
    
    # Map readiness status to status light
    status_light_map = {
        "OK": "GREEN",
        "WARN": "YELLOW",
        "BLOCK": "RED",
    }
    status_light = status_light_map.get(readiness["status"], "YELLOW")
    
    # Build headline (neutral, factual)
    runs = trend_ledger["runs"]
    total_runs = len(runs)
    blocking_count = len(readiness.get("blocking_components", []))
    
    if blocking_count > 0:
        headline = f"{blocking_count} component(s) with repeated performance breaches across {total_runs} run(s)."
    elif readiness["status"] == "BLOCK":
        recent_fails = readiness.get("recent_fail_count", 0)
        headline = f"{recent_fails} recent gate failure(s) in last {readiness.get('recent_runs_analyzed', 0)} run(s)."
    elif readiness["status"] == "WARN":
        recent_warns = readiness.get("recent_warn_count", 0)
        headline = f"{recent_warns} warning(s) in recent runs."
    else:
        headline = f"Performance stable across {total_runs} run(s)."
    
    # Build primary concerns (top components by worst delta in recent runs)
    primary_concerns = []
    
    # Get recent runs (last 3 or all if fewer)
    recent_runs = runs[-3:] if len(runs) >= 3 else runs
    
    # Collect component deltas from recent runs
    component_deltas: Dict[str, List[float]] = {}  # component -> [delta_pct values]
    
    for run in recent_runs:
        worst_comp = run.get("worst_component")
        worst_delta = run.get("worst_delta_pct", 0.0)
        
        if worst_comp:
            if worst_comp not in component_deltas:
                component_deltas[worst_comp] = []
            component_deltas[worst_comp].append(worst_delta)
    
    # Build concerns list (components with positive deltas = regressions)
    for comp_name, deltas in component_deltas.items():
        # Use worst (highest) delta from recent runs
        worst_delta = max(deltas) if deltas else 0.0
        
        # Only include if regression (positive delta) or in blocking list
        if worst_delta > 0 or comp_name in readiness.get("blocking_components", []):
            primary_concerns.append({
                "component": comp_name,
                "recent_delta_pct": round(worst_delta, 1),
            })
    
    # Sort by delta descending (worst first), limit to top 3
    primary_concerns.sort(key=lambda x: x["recent_delta_pct"], reverse=True)
    primary_concerns = primary_concerns[:3]
    
    return {
        "status_light": status_light,
        "headline": headline,
        "primary_concerns": primary_concerns,
        "total_runs": total_runs,
        "release_status": readiness["status"],
    }


# ---------------------------------------------------------------------------
# Phase V: Cross-Tile Perf Governance & Evidence Adapter
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Input Contract Validation
# ---------------------------------------------------------------------------

class PerfGovernanceInputError(Exception):
    """Raised when input contracts for perf governance are violated."""
    pass


def validate_perf_trend(perf_trend: Dict[str, Any]) -> None:
    """
    Validate perf_trend input contract.
    
    Required keys:
        - release_risk_level: "LOW" | "MEDIUM" | "HIGH"
        - runs: List of run dicts
    
    Raises:
        PerfGovernanceInputError: If contract is violated
    """
    if not isinstance(perf_trend, dict):
        raise PerfGovernanceInputError(f"perf_trend must be a dict, got {type(perf_trend)}")
    
    required_keys = ["release_risk_level", "runs"]
    missing = [k for k in required_keys if k not in perf_trend]
    if missing:
        raise PerfGovernanceInputError(f"perf_trend missing required keys: {missing}")
    
    risk_level = perf_trend.get("release_risk_level")
    if risk_level not in ("LOW", "MEDIUM", "HIGH"):
        raise PerfGovernanceInputError(
            f"perf_trend.release_risk_level must be LOW|MEDIUM|HIGH, got {risk_level}"
        )
    
    runs = perf_trend.get("runs", [])
    if not isinstance(runs, list):
        raise PerfGovernanceInputError(f"perf_trend.runs must be a list, got {type(runs)}")
    
    # Validate run structure (if runs exist)
    for i, run in enumerate(runs):
        if not isinstance(run, dict):
            raise PerfGovernanceInputError(f"perf_trend.runs[{i}] must be a dict, got {type(run)}")
        if "status" not in run:
            raise PerfGovernanceInputError(f"perf_trend.runs[{i}] missing required key: 'status'")


def validate_budget_trend(budget_trend: Dict[str, Any]) -> None:
    """
    Validate budget_trend input contract.
    
    Required keys:
        - risk_level: "LOW" | "MEDIUM" | "HIGH"
        - uplift_slices: List of slice names
    
    Raises:
        PerfGovernanceInputError: If contract is violated
    """
    if not isinstance(budget_trend, dict):
        raise PerfGovernanceInputError(f"budget_trend must be a dict, got {type(budget_trend)}")
    
    required_keys = ["risk_level", "uplift_slices"]
    missing = [k for k in required_keys if k not in budget_trend]
    if missing:
        raise PerfGovernanceInputError(f"budget_trend missing required keys: {missing}")
    
    risk_level = budget_trend.get("risk_level")
    if risk_level not in ("LOW", "MEDIUM", "HIGH"):
        raise PerfGovernanceInputError(
            f"budget_trend.risk_level must be LOW|MEDIUM|HIGH, got {risk_level}"
        )
    
    uplift_slices = budget_trend.get("uplift_slices", [])
    if not isinstance(uplift_slices, list):
        raise PerfGovernanceInputError(
            f"budget_trend.uplift_slices must be a list, got {type(uplift_slices)}"
        )


def validate_metric_conformance(metric_conformance: Dict[str, Any]) -> None:
    """
    Validate metric_conformance input contract.
    
    Required keys:
        - critical_slices: List of slice names
    
    Raises:
        PerfGovernanceInputError: If contract is violated
    """
    if not isinstance(metric_conformance, dict):
        raise PerfGovernanceInputError(
            f"metric_conformance must be a dict, got {type(metric_conformance)}"
        )
    
    required_keys = ["critical_slices"]
    missing = [k for k in required_keys if k not in metric_conformance]
    if missing:
        raise PerfGovernanceInputError(
            f"metric_conformance missing required keys: {missing}"
        )
    
    critical_slices = metric_conformance.get("critical_slices", [])
    if not isinstance(critical_slices, list):
        raise PerfGovernanceInputError(
            f"metric_conformance.critical_slices must be a list, got {type(critical_slices)}"
        )


def validate_perf_governance_inputs(
    perf_trend: Dict[str, Any],
    budget_trend: Dict[str, Any],
    metric_conformance: Dict[str, Any],
) -> None:
    """
    Validate all inputs for perf governance functions.
    
    Raises:
        PerfGovernanceInputError: If any contract is violated
    """
    validate_perf_trend(perf_trend)
    validate_budget_trend(budget_trend)
    validate_metric_conformance(metric_conformance)

def build_perf_joint_governance_view(
    perf_trend: Dict[str, Any],
    budget_trend: Dict[str, Any],
    metric_conformance: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a joint governance view combining performance, budget, and metric data.
    
    Integrates performance trends with budget risk and metric conformance to provide
    a unified view for release governance decisions.
    
    Args:
        perf_trend: Result from build_perf_trend_ledger() or similar
        budget_trend: Dict with budget risk information:
            - risk_level: "LOW" | "MEDIUM" | "HIGH"
            - slices: List of slice names with budget info
            - uplift_slices: List of slice names that are uplift slices
        metric_conformance: Dict with metric conformance information:
            - slices: List of slice names with conformance status
            - critical_slices: List of critical slice names
    
    Returns:
        Dict with:
            - perf_risk: "LOW" | "MEDIUM" | "HIGH"
            - slices_with_perf_regressions: List of slice names
            - slices_where_perf_blocks_uplift: List of slice names
            - summary_note: Human-readable summary
    """
    # Extract perf risk from trend ledger
    perf_risk_level = perf_trend.get("release_risk_level", "LOW")
    
    # Map risk levels to consistent format
    risk_map = {
        "LOW": "LOW",
        "MEDIUM": "MEDIUM",
        "HIGH": "HIGH",
    }
    perf_risk = risk_map.get(perf_risk_level, "LOW")
    
    # Extract slices with perf regressions
    # Assume perf_trend has runs with slice information, or use component breaches as proxy
    slices_with_perf_regressions = []
    
    # Check runs for slice-level regressions
    runs = perf_trend.get("runs", [])
    for run in runs:
        if run.get("status") in ("WARN", "BLOCK"):
            # Extract slice info if available
            slice_name = run.get("slice_name")
            if slice_name and slice_name not in slices_with_perf_regressions:
                slices_with_perf_regressions.append(slice_name)
    
    # If no slice-level info, use component breaches as indicator
    if not slices_with_perf_regressions:
        components_with_breaches = perf_trend.get("components_with_repeated_breaches", [])
        if components_with_breaches:
            # Mark all slices as having regressions if components breach
            # (conservative approach - can be refined with actual slice data)
            slices_with_perf_regressions = ["all_slices"]
    
    # Identify slices where perf blocks uplift
    # Uplift is blocked if:
    # 1. Perf regression on uplift slice
    # 2. Budget risk is HIGH on uplift slice
    slices_where_perf_blocks_uplift = []
    
    budget_risk = budget_trend.get("risk_level", "LOW")
    uplift_slices = budget_trend.get("uplift_slices", [])
    critical_slices = metric_conformance.get("critical_slices", [])
    
    # Check each uplift slice
    for slice_name in uplift_slices:
        # Check if this slice has perf regressions
        has_perf_regression = (
            slice_name in slices_with_perf_regressions or
            "all_slices" in slices_with_perf_regressions
        )
        
        # Check if budget risk is HIGH for this slice
        slice_budget_risk = budget_trend.get("slices", {}).get(slice_name, {}).get("risk_level", budget_risk)
        has_high_budget_risk = slice_budget_risk == "HIGH" or budget_risk == "HIGH"
        
        # Block uplift if perf regression AND high budget risk
        if has_perf_regression and has_high_budget_risk:
            slices_where_perf_blocks_uplift.append(slice_name)
    
    # Build summary note
    summary_parts = []
    
    if perf_risk == "HIGH":
        summary_parts.append(f"High performance risk ({perf_risk}).")
    elif perf_risk == "MEDIUM":
        summary_parts.append(f"Medium performance risk ({perf_risk}).")
    else:
        summary_parts.append(f"Low performance risk ({perf_risk}).")
    
    if slices_with_perf_regressions:
        if len(slices_with_perf_regressions) == 1 and slices_with_perf_regressions[0] == "all_slices":
            summary_parts.append("Performance regressions detected across all slices.")
        else:
            summary_parts.append(f"Performance regressions in {len(slices_with_perf_regressions)} slice(s).")
    
    if slices_where_perf_blocks_uplift:
        summary_parts.append(f"Uplift blocked on {len(slices_where_perf_blocks_uplift)} slice(s) due to perf + budget risk.")
    
    if not summary_parts:
        summary_parts.append("Performance within acceptable bounds.")
    
    summary_note = " ".join(summary_parts)
    
    return {
        "perf_risk": perf_risk,
        "slices_with_perf_regressions": slices_with_perf_regressions,
        "slices_where_perf_blocks_uplift": slices_where_perf_blocks_uplift,
        "summary_note": summary_note,
        "budget_risk": budget_risk,
        "uplift_slices": uplift_slices,
        "critical_slices": critical_slices,
    }


def summarize_perf_for_global_release(perf_joint_view: Dict[str, Any]) -> Dict[str, Any]:
    """
    Summarize performance joint view for global release decision.
    
    Maps the joint governance view into a release decision format suitable for
    integration with release bundles and evidence packs.
    
    Args:
        perf_joint_view: Result from build_perf_joint_governance_view()
        
    Returns:
        Dict with:
            - release_ok: bool
            - status: "OK" | "WARN" | "BLOCK"
            - blocking_components: List of component/slice names
            - headline: Short neutral summary
    """
    perf_risk = perf_joint_view.get("perf_risk", "LOW")
    slices_with_regressions = perf_joint_view.get("slices_with_perf_regressions", [])
    slices_blocking_uplift = perf_joint_view.get("slices_where_perf_blocks_uplift", [])
    budget_risk = perf_joint_view.get("budget_risk", "LOW")
    critical_slices = perf_joint_view.get("critical_slices", [])
    
    # Determine release status based on joint view
    # BLOCK if:
    # 1. Perf regressions + HIGH budget risk on uplift slices
    # 2. Perf risk is HIGH and affects critical slices
    if slices_blocking_uplift:
        status = "BLOCK"
        release_ok = False
        blocking_components = slices_blocking_uplift
        headline = f"Release blocked: Performance regressions + high budget risk on {len(slices_blocking_uplift)} uplift slice(s)."
    elif perf_risk == "HIGH" and any(slice_name in slices_with_regressions for slice_name in critical_slices):
        status = "BLOCK"
        release_ok = False
        blocking_components = [s for s in slices_with_regressions if s in critical_slices]
        headline = f"Release blocked: High performance risk on {len(blocking_components)} critical slice(s)."
    elif perf_risk == "HIGH":
        status = "BLOCK"
        release_ok = False
        blocking_components = slices_with_regressions if slices_with_regressions else ["performance"]
        headline = f"Release blocked: High performance risk detected."
    elif slices_with_regressions and budget_risk == "LOW":
        # Perf regressions on critical slices with consistent budget OK → WARN (not BLOCK)
        status = "WARN"
        release_ok = True  # WARN doesn't block release
        blocking_components = []
        headline = f"Release warning: Performance regressions on {len(slices_with_regressions)} slice(s) with stable budget."
    elif perf_risk == "MEDIUM":
        status = "WARN"
        release_ok = True
        blocking_components = []
        headline = f"Release warning: Medium performance risk detected."
    else:
        # Perf improvement everywhere → OK
        status = "OK"
        release_ok = True
        blocking_components = []
        headline = "Release ready: Performance within acceptable bounds."
    
    return {
        "release_ok": release_ok,
        "status": status,
        "blocking_components": blocking_components,
        "headline": headline,
        "perf_risk": perf_risk,
        "budget_risk": budget_risk,
        "slices_with_regressions": slices_with_regressions,
        "slices_blocking_uplift": slices_blocking_uplift,
    }


# ---------------------------------------------------------------------------
# Self-Consistency Check
# ---------------------------------------------------------------------------

def run_cycles_and_fingerprint(
    cycles: int,
    seed: int,
    slice_name: str,
    mode: str = "baseline",
) -> List[CycleFingerprint]:
    """
    Run CycleRunner and collect fingerprints for each cycle.
    """
    from experiments.run_fo_cycles import CycleRunner
    
    output_path = Path("results") / "perf" / f"equiv_test_{mode}_{seed}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    runner = CycleRunner(mode, output_path, slice_name=slice_name, system="pl")
    
    # Override the MDAP epoch seed for reproducibility
    import experiments.run_fo_cycles as fo_module
    original_seed = fo_module.MDAP_EPOCH_SEED
    fo_module.MDAP_EPOCH_SEED = seed
    
    fingerprints = []
    
    try:
        for i in range(cycles):
            result = runner.run_cycle(i)
            
            fp = CycleFingerprint(
                cycle_index=i,
                h_t=result["roots"]["h_t"],
                r_t=result["roots"]["r_t"],
                u_t=result["roots"]["u_t"],
                n_verified=result["derivation"]["verified"],
                n_abstained=result["derivation"]["abstained"],
                n_candidates=result["derivation"]["candidates"],
                status=result["status"],
                candidate_hash=result["derivation"]["candidate_hash"],
            )
            fingerprints.append(fp)
    finally:
        fo_module.MDAP_EPOCH_SEED = original_seed
    
    return fingerprints


def compare_fingerprints(
    baseline: List[CycleFingerprint],
    optimized: List[CycleFingerprint],
) -> List[Dict[str, Any]]:
    """
    Compare two sets of fingerprints and return mismatches.
    """
    mismatches = []
    
    for b, o in zip(baseline, optimized):
        diffs = {}
        
        if b.h_t != o.h_t:
            diffs["h_t"] = {"baseline": b.h_t, "optimized": o.h_t}
        if b.r_t != o.r_t:
            diffs["r_t"] = {"baseline": b.r_t, "optimized": o.r_t}
        if b.u_t != o.u_t:
            diffs["u_t"] = {"baseline": b.u_t, "optimized": o.u_t}
        if b.n_verified != o.n_verified:
            diffs["n_verified"] = {"baseline": b.n_verified, "optimized": o.n_verified}
        if b.n_abstained != o.n_abstained:
            diffs["n_abstained"] = {"baseline": b.n_abstained, "optimized": o.n_abstained}
        if b.n_candidates != o.n_candidates:
            diffs["n_candidates"] = {"baseline": b.n_candidates, "optimized": o.n_candidates}
        if b.status != o.status:
            diffs["status"] = {"baseline": b.status, "optimized": o.status}
        if b.candidate_hash != o.candidate_hash:
            diffs["candidate_hash"] = {"baseline": b.candidate_hash, "optimized": o.candidate_hash}
        
        if diffs:
            mismatches.append({
                "cycle": b.cycle_index,
                "diffs": diffs,
            })
    
    return mismatches


def verify_self_equivalence(
    cycles: int = 10,
    seed: int = 42,
    slice_name: str = "slice_medium",
    mode: str = "baseline",
) -> EquivalenceReport:
    """
    Run cycles twice and verify outputs are identical.
    """
    print(f"Running {cycles} cycles with seed={seed}, slice={slice_name}, mode={mode}")
    print("Run 1...")
    run1 = run_cycles_and_fingerprint(cycles, seed, slice_name, mode)
    
    print("Run 2...")
    run2 = run_cycles_and_fingerprint(cycles, seed, slice_name, mode)
    
    print("Comparing fingerprints...")
    mismatches = compare_fingerprints(run1, run2)
    
    report = EquivalenceReport(
        matched=(len(mismatches) == 0),
        cycles_compared=cycles,
        mismatches=mismatches,
        seed=seed,
        slice_name=slice_name,
        mode=mode,
    )
    
    return report


# ---------------------------------------------------------------------------
# Performance Ratchet Check
# ---------------------------------------------------------------------------

def check_perf_ratchet(
    baseline_path: Path,
    optimized_path: Path,
    min_improvement: float = 0.1,
) -> PerfRatchetResult:
    """
    Check that optimized performance meets minimum improvement threshold.
    """
    if not baseline_path.exists():
        return PerfRatchetResult(
            passed=False,
            baseline_avg_ms=0.0,
            optimized_avg_ms=0.0,
            improvement_pct=0.0,
            required_improvement_pct=min_improvement * 100,
            message=f"Baseline file not found: {baseline_path}",
        )
    
    if not optimized_path.exists():
        return PerfRatchetResult(
            passed=False,
            baseline_avg_ms=0.0,
            optimized_avg_ms=0.0,
            improvement_pct=0.0,
            required_improvement_pct=min_improvement * 100,
            message=f"Optimized file not found: {optimized_path}",
        )
    
    with open(baseline_path, 'r') as f:
        baseline = json.load(f)
    
    with open(optimized_path, 'r') as f:
        optimized = json.load(f)
    
    baseline_avg = baseline.get("avg_time_per_cycle_ms", baseline.get("avg_cycle_time_ms", 0))
    optimized_avg = optimized.get("avg_time_per_cycle_ms", optimized.get("avg_cycle_time_ms", 0))
    
    if baseline_avg == 0:
        return PerfRatchetResult(
            passed=False,
            baseline_avg_ms=baseline_avg,
            optimized_avg_ms=optimized_avg,
            improvement_pct=0.0,
            required_improvement_pct=min_improvement * 100,
            message="Baseline avg_time_per_cycle_ms is zero",
        )
    
    improvement = (baseline_avg - optimized_avg) / baseline_avg
    improvement_pct = improvement * 100
    
    threshold = baseline_avg * (1 - min_improvement)
    passed = optimized_avg <= threshold
    
    if passed:
        message = f"PASS: {improvement_pct:.1f}% improvement meets {min_improvement*100:.1f}% threshold"
    else:
        message = f"FAIL: {improvement_pct:.1f}% improvement does not meet {min_improvement*100:.1f}% threshold"
    
    return PerfRatchetResult(
        passed=passed,
        baseline_avg_ms=baseline_avg,
        optimized_avg_ms=optimized_avg,
        improvement_pct=improvement_pct,
        required_improvement_pct=min_improvement * 100,
        message=message,
    )


# ---------------------------------------------------------------------------
# SLO Enforcement
# ---------------------------------------------------------------------------

def check_slo(
    baseline_path: Path,
    optimized_path: Path,
    slo_path: Path,
) -> SLOResult:
    """
    Check performance against SLO baseline registry.
    
    Computes regression against historical reference and determines status:
    - OK: Within acceptable bounds
    - WARN: Slightly above threshold, needs attention
    - BLOCK: Grossly above threshold, CI should fail
    """
    # Load SLO baseline (validates config)
    slo = SLOBaseline.from_json(slo_path)
    
    # Load benchmark results
    if not baseline_path.exists():
        return SLOResult(
            status=SLOStatus.BLOCK,
            current_avg_ms=0.0,
            reference_avg_ms=slo.reference_avg_ms,
            baseline_avg_ms=0.0,
            optimized_avg_ms=0.0,
            regression_pct=0.0,
            improvement_pct=0.0,
            message=f"Baseline file not found: {baseline_path}",
        )
    
    if not optimized_path.exists():
        return SLOResult(
            status=SLOStatus.BLOCK,
            current_avg_ms=0.0,
            reference_avg_ms=slo.reference_avg_ms,
            baseline_avg_ms=0.0,
            optimized_avg_ms=0.0,
            regression_pct=0.0,
            improvement_pct=0.0,
            message=f"Optimized file not found: {optimized_path}",
        )
    
    with open(baseline_path, 'r') as f:
        baseline_data = json.load(f)
    
    with open(optimized_path, 'r') as f:
        optimized_data = json.load(f)
    
    baseline_avg = baseline_data.get("avg_time_per_cycle_ms", baseline_data.get("avg_cycle_time_ms", 0))
    optimized_avg = optimized_data.get("avg_time_per_cycle_ms", optimized_data.get("avg_cycle_time_ms", 0))
    cycles = optimized_data.get("cycles", optimized_data.get("measured_cycles", 0))
    
    # Validate minimum cycles
    if cycles < slo.min_cycles_for_validity:
        return SLOResult(
            status=SLOStatus.WARN,
            current_avg_ms=optimized_avg,
            reference_avg_ms=slo.reference_avg_ms,
            baseline_avg_ms=baseline_avg,
            optimized_avg_ms=optimized_avg,
            regression_pct=0.0,
            improvement_pct=0.0,
            message=f"Insufficient cycles: {cycles} < {slo.min_cycles_for_validity} required",
            details={"cycles": cycles, "min_required": slo.min_cycles_for_validity},
        )
    
    # Use optimized time as current measurement
    current_avg = optimized_avg
    
    # Calculate regression from historical reference
    if slo.reference_avg_ms > 0:
        regression_pct = ((current_avg - slo.reference_avg_ms) / slo.reference_avg_ms) * 100
    else:
        regression_pct = 0.0
    
    # Calculate improvement from this run's baseline
    if baseline_avg > 0:
        improvement_pct = ((baseline_avg - optimized_avg) / baseline_avg) * 100
    else:
        improvement_pct = 0.0
    
    # Extract component-level metrics
    components = extract_components(baseline_data, optimized_data, slo)
    
    # Generate narrative
    narrative = generate_narrative(improvement_pct, components)
    
    # Apply jitter tolerance
    effective_regression = max(0, regression_pct - slo.jitter_allowance_pct)
    
    # Determine status
    if effective_regression <= slo.warn_regression_pct:
        status = SLOStatus.OK
        message = f"Performance within SLO: {regression_pct:+.1f}% vs reference"
    elif effective_regression <= slo.max_regression_pct:
        status = SLOStatus.WARN
        message = f"Performance degraded: {regression_pct:+.1f}% vs reference (threshold: {slo.max_regression_pct}%)"
    elif effective_regression <= slo.block_regression_pct:
        status = SLOStatus.WARN
        message = f"Significant regression: {regression_pct:+.1f}% vs reference (approaching block threshold)"
    else:
        status = SLOStatus.BLOCK
        message = f"BLOCKED: {regression_pct:+.1f}% regression exceeds {slo.block_regression_pct}% threshold"
    
    return SLOResult(
        status=status,
        current_avg_ms=current_avg,
        reference_avg_ms=slo.reference_avg_ms,
        baseline_avg_ms=baseline_avg,
        optimized_avg_ms=optimized_avg,
        regression_pct=regression_pct,
        improvement_pct=improvement_pct,
        message=message,
        components=components,
        narrative=narrative,
        details={
            "effective_regression_pct": effective_regression,
            "jitter_allowance_pct": slo.jitter_allowance_pct,
            "cycles": cycles,
        },
    )


# ---------------------------------------------------------------------------
# Markdown Summary Generation
# ---------------------------------------------------------------------------

def generate_progress_bar(value: float, max_value: float, width: int = 20) -> str:
    """Generate a text-based progress bar."""
    if max_value <= 0:
        return "░" * width
    ratio = min(1.0, max(0.0, value / max_value))
    filled = int(ratio * width)
    return "█" * filled + "░" * (width - filled)


def generate_component_table(components: List[ComponentMetrics]) -> str:
    """Generate Markdown table for component-level breakdown."""
    if not components:
        return ""
    
    lines = [
        "### Component-Level Breakdown",
        "",
        "| Component | Baseline | Optimized | Δ% | Status |",
        "|-----------|----------|-----------|-----|--------|",
    ]
    
    for c in components:
        delta_str = f"{c.delta_pct:+.1f}%"
        lines.append(
            f"| {c.name} | `{c.baseline_avg_ms:.1f}ms` | `{c.optimized_avg_ms:.1f}ms` | `{delta_str}` | {c.status.emoji} |"
        )
    
    lines.append("")
    return "\n".join(lines)


def generate_markdown_summary(
    slo_result: SLOResult,
    baseline_path: Optional[Path] = None,
    optimized_path: Optional[Path] = None,
) -> str:
    """
    Generate a human-friendly Markdown summary for PR descriptions.
    """
    status = slo_result.status
    
    # Header with status
    lines = [
        "## 🏎️ Performance Ratchet Report",
        "",
        f"**Status:** {status.emoji} **{status.value}**",
        "",
    ]
    
    # Narrative summary (new!)
    if slo_result.narrative:
        lines.extend([
            "### Summary",
            "",
            f"> {slo_result.narrative}",
            "",
        ])
    
    # Timing summary table
    lines.extend([
        "### Timing Summary",
        "",
        "| Metric | Value | vs Reference |",
        "|--------|-------|--------------|",
        f"| Reference (historical) | `{slo_result.reference_avg_ms:.1f}ms` | — |",
        f"| Baseline (this run) | `{slo_result.baseline_avg_ms:.1f}ms` | — |",
        f"| Optimized (this run) | `{slo_result.optimized_avg_ms:.1f}ms` | `{slo_result.regression_pct:+.1f}%` |",
        "",
    ])
    
    # Visual comparison
    max_time = max(slo_result.reference_avg_ms, slo_result.baseline_avg_ms, slo_result.optimized_avg_ms, 1)
    
    lines.extend([
        "### Visual Comparison",
        "",
        "```",
        f"Reference:  {generate_progress_bar(slo_result.reference_avg_ms, max_time)} {slo_result.reference_avg_ms:.1f}ms",
        f"Baseline:   {generate_progress_bar(slo_result.baseline_avg_ms, max_time)} {slo_result.baseline_avg_ms:.1f}ms",
        f"Optimized:  {generate_progress_bar(slo_result.optimized_avg_ms, max_time)} {slo_result.optimized_avg_ms:.1f}ms",
        "```",
        "",
    ])
    
    # Component table (new!)
    component_table = generate_component_table(slo_result.components)
    if component_table:
        lines.append(component_table)
    
    # Improvement metrics
    if slo_result.improvement_pct != 0:
        if slo_result.improvement_pct > 0:
            improvement_indicator = "⬇️ faster"
        else:
            improvement_indicator = "⬆️ slower"
        
        lines.extend([
            "### Optimization Impact",
            "",
            f"- **Improvement over baseline:** `{slo_result.improvement_pct:+.1f}%` {improvement_indicator}",
            f"- **Regression vs reference:** `{slo_result.regression_pct:+.1f}%`",
            "",
        ])
    
    # SLO details
    if slo_result.details:
        lines.extend([
            "### SLO Details",
            "",
            f"- Effective regression (after jitter tolerance): `{slo_result.details.get('effective_regression_pct', 0):.1f}%`",
            f"- Jitter allowance: `{slo_result.details.get('jitter_allowance_pct', 0):.1f}%`",
            f"- Cycles measured: `{slo_result.details.get('cycles', 0)}`",
            "",
        ])
    
    # Status message
    lines.extend([
        "### Result",
        "",
        f"> {slo_result.message}",
        "",
    ])
    
    # CI guidance
    if status == SLOStatus.OK:
        lines.extend([
            "✅ **CI Status:** Pass — performance is within acceptable bounds.",
            "",
        ])
    elif status == SLOStatus.WARN:
        lines.extend([
            "⚠️ **CI Status:** Pass (with warning) — performance needs attention.",
            "",
            "Consider investigating if this regression persists across multiple runs.",
            "",
        ])
    else:  # BLOCK
        lines.extend([
            "❌ **CI Status:** Fail — performance regression exceeds acceptable threshold.",
            "",
            "**Action Required:** Investigate and fix the performance regression before merging.",
            "",
        ])
    
    # Footer
    lines.extend([
        "---",
        f"*Generated at {datetime.utcnow().isoformat()}Z by `verify_perf_equivalence.py`*",
    ])
    
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Performance Equivalence and Ratchet Verification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Self-consistency check:
  %(prog)s --cycles=10 --seed=42

  # Perf ratchet check with SLO:
  %(prog)s --baseline results/perf/baseline.json --optimized results/perf/optimized.json --slo config/perf_baseline.json

  # Generate summary for PR:
  %(prog)s --baseline results/perf/baseline.json --optimized results/perf/optimized.json --slo config/perf_baseline.json --output-summary results/perf/summary.md
        """
    )
    
    # Self-consistency mode arguments
    parser.add_argument(
        "--cycles",
        type=int,
        default=10,
        help="Number of cycles for self-consistency check (default: 10)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--slice",
        type=str,
        default="slice_medium",
        help="Curriculum slice name (default: slice_medium)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["baseline", "rfl"],
        default="baseline",
        help="Execution mode (default: baseline)"
    )
    
    # Perf ratchet mode arguments
    parser.add_argument(
        "--baseline",
        type=str,
        help="Path to baseline benchmark JSON file"
    )
    parser.add_argument(
        "--optimized",
        type=str,
        help="Path to optimized benchmark JSON file"
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.1,
        help="Minimum required improvement ratio (0.1 = 10%%, default: 0.1)"
    )
    parser.add_argument(
        "--slo",
        type=str,
        help="Path to SLO baseline JSON file (config/perf_baseline.json)"
    )
    parser.add_argument(
        "--output-summary",
        type=str,
        help="Path to write Markdown summary (for PR descriptions)"
    )
    parser.add_argument(
        "--behavior-only",
        action="store_true",
        help="Skip perf check, only verify behavioral equivalence"
    )
    parser.add_argument(
        "--slo-gate",
        action="store_true",
        help="Run SLO gate evaluation (PASS|WARN|FAIL) with component-level analysis"
    )
    parser.add_argument(
        "--json-output",
        type=str,
        help="Path to write JSON gate result (for programmatic consumption)"
    )
    
    args = parser.parse_args()
    
    results_dir = Path("results") / "perf"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine mode: SLO gate, SLO check, perf ratchet, or self-consistency
    if args.baseline and args.optimized:
        baseline_path = Path(args.baseline)
        optimized_path = Path(args.optimized)
        
        # Handle --slo-gate mode (Task 2)
        if args.slo_gate:
            if not args.slo:
                print("❌ Error: --slo-gate requires --slo config path", file=sys.stderr)
                return 1
            
            slo_path = Path(args.slo)
            
            print("=" * 70)
            print("PERFORMANCE SLO GATE EVALUATION")
            print("=" * 70)
            print(f"Baseline: {baseline_path}")
            print(f"Optimized: {optimized_path}")
            print(f"SLO Config: {slo_path}")
            print("=" * 70)
            
            try:
                gate_result = evaluate_perf_gate(baseline_path, optimized_path, slo_path)
            except SLOConfigError as e:
                print(f"\n❌ SLO Configuration Error:\n{e}", file=sys.stderr)
                return 1
            
            # Generate global health summary
            health_summary = summarize_perf_for_global_health(gate_result)
            
            print()
            print(f"Gate Status: {gate_result.gate_status.emoji} {gate_result.gate_status.value}")
            print(f"Overall Δ:   {gate_result.overall_delta_pct:+.1f}%")
            print()
            print("Summary:")
            print(f"  {gate_result.short_summary}")
            print()
            
            # Print component SLO results
            if gate_result.component_eval:
                comp_eval = gate_result.component_eval
                print(f"Components: {comp_eval.total_components} total")
                print(f"  ✅ OK:     {comp_eval.ok_count}")
                print(f"  ⚠️ WARN:   {comp_eval.warned_count}")
                print(f"  🚨 BREACH: {comp_eval.breached_count}")
                print()
                
                if comp_eval.worst_offender:
                    print(f"Worst Offender: {comp_eval.worst_offender} ({comp_eval.worst_delta_pct:+.1f}%)")
                    print()
                
                print("Component Details:")
                for c in comp_eval.components:
                    print(f"  {c.status.emoji} {c.name}: {c.baseline_avg_ms:.1f}ms → {c.optimized_avg_ms:.1f}ms ({c.delta_pct:+.1f}%)")
                    print(f"      Thresholds: WARN>{c.warn_threshold}%, BREACH>{c.breach_threshold}%")
                print()
            
            # Print breaches and warnings
            if gate_result.component_breaches:
                print(f"🚨 BREACHED: {', '.join(gate_result.component_breaches)}")
            if gate_result.component_warnings:
                print(f"⚠️ WARNED: {', '.join(gate_result.component_warnings)}")
            
            print()
            print("=" * 70)
            print(f"{gate_result.gate_status.emoji} GATE: {gate_result.gate_status.value}")
            print("=" * 70)
            
            # Print global health summary
            print()
            print("Global Health Signal:")
            print(f"  perf_ok: {health_summary.perf_ok}")
            print(f"  status:  {health_summary.status}")
            print(f"  message: {health_summary.message}")
            
            # Save JSON output
            if args.json_output:
                json_path = Path(args.json_output)
                json_path.parent.mkdir(parents=True, exist_ok=True)
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "gate": gate_result.to_dict(),
                        "health": health_summary.to_dict(),
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                    }, f, indent=2)
                print(f"\nJSON result saved to: {json_path}")
            
            # Save to default location
            gate_json_path = results_dir / "slo_gate_result.json"
            with open(gate_json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "gate": gate_result.to_dict(),
                    "health": health_summary.to_dict(),
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                }, f, indent=2)
            print(f"Gate result saved to: {gate_json_path}")
            
            return gate_result.gate_status.exit_code
        
        if args.slo:
            # SLO enforcement mode
            slo_path = Path(args.slo)
            
            print("=" * 70)
            print("PERFORMANCE SLO CHECK")
            print("=" * 70)
            print(f"Baseline: {baseline_path}")
            print(f"Optimized: {optimized_path}")
            print(f"SLO Config: {slo_path}")
            print("=" * 70)
            
            try:
                result = check_slo(baseline_path, optimized_path, slo_path)
            except SLOConfigError as e:
                print(f"\n❌ SLO Configuration Error:\n{e}", file=sys.stderr)
                return 1
            
            print()
            print(f"Reference avg:  {result.reference_avg_ms:.2f}ms (historical)")
            print(f"Baseline avg:   {result.baseline_avg_ms:.2f}ms (this run)")
            print(f"Optimized avg:  {result.optimized_avg_ms:.2f}ms (this run)")
            print(f"Regression:     {result.regression_pct:+.1f}% vs reference")
            print(f"Improvement:    {result.improvement_pct:+.1f}% vs baseline")
            print()
            
            # Print narrative
            if result.narrative:
                print("Narrative:")
                print(f"  {result.narrative}")
                print()
            
            # Print component breakdown
            if result.components:
                print("Component Breakdown:")
                for c in result.components:
                    print(f"  {c.name}: {c.baseline_avg_ms:.1f}ms → {c.optimized_avg_ms:.1f}ms ({c.delta_pct:+.1f}%) {c.status.emoji}")
                print()
            
            print("=" * 70)
            print(f"{result.status.emoji} {result.message}")
            print("=" * 70)
            
            # Generate and save Markdown summary
            summary = generate_markdown_summary(result, baseline_path, optimized_path)
            
            if args.output_summary:
                summary_path = Path(args.output_summary)
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                with open(summary_path, 'w', encoding='utf-8') as f:
                    f.write(summary)
                print(f"\nMarkdown summary saved to: {summary_path}")
            
            # Also print summary to stdout for easy copy-paste
            print("\n" + "=" * 70)
            print("MARKDOWN SUMMARY (for PR description)")
            print("=" * 70)
            print(summary)
            
            # Save result JSON
            report_path = results_dir / "slo_result.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                # Convert components to dicts
                result_dict = asdict(result)
                result_dict["status"] = result.status.value
                result_dict["ci_pass"] = result.ci_pass
                result_dict["components"] = [
                    {**asdict(c), "status": c.status.value}
                    for c in result.components
                ]
                json.dump(result_dict, f, indent=2, default=str)
            print(f"\nResult saved to: {report_path}")
            
            return 0 if result.ci_pass else 1
        
        else:
            # Simple perf ratchet mode (no SLO)
            print("=" * 70)
            print("PERFORMANCE RATCHET CHECK")
            print("=" * 70)
            print(f"Baseline: {baseline_path}")
            print(f"Optimized: {optimized_path}")
            print(f"Min improvement: {args.min_improvement * 100:.1f}%")
            print("=" * 70)
            
            result = check_perf_ratchet(
                baseline_path,
                optimized_path,
                args.min_improvement,
            )
            
            print()
            print(f"Baseline avg:  {result.baseline_avg_ms:.2f}ms")
            print(f"Optimized avg: {result.optimized_avg_ms:.2f}ms")
            print(f"Improvement:   {result.improvement_pct:+.1f}%")
            print(f"Required:      {result.required_improvement_pct:.1f}%")
            print()
            print("=" * 70)
            
            if result.passed:
                print(f"✓ {result.message}")
            else:
                print(f"✗ {result.message}")
            
            print("=" * 70)
            
            # Save result
            report_path = results_dir / "ratchet_result.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(result), f, indent=2)
            print(f"\nResult saved to: {report_path}")
            
            return 0 if result.passed else 1
    
    else:
        # Self-consistency mode
        report = verify_self_equivalence(
            cycles=args.cycles,
            seed=args.seed,
            slice_name=args.slice,
            mode=args.mode,
        )
        
        print()
        print("=" * 60)
        if report.matched:
            print("✓ EQUIVALENCE VERIFIED: All cycles produced identical outputs")
        else:
            print("✗ EQUIVALENCE FAILED: Mismatches detected!")
            for mismatch in report.mismatches:
                print(f"  Cycle {mismatch['cycle']}: {mismatch['diffs']}")
        print("=" * 60)
        
        # Save report
        report_path = results_dir / "equivalence_report.json"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(report), f, indent=2)
        
        print(f"Report saved to: {report_path}")
        
        return 0 if report.matched else 1


if __name__ == "__main__":
    sys.exit(main())
