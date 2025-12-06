"""
U2 Safety Envelope - Phase III Safety Contract

Provides safety status evaluation and performance monitoring for U2 experiments.
The safety envelope is an in-memory contract that assesses experiment safety
based on configuration, performance metrics, and evaluation lint results.
"""

from typing import Any, Dict, List, Literal, Optional
from dataclasses import dataclass, field

from .runner import U2Config
from .u2_safe_eval import SafeEvalLintResult


# Safety status values
SafetyStatus = Literal["OK", "WARN", "BLOCK"]

# Schema version for safety envelope
SAFETY_ENVELOPE_VERSION = "1.0.0"

# Performance thresholds (configurable)
DEFAULT_PERF_THRESHOLDS = {
    "max_cycle_duration_ms": 5000.0,  # 5 seconds per cycle
    "max_avg_cycle_duration_ms": 2000.0,  # 2 second average
    "max_eval_lint_issues": 10,  # Maximum number of lint issues to tolerate
}


@dataclass
class U2SafetyEnvelope:
    """
    Safety envelope for U2 experiments.
    
    Provides a typed contract for safety assessment based on:
    - Configuration validity
    - Performance metrics
    - Evaluation lint results
    """
    schema_version: str
    config: Dict[str, Any]  # Safe subset of U2Config (no secrets)
    perf_ok: bool
    eval_lint_issues: int
    safety_status: SafetyStatus
    
    # Optional detailed information
    perf_stats: Dict[str, Any] = field(default_factory=dict)
    top_eval_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "schema_version": self.schema_version,
            "config": self.config,
            "perf_ok": self.perf_ok,
            "eval_lint_issues": self.eval_lint_issues,
            "safety_status": self.safety_status,
            "perf_stats": self.perf_stats,
            "top_eval_issues": self.top_eval_issues,
            "warnings": self.warnings,
        }


def build_u2_safety_envelope(
    run_config: U2Config,
    perf_stats: Dict[str, Any],
    eval_lint_results: List[SafeEvalLintResult],
    perf_thresholds: Optional[Dict[str, Any]] = None,
) -> U2SafetyEnvelope:
    """
    Build a U2 safety envelope from experiment data.
    
    This function assesses the safety of a U2 experiment based on:
    1. Configuration validity (no dangerous settings)
    2. Performance metrics (within acceptable bounds)
    3. Evaluation lint results (no critical safety issues)
    
    Args:
        run_config: U2Config for the experiment (must be valid)
        perf_stats: Performance statistics dictionary with keys:
            - cycle_durations_ms: List of cycle durations
            - avg_cycle_duration_ms: Average cycle duration
            - max_cycle_duration_ms: Maximum cycle duration
        eval_lint_results: List of SafeEvalLintResult from safe_eval linting
        perf_thresholds: Optional custom performance thresholds
        
    Returns:
        U2SafetyEnvelope with safety assessment
    """
    if perf_thresholds is None:
        perf_thresholds = DEFAULT_PERF_THRESHOLDS
    
    warnings: List[str] = []
    
    # 1. Extract safe config subset (no secrets)
    safe_config = run_config.to_safe_dict()
    
    # 2. Evaluate performance metrics
    perf_ok = True
    
    max_cycle_duration = perf_stats.get("max_cycle_duration_ms", 0.0)
    avg_cycle_duration = perf_stats.get("avg_cycle_duration_ms", 0.0)
    
    if max_cycle_duration > perf_thresholds["max_cycle_duration_ms"]:
        perf_ok = False
        warnings.append(
            f"Max cycle duration ({max_cycle_duration:.1f}ms) exceeds threshold "
            f"({perf_thresholds['max_cycle_duration_ms']:.1f}ms)"
        )
    
    if avg_cycle_duration > perf_thresholds["max_avg_cycle_duration_ms"]:
        perf_ok = False
        warnings.append(
            f"Avg cycle duration ({avg_cycle_duration:.1f}ms) exceeds threshold "
            f"({perf_thresholds['max_avg_cycle_duration_ms']:.1f}ms)"
        )
    
    # 3. Evaluate lint results
    unsafe_count = sum(1 for result in eval_lint_results if not result.is_safe)
    total_issues = sum(len(result.issues) for result in eval_lint_results)
    
    # Collect top issues (up to 10)
    top_issues: List[str] = []
    for result in eval_lint_results:
        if not result.is_safe:
            for issue in result.issues:
                top_issues.append(f"{result.expression}: {issue}")
                if len(top_issues) >= 10:
                    break
        if len(top_issues) >= 10:
            break
    
    # 4. Determine safety status
    safety_status: SafetyStatus
    
    if unsafe_count > 0:
        # Any unsafe expressions are a BLOCK
        safety_status = "BLOCK"
        warnings.append(f"Found {unsafe_count} unsafe expressions")
    elif total_issues > perf_thresholds["max_eval_lint_issues"]:
        # Too many lint issues are a WARN
        safety_status = "WARN"
        warnings.append(f"Total lint issues ({total_issues}) exceeds threshold")
    elif not perf_ok:
        # Performance issues are a WARN
        safety_status = "WARN"
    else:
        # Everything looks good
        safety_status = "OK"
    
    # 5. Build envelope
    envelope = U2SafetyEnvelope(
        schema_version=SAFETY_ENVELOPE_VERSION,
        config=safe_config,
        perf_ok=perf_ok,
        eval_lint_issues=total_issues,
        safety_status=safety_status,
        perf_stats={
            "max_cycle_duration_ms": max_cycle_duration,
            "avg_cycle_duration_ms": avg_cycle_duration,
            "cycle_count": len(perf_stats.get("cycle_durations_ms", [])),
        },
        top_eval_issues=top_issues,
        warnings=warnings,
    )
    
    return envelope


def evaluate_safety_status(envelope: U2SafetyEnvelope) -> bool:
    """
    Evaluate whether an experiment passes safety checks.
    
    Args:
        envelope: U2SafetyEnvelope to evaluate
        
    Returns:
        True if safety status is OK or WARN, False if BLOCK
    """
    return envelope.safety_status != "BLOCK"
