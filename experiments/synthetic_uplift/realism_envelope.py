#!/usr/bin/env python3
"""
==============================================================================
PHASE II — SYNTHETIC TEST DATA ONLY
==============================================================================

Synthetic Realism Envelope Check
---------------------------------

This module validates that synthetic scenario parameters stay within
"realistic" bounds - not to claim empirical validity, but to ensure
synthetic data is plausible enough for stress-testing purposes.

Envelope Checks:
    - Variance bounds: per_cycle_sigma, per_item_sigma within limits
    - Correlation envelope: rho within [0, 0.9]
    - Drift amplitude envelope: amplitude within limits per mode
    - Rare event frequency bounds: trigger probabilities reasonable

Exit Codes:
    - 0: All envelopes satisfied
    - 1: At least one parameter out of envelope

Must NOT:
    - Produce claims about real-world behavior
    - Suggest synthetic data is empirically valid
    - Mix synthetic and real data

==============================================================================
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from experiments.synthetic_uplift.noise_models import SAFETY_LABEL


# ==============================================================================
# ENVELOPE BOUNDS
# ==============================================================================

@dataclass
class EnvelopeBounds:
    """
    Bounds for the realism envelope.
    
    These are NOT empirical bounds - they are reasonable ranges for
    synthetic data to remain plausible for stress-testing.
    """
    
    # Variance bounds
    max_per_cycle_sigma: float = 0.15
    max_per_item_sigma: float = 0.10
    
    # Correlation bounds
    min_correlation_rho: float = 0.0
    max_correlation_rho: float = 0.9
    
    # Drift amplitude bounds
    max_sinusoidal_amplitude: float = 0.25
    max_linear_slope: float = 0.005  # Per cycle
    max_step_delta: float = 0.40
    min_drift_period: int = 20  # Cycles
    
    # Rare event bounds
    max_rare_event_probability: float = 0.10  # Per cycle
    min_rare_event_duration: int = 1
    max_rare_event_magnitude: float = 0.70
    max_rare_events_per_scenario: int = 5
    
    # Probability bounds
    min_probability: float = 0.05
    max_probability: float = 0.95
    max_probability_spread: float = 0.60  # Max diff between baseline and RFL
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "variance": {
                "max_per_cycle_sigma": self.max_per_cycle_sigma,
                "max_per_item_sigma": self.max_per_item_sigma,
            },
            "correlation": {
                "min_rho": self.min_correlation_rho,
                "max_rho": self.max_correlation_rho,
            },
            "drift": {
                "max_sinusoidal_amplitude": self.max_sinusoidal_amplitude,
                "max_linear_slope": self.max_linear_slope,
                "max_step_delta": self.max_step_delta,
                "min_period": self.min_drift_period,
            },
            "rare_events": {
                "max_probability": self.max_rare_event_probability,
                "min_duration": self.min_rare_event_duration,
                "max_magnitude": self.max_rare_event_magnitude,
                "max_per_scenario": self.max_rare_events_per_scenario,
            },
            "probability": {
                "min": self.min_probability,
                "max": self.max_probability,
                "max_spread": self.max_probability_spread,
            },
        }


# Default bounds
DEFAULT_BOUNDS = EnvelopeBounds()


# ==============================================================================
# ENVELOPE VIOLATION
# ==============================================================================

@dataclass
class EnvelopeViolation:
    """A single envelope violation."""
    scenario: str
    parameter: str
    actual_value: Any
    bound_type: str  # "min" or "max"
    bound_value: Any
    severity: str = "error"  # "error" or "warning"
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "scenario": self.scenario,
            "parameter": self.parameter,
            "actual_value": self.actual_value,
            "bound_type": self.bound_type,
            "bound_value": self.bound_value,
            "severity": self.severity,
        }
    
    def __str__(self) -> str:
        """Human-readable string."""
        op = ">=" if self.bound_type == "max" else "<="
        return (
            f"[{self.severity.upper()}] {self.scenario}: "
            f"{self.parameter} = {self.actual_value} "
            f"({op} {self.bound_value})"
        )


# ==============================================================================
# ENVELOPE CHECK RESULT
# ==============================================================================

@dataclass
class EnvelopeCheckResult:
    """Result of envelope validation."""
    passed: bool = True
    violations: List[EnvelopeViolation] = field(default_factory=list)
    scenarios_checked: int = 0
    scenarios_passed: int = 0
    scenarios_failed: int = 0
    
    def add_violation(self, violation: EnvelopeViolation):
        """Add a violation."""
        self.violations.append(violation)
        if violation.severity == "error":
            self.passed = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "label": SAFETY_LABEL,
            "passed": self.passed,
            "scenarios_checked": self.scenarios_checked,
            "scenarios_passed": self.scenarios_passed,
            "scenarios_failed": self.scenarios_failed,
            "violations": [v.to_dict() for v in self.violations],
        }


# ==============================================================================
# ENVELOPE CHECKER
# ==============================================================================

class RealismEnvelopeChecker:
    """
    Validates scenario parameters against realism envelopes.
    
    This is NOT empirical validation - it's plausibility checking
    for synthetic stress-testing purposes.
    """
    
    def __init__(self, bounds: EnvelopeBounds = DEFAULT_BOUNDS):
        self.bounds = bounds
    
    def check_scenario(
        self,
        scenario_name: str,
        params: Dict[str, Any],
    ) -> List[EnvelopeViolation]:
        """
        Check a single scenario against envelope bounds.
        
        Args:
            scenario_name: Name of the scenario
            params: Scenario parameters dictionary
        
        Returns:
            List of violations found
        """
        violations = []
        
        # Check variance
        violations.extend(self._check_variance(scenario_name, params))
        
        # Check correlation
        violations.extend(self._check_correlation(scenario_name, params))
        
        # Check drift
        violations.extend(self._check_drift(scenario_name, params))
        
        # Check rare events
        violations.extend(self._check_rare_events(scenario_name, params))
        
        # Check probabilities
        violations.extend(self._check_probabilities(scenario_name, params))
        
        return violations
    
    def _check_variance(
        self,
        scenario_name: str,
        params: Dict[str, Any],
    ) -> List[EnvelopeViolation]:
        """Check variance parameters."""
        violations = []
        variance = params.get("variance", {})
        
        per_cycle = variance.get("per_cycle_sigma", 0.0)
        if per_cycle > self.bounds.max_per_cycle_sigma:
            violations.append(EnvelopeViolation(
                scenario=scenario_name,
                parameter="variance.per_cycle_sigma",
                actual_value=per_cycle,
                bound_type="max",
                bound_value=self.bounds.max_per_cycle_sigma,
            ))
        
        per_item = variance.get("per_item_sigma", 0.0)
        if per_item > self.bounds.max_per_item_sigma:
            violations.append(EnvelopeViolation(
                scenario=scenario_name,
                parameter="variance.per_item_sigma",
                actual_value=per_item,
                bound_type="max",
                bound_value=self.bounds.max_per_item_sigma,
            ))
        
        return violations
    
    def _check_correlation(
        self,
        scenario_name: str,
        params: Dict[str, Any],
    ) -> List[EnvelopeViolation]:
        """Check correlation parameters."""
        violations = []
        corr = params.get("correlation", {})
        
        rho = corr.get("rho", 0.0)
        
        if rho < self.bounds.min_correlation_rho:
            violations.append(EnvelopeViolation(
                scenario=scenario_name,
                parameter="correlation.rho",
                actual_value=rho,
                bound_type="min",
                bound_value=self.bounds.min_correlation_rho,
            ))
        
        if rho > self.bounds.max_correlation_rho:
            violations.append(EnvelopeViolation(
                scenario=scenario_name,
                parameter="correlation.rho",
                actual_value=rho,
                bound_type="max",
                bound_value=self.bounds.max_correlation_rho,
            ))
        
        return violations
    
    def _check_drift(
        self,
        scenario_name: str,
        params: Dict[str, Any],
    ) -> List[EnvelopeViolation]:
        """Check drift parameters."""
        violations = []
        drift = params.get("drift", {})
        mode = drift.get("mode", "none")
        
        if mode == "cyclical" or mode == "sinusoidal":
            amplitude = drift.get("amplitude", 0.0)
            if abs(amplitude) > self.bounds.max_sinusoidal_amplitude:
                violations.append(EnvelopeViolation(
                    scenario=scenario_name,
                    parameter="drift.amplitude",
                    actual_value=amplitude,
                    bound_type="max",
                    bound_value=self.bounds.max_sinusoidal_amplitude,
                ))
            
            period = drift.get("period", 100)
            if period < self.bounds.min_drift_period:
                violations.append(EnvelopeViolation(
                    scenario=scenario_name,
                    parameter="drift.period",
                    actual_value=period,
                    bound_type="min",
                    bound_value=self.bounds.min_drift_period,
                ))
        
        elif mode == "linear" or mode == "monotonic":
            slope = drift.get("slope", 0.0)
            if abs(slope) > self.bounds.max_linear_slope:
                violations.append(EnvelopeViolation(
                    scenario=scenario_name,
                    parameter="drift.slope",
                    actual_value=slope,
                    bound_type="max",
                    bound_value=self.bounds.max_linear_slope,
                ))
        
        elif mode == "shock" or mode == "step":
            delta = drift.get("shock_delta", 0.0)
            if abs(delta) > self.bounds.max_step_delta:
                violations.append(EnvelopeViolation(
                    scenario=scenario_name,
                    parameter="drift.shock_delta",
                    actual_value=delta,
                    bound_type="max",
                    bound_value=self.bounds.max_step_delta,
                ))
        
        return violations
    
    def _check_rare_events(
        self,
        scenario_name: str,
        params: Dict[str, Any],
    ) -> List[EnvelopeViolation]:
        """Check rare event parameters."""
        violations = []
        rare_events = params.get("rare_events", [])
        
        # Check count
        if len(rare_events) > self.bounds.max_rare_events_per_scenario:
            violations.append(EnvelopeViolation(
                scenario=scenario_name,
                parameter="rare_events.count",
                actual_value=len(rare_events),
                bound_type="max",
                bound_value=self.bounds.max_rare_events_per_scenario,
            ))
        
        # Check individual events
        for i, event in enumerate(rare_events):
            prob = event.get("trigger_probability", 0.0)
            if prob > self.bounds.max_rare_event_probability:
                violations.append(EnvelopeViolation(
                    scenario=scenario_name,
                    parameter=f"rare_events[{i}].trigger_probability",
                    actual_value=prob,
                    bound_type="max",
                    bound_value=self.bounds.max_rare_event_probability,
                ))
            
            magnitude = abs(event.get("magnitude", 0.0))
            if magnitude > self.bounds.max_rare_event_magnitude:
                violations.append(EnvelopeViolation(
                    scenario=scenario_name,
                    parameter=f"rare_events[{i}].magnitude",
                    actual_value=magnitude,
                    bound_type="max",
                    bound_value=self.bounds.max_rare_event_magnitude,
                ))
            
            duration = event.get("duration", 1)
            if duration < self.bounds.min_rare_event_duration:
                violations.append(EnvelopeViolation(
                    scenario=scenario_name,
                    parameter=f"rare_events[{i}].duration",
                    actual_value=duration,
                    bound_type="min",
                    bound_value=self.bounds.min_rare_event_duration,
                ))
        
        return violations
    
    def _check_probabilities(
        self,
        scenario_name: str,
        params: Dict[str, Any],
    ) -> List[EnvelopeViolation]:
        """Check probability parameters."""
        violations = []
        probs = params.get("probabilities", {})
        
        all_probs = []
        for mode, class_probs in probs.items():
            if isinstance(class_probs, dict):
                all_probs.extend(class_probs.values())
        
        for prob in all_probs:
            if prob < self.bounds.min_probability:
                violations.append(EnvelopeViolation(
                    scenario=scenario_name,
                    parameter="probabilities",
                    actual_value=prob,
                    bound_type="min",
                    bound_value=self.bounds.min_probability,
                ))
            
            if prob > self.bounds.max_probability:
                violations.append(EnvelopeViolation(
                    scenario=scenario_name,
                    parameter="probabilities",
                    actual_value=prob,
                    bound_type="max",
                    bound_value=self.bounds.max_probability,
                ))
        
        # Check spread between baseline and RFL
        baseline_probs = probs.get("baseline", {})
        rfl_probs = probs.get("rfl", {})
        
        for cls in baseline_probs:
            if cls in rfl_probs:
                spread = abs(baseline_probs[cls] - rfl_probs[cls])
                if spread > self.bounds.max_probability_spread:
                    violations.append(EnvelopeViolation(
                        scenario=scenario_name,
                        parameter=f"probabilities.spread.{cls}",
                        actual_value=spread,
                        bound_type="max",
                        bound_value=self.bounds.max_probability_spread,
                        severity="warning",
                    ))
        
        return violations
    
    def check_registry(
        self,
        registry_path: Optional[Path] = None,
    ) -> EnvelopeCheckResult:
        """
        Check all scenarios in the registry.
        
        Args:
            registry_path: Path to registry file (uses default if None)
        
        Returns:
            EnvelopeCheckResult with all violations
        """
        if registry_path is None:
            registry_path = Path(__file__).parent / "scenario_registry.json"
        
        with open(registry_path, "r", encoding="utf-8") as f:
            registry = json.load(f)
        
        result = EnvelopeCheckResult()
        scenarios = registry.get("scenarios", {})
        
        for name, scenario in scenarios.items():
            result.scenarios_checked += 1
            
            params = scenario.get("parameters", {})
            violations = self.check_scenario(name, params)
            
            if any(v.severity == "error" for v in violations):
                result.scenarios_failed += 1
            else:
                result.scenarios_passed += 1
            
            for v in violations:
                result.add_violation(v)
        
        return result


# ==============================================================================
# FORMATTING
# ==============================================================================

def format_envelope_report(result: EnvelopeCheckResult) -> str:
    """Format envelope check result as human-readable report."""
    lines = [
        f"\n{SAFETY_LABEL}",
        "",
        "=" * 60,
        "REALISM ENVELOPE CHECK",
        "=" * 60,
        f"Scenarios checked: {result.scenarios_checked}",
        f"Scenarios passed:  {result.scenarios_passed}",
        f"Scenarios failed:  {result.scenarios_failed}",
        "",
    ]
    
    if result.passed:
        lines.append("[PASS] All scenarios within envelope")
    else:
        lines.append("[FAIL] Envelope violations detected")
    
    if result.violations:
        lines.append("")
        lines.append("Violations:")
        lines.append("-" * 40)
        
        errors = [v for v in result.violations if v.severity == "error"]
        warnings = [v for v in result.violations if v.severity == "warning"]
        
        if errors:
            lines.append("\nErrors:")
            for v in errors:
                lines.append(f"  {v}")
        
        if warnings:
            lines.append("\nWarnings:")
            for v in warnings:
                lines.append(f"  {v}")
    
    lines.append("")
    return "\n".join(lines)


# ==============================================================================
# CLI FUNCTION
# ==============================================================================

def run_envelope_check(
    registry_path: Optional[Path] = None,
    verbose: bool = True,
) -> int:
    """
    Run envelope check and return exit code.
    
    Args:
        registry_path: Path to registry (uses default if None)
        verbose: Print report
    
    Returns:
        0 if passed, 1 if failed
    """
    checker = RealismEnvelopeChecker()
    result = checker.check_registry(registry_path)
    
    if verbose:
        print(format_envelope_report(result))
    
    return 0 if result.passed else 1


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    import sys
    
    exit_code = run_envelope_check()
    sys.exit(exit_code)

