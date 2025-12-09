#!/usr/bin/env python3
# PHASE II — NOT USED IN PHASE I
"""
Budget What-If Simulator
═══════════════════════════════════════════════════════════════════════════════

**HEURISTIC MODEL ONLY — NOT EMPIRICAL DATA**

This tool provides approximate what-if analysis for budget parameter changes.
It uses a simple deterministic model to estimate how budget metrics might change
when parameters are scaled. 

⚠️  IMPORTANT LIMITATIONS:
    - This is a STATIC HEURISTIC, not a simulation or replay.
    - Real-world results will vary based on slice characteristics.
    - Use only for rough planning; do not rely on exact numbers.
    - Does NOT integrate with production pipelines.

Usage:
    # Double the cycle budget
    uv run python experiments/budget_simulator.py baseline.json --scale-budget 2.0
    
    # Halve the timeout
    uv run python experiments/budget_simulator.py baseline.json --scale-timeout 0.5
    
    # Combined scenario
    uv run python experiments/budget_simulator.py baseline.json --scale-budget 1.5 --scale-timeout 0.8

Model Assumptions:
    - budget_exhausted_pct scales inversely with budget: pct_new ≈ pct_old / scale
    - timeout_abstentions_avg scales inversely with timeout: avg_new ≈ avg_old / scale
    - max_candidates_hit is unaffected by these parameters

═══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


# =============================================================================
# HEURISTIC MODEL
# =============================================================================

# Minimum values to prevent division by zero or unrealistic estimates
MIN_EXHAUSTED_PCT = 0.001
MIN_TIMEOUT_AVG = 0.001

# Maximum values (100% exhausted, etc.)
MAX_EXHAUSTED_PCT = 100.0
MAX_TIMEOUT_AVG = 100.0


def estimate_exhausted_pct(baseline_pct: float, budget_scale: float) -> float:
    """
    Estimate budget_exhausted_pct after scaling budget.
    
    HEURISTIC MODEL:
        If budget is doubled, roughly half as many cycles should run out of time.
        pct_new ≈ pct_old / budget_scale
    
    This is a simplification; real behavior depends on cycle duration distribution.
    
    Args:
        baseline_pct: Current budget_exhausted_pct
        budget_scale: Multiplier for cycle_budget_s (e.g., 2.0 = double budget)
        
    Returns:
        Estimated new budget_exhausted_pct
    """
    if budget_scale <= 0:
        raise ValueError("budget_scale must be positive")
    
    # Simple inverse scaling
    estimated = baseline_pct / budget_scale
    
    # Clamp to valid range
    return max(MIN_EXHAUSTED_PCT, min(MAX_EXHAUSTED_PCT, estimated))


def estimate_timeout_avg(baseline_avg: float, timeout_scale: float) -> float:
    """
    Estimate timeout_abstentions_avg after scaling timeout.
    
    HEURISTIC MODEL:
        If timeout is doubled, roughly half as many candidates should timeout.
        avg_new ≈ avg_old / timeout_scale
    
    This is a simplification; real behavior depends on verification time distribution.
    
    Args:
        baseline_avg: Current timeout_abstentions_avg
        timeout_scale: Multiplier for taut_timeout_s (e.g., 2.0 = double timeout)
        
    Returns:
        Estimated new timeout_abstentions_avg
    """
    if timeout_scale <= 0:
        raise ValueError("timeout_scale must be positive")
    
    # Simple inverse scaling
    estimated = baseline_avg / timeout_scale
    
    # Clamp to valid range
    return max(MIN_TIMEOUT_AVG, min(MAX_TIMEOUT_AVG, estimated))


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class SimulationScenario:
    """A what-if simulation scenario."""
    budget_scale: float
    timeout_scale: float
    
    def describe(self) -> str:
        """Human-readable description of the scenario."""
        parts = []
        if self.budget_scale != 1.0:
            if self.budget_scale > 1.0:
                parts.append(f"budget ×{self.budget_scale:.1f} (increased)")
            else:
                parts.append(f"budget ×{self.budget_scale:.1f} (decreased)")
        if self.timeout_scale != 1.0:
            if self.timeout_scale > 1.0:
                parts.append(f"timeout ×{self.timeout_scale:.1f} (increased)")
            else:
                parts.append(f"timeout ×{self.timeout_scale:.1f} (decreased)")
        return ", ".join(parts) if parts else "no changes"


@dataclass
class SimulationResult:
    """Result of a what-if simulation."""
    slice_name: str
    scenario: SimulationScenario
    baseline_metrics: Dict[str, float]
    simulated_metrics: Dict[str, float]
    baseline_status: str
    simulated_status: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "slice_name": self.slice_name,
            "scenario": {
                "budget_scale": self.scenario.budget_scale,
                "timeout_scale": self.scenario.timeout_scale,
                "description": self.scenario.describe(),
            },
            "baseline": {
                "metrics": self.baseline_metrics,
                "status": self.baseline_status,
            },
            "simulated": {
                "metrics": self.simulated_metrics,
                "status": self.simulated_status,
            },
            "status_change": self.baseline_status != self.simulated_status,
        }


# =============================================================================
# SIMULATION FUNCTIONS
# =============================================================================


def simulate_scenario(
    slice_name: str,
    baseline_metrics: Dict[str, float],
    baseline_status: str,
    scenario: SimulationScenario,
) -> SimulationResult:
    """
    Run a what-if simulation for a single slice.
    
    Args:
        slice_name: Name of the slice
        baseline_metrics: Current budget metrics
        baseline_status: Current health status
        scenario: Scaling parameters to simulate
        
    Returns:
        SimulationResult with estimated metrics and status
    """
    # Extract baseline values
    baseline_exhausted = baseline_metrics.get("budget_exhausted_pct", 0.0)
    baseline_timeout = baseline_metrics.get("timeout_abstentions_avg", 0.0)
    baseline_maxcand = baseline_metrics.get("max_candidates_hit_pct", 0.0)
    
    # Apply heuristic model
    simulated_exhausted = estimate_exhausted_pct(baseline_exhausted, scenario.budget_scale)
    simulated_timeout = estimate_timeout_avg(baseline_timeout, scenario.timeout_scale)
    
    # max_candidates_hit is unaffected by budget/timeout scaling
    simulated_maxcand = baseline_maxcand
    
    simulated_metrics = {
        "budget_exhausted_pct": round(simulated_exhausted, 4),
        "timeout_abstentions_avg": round(simulated_timeout, 4),
        "max_candidates_hit_pct": round(simulated_maxcand, 4),
    }
    
    # Re-classify health using the classifier
    from experiments.summarize_budget_usage import (
        BudgetSummary,
        classify_budget_health,
        THRESHOLD_EXHAUSTED_SAFE,
        THRESHOLD_EXHAUSTED_TIGHT,
        THRESHOLD_TIMEOUT_SAFE,
        THRESHOLD_TIMEOUT_TIGHT,
    )
    
    # Determine status from simulated metrics
    if simulated_exhausted > THRESHOLD_EXHAUSTED_TIGHT or simulated_timeout > THRESHOLD_TIMEOUT_TIGHT:
        simulated_status = "STARVED"
    elif simulated_exhausted >= THRESHOLD_EXHAUSTED_SAFE or simulated_timeout >= THRESHOLD_TIMEOUT_SAFE:
        simulated_status = "TIGHT"
    else:
        simulated_status = "SAFE"
    
    return SimulationResult(
        slice_name=slice_name,
        scenario=scenario,
        baseline_metrics=baseline_metrics,
        simulated_metrics=simulated_metrics,
        baseline_status=baseline_status,
        simulated_status=simulated_status,
    )


def load_baseline(path: Path) -> List[Dict[str, Any]]:
    """
    Load baseline data from a health JSON file.
    
    Args:
        path: Path to health JSON file
        
    Returns:
        List of slice data with metrics
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    results = []
    
    # Health JSON format
    if "health_report" in data:
        for entry in data["health_report"]:
            results.append({
                "slice_name": entry.get("slice", "unknown"),
                "status": entry.get("health", {}).get("status", "INVALID"),
                "metrics": entry.get("health", {}).get("metrics", {}),
            })
    
    # Basic JSON format
    elif "logs" in data:
        for log_entry in data["logs"]:
            budget = log_entry.get("budget", {})
            total_cycles = log_entry.get("total_cycles", 1)
            
            results.append({
                "slice_name": log_entry.get("slice", "unknown"),
                "status": "UNKNOWN",
                "metrics": {
                    "budget_exhausted_pct": budget.get("exhausted_pct", 0.0),
                    "timeout_abstentions_avg": budget.get("timeout_abstentions_avg", 0.0),
                    "max_candidates_hit_pct": budget.get("max_candidates_hit_pct", 0.0),
                },
            })
    
    return results


# =============================================================================
# OUTPUT FORMATTERS
# =============================================================================


STATUS_EMOJI = {
    "SAFE": "✅",
    "TIGHT": "⚠️",
    "STARVED": "🔥",
    "INVALID": "❓",
    "UNKNOWN": "❓",
}


def format_markdown(results: List[SimulationResult], scenario: SimulationScenario) -> str:
    """Format simulation results as Markdown."""
    lines = []
    
    lines.append("## 🔮 Budget What-If Simulation")
    lines.append("")
    lines.append("> **HEURISTIC MODEL ONLY — NOT EMPIRICAL DATA**")
    lines.append("> ")
    lines.append("> This is a static estimate. Real-world results will vary.")
    lines.append("")
    
    lines.append(f"### Scenario: {scenario.describe()}")
    lines.append("")
    
    lines.append("| Slice | Baseline | Simulated | Change |")
    lines.append("|-------|----------|-----------|--------|")
    
    for r in results:
        b_emoji = STATUS_EMOJI.get(r.baseline_status, "")
        s_emoji = STATUS_EMOJI.get(r.simulated_status, "")
        
        if r.baseline_status != r.simulated_status:
            change = f"{b_emoji} → {s_emoji}"
        else:
            change = "—"
        
        lines.append(
            f"| `{r.slice_name}` | {b_emoji} {r.baseline_status} | "
            f"{s_emoji} {r.simulated_status} | {change} |"
        )
    
    lines.append("")
    
    # Detailed metrics
    lines.append("### Detailed Metrics")
    lines.append("")
    lines.append("| Slice | Metric | Baseline | Simulated |")
    lines.append("|-------|--------|----------|-----------|")
    
    for r in results:
        lines.append(
            f"| `{r.slice_name}` | budget_exhausted_pct | "
            f"{r.baseline_metrics.get('budget_exhausted_pct', 0):.2f}% | "
            f"{r.simulated_metrics.get('budget_exhausted_pct', 0):.2f}% |"
        )
        lines.append(
            f"| | timeout_abstentions_avg | "
            f"{r.baseline_metrics.get('timeout_abstentions_avg', 0):.3f} | "
            f"{r.simulated_metrics.get('timeout_abstentions_avg', 0):.3f} |"
        )
    
    lines.append("")
    lines.append("---")
    lines.append("*This is a heuristic estimate. Actual results depend on slice characteristics.*")
    
    return "\n".join(lines)


def format_json(results: List[SimulationResult]) -> str:
    """Format simulation results as JSON."""
    output = {
        "phase": "PHASE II — NOT USED IN PHASE I",
        "warning": "HEURISTIC MODEL ONLY — NOT EMPIRICAL DATA",
        "results": [r.to_dict() for r in results],
    }
    return json.dumps(output, indent=2)


# =============================================================================
# CLI
# =============================================================================


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Budget What-If Simulator — HEURISTIC MODEL ONLY.\n\n"
            "⚠️  This is a static estimate, not a simulation or replay.\n"
            "    Real-world results will vary based on slice characteristics."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "baseline",
        type=str,
        help="Path to baseline health JSON file."
    )
    parser.add_argument(
        "--scale-budget",
        type=float,
        default=1.0,
        help="Multiplier for cycle_budget_s (e.g., 2.0 = double). Default: 1.0"
    )
    parser.add_argument(
        "--scale-timeout",
        type=float,
        default=1.0,
        help="Multiplier for taut_timeout_s (e.g., 0.5 = halve). Default: 1.0"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON."
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Write output to file instead of stdout."
    )
    
    args = parser.parse_args()
    
    print("PHASE II — NOT USED IN PHASE I", file=sys.stderr)
    print("⚠️  HEURISTIC MODEL ONLY — NOT EMPIRICAL DATA", file=sys.stderr)
    print("", file=sys.stderr)
    
    # Validate inputs
    baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        print(f"ERROR: Baseline file not found: {baseline_path}", file=sys.stderr)
        sys.exit(1)
    
    if args.scale_budget <= 0:
        print("ERROR: --scale-budget must be positive", file=sys.stderr)
        sys.exit(1)
    
    if args.scale_timeout <= 0:
        print("ERROR: --scale-timeout must be positive", file=sys.stderr)
        sys.exit(1)
    
    # Load baseline
    try:
        baseline_data = load_baseline(baseline_path)
    except Exception as e:
        print(f"ERROR: Failed to load baseline: {e}", file=sys.stderr)
        sys.exit(1)
    
    if not baseline_data:
        print("ERROR: No slice data found in baseline file.", file=sys.stderr)
        sys.exit(1)
    
    # Create scenario
    scenario = SimulationScenario(
        budget_scale=args.scale_budget,
        timeout_scale=args.scale_timeout,
    )
    
    # Run simulation
    results = []
    for entry in baseline_data:
        result = simulate_scenario(
            slice_name=entry["slice_name"],
            baseline_metrics=entry["metrics"],
            baseline_status=entry["status"],
            scenario=scenario,
        )
        results.append(result)
    
    # Format output
    if args.json:
        output = format_json(results)
    else:
        output = format_markdown(results, scenario)
    
    # Write output
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"Output written to: {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()

