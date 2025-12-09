#!/usr/bin/env python3
"""
Truth-Table Oracle Introspection CLI

PHASE II (Agent B2) - Diagnostic and Analysis Tool
===================================================

Provides deep inspection of truth-table oracle behavior for individual
formulas, including diagnostics and CHI estimation.

Usage:
    python experiments/inspect_truth_table_oracle.py --formula "p -> p"
    python experiments/inspect_truth_table_oracle.py --formula "p -> q" --timeout-ms 100
    python experiments/inspect_truth_table_oracle.py --formula "((p -> q) -> p) -> p" --mode chi-only

Exit Codes:
    0: Oracle ran successfully (TAUTOLOGY, NOT_TAUTOLOGY, or TIMEOUT result)
    1: CLI argument error or internal error

Output Modes:
    auto:            Full report with result, diagnostics, and CHI (default)
    diagnostic-only: Result and diagnostics without CHI analysis
    chi-only:        Result and CHI analysis without raw diagnostics

Flags:
    --emit-policy-signal: Output JSON HardnessPolicySignal block
    --diagnostic-report:  Force diagnostics and print incident report format
"""

from __future__ import annotations

import argparse
import os
import sys
from enum import Enum
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Enable diagnostics by default for this CLI
os.environ["TT_ORACLE_DIAGNOSTIC"] = "1"

from normalization.taut import (
    truth_table_is_tautology,
    TruthTableTimeout,
    get_last_diagnostics,
    clear_diagnostics,
    _cached_normalize,
    _cached_extract_atoms,
)
from normalization.tt_chi import (
    chi_from_diagnostics,
    format_chi_report,
    format_chi_compact,
    estimate_timeout_budget,
    classify_hardness,
    get_hardness_description,
    CHIResult,
    suggest_timeout_ms,
    HardnessPolicySignal,
    format_diagnostics_for_report,
)


# =============================================================================
# EXIT CODES
# =============================================================================

EXIT_SUCCESS = 0  # Oracle ran and returned a result (TAUTOLOGY/NOT/TIMEOUT)
EXIT_ERROR = 1    # CLI argument error or internal error


# =============================================================================
# OUTPUT MODE
# =============================================================================

class OutputMode(Enum):
    AUTO = "auto"
    DIAGNOSTIC_ONLY = "diagnostic-only"
    CHI_ONLY = "chi-only"


# =============================================================================
# RESULT FORMATTING
# =============================================================================

def format_result_banner(result: str, is_timeout: bool = False) -> str:
    """Create a visually distinct result banner."""
    if is_timeout:
        return """
╔══════════════════════════════════════════════════════════════╗
║                    ⏱️  TIMEOUT (ABSTAIN)                      ║
║           Could not determine within time budget              ║
╚══════════════════════════════════════════════════════════════╝
"""
    elif result == "TAUTOLOGY":
        return """
╔══════════════════════════════════════════════════════════════╗
║                    ✅ VERIFIED: TAUTOLOGY                     ║
║           Formula is true for all truth assignments           ║
╚══════════════════════════════════════════════════════════════╝
"""
    else:
        return """
╔══════════════════════════════════════════════════════════════╗
║                  ❌ NOT VERIFIED: NOT TAUTOLOGY               ║
║         Formula is false for at least one assignment          ║
╚══════════════════════════════════════════════════════════════╝
"""


def format_result_compact(result: str, is_timeout: bool = False) -> str:
    """Compact result line."""
    if is_timeout:
        return "Result: TIMEOUT (abstain - could not determine)"
    elif result == "TAUTOLOGY":
        return "Result: TAUTOLOGY (verified)"
    else:
        return "Result: NOT_TAUTOLOGY (not verified)"


def format_diagnostics_panel(diag: dict) -> str:
    """Format diagnostics as a human-readable report."""
    lines = [
        "┌──────────────────────────────────────────────────────────────┐",
        "│                    ORACLE DIAGNOSTICS                        │",
        "├──────────────────────────────────────────────────────────────┤",
    ]
    
    # Formula info
    formula = diag.get("formula", "N/A")
    if len(formula) > 50:
        formula = formula[:47] + "..."
    lines.append(f"│  Original Formula:    {formula:<38}│")
    
    norm = diag.get("normalized_formula", "N/A")
    if len(norm) > 50:
        norm = norm[:47] + "..."
    lines.append(f"│  Normalized:          {norm:<38}│")
    
    lines.append("├──────────────────────────────────────────────────────────────┤")
    
    # Complexity metrics
    atom_count = diag.get("atom_count", 0)
    assignment_count = diag.get("assignment_count", 0)
    evaluated = diag.get("assignments_evaluated", 0)
    
    lines.append(f"│  Atom Count:          {atom_count:<38}│")
    lines.append(f"│  Total Assignments:   {assignment_count:<38}│")
    lines.append(f"│  Assignments Checked: {evaluated:<38}│")
    
    lines.append("├──────────────────────────────────────────────────────────────┤")
    
    # Timing
    elapsed_ns = diag.get("elapsed_ns", 0)
    elapsed_us = elapsed_ns / 1000
    elapsed_ms = elapsed_ns / 1_000_000
    
    if elapsed_ms >= 1:
        lines.append(f"│  Elapsed Time:        {elapsed_ms:.3f} ms{'':<29}│")
    else:
        lines.append(f"│  Elapsed Time:        {elapsed_us:.1f} μs{'':<29}│")
    
    # Throughput
    if evaluated > 0:
        throughput = elapsed_ns / evaluated
        lines.append(f"│  Throughput:          {throughput:.1f} ns/assignment{'':<20}│")
    
    lines.append("├──────────────────────────────────────────────────────────────┤")
    
    # Flags
    short_circuit = diag.get("short_circuit_triggered", False)
    timeout = diag.get("timeout_flag", False)
    cache_norm = diag.get("cache_hit_normalize", False)
    cache_atoms = diag.get("cache_hit_atoms", False)
    
    lines.append(f"│  Short-Circuit:       {'Yes' if short_circuit else 'No':<38}│")
    lines.append(f"│  Timeout:             {'Yes' if timeout else 'No':<38}│")
    lines.append(f"│  Cache Hit (norm):    {'Yes' if cache_norm else 'No':<38}│")
    lines.append(f"│  Cache Hit (atoms):   {'Yes' if cache_atoms else 'No':<38}│")
    
    # Short-circuit details
    if short_circuit:
        failing = diag.get("short_circuit_at_assignment")
        if failing:
            atoms = _cached_extract_atoms(diag.get("normalized_formula", ""))
            assignment_str = ", ".join(
                f"{a}={'T' if v else 'F'}" 
                for a, v in zip(atoms, failing)
            )
            if len(assignment_str) > 38:
                assignment_str = assignment_str[:35] + "..."
            lines.append(f"│  Failing Assignment:  {assignment_str:<38}│")
    
    lines.append("└──────────────────────────────────────────────────────────────┘")
    
    return "\n".join(lines)


def format_recommendations(chi_result: CHIResult) -> str:
    """Format recommendations based on CHI analysis."""
    lines = [
        "┌──────────────────────────────────────────────────────────────┐",
        "│                    RECOMMENDATIONS                           │",
        "├──────────────────────────────────────────────────────────────┤",
    ]
    
    category = chi_result.hardness_category
    desc = chi_result.hardness_description
    
    if category == "extreme":
        lines.append("│  ⚠️  EXTREME hardness detected                                │")
        lines.append(f"│     {desc:<56}│")
        lines.append("│     Consider: alternative proof methods, formula splitting   │")
    elif category == "hard":
        lines.append("│  ⚠️  HARD hardness detected                                  │")
        lines.append(f"│     {desc:<56}│")
        lines.append("│     Consider: adding to pattern matcher or using timeout     │")
    elif chi_result.is_short_circuited:
        lines.append("│  ✅ Non-tautology detected efficiently via short-circuit     │")
        lines.append(f"│     Only {chi_result.efficiency_ratio*100:.0f}% of assignments evaluated{'':<24}│")
    else:
        lines.append("│  ✅ Formula evaluated within acceptable parameters           │")
        lines.append(f"│     {desc:<56}│")
    
    # Suggested timeout based on CHI (TASK 1)
    suggested = chi_result.suggested_timeout_ms
    lines.append(f"│  Suggested timeout: ~{suggested} ms (based on CHI){'':<21}│")
    
    lines.append("└──────────────────────────────────────────────────────────────┘")
    
    return "\n".join(lines)


# =============================================================================
# MAIN INSPECTION FUNCTION
# =============================================================================

def inspect_formula(
    formula: str,
    timeout_ms: Optional[int] = None,
    mode: OutputMode = OutputMode.AUTO,
    emit_policy_signal: bool = False,
    diagnostic_report: bool = False,
) -> int:
    """
    Inspect a single formula and print detailed report.
    
    Args:
        formula: The propositional formula to inspect
        timeout_ms: Optional timeout in milliseconds
        mode: Output mode (auto, diagnostic-only, chi-only)
        emit_policy_signal: If True, output JSON HardnessPolicySignal block
        diagnostic_report: If True, force diagnostics and print incident format
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    show_diagnostics = mode in (OutputMode.AUTO, OutputMode.DIAGNOSTIC_ONLY) or diagnostic_report
    show_chi = mode in (OutputMode.AUTO, OutputMode.CHI_ONLY)
    
    # Header
    print("\n" + "=" * 64)
    print("         TRUTH-TABLE ORACLE INSPECTION REPORT")
    print("=" * 64)
    
    # Normalize for display
    try:
        normalized = _cached_normalize(formula)
        atoms = _cached_extract_atoms(normalized)
    except Exception as e:
        print(f"\n❌ Error parsing formula: {e}")
        return EXIT_ERROR
    
    print(f"\nFormula:    {formula}")
    print(f"Normalized: {normalized}")
    print(f"Atoms:      {', '.join(atoms) if atoms else '(none)'}")
    print(f"Timeout:    {timeout_ms}ms" if timeout_ms else "Timeout:    None (unlimited)")
    print(f"Mode:       {mode.value}")
    
    # Clear any previous diagnostics
    clear_diagnostics()
    
    # Run the oracle
    result_str = "UNKNOWN"
    is_timeout = False
    
    try:
        result = truth_table_is_tautology(formula, timeout_ms=timeout_ms)
        result_str = "TAUTOLOGY" if result else "NOT_TAUTOLOGY"
    except TruthTableTimeout as e:
        result_str = "TIMEOUT"
        is_timeout = True
        print(f"\n⚠️  Timeout after {e.timeout_ms}ms")
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        return EXIT_ERROR
    
    # Result banner (always shown)
    print(format_result_banner(result_str, is_timeout))
    
    # Get diagnostics
    diag = get_last_diagnostics()
    
    # Diagnostics section
    if show_diagnostics and diag:
        print(format_diagnostics_panel(diag))
    
    # CHI Analysis section
    chi_result: Optional[CHIResult] = None
    if diag:
        try:
            chi_result = chi_from_diagnostics(diag)
        except Exception as e:
            print(f"\n⚠️  Could not compute CHI: {e}")
    
    if show_chi and chi_result:
        if mode == OutputMode.CHI_ONLY:
            # Compact CHI format
            print("\n" + format_chi_compact(chi_result))
        else:
            # Full CHI report
            print("\n" + format_chi_report(chi_result))
        
        # Recommendations with suggested timeout (TASK 1)
        print("\n" + format_recommendations(chi_result))
    elif show_chi and not chi_result and not show_diagnostics:
        # In chi-only mode but no diagnostics available
        print("\n⚠️  CHI analysis not available (no diagnostics)")
    
    # Policy Signal output (TASK 2)
    if emit_policy_signal and chi_result:
        policy_signal = HardnessPolicySignal.from_chi_result(chi_result)
        print("\n┌──────────────────────────────────────────────────────────────┐")
        print("│                    POLICY SIGNAL (JSON)                      │")
        print("├──────────────────────────────────────────────────────────────┤")
        print(policy_signal.to_json(indent=2))
        print("└──────────────────────────────────────────────────────────────┘")
    
    # Diagnostic Report for Incident Bundles (TASK 3)
    if diagnostic_report and diag:
        print("\n┌──────────────────────────────────────────────────────────────┐")
        print("│              INCIDENT BUNDLE DIAGNOSTIC REPORT               │")
        print("├──────────────────────────────────────────────────────────────┤")
        print(format_diagnostics_for_report(diag))
        print("└──────────────────────────────────────────────────────────────┘")
    
    print("\n" + "=" * 64)
    
    return EXIT_SUCCESS


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main() -> int:
    """
    CLI entry point.
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    parser = argparse.ArgumentParser(
        description="Inspect truth-table oracle behavior for a formula",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full inspection (default mode: auto)
    python experiments/inspect_truth_table_oracle.py --formula "p -> p"
    
    # With timeout
    python experiments/inspect_truth_table_oracle.py --formula "p -> q" --timeout-ms 100
    
    # Diagnostics only (no CHI)
    python experiments/inspect_truth_table_oracle.py --formula "p -> p" --mode diagnostic-only
    
    # CHI only (no raw diagnostics)
    python experiments/inspect_truth_table_oracle.py --formula "p -> p" --mode chi-only
    
    # Emit policy signal as JSON
    python experiments/inspect_truth_table_oracle.py --formula "p -> p" --emit-policy-signal
    
    # Generate incident bundle diagnostic report
    python experiments/inspect_truth_table_oracle.py --formula "p -> p" --diagnostic-report
    
    # Complex formula
    python experiments/inspect_truth_table_oracle.py --formula "(p -> q) -> ((q -> r) -> (p -> r))"

Exit Codes:
    0: Oracle ran successfully (TAUTOLOGY, NOT_TAUTOLOGY, or TIMEOUT)
    1: CLI argument error or internal error
"""
    )
    
    parser.add_argument(
        "--formula", "-f",
        type=str,
        required=True,
        help="Propositional formula to inspect (use quotes)"
    )
    
    parser.add_argument(
        "--timeout-ms", "-t",
        type=int,
        default=None,
        help="Timeout in milliseconds (default: unlimited)"
    )
    
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["auto", "diagnostic-only", "chi-only"],
        default="auto",
        help="Output mode: auto (full), diagnostic-only, chi-only (default: auto)"
    )
    
    # New flags for Phase II UX & Policy Layer
    parser.add_argument(
        "--emit-policy-signal",
        action="store_true",
        help="Output JSON HardnessPolicySignal block (for Evidence Packs)"
    )
    
    parser.add_argument(
        "--diagnostic-report",
        action="store_true",
        help="Force diagnostics and print incident bundle format"
    )
    
    # Legacy flags for backward compatibility
    parser.add_argument(
        "--diagnostic", "-d",
        action="store_true",
        help="[Deprecated] Use --mode diagnostic-only instead"
    )
    
    parser.add_argument(
        "--no-diagnostic",
        action="store_true",
        help="[Deprecated] Use --mode chi-only instead"
    )
    
    parser.add_argument(
        "--chi", "-c",
        action="store_true",
        help="[Deprecated] Use --mode chi-only instead"
    )
    
    parser.add_argument(
        "--no-chi",
        action="store_true",
        help="[Deprecated] Use --mode diagnostic-only instead"
    )
    
    try:
        args = parser.parse_args()
    except SystemExit as e:
        return EXIT_ERROR if e.code != 0 else EXIT_SUCCESS
    
    # Parse mode (handle legacy flags)
    mode = OutputMode.AUTO
    if args.mode == "diagnostic-only" or args.no_chi:
        mode = OutputMode.DIAGNOSTIC_ONLY
    elif args.mode == "chi-only" or (args.chi and args.no_diagnostic):
        mode = OutputMode.CHI_ONLY
    elif args.mode == "auto":
        mode = OutputMode.AUTO
    
    return inspect_formula(
        formula=args.formula,
        timeout_ms=args.timeout_ms,
        mode=mode,
        emit_policy_signal=args.emit_policy_signal,
        diagnostic_report=args.diagnostic_report,
    )


if __name__ == "__main__":
    sys.exit(main())
