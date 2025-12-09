#!/usr/bin/env python
# PHASE II — NOT USED IN PHASE I
"""
Curriculum Diagnostics CLI

Command-line tool for analyzing Phase II curriculum slices, including
decoy difficulty scoring, confusability analysis, and CI gates.

Usage:
    python -m experiments.curriculum_diagnostics --decoy-report
    python -m experiments.curriculum_diagnostics --decoy-map <slice>
    python -m experiments.curriculum_diagnostics --export-confusability <slice>
    python -m experiments.curriculum_diagnostics --verify-confusability <slice> [--explain]
    python -m experiments.curriculum_diagnostics --decoy-ci-summary
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from experiments.decoys.loader import CurriculumDecoyLoader
from experiments.decoys.scoring import (
    score_all_slices,
    generate_markdown_report,
    generate_json_report,
)
from experiments.decoys.confusability import (
    ConfusabilityMap,
    get_confusability_map,
    get_all_confusability_maps,
    compute_confusability,
)
from experiments.decoys.contract import (
    ConfusabilityContract,
    export_contract,
    SCHEMA_VERSION,
)
from experiments.decoys.risk import (
    build_family_risk_snapshot,
    compare_family_risk,
    summarize_confusability_for_global_health,
    RISK_SCHEMA_VERSION,
)


# =============================================================================
# CONFUSABILITY CONTRACT THRESHOLDS
# =============================================================================

# Default thresholds for confusability verification
# These are CI-friendly: soft warnings vs hard failures
DEFAULT_THRESHOLDS = {
    "near_avg_min": 0.7,      # Near-decoys must average >= 0.7 confusability
    "far_avg_max": 0.5,       # Far-decoys must average <= 0.5 confusability
    "near_individual_min": 0.4,  # No near-decoy below 0.4
}

# Output directory for confusability contracts
CONFUSABILITY_CONTRACT_DIR = Path("artifacts/phase_ii/decoy_confusability")


class VerificationStatus(Enum):
    """Status codes for confusability verification."""
    OK = "OK"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIPPED = "SKIPPED"  # For legacy slices with no decoys


@dataclass
class NarrativeBlock:
    """
    Three-block narrative explanation for failures.
    
    A. WHAT FAILED - Identifies the specific failure
    B. WHY IT FAILED - Threshold, value, expected range
    C. HOW TO EXAMINE - Non-prescriptive investigation guidance
    
    IMPORTANT: Block C must NOT contain prescriptive verbs ("fix", "change", "modify").
    It provides diagnostic guidance only.
    """
    what_failed: str      # Block A: What specific check failed
    why_it_failed: str    # Block B: Threshold, actual value, expected range
    how_to_examine: str   # Block C: Non-prescriptive investigation guidance
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "A_what_failed": self.what_failed,
            "B_why_it_failed": self.why_it_failed,
            "C_how_to_examine": self.how_to_examine,
        }
    
    def format_text(self) -> str:
        """Format as readable text with exact block headings."""
        return (
            f"    A. WHAT FAILED:\n"
            f"       {self.what_failed}\n"
            f"    B. WHY IT FAILED:\n"
            f"       {self.why_it_failed}\n"
            f"    C. HOW TO EXAMINE:\n"
            f"       {self.how_to_examine}"
        )


@dataclass
class FailureExplanation:
    """Detailed explanation of a verification failure."""
    threshold_name: str  # e.g., "near_avg_min"
    threshold_value: float
    actual_value: float
    formula_name: Optional[str] = None  # For individual formula failures
    formula_confusability: Optional[float] = None
    formula_difficulty: Optional[float] = None
    rationale: str = ""
    narrative: Optional[NarrativeBlock] = None  # v1.2: 3-block narrative
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "threshold_name": self.threshold_name,
            "threshold_value": round(self.threshold_value, 4),
            "actual_value": round(self.actual_value, 4),
            "rationale": self.rationale,
        }
        if self.formula_name:
            result["formula_name"] = self.formula_name
            result["formula_confusability"] = round(self.formula_confusability or 0, 4)
            result["formula_difficulty"] = round(self.formula_difficulty or 0, 4)
        if self.narrative:
            result["narrative"] = self.narrative.to_dict()
        return result


@dataclass
class VerificationResult:
    """Result of confusability verification for a slice."""
    slice_name: str
    status: VerificationStatus
    near_avg: float
    far_avg: float
    bridge_count: int
    has_decoys: bool
    reasons: List[str] = field(default_factory=list)
    explanations: List[FailureExplanation] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "slice_name": self.slice_name,
            "status": self.status.value,
            "near_avg": round(self.near_avg, 4),
            "far_avg": round(self.far_avg, 4),
            "bridge_count": self.bridge_count,
            "has_decoys": self.has_decoys,
            "reasons": self.reasons,
            "explanations": [e.to_dict() for e in self.explanations],
        }


# =============================================================================
# EXISTING COMMANDS
# =============================================================================

def cmd_decoy_report(
    args: argparse.Namespace,
    loader: CurriculumDecoyLoader,
) -> int:
    """Generate and display decoy difficulty report."""
    
    if args.slice:
        try:
            report = loader.get_slice_report(args.slice)
            reports = {args.slice: report}
        except (KeyError, ValueError) as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    else:
        reports = loader.get_all_reports()
    
    if not reports:
        print("No uplift slices found with formula_pool_entries.", file=sys.stderr)
        return 1
    
    if args.format == "json":
        output = json.dumps(generate_json_report(reports), indent=2)
    else:
        output = generate_markdown_report(reports)
    
    if args.output:
        Path(args.output).write_text(output, encoding='utf-8')
        print(f"Report written to {args.output}")
    else:
        print(output)
    
    return 0


def cmd_check_monotonicity(
    args: argparse.Namespace,
    loader: CurriculumDecoyLoader,
) -> int:
    """Check that near-decoys are harder than far-decoys."""
    warnings = loader.check_monotonicity()
    
    if warnings:
        print("Monotonicity Warnings:")
        for w in warnings:
            print(f"  - {w}")
        return 1
    else:
        print("✓ All slices pass monotonicity check (near > far difficulty)")
        return 0


def cmd_check_collisions(
    args: argparse.Namespace,
    loader: CurriculumDecoyLoader,
) -> int:
    """Check that no decoy hashes collide with targets."""
    errors = loader.check_target_collisions()
    
    if errors:
        print("CRITICAL ERRORS - Target Collisions Found:")
        for e in errors:
            print(f"  - {e}")
        return 1
    else:
        print("✓ No target collisions found")
        return 0


def cmd_summary(
    args: argparse.Namespace,
    loader: CurriculumDecoyLoader,
) -> int:
    """Print a brief summary of all slices."""
    slices = loader.list_uplift_slices()
    
    if not slices:
        print("No uplift slices found.", file=sys.stderr)
        return 1
    
    print("Phase II Uplift Slices with Decoy Framework:")
    print("-" * 60)
    print(f"{'Slice':<30} {'Targets':>8} {'Near':>6} {'Far':>6} {'Conf':>8}")
    print("-" * 60)
    
    for name in slices:
        report = loader.get_slice_report(name)
        print(
            f"{name:<30} {len(report.targets):>8} "
            f"{len(report.decoys_near):>6} {len(report.decoys_far):>6} "
            f"{report.confusability_index:>8.3f}"
        )
    
    print("-" * 60)
    
    mono_warnings = loader.check_monotonicity()
    collision_errors = loader.check_target_collisions()
    
    if mono_warnings:
        print(f"\n⚠ {len(mono_warnings)} monotonicity warning(s)")
    else:
        print("\n✓ Monotonicity OK")
    
    if collision_errors:
        print(f"✗ {len(collision_errors)} target collision(s)!")
    else:
        print("✓ No target collisions")
    
    return 0


def cmd_decoy_map(
    args: argparse.Namespace,
    loader: CurriculumDecoyLoader,
) -> int:
    """Generate confusability map for a slice."""
    slice_name = args.decoy_map
    
    try:
        cmap = ConfusabilityMap(slice_name, args.config)
        report = cmap.generate_report()
    except (KeyError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    if args.format == "json":
        output = json.dumps(report.to_dict(), indent=2)
    else:
        output = _format_confusability_map_markdown(report)
    
    if args.output:
        Path(args.output).write_text(output, encoding='utf-8')
        print(f"Confusability map written to {args.output}")
    else:
        print(output)
    
    return 0


def cmd_decoy_stats(
    args: argparse.Namespace,
    loader: CurriculumDecoyLoader,
) -> int:
    """Generate statistical summary of decoy distribution."""
    slice_name = args.decoy_stats
    
    try:
        cmap = ConfusabilityMap(slice_name, args.config)
        stats = cmap.get_decoy_stats()
    except (KeyError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    if args.format == "json":
        output = json.dumps(stats, indent=2)
    else:
        output = _format_decoy_stats_markdown(stats)
    
    if args.output:
        Path(args.output).write_text(output, encoding='utf-8')
        print(f"Decoy stats written to {args.output}")
    else:
        print(output)
    
    return 0


def cmd_decoy_quality(
    args: argparse.Namespace,
    loader: CurriculumDecoyLoader,
) -> int:
    """Assess decoy quality against design invariants."""
    slice_name = args.decoy_quality
    
    try:
        cmap = ConfusabilityMap(slice_name, args.config)
        quality = cmap.get_quality_assessment()
    except (KeyError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    if args.format == "json":
        output = json.dumps(quality, indent=2)
    else:
        output = _format_quality_assessment_markdown(quality)
    
    if args.output:
        Path(args.output).write_text(output, encoding='utf-8')
        print(f"Quality assessment written to {args.output}")
    else:
        print(output)
    
    return 1 if not quality["passed"] else 0


# =============================================================================
# CONFUSABILITY CONTRACT COMMANDS
# =============================================================================

def cmd_export_confusability(
    args: argparse.Namespace,
    loader: CurriculumDecoyLoader,
) -> int:
    """
    Export confusability contract to JSON file (deterministic, byte-stable).
    
    Uses the canonical schema defined in experiments/decoys/contract.py.
    No timestamp is included to ensure byte-stable exports.
    """
    slice_name = args.export_confusability
    
    try:
        contract = export_contract(slice_name, args.config)
    except (KeyError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        CONFUSABILITY_CONTRACT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = CONFUSABILITY_CONTRACT_DIR / f"{slice_name}.json"
    
    # Write deterministic JSON (sorted keys, no timestamp)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(contract.to_json(indent=2), encoding='utf-8')
    
    summary = contract.summary
    print(f"Confusability contract exported: {output_path}")
    print(f"  Schema version: {SCHEMA_VERSION}")
    print(f"  Targets: {summary.target_count}")
    print(f"  Near decoys: {summary.decoy_near_count} (avg conf: {summary.avg_confusability_near:.3f})")
    print(f"  Far decoys: {summary.decoy_far_count} (avg conf: {summary.avg_confusability_far:.3f})")
    print(f"  Bridges: {summary.bridge_count}")
    
    return 0


def verify_slice_confusability(
    slice_name: str,
    config_path: str,
    thresholds: Optional[Dict[str, float]] = None,
) -> VerificationResult:
    """
    Verify confusability invariants for a single slice.
    
    LEGACY/DECOY-AWARE CI POLICY:
    If a slice has no decoys, verification is SKIPPED and returns OK.
    This ensures legacy format (string lists) pass CI trivially.
    
    Invariants checked (only if decoys exist):
    1. Average near-decoy confusability >= near_avg_min (default 0.7)
    2. Average far-decoy confusability <= far_avg_max (default 0.5)
    3. No near-decoy has confusability < near_individual_min (default 0.4)
    
    Returns:
        VerificationResult with status OK, WARN, FAIL, or SKIPPED
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
    
    cmap = ConfusabilityMap(slice_name, config_path)
    report = cmap.generate_report()
    
    near_decoys = [f for f in report.formulas if f.role == 'decoy_near']
    far_decoys = [f for f in report.formulas if f.role == 'decoy_far']
    bridges = [f for f in report.formulas if f.role == 'bridge']
    
    has_decoys = len(near_decoys) + len(far_decoys) > 0
    
    near_avg = report.avg_near_confusability
    far_avg = report.avg_far_confusability
    
    # LEGACY-AWARE CI POLICY: Skip verification if no decoys
    if not has_decoys:
        return VerificationResult(
            slice_name=slice_name,
            status=VerificationStatus.SKIPPED,
            near_avg=0.0,
            far_avg=0.0,
            bridge_count=len(bridges),
            has_decoys=False,
            reasons=["No decoys present; confusability gate skipped"],
            explanations=[],
        )
    
    reasons = []
    explanations = []
    status = VerificationStatus.OK
    
    # Check 1: Near average
    near_avg_min = thresholds.get("near_avg_min", 0.7)
    if near_decoys and near_avg < near_avg_min:
        if near_avg < near_avg_min - 0.1:
            status = VerificationStatus.FAIL
            reasons.append(f"FAIL: near_avg ({near_avg:.3f}) < {near_avg_min} - 0.1")
        else:
            if status != VerificationStatus.FAIL:
                status = VerificationStatus.WARN
            reasons.append(f"WARN: near_avg ({near_avg:.3f}) < {near_avg_min}")
        
        explanations.append(FailureExplanation(
            threshold_name="near_avg_min",
            threshold_value=near_avg_min,
            actual_value=near_avg,
            rationale=(
                f"Average near-decoy confusability ({near_avg:.3f}) is below the "
                f"minimum threshold ({near_avg_min}). Near-decoys should be highly "
                f"confusable with targets to serve as effective distractors."
            ),
            narrative=NarrativeBlock(
                what_failed=(
                    f"Near-decoy average confusability check in slice '{slice_name}'"
                ),
                why_it_failed=(
                    f"Threshold: near_avg_min = {near_avg_min}. "
                    f"Actual value: {near_avg:.4f}. "
                    f"Expected range: [{near_avg_min}, 1.0]"
                ),
                how_to_examine=(
                    f"Inspect the {len(near_decoys)} near-decoys in this slice. "
                    f"Review their normalized forms and component scores. "
                    f"Compare syntactic and connective similarity to targets."
                ),
            ),
        ))
    
    # Check 2: Far average
    far_avg_max = thresholds.get("far_avg_max", 0.5)
    if far_decoys and far_avg > far_avg_max:
        if far_avg > far_avg_max + 0.1:
            status = VerificationStatus.FAIL
            reasons.append(f"FAIL: far_avg ({far_avg:.3f}) > {far_avg_max} + 0.1")
        else:
            if status != VerificationStatus.FAIL:
                status = VerificationStatus.WARN
            reasons.append(f"WARN: far_avg ({far_avg:.3f}) > {far_avg_max}")
        
        explanations.append(FailureExplanation(
            threshold_name="far_avg_max",
            threshold_value=far_avg_max,
            actual_value=far_avg,
            rationale=(
                f"Average far-decoy confusability ({far_avg:.3f}) exceeds the "
                f"maximum threshold ({far_avg_max}). Far-decoys should be clearly "
                f"distinguishable from targets."
            ),
            narrative=NarrativeBlock(
                what_failed=(
                    f"Far-decoy average confusability check in slice '{slice_name}'"
                ),
                why_it_failed=(
                    f"Threshold: far_avg_max = {far_avg_max}. "
                    f"Actual value: {far_avg:.4f}. "
                    f"Expected range: [0.0, {far_avg_max}]"
                ),
                how_to_examine=(
                    f"Inspect the {len(far_decoys)} far-decoys in this slice. "
                    f"Review their structural distance from targets. "
                    f"Compare atom sets and connective signatures."
                ),
            ),
        ))
    
    # Check 3: No near-decoy below individual minimum
    near_individual_min = thresholds.get("near_individual_min", 0.4)
    low_near_decoys = [f for f in near_decoys if f.confusability < near_individual_min]
    if low_near_decoys:
        status = VerificationStatus.FAIL
        for f in low_near_decoys:
            reasons.append(
                f"FAIL: near-decoy '{f.name}' has confusability {f.confusability:.3f} < {near_individual_min}"
            )
            explanations.append(FailureExplanation(
                threshold_name="near_individual_min",
                threshold_value=near_individual_min,
                actual_value=f.confusability,
                formula_name=f.name,
                formula_confusability=f.confusability,
                formula_difficulty=f.difficulty,
                rationale=(
                    f"Near-decoy '{f.name}' has confusability {f.confusability:.3f} which is "
                    f"below the minimum threshold ({near_individual_min}). This formula is "
                    f"too easily distinguished from targets to serve as a near-decoy."
                ),
                narrative=NarrativeBlock(
                    what_failed=(
                        f"Individual near-decoy confusability check for '{f.name}'"
                    ),
                    why_it_failed=(
                        f"Threshold: near_individual_min = {near_individual_min}. "
                        f"Actual confusability: {f.confusability:.4f}. "
                        f"Difficulty: {f.difficulty:.4f}. "
                        f"Expected range: [{near_individual_min}, 1.0]"
                    ),
                    how_to_examine=(
                        f"Inspect formula '{f.name}': {f.formula}. "
                        f"Review its normalized form and component breakdown. "
                        f"Compare structural similarity to target formulas in this slice."
                    ),
                ),
            ))
    
    return VerificationResult(
        slice_name=slice_name,
        status=status,
        near_avg=near_avg,
        far_avg=far_avg,
        bridge_count=len(bridges),
        has_decoys=True,
        reasons=reasons,
        explanations=explanations,
    )


def cmd_verify_confusability(
    args: argparse.Namespace,
    loader: CurriculumDecoyLoader,
) -> int:
    """
    Verify confusability invariants for a slice or all slices.
    
    With --explain flag, prints detailed failure explanations including:
    - Which thresholds failed
    - Which formulas caused failure (by name)
    - Their difficulty and confusability values
    - A rationale explaining the failure
    
    Exit codes:
        0 = all slices OK, WARN, or SKIPPED
        1 = any slice FAIL
    """
    # Get slice(s) to verify
    if args.verify_confusability == "all":
        slice_names = loader.list_uplift_slices()
    else:
        slice_names = [args.verify_confusability]
    
    results = []
    any_fail = False
    
    for slice_name in slice_names:
        try:
            result = verify_slice_confusability(slice_name, args.config)
            results.append(result)
            
            if result.status == VerificationStatus.FAIL:
                any_fail = True
        except (KeyError, ValueError) as e:
            print(f"Error verifying {slice_name}: {e}", file=sys.stderr)
            any_fail = True
    
    # Output results
    if args.format == "json":
        output = json.dumps([r.to_dict() for r in results], indent=2, sort_keys=True)
        if args.output:
            Path(args.output).write_text(output, encoding='utf-8')
        else:
            print(output)
    else:
        print("Confusability Verification Results:")
        print("=" * 70)
        
        for result in results:
            status_icon = {
                VerificationStatus.OK: "✓",
                VerificationStatus.WARN: "⚠",
                VerificationStatus.FAIL: "✗",
                VerificationStatus.SKIPPED: "○",
            }[result.status]
            
            print(f"\n{status_icon} {result.slice_name}: {result.status.value}")
            
            if result.status == VerificationStatus.SKIPPED:
                print("    No decoys present; confusability gate skipped")
            else:
                print(f"    near_avg={result.near_avg:.3f}, far_avg={result.far_avg:.3f}, bridges={result.bridge_count}")
            
            if result.reasons:
                for reason in result.reasons:
                    print(f"    → {reason}")
            
            # Print detailed explanations if --explain is set
            if getattr(args, 'explain', False) and result.explanations:
                print("\n    DETAILED EXPLANATIONS:")
                for i, exp in enumerate(result.explanations, 1):
                    print(f"\n    [{i}] Threshold: {exp.threshold_name}")
                    
                    # Print narrative blocks if available
                    if exp.narrative:
                        print(exp.narrative.format_text())
                    else:
                        # Fallback to simple format
                        print(f"        Expected: {exp.threshold_value:.4f}")
                        print(f"        Actual:   {exp.actual_value:.4f}")
                        if exp.formula_name:
                            print(f"        Formula:  {exp.formula_name}")
                            print(f"        Confusability: {exp.formula_confusability:.4f}")
                            print(f"        Difficulty:    {exp.formula_difficulty:.4f}")
                        print(f"        Rationale: {exp.rationale}")
        
        print("\n" + "=" * 70)
        
        ok_count = sum(1 for r in results if r.status == VerificationStatus.OK)
        warn_count = sum(1 for r in results if r.status == VerificationStatus.WARN)
        fail_count = sum(1 for r in results if r.status == VerificationStatus.FAIL)
        skip_count = sum(1 for r in results if r.status == VerificationStatus.SKIPPED)
        
        print(f"Summary: {ok_count} OK, {warn_count} WARN, {fail_count} FAIL, {skip_count} SKIPPED")
        
        if any_fail:
            print("\n❌ VERIFICATION FAILED - Fix issues before merge")
        else:
            print("\n✓ Verification passed")
    
    return 1 if any_fail else 0


def cmd_decoy_ci_summary(
    args: argparse.Namespace,
    loader: CurriculumDecoyLoader,
) -> int:
    """
    Print compact, greppable CI summary for all slices.
    
    Output format (one line per slice):
        slice_name: OK|WARN|FAIL|SKIPPED (near=X.XXX, far=X.XXX, bridges=N)
    
    No exit code change - use --verify-confusability for gating.
    """
    slice_names = loader.list_uplift_slices()
    
    if not slice_names:
        print("No uplift slices found.", file=sys.stderr)
        return 0
    
    for slice_name in slice_names:
        try:
            result = verify_slice_confusability(slice_name, args.config)
            if result.status == VerificationStatus.SKIPPED:
                print(f"{slice_name}: SKIPPED (no decoys)")
            else:
                print(
                    f"{slice_name}: {result.status.value} "
                    f"(near={result.near_avg:.3f}, far={result.far_avg:.3f}, bridges={result.bridge_count})"
                )
        except (KeyError, ValueError) as e:
            print(f"{slice_name}: ERROR ({e})")
    
    return 0


# =============================================================================
# FAMILY RISK COMMANDS
# =============================================================================

def cmd_family_risk_snapshot(
    args: argparse.Namespace,
    loader: CurriculumDecoyLoader,
) -> int:
    """
    Generate family risk snapshot for a slice.
    
    Output includes:
    - Risk level per family (LOW/MEDIUM/HIGH)
    - Risk counts
    - Summary notes (neutral language)
    """
    slice_name = args.family_risk_snapshot
    
    try:
        contract = export_contract(slice_name, args.config)
        contract_dict = contract.to_dict()
        
        snapshot = build_family_risk_snapshot(contract_dict)
        
        # Pretty print JSON
        output = json.dumps(snapshot, indent=2, sort_keys=True)
        
        if args.output:
            Path(args.output).write_text(output, encoding='utf-8')
            print(f"Risk snapshot written to: {args.output}")
        else:
            print(output)
        
        # Summary
        print(f"\n--- Summary ---", file=sys.stderr)
        print(f"Slice: {slice_name}", file=sys.stderr)
        print(f"Total families: {snapshot['total_family_count']}", file=sys.stderr)
        print(f"HIGH: {snapshot['high_risk_family_count']}", file=sys.stderr)
        print(f"MEDIUM: {snapshot['medium_risk_family_count']}", file=sys.stderr)
        print(f"LOW: {snapshot['low_risk_family_count']}", file=sys.stderr)
        
        return 0
    except (KeyError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_family_drift(
    args: argparse.Namespace,
    loader: CurriculumDecoyLoader,
) -> int:
    """
    Compare two risk snapshots for drift analysis.
    """
    old_path, new_path = args.family_drift
    
    try:
        old_snapshot = json.loads(Path(old_path).read_text(encoding='utf-8'))
        new_snapshot = json.loads(Path(new_path).read_text(encoding='utf-8'))
        
        drift = compare_family_risk(old_snapshot, new_snapshot)
        
        # Pretty print JSON
        output = json.dumps(drift, indent=2, sort_keys=True)
        
        if args.output:
            Path(args.output).write_text(output, encoding='utf-8')
            print(f"Drift analysis written to: {args.output}")
        else:
            print(output)
        
        # Summary
        print(f"\n--- Drift Summary ---", file=sys.stderr)
        print(f"Trend: {drift['net_risk_trend']}", file=sys.stderr)
        print(f"New families: {len(drift['families_new'])}", file=sys.stderr)
        print(f"Removed families: {len(drift['families_removed'])}", file=sys.stderr)
        print(f"Increased risk: {len(drift['families_increased_risk'])}", file=sys.stderr)
        print(f"Decreased risk: {len(drift['families_decreased_risk'])}", file=sys.stderr)
        
        return 0
    except FileNotFoundError as e:
        print(f"Error: Snapshot file not found: {e}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in snapshot: {e}", file=sys.stderr)
        return 1


def cmd_governance_signal(
    args: argparse.Namespace,
    loader: CurriculumDecoyLoader,
) -> int:
    """
    Generate governance health signal for a slice.
    
    Output:
    - confusability_ok: Boolean
    - high_risk_family_count
    - status: OK | ATTENTION | HOT
    - summary: Neutral description
    """
    slice_name = args.governance_signal
    
    try:
        contract = export_contract(slice_name, args.config)
        contract_dict = contract.to_dict()
        
        snapshot = build_family_risk_snapshot(contract_dict)
        signal = summarize_confusability_for_global_health(snapshot)
        
        # Pretty print JSON
        output = json.dumps(signal, indent=2, sort_keys=True)
        
        if args.output:
            Path(args.output).write_text(output, encoding='utf-8')
            print(f"Governance signal written to: {args.output}")
        else:
            print(output)
        
        # Human-readable status
        print(f"\n--- Governance Status ---", file=sys.stderr)
        print(f"Slice: {slice_name}", file=sys.stderr)
        print(f"Status: {signal['status']}", file=sys.stderr)
        print(f"OK: {signal['confusability_ok']}", file=sys.stderr)
        print(f"HIGH-risk families: {signal['high_risk_family_count']}", file=sys.stderr)
        
        return 0
    except (KeyError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


# =============================================================================
# MARKDOWN FORMATTERS
# =============================================================================

def _format_confusability_map_markdown(report) -> str:
    """Format confusability map as Markdown."""
    lines = [
        f"# Confusability Map: {report.slice_name}",
        "",
        "## Statistics",
        "",
        f"- **Avg Near Confusability**: {report.avg_near_confusability:.4f}",
        f"- **Avg Far Confusability**: {report.avg_far_confusability:.4f}",
        f"- **Near-Far Gap**: {report.near_far_gap:.4f}",
        f"- **Coverage Score**: {report.coverage_score:.4f}",
        "",
        "## Formula Map",
        "",
        "| Name | Role | Difficulty | Confusability | Syntactic | Connective | Atoms | Chain |",
        "|------|------|------------|---------------|-----------|------------|-------|-------|",
    ]
    
    role_order = {'target': 0, 'decoy_near': 1, 'decoy_far': 2, 'bridge': 3}
    sorted_formulas = sorted(
        report.formulas,
        key=lambda f: (role_order.get(f.role, 4), -f.confusability)
    )
    
    for f in sorted_formulas:
        comp = f.components
        lines.append(
            f"| {f.name} | {f.role} | {f.difficulty:.3f} | **{f.confusability:.3f}** | "
            f"{comp.get('syntactic', 0):.3f} | {comp.get('connective', 0):.3f} | "
            f"{comp.get('atom_similarity', 0):.3f} | {comp.get('chain_alignment', 0):.3f} |"
        )
    
    return "\n".join(lines)


def _format_decoy_stats_markdown(stats: Dict[str, Any]) -> str:
    """Format decoy statistics as Markdown."""
    lines = [
        f"# Decoy Statistics: {stats['slice_name']}",
        "",
        "## Distribution Summary",
        "",
        f"| Category | Count | Avg Diff | Avg Conf | Min Diff | Max Diff | Min Conf | Max Conf |",
        f"|----------|-------|----------|----------|----------|----------|----------|----------|",
    ]
    
    for category in ['near_decoys', 'far_decoys', 'bridges']:
        s = stats.get(category, {})
        if s.get('count', 0) > 0:
            lines.append(
                f"| {category.replace('_', ' ').title()} | {s['count']} | "
                f"{s.get('avg_difficulty', 0):.3f} | {s.get('avg_confusability', 0):.3f} | "
                f"{s.get('min_difficulty', 0):.3f} | {s.get('max_difficulty', 0):.3f} | "
                f"{s.get('min_confusability', 0):.3f} | {s.get('max_confusability', 0):.3f} |"
            )
    
    summary = stats.get('summary', {})
    lines.extend([
        "",
        "## Summary Metrics",
        "",
        f"- **Near-Far Gap**: {summary.get('near_far_gap', 0):.4f}",
        f"- **Coverage Score**: {summary.get('coverage_score', 0):.4f}",
        f"- **Target Count**: {stats.get('targets', {}).get('count', 0)}",
    ])
    
    return "\n".join(lines)


def _format_quality_assessment_markdown(quality: Dict[str, Any]) -> str:
    """Format quality assessment as Markdown."""
    passed_str = "✓ PASSED" if quality["passed"] else "✗ FAILED"
    
    lines = [
        f"# Decoy Quality Assessment: {quality['slice_name']}",
        "",
        f"## Result: {passed_str}",
        "",
        f"**Quality Score**: {quality['quality_score']:.3f} / 1.000",
        "",
        "## Metrics",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
    ]
    
    metrics = quality.get('metrics', {})
    for key, value in metrics.items():
        lines.append(f"| {key.replace('_', ' ').title()} | {value:.4f} |")
    
    if quality.get('issues'):
        lines.extend(["", "## Issues (MUST FIX)", ""])
        for issue in quality['issues']:
            lines.append(f"- ❌ {issue}")
    
    if quality.get('warnings'):
        lines.extend(["", "## Warnings", ""])
        for warning in quality['warnings']:
            lines.append(f"- ⚠ {warning}")
    
    if not quality.get('issues') and not quality.get('warnings'):
        lines.extend(["", "## Health Check", "", "✓ No issues or warnings detected."])
    
    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point."""
    
    parser = argparse.ArgumentParser(
        description="Curriculum Diagnostics CLI for Phase II",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Reports
  %(prog)s --decoy-report                    Generate Markdown report
  %(prog)s --decoy-map <slice>               Generate confusability map
  %(prog)s --decoy-stats <slice>             Get decoy statistics
  %(prog)s --decoy-quality <slice>           Assess quality invariants

  # Confusability Contracts
  %(prog)s --export-confusability <slice>    Export JSON contract (deterministic)
  %(prog)s --verify-confusability <slice>    Verify invariants (CI gate)
  %(prog)s --verify-confusability all        Verify all slices
  %(prog)s --verify-confusability all --explain  With detailed explanations
  %(prog)s --decoy-ci-summary                Compact greppable summary

  # Checks
  %(prog)s --check-monotonicity              Verify near > far difficulty
  %(prog)s --check-collisions                Check for hash collisions
  %(prog)s --summary                         Brief summary of all slices
        """,
    )
    
    parser.add_argument(
        "--config",
        default="config/curriculum_uplift_phase2.yaml",
        help="Path to curriculum YAML file",
    )
    
    # Commands (mutually exclusive)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--decoy-report", action="store_true",
                       help="Generate decoy difficulty report")
    group.add_argument("--check-monotonicity", action="store_true",
                       help="Check near > far difficulty")
    group.add_argument("--check-collisions", action="store_true",
                       help="Check for hash collisions")
    group.add_argument("--summary", action="store_true",
                       help="Brief summary of all slices")
    group.add_argument("--decoy-map", metavar="SLICE",
                       help="Generate confusability map")
    group.add_argument("--decoy-stats", metavar="SLICE",
                       help="Get decoy statistics")
    group.add_argument("--decoy-quality", metavar="SLICE",
                       help="Assess quality invariants")
    group.add_argument("--export-confusability", metavar="SLICE",
                       help="Export confusability contract to JSON (deterministic)")
    group.add_argument("--verify-confusability", metavar="SLICE",
                       help="Verify confusability invariants (use 'all' for all slices)")
    group.add_argument("--decoy-ci-summary", action="store_true",
                       help="Compact greppable CI summary")
    group.add_argument("--family-risk-snapshot", metavar="SLICE",
                       help="Generate family risk snapshot")
    group.add_argument("--family-drift", nargs=2, metavar=("OLD_SNAPSHOT", "NEW_SNAPSHOT"),
                       help="Compare two risk snapshots for drift")
    group.add_argument("--governance-signal", metavar="SLICE",
                       help="Generate governance health signal")
    
    # Options
    parser.add_argument("--slice", help="Slice for --decoy-report")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown",
                       help="Output format")
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument("--explain", action="store_true",
                       help="Show detailed explanations for failures (with --verify-confusability)")
    
    args = parser.parse_args(argv)
    
    # Load curriculum
    try:
        loader = CurriculumDecoyLoader(args.config)
    except FileNotFoundError:
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        return 1
    
    # Dispatch command
    if args.decoy_report:
        return cmd_decoy_report(args, loader)
    elif args.check_monotonicity:
        return cmd_check_monotonicity(args, loader)
    elif args.check_collisions:
        return cmd_check_collisions(args, loader)
    elif args.summary:
        return cmd_summary(args, loader)
    elif args.decoy_map:
        return cmd_decoy_map(args, loader)
    elif args.decoy_stats:
        return cmd_decoy_stats(args, loader)
    elif args.decoy_quality:
        return cmd_decoy_quality(args, loader)
    elif args.export_confusability:
        return cmd_export_confusability(args, loader)
    elif args.verify_confusability:
        return cmd_verify_confusability(args, loader)
    elif args.decoy_ci_summary:
        return cmd_decoy_ci_summary(args, loader)
    elif args.family_risk_snapshot:
        return cmd_family_risk_snapshot(args, loader)
    elif args.family_drift:
        return cmd_family_drift(args, loader)
    elif args.governance_signal:
        return cmd_governance_signal(args, loader)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
