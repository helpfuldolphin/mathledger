#!/usr/bin/env python3
"""
Mock Oracle CLI for Test Engineers.

A command-line interface for querying the deterministic mock verification
oracle. Useful for test planning, debugging, and generating test data.

ABSOLUTE SAFEGUARD: This CLI is for tests/dev ONLY — never for production.

Usage:
    python -m backend.verification.mock_oracle_cli \\
        --profile goal_hit \\
        --formula "(p->q)" \\
        --count 20 \\
        --json

    # CI mode for contract verification
    python -m backend.verification.mock_oracle_cli --ci

Exit Codes:
    0: Success (or --ci passes all checks)
    1: Invalid arguments or configuration error
    2: Runtime error during verification
    3: Contract violation detected (--ci mode only)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

# Enable mock oracle for CLI usage
os.environ["MATHLEDGER_ALLOW_MOCK_ORACLE"] = "1"

from backend.verification.mock_config import (
    ANALYTICS_SCHEMA_VERSION,
    DRIFT_STATUS_BROKEN,
    DRIFT_STATUS_DRIFTED,
    DRIFT_STATUS_IN_CONTRACT,
    MOCK_ORACLE_CONTRACT_VERSION,
    PROFILE_CONTRACTS,
    SCENARIOS,
    SLICE_PROFILES,
    MockOracleConfig,
    ProfileCoverageMap,
    Scenario,
    detect_scenario_drift,
    export_mock_oracle_contract,
    get_scenario,
    list_scenarios,
    summarize_scenario_results,
    verify_profile_contracts,
    verify_negative_control_result,
)
from backend.verification.mock_oracle import MockVerifiableOracle
from backend.verification.mock_exceptions import MockOracleCrashError


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        prog="mock_oracle_cli",
        description="Mock Verification Oracle CLI for Test Engineers",
        epilog="SAFEGUARD: For tests/dev only — never for production.",
    )
    
    parser.add_argument(
        "--profile",
        type=str,
        default="default",
        choices=list(SLICE_PROFILES.keys()),
        help="Slice profile to use (default: default)",
    )
    
    parser.add_argument(
        "--formula",
        type=str,
        help="Single formula to verify",
    )
    
    parser.add_argument(
        "--formulas-file",
        type=str,
        help="File containing formulas (one per line)",
    )
    
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of times to verify each formula (default: 1)",
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format",
    )
    
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Show coverage map for profile and exit",
    )
    
    parser.add_argument(
        "--negative-control",
        action="store_true",
        help="Enable negative control mode (all results return verified=False)",
    )
    
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=50,
        help="Timeout latency in milliseconds (default: 50)",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for deterministic behavior (default: 0)",
    )
    
    parser.add_argument(
        "--enable-crashes",
        action="store_true",
        help="Enable crash exceptions (default: disabled)",
    )
    
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show summary statistics after verification",
    )
    
    parser.add_argument(
        "--all-profiles",
        action="store_true",
        help="Show coverage for all profiles and exit",
    )
    
    parser.add_argument(
        "--ci",
        action="store_true",
        help="CI mode: verify all contracts and exit with code 3 on violation",
    )
    
    parser.add_argument(
        "--ci-nc-samples",
        type=int,
        default=200,
        help="Number of samples to test in negative control CI check (default: 200)",
    )
    
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show mock oracle contract version and exit",
    )
    
    # Scenario runner arguments
    parser.add_argument(
        "--scenario",
        type=str,
        help="Run a named scenario (e.g., 'default_sanity', 'goal_hit_stress')",
    )
    
    parser.add_argument(
        "--samples",
        type=int,
        help="Number of samples to run for scenario (overrides scenario default)",
    )
    
    parser.add_argument(
        "--assert-contract",
        action="store_true",
        help="Assert that observed distribution matches contract (exit 3 on violation)",
    )
    
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="List all available scenarios and exit",
    )
    
    parser.add_argument(
        "--filter-tags",
        type=str,
        help="Filter scenarios by tags (comma-separated, e.g., 'sanity,quick')",
    )
    
    # Contract export
    parser.add_argument(
        "--export-contract",
        action="store_true",
        help="Export full contract snapshot as JSON (for Evidence Packs)",
    )
    
    # Drift check (CI guard)
    parser.add_argument(
        "--drift-check",
        type=str,
        metavar="SCENARIO",
        help="Run scenario drift check for CI (exits non-zero if drift detected)",
    )
    
    parser.add_argument(
        "--drift-samples",
        type=int,
        default=500,
        help="Number of samples for drift check (default: 500)",
    )
    
    parser.add_argument(
        "--drift-seed",
        type=int,
        default=42,
        help="Fixed seed for reproducible drift check (default: 42)",
    )
    
    parser.add_argument(
        "--warning-threshold",
        type=float,
        default=3.0,
        help="Drift warning threshold in percentage points (default: 3.0)",
    )
    
    parser.add_argument(
        "--broken-threshold",
        type=float,
        default=10.0,
        help="Drift broken threshold in percentage points (default: 10.0)",
    )
    
    return parser


def format_result_human(result: Dict[str, Any], formula: str) -> str:
    """Format a single result for human-readable output."""
    lines = [
        f"Formula: {formula}",
        f"  Bucket: {result['bucket']}",
        f"  Verified: {result['verified']}",
        f"  Abstained: {result['abstained']}",
        f"  Timed Out: {result['timed_out']}",
        f"  Crashed: {result['crashed']}",
        f"  Reason: {result['reason']}",
        f"  Latency: {result['latency_ms']}ms",
        f"  Hash (mod 100): {result['hash_int'] % 100}",
    ]
    return "\n".join(lines)


def format_coverage_human(coverage: ProfileCoverageMap) -> str:
    """Format coverage map for human-readable output."""
    lines = [
        f"Profile: {coverage.profile_name}",
        f"  Verified:  {coverage.verified_pct:5.1f}%",
        f"  Failed:    {coverage.failed_pct:5.1f}%",
        f"  Abstain:   {coverage.abstain_pct:5.1f}%",
        f"  Timeout:   {coverage.timeout_pct:5.1f}%",
        f"  Error:     {coverage.error_pct:5.1f}%",
        f"  Crash:     {coverage.crash_pct:5.1f}%",
    ]
    return "\n".join(lines)


def run_verification(
    oracle: MockVerifiableOracle,
    formulas: List[str],
    count: int,
    json_output: bool,
) -> List[Dict[str, Any]]:
    """
    Run verification on formulas.
    
    Args:
        oracle: MockVerifiableOracle instance.
        formulas: List of formulas to verify.
        count: Number of times to verify each formula.
        json_output: If True, collect results for JSON output.
        
    Returns:
        List of result dictionaries.
    """
    results = []
    
    for formula in formulas:
        for i in range(count):
            try:
                result = oracle.verify(formula)
                result_dict = {
                    "formula": formula,
                    "iteration": i + 1,
                    "verified": result.verified,
                    "abstained": result.abstained,
                    "timed_out": result.timed_out,
                    "crashed": result.crashed,
                    "reason": result.reason,
                    "latency_ms": result.latency_ms,
                    "bucket": result.bucket,
                    "hash_int": result.hash_int,
                }
                results.append(result_dict)
                
                if not json_output:
                    if count > 1:
                        print(f"[{i+1}/{count}] {format_result_human(result_dict, formula)}")
                    else:
                        print(format_result_human(result_dict, formula))
                    print()
                    
            except MockOracleCrashError as e:
                result_dict = {
                    "formula": formula,
                    "iteration": i + 1,
                    "verified": False,
                    "abstained": False,
                    "timed_out": False,
                    "crashed": True,
                    "reason": "mock-crash-exception",
                    "latency_ms": 0,
                    "bucket": "crash",
                    "hash_int": e.hash_int,
                    "error": str(e),
                }
                results.append(result_dict)
                
                if not json_output:
                    print(f"CRASH: {formula}")
                    print(f"  Error: {e}")
                    print()
    
    return results


def run_ci_checks(nc_samples: int, json_output: bool) -> Tuple[bool, Dict[str, Any]]:
    """
    Run CI contract verification checks.
    
    Verifies:
    1. Profile contracts match SLICE_PROFILES
    2. Negative control contract holds for nc_samples inputs
    3. Determinism: same input → same output
    
    Args:
        nc_samples: Number of samples to test for negative control.
        json_output: If True, format output for JSON.
        
    Returns:
        Tuple of (all_passed, results_dict).
    """
    results = {
        "contract_version": MOCK_ORACLE_CONTRACT_VERSION,
        "checks": {},
        "passed": True,
    }
    
    # Check 1: Profile contracts
    contract_ok, contract_errors = verify_profile_contracts()
    results["checks"]["profile_contracts"] = {
        "passed": contract_ok,
        "errors": contract_errors,
    }
    if not contract_ok:
        results["passed"] = False
    
    if not json_output and not contract_ok:
        print("❌ PROFILE CONTRACT VIOLATIONS:")
        for err in contract_errors:
            print(f"   {err}")
    elif not json_output:
        print("✓ Profile contracts verified")
    
    # Check 2: Negative control contract
    nc_config = MockOracleConfig(negative_control=True)
    nc_oracle = MockVerifiableOracle(nc_config)
    nc_errors = []
    
    for i in range(nc_samples):
        formula = f"nc_test_formula_{i}_{'p' * (i % 10)}"
        result = nc_oracle.verify(formula)
        is_valid, violations = verify_negative_control_result(result)
        if not is_valid:
            nc_errors.append({
                "formula": formula,
                "violations": violations,
            })
    
    nc_ok = len(nc_errors) == 0
    results["checks"]["negative_control"] = {
        "passed": nc_ok,
        "samples_tested": nc_samples,
        "violations": len(nc_errors),
        "errors": nc_errors[:5] if nc_errors else [],  # Limit to first 5
    }
    if not nc_ok:
        results["passed"] = False
    
    if not json_output and not nc_ok:
        print(f"❌ NEGATIVE CONTROL VIOLATIONS: {len(nc_errors)} of {nc_samples}")
        for err in nc_errors[:3]:
            print(f"   Formula: {err['formula']}")
            for v in err['violations']:
                print(f"     - {v}")
    elif not json_output:
        print(f"✓ Negative control verified ({nc_samples} samples)")
    
    # Check 3: Stats suppression in NC mode
    stats_ok = nc_oracle.stats["total"] == 0
    results["checks"]["nc_stats_suppression"] = {
        "passed": stats_ok,
        "stats_total": nc_oracle.stats["total"],
    }
    if not stats_ok:
        results["passed"] = False
    
    if not json_output and not stats_ok:
        print(f"❌ NC STATS NOT SUPPRESSED: total={nc_oracle.stats['total']}")
    elif not json_output:
        print("✓ Negative control stats suppression verified")
    
    # Check 4: Determinism
    det_config = MockOracleConfig(slice_profile="default", seed=42)
    det_oracle1 = MockVerifiableOracle(det_config)
    det_oracle2 = MockVerifiableOracle(det_config)
    
    det_errors = []
    for i in range(50):
        formula = f"det_test_{i}"
        r1 = det_oracle1.verify(formula)
        r2 = det_oracle2.verify(formula)
        
        if (r1.verified != r2.verified or r1.bucket != r2.bucket or 
            r1.latency_ms != r2.latency_ms or r1.reason != r2.reason):
            det_errors.append({
                "formula": formula,
                "result1": {"verified": r1.verified, "bucket": r1.bucket},
                "result2": {"verified": r2.verified, "bucket": r2.bucket},
            })
    
    det_ok = len(det_errors) == 0
    results["checks"]["determinism"] = {
        "passed": det_ok,
        "samples_tested": 50,
        "errors": det_errors[:3] if det_errors else [],
    }
    if not det_ok:
        results["passed"] = False
    
    if not json_output and not det_ok:
        print(f"❌ DETERMINISM VIOLATIONS: {len(det_errors)}")
    elif not json_output:
        print("✓ Determinism verified (50 samples)")
    
    # Summary
    if not json_output:
        print()
        if results["passed"]:
            print(f"✓ ALL CI CHECKS PASSED (contract v{MOCK_ORACLE_CONTRACT_VERSION})")
        else:
            print(f"❌ CI CHECKS FAILED (contract v{MOCK_ORACLE_CONTRACT_VERSION})")
    
    return results["passed"], results


def run_scenario(
    scenario: Scenario,
    samples: int,
    json_output: bool,
    assert_contract: bool,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Run a scenario and compute observed distribution.
    
    Args:
        scenario: The scenario to run.
        samples: Number of samples to run.
        json_output: If True, format output for JSON.
        assert_contract: If True, check observed vs expected distribution.
        
    Returns:
        Tuple of (contract_ok, results_dict).
    """
    # Create oracle with scenario config
    config = MockOracleConfig(
        slice_profile=scenario.profile,
        negative_control=scenario.negative_control,
        seed=42,  # Deterministic for reproducibility
    )
    oracle = MockVerifiableOracle(config)
    
    # Generate test formulas
    formulas = [f"scenario_sample_{i}" for i in range(samples)]
    
    # Run verification
    bucket_counts = {
        "verified": 0,
        "failed": 0,
        "abstain": 0,
        "timeout": 0,
        "error": 0,
        "crash": 0,
        "negative_control": 0,
    }
    
    for formula in formulas:
        result = oracle.verify(formula)
        bucket = result.bucket
        if bucket in bucket_counts:
            bucket_counts[bucket] += 1
    
    # Compute observed percentages
    observed = {
        bucket: round(count / samples * 100, 2)
        for bucket, count in bucket_counts.items()
    }
    
    # Get expected percentages
    if scenario.negative_control:
        expected = {
            "verified": 0.0,
            "failed": 0.0,
            "abstain": 0.0,
            "timeout": 0.0,
            "error": 0.0,
            "crash": 0.0,
            "negative_control": 100.0,
        }
    else:
        expected = PROFILE_CONTRACTS.get(scenario.profile, {})
        # Add negative_control key for completeness
        expected = {**expected, "negative_control": 0.0}
    
    # Compute differences
    differences = {}
    for bucket in observed:
        exp_val = expected.get(bucket, 0.0)
        obs_val = observed[bucket]
        differences[bucket] = round(obs_val - exp_val, 2)
    
    # Check contract (using larger epsilon for sampling variance)
    # With N samples, expect variance ~ sqrt(p*(1-p)/N) * 100%
    # For N=100, max expected deviation ~5% for p=0.5
    contract_epsilon = max(10.0, 100.0 / (samples ** 0.5))  # Scale with samples
    contract_ok = True
    contract_violations = []
    
    if assert_contract:
        for bucket, exp_val in expected.items():
            obs_val = observed.get(bucket, 0.0)
            if abs(obs_val - exp_val) > contract_epsilon:
                contract_ok = False
                contract_violations.append({
                    "bucket": bucket,
                    "expected": exp_val,
                    "observed": obs_val,
                    "difference": differences.get(bucket, 0.0),
                    "epsilon": contract_epsilon,
                })
    
    results = {
        "scenario": scenario.to_dict(),
        "samples": samples,
        "observed": observed,
        "expected": expected,
        "differences": differences,
        "contract_check": {
            "enabled": assert_contract,
            "epsilon": contract_epsilon,
            "passed": contract_ok,
            "violations": contract_violations,
        } if assert_contract else None,
    }
    
    if not json_output:
        print(f"Scenario: {scenario.name}")
        print(f"Profile: {scenario.profile}")
        print(f"Description: {scenario.description}")
        print(f"Tags: {sorted(scenario.tags)}")
        print(f"Negative Control: {scenario.negative_control}")
        print(f"Samples: {samples}")
        print()
        print("Bucket Distribution:")
        print("-" * 60)
        print(f"{'Bucket':<18} {'Expected':>10} {'Observed':>10} {'Diff':>10}")
        print("-" * 60)
        for bucket in ["verified", "failed", "abstain", "timeout", "error", "crash", "negative_control"]:
            exp = expected.get(bucket, 0.0)
            obs = observed.get(bucket, 0.0)
            diff = differences.get(bucket, 0.0)
            diff_str = f"+{diff:.2f}" if diff > 0 else f"{diff:.2f}"
            print(f"{bucket:<18} {exp:>10.2f}% {obs:>10.2f}% {diff_str:>10}")
        print("-" * 60)
        
        if assert_contract:
            print()
            if contract_ok:
                print(f"[PASS] CONTRACT CHECK PASSED (epsilon={contract_epsilon:.1f}%)")
            else:
                print(f"[FAIL] CONTRACT CHECK FAILED (epsilon={contract_epsilon:.1f}%)")
                for v in contract_violations:
                    print(f"   {v['bucket']}: expected {v['expected']:.2f}%, got {v['observed']:.2f}%")
    
    return contract_ok, results


def format_scenario_list(scenarios: List[Scenario], json_output: bool) -> None:
    """Format and print scenario list."""
    if json_output:
        output = {
            "scenarios": [s.to_dict() for s in scenarios],
            "count": len(scenarios),
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"Available Scenarios ({len(scenarios)}):")
        print("=" * 70)
        for s in scenarios:
            nc_marker = " [NC]" if s.negative_control else ""
            print(f"\n{s.name}{nc_marker}")
            print(f"  Profile: {s.profile}")
            print(f"  Tags: {sorted(s.tags)}")
            print(f"  Description: {s.description}")
            print(f"  Default Samples: {s.samples_default}")


def run_drift_check(
    scenario_name: str,
    samples: int,
    seed: int,
    warning_threshold: float,
    broken_threshold: float,
    json_output: bool,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Run a scenario drift check for CI.
    
    This function runs a scenario with a fixed seed and sample count,
    then compares the results against the contract to detect drift.
    
    Args:
        scenario_name: Name of the scenario to run.
        samples: Number of samples to run.
        seed: Fixed seed for reproducibility.
        warning_threshold: Percentage points threshold for DRIFTED status.
        broken_threshold: Percentage points threshold for BROKEN status.
        json_output: If True, format output for JSON.
        
    Returns:
        Tuple of (passed, results_dict).
        passed is True if status is IN_CONTRACT.
    """
    try:
        scenario = get_scenario(scenario_name)
    except KeyError as e:
        return False, {
            "error": str(e),
            "status": "ERROR",
            "passed": False,
        }
    
    # Create oracle with fixed seed for reproducibility
    config = MockOracleConfig(
        slice_profile=scenario.profile,
        negative_control=scenario.negative_control,
        seed=seed,
    )
    oracle = MockVerifiableOracle(config)
    
    # Generate test formulas deterministically
    formulas = [f"drift_check_{seed}_{i}" for i in range(samples)]
    
    # Run verification
    outcomes = [oracle.verify(f) for f in formulas]
    
    # Compute analytics summary
    summary = summarize_scenario_results(scenario_name, outcomes)
    
    # Get contract and detect drift
    contract = export_mock_oracle_contract()
    drift_report = detect_scenario_drift(
        contract, summary, warning_threshold, broken_threshold
    )
    
    # Combine results
    results = {
        "analytics": summary,
        "drift_report": drift_report,
        "config": {
            "scenario": scenario_name,
            "samples": samples,
            "seed": seed,
            "warning_threshold": warning_threshold,
            "broken_threshold": broken_threshold,
        },
        "passed": drift_report["status"] == DRIFT_STATUS_IN_CONTRACT,
    }
    
    if not json_output:
        print(f"Drift Check: {scenario_name}")
        print(f"Samples: {samples} | Seed: {seed}")
        print("=" * 60)
        print()
        
        # Show distribution comparison
        print("Bucket Distribution:")
        print("-" * 60)
        print(f"{'Bucket':<18} {'Expected':>10} {'Observed':>10} {'Delta':>10}")
        print("-" * 60)
        
        emp = summary["empirical_distribution"]
        exp = summary["expected_distribution"]
        deltas = summary["deltas"]
        
        for bucket in ["verified", "failed", "abstain", "timeout", "error", "crash", "negative_control"]:
            e_val = exp.get(bucket, 0.0)
            o_val = emp.get(bucket, 0.0)
            d_val = deltas.get(bucket, 0.0)
            d_str = f"+{d_val:.2f}" if d_val > 0 else f"{d_val:.2f}"
            print(f"{bucket:<18} {e_val:>10.2f}% {o_val:>10.2f}% {d_str:>10}")
        print("-" * 60)
        print()
        
        # Show drift status
        status = drift_report["status"]
        max_drift = drift_report["max_drift"]
        
        if status == DRIFT_STATUS_IN_CONTRACT:
            print(f"[PASS] Status: {status}")
            print(f"       Max drift: {max_drift:.2f}% (within thresholds)")
        elif status == DRIFT_STATUS_DRIFTED:
            print(f"[WARN] Status: {status}")
            print(f"       Max drift: {max_drift:.2f}%")
            print()
            print("Drift signals:")
            for signal in drift_report["drift_signals"]:
                print(f"  - {signal['bucket']}: {signal['observed']:.2f}% (expected {signal['expected']:.2f}%)")
        else:  # BROKEN
            print(f"[FAIL] Status: {status}")
            print(f"       Max drift: {max_drift:.2f}%")
            print()
            print("Critical drift signals:")
            for signal in drift_report["drift_signals"]:
                sev = signal["severity"]
                print(f"  - [{sev}] {signal['bucket']}: {signal['observed']:.2f}% (expected {signal['expected']:.2f}%)")
        
        print()
        print(f"Recommendation: {drift_report['recommended_action']}")
    
    return results["passed"], results


def show_summary(oracle: MockVerifiableOracle, json_output: bool) -> Dict[str, Any]:
    """Show summary statistics."""
    stats = oracle.stats
    total = stats["total"]
    
    summary = {
        "total": total,
        "verified": stats["verified"],
        "failed": stats["failed"],
        "abstain": stats["abstain"],
        "timeout": stats["timeout"],
        "error": stats["error"],
        "crash": stats["crash"],
    }
    
    if total > 0:
        summary["percentages"] = {
            "verified": round(stats["verified"] / total * 100, 1),
            "failed": round(stats["failed"] / total * 100, 1),
            "abstain": round(stats["abstain"] / total * 100, 1),
            "timeout": round(stats["timeout"] / total * 100, 1),
            "error": round(stats["error"] / total * 100, 1),
            "crash": round(stats["crash"] / total * 100, 1),
        }
    
    if not json_output:
        print("=" * 50)
        print("SUMMARY")
        print("=" * 50)
        print(f"Total verifications: {total}")
        if total > 0:
            print(f"  Verified:  {stats['verified']:4d} ({summary['percentages']['verified']:5.1f}%)")
            print(f"  Failed:    {stats['failed']:4d} ({summary['percentages']['failed']:5.1f}%)")
            print(f"  Abstain:   {stats['abstain']:4d} ({summary['percentages']['abstain']:5.1f}%)")
            print(f"  Timeout:   {stats['timeout']:4d} ({summary['percentages']['timeout']:5.1f}%)")
            print(f"  Error:     {stats['error']:4d} ({summary['percentages']['error']:5.1f}%)")
            print(f"  Crash:     {stats['crash']:4d} ({summary['percentages']['crash']:5.1f}%)")
    
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)
    
    # Handle --version
    if args.version:
        if args.json:
            print(json.dumps({"contract_version": MOCK_ORACLE_CONTRACT_VERSION}))
        else:
            print(f"Mock Oracle Contract Version: {MOCK_ORACLE_CONTRACT_VERSION}")
        return 0
    
    # Handle --export-contract
    if args.export_contract:
        contract = export_mock_oracle_contract()
        print(json.dumps(contract, indent=2))
        return 0
    
    # Handle --list-scenarios
    if args.list_scenarios:
        filter_tags = None
        if args.filter_tags:
            filter_tags = set(t.strip() for t in args.filter_tags.split(","))
        scenarios = list_scenarios(filter_tags)
        format_scenario_list(scenarios, args.json)
        return 0
    
    # Handle --drift-check (CI guard)
    if args.drift_check:
        passed, results = run_drift_check(
            scenario_name=args.drift_check,
            samples=args.drift_samples,
            seed=args.drift_seed,
            warning_threshold=args.warning_threshold,
            broken_threshold=args.broken_threshold,
            json_output=args.json,
        )
        
        if args.json:
            print(json.dumps(results, indent=2))
        
        if not passed:
            return 3  # Contract violation / drift detected
        return 0
    
    # Handle --scenario
    if args.scenario:
        try:
            scenario = get_scenario(args.scenario)
        except KeyError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        
        samples = args.samples if args.samples else scenario.samples_default
        contract_ok, results = run_scenario(
            scenario, samples, args.json, args.assert_contract
        )
        
        if args.json:
            print(json.dumps(results, indent=2))
        
        if args.assert_contract and not contract_ok:
            return 3
        return 0
    
    # Handle --ci
    if args.ci:
        passed, results = run_ci_checks(args.ci_nc_samples, args.json)
        if args.json:
            print(json.dumps(results, indent=2))
        return 0 if passed else 3
    
    # Handle --all-profiles
    if args.all_profiles:
        all_coverage = {}
        for profile_name in SLICE_PROFILES:
            coverage = ProfileCoverageMap.from_profile(profile_name)
            all_coverage[profile_name] = coverage.to_dict()
            if not args.json:
                print(format_coverage_human(coverage))
                print()
        
        if args.json:
            print(json.dumps({"profiles": all_coverage}, indent=2))
        
        return 0
    
    # Handle --coverage
    if args.coverage:
        coverage = ProfileCoverageMap.from_profile(args.profile)
        
        if args.json:
            output = {
                "profile": args.profile,
                "coverage": coverage.to_dict(),
            }
            print(json.dumps(output, indent=2))
        else:
            print(format_coverage_human(coverage))
        
        return 0
    
    # Validate formula input
    formulas = []
    if args.formula:
        formulas.append(args.formula)
    
    if args.formulas_file:
        try:
            with open(args.formulas_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        formulas.append(line)
        except FileNotFoundError:
            print(f"Error: File not found: {args.formulas_file}", file=sys.stderr)
            return 1
        except IOError as e:
            print(f"Error reading file: {e}", file=sys.stderr)
            return 1
    
    if not formulas:
        print("Error: No formulas provided. Use --formula or --formulas-file.", file=sys.stderr)
        return 1
    
    # Create oracle configuration
    try:
        config = MockOracleConfig(
            slice_profile=args.profile,
            timeout_ms=args.timeout_ms,
            enable_crashes=args.enable_crashes,
            seed=args.seed,
            negative_control=args.negative_control,
        )
        oracle = MockVerifiableOracle(config)
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1
    
    # Run verification
    try:
        results = run_verification(oracle, formulas, args.count, args.json)
    except Exception as e:
        print(f"Runtime error: {e}", file=sys.stderr)
        return 2
    
    # Generate summary
    summary = None
    if args.summary:
        summary = show_summary(oracle, args.json)
    
    # Output JSON if requested
    if args.json:
        output = {
            "config": {
                "profile": args.profile,
                "timeout_ms": args.timeout_ms,
                "enable_crashes": args.enable_crashes,
                "seed": args.seed,
                "negative_control": args.negative_control,
            },
            "results": results,
        }
        if summary:
            output["summary"] = summary
        
        print(json.dumps(output, indent=2))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

