#!/usr/bin/env python3
"""
CI Pipeline Health Evaluator.

PHASE II - NOT RUN IN PHASE I
No uplift claims are made.

This script evaluates pipeline health from CI stage results and outputs
a degradation decision for use in GitHub Actions workflows.

Usage:
    python scripts/ci_pipeline_health.py --stage-results results.json
    python scripts/ci_pipeline_health.py --from-env
    python scripts/ci_pipeline_health.py --node-statuses statuses.json

    # Snapshot mode - save current health state
    python scripts/ci_pipeline_health.py --stage-results results.json --snapshot snapshot.json

    # Compare mode - compare current to previous snapshot
    python scripts/ci_pipeline_health.py --stage-results results.json \\
        --compare-to previous_snapshot.json

    # Explain mode - show detailed decision explanation
    python scripts/ci_pipeline_health.py --stage-results results.json --explain

Exit Codes:
    0: FULL_PIPELINE - all healthy, Delta-p permitted
    1: EVIDENCE_ONLY - critical failure, Delta-p forbidden, CI should halt
    2: DEGRADED_ANALYSIS - partial success, restricted Delta-p

Reference: docs/U2_PIPELINE_TOPOLOGY.md Sections 10-12
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.pipeline.topology_health import (
    DecisionExplanation,
    DegradationDecision,
    DegradationPolicyEngine,
    GovernanceLabel,
    HealthSnapshot,
    IntegrityCheck,
    NodeHealth,
    NodeStatus,
    PipelineHealth,
    PipelineMode,
    SnapshotComparison,
    TopologyHealthEvaluator,
    ValidationStatus,
    build_pipeline_health_snapshot,
    compare_health_snapshots,
)


# Exit codes for CI integration
EXIT_CODE_FULL_PIPELINE = 0
EXIT_CODE_EVIDENCE_ONLY = 1
EXIT_CODE_DEGRADED_ANALYSIS = 2


def parse_stage_results_from_env() -> Dict[str, Dict[str, Any]]:
    """
    Parse CI stage results from environment variables.

    Expected environment variables:
    - STAGE_GATE_CHECK_STATUS: OK, FAIL, SKIP
    - STAGE_PREREG_VERIFY_STATUS: OK, FAIL, SKIP
    - STAGE_CURRICULUM_LOAD_STATUS: OK, FAIL, SKIP
    - STAGE_DRY_RUN_STATUS: OK, FAIL, SKIP
    - STAGE_MANIFEST_INIT_STATUS: OK, FAIL, SKIP
    - STAGE_RUN_SLICE_GOAL_STATUS: OK, FAIL, SKIP
    - STAGE_RUN_SLICE_SPARSE_STATUS: OK, FAIL, SKIP
    - STAGE_RUN_SLICE_TREE_STATUS: OK, FAIL, SKIP
    - STAGE_RUN_SLICE_DEP_STATUS: OK, FAIL, SKIP
    - STAGE_EVAL_GOAL_STATUS: OK, FAIL, SKIP
    - STAGE_EVAL_SPARSE_STATUS: OK, FAIL, SKIP
    - STAGE_EVAL_TREE_STATUS: OK, FAIL, SKIP
    - STAGE_EVAL_DEP_STATUS: OK, FAIL, SKIP
    - STAGE_INTEGRITY_CHECK_STATUS: OK, FAIL, SKIP

    Returns:
        Dict of stage_name → {status, exit_code}
    """
    stage_env_map = {
        "gate-check": "STAGE_GATE_CHECK_STATUS",
        "prereg-verify": "STAGE_PREREG_VERIFY_STATUS",
        "curriculum-load": "STAGE_CURRICULUM_LOAD_STATUS",
        "dry-run": "STAGE_DRY_RUN_STATUS",
        "manifest-init": "STAGE_MANIFEST_INIT_STATUS",
        "run-slice-goal": "STAGE_RUN_SLICE_GOAL_STATUS",
        "run-slice-sparse": "STAGE_RUN_SLICE_SPARSE_STATUS",
        "run-slice-tree": "STAGE_RUN_SLICE_TREE_STATUS",
        "run-slice-dep": "STAGE_RUN_SLICE_DEP_STATUS",
        "eval-goal": "STAGE_EVAL_GOAL_STATUS",
        "eval-sparse": "STAGE_EVAL_SPARSE_STATUS",
        "eval-tree": "STAGE_EVAL_TREE_STATUS",
        "eval-dep": "STAGE_EVAL_DEP_STATUS",
        "integrity-check": "STAGE_INTEGRITY_CHECK_STATUS",
        "contamination-check": "STAGE_CONTAMINATION_CHECK_STATUS",
    }

    results: Dict[str, Dict[str, Any]] = {}

    for stage_name, env_var in stage_env_map.items():
        status = os.environ.get(env_var, "NOT_RUN")
        exit_code_var = env_var.replace("_STATUS", "_EXIT_CODE")
        exit_code_str = os.environ.get(exit_code_var)
        exit_code = int(exit_code_str) if exit_code_str else None

        results[stage_name] = {
            "status": status,
            "exit_code": exit_code,
        }

    return results


def parse_node_statuses_from_env() -> Dict[str, NodeStatus]:
    """
    Parse node statuses from environment variables.

    Expected environment variables:
    - NODE_N01_STATUS: OK, WARN, FAIL, SKIPPED, NOT_RUN
    - NODE_N02_STATUS: ...
    - etc.

    Returns:
        Dict of node_id → NodeStatus
    """
    node_ids = [
        "N01", "N02", "N03", "N04", "N05",
        "N10", "N11", "N12", "N13",
        "N20", "N21", "N22", "N23",
        "N30", "N40", "N41", "N50", "N60", "N70", "N80", "N90",
    ]

    statuses: Dict[str, NodeStatus] = {}

    for node_id in node_ids:
        env_var = f"NODE_{node_id}_STATUS"
        status_str = os.environ.get(env_var, "NOT_RUN")
        try:
            statuses[node_id] = NodeStatus(status_str)
        except ValueError:
            statuses[node_id] = NodeStatus.NOT_RUN

    return statuses


def load_stage_results(path: str) -> Dict[str, Dict[str, Any]]:
    """Load stage results from a JSON file."""
    with open(path) as f:
        return json.load(f)


def load_node_statuses(path: str) -> Dict[str, NodeStatus]:
    """Load node statuses from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    return {k: NodeStatus(v) for k, v in data.items()}


def format_github_output(decision: DegradationDecision) -> str:
    """Format decision for GitHub Actions output."""
    lines = [
        f"mode={decision.mode.value}",
        f"governance_label={decision.governance_label.value}",
        f"allow_delta_p={str(decision.allow_delta_p).lower()}",
        f"allowed_slices={','.join(sorted(decision.allowed_delta_p_slices))}",
        f"evidence_pack_status={decision.evidence_pack_status}",
        f"halt_ci={str(decision.halt_ci).lower()}",
        f"quarantine={str(decision.quarantine).lower()}",
    ]
    return "\n".join(lines)


def write_github_output(decision: DegradationDecision) -> None:
    """Write decision to GitHub Actions output file."""
    output_file = os.environ.get("GITHUB_OUTPUT")
    if output_file:
        with open(output_file, "a") as f:
            f.write(format_github_output(decision))
            f.write("\n")


def print_summary(
    health: Optional[PipelineHealth],
    decision: DegradationDecision,
    verbose: bool = False,
) -> None:
    """Print a summary of the health evaluation."""
    print("=" * 60)
    print("U2 Pipeline Health Evaluation")
    print("PHASE II - NOT RUN IN PHASE I")
    print("=" * 60)
    print()

    print(f"Pipeline Mode: {decision.mode.value}")
    print(f"Governance Label: {decision.governance_label.value}")
    print()

    print("Delta-p Computation:")
    print(f"  Allowed: {decision.allow_delta_p}")
    if decision.allowed_delta_p_slices:
        print(f"  Allowed Slices: {', '.join(sorted(decision.allowed_delta_p_slices))}")
    else:
        print("  Allowed Slices: (none)")
    print()

    print("Evidence Pack:")
    print(f"  Status: {decision.evidence_pack_status}")
    print(f"  Quarantine: {decision.quarantine}")
    print()

    print("CI Decision:")
    print(f"  Halt CI: {decision.halt_ci}")
    if decision.halt_reason:
        print(f"  Reason: {decision.halt_reason}")
    print()

    if decision.violations:
        print("Violations/Notes:")
        for v in decision.violations:
            print(f"  - {v}")
        print()

    if health and verbose:
        print("Pipeline Health Details:")
        print(f"  Successful Slices: {', '.join(health.successful_slices) or '(none)'}")
        print(f"  Failed Slices: {', '.join(health.failed_slices) or '(none)'}")
        print(f"  Failed Nodes: {', '.join(health.failed_nodes) or '(none)'}")
        print()

    print("=" * 60)
    print(f"Timestamp: {decision.timestamp}")
    print("=" * 60)


def print_comparison_summary(comparison: SnapshotComparison) -> None:
    """Print a summary of the snapshot comparison."""
    print("=" * 60)
    print("Health Trend Analysis")
    print("=" * 60)
    print()

    # Overall trend
    if comparison.degraded:
        print("Trend: DEGRADED (health worsened)")
    elif comparison.improved:
        print("Trend: IMPROVED (health improved)")
    elif comparison.unchanged:
        print("Trend: UNCHANGED (no significant changes)")
    else:
        print("Trend: MIXED (some improvements, some degradations)")
    print()

    # Mode transition
    if comparison.mode_delta:
        print(f"Mode Transition: {comparison.mode_delta}")
    if comparison.label_delta:
        print(f"Label Transition: {comparison.label_delta}")

    # Failure counts
    print()
    print("Failure Count Deltas:")
    print(f"  Hard-fail delta: {comparison.hard_fail_delta:+d}")
    print(f"  Soft-fail delta: {comparison.soft_fail_delta:+d}")
    print(f"  Slice-fail delta: {comparison.slice_fail_delta:+d}")

    # Pattern changes
    if comparison.new_patterns:
        print()
        print("New Patterns Detected:")
        for p in comparison.new_patterns:
            print(f"  - {p}")

    if comparison.removed_patterns:
        print()
        print("Patterns Resolved:")
        for p in comparison.removed_patterns:
            print(f"  - {p}")

    if comparison.any_new_critical_pattern:
        print()
        print("WARNING: New critical pattern(s) detected!")

    print()
    print("=" * 60)


def write_github_step_summary(explanation: DecisionExplanation) -> None:
    """Write explanation to GitHub Actions step summary."""
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a") as f:
            f.write("\n")
            f.write(explanation.format_for_ci())
            f.write("\n")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate U2 pipeline health for CI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--stage-results",
        type=str,
        help="Path to JSON file with stage results",
    )
    parser.add_argument(
        "--node-statuses",
        type=str,
        help="Path to JSON file with node statuses",
    )
    parser.add_argument(
        "--from-env",
        action="store_true",
        help="Parse stage results from environment variables",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to write decision JSON",
    )
    parser.add_argument(
        "--github-output",
        action="store_true",
        help="Write decision to GITHUB_OUTPUT",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output decision as JSON only",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error on any degradation",
    )
    parser.add_argument(
        "--snapshot",
        type=str,
        help="Path to save health snapshot JSON",
    )
    parser.add_argument(
        "--compare-to",
        type=str,
        dest="compare_to",
        help="Path to previous snapshot JSON for comparison",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Show detailed decision explanation",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        dest="run_id",
        help="Run ID for snapshot correlation",
    )

    args = parser.parse_args()

    # Load inputs
    health: Optional[PipelineHealth] = None
    decision: Optional[DegradationDecision] = None

    if args.node_statuses:
        # Evaluate from node statuses
        node_statuses = load_node_statuses(args.node_statuses)
        evaluator = TopologyHealthEvaluator()
        health = evaluator.evaluate_pipeline_health(node_statuses)

        # Convert to CI stage results for degradation engine
        ci_results = convert_health_to_stage_results(health)
        engine = DegradationPolicyEngine()
        decision = engine.evaluate_degradation(ci_results)

    elif args.stage_results:
        # Evaluate from stage results file
        ci_results = load_stage_results(args.stage_results)
        engine = DegradationPolicyEngine()
        decision = engine.evaluate_degradation(ci_results)

    elif args.from_env:
        # Evaluate from environment variables
        ci_results = parse_stage_results_from_env()
        engine = DegradationPolicyEngine()
        decision = engine.evaluate_degradation(ci_results)

    else:
        # Default: assume all OK for testing
        print("No input specified. Use --stage-results, --node-statuses, or --from-env")
        print("Running with default all-OK configuration for demonstration...")
        print()

        ci_results = {
            "gate-check": {"status": "OK"},
            "prereg-verify": {"status": "OK"},
            "curriculum-load": {"status": "OK"},
            "dry-run": {"status": "OK"},
            "manifest-init": {"status": "OK"},
            "run-slice-goal": {"status": "OK"},
            "run-slice-sparse": {"status": "OK"},
            "run-slice-tree": {"status": "OK"},
            "run-slice-dep": {"status": "OK"},
            "eval-goal": {"status": "OK"},
            "eval-sparse": {"status": "OK"},
            "eval-tree": {"status": "OK"},
            "eval-dep": {"status": "OK"},
            "integrity-check": {"status": "OK"},
        }
        engine = DegradationPolicyEngine()
        decision = engine.evaluate_degradation(ci_results)

    # Generate explanation if requested or needed for snapshot
    explanation: Optional[DecisionExplanation] = None
    if args.explain or args.snapshot:
        explanation = engine.explain(decision, ci_results, health)

    # Build snapshot if health data available
    snapshot: Optional[HealthSnapshot] = None
    if health and decision:
        snapshot = build_pipeline_health_snapshot(health, decision, args.run_id)

    # Compare to previous snapshot if requested
    comparison: Optional[SnapshotComparison] = None
    if args.compare_to and snapshot:
        try:
            with open(args.compare_to) as f:
                old_data = json.load(f)
            old_snapshot = HealthSnapshot.from_dict(old_data)
            comparison = compare_health_snapshots(old_snapshot, snapshot)
        except FileNotFoundError:
            print(f"Warning: Previous snapshot not found: {args.compare_to}")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to parse previous snapshot: {e}")

    # Output
    if args.json:
        output_data = decision.to_dict()
        if explanation:
            output_data["explanation"] = explanation.to_dict()
        if snapshot:
            output_data["snapshot"] = snapshot.to_dict()
        if comparison:
            output_data["comparison"] = comparison.to_dict()
        print(json.dumps(output_data, indent=2))
    else:
        print_summary(health, decision, verbose=args.verbose)

        # Print explanation if requested
        if args.explain and explanation:
            print()
            print("Decision Explanation:")
            print("-" * 40)
            print(explanation.format_for_ci())
            print()

        # Print comparison if available
        if comparison:
            print()
            print_comparison_summary(comparison)

    if args.output:
        output_data = decision.to_dict()
        if explanation:
            output_data["explanation"] = explanation.to_dict()
        if snapshot:
            output_data["snapshot"] = snapshot.to_dict()
        if comparison:
            output_data["comparison"] = comparison.to_dict()
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)

    # Save snapshot if requested
    if args.snapshot and snapshot:
        with open(args.snapshot, "w") as f:
            f.write(snapshot.to_json())
        if not args.json:
            print(f"Snapshot saved to: {args.snapshot}")

    if args.github_output:
        write_github_output(decision)
        # Also write explanation summary to step summary if available
        if explanation:
            write_github_step_summary(explanation)

    # Determine exit code
    if decision.mode == PipelineMode.FULL_PIPELINE:
        return EXIT_CODE_FULL_PIPELINE
    elif decision.mode == PipelineMode.EVIDENCE_ONLY:
        return EXIT_CODE_EVIDENCE_ONLY
    else:  # DEGRADED_ANALYSIS
        if args.strict:
            return EXIT_CODE_EVIDENCE_ONLY
        return EXIT_CODE_DEGRADED_ANALYSIS


def convert_health_to_stage_results(health: PipelineHealth) -> Dict[str, Dict[str, Any]]:
    """Convert PipelineHealth to CI stage results format."""
    results: Dict[str, Dict[str, Any]] = {}

    # Validation stages - assume OK unless in failed_nodes
    validation_nodes = {"N01", "N02", "N03", "N04", "N05"}
    validation_failed = bool(validation_nodes & set(health.failed_nodes))

    results["gate-check"] = {"status": "FAIL" if "N01" in health.failed_nodes else "OK"}
    results["prereg-verify"] = {"status": "FAIL" if "N02" in health.failed_nodes else "OK"}
    results["curriculum-load"] = {"status": "FAIL" if "N03" in health.failed_nodes else "OK"}
    results["dry-run"] = {"status": "FAIL" if "N04" in health.failed_nodes else "OK"}
    results["manifest-init"] = {"status": "FAIL" if "N05" in health.failed_nodes else "OK"}

    # Slice stages
    slice_map = {
        "goal": ("N10", "N20"),
        "sparse": ("N11", "N21"),
        "tree": ("N12", "N22"),
        "dep": ("N13", "N23"),
    }

    for slice_name, (runner_id, eval_id) in slice_map.items():
        runner_status = "FAIL" if runner_id in health.failed_nodes else "OK"
        eval_status = "FAIL" if eval_id in health.failed_nodes else "OK"

        if slice_name in health.successful_slices:
            runner_status = "OK"
            eval_status = "OK"
        elif slice_name in health.failed_slices:
            # At least one must have failed
            if runner_id not in health.failed_nodes and eval_id not in health.failed_nodes:
                runner_status = "FAIL"  # Default to runner failure

        results[f"run-slice-{slice_name}"] = {"status": runner_status}
        results[f"eval-{slice_name}"] = {"status": eval_status}

    results["integrity-check"] = {"status": "OK"}

    return results


if __name__ == "__main__":
    sys.exit(main())
