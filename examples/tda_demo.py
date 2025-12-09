#!/usr/bin/env python3
"""
TDA Integration Demo

Demonstrates the TDA governance integration with evidence fusion.
"""

import sys
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rfl.evidence_fusion import (
    fuse_evidence_summaries,
    TDAFields,
    TDAOutcome,
    RunEntry,
)
from rfl.hard_gate import (
    evaluate_hard_gate_decision,
    HardGateMode,
)
from rfl.runner_integration import (
    HardGateIntegration,
    create_policy_state_snapshot,
    create_mock_event_stats,
)


def demo_hard_gate_evaluation():
    """Demonstrate hard gate evaluation for a single cycle."""
    print("=" * 70)
    print("DEMO 1: Hard Gate Evaluation")
    print("=" * 70)
    
    # Create policy state
    policy_state = {
        "weights": {
            "len": 0.5,
            "depth": 0.3,
            "success": 0.2,
        },
        "version": "v1.0",
    }
    
    # Create event stats
    event_stats = {
        "total_events": 100,
        "blocked": 10,
        "passed": 90,
    }
    
    # Evaluate hard gate
    decision = evaluate_hard_gate_decision(
        cycle=0,
        policy_state=policy_state,
        event_stats=event_stats,
        mode=HardGateMode.SHADOW,
    )
    
    print(f"\nCycle 0 Decision:")
    print(f"  Outcome: {decision.outcome.value}")
    print(f"  HSS: {decision.tda_fields.HSS:.2f}")
    print(f"  Block Rate: {decision.tda_fields.block_rate:.2%}")
    print(f"  Reason: {decision.decision_reason}")
    print()


def demo_multi_cycle_evaluation():
    """Demonstrate hard gate evaluation across multiple cycles."""
    print("=" * 70)
    print("DEMO 2: Multi-Cycle Hard Gate Evaluation")
    print("=" * 70)
    
    integration = HardGateIntegration(mode=HardGateMode.SHADOW)
    
    # Evaluate 5 cycles with increasing block rate
    for cycle in range(5):
        policy_state = create_policy_state_snapshot(
            policy_weights={"len": 0.5, "depth": 0.3 + cycle * 0.05},
            success_count={},
            attempt_count={},
        )
        
        event_stats = create_mock_event_stats(
            total_events=100,
            blocked_events=5 + cycle * 10,  # Increasing block rate
        )
        
        decision = integration.evaluate_cycle(
            cycle=cycle,
            policy_state=policy_state,
            event_stats=event_stats,
        )
        
        print(f"\nCycle {cycle}:")
        print(f"  Outcome: {decision.outcome.value}")
        print(f"  HSS: {decision.tda_fields.HSS:.2f}")
        print(f"  Block Rate: {decision.tda_fields.block_rate:.2%}")
        print(f"  Reason: {decision.decision_reason}")
    
    # Get aggregate metrics
    print("\nAggregate Metrics:")
    metrics = integration.get_aggregate_metrics()
    print(f"  Mean HSS: {metrics['hss_aggregate'].get('mean_stability', 0):.2f}")
    print(f"  HSS Trend: {metrics['hss_aggregate'].get('trend', 'unknown')}")
    print(f"  Mean Block Rate: {metrics['mean_block_rate']:.2%}")
    print(f"  Total Cycles: {metrics['total_cycles']}")
    print()


def demo_evidence_fusion():
    """Demonstrate evidence fusion with TDA fields."""
    print("=" * 70)
    print("DEMO 3: Evidence Fusion with TDA")
    print("=" * 70)
    
    # Create baseline runs
    baseline_runs = [
        RunEntry(
            run_id=f"baseline_{i}",
            experiment_id="DEMO_EXP",
            slice_name="demo_slice",
            mode="baseline",
            coverage_rate=0.70 + i * 0.02,
            novelty_rate=0.5,
            throughput=10.0,
            success_rate=0.75,
            abstention_fraction=0.25,
            tda=TDAFields(
                HSS=0.80,
                block_rate=0.10,
                tda_outcome=TDAOutcome.PASS,
            ),
        )
        for i in range(3)
    ]
    
    # Create RFL runs with improved performance
    rfl_runs = [
        RunEntry(
            run_id=f"rfl_{i}",
            experiment_id="DEMO_EXP",
            slice_name="demo_slice",
            mode="rfl",
            coverage_rate=0.82 + i * 0.02,
            novelty_rate=0.6,
            throughput=12.0,
            success_rate=0.85,
            abstention_fraction=0.15,
            tda=TDAFields(
                HSS=0.85,
                block_rate=0.08,
                tda_outcome=TDAOutcome.PASS,
            ),
        )
        for i in range(3)
    ]
    
    # Fuse evidence
    summary = fuse_evidence_summaries(
        baseline_runs=baseline_runs,
        rfl_runs=rfl_runs,
        experiment_id="DEMO_EXP",
        slice_name="demo_slice",
        tda_hard_gate_mode="SHADOW",
    )
    
    print(f"\nFused Evidence Summary:")
    print(f"  Experiment: {summary.experiment_id}")
    print(f"  Slice: {summary.slice_name}")
    print(f"  Baseline Runs: {len(summary.baseline_runs)}")
    print(f"  RFL Runs: {len(summary.rfl_runs)}")
    print(f"  Baseline Mean Coverage: {summary.baseline_mean_coverage:.2%}")
    print(f"  RFL Mean Coverage: {summary.rfl_mean_coverage:.2%}")
    print(f"  Mean Block Rate: {summary.mean_block_rate:.2%}")
    print(f"  TDA Pass Rate: {summary.tda_pass_rate:.2%}")
    print(f"  Inconsistencies: {len(summary.inconsistencies)}")
    print(f"  Promotion Blocked: {summary.promotion_blocked}")
    print(f"  Fusion Hash: {summary.fusion_hash[:16]}...")
    
    if summary.inconsistencies:
        print("\nInconsistencies Detected:")
        for inc in summary.inconsistencies:
            print(f"  - [{inc.severity.upper()}] {inc.inconsistency_type.value}")
            print(f"    {inc.message}")
    
    print()


def demo_promotion_gate():
    """Demonstrate promotion gate with structural risk."""
    print("=" * 70)
    print("DEMO 4: Promotion Gate with Structural Risk")
    print("=" * 70)
    
    # Create RFL runs with high block rate (structural risk)
    rfl_runs = [
        RunEntry(
            run_id="rfl_risky",
            experiment_id="DEMO_EXP",
            slice_name="demo_slice",
            mode="rfl",
            coverage_rate=0.90,  # Good coverage
            novelty_rate=0.7,
            throughput=15.0,
            success_rate=0.9,
            abstention_fraction=0.1,
            tda=TDAFields(
                HSS=0.25,  # Structural instability
                block_rate=0.96,  # Critical blocking
                tda_outcome=TDAOutcome.BLOCK,
            ),
        )
    ]
    
    # Test in SHADOW mode
    print("\nSHADOW Mode:")
    summary_shadow = fuse_evidence_summaries(
        baseline_runs=[],
        rfl_runs=rfl_runs,
        experiment_id="DEMO_EXP",
        slice_name="demo_slice",
        tda_hard_gate_mode="SHADOW",
    )
    
    print(f"  Promotion Blocked: {summary_shadow.promotion_blocked}")
    print(f"  (In SHADOW mode, issues are logged but don't block)")
    
    # Test in ENFORCE mode
    print("\nENFORCE Mode:")
    summary_enforce = fuse_evidence_summaries(
        baseline_runs=[],
        rfl_runs=rfl_runs,
        experiment_id="DEMO_EXP",
        slice_name="demo_slice",
        tda_hard_gate_mode="ENFORCE",
    )
    
    print(f"  Promotion Blocked: {summary_enforce.promotion_blocked}")
    if summary_enforce.promotion_blocked:
        print(f"  Block Reason: {summary_enforce.promotion_block_reason}")
    
    print()


def main():
    """Run all demos."""
    print("\n")
    print("╔═════════════════════════════════════════════════════════════════════╗")
    print("║                   TDA Integration Demo                              ║")
    print("║              Phase II U2 Uplift Experiments Extension               ║")
    print("╚═════════════════════════════════════════════════════════════════════╝")
    print()
    
    demo_hard_gate_evaluation()
    demo_multi_cycle_evaluation()
    demo_evidence_fusion()
    demo_promotion_gate()
    
    print("=" * 70)
    print("All demos completed successfully!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
