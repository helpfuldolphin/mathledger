"""
Curriculum Stability Envelope Demonstration

Shows how to integrate the stability envelope with U2Runner and RFLRunner
to prevent invalid slice transitions during integration.

Usage:
    python examples/stability_envelope_demo.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from curriculum.stability_envelope import (
    CurriculumStabilityEnvelope,
    StabilityEnvelopeConfig,
)
from curriculum.stability_integration import (
    StabilityGateSpec,
    StabilityGateEvaluator,
    should_ratchet_with_stability,
    record_cycle_hss_metrics,
)
from curriculum.gates import make_first_organism_slice


def demo_basic_tracking():
    """Demonstrate basic HSS tracking."""
    print("=" * 70)
    print("Demo 1: Basic HSS Tracking")
    print("=" * 70)
    
    envelope = CurriculumStabilityEnvelope()
    
    # Simulate 20 cycles with stable HSS
    print("\nRecording 20 stable cycles...")
    for i in range(20):
        hss_value = 0.75 + (i % 3) * 0.02  # Small variation
        envelope.record_cycle(
            cycle_id=f"cycle_{i:03d}",
            slice_name="slice_uplift_goal",
            hss_value=hss_value,
            verified_count=10 + i,
            timestamp=f"2025-01-{i+1:02d}T00:00:00Z",
        )
    
    # Compute stability
    stability = envelope.compute_slice_stability("slice_uplift_goal")
    print(f"\nSlice: {stability.slice_name}")
    print(f"  Mean HSS: {stability.hss_mean:.4f}")
    print(f"  Std Dev:  {stability.hss_std:.4f}")
    print(f"  CV:       {stability.hss_cv:.4f}")
    print(f"  Stable:   {stability.is_stable}")
    print(f"  Suitability Score: {stability.suitability_score:.4f}")
    print(f"  Flags:    {', '.join(stability.flags) if stability.flags else 'none'}")


def demo_variance_spike_detection():
    """Demonstrate variance spike detection."""
    print("\n" + "=" * 70)
    print("Demo 2: Variance Spike Detection")
    print("=" * 70)
    
    envelope = CurriculumStabilityEnvelope()
    
    # Record baseline with low variance
    print("\nRecording baseline (20 cycles, low variance)...")
    for i in range(20):
        envelope.record_cycle(
            cycle_id=f"cycle_{i:03d}",
            slice_name="slice_uplift_sparse",
            hss_value=0.75 + (i % 2) * 0.01,
            verified_count=10,
            timestamp=f"2025-01-{i+1:02d}T00:00:00Z",
        )
    
    # Add variance spike
    print("Adding variance spike (10 cycles with high variance)...")
    spike_values = [0.9, 0.1, 0.95, 0.05, 0.85, 0.15, 0.9, 0.2, 0.95, 0.1]
    for i, hss in enumerate(spike_values):
        envelope.record_cycle(
            cycle_id=f"cycle_{20+i:03d}",
            slice_name="slice_uplift_sparse",
            hss_value=hss,
            verified_count=10,
            timestamp=f"2025-01-{20+i+1:02d}T00:00:00Z",
        )
    
    # Detect spike
    spike_detected, current_var = envelope.detect_variance_spike("slice_uplift_sparse")
    baseline_var = envelope.baseline_variance.get("slice_uplift_sparse", 0.0)
    
    print(f"\nVariance Spike Detected: {spike_detected}")
    print(f"  Baseline Variance: {baseline_var:.6f}")
    print(f"  Current Variance:  {current_var:.6f}")
    print(f"  Ratio: {current_var / baseline_var:.2f}x" if baseline_var > 0 else "")


def demo_slice_transition_control():
    """Demonstrate slice transition control."""
    print("\n" + "=" * 70)
    print("Demo 3: Slice Transition Control")
    print("=" * 70)
    
    envelope = CurriculumStabilityEnvelope()
    
    # Create stable slice
    print("\nCreating stable slice A...")
    for i in range(10):
        envelope.record_cycle(
            cycle_id=f"a_cycle_{i:03d}",
            slice_name="slice_a",
            hss_value=0.8 + (i % 2) * 0.02,
            verified_count=10,
            timestamp=f"2025-01-{i+1:02d}T00:00:00Z",
        )
    
    # Create unstable slice
    print("Creating unstable slice B (high variance)...")
    for i in range(10):
        envelope.record_cycle(
            cycle_id=f"b_cycle_{i:03d}",
            slice_name="slice_b",
            hss_value=0.9 if i % 2 == 0 else 0.1,
            verified_count=5,
            timestamp=f"2025-01-{i+1:02d}T00:00:00Z",
        )
    
    # Check transitions
    print("\n--- Transition from stable slice A ---")
    allowed, reason, details = envelope.check_slice_transition_allowed("slice_a", "slice_c")
    print(f"  Allowed: {allowed}")
    print(f"  Reason:  {reason}")
    
    print("\n--- Transition from unstable slice B ---")
    allowed, reason, details = envelope.check_slice_transition_allowed("slice_b", "slice_c")
    print(f"  Allowed: {allowed}")
    print(f"  Reason:  {reason}")


def demo_gate_integration():
    """Demonstrate integration with curriculum gates."""
    print("\n" + "=" * 70)
    print("Demo 4: Curriculum Gate Integration")
    print("=" * 70)
    
    envelope = CurriculumStabilityEnvelope()
    slice_cfg = make_first_organism_slice()
    
    # Record cycles
    print(f"\nRecording cycles for slice '{slice_cfg.name}'...")
    for i in range(10):
        envelope.record_cycle(
            cycle_id=f"cycle_{i:03d}",
            slice_name=slice_cfg.name,
            hss_value=0.85 + (i % 2) * 0.01,
            verified_count=12 + i,
            timestamp=f"2025-01-{i+1:02d}T00:00:00Z",
        )
    
    # Evaluate stability gate
    print("\nEvaluating stability gate...")
    spec = StabilityGateSpec(
        min_suitability_score=0.7,
        require_stable_slice=True,
        allow_variance_spikes=False,
    )
    evaluator = StabilityGateEvaluator(envelope, spec, slice_cfg)
    status = evaluator.evaluate()
    
    print(f"  Gate:    {status.gate}")
    print(f"  Passed:  {status.passed}")
    print(f"  Message: {status.message}")
    print(f"\n  Observed:")
    for key, value in status.observed.items():
        if key not in ["flags"]:
            print(f"    {key}: {value}")


def demo_suitability_scoring():
    """Demonstrate slice suitability scoring."""
    print("\n" + "=" * 70)
    print("Demo 5: Slice Suitability Scoring")
    print("=" * 70)
    
    envelope = CurriculumStabilityEnvelope()
    
    # Create three slices with different characteristics
    slices = {
        "excellent": (0.85, 0.01, 15),  # High mean, low variance, many proofs
        "moderate": (0.65, 0.05, 10),   # Medium mean, medium variance
        "poor": (0.25, 0.15, 3),         # Low mean, high variance, few proofs
    }
    
    print("\nCreating slices with different characteristics...")
    for slice_name, (base_hss, variance, base_proofs) in slices.items():
        for i in range(10):
            import random
            random.seed(i)
            hss_value = base_hss + (random.random() - 0.5) * variance * 2
            hss_value = max(0.0, min(1.0, hss_value))
            
            envelope.record_cycle(
                cycle_id=f"{slice_name}_cycle_{i:03d}",
                slice_name=slice_name,
                hss_value=hss_value,
                verified_count=base_proofs + i,
                timestamp=f"2025-01-{i+1:02d}T00:00:00Z",
            )
    
    # Get suitability scores
    scores = envelope.get_all_slice_suitability()
    
    print("\nSuitability Scores:")
    for slice_name, score in sorted(scores.items(), key=lambda x: -x[1]):
        stability = envelope.compute_slice_stability(slice_name)
        print(f"  {slice_name:12s}: {score:.4f}  (mean={stability.hss_mean:.3f}, cv={stability.hss_cv:.3f})")


def demo_full_report():
    """Demonstrate full stability report export."""
    print("\n" + "=" * 70)
    print("Demo 6: Full Stability Report")
    print("=" * 70)
    
    envelope = CurriculumStabilityEnvelope()
    
    # Record data for two slices
    print("\nRecording data for multiple slices...")
    for slice_name in ["slice_uplift_goal", "slice_uplift_sparse"]:
        for i in range(10):
            envelope.record_cycle(
                cycle_id=f"{slice_name}_cycle_{i:03d}",
                slice_name=slice_name,
                hss_value=0.75 + (i % 3) * 0.02,
                verified_count=10 + i,
                timestamp=f"2025-01-{i+1:02d}T00:00:00Z",
            )
    
    # Export report
    report = envelope.export_stability_report()
    
    print("\nStability Report:")
    print(f"  Config:")
    print(f"    max_hss_cv: {report['config']['max_hss_cv']}")
    print(f"    min_hss_threshold: {report['config']['min_hss_threshold']}")
    print(f"    min_cycles_for_stability: {report['config']['min_cycles_for_stability']}")
    
    print(f"\n  Slices: {len(report['slices'])}")
    for slice_name, metrics in report["slices"].items():
        print(f"\n    {slice_name}:")
        print(f"      is_stable: {metrics['is_stable']}")
        print(f"      suitability_score: {metrics['suitability_score']:.4f}")
        print(f"      hss_mean: {metrics['hss_mean']:.4f}")
        print(f"      hss_cv: {metrics['hss_cv']:.4f}")
        print(f"      variance_spike: {metrics['variance_spike']}")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("CURRICULUM STABILITY ENVELOPE DEMONSTRATION")
    print("=" * 70)
    print("\nThis demo shows how to integrate stability tracking with")
    print("U2Runner and RFLRunner to prevent invalid slice transitions.")
    print()
    
    demo_basic_tracking()
    demo_variance_spike_detection()
    demo_slice_transition_control()
    demo_gate_integration()
    demo_suitability_scoring()
    demo_full_report()
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\n✅ All demos executed successfully!")
    print("\nKey Takeaways:")
    print("  1. Stability envelope tracks HSS variance over time")
    print("  2. Variance spikes indicate curriculum instability")
    print("  3. Slice suitability scores guide uplift experiments")
    print("  4. Stability gates prevent transitions during instability")
    print("  5. Integration prevents cortex drift during First Light")


if __name__ == "__main__":
    main()
