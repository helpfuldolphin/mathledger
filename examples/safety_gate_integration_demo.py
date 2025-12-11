#!/usr/bin/env python3
"""
Safety Gate Integration Demo (Phase X Neural Link)

Demonstrates how to use the safety gate module to surface decisions
into First Light, global health, and evidence packs.

Usage:
    python examples/safety_gate_integration_demo.py
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.governance.safety_gate import (
    SafetyEnvelope,
    SafetyGateStatus,
    SafetyGateDecision,
    build_safety_gate_summary_for_first_light,
    build_safety_gate_tile_for_global_health,
    attach_safety_gate_to_evidence,
    build_global_health_surface,
)


def simulate_safety_gate_run():
    """Simulate a run with safety gate decisions."""
    
    print("=" * 80)
    print("PHASE X NEURAL LINK - SAFETY GATE INTEGRATION DEMO")
    print("=" * 80)
    print()
    
    # Simulate decisions from a 100-cycle run
    decisions = [
        SafetyGateDecision(cycle=10, status=SafetyGateStatus.WARN, 
                          reason="latency_spike", timestamp="2025-12-11T04:00:00Z"),
        SafetyGateDecision(cycle=25, status=SafetyGateStatus.WARN, 
                          reason="memory_pressure", timestamp="2025-12-11T04:01:00Z"),
        SafetyGateDecision(cycle=50, status=SafetyGateStatus.BLOCK, 
                          reason="critical_invariant_violation", timestamp="2025-12-11T04:02:00Z"),
        SafetyGateDecision(cycle=51, status=SafetyGateStatus.BLOCK, 
                          reason="critical_invariant_violation", timestamp="2025-12-11T04:02:01Z"),
        SafetyGateDecision(cycle=75, status=SafetyGateStatus.WARN, 
                          reason="latency_spike", timestamp="2025-12-11T04:03:00Z"),
    ]
    
    # Create envelope
    envelope = SafetyEnvelope(
        final_status=SafetyGateStatus.BLOCK,
        total_decisions=100,
        blocked_cycles=2,
        advisory_cycles=3,
        decisions=decisions,
    )
    
    print("📊 SAFETY ENVELOPE SUMMARY")
    print(f"   Final Status: {envelope.final_status.value}")
    print(f"   Total Decisions: {envelope.total_decisions}")
    print(f"   Blocked Cycles: {envelope.blocked_cycles}")
    print(f"   Advisory Cycles: {envelope.advisory_cycles}")
    print(f"   Unique Reasons: {envelope.get_reasons()}")
    print()
    
    # 1. First Light Summary
    print("1️⃣  FIRST LIGHT SUMMARY")
    print("-" * 80)
    first_light_summary = build_safety_gate_summary_for_first_light(envelope)
    print(json.dumps(first_light_summary, indent=2))
    print()
    
    # Show how to integrate into First Light output
    first_light_full = {
        "experiment_id": "first_organism_2025_12_11",
        "timestamp": "2025-12-11T04:30:00Z",
        "coverage": 0.92,
        "uplift": 1.15,
        "safety_gate_summary": first_light_summary,  # ← Integration point
    }
    
    print("   Integrated into First Light summary.json:")
    print(json.dumps(first_light_full, indent=2))
    print()
    
    # 2. Global Health Tile
    print("2️⃣  GLOBAL HEALTH TILE")
    print("-" * 80)
    health_tile = build_safety_gate_tile_for_global_health(envelope)
    print(json.dumps(health_tile, indent=2))
    print()
    
    # Show how to integrate into global health
    global_health = build_global_health_surface(
        tiles={
            "database": {"status_light": "GREEN", "headline": "DB OK"},
            "redis": {"status_light": "GREEN", "headline": "Redis OK"},
        },
        safety_envelope=envelope,
    )
    
    print("   Integrated into global_health.json:")
    print(json.dumps(global_health, indent=2))
    print()
    
    # 3. Evidence Pack Attachment
    print("3️⃣  EVIDENCE PACK ATTACHMENT")
    print("-" * 80)
    
    # Simulate existing evidence pack
    evidence_before = {
        "version": "1.0.0",
        "experiment": {
            "id": "first_organism_2025_12_11",
            "type": "rfl_experiment",
        },
        "artifacts": {
            "logs": ["results/fo_baseline.jsonl"],
            "figures": ["artifacts/dyno_chart.png"],
        },
        "governance": {
            "curriculum_hash": "abc123...",
        }
    }
    
    evidence_after = attach_safety_gate_to_evidence(evidence_before, envelope)
    
    print("   Original evidence (governance section):")
    print(json.dumps(evidence_before["governance"], indent=2))
    print()
    
    print("   Updated evidence (governance section with safety gate):")
    print(json.dumps(evidence_after["governance"], indent=2))
    print()
    
    # 4. Verify No Mutation
    print("4️⃣  VERIFY NO MUTATION")
    print("-" * 80)
    assert evidence_before["governance"] == {"curriculum_hash": "abc123..."}
    print("   ✅ Original evidence unchanged (no mutation)")
    print()
    
    # 5. Demonstrate Determinism
    print("5️⃣  VERIFY DETERMINISM")
    print("-" * 80)
    
    summary1 = build_safety_gate_summary_for_first_light(envelope)
    summary2 = build_safety_gate_summary_for_first_light(envelope)
    
    assert summary1 == summary2
    assert summary1["reasons"] == sorted(summary1["reasons"])
    print("   ✅ Deterministic output confirmed")
    print("   ✅ Reasons alphabetically sorted")
    print()
    
    # Summary
    print("=" * 80)
    print("✅ DEMO COMPLETE - ALL INTEGRATIONS VERIFIED")
    print("=" * 80)
    print()
    print("Key Features Demonstrated:")
    print("  ✓ First Light summary integration")
    print("  ✓ Global health tile with traffic light mapping")
    print("  ✓ Evidence pack attachment (top 3 reasons)")
    print("  ✓ No mutation of inputs")
    print("  ✓ Deterministic output")
    print()
    print("Integration Points:")
    print("  • First Light: Add 'safety_gate_summary' to summary.json")
    print("  • Global Health: Call build_global_health_surface() with envelope")
    print("  • Evidence Pack: Call attach_safety_gate_to_evidence() before sealing")
    print()


if __name__ == "__main__":
    simulate_safety_gate_run()
