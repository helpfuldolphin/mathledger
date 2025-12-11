# Safety Gate Quick Start Guide

**Phase X Neural Link** - 5-minute integration guide

---

## Quick Import

```python
from backend.governance import (
    SafetyEnvelope,
    SafetyGateStatus,
    SafetyGateDecision,
    build_safety_gate_summary_for_first_light,
    build_global_health_surface,
    attach_safety_gate_to_evidence,
)
```

---

## Three-Step Integration

### Step 1: Collect Decisions During Run

```python
# Initialize list
safety_decisions = []

# During each cycle where gate is evaluated
for cycle in range(total_cycles):
    # ... your cycle logic ...
    
    # If gate evaluates
    status = evaluate_gate_for_cycle(cycle)  # Your gate logic
    if status != SafetyGateStatus.PASS:
        safety_decisions.append(
            SafetyGateDecision(
                cycle=cycle,
                status=status,
                reason="latency_spike",  # Your reason
                timestamp=datetime.now(timezone.utc).isoformat()
            )
        )
```

### Step 2: Build Envelope at Completion

```python
# After run completes
envelope = SafetyEnvelope(
    final_status=compute_final_status(safety_decisions),
    total_decisions=total_cycles,
    blocked_cycles=sum(1 for d in safety_decisions if d.status == SafetyGateStatus.BLOCK),
    advisory_cycles=sum(1 for d in safety_decisions if d.status == SafetyGateStatus.WARN),
    decisions=safety_decisions,
)
```

### Step 3: Surface to Observability

```python
# First Light
first_light = load_or_create_first_light_summary()
first_light["safety_gate_summary"] = build_safety_gate_summary_for_first_light(envelope)
save_json(first_light, "artifacts/first_light_summary.json")

# Global Health
health = build_global_health_surface(
    tiles=load_existing_health_tiles(),
    safety_envelope=envelope
)
save_json(health, "artifacts/global_health.json")

# Evidence Pack
evidence = load_evidence_pack()
evidence_with_safety = attach_safety_gate_to_evidence(evidence, envelope)
seal_evidence_pack(evidence_with_safety)
```

---

## Status Reference

| Status | Meaning | Light | Use When |
|--------|---------|-------|----------|
| `PASS` | All checks passed | 🟢 GREEN | No issues detected |
| `WARN` | Advisory warnings | 🟡 YELLOW | Non-critical issues |
| `BLOCK` | Critical failures | 🔴 RED | Run should be blocked |

---

## Output Formats

### First Light Summary
```json
{
  "safety_gate_summary": {
    "final_status": "PASS",
    "total_decisions": 100,
    "blocked_cycles": 0,
    "advisory_cycles": 2,
    "reasons": ["latency_spike", "memory_pressure"]
  }
}
```

### Global Health Tile
```json
{
  "safety_gate": {
    "schema_version": "1.0.0",
    "status_light": "GREEN",
    "blocked_fraction": 0.0,
    "headline": "Safety gate: PASS",
    "total_decisions": 100,
    "blocked_cycles": 0,
    "advisory_cycles": 2
  }
}
```

### Evidence Pack
```json
{
  "governance": {
    "safety_gate": {
      "final_status": "PASS",
      "blocked_cycles": 0,
      "advisory_cycles": 2,
      "reasons": ["latency_spike", "memory_pressure"]
    }
  }
}
```

---

## Common Patterns

### Pattern 1: Simple Pass/Fail Gate
```python
decisions = []
for cycle in cycles:
    if should_block(cycle):
        decisions.append(SafetyGateDecision(
            cycle=cycle,
            status=SafetyGateStatus.BLOCK,
            reason="invariant_violation"
        ))

final_status = SafetyGateStatus.BLOCK if any(
    d.status == SafetyGateStatus.BLOCK for d in decisions
) else SafetyGateStatus.PASS

envelope = SafetyEnvelope(
    final_status=final_status,
    total_decisions=len(cycles),
    blocked_cycles=sum(1 for d in decisions if d.status == SafetyGateStatus.BLOCK),
    advisory_cycles=0,
    decisions=decisions,
)
```

### Pattern 2: Multi-Level Gate (PASS/WARN/BLOCK)
```python
def compute_final_status(decisions):
    """Compute final status from decisions."""
    if any(d.status == SafetyGateStatus.BLOCK for d in decisions):
        return SafetyGateStatus.BLOCK
    elif any(d.status == SafetyGateStatus.WARN for d in decisions):
        return SafetyGateStatus.WARN
    return SafetyGateStatus.PASS
```

### Pattern 3: Threshold-Based Gate
```python
# Example: Block if >10% cycles fail
blocked_count = sum(1 for d in decisions if d.status == SafetyGateStatus.BLOCK)
blocked_fraction = blocked_count / total_cycles

if blocked_fraction > 0.1:
    final_status = SafetyGateStatus.BLOCK
elif blocked_fraction > 0.05:
    final_status = SafetyGateStatus.WARN
else:
    final_status = SafetyGateStatus.PASS
```

---

## Checklist

Before committing integration:

- [ ] Import safety gate classes
- [ ] Collect SafetyGateDecision objects during run
- [ ] Build SafetyEnvelope at completion
- [ ] Add to First Light summary
- [ ] Update global health dashboard
- [ ] Attach to evidence pack
- [ ] Test determinism (run twice, compare)
- [ ] Add integration test

---

## Test Your Integration

```python
# Quick test
envelope = SafetyEnvelope(
    final_status=SafetyGateStatus.PASS,
    total_decisions=10,
    blocked_cycles=0,
    advisory_cycles=0,
)

summary = build_safety_gate_summary_for_first_light(envelope)
assert summary["final_status"] == "PASS"
print("✅ Integration working!")
```

---

## Need Help?

- **Full Guide**: `docs/SAFETY_GATE_INTEGRATION.md`
- **Demo**: `python examples/safety_gate_integration_demo.py`
- **Tests**: `tests/governance/test_safety_gate.py`
- **Summary**: `PHASE_X_SAFETY_GATE_IMPLEMENTATION.md`

---

**Ready to integrate!** 🚀
