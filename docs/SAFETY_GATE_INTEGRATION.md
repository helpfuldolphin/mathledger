# Safety Gate Integration Guide (Phase X Neural Link)

## Overview

The Safety Gate module (`backend/governance/safety_gate.py`) provides standardized interfaces for surfacing safety gate decisions into:
1. **First Light summaries** (First Organism output)
2. **Global health tiles** (dashboard monitoring)
3. **Evidence packs** (cryptographic audit trails)

**Design Constraint**: This module is behavior-preserving. It does NOT implement new gating logic—it only provides structures for surfacing existing gate decisions.

## Core Data Structures

### SafetyGateStatus

```python
from backend.governance.safety_gate import SafetyGateStatus

# Three states:
SafetyGateStatus.PASS   # All checks passed
SafetyGateStatus.WARN   # Advisory warnings (non-blocking)
SafetyGateStatus.BLOCK  # Critical failures (blocking)
```

### SafetyGateDecision

```python
from backend.governance.safety_gate import SafetyGateDecision

decision = SafetyGateDecision(
    cycle=10,
    status=SafetyGateStatus.WARN,
    reason="latency_spike",
    timestamp="2025-12-11T04:00:00Z",
    metadata={"latency_ms": 250}  # Optional
)
```

### SafetyEnvelope

Aggregates gate decisions across a full run:

```python
from backend.governance.safety_gate import SafetyEnvelope

envelope = SafetyEnvelope(
    final_status=SafetyGateStatus.PASS,
    total_decisions=100,
    blocked_cycles=0,
    advisory_cycles=2,
    decisions=[decision1, decision2, ...]
)

# Get deterministically ordered reasons
reasons = envelope.get_reasons()          # All unique reasons, sorted
top_3 = envelope.get_reasons(limit=3)    # Top 3 reasons
```

## Integration 1: First Light Summary

Add safety gate information to First Light (First Organism) summary output:

```python
from backend.governance.safety_gate import build_safety_gate_summary_for_first_light

# Build summary
safety_summary = build_safety_gate_summary_for_first_light(envelope)

# Integrate into First Light output
first_light_output = {
    "experiment_id": "fo_run_123",
    "timestamp": "2025-12-11T04:30:00Z",
    "coverage": 0.92,
    "uplift": 1.15,
    "safety_gate_summary": safety_summary,  # ← Add this field
}

# Write to summary.json
with open("artifacts/first_light_summary.json", "w") as f:
    json.dump(first_light_output, f, indent=2)
```

**Output Structure:**
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

## Integration 2: Global Health Tile

Add safety gate monitoring to the global health dashboard:

```python
from backend.governance.safety_gate import (
    build_safety_gate_tile_for_global_health,
    build_global_health_surface
)

# Option A: Build tile independently
safety_tile = build_safety_gate_tile_for_global_health(envelope)

# Option B: Use the surface builder (recommended)
global_health = build_global_health_surface(
    tiles={
        "database": {"status_light": "GREEN", "headline": "DB OK"},
        "redis": {"status_light": "GREEN", "headline": "Redis OK"},
        # ... other tiles
    },
    safety_envelope=envelope  # ← Pass envelope to add safety tile
)

# Write to global_health.json
with open("artifacts/global_health.json", "w") as f:
    json.dump(global_health, f, indent=2)
```

**Tile Structure:**
```json
{
  "safety_gate": {
    "schema_version": "1.0.0",
    "status_light": "GREEN",     // GREEN, YELLOW, or RED
    "blocked_fraction": 0.0,     // 0.0 to 1.0
    "headline": "Safety gate: PASS",
    "total_decisions": 100,
    "blocked_cycles": 0,
    "advisory_cycles": 2
  }
}
```

**Status Light Mapping:**
- `PASS` → `GREEN`
- `WARN` → `YELLOW`
- `BLOCK` → `RED`

## Integration 3: Evidence Pack

Attach safety gate data to cryptographic evidence packs:

```python
from backend.governance.safety_gate import attach_safety_gate_to_evidence

# Load existing evidence pack
evidence = {
    "version": "1.0.0",
    "experiment": {"id": "exp_123", "type": "rfl"},
    "artifacts": {...},
    "governance": {
        "curriculum_hash": "abc123...",
        # ... other governance data
    }
}

# Attach safety gate data (no mutation of original)
evidence_with_safety = attach_safety_gate_to_evidence(evidence, envelope)

# Seal the evidence pack
seal_evidence_pack(evidence_with_safety, output_path)
```

**Evidence Structure:**
```json
{
  "governance": {
    "curriculum_hash": "abc123...",
    "safety_gate": {
      "final_status": "PASS",
      "blocked_cycles": 0,
      "advisory_cycles": 2,
      "reasons": [
        "latency_spike",
        "memory_pressure"
        // Top 3 reasons only
      ]
    }
  }
}
```

**Important**: The function returns a NEW dictionary—it never mutates the input.

## Complete Integration Example

Here's a complete example showing all three integrations:

```python
from backend.governance.safety_gate import (
    SafetyEnvelope,
    SafetyGateStatus,
    SafetyGateDecision,
    build_safety_gate_summary_for_first_light,
    build_global_health_surface,
    attach_safety_gate_to_evidence,
)

def finalize_run_with_safety_gate(run_data, safety_decisions):
    """Finalize a run with safety gate integration."""
    
    # 1. Build safety envelope from decisions
    envelope = SafetyEnvelope(
        final_status=compute_final_status(safety_decisions),
        total_decisions=len(run_data["cycles"]),
        blocked_cycles=count_blocked(safety_decisions),
        advisory_cycles=count_advisory(safety_decisions),
        decisions=safety_decisions,
    )
    
    # 2. Add to First Light summary
    first_light = {
        "experiment_id": run_data["experiment_id"],
        "timestamp": run_data["timestamp"],
        "coverage": run_data["coverage"],
        "uplift": run_data["uplift"],
        "safety_gate_summary": build_safety_gate_summary_for_first_light(envelope),
    }
    save_json(first_light, "artifacts/first_light_summary.json")
    
    # 3. Update global health
    global_health = build_global_health_surface(
        tiles=get_existing_health_tiles(),
        safety_envelope=envelope,
    )
    save_json(global_health, "artifacts/global_health.json")
    
    # 4. Attach to evidence pack
    evidence = load_evidence_pack()
    evidence_with_safety = attach_safety_gate_to_evidence(evidence, envelope)
    seal_evidence_pack(evidence_with_safety)
    
    return envelope.final_status
```

## Testing

The module includes comprehensive tests in `tests/governance/test_safety_gate.py`:

```bash
# Run safety gate tests
pytest tests/governance/test_safety_gate.py -v

# Run demo
python examples/safety_gate_integration_demo.py
```

**Test Coverage:**
- ✓ JSON serialization (all outputs are JSON-safe)
- ✓ Determinism (same inputs → same outputs)
- ✓ No mutation (inputs never modified)
- ✓ Status light mapping (PASS/WARN/BLOCK → GREEN/YELLOW/RED)
- ✓ Reason ordering (alphabetically sorted)
- ✓ Evidence attachment (top 3 reasons)

## Design Principles

1. **Behavior-Preserving**: No new gating logic—only surfacing existing decisions
2. **Deterministic**: Same inputs always produce identical outputs
3. **Non-Mutating**: All functions return new data structures
4. **JSON-Safe**: All outputs are directly serializable
5. **Shadow Mode**: Global health integration doesn't affect other tiles

## Integration Checklist

When integrating safety gate into a new runner (U2, RFL, etc.):

- [ ] Collect `SafetyGateDecision` objects during run
- [ ] Build `SafetyEnvelope` at run completion
- [ ] Add to First Light summary via `build_safety_gate_summary_for_first_light()`
- [ ] Update global health via `build_global_health_surface()`
- [ ] Attach to evidence via `attach_safety_gate_to_evidence()`
- [ ] Verify determinism with repeated runs
- [ ] Add integration tests

## API Reference

### Functions

#### `build_safety_gate_summary_for_first_light(envelope: SafetyEnvelope) -> dict`
Returns a summary suitable for First Light output.

#### `build_safety_gate_tile_for_global_health(envelope: SafetyEnvelope) -> dict`
Returns a health tile with traffic light status.

#### `attach_safety_gate_to_evidence(evidence: dict, envelope: SafetyEnvelope) -> dict`
Returns new evidence dict with safety gate attached. Never mutates input.

#### `build_global_health_surface(tiles: dict, safety_envelope: Optional[SafetyEnvelope]) -> dict`
Builds complete health surface. If `safety_envelope` is None, safety tile is omitted.

## Future Work

- Integration with U2Runner and RFLRunner
- Real-time safety gate monitoring dashboard
- Historical trend analysis
- Alert thresholds and notifications
- Integration with SLO monitoring

## Questions?

See `examples/safety_gate_integration_demo.py` for a working example, or contact the Phase X Neural Link team.
