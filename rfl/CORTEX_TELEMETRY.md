# Cortex Telemetry — Neural Link Integration

**Status:** ✅ COMPLETE  
**Author:** rfl-policy-engineer  
**Date:** 2025-12-11

## Mission

Surface Cortex (hard gate + TDA) decisions into:
- First Light summaries
- Uplift Safety Engine v6
- Evidence Packs

**No new gating** — only telemetry. Cortex owns the gate; this module only exposes outcomes.

---

## Architecture

### 1. Core Data Structures

#### `TDAMode` Enum
```python
TDAMode.BLOCK      # Hard block on violations
TDAMode.DRY_RUN    # Log violations but don't block
TDAMode.SHADOW     # Silent monitoring only
```

#### `HardGateStatus` Enum
```python
HardGateStatus.OK     # No violations detected
HardGateStatus.WARN   # Advisory violations (DRY_RUN/SHADOW)
HardGateStatus.BLOCK  # Hard blocking violations (BLOCK mode)
```

#### `CortexDecision` Dataclass
Single Cortex gate decision record:
```python
@dataclass
class CortexDecision:
    decision_id: str
    cycle_id: Optional[int]
    item: str
    blocked: bool
    advisory: bool
    rationale: str
    tda_mode: TDAMode
    timestamp: Optional[str]
    metadata: Dict[str, Any]
```

#### `CortexEnvelope` Dataclass
Envelope of Cortex decisions for a run:
```python
@dataclass
class CortexEnvelope:
    tda_mode: TDAMode
    decisions: List[CortexDecision]
    metadata: Dict[str, Any]
    
    # Key methods:
    def total_decisions() -> int
    def blocked_decisions() -> int
    def advisory_decisions() -> int
    def compute_hard_gate_status() -> HardGateStatus
```

---

## Integration Points

### 1. First Light Summary

**Function:** `compute_cortex_summary(envelope: CortexEnvelope) -> Dict`

**Output Structure:**
```json
{
  "cortex_summary": {
    "total_decisions": 100,
    "blocked_decisions": 2,
    "advisory_decisions": 5,
    "tda_mode": "DRY_RUN",
    "hard_gate_status": "WARN"
  }
}
```

**Hard Gate Status Logic:**
- `BLOCK`: If BLOCK mode AND blocked_decisions > 0
- `WARN`: If (DRY_RUN or SHADOW) AND any violations
- `OK`: Otherwise

**Usage in RFLRunner:**
```python
# RFLRunner initialization
self.cortex_envelope = CortexEnvelope(tda_mode=TDAMode.DRY_RUN)

# Export results
results = {
    # ... other fields ...
    "cortex_summary": compute_cortex_summary(self.cortex_envelope)["cortex_summary"]
}
```

---

### 2. Uplift Safety Engine Hook

**Class:** `UpliftSafetyCortexAdapter`

**Factory Method:** `UpliftSafetyCortexAdapter.from_envelope(envelope: CortexEnvelope)`

**Output Structure:**
```json
{
  "cortex_gate_band": "MEDIUM",
  "hypothetical_block_rate": 0.15,
  "advisory_only": true
}
```

**Band Logic:**
- `HIGH`: blocked_decisions > 0 (in BLOCK mode)
- `MEDIUM`: advisory_decisions > 0 (any mode)
- `LOW`: no violations

**Hypothetical Block Rate:**
```python
hypothetical_block_rate = (blocked + advisory) / total
```

**Usage in RFLRunner:**
```python
results = {
    "uplift": {
        "bootstrap_ci": self.uplift_ci.to_dict() if self.uplift_ci else None,
        "safety_cortex_adapter": UpliftSafetyCortexAdapter.from_envelope(self.cortex_envelope).to_dict()
    }
}
```

**Integration with Uplift Safety Tensor:**
The adapter provides an **advisory weight only** that can be incorporated into uplift safety tensor construction. It does NOT alter core uplift computation.

---

### 3. Evidence Pack Governance Tile

**Function:** `attach_cortex_governance_to_evidence(evidence: Dict, envelope: CortexEnvelope) -> Dict`

**Output Structure:**
```json
{
  "version": "1.0.0",
  "experiment": { ... },
  "governance": {
    "cortex_gate": {
      "hard_gate_status": "BLOCK",
      "blocked_decisions": 2,
      "tda_mode": "BLOCK",
      "rationales": [
        "Hard violation: item1 exceeds safety threshold",
        "Hard violation: item2 failed validation",
        "Advisory: item3 approaching threshold"
      ],
      "total_decisions": 10,
      "advisory_decisions": 1
    }
  }
}
```

**Key Properties:**
- **No mutation:** Original evidence dict is NOT mutated (deep copy used)
- **Rationale limit:** Top 3 rationales included
- **Preserves existing governance:** Adds cortex_gate without overwriting other keys

**Usage in Evidence Pack Toolchain:**
```python
from rfl.cortex_telemetry import attach_cortex_governance_to_evidence, CortexEnvelope

toolchain = EvidencePackToolchain(repo_root)
toolchain.attach_cortex_governance(evidence_path, cortex_envelope)
```

---

## Configuration

### Environment Variables

**`CORTEX_TDA_MODE`**  
Controls TDA gating mode for RFLRunner.

**Values:**
- `BLOCK` (default if in production)
- `DRY_RUN` (default for development)
- `SHADOW`

**Example:**
```bash
export CORTEX_TDA_MODE=DRY_RUN
```

---

## Testing

**Total Tests:** 41 (28 unit + 13 integration)  
**Status:** ✅ All passing

### Test Categories

1. **Unit Tests** (`tests/rfl/test_cortex_telemetry.py`)
   - TDA mode and hard gate status enums
   - CortexDecision creation and serialization
   - CortexEnvelope decision tracking
   - Hard gate status computation logic
   - CortexSummary generation
   - UpliftSafetyCortexAdapter band logic
   - Evidence pack governance attachment
   - Determinism verification
   - JSON compatibility
   - No-mutation guarantees

2. **Integration Tests** (`tests/rfl/test_cortex_integration.py`)
   - First Light summary generation
   - Uplift Safety adapter integration
   - Evidence Pack governance tiles
   - End-to-end flow: Cortex → First Light → Uplift Safety → Evidence Pack
   - Determinism across complete flow
   - Invariant maintenance (no gating semantics alteration)

### Running Tests

```bash
# Run all Cortex telemetry tests
python3 -m pytest tests/rfl/test_cortex_telemetry.py -v
python3 -m pytest tests/rfl/test_cortex_integration.py -v

# Run all RFL tests
python3 -m pytest tests/rfl/ -v
```

---

## Design Principles

### 1. Read-Only Access
Cortex telemetry is **read-only**. It surfaces decisions but does NOT:
- Alter gating semantics
- Modify Cortex decisions
- Change TDA mode
- Mutate input data structures

**Cortex owns the gate. Telemetry only exposes.**

### 2. Determinism
All telemetry operations are **deterministic**:
- Same envelope → same summary
- Same envelope → same adapter
- Same envelope → same evidence governance

No randomness. No wall-clock time. No external entropy.

### 3. JSON Compatibility
All data structures are **JSON-serializable**:
- Use `to_dict()` methods for serialization
- All enums use `.value` for string representation
- No numpy arrays or non-JSON types

### 4. No Breaking Changes
Integration preserves existing behavior:
- RFLRunner exports unchanged (only adds cortex_summary)
- Evidence packs preserve existing governance
- Uplift safety adds advisory weight only

---

## Example Usage

### Full Integration Example

```python
from rfl.cortex_telemetry import (
    CortexEnvelope,
    CortexDecision,
    TDAMode,
    compute_cortex_summary,
    UpliftSafetyCortexAdapter,
    attach_cortex_governance_to_evidence,
)

# 1. Create Cortex envelope
envelope = CortexEnvelope(tda_mode=TDAMode.DRY_RUN)

# 2. Add decisions (would be done by Cortex gate in practice)
for i in range(10):
    envelope.add_decision(CortexDecision(
        decision_id=f"d{i:03d}",
        cycle_id=i,
        item=f"item_{i}",
        blocked=False,
        advisory=(i < 2),
        rationale=f"Advisory {i}" if i < 2 else "OK",
        tda_mode=TDAMode.DRY_RUN,
    ))

# 3. Generate First Light summary
first_light = compute_cortex_summary(envelope)
print(first_light["cortex_summary"])
# {
#   "total_decisions": 10,
#   "blocked_decisions": 0,
#   "advisory_decisions": 2,
#   "tda_mode": "DRY_RUN",
#   "hard_gate_status": "WARN"
# }

# 4. Generate Uplift Safety adapter
adapter = UpliftSafetyCortexAdapter.from_envelope(envelope)
print(adapter.to_dict())
# {
#   "cortex_gate_band": "MEDIUM",
#   "hypothetical_block_rate": 0.2,
#   "advisory_only": true
# }

# 5. Attach to Evidence Pack
evidence = {"version": "1.0.0", "experiment": {"id": "test"}}
evidence_with_cortex = attach_cortex_governance_to_evidence(evidence, envelope)
print(evidence_with_cortex["governance"]["cortex_gate"])
# {
#   "hard_gate_status": "WARN",
#   "blocked_decisions": 0,
#   "tda_mode": "DRY_RUN",
#   "rationales": ["Advisory 0", "Advisory 1"],
#   "total_decisions": 10,
#   "advisory_decisions": 2
# }
```

---

## Security & Safety

### Guardrails

1. **No External Entropy**
   - All telemetry operations are deterministic
   - No wall-clock time in decision logic
   - SeededRNG not required (pure data transformation)

2. **No Side Effects**
   - Telemetry never modifies Cortex state
   - Input envelopes are read-only
   - Evidence packs use deep copy to avoid mutation

3. **No Proxy Metrics**
   - Only formal verification outcomes (proof success/failure)
   - No human preference signals
   - No unverified feedback

### Verification

- All operations covered by determinism tests
- JSON compatibility verified for all outputs
- No-mutation guarantees tested
- Correct state mapping verified

---

## Future Extensions

### When Cortex is Fully Implemented

1. **Populate CortexEnvelope in RFLRunner:**
   ```python
   # In RFLRunner._execute_experiments()
   cortex_decision = evaluate_hard_gate_decision(item, context)
   self.cortex_envelope.add_decision(cortex_decision)
   ```

2. **Hook Uplift Safety Adapter:**
   ```python
   # In uplift safety tensor construction
   cortex_adapter = UpliftSafetyCortexAdapter.from_envelope(self.cortex_envelope)
   safety_tensor = build_safety_tensor(
       base_metrics=base_metrics,
       cortex_weight=cortex_adapter.to_dict()  # advisory only
   )
   ```

3. **Auto-attach to Evidence Packs:**
   ```python
   # In evidence pack creation
   if hasattr(runner, 'cortex_envelope'):
       evidence = attach_cortex_governance_to_evidence(
           evidence,
           runner.cortex_envelope
       )
   ```

---

## API Reference

### Functions

#### `compute_cortex_summary(envelope: CortexEnvelope) -> Dict[str, Any]`
Generate First Light cortex_summary from envelope.

**Returns:** `{"cortex_summary": {...}}`

#### `attach_cortex_governance_to_evidence(evidence: Dict, envelope: CortexEnvelope) -> Dict`
Attach Cortex governance tile to evidence pack.

**Returns:** New evidence dict with cortex_gate added (original unchanged)

### Classes

#### `CortexEnvelope`
- `total_decisions() -> int`
- `blocked_decisions() -> int`
- `advisory_decisions() -> int`
- `compute_hard_gate_status() -> HardGateStatus`
- `to_dict() -> Dict[str, Any]`

#### `CortexSummary`
- `from_envelope(envelope: CortexEnvelope) -> CortexSummary`
- `to_dict() -> Dict[str, Any]`

#### `UpliftSafetyCortexAdapter`
- `from_envelope(envelope: CortexEnvelope) -> UpliftSafetyCortexAdapter`
- `to_dict() -> Dict[str, Any]`

#### `CortexDecision`
- `to_dict() -> Dict[str, Any]`

---

## Change Log

### 2025-12-11: Initial Implementation
- Created `rfl/cortex_telemetry.py` with core data structures
- Integrated into RFLRunner for First Light summaries
- Added Uplift Safety adapter
- Added Evidence Pack governance tile support
- Comprehensive tests (41 passing)
- Documentation complete

---

## Contact

**Owner:** rfl-policy-engineer  
**Scope:** RFL policy implementation only  
**Related:** First Light, Uplift Safety Engine v6, Evidence Packs
