# Neural Link Integration: Runtime Safety Enforcement

## Overview

The Neural Link integration wires the `evaluate_hard_gate_decision()` function into U2Runner and RFLRunner as a **BLOCKING** call, ensuring that NO candidate executes without Cortex approval.

## Architecture

### Safety Components

```
┌─────────────────────────────────────────────────────────────┐
│                     U2Runner (Body)                         │
│                                                             │
│  ┌────────────┐                                            │
│  │  Frontier  │                                            │
│  │   Queue    │                                            │
│  └──────┬─────┘                                            │
│         │                                                   │
│         │ pop_candidate()                                  │
│         ▼                                                   │
│  ┌─────────────────────────────────────┐                  │
│  │   BLOCKING GATE EVALUATION          │                  │
│  │                                     │                  │
│  │   evaluate_hard_gate_decision()     │◄────────────────┤
│  │                                     │  U2SafetyContext │
│  │   • Depth check                    │                  │
│  │   • Complexity check               │                  │
│  │   • SLO envelope check             │                  │
│  │   • TDA attitude integration       │                  │
│  │                                     │                  │
│  └──────────┬──────────────────────────┘                  │
│             │                                               │
│             │ decision                                      │
│             ▼                                               │
│     ┌───────────────┐                                      │
│     │   APPROVED?   │                                      │
│     └───┬───────┬───┘                                      │
│         │       │                                           │
│      YES│       │NO                                         │
│         │       │                                           │
│         ▼       ▼                                           │
│    ┌────────┐ ┌────────┐                                   │
│    │Execute │ │ Block  │                                   │
│    │        │ │& Skip  │                                   │
│    └────────┘ └────────┘                                   │
└─────────────────────────────────────────────────────────────┘
```

### Key Types

#### U2SafetyContext

Tracks runtime safety metrics:
- Total candidates evaluated
- Approval/rejection/abstention counts
- Safety SLO rates
- TDA attitude signals
- Safety violation count

```python
@dataclass
class U2SafetyContext:
    total_candidates_evaluated: int = 0
    total_approvals: int = 0
    total_rejections: int = 0
    total_abstentions: int = 0
    
    approval_rate: float = 0.0
    rejection_rate: float = 0.0
    abstention_rate: float = 0.0
    
    tda_attitudes: Dict[str, Any] = field(default_factory=dict)
    safety_violations: int = 0
    last_decision: Optional[GateDecision] = None
```

#### SafetyEnvelope

Contains decision metadata and compliance attestation:
- Gate decision (APPROVED/REJECTED/ABSTAINED)
- Candidate ID and cycle
- Decision reason and confidence
- SLO compliance status
- Provenance (version, seed)

```python
@dataclass
class SafetyEnvelope:
    decision: GateDecision
    candidate_id: str
    cycle: int
    reason: str
    confidence: float  # 0.0 to 1.0
    slo_compliant: bool
    slo_violations: Dict[str, Any]
    gate_version: str
    deterministic_seed: Optional[str]
```

## Integration Points

### U2Runner.run_cycle()

**Before Integration:**
```python
while budget_remaining > 0 and not self.frontier.is_empty():
    candidate = self.frontier.pop()
    
    # Execute candidate immediately
    success, result = execute_fn(candidate.item, cycle)
```

**After Integration:**
```python
while budget_remaining > 0 and not self.frontier.is_empty():
    candidate = self.frontier.pop()
    
    # BLOCKING CALL: Cortex approval
    safety_envelope = evaluate_hard_gate_decision(
        candidate=candidate.item,
        cycle=cycle,
        safety_context=self.safety_context,
        prng=self.safety_prng.for_path("gate", str(cycle)),
        max_depth=self.config.max_depth,
    )
    
    # Block if not approved
    if safety_envelope.decision != GateDecision.APPROVED:
        continue  # Skip this candidate
    
    # Execute only if approved
    success, result = execute_fn(candidate.item, cycle)
```

## Safety Gate Logic

### Decision Flow

1. **Depth Check**: Reject if `candidate.depth > max_depth`
2. **Complexity Check**: Reject if `len(str(candidate)) > max_complexity`
3. **SLO Protection**: Abstain probabilistically if rejection_rate > 0.5
4. **TDA Integration**: Reject if TDA attitude signal < 0.3 (placeholder)

### Determinism Guarantees

- All decisions use the `safety_prng` for tie-breaking
- Same inputs + same PRNG seed = same decision
- Decision logic is pure (no external state except safety_context)
- PRNG state is included in snapshots for reproducibility

## Snapshot Integration

Safety context is fully serializable:

```python
# Save
snapshot = SnapshotData(
    ...,
    safety_context=self.safety_context.to_dict(),
    ...
)

# Restore
if snapshot.safety_context:
    self.safety_context = U2SafetyContext.from_dict(snapshot.safety_context)
```

## Type Safety

All safety types use proper type hints:

```python
def evaluate_hard_gate_decision(
    candidate: Any,
    cycle: int,
    safety_context: U2SafetyContext,
    prng: DeterministicPRNG,
    max_depth: int = 10,
    max_complexity: float = 1000.0,
) -> SafetyEnvelope:
    ...
```

## Correctness Properties

### P1: Blocking Enforcement
**Property**: NO candidate executes without passing the gate.

**Proof**: The `evaluate_hard_gate_decision()` call happens **before** `execute_fn()`. If decision ≠ APPROVED, execution is skipped via `continue`.

### P2: Determinism
**Property**: Same inputs + same seed → same decision.

**Proof**:
1. All candidate features are deterministic (depth, complexity)
2. All thresholds are constant (max_depth, max_complexity)
3. PRNG is used only for tie-breaking in SLO protection
4. Same PRNG seed → same random values → same decision

### P3: No Side Effects
**Property**: Gate evaluation doesn't modify external state except safety_context.

**Proof**: 
1. Function signature shows only safety_context is mutable
2. All other operations are pure (comparisons, calculations)
3. safety_context mutations are local to the runner

### P4: Snapshot Consistency
**Property**: Restored runner produces same decisions as original.

**Proof**:
1. safety_context is serialized in snapshot
2. PRNG state is serialized in snapshot
3. Restored state = original state
4. By P2, same state → same decisions

## Testing Strategy

Tests cover:

1. **Basic blocking**: Gate blocks deep/complex candidates
2. **Determinism**: Same seed → same decision
3. **Context tracking**: Metrics are correctly updated
4. **Serialization**: Context survives save/restore
5. **Runner integration**: Runner respects gate decisions
6. **Validation**: Envelope integrity checks

## Future Extensions

### TDA Attitude Integration

Currently a placeholder. Future integration points:

```python
# In evaluate_hard_gate_decision()
tda_signal = safety_context.tda_attitudes.get("approval_signal", 1.0)
if tda_signal < 0.3:
    decision = GateDecision.REJECTED
    reason = "tda_risk_signal"
```

### RFLRunner Integration

RFLRunner will call safety gate before policy updates:

```python
def run_with_attestation(self, attestation: AttestedRunContext) -> RflResult:
    # ... existing code ...
    
    # Safety gate check before policy update
    safety_envelope = evaluate_hard_gate_decision(
        candidate=attestation.statement_hash,
        cycle=self.first_organism_runs_total,
        safety_context=self.safety_context,
        prng=self.safety_prng,
    )
    
    if safety_envelope.decision != GateDecision.APPROVED:
        # Block policy update
        return RflResult(policy_update_applied=False, ...)
    
    # ... apply policy update ...
```

## Migration Path

1. ✅ **Phase 1**: Add safety module with basic checks (COMPLETE)
2. ✅ **Phase 2**: Integrate into U2Runner.run_cycle() (COMPLETE)
3. ⏳ **Phase 3**: Integrate into RFLRunner.run_with_attestation()
4. ⏳ **Phase 4**: Add TDA attitude signal integration
5. ⏳ **Phase 5**: Add advanced SLO tracking and adaptation

## Compliance

### Sober Truth Principles

- ✅ Behavior-preserving: Only rejects candidates, doesn't change logic
- ✅ Deterministic: Uses PRNG for reproducibility
- ✅ Testable: Full test coverage of gate logic
- ✅ Transparent: Clear decision reasons in envelope
- ✅ No normative language: Pure safety enforcement

### Governance Alignment

- Does NOT modify basis/ (frozen modules)
- Does NOT touch governance docs
- Does NOT change experiment outputs
- DOES add safety controls as specified in STRATCOM directive
