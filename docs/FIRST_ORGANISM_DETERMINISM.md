# First Organism Determinism Specification

> **Cryptographic Constraint**: The First Organism closed loop MUST produce
> byte-for-byte identical outputs when run with the same seed.

## Overview

The First Organism is the foundational integration test for MathLedger's
Reflexive Formal Learning (RFL) system. It exercises the complete path:

```
UI Event → Curriculum Gate → Derivation → Lean Verify (abstention) →
Dual-Attest seal H_t → RFL runner metabolism
```

This document specifies the determinism requirements and APIs for each stage.

## Cryptographic Invariants

### 1. No Hidden Entropy

The following sources of nondeterminism are **banned** in the First Organism path:

| Banned Pattern | Replacement |
|----------------|-------------|
| `datetime.now()` | `deterministic_timestamp_from_content()` |
| `datetime.utcnow()` | `deterministic_timestamp_from_content()` |
| `time.time()` | `deterministic_unix_timestamp()` |
| `uuid.uuid4()` | `deterministic_uuid()` or `deterministic_run_id()` |
| `random.*` | `SeededRNG(seed)` |
| `numpy.random.*` | `SeededRNG(seed)` with numpy backend |
| SQL `NOW()` | Content-derived timestamp parameter |
| SQL `CURRENT_TIMESTAMP` | Content-derived timestamp parameter |

### 2. RFC 8785 Canonical JSON

All JSON serialization in the First Organism path MUST use RFC 8785 canonical form:

```python
def rfc8785_canonicalize(obj: Any) -> str:
    return json.dumps(
        obj,
        separators=(",", ":"),
        sort_keys=True,
        ensure_ascii=True,
    )
```

Properties guaranteed:
- Keys sorted lexicographically at all nesting levels
- No whitespace between tokens
- ASCII-only output (non-ASCII characters escaped as `\uXXXX`)
- Consistent float representation

### 3. H_t Recomputability

The composite attestation root H_t MUST be recomputable from stored R_t and U_t:

```
H_t = SHA256(R_t || U_t)
```

Where:
- `R_t`: Reasoning Merkle root (over proof hashes)
- `U_t`: UI Merkle root (over UI event leaf hashes)
- `||`: Concatenation of hex strings

## Deterministic APIs

### `deterministic_ui_event(seed, payload)`

Creates a deterministic UI event with content-derived ID and timestamp.

```python
from backend.repro.first_organism_harness import deterministic_ui_event

event = deterministic_ui_event(
    seed=42,
    payload={"action": "toggle_abstain", "statement_hash": "abc123..."},
)

# Returns:
# DeterministicUIEvent(
#     event_id="ui-event-7f3c...",
#     timestamp="2025-01-01T00:00:42+00:00",
#     canonical_json='{"event_id":"ui-event-7f3c...","event_type":"select_statement",...}',
#     leaf_hash="sha256_of_canonical_payload",
#     payload={...},
# )
```

**Determinism Guarantee**: Same `seed` + `payload` → identical output.

### `deterministic_gate_verdict(seed, slice_name, metrics, gate_statuses, advance, reason)`

Creates a deterministic curriculum gate verdict.

```python
from backend.repro.first_organism_harness import deterministic_gate_verdict

verdict = deterministic_gate_verdict(
    seed=42,
    slice_name="first-organism-pl",
    metrics={"coverage": {"ci_lower": 0.95}},
    gate_statuses=[{"gate": "coverage", "passed": True}],
    advance=True,
    reason="all gates passed",
)

# Returns:
# DeterministicGateVerdict(
#     advance=True,
#     reason="all gates passed",
#     audit_json='{"active_slice":"first-organism-pl",...}',
#     audit_hash="sha256_of_audit_json",
#     timestamp="2025-01-01T00:00:42+00:00",
#     gate_statuses=(...),
# )
```

### `deterministic_derivation_result(seed, slice_name, status, n_candidates, n_abstained, abstained_hashes)`

Creates a deterministic derivation result.

```python
from backend.repro.first_organism_harness import deterministic_derivation_result

result = deterministic_derivation_result(
    seed=42,
    slice_name="first-organism-pl",
    status="abstain",
    n_candidates=10,
    n_abstained=3,
    abstained_hashes=["h1", "h2", "h3"],
)

# Returns:
# DeterministicDerivationResult(
#     run_id="derive-a1b2c3d4e5f6",
#     status="abstain",
#     n_candidates=10,
#     n_abstained=3,
#     abstained_hashes=("h1", "h2", "h3"),  # Sorted!
#     canonical_json='...',
#     result_hash="sha256_of_canonical_json",
# )
```

**Note**: `abstained_hashes` are automatically sorted for determinism.

### `deterministic_seal(seed, derivation_result, ui_events)`

Creates a deterministic dual-root attestation seal.

```python
from backend.repro.first_organism_harness import deterministic_seal

seal = deterministic_seal(
    seed=42,
    derivation_result=derivation_result,
    ui_events=[ui_event],
)

# Returns:
# DeterministicSealResult(
#     block_id="block-7f3c8e9d...",
#     reasoning_root="R_t: sha256 of reasoning leaves",
#     ui_root="U_t: sha256 of UI leaves",
#     composite_root="H_t: SHA256(R_t || U_t)",
#     sealed_at="2025-01-01T00:00:42+00:00",
#     attestation_json='...',
#     attestation_hash="sha256_of_attestation_json",
# )
```

**Invariant**: `composite_root == SHA256(reasoning_root + ui_root)`

### `deterministic_rfl_step(seed, seal_result, derivation_result, slice_name)`

Executes a deterministic RFL metabolism step.

```python
from backend.repro.first_organism_harness import deterministic_rfl_step

step = deterministic_rfl_step(
    seed=42,
    seal_result=seal,
    derivation_result=derivation_result,
    slice_name="first-organism-pl",
)

# Returns:
# DeterministicRflStep(
#     step_id="sha256_of_step_material",
#     policy_update_applied=True,
#     abstention_mass_delta=1.5,
#     symbolic_descent=-0.15,
#     ledger_entry_json='...',
#     ledger_entry_hash="sha256_of_ledger_entry_json",
# )
```

### `run_first_organism_deterministic(seed, **kwargs)`

Executes the complete First Organism closed loop.

```python
from backend.repro.first_organism_harness import run_first_organism_deterministic

result = run_first_organism_deterministic(seed=42)

# Returns:
# FirstOrganismResult(
#     seed=42,
#     ui_event=DeterministicUIEvent(...),
#     gate_verdict=DeterministicGateVerdict(...),
#     derivation_result=DeterministicDerivationResult(...),
#     seal_result=DeterministicSealResult(...),
#     rfl_step=DeterministicRflStep(...),
#     composite_root="H_t",
#     run_hash="sha256_of_entire_run",
# )
```

#### High-Complexity Feedback

The `slice_name` argument accepts any curriculum slice that exists in `config/curriculum.yaml`, including the newly added `first_organism_pl2_hard`. Calling the harness with this slice exercises the deterministic abstention path while still producing deterministic artifacts:

```python
result = run_first_organism_deterministic(
    seed=42,
    slice_name="first_organism_pl2_hard",
)

assert result.gate_verdict.advance is False
assert "abstention gate" in result.gate_verdict.reason
```

The gate verdict mirrors the thresholds defined for `first_organism_pl2_hard` (coverage 0.90, abstention 20%, velocity 120 pph), which allows the harness to report the run as allowed but not ratchet forward.

## Running the Determinism Test

### Local Development

```bash
# Run all determinism tests
pytest tests/integration/test_first_organism_determinism.py -v

# Run only bitwise reproducibility tests
pytest tests/integration/test_first_organism_determinism.py -k bitwise -v

# Run with coverage
pytest tests/integration/test_first_organism_determinism.py --cov=backend.repro
```

### CI Integration

Add to your CI workflow:

```yaml
- name: Run First Organism Determinism Tests
  run: |
    pytest tests/integration/test_first_organism_determinism.py \
      -v \
      --tb=short \
      -x  # Stop on first failure
```

### Verifying Determinism Manually

```python
from backend.repro.first_organism_harness import verify_determinism

# Runs the pipeline 5 times and verifies all outputs are identical
assert verify_determinism(seed=42, runs=5)
```

## Allowed Randomness

The following are **allowed** because they are explicitly seeded:

1. **SeededRNG**: `SeededRNG(seed)` from `backend.repro.determinism`
2. **NumPy with seed**: `np.random.seed(config.random_seed)` in RFL runner
3. **Bootstrap resampling**: Uses `random_state` parameter

## Debugging Nondeterminism

If the determinism test fails, use this debugging approach:

1. **Isolate the stage**: Run each stage separately with the same seed
2. **Compare canonical JSON**: Diff the `canonical_json` fields
3. **Check for banned patterns**: Search for `datetime.now`, `time.time`, etc.
4. **Verify hash inputs**: Print the inputs to each hash function

Example debugging script:

```python
from backend.repro.first_organism_harness import run_first_organism_deterministic

result1 = run_first_organism_deterministic(42)
result2 = run_first_organism_deterministic(42)

# Compare each stage
assert result1.ui_event.canonical_json == result2.ui_event.canonical_json, "UI event differs"
assert result1.gate_verdict.audit_json == result2.gate_verdict.audit_json, "Gate verdict differs"
assert result1.derivation_result.canonical_json == result2.derivation_result.canonical_json, "Derivation differs"
assert result1.seal_result.attestation_json == result2.seal_result.attestation_json, "Seal differs"
assert result1.rfl_step.ledger_entry_json == result2.rfl_step.ledger_entry_json, "RFL step differs"
```

## Migration Guide

### Replacing datetime.now()

Before:
```python
timestamp = datetime.now(timezone.utc).isoformat()
```

After:
```python
from backend.repro.determinism import deterministic_isoformat

timestamp = deterministic_isoformat(seed, content_hash, resolution="seconds")
```

### Replacing uuid.uuid4()

Before:
```python
run_id = str(uuid.uuid4())
```

After:
```python
from backend.repro.determinism import deterministic_run_id

run_id = deterministic_run_id("run", seed, content_hash, length=12)
```

### Replacing random.choice()

Before:
```python
import random
selected = random.choice(candidates)
```

After:
```python
from backend.repro.determinism import SeededRNG

rng = SeededRNG(seed)
selected = rng.choice(candidates, size=1)[0]
```

## References

- [RFC 8785: JSON Canonicalization Scheme](https://www.rfc-editor.org/rfc/rfc8785)
- [MathLedger Whitepaper §4.2: Dual Root Attestation](../docs/whitepaper.md)
- [First Organism Integration Test](../tests/integration/test_first_organism.py)
- [Determinism Helpers Module](../backend/repro/determinism.py)

