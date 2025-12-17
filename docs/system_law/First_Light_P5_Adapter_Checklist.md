# First Light P5 Real Telemetry Adapter Checklist

**Document Version:** 1.1.0
**Status:** Design specification only — no implementation yet
**Mode:** SHADOW (observation-only contract preserved)

This checklist defines the requirements for implementing `RealTelemetryAdapter` for Phase X P5.

---

## Overview

The `RealTelemetryAdapter` replaces `MockTelemetryProvider` in the P4 harness to connect the shadow twin to actual runner telemetry. The adapter must:
- Provide real observations to the P4 runner
- Maintain SHADOW MODE invariants
- Enable meaningful divergence measurement
- Support deterministic snapshot hashing

---

## 1. Interface Contract

### 1.1 Required Observation Fields

The adapter must provide observations with these exact fields:

| Field | Type | Range/Format | Description | Required |
|-------|------|--------------|-------------|----------|
| `cycle` | int | >= 0 | Monotonically increasing cycle identifier | Yes |
| `timestamp` | str | ISO8601 | Observation timestamp (UTC) | Yes |
| `H` | float | [0.0, 1.0] | Entropy/uncertainty metric | Yes |
| `rho` | float | [0.0, 1.0] | Density/concentration metric | Yes |
| `tau` | float | [0.0, 1.0] | Temperature/volatility metric | Yes |
| `beta` | float | [0.0, 1.0] | Beta coefficient (learning rate proxy) | Yes |
| `success` | bool | true/false | Whether the real runner succeeded this cycle | Yes |
| `blocked` | bool | true/false | Whether the real runner was blocked | Yes |
| `in_omega` | bool | true/false | Whether real runner is in safe region (Ω) | Yes |
| `hard_ok` | bool | true/false | Whether HARD mode check passed | Yes |
| `rsi` | float | [0.0, 1.0] | Real stability index | Yes |
| `abstention` | bool | true/false | Whether real runner abstained | Yes |
| `mode` | str | "SHADOW" | Must always be "SHADOW" for P5 | Yes |
| `source` | str | "REAL_RUNNER" | Identifies telemetry source | Yes |
| `metrics` | dict | {} | Additional telemetry metrics | Optional |

### 1.2 Required Invariants

The adapter must enforce these invariants at all times:

```
INVARIANT-1: mode == "SHADOW"
    All returned observations must have mode="SHADOW".
    Violation: Reject observation, raise RuntimeError.

INVARIANT-2: READ-ONLY
    Adapter has no control surfaces. It cannot:
    - Send commands to the real runner
    - Modify runner state
    - Trigger governance actions
    - Abort or stop the runner
    Violation: Methods must raise RuntimeError("SHADOW MODE VIOLATION").

INVARIANT-3: DETERMINISTIC HASHING
    Given the same observation data, the adapter must produce
    identical hash values for snapshot comparison.
    Hash function: SHA-256 over canonicalized JSON (sorted keys, no whitespace).
```

### 1.3 Observation Schema (JSON)

```json
{
  "cycle": 42,
  "timestamp": "2025-12-11T12:00:00.000000+00:00",
  "H": 0.65,
  "rho": 0.82,
  "tau": 0.20,
  "beta": 0.15,
  "success": true,
  "blocked": false,
  "in_omega": true,
  "hard_ok": true,
  "rsi": 0.78,
  "abstention": false,
  "mode": "SHADOW",
  "source": "REAL_RUNNER",
  "metrics": {}
}
```

---

## 2. Validation Steps (Pre-P5)

### 2.1 Test Adapter Against MockTelemetryProvider Traces

Run an A/B comparison to verify adapter compatibility:

```bash
# Step 1: Generate mock trace
uv run python scripts/usla_first_light_p4_harness.py \
    --cycles 100 --seed 42 \
    --telemetry-adapter mock \
    --output-dir results/validation/mock_trace

# Step 2: Run adapter with same harness (adapter must exist)
uv run python scripts/usla_first_light_p4_harness.py \
    --cycles 100 \
    --telemetry-adapter real \
    --adapter-config config/real_telemetry_adapter.json \
    --output-dir results/validation/real_trace
```

**Acceptance criteria:**
- [ ] Adapter does not crash
- [ ] All output files are created (6 artifacts)
- [ ] Logs are in expected JSONL format
- [ ] All observations have `mode="SHADOW"`
- [ ] All observations have required fields

### 2.2 Short P4 Run Sanity Check

Run a short P4 cycle with a trivial/easy runner to verify divergence is not 97%:

```python
def test_adapter_divergence_sanity():
    """
    Sanity check: With a trivial test case (synthetic easy runner),
    divergence should NOT be ~97%.

    If divergence is still ~97% with an easy case, the adapter
    is likely not connected correctly.
    """
    result = run_p4_short(cycles=50, adapter="real", easy_mode=True)

    # Divergence should be significantly lower than noise baseline
    assert result.divergence_rate < 0.80, (
        f"Divergence {result.divergence_rate:.2f} too close to noise baseline. "
        "Check adapter connection."
    )
```

### 2.3 SHADOW Invariants Verification

```python
def test_shadow_invariants_hold():
    """Verify SHADOW invariants are preserved with real adapter."""
    adapter = RealTelemetryAdapter(config, shadow_mode=True)

    # INVARIANT-1: mode must be SHADOW
    obs = adapter.get_observation()
    assert obs.mode == "SHADOW"

    # INVARIANT-2: READ-ONLY (mutation methods raise)
    with pytest.raises(RuntimeError, match="SHADOW MODE VIOLATION"):
        adapter.set_state({})
    with pytest.raises(RuntimeError, match="SHADOW MODE VIOLATION"):
        adapter.abort()
    with pytest.raises(RuntimeError, match="SHADOW MODE VIOLATION"):
        adapter.enforce_policy({})

    # INVARIANT-3: Deterministic hashing
    obs1 = adapter.get_observation()
    obs2 = adapter.get_observation()
    # Same cycle data should hash identically
    assert hash_observation(obs1) == hash_observation(obs1)
```

---

## 3. P5 Readiness Criteria

Before running a full 1000-cycle P5 golden run, ALL of the following must be true:

### 3.1 Schema & Interface Checks

- [ ] Adapter provides all 14 required fields
- [ ] All field types match specification
- [ ] All field ranges are valid (floats in [0,1], etc.)
- [ ] `mode="SHADOW"` on every observation
- [ ] `source="REAL_RUNNER"` on every observation

### 3.2 SHADOW Invariants Verified

- [ ] Adapter constructor requires `shadow_mode=True`
- [ ] Adapter raises `ValueError` on `shadow_mode=False`
- [ ] All 5 mutation methods raise `RuntimeError`
- [ ] No governance API calls in adapter code (static analysis)
- [ ] Existing 14 SHADOW MODE compliance tests pass

### 3.3 Short Test Runs Show Reasonable Divergence

- [ ] 50-cycle test run completes without error
- [ ] Divergence rate < 0.80 (significantly below noise baseline)
- [ ] No "severe" divergences (all should be minor/moderate)
- [ ] Twin predictions are not all identical (variance check)

### 3.4 Connection & Stability

- [ ] Adapter connects to real runner telemetry
- [ ] Adapter handles transient disconnects gracefully
- [ ] Adapter logs connection state changes
- [ ] Health check endpoint responds

**Definition of "reasonable divergence" for first pass:**
- Divergence < 0.80 (at least 20% below noise baseline of 0.97)
- This is a sanity check, not a quality bar
- Actual P5 target is < 0.30 (acceptable alignment)

---

## 4. Post-P5 Update Procedure

Once the real-telemetry P5 golden run is complete, the following documents MUST be updated:

### 4.1 Update Checklist

```
[ ] 1. Golden Run Summary: Update P3 + P4 metrics
       - File: docs/system_law/First_Light_Golden_Run_Summary.md
       - Update all metric tables with P5 values
       - Add "P5 Real Telemetry Results" section
       - Compare P5 divergence to mock baseline (0.97)

[ ] 2. External FAQ: Update example outputs
       - File: docs/system_law/First_Light_External_FAQ.md
       - Update divergence rate in Q&A
       - Add P5-specific Q&A if new issues discovered
       - Update "noise baseline" section with comparison

[ ] 3. Cross-Platform Validation: Mark P5 cells
       - File: docs/system_law/First_Light_External_Verification.md
       - Update Section 8.6 matrix with P5 validation status
       - Document any OS-specific issues with real adapter

[ ] 4. Status JSON: Add telemetry_source field
       - File: scripts/generate_first_light_status.py
       - Add "telemetry_source": "real" field
       - Bump schema_version to 1.1.0

[ ] 5. Evidence Pack: Archive with correct naming
       - Create: results/first_light/evidence_pack_first_light_real/
       - Verify manifest.json has correct metadata
       - Run integrity verification
```

### 4.2 Metrics to Update

| Document | Metric | Mock Value | P5 Value (TBD) |
|----------|--------|------------|----------------|
| Golden Run Summary | P4 divergence_rate | 0.972 | ___ |
| Golden Run Summary | P4 twin_success_accuracy | 0.886 | ___ |
| Golden Run Summary | P4 omega_prediction_accuracy | 0.973 | ___ |
| External FAQ | Divergence rate in examples | 97% | ___ |
| Status JSON | telemetry_source | "mock" | "real" |

### 4.3 Comparison Report

Generate a mock-vs-real comparison report:

```bash
uv run python scripts/compare_first_light_runs.py \
    --mock-pack results/first_light/evidence_pack_first_light_mock \
    --real-pack results/first_light/evidence_pack_first_light_real \
    --output results/first_light/p5_comparison_report.json
```

This script will be created as part of P5 implementation.

---

## 5. Verification Artifacts

After P5 implementation, the following artifacts must exist:

| Artifact | Location | Purpose |
|----------|----------|---------|
| Adapter implementation | `backend/topology/first_light/real_telemetry_adapter.py` | Core adapter code |
| Adapter tests | `tests/topology/first_light/test_real_telemetry_adapter.py` | Unit + integration tests |
| Connection config | `config/real_telemetry_adapter.json` | Production config template |
| P5 evidence pack | `results/first_light/evidence_pack_first_light_real/` | Real telemetry golden run |

---

## 6. SHADOW MODE Compliance Checklist

Before merging adapter implementation:

- [ ] Adapter constructor requires `shadow_mode=True`
- [ ] Adapter raises on `shadow_mode=False`
- [ ] All mutation methods raise `RuntimeError`
- [ ] All observations have `mode="SHADOW"`
- [ ] No governance API calls in adapter code
- [ ] Static analysis passes (grep for forbidden patterns)
- [ ] Existing 14 SHADOW MODE compliance tests still pass

---

## 7. Transition from Mock to Real

When executing the P5 golden run:

```bash
# Current (mock) - no change to command
uv run python scripts/usla_first_light_p4_harness.py \
    --cycles 1000 --seed 42 ...

# P5 (real) - add adapter flag (design only, flag not yet implemented)
uv run python scripts/usla_first_light_p4_harness.py \
    --cycles 1000 \
    --telemetry-adapter real \
    --adapter-config config/real_telemetry_adapter.json \
    ...
```

The harness will need a `--telemetry-adapter` flag to select between mock and real. This is a future implementation task.

---

**Document generated:** 2025-12-11
**Implementation status:** Design only — no code created
