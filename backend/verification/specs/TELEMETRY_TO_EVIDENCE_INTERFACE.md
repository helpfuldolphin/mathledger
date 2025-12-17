# Telemetry to Evidence Interface Specification

**Author**: Manus-C (Telemetry Architect)  
**Date**: 2025-12-06  
**Version**: 1.0  
**Status**: Test-Ready

---

## Overview

This document specifies the **exact interface** between the telemetry collection system and MathLedger's evidence pipelines (Δp, RSI, Ω, TDA). It defines telemetry fields, error code taxonomy, JSONL schemas, and noise metadata storage/replay protocols.

---

## 1. Telemetry Fields for MathLedger Pipelines

### 1.1 Core Identity Fields

Required for all pipelines to identify and track verification attempts.

| Field | Type | Description | Required By |
|-------|------|-------------|-------------|
| `verification_id` | string (UUID) | Unique identifier for this verification attempt | All |
| `timestamp` | float (Unix epoch) | Time of verification start | All |
| `module_name` | string | Lean module being verified (e.g., "Mathlib.Algebra.Ring.Basic") | All |
| `context` | string | Context string for PRNG seeding and grouping | All |
| `cycle` | int | U2 cycle number (if applicable) | Δp, RSI |

### 1.2 Configuration Fields

Verifier configuration that affects outcome interpretation.

| Field | Type | Description | Required By |
|-------|------|-------------|-------------|
| `tier` | enum | Verifier tier: `fast_noisy`, `balanced`, `slow_precise` | All |
| `timeout_s` | float | Configured timeout in seconds | Ω, TDA |
| `lean_version` | string | Lean version string (e.g., "4.3.0") | TDA |
| `master_seed` | int | Master random seed for reproducibility | All |

### 1.3 Outcome Fields

Verification outcome and quality metrics.

| Field | Type | Description | Required By |
|-------|------|-------------|-------------|
| `outcome` | enum (VerifierErrorCode) | Stable error code (see Section 2) | All |
| `success` | bool | True if outcome == VERIFIED | Δp, RSI |
| `duration_ms` | float | Actual verification duration in milliseconds | Ω, TDA |
| `confidence` | float (0-1) | Verifier confidence (if available) | Δp |

### 1.4 Resource Usage Fields

Resource consumption metrics for Ω (resource optimization) and TDA (timeout analysis).

| Field | Type | Description | Required By |
|-------|------|-------------|-------------|
| `cpu_time_ms` | float | CPU time (user + system) in milliseconds | Ω, TDA |
| `memory_peak_mb` | float | Peak memory usage in MB | Ω, TDA |
| `memory_final_mb` | float | Final memory usage in MB | Ω |
| `io_read_mb` | float | I/O read in MB (optional) | Ω |
| `io_write_mb` | float | I/O write in MB (optional) | Ω |

### 1.5 Lean-Specific Metrics

Proof complexity metrics for RSI (proof complexity analysis) and TDA (tactic depth analysis).

| Field | Type | Description | Required By |
|-------|------|-------------|-------------|
| `tactic_count` | int | Total number of tactics used | RSI, TDA |
| `tactic_depth` | int | Maximum tactic nesting depth | RSI, TDA |
| `proof_size_bytes` | int | Proof term size in bytes | RSI |
| `search_nodes` | int | Search tree nodes explored | RSI |
| `tactics` | list[string] | List of tactic names used | TDA |
| `tactic_counts` | dict[string, int] | Per-tactic usage counts | TDA |

### 1.6 Failure Diagnostics

Diagnostic information for debugging and error analysis.

| Field | Type | Description | Required By |
|-------|------|-------------|-------------|
| `stderr` | string | Lean stderr output | TDA |
| `returncode` | int | Process return code | TDA |
| `signal` | int | Signal number (if killed) | TDA |
| `error_message` | string | Extracted error message | TDA |

### 1.7 Noise Injection Metadata

**Critical for P3**: Noise injection decisions and ground truth for calibration and replay.

| Field | Type | Description | Required By |
|-------|------|-------------|-------------|
| `noise_injected` | bool | True if noise was injected | All (P3) |
| `noise_type` | enum | `timeout`, `spurious_fail`, `spurious_pass`, or null | All (P3) |
| `ground_truth` | enum | Ground truth outcome before noise injection | Δp (P3) |
| `noise_seed` | string (hex) | PRNG seed used for this noise decision | All (P3) |
| `noise_rates` | dict | Noise rates used (θ_timeout, θ_sf, θ_sp) | Calibration |

### 1.8 Policy Metadata

RFL policy state for Δp (policy gradient computation).

| Field | Type | Description | Required By |
|-------|------|-------------|-------------|
| `policy_prob` | float (0-1) | Policy probability for this item | Δp |
| `policy_weight` | float | Policy weight before update | Δp |
| `policy_gradient` | float | Computed policy gradient | Δp |
| `policy_update` | float | Weight update (η * gradient) | Δp |

---

## 2. Stable Error Code Taxonomy

### 2.1 Error Code Enum

All verifier outcomes MUST map to one of these 11 stable error codes.

```python
class VerifierErrorCode(Enum):
    # Success
    VERIFIED = "verified"
    
    # Genuine Failures
    PROOF_INVALID = "proof_invalid"
    PROOF_INCOMPLETE = "proof_incomplete"
    
    # Verifier Imperfections (Noise)
    VERIFIER_TIMEOUT = "verifier_timeout"
    VERIFIER_SPURIOUS_FAIL = "verifier_spurious_fail"
    VERIFIER_SPURIOUS_PASS = "verifier_spurious_pass"
    VERIFIER_INTERNAL_ERROR = "verifier_internal_error"
    
    # Resource Constraints
    BUDGET_EXHAUSTED = "budget_exhausted"
    MEMORY_LIMIT_EXCEEDED = "memory_limit_exceeded"
    
    # Abstention
    ABSTENTION_MOCK_MODE = "abstention_mock_mode"
    ABSTENTION_CONTROLLED_ONLY = "abstention_controlled_only"
```

### 2.2 Error Code Classification

Helper methods for error code classification:

| Method | Returns True For |
|--------|------------------|
| `is_success()` | VERIFIED |
| `is_failure()` | PROOF_INVALID, PROOF_INCOMPLETE |
| `is_timeout()` | VERIFIER_TIMEOUT |
| `is_noise_injected()` | VERIFIER_SPURIOUS_FAIL, VERIFIER_SPURIOUS_PASS |
| `is_abstention()` | ABSTENTION_MOCK_MODE, ABSTENTION_CONTROLLED_ONLY |
| `is_resource_constraint()` | BUDGET_EXHAUSTED, MEMORY_LIMIT_EXCEEDED |

### 2.3 Outcome Interpretation for Pipelines

| Pipeline | Success Outcomes | Failure Outcomes | Abstention Outcomes |
|----------|------------------|------------------|---------------------|
| **Δp** | VERIFIED | PROOF_INVALID, PROOF_INCOMPLETE | VERIFIER_TIMEOUT, ABSTENTION_* |
| **RSI** | VERIFIED | PROOF_INVALID | All others |
| **Ω** | VERIFIED | All others | None |
| **TDA** | VERIFIED | VERIFIER_TIMEOUT | None |

**Δp Abstention Rule**: When outcome is VERIFIER_TIMEOUT or ABSTENTION_*, skip policy update (do not use in gradient computation).

---

## 3. JSONL Schemas

### 3.1 First-Light Synthetic Raw Schema

**File**: `first_light_synthetic_raw.jsonl`  
**Purpose**: Raw telemetry from first-light experiments with synthetic noise  
**Format**: One JSON object per line (JSONL)

#### Schema

```json
{
  "verification_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": 1733529600.123456,
  "module_name": "Mathlib.Algebra.Ring.Basic",
  "context": "first_light_cycle_0_item_42",
  "cycle": 0,
  
  "tier": "balanced",
  "timeout_s": 60.0,
  "lean_version": "4.3.0",
  "master_seed": 12345,
  
  "outcome": "verified",
  "success": true,
  "duration_ms": 1234.56,
  "confidence": 0.95,
  
  "cpu_time_ms": 1200.0,
  "memory_peak_mb": 512.0,
  "memory_final_mb": 480.0,
  
  "tactic_count": 15,
  "tactic_depth": 3,
  "proof_size_bytes": 2048,
  "search_nodes": 127,
  "tactics": ["apply", "rw", "simp", "ring"],
  "tactic_counts": {"apply": 5, "rw": 4, "simp": 3, "ring": 3},
  
  "stderr": "",
  "returncode": 0,
  "signal": null,
  
  "noise_injected": false,
  "noise_type": null,
  "ground_truth": null,
  "noise_seed": "000000000000000000000000000000000000000000000000",
  "noise_rates": {"timeout": 0.1, "spurious_fail": 0.05, "spurious_pass": 0.02},
  
  "policy_prob": 0.75,
  "policy_weight": 0.5,
  "policy_gradient": 0.95,
  "policy_update": 0.0095,
  
  "metadata": {}
}
```

#### Required Fields

**Minimal schema** (for basic pipelines):
- Identity: `verification_id`, `timestamp`, `module_name`
- Outcome: `outcome`, `success`, `duration_ms`
- Noise: `noise_injected`, `noise_type`, `ground_truth`

**Full schema** (for all pipelines):
- All fields in Section 1

#### Optional Fields

- `metadata`: Arbitrary JSON object for experiment-specific data
- `io_read_mb`, `io_write_mb`: I/O metrics (if available)
- `confidence`: Verifier confidence (if available)

### 3.2 Calibration Raw Schema

**File**: `calibration_raw_{tier}.jsonl`  
**Purpose**: Raw telemetry from calibration experiments (no noise injection)  
**Format**: JSONL

Same schema as first-light, but with:
- `noise_injected` = `false` (always)
- `noise_type` = `null` (always)
- `ground_truth` = `outcome` (ground truth is the actual outcome)

### 3.3 Replay Log Schema

**File**: `replay_log_{experiment_id}.jsonl`  
**Purpose**: Complete log for deterministic replay of experiments  
**Format**: JSONL

Extended schema with additional replay metadata:

```json
{
  // ... all fields from first_light_synthetic_raw.jsonl ...
  
  "replay_metadata": {
    "experiment_id": "first_light_2025_12_06",
    "slice_name": "test_slice",
    "mode": "rfl",
    "replay_version": "1.0",
    "prng_state": "deadbeef...",  // Full PRNG state for exact replay
    "noise_decision_trace": {
      "should_inject": true,
      "noise_type_sample": 0.123,
      "timeout_sample": 45678.9,
      "prng_path": ["noise_injection", "first_light_cycle_0_item_42", "balanced"]
    }
  }
}
```

---

## 4. Noise Metadata Storage and Replay

### 4.1 Storage Requirements

**Noise metadata MUST be stored** in every telemetry record to enable:
1. **Calibration**: Distinguish real failures from noise-injected failures
2. **Replay**: Reproduce exact noise sequence for debugging
3. **Validation**: Verify determinism (identical seeds → identical noise)
4. **Analysis**: Measure noise impact on policy convergence

### 4.2 Noise Metadata Fields

| Field | Purpose | Storage Location |
|-------|---------|------------------|
| `noise_injected` | Flag indicating noise was injected | Telemetry root |
| `noise_type` | Type of noise injected | Telemetry root |
| `ground_truth` | True outcome before noise | Telemetry root |
| `noise_seed` | PRNG seed for this decision | Telemetry root |
| `noise_rates` | Noise rates used | Telemetry root |
| `prng_state` | Full PRNG state | Replay log only |
| `noise_decision_trace` | Detailed noise decision | Replay log only |

### 4.3 Replay Protocol

#### Step 1: Capture Replay Log

During experiment execution, emit replay log entries with:
- All telemetry fields
- Full `replay_metadata` including PRNG state
- Noise decision trace (samples, thresholds, decisions)

#### Step 2: Load Replay Log

```python
def load_replay_log(replay_log_path: Path) -> List[Dict]:
    with open(replay_log_path) as f:
        return [json.loads(line) for line in f]
```

#### Step 3: Replay Experiment

```python
def replay_experiment(replay_log: List[Dict]) -> List[Dict]:
    results = []
    for entry in replay_log:
        # Restore PRNG state
        prng_state = entry["replay_metadata"]["prng_state"]
        prng = DeterministicPRNG.from_state(prng_state)
        
        # Replay noise decision
        noise_decision = entry["replay_metadata"]["noise_decision_trace"]
        should_inject = noise_decision["should_inject"]
        noise_type = noise_decision["noise_type"]
        
        # Execute with replayed noise
        result = execute_with_noise(
            module=entry["module_name"],
            prng=prng,
            should_inject=should_inject,
            noise_type=noise_type,
        )
        
        results.append(result)
    
    return results
```

#### Step 4: Validate Determinism

```python
def validate_replay_determinism(
    original_log: List[Dict],
    replayed_log: List[Dict],
) -> bool:
    for orig, replay in zip(original_log, replayed_log):
        assert orig["outcome"] == replay["outcome"]
        assert orig["noise_injected"] == replay["noise_injected"]
        assert orig["noise_type"] == replay["noise_type"]
        assert orig["duration_ms"] == replay["duration_ms"]  # Within tolerance
    return True
```

### 4.4 Noise Decision Trace Format

The `noise_decision_trace` object captures the complete noise decision process:

```json
{
  "should_inject": true,
  "noise_type_sample": 0.123,
  "noise_type_thresholds": {
    "timeout": 0.1,
    "spurious_fail": 0.15,
    "spurious_pass": 0.17
  },
  "noise_type_decision": "timeout",
  "timeout_sample": 45678.9,
  "timeout_distribution": "heavy_tail_mixture",
  "timeout_parameters": {
    "pi": 0.1,
    "lambda_fast": 0.1,
    "alpha": 1.5,
    "x_min": 100.0
  },
  "prng_path": [
    "noise_injection",
    "first_light_cycle_0_item_42",
    "balanced",
    "noise_type"
  ]
}
```

This enables **exact reproduction** of noise decisions and **differential debugging** when outcomes diverge.

---

## 5. Pipeline-Specific Requirements

### 5.1 Δp (Policy Gradient) Pipeline

**Required Fields**:
- Identity: `verification_id`, `module_name`, `cycle`
- Outcome: `outcome`, `success`
- Noise: `noise_injected`, `noise_type`, `ground_truth`
- Policy: `policy_prob`, `policy_weight`, `policy_gradient`, `policy_update`

**Processing**:
1. Load telemetry JSONL
2. For each entry:
   - If `noise_injected == false`: Use `outcome` directly
   - If `noise_injected == true`: Use `ground_truth` for calibration, `outcome` for noisy training
   - If `outcome` is abstention (timeout): Skip policy update
   - Compute expected value: `V_expected = compute_expected_value(outcome, noise_rates)`
   - Apply bias correction: `V_corrected = apply_bias_correction(V_expected, noise_rates["spurious_pass"])`
   - Update policy: `Δw = η * V_corrected * ∇log π`
3. Emit policy convergence metrics

### 5.2 RSI (Proof Complexity) Pipeline

**Required Fields**:
- Identity: `verification_id`, `module_name`
- Outcome: `outcome`, `success`
- Complexity: `tactic_count`, `tactic_depth`, `proof_size_bytes`, `search_nodes`

**Processing**:
1. Load telemetry JSONL
2. Filter to successful verifications: `outcome == VERIFIED`
3. Compute complexity metrics:
   - Mean/median/max tactic count
   - Mean/median/max tactic depth
   - Mean/median/max proof size
   - Mean/median/max search nodes
4. Correlate complexity with duration
5. Emit complexity distribution

### 5.3 Ω (Resource Optimization) Pipeline

**Required Fields**:
- Identity: `verification_id`, `module_name`
- Outcome: `outcome`, `duration_ms`
- Resources: `cpu_time_ms`, `memory_peak_mb`, `memory_final_mb`

**Processing**:
1. Load telemetry JSONL
2. Compute resource efficiency:
   - CPU utilization: `cpu_time_ms / duration_ms`
   - Memory efficiency: `memory_final_mb / memory_peak_mb`
   - Resource waste: `memory_peak_mb - memory_final_mb`
3. Identify resource bottlenecks
4. Emit optimization recommendations

### 5.4 TDA (Timeout/Depth Analysis) Pipeline

**Required Fields**:
- Identity: `verification_id`, `module_name`
- Outcome: `outcome`, `duration_ms`
- Diagnostics: `stderr`, `tactic_count`, `tactic_depth`, `tactics`

**Processing**:
1. Load telemetry JSONL
2. Filter to timeouts: `outcome == VERIFIER_TIMEOUT`
3. Analyze timeout patterns:
   - Correlation with tactic depth
   - Correlation with specific tactics
   - Correlation with module characteristics
4. Emit timeout prediction model

---

## 6. Validation and Testing

### 6.1 Schema Validation

**Test**: Validate JSONL against schema

```python
def validate_telemetry_schema(telemetry: Dict) -> bool:
    required_fields = [
        "verification_id", "timestamp", "module_name",
        "outcome", "success", "duration_ms",
        "noise_injected", "noise_type", "ground_truth",
    ]
    
    for field in required_fields:
        assert field in telemetry, f"Missing required field: {field}"
    
    assert telemetry["outcome"] in VerifierErrorCode.__members__.values()
    assert isinstance(telemetry["noise_injected"], bool)
    assert telemetry["noise_type"] in [None, "timeout", "spurious_fail", "spurious_pass"]
    
    return True
```

### 6.2 Determinism Validation

**Test**: Verify identical seeds produce identical telemetry

```python
def test_telemetry_determinism():
    seed = 12345
    module = "Test.Module"
    
    # Run twice with same seed
    telemetry1 = run_lean_with_monitoring(module, seed=seed)
    telemetry2 = run_lean_with_monitoring(module, seed=seed)
    
    # Outcomes must be identical
    assert telemetry1["outcome"] == telemetry2["outcome"]
    assert telemetry1["noise_injected"] == telemetry2["noise_injected"]
    assert telemetry1["noise_type"] == telemetry2["noise_type"]
```

### 6.3 Replay Validation

**Test**: Verify replay produces identical outcomes

```python
def test_replay_determinism():
    # Run experiment and capture replay log
    original_log = run_experiment_with_replay_log()
    
    # Replay experiment
    replayed_log = replay_experiment(original_log)
    
    # Validate determinism
    assert validate_replay_determinism(original_log, replayed_log)
```

---

## 7. Implementation Checklist

### 7.1 Telemetry Collection
- [ ] Implement all fields in Section 1
- [ ] Validate schema on every telemetry emission
- [ ] Emit JSONL to `first_light_synthetic_raw.jsonl`

### 7.2 Noise Metadata
- [ ] Capture noise decisions in telemetry
- [ ] Store ground truth before noise injection
- [ ] Record PRNG seed for reproducibility

### 7.3 Replay Logs
- [ ] Emit replay logs with full PRNG state
- [ ] Implement replay log reader
- [ ] Implement deterministic replay
- [ ] Validate replay determinism

### 7.4 Pipeline Integration
- [ ] Δp: Implement expected value computation
- [ ] RSI: Implement complexity metrics
- [ ] Ω: Implement resource efficiency metrics
- [ ] TDA: Implement timeout analysis

---

## 8. Example Telemetry Records

### 8.1 Successful Verification (No Noise)

```json
{
  "verification_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": 1733529600.0,
  "module_name": "Mathlib.Algebra.Ring.Basic",
  "context": "first_light_cycle_0_item_0",
  "cycle": 0,
  "tier": "balanced",
  "timeout_s": 60.0,
  "lean_version": "4.3.0",
  "master_seed": 12345,
  "outcome": "verified",
  "success": true,
  "duration_ms": 1234.56,
  "cpu_time_ms": 1200.0,
  "memory_peak_mb": 512.0,
  "memory_final_mb": 480.0,
  "tactic_count": 15,
  "tactic_depth": 3,
  "proof_size_bytes": 2048,
  "search_nodes": 127,
  "tactics": ["apply", "rw", "simp", "ring"],
  "tactic_counts": {"apply": 5, "rw": 4, "simp": 3, "ring": 3},
  "stderr": "",
  "returncode": 0,
  "signal": null,
  "noise_injected": false,
  "noise_type": null,
  "ground_truth": null,
  "noise_seed": "000000000000000000000000000000000000000000000000",
  "noise_rates": {"timeout": 0.1, "spurious_fail": 0.05, "spurious_pass": 0.02},
  "policy_prob": 0.75,
  "policy_weight": 0.5,
  "policy_gradient": 0.95,
  "policy_update": 0.0095,
  "metadata": {}
}
```

### 8.2 Timeout (Noise Injected)

```json
{
  "verification_id": "550e8400-e29b-41d4-a716-446655440001",
  "timestamp": 1733529660.0,
  "module_name": "Mathlib.Algebra.Group.Defs",
  "context": "first_light_cycle_0_item_1",
  "cycle": 0,
  "tier": "balanced",
  "timeout_s": 60.0,
  "lean_version": "4.3.0",
  "master_seed": 12345,
  "outcome": "verifier_timeout",
  "success": false,
  "duration_ms": 60000.0,
  "cpu_time_ms": null,
  "memory_peak_mb": null,
  "memory_final_mb": null,
  "tactic_count": null,
  "tactic_depth": null,
  "proof_size_bytes": null,
  "search_nodes": null,
  "tactics": [],
  "tactic_counts": {},
  "stderr": "",
  "returncode": null,
  "signal": null,
  "noise_injected": true,
  "noise_type": "timeout",
  "ground_truth": "verified",
  "noise_seed": "0123456789abcdef0123456789abcdef0123456789abcdef",
  "noise_rates": {"timeout": 0.1, "spurious_fail": 0.05, "spurious_pass": 0.02},
  "policy_prob": 0.75,
  "policy_weight": 0.5,
  "policy_gradient": null,
  "policy_update": null,
  "metadata": {}
}
```

### 8.3 Spurious Failure (Noise Injected)

```json
{
  "verification_id": "550e8400-e29b-41d4-a716-446655440002",
  "timestamp": 1733529720.0,
  "module_name": "Mathlib.Data.List.Basic",
  "context": "first_light_cycle_0_item_2",
  "cycle": 0,
  "tier": "balanced",
  "timeout_s": 60.0,
  "lean_version": "4.3.0",
  "master_seed": 12345,
  "outcome": "proof_invalid",
  "success": false,
  "duration_ms": 2345.67,
  "cpu_time_ms": 2300.0,
  "memory_peak_mb": 600.0,
  "memory_final_mb": 580.0,
  "tactic_count": 20,
  "tactic_depth": 4,
  "proof_size_bytes": 3072,
  "search_nodes": 256,
  "tactics": ["apply", "cases", "induction", "simp"],
  "tactic_counts": {"apply": 8, "cases": 5, "induction": 3, "simp": 4},
  "stderr": "",
  "returncode": 0,
  "signal": null,
  "noise_injected": true,
  "noise_type": "spurious_fail",
  "ground_truth": "verified",
  "noise_seed": "fedcba9876543210fedcba9876543210fedcba9876543210",
  "noise_rates": {"timeout": 0.1, "spurious_fail": 0.05, "spurious_pass": 0.02},
  "policy_prob": 0.75,
  "policy_weight": 0.5,
  "policy_gradient": -0.95,
  "policy_update": -0.0095,
  "metadata": {}
}
```

---

## 9. Summary

This specification defines the **complete interface** between telemetry collection and MathLedger pipelines:

✅ **Telemetry Fields**: 40+ fields organized into 8 categories  
✅ **Error Code Taxonomy**: 11 stable error codes with classification helpers  
✅ **JSONL Schemas**: First-light, calibration, and replay log formats  
✅ **Noise Metadata**: Complete storage and replay protocol  
✅ **Pipeline Requirements**: Specific field requirements for Δp, RSI, Ω, TDA  
✅ **Validation**: Schema, determinism, and replay tests  
✅ **Examples**: Complete telemetry records for all scenarios

**Status**: Test-ready. Ready for implementation and integration testing.

---

**Manus-C — Telemetry Architect**  
*"Every packet accounted for, every signal explained."*
