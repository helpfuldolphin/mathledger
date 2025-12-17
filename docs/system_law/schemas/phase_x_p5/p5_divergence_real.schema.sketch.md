# P5 Real Telemetry Divergence Schema — SPEC-ONLY SKETCH

**Document ID**: P5-DIV-SCHEMA-SKETCH-001
**Status**: SPECIFICATION ONLY — No implementation until P5 activation
**Classification**: System Law — Schema Draft
**Last Updated**: 2025-12-11
**Parent Slot**: `Evidence_Pack_Spec_PhaseX.md` Section 5.6

---

## 1. Purpose

This document sketches the JSON Schema for `p4_shadow/p5_divergence_real.json`, the P5 real telemetry divergence report artifact reserved in the Evidence Pack specification.

**SHADOW MODE**: This schema describes observational artifacts only. No governance authority derives from P5 divergence values.

---

## 2. Schema Linkages

### 2.1 Upstream Dependencies

| Specification | Link Fields |
|---------------|-------------|
| **RTTS** (Real_Telemetry_Topology_Spec.md) | `manifold_validation`, `validation_window`, `mock_detection_flags` |
| **TDA** (TDA_PhaseX_Binding.md) | `tda_comparison.sns`, `tda_comparison.pcs`, `tda_comparison.drs`, `tda_comparison.hss` |
| **GGFL** (Global_Governance_Fusion_PhaseX.md) | `governance_signals`, `fusion_advisory` |

### 2.2 Cross-References

- Divergence decomposition per RTTS Section 3.2
- TDA metrics per `tda_metrics.schema.json`
- Validation criteria per RTTS Section 2.2

---

## 3. Field Classification

### 3.1 Minimal Fields (REQUIRED)

These fields MUST be present for the artifact to be valid:

| Field | Type | Description | Source |
|-------|------|-------------|--------|
| `schema_version` | string | Schema version: `"1.0.0"` | Generator |
| `run_id` | string | P5 run identifier | Runtime |
| `telemetry_source` | enum | `"real"` or `"mock"` | RTTS validation |
| `validation_status` | enum | `"VALIDATED_REAL"`, `"SUSPECTED_MOCK"`, `"UNVALIDATED"` | RTTS Section 2.2 |
| `validation_confidence` | float | Confidence score [0.0, 1.0] | RTTS validation |
| `total_cycles` | integer | Total cycles in P5 run | Runtime |
| `divergence_rate` | float | Fraction of cycles with divergence [0.0, 1.0] | P4 divergence logic |
| `mode` | string | `"SHADOW"` (required) | SHADOW MODE contract |

### 3.2 Recommended Fields (SHOULD)

These fields SHOULD be present for meaningful P5 analysis:

| Field | Type | Description | Source |
|-------|------|-------------|--------|
| `mock_baseline_divergence_rate` | float | P4 mock baseline divergence rate | P4 artifacts |
| `divergence_delta` | float | `divergence_rate - mock_baseline_divergence_rate` | Computed |
| `twin_tracking_accuracy` | object | Real-world twin prediction accuracy | Computed |
| `twin_tracking_accuracy.success` | float | Accuracy for success prediction [0.0, 1.0] | Computed |
| `twin_tracking_accuracy.omega` | float | Accuracy for Ω membership [0.0, 1.0] | Computed |
| `twin_tracking_accuracy.blocked` | float | Accuracy for block decision [0.0, 1.0] | Computed |
| `manifold_validation` | object | RTTS manifold constraint checks | RTTS Section 1.2 |
| `manifold_validation.boundedness_ok` | boolean | All values in [0,1] | RTTS 1.2.1 |
| `manifold_validation.continuity_ok` | boolean | No jumps exceed δ_max | RTTS 1.2.2 |
| `manifold_validation.correlation_ok` | boolean | Cross-correlations in expected range | RTTS 1.2.3 |
| `manifold_validation.violations` | array | List of violation codes (e.g., `["V3_JUMP_H"]`) | RTTS 2.2 |
| `tda_comparison` | object | TDA metric comparison: real vs twin | TDA binding |
| `tda_comparison.sns_delta` | float | SNS difference (real - twin) | Computed |
| `tda_comparison.pcs_delta` | float | PCS difference (real - twin) | Computed |
| `tda_comparison.drs_delta` | float | DRS difference (real - twin) | Computed |
| `tda_comparison.hss_delta` | float | HSS difference (real - twin) | Computed |
| `warm_start_calibration` | object | Twin warm-start calibration metrics | RTTS Section 4 |
| `warm_start_calibration.calibration_cycles` | integer | Cycles used for calibration | Runtime |
| `warm_start_calibration.initial_divergence` | float | Divergence at calibration start | Computed |
| `warm_start_calibration.final_divergence` | float | Divergence at calibration end | Computed |
| `warm_start_calibration.convergence_achieved` | boolean | Twin converged within tolerance | Computed |

### 3.3 Optional Diagnostics (MAY)

These fields MAY be present for deep diagnostic analysis:

| Field | Type | Description | Source |
|-------|------|-------------|--------|
| `divergence_decomposition` | object | Per RTTS Section 3.2 | Computed |
| `divergence_decomposition.bias` | float | `|mean(p_twin) - mean(p_real)|` | RTTS 3.2 |
| `divergence_decomposition.variance` | float | `|std(p_twin) - std(p_real)|` | RTTS 3.2 |
| `divergence_decomposition.timing` | float | `1 - max(xcorr(p_twin, p_real))` | RTTS 3.2 |
| `divergence_decomposition.structural` | float | Rate of sign changes | RTTS 3.2 |
| `pattern_classification` | enum | `"DRIFT"`, `"NOISE_AMPLIFICATION"`, `"PHASE_LAG"`, `"ATTRACTOR_MISS"`, `"TRANSIENT_MISS"`, `"STRUCTURAL_BREAK"`, `"NOMINAL"` | RTTS 3.1 |
| `pattern_confidence` | float | Classification confidence [0.0, 1.0] | Computed |
| `mock_detection_flags` | array | Triggered mock detection criteria | RTTS 2.1 |
| `noise_envelope` | object | Noise floor analysis | RTTS 1.3 |
| `noise_envelope.sigma_H` | float | Health measurement noise | RTTS 1.3.1 |
| `noise_envelope.sigma_rho` | float | Stability measurement noise | RTTS 1.3.1 |
| `noise_envelope.autocorr_lag1` | float | Temporal correlation | RTTS 1.3.2 |
| `noise_envelope.kurtosis` | float | Excess kurtosis | RTTS 1.3.2 |
| `governance_signals` | object | GGFL signal snapshot (observational) | GGFL 2.1 |
| `governance_signals.sig_top_status` | string | Topology signal status | GGFL |
| `governance_signals.sig_rpl_status` | string | Replay signal status | GGFL |
| `governance_signals.sig_tel_status` | string | Telemetry signal status | GGFL |
| `fusion_advisory` | object | GGFL fusion output (SHADOW only) | GGFL |
| `fusion_advisory.recommendation` | enum | `"ALLOW"`, `"WARN"`, `"BLOCK"` | GGFL |
| `fusion_advisory.conflict_detected` | boolean | Signal conflict detected | GGFL |
| `recalibration_recommendations` | array | Suggested twin parameter adjustments | RTTS 3.1 |
| `timing` | object | Execution timing metadata | Runtime |
| `timing.start_time` | string | ISO8601 start timestamp | Runtime |
| `timing.end_time` | string | ISO8601 end timestamp | Runtime |
| `timing.duration_seconds` | float | Total runtime | Runtime |

---

## 4. Informal JSON Schema (SPEC-ONLY)

```yaml
# SPEC-ONLY — Not for production validation
$id: https://mathledger.org/schemas/phase_x_p5/p5_divergence_real.v1.0.0.json
$schema: http://json-schema.org/draft-07/schema#
title: P5 Real Telemetry Divergence Report
version: 1.0.0
status: DRAFT — SPEC-ONLY

type: object
required:
  - schema_version
  - run_id
  - telemetry_source
  - validation_status
  - validation_confidence
  - total_cycles
  - divergence_rate
  - mode

properties:
  # === MINIMAL FIELDS ===
  schema_version:
    type: string
    const: "1.0.0"
  run_id:
    type: string
    pattern: "^p5_[0-9]{8}_[0-9]{6}.*$"
  telemetry_source:
    type: string
    enum: ["real", "mock"]
  validation_status:
    type: string
    enum: ["VALIDATED_REAL", "SUSPECTED_MOCK", "UNVALIDATED"]
  validation_confidence:
    type: number
    minimum: 0.0
    maximum: 1.0
  total_cycles:
    type: integer
    minimum: 1
  divergence_rate:
    type: number
    minimum: 0.0
    maximum: 1.0
  mode:
    type: string
    const: "SHADOW"

  # === RECOMMENDED FIELDS ===
  mock_baseline_divergence_rate:
    type: number
    minimum: 0.0
    maximum: 1.0
  divergence_delta:
    type: number
    description: "divergence_rate - mock_baseline_divergence_rate"
  twin_tracking_accuracy:
    type: object
    properties:
      success:
        type: number
        minimum: 0.0
        maximum: 1.0
      omega:
        type: number
        minimum: 0.0
        maximum: 1.0
      blocked:
        type: number
        minimum: 0.0
        maximum: 1.0
  manifold_validation:
    type: object
    properties:
      boundedness_ok:
        type: boolean
      continuity_ok:
        type: boolean
      correlation_ok:
        type: boolean
      violations:
        type: array
        items:
          type: string
          pattern: "^(V[0-9]|MOCK-[0-9]{3}).*$"
  tda_comparison:
    type: object
    properties:
      sns_delta:
        type: number
      pcs_delta:
        type: number
      drs_delta:
        type: number
      hss_delta:
        type: number
  warm_start_calibration:
    type: object
    properties:
      calibration_cycles:
        type: integer
        minimum: 0
      initial_divergence:
        type: number
      final_divergence:
        type: number
      convergence_achieved:
        type: boolean

  # === OPTIONAL DIAGNOSTICS ===
  divergence_decomposition:
    type: object
    properties:
      bias:
        type: number
        minimum: 0.0
      variance:
        type: number
        minimum: 0.0
      timing:
        type: number
        minimum: 0.0
        maximum: 1.0
      structural:
        type: number
        minimum: 0.0
  pattern_classification:
    type: string
    enum:
      - "DRIFT"
      - "NOISE_AMPLIFICATION"
      - "PHASE_LAG"
      - "ATTRACTOR_MISS"
      - "TRANSIENT_MISS"
      - "STRUCTURAL_BREAK"
      - "NOMINAL"
  pattern_confidence:
    type: number
    minimum: 0.0
    maximum: 1.0
  mock_detection_flags:
    type: array
    items:
      type: string
      pattern: "^MOCK-[0-9]{3}$"
  noise_envelope:
    type: object
    properties:
      sigma_H:
        type: number
      sigma_rho:
        type: number
      autocorr_lag1:
        type: number
      kurtosis:
        type: number
  governance_signals:
    type: object
    additionalProperties:
      type: string
  fusion_advisory:
    type: object
    properties:
      recommendation:
        type: string
        enum: ["ALLOW", "WARN", "BLOCK"]
      conflict_detected:
        type: boolean
  recalibration_recommendations:
    type: array
    items:
      type: object
      properties:
        parameter:
          type: string
        current_value:
          type: number
        suggested_value:
          type: number
        rationale:
          type: string
  timing:
    type: object
    properties:
      start_time:
        type: string
        format: date-time
      end_time:
        type: string
        format: date-time
      duration_seconds:
        type: number
```

---

## 5. Worked Example: P5 Divergence Record

```json
{
  "schema_version": "1.0.0",
  "run_id": "p5_20251215_143022_real_arithmetic",
  "telemetry_source": "real",
  "validation_status": "VALIDATED_REAL",
  "validation_confidence": 0.92,
  "total_cycles": 1000,
  "divergence_rate": 0.23,
  "mode": "SHADOW",

  "mock_baseline_divergence_rate": 0.97,
  "divergence_delta": -0.74,

  "twin_tracking_accuracy": {
    "success": 0.91,
    "omega": 0.88,
    "blocked": 0.85
  },

  "manifold_validation": {
    "boundedness_ok": true,
    "continuity_ok": true,
    "correlation_ok": true,
    "violations": []
  },

  "tda_comparison": {
    "sns_delta": 0.02,
    "pcs_delta": -0.01,
    "drs_delta": 0.05,
    "hss_delta": 0.03
  },

  "warm_start_calibration": {
    "calibration_cycles": 50,
    "initial_divergence": 0.65,
    "final_divergence": 0.18,
    "convergence_achieved": true
  },

  "divergence_decomposition": {
    "bias": 0.03,
    "variance": 0.02,
    "timing": 0.08,
    "structural": 0.10
  },

  "pattern_classification": "NOMINAL",
  "pattern_confidence": 0.87,

  "mock_detection_flags": [],

  "noise_envelope": {
    "sigma_H": 0.012,
    "sigma_rho": 0.008,
    "autocorr_lag1": 0.32,
    "kurtosis": 0.45
  },

  "governance_signals": {
    "sig_top_status": "OK",
    "sig_rpl_status": "OK",
    "sig_tel_status": "OK"
  },

  "fusion_advisory": {
    "recommendation": "ALLOW",
    "conflict_detected": false
  },

  "recalibration_recommendations": [],

  "timing": {
    "start_time": "2025-12-15T14:30:22Z",
    "end_time": "2025-12-15T14:35:47Z",
    "duration_seconds": 325.4
  }
}
```

### 5.1 Example Interpretation

This example P5 divergence record shows:

1. **Real telemetry validated** (`validation_status: "VALIDATED_REAL"`, confidence 92%)
2. **Divergence dramatically improved** from mock baseline (97% → 23%, delta of -74%)
3. **Twin tracking is strong** (91% success accuracy, 88% Ω accuracy)
4. **Manifold constraints satisfied** (no violations)
5. **TDA metrics aligned** (all deltas < 0.1)
6. **Warm-start converged** (from 65% initial to 18% final divergence)
7. **Pattern is NOMINAL** (no systematic divergence pattern detected)
8. **Governance signals healthy** (all OK, no conflict)

---

## 6. Smoke-Test Readiness Checklist

Before including `p5_divergence_real.json` in an Evidence Pack, verify:

### 6.1 Artifact Generation

- [ ] **GEN-01**: P5 harness completed with `telemetry_adapter=real`
- [ ] **GEN-02**: File exists at `p4_shadow/p5_divergence_real.json`
- [ ] **GEN-03**: File is valid JSON (parse succeeds)
- [ ] **GEN-04**: File size > 100 bytes (non-trivial content)

### 6.2 Minimal Field Validation

- [ ] **MIN-01**: `schema_version` equals `"1.0.0"`
- [ ] **MIN-02**: `run_id` matches pattern `p5_*`
- [ ] **MIN-03**: `telemetry_source` is `"real"` or `"mock"`
- [ ] **MIN-04**: `validation_status` is valid enum value
- [ ] **MIN-05**: `validation_confidence` in [0.0, 1.0]
- [ ] **MIN-06**: `total_cycles` >= 1
- [ ] **MIN-07**: `divergence_rate` in [0.0, 1.0]
- [ ] **MIN-08**: `mode` equals `"SHADOW"`

### 6.3 Recommended Field Presence

- [ ] **REC-01**: `mock_baseline_divergence_rate` present (for baseline comparison)
- [ ] **REC-02**: `twin_tracking_accuracy` object present
- [ ] **REC-03**: `manifold_validation` object present
- [ ] **REC-04**: `tda_comparison` object present (if TDA enabled)

### 6.4 Semantic Validation

- [ ] **SEM-01**: If `telemetry_source: "real"`, then `validation_status` should be `"VALIDATED_REAL"` or include explanation
- [ ] **SEM-02**: If `validation_status: "VALIDATED_REAL"`, then `validation_confidence` >= 0.8 (per RTTS)
- [ ] **SEM-03**: `divergence_rate` < `mock_baseline_divergence_rate` expected (real should track better than mock)
- [ ] **SEM-04**: No HARD FAULT mock detection flags (e.g., `MOCK-001`, `MOCK-002`, `MOCK-009`)

### 6.5 SHADOW MODE Compliance

- [ ] **SHD-01**: `mode` field is `"SHADOW"` (enforced)
- [ ] **SHD-02**: No `action` fields with enforcement values (only `"LOGGED_ONLY"` if present)
- [ ] **SHD-03**: `fusion_advisory.recommendation` is advisory only (no enforcement mechanism)

### 6.6 Evidence Pack Integration

- [ ] **EVP-01**: File hash computed and matches manifest entry (when bundled)
- [ ] **EVP-02**: File path matches reserved slot `p4_shadow/p5_divergence_real.json`
- [ ] **EVP-03**: Schema version in file matches Evidence Pack spec reference

---

## 7. Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2025-12-11 | Initial sketch — SPEC-ONLY |

---

**CLAUDE L: P5 Divergence Schema Sketch Ready.**
