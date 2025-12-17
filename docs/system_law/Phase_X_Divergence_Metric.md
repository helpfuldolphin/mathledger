# Phase X Divergence Metric Specification

**Status:** SPECIFICATION — BINDING
**Version:** 1.0.0
**Date:** 2025-12-10
**Binding To:** P4 DivergenceAnalyzer, p4_divergence_log.jsonl, p4_calibration_report.json

---

## 1. Purpose

This document formally defines the divergence metric used in Phase X P4 (Real-Runner Shadow Coupling) to measure the difference between the Shadow Twin's predictions and the Real Runner's observed behavior.

**SHADOW MODE CONTRACT:** This metric is for LOGGING and ANALYSIS only. Divergence values do NOT trigger any enforcement actions, remediation, or feedback to the real runner.

---

## 2. Divergence Metric Definition

### 2.1 Absolute Divergence

The absolute divergence at cycle `t` is defined as:

```
divergence(t) = |Δp_real(t) - Δp_twin(t)|
```

Where:
- `Δp_real(t)` = Delta-p observed from real runner at cycle t
- `Δp_twin(t)` = Delta-p predicted by shadow twin at cycle t

### 2.2 Percentage Divergence

The percentage divergence normalizes the absolute divergence:

```
divergence_pct(t) = divergence(t) / max(|Δp_real(t)|, ε)
```

Where:
- `ε = 0.001` (epsilon floor to prevent division by zero)

### 2.3 State-Level Divergence

For individual USLA state components:

```
H_diff(t) = |H_real(t) - H_twin(t)|
rho_diff(t) = |ρ_real(t) - ρ_twin(t)|
tau_diff(t) = |τ_real(t) - τ_twin(t)|
beta_diff(t) = |β_real(t) - β_twin(t)|
```

Binary mismatches:
```
omega_mismatch(t) = (in_omega_real(t) ≠ in_omega_twin(t))
success_mismatch(t) = (success_real(t) ≠ success_twin(t))
```

---

## 3. Severity Classification

### 3.1 Severity Bands

| Severity | Condition | Interpretation |
|----------|-----------|----------------|
| **NONE** | `divergence_pct < 0.01` | Twin and real are effectively aligned |
| **INFO** | `0.01 ≤ divergence_pct < 0.05` | Minor divergence, within normal variance |
| **WARN** | `0.05 ≤ divergence_pct < 0.15` | Moderate divergence, model may need calibration |
| **CRITICAL** | `divergence_pct ≥ 0.15` | Significant divergence, model validity questionable |

### 3.2 Threshold Constants

```python
DIVERGENCE_THRESHOLD_NONE = 0.01
DIVERGENCE_THRESHOLD_INFO = 0.05
DIVERGENCE_THRESHOLD_WARN = 0.15
EPSILON = 0.001
```

### 3.3 Classification Algorithm

```python
def classify_severity(divergence_pct: float) -> str:
    if divergence_pct < DIVERGENCE_THRESHOLD_NONE:
        return "NONE"
    elif divergence_pct < DIVERGENCE_THRESHOLD_INFO:
        return "INFO"
    elif divergence_pct < DIVERGENCE_THRESHOLD_WARN:
        return "WARN"
    else:
        return "CRITICAL"
```

---

## 4. Streak Thresholds

Consecutive divergence cycles are tracked for pattern detection:

| Streak Threshold | Severity Escalation |
|------------------|---------------------|
| 5 consecutive WARN+ | Log streak event |
| 10 consecutive WARN+ | Escalate to WARN streak |
| 20 consecutive WARN+ | Escalate to CRITICAL streak |
| 5 consecutive CRITICAL | Log critical streak event |

**SHADOW MODE:** Streak thresholds do NOT trigger any enforcement. They are for analysis and logging only.

---

## 5. Aggregate Metrics

### 5.1 Run-Level Statistics

At the end of a P4 run, compute:

```
mean_divergence = mean(divergence(t) for t in 1..N)
std_divergence = std(divergence(t) for t in 1..N)
max_divergence = max(divergence(t) for t in 1..N)
p95_divergence = percentile(divergence(t), 0.95)
p99_divergence = percentile(divergence(t), 0.99)
```

### 5.2 Validity Score

The twin validity score is computed as:

```
validity_score = 1.0 - min(1.0, mean_divergence_pct / DIVERGENCE_THRESHOLD_WARN)
```

Interpretation:
- `validity_score ≥ 0.8` → VALID
- `0.5 ≤ validity_score < 0.8` → MARGINAL
- `validity_score < 0.5` → INVALID

---

## 6. Binding to Code

### 6.1 Implementation Files

| Component | File Path | Status |
|-----------|-----------|--------|
| DivergenceAnalyzer | `backend/topology/first_light/divergence_analyzer.py` | Skeleton (NotImplementedError) |
| Divergence Thresholds | `backend/topology/first_light/divergence_analyzer.py:DivergenceThresholds` | Defined but not bound to this spec |

### 6.2 Implementation TODO

The following TODO marker must be added to the implementation file:

```python
# TODO[PhaseX-Divergence-Metric]: Implement severity classification per
# docs/system_law/Phase_X_Divergence_Metric.md
# Thresholds: NONE < 0.01, INFO < 0.05, WARN < 0.15, CRITICAL >= 0.15
```

### 6.3 Output Binding

This specification binds to the following output artifacts:

| Artifact | Field | Source |
|----------|-------|--------|
| `p4_divergence_log.jsonl` | `divergence`, `divergence_pct`, `severity` | Per-cycle divergence record |
| `p4_calibration_report.json` | `divergence_statistics.*`, `severity_distribution.*` | Run-level summary |
| `p4_calibration_report.json` | `calibration_assessment.validity_score` | Computed validity |

---

## 7. Rationale

### 7.1 Why These Thresholds?

The threshold values are derived from:

1. **NONE (< 1%):** Numerical precision floor. Divergence below 1% is indistinguishable from floating-point noise.

2. **INFO (1-5%):** Normal operational variance. Well-calibrated models should rarely exceed 5% divergence.

3. **WARN (5-15%):** Indicates systematic drift. The twin model may be based on outdated assumptions or missing features.

4. **CRITICAL (≥ 15%):** The twin model's predictions are significantly different from reality. Model validity is in question.

### 7.2 Why Percentage-Based?

Percentage-based thresholds normalize across different operating regimes. A 0.01 absolute divergence when Δp_real = 0.02 (50% error) is more concerning than 0.01 divergence when Δp_real = 0.5 (2% error).

---

## 8. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-10 | Initial specification |

---

*This specification is BINDING for all P4 implementation work.*
