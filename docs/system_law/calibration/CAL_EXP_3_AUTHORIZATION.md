# CAL-EXP-3 AUTHORIZATION

**Status:** CONDITIONAL AUTHORIZATION
**Authority:** CLAUDE V (Gatekeeper)
**Date:** 2025-12-13
**Scope:** Uplift Measurement (Δp)

---

## Authorization Statement

### AUTHORIZED TO MEASURE UPLIFT

**Condition:** Toolchain parity passes AND CAL-EXP-2 verifier passes.

This authorization permits measurement of Δp (uplift) under the following constraints:

1. All measurements are observational
2. No enforcement semantics
3. No external claims until pilot exit criteria met
4. No new science (metrics, formulas, thresholds)

---

## Pilot Status

### PILOT REMAINS WAITING MODE

| Status | Description |
|--------|-------------|
| Internal measurement | ✓ AUTHORIZED |
| Internal reporting | ✓ AUTHORIZED |
| External claims | ✗ NOT AUTHORIZED |
| Production deployment | ✗ NOT AUTHORIZED |
| Enforcement activation | ✗ NOT AUTHORIZED |

**No external claims may be made until pilot exit criteria are satisfied.**

---

## No New Science Reminder

### FROZEN ELEMENTS

| Element | Status | Document |
|---------|--------|----------|
| Metric definitions | FROZEN | `Phase_X_Divergence_Metric.md` |
| Severity thresholds | FROZEN | `Phase_X_Divergence_Metric.md` |
| Exit codes | FROZEN | `RUN_SHADOW_AUDIT_V0_1_CONTRACT.md` |
| CLI flags | FROZEN | `RUN_SHADOW_AUDIT_V0_1_CONTRACT.md` |
| Artifact schemas | FROZEN | `Evidence_Pack_Spec_PhaseX.md` |

### PERMITTED ACTIONS

| Action | Constraint |
|--------|------------|
| Measure existing metrics | Read-only |
| Compare configurations | No new formulas |
| Document observations | Safe templates only |
| Extend observation horizon | Cycles only, no structural changes |

### FORBIDDEN ACTIONS

| Action | Reason |
|--------|--------|
| Define new metrics | No new science |
| Modify thresholds | Frozen in spec |
| Add enforcement logic | SHADOW mode invariant |
| Change convergence algorithms | Requires UPGRADE-2 spec |

---

## Prerequisites Checklist

### Required Before Uplift Measurement

- [ ] **CAL-EXP-1 replicated**
  - Baseline metrics recorded
  - Two seeds verified (42, 43)
  - Canonical record frozen
  - Reference: `CAL_EXP_1.md`

- [ ] **CAL-EXP-2 verified**
  - 1000 cycles completed
  - Divergence reduction valid (§1)
  - No new pathology (§3)
  - Exit decision issued
  - Reference: `CAL_EXP_2_EXIT_DECISION.md`

- [ ] **Language hygiene passed**
  - All claims use safe templates
  - No forbidden phrases
  - Ends with "SHADOW MODE — observational only"
  - Reference: `CAL_EXP_2_LANGUAGE_CONSTRAINTS.md`

- [ ] **Reproducibility gate green**
  - Toolchain fingerprint recorded
  - Same seed produces same output
  - Fixtures hash committed
  - Reference: `CAL_EXP_2_GO_NO_GO.md` §1.2

---

## Verification Commands

### Toolchain Parity

```bash
uv run pytest \
    tests/ci/test_shadow_audit_sentinel.py \
    tests/ci/test_shadow_audit_guardrails.py \
    tests/integration/test_shadow_audit_e2e.py \
    -v --tb=short

# Expected: 21 passed, 0 failed
```

### CAL-EXP-2 Verifier

```bash
# Verify artifacts exist
ls results/cal_exp_2/*/RUN_METADATA.json
ls results/cal_exp_2/*/twin_accuracy.json

# Verify SHADOW mode
grep '"mode": "SHADOW"' results/cal_exp_2/*/*.json

# Verify no enforcement
grep -r '"enforcement": true' results/cal_exp_2/ && echo "FAIL" || echo "PASS"
```

### Combined Gate

```bash
uv run pytest \
    tests/ci/test_shadow_audit_sentinel.py \
    tests/ci/test_shadow_audit_guardrails.py \
    tests/integration/test_shadow_audit_e2e.py \
    -v --tb=short && \
grep -q '"mode": "SHADOW"' results/cal_exp_2/*/RUN_METADATA.json && \
echo "CAL-EXP-3: AUTHORIZED" || echo "CAL-EXP-3: NOT AUTHORIZED"
```

---

## Conditional Authorization Matrix

| Condition | Status | Result |
|-----------|--------|--------|
| Toolchain parity (21/21) | REQUIRED | — |
| CAL-EXP-2 exit decision | REQUIRED | — |
| SHADOW mode in all artifacts | REQUIRED | — |
| No enforcement=true | REQUIRED | — |

| All conditions met | → | **AUTHORIZED TO MEASURE UPLIFT** |
| Any condition failed | → | **NOT AUTHORIZED** |

---

## Boundaries

### HARD BOUNDARIES (Violation = Abort)

1. `enforcement=true` appears in any output
2. `action` field contains anything other than `LOGGED_ONLY`
3. `mode` field is not `SHADOW`
4. Any governance decision modified during measurement
5. New metric formula introduced
6. Threshold value changed

### SOFT BOUNDARIES (Violation = Document and Continue)

1. Warm-up divergence exceeds 1.5× baseline
2. Single window δp exceeds 0.10 (CRITICAL threshold)
3. Variance increases in final phase

---

## Authorization Record

| Prerequisite | Status | Date |
|--------------|--------|------|
| CAL-EXP-1 replicated | ✓ COMPLETE | 2025-12-12 |
| CAL-EXP-2 verified | ✓ COMPLETE | 2025-12-13 |
| Language hygiene | ✓ PASS | 2025-12-13 |
| Reproducibility gate | ✓ GREEN | 2025-12-13 |

### Authorization Granted

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│   CAL-EXP-3: CONDITIONAL AUTHORIZATION                  │
│                                                         │
│   Authorized to measure uplift (Δp)                     │
│   Pilot remains waiting mode                            │
│   No external claims                                    │
│   No new science                                        │
│                                                         │
│   Condition: Toolchain parity + CAL-EXP-2 verifier      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Sign-Off

| Role | Agent | Date |
|------|-------|------|
| Authorization | CLAUDE V | 2025-12-13 |
| Prerequisites Verification | CLAUDE V | 2025-12-13 |
| Boundary Definition | CLAUDE Y | 2025-12-13 |

---

**SHADOW MODE — observational only.**
