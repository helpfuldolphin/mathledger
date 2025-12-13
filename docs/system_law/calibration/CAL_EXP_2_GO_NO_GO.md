# CAL-EXP-2 GO/NO-GO CHECKLIST

**Document Type:** Operational Checklist (Non-Canonical)
**Purpose:** Pre-flight and validity verification for CAL-EXP-2 runs
**Owner:** CLAUDE V (Gatekeeper)
**Date:** 2025-12-13

---

## Notice

This checklist is **descriptive and operational**. It is NOT a CI gate.
Use this to verify CAL-EXP-2 run validity before interpreting results.

---

## 1. Preconditions

### 1.1 Toolchain Parity

- [ ] `uv run pytest tests/ci/test_shadow_audit_sentinel.py -v` passes (4 tests)
- [ ] `uv run pytest tests/ci/test_shadow_audit_guardrails.py -v` passes (9 tests)
- [ ] `scripts/run_shadow_audit.py --help` shows canonical flags only
- [ ] No uncommitted changes to `backend/topology/` or `backend/health/`

### 1.2 Fixtures Hash Recorded

- [ ] Input shadow logs exist at specified `--input` path
- [ ] Input directory contains `shadow_log*.jsonl` files
- [ ] Record SHA256 of input files before run:
  ```bash
  find <input_dir> -name "*.jsonl" -exec sha256sum {} \; > fixtures_hash.txt
  ```
- [ ] `fixtures_hash.txt` committed or recorded in run metadata

### 1.3 SHADOW Mode Asserted

- [ ] Environment variable `USLA_SHADOW_ENABLED=true` is set
- [ ] Environment variable `SHADOW_MODE_ENABLED=true` is set
- [ ] No `enforcement=true` in any configuration file
- [ ] Run output will contain `"mode": "SHADOW"`

---

## 2. Valid Run Criteria

### 2.1 Windowing Configuration

Per CAL-EXP-2 Canonical Record:

- [ ] Window size: **50 cycles**
- [ ] Total horizon: **1000 cycles minimum**
- [ ] Windows computed: `floor(total_cycles / window_size)`

### 2.2 Warm-Up Exclusion

Per Claude Y interpretation guardrail:

- [ ] First **400 cycles** excluded from convergence assessment
- [ ] Warm-up divergence (phases 2-3) is **expected behavior**
- [ ] Do NOT abort or recalibrate during warm-up phase
- [ ] Assessment begins at cycle 401

### 2.3 Learning Rate Configuration

Per UPGRADE-1 validated parameters:

- [ ] LR_H = 0.20
- [ ] LR_ρ = 0.15
- [ ] LR_τ = 0.02
- [ ] LR_β = 0.12

### 2.4 Determinism

- [ ] `--seed` flag provided for reproducibility
- [ ] Same seed produces identical `run_id`
- [ ] Timestamps excluded from determinism comparison

---

## 3. Fail Criteria

### 3.1 CRITICAL Streak Detection

A run is **INVALID** if:

- [ ] 3+ consecutive windows show δp > 0.10 (CRITICAL threshold)
- [ ] Any single window shows δp > 0.15 (hard ceiling)
- [ ] Monotonic divergence across all post-warm-up windows

### 3.2 Validity Regression

A run shows **REGRESSION** if:

- [ ] Final window δp > First post-warm-up window δp + 0.01
- [ ] Mean post-warm-up δp > CAL-EXP-1 baseline (0.0358)
- [ ] Variance increases across final 3 windows

### 3.3 Forbidden Edges

A run is **CONTAMINATED** if:

- [ ] `enforcement=true` appears in any output
- [ ] `action` field contains anything other than `LOGGED_ONLY`
- [ ] `mode` field is not `SHADOW`
- [ ] Any governance decision was modified during run
- [ ] Output contains forbidden phrases per `CAL_EXP_2_LANGUAGE_CONSTRAINTS.md`

---

## 4. Exit Criteria (Uplift/Δp Claims)

### 4.1 Minimum Requirements

Before ANY uplift claim can be made:

- [ ] Run completed with exit code 0
- [ ] All artifacts present (`run_summary.json`, `first_light_status.json`)
- [ ] `schema_version` = `"1.0.0"` in all outputs
- [ ] `mode` = `"SHADOW"` in all outputs
- [ ] `shadow_mode_compliance.no_enforcement` = `true`

### 4.2 Statistical Requirements

- [ ] Minimum 1000 cycles completed
- [ ] Minimum 600 post-warm-up cycles assessed
- [ ] At least 12 windows computed (post-warm-up)

### 4.3 Convergence Evidence

To claim "divergence reduced":

- [ ] Final window δp < First post-warm-up window δp
- [ ] OR: Mean of last 3 windows < Mean of first 3 post-warm-up windows
- [ ] Trend slope is negative or near-zero (< +0.001/window)

### 4.4 Floor Acknowledgment

- [ ] Acknowledge convergence floor at δp ≈ 0.025 (algorithmic limit)
- [ ] Plateau is **acceptable**, not a failure
- [ ] Breaking floor requires UPGRADE-2 (structural change)

### 4.5 Language Compliance

All result statements must:

- [ ] Use approved templates from `CAL_EXP_2_LANGUAGE_CONSTRAINTS.md`
- [ ] End with "SHADOW MODE — observational only"
- [ ] NOT use forbidden phrases (see Section 3.3)

---

## 5. GO/NO-GO Decision Matrix

| Condition | GO | NO-GO |
|-----------|-----|-------|
| Preconditions met | All checked | Any unchecked |
| Valid run criteria | All checked | Any unchecked |
| Fail criteria | None triggered | Any triggered |
| Exit criteria | All checked | Any unchecked |

### Decision

```
[ ] GO — Proceed with CAL-EXP-2 execution / Accept results
[ ] NO-GO — Do not proceed / Reject results
```

**Reason (if NO-GO):** _________________________________

---

## 6. Verification Commands

### Cross-Shell Preflight

If Git Bash shows `$'\377\376export': command not found` (UTF-16 BOM error):

```powershell
# Run once to fix .bashrc encoding (PowerShell)
powershell -ExecutionPolicy Bypass -File scripts/fix_bashrc_encoding.ps1
```

TDA windowed patterns storm guard sanity check:

```bash
# Bash or PowerShell
uv run python -m pytest tests/health/test_tda_windowed_patterns_adapter.py -q
# Expected: 55 tests passing
```

### Pre-Flight

```bash
# Toolchain parity
uv run pytest tests/ci/test_shadow_audit_sentinel.py \
              tests/ci/test_shadow_audit_guardrails.py -v --tb=short

# SHADOW mode assertion
echo "USLA_SHADOW_ENABLED=$USLA_SHADOW_ENABLED"
echo "SHADOW_MODE_ENABLED=$SHADOW_MODE_ENABLED"
```

### Post-Run Validation

```bash
# Artifact presence
ls results/cal_exp_2/*/run_summary.json
ls results/cal_exp_2/*/first_light_status.json

# SHADOW mode markers
grep '"mode": "SHADOW"' results/cal_exp_2/*/run_summary.json
grep '"no_enforcement": true' results/cal_exp_2/*/run_summary.json

# Forbidden content scan
grep -r '"enforcement": true' results/cal_exp_2/ && echo "FAIL: enforcement=true found"
grep -r '"action": "ENFORCED"' results/cal_exp_2/ && echo "FAIL: ENFORCED action found"
```

---

## 7. Sign-Off

| Check | Verified By | Date |
|-------|-------------|------|
| Preconditions | | |
| Valid Run Criteria | | |
| No Fail Criteria | | |
| Exit Criteria Met | | |

**Final Decision:** `[ ] GO` / `[ ] NO-GO`

**Signature:** _________________________________

---

**SHADOW MODE — This checklist is observational and operational only.**
