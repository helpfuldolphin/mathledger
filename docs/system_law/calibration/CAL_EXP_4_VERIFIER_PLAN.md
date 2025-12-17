# CAL-EXP-4: Verifier Delta Plan

**Status**: READY (Schemas Defined)
**Type**: Verifier Engineering Plan
**Author**: CLAUDE R (Verifier/CI Owner)
**Created**: 2025-12-17
**Updated**: 2025-12-17
**Dependency**: `CAL_EXP_4_VARIANCE_STRESS_SPEC.md` (assumed binding)

---

## Schema References

| Schema | Path | Version |
|--------|------|---------|
| Temporal Structure Audit | `schemas/cal_exp_4/temporal_structure_audit.schema.json` | 1.0.0 |
| Variance Profile Audit | `schemas/cal_exp_4/variance_profile_audit.schema.json` | 1.0.0 |

---

## Purpose

This document specifies the verifier changes required to validate CAL-EXP-4 runs. CAL-EXP-4 is a Phase-II stress test focused on detecting temporal/variance structure mismatches that would invalidate comparability between arms.

**CAL-EXP-4 Goal**: Prevent false comparability claims. If baseline and treatment arms exhibit structurally incompatible variance profiles or temporal dynamics, the run must fail-close or cap claims.

**This plan does NOT**:
- Define new metrics (temporal/variance features are "audit features," not metrics)
- Implement code (plan only)
- Reference pilot data
- Modify CAL-EXP-3 behavior

---

## 1. Scope Delta from CAL-EXP-3

CAL-EXP-4 verifier extends CAL-EXP-3 verifier with **additive checks only**. All CAL-EXP-3 checks remain in force.

| Aspect | CAL-EXP-3 | CAL-EXP-4 Delta |
|--------|-----------|-----------------|
| Focus | Measure uplift | Validate comparability |
| Artifacts | cycles.jsonl, run_config.json, ... | + temporal_structure_audit.json, + variance_profile_audit.json |
| Checks | Alignment, parity, validity | + Temporal structure checks, + Variance profile checks |
| Claim effect | Permits L4/L5 if valid | Caps or invalidates if variance mismatch |

---

## 2. New Required Artifacts (Additive)

### 2.1 Artifact Registry

CAL-EXP-4 requires all CAL-EXP-3 artifacts PLUS the following:

| Artifact | Path | Purpose | Required |
|----------|------|---------|----------|
| Temporal Structure Audit | `validity/temporal_structure_audit.json` | Per-arm temporal dynamics summary | YES |
| Variance Profile Audit | `validity/variance_profile_audit.json` | Per-arm variance profile comparison | YES |

### 2.2 Artifact Schemas

**Authoritative schemas**: See `schemas/cal_exp_4/` directory.

#### 2.2.1 `validity/temporal_structure_audit.json`

**Schema**: `schemas/cal_exp_4/temporal_structure_audit.schema.json` (v1.0.0)

**Required fields** (per schema):
- `schema_version`: Must be "1.0.0"
- `experiment_id`: Must be "CAL-EXP-4"
- `baseline_arm`: Temporal profile object (cycle_count, gaps, monotonicity, coverage)
- `treatment_arm`: Temporal profile object (same structure as baseline)
- `comparability`: Pass/fail flags (cycle_count_match, cycle_indices_identical, temporal_structure_pass)
- `thresholds`: Thresholds used for pass/fail determination
- `generated_at`: ISO8601 timestamp

**Key comparability fields** (verifier reads these):
| Field | Schema Path | Type | Verifier Check |
|-------|-------------|------|----------------|
| `cycle_count_match` | `.comparability.cycle_count_match` | bool | `temporal:cycle_count_match` |
| `cycle_indices_identical` | `.comparability.cycle_indices_identical` | bool | `window:arm_alignment` |
| `temporal_structure_pass` | `.comparability.temporal_structure_pass` | bool | `temporal:structure_compatible` |
| `monotonic_cycle_indices` | `.baseline_arm.monotonic_cycle_indices` | bool | `temporal:baseline_monotonic` |
| `monotonic_cycle_indices` | `.treatment_arm.monotonic_cycle_indices` | bool | `temporal:treatment_monotonic` |
| `temporal_coverage_ratio` | `.baseline_arm.temporal_coverage_ratio` | float | `temporal:coverage_ratio_baseline` |
| `temporal_coverage_ratio` | `.treatment_arm.temporal_coverage_ratio` | float | `temporal:coverage_ratio_treatment` |

#### 2.2.2 `validity/variance_profile_audit.json`

**Schema**: `schemas/cal_exp_4/variance_profile_audit.schema.json` (v1.0.0)

**Required fields** (per schema):
- `schema_version`: Must be "1.0.0"
- `experiment_id`: Must be "CAL-EXP-4"
- `baseline_arm`: Variance profile object (delta_p statistics)
- `treatment_arm`: Variance profile object (same structure as baseline)
- `comparability`: Pass/fail flags and computed ratios
- `thresholds`: Thresholds used for pass/fail determination
- `generated_at`: ISO8601 timestamp

**Key comparability fields** (verifier reads these):
| Field | Schema Path | Type | Verifier Check |
|-------|-------------|------|----------------|
| `variance_ratio` | `.comparability.variance_ratio` | float | `variance:ratio_within_threshold` |
| `variance_ratio_acceptable` | `.comparability.variance_ratio_acceptable` | bool | `variance:ratio_within_threshold` |
| `windowed_drift_acceptable` | `.comparability.windowed_drift_acceptable` | bool | `variance:windowed_drift_acceptable` |
| `profile_compatible` | `.comparability.profile_compatible` | bool | `variance:profile_compatible` |
| `variance_profile_pass` | `.comparability.variance_profile_pass` | bool | Final verdict |
| `claim_cap_applied` | `.comparability.claim_cap_applied` | bool | Claim level adjustment |
| `claim_cap_level` | `.comparability.claim_cap_level` | string/null | Max claim if capped |

**Threshold fields** (from spec):
| Field | Schema Path | Description |
|-------|-------------|-------------|
| `variance_ratio_max` | `.thresholds.variance_ratio_max` | Upper bound for acceptable ratio |
| `variance_ratio_min` | `.thresholds.variance_ratio_min` | Lower bound for acceptable ratio |
| `windowed_drift_max` | `.thresholds.windowed_drift_max` | Max drift across sub-windows |
| `claim_cap_threshold` | `.thresholds.claim_cap_threshold` | Ratio threshold triggering cap vs fail |

---

## 3. Check Specification

### 3.1 Check Categories

| Category | Purpose | Failure Mode |
|----------|---------|--------------|
| Artifact Presence | Required files exist | FAIL (invalidates) |
| SHADOW Mode Invariants | Non-blocking constraints | FAIL (invalidates) |
| Toolchain Parity | Execution environment match | FAIL (invalidates) |
| Window Alignment | Temporal boundaries match | FAIL (invalidates) |
| Temporal Structure | Cycle coverage and gaps | FAIL or WARN (per spec) |
| Variance Profile | Dispersion comparability | FAIL or WARN (per spec) |

### 3.2 Artifact Presence Checks

#### Check: `artifact:temporal_structure_audit`

| Field | Value |
|-------|-------|
| **Check Name** | `artifact:temporal_structure_audit` |
| **Inputs** | `validity/temporal_structure_audit.json` |
| **PASS** | File exists and parses as valid JSON |
| **FAIL** | File missing or invalid JSON |
| **WARN** | N/A |
| **Failure Message** | `FAIL: artifact:temporal_structure_audit: file missing or invalid` |
| **Invalidates** | YES |

#### Check: `artifact:variance_profile_audit`

| Field | Value |
|-------|-------|
| **Check Name** | `artifact:variance_profile_audit` |
| **Inputs** | `validity/variance_profile_audit.json` |
| **PASS** | File exists and parses as valid JSON |
| **FAIL** | File missing or invalid JSON |
| **WARN** | N/A |
| **Failure Message** | `FAIL: artifact:variance_profile_audit: file missing or invalid` |
| **Invalidates** | YES |

### 3.3 SHADOW Mode Invariants

#### Check: `shadow_mode:experiment_identity`

| Field | Value |
|-------|-------|
| **Check Name** | `shadow_mode:experiment_identity` |
| **Inputs** | `run_config.json`.experiment |
| **PASS** | experiment == "CAL-EXP-4" |
| **FAIL** | experiment != "CAL-EXP-4" |
| **WARN** | N/A |
| **Failure Message** | `FAIL: shadow_mode:experiment_identity: expected=CAL-EXP-4, actual={value}` |
| **Invalidates** | YES |

#### Check: `shadow_mode:enforcement_off`

| Field | Value |
|-------|-------|
| **Check Name** | `shadow_mode:enforcement_off` |
| **Inputs** | `RUN_METADATA.json`.enforcement OR `run_config.json`.enforcement |
| **PASS** | enforcement == false |
| **FAIL** | enforcement == true |
| **WARN** | N/A |
| **Failure Message** | `FAIL: shadow_mode:enforcement_off: enforcement must be false` |
| **Invalidates** | YES |

### 3.4 Toolchain Parity Checks

#### Check: `toolchain:hash_present`

| Field | Value |
|-------|-------|
| **Check Name** | `toolchain:hash_present` |
| **Inputs** | `validity/toolchain_hash.txt` |
| **PASS** | File exists with valid SHA-256 (64 hex chars) |
| **FAIL** | File missing or invalid hash format |
| **WARN** | N/A |
| **Failure Message** | `FAIL: toolchain:hash_present: {reason}` |
| **Invalidates** | YES |

#### Check: `toolchain:baseline_treatment_match`

| Field | Value |
|-------|-------|
| **Check Name** | `toolchain:baseline_treatment_match` |
| **Inputs** | `validity/toolchain_hash.txt`, per-arm hashes if recorded |
| **PASS** | Baseline hash == Treatment hash |
| **FAIL** | Hash mismatch |
| **WARN** | N/A |
| **Failure Message** | `FAIL: toolchain:baseline_treatment_match: hash mismatch` |
| **Invalidates** | YES |

### 3.5 Window Alignment Checks

#### Check: `window:pre_registered`

| Field | Value |
|-------|-------|
| **Check Name** | `window:pre_registered` |
| **Inputs** | `run_config.json`.windows.evaluation_window |
| **PASS** | evaluation_window present with start_cycle and end_cycle |
| **FAIL** | evaluation_window missing or incomplete |
| **WARN** | N/A |
| **Failure Message** | `FAIL: window:pre_registered: evaluation_window not declared` |
| **Invalidates** | YES |

#### Check: `window:arm_alignment`

| Field | Value |
|-------|-------|
| **Check Name** | `window:arm_alignment` |
| **Inputs** | `validity/temporal_structure_audit.json`.comparability.cycle_indices_identical |
| **PASS** | cycle_indices_identical == true |
| **FAIL** | cycle_indices_identical == false |
| **WARN** | N/A |
| **Failure Message** | `FAIL: window:arm_alignment: baseline and treatment cycle indices differ` |
| **Invalidates** | YES |

### 3.6 Temporal Structure Checks (CAL-EXP-4 Specific)

#### Check: `temporal:cycle_count_match`

| Field | Value |
|-------|-------|
| **Check Name** | `temporal:cycle_count_match` |
| **Inputs** | `validity/temporal_structure_audit.json`.comparability.cycle_count_match |
| **PASS** | cycle_count_match == true |
| **FAIL** | cycle_count_match == false |
| **WARN** | N/A |
| **Failure Message** | `FAIL: temporal:cycle_count_match: baseline={n1}, treatment={n2}` |
| **Invalidates** | YES |

#### Check: `temporal:baseline_monotonic`

| Field | Value |
|-------|-------|
| **Check Name** | `temporal:baseline_monotonic` |
| **Inputs** | `validity/temporal_structure_audit.json`.baseline_arm.monotonic_cycle_indices |
| **PASS** | monotonic_cycle_indices == true |
| **FAIL** | monotonic_cycle_indices == false |
| **WARN** | N/A |
| **Failure Message** | `FAIL: temporal:baseline_monotonic: cycle indices not monotonic` |
| **Invalidates** | YES |

#### Check: `temporal:treatment_monotonic`

| Field | Value |
|-------|-------|
| **Check Name** | `temporal:treatment_monotonic` |
| **Inputs** | `validity/temporal_structure_audit.json`.treatment_arm.monotonic_cycle_indices |
| **PASS** | monotonic_cycle_indices == true |
| **FAIL** | monotonic_cycle_indices == false |
| **WARN** | N/A |
| **Failure Message** | `FAIL: temporal:treatment_monotonic: cycle indices not monotonic` |
| **Invalidates** | YES |

#### Check: `temporal:coverage_ratio_baseline`

| Field | Value |
|-------|-------|
| **Check Name** | `temporal:coverage_ratio_baseline` |
| **Inputs** | `validity/temporal_structure_audit.json`.baseline_arm.temporal_coverage_ratio |
| **PASS** | temporal_coverage_ratio >= 1.0 (all expected cycles present) |
| **FAIL** | temporal_coverage_ratio < 1.0 |
| **WARN** | N/A |
| **Failure Message** | `FAIL: temporal:coverage_ratio_baseline: coverage={ratio}, expected=1.0` |
| **Invalidates** | YES |

#### Check: `temporal:coverage_ratio_treatment`

| Field | Value |
|-------|-------|
| **Check Name** | `temporal:coverage_ratio_treatment` |
| **Inputs** | `validity/temporal_structure_audit.json`.treatment_arm.temporal_coverage_ratio |
| **PASS** | temporal_coverage_ratio >= 1.0 (all expected cycles present) |
| **FAIL** | temporal_coverage_ratio < 1.0 |
| **WARN** | N/A |
| **Failure Message** | `FAIL: temporal:coverage_ratio_treatment: coverage={ratio}, expected=1.0` |
| **Invalidates** | YES |

#### Check: `temporal:structure_compatible`

| Field | Value |
|-------|-------|
| **Check Name** | `temporal:structure_compatible` |
| **Inputs** | `validity/temporal_structure_audit.json`.comparability.temporal_structure_compatible |
| **PASS** | temporal_structure_compatible == true |
| **FAIL** | temporal_structure_compatible == false |
| **WARN** | N/A |
| **Failure Message** | `FAIL: temporal:structure_compatible: arms have incompatible temporal structure` |
| **Invalidates** | YES (fail-close) |

### 3.7 Variance Profile Checks (CAL-EXP-4 Specific)

#### Check: `variance:ratio_within_threshold`

| Field | Value |
|-------|-------|
| **Check Name** | `variance:ratio_within_threshold` |
| **Inputs** | `validity/variance_profile_audit.json`.comparability.variance_ratio, .variance_ratio_threshold |
| **PASS** | variance_ratio_acceptable == true |
| **FAIL** | variance_ratio_acceptable == false AND spec requires fail-close |
| **WARN** | variance_ratio_acceptable == false AND spec allows claim cap |
| **Failure Message** | `FAIL: variance:ratio_within_threshold: ratio={r}, threshold={t}` OR `WARN: variance:ratio_within_threshold: ratio={r} exceeds threshold={t}, claim capped` |
| **Invalidates** | Depends on spec (fail-close vs claim-cap) |

#### Check: `variance:windowed_drift_acceptable`

| Field | Value |
|-------|-------|
| **Check Name** | `variance:windowed_drift_acceptable` |
| **Inputs** | `validity/variance_profile_audit.json`.comparability.windowed_drift_acceptable |
| **PASS** | windowed_drift_acceptable == true |
| **FAIL** | windowed_drift_acceptable == false AND spec requires fail-close |
| **WARN** | windowed_drift_acceptable == false AND spec allows claim cap |
| **Failure Message** | `FAIL: variance:windowed_drift_acceptable: drift={d} exceeds limit` OR `WARN: variance:windowed_drift_acceptable: drift={d}, claim capped` |
| **Invalidates** | Depends on spec (fail-close vs claim-cap) |

#### Check: `variance:profile_compatible`

| Field | Value |
|-------|-------|
| **Check Name** | `variance:profile_compatible` |
| **Inputs** | `validity/variance_profile_audit.json`.comparability.profile_compatible |
| **PASS** | profile_compatible == true |
| **FAIL** | profile_compatible == false |
| **WARN** | N/A (this is the aggregate fail-close check) |
| **Failure Message** | `FAIL: variance:profile_compatible: variance profiles not comparable` |
| **Invalidates** | YES (fail-close) |

---

## 4. Failure Message Format

All failure messages follow a strict format to ensure neutrality and parseability:

```
{STATUS}: {check_name}: {description}
```

Where:
- `{STATUS}` is one of: `PASS`, `FAIL`, `WARN`
- `{check_name}` is the stable string identifier (e.g., `variance:ratio_within_threshold`)
- `{description}` is a neutral, single-line description with key=value pairs

**Examples**:
```
PASS: artifact:temporal_structure_audit: file present and valid
FAIL: temporal:cycle_count_match: baseline=800, treatment=795
WARN: variance:ratio_within_threshold: ratio=3.2, threshold=2.0, claim capped
FAIL: variance:profile_compatible: variance profiles not comparable
```

**Forbidden patterns**:
- No alarm words ("error", "critical", "danger", "alert")
- No speculative language ("may", "might", "could")
- No causal claims ("caused by", "due to", "because")
- No emotional language ("unfortunately", "sadly")

---

## 5. Unit Test Matrix

### 5.1 Test Categories

| Category | Purpose | Count |
|----------|---------|-------|
| Artifact Presence | Verify file existence checks | 4 |
| SHADOW Mode | Verify enforcement/identity checks | 4 |
| Toolchain Parity | Verify hash checks | 4 |
| Window Alignment | Verify window checks | 4 |
| Temporal Structure | Verify temporal checks | 12 |
| Variance Profile | Verify variance checks | 12 |
| Integration | Full run verification | 4 |
| Adversarial | Edge cases and attack vectors | 8 |

### 5.2 Artifact Presence Tests

| Test ID | Description | Expected |
|---------|-------------|----------|
| `test_artifact_temporal_present` | temporal_structure_audit.json exists | PASS |
| `test_artifact_temporal_missing` | temporal_structure_audit.json missing | FAIL |
| `test_artifact_variance_present` | variance_profile_audit.json exists | PASS |
| `test_artifact_variance_missing` | variance_profile_audit.json missing | FAIL |

### 5.3 SHADOW Mode Tests

| Test ID | Description | Expected |
|---------|-------------|----------|
| `test_shadow_experiment_cal_exp_4` | experiment=CAL-EXP-4 | PASS |
| `test_shadow_experiment_wrong` | experiment=CAL-EXP-3 | FAIL |
| `test_shadow_enforcement_false` | enforcement=false | PASS |
| `test_shadow_enforcement_true` | enforcement=true | FAIL |

### 5.4 Toolchain Parity Tests

| Test ID | Description | Expected |
|---------|-------------|----------|
| `test_toolchain_hash_valid` | 64 hex char hash present | PASS |
| `test_toolchain_hash_missing` | toolchain_hash.txt missing | FAIL |
| `test_toolchain_hash_invalid` | hash too short | FAIL |
| `test_toolchain_hash_mismatch` | baseline != treatment hash | FAIL |

### 5.5 Window Alignment Tests

| Test ID | Description | Expected |
|---------|-------------|----------|
| `test_window_pre_registered` | evaluation_window present | PASS |
| `test_window_missing` | evaluation_window missing | FAIL |
| `test_window_arm_aligned` | identical cycle indices | PASS |
| `test_window_arm_misaligned` | different cycle indices | FAIL |

### 5.6 Temporal Structure Tests

| Test ID | Description | Expected |
|---------|-------------|----------|
| `test_temporal_cycle_count_match` | baseline=800, treatment=800 | PASS |
| `test_temporal_cycle_count_mismatch` | baseline=800, treatment=795 | FAIL |
| `test_temporal_baseline_monotonic` | indices strictly increasing | PASS |
| `test_temporal_baseline_non_monotonic` | indices not monotonic | FAIL |
| `test_temporal_treatment_monotonic` | indices strictly increasing | PASS |
| `test_temporal_treatment_non_monotonic` | indices not monotonic | FAIL |
| `test_temporal_coverage_full` | ratio=1.0 | PASS |
| `test_temporal_coverage_partial` | ratio=0.95 | FAIL |
| `test_temporal_structure_compatible` | compatible=true | PASS |
| `test_temporal_structure_incompatible` | compatible=false | FAIL |
| `test_temporal_gap_zero` | no gaps in cycles | PASS |
| `test_temporal_gap_large` | large gaps in cycles | Context-dependent |

### 5.7 Variance Profile Tests

| Test ID | Description | Expected |
|---------|-------------|----------|
| `test_variance_ratio_equal` | ratio=1.0 | PASS |
| `test_variance_ratio_within_threshold` | ratio=1.5, threshold=2.0 | PASS |
| `test_variance_ratio_exceeds_threshold` | ratio=3.0, threshold=2.0 | FAIL/WARN |
| `test_variance_ratio_extreme` | ratio=10.0 | FAIL |
| `test_variance_drift_zero` | no windowed drift | PASS |
| `test_variance_drift_small` | drift within limit | PASS |
| `test_variance_drift_large` | drift exceeds limit | FAIL/WARN |
| `test_variance_profile_compatible` | profile_compatible=true | PASS |
| `test_variance_profile_incompatible` | profile_compatible=false | FAIL |
| `test_variance_baseline_zero` | baseline variance=0 (pathology) | FAIL |
| `test_variance_treatment_zero` | treatment variance=0 (pathology) | FAIL |
| `test_variance_ratio_undefined` | division by zero (baseline=0) | FAIL |

### 5.8 Integration Tests

| Test ID | Description | Expected |
|---------|-------------|----------|
| `test_integration_valid_run` | All checks pass | PASS |
| `test_integration_invalid_temporal` | Temporal structure fails | FAIL |
| `test_integration_invalid_variance` | Variance profile fails | FAIL |
| `test_integration_claim_capped` | Variance exceeds threshold, claim capped | WARN (partial pass) |

### 5.9 Adversarial Tests

| Test ID | Description | Expected |
|---------|-------------|----------|
| `test_adversarial_nan_variance` | NaN in variance values | FAIL |
| `test_adversarial_inf_variance` | Infinity in variance values | FAIL |
| `test_adversarial_negative_variance` | Negative variance | FAIL |
| `test_adversarial_empty_cycles` | Empty cycles.jsonl | FAIL |
| `test_adversarial_malformed_json` | Invalid JSON in audit files | FAIL |
| `test_adversarial_missing_fields` | Required fields missing | FAIL |
| `test_adversarial_extra_fields` | Extra fields in audit (should pass) | PASS |
| `test_adversarial_schema_version_mismatch` | schema_version != 1.0.0 | FAIL |

---

## 6. CI Workflow Sketch

### 6.1 Workflow File

**Path**: `.github/workflows/cal-exp-4-verification.yml`

```yaml
# CAL-EXP-4 Run Verification (NON-GATING)
#
# SHADOW MODE CONTRACT:
# - This workflow is ADVISORY ONLY (never blocks merges)
# - Runs verification if results/cal_exp_4/**/run_config.json exists
# - Validates: temporal structure, variance profile comparability
# - All outputs have mode="SHADOW", enforcement=false
# - continue-on-error: true for verification steps

name: CAL-EXP-4 Verification

on:
  push:
    branches: [integrate/ledger-v0.1, master]
    paths:
      - 'results/cal_exp_4/**'
      - 'scripts/verify_cal_exp_4_run.py'
      - 'tests/ci/test_verify_cal_exp_4_run.py'

  pull_request:
    branches: [integrate/ledger-v0.1, master]
    paths:
      - 'results/cal_exp_4/**'
      - 'scripts/verify_cal_exp_4_run.py'
      - 'tests/ci/test_verify_cal_exp_4_run.py'

  workflow_dispatch:
    inputs:
      run_dir:
        description: 'CAL-EXP-4 run directory'
        required: false
        default: ''
        type: string

env:
  USLA_SHADOW_ENABLED: 'true'
  SHADOW_MODE_ENABLED: 'true'
  PYTHONUTF8: '1'

jobs:
  # -------------------------------------------------------------------------
  # Unit Tests (BLOCKING on script failures only)
  # -------------------------------------------------------------------------
  verifier-tests:
    name: Verifier Unit Tests
    runs-on: ubuntu-latest
    continue-on-error: false  # Script failures block

    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v1

      - name: Install dependencies
        run: uv sync

      - name: Run verifier unit tests
        run: |
          uv run pytest tests/ci/test_verify_cal_exp_4_run.py \
            -v --tb=short \
            -m unit \
            --junit-xml=results/verifier-tests.xml

      - name: Upload test results
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: cal-exp-4-verifier-test-results
          path: results/verifier-tests.xml
          retention-days: 30

  # -------------------------------------------------------------------------
  # Run Verification (NON-GATING - advisory only)
  # -------------------------------------------------------------------------
  verify-runs:
    name: Verify CAL-EXP-4 Runs
    runs-on: ubuntu-latest
    continue-on-error: true  # CRITICAL: Advisory only
    needs: verifier-tests

    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v1

      - name: Install dependencies
        run: uv sync

      - name: Check for CAL-EXP-4 runs
        id: check-runs
        run: |
          echo "=== CHECKING FOR CAL-EXP-4 RUNS ==="

          if [ ! -d "results/cal_exp_4" ]; then
            echo "No results/cal_exp_4 directory found."
            echo "has_runs=false" >> $GITHUB_OUTPUT
            exit 0
          fi

          RUN_DIRS=$(find results/cal_exp_4 -maxdepth 2 -name "run_config.json" -type f 2>/dev/null | xargs -I{} dirname {} || true)

          if [ -z "$RUN_DIRS" ]; then
            echo "No run directories found"
            echo "has_runs=false" >> $GITHUB_OUTPUT
            exit 0
          fi

          echo "has_runs=true" >> $GITHUB_OUTPUT

      - name: Verify all runs
        if: steps.check-runs.outputs.has_runs == 'true'
        continue-on-error: true
        run: |
          echo "=== VERIFYING ALL CAL-EXP-4 RUNS ==="
          echo "SHADOW MODE: Advisory only, non-blocking"

          mkdir -p results/verification_reports

          find results/cal_exp_4 -maxdepth 2 -name "run_config.json" -type f | while read config_file; do
            run_dir=$(dirname "$config_file")
            RUN_ID=$(basename "$run_dir")

            echo "----------------------------------------"
            echo "Verifying: $run_dir"

            if uv run python scripts/verify_cal_exp_4_run.py \
                --run-dir "$run_dir" \
                --output-report "results/verification_reports/${RUN_ID}_verification_report.json"; then
              echo "RESULT: PASS"
            else
              echo "RESULT: FAIL (advisory only)"
            fi
          done

      - name: Upload verification reports
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: cal-exp-4-verification-reports-${{ github.run_number }}
          path: results/verification_reports/
          retention-days: 90

  # -------------------------------------------------------------------------
  # Summary (Always runs, NON-GATING)
  # -------------------------------------------------------------------------
  summary:
    name: Verification Summary
    runs-on: ubuntu-latest
    if: always()
    needs: [verifier-tests, verify-runs]

    steps:
      - name: Report Status
        run: |
          echo "========================================"
          echo "CAL-EXP-4 VERIFICATION SUMMARY"
          echo "========================================"
          echo ""
          echo "SHADOW MODE CONTRACT:"
          echo "  - Verification results: ADVISORY ONLY"
          echo "  - This workflow NEVER blocks merges"
          echo "  - Mode marker: SHADOW"
          echo "  - Enforcement: FALSE"
          echo ""
          echo "CHECKS PERFORMED:"
          echo "  - Artifact presence (temporal_structure_audit, variance_profile_audit)"
          echo "  - Temporal structure comparability"
          echo "  - Variance profile comparability (fail-close rules)"
          echo "  - All CAL-EXP-3 checks (inherited)"
          echo ""
          echo "RESULTS:"
          echo "  Verifier Tests: ${{ needs.verifier-tests.result }}"
          echo "  Run Verification: ${{ needs.verify-runs.result }}"

          if [ "${{ needs.verifier-tests.result }}" = "failure" ]; then
            echo ""
            echo "STATUS: VERIFIER TESTS FAILED (script issue)"
            exit 1
          fi

          echo ""
          echo "STATUS: COMPLETE (verification results are advisory)"
```

### 6.2 Trigger Scope

| Trigger | Condition |
|---------|-----------|
| Push | Changes to `results/cal_exp_4/**`, `scripts/verify_cal_exp_4_run.py`, `tests/ci/test_verify_cal_exp_4_run.py` |
| PR | Same paths as push |
| Manual | `workflow_dispatch` with optional `run_dir` input |

### 6.3 Gating Rules

| Job | Gating Behavior |
|-----|-----------------|
| `verifier-tests` | **GATING** - Script failures block (tests verifier code, not run results) |
| `verify-runs` | **NON-GATING** - `continue-on-error: true` (SHADOW MODE) |
| `summary` | **NON-GATING** - Always runs |

---

## 7. Implementation Readiness Checklist

Before coding begins, the following MUST exist:

### 7.1 Specification Dependencies

| Item | Status | Owner |
|------|--------|-------|
| `CAL_EXP_4_VARIANCE_STRESS_SPEC.md` merged | ASSUMED BINDING | Topologist |
| Variance ratio threshold defined | CONFIGURABLE (schema has `thresholds.variance_ratio_max`) | Topologist |
| Windowed drift threshold defined | CONFIGURABLE (schema has `thresholds.windowed_drift_max`) | Topologist |
| Fail-close vs claim-cap rules specified | CONFIGURABLE (schema has `claim_cap_applied`, `claim_cap_level`) | Topologist |
| Claim level cap rules specified | CONFIGURABLE (schema supports L0-L5 cap) | Topologist |

### 7.2 Schema Dependencies

| Item | Status | Owner |
|------|--------|-------|
| `schemas/cal_exp_4/temporal_structure_audit.schema.json` | **DONE** (v1.0.0) | CLAUDE R |
| `schemas/cal_exp_4/variance_profile_audit.schema.json` | **DONE** (v1.0.0) | CLAUDE R |

### 7.3 Code Dependencies

| Item | Status | Owner |
|------|--------|-------|
| `scripts/verify_cal_exp_3_run.py` exists and tested | **DONE** | CLAUDE R |
| Python 3.11+ available | **DONE** | Infrastructure |
| pytest with markers available | **DONE** | Infrastructure |

### 7.4 Documentation Dependencies

| Item | Status | Owner |
|------|--------|-------|
| `CAL_EXP_4_INDEX.md` created | PENDING | Topologist |
| Check names finalized | **DONE** (17 checks defined in §3) | CLAUDE R |
| Failure message formats approved | **DONE** (neutral format per §4) | CLAUDE R |

### 7.5 Pre-Implementation Approval

| Item | Status | Approver |
|------|--------|----------|
| Spec merged and CANONICAL | ASSUMED | STRATCOM |
| This verifier plan approved | READY FOR REVIEW | STRATCOM |
| Check semantics match spec invalidations | READY (schema fields bound to checks) | Topologist + CLAUDE R |
| Schemas committed | **DONE** | CLAUDE R |

---

## 8. Open Questions (Resolved via Schema)

The following were open questions, now resolved by schema design:

| Question | Resolution |
|----------|------------|
| Variance ratio threshold | Configurable via `thresholds.variance_ratio_max` / `variance_ratio_min` in audit artifact |
| Windowed drift threshold | Configurable via `thresholds.windowed_drift_max` in audit artifact |
| Fail-close vs claim-cap | Schema has both `variance_profile_pass` (fail-close) and `claim_cap_applied`/`claim_cap_level` (cap) |
| Claim cap level | Schema supports `claim_cap_level` with values L0-L5 or null |
| Temporal coverage tolerance | Schema defaults to 1.0 via `thresholds.min_coverage_ratio` |
| Sub-window definition | Schema has `windowed_analysis` section with `window_count` and per-window arrays |

**Implementation note**: The harness that generates the audit artifacts is responsible for setting threshold values per the spec. The verifier reads the pass/fail flags from the artifact and does not recompute thresholds.

---

## 9. Schema Validation Contract

The verifier MUST validate audit artifacts against schemas before reading fields:

```python
# Pseudo-code for verifier
def validate_audit_artifact(path: Path, schema_path: Path) -> bool:
    """Validate JSON artifact against JSON Schema."""
    artifact = json.load(path)
    schema = json.load(schema_path)
    # Use jsonschema or equivalent
    return jsonschema.validate(artifact, schema)
```

**Fail-close behavior**: If schema validation fails, the entire run is INVALID. The verifier does not attempt to read fields from malformed artifacts.

**Schema version enforcement**: The verifier MUST check `schema_version == "1.0.0"`. Future schema versions require verifier updates.

---

**SHADOW MODE** - observational only, non-blocking.

*Schemas defined. Implementation unblocked pending verifier code.*
