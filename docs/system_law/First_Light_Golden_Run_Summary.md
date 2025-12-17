# First Light Golden Run Summary

**Schema Version:** 1.0.0
**Run Date:** 2025-12-11T04:49:05Z
**Mode:** SHADOW (observation-only)

---

## Configuration (Frozen)

| Parameter     | Value              |
|---------------|--------------------|
| Slice         | arithmetic_simple  |
| Runner        | u2                 |
| Cycles        | 1000               |
| Seed          | 42                 |
| tau_0         | 0.20               |
| Window Size   | 50                 |

Config frozen in: `docs/system_law/first_light_golden_run_config.json`

---

## P3 Synthetic Results

| Metric                | Value   | Threshold | Status |
|-----------------------|---------|-----------|--------|
| Success Rate (final)  | 85.2%   | >= 75%    | PASS   |
| Delta-P Success Slope | +0.0123 | > 0       | PASS   |
| Mean RSI              | 0.745   | >= 0.6    | PASS   |
| Omega Occupancy       | 85.1%   | >= 90%    | FAIL   |
| CDI-010 Activations   | 0       | = 0       | PASS   |
| HARD-OK Rate          | 98.1%   | >= 80%    | PASS   |

**Red Flag Summary:** 318 total (138 INFO, 161 WARNING, 19 CRITICAL)
**Hypothetical Abort:** Yes (omega occupancy below threshold)

---

## P4 Shadow Coupling Results

| Metric                     | Value   |
|----------------------------|---------|
| Real Success Rate          | 92.7%   |
| Twin Success Accuracy      | 88.6%   |
| Blocked Prediction Acc.    | 100.0%  |
| Omega Prediction Acc.      | 97.3%   |
| HARD-OK Prediction Acc.    | 96.4%   |
| Divergence Rate            | 97.2%   |
| Max Divergence Streak      | 165     |

**Divergence Breakdown:**
- By Type: state=184, outcome=37, combined=77
- By Severity: minor=710, moderate=261, severe=0

---

## Noise Baseline & P5 Expectations

### Current P4 Divergence (Mock Baseline)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| P4 divergence_rate (mock baseline) | **~0.97** | Noise baseline — alarm system working correctly |

The 97% divergence rate establishes the **noise baseline**: how the alarm system behaves when connected to `MockTelemetryProvider` (a noisy, non-physical source) and an uncalibrated Twin. The Twin correctly refuses to hallucinate alignment with mock noise.

### P5 Target Bands (Provisional)

The following bands are **calibration hypotheses, not guarantees**. They will be updated after the first real-telemetry run.

| Divergence Rate | Interpretation | Action |
|-----------------|----------------|--------|
| < 0.30 | **Acceptable alignment** | Real-telemetry target band |
| 0.30 – 0.70 | **Needs investigation** | Twin may need recalibration |
| > 0.70 | **Unacceptable alignment** | Twin not tracking real behavior; requires intervention |

### P5 Comparison Expectations

When Phase X P5 is executed:
- Divergence should drop **significantly** below the 0.97 mock baseline
- Ideally divergence < 0.30 (acceptable alignment)
- If divergence remains > 0.70, the Twin requires fundamental recalibration

**Note:** These are provisional targets based on first-principles reasoning. Actual thresholds will be refined based on P5 empirical data.

---

## SHADOW MODE Compliance

| Requirement                      | Verified |
|----------------------------------|----------|
| No governance API calls          | YES      |
| No real state mutation           | YES      |
| All divergences action=LOGGED_ONLY | YES    |
| All observations mode=SHADOW     | YES      |
| Config rejects shadow_mode=False | YES      |

**Compliance Tests:** 14/14 PASSED

---

## Evidence Pack Contents

```
results/first_light/evidence_pack_first_light/
+-- p3_synthetic/           (6 files)
|   +-- synthetic_raw.jsonl
|   +-- stability_report.json
|   +-- red_flag_matrix.json
|   +-- metrics_windows.json
|   +-- tda_metrics.json
|   +-- run_config.json
+-- p4_shadow/              (6 files)
|   +-- real_cycles.jsonl
|   +-- twin_predictions.jsonl
|   +-- divergence_log.jsonl
|   +-- p4_summary.json
|   +-- twin_accuracy.json
|   +-- run_config.json
+-- compliance/
|   +-- compliance_narrative.md
+-- visualizations/
|   +-- README.md
+-- manifest.json
```

**Total Files:** 14
**Manifest SHA-256 Integrity:** All files hashed

### Proof Log Hash Snapshot (Optional)

Shadow-mode provenance teams can add `compliance/proof_log_snapshot.json` next to the compliance narrative to capture the canonical proof log fingerprint. Run:

```
python -m scripts.first_light_proof_hash_snapshot \
    --proof-log results/first_light/golden_run/p3/proofs.jsonl \
    --output results/first_light/evidence_pack_first_light/compliance/proof_log_snapshot.json
```

Run `python scripts/build_first_light_evidence_pack.py ... --include-proof-snapshot` or set `FIRST_LIGHT_INCLUDE_PROOF_SNAPSHOT=1` (with optional `FIRST_LIGHT_PROOF_LOG=<path>`) when invoking the builder to have it execute the same snapshot automatically without breaking SHADOW-mode pack generation if the proof log is missing.

---

## Determinism Verification

| Artifact                 | Run 1 vs Run 2 |
|--------------------------|----------------|
| P3 stability_report.json | IDENTICAL      |
| P3 tda_metrics.json      | IDENTICAL      |
| P3 metrics_windows.json  | IDENTICAL      |
| P3 synthetic_raw.jsonl   | IDENTICAL*     |
| P4 p4_summary.json       | IDENTICAL      |
| P4 twin_accuracy.json    | IDENTICAL      |
| P4 real_cycles.jsonl     | IDENTICAL*     |
| P4 twin_predictions.jsonl| IDENTICAL*     |

*Ignoring timestamp fields

---

## Artifacts Location

| Artifact                    | Path                                              |
|-----------------------------|---------------------------------------------------|
| Golden Config               | docs/system_law/first_light_golden_run_config.json|
| P3 Golden Run               | results/first_light/golden_run/p3/                |
| P4 Golden Run               | results/first_light/golden_run/p4/                |
| Evidence Pack               | results/first_light/evidence_pack_first_light/    |
| Compliance Tests            | tests/integration/test_shadow_mode_compliance.py  |
| Determinism Script          | scripts/verify_first_light_determinism.py         |
| Evidence Builder            | scripts/build_first_light_evidence_pack.py        |

---

## Key Observations

1. **P3 omega_occupancy (85.1%) below 90% threshold** - synthetic twin exits safe region more than expected; this would trigger hypothetical abort under live governance.

2. **P4 divergence rate (97.2%) is high** - expected for shadow mode with mock telemetry; indicates twin predictions differ from real outcomes on most cycles.

3. **Twin success prediction accuracy (88.6%)** - reasonable correlation between shadow twin and real runner outcomes.

4. **No severe divergences** - all 972 divergences classified as minor (710) or moderate (261).

5. **SHADOW MODE contract fully honored** - all 14 compliance tests pass; no governance mutation occurred.

---

## Compliance Statement

This First Light Golden Run was executed in **SHADOW MODE only**. No governance decisions were modified or enforced. All divergence records have `action="LOGGED_ONLY"`. All cycle observations have `mode="SHADOW"`.

This summary is machine-generated from artifacts produced by:
- `scripts/usla_first_light_harness.py` (P3)
- `scripts/usla_first_light_p4_harness.py` (P4)
- `scripts/build_first_light_evidence_pack.py` (packaging)

---

## Portability & Paths

**Current Status:** The `manifest.json` file paths use OS-native separators. On Windows (where this golden run was generated), paths appear with backslashes:

```json
"path": "evidence_pack_first_light\\p3_synthetic\\synthetic_raw.jsonl"
```

**Cross-Platform Note:** External verifiers on Unix/macOS systems should normalize these paths when parsing the manifest. The verification script in `First_Light_External_Verification.md` includes path normalization logic.

**Future Revision:** A subsequent update will standardize `manifest.json` to use POSIX-style forward slashes (`/`) for cross-platform compatibility. This is a documentation/tooling change only and does not affect artifact integrity — SHA-256 hashes remain valid regardless of path separator convention in the manifest.

---

## Phase X P5 Update Path (Real Telemetry)

This section outlines how this document will be updated when real telemetry is integrated in Phase X P5.

### P5 Divergence Expectations

The current 97.2% divergence rate reflects MockTelemetryProvider + uncalibrated twin — this is expected and validates that divergence detection works correctly under stress.

**Target bands for P5 (tentative, subject to calibration):**

| Divergence Rate | Interpretation |
|-----------------|----------------|
| < 0.30 (30%) | **Acceptable** — twin tracks real runner well |
| 0.30 – 0.50 | **Marginal** — may need recalibration |
| 0.50 – 0.70 | **Concerning** — investigate model assumptions |
| > 0.70 (70%) | **Unstable** — requires intervention before proceeding |

A successful P5 run should show divergence **significantly below 97.2%** (ideally < 30%). These thresholds will be refined based on initial P5 results and statistical analysis.

### What Will Change

| Metric/Artifact | Current (Mock) | Expected (Real) |
|-----------------|----------------|-----------------|
| P4 divergence_rate | 97.2% | Should decrease significantly (twin tracks real) |
| P4 twin_success_accuracy | 88.6% | Should increase (meaningful correlation) |
| P4 omega_prediction_accuracy | 97.3% | May change based on real stability patterns |
| P3 thresholds | May need tuning | Calibrated to real behavior |
| Telemetry source | MockTelemetryProvider | Real runner adapter |

### What Will Stay the Same

- Harness invocation commands (same flags, same scripts)
- Evidence pack structure (p3_synthetic/, p4_shadow/, manifest.json)
- Determinism verification procedure
- SHADOW MODE compliance tests (14 tests)
- Hash verification workflow
- Status JSON schema

### New Caveats Expected

1. **Real-runner stability:** Real telemetry may show different stability patterns than synthetic data. Omega occupancy thresholds may need adjustment.

2. **U2 integration latency:** Real telemetry introduces I/O latency. Determinism checks may need timing tolerance.

3. **Data volume:** Real runs may produce larger JSONL files. Evidence pack size may increase.

4. **External dependencies:** Real telemetry requires runner infrastructure to be operational.

### Document Update Procedure

When Phase X P5 is ready:

1. Run new golden run with real telemetry adapter
2. Update this summary with new metric values
3. Add "P5 Real Telemetry" section with comparison to mock baseline
4. Update Known Caveats with any new issues discovered
5. Regenerate evidence pack and update manifest hashes

---

**Generated:** 2025-12-11
**Run IDs:** fl_20251211_044905_seed42 (P3), p4_20251211_044926 (P4)
