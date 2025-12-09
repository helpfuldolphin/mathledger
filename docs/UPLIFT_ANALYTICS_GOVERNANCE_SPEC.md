# Uplift Analytics Governance Specification

**Version**: 1.0.0
**Status**: PHASE II — NOT RUN IN PHASE I
**Date**: 2025-12-06
**Author**: Claude I (Analytics Constitutional Engineer)

---

## 1. Abstract

This specification defines the **Governance Law** for U2 uplift analytics. It establishes the constitutional rules that govern how uplift experiments are validated, how results are certified, and how governance decisions (PROCEED/HOLD/ROLLBACK) are made.

The Governance Law is the **legal layer** on top of the analytics computation layer. It does not modify thresholds or Δp semantics; it verifies that analyses conform to preregistered methods and produce valid, reproducible results.

---

## 2. Scope

This specification governs:

1. **Verification of summary.json** — The primary analysis output
2. **Verification of manifest.json** — The experiment configuration and reproducibility record
3. **Verification of telemetry summaries** — Real-time metrics collected during execution
4. **Governance decision certification** — Whether PROCEED/HOLD/ROLLBACK is lawful

This specification does NOT govern:

- Threshold values (defined in PREREG_UPLIFT_U2.yaml)
- Δp computation semantics (defined in fo_analytics.py theoretical framework)
- RFL policy mechanics (defined in RFL_LAW.md)

---

## 3. Constitutional Principles

### 3.1 Principle of Determinism (P-DET)

> All analytics computations MUST be deterministic. Given identical inputs and seeds, the output MUST be byte-for-byte identical.

**Rationale**: Non-deterministic results cannot be audited or reproduced. Any randomness must be seeded.

### 3.2 Principle of Preregistration (P-PRE)

> No analysis parameter may be modified after data collection begins. All methods, thresholds, and sample sizes MUST be locked in PREREG_UPLIFT_U2.yaml before the first cycle runs.

**Rationale**: Post-hoc modifications enable p-hacking and invalidate statistical guarantees.

### 3.3 Principle of Traceability (P-TRC)

> Every governance decision MUST trace to specific data points, computations, and criteria. The path from raw logs to final decision MUST be reconstructible.

**Rationale**: Auditors must be able to verify any claim by following the evidence chain.

### 3.4 Principle of Conservatism (P-CON)

> When in doubt, HOLD. Ambiguous results or marginal passes MUST trigger human review rather than automatic PROCEED.

**Rationale**: False positives (claiming uplift when none exists) are more harmful than false negatives.

### 3.5 Principle of Transparency (P-TRN)

> All violations, warnings, and marginal conditions MUST be reported. No failure may be silently ignored.

**Rationale**: Hidden failures undermine trust and auditability.

---

## 4. Rule Categories

The Governance Law comprises four rule categories:

| Category | Prefix | Scope | Count |
|----------|--------|-------|-------|
| **Governance Rules** | GOV- | Decision-making and threshold compliance | 12 |
| **Reproducibility Rules** | REP- | Determinism and seed management | 8 |
| **Manifest Rules** | MAN- | Configuration and artifact integrity | 10 |
| **Invariant Rules** | INV- | Data and statistical invariants | 13 |

Total: **43 rules**

---

## 5. Governance Rules (GOV-)

### GOV-1: Threshold Compliance

Each slice MUST be evaluated against its predefined success criteria.

| Slice | min_SR | max_AR | min_Δ_Θ% | min_n |
|-------|--------|--------|----------|-------|
| prop_depth4 | 0.95 | 0.02 | 5.0 | 500 |
| fol_eq_group | 0.85 | 0.10 | 3.0 | 300 |
| fol_eq_ring | 0.80 | 0.15 | 2.0 | 300 |
| linear_arith | 0.70 | 0.20 | 0.0 | 200 |

**Severity**: INVALIDATING
**Inputs**: summary.json → slices → {slice_id} → success_rate, abstention_rate, throughput_uplift_pct, n_rfl

### GOV-2: Decision Exclusivity

Exactly one of {PROCEED, HOLD, ROLLBACK} MUST be true.

**Severity**: INVALIDATING
**Inputs**: summary.json → governance → recommendation

### GOV-3: Decision Consistency

- PROCEED ⟹ all_slices_pass = true
- all_slices_pass = false ⟹ recommendation ∈ {HOLD, ROLLBACK}

**Severity**: INVALIDATING
**Inputs**: summary.json → governance → {all_slices_pass, recommendation}

### GOV-4: Failing Slice Identification

- ∀s ∈ failing_slices: PASS(s) = false
- ∀s ∉ failing_slices: PASS(s) = true
- passing_slices ∪ failing_slices = all slices

**Severity**: INVALIDATING
**Inputs**: summary.json → governance → {passing_slices, failing_slices}

### GOV-5: Marginal Case Flagging

If any slice has: CI_lower < threshold < CI_upper (CI overlaps threshold), the case MUST be flagged as marginal.

**Severity**: WARNING
**Inputs**: summary.json → slices → {slice_id} → throughput → {ci_low, ci_high}

### GOV-6: HOLD Rationale Required

If recommendation = HOLD, a rationale MUST be provided identifying which slices are marginal.

**Severity**: WARNING
**Inputs**: summary.json → governance → {recommendation, rationale}

### GOV-7: ROLLBACK Rationale Required

If recommendation = ROLLBACK, a rationale MUST be provided identifying which slices failed and why.

**Severity**: INVALIDATING
**Inputs**: summary.json → governance → {recommendation, rationale, failing_slices}

### GOV-8: Sample Size Minimum

Each slice MUST meet its minimum sample size requirement (n_rfl ≥ min_n).

**Severity**: INVALIDATING
**Inputs**: summary.json → slices → {slice_id} → n_rfl; PREREG → slice_criteria → min_n

### GOV-9: All Slices Present

All four slices (prop_depth4, fol_eq_group, fol_eq_ring, linear_arith) MUST be present in the summary.

**Severity**: INVALIDATING
**Inputs**: summary.json → slices (keys)

### GOV-10: No Unreported Failures

No slice failure may be omitted from failing_slices.

**Severity**: INVALIDATING
**Inputs**: summary.json → slices (computed PASS predicate) vs governance → failing_slices

### GOV-11: Confidence Level Match

The confidence level used MUST match the preregistered level (default 0.95).

**Severity**: INVALIDATING
**Inputs**: summary.json → reproducibility → confidence; PREREG → confidence_level

### GOV-12: Statistical Method Match

Wilson CI MUST be used for proportions; bootstrap MUST be used for continuous metrics.

**Severity**: INVALIDATING
**Inputs**: summary.json → reproducibility (implicit); analysis code audit

---

## 6. Reproducibility Rules (REP-)

### REP-1: Baseline Seed Documented

manifest.json MUST contain seed_baseline as a positive integer.

**Severity**: INVALIDATING
**Inputs**: manifest.json → config → seed_baseline

### REP-2: RFL Seed Documented

manifest.json MUST contain seed_rfl as a positive integer.

**Severity**: INVALIDATING
**Inputs**: manifest.json → config → seed_rfl

### REP-3: Bootstrap Seed Documented

summary.json MUST contain bootstrap_seed as a positive integer.

**Severity**: INVALIDATING
**Inputs**: summary.json → reproducibility → bootstrap_seed

### REP-4: Seed Distinctness

seed_baseline ≠ seed_rfl ≠ bootstrap_seed (all three must be distinct).

**Severity**: WARNING
**Inputs**: manifest.json, summary.json → seeds

### REP-5: Bootstrap Iterations Minimum

n_bootstrap ≥ 10,000.

**Severity**: INVALIDATING
**Inputs**: summary.json → reproducibility → n_bootstrap

### REP-6: Determinism Verification

Re-running analysis with same inputs and seeds MUST produce identical summary.json (excluding generated_at timestamp).

**Severity**: INVALIDATING
**Inputs**: Two summary.json from same inputs

### REP-7: Code Version Recorded

manifest.json MUST contain analysis_code_version (git commit hash or version string).

**Severity**: WARNING
**Inputs**: manifest.json → metadata → analysis_code_version

### REP-8: Raw Data Preserved

Paths to raw JSONL logs MUST be recorded and files MUST exist.

**Severity**: INVALIDATING
**Inputs**: manifest.json → artifacts → {baseline_logs, rfl_logs}

---

## 7. Manifest Rules (MAN-)

### MAN-1: Experiment ID Present

manifest.json MUST contain experiment_id (non-empty string).

**Severity**: INVALIDATING
**Inputs**: manifest.json → experiment_id

### MAN-2: Preregistration Reference

manifest.json MUST reference the preregistration file (prereg_ref).

**Severity**: INVALIDATING
**Inputs**: manifest.json → prereg_ref

### MAN-3: Preregistration File Exists

The referenced preregistration file MUST exist and be readable.

**Severity**: INVALIDATING
**Inputs**: filesystem check on prereg_ref path

### MAN-4: Slice Configuration Complete

manifest.json MUST contain configuration for all four slices.

**Severity**: INVALIDATING
**Inputs**: manifest.json → config → slices (keys)

### MAN-5: Artifact Checksums Present

manifest.json MUST contain SHA-256 checksums for all artifact files.

**Severity**: WARNING
**Inputs**: manifest.json → checksums

### MAN-6: Artifact Checksums Valid

All recorded checksums MUST match the actual file checksums.

**Severity**: INVALIDATING
**Inputs**: manifest.json → checksums vs computed checksums

### MAN-7: Created Timestamp Present

manifest.json MUST contain created_at in ISO 8601 format.

**Severity**: WARNING
**Inputs**: manifest.json → created_at

### MAN-8: Schema Version Present

manifest.json MUST declare its schema version.

**Severity**: WARNING
**Inputs**: manifest.json → $schema or schema_version

### MAN-9: No Extraneous Slices

manifest.json MUST NOT contain slice configurations for undefined slices.

**Severity**: WARNING
**Inputs**: manifest.json → config → slices vs SLICE_IDS

### MAN-10: Derivation Parameters Recorded

For each slice, derivation parameters (steps, depth, breadth, total) MUST be recorded.

**Severity**: WARNING
**Inputs**: manifest.json → config → slices → {slice_id} → derivation_params

---

## 8. Invariant Rules (INV-)

### INV-D1: Cycle Index Continuity

For each condition, cycle indices MUST be consecutive with no gaps.

**Severity**: INVALIDATING
**Inputs**: telemetry summary or raw JSONL → cycle indices

### INV-D2: Timestamp Monotonicity

Timestamps within a condition MUST be strictly increasing.

**Severity**: WARNING
**Inputs**: telemetry summary or raw JSONL → timestamps

### INV-D3: Verification Bound

proofs_succeeded ≤ proofs_attempted for all cycles.

**Severity**: INVALIDATING
**Inputs**: telemetry summary → per-cycle metrics

### INV-D4: Abstention Bound

abstention_count ≤ proofs_attempted for all cycles.

**Severity**: INVALIDATING
**Inputs**: telemetry summary → per-cycle metrics

### INV-D5: Duration Positivity

duration_seconds > 0 for all cycles.

**Severity**: INVALIDATING
**Inputs**: telemetry summary → per-cycle metrics

### INV-D6: Hash Format Validity

ht_hash MUST match regex ^[0-9a-f]{64}$.

**Severity**: INVALIDATING
**Inputs**: telemetry summary → per-cycle ht_hash

### INV-S1: Wilson CI Bounds

For Wilson CIs: 0 ≤ ci_low ≤ ci_high ≤ 1.

**Severity**: INVALIDATING
**Inputs**: summary.json → slices → success_rate/abstention_rate → {ci_low, ci_high}

### INV-S2: Bootstrap CI Ordering

For all CIs: ci_low ≤ ci_high.

**Severity**: INVALIDATING
**Inputs**: summary.json → slices → all CI fields

### INV-S3: Point Estimate Plausibility

Point estimates SHOULD be within their CIs (warning if outside).

**Severity**: WARNING
**Inputs**: summary.json → slices → {baseline, rfl, ci_low, ci_high}

### INV-S4: Rate Bounds

All rate metrics (success_rate, abstention_rate) MUST be in [0, 1].

**Severity**: INVALIDATING
**Inputs**: summary.json → slices → rate fields

### INV-G1: Decision Uniqueness

Exactly one governance decision MUST be set.

**Severity**: INVALIDATING
**Inputs**: summary.json → governance → recommendation

### INV-G2: Pass Consistency

PROCEED requires all_slices_pass = true.

**Severity**: INVALIDATING
**Inputs**: summary.json → governance

### INV-G3: Slice Partition

passing_slices ∪ failing_slices = set(slices) ∧ passing_slices ∩ failing_slices = ∅.

**Severity**: INVALIDATING
**Inputs**: summary.json → governance → {passing_slices, failing_slices}

---

## 9. Rule Severity Classification

### 9.1 Severity Levels

| Level | Code | Meaning | Effect |
|-------|------|---------|--------|
| **INVALIDATING** | I | Rule violation invalidates the analysis | FAIL verdict, block Evidence Pack |
| **WARNING** | W | Rule violation is concerning but not fatal | WARN in report, flag for review |
| **COSMETIC** | C | Minor formatting/style issue | Note in report, no action required |

### 9.2 Severity by Rule

| Rule | Severity | Automatable |
|------|----------|-------------|
| GOV-1 | I | Yes |
| GOV-2 | I | Yes |
| GOV-3 | I | Yes |
| GOV-4 | I | Yes |
| GOV-5 | W | Yes |
| GOV-6 | W | Partial |
| GOV-7 | I | Partial |
| GOV-8 | I | Yes |
| GOV-9 | I | Yes |
| GOV-10 | I | Yes |
| GOV-11 | I | Yes |
| GOV-12 | I | Partial |
| REP-1 | I | Yes |
| REP-2 | I | Yes |
| REP-3 | I | Yes |
| REP-4 | W | Yes |
| REP-5 | I | Yes |
| REP-6 | I | Yes |
| REP-7 | W | Yes |
| REP-8 | I | Yes |
| MAN-1 | I | Yes |
| MAN-2 | I | Yes |
| MAN-3 | I | Yes |
| MAN-4 | I | Yes |
| MAN-5 | W | Yes |
| MAN-6 | I | Yes |
| MAN-7 | W | Yes |
| MAN-8 | W | Yes |
| MAN-9 | W | Yes |
| MAN-10 | W | Yes |
| INV-D1 | I | Yes |
| INV-D2 | W | Yes |
| INV-D3 | I | Yes |
| INV-D4 | I | Yes |
| INV-D5 | I | Yes |
| INV-D6 | I | Yes |
| INV-S1 | I | Yes |
| INV-S2 | I | Yes |
| INV-S3 | W | Yes |
| INV-S4 | I | Yes |
| INV-G1 | I | Yes |
| INV-G2 | I | Yes |
| INV-G3 | I | Yes |

**Summary**: 27 INVALIDATING, 12 WARNING, 0 COSMETIC

---

## 10. Governance Decision Tree

```
                    ┌─────────────────────────────────────┐
                    │         Run Governance Verifier     │
                    └─────────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │    Any INVALIDATING violations?     │
                    └─────────────────────────────────────┘
                           │                    │
                          YES                   NO
                           │                    │
                           ▼                    ▼
              ┌────────────────────┐  ┌────────────────────┐
              │   VERDICT: FAIL   │  │  Any WARNING rules │
              │   Block Evidence  │  │     violated?      │
              │   Pack generation │  └────────────────────┘
              └────────────────────┘         │        │
                                            YES       NO
                                             │        │
                                             ▼        ▼
                              ┌──────────────────┐  ┌──────────────────┐
                              │  VERDICT: WARN   │  │  VERDICT: PASS   │
                              │  Flag for review │  │  Proceed to next │
                              │  Continue with   │  │  stage           │
                              │  caution         │  └──────────────────┘
                              └──────────────────┘
```

---

## 11. Integration Points

### 11.1 CI Pipeline Integration

The Governance Verifier MUST be invoked as a CI check:

```yaml
# .github/workflows/u2-governance.yml
governance-verify:
  runs-on: ubuntu-latest
  steps:
    - name: Verify Governance Rules
      run: |
        python -m backend.governance.verifier \
          --summary results/statistical_summary.json \
          --manifest results/manifest.json \
          --telemetry results/telemetry_summary.json \
          --output results/governance_report.json
    - name: Check Verdict
      run: |
        verdict=$(jq -r '.verdict' results/governance_report.json)
        if [ "$verdict" = "FAIL" ]; then
          echo "Governance verification failed"
          exit 1
        fi
```

### 11.2 Evidence Pack Integration

The governance report MUST be included in the Evidence Pack:

```
evidence_pack_v2/
├── summary.json
├── manifest.json
├── telemetry_summary.json
├── governance_report.json     ← Governance Verifier output
├── raw_logs/
│   ├── u2_prop_depth4_baseline.jsonl
│   └── ...
└── attestation.json
```

### 11.3 Attestation Integration

The attestation MUST reference the governance verdict:

```json
{
  "attestation_id": "...",
  "governance": {
    "verdict": "PASS",
    "violations": [],
    "warnings": ["REP-4"],
    "report_hash": "sha256:..."
  }
}
```

---

## 12. Changelog

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-06 | Claude I | Initial specification |

---

## 13. References

- `backend/metrics/fo_analytics.py` — Theoretical framework (Sections 4-5)
- `docs/RFL_LAW.md` — RFL metabolic contract
- `PREREG_UPLIFT_U2.yaml` — Preregistration (Phase II)
- `docs/DETERMINISM_CONTRACT.md` — Determinism requirements
