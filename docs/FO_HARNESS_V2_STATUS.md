# First Organism Harness V2 - Status

## Classification: Phase II - Experimental

**This harness is NOT used in Evidence Pack v1.**

---

## What This File Is

`tests/test_first_organism_harness_v2.py` is an **experimental** unified test class created to consolidate fragmented First Organism test patterns. It provides:

- Composable phase-based testing (UI, Curriculum, Derivation, Attestation, RFL)
- Mock-first architecture (runs without Postgres/Redis)
- Determinism verification (H_t identical across runs)

## What This File Is NOT

- **NOT** the authoritative First Organism test
- **NOT** used to certify Evidence Pack v1
- **NOT** a CI gate for any production claims

## Authoritative Phase I Evidence

The actual Evidence Pack v1 relies exclusively on:

| Artifact | Path | Purpose |
|----------|------|---------|
| Authoritative FO Test | `tests/integration/test_first_organism.py` | Produces sealed attestation |
| Attestation Artifact | `artifacts/first_organism/attestation.json` | H_t proof for Cursor P |
| Baseline Data | `results/fo_baseline.jsonl` | Partial baseline run logs |
| RFL Data | `results/fo_rfl.jsonl` | Partial RFL run logs (330 cycles, degenerate) |

## RFL Evidence Clarification

The RFL evidence in `results/fo_rfl.jsonl` represents a **partial, degenerate run**:

- **Cycles completed**: 330 (not 1000)
- **Status**: Degenerate (policy did not converge)
- **Usage**: Phase I prototype data only

**Harness V2 does not consume or validate RFL evidence.**

The harness operates in mock-only mode with synthetic data. It has no integration with:
- `results/fo_rfl.jsonl`
- `results/fo_baseline.jsonl`
- Any actual RFL runner output
- Any sealed manifests or Dyno Chart data

RFL validation, if performed, must use the authoritative test at `tests/integration/test_first_organism.py`, not this experimental harness.

## Why This Harness Exists

The existing FO test files contain significant duplication:
- `test_first_organism.py` (~2100 lines)
- `test_first_organism_determinism.py`
- `test_first_organism_ledger.py`
- Plus 10+ other `test_first_organism_*.py` files

Harness V2 was created to explore consolidation for **future** development. It is quarantined from Phase I claims.

## Do NOT

- Reference this harness in the research paper
- Use it to support any claims in the pitch deck
- Make CI depend on it for Evidence Pack certification
- Treat its outputs as authoritative

## Status

- **Created**: 2025-01-30 (Phase II work)
- **Status**: Experimental, not production
- **Tests**: 12 passing (mock-only mode)
- **Integration**: None with Phase I evidence chain

---

## Phase II Uplift Extension (Design Only)

**Phase II uplift mode is NOT implemented. As of Phase I, Harness V2 remains synthetic-only.**

This section describes how Harness V2 *could* be extended into an "Uplift Experiment Harness" for future non-degenerate runs. This is design documentation only—no code exists to support uplift mode.

### What an Uplift Experiment Scenario Looks Like

An uplift experiment would compare baseline derivation (no RFL policy) against RFL-guided derivation to measure whether policy feedback reduces abstention rates.

**Inputs (future, not yet implemented):**
- `--baseline-log=<path>`: JSONL file with baseline run cycles (e.g., `results/fo_baseline_v2.jsonl`)
- `--rfl-log=<path>`: JSONL file with RFL-guided run cycles (e.g., `results/fo_rfl_v2.jsonl`)
- `--slice=<name>`: Curriculum slice identifier (must not be a known Phase I plumbing slice)
- `--manifest=<path>`: Sealed manifest with H_t roots for both runs

**Preconditions for Legitimate Uplift (must ALL be true):**
1. Baseline abstention rate is non-degenerate: `0.20 <= baseline_abstention <= 0.80`
2. RFL abstention rate is strictly lower than baseline: `rfl_abstention < baseline_abstention`
3. Both logs have >= 1000 complete cycles
4. Slice name is NOT in the blocked list: `["first-organism-pl", "fo-plumbing", "phase1-*"]`
5. Manifest contains valid H_t roots for both baseline and RFL runs
6. Manifest is cryptographically sealed (SHA256 of contents matches header)

**Checks the harness would perform:**
- Parse both JSONL files and compute per-cycle abstention rates
- Reject if either log is all-abstain (abstention_rate > 0.99)
- Reject if either log is all-success (abstention_rate < 0.01)
- Reject if cycle counts differ by more than 5%
- Reject if slice name matches Phase I plumbing patterns

**Outputs (future, not yet implemented):**
- `uplift_summary.json`: Baseline vs RFL abstention rates, delta, confidence interval
- `abstention_trajectory.png`: Line plot of per-cycle abstention for both runs
- `uplift_verdict.txt`: PASS/FAIL with reasoning

### Proposed CLI Interface (Phase II)

```
python -m tests.test_first_organism_harness_v2 \
    --mode=uplift \
    --baseline-log=results/fo_baseline_v2.jsonl \
    --rfl-log=results/fo_rfl_v2.jsonl \
    --slice=wide-slice-pl4 \
    --manifest=artifacts/uplift_manifest.json
```

### Why Phase I Evidence Does NOT Satisfy These Preconditions

| Precondition | Phase I Status | Reason |
|--------------|----------------|--------|
| Baseline abstention 0.20–0.80 | FAIL | `fo_baseline.jsonl` has ~100% abstention (degenerate) |
| RFL abstention < baseline | FAIL | `fo_rfl.jsonl` has ~100% abstention (degenerate) |
| >= 1000 cycles | FAIL | `fo_rfl.jsonl` has only 330 cycles |
| Slice not Phase I plumbing | FAIL | Current slices are all Phase I test slices |
| Manifest sealed | N/A | No uplift manifest exists |

**Conclusion:** Phase I logs cannot be used with uplift mode. The harness would reject them at the precondition check stage.

### Rejection Behavior (Future Implementation)

When invoked in uplift mode with invalid inputs, the harness would:

1. Print explicit rejection reason:
   ```
   [REJECT] Uplift mode preconditions not met:
     - fo_rfl.jsonl: abstention_rate=1.00 > 0.99 (degenerate, all-abstain)
     - fo_rfl.jsonl: cycle_count=330 < 1000 (insufficient data)
     - Slice 'first-organism-pl' is in Phase I blocked list
   ```

2. Exit with non-zero status code

3. Write no output files

4. Log rejection to `artifacts/uplift_rejections.log` for audit

### Current State

| Capability | Implemented? |
|------------|--------------|
| `--mode=synthetic` (mock data) | Yes |
| `--mode=uplift` (real log comparison) | **No** |
| JSONL log parsing | **No** |
| Precondition validation | **No** |
| Uplift metrics/plots | **No** |
| CLI argument parsing | **No** |

**Harness V2 is the natural home for uplift experiments in Phase II, but it is completely inert for uplift right now.**

---

## FAQ

**Q: Does the existence of RFL logs (`fo_rfl.jsonl`) activate Harness V2?**

A: No. Harness V2 is completely independent of RFL logs. It uses only synthetic mock data and has no code path that reads, parses, or validates any `.jsonl` files from `results/`. The presence or absence of RFL evidence has no effect on Harness V2 behavior.

**Q: Can I run uplift experiments with Harness V2 today?**

A: No. Uplift mode is not implemented. The `--mode=uplift` interface described above is a Phase II design specification only. Attempting to pass these arguments today would result in an unrecognized argument error.

**Q: What would happen if someone tried to use Phase I logs with future uplift mode?**

A: The harness would reject them. Phase I logs (`fo_baseline.jsonl`, `fo_rfl.jsonl`) are degenerate (all-abstain) and would fail the precondition check: `abstention_rate > 0.99` triggers immediate rejection with an explicit error message.
