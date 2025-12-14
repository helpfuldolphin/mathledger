# CAL-EXP-3: Implementation Plan

**Status**: IMPLEMENTATION PLAN (PROVISIONAL)
**Authority**: Derived from `CAL_EXP_3_UPLIFT_SPEC.md`
**Date**: 2025-12-13
**Scope**: Execution machinery only
**Mutability**: Editable until execution begins

---

## Purpose

This document translates the binding charter (`CAL_EXP_3_UPLIFT_SPEC.md`) into concrete execution steps. It specifies harness structure, seed discipline, window registration, artifact layout, and verifier extensions.

**This document MUST NOT**:
- Introduce new metrics
- Add new claims
- Modify uplift definitions
- Reference pilot or external data

**Goal**: Execution without interpretation.

---

## 1. Harness Structure

### 1.1 Dual-Arm Architecture

CAL-EXP-3 requires two parallel execution arms with identical inputs and divergent learning state.

```
┌─────────────────────────────────────────────────────────────┐
│                    CAL-EXP-3 HARNESS                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐         ┌─────────────────┐           │
│  │  BASELINE ARM   │         │  TREATMENT ARM  │           │
│  │  (Learning OFF) │         │  (Learning ON)  │           │
│  ├─────────────────┤         ├─────────────────┤           │
│  │ RFL: DISABLED   │         │ RFL: ENABLED    │           │
│  │ Params: FIXED   │         │ Params: ADAPTIVE│           │
│  │ Seed: S         │         │ Seed: S         │           │
│  │ Corpus: C       │         │ Corpus: C       │           │
│  └────────┬────────┘         └────────┬────────┘           │
│           │                           │                     │
│           ▼                           ▼                     │
│  ┌─────────────────┐         ┌─────────────────┐           │
│  │ Δp_baseline[]   │         │ Δp_treatment[]  │           │
│  └─────────────────┘         └─────────────────┘           │
│                                                             │
│                       ▼                                     │
│              ┌─────────────────┐                           │
│              │  ΔΔp COMPUTATION │                          │
│              │  (Post-Execution)│                          │
│              └─────────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Arm Configuration

**Reference**: `CAL_EXP_3_UPLIFT_SPEC.md` § "Baseline Arm (Control)" and "Treatment Arm (Learning ON)"

| Parameter | Baseline Arm | Treatment Arm |
|-----------|--------------|---------------|
| `learning_enabled` | `false` | `true` |
| `rfl_active` | `false` | `true` |
| `parameter_adaptation` | `false` | `true` |
| `seed` | `S` (shared) | `S` (shared) |
| `corpus_path` | `C` (shared) | `C` (shared) |
| `initial_state` | `I` (shared) | `I` (shared) |

### 1.3 Execution Order

Arms MUST be executed in isolation to prevent cross-contamination:

```
1. Generate shared corpus C
2. Record seed S
3. Snapshot initial state I
4. Execute BASELINE arm (learning OFF) → Δp_baseline[]
5. Reset to initial state I
6. Execute TREATMENT arm (learning ON) → Δp_treatment[]
7. Compute ΔΔp from windowed analysis
```

**Alternative**: Parallel execution is permitted if and only if both arms operate on independent copies of state with no shared mutable resources.

---

## 2. Seed Discipline

**Reference**: `CAL_EXP_3_UPLIFT_SPEC.md` § "Validity Conditions" (Toolchain parity)

### 2.1 Seed Requirements

| Requirement | Specification |
|-------------|---------------|
| Seed source | Single integer seed `S` |
| Seed logging | MUST be recorded in `run_config.json` |
| Seed sharing | MUST be identical across both arms |
| Seed determinism | Given `S`, both arms produce deterministic outputs |

### 2.2 Seed Registration

Before execution begins:

```json
{
  "seed": 42,
  "seed_registered_at": "2025-12-13T00:00:00Z",
  "seed_source": "pre-registered"
}
```

**Post-hoc seed selection is forbidden** per `CAL_EXP_3_UPLIFT_SPEC.md` § "Explicit Invalidations" (post-hoc hypothesis).

---

## 3. Window Registration

**Reference**: `CAL_EXP_3_UPLIFT_SPEC.md` § "Formal Definition of Uplift" (evaluation window W)

### 3.1 Pre-Registration Requirement

Evaluation windows MUST be declared before execution begins.

```json
{
  "windows": {
    "warm_up_exclusion": {
      "start_cycle": 0,
      "end_cycle": 200,
      "included_in_analysis": false
    },
    "evaluation_window": {
      "start_cycle": 201,
      "end_cycle": 1000,
      "included_in_analysis": true
    }
  },
  "window_registered_at": "2025-12-13T00:00:00Z"
}
```

### 3.2 Window Boundaries

| Window | Cycles | Purpose |
|--------|--------|---------|
| Warm-up exclusion | 0-200 | Excluded from ΔΔp computation |
| Evaluation window | 201-1000 | Included in ΔΔp computation |

**Reference**: Warm-up exclusion prevents F1.4 (Warm-up inclusion) per `CAL_EXP_3_UPLIFT_SPEC.md` § "Failure Taxonomy".

### 3.3 Sub-Window Analysis (Optional)

For windowed analysis reporting:

| Sub-Window | Cycles | Label |
|------------|--------|-------|
| W1 | 201-400 | Early |
| W2 | 401-600 | Mid |
| W3 | 601-800 | Late |
| W4 | 801-1000 | Final |

Sub-windows enable detection of F3.3 (Monotonicity assumption).

### 3.4 Window Semantics (Binding)

| Property | Specification |
|----------|---------------|
| Cycle range bounds | **Inclusive** on both ends (e.g., 201-1000 includes cycles 201 and 1000) |
| Missing-cycle handling | **INVALIDATION** — run is void if any cycle in evaluation window is missing |
| Arm alignment | Baseline and treatment MUST have identical cycle indices within evaluation window |

**Rationale**: Missing cycles indicate harness failure or data corruption. There is no valid imputation strategy for Δp values. Arm misalignment prevents valid ΔΔp computation.

---

## 4. Artifact Layout

**Reference**: `CAL_EXP_3_UPLIFT_SPEC.md` § "Required Reporting"

### 4.1 Directory Structure

```
results/cal_exp_3/<run_id>/
├── run_config.json           # Experiment configuration
├── baseline/
│   ├── cycles.jsonl          # Per-cycle Δp values (learning OFF)
│   └── summary.json          # Baseline arm summary statistics
├── treatment/
│   ├── cycles.jsonl          # Per-cycle Δp values (learning ON)
│   └── summary.json          # Treatment arm summary statistics
├── analysis/
│   ├── uplift_report.json    # ΔΔp computation and validity checks
│   └── windowed_analysis.json # Per-window breakdown
├── validity/
│   ├── toolchain_hash.txt    # SHA-256 of runtime environment
│   ├── corpus_manifest.json  # Hash of input corpus
│   └── validity_checks.json  # Pass/fail for each validity condition
└── RUN_METADATA.json         # Final verdict and claim level
```

### 4.2 Required Fields per Artifact

**`run_config.json`**:
```json
{
  "experiment": "CAL-EXP-3",
  "spec_reference": "CAL_EXP_3_UPLIFT_SPEC.md",
  "seed": <int>,
  "cycles": <int>,
  "windows": { ... },
  "baseline_config": {
    "learning_enabled": false,
    "rfl_active": false
  },
  "treatment_config": {
    "learning_enabled": true,
    "rfl_active": true
  },
  "registered_at": "<ISO8601>"
}
```

**`baseline/cycles.jsonl`** (per line):
```json
{
  "cycle": <int>,
  "delta_p": <float>,
  "timestamp": "<ISO8601>"
}
```

**`treatment/cycles.jsonl`** (per line):
```json
{
  "cycle": <int>,
  "delta_p": <float>,
  "timestamp": "<ISO8601>"
}
```

**`analysis/uplift_report.json`**:
```json
{
  "baseline_mean_delta_p": <float>,
  "treatment_mean_delta_p": <float>,
  "delta_delta_p": <float>,
  "standard_error": <float>,
  "evaluation_window": {
    "start_cycle": <int>,
    "end_cycle": <int>
  },
  "n_baseline": <int>,
  "n_treatment": <int>
}
```

**`RUN_METADATA.json`**:
```json
{
  "experiment": "CAL-EXP-3",
  "run_id": "<uuid>",
  "verdict": "<L0|L1|L2|L3|L4|L5>",
  "delta_delta_p": <float>,
  "validity_passed": <bool>,
  "claim_permitted": "<string>",
  "generated_at": "<ISO8601>"
}
```

### 4.3 Artifact Determinism Rules

For reproducibility verification and byte-comparison of artifacts:

| Rule | Specification |
|------|---------------|
| Time-variant fields | `timestamp`, `generated_at`, `registered_at`, `window_registered_at` — strip before comparison |
| JSON canonicalization | `sort_keys=True`, no trailing whitespace, UTF-8 encoding |
| JSONL newline discipline | Unix newlines (`\n`) only, one JSON object per line, no blank lines |
| Per-cycle line fields | `cycle` (int), `delta_p` (float) — `timestamp` is auxiliary metadata, excluded from content comparison |
| No random identifiers | Per-cycle lines MUST NOT contain UUIDs or random IDs; `run_id` appears only in `RUN_METADATA.json` |

**Canonical per-cycle line format**:
```json
{"cycle": 201, "delta_p": 0.847}
```

**Non-canonical (forbidden)**:
```json
{"cycle": 201, "delta_p": 0.847, "id": "a1b2c3d4-..."}
```

---

## 5. Verifier Extensions

### 5.1 Validity Checker

A post-execution validator MUST verify all conditions from `CAL_EXP_3_UPLIFT_SPEC.md` § "Validity Conditions".

| Check | Input | Validation |
|-------|-------|------------|
| Toolchain parity | `validity/toolchain_hash.txt` | Baseline hash == Treatment hash |
| Corpus identity | `validity/corpus_manifest.json` | Baseline corpus == Treatment corpus |
| Window alignment | `run_config.json` | Windows declared before execution |
| No pathology | `baseline/cycles.jsonl`, `treatment/cycles.jsonl` | No NaN, no missing cycles |

### 5.2 Claim Level Assigner

Based on `CAL_EXP_3_UPLIFT_SPEC.md` § "Claim Strength Ladder":

```python
def assign_claim_level(report: dict, validity: dict) -> str:
    if not both_arms_completed(report):
        return "L0"
    if not measurements_obtained(report):
        return "L1"
    if not delta_delta_p_computed(report):
        return "L2"
    if not exceeds_noise_floor(report):
        return "L3"
    if not all_validity_conditions(validity):
        return "L3"  # Cannot advance to L4 without validity
    return "L4"
```

#### 5.2.1 Claim Level Rules (Explicit)

| Level | Requirements | Run Count |
|-------|--------------|-----------|
| L4 | All validity conditions pass AND \|ΔΔp\| > noise_floor | Single run permitted |
| L5 | L4 achieved across ≥3 independent run-pairs with identical toolchain fingerprint and pre-registered windows | Multiple runs required |

**Clarifications**:
- L4 is achievable on a **single run** if all validity conditions hold and the effect exceeds noise floor
- L5 ("replicated") requires **≥3 independent run-pairs** (6 total arm executions)
- All L5 runs must share identical `toolchain_fingerprint` and pre-registered window definitions
- Statistical tests (paired t-test, bootstrap CI) are **OPTIONAL** and **non-binding** for claim level assignment
- If statistical tests are reported, they must be labeled as "supplementary analysis" and not affect the claim level

### 5.3 Noise Floor Estimation

The noise floor is estimated from baseline arm variance:

```
noise_floor = 2 * std(Δp_baseline) / sqrt(n)
```

ΔΔp exceeds noise floor if:
```
|ΔΔp| > noise_floor
```

### 5.4 Toolchain Manifest Alignment

CAL-EXP-3 uses the existing toolchain manifest schema for provenance verification.

**Reference**: `schemas/toolchain_manifest.schema.json`

| Field | Requirement for CAL-EXP-3 |
|-------|---------------------------|
| `schema_version` | Must be `"1.0.0"` |
| `experiment_id` | Must be `"CAL-EXP-3"` |
| `provenance_level` | `"full"` required for L4 claims; `"partial"` caps at L3 |
| `uv_lock_hash` | Required (SHA-256 of uv.lock) |
| `toolchain_fingerprint` | Required if `provenance_level=full` |

**Validity condition**: Baseline and treatment arms MUST have identical `toolchain_fingerprint` values. Mismatch triggers F1.1 (Toolchain drift) and invalidates the run.

**Minimal manifest for CAL-EXP-3** (provenance_level=full):
```json
{
  "schema_version": "1.0.0",
  "experiment_id": "CAL-EXP-3",
  "provenance_level": "full",
  "toolchain_fingerprint": "<sha256>",
  "uv_lock_hash": "<sha256>",
  "timestamp": "<ISO8601>"
}
```

---

## 6. Execution Checklist

### 6.1 Pre-Execution

| Step | Action | Artifact |
|------|--------|----------|
| 1 | Register seed | `run_config.json` |
| 2 | Register windows | `run_config.json` |
| 3 | Generate corpus | `validity/corpus_manifest.json` |
| 4 | Snapshot initial state | (internal) |
| 5 | Record toolchain hash | `validity/toolchain_hash.txt` |

### 6.2 Execution

| Step | Action | Artifact |
|------|--------|----------|
| 6 | Execute baseline arm | `baseline/cycles.jsonl` |
| 7 | Compute baseline summary | `baseline/summary.json` |
| 8 | Reset to initial state | (internal) |
| 9 | Execute treatment arm | `treatment/cycles.jsonl` |
| 10 | Compute treatment summary | `treatment/summary.json` |

### 6.3 Post-Execution

| Step | Action | Artifact |
|------|--------|----------|
| 11 | Run validity checker | `validity/validity_checks.json` |
| 12 | Compute ΔΔp | `analysis/uplift_report.json` |
| 13 | Compute windowed analysis | `analysis/windowed_analysis.json` |
| 14 | Assign claim level | `RUN_METADATA.json` |

---

## 7. Failure Detection

**Reference**: `CAL_EXP_3_UPLIFT_SPEC.md` § "Failure Taxonomy"

### 7.1 Automated Detection

| Failure | Detection Method | Response |
|---------|------------------|----------|
| F1.1 Toolchain drift | Hash mismatch | Abort, report invalid |
| F1.2 Corpus contamination | Manifest mismatch | Abort, report invalid |
| F1.3 Window misalignment | Config audit | Abort, report invalid |
| F1.4 Warm-up inclusion | Window bounds check | Exclude warm-up cycles |
| F2.1 Parameter leakage | Config diff | Abort, report invalid |
| F2.2 Seed divergence | Seed log diff | Abort, report invalid |
| F2.3 External ingestion | Isolation audit | Run INVALIDATED |
| F4.1 Noise floor confusion | Statistical test | Cap claim at L3 |
| F4.4 Variance suppression | Report includes std | (enforced by schema) |

#### 7.1.1 External Ingestion Negative Proof (F2.3)

**Reference**: `CAL_EXP_3_UPLIFT_SPEC.md` § "Explicit Invalidations" (external data ingestion)

The harness MUST verify that during execution:
- **No network calls** were made (socket operations logged and checked post-run)
- **No file reads** occurred outside the pre-registered corpus path

| Check | Method | Outcome |
|-------|--------|---------|
| Network isolation | Socket activity log empty | Pass/Fail |
| Filesystem isolation | All reads within `corpus_path` | Pass/Fail |

**Fail-close behavior**: If either check fails, the run is **INVALIDATED**. This is observational (logged post-run), not runtime-blocking.

**Artifact**: `validity/isolation_audit.json`
```json
{
  "network_calls": [],
  "file_reads_outside_corpus": [],
  "isolation_passed": true
}
```

### 7.2 Manual Review Required

| Failure | Reason |
|---------|--------|
| F2.4 Observer effect | Requires instrumentation review |
| F3.1-F3.4 Interpretive | Requires human judgment on claims |
| F4.2 Single-run inference | Requires multi-run design |
| F4.3 Cherry-picked window | Requires pre-registration audit |

---

## 8. Claim Generation

### 8.1 Permitted Claim Templates

**Reference**: `CAL_EXP_3_UPLIFT_SPEC.md` § "Valid Claims"

```python
def generate_claim(report: dict, level: str) -> str:
    ddp = report["delta_delta_p"]
    se = report["standard_error"]
    window = report["evaluation_window"]

    if level == "L4":
        return f"Measured ΔΔp of {ddp:+.4f} +/- {se:.4f} in cycles {window['start_cycle']}-{window['end_cycle']} under CAL-EXP-3 conditions"
    elif level == "L3":
        return f"Computed ΔΔp of {ddp:+.4f}; validity conditions not fully verified"
    elif level == "L2":
        return f"ΔΔp computed but within noise floor"
    else:
        return f"Experiment reached level {level}; no uplift claim permitted"
```

### 8.2 Forbidden Claim Filter

Any generated claim MUST be filtered against `CAL_EXP_3_UPLIFT_SPEC.md` § "Invalid Claims":

```python
FORBIDDEN_PATTERNS = [
    r"learning works",
    r"system improved",
    r"intelligence",
    r"generalization proven",
    r"uplift will continue",
    r"statistically significant"  # without formal test
]
```

---

## 9. Integration Points

### 9.1 Existing Infrastructure

| Component | Role in CAL-EXP-3 |
|-----------|-------------------|
| `backend/rfl/runner.py` | RFL execution (treatment arm) |
| `backend/topology/first_light/` | Harness template |
| `METRIC_DEFINITIONS.md@v1.1.0` | Δp definition |

### 9.2 New Components Required

| Component | Purpose |
|-----------|---------|
| `scripts/run_cal_exp_3.py` | Harness orchestrator |
| `backend/calibration/uplift_analyzer.py` | ΔΔp computation |
| `backend/calibration/validity_checker.py` | Validity condition verifier |

### 9.3 Schema References

All output artifacts MUST conform to schemas in:
```
docs/system_law/schemas/cal_exp_3/
├── run_config.schema.json
├── cycles.schema.json
├── uplift_report.schema.json
└── run_metadata.schema.json
```

---

## 10. Authorization Gate

**This implementation plan does NOT authorize execution.**

Execution requires:
1. STRATCOM approval of this implementation plan
2. Schema creation and validation
3. Harness code review
4. Explicit execution authorization

---

**SHADOW MODE** — observational only.

*Execution without interpretation.*
