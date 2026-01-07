# Evaluation Contract Checklist

**Purpose:** Define minimum requirements for external measurement tools to be admitted as governance-relevant evidence in MathLedger.

**Status:** Reference document
**Not included in fm.tex**

---

## Core Principle

**Measurement ≠ Authority.**

A measurement tool can produce useful signals without those signals being admissible as governance evidence. Admission requires a pinned evaluation contract that enables deterministic replay or bounded distributional reporting.

---

## Required Contract Fields

Any external measurement tool must pin the following fields before its outputs can be treated as authority-bearing:

### 1. Dataset Specification

| Field | Requirement |
|-------|-------------|
| `DATASET_ID` | Unique identifier |
| `DATASET_HASH` | sha256 of canonical dataset file |
| `DATASET_VERSION` | Semantic version or commit hash |
| `DATASET_LOCATION` | URL or artifact path (must be retrievable) |

**Failure mode if missing:** Different evaluations use different inputs; results not comparable.

### 2. Model Specification

| Field | Requirement |
|-------|-------------|
| `MODEL_ID` | Exact model identifier (e.g., `claude-3-7-sonnet-20250219`) |
| `MODEL_VERSION_LOCK` | Commitment to not change without version increment |
| `MODEL_TEMPERATURE` | Fixed value (preferably 0 for determinism) |
| `MODEL_SEED` | Fixed seed if available, or explicit "non-deterministic" with envelope |

**Failure mode if missing:** Model drift changes scores silently; replay impossible.

### 3. Prompt Specification

| Field | Requirement |
|-------|-------------|
| `PROMPT_ID` | Unique identifier |
| `PROMPT_HASH` | sha256 of exact prompt text |
| `PROMPT_TEXT` | Full text (for audit) |
| `PROMPT_VERSION` | Semantic version or commit hash |

**Failure mode if missing:** Prompt drift changes behavior without notice.

### 4. Threshold / Scoring Specification

| Field | Requirement |
|-------|-------------|
| `THRESHOLD_VALUE` | Exact threshold (e.g., τ=0.8) |
| `THRESHOLD_RATIONALE` | Why this value? (even if arbitrary, document it) |
| `SCORING_FUNCTION` | Exact function or algorithm reference |
| `EDGE_CASE_HANDLING` | How ties, nulls, short responses are handled |

**Failure mode if missing:** Threshold changes flip pass/fail; no audit trail.

### 5. Infrastructure Specification

| Field | Requirement |
|-------|-------------|
| `MIN_RPM` | Minimum requests per minute required |
| `MIN_TPM` | Minimum tokens per minute required |
| `MAX_CONCURRENT` | Maximum concurrent requests |
| `BACKOFF_CONTRACT` | Retry policy (exponential backoff params, max retries) |
| `TIMEOUT` | Per-request timeout |

**Failure mode if missing:** Partial runs due to rate limits; incomplete data treated as complete.

### 6. Replay Bundle Specification

| Field | Requirement |
|-------|-------------|
| `REPLAY_BUNDLE_FORMAT` | Artifact structure for deterministic replay |
| `INPUTS_LOGGED` | All inputs captured |
| `OUTPUTS_LOGGED` | All outputs captured (including intermediate) |
| `RANDOMNESS_LOGGED` | Seeds or stochastic outputs logged |
| `REPLAY_COMMAND` | Exact command to reproduce from bundle |

**Failure mode if missing:** Cannot verify claimed results; no audit trail.

---

## Determinism Requirements

### Option A: Full Determinism
- All temperatures = 0
- All seeds fixed
- Replay produces identical outputs

### Option B: Distributional Envelope
- Stochastic components acknowledged
- Multiple runs required (e.g., n=10)
- Report mean ± std, or confidence interval
- Envelope bounds documented

**If neither A nor B is satisfied:** Tool outputs are observer-only signals, not admissible evidence.

---

## Admission Checklist

Before admitting an external measurement tool as governance evidence:

- [ ] Dataset hash pinned and artifact retrievable
- [ ] Model version locked and documented
- [ ] All prompts hash-pinned
- [ ] Thresholds documented with rationale
- [ ] Infrastructure requirements specified
- [ ] Replay bundle format defined
- [ ] Either full determinism OR distributional envelope documented
- [ ] Failure modes enumerated
- [ ] Version increment policy documented

**If any box is unchecked:** Classify as ADV (Advisory) only.

---

## Example: Ariadne Status

| Requirement | Ariadne Status |
|-------------|----------------|
| Dataset pinned | No (not committed) |
| Model version locked | Yes (judge pinned to claude-3-7-sonnet-20250219) |
| Prompts hash-pinned | Partial (located but not hash-pinned) |
| Thresholds documented | Yes (τ=0.8, min_len=10) |
| Infrastructure specified | No (no minimum RPM/TPM) |
| Replay bundle defined | No |
| Determinism/envelope | No (critic temp=0.8, no seed control) |

**Conclusion:** Not admissible as governance evidence. Classify as ADV/observer-only.

---

## Usage

When evaluating any external measurement tool for MathLedger integration:

1. Complete this checklist
2. Document gaps in parking_lot entry for that tool
3. Classify as ADV until all requirements met
4. Re-evaluate when upstream addresses gaps

---

## Status

Reference document.
Not a claim.
Not an obligation.
Not included in fm.tex.
