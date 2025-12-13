# CAL-EXP-1: First Internal Calibration Experiment

---

> **STATUS**: EXPERIMENT DESIGN DOCUMENT
>
> **Purpose**: Instrument calibration (not system validation)
>
> **Phase**: Pre-pilot, pre-uplift, pre-enforcement
>
> **Date**: 2025-12-13

---

## 1. Experiment Objective

CAL-EXP-1 validates that MathLedger's observational instruments behave as documented when subjected to controlled perturbations. The experiment tests whether:

1. **Sensitive signals** respond to perturbations they claim to detect
2. **Invariant signals** remain stable under perturbations they should not detect
3. **Barrier enforcement** correctly blocks violations of FORB-XX forbidden edges

This is an instrument calibration experiment. It does not evaluate system performance, model quality, or operational readiness. It answers one question: **Do MathLedger's signals respond to stimuli in the documented manner?**

---

## 2. Hypotheses (Explicit & Testable)

### H1: Determinism Detection

**If** proof ordering is perturbed while proof content remains byte-identical, **then** `determinism_rate` should remain 1.0 (match), **while** `tier_skew` and `semantic_drift` signals must remain unchanged.

*Rationale*: `determinism_rate` measures content equivalence, not ordering. Ordering perturbation should not affect content-based signals.

### H2: Replay Sensitivity

**If** a single proof artifact is modified (content change) during replay, **then** `determinism_rate` should drop below 1.0, **while** monotone ledger invariants (`height`, `hash_chain`, `timestamp`) must remain unviolated (they govern block structure, not proof content).

*Rationale*: `determinism_rate` is sensitive to content divergence; monotone guard governs block ordering.

### H3: Monotone Guard Rejection

**If** a block with `height <= prev_height` is submitted to `check_monotone_ledger()`, **then** a `MonotoneViolation` with `violation_type="height"` must be produced, **while** `determinism_rate` must remain unaffected (it operates at the attestation layer, not the block layer).

*Rationale*: FORB-04 (height monotonicity) is enforced by BARR-02. This tests the barrier.

### H4: Timestamp Regression Rejection

**If** a block with `timestamp < prev_timestamp` is submitted to `check_monotone_ledger()`, **then** a `MonotoneViolation` with `violation_type="timestamp"` must be produced, **while** `tier_skew` and `semantic_drift` signals must remain unaffected.

*Rationale*: FORB-06 (timestamp monotonicity) is enforced by BARR-02.

### H5: TDA Envelope Sensitivity

**If** synthetic cycle data is injected with `SNS > 0.6` (anomaly threshold), **then** a `TDA_SNS_ANOMALY` red-flag must be logged with `action="LOGGED_ONLY"`, **while** the runner control flow must remain unmodified (SHADOW mode non-interference).

*Rationale*: Red-flags are observational only (FORB-03). The signal must fire; enforcement must not occur.

### H6: Tier Skew Detection

**If** verifier telemetry is injected with `timeout_rate(FAST) < timeout_rate(BALANCED)` across sufficient samples, **then** `tier_skew_detector` must produce an alarm with `p_value < alpha`, **while** `determinism_rate` and `monotone_guard` must remain unaffected.

*Rationale*: `tier_skew` operates on verifier-layer telemetry, independent of attestation and block clocks.

---

## 3. Controlled Variables (Held Constant)

| Variable | Held Value | Rationale |
|----------|------------|-----------|
| **Toolchain** | Lean 4.x as currently deployed; Python 3.11+ with `uv sync` dependencies | Avoids confounding from toolchain drift |
| **Input corpus** | Static fixture set in `tests/fixtures/` | Deterministic inputs enable reproducibility |
| **SHADOW mode** | `shadow_mode=True` for all P3/P4 components | Required by P3/P4 design; enables FORB-02/FORB-08 testing |
| **Configuration flags** | Default values from `config/` as of experiment start | Avoids confounding from configuration drift |
| **Clock domains** | All 5 clocks (USLA Cycle, Block, Attestation, RFL Epoch, Governance) operate independently | Tests signal isolation across clock domains |
| **Barrier logic** | `check_monotone_ledger()`, `compute_composite_root()`, `RFLEventGate` unchanged | Barriers are the invariants under test |
| **PRNG seed** | Fixed seed for any stochastic components | Enables deterministic replay |

---

## 4. Perturbations (What You Will Change)

### P1: Proof Ordering Perturbation

**Description**: Shuffle the order of proof artifacts within a single run while preserving byte-identical content.

**Implementation**: Re-order `proof_parents` edges in memory before computing Merkle root. Content hashes remain identical; ordering changes.

**Reversibility**: Sorting by original index restores original order.

**Invariant Safety**: Does not violate BARR-02 (block ordering) as perturbation is pre-sealing.

### P2: Content Modification Perturbation

**Description**: Inject a single-byte modification into one proof artifact during replay comparison.

**Implementation**: Flip one byte in artifact body after initial hash computation, then re-run replay comparison.

**Reversibility**: Restore original byte.

**Invariant Safety**: Does not affect block structure; tests attestation-layer sensitivity only.

### P3: Height Regression Injection

**Description**: Construct a block sequence where `blocks[n].height <= blocks[n-1].height`.

**Implementation**: Override `height` field in synthetic block fixture.

**Reversibility**: Restore monotone sequence.

**Invariant Safety**: Tests FORB-04 enforcement; must be blocked by `check_monotone_ledger()`.

### P4: Timestamp Regression Injection

**Description**: Construct a block sequence where `blocks[n].timestamp < blocks[n-1].timestamp`.

**Implementation**: Override `timestamp` field in synthetic block fixture.

**Reversibility**: Restore monotone sequence.

**Invariant Safety**: Tests FORB-06 enforcement; must be blocked by `check_monotone_ledger()`.

### P5: TDA Threshold Breach Injection

**Description**: Inject synthetic cycle observations with `SNS=0.7` (above 0.6 anomaly threshold).

**Implementation**: Call `TDAMonitor.observe_cycle()` with elevated SNS value.

**Reversibility**: Reset monitor state.

**Invariant Safety**: Does not affect real execution; tests red-flag observation layer.

### P6: Tier Skew Injection

**Description**: Inject verifier telemetry where FAST tier has lower timeout rate than BALANCED tier.

**Implementation**: Call `TierSkewDetector.update()` with synthetic `LeanVerificationTelemetry` objects.

**Reversibility**: Reset detector state.

**Invariant Safety**: Does not affect real verification; tests tier skew detection.

---

## 5. Expected Signal Behavior

| Signal | Perturbation | Expected Direction | Reason | Failure Interpretation |
|--------|-------------|-------------------|--------|------------------------|
| `determinism_rate` | P1 (ordering) | **No change** (1.0) | Content unchanged; ordering is not content | Signal moved = instrument mis-wiring |
| `determinism_rate` | P2 (content) | **Drops** (< 1.0) | Content divergence detected | Signal unchanged = instrument failure |
| `monotone_guard.height` | P3 (height) | **Violation produced** | FORB-04 enforced | No violation = barrier failure |
| `monotone_guard.timestamp` | P4 (timestamp) | **Violation produced** | FORB-06 enforced | No violation = barrier failure |
| `tier_skew` | P6 (skew) | **Alarm produced** | Invariant violated | No alarm = detector failure |
| `TDA_SNS_ANOMALY` | P5 (threshold) | **Red-flag logged** | Threshold exceeded | No red-flag = observer failure |
| `runner_control_flow` | P5 (threshold) | **No change** | SHADOW mode; FORB-03 | Control flow modified = SHADOW violation |
| `monotone_guard` | P2 (content) | **No change** | Block structure unaffected | Violation = cross-layer leakage |
| `semantic_drift` | P1 (ordering) | **No change** | Operates on corpus, not ordering | Signal moved = scope misunderstanding |

---

## 6. Pass / Fail Criteria (Strict)

### PASS Conditions

The experiment PASSES if and only if:

1. **All hypotheses hold**: H1-H6 exhibit expected behavior as documented in Section 5.
2. **No invariant violations from non-violating perturbations**: Perturbations P1, P2, P5, P6 do not trigger monotone guard violations.
3. **Invariant violations from violating perturbations**: Perturbations P3, P4 produce the documented violations.
4. **SHADOW mode non-interference**: P5 logs red-flag but does not modify control flow.
5. **Determinism on re-run**: Running CAL-EXP-1 twice with identical inputs produces identical evidence packs (excluding timestamps).

### FAIL Conditions

The experiment FAILS if any of the following occur:

1. **Signal does not respond when expected**: e.g., `determinism_rate` remains 1.0 under P2.
2. **Signal responds when it should not**: e.g., `monotone_guard` produces violation under P2.
3. **Invariant violation not produced**: e.g., P3 does not produce height violation.
4. **SHADOW violation**: e.g., red-flag under P5 triggers abort or governance modification.
5. **Non-determinism**: Two runs with identical inputs produce different evidence packs.

### EXPERIMENT INVALIDATION

CAL-EXP-1 results must be discarded if:

1. Any fixture file is missing or corrupted.
2. Toolchain version differs from documented version.
3. SHADOW mode is not enabled (`shadow_mode=False` detected).
4. PRNG seed is not fixed.

---

## 7. Invalid Experiment Conditions

The following conditions invalidate CAL-EXP-1 results entirely:

| Condition | Detection Method | Consequence |
|-----------|------------------|-------------|
| **Missing fixtures** | Pre-run check for `tests/fixtures/` contents | Abort experiment |
| **Toolchain mismatch** | Compare `uv sync` hash against recorded baseline | Discard results |
| **SHADOW mode disabled** | Check `config.shadow_mode` at runner initialization | Abort experiment |
| **Broken replay infrastructure** | `verify_first_light_determinism.py` fails on known-good inputs | Discard results |
| **Non-deterministic infrastructure** | Two baseline runs diverge on hash values | Discard results |
| **Clock synchronization failure** | Any clock domain fails to advance monotonically | Discard results |
| **Environment variable contamination** | Unexpected `DERIVE_*` or `USLA_*` overrides | Discard results |

---

## 8. Evidence Artifacts Produced

CAL-EXP-1 produces the following artifacts. No charts, summaries, or interpretive narratives are generated.

### Per-Hypothesis Artifacts

| Artifact | Format | Contents |
|----------|--------|----------|
| `h1_ordering_perturbation.jsonl` | JSONL | Raw signal values before/after P1 |
| `h2_content_perturbation.jsonl` | JSONL | Raw `determinism_rate` and monotone guard outputs under P2 |
| `h3_height_rejection.jsonl` | JSONL | `MonotoneViolation` output under P3 |
| `h4_timestamp_rejection.jsonl` | JSONL | `MonotoneViolation` output under P4 |
| `h5_tda_redflag.jsonl` | JSONL | TDARedFlag output and control flow trace under P5 |
| `h6_tier_skew.jsonl` | JSONL | TierSkewDetector alarm output under P6 |

### Aggregate Artifacts

| Artifact | Format | Contents |
|----------|--------|----------|
| `cal_exp_1_manifest.json` | JSON | Run metadata: toolchain hash, fixture hash, PRNG seed, timestamps |
| `cal_exp_1_hypothesis_matrix.json` | JSON | Boolean pass/fail for each hypothesis (no interpretation) |
| `cal_exp_1_evidence_pack.tar.gz` | Compressed archive | All JSONL files plus manifest |
| `cal_exp_1_evidence_pack.sha256` | Text | SHA-256 of evidence pack for integrity verification |

### Hash Commitments

| Artifact | Purpose |
|----------|---------|
| `cal_exp_1_manifest.json:evidence_merkle_root` | Merkle root over all JSONL artifact hashes |
| `cal_exp_1_manifest.json:fixture_hash` | SHA-256 of `tests/fixtures/` directory |
| `cal_exp_1_manifest.json:toolchain_hash` | SHA-256 of `uv.lock` |

---

## 9. Explicit Non-Claims Block (Mandatory)

### What CAL-EXP-1 Does NOT Prove

1. **CAL-EXP-1 does NOT prove MathLedger is correct.** It tests whether signals respond to perturbations. Signal correctness is a necessary but not sufficient condition for system correctness.

2. **CAL-EXP-1 does NOT prove MathLedger is safe.** Safety requires demonstrated non-interference in operational contexts. CAL-EXP-1 tests SHADOW mode compliance in controlled settings.

3. **CAL-EXP-1 does NOT prove MathLedger detects all failures.** It tests documented failure classes only. Undocumented failure modes are outside scope.

4. **CAL-EXP-1 does NOT evaluate model quality.** MathLedger does not claim responsibility for model quality (per MATHLEDGER_RESPONSIBILITY_BOUNDARY.md). CAL-EXP-1 tests the ledger layer, not the model layer.

5. **CAL-EXP-1 does NOT provide performance benchmarks.** Timing data is captured for reproducibility, not for optimization.

6. **CAL-EXP-1 does NOT authorize any pilot or deployment.** CAL-EXP-1 is pre-pilot. Successful completion is a prerequisite for, not a grant of, pilot authorization.

### What Conclusions Must NOT Be Drawn

- "CAL-EXP-1 passed, therefore MathLedger is ready for production." **INVALID.**
- "CAL-EXP-1 passed, therefore the system is safe." **INVALID.**
- "CAL-EXP-1 passed, therefore signals are accurate." **INVALID** (signals are responsive, not necessarily accurate).
- "CAL-EXP-1 failed, therefore the system is unsafe." **INVALID** (failure indicates instrument mis-calibration, not system risk).

### Why CAL-EXP-1 Precedes Pilot

CAL-EXP-1 establishes instrument baseline. Without confirmed instrument behavior, pilot observations cannot be interpreted. A pilot that detects anomalies cannot distinguish between:

1. Real system failure (interesting)
2. Instrument mis-calibration (confounding)
3. Experiment design error (invalidating)

CAL-EXP-1 eliminates category (2) and provides evidence against category (3) by documenting instrument behavior under controlled perturbation.

---

## Appendix: Determinism Exclusions

### Time-Variant Keys

The following JSON keys are excluded from normalized Merkle root computation. These keys represent execution-time metadata that varies between runs but do not affect the semantic correctness of experiment results.

| Key | Description | Rationale |
|-----|-------------|-----------|
| `timestamp` | Execution timestamp (ISO 8601) | Varies with wall-clock time |
| `created_at` | Creation timestamp | Varies with wall-clock time |
| `updated_at` | Update timestamp | Varies with wall-clock time |
| `run_timestamp` | Run-specific timestamp | Varies with wall-clock time |

### Normalization Rule

Before computing `evidence_merkle_root_normalized`:

1. Parse each artifact as JSON
2. Recursively strip all keys in the TIME_VARIANT_KEYS set
3. Serialize with `json.dumps(obj, sort_keys=True)`
4. Compute SHA-256 hash

This ensures that two runs with identical inputs produce identical normalized Merkle roots, even if wall-clock timestamps differ.

### Merkle Root Types

| Root Type | Includes Timestamps | Use Case |
|-----------|---------------------|----------|
| `evidence_merkle_root_raw` | Yes | Exact byte-level audit trail |
| `evidence_merkle_root_normalized` | No | Reproducibility verification |

For replication verification, compare `evidence_merkle_root_normalized` values. Raw roots may differ due to timestamp variance.

---

## Appendix A: Traceability to Verified Claims

| Hypothesis | Verified Claim ID | Test Evidence |
|------------|-------------------|---------------|
| H1 | BARR-01 (dual attestation) | `test_compute_composite_root_valid` |
| H2 | CLK-05, BARR-02 | `determinism_rate` signal, `test_seal_block_with_dual_roots_basic` |
| H3 | FORB-04 | `test_height_violation` |
| H4 | FORB-06 | `test_timestamp_violation` |
| H5 | FORB-03 | `TDARedFlag.action="LOGGED_ONLY"` |
| H6 | CLK-03, TDA window clock | `tier_skew_detector` alarm production |

---

## Appendix B: Perturbation-Signal Isolation Matrix

This matrix documents which signals should remain stable under each perturbation.

| Perturbation | Sensitive Signals | Must-Be-Invariant Signals |
|--------------|-------------------|---------------------------|
| P1 (ordering) | *None* | `determinism_rate`, `semantic_drift`, `tier_skew`, `monotone_guard` |
| P2 (content) | `determinism_rate` | `monotone_guard`, `tier_skew`, `TDA_*` |
| P3 (height) | `monotone_guard.height` | `determinism_rate`, `tier_skew`, `semantic_drift` |
| P4 (timestamp) | `monotone_guard.timestamp` | `determinism_rate`, `tier_skew`, `semantic_drift` |
| P5 (TDA threshold) | `TDA_SNS_ANOMALY` | `runner_control_flow`, `monotone_guard`, `determinism_rate` |
| P6 (tier skew) | `tier_skew` | `determinism_rate`, `monotone_guard`, `semantic_drift` |

---

*Document Version: 1.0.0*
*Generated: 2025-12-13*
*Status: EXPERIMENT DESIGN (Implementation requires explicit authorization)*
