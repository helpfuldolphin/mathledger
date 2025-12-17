# First Light External Verifier FAQ

**Document Version:** 1.0.0
**Audience:** External auditors, safety engineers, skeptical reviewers

This FAQ addresses common questions from external verifiers examining the First Light evidence pack.

---

## General Questions

### What is First Light?

First Light is the initial shadow-mode validation run of the MathLedger system's uplift monitoring infrastructure. It generates synthetic data (P3) and compares a shadow twin's predictions against a mock real runner (P4) to validate the monitoring and divergence detection systems before connecting to real telemetry.

### What does "Phase X" mean?

Phase X refers to the pre-production validation phase. All First Light runs are Phase X artifacts — they use synthetic or mock data and operate in SHADOW MODE. Phase X is explicitly distinct from production operation.

### What is the purpose of the evidence pack?

The evidence pack provides a complete, self-contained bundle of artifacts that allows independent verification of:
- System behavior under controlled conditions
- Determinism of outputs given identical inputs
- SHADOW MODE compliance (no governance mutations)
- Integrity of all artifacts via SHA-256 hashes

---

## Metric-Specific Questions

### Why is P3 omega_occupancy (85.1%) below the 90% threshold?

**Short answer:** The synthetic twin spends more time outside the designated "safe region" (Omega) than the threshold allows.

**Technical detail:** The omega region is defined by stability criteria (RSI >= threshold, no CDI activations, etc.). When the synthetic runner's metrics fall outside these bounds, it counts as an omega exit. An occupancy rate of 85.1% means the system was in the safe region 85.1% of cycles.

**Implication:** Under live governance, this would trigger a hypothetical abort condition. However, because First Light operates in SHADOW MODE:
- No actual abort occurred
- The condition is logged but not enforced
- The `hypothetical_abort: true` flag in `stability_report.json` records this

**What it means for safety:** This indicates the synthetic model has stability characteristics that would not pass production criteria. This is expected behavior for a first validation run — the threshold or model parameters may need tuning before real deployment.

### Why is the P4 divergence rate ≈97% in the current evidence pack?

**Short answer:** The current P4 run uses `MockTelemetryProvider` and an uncalibrated Twin. The 97% divergence rate is the **noise baseline** — this is how the alarm system behaves when connected to a noisy, non-physical source.

**Technical framing:** The Twin correctly refuses to hallucinate alignment with mock noise. When fed synthetic observations from an independent random process, the Twin's predictions diverge from the mock "real" data on nearly every cycle. This is correct behavior:

- The Twin is not designed to predict mock data
- The mock provider generates observations uncorrelated with the Twin's internal state
- High divergence demonstrates the alarm system is working — it detects mismatch

**This validates:**
1. Divergence detection is functioning correctly
2. The system correctly identifies when Twin ≠ Real
3. All divergences are properly logged with `action="LOGGED_ONLY"`
4. The Twin does not fabricate false agreement

**Phase X P5 will re-measure divergence:** When the P4 harness is connected to actual runner telemetry and a calibrated Twin, divergence will be re-measured. The expectation is that divergence will drop significantly because the Twin is designed to model real runner behavior. If divergence remains high with real data, this indicates a calibration problem requiring investigation.

### What do the divergence severity levels mean?

| Severity | Definition |
|----------|------------|
| **minor** | Small numeric differences within acceptable tolerance (e.g., success probability differs by < 0.05) |
| **moderate** | Larger differences or state disagreements that warrant logging (e.g., twin predicts omega=True, real shows omega=False) |
| **severe** | Critical disagreements that would trigger governance action in ACTIVE mode (none observed in this run) |

The current run shows 710 minor, 261 moderate, and 0 severe divergences.

### What does "noise baseline" mean for the current evidence pack?

The 97% divergence rate is the **noise baseline** — it establishes how the alarm system behaves when connected to a non-physical, uncorrelated signal source. This baseline serves several purposes:

1. **Stress test:** Demonstrates the system handles high-divergence scenarios without errors or crashes.

2. **Alarm validation:** Confirms the Twin correctly flags mismatch rather than fabricating false agreement.

3. **Baseline for comparison:** P5 (real telemetry) results will be compared against this baseline. A significant drop in divergence indicates the Twin is successfully modeling real behavior.

**The noise baseline is NOT:**
- Evidence of a bug (mismatch with mock data is expected)
- A safety concern (SHADOW MODE is designed so no action is taken)
- A measure of Twin accuracy (mock data is not ground truth)

**Reviewing the baseline:**
- ~97% divergence with mock data = alarm system working correctly
- If P5 shows ~97% divergence with real data = Twin calibration problem
- If P5 shows <30% divergence with real data = Twin tracking real behavior

### What are the P5 divergence expectations?

Phase X P5 will rerun P4 with real telemetry from an actual runner and a calibrated twin. The following target bands are **tentative and subject to calibration**:

| Divergence Rate | Interpretation |
|-----------------|----------------|
| < 0.30 (30%) | **Acceptable** — twin tracks real runner reasonably well |
| 0.30 – 0.50 | **Marginal** — twin may need recalibration |
| 0.50 – 0.70 | **Concerning** — investigate twin model assumptions |
| > 0.70 (70%) | **Unstable** — twin is not tracking real behavior; requires intervention |

These bands will be refined based on:
- Initial P5 golden run results
- Statistical analysis of divergence patterns
- Comparison against mock baseline (current 97.2%)

A successful P5 run should show divergence rate **significantly below the mock baseline** (ideally < 0.30). If P5 divergence remains above 0.70, this indicates the twin model needs fundamental recalibration before proceeding.

---

## SHADOW MODE Questions

### What are the design goals of SHADOW MODE?

SHADOW MODE is designed with these properties:

1. **No governance writes:** The harnesses do not call `governance.enforce()`, `governance.abort()`, `governance.reject()`, or similar mutation APIs.

2. **No state mutation:** The P4 runner does not call `set_state()`, `modify_governance()`, `abort()`, or `stop()` on the telemetry provider.

3. **Observation-only logging:** All divergence records have `action="LOGGED_ONLY"` — the system observes and records but takes no action.

4. **Mode marking:** All cycle observations and summary records have `mode="SHADOW"` to clearly distinguish them from any future ACTIVE mode data.

### What is SHADOW MODE NOT designed to do?

SHADOW MODE is not designed for the following:

1. **Accuracy of the twin model:** The twin may produce poor predictions; SHADOW MODE is only designed to not act on them.

2. **Real-world applicability:** Synthetic data may not represent production conditions.

3. **Threshold correctness:** The omega occupancy threshold (90%) may be too strict or too lenient for real deployment.

4. **Future behavior:** SHADOW MODE is a contract for this specific run; future runs must be independently verified.

### How is SHADOW MODE enforced?

SHADOW MODE is enforced at multiple levels:

1. **Configuration validation:** `FirstLightConfig.validate()` and `FirstLightConfigP4.validate()` both reject `shadow_mode=False`.

2. **Runtime enforcement:** `FirstLightShadowRunnerP4` raises `ValueError` if instantiated with non-shadow config.

3. **Data structure defaults:** `RealCycleObservation`, `TwinCycleObservation`, and `DivergenceSnapshot` all default to shadow mode values.

4. **Static analysis:** The compliance tests scan harness source code for forbidden governance API patterns.

5. **Output verification:** All JSONL records are verified to contain correct mode/action markers.

### How do I verify SHADOW MODE compliance myself?

Run the compliance test suite:

```bash
uv run python -m pytest tests/integration/test_shadow_mode_compliance.py -v
```

Expected: 14/14 tests pass. Any failure indicates a potential SHADOW MODE violation.

---

## Integrity Questions

### How can I verify the evidence pack has not been altered?

The evidence pack includes cryptographic integrity verification:

1. **Per-file hashes:** `manifest.json` contains SHA-256 hashes for all 14 files. Run the integrity verifier to confirm:

   ```bash
   uv run python scripts/verify_evidence_pack_integrity.py
   ```

2. **Manifest hash:** The manifest itself can be hashed and compared against an external record:

   ```bash
   uv run python -c "
   import hashlib
   with open('results/first_light/evidence_pack_first_light/manifest.json', 'rb') as f:
       print(hashlib.sha256(f.read()).hexdigest())
   "
   ```

3. **Determinism verification:** Run the harnesses yourself with the same seed/config and compare outputs:

   ```bash
   uv run python scripts/verify_first_light_determinism.py
   ```

If someone edited the evidence pack:
- File hashes would not match manifest
- Determinism check would fail
- Manifest hash would differ from any external record

### Can I reproduce the evidence pack from scratch?

Yes. Run these commands:

### How do I verify schema correctness?

- **Fast path:** Run the status generator with schema validation enabled (non-gating, advisory only):
  ```bash
  uv run python scripts/generate_first_light_status.py \
    --p3-dir results/first_light/golden_run/p3 \
    --p4-dir results/first_light/golden_run/p4 \
    --evidence-pack-dir results/first_light/evidence_pack_first_light \
    --validate-schemas
  ```
  The resulting `first_light_status.json` will include `schemas_ok` (True/False/None) and emit warnings for any failures. Phase X treats this as informational in SHADOW MODE; no gating is performed.
- **Direct check:** Validate individual artifacts against their schemas:
  ```bash
  uv run python tools/evidence_schema_check.py results/first_light/evidence_pack_first_light/p3_synthetic/synthetic_raw.jsonl schemas/evidence/first_light_synthetic_raw.schema.json
  uv run python tools/evidence_schema_check.py results/first_light/evidence_pack_first_light/p3_synthetic/red_flag_matrix.json schemas/evidence/first_light_red_flag_matrix.schema.json
  uv run python tools/evidence_schema_check.py results/first_light/evidence_pack_first_light/p4_shadow/divergence_log.jsonl schemas/evidence/p4_divergence_log.schema.json
  ```
  Any failures should be investigated but do not halt Phase X workflows until promoted to a gate.

```bash
# P3 Harness
uv run python scripts/usla_first_light_harness.py \
    --cycles 1000 --seed 42 --slice arithmetic_simple \
    --runner-type u2 --tau-0 0.20 --window-size 50 \
    --output-dir results/my_verification/p3

# P4 Harness
uv run python scripts/usla_first_light_p4_harness.py \
    --cycles 1000 --seed 42 --slice arithmetic_simple \
    --runner-type u2 --tau-0 0.20 \
    --output-dir results/my_verification/p4

# Build Evidence Pack
uv run python scripts/build_first_light_evidence_pack.py \
    --p3-dir results/my_verification/p3 \
    --p4-dir results/my_verification/p4 \
    --output-dir results/my_verification/evidence_pack
```

Your `stability_report.json` and `p4_summary.json` should contain identical metric values (ignoring timestamps).

### Why do JSONL files differ between runs if they're supposed to be deterministic?

JSONL files contain per-record `timestamp` fields that vary between runs. The determinism check compares content *after stripping timestamp fields*. All mathematical and logical outputs are identical; only ephemeral metadata differs.

---

## Narrative Consistency Index (NCI) Questions

### What does NCI tell me once the runner uses real telemetry?

When the runner transitions to real telemetry (Phase X P5+), the NCI provides three distinct categories of insight:

**1. Documentation-Telemetry Alignment**

The NCI's Telemetry Consistency Laws (TCL) validate that your documentation accurately describes the actual telemetry being produced:

- **TCL-002 violations** indicate documentation uses non-canonical field names (e.g., "Ht" instead of "H", "RSI" instead of "rho"). Under real telemetry, these become operational concerns — operators may misinterpret live data if docs use different terminology.

- **TCL-004 drift detection** becomes operationally significant. When real telemetry schema evolves (new fields, deprecated events), NCI flags documentation that has not been updated. Under mock/synthetic telemetry, this is a specification exercise; under real telemetry, stale docs can cause operational confusion.

**2. Slice Configuration Accuracy**

The Slice Identity Consistency Laws (SIC) validate that slice documentation matches deployed configurations:

- Under real telemetry, slice parameters (depth limits, atom bounds, theory IDs) are validated against what the runner actually uses
- Documentation claiming capabilities beyond the deployed slice configuration will be flagged

**3. Operational Health Signal**

The NCI contributes to the Global Governance Fusion Layer (GGFL) through the Narrative Signal (SIG-NAR):

| NCI Condition | GGFL Impact |
|---------------|-------------|
| `global_nci < 0.60` | WARNING recommendation |
| `slo_status == BREACH` | WARNING (may contribute to escalation) |
| `telemetry_drift_detected == true` | WARNING + operational flag |

Under real telemetry, a low NCI score suggests documentation debt that may affect operator review of live system behavior. This is not a safety concern per se (the runner operates correctly regardless of documentation quality), but it affects operational readiness and auditability.

**What NCI does NOT tell you under real telemetry:**

- NCI does not validate telemetry correctness — it validates documentation consistency
- NCI does not detect runner bugs — it detects doc/spec divergence
- NCI does not replace functional testing — it complements it with documentation health

**Recommended action when NCI flags issues under real telemetry:**

1. Review flagged documents for accuracy against live telemetry schema
2. Prioritize TCL-002 violations (terminology consistency)
3. Establish a documentation SLO (e.g., "update docs within 24h of schema changes")
4. Include NCI check in release readiness criteria

---

## Future Changes Questions

### What changes when real telemetry is wired in (Phase X P5)?

| Aspect | Current (Mock) | Future (Real) |
|--------|----------------|---------------|
| P4 telemetry source | `MockTelemetryProvider` | Real runner adapter |
| Expected divergence rate | ~97% (uncorrelated) | Should be much lower (twin tracks real) |
| Twin accuracy | Baseline only | Meaningful correlation metric |
| Stability metrics | Synthetic | Real system behavior |

The harness invocation, evidence pack structure, and verification procedures remain the same. Only the data source changes.

### Will the thresholds change?

Possibly. The omega occupancy threshold (90%) and other criteria may be adjusted based on:
- Real telemetry behavior patterns
- Operational requirements
- Safety analysis results

Any threshold changes will be documented in updated configuration files and will require new golden runs.

### When does SHADOW MODE end?

SHADOW MODE remains in effect for all Phase X work. Transition to ACTIVE MODE requires:

1. Real telemetry integration and validation
2. Twin calibration against real data
3. Threshold tuning based on real behavior
4. Governance review and approval
5. Explicit configuration change (`shadow_mode=False`)

This transition is not imminent and will be documented extensively before occurring.

---

## Troubleshooting Questions

### The integrity verifier reports MISMATCH. What should I do?

1. **Check if the file was modified:** Compare the actual file content to the expected hash.

2. **Check for encoding issues:** Windows/Unix line ending differences can cause hash mismatches.

3. **Re-run the harness:** Generate fresh artifacts and compare.

4. **Check manifest path format:** The verifier normalizes backslashes; validate your paths are correct.

If mismatches persist after fresh generation, file an issue with full output.

### The determinism check reports DIFFERS. What should I do?

1. **Check the specific field:** Small floating-point differences (< 1e-9) may be acceptable numerical noise.

2. **Check for state pollution:** Leftover files from previous runs can affect results. Clean `results/determinism_test/` and retry.

3. **Check environment:** Different Python versions or platforms may have subtle floating-point differences.

4. **Check seed:** Validate that both runs used `--seed 42`.

If differences are in meaningful fields (success rates, counts), investigate the root cause before trusting the artifacts.

### SHADOW MODE compliance tests fail. What should I do?

Any compliance test failure is a serious issue. Do not trust artifacts from a run with compliance failures.

1. **Read the failure message:** It will indicate which specific check failed.

2. **Check recent code changes:** Was governance code accidentally introduced?

3. **Check configuration:** Was `shadow_mode=False` somehow set?

4. **File an issue:** SHADOW MODE violations must be investigated and resolved before any artifacts can be used.

---

## Contact and Support

For questions not covered here:
- File an issue at the repository with `[first-light-verification]` label
- Include full command output and any error messages
- Specify your OS, Python version, and uv version

---

**Document generated:** 2025-12-11
**Applies to:** Phase X First Light Golden Run (mock telemetry)
