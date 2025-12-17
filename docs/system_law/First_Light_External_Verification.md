# First Light External Verification Guide

**Document Version:** 1.1.0
**Target Audience:** Senior infrastructure or safety engineer performing independent verification
**Validation Platform:** Windows 11, Python 3.11.9, uv package manager

---

## 1. Prerequisites

### 1.1 System Requirements

| Requirement | Validated Version | Notes |
|-------------|-------------------|-------|
| Python | 3.11.9 | 3.10+ should work; 3.11 recommended |
| OS | Windows 11 | Linux/macOS untested but likely compatible |
| Package Manager | uv 0.4+ | Or pip with requirements.txt |
| Disk Space | ~500MB | For dependencies + artifact generation |

### 1.2 Repository Setup

```bash
# Clone the repository
git clone <repo-url> mathledger
cd mathledger

# Install dependencies using uv (recommended)
uv sync

# Alternative: pip installation
pip install -e .
```

Verify installation:

```bash
uv run python -c "from backend.topology.first_light.runner import FirstLightShadowRunner; print('OK')"
```

Expected output: `OK`

### 1.3 Artifact Locations

| Artifact Type | Default Path |
|---------------|--------------|
| Canonical Config | `docs/system_law/first_light_golden_run_config.json` |
| P3 Golden Run | `results/first_light/golden_run/p3/fl_<timestamp>_seed42/` |
| P4 Golden Run | `results/first_light/golden_run/p4/p4_<timestamp>/` |
| Evidence Pack | `results/first_light/evidence_pack_first_light/` |
| Compliance Tests | `tests/integration/test_shadow_mode_compliance.py` |
| Determinism Script | `scripts/verify_first_light_determinism.py` |
| Integrity Verifier | `scripts/verify_evidence_pack_integrity.py` |
| Status Generator | `scripts/generate_first_light_status.py` |

---

## 2. Step-by-Step Local Reproduction

### 2.0 Containerized Harness Checks (Clean Environment)

Recommended one-liners for external auditors (clean/air-gapped repro):

```bash
docker compose -f docker-compose.p3.yml run --rm p3-harness
docker compose -f docker-compose.p3.yml run --rm p4-harness
```

These build the slim Python images, install project deps, and execute the harness smoke tests in isolated containers. Docker paths are optional but strongly recommended for air-gapped repro.

**Outputs:** P3 smoke is an in-container pytest. By default its artifacts live under `/tmp/pytest-of-root` in the container and are ephemeral. P4 smoke writes under `results/p4_compose/p4_<timestamp>` inside the container unless you mount a host volume.

For a JSON summary of both runs *and* a persistent copy of P3 pytest artifacts on the host, run:

```bash
python scripts/run_first_light_docker_smoke.py --p3-bind-mount <HOST_DIR>
```

This bind-mounts `<HOST_DIR>` into the P3 container and copies `/tmp/pytest-of-root` to that host directory after the pytest run. The JSON summary includes `p3_host_dir` when this flag is used.

### 2.1 Run P3 Harness (Synthetic First Light)

**Command:**

```bash
uv run python scripts/usla_first_light_harness.py \
    --cycles 1000 \
    --seed 42 \
    --slice arithmetic_simple \
    --runner-type u2 \
    --tau-0 0.20 \
    --window-size 50 \
    --output-dir results/verification/p3
```

> **Note:** New callers MUST use `scripts/usla_first_light_harness.py`; `scripts/first_light_p3_harness.py` is retained only as a legacy wrapper for older automation.
> If the wrapper is triggered (for compatibility runs), verify the console logs include the banner `LEGACY ONLY — DO NOT USE FOR NEW RUNS` and confirm it still mirrors `first_light_synthetic_raw.jsonl` alongside the canonical artifacts.

**Expected Artifacts:**

| File | Approximate Size | Description |
|------|------------------|-------------|
| `synthetic_raw.jsonl` | 330-340 KB | Raw cycle observations |
| `stability_report.json` | 2-3 KB | Aggregated metrics |
| `red_flag_matrix.json` | 700-800 bytes | Red flag occurrences |
| `metrics_windows.json` | 10-12 KB | Windowed metrics |
| `tda_metrics.json` | 4-5 KB | TDA analysis |
| `run_config.json` | 500-600 bytes | Frozen config |

> **Test-only note:** If the P3 harness is invoked with `--pathology`, the evidence pack manifest includes a SHADOW-only annotation under `evidence.data.p3_pathology` describing the injected stress; external verifiers should treat this as non-production metadata.

**Verification (inspect stability_report.json):**

```bash
# Using Python (cross-platform)
uv run python -c "
import json
with open('results/verification/p3/fl_*/stability_report.json'.replace('*', '<your-run-id>')) as f:
    r = json.load(f)
print(f\"success_rate: {r['metrics']['success_rate']}\")
print(f\"omega_occupancy: {r['metrics']['omega']['occupancy_rate']}\")
print(f\"schema_version: {r['schema_version']}\")
"
```

**Expected values (approximate):**
- `success_rate`: 0.85 +/- 0.02
- `omega_occupancy`: 0.85 +/- 0.02
- `schema_version`: "1.0.0"

If `success_rate` is below 0.75 or above 0.95, investigate. The synthetic runner with seed=42 should produce consistent results.

### 2.2 Run P4 Harness (Shadow Coupling)

**Command:**

```bash
uv run python scripts/usla_first_light_p4_harness.py \
    --cycles 1000 \
    --seed 42 \
    --slice arithmetic_simple \
    --runner-type u2 \
    --tau-0 0.20 \
    --output-dir results/verification/p4
```

**Expected Artifacts:**

| File | Approximate Size | Description |
|------|------------------|-------------|
| `real_cycles.jsonl` | 410-430 KB | Real runner observations |
| `twin_predictions.jsonl` | 300-320 KB | Shadow twin predictions |
| `divergence_log.jsonl` | 620-640 KB | All divergence events |
| `p4_summary.json` | 2-3 KB | Aggregated P4 metrics |
| `twin_accuracy.json` | 600-700 bytes | Twin accuracy breakdown |
| `run_config.json` | 350-400 bytes | Frozen config |

**Verification (inspect p4_summary.json):**

```bash
uv run python -c "
import json
from pathlib import Path
p4_dir = list(Path('results/verification/p4').glob('p4_*'))[0]
with open(p4_dir / 'p4_summary.json') as f:
    s = json.load(f)
print(f\"mode: {s['mode']}\")
print(f\"success_rate: {s['uplift_metrics']['u2_success_rate_final']}\")
print(f\"divergence_rate: {s['divergence_analysis']['divergence_rate']}\")
print(f\"twin_success_accuracy: {s['twin_accuracy']['success_prediction_accuracy']}\")
"
```

**Expected values:**
- `mode`: "SHADOW" (required — if not "SHADOW", this is a compliance failure)
- `success_rate`: 0.92 +/- 0.03
- `divergence_rate`: 0.95+ (high divergence expected with mock telemetry)
- `twin_success_accuracy`: 0.88 +/- 0.03

**Critical check — verify all divergence records have action="LOGGED_ONLY":**

```bash
uv run python -c "
import json
from pathlib import Path
p4_dir = list(Path('results/verification/p4').glob('p4_*'))[0]
with open(p4_dir / 'divergence_log.jsonl') as f:
    violations = [i for i, line in enumerate(f) if json.loads(line).get('action') != 'LOGGED_ONLY']
if violations:
    print(f'FAIL: {len(violations)} records without action=LOGGED_ONLY')
else:
    print('PASS: All divergence records have action=LOGGED_ONLY')
"
```

Expected output: `PASS: All divergence records have action=LOGGED_ONLY`

### 2.3 Build Evidence Pack

**Command:**

```bash
uv run python scripts/build_first_light_evidence_pack.py \
    --p3-dir results/verification/p3 \
    --p4-dir results/verification/p4 \
    --output-dir results/verification/evidence_pack
```

This command now emits Δp / RSI / Ω SVGs under `visualizations/` alongside the manifest entries. Treat them as standard verification checkpoints for every First Light evidence pack.

#### Plot Artifacts (Δp / RSI / Ω)

| File | Interpretation | Primary Source |
|------|----------------|----------------|
| `visualizations/delta_p_vs_cycles.svg` | Cycle-by-cycle Δp between twins and real trajectories; large spikes signal divergence pressure. | `p4_shadow/divergence_log.jsonl` |
| `visualizations/rsi_vs_cycles.svg` | Rolling RSI trend for the synthetic run; visually confirm mean RSI claims. | `p3_synthetic/metrics_windows.json` |
| `visualizations/omega_occupancy_vs_cycles.svg` | Omega occupancy over windows; correlates with stability statements in the narrative. | `p3_synthetic/metrics_windows.json` |

Open each SVG to confirm the qualitative story (e.g., Δp contained, RSI trending upward, Ω occupancy >80%) before filing the verification report.

### 2.4 Schema Checker (Non-gating CI Sketch)

- **Proposed CI step (non-gating):** Run the status generator with schema validation enabled to record `schemas_ok` without blocking merges.
  ```bash
  uv run python scripts/generate_first_light_status.py \
    --p3-dir results/first_light/golden_run/p3 \
    --p4-dir results/first_light/golden_run/p4 \
    --evidence-pack-dir results/first_light/evidence_pack_first_light \
    --pipeline ci \
    --validate-schemas
  ```
  This emits warnings for any schema failures and sets `schemas_ok` in `first_light_status.json`; pipelines remain observational.
- **External reproduction:** Run the same command locally, or invoke the checker directly:
  ```bash
  uv run python tools/evidence_schema_check.py results/first_light/evidence_pack_first_light/p3_synthetic/synthetic_raw.jsonl schemas/evidence/first_light_synthetic_raw.schema.json
  uv run python tools/evidence_schema_check.py results/first_light/evidence_pack_first_light/p3_synthetic/red_flag_matrix.json schemas/evidence/first_light_red_flag_matrix.schema.json
  uv run python tools/evidence_schema_check.py results/first_light/evidence_pack_first_light/p4_shadow/divergence_log.jsonl schemas/evidence/p4_divergence_log.schema.json
  ```
  Failures should be treated as advisory until the CI job is promoted to a gate.

This appends publication-ready SVGs under `visualizations/` (Δp, RSI, Ω occupancy) and records them in `manifest.json`. Reviewers can open each SVG to confirm the trajectories align with the reported statistics before proceeding.

**Expected Output Structure:**

```
results/verification/evidence_pack/
+-- p3_synthetic/           (6 files)
+-- p4_shadow/              (6 files)
+-- compliance/
|   +-- compliance_narrative.md
+-- visualizations/
|   +-- README.md
|   +-- delta_p_vs_cycles.svg
|   +-- rsi_vs_cycles.svg
|   +-- omega_occupancy_vs_cycles.svg
+-- manifest.json
```

When present, the Ledger Guard governed artifact is stored at `governance/ledger_guard_summary.json`. In `manifest.json`, governed artifacts that are explicitly schema-versioned and hash-locked are referenced under `governance.schema_versioned.*` (for Ledger Guard: `governance.schema_versioned.ledger_guard_summary`), which records the `schema_version` and `sha256` for the on-disk JSON so auditors can verify both the file path and its integrity deterministically. Trust boundary: the manifest reference is canonical; a disk-only summary (present on disk but not referenced under `governance.schema_versioned`) is lower confidence, and `first_light_status.json` reports this via `signals.ledger_guard.extraction_source` (`MANIFEST_REFERENCE` vs `DISK_FILE`).

**Total: 14 files**

**Verification (check manifest.json):**

```bash
uv run python -c "
import json
with open('results/verification/evidence_pack/manifest.json') as f:
    m = json.load(f)
print(f\"mode: {m['mode']}\")
print(f\"file_count: {m['file_count']}\")
print(f\"shadow_mode_compliance: {m['shadow_mode_compliance']}\")
"
```

**Expected:**
- `mode`: "SHADOW"
- `file_count`: 14
- `shadow_mode_compliance`: `{'all_divergence_logged_only': True, 'no_governance_modification': True, 'no_abort_enforcement': True}`

---

## 3. Determinism Verification Procedure

### 3.1 Run Determinism Check

**Command:**

```bash
uv run python scripts/verify_first_light_determinism.py
```

This script:
1. Runs P3 harness twice with identical configuration (seed=42, 100 cycles)
2. Runs P4 harness twice with identical configuration
3. Compares key artifacts between runs

**Execution time:** 30-60 seconds

### 3.2 Understanding "Identical Modulo Timestamps"

The comparison logic explicitly ignores:
- `timestamp` fields (per-record creation time)
- `start_time` / `end_time` fields (run timing)
- `run_id` fields (contain embedded timestamps like `p4_20251211_044926`)
- `timing` objects in summary reports

All other fields — numeric values, configuration, metrics, flags — must be **byte-identical** between runs.

**Rationale:** Timestamps are ephemeral metadata that vary between executions. The determinism contract is: "given the same seed and config, the mathematical/logical outputs are identical."

### 3.3 Interpreting Output

**PASS output:**

```
============================================================
First-Light Determinism Verification
============================================================

Running P3 harness (run 1)...
  Output: results/determinism_test/p3_run1/fl_20251211_123456_seed42
Running P3 harness (run 2)...
  Output: results/determinism_test/p3_run2/fl_20251211_123457_seed42
Running P4 harness (run 1)...
  Output: results/determinism_test/p4_run1/p4_20251211_123458
Running P4 harness (run 2)...
  Output: results/determinism_test/p4_run2/p4_20251211_123459

Comparing artifacts...

P3 Artifacts:
  stability_report.json: IDENTICAL
  tda_metrics.json: IDENTICAL
  metrics_windows.json: IDENTICAL

P4 Artifacts:
  p4_summary.json: IDENTICAL
  twin_accuracy.json: IDENTICAL

Raw Data (content comparison, ignoring timestamps):
  P3 synthetic_raw.jsonl: IDENTICAL (ignoring timestamps)
  P4 real_cycles.jsonl: IDENTICAL (ignoring timestamps)
  P4 twin_predictions.jsonl: IDENTICAL (ignoring timestamps)

============================================================
DETERMINISM CHECK: PASSED
All artifacts are identical between runs with same seed/config.
============================================================
```

**INVESTIGATE output:**

If any line shows `DIFFERS`, the output includes a brief diff description:

```
  stability_report.json: DIFFERS - .metrics.success_rate: 0.852 vs 0.847
```

**Triage steps for DIFFERS:**
1. Check if the differing field is truly deterministic (not a floating-point edge case)
2. Verify both runs used identical seed and config
3. Check for external state pollution (leftover files, environment variables)
4. If the difference is < 0.001 in floating-point values, this may be acceptable numerical noise

---

## 4. SHADOW MODE Verification Procedure

### 4.1 Run Compliance Tests

**Command:**

```bash
uv run python -m pytest tests/integration/test_shadow_mode_compliance.py -v
```

**Expected output:**

```
============================= test session starts =============================
...
collected 14 items

tests/integration/test_shadow_mode_compliance.py::TestP3ShadowModeCompliance::test_p3_harness_no_governance_api_calls PASSED
tests/integration/test_shadow_mode_compliance.py::TestP3ShadowModeCompliance::test_p3_harness_shadow_mode_enforced_in_config PASSED
tests/integration/test_shadow_mode_compliance.py::TestP3ShadowModeCompliance::test_p3_output_always_shadow_mode PASSED
tests/integration/test_shadow_mode_compliance.py::TestP4ShadowModeCompliance::test_p4_config_rejects_non_shadow_mode PASSED
tests/integration/test_shadow_mode_compliance.py::TestP4ShadowModeCompliance::test_p4_runner_enforces_shadow_mode PASSED
tests/integration/test_shadow_mode_compliance.py::TestP4ShadowModeCompliance::test_p4_runner_never_mutates_real_state PASSED
tests/integration/test_shadow_mode_compliance.py::TestP4ShadowModeCompliance::test_p4_divergence_snapshots_all_logged_only PASSED
tests/integration/test_shadow_mode_compliance.py::TestP4ShadowModeCompliance::test_p4_real_observations_all_shadow_mode PASSED
tests/integration/test_shadow_mode_compliance.py::TestP4ShadowModeCompliance::test_p4_harness_output_all_shadow_mode PASSED
tests/integration/test_shadow_mode_compliance.py::TestP4ShadowModeCompliance::test_p4_red_flag_status_logged_only PASSED
tests/integration/test_shadow_mode_compliance.py::TestP4ShadowModeCompliance::test_p4_divergence_status_logged_only PASSED
tests/integration/test_shadow_mode_compliance.py::TestShadowModeDataStructures::test_real_cycle_observation_default_shadow PASSED
tests/integration/test_shadow_mode_compliance.py::TestShadowModeDataStructures::test_twin_cycle_observation_default_shadow PASSED
tests/integration/test_shadow_mode_compliance.py::TestShadowModeDataStructures::test_divergence_snapshot_default_logged_only PASSED

============================= 14 passed in 1.xx s =============================
```

### 4.2 What the Tests Enforce

| Test Category | Enforcement |
|---------------|-------------|
| **No Governance Writes** | P3 harness source code is statically analyzed for forbidden patterns: `governance.enforce`, `governance.abort`, `governance.block`, `real_runner.stop`, `execute_abort`, etc. |
| **No Abort/Stop Calls** | P4 runner is wrapped with a `TrackedMockProvider` that raises `RuntimeError` if `set_state()`, `modify_governance()`, `abort()`, or `stop()` are called. The test runs 50 cycles and verifies zero mutation calls. |
| **mode="SHADOW" Everywhere** | All `RealCycleObservation` and `TwinCycleObservation` objects default to `mode="SHADOW"`. P4 summary and all JSONL records are verified to contain `mode="SHADOW"`. |
| **action="LOGGED_ONLY" Everywhere** | All `DivergenceSnapshot` objects default to `action="LOGGED_ONLY"`. Every line in `divergence_log.jsonl` is verified to have `action="LOGGED_ONLY"`. |
| **Config Validation** | `FirstLightConfig` and `FirstLightConfigP4` both reject `shadow_mode=False` at validation time. `FirstLightShadowRunnerP4` raises `ValueError` if instantiated with non-shadow config. |

### 4.3 Failure Interpretation

Any test failure indicates a SHADOW MODE violation. Examples:

- `test_p4_runner_never_mutates_real_state FAILED` — the runner called a mutation method on the telemetry provider
- `test_p4_divergence_snapshots_all_logged_only FAILED` — at least one divergence snapshot has `action != "LOGGED_ONLY"`

A SHADOW MODE violation is a **blocking issue** that must be investigated before any artifacts from that run can be trusted.

---

## 5. Evidence Pack Structure & Hash Checking

### 5.1 Directory Layout

```
evidence_pack_first_light/
+-- p3_synthetic/
|   +-- synthetic_raw.jsonl      # Raw P3 cycle data
|   +-- stability_report.json    # Aggregated metrics
|   +-- red_flag_matrix.json     # Red flag occurrences
|   +-- metrics_windows.json     # Windowed metrics
|   +-- tda_metrics.json         # TDA analysis
|   +-- run_config.json          # Frozen config
+-- p4_shadow/
|   +-- real_cycles.jsonl        # Real runner observations
|   +-- twin_predictions.jsonl   # Shadow twin predictions
|   +-- divergence_log.jsonl     # All divergence events
|   +-- p4_summary.json          # Aggregated P4 metrics
|   +-- twin_accuracy.json       # Twin accuracy breakdown
|   +-- run_config.json          # Frozen config
+-- compliance/
|   +-- compliance_narrative.md  # SHADOW MODE declaration
+-- visualizations/
|   +-- README.md                # Placeholder for future viz
+-- manifest.json                # SHA-256 hashes + metadata
```

### 5.2 Validate Manifest Hashes Against Files

Use the canonical integrity verification script:

**Command:**

```bash
uv run python scripts/verify_evidence_pack_integrity.py
```

**With custom pack directory:**

```bash
uv run python scripts/verify_evidence_pack_integrity.py --pack-dir results/verification/evidence_pack
```

**With JSON output for automation:**

```bash
uv run python scripts/verify_evidence_pack_integrity.py --json-output integrity_report.json
```

The script:
- Loads `manifest.json` from the evidence pack directory
- Normalizes paths (Windows backslashes → POSIX forward slashes)
- Recomputes SHA-256 for every file listed in the manifest
- Reports OK/MISSING/MISMATCH for each file
- Emits a summary with total counts
- Outputs the manifest's own SHA-256 for tamper detection

**SHADOW MODE behavior:** The script never modifies any files and always exits 0 (except for fatal I/O errors). The `status` field in JSON output indicates PASSED/FAILED for downstream consumption.

**Expected output:**

```
OK: p3_synthetic/synthetic_raw.jsonl
OK: p3_synthetic/stability_report.json
OK: p3_synthetic/red_flag_matrix.json
OK: p3_synthetic/metrics_windows.json
OK: p3_synthetic/tda_metrics.json
OK: p3_synthetic/run_config.json
OK: p4_shadow/real_cycles.jsonl
OK: p4_shadow/twin_predictions.jsonl
OK: p4_shadow/divergence_log.jsonl
OK: p4_shadow/p4_summary.json
OK: p4_shadow/twin_accuracy.json
OK: p4_shadow/run_config.json
OK: visualizations/README.md
OK: compliance/compliance_narrative.md

INTEGRITY CHECK: PASSED
```

### 5.4 Evidence Gap Reporter (read-only)

Use the gap reporter to check which required artifacts are present or missing in an evidence pack directory. The tool is advisory only and does not modify files.

```bash
python -m tools.evidence_gap_report --root results/first_light/evidence_pack_first_light
```

Example output:

```json
{
  "present": [
    "manifest.json",
    "p3_synthetic/synthetic_raw.jsonl",
    "p4_shadow/divergence_log.jsonl"
  ],
  "missing": [
    "p3_synthetic/stability_report.json",
    "p3_synthetic/red_flag_matrix.json",
    "p3_synthetic/metrics_windows.json",
    "p3_synthetic/tda_metrics.json",
    "p4_shadow/real_cycles.jsonl",
    "p4_shadow/twin_predictions.jsonl",
    "p4_shadow/p4_summary.json",
    "p4_shadow/twin_accuracy.json",
    "p4_shadow/tda_metrics.json",
    "compliance/compliance_narrative.md"
  ]
}
```

### 5.3 Compute Manifest Hash (Tamper Detection)

To verify the manifest itself hasn't been modified, compute its SHA-256:

```bash
# Unix/macOS
shasum -a 256 results/first_light/evidence_pack_first_light/manifest.json

# Windows PowerShell
Get-FileHash -Algorithm SHA256 results/first_light/evidence_pack_first_light/manifest.json

# Python (cross-platform)
uv run python -c "
import hashlib
with open('results/first_light/evidence_pack_first_light/manifest.json', 'rb') as f:
    print(hashlib.sha256(f.read()).hexdigest())
"
```

Record this hash externally (e.g., in a secure audit log). Any future verification should produce the same hash for an unmodified manifest.

---

## 6. Lean Shadow First-Light Summary

### 6.1 What is the Lean Shadow First-Light Summary?

The Lean Shadow First-Light Summary is a structural health annex for the proof pipeline that appears in evidence packs under `governance.lean_shadow.first_light_summary`. This summary explains whether the proof pipeline itself is stable, independent of the core governance signals.

**Important:** This summary is intended as a structural health annex for the proof pipeline, not a hard gate. It is purely observational and does not block or gate any operations.

### 6.2 Understanding the Fields

The Lean Shadow summary contains four key fields:

#### `structural_error_rate` (float, 0.0-1.0)

**What it means:** The proportion of verification failures that are due to structural issues (logical errors, type mismatches, formula complexity issues) rather than resource limits (timeouts, memory constraints).

**In practice:**
- **Low (0.0-0.2):** Most failures are resource-related, suggesting the pipeline logic is sound but may need budget adjustments.
- **Medium (0.2-0.5):** Mix of structural and resource issues. Review anomaly patterns to identify systematic problems.
- **High (>0.5):** Predominantly structural failures. This suggests potential issues with formula encoding, type checking, or logical consistency in the verification pipeline.

**Example interpretation:**
- `structural_error_rate: 0.15` → 15% of failures are structural. This is acceptable; most issues are resource constraints.
- `structural_error_rate: 0.45` → 45% structural failures. Review the `dominant_anomalies` to identify patterns.

#### `shadow_resource_band` ("LOW" | "MEDIUM" | "HIGH")

**What it means:** A classification of resource consumption patterns based on formula complexity and resource limit hit rates.

**In practice:**
- **LOW:** Simple formulas, minimal resource pressure. Pipeline is operating well within resource bounds.
- **MEDIUM:** Moderate complexity, occasional resource limits. Pipeline is functioning but may benefit from optimization.
- **HIGH:** High complexity or frequent resource limits. Pipeline may be approaching capacity constraints.

**Example interpretation:**
- `shadow_resource_band: "LOW"` → Pipeline is handling verification requests efficiently.
- `shadow_resource_band: "HIGH"` → Pipeline is under resource pressure. Consider reviewing timeout/memory budgets or formula complexity.

#### `dominant_anomalies` (List[str], up to 5)

**What it means:** Hash-based signatures of the most common error patterns observed during verification. These are deterministic identifiers that cluster similar failure modes.

**In practice:**
- Each signature represents a class of errors (e.g., type mismatches, timeout patterns, memory issues).
- Multiple occurrences of the same signature indicate a systematic issue rather than random failures.
- Review these signatures to identify recurring problems in the verification pipeline.

**Example interpretation:**
- `dominant_anomalies: ["abc123", "def456"]` → Two distinct error patterns are dominant. Investigate what these patterns represent.
- `dominant_anomalies: []` → No recurring error patterns. Failures are diverse or infrequent.

#### `status` ("OK" | "WARN" | "BLOCK")

**What it means:** An overall health indicator derived from `structural_error_rate` and `shadow_resource_band`.

**Status logic:**
- **OK:** `structural_error_rate < 0.2` AND `shadow_resource_band == "LOW"`
- **WARN:** `structural_error_rate >= 0.2` OR `shadow_resource_band == "MEDIUM"`
- **BLOCK:** `structural_error_rate > 0.5` OR `shadow_resource_band == "HIGH"`

**Important:** This status is **observational only**. It does not gate or block any operations. It provides context for understanding pipeline health.

### 6.3 Interpreting Discrepancies

#### When Everything Else is GREEN but Lean Shadow Says WARN

**Scenario:** Core governance signals (USLA, TDA, semantic alignment) all show GREEN, but Lean Shadow summary shows WARN.

**What this means:**
- The verification pipeline itself may have structural issues that don't immediately affect governance signals.
- This could indicate:
  - Tooling issues (Lean integration, formula encoding)
  - Formula complexity mismatches
  - Resource budget calibration problems

**What a reviewer should do:**
1. **Inspect anomalies:** Review `dominant_anomalies` to identify specific error patterns.
2. **Verify Lean integration:** Check if Lean adapter is functioning correctly (version, availability, configuration).
3. **Treat as potential tooling issue vs core-governance issue:**
   - If anomalies are tooling-related (Lean version mismatches, encoding errors), this is a tooling issue.
   - If anomalies indicate logical inconsistencies in proof generation, this may be a core-governance issue.
4. **Review resource bands:** If `shadow_resource_band` is MEDIUM or HIGH, consider whether resource budgets need adjustment.

**Example:**
```json
{
  "governance": {
    "usla": {"status": "GREEN"},
    "tda": {"status": "GREEN"},
    "semantic": {"status": "GREEN"},
    "lean_shadow": {
      "first_light_summary": {
        "status": "WARN",
        "structural_error_rate": 0.25,
        "shadow_resource_band": "LOW",
        "dominant_anomalies": ["type_mismatch_abc123"]
      }
    }
  }
}
```

**Interpretation:** Core governance is healthy, but Lean verification is showing structural issues. The `type_mismatch_abc123` anomaly suggests formula encoding or type checking problems. This is likely a tooling issue rather than a core-governance problem, but should be investigated.

### 6.4 When to Escalate

**The Lean Shadow summary does NOT trigger escalation or gating.** It is purely informational.

However, reviewers may want to investigate further if:
- `status: "BLOCK"` with high `structural_error_rate` (>0.5) — suggests systematic pipeline issues
- Multiple distinct anomalies in `dominant_anomalies` — indicates diverse failure modes
- `shadow_resource_band: "HIGH"` — suggests resource capacity concerns

**Remember:** This is a structural health annex, not a gate. Use it to understand pipeline stability, not to block operations.

### 6.5 How to Read Lean Shadow vs Structural Cohesion in P5

**P5 Structural Regime Monitor:** In P5 calibration experiments, Lean Shadow serves as a "Proof Pipeline Regime Detector" that monitors structural health across calibration runs. This section explains how to interpret the cross-check between Lean Shadow signals and Structural Cohesion signals.

#### 6.5.1 Understanding the Cross-Check

The Lean Shadow vs Structural Cohesion cross-check appears in evidence packs under `governance.structure.lean_cross_check`. This comparison detects consistency, tension, or conflict between:

- **Lean Shadow Signal:** Proof pipeline structural health (verification layer)
- **Structural Cohesion Signal:** Architectural structural health (DAG/Topology/HT layers)

**Critical Principle:** This cross-check is **advisory only**. It does not gate or block any operations. It provides context for understanding alignment between verification pipeline health and architectural layer health.

#### 6.5.2 Cross-Check Status Values

The cross-check produces three possible status values:

| Status | Meaning | Interpretation |
|--------|---------|----------------|
| **CONSISTENT** | Both signals agree | Lean Shadow and Structural Cohesion are aligned (both OK/CONSISTENT, both WARN/TENSION, or both BLOCK/CONFLICT) |
| **TENSION** | Signals partially disagree | One signal is worse than the other but not opposite (e.g., Lean WARN vs Structural CONSISTENT) |
| **CONFLICT** | Signals strongly disagree | Signals are opposite (e.g., Lean OK vs Structural CONFLICT, or Lean BLOCK vs Structural CONSISTENT) |

**Example CONSISTENT:**
```json
{
  "governance": {
    "structure": {
      "lean_cross_check": {
        "status": "CONSISTENT",
        "lean_signal_severity": "OK",
        "structural_signal_severity": "CONSISTENT",
        "advisory_notes": [
          "Both Lean shadow and structural cohesion signals indicate healthy state."
        ]
      }
    }
  }
}
```

**Example TENSION:**
```json
{
  "governance": {
    "structure": {
      "lean_cross_check": {
        "status": "TENSION",
        "lean_signal_severity": "WARN",
        "structural_signal_severity": "CONSISTENT",
        "advisory_notes": [
          "Signals partially disagree: Lean shadow is WARN while structural cohesion is CONSISTENT. Review both signals for context."
        ]
      }
    }
  }
}
```

**Example CONFLICT:**
```json
{
  "governance": {
    "structure": {
      "lean_cross_check": {
        "status": "CONFLICT",
        "lean_signal_severity": "OK",
        "structural_signal_severity": "CONFLICT",
        "advisory_notes": [
          "Signals strongly disagree: Lean shadow is OK but structural cohesion is CONFLICT. This suggests a potential mismatch between verification pipeline and architectural layers."
        ]
      }
    }
  }
}
```

#### 6.5.3 Per-Experiment Structural Summary

For P5 calibration experiments (CAL-EXP-1, CAL-EXP-2, CAL-EXP-3), each experiment produces a structural summary that appears in the calibration report under `structural_summary`. This summary aggregates Lean Shadow tiles across the experiment to detect:

- **Mean/Max Error Rates:** Average and peak structural error rates across the experiment
- **Anomaly Bursts:** Sustained periods (3+ consecutive tiles) with high error rates (≥0.3)
- **Dominant Anomalies:** Most frequent error patterns across the experiment

**Example Structural Summary:**
```json
{
  "structural_summary": {
    "schema_version": "1.0.0",
    "mean_structural_error_rate": 0.15,
    "max_structural_error_rate": 0.35,
    "anomaly_bursts": [
      {
        "start_index": 10,
        "end_index": 12,
        "length": 3,
        "mean_error_rate": 0.33,
        "max_error_rate": 0.35
      }
    ],
    "dominant_anomalies": ["type_mismatch_abc123", "timeout_pattern_def456"]
  }
}
```

**Interpreting Anomaly Bursts:**
- **Sustained Bursts (3+ consecutive):** Indicates systematic structural issues rather than isolated spikes
- **Single Spikes (<3 consecutive):** Ignored as transient noise
- **Multiple Bursts:** Suggests recurring structural problems that may need investigation

#### 6.5.4 When to Investigate Cross-Check Disagreements

**CONSISTENT Status:**
- **Both OK/CONSISTENT:** Healthy state — verification pipeline and architectural layers are aligned
- **Both WARN/TENSION:** Moderate concerns — both layers indicate issues; signals are aligned
- **Both BLOCK/CONFLICT:** Significant issues — both layers indicate problems; signals are aligned

**TENSION Status:**
- **Lean WARN vs Structural CONSISTENT:** Verification pipeline shows issues but architectural layers are healthy
  - **Possible causes:** Tooling issues (Lean integration, formula encoding), formula complexity mismatches, resource budget calibration problems
  - **Action:** Review `dominant_anomalies` in Lean Shadow summary; verify Lean adapter configuration
- **Lean CONSISTENT vs Structural TENSION:** Architectural layers show issues but verification pipeline is healthy
  - **Possible causes:** DAG/Topology/HT layer inconsistencies that don't affect verification pipeline
  - **Action:** Review Structural Cohesion details (DAG status, topology status, HT status)

**CONFLICT Status:**
- **Lean OK vs Structural CONFLICT:** Verification pipeline is healthy but architectural layers show significant issues
  - **Possible causes:** Architectural layer problems (DAG cycles, topology violations, HT inconsistencies) that don't affect verification pipeline operation
  - **Action:** Investigate Structural Cohesion violations; verify DAG/Topology/HT layer integrity
- **Lean BLOCK vs Structural CONSISTENT:** Verification pipeline shows significant issues but architectural layers are healthy
  - **Possible causes:** Verification pipeline tooling issues (Lean adapter failures, formula encoding problems) that don't affect architectural layers
  - **Action:** Review Lean Shadow `dominant_anomalies`; verify Lean adapter availability and configuration

#### 6.5.5 Advisory Notes Interpretation

The cross-check includes `advisory_notes` that provide context-specific guidance:

- **High Error Rate Notes:** Appear when `mean_structural_error_rate > 0.3`
  - Suggests reviewing verification pipeline stability
- **Low Cohesion Notes:** Appear when `cohesion_score < 0.7`
  - Suggests reviewing DAG/Topology/HT layer consistency
- **Anomaly Burst Notes:** Appear when `anomaly_bursts` is non-empty
  - Indicates sustained structural issues rather than isolated spikes

**Example Advisory Notes:**
```json
{
  "advisory_notes": [
    "Signals partially disagree: Lean shadow is WARN while structural cohesion is CONSISTENT. Review both signals for context.",
    "Lean shadow shows elevated mean error rate (35.0%). Consider reviewing verification pipeline stability.",
    "Detected 1 anomaly burst(s) in Lean shadow. This indicates sustained structural issues rather than isolated spikes."
  ]
}
```

#### 6.5.6 P5 Calibration Experiment Workflow

When reviewing P5 calibration experiment evidence packs:

1. **Check Structural Summary:** Review `structural_summary` in each CAL-EXP report
   - Verify `mean_structural_error_rate` is within acceptable range (<0.3 for healthy)
   - Check for `anomaly_bursts` — sustained bursts indicate systematic issues
   - Review `dominant_anomalies` to identify recurring error patterns

2. **Check Cross-Check:** Review `governance.structure.lean_cross_check` in evidence pack
   - Verify `status` (CONSISTENT/TENSION/CONFLICT)
   - Read `advisory_notes` for context-specific guidance
   - Compare `lean_signal_severity` vs `structural_signal_severity`

3. **Interpret Disagreements:**
   - **TENSION:** Partial disagreement — review both signals for context
   - **CONFLICT:** Strong disagreement — investigate potential mismatch between verification pipeline and architectural layers

4. **Escalate if Needed:**
   - **CONFLICT with Lean BLOCK:** Investigate verification pipeline tooling issues
   - **CONFLICT with Structural CONFLICT:** Investigate architectural layer integrity
   - **Multiple Bursts:** Review recurring structural problems across experiments

**Remember:** The cross-check is **advisory only**. It does not gate or block operations. Use it to understand alignment between verification pipeline health and architectural layer health, not to block calibration experiments.

---

## 7. Control Arm Calibration Panel

### 7.1 What is the Control Arm Calibration Panel?

The Control Arm Calibration Panel is a P5 calibration comparator that appears in evidence packs under `governance.mock_oracle_panel`. This panel compares control arm (mock oracle) summaries against twin (real verifier) summaries across multiple CAL-EXP-* experiments to prove we can distinguish noise (expected stochasticity) from signal (actual verification behavior).

**Critical Principle:** Control ≠ twin is **expected and good**. Equality would be a **red flag** indicating overfitting or lack of sensitivity.

### 7.2 Understanding the Panel Structure

The panel contains:

#### `schema_version` (string)
Schema version for the panel format (currently "1.0.0").

#### `experiments` (List[str])
List of experiment names analyzed (e.g., `["CAL-EXP-001", "CAL-EXP-002"]`).

#### `control_vs_twin_delta` (Dict[str, Dict])
Dictionary mapping experiment names to delta metrics:

```json
{
  "CAL-EXP-001": {
    "abstention_rate_delta": 0.08,
    "invalid_rate_delta": 0.13,
    "status_match": false
  }
}
```

- **`abstention_rate_delta`**: Absolute difference between control and twin abstention rates
- **`invalid_rate_delta`**: Absolute difference between control and twin invalid rates
- **`status_match`**: Boolean indicating whether control and twin have matching status

#### `red_flags` (List[str])
List of red flag descriptions (empty if all checks pass). Red flags are raised when:
- Control and twin metrics are too similar (within 1% on both metrics) — suggests overfitting or lack of sensitivity
- Control and twin have matching status with similar rates — suggests pipeline not distinguishing between expected stochasticity and actual behavior

### 7.3 Interpreting Control vs Twin Differences

**Expected Behavior:**
- Control (mock oracle) should show baseline stochasticity based on its profile (e.g., `uniform` ≈ 33% abstention, 33% invalid)
- Twin (real verifier) should show actual verification behavior, which may differ significantly from control
- **Differences are good** — they prove the pipeline can distinguish noise from signal

**Red Flags (What Would Indicate a Problem):**

1. **Too Similar (within 1%):**
   ```json
   {
     "abstention_rate_delta": 0.005,
     "invalid_rate_delta": 0.003
   }
   ```
   **Interpretation:** Control and twin are nearly identical. This suggests:
   - The pipeline may be overfitting to deterministic patterns
   - The pipeline may lack sensitivity to actual verification behavior
   - The mock oracle may not be providing a proper negative control baseline

2. **Matching Status with Similar Rates:**
   ```json
   {
     "status_match": true,
     "abstention_rate_delta": 0.02,
     "invalid_rate_delta": 0.03
   }
   ```
   **Interpretation:** Control and twin have the same status with very similar rates. This suggests the pipeline is not distinguishing between expected stochasticity (control) and actual behavior (twin).

**Good Signs (What Indicates Healthy Calibration):**

1. **Significant Differences:**
   ```json
   {
     "abstention_rate_delta": 0.25,
     "invalid_rate_delta": 0.15,
     "status_match": false
   }
   ```
   **Interpretation:** Control and twin differ appropriately. This proves the pipeline can distinguish between expected stochasticity and actual verification behavior.

2. **No Red Flags:**
   An empty `red_flags` list indicates all experiments passed calibration checks.

### 7.4 Control Arm Summary Format

Each CAL-EXP-* experiment produces a control arm summary (stored in `calibration/control_arm_*.json`):

```json
{
  "schema_version": "1.0.0",
  "status": "OK",
  "abstention_rate": 0.33,
  "invalid_rate": 0.33,
  "total_queries": 100,
  "profile": "uniform"
}
```

**Fields:**
- `schema_version`: Schema version for the summary format
- `status`: Fleet status ("OK" | "DRIFTING" | "BROKEN")
- `abstention_rate`: Fraction of abstentions (0.0-1.0)
- `invalid_rate`: Fraction of invalid results (0.0-1.0)
- `total_queries`: Total number of queries processed
- `profile`: Profile name used (e.g., "uniform", "timeout_heavy", "invalid_heavy")

### 7.5 Verification Checklist

When reviewing an evidence pack with a control arm calibration panel, verify:

- [ ] `governance.mock_oracle_panel` is present
- [ ] `schema_version` matches expected version ("1.0.0")
- [ ] `experiments` list contains all CAL-EXP-* experiments
- [ ] `control_vs_twin_delta` contains entries for all experiments
- [ ] `red_flags` list is empty or contains interpretable warnings
- [ ] Control and twin metrics differ appropriately (they should not match)
- [ ] Delta values are reasonable (typically > 0.05 for healthy calibration)

**Important:** The control arm calibration panel is a **calibration/control signal only**. It does not gate or block any operations. It serves purely to validate that the pipeline can distinguish between expected stochasticity (control) and actual verification behavior (twin).

---

## 8. Telemetry × Behavior Consistency Panel

### 8.1 What is the Telemetry × Behavior Consistency Panel?

The Telemetry × Behavior Consistency Panel is a **calibration-only diagnostic** that appears in evidence packs under `governance.telemetry_behavior_panel`. This panel provides a cross-experiment view of telemetry-behavior alignment to answer: "Do telemetry anomalies agree with behavioral issues across CAL-EXP-1/2/3?"

**Critical Principle:** This panel is **observational and advisory only**. It does not gate or block any operations. It serves purely as a diagnostic tool to identify where telemetry governance signals and behavior metrics (readiness/performance) disagree across calibration experiments.

### 8.2 Understanding the Panel Structure

The panel contains:

#### `schema_version` (string)
Schema version for the panel format (currently "1.0.0").

#### `total_experiments` (int)
Total number of calibration experiments analyzed (e.g., 3 for CAL-EXP-1, CAL-EXP-2, CAL-EXP-3).

#### `consistency_counts` (Dict[str, int])
Counts of consistency statuses across experiments:

```json
{
  "CONSISTENT": 2,
  "INCONSISTENT": 1,
  "PARTIAL": 0
}
```

- **`CONSISTENT`**: Telemetry and behavior metrics are aligned (e.g., both GREEN, or both showing issues)
- **`INCONSISTENT`**: Telemetry shows issues (RED/YELLOW) while behavior metrics are healthy (GREEN), or vice versa
- **`PARTIAL`**: Mixed alignment pattern (e.g., telemetry aligns with readiness but not with performance)

#### `inconsistent_experiments` (List[Dict])
List of experiments showing inconsistency, with brief reasons:

```json
[
  {
    "cal_id": "cal_exp2",
    "reason": "telemetry YELLOW vs readiness GREEN"
  }
]
```

#### `summary` (string)
Neutral summary text describing the overall consistency pattern across experiments.

### 8.3 Per-Experiment Consistency Snapshots

Each calibration experiment produces a consistency snapshot (stored in `calibration/telemetry_behavior_consistency_<cal_id>.json`):

```json
{
  "schema_version": "1.0.0",
  "cal_id": "cal_exp1",
  "consistency_status": "INCONSISTENT",
  "telemetry_status": "YELLOW",
  "readiness_status": "GREEN",
  "perf_status": "GREEN"
}
```

These snapshots capture the consistency status between telemetry governance and behavior metrics for a single experiment.

### 8.4 Interpreting Consistency Patterns

**Expected Behavior:**
- **CONSISTENT**: Telemetry and behavior metrics align (e.g., both GREEN when healthy, both RED/YELLOW when issues detected)
- **INCONSISTENT**: Telemetry shows issues while behavior metrics appear healthy, or vice versa — this may indicate:
  - Telemetry anomalies that do not align with behavioral issues
  - Behavior metrics that miss issues detected by telemetry
  - Timing differences between telemetry and behavior metric collection
- **PARTIAL**: Mixed alignment (e.g., telemetry aligns with readiness but not with performance)

**What This Panel Tells You:**
- **All CONSISTENT**: Telemetry and behavior metrics are well-aligned across all experiments
- **Some INCONSISTENT**: Mixed alignment pattern — some experiments show telemetry-behavior disagreement
- **All INCONSISTENT**: Telemetry anomalies do not align with behavioral issues across experiments — may indicate systematic misalignment

**Important:** The telemetry × behavior consistency panel is a **calibration-only diagnostic**. It does not gate or block any operations. It serves purely as a way to see where telemetry and behavior disagree across experiments.

---

## 9. Known Caveats

### 8.1 P3 Omega Occupancy Below Threshold

**Observation:** P3 omega_occupancy = 85.1%, threshold = 90%

**Meaning:** The synthetic twin spends ~15% of cycles outside the designated "safe region" (Omega). Under live governance, this would trigger a hypothetical abort condition.

**Implication:** This is observation-only. No abort was executed because SHADOW MODE is enforced. The flag is recorded in `stability_report.json` under `red_flag_summary.hypothetical_abort: true`.

**Action required:** None for verification purposes. This is a legitimate signal that the synthetic model has stability concerns. Future work may tune parameters or investigate root cause.

### 8.2 P4 High Divergence Rate

**Observation:** P4 divergence_rate = 97.2%

**Meaning:** The shadow twin's predictions diverged from the real runner's observations on 97.2% of cycles.

**Cause:** P4 uses `MockTelemetryProvider`, which generates synthetic "real" data that doesn't correlate with the twin's internal state. This is expected behavior for mock-mode testing.

**Implication:** When real telemetry is integrated (Phase X P5), divergence rate should drop significantly. The current high rate validates that divergence detection is working correctly.

### 6.3 Manifest Path Format

**Observation:** `manifest.json` paths use OS-native separators (backslashes on Windows).

**Implication:** Cross-platform consumers may need to normalize paths. A future revision will standardize to POSIX-style forward slashes.

---

## 7. Compliance Statement

This First Light Golden Run was executed in **SHADOW MODE only**.

- No governance decisions were modified or enforced
- All divergence records have `action="LOGGED_ONLY"`
- All cycle observations have `mode="SHADOW"`
- The P3 and P4 harnesses contain no governance API calls

**SIG-PAT Invariant:** SIG-PAT (P5 Patterns Panel signal) cannot be the sole cause of a BLOCK decision; the `conflict=false` invariant and LOW precedence (GGFL precedence=9)[^sigpat] are enforced by `tests/health/test_p5_patterns_panel.py::TestSoloWarnCannotBlock::test_solo_p5_patterns_warn_cannot_cause_block`. Non-gating: this invariant affects explanation only, not exit codes.

[^sigpat]: Precedence=9 implies advisory signal; tests verify solo warn case, code enforces `conflict=false` universally.

**Phase X Status:** Shadow observation complete. No claims are made about uplift, performance, or safety beyond what the artifacts directly show.

If you run the commands in this document, you will see exactly what we saw.

---

## 8. Cross-Platform Validation Plan

This section defines the validation matrix for cross-platform verification. **This is a specification only** — no CI jobs exist yet.

### 8.1 Validation Matrix

| OS | Python | Status | Notes |
|----|--------|--------|-------|
| Windows 11 | 3.11.9 | **Validated** | Primary development platform |
| Windows 11 | 3.10.x | Untested | Should work; test before release |
| Windows 11 | 3.12.x | Untested | May have typing changes |
| Ubuntu 22.04 LTS | 3.11.x | Untested | Primary Linux target |
| Ubuntu 22.04 LTS | 3.10.x | Untested | Minimum supported version |
| Ubuntu 22.04 LTS | 3.12.x | Untested | May require dependency updates |
| macOS Sonoma | 3.11.x | Untested | ARM64 (Apple Silicon) |
| macOS Sonoma | 3.10.x | Untested | Homebrew Python |
| macOS Sonoma | 3.12.x | Untested | May require dependency updates |

### 8.2 Commands Per Cell

For each (OS, Python) cell, run:

```bash
# 1. Setup
uv sync  # or: pip install -e .

# 2. P3 Harness (short run for validation)
uv run python scripts/usla_first_light_harness.py \
    --cycles 100 --seed 42 --slice arithmetic_simple \
    --runner-type u2 --tau-0 0.20 --window-size 50 \
    --output-dir results/validation/p3

# 3. P4 Harness (short run for validation)
uv run python scripts/usla_first_light_p4_harness.py \
    --cycles 100 --seed 42 --slice arithmetic_simple \
    --runner-type u2 --tau-0 0.20 \
    --output-dir results/validation/p4

# 4. Evidence Pack Build
uv run python scripts/build_first_light_evidence_pack.py \
    --p3-dir results/validation/p3 \
    --p4-dir results/validation/p4 \
    --output-dir results/validation/evidence_pack

# 5. Integrity Verification
uv run python scripts/verify_evidence_pack_integrity.py \
    --pack-dir results/validation/evidence_pack

# 6. SHADOW MODE Compliance
uv run python -m pytest tests/integration/test_shadow_mode_compliance.py -v
```

### 8.3 Expected Failure Modes

| Failure Mode | Likely OS | Symptom | Triage |
|--------------|-----------|---------|--------|
| Path separator issues | Linux/macOS | `FileNotFoundError` on manifest paths | Check `normalize_path()` in integrity verifier |
| Locale/encoding | Linux | `UnicodeDecodeError` on JSONL files | Ensure UTF-8 locale (`LANG=en_US.UTF-8`) |
| Float formatting | All | Determinism check shows tiny diffs | Check if diff < 1e-9 (acceptable numerical noise) |
| Missing dependencies | All | `ModuleNotFoundError` | Run `uv sync` or check `pyproject.toml` |
| Permission denied | Linux/macOS | `PermissionError` on results/ | Check directory permissions, run as correct user |
| Line endings | Windows→Linux | JSON parse errors | Ensure CRLF→LF conversion on checkout |
| Timestamp format | macOS | ISO8601 parsing differences | Check `datetime.fromisoformat()` compatibility |

### 8.4 Validation Acceptance Criteria

A cell passes if:
- [ ] `uv sync` completes without errors
- [ ] P3 harness produces 6 artifacts
- [ ] P4 harness produces 6 artifacts
- [ ] Evidence pack has 14 files
- [ ] Integrity verifier reports `PASSED`
- [ ] SHADOW MODE compliance: 14/14 tests pass
- [ ] No unhandled exceptions in any step

### 8.5 Triage Procedure

If a cell fails:

1. **Capture full output** including stack traces
2. **Identify failure category** from table above
3. **Check if OS-specific** by comparing to Windows baseline
4. **Document workaround** if one exists
5. **File issue** with `[cross-platform]` label

Do not modify the codebase to fix cross-platform issues until:
- The issue is documented
- The fix is reviewed
- Windows baseline still passes after fix

### 8.6 Phase X P5 Validation Extension

When P5 (real telemetry) is ready, the validation matrix will be extended:

**Extended Matrix (P5):**

| OS | Python | Mock Status | P5 Status | Notes |
|----|--------|-------------|-----------|-------|
| Windows 11 | 3.11.9 | **Validated** | P5: not yet validated | Primary platform |
| Ubuntu 22.04 LTS | 3.11.x | Untested | P5: not yet validated | CI target |
| macOS Sonoma | 3.11.x | Untested | P5: not yet validated | Developer platform |

**P5 Validation Procedure:**

For each validated mock cell, re-run with real telemetry:

```bash
# P5 validation (when adapter is available)
uv run python scripts/usla_first_light_p4_harness.py \
    --cycles 100 --seed 42 --slice arithmetic_simple \
    --runner-type u2 --tau-0 0.20 \
    --telemetry-adapter real \
    --adapter-config config/real_telemetry_adapter.json \
    --output-dir results/validation/p4_real
```

**P5 Acceptance Criteria (in addition to 8.4):**

- [ ] Real telemetry adapter connects successfully
- [ ] P4 divergence rate < mock baseline (97.2%)
- [ ] P4 divergence rate in acceptable range (ideally < 30%)
- [ ] No adapter-specific errors on target OS
- [ ] Determinism check passes (with real telemetry timing tolerance)

**Comparison Report:**

After P5 validation, generate comparison:

```bash
uv run python scripts/compare_first_light_runs.py \
    --mock-pack results/validation/evidence_pack_mock \
    --real-pack results/validation/evidence_pack_real \
    --output comparison_report.json
```

This script will be created when P5 is implemented.

### 8.7 Distinguishing Mock vs Real Evidence Packs

External verifiers will receive evidence packs from different telemetry sources. This section describes how to identify the source.

**Naming Convention:**

| Telemetry Source | Evidence Pack Name | Run Directory Pattern |
|------------------|--------------------|-----------------------|
| Mock (current) | `evidence_pack_first_light_mock` | `fl_<timestamp>_seed42_mock` |
| Real (P5) | `evidence_pack_first_light_real` | `fl_<timestamp>_seed42_real` |
| Legacy (implicit mock) | `evidence_pack_first_light` | `fl_<timestamp>_seed42` |

**Detection Methods:**

1. **Check directory/pack name suffix:**
   - `_mock` → Mock telemetry
   - `_real` → Real telemetry
   - No suffix → Legacy mock (treat as mock)

2. **Check `first_light_status.json` (when present):**
   ```json
   {
     "schema_version": "1.2.0",
     "telemetry_source": "mock" | "real_synthetic" | "real_trace",
     ...
   }
   ```
   - `schema_version >= 1.2.0` includes `telemetry_source` + `proof_snapshot_present`
   - `schema_version < 1.2.0` �+' implicit mock

3. **Check `run_config.json` in P4 directory:**
   ```json
   {
     "telemetry_adapter": "mock" | "real",
     ...
   }
   ```
   - Field present → use its value
   - Field absent → implicit mock

4. **Check `p4_summary.json` source field (P5+):**
   ```json
   {
     "telemetry_source": "mock" | "real_synthetic" | "real_trace",
     ...
   }
   ```

**Verification Code (Python):**

```python
def detect_telemetry_source(evidence_pack_dir: Path) -> str:
    """
    Detect telemetry source from evidence pack.

    Returns: "mock" | "real_synthetic" | "real_trace" | "unknown"
    """
    # Method 1: Directory name
    dir_name = evidence_pack_dir.name.lower()
    if "_real_trace" in dir_name or dir_name.endswith("_real"):
        return "real_trace"
    if "_real_synth" in dir_name or "_real_synthetic" in dir_name:
        return "real_synthetic"
    if "_mock" in dir_name:
        return "mock"

    # Method 2: Status JSON
    status_file = evidence_pack_dir / "first_light_status.json"
    if status_file.exists():
        with open(status_file) as f:
            status = json.load(f)
        source = status.get("telemetry_source")
        if source in {"mock", "real_synthetic", "real_trace"}:
            return source

    # Method 3: P4 run_config
    p4_config = evidence_pack_dir / "p4_shadow" / "run_config.json"
    if p4_config.exists():
        with open(p4_config) as f:
            config = json.load(f)
        adapter = config.get("telemetry_adapter") or config.get("telemetry_source")
        if adapter in {"mock", "real_synthetic", "real_trace"}:
            return adapter

    # Method 4: Manifest (when status is missing)
    manifest_file = evidence_pack_dir / "manifest.json"
    if manifest_file.exists():
        with open(manifest_file) as f:
            manifest = json.load(f)
        source = manifest.get("telemetry_source") or manifest.get("configuration", {}).get("telemetry_source")
        if source in {"mock", "real_synthetic", "real_trace"}:
            return source

    return "unknown"
```

**Interpreting Results by Source:**

| Metric | Mock Interpretation | Real Interpretation |
|--------|---------------------|---------------------|
| P4 divergence_rate ~97% | **Expected** — noise baseline | **Problem** — twin not tracking |
| P4 divergence_rate <30% | **Unexpected** — check for bugs | **Good** — acceptable alignment |
| P4 twin_success_accuracy | Baseline only (no correlation) | Meaningful accuracy metric |

**Warning:** Do not compare mock and real packs directly without accounting for the fundamental difference in what they measure. Mock packs measure alarm system behavior; real packs measure twin accuracy.

---

## 9. Generating first_light_status.json

The status generator script creates a machine-readable summary of First Light run status.

**Command:**

```bash
uv run python scripts/generate_first_light_status.py \
    --p3-dir results/first_light/golden_run/p3 \
    --p4-dir results/first_light/golden_run/p4 \
    --evidence-pack-dir results/first_light/evidence_pack_first_light
```

**Output:** `first_light_status.json` in the evidence pack directory.

**Example JSON:**

```json
{
  "schema_version": "1.2.0",
  "timestamp": "2025-12-11T12:00:00Z",
  "mode": "SHADOW",
  "pipeline": "local",
  "telemetry_source": "mock",
  "proof_snapshot_present": false,
  "shadow_mode_ok": true,
  "determinism_ok": true,
  "p3_harness_ok": true,
  "p4_harness_ok": true,
  "evidence_pack_ok": true,
  "schemas_ok": true,
  "metrics_snapshot": {
    "p3_success_rate": 0.852,
    "p3_omega_occupancy": 0.851,
    "p4_success_rate": 0.927,
    "p4_divergence_rate": 0.972,
    "p4_twin_accuracy": 0.886
  },
  "warnings": [
    "p3_omega_occupancy below 90% threshold"
  ]
}
```

When `proof_snapshot_present=true`, the status may also include `proof_snapshot_integrity`, an advisory object `{ ok, failure_codes, notes, extraction_source }` set by `scripts/verify_proof_snapshot_integrity.py`. `ok=true` means the snapshot’s file hash matches the manifest `sha256`, the snapshot’s `canonical_hash` matches a recomputation over `proof_hashes`, and `entry_count` equals the number of hashes. `failure_codes` uses canonical values (`MISSING_FILE`, `SHA256_MISMATCH`, `CANONICAL_HASH_MISMATCH`, `ENTRY_COUNT_MISMATCH`). `extraction_source` is `STATUS_CHECK` when computed by `generate_first_light_status.py`, `MANIFEST` when copied from a precomputed manifest field, and `MISSING` when no proof snapshot is present. Any `ok=false` verdict is advisory only in SHADOW mode: it does not invalidate the evidence pack or gate CI, but signals that an external verifier should re-run the integrity checker and inspect `notes` for details.

The script is purely observational - it reads existing artifacts and summarizes status. It does not run harnesses or tests.

### 9.1 Interpreting CONSISTENT / PARTIAL / INCONSISTENT (No Gating)

When `first_light_status.json` includes `signals.policy_drift_vs_nci`, the `consistency_status` field summarizes whether Policy Drift and NCI (Narrative Consistency Index) are reporting comparable severity:

- `CONSISTENT`: Same severity tier (e.g., policy `OK` vs NCI `OK`, policy `WARN` vs NCI `WARN`).
- `PARTIAL`: One-tier mismatch (e.g., policy `WARN` vs NCI `OK`, policy `BLOCK` vs NCI `WARN`).
- `INCONSISTENT`: Two-tier mismatch (e.g., policy `OK` vs NCI `BREACH`).

This is advisory only in SHADOW mode; it should inform reviewer focus, not block execution.

### 9.2 CTRPK Signal Precedence (Manifest > CLI)

When `first_light_status.json` includes `signals.ctrpk` (Curriculum Transition Requests Per 1K Cycles), the CTRPK data may come from two sources:

1. **Manifest CTRPK**: Embedded in the evidence pack manifest under `governance.curriculum.ctrpk`
2. **CLI CTRPK**: Provided via `--ctrpk-json` command-line argument

**Precedence Rule:**

- **Manifest CTRPK takes precedence over CLI CTRPK.** When both are present, the manifest values are used and the `source` field is set to `"manifest"`.
- When only CLI CTRPK is provided, those values are used and `source` is set to `"cli"`.
- When neither is present, no CTRPK signal is included in the status output.

**Mismatch Warning Semantics:**

When both manifest and CLI CTRPK are provided with **different** values (value or status mismatch), a warning is emitted:

```
CTRPK mismatch: CLI (value=X.XX, status=Y) vs manifest (value=Z.ZZ, status=W). Using manifest CTRPK.
```

This warning signals that the CLI-provided CTRPK differs from what was embedded in the evidence pack. External verifiers should investigate why the CLI value differs—this may indicate:

- Stale CLI input from a previous run
- Evidence pack built with different signal parameters
- Intentional override for comparison purposes

**Why Manifest Takes Precedence:**

The manifest CTRPK is **cryptographically bound** to the evidence pack via SHA-256 hash. This ensures:

1. **Determinism**: The same evidence pack always produces the same CTRPK signal
2. **Auditability**: The CTRPK value is verifiable against the manifest hash
3. **Reproducibility**: External verifiers can independently verify the CTRPK source

The CLI argument exists primarily for:
- Testing and development
- Backward compatibility with runs that predate manifest CTRPK embedding
- Explicit override scenarios (with documented mismatch warning)

**SHADOW MODE CONTRACT:**

- CTRPK is purely observational and advisory only
- CTRPK status (OK/WARN/BLOCK) and trend (IMPROVING/STABLE/DEGRADING) do NOT gate any operations
- Mismatch warnings are informational; they do not prevent status generation

**Example with Manifest Precedence:**

```bash
# CLI provides different CTRPK than manifest
uv run python scripts/generate_first_light_status.py \
    --p3-dir results/first_light/golden_run/p3 \
    --p4-dir results/first_light/golden_run/p4 \
    --evidence-pack-dir results/first_light/evidence_pack_first_light \
    --ctrpk-json results/cli_ctrpk.json
```

If `cli_ctrpk.json` contains `{"value": 1.0, "status": "OK"}` but manifest has `{"value": 3.0, "status": "WARN"}`, the output will:
- Use manifest values (value=3.0, status=WARN)
- Set `signals.ctrpk.source = "manifest"`
- Emit warning: `CTRPK mismatch: CLI (value=1.00, status=OK) vs manifest (value=3.00, status=WARN). Using manifest CTRPK.`

---

## 10. Snapshot Runbook Summary (Auto-Resume Rationale)

**PHASE II — NOT USED IN PHASE I**

The snapshot runbook summary provides a machine-readable explanation of why a particular resume choice was made during U2 experiment orchestration. This summary is included in evidence packs under `manifest["operations"]["auto_resume"]` when snapshot planning is enabled.

### 10.1 Understanding the Runbook Summary Fields

When present in an evidence pack manifest, the `auto_resume` section contains:

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | string | Schema version (currently "1.0.0") |
| `runs_analyzed` | integer | Number of experiment runs analyzed for resume decisions |
| `preferred_run_id` | string \| null | Identifier of the selected run for resuming (null if NEW_RUN) |
| `preferred_snapshot_path` | string \| null | Path to the selected snapshot file (null if NEW_RUN) |
| `mean_coverage_pct` | float | Average snapshot coverage percentage across all analyzed runs |
| `max_gap` | integer | Largest gap (in cycles) between consecutive snapshots across all runs |
| `reason` | string | Human-readable explanation of the resume decision |
| `status` | string | Decision status: "RESUME", "NEW_RUN", or "NO_ACTION" |
| `has_resume_targets` | boolean | Whether any viable resume targets were found |

**Example Runbook Summary:**

```json
{
  "schema_version": "1.0.0",
  "runs_analyzed": 3,
  "preferred_run_id": "run_20250101_120000",
  "preferred_snapshot_path": "results/u2_snapshots/run_20250101_120000/snapshots/snapshot_0050.json",
  "mean_coverage_pct": 25.5,
  "max_gap": 15,
  "reason": "Selected run 'run_20250101_120000' for resume: coverage 3.0% (priority score 14.20), mean coverage across all runs 25.5%, max gap 15 cycles",
  "status": "RESUME",
  "has_resume_targets": true
}
```

### 10.2 Interpreting RESUME vs NEW_RUN in SHADOW Mode

**SHADOW MODE CONTRACT:**
- Snapshot planning is **purely observational** and **advisory only**
- No governance decisions are made or modified based on snapshot status
- All snapshot planning operates in SHADOW MODE (no run blocking)

**RESUME Status:**
- Indicates that viable resume targets were found and a preferred snapshot was selected
- The `preferred_run_id` and `preferred_snapshot_path` identify the chosen resume point
- The `reason` field explains why this particular run was selected (typically based on coverage deficit and priority scoring)
- **In SHADOW MODE:** This is a recommendation only; no actual resume occurs unless explicitly triggered by operator

**NEW_RUN Status:**
- Indicates that no viable resume targets were found, or analysis failed
- `preferred_run_id` and `preferred_snapshot_path` will be `null`
- The `reason` field explains why NEW_RUN was chosen:
  - "No runs found in snapshot root" — snapshot root was empty
  - "No viable resume targets found among N runs" — runs exist but none are resumable (e.g., all corrupted, all have very low coverage)
  - "Error during auto-resume analysis" — analysis failed (logged but non-blocking)
- **In SHADOW MODE:** This is a recommendation only; the system will start a new run

**NO_ACTION Status:**
- Rare status indicating analysis completed but no clear recommendation
- Typically indicates edge cases or ambiguous planning state

### 10.3 Verification Checklist for Snapshot Runbook

When examining an evidence pack that includes `operations.auto_resume`:

- [ ] `schema_version` is present and matches expected version ("1.0.0")
- [ ] `runs_analyzed` is a non-negative integer
- [ ] `status` is one of: "RESUME", "NEW_RUN", "NO_ACTION"
- [ ] If `status == "RESUME"`: `preferred_run_id` and `preferred_snapshot_path` are non-null
- [ ] If `status == "NEW_RUN"`: `preferred_run_id` and `preferred_snapshot_path` are null
- [ ] `mean_coverage_pct` is between 0.0 and 100.0
- [ ] `max_gap` is a non-negative integer
- [ ] `reason` is a non-empty string explaining the decision
- [ ] `has_resume_targets` is a boolean consistent with `status` (true for RESUME, false for NEW_RUN)

**External Verifier Guidance:**

1. **For RESUME decisions:** Verify that the `preferred_snapshot_path` exists and is valid (if you have access to the snapshot root). The `reason` should explain why this run was prioritized (e.g., lowest coverage, largest gap).

2. **For NEW_RUN decisions:** Verify that the `reason` provides a clear explanation. If `runs_analyzed == 0`, this is expected for a fresh snapshot root. If `runs_analyzed > 0` but `has_resume_targets == false`, investigate why no runs were resumable (corruption, very low coverage, etc.).

3. **Coverage interpretation:** `mean_coverage_pct` indicates the average snapshot coverage across all analyzed runs. Low coverage (< 10%) suggests sparse checkpointing; high coverage (> 70%) suggests good checkpoint discipline.

4. **Gap interpretation:** `max_gap` indicates the largest gap between consecutive snapshots. Large gaps (> 50% of total cycles) suggest potential data loss risk if a failure occurs mid-gap.

### 10.4 Using the Snapshot Inspector Tool

The `scripts/u2_snapshot_inspect.py` tool allows operators to inspect snapshot planning decisions locally:

```bash
# Human-readable summary
uv run python scripts/u2_snapshot_inspect.py --snapshot-root results/u2_snapshots

# JSON output for scripting
uv run python scripts/u2_snapshot_inspect.py --snapshot-root results/u2_snapshots --json
```

**Example: Sanity-Checking Auto-Resume Before P5 Deployment**

Before trusting auto-resume in a P5 (production) setting, a human operator should verify:

1. **Run the inspector:**
   ```bash
   uv run python scripts/u2_snapshot_inspect.py --snapshot-root results/u2_snapshots --json > snapshot_plan.json
   ```

2. **Extract the runbook summary:**
   ```python
   import json
   with open("snapshot_plan.json") as f:
       data = json.load(f)
   
   # Get runbook summary (if available in full output)
   from experiments.u2.snapshot_history import (
       build_multi_run_snapshot_history,
       plan_future_runs,
       build_snapshot_runbook_summary,
   )
   
   # Rebuild runbook from inspector output
   multi_history = data["multi_history"]
   plan = data["plan"]
   runbook = build_snapshot_runbook_summary(multi_history, plan)
   
   print(f"Status: {runbook['status']}")
   print(f"Reason: {runbook['reason']}")
   print(f"Mean Coverage: {runbook['mean_coverage_pct']:.1f}%")
   print(f"Max Gap: {runbook['max_gap']} cycles")
   ```

3. **Verify decision makes sense:**
   - If `status == "RESUME"`: Check that `preferred_snapshot_path` exists and is recent
   - If `status == "NEW_RUN"`: Verify that `reason` explains why (empty root, no viable targets, etc.)
   - Check that `mean_coverage_pct` is reasonable for your checkpointing policy
   - Check that `max_gap` is acceptable (not too large relative to total cycles)

4. **Cross-reference with evidence pack:**
   - Compare the runbook summary in the evidence pack manifest with local inspection
   - Verify that `runs_analyzed` matches the number of run directories in snapshot root
   - Verify that `reason` is consistent with observed snapshot state

**Key Principle:** In SHADOW MODE, snapshot planning is advisory only. The runbook summary explains what the planner would do, but does not enforce any behavior. Before trusting auto-resume in P5, operators should verify that the planning logic produces sensible recommendations.

### 10.5 Calibration Experiment Runbooks (CAL-EXP-1, 2, 3)

For calibration experiments (CAL-EXP-1 warm-start, CAL-EXP-2 long-window convergence, CAL-EXP-3 regime-change resilience), the snapshot inspector can generate experiment-specific runbooks:

```bash
# CAL-EXP-1: Warm-start stability analysis
uv run python scripts/u2_snapshot_inspect.py \
    --snapshot-root results/cal_exp1_snapshots \
    --calibration-experiment CAL-EXP-1 \
    --json > cal_exp1_runbook.json

# CAL-EXP-2: Long-window convergence analysis
uv run python scripts/u2_snapshot_inspect.py \
    --snapshot-root results/cal_exp2_snapshots \
    --calibration-experiment CAL-EXP-2 \
    --json > cal_exp2_runbook.json

# CAL-EXP-3: Regime-change resilience analysis
uv run python scripts/u2_snapshot_inspect.py \
    --snapshot-root results/cal_exp3_snapshots \
    --calibration-experiment CAL-EXP-3 \
    --json > cal_exp3_runbook.json
```

**Experiment-Specific Fields:**

- **CAL-EXP-1** includes `stability_indicators` with `coverage_consistency` and `gap_risk` assessments
- **CAL-EXP-2** includes `convergence_indicators` with `coverage_trend` and `gap_stability` assessments
- **CAL-EXP-3** includes `resilience_indicators` with `coverage_robustness` and `gap_tolerance` assessments

### 10.6 Multi-Run Comparison Analysis

The snapshot inspector supports multi-run comparison to detect stability deltas, max_gap issues, and coverage regression:

```bash
# Compare current state with previous history
uv run python scripts/u2_snapshot_inspect.py \
    --snapshot-root results/u2_snapshots \
    --compare \
    --previous-history previous_multi_history.json \
    --json > comparison_analysis.json
```

**Comparison Analysis Fields:**

- **stability_deltas**: Coverage mean/std, gap mean, status distribution, coverage stability assessment
- **max_gap_analysis**: Global max gap, list of runs with problematic gaps (>50% of total cycles), risk levels
- **coverage_regression**: Detection flag, severity (high/medium/low), mean degradation percentage, previous vs. current coverage

**Interpretation:**

- **Stability deltas**: Low std (<5%) indicates stable checkpointing; high std (>15%) indicates inconsistent checkpointing
- **Problematic gaps**: Gaps >50% of total cycles indicate high data loss risk if failure occurs mid-gap
- **Coverage regression**: Degradation >20% indicates significant checkpointing discipline regression

### 10.7 P5 Acceptance Gate 4 (Operational) Support

Snapshot runbook summaries provide evidence for **P5 Acceptance Gate 4 (Operational)** criteria:

| Gate Criterion | Snapshot Evidence | Interpretation |
|----------------|-------------------|----------------|
| **P5-OPS-001**: Shadow mode percentage | `status` field shows SHADOW-only decisions | All snapshot planning operates in SHADOW MODE (advisory only) |
| **P5-OPS-002**: Governance interventions max | `has_resume_targets` and `status` show no enforcement | Snapshot planning never enforces governance; all decisions are advisory |
| **P5-OPS-003**: Schema validation pass | `schema_version` present and valid | Runbook schema is versioned and validated |
| **P5-OPS-004**: Human review completed | `reason` field provides human-readable explanation | Runbook provides auditable trail for human review |

**Operational Readiness Assessment:**

The snapshot runbook summary demonstrates operational readiness by:

1. **Transparency**: The `reason` field explains every resume decision in human-readable terms
2. **Auditability**: All metrics (`mean_coverage_pct`, `max_gap`, `runs_analyzed`) are included for verification
3. **Safety**: SHADOW MODE contract ensures no governance enforcement; all decisions are advisory
4. **Continuity**: `preferred_snapshot_path` and `preferred_run_id` provide clear resume targets for operational continuity

**For External Verifiers:**

When evaluating P5 Gate 4 (Operational), verify:

- [ ] Snapshot runbook summary is present in evidence pack manifest (`operations.auto_resume`)
- [ ] `status` field indicates SHADOW MODE operation (no enforcement)
- [ ] `reason` field provides clear, neutral explanation of resume decisions
- [ ] `mean_coverage_pct` and `max_gap` are within acceptable operational ranges
- [ ] Multi-run comparison (if available) shows stable checkpointing discipline
- [ ] No coverage regression detected (if comparison history provided)

**Key Principle:** Snapshot planning supports operational readiness by providing transparent, auditable, and safe resume decision-making. All planning operates in SHADOW MODE, ensuring no governance enforcement until explicitly authorized by human operators.

---

## 11. Phase X P5 Update Path (Real Telemetry)

This section outlines what changes when real telemetry is integrated in Phase X P5.

### 11.1 What Numbers Will Change

| Metric | Current (Mock) | Expected Change |
|--------|----------------|-----------------|
| P4 divergence_rate | 97.2% | **Decrease significantly** — twin should track real runner |
| P4 twin_success_accuracy | 88.6% | **Increase** — meaningful correlation expected |
| P4 max_divergence_streak | 165 | **Decrease** — fewer consecutive divergences |
| P3 omega_occupancy | 85.1% | **May change** — depends on real stability patterns |

### 11.2 What Procedures Stay the Same

These verification procedures are unchanged with real telemetry:

1. **P3 Harness invocation:** Same command, same flags
   ```bash
   uv run python scripts/usla_first_light_harness.py --cycles 1000 --seed 42 ...
   ```

2. **P4 Harness invocation:** Same command, different telemetry adapter (internal change)
   ```bash
   uv run python scripts/usla_first_light_p4_harness.py --cycles 1000 --seed 42 ...
   ```

3. **Evidence pack build:** Same script, same structure
   ```bash
   uv run python scripts/build_first_light_evidence_pack.py ...
   ```

4. **Integrity verification:** Same script
   ```bash
   uv run python scripts/verify_evidence_pack_integrity.py
   ```

5. **SHADOW MODE compliance:** Same 14 tests
   ```bash
   uv run python -m pytest tests/integration/test_shadow_mode_compliance.py -v
   ```

6. **Determinism check:** Same script (may need timing tolerance for I/O latency)
   ```bash
   uv run python scripts/verify_first_light_determinism.py
   ```

### 11.3 New Caveats to Expect

1. **Real-runner availability:** Verification requires the real runner infrastructure to be operational. Mock mode remains available as fallback.

2. **I/O latency:** Real telemetry introduces network/IPC latency. Cycle timing may vary more than with mock data.

3. **Non-determinism sources:** Real runners may have non-deterministic elements (timing, external state). Determinism checks may need to account for this.

4. **Data volume:** Real telemetry may produce larger JSONL files with richer observations.

5. **Threshold calibration:** Omega occupancy and other thresholds may need adjustment based on real behavior patterns.

### 11.4 Runtime Profile Snapshot for P5 Calibration Drift Detection

The runtime profile snapshot (see Section 9.5) provides an additional signal for P5 calibration drift detection. When comparing evidence packs across calibration runs:

1. **Profile Stability Tracking**: Monitor `profile_stability` across runs. A significant drop (e.g., from 0.95 to 0.80) may indicate:
   - New feature flags introduced that violate profile constraints
   - Profile constraints tightened (intentional or accidental)
   - Runtime environment changes affecting flag evaluation

2. **NO_RUN Rate Trends**: Track `no_run_rate` over time. An increasing trend suggests:
   - Profile constraints becoming more restrictive
   - New forbidden flag combinations emerging
   - Environment context changes (e.g., dev → ci → prod migration)

3. **Status Light Transitions**: Monitor status light changes (GREEN → YELLOW → RED):
   - **GREEN → YELLOW**: Early warning of profile constraint drift
   - **YELLOW → RED**: Significant drift requiring profile review
   - **RED → GREEN**: Profile constraints relaxed or issues resolved

**P5 Calibration Use Case**: When calibrating noise models across multiple P5 runs, compare runtime profile snapshots to ensure:
- Profile constraints remain consistent across calibration runs
- No unexpected flag configuration changes occurred
- Runtime environment stability (same profile, same constraints)

**Example Drift Detection**:
```json
{
  "baseline_run": {
    "profile": "prod-hardened",
    "profile_stability": 0.95,
    "no_run_rate": 0.05,
    "status_light": "GREEN"
  },
  "current_run": {
    "profile": "prod-hardened",
    "profile_stability": 0.82,
    "no_run_rate": 0.12,
    "status_light": "YELLOW"
  },
  "drift_detected": true,
  "advisory": "Profile stability dropped 13 percentage points; review flag configuration changes"
}
```

**SHADOW MODE**: Runtime profile drift detection is advisory only. It does not block calibration runs or invalidate evidence packs. It provides visibility into runtime configuration stability for calibration analysis.

### 11.5 Runtime Profile Calibration Correlation (Cross-Window Analysis)

The runtime profile calibration annex (`evidence["governance"]["runtime_profile_calibration"]`) provides cross-window correlation analysis between runtime profile metrics and CAL-EXP-1 window metrics. This annex correlates `profile_stability` and `no_run_rate` (from the runtime profile snapshot) with per-window `divergence_rate` and `mean_delta_p` metrics. Windows are annotated with `runtime_profile_confounded: bool` flags when `status_light=RED` and either a divergence spike (divergence_rate >= 0.8) or high delta_p (|mean_delta_p| >= 0.05) occurs in the same window. The annex includes windowed correlations (Pearson correlation coefficients rounded to 6 decimals), correlation reasons (explaining why correlations may be `None`: "INSUFFICIENT_POINTS", "ZERO_VARIANCE_X", "ZERO_VARIANCE_Y", "NON_NUMERIC_INPUT"), confounding summary (count of confounded windows, their indices, and confound reasons), and annotated windows with confounding flags. **Correlation Interpretation**: When `correlation=None` with a reason in `correlation_reasons`, this indicates a statistical degeneracy (e.g., constant profile metrics result in zero variance, making correlation undefined). Correlation is a diagnostic hint for potential relationships, not causal proof. **Example**: If `profile_stability=0.95` (constant) is correlated with varying `divergence_rate` values, the correlation will be `None` with reason "ZERO_VARIANCE_X" because constant X has zero variance. **SHADOW MODE**: All correlation analysis is advisory only and does not gate calibration results or invalidate evidence packs. Missing runtime profile snapshot results in graceful no-op (annex is not generated). The annex is deterministic and JSON-serializable for reproducible analysis.

### 11.6 Noise vs Reality: p5_source Provenance Enum

The `signals.noise_vs_reality.p5_source` field in `first_light_status.json` uses a frozen enum to indicate the provenance of P5 divergence data used in noise-vs-reality comparison. This field is critical for understanding how the P3 synthetic noise model coverage was validated against real telemetry divergence.

**p5_source Values (Frozen Enum):**

| Value | Meaning |
|-------|---------|
| `p5_real_validated` | P5 data from `p5_divergence_real.json` with RTTS validation status `VALIDATED_REAL` — highest confidence real telemetry |
| `p5_suspected_mock` | P5 data from `p5_divergence_real.json` with RTTS validation status `SUSPECTED_MOCK` — telemetry failed mock detection criteria |
| `p5_real_adapter` | P5 data from `p5_divergence_real.json` with unvalidated status — real adapter without RTTS validation |
| `p5_jsonl_fallback` | Fallback to raw `divergence_log.jsonl` — no structured P5 report available |

**Coercion Behavior:** Unknown or missing `p5_source` values are coerced to `p5_jsonl_fallback` with an advisory note in `p5_source_advisory`. When `p5_source_advisory` is non-null, it indicates the original value was coerced and explains why.

**SHADOW MODE:** The `p5_source` field is purely observational provenance tracking. It does not gate any operations or modify governance decisions.

### 11.7 Noise vs Reality: extraction_source Enum

The `signals.noise_vs_reality.extraction_source` field indicates where the noise_vs_reality signal was extracted from:

| Value | Meaning |
|-------|---------|
| `MANIFEST` | Extracted from evidence pack manifest (canonical source) |
| `EVIDENCE_JSON` | Fallback extraction from evidence.json file |
| `MISSING` | No noise_vs_reality data available |

**SHADOW MODE:** The `extraction_source` field is purely observational provenance tracking.

### 11.8 How to Read top_factor (coverage_ratio vs exceedance_rate)

The `top_factor` field in the noise_vs_reality signal identifies which metric is the primary driver when coverage is MARGINAL or INSUFFICIENT. This helps reviewers quickly identify whether the issue is:

1. **coverage_ratio** — P3 synthetic noise coverage is insufficient relative to P5 divergence
2. **exceedance_rate** — P5 divergence exceeds P3 noise predictions too frequently

**Reading top_factor:**

| top_factor | top_factor_value | Interpretation |
|------------|------------------|----------------|
| `coverage_ratio` | 0.75 | P3 noise model explains 75% of observed divergence. Gap = 25%. |
| `exceedance_rate` | 0.15 | In 15% of cycles, P5 divergence exceeded P3 noise predictions. |

**Example 1: Low Coverage**
```json
{
  "verdict": "MARGINAL",
  "top_factor": "coverage_ratio",
  "top_factor_value": 0.65,
  "advisory_warning": "MARGINAL: coverage_ratio=0.65 [jsonl]"
}
```
*Interpretation:* P3 noise model covers only 65% of real divergence. Consider expanding P3 noise scenarios (e.g., add correlated failures, degradation modes).

**Example 2: High Exceedance**
```json
{
  "verdict": "MARGINAL",
  "top_factor": "exceedance_rate",
  "top_factor_value": 0.12,
  "advisory_warning": "MARGINAL: exceedance_rate=12.0% [real]"
}
```
*Interpretation:* In 12% of cycles, real telemetry showed divergence that P3 noise model did not predict. May indicate unmodeled pathologies or structural drift.

**Advisory Warning Format:**
The `advisory_warning` field follows a stable single-line format:
```
VERDICT: factor=value [source_abbrev]
```
Where `source_abbrev` is: `real` (p5_real_validated), `mock?` (p5_suspected_mock), `adapter` (p5_real_adapter), `jsonl` (p5_jsonl_fallback).

### 11.9 Golden Bundle Entry: noise_vs_reality Signal

Copy-pasteable reference for `signals.noise_vs_reality` in `first_light_status.json`:

```json
{
  "noise_vs_reality": {
    "extraction_source": "MANIFEST",
    "verdict": "MARGINAL",
    "advisory_severity": "WARN",
    "coverage_ratio": 0.72,
    "p3_noise_rate": 0.08,
    "p5_divergence_rate": 0.11,
    "p5_source": "p5_real_validated",
    "p5_source_advisory": null,
    "summary_sha256": "a1b2c3d4e5f6...",
    "top_factor": "coverage_ratio",
    "top_factor_value": 0.72
  }
}
```

**Single-Line Warning Format:**
```
Noise vs reality: VERDICT: factor=value [source_abbrev]
```
Example: `Noise vs reality: MARGINAL: coverage_ratio=0.72 [real]`

### 11.10 Golden Bundle Entry: nci_p5 Signal

Copy-pasteable reference for `signals.nci_p5` in `first_light_status.json`:

```json
{
  "nci_p5": {
    "extraction_source": "MANIFEST_SIGNAL",
    "path": "nci_p5_signal.json",
    "sha256": "a1b2c3d4e5f6...",
    "detection_path": "root",
    "mode": "DOC_ONLY",
    "global_nci": 0.85,
    "confidence": 0.75,
    "slo_status": "OK",
    "recommendation": "NONE",
    "tcl_aligned": true,
    "sic_aligned": true,
    "shadow_mode": true,
    "drivers": [],
    "result_path": null,
    "result_sha256": null
  }
}
```

**Field Reference:**

| Field | Type | Description |
|-------|------|-------------|
| `extraction_source` | enum | `MANIFEST_SIGNAL` \| `MANIFEST_RESULT` \| `EVIDENCE_JSON` \| `MISSING` |
| `path` | string | Relative path to primary artifact (signal or result file) |
| `sha256` | string | SHA-256 hash of primary artifact |
| `detection_path` | enum | `root` \| `calibration` — where artifact was found |
| `mode` | string | NCI analysis mode: `DOC_ONLY`, `TELEMETRY_CHECKED`, etc. |
| `global_nci` | float | Global Narrative Consistency Index (0.0–1.0) |
| `confidence` | float | Confidence score (0.0–1.0) |
| `slo_status` | enum | `OK` \| `WARN` \| `BREACH` |
| `recommendation` | string | `NONE`, `REVIEW`, `WARNING`, etc. |
| `tcl_aligned` | bool | Telemetry Consistency Layer alignment status |
| `sic_aligned` | bool | Schema Integrity Check alignment status |
| `shadow_mode` | bool | Always `true` in shadow mode (non-gating) |
| `drivers` | array | Reason codes (max 3): see below |
| `result_path` | string? | Cross-reference to result file if both exist |
| `result_sha256` | string? | SHA-256 of result file if present |

**Driver Codes:**

| Code | Trigger Condition |
|------|-------------------|
| `DRIVER_SLO_BREACH` | `slo_status == "BREACH"` |
| `DRIVER_RECOMMENDATION_NON_NONE` | `recommendation != "NONE"` |
| `DRIVER_CONFIDENCE_LOW` | `confidence < 0.5` |

Maximum 3 drivers emitted per signal.

**Warning Emission Rule:**

A single advisory warning is emitted if and only if:
```
slo_status == "BREACH"  OR  recommendation != "NONE"
```

Truth table:

| slo_status | recommendation | Warning? |
|------------|----------------|----------|
| OK | NONE | ❌ No |
| OK | REVIEW | ✅ Yes |
| BREACH | NONE | ✅ Yes |
| BREACH | REVIEW | ✅ Yes |

**Sample Warning Line:**
```
NCI BREACH: 72% consistency (confidence 65%)
```

**SHADOW MODE:** All NCI P5 signals are observational only. Detection does not gate pack generation. Status signals are advisory only.

---

**Document generated:** 2025-12-11
**Validated on:** Windows 11, Python 3.11.9, uv 0.4+



## Schema Validation Report (SHADOW)

- Run schema validation in-place (non-gating):
  ```bash
  uv run python scripts/generate_first_light_status.py \
    --p3-dir results/first_light/golden_run/p3 \
    --p4-dir results/first_light/golden_run/p4 \
    --evidence-pack-dir results/first_light/evidence_pack_first_light \
    --validate-schemas [--schema-root <offline-schema-path>]
  ```
- Interpretation:
  - `first_light_status.json` includes `schemas_ok` (True/False/None) and schema warnings.
  - `schemas_ok_summary` adds reviewer-friendly counts (`pass`, `fail`, `missing`), `extraction_source` (`REPORT_FILE` or `MISSING`), a deterministic `top_reason_code` (count desc, then code asc), and a `top_failures` shortlist (max 5) sorted by artifact name. Each failure includes `artifact`, `reason_code`, `schema_path`, and a short `note`.
  - `schemas_validation_report.json` sits next to the status file and lists each artifact (`payload`, `schema`, `status`, `reason_code`, `errors`).
  - Status values: `pass`, `fail`, `missing_schema`, or `missing_payload`. Missing schemas are treated as WARN, not gates.
  - Diagnosing problems:
    - **Missing artifacts** (`missing_*` or `missing` count > 0): check the `payload`/`schema` paths in the report. A `missing_payload` means the harness did not emit the artifact; a `missing_schema` means the schema registry (local `schemas/evidence` or `--schema-root`) is incomplete.
    - **Schema drift** (`fail` count > 0): payload exists but violates its schema. Inspect `errors` for the failing artifact, then diff the payload against the referenced schema to decide whether to regenerate artifacts or update schemas in lockstep.
  - Acting on `reason_code` (recommended auditor action):
    - `MISSING_PAYLOAD` → Re-run harness; confirm the referenced `payload` exists in the run directory.
    - `MISSING_SCHEMA` → Sync schema registry; ensure the referenced `schema_path` exists (or supply `--schema-root`).
    - `SCHEMA_ROOT_NOT_FOUND` → Fix `--schema-root` or restore local `schemas/evidence`.
    - `SCHEMA_VALIDATION_FAILED` → Treat as drift; use report `errors` to pinpoint violations, then update payload generators and schemas together.
- External auditors can copy schemas to an airgapped registry and point the checker at it via `--schema-root`.
