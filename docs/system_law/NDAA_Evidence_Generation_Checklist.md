# NDAA FY26 AI Governance: Evidence Generation Checklist

**Document ID:** ML-COMP-NDAA-FY26-V1.1-CHECKLIST
**Date:** 2025-12-10
**Classification:** INTERNAL / AUDITOR-FACING
**Status:** BINDING
**Precondition:** `ndaa_compliance_spec_validated.md`

---

## 1. Introduction & Scope

It is critical for auditors to understand the behavior of the automated NDAA validation process. A "failed" NDAA validation (i.e., the validation script exits with a non-zero code) signifies that one or more required evidentiary artifacts are either missing, corrupted, or fail their automated verification checks. This *does not* block the continuous integration (CI) pipeline; this non-gating design ensures that development velocity is maintained while providing continuous compliance feedback. This architectural choice balances the imperative for continuous regulatory compliance auditing with the critical need to maintain rapid development velocity, preventing compliance checks from impeding iterative progress. Reviewers should interpret the detailed `ndaa_validation_report.json` artifact (uploaded in CI) to pinpoint specific failures, identify root causes, and prioritize remediation, distinguishing between critical compliance gaps and minor discrepancies.

This document provides a concrete, step-by-step checklist for generating the evidentiary artifacts required to support the claims made in `ndaa_compliance_spec_validated.md`. It maps each NDAA pillar to specific scripts, commands, expected outputs, and verification procedures.

This checklist is designed for both internal validation and to provide a transparent, reproducible workflow for a third-party auditor. Adherence to this process ensures that the generated evidence is authentic, verifiable, and directly traceable to the system's governance laws.

```text
INVARIANT: NDAA validation MUST NEVER gate CI in SHADOW mode.
```
*Enforcement Trace:* This invariant is enforced by configuration in `.github/workflows/ndaa_evidence_validation.yml`, which sets `continue-on-error: true` for the validation step and `if: always()` for artifact upload.

---

## 2. Minimal Auditor Workflow

An auditor can verify the entire compliance claim by following these four steps. This workflow is designed to be completed with a minimal set of commands, demonstrating the integrated and automated nature of the compliance posture.

### Step 1: Reproduce Evidence Generation

Execute the primary Phase X test harnesses. These scripts run the system in both synthetic (P3) and shadow (P4) modes and generate the full suite of log artifacts.

```bash
# Run the P3 synthetic experiment (wind tunnel test)
# This validates the internal consistency of the governance model.
python ledgerctl.py run-experiment p3 --cycles 1000 --output-dir results/audit_p3

# Run the P4 real-runner shadow experiment (flight test)
# This validates the model's correspondence to real telemetry.
python ledgerctl.py run-experiment p4 --cycles 1000 --output-dir results/audit_p4
```

### Step 2: Verify Artifact Integrity

Verify that the generated log files have not been tampered with by checking their hashes against a pre-computed manifest. This confirms the raw evidence is authentic.

```bash
# Generate a manifest of the new artifacts
python verify_config_hashes.py generate --dir results/audit_p3 > manifest_p3_generated.json
python verify_config_hashes.py generate --dir results/audit_p4 > manifest_p4_generated.json

# Compare the generated manifests against the trusted manifests from the evidence package
# (Assuming trusted manifests are provided)
diff manifest_p3_trusted.json manifest_p3_generated.json
diff manifest_p4_trusted.json manifest_p4_generated.json
```

### Step 3: Run Compliance Verification Tests

Execute the automated test suite designed to parse the generated artifacts and verify they meet the system's own integrity and governance laws.

```bash
# Run the high-level integrity audit on the generated results
uv run pytest test_integrity_audit.py --log-dir results/audit_p4

# Run the specific test for validating P4 divergence logs against the spec
uv run pytest test_migration_validation.py --log-file results/audit_p4/p4_divergence_log.jsonl
```

### Step 4: Inspect Final Status Reports

Review the human-readable summary reports. If the previous steps passed, these JSON files represent the final, validated state of the system's compliance.

```bash
# Inspect the P3 Stability Report for internal consistency pass/fail
cat results/audit_p3/first_light_stability_report.json | jq .criteria_evaluation

# Inspect the P4 Calibration Report for model-reality correspondence
cat results/audit_p4/p4_calibration_report.json | jq .calibration_assessment
```

---

## 3. Pillar-by-Pillar Evidence Generation

### Pillar 1: Risk-Informed Strategy

**Claim:** Risks are formally identified, measured, and mitigated through proactive monitoring and signal fusion.

| Step | Action | Command / Procedure | Expected Output | Verification |
| :--- | :--- | :--- | :--- | :--- |
| **1.1** | **Generate Red-Flag Logs** | Run the P3 synthetic harness which simulates various failure modes. | `results/audit_p3/first_light_red_flag_matrix.json` | The JSON file must validate against its schema. The `summary.total_flags` should be greater than zero in a pathological run. |
| **1.2** | **Generate Fused Safety Signal** | Execute the replay safety test suite, which invokes the signal fusion logic. | Log output from `test_replay_safety_governance_signal.py` showing `[CONFLICT]` and `[DIVERGENT]` states. | `uv run pytest tests/test_replay_safety_governance_signal.py` must pass. |
| **1.3** | **Generate Stability Report** | The P3 run automatically generates a final report summarizing risk metrics. | `results/audit_p3/first_light_stability_report.json` | The file must validate against its schema. The `.criteria_evaluation.all_passed` field determines the final outcome. |

### Pillar 2: Technical Controls

**Claim:** Verifiable mechanisms in the architecture enforce system laws, such as shadow mode and configuration integrity.

| Step | Action | Command / Procedure | Expected Output | Verification |
| :--- | :--- | :--- | :--- | :--- |
| **2.1** | **Generate Divergence Logs**| Run the P4 shadow harness to compare the real runner vs. the twin. | `results/audit_p4/p4_divergence_log.jsonl` | The file must conform to `p4_divergence_log.schema.json`. Each entry must contain `action: "LOGGED_ONLY"`. |
| **2.2** | **Verify Config Integrity** | Run the slice identity drift guard tests. | Test logs showing detection of `PARAMETRIC` and `SEMANTIC` drift. | `uv run pytest curriculum/test_slice_drift_guard.py` must pass. |
| **2.3** | **Verify Shadow Invariant**| Run the P4 invariant test suite, which includes checks that try to violate shadow mode. | All tests passing, confirming that architectural barriers prevent observation from influencing control. | `uv run pytest tests/first_light/test_p4_invariants.py` must pass. |

### Pillar 3: Human Override Capability

**Claim:** The system is architecturally incapable of autonomous action and is designed to present clear data to human operators.

| Step | Action | Command / Procedure | Expected Output | Verification |
| :--- | :--- | :--- | :--- | :--- |
| **3.1** | **Confirm Non-Enforcement** | Inspect the P3 Red Flag Matrix generated in step 1.1. | The `hypothetical_abort.would_abort` field may be `true`, but no actual abort occurs. | Manually inspect `first_light_red_flag_matrix.json` and confirm the run completed all cycles. |
| **3.2** | **Generate Operator Tile** | Run the test that builds the console tile for slice identity. | A valid JSON object conforming to `slice_identity_console_tile.schema.json`. | `uv run pytest tests/slice_identity/test_console_tile.py` must pass. |
| **3.3** | **Generate Divergence Report**| The P4 run from Step 1 generates a final calibration report. | `results/audit_p4/p4_calibration_report.json` | The `.calibration_assessment.recommendations` field provides clear, human-readable steps for an operator. |

### Pillar 4: Auditability

**Claim:** All system events are logged to structured, immutable, and verifiable artifacts from a stable, known baseline.

| Step | Action | Command / Procedure | Expected Output | Verification |
| :--- | :--- | :--- | :--- | :--- |
| **4.1** | **Generate Full Log Suite** | The P3/P4 runs from Step 1 generate all necessary raw logs. | The `results/audit_p3` and `results/audit_p4` directories populated with JSON and JSONL files. | All generated files must validate against their corresponding schemas in `docs/system_law/schemas/`. |
| **4.2** | **Verify Log Hashes**| Use the `verify_config_hashes.py` script to confirm immutability. | Command returns exit code 0, indicating all hashes match the manifest. | `python verify_config_hashes.py check --manifest manifest_p4_trusted.json` |
| **4.3** | **Verify Baseline Integrity**| Run the test that validates the Slice Identity invariants. | All tests passing, confirming `SI-001` through `SI-006` are enforced. | `uv run pytest tests/slice_identity/test_invariants.py` must pass. |

---

## 4. Interpretation Rules: Mock (P3) vs. Real (P4) Telemetry

The evidence package contains artifacts from two distinct phases. Correctly interpreting them is crucial for a fair audit.

-   **Phase P3 (Synthetic Telemetry): The "Wind Tunnel"**
    -   **Purpose:** To validate the *internal consistency and logical correctness* of the governance model itself.
    -   **Interpretation:** P3 artifacts (e.g., `first_light_stability_report.json`) prove that the system behaves as designed in a perfectly controlled, synthetic environment. It answers the question: "Does the governance logic work correctly on paper?"
    -   **Example:** A successful P3 run shows that the red-flag detection system correctly identifies a simulated `RSI_COLLAPSE` event.

-   **Phase P4 (Real Telemetry Shadow): The "Flight Test"**
    -   **Purpose:** To validate the *correspondence of the governance model to reality*.
    -   **Interpretation:** P4 artifacts (e.g., `p4_divergence_log.jsonl`) prove that the model's predictions accurately reflect the behavior of the live, operational system. It answers the question: "Does the governance model accurately describe the real world?"
    -   **Example:** A successful P4 run shows low divergence between the twin's predictions and the real runner's telemetry, meaning the model is well-calibrated.

An auditor should first confirm the model is logically sound via P3 evidence, then confirm it is empirically valid via P4 evidence.
