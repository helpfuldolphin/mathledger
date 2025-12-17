# SKELETON_POLICY.md - Substrate Governance Skeleton Baseline

## Purpose

This document outlines the policy and usage of the `SKELETON_BASELINE_v1` for Substrate Governance. This skeleton implementation provides a minimal, non-invasive framework to integrate and test the governance gating logic within higher-level Phase X surfaces without requiring immediate integration with actual Trusted Platform Module (TPM) or Integrity Measurement Architecture (IMA) attestation sources.

Its primary role is to enable early-stage integration, validation of API contracts (e.g., `global_health.json` schema), and the operationalization of CI gating behavior in a "shadow-only" mode.

## Scope

This policy applies to the `scripts/substrate_governance_check.py` script and its associated schema (`docs/governance/substrate/substrate_schema_draft07.json`).

## Key Characteristics & Limitations

1.  **Non-Attestation:** The skeleton *does not* perform any actual hardware attestation, TPM measurements, or filesystem integrity checks. All `identity_stability_index` and `drift_flags` are derived from a local `global_health.json` file provided as an argument.

2.  **Schema Validation:** The script rigorously validates the provided `global_health.json` against the official Draft-07 schema, ensuring data contract adherence.

3.  **Shadow-Only Mode (Phase X Default):**
    *   By default, the `substrate_governance_check.py` script operates in `--shadow-only` mode.
    *   In this mode, regardless of the detected `status_light` (`RED` or `BLOCK`), the script will *always* exit with code `0` (success).
    *   Critical advisories and detailed failure messages (as defined in the Substrate Governance Integration Plan) will be printed to `stderr`.
    *   This mode is crucial for Phase X deployments, allowing the governance mechanism to run in production pipelines, collect telemetry, and provide alerts *without* blocking deployments. This enables monitoring and analysis of potential issues before hard enforcement.

4.  **Hard Enforcement Mode:**
    *   The `--no-shadow-only` flag can be used to disable shadow mode.
    *   In this mode, `RED` status will result in exit code `64`, and `BLOCK` status will result in exit code `65`, thereby failing the CI job or deployment process.
    *   This mode is intended for testing, development, and eventual production hard-gating once the attestation infrastructure is fully mature and trusted.

## Integration Guidelines

*   **CI Workflows:** When integrating into Phase X CI/CD pipelines, ensure the `substrate_governance_check.py` script is invoked with the `--shadow-only` flag (which is the default behavior if neither `--shadow-only` nor `--no-shadow-only` is explicitly provided).
*   **Monitoring:** Pipeline logs should be configured to capture `stderr` output from the governance check script to ensure advisories are visible and actionable.
*   **Transition to Hard Gate:** The transition from shadow-only to hard enforcement (`--no-shadow-only`) in production environments must be a deliberate, risk-assessed decision, following a formal governance review and sign-off process, after sufficient confidence is gained from shadow-mode operation and actual attestation data.

## Verification

A dedicated CI job (`.github/workflows/skeleton_verification.yml`) is in place to verify that the `scripts/substrate_governance_check.py` module can be imported without unexpected side effects, confirming its readiness for integration into larger Python environments.
