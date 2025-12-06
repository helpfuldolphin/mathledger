# Vibe Audit Report: Serialization & Canonicalization
**Date:** 2025-11-27
**Auditor:** Gemini A (Global Orchestrator)
**Status:** VCP 2.2 Initialization

## Executive Summary
The codebase exhibits significant entropy in JSON serialization, violating the "No hidden randomness" and "RFC 8785" constraints of the VSD. We identified ~195 instances of `json.dumps` with varying parameters (`sort_keys`, `separators`, `ensure_ascii`).

## Critical Violations (Must Fix for First Organism)

### 1. Ledger & Block Serialization
*   **File:** `backend/ledger/ingest.py`
*   **File:** `backend/axiom_engine/derive_worker.py`
*   **Issue:** Uses `json.dumps(..., separators=(",", ":"), sort_keys=True)`. This is an *ad-hoc* canonicalization attempt that differs from strict RFC 8785 (JCS) in edge cases (unicode, float handling).
*   **Remediation:** Replace with `backend.basis.canon.canonical_json_dump`.

### 2. Job Payload Hashing
*   **File:** `backend/axiom_engine/derive.py`
*   **Issue:** Job data is serialized with default `json.dumps` before being put on Redis.
*   **Risk:** Workers might hash the job differently if their python environment differs (e.g. default separators changed in Python versions, though unlikely).
*   **Remediation:** Use `canonical_json_dump` for all job payloads.

### 3. Metrics & Telemetry
*   **File:** `backend/metrics_cartographer.py`
*   **File:** `backend/metrics/fo_feedback.py`
*   **Issue:** Inconsistent use of `sort_keys=True`.
*   **Remediation:** Low priority for consensus, but high priority for "Vibe". Migrate to `basis`.

## Migration Plan (Track A)
1.  **Phase 1 (Substrate):** Fix `backend/ledger` and `backend/axiom_engine` to use `backend.basis.canon`.
2.  **Phase 2 (Tools):** Update `scripts/` and `tools/` to use `basis` where artifacts are generated.
3.  **Phase 3 (Cleanup):** Ban `json.dumps` via linter rules (except in specific UI/display logic).

## Layering Check
*   **Whitepaper:** Requires "Merkle-rooted blocks".
*   **VSD:** Requires "RFC 8785".
*   **Code:** Currently uses Ad-Hoc JSON.
*   **Gap:** **CRITICAL**. The ledger root hash is currently unstable across different JSON implementations. This confirms the necessity of Track A.

