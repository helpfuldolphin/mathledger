# Claim Promotion Register

Status: ACTIVE  
Mode: ACE claim promotion registry for clean-room `mathledger`

## Purpose

This register is the canonical source for wave-scoped claims.
Each wave must define:
- one narrow falsifiable claim
- explicit tier classification
- closure artifacts (tests, assertions, or hostile checks)

No claim may be treated as promoted without a closure artifact.

## Tier Meanings

- Tier A (Enforced): violation is mechanically unavoidable to detect.
- Tier B (Visible): violation is replay-visible but not blocked.
- Tier C (Documented): deferred or aspirational only.

## Wave Register

| Wave | Claim ID | Status | Tier | Narrow Claim | Falsifier | Required Closure Artifacts |
| --- | --- | --- | --- | --- | --- | --- |
| A0 | ML-A0-BOOTSTRAP-DETERMINISTIC | Locally Validated | A | Repository test harness and package layout are deterministic and runnable from a clean checkout. | `pytest` fails from clean checkout, or imports rely on ambient path hacks. | `pyproject.toml`, package layout, deterministic smoke test run. |
| A1 | ML-A1-BASIS-GENOME | Locally Validated | A | Basis primitives (normalization, hashing, Merkle, block sealing, dual attestation, curriculum I/O) are deterministic and pure. | Same inputs produce divergent outputs or read ambient state. | Wave A1 test suite passing in `tests/test_wave_a1_basis_core.py`. |
| A2 | ML-A2-GOV-KERNEL | Locally Validated | A | Trust class routing and abstention preservation are structurally enforced. | ADV enters authority path or ABSTAINED can be dropped/null. | Route gate tests, abstention gate tests, fail-closed checks. |
| A3 | ML-A3-UVIL-BOUNDARY | Locally Validated | A | Exploration IDs never enter committed identity paths. | `proposal_id` appears in committed hash identity. | Commit/verification boundary tests. |
| A4 | ML-A4-EVIDENCE-REPLAY | Locally Validated | A | U_t, R_t, H_t replay re-compute exactly from evidence artifacts. | Replay mismatch without mutation. | Replay verifier tests, tamper-negative tests. |
| A5 | ML-A5-AAK-BRIDGE | Locally Validated | A | Lane B evidence references can link to Lane A hashes without authority escalation. | Lane B payload modifies authority status. | Contract tests against AAK context contract. |
| A6 | ML-A6-CPF-ADAPTER | Locally Validated | B | CPF outputs normalize into deterministic context snapshots for experimentation. | Same CPF payload yields divergent normalized output. | Schema adapter tests and canonical serialization checks. |
| A7 | ML-A7-EXPERIMENT-RUNNER | Locally Validated | B | Three-lane runner executes reproducible pipeline with explicit lane boundaries. | Runner mixes authority and advisory artifacts silently. | End-to-end reproducibility test and boundary assertions. |
| A7R | ML-A7R-SCENARIO-SUITE | Locally Validated | B | Published adversarial and benign scenarios execute through A7 with deterministic gate outcomes and sealed evidence artifacts. | Scenario outputs diverge from expected gates or evidence artifacts are not emitted per step. | `tests/test_wave_a7_scenario_suite.py`, `docs/results/scenario_suite/results.md`, per-step `evidence_pack.json` and bridge packets. |
| A7L | ML-A7L-LIVE-MODEL-LAYER | Locally Validated (Mock) | B | A thin live-model harness captures OpenRouter responses, classifies text into CPF indicators, and routes outputs through A7 governance with replay artifacts. | Live-layer output cannot be routed through A7 deterministically or capture logs/evidence are missing. | `src/mathledger/integration/live_model_layer.py`, `tests/test_wave_a7_live_model_layer.py` |
| A8 | ML-A8-USLA-TDA-SIDECAR | Pending | B | USLA/TDA metrics run as advisory sidecar only and never gate authority in this phase. | Sidecar output influences Lane A verdicts. | Sidecar contract tests and no-authority assertions. |
| A9 | ML-A9-USLA-TDA-PROMOTION | Pending | A | USLA/TDA authority gating is enabled only after hostile replay closure. | Gating enabled without closure artifacts. | Hostile audit vectors, replay closure dossier. |

## Promotion Readiness Checklist

For each wave:
1. claim is one sentence and falsifiable
2. tier assigned before implementation
3. no hidden scope expansion
4. closure artifact exists
5. audit path is executable
6. versioning and tags are coherent

## Local Closure Evidence

- Date: 2026-02-23
- Command: `cd C:\dev\mathledger && pytest -q`
- Result: `51 passed`
