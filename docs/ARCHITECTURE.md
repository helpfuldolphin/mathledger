# MathLedger Architecture

**Status:** Active  
**Updated:** 2025-11-26

This document is the **stakeholder-facing narrative** of MathLedger’s architecture. It references the whitepaper, the First Organism, and the RFL framework without duplicating the deeper architectural audit trail housed under `docs/architecture/README.md`. That subdirectory continues to hold the rigorous analysis (duplicate crypto, complexity hotspots, risk scoring) for architects and auditors.

---

## 1. High-Level Perspective

MathLedger is a Reflexive Formal Learning (RFL) system whose control loop is captured in the whitepaper (`docs/whitepaper.md`). The Chain of Verifiable Cognition—`U_t → R_t → H_t → RFL`—is realized by the **First Organism** and validated by `tests/integration/test_first_organism.py`. This integration test, the attestation spec, and the RFL implementation together form the **reference implementation pathway** from theory to deployed artifacts.

Key references:
* **Theory:** `docs/whitepaper.md`
* **Spec:** `docs/ATTESTATION_SPEC.md`, `docs/RFL_IMPLEMENTATION_SUMMARY.md`
* **Test Harness:** `tests/integration/test_first_organism.py`
* **Artifacts:** `artifacts/first_organism/attestation.json`, `apps/ui/test/fixtures/first_organism_attestation.json`

---

## 2. Core Operational Pillars

### 2.1. The First Organism (MVDP)
* Operationalizes the Chain of Verifiable Cognition.
* Spec: `docs/FIRST_ORGANISM.md`.
* Implementation surfaces: `backend/axiom_engine`, `backend/rfl`, `basis/attestation`.
* Evidence: Integration logs, coverage artifacts, canonical attestation fixture.

### 2.2. Dual-Root Attestation
* Ensures reasoning (`R_t`) and UI inputs (`U_t`) receive deterministic Merkle roots.
* Implementation: `basis/attestation/dual_root.py`.
* Guided by `docs/ATTESTATION_SPEC.md`.  
* Audit trail: `docs/architecture/README.md` documents determinism proofs and dual-root expectations.

### 2.3. Reflexive Formal Learning
* Runner: `backend/rfl/runner.py`.
* Statistics: `backend/rfl/bootstrap_stats.py`.
* Verification spec: `docs/RFL_IMPLEMENTATION_SUMMARY.md`.

---

## 3. Chain of Artifacts

1. **Derivation ($U_t$)** writes candidate statements and logs.
2. **Experiment Result ($R_t$)** records metrics, proof counts, proof statuses.
3. **Attestation ($H_t$)** dual-root seals the run; the canonical JSON flows into `apps/ui/test/fixtures/first_organism_attestation.json`.
4. **RFL Verdict** computes bootstrap CIs, emits `[PASS/FAIL/ABSTAIN]`, and exports results/curves.

Each stage links a whitepaper chapter, a spec, and an integration test, enforcing the narrative: Theory → Execution → Proof → Asset.

---

## 4. Repository Structure & Responsibilities

* `backend/`: APIs, orchestrator, worker runtimes, derivation, RFL runner.
* `basis/`: Canonical attestation and cryptographic primitives (dual-root logic).
* `substrate/`: Lean tooling, proving environments, reproducibility harnesses.
* `tests/integration/`: End-to-end deliveries; the First Organism test is the definitive proof of life.
* `artifacts/`: Durable evidences (`attestation.json`, `rfl_results.json`, plots).

Architectural redundancies are resolved by pointing to this document for high-level storytelling and `docs/architecture/README.md` plus its supporting materials for deep-dive remediation work.

