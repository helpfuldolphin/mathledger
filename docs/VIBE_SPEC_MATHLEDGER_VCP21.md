# VIBE SPECIFICATION DOCUMENT (VSD) — VCP 2.1
**Status:** LIVING | **Version:** 2.1 | **Scope:** GLOBAL

## 1. The Vibe
**MathLedger is not a playground; it is a precision instrument.**
We are building the "ImageNet for theorem proving" — a cryptographically auditable substrate for mathematical truth. The codebase must reflect the rigorous, deductive nature of the subject matter.

*   **Ultra-clean:** Code should look like it was written by a single, highly disciplined author.
*   **Investor-grade:** Robust, documented, and metrics-driven. No "hacky" scripts in production paths.
*   **Cryptographically Precise:** Every byte matters. Canonicalization (RFC 8785) is not optional; it is the law.
*   **Abstention-First:** Better to crash or return `None` than to yield a hallucination.
*   **No Slop:** Zero tolerance for unused imports, dead code, commented-out blocks, or vague variable names.

## 2. Intent & Mission
**From Spanning Set to Minimal Basis.**
The current `mathledger/` repository is a "spanning set" — it contains all necessary ideas but in a messy, overcomplete form. Our mission is to distill this into a **Minimal Basis**: a compact, orthogonal set of modules that can be extracted into a clean `basis/` directory (and eventually a separate repo).

## 3. Global Constraints (The Invariants)

### 3.1 Determinism & Canonicalization
*   **RFC 8785:** All JSON serialization for hashes, signatures, or ledger storage MUST use RFC 8785 (JCS).
*   **No Hidden Randomness:** Random seeds must be explicit inputs to functions. No `import random` calls deep in the library logic.
*   **Bit-Perfect Replay:** Re-running a derivation sequence with the same seed MUST yield the exact same artifact hashes.

### 3.2 Security & Integrity
*   **Dual Attestation:** Every artifact must support the `(R_t, U_t, H_t)` triple attestation structure (Reasoning, User, Hash).
*   **Proof-or-Abstain:** Functions claiming to verify a proof must either return a cryptographically valid result or an explicit failure. No "best effort" success.
*   **Input Validation:** All external inputs (API, CLI) must be strictly validated via Pydantic schemas before processing.

### 3.3 Operational Continuity
*   **Shim Strategy:** We cannot break `uv run`, `pytest`, or `backend/worker.py` immediately.
*   **Migration:** New logic goes into parallel "v2" or "clean" modules. Old paths are deprecated via shims, then removed.

## 4. Coding Style & Rules

### 4.1 Python (Backend)
*   **Type Hints:** 100% coverage for public APIs. `mypy --strict` goal.
*   **Error Handling:** Use custom exception hierarchies (e.g., `MathLedgerError`, `IntegrityError`).
*   **Imports:** Absolute imports preferred. Group by standard lib, 3rd party, local.
*   **Docstrings:** Google style. Mandatory for all public classes and functions.

### 4.2 Lean (Substrate)
*   **Structure:** Follow standard Lean 4 project layout.
*   **Namespaces:** Use `MathLedger.*` to avoid collisions.

## 5. Refactor Priorities
1.  **Substrate Hardening:** `backend/logic` and `backend/axiom_engine` must be mathematically perfect.
2.  **Ledger Integrity:** `backend/ledger` must implement the Whitepaper's Merkle structures exactly.
3.  **RFL Control Loop:** `backend/rfl` must be the single source of truth for "forgetting" and "promotion".

## 6. Integration Requirements (First Organism)
The "First Organism" loop is the acceptance test for VCP 2.1:
1.  **Boot:** System starts from `genesis.json`.
2.  **Derive:** Generates new theorems (Prop Logic).
3.  **Verify:** Lean confirms them.
4.  **Forget:** RFL prunes weak paths.
5.  **Attest:** Ledger seals the block.
6.  **Rest:** System shuts down and restarts with perfect state recovery.

**Success = automated, bit-perfect execution of this loop 100 times.**
