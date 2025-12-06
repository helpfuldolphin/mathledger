# Vibe Style Guide — MathLedger

**Authority:** Vibe Compliance Guardian
**Version:** 1.0
**Scope:** First Organism Path (UI → RFL)

## 1. Core Philosophy
**"Financial-grade cryptographic substrate with research-grade clarity."**

Every line of code, log message, and documentation fragment must read as if it belongs to a serious, investor-ready research-engineering product. We are not building a toy; we are building a sovereign mathematical ledger.

### 1.1 The "Proof-or-Abstain" Principle
- If a component cannot guarantee correctness, it must **explicitly abstain**.
- Silence is failure. Explicit abstention is a valid, handled state.
- Use `ABSTAIN:` prefix in logs/errors for controlled failures.

## 2. Naming Conventions

### 2.1 Modules and Scripts
- **Snake Case:** `ledger_ingestor.py`, `dual_root.py`.
- **Verbs for Actions:** `verify_dual_root.py` (not `verification.py` unless it's a library).
- **No "Utils" or "Helpers":** Be specific. `hashing.py` instead of `crypto_utils.py`.

### 2.2 Classes (Types)
- **PascalCase:** `LedgerIngestor`, `DualAttestationMetadata`, `AttestedRunContext`.
- **Descriptive & Serious:**
  - BAD: `Runner`, `Checker`, `Blob`.
  - GOOD: `RFLRunner`, `IntegritySentinel`, `CanonicalStatement`.

### 2.3 Functions and Variables
- **Snake Case:** `compute_merkle_root`, `ingest_block`.
- **No Metasyntactic Variables:** Never use `foo`, `bar`, `baz`, `tmp`, `junk`.
- **Explicit Types:** `candidate_hash` (not `h`), `block_height` (not `n`).

## 3. Logging and Error Policy

### 3.1 Tone
- **Professional:** No jokes, no slang, no cutesy phrases.
- **No:** "Oops", "Whoops", "Boom", "Crunching numbers...", "Sanity check failed".
- **Yes:** "Integrity violation detected", "Computation started", "Invariant failed".

### 3.2 Structure
- **Cause & Remediation:** Errors must state *what* happened and *why*.
- **Format:** `[COMPONENT] <Status>: <Details> (Ctx=<Context>)`
- **Example:** `[LEDGER] FAILED: Hash mismatch in block 42 (Expected=0x123, Actual=0x456)`

### 3.3 Abstention
- When a proof or process voluntarily halts due to lack of resources/capabilities (not a bug):
  - Log: `[ABSTAIN] <Component>: <Reason>`
  - Exception: Raise `AbstentionError` (or similar) with the message.

## 4. Documentation

### 4.1 Docstrings
- **Mandatory** for all public modules, classes, and functions in the critical path.
- **Cross-Linking:** Reference specific sections of the whitepaper or spec.
- **Format:**
  ```python
  def compute_composite_root(reasoning_root: str, ui_root: str) -> str:
      """
      Compute the dual-root composite hash H_t.

      Combines the reasoning merkle root R_t and the UI event merkle root U_t
      into the final block commit.

      Reference: MathLedger Whitepaper §4.2 (Dual Attestation).
      """
  ```

## 5. Banned Patterns
- **Comments:** `TODO`, `FIXME`, `HACK`, `WIP`, `SLOP` (unless accompanied by a tracking ticket/issue ID, but ideally remove).
- **Language:** "hacky", "quick fix", "ignore this", "garbage".
- **Code:** Commented-out blocks of code (delete them).

## 6. Integration Test Output
The First Organism integration test must conclude with:
`[PASS] First Organism: UI→RFL closed-loop attested (H_t=<hash>) — organism alive.`

## 7. Enforcement
- Run `make vibe-check` or `uv run python tools/vibe_check.py` from the workspace root before pushing changes in the First Organism path.
- The Vibe Check now covers the orchestrator and interface/API layers in addition to the previously protected ledger, derivation, and RFL modules.
- Our CI workflow invokes a dedicated `vibe_check` job that gates the downstream `browsermcp`, `reasoning`, and `dual-attestation` jobs; any failures must be addressed before merging.
