# Evidence Pack Verifier Specification

**Purpose**: Static webpage that verifies evidence packs locally (no backend).

**Status**: Specification
**Target**: `site/verifier/index.html` (hosted on mathledger.ai)

---

## Overview

A user uploads an `evidence_pack.json` file. The page recomputes `U_t`, `R_t`, and `H_t` using the **exact same** canonicalization and hashing logic as the Python codebase. The page displays PASS or FAIL with a readable diff.

---

## Evidence Pack Schema

### Required Fields

```json
{
  "schema_version": "v1",
  "uvil_events": [
    {
      "event_id": "string (content-derived)",
      "event_type": "COMMIT | EDIT | PROMOTE",
      "committed_partition_id": "string",
      "user_fingerprint": "string",
      "epoch": "integer"
    }
  ],
  "reasoning_artifacts": [
    {
      "artifact_id": "string (content-derived)",
      "claim_id": "string",
      "trust_class": "MV | FV | PA",
      "validation_outcome": "VERIFIED | REFUTED | ABSTAINED",
      "proof_payload": { ... }
    }
  ],
  "u_t": "64-char hex (recorded UI root)",
  "r_t": "64-char hex (recorded reasoning root)",
  "h_t": "64-char hex (recorded composite root)"
}
```

### Validation Rules

1. `uvil_events` is an array (may be empty)
2. `reasoning_artifacts` is an array (may be empty)
3. `trust_class` must NOT be `ADV` (ADV excluded from R_t)
4. `validation_outcome` must be present (abstention preservation)
5. `u_t`, `r_t`, `h_t` must be 64-character lowercase hex strings

---

## Canonicalization Contract

### RFC 8785-Style Canonicalization

The verifier uses RFC 8785 JSON Canonicalization Scheme:

1. **Keys sorted lexicographically** (Unicode code point order)
2. **No insignificant whitespace**
3. **Unicode escapes normalized**
4. **Numbers in standard form** (no leading zeros, no trailing zeros after decimal)

### Leaf Contract (CRITICAL)

```
compute_ui_root() and compute_reasoning_root() expect RAW DICT PAYLOADS,
NOT pre-hashed strings. Canonicalization happens internally via _canonicalize_leaf().
```

**Domain Separation:**
- UI leaves: `0xA1` + "ui-leaf"
- Reasoning leaves: `0xA0` + "reasoning-leaf"

**Empty Set Sentinels:**
- Empty UI events: `SHA256(b"UI:EMPTY")`
- Empty reasoning artifacts: `SHA256(b"REASONING:EMPTY")`

---

## Verification Algorithm

```javascript
function verifyEvidencePack(pack) {
  // 1. Canonicalize and hash uvil_events
  const recomputed_u_t = computeUiRoot(pack.uvil_events);

  // 2. Canonicalize and hash reasoning_artifacts
  const recomputed_r_t = computeReasoningRoot(pack.reasoning_artifacts);

  // 3. Compute composite: H_t = SHA256(R_t || U_t)
  const recomputed_h_t = sha256(recomputed_r_t + recomputed_u_t);

  // 4. Compare all three
  return {
    u_t_match: recomputed_u_t === pack.u_t,
    r_t_match: recomputed_r_t === pack.r_t,
    h_t_match: recomputed_h_t === pack.h_t,
    overall: all_match
  };
}
```

---

## UI Requirements

### Upload Panel
- File input accepting `.json`
- Drag-and-drop support
- Client-side only (no upload to server)

### Result Banner
- **PASS (green)**: All three hashes match
- **FAIL (red)**: At least one hash mismatch

### Diff Panel (on FAIL)
| Root | Recorded | Recomputed | Status |
|------|----------|------------|--------|
| U_t  | `abc123...` | `abc123...` | MATCH |
| R_t  | `def456...` | `xyz789...` | MISMATCH |
| H_t  | `ghi012...` | `uvw345...` | MISMATCH |

### Metadata Display
- Schema version
- Event counts (uvil_events, reasoning_artifacts)
- Trust class distribution

---

## Security Stance

1. **No network calls**: Verification runs entirely in browser
2. **No data exfiltration**: Uploaded file never leaves client
3. **No external dependencies**: All crypto implemented inline (or vendored)
4. **Deterministic**: Same input always produces same output

---

## Explicit Non-Claims

This verifier does NOT:

1. **Verify claim truth**: It verifies hash integrity, not whether claims are correct
2. **Validate trust class assignment**: It checks structure, not appropriateness
3. **Authenticate users**: No identity verification
4. **Guarantee security**: It's an auditor tool, not a security seal
5. **Replace human review**: Passing verification does not mean the evidence pack is trustworthy

The verifier answers ONE question:
> "Does this evidence pack's recorded hashes match what would be recomputed from its raw payloads?"

---

## Authoritative Test Vectors

**Canonical Vector File**: `releases/evidence_pack_verifier_vectors.v0.2.0.json`

This file is the **authoritative** source for verifier test vectors. It is:
- Version-pinned (v0.2.0)
- Generated using the SAME functions as `replay_verify`
- Repo-tracked and deterministic

### Vector Contents

| Category | Count | Description |
|----------|-------|-------------|
| Valid Packs | 2 | Expected result: PASS |
| Invalid Packs | 3 | Expected result: FAIL (different reasons) |
| Canonicalization Tests | 2 | RFC 8785 edge cases |

### Invalid Pack Failure Reasons

| Name | Failure Reason | What's Wrong |
|------|----------------|--------------|
| `invalid_tampered_ht` | `h_t_mismatch` | h_t field set to zeros |
| `invalid_tampered_reasoning_leaf` | `r_t_mismatch` | reasoning_artifacts modified after r_t recorded |
| `invalid_missing_validation_outcome` | `missing_required_field` | validation_outcome field missing |

### Vector Schema

Each vector has:
```json
{
  "name": "human_readable_name",
  "description": "What this test case validates",
  "expected_result": "PASS | FAIL",
  "expected_failure_reason": null | "h_t_mismatch" | "r_t_mismatch" | ...,
  "pack": { ... }
}
```

### Regeneration

To regenerate vectors (requires audit approval for production):
```bash
uv run python scripts/generate_verifier_vectors.py
```

### Validation Tests

```bash
uv run pytest tests/governance/test_verifier_vectors_artifact.py -v
```

---

## Implementation Phases

### Phase 1: Python Test Vectors
- Generate deterministic test vectors from Python
- Export as JSON for JS verifier validation
- Include golden hashes and expected outcomes

### Phase 2: JS Implementation
- Port `rfc8785_canonicalize()` to JavaScript
- Port `compute_ui_root()`, `compute_reasoning_root()`, `compute_composite_root()`
- Validate against Python test vectors

### Phase 3: Static Page
- Build single-file HTML with inline JS
- Add to `site/verifier/index.html`
- Deploy via Cloudflare Pages

---

## Cross-Reference

| Python Function | Location | JS Equivalent |
|-----------------|----------|---------------|
| `rfc8785_canonicalize()` | `substrate/crypto/core.py:35` | `canonicalize()` |
| `compute_ui_root()` | `attestation/dual_root.py:295` | `computeUiRoot()` |
| `compute_reasoning_root()` | `attestation/dual_root.py:282` | `computeReasoningRoot()` |
| `compute_composite_root()` | `attestation/dual_root.py:308` | `computeCompositeRoot()` |
| `_canonicalize_leaf()` | `attestation/dual_root.py:162` | `canonicalizeLeaf()` |

---

## Contract Enforcement

This specification is enforced by:
- `releases/evidence_pack_verifier_vectors.v0.2.0.json` (Authoritative vectors)
- `tests/governance/test_verifier_vectors_artifact.py` (Artifact validation)
- `tests/governance/test_evidence_pack_replay_vectors.py` (Replay testing)
- Future: `site/verifier/test.js` (JS validation against vectors)

---

**Author**: Claude A
**Date**: 2026-01-02
