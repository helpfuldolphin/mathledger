# Hash Pipeline Specification

## Overview

This document specifies the canonical hash pipeline for MathLedger statements.
All statement hashes **must** satisfy the whitepaper identity:

```
hash(s) = SHA256(DOMAIN_STMT || canonical_bytes(s))
```

Where:
- `DOMAIN_STMT = 0x02` (domain separation tag for statements)
- `canonical_bytes(s)` = `normalize(s).encode("ascii")`
- `normalize(s)` applies NNF, right-association, commutative flattening, and Unicode→ASCII mapping

## Normalization Rules

### 1. Unicode → ASCII Mapping

The `_SYMBOL_MAP` in `normalization/canon.py` defines the authoritative mapping:

| Unicode | ASCII | Description |
|---------|-------|-------------|
| `→` `⇒` `⟹` | `->` | Implication |
| `↔` `⇔` | `<->` | Biconditional |
| `∧` `⋀` | `/\` | Conjunction |
| `∨` `⋁` | `\/` | Disjunction |
| `¬` `￢` | `~` | Negation |
| `（` `）` | `()` | Parentheses |
| `⟨` `⟩` | `()` | Angle brackets |

### 2. Whitespace Handling

All whitespace (spaces, tabs, newlines, non-breaking spaces) is stripped from the output.

### 3. Right-Association of Implications

Implications are right-associative by default:
- `p -> q -> r` normalizes to `p->q->r` (right-associated)
- `(p -> q) -> r` normalizes to `(p->q)->r` (left-association preserved with parens)

### 4. Commutative Flattening

Conjunctions and disjunctions are:
1. Flattened: `(p /\ q) /\ r` → `p /\ q /\ r`
2. Sorted: Operands are sorted lexicographically for determinism
3. Deduplicated: `p /\ p` → `p`

### 5. Idempotency

- `p /\ p` → `p`
- `p \/ p` → `p`

### 6. Parenthesis Stripping

Redundant outer parentheses are removed: `((p))` → `p`

## Canonical Bytes Encoding

The `canonical_bytes(s)` function:

1. Calls `normalize(s)` to produce the canonical form
2. Encodes the result as ASCII bytes
3. Raises `ValueError` if non-ASCII characters remain (defensive guard)

```python
def canonical_bytes(s: Optional[str]) -> bytes:
    normalized = normalize("" if s is None else s)
    try:
        return normalized.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(f"normalized statement is not ASCII-clean: {normalized!r}") from exc
```

## Hash Computation

The `hash_statement(s)` function:

```python
def hash_statement(statement: str) -> str:
    canonical = canonical_bytes(statement)
    return sha256_hex(canonical, domain=DOMAIN_STMT)
```

Where `sha256_hex(data, domain)` computes `SHA256(domain + data).hexdigest()`.

## Callsite Audit

All statement hashing in the codebase must use the canonical pipeline:

### Derivation Pipeline
- `backend/axiom_engine/derive_utils.py::sha256_statement()` → uses `canonical_bytes`
- `backend/axiom_engine/pipeline.py` → uses `sha256_statement`
- `backend/axiom_engine/derive_core.py` → uses `sha256_statement`

### Ledger Ingestion
- `ledger/ingest.py::LedgerIngestor._upsert_statement()` → uses `hash_statement`

### Crypto Helpers
- `backend/crypto/hashing.py::hash_statement()` → uses `canonical_bytes`
- `backend/crypto/core.py::hash_statement()` → uses `canonical_bytes`

### DAG Repository
- `backend/dag/proof_dag.py::ProofDagRepository.insert_edge()` → receives pre-computed hashes
- Parent/child hashes are computed by callers using the canonical pipeline

### Ledger Hash Contract Helpers
- `ledger/ingest.py::verify_hash_contract()` → verifies hash matches canonical computation
- `ledger/ingest.py::assert_hash_contract()` → raises AssertionError on mismatch

## AST-Based Normalization (Draft)

The `normalization/ast_canon.py` module provides a robust AST-based alternative:

```python
from normalization.ast_canon import (
    parse_ast,
    normalize_ast,
    serialize_ast,
    canonical_bytes_ast,
)

# Parse → Normalize → Serialize
ast = parse_ast("(p ∧ q) → r")
normalized = normalize_ast(ast)
payload = serialize_ast(normalized)  # b"(p/\\q)->r"
```

Benefits over string-based normalization:
- Proper operator precedence handling
- Structural equivalence checking
- Double negation elimination
- Extensible for FOL (quantifiers, binders)

## Integration-Level Hash Audit

To verify hash integrity from raw database rows:

### Step 1: Extract Statements

```sql
SELECT id, hash, content_norm FROM statements WHERE hash IS NOT NULL;
```

### Step 2: Recompute Hashes

```python
from normalization.canon import canonical_bytes
from backend.crypto.hashing import DOMAIN_STMT
import hashlib

def recompute_hash(content_norm: str) -> str:
    canonical = canonical_bytes(content_norm)
    return hashlib.sha256(DOMAIN_STMT + canonical).hexdigest()

# For each row:
expected = recompute_hash(row.content_norm)
assert row.hash == expected, f"Hash mismatch for statement {row.id}"
```

### Step 3: Verify Block Attestations

```sql
SELECT b.id, b.reasoning_merkle_root, b.ui_merkle_root, b.composite_attestation_root
FROM blocks b;
```

```python
from attestation.dual_root import compute_composite_root

# For each block:
recomputed_h_t = compute_composite_root(row.reasoning_merkle_root, row.ui_merkle_root)
assert row.composite_attestation_root == recomputed_h_t, f"H_t mismatch for block {row.id}"
```

## Test Coverage

### Unit Tests (`tests/test_canon.py`)

- `TestCanonicalHashing::test_hash_statement_normalizes_unicode`
- `TestCanonicalHashing::test_canonical_bytes_ascii_encoding`
- `TestCanonicalHashing::test_first_organism_statement_hash`
- `TestCanonicalHashing::test_canonical_bytes_is_ascii_only`
- `TestCanonicalHashing::test_right_association_preserved`
- `TestCanonicalHashing::test_commutative_flattening_conjunction`
- `TestCanonicalHashing::test_commutative_flattening_disjunction`
- `TestCanonicalHashing::test_unicode_comprehensive_mapping`
- `TestCanonicalHashing::test_hash_stability_across_representations`
- `TestCanonicalHashing::test_whitespace_variations`

### Integration Tests (`tests/integration/test_first_organism.py`)

- `assert_hash_contract()` helper verifies hash identity at integration boundaries
- Hash contract assertions after derivation pipeline produces candidates
- Hash contract assertions after block sealing

## Invariants

1. **Determinism**: Same statement always produces same hash
2. **Unicode Equivalence**: Unicode and ASCII forms hash identically
3. **Whitespace Invariance**: Whitespace variations hash identically
4. **ASCII Purity**: `canonical_bytes` output is always ASCII-only
5. **Recomputability**: All stored hashes can be recomputed from normalized content

## Future Considerations

### FOL Extension

When extending to First-Order Logic:
- Add binder-aware alpha-renaming before normalization
- Ensure quantifier scope is preserved
- Update `_SYMBOL_MAP` with FOL symbols (∀, ∃, etc.)

### AST-Based Normalization

Current normalization operates on strings. Future work may:
- Parse to AST before normalization
- Normalize on AST structure
- Serialize AST to canonical bytes
- This would enable more robust structural equivalence

---

*Last updated: 2025-11-26*
*Bound to: `tests/test_canon.py`, `tests/integration/test_first_organism.py`*

