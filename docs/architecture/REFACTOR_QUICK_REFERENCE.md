# Architecture Refactor Quick Reference

**Related:** [OVERSIGHT_REPORT.md](./OVERSIGHT_REPORT.md)

## Quick Migration Guide

### When writing new code:

#### ✅ DO: Use centralized crypto utilities
```python
# Good - Use centralized module with domain separation
from backend.crypto.hashing import sha256_hex, merkle_root, DOMAIN_STMT

statement_hash = sha256_hex(statement, domain=DOMAIN_STMT)
tree_root = merkle_root(leaves)
```

#### ❌ DON'T: Implement local crypto functions
```python
# Bad - Local implementation
def compute_sha256(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()
```

---

#### ✅ DO: Use RFC 8785 canonical module (when available)
```python
# Good - Will be available after Phase 1 refactor
from backend.core.crypto.canon import canonicalize_json

canonical = canonicalize_json(data)
```

#### ❌ DON'T: Copy-paste canonicalization
```python
# Bad - Duplicate implementation
def canonicalize(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'))
```

---

#### ✅ DO: Use standardized output (when available)
```python
# Good - Will be available after Phase 2 refactor
from backend.core.output.status import emit_pass, emit_fail, emit_abstain

emit_pass(f"Test completed: {test_count} tests")
emit_fail(f"Validation failed: {error}")
emit_abstain("Insufficient data for analysis")
```

#### ❌ DON'T: Use ad-hoc print statements
```python
# Bad - Inconsistent format
print(f"PASS: Test completed")
print("Test failed!")  # Missing [FAIL] tag
```

---

## Module Import Reference

### Current (Before Refactor)

```python
# Cryptographic hashing (AVAILABLE NOW)
from backend.crypto.hashing import (
    sha256_hex,           # SHA-256 returning hex string
    sha256_bytes,         # SHA-256 returning bytes
    hash_statement,       # Statement hashing with STMT domain
    merkle_root,          # Merkle root with domain separation
    compute_merkle_proof, # Merkle proof generation
    DOMAIN_STMT,          # Domain tags for second-preimage protection
    DOMAIN_LEAF,
    DOMAIN_NODE,
    DOMAIN_BLCK
)

# Authentication
from backend.crypto.auth import generate_token, verify_token

# Protocol handshake
from backend.crypto.handshake import perform_handshake
```

### Future (After Refactor)

```python
# Canonical JSON (PLANNED - Phase 1)
from backend.core.crypto.canon import canonicalize_json

# Standardized output (PLANNED - Phase 2)
from backend.core.output.status import emit_pass, emit_fail, emit_abstain

# Audit utilities (PLANNED - Phase 2)
from backend.core.audit.metrics import track_operation, get_metrics
```

---

## Pre-Commit Checklist

Before submitting a PR, verify:

- [ ] No new `def compute_sha256()` functions in tools/ or scripts/
- [ ] No new `def canonicalize()` functions outside backend/core/
- [ ] All crypto operations use `backend.crypto.*` imports
- [ ] Output uses `[PASS]`, `[FAIL]`, or `[ABSTAIN]` format
- [ ] Functions are < 30 lines (McCabe complexity < 10)
- [ ] Files are < 600 lines (except generated files)

---

## Files Currently Being Refactored

**Do not add new crypto implementations to:**
- ⚠️ `tools/ci/*.py` - Use backend.crypto instead
- ⚠️ `tools/repro/*.py` - Use backend.crypto instead
- ⚠️ `tools/perf/*.py` - Use backend.crypto instead
- ⚠️ `scripts/*.py` - Use backend.crypto instead

**High-complexity files under review:**
- ⚠️ `backend/logic/canon.py` - Being decomposed
- ⚠️ `backend/orchestrator/app.py` - Being split into routes/handlers/ui
- ⚠️ `backend/axiom_engine/derive.py` - Extracting smoke tests

---

## Common Patterns to Avoid

### Pattern 1: Embedding tests in production code
```python
# Bad - test embedded in production module
def main():
    if "--test" in sys.argv:
        run_smoke_test()
    else:
        run_production()
```

**Fix:** Move tests to `tests/` directory

---

### Pattern 2: God objects with too many responsibilities
```python
# Bad - too many responsibilities
class Orchestrator:
    def handle_api_request(self): ...
    def render_ui(self): ...
    def query_database(self): ...
    def format_metrics(self): ...
```

**Fix:** Split into separate modules (routes, handlers, services)

---

### Pattern 3: Deep nesting
```python
# Bad - deep nesting increases complexity
def process(data):
    if condition1:
        if condition2:
            if condition3:
                if condition4:
                    return result
```

**Fix:** Use early returns, extract helper functions

---

## Complexity Thresholds

| Metric | Threshold | Tool |
|--------|-----------|------|
| Lines per file | 600 | `wc -l` |
| Lines per function | 30 | Manual review |
| Cyclomatic complexity | 10 (Grade B) | `radon cc` |
| Cognitive complexity | 15 | Manual review |

**Check your code:**
```bash
# Check complexity of a file
radon cc -s your_file.py

# Check complexity of entire module
radon cc -s -a backend/your_module/

# Get average complexity
radon cc -s -a --total-average backend/
```

---

## Contact

Questions about refactoring? See:
- Full analysis: [OVERSIGHT_REPORT.md](./OVERSIGHT_REPORT.md)
- Contributing guide: [../../CONTRIBUTING.md](../../CONTRIBUTING.md)
- Agent guidelines: [../../AGENTS.md](../../AGENTS.md)

---

*Last updated: 2025-11-02*
