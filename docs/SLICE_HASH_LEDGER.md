# Slice Hash Ledger Specification

> **STATUS: PHASE II — NOT USED FOR FO/PHASE I RUNS**
>
> This specification governs cryptographic traceability for Phase II curriculum slices.
> It is not part of Evidence Pack v1 and does not apply to Phase I artifacts.

**Version:** 1.0.0
**Author:** Claude E — Hash Integrity Inspector (Phase II Slices)
**Date:** 2025-12-06
**Bound to:** `scripts/verify_slice_hashes.py`, `config/curriculum_uplift_phase2.yaml`

---

## 1. Purpose

This document specifies the **Slice Hash Ledger** — an append-only cryptographic audit trail that ensures:

1. **Immutability**: Formula-to-hash bindings cannot be silently modified
2. **Traceability**: Every hash can be traced to its canonical derivation
3. **Governance**: Slice modifications require explicit attestation
4. **Reproducibility**: Any observer can recompute and verify hashes

The ledger provides the foundation for Phase II governance claims about curriculum integrity.

---

## 2. Core Identity: The Hash Pipeline

All slice formula hashes **must** satisfy the whitepaper identity:

$$
\text{hash}(s) = \text{SHA256}(\mathcal{D} \| \mathcal{E}(\mathcal{N}(s)))
$$

| Symbol | Name | Implementation | Location |
|--------|------|----------------|----------|
| $\mathcal{N}$ | Normalization | `normalize(expr)` | `normalization/canon.py` |
| $\mathcal{E}$ | Encoding | `str.encode("ascii")` | `normalization/canon.py:canonical_bytes` |
| $\mathcal{D}$ | Domain Tag | `DOMAIN_STMT = 0x02` | `substrate/crypto/hashing.py` |
| $\|$ | Concatenation | `domain + payload` | — |

### 2.1 Canonical Computation

```python
from normalization.canon import normalize, canonical_bytes
from substrate.crypto.hashing import hash_statement, DOMAIN_STMT

# The canonical pipeline
def compute_slice_hash(formula: str) -> str:
    """
    Compute the canonical hash for a slice formula.

    Pipeline: formula → normalize → ascii_bytes → domain_prefix → SHA256 → hex
    """
    return hash_statement(formula)

# Equivalently:
def compute_slice_hash_explicit(formula: str) -> str:
    import hashlib
    normalized = normalize(formula)
    payload = normalized.encode("ascii")
    return hashlib.sha256(DOMAIN_STMT + payload).hexdigest()
```

### 2.2 Domain Separation

The `DOMAIN_STMT = 0x02` prefix prevents second-preimage attacks and ensures slice hashes cannot collide with:

| Domain | Tag | Purpose |
|--------|-----|---------|
| `LEAF` | `0x00` | Merkle tree leaf nodes |
| `NODE` | `0x01` | Merkle tree internal nodes |
| `STMT` | `0x02` | **Statement/formula content** |
| `BLCK` | `0x03` | Block header identity |

---

## 3. Append-Only Ledger Format

The Slice Hash Ledger is represented as a YAML-compatible structure with the following schema:

### 3.1 Ledger Entry Schema

```yaml
# Individual ledger entry
entry:
  formula: "<raw formula string>"          # Original input
  normalized: "<canonical form>"           # After normalize()
  hash: "<64-char hex digest>"             # SHA256 output
  role: "<semantic role>"                  # axiom, intermediate, target, decoy
  slice: "<slice name>"                    # Owning slice
  added_at: "<ISO 8601 timestamp>"         # First appearance
  added_by: "<agent identifier>"           # Claude E, etc.
  attestation_hash: "<optional>"           # Hash of attestation record
```

### 3.2 Slice Manifest Schema

Each slice maintains a manifest in `config/curriculum_uplift_phase2.yaml`:

```yaml
slice_name:
  formula_pool_entries:
    - formula: "p->q->p"
      hash: "248e2c30377c23e7a10d20d203eef09b9a136c30729ece89910908a0f36c89b1"
      role: axiom_k
      # Optional: chain_hop, required, etc.
```

### 3.3 Ledger Invariants

1. **Append-Only**: Entries may be added but never removed or modified
2. **Hash Stability**: `hash(formula)` is immutable once recorded
3. **Normalization Idempotency**: `normalize(normalize(s)) == normalize(s)`
4. **No Silent Updates**: Hash changes require explicit re-attestation

---

## 4. Hash Lineage Invariants

### 4.1 Formula → Hash Binding

Once a formula-hash pair is recorded in the ledger, the binding is permanent:

```
INVARIANT: ∀ entry ∈ Ledger:
    hash_statement(entry.formula) == entry.hash
```

Verification: `scripts/verify_slice_hashes.py`

### 4.2 Normalization Stability

The normalized form must be stable across codebase versions:

```
INVARIANT: ∀ formula:
    normalize(formula) @ version_N == normalize(formula) @ version_N+1
```

Breaking this invariant would cause **hash drift** (see Section 7).

### 4.3 Role Consistency

Each formula has exactly one semantic role within a slice:

```
INVARIANT: ∀ slice, formula:
    |{role : (formula, role) ∈ slice.entries}| ≤ 1
```

### 4.4 Target Hash Integrity

Target hashes referenced in `success_metric` must exist in `formula_pool_entries`:

```
INVARIANT: ∀ slice:
    slice.success_metric.target_hashes ⊆ {e.hash : e ∈ slice.formula_pool_entries}
```

---

## 5. Canonical Serialization Rules

### 5.1 Formula Serialization

Formulas are serialized using strict ASCII with the following rules:

| Input | Normalized Output |
|-------|-------------------|
| `p → q` | `p->q` |
| `p ∧ q` | `p/\q` |
| `p ∨ q` | `p\/q` |
| `¬p` | `~p` |
| `(p -> q) -> r` | `(p->q)->r` |
| `p -> q -> r` | `p->q->r` |
| `  p  ->  q  ` | `p->q` |

### 5.2 Hash Serialization

Hashes are serialized as lowercase 64-character hexadecimal strings:

```
Valid:   248e2c30377c23e7a10d20d203eef09b9a136c30729ece89910908a0f36c89b1
Invalid: 248E2C30...  (uppercase)
Invalid: 0x248e2c...  (prefix)
Invalid: 248e2c30     (truncated)
```

### 5.3 YAML Serialization

Ledger entries in YAML use quoted strings for all values:

```yaml
# Correct
- formula: "p->q->p"
  hash: "248e2c30377c23e7a10d20d203eef09b9a136c30729ece89910908a0f36c89b1"

# Incorrect (unquoted)
- formula: p->q->p
  hash: 248e2c30377c23e7a10d20d203eef09b9a136c30729ece89910908a0f36c89b1
```

### 5.4 JSON Serialization (Audit Reports)

JSON audit reports use RFC 8785 compatible formatting:

```json
{
  "formula": "p->q->p",
  "normalized": "p->q->p",
  "hash": "248e2c30377c23e7a10d20d203eef09b9a136c30729ece89910908a0f36c89b1",
  "role": "axiom_k"
}
```

---

## 6. Requirements for Future Slice Authors

### 6.1 Pre-Submission Checklist

Before adding formulas to a slice, authors **must**:

- [ ] Compute hashes using `hash_statement()` from canonical pipeline
- [ ] Verify normalized form matches expectations
- [ ] Run `verify_slice_hashes.py` with zero mismatches
- [ ] Document formula roles (axiom, intermediate, target, decoy)
- [ ] Ensure target hashes are included in `formula_pool_entries`

### 6.2 Hash Computation Procedure

```python
from normalization.canon import normalize
from substrate.crypto.hashing import hash_statement

# Step 1: Define formula
formula = "((p -> q) -> p) -> p"  # Peirce's law

# Step 2: Normalize
normalized = normalize(formula)
print(f"Normalized: {normalized}")  # ((p->q)->p)->p

# Step 3: Compute hash
formula_hash = hash_statement(formula)
print(f"Hash: {formula_hash}")  # 1065e56b3c644ae6f05a947337e6a8afa7e968866b353a1930a94950ac4826cf

# Step 4: Add to YAML
entry = f'''
- formula: "{normalized}"
  hash: "{formula_hash}"
  role: target_peirce
'''
```

### 6.3 Slice Modification Protocol

To modify an existing slice:

1. **Never delete entries** — mark as `deprecated: true` if needed
2. **Never change existing hashes** — this would break lineage
3. **Add new entries** with new formulas/hashes
4. **Re-run verification** after any change
5. **Document changes** in slice changelog

### 6.4 Forbidden Practices

| Practice | Reason |
|----------|--------|
| Manual hash computation | Risk of pipeline deviation |
| Editing normalized forms | Hash/formula mismatch |
| Removing entries | Breaks append-only invariant |
| Using non-canonical hashing | CVE-2012-2459 vulnerability |
| Skipping verification | Undetected drift |

---

## 7. Hash Drift Detection

**Hash drift** occurs when the computed hash for a formula changes between codebase versions. This is a **critical integrity violation**.

### 7.1 Causes of Hash Drift

| Cause | Example | Detection |
|-------|---------|-----------|
| Normalization rule change | Right-assoc → left-assoc | `verify_slice_hashes.py` MISMATCH |
| Unicode mapping change | `→` maps to `=>` | Normalized form differs |
| Encoding change | UTF-8 → UTF-16 | Binary payload differs |
| Domain tag change | `0x02` → `0x12` | All hashes change |

### 7.2 Detection Mechanism

The `verify_slice_hashes.py` script detects drift by recomputing all hashes:

```bash
uv run python scripts/verify_slice_hashes.py config/curriculum_uplift_phase2.yaml
```

**Output on drift detection:**

```
--- Verifying Slice: slice_uplift_goal ---
  [MISMATCH] Formula: ((p->q)->p)->p
    Normalized: ((p->q)->p)->p
    Expected:   1065e56b3c644ae6f05a947337e6a8afa7e968866b353a1930a94950ac4826cf
    Actual:     a1b2c3d4e5f6... (different)
    Role:       target_peirce

RESULT: FAILED — 1 hash mismatch(es) detected
```

### 7.3 Drift Response Protocol

When drift is detected:

1. **STOP** — Do not proceed with experiments
2. **Diagnose** — Identify which component changed
3. **Assess** — Determine if change is intentional
4. **If unintentional** — Revert the change
5. **If intentional** — Create migration plan:
   - Document the normalization change
   - Update all affected hashes
   - Re-attest the ledger
   - Increment ledger version

### 7.4 Governance-Grade Drift Explanation

For governance purposes, a hash drift report must include:

```yaml
drift_report:
  detected_at: "2025-12-06T12:00:00Z"
  affected_slices:
    - slice_uplift_goal
    - slice_uplift_tree
  affected_entries: 12
  root_cause:
    component: "normalization/canon.py"
    change_type: "right-association handling"
    commit: "<git commit hash>"
  impact:
    experiments_blocked: true
    data_affected: "None (detected before runs)"
  resolution:
    action: "Reverted normalization change"
    verified_at: "2025-12-06T14:00:00Z"
    verifier: "scripts/verify_slice_hashes.py"
```

---

## 8. Audit Trail & Attestation

### 8.1 Verification Artifacts

Each verification run produces:

1. **Console output** — Human-readable summary
2. **JSON report** — Machine-readable audit (`--report flag`)
3. **Exit code** — 0 (pass) or 1 (fail)

### 8.2 JSON Report Schema

```json
{
  "phase": "II",
  "status": "PHASE_II_DESIGNED",
  "config_path": "config/curriculum_uplift_phase2.yaml",
  "config_version": 2,
  "slices": {
    "slice_uplift_goal": {
      "system": "pl",
      "total_entries": 14,
      "checked_entries": 14,
      "passed_entries": 14,
      "failed_entries": 0,
      "mismatches": []
    }
  },
  "summary": {
    "total_slices": 4,
    "total_entries": 51,
    "total_checked": 51,
    "total_passed": 51,
    "total_failed": 0
  }
}
```

### 8.3 Pre-Experiment Attestation

Before running U2 experiments, the following attestation is required:

```yaml
pre_experiment_attestation:
  experiment_id: "uplift_u2_goal_001"
  slice: "slice_uplift_goal"
  ledger_verification:
    timestamp: "<ISO 8601>"
    tool: "scripts/verify_slice_hashes.py"
    config: "config/curriculum_uplift_phase2.yaml"
    result: "PASS"
    total_checked: 14
    total_failed: 0
    report_hash: "<SHA256 of JSON report>"
```

---

## 9. Integration with U2 Runner

### 9.1 Pre-Run Verification

The U2 runner **should** invoke hash verification before starting:

```python
def pre_run_checks(config_path: str) -> bool:
    """Verify slice hashes before experiment run."""
    import subprocess
    result = subprocess.run(
        ["python", "scripts/verify_slice_hashes.py", config_path, "-q"],
        capture_output=True
    )
    return result.returncode == 0
```

### 9.2 Runtime Hash Contract

During derivation, any produced hash can be verified:

```python
from normalization.canon import normalize
from substrate.crypto.hashing import hash_statement

def assert_hash_contract(formula: str, claimed_hash: str) -> None:
    """Verify a formula-hash binding at runtime."""
    computed = hash_statement(formula)
    if computed != claimed_hash:
        raise HashContractViolation(
            f"Hash mismatch: {formula} → {computed} != {claimed_hash}"
        )
```

---

## 10. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-06 | Initial specification |

---

## 11. References

- `docs/SLICE_HASH_EXECUTION_BINDING.md` — Execution binding and reconciliation
- `docs/U2_GOVERNANCE_PIPELINE.md` — Governance verification pipeline
- `docs/HASHING_SPEC.md` — Core hashing specification
- `docs/HASH_PIPELINE_SPEC.md` — Pipeline implementation details
- `docs/VSD_PHASE_2.md` — Phase II architecture
- `scripts/verify_slice_hashes.py` — Verification tool
- `config/curriculum_uplift_phase2.yaml` — Phase II slice definitions
- `experiments/prereg/PREREG_UPLIFT_U2.yaml` — Preregistration template

---

*— Claude E, Hash Law (Phase II Slices)*
