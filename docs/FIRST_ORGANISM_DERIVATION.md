# First Organism Derivation: Abstention Protocol

> **Protocol Version:** 2.0  
> **Telemetry Version:** 1.1.0  
> **Status:** Active  
> **Last Updated:** 2025-11-27  
> **Owner:** Derivation Configurator & Abstention Engineer (Cursor E)

---

## 1. Purpose

This document specifies the **First Organism Abstention Experiment**—a controlled derivation scenario that **guarantees** the Lean verifier's abstention path is triggered. The experiment validates the complete causal chain:

```
UI Event → Curriculum Gate → Derivation → Lean Abstain → Ledger Ingest → Dual Attestation (Hₜ) → RFL Metabolism
```

The abstention is **not a failure**—it is a **designed, guaranteed outcome** that demonstrates the organism's ability to detect and record its own epistemic limitations.

---

## 2. Abstention Guarantee

### 2.1 Why Abstention is Guaranteed

The `make_first_organism_derivation_config()` function constructs a configuration that **always produces at least one abstention**. This is guaranteed by the following invariants:

1. **Seeds**: `p` and `(p -> q)` are seeded into the pipeline
2. **MP Derivation**: Modus Ponens fires: `p + (p -> q) ⊢ q`
3. **Non-Tautology**: `q` is a bare propositional atom, which is **never** a tautology
   - A tautology must be true under ALL valuations
   - `q` is false when `q := false`
4. **Verification Cascade**:
   - Pattern matcher: `q` is not a known tautology schema
   - Truth-table: `q` evaluates to false for some valuations → rejected
   - Lean fallback: disabled (`ML_ENABLE_LEAN_FALLBACK` not set) → returns `lean-disabled`

**Result**: The verifier abstains, and `q` is recorded in `DerivationResult.abstained_candidates`.

### 2.2 Configuration Validation

The `FirstOrganismDerivationConfig` includes a `validate()` method that verifies:

```python
def validate(self) -> bool:
    if not self.seed_statements:
        return False
    if not _is_guaranteed_non_tautology(self.guaranteed_non_tautology):
        return False
    if self.abstention_method not in ABSTENTION_METHODS:
        return False
    return True
```

---

## 3. Experimental Configuration

### 3.1 Slice Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `atoms` | 2 | Minimal alphabet (`p`, `q`) keeps combinatorial space tiny |
| `depth_max` | 2 | Allows `p -> q` but prevents deep nesting |
| `mp_depth` | 1 | Single Modus Ponens round |
| `breadth_max` | 4 | Limits new statements per step |
| `total_max` | 4 | Caps total candidates for determinism |
| `axiom_instances` | **0** | **CRITICAL**: No axiom seeding, use seeds only |
| `formula_pool` | 8 | Small enumeration pool |
| `lean_timeout_s` | 0.001 | Effectively disables Lean (too short to succeed) |

### 3.2 Seed Statements

| Expression | Normalized | Rule | Purpose |
|------------|------------|------|---------|
| `p` | `p` | `seed:atom` | Antecedent for MP |
| `(p->(q))` | `p->q` | `seed:implication` | Implication for MP |

### 3.3 Expected Derivation

```
p + (p -> q) ⊢ q
```

Where `q` is the **guaranteed non-tautology** that triggers abstention.

---

## 4. Canonical Metadata

All metadata is normalized for deterministic serialization:

### 4.1 Parent Hash Ordering

Parent hashes are **always sorted** for canonical ordering:

```python
def _canonical_parents(parents: Tuple[str, ...]) -> Tuple[str, ...]:
    return tuple(sorted(parents))
```

### 4.2 Pretty Form

Pretty forms are **derived from normalized forms**, not stored separately:

```python
def _canonical_pretty(normalized: str) -> str:
    return normalize_pretty(normalized)
```

### 4.3 Statement Fingerprint

Each statement has a deterministic fingerprint including provenance:

```python
def _statement_fingerprint(normalized: str, parents: Tuple[str, ...]) -> str:
    canonical_parents = _canonical_parents(parents)
    payload = f"{normalized}|{','.join(canonical_parents)}"
    return hashlib.sha256(payload.encode("ascii")).hexdigest()[:16]
```

---

## 5. Telemetry (v1.1.0)

### 5.1 DERIVATION_SUMMARY Format

Every call to `run_slice_for_test()` emits structured telemetry:

```json
{
  "telemetry_version": "1.1.0",
  "timestamp": "2025-11-27T12:00:00.000000+00:00",
  "slice": "first-organism-abstention-slice",
  "duration_ms": 1.23,
  "bounds_fingerprint": "abc123def456",
  "metrics": {
    "n_candidates": 1,
    "n_verified": 0,
    "n_abstain": 1,
    "abstention_rate": 1.0
  },
  "filtering": {
    "axioms_seeded": 0,
    "axioms_rejected": 0,
    "mp_candidates_rejected": 1,
    "depth_filtered": 0,
    "atom_filtered": 0,
    "duplicate_filtered": 0
  },
  "abstained": [
    {
      "hash": "abc123...",
      "normalized": "q",
      "pretty": "q",
      "method": "lean-disabled",
      "rule": "mp",
      "mp_depth": 1,
      "parents": ["hash1", "hash2"],
      "fingerprint": "fp123..."
    }
  ]
}
```

### 5.2 Log Line Format

A compact single-line format is also emitted:

```
DERIVATION_SUMMARY slice=first-organism-abstention-slice candidates=1 verified=0 abstain=1 duration_ms=1.23 abstain_rate=1.0000
```

### 5.3 Extended Filtering Metrics

| Metric | Description |
|--------|-------------|
| `axioms_seeded` | Axiom instances that passed verification |
| `axioms_rejected` | Axiom instances that failed verification |
| `mp_candidates_rejected` | MP-derived candidates that failed verification |
| `depth_filtered` | Candidates filtered by depth cap |
| `atom_filtered` | Candidates filtered by atom cap |
| `duplicate_filtered` | Candidates filtered as duplicates |

---

## 6. Verification Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│ Layer 1: Pattern Matching                                       │
│   - Instant recognition of known tautology schemata             │
│   - e.g., p -> p, p -> (q -> p), (p -> (q -> r)) -> ...        │
│   - Result: verified=True, method="pattern"                     │
├─────────────────────────────────────────────────────────────────┤
│ Layer 2: Truth Table Evaluation                                 │
│   - Deterministic O(2^n) check over all valuations              │
│   - If all valuations yield True → tautology                    │
│   - Result: verified=True, method="truth-table"                 │
├─────────────────────────────────────────────────────────────────┤
│ Layer 3: Lean Fallback (Optional)                               │
│   - Enabled via ML_ENABLE_LEAN_FALLBACK=1                       │
│   - Timeout controlled by lean_timeout_s                        │
│   - If disabled: verified=False, method="lean-disabled"         │
│   - If timeout: verified=False, method="lean-timeout"           │
└─────────────────────────────────────────────────────────────────┘
```

For the First Organism experiment:
- `q` fails Layer 1 (not a pattern)
- `q` fails Layer 2 (not a tautology)
- `q` hits Layer 3, which returns `lean-disabled`

---

## 7. Property-Based Tests

The abstention generator is validated by property-based tests using Hypothesis:

### 7.1 Core Invariants

| Test | Invariant |
|------|-----------|
| `test_first_organism_produces_abstention` | `result.has_abstention == True` |
| `test_abstention_method_is_expected` | All methods in `ABSTENTION_METHODS` |
| `test_guaranteed_non_tautology_is_abstained` | `q` is in abstained set |
| `test_multiple_runs_produce_same_result` | Deterministic hashes |

### 7.2 Hypothesis Properties

```python
@given(st.integers(min_value=1, max_value=5))
def test_multiple_iterations_deterministic(iterations: int):
    """Multiple iterations produce deterministic results."""
    ...

@given(st.sampled_from(["p", "q", "r", "s"]))
def test_single_atoms_are_non_tautologies(atom: str):
    """Single atoms are always non-tautologies."""
    ...
```

---

## 8. Code Paths

### 8.1 Configuration

```
derivation/pipeline.py
├── make_first_organism_derivation_config()  → FirstOrganismDerivationConfig
├── make_first_organism_derivation_slice()   → CurriculumSlice
├── make_first_organism_seed_statements()    → Tuple[StatementRecord, ...]
├── FirstOrganismDerivationConfig
│   ├── slice_cfg: CurriculumSlice
│   ├── seed_statements: Tuple[StatementRecord, ...]
│   ├── expected_abstention_reason: str
│   ├── guaranteed_non_tautology: str  # "q"
│   ├── abstention_method: str         # "lean-disabled"
│   └── validate() → bool
└── run_slice_for_test()                     → DerivationResult
```

### 8.2 Canonical Helpers

```
derivation/pipeline.py
├── _canonical_parents(parents) → Tuple[str, ...]
├── _canonical_pretty(normalized) → str
├── _statement_fingerprint(normalized, parents) → str
└── _is_guaranteed_non_tautology(normalized) → bool
```

### 8.3 Tests

```
tests/test_abstention_determinism.py
├── TestAbstentionGuarantee
│   ├── test_first_organism_config_is_valid
│   ├── test_first_organism_produces_abstention
│   ├── test_abstention_method_is_expected
│   ├── test_abstention_guaranteed_property
│   └── test_guaranteed_non_tautology_is_abstained
├── TestDeterminism
│   ├── test_multiple_runs_produce_same_result
│   ├── test_summary_json_is_deterministic
│   └── test_seed_statements_are_deterministic
├── TestCanonicalMetadata
│   ├── test_parents_are_sorted
│   ├── test_pretty_is_derived_from_normalized
│   ├── test_fingerprint_is_deterministic
│   └── test_hash_matches_normalized
├── TestTelemetry
│   ├── test_summary_has_required_fields
│   ├── test_abstained_statements_have_required_fields
│   ├── test_json_is_valid
│   └── test_log_line_format
└── TestPropertyBasedDeterminism (Hypothesis)
    ├── test_multiple_iterations_deterministic
    ├── test_single_atoms_are_non_tautologies
    ├── test_canonical_parents_is_sorted
    ├── test_normalize_is_idempotent
    └── test_fingerprint_is_deterministic
```

---

## 9. AST Normalization (Future FOL)

The pipeline includes hooks for future First-Order Logic support:

```python
class NormalizationStrategy(Enum):
    PROPOSITIONAL = "pl"      # Current
    FIRST_ORDER = "fol"       # Future
    HIGHER_ORDER = "hol"      # Future

@dataclass(frozen=True)
class ASTNormalizationConfig:
    strategy: NormalizationStrategy = NormalizationStrategy.PROPOSITIONAL
    
    # Propositional options
    flatten_associative: bool = True
    sort_commutative: bool = True
    deduplicate_idempotent: bool = True
    
    # FOL options (future)
    alpha_normalize: bool = False
    skolemize: bool = False
    prenex_normal_form: bool = False
```

---

## 10. Running the Experiment

### 10.1 Standalone (No DB)

```bash
uv run pytest tests/test_abstention_determinism.py -v
```

### 10.2 Property-Based Tests

```bash
uv run pytest tests/test_abstention_determinism.py::TestPropertyBasedDeterminism -v
```

### 10.3 Integration Test

```bash
uv run pytest tests/integration/test_first_organism.py::test_first_organism_closed_loop_standalone -v
```

---

## 11. Invariants

| ID | Invariant | Assertion |
|----|-----------|-----------|
| **I1** | Abstention guaranteed | `result.has_abstention == True` |
| **I2** | Method is expected | `method in ABSTENTION_METHODS` |
| **I3** | Parents sorted | `parents == tuple(sorted(parents))` |
| **I4** | Pretty canonical | `pretty == normalize_pretty(normalized)` |
| **I5** | Fingerprint deterministic | `fp1 == fp2` for same inputs |
| **I6** | Summary JSON valid | `json.loads(summary.to_json())` succeeds |
| **I7** | Hₜ recomputable | `compute_composite_root(Rₜ, Uₜ) == stored_Hₜ` |

---

## 12. Troubleshooting

### 12.1 No Abstention Observed

**Symptom:** `has_abstention == False`

**Causes:**
1. `axiom_instances > 0` — axioms are being seeded and verified
2. Seeds not correctly normalized
3. MP not firing (antecedent not in known set)

**Fix:** Ensure `axiom_instances=0` in slice params.

### 12.2 Wrong Verification Method

**Symptom:** `method == "lean"` instead of `lean-disabled`

**Cause:** `ML_ENABLE_LEAN_FALLBACK=1` is set in environment.

**Fix:** Unset the env var.

### 12.3 Non-Deterministic Results

**Symptom:** Different hashes on repeated runs

**Cause:** Non-deterministic ordering in pipeline.

**Fix:** Ensure all parent tuples are sorted via `_canonical_parents()`.

---

## 13. References

- **Whitepaper:** MathLedger Symbolic Descent (Section 4.2)
- **Code:** `derivation/pipeline.py`, `derivation/verification.py`
- **Tests:** `tests/test_abstention_determinism.py`
- **Integration:** `tests/integration/test_first_organism.py`
- **Attestation:** `attestation/dual_root.py`

---

*This document is the authoritative specification for the First Organism abstention experiment. All changes must be reviewed by the Derivation Configurator & Abstention Engineer.*
