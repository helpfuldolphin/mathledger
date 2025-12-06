# First Organism Derivation Specification

**Version:** 1.0.0
**Created:** 2025-11-29
**Status:** SPEC / CONFIG DOCUMENTATION

---

> **IMPORTANT: Implementation Status**
>
> This document and the accompanying YAML files (`first_organism_slice.yaml`,
> `first_organism_seeds.yaml`) are **configuration documentation only**.
>
> **The live FO experiment uses Python-hardcoded configuration:**
> - `make_first_organism_derivation_config()` in `derivation/pipeline.py`
> - `make_first_organism_seed_statements()` in `derivation/pipeline.py`
>
> **YAML loading is NOT YET WIRED into the runtime.**
>
> The YAML values match the Python implementation (see comparison table below),
> but the Python code is the source of truth for Phase I Evidence Pack runs.
>
> **PHASE-I RFL NOTE:**
> All RFL runs (50-cycle baseline, 330-cycle experiments) use Python-only
> configuration. Neither the baseline nor RFL experiments read these YAML files.
> These files are documentation only and are NOT used in Phase-I Evidence Pack.

---

## Overview

This document specifies the canonical derivation configuration for the **First Organism (FO)** abstention experiment. The FO experiment is a controlled test that validates MathLedger's ability to:

1. Execute deterministic derivation under tight bounds
2. Produce verifiable abstention telemetry
3. Generate attestation artifacts for downstream certification

The configuration is documented in two YAML files (spec only, not runtime):
- `first_organism_slice.yaml`: Slice bounds and gate configuration
- `first_organism_seeds.yaml`: Pre-seeded statements for MP derivation

## Design Rationale

### Why a Controlled Slice?

The First Organism test is not a stress test or exploration. It is a **controlled experiment** that must produce **predictable, reproducible results**. This requires:

1. **Minimal search space**: Tight bounds prevent combinatorial explosion
2. **Deterministic seeds**: Pre-defined statements ensure identical derivation paths
3. **Disabled external dependencies**: Lean fallback is disabled for hermetic execution
4. **Guaranteed abstention**: The slice is designed to always produce at least one abstention

### Why Guaranteed Abstention?

Abstention is the core signal that FO validates. It proves that:

1. The verifier correctly rejects non-tautologies
2. Telemetry captures abstained statements with full provenance
3. The pipeline distinguishes verified statements from abstentions
4. Attestation artifacts include abstention metrics

Without guaranteed abstention, the FO test would be non-deterministic and unsuitable for regression testing.

## Current Implementation vs YAML Config

The following table shows the correspondence between the **active Python implementation**
(used in Phase I FO runs) and the **YAML documentation** (spec only).

### Slice Bounds Comparison

| Parameter | Python (ACTIVE) | YAML (SPEC) | Status |
|-----------|-----------------|-------------|--------|
| `max_atoms` | 2 | 2 | MATCH - Not used in Phase-I RFL |
| `max_formula_depth` | 2 | 2 | MATCH - Not used in Phase-I RFL |
| `max_mp_depth` | 1 | 1 | MATCH - Not used in Phase-I RFL |
| `max_breadth` | 4 | 4 | MATCH - Not used in Phase-I RFL |
| `max_total` | 4 | 4 | MATCH - Not used in Phase-I RFL |
| `max_axiom_instances` | 0 | 0 | MATCH - Not used in Phase-I RFL |
| `max_formula_pool` | 8 | 8 | MATCH - Not used in Phase-I RFL |
| `lean_timeout_s` | 0.001 | 0.001 | MATCH - Not used in Phase-I RFL |
| Gate configs | Hardcoded in Python | Defined in YAML | NOT ACTIVE - Not used in Phase-I RFL |

### Seed Statements Comparison

| Seed | Python (ACTIVE) | YAML (SPEC) | Status |
|------|-----------------|-------------|--------|
| Atom `p` | `"p"` | `"p"` | MATCH - Not used in Phase-I RFL |
| Implication | `"(p->(q))"` | `"(p->(q))"` | MATCH - Not used in Phase-I RFL |
| Rule (p) | `"seed:atom"` | `"seed:atom"` | MATCH - Not used in Phase-I RFL |
| Rule (p→q) | `"seed:implication"` | `"seed:implication"` | MATCH - Not used in Phase-I RFL |

**Source of Truth**: `derivation/pipeline.py` lines 739-894

---

## Parameter Choices

### Slice Bounds

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `max_atoms` | 2 | Minimal set (p, q) for interesting MP. 2^2 = 4 valuations for instant truth-table. |
| `max_formula_depth` | 2 | Allows `(p -> q)` but prevents deep nesting. Minimal for MP derivation. |
| `max_mp_depth` | 1 | Single MP round derives q from {p, p→q}. More rounds unnecessary. |
| `max_breadth` | 4 | Tight cap for deterministic behavior. Only a few statements expected. |
| `max_total` | 4 | Same as breadth for single-step runs. Prevents runaway derivation. |
| `max_axiom_instances` | 0 | **Critical**: Disables axiom seeding. Only explicit seeds participate. |
| `max_formula_pool` | 8 | Small pool for minimal memory. Mostly unused since axiom_instances=0. |
| `lean_timeout_s` | 0.001 | **Critical**: Effectively disables Lean. Forces "lean-disabled" abstention. |

### Seed Statements

| Expression | Normalized | Rule | Purpose |
|------------|------------|------|---------|
| `p` | `p` | `seed:atom` | Antecedent for MP |
| `(p -> q)` | `(p->(q))` | `seed:implication` | Major premise for MP |

### Expected Derivation

From seeds `{p, (p -> q)}`, Modus Ponens derives:

```
p, (p -> q)  ⊢  q
```

The formula `q` is a bare propositional atom, which is **not a tautology**.

### Why `q` Is Not a Tautology

A tautology must be true under **all** valuations. For `q`:

| q | Formula Value |
|---|---------------|
| T | T |
| F | F |

Since valuation `{q = F}` makes the formula false, `q` is not a tautology.

### Verification Cascade

When the pipeline attempts to verify `q`:

1. **Pattern Matching**: `q` matches no known tautology schema (K, S, etc.)
2. **Truth-Table**: Evaluates to `{T, F}` → not a tautology → **rejected**
3. **Lean Fallback**: Disabled (timeout=0.001s) → returns `method="lean-disabled"`

**Result**: Statement `q` is recorded as abstained with:
- `verification_method = "lean-disabled"`
- `rule = "mp"`
- `mp_depth = 1`
- `parents = [hash(p), hash(p -> q)]`

## Abstention Guarantee Proof

**Theorem**: The First Organism slice configuration, when run with the specified seeds, **always** produces at least one abstention.

**Proof**:

1. **Premise**: Seeds are `p` and `(p -> q)` with `axiom_instances = 0`.
2. **Derivation**: MP applies because:
   - `p` exists in the working set
   - `(p -> q)` is an implication with antecedent `p`
   - Consequent `q` is not in the working set
3. **Candidate Generation**: The pipeline generates candidate `q` with:
   - `normalized = "q"`
   - `mp_depth = 1`
   - `parents = (hash(p), hash(p -> q))`
4. **Verification**:
   - Pattern: No match (bare atom)
   - Truth-table: `q` is not a tautology (false when q=F)
   - Lean: Disabled → returns `(False, "lean-disabled")`
5. **Recording**: `q` is added to `abstained_candidates` list.
6. **Telemetry**: `DerivationSummary.n_abstain >= 1`

**QED**: By construction, the slice always produces at least one abstention. ∎

## Gate Configuration

The slice gates are configured to **not block** on abstention:

| Gate | Setting | Purpose |
|------|---------|---------|
| `coverage.ci_lower_min` | 0.01 | Minimal coverage requirement |
| `abstention.max_rate_pct` | 100.0 | Allow full abstention |
| `abstention.max_mass` | 1000 | Allow many abstentions |
| `velocity.min_pph` | 0.01 | Minimal velocity requirement |
| `caps.min_attempt_mass` | 1 | At least one attempt |

These settings ensure the FO test passes even when most candidates are abstained.

## Integration Points

### Pipeline Integration

The seeds and slice are loaded via:

```python
from derivation.pipeline import (
    make_first_organism_derivation_config,
    make_first_organism_seed_statements,
    run_slice_for_test,
)

# Get canonical config
config = make_first_organism_derivation_config()

# Run derivation
result = run_slice_for_test(
    config.slice_cfg,
    existing=config.seed_statements,
    limit=1,
)

# Validate abstention
assert result.has_abstention
assert result.n_abstained >= 1
assert any(s.normalized == "q" for s in result.abstained_candidates)
```

### Test Fixture

The primary test is:
```
tests/integration/test_first_organism.py::test_first_organism_closed_loop_happy_path
```

Enable with:
```bash
FIRST_ORGANISM_TESTS=true pytest -m first_organism
```

### Telemetry Output

On successful FO run:
```
DERIVATION_SUMMARY={"slice":"first-organism-slice","n_candidates":1,"n_verified":0,"n_abstain":1,...}
DERIVATION_SUMMARY slice=first-organism-slice candidates=1 verified=0 abstain=1 ...
[PASS] FIRST ORGANISM ALIVE
```

## Determinism Requirements

| Aspect | Requirement |
|--------|-------------|
| RNG Seed | Fixed at 42 (reserved for future stochastic ops) |
| Hash Algorithm | SHA-256 |
| Normalization | Propositional ASCII canonical form |
| Timestamp | Deterministic from content (hermetic) |
| Lean Kernel | Not required (disabled) |
| External I/O | None (hermetic) |

## Security Considerations

The FO test runs under `RUNTIME_ENV=test_hardened`:

1. **No production credentials**: Uses dedicated test database
2. **Rate limiting**: High limits for integration tests
3. **Isolated queue**: Separate Redis queue key
4. **No Lean process**: No subprocess execution

See `config/first_organism.env.template` for secure environment setup.

## Phase I vs Phase II

### Phase I (Implemented - Evidence Pack v1)

The following are **actually implemented and used** in Evidence Pack v1 FO runs:

- `make_first_organism_derivation_config()` - Python function (ACTIVE)
- `make_first_organism_seed_statements()` - Python function (ACTIVE)
- `run_slice_for_test()` - Pipeline entry point (ACTIVE)
- `test_first_organism_closed_loop_happy_path` - Integration test (ACTIVE)
- `artifacts/first_organism/attestation.json` - Attestation artifact (ACTIVE)
- FO closed-loop logs in `fo_baseline/` and `fo_rfl/` directories (ACTIVE)

### Phase II (Future Work - Not Yet Implemented)

The following are **documented but not wired into runtime**:

- `first_organism_slice.yaml` - YAML slice config (SPEC ONLY)
- `first_organism_seeds.yaml` - YAML seed config (SPEC ONLY)
- YAML loader functions - Not implemented
- External config file parsing - Not implemented
- Gate configuration from YAML - Not implemented

**Do not claim that FO config comes from YAML in any Phase I documentation.**

---

## Phase II Uplift-Capable Derivation Regime

> **STATUS: PHASE-II DESIGN ONLY - NOT IMPLEMENTED**
>
> This section describes a future derivation regime designed to produce **mixed
> success/abstain patterns** suitable for measuring RFL uplift. None of this is
> implemented in Phase I. This is a design target for code agents.

### Problem: Phase I Degeneracy

The Phase I FO slice produces **100% abstention** because:

1. Seeds `{p, (p -> q)}` only yield MP derivation of `q`
2. `q` is a bare atom → not a tautology → abstained
3. `axiom_instances = 0` prevents any K/S axiom seeding
4. Result: `n_verified = 0`, `n_abstain = 1` on every run

This is useful for testing abstention telemetry, but **useless for measuring RFL uplift**
because there's no variance in the success/abstain ratio.

### Goal: Mixed Success/Abstain Pattern

For RFL to show measurable uplift, we need a derivation regime that produces:

- **Target abstention band**: 20–70%
- **Mix of outcomes**: Some tautologies verified, some non-tautologies abstained
- **Deterministic**: Same seeds + bounds → same success/abstain counts

### Proposed Phase II Uplift Slice

#### Slice Parameters

| Parameter | Phase I (Degenerate) | Phase II (Uplift) | Rationale |
|-----------|---------------------|-------------------|-----------|
| `max_atoms` | 2 | 3 | More atoms → richer formula space |
| `max_formula_depth` | 2 | 3 | Deeper formulas → more MP opportunities |
| `max_mp_depth` | 1 | 2 | Two MP rounds → longer derivation chains |
| `max_breadth` | 4 | 32 | Wider exploration per step |
| `max_total` | 4 | 64 | More total candidates |
| `max_axiom_instances` | 0 | 16 | **Critical**: Enable K/S axiom seeding |
| `max_formula_pool` | 8 | 32 | Larger substitution pool |
| `lean_timeout_s` | 0.001 | 0.001 | Keep Lean disabled for determinism |

#### Why This Produces Mixed Outcomes

1. **K Axiom instances** (e.g., `p -> (q -> p)`) are **tautologies** → verified via truth-table
2. **S Axiom instances** (e.g., `(p -> (q -> r)) -> ((p -> q) -> (p -> r))`) are **tautologies** → verified
3. **MP derivations from axioms** may produce tautologies (if consequent is tautology) or non-tautologies
4. **Non-tautology consequents** (like bare atoms or contingent implications) → abstained

Example derivation trace:
```
AXIOM K[p,q]: p -> (q -> p)              → VERIFIED (tautology)
AXIOM K[q,p]: q -> (p -> q)              → VERIFIED (tautology)
SEED: p                                   → SEED (given)
MP: p, (p -> (q -> p)) ⊢ (q -> p)        → VERIFIED (tautology: q -> p when p=T)
                                           Wait, (q -> p) is NOT a tautology!
                                           When q=T, p=F: (T -> F) = F
                                           → ABSTAINED
```

Actually, let me recalculate. `(q -> p)` is NOT a tautology (false when q=T, p=F).
So even with axiom seeding, MP can derive non-tautologies.

#### Verifier Mode Requirements

| Verifier Mode | Expected Pattern | Suitable for Uplift? |
|---------------|------------------|---------------------|
| Truth-table only (Lean disabled) | ~30-50% abstention | **Yes** - deterministic |
| Truth-table + Lean enabled | ~10-30% abstention | Possible but non-hermetic |

**Recommendation**: Use truth-table only mode for Phase II uplift experiments.
This keeps runs hermetic while allowing measurable abstention variance.

### Proposed Phase II Uplift Seeds

Instead of the degenerate `{p, (p -> q)}` seeds, use a mix that enables both
tautology verification and non-tautology abstention:

| Seed | Type | Expected Outcome |
|------|------|------------------|
| `p -> (q -> p)` | K axiom instance | VERIFIED (tautology) |
| `(p -> (q -> r)) -> ((p -> q) -> (p -> r))` | S axiom instance | VERIFIED (tautology) |
| `p` | Atom | SEED (given) |
| `(p -> q)` | Contingent implication | SEED (given) |
| `q` | Atom | SEED (given) |

With these seeds and `max_mp_depth = 2`, MP can derive:
- From `p` and `p -> (q -> p)`: derive `(q -> p)` → NOT a tautology → ABSTAINED
- From `q` and K axiom: derive `(p -> q)` → NOT a tautology → ABSTAINED
- From S axiom and other implications: may derive tautologies or non-tautologies

**Expected abstention rate**: 30-50% depending on exact derivation order.

### Implementation Checklist for Code Agents

To implement Phase II uplift regime:

1. **Add YAML loader** to `derivation/pipeline.py`:
   ```python
   def load_slice_from_yaml(path: Path) -> CurriculumSlice:
       """Load slice config from YAML file."""
       ...
   ```

2. **Add environment toggle**:
   ```bash
   FO_CONFIG_SOURCE=yaml  # or 'python' (default)
   FO_SLICE_YAML=derivation/first_organism_slice.yaml
   ```

3. **Wire YAML into FO harness**:
   - Modify `make_first_organism_derivation_config()` to check `FO_CONFIG_SOURCE`
   - If `yaml`, load from `FO_SLICE_YAML` using `phase_ii_uplift_slice` section
   - If `python`, use existing hardcoded config

4. **Add test for mixed outcomes**:
   ```python
   def test_phase_ii_uplift_mixed_outcomes():
       """Phase II: Verify mixed success/abstain pattern."""
       config = load_uplift_config()
       result = run_slice_for_test(config.slice_cfg, existing=config.seeds)
       assert result.n_verified > 0, "Expected some tautologies verified"
       assert result.n_abstained > 0, "Expected some non-tautologies abstained"
       abstention_rate = result.n_abstained / (result.n_verified + result.n_abstained)
       assert 0.20 <= abstention_rate <= 0.70, f"Abstention {abstention_rate:.0%} outside target band"
   ```

5. **Update RFL runner** to use Phase II config for uplift experiments.

### Migration Path

| Step | Action | Blocks |
|------|--------|--------|
| 1 | Implement YAML loader | None |
| 2 | Add `phase_ii_uplift_slice` to YAML | Step 1 |
| 3 | Add environment toggle | Step 1 |
| 4 | Wire toggle into FO harness | Steps 1-3 |
| 5 | Run Phase II uplift experiments | Step 4 |
| 6 | Validate abstention band | Step 5 |
| 7 | Update RFL runner for uplift | Step 6 |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-11-29 | Initial canonical specification |
| 1.0.1 | 2025-11-30 | Added Phase I/II status markers and implementation comparison table |
| 1.1.0 | 2025-11-30 | Added Phase II Uplift-Capable Derivation Regime design |

## Related Documents

- `derivation/first_organism_slice.yaml`: Slice bounds configuration (SPEC ONLY)
- `derivation/first_organism_seeds.yaml`: Seed statement definitions (SPEC ONLY)
- `derivation/pipeline.py`: Pipeline implementation (ACTIVE - source of truth)
- `derivation/verification.py`: Verifier implementation (ACTIVE)
- `tests/integration/test_first_organism.py`: Integration test (ACTIVE)
- `docs/FIRST_ORGANISM_LOCAL_DEV.md`: Local development guide
