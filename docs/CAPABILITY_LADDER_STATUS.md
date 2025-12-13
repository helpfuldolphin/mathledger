# Capability Ladder Status

**Version:** 1.0.0
**Last Updated:** 2025-12-13
**Status:** DORMANT (Historical Design Artifact)

---

## Overview

The "Capability Ladder" refers to the original architectural vision for progressive complexity advancement in MathLedger:

```
Propositional Logic (PL)
    ↓
First-Order Logic with Equality (FOL=)
    ↓
Equational Theories (Group, Ring, Field)
    ↓
Linear Arithmetic
    ↓
Higher Theories...
```

This document explicitly clarifies the current status of this design to prevent confusion for collaborators.

---

## Current Status: DORMANT

The Capability Ladder is:
- **Documented**: Referenced in `docs/FieldManual/fm.tex` and historical design docs
- **Partially Implemented**: Database schema supports `theories` table with `logic` field
- **Not Actively Enforced**: No runtime gating between logic tiers
- **Not Deprecated**: Remains a valid long-term architectural goal

### Why Dormant?

The First Light calibration phase (P3/P4/P5) focuses on:
1. Establishing governance infrastructure (USLA/TDA)
2. Proving shadow mode stability
3. Validating dual-root attestation

Advancing the capability ladder requires:
1. Stable governance signals
2. Verified slice progression within each tier
3. Cross-tier verification infrastructure

The decision was made to stabilize governance before expanding the logic frontier.

---

## What IS Active (Slice System)

Within Propositional Logic (the current active tier), the **Slice System** is fully operational:

```python
# curriculum/gates.py - Active slice progression
SLICE_LADDER = [
    SliceSpec(max_atoms=4, max_depth=4, name="beginner"),
    SliceSpec(max_atoms=4, max_depth=5, name="intermediate_depth"),
    SliceSpec(max_atoms=5, max_depth=5, name="intermediate_atoms"),
    SliceSpec(max_atoms=5, max_depth=6, name="advanced"),
    ...
]
```

This provides bounded complexity progression **within** a logic tier, not **across** tiers.

---

## Future Reactivation Path

To reactivate the full Capability Ladder:

### Phase 1: Cross-Tier Infrastructure
- [ ] Implement `tier_gate.py` module for logic tier transitions
- [ ] Add `tier_id` to proof attestations
- [ ] Create cross-tier verification tests

### Phase 2: FOL= Integration
- [ ] Integrate Lean FOL axioms
- [ ] Add quantifier handling to normalizer
- [ ] Implement term unification

### Phase 3: Equational Theories
- [ ] Add theory-specific axiom modules
- [ ] Implement rewrite rule verification
- [ ] Create theory inheritance model

### Prerequisites
- [ ] First Light P5 completion (shadow mode graduation)
- [ ] Governance signal stability (H, rho, tau within bounds)
- [ ] UVI integration for tier advancement approval

---

## References

| Document | Relevance |
|----------|-----------|
| `docs/FieldManual/fm.tex` | Historical capability ladder description |
| `curriculum/gates.py` | Active slice system implementation |
| `migrations/001_init.sql` | `theories` table schema |
| `backend/axiom_engine/derive_core.py` | System-aware derivation |

---

## Decision Record

| Date | Decision | Rationale |
|------|----------|-----------|
| 2025-Q4 | Mark ladder DORMANT | Focus on governance stability before logic expansion |
| TBD | Reactivate ladder | After P5 completion and shadow mode graduation |

---

## Contact

For questions about capability ladder reactivation, consult:
- Repository maintainers
- `docs/FieldManual/fm.tex` for historical context
- VSD.md for current architectural priorities
