# MathLedger Theory Packs

This document records the inductive progression of axiomatic systems used in MathLedger.
Each section defines:
- **Parent theory** (the base system extended)
- **New symbols/operations**
- **New axioms/inference rules**
- **Rationale** (why this extension matters)
- **Status** (planned / in-progress / completed)

---

## Stage A — Propositional Logic

**Parent:** None (base)

**Symbols:**
- Variables: p, q, r, ...
- Connectives: ∧, ∨, →, ¬

**Axioms:**
- Hilbert-style or natural deduction rules (impl, conj, disj, neg).
- Semantics: classical tautologies.

**Rationale:**
Foundation for statement normalization, hashing, and prover orchestration.
Provides first ~10k+ theorems for bootstrapping the ledger.

**Status:** In-progress (2025-09-07)

---

## Stage B — Equational Logic (Monoids)

**Parent:** Propositional Logic

**Symbols:**
- Constant: e
- Binary operation: *

**Axioms:**
- (x * y) * z = x * (y * z) (Associativity)
- e * x = x (Left identity)
- x * e = x (Right identity)

**Rationale:**
First algebraic structure, introduces rewrite-based reasoning in Lean.
Lemmas become reusable for all algebraic extensions.

**Status:** Planned

---

## Stage C — First-Order Logic with Equality (QF_UF)

**Parent:** Equational Logic

**Symbols:**
- Variables over domain
- Equality predicate =

**Axioms:**
- Reflexivity: ∀x, x = x
- Symmetry: x = y → y = x
- Transitivity: x = y ∧ y = z → x = z
- Substitution principle

**Rationale:**
Adds general equality reasoning; bridge from algebra to arithmetic.

**Status:** Planned

---

## Stage D — Arithmetic Fragments (Peano Light)

**Parent:** FOL with Equality

**Symbols:**
- Constant: 0
- Function: S (successor)
- Function: +

**Axioms:**
- ∀x, x + 0 = x
- ∀x,y, x + S(y) = S(x + y)
- Peano successor axioms (injectivity, non-zero).

**Rationale:**
Foundation of number theory, induction schemas for generalization.

**Status:** Planned

---

## Stage E — Category Theory Mini-Fragment

**Parent:** FOL with Equality

**Symbols:**
- Objects, morphisms
- Identity, composition

**Axioms:**
- Associativity of composition
- Identity laws

**Rationale:**
Step into structural/higher-order mathematics; relevant for ML labs (graphs, categories, type theory).

**Status:** Planned

---
