# MathLedger Field Manual

**Status**: Working document. Do not rewrite into a "pristine" version yet.

## What This Document Is

The Field Manual (`fm.tex` / `fm.pdf`) is a **catalog of obligations** that the demo must eventually satisfy. It functions as a **pressure surface**: any claim made in the FM that the demo cannot currently enforce is either:

- A Tier B/C invariant (provable gap, documented)
- A specification error (FM claim is wrong, must be retracted)
- A demo deficiency (must be fixed before the claim becomes Tier A)

The FM is not marketing material. It is an internal ledger of what we said the system would do.

## Relationship to Demo/Site

```
FM (obligations)  →  Demo (executable slice)  →  Versions (monotone closure)
       ↓                      ↓                          ↓
  pressure surface     forces honesty            immutable artifact
```

- The demo/site is the **executable slice** of the FM's claims.
- Each release version is a **monotone closure** of enforced invariants.
- The FM creates pressure to close gaps; the demo creates evidence that gaps are closed.

## How to Use This FM with the Demo Flywheel

### Step 1: Find an Invariant
Pick any claim from the FM (e.g., "H_t = SHA256(R_t || U_t)").

### Step 2: Ask the Violation Question
> Can the current demo violate this invariant without leaving a replay-visible scar?

- **If NO**: The invariant is Tier A (enforced). The demo cannot lie about this.
- **If YES**: The invariant is Tier B or C (not enforced). Document the gap.

### Step 3: Propose Minimal Promotion
If the invariant is Tier B/C, propose the **minimal code change** that would make violation detectable. This is how invariants graduate from C → B → A.

### Step 4: Verify with Hostile Replay
After implementing the promotion, verify that:
1. A hostile auditor cannot forge a passing evidence pack
2. The invariant appears in `/demo/health` or equivalent audit surface
3. The test suite includes a regression test for the invariant

## When We Will Do a Pristine Rewrite

The FM will be rewritten into a clean, publishable document **only after**:

1. **Tier A saturation for Phase I**: All Phase I invariants are enforced in the demo, with test coverage and replay verification.

2. **Tier C explicitly Phase II+**: Any invariant that cannot be enforced in Phase I is either:
   - Retracted from the FM
   - Explicitly marked as Phase II+ scope
   - Promoted to Tier B with a clear path to A

3. **Convergence**: The FM and demo agree on what is enforced. No FM claim exists that the demo silently fails to satisfy.

Until these conditions are met, the FM remains a working document—a pressure surface, not a polished artifact.

## Files in This Directory

| File | Description |
|------|-------------|
| `fm.tex` | LaTeX source (canonical) |
| `fm.pdf` | Compiled output |
| `references.bib` | Bibliography |
| `README.md` | This file |

## Build Instructions

```bash
cd docs/field_manual
pdflatex fm.tex
bibtex fm
pdflatex fm.tex
pdflatex fm.tex
```

Or with latexmk:
```bash
latexmk -pdf fm.tex
```

---

**Do not** use this document in external communications until the pristine rewrite is complete.
