# Field Manual

**Status:** Working constraint artifact (not marketing documentation)

The Field Manual (fm.tex/fm.pdf) is an internal document that tracks obligations, gaps, and the reasoning behind version promotions. It is published here for transparency, not as polished documentation.

## Why It Exists

The Field Manual serves as an **obligation ledger**:

1. **Surfaces gaps** — Documents what the system cannot yet do
2. **Drives promotions** — Version upgrades require addressing FM obligations
3. **Prevents drift** — Written commitments constrain future behavior
4. **Auditor artifact** — External reviewers can verify claims against FM entries

## What It Is NOT

- Not a user guide
- Not marketing material
- Not rewritten for readability
- Not guaranteed to be current (check version date)

## Downloads

- [fm.pdf](fm.pdf) — Compiled Field Manual (PDF)
- [fm.tex](fm.tex) — LaTeX source

## How Auditors Should Use This

1. **Cross-reference claims** — If a feature is claimed, check FM for caveats
2. **Look for "TODO" and "OBLIGATION"** — These are explicit acknowledgments of gaps
3. **Compare versions** — FM changes between versions show what was addressed
4. **Trust gaps over features** — The gaps we document are more honest than features we claim

## Relationship to Other Docs

| Document | Purpose |
|----------|---------|
| Scope Lock | What this version does/doesn't demonstrate |
| Invariants | Tier A/B/C enforcement status |
| Field Manual | Obligation ledger driving version promotions |
| Hostile Rehearsal | Prepared answers to skeptical questions |

The Field Manual is the "why we promoted" document; other docs are "what we claim now."
