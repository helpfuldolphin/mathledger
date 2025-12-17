# Appendix: How to Interpret a CAL-EXP-3 Mismatch

## What a Mismatch Means

A mismatch occurs when a CAL-EXP-3 evidence pack differs from a golden reference manifest. The verification system computes cryptographic hashes of deterministic artifacts and compares them against known-good values.

**MATCH** = The evidence pack is bitwise-identical to the reference for all critical artifacts. The claim "this run produced ΔΔp = +0.032" is reproducible.

**MISMATCH** = At least one critical artifact differs. The run cannot be confirmed as equivalent to the reference. This does not mean fraud—it means the comparison is invalid.

---

## Why Mismatches Invalidate Comparison

CAL-EXP-3 measures uplift (ΔΔp) between two arms under controlled conditions. For this measurement to be meaningful, the following must be identical across comparable runs:

| Artifact | Why It Must Match |
|----------|-------------------|
| `baseline/cycles.jsonl` | Raw Δp values determine the baseline mean |
| `treatment/cycles.jsonl` | Raw Δp values determine the treatment mean |
| `corpus_manifest.json` | Different inputs → different outputs → incomparable |
| `toolchain_hash.txt` | Different runtime → potential behavioral divergence |

If any of these differ, the runs are **not the same experiment**. Comparing them would be like comparing clinical trial results from different patient populations.

**Conservative stance**: We reject comparison rather than risk false equivalence.

---

## Why Allowed-to-Differ Fields Are Excluded

Some fields vary between executions without affecting the scientific validity of the measurement:

| Field | Why Excluded |
|-------|--------------|
| `generated_at` | Wall-clock timestamp of execution |
| `run_id` | Unique identifier for logging/tracking |
| `timestamp` | Verifier execution time |
| `run_dir` | Filesystem path (machine-specific) |

These are **metadata about the execution**, not **data from the execution**. Including them would cause every run to mismatch trivially, defeating the purpose of determinism verification.

**Design principle**: Hash what matters, ignore what doesn't.

---

## Why This Strengthens the Evidence

A naive approach would compare only the final ΔΔp value. If two runs report ΔΔp = +0.032, they "match." This is weak evidence—it proves nothing about how the value was computed.

The golden manifest approach is stronger because it verifies:

1. **Input identity** — Same corpus, same seed, same toolchain
2. **Execution determinism** — Bitwise-identical cycle-by-cycle outputs
3. **Computation integrity** — Uplift derived from verified intermediate artifacts

An evaluator can confirm: "This evidence pack was produced by the same deterministic process as the reference, not by coincidence or manipulation."

**Analogy**: A financial audit doesn't just check that the final balance is correct—it traces every transaction. Similarly, we don't just check ΔΔp—we verify the entire artifact chain.

---

## Summary

| Scenario | Interpretation |
|----------|----------------|
| All artifacts MATCH | Evidence pack is reproducible and equivalent to reference |
| Any critical artifact MISMATCH | Comparison invalid; runs are not equivalent |
| Only allowed-to-differ fields differ | Expected; these are excluded from comparison |
| Invariant mismatch (seed, cycles, etc.) | Runs are fundamentally different experiments |

**Bottom line**: A MATCH verdict means the evidence is cryptographically tied to a known-good reference. A MISMATCH verdict means the comparison cannot be made—not that the run is invalid, but that it cannot be confirmed as equivalent.
