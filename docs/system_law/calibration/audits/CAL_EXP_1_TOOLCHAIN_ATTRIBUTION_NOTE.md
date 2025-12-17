# CAL-EXP-1 Toolchain Attribution Note

**Status**: CANONICAL
**Date**: 2025-12-13
**Author**: Claude X (Toolchain Snapshot & Reproducibility Engineer)

---

## Purpose

This document records the toolchain provenance situation for CAL-EXP-1 and CAL-EXP-2, establishing what claims can and cannot be made about toolchain parity.

## Recorded State

### CAL-EXP-1 (Instrument Calibration)

CAL-EXP-1 was executed before the full toolchain snapshot infrastructure was implemented.

**What was recorded:**
- `toolchain_hash`: `d088f20824a5bbc4cd1bf5f02d34a6758752363f417bed1a99970773b8dacfdc`

**This hash represents:**
- SHA-256 of `uv.lock` only

**What was NOT recorded:**
- Lean toolchain version hash
- Lake manifest hash
- Lakefile hash
- Combined toolchain fingerprint

### CAL-EXP-2 (P4 Divergence Minimization)

CAL-EXP-2 was executed after the full toolchain snapshot infrastructure was implemented.

**What was recorded:**
- `toolchain_fingerprint`: `b828a2185e017e172db966d3158e8e2b91b00a37f0cd7de4c4f7cf707130a20a`
- `uv_lock_hash`: `d088f20824a5bbc4cd1bf5f02d34a6758752363f417bed1a99970773b8dacfdc`
- `lean_toolchain_hash`: `410d5c912b1a040c79883f5e0bb55e733888534e2006eefe186e631c24864546`
- `lake_manifest_hash`: `f13722c8f13f52ef06e5fc123ba449287887018f2b071ad4da2d8f580045dd3e`

---

## Git History Analysis

### Commit Range Check

**Relevant commits between CAL-EXP-1 and CAL-EXP-2:**
```
caf6f70 docs(audit): CAL-EXP-2 freeze integrity report
f160c08 docs(cal-exp-2): add freeze attestation for P4 divergence experiment
16a828a docs(audit): add CAL-EXP-1 replication attestation and report
55bd258 fix(tda): track TDA, ht, synthetic, rfl modules and CAL-EXP-1 harness
854e661 feat: Add TDA windowed patterns hook to status generator
...
f7f80e9 Initial commit
```

### Lean Toolchain File Changes

**Query:**
```bash
git log --oneline f7f80e9..HEAD -- backend/lean_proj/lean-toolchain backend/lean_proj/lake-manifest.json
```

**Result:** (empty)

**Interpretation:** No commits between the initial commit and HEAD have modified `lean-toolchain` or `lake-manifest.json`.

### File Stability Evidence

| File | Last Modified | Commit |
|------|---------------|--------|
| `backend/lean_proj/lean-toolchain` | Initial | `f7f80e9` |
| `backend/lean_proj/lake-manifest.json` | Initial | `f7f80e9` |
| `uv.lock` | — | Hash matches between experiments |

---

## Attribution Claims

### What CAN Be Claimed

1. **uv.lock hash matches between CAL-EXP-1 and CAL-EXP-2**
   - CAL-EXP-1: `d088f208...`
   - CAL-EXP-2: `d088f208...`
   - Verdict: IDENTICAL

2. **No git changes to Lean toolchain files between experiments**
   - `lean-toolchain`: unchanged since initial commit
   - `lake-manifest.json`: unchanged since initial commit
   - Evidence: `git log` query returns empty

3. **CAL-EXP-2 divergence delta is attributable to parameter tuning**
   - Given: uv.lock matches AND Lean files unchanged in git
   - Therefore: toolchain is stable (high confidence)
   - Therefore: δp change is due to LR config, not toolchain drift

### What CANNOT Be Claimed

1. **"Full toolchain fingerprint match for CAL-EXP-1"**
   - CAL-EXP-1 did not record the full fingerprint
   - We cannot retroactively compute what fingerprint it would have had
   - We can only infer stability from git history

2. **"Cryptographic proof of identical toolchain"**
   - CAL-EXP-1's partial recording prevents cryptographic certainty
   - The claim is inferential, not cryptographic

---

## Recommendation

For future experiments:

1. All experiments MUST call `capture_toolchain_snapshot()` before execution
2. The full `toolchain_fingerprint` MUST be recorded in the manifest
3. Experiments with missing toolchain data should be flagged as "partial provenance"

---

## Artifact References

| Artifact | Location | Status |
|----------|----------|--------|
| CAL-EXP-1 manifest | `results/cal_exp_1/cal_exp_1_manifest.json` | Partial (uv.lock only) |
| CAL-EXP-2 manifest | `docs/system_law/calibration/audits/cal_exp_2_toolchain_manifest.json` | Full |
| Toolchain baseline | `toolchain_baseline.json` | Full |

---

## Conclusion

CAL-EXP-2 divergence minimization results are **attributable to parameter tuning** based on:
- Direct evidence: uv.lock hash match
- Indirect evidence: git history shows no Lean toolchain changes

This attribution is **high confidence but not cryptographically proven** for CAL-EXP-1 due to its partial toolchain recording.

---

*This note closes the "CAL-EXP-1 didn't have full toolchain" gap without making unsupported retroactive claims.*
