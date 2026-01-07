# Ariadne Faithfulness Pressure

**Status:** PARKED — Observer-only pressure signal; not admissible as verifier
**Classification:** ADV (Advisory)
**Phase:** II+ (not in scope for Phase I)
**Not included in fm.tex**

---

## Decision

**STATUS: Park Ariadne (observer-only) — not admissible as verifier until evaluation contract pinned.**

Re-open only if upstream ships:
1. Frozen, hash-pinned evaluation dataset (500 queries)
2. Deterministic or fully-logged critic outputs (seed control or distributional envelope)
3. Explicit infrastructure requirements (minimum RPM/TPM, backoff contract)
4. Replay bundle artifact specification

---

## Tool Summary

**Project Ariadne (AriadneXAI):** XAI/faithfulness measurement instrument using causal interventions on intermediate reasoning + judge-based semantic similarity thresholding.

**Core mechanism:**
1. Generate counterfactual interventions on reasoning traces (critic LLM)
2. Measure semantic similarity between original and counterfactual answers (judge LLM)
3. Threshold similarity score to determine "faithfulness"

**Classification:** Measurement instrument (observer-only), not a verifier or closure system.

---

## Authoritative Technical Facts

| Field | Value | Source |
|-------|-------|--------|
| Judge prompt location | `semantic_scorer.py`, function `_llm_judge_similarity` | Sourena (author) |
| Judge model | `claude-3-7-sonnet-20250219` (version-pinned, intended immutable) | Sourena |
| Similarity threshold τ | 0.8 | Sourena |
| Minimum answer length | >10 characters | Sourena |
| Critic temperature | 0.8 (non-deterministic) | Sourena |
| Canonical dataset | NOT COMMITTED; repo provides 30-query template | Sourena |
| Expected dataset | 500 queries across 3 categories (user must expand) | Sourena |
| Repro command | `python experimental_setup.py` | Sourena |

---

## Internal Reproduction Attempt

**Result:** NOT REPRODUCED (numerical-claim level)

This verdict is scoped to numerical-claim reproducibility: the specific headline numbers from the paper could not be independently verified. Procedural executability may still hold given sufficient API infrastructure (adequate RPM/TPM) and a user-supplied 500-query dataset matching the paper's distribution.

**Blocking factors:**
1. **Missing dataset artifact:** `research_dataset_500.jsonl` not committed; no hash to pin.
2. **Infrastructure underspecification:** No explicit minimum RPM/TPM; MAX_CONCURRENT=10 caused 0 successful audits under low-tier API keys.
3. **Backoff reliability unclear:** Execution contract does not specify retry/backoff guarantees.
4. **Non-deterministic critic:** Temperature 0.8 produces different interventions on each run; no seed control.

**Conclusion:** Tool measures something, but what it measures is not replay-verifiable without additional contract pinning.

---

## Conceptual Resolution

| Concept | Ariadne | MathLedger |
|---------|---------|------------|
| Role | Measurement instrument | Governance substrate |
| Authority | None (observer-only) | Fail-closed, artifact-bound |
| Input requirements | Flexible (user provides) | Frozen or envelope-bounded |
| Reproducibility | Not guaranteed | Required for authority |

**Key insight:** Measurement ≠ Authority. A measurement tool can produce useful signals without those signals being admissible as governance evidence. Ariadne is the former; MathLedger demands the latter.

---

## Integration Criteria (Future)

For Ariadne to become an admissible measurement route inside MathLedger, the following must be satisfied:

### Required Pinned Fields

| Field | Current State | Required State |
|-------|---------------|----------------|
| Dataset | Not committed | Hash-pinned artifact |
| Critic temperature | 0.8 (stochastic) | Fixed with seed, OR distributional envelope logged |
| Judge model | Pinned (good) | Maintain version lock |
| Judge prompt | Located but mutable | Hash-pinned, immutable |
| Judge temperature | Unspecified | Must be pinned (preferably 0) |
| Infrastructure | Unspecified | Minimum RPM/TPM documented |
| Backoff | Unspecified | Explicit retry contract |
| Replay bundle | None | Full artifact package for deterministic replay |

### Minimum Viable Contract

```
DATASET_HASH      = sha256(research_dataset_500.jsonl)
JUDGE_MODEL       = claude-3-7-sonnet-20250219
JUDGE_PROMPT_HASH = sha256(judge_prompt_text)
JUDGE_TEMP        = 0.0
CRITIC_TEMP       = 0.8
CRITIC_SEED       = <fixed or logged>
TAU               = 0.8
MIN_RPM           = <specified>
BACKOFF_CONTRACT  = <specified>
```

---

## Failure Modes (If Integrated Prematurely)

1. **Dataset drift:** Different users run different 500-query sets; results not comparable.
2. **Critic variance:** Non-deterministic interventions produce different faithfulness scores on replay.
3. **Judge drift:** Model version changes without notice; scores shift.
4. **Infrastructure flakiness:** Rate limits cause partial runs; incomplete data treated as complete.
5. **Threshold brittleness:** τ=0.8 is arbitrary; small perturbations flip pass/fail.

---

## Recommended Posture

**Do not integrate now.**

Ariadne is a useful research probe for understanding faithfulness, but it does not meet MathLedger's admissibility requirements for governance-relevant evidence.

**Park conditions:**
- Treat as ADV/observer-only signal
- Do not admit outputs as authority-bearing artifacts
- Do not cite Ariadne scores in closure matrices

**Re-open conditions:**
- Upstream commits canonical dataset with hash
- Upstream documents infrastructure requirements
- Upstream provides replay bundle specification
- OR: MathLedger wraps Ariadne in a determinizing harness (seed control, output logging, envelope reporting)

---

## Generalizable Lesson

See: [Evaluation Contract Checklist](evaluation_contract_checklist.md)

Any external measurement tool seeking admission as governance evidence must satisfy the evaluation contract checklist. Ariadne currently fails on: dataset pinning, critic determinism, infrastructure specification, and replay bundle.

---

## Status

Awareness only.
Not a claim.
Not an obligation.
Not included in fm.tex.

**Activation condition:**
Only activate if:
- Upstream pins evaluation contract, OR
- MathLedger builds determinizing wrapper.

**Until then:**
This remains parked.
