# Org Policy Alignment Pressure

**Status:** Parked (Phase II+ consideration)
**Not included in fm.tex**

---

## Pressure

- "Org-specific denylist compliance is systematically weak; evaluation needs to be versioned + replayable."
- Current org policy enforcement is opaque, non-reproducible, and unauditable
- No artifact trail exists for what policy was applied, when, or how

---

## Integration Idea (Phase II+)

Add a PA verifier route like `ORG_POLICY_CHECK_v1` with:

- **Policy bundle hash** — immutable reference to the policy version applied
- **Judge model+prompt hash** — if LLM-judged, hash of (model_id, prompt_template, version)
- **Confidence / disagreement signals** — explicit uncertainty reporting
- **Mandatory allowed outcomes:**
  - PASS
  - FAIL
  - ABSTAIN
  - ESCALATE

---

## Design Constraints

- All policy checks must be replayable given the same inputs and policy bundle
- Policy bundle updates must increment version and produce new hash
- Judge model changes invalidate prior comparisons (version break)

---

## Explicit Non-Claim

"This does not prove correctness; it proves what policy boundary was applied and how."

- We do not claim the policy is correct
- We do not claim the judge is correct
- We claim only: this input, this policy, this judge, this outcome, this artifact

---

## Open Questions

- How to handle policy bundle confidentiality (hash-only vs full disclosure)?
- How to version judge prompts without leaking evaluation criteria?
- What threshold triggers ESCALATE vs ABSTAIN?
- How to handle multi-policy composition (AND/OR/precedence)?

---

## Related: Professional-Duty Regimes

**Context:**
Beyond org-specific policies, some domains impose external professional duties (law, medicine, accounting) where verification scope is defined by regulation, case law, or professional standards—not by organizational preference.

**Implications:**
- Policy bundles may need to reference external duty frameworks (e.g., legal jurisdiction, professional licensing body)
- Verification scope is externally constrained; org cannot unilaterally reduce it
- Liability flows from professional duty, not just org policy
- ADV/PA routes do not satisfy professional verification requirements; only MV-equivalent routes may reduce verification cost

**Reference:**
See `verification_value_paradox_pressure.md` for analysis of verification cost in high-duty domains.

---

## Status

Awareness only.
Not a claim.
Not an obligation.
Not included in fm.tex.

**Activation condition:**
Only activate if:
- MathLedger addresses org-specific compliance as a first-class concern, AND
- A versioned policy bundle format is defined.

**Until then:**
This remains parked.
