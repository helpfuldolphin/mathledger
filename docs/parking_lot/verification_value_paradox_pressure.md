# Verification-Value Paradox Pressure

**Status:** PARKED
**Classification:** ADV / pressure signal
**Phase:** II+ consideration
**Not included in fm.tex**

---

## Source

Yuvaraj, "The Verification-Value Paradox: A Normative Critique of Gen AI in Legal Practice" (arXiv:2510.20109)

---

## The Paradox

Net value from AI assistance is: **N = EG − V**, where EG is efficiency gain and V is verification cost. In high-duty domains (law, medicine, finance), verification is mandatory and scope is broad—not merely "citations exist" but accuracy, relevance, context, and fitness for purpose. When V approaches or exceeds EG, net value collapses or goes negative. The efficiency promise of generative AI is neutralized by the professional obligation to verify its outputs.

This is not a bug in AI adoption; it is a structural feature of domains where the duty of verification is non-negotiable.

---

## Implications for MathLedger

- **Verification cost is first-class.** If MathLedger routes do not demonstrably reduce V, value claims collapse under professional-duty scrutiny.

- **Trust typing prevents false economy.** PA (procedurally attested) and ADV (advisory) routes do not reduce verification burden—they merely document that verification was not performed mechanically. Only MV (mechanically verified) routes can credibly claim to reduce V.

- **Phase II priority: expand MV-route coverage.** In high-duty domains, the value proposition depends on MV routes that shift verification cost from human reviewers to replayable, auditable evidence packs.

- **Abstention preserves value.** ABSTAIN outcomes prevent false confidence. A system that abstains rather than emits unverifiable claims preserves net value by not imposing unresolvable verification cost.

- **Auditability is load-bearing.** Evidence packs, replay bundles, and closure matrices exist precisely to reduce V for downstream reviewers. Without these, efficiency gains are illusory.

- **ACE concept (documentation only): `verification_cost_proxy`.** Future Phase II design may include an explicit field or annotation indicating the expected verification burden per trust class. Not implemented; awareness only.

---

## Mapping to Trust Classes

| Trust Class | Verification Cost (V) | Implication |
|-------------|----------------------|-------------|
| MV | Low (mechanically verified; reviewer audits process, not output) | Value-positive in high-duty domains |
| PA | Medium (reviewer must validate procedure was correct) | Marginal value; verification still required |
| ADV | High (reviewer must independently verify claim) | No reduction in V; advisory only |
| FV | Lowest (formally verified; mathematical proof) | Maximum value; not yet implemented |

---

## Non-Claims

- This pressure does not assert that MathLedger currently reduces verification cost. Phase I routes are limited; MV coverage is narrow.
- This pressure does not assert that all professional domains require the same V. Verification scope varies by jurisdiction, profession, and context.
- This pressure does not propose a specific implementation. It identifies a structural constraint that Phase II must address.
- This pressure does not claim that high V makes AI unusable—only that value propositions must account for it honestly.

---

## Activation Condition

Only activate if:
- MathLedger explicitly targets high-duty domains (law, medicine, finance, regulated industries), AND
- Phase II planning prioritizes MV-route expansion.

**Until then:**
This remains parked as a pressure signal.

---

## Status

Awareness only.
Not a claim.
Not an obligation.
Not included in fm.tex.
