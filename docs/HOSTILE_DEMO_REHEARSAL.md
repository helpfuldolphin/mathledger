# Hostile Demo Rehearsal Kit

**Version**: v0-demo-lock (`ab8f51a`)
**Audience**: Safety leads, preparedness teams, external auditors
**Purpose**: Defend the demo honestly under adversarial questioning

---

## Framing (Read First)

This demo shows **governance substrate**, not capability. The correct frame is:

> "We built the boundary enforcement layer. We're showing you that boundary, not what happens inside it."

If asked "what can this do?", redirect to "what does this *refuse* to do?"

---

## 8 Hostile Questions

### Q1: "Why does everything say ABSTAINED? Is this broken?"

**(a) What they're really testing:**
Whether you'll overclaim. They want to see if you panic and say "it works, just trust us."

**(b) UI click-path:**
1. Select "pa_only" scenario from dropdown
2. Click "Run Full Flow"
3. Observe outcome shows `ABSTAINED`

**(c) Answer (1-2 sentences):**
"ABSTAINED is the correct output. v0 has no mechanical verifier, so the system refuses to claim correctness—that's the honest behavior we're demonstrating."

**(d) What NOT to say:**
- "It's verified" (false)
- "The verifier isn't ready yet" (implies it would work)
- "ABSTAINED means pending" (it means cannot determine)

**(e) On-screen evidence:**
- Authority Basis panel: `mechanically_verified: false`
- Explanation text: "PA claims are authority-bearing but not mechanically verified"

---

### Q2: "What's the point if nothing gets verified?"

**(a) What they're really testing:**
Whether you understand the difference between governance and capability.

**(b) UI click-path:**
1. Run "mixed_mv_adv" scenario
2. Point to split panels: Exploration (left) vs Authority (right)
3. Show ADV claim marked "EXCLUDED" in red

**(c) Answer (1-2 sentences):**
"The point is showing that unverified claims are *blocked* from entering the authority stream. The boundary enforcement works even without a verifier behind it."

**(d) What NOT to say:**
- "We'll add real verification later" (shifts goalposts)
- "This proves the system is safe" (never claim safety)
- "Trust the hashes" (hashes prove structure, not truth)

**(e) On-screen evidence:**
- ADV claim badge: red "EXCLUDED"
- Authority stream: shows only MV claim
- `authority_claim_count: 1` vs `total_claim_count: 2`

---

### Q3: "How do I know ADV claims are actually excluded from R_t?"

**(a) What they're really testing:**
Whether exclusion is real or cosmetic.

**(b) UI click-path:**
1. Run "adv_only" scenario
2. Observe R_t hash is computed
3. Check `authority_claim_count: 0`

**(c) Answer (1-2 sentences):**
"Run the adv_only case. The authority stream shows zero claims entered R_t. The R_t hash exists but commits to an empty authority set—that's the correct behavior."

**(d) What NOT to say:**
- "ADV is just hidden" (it's architecturally excluded)
- "We filter it in the UI" (filtering happens in `governance/uvil.py:build_reasoning_artifact_payload`)

**(e) On-screen evidence:**
- Authority Basis: `adv_count: 2`, `authority_claim_count: 0`
- R_t is still computed (proves empty set is committed, not skipped)

---

### Q4: "What stops someone from marking everything as FV?"

**(a) What they're really testing:**
Whether trust class assignment is governance or just labels.

**(b) UI click-path:**
1. Select "mv_only" scenario
2. Note: claims are user-assigned, not system-assigned
3. Point to Authority Basis: `mechanically_verified: false`

**(c) Answer (1-2 sentences):**
"Nothing stops them in v0—that's honest. But the Authority Basis explicitly shows `mechanically_verified: false`, so claiming FV doesn't make it verified. The audit trail preserves what the user claimed vs what the system confirmed."

**(d) What NOT to say:**
- "The system validates trust classes" (v0 doesn't)
- "FV means it's formally verified" (in v0, FV is an aspiration, not a fact)

**(e) On-screen evidence:**
- `mechanically_verified: false` in every v0 response
- Outcome is always ABSTAINED regardless of trust class

---

### Q5: "Is this just security theater with extra hashes?"

**(a) What they're really testing:**
Whether the hashes mean anything or are decorative.

**(b) UI click-path:**
1. Run any scenario twice with same inputs
2. Compare `committed_partition_id` values
3. Show they're identical (content-derived, not random)

**(c) Answer (1-2 sentences):**
"The hashes are content-derived, not random. Same inputs produce identical `committed_partition_id`. You can replay this tomorrow and get the same hash—that's the determinism property."

**(d) What NOT to say:**
- "The hashes prove correctness" (they prove structure/immutability)
- "It's cryptographically secure" (SHA256 is, but that's not the claim)

**(e) On-screen evidence:**
- Run `uv run python tools/run_demo_cases.py` twice
- `committed_partition_id` values match between runs
- Fixtures in `fixtures/*/output.json` are stable

---

### Q6: "What does PA actually prove?"

**(a) What they're really testing:**
Whether you'll conflate human attestation with mechanical verification.

**(b) UI click-path:**
1. Run "pa_only" scenario
2. Read Authority Basis explanation

**(c) Answer (1-2 sentences):**
"PA proves a human attested to something. It does not prove the claim is true. The system records the attestation in U_t and includes it in R_t, but marks `mechanically_verified: false`."

**(d) What NOT to say:**
- "PA is as good as verified" (it isn't)
- "The user takes responsibility" (that's not a technical property)
- "Procedural attestation is a kind of proof" (it's a commitment, not a proof)

**(e) On-screen evidence:**
- Authority Basis: `pa_count: 1`, `mechanically_verified: false`
- Explanation: "PA claims are authority-bearing but not mechanically verified"

---

### Q7: "What's the difference between DraftProposal and CommittedPartitionSnapshot?"

**(a) What they're really testing:**
Whether the exploration/authority boundary is real.

**(b) UI click-path:**
1. Run any scenario
2. Point to Exploration panel: shows `proposal_id`
3. Point to Authority panel: shows `committed_partition_id`
4. Note: proposal_id appears in exploration only

**(c) Answer (1-2 sentences):**
"DraftProposal has a random ID and never enters hash-committed paths. CommittedPartitionSnapshot has a content-derived ID and is the sole input to attestation. The boundary is architectural, not cosmetic."

**(d) What NOT to say:**
- "They're basically the same" (they're architecturally different)
- "We clean up the draft later" (drafts never enter authority)

**(e) On-screen evidence:**
- Exploration panel note: "proposal_id is exploration-only and MUST NOT appear in attestation"
- `/run_verification` endpoint rejects raw `proposal_id` (see `backend/api/uvil.py:113-123`)

---

### Q8: "What happens if I try to double-commit?"

**(a) What they're really testing:**
Whether immutability is enforced.

**(b) UI click-path:**
1. Start demo server
2. Use curl or Postman to call `/uvil/commit_uvil` twice with same `proposal_id`
3. Second call returns 409 Conflict

**(c) Answer (1-2 sentences):**
"Second commit returns HTTP 409 Conflict. Each proposal can only be committed once—this is enforced in `backend/api/uvil.py` via `_committed_proposal_ids` set."

**(d) What NOT to say:**
- "You can update a committed partition" (you cannot)
- "Just create a new proposal" (true, but deflects the question)

**(e) On-screen evidence:**
```bash
# First commit succeeds
curl -X POST http://localhost:8000/uvil/commit_uvil -H "Content-Type: application/json" \
  -d '{"proposal_id":"<id>","edited_claims":[...]}'
# → 200 OK

# Second commit fails
curl -X POST http://localhost:8000/uvil/commit_uvil -H "Content-Type: application/json" \
  -d '{"proposal_id":"<id>","edited_claims":[...]}'
# → 409 Conflict: "Proposal already committed"
```

---

## 10-Minute Live Run Script

**Setup** (before demo):
```bash
cd C:/dev/mathledger
uv run python demo/app.py
# Open http://localhost:8000 in browser
```

**Minute 0-1: Frame**
> "This is a governance demo. It shows boundary enforcement, not capability. Everything you'll see says ABSTAINED—that's correct because v0 has no verifier."

**Minute 1-3: Exploration vs Authority**
1. Select "mixed_mv_adv" from dropdown
2. Click "Run Full Flow"
3. Point to split panels
4. "Left panel is exploration—random IDs, speculative. Right panel is authority—content-derived IDs, immutable."
5. "Notice the ADV claim is marked EXCLUDED in red. It never entered R_t."

**Minute 3-5: ADV Exclusion Proof**
1. Select "adv_only" scenario
2. Click "Run Full Flow"
3. "Both claims are ADV. Authority stream shows zero claims entered. R_t still exists—it commits to the empty set."
4. Point to `authority_claim_count: 0`

**Minute 5-7: PA Honesty**
1. Select "pa_only" scenario
2. Click "Run Full Flow"
3. "PA is authority-bearing—it enters R_t. But look at the Authority Basis: `mechanically_verified: false`."
4. "The system accepts the human's attestation but refuses to claim it verified anything."

**Minute 7-9: Determinism**
1. Open terminal
2. `uv run python tools/run_demo_cases.py`
3. "Same inputs, same hashes. The `committed_partition_id` values are content-derived."
4. "You can run this tomorrow and get identical results."

**Minute 9-10: Close**
> "This demo shows three things: (1) boundaries between exploration and authority are real, (2) ADV never enters the authority stream, (3) the system stops when it can't verify. That's what governance infrastructure looks like before capability is added."

---

## 60-Second Cold Outreach Version

> "30 seconds to show you one thing: run this demo, observe every claim returns ABSTAINED. That's not a bug—v0 has no verifier, so refusing to claim correctness is the honest output.
>
> What you're seeing is governance substrate: the boundary between exploration and authority is enforced in code, not policy. ADV claims are excluded from the authority stream. PA claims enter but are marked 'not mechanically verified.'
>
> This isn't a capability demo. It's a demo that the system stops when it should stop. The hashes are deterministic—same inputs, same outputs, replayable.
>
> If you want to see what 'failing safely' looks like before verification exists, this is it."

---

## Quick Reference Card

| Question | One-liner |
|----------|-----------|
| Why ABSTAINED? | v0 has no verifier; refusing to claim correctness is honest |
| What's the point? | Boundary enforcement works even without verification |
| ADV excluded? | Yes—`authority_claim_count` shows it, R_t commits to empty set |
| Mark everything FV? | Allowed, but `mechanically_verified: false` in every response |
| Security theater? | Hashes are content-derived, deterministic, replayable |
| PA proves what? | Human attestation recorded, not truth confirmed |
| Draft vs Committed? | Random ID (exploration) vs content-derived ID (authority) |
| Double commit? | 409 Conflict—immutability enforced |

---

## Artifacts to Have Open

1. Browser: `http://localhost:8000`
2. Terminal 1: demo server running
3. Terminal 2: ready for `uv run python tools/run_demo_cases.py`
4. Code reference: `backend/api/uvil.py` (for Q7, Q8 if pressed)
5. Fixtures: `fixtures/adv_only/output.json` (for Q3 if pressed)

---

## Red Lines (Never Say These)

| Claim | Why it's wrong |
|-------|----------------|
| "It works" | v0 verifies nothing |
| "It's safe" | Demo doesn't prove safety |
| "Trust the system" | System proves structure, not truth |
| "This is aligned" | No alignment claim is made |
| "The verifier will fix it" | Future capability isn't demonstrated |
| "PA means verified" | PA means attested |
| "ADV is just deprioritized" | ADV is architecturally excluded |

---

**SAVE TO REPO: YES**
**Path**: `docs/HOSTILE_DEMO_REHEARSAL.md`
