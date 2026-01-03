# Hostile Demo Rehearsal Kit

**Version**: v0.1 (MV arithmetic validator added)
**Audience**: Safety leads, preparedness teams, external auditors
**Purpose**: Defend the demo honestly under adversarial questioning

---

## Framing (Read First)

This demo shows **governance substrate**, not capability. The correct frame is:

> "We built the boundary enforcement layer. We're showing you that boundary, not what happens inside it."

If asked "what can this do?", redirect to "what does this *refuse* to do?"

---

## 10 Hostile Questions

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

### Q9: "Why does THIS claim verify but not that one?"

**(a) What they're really testing:**
Whether verification is real or arbitrary. This is the key question once MV arithmetic validation exists.

**(b) UI click-path:**
1. Run "mv_arithmetic_verified" scenario: claim is "2 + 2 = 4" marked MV
2. Observe outcome: `VERIFIED`
3. Run "same_claim_as_pa" scenario: claim is "2 + 2 = 4" marked PA
4. Observe outcome: `ABSTAINED`
5. Run "mv_arithmetic_refuted" scenario: claim is "2 + 2 = 5" marked MV
6. Observe outcome: `REFUTED`

**(c) Answer (1-2 sentences):**
"VERIFIED only occurs when: (1) the claim is marked MV, AND (2) the arithmetic validator can parse and confirm it. Same text marked PA doesn't go through the validator—that's the trust class distinction. `2 + 2 = 5` marked MV returns REFUTED because the validator ran and found it false."

**(d) What NOT to say:**
- "The system knows math" (it knows one pattern: `a op b = c`)
- "MV claims are verified" (only if the validator can parse them)
- "This proves the system is intelligent" (it's a regex + arithmetic)

**(e) On-screen evidence:**
- mv_arithmetic_verified: `outcome: VERIFIED`, `mechanically_verified: true`
- same_claim_as_pa: `outcome: ABSTAINED`, `mechanically_verified: false`
- mv_arithmetic_refuted: `outcome: REFUTED`, `mechanically_verified: true`
- Authority Basis shows `mv_validation: { verified: 1, refuted: 0, abstained: 0 }`

**(f) Key insight:**
The trust class determines *which validator runs*. The validator determines the outcome. PA bypasses all validators. MV goes through the arithmetic validator. FV would go through a formal proof checker (not implemented). The governance layer routes; the verifier layer decides.

---

### Q10: "Prove this isn't just security theater with fancy hashes."

**(a) What they're really testing:**
Whether the hashes are decorative or functional. This is the "show me or shut up" question.

**(b) UI click-path:**
1. Run any MV verified scenario (e.g., mv_arithmetic_verified)
2. After verification, scroll to "Audit Verification" section
3. Click "Download Evidence Pack" → saves JSON file
4. Click "Replay & Verify" → shows PASS
5. Open the downloaded JSON, change one character in `reasoning_artifacts[0].claim_id`
6. Use curl to POST the tampered pack to `/uvil/replay_verify`
7. Observe: FAIL with diff showing which hash diverged

**(c) Answer (1-2 sentences):**
"Download the evidence pack. Tamper with any field. Run replay verification. It fails and shows you exactly which hash diverged. That's not theater—that's tamper detection."

**(d) What NOT to say:**
- "Trust the hashes" (show them, don't assert them)
- "The cryptography is sound" (irrelevant—the demo is about structure)
- "We use SHA256" (implementation detail, not the point)

**(e) On-screen evidence:**
```bash
# Download evidence pack
curl http://localhost:8000/uvil/evidence_pack/<committed_id> > pack.json

# Tamper with it (change one character)
# Then replay:
curl -X POST http://localhost:8000/uvil/replay_verify \
  -H "Content-Type: application/json" \
  -d @pack.json
# → {"result": "FAIL", "diff": {...}}
```

**(f) Key insight:**
The evidence pack is self-contained. No external API calls. No network access. Anyone with the pack can:
1. Recompute U_t from uvil_events
2. Recompute R_t from reasoning_artifacts
3. Recompute H_t = SHA256(R_t || U_t)
4. Compare to recorded values

If all match: PASS. If any differ: FAIL with diff. There is no way to tamper without detection. This is the audit instrument.

---

## 10-Minute Live Run Script

**Setup** (before demo):
```bash
cd C:/dev/mathledger
uv run python demo/app.py
# Open http://localhost:8000 in browser
```

**Minute 0-1: Frame**
> "This is a governance demo. It shows boundary enforcement and routing. Most claims return ABSTAINED. One type—MV with simple arithmetic—actually verifies. That's intentional: we show the full spectrum."

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

**Minute 7-8: MV Verification (The Key Demo)**
1. Select "mv_arithmetic_verified" scenario
2. Click "Run Full Flow"
3. "This claim is `2 + 2 = 4` marked MV. Outcome: VERIFIED."
4. Select "same_claim_as_pa" scenario
5. "Same text, but marked PA. Outcome: ABSTAINED."
6. "Trust class determines routing. MV goes to validator. PA bypasses it."
7. Select "mv_arithmetic_refuted"
8. "`2 + 2 = 5` marked MV. Outcome: REFUTED. The validator ran and found it false."

**Minute 8-9: Determinism**
1. Open terminal
2. `uv run python tools/run_demo_cases.py`
3. "All 9 cases pass. Same inputs, same hashes."
4. "You can run this tomorrow and get identical results."

**Minute 9-10: Close**
> "This demo shows four things: (1) boundaries between exploration and authority are real, (2) ADV never enters the authority stream, (3) the system stops when it can't verify, (4) when verification exists (MV arithmetic), it runs and returns VERIFIED or REFUTED. That's the full governance stack."

---

## 60-Second Cold Outreach Version

> "30 seconds to show you one thing: run `2 + 2 = 4` as MV—it returns VERIFIED. Run `2 + 2 = 5` as MV—it returns REFUTED. Run `2 + 2 = 4` as PA—it returns ABSTAINED.
>
> Same claim text, different trust class, different outcome. That's governance routing: the trust class determines which validator runs. MV goes to arithmetic. PA bypasses validators. ADV never enters the authority stream at all.
>
> This isn't a capability demo. It's a demo of boundary enforcement. The system verifies what it can, refuses to claim what it can't, and excludes what it shouldn't.
>
> If you want to see what 'honest verification infrastructure' looks like, this is it."

---

## Quick Reference Card

| Question | One-liner |
|----------|-----------|
| Why ABSTAINED? | No verifier for that trust class; refusing to claim is honest |
| What's the point? | Boundary enforcement works even without verification |
| ADV excluded? | Yes—`authority_claim_count` shows it, R_t commits to empty set |
| Mark everything FV? | Allowed, but `mechanically_verified: false` without FV verifier |
| Security theater? | Download, tamper, replay → FAIL. That's tamper detection, not theater. |
| PA proves what? | Human attestation recorded, not truth confirmed |
| Draft vs Committed? | Random ID (exploration) vs content-derived ID (authority) |
| Double commit? | 409 Conflict—immutability enforced |
| Why THIS verifies? | MV + parseable arithmetic → validator runs → VERIFIED/REFUTED |
| Prove it's not theater? | Evidence pack + replay verify: tamper → FAIL with diff |

---

## Artifacts to Have Open

1. Browser: `http://localhost:8000`
2. Terminal 1: demo server running
3. Terminal 2: ready for `uv run python tools/run_demo_cases.py`
4. Terminal 3: ready for curl commands (Q10 evidence pack tamper test)
5. Code reference: `backend/api/uvil.py` (for Q7, Q8, Q10 if pressed)
6. Code reference: `governance/mv_validator.py` (for Q9 if pressed)
7. Fixtures: `fixtures/mv_arithmetic_verified/output.json` (for Q9 comparison)
8. Downloaded evidence pack JSON (for Q10 tamper demo)

---

## Red Lines (Never Say These)

| Claim | Why it's wrong |
|-------|----------------|
| "It works" | Only arithmetic MV claims verify; everything else abstains |
| "It's safe" | Demo doesn't prove safety |
| "Trust the system" | System proves structure, not truth |
| "This is aligned" | No alignment claim is made |
| "The verifier will fix it" | Future capability isn't demonstrated |
| "PA means verified" | PA means attested |
| "ADV is just deprioritized" | ADV is architecturally excluded |
| "The system knows math" | It knows one pattern: `a op b = c` |
| "MV always verifies" | Only parseable arithmetic; unparseable MV → ABSTAINED |

---

**SAVE TO REPO: YES**
**Path**: `docs/HOSTILE_DEMO_REHEARSAL.md`
