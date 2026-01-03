# How the MathLedger Demo Explains Itself

This document accompanies the front-facing demo. It explains what the demo enforces, what it refuses to do, and why those refusals are the point.

---

## What the Demo Is Actually Doing

The demo separates exploration from authority.

**Exploration phase**: When you enter a problem, the system generates a `DraftProposal`. This is a set of candidate claims with suggested trust classes. The proposal has a random identifier. Nothing in this phase is committed. You can edit freely. The system makes suggestions. None of it matters yet.

**Authority phase**: When you click "Commit," the system creates a `CommittedPartitionSnapshot`. This has a content-derived identifier (a hash of what you committed). The claims become immutable. The system records a `UVIL_Event` documenting your action. Only now do the claims enter the attestation structure.

The boundary is explicit: exploration identifiers never appear in committed data. The random `proposal_id` from the draft phase is not present in any hash-committed payload. This is not an implementation detail. It is the design.

Exploration is intentionally unconstrained because suggestions are not claims. Authority is explicitly bound because claims have consequences.

---

## Why the System Stops

The demo produces three possible outcomes: `VERIFIED`, `REFUTED`, or `ABSTAINED`.

`VERIFIED` means the system found a machine-checkable proof that the claim holds.

`REFUTED` means the system found a machine-checkable proof that the claim does not hold.

`ABSTAINED` means the system did not find either. This is not a failure. This is the correct output when the system cannot establish the claim.

The system does not guess. It does not approximate. It does not hedge. When it cannot verify or refute, it stops and says so.

This stopping is correctness, not caution. A system that produces confident outputs when it lacks grounds for confidence is broken. A system that reports the limits of what it can establish is working as intended.

---

## Why ADV Exists

Claims in the demo have trust classes. Three of them (FV, MV, PA) are authority-bearing and can enter the reasoning attestation. One of them (ADV) cannot.

ADV stands for Advisory. It means: this is a suggestion, not a claim.

The system is allowed to produce ADV content. It can speculate. It can offer best guesses. It can generate plausible-sounding statements. All of this is permitted in the exploration phase.

But ADV content is explicitly inert. It does not enter the reasoning root (R_t). It does not contribute to the attestation. It is recorded, but it has no authority.

This is not humility theater. The system is not pretending to be modest. It is making a structural distinction: suggestions exist, but they are not claims. The demo enforces this distinction at the data model level. ADV content cannot leak into authority-bearing paths because the code rejects it.

---

## What the Hashes Mean

The demo produces three values: U_t, R_t, and H_t.

**U_t** is the UI Merkle root. It commits to everything the human did: which claims they committed, which edits they made, which actions they took. If the human's actions change, U_t changes.

**R_t** is the Reasoning Merkle root. It commits to everything the system established: which claims were verified, which proofs were produced, which artifacts support those claims. If the system's reasoning changes, R_t changes.

**H_t** is the composite root. It binds U_t and R_t together. It represents the full state of what happened: human actions and system reasoning, tied to each other.

These values are deterministic. Given the same inputs, the system produces the same hashes. If you run the demo twice with identical actions, you get identical values.

This enables two properties:

1. **Replayability**: Anyone with the inputs can verify the outputs.
2. **Non-silent drift**: If the system's behavior changes, the hashes change. There is no way for the system to produce different outputs while claiming nothing changed.

You do not need to understand cryptography to use these values. You need to understand one thing: if the hash is the same, the content is the same. If the content changed, the hash changed.

---

## Evidence Pack + Replay Verification

The demo provides audit-grade closure through evidence packs.

**What is an evidence pack?**

An evidence pack is a self-contained JSON file containing:
- The committed partition snapshot (all claims and their trust classes)
- UVIL events (what the human did)
- Reasoning artifacts (what the system established)
- The attestation hashes (U_t, R_t, H_t)
- The formula note explaining how H_t is computed
- Replay instructions for independent verification

**Why evidence packs matter:**

Evidence packs enable independent verification. An auditor with the evidence pack can:
1. Recompute U_t from the UVIL events
2. Recompute R_t from the reasoning artifacts
3. Recompute H_t = SHA256(R_t || U_t)
4. Compare the computed values to the recorded values

If all three match, the attestation is verified. If any differ, something changed.

**No external calls required:**

Replay verification is fully local. The evidence pack contains everything needed. No API calls, no network access, no external dependencies. The computation is deterministic and reproducible.

**Tamper detection:**

If anyone modifies the evidence pack—changing a claim, altering a trust class, editing an artifact—the replay verification fails. The hashes will not match. There is no way to tamper with the evidence without detection.

**How to use:**

1. After verification, click "Download Evidence Pack" to save the JSON file
2. Click "Replay & Verify" to recompute and compare hashes locally
3. A PASS result confirms the attestation is intact
4. A FAIL result shows which hashes diverged

This is the audit trail. Same inputs, same hashes, same evidence. Every time.

**What replay does NOT prove:**

Replay verification proves structural integrity, not truth. A PASS result means: the recorded hashes match what the inputs produce. It does not mean: the claims are correct, the verification was sound, or the system behaved safely. The evidence pack proves the audit trail is intact. It proves nothing about what that trail represents.

---

## What This Demo Refuses to Claim

This demo does not claim the system is aligned with human values.

This demo does not claim the system is intelligent.

This demo does not claim the system is safe.

This demo does not claim the system will behave well in novel situations.

This demo claims one thing: the system's governance is legible. You can see what the human committed. You can see what the system established. You can see where the boundary is. You can verify that the boundary was not crossed.

That is all. Governance and epistemic legibility. Nothing else.

---

## How to Read the Demo as Evidence

**Justified conclusions:**

- The system separates exploration from authority at the data model level.
- The system produces deterministic attestations for committed content.
- The system refuses to include ADV content in authority-bearing paths.
- The system stops and reports ABSTAINED when it cannot verify or refute.
- The human's actions and the system's reasoning are recorded in separate, verifiable structures.

**Explicitly not justified:**

- The system will behave correctly in all cases.
- The system's verification is complete or sound in any formal sense.
- The system's abstention is always appropriate.
- The governance structure prevents all possible failures.
- The demo generalizes to production systems.

The demo is evidence that a specific set of invariants are enforced in a specific implementation. It is not evidence of anything beyond that.

---

## If This Feels Restrictive, That Is the Point

The demo does less than you might expect. It stops when it could guess. It refuses to promote ADV content. It separates phases that could be merged. It records hashes that could be omitted.

These restrictions are not limitations to be overcome. They are the design.

A system that does more than it can justify is not more capable. It is less trustworthy. A system that clearly reports its boundaries is not weaker. It is legible.

The demo is shaped this way because the alternative—a system that blurs the line between suggestion and claim, between exploration and authority, between what it knows and what it guesses—is worse. Not harder to build. Worse.

This demo is boring. That is also the point.

---

## Where the Demo Explains Itself in the UI

As of v0.2.0, the demo includes integrated self-explanation that appears directly in the interface. This is not decorative. These explanations are structurally enforced via regression tests.

### UI Self-Explanation Integration Points

| Location | What It Explains | Key Copy |
|----------|-----------------|----------|
| Framing Box (top) | What the system does NOT do | "The system does not decide what is true. It decides what is justified under a declared verification route." |
| Framing Expandable | What "justified" means | "A claim is justified when it passes through attestation with a declared trust class." |
| Trust Class Selector | What each trust class means | Tooltips explain FV, MV, PA, ADV |
| Trust Class Note | Trust class determines route, not correctness | "Trust class determines verification route, not correctness." |
| Transition Note | What changes at commit | "The random proposal_id is discarded. The committed_id is derived from claim content." |
| Outcome Display | What each outcome means | VERIFIED/REFUTED/ABSTAINED explanations |
| ABSTAINED Explanation | Why ABSTAINED is first-class | "ABSTAINED is recorded in R_t. It is a first-class outcome, not a missing value." |
| ADV Badge | Why ADV is excluded | "ADV claims are exploration-only. They do not enter R_t." |
| Hash Labels | What U_t, R_t, H_t commit to | Tooltips on each hash label |
| Evidence Pack Details | What the pack contains | Expandable explanation of contents |
| Boundary Demo | What the demo proves | "Same claim text, different trust class → different outcome." |

### UI Doc Sources

Auditors reviewing the demo UI should reference:

1. **DEMO_SELF_EXPLANATION_UI_PLAN.md** - Full specification of all 9 integration points
2. **demo/app.py:UI_COPY** - Canonical copy strings (line ~83)
3. **tests/governance/test_ui_copy_drift.py** - Regression tests for copy integrity
4. **tools/run_demo_cases.py --ui-tests** - Runtime self-explanation verification

### Enforcement

Self-explanation content is enforced through:

1. **UI_COPY dictionary**: All explanation strings are canonical and versioned
2. **Copy drift tests**: `test_ui_copy_drift.py` verifies required phrases are present
3. **Regression harness**: `run_demo_cases.py --ui-tests` checks endpoint availability
4. **No capability claims**: Tests verify absence of "safe", "aligned", "intelligent"

### What the UI Copy Does NOT Say

The UI self-explanation follows strict constraints:

- Does NOT say "the system believes" (anthropomorphizing)
- Does NOT say "verified correct" (overclaiming validator capability)
- Does NOT say "safe" or "aligned" (capability claims)
- Does NOT use emojis or exclamation marks (false enthusiasm)
- Does NOT promise future capabilities (scope creep)

The copy style is deliberately flat. It explains what happens, not what it means for safety or alignment.
