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
