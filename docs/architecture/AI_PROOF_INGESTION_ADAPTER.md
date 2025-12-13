# AI Proof Ingestion Adapter — Design Specification

**Version:** 1.0.0
**Status:** APPROVED DESIGN (Implementation Pending)
**Approved:** 2025-12-13
**Purpose:** Define a minimal, controlled interface for ingesting AI-generated proofs into MathLedger without compromising provenance, stability, or narrative discipline.

---

## Executive Summary

This specification defines the AI Proof Ingestion Adapter — a controlled intake port for externally-generated mathematical proofs. The adapter treats AI-generated proofs as untrusted external input, subjects them to the same Lean 4 verification as internal derivations, and records them with full cryptographic provenance.

**Key Property:** MathLedger remains a ledger of verified truths. AI proofs are verified, not trusted.

---

## 1. Problem Statement

MathLedger currently generates statements via axiom engine and verifies them in Lean 4. The system does not yet accept externally-generated proofs from AI systems (GPT, Claude, Symbolica, etc.).

When AI-generated proofs arrive, they will stress every assumption:
- **Provenance**: Who/what generated this? Can we trace it?
- **Stability**: Will high-volume noisy input destabilize USLA metrics?
- **Narrative**: Are we still a "ledger of truths" or becoming an "AI output validator"?

This adapter answers "yes, we can ingest AI proofs" while preserving all three guarantees.

---

## 2. Design Principles

### 2.1 Proof-or-Abstain (Unchanged)

AI-generated proofs receive the same treatment as internally-derived statements:
- **VERIFIED**: Lean 4 accepts the proof term
- **ABSTAIN**: Lean 4 rejects or times out

No new status. No "AI-verified" or "partially checked" categories. The ledger records only what Lean confirms.

### 2.2 Provenance Chain (Extended)

Every ingested proof carries mandatory metadata:

```json
{
  "source_type": "external_ai",
  "source_id": "gpt-4-turbo-2025-01",
  "submission_id": "uuid",
  "submitted_at": "ISO-8601",
  "submitter_attestation": "sha256-of-request",
  "raw_output_hash": "sha256-of-ai-response"
}
```

This metadata enters the Reasoning Root (R_t) alongside the proof artifact. The ledger never records a proof without knowing its origin.

### 2.3 Rate Limiting (New)

AI systems can generate thousands of proofs per hour. Unbounded ingestion would:
- Overwhelm Lean verification workers
- Flood USLA telemetry with noise
- Create governance signal instability

The adapter enforces:
- **Per-source rate limit**: 100 submissions/hour per `source_id`
- **Global rate limit**: 500 total AI submissions/hour
- **Backpressure signal**: Return 429 with retry-after, not silent drop

### 2.4 Shadow Mode First (Mandatory)

For the first N submissions (configurable, default 1000), all AI-ingested proofs operate in **Shadow Mode**:
- Verification runs normally
- Results recorded in ledger
- USLA/TDA metrics computed
- **No governance enforcement triggered**
- **No slice advancement credit**

This creates a calibration period to observe AI proof behavior before trusting it for system progression.

---

## 3. Architectural Position

### 3.1 Layer Mapping

| Layer | Role | In MathLedger |
|-------|------|---------------|
| Producers | Generate goods | Humans, AI proof generators |
| Goods | What is produced | Proof artifacts |
| Customs & Courts | Decide admissibility | Lean kernel |
| Land Registry | Permanent record | Ledger |
| Census / VIN | Provenance | `proof_provenance` + Merkle roots |
| Central Bank | Stability | USLA |
| Civil Engineers | Structural integrity | TDA |
| Trial Period | Import quarantine | Shadow mode |
| Naturalization | Conditional trust | UVI graduation |

### 3.2 What This Adapter Does NOT Do

| Excluded | Rationale |
|----------|-----------|
| AI model fine-tuning | MathLedger verifies, does not train |
| Proof synthesis | We ingest, we don't generate |
| Multi-model consensus | One source per submission; no voting |
| Automatic trust escalation | Shadow mode exit requires manual review |
| Lean tactic generation | Proof terms only, not tactic scripts |
| AI quality scoring | Binary only: verified or abstain |
| Model endorsement | Provenance is descriptive, not evaluative |

---

## 4. Component Design

### 4.1 Ingestion Endpoint

```
POST /api/v1/ingest/ai-proof
Authorization: Bearer <api-key>
Content-Type: application/json

{
  "statement": "(p → q) → (q → r) → (p → r)",
  "proof_term": "fun h1 h2 hp => h2 (h1 hp)",
  "source_metadata": {
    "source_type": "external_ai",
    "source_id": "gpt-4-turbo-2025-01",
    "model_temperature": 0.0,
    "prompt_hash": "sha256-of-prompt"
  }
}
```

**Response:**
```json
{
  "submission_id": "uuid",
  "status": "queued",
  "estimated_verification_time_ms": 5000,
  "shadow_mode": true
}
```

### 4.2 Verification Pipeline (Reused)

The existing `backend/worker.py` handles verification. The adapter:
1. Wraps the AI submission in standard job format
2. Adds `source_metadata` to job payload
3. Enqueues to existing Redis queue
4. Worker processes identically to internal jobs

No new verification path. Same Lean, same timeout, same abstention logic.

### 4.3 Provenance Recorder

After verification completes:
1. Standard proof record created in `proofs` table
2. New `proof_provenance` record links proof to source metadata
3. Source metadata hashed and included in R_t Merkle tree

### 4.4 Shadow Mode Gate

Proofs with `shadow_mode = true`:
- Count toward metrics
- Do not advance slice progression
- Do not trigger governance enforcement
- Are marked visibly in UI ("Shadow")

### 4.5 USLA Integration

AI-ingested proofs generate telemetry events tagged with `source_type: external_ai`. This enables:
- Separate H/ρ/τ tracking for AI vs. internal proofs
- Divergence detection between AI and internal verification rates
- Pattern analysis for AI-specific failure modes

No new USLA metrics. Existing infrastructure, new tags.

---

## 5. Database Schema Additions

```sql
-- Provenance tracking for external proofs
CREATE TABLE proof_provenance (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  proof_id UUID NOT NULL REFERENCES proofs(id) ON DELETE CASCADE,
  source_type TEXT NOT NULL,
  source_id TEXT NOT NULL,
  submission_id UUID NOT NULL UNIQUE,
  raw_output_hash TEXT NOT NULL,
  submitter_attestation TEXT,
  metadata JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_provenance_source ON proof_provenance(source_id);
CREATE INDEX idx_provenance_submission ON proof_provenance(submission_id);

-- Shadow mode flag
ALTER TABLE proofs ADD COLUMN shadow_mode BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE proofs ADD COLUMN source_type TEXT DEFAULT 'internal';
```

---

## 6. Graduation Criteria (Shadow → Production)

An AI source graduates from shadow mode when:

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| Volume | ≥1000 submissions | Statistical significance |
| Time window | ≥7 days in shadow | Temporal diversity |
| Verification rate | ≥95% success | Quality threshold |
| USLA stability | H within bounds | No destabilization |
| Human review | UVI CONFIRMATION | Explicit approval |

Graduation is per-source, not global. One model may graduate while another remains in shadow.

---

## 7. Rate Limiting Specification

```python
RATE_LIMITS = {
    "per_source_hour": 100,      # Max submissions per source_id per hour
    "global_hour": 500,          # Max total AI submissions per hour
    "burst_window_seconds": 60,  # Sliding window for burst detection
    "burst_max": 20,             # Max submissions in burst window
}
```

Exceeded limits return:
```json
{
  "error": "rate_limit_exceeded",
  "retry_after_seconds": 3600,
  "limit_type": "per_source_hour"
}
```

---

## 8. Narrative Discipline

This adapter preserves the core claim:

> "MathLedger is a cryptographically attested ledger of formally verified mathematical truths."

AI-generated proofs are:
- **Verified** by the same Lean kernel as internal proofs
- **Attested** with full provenance in the Merkle tree
- **Recorded** only if they pass verification

The system does NOT:
- Grade AI quality (only binary: verified or abstain)
- Endorse AI models (provenance is descriptive, not evaluative)
- Depend on AI for system progression (shadow mode isolation)

---

## 9. Implementation Sequence

| Phase | Scope | Estimated Effort |
|-------|-------|------------------|
| 1 | Database migrations (`proof_provenance`, `shadow_mode`) | 1 day |
| 2 | Ingestion endpoint (`/api/v1/ingest/ai-proof`) | 1 day |
| 3 | Rate limiter middleware (Redis-backed) | 0.5 day |
| 4 | Telemetry tagging (`source_type` in events) | 0.5 day |
| 5 | Shadow mode filter in slice progression | 0.5 day |
| 6 | UVI graduation flow | 0.5 day |
| 7 | Integration testing | 1 day |

**Total estimated scope:** 500–800 lines of code, 5 days implementation + testing

---

## 10. Open Questions (For Implementation Phase)

| Question | Options | Recommendation |
|----------|---------|----------------|
| Proof format | Lean 4 only vs. multi-prover | Start Lean 4 only, extend later |
| Source identity verification | API key binding vs. cryptographic attestation | API key binding for v1 |
| Public vs. private pilot | Open API vs. invite-only | Invite-only for first 90 days |

---

## 11. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Verification latency | <10s p99 | Prometheus histogram |
| Shadow mode graduation rate | >50% of sources | Per-source tracking |
| USLA stability under AI load | H within ±0.1 of baseline | TDA monitoring |
| Zero false positives | 0 ABSTAIN→VERIFIED inversions | Audit log review |

---

## Appendix A: Relationship to Existing Components

| Component | Relationship |
|-----------|--------------|
| `backend/worker.py` | Unchanged — processes AI jobs identically |
| `attestation/dual_root.py` | Extended — provenance enters R_t |
| `backend/health/*.py` | Unchanged — receives tagged events |
| `curriculum/gates.py` | Modified — respects shadow_mode flag |
| `backend/uvi/` | Used — graduation requires UVI CONFIRMATION |

---

## Appendix B: Strategic Position

This adapter makes MathLedger:

> **A neutral, verifiable intake port for industrial-scale proof generation.**

When AI systems (GPT-N, Symbolica, fine-tuned provers) generate thousands of formal artifacts per day, they will need:
- Provenance tracking
- Attestation
- Replay capability
- Drift detection
- Auditability

MathLedger provides all five. The adapter is the interface that makes this capability accessible without surrendering epistemic control.

---

**End of Specification**
