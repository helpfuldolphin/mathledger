# OpenForesight Forecasting Pressure

**Status:** PARKED
**Classification:** PA / ADV (observer-only; procedural attestation candidate)
**Phase:** II+ consideration
**Not included in fm.tex**

---

## Source

"Scaling Open-Ended Reasoning To Predict the Future" (OpenForecaster / OpenForesight)

---

## Why It Matters to MathLedger

Calibrated probabilistic forecasting represents a pressure vector for governance systems:

1. **Calibration as governance artifact.** If forecasts are well-calibrated, users demand that calibration itself become a verifiable property—not just accuracy.

2. **Probabilistic claims need typed routes.** MathLedger currently handles binary outcomes (VERIFIED / REFUTED / ABSTAINED). Probabilistic claims (e.g., "70% confidence event X occurs by date Y") require a new route type with explicit uncertainty semantics.

3. **Leakage discipline is load-bearing.** The paper's offline corpus approach (CCNews snapshots, retrieval cutoff ≤1 month before resolution) demonstrates that leakage control is a first-class reproducibility constraint. This aligns with MathLedger's artifact-binding requirements.

4. **Resolution ambiguity is a failure mode.** Forecasting claims require an explicit resolution source and criteria. Without this, "did the prediction come true?" becomes a judgment call, not a verifiable fact.

---

## Authoritative Technical Facts (from paper)

| Field | Value |
|-------|-------|
| Training corpus | CCNews offline snapshots (leakage-controlled) |
| Retrieval cutoff | ≤1 month before resolution date |
| Dataset size | ~52k question-answer pairs |
| Question generation | Synthetic pipeline from news articles |
| Leakage removal | Explicit filtering stage |
| Answer-matching judge | LLM-based (model not pinned in paper) |
| Calibration metric | Adapted Brier score for free-form answers |
| Training method | GRPO (reinforcement learning) |
| Key finding | Accuracy-only training hurts calibration; accuracy+Brier recommended |
| Calibration claim | Calibration generalizes across question types |

---

## Threats / Failure Modes If Used As Authority

| Threat | Description |
|--------|-------------|
| **Leakage** | If retrieval cutoff is not enforced, model may have seen resolution before prediction |
| **Judge drift** | Answer-matching judge is LLM-based; model/prompt changes silently shift scores |
| **Resolution ambiguity** | "Did event X happen?" may be contested; no canonical resolution source |
| **Non-verifiable outcomes** | Some predictions have no objective resolution (opinions, counterfactuals) |
| **Corpus version drift** | Different CCNews snapshots produce different training; no hash pinning |
| **Calibration overfitting** | Calibration may be tuned to held-out set rather than generalizing |

---

## Admissibility Routing Proposal (Phase II)

For OpenForesight-style forecasting to become a Procedurally Attested (PA) route in MathLedger:

### Required Contract Fields

| Field | Description |
|-------|-------------|
| `CORPUS_SNAPSHOT_HASH` | sha256 of offline news corpus used |
| `RETRIEVAL_CUTOFF_POLICY` | Explicit rule (e.g., "≤30 days before resolution") |
| `QUESTION_GEN_PROMPT_HASH` | sha256 of question generation prompt |
| `QUESTION_GEN_MODEL_ID` | Model used for synthetic question generation |
| `JUDGE_MODEL_ID` | Model used for answer matching |
| `JUDGE_PROMPT_HASH` | sha256 of judge prompt |
| `RESOLUTION_SOURCE` | Canonical source for determining outcome |
| `RESOLUTION_CRITERIA` | Explicit criteria for YES/NO/AMBIGUOUS |
| `PREDICTION_TIMESTAMP` | When prediction was made |
| `RESOLUTION_TIMESTAMP` | When outcome was determined |
| `CONFIDENCE` | Stated probability (0.0–1.0) |
| `REPLAY_BUNDLE` | Full artifact package for deterministic replay |

### Route Schema (Candidate)

```
PROBABILISTIC_CLAIM_v1 {
  prediction: string,
  confidence: float,
  resolution_source: string,
  resolution_criteria: string,
  retrieval_cutoff: date,
  corpus_hash: sha256,
  judge_model_id: string,
  judge_prompt_hash: sha256,
  prediction_timestamp: datetime,
  resolution_timestamp: datetime | null,
  outcome: PENDING | CORRECT | INCORRECT | AMBIGUOUS,
  replay_bundle_hash: sha256
}
```

### Trust Class

- **Before resolution:** ADV (advisory) — prediction is a probabilistic claim, not authority
- **After resolution:** PA (procedurally attested) — outcome determined by pinned resolution source and criteria
- **Never MV:** Forecasting inherently involves judgment; no mechanical verification of "will X happen"

---

## Re-open Conditions

Only activate if:
1. MathLedger addresses probabilistic claims as a first-class concern, AND
2. A standard PA route schema for forecasting is defined, AND
3. Resolution sources and criteria can be pinned and versioned.

**Until then:**
This remains parked as a pressure signal.

---

## Generalizable Lesson

See: [Evaluation Contract Checklist](evaluation_contract_checklist.md)

Forecasting systems that seek governance admission must satisfy the evaluation contract checklist, with additional fields for:
- Temporal cutoffs (leakage prevention)
- Resolution source specification
- Calibration artifact logging

---

## Non-Claims

- This pressure does not assert that MathLedger should implement forecasting routes in Phase I.
- This pressure does not assert that calibration alone constitutes authority.
- This pressure does not assert that OpenForesight's specific methodology is correct or optimal.
- This pressure does not propose immediate integration; it identifies a structural demand that Phase II may need to address.

---

## Status

Awareness only.
Not a claim.
Not an obligation.
Not included in fm.tex.
