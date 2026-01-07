# Future Pressure Candidates

**Status:** Awareness only
**Not included in fm.tex**

---

## Latent Pressure Candidates (Not Activated)

### Continual Learning vs No Silent Authority

**Context:**
Speculative claims (unverified) suggest that deployed LLM systems may eventually support persistent or continual learning across sessions.

**Relevance:**
If persistent learning becomes real, it directly collides with MathLedger's core obligation:
"No Silent Authority" — nothing may influence durable learning without explicit, inspectable attestation.

**Potential collision:**
- Persistent learning encourages silent accumulation of state.
- MathLedger requires all durable state transitions to be artifact-bound, versioned, and auditable.
- This creates tension between adaptivity and epistemic legibility.

**Status:**
Awareness only.
Not a claim.
Not an obligation.
Not included in fm.tex.

**Activation condition:**
Only activate if:
- Persistent learning is demonstrably deployed in production systems, AND
- MathLedger chooses to address learning-over-time as a first-class concern.

**Until then:**
This remains parked.

---

### Monte-Carlo-Governed Control Planes

**Context:**
Monte Carlo policy evaluation is becoming a default enterprise architecture for AI-assisted decision systems.

**Relevance:**
You will be asked: "Why not just do this?"
Your answer must already be written.

**Position:**
Control planes that govern action selection via Monte Carlo evaluation lack replay-verifiable non-learning guarantees and fail-closed authority semantics.

**Potential collision:**
- Monte Carlo methods optimize expected outcomes, not auditable guarantees
- No deterministic replay path exists for stochastic policy evaluation
- Authority delegation becomes implicit rather than explicit

**Pressing level:** Medium-High (enterprise adoption vector)

**Status:**
Awareness only.
Not a claim.
Not an obligation.
Not included in fm.tex.

**Activation condition:**
Only activate if:
- Enterprise adoption pressure forces explicit response, AND
- MathLedger must differentiate from MCTS/MC-based governance architectures.

**Until then:**
This remains parked.

---

### Verification-Value Paradox (Legal/Professional Domains)

**Context:**
In high-duty domains (law, medicine, finance), verification is mandatory and scope is broad. Net value N = EG − V; when verification cost V approaches efficiency gain EG, value collapses.

**Relevance:**
MathLedger's value proposition in professional domains depends on MV routes that demonstrably reduce V. ADV and PA routes do not reduce verification burden.

**Reference:**
See `verification_value_paradox_pressure.md` for full analysis.

**Status:**
Awareness only. Phase II pressure.

---

### Skills-Enabled Agent Benchmarks (e.g., SkillsBench)

**Context:**
Increasing agent task success on real-world workflows (docs, git, data pipelines) raises the risk of unverified outputs being treated as authoritative.

**Relevance:**
Task success ≠ claim admissibility. As agents reliably complete multi-step tasks, enterprise users may conflate "it worked" with "it's verified." This strengthens the need for a post-capability governance layer.

**Potential collision:**
- High benchmark scores encourage deployment without governance
- Artifacts produced by successful tasks inherit no trust class
- "Task completed" is not an authority signal

**Status:**
Awareness only. Strengthens Phase II MV-route expansion pressure.

---

### OpenForesight / OpenForecaster (Forecasting as Training Target)

**Context:**
Forecasting systems trained on calibration (not just accuracy) demonstrate that well-calibrated probabilistic outputs are achievable. Key innovation: offline corpus with leakage discipline (retrieval cutoff ≤1 month before resolution).

**Relevance:**
Pressures MathLedger to define how probabilistic claims acquire trust class. Calibration generalization claims require:
- Corpus versioning and hash pinning
- Retrieval cutoff enforcement
- Resolution source specification
- Replay bundles for probabilistic claims

**Classification:** PA / ADV (procedural attestation candidate; observer-only until route defined)

**Reference:**
See `openforesight_forecasting_pressure.md` for full analysis.

**Status:**
Awareness only. Phase II pressure for probabilistic claim routing.

---

### Forensic Agent Auditing (Lane B Evidence Substrate)

**Context:**
Enterprise agent deployments require forensic traceability for dispute resolution, compliance audits, and incident response—independent of governance verdicts.

**Relevance:**
As agents handle revenue-facing and regulated workflows, enterprises will demand hash-chained event logs, replay bundles, and tamper-evident evidence packaging. This creates pressure to clarify the boundary between forensic evidence (Lane B) and governance authority (Lane A).

**Key constraint:**
Forensic evidence does not substitute for governance. Evidence artifacts may inform human judgment but cannot produce VERIFIED / REFUTED outcomes or assign trust class.

**Classification:** Lane B / Evidence-only

**Reference:**
See `agent_audit_kit_lane_b.md` for architectural separation.

**Status:**
Awareness only. Clarifies Lane A / Lane B boundary.