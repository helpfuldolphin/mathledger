# MathLedger: A Verifiable Ledger of Mathematical Truths

## Abstract
We present **MathLedger**, a system that constructs a **ledger of mathematics**: a monotone, auditable, and queryable record of all provable truths within bounded axiomatic frameworks. MathLedger automates the generation of statements, derives proofs via axiomatic inference, verifies them in Lean 4, and records every result as a record entry with Merkle-style provenance. Each record entry saturates a **slice of a theory**—beginning with propositional logic and climbing a **logic ladder** through first-order logic with equality, equational theories, and linear arithmetic. The result is **authentic synthetic data**: infinite in scope, but Lean-verified, normalized, deduplicated, and cryptographically sealed.

MathLedger is both a **backend factory** and a **frontend interface**. The backend generates and records proofs automatically, night after night. The frontend exposes a typed API, a lightweight UI, and an AI wrapper so that humans and models can traverse, query, and even generate textbooks and research papers from the ledger itself. This duality makes MathLedger both the substrate for large-scale reasoning research and the platform through which future mathematics is authored.

---

## 1. Motivation
Mathematics is the cleanest substrate for reasoning. Yet no system today offers **scalable, verifiable, and extensible data** for machine reasoning research:
- **Scraped corpora** (e.g. arXiv, web math data) are noisy and unverifiable.
- **Benchmarks** (MATH, GSM8K, MiniF2F) are tiny, brittle, and inconsistent.
- **LLM proofs** hallucinate; internal labs hand-curate datasets at great cost.

MathLedger addresses this by **automating verification generation and validation** into a *ledger*: every statement is derived from axioms, every verification is validated in Lean, every record entry is auditable. Unlike scraped data, MathLedger grows without bound. Unlike RL-provers, its automation is deterministic and reproducible. It is the first source of **authentic synthetic reasoning data** at scale.

---

## 2. System Overview
### Architecture
[Enumerator] → [Derivation Engine] → [Lean Verifier] → [Normalizer/Hasher] → [Ledger DB] → [Block Builder] → [Curriculum YAML]

- **Enumerator**: generates candidate formulas bounded by atoms and depth, deduped via AST normalization.
- **Derivation Engine**: instantiates axioms, applies Modus Ponens and substitution, derives new statements.
- **Lean Verifier**: invokes Lean tactics (`taut`, `aesop`, `by decide`) or internal checkers, validates correctness.
- **Normalizer/Hasher**: canonicalizes formulas, computes SHA-256 hash IDs.
- **Ledger DB**: Postgres tables: `theories`, `statements`, `verifications`, `dependencies`, `runs`, `records`.
- **Record Builder**: groups verifications into a record entry, computes Merkle root over verification IDs, appends record to ledger.
-- **Curriculum YAML**: defines per-system slices (atoms, depth, breadth, total caps), ratchets to the next slice when thresholds are met.

Mathematical activity follows the **Chain of Verifiable Cognition**. The living implementation of that chain is the **First Organism**; it is documented in `docs/FIRST_ORGANISM.md`, validated by `tests/integration/test_first_organism.py`, sealed via `basis/attestation/dual_root.py`, and referenced in the attestation and RFL specs (`docs/ATTESTATION_SPEC.md`, `docs/RFL_IMPLEMENTATION_SUMMARY.md`). Every documented architecture milestone therefore has a concrete test and artifact associated with it, transforming the whitepaper from theory into diagnosable reality.

### Data Model
- `theories`: defines systems (PL, FOL=, Group, Ring).
- `statements`: `id, hash, text, system_id, depth, norm_jsonb, created_at`.
- `verifications`: `id, statement_id, method, prover, derivation_rule, status, duration_ms`.
- `dependencies`: DAG edges (premises → conclusion).
- `runs`: execution configs, start/finish times, summary stats.
- `records`: record headers with `run_id, system_id, root_hash, counts`.

---

## 3. The Logic Ladder
MathLedger advances inductively, system by system, slice by slice:

1. **Propositional Logic (PL)**
   - Connectives: ¬, ∧, ∨, →.
   - Axioms: K and S schemas.
   - Inference: Modus Ponens, Substitution.
   - Verification methods: Lean tactics (`taut`, `aesop`), internal truth tables.
   - Curricula: atoms ≤4, depth ≤4 → depth ≤5 → atoms ≤5, depth ≤6.

2. **First-Order Logic with Equality (FOL=)**
   - Quantifiers ∀, ∃.
   - Ground term instantiation bounded by depth.
   - Congruence closure for =.
   - Verification methods: Lean FOL tactics, Herbrand expansions.

3. **Equational Theories (Monoids → Groups → Rings)**
   - Rewrite rules: associativity, identity, inverses.
   - Verification methods: Knuth–Bendix completion, Lean `simp`.
   - Lemmas: algebraic identities, cancellations, homomorphisms.

4. **Linear Arithmetic (QF-LIA, QF-LRA)**
   - Verification methods: `linarith`, cutting-plane certs.
   - Lemmas: inequalities, bounds, linear dependencies.

Each system is saturated slice by slice, with **ratchet logic**: only when success rates and coverage thresholds are met does MathLedger advance to the next slice.

---

## 4. Algorithms
### Enumeration
- AST grammar: `Var | Not | And | Or | Imp`.
- Deduplication: normalize NNF, sort commutative operands, alpha-rename, hash.

### Derivation
- Apply axioms via substitution (bounded depth).
- Apply Modus Ponens on pairs (p, p→q) to yield q.
- Derivation depth = min(parent depths) + 1.

### Verification
- Try Lean tactics with bounded timeout.
- Fallback: truth-table evaluation for PL.
- Status: `success`, `failure`, `internal_only`.

### Ingestion
- Persist only **new hashes**.
- UPSERT operations support idempotency.
- Canonical proof chosen per statement.

### Record Construction
- After each run, collect successful verifications.
- Compute Merkle root: SHA-256 over verification ID list.
- Insert record header into `records`, append to ledger.

---

## 5. Ledger Semantics
MathLedger is a **monotone, auditable ledger**:
- Every record extends the DAG of mathematics; none retract.
- Each record header commits to verifications via Merkle root.
- Provenance is explicit: every statement links to axioms and parent proofs.
- Determinism: same slice config → identical hashes, verifications, records.
- Reproducibility: any record can be independently re-verified in Lean.

---

## 6. Evaluation
### Metrics Captured
- Verification throughput (verifications/sec).
- Success rate (Lean acceptance vs failures).
- Dedupe ratio (unique proofs vs raw generation).
- Lemma reuse (frequency of derived lemmas across records).
- Depth coverage (max derivation depth).

### Scaling Laws
MathLedger produces scaling curves analogous to LLMs:
- **Proofs/sec vs depth** (compute cost vs coverage).
- **Lemma hit-rate vs record index** (knowledge reuse curve).
- **Success % vs slice size** (robustness as domains widen).

---

## 7. Interface
### API
- FastAPI, typed with Pydantic models.
- Endpoints:
  - `/metrics`: JSON of system stats.
  - `/theories`: ladder view.
  - `/records/latest?system=pl`: latest record header.
  - `/statements?hash=...`: statement detail, proofs, parents.
  - `/lemmas/top`: most reused lemmas.

### UI
- `/ui/theories`: ladder visualization.
- `/ui/theories/pl`: active slice, block timeline.
- `/ui/s/<hash>`: statement detail page with proofs, dependencies.

### AI Wrapper
- Models tool-call the ledger API: search, traverse, export, run_derive.
- Humans interact via ChatGPT-5 layer: query, learn, explore.
- Both can generate textbooks, exercises, or research papers with every claim hyperlinked to ledger hashes.

---

## 8. Roadmap
- **Phase 1 (PL, live now):** Bounded-complete propositional logic, K+S axioms, Modus Ponens, substitution. Records finalized nightly.
- **Phase 2 (FOL=):** Equality reasoning, Herbrand instantiation, congruence closure.
- **Phase 3 (Equational theories):** Groups, rings, algebraic identities via rewrite proofs.
- **Phase 4 (Linear arithmetic):** Inequalities, bounded model proofs.
- **Phase 5:** Extend to topology, category theory, analysis — leveraging mathlib’s corpus as seed axioms.

At each phase, MathLedger saturates bounded slices, ratchets difficulty, and records entries.

---

## 9. Business Positioning
MathLedger is the **data and infrastructure layer** for reasoning research.
- **Labs don’t need another RL prover.** They need **clean, limitless, verified data**.
- MathLedger delivers **authentic synthetic data**: infinite in scope, but Lean-verified.
- The ladder structure gives labs a natural **curriculum for training and evaluation**.
- The API provides a **tool-use substrate** for models to practice calling external reasoning engines.

**Acquisition thesis:**
- Owning MathLedger means owning the **substrate of mathematical reasoning data**.
- It is the reasoning equivalent of owning the ImageNet of 2010 or the Bitcoin ledger of 2009.
- It is both **a factory** (backend records) and **an author** (frontend AI wrapper) — producing textbooks, conjectures, and research papers from the ledger itself.

---

## 10. Conclusion
MathLedger makes mathematics a **living, verifiable dataset**. It climbs the ladder of logic slice by slice, record by record, with every truth cryptographically sealed and every verification reproducible. On the backend, it is an **automated factory of authentic synthetic reasoning data**. On the frontend, it is an **interface for humans and AIs to explore, learn, and create mathematics**.

Just as Bitcoin demonstrated a new substrate for money, MathLedger demonstrates a new substrate for **computational reasoning itself**. It does not automate subjective understanding — it scaffolds it, extends it, and makes it auditable at scale.

MathLedger is the **ledger of mathematics**.
