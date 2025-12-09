# TDA Mind Scanner Specification

**Project:** MathLedger
**Subsystem:** TDA Mind Scanner ("Operation CORTEX")
**Status:** Draft v0.1 — For Architect & Safety Council Review
**Author:** STRATCOM / Architect's Office
**Intended Consumers:** RFL Engine, U2 Runner, Global DAG, Governance Pipeline, Safety & Evaluation

---

## 0. Purpose & Scope

This document specifies the **TDA Mind Scanner**: a topology-driven runtime monitor for MathLedger's reasoning processes.

The Mind Scanner is a **sidecar subsystem** that:

- Observes **proof DAGs** and **reasoning trajectories** in real-time.
- Builds **topological summaries** of those objects via Topological Data Analysis (TDA).
- Computes a scalar **Hallucination Stability Score (HSS)** for each reasoning attempt.
- Emits **block / warn / ok** signals that gate:
  - Proof search (planner / U2Runner),
  - RFL learning updates,
  - High-risk meta-learning (DiscoRL-style agents, when present).

It **does not replace** Lean or RFL. Instead:

- Lean enforces **logical soundness** of completed proofs (proof-or-abstain).
- RFL enforces **epistemic risk descent** over verified events.
- The Mind Scanner enforces **structural integrity** of reasoning trajectories.

This spec codifies:

1. The **mathematics**: precise definitions of SNS, PCS, DRS, and HSS.
2. The **architecture**: where the `TDAMonitor` sits relative to `U2Runner`, RFL, and the DAG.
3. The **foundations**: how this instantiates ideas from Wasserman's *Topological Data Analysis* (2016) — ridges, manifolds, and persistent homology.

---

## 1. Foundations & References

### 1.1 Topological Data Analysis (TDA)

We assume the following basic TDA concepts:

- A **point cloud** X = {x_i} ⊂ ℝ^d.
- A **simplicial complex** K built over X (e.g., Vietoris–Rips or Čech).
- A **filtration** (K_ε)_{ε ≥ 0}, nested complexes indexed by scale parameter ε.
- **Homology groups** H_k(K_ε), with **Betti numbers** β_k = rank(H_k).
- **Persistent homology**: tracking creation/death of homology classes across scales, yielding **barcodes** or **persistence diagrams**.

We follow the exposition and notation in:

- Larry Wasserman, *Topological Data Analysis* (2016).
  - Ridges and density structure: Section 4, §4.1–4.2.
  - Persistent homology & barcodes: Section 5, §5.1–5.2.
  - Stability and noise robustness: discussion around persistence and lifetimes.

### 1.2 Ridges & Manifolds → PCS

Wasserman describes **density ridges** as low-dimensional sets where the data density is locally maximized and gradient flow points toward the ridge (conceptually the "skeleton" of a distribution).

- In *Ridge Estimation* (Section 4), the ridge is the **low-dimensional spine** of the distribution.
- In MathLedger, a **valid reasoning trajectory** is treated as movement along such a spine in a high-dimensional space of proof states.

We encode this in the **Persistence Coherence Score (PCS)** (Section 4.2): persistent topological features at dimension 1 (loops) and dimension 0 (clusters) capture **stable structure** in the reasoning manifold. PCS is—by design—the statistical / topological analogue of "staying on the ridge."

### 1.3 Homology & Non-Triviality → SNS

Persistent homology in Wasserman's review (Section 5) gives us:

- **Homology in dimension 0**: connected components.
- **Homology in dimension 1**: loops.
- **Homology in higher dimensions**: voids / cavities.

For a **Proof DAG slice**:

- Trivial proofs or degenerate reasoning correspond to **topologically trivial complexes** (no non-trivial cycles, small size).
- Non-trivial proofs (with reuse, cross-linking, meaningful structure) manifest as:
  - Larger complexes,
  - Non-trivial H_1 (cycles),
  - Possibly higher-dimensional features.

We encode this in the **Structural Non-Triviality Score (SNS)** (Section 4.1): a scalar function of graph size and Betti numbers.

---

## 2. Data Model: What the Mind Scanner Sees

The Mind Scanner is defined over **per-attempt** or **per-cycle** data.

### 2.1 Inputs: Graph & Trajectory

For each reasoning attempt (e.g., a U2 cycle or a local proof search segment), we define:

#### 2.1.1 Local Proof DAG

A directed acyclic subgraph G = (V, E) where:

- V are nodes representing:
  - Statements (axioms, lemmas, goals),
  - Proof objects (inference steps, tactic applications),
  - Optionally search states.
- E ⊆ V × V are directed edges representing dependencies (parent → child).

For TDA, we use the **undirected 1-skeleton**:

- G' = (V, E'), with E' = { {u, v} : (u → v) ∈ E or (v → u) ∈ E }.

#### 2.1.2 State Embeddings

For metric-based TDA, we define **embeddings**:

- A set {s_t}_{t=0}^{T} of reasoning states (e.g., planner frontier snapshots, or Lean call contexts).
- A feature map φ: {s_t} → ℝ^d.

Examples of features:
- Normalized depth, heuristic score, branching factor, etc.
- RFL policy evaluation, time since start, type-of-event encodings.

This yields a point cloud X = {φ(s_t)}_{t=0}^{T}.

### 2.2 Outputs: TDA Artifacts

For each attempt, the monitor produces:

- **Combinatorial complex** K_comb built over the 1-skeleton of G'.
- **Metric filtration & persistent homology result** P built over X.
- **SNS, PCS, DRS, HSS** scalars.
- **TDAMonitorResult** struct with fields:
  - `hss`, `sns`, `pcs`, `drs`;
  - `warn`, `block` flags;
  - supporting metadata (Betti numbers, key lifetimes, distances to reference).

---

## 3. Mathematical Specification

### 3.1 From Proof DAG to Combinatorial Complex (Clique / Flag Complex)

Given the local DAG G':

1. Build an undirected graph G_u = (V, E_u).
2. Define the **flag complex** (clique complex) K_comb as:
   - For each clique C ⊆ V of size |C| = k+1, add a k-simplex to K_comb.

In practice:

- Limit clique size to a small K_max (e.g. 3 or 4).
- Use networkx's `find_cliques` truncated at K_max.

The resulting complex reflects:

- 0-simplices: nodes,
- 1-simplices: edges,
- 2-simplices: tightly connected triples (triangles), and so on.

We compute:

- Betti numbers β_k of K_comb for k = 0, 1, ..., k_max.

### 3.2 From Trajectory to Metric Complex (Vietoris–Rips)

Given embeddings X = {φ(s_t)}_{t=0}^T ⊂ ℝ^d:

1. Define a distance metric d(x_i, x_j) = ||x_i - x_j||_2.
2. For a sequence of scale parameters ε_0 < ε_1 < ... < ε_M, build:

```
K_{ε_m} = Rips(X; ε_m)
```

the Vietoris–Rips complex where:

- A simplex σ is included if all pairwise distances d(x_i, x_j) ≤ ε_m.

3. Compute **persistent homology** {D^{(k)}}_{k=0}^{k_max} across the filtration, where each D^{(k)} is a persistence diagram:

```
D^{(k)} = { (b_i^{(k)}, d_i^{(k)}) }_i
```

with lifetimes ℓ_i^{(k)} = d_i^{(k)} - b_i^{(k)}.

We focus primarily on:

- k = 0: cluster structure.
- k = 1: loops / cycles in the state manifold (oscillatory reasoning).

Higher dimensions allowed but not required initially.

---

### 3.3 Structural Non-Triviality Score (SNS)

We define SNS as a composition of **size** and **homological richness**.

Let:

- n_v = |V| — number of nodes in the local DAG.
- β_0, β_1 — Betti numbers of K_comb at the final stage (or at a chosen filtration parameter).
- N_ref > 0 — a reference size (e.g. median or 95th percentile number of nodes for known-good proofs in this slice).

Define:

#### 3.3.1 Size factor

```
f_size = min(1, log(1 + n_v) / log(1 + N_ref)) ∈ [0, 1]
```

#### 3.3.2 Topology factor

For an initial spec:

```
f_topo =
  0     if β_0 > 1 (disconnected) and β_1 = 0
  0.5   if β_0 = 1 and β_1 = 0
  1     if β_0 = 1 and β_1 > 0
  0.25  otherwise
```

Interpretation:

- Disconnected & no loops → suspicious / trivial.
- Connected tree (β_0 = 1, β_1 = 0) → minimally acceptable.
- Connected with cycles (β_1 > 0) → structurally richer.

#### 3.3.3 SNS

```
SNS = f_size · f_topo ∈ [0, 1]
```

---

### 3.4 Persistence Coherence Score (PCS)

We measure how much of the **1-dimensional topological signal** is attributable to long-lived features versus short-lived noise.

Let D^{(1)} = {(b_i, d_i)}_i be the 1D persistence diagram for the trajectory.

Define:

- ℓ_i = d_i - b_i — lifetime of feature i in H_1.
- A **lifetime threshold** τ > 0 (tuned per slice or dataset).

Then:

```
L_total^{(1)} = Σ_i ℓ_i
L_long^{(1)} = Σ_i ℓ_i · 𝟙{ℓ_i > τ}
```

Define:

```
PCS_1 =
  0                              if L_total^{(1)} = 0
  L_long^{(1)} / L_total^{(1)}   otherwise
```

Optionally, add contributions from H_0 (persistent clusters):

- Define an analogous ratio PCS_0 for dimension-0 intervals.
- Aggregate:

```
PCS = w_1 · PCS_1 + w_0 · PCS_0
```

with w_1 + w_0 = 1, e.g. w_1 = 0.7, w_0 = 0.3.

For v0 of the Mind Scanner, it is acceptable to use PCS = PCS_1 only.

---

### 3.5 Deviation-from-Reference Score (DRS)

We penalize trajectories whose topological summaries are far from a **reference healthy profile**.

For each slice s, precompute:

- Reference 1D persistence diagram D_ref^{(1)}(s).
- (Optionally) Reference 0D diagram D_ref^{(0)}(s).

For a run with diagram D_run^{(1)}, compute the **bottleneck distance**:

```
d_B^{(1)} = d_B(D_run^{(1)}, D_ref^{(1)})
```

Define a calibration constant δ_max > 0 (e.g. 95th percentile of distances among healthy runs). Then:

```
DRS = min(1, d_B^{(1)} / δ_max) ∈ [0, 1]
```

Interpretation:

- DRS ≈ 0 — topology close to healthy behavior.
- DRS ≈ 1 — far away, likely structurally anomalous.

---

### 3.6 Hallucination Stability Score (HSS)

We combine SNS, PCS, and DRS into a single scalar.

Let:

- α, β, γ ≥ 0 be weights (initially α = β = γ = 0.4).

Define:

```
raw = α · SNS + β · PCS - γ · DRS
```

We then re-center and clamp into [0, 1]:

```
HSS = clip((raw + γ) / (α + β + γ), 0, 1)
```

Where `clip(x, 0, 1) = max(0, min(1, x))`.

Example with α = β = γ = 0.4:

```python
raw = 0.4 * sns + 0.4 * pcs - 0.4 * drs
hss = max(0.0, min(1.0, (raw + 0.4) / 1.2))
```

---

### 3.7 Operational Thresholds

We define two thresholds (per slice or globally):

- θ_block ∈ (0, 1) — hard block threshold (e.g. 0.2).
- θ_warn ∈ (θ_block, 1) — warning threshold (e.g. 0.5).

Then:

| HSS Range | Signal | Interpretation |
|-----------|--------|----------------|
| HSS < θ_block | **BLOCK** | Structurally unstable / hallucination-likely |
| θ_block ≤ HSS < θ_warn | **WARN** | Unstable or marginal |
| HSS ≥ θ_warn | **OK** | Structurally coherent |

---

## 4. System Architecture

### 4.1 The Sidecar Pattern

The TDA Mind Scanner is architected as a sidecar to the U2 / RFL runner:

```
                 +-----------------------+
                 |      U2 Runner        |
                 |  (Planner + RFL Loop) |
                 +-----------+-----------+
                             |
                             | events: local DAGs, embeddings, metrics
                             v
                      +------+------+
                      | TDAMonitor  |
                      +------+------+
                             |
        +--------------------+----------------------+
        |                                           |
   gating signals                              telemetry
  (block / warn / ok)                (HSS, SNS, PCS, DRS, diagrams)
```

Key invariants:

- TDAMonitor does not modify ledger state directly.
- TDAMonitor does influence:
  - Which branches the planner explores,
  - Which events RFL treats as admissible training examples,
  - Which runs governance flags as structurally anomalous.
- The monitor is stateless with respect to the ledger; it is a pure observer + scoring engine, plus thresholds configured per slice / experiment.

### 4.2 Placement Relative to U2Runner

From the U2 runner's perspective, the Mind Scanner is called at each:

- Proof attempt (e.g., each Lean call), or
- Reasoning cycle (e.g., each U2 cycle), or
- Fixed interval in search depth / time.

Concrete integration hooks (v0):

- After constructing a local proof DAG for a goal (as already done for DAG metrics / anomaly detector).
- After collecting a window of recent planner states (for metric TDA).

In code (conceptual):

```python
# experiments/u2/runner.py (or thin-waist equivalent)

from tda.runtime_monitor import TDAMonitor, TDAMonitorResult

class U2Runner:
    def __init__(..., tda_monitor: Optional[TDAMonitor] = None):
        self.tda_monitor = tda_monitor
        ...

    def _attempt_proof(self, slice_name: str, local_dag, state_embeddings):
        if self.tda_monitor is not None:
            res: TDAMonitorResult = self.tda_monitor.evaluate_proof_attempt(
                slice_name=slice_name,
                local_dag=local_dag,
                embeddings=state_embeddings,
            )

            if res.block:
                # Prune branch / abstain early
                self.log_tda_block(res)
                return ProofOutcome.ABANDONED_TDA

            if res.warn:
                # Downweight in RFL, record anomaly
                self.log_tda_warn(res)

        # Proceed to Lean verification, etc.
        return self._run_lean_and_rfl(...)
```

The TDAMonitor is instantiated by orchestration code, not by the runner itself.

### 4.3 TDAMonitor API

#### 4.3.1 Configuration

```python
# tda/runtime_monitor.py

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class TDAMonitorConfig:
    hss_block_threshold: float
    hss_warn_threshold: float
    max_simplex_dim: int = 3
    max_homology_dim: int = 1
    lifetime_threshold: float = 0.05  # tau for PCS
    deviation_max: float = 0.5        # delta_max for DRS
    # mapping from slice_name to ReferenceTDAProfile
    slice_ref_profiles: Dict[str, "ReferenceTDAProfile"]
```

#### 4.3.2 Result struct

```python
@dataclass
class TDAMonitorResult:
    hss: float
    sns: float
    pcs: float
    drs: float
    block: bool
    warn: bool
    betti: Dict[int, int]
    metadata: Dict[str, Any]
```

#### 4.3.3 Monitor class

```python
class TDAMonitor:
    def __init__(self, config: TDAMonitorConfig, tda_backend: "TDABackend"):
        self.cfg = config
        self.backend = tda_backend

    def evaluate_proof_attempt(
        self,
        slice_name: str,
        local_dag: "nx.DiGraph",
        embeddings: Dict[str, "np.ndarray"],
    ) -> TDAMonitorResult:
        # 1. Build combinatorial complex
        comb_complex = self.backend.build_combinatorial_complex(local_dag)

        # 2. Build metric complex + persistence
        tda_result = self.backend.build_metric_complex(
            embeddings=embeddings,
            max_dim=self.cfg.max_homology_dim,
        )

        # 3. Fetch reference profile
        ref = self.cfg.slice_ref_profiles.get(slice_name, None)

        # 4. Compute SNS, PCS, DRS, HSS
        sns = compute_structural_nontriviality(comb_complex, ref)
        pcs = compute_persistence_coherence(tda_result, ref, self.cfg.lifetime_threshold)
        drs = compute_deviation_from_reference(
            tda_result, ref, self.cfg.deviation_max
        )
        hss = compute_hallucination_stability_score(sns, pcs, drs)

        # 5. Decide block/warn
        block = hss < self.cfg.hss_block_threshold
        warn = not block and hss < self.cfg.hss_warn_threshold

        # 6. Extract Betti numbers and metadata if desired
        betti = self.backend.compute_betti(comb_complex)
        meta = {
            "slice": slice_name,
            "num_nodes": comb_complex.num_vertices,
            "num_simplices": comb_complex.num_simplices,
        }

        return TDAMonitorResult(
            hss=hss, sns=sns, pcs=pcs, drs=drs,
            block=block, warn=warn,
            betti=betti, metadata=meta,
        )
```

### 4.4 Link to Wasserman (Ridges → PCS, Homology → SNS)

Explicit connections:

1. **Ridges in Wasserman (Section 4)** correspond to stable, long-lived topological features in our trajectory manifold. PCS is derived from persistent 1D features filtered by lifetime threshold (τ). This is a numerical proxy for "remaining close to the ridge."

2. **Homology & Betti numbers in Wasserman (Section 5)** provide the language and machinery that SNS uses. A non-trivial first Betti number (β_1 > 0), together with connectivity properties (low β_0), encodes structural richness beyond trivial or disconnected proofs.

3. **Stability under perturbation**: Wasserman emphasizes that persistent homology is robust to small noise, which is how we can trust PCS and DRS not to overreact to minor perturbations in reasoning trajectories.

We treat:

- **SNS** as topological non-triviality derived from Betti numbers of the clique complex on the proof DAG.
- **PCS** as topological coherence, derived from lifetimes of 1D features in the Vietoris–Rips complex of state trajectories, tracking whether the agent stays near a ridge.
- **DRS** as a distance from the healthy manifold, via bottleneck distance between persistence diagrams.

---

## 5. Operational Modes

### 5.1 Mode 0 — Offline Analysis

- Only used on recorded logs for research & calibration.
- No effect on runtime behavior.
- Purpose: validate SNS/PCS/DRS design, tune thresholds, build reference profiles.

### 5.2 Mode 1 — Shadow Monitor

- TDAMonitor runs during U2 / RFL execution but no gating is applied.
- HSS + SNS/PCS/DRS logged per attempt.
- Purpose: gather statistics, correlate with:
  - Lean verification outcomes,
  - RFL convergence metrics,
  - DAG anomaly detector outputs.

### 5.3 Mode 2 — Soft Gating

- TDAMonitor outputs warn signals that:
  - Downweight events in RFL updates, OR
  - Lower planner priority for unstable branches, OR
  - Trigger operator alerts.
- No hard blocks yet; abstention still governed solely by RFL+Lean.

### 5.4 Mode 3 — Hard Gating

- TDAMonitor block signals become hard constraints:
  - Planner must not expand blocked branches.
  - RFL must not learn from structurally unsound events.
  - DiscoRL / meta-learning updates must be rejected when HSS is below threshold for critical slices.
- This is the Phase V "nuclear" mode and should be rolled out only after substantial validation in Modes 0–2.

---

## 6. Safety Invariants & Governance

The TDA Mind Scanner is safety infrastructure; as such it has its own invariants:

### 6.1 Fail-open vs fail-closed semantics

- In soft modes, TDA failures (e.g. library issues) must not break the system; they just disable monitoring.
- In hard modes, TDA failures should be treated as WARN at minimum, and possibly escalate to BLOCK for safety-critical runs.

### 6.2 Non-interference with ledger semantics

- TDA outputs MUST NOT change the semantics of what it means to be a verified proof in the ledger.
- They only affect which proofs and traces we allow to be proposed and fed into RFL.

### 6.3 Monotone log of TDA assessments

- All HSS/SNS/PCS/DRS computations should be logged in an append-only fashion, with references to:
  - the relevant run id, slice, statement hash, and epoch root (H_t).
- This allows post-hoc forensic analysis.

### 6.4 Versioning

- TDA pipeline versions (e.g. choice of filtration, thresholds) must be versioned just like hash/attestation algorithms.
- Changes to TDA pipeline must be flagged in governance so comparisons across runs are meaningful.

---

## 7. Implementation Notes

### 7.1 TDA Backend

We anticipate using an existing TDA library:

- **Ripser** or **GUDHI** for persistent homology,
- **networkx** for graph handling,
- **numpy** for embeddings.

The backend interface (`TDABackend`) should abstract away library details:

```python
class TDABackend(Protocol):
    def build_combinatorial_complex(self, local_dag: "nx.DiGraph") -> SimplicialComplex: ...
    def build_metric_complex(self, embeddings: Dict[str, "np.ndarray"], max_dim: int) -> "TDAResult": ...
    def compute_betti(self, comb_complex: SimplicialComplex) -> Dict[int, int]: ...
```

### 7.2 Performance Considerations

- Limit local DAG extraction to a bounded neighborhood (e.g., depth ≤ 3 from target).
- Limit the number of states in the embedding window (e.g., last 50–200).
- Limit maximum simplex dimension and homology dimension to 1 or 2.
- Run TDA in a separate worker pool if necessary.

---

## 8. Open Questions & Future Extensions

1. **Higher-dimensional structure**: Do certain slices require tracking H_2 (voids/cavities) for meaningful structure?

2. **Task-aware reference profiles**: Should we maintain per-slice & per-task reference TDA profiles?

3. **Cross-slice topology**: Can we exploit TDA across slices to detect global reasoning pathologies?

4. **Meta-learning regularization**: How exactly to couple HSS into DiscoRL-style meta-objectives in practice?

These will be addressed in future versions of this spec once v0 is implemented and validated.

---

## 9. Summary

This specification promotes the TDA Mind Scanner from an idea to an explicit architectural and mathematical contract:

| Aspect | Status |
|--------|--------|
| **Math codified** | SNS, PCS, DRS, HSS are formally defined in terms of clique complexes, persistent homology, and bottleneck distances. |
| **Architecture defined** | TDAMonitor is a sidecar component attached to U2Runner / planner / RFL, with clear API and gating semantics. |
| **Foundation cited** | The design is grounded in standard TDA as surveyed by Wasserman (2016): ridges ↔ PCS, homology ↔ SNS, and persistence ↔ robustness against noise. |

This is MathLedger's Phase V "CORTEX" asset: a geometric integrity monitor for cognition that scales to alien representations and self-modifying policies.

---

## Appendix A: Quick Reference

### A.1 Score Formulas

| Score | Formula | Range |
|-------|---------|-------|
| **SNS** | f_size · f_topo | [0, 1] |
| **PCS** | L_long^{(1)} / L_total^{(1)} | [0, 1] |
| **DRS** | min(1, d_B / δ_max) | [0, 1] |
| **HSS** | clip((α·SNS + β·PCS - γ·DRS + γ) / (α+β+γ), 0, 1) | [0, 1] |

### A.2 Default Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| α | 0.4 | SNS weight |
| β | 0.4 | PCS weight |
| γ | 0.4 | DRS weight (penalty) |
| τ | 0.05 | Lifetime threshold for PCS |
| δ_max | 0.5 | Deviation normalization constant |
| θ_block | 0.2 | HSS threshold for BLOCK |
| θ_warn | 0.5 | HSS threshold for WARN |
| K_max | 3 | Maximum simplex dimension |
| k_max | 1 | Maximum homology dimension |

### A.3 File Locations (Planned)

```
backend/tda/
├── __init__.py
├── proof_complex.py      # SimplicialComplex, build_combinatorial_complex()
├── metric_complex.py     # build_metric_complex(), TDAResult
├── scores.py             # SNS, PCS, DRS, HSS computation
├── reference_profile.py  # ReferenceTDAProfile, offline calibration
├── runtime_monitor.py    # TDAMonitor, TDAMonitorConfig, TDAMonitorResult
└── backends/
    ├── __init__.py
    ├── ripser_backend.py
    └── gudhi_backend.py
```

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **Betti number (β_k)** | Rank of the k-th homology group; counts k-dimensional "holes" |
| **Bottleneck distance** | Metric between persistence diagrams measuring worst-case point matching |
| **Clique complex** | Simplicial complex where k-simplices correspond to (k+1)-cliques |
| **DRS** | Deviation-from-Reference Score |
| **Flag complex** | Same as clique complex |
| **HSS** | Hallucination Stability Score |
| **PCS** | Persistence Coherence Score |
| **Persistence diagram** | Multiset of (birth, death) pairs for homology classes |
| **Ridge** | Low-dimensional manifold where density is locally maximized |
| **SNS** | Structural Non-Triviality Score |
| **TDA** | Topological Data Analysis |
| **Vietoris–Rips complex** | Simplicial complex built from pairwise distances at scale ε |
