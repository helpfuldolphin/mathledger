# RFL Uplift Theory: Symbolic Dynamics of Abstention Decay

---

> **STATUS: PHASE II — THEORETICAL ANALYSIS ONLY**
>
> This document contains mathematical theory that has **NOT** been empirically verified in Evidence Pack v1.
>
> **What this document is:**
> - A theoretical framework proposing how abstention rates *should* behave under idealized conditions
> - Mathematical conjectures with pseudo-proofs, not rigorous proofs
> - Future work intended to guide Phase II experiments
>
> **What this document is NOT:**
> - A claim that logistic convergence has been observed in practice
> - A validated model fit to actual RFL run data
> - Part of the Phase I evidence base
>
> **Phase I evidence (the only hard facts):**
> - FO closed-loop test with attestation.json
> - 1000-cycle abstention comparison (fo_baseline vs fo_rfl)
> - Sealed manifests in evidence pack
>
> **No curve fitting, no parametric validation, no residual analysis has been performed.**
> All theorems below are conjectures awaiting experimental saturation.

---

## Abstract

This document derives the *expected* dynamics of the abstention rate under the Reinforcement From Learning (RFL) loop with entropy injection via Wide Slice exploration. We *conjecture* that the abstention process forms a supermartingale with negative drift. This is a **theoretical hypothesis**, not an empirically validated claim. The framework is provided as future work to guide Phase II experimentation.

---

## 1. Preliminaries and Definitions

### 1.1 State Space

Let $\mathcal{S}$ denote the space of mathematical statements expressible in the bounded axiomatic framework. Define:

- $\mathcal{K}_t \subset \mathcal{S}$: The **knowledge frontier** at time $t$ (verified statements)
- $\mathcal{U}_t = \mathcal{S} \setminus \mathcal{K}_t$: The **unknown region** at time $t$
- $\mathcal{A}_t \subset \mathcal{U}_t$: The **abstention set** — statements the system cannot derive

### 1.2 Abstention Rate

**Definition 1.1** (Abstention Rate). The abstention rate at time $t$ is:

$$A(t) = \frac{|\mathcal{A}_t \cap \mathcal{C}_t|}{|\mathcal{C}_t|}$$ 

where $\mathcal{C}_t$ is the set of **candidate statements** presented to the derivation engine at step $t$.

### 1.3 Information-Theoretic Quantities

**Definition 1.2** (Derivation Entropy). The entropy of the derivation distribution at time $t$:

$$H_t = -\sum_{s \in \mathcal{C}_t} p_t(s) \log p_t(s)$$ 

where $p_t(s)$ is the probability of successfully deriving statement $s$ given current knowledge $\mathcal{K}_t$.

**Definition 1.3** (Learning Signal). The learning signal $L_t$ is the information gain from step $t$:

$$L_t = H(\mathcal{K}_{t+1} | \mathcal{K}_t) = \log \frac{|\mathcal{K}_{t+1}|}{|\mathcal{K}_t|}$$ 

---

## 2. Wide Slice Entropy Injection

### 2.1 Slice Policy

The Wide Slice policy $\pi_W$ samples candidate statements from an enlarged region of the formula space:

$$\mathcal{C}_t^{(\pi_W)} = \{s \in \mathcal{S} : \text{depth}(s) \leq d_{\max}, \text{atoms}(s) \leq a_{\max}\}$$ 

with $d_{\max}, a_{\max}$ set to expand coverage beyond the current derivation frontier.

### 2.2 Variance Amplification

**Lemma 2.1** (Variance Under Wide Slice). Under the Wide Slice policy, the variance of derivability scores increases:

$$\text{Var}_{\pi_W}[D(s)] > \text{Var}_{\pi_N}[D(s)]$$ 

where $D(s) \in \{0, 1\}$ indicates successful derivation and $\pi_N$ is the narrow (conservative) policy.

*Proof Sketch.* The Wide Slice includes statements at varying distances from the knowledge frontier. Near-frontier statements have $p(D=1) \approx 1$, while far-frontier statements have $p(D=1) \approx 0$. This bimodal distribution maximizes variance compared to a narrow policy that samples only near-frontier statements with $p(D=1) \approx 0.5$. ∎

### 2.3 Learning Signal Amplification

**Proposition 2.2** (Entropy-Signal Correspondence). Higher variance in derivability implies higher learning signal:

$$\text{Var}[D] \uparrow \implies L_t \uparrow$$ 

*Proof.* By the data processing inequality and the structure of the derivation DAG:

$$L_t = I(\mathcal{K}_{t+1}; \mathcal{C}_t) \geq I(D; \mathcal{C}_t) = H(D) - H(D|\mathcal{C}_t)$$ 

Since $H(D)$ is maximized when $\text{Var}[D]$ is maximized (for binary $D$), higher variance yields higher mutual information, hence higher learning signal. ∎

---

## 3. Abstention Dynamics

### 3.1 The Abstention Process

Define the abstention process $\{X_t\}_{t \geq 0}$ as:

$$X_t = A(t) \cdot |\mathcal{U}_0|$$ 

This represents the absolute count of "persistently unknown" statements, scaled by initial uncertainty.

### 3.2 Filtration and Adaptedness

Let $\mathcal{F}_t = \sigma(\mathcal{K}_0, \mathcal{K}_1, \ldots, \mathcal{K}_t)$ be the natural filtration. The process $X_t$ is $\mathcal{F}_t$-adapted since $\mathcal{A}_t$ is determined by $\mathcal{K}_t$.

### 3.3 Main Conjecture

**Conjecture 3.1** (Supermartingale Property). *[UNVERIFIED — Phase II]* Under entropy injection via Wide Slice policy, the abstention process $\{X_t\}_{t \geq 0}$ is conjectured to be a supermartingale with negative drift. Specifically:

$$\mathbb{E}[X_{t+1} | \mathcal{F}_t] \leq X_t - \delta_t$$ 

where $\delta_t > 0$ is the **drift coefficient** satisfying:

$$\delta_t \geq c \cdot \text{Var}_{\pi_W}[D] \cdot (1 - A(t))$$ 

for some constant $c > 0$ depending on the axiom system.

*Proof.*

**Step 1: Decomposition of Abstention Change**

$$X_{t+1} - X_t = -|\mathcal{A}_t \cap \mathcal{K}_{t+1}| + |\mathcal{A}_{t+1} \cap \mathcal{C}_{t+1}| - |\mathcal{A}_t \cap \mathcal{C}_t|$$ 

The first term represents statements that *leave* the abstention set (successful derivations). The remaining terms account for the change in candidate sets.

**Step 2: Bounding the Exit Rate**

Under Wide Slice, the probability of deriving a previously-abstained statement increases due to:
1. **Path Discovery**: New derivations create intermediate lemmas
2. **Frontier Expansion**: $\partial \mathcal{K}_t$ grows, enabling more Modus Ponens applications

Let $\rho_t = \mathbb{P}(s \in \mathcal{K}_{t+1} | s \in \mathcal{A}_t)$ be the exit probability. Then:

$$\mathbb{E}[|\mathcal{A}_t \cap \mathcal{K}_{t+1}|] = \rho_t \cdot |\mathcal{A}_t \cap \mathcal{C}_t|$$ 

**Step 3: Variance-Exit Relationship**

The exit probability $\rho_t$ is bounded below by the learning signal:

$$\rho_t \geq \frac{L_t}{H_{\max}} \geq \frac{c' \cdot \text{Var}[D]}{H_{\max}}$$ 

where $H_{\max} = \log |\mathcal{C}_t|$ is the maximum entropy.

**Step 4: Drift Bound**

Combining steps:

$$\mathbb{E}[X_{t+1} - X_t | \mathcal{F}_t] \leq -\rho_t \cdot X_t \cdot \frac{|\mathcal{C}_t|}{ |\mathcal{U}_0|} \leq -c \cdot \text{Var}[D] \cdot (1 - A(t))$$ 

The factor $(1 - A(t))$ appears because higher knowledge coverage enables more derivation pathways. ∎

---

## 4. Expected Shape of A(t)

### 4.1 Differential Approximation

For large $|\mathcal{S}|$, the abstention rate evolves approximately as:

$$\frac{dA}{dt} = -\lambda(t) \cdot A(t) \cdot (1 - A(t))$$ 

where $\lambda(t) = c \cdot \text{Var}_{\pi_W}[D(t)]$ is the time-varying learning rate.

### 4.2 Solution Form

**Conjecture 4.1** (Logistic Decay). *[UNVERIFIED — Phase II]* If $\lambda(t) \approx \lambda$ is approximately constant, then:

$$A(t) = \frac{A_0 \cdot e^{-\lambda t}}{1 - A_0 + A_0 \cdot e^{-\lambda t}}$$ 

This is a **logistic decay** curve with:
- Initial abstention rate $A_0 = A(0)$
- Decay rate $\lambda$ proportional to slice variance
- Asymptotic limit $\lim_{t \to \infty} A(t) = 0$

### 4.3 Phase Characterization

The abstention curve exhibits three phases:

| Phase | Time Range | Behavior | Mechanism |
|-------|------------|----------|-----------|
| **I. Plateau** | $t < t_1$ | $A(t) \approx A_0$ | Building foundational lemmas |
| **II. Descent** | $t_1 < t < t_2$ | $A(t)$ drops rapidly | Combinatorial explosion of derivations |
| **III. Tail** | $t > t_2$ | $A(t) \to 0$ slowly | Diminishing returns on easy proofs |

The transition times $t_1, t_2$ depend on the axiom system's branching factor.

---

## 5. Heteroskedasticity of Observations

### 5.1 Observation Noise Model

Empirical measurements of $\hat{A}(t)$ include noise:

$$\hat{A}(t) = A(t) + \epsilon_t$$ 

where $\epsilon_t$ is zero-mean but **heteroskedastic**:

$$\text{Var}(\epsilon_t) = \sigma^2(t) = \frac{A(t)(1 - A(t))}{n_t}$$ 

with $n_t = |\mathcal{C}_t|$ the sample size at step $t$.

### 5.2 Implications for Inference

**Proposition 5.1** (Variance Evolution). The observation variance follows:

$$\sigma^2(t) \propto A(t)(1 - A(t))$$ 

This is maximized at $A(t) = 0.5$ (Phase II) and minimized at the boundaries (Phases I and III).

**Corollary 5.2** (Confidence Interval Width). Bootstrap confidence intervals should be wider during Phase II descent and narrower during plateau/tail phases.

---

## 6. Convergence Guarantees

### 6.1 Almost Sure Convergence

**Conjecture 6.1** (Convergence). *[UNVERIFIED — Phase II]* Under the Wide Slice policy with bounded candidate sets:

$$A(t) \xrightarrow{a.s.} 0 \text{ as } t \to \infty$$ 

*Proof Sketch.* By Theorem 3.1, $\{X_t\}$ is a non-negative supermartingale. By the Martingale Convergence Theorem, $X_t \to X_\infty$ a.s. for some $X_\infty \geq 0$.

The negative drift condition $\delta_t \geq c \cdot (1 - A(t))$ implies that if $X_\infty > 0$, then $\sum_t \delta_t = \infty$, contradicting convergence. Hence $X_\infty = 0$ a.s., and therefore $A(t) \to 0$. ∎

### 6.2 Rate of Convergence

**Conjecture 6.2** (Convergence Rate). *[UNVERIFIED — Phase II]* The abstention rate decays at least exponentially:

$$A(t) \leq A_0 \cdot e^{-\bar{\lambda} t}$$ 

where $\bar{\lambda} = \inf_{s \leq t} \lambda(s) > 0$ under sustained entropy injection.

---

## 7. Summary of Conjectures

> **All results below are UNVERIFIED conjectures for Phase II investigation.**
> None have been validated against actual RFL run data.

| Result | Statement | Status |
|--------|-----------|--------|
| **Lemma 2.1** | Wide Slice increases derivability variance | Conjecture |
| **Proposition 2.2** | Higher variance yields higher learning signal | Conjecture |
| **Conjecture 3.1** | Abstention process is a supermartingale with negative drift | **UNVERIFIED** |
| **Conjecture 4.1** | Abstention follows logistic decay | **UNVERIFIED** |
| **Proposition 5.1** | Observation noise is heteroskedastic | Conjecture |
| **Conjecture 6.1** | Abstention converges to zero almost surely | **UNVERIFIED** |
| **Conjecture 6.2** | Convergence is at least exponential | **UNVERIFIED** |

---

## 8. Practical Implications (Phase II — If Theory Validates)

> **These recommendations are contingent on empirical validation of the above conjectures.**

1. **Window Size Selection**: Use windows capturing Phase II for maximum signal
2. **Smoothing**: Apply variance-weighted smoothing to account for heteroskedasticity
3. **Monotonicity Testing**: Expect descending trend; deviations indicate policy failure
4. **Bootstrap CI**: Use stratified bootstrap respecting phase boundaries

---

## 9. Relation to Phase I Evidence

**What Phase I actually demonstrated:**
- The FO closed-loop test showed the RFL harness can execute derivation cycles
- Abstention counts were recorded in fo_baseline and fo_rfl directories

**What Phase I did NOT demonstrate:**
- That abstention follows a logistic decay curve
- That the supermartingale property holds
- Any parametric fit quality or goodness-of-fit metrics
- Convergence rates or asymptotic behavior
- Any measurable "uplift" in the sense defined by this theory

---

### 9.1 Phase-I RFL Empirical Reality

> **SOBER TRUTH: The actual RFL logs do not demonstrate uplift.**

The following RFL output files exist on disk:

| File | Cycles | Outcome | Uplift Evidence |
|------|--------|---------|-----------------|
| `fo_rfl_50.jsonl` | ~50 | Partial, small exploratory run | **None** |
| `fo_rfl.jsonl` | 330 | All-abstain (100% abstention throughout) | **None** |

**Critical clarification:**

> **None of these logs constitute empirical uplift.**

- `fo_rfl_50.jsonl`: A short exploratory run, insufficient sample size for any statistical claim
- `fo_rfl.jsonl`: 330 cycles with **zero successful derivations** — the system abstained on every candidate

The theoretical framework in this document (logistic decay, supermartingale drift, convergence) **has no empirical support** in the Phase I evidence. The RFL loop executed, but did not produce the abstention reduction that would validate these conjectures.

**Implication:** All conjectures in Sections 3–6 remain purely theoretical. Phase II would need to:
1. Debug why the 330-cycle run produced all-abstain
2. Achieve actual derivation successes under RFL
3. Only then attempt curve fitting and conjecture validation

---

**This document is future work.** The theoretical framework here is intended to guide Phase II experiments where we would:
1. Instrument derivation runs to capture per-step abstention rates
2. Fit the logistic model and compute residuals
3. Test the supermartingale property empirically
4. Validate or refute the conjectures above

---

### 9.2 Phase II Uplift Experiment Family

> **FIREWALL: Phase I = negative control (all-abstain, Lean-disabled). Phase II uplift protocols = NOT YET RUN.**
>
> The experimental designs below are **preregistration** for future work. None have been executed.

#### 9.2.1 Why Phase I Shows All-Abstain

Before designing Phase II, we must understand why Phase I produced degenerate results:

| Factor | Phase I Configuration | Consequence |
|--------|----------------------|-------------|
| Verifier | Truth-table only, Lean disabled | Cannot verify non-tautological derivations |
| Slice | First-organism PL (atoms ≤ 2, depth ≤ 3) | Extremely narrow; most candidates trivially decidable or unreachable |
| Policy | Static; no learnable parameters | Nothing for RFL to optimize |
| Candidate source | Fixed axiom instantiation | No exploration beyond initial frontier |

**Root cause conjecture:** The Phase I slice is *epistemically closed* under truth-table verification—every statement is either trivially true (tautology) or trivially undecidable (not a tautology). There is no "learnable boundary" for RFL to exploit.

#### 9.2.2 Experimental Regime A: Lean-Enabled PL with Derivation Depth

**Objective:** Create a slice where some statements are provable *only* via multi-step derivation, not direct truth-table check.

**Configuration:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Logic | Propositional Logic | Continuity with Phase I |
| Verifier | **Lean-enabled** (truth-table fallback) | Enables derivation-based proofs |
| Slice | atoms ≤ 4, depth ≤ 6 | Larger than Phase I; includes non-trivial derivations |
| Axioms | Łukasiewicz 3-axiom system | Well-studied; known derivation complexity |
| Candidate source | Modus Ponens closure + random instantiation | Mix of reachable and unreachable |

**Policy Space (what RFL can learn):**

```
π_θ : (knowledge_frontier, candidate) → {attempt_derive, abstain, prioritize}
```

- **Tactic prior**: P(apply_MP | pattern) — learned from successful derivations
- **Depth budget allocation**: How many inference steps to attempt before abstaining
- **Candidate ranking**: Which candidates to attempt first given limited compute

**Epistemic Risk Functional:**

$$R(\pi) = \alpha \cdot \text{AbstentionRate}(\pi) + \beta \cdot \text{FalsePositiveRate}(\pi) + \gamma \cdot \text{ComputeCost}(\pi)$$ 

where:
- AbstentionRate = fraction of provable statements on which we abstain
- FalsePositiveRate = fraction of claimed proofs that fail Lean verification (should be 0)
- ComputeCost = average inference steps per candidate

**Conjecture A.1** *[Phase II — Not Yet Tested]*: Under Regime A, RFL should reduce abstention from α₀ ≈ 0.7 to α₁ ≈ 0.3 over T = 500 cycles, as the tactic prior learns which MP applications are productive.

**Conjecture A.2** *[Phase II — Not Yet Tested]*: The abstention curve should exhibit Phase II descent (Section 4.3) once the policy discovers key "lemma bridges" that unlock clusters of derivations.

**Observable signature:** Stepwise drops in abstention correlated with discovery of high-utility intermediate lemmas.

---

#### 9.2.3 Experimental Regime B: Equational Theory (Group Axioms)

**Objective:** Test RFL on a richer algebraic structure where proof search is genuinely hard.

**Configuration:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Logic | Equational logic with equality | First-order fragment |
| Theory | Group axioms (associativity, identity, inverse) | Simple but non-trivial |
| Verifier | Lean-enabled (equational rewriting) | Required for equality proofs |
| Slice | term depth ≤ 5, variables ≤ 3 | Tractable but not trivial |
| Candidate source | Random term equations + known theorems | Mix of true/false/hard |

**Policy Space:**

```
π_θ : (rewrite_state, goal) → {apply_rewrite_rule, orient_equation, abstain}
```

- **Rewrite ordering**: Which rule to apply first (associativity vs. inverse)
- **Orientation heuristic**: For unoriented equations, which direction to rewrite
- **Depth-first vs. breadth-first**: Search strategy selection

**Epistemic Risk Functional:**

$$R(\pi) = \alpha \cdot \text{Abstention} + \beta \cdot \text{ProofLength} + \gamma \cdot \text{SearchNodes}$$ 

ProofLength and SearchNodes measure efficiency, not just correctness.

**Conjecture B.1** *[Phase II — Not Yet Tested]*: Under Regime B, RFL should learn to prioritize associativity rewrites that "normalize" terms, reducing average proof length by 30-50% over T = 1000 cycles.

**Conjecture B.2** *[Phase II — Not Yet Tested]*: The "hard slice boundary" (statements requiring > 10 rewrite steps) should contract as RFL discovers efficient rewrite sequences.

**Observable signature:** Bimodal distribution of proof lengths collapses toward the shorter mode as policy improves.

---

#### 9.2.4 Minimum Viable Uplift Criteria

For any Phase II run to constitute "uplift evidence," it must satisfy:

1. **Non-degenerate baseline**: The Lean-enabled verifier must successfully verify ≥ 10% of candidates in the control condition (no RFL, random policy)
2. **Measurable improvement**: RFL condition must show statistically significant reduction in abstention (p < 0.05, Mann-Whitney U)
3. **No false positives**: Zero Lean verification failures in the RFL condition
4. **Reproducibility**: Effect must replicate across ≥ 3 independent runs with different random seeds

**Phase I failed criterion #1**: Truth-table-only verification produced 0% successful derivations, making uplift measurement impossible.

---

#### 9.2.5 Tie to Phase I All-Abstain Result

The Phase I all-abstain result is **informative**, not a failure:

| Phase I Observation | Phase II Design Implication |
|---------------------|----------------------------|
| 100% abstention | Must enable Lean verifier for non-tautological proofs |
| 330 cycles, no progress | Policy must have learnable parameters (Phase I had none) |
| Narrow slice (atoms ≤ 2) | Must expand slice to include derivation-requiring statements |
| No intermediate lemmas discovered | Must seed with or enable discovery of useful lemmas |

**Conjecture 9.1** *[Phase II — Not Yet Tested]*: Phase I all-abstain is a necessary negative control demonstrating that uplift requires (a) a non-trivial verification pathway and (b) a learnable policy. Without both, RFL reduces to random search over an epistemically closed space.

---

---

## 10. U1 Experiment Status (Phase II Preregistration)

> **STATUS: PREREGISTERED BUT NOT EXECUTED**
>
> Experiment U1 (`uplift_u1_slice_medium`) is preregistered in `experiments/prereg/PREREG_UPLIFT_U1.md`
> but **no experimental runs have been performed**. The `results/uplift_u1/` directory does not exist.

> **CANONICAL STATUS: U1 is the canonical first uplift experiment for Phase II.**
>
> - U1 uses `slice_medium` with truth-table verification (no Lean complexity)
> - Effect size threshold: **10 percentage points** (0.10 absolute reduction in abstention)
> - Lean-enabled uplift slices (`slice_pl_uplift_a/b/c`) are deferred to **Phase IIb**
> - U1b (Lean-enabled) will only be designed **after U1 results are analyzed**

### 10.1 U1 Preregistration Summary

| Field | Value |
|-------|-------|
| Experiment ID | `uplift_u1_slice_medium` |
| Status | **CANONICAL — PREREGISTERED — NOT EXECUTED** |
| Slice | `slice_medium` (atoms=5, depth_max=7) |
| Verifier | Truth-table-only (no Lean) |
| Effect Size Threshold | **≥ 10 percentage points** (0.10) |
| Planned Cycles | 500 baseline + 500 RFL |
| Results Directory | `results/uplift_u1/` — **DOES NOT EXIST** |

### 10.2 Why U1 Has Not Been Run

The Phase I runs (`fo_rfl.jsonl`, `fo_rfl_1000.jsonl`) used:
- `method='lean-disabled'` → All verification attempts abstain
- `slice_name='first-organism-pl'` → Not the `slice_medium` in U1 prereg
- 100% abstention rate throughout all 1000+ cycles

**Result:** Phase I runs are plumbing tests that verify RFL metabolism executes. They do not constitute uplift experiments because:
1. The verifier is disabled (always abstains)
2. There is no learnable boundary for RFL to exploit
3. Baseline and RFL produce identical 100% abstention

### 10.3 Phase I vs U1 Comparison

| Attribute | Phase I (Actual) | U1 (Preregistered) |
|-----------|------------------|---------------------|
| Slice | `first-organism-pl` | `slice_medium` |
| Atoms | ~2-3 | 5 |
| Depth | ~3 | 7 |
| Verifier | Disabled (mock) | Truth-table (real) |
| Expected Abstention | 100% (degenerate) | 30-70% (non-degenerate) |
| Uplift Measurable | **NO** | **YES (if run)** |

### 10.4 U1 Outcome Classification (Pending)

When U1 is eventually executed, outcomes will be classified as:

| Validity | Uplift Criteria | Classification |
|----------|-----------------|----------------|
| Invalid (α outside 0.10-0.80) | N/A | **INVALID** — Experiment failed validity |
| Valid | α_rfl ≥ α_baseline | **NULL** — No uplift detected |
| Valid | α_rfl < α_baseline but Δ < 10pp | **NULL** — Trend only, not uplift |
| Valid | α_rfl < α_baseline, Δ ≥ 10pp, CI excludes 0 | **POSITIVE** — Uplift detected |

**Effect Size Definition:**
> Uplift = absolute abstention reduction of at least **10 percentage points** (0.10),
> with a 95% bootstrap confidence interval that excludes zero.

**Current Status:** Cannot classify — experiment not run.

### 10.5 Relationship to Phase II Uplift Slices

U1 uses `slice_medium` which differs from the Lean-enabled uplift slices designed in `docs/curriculum_gate_equations.md` Section 10:

| Slice | Verifier | Status | Phase |
|-------|----------|--------|-------|
| `slice_medium` | Truth-table | **CANONICAL U1** — Preregistered, not run | **Phase II** |
| `slice_pl_uplift_a` | Lean-enabled | Design only, deferred | Phase IIb |
| `slice_pl_uplift_b` | Lean-enabled | Design only, deferred | Phase IIb |
| `slice_pl_uplift_c` | Lean-enabled | Design only, deferred | Phase IIb |

**Sequencing:**
1. **Run U1 first** — Truth-table verification on `slice_medium` is simpler and safer
2. **Analyze U1 results** — Determine if INVALID/NULL/POSITIVE
3. **Only then design U1b** — Lean-enabled experiments are Phase IIb, contingent on U1 outcome

**None of these slices have been exercised in actual experiments.**

---

## References

- Doob, J.L. (1953). *Stochastic Processes*. Wiley.
- Williams, D. (1991). *Probability with Martingales*. Cambridge.
- Cover, T.M. & Thomas, J.A. (2006). *Elements of Information Theory*. Wiley.

---

*Document generated for MathLedger RFL Loop Analysis*

---

**PHASE II — NOT PART OF EVIDENCE PACK v1**

### Document Revision History

| Date | Revision | Author |
|------|----------|--------|
| 2025-11-30 | Added Section 10 (U1 Experiment Status) | CLAUDE G |
| 2025-11-30 | Clarified U1 is preregistered but not executed | CLAUDE G |
| 2025-11-30 | Added Phase I vs U1 comparison table | CLAUDE G |

---

## 11. Formalism for Phase II Asymmetric Uplift Analysis

> **STATUS: PHASE II — THEORETICAL ANALYSIS ONLY**
>
> This section extends the prior theoretical framework to the specific context of the four asymmetric uplift experiments (U2, U3, U4, U5) planned for Phase II. These experiments introduce heterogeneous verification environments. The formalism below is preregistration for the analysis of these future experiments.
>
> **ABSOLUTE SAFEGUARD:** This theory is speculative and MUST NOT be interpreted as empirical evidence. No experiments related to these slices have been run.

---

### 11.1 Slice-Specific Success Metrics

Let $\mathcal{C}_i$ be the candidate set for slice $i \in \{U2, U3, U4, U5\}$. Let $\pi$ be the RFL policy and $\pi_0$ be the baseline (random) policy. The primary success metric is the **abstention rate**, $A_i(\pi, t)$, for slice $i$ under policy $\pi$ at cycle $t$.

**Definition 11.1 (Uplift Gain).** The raw uplift gain for slice $i$ at time $T$ is the reduction in abstention rate compared to baseline:

$$ G_i(\pi, T) = A_i(\pi_0, T) - A_i(\pi, T) $$ 

**Definition 11.2 (Slice-Specific Success Criterion).** An experiment on slice $i$ demonstrates **conclusive uplift** if, over $N$ cycles, the time-averaged uplift gain is statistically significant and exceeds a preregistered threshold $\tau_i$.

$$ \frac{1}{N} \sum_{t=1}^{N} G_i(\pi, t) > \tau_i \quad \text{and} \quad p\text{-value}(\text{H}_0: \mathbb{E}[G_i] \leq 0) < 0.05 $$ 

The thresholds $\{\tau_i\}$ are set based on the hypothesized difficulty of each slice:
- **$\tau_{U2}$ (Tautology Enrichment):** High expected uplift. $\tau_{U2} = 0.20$. The enriched density of tautologies should be easily exploited.
- **$\tau_{U3}$ (Contradiction Enrichment):** Medium expected uplift. $\tau_{U3} = 0.15$. Identifying contradictions is harder than tautologies.
- **$\tau_{U4}$ (Conjecture Depletion):** Low expected uplift. $\tau_{U4} = 0.05$. A sparse environment offers fewer learning opportunities.
- **$\tau_{U5}$ (Axiom Expansion):** High but variable uplift. $\tau_{U5} = 0.10$. Success depends on the discovery of a few key lemmas.

---

### 11.2 Stochastic Approximation Framing

Uplift estimation can be framed as a stochastic approximation problem. The RFL policy $\pi_\theta$ is parameterized by $\theta \in \Theta$. The goal is to find $\theta^*$ that minimizes the expected abstention rate.

Let $J_i(\theta) = \mathbb{E}[A_i(\pi_\theta)]$. The RFL update rule is a stochastic gradient descent step on this objective:

$$ \theta_{t+1} = \theta_t - \alpha_t \nabla_\theta \hat{J}_i(\theta_t) $$ 

where $\hat{J}_i(\theta_t)$ is a noisy measurement of the abstention rate on a batch of candidates from slice $i$ at cycle $t$.

**Definition 11.3 (Uplift Process as a Robbins-Monro Algorithm).** Let $X_t = A_i(\pi_{\theta_t}, t)$ be the abstention rate. The learning process seeks the root of the function $f(\theta) = \nabla_\theta J_i(\theta)$. The update is:

$$ \theta_{t+1} = \theta_t - \alpha_t h(C_t, \theta_t) $$ 

where $h(C_t, \theta_t)$ is an unbiased estimator of $\nabla_\theta J_i(\theta)$ computed from candidate set $C_t$. This is a Robbins-Monro process.

---

### 11.3 Convergence Criteria for Asymmetric Slices

Convergence in Phase II is defined by the stability of the policy and the stationarity of the abstention rate.

**Criterion 11.1 (Policy Convergence).** The policy $\pi_{\theta_t}$ has converged at cycle $T$ if the parameter vector is stable:

$$ \frac{\|\theta_T - \theta_{T-k}\|_2}{\|\theta_T\|_2} < \epsilon_\theta \quad \text{for a window } k $$ 

This indicates the policy is no longer making significant updates.

**Criterion 11.2 (Abstention Rate Stationarity).** The uplift process for slice $i$ has converged if the time-series of the abstention rate $A_i(t)$ becomes stationary. This can be tested via an Augmented Dickey-Fuller (ADF) test on the recent history of $A_i(t)$.

$$ \text{ADF-statistic}( \{A_i(t)\}_{t=T-k}^T ) < \text{critical value} $$ 

**Criterion 11.3 (Uplift Saturation).** Uplift is saturated when the marginal gain approaches zero. Let $\bar{G}_i(T) = \frac{1}{T}\sum_{t=1}^T G_i(t)$. Saturation occurs when:

$$ \frac{d\bar{G}_i(T)}{dT} < \epsilon_g $$ 

This means that further training yields diminishing returns.

---

### 11.4 Phase II Stability Considerations

The introduction of asymmetric slices and more complex verification (e.g., Lean-enabled) introduces new stability risks.

**Risk 1: Policy Oscillation.** In a heterogeneous environment (e.g., mixing candidates from U2 and U4), the optimal policy may oscillate. A policy specialized for tautology detection (U2) may perform poorly on conjecture-sparse slices (U4), and vice-versa.
- **Mitigation:** Implement separate policy heads for different slice types or use a Mixture-of-Experts model.

**Risk 2: Verifier Latency Exploits.** If the Lean verifier has non-uniform response times (e.g., faster for trivial tautologies), the RFL loop could learn to "game the system" by prioritizing candidates that are cheap to verify, regardless of their epistemic value.
- **Mitigation:** Normalize rewards by verification cost or use a fixed-time budget for verification attempts. This is part of the Epistemic Risk Functional (Section 9.2.2).

**Risk 3: Catastrophic Forgetting.** A policy trained extensively on one slice (e.g., U5 with new axioms) might unlearn heuristics valuable for the base theory.
- **Mitigation:** Use experience replay with samples from all slices, or employ regularization techniques like Elastic Weight Consolidation (EWC).

---

### 11.5 Expected Behavior Under Asymmetric Slices

The four uplift slices are designed to probe different aspects of the RFL process.

- **U2 (Tautology-Enriched):** We expect a rapid and significant decrease in the abstention rate. This slice is "easy" and serves as a positive control. The learning curve for $A_{U2}(t)$ should resemble the idealized logistic decay (Conjecture 4.1) most closely.

- **U3 (Contradiction-Enriched):** We expect a slower decrease in abstention compared to U2. Proving a contradiction $`\neg P`$ is equivalent to proving $`P \rightarrow false`$, which can be harder. The uplift gain $G_{U3}$ should be positive but smaller than $G_{U2}$.

- **U4 (Conjecture-Depleted):** We expect a very small, possibly zero, uplift gain. With few provable statements, there is little learning signal. The abstention rate $A_{U4}(t)$ should remain high and flat. This slice acts as a negative control for the learning process itself. A significant drop in $A_{U4}(t)$ would be anomalous and suggest the policy is learning a spurious correlation.

- **U5 (Axiom-Expansion):** We expect a "step-function" behavior in the abstention rate. The rate $A_{U5}(t)$ should remain high until the RFL policy discovers how to use the new axiom to prove a key lemma. Upon discovery, the abstention rate should drop sharply as many previously unprovable statements become derivable. The timing and magnitude of this drop are stochastic.