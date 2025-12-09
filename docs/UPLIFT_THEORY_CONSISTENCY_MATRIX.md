# UPLIFT_THEORY_CONSISTENCY_MATRIX.md

> **STATUS: PHASE II — NOT RUN IN PHASE I**
>
> This document provides the authoritative binding between theoretical constructs
> (RFL_UPLIFT_THEORY.md) and their implementation counterparts across all system layers.

---

## 1. Document Purpose

This matrix ensures **end-to-end traceability** from:

```
Theory (Mathematical Definition)
    ↓
Implementation (Python Code / JSON Schema)
    ↓
Metric (Computed Value)
    ↓
Telemetry (Redis / JSONL Output)
    ↓
Manifest (Experiment Record)
```

Every theoretical variable must map cleanly through this chain. Any gap indicates
an implementation deficiency or a theory revision requirement.

---

## 2. Master Consistency Matrix

### 2.1 Slice A: Goal-Hit Metric (`slice_uplift_goal`)

| Theory Variable | Definition | Implementation Field | Metric Computation | Telemetry Key | Manifest Field |
|-----------------|------------|---------------------|-------------------|---------------|----------------|
| $\mathcal{C}_t$ | Candidate set at cycle $t$ | `record["candidates"]["hashes"]` | `len(candidates.hashes)` | `ml:u2:cycle:{t}:candidates_count` | `results.cycles[t].candidates.total` |
| $\mathcal{V}_t$ | Verified set at cycle $t$ | `record["verified"]["hashes"]` | `len(verified.hashes)` | `ml:u2:cycle:{t}:verified_count` | `results.cycles[t].verified.count` |
| $\mathcal{G}$ | Target goal set | `record["goals"]["target_hashes"]` | Config-defined | `ml:u2:config:goal_hashes` | `configuration.goals.target_hashes` |
| $\mathbf{1}_{\text{goal}}(t)$ | Goal-hit indicator (Def 12.1) | `record["metrics"]["goal_hit"]` | `bool(set(V_t) & set(G))` | `ml:u2:cycle:{t}:goal_hit` | `results.cycles[t].metrics.goal_hit` |
| $\rho_{\text{goal}}(\pi, T)$ | Goal-hit rate (Def 12.2) | Derived post-run | `sum(goal_hit) / T` | `ml:u2:summary:goal_hit_rate` | `results.summary.goal_hit_rate` |
| $\text{success}_A(t)$ | Success criterion (Def 12.3) | Derived | `goal_hit AND len(V_t) >= 3` | `ml:u2:cycle:{t}:success_a` | `results.cycles[t].success` |
| $M_A(\pi, T)$ | Primary metric | Derived post-run | `mean(success_A)` | `ml:u2:summary:m_a` | `results.summary.primary_metric` |
| $\Delta_A(T)$ | Uplift | Derived post-run | `M_A(rfl) - M_A(baseline)` | `ml:u2:comparison:delta_a` | `comparison.delta` |

---

### 2.2 Slice B: Sparse Density Metric (`slice_uplift_sparse`)

| Theory Variable | Definition | Implementation Field | Metric Computation | Telemetry Key | Manifest Field |
|-----------------|------------|---------------------|-------------------|---------------|----------------|
| $\delta_t$ | Verification density (Def 12.4) | `record["metrics"]["verification_density"]` | `verified.count / candidates.total` | `ml:u2:cycle:{t}:density` | `results.cycles[t].metrics.verification_density` |
| $B_{\max}$ | Budget cap | Config constant | `40` | `ml:u2:config:budget_max` | `configuration.budget_max` |
| $\text{success}_B(t)$ | Success criterion (Def 12.5) | Derived | `verified.count >= 5 AND candidates.total <= 40` | `ml:u2:cycle:{t}:success_b` | `results.cycles[t].success` |
| $M_B(\pi, T)$ | Density-weighted success (Def 12.6) | Derived post-run | `sum(success_B * delta) / T` | `ml:u2:summary:m_b` | `results.summary.primary_metric` |
| $\bar{\delta}(\pi, T)$ | Mean density | Derived post-run | `mean(delta_t)` | `ml:u2:summary:mean_density` | `results.summary.mean_density` |

---

### 2.3 Slice C: Chain-Depth Metric (`slice_uplift_tree`)

| Theory Variable | Definition | Implementation Field | Metric Computation | Telemetry Key | Manifest Field |
|-----------------|------------|---------------------|-------------------|---------------|----------------|
| $d(v)$ | Proof depth (Def 12.7) | `record["verified"]["depths"][i]` | From proof DAG | `ml:u2:cycle:{t}:depths` | `results.cycles[t].verified.depths` |
| $d_{\min}$ | Minimum chain depth | Config constant | `3` | `ml:u2:config:d_min` | `configuration.d_min` |
| $\text{success}_C(t)$ | Chain-success (Def 12.8) | Derived | `any(d >= d_min for d in depths if h in G)` | `ml:u2:cycle:{t}:success_c` | `results.cycles[t].success` |
| $\bar{d}(\pi, T)$ | Avg realized depth (Def 12.9) | Derived post-run | `sum(all_depths) / count(all_depths)` | `ml:u2:summary:mean_depth` | `results.summary.mean_depth` |
| $M_C(\pi, T)$ | Chain-success rate | Derived post-run | `mean(success_C)` | `ml:u2:summary:m_c` | `results.summary.primary_metric` |
| $M_C^{(2)}(\pi, T)$ | Depth efficiency | Derived post-run | `mean_depth(rfl) - mean_depth(baseline)` | `ml:u2:comparison:depth_delta` | `comparison.depth_delta` |

---

### 2.4 Slice D: Joint-Goal Metric (`slice_uplift_dependency`)

| Theory Variable | Definition | Implementation Field | Metric Computation | Telemetry Key | Manifest Field |
|-----------------|------------|---------------------|-------------------|---------------|----------------|
| $\mathcal{G} = \{g_1, \ldots, g_k\}$ | Required goals (Def 12.10) | `record["goals"]["target_hashes"]` | Config-defined | `ml:u2:config:required_goals` | `configuration.goals.required_hashes` |
| $\text{success}_D(t)$ | Joint-success (Def 12.11) | `record["metrics"]["joint_success"]` | `set(G) <= set(V_t)` | `ml:u2:cycle:{t}:joint_success` | `results.cycles[t].metrics.joint_success` |
| $\kappa_t$ | Partial coverage (Def 12.12) | `record["metrics"]["partial_coverage"]` | `len(V_t & G) / len(G)` | `ml:u2:cycle:{t}:partial_coverage` | `results.cycles[t].metrics.partial_coverage` |
| $M_D(\pi, T)$ | Joint-success rate | Derived post-run | `mean(success_D)` | `ml:u2:summary:m_d` | `results.summary.primary_metric` |
| $M_D^{(2)}(\pi, T)$ | Mean partial coverage | Derived post-run | `mean(kappa_t)` | `ml:u2:summary:mean_partial` | `results.summary.mean_partial_coverage` |

---

### 2.5 Policy Parameters

| Theory Variable | Definition | Implementation Field | Metric Computation | Telemetry Key | Manifest Field |
|-----------------|------------|---------------------|-------------------|---------------|----------------|
| $\theta_t$ | Policy params (§13.2) | `record["policy"]["theta"]` | Direct read | `ml:u2:cycle:{t}:theta` | `results.cycles[t].policy.theta` |
| $\theta_t - \theta_{t-1}$ | Param delta | `record["policy"]["theta_delta"]` | `theta[t] - theta[t-1]` | `ml:u2:cycle:{t}:theta_delta` | `results.cycles[t].policy.theta_delta` |
| $\|\nabla J(\theta_t)\|_2$ | Gradient norm (§15.7) | `record["policy"]["gradient_norm"]` | L2 norm of gradient | `ml:u2:cycle:{t}:grad_norm` | `results.cycles[t].policy.gradient_norm` |
| $\alpha_t$ | Learning rate | `record["policy"]["learning_rate"]` | Scheduled or adaptive | `ml:u2:cycle:{t}:lr` | `results.cycles[t].policy.learning_rate` |

---

### 2.6 Convergence Diagnostics

| Theory Variable | Definition | Implementation Field | Metric Computation | Telemetry Key | Manifest Field |
|-----------------|------------|---------------------|-------------------|---------------|----------------|
| $\Psi_T^{(w)}$ | Policy stability index (Def 14.1) | Derived | `mean(norm(delta) / norm(theta))` over window | `ml:u2:diagnostic:psi` | `diagnostics.policy_stability_index` |
| $\mathcal{O}_T$ | Oscillation index (Def 15.2) | Derived | `count(reversals) / (T-2)` | `ml:u2:diagnostic:oscillation` | `diagnostics.oscillation_index` |
| $\text{Stationary}_T^{(i)}$ | Metric stationarity (Def 14.2) | Derived | ADF test p-value < 0.05 | `ml:u2:diagnostic:stationary` | `diagnostics.metric_stationary` |
| $W_T$ | CI width (Def 14.3) | Derived | `ci_upper - ci_lower` | `ml:u2:diagnostic:ci_width` | `diagnostics.ci_width` |

---

### 2.7 Abstention & Success Tracking

| Theory Variable | Definition | Implementation Field | Metric Computation | Telemetry Key | Manifest Field |
|-----------------|------------|---------------------|-------------------|---------------|----------------|
| $\mathcal{A}_t$ | Abstention set (§12.1) | `record["abstained"]["hashes"]` | `C_t - V_t` | `ml:u2:cycle:{t}:abstained` | `results.cycles[t].abstained.hashes` |
| $\alpha_t$ (abstention rate) | Abstention rate (Def 1.1) | `record["metrics"]["abstention_rate"]` | `len(A_t) / len(C_t)` | `ml:u2:cycle:{t}:abstention_rate` | `results.cycles[t].metrics.abstention_rate` |
| $H_t$ | Composite root | `record["H_t"]` | SHA256 attestation | `ml:u2:cycle:{t}:H_t` | `results.cycles[t].H_t` |

---

## 3. Telemetry Schema (Redis Keys)

### 3.1 Per-Cycle Keys (Ephemeral)

```
ml:u2:cycle:{cycle_id}:candidates_count    → int
ml:u2:cycle:{cycle_id}:verified_count      → int
ml:u2:cycle:{cycle_id}:abstention_rate     → float
ml:u2:cycle:{cycle_id}:goal_hit            → bool
ml:u2:cycle:{cycle_id}:joint_success       → bool
ml:u2:cycle:{cycle_id}:partial_coverage    → float
ml:u2:cycle:{cycle_id}:density             → float
ml:u2:cycle:{cycle_id}:depths              → list[int]
ml:u2:cycle:{cycle_id}:theta               → list[float]
ml:u2:cycle:{cycle_id}:theta_delta         → list[float]
ml:u2:cycle:{cycle_id}:grad_norm           → float
ml:u2:cycle:{cycle_id}:H_t                 → string (64-char hex)
ml:u2:cycle:{cycle_id}:timestamp           → string (ISO8601)
```

### 3.2 Summary Keys (Persistent)

```
ml:u2:summary:{experiment_id}:goal_hit_rate     → float
ml:u2:summary:{experiment_id}:mean_density      → float
ml:u2:summary:{experiment_id}:mean_depth        → float
ml:u2:summary:{experiment_id}:joint_success_rate → float
ml:u2:summary:{experiment_id}:primary_metric    → float
ml:u2:summary:{experiment_id}:total_cycles      → int
```

### 3.3 Diagnostic Keys

```
ml:u2:diagnostic:{experiment_id}:psi            → float
ml:u2:diagnostic:{experiment_id}:oscillation    → float
ml:u2:diagnostic:{experiment_id}:stationary     → bool
ml:u2:diagnostic:{experiment_id}:ci_width       → float
ml:u2:diagnostic:{experiment_id}:pattern        → string (e.g., "A.1")
```

### 3.4 Comparison Keys

```
ml:u2:comparison:{experiment_id}:delta          → float
ml:u2:comparison:{experiment_id}:ci_lower       → float
ml:u2:comparison:{experiment_id}:ci_upper       → float
ml:u2:comparison:{experiment_id}:excludes_zero  → bool
ml:u2:comparison:{experiment_id}:outcome        → string (POSITIVE/NULL/INVALID)
```

---

## 4. Manifest Schema Binding

### 4.1 Experiment Manifest Structure

```json
{
  "manifest_version": "2.0",
  "experiment_id": "<from PREREG_UPLIFT_U2.yaml>",
  "slice_name": "<slice_uplift_goal|sparse|tree|dependency>",
  "mode": "<baseline|rfl>",

  "preregistration": {
    "prereg_path": "experiments/prereg/PREREG_UPLIFT_U2.yaml",
    "prereg_hash_sha256": "<64-char hex>",
    "slice_config_hash": "<from prereg>"
  },

  "configuration": {
    "seed": "<int>",
    "cycles": "<int>",
    "budget_max": 40,
    "d_min": 3,
    "goals": {
      "target_hashes": ["<sha256>", ...],
      "required_hashes": ["<sha256>", ...]
    }
  },

  "execution": {
    "started_at": "<ISO8601>",
    "completed_at": "<ISO8601>",
    "duration_seconds": "<float>",
    "python_version": "<string>",
    "platform": "<string>"
  },

  "results": {
    "total_cycles": "<int>",
    "cycles": [
      {
        "cycle": 1,
        "candidates": {"total": "<int>", "hashes": [...]},
        "verified": {"count": "<int>", "hashes": [...], "depths": [...]},
        "abstained": {"count": "<int>", "hashes": [...]},
        "goals": {"hit_count": "<int>", "hit_hashes": [...]},
        "policy": {"theta": [...], "theta_delta": [...], "gradient_norm": "<float>"},
        "metrics": {
          "abstention_rate": "<float>",
          "verification_density": "<float>",
          "goal_hit": "<bool>",
          "joint_success": "<bool>",
          "partial_coverage": "<float>",
          "max_depth": "<int>"
        },
        "H_t": "<sha256>",
        "success": "<bool>"
      }
    ],
    "summary": {
      "goal_hit_rate": "<float>",
      "mean_density": "<float>",
      "mean_depth": "<float>",
      "joint_success_rate": "<float>",
      "mean_partial_coverage": "<float>",
      "primary_metric": "<float>",
      "abstention_rate_final": "<float>"
    }
  },

  "diagnostics": {
    "policy_stability_index": "<float>",
    "oscillation_index": "<float>",
    "metric_stationary": "<bool>",
    "ci_width": "<float>",
    "pattern_detected": "<string>",
    "convergence_achieved": "<bool>",
    "early_stop_reason": "<string|null>"
  },

  "comparison": {
    "baseline_manifest": "<path or null>",
    "delta": "<float>",
    "ci_95_lower": "<float>",
    "ci_95_upper": "<float>",
    "ci_excludes_zero": "<bool>",
    "outcome": "<POSITIVE|NULL|INVALID>",
    "outcome_reason": "<string>"
  },

  "artifacts": {
    "log_jsonl": "<path>",
    "attestation_json": "<path>",
    "diagnostic_json": "<path>"
  }
}
```

### 4.2 Theory → Manifest Field Mapping Table

| Theory Reference | Manifest Path | Type | Required |
|-----------------|---------------|------|----------|
| Def 12.1 ($\mathbf{1}_{\text{goal}}$) | `results.cycles[*].metrics.goal_hit` | bool | Yes (Slice A) |
| Def 12.2 ($\rho_{\text{goal}}$) | `results.summary.goal_hit_rate` | float | Yes (Slice A) |
| Def 12.4 ($\delta_t$) | `results.cycles[*].metrics.verification_density` | float | Yes (Slice B) |
| Def 12.7 ($d(v)$) | `results.cycles[*].verified.depths` | list[int] | Yes (Slice C) |
| Def 12.9 ($\bar{d}$) | `results.summary.mean_depth` | float | Yes (Slice C) |
| Def 12.11 ($\text{success}_D$) | `results.cycles[*].metrics.joint_success` | bool | Yes (Slice D) |
| Def 12.12 ($\kappa_t$) | `results.cycles[*].metrics.partial_coverage` | float | Yes (Slice D) |
| Def 14.1 ($\Psi_T^{(w)}$) | `diagnostics.policy_stability_index` | float | Yes |
| Def 15.2 ($\mathcal{O}_T$) | `diagnostics.oscillation_index` | float | Yes |
| §17.4 ($\hat{\Delta}_i$) | `comparison.delta` | float | Yes |
| Table 19.1 | `comparison.outcome` | string | Yes |

---

## 5. Conjecture Dependency Diagrams

### 5.1 Conjecture Dependency Graph

```
                    ┌─────────────────────────────────────────────────────────────┐
                    │                    PHASE II CONJECTURES                      │
                    │                   Dependency Structure                       │
                    └─────────────────────────────────────────────────────────────┘

    ┌───────────────────────────────────────────────────────────────────────────────────┐
    │                              FOUNDATIONAL LAYER                                   │
    │                                                                                   │
    │   ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐            │
    │   │  Lemma 2.1      │     │  Prop 2.2       │     │  Prop 5.1       │            │
    │   │  Variance       │────▶│  Learning       │     │  Heteroskedas-  │            │
    │   │  Amplification  │     │  Signal         │     │  ticity         │            │
    │   └────────┬────────┘     └────────┬────────┘     └─────────────────┘            │
    │            │                       │                                              │
    └────────────┼───────────────────────┼──────────────────────────────────────────────┘
                 │                       │
                 ▼                       ▼
    ┌───────────────────────────────────────────────────────────────────────────────────┐
    │                              DYNAMICS LAYER                                       │
    │                                                                                   │
    │   ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐            │
    │   │  Conj 3.1       │────▶│  Conj 4.1       │     │  Thm 13.2       │            │
    │   │  Supermartingale│     │  Logistic       │     │  Multi-Goal     │            │
    │   │  Property       │     │  Decay          │     │  Convergence    │            │
    │   └────────┬────────┘     └────────┬────────┘     └────────┬────────┘            │
    │            │                       │                       │                      │
    └────────────┼───────────────────────┼───────────────────────┼──────────────────────┘
                 │                       │                       │
                 ▼                       ▼                       ▼
    ┌───────────────────────────────────────────────────────────────────────────────────┐
    │                              CONVERGENCE LAYER                                    │
    │                                                                                   │
    │   ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐            │
    │   │  Conj 6.1       │     │  Conj 6.2       │     │  Thm 15.1       │            │
    │   │  A.S.           │     │  Exponential    │     │  Local          │            │
    │   │  Convergence    │     │  Rate           │     │  Stability      │            │
    │   └────────┬────────┘     └─────────────────┘     └────────┬────────┘            │
    │            │                                               │                      │
    └────────────┼───────────────────────────────────────────────┼──────────────────────┘
                 │                                               │
                 ▼                                               ▼
    ┌───────────────────────────────────────────────────────────────────────────────────┐
    │                              STRUCTURE LAYER                                      │
    │                                                                                   │
    │   ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐            │
    │   │  Conj 15.4      │     │  Thm 15.3       │     │  Thm 15.5       │            │
    │   │  Basin          │◀────│  Lyapunov       │     │  Trajectory     │            │
    │   │  Structure      │     │  Stability      │     │  Classification │            │
    │   └─────────────────┘     └─────────────────┘     └─────────────────┘            │
    │                                                                                   │
    └───────────────────────────────────────────────────────────────────────────────────┘


    LEGEND:
    ───▶  "depends on" / "requires"
    ◀───  "informs" / "constrains"
```

### 5.2 Slice-Specific Conjecture Relevance

```
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │                     SLICE → CONJECTURE RELEVANCE MATRIX                         │
    └─────────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
    │              │ Lemma   │ Prop    │ Conj    │ Conj    │ Thm     │ Conj    │ Conj    │
    │   SLICE      │ 2.1     │ 2.2     │ 3.1     │ 4.1     │ 13.2    │ 15.4    │ 6.1     │
    ├──────────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
    │ A (Goal)     │   ◐     │   ●     │   ●     │   ●     │   ●     │   ◐     │   ●     │
    ├──────────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
    │ B (Sparse)   │   ●     │   ●     │   ●     │   ◐     │   ◐     │   ●     │   ◐     │
    ├──────────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
    │ C (Chain)    │   ◐     │   ◐     │   ●     │   ◐     │   ●     │   ●     │   ●     │
    ├──────────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
    │ D (Joint)    │   ○     │   ○     │   ◐     │   ○     │   ●     │   ●     │   ◐     │
    └──────────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘

    LEGEND:
    ●  Primary relevance (slice directly tests conjecture)
    ◐  Secondary relevance (conjecture applies but not primary test)
    ○  Minimal relevance (conjecture may apply indirectly)
```

### 5.3 Evidence Flow Diagram

```
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │                         EVIDENCE FLOW: OBSERVATION → CONJECTURE                 │
    └─────────────────────────────────────────────────────────────────────────────────┘

                                    OBSERVATIONS
                                         │
            ┌────────────────────────────┼────────────────────────────┐
            │                            │                            │
            ▼                            ▼                            ▼
    ┌───────────────┐          ┌───────────────┐          ┌───────────────┐
    │   Δp > 0      │          │   Δp ≈ 0      │          │   Δp < 0      │
    │   CI excl. 0  │          │   CI incl. 0  │          │   CI excl. 0  │
    └───────┬───────┘          └───────┬───────┘          └───────┬───────┘
            │                          │                          │
            ▼                          ▼                          ▼
    ┌───────────────┐          ┌───────────────┐          ┌───────────────┐
    │   SUPPORTS    │          │  CONSISTENT   │          │  CONTRADICTS  │
    │               │          │      or       │          │               │
    │   Conj 3.1    │          │  INCONCLUSIVE │          │   Conj 3.1    │
    │   Conj 4.1    │          │               │          │   (if α↑)     │
    │   Conj 13.2   │          │               │          │               │
    └───────────────┘          └───────────────┘          └───────────────┘

                              SECONDARY EVIDENCE
                                     │
            ┌────────────────────────┼────────────────────────┐
            │                        │                        │
            ▼                        ▼                        ▼
    ┌───────────────┐        ┌───────────────┐        ┌───────────────┐
    │   Ψ < 0.01    │        │   O < 0.20    │        │  Step jumps   │
    │   (stable)    │        │   (no osc)    │        │  in d̄(t)     │
    └───────┬───────┘        └───────┬───────┘        └───────┬───────┘
            │                        │                        │
            ▼                        ▼                        ▼
    ┌───────────────┐        ┌───────────────┐        ┌───────────────┐
    │   SUPPORTS    │        │   SUPPORTS    │        │   SUPPORTS    │
    │   Thm 15.1    │        │   Thm 15.5    │        │   Conj 15.4   │
    │   Conj 6.1    │        │   (convergent)│        │   (nested     │
    │               │        │               │        │    basins)    │
    └───────────────┘        └───────────────┘        └───────────────┘
```

### 5.4 Conjecture Falsification Conditions

```
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │                    CONJECTURE FALSIFICATION CONDITIONS                          │
    └─────────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────┐
    │   Conj 3.1      │──── FALSIFIED IF ────▶ α_t monotonically INCREASING
    │   Supermartingale│                        with p < 0.05 (Mann-Kendall)
    └─────────────────┘

    ┌─────────────────┐
    │   Conj 4.1      │──── FALSIFIED IF ────▶ Alternative model (linear, step)
    │   Logistic Decay│                        fits R² > logistic_R² + 0.10
    └─────────────────┘

    ┌─────────────────┐
    │   Thm 13.2      │──── FALSIFIED IF ────▶ θ_t diverges OR
    │   Multi-Goal    │                        Ψ_T > 0.10 after T_max cycles
    │   Convergence   │
    └─────────────────┘

    ┌─────────────────┐
    │   Thm 15.1      │──── FALSIFIED IF ────▶ θ → ∞ (unbounded growth)
    │   Local Stability│
    └─────────────────┘

    ┌─────────────────┐
    │   Conj 15.4     │──── FALSIFIED IF ────▶ Basin structure clearly mismatches
    │   Basin Structure│                        prediction (e.g., Slice A has
    └─────────────────┘                         multiple basins, D has single)

    ┌─────────────────┐
    │   Conj 6.1      │──── FALSIFIED IF ────▶ α_t does NOT converge to 0
    │   A.S. Convergence│                       after T_max cycles (stationary
    └─────────────────┘                         at non-zero level)
```

---

## 6. Implementation Verification Checklist

### 6.1 Pre-Run Verification

| Check | Theory Reference | Implementation Location | Verification Method |
|-------|-----------------|------------------------|---------------------|
| Goal hashes defined | Def 12.10 | `config.goals.target_hashes` | Assert non-empty list |
| Budget cap set | §12.3 | `config.budget_max` | Assert == 40 |
| Depth threshold set | §12.4 | `config.d_min` | Assert == 3 |
| Policy params initialized | §13.2 | `policy.theta` | Assert vector exists |
| Seed deterministic | §17.1 | `config.seed` | Assert matches prereg |

### 6.2 Per-Cycle Verification

| Check | Theory Reference | Field | Verification Method |
|-------|-----------------|-------|---------------------|
| Candidates logged | §12.1 | `candidates.total` | Assert > 0 |
| Verified subset of candidates | §12.1 | `verified.hashes` | Assert ⊆ candidates |
| Abstention complement | §12.1 | `abstained.count` | Assert == candidates - verified |
| Density computed correctly | Def 12.4 | `verification_density` | Assert == verified/candidates |
| Goal-hit computed correctly | Def 12.1 | `goal_hit` | Assert == bool(V ∩ G) |
| H_t present | §17.1 | `H_t` | Assert 64-char hex |

### 6.3 Post-Run Verification

| Check | Theory Reference | Location | Verification Method |
|-------|-----------------|----------|---------------------|
| Summary metrics computed | §17.2 | `summary.*` | Assert all fields present |
| Diagnostics computed | §18 | `diagnostics.*` | Assert Ψ, O present |
| CI computed | §17.4 | `comparison.ci_*` | Assert bounds exist |
| Outcome determined | §19 | `comparison.outcome` | Assert in {POSITIVE, NULL, INVALID} |

---

## 7. Cross-Reference Index

### 7.1 Theory Document References

| Symbol | RFL_UPLIFT_THEORY.md Section | This Document Section |
|--------|------------------------------|----------------------|
| $\mathcal{C}_t$ | §12.1 | §2.1 |
| $\mathcal{V}_t$ | §12.1 | §2.1 |
| $\mathcal{G}$ | §12.2, §12.10 | §2.1, §2.4 |
| $\mathbf{1}_{\text{goal}}$ | Def 12.1 | §2.1 |
| $\delta_t$ | Def 12.4 | §2.2 |
| $d(v)$ | Def 12.7 | §2.3 |
| $\kappa_t$ | Def 12.12 | §2.4 |
| $\theta_t$ | §13.2 | §2.5 |
| $\Psi_T^{(w)}$ | Def 14.1 | §2.6 |
| $\mathcal{O}_T$ | Def 15.2 | §2.6 |
| $\hat{\Delta}_i$ | Def 17.1 | §2.1–2.4 |

### 7.2 Code References

| Implementation | File | Line Reference |
|----------------|------|----------------|
| Telemetry emission | `backend/metrics/first_organism_telemetry.py` | L85-143 |
| Manifest schema | `docs/evidence/experiment_manifest_schema.json` | Full file |
| Diagnostic functions | `RFL_UPLIFT_THEORY.md` | §18.2–18.6 |
| Estimator functions | `RFL_UPLIFT_THEORY.md` | §17.2–17.4 |

---

## 8. Revision History

| Date | Revision | Author |
|------|----------|--------|
| 2025-12-06 | Initial creation with master consistency matrix | CLAUDE M |
| 2025-12-06 | Added conjecture dependency diagrams | CLAUDE M |
| 2025-12-06 | Added telemetry schema and manifest binding | CLAUDE M |
| 2025-12-06 | Added implementation verification checklist | CLAUDE M |

---

**PHASE II — NOT RUN IN PHASE I**

**ALL MAPPINGS ARE DESIGN SPECIFICATIONS PENDING IMPLEMENTATION VALIDATION**
