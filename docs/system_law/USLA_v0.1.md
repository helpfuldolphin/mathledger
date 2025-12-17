# USLA v0.1 — Unified System Law Abstraction

**Document Version:** 0.1.0
**Status:** Canonical System Law
**Classification:** Architecture Root

---

## 1. System Definition

The MathLedger governance-topology organism is a discrete-time controlled dynamical system **(Ω, X, U, F, G, Θ)** where:

| Symbol | Definition | Description |
|--------|------------|-------------|
| **Ω ⊂ ℝ⁶** | Safe control region | Polytope defining stable operation |
| **X ⊂ ℝⁿ** | State manifold | All system state variables |
| **U = {0, 1}** | Control action space | ALLOW (0), BLOCK (1) |
| **F: X × U × Θ → X** | State transition operator | Deterministic update law |
| **G: X → U** | Governance policy | Blocking decision function |
| **Θ ⊂ ℝᵐ** | Parameter manifold | Thresholds, sensitivities, tolerances |

---

## 2. System Laws

### Law 1: Governance Law
```
G(x) = 𝟙[H < τ(x; θ) ∧ ¬W(x)]
```
The governance policy blocks when HSS falls below adaptive threshold AND exception window is not active.

### Law 2: Threshold Law
```
τ(x; θ) = τ₀ · (1 + α_D · Ḋ) · (1 + α_B · (B - B₀)) · (1 - α_S · S) · γ(C)
```
Adaptive threshold responds to depth velocity, branch factor deviation, shear, and convergence class.

### Law 3: Stability Law
```
ρₜ₊₁ = αρₜ + (1-α)S(xₜ)
```
Rolling Stability Index is exponentially smoothed instantaneous stability.

### Law 4: Invariant Law
```
I(x) = ⋀ᵢ Iᵢ(x) ≥ εᵢ
```
System invariants must remain satisfied within tolerance.

### Law 5: Safe Region Law
```
x ∈ Ω ⟺ (H ≥ H_min) ∧ (|Ḋ| ≤ Ḋ_max) ∧ (B ≤ B_max) ∧ (S ≤ S_max) ∧ (C ≠ 2)
```
Safe region is intersection of five half-spaces plus convergence constraint.

### Law 6: Defect Law
```
D(x) = {d ∈ CDI : trigger_d(x) > threshold_d}
```
Active defects are those whose trigger predicates exceed thresholds.

### Law 7: Activation Law
```
HARD_OK(x) ⟺ (x ∈ Ω) ∧ (I(x)) ∧ (D(x) = ∅) ∧ (ρ ≥ ρ_min)
```
HARD mode activation requires safe region membership, invariant satisfaction, no defects, and minimum stability.

---

## 3. Canonical State Vector

**x ∈ ℝ¹⁵** defined as:

| Index | Symbol | Description | Domain |
|-------|--------|-------------|--------|
| 1 | H | HSS (health signal) | [0, 1] |
| 2 | D | Proof depth | ℤ⁺ |
| 3 | Ḋ | Depth velocity | ℝ |
| 4 | B | Branch factor | ℝ⁺ |
| 5 | S | Semantic shear | [0, 1] |
| 6 | C | Convergence class | {0, 1, 2} |
| 7 | ρ | Rolling Stability Index | [0, 1] |
| 8 | τ | Effective threshold | [0.1, 0.5] |
| 9 | J | Jacobian max sensitivity | ℝ⁺ |
| 10 | W | Exception window active | {0, 1} |
| 11 | β | Block rate (rolling) | [0, 1] |
| 12 | κ | Coupling strength | [0, 1] |
| 13 | ν | Variance velocity | ℝ |
| 14 | δ | CDI defect count | ℤ⁺ |
| 15 | Γ | TGRS (readiness score) | [0, 1] |

---

## 4. Parameter Manifold Θ

### Threshold Parameters
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| τ₀ | 0.2 | [0.1, 0.4] | Base HSS threshold |
| α_D | 0.02 | [0, 0.05] | Depth velocity sensitivity |
| α_B | 0.01 | [0, 0.03] | Branch factor sensitivity |
| α_S | 0.1 | [0, 0.2] | Shear sensitivity |
| B₀ | 2.0 | [1.5, 3.0] | Nominal branch factor |

### Convergence Modifiers
| Parameter | Default | Description |
|-----------|---------|-------------|
| γ_converging | 1.0 | Threshold modifier when CONVERGING |
| γ_oscillating | 1.1 | Threshold modifier when OSCILLATING |
| γ_diverging | 1.3 | Threshold modifier when DIVERGING |

### Stability Parameters
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| α_ρ | 0.9 | [0.8, 0.95] | RSI smoothing factor |
| ρ_min | 0.4 | [0.3, 0.5] | Minimum RSI for HARD mode |

### Safe Region Bounds
| Parameter | Default | Description |
|-----------|---------|-------------|
| H_min | 0.3 | HSS floor |
| Ḋ_max | 2.0 | Depth velocity ceiling |
| B_max | 8.0 | Branch factor ceiling |
| S_max | 0.4 | Shear ceiling |

---

## 5. Subsystem Hierarchy

All subsystems derive from USLA primitives:

```
USLA v0.1
├── F (State Transition Operator)
│   ├── Observation Layer
│   ├── Dynamics Classification
│   ├── Governance Computation
│   └── Stability Assessment
├── G (Governance Control Law)
│   ├── Adaptive Threshold τ(x)
│   ├── Exception Window W(x)
│   └── Block Decision
├── Ω (Safe Region)
│   ├── H-boundary (HSS floor)
│   ├── Ḋ-boundary (depth velocity)
│   ├── B-boundary (branch factor)
│   ├── S-boundary (shear)
│   └── C-constraint (convergence)
├── I (Invariants)
│   ├── INV-001: Shear Monotonicity
│   ├── INV-002: BF-Depth Gradient
│   ├── INV-003: HSS-Variance Lipschitz
│   ├── INV-004: Cut Coherence
│   ├── INV-005: Stability-of-Stability
│   ├── INV-006: Block Rate Stationarity
│   ├── INV-007: Exception Conservation
│   └── INV-008: Depth Boundedness
├── D (Defect Ontology)
│   ├── CDI-001 through CDI-010
│   └── Severity Classification
├── Γ (TGRS Readiness)
│   ├── H_score, C_score, S_score, B_score, P_score
│   └── Weighted Combination
├── ρ (Rolling Stability Index)
│   ├── Instantaneous Stability S(x)
│   └── Exponential Smoothing
└── J (Jacobian Sensitivity)
    ├── Partial Derivatives
    └── Stability Margin
```

---

## 6. Implementation Mapping

| USLA Component | Implementation Module |
|----------------|----------------------|
| F | `backend/topology/usla_simulator.py` |
| G | `backend/tda/governance.py` |
| Ω | `backend/topology/safe_region.py` |
| I | `backend/topology/invariant_monitor.py` |
| D | `backend/topology/cdi_detector.py` |
| Γ | `backend/topology/hard_mode_gate.py` |
| ρ | `backend/topology/stability_index.py` |
| J | `backend/topology/jacobian_monitor.py` |
| τ(x) | `backend/tda/governance.py::compute_adaptive_threshold` |

---

## 7. Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2025-12-09 | Initial USLA formalization from Phase VIII/IX |

