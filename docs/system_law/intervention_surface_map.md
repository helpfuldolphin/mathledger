# Intervention Surface Map

**Document Version:** 0.1.0
**Status:** Operator Reference
**Parent:** USLA v0.1

---

## 1. Overview

The Intervention Surface Map identifies the minimal number of system surfaces where human or AI agents can safely intervene without destabilizing the organism.

Each surface specifies:
- **Modifiable parameters**: What can be changed
- **Non-modifiable invariants**: What must not be changed
- **Response curve**: Expected system behavior
- **Red-flag outcomes**: Conditions requiring immediate rollback

---

## 2. Intervention Surfaces

### Surface 1: Threshold Surface (τ₀)

| Aspect | Specification |
|--------|---------------|
| **Modifiable** | τ₀ ∈ [0.1, 0.4] |
| **Non-Modifiable** | τ₀ < 0.1 destabilizes; τ₀ > 0.4 over-blocks |
| **Response Curve** | ∂β/∂τ₀ ≈ -J (Jacobian); linear in stable region |
| **Red-Flag** | J > 15 after adjustment |

**Intervention Protocol:**
1. Compute current Jacobian J
2. If J < 5: safe to adjust ±0.05
3. If 5 ≤ J < 10: adjust ±0.02 max
4. If J ≥ 10: do not adjust; system too sensitive

---

### Surface 2: Sensitivity Surface (α_D, α_B, α_S)

| Aspect | Specification |
|--------|---------------|
| **Modifiable** | ±50% of default values |
| **Non-Modifiable** | Signs must preserve semantics |
| **Response Curve** | Near-linear in stable region |
| **Red-Flag** | Sign flip; any α > 0.2 |

**Default Values:**
- α_D = 0.02 (depth velocity sensitivity)
- α_B = 0.01 (branch factor sensitivity)
- α_S = 0.1 (shear sensitivity)

**Semantic Constraints:**
- α_D > 0: Higher depth velocity → higher threshold (more lenient)
- α_B > 0: Higher branch factor → higher threshold
- α_S > 0: Higher shear → lower threshold (stricter)

---

### Surface 3: Convergence Modifier Surface (γ)

| Aspect | Specification |
|--------|---------------|
| **Modifiable** | γ_oscillating ∈ [1.0, 1.3], γ_diverging ∈ [1.1, 1.5] |
| **Non-Modifiable** | γ_converging = 1.0 (reference point) |
| **Response Curve** | Step response at C transition |
| **Red-Flag** | γ_diverging < 1.0 (inverts safety semantics) |

**Interpretation:**
- γ > 1.0: Threshold relaxed (more lenient when system stressed)
- γ = 1.0: Nominal threshold
- γ < 1.0: FORBIDDEN (would tighten when system needs relief)

---

### Surface 4: RSI Smoothing Surface (α_ρ)

| Aspect | Specification |
|--------|---------------|
| **Modifiable** | α_ρ ∈ [0.8, 0.95] |
| **Non-Modifiable** | α_ρ < 0.7 loses temporal memory |
| **Response Curve** | Smoothing bandwidth = 1/(1-α_ρ) cycles |
| **Red-Flag** | Oscillation onset when α_ρ < 0.75 |

**Effect:**
- α_ρ = 0.9: Effective window ≈ 10 cycles
- α_ρ = 0.95: Effective window ≈ 20 cycles
- α_ρ = 0.8: Effective window ≈ 5 cycles (responsive but noisy)

---

### Surface 5: Exception Window Surface (N_max, cooldown)

| Aspect | Specification |
|--------|---------------|
| **Modifiable** | N_max ∈ [5, 20], cooldown ∈ [10, 50] |
| **Non-Modifiable** | N_max = 0 disables exception mechanism entirely |
| **Response Curve** | Utilization ratio proportional to 1/N_max |
| **Red-Flag** | CDI-007 (exhaustion) triggers |

**Trade-offs:**
- Higher N_max: More override capacity, but risk of chronic bypass
- Higher cooldown: Less frequent activation, but longer recovery

---

### Surface 6: Safe Region Bounds

| Aspect | Specification |
|--------|---------------|
| **Modifiable** | ±20% relaxation of each bound |
| **Non-Modifiable** | H_min > 0 required; all bounds must be positive |
| **Response Curve** | Proportional Ω volume change |
| **Red-Flag** | Bound violation cascade (one violation triggers others) |

**Default Bounds:**
| Bound | Default | Min | Max |
|-------|---------|-----|-----|
| H_min | 0.3 | 0.24 | 0.36 |
| Ḋ_max | 2.0 | 1.6 | 2.4 |
| B_max | 8.0 | 6.4 | 9.6 |
| S_max | 0.4 | 0.32 | 0.48 |

---

### Surface 7: Invariant Tolerance Surface (ε)

| Aspect | Specification |
|--------|---------------|
| **Modifiable** | ε ∈ [0.5×, 2.0×] of default |
| **Non-Modifiable** | ε = 0 trivializes invariant |
| **Response Curve** | Violation frequency ∝ 1/ε |
| **Red-Flag** | False-positive storm (>50% cycles violating) |

**Per-Invariant Defaults:**
| Invariant | Default ε | Description |
|-----------|-----------|-------------|
| INV-001 | 0.05 | Shear increase per cycle |
| INV-002 | 1.0 | BF-depth gradient |
| INV-003 | 0.02 | Variance Lipschitz constant |
| INV-004 | 0.1 | Cut coherence minimum |
| INV-005 | 0.01 | Second-order stability |
| INV-006 | 0.1 | Block rate drift |
| INV-007 | 0.2 | Exception ratio maximum |
| INV-008 | 20 | Depth maximum |

---

### Surface 8: CDI Threshold Surface

| Aspect | Specification |
|--------|---------------|
| **Modifiable** | Per-defect trigger thresholds |
| **Non-Modifiable** | Defect classification (STRUCTURAL, etc.) |
| **Response Curve** | Detection sensitivity |
| **Red-Flag** | CDI-006/007 suppression (masks critical defects) |

**High-Risk CDIs (do not suppress):**
- CDI-006: Complexity Avoidance (CRITICAL)
- CDI-007: Exception Exhaustion (HIGH/CRITICAL)
- CDI-010: Fixed-Point Multiplicity (HIGH/CRITICAL)

---

### Surface 9: Curriculum Surface

| Aspect | Specification |
|--------|---------------|
| **Modifiable** | Slice duration, slice ordering |
| **Non-Modifiable** | Axiom content (mathematical truth) |
| **Response Curve** | Proof complexity distribution shift |
| **Red-Flag** | Complexity avoidance pattern (CDI-006) |

**Safe Interventions:**
- Extend slice duration when block_rate > 0.3
- Insert transition slices at semantic boundaries
- Pause curriculum on HIGH severity depth instability

---

### Surface 10: TGRS Weight Surface

| Aspect | Specification |
|--------|---------------|
| **Modifiable** | w₁...w₅ ∈ [0, 0.5] |
| **Non-Modifiable** | Σwᵢ = 1 (normalization) |
| **Response Curve** | Score distribution shift |
| **Red-Flag** | Any single weight > 0.5 (dominance) |

**Default Weights:**
| Weight | Component | Default |
|--------|-----------|---------|
| w₁ | H_score (HSS health) | 0.25 |
| w₂ | C_score (Convergence) | 0.25 |
| w₃ | S_score (Shear) | 0.15 |
| w₄ | B_score (Branch factor) | 0.15 |
| w₅ | P_score (Predicates) | 0.20 |

---

## 3. Intervention Decision Tree

```
INTERVENTION REQUEST:
│
├─ Is target a modifiable parameter?
│  ├─ NO → REJECT
│  └─ YES → Continue
│
├─ Is current Jacobian J < 10?
│  ├─ NO → REJECT (system too sensitive)
│  └─ YES → Continue
│
├─ Is system in safe region Ω?
│  ├─ NO → REJECT (stabilize first)
│  └─ YES → Continue
│
├─ Will change violate non-modifiable constraint?
│  ├─ YES → REJECT
│  └─ NO → Continue
│
├─ Is change magnitude within allowed range?
│  ├─ NO → REDUCE magnitude
│  └─ YES → Continue
│
├─ Simulate intervention (shadow mode):
│  ├─ Red-flag predicted → REJECT
│  └─ No red-flag → APPROVE with monitoring
│
└─ APPLY intervention with rollback capability
```

---

## 4. Intervention Risk Matrix

| Surface | Risk Level | Recovery Time | Monitoring Period |
|---------|------------|---------------|-------------------|
| Threshold τ₀ | MEDIUM | 5-10 cycles | 20 cycles |
| Sensitivity α | LOW | 2-5 cycles | 10 cycles |
| Convergence γ | LOW | 1-2 cycles | 5 cycles |
| RSI α_ρ | LOW | 10-20 cycles | 30 cycles |
| Exception window | MEDIUM | 10-20 cycles | 50 cycles |
| Safe region bounds | HIGH | 5-10 cycles | 30 cycles |
| Invariant ε | MEDIUM | Immediate | 20 cycles |
| CDI thresholds | HIGH | Immediate | 50 cycles |
| Curriculum | HIGH | 20-50 cycles | 100 cycles |
| TGRS weights | LOW | Immediate | 10 cycles |

---

## 5. Emergency Rollback Protocol

If any red-flag condition detected after intervention:

1. **Immediate**: Revert parameter to pre-intervention value
2. **Within 1 cycle**: Activate exception window if not active
3. **Within 5 cycles**: Verify system returns to safe region
4. **Within 10 cycles**: Re-evaluate TGRS
5. **Log**: Record intervention attempt and failure for analysis

