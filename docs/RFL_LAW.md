# RFL Law: Metabolic Reflexivity Contract

**Version**: 1.0.0
**Date**: 2025-11-25
**Status**: Canonical Specification

---

## Abstract

The **RFL Law** defines the deterministic, auditable transformation from a sealed attestation payload (H_t) to an RFL ledger update. This document formalizes the metabolic reflexivity contract: given a specific H_t with a known abstention profile, the exact RFL ledger update is uniquely determined.

---

## 1. Definitions

### 1.1 Core Symbols

| Symbol | Type | Definition |
|--------|------|------------|
| **H_t** | 64-char hex | Composite attestation root = SHA256(R_t \|\| U_t) |
| **R_t** | 64-char hex | Reasoning Merkle root over proof artifacts |
| **U_t** | 64-char hex | UI Merkle root over human interaction artifacts |
| **α_rate** | float [0,1] | Abstention rate = abstentions / total_attempts |
| **α_mass** | float ≥ 0 | Abstention mass = raw abstention count |
| **τ** | float [0,1] | Abstention tolerance threshold (default 0.10) |
| **Δα** | float | Abstention mass delta = α_mass - (τ × attempt_mass) |
| **∇_sym** | float | Symbolic descent = -(α_rate - τ) |
| **step_id** | 64-char hex | Deterministic step identifier |

### 1.2 Input Context

The **AttestedRunContext** is the sealed input from derivation:

```python
@dataclass
class AttestedRunContext:
    slice_id: str               # Curriculum slice identifier
    statement_hash: str         # Statement being attested
    proof_status: str           # "success" | "failure" | "abstain"
    block_id: int               # Ledger block number
    composite_root: str         # H_t (64-char hex)
    reasoning_root: str         # R_t (64-char hex)
    ui_root: str                # U_t (64-char hex)
    abstention_metrics: {
        "rate": float,          # α_rate
        "mass": float           # α_mass
    }
    policy_id: Optional[str]    # Policy identifier
    metadata: Dict[str, Any]    # Additional context
```

### 1.3 Output Ledger Entry

The **RunLedgerEntry** is the deterministic output:

```python
@dataclass
class RunLedgerEntry:
    run_id: str                 # = step_id (deterministic)
    slice_name: str             # Resolved curriculum slice
    status: str                 # "attestation"
    coverage_rate: float        # From metadata
    novelty_rate: float         # From metadata
    throughput: float           # From metadata
    success_rate: float         # From metadata
    abstention_fraction: float  # = α_rate
    policy_reward: float        # Computed reward signal
    symbolic_descent: float     # = ∇_sym
    budget_spent: int           # = attempt_mass
    derive_steps: int           # From slice config
    max_breadth: int            # From slice config
    max_total: int              # From slice config
    abstention_breakdown: Dict  # Category breakdown
```

---

## 2. The RFL Law

### 2.1 Formal Statement

**RFL Law**: Given a valid AttestedRunContext with composite root H_t and abstention profile (α_rate, α_mass), the RFL transformation produces a unique RunLedgerEntry with deterministic fields computed as follows:

#### 2.1.1 Step ID Computation

```
step_id = SHA256(experiment_id | slice_name | policy_id | H_t)
```

Where:
- `experiment_id`: RFL experiment identifier (from config)
- `slice_name`: Resolved curriculum slice name
- `policy_id`: Policy identifier or "default"
- `H_t`: Composite attestation root
- `|`: String concatenation with pipe separator

**Determinism Guarantee**: Same inputs → identical step_id across all invocations.

#### 2.1.2 Abstention Mass Delta

```
Δα = α_mass - (τ × attempt_mass)
```

Where:
- `α_mass`: Abstention mass from context
- `τ`: Abstention tolerance (default 0.10)
- `attempt_mass`: Total attempts from metadata (default max(α_mass, 1.0))

**Policy Update Trigger**: `policy_update_applied = (|Δα| > 1e-9) OR (|α_rate - τ| > 1e-9)`

#### 2.1.3 Symbolic Descent Gradient

```
∇_sym = -(α_rate - τ)
```

**Interpretation**:
- ∇_sym > 0: Abstention below tolerance → positive descent (improvement)
- ∇_sym = 0: Abstention at tolerance → neutral
- ∇_sym < 0: Abstention above tolerance → negative descent (regression)

#### 2.1.4 Policy Reward Signal

```
reward = max(0.0, 1.0 - max(α_rate, 0.0))
```

**Range**: [0.0, 1.0]

---

## 3. Determinism Contract

### 3.1 Byte-for-Byte Reproducibility

Given identical:
1. RFLConfig (experiment_id, curriculum, abstention_tolerance)
2. AttestedRunContext (H_t, R_t, U_t, abstention_metrics)

The RFL transformation MUST produce:
- Identical `step_id` (64-char hex)
- Identical `symbolic_descent` (float)
- Identical `abstention_mass_delta` (float)
- Identical `policy_reward` (float)
- Identical `ledger_entry` fields

### 3.2 Independence Properties

The RFL transformation is:
- **Stateless**: No hidden state affects output
- **Pure**: Same inputs → same outputs
- **Time-Independent**: Timestamps derived from attestation metadata, not wall-clock

### 3.3 Auditability

Every RFL ledger update records:
1. **Source Root**: H_t from which the update was derived
2. **Step ID**: Deterministic identifier linking to inputs
3. **Abstention Breakdown**: Per-category abstention counts
4. **Policy Context**: Full slice configuration used

---

## 4. Verification Protocol

### 4.1 Given H_t, Verify Ledger Update

```python
def verify_rfl_update(
    attestation: AttestedRunContext,
    expected_entry: RunLedgerEntry,
    config: RFLConfig
) -> bool:
    """
    Verify that a ledger entry was correctly produced from an attestation.

    Returns True if the entry matches the deterministic computation.
    """
    # 1. Recompute step_id
    slice_cfg = resolve_slice(config.curriculum, attestation.slice_id)
    policy_id = attestation.policy_id or "default"
    step_material = f"{config.experiment_id}|{slice_cfg.name}|{policy_id}|{attestation.composite_root}"
    expected_step_id = hashlib.sha256(step_material.encode()).hexdigest()

    if expected_entry.run_id != expected_step_id:
        return False

    # 2. Verify abstention metrics
    attempt_mass = attestation.metadata.get("attempt_mass", max(attestation.abstention_mass, 1.0))
    expected_delta = attestation.abstention_mass - (config.abstention_tolerance * attempt_mass)
    expected_descent = -(attestation.abstention_rate - config.abstention_tolerance)
    expected_reward = max(0.0, 1.0 - max(attestation.abstention_rate, 0.0))

    if abs(expected_entry.symbolic_descent - expected_descent) > 1e-9:
        return False
    if abs(expected_entry.policy_reward - expected_reward) > 1e-9:
        return False

    # 3. Verify source traceability
    # (The entry must be traceable to H_t via step_id)

    return True
```

### 4.2 Test Protocol

For each test case:
1. Construct AttestedRunContext with specific H_t and abstention profile
2. Invoke `RFLRunner.run_with_attestation()`
3. Assert ledger entry matches expected deterministic values
4. Repeat N times to verify stability

---

## 5. Examples

### 5.1 High Abstention (α_rate = 0.35)

**Input**:
```json
{
  "H_t": "6a006e789be39105...",
  "α_rate": 0.35,
  "α_mass": 7.0,
  "attempt_mass": 20.0,
  "τ": 0.10
}
```

**Computed Values**:
```
Δα = 7.0 - (0.10 × 20.0) = 7.0 - 2.0 = 5.0
∇_sym = -(0.35 - 0.10) = -0.25
reward = max(0, 1.0 - 0.35) = 0.65
policy_update_applied = True (|Δα| > 0)
```

**Ledger Entry**:
```json
{
  "run_id": "<sha256(exp|slice|policy|H_t)>",
  "status": "attestation",
  "abstention_fraction": 0.35,
  "symbolic_descent": -0.25,
  "policy_reward": 0.65
}
```

### 5.2 Zero Abstention (α_rate = 0.0)

**Input**:
```json
{
  "H_t": "abc123...",
  "α_rate": 0.0,
  "α_mass": 0.0,
  "attempt_mass": 20.0,
  "τ": 0.10
}
```

**Computed Values**:
```
Δα = 0.0 - (0.10 × 20.0) = -2.0
∇_sym = -(0.0 - 0.10) = 0.10
reward = max(0, 1.0 - 0.0) = 1.0
policy_update_applied = True (|Δα| > 0)
```

### 5.3 At Tolerance Boundary (α_rate = 0.10)

**Input**:
```json
{
  "H_t": "def456...",
  "α_rate": 0.10,
  "α_mass": 2.0,
  "attempt_mass": 20.0,
  "τ": 0.10
}
```

**Computed Values**:
```
Δα = 2.0 - (0.10 × 20.0) = 0.0
∇_sym = -(0.10 - 0.10) = 0.0
reward = max(0, 1.0 - 0.10) = 0.90
policy_update_applied = False (|Δα| = 0 AND |α_rate - τ| = 0)
```

---

## 6. Consistency with Whitepaper

### 6.1 Whitepaper RFL Principles

From `docs/whitepaper.md`:

> "Every statement is derived from axioms, every proof is verified in Lean, every block is auditable."

The RFL Law extends this principle to the metabolism layer:
- **Every H_t** produces a deterministic ledger update
- **Every abstention** is quantified in the symbolic descent
- **Every policy update** is traceable to its source root

### 6.2 Ledger Semantics Alignment

From whitepaper Section 5 (Ledger Semantics):

> "MathLedger is a **monotone, auditable ledger**... Determinism: same slice config → identical hashes, proofs, blocks."

The RFL Law maintains this guarantee at the metabolism layer:
- Same H_t + config → identical RFL ledger entry
- Policy updates are append-only (monotone)
- All updates are auditable via step_id → H_t tracing

---

## 7. Implementation Notes

### 7.1 Current Implementation

The RFL Law is implemented in:
- `rfl/runner.py`: `RFLRunner.run_with_attestation()`
- `substrate/bridge/context.py`: `AttestedRunContext`
- `attestation/dual_root.py`: H_t computation

### 7.2 Key Code Paths

```python
# Step ID computation (runner.py:437-440)
step_material = (
    f"{self.config.experiment_id}|{slice_cfg.name}|{policy_id}|{attestation.composite_root}"
)
step_id = hashlib.sha256(step_material.encode("utf-8")).hexdigest()

# Symbolic descent (runner.py:453)
symbolic_descent = -abstention_rate_delta
# where abstention_rate_delta = attestation.abstention_rate - self.config.abstention_tolerance

# Policy reward (runner.py:452)
reward = max(0.0, 1.0 - max(attestation.abstention_rate, 0.0))
```

---

## 8. Conclusion

The RFL Law establishes a **formal contract** between the attestation layer (H_t production) and the metabolism layer (RFL policy updates). This contract guarantees:

1. **Determinism**: Given H_t and abstention profile, the ledger update is uniquely determined
2. **Auditability**: Every update traces back to its source attestation root
3. **Consistency**: The transformation aligns with whitepaper ledger semantics

The symbolic descent gradient (∇_sym) provides a **continuous feedback signal** for policy refinement, enabling the system to learn from abstention patterns and improve coverage over time.

**This is the law by which H_t metabolizes into policy.**
