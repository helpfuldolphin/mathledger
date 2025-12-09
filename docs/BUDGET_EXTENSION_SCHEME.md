# Budget Extension Scheme: v2 and v3

> **STATUS: PHASE II — FORWARD-LOOKING DESIGN**
>
> This document proposes a forward-compatible extension scheme for the verifier
> budget system. It defines versioned schemas that maintain backward compatibility
> while enabling new capabilities in future phases.

---

## 1. Version Evolution Overview

```
v1.0 (Current)       v2.0 (Phase IIb)      v3.0 (Phase III)
─────────────────    ─────────────────     ─────────────────
• taut_timeout_s     • taut_timeout_s      • taut_timeout_s
• cycle_budget_s     • cycle_budget_s      • cycle_budget_s
• max_candidates     • max_candidates      • max_candidates
                     • lean_timeout_s      • lean_timeout_s
                     • lean_memory_mb      • lean_memory_mb
                     • max_lean_calls      • max_lean_calls
                                           • gpu_timeout_s
                                           • parallel_workers
                                           • adaptive_budget
```

---

## 2. Schema Version 1.0 (Current)

### 2.1 Schema Definition

```yaml
# config/verifier_budget_phase2.yaml
# Schema Version: 1.0

schema_version: "1.0"
phase: "II"

defaults:
  taut_timeout_s: 0.10      # Required: Per-statement timeout
  cycle_budget_s: 5.0       # Required: Cycle wall-clock budget
  max_candidates_per_cycle: 100  # Required: Candidate cap

slices:
  slice_uplift_goal:
    # Slice-specific overrides (all optional)
    cycle_budget_s: 5.0

  slice_uplift_sparse:
    cycle_budget_s: 6.0
    max_candidates_per_cycle: 120
```

### 2.2 Data Class (v1.0)

```python
@dataclass(frozen=True, slots=True)
class VerifierBudget:
    """Schema v1.0 — Phase II Budget Parameters."""
    cycle_budget_s: float
    taut_timeout_s: float
    max_candidates_per_cycle: int

    # Schema version for forward compatibility
    schema_version: str = "1.0"
```

### 2.3 Compatibility Guarantees

- **v1.0 configs will always load in v2.0+ loaders**
- **v1.0 data classes are a strict subset of v2.0+**
- **v1.0 manifests remain valid indefinitely**

---

## 3. Schema Version 2.0 (Phase IIb — Lean Integration)

### 3.1 New Capabilities

Version 2.0 adds Lean verification budget parameters:

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `lean_timeout_s` | float | Per-statement Lean timeout | 30.0 |
| `lean_memory_mb` | int | Lean process memory limit | 2048 |
| `max_lean_calls_per_cycle` | int | Lean call quota per cycle | 10 |
| `lean_enabled` | bool | Master enable for Lean | false |
| `tier2_fallback` | str | Fallback behavior on Lean failure | "tier1" |

### 3.2 Schema Definition (v2.0)

```yaml
# config/verifier_budget_phase2b.yaml
# Schema Version: 2.0

schema_version: "2.0"
phase: "IIb"

defaults:
  # v1.0 parameters (required)
  taut_timeout_s: 0.10
  cycle_budget_s: 5.0
  max_candidates_per_cycle: 100

  # v2.0 parameters (optional, have defaults)
  lean_enabled: false
  lean_timeout_s: 30.0
  lean_memory_mb: 2048
  max_lean_calls_per_cycle: 10
  tier2_fallback: "tier1"  # "tier1" | "abstain" | "fail"

slices:
  slice_uplift_goal_lean:
    lean_enabled: true
    lean_timeout_s: 20.0
    max_lean_calls_per_cycle: 5

  slice_uplift_sparse:
    # No Lean, uses v1.0 parameters only
    cycle_budget_s: 6.0
```

### 3.3 Data Class (v2.0)

```python
@dataclass(frozen=True, slots=True)
class VerifierBudgetV2:
    """Schema v2.0 — Phase IIb Budget Parameters with Lean Support."""

    # v1.0 parameters (always present)
    cycle_budget_s: float
    taut_timeout_s: float
    max_candidates_per_cycle: int

    # v2.0 parameters (new in this version)
    lean_enabled: bool = False
    lean_timeout_s: float = 30.0
    lean_memory_mb: int = 2048
    max_lean_calls_per_cycle: int = 10
    tier2_fallback: str = "tier1"

    schema_version: str = "2.0"

    @classmethod
    def from_v1(cls, v1: VerifierBudget) -> "VerifierBudgetV2":
        """Upgrade v1.0 budget to v2.0 with defaults."""
        return cls(
            cycle_budget_s=v1.cycle_budget_s,
            taut_timeout_s=v1.taut_timeout_s,
            max_candidates_per_cycle=v1.max_candidates_per_cycle,
            # v2.0 fields use defaults
        )
```

### 3.4 Loader Enhancement (v2.0)

```python
def load_budget_for_slice_v2(
    slice_name: str,
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> VerifierBudgetV2:
    """
    Load budget with v2.0 schema support.

    - v1.0 configs are upgraded automatically
    - v2.0 configs are loaded directly
    """
    config = _load_yaml(path)

    schema_version = config.get("schema_version", "1.0")

    if schema_version == "1.0":
        # Load as v1.0 and upgrade
        v1_budget = load_budget_for_slice(slice_name, path)
        return VerifierBudgetV2.from_v1(v1_budget)

    elif schema_version.startswith("2."):
        # Load v2.0 directly
        return _load_v2_budget(config, slice_name)

    else:
        raise ValueError(f"Unknown schema version: {schema_version}")
```

---

## 4. Schema Version 3.0 (Phase III — Parallel & Adaptive)

### 4.1 New Capabilities

Version 3.0 adds parallel verification and adaptive budgeting:

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `parallel_workers` | int | Number of parallel verification workers | 1 |
| `gpu_enabled` | bool | Enable GPU-accelerated verification | false |
| `gpu_timeout_s` | float | GPU kernel timeout | 10.0 |
| `adaptive_budget` | AdaptiveBudgetConfig | Adaptive budget settings | None |
| `resource_priority` | str | Resource allocation priority | "normal" |

### 4.2 Schema Definition (v3.0)

```yaml
# config/verifier_budget_phase3.yaml
# Schema Version: 3.0

schema_version: "3.0"
phase: "III"

defaults:
  # v1.0 parameters
  taut_timeout_s: 0.10
  cycle_budget_s: 5.0
  max_candidates_per_cycle: 100

  # v2.0 parameters
  lean_enabled: false
  lean_timeout_s: 30.0
  lean_memory_mb: 2048
  max_lean_calls_per_cycle: 10
  tier2_fallback: "tier1"

  # v3.0 parameters
  parallel_workers: 1
  gpu_enabled: false
  gpu_timeout_s: 10.0
  resource_priority: "normal"  # "low" | "normal" | "high"

  # v3.0 adaptive budget (optional block)
  adaptive_budget: null  # or AdaptiveBudgetConfig

slices:
  slice_parallel_fast:
    parallel_workers: 4
    cycle_budget_s: 2.0  # Reduced due to parallelism

  slice_gpu_accelerated:
    gpu_enabled: true
    gpu_timeout_s: 5.0
    lean_enabled: false  # GPU replaces Lean

  slice_adaptive_exploration:
    adaptive_budget:
      enabled: true
      min_cycle_budget_s: 1.0
      max_cycle_budget_s: 30.0
      adjustment_factor: 0.1
      target_utilization: 0.8
```

### 4.3 Adaptive Budget Configuration

```yaml
# Nested configuration for adaptive budgeting
adaptive_budget:
  enabled: bool           # Master enable
  min_cycle_budget_s: float  # Floor for adaptation
  max_cycle_budget_s: float  # Ceiling for adaptation
  adjustment_factor: float   # Per-cycle adjustment (0.0-1.0)
  target_utilization: float  # Target budget utilization (0.0-1.0)
  warmup_cycles: int         # Cycles before adaptation starts
  history_window: int        # Cycles to consider for adaptation
```

### 4.4 Data Class (v3.0)

```python
@dataclass(frozen=True)
class AdaptiveBudgetConfig:
    """Configuration for adaptive budget adjustment."""
    enabled: bool = False
    min_cycle_budget_s: float = 1.0
    max_cycle_budget_s: float = 60.0
    adjustment_factor: float = 0.1
    target_utilization: float = 0.8
    warmup_cycles: int = 10
    history_window: int = 20


@dataclass(frozen=True, slots=True)
class VerifierBudgetV3:
    """Schema v3.0 — Phase III Budget Parameters with Parallelism."""

    # v1.0 parameters
    cycle_budget_s: float
    taut_timeout_s: float
    max_candidates_per_cycle: int

    # v2.0 parameters
    lean_enabled: bool = False
    lean_timeout_s: float = 30.0
    lean_memory_mb: int = 2048
    max_lean_calls_per_cycle: int = 10
    tier2_fallback: str = "tier1"

    # v3.0 parameters
    parallel_workers: int = 1
    gpu_enabled: bool = False
    gpu_timeout_s: float = 10.0
    resource_priority: str = "normal"
    adaptive_budget: Optional[AdaptiveBudgetConfig] = None

    schema_version: str = "3.0"

    @classmethod
    def from_v2(cls, v2: VerifierBudgetV2) -> "VerifierBudgetV3":
        """Upgrade v2.0 budget to v3.0 with defaults."""
        return cls(
            # v1.0 fields
            cycle_budget_s=v2.cycle_budget_s,
            taut_timeout_s=v2.taut_timeout_s,
            max_candidates_per_cycle=v2.max_candidates_per_cycle,
            # v2.0 fields
            lean_enabled=v2.lean_enabled,
            lean_timeout_s=v2.lean_timeout_s,
            lean_memory_mb=v2.lean_memory_mb,
            max_lean_calls_per_cycle=v2.max_lean_calls_per_cycle,
            tier2_fallback=v2.tier2_fallback,
            # v3.0 fields use defaults
        )

    @classmethod
    def from_v1(cls, v1: VerifierBudget) -> "VerifierBudgetV3":
        """Upgrade v1.0 budget to v3.0 with defaults."""
        v2 = VerifierBudgetV2.from_v1(v1)
        return cls.from_v2(v2)
```

---

## 5. Migration Strategy

### 5.1 Backward Compatibility Rules

1. **New fields always have defaults** — Old configs remain valid
2. **Old fields are never removed** — Only deprecated
3. **Schema version is always present** — Default to "1.0" if missing
4. **Loaders support all previous versions** — Automatic upgrade

### 5.2 Upgrade Path

```
┌─────────────────────────────────────────────────────────────┐
│                    Config File Loading                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ Read schema_    │
                    │ version field   │
                    └────────┬────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
           ▼                 ▼                 ▼
     ┌───────────┐     ┌───────────┐     ┌───────────┐
     │ v1.0      │     │ v2.0      │     │ v3.0      │
     │ Parser    │     │ Parser    │     │ Parser    │
     └─────┬─────┘     └─────┬─────┘     └─────┬─────┘
           │                 │                 │
           ▼                 ▼                 │
     ┌───────────┐     ┌───────────┐           │
     │ Upgrade   │────▶│ Upgrade   │───────────┤
     │ to v2.0   │     │ to v3.0   │           │
     └───────────┘     └───────────┘           │
                                               │
                              ┌────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ VerifierBudget  │
                    │ V3 (current)    │
                    └─────────────────┘
```

### 5.3 Deprecation Policy

| Version | Status | Support Until |
|---------|--------|---------------|
| v1.0 | Active (Phase II) | Phase IV |
| v2.0 | Planned (Phase IIb) | Phase V |
| v3.0 | Planned (Phase III) | Indefinite |

**Deprecation warnings:**
```python
import warnings

def load_budget_v1(config, slice_name):
    warnings.warn(
        "v1.0 budget schema is deprecated. "
        "Consider upgrading to v2.0 for Lean support.",
        DeprecationWarning,
        stacklevel=2,
    )
    # ... load logic
```

---

## 6. API Evolution

### 6.1 Unified Loader Interface

```python
# Always use the latest loader, which handles all versions
from backend.verification.budget_loader import load_budget_for_slice

# Returns VerifierBudget (latest version)
budget = load_budget_for_slice("slice_uplift_goal")

# Version-specific access if needed
if budget.schema_version.startswith("3."):
    # Use v3.0 features
    if budget.adaptive_budget and budget.adaptive_budget.enabled:
        configure_adaptive_runner(budget)
elif budget.schema_version.startswith("2."):
    # Use v2.0 features
    if budget.lean_enabled:
        configure_lean_runner(budget)
else:
    # v1.0 - basic features only
    configure_basic_runner(budget)
```

### 6.2 Type Aliases for Clarity

```python
# Type aliases for documentation
VerifierBudget = VerifierBudgetV3  # Current version
VerifierBudgetLatest = VerifierBudgetV3

# Explicit version types
VerifierBudgetV1 = VerifierBudget  # Original
VerifierBudgetV2 = VerifierBudgetV2  # Lean support
VerifierBudgetV3 = VerifierBudgetV3  # Parallel + adaptive
```

### 6.3 Feature Detection

```python
def has_lean_support(budget: VerifierBudget) -> bool:
    """Check if budget supports Lean verification."""
    return hasattr(budget, 'lean_enabled') and budget.schema_version >= "2.0"


def has_parallel_support(budget: VerifierBudget) -> bool:
    """Check if budget supports parallel verification."""
    return hasattr(budget, 'parallel_workers') and budget.schema_version >= "3.0"


def has_adaptive_support(budget: VerifierBudget) -> bool:
    """Check if budget supports adaptive budgeting."""
    return (
        hasattr(budget, 'adaptive_budget')
        and budget.adaptive_budget is not None
        and budget.adaptive_budget.enabled
    )
```

---

## 7. Manifest Compatibility

### 7.1 Manifest Schema Evolution

```json
{
  "manifest_version": "2.0",
  "budget": {
    "schema_version": "1.0",
    "cycle_budget_s": 5.0,
    "taut_timeout_s": 0.10,
    "max_candidates_per_cycle": 100
  }
}
```

Becomes in v2.0:

```json
{
  "manifest_version": "3.0",
  "budget": {
    "schema_version": "2.0",
    "cycle_budget_s": 5.0,
    "taut_timeout_s": 0.10,
    "max_candidates_per_cycle": 100,
    "lean_enabled": true,
    "lean_timeout_s": 30.0,
    "lean_memory_mb": 2048,
    "max_lean_calls_per_cycle": 10
  }
}
```

### 7.2 Cross-Version Comparison

Manifests from different schema versions can be compared:

```python
def compare_manifests(m1: Manifest, m2: Manifest) -> ComparisonResult:
    """Compare manifests potentially from different schema versions."""
    # Normalize both to v3.0 for comparison
    b1 = normalize_to_v3(m1.budget)
    b2 = normalize_to_v3(m2.budget)

    # Compare only v1.0 fields if either is v1.0
    if m1.budget.schema_version == "1.0" or m2.budget.schema_version == "1.0":
        return compare_v1_fields(b1, b2)

    # Full comparison for matching versions
    return full_comparison(b1, b2)
```

---

## 8. Testing Strategy

### 8.1 Version Compatibility Tests

```python
class TestVersionCompatibility:
    """Tests for schema version compatibility."""

    def test_v1_loads_in_v2_loader(self):
        """v1.0 config loads correctly in v2.0 loader."""
        v1_config = create_v1_config()
        budget = load_budget_v2(v1_config)
        assert budget.schema_version == "2.0"
        assert budget.lean_enabled == False  # Default

    def test_v1_loads_in_v3_loader(self):
        """v1.0 config loads correctly in v3.0 loader."""
        v1_config = create_v1_config()
        budget = load_budget_v3(v1_config)
        assert budget.schema_version == "3.0"
        assert budget.parallel_workers == 1  # Default

    def test_upgrade_chain_v1_to_v3(self):
        """Full upgrade chain from v1.0 to v3.0."""
        v1 = VerifierBudget(cycle_budget_s=5.0, taut_timeout_s=0.1, max_candidates_per_cycle=100)
        v2 = VerifierBudgetV2.from_v1(v1)
        v3 = VerifierBudgetV3.from_v2(v2)

        # Original fields preserved
        assert v3.cycle_budget_s == 5.0
        assert v3.taut_timeout_s == 0.1
        assert v3.max_candidates_per_cycle == 100

        # New fields have defaults
        assert v3.lean_enabled == False
        assert v3.parallel_workers == 1
```

### 8.2 Regression Tests

```python
class TestNoRegression:
    """Tests ensuring upgrades don't change v1.0 behavior."""

    @pytest.mark.parametrize("slice_name", VALID_SLICES)
    def test_v1_behavior_preserved(self, slice_name):
        """v1.0 behavior unchanged after upgrade."""
        v1_budget = load_budget_v1(slice_name)
        v3_budget = load_budget_v3(slice_name)

        # Run identical experiments
        result_v1 = run_with_v1_budget(v1_budget, seed=42)
        result_v3 = run_with_v3_budget(v3_budget, seed=42)

        # Outcomes should be identical
        assert result_v1.outcomes == result_v3.outcomes
```

---

## 9. Implementation Timeline

| Phase | Schema Version | Key Features | Status |
|-------|----------------|--------------|--------|
| Phase II | v1.0 | Basic budget | **Active** |
| Phase IIb | v2.0 | Lean integration | Planned |
| Phase III | v3.0 | Parallel + adaptive | Future |
| Phase IV | v3.1 | GPU acceleration | Future |

---

## 10. References

- `docs/VERIFIER_BUDGET_THEORY.md` — Theoretical foundation
- `docs/BUDGET_STRESS_TEST_PLAN.md` — Testing strategy
- `backend/lean_control_sandbox_plan.md` §16 — Lean integration notes
- `config/verifier_budget_phase2.yaml` — Current v1.0 schema

---

*End of Budget Extension Scheme document.*
