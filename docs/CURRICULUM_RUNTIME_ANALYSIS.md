# Curriculum Runtime & Slice Execution Analysis

**Manus-E Report**  
**Date**: 2025-12-06  
**Repository**: helpfuldolphin/mathledger  

---

## Executive Summary

I have completed a comprehensive examination of the MathLedger curriculum runtime, slice execution, and drift detection systems. The codebase demonstrates a well-structured approach to curriculum management with deterministic execution guarantees, but **curriculum fingerprinting and drift guards are not yet implemented**.

---

## 1. Curriculum Schema & Loader (V2)

### 1.1 Schema Architecture

**Location**: `backend/frontier/curriculum.py`

The curriculum system implements a **Version 2 schema** with the following structure:

```yaml
version: 2
systems:
  <system_slug>:
    description: <string>
    active: <slice_name>
    invariants:
      monotonic_axes: [atoms, depth_max]
      attestation_required: <bool>
      attestation_window_minutes: <int>
    slices:
      - name: <string>
        params: <dict>
        gates: <dict>
        completed_at: <iso8601_timestamp>  # optional
```

### 1.2 Core Data Structures

**CurriculumSlice** (`@dataclass`):
- `name`: Unique identifier
- `params`: Dictionary of slice parameters (atoms, depth_max, breadth_max, total_max, etc.)
- `gates`: SliceGates object containing 4 gate types
- `completed_at`: Optional ISO-8601 timestamp
- `metadata`: Optional additional fields

**SliceGates** (`@dataclass`):
- `coverage`: CoverageGateSpec (ci_lower_min, sample_min, require_attestation)
- `abstention`: AbstentionGateSpec (max_rate_pct, max_mass)
- `velocity`: VelocityGateSpec (min_pph, stability_cv_max, window_minutes)
- `caps`: CapsGateSpec (min_attempt_mass, min_runtime_minutes, backlog_max)

**CurriculumSystem** (`@dataclass`):
- `version`: Schema version (currently 2)
- `slug`: System identifier (e.g., "pl", "fol")
- `description`: Human-readable description
- `slices`: List of CurriculumSlice objects
- `active_index`: Index of currently active slice
- `invariants`: Dictionary of system-level constraints

### 1.3 Loader Implementation

**Function**: `load(system_slug: str, *, validate: bool = True) -> CurriculumSystem`

**Location**: `backend/frontier/curriculum.py:1112-1141`

**Process**:
1. Locates `config/curriculum.yaml` relative to module path
2. Loads YAML with fallback parser (supports environments without PyYAML)
3. Validates configuration if `validate=True`
4. Constructs `CurriculumSystem` from config

**Configuration Path**: `config/curriculum.yaml`

### 1.4 Validation System

**Function**: `validate_curriculum_config(config: Dict[str, Any]) -> List[str]`

**Validation Checks**:
- ✅ Version field presence and compatibility (must be 2)
- ✅ Systems dictionary structure
- ✅ Slice name uniqueness
- ✅ Required gate presence (coverage, abstention, velocity, caps)
- ✅ Monotonicity invariants across slices
- ✅ Active slice existence
- ✅ Parameter type validation

**Monotonicity Enforcement**:
```python
def _validate_monotonicity(slug: str, slices: List[Dict], axes: List[str]) -> List[str]
```
Ensures that specified axes (e.g., `atoms`, `depth_max`) are non-decreasing across slices.

**Current Monotonic Axes** (for "pl" system):
- `atoms`
- `depth_max`

---

## 2. Slice Execution Integration

### 2.1 RFL Runner Architecture

**Location**: `rfl/runner.py`

**Class**: `RFLRunner`

The RFL (Reflexive Formal Learning) runner orchestrates slice execution through a 40-run experiment suite.

**Key Components**:

1. **Configuration Management** (`rfl/config.py`):
   - `RFLConfig`: Top-level experiment configuration
   - `CurriculumSlice`: Slice-specific parameters
   - Curriculum ladder with run-to-slice mapping

2. **Slice Resolution**:
   ```python
   def resolve_slice(self, run_index: int) -> CurriculumSlice
   ```
   Maps 1-indexed run numbers to curriculum slices based on `start_run` and `end_run`.

3. **Execution Flow**:
   - Phase 1: Execute experiments with slice-specific parameters
   - Phase 2: Compute coverage metrics
   - Phase 3: Compute uplift metrics
   - Phase 4: Verify metabolism (RFL gates)
   - Phase 5: Export results

### 2.2 Slice Parameter Binding

**Location**: `rfl/runner.py:266-277` (resolve_slice method)

Each run is bound to a slice, which provides:
- `derive_steps`: Number of derivation steps
- `max_breadth`: Breadth limit for search
- `max_total`: Total proof budget
- `depth_max`: Maximum formula depth

### 2.3 Gate Evaluation

**Class**: `GateEvaluator` (`backend/frontier/curriculum.py:872`)

**Function**: `should_ratchet(metrics: Dict, system_cfg: CurriculumSystem) -> GateVerdict`

**Process**:
1. Normalize metrics from raw dictionary
2. Evaluate all 4 gates against active slice thresholds
3. Return `GateVerdict` with advance decision and audit log

**Gate Types**:
1. **Coverage Gate**: Bootstrap CI lower bound ≥ threshold
2. **Abstention Gate**: Abstention rate and mass within limits
3. **Velocity Gate**: Proof throughput and stability checks
4. **Caps Gate**: Minimum attempt mass, runtime, and backlog constraints

### 2.4 Slice Advancement

**Function**: `activate_next_slice(system_slug: str, attestation: Optional[Dict]) -> CurriculumSystem`

**Location**: `backend/frontier/curriculum.py:1185-1235`

**Process**:
1. Create backup of `curriculum.yaml`
2. Mark current slice as completed with deterministic timestamp
3. Append attestation record if provided
4. Update `active` field to next incomplete slice
5. Write updated configuration back to YAML
6. Reload and return updated `CurriculumSystem`

**Determinism**: Uses `deterministic_timestamp_from_content()` for reproducible timestamps.

---

## 3. Drift Detection & Determinism

### 3.1 Drift Sentinel

**Location**: `tools/repro/drift_sentinel.py`

**Purpose**: Detect nondeterministic operations in Python code.

**Monitored Patterns**:
- `time.time()` → Use `deterministic_unix_timestamp()`
- `datetime.utcnow()` / `datetime.now()` → Use `deterministic_timestamp()`
- `uuid.uuid4()` → Use `deterministic_uuid()`
- `np.random.*` / `random.*` → Use `SeededRNG()`

**Whitelist System**:
- File-level whitelist: Skip entire files
- Function-level whitelist: Skip specific functions
- `@deterministic_ok` decorator: Mark functions as exempt

**Usage**:
```bash
python tools/repro/drift_sentinel.py --all
python tools/repro/drift_sentinel.py --staged  # Pre-commit hook
```

**Output**:
- `artifacts/repro/drift_report.json`: Violation report
- `artifacts/repro/drift_patch.diff`: Suggested fixes

### 3.2 Determinism Utilities

**Location**: `backend/repro/determinism.py`

**Core Functions**:

1. **Timestamps**:
   - `deterministic_timestamp(seed: int) -> datetime`
   - `deterministic_unix_timestamp(seed: int) -> int`
   - `deterministic_timestamp_from_content(*parts) -> datetime`

2. **UUIDs**:
   - `deterministic_uuid(content: str, namespace: str) -> str`
   - `deterministic_uuid_from_hash(hash_hex: str) -> str`

3. **Random Number Generation**:
   - `SeededRNG(seed: int)`: Deterministic RNG wrapper
   - Methods: `random()`, `randint()`, `choice()`, `shuffle()`

4. **Hashing**:
   - `deterministic_hash(content: Any, algorithm: str) -> str`
   - `deterministic_seed_from_content(*parts) -> int`

5. **Identifiers**:
   - `deterministic_run_id(prefix: str, *parts, length: int) -> str`
   - `deterministic_slug(*parts, length: int) -> str`

**Epoch**: Fixed at `2025-01-01T00:00:00Z` for reproducibility.

### 3.3 Autofix Tools

**Location**: `tools/repro/`

- `autofix_drift.py`: Basic drift auto-fixer
- `autofix_drift_v3.py`: Enhanced version with better pattern matching
- `autofix_drift_v3_2.py`: Latest iteration with improved heuristics

---

## 4. Gap Analysis: Missing Components

### 4.1 ❌ Curriculum Fingerprinting

**Status**: **NOT IMPLEMENTED**

**Required Functionality**:
- Compute deterministic hash of curriculum configuration
- Include all slice parameters, gates, and invariants
- Stable across runs for reproducibility
- Detect configuration drift between runs

**Proposed Implementation**:
```python
def compute_curriculum_fingerprint(system: CurriculumSystem) -> str:
    """
    Compute deterministic fingerprint of curriculum configuration.
    
    Returns:
        SHA-256 hash of canonical curriculum representation
    """
    canonical = {
        "version": system.version,
        "slug": system.slug,
        "slices": [
            {
                "name": s.name,
                "params": sorted(s.params.items()),
                "gates": {
                    "coverage": asdict(s.gates.coverage),
                    "abstention": asdict(s.gates.abstention),
                    "velocity": asdict(s.gates.velocity),
                    "caps": asdict(s.gates.caps),
                }
            }
            for s in system.slices
        ],
        "invariants": sorted(system.invariants.items())
    }
    return deterministic_hash(canonical)
```

### 4.2 ❌ Runtime Drift Guards

**Status**: **NOT IMPLEMENTED**

**Required Functionality**:
- Detect changes to curriculum schema between runs
- Detect changes to slice parameters during execution
- Detect changes to gate thresholds
- Fail-fast on unexpected configuration drift

**Proposed Implementation**:
```python
@dataclass
class CurriculumDriftCheck:
    """Runtime drift guard for curriculum configuration."""
    
    baseline_fingerprint: str
    baseline_version: int
    baseline_slice_count: int
    
    def check(self, system: CurriculumSystem) -> List[str]:
        """
        Check for drift against baseline.
        
        Returns:
            List of drift violations (empty if clean)
        """
        violations = []
        
        current_fingerprint = compute_curriculum_fingerprint(system)
        if current_fingerprint != self.baseline_fingerprint:
            violations.append(
                f"Curriculum fingerprint mismatch: "
                f"expected {self.baseline_fingerprint[:12]}..., "
                f"got {current_fingerprint[:12]}..."
            )
        
        if system.version != self.baseline_version:
            violations.append(
                f"Version drift: expected {self.baseline_version}, "
                f"got {system.version}"
            )
        
        if len(system.slices) != self.baseline_slice_count:
            violations.append(
                f"Slice count drift: expected {self.baseline_slice_count}, "
                f"got {len(system.slices)}"
            )
        
        return violations
```

### 4.3 ❌ Slice Parameter Drift Detection

**Status**: **NOT IMPLEMENTED**

**Required Functionality**:
- Detect mid-run changes to slice parameters
- Validate parameter consistency across curriculum ladder
- Ensure monotonicity is preserved during runtime

**Proposed Implementation**:
```python
def validate_slice_parameter_drift(
    slice_cfg: CurriculumSlice,
    expected_params: Dict[str, Any]
) -> List[str]:
    """
    Validate that slice parameters match expected values.
    
    Args:
        slice_cfg: Current slice configuration
        expected_params: Expected parameter values
    
    Returns:
        List of drift violations
    """
    violations = []
    
    for key, expected_value in expected_params.items():
        actual_value = slice_cfg.params.get(key)
        if actual_value != expected_value:
            violations.append(
                f"Parameter drift in '{slice_cfg.name}.{key}': "
                f"expected {expected_value}, got {actual_value}"
            )
    
    return violations
```

### 4.4 ❌ Metric Schema Drift Detection

**Status**: **NOT IMPLEMENTED**

**Required Functionality**:
- Validate that metrics conform to expected schema
- Detect missing or extra fields in metric payloads
- Ensure gate evaluation receives correct metric structure

**Proposed Implementation**:
```python
def validate_metric_schema(
    metrics: Dict[str, Any],
    schema_version: str = "v1"
) -> List[str]:
    """
    Validate metrics against expected schema.
    
    Args:
        metrics: Raw metrics dictionary
        schema_version: Expected schema version
    
    Returns:
        List of schema violations
    """
    violations = []
    
    required_paths = [
        ("metrics", "rfl", "coverage", "ci_lower"),
        ("metrics", "rfl", "coverage", "sample_size"),
        ("metrics", "success_rates", "abstention_rate"),
        ("metrics", "curriculum", "active_slice", "attempt_mass"),
        ("metrics", "throughput", "proofs_per_hour"),
        ("provenance", "merkle_hash"),
    ]
    
    for path in required_paths:
        node = metrics
        for part in path:
            if not isinstance(node, dict) or part not in node:
                violations.append(f"Missing required field: {'.'.join(path)}")
                break
            node = node[part]
    
    return violations
```

---

## 5. Current Curriculum Configuration

### 5.1 Propositional Logic (PL) System

**Location**: `config/curriculum.yaml`

**Active Slice**: `slice_hard`

**Slice Inventory** (9 slices total):

| Slice Name | Atoms | Depth | Breadth | Total | Status |
|------------|-------|-------|---------|-------|--------|
| `slice_debug_uplift` | 2 | 2 | 8 | 16 | Incomplete |
| `slice_easy_fo` | 3 | 3 | 300 | 1000 | Incomplete |
| `slice_uplift_proto` | 3 | 4 | 3 | 10 | Incomplete |
| `atoms4-depth4` | 4 | 4 | 500 | 2000 | ✅ Completed |
| `atoms4-depth5` | 4 | 5 | 1000 | 5000 | ✅ Completed |
| `atoms5-depth6` | 5 | 6 | 2000 | 10000 | Incomplete |
| `slice_medium` | 5 | 7 | 1500 | 8000 | Incomplete |
| `first_organism_pl2_hard` | 6 | 8 | 2000 | 10000 | Incomplete |
| `slice_hard` | 7 | 12 | 3000 | 15000 | **Active** |

**Monotonicity Axes**: `atoms`, `depth_max`

**Invariants**:
- Attestation required: `true`
- Attestation window: 60 minutes

### 5.2 Wide Slice Configuration

**Designated Slice**: `slice_medium`

**Purpose**: RFL uplift experiments (Operation Dyno Chart)

**Parameters**:
- Atoms: 5
- Depth: 7
- Breadth: 1500
- Total: 8000

**Gates**:
- Coverage: CI lower ≥ 0.85, sample ≥ 20
- Abstention: Rate ≤ 15%, mass ≤ 800
- Velocity: PPH ≥ 150, CV ≤ 0.12
- Caps: Mass ≥ 3000, runtime ≥ 20min, backlog ≤ 0.40

---

## 6. Integration Points

### 6.1 U2 Uplift Pipeline

**Expected Integration**:
1. Load curriculum slice for current run
2. Apply slice parameters to derivation engine
3. Execute derivation with slice constraints
4. Collect metrics in v1-compliant format
5. Evaluate gates against slice thresholds
6. Advance slice if gates pass

**Current Gap**: No explicit U2 integration code found in repository.

### 6.2 Attestation System

**Location**: `backend/frontier/curriculum.py:1214-1219`

**Mechanism**:
- Attestation records appended to completed slices
- Structure: `{sealed_at: <timestamp>, audit: <dict>}`
- Stored in `curriculum.yaml` under `slice.attestations[]`

**Integration with Gates**:
- `require_attestation` flag in coverage gate
- Attestation hash validated in `NormalizedMetrics.from_raw()`

### 6.3 Metrics Normalization

**Class**: `NormalizedMetrics` (`backend/frontier/curriculum.py:767`)

**Source Paths** (flexible extraction):
```python
coverage_ci_lower = _first_available(metrics, [
    ("metrics", "rfl", "coverage", "ci_lower"),
    ("coverage", "ci_lower"),
    ("ci_lower",)
])
```

**Normalized Fields**:
- `coverage_ci_lower`, `coverage_sample_size`
- `abstention_rate_pct`
- `attempt_mass`, `slice_runtime_minutes`
- `proof_velocity_pph`, `velocity_cv`
- `backlog_fraction`
- `attestation_hash`

---

## 7. Test Coverage

### 7.1 Curriculum Tests

**Location**: `tests/frontier/`

**Files**:
- `test_curriculum_gates.py`: Gate evaluation logic
- `test_curriculum_slices.py`: Slice configuration validation
- `test_first_organism_hard.py`: First Organism slice tests

**Coverage Areas**:
- ✅ Wide Slice configuration
- ✅ Monotonicity constraints
- ✅ Gate threshold validation
- ✅ Slice accessibility
- ✅ Completion tracking

### 7.2 Integration Tests

**Location**: `tests/integration/test_wide_slice_logs.py`

**Purpose**: Validate Wide Slice logging and metrics.

---

## 8. Recommendations

### 8.1 Immediate Actions

1. **Implement Curriculum Fingerprinting**:
   - Add `compute_curriculum_fingerprint()` to `backend/frontier/curriculum.py`
   - Include fingerprint in audit logs
   - Store fingerprint in experiment metadata

2. **Add Drift Guards**:
   - Create `CurriculumDriftCheck` class
   - Integrate into `RFLRunner.__init__()`
   - Fail-fast on fingerprint mismatch

3. **Enhance Slice Execution**:
   - Add slice parameter validation before each run
   - Log slice transitions with fingerprints
   - Verify monotonicity at runtime

### 8.2 Medium-Term Improvements

1. **Metric Schema Validation**:
   - Implement `validate_metric_schema()`
   - Add to `NormalizedMetrics.from_raw()`
   - Emit warnings on schema drift

2. **Curriculum Versioning**:
   - Add `curriculum_version` to experiment metadata
   - Track curriculum changes in version control
   - Support curriculum rollback

3. **Drift Reporting**:
   - Extend drift sentinel to check curriculum files
   - Generate curriculum drift reports
   - Integrate with CI/CD pipeline

### 8.3 Long-Term Enhancements

1. **Curriculum Provenance**:
   - Track curriculum lineage across experiments
   - Link slices to experiment outcomes
   - Support curriculum A/B testing

2. **Dynamic Slice Adjustment**:
   - Allow runtime slice parameter tuning
   - Implement adaptive gate thresholds
   - Support curriculum learning

---

## 9. Invariant Compliance

### 9.1 Determinism ✅

**Status**: **COMPLIANT**

- All timestamps use `deterministic_timestamp()`
- Slice advancement uses content-based timestamps
- RNG operations use `SeededRNG`

### 9.2 Full Specification ✅

**Status**: **COMPLIANT**

- Curriculum schema is fully specified (Version 2)
- All slices have complete parameter sets
- Gates have explicit thresholds

### 9.3 Drift Detection ⚠️

**Status**: **PARTIALLY COMPLIANT**

- ✅ Code-level drift detection (drift_sentinel.py)
- ❌ Curriculum fingerprinting not implemented
- ❌ Runtime drift guards not implemented
- ❌ Metric schema drift detection not implemented

### 9.4 Fingerprint Stability ❌

**Status**: **NOT IMPLEMENTED**

- No fingerprinting mechanism exists
- Cannot verify stability across runs
- No baseline for drift comparison

---

## 10. Conclusion

The MathLedger curriculum system provides a **solid foundation** for deterministic slice execution with comprehensive gate evaluation and validation. However, **critical drift detection capabilities are missing**:

1. ❌ Curriculum fingerprinting
2. ❌ Runtime drift guards
3. ❌ Slice parameter drift detection
4. ❌ Metric schema drift detection

**Next Steps**:
1. Implement curriculum fingerprinting
2. Add runtime drift checks to RFL runner
3. Integrate drift guards into U2 uplift pipeline
4. Extend test coverage for drift scenarios

**Priority**: **HIGH** — Drift detection is essential for reproducibility and scientific validity of RFL experiments.

---

**End of Report**
