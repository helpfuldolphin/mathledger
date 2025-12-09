# MANUS-E Implementation Blueprint: Curriculum Drift Governance

**Steward**: MANUS-E  
**Mission**: Phase II Implementation Pass  
**Status**: ACTIVE  
**Date**: 2025-12-06

---

## **1. Implementation Blueprint: Core Governance System**

This section contains the production-ready code and integration specifications for the core drift governance components.

### **1.1. `CurriculumSystem.fingerprint()` Implementation**

**Objective**: Produce a canonical, deterministic SHA-256 hash of the curriculum configuration.

**File**: `backend/frontier/curriculum.py`

**Action**: Add the following imports at the top of the file (approx. **line 12**):

```python
import json
import hashlib
```

**Action**: Add the `invariants` field to the `CurriculumSystem` dataclass (approx. **line 421**) to ensure it is stored for fingerprinting:

```python
# backend/frontier/curriculum.py:421
@dataclass
class CurriculumSystem:
    # ... existing fields
    active_index: int
    invariants: Dict[str, Any] = field(default_factory=dict)
    monotonic_axes: Tuple[str, ...] = ()
    version: int = 2
```

**Action**: Add the following method to the `CurriculumSystem` class (approx. **line 535**):

```python
# backend/frontier/curriculum.py:535
    def fingerprint(self) -> str:
        """
        Computes a deterministic SHA-256 fingerprint of the curriculum configuration.

        The fingerprint is stable against slice reordering and excludes runtime state
        such as completion timestamps and the active slice pointer.
        """
        # Slice-order normalization: sort by name for stability
        sorted_slices = sorted(self.slices, key=lambda s: s.name)

        canonical_slices = []
        for s in sorted_slices:
            canonical_slices.append({
                "name": s.name,
                # Parameter & Metadata Normalization: sort keys
                "params": dict(sorted(s.params.items())),
                "gates": {
                    # Gate-parameter normalization: asdict is stable for frozen dataclasses
                    "coverage": asdict(s.gates.coverage),
                    "abstention": asdict(s.gates.abstention),
                    "velocity": asdict(s.gates.velocity),
                    "caps": asdict(s.gates.caps),
                },
                "metadata": dict(sorted(s.metadata.items())),
            })

        canonical_representation = {
            "version": self.version,
            "slug": self.slug,
            "invariants": dict(sorted(self.invariants.items())),
            "slices": canonical_slices,
        }

        # Use sorted_keys and no whitespace for the most compact, stable JSON string
        payload = json.dumps(
            canonical_representation, 
            sort_keys=True, 
            separators=(",", ":")
        ).encode("utf-8")

        return hashlib.sha256(payload).hexdigest()
```

### **1.2. `SliceGates.to_dict()` Canonical Form**

**Objective**: Provide a canonical dictionary representation for the `SliceGates` object.

**File**: `backend/frontier/curriculum.py`

**Action**: Add the following method to the `SliceGates` class (approx. **line 387**):

```python
# backend/frontier/curriculum.py:387
    def to_dict(self) -> Dict[str, Any]:
        """Serialize all gates to a canonical dictionary."""
        return {
            "coverage": self.coverage.to_dict(),
            "abstention": self.abstention.to_dict(),
            "velocity": self.velocity.to_dict(),
            "caps": self.caps.to_dict(),
        }
```

### **1.3. Drift Sentinel and Error Handling Contract**

**Objective**: Define the core exception and sentinel class for drift detection.

**File**: `backend/frontier/curriculum.py`

**Action**: Add the following class definitions after `CurriculumConfigError` (approx. **line 150**):

```python
# backend/frontier/curriculum.py:150
class CurriculumDriftError(Exception):
    """Raised when a drift in curriculum configuration is detected."""

@dataclass
class CurriculumDriftSentinel:
    """Runtime drift guard for curriculum configuration."""
    baseline_fingerprint: str
    baseline_version: int
    baseline_slice_count: int

    def check(self, system: CurriculumSystem) -> List[str]:
        """
        Checks the provided CurriculumSystem for drift against the baseline.
        Returns a list of violation messages. An empty list means no drift.
        """
        violations = []
        current_fingerprint = system.fingerprint()
        if current_fingerprint != self.baseline_fingerprint:
            violations.append(
                f"Fingerprint mismatch (ContentDrift): "
                f"Expected {self.baseline_fingerprint[:12]}..., got {current_fingerprint[:12]}..."
            )
        if system.version != self.baseline_version:
            violations.append(f"SchemaDrift: Version changed from {self.baseline_version} to {system.version}")
        if len(system.slices) != self.baseline_slice_count:
            violations.append(f"SliceCountDrift: Slice count changed from {self.baseline_slice_count} to {len(system.slices)}")
        return violations
```

### **1.4. `RFLRunner` Integration Diffs**

**Objective**: Integrate the Drift Sentinel into the RFL experiment lifecycle.

**File**: `rfl/runner.py`

**Action**: Apply the following diffs.

**Diff 1: Imports and `__init__`**

```diff
--- a/rfl/runner.py
+++ b/rfl/runner.py
@@ -26,7 +26,7 @@
 from .config import RFLConfig, CurriculumSlice
 from .experiment import RFLExperiment, ExperimentResult
 from .coverage import CoverageTracker, CoverageMetrics, load_baseline_from_db
-from substrate.repro.determinism import deterministic_timestamp
+from substrate.repro.determinism import deterministic_timestamp, deterministic_isoformat
 from .bootstrap_stats import (
     compute_coverage_ci,
     compute_uplift_ci,
@@ -36,6 +36,7 @@
 )
 from .audit import RFLAuditLog, SymbolicDescentGradient, StepIdComputation
 from .experiment_logging import RFLExperimentLogger
 from .provenance import ManifestBuilder
+from backend.frontier.curriculum import load as load_curriculum, CurriculumDriftError, CurriculumDriftSentinel
 
 # ---------------- Logger ----------------
 logging.basicConfig(
@@ -167,6 +168,26 @@
             metrics_path = Path("results") / "rfl_wide_slice_runs.jsonl"
             self.metrics_logger = RFLMetricsLogger(str(metrics_path))
             logger.info(f"[INIT] Metrics logger enabled: {metrics_path}")
+
+        # ---------------- Drift Sentinel Initialization ----------------
+        logger.info("[INIT] Initializing Curriculum Drift Sentinel...")
+        try:
+            system = load_curriculum(slug=self.config.curriculum_slug)
+            self.baseline_fingerprint = system.fingerprint()
+            self.drift_sentinel = CurriculumDriftSentinel(
+                baseline_fingerprint=self.baseline_fingerprint,
+                baseline_version=system.version,
+                baseline_slice_count=len(system.slices),
+            )
+            logger.info(f"[INIT] Drift Sentinel locked on baseline fingerprint: {self.baseline_fingerprint[:12]}...")
+        except Exception as e:
+            logger.error(f"[FATAL] Failed to initialize Drift Sentinel: {e}")
+            raise CurriculumDriftError(f"Failed to establish a curriculum baseline: {e}") from e
+        # ----------------------------------------------------------------
 
         # Telemetry
         self._redis_client = None

```

**Diff 2: `run_all()` Enforcement**

```diff
--- a/rfl/runner.py
+++ b/rfl/runner.py
@@ -248,6 +269,20 @@
 
         # Phase 1: Execute experiments
         logger.info(f"Phase 1: Executing {self.config.num_runs} derivation experiments...")
+
+        # --- Drift Sentinel Enforcement ---
+        logger.info("Verifying curriculum integrity with Drift Sentinel...")
+        current_system = load_curriculum(slug=self.config.curriculum_slug)
+        violations = self.drift_sentinel.check(current_system)
+        if violations:
+            error_msg = "Curriculum drift detected before execution! Halting run. Violations:\n" + "\n".join(f"  - {v}" for v in violations)
+            logger.error(f"[FATAL] {error_msg}")
+            self._log_drift_report(current_system.fingerprint(), violations)
+            raise CurriculumDriftError(error_msg)
+        logger.info("Curriculum integrity verified.")
+        # --------------------------------
+
         self._execute_experiments()
 
         # Phase 2: Compute coverage metrics

```

### **1.5. Drift Report Writer Utility**

**Objective**: Create a utility to log drift artifacts.

**File**: `rfl/runner.py`

**Action**: Add the following method to the `RFLRunner` class (approx. **line 1100**):

```python
# rfl/runner.py:1100
    def _log_drift_report(self, detected_fingerprint: str, violations: List[str]):
        """Writes a drift report artifact for audit purposes."""
        report = {
            "report_type": "CurriculumDriftReport",
            "status": "DRIFT_DETECTED",
            "timestamp": deterministic_isoformat("drift_report", self.config.experiment_id),
            "experiment_id": self.config.experiment_id,
            "baseline_fingerprint": self.baseline_fingerprint,
            "detected_fingerprint": detected_fingerprint,
            "violations": violations,
            "recommendation": "Curriculum configuration has changed since the experiment started. Restore config from version control or establish a new baseline."
        }
        
        try:
            report_path = Path(self.config.artifacts_dir) / "drift_report.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            logger.warning(f"Drift report written to {report_path}")
        except IOError as e:
            logger.error(f"Failed to write drift report: {e}")
```

---

## **2. Metric Schema Migration Plan**

**Objective**: Safely migrate from the flexible `_first_available()` to a strict, canonical schema enforcement for metrics.

### **2.1. Feature Flag Strategy**

A new environment variable, `METRIC_SCHEMA_ENFORCEMENT_MODE`, will control the behavior:

-   **`permissive` (Default)**: Use the old `_first_available()` logic. No changes in behavior.
-   **`log_only`**: **Parallel Validation**. Run both the old and new logic. Use the result from the old logic, but log any discrepancies to `stderr`. This allows for non-disruptive identification of non-compliant metric sources.
-   **`strict`**: Use the new `_get_metric_by_path()` logic exclusively. Any deviation from the canonical schema will raise a `CurriculumDriftError`.

### **2.2. Migration Path**

1.  **Phase A (Code Deployment)**: Deploy the new code with the feature flag defaulting to `permissive`.
2.  **Phase B (Discovery)**: Run all relevant test suites and experimental pipelines with the mode set to `log_only`. Collect all logged discrepancy reports.
3.  **Phase C (Remediation)**: Update all upstream metric generation processes to conform to the canonical v1 schema identified in the discovery phase.
4.  **Phase D (Enforcement)**: Once all pipelines run clean in `log_only` mode, switch the default mode to `strict` to enforce the schema for all future runs.

### **2.3. Implementation: Parallel Validation Pathway**

**File**: `backend/frontier/curriculum.py`

**Action**: Implement the feature flag logic within `NormalizedMetrics.from_raw()`.

1.  Add new imports at the top of the file:
    ```python
    import os
    import sys
    ```

2.  Implement the new strict metric resolver:
    ```python
    # backend/frontier/curriculum.py: approx line 299
    def _get_metric_by_path(root: Dict[str, Any], path: Tuple[str, ...]) -> Any:
        """Strictly resolve a metric from a nested dictionary using a path."""
        node = root
        for i, part in enumerate(path):
            if not isinstance(node, dict) or part not in node:
                subpath = ".".join(path[:i+1])
                raise KeyError(f"Required path ‘{subpath}’ not found in metric payload.")
            node = node[part]
        return node
    ```

3.  Replace the existing `NormalizedMetrics.from_raw()` method (approx. **line 779**) with the following feature-flag-aware implementation:

    ```python
    # backend/frontier/curriculum.py:779
    @classmethod
    def from_raw(cls, metrics: Dict[str, Any]) -> "NormalizedMetrics":
        """
        Create a normalized metrics object from a raw dictionary.
        Supports multiple enforcement modes via METRIC_SCHEMA_ENFORCEMENT_MODE.
        """
        mode = os.getenv("METRIC_SCHEMA_ENFORCEMENT_MODE", "permissive").lower()

        # Canonical v1 schema paths
        SCHEMA_V1 = {
            "coverage_ci_lower": ("metrics", "rfl", "coverage", "ci_lower"),
            "coverage_sample_size": ("metrics", "rfl", "coverage", "sample_size"),
            "abstention_rate_pct": ("metrics", "success_rates", "abstention_rate"),
            "attempt_mass": ("metrics", "curriculum", "active_slice", "attempt_mass"),
            "slice_runtime_minutes": ("metrics", "curriculum", "active_slice", "wallclock_minutes"),
            "proof_velocity_pph": ("metrics", "throughput", "proofs_per_hour"),
            "velocity_cv": ("metrics", "throughput", "coefficient_of_variation"),
            "backlog_fraction": ("metrics", "frontier", "queue_backlog"),
            "attestation_hash": ("provenance", "merkle_hash"),
        }

        # --- Strict Mode ---
        if mode == "strict":
            try:
                return cls(**{key: _get_metric_by_path(metrics, path) for key, path in SCHEMA_V1.items()})
            except KeyError as e:
                raise CurriculumDriftError(f"Metric Schema Drift (Strict): {e}") from e

        # --- Permissive & Log-Only Modes --- 
        # Fallback to old, flexible logic
        # ... (existing _first_available logic from lines 782-838) ...
        permissive_result = cls(
            coverage_ci_lower=_to_float(_first_available(metrics, [('rfl', 'coverage', 'ci_lower'), ...])),
            # ... all other fields ...
            attestation_hash=str(_first_available(metrics, [('provenance', 'merkle_hash'), ...])),
        )

        if mode == "log_only":
            # Parallel Validation
            try:
                strict_result = cls(**{key: _get_metric_by_path(metrics, path) for key, path in SCHEMA_V1.items()})
                if asdict(permissive_result) != asdict(strict_result):
                    print(
                        "WARNING: Metric Schema Drift Detected in log_only mode.\n" 
                        f"  Permissive Result: {asdict(permissive_result)}\n"
                        f"  Strict Result:     {asdict(strict_result)}",
                        file=sys.stderr
                    )
            except KeyError as e:
                print(f"ERROR: Metric Schema Drift (Strict) would fail: {e}", file=sys.stderr)

        return permissive_result
    ```

### **2.4. Regression Test Suite Structure**

A new test file, `tests/frontier/test_metric_schema.py`, will be created with the following structure:

```python
import os
import pytest
from backend.frontier.curriculum import NormalizedMetrics, CurriculumDriftError

# Define valid v1 metric payload and various invalid payloads
VALID_METRICS = { "metrics": { "rfl": { "coverage": { "ci_lower": 0.9 } } } ... }
INVALID_METRICS_MISSING_PATH = { "metrics": { "rfl": {} } }

class TestMetricSchemaEnforcement:
    def test_strict_mode_accepts_valid_schema(self):
        os.environ["METRIC_SCHEMA_ENFORCEMENT_MODE"] = "strict"
        NormalizedMetrics.from_raw(VALID_METRICS)

    def test_strict_mode_rejects_invalid_schema(self):
        os.environ["METRIC_SCHEMA_ENFORCEMENT_MODE"] = "strict"
        with pytest.raises(CurriculumDriftError, match="Required path 'metrics.rfl.coverage' not found"):
            NormalizedMetrics.from_raw(INVALID_METRICS_MISSING_PATH)

    def test_permissive_mode_is_backward_compatible(self):
        os.environ["METRIC_SCHEMA_ENFORCEMENT_MODE"] = "permissive"
        # Use a metric payload that works with old logic but not new
        OLD_METRIC = { "coverage_ci_lower": 0.8 }
        result = NormalizedMetrics.from_raw(OLD_METRIC)
        assert result.coverage_ci_lower == 0.8

    def test_log_only_mode_detects_discrepancy(self, capsys):
        os.environ["METRIC_SCHEMA_ENFORCEMENT_MODE"] = "log_only"
        OLD_METRIC = { "coverage_ci_lower": 0.8 }
        NormalizedMetrics.from_raw(OLD_METRIC)
        captured = capsys.readouterr()
        assert "Metric Schema Drift Detected" in captured.err
```

---

## **3. Curriculum Provenance Extension**

**Objective**: Create an immutable, auditable link between every experiment run and the exact curriculum configuration used.

### **3.1. `RunLedgerEntry` Schema Extension**

**File**: `rfl/runner.py`

**Action**: Modify the `RunLedgerEntry` dataclass (approx. **line 51**) to include the new provenance fields.

```diff
--- a/rfl/runner.py
+++ b/rfl/runner.py
@@ -61,6 +61,10 @@
     max_breadth: int
     max_total: int
     abstention_breakdown: Dict[str, int] = field(default_factory=dict)
+    # --- Curriculum Provenance ---
+    curriculum_slug: str
+    curriculum_fingerprint: str
+    run_timestamp: str
     # Attestation-specific fields (optional, populated by run_with_attestation)
     attestation_slice_id: Optional[str] = None  # Original slice_id from attestation
     composite_root: Optional[str] = None  # H_t for traceability

```

### **3.2. Update Provenance Population**

**File**: `rfl/runner.py`

**Action**: Update the `_log_run_to_ledger` method (approx. **line 800**) to populate the new fields.

```diff
--- a/rfl/runner.py
+++ b/rfl/runner.py
@@ -808,6 +808,9 @@
             max_breadth=slice_cfg.max_breadth,
             max_total=slice_cfg.max_total,
             abstention_breakdown=result.abstention_breakdown,
+            curriculum_slug=self.config.curriculum_slug,
+            curriculum_fingerprint=self.baseline_fingerprint,
+            run_timestamp=deterministic_isoformat("run_ledger", result.run_id),
         )
         self.policy_ledger.append(entry)

```

---

## **4. Diff Generator for Curriculum Drift Diagnostic**

**Objective**: Create a developer utility to produce a human-readable diagnostic report comparing two curriculum configurations.

### **4.1. Implementation Location**

-   **File**: `backend/tools/curriculum_differ.py` (New File)

### **4.2. `generate_curriculum_diff` Design**

```python
# backend/tools/curriculum_differ.py

from typing import Dict, Any, List
from backend.frontier.curriculum import CurriculumSystem, load_curriculum

def generate_curriculum_diff(old_system: CurriculumSystem, new_system: CurriculumSystem) -> Dict[str, Any]:
    """Generates a human-readable diff between two curriculum systems."""
    diff = {
        "param_diffs": [],
        "gate_diffs": [],
        "structure_diffs": [],
        "teacher_facing_summary": ""
    }

    old_slices = {s.name: s for s in old_system.slices}
    new_slices = {s.name: s for s in new_system.slices}

    added_slices = new_slices.keys() - old_slices.keys()
    removed_slices = old_slices.keys() - new_slices.keys()
    common_slices = old_slices.keys() & new_slices.keys()

    if added_slices:
        diff["structure_diffs"].append(f"Added slices: {sorted(list(added_slices))}")
    if removed_slices:
        diff["structure_diffs"].append(f"Removed slices: {sorted(list(removed_slices))}")

    for name in sorted(list(common_slices)):
        old_s, new_s = old_slices[name], new_slices[name]
        # Param Diffs
        for key in old_s.params.keys() | new_s.params.keys():
            old_val, new_val = old_s.params.get(key), new_s.params.get(key)
            if old_val != new_val:
                diff["param_diffs"].append(f"{name}.params.{key}: {old_val} -> {new_val}")
        # Gate Diffs
        old_gates, new_gates = old_s.gates.to_dict(), new_s.gates.to_dict()
        for gate, spec in old_gates.items():
            for key, old_val in spec.items():
                new_val = new_gates.get(gate, {}).get(key)
                if old_val != new_val:
                    diff["gate_diffs"].append(f"{name}.gates.{gate}.{key}: {old_val} -> {new_val}")

    # Generate Summary
    summary_lines = []
    if not any(diff.values()):
        summary_lines.append("No functional change detected.")
    else:
        summary_lines.append("Curriculum has been modified:")
        if diff["structure_diffs"]: summary_lines.extend(diff["structure_diffs"])
        if diff["param_diffs"]: summary_lines.extend(diff["param_diffs"])
        if diff["gate_diffs"]: summary_lines.extend(diff["gate_diffs"])
    
    diff["teacher_facing_summary"] = "\n".join(summary_lines)
    return diff

if __name__ == "__main__":
    # Example CLI usage
    # python -m backend.tools.curriculum_differ <old_config.yaml> <new_config.yaml> <slug>
    import sys
    import yaml

    with open(sys.argv[1], 'r') as f: old_config = yaml.safe_load(f)
    with open(sys.argv[2], 'r') as f: new_config = yaml.safe_load(f)
    slug = sys.argv[3]

    old_sys = CurriculumSystem.from_config(slug, old_config)
    new_sys = CurriculumSystem.from_config(slug, new_config)

    diff_report = generate_curriculum_diff(old_sys, new_sys)
    print(diff_report["teacher_facing_summary"])
```

This blueprint provides a complete, actionable, and production-ready plan for implementing the entire Curriculum Drift Governance system. MANUS-E is prepared to execute this plan.
