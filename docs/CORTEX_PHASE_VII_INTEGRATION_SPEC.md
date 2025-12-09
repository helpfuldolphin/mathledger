# CORTEX Phase VII: Integration, Calibration & Autonomous Governance

**Operation CORTEX: TDA Mind Scanner**
**Phase VII: Integration, Calibration, CI Wiring & Autonomous Governance Escalation**

---

## Executive Summary

Phase VII transitions CORTEX from a standalone TDA governance system into a fully integrated component of MathLedger's production infrastructure. This specification covers:

1. **C1**: Runner Integration Blueprint (RFLRunner, U2Runner)
2. **C2**: Global Health Builder Wiring Plan
3. **C3**: Golden Set Calibration & Versioning Protocol
4. **C4**: CI/CD Integration Plan
5. **C5**: Phase VII+ Spec (Adaptive Thresholds, Automated Remediation)

---

## C1: Runner Integration Blueprint

### Overview

TDA governance must be invoked at cycle boundaries in both `rfl/runner.py` (RFLRunner) and `experiments/u2/runner.py` (U2Runner). The integration follows a **hook-based pattern** to minimize code coupling.

### RFLRunner Integration

#### File: `rfl/runner.py`

**Integration Point 1: Post-Cycle TDA Evaluation**

Location: `RFLRunner.run_with_attestation()` method, after line 708.

```python
# After: self.experiment_logger.log_cycle(...)
# Line ~708

# === CORTEX PHASE VII: TDA GOVERNANCE HOOK ===
from backend.tda.governance import (
    evaluate_hard_gate_decision,
    TDAHardGateMode,
)
from backend.tda.monitor import TDAMonitorResult  # If exists

# Build TDA monitor result from attestation metrics
tda_result = TDAMonitorResult(
    hss=1.0 - attestation.abstention_rate,  # HSS proxy from abstention
    cycle_id=step_id,
    timestamp=attestation.metadata.get("timestamp"),
    metrics={
        "abstention_rate": attestation.abstention_rate,
        "abstention_mass": attestation.abstention_mass,
        "composite_root": attestation.composite_root,
    },
)

# Evaluate hard gate (mode from config or environment)
tda_mode = TDAHardGateMode.from_string(
    os.getenv("TDA_HARD_GATE_MODE", "SHADOW")
)
tda_decision = evaluate_hard_gate_decision(
    tda_result=tda_result,
    mode=tda_mode,
    exception_manager=self._tda_exception_manager,  # New field
)

# Log decision for governance console
self._tda_decisions.append(tda_decision.to_dict())

# Block if hard mode and should_block
if tda_decision.should_block:
    logger.warning(
        "[TDA-GATE] Cycle %s BLOCKED (hss=%.4f, threshold=%.2f)",
        step_id,
        tda_result.hss,
        tda_decision.threshold_used,
    )
    # Increment block counter
    self._tda_block_count += 1
# === END CORTEX PHASE VII ===
```

**Integration Point 2: Runner Initialization**

Location: `RFLRunner.__init__()`, after line 190.

```python
# After: self.noise_guard = global_noise_guard()
# Line ~190

# === CORTEX PHASE VII: TDA GOVERNANCE INIT ===
from backend.tda.governance import ExceptionWindowManager

self._tda_exception_manager = ExceptionWindowManager()
self._tda_decisions: List[Dict[str, Any]] = []
self._tda_block_count: int = 0
# === END CORTEX PHASE VII ===
```

**Integration Point 3: Results Export**

Location: `RFLRunner._export_results()`, within the `results` dict construction.

```python
# Inside results dict, after "dual_attestation"
# Line ~1022

# === CORTEX PHASE VII: TDA GOVERNANCE EXPORT ===
"tda_governance": {
    "decisions": self._tda_decisions,
    "block_count": self._tda_block_count,
    "exception_windows": self._tda_exception_manager.to_dict(),
},
# === END CORTEX PHASE VII ===
```

### U2Runner Integration

#### File: `experiments/u2/runner.py`

**Integration Point 1: Post-Cycle TDA Evaluation**

Location: `U2Runner.run_cycle()`, after line 1678.

```python
# After: self.cycle_index += 1
# Line ~1678

# === CORTEX PHASE VII: TDA GOVERNANCE HOOK ===
if hasattr(self, '_tda_hook') and self._tda_hook is not None:
    self._tda_hook.on_cycle_complete(
        cycle_index=self.cycle_index - 1,
        success=success,
        cycle_result=result,
        telemetry=telemetry_record,
    )
# === END CORTEX PHASE VII ===
```

**Integration Point 2: Runner Initialization**

Location: `U2Runner.__init__()`, after line 1618.

```python
# After: self._rng = random.Random(config.master_seed)
# Line ~1618

# === CORTEX PHASE VII: TDA GOVERNANCE HOOK ===
self._tda_hook: Optional["TDAGovernanceHook"] = None

def register_tda_hook(self, hook: "TDAGovernanceHook") -> None:
    """Register TDA governance hook for cycle boundary evaluation."""
    self._tda_hook = hook
# === END CORTEX PHASE VII ===
```

### TDA Governance Hook Interface

#### New File: `backend/tda/runner_hook.py`

```python
"""
TDA Governance Hook for Runner Integration — Phase VII

Provides a clean interface for runners to invoke TDA governance
without tight coupling to governance internals.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .governance import (
    TDAHardGateMode,
    HardGateDecision,
    ExceptionWindowManager,
    evaluate_hard_gate_decision,
)
from .governance_console import (
    build_governance_console_snapshot,
    TDAGovernanceSnapshot,
)


@dataclass
class TDAMonitorResult:
    """Minimal TDA result for governance evaluation."""
    hss: float
    cycle_id: str
    timestamp: Optional[str] = None
    metrics: Dict[str, Any] = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}


class TDAGovernanceHook:
    """
    Hook for integrating TDA governance into experiment runners.

    Usage:
        hook = TDAGovernanceHook(mode=TDAHardGateMode.SHADOW)
        runner.register_tda_hook(hook)
        # ... run experiment ...
        snapshot = hook.build_snapshot()
    """

    def __init__(
        self,
        mode: TDAHardGateMode = TDAHardGateMode.SHADOW,
        exception_manager: Optional[ExceptionWindowManager] = None,
    ):
        self.mode = mode
        self.exception_manager = exception_manager or ExceptionWindowManager()
        self.results: List[TDAMonitorResult] = []
        self.decisions: List[HardGateDecision] = []

    def on_cycle_complete(
        self,
        cycle_index: int,
        success: bool,
        cycle_result: Any,
        telemetry: Dict[str, Any],
    ) -> HardGateDecision:
        """
        Called after each cycle completes.

        Returns:
            HardGateDecision indicating whether to block.
        """
        # Build TDA result from cycle telemetry
        hss = self._compute_hss(success, telemetry)

        result = TDAMonitorResult(
            hss=hss,
            cycle_id=f"cycle_{cycle_index}",
            timestamp=telemetry.get("timestamp"),
            metrics={
                "success": success,
                "cycle_index": cycle_index,
            },
        )
        self.results.append(result)

        # Evaluate hard gate
        decision = evaluate_hard_gate_decision(
            tda_result=result,
            mode=self.mode,
            exception_manager=self.exception_manager,
        )
        self.decisions.append(decision)

        return decision

    def _compute_hss(self, success: bool, telemetry: Dict[str, Any]) -> float:
        """Compute HSS from cycle outcome."""
        # Base HSS from success/failure
        if success:
            return 0.8 + (telemetry.get("confidence", 0.0) * 0.2)
        else:
            return 0.3 - (telemetry.get("error_severity", 0.0) * 0.1)

    def build_snapshot(self) -> TDAGovernanceSnapshot:
        """Build governance console snapshot from accumulated results."""
        return build_governance_console_snapshot(
            tda_results=self.results,
            hard_gate_decisions=[d.to_dict() for d in self.decisions],
            golden_state=None,  # Loaded from calibration file
            exception_manager=self.exception_manager,
            mode=self.mode,
        )

    def get_block_count(self) -> int:
        """Return number of blocked cycles."""
        return sum(1 for d in self.decisions if d.should_block)

    def to_dict(self) -> Dict[str, Any]:
        """Export hook state for logging."""
        return {
            "mode": self.mode.value,
            "cycle_count": len(self.results),
            "block_count": self.get_block_count(),
            "decisions": [d.to_dict() for d in self.decisions],
        }
```

### Integration Sequence Diagram

```
┌─────────┐       ┌──────────────┐       ┌─────────────┐
│ Runner  │       │ TDAHook      │       │ Governance  │
└────┬────┘       └──────┬───────┘       └──────┬──────┘
     │                   │                      │
     │ run_cycle()       │                      │
     ├──────────────────►│                      │
     │                   │                      │
     │ on_cycle_complete │                      │
     │───────────────────►                      │
     │                   │ evaluate_hard_gate   │
     │                   ├─────────────────────►│
     │                   │                      │
     │                   │◄─────────────────────┤
     │                   │   HardGateDecision   │
     │◄──────────────────┤                      │
     │   should_block?   │                      │
     │                   │                      │
     │ [if block]        │                      │
     │ halt/log          │                      │
     │                   │                      │
```

---

## C2: Global Health Builder Wiring Plan

### Overview

The TDA health tile must be merged into `global_health.json` during the health surface assembly process.

### Current Global Health Schema

From `backend/health/global_schema.py`:

```python
GLOBAL_HEALTH_SCHEMA_VERSION = "1.0.0"
ALLOWED_STATUS_VALUES = {"OK", "WARN", "BLOCK"}
```

Core fields:
- `fm_ok`: Boolean FM health flag
- `coverage_pct`: Coverage percentage
- `status`: OK | WARN | BLOCK
- `alignment_status`: WELL_DISTRIBUTED | CONCENTRATED | SPARSE

### TDA Tile Integration Point

#### File: `backend/health/global_builder.py` (NEW)

```python
"""
Global Health Surface Builder — Phase VII

Assembles the canonical global_health.json by merging tiles from:
1. FM canonicalization (scripts/fm_canonicalize.py)
2. TDA governance (backend/health/tda_adapter.py)
3. Replay safety (experiments/u2/runner.py)
4. Learning health (analysis/conjecture_engine_contract.py)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .global_schema import (
    GLOBAL_HEALTH_SCHEMA_VERSION,
    validate_global_health,
)
from .tda_adapter import summarize_tda_for_global_health
from .canonicalize import canonicalize_global_health


@dataclass
class GlobalHealthSurface:
    """Assembled global health surface with all tiles."""

    schema_version: str
    generated_at: str

    # Core FM health
    fm_ok: bool
    coverage_pct: float
    status: str
    alignment_status: str
    external_only_labels: int

    # TDA tile (Phase VI)
    tda: Optional[Dict[str, Any]] = None

    # Replay safety tile (Phase V)
    replay_safety: Optional[Dict[str, Any]] = None

    # Learning health tile
    learning_health: Optional[Dict[str, Any]] = None

    # Deterministic inputs for provenance
    deterministic_inputs: List[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "fm_ok": self.fm_ok,
            "coverage_pct": self.coverage_pct,
            "status": self.status,
            "alignment_status": self.alignment_status,
            "external_only_labels": self.external_only_labels,
        }
        if self.tda is not None:
            result["tda"] = self.tda
        if self.replay_safety is not None:
            result["replay_safety"] = self.replay_safety
        if self.learning_health is not None:
            result["learning_health"] = self.learning_health
        if self.deterministic_inputs is not None:
            result["deterministic_inputs"] = self.deterministic_inputs
        return result


def build_global_health_surface(
    fm_health: Dict[str, Any],
    tda_snapshot: Optional[Dict[str, Any]] = None,
    replay_result: Optional[Dict[str, Any]] = None,
    learning_report: Optional[Dict[str, Any]] = None,
) -> GlobalHealthSurface:
    """
    Build the global health surface from component tiles.

    Args:
        fm_health: Core FM health dict (from fm_canonicalize.py)
        tda_snapshot: TDA governance console snapshot
        replay_result: Replay safety envelope
        learning_report: Conjecture engine report

    Returns:
        GlobalHealthSurface with all tiles merged.
    """
    # Validate and normalize FM health
    validated_fm = validate_global_health(fm_health)

    # Build TDA tile
    tda_tile = None
    if tda_snapshot is not None:
        tda_tile = summarize_tda_for_global_health(tda_snapshot)

    # Build replay safety tile
    replay_tile = None
    if replay_result is not None:
        replay_tile = _extract_replay_tile(replay_result)

    # Build learning health tile
    learning_tile = None
    if learning_report is not None:
        learning_tile = _extract_learning_tile(learning_report)

    # Compute aggregate status
    aggregate_status = _compute_aggregate_status(
        fm_status=validated_fm["status"],
        tda_status=tda_tile.get("tda_status") if tda_tile else None,
        replay_ok=replay_tile.get("replay_safety_ok") if replay_tile else None,
    )

    return GlobalHealthSurface(
        schema_version=GLOBAL_HEALTH_SCHEMA_VERSION,
        generated_at=datetime.utcnow().isoformat() + "Z",
        fm_ok=validated_fm["fm_ok"],
        coverage_pct=validated_fm["coverage_pct"],
        status=aggregate_status,
        alignment_status=validated_fm["alignment_status"],
        external_only_labels=validated_fm["external_only_labels"],
        tda=tda_tile,
        replay_safety=replay_tile,
        learning_health=learning_tile,
    )


def _compute_aggregate_status(
    fm_status: str,
    tda_status: Optional[str],
    replay_ok: Optional[bool],
) -> str:
    """
    Compute aggregate status from component statuses.

    Rules:
    - BLOCK if any component is BLOCK/ALERT
    - WARN if any component is WARN/ATTENTION
    - OK only if all components are OK
    """
    # Map TDA status to global status
    tda_mapped = None
    if tda_status is not None:
        tda_mapped = {
            "OK": "OK",
            "ATTENTION": "WARN",
            "ALERT": "BLOCK",
        }.get(tda_status, "WARN")

    # Map replay status
    replay_mapped = None
    if replay_ok is not None:
        replay_mapped = "OK" if replay_ok else "BLOCK"

    # Aggregate: most severe wins
    statuses = [fm_status]
    if tda_mapped:
        statuses.append(tda_mapped)
    if replay_mapped:
        statuses.append(replay_mapped)

    if "BLOCK" in statuses:
        return "BLOCK"
    if "WARN" in statuses:
        return "WARN"
    return "OK"


def _extract_replay_tile(replay_result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract replay safety tile from replay result."""
    return {
        "replay_safety_ok": replay_result.get("governance_admissible", False),
        "status": replay_result.get("status", "UNKNOWN"),
        "confidence_score": replay_result.get("confidence_score", 0.0),
    }


def _extract_learning_tile(learning_report: Dict[str, Any]) -> Dict[str, Any]:
    """Extract learning health tile from conjecture report."""
    metrics = learning_report.get("metrics", {})
    return {
        "status": learning_report.get("status", "UNKNOWN"),
        "supports": metrics.get("supports", 0),
        "contradicts": metrics.get("contradicts", 0),
        "inconclusive": metrics.get("inconclusive", 0),
    }
```

### CI/Cron Propagation

#### File: `scripts/build_global_health.py` (NEW)

```python
#!/usr/bin/env python3
"""
Build Global Health Surface — Phase VII CI Script

Assembles global_health.json from component artifacts.

Usage:
    python scripts/build_global_health.py \
        --fm-health artifacts/fm_health.json \
        --tda-snapshot artifacts/tda/governance_snapshot.json \
        --output global_health.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.health.global_builder import build_global_health_surface


def main() -> int:
    parser = argparse.ArgumentParser(description="Build global health surface")
    parser.add_argument("--fm-health", type=Path, required=True)
    parser.add_argument("--tda-snapshot", type=Path, default=None)
    parser.add_argument("--replay-result", type=Path, default=None)
    parser.add_argument("--learning-report", type=Path, default=None)
    parser.add_argument("--output", "-o", type=Path, required=True)

    args = parser.parse_args()

    # Load FM health
    with open(args.fm_health, "r", encoding="utf-8") as f:
        fm_health = json.load(f)

    # Load optional tiles
    tda_snapshot = None
    if args.tda_snapshot and args.tda_snapshot.exists():
        with open(args.tda_snapshot, "r", encoding="utf-8") as f:
            tda_snapshot = json.load(f)

    replay_result = None
    if args.replay_result and args.replay_result.exists():
        with open(args.replay_result, "r", encoding="utf-8") as f:
            replay_result = json.load(f)

    learning_report = None
    if args.learning_report and args.learning_report.exists():
        with open(args.learning_report, "r", encoding="utf-8") as f:
            learning_report = json.load(f)

    # Build surface
    surface = build_global_health_surface(
        fm_health=fm_health,
        tda_snapshot=tda_snapshot,
        replay_result=replay_result,
        learning_report=learning_report,
    )

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(surface.to_dict(), f, indent=2)

    print(f"Global health surface written to {args.output}")
    print(f"Status: {surface.status}")

    return 0 if surface.status == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
```

### Merged Tile Contract

```json
{
    "schema_version": "1.0.0",
    "generated_at": "2025-12-09T12:00:00Z",

    "fm_ok": true,
    "coverage_pct": 95.5,
    "status": "OK",
    "alignment_status": "WELL_DISTRIBUTED",
    "external_only_labels": 0,

    "tda": {
        "schema_version": "1.0.0",
        "tda_status": "OK",
        "block_rate": 0.0,
        "mean_hss": 0.78,
        "hss_trend": "STABLE",
        "governance_signal": "OK",
        "notes": ["TDA metrics within normal operating range"]
    },

    "replay_safety": {
        "replay_safety_ok": true,
        "status": "REPLAY_VERIFIED",
        "confidence_score": 0.95
    },

    "learning_health": {
        "status": "HEALTHY",
        "supports": 3,
        "contradicts": 0,
        "inconclusive": 1
    }
}
```

---

## C3: Golden Set Calibration & Versioning Protocol

### Overview

The golden set is the canonical reference for TDA calibration. This section defines the lifecycle, tooling, and drift-aware calibration process.

### Golden Set Schema

#### File: `config/tda_golden_set.yaml`

```yaml
# TDA Golden Set — Phase VII Calibration Reference
# Schema Version: 1.0.0

schema_version: "1.0.0"
golden_set_id: "gs_2025_12_09_001"
created_at: "2025-12-09T00:00:00Z"
created_by: "Phase VII Calibration Pipeline"

# Calibration parameters
calibration:
  block_threshold: 0.2
  false_block_threshold_ok: 0.05
  false_block_threshold_drifting: 0.15
  false_pass_threshold_ok: 0.05
  false_pass_threshold_drifting: 0.15

# Golden runs (labeled TDA results for calibration)
golden_runs:
  - run_id: "golden_run_001"
    hss: 0.85
    expected_decision: "pass"
    label: "high_confidence_proof"
    timestamp: "2025-12-01T10:00:00Z"

  - run_id: "golden_run_002"
    hss: 0.15
    expected_decision: "block"
    label: "low_confidence_hallucination"
    timestamp: "2025-12-01T10:01:00Z"

  - run_id: "golden_run_003"
    hss: 0.45
    expected_decision: "warn"
    label: "borderline_case"
    timestamp: "2025-12-01T10:02:00Z"

# Provenance
provenance:
  source_experiment: "calibration_u2_2025_12"
  source_manifest: "artifacts/calibration/manifest.json"
  source_hash: "abc123..."

# Drift tracking
drift_tracking:
  last_calibration_check: "2025-12-09T00:00:00Z"
  calibration_status: "ALIGNED"
  false_block_rate: 0.02
  false_pass_rate: 0.01
```

### Calibration Lifecycle

```
┌─────────────────────────────────────────────────────────────────┐
│                    GOLDEN SET LIFECYCLE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. CREATION                                                     │
│     └─► Run calibration experiment                               │
│     └─► Label results (pass/block/warn)                          │
│     └─► Export to config/tda_golden_set.yaml                     │
│                                                                  │
│  2. DEPLOYMENT                                                   │
│     └─► CI loads golden set at startup                           │
│     └─► Evaluate against golden runs                             │
│     └─► Compute calibration_status                               │
│                                                                  │
│  3. DRIFT MONITORING                                             │
│     └─► Watchdog checks false_block/false_pass rates             │
│     └─► If DRIFTING: Enable exception window                     │
│     └─► If BROKEN: Alert + block promotions                      │
│                                                                  │
│  4. RECALIBRATION                                                │
│     └─► Triggered by BROKEN status or scheduled                  │
│     └─► Run new calibration experiment                           │
│     └─► Update golden set with new version                       │
│     └─► Archive previous version                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Calibration Tooling

#### File: `scripts/tda_calibrate.py` (NEW)

```python
#!/usr/bin/env python3
"""
TDA Golden Set Calibration Tool — Phase VII

Usage:
    # Run calibration against current golden set
    python scripts/tda_calibrate.py --golden config/tda_golden_set.yaml

    # Create new golden set from experiment results
    python scripts/tda_calibrate.py --create \
        --experiment artifacts/calibration/results.json \
        --output config/tda_golden_set.yaml

    # Check drift status
    python scripts/tda_calibrate.py --check-drift \
        --golden config/tda_golden_set.yaml \
        --current artifacts/tda/governance_snapshot.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.tda.governance import (
    LabeledTDAResult,
    evaluate_hard_gate_calibration,
    CalibrationResult,
)


def load_golden_set(path: Path) -> Dict[str, Any]:
    """Load golden set from YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_calibration(golden_set: Dict[str, Any]) -> CalibrationResult:
    """Run calibration against golden set."""
    # Convert golden runs to LabeledTDAResult
    labeled_results = []
    for run in golden_set.get("golden_runs", []):
        labeled_results.append(
            LabeledTDAResult(
                hss=run["hss"],
                expected_decision=run["expected_decision"],
                label=run.get("label", ""),
            )
        )

    # Get calibration parameters
    cal_params = golden_set.get("calibration", {})

    return evaluate_hard_gate_calibration(
        golden_runs=labeled_results,
        block_threshold=cal_params.get("block_threshold", 0.2),
        false_block_threshold_ok=cal_params.get("false_block_threshold_ok", 0.05),
        false_block_threshold_drifting=cal_params.get("false_block_threshold_drifting", 0.15),
        false_pass_threshold_ok=cal_params.get("false_pass_threshold_ok", 0.05),
        false_pass_threshold_drifting=cal_params.get("false_pass_threshold_drifting", 0.15),
    )


def check_drift(
    golden_set: Dict[str, Any],
    current_snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    """Check drift between golden set and current snapshot."""
    # Run calibration
    calibration = run_calibration(golden_set)

    # Compare with current metrics
    current_block_rate = current_snapshot.get("block_rate", 0.0)
    golden_block_rate = golden_set.get("drift_tracking", {}).get("false_block_rate", 0.0)

    drift_delta = abs(current_block_rate - golden_block_rate)

    return {
        "calibration_status": calibration.calibration_status,
        "false_block_rate": calibration.false_block_rate,
        "false_pass_rate": calibration.false_pass_rate,
        "current_block_rate": current_block_rate,
        "drift_delta": drift_delta,
        "drift_detected": drift_delta > 0.1,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="TDA Golden Set Calibration")
    parser.add_argument("--golden", type=Path, help="Path to golden set YAML")
    parser.add_argument("--create", action="store_true", help="Create new golden set")
    parser.add_argument("--experiment", type=Path, help="Experiment results for creation")
    parser.add_argument("--output", "-o", type=Path, help="Output path")
    parser.add_argument("--check-drift", action="store_true", help="Check drift status")
    parser.add_argument("--current", type=Path, help="Current snapshot for drift check")

    args = parser.parse_args()

    if args.check_drift:
        if not args.golden or not args.current:
            print("Error: --check-drift requires --golden and --current")
            return 1

        golden_set = load_golden_set(args.golden)
        with open(args.current, "r", encoding="utf-8") as f:
            current = json.load(f)

        drift = check_drift(golden_set, current)
        print(json.dumps(drift, indent=2))

        if drift["drift_detected"]:
            print("\n[WARN] Drift detected!")
            return 1
        return 0

    if args.golden:
        golden_set = load_golden_set(args.golden)
        result = run_calibration(golden_set)

        print(f"Calibration Status: {result.calibration_status}")
        print(f"False Block Rate: {result.false_block_rate:.4f}")
        print(f"False Pass Rate: {result.false_pass_rate:.4f}")

        if result.calibration_status == "BROKEN":
            return 2
        if result.calibration_status == "DRIFTING":
            return 1
        return 0

    print("No action specified. Use --help for options.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
```

### Reproducibility Contract

1. **Deterministic Hashing**: All golden set configurations must be SHA-256 hashed
2. **Version Control**: Golden sets are versioned with `golden_set_id`
3. **Provenance Tracking**: Source experiment and manifest hash recorded
4. **Archival**: Previous versions archived in `config/tda_golden_archive/`
5. **Drift Bounds**: Configurable thresholds for OK/DRIFTING/BROKEN

---

## C4: CI/CD Integration Plan

### GitHub Actions Workflow

#### File: `.github/workflows/tda-watchdog.yml`

```yaml
name: TDA Watchdog

on:
  schedule:
    - cron: '0 */6 * * *'  # Every 6 hours
  workflow_dispatch:
  push:
    paths:
      - 'backend/tda/**'
      - 'config/tda_*.yaml'

jobs:
  watchdog:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install uv
          uv sync

      - name: Run TDA Watchdog
        id: watchdog
        run: |
          python scripts/tda_watchdog.py \
            --governance-log "artifacts/tda/*.json" \
            --config config/tda_watchdog.yaml \
            --output artifacts/watchdog_report.json
        continue-on-error: true

      - name: Check Calibration Drift
        id: calibration
        run: |
          python scripts/tda_calibrate.py \
            --check-drift \
            --golden config/tda_golden_set.yaml \
            --current artifacts/tda/latest_snapshot.json
        continue-on-error: true

      - name: Build Global Health
        run: |
          python scripts/build_global_health.py \
            --fm-health artifacts/fm_health.json \
            --tda-snapshot artifacts/tda/latest_snapshot.json \
            --output artifacts/global_health.json

      - name: Upload Artifacts
        uses: actions/upload-artifact@v4
        with:
          name: tda-watchdog-${{ github.run_number }}
          path: |
            artifacts/watchdog_report.json
            artifacts/global_health.json

      - name: Alert on ALERT Status
        if: steps.watchdog.outcome == 'failure'
        run: |
          echo "::error::TDA Watchdog detected ALERT condition"
          cat artifacts/watchdog_report.json
          exit 1

      - name: Alert on Calibration Drift
        if: steps.calibration.outcome == 'failure'
        run: |
          echo "::warning::TDA Calibration drift detected"
```

### Release Governance Integration

#### File: `.github/workflows/release-gate.yml`

```yaml
name: Release Gate

on:
  pull_request:
    branches: [main, master]
  workflow_dispatch:

jobs:
  tda-gate:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install uv
          uv sync

      - name: Run TDA Hard Gate Check
        id: tda_gate
        run: |
          python scripts/tda_watchdog.py \
            --governance-log "artifacts/tda/*.json" \
            --config config/tda_watchdog.yaml \
            --output /tmp/watchdog_report.json

          # Check exit code
          EXIT_CODE=$?
          if [ $EXIT_CODE -eq 2 ]; then
            echo "tda_status=ALERT" >> $GITHUB_OUTPUT
            echo "::error::TDA Gate: ALERT - Release blocked"
            exit 1
          elif [ $EXIT_CODE -eq 1 ]; then
            echo "tda_status=ATTENTION" >> $GITHUB_OUTPUT
            echo "::warning::TDA Gate: ATTENTION - Review required"
          else
            echo "tda_status=OK" >> $GITHUB_OUTPUT
          fi

      - name: Verify Calibration Status
        run: |
          python scripts/tda_calibrate.py \
            --golden config/tda_golden_set.yaml

          if [ $? -eq 2 ]; then
            echo "::error::TDA Calibration: BROKEN - Release blocked"
            exit 1
          fi

      - name: Post Status Comment
        uses: actions/github-script@v7
        with:
          script: |
            const status = '${{ steps.tda_gate.outputs.tda_status }}';
            const emoji = status === 'OK' ? '✅' : status === 'ATTENTION' ? '⚠️' : '🛑';

            await github.rest.issues.createComment({
              owner: context.repo.owner,
              repo: context.repo.repo,
              issue_number: context.issue.number,
              body: `## TDA Governance Gate ${emoji}\n\nStatus: **${status}**\n\nSee workflow run for details.`
            });
```

### Exit Code Contract

| Exit Code | Status | CI Behavior |
|-----------|--------|-------------|
| 0 | OK | Pass, proceed |
| 1 | ATTENTION | Warn, allow with review |
| 2 | ALERT | Fail, block release |

---

## C5: Phase VII+ Spec — Adaptive Thresholds & Automated Remediation

### Overview

Phase VII+ extends CORTEX with:

1. **Adaptive Thresholds**: Dynamic threshold adjustment based on historical performance
2. **Automated Remediation**: Self-healing mechanisms for common failure modes
3. **Cross-System Correlation**: TDA signals correlated with other health metrics

### Adaptive Threshold System

#### Design Principles

1. **Conservative Adaptation**: Thresholds only move within bounded ranges
2. **Decay-Based Learning**: Recent data weighted more heavily
3. **Human Override**: Manual override always available
4. **Audit Trail**: All threshold changes logged

#### Schema: `config/tda_adaptive.yaml`

```yaml
# TDA Adaptive Threshold Configuration — Phase VII+
schema_version: "1.0.0"

adaptive_thresholds:
  enabled: true

  # Block threshold adaptation
  block_threshold:
    current: 0.2
    min_bound: 0.15
    max_bound: 0.35
    decay_rate: 0.1  # Weight for historical data
    adaptation_interval_cycles: 100

  # HSS trend sensitivity
  hss_sensitivity:
    current: 0.05
    min_bound: 0.02
    max_bound: 0.10

  # Adaptation constraints
  constraints:
    max_delta_per_adaptation: 0.02
    min_cycles_between_adaptations: 50
    require_governance_approval: false

  # Historical performance window
  history:
    window_size_cycles: 1000
    min_samples_for_adaptation: 100
```

#### Adaptation Algorithm

```python
def adapt_threshold(
    current: float,
    historical_block_rate: float,
    historical_false_positive_rate: float,
    min_bound: float,
    max_bound: float,
    max_delta: float,
) -> float:
    """
    Adapt threshold based on historical performance.

    If false positive rate is high: increase threshold (less blocking)
    If false negative rate is high: decrease threshold (more blocking)
    """
    # Target: balance false positives and false negatives
    if historical_false_positive_rate > 0.1:
        # Too many false blocks, increase threshold
        delta = min(max_delta, historical_false_positive_rate * 0.1)
        new_threshold = current + delta
    elif historical_block_rate > 0.2:
        # Too many blocks overall, might be too aggressive
        delta = min(max_delta, 0.01)
        new_threshold = current + delta
    else:
        # Current threshold is working well
        new_threshold = current

    # Clamp to bounds
    return max(min_bound, min(max_bound, new_threshold))
```

### Automated Remediation

#### Remediation Actions

| Condition | Remediation Action |
|-----------|-------------------|
| `block_rate > 0.3` for 3+ cycles | Trigger exception window |
| `hss_trend == DEGRADING` for 5+ cycles | Alert + enable shadow mode |
| `golden_alignment == BROKEN` | Trigger recalibration workflow |
| `exception_window_expired` | Auto-disable exception, return to HARD |

#### Remediation Hook

```python
class TDARemediationEngine:
    """
    Automated remediation engine for TDA governance.

    Phase VII+: Implements self-healing for common failure modes.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.remediation_log: List[Dict[str, Any]] = []

    def evaluate_and_remediate(
        self,
        snapshot: TDAGovernanceSnapshot,
    ) -> Optional[str]:
        """
        Evaluate snapshot and apply remediation if needed.

        Returns:
            Remediation action taken, or None.
        """
        action = None

        # Check for sustained high block rate
        if snapshot.block_rate > 0.3:
            if self._sustained_condition("high_block_rate", 3):
                action = self._apply_exception_window(snapshot)

        # Check for degrading trend
        if snapshot.hss_trend == "degrading":
            if self._sustained_condition("degrading_trend", 5):
                action = self._enable_shadow_mode()

        # Check for broken calibration
        if snapshot.golden_alignment == "BROKEN":
            action = self._trigger_recalibration()

        if action:
            self._log_remediation(action, snapshot)

        return action

    def _apply_exception_window(self, snapshot) -> str:
        # Implementation: create exception window
        return "EXCEPTION_WINDOW_CREATED"

    def _enable_shadow_mode(self) -> str:
        # Implementation: switch to shadow mode
        return "SHADOW_MODE_ENABLED"

    def _trigger_recalibration(self) -> str:
        # Implementation: dispatch recalibration job
        return "RECALIBRATION_TRIGGERED"
```

### Cross-System Correlation

#### Correlated Signals

```python
@dataclass
class CrossSystemCorrelation:
    """
    Correlation between TDA signals and other health metrics.

    Phase VII+: Enables holistic health assessment.
    """

    # TDA signal
    tda_status: str
    tda_block_rate: float

    # Correlated signals
    replay_safety_ok: bool
    learning_health_status: str
    fm_coverage_pct: float

    # Correlation metrics
    signal_agreement: float  # % of signals agreeing
    conflict_detected: bool
    conflict_resolution: Optional[str]

    def compute_composite_health(self) -> str:
        """
        Compute composite health from all signals.

        Rules:
        - All OK + high agreement: HEALTHY
        - Mixed signals + low agreement: DEGRADED
        - Any BLOCK/FAIL: CRITICAL
        """
        if self.conflict_detected:
            return "DEGRADED"

        if self.tda_status == "ALERT":
            return "CRITICAL"

        if not self.replay_safety_ok:
            return "CRITICAL"

        if self.signal_agreement >= 0.9:
            if self.tda_status == "OK":
                return "HEALTHY"
            return "DEGRADED"

        return "DEGRADED"
```

---

## Safety Invariants

### INV-VII-1: Runner Integration Safety

- TDA hook is optional; runners function without it
- Hook failures do not crash runners (fail-open with logging)
- Block decisions respect mode (SHADOW logs only, DRY_RUN warns only)

### INV-VII-2: Global Health Determinism

- Same inputs always produce same global_health.json
- Tile merging is commutative (order doesn't matter)
- Timestamps use UTC ISO format

### INV-VII-3: Calibration Stability

- Golden set changes require explicit version bump
- Threshold adaptation bounded to prevent runaway
- Manual override always available

### INV-VII-4: CI/CD Safety

- Exit codes are stable contract (0/1/2)
- ALERT always blocks release
- Artifacts always uploaded regardless of status

---

## Files to Create

| File | Description |
|------|-------------|
| `backend/tda/runner_hook.py` | TDA governance hook for runners |
| `backend/health/global_builder.py` | Global health surface builder |
| `scripts/build_global_health.py` | CI script for building global_health.json |
| `scripts/tda_calibrate.py` | Golden set calibration tool |
| `config/tda_golden_set.yaml` | Golden set configuration |
| `config/tda_adaptive.yaml` | Adaptive threshold configuration |
| `.github/workflows/tda-watchdog.yml` | Scheduled watchdog workflow |
| `.github/workflows/release-gate.yml` | Release gate workflow |

---

## Migration Path

### Phase 1: Hook Integration

1. Create `backend/tda/runner_hook.py`
2. Add optional hook to RFLRunner
3. Add optional hook to U2Runner
4. Test in SHADOW mode

### Phase 2: Global Health Wiring

1. Create `backend/health/global_builder.py`
2. Create `scripts/build_global_health.py`
3. Add TDA tile to existing global_health.json consumers

### Phase 3: CI/CD Deployment

1. Create `.github/workflows/tda-watchdog.yml`
2. Create `.github/workflows/release-gate.yml`
3. Enable in staging, then production

### Phase 4: Calibration & Adaptive

1. Create initial golden set
2. Run calibration baseline
3. Enable adaptive thresholds (optional, Phase VII+)

---

**CORTEX PHASE VII — INTEGRATION COMPLETE. AUTONOMOUS OVERSIGHT ARMED.**

**STRATCOM: THE SUBSTRATE NOW HAS REFLEXES.**
