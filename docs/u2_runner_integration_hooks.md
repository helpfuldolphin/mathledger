# U2 Runner Integration Hooks

**Status**: REAL-READY Integration Plan  
**Target**: `experiments/u2/runner.py`  
**Date**: 2025-12-06  
**Engineer**: Manus-F

---

## Overview

This document specifies the exact integration hooks for connecting the U2 Bridge Layer components to the existing `experiments/u2/runner.py`.

**Components to integrate**:
1. P3 Metric Extractor (`backend/u2/p3_metric_extractor.py`)
2. Provenance Bundle v2 Generator (`backend/u2/provenance_bundle_v2.py`)
3. LeanExecutor (`backend/u2/lean_executor_hardened.py`)

---

## Current Runner Structure

**File**: `experiments/u2/runner.py`

**Key Classes**:
- `U2Config`: Experiment configuration
- `CycleResult`: Single cycle execution result
- `U2Runner`: Main execution engine

**Key Methods**:
- `U2Runner.__init__(config)`: Initialize runner
- `U2Runner.run_cycle(cycle, execute_fn)`: Execute single cycle
- `U2Runner.run_experiment(execute_fn, total_cycles)`: Run full experiment

---

## Integration Hook 1: P3 Metric Extractor

### Location
**File**: `experiments/u2/runner.py`  
**Method**: `U2Runner.run_experiment()` (after experiment completion)

### Integration Code

```python
# REAL-READY
# Add to imports at top of file
from backend.u2.p3_metric_extractor import P3MetricExtractor

# Add to U2Runner.run_experiment() after experiment loop
def run_experiment(self, execute_fn, total_cycles):
    """Run full U2 experiment."""
    # ... existing experiment loop ...
    
    # After experiment completion, extract P3 metrics
    if self.config.output_dir:
        trace_path = self.config.output_dir / "trace.jsonl"
        metrics_path = self.config.output_dir / "p3_metrics.json"
        
        if trace_path.exists():
            extractor = P3MetricExtractor()
            metrics = extractor.extract(trace_path)
            extractor.save_metrics(metrics, metrics_path)
            
            # Log metrics
            print(f"P3 Metrics:")
            print(f"  Ω (proven statements): {len(metrics.omega)}")
            print(f"  Δp (count): {metrics.delta_p}")
            print(f"  RSI (executions/s): {metrics.rsi:.2f}")
```

### Configuration Changes

Add to `U2Config`:
```python
@dataclass
class U2Config:
    # ... existing fields ...
    
    # P3 metrics
    enable_p3_metrics: bool = True  # Enable P3 metric extraction
```

### Testing

```bash
# Run experiment with P3 metrics
python experiments/run_uplift_u2.py --enable-p3-metrics

# Verify p3_metrics.json exists
cat artifacts/u2/test_exp/p3_metrics.json
```

---

## Integration Hook 2: Provenance Bundle v2 Generator

### Location
**File**: `experiments/u2/runner.py`  
**Method**: `U2Runner.run_experiment()` (after P3 metrics extraction)

### Integration Code

```python
# REAL-READY
# Add to imports at top of file
from backend.u2.provenance_bundle_v2 import (
    ProvenanceBundleV2Generator,
    SliceMetadata,
)

# Add to U2Runner.run_experiment() after P3 metrics extraction
def run_experiment(self, execute_fn, total_cycles):
    """Run full U2 experiment."""
    # ... existing experiment loop ...
    # ... P3 metrics extraction ...
    
    # After P3 metrics, generate provenance bundle
    if self.config.output_dir and self.config.enable_provenance_bundle:
        bundle_path = self.config.output_dir / "provenance_bundle_v2.json"
        
        # Create slice metadata
        slice_metadata = SliceMetadata(
            slice_name=self.config.slice_name,
            master_seed=int_to_hex_seed(self.config.master_seed),
            total_cycles=total_cycles,
            policy_config={"mode": self.config.mode},
            feature_set_version="v1.0.0",
            executor_config=self.config.slice_config.get("executor_config", {}),
            budget_config={
                "cycle_budget_s": self.config.cycle_budget_s,
                "max_candidates_per_cycle": self.config.max_candidates_per_cycle,
            },
        )
        
        # Generate bundle
        generator = ProvenanceBundleV2Generator()
        bundle = generator.generate(
            experiment_id=self.config.experiment_id,
            slice_metadata=slice_metadata,
            artifacts_dir=self.config.output_dir,
            output_path=bundle_path,
        )
        
        print(f"Provenance Bundle v2:")
        print(f"  Content Merkle Root: {bundle.bundle_header.content_merkle_root[:16]}...")
        print(f"  Metadata Hash: {bundle.bundle_header.metadata_hash[:16]}...")
```

### Configuration Changes

Add to `U2Config`:
```python
@dataclass
class U2Config:
    # ... existing fields ...
    
    # Provenance bundle
    enable_provenance_bundle: bool = True  # Enable provenance bundle generation
```

### Testing

```bash
# Run experiment with provenance bundle
python experiments/run_uplift_u2.py --enable-provenance-bundle

# Verify provenance_bundle_v2.json exists
cat artifacts/u2/test_exp/provenance_bundle_v2.json
```

---

## Integration Hook 3: LeanExecutor

### Location
**File**: `experiments/u2/runner.py`  
**Method**: `U2Runner.__init__()` (executor selection)

### Integration Code

```python
# REAL-READY
# Add to imports at top of file
from backend.u2.lean_executor_hardened import create_executor

# Add to U2Config
@dataclass
class U2Config:
    # ... existing fields ...
    
    # Executor configuration
    executor_type: str = "propositional"  # "propositional" or "lean"
    executor_timeout_s: int = 5  # Timeout for verification
    allow_executor_stub: bool = False  # Allow stub if executor not available

# Modify U2Runner.__init__() to create executor
def __init__(self, config: U2Config):
    """Initialize U2 runner."""
    self.config = config
    
    # ... existing PRNG and frontier initialization ...
    
    # Create executor
    self.executor = create_executor(
        executor_type=config.executor_type,
        timeout_seconds=config.executor_timeout_s,
        allow_stub=config.allow_executor_stub,
    )
    
    print(f"Using executor: {self.executor.__class__.__name__}")

# Modify U2Runner.run_cycle() to use self.executor
def run_cycle(self, cycle, execute_fn):
    """Execute single cycle."""
    # Replace execute_fn with self.executor.verify
    # ... existing cycle logic ...
```

### Configuration Changes

**Command-line arguments** (in `experiments/run_uplift_u2.py`):
```python
parser.add_argument("--executor", choices=["propositional", "lean"], default="propositional")
parser.add_argument("--executor-timeout", type=int, default=5)
parser.add_argument("--allow-stub", action="store_true")
```

### Testing

```bash
# Run with propositional executor (default)
python experiments/run_uplift_u2.py

# Run with Lean executor (requires Lean 4 installed)
python experiments/run_uplift_u2.py --executor lean

# Run with Lean executor stub (for testing)
export U2_LEAN_ALLOW_STUB=1
python experiments/run_uplift_u2.py --executor lean --allow-stub
```

---

## Full Integration Example

### Modified `experiments/u2/runner.py`

```python
# REAL-READY
"""
U2 Planner Runner (with Bridge Layer Integration)
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from rfl.prng import DeterministicPRNG, int_to_hex_seed

from .frontier import FrontierManager, BeamAllocator
from .logging import U2TraceLogger
from .policy import create_policy, SearchPolicy
from .schema import EventType
from .snapshots import SnapshotData, create_snapshot_name, save_snapshot

# Bridge Layer imports
from backend.u2.p3_metric_extractor import P3MetricExtractor
from backend.u2.provenance_bundle_v2 import (
    ProvenanceBundleV2Generator,
    SliceMetadata,
)
from backend.u2.lean_executor_hardened import create_executor


@dataclass
class U2Config:
    """Configuration for U2 experiment."""
    
    experiment_id: str
    slice_name: str
    mode: str  # "baseline" or "rfl"
    total_cycles: int
    master_seed: int
    
    # Optional parameters
    snapshot_interval: int = 0
    snapshot_dir: Optional[Path] = None
    output_dir: Optional[Path] = None
    slice_config: Dict[str, Any] = field(default_factory=dict)
    
    # Search parameters
    max_beam_width: int = 100
    max_depth: int = 10
    
    # Budget parameters
    cycle_budget_s: float = 60.0
    max_candidates_per_cycle: int = 100
    
    # Bridge Layer parameters
    enable_p3_metrics: bool = True
    enable_provenance_bundle: bool = True
    executor_type: str = "propositional"  # "propositional" or "lean"
    executor_timeout_s: int = 5
    allow_executor_stub: bool = False


class U2Runner:
    """U2 Planner execution engine (with Bridge Layer integration)."""
    
    def __init__(self, config: U2Config):
        """Initialize U2 runner."""
        self.config = config
        
        # Initialize PRNG hierarchy
        master_seed_hex = int_to_hex_seed(config.master_seed)
        self.master_prng = DeterministicPRNG(master_seed_hex)
        self.slice_prng = self.master_prng.for_path("slice", config.slice_name)
        
        # Initialize frontier manager
        self.frontier = FrontierManager(
            max_beam_width=config.max_beam_width,
            max_depth=config.max_depth,
            prng=self.slice_prng.for_path("frontier"),
        )
        
        # Initialize executor
        self.executor = create_executor(
            executor_type=config.executor_type,
            timeout_seconds=config.executor_timeout_s,
            allow_stub=config.allow_executor_stub,
        )
        
        print(f"Using executor: {self.executor.__class__.__name__}")
    
    def run_experiment(self, execute_fn, total_cycles):
        """Run full U2 experiment with Bridge Layer integration."""
        # ... existing experiment loop ...
        
        # After experiment completion, extract P3 metrics
        if self.config.output_dir and self.config.enable_p3_metrics:
            trace_path = self.config.output_dir / "trace.jsonl"
            metrics_path = self.config.output_dir / "p3_metrics.json"
            
            if trace_path.exists():
                extractor = P3MetricExtractor()
                metrics = extractor.extract(trace_path)
                extractor.save_metrics(metrics, metrics_path)
                
                print(f"P3 Metrics:")
                print(f"  Ω (proven statements): {len(metrics.omega)}")
                print(f"  Δp (count): {metrics.delta_p}")
                print(f"  RSI (executions/s): {metrics.rsi:.2f}")
        
        # Generate provenance bundle
        if self.config.output_dir and self.config.enable_provenance_bundle:
            bundle_path = self.config.output_dir / "provenance_bundle_v2.json"
            
            slice_metadata = SliceMetadata(
                slice_name=self.config.slice_name,
                master_seed=int_to_hex_seed(self.config.master_seed),
                total_cycles=total_cycles,
                policy_config={"mode": self.config.mode},
                feature_set_version="v1.0.0",
                executor_config={"type": self.config.executor_type},
                budget_config={
                    "cycle_budget_s": self.config.cycle_budget_s,
                    "max_candidates_per_cycle": self.config.max_candidates_per_cycle,
                },
            )
            
            generator = ProvenanceBundleV2Generator()
            bundle = generator.generate(
                experiment_id=self.config.experiment_id,
                slice_metadata=slice_metadata,
                artifacts_dir=self.config.output_dir,
                output_path=bundle_path,
            )
            
            print(f"Provenance Bundle v2:")
            print(f"  Content Merkle Root: {bundle.bundle_header.content_merkle_root[:16]}...")
            print(f"  Metadata Hash: {bundle.bundle_header.metadata_hash[:16]}...")
```

---

## Smoke-Test Integration

### Step 1: Run Experiment with All Bridge Layer Components

```bash
cd C:\dev\mathledger

# Set PYTHONPATH
$env:PYTHONPATH = "C:\dev\mathledger"

# Run experiment with all bridge layer features
python experiments/run_uplift_u2.py `
    --experiment-id test_bridge `
    --slice-name test_slice `
    --mode baseline `
    --total-cycles 10 `
    --enable-p3-metrics `
    --enable-provenance-bundle `
    --executor propositional
```

### Step 2: Verify Outputs

```bash
# Verify P3 metrics
cat artifacts/u2/test_bridge/test_slice/p3_metrics.json

# Verify provenance bundle
cat artifacts/u2/test_bridge/test_slice/provenance_bundle_v2.json

# Verify trace
cat artifacts/u2/test_bridge/test_slice/trace.jsonl
```

### Step 3: Test with Lean Executor (if Lean installed)

```bash
# Run with Lean executor
python experiments/run_uplift_u2.py `
    --experiment-id test_lean `
    --slice-name test_slice `
    --mode baseline `
    --total-cycles 10 `
    --executor lean
```

### Step 4: Test with Lean Executor Stub (for testing)

```bash
# Enable stub
$env:U2_LEAN_ALLOW_STUB = "1"

# Run with stub
python experiments/run_uplift_u2.py `
    --experiment-id test_stub `
    --slice-name test_slice `
    --mode baseline `
    --total-cycles 10 `
    --executor lean `
    --allow-stub
```

---

## Implementation Checklist

- [ ] Add Bridge Layer imports to `experiments/u2/runner.py`
- [ ] Add Bridge Layer configuration fields to `U2Config`
- [ ] Add executor initialization to `U2Runner.__init__()`
- [ ] Add P3 metrics extraction to `U2Runner.run_experiment()`
- [ ] Add provenance bundle generation to `U2Runner.run_experiment()`
- [ ] Add command-line arguments to `experiments/run_uplift_u2.py`
- [ ] Test with propositional executor
- [ ] Test with Lean executor (if installed)
- [ ] Test with Lean executor stub
- [ ] Verify P3 metrics JSON output
- [ ] Verify provenance bundle v2 JSON output

---

## Status

**Integration Plan**: ✅ REAL-READY  
**Target File**: `experiments/u2/runner.py` (confirmed to exist)  
**Dependencies**: All Bridge Layer components implemented and tested (9/9 tests passing)  
**Ready for**: Implementation in local repository at `C:\dev\mathledger`
