# U2 Integration Patch — Noisy Verifier Integration

**Author**: Manus-C (Telemetry Architect)  
**Date**: 2025-12-06  
**Target**: `experiments/run_uplift_u2.py`

---

## Overview

This document describes the minimal changes needed to integrate the noisy verifier
regime into the U2 uplift experiment runner (`experiments/run_uplift_u2.py`).

The integration is designed to be:
- **Minimal**: Only a few lines of code changes
- **Backward compatible**: Can be disabled via flag
- **Drop-in**: Uses existing execution function interface

---

## Changes Required

### 1. Add Import for Noisy Execution Function

**Location**: After line 37 (after `from rfl.prng import ...`)

```python
# Noisy Verifier Integration (Agent C — Manus-C)
from backend.verification.u2_integration import (
    create_noisy_execute_fn,
    outcome_to_rfl_feedback,
    should_update_rfl_policy,
    get_rfl_feedback_metadata,
)
```

### 2. Add Noise Configuration Loading

**Location**: In `run_experiment()` function, after budget loading (after line 229)

```python
# Noisy Verifier Configuration (Agent C)
# Load noise config for Phase II slices
noise_enabled = config.get("noise_enabled", False)
if is_phase2_slice(slice_name) and noise_enabled:
    print(f"INFO: Noise injection enabled for Phase II slice '{slice_name}'")
    use_noisy_verifier = True
else:
    print(f"INFO: Noise injection disabled for slice '{slice_name}'")
    use_noisy_verifier = False
```

### 3. Replace Execution Function Creation

**Location**: Replace `create_execute_fn()` call (around line 260)

**Original**:
```python
# 2. Create U2 runner with config
u2_config = U2Config(
    experiment_id=f"u2_{slice_name}_{mode}",
    slice_name=slice_name,
    mode=mode,
    total_cycles=cycles,
    master_seed=seed,
    snapshot_interval=snapshot_interval,
    snapshot_dir=snapshot_dir,
    output_dir=out_dir,
    slice_config=slice_config,
)

runner = U2Runner(u2_config)
```

**Modified**:
```python
# 2. Create execution function (with optional noise injection)
if use_noisy_verifier:
    execute_fn = create_noisy_execute_fn(
        slice_name=slice_name,
        master_seed=seed,
        noise_enabled=True,
        use_escalation=config.get("use_escalation", True),
    )
    print(f"INFO: Using noisy verifier with seed {seed}")
else:
    execute_fn = create_execute_fn(slice_name)
    print(f"INFO: Using standard verifier (no noise)")

# 3. Create U2 runner with config
u2_config = U2Config(
    experiment_id=f"u2_{slice_name}_{mode}",
    slice_name=slice_name,
    mode=mode,
    total_cycles=cycles,
    master_seed=seed,
    snapshot_interval=snapshot_interval,
    snapshot_dir=snapshot_dir,
    output_dir=out_dir,
    slice_config=slice_config,
    execute_fn=execute_fn,  # Pass execution function to runner
)

runner = U2Runner(u2_config)
```

### 4. Add Noise Telemetry to Cycle Results

**Location**: In cycle execution loop (if accessible in U2Runner)

If `U2Runner` is implemented in the same file, add telemetry after each verification:

```python
# After verification call
outcome = execute_fn(item, seed)
success, result = outcome

# Log noise telemetry if available
if "noise_injected" in result and result["noise_injected"]:
    print(f"  [NOISE] {result['noise_type']} injected on item {item}")
    print(f"          Tier: {result['tier']}, Attempts: {result['attempt_count']}")

# Determine RFL feedback
if should_update_rfl_policy(result):
    feedback = outcome_to_rfl_feedback(result)
    metadata = get_rfl_feedback_metadata(result)
    # Update RFL policy with feedback and metadata
    # (existing RFL update logic here)
```

---

## Configuration Changes

### Add to Experiment Config YAML

**File**: `config/u2_experiment_config.yaml` (or similar)

```yaml
# Noisy Verifier Configuration
noise_enabled: true
use_escalation: true

slices:
  arithmetic_simple:
    noise_enabled: true
    # ... existing slice config
  
  algebra_expansion:
    noise_enabled: true
    # ... existing slice config
```

---

## Command-Line Flag (Optional)

Add command-line argument to enable/disable noise:

```python
parser.add_argument(
    "--noise",
    action="store_true",
    help="Enable noisy verifier regime (Phase II)",
)
parser.add_argument(
    "--no-noise",
    action="store_true",
    help="Disable noisy verifier regime (default for Phase I)",
)
```

Then in `run_experiment()`:

```python
# Override noise setting from command line
if args.noise:
    noise_enabled = True
elif args.no_noise:
    noise_enabled = False
else:
    noise_enabled = config.get("noise_enabled", False)
```

---

## Testing the Integration

### Test 1: Run with Noise Disabled (Baseline)

```bash
python experiments/run_uplift_u2.py \
    --slice arithmetic_simple \
    --cycles 10 \
    --seed 42 \
    --mode rfl \
    --no-noise
```

**Expected**: Standard verifier behavior, no noise injection.

### Test 2: Run with Noise Enabled

```bash
python experiments/run_uplift_u2.py \
    --slice arithmetic_simple \
    --cycles 10 \
    --seed 42 \
    --mode rfl \
    --noise
```

**Expected**: Noise injection active, telemetry shows noise events.

### Test 3: Verify Determinism

```bash
# Run 1
python experiments/run_uplift_u2.py \
    --slice arithmetic_simple \
    --cycles 10 \
    --seed 99999 \
    --mode rfl \
    --noise \
    --out-dir /tmp/run1

# Run 2 (same seed)
python experiments/run_uplift_u2.py \
    --slice arithmetic_simple \
    --cycles 10 \
    --seed 99999 \
    --mode rfl \
    --noise \
    --out-dir /tmp/run2

# Compare results
diff /tmp/run1/results.json /tmp/run2/results.json
```

**Expected**: Identical results (deterministic noise).

---

## Rollback Plan

If issues arise, the integration can be disabled by:

1. Setting `noise_enabled: false` in config
2. Using `--no-noise` flag
3. Commenting out the import and using original `create_execute_fn()`

The original behavior is preserved and can be restored instantly.

---

## Notes for U2Runner Implementation

If `U2Runner` is not yet implemented (imports fail), the integration can still
proceed by:

1. Using the noisy execution function directly in the experiment loop
2. Manually logging outcomes and telemetry
3. Implementing a minimal U2Runner stub that accepts `execute_fn`

Example stub:

```python
class U2Runner:
    def __init__(self, config):
        self.config = config
        self.execute_fn = config.execute_fn
    
    def run_cycle(self, item, seed):
        success, result = self.execute_fn(item, seed)
        return CycleResult(success=success, result=result)
```

---

**End of Integration Patch**
