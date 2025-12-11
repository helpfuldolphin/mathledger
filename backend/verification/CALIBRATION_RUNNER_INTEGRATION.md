# Calibration Runner Integration — REAL-READY

**Author**: Manus-C (Telemetry Architect)  
**Date**: 2025-12-06  
**Status**: REAL-READY (Aligned with actual repository structure)

---

## Overview

This document provides **REAL-READY** integration for the calibration runner, aligned with the existing `backend/verification/calibration/calibrate_noise.py` structure.

---

## Current Status

### Existing Files (REAL)
- `backend/verification/calibration/calibrate_noise.py` — CLI exists (DEMO-SCAFFOLD)
- `backend/verification/calibration/statistical_fitting.py` — Fitting functions exist (DEMO-SCAFFOLD)
- `backend/verification/calibration/__init__.py` — Package init exists

### Issues with Existing Code
1. **Import Error**: References `backend.verification.telemetry` which doesn't exist
2. **Should Reference**: `backend.verification.telemetry_runtime` (REAL-READY from previous section)
3. **Missing**: Actual module list for calibration
4. **Missing**: Integration with real Lean executor

---

## Integration Fixes (REAL-READY)

### 1. Fix Imports in `calibrate_noise.py`

**File**: `backend/verification/calibration/calibrate_noise.py`

**Line 26 (CURRENT — BROKEN)**:
```python
from backend.verification.telemetry import run_lean_with_monitoring
```

**Line 26 (REAL-READY FIX)**:
```python
# REAL-READY
from backend.verification.telemetry_runtime import run_lean_with_monitoring, LeanVerificationTelemetry
from backend.verification.lean_executor import construct_lean_command
```

---

### 2. Add Worker Function (REAL-READY)

**File**: `backend/verification/calibration/calibrate_noise.py`

**Location**: After line 100 (after `load_modules()`)

```python
# REAL-READY
def calibrate_single_module(args: tuple) -> Dict[str, Any]:
    """
    Calibrate a single module (worker function for multiprocessing).
    
    Args:
        args: (module_name, tier, timeout_s, working_dir, seed)
    
    Returns:
        Dict with telemetry data
    """
    module_name, tier, timeout_s, working_dir, seed = args
    
    # Construct Lean command
    # NOTE: This assumes modules are in Mathlib format
    # Adjust path construction based on actual module structure
    module_path = Path(working_dir) / f"{module_name.replace('.', '/')}.lean"
    
    if not module_path.exists():
        # Module not found: return error
        return {
            "module_name": module_name,
            "tier": tier,
            "outcome": "module_not_found",
            "success": False,
            "duration_ms": 0.0,
        }
    
    lean_command = construct_lean_command(
        module_path=module_path,
        timeout_s=timeout_s,
        use_lake=True,
        trace_tactics=True,
    )
    
    # Run with monitoring
    telemetry = run_lean_with_monitoring(
        module_name=module_name,
        lean_command=lean_command,
        timeout_s=timeout_s,
        working_dir=working_dir,
        verification_id=f"calibration_{tier}_{module_name}_{seed}",
        context=f"calibration_{tier}",
        tier=tier.lower(),
        master_seed=seed,
        noise_config=None,  # No noise injection during calibration
    )
    
    # Return telemetry as dict
    return telemetry.to_dict()


def calibrate_tier(
    tier: str,
    modules: List[str],
    n_samples: int,
    timeout_s: float,
    working_dir: Path,
    seed: int,
    workers: int,
) -> List[Dict[str, Any]]:
    """
    Calibrate a single tier.
    
    Args:
        tier: Tier name (FAST, BALANCED, SLOW)
        modules: List of module names
        n_samples: Number of samples to collect
        timeout_s: Timeout in seconds
        working_dir: Working directory for Lean
        seed: Random seed
        workers: Number of worker processes
    
    Returns:
        List of telemetry dicts
    """
    
    print(f"\n{'='*60}")
    print(f"Calibrating {tier} tier")
    print(f"  Samples: {n_samples}")
    print(f"  Timeout: {timeout_s}s")
    print(f"  Workers: {workers}")
    print(f"{'='*60}\n")
    
    # Sample modules (with replacement if needed)
    import random
    random.seed(seed)
    sampled_modules = random.choices(modules, k=n_samples)
    
    # Prepare worker args
    worker_args = [
        (module, tier, timeout_s, working_dir, seed + i)
        for i, module in enumerate(sampled_modules)
    ]
    
    # Run in parallel
    results = []
    with Pool(processes=workers) as pool:
        for i, result in enumerate(pool.imap_unordered(calibrate_single_module, worker_args)):
            results.append(result)
            
            # Progress update every 100 samples
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i + 1}/{n_samples} samples")
    
    print(f"\n  Completed: {len(results)}/{n_samples} samples")
    
    return results
```

---

### 3. Add Main Calibration Logic (REAL-READY)

**File**: `backend/verification/calibration/calibrate_noise.py`

**Location**: After worker functions (after line ~200)

```python
# REAL-READY
def main():
    """Main calibration entry point."""
    args = parse_args()
    
    # Load modules
    if args.modules:
        modules = load_modules(args.modules)
    else:
        # Default: use a small set of test modules
        print("WARNING: No modules file specified, using default test modules")
        modules = [
            "Mathlib.Algebra.Ring.Basic",
            "Mathlib.Data.Nat.Basic",
            "Mathlib.Init.Data.Nat.Lemmas",
        ]
    
    print(f"Loaded {len(modules)} modules")
    
    # Tier configuration
    tier_config = {
        "FAST": {"timeout_s": 30.0},
        "BALANCED": {"timeout_s": 60.0},
        "SLOW": {"timeout_s": 120.0},
    }
    
    # Working directory (assume Mathlib root)
    working_dir = Path.cwd()
    
    # Calibrate each tier
    all_results = {}
    for tier in args.tiers:
        timeout_s = tier_config[tier]["timeout_s"]
        
        results = calibrate_tier(
            tier=tier,
            modules=modules,
            n_samples=args.n,
            timeout_s=timeout_s,
            working_dir=working_dir,
            seed=args.seed,
            workers=args.workers,
        )
        
        all_results[tier] = results
    
    # Fit statistical models
    print(f"\n{'='*60}")
    print("Fitting statistical models")
    print(f"{'='*60}\n")
    
    calibrated_models = {}
    for tier, results in all_results.items():
        print(f"\nTier: {tier}")
        
        # Count outcomes
        outcomes = [r["outcome"] for r in results]
        outcome_counts = defaultdict(int)
        for outcome in outcomes:
            outcome_counts[outcome] += 1
        
        # Calculate rates
        n_total = len(results)
        n_timeout = outcome_counts.get("verifier_timeout", 0)
        n_fail = outcome_counts.get("proof_invalid", 0)
        n_success = outcome_counts.get("verified", 0)
        
        timeout_rate = n_timeout / n_total if n_total > 0 else 0.0
        fail_rate = n_fail / n_total if n_total > 0 else 0.0
        success_rate = n_success / n_total if n_total > 0 else 0.0
        
        # Wilson confidence intervals
        timeout_ci = wilson_confidence_interval(n_timeout, n_total)
        fail_ci = wilson_confidence_interval(n_fail, n_total)
        
        print(f"  Timeout rate: {timeout_rate:.3f} (95% CI: [{timeout_ci[0]:.3f}, {timeout_ci[1]:.3f}])")
        print(f"  Fail rate: {fail_rate:.3f} (95% CI: [{fail_ci[0]:.3f}, {fail_ci[1]:.3f}])")
        print(f"  Success rate: {success_rate:.3f}")
        
        # Fit timeout distribution (if timeouts observed)
        timeout_durations = [
            r["duration_ms"] for r in results
            if r["outcome"] == "verifier_timeout"
        ]
        
        if timeout_durations:
            timeout_dist = fit_timeout_distribution(timeout_durations)
            print(f"  Timeout distribution: {timeout_dist['best_model']}")
        else:
            timeout_dist = {"best_model": "none", "params": {}}
        
        # Store calibrated model
        calibrated_models[tier] = {
            "timeout_rate": timeout_rate,
            "timeout_rate_ci": timeout_ci,
            "fail_rate": fail_rate,
            "fail_rate_ci": fail_ci,
            "success_rate": success_rate,
            "timeout_distribution": timeout_dist,
            "n_samples": n_total,
        }
    
    # Export to YAML
    if args.export and not args.dry_run:
        export_data = {
            "calibration_metadata": {
                "timestamp": time.time(),
                "tiers": args.tiers,
                "n_samples_per_tier": args.n,
                "seed": args.seed,
                "n_modules": len(modules),
            },
            "calibrated_models": calibrated_models,
        }
        
        args.export.parent.mkdir(parents=True, exist_ok=True)
        with open(args.export, 'w') as f:
            yaml.dump(export_data, f, default_flow_style=False)
        
        print(f"\n{'='*60}")
        print(f"Calibrated models exported to: {args.export}")
        print(f"{'='*60}\n")
    
    return calibrated_models


if __name__ == "__main__":
    main()
```

---

### 4. Create Module List File (REAL-READY)

**File**: `config/calibration_modules.txt` (NEW)

```txt
# REAL-READY
# Calibration module list
# Format: One Lean module name per line
# These are example Mathlib modules for calibration

Mathlib.Algebra.Ring.Basic
Mathlib.Algebra.Group.Defs
Mathlib.Data.Nat.Basic
Mathlib.Data.Int.Basic
Mathlib.Data.List.Basic
Mathlib.Init.Data.Nat.Lemmas
Mathlib.Init.Data.List.Lemmas
Mathlib.Tactic.Ring
Mathlib.Tactic.Linarith
Mathlib.Tactic.Omega

# Add more modules as needed
# For full calibration, use 100+ modules
```

---

## Smoke-Test Readiness Checklist

### Files to Edit (Diffs)

1. **`backend/verification/calibration/calibrate_noise.py`** (3 diffs)
   - Line 26: Fix import (1 line change)
   - After line 100: Add `calibrate_single_module()` (~40 lines)
   - After line 100: Add `calibrate_tier()` (~50 lines)
   - After line ~200: Add `main()` (~100 lines)

### Files to Create

2. **`config/calibration_modules.txt`** (~10-100 lines)
   - List of Lean modules for calibration
   - Start with 10 modules for smoke test
   - Expand to 100+ for full calibration

### Commands to Run (Smoke Test with 100 samples)

```bash
# 1. Navigate to repository
cd /home/ubuntu/mathledger

# 2. Create module list file
mkdir -p config
cat > config/calibration_modules_smoke.txt << 'EOF'
Mathlib.Algebra.Ring.Basic
Mathlib.Data.Nat.Basic
Mathlib.Init.Data.Nat.Lemmas
EOF

# 3. Apply import fix to calibrate_noise.py
# (Manual edit: change line 26 as specified above)

# 4. Add worker functions and main() to calibrate_noise.py
# (Manual edit: add code blocks as specified above)

# 5. Test import
python3 -c "
from backend.verification.calibration.calibrate_noise import parse_args
print('Import successful')
"

# 6. Run smoke test (100 samples, 1 tier)
python3 -m backend.verification.calibration.calibrate_noise \
    --tiers BALANCED \
    --n 100 \
    --modules config/calibration_modules_smoke.txt \
    --workers 4 \
    --seed 42 \
    --export artifacts/calibration_smoke_test.yaml

# Expected duration: 10-20 minutes (depending on Lean availability)
# Expected output: 
#   - Progress updates every 100 samples
#   - Statistical summary (timeout rate, fail rate, success rate)
#   - Calibrated model exported to artifacts/calibration_smoke_test.yaml
```

### Expected Observable Artifacts

1. **Import Test**: No import errors
2. **Progress Updates**: Printed every 100 samples
3. **Statistical Summary**: Timeout rate, fail rate, success rate with confidence intervals
4. **YAML Export**: `artifacts/calibration_smoke_test.yaml` created with calibrated model
5. **No Crashes**: Calibration completes without exceptions

---

## Full Calibration Command (10,000 samples)

```bash
# After smoke test passes, run full calibration
python3 -m backend.verification.calibration.calibrate_noise \
    --tiers FAST BALANCED SLOW \
    --n 10000 \
    --modules config/calibration_modules.txt \
    --workers 16 \
    --seed 42 \
    --export artifacts/calibration_full.yaml

# Expected duration: 56-84 hours (2-3.5 days compute time)
```

---

## Troubleshooting

### Issue: "Module not found"
**Cause**: Module path construction incorrect  
**Fix**: Adjust `module_path` construction in `calibrate_single_module()` based on actual Mathlib structure

### Issue: "Lean command failed"
**Cause**: Lean not installed or not in PATH  
**Fix**: Install Lean 4 and Lake, ensure `lake env lean --version` works

### Issue: "Import error: backend.verification.telemetry"
**Cause**: Old import still present  
**Fix**: Apply import fix (line 26) as specified above

### Issue: "Low throughput"
**Cause**: Too few workers or slow Lean execution  
**Fix**: Increase `--workers` or reduce `--n` for smoke test

---

**Status**: REAL-READY  
**Confidence**: High (fixes existing DEMO-SCAFFOLD code)  
**Risk**: Medium (depends on Lean installation and module availability)

---

**Manus-C — Telemetry Architect**  
*"Every packet accounted for, every signal explained."*
