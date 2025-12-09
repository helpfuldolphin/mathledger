# RFL Noise Integration Guide

**Author**: Manus-C (Telemetry Architect)  
**Date**: 2025-12-06  
**Status**: Ready for Integration

---

## Overview

This guide provides step-by-step instructions for integrating noise-robust RFL updates into the U2 runner and policy modules.

---

## Files Delivered

1. **`backend/verification/rfl_integration.py`** — Core RFL integration functions
2. **`backend/verification/patches/u2_runner_noise_integration.patch`** — Patch for U2 runner
3. **`backend/verification/patches/policy_noise_integration.patch`** — Patch for policy module
4. **`backend/verification/patches/INTEGRATION_GUIDE.md`** — This guide

---

## Step-by-Step Integration

### Step 1: Apply U2 Runner Patch

```bash
cd /home/ubuntu/mathledger
git apply backend/verification/patches/u2_runner_noise_integration.patch
```

This patch adds:
- Noise configuration loading in `U2Config`
- `_execute_with_noise()` method for noise injection
- Modified `run_cycle()` to use noise injection when enabled

### Step 2: Apply Policy Module Patch

```bash
git apply backend/verification/patches/policy_noise_integration.patch
```

This patch modifies:
- `SearchPolicy.update()` signature to accept continuous `value` instead of binary `success`
- `RFLPolicy.update()` to use continuous value directly

### Step 3: Create Noise Configuration File

Create `config/noise_phase3.yaml`:

```yaml
noise_models:
  balanced:
    timeout_rate: 0.1
    spurious_fail_rate: 0.05
    spurious_pass_rate: 0.02
  fast_noisy:
    timeout_rate: 0.2
    spurious_fail_rate: 0.1
    spurious_pass_rate: 0.05
  slow_precise:
    timeout_rate: 0.05
    spurious_fail_rate: 0.02
    spurious_pass_rate: 0.01
```

### Step 4: Update U2 Runner Invocation

Modify `experiments/run_uplift_u2.py` to enable noise injection:

```python
u2_config = U2Config(
    experiment_id=f"u2_{slice_name}_{mode}",
    slice_name=slice_name,
    mode=mode,
    total_cycles=cycles,
    master_seed=seed,
    # Phase III: Enable noise injection
    enable_noise_injection=True,
    noise_config_path=Path("config/noise_phase3.yaml"),
    enable_replay_log=True,
    replay_log_path=Path(f"logs/replay_{slice_name}_{mode}.jsonl"),
)
```

### Step 5: Test Integration

Run integration tests:

```bash
python -m pytest tests/verification/phase4/test_rfl_integration.py -v
```

### Step 6: Run End-to-End Experiment

```bash
python experiments/run_uplift_u2.py \
    --slice test_slice \
    --mode rfl \
    --cycles 100 \
    --seed 12345
```

---

## Verification Checklist

- [ ] U2 runner patch applied successfully
- [ ] Policy module patch applied successfully
- [ ] Noise configuration file created
- [ ] Integration tests pass
- [ ] End-to-end experiment runs without errors
- [ ] Replay log generated (if enabled)
- [ ] Policy weights updated with noise-robust values

---

## Mathematical Formulas

### Expected Value Computation

```
V_expected = 2 * P(y = VALID | o) - 1
```

where:
- **For VERIFIED**: `P(y = VALID | o = VERIFIED) = (1 - θ_sf - θ_t) / ((1 - θ_sf - θ_t) + θ_sp)`
- **For FAILED**: `P(y = VALID | o = FAILED) = θ_sf / (θ_sf + (1 - θ_sp - θ_t))`
- **For TIMEOUT**: `P(y = VALID | o = TIMEOUT) = 0.5` (no information)

### Bias Correction

```
V_corrected = V_expected / (1 - 2θ_sp)
```

### Policy Update

```
Δπ = η * V_corrected * ∇log π
```

---

## Troubleshooting

### Patch Fails to Apply

If patch fails, manually apply changes:

1. Open `experiments/u2/runner.py`
2. Add imports at top
3. Add noise config fields to `U2Config`
4. Add `_execute_with_noise()` method
5. Modify `run_cycle()` to use noise injection

### Noise Not Injected

Check:
- `enable_noise_injection=True` in `U2Config`
- Noise config file exists and is valid YAML
- Noise rates are non-zero

### Policy Not Updating

Check:
- Policy module patch applied correctly
- `update()` signature changed to accept `value` instead of `success`
- RFL mode enabled (`mode="rfl"`)

---

## Next Steps

After successful integration:

1. **Calibrate noise models** using `calibrate_noise` CLI
2. **Run validation experiments** to verify policy convergence
3. **Deploy drift radar** to monitor noise drift
4. **Enable replay logs** for deterministic reproduction

---

**Manus-C — Telemetry Architect**  
*"Every packet accounted for, every signal explained."*
