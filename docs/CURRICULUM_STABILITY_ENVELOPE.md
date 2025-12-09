# Curriculum Stability Envelope

**Status:** Implemented  
**Version:** 1.0  
**Integration Point:** First Light - Cortex Calibration

## Overview

The Curriculum Stability Envelope prevents invalid curriculum slice transitions during integration by tracking TDA-driven HSS (Homological Spanning Set) metrics. It ensures curriculum topology remains stable while the Cortex evaluates gate decisions, preventing drift during First Light calibration.

## Purpose

During First Light integration:
- The **Cortex** evaluates topology distributions via `evaluate_hard_gate_decision()`
- The **Curriculum** defines the topology distribution via slice parameters
- **Both must remain stable** during calibration

The Stability Envelope enforces this invariant by:
1. Detecting HSS variance spikes that indicate topology instability
2. Flagging slices with repeated low-HSS regions (unsuitable for uplift)
3. Computing slice suitability scores for experiment selection
4. Blocking slice transitions when instability is detected

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    U2Runner / RFLRunner                      │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │          Cycle Execution (per run)                   │   │
│  │                                                      │   │
│  │  1. Execute cycle                                    │   │
│  │  2. Collect metrics (HSS, verified count)           │   │
│  │  3. Record to Stability Envelope                    │   │
│  └──────────────────────────────────────────────────────┘   │
│                           │                                  │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │       CurriculumStabilityEnvelope                    │   │
│  │                                                      │   │
│  │  • Track HSS history per slice                      │   │
│  │  • Compute variance statistics                      │   │
│  │  • Detect variance spikes                           │   │
│  │  • Calculate suitability scores                     │   │
│  └──────────────────────────────────────────────────────┘   │
│                           │                                  │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │          Before Slice Transition                     │   │
│  │                                                      │   │
│  │  1. Evaluate all standard gates                     │   │
│  │     (coverage, abstention, velocity, caps)          │   │
│  │  2. Evaluate stability gate                         │   │
│  │  3. Block transition if unstable                    │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. HSS Metrics Tracking

```python
from curriculum.stability_envelope import CurriculumStabilityEnvelope

envelope = CurriculumStabilityEnvelope()

# Record after each cycle
envelope.record_cycle(
    cycle_id="cycle_001",
    slice_name="slice_uplift_goal",
    hss_value=0.75,           # HSS score (0.0-1.0)
    verified_count=10,         # Number of verified proofs
    timestamp="2025-01-01T00:00:00Z"
)
```

### 2. Stability Evaluation

```python
# Compute stability metrics for a slice
stability = envelope.compute_slice_stability("slice_uplift_goal")

print(f"Stable: {stability.is_stable}")
print(f"Suitability Score: {stability.suitability_score:.3f}")
print(f"CV: {stability.hss_cv:.4f}")
print(f"Flags: {stability.flags}")
```

**Output:**
```
Stable: True
Suitability Score: 0.887
CV: 0.0215
Flags: []
```

### 3. Variance Spike Detection

```python
# Detect if recent HSS variance spiked
spike_detected, current_var = envelope.detect_variance_spike(
    "slice_uplift_goal",
    window_size=10
)

if spike_detected:
    print(f"⚠️  HSS variance spike detected!")
    print(f"   Current: {current_var:.6f}")
    print(f"   Baseline: {envelope.baseline_variance['slice_uplift_goal']:.6f}")
```

### 4. Slice Transition Control

```python
# Check if transition is allowed
allowed, reason, details = envelope.check_slice_transition_allowed(
    from_slice="slice_uplift_goal",
    to_slice="slice_uplift_sparse"
)

if not allowed:
    print(f"❌ Transition blocked: {reason}")
else:
    print(f"✅ Transition allowed: {reason}")
```

## Integration with Runners

### U2Runner Integration

```python
from experiments.u2.runner import U2Runner, U2Config
from curriculum.stability_envelope import CurriculumStabilityEnvelope
from curriculum.stability_integration import record_cycle_hss_metrics

# Initialize runner and envelope
config = U2Config(...)
runner = U2Runner(config)
envelope = CurriculumStabilityEnvelope()

# Execute cycles and record HSS
for cycle in range(config.total_cycles):
    result = runner.run_cycle(cycle, execute_fn)
    
    # Extract HSS from result metrics
    record_cycle_hss_metrics(
        envelope=envelope,
        cycle_id=f"cycle_{cycle:03d}",
        slice_name=config.slice_name,
        metrics=result.metadata,
        timestamp=result.timestamp
    )
    
    # Check stability periodically
    if (cycle + 1) % 10 == 0:
        stability = envelope.compute_slice_stability(config.slice_name)
        if not stability.is_stable:
            print(f"⚠️  Slice instability detected at cycle {cycle}")
```

### RFLRunner Integration

```python
from rfl.runner import RFLRunner, RFLConfig
from curriculum.stability_envelope import CurriculumStabilityEnvelope
from curriculum.stability_integration import should_ratchet_with_stability

runner = RFLRunner(config)
envelope = CurriculumStabilityEnvelope()

# During run_with_attestation()
result = runner.run_with_attestation(attestation)

# Record HSS metrics
record_cycle_hss_metrics(
    envelope=envelope,
    cycle_id=result.step_id,
    slice_name=attestation.slice_id,
    metrics=attestation.metadata,
    timestamp=attestation.metadata.get("timestamp")
)

# Before slice ratchet
slice_cfg = runner._resolve_slice(attestation.slice_id)
verdict = should_ratchet_with_stability(
    metrics=attestation.metadata,
    slice_cfg=slice_cfg,
    envelope=envelope
)

if not verdict.advance:
    print(f"❌ Ratchet blocked by stability: {verdict.reason}")
```

## Configuration

### Stability Envelope Config

```python
from curriculum.stability_envelope import StabilityEnvelopeConfig

config = StabilityEnvelopeConfig(
    # Variance thresholds
    max_hss_cv=0.25,                    # Max coefficient of variation
    min_hss_threshold=0.3,               # Minimum HSS considered "low"
    max_low_hss_ratio=0.2,               # Max 20% low-HSS cycles
    
    # Suitability scoring weights
    weight_mean=0.4,                     # 40% weight on mean HSS
    weight_stability=0.3,                # 30% weight on low CV
    weight_consistency=0.3,              # 30% weight on few low-HSS
    
    # Detection parameters
    min_cycles_for_stability=5,          # Need 5+ cycles for evaluation
    variance_spike_threshold=2.0,        # 2x baseline variance = spike
)
```

### Stability Gate Config

```python
from curriculum.stability_integration import StabilityGateSpec

gate_spec = StabilityGateSpec(
    enabled=True,                        # Enable stability gate
    min_suitability_score=0.6,           # Minimum score to pass
    allow_variance_spikes=False,         # Block on variance spikes
    require_stable_slice=True,           # Slice must be stable
)
```

## Stability Metrics

### Suitability Score

The suitability score (0.0-1.0) is a weighted combination of:

```
score = 0.4 × mean_component + 0.3 × stability_component + 0.3 × consistency_component

where:
  mean_component = normalized mean HSS (higher is better)
  stability_component = 1.0 - (CV / max_CV)  (lower CV is better)
  consistency_component = 1.0 - (low_ratio / max_ratio)  (fewer low-HSS is better)
```

**Interpretation:**
- **0.8-1.0:** Excellent - highly suitable for uplift experiments
- **0.6-0.8:** Good - suitable with monitoring
- **0.4-0.6:** Marginal - requires improvement before uplift
- **0.0-0.4:** Poor - unsuitable for uplift evaluation

### Stability Flags

| Flag | Meaning | Action |
|------|---------|--------|
| `insufficient_data` | < 5 cycles recorded | Wait for more data |
| `insufficient_cycles` | < min_cycles_for_stability | Continue collecting |
| `high_variance` | CV > max_hss_cv | Investigate slice parameters |
| `repeated_low_hss` | > max_low_hss_ratio low cycles | Adjust slice difficulty |

### Variance Spike Detection

A variance spike is detected when:
```
recent_variance > baseline_variance × spike_threshold
```

Default spike threshold is 2.0x, meaning recent variance must be at least double the baseline to trigger.

## Slice Transition Decision Tree

```
                    ┌─────────────────────┐
                    │  Check Transition   │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │ Sufficient Data?    │
                    │  (≥5 cycles)        │
                    └──────────┬──────────┘
                               │
                        ┌──────┴──────┐
                        │             │
                       NO            YES
                        │             │
                        │      ┌──────▼──────────┐
                        │      │ HSS CV Stable?  │
                        │      │  (CV < 0.25)    │
                        │      └──────┬──────────┘
                        │             │
                        │      ┌──────┴──────┐
                        │      │             │
                        │     NO            YES
                        │      │             │
                        │      │      ┌──────▼──────────┐
                        │      │      │ Low-HSS Ratio?  │
                        │      │      │  (< 0.2)        │
                        │      │      └──────┬──────────┘
                        │      │             │
                        │      │      ┌──────┴──────┐
                        │      │      │             │
                        │      │     NO            YES
                        │      │      │             │
                        │      │      │      ┌──────▼──────────┐
                        │      │      │      │ Variance Spike? │
                        │      │      │      └──────┬──────────┘
                        │      │      │             │
                        │      │      │      ┌──────┴──────┐
                        │      │      │      │             │
                        │      │      │     YES           NO
                        │      │      │      │             │
                        ▼      ▼      ▼      ▼             ▼
                    ┌──────────────────────────────────────────┐
                    │           BLOCK TRANSITION               │
                    └──────────────────────────────────────────┘
                                                               │
                                                               ▼
                                                    ┌──────────────────┐
                                                    │ ALLOW TRANSITION │
                                                    └──────────────────┘
```

## HSS Extraction

The system automatically extracts HSS values from metrics dictionaries:

```python
from curriculum.stability_integration import record_cycle_hss_metrics

# Metrics can come from various sources
metrics = {
    "metrics": {
        "tda": {"hss": 0.82},              # Priority 1: Direct TDA
        "topology": {"diversity": 0.78},    # Priority 2: Topology
        "rfl": {
            "coverage": {"ci_lower": 0.75}  # Priority 3: Coverage proxy
        },
        "proofs": {"verified": 15}
    }
}

# Automatically extracts HSS value (tries tda.hss first, falls back)
record_cycle_hss_metrics(envelope, "cycle_001", "slice_a", metrics, timestamp)
```

## Testing

Run the comprehensive test suite:

```bash
python -m pytest tests/test_curriculum_stability_envelope.py -v
```

Or run the interactive demonstration:

```bash
python examples/stability_envelope_demo.py
```

The demo shows:
1. Basic HSS tracking
2. Variance spike detection
3. Slice transition control
4. Gate integration
5. Suitability scoring
6. Full report export

## Production Usage

### First Light Integration

During First Light calibration:

```python
# 1. Initialize envelope at start
envelope = CurriculumStabilityEnvelope()

# 2. Record HSS after each Δp cycle
for cycle in range(total_cycles):
    # Execute U2Runner cycle
    result = u2_runner.run_cycle(cycle, evaluate_hard_gate_decision)
    
    # Record HSS to envelope
    record_cycle_hss_metrics(
        envelope=envelope,
        cycle_id=f"fo_cycle_{cycle:03d}",
        slice_name=current_slice.name,
        metrics=result.metrics,
        timestamp=result.timestamp
    )

# 3. Before any slice transition, check stability
allowed, reason, details = envelope.check_slice_transition_allowed(
    from_slice=current_slice.name,
    to_slice=next_slice.name
)

if not allowed:
    log.error(f"Slice transition blocked: {reason}")
    # Keep current slice, continue calibration
else:
    log.info(f"Slice transition allowed: {reason}")
    # Proceed with ratchet
```

### Continuous Monitoring

Export stability report periodically:

```python
# Generate report
report = envelope.export_stability_report()

# Save to file
import json
with open("stability_report.json", "w") as f:
    json.dump(report, f, indent=2)

# Check all slice suitability
scores = envelope.get_all_slice_suitability()
for slice_name, score in scores.items():
    if score < 0.6:
        print(f"⚠️  {slice_name}: suitability score {score:.3f} below threshold")
```

## Failure Modes

### 1. Insufficient Data

**Symptom:** `insufficient_data` flag  
**Cause:** Fewer than 5 cycles recorded  
**Action:** Continue collecting data, defer stability checks

### 2. High Variance

**Symptom:** `high_variance` flag, CV > 0.25  
**Cause:** Slice parameters cause inconsistent HSS  
**Action:** Review slice difficulty parameters, consider tightening bounds

### 3. Repeated Low-HSS

**Symptom:** `repeated_low_hss` flag  
**Cause:** Slice too difficult or inappropriate for current stage  
**Action:** Adjust slice parameters, verify monotonicity

### 4. Variance Spike

**Symptom:** `spike_detected=True`  
**Cause:** Recent cycles show sudden HSS instability  
**Action:** Investigate recent changes, block transitions until resolved

## Future Extensions

1. **Multi-dimensional HSS:** Track multiple topology metrics (Betti numbers, persistence)
2. **Adaptive Thresholds:** Learn optimal thresholds from historical data
3. **Predictive Stability:** Forecast instability before it occurs
4. **Slice Recommendations:** Suggest optimal next slice based on stability trends

## References

- **TDA Integration:** Topological Data Analysis drives HSS computation
- **Curriculum Gates:** `curriculum/gates.py` - Standard gate system
- **U2Runner:** `experiments/u2/runner.py` - Uplift experiment runner
- **RFLRunner:** `rfl/runner.py` - RFL orchestration runner

## See Also

- [Phase II RFL Uplift Plan](PHASE2_RFL_UPLIFT_PLAN.md)
- [Curriculum Slice Definitions](../config/curriculum_uplift_phase2.yaml)
- [First Organism Integration](../docs/FIRST_ORGANISM_INTEGRATION.md)
