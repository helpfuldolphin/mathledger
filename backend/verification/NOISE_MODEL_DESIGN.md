# Imperfect Verifier Noise Model — Phase II Design

**Author**: Manus-C (Telemetry Architect)  
**Date**: 2025-12-06  
**Status**: Implementation Specification

---

## Mission

Implement deterministic, reproducible noise models for imperfect verifiers in Phase-II. All verifier failures must be:
- **Deterministic**: Identical seeds produce identical outcomes
- **Reproducible**: No stochastic behavior without explicit seeding
- **Observable**: All outcomes map to stable error codes
- **Traceable**: Full telemetry for debugging and analysis

---

## Design Principles

### 1. Deterministic Noise Generation

All noise is generated using **seeded PRNGs** from `rfl.prng.DeterministicPRNG`:
- Master seed flows hierarchically through the system
- Each verifier call derives a unique subseed from: `(master_seed, cycle_id, item_id, attempt_count)`
- No `random.random()` or unseeded randomness allowed

### 2. Verifier Failure Taxonomy

We model three classes of verifier imperfection:

| Failure Type | Error Code | Description | Reproducible? |
|--------------|-----------|-------------|---------------|
| **Timeout** | `VERIFIER_TIMEOUT` | Verifier exceeds time budget | Yes (seeded) |
| **Spurious Failure** | `VERIFIER_SPURIOUS_FAIL` | Verifier fails on valid proof | Yes (seeded) |
| **Spurious Success** | `VERIFIER_SPURIOUS_PASS` | Verifier passes invalid proof | Yes (seeded) |

### 3. Noise Model Parameters

Configurable per-slice in `config/verifier_noise_phase2.yaml`:

```yaml
slices:
  arithmetic_simple:
    noise_enabled: true
    timeout_rate: 0.05        # 5% of calls timeout
    spurious_fail_rate: 0.02  # 2% false negatives
    spurious_pass_rate: 0.01  # 1% false positives
    timeout_distribution:
      type: "uniform"         # or "exponential", "fixed"
      min_ms: 500
      max_ms: 2000
```

### 4. Mixed-Verifier Tier Routing

Support for multiple verifier tiers with different noise profiles:

```python
class VerifierTier(Enum):
    FAST_NOISY = "fast_noisy"      # High noise, low latency
    BALANCED = "balanced"           # Medium noise, medium latency
    SLOW_PRECISE = "slow_precise"  # Low noise, high latency
```

Routing policy:
- Initial attempts use `FAST_NOISY`
- On timeout/failure, escalate to `BALANCED`
- Critical proofs use `SLOW_PRECISE`

---

## Implementation Components

### Component 1: Noise Sampler (`backend/verification/noise_sampler.py`)

```python
class NoiseSampler:
    """Deterministic noise injection for verifier calls."""
    
    def __init__(self, config: NoiseConfig, seed: int):
        self.config = config
        self.prng = DeterministicPRNG(int_to_hex_seed(seed))
    
    def should_timeout(self, context: str) -> bool:
        """Deterministically decide if this call times out."""
        rng = self.prng.for_path("timeout", context)
        return rng.random() < self.config.timeout_rate
    
    def should_spurious_fail(self, context: str) -> bool:
        """Deterministically decide if valid proof fails."""
        rng = self.prng.for_path("spurious_fail", context)
        return rng.random() < self.config.spurious_fail_rate
    
    def should_spurious_pass(self, context: str) -> bool:
        """Deterministically decide if invalid proof passes."""
        rng = self.prng.for_path("spurious_pass", context)
        return rng.random() < self.config.spurious_pass_rate
    
    def sample_timeout_duration(self, context: str) -> float:
        """Sample timeout duration from configured distribution."""
        rng = self.prng.for_path("timeout_duration", context)
        if self.config.timeout_distribution.type == "uniform":
            return rng.uniform(
                self.config.timeout_distribution.min_ms,
                self.config.timeout_distribution.max_ms
            ) / 1000.0
        # ... other distributions
```

### Component 2: Verifier Error Codes (`backend/verification/error_codes.py`)

```python
class VerifierErrorCode(Enum):
    """Stable error codes for verifier outcomes."""
    
    # Success states
    VERIFIED = "VERIFIED"
    
    # Failure states
    PROOF_INVALID = "PROOF_INVALID"
    VERIFIER_TIMEOUT = "VERIFIER_TIMEOUT"
    VERIFIER_SPURIOUS_FAIL = "VERIFIER_SPURIOUS_FAIL"
    VERIFIER_SPURIOUS_PASS = "VERIFIER_SPURIOUS_PASS"
    VERIFIER_INTERNAL_ERROR = "VERIFIER_INTERNAL_ERROR"
    
    # Budget exhaustion
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"

@dataclass(frozen=True)
class VerifierOutcome:
    """Complete verifier outcome with telemetry."""
    
    error_code: VerifierErrorCode
    success: bool
    duration_ms: float
    tier: VerifierTier
    noise_injected: bool
    noise_type: Optional[str]
    attempt_count: int
    metadata: Dict[str, Any]
```

### Component 3: Noisy Lean Wrapper (`backend/verification/noisy_lean_wrapper.py`)

```python
class NoisyLeanWrapper:
    """Wraps Lean verifier with deterministic noise injection."""
    
    def __init__(
        self,
        base_runner: BuildRunner,
        noise_sampler: NoiseSampler,
        tier: VerifierTier
    ):
        self.base_runner = base_runner
        self.noise_sampler = noise_sampler
        self.tier = tier
    
    def verify(
        self,
        module_name: str,
        context: str,
        timeout: int
    ) -> VerifierOutcome:
        """Execute verification with noise injection."""
        
        start_time = time.time()
        noise_injected = False
        noise_type = None
        
        # Check for timeout injection
        if self.noise_sampler.should_timeout(context):
            noise_injected = True
            noise_type = "timeout"
            timeout_duration = self.noise_sampler.sample_timeout_duration(context)
            time.sleep(timeout_duration)
            return VerifierOutcome(
                error_code=VerifierErrorCode.VERIFIER_TIMEOUT,
                success=False,
                duration_ms=(time.time() - start_time) * 1000,
                tier=self.tier,
                noise_injected=True,
                noise_type="timeout",
                attempt_count=1,
                metadata={"simulated_timeout_ms": timeout_duration * 1000}
            )
        
        # Run base verifier
        result = self.base_runner(module_name)
        duration_ms = (time.time() - start_time) * 1000
        
        # Determine ground truth
        ground_truth_success = (result.returncode == 0)
        
        # Check for spurious failures
        if ground_truth_success and self.noise_sampler.should_spurious_fail(context):
            noise_injected = True
            noise_type = "spurious_fail"
            return VerifierOutcome(
                error_code=VerifierErrorCode.VERIFIER_SPURIOUS_FAIL,
                success=False,
                duration_ms=duration_ms,
                tier=self.tier,
                noise_injected=True,
                noise_type="spurious_fail",
                attempt_count=1,
                metadata={"ground_truth": "VERIFIED"}
            )
        
        # Check for spurious passes
        if not ground_truth_success and self.noise_sampler.should_spurious_pass(context):
            noise_injected = True
            noise_type = "spurious_pass"
            return VerifierOutcome(
                error_code=VerifierErrorCode.VERIFIER_SPURIOUS_PASS,
                success=True,
                duration_ms=duration_ms,
                tier=self.tier,
                noise_injected=True,
                noise_type="spurious_pass",
                attempt_count=1,
                metadata={"ground_truth": "FAILED"}
            )
        
        # No noise injected, return ground truth
        error_code = VerifierErrorCode.VERIFIED if ground_truth_success else VerifierErrorCode.PROOF_INVALID
        return VerifierOutcome(
            error_code=error_code,
            success=ground_truth_success,
            duration_ms=duration_ms,
            tier=self.tier,
            noise_injected=False,
            noise_type=None,
            attempt_count=1,
            metadata={}
        )
```

### Component 4: Tier Router (`backend/verification/tier_router.py`)

```python
class VerifierTierRouter:
    """Routes verification requests to appropriate tier with escalation."""
    
    def __init__(self, config: Dict[str, Any], seed: int):
        self.tiers = self._init_tiers(config, seed)
        self.escalation_policy = config.get("escalation_policy", "on_failure")
    
    def verify_with_escalation(
        self,
        module_name: str,
        context: str,
        max_attempts: int = 3
    ) -> VerifierOutcome:
        """Verify with automatic tier escalation on failure."""
        
        tier_sequence = [
            VerifierTier.FAST_NOISY,
            VerifierTier.BALANCED,
            VerifierTier.SLOW_PRECISE
        ]
        
        for attempt, tier in enumerate(tier_sequence[:max_attempts], start=1):
            wrapper = self.tiers[tier]
            outcome = wrapper.verify(
                module_name,
                f"{context}_attempt_{attempt}",
                timeout=self._get_timeout_for_tier(tier)
            )
            
            # Update attempt count
            outcome = dataclasses.replace(outcome, attempt_count=attempt)
            
            # Success or final attempt
            if outcome.success or attempt == max_attempts:
                return outcome
            
            # Log escalation
            print(f"INFO: Escalating from {tier.value} to next tier after {outcome.error_code.value}")
        
        return outcome
```

---

## Integration with U2 Runtime

### Modification Points

1. **`experiments/run_uplift_u2.py`**:
   - Load noise config alongside budget config
   - Initialize `NoiseSampler` with master seed
   - Pass noisy wrapper to execution function

2. **`create_execute_fn()` in `run_uplift_u2.py`**:
   - Replace direct Lean calls with `NoisyLeanWrapper.verify()`
   - Log `VerifierOutcome` to trace logs
   - Map outcomes to RFL feedback signals

3. **RFL Update Logic**:
   - Treat `VERIFIER_TIMEOUT` as abstention (no feedback)
   - Treat `VERIFIER_SPURIOUS_FAIL` as negative feedback (with caution flag)
   - Treat `VERIFIER_SPURIOUS_PASS` as positive feedback (with caution flag)
   - Log all noise injections for post-hoc analysis

---

## Testing Requirements

### Test 1: Seed Reproducibility
```python
def test_identical_seeds_produce_identical_noise():
    """Verify that identical seeds produce identical noise signatures."""
    seed = 42
    config = load_noise_config("test_slice")
    
    sampler1 = NoiseSampler(config, seed)
    sampler2 = NoiseSampler(config, seed)
    
    contexts = [f"context_{i}" for i in range(100)]
    
    for ctx in contexts:
        assert sampler1.should_timeout(ctx) == sampler2.should_timeout(ctx)
        assert sampler1.should_spurious_fail(ctx) == sampler2.should_spurious_fail(ctx)
        assert sampler1.should_spurious_pass(ctx) == sampler2.should_spurious_pass(ctx)
```

### Test 2: Noise Rate Validation
```python
def test_noise_rates_match_config():
    """Verify that empirical noise rates match configured rates."""
    seed = 12345
    config = NoiseConfig(
        timeout_rate=0.1,
        spurious_fail_rate=0.05,
        spurious_pass_rate=0.02
    )
    sampler = NoiseSampler(config, seed)
    
    n_samples = 10000
    contexts = [f"context_{i}" for i in range(n_samples)]
    
    timeout_count = sum(sampler.should_timeout(ctx) for ctx in contexts)
    fail_count = sum(sampler.should_spurious_fail(ctx) for ctx in contexts)
    pass_count = sum(sampler.should_spurious_pass(ctx) for ctx in contexts)
    
    assert abs(timeout_count / n_samples - 0.1) < 0.01
    assert abs(fail_count / n_samples - 0.05) < 0.01
    assert abs(pass_count / n_samples - 0.02) < 0.01
```

### Test 3: End-to-End Determinism
```python
def test_u2_run_determinism_with_noise():
    """Verify that entire U2 runs are deterministic with noise."""
    config = load_test_config()
    seed = 99999
    
    # Run 1
    result1 = run_experiment(
        slice_name="test_slice",
        cycles=10,
        seed=seed,
        mode="rfl",
        out_dir=Path("/tmp/test1"),
        config=config
    )
    
    # Run 2 with same seed
    result2 = run_experiment(
        slice_name="test_slice",
        cycles=10,
        seed=seed,
        mode="rfl",
        out_dir=Path("/tmp/test2"),
        config=config
    )
    
    # Compare outcomes
    assert result1.final_state_hash == result2.final_state_hash
    assert result1.trace_events == result2.trace_events
```

---

## Telemetry and Observability

### Metrics to Track

1. **Noise Injection Rates** (per tier, per slice):
   - `verifier.noise.timeout.rate`
   - `verifier.noise.spurious_fail.rate`
   - `verifier.noise.spurious_pass.rate`

2. **Tier Escalation Metrics**:
   - `verifier.escalation.count` (by tier transition)
   - `verifier.escalation.success_rate` (after escalation)

3. **RFL Robustness Metrics**:
   - `rfl.feedback.noise_flagged.count` (feedback with noise flag)
   - `rfl.policy.noise_resilience` (policy performance under noise)

### Trace Event Schema

```python
@dataclass
class VerifierNoiseEvent:
    """Trace event for noise injection."""
    
    event_type: str = "verifier_noise"
    cycle_id: int
    item_id: str
    tier: str
    noise_type: Optional[str]
    error_code: str
    duration_ms: float
    attempt_count: int
    seed_context: str
```

---

## Configuration Schema

### `config/verifier_noise_phase2.yaml`

```yaml
# Verifier Noise Configuration for Phase II
# All noise is deterministic and seeded for reproducibility

global:
  noise_enabled: true
  default_tier: "balanced"
  escalation_policy: "on_failure"  # or "on_timeout", "always"
  max_escalation_attempts: 3

tiers:
  fast_noisy:
    timeout_rate: 0.10
    spurious_fail_rate: 0.05
    spurious_pass_rate: 0.02
    base_timeout_s: 30
  
  balanced:
    timeout_rate: 0.05
    spurious_fail_rate: 0.02
    spurious_pass_rate: 0.01
    base_timeout_s: 60
  
  slow_precise:
    timeout_rate: 0.01
    spurious_fail_rate: 0.005
    spurious_pass_rate: 0.001
    base_timeout_s: 120

slices:
  arithmetic_simple:
    noise_enabled: true
    tier_overrides:
      fast_noisy:
        timeout_rate: 0.08
    timeout_distribution:
      type: "uniform"
      min_ms: 500
      max_ms: 1500
  
  algebra_expansion:
    noise_enabled: true
    tier_overrides:
      fast_noisy:
        timeout_rate: 0.12
    timeout_distribution:
      type: "exponential"
      mean_ms: 1000
```

---

## Invariants

1. **No Unseeded Randomness**: All stochastic behavior must use `DeterministicPRNG` with explicit seeds.
2. **Stable Error Codes**: All verifier outcomes map to `VerifierErrorCode` enum values.
3. **Reproducibility**: Identical seeds produce identical noise signatures across runs.
4. **Telemetry Completeness**: Every verifier call logs a `VerifierOutcome` with full metadata.
5. **RFL Robustness**: RFL updates must handle all error codes gracefully without silent failures.

---

## Implementation Checklist

- [ ] Create `backend/verification/` directory structure
- [ ] Implement `error_codes.py` with `VerifierErrorCode` and `VerifierOutcome`
- [ ] Implement `noise_sampler.py` with `NoiseSampler` class
- [ ] Implement `noisy_lean_wrapper.py` with `NoisyLeanWrapper` class
- [ ] Implement `tier_router.py` with `VerifierTierRouter` class
- [ ] Implement `config_loader.py` for loading `verifier_noise_phase2.yaml`
- [ ] Create `config/verifier_noise_phase2.yaml` with default noise parameters
- [ ] Modify `experiments/run_uplift_u2.py` to integrate noise models
- [ ] Add trace events for noise injection to `experiments/u2/logging.py`
- [ ] Implement test suite in `tests/verification/test_noise_model.py`
- [ ] Add integration tests for U2 runner with noise
- [ ] Document noise model in `docs/phase2/verifier_noise.md`

---

**End of Design Document**
