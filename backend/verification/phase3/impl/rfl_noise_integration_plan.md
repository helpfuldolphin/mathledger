# Task C4: RFL Noise Integration Migration Plan

**Author**: Manus-C (Telemetry Architect)  
**Date**: 2025-12-06  
**Status**: Migration Ready  
**Target**: `experiments/u2/runner.py`, `experiments/u2/policy.py`

---

## 1. Overview

This document provides a complete migration plan for integrating noise-robust RFL updates into the U2 runner. The integration includes:

1. **V_expected computation** — Replace binary feedback with expected value under verifier epistemic uncertainty
2. **Bias correction** — Apply correction factor to ensure RFL-stability
3. **Abstention handling** — Skip updates for timeouts, adjust effective learning rate
4. **Multi-tier aggregation** — Combine outcomes from multiple verifier tiers
5. **Replay log emission** — Emit noise decisions to replay log for deterministic reproduction

---

## 2. Architecture Overview

### 2.1 Current RFL Flow (Phase I/II)

```
Verification → Binary Outcome (success/failure) → Policy Update
                                                      ↓
                                              Δπ = η * V * ∇log π
```

### 2.2 New Noise-Robust RFL Flow (Phase III)

```
Verification → Noisy Outcome (VERIFIED/FAILED/TIMEOUT) → Expected Value Computation
                                                              ↓
                                                      V_expected = E[V | outcome, θ]
                                                              ↓
                                                      Bias Correction
                                                              ↓
                                                      V_corrected = V_expected / (1 - 2θ_sp)
                                                              ↓
                                                      Abstention Check
                                                              ↓
                                                      Policy Update (if not timeout)
                                                              ↓
                                                      Δπ = η_eff * V_corrected * ∇log π
                                                              ↓
                                                      Replay Log Emission
```

---

## 3. Module Structure

### 3.1 New Modules

```
backend/verification/phase3/
├── rfl_integration.py              # Main RFL integration module
│   ├── compute_expected_value()    # V_expected computation
│   ├── apply_bias_correction()     # Bias correction
│   ├── update_rfl_policy_noisy()   # Noise-robust RFL update
│   └── update_rfl_policy_multitier()  # Multi-tier aggregation
├── replay_log_writer.py            # Replay log emission
└── noise_config_loader.py          # Load noise config from YAML
```

### 3.2 Modified Modules

```
experiments/u2/
├── runner.py                       # U2Runner modifications
│   ├── __init__: Load noise config
│   ├── run_cycle: Inject noise, emit replay log
│   └── update_policy: Use noise-robust RFL update
└── policy.py                       # SearchPolicy modifications
    └── update: Accept V_corrected instead of binary V
```

---

## 4. Diff Map: U2Runner Modifications

### 4.1 File: `experiments/u2/runner.py`

#### Diff 1: Import noise integration modules

```diff
+from backend.verification.phase3.rfl_integration import (
+    compute_expected_value,
+    apply_bias_correction,
+    update_rfl_policy_noisy,
+)
+from backend.verification.phase3.replay_log_writer import NoiseReplayLogWriter
+from backend.verification.phase3.noise_config_loader import load_noise_config
+from backend.verification.error_codes import VerifierErrorCode
```

#### Diff 2: Add noise config to U2Config

```diff
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
     
+    # Phase III: Noise configuration
+    noise_config_path: Optional[Path] = None
+    enable_noise_injection: bool = False
+    enable_replay_log: bool = False
+    replay_log_path: Optional[Path] = None
```

#### Diff 3: Initialize noise config and replay log writer in U2Runner.__init__

```diff
 class U2Runner:
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
         
+        # Phase III: Load noise configuration
+        self.noise_config = None
+        if config.enable_noise_injection and config.noise_config_path:
+            self.noise_config = load_noise_config(config.noise_config_path)
+            print(f"INFO: Loaded noise config from {config.noise_config_path}")
+        
+        # Phase III: Initialize replay log writer
+        self.replay_log_writer = None
+        if config.enable_replay_log and config.replay_log_path:
+            self.replay_log_writer = NoiseReplayLogWriter(config.replay_log_path)
+            print(f"INFO: Replay log enabled at {config.replay_log_path}")
```

#### Diff 4: Modify run_cycle to inject noise and emit replay log

```diff
 def run_cycle(
     self,
     cycle: int,
     execute_fn: Callable[[str, int], Tuple[bool, Any]],
     trace_ctx: Optional[TracedExperimentContext] = None,
 ) -> CycleResult:
     """Run a single cycle."""
     
     # ... existing code ...
     
     # Execute verification
-    success, result = execute_fn(item, cycle)
+    # Phase III: Wrap execute_fn with noise injection
+    if self.noise_config:
+        success, result, noise_metadata = self._execute_with_noise(
+            execute_fn, item, cycle
+        )
+    else:
+        success, result = execute_fn(item, cycle)
+        noise_metadata = None
+    
+    # Phase III: Emit replay log entry
+    if self.replay_log_writer and noise_metadata:
+        self.replay_log_writer.write_noise_decision(
+            verification_id=noise_metadata["verification_id"],
+            cycle_id=cycle,
+            item=item,
+            context=noise_metadata["context"],
+            tier=noise_metadata["tier"],
+            seed=noise_metadata["seed"],
+            noise_decision=noise_metadata["noise_decision"],
+            verifier_outcome=noise_metadata["verifier_outcome"],
+        )
     
     # ... rest of existing code ...
```

#### Diff 5: Add _execute_with_noise helper method

```diff
+def _execute_with_noise(
+    self,
+    execute_fn: Callable[[str, int], Tuple[bool, Any]],
+    item: str,
+    cycle: int,
+) -> Tuple[bool, Any, Dict[str, Any]]:
+    """Execute verification with noise injection.
+    
+    Args:
+        execute_fn: Original execution function
+        item: Item to verify
+        cycle: Cycle number
+    
+    Returns:
+        Tuple of (success, result, noise_metadata)
+    """
+    from backend.verification.telemetry_runtime import run_lean_with_monitoring
+    from backend.verification.error_codes import VerifierTier
+    
+    # Determine tier (default to BALANCED)
+    tier = VerifierTier.BALANCED
+    timeout_s = 60.0
+    
+    # Create context for hierarchical seeding
+    context = f"cycle_{cycle}_item_{item}"
+    
+    # Run verification with noise injection
+    telemetry = run_lean_with_monitoring(
+        module_name=item,
+        tier=tier,
+        timeout_s=timeout_s,
+        context=context,
+        master_seed=self.config.master_seed,
+        noise_config=self.noise_config,
+    )
+    
+    # Convert telemetry outcome to success flag
+    success = (telemetry.outcome == VerifierErrorCode.VERIFIED)
+    
+    # Package result
+    result = {
+        "outcome": telemetry.outcome.value,
+        "success": success,
+        "duration_ms": telemetry.duration_ms,
+        "noise_injected": telemetry.noise_injected,
+        "ground_truth": telemetry.ground_truth,
+    }
+    
+    # Package noise metadata for replay log
+    noise_metadata = {
+        "verification_id": telemetry.verification_id,
+        "context": context,
+        "tier": tier.value,
+        "seed": {
+            "master_seed": self.config.master_seed,
+            "context_hash": hash(context),
+        },
+        "noise_decision": {
+            "noise_injected": telemetry.noise_injected,
+            "noise_type": telemetry.noise_type,
+        },
+        "verifier_outcome": {
+            "outcome": telemetry.outcome.value,
+            "success": success,
+            "duration_ms": telemetry.duration_ms,
+            "ground_truth": telemetry.ground_truth,
+        },
+    }
+    
+    return success, result, noise_metadata
```

#### Diff 6: Modify policy update to use noise-robust RFL

```diff
 def update_policy(
     self,
     item: str,
-    success: bool,
+    outcome: VerifierErrorCode,
     cycle: int,
     trace_ctx: Optional[TracedExperimentContext] = None,
 ) -> None:
     """Update RFL policy based on verification outcome."""
     
     if self.config.mode != "rfl":
         return  # No policy update for baseline mode
     
-    # Phase I/II: Binary feedback
-    value = 1.0 if success else -1.0
+    # Phase III: Noise-robust RFL update
+    if self.noise_config:
+        # Extract noise rates from config
+        tier_config = self.noise_config.get("balanced", {})
+        theta_timeout = tier_config.get("timeout_rate", 0.0)
+        theta_spurious_fail = tier_config.get("spurious_fail_rate", 0.0)
+        theta_spurious_pass = tier_config.get("spurious_pass_rate", 0.0)
+        
+        # Compute expected value
+        value_expected = compute_expected_value(
+            outcome=outcome,
+            theta_spurious_fail=theta_spurious_fail,
+            theta_spurious_pass=theta_spurious_pass,
+            theta_timeout=theta_timeout,
+        )
+        
+        # Apply bias correction
+        value_corrected = apply_bias_correction(
+            value_expected=value_expected,
+            theta_spurious_pass=theta_spurious_pass,
+        )
+        
+        # Abstention handling: skip update if timeout
+        if outcome == VerifierErrorCode.VERIFIER_TIMEOUT:
+            print(f"INFO: Abstaining from policy update for {item} (timeout)")
+            return
+        
+        value = value_corrected
+    else:
+        # Phase I/II: Binary feedback (fallback)
+        value = 1.0 if outcome == VerifierErrorCode.VERIFIED else -1.0
     
     # Update policy
     self.policy.update(item, value, cycle)
     self.policy_update_count += 1
```

---

## 5. Diff Map: SearchPolicy Modifications

### 5.1 File: `experiments/u2/policy.py`

#### Diff 1: Modify update signature to accept continuous value

```diff
 class SearchPolicy:
     """Base class for search policies."""
     
     def update(
         self,
         item: str,
-        success: bool,
+        value: float,
         cycle: int,
     ) -> None:
         """Update policy based on feedback.
         
         Args:
             item: Item that was verified
-            success: Whether verification succeeded
+            value: Feedback value (continuous in [-1, 1])
             cycle: Cycle number
         """
         raise NotImplementedError
```

#### Diff 2: Update RFLPolicy.update to use continuous value

```diff
 class RFLPolicy(SearchPolicy):
     """RFL policy with policy gradient updates."""
     
     def update(
         self,
         item: str,
-        success: bool,
+        value: float,
         cycle: int,
     ) -> None:
         """Update policy weights using policy gradient."""
         
-        # Convert success to value
-        value = 1.0 if success else -1.0
-        
         # Compute gradient
         grad = self._compute_gradient(item, value)
         
         # Update weights
         self.weights[item] = self.weights.get(item, 0.0) + self.learning_rate * grad
```

---

## 6. Implementation: RFL Integration Module

### 6.1 File: `backend/verification/phase3/rfl_integration.py`

```python
"""
RFL Integration with Noise-Robust Updates

This module implements noise-robust RFL policy updates that account for
verifier epistemic uncertainty.

Author: Manus-C (Telemetry Architect)
Date: 2025-12-06
"""

from typing import Dict, Any, Optional
from backend.verification.error_codes import VerifierErrorCode


def compute_expected_value(
    outcome: VerifierErrorCode,
    theta_spurious_fail: float,
    theta_spurious_pass: float,
    theta_timeout: float,
) -> float:
    """Compute expected value under verifier epistemic uncertainty.
    
    Uses Bayes' theorem to compute posterior probability of ground truth
    validity given verifier outcome and noise rates.
    
    Args:
        outcome: Verifier outcome
        theta_spurious_fail: Spurious failure rate
        theta_spurious_pass: Spurious pass rate
        theta_timeout: Timeout rate
    
    Returns:
        Expected value in [-1, 1]
    """
    
    if outcome == VerifierErrorCode.VERIFIED:
        # P(y = VALID | o = VERIFIED)
        p_valid_given_verified = (
            (1 - theta_spurious_fail - theta_timeout) /
            ((1 - theta_spurious_fail - theta_timeout) + theta_spurious_pass)
        )
        return 2 * p_valid_given_verified - 1
    
    elif outcome == VerifierErrorCode.PROOF_INVALID:
        # P(y = VALID | o = FAILED)
        p_valid_given_failed = (
            theta_spurious_fail /
            (theta_spurious_fail + (1 - theta_spurious_pass - theta_timeout))
        )
        return 2 * p_valid_given_failed - 1
    
    elif outcome == VerifierErrorCode.VERIFIER_TIMEOUT:
        # No information from timeout
        return 0.0
    
    else:
        # Other outcomes (errors, abstentions) treated as no information
        return 0.0


def apply_bias_correction(
    value_expected: float,
    theta_spurious_pass: float,
) -> float:
    """Apply bias correction to ensure RFL-stability.
    
    Corrects for bias introduced by spurious passes.
    
    Args:
        value_expected: Expected value before correction
        theta_spurious_pass: Spurious pass rate
    
    Returns:
        Bias-corrected value
    """
    
    # Avoid division by zero
    if theta_spurious_pass >= 0.5:
        # Degenerate case: too much noise
        return value_expected
    
    # Bias correction factor
    correction_factor = 1 - 2 * theta_spurious_pass
    
    # Apply correction
    value_corrected = value_expected / correction_factor
    
    # Clamp to [-1, 1]
    value_corrected = max(-1.0, min(1.0, value_corrected))
    
    return value_corrected


def update_rfl_policy_noisy(
    policy_weights: Dict[str, float],
    item: str,
    outcome: VerifierErrorCode,
    theta_spurious_fail: float,
    theta_spurious_pass: float,
    theta_timeout: float,
    learning_rate: float,
) -> Dict[str, float]:
    """Update RFL policy with noise-robust update.
    
    Args:
        policy_weights: Current policy weights
        item: Item that was verified
        outcome: Verifier outcome
        theta_spurious_fail: Spurious failure rate
        theta_spurious_pass: Spurious pass rate
        theta_timeout: Timeout rate
        learning_rate: Learning rate
    
    Returns:
        Updated policy weights
    """
    
    # Compute expected value
    value_expected = compute_expected_value(
        outcome=outcome,
        theta_spurious_fail=theta_spurious_fail,
        theta_spurious_pass=theta_spurious_pass,
        theta_timeout=theta_timeout,
    )
    
    # Apply bias correction
    value_corrected = apply_bias_correction(
        value_expected=value_expected,
        theta_spurious_pass=theta_spurious_pass,
    )
    
    # Abstention handling: skip update if timeout
    if outcome == VerifierErrorCode.VERIFIER_TIMEOUT:
        return policy_weights  # No update
    
    # Compute gradient (simplified: assume log-linear policy)
    grad = value_corrected  # ∇log π(item) ≈ 1 for log-linear
    
    # Update weight
    new_weights = policy_weights.copy()
    new_weights[item] = new_weights.get(item, 0.0) + learning_rate * grad
    
    return new_weights


def update_rfl_policy_multitier(
    policy_weights: Dict[str, float],
    item: str,
    outcomes: Dict[str, VerifierErrorCode],
    noise_configs: Dict[str, Dict[str, float]],
    learning_rate: float,
) -> Dict[str, float]:
    """Update RFL policy with multi-tier aggregation.
    
    Args:
        policy_weights: Current policy weights
        item: Item that was verified
        outcomes: Dict mapping tier name to outcome
        noise_configs: Dict mapping tier name to noise config
        learning_rate: Learning rate
    
    Returns:
        Updated policy weights
    """
    
    # Confidence weights for each tier
    tier_weights = {
        "fast_noisy": 0.2,
        "balanced": 0.5,
        "slow_precise": 1.0,
    }
    
    # Compute weighted expected value
    total_weight = 0.0
    weighted_value = 0.0
    
    for tier, outcome in outcomes.items():
        tier_config = noise_configs.get(tier, {})
        
        # Compute expected value for this tier
        value_expected = compute_expected_value(
            outcome=outcome,
            theta_spurious_fail=tier_config.get("spurious_fail_rate", 0.0),
            theta_spurious_pass=tier_config.get("spurious_pass_rate", 0.0),
            theta_timeout=tier_config.get("timeout_rate", 0.0),
        )
        
        # Apply bias correction
        value_corrected = apply_bias_correction(
            value_expected=value_expected,
            theta_spurious_pass=tier_config.get("spurious_pass_rate", 0.0),
        )
        
        # Aggregate with confidence weight
        weight = tier_weights.get(tier, 0.5)
        weighted_value += weight * value_corrected
        total_weight += weight
    
    # Normalize
    if total_weight > 0:
        aggregated_value = weighted_value / total_weight
    else:
        aggregated_value = 0.0
    
    # Update policy
    grad = aggregated_value
    new_weights = policy_weights.copy()
    new_weights[item] = new_weights.get(item, 0.0) + learning_rate * grad
    
    return new_weights
```

---

## 7. Implementation: Replay Log Writer

### 7.1 File: `backend/verification/phase3/replay_log_writer.py`

```python
"""
Noise Replay Log Writer

Emits noise decisions to replay log for deterministic reproduction.

Author: Manus-C (Telemetry Architect)
Date: 2025-12-06
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional


class NoiseReplayLogWriter:
    """Writer for noise replay logs."""
    
    def __init__(self, log_path: Path):
        """Initialize replay log writer.
        
        Args:
            log_path: Path to replay log file (JSONL format)
        """
        self.log_path = log_path
        self.log_file = open(log_path, "w")
    
    def write_noise_decision(
        self,
        verification_id: str,
        cycle_id: int,
        item: str,
        context: str,
        tier: str,
        seed: Dict[str, Any],
        noise_decision: Dict[str, Any],
        verifier_outcome: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write noise decision to replay log.
        
        Args:
            verification_id: Unique verification ID
            cycle_id: Cycle number
            item: Item that was verified
            context: Context string for PRNG seeding
            tier: Verifier tier
            seed: Seed information
            noise_decision: Noise decision details
            verifier_outcome: Verifier outcome details
            metadata: Optional metadata
        """
        
        entry = {
            "version": "1.0",
            "entry_type": "noise_decision",
            "timestamp": time.time(),
            "verification_id": verification_id,
            "cycle_id": cycle_id,
            "item": item,
            "context": context,
            "tier": tier,
            "seed": seed,
            "noise_decision": noise_decision,
            "verifier_outcome": verifier_outcome,
            "metadata": metadata or {},
        }
        
        self.log_file.write(json.dumps(entry) + "\n")
        self.log_file.flush()  # Ensure immediate write
    
    def close(self) -> None:
        """Close replay log file."""
        self.log_file.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
```

---

## 8. Implementation: Noise Config Loader

### 8.1 File: `backend/verification/phase3/noise_config_loader.py`

```python
"""
Noise Configuration Loader

Loads noise configuration from YAML files.

Author: Manus-C (Telemetry Architect)
Date: 2025-12-06
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_noise_config(config_path: Path) -> Dict[str, Any]:
    """Load noise configuration from YAML file.
    
    Args:
        config_path: Path to noise configuration YAML
    
    Returns:
        Noise configuration dict
    """
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Validate required fields
    if "noise_models" not in config:
        raise ValueError("Noise config missing 'noise_models' field")
    
    return config["noise_models"]
```

---

## 9. Integration Sequence

### 9.1 Step-by-Step Integration

**Step 1**: Implement `rfl_integration.py` with all functions  
**Step 2**: Implement `replay_log_writer.py`  
**Step 3**: Implement `noise_config_loader.py`  
**Step 4**: Apply Diff 1-3 to `experiments/u2/runner.py` (imports, config, __init__)  
**Step 5**: Apply Diff 4-6 to `experiments/u2/runner.py` (run_cycle, _execute_with_noise, update_policy)  
**Step 6**: Apply Diff 1-2 to `experiments/u2/policy.py` (update signature)  
**Step 7**: Write integration tests  
**Step 8**: Run end-to-end test with noise injection  
**Step 9**: Validate replay log determinism  
**Step 10**: Deploy to production

---

## 10. Testing Strategy

### 10.1 Unit Tests

```python
# tests/verification/test_rfl_integration.py

def test_compute_expected_value_verified():
    """Test expected value for VERIFIED outcome."""
    value = compute_expected_value(
        outcome=VerifierErrorCode.VERIFIED,
        theta_spurious_fail=0.05,
        theta_spurious_pass=0.02,
        theta_timeout=0.1,
    )
    
    # Should be positive (closer to +1)
    assert value > 0.5


def test_compute_expected_value_failed():
    """Test expected value for FAILED outcome."""
    value = compute_expected_value(
        outcome=VerifierErrorCode.PROOF_INVALID,
        theta_spurious_fail=0.05,
        theta_spurious_pass=0.02,
        theta_timeout=0.1,
    )
    
    # Should be negative (closer to -1)
    assert value < -0.5


def test_compute_expected_value_timeout():
    """Test expected value for TIMEOUT outcome."""
    value = compute_expected_value(
        outcome=VerifierErrorCode.VERIFIER_TIMEOUT,
        theta_spurious_fail=0.05,
        theta_spurious_pass=0.02,
        theta_timeout=0.1,
    )
    
    # Should be zero (no information)
    assert value == 0.0


def test_bias_correction():
    """Test bias correction."""
    value_corrected = apply_bias_correction(
        value_expected=0.8,
        theta_spurious_pass=0.1,
    )
    
    # Corrected value should be higher
    assert value_corrected > 0.8
```

### 10.2 Integration Tests

```python
# tests/verification/test_u2_noise_integration.py

def test_u2_runner_with_noise():
    """Test U2 runner with noise injection."""
    
    config = U2Config(
        experiment_id="test_noise",
        slice_name="test_slice",
        mode="rfl",
        total_cycles=10,
        master_seed=12345,
        enable_noise_injection=True,
        noise_config_path=Path("config/noise_test.yaml"),
        enable_replay_log=True,
        replay_log_path=Path("test_replay.jsonl"),
    )
    
    runner = U2Runner(config)
    
    # Run cycles
    for cycle in range(10):
        result = runner.run_cycle(cycle, mock_execute_fn)
        assert result.success is not None
    
    # Verify replay log exists
    assert Path("test_replay.jsonl").exists()


def test_replay_log_determinism():
    """Test that replay log enables deterministic reproduction."""
    
    # Run experiment twice with same seed
    for run in range(2):
        config = U2Config(
            experiment_id=f"test_determinism_{run}",
            slice_name="test_slice",
            mode="rfl",
            total_cycles=10,
            master_seed=12345,
            enable_noise_injection=True,
            noise_config_path=Path("config/noise_test.yaml"),
            enable_replay_log=True,
            replay_log_path=Path(f"test_replay_{run}.jsonl"),
        )
        
        runner = U2Runner(config)
        
        for cycle in range(10):
            runner.run_cycle(cycle, mock_execute_fn)
    
    # Compare replay logs
    with open("test_replay_0.jsonl") as f0, open("test_replay_1.jsonl") as f1:
        log0 = [json.loads(line) for line in f0]
        log1 = [json.loads(line) for line in f1]
    
    # Outcomes should be identical
    for entry0, entry1 in zip(log0, log1):
        assert entry0["verifier_outcome"] == entry1["verifier_outcome"]
```

---

## 11. Deployment Checklist

- [ ] Implement `rfl_integration.py` with all functions
- [ ] Implement `replay_log_writer.py`
- [ ] Implement `noise_config_loader.py`
- [ ] Apply all diffs to `experiments/u2/runner.py`
- [ ] Apply all diffs to `experiments/u2/policy.py`
- [ ] Write unit tests for RFL integration functions
- [ ] Write integration tests for U2 runner with noise
- [ ] Test replay log determinism
- [ ] Create example noise configuration YAML
- [ ] Document RFL integration in README
- [ ] Run end-to-end test on real Mathlib proofs
- [ ] Validate policy convergence with noise
- [ ] Deploy to production

---

**Manus-C — Telemetry Architect**  
*"Every packet accounted for, every signal explained."*

**Status**: Migration Ready  
**Next**: Task C5 (Drift Radar Code Map)
