# Deterministic Replay: End-to-End Plan

**Author**: Manus-F  
**Date**: 2025-12-06  
**Status**: Implementation Plan (Ready for Execution)

---

## Overview

This document provides a **complete end-to-end plan** for implementing deterministic replay of U2 Planner experiments. Replay is the ultimate test of determinism: given the same master seed and configuration, the planner must produce identical results.

---

## 1. Replay Engine Flowchart

```
┌─────────────────────────────────────────────────────────────┐
│                    REPLAY ENGINE WORKFLOW                    │
└─────────────────────────────────────────────────────────────┘

┌──────────────────┐
│  Load Original   │
│  Provenance      │
│  Bundle          │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Extract Config  │
│  - master_seed   │
│  - slice_name    │
│  - total_cycles  │
│  - mode          │
│  - beam_width    │
│  - max_depth     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Initialize      │
│  Replay Runner   │
│  - Same PRNG     │
│  - Same frontier │
│  - Same policy   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Run Experiment  │
│  (Identical to   │
│   Original)      │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Collect Replay  │
│  Artifacts       │
│  - trace.jsonl   │
│  - snapshots     │
│  - final state   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Compare Hashes  │
│  - Trace hash    │
│  - State hash    │
│  - Per-cycle     │
└────────┬─────────┘
         │
         ▼
    ┌───┴───┐
    │ Match?│
    └───┬───┘
        │
    ┌───┴───────────┐
    │               │
    ▼               ▼
┌────────┐    ┌──────────┐
│  PASS  │    │  FAIL    │
│        │    │          │
│ Replay │    │ Generate │
│ Success│    │ Diff     │
│        │    │ Report   │
└────────┘    └──────────┘
```

---

## 2. Expected Invariants

### 2.1. Trace Invariant

**Statement**: Same master seed → identical trace hash

**Formal Definition**:
```
∀ seed, config:
  hash(trace(run(seed, config))) = hash(trace(replay(seed, config)))
```

**Verification**:
```python
def verify_trace_invariant(original_trace_hash, replay_trace_hash):
    return original_trace_hash == replay_trace_hash
```

### 2.2. State Invariant

**Statement**: Same master seed → identical final state hash

**Formal Definition**:
```
∀ seed, config:
  hash(state(run(seed, config))) = hash(state(replay(seed, config)))
```

**Verification**:
```python
def verify_state_invariant(original_state_hash, replay_state_hash):
    return original_state_hash == replay_state_hash
```

### 2.3. Per-Cycle Invariant

**Statement**: Same master seed → identical per-cycle hashes

**Formal Definition**:
```
∀ seed, config, cycle:
  hash(trace_cycle(run(seed, config), cycle)) = 
  hash(trace_cycle(replay(seed, config), cycle))
```

**Verification**:
```python
def verify_per_cycle_invariant(original_cycle_hashes, replay_cycle_hashes):
    for cycle in original_cycle_hashes:
        if original_cycle_hashes[cycle] != replay_cycle_hashes[cycle]:
            return False, cycle
    return True, None
```

### 2.4. Frontier Invariant

**Statement**: Same master seed → identical frontier state at each cycle

**Formal Definition**:
```
∀ seed, config, cycle:
  hash(frontier(run(seed, config), cycle)) = 
  hash(frontier(replay(seed, config), cycle))
```

**Verification**:
```python
def verify_frontier_invariant(original_frontier_hash, replay_frontier_hash):
    return original_frontier_hash == replay_frontier_hash
```

---

## 3. Hash-Threshold Test Suite

### 3.1. Test Categories

#### Category 1: Exact Match Tests

**Purpose**: Verify bit-for-bit identical outputs

**Tests**:
1. **Trace Hash Match**: Full trace SHA-256 must match exactly
2. **State Hash Match**: Final state SHA-256 must match exactly
3. **Frontier Hash Match**: Frontier state SHA-256 must match exactly

**Threshold**: 0 tolerance (exact match required)

#### Category 2: Per-Cycle Tests

**Purpose**: Verify determinism at cycle granularity

**Tests**:
1. **Per-Cycle Trace Hash**: Each cycle's trace hash must match
2. **Per-Cycle Frontier Size**: Frontier size at each cycle must match
3. **Per-Cycle Execution Count**: Number of executions per cycle must match

**Threshold**: 0 tolerance (exact match required)

#### Category 3: Statistical Tests

**Purpose**: Verify statistical properties (sanity checks)

**Tests**:
1. **Total Execution Count**: Total executions must match
2. **Success Rate**: Success rate must match (within floating-point precision)
3. **Average Execution Time**: Average time must match (within 1ms tolerance)

**Threshold**: 
- Execution count: 0 tolerance
- Success rate: 1e-6 tolerance
- Execution time: 1ms tolerance

### 3.2. Test Suite Implementation

```python
@dataclass
class ReplayTestResult:
    """Result of a single replay test."""
    test_name: str
    passed: bool
    expected: Any
    actual: Any
    diff: Optional[str]

class ReplayTestSuite:
    """
    Comprehensive test suite for deterministic replay.
    """
    
    def __init__(self, original_bundle: Path, replay_bundle: Path):
        """
        Initialize test suite.
        
        Args:
            original_bundle: Path to original provenance bundle
            replay_bundle: Path to replay provenance bundle
        """
        self.original_bundle = original_bundle
        self.replay_bundle = replay_bundle
        self.results = []
    
    def run_all_tests(self) -> List[ReplayTestResult]:
        """Run all replay tests."""
        # Category 1: Exact Match Tests
        self.test_trace_hash_match()
        self.test_state_hash_match()
        self.test_frontier_hash_match()
        
        # Category 2: Per-Cycle Tests
        self.test_per_cycle_trace_hashes()
        self.test_per_cycle_frontier_sizes()
        self.test_per_cycle_execution_counts()
        
        # Category 3: Statistical Tests
        self.test_total_execution_count()
        self.test_success_rate()
        self.test_average_execution_time()
        
        return self.results
    
    def test_trace_hash_match(self):
        """Test: Full trace hash must match exactly."""
        original_hash = self._load_trace_hash(self.original_bundle)
        replay_hash = self._load_trace_hash(self.replay_bundle)
        
        passed = (original_hash == replay_hash)
        self.results.append(ReplayTestResult(
            test_name="trace_hash_match",
            passed=passed,
            expected=original_hash,
            actual=replay_hash,
            diff=None if passed else f"Expected {original_hash}, got {replay_hash}",
        ))
    
    def test_state_hash_match(self):
        """Test: Final state hash must match exactly."""
        original_hash = self._load_state_hash(self.original_bundle)
        replay_hash = self._load_state_hash(self.replay_bundle)
        
        passed = (original_hash == replay_hash)
        self.results.append(ReplayTestResult(
            test_name="state_hash_match",
            passed=passed,
            expected=original_hash,
            actual=replay_hash,
            diff=None if passed else f"Expected {original_hash}, got {replay_hash}",
        ))
    
    def test_frontier_hash_match(self):
        """Test: Frontier state hash must match exactly."""
        original_hash = self._load_frontier_hash(self.original_bundle)
        replay_hash = self._load_frontier_hash(self.replay_bundle)
        
        passed = (original_hash == replay_hash)
        self.results.append(ReplayTestResult(
            test_name="frontier_hash_match",
            passed=passed,
            expected=original_hash,
            actual=replay_hash,
            diff=None if passed else f"Expected {original_hash}, got {replay_hash}",
        ))
    
    def test_per_cycle_trace_hashes(self):
        """Test: Per-cycle trace hashes must match."""
        original_hashes = self._load_per_cycle_hashes(self.original_bundle)
        replay_hashes = self._load_per_cycle_hashes(self.replay_bundle)
        
        passed = True
        diff_cycles = []
        for cycle in original_hashes:
            if original_hashes[cycle] != replay_hashes.get(cycle):
                passed = False
                diff_cycles.append(cycle)
        
        self.results.append(ReplayTestResult(
            test_name="per_cycle_trace_hashes",
            passed=passed,
            expected=original_hashes,
            actual=replay_hashes,
            diff=None if passed else f"Mismatched cycles: {diff_cycles}",
        ))
    
    def test_per_cycle_frontier_sizes(self):
        """Test: Per-cycle frontier sizes must match."""
        original_sizes = self._load_per_cycle_frontier_sizes(self.original_bundle)
        replay_sizes = self._load_per_cycle_frontier_sizes(self.replay_bundle)
        
        passed = (original_sizes == replay_sizes)
        self.results.append(ReplayTestResult(
            test_name="per_cycle_frontier_sizes",
            passed=passed,
            expected=original_sizes,
            actual=replay_sizes,
            diff=None if passed else f"Frontier sizes differ",
        ))
    
    def test_per_cycle_execution_counts(self):
        """Test: Per-cycle execution counts must match."""
        original_counts = self._load_per_cycle_execution_counts(self.original_bundle)
        replay_counts = self._load_per_cycle_execution_counts(self.replay_bundle)
        
        passed = (original_counts == replay_counts)
        self.results.append(ReplayTestResult(
            test_name="per_cycle_execution_counts",
            passed=passed,
            expected=original_counts,
            actual=replay_counts,
            diff=None if passed else f"Execution counts differ",
        ))
    
    def test_total_execution_count(self):
        """Test: Total execution count must match."""
        original_count = self._load_total_execution_count(self.original_bundle)
        replay_count = self._load_total_execution_count(self.replay_bundle)
        
        passed = (original_count == replay_count)
        self.results.append(ReplayTestResult(
            test_name="total_execution_count",
            passed=passed,
            expected=original_count,
            actual=replay_count,
            diff=None if passed else f"Expected {original_count}, got {replay_count}",
        ))
    
    def test_success_rate(self):
        """Test: Success rate must match (within tolerance)."""
        original_rate = self._load_success_rate(self.original_bundle)
        replay_rate = self._load_success_rate(self.replay_bundle)
        
        tolerance = 1e-6
        passed = abs(original_rate - replay_rate) < tolerance
        self.results.append(ReplayTestResult(
            test_name="success_rate",
            passed=passed,
            expected=original_rate,
            actual=replay_rate,
            diff=None if passed else f"Expected {original_rate}, got {replay_rate}",
        ))
    
    def test_average_execution_time(self):
        """Test: Average execution time must match (within tolerance)."""
        original_time = self._load_average_execution_time(self.original_bundle)
        replay_time = self._load_average_execution_time(self.replay_bundle)
        
        tolerance_ms = 1.0
        passed = abs(original_time - replay_time) < tolerance_ms
        self.results.append(ReplayTestResult(
            test_name="average_execution_time",
            passed=passed,
            expected=original_time,
            actual=replay_time,
            diff=None if passed else f"Expected {original_time}ms, got {replay_time}ms",
        ))
    
    # Helper methods to load data from bundles
    def _load_trace_hash(self, bundle: Path) -> str:
        """Load full trace hash from bundle."""
        cert_path = bundle / "reproducibility_certificate.json"
        with open(cert_path, 'r') as f:
            cert = json.load(f)
        return cert["hashes"]["full_trace_hash"]
    
    def _load_state_hash(self, bundle: Path) -> str:
        """Load final state hash from bundle."""
        cert_path = bundle / "reproducibility_certificate.json"
        with open(cert_path, 'r') as f:
            cert = json.load(f)
        return cert["hashes"]["final_state_hash"]
    
    def _load_frontier_hash(self, bundle: Path) -> str:
        """Load frontier state hash from bundle."""
        cert_path = bundle / "reproducibility_certificate.json"
        with open(cert_path, 'r') as f:
            cert = json.load(f)
        return cert["hashes"]["frontier_state_hash"]
    
    def _load_per_cycle_hashes(self, bundle: Path) -> Dict[int, str]:
        """Load per-cycle trace hashes from bundle."""
        cert_path = bundle / "reproducibility_certificate.json"
        with open(cert_path, 'r') as f:
            cert = json.load(f)
        return {int(k): v for k, v in cert["hashes"]["per_cycle_hashes"].items()}
    
    def _load_per_cycle_frontier_sizes(self, bundle: Path) -> Dict[int, int]:
        """Load per-cycle frontier sizes from telemetry."""
        telemetry_path = bundle / "telemetry" / "frontier_evolution.json"
        with open(telemetry_path, 'r') as f:
            telemetry = json.load(f)
        return {int(k): v["size"] for k, v in telemetry.items()}
    
    def _load_per_cycle_execution_counts(self, bundle: Path) -> Dict[int, int]:
        """Load per-cycle execution counts from telemetry."""
        telemetry_path = bundle / "telemetry" / "telemetry.json"
        with open(telemetry_path, 'r') as f:
            telemetry = json.load(f)
        return {int(k): v["executions"] for k, v in telemetry.items()}
    
    def _load_total_execution_count(self, bundle: Path) -> int:
        """Load total execution count from telemetry."""
        telemetry_path = bundle / "telemetry" / "telemetry.json"
        with open(telemetry_path, 'r') as f:
            telemetry = json.load(f)
        return sum(v["executions"] for v in telemetry.values())
    
    def _load_success_rate(self, bundle: Path) -> float:
        """Load overall success rate from telemetry."""
        telemetry_path = bundle / "telemetry" / "telemetry.json"
        with open(telemetry_path, 'r') as f:
            telemetry = json.load(f)
        total_executions = sum(v["executions"] for v in telemetry.values())
        total_successes = sum(v["successes"] for v in telemetry.values())
        return total_successes / total_executions if total_executions > 0 else 0.0
    
    def _load_average_execution_time(self, bundle: Path) -> float:
        """Load average execution time from telemetry."""
        telemetry_path = bundle / "telemetry" / "telemetry.json"
        with open(telemetry_path, 'r') as f:
            telemetry = json.load(f)
        total_time = sum(v["total_time_ms"] for v in telemetry.values())
        total_executions = sum(v["executions"] for v in telemetry.values())
        return total_time / total_executions if total_executions > 0 else 0.0
    
    def generate_report(self) -> str:
        """Generate test report."""
        passed_count = sum(1 for r in self.results if r.passed)
        total_count = len(self.results)
        
        report = []
        report.append("=" * 60)
        report.append("DETERMINISTIC REPLAY TEST REPORT")
        report.append("=" * 60)
        report.append(f"Total Tests: {total_count}")
        report.append(f"Passed: {passed_count}")
        report.append(f"Failed: {total_count - passed_count}")
        report.append("")
        
        for result in self.results:
            status = "[PASS]" if result.passed else "[FAIL]"
            report.append(f"{status} {result.test_name}")
            if not result.passed and result.diff:
                report.append(f"  Diff: {result.diff}")
        
        report.append("=" * 60)
        
        if passed_count == total_count:
            report.append("VERDICT: REPLAY SUCCESS - All invariants verified")
        else:
            report.append("VERDICT: REPLAY FAILURE - Determinism violated")
        
        report.append("=" * 60)
        
        return "\n".join(report)
```

---

## 4. Conformance Validator Contract

### 4.1. Purpose

Define the contract that all replay implementations must satisfy.

### 4.2. Validator Interface

```python
class ConformanceValidator:
    """
    Validates that a replay implementation conforms to determinism contract.
    """
    
    def validate(self, original_bundle: Path, replay_bundle: Path) -> ValidationResult:
        """
        Validate replay conformance.
        
        Args:
            original_bundle: Path to original provenance bundle
            replay_bundle: Path to replay provenance bundle
            
        Returns:
            ValidationResult with pass/fail status and details
        """
        pass

@dataclass
class ValidationResult:
    """Result of conformance validation."""
    passed: bool
    test_results: List[ReplayTestResult]
    invariants_verified: List[str]
    invariants_violated: List[str]
    report: str
```

### 4.3. Conformance Contract

All replay implementations MUST satisfy:

1. **Trace Invariant**: Same seed → identical trace hash
2. **State Invariant**: Same seed → identical final state hash
3. **Per-Cycle Invariant**: Same seed → identical per-cycle hashes
4. **Frontier Invariant**: Same seed → identical frontier state hash

**Verification**:
```python
def verify_conformance(original_bundle, replay_bundle):
    validator = ConformanceValidator()
    result = validator.validate(original_bundle, replay_bundle)
    
    if result.passed:
        print("✓ Replay implementation conforms to determinism contract")
    else:
        print("✗ Replay implementation violates determinism contract")
        print(f"Violated invariants: {result.invariants_violated}")
    
    return result
```

---

## 5. Replay Engine Implementation

### 5.1. ReplayEngine Class

```python
class ReplayEngine:
    """
    Executes deterministic replay of U2 experiments.
    """
    
    def __init__(self, original_bundle: Path):
        """
        Initialize replay engine.
        
        Args:
            original_bundle: Path to original provenance bundle
        """
        self.original_bundle = original_bundle
        self.config = self._load_config()
    
    def replay(self, output_dir: Path) -> Path:
        """
        Replay experiment and generate new provenance bundle.
        
        Args:
            output_dir: Output directory for replay bundle
            
        Returns:
            Path to replay provenance bundle
        """
        print(f"Replaying experiment: {self.config.experiment_id}")
        print(f"Master seed: {self.config.master_seed}")
        print(f"Total cycles: {self.config.total_cycles}")
        
        # Initialize U2 runner with same config
        runner = U2Runner(self.config)
        
        # Run experiment
        execution_started = datetime.utcnow().isoformat() + "Z"
        
        for cycle in range(self.config.total_cycles):
            runner.run_cycle(cycle, execute_fn)
            print(f"Cycle {cycle} complete")
        
        execution_completed = datetime.utcnow().isoformat() + "Z"
        
        # Generate provenance bundle
        bundle_generator = ProvenanceBundleGenerator()
        replay_bundle_path = output_dir / f"replay_bundle_{self.config.experiment_id}"
        
        bundle_generator.generate(
            config=self.config,
            output_dir=output_dir,
            worker_trace_paths=[runner.trace_path],
            initial_state_hash=runner.initial_state_hash,
            final_state_hash=runner.final_state_hash,
            frontier_state_hash=runner.frontier_state_hash,
            execution_started=execution_started,
            execution_completed=execution_completed,
        )
        
        print(f"Replay bundle generated: {replay_bundle_path}")
        
        return replay_bundle_path
    
    def _load_config(self) -> U2Config:
        """Load configuration from original bundle."""
        config_path = self.original_bundle / "config" / "experiment_config.json"
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return U2Config(**config_dict)
    
    def verify(self, replay_bundle: Path) -> ValidationResult:
        """
        Verify replay against original.
        
        Args:
            replay_bundle: Path to replay provenance bundle
            
        Returns:
            ValidationResult
        """
        test_suite = ReplayTestSuite(self.original_bundle, replay_bundle)
        test_results = test_suite.run_all_tests()
        
        passed = all(r.passed for r in test_results)
        
        invariants_verified = [r.test_name for r in test_results if r.passed]
        invariants_violated = [r.test_name for r in test_results if not r.passed]
        
        report = test_suite.generate_report()
        
        return ValidationResult(
            passed=passed,
            test_results=test_results,
            invariants_verified=invariants_verified,
            invariants_violated=invariants_violated,
            report=report,
        )
```

### 5.2. Usage Example

```python
# Replay experiment
engine = ReplayEngine(original_bundle=Path("provenance_bundle_exp123"))
replay_bundle = engine.replay(output_dir=Path("replay_output"))

# Verify replay
result = engine.verify(replay_bundle)

# Print report
print(result.report)

# Check result
if result.passed:
    print("✓ Replay successful - determinism verified")
else:
    print("✗ Replay failed - determinism violated")
    print(f"Violated invariants: {result.invariants_violated}")
```

---

## 6. Implementation Checklist

### Phase 1: Test Suite
- [ ] Implement `ReplayTestResult` dataclass
- [ ] Implement `ReplayTestSuite` class
- [ ] Implement all test methods
- [ ] Test suite with mock bundles

### Phase 2: Conformance Validator
- [ ] Implement `ValidationResult` dataclass
- [ ] Implement `ConformanceValidator` class
- [ ] Test validator with mock bundles

### Phase 3: Replay Engine
- [ ] Implement `ReplayEngine` class
- [ ] Implement config loading
- [ ] Implement replay execution
- [ ] Implement verification

### Phase 4: End-to-End Testing
- [ ] Run original experiment
- [ ] Generate provenance bundle
- [ ] Replay experiment
- [ ] Verify all invariants
- [ ] Generate conformance report

---

**Status**: Plan complete. Ready for implementation.
