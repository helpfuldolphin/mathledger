# Task C1: Telemetry Runtime Implementation Plan

**Author**: Manus-C (Telemetry Architect)  
**Date**: 2025-12-06  
**Status**: Implementation Ready  
**Target**: `backend/verification/telemetry_runtime.py`

---

## 1. Overview

The **Telemetry Runtime** is responsible for executing Lean verification with comprehensive instrumentation, capturing all metrics needed for calibration, noise injection, and drift detection.

**Core Function**: `run_lean_with_monitoring(module_name, tier, timeout_s, context, master_seed, noise_config) → LeanVerificationTelemetry`

**Key Requirements**:
- Process sandboxing with resource limits
- High-frequency resource monitoring (CPU, memory)
- Lean output parsing with fallback for unknown formats
- Coordinated PRNG seeding for reproducibility
- Complete error taxonomy with stable error codes
- Retry logic for transient failures

---

## 2. Implementation Architecture

### 2.1 Module Structure

```
backend/verification/
├── telemetry_runtime.py          # Main telemetry runtime
├── lean_output_parser.py          # Lean output parsing with adapters
├── resource_monitor.py            # psutil-based resource monitoring
├── process_sandbox.py             # Process sandboxing and limits
├── error_taxonomy.py              # Error classification and retry logic
└── telemetry_schema.py            # LeanVerificationTelemetry dataclass
```

### 2.2 Dependencies

```python
# Standard library
import subprocess
import time
import uuid
import hashlib
import signal
import os
import re
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

# Third-party
import psutil  # pip install psutil
import yaml    # pip install pyyaml

# Internal
from backend.verification.error_codes import VerifierErrorCode, VerifierTier
from backend.verification.noise_sampler import DeterministicPRNG
```

---

## 3. Core Implementation: `run_lean_with_monitoring`

### 3.1 Function Signature

```python
def run_lean_with_monitoring(
    module_name: str,
    tier: VerifierTier,
    timeout_s: float,
    context: str,
    master_seed: int,
    noise_config: Optional[Dict[str, Any]] = None,
    working_dir: Optional[Path] = None,
    lean_version: Optional[str] = None,
) -> LeanVerificationTelemetry:
    """Run Lean verification with comprehensive monitoring.
    
    Args:
        module_name: Lean module to verify (e.g., "Mathlib.Algebra.Ring.Basic")
        tier: Verifier tier (FAST_NOISY, BALANCED, SLOW_PRECISE)
        timeout_s: Timeout in seconds
        context: Context string for hierarchical PRNG seeding
        master_seed: Master seed for deterministic noise injection
        noise_config: Optional noise configuration (if None, no noise injection)
        working_dir: Working directory for Lean build (default: current dir)
        lean_version: Lean version string (if None, auto-detect)
    
    Returns:
        LeanVerificationTelemetry with complete metrics
    
    Raises:
        TelemetryRuntimeError: If telemetry collection fails catastrophically
    """
```

### 3.2 Implementation Steps

**Step 1: Initialize Telemetry Record**

```python
    # Generate verification ID
    verification_id = str(uuid.uuid4())
    
    # Record start time
    start_time = time.time()
    timestamp = start_time
    
    # Auto-detect Lean version if not provided
    if lean_version is None:
        lean_version = detect_lean_version(working_dir)
    
    # Initialize metadata
    metadata = {
        "working_dir": str(working_dir) if working_dir else os.getcwd(),
        "master_seed": master_seed,
        "noise_config": noise_config,
    }
```

**Step 2: Set Up Process Sandbox**

```python
    # Create sandbox with resource limits
    sandbox = ProcessSandbox(
        timeout_s=timeout_s,
        memory_limit_mb=get_memory_limit_for_tier(tier),
        cpu_limit_percent=get_cpu_limit_for_tier(tier),
    )
    
    # Prepare Lean build command
    cmd = ["lake", "build", module_name]
    
    # Set up environment variables
    env = os.environ.copy()
    env["LEAN_PATH"] = str(working_dir) if working_dir else os.getcwd()
```

**Step 3: Start Process with Resource Monitoring**

```python
    # Start Lean process
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        cwd=working_dir,
        preexec_fn=sandbox.apply_limits,  # Apply resource limits before exec
    )
    
    # Initialize resource monitor
    monitor = ResourceMonitor(
        pid=proc.pid,
        sampling_interval_s=0.1,  # Sample every 100ms
    )
    
    # Start monitoring in background thread
    monitor.start()
```

**Step 4: Wait for Completion with Timeout**

```python
    try:
        # Wait for process completion with timeout
        stdout, stderr = proc.communicate(timeout=timeout_s)
        returncode = proc.returncode
        signal_num = None
        
    except subprocess.TimeoutExpired:
        # Timeout occurred, kill process
        proc.kill()
        stdout, stderr = proc.communicate()
        returncode = -signal.SIGKILL
        signal_num = signal.SIGKILL
        
    finally:
        # Stop resource monitoring
        monitor.stop()
        
        # Compute duration
        duration_ms = (time.time() - start_time) * 1000
```

**Step 5: Extract Resource Metrics**

```python
    # Get resource usage from monitor
    resource_metrics = monitor.get_metrics()
    
    cpu_time_ms = resource_metrics["cpu_time_ms"]
    memory_peak_mb = resource_metrics["memory_peak_mb"]
    memory_final_mb = resource_metrics["memory_final_mb"]
```

**Step 6: Parse Lean Output**

```python
    # Parse Lean output for tactic-level metrics
    parser = LeanOutputParser(lean_version=lean_version)
    
    try:
        lean_metrics = parser.parse(stderr)
        tactic_count = lean_metrics.get("tactic_count")
        tactic_depth = lean_metrics.get("tactic_depth")
        proof_size_bytes = lean_metrics.get("proof_size_bytes")
        search_nodes = lean_metrics.get("search_nodes")
    except LeanOutputParseError as e:
        # Parsing failed, use None for all metrics
        tactic_count = None
        tactic_depth = None
        proof_size_bytes = None
        search_nodes = None
        metadata["parse_error"] = str(e)
```

**Step 7: Classify Outcome**

```python
    # Classify outcome using error taxonomy
    outcome, ground_truth = classify_outcome(
        returncode=returncode,
        signal=signal_num,
        stderr=stderr,
        duration_ms=duration_ms,
        timeout_s=timeout_s,
        memory_peak_mb=memory_peak_mb,
        memory_limit_mb=sandbox.memory_limit_mb,
    )
    
    success = (outcome == VerifierErrorCode.VERIFIED)
```

**Step 8: Apply Noise Injection (if configured)**

```python
    # Initialize noise injection state
    noise_injected = False
    noise_type = None
    
    if noise_config is not None:
        # Create PRNG for noise decision
        prng = DeterministicPRNG(int_to_hex_seed(master_seed))
        prng_context = prng.for_path("noise", context)
        
        # Check if noise should be injected
        noise_decision = should_inject_noise(
            outcome=outcome,
            tier=tier,
            noise_config=noise_config,
            prng=prng_context,
        )
        
        if noise_decision["inject"]:
            # Inject noise by modifying outcome
            noise_injected = True
            noise_type = noise_decision["noise_type"]
            ground_truth = outcome.value  # Store original outcome
            outcome = noise_decision["outcome"]  # Replace with noisy outcome
            
            # Update success flag
            success = (outcome == VerifierErrorCode.VERIFIED)
```

**Step 9: Construct Telemetry Record**

```python
    # Construct telemetry record
    telemetry = LeanVerificationTelemetry(
        # Identity
        verification_id=verification_id,
        timestamp=timestamp,
        module_name=module_name,
        context=context,
        
        # Configuration
        tier=tier,
        timeout_s=timeout_s,
        lean_version=lean_version,
        
        # Outcome
        outcome=outcome,
        success=success,
        duration_ms=duration_ms,
        
        # Resource Usage
        cpu_time_ms=cpu_time_ms,
        memory_peak_mb=memory_peak_mb,
        memory_final_mb=memory_final_mb,
        
        # Lean-Specific Metrics
        tactic_count=tactic_count,
        tactic_depth=tactic_depth,
        proof_size_bytes=proof_size_bytes,
        search_nodes=search_nodes,
        
        # Failure Diagnostics
        stderr=stderr,
        returncode=returncode,
        signal=signal_num,
        
        # Noise Injection
        noise_injected=noise_injected,
        noise_type=noise_type,
        ground_truth=ground_truth,
        
        # Metadata
        metadata=metadata,
    )
    
    return telemetry
```

---

## 4. Process Sandboxing

### 4.1 ProcessSandbox Class

```python
@dataclass
class ProcessSandbox:
    """Process sandbox with resource limits."""
    
    timeout_s: float
    memory_limit_mb: float
    cpu_limit_percent: float
    
    def apply_limits(self) -> None:
        """Apply resource limits to current process (called via preexec_fn).
        
        This function is called in the child process before exec.
        """
        import resource
        
        # Set memory limit (RLIMIT_AS = address space)
        memory_limit_bytes = int(self.memory_limit_mb * 1024 * 1024)
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))
        
        # Set CPU time limit (RLIMIT_CPU)
        cpu_limit_s = int(self.timeout_s * 2)  # 2x timeout for safety
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit_s, cpu_limit_s))
        
        # Set nice level (lower priority)
        os.nice(10)


def get_memory_limit_for_tier(tier: VerifierTier) -> float:
    """Get memory limit in MB for tier."""
    limits = {
        VerifierTier.FAST_NOISY: 4096,    # 4 GB
        VerifierTier.BALANCED: 8192,      # 8 GB
        VerifierTier.SLOW_PRECISE: 16384, # 16 GB
    }
    return limits[tier]


def get_cpu_limit_for_tier(tier: VerifierTier) -> float:
    """Get CPU limit as percentage for tier."""
    limits = {
        VerifierTier.FAST_NOISY: 50.0,    # 50% of one core
        VerifierTier.BALANCED: 100.0,     # 100% of one core
        VerifierTier.SLOW_PRECISE: 200.0, # 200% (two cores)
    }
    return limits[tier]
```

---

## 5. Resource Monitoring

### 5.1 ResourceMonitor Class

```python
import threading
from collections import defaultdict

@dataclass
class ResourceMonitor:
    """High-frequency resource monitor using psutil."""
    
    pid: int
    sampling_interval_s: float
    
    def __post_init__(self):
        self.samples = defaultdict(list)
        self.running = False
        self.thread = None
        
    def start(self) -> None:
        """Start monitoring in background thread."""
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
    
    def stop(self) -> None:
        """Stop monitoring."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
    
    def _monitor_loop(self) -> None:
        """Monitoring loop (runs in background thread)."""
        try:
            process = psutil.Process(self.pid)
            
            while self.running:
                try:
                    # Sample CPU times
                    cpu_times = process.cpu_times()
                    self.samples["cpu_user"].append(cpu_times.user)
                    self.samples["cpu_system"].append(cpu_times.system)
                    
                    # Sample memory usage
                    memory_info = process.memory_info()
                    self.samples["memory_rss"].append(memory_info.rss / (1024 * 1024))  # MB
                    
                    # Sleep until next sample
                    time.sleep(self.sampling_interval_s)
                    
                except psutil.NoSuchProcess:
                    # Process terminated
                    break
                    
        except Exception as e:
            # Monitoring failed, log error but don't crash
            print(f"Resource monitoring error: {e}")
    
    def get_metrics(self) -> Dict[str, float]:
        """Get aggregated resource metrics."""
        if not self.samples["cpu_user"]:
            # No samples collected
            return {
                "cpu_time_ms": 0.0,
                "memory_peak_mb": 0.0,
                "memory_final_mb": 0.0,
            }
        
        # Compute CPU time (user + system)
        cpu_time_s = (
            (self.samples["cpu_user"][-1] - self.samples["cpu_user"][0]) +
            (self.samples["cpu_system"][-1] - self.samples["cpu_system"][0])
        )
        cpu_time_ms = cpu_time_s * 1000
        
        # Compute memory peak and final
        memory_peak_mb = max(self.samples["memory_rss"])
        memory_final_mb = self.samples["memory_rss"][-1]
        
        return {
            "cpu_time_ms": cpu_time_ms,
            "memory_peak_mb": memory_peak_mb,
            "memory_final_mb": memory_final_mb,
        }
```

---

## 6. Lean Output Parsing

### 6.1 LeanOutputParser Class

```python
@dataclass
class LeanMetrics:
    """Parsed Lean metrics."""
    tactic_count: Optional[int] = None
    tactic_depth: Optional[int] = None
    proof_size_bytes: Optional[int] = None
    search_nodes: Optional[int] = None


class LeanOutputParseError(Exception):
    """Raised when Lean output parsing fails."""
    pass


class LeanOutputParser:
    """Parser for Lean verification output with fallback adapters."""
    
    def __init__(self, lean_version: str):
        self.lean_version = lean_version
        
        # Select parser adapter based on Lean version
        if lean_version.startswith("4."):
            self.adapter = Lean4OutputAdapter()
        elif lean_version.startswith("3."):
            self.adapter = Lean3OutputAdapter()
        else:
            self.adapter = GenericLeanOutputAdapter()
    
    def parse(self, stderr: str) -> Dict[str, Optional[int]]:
        """Parse Lean stderr output.
        
        Args:
            stderr: Lean stderr output
        
        Returns:
            Dict with tactic_count, tactic_depth, proof_size_bytes, search_nodes
        
        Raises:
            LeanOutputParseError: If parsing fails
        """
        try:
            metrics = self.adapter.parse(stderr)
            return {
                "tactic_count": metrics.tactic_count,
                "tactic_depth": metrics.tactic_depth,
                "proof_size_bytes": metrics.proof_size_bytes,
                "search_nodes": metrics.search_nodes,
            }
        except Exception as e:
            raise LeanOutputParseError(f"Failed to parse Lean output: {e}")


class Lean4OutputAdapter:
    """Parser adapter for Lean 4 output."""
    
    def parse(self, stderr: str) -> LeanMetrics:
        metrics = LeanMetrics()
        
        # Parse tactic count (count lines with "[Tactic]" prefix)
        metrics.tactic_count = stderr.count("[Tactic]")
        
        # Parse tactic depth (extract from "[Tactic.depth N]" lines)
        depth_matches = re.findall(r"\[Tactic\.depth (\d+)\]", stderr)
        if depth_matches:
            metrics.tactic_depth = max(int(d) for d in depth_matches)
        
        # Parse proof size (extract from "[Kernel] proof size: N bytes")
        size_match = re.search(r"\[Kernel\] proof size: (\d+) bytes", stderr)
        if size_match:
            metrics.proof_size_bytes = int(size_match.group(1))
        
        # Parse search nodes (extract from "[Search] explored N nodes")
        nodes_match = re.search(r"\[Search\] explored (\d+) nodes", stderr)
        if nodes_match:
            metrics.search_nodes = int(nodes_match.group(1))
        
        return metrics


class Lean3OutputAdapter:
    """Parser adapter for Lean 3 output."""
    
    def parse(self, stderr: str) -> LeanMetrics:
        metrics = LeanMetrics()
        
        # Lean 3 has different output format
        # Count tactic invocations (heuristic: lines with "tactic" keyword)
        metrics.tactic_count = len(re.findall(r"\btactic\b", stderr, re.IGNORECASE))
        
        # Other metrics not available in Lean 3
        metrics.tactic_depth = None
        metrics.proof_size_bytes = None
        metrics.search_nodes = None
        
        return metrics


class GenericLeanOutputAdapter:
    """Fallback parser adapter for unknown Lean versions."""
    
    def parse(self, stderr: str) -> LeanMetrics:
        metrics = LeanMetrics()
        
        # Use heuristics for unknown format
        # Count lines (rough proxy for tactic count)
        metrics.tactic_count = len(stderr.splitlines())
        
        # Other metrics unavailable
        metrics.tactic_depth = None
        metrics.proof_size_bytes = None
        metrics.search_nodes = None
        
        return metrics


def detect_lean_version(working_dir: Optional[Path] = None) -> str:
    """Detect Lean version by running `lean --version`."""
    try:
        result = subprocess.run(
            ["lean", "--version"],
            capture_output=True,
            text=True,
            cwd=working_dir,
            timeout=5.0,
        )
        
        # Parse version from output (e.g., "Lean (version 4.3.0)")
        match = re.search(r"version\s+(\d+\.\d+\.\d+)", result.stdout)
        if match:
            return match.group(1)
        else:
            return "unknown"
            
    except Exception:
        return "unknown"
```

---

## 7. Error Taxonomy and Classification

### 7.1 classify_outcome Function

```python
def classify_outcome(
    returncode: int,
    signal: Optional[int],
    stderr: str,
    duration_ms: float,
    timeout_s: float,
    memory_peak_mb: float,
    memory_limit_mb: float,
) -> Tuple[VerifierErrorCode, Optional[str]]:
    """Classify verifier outcome using error taxonomy.
    
    Args:
        returncode: Process return code
        signal: Signal number if killed (e.g., SIGKILL)
        stderr: Lean stderr output
        duration_ms: Actual duration in milliseconds
        timeout_s: Configured timeout in seconds
        memory_peak_mb: Peak memory usage in MB
        memory_limit_mb: Memory limit in MB
    
    Returns:
        Tuple of (outcome, ground_truth) where ground_truth is None if no noise
    """
    
    # Case 1: Success (returncode 0)
    if returncode == 0:
        return VerifierErrorCode.VERIFIED, None
    
    # Case 2: Timeout (killed by SIGKILL or duration exceeds timeout)
    if signal == signal.SIGKILL or duration_ms > timeout_s * 1000:
        return VerifierErrorCode.VERIFIER_TIMEOUT, None
    
    # Case 3: Memory exhaustion (OOM or near limit)
    if "out of memory" in stderr.lower() or "std::bad_alloc" in stderr:
        return VerifierErrorCode.MEMORY_LIMIT_EXCEEDED, None
    
    if memory_peak_mb > memory_limit_mb * 0.95:  # Within 5% of limit
        return VerifierErrorCode.MEMORY_LIMIT_EXCEEDED, None
    
    # Case 4: Proof invalid (explicit error messages)
    invalid_patterns = [
        "type mismatch",
        "failed to synthesize",
        "tactic failed",
        "unsolved goals",
        "declaration uses sorry",
    ]
    
    for pattern in invalid_patterns:
        if pattern in stderr.lower():
            return VerifierErrorCode.PROOF_INVALID, None
    
    # Case 5: Proof incomplete (partial proof)
    if "sorry" in stderr.lower() or "admit" in stderr.lower():
        return VerifierErrorCode.PROOF_INCOMPLETE, None
    
    # Case 6: Verifier internal error (crash, assertion failure)
    error_patterns = [
        "internal error",
        "assertion failed",
        "segmentation fault",
        "panic",
    ]
    
    for pattern in error_patterns:
        if pattern in stderr.lower():
            return VerifierErrorCode.VERIFIER_INTERNAL_ERROR, None
    
    # Case 7: Unknown failure (default)
    return VerifierErrorCode.PROOF_INVALID, None
```

### 7.2 Retry Logic

```python
@dataclass
class RetryPolicy:
    """Retry policy for transient failures."""
    
    max_retries: int = 3
    retry_delay_s: float = 1.0
    retry_on_errors: List[VerifierErrorCode] = field(default_factory=lambda: [
        VerifierErrorCode.VERIFIER_INTERNAL_ERROR,
        VerifierErrorCode.VERIFIER_TIMEOUT,  # Retry timeouts once
    ])
    
    def should_retry(self, outcome: VerifierErrorCode, attempt: int) -> bool:
        """Check if outcome should be retried."""
        return (
            attempt < self.max_retries and
            outcome in self.retry_on_errors
        )


def run_lean_with_retry(
    module_name: str,
    tier: VerifierTier,
    timeout_s: float,
    context: str,
    master_seed: int,
    retry_policy: Optional[RetryPolicy] = None,
    **kwargs,
) -> LeanVerificationTelemetry:
    """Run Lean verification with retry logic.
    
    Args:
        module_name: Lean module to verify
        tier: Verifier tier
        timeout_s: Timeout in seconds
        context: Context string for PRNG seeding
        master_seed: Master seed
        retry_policy: Retry policy (if None, no retries)
        **kwargs: Additional arguments for run_lean_with_monitoring
    
    Returns:
        LeanVerificationTelemetry from successful attempt or final failure
    """
    
    if retry_policy is None:
        retry_policy = RetryPolicy(max_retries=0)  # No retries
    
    attempt = 0
    last_telemetry = None
    
    while True:
        # Run verification
        telemetry = run_lean_with_monitoring(
            module_name=module_name,
            tier=tier,
            timeout_s=timeout_s,
            context=context,
            master_seed=master_seed,
            **kwargs,
        )
        
        last_telemetry = telemetry
        attempt += 1
        
        # Check if retry needed
        if not retry_policy.should_retry(telemetry.outcome, attempt):
            break
        
        # Wait before retry
        time.sleep(retry_policy.retry_delay_s)
    
    # Add retry metadata
    last_telemetry.metadata["retry_attempts"] = attempt
    
    return last_telemetry
```

---

## 8. Coordinated PRNG Seeding Strategy

### 8.1 Seeding Hierarchy

```
master_seed (experiment-level)
    ↓
context_seed = hash(master_seed, context)
    ↓
noise_seed = hash(context_seed, "noise", noise_type)
    ↓
sample_seed = hash(noise_seed, item, cycle)
```

### 8.2 Implementation

```python
def int_to_hex_seed(seed: int) -> str:
    """Convert integer seed to hex string for DeterministicPRNG."""
    return f"{seed:016x}"


def context_to_seed(master_seed: int, context: str) -> int:
    """Derive context-specific seed from master seed and context string."""
    # Hash master_seed + context using SHA256
    h = hashlib.sha256()
    h.update(master_seed.to_bytes(8, byteorder="big"))
    h.update(context.encode("utf-8"))
    
    # Take first 8 bytes as integer seed
    return int.from_bytes(h.digest()[:8], byteorder="big")


def should_inject_noise(
    outcome: VerifierErrorCode,
    tier: VerifierTier,
    noise_config: Dict[str, Any],
    prng: DeterministicPRNG,
) -> Dict[str, Any]:
    """Determine if noise should be injected.
    
    Args:
        outcome: Original verifier outcome
        tier: Verifier tier
        noise_config: Noise configuration dict
        prng: PRNG for noise decision (already seeded with context)
    
    Returns:
        Dict with "inject" (bool), "noise_type" (str), "outcome" (VerifierErrorCode)
    """
    
    # Extract noise rates for tier
    tier_config = noise_config.get(tier.value, {})
    timeout_rate = tier_config.get("timeout_rate", 0.0)
    spurious_fail_rate = tier_config.get("spurious_fail_rate", 0.0)
    spurious_pass_rate = tier_config.get("spurious_pass_rate", 0.0)
    
    # Sample noise decision
    prng_timeout = prng.for_path("timeout")
    prng_spurious_fail = prng.for_path("spurious_fail")
    prng_spurious_pass = prng.for_path("spurious_pass")
    
    # Check timeout injection (always possible)
    if prng_timeout.random() < timeout_rate:
        return {
            "inject": True,
            "noise_type": "timeout",
            "outcome": VerifierErrorCode.VERIFIER_TIMEOUT,
        }
    
    # Check spurious failure (only if original outcome is VERIFIED)
    if outcome == VerifierErrorCode.VERIFIED:
        if prng_spurious_fail.random() < spurious_fail_rate:
            return {
                "inject": True,
                "noise_type": "spurious_fail",
                "outcome": VerifierErrorCode.PROOF_INVALID,
            }
    
    # Check spurious pass (only if original outcome is failure)
    if outcome in [VerifierErrorCode.PROOF_INVALID, VerifierErrorCode.PROOF_INCOMPLETE]:
        if prng_spurious_pass.random() < spurious_pass_rate:
            return {
                "inject": True,
                "noise_type": "spurious_pass",
                "outcome": VerifierErrorCode.VERIFIED,
            }
    
    # No noise injection
    return {
        "inject": False,
        "noise_type": None,
        "outcome": outcome,
    }
```

---

## 9. Testing Strategy

### 9.1 Unit Tests

```python
# tests/verification/test_telemetry_runtime.py

def test_run_lean_with_monitoring_success():
    """Test successful verification."""
    telemetry = run_lean_with_monitoring(
        module_name="Mathlib.Data.Nat.Basic",
        tier=VerifierTier.BALANCED,
        timeout_s=60.0,
        context="test_success",
        master_seed=12345,
    )
    
    assert telemetry.outcome == VerifierErrorCode.VERIFIED
    assert telemetry.success is True
    assert telemetry.duration_ms > 0
    assert telemetry.cpu_time_ms > 0
    assert telemetry.memory_peak_mb > 0


def test_run_lean_with_monitoring_timeout():
    """Test timeout handling."""
    telemetry = run_lean_with_monitoring(
        module_name="Mathlib.Algebra.Ring.Basic",  # Assume this times out
        tier=VerifierTier.FAST_NOISY,
        timeout_s=1.0,  # Very short timeout
        context="test_timeout",
        master_seed=12345,
    )
    
    assert telemetry.outcome == VerifierErrorCode.VERIFIER_TIMEOUT
    assert telemetry.success is False
    assert telemetry.duration_ms >= 1000  # At least 1 second


def test_noise_injection_determinism():
    """Test that noise injection is deterministic."""
    # Run twice with same seed
    telemetry1 = run_lean_with_monitoring(
        module_name="Mathlib.Data.Nat.Basic",
        tier=VerifierTier.BALANCED,
        timeout_s=60.0,
        context="test_determinism",
        master_seed=12345,
        noise_config={"balanced": {"timeout_rate": 0.5}},
    )
    
    telemetry2 = run_lean_with_monitoring(
        module_name="Mathlib.Data.Nat.Basic",
        tier=VerifierTier.BALANCED,
        timeout_s=60.0,
        context="test_determinism",
        master_seed=12345,
        noise_config={"balanced": {"timeout_rate": 0.5}},
    )
    
    # Outcomes should be identical
    assert telemetry1.outcome == telemetry2.outcome
    assert telemetry1.noise_injected == telemetry2.noise_injected
    assert telemetry1.noise_type == telemetry2.noise_type


def test_resource_monitoring():
    """Test resource monitoring accuracy."""
    telemetry = run_lean_with_monitoring(
        module_name="Mathlib.Data.Nat.Basic",
        tier=VerifierTier.BALANCED,
        timeout_s=60.0,
        context="test_resources",
        master_seed=12345,
    )
    
    # Resource metrics should be non-zero
    assert telemetry.cpu_time_ms > 0
    assert telemetry.memory_peak_mb > 0
    assert telemetry.memory_final_mb > 0
    
    # Peak should be >= final
    assert telemetry.memory_peak_mb >= telemetry.memory_final_mb
```

### 9.2 Integration Tests

```python
def test_end_to_end_verification():
    """Test end-to-end verification with all components."""
    # Run verification with noise injection, retry, and full telemetry
    telemetry = run_lean_with_retry(
        module_name="Mathlib.Data.Nat.Basic",
        tier=VerifierTier.BALANCED,
        timeout_s=60.0,
        context="test_e2e",
        master_seed=12345,
        noise_config={
            "balanced": {
                "timeout_rate": 0.1,
                "spurious_fail_rate": 0.05,
                "spurious_pass_rate": 0.02,
            }
        },
        retry_policy=RetryPolicy(max_retries=3),
    )
    
    # Verify telemetry completeness
    assert telemetry.verification_id is not None
    assert telemetry.timestamp > 0
    assert telemetry.module_name == "Mathlib.Data.Nat.Basic"
    assert telemetry.tier == VerifierTier.BALANCED
    assert telemetry.outcome is not None
    assert "retry_attempts" in telemetry.metadata
```

---

## 10. Performance Considerations

### 10.1 Overhead Analysis

**Resource Monitoring Overhead**:
- psutil sampling every 100ms: ~0.1% CPU overhead
- Background thread: ~1 MB memory overhead
- Total impact: negligible for verification workloads

**Lean Output Parsing Overhead**:
- Regex parsing: O(n) where n = stderr length
- Typical stderr: 1-10 KB
- Parsing time: <1 ms
- Total impact: negligible

**Noise Injection Overhead**:
- PRNG operations: ~10 μs per random() call
- 3 random() calls per verification (timeout, spurious_fail, spurious_pass)
- Total: ~30 μs
- Total impact: negligible

**Overall Overhead**: <1% of verification time

### 10.2 Optimization Strategies

**Strategy 1: Lazy Parsing** — Only parse Lean output if tactic metrics are needed (e.g., skip parsing for timeouts)

**Strategy 2: Sampling Rate Adjustment** — Reduce psutil sampling rate to 200ms or 500ms for long-running verifications

**Strategy 3: Async Logging** — Write telemetry to log asynchronously to avoid blocking verification loop

---

## 11. Error Handling and Edge Cases

### 11.1 Edge Cases

**Edge Case 1: Process Dies Immediately** — If Lean process dies before monitoring starts, resource metrics will be zero. Handle gracefully with default values.

**Edge Case 2: Lean Output Format Changes** — If Lean version changes and output format is incompatible, parsing will fail. Use fallback adapter with heuristics.

**Edge Case 3: Memory Limit Exceeded** — If process exceeds memory limit, it will be killed by OS. Detect via returncode or signal and classify as MEMORY_LIMIT_EXCEEDED.

**Edge Case 4: Timeout Exactly at Boundary** — If verification completes exactly at timeout boundary, classify based on returncode (success if 0, timeout if killed).

**Edge Case 5: PRNG Seed Collision** — If two contexts hash to same seed, noise decisions will be identical. Use context string with sufficient entropy (e.g., include cycle number, item name).

### 11.2 Error Recovery

**Recovery 1: Parsing Failure** — If Lean output parsing fails, set tactic metrics to None and continue with other metrics.

**Recovery 2: Resource Monitoring Failure** — If psutil fails (e.g., process dies before monitoring starts), set resource metrics to zero and continue.

**Recovery 3: Sandbox Failure** — If resource limits cannot be applied, log warning and continue without limits.

**Recovery 4: Noise Config Missing** — If noise_config is None or missing tier, disable noise injection and continue.

---

## 12. Deployment Checklist

- [ ] Implement `LeanVerificationTelemetry` dataclass in `telemetry_schema.py`
- [ ] Implement `ProcessSandbox` class in `process_sandbox.py`
- [ ] Implement `ResourceMonitor` class in `resource_monitor.py`
- [ ] Implement `LeanOutputParser` with adapters in `lean_output_parser.py`
- [ ] Implement `classify_outcome` in `error_taxonomy.py`
- [ ] Implement `run_lean_with_monitoring` in `telemetry_runtime.py`
- [ ] Implement `run_lean_with_retry` in `telemetry_runtime.py`
- [ ] Implement `should_inject_noise` in `telemetry_runtime.py`
- [ ] Write unit tests for all components
- [ ] Write integration tests for end-to-end verification
- [ ] Test on real Lean proofs (Mathlib, curriculum)
- [ ] Benchmark overhead and optimize if needed
- [ ] Document all functions with docstrings
- [ ] Add logging for debugging (use Python logging module)
- [ ] Create example usage scripts

---

**Manus-C — Telemetry Architect**  
*"Every packet accounted for, every signal explained."*

**Status**: Implementation Ready  
**Next**: Task C2 (Calibration Runner)
