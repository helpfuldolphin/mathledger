# Phase III: Noise Replay Log Specification

**Author**: Manus-C (Telemetry Architect)  
**Date**: 2025-12-06  
**Mission**: Define deterministic replay semantics for noise injection debugging and analysis

---

## 1. Overview

**Noise replay logs** enable deterministic reproduction of verifier behavior for debugging, analysis, and validation. Given a replay log, the system can reconstruct the exact sequence of noise decisions and verifier outcomes, enabling:

**Use Case 1: Debugging** — Reproduce rare noise patterns that cause unexpected behavior  
**Use Case 2: Differential Analysis** — Compare outcomes with and without noise injection  
**Use Case 3: Anomaly Detection** — Identify unusual noise patterns that deviate from expected distributions  
**Use Case 4: Validation** — Verify that noise injection is deterministic and reproducible

---

## 2. Replay Log Schema

### 2.1 Log Entry Format

Each log entry captures a single noise decision:

```json
{
  "version": "1.0",
  "entry_type": "noise_decision",
  "timestamp": 1733518800.123,
  "verification_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "cycle_id": 42,
  "item": "Mathlib.Algebra.Ring.Basic.theorem_1",
  "context": "cycle_42_item_Mathlib.Algebra.Ring.Basic.theorem_1",
  "tier": "balanced",
  "seed": {
    "master_seed": 12345,
    "context_hash": "8f3a9b2c1d4e5f67",
    "noise_type_path": "timeout"
  },
  "noise_decision": {
    "noise_type": "timeout",
    "injected": true,
    "probability": 0.05,
    "sampled_value": 0.0234,
    "decision": "inject"
  },
  "noise_parameters": {
    "timeout_duration_ms": 1234.5,
    "distribution": "lognormal",
    "distribution_params": {
      "mu": 5.2,
      "sigma": 0.8
    }
  },
  "verifier_outcome": {
    "outcome": "VERIFIER_TIMEOUT",
    "success": false,
    "duration_ms": 1234.5,
    "noise_injected": true,
    "ground_truth": "VERIFIED"
  },
  "metadata": {
    "lean_version": "4.3.0",
    "experiment_id": "uplift_phase2_run_001",
    "slice_name": "arithmetic_simple"
  }
}
```

### 2.2 Schema Fields

**version**: Schema version (semantic versioning)

**entry_type**: Type of log entry (`noise_decision`, `verifier_outcome`, `policy_update`)

**timestamp**: Unix timestamp (seconds since epoch) with millisecond precision

**verification_id**: Unique identifier for this verification call (UUID)

**cycle_id**: Cycle number in U2 experiment

**item**: Item being verified (e.g., proof module name)

**context**: Context string used for hierarchical seeding

**tier**: Verifier tier (`fast_noisy`, `balanced`, `slow_precise`)

**seed**: Seed information for reproducibility
- `master_seed`: Master seed for entire experiment
- `context_hash`: Hash of context string (for verification)
- `noise_type_path`: Path components for PRNG derivation

**noise_decision**: Noise injection decision
- `noise_type`: Type of noise (`timeout`, `spurious_fail`, `spurious_pass`)
- `injected`: Whether noise was injected (boolean)
- `probability`: Configured noise probability (0-1)
- `sampled_value`: Random value sampled from PRNG (0-1)
- `decision`: Decision result (`inject` or `no_inject`)

**noise_parameters**: Parameters for noise injection (if injected)
- `timeout_duration_ms`: Sampled timeout duration
- `distribution`: Distribution type (`uniform`, `exponential`, `lognormal`, etc.)
- `distribution_params`: Distribution parameters

**verifier_outcome**: Final verifier outcome
- `outcome`: VerifierErrorCode
- `success`: Boolean success flag
- `duration_ms`: Actual duration
- `noise_injected`: Whether noise was injected
- `ground_truth`: Ground truth outcome (if known)

**metadata**: Additional context
- `lean_version`: Lean version
- `experiment_id`: Experiment identifier
- `slice_name`: Slice name

---

## 3. Deterministic Replay Semantics

### 3.1 Replay Algorithm

Given a replay log, reconstruct the exact sequence of noise decisions:

```python
def replay_noise_log(
    log_entries: List[Dict[str, Any]],
) -> List[VerifierOutcome]:
    """Replay noise decisions from log.
    
    Args:
        log_entries: List of log entries (parsed JSON)
    
    Returns:
        List of reconstructed verifier outcomes
    """
    
    outcomes = []
    
    for entry in log_entries:
        # Extract seed information
        master_seed = entry["seed"]["master_seed"]
        context = entry["context"]
        noise_type = entry["noise_decision"]["noise_type"]
        
        # Reconstruct PRNG
        prng = DeterministicPRNG(int_to_hex_seed(master_seed))
        prng_noise = prng.for_path(noise_type, context)
        
        # Replay noise decision
        sampled_value = prng_noise.random()
        probability = entry["noise_decision"]["probability"]
        injected = sampled_value < probability
        
        # Verify against log
        assert sampled_value == entry["noise_decision"]["sampled_value"], \
            f"Sampled value mismatch: {sampled_value} vs {entry['noise_decision']['sampled_value']}"
        assert injected == entry["noise_decision"]["injected"], \
            f"Injection decision mismatch: {injected} vs {entry['noise_decision']['injected']}"
        
        # Reconstruct outcome
        if injected:
            if noise_type == "timeout":
                # Replay timeout duration sampling
                dist = entry["noise_parameters"]["distribution"]
                params = entry["noise_parameters"]["distribution_params"]
                duration_ms = sample_from_distribution(prng_noise, dist, params)
                
                outcome = timeout_outcome(
                    duration_ms=duration_ms,
                    tier=VerifierTier[entry["tier"].upper()],
                    noise_injected=True,
                )
            elif noise_type == "spurious_fail":
                outcome = spurious_fail_outcome(
                    duration_ms=entry["verifier_outcome"]["duration_ms"],
                    tier=VerifierTier[entry["tier"].upper()],
                )
            elif noise_type == "spurious_pass":
                outcome = spurious_pass_outcome(
                    duration_ms=entry["verifier_outcome"]["duration_ms"],
                    tier=VerifierTier[entry["tier"].upper()],
                )
        else:
            # No noise injected, use ground truth
            if entry["verifier_outcome"]["ground_truth"] == "VERIFIED":
                outcome = verified_outcome(
                    duration_ms=entry["verifier_outcome"]["duration_ms"],
                    tier=VerifierTier[entry["tier"].upper()],
                )
            else:
                outcome = proof_invalid_outcome(
                    duration_ms=entry["verifier_outcome"]["duration_ms"],
                    tier=VerifierTier[entry["tier"].upper()],
                )
        
        outcomes.append(outcome)
    
    return outcomes
```

### 3.2 Verification

Replay must produce **identical outcomes** to original run:

```python
def verify_replay(
    original_outcomes: List[VerifierOutcome],
    replayed_outcomes: List[VerifierOutcome],
) -> bool:
    """Verify that replayed outcomes match original outcomes.
    
    Args:
        original_outcomes: Outcomes from original run
        replayed_outcomes: Outcomes from replay
    
    Returns:
        True if all outcomes match, False otherwise
    """
    
    if len(original_outcomes) != len(replayed_outcomes):
        return False
    
    for orig, replay in zip(original_outcomes, replayed_outcomes):
        if orig.error_code != replay.error_code:
            return False
        if orig.success != replay.success:
            return False
        if orig.noise_injected != replay.noise_injected:
            return False
        if abs(orig.duration_ms - replay.duration_ms) > 0.001:  # 1μs tolerance
            return False
    
    return True
```

---

## 4. Differential Debugging Methodology

### 4.1 Counterfactual Analysis

Compare outcomes with and without noise injection:

```python
def counterfactual_analysis(
    log_entries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Perform counterfactual analysis: what if noise was not injected?
    
    Args:
        log_entries: Replay log entries
    
    Returns:
        Analysis report with outcome differences
    """
    
    # Replay with noise (original)
    outcomes_with_noise = replay_noise_log(log_entries)
    
    # Replay without noise (counterfactual)
    outcomes_without_noise = []
    for entry in log_entries:
        # Use ground truth outcome
        if entry["verifier_outcome"]["ground_truth"] == "VERIFIED":
            outcome = verified_outcome(
                duration_ms=entry["verifier_outcome"]["duration_ms"],
                tier=VerifierTier[entry["tier"].upper()],
            )
        else:
            outcome = proof_invalid_outcome(
                duration_ms=entry["verifier_outcome"]["duration_ms"],
                tier=VerifierTier[entry["tier"].upper()],
            )
        outcomes_without_noise.append(outcome)
    
    # Compute differences
    differences = []
    for i, (with_noise, without_noise) in enumerate(zip(outcomes_with_noise, outcomes_without_noise)):
        if with_noise.error_code != without_noise.error_code:
            differences.append({
                "index": i,
                "item": log_entries[i]["item"],
                "with_noise": with_noise.error_code.value,
                "without_noise": without_noise.error_code.value,
                "noise_type": log_entries[i]["noise_decision"]["noise_type"],
            })
    
    return {
        "total_outcomes": len(outcomes_with_noise),
        "differences": len(differences),
        "difference_rate": len(differences) / len(outcomes_with_noise),
        "difference_details": differences,
    }
```

### 4.2 Noise Impact Analysis

Measure the impact of noise on RFL policy:

```python
def noise_impact_analysis(
    log_entries: List[Dict[str, Any]],
    initial_policy: Dict[str, float],
    learning_rate: float,
) -> Dict[str, Any]:
    """Analyze impact of noise on RFL policy.
    
    Args:
        log_entries: Replay log entries
        initial_policy: Initial policy distribution
        learning_rate: RFL learning rate
    
    Returns:
        Analysis report with policy divergence metrics
    """
    
    # Simulate policy updates with noise
    policy_with_noise = initial_policy.copy()
    for entry in log_entries:
        outcome = VerifierErrorCode[entry["verifier_outcome"]["outcome"]]
        item = entry["item"]
        policy_with_noise = update_rfl_policy_noisy(
            policy_with_noise,
            item,
            outcome,
            theta_spurious_fail=0.02,
            theta_spurious_pass=0.01,
            theta_timeout=0.05,
            learning_rate=learning_rate,
            abstention_rate=0.05,
        )
    
    # Simulate policy updates without noise (ground truth)
    policy_without_noise = initial_policy.copy()
    for entry in log_entries:
        ground_truth = entry["verifier_outcome"]["ground_truth"]
        outcome = VerifierErrorCode.VERIFIED if ground_truth == "VERIFIED" else VerifierErrorCode.PROOF_INVALID
        item = entry["item"]
        policy_without_noise = update_rfl_policy_noisy(
            policy_without_noise,
            item,
            outcome,
            theta_spurious_fail=0.0,
            theta_spurious_pass=0.0,
            theta_timeout=0.0,
            learning_rate=learning_rate,
            abstention_rate=0.0,
        )
    
    # Compute policy divergence (KL divergence)
    kl_divergence = sum(
        policy_without_noise[item] * np.log(policy_without_noise[item] / policy_with_noise[item])
        for item in policy_without_noise.keys()
        if policy_with_noise[item] > 0
    )
    
    return {
        "kl_divergence": kl_divergence,
        "policy_with_noise": policy_with_noise,
        "policy_without_noise": policy_without_noise,
    }
```

---

## 5. Noise Anomaly Detection

### 5.1 Anomaly Types

**Anomaly 1: Unexpected Noise Rate** — Empirical noise rate deviates significantly from configured rate

**Anomaly 2: Correlated Noise** — Noise events are correlated across items (should be independent)

**Anomaly 3: Temporal Clustering** — Noise events cluster in time (should be uniformly distributed)

**Anomaly 4: Seed Collision** — Multiple items have identical noise decisions (seed collision)

**Anomaly 5: Distribution Mismatch** — Timeout durations do not match configured distribution

### 5.2 Detection Algorithms

**Algorithm 1: Chi-Square Test for Noise Rate**

```python
def detect_noise_rate_anomaly(
    log_entries: List[Dict[str, Any]],
    configured_rate: float,
    significance_level: float = 0.05,
) -> Dict[str, Any]:
    """Detect anomalies in noise injection rate using chi-square test.
    
    Args:
        log_entries: Replay log entries
        configured_rate: Configured noise rate (0-1)
        significance_level: Significance level for hypothesis test
    
    Returns:
        Detection result with p-value and anomaly flag
    """
    
    # Count noise injections
    n_total = len(log_entries)
    n_injected = sum(1 for entry in log_entries if entry["noise_decision"]["injected"])
    
    # Expected counts
    expected_injected = n_total * configured_rate
    expected_not_injected = n_total * (1 - configured_rate)
    
    # Chi-square statistic
    chi_square = (
        (n_injected - expected_injected) ** 2 / expected_injected +
        (n_total - n_injected - expected_not_injected) ** 2 / expected_not_injected
    )
    
    # P-value (chi-square distribution with 1 degree of freedom)
    from scipy.stats import chi2
    p_value = 1 - chi2.cdf(chi_square, df=1)
    
    # Anomaly if p-value < significance level
    is_anomaly = p_value < significance_level
    
    return {
        "n_total": n_total,
        "n_injected": n_injected,
        "empirical_rate": n_injected / n_total,
        "configured_rate": configured_rate,
        "chi_square": chi_square,
        "p_value": p_value,
        "is_anomaly": is_anomaly,
    }
```

**Algorithm 2: Runs Test for Independence**

```python
def detect_correlation_anomaly(
    log_entries: List[Dict[str, Any]],
    significance_level: float = 0.05,
) -> Dict[str, Any]:
    """Detect correlation in noise decisions using runs test.
    
    Args:
        log_entries: Replay log entries (must be time-ordered)
        significance_level: Significance level for hypothesis test
    
    Returns:
        Detection result with runs test statistic and anomaly flag
    """
    
    # Extract binary sequence of noise decisions
    sequence = [1 if entry["noise_decision"]["injected"] else 0 for entry in log_entries]
    
    # Count runs (consecutive sequences of same value)
    runs = 1
    for i in range(1, len(sequence)):
        if sequence[i] != sequence[i-1]:
            runs += 1
    
    # Expected runs under independence
    n_total = len(sequence)
    n_ones = sum(sequence)
    n_zeros = n_total - n_ones
    
    expected_runs = (2 * n_ones * n_zeros) / n_total + 1
    variance_runs = (2 * n_ones * n_zeros * (2 * n_ones * n_zeros - n_total)) / (n_total ** 2 * (n_total - 1))
    
    # Z-statistic
    z_stat = (runs - expected_runs) / np.sqrt(variance_runs)
    
    # P-value (two-tailed test)
    from scipy.stats import norm
    p_value = 2 * (1 - norm.cdf(abs(z_stat)))
    
    # Anomaly if p-value < significance level
    is_anomaly = p_value < significance_level
    
    return {
        "n_total": n_total,
        "runs": runs,
        "expected_runs": expected_runs,
        "z_stat": z_stat,
        "p_value": p_value,
        "is_anomaly": is_anomaly,
    }
```

**Algorithm 3: Kolmogorov-Smirnov Test for Distribution**

```python
def detect_distribution_anomaly(
    log_entries: List[Dict[str, Any]],
    configured_distribution: str,
    configured_params: Dict[str, float],
    significance_level: float = 0.05,
) -> Dict[str, Any]:
    """Detect anomalies in timeout duration distribution using KS test.
    
    Args:
        log_entries: Replay log entries
        configured_distribution: Configured distribution type
        configured_params: Configured distribution parameters
        significance_level: Significance level for hypothesis test
    
    Returns:
        Detection result with KS statistic and anomaly flag
    """
    
    # Extract timeout durations
    durations = [
        entry["noise_parameters"]["timeout_duration_ms"]
        for entry in log_entries
        if entry["noise_decision"]["injected"] and entry["noise_decision"]["noise_type"] == "timeout"
    ]
    
    if len(durations) == 0:
        return {"is_anomaly": False, "reason": "no_timeout_data"}
    
    # Get theoretical CDF
    from scipy import stats
    if configured_distribution == "exponential":
        dist = stats.expon(scale=configured_params["mean"])
    elif configured_distribution == "lognormal":
        dist = stats.lognorm(s=configured_params["sigma"], scale=np.exp(configured_params["mu"]))
    elif configured_distribution == "gamma":
        dist = stats.gamma(a=configured_params["alpha"], scale=1/configured_params["beta"])
    else:
        return {"is_anomaly": False, "reason": "unknown_distribution"}
    
    # Kolmogorov-Smirnov test
    ks_stat, p_value = stats.kstest(durations, dist.cdf)
    
    # Anomaly if p-value < significance level
    is_anomaly = p_value < significance_level
    
    return {
        "n_samples": len(durations),
        "ks_stat": ks_stat,
        "p_value": p_value,
        "is_anomaly": is_anomaly,
    }
```

---

## 6. Log Storage and Indexing

### 6.1 Storage Format

**Format 1: JSONL (JSON Lines)** — One JSON object per line, append-only

```
{"version": "1.0", "entry_type": "noise_decision", ...}
{"version": "1.0", "entry_type": "noise_decision", ...}
{"version": "1.0", "entry_type": "noise_decision", ...}
```

**Format 2: Parquet** — Columnar storage for efficient querying

**Format 3: SQLite** — Relational database for complex queries

### 6.2 Indexing

Create indices for fast querying:

**Index 1: Timestamp** — For time-range queries

**Index 2: Item** — For item-specific queries

**Index 3: Noise Type** — For noise-type-specific queries

**Index 4: Verification ID** — For single-verification queries

### 6.3 Compression

Compress logs using:

**Compression 1: gzip** — Standard compression (10x reduction)

**Compression 2: zstd** — Fast compression (8x reduction, 3x faster than gzip)

**Compression 3: Delta encoding** — For timestamp and duration fields

---

## 7. Implementation

### 7.1 Log Writer

```python
class NoiseReplayLogWriter:
    """Writer for noise replay logs."""
    
    def __init__(self, log_path: Path, compression: str = "none"):
        """Initialize log writer.
        
        Args:
            log_path: Path to log file
            compression: Compression type ("none", "gzip", "zstd")
        """
        self.log_path = log_path
        self.compression = compression
        
        if compression == "gzip":
            import gzip
            self.file = gzip.open(log_path, "wt")
        elif compression == "zstd":
            import zstandard as zstd
            self.file = zstd.open(log_path, "wt")
        else:
            self.file = open(log_path, "w")
    
    def write_entry(self, entry: Dict[str, Any]) -> None:
        """Write a single log entry.
        
        Args:
            entry: Log entry (dict)
        """
        import json
        self.file.write(json.dumps(entry) + "\n")
        self.file.flush()
    
    def close(self) -> None:
        """Close log file."""
        self.file.close()
```

### 7.2 Log Reader

```python
class NoiseReplayLogReader:
    """Reader for noise replay logs."""
    
    def __init__(self, log_path: Path):
        """Initialize log reader.
        
        Args:
            log_path: Path to log file
        """
        self.log_path = log_path
        
        # Detect compression
        if log_path.suffix == ".gz":
            import gzip
            self.file = gzip.open(log_path, "rt")
        elif log_path.suffix == ".zst":
            import zstandard as zstd
            self.file = zstd.open(log_path, "rt")
        else:
            self.file = open(log_path, "r")
    
    def read_entries(self) -> Iterator[Dict[str, Any]]:
        """Read log entries lazily.
        
        Yields:
            Log entries (dicts)
        """
        import json
        for line in self.file:
            yield json.loads(line)
    
    def close(self) -> None:
        """Close log file."""
        self.file.close()
```

---

## 8. Usage Examples

### 8.1 Recording Noise Decisions

```python
# Initialize log writer
log_writer = NoiseReplayLogWriter(
    log_path=Path("/tmp/noise_replay.jsonl.gz"),
    compression="gzip",
)

# During verification
for cycle_id, item in enumerate(items):
    # Sample noise decision
    should_timeout = noise_sampler.should_timeout(item)
    
    # Record decision
    log_writer.write_entry({
        "version": "1.0",
        "entry_type": "noise_decision",
        "timestamp": time.time(),
        "verification_id": str(uuid.uuid4()),
        "cycle_id": cycle_id,
        "item": item,
        "context": f"cycle_{cycle_id}_item_{item}",
        "tier": "balanced",
        "seed": {
            "master_seed": master_seed,
            "context_hash": hashlib.sha256(item.encode()).hexdigest()[:16],
            "noise_type_path": "timeout",
        },
        "noise_decision": {
            "noise_type": "timeout",
            "injected": should_timeout,
            "probability": 0.05,
            "sampled_value": 0.0234,  # From PRNG
            "decision": "inject" if should_timeout else "no_inject",
        },
        # ... rest of fields
    })

log_writer.close()
```

### 8.2 Replaying Noise Decisions

```python
# Initialize log reader
log_reader = NoiseReplayLogReader(
    log_path=Path("/tmp/noise_replay.jsonl.gz"),
)

# Replay
outcomes = replay_noise_log(list(log_reader.read_entries()))

log_reader.close()

# Verify
assert verify_replay(original_outcomes, outcomes)
```

### 8.3 Anomaly Detection

```python
# Load log
log_reader = NoiseReplayLogReader(Path("/tmp/noise_replay.jsonl.gz"))
log_entries = list(log_reader.read_entries())
log_reader.close()

# Detect anomalies
rate_anomaly = detect_noise_rate_anomaly(log_entries, configured_rate=0.05)
correlation_anomaly = detect_correlation_anomaly(log_entries)
distribution_anomaly = detect_distribution_anomaly(
    log_entries,
    configured_distribution="lognormal",
    configured_params={"mu": 5.2, "sigma": 0.8},
)

# Report
if rate_anomaly["is_anomaly"]:
    print(f"WARNING: Noise rate anomaly detected (p={rate_anomaly['p_value']:.4f})")
if correlation_anomaly["is_anomaly"]:
    print(f"WARNING: Correlation anomaly detected (p={correlation_anomaly['p_value']:.4f})")
if distribution_anomaly["is_anomaly"]:
    print(f"WARNING: Distribution anomaly detected (p={distribution_anomaly['p_value']:.4f})")
```

---

## 9. Summary

Noise replay logs provide deterministic reproducibility for verifier behavior, enabling debugging, differential analysis, and anomaly detection. Key features:

**1. Complete Telemetry**: Every noise decision is logged with full context

**2. Deterministic Replay**: Identical seeds produce identical outcomes

**3. Counterfactual Analysis**: Compare outcomes with and without noise

**4. Anomaly Detection**: Statistical tests for unexpected noise patterns

**5. Efficient Storage**: Compression and indexing for large-scale logs

---

**Manus-C — Telemetry Architect**  
*"Every packet accounted for, every signal explained."*
