# TDA Pipeline Attestation Integration Guide

## Overview

This guide shows how to integrate TDA pipeline attestation into existing RFL and U2 runners.

## Quick Start

### Step 1: Import the Integration Module

```python
from attestation.experiment_integration import (
    create_rfl_attestation_block,
    save_attestation_block,
)
```

### Step 2: Generate Attestation After Each Run

In your runner's main loop, after completing an experiment run:

```python
# After experiment completes
block = create_rfl_attestation_block(
    run_id=run_id,
    experiment_id=self.config.experiment_id,
    reasoning_events=proof_events,  # List of proof artifacts
    ui_events=ui_events,            # List of UI events
    rfl_config=self.config.to_dict(),
    gate_decisions=gate_decisions,   # Dict of gate names -> decisions
    prev_block_hash=prev_block_hash, # Hash from previous run
    block_number=run_index,
)

# Save attestation block
attestation_path = output_dir / f"run_{run_index:03d}" / "attestation.json"
save_attestation_block(block, attestation_path)

# Store hash for next block
prev_block_hash = block.compute_block_hash()
```

### Step 3: Verify Attestation Chain After Suite Completes

```python
from attestation.chain_verifier import verify_experiment_attestation_chain

# After all runs complete, verify the chain
result = verify_experiment_attestation_chain(blocks, strict_tda=True)

if not result.is_valid:
    logger.error(f"Attestation verification failed: {result.error_message}")
    sys.exit(int(result.error_code))
```

## Detailed Integration Patterns

### RFL Runner Integration

#### In `RFLRunner.__init__`:

```python
def __init__(self, config: RFLConfig):
    # Existing initialization
    ...
    
    # Add attestation tracking
    self.attestation_blocks = []
    self.prev_block_hash = None
```

#### In `RFLRunner._execute_experiments`:

```python
def _execute_experiments(self) -> None:
    """Execute all derivation experiments sequentially."""
    np.random.seed(self.config.random_seed)
    
    for i in range(self.config.num_runs):
        run_id = f"rfl_{self.config.experiment_id}_run_{i+1:02d}"
        slice_cfg = self.config.resolve_slice(i + 1)
        
        # Run experiment
        result = self.experiment.run(...)
        
        # Collect proof and UI events for attestation
        proof_events = self._collect_proof_events(result)
        ui_events = self._collect_ui_events(result)
        
        # Create attestation block
        block = create_rfl_attestation_block(
            run_id=run_id,
            experiment_id=self.config.experiment_id,
            reasoning_events=proof_events,
            ui_events=ui_events,
            rfl_config=self.config.to_dict(),
            gate_decisions=self._evaluate_gates(result),
            prev_block_hash=self.prev_block_hash,
            block_number=i,
        )
        
        self.attestation_blocks.append(block)
        self.prev_block_hash = block.compute_block_hash()
        
        # Save attestation
        attestation_path = self.output_dir / f"run_{i+1:03d}" / "attestation.json"
        save_attestation_block(block, attestation_path)
```

#### Add Gate Evaluation Helper:

```python
def _evaluate_gates(self, result: ExperimentResult) -> Dict[str, str]:
    """
    Evaluate gate decisions for this run.
    
    Returns:
        Dict mapping gate names to decisions ("PASS", "FAIL", "ABANDONED_TDA")
    """
    gates = {}
    
    # G1: Coverage gate
    if result.coverage_rate >= self.config.coverage_threshold:
        gates["G1_COVERAGE"] = "PASS"
    else:
        gates["G1_COVERAGE"] = "FAIL"
    
    # G2: Uplift gate (if applicable)
    if hasattr(result, "uplift_rate"):
        if result.uplift_rate > 1.0:
            gates["G2_UPLIFT"] = "PASS"
        else:
            gates["G2_UPLIFT"] = "FAIL"
    
    # G3: Hard gate decisions
    if self._should_abandon_tda(result):
        gates["G3_HERMETIC"] = "ABANDONED_TDA"
    else:
        gates["G3_HERMETIC"] = "PASS"
    
    return gates
```

#### Verify Chain After All Runs:

```python
def run_all(self) -> Dict[str, Any]:
    """Execute all experiments and verify attestation chain."""
    
    # ... existing execution code ...
    
    # Verify attestation chain
    logger.info("Verifying attestation chain...")
    result = verify_experiment_attestation_chain(
        self.attestation_blocks,
        strict_tda=True
    )
    
    if not result.is_valid:
        logger.error(f"❌ Attestation verification failed: {result.error_message}")
        if result.divergences:
            for div in result.divergences:
                logger.error(f"   {div}")
        sys.exit(int(result.error_code))
    
    logger.info("✅ Attestation chain verified successfully")
    
    return results
```

### U2 Runner Integration

The pattern is similar for U2 runners:

```python
from attestation.experiment_integration import create_u2_attestation_block

# In U2Runner.run():
for cycle_id in range(self.config.total_cycles):
    # Run cycle
    cycle_results = self._run_cycle(cycle_id)
    
    # Create attestation block
    block = create_u2_attestation_block(
        run_id=f"cycle_{cycle_id:03d}",
        experiment_id=self.config.experiment_id,
        reasoning_events=cycle_results,
        ui_events=self._collect_ui_events(),
        u2_config=self.config.__dict__,
        gate_decisions=self._evaluate_gates(cycle_results),
        prev_block_hash=self.prev_block_hash,
        block_number=cycle_id,
    )
    
    # Save and track
    save_attestation_block(block, output_path)
    self.attestation_blocks.append(block)
    self.prev_block_hash = block.compute_block_hash()
```

## Collecting Events for Attestation

### Proof/Reasoning Events

These should include all verifiable artifacts:

```python
def _collect_proof_events(self, result: ExperimentResult) -> List[Any]:
    """Collect reasoning artifacts for attestation."""
    events = []
    
    # Add derivation results
    for statement in result.derived_statements:
        events.append({
            "statement": statement.normalized,
            "hash": statement.hash,
            "verification_method": statement.verification_method,
        })
    
    # Add abstention records
    for abstained in result.abstained_statements:
        events.append({
            "statement": abstained.normalized,
            "abstention_reason": abstained.reason,
        })
    
    return events
```

### UI Events

These should include human interaction traces:

```python
def _collect_ui_events(self, result: ExperimentResult) -> List[Any]:
    """Collect UI events for attestation."""
    events = []
    
    # Add configuration changes
    if hasattr(result, "config_changes"):
        events.extend(result.config_changes)
    
    # Add user interventions
    if hasattr(result, "user_interventions"):
        events.extend(result.user_interventions)
    
    # Add timestamps and metadata
    events.append({
        "event_type": "run_completed",
        "timestamp": result.end_time,
        "run_id": result.run_id,
    })
    
    return events
```

## Hard Gate Decision Binding

### Example Gate Evaluation

```python
def _evaluate_gates(self, result: ExperimentResult) -> Dict[str, str]:
    """Evaluate all gates and return decisions."""
    gates = {}
    
    # G1: Coverage ≥ threshold
    gates["G1"] = "PASS" if result.coverage_rate >= 0.92 else "FAIL"
    
    # G2: Uplift > 1.0
    gates["G2"] = "PASS" if result.uplift_rate > 1.0 else "FAIL"
    
    # G3: Manifest integrity
    gates["G3"] = "PASS" if self._verify_manifest(result) else "FAIL"
    
    # G4: Hermetic execution (may be abandoned for TDA reasons)
    if self._is_hermetic_required() and not self._verify_hermetic(result):
        # Check if TDA constraints prevent hermetic verification
        if self._tda_prevents_hermetic():
            gates["G4"] = "ABANDONED_TDA"
        else:
            gates["G4"] = "FAIL"
    else:
        gates["G4"] = "PASS"
    
    # G5: Velocity within bounds
    gates["G5"] = "PASS" if result.velocity_ok else "FAIL"
    
    return gates
```

### ABANDONED_TDA Decision Criteria

Use `ABANDONED_TDA` when:

1. **Budget Constraints**: TDA budget exhausted before gate could be evaluated
2. **Time Limits**: TDA timeout reached before verification completed
3. **Resource Limits**: TDA resource constraints prevent full verification
4. **Configuration Conflicts**: TDA configuration makes gate inapplicable

Example:

```python
def _tda_prevents_hermetic(self) -> bool:
    """Check if TDA constraints prevent hermetic verification."""
    # Example: Hermetic verification requires network isolation,
    # but TDA config allows network for baseline comparison
    if self.config.tda_mode == "baseline_comparison":
        if self.config.allow_network_for_baseline:
            return True
    
    # Example: Budget exhausted
    if self.tda_budget.is_exhausted():
        return True
    
    return False
```

## CI Integration

### Add to `.github/workflows/`:

```yaml
name: Attestation Chain Verification

on:
  push:
    branches: [main]
  pull_request:

jobs:
  verify-attestation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Verify Attestation Chain
        run: |
          python scripts/verify_attestation_chain.py \
            --strict-tda \
            --verbose \
            artifacts/experiment_output/
        
      - name: Check Exit Code
        if: failure()
        run: |
          echo "Attestation verification failed!"
          echo "Exit code 4 indicates TDA configuration drift"
          exit 1
```

## Testing Your Integration

### Unit Test Template

```python
import pytest
from attestation.experiment_integration import create_rfl_attestation_block
from attestation.chain_verifier import verify_experiment_attestation_chain

def test_rfl_attestation_generation():
    """Test that RFL runner generates valid attestation blocks."""
    
    # Create mock config
    rfl_config = {...}
    
    # Generate block
    block = create_rfl_attestation_block(
        run_id="test_run_001",
        experiment_id="test_exp",
        reasoning_events=["proof1"],
        ui_events=["event1"],
        rfl_config=rfl_config,
        gate_decisions={"G1": "PASS"},
        block_number=0,
    )
    
    # Verify block integrity
    is_valid, error = block.verify_integrity()
    assert is_valid, f"Block invalid: {error}"

def test_rfl_chain_verification():
    """Test that RFL runner generates verifiable chains."""
    
    # Create multiple blocks
    blocks = [...]
    
    # Verify chain
    result = verify_experiment_attestation_chain(blocks, strict_tda=True)
    assert result.is_valid
```

## Troubleshooting

### Issue: TDA Hash Mismatch

**Symptom**: Verification fails with "TDA pipeline hash mismatch"

**Solution**: Ensure all required fields are present in the config dict:
- `max_breadth`, `max_depth`, `max_total`
- `verifier_tier`, `verifier_timeout`
- `slice_id`, `slice_config_hash`
- `abstention_strategy`

### Issue: Exit Code 4 (TDA Divergence)

**Symptom**: Chain verification fails with exit code 4

**Solution**: This indicates configuration changed between runs. Options:
1. **Legitimate Change**: Update experiment series ID
2. **Bug**: Fix configuration to be consistent
3. **Intentional Drift**: Use non-strict mode (`strict_tda=False`)

### Issue: Missing Events

**Symptom**: Attestation blocks have empty event lists

**Solution**: Ensure event collection happens before attestation generation:
```python
# WRONG: Events collected after attestation
block = create_attestation_block(...)
events = collect_events()

# CORRECT: Events collected first
events = collect_events()
block = create_attestation_block(..., reasoning_events=events)
```

## Performance Considerations

- **Block Generation**: ~1ms per block (negligible overhead)
- **Chain Verification**: ~0.1ms per block (O(n) in chain length)
- **File I/O**: Largest cost (~10ms per attestation.json write)

**Recommendation**: Generate attestations asynchronously if running thousands of blocks.

## Security Best Practices

1. **Never modify attestation files after generation**
2. **Store attestation.json in write-once storage** (e.g., append-only logs)
3. **Include attestation hashes in experiment manifests**
4. **Verify chains before trusting results**
5. **Use strict TDA mode in production**

## Next Steps

1. Integrate attestation generation into your runner
2. Add gate evaluation logic
3. Test with sample runs
4. Add CI verification
5. Document your gate decision criteria
