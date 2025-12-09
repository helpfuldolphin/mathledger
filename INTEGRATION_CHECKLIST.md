# TDA Pipeline Attestation - Integration Checklist

## Overview

This checklist guides the integration of TDA pipeline attestation into existing RFL and U2 runners. The infrastructure is complete; only wiring is needed.

## Prerequisites

✅ All infrastructure implemented:
- [x] TDA pipeline hashing
- [x] Attestation chain verifier
- [x] Integration helpers
- [x] CLI verification tool
- [x] Tests and documentation

## Integration Steps

### Phase 1: RFL Runner Integration

**File**: `rfl/runner.py`

#### Step 1.1: Add Imports

```python
# At top of rfl/runner.py
from attestation import (
    create_rfl_attestation_block,
    save_attestation_block,
    verify_experiment_attestation_chain,
)
```

#### Step 1.2: Initialize Attestation Tracking

In `RFLRunner.__init__()`:

```python
def __init__(self, config: RFLConfig):
    # ... existing code ...
    
    # Add attestation tracking
    self.attestation_blocks: List[ExperimentBlock] = []
    self.prev_block_hash: Optional[str] = None
```

#### Step 1.3: Generate Attestation After Each Run

In `RFLRunner._execute_experiments()`, after each experiment completes:

```python
# Collect events for attestation
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

#### Step 1.4: Add Event Collection Helpers

```python
def _collect_proof_events(self, result: ExperimentResult) -> List[Any]:
    """Collect reasoning artifacts for attestation."""
    events = []
    for statement in result.derived_statements:
        events.append({
            "statement": statement.normalized,
            "hash": statement.hash,
            "verification_method": statement.verification_method,
        })
    return events

def _collect_ui_events(self, result: ExperimentResult) -> List[Any]:
    """Collect UI events for attestation."""
    return [{
        "event_type": "run_completed",
        "timestamp": result.end_time,
        "run_id": result.run_id,
    }]
```

#### Step 1.5: Add Gate Evaluation

```python
def _evaluate_gates(self, result: ExperimentResult) -> Dict[str, str]:
    """Evaluate gate decisions for this run."""
    gates = {}
    
    # G1: Coverage gate
    gates["G1_COVERAGE"] = (
        "PASS" if result.coverage_rate >= self.config.coverage_threshold
        else "FAIL"
    )
    
    # G2: Uplift gate
    if hasattr(result, "uplift_rate"):
        gates["G2_UPLIFT"] = (
            "PASS" if result.uplift_rate > 1.0
            else "FAIL"
        )
    
    # Add more gates as needed
    return gates
```

#### Step 1.6: Verify Chain After All Runs

In `RFLRunner.run_all()`, after all experiments complete:

```python
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
```

**Status**: [ ] Completed

---

### Phase 2: U2 Runner Integration

**File**: `experiments/u2/runner.py` (or wherever U2Runner is defined)

#### Step 2.1: Add Imports

```python
from attestation import (
    create_u2_attestation_block,
    save_attestation_block,
    verify_experiment_attestation_chain,
)
```

#### Step 2.2: Initialize Attestation Tracking

In `U2Runner.__init__()`:

```python
def __init__(self, config: U2Config):
    # ... existing code ...
    
    # Add attestation tracking
    self.attestation_blocks: List[ExperimentBlock] = []
    self.prev_block_hash: Optional[str] = None
```

#### Step 2.3: Generate Attestation After Each Cycle

In U2 runner's main loop, after each cycle completes:

```python
# Collect events
cycle_events = self._collect_cycle_events(cycle_results)
ui_events = self._collect_ui_events()

# Create attestation block
block = create_u2_attestation_block(
    run_id=f"cycle_{cycle_id:03d}",
    experiment_id=self.config.experiment_id,
    reasoning_events=cycle_events,
    ui_events=ui_events,
    u2_config=self._extract_u2_config(),
    gate_decisions=self._evaluate_gates(cycle_results),
    prev_block_hash=self.prev_block_hash,
    block_number=cycle_id,
)

self.attestation_blocks.append(block)
self.prev_block_hash = block.compute_block_hash()

# Save attestation
attestation_path = self.output_dir / f"cycle_{cycle_id:03d}" / "attestation.json"
save_attestation_block(block, attestation_path)
```

#### Step 2.4: Extract U2 Config

```python
def _extract_u2_config(self) -> Dict[str, Any]:
    """Extract U2 config in format expected by TDA hashing."""
    return {
        "max_breadth": self.config.get("max_breadth", 0),
        "max_depth": self.config.get("max_depth", 0),
        "total_cycles": self.config.total_cycles,
        "verifier_tier": self.config.get("verifier_tier", "tier1"),
        "verifier_timeout": self.config.get("verifier_timeout", 10.0),
        "verifier_budget": self.config.get("verifier_budget"),
        "slice_name": self.config.slice_name,
        "slice_config": self.config.slice_config,
        "abstention_strategy": self.config.get("abstention_strategy", "conservative"),
    }
```

#### Step 2.5: Verify Chain After All Cycles

At the end of U2 run:

```python
# Verify attestation chain
logger.info("Verifying attestation chain...")
result = verify_experiment_attestation_chain(
    self.attestation_blocks,
    strict_tda=True
)

if not result.is_valid:
    logger.error(f"❌ Attestation verification failed")
    sys.exit(int(result.error_code))

logger.info("✅ Attestation chain verified")
```

**Status**: [ ] Completed

---

### Phase 3: CI/CD Integration

**File**: `.github/workflows/first-light.yml` (or appropriate workflow)

#### Step 3.1: Add Verification Step

```yaml
- name: Verify Attestation Chain
  run: |
    python scripts/verify_attestation_chain.py \
      --strict-tda \
      --verbose \
      artifacts/first_light/
  continue-on-error: false

- name: Check for TDA Divergence
  if: failure()
  run: |
    echo "::error::Attestation verification failed!"
    echo "Check for exit code 4 (TDA divergence)"
    exit 1
```

**Status**: [ ] Completed

---

### Phase 4: First Light Run

#### Step 4.1: Execute First Light Run

```bash
# Run RFL experiment with attestation
python experiments/run_uplift_u2.py \
  --config configs/first_light.yaml \
  --output artifacts/first_light/
```

#### Step 4.2: Verify Attestation Chain

```bash
# Verify the generated chain
python scripts/verify_attestation_chain.py \
  --strict-tda \
  --verbose \
  artifacts/first_light/
```

**Expected Output**:
```
✅ Attestation chain verification PASSED
```

**Status**: [ ] Completed

---

## Validation Checklist

After integration, verify:

- [ ] Attestation blocks generated for each run
- [ ] TDA pipeline hash present and correct
- [ ] Gate decisions recorded (including ABANDONED_TDA if applicable)
- [ ] Chain linkage valid (prev_block_hash correct)
- [ ] CLI verification succeeds
- [ ] Exit code 4 triggered on configuration drift (test with intentional drift)
- [ ] CI/CD pipeline includes verification step

## Testing

### Test 1: Valid Chain

```bash
# Run experiment
python rfl/runner.py --config test_config.yaml

# Verify
python scripts/verify_attestation_chain.py artifacts/test_exp/

# Expected: Exit code 0
```

### Test 2: TDA Divergence Detection

```bash
# Run first experiment
python rfl/runner.py --config config1.yaml

# Modify config (e.g., change max_breadth)
# Run second experiment with modified config
python rfl/runner.py --config config2.yaml

# Verify chain (should detect drift)
python scripts/verify_attestation_chain.py --strict-tda artifacts/

# Expected: Exit code 4 (TDA divergence)
```

### Test 3: Hard Gate Binding

```bash
# Run experiment that produces ABANDONED_TDA gate
python rfl/runner.py --config abandoned_config.yaml

# Verify attestation includes gate decision
cat artifacts/run_001/attestation.json | jq .gate_decisions

# Expected: {"G4": "ABANDONED_TDA", ...}
```

## Troubleshooting

### Issue: Missing attestation.json

**Symptom**: No attestation files in output directory

**Solution**: 
- Check that `save_attestation_block()` is called
- Verify output path is correct
- Ensure directory exists before saving

### Issue: TDA Hash Mismatch

**Symptom**: Block verification fails with "TDA pipeline hash mismatch"

**Solution**:
- Ensure all required fields present in config
- Check config extraction helper matches expected format
- Use `TDA_INTEGRATION_GUIDE.md` as reference

### Issue: Chain Linkage Broken

**Symptom**: Exit code 3 (chain linkage broken)

**Solution**:
- Verify `prev_block_hash` is updated after each block
- Check blocks are saved in correct order
- Ensure `block_number` increments correctly

## Success Criteria

✅ Integration complete when:
- [ ] RFL runner generates valid attestation blocks
- [ ] U2 runner generates valid attestation blocks
- [ ] CLI verification passes for both runners
- [ ] CI/CD includes attestation verification
- [ ] First Light run produces complete attestation chain
- [ ] Exit code 4 works correctly for drift detection

## Documentation References

- **Architecture**: docs/TDA_PIPELINE_ATTESTATION.md
- **Integration Guide**: docs/TDA_INTEGRATION_GUIDE.md
- **Security**: TDA_SECURITY_SUMMARY.md
- **Summary**: TDA_ATTESTATION_SUMMARY.md
- **Example**: examples/tda_attestation_demo.py

## Support

For questions or issues:
1. Review integration guide: docs/TDA_INTEGRATION_GUIDE.md
2. Check example: examples/tda_attestation_demo.py
3. Validate with tests: tests/test_tda_pipeline_attestation.py

---

**Next Action**: Begin Phase 1 (RFL Runner Integration)
