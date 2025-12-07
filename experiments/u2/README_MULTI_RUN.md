# Multi-Run Evidence Fusion for U2 Experiments

**PHASE II — NOT USED IN PHASE I**

## Overview

The multi-run evidence fusion system provides cross-run validation and evidence aggregation for Phase II U2 experiments. It checks for determinism, consistency, and completeness across multiple experiment runs before promotion.

## Features

### Evidence Fusion (`evidence_fusion.py`)

The `fuse_evidence_summaries()` function aggregates multiple U2 experiment runs and validates:

1. **Determinism Violations**: Detects when runs with identical parameters produce different results
2. **Missing Artifacts**: Checks for missing result files or manifests
3. **Conflicting Slice Names**: Identifies slice name conflicts with same configuration
4. **Run Ordering Anomalies**: Detects issues like RFL runs without corresponding baseline runs
5. **RFL Policy Completeness**: Verifies RFL runs have complete policy statistics

### CLI (`cli.py`)

The `promotion-precheck` command provides a simple interface for validation:

```bash
python3 -m experiments.u2.cli promotion-precheck run1/manifest.json run2/manifest.json ...
```

**Exit Codes:**
- `0` (PASS): All checks passed, runs are ready for promotion
- `1` (WARN): Non-critical issues detected, review recommended
- `2` (BLOCK): Critical issues detected, promotion blocked

## Usage Examples

### Basic Usage

```python
from experiments.u2.evidence_fusion import fuse_evidence_summaries
import json

# Load run manifests
manifests = []
for path in ["run1/manifest.json", "run2/manifest.json"]:
    with open(path) as f:
        manifests.append(json.load(f))

# Fuse evidence
fused = fuse_evidence_summaries(manifests)

# Check status
if fused.pass_status == "PASS":
    print("✅ Ready for promotion")
elif fused.pass_status == "WARN":
    print("⚠️  Review recommended")
    print(f"Anomalies: {fused.run_ordering_anomalies}")
else:  # BLOCK
    print("❌ Promotion blocked")
    print(f"Violations: {fused.determinism_violations}")
    print(f"Missing: {fused.missing_artifacts}")
```

### CLI Usage

```bash
# Check single run
python3 -m experiments.u2.cli promotion-precheck results/slice_a/manifest.json

# Check multiple runs (baseline + RFL)
python3 -m experiments.u2.cli promotion-precheck \
    results/slice_a_baseline/manifest.json \
    results/slice_a_rfl/manifest.json

# Use in CI/CD
python3 -m experiments.u2.cli promotion-precheck results/**/manifest.json
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "PASS: Proceeding with promotion"
elif [ $EXIT_CODE -eq 1 ]; then
    echo "WARN: Review required before promotion"
else
    echo "BLOCK: Cannot promote, fix issues first"
    exit 1
fi
```

### Evidence Integration

```python
from experiments.u2.evidence_fusion import (
    fuse_evidence_summaries,
    inject_multi_run_fusion_into_evidence
)

# Fuse evidence
fused = fuse_evidence_summaries(manifests)

# Inject into existing evidence pack
evidence_pack = {"experiment": "U2_SLICE_A", "version": "1.0"}
combined = inject_multi_run_fusion_into_evidence(evidence_pack, fused)

# Save combined evidence
with open("evidence_pack.json", "w") as f:
    json.dump(combined, f, indent=2)
```

## Validation Rules

### PASS Conditions

All of the following must be true:
- No determinism violations
- No missing artifacts
- No conflicting slice names
- All required fields present in manifests

### WARN Conditions

Any of the following triggers a warning:
- RFL runs without corresponding baseline runs
- Incomplete RFL policy statistics
- Other non-critical anomalies

### BLOCK Conditions

Any of the following blocks promotion:
- Determinism violations (same parameters, different results)
- Missing artifact files
- Conflicting slice names with same configuration
- Missing required fields in manifests

## Manifest Requirements

Each run manifest must include:

```json
{
  "label": "PHASE II — NOT USED IN PHASE I",
  "slice": "SLICE_NAME",
  "mode": "baseline" | "rfl",
  "cycles": 100,
  "initial_seed": 42,
  "slice_config_hash": "abc123...",
  "ht_series_hash": "def456...",
  "outputs": {
    "results": "/path/to/results.jsonl",
    "manifest": "/path/to/manifest.json"
  },
  "policy_stats": {  // Required for RFL mode
    "update_count": 10,
    "success_count": {...},
    "attempt_count": {...}
  }
}
```

## Sober Truth Guardrails

⚠️ **CRITICAL CONSTRAINTS:**

1. ❌ Does NOT interpret results as uplift evidence
2. ❌ Does NOT claim uplift until G1-G5 gates ALL pass
3. ❌ Does NOT reinterpret Phase I logs
4. ❌ Does NOT fabricate or extrapolate metrics
5. ✅ ONLY validates cross-run consistency
6. ✅ Labels all outputs "PHASE II — NOT USED IN PHASE I"

## Testing

Run the test suite:

```bash
python3 -m pytest tests/test_u2_multi_run_evidence.py -v
```

All 35 tests should pass, covering:
- Validation logic
- Determinism checking
- Missing artifact detection
- Slice name conflict detection
- Run ordering anomaly detection
- RFL policy completeness
- CLI exit codes
- Evidence injection

## Integration Points

### Pre-Promotion Gate

Add to CI pipeline before promotion:

```yaml
- name: Multi-Run Pre-Check
  run: |
    python3 -m experiments.u2.cli promotion-precheck \
      artifacts/phase_ii/**/*_manifest.json
```

### Evidence Pack Generation

Integrate with existing evidence pack workflow:

```python
from experiments.u2.evidence_fusion import fuse_evidence_summaries

# Load all run manifests
manifests = load_all_manifests("artifacts/phase_ii/")

# Fuse and validate
fused = fuse_evidence_summaries(manifests)

# Add to evidence pack
evidence_pack["multi_run_validation"] = fused.to_dict()
```

## Future Extensions

Potential future enhancements (NOT currently implemented):

- Cross-slice consistency checking
- Temporal consistency across experiment waves
- Statistical significance testing for uplift claims
- Automated gate (G1-G5) compliance checking
- Integration with preregistration validation

## Security & Correctness

- **Zero security vulnerabilities** (verified with CodeQL)
- **Input validation** on all manifest fields
- **Type safety** via PassStatus class
- **Error handling** for missing/corrupt files
- **Deterministic** processing (same inputs → same outputs)
