# Curriculum Stability Envelope - Integration Guide

**Version:** 1.0.0  
**Date:** 2025-12-11  
**Author:** Curriculum Architect Agent

---

## Table of Contents

1. [Introduction](#introduction)
2. [Architecture](#architecture)
3. [Usage Examples](#usage-examples)
4. [JSON Schema Reference](#json-schema-reference)
5. [Configuration](#configuration)
6. [Testing](#testing)
7. [Migration Guide](#migration-guide)
8. [Governance Properties](#governance-properties)
9. [FAQ](#faq)

---

## Introduction

The Curriculum Stability Envelope is a governance system that tracks curriculum health and binds stability metrics into P3 First Light, P4 Calibration, Evidence Packs, and Uplift Council advisory.

### Key Features

- **HSS Tracking:** Homogeneity-Stability Score (0.0-1.0) per slice
- **Variance Monitoring:** Temporal stability analysis
- **Suitability Scoring:** Per-slice fitness for production (0.0-1.0)
- **Status Light:** Three-level health indicator (GREEN/YELLOW/RED)
- **Shadow Mode:** Observes and reports, never blocks execution
- **Deterministic:** Same inputs always produce same outputs
- **Non-Mutating:** Evidence Pack integration never modifies originals

### Shadow Mode Guarantee

All stability assessments are **observational only**:
- ✅ Track and report curriculum health
- ✅ Flag problematic slices for review
- ✅ Provide advisory to Uplift Council
- ❌ Never block experiment execution
- ❌ Never gate production deployments

---

## Architecture

```
curriculum/
├── stability.py       # Core envelope implementation
│   ├── compute_hss()
│   ├── compute_variance_metric()
│   ├── compute_suitability_score()
│   ├── build_stability_envelope()
│   ├── attach_curriculum_stability_to_evidence()
│   ├── summarize_curriculum_stability_for_council()
│   ├── add_stability_to_first_light()
│   └── add_stability_to_p4_calibration()
│
├── integration.py     # Runner integration helpers
│   ├── add_stability_to_rfl_results()
│   ├── add_stability_to_u2_results()
│   ├── create_evidence_pack_with_stability()
│   └── create_p4_calibration_report_with_stability()
│
└── README.md          # Quick reference
```

---

## Usage Examples

### Example 1: Basic Envelope Creation

```python
from curriculum.stability import build_stability_envelope

# Define current slice metrics
slice_metrics = [
    {
        "slice_name": "slice_a",
        "params": {
            "atoms": 5,
            "depth_max": 6,
            "breadth_max": 1500
        },
        "coverage_rate": 0.85,
        "abstention_rate": 0.10,
    },
    {
        "slice_name": "slice_b",
        "params": {
            "atoms": 6,
            "depth_max": 8,
            "breadth_max": 2000
        },
        "coverage_rate": 0.80,
        "abstention_rate": 0.15,
    }
]

# Optional: provide historical data for better stability assessment
historical_data = {
    "slice_a": [
        {"coverage_rate": 0.88, "abstention_rate": 0.08},
        {"coverage_rate": 0.86, "abstention_rate": 0.09},
        {"coverage_rate": 0.85, "abstention_rate": 0.10},
    ],
    "slice_b": [
        {"coverage_rate": 0.82, "abstention_rate": 0.14},
        {"coverage_rate": 0.81, "abstention_rate": 0.15},
    ]
}

# Build envelope
envelope = build_stability_envelope(slice_metrics, historical_data)

# Inspect results
print(f"Status Light: {envelope.status_light}")
print(f"Mean HSS: {envelope.mean_hss:.3f}")
print(f"HSS Variance: {envelope.hss_variance:.3f}")
print(f"Flagged Slices: {envelope.slices_flagged}")
print(f"Stable Slices: {envelope.stable_slices}")
print(f"Unstable Slices: {envelope.unstable_slices}")

# Per-slice suitability
for slice_name, score in envelope.suitability_scores.items():
    print(f"  {slice_name}: {score:.3f}")
```

### Example 2: RFLRunner Integration

```python
from curriculum.integration import add_stability_to_rfl_results

class RFLRunner:
    def _export_results(self) -> Dict[str, Any]:
        # Build base results dict
        results = {
            "experiment_id": self.config.experiment_id,
            "execution_summary": {
                "total_runs": len(self.run_results),
                "successful_runs": sum(1 for r in self.run_results if r.status == "success"),
            },
            "runs": [r.to_dict() for r in self.run_results],
            # ... other results
        }
        
        # ✨ ADD STABILITY ENVELOPE
        results = add_stability_to_rfl_results(
            results,
            runner=self,
            include_council=True  # Optional: include Uplift Council advisory
        )
        
        # Export to JSON
        results_path = Path(self.config.artifacts_dir) / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
```

**Result Structure:**

```json
{
  "experiment_id": "rfl_experiment_001",
  "execution_summary": {...},
  "runs": [...],
  "curriculum_stability_envelope": {
    "mean_HSS": 0.75,
    "HSS_variance": 0.08,
    "low_HSS_fraction": 0.2,
    "slices_flagged": ["slice_b"],
    "suitability_scores": {
      "slice_a": 0.85,
      "slice_b": 0.55
    },
    "status_light": "YELLOW",
    "stable_slices": ["slice_a"],
    "unstable_slices": ["slice_b"],
    "HSS_variance_spikes": ["slice_b"]
  },
  "uplift_council_advisory": {
    "status": "WARN",
    "blocked_slices": [],
    "marginal_slices": ["slice_b"],
    "mean_hss": 0.75,
    "hss_variance": 0.08,
    "status_light": "YELLOW"
  }
}
```

### Example 3: Evidence Pack Integration

```python
from curriculum.integration import create_evidence_pack_with_stability

# Create base evidence pack
evidence = {
    "experiment_id": "test_001",
    "timestamp": "2025-12-11T00:00:00Z",
    "results": {
        "coverage": 0.85,
        "uplift": 1.15
    },
    "artifacts": {
        "logs": ["run_001.jsonl"],
        "figures": ["coverage_curve.png"]
    }
}

# Define slice metrics (extracted from runner or experiment)
slice_metrics = [
    {
        "slice_name": "slice_a",
        "params": {"atoms": 5, "depth_max": 6, "breadth_max": 1500},
        "coverage_rate": 0.85,
        "abstention_rate": 0.10,
    }
]

# Create evidence pack with stability (non-mutating)
new_evidence = create_evidence_pack_with_stability(evidence, slice_metrics)

# ✅ Original evidence is unchanged
assert "governance" not in evidence

# ✅ New evidence has stability tile
assert "governance" in new_evidence
assert "curriculum_stability" in new_evidence["governance"]

# Save to disk
with open("evidence_pack_v1.json", 'w') as f:
    json.dump(new_evidence, f, indent=2)
```

**Evidence Pack Structure:**

```json
{
  "experiment_id": "test_001",
  "timestamp": "2025-12-11T00:00:00Z",
  "results": {...},
  "artifacts": {...},
  "governance": {
    "curriculum_stability": {
      "status_light": "GREEN",
      "slices_flagged": [],
      "suitability_scores": {
        "slice_a": 0.85
      }
    }
  }
}
```

### Example 4: P4 Calibration Report

```python
from curriculum.integration import create_p4_calibration_report_with_stability

# Base calibration data
calibration_data = {
    "calibration_id": "p4_001",
    "timestamp": "2025-12-11T00:00:00Z",
    "baseline_metrics": {...},
    "treatment_metrics": {...}
}

# Slice metrics (extracted from both baseline and treatment runs)
slice_metrics = [
    {
        "slice_name": "slice_a",
        "params": {"atoms": 5, "depth_max": 6, "breadth_max": 1500},
        "coverage_rate": 0.90,
        "abstention_rate": 0.05,
    },
    {
        "slice_name": "slice_b",
        "params": {"atoms": 6, "depth_max": 8, "breadth_max": 2000},
        "coverage_rate": 0.50,
        "abstention_rate": 0.40,
    }
]

# Create P4 report with stability
report = create_p4_calibration_report_with_stability(
    calibration_data,
    slice_metrics,
    historical_data=None  # Optional
)

# ✅ Report now includes curriculum_stability section
print(report["curriculum_stability"])
```

**P4 Report Structure:**

```json
{
  "calibration_id": "p4_001",
  "timestamp": "2025-12-11T00:00:00Z",
  "baseline_metrics": {...},
  "treatment_metrics": {...},
  "curriculum_stability": {
    "stable_slices": ["slice_a"],
    "unstable_slices": ["slice_b"],
    "HSS_variance_spikes": ["slice_b"],
    "stability_gate_decisions": {
      "slice_a": "ALLOW",
      "slice_b": "BLOCK"
    }
  }
}
```

**Note:** `stability_gate_decisions` is in **shadow mode** - it shows what WOULD be blocked but does not actually block execution.

### Example 5: Uplift Council Advisory

```python
from curriculum.stability import (
    build_stability_envelope,
    summarize_curriculum_stability_for_council
)

# Build envelope (as shown in Example 1)
envelope = build_stability_envelope(slice_metrics, historical_data)

# Get council-level advisory
advisory = summarize_curriculum_stability_for_council(envelope)

# Interpret status
if advisory["status"] == "OK":
    print("✅ Curriculum is healthy")
elif advisory["status"] == "WARN":
    print("⚠️  Some slices need attention:")
    for slice_name in advisory["marginal_slices"]:
        print(f"  - {slice_name}")
elif advisory["status"] == "BLOCK":
    print("🚫 Critical slices detected:")
    for slice_name in advisory["blocked_slices"]:
        print(f"  - {slice_name}")

# Advisory structure
print(json.dumps(advisory, indent=2))
```

**Advisory Structure:**

```json
{
  "status": "WARN",
  "blocked_slices": [],
  "marginal_slices": ["slice_b"],
  "mean_hss": 0.70,
  "hss_variance": 0.08,
  "status_light": "YELLOW"
}
```

---

## JSON Schema Reference

### CurriculumStabilityEnvelope

```json
{
  "type": "object",
  "properties": {
    "mean_HSS": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    "HSS_variance": {"type": "number", "minimum": 0.0},
    "low_HSS_fraction": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    "slices_flagged": {"type": "array", "items": {"type": "string"}},
    "suitability_scores": {
      "type": "object",
      "additionalProperties": {"type": "number", "minimum": 0.0, "maximum": 1.0}
    },
    "status_light": {"type": "string", "enum": ["GREEN", "YELLOW", "RED"]},
    "stable_slices": {"type": "array", "items": {"type": "string"}},
    "unstable_slices": {"type": "array", "items": {"type": "string"}},
    "HSS_variance_spikes": {"type": "array", "items": {"type": "string"}}
  },
  "required": [
    "mean_HSS", "HSS_variance", "low_HSS_fraction",
    "slices_flagged", "suitability_scores", "status_light"
  ]
}
```

### Evidence Pack Governance Tile

```json
{
  "type": "object",
  "properties": {
    "governance": {
      "type": "object",
      "properties": {
        "curriculum_stability": {
          "type": "object",
          "properties": {
            "status_light": {"type": "string", "enum": ["GREEN", "YELLOW", "RED"]},
            "slices_flagged": {"type": "array", "items": {"type": "string"}},
            "suitability_scores": {
              "type": "object",
              "additionalProperties": {"type": "number"}
            }
          },
          "required": ["status_light", "slices_flagged", "suitability_scores"]
        }
      }
    }
  }
}
```

### P4 Calibration Stability Block

```json
{
  "type": "object",
  "properties": {
    "curriculum_stability": {
      "type": "object",
      "properties": {
        "stable_slices": {"type": "array", "items": {"type": "string"}},
        "unstable_slices": {"type": "array", "items": {"type": "string"}},
        "HSS_variance_spikes": {"type": "array", "items": {"type": "string"}},
        "stability_gate_decisions": {
          "type": "object",
          "additionalProperties": {"type": "string", "enum": ["ALLOW", "BLOCK"]}
        }
      },
      "required": [
        "stable_slices", "unstable_slices",
        "HSS_variance_spikes", "stability_gate_decisions"
      ]
    }
  }
}
```

---

## Configuration

### Default Thresholds

```python
DEFAULT_HSS_THRESHOLD = 0.7          # Below this is "low HSS"
DEFAULT_VARIANCE_THRESHOLD = 0.15    # Above this is "high variance"
DEFAULT_SUITABILITY_THRESHOLD = 0.6  # Below this flags as marginal
```

### Custom Thresholds

```python
from curriculum.stability import build_stability_envelope

custom_thresholds = {
    "hss": 0.75,           # Stricter HSS requirement
    "variance": 0.10,      # Lower variance tolerance
    "suitability": 0.65,   # Higher suitability bar
}

envelope = build_stability_envelope(
    slice_metrics,
    historical_data=None,
    thresholds=custom_thresholds
)
```

---

## Testing

### Unit Tests

```bash
# Test stability module
python3 -m pytest tests/test_curriculum_stability.py -v

# Test integration helpers
python3 -m pytest tests/test_curriculum_integration.py -v
```

### Manual Integration Test

```python
import sys
sys.path.insert(0, '.')
from curriculum.stability import build_stability_envelope

slice_metrics = [{
    "slice_name": "test_slice",
    "params": {"atoms": 5, "depth_max": 6, "breadth_max": 1500},
    "coverage_rate": 0.85,
    "abstention_rate": 0.10,
}]

envelope = build_stability_envelope(slice_metrics)
assert envelope.status_light in ["GREEN", "YELLOW", "RED"]
print(f"✓ Status: {envelope.status_light}")
print(f"✓ Mean HSS: {envelope.mean_hss:.3f}")
```

---

## Migration Guide

### Step 1: Import Helper

```python
# In your runner file (e.g., rfl/runner.py)
from curriculum.integration import add_stability_to_rfl_results
```

### Step 2: Update _export_results()

```python
def _export_results(self) -> Dict[str, Any]:
    # ... existing results building code ...
    
    # ✨ ADD THIS LINE
    results = add_stability_to_rfl_results(results, self)
    
    # ... existing export code ...
    return results
```

### Step 3: Verify Output

Run your experiment and check that the output JSON contains `curriculum_stability_envelope`.

### Step 4: (Optional) Add Council Advisory

```python
results = add_stability_to_rfl_results(
    results,
    self,
    include_council=True  # ← Add this
)
```

---

## Governance Properties

### 1. Shadow Mode

✅ **Guaranteed:** No execution blocking
- Stability assessments are purely observational
- Reports flag problematic slices but do not halt experiments
- P4 gate decisions show what WOULD be blocked, not what IS blocked

### 2. Determinism

✅ **Guaranteed:** Reproducible outputs
- Same slice metrics → same envelope
- No random number generation
- No timestamp dependencies (except metadata)
- JSON-serializable throughout

### 3. Non-Mutation

✅ **Guaranteed:** Original data unchanged
- Evidence Pack integration uses deep copy
- Original dicts are never modified
- Safe for concurrent usage

### 4. Additive Compatibility

✅ **Guaranteed:** Backward compatible
- Adds new fields, doesn't modify existing ones
- Phase I experiments unaffected
- Graceful degradation if stability data unavailable

---

## FAQ

### Q: Does this affect experiment execution?

**A:** No. All stability assessments are in shadow mode and do not block execution.

### Q: What happens if HSS is low?

**A:** The slice is flagged in `slices_flagged` and `unstable_slices`, and the status_light may turn YELLOW or RED. However, execution continues normally.

### Q: Can I customize the thresholds?

**A:** Yes. Pass a `thresholds` dict to `build_stability_envelope()` with custom values for `hss`, `variance`, and `suitability`.

### Q: How often should historical data be updated?

**A:** Historical data should be updated after each run. In RFLRunner, the policy ledger serves as historical data automatically.

### Q: Is this compatible with Phase I experiments?

**A:** Yes. The stability envelope is additive and backward-compatible. It adds new fields but doesn't modify existing outputs.

### Q: What if I don't have historical data?

**A:** The envelope works fine without historical data. HSS will use neutral defaults for temporal stability component (0.5).

### Q: Can I disable stability tracking?

**A:** Yes. Simply don't call the integration helpers. The stability module is opt-in.

### Q: How do I interpret the status light?

- **GREEN:** All good, curriculum is stable
- **YELLOW:** Some slices need attention, monitor closely
- **RED:** Many slices problematic, investigate immediately

### Q: What does "shadow mode" mean for P4 gate decisions?

**A:** It means `stability_gate_decisions` shows what WOULD be blocked if gating were enabled, but it doesn't actually block anything. It's purely informational.

---

## References

- [Curriculum Stability Appendix](./curriculum_stability_appendix.md) - Scientific rationale
- [Phase II RFL Uplift Plan](./PHASE2_RFL_UPLIFT_PLAN.md) - Experimental context
- [Curriculum Gates](../curriculum/gates.py) - Existing gate logic
- [VSD Phase 2](./VSD_PHASE_2.md) - Governance framework

---

## Contact

For questions or issues:
- Review [curriculum/README.md](../curriculum/README.md)
- Check test files: `tests/test_curriculum_stability.py`, `tests/test_curriculum_integration.py`
- Contact: Curriculum Architect Agent

**Version:** 1.0.0  
**Last Updated:** 2025-12-11
