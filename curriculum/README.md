# Curriculum

Curriculum slice definitions, progression logic, and advancement policies live here:

- Formal definitions of learning stages and gating criteria
- Scheduling of RFL loops across cohorts
- Interfaces for monitoring curriculum alignment with the ledger
- **NEW:** Curriculum Stability Envelope for P3/P4 governance

Treat curriculum specs as code: version them, hash them, and feed them into attestation pipelines.

---

## Curriculum Stability Envelope

**Version:** 1.0.0  
**Status:** Production (Shadow Mode)  
**Author:** Curriculum Architect Agent

### Overview

The Curriculum Stability Envelope is a governance system that tracks curriculum health via HSS (Homogeneity-Stability Score) metrics, variance tracking, and suitability scoring. It provides bindings for P3 First Light, P4 Calibration, Evidence Packs, and Uplift Council advisory.

**Key Property:** All stability assessments operate in **shadow mode** - they observe and report but do not block execution.

### Core Concepts

#### HSS (Homogeneity-Stability Score)

HSS is a composite metric (0.0-1.0, higher is better) that combines:
- **Parameter Homogeneity** (30%): How "normal" are the slice parameters
- **Temporal Stability** (40%): How consistent are metrics over time
- **Coverage Consistency** (30%): How stable is coverage (inverse of abstention)

#### Suitability Score

Per-slice suitability (0.0-1.0) indicates fitness for production use:
- **Formula:** `(HSS * 0.5) + ((1 - variance) * 0.3) + (coverage * 0.2)`
- **Threshold:** Default 0.6 (below this is flagged as marginal)
- **Critical:** Below 0.3 is considered critical/blocked

#### Status Light

Three-level indicator for curriculum health:
- **GREEN:** All slices healthy, low_HSS_fraction < 25%
- **YELLOW:** Some slices marginal, 25% ≤ low_HSS_fraction ≤ 50%
- **RED:** Many slices unhealthy, low_HSS_fraction > 50%

### Quick Start

```python
from curriculum.stability import build_stability_envelope

# Define slice metrics
slice_metrics = [{
    "slice_name": "slice_a",
    "params": {"atoms": 5, "depth_max": 6, "breadth_max": 1500},
    "coverage_rate": 0.85,
    "abstention_rate": 0.10,
}]

# Build envelope
envelope = build_stability_envelope(slice_metrics)
print(f"Status: {envelope.status_light}")
```

### Integration Points

1. **RFLRunner:** `add_stability_to_rfl_results(results, runner)`
2. **Evidence Packs:** `create_evidence_pack_with_stability(evidence, slice_metrics)`
3. **P4 Reports:** `create_p4_calibration_report_with_stability(calibration, slice_metrics)`
4. **Uplift Council:** `summarize_curriculum_stability_for_council(envelope)`

### Documentation

See [curriculum_stability_appendix.md](../docs/curriculum_stability_appendix.md) for scientific rationale and governance properties.

### Testing

```bash
python3 -m pytest tests/test_curriculum_stability.py -v
python3 -m pytest tests/test_curriculum_integration.py -v
```

