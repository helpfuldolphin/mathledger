# TDA-Aware Evidence Fusion — Phase II

## Overview

This module implements **TDA-aware evidence fusion** for multi-run RFL experiments. It detects conflicts between uplift decisions and TDA (Timeout/Depth Analysis) outcomes, enabling early detection of hidden instabilities before promotion.

**Status:** PHASE II — NOT YET ACTIVATED

**Location:**
- `experiments/evidence_fusion.py` — Core fusion logic
- `experiments/promotion_precheck.py` — CLI for promotion precheck
- `tests/test_evidence_fusion.py` — Comprehensive test suite

## Schema Extension

Each run in a multi-run summary now carries:

```json
{
  "run_id": "U2_EXP_001_run_42",
  "uplift": {
    "delta_p": 0.12,
    "abstention_rate": 0.23,
    "promotion_decision": "PASS"
  },
  "tda": {
    "HSS": 0.85,
    "block_rate": 0.0,
    "tda_outcome": "OK"
  },
  "metadata": {
    "slice": "SLICE_A",
    "seed": "0x42ab...",
    "cycles": 200
  }
}
```

### Fields

**Uplift Metrics:**
- `delta_p`: Delta probability (uplift relative to baseline)
- `abstention_rate`: Fraction of cycles with abstention
- `promotion_decision`: `PASS` | `WARN` | `BLOCK`

**TDA Metrics:**
- `HSS`: Hidden State Score (0.0 to 1.0, higher is better)
- `block_rate`: Fraction of cycles blocked by TDA
- `tda_outcome`: `OK` | `ATTENTION` | `BLOCK`

## Alignment Rules

The fusion engine detects three alignment states:

### 1. OK (No Issues)
- All runs have consistent uplift/TDA decisions
- No conflicts, no hidden instability
- **Exit code:** 0

### 2. WARN (Hidden Instability)
- One or more runs have `PASS` uplift but low HSS
- Indicates potential reproducibility issues
- **Exit code:** 0 (with warning to stderr)

### 3. BLOCK (Uplift/TDA Conflict)
- One or more runs have `PASS` uplift but `BLOCK` TDA
- Critical misalignment requiring investigation
- **Exit code:** 1 (advisory only)

## Usage

### 1. Fusing Evidence Summaries

```bash
# Fuse multiple run summaries with TDA conflict detection
python experiments/evidence_fusion.py \
  input_runs.json \
  fused_output.json \
  --hss-threshold 0.7
```

**Input format** (`input_runs.json`):
```json
{
  "runs": [
    {
      "run_id": "...",
      "uplift": {...},
      "tda": {...}
    },
    ...
  ]
}
```

**Output format** (`fused_output.json`):
```json
{
  "runs": [...],
  "tda_alignment": {
    "conflicted_runs": [],
    "hidden_instability_runs": [],
    "alignment_status": "OK"
  },
  "metadata": {}
}
```

### 2. Running Promotion Precheck

```bash
# Run promotion precheck with TDA alignment analysis
python experiments/promotion_precheck.py fused_output.json

# Exit codes:
#   0 = OK or WARN (no blocking issues)
#   1 = BLOCK (advisory — TDA conflict detected)
#   2 = ERROR (system/configuration error)
```

## Example Scenarios

### Scenario 1: All Clear (OK)

```python
from evidence_fusion import *

runs = [
    RunEvidence(
        run_id="run1",
        uplift=UpliftMetrics(
            delta_p=0.12,
            abstention_rate=0.23,
            promotion_decision=PromotionDecision.PASS,
        ),
        tda=TDAMetrics(
            HSS=0.90,
            block_rate=0.0,
            tda_outcome=TDAOutcome.OK,
        ),
    ),
]

fused = fuse_evidence_summaries(runs)
assert fused.tda_alignment.alignment_status == AlignmentStatus.OK
```

**Output:**
```
✓ No TDA alignment issues detected
  • run1: uplift=PASS, TDA=OK, HSS=0.900
```

### Scenario 2: Hidden Instability (WARN)

```python
runs = [
    RunEvidence(
        run_id="run2",
        uplift=UpliftMetrics(
            delta_p=0.15,
            abstention_rate=0.20,
            promotion_decision=PromotionDecision.PASS,
        ),
        tda=TDAMetrics(
            HSS=0.55,  # Low HSS
            block_rate=0.0,
            tda_outcome=TDAOutcome.OK,
        ),
    ),
]

fused = fuse_evidence_summaries(runs, hss_threshold=0.7)
assert fused.tda_alignment.alignment_status == AlignmentStatus.WARN
```

**Output:**
```
⚠️  WARNING: Hidden instability detected
  • run2: uplift=PASS, HSS=0.550 (below threshold)

This indicates potential hidden state instability that may affect
reproducibility or reliability. Review TDA metrics before promotion.
```

### Scenario 3: Uplift/TDA Conflict (BLOCK)

```python
runs = [
    RunEvidence(
        run_id="run3",
        uplift=UpliftMetrics(
            delta_p=0.18,
            abstention_rate=0.18,
            promotion_decision=PromotionDecision.PASS,
        ),
        tda=TDAMetrics(
            HSS=0.85,
            block_rate=0.30,
            tda_outcome=TDAOutcome.BLOCK,  # TDA blocks
        ),
    ),
]

fused = fuse_evidence_summaries(runs)
assert fused.tda_alignment.alignment_status == AlignmentStatus.BLOCK
```

**Output:**
```
✗ BLOCK: Uplift/TDA conflict detected
  • run3: uplift=PASS, TDA=BLOCK, HSS=0.850, block_rate=0.300

This is a critical misalignment: uplift metrics suggest promotion,
but TDA analysis indicates blocking issues. Do NOT promote until
this conflict is resolved.
```

## Integration with Existing Workflow

### Step 1: Run Multi-Run Experiment

```bash
# Run multiple seeds/slices and collect evidence
for seed in 42 43 44; do
  python experiments/run_uplift_u2.py \
    --slice SLICE_A \
    --mode rfl \
    --seed $seed \
    --cycles 200 \
    --output results/phase2/U2_EXP_001/run_$seed.jsonl
done
```

### Step 2: Extract Evidence Summaries

```bash
# Extract uplift and TDA metrics from each run
python scripts/extract_evidence.py \
  results/phase2/U2_EXP_001/ \
  evidence_summaries.json
```

### Step 3: Fuse Evidence with TDA Awareness

```bash
# Fuse summaries with conflict detection
python experiments/evidence_fusion.py \
  evidence_summaries.json \
  fused_evidence.json \
  --hss-threshold 0.7
```

### Step 4: Run Promotion Precheck

```bash
# Check for alignment issues before promotion
python experiments/promotion_precheck.py fused_evidence.json

# Exit code 0 = proceed (with warnings if WARN)
# Exit code 1 = advisory block (review TDA conflict)
```

## Constraints

### Advisory-Only Blocking

**IMPORTANT:** This tool provides **advisory guidance only**. It does NOT:
- Claim that uplift has been achieved
- Certify that promotion is safe
- Override human judgment or governance decisions

An advisory BLOCK (exit code 1) means:
> "There is a detectable misalignment between uplift metrics and TDA analysis.
> Investigate before proceeding, but this is not a hard gate."

### No Automatic Promotion

This tool is a **pre-check only**. It does not:
- Automatically promote `basis/` to production
- Modify any governance documents
- Update any canonical configurations

All promotion decisions require manual review and governance approval.

## Testing

Comprehensive test suite in `tests/test_evidence_fusion.py`:

```bash
# Run all tests
pytest tests/test_evidence_fusion.py -v

# Run specific test classes
pytest tests/test_evidence_fusion.py::TestEvidenceFusion -v
pytest tests/test_evidence_fusion.py::TestPromotionPrecheck -v
```

Test coverage:
- ✅ PASS uplift + OK TDA → alignment OK
- ✅ PASS uplift + BLOCK TDA → alignment BLOCK
- ✅ PASS uplift + low HSS → alignment WARN
- ✅ CLI exit codes for all scenarios
- ✅ Serialization/deserialization roundtrips
- ✅ Edge cases (empty runs, invalid files)

## References

- **VSD Phase 2:** `docs/VSD_PHASE_2.md` (governance framework)
- **Canonical Basis Plan:** `canonical_basis_plan.md` (promotion criteria)
- **Agent Instructions:** `.github/agents/rfl-uplift-experiments.md`
- **TDA Spec:** `backend/verification/specs/TELEMETRY_TO_EVIDENCE_INTERFACE.md`
