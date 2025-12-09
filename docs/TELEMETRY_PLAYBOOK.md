# Telemetry Playbook

**PHASE II — Behavioral Telemetry Visualization Guide**

> **DISCLAIMER**: All telemetry outputs are purely descriptive. They visualize
> structural and behavioral patterns in experiment logs without making claims
> about performance, uplift, or statistical significance.

---

## Overview

The Behavioral Telemetry Visualization Suite generates developer-facing plots
that help understand how the system behaves during experiments. These are
**diagnostic tools**, not evidence of improvement.

### What You Get

A "telemetry pack" contains:

| File | Description |
|------|-------------|
| `abstention_heatmap.png` | 2D heatmap: cycles × abstention rate |
| `chain_depth_density.png` | Overlapping histograms of chain depths |
| `candidate_entropy.png` | Rolling entropy of candidate ordering |
| `metric_volatility.png` | Rolling standard deviation of a metric |
| `pack_index.json` | Quick-reference index with plot checksums |
| `telemetry_manifest.json` | Reproducibility manifest with SHA-256 hashes |
| `pack_summary.json` | Pack metadata and parameter record |

---

## Quick Start

### Generate a Telemetry Pack

```bash
uv run python experiments/behavioral_telemetry_viz.py \
    --pack \
    --baseline results/fo_baseline.jsonl \
    --rfl results/fo_rfl.jsonl \
    --out-dir artifacts/telemetry_pack
```

### Verify an Existing Pack

```bash
uv run python experiments/behavioral_telemetry_viz.py \
    --verify \
    --out-dir artifacts/telemetry_pack
```

### Generate Individual Plots

```bash
# Just the abstention heatmap
uv run python experiments/behavioral_telemetry_viz.py \
    --plot-type abstention_heatmap \
    --baseline results/fo_baseline.jsonl \
    --out-dir artifacts/single_plot

# Just the entropy trajectory
uv run python experiments/behavioral_telemetry_viz.py \
    --plot-type entropy \
    --baseline results/fo_baseline.jsonl \
    --rfl results/fo_rfl.jsonl \
    --out-dir artifacts/single_plot
```

---

## Task Recipes

These recipes show how to use telemetry plots for specific diagnostic questions.
Each recipe references only descriptive plots and keys.

> **IMPORTANT**: These are descriptive patterns, not claims of uplift or correctness.
> The plots show "what happened," not "what is better."

### Recipe 1: "Is RFL more volatile than baseline?"

**Goal**: Compare the variability of outcomes between baseline and RFL runs.

**Steps**:
1. Open `metric_volatility.png`
2. Compare the two lines:
   - Gray dashed line = baseline rolling std
   - Black solid line = RFL rolling std
3. Look at the relative height of the lines over cycles

**What to look for**:
- If the RFL line is consistently higher → RFL exhibits more variability
- If the RFL line is consistently lower → RFL exhibits less variability
- If lines cross frequently → volatility patterns differ by phase

**Keys in manifest**: `rolling_metric_volatility` entry in `telemetry_manifest.json`

**This is a descriptive pattern, not a claim of uplift or correctness.**
Higher or lower volatility is neither good nor bad without domain context.

---

### Recipe 2: "Is abstention drift happening mid-run?"

**Goal**: Detect if abstention behavior changes during the experiment.

**Steps**:
1. Open `abstention_heatmap.png`
2. Look for diagonal patterns or vertical color shifts
3. Check if the color concentration moves up or down the y-axis as x increases

**What to look for**:
- Horizontal color bands → stable abstention rate throughout
- Diagonal drift upward → abstention rate increasing over cycles
- Diagonal drift downward → abstention rate decreasing over cycles
- Sudden vertical shift → abrupt change in abstention behavior

**Keys in pack_index.json**:
```json
{
  "plots": [
    {"name": "abstention_heatmap", "filename": "abstention_heatmap.png", ...}
  ]
}
```

**This is a descriptive pattern, not a claim of uplift or correctness.**
Abstention drift may be expected or unexpected depending on the experiment.

---

### Recipe 3: "What depth region do most failures occur in?"

**Goal**: Understand the distribution of chain depths across runs.

**Steps**:
1. Open `chain_depth_density.png`
2. Identify the peaks (most common depths) for each distribution
3. Compare baseline (gray) vs RFL (black) peak locations

**What to look for**:
- Single sharp peak → most derivations reach a specific depth
- Broad distribution → diverse chain depths
- Shifted peaks between baseline/RFL → different depth profiles
- Left-skewed (peaks at low depth) → shallow derivations dominate

**To correlate with failures** (requires additional log analysis):
```python
from experiments.behavioral_telemetry_viz import load_jsonl, extract_chain_depth_series

records = load_jsonl('results/fo_baseline.jsonl')
df = extract_chain_depth_series(records)
# Cross-reference with success/failure status in original logs
```

**This is a descriptive pattern, not a claim of uplift or correctness.**
Depth distribution describes structure, not quality.

---

## Plot Descriptions

### 1. Abstention Heatmap (`abstention_heatmap.png`)

**What it shows**: A 2D histogram where:
- X-axis: Cycle bins (grouped by `--bin-size`, default 20)
- Y-axis: Abstention rate (0 to 1)
- Color: Frequency count (darker = more observations)

**How to read it**:
- Horizontal bands = consistent abstention rate over time
- Vertical streaks = sudden changes in abstention behavior
- Diagonal patterns = gradual drift in abstention rate

**What it does NOT show**: Whether abstention is "good" or "bad" — it only
shows the distribution of abstention rates over cycles.

---

### 2. Chain Depth Density (`chain_depth_density.png`)

**What it shows**: Overlapping normalized histograms of chain depth values:
- Gray (dashed): Baseline run
- Black (solid): RFL run

**How to read it**:
- Peaks indicate common chain depths
- Spread indicates variability in derivation complexity
- Overlapping distributions = similar behavioral profiles

**What it does NOT show**: Whether deeper chains are better. This is purely
a structural comparison.

---

### 3. Candidate Ordering Entropy (`candidate_entropy.png`)

**What it shows**: Rolling mean of entropy over cycles:
- High entropy = more uniform candidate selection
- Low entropy = more deterministic candidate selection

**How to read it**:
- Flat lines = stable selection behavior
- Trends = changing selection patterns over time
- Spikes = sudden behavioral shifts

**What it does NOT show**: Whether high or low entropy is desirable. This
depends entirely on the experiment context.

---

### 4. Metric Volatility (`metric_volatility.png`)

**What it shows**: Rolling standard deviation of a metric (default:
`derivation.verified`):
- High volatility = highly variable outcomes
- Low volatility = consistent outcomes

**How to read it**:
- Declining volatility = system stabilizing
- Increasing volatility = system becoming less predictable
- Flat volatility = consistent behavior

**What it does NOT show**: Whether stability or variability is better. This
is context-dependent.

---

## Reproducibility Manifest

The `telemetry_manifest.json` file ensures A/B reproducibility:

```json
{
  "manifest_version": "1.0.0",
  "generator": "behavioral_telemetry_viz",
  "phase": "PHASE II",
  "rcparams_hash_sha256": "abc123...",
  "entries": [
    {
      "plot_type": "abstention_heatmap",
      "output_path": "artifacts/.../abstention_heatmap.png",
      "file_hash_sha256": "def456...",
      "parameter_hash_sha256": "ghi789...",
      "parameters": { ... }
    }
  ]
}
```

### Verifying Reproducibility

To check that a pack was generated correctly:

```bash
uv run python experiments/behavioral_telemetry_viz.py \
    --verify --out-dir artifacts/telemetry_pack
```

To compare two packs programmatically:

```python
from experiments.behavioral_telemetry_viz import VisualizationManifest

m1 = VisualizationManifest.load('pack_a/telemetry_manifest.json')
m2 = VisualizationManifest.load('pack_b/telemetry_manifest.json')

report = m1.verify_reproducibility(m2)
print(f"All match: {report['all_match']}")
```

---

## Configuration Options

| Flag | Default | Description |
|------|---------|-------------|
| `--baseline` | (required) | Path to baseline JSONL file |
| `--rfl` | (required for comparison plots) | Path to RFL JSONL file |
| `--out-dir` | `artifacts/behavioral_telemetry` | Output directory |
| `--window` | `20` | Rolling window size for smoothing |
| `--bin-size` | `20` | Cycle bin size for heatmap |
| `--metric-path` | `derivation.verified` | Metric for volatility chart |
| `--pack` | (flag) | Generate complete telemetry pack |
| `--verify` | (flag) | Verify existing pack integrity |

---

## Integration with Behavior Atlas

The telemetry pack can be combined with the Behavior Atlas for comprehensive
system profiling. The Atlas provides **structural analysis** (clustering,
fingerprints, matrices) while the Telemetry Pack provides **temporal analysis**
(trajectories, heatmaps, volatility).

> **PHASE II — Descriptive only. No uplift claims.**

### Workflow 1: Generate Atlas + Telemetry Separately

```python
from pathlib import Path

# PHASE II — Descriptive only, no uplift claims.

# Step 1: Generate atlas (structural clustering)
from experiments.u2_behavior_atlas import build_behavior_atlas

atlas, baseline_records = build_behavior_atlas(
    input_dir=Path('results'),
    n_clusters=4,
    clustering_seed=42,  # deterministic
)
# Outputs:
#   atlas.slice_profiles     -> per-slice structural data
#   atlas.js_divergence_matrix  -> behavioral distances
#   atlas.archetypes         -> cluster labels

# Step 2: Generate telemetry pack (temporal visualization)
from experiments.behavioral_telemetry_viz import generate_telemetry_pack

pack_meta = generate_telemetry_pack(
    baseline_path='results/fo_baseline.jsonl',
    rfl_path='results/fo_rfl.jsonl',
    out_dir='artifacts/telemetry',
    window=20,  # rolling window size
)
# Outputs:
#   artifacts/telemetry/abstention_heatmap.png
#   artifacts/telemetry/chain_depth_density.png
#   artifacts/telemetry/candidate_entropy.png
#   artifacts/telemetry/metric_volatility.png
#   artifacts/telemetry/pack_index.json
#   artifacts/telemetry/telemetry_manifest.json
```

### Workflow 2: Combined Analysis (One-Shot)

```python
from pathlib import Path
from experiments.u2_behavior_atlas import generate_combined_analysis

# PHASE II — Descriptive only, no uplift claims.

meta = generate_combined_analysis(
    input_dir=Path('results'),
    baseline_jsonl='results/fo_baseline.jsonl',
    rfl_jsonl='results/fo_rfl.jsonl',
    out_dir=Path('artifacts/combined'),
    n_clusters=4,
    seed=42,
    window=20,
)

# Combined output structure:
#   artifacts/combined/atlas/behavior_atlas.json
#   artifacts/combined/atlas/fingerprints.json
#   artifacts/combined/atlas/js_divergence_heatmap.png
#   artifacts/combined/atlas/trend_similarity_heatmap.png
#   artifacts/combined/atlas/archetype_distribution.png
#   artifacts/combined/telemetry/abstention_heatmap.png
#   artifacts/combined/telemetry/chain_depth_density.png
#   artifacts/combined/telemetry/candidate_entropy.png
#   artifacts/combined/telemetry/metric_volatility.png
#   artifacts/combined/telemetry/pack_index.json
#   artifacts/combined/telemetry/telemetry_manifest.json
#   artifacts/combined/combined_analysis.json
```

### Workflow 3: Load and Inspect Combined Metadata

```python
import json
from experiments.behavioral_telemetry_viz import (
    load_pack_index,
    validate_pack_index_against_manifest,
)

# PHASE II — Descriptive only, no uplift claims.

# Load pack index
index = load_pack_index('artifacts/telemetry')
print(f"Baseline: {index['baseline_log']}")
print(f"RFL: {index['rfl_log']}")
print(f"Plots: {[p['name'] for p in index['plots']]}")

# Validate integrity
report = validate_pack_index_against_manifest('artifacts/telemetry')
assert report['valid'], f"Pack invalid: {report}"

# Load atlas for structural patterns
with open('artifacts/atlas/behavior_atlas.json') as f:
    atlas = json.load(f)

# Where to look:
#   - STRUCTURAL patterns: atlas['js_divergence_matrix'], atlas['archetypes']
#   - TEMPORAL patterns: abstention_heatmap.png, candidate_entropy.png
```

### What to Look For

| Analysis Type | Artifacts | Questions |
|---------------|-----------|-----------|
| **Structural** | `behavior_atlas.json`, heatmaps | How do slices cluster? Which slices behave similarly? |
| **Temporal** | `candidate_entropy.png`, `metric_volatility.png` | How does behavior change over cycles? |
| **Abstention** | `abstention_heatmap.png` | Is abstention rate stable or drifting? |
| **Depth** | `chain_depth_density.png` | What's the distribution of derivation depth? |

> **Remember**: These are descriptive patterns. They describe what happened, not
> what should happen or what is "better."

---

## Output Directory Structure

After running `--pack`, you'll have:

```
artifacts/telemetry_pack/
├── abstention_heatmap.png      # 2D heatmap
├── chain_depth_density.png     # Histogram overlay
├── candidate_entropy.png       # Entropy trajectory
├── metric_volatility.png       # Volatility chart
├── pack_index.json             # Quick-reference index
├── telemetry_manifest.json     # Reproducibility manifest
└── pack_summary.json           # Pack metadata
```

### Pack Index Format

The `pack_index.json` provides a developer-facing quick reference:

```json
{
  "generated_at": "2025-01-15T12:34:56Z",
  "baseline_log": "results/fo_baseline.jsonl",
  "rfl_log": "results/fo_rfl.jsonl",
  "plots": [
    {"name": "abstention_heatmap", "filename": "abstention_heatmap.png", "checksum": "abc123..."},
    {"name": "chain_depth_density", "filename": "chain_depth_density.png", "checksum": "def456..."},
    {"name": "candidate_ordering_entropy", "filename": "candidate_entropy.png", "checksum": "ghi789..."},
    {"name": "rolling_metric_volatility", "filename": "metric_volatility.png", "checksum": "jkl012..."}
  ],
  "manifest_hash": "xyz789..."
}
```

To validate the index matches the manifest:

```python
from experiments.behavioral_telemetry_viz import validate_pack_index_against_manifest

report = validate_pack_index_against_manifest('artifacts/telemetry_pack')
print(f"Valid: {report['valid']}")
```

---

## Troubleshooting

### "No data available" in heatmap

The input JSONL has no records with `derivation.abstained` or `derivation.candidates`.
Check that your log format matches the expected schema.

### Flat lines in all plots

All cycles have identical values. This could mean:
- Very short experiment (< 50 cycles)
- Deterministic system with no variation
- Incorrect metric path

### Checksum mismatch on verification

The PNG files were regenerated with different parameters or on a different
system. Re-run `--pack` with the same parameters to restore consistency.

---

## PHASE II Safeguards

All outputs include:

1. **Watermark**: "PHASE II — NOT EVIDENCE" in figure corner
2. **Metadata label**: `"phase": "PHASE II"` in all JSON files
3. **Disclaimer**: "Descriptive only — not admissible as uplift evidence"

These markers ensure that telemetry visualizations are never mistaken for
statistical evidence of performance improvement.

---

## See Also

- `experiments/behavioral_telemetry_viz.py` — Source code
- `experiments/uplift_visualization.py` — Uplift curve pipeline
- `experiments/u2_behavior_atlas.py` — Structural clustering
- `tests/test_behavioral_telemetry_viz.py` — Test suite

