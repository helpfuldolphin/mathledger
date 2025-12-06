# Curriculum

Curriculum slice definitions, progression logic, and advancement policies live here:

- Formal definitions of learning stages and gating criteria
- Scheduling of RFL loops across cohorts
- Interfaces for monitoring curriculum alignment with the ledger

Treat curriculum specs as code: version them, hash them, and feed them into attestation pipelines.

---

## Phase III: Drift Radar & Promotion Guard

Phase III adds comprehensive drift detection capabilities on top of Phase II's uplift curriculum framework.

### Key Features

- **CurriculumLoaderV2**: Loads and validates Phase II uplift slice configurations
- **CurriculumFingerprint**: Generates canonical SHA-256 fingerprints for curriculum state
- **Drift History Ledger**: Tracks curriculum changes across multiple runs
- **Drift Classification**: Categorizes changes as NONE, MINOR, or MAJOR
- **Promotion Guard**: Evaluates curriculum stability for production promotion
- **Global Health Summary**: Provides dashboard-ready health metrics

### Quick Start

#### List Slices

```bash
python3 curriculum/cli.py --list-slices
```

#### Show Slice Details

```bash
python3 curriculum/cli.py --show-slice slice_uplift_goal
```

#### Generate Fingerprint

```bash
python3 curriculum/cli.py --fingerprint --run-id baseline --output fingerprint.json
```

#### Check for Drift

```bash
python3 curriculum/cli.py --check-against reference_fingerprint.json
```

Exit codes:
- `0`: No drift or non-blocking drift detected
- `1`: Blocking drift detected (should fail CI)

#### Build Drift History

```bash
python3 curriculum/cli.py --drift-history fp1.json fp2.json fp3.json
```

### Python API

#### Load Curriculum

```python
from curriculum import CurriculumLoaderV2

curriculum = CurriculumLoaderV2.load()
slices = curriculum.list_slices()
```

#### Generate Fingerprint

```python
from curriculum import CurriculumLoaderV2, CurriculumFingerprint

curriculum = CurriculumLoaderV2.load()
fingerprint = CurriculumFingerprint.generate(curriculum, run_id="test-run")
fingerprint.save("fingerprint.json")
```

#### Detect Drift

```python
from curriculum import CurriculumFingerprint, compute_curriculum_diff, classify_curriculum_drift_event

fp1 = CurriculumFingerprint.load_from_file("fp1.json")
fp2 = CurriculumFingerprint.load_from_file("fp2.json")

diff = compute_curriculum_diff(fp1, fp2)
classification = classify_curriculum_drift_event(diff)

if classification['blocking']:
    print(f"❌ Blocking drift: {classification['reasons']}")
```

#### Evaluate Promotion

```python
from curriculum import build_curriculum_drift_history, evaluate_curriculum_for_promotion

history = build_curriculum_drift_history(["fp1.json", "fp2.json", "fp3.json"])
promotion = evaluate_curriculum_for_promotion(history)

if promotion['promotion_ok']:
    print("✅ Safe to promote")
```

### Drift Severity Levels

- **NONE**: No changes detected
- **MINOR**: Parameter tweaks within slices (non-blocking)
- **MAJOR**: Structural changes - slices added/removed, schema changed (blocking)

### Testing

```bash
python3 -m pytest tests/curriculum/ -v
```

43 tests covering:
- Phase II loader validation
- Fingerprint generation and determinism
- Drift detection and classification
- Promotion guard logic
- CLI functionality

### Files

- `phase2_loader.py`: Core loader, fingerprint, and diff logic
- `drift_radar.py`: History ledger, classification, promotion guard
- `cli.py`: Command-line interface
- `tests/curriculum/`: Comprehensive test suite

