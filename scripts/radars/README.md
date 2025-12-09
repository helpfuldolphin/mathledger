# Drift Radars

Automated detection systems for unintended changes to critical project components.

---

## Overview

Drift radars monitor for accidental or intentional changes that violate project invariants, governance rules, or architectural boundaries. Each radar is specialized for a specific domain.

---

## Available Radars

### 1. Curriculum Drift Radar

**Script**: `curriculum_drift_radar.py`  
**Purpose**: Detects changes to curriculum structure, problem definitions, difficulty scores, and topic taxonomies.

**Detects**:
- Schema violations
- Taxonomy changes (topics added/removed)
- Content hash changes
- Difficulty score shifts (>10%)
- Topic reassignments

**Usage**:
```bash
python scripts/radars/curriculum_drift_radar.py \
  --baseline snapshots/curriculum_baseline.json \
  --current snapshots/curriculum_current.json \
  --output artifacts/drift/
```

---

### 2. HT Triangle Drift Radar

**Script**: `ht_triangle_drift_radar.py`  
**Purpose**: Monitors Hash-Time-State triangle integrity for determinism and reproducibility.

**Detects**:
- Hash function changes
- Timestamp format violations
- State tree structural changes
- Merkle root mismatches

**Usage**:
```bash
python scripts/radars/ht_triangle_drift_radar.py \
  --baseline snapshots/ht_baseline.json \
  --current snapshots/ht_current.json \
  --output artifacts/drift/
```

---

### 3. Ledger Drift Radar

**Script**: `ledger_drift_radar.py`  
**Purpose**: Detects changes to ledger state, block hashes, and Merkle roots to ensure cryptographic integrity.

**Detects**:
- Chain ID changes
- Block height mismatches
- Hash chain breaks
- State hash divergence
- Merkle root changes

**Usage**:
```bash
python scripts/radars/ledger_drift_radar.py \
  --baseline snapshots/ledger_baseline.json \
  --current snapshots/ledger_current.json \
  --output artifacts/drift/
```

---

### 4. Telemetry Drift Radar

**Script**: `telemetry_drift_radar.py`  
**Purpose**: Detects changes to telemetry event schemas, field types, and required fields.

**Detects**:
- Event removals
- Schema field type changes
- Required field removals
- Description changes

**Usage**:
```bash
python scripts/radars/telemetry_drift_radar.py \
  --baseline snapshots/telemetry_baseline.json \
  --current snapshots/telemetry_current.json \
  --output artifacts/drift/
```

---

### 5. Documentation Governance Drift Radar ⭐ NEW

**Script**: `doc_governance_drift_radar.py`  
**Purpose**: Guards narrative integrity during First Light by detecting governance violations in documentation.

**Detects**:
- Uplift claims without "integrated-run pending" disclaimer
- TDA enforcement claims before runner wiring complete
- Contradictions to Phase I-II disclaimers
- Missing Phase boundary labels

**Usage**:
```bash
# Watchdog mode (9 key governance docs)
python scripts/radars/doc_governance_drift_radar.py \
  --mode=watchdog \
  --docs=docs/ \
  --output=artifacts/drift/

# Full scan (all markdown files)
python scripts/radars/doc_governance_drift_radar.py \
  --mode=full-scan \
  --docs=docs/ \
  --output=artifacts/drift/

# PR diff mode
git diff origin/main...HEAD -- '*.md' > artifacts/drift/pr_diff.patch
python scripts/radars/doc_governance_drift_radar.py \
  --mode=pr-diff \
  --docs=docs/ \
  --output=artifacts/drift/
```

**Integration Scripts**:
- `scripts/governance-watchdog.sh` (Unix/Linux/macOS)
- `scripts/governance-watchdog.ps1` (Windows/PowerShell)

**CI/CD**: See `.github/workflows/doc-governance-radar.yml`

**Documentation**: `docs/governance/DOC_GOVERNANCE_DRIFT_RADAR.md`

---

## Exit Codes

All radars use a consistent exit code scheme:

| Code | Status | Meaning |
|------|--------|---------|
| 0 | PASS | No drift detected |
| 1 | FAIL | Critical drift detected |
| 2 | WARN | Non-critical drift detected |
| 3 | ERROR | Infrastructure failure (missing files, invalid JSON) |
| 4 | SKIP | No baseline available or no files to scan |

---

## Output Format

All radars produce two artifacts:

### 1. Machine-Readable Report (JSON)

**File**: `{output_dir}/{radar_name}_drift_report.json`

```json
{
  "version": "1.0.0",
  "radar": "curriculum",
  "status": "FAIL",
  "drifts": [
    {
      "type": "content_changed",
      "severity": "CRITICAL",
      "problem_id": "p001",
      "message": "Problem 'p001' content changed"
    }
  ],
  "summary": {
    "critical": 1,
    "warning": 0,
    "info": 0
  }
}
```

### 2. Human-Readable Summary (Markdown)

**File**: `{output_dir}/{radar_name}_drift_summary.md`

- Status (PASS/WARN/FAIL)
- Summary counts (critical/warning/info)
- Detailed drift descriptions
- File/line references (where applicable)

---

## Common Workflow

### 1. Create Baseline Snapshot

```bash
# Example: Curriculum baseline
python scripts/export_curriculum_snapshot.py \
  --output snapshots/curriculum_baseline.json
```

### 2. Make Changes

```bash
# Edit curriculum, code, or documentation
vim config/curriculum.yaml
git commit -m "curriculum: adjust difficulty scores"
```

### 3. Run Radar

```bash
# Generate current snapshot
python scripts/export_curriculum_snapshot.py \
  --output snapshots/curriculum_current.json

# Run drift detection
python scripts/radars/curriculum_drift_radar.py \
  --baseline snapshots/curriculum_baseline.json \
  --current snapshots/curriculum_current.json \
  --output artifacts/drift/
```

### 4. Review Results

```bash
# Check exit code
echo $?

# Review report
cat artifacts/drift/curriculum_drift_summary.md
```

### 5. Address or Approve

- **If FAIL**: Fix violations or update baseline if intentional
- **If WARN**: Review and decide whether to address
- **If PASS**: Changes are safe

---

## Integration

### Pre-Commit Hook

```bash
#!/bin/bash
# .git/hooks/pre-commit

# Run governance radar on docs
if git diff --cached --name-only | grep -q '\.md$'; then
    ./scripts/governance-watchdog.sh watchdog
    if [ $? -eq 1 ]; then
        echo "❌ COMMIT BLOCKED: Documentation governance violations"
        exit 1
    fi
fi
```

### CI/CD Pipeline

```yaml
# .github/workflows/drift-check.yml
- name: Run Drift Radars
  run: |
    # Curriculum
    python scripts/radars/curriculum_drift_radar.py \
      --baseline snapshots/baseline.json \
      --current snapshots/current.json \
      --output artifacts/drift/
    
    # Documentation governance
    ./scripts/governance-watchdog.sh full-scan
```

---

## First Light Governance

During the **First Light integration sprint**, the Documentation Governance Radar enforces:

> "The organism does not move unless the Cortex approves."

**Blocked until First Light completes**:
- ❌ Uplift claims without "integrated-run pending" disclaimer
- ❌ TDA enforcement claims before runner wiring complete
- ❌ Contradictions to Phase I-II disclaimers

See `docs/governance/DOC_GOVERNANCE_DRIFT_RADAR.md` for full details.

---

## Troubleshooting

### False Positives

**Q**: Radar flags a legitimate change.

**A**: Update the baseline snapshot to reflect the new intended state.

```bash
# After reviewing and approving changes
cp snapshots/current.json snapshots/baseline.json
git add snapshots/baseline.json
git commit -m "drift: update baseline after approved changes"
```

---

### False Negatives

**Q**: Radar didn't catch a real drift.

**A**: The detection patterns may need refinement.

1. Open an issue with the missed case
2. Propose pattern improvements
3. Add test case to prevent regression

---

## Development

### Adding a New Radar

1. **Create radar script** in `scripts/radars/`:
   ```python
   #!/usr/bin/env python3
   class MyDriftRadar:
       def __init__(self, baseline_path, current_path, output_dir):
           # Initialize
       
       def detect_drift(self):
           # Detection logic
       
       def run(self) -> int:
           # Returns exit code (0-4)
   ```

2. **Follow naming convention**: `{domain}_drift_radar.py`

3. **Use standard exit codes**: 0 (PASS), 1 (FAIL), 2 (WARN), 3 (ERROR), 4 (SKIP)

4. **Output standard artifacts**:
   - `{domain}_drift_report.json` (machine-readable)
   - `{domain}_drift_summary.md` (human-readable)

5. **Add documentation**: Update this README and create domain-specific docs

6. **Add tests**: Create test cases in `tests/radars/`

---

## Version History

| Radar | Version | Added |
|-------|---------|-------|
| Curriculum | 1.0.0 | 2025-Q3 |
| HT Triangle | 1.0.0 | 2025-Q3 |
| Ledger | 1.0.0 | 2025-Q3 |
| Telemetry | 1.0.0 | 2025-Q3 |
| Doc Governance | 1.0.0 | 2025-12-09 (First Light) |

---

## References

- **STRATCOM Directive**: First Light integration sprint
- **Drift Detection Theory**: Monitoring invariants across state changes
- **Phase Governance**: `docs/RFL_PHASE_I_TRUTH_SOURCE.md`, `docs/PHASE2_RFL_UPLIFT_PLAN.md`

---

**Maintained by**: Infrastructure team + specialized agents  
**Status**: Active monitoring during all development phases
