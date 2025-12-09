# Documentation Governance Drift Radar

**Status**: ACTIVE - First Light Integration Sprint  
**Authority**: STRATCOM Directive - Priority Zero  
**Owner**: doc-weaver agent

---

## Purpose

Guards narrative integrity during First Light by detecting governance violations in documentation:

1. **Uplift claims** without "integrated-run pending" disclaimer
2. **TDA enforcement claims** before runner wiring is complete
3. **Contradictions** to Phase I-II disclaimers

> "The organism does not move unless the Cortex approves."

No document may imply "organism alive" until First Light run completes.

---

## STRATCOM Context

**First Light Mission**: Wire `evaluate_hard_gate_decision()` into U2Runner + RFLRunner to produce the first integrated uplift run (Δp + HSS traces).

**Governance Order**: Until First Light completes, all documentation must maintain narrative integrity:
- No uplift claims without integrated-run disclaimer
- No TDA enforcement claims before wiring complete
- No weakening of Phase I-II boundaries

---

## Installation & Usage

### Quick Start

```bash
# Watchdog mode - monitor key governance docs
python scripts/radars/doc_governance_drift_radar.py \
  --mode=watchdog \
  --docs=docs/ \
  --output=artifacts/drift/

# Full scan - all markdown files
python scripts/radars/doc_governance_drift_radar.py \
  --mode=full-scan \
  --docs=docs/ \
  --output=artifacts/drift/

# PR diff mode - scan changes in PR
git diff origin/main...HEAD -- '*.md' > artifacts/drift/pr_diff.patch
python scripts/radars/doc_governance_drift_radar.py \
  --mode=pr-diff \
  --docs=docs/ \
  --output=artifacts/drift/
```

### Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `full-scan` | Scan all markdown files | Pre-commit validation |
| `watchdog` | Monitor 9 key governance docs | Continuous monitoring |
| `pr-diff` | Scan git diff for violations | PR checks |

---

## Violation Types

### Critical Violations (Exit Code 1)

#### 1. Uplift Claims Without Disclaimer

**Pattern**: Positive uplift claims without "integrated-run pending" phrase.

**Examples**:
```markdown
❌ BAD:  "The system demonstrates 15% uplift in success rate"
❌ BAD:  "RFL shows significant improvement in abstention"
✅ GOOD: "Expected uplift (integrated-run pending): 15% improvement"
✅ GOOD: "Phase II will demonstrate uplift (NOT YET RUN)"
```

**Allowed patterns** (no disclaimer needed):
- "No uplift" / "Zero uplift" (negations)
- "Uplift evidence gate" / "Criteria for uplift" (gate definitions)
- "Target uplift" / "Expected uplift" (goals, not claims)

#### 2. TDA Enforcement Claims

**Pattern**: Claims that TDA is actively enforcing before runner wiring complete.

**Examples**:
```markdown
❌ BAD:  "TDA enforces governance decisions"
❌ BAD:  "evaluate_hard_gate_decision() is active"
✅ GOOD: "TDA will enforce once runners are wired"
✅ GOOD: "evaluate_hard_gate_decision() integration pending"
```

#### 3. Phase I Uplift Claims

**Pattern**: Claiming uplift in Phase I sections (Phase I has no uplift by design).

**Examples**:
```markdown
❌ BAD:  "Phase I demonstrates 20% uplift"
✅ GOOD: "Phase I has zero uplift (negative control)"
```

### Warning Violations (Exit Code 2)

#### Incomplete Phase I Disclaimers

When discussing Phase I without proper context:
- Missing "negative control" label
- Missing "100% abstention" note
- Missing "infrastructure validation only" qualifier

---

## Detection Logic

### Uplift Claim Detection

1. **Positive claim patterns**:
   - `demonstrates?|shows?|proves? ... uplift`
   - `uplift ... observed|measured|detected`
   - `significant uplift`
   - `Δp > 0.X` (positive delta-p values)

2. **Negation patterns** (allowed):
   - `no uplift`, `zero uplift`, `without uplift`
   - `does not uplift`, `cannot uplift`

3. **Gate/criteria patterns** (allowed):
   - `uplift evidence gate`, `criteria for uplift`
   - `if/when/should uplift`, `future/potential/target uplift`

4. **Context window**: 7 lines before and after claim
   - Must contain disclaimer or qualifier

5. **Quoted examples**: Skipped (examples of what NOT to say)

### TDA Enforcement Detection

1. **Enforcement patterns**:
   - `TDA ... enforces?`
   - `evaluate_hard_gate_decision() ... active|live|enforcing`
   - `TDA ... blocks?|prevents?|stops?`

2. **Future qualifier** (allowed):
   - `will|future|planned ... wired|integrated`
   - `once|after|when ... connected`

### Phase Disclaimer Detection

1. **Phase I markers**: `Phase I`, `negative control`, etc.
2. **Required context**:
   - "negative control" for Phase I discussions
   - "100% abstention" for RFL claims
   - "infrastructure validation only" for testing claims

---

## Output Artifacts

### 1. Machine-Readable Report

**File**: `artifacts/drift/doc_governance_drift_report.json`

```json
{
  "version": "1.0.0",
  "radar": "doc_governance",
  "mode": "full-scan",
  "status": "FAIL",
  "violations": [
    {
      "type": "uplift_claim_without_disclaimer",
      "severity": "CRITICAL",
      "file": "docs/EXAMPLE.md",
      "line": 42,
      "snippet": "demonstrates 15% uplift",
      "message": "Positive uplift claim without disclaimer"
    }
  ],
  "summary": {
    "critical": 1,
    "warning": 0,
    "info": 0
  }
}
```

### 2. Human-Readable Summary

**File**: `artifacts/drift/doc_governance_drift_summary.md`

Markdown report with:
- Status (PASS/WARN/FAIL)
- Summary counts
- Governance rules
- Detailed violations with file/line references

---

## Exit Codes

| Code | Status | Meaning |
|------|--------|---------|
| 0 | PASS | No violations detected |
| 1 | FAIL | Critical violations (organism not alive) |
| 2 | WARN | Non-critical violations |
| 3 | ERROR | Infrastructure failure |
| 4 | SKIP | No files to scan |

---

## Integration

### Pre-Commit Hook

```bash
#!/bin/bash
# .git/hooks/pre-commit

python scripts/radars/doc_governance_drift_radar.py \
  --mode=watchdog \
  --docs=docs/ \
  --output=artifacts/drift/

if [ $? -eq 1 ]; then
  echo "❌ COMMIT BLOCKED: Documentation governance violations detected"
  echo "   Review artifacts/drift/doc_governance_drift_summary.md"
  exit 1
fi
```

### CI/PR Check

```yaml
# .github/workflows/doc-governance.yml
name: Documentation Governance

on:
  pull_request:
    paths:
      - '**.md'

jobs:
  governance-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Generate PR diff
        run: |
          git fetch origin main
          git diff origin/main...HEAD -- '*.md' > artifacts/drift/pr_diff.patch
      
      - name: Run governance radar
        run: |
          python scripts/radars/doc_governance_drift_radar.py \
            --mode=pr-diff \
            --docs=docs/ \
            --output=artifacts/drift/
      
      - name: Upload report
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: governance-report
          path: artifacts/drift/
```

---

## Key Governance Documents (Watchdog Mode)

The following documents are monitored in watchdog mode:

1. **PHASE2_RFL_UPLIFT_PLAN.md** - Phase II uplift plan
2. **RFL_PHASE_I_TRUTH_SOURCE.md** - Phase I ground truth
3. **VSD_PHASE_2.md** - VSD Phase 2 governance
4. **README.md** - Repository root readme
5. **CLAUDE.md** - Agent instructions
6. Plus any top-level governance docs

---

## Baseline Results

**Current State** (as of First Light):

| Mode | Files Scanned | Critical | Warning | Status |
|------|---------------|----------|---------|--------|
| full-scan | 180 | 9 | 112 | FAIL |
| watchdog | 9 | 0 | 4 | WARN |

**Critical Violations**:
- 1 in `docs/fleet/FPT_Absolute_Readiness_CERTIFIED.md` (uplift claim)
- 4 in Cursor audit docs (quoted examples need context)
- 4 in experiment docs (Phase I section uplift claims)

**Action Required**: Before First Light completes, either:
1. Add "integrated-run pending" disclaimers to uplift claims
2. Remove absolute uplift claims and replace with "expected" or "target"
3. Add Phase II disclaimer to sections claiming uplift

---

## Schema Drift Check (Future)

**TODO**: Add governance schema validation for TDA documentation.

Expected schema fields:
- `governance_version`: Version of TDA governance schema
- `enforcement_status`: "pending" | "active" | "disabled"
- `wiring_complete`: boolean
- `first_light_date`: ISO timestamp or null

Drift detection:
- Schema version mismatch
- Enforcement status changing to "active" before wiring complete
- Missing required fields

---

## Troubleshooting

### False Positives

**Q**: Radar flags my "no uplift" statement.

**A**: Check that negation word (`no`, `zero`, `without`) is in same sentence. If separated, radar may miss negation.

**Fix**: Keep negation close to "uplift" keyword.

---

**Q**: Radar flags my gate definition as uplift claim.

**A**: Ensure you use gate keywords: `criteria`, `if`, `when`, `should`, `would`, `future`, `target`, `goal`.

**Fix**: Rephrase as conditional or future state.

---

**Q**: Radar flags quoted example in "DON'T" section.

**A**: Make sure quote is on same line as opening quote character, or use ❌ marker.

**Fix**: Format as `❌ BAD: "statement"` or `- "statement"`

---

### False Negatives

**Q**: Radar didn't catch an uplift claim.

**A**: Check if claim uses non-standard phrasing or is spread across multiple lines.

**Report**: Open issue with snippet for pattern improvement.

---

## Maintenance

### Adding New Patterns

1. Edit `doc_governance_drift_radar.py`
2. Add pattern to appropriate list:
   - `self.uplift_patterns` - positive claim patterns
   - `self.tda_enforcement_patterns` - TDA enforcement
   - `negation_patterns` - allowed negations
   - `gate_patterns` - allowed gate/criteria phrases

3. Test with baseline:
   ```bash
   python scripts/radars/doc_governance_drift_radar.py --mode=full-scan
   ```

4. Commit with test results

### Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-09 | Initial release for First Light |

---

## References

- **STRATCOM Directive**: First Light integration sprint
- **Phase I Truth Source**: `docs/RFL_PHASE_I_TRUTH_SOURCE.md`
- **Phase II Plan**: `docs/PHASE2_RFL_UPLIFT_PLAN.md`
- **VSD Governance**: `docs/VSD_PHASE_2.md`

---

**Status**: ACTIVE - Monitoring active during First Light sprint  
**Last Updated**: 2025-12-09  
**Authority**: doc-weaver agent under STRATCOM directive
