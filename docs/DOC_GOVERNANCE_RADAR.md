# Doc Governance Radar

The Doc Governance Radar is a documentation enforcement tool that guards the "First Light" narrative and ensures Phase boundaries are properly maintained across MathLedger documentation.

## Purpose

This radar detects:
- **Premature uplift claims** - Claims of uplift without proper evidence citations
- **TDA enforcement claims** - Claims that TDA is live without proper wiring
- **Phase X language violations** - Incorrect descriptions of P3 (must be "synthetic wind tunnel") or P4 (must be "shadow/no control authority")
- **Substrate alignment claims** - Claims that Substrate solves alignment

## Usage

```bash
# Run from repository root
python3 scripts/radars/doc_governance_radar.py --repo-root . --output artifacts/drift

# Specify custom paths
python3 scripts/radars/doc_governance_radar.py --repo-root /path/to/repo --output /path/to/output
```

## Exit Codes

- **0 (PASS)**: No violations detected
- **1 (FAIL)**: Critical violations detected (premature claims, incorrect phase language)
- **2 (WARN)**: Non-critical issues detected
- **3 (ERROR)**: Infrastructure failure (missing files, read errors)
- **4 (SKIP)**: No documents to check

## Watchlist

The radar monitors:

### Core Documents
- `README.md`
- `docs/PHASE2_RFL_UPLIFT_PLAN.md`
- `docs/RFL_PHASE_I_TRUTH_SOURCE.md`

### Phase X Documents
- `docs/system_law/Phase_X_Prelaunch_Review.md`
- `docs/system_law/Phase_X_Divergence_Metric.md`
- `docs/system_law/Phase_X_P3P4_TODO.md`
- `docs/CORTEX_INTEGRATION.md`
- `docs/TDA_MODES.md`

## Detection Rules

### Uplift Claims

**FAIL**: Claiming uplift without evidence
```markdown
❌ We proved uplift in our experiments.
❌ Uplift achieved across all metrics.
```

**PASS**: Uplift claims with proper evidence citations
```markdown
✅ We demonstrated uplift (see P3/P4 evidence package).
✅ Results pending: integrated-run pending for final validation.
✅ All G1-G5 gates passed with uplift confirmed.
```

### TDA Enforcement Claims

**FAIL**: Claiming TDA is live without qualification
```markdown
❌ TDA enforcement live as of today.
❌ TDA is now the final arbiter of all decisions.
```

**PASS**: TDA claims with proper qualifiers
```markdown
✅ TDA enforcement (not yet wired) will be implemented.
✅ TDA design pending integration.
✅ TDA hooks planned for next phase.
```

### Phase X Language

**P3 - FAIL**: Describing P3 as real world
```markdown
❌ P3 will run in real world conditions.
❌ P3 is our production environment.
```

**P3 - PASS**: Describing P3 as synthetic
```markdown
✅ P3 is a synthetic wind tunnel environment.
✅ P3 simulates conditions in a controlled setting.
```

**P4 - FAIL**: Claiming P4 has control authority
```markdown
❌ P4 has control over the system.
❌ P4 controls all critical paths.
```

**P4 - PASS**: Describing P4 as shadow mode
```markdown
✅ P4 runs in shadow mode with no control authority.
✅ P4 has no control over production systems.
```

### Substrate Alignment

**FAIL**: Claiming Substrate solves alignment
```markdown
❌ The Substrate solves alignment completely.
❌ With this system, alignment is guaranteed.
```

**PASS**: Accurate capability statements
```markdown
✅ The Substrate provides verification capabilities.
✅ The Substrate assists with, but does not solve, alignment.
```

## Output

The radar generates two files:

### `doc_governance_report.json`
Machine-readable JSON report with all violations:
```json
{
  "version": "1.0.0",
  "radar": "doc_governance",
  "status": "FAIL",
  "violations": [
    {
      "type": "premature_uplift_claim",
      "severity": "CRITICAL",
      "filepath": "README.md",
      "line": 42,
      "matched_text": "proved uplift",
      "message": "Claimed 'proved uplift' without evidence citation at line 42"
    }
  ],
  "summary": {
    "critical": 1,
    "warning": 0,
    "info": 0
  }
}
```

### `doc_governance_summary.md`
Human-readable Markdown summary with violation details.

## Integration with CI

To integrate with CI workflows, add a step like:

```yaml
- name: Doc Governance Check
  run: |
    python3 scripts/radars/doc_governance_radar.py --repo-root . --output artifacts/drift
    if [ $? -eq 1 ]; then
      echo "❌ Documentation governance violations detected"
      cat artifacts/drift/doc_governance_summary.md
      exit 1
    fi
```

## Testing

Run the test suite:
```bash
python3 -m pytest tests/test_doc_governance_radar.py -v
```

All tests validate:
- Violation detection (uplift claims, TDA claims, phase language, alignment claims)
- Proper qualifier recognition (evidence citations, "not yet wired", etc.)
- Exit code behavior
- Report artifact generation
- Line number accuracy
- Case-insensitive detection

## Sober Truth Guardrails

The Doc Governance Radar enforces:

- ❌ **NEVER** weaken or remove Phase I disclaimers
- ❌ **NEVER** add claims of demonstrated uplift without gate evidence
- ❌ **NEVER** reinterpret Phase I negative control as "partial success"
- ❌ **NEVER** claim TDA is operational without actual wiring
- ❌ **NEVER** describe P3 as "real world" or P4 as having "control authority"
- ✅ **ALWAYS** require evidence citations for uplift claims
- ✅ **ALWAYS** label Phase II sections clearly
- ✅ **ALWAYS** maintain accurate capability statements
