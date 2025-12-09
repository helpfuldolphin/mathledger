# Drift Radar → Governance Adaptor

**Author**: Manus-B (Ledger Replay Architect & PQ Migration Officer)  
**Phase**: IV - Consensus Integration & Enforcement  
**Date**: 2025-12-09  
**Status**: Implementation Complete

---

## Purpose

Map drift severity levels (LOW→CRITICAL) to governance signals (OK/WARN/BLOCK) for CI/CD enforcement.

**Key Features**:
1. Severity → Signal mapping
2. Configurable governance policies
3. Evidence pack generation
4. Remediation guidance

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Drift Radar Scanner                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Schema       │  │ Hash-Delta   │  │ Metadata     │      │
│  │ Drift        │  │ Drift        │  │ Drift        │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ Drift Signals (with severity)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Governance Adaptor                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Severity     │  │ Policy       │  │ Evidence     │      │
│  │ Mapping      │  │ Evaluation   │  │ Pack         │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ Governance Signal (OK/WARN/BLOCK)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    CI/CD Enforcement                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Merge Gate   │  │ PR Comment   │  │ Console      │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

---

## Severity → Signal Mapping

### Drift Severity Levels

| Severity | Description | Example |
|----------|-------------|---------|
| **LOW** | Benign drift, no action required | Backward-compatible schema evolution |
| **MEDIUM** | Should be investigated | New metadata field added |
| **HIGH** | Requires immediate attention | Hash computation changed |
| **CRITICAL** | Chain integrity compromised | Schema changed without migration |

---

### Governance Signals

| Signal | Description | Action |
|--------|-------------|--------|
| **OK** | No action required | Proceed with merge |
| **WARN** | Warning issued | Proceed with caution, review recommended |
| **BLOCK** | Block merge | Manual review required before merge |
| **EMERGENCY** | Emergency stop | Rollback required, incident response |

---

### Mapping Rules

#### Strict Policy (Default)

| Drift Severity | Max Allowed | Governance Signal |
|----------------|-------------|-------------------|
| CRITICAL | 0 | BLOCK |
| HIGH | 0 | WARN → BLOCK if > 0 |
| MEDIUM | 2 | WARN if > 2 |
| LOW | 5 | WARN if > 5 |

**Behavior**:
- Any CRITICAL drift → BLOCK
- Any HIGH drift → WARN (upgrade to BLOCK if > 0)
- MEDIUM drift > 2 → WARN
- LOW drift > 5 → WARN

---

#### Moderate Policy

| Drift Severity | Max Allowed | Governance Signal |
|----------------|-------------|-------------------|
| CRITICAL | 0 | BLOCK |
| HIGH | 2 | WARN if > 2 |
| MEDIUM | 5 | WARN if > 5 |
| LOW | 10 | WARN if > 10 |

**Behavior**:
- Any CRITICAL drift → BLOCK
- HIGH drift > 2 → WARN
- MEDIUM drift > 5 → WARN
- LOW drift > 10 → WARN

---

#### Permissive Policy

| Drift Severity | Max Allowed | Governance Signal |
|----------------|-------------|-------------------|
| CRITICAL | 1 | WARN if > 1 |
| HIGH | 5 | WARN if > 5 |
| MEDIUM | 10 | WARN if > 10 |
| LOW | 20 | WARN if > 20 |

**Behavior**:
- CRITICAL drift > 1 → WARN
- HIGH drift > 5 → WARN
- MEDIUM drift > 10 → WARN
- LOW drift > 20 → WARN

---

## Evidence Pack Structure

```python
@dataclass
class EvidencePack:
    signal: GovernanceSignal           # OK | WARN | BLOCK | EMERGENCY
    drift_counts: Dict[DriftSeverity, int]  # Counts by severity
    drift_signals: List[Dict[str, Any]]     # Full drift signals
    policy: str                         # Policy name
    timestamp: str                      # ISO 8601 timestamp
    metadata: Dict[str, Any]            # Additional metadata
```

### Example Evidence Pack

```json
{
  "signal": "BLOCK",
  "drift_counts": {
    "LOW": 2,
    "MEDIUM": 1,
    "HIGH": 1,
    "CRITICAL": 1
  },
  "drift_signals": [
    {
      "type": "SCHEMA_DRIFT",
      "severity": "CRITICAL",
      "message": "canonical_proofs schema changed without migration",
      "block_number": 12345
    },
    {
      "type": "HASH_DELTA_DRIFT",
      "severity": "HIGH",
      "message": "Hash computation changed",
      "block_number": 12346
    }
  ],
  "policy": "strict",
  "timestamp": "2025-12-09T12:00:00Z",
  "metadata": {
    "total_signals": 5,
    "policy_thresholds": {
      "LOW": 5,
      "MEDIUM": 2,
      "HIGH": 0,
      "CRITICAL": 0
    }
  }
}
```

---

## Console Output Format

```
============================================================
GOVERNANCE EVIDENCE PACK
============================================================
Signal: BLOCK
Policy: strict
Timestamp: 2025-12-09T12:00:00Z

Drift Counts:
  LOW: 2
  MEDIUM: 1
  HIGH: 1
  CRITICAL: 1

Total Drift Signals: 5

Top 5 Drift Signals:
  1. [CRITICAL] SCHEMA_DRIFT: canonical_proofs schema changed without migration
  2. [HIGH] HASH_DELTA_DRIFT: Hash computation changed
  3. [MEDIUM] METADATA_DRIFT: attestation_metadata field added
  4. [LOW] STATEMENT_DRIFT: canonical_statements format changed
  5. [LOW] SCHEMA_DRIFT: Backward-compatible field added
============================================================
```

---

## Remediation Guidance

### BLOCK Signal

```
MERGE BLOCKED - Manual review required

Remediation steps:
1. Investigate CRITICAL drift signals:
   - SCHEMA_DRIFT: canonical_proofs schema changed without migration
   Action: Revert changes or fix root cause

2. Investigate HIGH drift signals:
   - HASH_DELTA_DRIFT: Hash computation changed
   Action: Review and document intentional changes
```

---

### WARN Signal

```
WARNING - Proceed with caution

Review:
1. MEDIUM drift signals:
   - METADATA_DRIFT: attestation_metadata field added
```

---

### OK Signal

```
OK - No action required
```

---

## Usage

### Python API

```python
from backend.ledger.drift.governance import create_governance_adaptor

# Create adaptor with strict policy
adaptor = create_governance_adaptor("strict")

# Drift signals from drift radar
drift_signals = [
    {
        "type": "SCHEMA_DRIFT",
        "severity": "CRITICAL",
        "message": "canonical_proofs schema changed without migration",
        "block_number": 12345,
    },
    {
        "type": "HASH_DELTA_DRIFT",
        "severity": "HIGH",
        "message": "Hash computation changed",
        "block_number": 12346,
    },
]

# Evaluate
evidence_pack = adaptor.evaluate_drift_signals(drift_signals)

# Check signal
if adaptor.should_block_merge(evidence_pack):
    print("MERGE BLOCKED")
    guidance = adaptor.get_remediation_guidance(evidence_pack)
    for line in guidance:
        print(line)
elif adaptor.should_warn(evidence_pack):
    print("WARNING")
else:
    print("OK")

# Print evidence pack
print(evidence_pack.to_console_output())
```

---

### CLI Usage

```bash
# Run drift radar with governance adaptor
python3 scripts/drift_radar_scan.py \
  --database-url $DATABASE_URL \
  --start-block 0 \
  --end-block 1000 \
  --governance-policy strict \
  --output evidence_pack.json

# Check governance signal
python3 scripts/check_governance_signal.py \
  --evidence-pack evidence_pack.json \
  --fail-on-block
```

---

### CI Integration

```yaml
# .github/workflows/drift-governance.yml
- name: Run drift radar with governance
  run: |
    python3 scripts/drift_radar_scan.py \
      --database-url $DATABASE_URL \
      --start-block 0 \
      --end-block 1000 \
      --governance-policy strict \
      --output evidence_pack.json

- name: Check governance signal
  run: |
    python3 scripts/check_governance_signal.py \
      --evidence-pack evidence_pack.json \
      --fail-on-block

- name: Post evidence pack to PR
  if: github.event_name == 'pull_request'
  uses: actions/github-script@v7
  with:
    script: |
      const fs = require('fs');
      const evidence = JSON.parse(fs.readFileSync('evidence_pack.json', 'utf8'));
      
      const comment = `## Drift Governance Report
      
      **Signal**: ${evidence.signal}
      **Policy**: ${evidence.policy}
      
      ### Drift Counts
      - CRITICAL: ${evidence.drift_counts.CRITICAL}
      - HIGH: ${evidence.drift_counts.HIGH}
      - MEDIUM: ${evidence.drift_counts.MEDIUM}
      - LOW: ${evidence.drift_counts.LOW}
      
      [View Full Report](${evidence.report_url})
      `;
      
      github.rest.issues.createComment({
        issue_number: context.issue.number,
        owner: context.repo.owner,
        repo: context.repo.repo,
        body: comment
      });
```

---

## Policy Selection Guide

### When to Use Strict Policy

- **Production deployments**
- **Main branch**
- **Release branches**
- **Critical infrastructure changes**

**Rationale**: Zero tolerance for CRITICAL/HIGH drift.

---

### When to Use Moderate Policy

- **Develop branch**
- **Feature branches (mature)**
- **Integration testing**

**Rationale**: Allow some HIGH drift for experimentation, but block CRITICAL.

---

### When to Use Permissive Policy

- **Feature branches (early)**
- **Experimental branches**
- **Local development**

**Rationale**: Allow exploration, but warn on excessive drift.

---

## Integration Checklist

- [x] Implement `GovernanceAdaptor` class
- [x] Define `GovernancePolicy` with severity thresholds
- [x] Implement `EvidencePack` structure
- [x] Implement console output formatting
- [x] Implement remediation guidance
- [ ] Write CLI scripts (`drift_radar_scan.py`, `check_governance_signal.py`)
- [ ] Integrate into CI pipeline
- [ ] Write integration tests (10+ tests)
- [ ] Update documentation

---

## Conclusion

The Drift Radar → Governance Adaptor provides a **contract mapping** between drift severity and governance signals, enabling automated enforcement of ledger integrity in CI/CD pipelines.

**Key Benefits**:
1. **Automated enforcement**: No manual review for OK signals
2. **Clear escalation**: WARN → BLOCK based on severity
3. **Evidence-based**: Full evidence pack for debugging
4. **Configurable**: 3 policies (strict/moderate/permissive)

**Status**: Implementation complete, CLI integration pending.

---

**"Keep it blue, keep it clean, keep it sealed."**  
— Manus-B, Ledger Replay Architect & PQ Migration Officer
