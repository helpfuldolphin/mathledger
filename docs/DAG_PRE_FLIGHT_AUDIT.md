# DAG Pre-Flight Audit Specification

**PHASE II — NOT RUN IN PHASE I**

**Author:** CLAUDE G — DAG Law
**Status:** Governance Document
**Version:** 1.0.0

---

## 1. Purpose

This document defines the **Pre-Flight DAG Audit** — a mandatory structural health check that must pass before any U2 experiment audit (`proof_dag_u2_audit.py`) is permitted to execute.

The Pre-Flight Audit ensures:
- The global DAG is structurally sound
- No latent corruption exists that would invalidate audit results
- Baseline and RFL runs are structurally compatible for comparison
- Audit conclusions rest on verified foundations

**Principle:** No audit may proceed on a corrupted or inconsistent DAG.

---

## 2. Scope Definition

### 2.1 What Constitutes "The DAG"

For Pre-Flight purposes, "the DAG" includes:

| Component | Source | Scope |
|-----------|--------|-------|
| **Global Ledger DAG** | `proof_parents` table | All historical edges |
| **Experiment Log DAG** | JSONL log files | Edges from current experiment |
| **Merged DAG** | Union of above | Full provenance graph |

### 2.2 Historical Scope

The Pre-Flight Audit operates on:

```
Historical Window = {
    FULL:     All edges ever recorded (default for production)
    BOUNDED:  Edges from last N blocks (for performance)
    EXPERIMENT: Only edges from current experiment logs
}
```

**Default:** `FULL` — The entire historical DAG must be healthy.

**Override:** `BOUNDED` mode requires explicit justification and records the bound in audit metadata.

### 2.3 In-Scope Artifacts

| Artifact | Required | Purpose |
|----------|----------|---------|
| `proof_parents` table | Yes (if DB mode) | Historical provenance |
| Baseline log (`.jsonl`) | Yes | Baseline DAG snapshot |
| RFL log (`.jsonl`) | Yes | Treatment DAG snapshot |
| Axiom manifest | Yes | Defines valid axiom set |
| Slice configuration | Yes | Defines depth/atom bounds |

---

## 3. Pre-Flight Health Checklist

### 3.1 Mandatory Checks (MUST PASS)

All checks in this section must pass for the audit to proceed.

#### CHECK-001: Acyclicity (INV-001)

**Requirement:** The merged DAG contains no cycles.

**Verification:**
```
For DAG G = (V, E):
    Run Kahn's algorithm (topological sort)
    PASS if all vertices visited
    FAIL if any vertex has non-zero in-degree after completion
```

**Failure Action:** ABORT audit. Cycles indicate fundamental corruption.

**Scope:** Full merged DAG (Global + Experiment logs)

---

#### CHECK-002: No Self-Loops (INV-002)

**Requirement:** No vertex references itself as a parent.

**Verification:**
```
For each edge (child, parent) in E:
    FAIL if child == parent
```

**Failure Action:** ABORT audit. Self-loops are degenerate cycles.

**Scope:** Full merged DAG

---

#### CHECK-003: Hash Integrity (INV-004)

**Requirement:** Each hash maps to exactly one normalized formula.

**Verification:**
```
Build map: hash → Set[normalized_formula]
For each hash h:
    FAIL if |map[h]| > 1
```

**Failure Action:** ABORT audit. Hash collision undermines all provenance.

**Scope:** All statements in Global Ledger + Experiment logs

---

#### CHECK-004: Parent Resolution

**Requirement:** All parent references resolve to known vertices OR are in the allowed axiom set.

**Verification:**
```
Let V = all known vertices (statements)
Let A = allowed axiom hashes (from axiom manifest)
For each edge (child, parent) in E:
    PASS if parent ∈ V OR parent ∈ A
    FAIL otherwise (dangling reference)
```

**Failure Action:**
- If dangling count = 0: PASS
- If dangling count ≤ DANGLING_TOLERANCE: WARN, continue with flag
- If dangling count > DANGLING_TOLERANCE: ABORT

**Default DANGLING_TOLERANCE:** 0 (strict mode)

**Scope:** Full merged DAG

---

#### CHECK-005: Axiom Set Validity

**Requirement:** The axiom manifest is present and parseable.

**Verification:**
```
Load axiom manifest from configured path
PASS if:
    - File exists
    - JSON/YAML parses successfully
    - Contains non-empty axiom hash list
    - All axiom hashes are valid SHA256 (64 hex chars)
```

**Failure Action:** ABORT audit. Cannot distinguish axioms from dangling refs.

**Scope:** Configuration files

---

#### CHECK-006: Log File Integrity

**Requirement:** Experiment log files are complete and parseable.

**Verification:**
```
For each log file (baseline, rfl):
    PASS if:
        - File exists and is readable
        - All lines are valid JSON
        - No truncated final line
        - Contains at least 1 cycle record
```

**Failure Action:** ABORT audit. Incomplete logs yield incomplete DAG.

**Scope:** Experiment log files

---

### 3.2 Conditional Checks (Context-Dependent)

#### CHECK-007: Depth Bound Compliance (if slice specifies max_depth)

**Requirement:** No vertex exceeds configured maximum depth.

**Verification:**
```
For each vertex v in experiment DAG:
    Compute d(v)
    WARN if d(v) > MAX_CONFIGURED_DEPTH
    FAIL if d(v) > MAX_CONFIGURED_DEPTH + DEPTH_TOLERANCE
```

**Default DEPTH_TOLERANCE:** 2

**Failure Action:** WARN or ABORT based on tolerance

**Scope:** Experiment DAG only (not historical)

---

#### CHECK-008: Temporal Consistency (if timestamps available)

**Requirement:** Child statements do not precede their parents in log order.

**Verification:**
```
For each edge (child, parent) where both have timestamps:
    WARN if timestamp(child) < timestamp(parent)
```

**Failure Action:** WARN only (logging order may differ from derivation order)

**Scope:** Experiment logs

---

### 3.3 Check Summary Matrix

| Check | Code | Severity | Scope | Abort On Fail |
|-------|------|----------|-------|---------------|
| Acyclicity | CHECK-001 | CRITICAL | Full | Yes |
| No Self-Loops | CHECK-002 | CRITICAL | Full | Yes |
| Hash Integrity | CHECK-003 | CRITICAL | Full | Yes |
| Parent Resolution | CHECK-004 | ERROR | Full | Conditional |
| Axiom Set Valid | CHECK-005 | ERROR | Config | Yes |
| Log Integrity | CHECK-006 | ERROR | Logs | Yes |
| Depth Bounds | CHECK-007 | WARNING | Experiment | No |
| Temporal Order | CHECK-008 | WARNING | Logs | No |

---

## 4. Drift Eligibility Thresholds

### 4.1 Purpose

Before comparing baseline vs RFL runs, both must be structurally compatible. Excessive structural divergence may indicate:
- Different axiom sets used
- Different slice configurations
- Data corruption in one run
- Non-deterministic derivation behavior

### 4.2 Structural Compatibility Requirements

For an audit comparing Baseline DAG G_b and RFL DAG G_r:

#### DRIFT-001: Axiom Set Alignment

**Requirement:** Both runs use identical axiom sets.

**Verification:**
```
Let A_b = axioms in G_b (vertices with no parents)
Let A_r = axioms in G_r (vertices with no parents)
Let A_manifest = configured axiom set

PASS if A_b ⊆ A_manifest AND A_r ⊆ A_manifest
WARN if A_b ≠ A_r (different axiom usage)
FAIL if A_b ⊄ A_manifest OR A_r ⊄ A_manifest (unknown axioms)
```

---

#### DRIFT-002: Vertex Set Divergence

**Requirement:** Vertex sets must not diverge beyond threshold.

**Verification:**
```
Divergence = |V_b Δ V_r| / max(|V_b|, |V_r|)

PASS if Divergence ≤ MAX_VERTEX_DIVERGENCE
WARN if Divergence ≤ 2 × MAX_VERTEX_DIVERGENCE
FAIL if Divergence > 2 × MAX_VERTEX_DIVERGENCE
```

**Default MAX_VERTEX_DIVERGENCE:** 0.5 (50%)

**Rationale:** RFL may explore different derivation paths, but extreme divergence suggests incompatible runs.

---

#### DRIFT-003: Edge Set Divergence

**Requirement:** Edge sets must not diverge beyond threshold.

**Verification:**
```
Divergence = |E_b Δ E_r| / max(|E_b|, |E_r|)

PASS if Divergence ≤ MAX_EDGE_DIVERGENCE
WARN if Divergence ≤ 2 × MAX_EDGE_DIVERGENCE
FAIL if Divergence > 2 × MAX_EDGE_DIVERGENCE
```

**Default MAX_EDGE_DIVERGENCE:** 0.6 (60%)

---

#### DRIFT-004: Depth Distribution Alignment

**Requirement:** Depth distributions must be comparable.

**Verification:**
```
Let D_b = max_depth(G_b)
Let D_r = max_depth(G_r)

PASS if |D_b - D_r| ≤ MAX_DEPTH_DIFFERENCE
WARN if |D_b - D_r| ≤ 2 × MAX_DEPTH_DIFFERENCE
FAIL if |D_b - D_r| > 2 × MAX_DEPTH_DIFFERENCE
```

**Default MAX_DEPTH_DIFFERENCE:** 3

---

#### DRIFT-005: Cycle Count Alignment

**Requirement:** Both runs must have comparable cycle counts.

**Verification:**
```
Let C_b = cycle count in baseline log
Let C_r = cycle count in RFL log

PASS if C_b == C_r (ideal)
WARN if |C_b - C_r| ≤ CYCLE_TOLERANCE
FAIL if |C_b - C_r| > CYCLE_TOLERANCE
```

**Default CYCLE_TOLERANCE:** 10

---

### 4.3 Drift Threshold Summary

| Check | Metric | Pass | Warn | Fail |
|-------|--------|------|------|------|
| DRIFT-001 | Axiom alignment | Identical manifest | Different usage | Unknown axioms |
| DRIFT-002 | Vertex divergence | ≤ 50% | ≤ 100% | > 100% |
| DRIFT-003 | Edge divergence | ≤ 60% | ≤ 120% | > 120% |
| DRIFT-004 | Depth difference | ≤ 3 | ≤ 6 | > 6 |
| DRIFT-005 | Cycle count | Equal | ± 10 | > ± 10 |

---

## 5. CI/Audit Hook Specification

### 5.1 Pre-Flight Gate Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    CI/CD Pipeline                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │ Experiment   │───▶│  Pre-Flight  │───▶│  U2 Audit │ │
│  │ Logs Ready   │    │  DAG Audit   │    │  Allowed  │ │
│  └──────────────┘    └──────────────┘    └───────────┘ │
│                             │                           │
│                             ▼                           │
│                      ┌──────────────┐                   │
│                      │  FAIL: Abort │                   │
│                      │  No Audit    │                   │
│                      └──────────────┘                   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 5.2 Required Scripts

#### Script 1: `preflight_dag_check.py`

**Purpose:** Execute all CHECK-* validations

**Invocation:**
```bash
uv run python experiments/preflight_dag_check.py \
    --baseline results/uplift_u2_baseline.jsonl \
    --rfl results/uplift_u2_rfl.jsonl \
    --axiom-manifest config/axioms_pl.yaml \
    --slice-config config/slice_uplift_tree.yaml \
    --scope FULL \
    --out results/preflight_report.json
```

**Exit Codes:**
- 0: All checks PASS
- 1: WARN conditions (audit may proceed with flag)
- 2: FAIL conditions (audit blocked)

---

#### Script 2: `drift_eligibility_check.py`

**Purpose:** Execute all DRIFT-* validations

**Invocation:**
```bash
uv run python experiments/drift_eligibility_check.py \
    --baseline results/uplift_u2_baseline.jsonl \
    --rfl results/uplift_u2_rfl.jsonl \
    --thresholds config/drift_thresholds.yaml \
    --out results/drift_eligibility_report.json
```

**Exit Codes:**
- 0: Runs are eligible for comparison
- 1: WARN conditions (comparison may proceed with caveats)
- 2: FAIL conditions (runs incompatible, no comparison)

---

### 5.3 CI Integration

#### GitHub Actions Example

```yaml
name: U2 Audit Pipeline

on:
  workflow_dispatch:
    inputs:
      baseline_log:
        required: true
      rfl_log:
        required: true

jobs:
  preflight:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Pre-Flight DAG Check
        id: preflight
        run: |
          uv run python experiments/preflight_dag_check.py \
            --baseline ${{ inputs.baseline_log }} \
            --rfl ${{ inputs.rfl_log }} \
            --axiom-manifest config/axioms_pl.yaml \
            --scope FULL \
            --out preflight_report.json
        continue-on-error: false

      - name: Drift Eligibility Check
        id: drift
        run: |
          uv run python experiments/drift_eligibility_check.py \
            --baseline ${{ inputs.baseline_log }} \
            --rfl ${{ inputs.rfl_log }} \
            --out drift_report.json
        continue-on-error: false

      - name: Upload Pre-Flight Reports
        uses: actions/upload-artifact@v4
        with:
          name: preflight-reports
          path: |
            preflight_report.json
            drift_report.json

  audit:
    needs: preflight
    runs-on: ubuntu-latest
    steps:
      - name: Run U2 Audit
        run: |
          uv run python experiments/proof_dag_u2_audit.py \
            --baseline ${{ inputs.baseline_log }} \
            --rfl ${{ inputs.rfl_log }} \
            --preflight-report preflight_report.json
```

---

### 5.4 Local Development Hook

For local runs, add to experiment runner:

```python
# In proof_dag_u2_audit.py

def main():
    # MANDATORY: Pre-flight check before any audit
    preflight_result = run_preflight_check(
        baseline_log=args.baseline,
        rfl_log=args.rfl,
        axiom_manifest=args.axioms,
        scope=args.scope,
    )

    if preflight_result.status == "FAIL":
        print("PRE-FLIGHT FAILED. Audit cannot proceed.")
        print(f"Failures: {preflight_result.failures}")
        sys.exit(2)

    if preflight_result.status == "WARN":
        print("PRE-FLIGHT WARNINGS. Audit will proceed with caveats.")
        print(f"Warnings: {preflight_result.warnings}")

    # Only now proceed with audit...
    run_audit(...)
```

---

### 5.5 Bypass Protocol

**No bypass is permitted for CRITICAL checks (CHECK-001, CHECK-002, CHECK-003).**

For ERROR-level checks, bypass requires:

1. **Explicit flag:** `--bypass-preflight-errors`
2. **Justification:** `--bypass-reason "reason text"`
3. **Audit log entry:** Bypass recorded in audit metadata
4. **Human approval:** In CI, requires manual workflow approval

```bash
# Bypass example (use with extreme caution)
uv run python experiments/proof_dag_u2_audit.py \
    --baseline baseline.jsonl \
    --rfl rfl.jsonl \
    --bypass-preflight-errors \
    --bypass-reason "Known axiom manifest update, dangling refs expected"
```

---

## 6. Pre-Flight Report Format

### 6.1 JSON Schema

```json
{
    "preflight_version": "1.0.0",
    "timestamp": "2025-01-15T12:00:00Z",
    "label": "PHASE II — NOT RUN IN PHASE I",

    "inputs": {
        "baseline_log": "results/baseline.jsonl",
        "rfl_log": "results/rfl.jsonl",
        "axiom_manifest": "config/axioms_pl.yaml",
        "scope": "FULL"
    },

    "checks": {
        "CHECK-001": {
            "name": "Acyclicity",
            "status": "PASS",
            "details": {
                "vertices_checked": 1250,
                "cycles_found": 0
            }
        },
        "CHECK-002": {
            "name": "No Self-Loops",
            "status": "PASS",
            "details": {
                "edges_checked": 2340,
                "self_loops_found": 0
            }
        },
        "CHECK-003": {
            "name": "Hash Integrity",
            "status": "PASS",
            "details": {
                "hashes_checked": 1250,
                "collisions_found": 0
            }
        },
        "CHECK-004": {
            "name": "Parent Resolution",
            "status": "PASS",
            "details": {
                "parents_checked": 2340,
                "dangling_found": 0,
                "resolved_to_axioms": 45
            }
        },
        "CHECK-005": {
            "name": "Axiom Set Valid",
            "status": "PASS",
            "details": {
                "axiom_count": 3,
                "axiom_hashes": ["abc...", "def...", "ghi..."]
            }
        },
        "CHECK-006": {
            "name": "Log Integrity",
            "status": "PASS",
            "details": {
                "baseline_lines": 500,
                "rfl_lines": 500,
                "parse_errors": 0
            }
        }
    },

    "drift_eligibility": {
        "DRIFT-001": {"status": "PASS", "axiom_alignment": true},
        "DRIFT-002": {"status": "PASS", "vertex_divergence": 0.23},
        "DRIFT-003": {"status": "PASS", "edge_divergence": 0.31},
        "DRIFT-004": {"status": "PASS", "depth_difference": 1},
        "DRIFT-005": {"status": "PASS", "cycle_difference": 0}
    },

    "summary": {
        "overall_status": "PASS",
        "critical_failures": 0,
        "errors": 0,
        "warnings": 0,
        "audit_eligible": true
    }
}
```

---

## 7. Failure Recovery Procedures

### 7.1 CHECK-001 Failure (Cycles Detected)

**Diagnosis:**
1. Identify cycle nodes from report
2. Trace edges forming the cycle
3. Check for data corruption or logging bugs

**Recovery:**
- If in experiment logs: Regenerate experiment with fixed derivation engine
- If in global ledger: Database repair required (out of scope for normal audit)

**Prevention:**
- Derivation engine must validate acyclicity before recording edges

---

### 7.2 CHECK-003 Failure (Hash Collision)

**Diagnosis:**
1. Identify colliding hashes
2. Retrieve all formulas mapping to those hashes
3. Verify normalization function

**Recovery:**
- If normalization bug: Fix normalizer, rehash affected statements
- If true collision: Cryptographic emergency (extremely unlikely with SHA256)

**Prevention:**
- Normalize before hashing (always)
- Test normalization determinism regularly

---

### 7.3 CHECK-004 Failure (Dangling Parents)

**Diagnosis:**
1. List all dangling parent hashes
2. Check if they should be axioms (missing from manifest)
3. Check if parent statements were lost

**Recovery:**
- If missing axioms: Update axiom manifest
- If lost statements: Investigate data loss, restore from backup

**Prevention:**
- Insert parent statements before child statements
- Validate parent existence at derivation time

---

## 8. Configuration Reference

### 8.1 Default Thresholds

```yaml
# config/preflight_thresholds.yaml

preflight:
  scope: FULL
  dangling_tolerance: 0
  depth_tolerance: 2

drift:
  max_vertex_divergence: 0.5
  max_edge_divergence: 0.6
  max_depth_difference: 3
  cycle_tolerance: 10

bypass:
  allow_error_bypass: false
  require_justification: true
```

### 8.2 Strict Mode (Production)

```yaml
preflight:
  scope: FULL
  dangling_tolerance: 0
  depth_tolerance: 0

drift:
  max_vertex_divergence: 0.3
  max_edge_divergence: 0.4
  max_depth_difference: 2
  cycle_tolerance: 0

bypass:
  allow_error_bypass: false
```

### 8.3 Relaxed Mode (Development)

```yaml
preflight:
  scope: EXPERIMENT
  dangling_tolerance: 5
  depth_tolerance: 5

drift:
  max_vertex_divergence: 0.8
  max_edge_divergence: 0.9
  max_depth_difference: 10
  cycle_tolerance: 50

bypass:
  allow_error_bypass: true
  require_justification: true
```

---

## 9. Audit Trail Requirements

Every Pre-Flight execution must record:

| Field | Description | Retention |
|-------|-------------|-----------|
| `preflight_id` | Unique identifier (UUID) | Permanent |
| `timestamp` | Execution time (ISO 8601) | Permanent |
| `inputs_hash` | SHA256 of input file contents | Permanent |
| `config_hash` | SHA256 of threshold config | Permanent |
| `result` | PASS / WARN / FAIL | Permanent |
| `check_details` | Per-check results | 90 days |
| `operator` | User/CI job that ran check | Permanent |

---

## 10. References

- `docs/U2_DAG_TRUTH_MODEL.md` — Formal graph model and invariants
- `docs/DAG_DRIFT_AUDITOR_SPEC.md` — Drift measurement specification
- `experiments/derivation_chain_analysis.py` — DAG analysis implementation
- `experiments/analyze_chain_depth_u2.py` — Chain depth CLI tool

---

*PHASE II — NOT RUN IN PHASE I*
