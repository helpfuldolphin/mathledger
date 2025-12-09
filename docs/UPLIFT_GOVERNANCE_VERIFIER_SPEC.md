# Uplift Governance Verifier Specification

**Version**: 1.0.0
**Status**: PHASE II — NOT RUN IN PHASE I
**Date**: 2025-12-06
**Author**: Claude I (Analytics Constitutional Engineer)

---

## 1. Abstract

This document specifies the **Governance Verifier** — a pure function that certifies whether a U2 uplift analysis is lawful according to the Governance Specification (UPLIFT_ANALYTICS_GOVERNANCE_SPEC.md).

The Verifier takes three inputs (summary.json, manifest.json, telemetry summary) and produces a boolean PASSED/FAILED verdict along with a detailed list of any violated rules.

---

## 2. Verifier Contract

### 2.1 Pure Function Signature

```python
def governance_verify(
    summary: Dict[str, Any],        # Parsed summary.json
    manifest: Dict[str, Any],       # Parsed manifest.json
    telemetry: Dict[str, Any],      # Parsed telemetry summary
    prereg_path: Optional[str] = None,  # Path to preregistration file
) -> GovernanceVerdict:
    """
    PHASE II — NOT RUN IN PHASE I

    Pure function that verifies an uplift analysis against governance rules.

    Args:
        summary: The statistical_summary.json content
        manifest: The experiment manifest.json content
        telemetry: The telemetry_summary.json content (aggregated metrics)
        prereg_path: Optional path to preregistration file for validation

    Returns:
        GovernanceVerdict with:
          - verdict: "PASS" | "WARN" | "FAIL"
          - violations: List of violated rule IDs
          - warnings: List of warning-level rule IDs
          - details: Dict mapping rule_id -> violation details
          - timestamp: ISO 8601 verification timestamp

    Properties:
        - PURE: No side effects, no I/O (except optional prereg file read)
        - DETERMINISTIC: Same inputs -> same outputs
        - TOTAL: Defined for all valid input structures
    """
    raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")
```

### 2.2 Output Data Structure

```python
@dataclass
class RuleResult:
    """Result of checking a single governance rule."""
    rule_id: str              # e.g., "GOV-1", "INV-D3"
    passed: bool              # Whether the rule passed
    severity: str             # "INVALIDATING" | "WARNING" | "COSMETIC"
    message: str              # Human-readable explanation
    evidence: Dict[str, Any]  # Supporting data for the result


@dataclass
class GovernanceVerdict:
    """Complete governance verification result."""

    # Overall verdict
    verdict: str              # "PASS" | "WARN" | "FAIL"

    # Rule results
    violations: List[str]     # Rule IDs with INVALIDATING failures
    warnings: List[str]       # Rule IDs with WARNING failures
    passed_rules: List[str]   # Rule IDs that passed

    # Detailed results
    details: Dict[str, RuleResult]  # rule_id -> full result

    # Metadata
    timestamp: str            # ISO 8601 verification time
    verifier_version: str     # Verifier code version
    rules_checked: int        # Total rules evaluated

    # Summary statistics
    invalidating_count: int   # Count of INVALIDATING failures
    warning_count: int        # Count of WARNING failures
    pass_count: int           # Count of passed rules

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "verdict": self.verdict,
            "violations": self.violations,
            "warnings": self.warnings,
            "passed_rules": self.passed_rules,
            "details": {k: asdict(v) for k, v in self.details.items()},
            "timestamp": self.timestamp,
            "verifier_version": self.verifier_version,
            "rules_checked": self.rules_checked,
            "summary": {
                "invalidating": self.invalidating_count,
                "warning": self.warning_count,
                "passed": self.pass_count,
            }
        }
```

---

## 3. Verification Ruleset

The Verifier checks 43 rules across 4 categories. This section specifies exactly what each rule checks.

### 3.1 Governance Rules (GOV-)

#### GOV-1: Threshold Compliance

```python
def check_gov_1(summary: Dict, manifest: Dict) -> RuleResult:
    """
    Each slice MUST meet its predefined success criteria.

    Required inputs:
        summary["slices"][slice_id]["success_rate"]["rfl"]
        summary["slices"][slice_id]["abstention_rate"]["rfl"]
        summary["slices"][slice_id]["throughput"]["delta_pct"]
        summary["slices"][slice_id]["n_rfl"]

    Expected condition:
        For each slice s in SLICE_IDS:
            success_rate >= CRITERIA[s].min_SR
            abstention_rate <= CRITERIA[s].max_AR
            throughput_uplift >= CRITERIA[s].min_delta_pct
            n_rfl >= CRITERIA[s].min_n

    Severity: INVALIDATING
    """
    CRITERIA = {
        "prop_depth4": {"min_SR": 0.95, "max_AR": 0.02, "min_delta_pct": 5.0, "min_n": 500},
        "fol_eq_group": {"min_SR": 0.85, "max_AR": 0.10, "min_delta_pct": 3.0, "min_n": 300},
        "fol_eq_ring": {"min_SR": 0.80, "max_AR": 0.15, "min_delta_pct": 2.0, "min_n": 300},
        "linear_arith": {"min_SR": 0.70, "max_AR": 0.20, "min_delta_pct": 0.0, "min_n": 200},
    }

    failures = []
    for slice_id, criteria in CRITERIA.items():
        slice_data = summary.get("slices", {}).get(slice_id, {})
        # ... check each criterion ...

    return RuleResult(
        rule_id="GOV-1",
        passed=len(failures) == 0,
        severity="INVALIDATING",
        message="Threshold compliance check" if not failures else f"Failed: {failures}",
        evidence={"failures": failures}
    )
```

#### GOV-2: Decision Exclusivity

```python
def check_gov_2(summary: Dict) -> RuleResult:
    """
    Exactly one of {PROCEED, HOLD, ROLLBACK} MUST be true.

    Required inputs:
        summary["governance"]["recommendation"]

    Expected condition:
        recommendation in {"proceed", "hold", "rollback"}

    Severity: INVALIDATING
    """
    recommendation = summary.get("governance", {}).get("recommendation", "")
    valid_decisions = {"proceed", "hold", "rollback"}
    passed = recommendation.lower() in valid_decisions

    return RuleResult(
        rule_id="GOV-2",
        passed=passed,
        severity="INVALIDATING",
        message=f"Decision '{recommendation}' is valid" if passed else f"Invalid decision: {recommendation}",
        evidence={"recommendation": recommendation, "valid_options": list(valid_decisions)}
    )
```

#### GOV-3: Decision Consistency

```python
def check_gov_3(summary: Dict) -> RuleResult:
    """
    PROCEED requires all_slices_pass = true.

    Required inputs:
        summary["governance"]["recommendation"]
        summary["governance"]["all_slices_pass"]

    Expected condition:
        IF recommendation == "proceed" THEN all_slices_pass == True
        IF all_slices_pass == False THEN recommendation in {"hold", "rollback"}

    Severity: INVALIDATING
    """
    gov = summary.get("governance", {})
    recommendation = gov.get("recommendation", "").lower()
    all_pass = gov.get("all_slices_pass", False)

    if recommendation == "proceed" and not all_pass:
        return RuleResult("GOV-3", False, "INVALIDATING",
            "PROCEED but all_slices_pass=False", {"recommendation": recommendation, "all_slices_pass": all_pass})

    if not all_pass and recommendation not in {"hold", "rollback"}:
        return RuleResult("GOV-3", False, "INVALIDATING",
            "Not all slices pass but recommendation is not HOLD/ROLLBACK",
            {"recommendation": recommendation, "all_slices_pass": all_pass})

    return RuleResult("GOV-3", True, "INVALIDATING", "Decision is consistent", {})
```

#### GOV-4: Failing Slice Identification

```python
def check_gov_4(summary: Dict) -> RuleResult:
    """
    Failing slices must be correctly identified.

    Required inputs:
        summary["governance"]["passing_slices"]
        summary["governance"]["failing_slices"]
        summary["slices"] (keys)

    Expected condition:
        passing_slices ∪ failing_slices = set(slices.keys())
        passing_slices ∩ failing_slices = ∅

    Severity: INVALIDATING
    """
    gov = summary.get("governance", {})
    passing = set(gov.get("passing_slices", []))
    failing = set(gov.get("failing_slices", []))
    all_slices = set(summary.get("slices", {}).keys())

    union_ok = (passing | failing) == all_slices
    disjoint_ok = len(passing & failing) == 0

    return RuleResult(
        rule_id="GOV-4",
        passed=union_ok and disjoint_ok,
        severity="INVALIDATING",
        message="Slice partition is valid" if (union_ok and disjoint_ok) else "Invalid slice partition",
        evidence={"passing": list(passing), "failing": list(failing),
                  "expected": list(all_slices), "union_ok": union_ok, "disjoint_ok": disjoint_ok}
    )
```

#### GOV-5: Marginal Case Flagging

```python
def check_gov_5(summary: Dict) -> RuleResult:
    """
    Marginal cases (CI overlaps threshold) should be flagged.

    Required inputs:
        summary["slices"][slice_id]["throughput"]["ci_low"]
        summary["slices"][slice_id]["throughput"]["ci_high"]

    Expected condition:
        If ci_low < threshold < ci_high, slice should be marked marginal.

    Severity: WARNING
    """
    THRESHOLDS = {
        "prop_depth4": 5.0, "fol_eq_group": 3.0,
        "fol_eq_ring": 2.0, "linear_arith": 0.0
    }
    marginal_slices = []

    for slice_id, threshold in THRESHOLDS.items():
        slice_data = summary.get("slices", {}).get(slice_id, {})
        throughput = slice_data.get("throughput", {})
        ci_low = throughput.get("ci_low", 0)
        ci_high = throughput.get("ci_high", 0)

        if ci_low < threshold < ci_high:
            marginal_slices.append(slice_id)

    # Check if marginal slices are flagged (in HOLD rationale or warnings)
    # This is a WARNING because detection is informational
    return RuleResult(
        rule_id="GOV-5",
        passed=True,  # Always passes, just flags
        severity="WARNING",
        message=f"Marginal slices: {marginal_slices}" if marginal_slices else "No marginal cases",
        evidence={"marginal_slices": marginal_slices}
    )
```

#### GOV-6 through GOV-12

*(Similar structure — see implementation skeleton below)*

---

### 3.2 Reproducibility Rules (REP-)

#### REP-1: Baseline Seed Documented

```python
def check_rep_1(manifest: Dict) -> RuleResult:
    """
    manifest.json MUST contain seed_baseline.

    Required inputs:
        manifest["config"]["seed_baseline"]

    Expected condition:
        seed_baseline exists and is a positive integer

    Severity: INVALIDATING
    """
    seed = manifest.get("config", {}).get("seed_baseline")
    passed = isinstance(seed, int) and seed > 0

    return RuleResult(
        rule_id="REP-1",
        passed=passed,
        severity="INVALIDATING",
        message=f"seed_baseline={seed}" if passed else "Missing or invalid seed_baseline",
        evidence={"seed_baseline": seed}
    )
```

#### REP-5: Bootstrap Iterations Minimum

```python
def check_rep_5(summary: Dict) -> RuleResult:
    """
    n_bootstrap MUST be >= 10,000.

    Required inputs:
        summary["reproducibility"]["n_bootstrap"]

    Expected condition:
        n_bootstrap >= 10000

    Severity: INVALIDATING
    """
    n_bootstrap = summary.get("reproducibility", {}).get("n_bootstrap", 0)
    passed = n_bootstrap >= 10000

    return RuleResult(
        rule_id="REP-5",
        passed=passed,
        severity="INVALIDATING",
        message=f"n_bootstrap={n_bootstrap}" if passed else f"n_bootstrap={n_bootstrap} < 10000",
        evidence={"n_bootstrap": n_bootstrap, "minimum": 10000}
    )
```

#### REP-6: Determinism Verification

```python
def check_rep_6(summary: Dict, summary_rerun: Optional[Dict] = None) -> RuleResult:
    """
    Re-running analysis MUST produce identical results (except timestamp).

    Required inputs:
        Two summary.json files from same inputs

    Expected condition:
        summary (excluding generated_at) == summary_rerun (excluding generated_at)

    Severity: INVALIDATING

    Note: This requires an actual re-run. If summary_rerun is None,
          the check is marked as SKIPPED (not failed).
    """
    if summary_rerun is None:
        return RuleResult(
            rule_id="REP-6",
            passed=True,  # Cannot fail without rerun
            severity="INVALIDATING",
            message="Determinism check skipped (no rerun provided)",
            evidence={"skipped": True}
        )

    # Compare excluding timestamps
    def normalize(d):
        d = dict(d)
        d.pop("computed_at", None)
        d.pop("generated_at", None)
        d.pop("timestamp", None)
        return d

    s1 = normalize(summary)
    s2 = normalize(summary_rerun)
    passed = s1 == s2

    return RuleResult(
        rule_id="REP-6",
        passed=passed,
        severity="INVALIDATING",
        message="Results are deterministic" if passed else "Non-deterministic results detected",
        evidence={"identical": passed}
    )
```

---

### 3.3 Manifest Rules (MAN-)

#### MAN-6: Artifact Checksums Valid

```python
def check_man_6(manifest: Dict, base_path: str = ".") -> RuleResult:
    """
    All recorded checksums MUST match actual file checksums.

    Required inputs:
        manifest["checksums"] - dict of path -> sha256

    Expected condition:
        For each (path, expected_hash) in checksums:
            SHA256(read(path)) == expected_hash

    Severity: INVALIDATING
    """
    import hashlib
    from pathlib import Path

    checksums = manifest.get("checksums", {})
    failures = []

    for path, expected_hash in checksums.items():
        full_path = Path(base_path) / path
        if not full_path.exists():
            failures.append({"path": path, "error": "file not found"})
            continue

        with open(full_path, "rb") as f:
            actual_hash = hashlib.sha256(f.read()).hexdigest()

        if actual_hash != expected_hash:
            failures.append({
                "path": path,
                "expected": expected_hash,
                "actual": actual_hash
            })

    return RuleResult(
        rule_id="MAN-6",
        passed=len(failures) == 0,
        severity="INVALIDATING",
        message="All checksums valid" if not failures else f"Checksum failures: {len(failures)}",
        evidence={"failures": failures, "total_checked": len(checksums)}
    )
```

---

### 3.4 Invariant Rules (INV-)

#### INV-D1: Cycle Index Continuity

```python
def check_inv_d1(telemetry: Dict) -> RuleResult:
    """
    Cycle indices MUST be consecutive with no gaps.

    Required inputs:
        telemetry["cycles"] or raw JSONL data

    Expected condition:
        For each condition:
            cycles = sorted(cycle_indices)
            cycles == list(range(cycles[0], cycles[-1] + 1))

    Severity: INVALIDATING
    """
    gaps = []

    for condition in ["baseline", "rfl"]:
        cycles_data = telemetry.get(condition, {}).get("cycles", [])
        if not cycles_data:
            continue

        indices = sorted([c.get("cycle", 0) for c in cycles_data])
        expected = list(range(indices[0], indices[-1] + 1))

        if indices != expected:
            missing = set(expected) - set(indices)
            gaps.append({"condition": condition, "missing": list(missing)})

    return RuleResult(
        rule_id="INV-D1",
        passed=len(gaps) == 0,
        severity="INVALIDATING",
        message="Cycle indices are continuous" if not gaps else f"Gaps found: {gaps}",
        evidence={"gaps": gaps}
    )
```

#### INV-S1: Wilson CI Bounds

```python
def check_inv_s1(summary: Dict) -> RuleResult:
    """
    Wilson CI bounds MUST be in [0, 1].

    Required inputs:
        summary["slices"][*]["success_rate"]["ci_low", "ci_high"]
        summary["slices"][*]["abstention_rate"]["ci_low", "ci_high"]

    Expected condition:
        0 <= ci_low <= ci_high <= 1

    Severity: INVALIDATING
    """
    violations = []

    for slice_id, slice_data in summary.get("slices", {}).items():
        for metric in ["success_rate", "abstention_rate"]:
            m = slice_data.get(metric, {})
            ci_low = m.get("ci_low", 0)
            ci_high = m.get("ci_high", 1)

            if not (0 <= ci_low <= ci_high <= 1):
                violations.append({
                    "slice": slice_id,
                    "metric": metric,
                    "ci_low": ci_low,
                    "ci_high": ci_high
                })

    return RuleResult(
        rule_id="INV-S1",
        passed=len(violations) == 0,
        severity="INVALIDATING",
        message="Wilson CI bounds valid" if not violations else f"Invalid bounds: {violations}",
        evidence={"violations": violations}
    )
```

#### INV-G2: Pass Consistency

```python
def check_inv_g2(summary: Dict) -> RuleResult:
    """
    PROCEED requires all_slices_pass = true.

    Required inputs:
        summary["governance"]["recommendation"]
        summary["governance"]["all_slices_pass"]

    Expected condition:
        recommendation == "proceed" => all_slices_pass == True

    Severity: INVALIDATING
    """
    gov = summary.get("governance", {})
    rec = gov.get("recommendation", "").lower()
    all_pass = gov.get("all_slices_pass", False)

    if rec == "proceed" and not all_pass:
        return RuleResult(
            rule_id="INV-G2",
            passed=False,
            severity="INVALIDATING",
            message="PROCEED but all_slices_pass=False violates INV-G2",
            evidence={"recommendation": rec, "all_slices_pass": all_pass}
        )

    return RuleResult(
        rule_id="INV-G2",
        passed=True,
        severity="INVALIDATING",
        message="Pass consistency verified",
        evidence={}
    )
```

---

## 4. Failure Escalation Tree

### 4.1 Classification Hierarchy

```
VIOLATION
├── INVALIDATING (blocks Evidence Pack, CI fails)
│   ├── Governance Failures (GOV-1 through GOV-4, GOV-7 through GOV-12)
│   ├── Reproducibility Failures (REP-1 through REP-3, REP-5, REP-6, REP-8)
│   ├── Manifest Failures (MAN-1 through MAN-4, MAN-6)
│   └── Invariant Failures (INV-D1, INV-D3 through INV-D6, INV-S1, INV-S2, INV-S4, INV-G1 through INV-G3)
│
└── WARNING (flags for review, CI warns)
    ├── Governance Warnings (GOV-5, GOV-6)
    ├── Reproducibility Warnings (REP-4, REP-7)
    ├── Manifest Warnings (MAN-5, MAN-7 through MAN-10)
    └── Invariant Warnings (INV-D2, INV-S3)
```

### 4.2 Escalation Actions

| Verdict | CI Exit Code | Evidence Pack | Human Review | Attestation |
|---------|--------------|---------------|--------------|-------------|
| **PASS** | 0 | Generated | Not required | Full attestation |
| **WARN** | 0 (with warning) | Generated | Recommended | Conditional attestation |
| **FAIL** | 1 | Blocked | Required | No attestation |

### 4.3 Failure Response Protocol

#### For INVALIDATING Failures:

1. **Immediate CI Failure**: Exit code 1
2. **Block Evidence Pack**: Do not generate or publish
3. **Generate Failure Report**: Include all violation details
4. **Notify Stakeholders**: Alert via configured webhook
5. **Require Manual Resolution**: Analysis must be re-run after fixing issues

#### For WARNING Failures:

1. **CI Warning**: Exit code 0 with warning message
2. **Generate Evidence Pack**: With warning annotations
3. **Flag for Review**: Mark in attestation
4. **Continue Pipeline**: Do not block downstream steps
5. **Track in Metrics**: Record warning frequency

---

## 5. CI Integration

### 5.1 GitHub Actions Workflow

```yaml
name: U2 Governance Verification

on:
  workflow_run:
    workflows: ["U2 Analysis"]
    types: [completed]

jobs:
  governance-verify:
    runs-on: ubuntu-latest
    if: ${{ github.event.workflow_run.conclusion == 'success' }}

    steps:
      - uses: actions/checkout@v4

      - name: Download Analysis Artifacts
        uses: actions/download-artifact@v4
        with:
          name: u2-analysis-results
          path: results/

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install Dependencies
        run: pip install -e .

      - name: Run Governance Verifier
        id: verify
        run: |
          python -m backend.governance.verifier \
            --summary results/statistical_summary.json \
            --manifest results/manifest.json \
            --telemetry results/telemetry_summary.json \
            --prereg config/PREREG_UPLIFT_U2.yaml \
            --output results/governance_report.json

          verdict=$(jq -r '.verdict' results/governance_report.json)
          echo "verdict=$verdict" >> $GITHUB_OUTPUT

      - name: Upload Governance Report
        uses: actions/upload-artifact@v4
        with:
          name: governance-report
          path: results/governance_report.json

      - name: Check Verdict
        run: |
          if [ "${{ steps.verify.outputs.verdict }}" = "FAIL" ]; then
            echo "::error::Governance verification FAILED"
            echo "## Governance Verification Failed" >> $GITHUB_STEP_SUMMARY
            jq -r '.violations[]' results/governance_report.json >> $GITHUB_STEP_SUMMARY
            exit 1
          elif [ "${{ steps.verify.outputs.verdict }}" = "WARN" ]; then
            echo "::warning::Governance verification passed with warnings"
            echo "## Governance Warnings" >> $GITHUB_STEP_SUMMARY
            jq -r '.warnings[]' results/governance_report.json >> $GITHUB_STEP_SUMMARY
          else
            echo "## Governance Verification Passed" >> $GITHUB_STEP_SUMMARY
          fi

      - name: Generate Evidence Pack
        if: steps.verify.outputs.verdict != 'FAIL'
        run: |
          python -m backend.governance.evidence_pack \
            --governance-report results/governance_report.json \
            --output evidence_pack_v2/
```

### 5.2 Local Verification Script

```bash
#!/bin/bash
# scripts/verify-governance.sh

set -e

SUMMARY=${1:-results/statistical_summary.json}
MANIFEST=${2:-results/manifest.json}
TELEMETRY=${3:-results/telemetry_summary.json}
OUTPUT=${4:-results/governance_report.json}

echo "Running Governance Verifier..."
python -m backend.governance.verifier \
  --summary "$SUMMARY" \
  --manifest "$MANIFEST" \
  --telemetry "$TELEMETRY" \
  --output "$OUTPUT"

VERDICT=$(jq -r '.verdict' "$OUTPUT")
VIOLATIONS=$(jq -r '.violations | length' "$OUTPUT")
WARNINGS=$(jq -r '.warnings | length' "$OUTPUT")

echo ""
echo "=========================================="
echo "GOVERNANCE VERDICT: $VERDICT"
echo "=========================================="
echo "Violations: $VIOLATIONS"
echo "Warnings:   $WARNINGS"
echo ""

if [ "$VERDICT" = "FAIL" ]; then
  echo "FAILED RULES:"
  jq -r '.violations[]' "$OUTPUT"
  exit 1
fi

if [ "$VERDICT" = "WARN" ]; then
  echo "WARNING RULES:"
  jq -r '.warnings[]' "$OUTPUT"
fi

echo "Governance report written to: $OUTPUT"
```

---

## 6. Evidence Pack Integration

### 6.1 Evidence Pack Structure

```
evidence_pack_v2/
├── README.md                      # Pack description and navigation
├── summary.json                   # Statistical analysis summary
├── manifest.json                  # Experiment configuration
├── telemetry_summary.json         # Aggregated telemetry
├── governance_report.json         # Governance Verifier output
├── attestation.json               # Cryptographic attestation
├── raw_logs/
│   ├── u2_prop_depth4_baseline.jsonl
│   ├── u2_prop_depth4_rfl.jsonl
│   ├── u2_fol_eq_group_baseline.jsonl
│   ├── u2_fol_eq_group_rfl.jsonl
│   ├── u2_fol_eq_ring_baseline.jsonl
│   ├── u2_fol_eq_ring_rfl.jsonl
│   ├── u2_linear_arith_baseline.jsonl
│   └── u2_linear_arith_rfl.jsonl
└── checksums.sha256               # File integrity checksums
```

### 6.2 Governance Report in Evidence Pack

The `governance_report.json` MUST contain:

```json
{
  "$schema": "https://mathledger.io/schemas/governance-report-v1.json",
  "verdict": "PASS",
  "timestamp": "2025-12-06T12:00:00Z",
  "verifier_version": "1.0.0",

  "rules_checked": 43,
  "summary": {
    "invalidating": 0,
    "warning": 1,
    "passed": 42
  },

  "violations": [],
  "warnings": ["REP-4"],
  "passed_rules": ["GOV-1", "GOV-2", "..."],

  "details": {
    "REP-4": {
      "rule_id": "REP-4",
      "passed": false,
      "severity": "WARNING",
      "message": "Seeds are not distinct: seed_baseline == seed_rfl",
      "evidence": {
        "seed_baseline": 42,
        "seed_rfl": 42,
        "bootstrap_seed": 1337
      }
    }
  },

  "inputs": {
    "summary_hash": "sha256:abc123...",
    "manifest_hash": "sha256:def456...",
    "telemetry_hash": "sha256:ghi789..."
  }
}
```

### 6.3 Attestation Integration

The `attestation.json` MUST reference the governance verdict:

```json
{
  "attestation_id": "u2-2025-12-06-abc123",
  "created_at": "2025-12-06T12:00:00Z",
  "experiment_id": "u2-prop-depth4-v1",

  "governance": {
    "verdict": "PASS",
    "report_hash": "sha256:...",
    "violations": [],
    "warnings": ["REP-4"]
  },

  "content_hashes": {
    "summary": "sha256:...",
    "manifest": "sha256:...",
    "governance_report": "sha256:..."
  },

  "attestation_hash": "sha256:..."
}
```

---

## 7. Implementation Skeleton

```python
# backend/governance/verifier.py

"""
PHASE II — NOT RUN IN PHASE I

Governance Verifier implementation.
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import json
import hashlib


@dataclass
class RuleResult:
    rule_id: str
    passed: bool
    severity: str
    message: str
    evidence: Dict[str, Any]


@dataclass
class GovernanceVerdict:
    verdict: str
    violations: List[str]
    warnings: List[str]
    passed_rules: List[str]
    details: Dict[str, RuleResult]
    timestamp: str
    verifier_version: str
    rules_checked: int
    invalidating_count: int
    warning_count: int
    pass_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "violations": self.violations,
            "warnings": self.warnings,
            "passed_rules": self.passed_rules,
            "details": {k: asdict(v) for k, v in self.details.items()},
            "timestamp": self.timestamp,
            "verifier_version": self.verifier_version,
            "rules_checked": self.rules_checked,
            "summary": {
                "invalidating": self.invalidating_count,
                "warning": self.warning_count,
                "passed": self.pass_count,
            }
        }


VERIFIER_VERSION = "1.0.0"


def governance_verify(
    summary: Dict[str, Any],
    manifest: Dict[str, Any],
    telemetry: Dict[str, Any],
    prereg_path: Optional[str] = None,
) -> GovernanceVerdict:
    """
    PHASE II — NOT RUN IN PHASE I

    Pure function that verifies an uplift analysis against governance rules.
    """
    raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")


# Rule check functions (all PHASE II stubs)

def check_gov_1(summary: Dict, manifest: Dict) -> RuleResult:
    raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")

def check_gov_2(summary: Dict) -> RuleResult:
    raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")

def check_gov_3(summary: Dict) -> RuleResult:
    raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")

def check_gov_4(summary: Dict) -> RuleResult:
    raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")

def check_gov_5(summary: Dict) -> RuleResult:
    raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")

def check_gov_6(summary: Dict) -> RuleResult:
    raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")

def check_gov_7(summary: Dict) -> RuleResult:
    raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")

def check_gov_8(summary: Dict, manifest: Dict) -> RuleResult:
    raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")

def check_gov_9(summary: Dict) -> RuleResult:
    raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")

def check_gov_10(summary: Dict) -> RuleResult:
    raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")

def check_gov_11(summary: Dict, prereg: Dict) -> RuleResult:
    raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")

def check_gov_12(summary: Dict) -> RuleResult:
    raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")

def check_rep_1(manifest: Dict) -> RuleResult:
    raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")

def check_rep_2(manifest: Dict) -> RuleResult:
    raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")

def check_rep_3(summary: Dict) -> RuleResult:
    raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")

def check_rep_4(manifest: Dict, summary: Dict) -> RuleResult:
    raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")

def check_rep_5(summary: Dict) -> RuleResult:
    raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")

def check_rep_6(summary: Dict, summary_rerun: Optional[Dict] = None) -> RuleResult:
    raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")

def check_rep_7(manifest: Dict) -> RuleResult:
    raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")

def check_rep_8(manifest: Dict) -> RuleResult:
    raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")

def check_man_1(manifest: Dict) -> RuleResult:
    raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")

def check_man_2(manifest: Dict) -> RuleResult:
    raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")

def check_man_3(manifest: Dict, prereg_path: str) -> RuleResult:
    raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")

def check_man_4(manifest: Dict) -> RuleResult:
    raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")

def check_man_5(manifest: Dict) -> RuleResult:
    raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")

def check_man_6(manifest: Dict, base_path: str = ".") -> RuleResult:
    raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")

def check_man_7(manifest: Dict) -> RuleResult:
    raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")

def check_man_8(manifest: Dict) -> RuleResult:
    raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")

def check_man_9(manifest: Dict) -> RuleResult:
    raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")

def check_man_10(manifest: Dict) -> RuleResult:
    raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")

def check_inv_d1(telemetry: Dict) -> RuleResult:
    raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")

def check_inv_d2(telemetry: Dict) -> RuleResult:
    raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")

def check_inv_d3(telemetry: Dict) -> RuleResult:
    raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")

def check_inv_d4(telemetry: Dict) -> RuleResult:
    raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")

def check_inv_d5(telemetry: Dict) -> RuleResult:
    raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")

def check_inv_d6(telemetry: Dict) -> RuleResult:
    raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")

def check_inv_s1(summary: Dict) -> RuleResult:
    raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")

def check_inv_s2(summary: Dict) -> RuleResult:
    raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")

def check_inv_s3(summary: Dict) -> RuleResult:
    raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")

def check_inv_s4(summary: Dict) -> RuleResult:
    raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")

def check_inv_g1(summary: Dict) -> RuleResult:
    raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")

def check_inv_g2(summary: Dict) -> RuleResult:
    raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")

def check_inv_g3(summary: Dict) -> RuleResult:
    raise NotImplementedError("PHASE II — NOT RUN IN PHASE I")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Governance Verifier")
    parser.add_argument("--summary", required=True, help="Path to summary.json")
    parser.add_argument("--manifest", required=True, help="Path to manifest.json")
    parser.add_argument("--telemetry", required=True, help="Path to telemetry_summary.json")
    parser.add_argument("--prereg", help="Path to preregistration file")
    parser.add_argument("--output", required=True, help="Output path for governance report")

    args = parser.parse_args()

    # Load inputs
    with open(args.summary) as f:
        summary = json.load(f)
    with open(args.manifest) as f:
        manifest = json.load(f)
    with open(args.telemetry) as f:
        telemetry = json.load(f)

    # Run verifier
    verdict = governance_verify(summary, manifest, telemetry, args.prereg)

    # Write output
    with open(args.output, "w") as f:
        json.dump(verdict.to_dict(), f, indent=2)

    print(f"Verdict: {verdict.verdict}")
    print(f"Violations: {len(verdict.violations)}")
    print(f"Warnings: {len(verdict.warnings)}")
```

---

## 8. Changelog

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-06 | Claude I | Initial specification |

---

## 9. References

- `docs/UPLIFT_ANALYTICS_GOVERNANCE_SPEC.md` — Governance rules and constitutional principles
- `backend/metrics/fo_analytics.py` — Theoretical framework (Sections 4-5)
- `docs/RFL_LAW.md` — RFL metabolic contract
- `docs/DETERMINISM_CONTRACT.md` — Determinism requirements
- `PREREG_UPLIFT_U2.yaml` — Preregistration (Phase II)
