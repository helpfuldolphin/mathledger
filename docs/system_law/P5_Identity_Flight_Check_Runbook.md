# P5 Identity Flight Check Runbook

**Document Version:** 1.0.0
**Status:** Operational Guidance
**Phase:** P5 (Real Telemetry Transition)
**Date:** 2025-12-11

---

## Table of Contents

1. [Purpose](#1-purpose)
2. [Failure Drill Scenarios](#2-failure-drill-scenarios)
3. [Identity Flight Check Script Specification](#3-identity-flight-check-script-specification)
4. [P5 Identity Flight Check Runbook (12 Steps)](#4-p5-identity-flight-check-runbook-12-steps)
5. [Smoke-Test Readiness Checklist](#5-smoke-test-readiness-checklist)

---

## 1. Purpose

This runbook provides operational guidance for engineers enabling `RealTelemetryAdapter` in P5. It includes:

- Failure drill scenarios to recognize identity issues
- A specification for automated pre-flight checks
- A 12-step manual runbook
- A smoke-test readiness checklist

**Prerequisite**: Read `SliceIdentity_PhaseX_Invariants.md` Section 7 (P5 Real Telemetry Identity Failure Modes).

---

## 2. Failure Drill Scenarios

### 2.1 FM-001: Config Source Divergence

#### What Engineer Sees

**Logs:**
```
2025-12-11T14:32:01.123Z [ERROR] slice_identity: SI-001 FAIL: Fingerprint mismatch
2025-12-11T14:32:01.124Z [ERROR] slice_identity: computed=a3f8c9d2e1... baseline=7b2e4f6a8c...
2025-12-11T14:32:01.125Z [WARN]  first_light: identity_verified=false, advisory_block=true
2025-12-11T14:32:01.126Z [INFO]  telemetry_adapter: Source switched to REAL_TELEMETRY
2025-12-11T14:32:01.127Z [ERROR] evidence_pack: evidence_admissibility=INADMISSIBLE
```

**Metrics:**
```
slice_identity_verified{slice="arithmetic_simple"} 0
slice_identity_fingerprint_match{slice="arithmetic_simple"} 0
p5_identity_failures_total{failure_mode="FM-001"} 1
```

**Console Tile:**
```json
{
  "tile_type": "slice_identity",
  "status": "ERROR",
  "headline": "Identity BLOCK advisory (critical invariant failure)",
  "alerts": [
    {"level": "ERROR", "message": "SI-001 FAIL: Fingerprint mismatch"}
  ]
}
```

#### Hypothesized Root Cause

| Hypothesis | Investigation |
|------------|---------------|
| Env var override | Check `SLICE_DEPTH_MAX`, `SLICE_ATOMS` env vars in prod |
| Config management divergence | Compare Consul/etcd values vs file config |
| Feature flag enabled | Check LaunchDarkly/Split.io flags for slice params |
| Local dev config leaked | Verify no `.env.local` or dev overrides in prod |

#### Drill Resolution
```bash
# 1. Dump production config
kubectl exec -it runner-pod -- cat /app/config/slice.yaml > prod_config.yaml

# 2. Dump synthetic config
cat tests/fixtures/slice_config.yaml > synthetic_config.yaml

# 3. Diff
diff -u synthetic_config.yaml prod_config.yaml

# 4. Identify divergent keys, align, re-deploy
```

---

### 2.2 FM-002: Runtime Parameter Injection

#### What Engineer Sees

**Logs:**
```
2025-12-11T14:32:01.123Z [INFO]  first_light: cycle=1 identity_stable=true
2025-12-11T14:32:02.456Z [INFO]  first_light: cycle=2 identity_stable=true
...
2025-12-11T14:35:42.789Z [INFO]  autoscaler: Adjusting resource limits for runner-pod
2025-12-11T14:35:43.012Z [WARN]  slice_identity: SI-002 WARN: Drift detected mid-run
2025-12-11T14:35:43.013Z [WARN]  first_light: cycle=142 identity_stable=false
2025-12-11T14:35:43.014Z [WARN]  slice_identity: changed_params=["resource_limit_mb"]
```

**Metrics:**
```
slice_identity_drift_events_total{slice="arithmetic_simple",severity="PARAMETRIC"} 1
slice_identity_stable{slice="arithmetic_simple"} 0
p5_identity_failures_total{failure_mode="FM-002"} 1
```

**Console Tile:**
```json
{
  "tile_type": "slice_identity",
  "status": "WARN",
  "headline": "Drift detected (1 events)",
  "identity_summary": {
    "consecutive_stable_cycles": 0,
    "drift_events_24h": 1
  }
}
```

#### Hypothesized Root Cause

| Hypothesis | Investigation |
|------------|---------------|
| Auto-scaler modified config | Check HPA/VPA events during run window |
| Canary deployment triggered | Check deployment rollout status |
| Hot-reload system active | Check config reload timestamps |
| Sidecar injection modified env | Check istio/envoy proxy injection |

#### Drill Resolution
```bash
# 1. Check for HPA/VPA events
kubectl get events --field-selector reason=SuccessfulRescale

# 2. Disable auto-scaling during P5 window
kubectl patch hpa runner-hpa -p '{"spec":{"minReplicas":3,"maxReplicas":3}}'

# 3. Lock config reload
kubectl annotate deployment runner config.reload.enabled=false

# 4. Re-run P5 with config lock
```

---

### 2.3 FM-003: Environment-Specific Gates

#### What Engineer Sees

**Logs:**
```
2025-12-11T14:32:01.123Z [ERROR] slice_identity: SI-005 FAIL: Fingerprint mismatch
2025-12-11T14:32:01.124Z [DEBUG] slice_identity: Gate divergence detected
2025-12-11T14:32:01.125Z [DEBUG] slice_identity: gates.coverage.ci_lower_min: synthetic=0.80, prod=0.70
2025-12-11T14:32:01.126Z [DEBUG] slice_identity: gates.abstention.max_rate_pct: synthetic=15, prod=20
2025-12-11T14:32:01.127Z [WARN]  p4_drift_context: identity_divergence_type=FINGERPRINT_MISMATCH
```

**Metrics:**
```
slice_identity_fingerprint_match{slice="arithmetic_simple"} 0
slice_gate_divergence{gate="coverage.ci_lower_min"} 1
slice_gate_divergence{gate="abstention.max_rate_pct"} 1
p5_identity_failures_total{failure_mode="FM-003"} 1
```

**Console Tile:**
```json
{
  "tile_type": "slice_identity",
  "status": "ERROR",
  "headline": "Identity verification failed: 1 violation(s)",
  "alerts": [
    {"level": "ERROR", "message": "SI-005 FAIL: Fingerprint mismatch - gates diverge"}
  ]
}
```

#### Hypothesized Root Cause

| Hypothesis | Investigation |
|------------|---------------|
| Prod gates more permissive | Compare gate values in prod vs test |
| Regional config differences | Check if multi-region deploys have different gates |
| A/B test gate variants | Check experiment framework for gate overrides |
| Legacy gate migration incomplete | Verify all environments migrated to new gate schema |

#### Drill Resolution
```bash
# 1. Extract prod gate config
kubectl exec runner-pod -- python -c "from config import gates; print(gates)"

# 2. Compare against synthetic
python -c "
from tests.fixtures import synthetic_config
from backend.topology.first_light import compute_slice_fingerprint

syn_gates = synthetic_config['gates']
prod_gates = {...}  # from step 1

for key in set(syn_gates) | set(prod_gates):
    if syn_gates.get(key) != prod_gates.get(key):
        print(f'DIVERGE: {key}: syn={syn_gates.get(key)} prod={prod_gates.get(key)}')
"

# 3. Align gates or document acceptable divergence
```

---

### 2.4 FM-004: Curriculum Version Skew

#### What Engineer Sees

**Logs:**
```
2025-12-11T14:32:01.123Z [ERROR] slice_identity: SI-006 FAIL: Version incompatible
2025-12-11T14:32:01.124Z [ERROR] slice_identity: curriculum_version: synthetic=1.2.0, prod=1.1.0
2025-12-11T14:32:01.125Z [WARN]  provenance: Curriculum fingerprint mismatch
2025-12-11T14:32:01.126Z [WARN]  provenance: synthetic_fp=abc123... prod_fp=def456...
2025-12-11T14:32:01.127Z [ERROR] evidence_pack: SI-004 provenance chain broken
```

**Metrics:**
```
curriculum_version{env="synthetic"} 1.2
curriculum_version{env="production"} 1.1
slice_identity_invariant_status{invariant="SI-006"} 0
p5_identity_failures_total{failure_mode="FM-004"} 1
```

**Console Tile:**
```json
{
  "tile_type": "slice_identity",
  "status": "ERROR",
  "headline": "Identity verification failed: 1 violation(s)",
  "invariant_status": {
    "SI-006": "FAIL"
  }
}
```

#### Hypothesized Root Cause

| Hypothesis | Investigation |
|------------|---------------|
| Prod rollback to old version | Check deployment history |
| Synthetic ahead of prod | Verify test env uses same version as prod |
| Blue/green serving old version | Check active deployment color |
| Curriculum update not deployed | Verify CI/CD completed curriculum update |

#### Drill Resolution
```bash
# 1. Check deployed curriculum version
kubectl exec runner-pod -- cat /app/curriculum/VERSION

# 2. Check synthetic curriculum version
cat curriculum/VERSION

# 3. If major version differs, treat P5 as new baseline
# 4. If minor/patch differs, update prod or downgrade synthetic

# 5. Re-verify after alignment
python -m backend.topology.first_light.slice_identity --verify-curriculum
```

---

## 3. Identity Flight Check Script Specification

### 3.1 Script Overview

```python
#!/usr/bin/env python3
"""
P5 Identity Flight Check Script

SPEC-ONLY: This is a specification for implementation.
Status: NOT YET IMPLEMENTED

Usage:
    python check_p5_identity_alignment.py \
        --synthetic-config path/to/synthetic.yaml \
        --prod-config path/to/prod.yaml \
        --p4-evidence-pack path/to/evidence.json

Output:
    OK          - All checks pass, safe to enable RealTelemetryAdapter
    INVESTIGATE - Non-critical issues found, review before proceeding
    BLOCK       - Critical issues found, do not enable RealTelemetryAdapter
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class CheckResult(Enum):
    OK = "OK"
    INVESTIGATE = "INVESTIGATE"
    BLOCK = "BLOCK"


@dataclass
class CheckReport:
    """Report from P5 identity alignment check."""

    overall_status: CheckResult = CheckResult.OK
    checks: List[Dict[str, Any]] = field(default_factory=list)
    blocking_issues: List[str] = field(default_factory=list)
    investigation_items: List[str] = field(default_factory=list)

    def add_check(
        self,
        name: str,
        status: CheckResult,
        details: str,
        invariant: Optional[str] = None,
    ) -> None:
        """Add a check result."""
        self.checks.append({
            "name": name,
            "status": status.value,
            "details": details,
            "invariant": invariant,
        })

        if status == CheckResult.BLOCK:
            self.blocking_issues.append(f"{name}: {details}")
            self.overall_status = CheckResult.BLOCK
        elif status == CheckResult.INVESTIGATE and self.overall_status != CheckResult.BLOCK:
            self.investigation_items.append(f"{name}: {details}")
            self.overall_status = CheckResult.INVESTIGATE

    def to_report(self) -> str:
        """Generate human-readable report."""
        lines = [
            "=" * 60,
            "P5 IDENTITY FLIGHT CHECK REPORT",
            "=" * 60,
            "",
            f"Overall Status: {self.overall_status.value}",
            "",
        ]

        if self.blocking_issues:
            lines.append("BLOCKING ISSUES:")
            for issue in self.blocking_issues:
                lines.append(f"  ❌ {issue}")
            lines.append("")

        if self.investigation_items:
            lines.append("INVESTIGATION ITEMS:")
            for item in self.investigation_items:
                lines.append(f"  ⚠️  {item}")
            lines.append("")

        lines.append("CHECK DETAILS:")
        for check in self.checks:
            icon = {"OK": "✅", "INVESTIGATE": "⚠️", "BLOCK": "❌"}[check["status"]]
            lines.append(f"  {icon} [{check.get('invariant', 'N/A')}] {check['name']}")
            lines.append(f"      {check['details']}")

        lines.extend(["", "=" * 60])

        return "\n".join(lines)


def check_p5_identity_alignment(
    synthetic_config: Dict[str, Any],
    prod_config: Dict[str, Any],
    p4_evidence_pack: Optional[Dict[str, Any]] = None,
) -> CheckReport:
    """
    Compare synthetic vs production configs for P5 alignment.

    SPEC-ONLY: Implementation sketch.

    Args:
        synthetic_config: Config used in P3/P4 synthetic runs
        prod_config: Config from production environment
        p4_evidence_pack: Optional P4 evidence pack for baseline comparison

    Returns:
        CheckReport with OK / INVESTIGATE / BLOCK status
    """
    report = CheckReport()

    # -------------------------------------------------------------------------
    # CHECK 1: SI-001 - Fingerprint Match
    # -------------------------------------------------------------------------
    # SPEC: compute_slice_fingerprint() from slice_identity.py
    syn_fp = _compute_fingerprint(synthetic_config)
    prod_fp = _compute_fingerprint(prod_config)

    if syn_fp == prod_fp:
        report.add_check(
            name="Fingerprint Match",
            status=CheckResult.OK,
            details=f"Fingerprints match: {syn_fp[:16]}...",
            invariant="SI-001",
        )
    else:
        report.add_check(
            name="Fingerprint Match",
            status=CheckResult.BLOCK,
            details=f"MISMATCH: syn={syn_fp[:16]}... prod={prod_fp[:16]}...",
            invariant="SI-001",
        )

    # -------------------------------------------------------------------------
    # CHECK 2: SI-002 - Config Immutability Controls
    # -------------------------------------------------------------------------
    # SPEC: Check for hot-reload, auto-scaling indicators
    hot_reload = prod_config.get("_meta", {}).get("hot_reload_enabled", False)

    if not hot_reload:
        report.add_check(
            name="Config Immutability",
            status=CheckResult.OK,
            details="Hot-reload disabled",
            invariant="SI-002",
        )
    else:
        report.add_check(
            name="Config Immutability",
            status=CheckResult.INVESTIGATE,
            details="Hot-reload enabled - may cause mid-run drift",
            invariant="SI-002",
        )

    # -------------------------------------------------------------------------
    # CHECK 3: SI-003 - Drift Detection Wiring
    # -------------------------------------------------------------------------
    # SPEC: Verify drift_guard hooks configured
    drift_guard_enabled = prod_config.get("_meta", {}).get("drift_guard_enabled", True)

    if drift_guard_enabled:
        report.add_check(
            name="Drift Detection",
            status=CheckResult.OK,
            details="Drift guard enabled",
            invariant="SI-003",
        )
    else:
        report.add_check(
            name="Drift Detection",
            status=CheckResult.BLOCK,
            details="Drift guard DISABLED - cannot detect runtime drift",
            invariant="SI-003",
        )

    # -------------------------------------------------------------------------
    # CHECK 4: SI-004 - Provenance Chain
    # -------------------------------------------------------------------------
    # SPEC: Compare curriculum fingerprints
    syn_curriculum_fp = synthetic_config.get("_curriculum_fingerprint")
    prod_curriculum_fp = prod_config.get("_curriculum_fingerprint")

    if syn_curriculum_fp and prod_curriculum_fp:
        if syn_curriculum_fp == prod_curriculum_fp:
            report.add_check(
                name="Curriculum Fingerprint",
                status=CheckResult.OK,
                details=f"Curriculum match: {syn_curriculum_fp[:16]}...",
                invariant="SI-004",
            )
        else:
            report.add_check(
                name="Curriculum Fingerprint",
                status=CheckResult.INVESTIGATE,
                details=f"Curriculum diverge: syn={syn_curriculum_fp[:16]}... prod={prod_curriculum_fp[:16]}...",
                invariant="SI-004",
            )
    else:
        report.add_check(
            name="Curriculum Fingerprint",
            status=CheckResult.INVESTIGATE,
            details="Curriculum fingerprint not available in one or both configs",
            invariant="SI-004",
        )

    # -------------------------------------------------------------------------
    # CHECK 5: SI-005 - P4 Evidence Baseline
    # -------------------------------------------------------------------------
    # SPEC: Compare against P4 evidence pack baseline
    if p4_evidence_pack:
        p4_baseline_fp = p4_evidence_pack.get("governance", {}).get(
            "slice_identity", {}
        ).get("baseline_fingerprint")

        if p4_baseline_fp:
            if prod_fp == p4_baseline_fp:
                report.add_check(
                    name="P4 Evidence Baseline",
                    status=CheckResult.OK,
                    details="Production matches P4 evidence baseline",
                    invariant="SI-005",
                )
            else:
                report.add_check(
                    name="P4 Evidence Baseline",
                    status=CheckResult.BLOCK,
                    details=f"Production differs from P4 baseline: prod={prod_fp[:16]}... p4={p4_baseline_fp[:16]}...",
                    invariant="SI-005",
                )
        else:
            report.add_check(
                name="P4 Evidence Baseline",
                status=CheckResult.INVESTIGATE,
                details="P4 evidence pack missing baseline fingerprint",
                invariant="SI-005",
            )
    else:
        report.add_check(
            name="P4 Evidence Baseline",
            status=CheckResult.INVESTIGATE,
            details="No P4 evidence pack provided for comparison",
            invariant="SI-005",
        )

    # -------------------------------------------------------------------------
    # CHECK 6: SI-006 - Version Compatibility
    # -------------------------------------------------------------------------
    # SPEC: Compare slice versions
    syn_version = synthetic_config.get("version", "0.0.0")
    prod_version = prod_config.get("version", "0.0.0")

    syn_major = int(syn_version.split(".")[0])
    prod_major = int(prod_version.split(".")[0])

    if syn_version == prod_version:
        report.add_check(
            name="Version Compatibility",
            status=CheckResult.OK,
            details=f"Versions match: {syn_version}",
            invariant="SI-006",
        )
    elif syn_major == prod_major:
        report.add_check(
            name="Version Compatibility",
            status=CheckResult.INVESTIGATE,
            details=f"Minor version diff: syn={syn_version} prod={prod_version}",
            invariant="SI-006",
        )
    else:
        report.add_check(
            name="Version Compatibility",
            status=CheckResult.BLOCK,
            details=f"MAJOR version diff: syn={syn_version} prod={prod_version} - treat as new baseline",
            invariant="SI-006",
        )

    # -------------------------------------------------------------------------
    # CHECK 7: Parameter-by-Parameter Diff
    # -------------------------------------------------------------------------
    # SPEC: Detailed param comparison for diagnostics
    syn_params = synthetic_config.get("params", {})
    prod_params = prod_config.get("params", {})

    differing_params = []
    all_keys = set(syn_params.keys()) | set(prod_params.keys())
    for key in sorted(all_keys):
        syn_val = syn_params.get(key)
        prod_val = prod_params.get(key)
        if syn_val != prod_val:
            differing_params.append(f"{key}: syn={syn_val} prod={prod_val}")

    if not differing_params:
        report.add_check(
            name="Parameter Alignment",
            status=CheckResult.OK,
            details="All parameters match",
            invariant="N/A",
        )
    else:
        report.add_check(
            name="Parameter Alignment",
            status=CheckResult.INVESTIGATE if len(differing_params) <= 2 else CheckResult.BLOCK,
            details=f"Differing params: {'; '.join(differing_params[:5])}{'...' if len(differing_params) > 5 else ''}",
            invariant="N/A",
        )

    # -------------------------------------------------------------------------
    # CHECK 8: Gate Alignment
    # -------------------------------------------------------------------------
    # SPEC: Compare gate configurations
    syn_gates = synthetic_config.get("gates", {})
    prod_gates = prod_config.get("gates", {})

    differing_gates = []

    def _flatten_gates(gates: Dict, prefix: str = "") -> Dict[str, Any]:
        result = {}
        for k, v in gates.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                result.update(_flatten_gates(v, key))
            else:
                result[key] = v
        return result

    syn_flat = _flatten_gates(syn_gates)
    prod_flat = _flatten_gates(prod_gates)
    all_gate_keys = set(syn_flat.keys()) | set(prod_flat.keys())

    for key in sorted(all_gate_keys):
        if syn_flat.get(key) != prod_flat.get(key):
            differing_gates.append(f"{key}: syn={syn_flat.get(key)} prod={prod_flat.get(key)}")

    if not differing_gates:
        report.add_check(
            name="Gate Alignment",
            status=CheckResult.OK,
            details="All gates match",
            invariant="N/A",
        )
    else:
        report.add_check(
            name="Gate Alignment",
            status=CheckResult.INVESTIGATE,
            details=f"Differing gates: {'; '.join(differing_gates[:3])}{'...' if len(differing_gates) > 3 else ''}",
            invariant="N/A",
        )

    return report


def _compute_fingerprint(config: Dict[str, Any]) -> str:
    """
    Compute slice fingerprint.

    SPEC: Delegates to compute_slice_fingerprint() from slice_identity.py
    """
    import hashlib
    import json

    relevant = {}
    if "params" in config:
        relevant["params"] = config["params"]
    if "gates" in config:
        relevant["gates"] = config["gates"]
    if not relevant:
        relevant = {
            k: v for k, v in config.items()
            if not k.startswith("_") and k not in ("name", "version", "description")
        }

    canonical = json.dumps(relevant, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


# =============================================================================
# CLI Entry Point (SPEC-ONLY)
# =============================================================================

def main():
    """
    CLI entry point.

    SPEC-ONLY: Implementation sketch.
    """
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="P5 Identity Flight Check")
    parser.add_argument("--synthetic-config", required=True, help="Path to synthetic config")
    parser.add_argument("--prod-config", required=True, help="Path to production config")
    parser.add_argument("--p4-evidence-pack", help="Path to P4 evidence pack JSON")
    parser.add_argument("--output-format", choices=["text", "json"], default="text")
    args = parser.parse_args()

    # Load configs
    with open(args.synthetic_config) as f:
        synthetic = yaml.safe_load(f)

    with open(args.prod_config) as f:
        prod = yaml.safe_load(f)

    evidence = None
    if args.p4_evidence_pack:
        with open(args.p4_evidence_pack) as f:
            evidence = json.load(f)

    # Run check
    report = check_p5_identity_alignment(synthetic, prod, evidence)

    # Output
    if args.output_format == "json":
        print(json.dumps({
            "status": report.overall_status.value,
            "checks": report.checks,
            "blocking_issues": report.blocking_issues,
            "investigation_items": report.investigation_items,
        }, indent=2))
    else:
        print(report.to_report())

    # Exit code
    if report.overall_status == CheckResult.BLOCK:
        sys.exit(2)
    elif report.overall_status == CheckResult.INVESTIGATE:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
```

---

## 4. P5 Identity Flight Check Runbook (12 Steps)

### Pre-Requisites

- P4 evidence pack from successful P4 run
- Access to production Kubernetes cluster
- Access to synthetic test environment
- `check_p5_identity_alignment.py` script available

### Step 1: Establish Baseline Checkpoint

```bash
# Record current state before any changes
date > p5_flight_check_$(date +%Y%m%d_%H%M%S).log
echo "Starting P5 Identity Flight Check" >> $LOG

# Capture P4 evidence baseline
cp evidence/latest_p4_evidence_pack.json p5_baseline_evidence.json
```

**Exit Criteria**: P4 evidence pack file exists and is valid JSON.

### Step 2: Extract Production Configuration

```bash
# Get slice config from production
kubectl exec -it $(kubectl get pod -l app=runner -o jsonpath='{.items[0].metadata.name}') \
    -- cat /app/config/slice.yaml > prod_slice_config.yaml

# Get curriculum version
kubectl exec -it $(kubectl get pod -l app=runner -o jsonpath='{.items[0].metadata.name}') \
    -- cat /app/curriculum/VERSION
```

**Exit Criteria**: `prod_slice_config.yaml` exists and contains valid YAML.

### Step 3: Extract Synthetic Configuration

```bash
# Get slice config used in P4 synthetic runs
cp tests/fixtures/slice_config.yaml synthetic_slice_config.yaml

# Verify it matches P4 evidence pack
python -c "
import json
with open('p5_baseline_evidence.json') as f:
    evidence = json.load(f)
print('P4 baseline fingerprint:', evidence['governance']['slice_identity']['baseline_fingerprint'][:16])
"
```

**Exit Criteria**: `synthetic_slice_config.yaml` exists and fingerprint is recorded.

### Step 4: Run Automated Alignment Check

```bash
# Run the flight check script
python check_p5_identity_alignment.py \
    --synthetic-config synthetic_slice_config.yaml \
    --prod-config prod_slice_config.yaml \
    --p4-evidence-pack p5_baseline_evidence.json \
    --output-format text | tee flight_check_report.txt

# Check exit code
echo "Exit code: $?"
```

**Exit Criteria**:
- Exit code 0 → Proceed to Step 5
- Exit code 1 → Review investigation items, proceed with caution
- Exit code 2 → STOP, resolve blocking issues before proceeding

### Step 5: Verify SI-001 (Fingerprint Match)

```bash
# Manual fingerprint verification
python -c "
from backend.topology.first_light import compute_slice_fingerprint
import yaml

with open('synthetic_slice_config.yaml') as f:
    syn = yaml.safe_load(f)
with open('prod_slice_config.yaml') as f:
    prod = yaml.safe_load(f)

syn_fp = compute_slice_fingerprint(syn)
prod_fp = compute_slice_fingerprint(prod)

print(f'Synthetic: {syn_fp}')
print(f'Production: {prod_fp}')
print(f'Match: {syn_fp == prod_fp}')
"
```

**Exit Criteria**: Fingerprints match OR divergence is documented and accepted.

### Step 6: Verify SI-002 (Config Immutability)

```bash
# Check HPA/VPA status
kubectl get hpa -l app=runner
kubectl get vpa -l app=runner

# Check for config reload annotations
kubectl get deployment runner -o jsonpath='{.metadata.annotations}'

# Verify no hot-reload
kubectl exec -it $(kubectl get pod -l app=runner -o jsonpath='{.items[0].metadata.name}') \
    -- env | grep -i reload
```

**Exit Criteria**: No auto-scaling or hot-reload affecting slice config during P5 window.

### Step 7: Verify SI-003 (Drift Detection Wiring)

```bash
# Verify drift guard is configured
kubectl exec -it $(kubectl get pod -l app=runner -o jsonpath='{.items[0].metadata.name}') \
    -- python -c "from curriculum.slice_drift_guard import compute_slice_drift_and_provenance; print('OK')"

# Check drift events route to collector
kubectl logs -l app=runner --since=1h | grep -i "drift_event"
```

**Exit Criteria**: Drift guard module loads without error.

### Step 8: Verify SI-004 (Provenance Chain)

```bash
# Compare curriculum fingerprints
python -c "
from curriculum.slice_drift_guard import build_curriculum_fingerprint
import yaml

# Load curriculum from both environments
# (Implementation depends on your curriculum loading mechanism)
print('Verify curriculum fingerprints match between synthetic and prod')
"
```

**Exit Criteria**: Curriculum fingerprints match OR gap is documented.

### Step 9: Verify SI-005 (P4 Evidence Binding)

```bash
# Verify production can bind to P4 evidence
python -c "
import json
from backend.topology.first_light import verify_slice_identity_for_p3
import yaml

with open('prod_slice_config.yaml') as f:
    prod = yaml.safe_load(f)

with open('p5_baseline_evidence.json') as f:
    evidence = json.load(f)

baseline_fp = evidence['governance']['slice_identity'].get('baseline_fingerprint')

result = verify_slice_identity_for_p3(
    slice_config=prod,
    baseline_fingerprint=baseline_fp,
)

print(f'identity_verified: {result.identity_verified}')
print(f'fingerprint_match: {result.fingerprint_match}')
print(f'violations: {result.violations}')
"
```

**Exit Criteria**: `identity_verified=True` and `fingerprint_match=True`.

### Step 10: Verify SI-006 (Version Compatibility)

```bash
# Compare versions
echo "Synthetic version:"
grep -E "^version:" synthetic_slice_config.yaml

echo "Production version:"
grep -E "^version:" prod_slice_config.yaml

# Check for major version mismatch
python -c "
import yaml
with open('synthetic_slice_config.yaml') as f:
    syn_ver = yaml.safe_load(f).get('version', '0.0.0')
with open('prod_slice_config.yaml') as f:
    prod_ver = yaml.safe_load(f).get('version', '0.0.0')

syn_major = int(syn_ver.split('.')[0])
prod_major = int(prod_ver.split('.')[0])

if syn_major != prod_major:
    print(f'WARNING: Major version mismatch! syn={syn_ver} prod={prod_ver}')
    print('Treat P5 as new baseline, not comparison to P4')
else:
    print(f'OK: Compatible versions syn={syn_ver} prod={prod_ver}')
"
```

**Exit Criteria**: Versions compatible OR major diff documented as new baseline.

### Step 11: Final Gate Decision

```bash
# Review all checks
cat flight_check_report.txt

# Make go/no-go decision
echo "============================================"
echo "P5 IDENTITY FLIGHT CHECK FINAL DECISION"
echo "============================================"
echo ""
echo "Review the above report and enter decision:"
echo "  GO       - Enable RealTelemetryAdapter"
echo "  NO-GO    - Do not enable, resolve issues first"
echo "  DEFER    - Defer decision, need more investigation"
echo ""
read -p "Decision: " DECISION
echo "Decision recorded: $DECISION at $(date)" >> flight_check_report.txt
```

**Exit Criteria**: Explicit GO/NO-GO/DEFER decision recorded.

### Step 12: Enable RealTelemetryAdapter (GO only)

```bash
# Only execute if Step 11 decision was GO

# Enable real telemetry adapter
kubectl set env deployment/runner TELEMETRY_ADAPTER=real

# Monitor first cycles
kubectl logs -f -l app=runner --since=1m | grep -E "identity_stable|identity_verified"

# Verify no immediate failures
sleep 60
kubectl logs -l app=runner --since=2m | grep -c "SI-00[1-6] FAIL"
```

**Exit Criteria**: Real telemetry enabled, no identity failures in first 60 seconds.

---

## 5. Smoke-Test Readiness Checklist

### P5 Identity Pre-Flight Smoke Test

Complete this checklist before enabling `RealTelemetryAdapter`.

```
P5 IDENTITY PRE-FLIGHT SMOKE TEST CHECKLIST
============================================

Date: ____________  Engineer: ____________  Run ID: ____________

CONFIGURATION ALIGNMENT
-----------------------
[ ] Prod config extracted (Step 2)
[ ] Synthetic config extracted (Step 3)
[ ] Automated flight check passed (Step 4)
    Exit code: ____  (0=OK, 1=INVESTIGATE, 2=BLOCK)

INVARIANT VERIFICATION
----------------------
[ ] SI-001: Fingerprints match
    Synthetic: ________________
    Production: ________________

[ ] SI-002: Config immutability verified
    HPA disabled/locked: [ ] Yes [ ] No [ ] N/A
    Hot-reload disabled: [ ] Yes [ ] No [ ] N/A

[ ] SI-003: Drift detection wired
    slice_drift_guard.py loads: [ ] Yes [ ] No

[ ] SI-004: Provenance chain continuous
    Curriculum FP match: [ ] Yes [ ] No [ ] Documented gap

[ ] SI-005: P4 evidence binding verified
    identity_verified: [ ] True [ ] False
    fingerprint_match: [ ] True [ ] False

[ ] SI-006: Version compatible
    Synthetic version: ________
    Production version: ________
    Major match: [ ] Yes [ ] No (new baseline)

FINAL GATE
----------
[ ] Flight check report reviewed
[ ] All blocking issues resolved
[ ] Investigation items documented
[ ] GO/NO-GO decision made

Decision: [ ] GO  [ ] NO-GO  [ ] DEFER
Reason: ________________________________

ENABLEMENT (GO only)
--------------------
[ ] RealTelemetryAdapter enabled
[ ] First 60s monitored, no failures
[ ] P5 run initiated

Sign-off: ________________  Date: ____________
```

---

## Appendix: Quick Reference

### Exit Codes

| Code | Status | Action |
|------|--------|--------|
| 0 | OK | Safe to enable RealTelemetryAdapter |
| 1 | INVESTIGATE | Review items, proceed with caution |
| 2 | BLOCK | Do not enable, resolve issues first |

### Failure Mode Quick Reference

| FM | Invariant | Quick Check |
|----|-----------|-------------|
| FM-001 | SI-001 | `compute_slice_fingerprint()` mismatch |
| FM-002 | SI-002 | `identity_stable=false` mid-run |
| FM-003 | SI-005 | Gate value differences |
| FM-004 | SI-006 | Curriculum/slice version mismatch |

### Emergency Rollback

If P5 identity failures occur after enabling real telemetry:

```bash
# Immediate rollback
kubectl set env deployment/runner TELEMETRY_ADAPTER=synthetic

# Capture evidence
kubectl logs -l app=runner --since=10m > p5_failure_logs.txt

# File incident
echo "P5 identity failure at $(date)" | tee -a incidents.log
```

---

*This runbook is part of the MathLedger Phase X operational documentation.*
