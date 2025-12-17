# Slice Identity Phase X Invariants

**Document Version:** 1.1.0
**Status:** Design Specification
**Phase:** X (SHADOW MODE ONLY)
**Date:** 2025-12-11

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Identity Invariants for Phase X](#2-identity-invariants-for-phase-x)
3. [Ledger Stability Requirements for P3/P4 Runs](#3-ledger-stability-requirements-for-p3p4-runs)
4. [Identity-to-P4 Divergence Implications](#4-identity-to-p4-divergence-implications)
5. [Governance Integration](#5-governance-integration)
6. [Schema References](#6-schema-references)
7. [P5 Real Telemetry Identity Failure Modes](#7-p5-real-telemetry-identity-failure-modes)
8. [TODO Anchors](#8-todo-anchors)

---

## 1. Executive Summary

Slice identity forms the foundational invariant layer that enables Phase X P3/P4 evidence chains to maintain admissibility. Without stable slice identity, the shadow observation infrastructure cannot establish the baseline conditions required for meaningful divergence analysis.

### Why Slice Identity Matters

In Phase X, the shadow observation layer compares real runner telemetry against twin predictions. For this comparison to be meaningful:

1. **The slice under observation must be uniquely identifiable** across time and system restarts
2. **Slice parameters must be immutable** during an observation run
3. **Slice drift must be detectable and logged** as a first-class governance signal
4. **Identity violations must be captured as P4 divergence evidence**

### Relationship to MathLedger Truth Anchor

MathLedger establishes verifiable mathematical truth through:
- Lean 4 verification of derivation claims
- Cryptographic commitment to proof chains
- Immutable ledger blocks with Merkle roots

Slice identity extends this trust model to the curriculum layer:
- Slice fingerprints commit to parameter configurations
- Drift detection identifies runtime deviations
- Identity invariants ensure P3/P4 observations reference stable baselines

---

## 2. Identity Invariants for Phase X

### 2.1 Invariant Table

| ID | Invariant | Description | Enforcement | Violation Impact |
|----|-----------|-------------|-------------|------------------|
| SI-001 | **Unique Slice Fingerprint** | Each slice configuration produces a unique cryptographic fingerprint | SHA-256 over canonical JSON | P4 evidence inadmissible |
| SI-002 | **Immutable Run Baseline** | Slice parameters cannot change during a P3/P4 run | Config freeze at initialization | Run invalidation |
| SI-003 | **Drift Detection Required** | Any parameter deviation must be logged as drift event | `slice_drift_guard.py` hooks | Shadow observation gaps |
| SI-004 | **Provenance Chain Continuity** | Every drift event links to curriculum fingerprint | Event schema enforcement | Audit trail broken |
| SI-005 | **Identity-P4 Binding** | P4 divergence analysis must include slice identity verification | Schema validation | False divergence signals |
| SI-006 | **Cross-Run Identity Stability** | Same slice name must map to compatible parameters across runs | Semantic versioning | Historical comparison invalid |

### 2.2 Formal Definitions

#### SI-001: Unique Slice Fingerprint

```
fingerprint(slice) = SHA256(canonical_json(slice.params ∪ slice.gates))
```

Where `canonical_json` produces:
- Sorted keys
- Minimal whitespace separators
- Deterministic floating-point representation

#### SI-002: Immutable Run Baseline

```
∀ cycle c ∈ run R:
  slice_config(c) = slice_config(R.init)
```

No parameter may be modified after run initialization.

#### SI-003: Drift Detection

```
drift_event(baseline, current) = {
  changed_params: [p | p ∈ params ∧ baseline[p] ≠ current[p]],
  severity: max(classification(p) for p in changed_params),
  status: severity_to_status(severity)
}
```

Classification follows constraint rules:
- `increasing` params: current < baseline → SEMANTIC violation
- `decreasing` params: current > baseline → SEMANTIC violation
- `boolean_true` params: baseline=true, current=false → SEMANTIC violation

#### SI-004: Provenance Chain

```
provenance_event = {
  curriculum_fingerprint: fingerprint(curriculum),
  slice_name: string,
  drift_snapshot: drift_event,
  emitted_at: deterministic_timestamp(...)
}
```

Every drift observation creates an immutable provenance record.

#### SI-005: Identity-P4 Binding

P4 divergence snapshots MUST include:
```
{
  "slice_identity": {
    "fingerprint": "abc123...",
    "name": "arithmetic_simple",
    "version": "1.0.0"
  },
  "identity_verified": true,
  "identity_drift_detected": false
}
```

#### SI-006: Cross-Run Stability

Slice names follow semantic versioning compatibility:
- Major version change → breaking parameter changes allowed
- Minor version change → additive changes only
- Patch version change → no parameter changes

---

## 3. Ledger Stability Requirements for P3/P4 Runs

### 3.1 Admissibility Conditions

For a P3/P4 shadow run to produce admissible evidence:

| Condition | Requirement | Verification Method |
|-----------|-------------|---------------------|
| **Ledger Online** | Database accessible throughout run | Health check at cycle boundaries |
| **Block Continuity** | No gaps in block sequence | Block number monotonicity check |
| **Proof Chain Integrity** | All referenced proofs exist | Foreign key validation |
| **Slice Registry Stable** | Curriculum file unchanged | Fingerprint comparison |
| **Identity Lock** | Target slice parameters frozen | Config hash verification |

### 3.2 Pre-Run Checklist

Before initiating a P3/P4 run:

```
1. Verify DATABASE_URL accessible
2. Query max(block_id) for continuity baseline
3. Load curriculum file, compute fingerprint
4. Resolve target slice by name
5. Compute slice fingerprint
6. Store (run_id, curriculum_fp, slice_fp) as immutable header
7. Initialize shadow observation with frozen config
```

### 3.3 Mid-Run Stability Monitoring

During each cycle:

```
1. Re-compute slice fingerprint from active config
2. Compare against frozen baseline
3. If drift detected:
   a. Log SliceIdentityDriftEvent
   b. Set run.identity_drift_flag = true
   c. Continue observation (SHADOW MODE: never abort)
4. Include identity_stable: bool in cycle log
```

### 3.4 Post-Run Validation

After run completion:

```
1. Verify all cycles logged identity status
2. Count identity_drift_events
3. If drift_count > 0:
   a. Mark run as IDENTITY_COMPROMISED
   b. Evidence admissibility = PARTIAL
4. Include identity summary in run report
```

---

## 4. Identity-to-P4 Divergence Implications

### 4.1 How Identity Drift Affects P4 Analysis

P4 compares real runner telemetry against shadow twin predictions. If slice identity drifts:

| Scenario | Twin Behavior | Divergence Analysis | Admissibility |
|----------|---------------|---------------------|---------------|
| No drift | Uses frozen baseline | Valid comparison | Full |
| Parametric drift | Twin uses old params | Divergence inflated | Partial with warning |
| Semantic drift | Twin predictions invalid | Comparison meaningless | Inadmissible |
| Identity mismatch | Twin/real on different slices | False divergence | Rejected |

### 4.2 Divergence Attribution

When divergence is detected, the analysis must determine:

```
divergence_attribution = {
  caused_by_slice_drift: bool,
  caused_by_governance_change: bool,
  caused_by_runner_behavior: bool,
  caused_by_twin_model_error: bool
}
```

Slice identity drift as a cause:
- Indicates infrastructure instability, not governance failure
- Should be logged separately from behavioral divergence
- Does not count toward divergence threshold triggers

### 4.3 Identity Divergence Event Schema

When slice identity diverges between real and twin:

```json
{
  "event_type": "SLICE_IDENTITY_DIVERGENCE",
  "cycle": 142,
  "timestamp": "2025-12-10T14:32:01.123Z",

  "real_identity": {
    "fingerprint": "abc123...",
    "name": "arithmetic_simple",
    "params_hash": "def456..."
  },

  "twin_identity": {
    "fingerprint": "abc123...",
    "name": "arithmetic_simple",
    "params_hash": "abc123..."
  },

  "divergence_type": "PARAMS_MISMATCH",
  "affected_params": ["depth_max", "breadth_max"],

  "impact": {
    "evidence_admissibility": "PARTIAL",
    "divergence_analysis_valid": false,
    "recommended_action": "INVESTIGATE_DRIFT_SOURCE"
  }
}
```

### 4.4 Cascade Effects

Identity drift can cascade through the P4 evidence chain:

```
Slice Identity Drift
       │
       ├──► Invalidates twin baseline
       │         │
       │         └──► All subsequent divergence measurements suspect
       │
       ├──► Breaks provenance chain continuity
       │         │
       │         └──► Historical comparison not possible
       │
       └──► Triggers governance signal
                 │
                 └──► Console tile shows identity alert
```

---

## 5. Governance Integration

### 5.1 Identity Governance Signals

Slice identity produces three governance signal types:

| Signal | Condition | Severity | Response |
|--------|-----------|----------|----------|
| `IDENTITY_STABLE` | No drift detected | INFO | Continue normal operation |
| `IDENTITY_DRIFT_PARAMETRIC` | Non-semantic param change | WARNING | Log and continue, flag evidence |
| `IDENTITY_DRIFT_SEMANTIC` | Constraint-violating change | ERROR | Log prominently, evidence inadmissible |

### 5.2 Console Tile Integration

The slice identity console tile provides:

```json
{
  "tile_type": "slice_identity",
  "status": "OK|WARN|ERROR",
  "headline": "Slice identity stable" | "Drift detected in 2 params",

  "identity_summary": {
    "current_slice": "arithmetic_simple",
    "fingerprint_stable": true,
    "drift_events_24h": 0,
    "last_drift_at": null
  },

  "active_alerts": []
}
```

### 5.3 P3 Run Pre-Flight Check

Before P3 execution authorization:

```python
def verify_slice_identity_for_p3(
    curriculum_fingerprint: str,
    slice_name: str,
    slice_config: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    """
    Pre-flight identity verification for P3 runs.

    Returns:
        (is_valid, list_of_issues)
    """
    issues = []

    # SI-001: Verify fingerprint computable
    try:
        fp = compute_slice_fingerprint(slice_config)
    except Exception as e:
        issues.append(f"SI-001 FAIL: Cannot compute fingerprint: {e}")

    # SI-004: Verify provenance chain exists
    if not curriculum_fingerprint:
        issues.append("SI-004 FAIL: No curriculum fingerprint provided")

    # SI-006: Check version compatibility
    version = slice_config.get("version", "0.0.0")
    if not is_compatible_version(slice_name, version):
        issues.append(f"SI-006 FAIL: Version {version} incompatible with registry")

    return len(issues) == 0, issues
```

---

## 6. Schema References

This document defines requirements for three schemas:

### 6.1 slice_identity_drift_view.schema.json

Purpose: Structured view of slice identity drift events for analysis dashboards.

Location: `docs/system_law/schemas/slice_identity/slice_identity_drift_view.schema.json`

### 6.2 slice_identity_console_tile.schema.json

Purpose: Console tile payload for real-time identity status monitoring.

Location: `docs/system_law/schemas/slice_identity/slice_identity_console_tile.schema.json`

### 6.3 slice_identity_governance_signal.schema.json

Purpose: Governance signal emitted when identity invariants are violated.

Location: `docs/system_law/schemas/slice_identity/slice_identity_governance_signal.schema.json`

---

## 7. P5 Real Telemetry Identity Failure Modes

### 7.1 Overview

P5 introduces the `RealTelemetryAdapter`, which transitions from synthetic/offline telemetry to live production data. This transition represents a critical boundary where identity invariants that passed in P3/P4 may suddenly fail due to environmental differences between synthetic and deployed configurations.

**Key Risk**: Identity drift that appears *only* after switching to real telemetry indicates a configuration mismatch between the synthetic test environment and the actual deployed system.

### 7.2 Failure Mode Analysis

#### FM-001: Config Source Divergence

**Symptom**: SI-001 (Unique Slice Fingerprint) fails on first real telemetry cycle despite passing all P3/P4 runs.

**Cause**: The slice configuration loaded in production differs from the synthetic baseline:
- Environment variables override file-based config
- Production config management system (Consul, etcd, etc.) returns different values
- Feature flags enabled in production but not in test

**Detection**: Compare `computed_fingerprint` from real adapter against `baseline_fingerprint` from P4 evidence pack.

**Resolution**: Audit config loading path; ensure synthetic runs use production-equivalent config sources.

#### FM-002: Runtime Parameter Injection

**Symptom**: SI-002 (Immutable Run Baseline) violated mid-run when real telemetry is active.

**Cause**: Production infrastructure injects or modifies parameters:
- Auto-scaling adjusts resource limits
- Canary deployments change config mid-run
- Hot-reload systems update slice parameters

**Detection**: `identity_stable: false` in cycle logs after telemetry source switch.

**Resolution**: Lock slice config at run initialization; disable hot-reload during P5 observation windows.

#### FM-003: Environment-Specific Gates

**Symptom**: SI-005 (Identity-P4 Binding) shows `FINGERPRINT_MISMATCH` type divergence.

**Cause**: Gate configurations include environment-specific thresholds:
- Production gates are more permissive than test gates
- CI/CD gates differ from production gates
- Regional deployments have different gate values

**Detection**: `identity_divergence_type: "FINGERPRINT_MISMATCH"` in P4 drift context.

**Resolution**: Explicitly version gate configurations; verify gate fingerprints match across environments.

#### FM-004: Curriculum Version Skew

**Symptom**: SI-006 (Cross-Run Identity Stability) fails when comparing P5 runs against P4 baseline.

**Cause**: Production deployment has different curriculum version:
- Curriculum updated in production but not in test
- Rollback left production on older version
- Blue/green deployment serves mixed versions

**Detection**: Curriculum fingerprint mismatch in provenance chain.

**Resolution**: Pin curriculum version in deployment manifest; verify version before enabling real telemetry.

### 7.3 P5 Pre-Flight Checklist

Before enabling `RealTelemetryAdapter` for P5 observations:

```
P5 IDENTITY VERIFICATION CHECKLIST

□ 1. SI-001: Compute slice fingerprint from PRODUCTION config source
      - Load config exactly as production runner does
      - Compare against P4 evidence pack baseline_slice_fingerprint
      - PASS: Fingerprints match
      - FAIL: Document mismatch, investigate config divergence

□ 2. SI-002: Verify config immutability controls
      - Confirm no hot-reload enabled for target slice
      - Confirm no auto-scaling affects slice parameters
      - Confirm no canary/shadow deployments in progress
      - PASS: Config lock mechanism verified
      - FAIL: Defer P5 until stability window

□ 3. SI-003: Initialize drift detection for real telemetry
      - Configure slice_drift_guard.py with production baseline
      - Verify drift events route to P5 evidence collector
      - PASS: Drift detection active
      - FAIL: Wire up monitoring before proceeding

□ 4. SI-004: Establish provenance chain from production
      - Compute curriculum fingerprint from deployed curriculum
      - Compare against P4 evidence pack curriculum_fingerprint
      - Link P5 run to P4 evidence chain
      - PASS: Provenance chain continuous
      - FAIL: Document gap, flag evidence as PARTIAL

□ 5. SI-005: Pre-verify identity binding
      - Call verify_slice_identity_for_p3() with production config
      - Confirm identity_verified = true
      - Confirm advisory_block = false
      - PASS: Identity binding verified
      - FAIL: Do not enable real telemetry

□ 6. SI-006: Check version compatibility
      - Verify production slice version matches P4 evidence version
      - If major version differs, treat as new baseline (not comparison)
      - PASS: Versions compatible
      - FAIL: Document version skew in P5 run header

FINAL GATE:
  All 6 checks PASS → Enable RealTelemetryAdapter
  Any check FAIL → Document, remediate, re-check
```

### 7.4 Diagnostic Queries

When P5 identity failures occur, use these diagnostic queries:

```python
# Compare synthetic vs production config
def diagnose_p5_identity_failure(
    synthetic_config: Dict[str, Any],
    production_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Diagnose identity mismatch between synthetic and production.
    """
    syn_fp = compute_slice_fingerprint(synthetic_config)
    prod_fp = compute_slice_fingerprint(production_config)

    if syn_fp == prod_fp:
        return {"match": True, "diagnosis": "No config divergence"}

    # Find differing keys
    syn_params = synthetic_config.get("params", {})
    prod_params = production_config.get("params", {})

    differing = []
    all_keys = set(syn_params.keys()) | set(prod_params.keys())
    for key in all_keys:
        syn_val = syn_params.get(key)
        prod_val = prod_params.get(key)
        if syn_val != prod_val:
            differing.append({
                "param": key,
                "synthetic": syn_val,
                "production": prod_val,
            })

    return {
        "match": False,
        "synthetic_fingerprint": syn_fp,
        "production_fingerprint": prod_fp,
        "differing_params": differing,
        "diagnosis": "Config divergence detected",
        "recommended_action": "Align production config with synthetic baseline",
    }
```

### 7.5 Evidence Admissibility Under P5 Identity Failure

| Failure Mode | Evidence Impact | Recommended Action |
|--------------|-----------------|-------------------|
| FM-001 (Config Source Divergence) | INADMISSIBLE | Halt P5, align configs, restart |
| FM-002 (Runtime Injection) | PARTIAL | Flag affected cycles, continue observation |
| FM-003 (Environment Gates) | PARTIAL | Document gate differences, adjust thresholds |
| FM-004 (Curriculum Skew) | PARTIAL or INADMISSIBLE | Depends on version delta; major = new baseline |

**Key Principle**: P5 identity failures indicate infrastructure issues, not governance failures. They should be resolved before drawing conclusions about system behavior under real telemetry.

---

## 8. TODO Anchors

### Evidence Chain Integration (Whitepaper)

```
<!-- TODO: EVIDENCE_CHAIN_WHITEPAPER_INTEGRATION
Location: docs/whitepaper.md
Section: Evidence Chain Architecture

Add subsection explaining how slice identity invariants form the
foundation of the P3/P4 evidence chain:

1. Slice fingerprint as trust anchor
2. Identity drift as evidence contamination
3. Admissibility criteria for governance decisions

Reference this document: docs/system_law/SliceIdentity_PhaseX_Invariants.md
Reference schemas: slice_identity_*.schema.json

Priority: HIGH
Tracking: CLAUDE_E_SLICE_IDENTITY
-->
```

### Slice Identity Checks Pre-P3 Run

```
<!-- TODO: SLICE_IDENTITY_PRE_P3_CHECKS
Location: backend/topology/first_light/runner.py (when implemented)
Function: FirstLightShadowRunner.__init__

Add pre-flight identity verification:

1. Call verify_slice_identity_for_p3()
2. Store frozen slice fingerprint in run metadata
3. Initialize drift monitoring hooks
4. Log identity baseline to run header

Implementation notes:
- Must complete before first cycle
- Failure should prevent run start (even in SHADOW MODE)
- Store issues in run.identity_preflight_issues

Reference: SliceIdentity_PhaseX_Invariants.md Section 5.3
Priority: REQUIRED for P3 execution authorization
Tracking: CLAUDE_E_SLICE_IDENTITY
-->
```

### Drift Guard Integration

```
<!-- TODO: DRIFT_GUARD_P4_INTEGRATION
Location: curriculum/slice_drift_guard.py
Function: compute_slice_drift_and_provenance

Extend to emit P4-compatible identity signals:

1. Add slice_fingerprint to return payload
2. Emit governance signal on SEMANTIC drift
3. Include identity_invariant_violated field
4. Support P4 divergence analyzer consumption

Reference: SliceIdentity_PhaseX_Invariants.md Section 4
Priority: MEDIUM (after P4 execution authorization)
Tracking: CLAUDE_E_SLICE_IDENTITY
-->
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-10 | Initial specification |
| 1.1.0 | 2025-12-11 | Added Section 7: P5 Real Telemetry Identity Failure Modes |

---

*This document is part of the MathLedger system law corpus. It establishes binding invariants for Phase X operations.*
