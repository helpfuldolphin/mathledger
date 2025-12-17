# Replay Safety P5 Engineering Plan

> **Status**: ENGINEERING SPECIFICATION
> **Author**: CLAUDE A (Replay Governance Layer)
> **Date**: 2025-12-11
> **Dependencies**: Replay_Safety_Governance_Law.md (Appendix C, D)

---

## 1. Signal Shape → Code Plan

### 1.1 P5 Replay Safety Extractor

Extracts replay safety signal from raw replay logs in production (P5) context.

```python
# REAL-READY: backend/health/replay_governance_adapter.py

def extract_p5_replay_safety_from_logs(
    replay_logs: List[Dict[str, Any]],
    production_run_id: str,
    expected_hashes: Optional[Dict[str, str]] = None,
    *,
    telemetry_source: str = "real",
) -> Dict[str, Any]:
    """Extract P5 replay safety signal from raw replay logs.

    This function processes production replay logs and produces a replay safety
    signal suitable for P5 real-telemetry validation.

    SHADOW MODE CONTRACT:
    - This function is read-only and side-effect free
    - The signal it produces does NOT influence any governance decisions
    - This is purely for observability and evidence collection

    Args:
        replay_logs: List of replay log entries from production run.
            Each entry should contain:
            - "trace_hash": SHA-256 hash of trace data
            - "timestamp": ISO 8601 timestamp
            - "cycle_id": Cycle identifier
            - "determinism_check": Optional dict with verification results
        production_run_id: Unique identifier for the production run.
        expected_hashes: Optional dict mapping cycle_id -> expected_hash
            for determinism verification.
        telemetry_source: Source identifier ("real" for P5, "shadow" for P4,
            "synthetic" for P3).

    Returns:
        P5 replay safety signal with:
        - status: ok | warn | block
        - determinism_rate: float (0.0-1.0)
        - hash_match_count: int
        - hash_mismatch_count: int
        - critical_incidents: List[Dict]
        - telemetry_source: str
        - production_run_id: str
        - replay_latency_ms: Optional[float]
        - p5_grade: bool (True if meets P5 requirements)

    P5 Band Thresholds (from Appendix C):
        GREEN: determinism_rate >= 0.85
        YELLOW: 0.70 <= determinism_rate < 0.85
        RED: determinism_rate < 0.70
    """
    pass  # Implementation placeholder
```

### 1.2 P5 Replay Safety Tile Builder

Builds a global health tile from P5 replay safety signal.

```python
# REAL-READY: backend/health/replay_governance_adapter.py

def build_p5_replay_governance_tile(
    p5_signal: Dict[str, Any],
    radar: Dict[str, Any],
    promotion_eval: Dict[str, Any],
    *,
    include_p5_extensions: bool = True,
) -> Dict[str, Any]:
    """Build P5-grade replay governance tile for global health surface.

    This function extends build_replay_console_tile() with P5-specific fields
    and validation.

    SHADOW MODE CONTRACT:
    - This function is read-only and side-effect free
    - The tile it produces does NOT influence any governance decisions
    - The tile does NOT influence any other tiles
    - No control flow depends on this tile

    Args:
        p5_signal: P5 replay safety signal from extract_p5_replay_safety_from_logs().
        radar: Replay governance radar view.
        promotion_eval: Promotion evaluation result.
        include_p5_extensions: If True, include P5-only fields
            (telemetry_source, production_run_id, replay_latency_ms).

    Returns:
        P5 replay governance tile with:
        - All fields from build_replay_console_tile()
        - p5_grade: bool (True if signal meets P5 requirements)
        - telemetry_source: "real" (P5-only)
        - production_run_id: str (P5-only)
        - replay_latency_ms: float (P5-only, optional)
        - determinism_band: "GREEN" | "YELLOW" | "RED"
        - phase: "P5"

    P5 Grade Requirements:
        - telemetry_source == "real"
        - production_run_id is present and non-empty
        - determinism_rate is computed from actual replay logs
    """
    pass  # Implementation placeholder
```

### 1.3 P5 Replay Safety Evidence Attachment

Attaches P5 replay safety to evidence pack with P5-specific validation.

```python
# REAL-READY: backend/health/replay_governance_adapter.py

def attach_p5_replay_governance_to_evidence(
    evidence: Dict[str, Any],
    p5_signal: Dict[str, Any],
    p5_tile: Dict[str, Any],
    *,
    validate_p5_grade: bool = True,
) -> Dict[str, Any]:
    """Attach P5 replay governance to evidence pack.

    NON-MUTATING: Returns a new dict, does not modify inputs.

    SHADOW MODE CONTRACT:
    - This function is read-only and side-effect free
    - The attachment is advisory only, does NOT gate any operations
    - This is purely for observability and evidence collection

    Args:
        evidence: Existing evidence pack (not modified).
        p5_signal: P5 replay safety signal from extract_p5_replay_safety_from_logs().
        p5_tile: P5 replay governance tile from build_p5_replay_governance_tile().
        validate_p5_grade: If True, raise ValueError if signal is not P5-grade.

    Returns:
        New evidence dict with P5 replay governance attached under:
        - evidence["governance"]["replay"]: Collapsed governance signal
        - evidence["governance"]["replay_p5"]: P5-specific extension fields
        - evidence["replay_safety_ok"]: bool
        - evidence["replay_p5_grade"]: bool

    Raises:
        ValueError: If validate_p5_grade=True and signal is not P5-grade.

    P5 Grade Validation:
        - p5_signal["telemetry_source"] == "real"
        - p5_signal["production_run_id"] is present
        - p5_tile["phase"] == "P5"
    """
    pass  # Implementation placeholder
```

---

## 2. P4 → P5 Replay Transition Checklist

```
# P4 → P5 REPLAY TRANSITION CHECKLIST
# Reference: Replay_Safety_Governance_Law.md Appendix C.5

[ ] 1. P3 BASELINE VERIFICATION
    - [ ] P3 synthetic replay signals archived
    - [ ] P3 determinism_rate >= 0.95 confirmed
    - [ ] P3 signal shape matches schema v1.0.0

[ ] 2. P4 SHADOW REFERENCE
    - [ ] P4 shadow signals collected for >= N cycles
    - [ ] P4 vs P3 alignment >= 95% on status field
    - [ ] P4 determinism_rate within 0.90-1.0 band
    - [ ] No P4 BLOCK signals unexplained

[ ] 3. P5 FIELD REQUIREMENTS
    - [ ] telemetry_source = "real" present
    - [ ] production_run_id populated (non-empty)
    - [ ] replay_latency_ms captured (optional but recommended)
    - [ ] external_correlation_id linked (if available)

[ ] 4. P5 GRADE VALIDATION
    - [ ] Determinism computed from actual replay logs
    - [ ] Hash verification performed on real traces
    - [ ] Signal passes extract_p5_replay_safety_from_logs()

[ ] 5. CANARY COMPARISON
    - [ ] P5 real runs parallel with P4 shadow for N cycles
    - [ ] P5 vs P4 divergence on status < 10%
    - [ ] No unexplained P5 BLOCK while P4 is OK

[ ] 6. SIGN-OFF
    - [ ] Manual review of first 10 P5 signals
    - [ ] Auditor checklist (Section 4) passed
    - [ ] P5 signals marked authoritative
```

---

## 3. Auditor Script

```python
# SPEC-ONLY: Auditor verification script for P5 Replay Safety
# Reference: Replay_Safety_Governance_Law.md Appendix D

def audit_p5_replay_safety(evidence_pack: Dict[str, Any]) -> str:
    """
    5-Step Auditor Script for P5 Replay Safety Verification.

    Returns: "P5_REPLAY_OK" | "P5_REPLAY_WARN" | "P5_REPLAY_INVESTIGATE"
    """

    # STEP 1: Locate replay_governance in evidence pack
    governance = evidence_pack.get("governance", {})
    replay_gov = governance.get("replay") or governance.get("replay_governance")

    if replay_gov is None:
        return "P5_REPLAY_INVESTIGATE"  # Missing replay governance

    # STEP 2: Extract status, alignment, conflict
    status = replay_gov.get("status", "").lower()
    alignment = replay_gov.get("governance_alignment", replay_gov.get("alignment", ""))
    conflict = replay_gov.get("conflict", False)

    # STEP 3: Check P5-grade indicators
    is_p5_grade = (
        replay_gov.get("telemetry_source") == "real" or
        evidence_pack.get("replay_p5_grade", False) or
        governance.get("replay_p5") is not None
    )

    if not is_p5_grade:
        # Not P5-grade evidence, investigate
        return "P5_REPLAY_INVESTIGATE"

    # STEP 4: Apply verdict logic
    if status == "block":
        return "P5_REPLAY_INVESTIGATE"

    if conflict:
        return "P5_REPLAY_INVESTIGATE"

    if alignment == "divergent":
        return "P5_REPLAY_INVESTIGATE"

    if status == "warn":
        return "P5_REPLAY_WARN"

    if status == "ok" and alignment in ("aligned", ""):
        return "P5_REPLAY_OK"

    # STEP 5: Default to investigate for unexpected states
    return "P5_REPLAY_INVESTIGATE"


# Example usage:
# verdict = audit_p5_replay_safety(evidence_pack)
# print(f"Auditor Verdict: {verdict}")
```

---

## 4. Smoke-Test Readiness Checklist

```
# P5 REPLAY SAFETY WIRING — SMOKE-TEST READINESS

## A. Module Availability
[ ] backend/health/replay_governance_adapter.py exists
[ ] All P5 functions importable without error
[ ] Schema files readable at runtime

## B. Signal Flow
[ ] extract_p5_replay_safety_from_logs() returns valid structure
[ ] build_p5_replay_governance_tile() produces JSON-safe output
[ ] attach_p5_replay_governance_to_evidence() is non-mutating

## C. P5 Field Presence
[ ] Tile includes telemetry_source="real"
[ ] Tile includes production_run_id (non-empty)
[ ] Tile includes phase="P5"
[ ] Tile includes determinism_band (GREEN/YELLOW/RED)

## D. P5 Band Logic
[ ] determinism_rate >= 0.85 → GREEN
[ ] 0.70 <= determinism_rate < 0.85 → YELLOW
[ ] determinism_rate < 0.70 → RED

## E. GGFL Integration
[ ] replay_for_alignment_view() handles P5 signals
[ ] [Replay] prefix stripping works for P5 reasons
[ ] top_reasons limited to 5

## F. Evidence Attachment
[ ] P5 evidence includes governance.replay
[ ] P5 evidence includes governance.replay_p5 (extension fields)
[ ] replay_p5_grade=true when valid P5 signal

## G. Auditor Script
[ ] audit_p5_replay_safety() returns valid verdict
[ ] P5_REPLAY_OK for nominal P5 signal
[ ] P5_REPLAY_WARN for status=warn
[ ] P5_REPLAY_INVESTIGATE for status=block or conflict=true

## H. SHADOW Mode
[ ] No function modifies control flow
[ ] All tiles include shadow_mode_contract
[ ] No gating logic present

## I. Documentation
[ ] Appendix C (P5 Expectations) references match code
[ ] Appendix D (Auditor Guide) examples are valid
[ ] P4→P5 Transition Checklist is actionable
```

---

## 5. Implementation Roadmap

| Phase | Deliverable | Depends On | Status |
|-------|-------------|------------|--------|
| P5-A | Function signatures (this doc) | Appendix C | COMPLETE |
| P5-B | Stub implementations | P5-A | PENDING |
| P5-C | Unit tests for P5 functions | P5-B | PENDING |
| P5-D | Integration with global_surface.py | P5-B | PENDING |
| P5-E | Smoke test harness | P5-C, P5-D | PENDING |
| P5-F | Canary run (P5 parallel P4) | P5-E | PENDING |
| P5-G | Sign-off and P5 activation | P5-F | PENDING |

---

**CLAUDE A: P5 Replay Wiring Plan Ready.**

---

## Appendix E: Replay P5 Robustness (v1.1.0)

The P5 replay signal extraction pipeline has been hardened to handle messy real-world production log artifacts. The robustness enhancements ensure that the pipeline completes and produces observable telemetry even when encountering imperfect data.

**Robustness Features:**
- **Rotated JSONL Support**: Directories may contain multiple `.jsonl` segment files (e.g., `segment_001.jsonl`, `segment_002.jsonl`). All segments are concatenated with deterministic ordering (sorted by filename).
- **Gzip Skip with Warning**: `.jsonl.gz` files are detected but skipped with an advisory warning (`skipped_gz_count > 0`). The gzip stdlib is not required as a dependency.
- **Malformed Line Tolerance**: Invalid JSON lines are skipped (not fatal), and counted in `malformed_line_count` with per-line advisory warnings.
- **Schema Guard**: When required P5 fields (`cycle_id`, `trace_hash`, `timestamp`) are missing, extraction continues but `schema_ok=false` is surfaced with specific advisory warnings.
- **Absolute Path Support**: The `explicit_path` parameter allows CLI tools to pass logs from arbitrary locations outside the run directory.
- **Deterministic Ordering**: All file lists and advisory warnings are sorted to ensure identical output across runs.

**All robustness features are SHADOW MODE compliant—they do not gate any operations, only provide enhanced observability.**

---

*Document Version: 1.1.0*
*Last Updated: 2025-12-12*
*Status: Engineering Specification*
