# PHASE II — NOT RUN IN PHASE I
"""
The Admissibility Law Engine for U2 Evidence Dossiers (MAAS v2).
"""
import dataclasses
import datetime
import hashlib
import json
import os
import uuid
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence
import yaml
from backend.promotion.u2_evidence import U2Dossier, CORE_ARTIFACT_IDS, REQUIRED_ENVIRONMENTS

GOVV_REQUIRED_FIELDS = ["report_id", "verified_at", "verifier_version", "dossier_hash", "gates", "overall_status", "artifact_inventory", "signature"]
GOVV_REQUIRED_GATES = ["G1", "G2", "G3", "G4"]

@dataclasses.dataclass
class AdmissibilityError:
    error_id: str
    message: str
    offending_artifact_id: str
    phase: int = 0

@dataclasses.dataclass
class AdmissibilityVerdict:
    status: str
    reason: Optional[str] = None
    code: Optional[str] = None

@dataclasses.dataclass
class AdmissibilityReport:
    decision_id: str
    executed_at: str
    verdict: AdmissibilityVerdict
    errors: List[AdmissibilityError] = dataclasses.field(default_factory=list)
    warnings: List[str] = dataclasses.field(default_factory=list)
    execution_time_ms: int = 0
    def to_json(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

def compute_dossier_hash(dossier: U2Dossier, exclude_a8: bool = False) -> str:
    """
    Computes a deterministic hash of the dossier's artifact *contents*.
    """
    hasher = hashlib.sha256()
    paths_to_hash = []
    
    for artifact_id in sorted(dossier.artifacts.keys()):
        if exclude_a8 and artifact_id == "A8":
            continue
        artifact = dossier.artifacts[artifact_id]
        if artifact.found:
            paths_to_hash.extend(sorted(artifact.paths))
            
    for path in sorted(paths_to_hash):
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
        except IOError:
            # This will be caught by other checks, but we should not crash here.
            pass
            
    return hasher.hexdigest()

def load_governance_report(dossier: U2Dossier) -> Optional[Dict[str, Any]]:
    if not (artifact := dossier.artifacts.get("A8")) or not artifact.found: return None
    try:
        with open(artifact.path, "r", encoding="utf-8") as f: return json.load(f)
    except (IOError, json.JSONDecodeError): return None

def _phase1_governance_precondition(dossier: U2Dossier, governance_report: Optional[Dict[str, Any]], errors: List[AdmissibilityError]) -> Optional[AdmissibilityVerdict]:
    if governance_report is None:
        errors.append(AdmissibilityError("HE-GV1", "Missing Governance Verifier Report (A8)", "A8", 1))
        return AdmissibilityVerdict("NOT_ADMISSIBLE", "Missing Governance Verifier Report", "HE-GV1")
    current_dossier_hash = compute_dossier_hash(dossier, exclude_a8=True)
    if governance_report.get("dossier_hash") != current_dossier_hash:
        errors.append(AdmissibilityError("HE-GV4", "Dossier hash mismatch between A8 and reality", "A8", 1))
        return AdmissibilityVerdict("NOT_ADMISSIBLE", "Dossier hash mismatch", "HE-GV4")
    if governance_report.get("overall_status") != "PASS":
        errors.append(AdmissibilityError("HE-GV2", "A8 overall_status is not 'PASS'", "A8", 1))
        return AdmissibilityVerdict("NOT_ADMISSIBLE", "Governance verification failed", "HE-GV2")
    return None

def _phase2_core_artifact_presence(dossier: U2Dossier, errors: List[AdmissibilityError]) -> Optional[AdmissibilityVerdict]:
    for artifact_id in CORE_ARTIFACT_IDS:
        if not (artifact := dossier.artifacts.get(artifact_id)) or not artifact.found:
            errors.append(AdmissibilityError("HE-S1", f"Core artifact not found: {artifact_id}", artifact_id, 2))
            return AdmissibilityVerdict("NOT_ADMISSIBLE", f"Missing core artifact: {artifact_id}", "HE-S1")
    if len(dossier.artifacts["A3"].paths) < len(REQUIRED_ENVIRONMENTS):
        errors.append(AdmissibilityError("HE-S3", f"Fewer than {len(REQUIRED_ENVIRONMENTS)} manifests found", "A3", 2))
        return AdmissibilityVerdict("NOT_ADMISSIBLE", "Incomplete environment manifests", "HE-S3")
    return None

def _phase4_artifact_integrity(dossier: U2Dossier, errors: List[AdmissibilityError]) -> Optional[AdmissibilityVerdict]:
    for path in dossier.artifacts["A3"].paths:
        try:
            with open(path, 'r') as f: manifest = json.load(f)
            if 'manifest_hash' not in manifest:
                errors.append(AdmissibilityError("DOSSIER-11", f"Manifest self-hash missing in '{path}'", "A3", 4))
                continue
            original_hash = manifest.pop('manifest_hash')
            canonical_content = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode('utf-8')
            recalculated_hash = hashlib.sha256(canonical_content).hexdigest()
            if original_hash.lower() != recalculated_hash.lower():
                errors.append(AdmissibilityError("DOSSIER-11", f"Manifest self-hash mismatch for '{path}'", "A3", 4))
        except (IOError, json.JSONDecodeError) as e:
            errors.append(AdmissibilityError("DOSSIER-17", f"Failed to read or parse manifest '{path}': {e}", "A3", 4))
    if errors: return AdmissibilityVerdict("NOT_ADMISSIBLE", "Artifact integrity check failed", errors[-1].error_id)
    return None

def _phase3_structural_integrity(dossier: U2Dossier, errors: List[AdmissibilityError]) -> Optional[AdmissibilityVerdict]: return None
def _phase5_relationship_integrity(dossier: U2Dossier, errors: List[AdmissibilityError]) -> Optional[AdmissibilityVerdict]: return None

def check_admissibility(dossier: U2Dossier) -> AdmissibilityReport:
    start_time = time.time()
    errors: List[AdmissibilityError] = []
    final_verdict: Optional[AdmissibilityVerdict] = None
    governance_report = load_governance_report(dossier)
    all_phases = [
        (_phase1_governance_precondition, (dossier, governance_report, errors)),
        (_phase2_core_artifact_presence, (dossier, errors)),
        (_phase3_structural_integrity, (dossier, errors)),
        (_phase4_artifact_integrity, (dossier, errors)),
        (_phase5_relationship_integrity, (dossier, errors)),
    ]
    for phase_func, args in all_phases:
        verdict = phase_func(*args)
        if verdict:
            final_verdict = verdict
            break
    if final_verdict is None: final_verdict = AdmissibilityVerdict("ADMISSIBLE")
    return AdmissibilityReport(
        decision_id=str(uuid.uuid4()), executed_at=datetime.datetime.utcnow().isoformat() + "Z",
        verdict=final_verdict, errors=errors,
        execution_time_ms=int((time.time() - start_time) * 1000)
    )

def build_admissibility_snapshot(report: AdmissibilityReport) -> Dict[str, Any]:
    error_codes = sorted(list(set(e.error_id for e in report.errors)))
    return {
        "schema_version": "1.0.0",
        "decision_id": report.decision_id,
        "run_id": report.verdict.run_id if hasattr(report.verdict, 'run_id') else "",
        "verdict": report.verdict.status,
        "phase_failures": error_codes,
        "core_artifacts_present": all(e.error_id != "HE-S1" for e in report.errors),
        "executed_at": report.executed_at,
    }

def build_admissibility_history(snapshots: Sequence[Dict[str, Any]], rolling_window: int = 10) -> Dict[str, Any]:
    total_cases = len(snapshots)
    if total_cases == 0: return {"total_cases": 0}
    admissible_count = sum(1 for s in snapshots if s["verdict"] == "ADMISSIBLE")
    last_n = snapshots[-rolling_window:]
    rolling_admissible = sum(1 for s in last_n if s["verdict"] == "ADMISSIBLE")
    rolling_rate = rolling_admissible / len(last_n) if last_n else 0.0
    all_error_codes = [code for s in snapshots if s["verdict"] != "ADMISSIBLE" for code in s.get("phase_failures", [])]
    error_freq = Counter(all_error_codes)
    top_blockers = {code: count for code, count in error_freq.most_common(5)}
    phase_failure_dist = {"phase_1_governance": 0, "phase_2_artifacts": 0} 
    return { "total_cases": total_cases, "admissible_count": admissible_count, "inadmissible_count": total_cases - admissible_count, "admissibility_rate": admissible_count / total_cases, "rolling_admissibility_rate (last_10)": rolling_rate, "top_recurrent_blockers": top_blockers, "phase_failure_distribution": phase_failure_dist }

def summarize_admissibility_for_global_health(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a compact summary for embedding in global_health.json."""
    verdict = snapshot.get("verdict", "UNKNOWN")
    is_admissible = verdict == "ADMISSIBLE"
    error_codes = snapshot.get("error_codes", snapshot.get("phase_failures", []))
    blocking_reasons = error_codes[:3] if not is_admissible else []
    return {
        "is_evidence_admissible": is_admissible,
        "blocking_reasons": blocking_reasons,
        "status": "OK" if is_admissible else "BLOCKED",
    }


# =============================================================================
# PHASE III: ADMISSIBILITY ANALYTICS & DIRECTOR LIGHT MAPPING
# =============================================================================

PHASE_NAMES = [
    "phase_1_governance",
    "phase_2_artifacts",
    "phase_3_structure",
    "phase_4_integrity",
    "phase_5_relationships",
]

RECURRENCE_THRESHOLD = 0.3


def build_admissibility_analytics(timeline: Dict[str, Any]) -> Dict[str, Any]:
    """Compute analytics from an admissibility timeline."""
    total_cases = timeline.get("total_cases", 0)
    admissible_count = timeline.get("admissible_count", 0)
    inadmissible_count = timeline.get("inadmissible_count", 0)
    error_code_frequency = timeline.get("error_code_frequency", {})
    phase_failure_distribution = timeline.get("phase_failure_distribution", {})

    long_run_rate = admissible_count / total_cases if total_cases > 0 else 0.0

    recurrent_errors = []
    if inadmissible_count > 0:
        for code, count in error_code_frequency.items():
            if count / inadmissible_count >= RECURRENCE_THRESHOLD:
                recurrent_errors.append(code)
    recurrent_errors = sorted(recurrent_errors)

    dominant_failure_phase = None
    max_failures = 0
    for phase, count in phase_failure_distribution.items():
        if count > max_failures:
            max_failures = count
            dominant_failure_phase = phase

    total_phase_failures = sum(phase_failure_distribution.values())
    failure_concentration = max_failures / total_phase_failures if total_phase_failures > 0 else 0.0

    error_diversity = len(error_code_frequency)

    return {
        "long_run_admissibility_rate": round(long_run_rate, 4),
        "recurrent_errors": recurrent_errors,
        "dominant_failure_phase": dominant_failure_phase,
        "failure_concentration": round(failure_concentration, 4),
        "error_diversity": error_diversity,
    }


def map_admissibility_to_director_light(snapshot: Dict[str, Any]) -> str:
    """Map an admissibility snapshot to a Director Console traffic light (GREEN|YELLOW|RED)."""
    verdict = snapshot.get("verdict", "UNKNOWN")

    if verdict == "ADMISSIBLE":
        return "GREEN"

    error_codes = snapshot.get("error_codes", snapshot.get("phase_failures", []))
    critical_prefixes = ("HE-GV", "HE-I")
    for code in error_codes:
        if any(code.startswith(prefix) for prefix in critical_prefixes):
            return "RED"

    return "YELLOW"


def summarize_admissibility_for_global_dashboard(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a summary for the global dashboard with traffic light status."""
    verdict = snapshot.get("verdict", "UNKNOWN")
    is_admissible = verdict == "ADMISSIBLE"
    error_codes = snapshot.get("error_codes", snapshot.get("phase_failures", []))

    blockers = error_codes[:3] if not is_admissible else []
    status_light = map_admissibility_to_director_light(snapshot)

    return {
        "admissible": is_admissible,
        "blockers": blockers,
        "status_light": status_light,
    }


# =============================================================================
# PHASE IV: EVIDENCE ADMISSIBILITY COMPASS & GLOBAL DASHBOARD TILE
# =============================================================================

# Thresholds for compass status determination
COMPASS_STABLE_THRESHOLD = 0.8  # >= 80% admissibility rate = STABLE
COMPASS_CRITICAL_THRESHOLD = 0.5  # < 50% admissibility rate = CRITICAL


def build_admissibility_compass(analytics: Dict[str, Any]) -> Dict[str, Any]:
    """Build an admissibility compass view from analytics.

    The compass provides directional guidance for evidence pipelines:
    - STABLE: Pipeline is healthy (>= 80% admissibility rate, no recurrent errors)
    - DEGRADING: Pipeline needs attention (50-80% rate, or recurrent errors)
    - CRITICAL: Pipeline requires immediate action (< 50% rate)
    """
    long_run_rate = analytics.get("long_run_admissibility_rate", 0.0)
    recurrent_errors = analytics.get("recurrent_errors", [])
    dominant_failure_phase = analytics.get("dominant_failure_phase")
    failure_concentration = analytics.get("failure_concentration", 0.0)

    # Determine compass status
    if long_run_rate >= COMPASS_STABLE_THRESHOLD and not recurrent_errors:
        compass_status = "STABLE"
    elif long_run_rate < COMPASS_CRITICAL_THRESHOLD:
        compass_status = "CRITICAL"
    else:
        compass_status = "DEGRADING"

    return {
        "compass_status": compass_status,
        "dominant_failure_phase": dominant_failure_phase,
        "recurrent_errors": recurrent_errors,
        "failure_concentration": failure_concentration,
    }


def evaluate_admissibility_for_promotion(
    analytics: Dict[str, Any],
    compass: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate whether the evidence pipeline is promotable.

    Determines if the evidence quality is sufficient for promotion:
    - OK: Compass is STABLE, no blocking errors
    - WARN: Compass is DEGRADING, promotion possible with caution
    - BLOCK: Compass is CRITICAL, promotion not recommended
    """
    compass_status = compass.get("compass_status", "CRITICAL")
    recurrent_errors = compass.get("recurrent_errors", [])

    # Identify blocking error codes (HE-GV* and HE-I* are promotion blockers)
    blocking_error_codes = []
    for code in recurrent_errors:
        if code.startswith("HE-GV") or code.startswith("HE-I"):
            blocking_error_codes.append(code)
    blocking_error_codes = sorted(blocking_error_codes)

    # Determine promotion status
    if compass_status == "STABLE":
        promotion_ok = True
        status = "OK"
    elif compass_status == "DEGRADING":
        # DEGRADING with blocking errors = BLOCK
        if blocking_error_codes:
            promotion_ok = False
            status = "BLOCK"
        else:
            promotion_ok = True
            status = "WARN"
    else:  # CRITICAL
        promotion_ok = False
        status = "BLOCK"

    return {
        "promotion_ok": promotion_ok,
        "status": status,
        "blocking_error_codes": blocking_error_codes,
    }


def build_admissibility_director_panel(
    snapshot: Dict[str, Any],
    compass: Dict[str, Any],
    promotion_eval: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the Director's admissibility panel for dashboard display.

    Combines snapshot, compass, and promotion evaluation into a unified
    panel view suitable for the Director Console.
    """
    status_light = map_admissibility_to_director_light(snapshot)
    compass_status = compass.get("compass_status", "CRITICAL")
    promotion_status = promotion_eval.get("status", "BLOCK")

    # Generate neutral headline
    if status_light == "GREEN" and compass_status == "STABLE":
        headline = "Evidence pipeline operating normally"
    elif status_light == "GREEN":
        headline = "Current evidence admissible, pipeline showing degradation"
    elif compass_status == "CRITICAL":
        headline = "Evidence pipeline requires immediate attention"
    elif promotion_status == "BLOCK":
        headline = "Evidence pipeline blocked for promotion"
    elif compass_status == "DEGRADING":
        headline = "Evidence pipeline degrading, monitoring recommended"
    else:
        headline = "Evidence admissibility check in progress"

    # Extract long_run_admissibility_rate from snapshot if available
    long_run_rate = snapshot.get("long_run_admissibility_rate", None)

    result = {
        "status_light": status_light,
        "headline": headline,
    }

    # Only include long_run_admissibility_rate if available
    if long_run_rate is not None:
        result["long_run_admissibility_rate"] = long_run_rate

    return result


# =============================================================================
# PHASE V: ADMISSIBILITY AS GLOBAL GATE
# =============================================================================

def summarize_admissibility_for_global_console(
    snapshot: Dict[str, Any],
    compass: Dict[str, Any],
    promotion_eval: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate a unified summary for the global console.

    Combines snapshot, compass, and promotion evaluation into a single
    structure suitable for the global console display.
    """
    verdict = snapshot.get("verdict", "UNKNOWN")
    is_admissible = verdict == "ADMISSIBLE"
    status_light = map_admissibility_to_director_light(snapshot)
    dominant_failure_phase = compass.get("dominant_failure_phase")
    compass_status = compass.get("compass_status", "CRITICAL")
    promotion_status = promotion_eval.get("status", "BLOCK")

    # Generate headline based on combined state
    if is_admissible and compass_status == "STABLE":
        headline = "Evidence pipeline operating normally"
    elif is_admissible and compass_status == "DEGRADING":
        headline = "Current evidence admissible, pipeline showing degradation"
    elif compass_status == "CRITICAL":
        headline = "Evidence pipeline requires immediate attention"
    elif promotion_status == "BLOCK":
        headline = "Evidence pipeline blocked for promotion"
    elif not is_admissible:
        headline = "Evidence currently not admissible"
    else:
        headline = "Evidence admissibility check in progress"

    return {
        "admissible": is_admissible,
        "status_light": status_light,
        "dominant_failure_phase": dominant_failure_phase,
        "headline": headline,
    }


def attach_admissibility_to_evidence_chain(
    evidence_chain: Dict[str, Any],
    compass: Dict[str, Any],
    promotion_eval: Dict[str, Any],
) -> Dict[str, Any]:
    """Attach admissibility compass data to an evidence chain.

    Adds an admissibility_compass subtree to the evidence chain containing:
    - status: The compass status (STABLE|DEGRADING|CRITICAL)
    - recurrent_errors: List of recurrent error codes
    - failure_concentration: How concentrated failures are in dominant phase
    - promotion_blocked: Whether promotion is blocked
    - blocking_error_codes: Error codes blocking promotion

    Returns a new dict with the admissibility data attached (does not mutate input).
    """
    compass_status = compass.get("compass_status", "CRITICAL")
    recurrent_errors = compass.get("recurrent_errors", [])
    failure_concentration = compass.get("failure_concentration", 0.0)
    promotion_blocked = not promotion_eval.get("promotion_ok", False)
    blocking_error_codes = promotion_eval.get("blocking_error_codes", [])

    # Create admissibility subtree
    admissibility_compass = {
        "status": compass_status,
        "recurrent_errors": recurrent_errors,
        "failure_concentration": failure_concentration,
        "promotion_blocked": promotion_blocked,
        "blocking_error_codes": blocking_error_codes,
    }

    # Create new chain with admissibility attached (immutable)
    result = dict(evidence_chain)
    result["admissibility_compass"] = admissibility_compass

    return result


def to_governance_signal(
    analytics: Dict[str, Any],
    compass: Dict[str, Any],
    promotion_eval: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate a governance signal for CLAUDE I integration.

    PRIORITY: ADMISSIBILITY_GATE is the FINAL sanity check before any evidence
    is promoted or published. It takes precedence over other promotion layers:

        1. Evidence Collection    → artifacts gathered
        2. Statistical Analysis   → metrics computed
        3. Governance Verifier    → G1-G4 gates checked
        4. ADMISSIBILITY_GATE     → THIS LAYER (final veto power)

    If ADMISSIBILITY_GATE returns HALT or BLOCK, promotion MUST NOT proceed
    regardless of what other layers indicate. This ensures evidence integrity
    is never compromised by upstream false positives.

    Signal Structure (for CLAUDE I consumption):
        - signal_type: Always "ADMISSIBILITY_GATE" for routing
        - recommendation: PROCEED | PROCEED_WITH_CAUTION | BLOCK | HALT
        - confidence: HIGH | MEDIUM (how certain is the recommendation)
        - promotion_status: OK | WARN | BLOCK (from promotion evaluator)
        - compass_status: STABLE | DEGRADING | CRITICAL
        - blocking_error_codes: List of HE-GV*/HE-I* codes blocking promotion
        - recurrent_errors: All recurrent errors (>= 30% frequency)
        - critical_issues: Human-readable list of issues for governance review

    Integration:
        CLAUDE I should check signal_type == "ADMISSIBILITY_GATE" and respect
        the recommendation field. HALT means stop all promotion activity.
        BLOCK means this specific evidence cannot be promoted.
    """
    compass_status = compass.get("compass_status", "CRITICAL")
    promotion_ok = promotion_eval.get("promotion_ok", False)
    promotion_status = promotion_eval.get("status", "BLOCK")
    blocking_error_codes = promotion_eval.get("blocking_error_codes", [])
    recurrent_errors = compass.get("recurrent_errors", [])
    long_run_rate = analytics.get("long_run_admissibility_rate", 0.0)
    dominant_failure_phase = compass.get("dominant_failure_phase")

    # Determine governance recommendation
    if promotion_ok and compass_status == "STABLE":
        recommendation = "PROCEED"
        confidence = "HIGH"
    elif promotion_ok and compass_status == "DEGRADING":
        recommendation = "PROCEED_WITH_CAUTION"
        confidence = "MEDIUM"
    elif compass_status == "CRITICAL":
        recommendation = "HALT"
        confidence = "HIGH"
    else:
        recommendation = "BLOCK"
        confidence = "HIGH"

    # Identify critical issues requiring governance attention
    critical_issues = []
    if compass_status == "CRITICAL":
        critical_issues.append("Admissibility rate below critical threshold")
    if blocking_error_codes:
        critical_issues.append(f"Blocking errors: {', '.join(blocking_error_codes)}")
    if dominant_failure_phase:
        critical_issues.append(f"Dominant failure phase: {dominant_failure_phase}")

    return {
        "signal_type": "ADMISSIBILITY_GATE",
        "recommendation": recommendation,
        "confidence": confidence,
        "promotion_status": promotion_status,
        "compass_status": compass_status,
        "long_run_admissibility_rate": long_run_rate,
        "blocking_error_codes": blocking_error_codes,
        "recurrent_errors": recurrent_errors,
        "critical_issues": critical_issues,
    }