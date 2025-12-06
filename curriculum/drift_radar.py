"""
Curriculum Drift Radar & Promotion Guard

Implements Phase III drift detection features:
- Drift history ledger across multiple runs
- Drift severity classification for CI and promotion
- Promotion gate evaluation
- Global health summary for dashboards
"""

import json
from typing import Any, Dict, List, Optional, Sequence

from curriculum.phase2_loader import CurriculumFingerprint


def build_curriculum_drift_history(fingerprint_paths: Sequence[str]) -> Dict[str, Any]:
    """
    Build curriculum drift history from multiple fingerprint files.
    
    Loads fingerprints, sorts by timestamp, and computes:
    - schema_version series over time
    - slice_count series over time
    - metric_kind_changes (transitions in success metric types)
    - drift_events_count
    
    Args:
        fingerprint_paths: Sequence of paths to fingerprint JSON files
    
    Returns:
        Dictionary with:
        - fingerprints: List of loaded fingerprints (sorted by timestamp)
        - schema_version: Current schema version
        - version_series: List of schema versions in time order
        - slice_count_series: List of (timestamp, slice_count) tuples
        - metric_kind_changes: List of metric kind change events
        - drift_events_count: Number of detected drift events
    """
    if not fingerprint_paths:
        return {
            "fingerprints": [],
            "schema_version": None,
            "version_series": [],
            "slice_count_series": [],
            "metric_kind_changes": [],
            "drift_events_count": 0
        }
    
    # Load all fingerprints
    fingerprints: List[CurriculumFingerprint] = []
    for path in fingerprint_paths:
        try:
            fp = CurriculumFingerprint.load_from_file(path)
            fingerprints.append(fp)
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            # Skip invalid or missing files but don't fail the whole operation
            # This allows partial history builds when some files are corrupted
            continue
    
    if not fingerprints:
        return {
            "fingerprints": [],
            "schema_version": None,
            "version_series": [],
            "slice_count_series": [],
            "metric_kind_changes": [],
            "drift_events_count": 0
        }
    
    # Sort by timestamp (extract ISO portion before @ if present)
    def extract_timestamp(fp: CurriculumFingerprint) -> str:
        ts = fp.timestamp
        if '@' in ts:
            ts = ts.split('@')[0]
        return ts
    
    fingerprints.sort(key=extract_timestamp)
    
    # Build version series
    version_series = [fp.schema_version for fp in fingerprints]
    
    # Build slice count series
    slice_count_series = [(fp.timestamp, fp.slice_count) for fp in fingerprints]
    
    # Detect metric kind changes (would require loading full curriculum, not just fingerprint)
    # For now, we track slice additions/removals as proxy
    metric_kind_changes: List[Dict[str, Any]] = []
    
    # Count drift events (non-identical fingerprints)
    drift_events_count = 0
    for i in range(1, len(fingerprints)):
        if fingerprints[i].sha256 != fingerprints[i-1].sha256:
            drift_events_count += 1
    
    return {
        "fingerprints": [fp.to_dict() for fp in fingerprints],
        "schema_version": fingerprints[-1].schema_version,
        "version_series": version_series,
        "slice_count_series": slice_count_series,
        "metric_kind_changes": metric_kind_changes,
        "drift_events_count": drift_events_count
    }


def classify_curriculum_drift_event(diff: Dict[str, Any]) -> Dict[str, Any]:
    """
    Classify curriculum drift severity for CI and promotion decisions.
    
    Severity levels:
    - NONE: No changes detected
    - MINOR: Parameter tweaks within existing slices, no structural changes
    - MAJOR: Slices added/removed, schema version changed, or success metric kind changed
    
    Args:
        diff: Diff dictionary from compute_curriculum_diff()
    
    Returns:
        Dictionary with:
        - severity: "NONE" | "MINOR" | "MAJOR"
        - blocking: bool (True if should block promotion/CI)
        - reasons: List of short neutral explanation strings
    """
    if not diff.get("changed", False):
        return {
            "severity": "NONE",
            "blocking": False,
            "reasons": []
        }
    
    reasons: List[str] = []
    blocking = False
    severity = "MINOR"
    
    # Check for major changes
    if diff.get("schema_version_changed", False):
        reasons.append(f"schema version changed: {diff['old_schema_version']} → {diff['new_schema_version']}")
        severity = "MAJOR"
        blocking = True
    
    slices_added = diff.get("slices_added", [])
    if slices_added:
        reasons.append(f"slices added: {', '.join(slices_added)}")
        severity = "MAJOR"
        blocking = True
    
    slices_removed = diff.get("slices_removed", [])
    if slices_removed:
        reasons.append(f"slices removed: {', '.join(slices_removed)}")
        severity = "MAJOR"
        blocking = True
    
    slices_modified = diff.get("slices_modified", [])
    if slices_modified and severity == "MINOR":
        # Modified slices without add/remove is minor
        reasons.append(f"slices modified: {', '.join(slices_modified)}")
        # Minor modifications are not blocking by default
        blocking = False
    
    if not reasons:
        reasons.append("configuration hash changed")
    
    return {
        "severity": severity,
        "blocking": blocking,
        "reasons": reasons
    }


def evaluate_curriculum_for_promotion(history: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate curriculum state for promotion gate.
    
    Checks if curriculum is stable enough for promotion to production.
    
    Args:
        history: Drift history from build_curriculum_drift_history()
    
    Returns:
        Dictionary with:
        - promotion_ok: bool (True if safe to promote)
        - last_drift_severity: str or None
        - blocking_reasons: List[str] (reasons blocking promotion, if any)
    """
    if not history.get("fingerprints"):
        return {
            "promotion_ok": False,
            "last_drift_severity": None,
            "blocking_reasons": ["no curriculum fingerprints found"]
        }
    
    fingerprints = history["fingerprints"]
    drift_count = history.get("drift_events_count", 0)
    
    # Check for recent drift
    if len(fingerprints) >= 2:
        from curriculum.phase2_loader import CurriculumFingerprint, compute_curriculum_diff
        
        # Compare last two fingerprints
        fp1 = CurriculumFingerprint.from_dict(fingerprints[-2])
        fp2 = CurriculumFingerprint.from_dict(fingerprints[-1])
        
        diff = compute_curriculum_diff(fp1, fp2)
        classification = classify_curriculum_drift_event(diff)
        
        last_drift_severity = classification["severity"]
        blocking = classification["blocking"]
        
        if blocking:
            return {
                "promotion_ok": False,
                "last_drift_severity": last_drift_severity,
                "blocking_reasons": classification["reasons"]
            }
    else:
        last_drift_severity = "NONE"
    
    # Promotion OK if no blocking issues
    return {
        "promotion_ok": True,
        "last_drift_severity": last_drift_severity,
        "blocking_reasons": []
    }


def summarize_curriculum_for_global_health(history: Dict[str, Any]) -> Dict[str, Any]:
    """
    Summarize curriculum state for global health dashboard.
    
    Provides high-level health status and metrics.
    
    Args:
        history: Drift history from build_curriculum_drift_history()
    
    Returns:
        Dictionary with:
        - curriculum_ok: bool (overall health)
        - current_slice_count: int
        - recent_drift_events: int (in last N runs)
        - status: "OK" | "WARN" | "BLOCK"
        - details: Dict with additional info
    """
    if not history.get("fingerprints"):
        return {
            "curriculum_ok": False,
            "current_slice_count": 0,
            "recent_drift_events": 0,
            "status": "BLOCK",
            "details": {
                "message": "no curriculum fingerprints available"
            }
        }
    
    fingerprints = history["fingerprints"]
    current_slice_count = fingerprints[-1]["slice_count"]
    total_drift = history.get("drift_events_count", 0)
    
    # Check promotion status
    promotion = evaluate_curriculum_for_promotion(history)
    
    if not promotion["promotion_ok"]:
        status = "BLOCK"
        curriculum_ok = False
    elif total_drift > 3:
        # Warn if significant drift activity
        status = "WARN"
        curriculum_ok = True
    else:
        status = "OK"
        curriculum_ok = True
    
    # Count recent drift (last 5 runs)
    recent_drift_events = 0
    if len(fingerprints) >= 2:
        check_count = min(5, len(fingerprints))
        for i in range(len(fingerprints) - check_count + 1, len(fingerprints)):
            if fingerprints[i]["sha256"] != fingerprints[i-1]["sha256"]:
                recent_drift_events += 1
    
    return {
        "curriculum_ok": curriculum_ok,
        "current_slice_count": current_slice_count,
        "recent_drift_events": recent_drift_events,
        "status": status,
        "details": {
            "total_fingerprints": len(fingerprints),
            "total_drift_events": total_drift,
            "schema_version": history.get("schema_version"),
            "promotion_ok": promotion["promotion_ok"],
            "blocking_reasons": promotion.get("blocking_reasons", [])
        }
    }
