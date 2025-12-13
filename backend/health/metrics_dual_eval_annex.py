"""
Metrics Dual Evaluation Annex.

Surfaces HYBRID mode evaluation results as advisory artifacts for
comparing MOCK vs REAL threshold bands without changing decisions.

SHADOW MODE: Observation-only. No control paths.
This module provides advisory comparison data but NEVER changes governance status.

REAL-READY: Designed for P5 migration comparison phase.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.health.metrics_thresholds import (
    MODE_HYBRID,
    MODE_MOCK,
    MODE_REAL,
    check_in_band,
    evaluate_with_dual_thresholds,
    get_all_safe_bands,
    get_threshold,
    get_threshold_mode,
    get_threshold_pair,
    list_threshold_names,
)

DUAL_EVAL_ANNEX_SCHEMA_VERSION = "1.2.0"  # Bumped for reason codes + extraction_source

# ==============================================================================
# Reason Codes (GGFL driver codes)
# ==============================================================================

DRIVER_BANDS_DIFFER_PRESENT = "DRIVER_BANDS_DIFFER_PRESENT"
DRIVER_TOP_METRICS_PRESENT = "DRIVER_TOP_METRICS_PRESENT"

# Extraction source constants
EXTRACTION_SOURCE_MANIFEST = "MANIFEST"
EXTRACTION_SOURCE_EVIDENCE_JSON = "EVIDENCE_JSON"
EXTRACTION_SOURCE_MISSING = "MISSING"


# ==============================================================================
# Canonical Artifact Detection
# ==============================================================================


def _compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def discover_metrics_json(run_dir: Path) -> Tuple[Optional[Path], Optional[str]]:
    """
    Discover metrics_windows.json from canonical locations within run_dir.

    SHADOW MODE: No env reliance. No enforcement.
    Searches deterministically in priority order.

    Args:
        run_dir: Base run directory to search

    Returns:
        Tuple of (path, selection_reason) or (None, None) if not found.
        - path: Absolute path to discovered metrics JSON
        - selection_reason: String describing which candidate was selected

    Candidate paths (priority order):
        1. <run_dir>/metrics_windows.json
        2. <run_dir>/p3/metrics_windows.json
        3. <run_dir>/fl_*/metrics_windows.json (most recent fl_ subdir)
    """
    if not run_dir.exists():
        return None, None

    # Candidate 1: Direct path
    direct_path = run_dir / "metrics_windows.json"
    if direct_path.exists():
        return direct_path, "direct:<run_dir>/metrics_windows.json"

    # Candidate 2: p3 subdirectory
    p3_path = run_dir / "p3" / "metrics_windows.json"
    if p3_path.exists():
        return p3_path, "p3_subdir:<run_dir>/p3/metrics_windows.json"

    # Candidate 3: fl_* subdirectory (deterministic: most recent by mtime)
    fl_dirs = sorted(
        run_dir.glob("fl_*"),
        key=lambda p: p.stat().st_mtime if p.is_dir() else 0,
        reverse=True,
    )
    for fl_dir in fl_dirs:
        fl_metrics = fl_dir / "metrics_windows.json"
        if fl_metrics.exists():
            return fl_metrics, f"fl_subdir:{fl_dir.name}/metrics_windows.json"

    # Candidate 4: p4_* subdirectory (for P4 runs)
    p4_dirs = sorted(
        run_dir.glob("p4_*"),
        key=lambda p: p.stat().st_mtime if p.is_dir() else 0,
        reverse=True,
    )
    for p4_dir in p4_dirs:
        p4_metrics = p4_dir / "metrics_windows.json"
        if p4_metrics.exists():
            return p4_metrics, f"p4_subdir:{p4_dir.name}/metrics_windows.json"

    return None, None


def load_metrics_from_run_dir(run_dir: Path) -> Tuple[Optional[Dict[str, float]], Optional[str], Optional[str]]:
    """
    Load metrics payload from canonical location in run_dir.

    SHADOW MODE: No env reliance. No enforcement.

    Args:
        run_dir: Base run directory to search

    Returns:
        Tuple of (metrics_payload, selection_path, sha256) or (None, None, None) if not found.
    """
    path, selection_reason = discover_metrics_json(run_dir)
    if path is None:
        return None, None, None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Extract metrics from windows format (use last window or aggregate)
        metrics_payload = _extract_metrics_from_windows(data)
        sha256 = _compute_sha256(path)

        return metrics_payload, selection_reason, sha256
    except (json.JSONDecodeError, IOError, KeyError):
        return None, None, None


def _extract_metrics_from_windows(data: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract metrics payload from metrics_windows.json format.

    Handles both single-window and multi-window formats.
    Uses last window for time-series data (most recent state).
    """
    # If data has "windows" array, use last window
    windows = data.get("windows", [])
    if windows:
        last_window = windows[-1]
        return _extract_single_window_metrics(last_window)

    # If data is direct metrics (flat structure)
    return _extract_single_window_metrics(data)


def _extract_single_window_metrics(window: Dict[str, Any]) -> Dict[str, float]:
    """Extract canonical metrics from a single window/report."""
    metrics: Dict[str, float] = {}

    # Direct field mapping
    field_mappings = {
        "drift_magnitude": ["drift_magnitude", "drift", "drift_mag"],
        "success_rate": ["success_rate", "sr", "success"],
        "budget_utilization": ["budget_utilization", "budget_util", "budget"],
        "abstention_rate": ["abstention_rate", "abstention", "abs_rate"],
        "block_rate": ["block_rate", "br", "blocked_rate"],
    }

    for canonical_name, candidates in field_mappings.items():
        for candidate in candidates:
            if candidate in window:
                value = window[candidate]
                if isinstance(value, (int, float)):
                    metrics[canonical_name] = float(value)
                    break

    # Also check nested "metrics" key
    nested_metrics = window.get("metrics", {})
    if nested_metrics:
        for canonical_name, candidates in field_mappings.items():
            if canonical_name in metrics:
                continue  # Already found
            for candidate in candidates:
                if candidate in nested_metrics:
                    value = nested_metrics[candidate]
                    if isinstance(value, (int, float)):
                        metrics[canonical_name] = float(value)
                        break

    return metrics


# ==============================================================================
# Dual Evaluation Annex Builder
# ==============================================================================


def build_metrics_dual_eval_annex(
    metrics_payload: Dict[str, float],
    *,
    mode: str = MODE_HYBRID,
) -> Dict[str, Any]:
    """
    Build dual evaluation annex for MOCK vs REAL threshold comparison.

    SHADOW MODE: Advisory-only. Does NOT change governance status.
    This annex is purely for observation and comparison logging.

    Args:
        metrics_payload: Dict with metric values:
            - drift_magnitude: float
            - success_rate: float (percentage)
            - budget_utilization: float (percentage)
            - abstention_rate: float (percentage)
            - block_rate: float (ratio)
        mode: Evaluation mode (default HYBRID for dual comparison)

    Returns:
        Schema-versioned annex with:
        - schema_version: "1.0.0"
        - mode: evaluation mode used
        - advisory_only: True (always)
        - per_metric: Dict[metric_name, {mock_band, real_band, delta, in_safe_band}]
        - summary: {total_metrics, in_band_count, out_of_band_count, out_of_band_metrics}
        - dual_verdict: {mock_status, real_status, diverges} (if HYBRID)
        - timestamp_note: str

    Example:
        >>> annex = build_metrics_dual_eval_annex({"drift_magnitude": 0.32})
        >>> annex["advisory_only"]
        True
        >>> annex["per_metric"]["drift_magnitude"]["mock_band"]
        "YELLOW"
    """
    # Always advisory - never changes decisions
    annex: Dict[str, Any] = {
        "schema_version": DUAL_EVAL_ANNEX_SCHEMA_VERSION,
        "mode": mode,
        "advisory_only": True,
        "timestamp_note": "Annex generated for threshold comparison only.",
    }

    # Build per-metric analysis
    per_metric = _build_per_metric_analysis(metrics_payload)
    annex["per_metric"] = per_metric

    # Build summary counts
    annex["summary"] = _build_summary(per_metric)

    # If HYBRID mode, include dual verdict comparison
    if mode == MODE_HYBRID:
        # Compute both MOCK and REAL verdicts directly
        # (don't rely on env var - use explicit mode parameter)
        from backend.health.metrics_thresholds import _evaluate_single
        mock_verdict = _evaluate_single(metrics_payload, MODE_MOCK)
        real_verdict = _evaluate_single(metrics_payload, MODE_REAL)
        diverges = mock_verdict["status"] != real_verdict["status"]

        annex["dual_verdict"] = {
            "mock_status": mock_verdict["status"],
            "real_status": real_verdict["status"],
            "diverges": diverges,
            "divergent_thresholds": _find_divergent_thresholds_inline(metrics_payload)
            if diverges
            else [],
        }
    else:
        # Single mode - no dual verdict
        annex["dual_verdict"] = None

    return annex


def _find_divergent_thresholds_inline(metrics_payload: Dict[str, float]) -> List[str]:
    """
    Find thresholds causing MOCK/REAL divergence.

    Inline version that doesn't rely on environment.
    """
    from backend.health.metrics_thresholds import get_threshold

    divergent: List[str] = []

    # Drift
    drift_mag = metrics_payload.get("drift_magnitude", 0.0)
    mock_drift_warn = get_threshold("drift_warn", MODE_MOCK)
    real_drift_warn = get_threshold("drift_warn", MODE_REAL)
    if mock_drift_warn <= drift_mag < real_drift_warn:
        divergent.append("drift_warn")

    # Success rate (inverted)
    success_rate = metrics_payload.get("success_rate", 100.0)
    mock_sr_warn = get_threshold("success_rate_warn", MODE_MOCK)
    real_sr_warn = get_threshold("success_rate_warn", MODE_REAL)
    if real_sr_warn <= success_rate < mock_sr_warn:
        divergent.append("success_rate_warn")

    # Budget utilization
    budget_util = metrics_payload.get("budget_utilization", 0.0)
    mock_budget_warn = get_threshold("budget_warn", MODE_MOCK)
    real_budget_warn = get_threshold("budget_warn", MODE_REAL)
    if mock_budget_warn <= budget_util < real_budget_warn:
        divergent.append("budget_warn")

    # Abstention rate
    abstention = metrics_payload.get("abstention_rate", 0.0)
    mock_abs_warn = get_threshold("abstention_warn", MODE_MOCK)
    real_abs_warn = get_threshold("abstention_warn", MODE_REAL)
    if mock_abs_warn <= abstention < real_abs_warn:
        divergent.append("abstention_warn")

    # Block rate
    block_rate = metrics_payload.get("block_rate", 0.0)
    mock_br_warn = get_threshold("block_rate_warn", MODE_MOCK)
    real_br_warn = get_threshold("block_rate_warn", MODE_REAL)
    if mock_br_warn <= block_rate < real_br_warn:
        divergent.append("block_rate_warn")

    return sorted(divergent)


def _build_per_metric_analysis(
    metrics_payload: Dict[str, float],
) -> Dict[str, Dict[str, Any]]:
    """
    Build per-metric threshold band analysis.

    For each metric, determines:
    - mock_band: status under MOCK thresholds (GREEN/YELLOW/RED)
    - real_band: status under REAL thresholds (GREEN/YELLOW/RED)
    - delta: difference in treatment
    - in_safe_band: whether P3/P5 delta is within safe comparison band
    """
    result: Dict[str, Dict[str, Any]] = {}

    # Map metrics payload keys to threshold names
    metric_mappings = {
        "drift_magnitude": ("drift_warn", "drift_critical"),
        "success_rate": ("success_rate_warn", "success_rate_critical"),
        "budget_utilization": ("budget_warn", "budget_critical"),
        "abstention_rate": ("abstention_warn", "abstention_critical"),
        "block_rate": ("block_rate_warn", "block_rate_critical"),
    }

    for metric_name, (warn_key, critical_key) in metric_mappings.items():
        value = metrics_payload.get(metric_name)
        if value is None:
            continue

        # Get thresholds for both modes
        mock_warn = get_threshold(warn_key, MODE_MOCK)
        mock_critical = get_threshold(critical_key, MODE_MOCK)
        real_warn = get_threshold(warn_key, MODE_REAL)
        real_critical = get_threshold(critical_key, MODE_REAL)

        # Determine band for each mode
        mock_band = _classify_band(metric_name, value, mock_warn, mock_critical)
        real_band = _classify_band(metric_name, value, real_warn, real_critical)

        # Check if in safe comparison band (using value as both P3 and P5)
        # For actual comparison, we'd need both P3 and P5 values
        safe_bands = get_all_safe_bands()
        safe_band_key = _get_safe_band_key(metric_name)
        in_safe_band = True  # Default when comparing same value
        safe_band_width = safe_bands.get(safe_band_key, 0.0)

        result[metric_name] = {
            "value": value,
            "mock_band": mock_band,
            "real_band": real_band,
            "bands_match": mock_band == real_band,
            "mock_thresholds": {"warn": mock_warn, "critical": mock_critical},
            "real_thresholds": {"warn": real_warn, "critical": real_critical},
            "safe_band_width": safe_band_width,
            "in_safe_band": in_safe_band,
        }

    return result


def _classify_band(
    metric_name: str,
    value: float,
    warn_threshold: float,
    critical_threshold: float,
) -> str:
    """
    Classify metric value into band (GREEN/YELLOW/RED).

    Handles inverted metrics (success_rate - lower is worse).
    """
    # Success rate is inverted (lower is worse)
    if metric_name == "success_rate":
        if value < critical_threshold:
            return "RED"
        if value < warn_threshold:
            return "YELLOW"
        return "GREEN"

    # All other metrics: higher is worse
    if value >= critical_threshold:
        return "RED"
    if value >= warn_threshold:
        return "YELLOW"
    return "GREEN"


def _get_safe_band_key(metric_name: str) -> str:
    """Map metric payload key to safe band key."""
    mapping = {
        "drift_magnitude": "drift_magnitude",
        "success_rate": "success_rate",
        "budget_utilization": "budget_utilization",
        "abstention_rate": "abstention_rate",
        "block_rate": "block_rate",
    }
    return mapping.get(metric_name, metric_name)


def _build_summary(per_metric: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Build summary counts from per-metric analysis."""
    total = len(per_metric)
    bands_match = sum(1 for m in per_metric.values() if m.get("bands_match", False))
    bands_differ = total - bands_match

    differing_metrics = sorted(
        [name for name, data in per_metric.items() if not data.get("bands_match", True)]
    )

    return {
        "total_metrics": total,
        "bands_match_count": bands_match,
        "bands_differ_count": bands_differ,
        "differing_metrics": differing_metrics,
        "all_bands_match": bands_differ == 0,
    }


# ==============================================================================
# Evidence Pack Hook
# ==============================================================================


def attach_dual_eval_to_evidence(
    evidence: Dict[str, Any],
    metrics_payload: Optional[Dict[str, float]] = None,
    *,
    mode: str = MODE_HYBRID,
    source_path: Optional[str] = None,
    source_sha256: Optional[str] = None,
    annex_artifact_path: Optional[str] = None,
    annex_artifact_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Attach dual evaluation annex to evidence pack.

    SHADOW MODE: Advisory-only. Does NOT change governance status.
    Attaches under evidence["governance"]["metrics"]["dual_eval"].
    Also adds manifest reference under evidence["governance"]["metrics"]["dual_eval_reference"].
    Wires artifact reference via build_dual_eval_artifact_reference().

    Args:
        evidence: Existing evidence pack dict (read-only, not modified)
        metrics_payload: Metrics to evaluate (if None, skips attachment)
        mode: Evaluation mode (default HYBRID)
        source_path: Optional path describing where metrics were loaded from
        source_sha256: Optional SHA256 hash of source file for manifest reference
        annex_artifact_path: Optional path to annex artifact file (for manifest)
        annex_artifact_sha256: Optional SHA256 of annex artifact file (for manifest)

    Returns:
        New dict with evidence contents plus dual_eval annex and reference attached.

    Example:
        >>> evidence = {"timestamp": "2025-01-01", "governance": {"metrics": {}}}
        >>> enriched = attach_dual_eval_to_evidence(
        ...     evidence, {"drift_magnitude": 0.32},
        ...     source_path="p3_subdir:metrics_windows.json",
        ...     source_sha256="abc123...",
        ...     annex_artifact_sha256="def456..."
        ... )
        >>> "dual_eval" in enriched["governance"]["metrics"]
        True
        >>> "dual_eval_reference" in enriched["governance"]["metrics"]
        True
    """
    # Create a copy to avoid mutating the original
    enriched: Dict[str, Any] = {}
    for key, value in evidence.items():
        enriched[key] = value

    # Skip if no metrics payload
    if metrics_payload is None or len(metrics_payload) == 0:
        return enriched

    # Build the annex
    annex = build_metrics_dual_eval_annex(metrics_payload, mode=mode)

    # Ensure governance structure exists
    if "governance" not in enriched:
        enriched["governance"] = {}
    else:
        enriched["governance"] = dict(enriched["governance"])

    if "metrics" not in enriched["governance"]:
        enriched["governance"]["metrics"] = {}
    else:
        enriched["governance"]["metrics"] = dict(enriched["governance"]["metrics"])

    # Attach the annex
    enriched["governance"]["metrics"]["dual_eval"] = annex

    # Build and attach manifest reference using artifact reference builder
    # Use annex artifact sha256 if provided, else fall back to source sha256
    ref_sha256 = annex_artifact_sha256 or source_sha256
    if ref_sha256:
        artifact_ref = build_dual_eval_artifact_reference(
            annex_path=Path(annex_artifact_path) if annex_artifact_path else None,
            annex_sha256=ref_sha256,
        )
        if artifact_ref:
            # Add source provenance to artifact reference
            artifact_ref["source_path"] = source_path
            enriched["governance"]["metrics"]["dual_eval_reference"] = artifact_ref

    return enriched


# ==============================================================================
# Status Summary for CLI
# ==============================================================================


def build_dual_eval_status_summary(
    metrics_payload: Dict[str, float],
    *,
    mode: str = MODE_HYBRID,
    source_path: Optional[str] = None,
    source_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build compact status summary for CLI output.

    SHADOW MODE: Advisory-only. Does NOT change governance status.
    Designed for --metrics-json CLI flag output.

    Args:
        metrics_payload: Metrics to evaluate
        mode: Evaluation mode (default HYBRID)
        source_path: Optional selection path describing where metrics were found
        source_sha256: Optional SHA256 hash of the source file (required if source_path provided)

    Returns:
        Standardized status signal with:
        - schema_version: str (passthrough from annex)
        - mode: "SHADOW" (constant marker)
        - advisory_only: True
        - diverges: bool (HYBRID verdict divergence)
        - bands_differ_count: int
        - differing_metrics: List[str] (sorted)
        - mock_status: str
        - real_status: str
        - source_path: Optional[str] (only if source_sha256 also provided)
        - source_sha256: Optional[str]

    Example:
        >>> summary = build_dual_eval_status_summary({"drift_magnitude": 0.32})
        >>> summary["advisory_only"]
        True
        >>> summary["mode"]
        "SHADOW"
        >>> summary["schema_version"]
        "1.1.0"
    """
    annex = build_metrics_dual_eval_annex(metrics_payload, mode=mode)

    # Standardized signal shape with schema_version passthrough and SHADOW marker
    summary: Dict[str, Any] = {
        "schema_version": DUAL_EVAL_ANNEX_SCHEMA_VERSION,
        "mode": "SHADOW",  # Constant marker per spec
        "advisory_only": True,
        # Standardized fields (per spec)
        "diverges": False,
        "bands_differ_count": annex["summary"]["bands_differ_count"],
        "differing_metrics": sorted(annex["summary"]["differing_metrics"]),  # Always sorted
        "mock_status": "UNKNOWN",
        "real_status": "UNKNOWN",
    }

    if annex["dual_verdict"] is not None:
        summary["diverges"] = annex["dual_verdict"]["diverges"]
        summary["mock_status"] = annex["dual_verdict"]["mock_status"]
        summary["real_status"] = annex["dual_verdict"]["real_status"]

    # Source provenance: only include source_path if source_sha256 is also present
    # This ensures provenance is always verifiable
    if source_sha256:
        summary["source_sha256"] = source_sha256
        if source_path:
            summary["source_path"] = source_path

    return summary


# ==============================================================================
# Artifact Reference for Manifest
# ==============================================================================


def build_dual_eval_artifact_reference(
    annex_path: Optional[Path] = None,
    annex_sha256: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Build artifact reference for dual-eval annex for manifest inclusion.

    SHADOW MODE: Advisory-only. Does NOT change governance status.

    Args:
        annex_path: Path to the dual-eval annex artifact file
        annex_sha256: SHA256 hash of the annex artifact file

    Returns:
        Artifact reference dict or None if no valid reference.
        Reference includes:
        - artifact_type: "dual_eval_annex"
        - schema_version: str
        - path: str (if provided)
        - sha256: str (if provided)

    Example:
        >>> ref = build_dual_eval_artifact_reference(
        ...     annex_path=Path("p3/dual_eval_annex.json"),
        ...     annex_sha256="abc123..."
        ... )
        >>> ref["artifact_type"]
        "dual_eval_annex"
    """
    # Require at least sha256 for a valid reference
    if not annex_sha256:
        return None

    ref: Dict[str, Any] = {
        "artifact_type": "dual_eval_annex",
        "schema_version": DUAL_EVAL_ANNEX_SCHEMA_VERSION,
        "sha256": annex_sha256,
    }

    if annex_path:
        ref["path"] = str(annex_path)

    return ref


# ==============================================================================
# GGFL Alignment View Adapter
# ==============================================================================


def metrics_dual_eval_for_alignment_view(
    metrics_payload: Optional[Dict[str, float]] = None,
    *,
    source_path: Optional[str] = None,
    source_sha256: Optional[str] = None,
    extraction_source: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Build GGFL-compliant alignment view signal for metrics dual-eval.

    SHADOW MODE: Advisory-only. Does NOT change governance status.
    Returns None if metrics_payload is None or empty (explicit optional).

    Args:
        metrics_payload: Metrics to evaluate (if None, returns None)
        source_path: Optional path for artifact reference
        source_sha256: Optional SHA256 for artifact reference
        extraction_source: MANIFEST | EVIDENCE_JSON | MISSING (provenance)

    Returns:
        GGFL-compliant signal with fixed shape, or None if no metrics:
        - signal_type: "SIG-MET"
        - status: "ok" | "warn" (warn if diverges true)
        - conflict: false (always)
        - drivers: List[str] max 3 items (reason codes)
        - driver_codes: List[str] (DRIVER_BANDS_DIFFER_PRESENT, DRIVER_TOP_METRICS_PRESENT)
        - summary: str (1 sentence)
        - artifact_ref: Optional[Dict] (path + sha256 if provided)
        - extraction_source: str (MANIFEST | EVIDENCE_JSON | MISSING)

    Example:
        >>> signal = metrics_dual_eval_for_alignment_view({"drift_magnitude": 0.32})
        >>> signal["signal_type"]
        "SIG-MET"
        >>> signal["status"]
        "warn"
        >>> "DRIVER_BANDS_DIFFER_PRESENT" in signal["driver_codes"]
        True
    """
    # Explicit optional: return None for missing/empty metrics
    if metrics_payload is None or len(metrics_payload) == 0:
        return None

    # Build underlying summary
    summary_data = build_dual_eval_status_summary(
        metrics_payload,
        mode=MODE_HYBRID,
        source_path=source_path,
        source_sha256=source_sha256,
    )

    # Determine status: warn if diverges, ok otherwise
    diverges = summary_data.get("diverges", False)
    status = "warn" if diverges else "ok"

    # Build drivers with reason codes
    bands_count = summary_data.get("bands_differ_count", 0)
    differing = sorted(summary_data.get("differing_metrics", []))

    drivers: List[str] = []
    driver_codes: List[str] = []

    if bands_count > 0:
        drivers.append(f"{bands_count} band(s) differ")
        driver_codes.append(DRIVER_BANDS_DIFFER_PRESENT)

    # Add top 2 differing metrics (total max 3 drivers)
    if differing:
        drivers.extend(differing[:2])
        driver_codes.append(DRIVER_TOP_METRICS_PRESENT)

    # Build 1-sentence summary
    mock_status = summary_data.get("mock_status", "UNKNOWN")
    real_status = summary_data.get("real_status", "UNKNOWN")
    if diverges:
        top3_str = ", ".join(differing[:3]) if differing else "none"
        summary_sentence = (
            f"MOCK/REAL thresholds diverge ({bands_count} bands); "
            f"top: {top3_str}."
        )
    else:
        summary_sentence = (
            f"MOCK/REAL thresholds aligned (MOCK={mock_status}, REAL={real_status})."
        )

    # Determine extraction_source if not provided
    if extraction_source is None:
        if source_sha256:
            extraction_source = EXTRACTION_SOURCE_MANIFEST
        else:
            extraction_source = EXTRACTION_SOURCE_MISSING

    # Build GGFL signal
    signal: Dict[str, Any] = {
        "signal_type": "SIG-MET",
        "status": status,
        "conflict": False,  # Never conflicts (advisory only)
        "drivers": drivers[:3],  # Cap at 3
        "driver_codes": driver_codes,
        "summary": summary_sentence,
        "extraction_source": extraction_source,
    }

    # Include artifact reference if sha256 provided
    if source_sha256:
        signal["artifact_ref"] = {
            "sha256": source_sha256,
        }
        if source_path:
            signal["artifact_ref"]["path"] = source_path

    return signal


def format_dual_eval_warning_line(
    summary_data: Dict[str, Any],
) -> Optional[str]:
    """
    Format single warning line for dual-eval divergence.

    SHADOW MODE: Advisory-only. Does NOT change governance status.
    Enforces single warning cap: one line only.

    Args:
        summary_data: Output from build_dual_eval_status_summary()

    Returns:
        Single warning line string, or None if no divergence.
        Format: "Metrics dual-eval: N band(s) differ (MOCK=X, REAL=Y); top3: [a, b, c]"
    """
    if not summary_data.get("diverges", False):
        return None

    bands_count = summary_data.get("bands_differ_count", 0)
    differing = sorted(summary_data.get("differing_metrics", []))
    mock_status = summary_data.get("mock_status", "UNKNOWN")
    real_status = summary_data.get("real_status", "UNKNOWN")

    # Top 3 sorted for determinism
    top3 = differing[:3]
    top3_str = ", ".join(top3) if top3 else "none"

    return (
        f"Metrics dual-eval: {bands_count} band(s) differ "
        f"(MOCK={mock_status}, REAL={real_status}); "
        f"top3: [{top3_str}]"
    )


# ==============================================================================
# Exports
# ==============================================================================

__all__ = [
    "DUAL_EVAL_ANNEX_SCHEMA_VERSION",
    "build_metrics_dual_eval_annex",
    "attach_dual_eval_to_evidence",
    "build_dual_eval_status_summary",
    # Canonical detection
    "discover_metrics_json",
    "load_metrics_from_run_dir",
    # Artifact reference
    "build_dual_eval_artifact_reference",
    # GGFL alignment view adapter
    "metrics_dual_eval_for_alignment_view",
    "format_dual_eval_warning_line",
    # Reason codes
    "DRIVER_BANDS_DIFFER_PRESENT",
    "DRIVER_TOP_METRICS_PRESENT",
    # Extraction source constants
    "EXTRACTION_SOURCE_MANIFEST",
    "EXTRACTION_SOURCE_EVIDENCE_JSON",
    "EXTRACTION_SOURCE_MISSING",
]
