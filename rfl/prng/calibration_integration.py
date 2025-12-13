"""
PRNG Drift → Calibration Trajectory Lens

Integrates PRNG drift analysis with CAL-EXP-1 calibration windows to identify
correlations between PRNG volatility and calibration behavior.

SHADOW MODE CONTRACT:
- All outputs are observational only
- No inference beyond correlation
- No enforcement language
- Purely diagnostic for calibration analysis
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from rfl.prng.governance import DriftStatus, build_prng_governance_tile, build_prng_drift_radar


def align_prng_drift_to_windows(
    windows: List[Dict[str, Any]],
    prng_tiles: Optional[List[Dict[str, Any]]] = None,
    prng_history: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Align PRNG drift deltas to CAL-EXP-1 window indices.

    For each window, computes PRNG drift status and volatility metrics.
    If PRNG data is not provided, returns windows with empty PRNG fields.

    Args:
        windows: List of CAL-EXP-1 window dicts, each with:
            - window_index: int
            - window_start: int
            - window_end: int
            - mean_delta_p: float
            - (other window metrics)
        prng_tiles: Optional list of PRNG governance tiles (one per window).
                    Each tile should have drift_status and blocking_rules.
        prng_history: Optional PRNG governance history. If provided, will compute
                     tiles per window from history.

    Returns:
        List of window dicts with added PRNG fields:
        - prng_drift_status: "STABLE" | "DRIFTING" | "VOLATILE" | null
        - prng_volatile_runs: int (count of volatile runs in this window)
        - prng_blocking_rules: List[str] (rule IDs)
    """
    enriched_windows = []
    
    for i, window in enumerate(windows):
        enriched = window.copy()
        
        # Determine PRNG drift status for this window
        if prng_tiles and i < len(prng_tiles):
            tile = prng_tiles[i]
            enriched["prng_drift_status"] = tile.get("drift_status")
            enriched["prng_blocking_rules"] = tile.get("blocking_rules", [])
            # For per-window tiles, volatile_runs is 1 if drift_status is VOLATILE, else 0
            enriched["prng_volatile_runs"] = 1 if tile.get("drift_status") == DriftStatus.VOLATILE.value else 0
        elif prng_history:
            # Compute tile from history for this window
            # For now, use overall history (could be refined to window-specific history)
            radar = build_prng_drift_radar(prng_history)
            tile = build_prng_governance_tile(prng_history, radar=radar)
            enriched["prng_drift_status"] = tile.get("drift_status")
            enriched["prng_blocking_rules"] = tile.get("blocking_rules", [])
            enriched["prng_volatile_runs"] = radar.get("volatile_runs", 0)
        else:
            # No PRNG data available
            enriched["prng_drift_status"] = None
            enriched["prng_blocking_rules"] = []
            enriched["prng_volatile_runs"] = 0
        
        enriched_windows.append(enriched)
    
    return enriched_windows


def compute_per_window_prng_volatility_deltas(
    windows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Produce per-window PRNG volatility deltas.

    Computes the change in PRNG volatility between consecutive windows.

    Args:
        windows: List of window dicts (from align_prng_drift_to_windows).

    Returns:
        List of delta dicts, one per window (first window has no delta):
        - window_index: int
        - prng_volatility_delta: int (change in volatile_runs from previous window)
        - prng_drift_status_transition: str | null
            ("STABLE→DRIFTING", "DRIFTING→VOLATILE", etc., or null if no change)
    """
    deltas = []
    
    for i, window in enumerate(windows):
        delta = {
            "window_index": window.get("window_index", i),
        }
        
        if i == 0:
            # First window has no delta
            delta["prng_volatility_delta"] = 0
            delta["prng_drift_status_transition"] = None
        else:
            prev_window = windows[i - 1]
            prev_volatile = prev_window.get("prng_volatile_runs", 0)
            curr_volatile = window.get("prng_volatile_runs", 0)
            
            delta["prng_volatility_delta"] = curr_volatile - prev_volatile
            
            # Check for drift status transition
            prev_status = prev_window.get("prng_drift_status")
            curr_status = window.get("prng_drift_status")
            
            if prev_status and curr_status and prev_status != curr_status:
                delta["prng_drift_status_transition"] = f"{prev_status}→{curr_status}"
            else:
                delta["prng_drift_status_transition"] = None
        
        deltas.append(delta)
    
    return deltas


def compute_prng_confounded_windows(
    windows: List[Dict[str, Any]],
    volatility_deltas: Optional[List[Dict[str, Any]]] = None,
) -> List[int]:
    """
    Compute prng_confounded_window metric.

    A window is confounded if:
    - PRNG transitions to VOLATILE (from STABLE or DRIFTING)
    - AND delta_p worsens or stalls (mean_delta_p increases or stays same)

    Args:
        windows: List of window dicts with PRNG fields and mean_delta_p.
        volatility_deltas: Optional pre-computed volatility deltas.
                         If None, will be computed from windows.

    Returns:
        List of window indices where PRNG confounded calibration.
    """
    if not windows:
        return []
    
    if volatility_deltas is None:
        volatility_deltas = compute_per_window_prng_volatility_deltas(windows)
    
    confounded_indices = []
    
    for i, (window, delta) in enumerate(zip(windows, volatility_deltas)):
        # Check if PRNG transitioned to VOLATILE
        transition = delta.get("prng_drift_status_transition")
        is_volatile_transition = (
            transition is not None
            and transition.endswith(f"→{DriftStatus.VOLATILE.value}")
        )
        
        if not is_volatile_transition:
            continue
        
        # Check if delta_p worsened or stalled
        if i == 0:
            # First window: no previous delta_p to compare
            continue
        
        prev_window = windows[i - 1]
        prev_delta_p = prev_window.get("mean_delta_p", 0.0)
        curr_delta_p = window.get("mean_delta_p", 0.0)
        
        # delta_p worsened if it increased, stalled if it stayed the same (within tolerance)
        delta_p_worsened = curr_delta_p > prev_delta_p
        delta_p_stalled = abs(curr_delta_p - prev_delta_p) < 0.001  # Small tolerance
        
        if delta_p_worsened or delta_p_stalled:
            confounded_indices.append(window.get("window_index", i))
    
    return sorted(confounded_indices)  # Deterministic ordering


def build_prng_window_audit_table(
    windows: List[Dict[str, Any]],
    max_windows: int = 10,
) -> Dict[str, Any]:
    """
    Build co-registered per-window audit table for PRNG reconciliation.

    Creates a table aligning calibration metrics (delta_p, divergence_rate) with
    PRNG drift metrics (drift_status, volatile_runs, blocking_rules) per window.

    For runs with many windows, the table is bounded to max_windows by selecting:
    - First window
    - Last window
    - Evenly spaced middle windows

    Args:
        windows: List of window dicts (from align_prng_drift_to_windows).
        max_windows: Maximum number of windows to include in audit table (default: 10).

    Returns:
        Dict with:
        - schema_version: "1.0.0"
        - total_windows: int (total number of windows)
        - selection_strategy: "first_last_even_spacing" (selection algorithm identifier)
        - selected_window_indices: List[int] (window indices included in table, sorted)
        - max_windows: int (maximum windows bound used)
        - rows: List[Dict] with per-window audit data:
            - window_index: int
            - mean_delta_p: float
            - divergence_rate: float
            - prng_drift_status: str | null
            - prng_volatile_runs: int
            - prng_blocking_rules: List[str]
            - prng_volatility_delta: int (from previous window)
            - prng_confounded: bool (True if window is confounded)
    """
    if not windows:
        return {
            "schema_version": "1.0.0",
            "total_windows": 0,
            "selection_strategy": "first_last_even_spacing",
            "selected_window_indices": [],
            "max_windows": max_windows,
            "rows": [],
        }
    
    total_windows = len(windows)
    
    # Compute volatility deltas and confounded windows
    volatility_deltas = compute_per_window_prng_volatility_deltas(windows)
    confounded_indices = set(compute_prng_confounded_windows(windows, volatility_deltas))
    
    # Select windows for audit table (bounded to max_windows)
    if total_windows <= max_windows:
        selected_windows = windows
        selected_deltas = volatility_deltas
    else:
        # Select first, last, and evenly spaced middle windows
        selected_indices = _select_bounded_window_indices(total_windows, max_windows)
        selected_windows = [windows[i] for i in selected_indices]
        selected_deltas = [volatility_deltas[i] for i in selected_indices]
    
    # Build audit rows
    rows = []
    for window, delta in zip(selected_windows, selected_deltas):
        window_index = window.get("window_index", 0)
        row = {
            "window_index": window_index,
            "mean_delta_p": window.get("mean_delta_p"),
            "divergence_rate": window.get("divergence_rate"),
            "prng_drift_status": window.get("prng_drift_status"),
            "prng_volatile_runs": window.get("prng_volatile_runs", 0),
            "prng_blocking_rules": window.get("prng_blocking_rules", []),
            "prng_volatility_delta": delta.get("prng_volatility_delta", 0),
            "prng_confounded": window_index in confounded_indices,
        }
        rows.append(row)
    
    # Build selection metadata
    if total_windows <= max_windows:
        selected_indices = list(range(total_windows))
    else:
        selected_indices = _select_bounded_window_indices(total_windows, max_windows)
    
    return {
        "schema_version": "1.0.0",
        "total_windows": total_windows,
        "selection_strategy": "first_last_even_spacing",
        "selected_window_indices": selected_indices,
        "max_windows": max_windows,
        "rows": rows,
    }


def _select_bounded_window_indices(total_windows: int, max_windows: int) -> List[int]:
    """
    Select window indices for bounded audit table.

    Returns first, last, and evenly spaced middle indices.

    Args:
        total_windows: Total number of windows.
        max_windows: Maximum number of windows to select.

    Returns:
        List of selected window indices (sorted, deterministic).
    """
    if total_windows <= max_windows:
        return list(range(total_windows))
    
    # Always include first and last
    selected = {0, total_windows - 1}
    
    # Fill remaining slots with evenly spaced middle windows
    remaining_slots = max_windows - 2
    if remaining_slots > 0:
        # Calculate step size for evenly spaced selection
        step = (total_windows - 1) / (remaining_slots + 1)
        for i in range(1, remaining_slots + 1):
            idx = int(round(i * step))
            # Ensure we don't duplicate first/last
            if idx > 0 and idx < total_windows - 1:
                selected.add(idx)
    
    return sorted(selected)  # Deterministic ordering


def build_prng_calibration_annex(
    windows: List[Dict[str, Any]],
    prng_tiles: Optional[List[Dict[str, Any]]] = None,
    prng_history: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build PRNG calibration annex for evidence["governance"]["p5_calibration"].

    Integrates PRNG drift analysis with calibration windows to provide
    correlation context for calibration behavior.

    Args:
        windows: List of CAL-EXP-1 window dicts.
        prng_tiles: Optional list of PRNG governance tiles (one per window).
        prng_history: Optional PRNG governance history.

    Returns:
        Dict with:
        - schema_version: "1.0.0"
        - prng_confounded_windows: List[int] (window indices where PRNG confounded)
        - prng_delta_summary: Dict with:
            - total_windows: int
            - windows_with_prng_data: int
            - windows_with_volatile_prng: int
            - windows_with_drift_transitions: int
            - prng_drift_status_progression: List[str] (status per window)
    """
    # Align PRNG drift to windows
    enriched_windows = align_prng_drift_to_windows(windows, prng_tiles, prng_history)
    
    # Compute volatility deltas
    volatility_deltas = compute_per_window_prng_volatility_deltas(enriched_windows)
    
    # Compute confounded windows
    confounded_indices = compute_prng_confounded_windows(enriched_windows, volatility_deltas)
    
    # Build summary
    windows_with_prng_data = sum(
        1 for w in enriched_windows if w.get("prng_drift_status") is not None
    )
    windows_with_volatile_prng = sum(
        1 for w in enriched_windows
        if w.get("prng_drift_status") == DriftStatus.VOLATILE.value
    )
    windows_with_drift_transitions = sum(
        1 for d in volatility_deltas
        if d.get("prng_drift_status_transition") is not None
    )
    
    # Build drift status progression (deterministic ordering)
    drift_status_progression = [
        w.get("prng_drift_status") or "UNKNOWN"
        for w in enriched_windows
    ]
    
    # Build window audit table
    audit_table = build_prng_window_audit_table(enriched_windows, max_windows=10)
    
    return {
        "schema_version": "1.0.0",
        "prng_confounded_windows": confounded_indices,
        "prng_delta_summary": {
            "total_windows": len(windows),
            "windows_with_prng_data": windows_with_prng_data,
            "windows_with_volatile_prng": windows_with_volatile_prng,
            "windows_with_drift_transitions": windows_with_drift_transitions,
            "prng_drift_status_progression": drift_status_progression,
        },
        "prng_window_audit_table": audit_table,
    }


def attach_prng_calibration_annex_to_evidence(
    evidence: Dict[str, Any],
    windows: List[Dict[str, Any]],
    prng_tiles: Optional[List[Dict[str, Any]]] = None,
    prng_history: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Attach PRNG calibration annex to evidence["governance"]["p5_calibration"].

    Non-mutating: returns new dict, original unchanged.

    Args:
        evidence: Existing evidence dict (read-only, not modified).
        windows: List of CAL-EXP-1 window dicts.
        prng_tiles: Optional list of PRNG governance tiles.
        prng_history: Optional PRNG governance history.

    Returns:
        New dict with PRNG calibration annex attached under
        evidence["governance"]["p5_calibration"]["prng"].
    """
    import copy
    
    # Deep copy to ensure non-mutation
    enriched = copy.deepcopy(evidence)
    
    # Build annex
    annex = build_prng_calibration_annex(windows, prng_tiles, prng_history)
    
    # Attach to evidence["governance"]["p5_calibration"]["prng"]
    if "governance" not in enriched:
        enriched["governance"] = {}
    if "p5_calibration" not in enriched["governance"]:
        enriched["governance"]["p5_calibration"] = {}
    
    enriched["governance"]["p5_calibration"]["prng"] = annex
    
    return enriched


__all__ = [
    "align_prng_drift_to_windows",
    "compute_per_window_prng_volatility_deltas",
    "compute_prng_confounded_windows",
    "build_prng_window_audit_table",
    "build_prng_calibration_annex",
    "attach_prng_calibration_annex_to_evidence",
]

