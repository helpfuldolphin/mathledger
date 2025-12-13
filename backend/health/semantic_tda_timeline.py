"""Semantic-TDA correlation timeline extractor.

STATUS: PHASE V — SEMANTIC/TDA CROSS-TIE (UPGRADE-2)

Provides windowed correlation extraction for semantic-TDA alignment tracking
over time. Supports integration with TDA pattern classifier for divergence
pattern detection.

SHADOW MODE CONTRACT:
- All functions are read-only and side-effect free
- Outputs are purely observational and do NOT influence governance decisions
- No control flow depends on these outputs
"""

from typing import Any, Dict, List, Optional

from experiments.semantic_consistency_audit import correlate_semantic_and_tda_signals

SEMANTIC_TDA_TIMELINE_SCHEMA_VERSION = "1.0.0"
DEFAULT_WINDOW_SIZE = 10  # 10-cycle windows


def extract_correlation_timeline(
    semantic_timeline_history: List[Dict[str, Any]],
    tda_health_history: List[Dict[str, Any]],
    window_size: int = DEFAULT_WINDOW_SIZE,
) -> Dict[str, Any]:
    """
    Extract semantic-TDA correlation timeline per window.

    STATUS: PHASE V — SEMANTIC/TDA CROSS-TIE (UPGRADE-2)

    Computes correlation between semantic and TDA signals for each window
    of cycles, creating a time series of alignment between the two systems.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned timeline is purely observational
    - No control flow depends on the timeline contents

    Args:
        semantic_timeline_history: List of semantic timeline snapshots, one per cycle.
            Each snapshot should have: timeline, runs_with_critical_signals,
            node_disappearance_events, trend, semantic_status_light
        tda_health_history: List of TDA health snapshots, one per cycle.
            Each snapshot should have: tda_status, block_rate, hss_trend, governance_signal
        window_size: Number of cycles per window (default: 10)

    Returns:
        Dictionary with:
        - schema_version
        - window_size: Window size used
        - total_windows: Number of windows computed
        - windows: List of window correlation records, each with:
            - window_index: 0-indexed window number
            - start_cycle: First cycle in window
            - end_cycle: Last cycle in window
            - correlation_coefficient: Correlation for this window
            - num_key_slices: Number of slices where both signal
            - key_slices: List of slice names (truncated to first 5)
            - semantic_status: Semantic status for window
            - tda_status: TDA status for window
    """
    if len(semantic_timeline_history) != len(tda_health_history):
        raise ValueError(
            f"History lengths must match: semantic={len(semantic_timeline_history)}, "
            f"tda={len(tda_health_history)}"
        )

    if len(semantic_timeline_history) == 0:
        return {
            "schema_version": SEMANTIC_TDA_TIMELINE_SCHEMA_VERSION,
            "window_size": window_size,
            "total_windows": 0,
            "windows": [],
        }

    # Group cycles into windows
    total_cycles = len(semantic_timeline_history)
    total_windows = (total_cycles + window_size - 1) // window_size  # Ceiling division

    windows = []
    for window_idx in range(total_windows):
        start_cycle = window_idx * window_size
        end_cycle = min(start_cycle + window_size, total_cycles)

        # Extract window slices
        semantic_window = semantic_timeline_history[start_cycle:end_cycle]
        tda_window = tda_health_history[start_cycle:end_cycle]

        # Aggregate window data for correlation
        # For semantic: merge timelines, collect critical signals
        aggregated_semantic = _aggregate_semantic_window(semantic_window)
        # For TDA: use most recent snapshot or aggregate
        aggregated_tda = _aggregate_tda_window(tda_window)

        # Compute correlation for this window
        correlation = correlate_semantic_and_tda_signals(aggregated_semantic, aggregated_tda)

        # Extract key information
        key_slices = correlation.get("slices_where_both_signal", [])
        semantic_status = aggregated_semantic.get("semantic_status_light", "GREEN")
        tda_status = aggregated_tda.get("tda_status", "OK")

        windows.append({
            "window_index": window_idx,
            "start_cycle": start_cycle,
            "end_cycle": end_cycle - 1,  # Inclusive end
            "correlation_coefficient": correlation.get("correlation_coefficient", 0.0),
            "num_key_slices": len(key_slices),
            "key_slices": key_slices[:5],  # Truncate to first 5
            "semantic_status": semantic_status,
            "tda_status": tda_status,
        })

    return {
        "schema_version": SEMANTIC_TDA_TIMELINE_SCHEMA_VERSION,
        "window_size": window_size,
        "total_windows": total_windows,
        "windows": windows,
    }


def _aggregate_semantic_window(semantic_window: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate semantic timeline snapshots for a window."""
    if not semantic_window:
        return {
            "timeline": [],
            "runs_with_critical_signals": [],
            "node_disappearance_events": [],
            "trend": "STABLE",
            "semantic_status_light": "GREEN",
        }

    # Collect all critical signals
    all_critical_runs = set()
    all_disappearance_events = []
    all_timeline_entries = []
    statuses = []

    for snapshot in semantic_window:
        all_critical_runs.update(snapshot.get("runs_with_critical_signals", []))
        all_disappearance_events.extend(snapshot.get("node_disappearance_events", []))
        all_timeline_entries.extend(snapshot.get("timeline", []))
        statuses.append(snapshot.get("semantic_status_light", "GREEN"))

    # Determine overall status (most severe)
    status_priority = {"RED": 2, "YELLOW": 1, "GREEN": 0}
    overall_status = max(statuses, key=lambda s: status_priority.get(s, 0), default="GREEN")

    # Determine trend (most common, or DRIFTING if any critical)
    trends = [s.get("trend", "STABLE") for s in semantic_window]
    if any("DRIFTING" in t for t in trends) or all_critical_runs:
        overall_trend = "DRIFTING"
    elif any("VOLATILE" in t for t in trends):
        overall_trend = "VOLATILE"
    else:
        overall_trend = "STABLE"

    return {
        "timeline": all_timeline_entries,
        "runs_with_critical_signals": sorted(list(all_critical_runs)),
        "node_disappearance_events": all_disappearance_events,
        "trend": overall_trend,
        "semantic_status_light": overall_status,
    }


def _aggregate_tda_window(tda_window: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate TDA health snapshots for a window."""
    if not tda_window:
        return {
            "tda_status": "OK",
            "block_rate": 0.0,
            "hss_trend": "STABLE",
            "governance_signal": "OK",
        }

    # Use most recent snapshot (or aggregate if needed)
    # For now, use most recent as representative
    latest = tda_window[-1]

    # Optionally aggregate: average block_rate, most severe status
    block_rates = [s.get("block_rate", 0.0) for s in tda_window if "block_rate" in s]
    avg_block_rate = sum(block_rates) / len(block_rates) if block_rates else 0.0

    statuses = [s.get("tda_status", "OK") for s in tda_window]
    status_priority = {"ALERT": 2, "ATTENTION": 1, "OK": 0}
    overall_status = max(statuses, key=lambda s: status_priority.get(s, 0), default="OK")

    return {
        "tda_status": overall_status,
        "block_rate": avg_block_rate,
        "hss_trend": latest.get("hss_trend", "STABLE"),
        "governance_signal": latest.get("governance_signal", "OK"),
    }


def extract_correlation_trends(
    correlation_timeline: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Extract correlation trends from timeline for pattern classifier integration.

    STATUS: PHASE V — SEMANTIC/TDA CROSS-TIE (UPGRADE-2)

    Computes trend statistics from correlation timeline to support TDA pattern
    classifier inputs. Provides correlation slope, variance, and regime classification.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned trends are purely observational
    - No control flow depends on the trend contents

    Args:
        correlation_timeline: Output from extract_correlation_timeline()

    Returns:
        Dictionary with:
        - schema_version
        - correlation_mean: Mean correlation across all windows
        - correlation_variance: Variance of correlation values
        - correlation_slope: Linear regression slope (trend direction)
        - correlation_regime: "ALIGNED" | "MISALIGNED" | "VOLATILE" | "STABLE"
        - windows_with_high_correlation: Count of windows with correlation >= 0.7
        - windows_with_negative_correlation: Count of windows with correlation <= -0.3
    """
    windows = correlation_timeline.get("windows", [])
    if not windows:
        return {
            "schema_version": SEMANTIC_TDA_TIMELINE_SCHEMA_VERSION,
            "correlation_mean": 0.0,
            "correlation_variance": 0.0,
            "correlation_slope": 0.0,
            "correlation_regime": "STABLE",
            "windows_with_high_correlation": 0,
            "windows_with_negative_correlation": 0,
        }

    correlations = [w.get("correlation_coefficient", 0.0) for w in windows]

    # Compute mean
    correlation_mean = sum(correlations) / len(correlations) if correlations else 0.0

    # Compute variance
    if len(correlations) > 1:
        variance = sum((c - correlation_mean) ** 2 for c in correlations) / len(correlations)
    else:
        variance = 0.0

    # Compute slope (simple linear regression)
    n = len(correlations)
    if n > 1:
        x_values = list(range(n))
        x_mean = sum(x_values) / n
        y_mean = correlation_mean
        numerator = sum((x_values[i] - x_mean) * (correlations[i] - y_mean) for i in range(n))
        denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))
        slope = numerator / denominator if denominator != 0 else 0.0
    else:
        slope = 0.0

    # Classify regime
    high_corr_count = sum(1 for c in correlations if c >= 0.7)
    neg_corr_count = sum(1 for c in correlations if c <= -0.3)

    if variance > 0.1:
        regime = "VOLATILE"
    elif correlation_mean >= 0.7:
        regime = "ALIGNED"
    elif correlation_mean <= -0.3:
        regime = "MISALIGNED"
    else:
        regime = "STABLE"

    return {
        "schema_version": SEMANTIC_TDA_TIMELINE_SCHEMA_VERSION,
        "correlation_mean": round(correlation_mean, 3),
        "correlation_variance": round(variance, 3),
        "correlation_slope": round(slope, 4),
        "correlation_regime": regime,
        "windows_with_high_correlation": high_corr_count,
        "windows_with_negative_correlation": neg_corr_count,
    }


def visualize_correlation_trajectory(
    correlation_timeline: Dict[str, Any],
    format: str = "ascii",
) -> str:
    """
    Generate visualization stub for cross-system alignment trajectory.

    STATUS: PHASE V — SEMANTIC/TDA CROSS-TIE (UPGRADE-2)

    Creates a simple visualization of correlation over time, showing where
    semantic and TDA systems agree or disagree across windows.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from string construction)
    - The returned visualization is purely observational
    - No control flow depends on the visualization contents

    Args:
        correlation_timeline: Output from extract_correlation_timeline()
        format: Output format ("ascii" or "json")

    Returns:
        Visualization string (ASCII chart) or JSON summary
    """
    windows = correlation_timeline.get("windows", [])
    if not windows:
        if format == "json":
            return '{"summary": "No windows to visualize"}'
        return "No correlation data available.\n"

    if format == "json":
        # JSON summary format
        correlations = [w.get("correlation_coefficient", 0.0) for w in windows]
        trends = extract_correlation_trends(correlation_timeline)
        return (
            '{\n'
            f'  "schema_version": "{SEMANTIC_TDA_TIMELINE_SCHEMA_VERSION}",\n'
            f'  "total_windows": {len(windows)},\n'
            f'  "correlation_range": [{min(correlations):.3f}, {max(correlations):.3f}],\n'
            f'  "correlation_mean": {trends["correlation_mean"]:.3f},\n'
            f'  "correlation_regime": "{trends["correlation_regime"]}",\n'
            f'  "windows": [\n'
            + ",\n".join(
                f'    {{"window": {w["window_index"]}, "cycles": [{w["start_cycle"]}, {w["end_cycle"]}], '
                f'"correlation": {w["correlation_coefficient"]:.3f}, "slices": {w["num_key_slices"]}}}'
                for w in windows
            )
            + "\n  ]\n}"
        )

    # ASCII visualization
    lines = []
    lines.append("Semantic-TDA Correlation Trajectory")
    lines.append("=" * 60)
    lines.append(f"Total Windows: {len(windows)}")
    lines.append("")

    # Create simple ASCII chart
    correlations = [w.get("correlation_coefficient", 0.0) for w in windows]
    min_corr = min(correlations) if correlations else 0.0
    max_corr = max(correlations) if correlations else 0.0
    range_corr = max_corr - min_corr if max_corr != min_corr else 1.0

    chart_width = 50
    lines.append("Correlation over Windows:")
    lines.append("")
    lines.append("Window | Cycles      | Correlation | Chart")
    lines.append("-" * 60)

    for w in windows:
        window_idx = w["window_index"]
        start = w["start_cycle"]
        end = w["end_cycle"]
        corr = w["correlation_coefficient"]
        num_slices = w["num_key_slices"]

        # Normalize correlation to chart position
        normalized = (corr - min_corr) / range_corr if range_corr > 0 else 0.5
        chart_pos = int(normalized * chart_width)

        # Create bar chart
        bar = "█" * chart_pos + "░" * (chart_width - chart_pos)

        lines.append(
            f"{window_idx:6d} | {start:4d}-{end:4d} | {corr:11.3f} | {bar} "
            f"({num_slices} slices)"
        )

    lines.append("")
    lines.append(f"Range: [{min_corr:.3f}, {max_corr:.3f}]")
    lines.append("")

    # Add trend summary
    trends = extract_correlation_trends(correlation_timeline)
    lines.append("Trend Summary:")
    lines.append(f"  Mean Correlation: {trends['correlation_mean']:.3f}")
    lines.append(f"  Variance: {trends['correlation_variance']:.3f}")
    lines.append(f"  Slope: {trends['correlation_slope']:.4f}")
    lines.append(f"  Regime: {trends['correlation_regime']}")

    return "\n".join(lines)


def compute_phase_lag_index(
    correlation_timeline: Dict[str, Any],
    trends: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compute phase-lag index based on correlation slope, lagged sign changes, and alignment strength.

    STATUS: PHASE V — SEMANTIC/TDA CROSS-TIE (UPGRADE-2)

    Quantifies phase lag severity numerically as an index in [0, 1], where:
    - 0.0 = No phase lag (systems perfectly aligned)
    - 1.0 = Maximum phase lag (systems severely misaligned)

    The index combines:
    1. Correlation slope: Negative or oscillating slopes indicate lag
    2. Lagged sign changes: Frequent sign flips indicate temporal misalignment
    3. Alignment strength: Low alignment suggests phase lag

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned index is purely observational
    - No control flow depends on the phase lag index

    Args:
        correlation_timeline: Output from extract_correlation_timeline()
        trends: Optional pre-computed trends from extract_correlation_trends()
            If not provided, will be computed internally

    Returns:
        Dictionary with:
        - schema_version
        - phase_lag_index: Float in [0, 1] indicating phase lag severity
        - correlation_slope_contribution: Contribution from slope (0-1)
        - sign_change_contribution: Contribution from sign changes (0-1)
        - alignment_contribution: Contribution from alignment strength (0-1)
        - worst_windows: List of window indices contributing most to phase lag
    """
    windows = correlation_timeline.get("windows", [])
    if not windows:
        return {
            "schema_version": SEMANTIC_TDA_TIMELINE_SCHEMA_VERSION,
            "phase_lag_index": 0.0,
            "correlation_slope_contribution": 0.0,
            "sign_change_contribution": 0.0,
            "alignment_contribution": 0.0,
            "worst_windows": [],
        }

    # Compute trends if not provided
    if trends is None:
        trends = extract_correlation_trends(correlation_timeline)

    correlations = [w.get("correlation_coefficient", 0.0) for w in windows]

    # 1. Correlation slope contribution
    # Negative slope or high variance in slope indicates lag
    correlation_slope = trends.get("correlation_slope", 0.0)
    correlation_variance = trends.get("correlation_variance", 0.0)

    # Normalize slope: negative slopes indicate lag
    # Map slope from [-0.1, 0.1] to [0, 1] where negative = high lag
    slope_contribution = 0.0
    if correlation_slope < 0:
        # Negative slope indicates deteriorating alignment (lag)
        slope_contribution = min(1.0, abs(correlation_slope) * 10.0)  # Scale by 10
    elif correlation_variance > 0.1:
        # High variance also indicates instability (potential lag)
        slope_contribution = min(1.0, correlation_variance * 5.0)  # Scale by 5

    # 2. Lagged sign changes contribution
    # Count sign changes in correlation sequence
    sign_changes = 0
    for i in range(1, len(correlations)):
        if (correlations[i] >= 0) != (correlations[i - 1] >= 0):
            sign_changes += 1

    # Normalize sign change rate: more changes = more lag
    sign_change_rate = sign_changes / len(correlations) if len(correlations) > 1 else 0.0
    # High sign change rate (e.g., > 0.3) indicates severe lag
    sign_change_contribution = min(1.0, sign_change_rate * 2.0)  # Scale by 2

    # 3. Alignment strength contribution
    # Low alignment strength indicates phase lag
    correlation_mean = trends.get("correlation_mean", 0.0)
    # Map correlation_mean from [-1, 1] to alignment strength [0, 1]
    # Low correlation = high lag contribution
    alignment_strength = (correlation_mean + 1.0) / 2.0  # [0, 1]
    alignment_contribution = 1.0 - alignment_strength  # Invert: low alignment = high lag

    # Combine contributions with weighted average
    # Weights: slope=0.4, sign_changes=0.3, alignment=0.3
    phase_lag_index = (
        0.4 * slope_contribution
        + 0.3 * sign_change_contribution
        + 0.3 * alignment_contribution
    )
    phase_lag_index = max(0.0, min(1.0, phase_lag_index))  # Clamp to [0, 1]

    # Identify worst windows (highest contribution to phase lag)
    # Score each window based on correlation instability
    window_scores = []
    for i, w in enumerate(windows):
        corr = w.get("correlation_coefficient", 0.0)
        score = 0.0

        # Low absolute correlation contributes to lag
        score += (1.0 - abs(corr)) * 0.5

        # Sign changes with neighbors
        if i > 0:
            prev_corr = windows[i - 1].get("correlation_coefficient", 0.0)
            if (corr >= 0) != (prev_corr >= 0):
                score += 0.3
        if i < len(windows) - 1:
            next_corr = windows[i + 1].get("correlation_coefficient", 0.0)
            if (corr >= 0) != (next_corr >= 0):
                score += 0.2

        window_scores.append((i, score))

    # Sort by score (descending) and take top 3 worst windows
    window_scores.sort(key=lambda x: x[1], reverse=True)
    worst_windows = [w[0] for w in window_scores[:3] if w[1] > 0.3]

    return {
        "schema_version": SEMANTIC_TDA_TIMELINE_SCHEMA_VERSION,
        "phase_lag_index": round(phase_lag_index, 3),
        "correlation_slope_contribution": round(slope_contribution, 3),
        "sign_change_contribution": round(sign_change_contribution, 3),
        "alignment_contribution": round(alignment_contribution, 3),
        "worst_windows": worst_windows,
    }


def visualize_phase_lag(
    correlation_timeline: Dict[str, Any],
    phase_lag_data: Optional[Dict[str, Any]] = None,
    format: str = "ascii",
) -> str:
    """
    Generate visualization for phase lag index over time.

    STATUS: PHASE V — SEMANTIC/TDA CROSS-TIE (UPGRADE-2)

    Creates visualization showing phase lag index trajectory and worst windows.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from string construction)
    - The returned visualization is purely observational
    - No control flow depends on the visualization contents

    Args:
        correlation_timeline: Output from extract_correlation_timeline()
        phase_lag_data: Optional pre-computed phase lag data from compute_phase_lag_index()
            If not provided, will be computed internally
        format: Output format ("ascii" or "json")

    Returns:
        Visualization string (ASCII chart) or JSON summary
    """
    windows = correlation_timeline.get("windows", [])
    if not windows:
        if format == "json":
            return '{"summary": "No windows to visualize"}'
        return "No correlation data available for phase lag analysis.\n"

    # Compute phase lag if not provided
    if phase_lag_data is None:
        trends = extract_correlation_trends(correlation_timeline)
        phase_lag_data = compute_phase_lag_index(correlation_timeline, trends=trends)

    phase_lag_index = phase_lag_data.get("phase_lag_index", 0.0)
    worst_windows = phase_lag_data.get("worst_windows", [])

    if format == "json":
        # JSON summary format
        correlations = [w.get("correlation_coefficient", 0.0) for w in windows]
        return (
            '{\n'
            f'  "schema_version": "{SEMANTIC_TDA_TIMELINE_SCHEMA_VERSION}",\n'
            f'  "phase_lag_index": {phase_lag_index:.3f},\n'
            f'  "correlation_slope_contribution": {phase_lag_data.get("correlation_slope_contribution", 0.0):.3f},\n'
            f'  "sign_change_contribution": {phase_lag_data.get("sign_change_contribution", 0.0):.3f},\n'
            f'  "alignment_contribution": {phase_lag_data.get("alignment_contribution", 0.0):.3f},\n'
            f'  "worst_windows": {worst_windows},\n'
            f'  "windows": [\n'
            + ",\n".join(
                f'    {{"window": {w["window_index"]}, "cycles": [{w["start_cycle"]}, {w["end_cycle"]}], '
                f'"correlation": {w["correlation_coefficient"]:.3f}, "phase_lag_contribution": '
                f'{(1.0 - abs(w.get("correlation_coefficient", 0.0))) * 0.5:.3f}}}'
                for w in windows
            )
            + "\n  ]\n}"
        )

    # ASCII visualization
    lines = []
    lines.append("Semantic-TDA Phase Lag Analysis")
    lines.append("=" * 60)
    lines.append(f"Phase Lag Index: {phase_lag_index:.3f} (0.0 = aligned, 1.0 = severe lag)")
    lines.append("")

    # Show contributions
    lines.append("Contributions to Phase Lag:")
    lines.append(f"  Correlation Slope: {phase_lag_data.get('correlation_slope_contribution', 0.0):.3f}")
    lines.append(f"  Sign Changes: {phase_lag_data.get('sign_change_contribution', 0.0):.3f}")
    lines.append(f"  Alignment Strength: {phase_lag_data.get('alignment_contribution', 0.0):.3f}")
    lines.append("")

    # Show worst windows
    if worst_windows:
        lines.append("Worst Windows (Highest Phase Lag Contribution):")
        for win_idx in worst_windows:
            if win_idx < len(windows):
                w = windows[win_idx]
                lines.append(
                    f"  Window {win_idx}: cycles [{w['start_cycle']}-{w['end_cycle']}], "
                    f"correlation={w.get('correlation_coefficient', 0.0):.3f}"
                )
        lines.append("")

    # Create phase lag index chart over time
    lines.append("Phase Lag Index over Windows:")
    lines.append("")
    lines.append("Window | Cycles      | Correlation | Phase Lag | Chart")
    lines.append("-" * 70)

    chart_width = 40
    for w in windows:
        window_idx = w["window_index"]
        start = w["start_cycle"]
        end = w["end_cycle"]
        corr = w["correlation_coefficient"]

        # Compute per-window phase lag contribution
        window_lag_contrib = (1.0 - abs(corr)) * 0.5
        # Add sign change penalty
        if window_idx > 0:
            prev_corr = windows[window_idx - 1].get("correlation_coefficient", 0.0)
            if (corr >= 0) != (prev_corr >= 0):
                window_lag_contrib += 0.3

        # Normalize to chart position
        chart_pos = int(window_lag_contrib * chart_width)
        bar = "█" * chart_pos + "░" * (chart_width - chart_pos)

        lines.append(
            f"{window_idx:6d} | {start:4d}-{end:4d} | {corr:11.3f} | "
            f"{window_lag_contrib:9.3f} | {bar}"
        )

    lines.append("")
    lines.append(f"Overall Phase Lag Index: {phase_lag_index:.3f}")

    return "\n".join(lines)


def explain_phase_lag_vs_divergence(
    phase_lag: Dict[str, Any],
    divergence_decomp: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Explain phase lag in terms of state lag vs outcome noise.

    STATUS: PHASE V — SEMANTIC/TDA CROSS-TIE (UPGRADE-2)

    Connects phase-lag index to calibration reconciliation by distinguishing
    between state lag (temporal misalignment in state predictions) and outcome
    noise (random outcome prediction errors). This enables evidence-based
    diagnosis of whether phase lag indicates true state lag or generalized failure.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned interpretation is purely observational
    - No control flow depends on the interpretation

    Args:
        phase_lag: Output from compute_phase_lag_index()
            Expected keys: phase_lag_index, correlation_slope_contribution,
            sign_change_contribution, alignment_contribution
        divergence_decomp: Divergence decomposition dictionary
            Expected keys: state_divergence_rate, success_divergence_rate
            (or outcome_divergence_rate_success)

    Returns:
        Dictionary with:
        - schema_version
        - phase_lag_index: Phase lag index from input
        - state_divergence_rate: State divergence rate from decomposition
        - outcome_divergence_rate_success: Success/outcome divergence rate
        - interpretation: "STATE_LAG_DOMINANT" | "OUTCOME_NOISE_DOMINANT" | "MIXED" | "INSUFFICIENT_DATA"
        - notes: List of neutral explanatory notes
    """
    phase_lag_index = phase_lag.get("phase_lag_index")

    # Extract divergence rates (use None as default to detect missing keys)
    state_divergence_rate = divergence_decomp.get("state_divergence_rate")
    # Try multiple possible keys for outcome divergence
    outcome_divergence_rate_success = divergence_decomp.get("outcome_divergence_rate_success")
    if outcome_divergence_rate_success is None:
        outcome_divergence_rate_success = divergence_decomp.get("success_divergence_rate")

    # Determine interpretation
    notes = []
    interpretation = "INSUFFICIENT_DATA"

    # Check if we have sufficient data
    # Check for None explicitly (not just falsy values, since 0.0 is valid)
    if state_divergence_rate is None or outcome_divergence_rate_success is None:
        notes.append("Divergence decomposition missing required rates")
        interpretation = "INSUFFICIENT_DATA"
    elif phase_lag_index is None:
        notes.append("Phase lag index invalid or missing")
        interpretation = "INSUFFICIENT_DATA"
    else:
        # Determine interpretation based on relative rates and phase lag
        # High phase lag + high state divergence = state lag dominant
        # High phase lag + low state divergence + high outcome divergence = outcome noise dominant
        # High phase lag + both high = mixed

        # Thresholds for "high"
        HIGH_PHASE_LAG = 0.4
        HIGH_STATE_DIVERGENCE = 0.3
        HIGH_OUTCOME_DIVERGENCE = 0.2

        phase_lag_high = phase_lag_index >= HIGH_PHASE_LAG
        state_div_high = state_divergence_rate >= HIGH_STATE_DIVERGENCE
        outcome_div_high = outcome_divergence_rate_success >= HIGH_OUTCOME_DIVERGENCE

        if not phase_lag_high:
            # Low phase lag: systems are aligned, divergence is likely noise
            if state_div_high and outcome_div_high:
                interpretation = "MIXED"
                notes.append("Low phase lag but both state and outcome divergence present")
            elif state_div_high:
                interpretation = "MIXED"
                notes.append("Low phase lag but state divergence present (may indicate calibration issue)")
            else:
                interpretation = "OUTCOME_NOISE_DOMINANT"
                notes.append("Low phase lag suggests systems aligned; outcome divergence likely noise")
        else:
            # High phase lag: temporal misalignment present
            if state_div_high and not outcome_div_high:
                # High state divergence, low outcome divergence = state lag
                interpretation = "STATE_LAG_DOMINANT"
                notes.append(
                    f"High phase lag ({phase_lag_index:.3f}) correlates with high state divergence "
                    f"({state_divergence_rate:.3f}), indicating temporal misalignment in state predictions"
                )
            elif not state_div_high and outcome_div_high:
                # Low state divergence, high outcome divergence = outcome noise
                interpretation = "OUTCOME_NOISE_DOMINANT"
                notes.append(
                    f"High phase lag ({phase_lag_index:.3f}) but low state divergence "
                    f"({state_divergence_rate:.3f}); outcome divergence ({outcome_divergence_rate_success:.3f}) "
                    "likely due to outcome prediction noise rather than state lag"
                )
            elif state_div_high and outcome_div_high:
                # Both high = mixed
                interpretation = "MIXED"
                notes.append(
                    f"High phase lag ({phase_lag_index:.3f}) with both state "
                    f"({state_divergence_rate:.3f}) and outcome ({outcome_divergence_rate_success:.3f}) "
                    "divergence; suggests both temporal misalignment and outcome noise"
                )
            else:
                # High phase lag but low divergence rates = unclear
                interpretation = "MIXED"
                notes.append(
                    f"High phase lag ({phase_lag_index:.3f}) but low divergence rates; "
                    "may indicate subtle temporal misalignment not yet manifesting as divergence"
                )

    # Add additional context notes
    if interpretation != "INSUFFICIENT_DATA":
        if phase_lag_index >= 0.6:
            notes.append("Phase lag index indicates severe temporal misalignment")
        elif phase_lag_index >= 0.4:
            notes.append("Phase lag index indicates significant temporal misalignment")

        if state_divergence_rate >= 0.5:
            notes.append("State divergence rate is high, suggesting twin state predictions lag real state")
        elif state_divergence_rate >= 0.3:
            notes.append("State divergence rate is moderate")

        if outcome_divergence_rate_success >= 0.3:
            notes.append("Outcome divergence rate is high, suggesting outcome prediction errors")
        elif outcome_divergence_rate_success >= 0.15:
            notes.append("Outcome divergence rate is moderate")

    # Thresholds used for interpretation
    thresholds = {
        "high_phase_lag": 0.4,
        "high_state_divergence": 0.3,
        "high_outcome_divergence": 0.2,
    }

    # Basis: source of data
    basis = {
        "phase_lag": "semantic_tda_timeline",
        "divergence_decomp": "runtime_profile_calibration.decompose_divergence_components",
    }

    return {
        "schema_version": SEMANTIC_TDA_TIMELINE_SCHEMA_VERSION,
        "phase_lag_index": round(phase_lag_index, 3) if phase_lag_index is not None else None,
        "state_divergence_rate": round(state_divergence_rate, 4) if state_divergence_rate is not None else None,
        "outcome_divergence_rate_success": round(outcome_divergence_rate_success, 4) if outcome_divergence_rate_success is not None else None,
        "interpretation": interpretation,
        "notes": notes,
        "thresholds": thresholds,
        "basis": basis,
    }


__all__ = [
    "SEMANTIC_TDA_TIMELINE_SCHEMA_VERSION",
    "DEFAULT_WINDOW_SIZE",
    "extract_correlation_timeline",
    "extract_correlation_trends",
    "visualize_correlation_trajectory",
    "compute_phase_lag_index",
    "visualize_phase_lag",
    "explain_phase_lag_vs_divergence",
]



