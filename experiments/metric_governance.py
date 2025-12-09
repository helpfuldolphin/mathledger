"""
PHASE III — LIVE METRIC GOVERNANCE
Core logic for metric conformance comparison, promotion gating, and timeline analysis.
"""
import json
import os
import statistics
from typing import Any, Dict, List, Tuple, Optional

# --- Constants ---
POLICY_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'metric_promotion_policy.json')
CONFORMANCE_LEVELS = ["L0_reproducible", "L1_deterministic", "L2_domain_coverage"]

# --- Core Functions ---

def load_promotion_policy(policy_path: str = POLICY_PATH) -> Dict[str, Any]:
    """Loads the metric promotion policy file."""
    with open(policy_path, 'r') as f:
        return json.load(f)

def get_policy_for_metric(metric_name: str, policy: Dict[str, Any]) -> Dict[str, Any]:
    """Finds the appropriate policy for a given metric name."""
    for family, family_policy in policy.items():
        if metric_name.startswith(family):
            return family_policy
    return policy["default"]

def compare_snapshots(baseline: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Compares two snapshots and identifies all regressions."""
    comparison = {"regressions": []}
    for level in CONFORMANCE_LEVELS:
        if candidate["levels"][level]["status"] != "PASS":
            comparison["regressions"].append({
                "level": level,
                "reason": f"Candidate failed required level {level}.",
                "severity": "CRITICAL"
            })
    if candidate["levels"]["L3_regression"]["status"] != "PASS":
        comparison["regressions"].append({
            "level": "L3_regression",
            "reason": candidate["levels"]["L3_regression"].get("details", "L3 regression check failed."),
            "severity": "MINOR"
        })
    return comparison

def can_promote_metric(
    baseline: Dict[str, Any], 
    candidate: Dict[str, Any],
    policy: Optional[Dict[str, Any]] = None
) -> Tuple[bool, str]:
    """
    Determines if a metric is eligible for promotion based on conformance
    snapshots and a promotion policy.
    """
    if policy is None:
        policy = load_promotion_policy()
    
    metric_policy = get_policy_for_metric(candidate["metric_name"], policy)
    comparison = compare_snapshots(baseline, candidate)
    
    blockers = []
    
    # Check for critical regressions up to the required level
    required_level_index = CONFORMANCE_LEVELS.index(metric_policy["required_level"])
    for i, level in enumerate(CONFORMANCE_LEVELS):
        if i <= required_level_index:
            if candidate["levels"][level]["status"] != "PASS":
                blockers.append(f"failed required level {level}")

    # Check for L3 regression if not allowed
    if not metric_policy["allow_l3_regression"]:
        if any(reg["level"] == "L3_regression" for reg in comparison["regressions"]):
            blockers.append("L3 regression is not tolerated by policy")

    if blockers:
        return False, f"Promotion denied for '{candidate['metric_name']}': {'; '.join(blockers)}."
        
    return True, f"Promotion approved for '{candidate['metric_name']}'. No blocking regressions found."

def build_metric_conformance_timeline(
    snapshot_paths: List[str],
    flapping_window: int = 5,
    flapping_threshold: int = 3,
    drift_window: int = 3,
    regression_outlier_threshold: float = 2.0
) -> Dict[str, Any]:
    """
    Processes a list of snapshot files to build a conformance history for each metric,
    including advanced analytics for flapping, drift, and outlier regressions.
    """
    snapshots_by_metric: Dict[str, List[Dict[str, Any]]] = {}
    for path in snapshot_paths:
        with open(path, 'r') as f:
            snapshot = json.load(f)
            metric_name = snapshot["metric_name"]
            if metric_name not in snapshots_by_metric:
                snapshots_by_metric[metric_name] = []
            snapshots_by_metric[metric_name].append(snapshot)

    for metric_name in snapshots_by_metric:
        snapshots_by_metric[metric_name].sort(key=lambda s: s["timestamp_utc"])

    timeline: Dict[str, Any] = {}
    for metric_name, snapshots in snapshots_by_metric.items():
        total_runs = len(snapshots)
        
        # Basic regression count and streak
        regressions = [s for s in snapshots if s["levels"]["L3_regression"]["status"] != "PASS"]
        
        streak_type = None
        streak_count = 0
        if snapshots:
            last_snapshot = snapshots[-1]
            current_status = last_snapshot["levels"]["L3_regression"]["status"]
            for snapshot in reversed(snapshots):
                if snapshot["levels"]["L3_regression"]["status"] == current_status:
                    streak_count += 1
                else:
                    break
            streak_type = current_status
        
        # --- Advanced Analytics ---
        
        # Metric Flapping
        is_flapping = False
        if total_runs >= flapping_window:
            flapping_events = 0
            recent_statuses = [s["levels"]["L3_regression"]["status"] for s in snapshots[-flapping_window:]]
            for i in range(flapping_window - 1):
                if recent_statuses[i] != recent_statuses[i+1]:
                    flapping_events += 1
            if flapping_events >= flapping_threshold:
                is_flapping = True

        # Long-term Drift
        l3_values = [s["levels"]["L3_regression"]["value"] for s in snapshots if "value" in s["levels"]["L3_regression"]]
        long_term_drift_magnitude = 0.0
        if len(l3_values) > drift_window:
            historical_avg = statistics.mean(l3_values[:-drift_window])
            current_avg = statistics.mean(l3_values[-drift_window:])
            long_term_drift_magnitude = current_avg - historical_avg

        # Outliers in Performance Regressions
        is_regression_outlier = False
        regression_magnitudes = []
        for i in range(1, len(snapshots)):
            if snapshots[i]["levels"]["L3_regression"]["status"] == "FAIL":
                prev_value = snapshots[i-1]["levels"]["L3_regression"].get("value")
                curr_value = snapshots[i]["levels"]["L3_regression"].get("value")
                if prev_value is not None and curr_value is not None:
                    regression_magnitudes.append(abs(curr_value - prev_value))
        
        if regression_magnitudes and snapshots[-1]["levels"]["L3_regression"]["status"] == "FAIL":
            last_regression_magnitude = abs(snapshots[-1]["levels"]["L3_regression"].get("value", 0) - snapshots[-2]["levels"]["L3_regression"].get("value", 0)) if total_runs >= 2 else 0
            if last_regression_magnitude > regression_outlier_threshold * statistics.mean(regression_magnitudes):
                is_regression_outlier = True

        timeline[metric_name] = {
            "total_runs": total_runs,
            "regression_count": len(regressions),
            "last_n_snapshots": snapshots[-5:], # Keep last 5 for brevity
            "current_streak": {
                "status": streak_type,
                "count": streak_count,
            },
            "advanced_analytics": {
                "is_flapping": is_flapping,
                "long_term_drift": long_term_drift_magnitude,
                "is_regression_outlier": is_regression_outlier,
            }
        }
    return timeline

if __name__ == '__main__':
    print("--- Running policy-based governance checks ---")
    
    # --- Load test data using paths relative to this script ---
    script_dir = os.path.dirname(__file__)
    
    with open(os.path.join(script_dir, '..', 'artifacts', 'snapshots', 'baseline', 'uplift_u2_density.json'), 'r') as f:
        baseline_density = json.load(f)
    with open(os.path.join(script_dir, '..', 'artifacts', 'snapshots', 'candidate', 'uplift_u2_density.json'), 'r') as f:
        candidate_density = json.load(f) # Has L3 regression
    
    with open(os.path.join(script_dir, '..', 'artifacts', 'snapshots', 'baseline', 'uplift_u2_chain_length.json'), 'r') as f:
        baseline_chain = json.load(f)
    with open(os.path.join(script_dir, '..', 'artifacts', 'snapshots', 'candidate', 'uplift_u2_chain_length.json'), 'r') as f:
        candidate_chain = json.load(f) # No regression

    # --- Test Case 1: uplift_u2_density (L3 regression is allowed) ---
    can_promote, reason = can_promote_metric(baseline_density, candidate_density)
    print(f"\nMetric '{candidate_density['metric_name']}' Promotion Status: {can_promote}")
    print(f"Reason: {reason}")
    # This should be TRUE because the 'uplift_u2' policy allows L3 regressions
    assert can_promote 

    # --- Test Case 2: uplift_u2_chain_length (no regression) ---
    can_promote, reason = can_promote_metric(baseline_chain, candidate_chain)
    print(f"\nMetric '{candidate_chain['metric_name']}' Promotion Status: {can_promote}")
    print(f"Reason: {reason}")
    assert can_promote
    
    # --- Test Case 3: A metric failing under the default policy ---
    # We'll re-use the density metric but apply a custom 'default' policy to it
    # by filtering the policy dictionary.
    policy = load_promotion_policy()
    default_policy_only = {"default": policy["default"]}
    
    can_promote, reason = can_promote_metric(baseline_density, candidate_density, policy=default_policy_only)
    print(f"\nMetric '{candidate_density['metric_name']}' (under default policy) Promotion Status: {can_promote}")
    print(f"Reason: {reason}")
    # This should be FALSE because the default policy does NOT allow L3 regressions
    assert not can_promote

    print("\n--- Policy-based checks complete ---")


    # --- Test Timeline Builder ---
    print("\n--- Testing Timeline Builder ---")
    import glob
    # Use glob to find all our test snapshots
    snapshot_files = glob.glob(os.path.join(script_dir, '..', 'artifacts', 'snapshots', '**', '*.json'), recursive=True)
    
    timeline = build_metric_conformance_timeline(snapshot_files)
    print(json.dumps(timeline, indent=2))

    # Assertions for the density metric
    density_timeline = timeline.get("uplift_u2_density", {})
    assert density_timeline.get("total_runs") == 4
    assert density_timeline.get("regression_count") == 1
    assert density_timeline.get("current_streak", {}).get("status") == "FAIL"
    assert density_timeline.get("current_streak", {}).get("count") == 1
    
    # Assertions for advanced analytics
    assert density_timeline.get("advanced_analytics", {}).get("is_flapping") is False
    assert density_timeline.get("advanced_analytics", {}).get("long_term_drift") != 0.0
    assert density_timeline.get("advanced_analytics", {}).get("is_regression_outlier") is False


    print("\n--- Timeline builder checks complete ---")


def generate_reports(timeline: Dict[str, Any], policy_registry: Dict[str, Any]) -> Tuple[str, str]:
    """
    Generates developer-oriented and director-oriented reports from timeline data.
    """
    developer_report = "# Metric Conformance Report (Developer View)\n\n"
    director_summary = "# Metric Conformance Summary (Director's View)\n\n"

    overall_status = "GREEN" # Assume all good until critical failure
    
    for metric_name, data in timeline.items():
        dev_section = f"## Metric: {metric_name}\n"
        dev_section += f"- Total Runs: {data['total_runs']}\n"
        dev_section += f"- Regression Count: {data['regression_count']}\n"
        dev_section += f"- Current Streak (L3): {data['current_streak']['status']} for {data['current_streak']['count']} runs\n"
        dev_section += f"- Advanced Analytics:\n"
        for key, value in data['advanced_analytics'].items():
            dev_section += f"  - {key.replace('_', ' ').title()}: {value}\n"
        dev_section += f"- Last 5 Snapshots (most recent first):\n"
        for snapshot in reversed(data['last_n_snapshots'][-5:]):
            dev_section += f"  - {snapshot['timestamp_utc']} (SHA: {snapshot['git_sha'][:7]}): L3 Status: {snapshot['levels']['L3_regression']['status']} (Value: {snapshot['levels']['L3_regression'].get('value', 'N/A')})\n"
        
        developer_report += dev_section + "\n"

        # Director's view logic
        status_color = "GREEN"
        summary_lines = []
        if data["regression_count"] > 0:
            status_color = "YELLOW"
            summary_lines.append(f"  - {metric_name}: L3 regressions detected ({data['regression_count']} total).")
        if data['advanced_analytics']['is_flapping']:
            status_color = "YELLOW"
            summary_lines.append(f"  - {metric_name}: Flapping detected (potential instability).")
        if data['advanced_analytics']['is_regression_outlier']:
            status_color = "RED" # Critical for director
            summary_lines.append(f"  - {metric_name}: OUTLIER REGRESSION DETECTED. IMMEDIATE ATTENTION.")
            overall_status = "RED" # Override to RED if critical

        if status_color == "GREEN" and not summary_lines:
             summary_lines.append(f"  - {metric_name}: All good.")
        
        director_summary += "\n" + "\n".join(summary_lines)

    # Final overall status for director
    director_summary_overall = f"## Overall Metric Health: {overall_status}\n"
    if overall_status == "RED":
        director_summary_overall += "Immediate action required for critical issues.\n"
    elif overall_status == "YELLOW":
        director_summary_overall += "Review warnings and potential areas of concern.\n"
    else:
        director_summary_overall += "All metrics appear healthy according to current policies.\n"

    director_summary = director_summary_overall + director_summary
    
    return developer_report, director_summary


if __name__ == '__main__':
    # (previous main block remains)

    # --- Test Reports Generation ---
    print("\n--- Testing Reports Generation ---")
    
    # Reload policy for reports
    policy_registry_for_reports = load_promotion_policy()

    developer_report, director_summary = generate_reports(timeline, policy_registry_for_reports)
    
    print("\n" + "="*80 + "\nDeveloper Report:\n" + "="*80)
    print(developer_report)
    
    print("\n" + "="*80 + "\nDirector's Summary:\n" + "="*80)
    print(director_summary)

    print("\n--- Reports generation checks complete ---")

