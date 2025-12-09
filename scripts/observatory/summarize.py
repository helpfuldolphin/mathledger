# scripts/observatory/summarize.py
"""
Provides high-level summary functions for governance and global health dashboards.
"""
import json
from pathlib import Path
from typing import List, Dict, Any

def summarize_hash_observatory_for_global_health(
    history_log_path: Path,
    recent_run_count: int = 10
) -> Dict[str, Any]:
    """
    Consumes recent history from the observatory log and emits a global health tile.
    """
    if not history_log_path.exists():
        return {
            "hash_observatory_ok": False,
            "status": "BLOCK",
            "recent_failures": 0,
            "drift_signatures": ["HISTORY_LOG_NOT_FOUND"],
            "headline": "Observatory history log not found. Audits may not be running."
        }

    try:
        with open(history_log_path, "r") as f:
            # Read the last N lines for "recent" history
            lines = f.readlines()
            recent_lines = lines[-recent_run_count:]
            history_records = [json.loads(line) for line in recent_lines]
    except (IOError, json.JSONDecodeError) as e:
        return {
            "hash_observatory_ok": False,
            "status": "BLOCK",
            "recent_failures": 0,
            "drift_signatures": ["HISTORY_LOG_CORRUPT"],
            "headline": f"Observatory history log is corrupt or unreadable: {e}"
        }

    if not history_records:
        return {
            "hash_observatory_ok": False,
            "status": "WARN",
            "recent_failures": 0,
            "drift_signatures": ["NO_RECENT_AUDITS"],
            "headline": "No recent audit records found in history log."
        }

    recent_failures = sum(1 for rec in history_records if rec.get("auditor_exit_code") != 0)
    
    # Check the status of the most recent run
    last_run = history_records[-1]
    last_status = last_run.get("audit_summary", {}).get("status", "FAIL")

    status = "OK"
    if recent_failures > 0:
        # If there are any failures in the recent window, status is at least WARN
        status = "WARN"
    if last_status == "FAIL":
        # If the *very last* run failed, the status is more severe
        status = "BLOCK"

    # Collect unique drift signatures from failed runs
    drift_signatures = set()
    for rec in history_records:
        if rec.get("auditor_exit_code") != 0:
            summary = rec.get("audit_summary", {})
            for slice_name in summary.get("drift_slices", []):
                drift_signatures.add(slice_name)

    headline = "All systems nominal."
    if status == "WARN":
        headline = f"{recent_failures} drift events detected in the last {len(history_records)} runs."
    if status == "BLOCK":
        headline = f"Critical drift detected in last audit run. Failing slices: {', '.join(sorted(list(drift_signatures)))}."

    return {
        "hash_observatory_ok": recent_failures == 0,
        "status": status,
        "recent_failures": recent_failures,
        "drift_signatures": sorted(list(drift_signatures)),
        "headline": headline
    }
