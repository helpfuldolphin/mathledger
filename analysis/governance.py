# Copyright 2025 MathLedger
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Uplift Governance and Cross-Run Analysis Module (governance.py)
===============================================================

This module provides higher-level functions for interpreting the output of the
conjecture engine in the context of project governance and historical trends.

Author: Gemini M, Dynamics-Theory Unification Analyst
"""

from typing import Any, Dict, List, Sequence, Tuple

# --- Task 3: Global Health Simplified Signal ---

def summarize_conjecture_status_for_global_health(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """
    Distills a single conjecture report snapshot into a simplified, high-level
    "learning health" signal.

    Args:
        snapshot: A dictionary representing a single conjecture_report.json.

    Returns:
        A dictionary containing the learning health summary.
    """
    statuses = [v['status'] for k, v in snapshot.items() 
                if k.startswith("Conjecture") or k == "Phase II Uplift"]
    
    if not statuses:
        return {
            "learning_health": "INCONCLUSIVE",
            "supports_vs_contradicts_ratio": 0,
            "any_key_conjecture_contradicted": False,
            "details": "No conjecture statuses found in snapshot."
        }

    num_supports = statuses.count("SUPPORTS")
    num_contradicts = statuses.count("CONTRADICTS")
    num_inconclusive = statuses.count("INCONCLUSIVE")
    
    # Define "key" conjectures whose failure is particularly alarming
    key_conjectures = ["Phase II Uplift", "Conjecture 3.1"]
    any_key_conjecture_contradicted = any(
        snapshot.get(c, {}).get("status") == "CONTRADICTS" for c in key_conjectures
    )

    # Determine learning_health
    if any_key_conjecture_contradicted or num_contradicts > num_supports:
        learning_health = "UNHEALTHY"
    elif num_supports > 0 and num_contradicts == 0:
        learning_health = "HEALTHY"
    elif num_supports > 0:
        learning_health = "MIXED"
    else:
        learning_health = "INCONCLUSIVE"
        
    ratio = num_supports / num_contradicts if num_contradicts > 0 else float('inf')

    return {
        "learning_health": learning_health,
        "supports_vs_contradicts_ratio": ratio,
        "any_key_conjecture_contradicted": any_key_conjecture_contradicted
    }

# --- Task 1: Uplift Governance Integration Helper ---

def _summarize_conjectures_for_governance(dynamics_summary: Dict[str, Any]) -> str:
    """Helper to produce a simple OK/WARN/ATTENTION status."""
    health_summary = summarize_conjecture_status_for_global_health(dynamics_summary)
    
    if health_summary["learning_health"] == "UNHEALTHY":
        return "ATTENTION"
    if health_summary["learning_health"] == "MIXED" or health_summary["learning_health"] == "INCONCLUSIVE":
        return "WARN"
    return "OK"

def combine_conjectures_with_governance(governance_posture: Dict, dynamics_summary: Dict) -> Dict[str, Any]:
    """
    Merges a dynamics analysis summary with a governance posture document
    to produce a final, actionable readiness assessment.

    Args:
        governance_posture: A dictionary describing the current project governance state.
        dynamics_summary: A dictionary representing a single conjecture_report.json.

    Returns:
        A dictionary containing the combined governance and dynamics status.
    """
    dynamics_gov_status = _summarize_conjectures_for_governance(dynamics_summary)
    
    # Rule: Governance is blocking if any manual gate is not 'passed'.
    governance_blocking = any(
        gate_status != 'passed' for gate_status in governance_posture.get("gates", {}).values()
    )

    # Combine statuses. The worse of the two statuses takes precedence.
    status_priority = {"ATTENTION": 2, "WARN": 1, "OK": 0}
    gov_posture_status = governance_posture.get("status", "OK")
    
    if status_priority[dynamics_gov_status] > status_priority[gov_posture_status]:
        final_dynamics_status = dynamics_gov_status
    else:
        final_dynamics_status = gov_posture_status

    # Rule: Uplift is "ready" only if nothing is blocking and dynamics status is OK.
    uplift_readiness_flag = (
        not governance_blocking 
        and final_dynamics_status == "OK"
    )

    return {
        "governance_blocking": governance_blocking,
        "dynamics_status": final_dynamics_status,
        "uplift_readiness_flag": uplift_readiness_flag,
        "components": {
            "governance_posture": governance_posture,
            "dynamics_summary": dynamics_gov_status,
        }
    }


def summarize_policy_identity_for_global_health(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze policy identity across runtime rows, flagging conflicting hashes/versions per policy.

    Args:
        rows: Sequence of dictionaries containing at least policy_name/policy_hash/policy_version.

    Returns:
        Summary dict with status and per-policy issue list.
    """
    if not rows:
        return {
            "status": "UNKNOWN",
            "policy_count": 0,
            "issues": [],
            "details": "No policy rows supplied.",
        }

    def _logical_key(row: Dict[str, Any]) -> str:
        return (
            row.get("policy_name")
            or row.get("policy_id")
            or row.get("system")
            or "default"
        )

    issues: List[Dict[str, Any]] = []
    groups: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        key = _logical_key(row)
        group = groups.setdefault(
            key,
            {"hashes": set(), "versions": set(), "rows": []},
        )
        policy_hash = row.get("policy_hash")
        policy_version = row.get("policy_version")
        if policy_hash:
            group["hashes"].add(policy_hash)
        if policy_version:
            group["versions"].add(policy_version)
        group["rows"].append(row)

    for key, group in groups.items():
        hashes = sorted(group["hashes"])
        versions = sorted(group["versions"])
        if len(hashes) > 1 or len(versions) > 1:
            issues.append(
                {
                    "policy": key,
                    "hashes": hashes,
                    "versions": versions,
                    "occurrences": len(group["rows"]),
                }
            )

    status = "OK"
    if issues:
        if any(len(issue["hashes"]) > 1 for issue in issues):
            status = "ATTENTION"
        else:
            status = "WARN"

    return {
        "status": status,
        "policy_count": len(groups),
        "issues": issues,
    }


# --- Task 2: Conjecture History Timeline ---

def build_conjecture_timeline(snapshots: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Builds a historical timeline of conjecture statuses from a sequence of
    conjecture report snapshots.

    Args:
        snapshots: An ordered sequence of dictionaries, where each is a
                   conjecture_report.json.

    Returns:
        A dictionary containing the historical evolution and transition counts.
    """
    if not snapshots:
        return {}

    timeline = {}
    transition_counts = {}

    # Initialize with all conjectures found in the first snapshot
    first_snapshot = snapshots[0]
    all_conjecture_keys = [k for k in first_snapshot.keys() if k.startswith("Conjecture") or k == "Phase II Uplift"]
    
    for key in all_conjecture_keys:
        timeline[key] = []
        transition_counts[key] = {"to_CONTRADICTS": 0}

    # Populate the timeline
    for snapshot in snapshots:
        for key in all_conjecture_keys:
            if key in snapshot and 'status' in snapshot[key]:
                status = snapshot[key]['status']
                
                # Check for transition to CONTRADICTS
                if len(timeline[key]) > 0 and status == "CONTRADICTS" and timeline[key][-1] != "CONTRADICTS":
                    transition_counts[key]["to_CONTRADICTS"] += 1
                
                timeline[key].append(status)
            else:
                # Append a null value if a conjecture is missing in a snapshot
                timeline[key].append(None)
    
    return {
        "conjecture_evolution": timeline,
        "transition_metrics": transition_counts
    }
