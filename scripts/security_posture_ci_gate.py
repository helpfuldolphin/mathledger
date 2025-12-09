# scripts/security_posture_ci_gate.py
"""
GEMINI-K: SIGMA-III - CI Security Gate

This script serves as the primary CI entry point for determinism governance.
It performs the following actions:
1. Runs the canonical posture check script (`security_posture_check.py`).
2. Classifies the resulting posture into a known scenario.
3. Maps the scenario to a severity level and a recommended action.
4. Writes a consolidated `security_governance_snapshot.json` artifact.
5. Exits with a code corresponding to the severity level, allowing CI
   workflows to fail, warn, or pass accordingly.

Exit Codes:
- 0: Severity is OK.
- 1: Severity is ATTENTION.
- 2: Severity is CRITICAL.
"""

import sys
import subprocess
import json
from pathlib import Path
from typing import Dict, Any

from backend.security.posture import classify_security_scenario, SecurityPosture

# --- Governance Mapping ---

# This mapping defines the governance response to each security scenario.
SCENARIO_GOVERNANCE_MAP: Dict[str, Dict[str, Any]] = {
    "HEALTHY_GREEN": {
        "severity": "OK",
        "recommended_action": "No action needed. Determinism posture is nominal.",
        "exit_code": 0,
    },
    "VALIDATION_FAIL_AMBER": {
        "severity": "ATTENTION",
        "recommended_action": "Investigate last-mile validation failures. Core determinism is intact, but downstream consumers may be affected.",
        "exit_code": 1,
    },
    "REPLAY_FAILURE_RED": {
        "severity": "CRITICAL",
        "recommended_action": "BLOCK MERGE. A core determinism invariant has failed (replay). Investigate root cause immediately.",
        "exit_code": 2,
    },
    "SEED_DRIFT_RED": {
        "severity": "CRITICAL",
        "recommended_action": "BLOCK MERGE. A core determinism invariant has failed (seed drift). Investigate PRNG or manifest pipeline.",
        "exit_code": 2,
    },
    "unknown": {
        "severity": "CRITICAL",
        "recommended_action": "BLOCK MERGE. The security posture does not match any known scenario, indicating a novel or unexpected failure mode.",
        "exit_code": 2,
    },
}

def main():
    """Main entry point for the CI gate."""
    print("--- GEMINI-K: Executing Security Posture CI Gate ---")

    # 1. Run the posture check script to get the canonical posture.
    # We pass through the command-line arguments to allow the workflow
    # to define the posture being checked.
    try:
        check_command = [sys.executable, "scripts/security_posture_check.py", "--json-output"] + sys.argv[1:]
        result = subprocess.run(
            check_command,
            capture_output=True,
            text=True,
            check=True,
        )
        posture_summary = json.loads(result.stdout)
        posture: SecurityPosture = posture_summary["components"]
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
        print(f"::error::Failed to retrieve a valid posture from security_posture_check.py: {e}", file=sys.stderr)
        if hasattr(e, 'stderr'):
            print(e.stderr, file=sys.stderr)
        sys.exit(2) # Fail critically if the check itself fails

    # 2. Classify the posture into a scenario.
    scenario_id = classify_security_scenario(posture)
    print(f"Posture classified as scenario: {scenario_id}")

    # 3. Get the governance policy for the scenario.
    governance = SCENARIO_GOVERNANCE_MAP.get(scenario_id, SCENARIO_GOVERNANCE_MAP["unknown"])
    severity = governance["severity"]
    print(f"Scenario severity: {severity}")

    # 4. Create the consolidated snapshot artifact.
    snapshot = {
        "scenario_id": scenario_id,
        "severity": severity,
        "recommended_action": governance["recommended_action"],
        "posture_summary": posture_summary,
    }
    
    snapshot_path = Path("security_governance_snapshot.json")
    with open(snapshot_path, 'w') as f:
        json.dump(snapshot, f, indent=2)
    print(f"Governance snapshot written to {snapshot_path}")

    # 5. Exit with the appropriate code.
    final_exit_code = governance["exit_code"]
    if final_exit_code == 1:
        print(f"::warning::Security posture requires ATTENTION. {governance['recommended_action']}")
    elif final_exit_code == 2:
        print(f"::error::Security posture is CRITICAL. {governance['recommended_action']}")

    sys.exit(final_exit_code)

if __name__ == "__main__":
    main()
