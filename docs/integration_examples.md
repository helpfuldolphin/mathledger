# GEMINI-K: SIGMA-III - Integration Examples
_This document provides examples for integrating the determinism security posture spine into higher-level systems._

---

## Director's Console Integration

This example demonstrates how a "Director's Console" or a global health dashboard can consume the output of the security posture system to drive a top-level "traffic light" status indicator.

The core principle is that the security posture of the U2 compute substrate is a critical input to the overall system health. A breach of determinism (`RED` security level) should immediately signal a critical failure at the global level.

### Pseudo-code Example

This Python pseudo-code illustrates a typical pipeline within a master control program or dashboard backend.

```python
import json
import subprocess
from typing import Dict, Any

# Assume this function is provided by the posture spine module
from backend.security.posture import merge_into_global_health

def run_subprocess_and_get_json(command: list) -> Dict[str, Any]:
    """Runs a command and returns its JSON output."""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        print(f"Error getting posture summary: {e}")
        # Return a failsafe error state
        return {
            "security_level": "RED",
            "is_ok": False,
            "narrative": f"Failed to execute security check: {e}",
            "components": {},
        }

def update_director_console_status():
    """
    Main loop for updating the Director's Console health status.
    """
    # 1. Initialize the global health state for this update cycle.
    # The default state is GREEN.
    global_health = {
        "traffic_light": "GREEN",
        "subsystems": {
            # Other subsystems might already be populated here
            "data_ingest": {"status": "OK"},
            "api_server": {"status": "OK"},
        }
    }

    # 2. Execute the security posture check CLI to get the canonical summary.
    # In a real system, the boolean flags would be determined by running
    # other audit scripts (e.g., the cross-run determinism check).
    print("Fetching security determinism posture...")
    posture_check_command = [
        "python", "scripts/security_posture_check.py",
        "--json-output",
        "--replay-ok", # This would be dynamically determined
        "--seed-pure", # This would be dynamically determined
        "--last-mile-pass", # This would be dynamically determined
    ]
    security_summary = run_subprocess_and_get_json(posture_check_command)

    # 3. Merge the security summary into the global health object.
    # The `merge_into_global_health` function contains the governance logic
    # for how a subsystem's status affects the global status.
    global_health = merge_into_global_health(global_health, security_summary)
    
    # The `merge_into_global_health` function automatically promotes the
    # global traffic_light to AMBER or RED based on the security level,
    # so no extra `if/else` logic is needed here.

    # 4. Render the final dashboard using the consolidated health object.
    print("\n--- DIRECTOR'S CONSOLE FINAL STATE ---")
    print(json.dumps(global_health, indent=2))
    print("------------------------------------")
    # render_dashboard(global_health)

if __name__ == "__main__":
    update_director_console_status()

```

### Key Takeaways

1.  **Command-Line Interface:** The `security_posture_check.py` script serves as a stable, version-controlled interface. The console backend doesn't need to import the Python logic directly; it only needs to execute the script.
2.  **JSON as Data Interchange:** The `--json-output` flag provides a machine-readable format, avoiding the need to parse human-readable text.
3.  **Centralized Governance Logic:** The `merge_into_global_health` function contains the critical business logic (e.g., "a RED security level forces the global status to RED"). This prevents logic from being duplicated or diverging in different consumer applications. The console's only job is to call the function and respect its output.
