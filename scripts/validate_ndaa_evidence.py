# scripts/validate_ndaa_evidence.py

import argparse
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Tuple

# Add jsonschema to requirements if not present
try:
    import jsonschema
except ImportError:
    print("Error: jsonschema is not installed. Please run 'pip install jsonschema'")
    sys.exit(1)

# --- Globals ---
ASCII_ONLY = False

# --- Color Codes for Output ---
C_RED = "\033[91m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_BLUE = "\033[94m"
C_RESET = "\033[0m"

def print_color(color: str, text: str):
    """Prints text in a given color, respecting ASCII_ONLY mode."""
    if ASCII_ONLY:
        print(text)
    else:
        print(f"{color}{text}{C_RESET}")

def run_command(command: str) -> Tuple[bool, str]:
    """Runs a shell command and returns (success, output)."""
    print_color(C_BLUE, f"  > Running verification command: {command}")
    try:
        process = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        output = process.stdout or ""
        if process.stderr:
            output += f"\nSTDERR:\n{process.stderr}"
        print_color(C_GREEN, "  > Command successful.")
        return True, output
    except subprocess.CalledProcessError as e:
        output = e.stdout or ""
        if e.stderr:
            output += f"\nSTDERR:\n{e.stderr}"
        print_color(C_RED, f"  > Command failed with exit code {e.returncode}.")
        return False, output
    except Exception as e:
        err_msg = f"  > An unexpected error occurred while running command: {e}"
        print_color(C_RED, err_msg)
        return False, err_msg


def validate_schema(schema_path: str, instance_path: str) -> Tuple[bool, str]:
    """Validates a JSON instance against a schema."""
    message = f"Validating {instance_path} against schema {schema_path}"
    print_color(C_BLUE, f"  > {message}")
    try:
        with open(schema_path, 'r') as f:
            schema = json.load(f)
        with open(instance_path, 'r') as f:
            instance = json.load(f)
        jsonschema.validate(instance=instance, schema=schema)
        success_msg = "Schema validation successful."
        print_color(C_GREEN, f"  > {success_msg}")
        return True, success_msg
    except FileNotFoundError as e:
        err_msg = f"File not found: {e.filename}"
        print_color(C_RED, f"  > {err_msg}")
        return False, err_msg
    except json.JSONDecodeError as e:
        err_msg = f"Invalid JSON in file: {e}"
        print_color(C_RED, f"  > {err_msg}")
        return False, err_msg
    except jsonschema.ValidationError as e:
        err_msg = f"Schema validation failed: {e.message}"
        print_color(C_RED, f"  > {err_msg}")
        return False, err_msg
    except Exception as e:
        err_msg = f"An unexpected error occurred during schema validation: {e}"
        print_color(C_RED, f"  > {err_msg}")
        return False, err_msg

def check_artifact_exists(base_dir: str, artifact_path: str) -> Tuple[bool, str]:
    """Checks if an artifact exists in the evidence directory."""
    full_path = os.path.join(base_dir, artifact_path)
    message = f"Checking for artifact: {full_path}"
    print_color(C_BLUE, f"  > {message}")
    if os.path.exists(full_path):
        success_msg = "Artifact found."
        print_color(C_GREEN, f"  > {success_msg}")
        return True, success_msg
    else:
        err_msg = "Artifact NOT found."
        print_color(C_RED, f"  > {err_msg}")
        return False, err_msg

def validate_checklist(checklist_path: str, base_dir: str, report_path: str = None):
    """
    Validates an evidence pack against a machine-readable checklist.
    """
    print_color(C_YELLOW, f"--- Starting NDAA Evidence Validation ---")
    print(f"Checklist: {checklist_path}")
    print(f"Evidence Base Directory: {base_dir}\n")

    report = {
        "summary": {
            "status": "FAILURE",
            "total_steps": 0,
            "passed_steps": 0,
            "failed_steps": 0
        },
        "results": []
    }

    try:
        with open(checklist_path, 'r') as f:
            checklist = json.load(f)
    except FileNotFoundError:
        msg = f"FATAL: Checklist file not found at {checklist_path}"
        print_color(C_RED, msg)
        report["summary"]["error"] = msg
        if report_path:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
        sys.exit(1)
    except json.JSONDecodeError:
        msg = f"FATAL: Could not parse JSON checklist at {checklist_path}"
        print_color(C_RED, msg)
        report["summary"]["error"] = msg
        if report_path:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
        sys.exit(1)

    for pillar in checklist.get("pillars", []):
        pillar_name = pillar.get('pillar_name', 'Unknown Pillar')
        print_color(C_YELLOW, f"\n--- Pillar: {pillar_name} ---")

        for step in pillar.get("steps", []):
            report["summary"]["total_steps"] += 1
            step_id = step.get('step_id', 'N/A')
            action = step.get('action', 'N/A')
            print(f"\n{C_BLUE}Step {step_id}: {action}{C_RESET}")

            step_result = {
                "step_id": step_id,
                "action": action,
                "pillar": pillar_name,
                "status": "FAILURE",
                "checks": []
            }

            # 1. Check for expected artifact
            artifact_path = step.get("expected_output", {}).get("artifact_path")
            if not artifact_path:
                msg = "MISCONFIG: No artifact_path in checklist for this step."
                print_color(C_RED, f"  > {msg}")
                step_result["checks"].append({"check": "artifact_existence", "status": "FAILURE", "details": msg})
            elif artifact_path.startswith("_test_output_"):
                msg = f"Skipping existence check for test-only artifact: {artifact_path}"
                print_color(C_YELLOW, f"  > {msg}")
                step_result["checks"].append({"check": "artifact_existence", "status": "PASSED", "details": msg})
            else:
                exists, msg = check_artifact_exists(base_dir, artifact_path)
                status = "PASSED" if exists else "FAILURE"
                step_result["checks"].append({"check": "artifact_existence", "status": status, "details": msg})

            # 2. Run verification command if artifact exists
            if all(check["status"] == "PASSED" for check in step_result["checks"]):
                command = step.get("verification", {}).get("command")
                if not command:
                    msg = "MISCONFIG: No verification command in checklist for this step."
                    print_color(C_RED, f"  > {msg}")
                    step_result["checks"].append({"check": "verification_command", "status": "FAILURE", "details": msg})
                else:
                    full_artifact_path = os.path.join(base_dir, artifact_path) if artifact_path else ""
                    formatted_command = command.replace("{artifact_path}", full_artifact_path)
                    success, output = run_command(formatted_command)
                    status = "PASSED" if success else "FAILURE"
                    step_result["checks"].append({"check": "verification_command", "status": status, "details": output})

            # Determine final step status
            if all(check["status"] == "PASSED" for check in step_result["checks"]):
                step_result["status"] = "PASSED"
                report["summary"]["passed_steps"] += 1
                print_color(C_GREEN, f"Step {step_id} [PASS]")
            else:
                report["summary"]["failed_steps"] += 1
                print_color(C_RED, f"Step {step_id} [FAIL]")

            report["results"].append(step_result)

    summary = report["summary"]
    summary["failed_steps"] = summary["total_steps"] - summary["passed_steps"]
    
    print_color(C_YELLOW, f"\n--- Validation Summary ---")
    print(f"Total Steps: {summary['total_steps']}")
    print(f"Steps Passed: {summary['passed_steps']}")
    print(f"Steps FAILED: {summary['failed_steps']}")

    if summary["passed_steps"] == summary["total_steps"]:
        summary["status"] = "SUCCESS"
        success_msg = "\nSUCCESS: All evidence checklist items passed."
        print_color(C_GREEN, success_msg if not ASCII_ONLY else "[+] SUCCESS: All evidence checklist items passed.")
        exit_code = 0
    else:
        summary["status"] = "FAILURE"
        fail_msg = "\nFAILURE: One or more evidence checklist items failed."
        print_color(C_RED, fail_msg if not ASCII_ONLY else "[-] FAILURE: One or more evidence checklist items failed.")
        exit_code = 1

    if report_path:
        print(f"\nWriting machine-readable report to {report_path}")
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
        except Exception as e:
            print_color(C_RED, f"Error writing report file: {e}")

    sys.exit(exit_code)

def main():
    global ASCII_ONLY
    parser = argparse.ArgumentParser(
        description="Validate an NDAA evidence pack against a machine-readable checklist.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--pack-dir",
        help="Path to the root directory of the generated evidence pack."
    )
    group.add_argument(
        "--manifest",
        help="Path to a manifest file. Artifact paths will be inferred relative to this file's directory."
    )
    parser.add_argument(
        "--checklist",
        default="docs/system_law/ndaa_evidence_checklist.json",
        help="Path to the machine-readable checklist JSON file."
    )
    parser.add_argument(
        "--json-report-out",
        help="Path to write a machine-readable JSON report."
    )
    parser.add_argument(
        "--ascii-only",
        action="store_true",
        help="Disable color codes and Unicode characters for Windows-safe/CI-friendly output."
    )

    args = parser.parse_args()

    if args.ascii_only:
        ASCII_ONLY = True

    base_dir = ""
    if args.pack_dir:
        base_dir = args.pack_dir
    elif args.manifest:
        base_dir = os.path.dirname(os.path.abspath(args.manifest))

    validate_checklist(args.checklist, base_dir, args.json_report_out)

if __name__ == "__main__":
    main()