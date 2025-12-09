# PHASE II — U2 UPLIFT EXPERIMENT
"""
Checks the alignment between the U2 Environment Enforcement Specification
and the implementation in the validator script.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

def parse_spec_file(spec_path):
    """
    Parses the markdown spec file to extract all RULE-IDs and the spec version.
    """
    if not spec_path.exists():
        print(f"ERROR: Specification file not found at '{spec_path}'", file=sys.stderr)
        return None, None
    
    rule_ids = set()
    spec_version = "unknown"
    try:
        content = spec_path.read_text(encoding="utf-8")
        
        # Extract Spec Version
        version_match = re.search(r"\*\*ID:\*\* `(U2_EES_v[\d\.]+)", content)
        if version_match:
            spec_version = version_match.group(1)

        # Extract Rule IDs
        for line in content.splitlines():
            if line.strip().startswith("| **RULE-"):
                match = re.search(r"RULE-(\d{3})", line)
                if match:
                    rule_ids.add(f"RULE-{match.group(1)}")
        
        return rule_ids, spec_version
    except Exception as e:
        print(f"ERROR: Failed to parse spec file '{spec_path}': {e}", file=sys.stderr)
        return None, None

def get_rules_from_code(validator_path):
    """Executes the validator script with --report-covered-rules to get implemented rules."""
    if not validator_path.exists():
        print(f"ERROR: Validator script not found at '{validator_path}'", file=sys.stderr)
        return None
        
    try:
        command = [sys.executable, str(validator_path), "--report-covered-rules"]
        result = subprocess.run(command, capture_output=True, text=True, check=True, encoding="utf-8")
        return set(json.loads(result.stdout))
    except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError) as e:
        print(f"ERROR: Failed to get rules from validator script '{validator_path}': {e}", file=sys.stderr)
        if isinstance(e, subprocess.CalledProcessError):
            print(f"Stderr: {e.stderr}", file=sys.stderr)
        return None

def main():
    """Main execution function."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    spec_path = project_root / "docs" / "U2_ENVIRONMENT_ENFORCEMENT_SPEC.md"
    validator_path = script_dir / "validate_u2_environment.py"
    report_path = project_root / "environment_spec_alignment_report.json"

    print(f"Spec file:      {spec_path}")
    print(f"Validator file: {validator_path}")
    print(f"Report file:    {report_path}")

    spec_rules, spec_version = parse_spec_file(spec_path)
    if spec_rules is None or not spec_rules:
        print(f"ERROR: No rules found in spec file '{spec_path}'. Check parsing logic and file content.", file=sys.stderr)
        sys.exit(1)

    code_rules = get_rules_from_code(validator_path)
    if code_rules is None:
        sys.exit(1)

    rules_in_spec_not_in_code = sorted(list(spec_rules - code_rules))
    rules_in_code_not_in_spec = sorted(list(code_rules - code_rules))
    
    aligned = not rules_in_spec_not_in_code and not rules_in_code_not_in_spec
    status = "PASS" if aligned else "FAIL"
    
    commit_sha = os.environ.get("GITHUB_SHA", "local_run")
    
    print(f"\nAlignment Status: {status}")

    report = {
        "alignment_status": status,
        "spec_version": spec_version,
        "commit_sha": commit_sha,
        "spec_file": str(spec_path),
        "code_file": str(validator_path),
        "total_rules_in_spec": len(spec_rules),
        "total_rules_in_code": len(code_rules),
        "aligned_rules_count": len(spec_rules.intersection(code_rules)),
        "rule_coverage_percentage": (len(spec_rules.intersection(code_rules)) / len(spec_rules)) * 100 if spec_rules else 0,
        "rules_in_spec_not_in_code": rules_in_spec_not_in_code,
        "rules_in_code_not_in_spec": rules_in_code_not_in_spec,
    }

    try:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Alignment report successfully written to '{report_path}'")
    except IOError as e:
        print(f"ERROR: Failed to write report file: {e}", file=sys.stderr)
        sys.exit(1)
        
    if not aligned:
        print("\nMisalignment detected:")
        if rules_in_spec_not_in_code:
            print(f"- Rules in spec but not in code: {rules_in_spec_not_in_code}")
        if rules_in_code_not_in_spec:
            print(f"- Rules in code but not in spec: {rules_in_code_not_in_spec}")
        sys.exit(1)
        
    sys.exit(0)

if __name__ == "__main__":
    main()
