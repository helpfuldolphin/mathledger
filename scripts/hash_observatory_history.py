#!/usr/bin/env python3
# scripts/hash_observatory_history.py
import subprocess
import json
import sys
from pathlib import Path
from datetime import datetime, timezone
import argparse

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Builds a historical record of slice integrity audits.")
    parser.add_argument("--auditor-script", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent.parent)
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    project_root = args.project_root
    auditor_script = args.auditor_script or project_root / "scripts" / "hash_reconciliation_auditor.py"
    history_log_path = project_root / "artifacts" / "hash_observatory" / "history.jsonl"
    
    config_path = project_root / "config" / "curriculum_uplift_phase2_hashed.yaml"
    prereg_path = project_root / "PREREG_UPLIFT_U2.yaml"
    manifest_path = project_root / "execution_manifest.json"
    
    history_log_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not all([p.exists() for p in [config_path, prereg_path, manifest_path]]):
        print("WARN: Skipping history run, input artifacts not found.", file=sys.stderr)
        sys.exit(0)

    result = subprocess.run([
        sys.executable, str(auditor_script),
        "--config", str(config_path),
        "--prereg", str(prereg_path),
        "--manifest", str(manifest_path),
        "--integrity-only"
    ], capture_output=True, text=True, check=False)
    
    exit_code = result.returncode
    summary_json = {}
    
    try:
        summary_json = json.loads(result.stdout) if exit_code in [0, 1] and result.stdout else {}
    except json.JSONDecodeError:
        exit_code = 2
        summary_json = {"error": "Invalid JSON from auditor"}

    history_record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "audit_summary": summary_json,
        "auditor_exit_code": exit_code,
    }
    
    with open(history_log_path, "a") as f:
        f.write(json.dumps(history_record) + "\n")
    print(f"Successfully appended record to '{history_log_path}'")

if __name__ == "__main__":
    main()
