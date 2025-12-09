
"""
Uplift Governance Pipeline Orchestrator (v4)

This script orchestrates the full evidence pipeline, chaining the v2
governance verifier with the v2 dynamics-theory conjecture engine.
"""

import argparse
import json
import hashlib
import sys
from pathlib import Path
import subprocess
import datetime

# Add project root to sys.path to resolve local package imports
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.analytics.governance_verifier import verify_summary_file
from analysis.conjecture_engine import run_conjecture_analysis

def get_git_commit_hash() -> str:
    """Get the current Git commit hash."""
    try:
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT
        ).decode("utf-8").strip()
        return commit_hash
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "N/A"

def _load_raw_logs(log_path: Path) -> list:
    """Helper to load a JSONL file into a list of dicts."""
    records = []
    with log_path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def _archive_artifacts(output_path: Path, *input_paths: Path):
    """Copies the final report and all its inputs to a timestamped archive directory."""
    run_timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_dir = Path("artifacts/governance") / run_timestamp
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy all input files
    for p in input_paths:
        if p and p.exists():
            shutil.copy(p, archive_dir / p.name)
            
    # Copy the final report
    shutil.copy(output_path, archive_dir / output_path.name)
    print(f"Archived artifacts to: {archive_dir}")

def run_pipeline(args: argparse.Namespace) -> bool:
    """
    Runs the full governance pipeline.
    Returns True if the evidence is admissible (PASS or WARN), False otherwise.
    """
    output_path = Path(args.output_path)
    summary_path = Path(args.summary_path)

    # 1. Governance Verification (Hard Gate)
    try:
        verdict = verify_summary_file(
            summary_path=str(summary_path),
            manifest_path=getattr(args, 'manifest_path', None),
            telemetry_path=getattr(args, 'telemetry_path', None),
            prereg_path=getattr(args, 'prereg_path', None),
        )
        print("--- VERDICT ---")
        print(json.dumps(verdict.to_dict(), indent=2))
        print("---------------")
    except Exception as e:
        final_report = {
            "status": "ERROR",
            "error_message": f"Governance verifier crashed: {e}",
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(final_report, f, indent=2)
        return False

    if verdict.status == "FAIL":
        final_report = {
            "status": "FAILED",
            "governance_verdict": verdict.to_dict(),
            "conjecture_report": None,
            "provenance": {"analytics_code_version": get_git_commit_hash()}
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(final_report, f, indent=2)
        return False

    # 2. Load Raw Data (only if governance passes)
    try:
        baseline_records = _load_raw_logs(Path(args.baseline_log_path))
        rfl_records = _load_raw_logs(Path(args.rfl_log_path))
    except Exception as e:
         final_report = {
            "status": "ERROR",
            "error_message": f"Failed to load raw log files: {e}",
            "governance_verdict": verdict.to_dict(),
         }
         output_path.parent.mkdir(parents=True, exist_ok=True)
         with output_path.open("w") as f:
            json.dump(final_report, f, indent=2)
         return False

    # 3. Run Conjecture Engine
    conjecture_params = {
        "slice_uplift_threshold": 0.10,
        "conjectures_to_test": ["Phase II Uplift"],
        "thresholds": {
            'stagnation_std_thresh': 0.01,
            'trend_tau_thresh': -0.2,
            'oscillation_omega_thresh': 0.3,
            'step_size_thresh': 0.1,
        }
    }
    conjecture_report = run_conjecture_analysis(
        baseline_records, rfl_records, **conjecture_params
    )

    # 4. Aggregate Final Report
    final_report = {
        "status": verdict.status,
        "governance_verdict": verdict.to_dict(),
        "conjecture_report": conjecture_report,
        "provenance": {
            "summary_checksum": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
            "analytics_code_version": get_git_commit_hash(),
        }
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(final_report, f, indent=2)
    
    return True

if __name__ == "__main__":
    import shutil
    parser = argparse.ArgumentParser(description="Run the Full Uplift Governance and Conjecture Pipeline.")
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--baseline-log-path", required=True)
    parser.add_argument("--rfl-log-path", required=True)
    parser.add_argument("--manifest-path")
    parser.add_argument("--telemetry-path")
    parser.add_argument("--prereg-path")
    parser.add_argument("--output-path", default="results/uplift_governance_report.json")
    parser.add_argument("--archive", action="store_true", help="Archive artifacts to artifacts/governance/")
    args = parser.parse_args()

    is_admissible = run_pipeline(args)
    
    print(f"Pipeline complete. Report generated at: {args.output_path}")
    
    if args.archive:
        input_paths = [Path(p) for p in [args.summary_path, args.manifest_path, args.telemetry_path, args.prereg_path] if p]
        _archive_artifacts(Path(args.output_path), *input_paths)

    if not is_admissible:
        print("Verdict: FAILED or ERROR. Evidence is not admissible.")
        sys.exit(1)
    else:
        print(f"Verdict: {is_admissible}. Evidence is admissible for review.")
        sys.exit(0)
