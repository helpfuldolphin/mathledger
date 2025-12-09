# scripts/run_telemetry_audit.py

import argparse
import json
import sys
import os
from pathlib import Path

# Ensure experiments and backend modules can be imported
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from experiments.telemetry_consistency_auditor import TelemetryConsistencyAuditor, FileSystemDataLoader

def main():
    """
    Main entry point for running the telemetry consistency audit from the command line.
    """
    parser = argparse.ArgumentParser(description="Run Telemetry Consistency Audit on U2 experiment output.")
    parser.add_argument(
        "source_path",
        type=str,
        help="Path to the experiment output directory containing summary.json and cycles.jsonl."
    )
    args = parser.parse_args()

    print(f"--- Starting Telemetry Consistency Audit for: {args.source_path} ---")

    try:
        # 1. Load data using the pluggable loader
        loader = FileSystemDataLoader()
        cycle_events, summary = loader.load_data(args.source_path)
        print(f"Successfully loaded {len(cycle_events)} cycle events and 1 summary manifest.")

        # 2. Instantiate and run the auditor
        auditor = TelemetryConsistencyAuditor(cycle_events, summary)
        report = auditor.run_audit()

        # 3. Save the report to the specified artifacts directory
        report_path = Path(args.source_path) / "artifacts" / "telemetry"
        report_path.mkdir(parents=True, exist_ok=True)
        report_file = report_path / "telemetry_consistency_report.json"

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, sort_keys=True)
        
        print(f"\nAudit complete. Report generated at: {report_file}")
        
        # 4. Print report to stdout for CI logs
        print("\n--- AUDIT REPORT ---")
        print(json.dumps(report, indent=2, sort_keys=True))
        print("--- END REPORT ---")

        # 5. Determine exit code based on report status
        has_failures = any(check["status"] == "FAILED" for check in report["checks"])
        if has_failures:
            print("\nAUDIT FAILED: One or more consistency checks failed.")
            sys.exit(1)
        else:
            print("\nAUDIT PASSED: All consistency checks passed.")
            sys.exit(0)

    except FileNotFoundError as e:
        print(f"\n[ERROR] Input data not found: {e}", file=sys.stderr)
        sys.exit(2) # Use a different exit code for setup errors
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred during the audit: {e}", file=sys.stderr)
        sys.exit(3)

if __name__ == "__main__":
    main()
