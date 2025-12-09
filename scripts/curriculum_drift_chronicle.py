# PHASE IV — NOT USED IN PHASE I
# File: scripts/curriculum_drift_chronicle.py
"""
Chronicle Archiver & CI Job Runner

This script operationalizes the curriculum intelligence engine. It:
1. Scans a directory of drift reports.
2. Generates the high-level chronicle and director panel.
3. Appends key metrics to a historical log.
4. Exits with a status code based on promotion readiness for CI gating.
"""
import argparse
import json
import sys
import os
import glob
from datetime import datetime, timezone

# Ensure the experiments module can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiments.curriculum_linter_v3 import (
    build_curriculum_drift_chronicle,
    evaluate_curriculum_for_promotion,
    build_curriculum_director_panel
)

def summarize_for_global_console(promotion_eval, chronicle):
    """Implements the Global Console Adapter."""
    return {
        "schema_version": "1.0.0",
        "curriculum_ok": promotion_eval.get('curriculum_ready', False),
        "status": promotion_eval.get('status', 'UNKNOWN'),
        "headline": promotion_eval.get('rationale', 'Could not determine status.'),
        "slices_with_recurrent_drift": chronicle.get('recurrent_drift_paths', [])
    }

def append_to_history(history_file, chronicle, promotion_eval):
    """Appends a summary of the latest evaluation to a JSONL history file."""
    history_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trend": chronicle.get('drift_trend'),
        "status": promotion_eval.get('status'),
        "critical_count": chronicle.get('drift_events_series', [{}])[-1].get('counts', {}).get('CRITICAL', 0),
        "warning_count": chronicle.get('drift_events_series', [{}])[-1].get('counts', {}).get('WARNING', 0),
    }
    with open(history_file, 'a') as f:
        f.write(json.dumps(history_entry) + '\n')

def main():
    parser = argparse.ArgumentParser(description="Curriculum Drift Chronicle Processor")
    parser.add_argument("--reports-glob", required=True, help="Glob pattern for input drift reports (e.g., 'artifacts/drift/*.json')")
    parser.add_argument("--output-panel", default="artifacts/curriculum/curriculum_director_panel.json", help="Output path for the director panel JSON.")
    parser.add_argument("--history-file", default="artifacts/curriculum/drift_chronicle_history.jsonl", help="Path to the JSONL history file to append to.")
    parser.add_argument("--append-history", action="store_true", help="Enable appending to the history file.")
    args = parser.parse_args()

    report_files = glob.glob(args.reports_glob)
    if not report_files:
        print(f"No reports found matching glob: {args.reports_glob}", file=sys.stderr)
        sys.exit(1) # Fail if no reports are found

    reports = []
    for rf in sorted(report_files):
        with open(rf, 'r') as f:
            reports.append(json.load(f))
    
    latest_report = sorted(reports, key=lambda r: r.get('report_generated_utc', ''))[-1]

    # 1. Build Chronicle
    chronicle = build_curriculum_drift_chronicle(reports)
    
    # 2. Evaluate for Promotion
    promotion_eval = evaluate_curriculum_for_promotion(latest_report, chronicle)
    
    # 3. Build Director Panel
    director_panel = build_curriculum_director_panel(latest_report, promotion_eval, chronicle)
    os.makedirs(os.path.dirname(args.output_panel), exist_ok=True)
    with open(args.output_panel, 'w') as f:
        json.dump(director_panel, f, indent=2)
    print(f"[INFO] Director Panel written to {args.output_panel}")

    # 4. Generate Global Console Summary
    console_summary = summarize_for_global_console(promotion_eval, chronicle)
    print("\n--- Global Console Summary ---")
    print(json.dumps(console_summary, indent=2))

    # 5. Append to History if enabled
    if args.append_history:
        os.makedirs(os.path.dirname(args.history_file), exist_ok=True)
        append_to_history(args.history_file, chronicle, promotion_eval)
        print(f"[INFO] Appended current status to {args.history_file}")

    # 6. Exit with status code for CI
    if promotion_eval['status'] == 'BLOCK':
        print("\n[FAIL] Promotion status is BLOCK. Exiting with status 1.", file=sys.stderr)
        sys.exit(1)
    else:
        print("\n[PASS] Promotion status is OK or WARN. Exiting with status 0.")
        sys.exit(0)

if __name__ == "__main__":
    main()
