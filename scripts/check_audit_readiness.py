#!/usr/bin/env python3
"""
check_audit_readiness.py - U2 Audit Pre-flight Check

PHASE II — NOT USED IN PHASE I

This script runs a pre-flight check to determine if the system is ready for a
full U2 DAG audit. It checks for the existence of required data sources and
system prerequisites, generating a JSON report of its findings.
"""

import argparse
import json
import psycopg2
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

REQUIRED_DATA_SOURCES = {
    "tables": [
        "u2_experiments", "u2_statements", "u2_proof_parents",
        "u2_goal_attributions", "u2_dag_snapshots", "statements", "proof_parents"
    ],
    "files": ["manifest.json", "PREREG_UPLIFT_U2.yaml"]
}

def check_db_tables(conn: Any) -> List[Dict[str, Any]]:
    """Checks for the existence of required tables using information_schema."""
    status_report = []
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public';
            """)
            existing_tables = {row[0] for row in cursor.fetchall()}
    except psycopg2.Error as e:
        # If we can't even query, all table checks fail.
        for table_name in REQUIRED_DATA_SOURCES["tables"]:
            status_report.append({
                "source_name": table_name,
                "status": "ACCESS_DENIED",
                "details": f"Database query failed: {e}"
            })
        return status_report

    for table_name in REQUIRED_DATA_SOURCES["tables"]:
        status = "FOUND" if table_name in existing_tables else "NOT_FOUND"
        status_report.append({
            "source_name": table_name,
            "status": status,
            "details": f"Database table '{table_name}'"
        })
    return status_report

def check_fs_files(base_dir: Path) -> List[Dict[str, Any]]:
    """Checks for the existence of required files."""
    status_report = []
    for file_name in REQUIRED_DATA_SOURCES["files"]:
        file_path = base_dir / file_name
        status = "FOUND" if file_path.exists() else "NOT_FOUND"
        status_report.append({
            "source_name": file_name,
            "status": status,
            "details": f"Required file at '{file_path}'"
        })
    return status_report

def check_prerequisites(conn: Any) -> List[Dict[str, Any]]:
    """Checks system-level prerequisites like DB extensions."""
    # This is a placeholder for more complex checks.
    prereq_report = []
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM pg_extension WHERE extname = 'pg_trgm';")
            trgm_exists = cursor.fetchone()[0] > 0
            prereq_report.append({
                "check_name": "DB_EXTENSION_PG_TRGM",
                "status": "MET" if trgm_exists else "NOT_MET",
                "expected_value": "installed",
                "actual_value": "installed" if trgm_exists else "not installed"
            })
    except psycopg2.Error as e:
        prereq_report.append({
            "check_name": "DB_EXTENSION_PG_TRGM",
            "status": "UNKNOWN",
            "details": f"Database query failed: {e}"
        })
    return prereq_report

def generate_completeness_report(
    exp_id: str, 
    db_conn: Optional[Any], 
    log_dir: Path
) -> Dict[str, Any]:
    """
    Generates the completeness report by running all readiness checks.
    Accepts an active DB connection to allow for testing.
    """
    if not db_conn:
        data_source_status = [
            {"source_name": t, "status": "ACCESS_DENIED", "details": "No DB connection"}
            for t in REQUIRED_DATA_SOURCES["tables"]
        ]
        prerequisite_status = [
             {"check_name": "DB_EXTENSION_PG_TRGM", "status": "UNKNOWN", "details": "No DB connection"}
        ]
    else:
        data_source_status = check_db_tables(db_conn)
        prerequisite_status = check_prerequisites(db_conn)

    data_source_status.extend(check_fs_files(log_dir))

    is_ready = all(s["status"] == "FOUND" for s in data_source_status) and \
               all(p["status"] == "MET" for p in prerequisite_status)
    completeness_status = "READY" if is_ready else "INCOMPLETE"

    report = {
        "report_metadata": {
            "experiment_id": exp_id,
            "timestamp": datetime.now().isoformat(),
        },
        "completeness_status": completeness_status,
        "data_source_status": data_source_status,
        "prerequisite_checks": prerequisite_status,
        "invariant_coverage": {
            "total_invariants_in_spec": 11,
            "invariants_to_be_run": REQUIRED_DATA_SOURCES["tables"] if is_ready else [],
            "invariants_skipped_due_to_missing_data": [] if is_ready else ["ALL"]
        }
    }
    return report

def main():
    parser = argparse.ArgumentParser(description="U2 Audit Pre-flight Check")
    parser.add_argument('--exp-id', required=True, help="Experiment ID to check readiness for.")
    parser.add_argument('--db-url', required=True, help="Database connection string.")
    parser.add_argument('--log-dir', required=True, help="Directory containing logs and manifests.")
    parser.add_argument('--output', required=True, help="Path to write the audit_completeness_report.json.")
    args = parser.parse_args()

    db_conn = None
    try:
        db_conn = psycopg2.connect(args.db_url)
        report = generate_completeness_report(args.exp_id, db_conn, Path(args.log_dir))
    except psycopg2.Error as e:
        print(f"FATAL: Could not connect to database at '{args.db_url}'. Error: {e}", file=sys.stderr)
        # Generate a failure report even if we can't connect
        report = generate_completeness_report(args.exp_id, None, Path(args.log_dir))
    finally:
        if db_conn:
            db_conn.close()

    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Audit completeness report written to {args.output}")
    if report["completeness_status"] != "READY":
        sys.exit(1)

if __name__ == "__main__":
    main()