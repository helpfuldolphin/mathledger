#!/usr/bin/env python3
"""
Determinism Ledger Engine
==========================

JSON-first audit engine that records determinism test results, tracks
reproducibility hashes, and generates long-horizon trend reports.

Usage:
    # Record a determinism audit run
    python tools/determinism_ledger.py record \\
        --report determinism_report.json \\
        --python-version 3.11 \\
        --commit abc123 \\
        --pr 42

    # Generate trend report
    python tools/determinism_ledger.py trend \\
        --output trend_report.json \\
        --lookback-days 90

    # Query ledger
    python tools/determinism_ledger.py query \\
        --commit abc123

Ledger Format:
    The ledger is stored as a JSONL file at `artifacts/determinism_ledger.jsonl`.
    Each line is a JSON object representing a single audit run.
"""

import argparse
import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict


LEDGER_PATH = Path("artifacts/determinism_ledger.jsonl")


@dataclass
class DeterminismAuditRecord:
    """A single determinism audit record."""
    timestamp: str
    commit: str
    python_version: str
    reproducibility_hash: str
    drift_count: int
    offending_modules: List[str]
    pr_number: Optional[int] = None
    release_tag: Optional[str] = None
    scheduled: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DeterminismAuditRecord':
        """Create from dictionary."""
        return cls(**data)


def ensure_ledger_exists():
    """Ensure the ledger file and directory exist."""
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not LEDGER_PATH.exists():
        LEDGER_PATH.touch()


def append_to_ledger(record: DeterminismAuditRecord):
    """Append a record to the ledger (idempotent)."""
    ensure_ledger_exists()
    
    # Check if record already exists (by commit + python_version)
    existing_records = read_ledger()
    for existing in existing_records:
        if (existing.commit == record.commit and 
            existing.python_version == record.python_version):
            print(f"Record already exists for commit {record.commit} (Python {record.python_version})")
            return
    
    # Append new record
    with open(LEDGER_PATH, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record.to_dict(), sort_keys=True) + '\n')
    
    print(f"✓ Recorded audit for commit {record.commit} (Python {record.python_version})")


def read_ledger() -> List[DeterminismAuditRecord]:
    """Read all records from the ledger."""
    ensure_ledger_exists()
    
    records = []
    with open(LEDGER_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    records.append(DeterminismAuditRecord.from_dict(data))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping malformed line: {e}")
    
    return records


def record_audit(
    report_path: Path,
    python_version: str,
    commit: str,
    pr_number: Optional[int] = None,
    release_tag: Optional[str] = None,
    scheduled: bool = False,
):
    """
    Record a determinism audit run to the ledger.
    
    Args:
        report_path: Path to determinism_report.json
        python_version: Python version (e.g., "3.11")
        commit: Git commit SHA
        pr_number: Optional PR number
        release_tag: Optional release tag
        scheduled: Whether this is a scheduled audit
    """
    # Load the determinism report
    with open(report_path, 'r', encoding='utf-8') as f:
        report = json.load(f)
    
    # Extract key fields
    reproducibility_hash = report.get('reproducibility_hash', '')
    drift_vectors = report.get('drift_vectors', [])
    offending_modules = report.get('offending_modules', [])
    
    # Create audit record
    record = DeterminismAuditRecord(
        timestamp=datetime.utcnow().isoformat() + 'Z',
        commit=commit,
        python_version=python_version,
        reproducibility_hash=reproducibility_hash,
        drift_count=len(drift_vectors),
        offending_modules=offending_modules,
        pr_number=pr_number,
        release_tag=release_tag,
        scheduled=scheduled,
    )
    
    # Append to ledger
    append_to_ledger(record)
    
    # Check for violations
    if record.drift_count > 0:
        print(f"⚠️  WARNING: {record.drift_count} drift vectors detected")
        print(f"   Offending modules: {', '.join(offending_modules)}")
        return 1
    else:
        print(f"✅ Zero drift verified")
        return 0


def generate_trend_report(
    output_path: Path,
    lookback_days: int = 90,
):
    """
    Generate a trend report analyzing determinism over time.
    
    Args:
        output_path: Path to write the trend report JSON
        lookback_days: Number of days to look back
    """
    records = read_ledger()
    
    # Filter to lookback window
    cutoff = datetime.utcnow() - timedelta(days=lookback_days)
    recent_records = [
        r for r in records
        if datetime.fromisoformat(r.timestamp.replace('Z', '')) >= cutoff
    ]
    
    # Group by commit
    by_commit: Dict[str, List[DeterminismAuditRecord]] = {}
    for record in recent_records:
        if record.commit not in by_commit:
            by_commit[record.commit] = []
        by_commit[record.commit].append(record)
    
    # Analyze trends
    total_audits = len(recent_records)
    audits_with_drift = sum(1 for r in recent_records if r.drift_count > 0)
    drift_rate = audits_with_drift / max(1, total_audits)
    
    # Identify most problematic modules
    module_drift_counts: Dict[str, int] = {}
    for record in recent_records:
        for module in record.offending_modules:
            module_drift_counts[module] = module_drift_counts.get(module, 0) + 1
    
    top_offenders = sorted(
        module_drift_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]
    
    # Cross-version analysis
    by_version: Dict[str, List[DeterminismAuditRecord]] = {}
    for record in recent_records:
        if record.python_version not in by_version:
            by_version[record.python_version] = []
        by_version[record.python_version].append(record)
    
    version_drift_rates = {
        version: sum(1 for r in records if r.drift_count > 0) / max(1, len(records))
        for version, records in by_version.items()
    }
    
    # Build trend report
    trend_report = {
        "generated_at": datetime.utcnow().isoformat() + 'Z',
        "lookback_days": lookback_days,
        "summary": {
            "total_audits": total_audits,
            "audits_with_drift": audits_with_drift,
            "drift_rate": drift_rate,
            "unique_commits": len(by_commit),
        },
        "top_offending_modules": [
            {"module": module, "drift_count": count}
            for module, count in top_offenders
        ],
        "by_python_version": {
            version: {
                "total_audits": len(records),
                "drift_rate": version_drift_rates[version],
            }
            for version, records in by_version.items()
        },
        "recent_audits": [
            r.to_dict() for r in sorted(
                recent_records,
                key=lambda x: x.timestamp,
                reverse=True
            )[:20]
        ],
    }
    
    # Write report
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(trend_report, f, indent=2, sort_keys=True)
    
    print(f"✓ Trend report written to {output_path}")
    print(f"  Total audits: {total_audits}")
    print(f"  Drift rate: {drift_rate:.2%}")
    print(f"  Top offender: {top_offenders[0][0] if top_offenders else 'None'}")


def query_ledger(
    commit: Optional[str] = None,
    pr_number: Optional[int] = None,
    release_tag: Optional[str] = None,
):
    """
    Query the ledger for specific records.
    
    Args:
        commit: Filter by commit SHA
        pr_number: Filter by PR number
        release_tag: Filter by release tag
    """
    records = read_ledger()
    
    # Apply filters
    if commit:
        records = [r for r in records if r.commit == commit]
    if pr_number is not None:
        records = [r for r in records if r.pr_number == pr_number]
    if release_tag:
        records = [r for r in records if r.release_tag == release_tag]
    
    # Display results
    if not records:
        print("No matching records found")
        return
    
    print(f"Found {len(records)} matching record(s):")
    print()
    
    for record in records:
        print(f"Commit: {record.commit}")
        print(f"  Python: {record.python_version}")
        print(f"  Timestamp: {record.timestamp}")
        print(f"  Reproducibility Hash: {record.reproducibility_hash}")
        print(f"  Drift Count: {record.drift_count}")
        if record.offending_modules:
            print(f"  Offending Modules: {', '.join(record.offending_modules)}")
        if record.pr_number:
            print(f"  PR: #{record.pr_number}")
        if record.release_tag:
            print(f"  Release: {record.release_tag}")
        print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Determinism Ledger Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Record command
    record_parser = subparsers.add_parser('record', help='Record a determinism audit')
    record_parser.add_argument('--report', type=Path, required=True,
                               help='Path to determinism_report.json')
    record_parser.add_argument('--python-version', required=True,
                               help='Python version (e.g., 3.11)')
    record_parser.add_argument('--commit', required=True,
                               help='Git commit SHA')
    record_parser.add_argument('--pr', type=int,
                               help='PR number')
    record_parser.add_argument('--release', type=str,
                               help='Release tag')
    record_parser.add_argument('--scheduled', action='store_true',
                               help='Mark as scheduled audit')
    
    # Trend command
    trend_parser = subparsers.add_parser('trend', help='Generate trend report')
    trend_parser.add_argument('--output', type=Path, required=True,
                              help='Output path for trend report')
    trend_parser.add_argument('--lookback-days', type=int, default=90,
                              help='Number of days to look back (default: 90)')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query the ledger')
    query_parser.add_argument('--commit', type=str,
                              help='Filter by commit SHA')
    query_parser.add_argument('--pr', type=int,
                              help='Filter by PR number')
    query_parser.add_argument('--release', type=str,
                              help='Filter by release tag')
    
    args = parser.parse_args()
    
    if args.command == 'record':
        exit_code = record_audit(
            report_path=args.report,
            python_version=args.python_version,
            commit=args.commit,
            pr_number=args.pr,
            release_tag=args.release,
            scheduled=args.scheduled,
        )
        return exit_code
    
    elif args.command == 'trend':
        generate_trend_report(
            output_path=args.output,
            lookback_days=args.lookback_days,
        )
        return 0
    
    elif args.command == 'query':
        query_ledger(
            commit=args.commit,
            pr_number=args.pr,
            release_tag=args.release,
        )
        return 0
    
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    exit(main())
