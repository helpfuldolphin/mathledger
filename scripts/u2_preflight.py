#!/usr/bin/env python3
"""
U2 Pre-Flight Audit Tool

PHASE II — NOT RUN IN PHASE I

Implements the 25-item pre-flight checklist from U2_PRE_FLIGHT_AUDIT_PLAYBOOK.md.
Determines whether a U2 experiment is eligible for full audit.

Exit Codes:
    0 - ELIGIBLE or ELIGIBLE_WARNED
    1 - BLOCKED_FIXABLE or INADMISSIBLE
    2 - Invalid arguments or runtime error
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

# Conditional imports for database
try:
    import psycopg2
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class CheckStatus(str, Enum):
    """Status of an individual check."""
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"


class EligibilityStatus(str, Enum):
    """Final eligibility status for the experiment."""
    ELIGIBLE = "ELIGIBLE"
    ELIGIBLE_PARTIAL = "ELIGIBLE_PARTIAL"
    ELIGIBLE_WARNED = "ELIGIBLE_WARNED"
    BLOCKED_FIXABLE = "BLOCKED_FIXABLE"
    INADMISSIBLE = "INADMISSIBLE"


class FailureType(str, Enum):
    """Type of failure - determines admissibility."""
    FATAL = "FATAL"   # Forever non-admissible
    STOP = "STOP"     # Fixable, blocks audit
    WARN = "WARN"     # Advisory, audit may proceed


# Valid experiment statuses
VALID_STATUSES = {'pending', 'running', 'completed', 'validated', 'rolled_back', 'failed'}
AUDITABLE_STATUSES = {'running', 'completed', 'validated', 'rolled_back', 'failed'}

# Required U2 tables
U2_REQUIRED_TABLES = [
    'u2_experiments',
    'u2_statements',
    'u2_proof_parents',
    'u2_goal_attributions',
    'u2_dag_snapshots',
]

# Required baseline tables
BASELINE_REQUIRED_TABLES = [
    'statements',
    'proof_parents',
    'theories',
]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CheckResult:
    """Result of a single pre-flight check."""
    id: str                          # e.g., "PRE-1.1"
    gate: str                        # e.g., "PRE-1"
    name: str                        # Human-readable name
    status: CheckStatus
    message: str
    failure_type: Optional[FailureType] = None  # Only set if status != PASS

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "gate": self.gate,
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "failure_type": self.failure_type.value if self.failure_type else None,
        }


@dataclass
class GateResult:
    """Result of a gate (group of checks)."""
    gate: str
    status: CheckStatus
    checks: list[CheckResult] = field(default_factory=list)
    passed: int = 0
    failed: int = 0
    warned: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "gate": self.gate,
            "status": self.status.value,
            "passed": self.passed,
            "failed": self.failed,
            "warned": self.warned,
            "check_ids": [c.id for c in self.checks],
        }


@dataclass
class PreFlightReport:
    """Complete pre-flight audit report."""
    experiment_id: str
    preflight_timestamp: str
    eligibility_status: EligibilityStatus
    gates: dict[str, GateResult] = field(default_factory=dict)
    checks: list[CheckResult] = field(default_factory=list)
    fatal_reasons: list[str] = field(default_factory=list)
    stop_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    recommendation: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        # Sort checks by ID for determinism
        sorted_checks = sorted(self.checks, key=lambda c: c.id)

        return {
            "experiment_id": self.experiment_id,
            "preflight_timestamp": self.preflight_timestamp,
            "eligibility_status": self.eligibility_status.value,
            "gates": {k: v.to_dict() for k, v in sorted(self.gates.items())},
            "checks": [c.to_dict() for c in sorted_checks],
            "fatal_reasons": sorted(self.fatal_reasons),
            "stop_reasons": sorted(self.stop_reasons),
            "warnings": sorted(self.warnings),
            "recommendation": self.recommendation,
            "notes": self.notes,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)


# =============================================================================
# PRE-1: REGISTRATION & IDENTITY
# =============================================================================

def _check_PRE1_registration_identity(
    exp_id: str,
    db_conn: Any
) -> list[CheckResult]:
    """
    PRE-1: Registration & Identity checks (5 items).

    Verifies:
    1. Experiment ID exists in u2_experiments
    2. Theory ID is valid FK
    3. Slice ID is non-empty
    4. Status is recognized
    5. Start time is set (if running/completed)
    """
    results = []

    # Check 1: Experiment ID exists
    check_id = "PRE-1.1"
    try:
        with db_conn.cursor() as cur:
            cur.execute(
                "SELECT experiment_id, theory_id, slice_id, status, start_time "
                "FROM u2_experiments WHERE experiment_id = %s",
                (exp_id,)
            )
            row = cur.fetchone()

        if row is None:
            results.append(CheckResult(
                id=check_id,
                gate="PRE-1",
                name="Experiment ID exists",
                status=CheckStatus.FAIL,
                message=f"Experiment '{exp_id}' not found in u2_experiments",
                failure_type=FailureType.STOP,
            ))
            # Early return - can't check other fields
            return results

        exp_data = {
            'experiment_id': row[0],
            'theory_id': row[1],
            'slice_id': row[2],
            'status': row[3],
            'start_time': row[4],
        }

        results.append(CheckResult(
            id=check_id,
            gate="PRE-1",
            name="Experiment ID exists",
            status=CheckStatus.PASS,
            message=f"Experiment '{exp_id}' found in u2_experiments",
        ))

    except Exception as e:
        results.append(CheckResult(
            id=check_id,
            gate="PRE-1",
            name="Experiment ID exists",
            status=CheckStatus.FAIL,
            message=f"Database error: {e}",
            failure_type=FailureType.STOP,
        ))
        return results

    # Check 2: Theory ID is valid FK
    check_id = "PRE-1.2"
    try:
        with db_conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM theories WHERE id = %s",
                (exp_data['theory_id'],)
            )
            theory_exists = cur.fetchone() is not None

        if theory_exists:
            results.append(CheckResult(
                id=check_id,
                gate="PRE-1",
                name="Theory ID is valid FK",
                status=CheckStatus.PASS,
                message=f"Theory ID {exp_data['theory_id']} exists",
            ))
        else:
            results.append(CheckResult(
                id=check_id,
                gate="PRE-1",
                name="Theory ID is valid FK",
                status=CheckStatus.FAIL,
                message=f"Theory ID {exp_data['theory_id']} not found in theories table",
                failure_type=FailureType.STOP,
            ))
    except Exception as e:
        results.append(CheckResult(
            id=check_id,
            gate="PRE-1",
            name="Theory ID is valid FK",
            status=CheckStatus.FAIL,
            message=f"Database error: {e}",
            failure_type=FailureType.STOP,
        ))

    # Check 3: Slice ID is non-empty
    check_id = "PRE-1.3"
    slice_id = exp_data.get('slice_id')
    if slice_id and str(slice_id).strip():
        results.append(CheckResult(
            id=check_id,
            gate="PRE-1",
            name="Slice ID is non-empty",
            status=CheckStatus.PASS,
            message=f"Slice ID is '{slice_id}'",
        ))
    else:
        results.append(CheckResult(
            id=check_id,
            gate="PRE-1",
            name="Slice ID is non-empty",
            status=CheckStatus.FAIL,
            message="Slice ID is null or empty",
            failure_type=FailureType.STOP,
        ))

    # Check 4: Status is recognized
    check_id = "PRE-1.4"
    status = exp_data.get('status')
    if status in VALID_STATUSES:
        results.append(CheckResult(
            id=check_id,
            gate="PRE-1",
            name="Status is recognized",
            status=CheckStatus.PASS,
            message=f"Status '{status}' is valid",
        ))
    else:
        results.append(CheckResult(
            id=check_id,
            gate="PRE-1",
            name="Status is recognized",
            status=CheckStatus.FAIL,
            message=f"Status '{status}' not in {VALID_STATUSES}",
            failure_type=FailureType.STOP,
        ))

    # Check 5: Start time is set (if running/completed)
    check_id = "PRE-1.5"
    status = exp_data.get('status')
    start_time = exp_data.get('start_time')

    if status == 'pending':
        # Start time not required for pending
        results.append(CheckResult(
            id=check_id,
            gate="PRE-1",
            name="Start time set (if applicable)",
            status=CheckStatus.PASS,
            message="Start time not required for pending status",
        ))
    elif start_time is not None:
        results.append(CheckResult(
            id=check_id,
            gate="PRE-1",
            name="Start time set (if applicable)",
            status=CheckStatus.PASS,
            message=f"Start time is {start_time}",
        ))
    else:
        results.append(CheckResult(
            id=check_id,
            gate="PRE-1",
            name="Start time set (if applicable)",
            status=CheckStatus.FAIL,
            message=f"Start time is null but status is '{status}'",
            failure_type=FailureType.STOP,
        ))

    return results


# =============================================================================
# PRE-2: PREREGISTRATION INTEGRITY
# =============================================================================

def _canonicalize_json(obj: Any) -> str:
    """Canonicalize JSON for hash computation."""
    return json.dumps(obj, sort_keys=True, separators=(',', ':'))


def _compute_prereg_hash(entry: dict) -> str:
    """Compute SHA-256 hash of a preregistration entry."""
    canonical = _canonicalize_json(entry)
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def _check_PRE2_preregistration_integrity(
    exp_id: str,
    prereg_path: Path,
    db_conn: Any
) -> list[CheckResult]:
    """
    PRE-2: Preregistration Integrity checks (3 items).

    Verifies:
    6. Experiment in PREREG_UPLIFT_U2.yaml
    7. Preregistration hash matches
    8. Prereg file has no uncommitted changes (advisory)
    """
    results = []

    # Check 6: Experiment in prereg file
    check_id = "PRE-2.6"

    if not HAS_YAML:
        results.append(CheckResult(
            id=check_id,
            gate="PRE-2",
            name="Experiment in prereg",
            status=CheckStatus.FAIL,
            message="PyYAML not installed - cannot parse prereg file",
            failure_type=FailureType.STOP,
        ))
        return results

    if not prereg_path.exists():
        results.append(CheckResult(
            id=check_id,
            gate="PRE-2",
            name="Experiment in prereg",
            status=CheckStatus.FAIL,
            message=f"Prereg file not found: {prereg_path}",
            failure_type=FailureType.FATAL,
        ))
        return results

    try:
        with open(prereg_path, 'r', encoding='utf-8') as f:
            prereg_data = yaml.safe_load(f)
    except Exception as e:
        results.append(CheckResult(
            id=check_id,
            gate="PRE-2",
            name="Experiment in prereg",
            status=CheckStatus.FAIL,
            message=f"Failed to parse prereg file: {e}",
            failure_type=FailureType.FATAL,
        ))
        return results

    # Find experiment entry
    experiments = prereg_data.get('experiments', [])
    prereg_entry = None
    for entry in experiments:
        if entry.get('experiment_id') == exp_id:
            prereg_entry = entry
            break

    if prereg_entry is None:
        results.append(CheckResult(
            id=check_id,
            gate="PRE-2",
            name="Experiment in prereg",
            status=CheckStatus.FAIL,
            message=f"Experiment '{exp_id}' not found in {prereg_path.name}",
            failure_type=FailureType.FATAL,
        ))
        return results

    results.append(CheckResult(
        id=check_id,
        gate="PRE-2",
        name="Experiment in prereg",
        status=CheckStatus.PASS,
        message=f"Experiment '{exp_id}' found in prereg",
    ))

    # Check 7: Preregistration hash matches
    check_id = "PRE-2.7"
    computed_hash = _compute_prereg_hash(prereg_entry)

    try:
        with db_conn.cursor() as cur:
            cur.execute(
                "SELECT preregistration_hash FROM u2_experiments WHERE experiment_id = %s",
                (exp_id,)
            )
            row = cur.fetchone()
            stored_hash = row[0] if row else None
    except Exception as e:
        results.append(CheckResult(
            id=check_id,
            gate="PRE-2",
            name="Prereg hash matches",
            status=CheckStatus.FAIL,
            message=f"Database error: {e}",
            failure_type=FailureType.STOP,
        ))
        stored_hash = None

    if stored_hash is None:
        results.append(CheckResult(
            id=check_id,
            gate="PRE-2",
            name="Prereg hash matches",
            status=CheckStatus.FAIL,
            message="No preregistration hash stored in database",
            failure_type=FailureType.STOP,
        ))
    elif stored_hash == computed_hash:
        results.append(CheckResult(
            id=check_id,
            gate="PRE-2",
            name="Prereg hash matches",
            status=CheckStatus.PASS,
            message=f"Hash matches: {computed_hash[:16]}...",
        ))
    else:
        results.append(CheckResult(
            id=check_id,
            gate="PRE-2",
            name="Prereg hash matches",
            status=CheckStatus.FAIL,
            message=f"Hash mismatch: stored={stored_hash[:16]}... computed={computed_hash[:16]}...",
            failure_type=FailureType.FATAL,
        ))

    # Check 8: Prereg file has no uncommitted changes (advisory)
    check_id = "PRE-2.8"
    try:
        import subprocess
        result = subprocess.run(
            ['git', 'status', '--porcelain', str(prereg_path)],
            capture_output=True,
            text=True,
            cwd=prereg_path.parent.parent,  # Assume repo root
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip() == '':
            results.append(CheckResult(
                id=check_id,
                gate="PRE-2",
                name="Prereg committed",
                status=CheckStatus.PASS,
                message="Prereg file has no uncommitted changes",
            ))
        elif result.returncode == 0:
            results.append(CheckResult(
                id=check_id,
                gate="PRE-2",
                name="Prereg committed",
                status=CheckStatus.WARN,
                message=f"Prereg file has uncommitted changes: {result.stdout.strip()}",
                failure_type=FailureType.WARN,
            ))
        else:
            results.append(CheckResult(
                id=check_id,
                gate="PRE-2",
                name="Prereg committed",
                status=CheckStatus.WARN,
                message="Could not check git status (not a git repo?)",
                failure_type=FailureType.WARN,
            ))
    except Exception as e:
        results.append(CheckResult(
            id=check_id,
            gate="PRE-2",
            name="Prereg committed",
            status=CheckStatus.WARN,
            message=f"Could not check git status: {e}",
            failure_type=FailureType.WARN,
        ))

    return results


# =============================================================================
# PRE-3: BASELINE SNAPSHOT
# =============================================================================

def _check_PRE3_baseline_snapshot(
    exp_id: str,
    db_conn: Any
) -> list[CheckResult]:
    """
    PRE-3: Baseline Snapshot checks (5 items).

    Verifies:
    9. Snapshot record exists
    10. Merkle root is non-null
    11. Statement count recorded
    12. Edge count recorded
    13. Snapshot timestamp is before experiment start
    """
    results = []

    # Check 9: Snapshot record exists
    check_id = "PRE-3.9"
    try:
        with db_conn.cursor() as cur:
            cur.execute(
                "SELECT experiment_id, root_merkle_hash, statement_count, "
                "edge_count, snapshot_timestamp "
                "FROM u2_dag_snapshots WHERE experiment_id = %s",
                (exp_id,)
            )
            snapshot_row = cur.fetchone()
    except Exception as e:
        results.append(CheckResult(
            id=check_id,
            gate="PRE-3",
            name="Snapshot exists",
            status=CheckStatus.FAIL,
            message=f"Database error: {e}",
            failure_type=FailureType.STOP,
        ))
        return results

    if snapshot_row is None:
        results.append(CheckResult(
            id=check_id,
            gate="PRE-3",
            name="Snapshot exists",
            status=CheckStatus.FAIL,
            message=f"No snapshot found for experiment '{exp_id}'",
            failure_type=FailureType.STOP,
        ))
        return results

    snapshot = {
        'experiment_id': snapshot_row[0],
        'root_merkle_hash': snapshot_row[1],
        'statement_count': snapshot_row[2],
        'edge_count': snapshot_row[3],
        'snapshot_timestamp': snapshot_row[4],
    }

    results.append(CheckResult(
        id=check_id,
        gate="PRE-3",
        name="Snapshot exists",
        status=CheckStatus.PASS,
        message="Snapshot record found",
    ))

    # Check 10: Merkle root is non-null
    check_id = "PRE-3.10"
    merkle = snapshot.get('root_merkle_hash')
    if merkle and len(str(merkle)) == 64:
        results.append(CheckResult(
            id=check_id,
            gate="PRE-3",
            name="Merkle root valid",
            status=CheckStatus.PASS,
            message=f"Merkle root: {str(merkle)[:16]}...",
        ))
    elif merkle:
        results.append(CheckResult(
            id=check_id,
            gate="PRE-3",
            name="Merkle root valid",
            status=CheckStatus.FAIL,
            message=f"Merkle root invalid length: {len(str(merkle))} (expected 64)",
            failure_type=FailureType.STOP,
        ))
    else:
        results.append(CheckResult(
            id=check_id,
            gate="PRE-3",
            name="Merkle root valid",
            status=CheckStatus.FAIL,
            message="Merkle root is null",
            failure_type=FailureType.STOP,
        ))

    # Check 11: Statement count recorded
    check_id = "PRE-3.11"
    stmt_count = snapshot.get('statement_count')
    if stmt_count is not None and stmt_count > 0:
        results.append(CheckResult(
            id=check_id,
            gate="PRE-3",
            name="Statement count recorded",
            status=CheckStatus.PASS,
            message=f"Statement count: {stmt_count}",
        ))
    elif stmt_count == 0:
        results.append(CheckResult(
            id=check_id,
            gate="PRE-3",
            name="Statement count recorded",
            status=CheckStatus.WARN,
            message="Statement count is 0 (empty baseline?)",
            failure_type=FailureType.WARN,
        ))
    else:
        results.append(CheckResult(
            id=check_id,
            gate="PRE-3",
            name="Statement count recorded",
            status=CheckStatus.FAIL,
            message="Statement count is null",
            failure_type=FailureType.STOP,
        ))

    # Check 12: Edge count recorded
    check_id = "PRE-3.12"
    edge_count = snapshot.get('edge_count')
    if edge_count is not None and edge_count >= 0:
        results.append(CheckResult(
            id=check_id,
            gate="PRE-3",
            name="Edge count recorded",
            status=CheckStatus.PASS,
            message=f"Edge count: {edge_count}",
        ))
    else:
        results.append(CheckResult(
            id=check_id,
            gate="PRE-3",
            name="Edge count recorded",
            status=CheckStatus.FAIL,
            message="Edge count is null",
            failure_type=FailureType.STOP,
        ))

    # Check 13: Snapshot timestamp before experiment start
    check_id = "PRE-3.13"
    snapshot_ts = snapshot.get('snapshot_timestamp')

    try:
        with db_conn.cursor() as cur:
            cur.execute(
                "SELECT start_time FROM u2_experiments WHERE experiment_id = %s",
                (exp_id,)
            )
            row = cur.fetchone()
            start_time = row[0] if row else None
    except Exception as e:
        results.append(CheckResult(
            id=check_id,
            gate="PRE-3",
            name="Snapshot before start",
            status=CheckStatus.FAIL,
            message=f"Database error: {e}",
            failure_type=FailureType.STOP,
        ))
        return results

    if start_time is None:
        # Experiment hasn't started yet - snapshot timing is OK
        results.append(CheckResult(
            id=check_id,
            gate="PRE-3",
            name="Snapshot before start",
            status=CheckStatus.PASS,
            message="Experiment not started yet; snapshot timing valid",
        ))
    elif snapshot_ts is None:
        results.append(CheckResult(
            id=check_id,
            gate="PRE-3",
            name="Snapshot before start",
            status=CheckStatus.FAIL,
            message="Snapshot timestamp is null",
            failure_type=FailureType.STOP,
        ))
    elif snapshot_ts < start_time:
        results.append(CheckResult(
            id=check_id,
            gate="PRE-3",
            name="Snapshot before start",
            status=CheckStatus.PASS,
            message=f"Snapshot ({snapshot_ts}) < Start ({start_time})",
        ))
    else:
        results.append(CheckResult(
            id=check_id,
            gate="PRE-3",
            name="Snapshot before start",
            status=CheckStatus.FAIL,
            message=f"FATAL: Snapshot ({snapshot_ts}) >= Start ({start_time}) - baseline contaminated",
            failure_type=FailureType.FATAL,
        ))

    return results


# =============================================================================
# PRE-4: LOG DIRECTORY INTEGRITY
# =============================================================================

def _check_PRE4_log_directory_integrity(
    exp_id: str,
    run_dir: Path
) -> list[CheckResult]:
    """
    PRE-4: Log Directory Integrity checks (7 items).

    Verifies:
    14. Log directory exists
    15. manifest.json exists
    16. manifest.json is valid JSON
    17. At least one cycle log exists
    18. Cycle logs are parseable JSONL
    19. verifications.jsonl exists
    20. verifications.jsonl is parseable
    """
    results = []
    log_dir = run_dir / "logs" / "u2" / exp_id

    # Check 14: Log directory exists
    check_id = "PRE-4.14"
    if log_dir.exists() and log_dir.is_dir():
        results.append(CheckResult(
            id=check_id,
            gate="PRE-4",
            name="Log directory exists",
            status=CheckStatus.PASS,
            message=f"Directory exists: {log_dir}",
        ))
    else:
        results.append(CheckResult(
            id=check_id,
            gate="PRE-4",
            name="Log directory exists",
            status=CheckStatus.FAIL,
            message=f"Directory not found: {log_dir}",
            failure_type=FailureType.STOP,
        ))
        return results  # Can't check other files

    # Check 15: manifest.json exists
    check_id = "PRE-4.15"
    manifest_path = log_dir / "manifest.json"
    if manifest_path.exists():
        results.append(CheckResult(
            id=check_id,
            gate="PRE-4",
            name="manifest.json exists",
            status=CheckStatus.PASS,
            message="manifest.json found",
        ))
    else:
        results.append(CheckResult(
            id=check_id,
            gate="PRE-4",
            name="manifest.json exists",
            status=CheckStatus.FAIL,
            message="manifest.json not found",
            failure_type=FailureType.STOP,
        ))

    # Check 16: manifest.json is valid JSON
    check_id = "PRE-4.16"
    if manifest_path.exists():
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                json.load(f)
            results.append(CheckResult(
                id=check_id,
                gate="PRE-4",
                name="manifest.json valid",
                status=CheckStatus.PASS,
                message="manifest.json is valid JSON",
            ))
        except json.JSONDecodeError as e:
            results.append(CheckResult(
                id=check_id,
                gate="PRE-4",
                name="manifest.json valid",
                status=CheckStatus.FAIL,
                message=f"manifest.json parse error: {e}",
                failure_type=FailureType.STOP,
            ))
    else:
        results.append(CheckResult(
            id=check_id,
            gate="PRE-4",
            name="manifest.json valid",
            status=CheckStatus.FAIL,
            message="Cannot validate - file missing",
            failure_type=FailureType.STOP,
        ))

    # Check 17: At least one cycle log exists
    check_id = "PRE-4.17"
    cycle_logs = list(log_dir.glob("cycle_*.jsonl"))
    if cycle_logs:
        results.append(CheckResult(
            id=check_id,
            gate="PRE-4",
            name="Cycle logs exist",
            status=CheckStatus.PASS,
            message=f"Found {len(cycle_logs)} cycle log(s)",
        ))
    else:
        results.append(CheckResult(
            id=check_id,
            gate="PRE-4",
            name="Cycle logs exist",
            status=CheckStatus.FAIL,
            message="No cycle_*.jsonl files found",
            failure_type=FailureType.STOP,
        ))

    # Check 18: Cycle logs are parseable JSONL
    check_id = "PRE-4.18"
    if cycle_logs:
        parse_errors = []
        for log_file in cycle_logs:
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:  # Skip empty lines
                            json.loads(line)
            except json.JSONDecodeError as e:
                parse_errors.append(f"{log_file.name}:{line_num}: {e}")

        if not parse_errors:
            results.append(CheckResult(
                id=check_id,
                gate="PRE-4",
                name="Cycle logs parseable",
                status=CheckStatus.PASS,
                message="All cycle logs are valid JSONL",
            ))
        else:
            results.append(CheckResult(
                id=check_id,
                gate="PRE-4",
                name="Cycle logs parseable",
                status=CheckStatus.FAIL,
                message=f"Parse errors: {'; '.join(parse_errors[:3])}",
                failure_type=FailureType.STOP,
            ))
    else:
        results.append(CheckResult(
            id=check_id,
            gate="PRE-4",
            name="Cycle logs parseable",
            status=CheckStatus.FAIL,
            message="No cycle logs to validate",
            failure_type=FailureType.STOP,
        ))

    # Check 19: verifications.jsonl exists
    check_id = "PRE-4.19"
    verif_path = log_dir / "verifications.jsonl"
    if verif_path.exists():
        results.append(CheckResult(
            id=check_id,
            gate="PRE-4",
            name="verifications.jsonl exists",
            status=CheckStatus.PASS,
            message="verifications.jsonl found",
        ))
    else:
        results.append(CheckResult(
            id=check_id,
            gate="PRE-4",
            name="verifications.jsonl exists",
            status=CheckStatus.WARN,
            message="verifications.jsonl not found (verification receipts unavailable)",
            failure_type=FailureType.WARN,
        ))

    # Check 20: verifications.jsonl is parseable
    check_id = "PRE-4.20"
    if verif_path.exists():
        try:
            with open(verif_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        json.loads(line)
            results.append(CheckResult(
                id=check_id,
                gate="PRE-4",
                name="verifications.jsonl parseable",
                status=CheckStatus.PASS,
                message="verifications.jsonl is valid JSONL",
            ))
        except json.JSONDecodeError as e:
            results.append(CheckResult(
                id=check_id,
                gate="PRE-4",
                name="verifications.jsonl parseable",
                status=CheckStatus.FAIL,
                message=f"Parse error at line {line_num}: {e}",
                failure_type=FailureType.STOP,
            ))
    else:
        results.append(CheckResult(
            id=check_id,
            gate="PRE-4",
            name="verifications.jsonl parseable",
            status=CheckStatus.WARN,
            message="Cannot validate - file missing",
            failure_type=FailureType.WARN,
        ))

    return results


# =============================================================================
# PRE-5: DATABASE CONNECTIVITY
# =============================================================================

def _check_PRE5_database_connectivity(
    db_url: str
) -> tuple[list[CheckResult], Any]:
    """
    PRE-5: Database Connectivity checks (3 items).

    Verifies:
    21. Database connection succeeds
    22. All U2 tables exist
    23. Baseline tables exist

    Returns (results, connection) - connection may be None if failed.
    """
    results = []
    conn = None

    # Check 21: Database connection succeeds
    check_id = "PRE-5.21"
    if not HAS_PSYCOPG2:
        results.append(CheckResult(
            id=check_id,
            gate="PRE-5",
            name="Database connection",
            status=CheckStatus.FAIL,
            message="psycopg2 not installed",
            failure_type=FailureType.STOP,
        ))
        return results, None

    try:
        conn = psycopg2.connect(db_url)
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        results.append(CheckResult(
            id=check_id,
            gate="PRE-5",
            name="Database connection",
            status=CheckStatus.PASS,
            message="Database connection successful",
        ))
    except Exception as e:
        results.append(CheckResult(
            id=check_id,
            gate="PRE-5",
            name="Database connection",
            status=CheckStatus.FAIL,
            message=f"Connection failed: {e}",
            failure_type=FailureType.STOP,
        ))
        return results, None

    # Check 22: All U2 tables exist
    check_id = "PRE-5.22"
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public'"
            )
            existing_tables = {row[0] for row in cur.fetchall()}

        missing_u2 = [t for t in U2_REQUIRED_TABLES if t not in existing_tables]

        if not missing_u2:
            results.append(CheckResult(
                id=check_id,
                gate="PRE-5",
                name="U2 tables exist",
                status=CheckStatus.PASS,
                message=f"All {len(U2_REQUIRED_TABLES)} U2 tables present",
            ))
        else:
            results.append(CheckResult(
                id=check_id,
                gate="PRE-5",
                name="U2 tables exist",
                status=CheckStatus.FAIL,
                message=f"Missing U2 tables: {missing_u2}",
                failure_type=FailureType.STOP,
            ))
    except Exception as e:
        results.append(CheckResult(
            id=check_id,
            gate="PRE-5",
            name="U2 tables exist",
            status=CheckStatus.FAIL,
            message=f"Error checking tables: {e}",
            failure_type=FailureType.STOP,
        ))

    # Check 23: Baseline tables exist
    check_id = "PRE-5.23"
    try:
        missing_baseline = [t for t in BASELINE_REQUIRED_TABLES if t not in existing_tables]

        if not missing_baseline:
            results.append(CheckResult(
                id=check_id,
                gate="PRE-5",
                name="Baseline tables exist",
                status=CheckStatus.PASS,
                message=f"All {len(BASELINE_REQUIRED_TABLES)} baseline tables present",
            ))
        else:
            results.append(CheckResult(
                id=check_id,
                gate="PRE-5",
                name="Baseline tables exist",
                status=CheckStatus.FAIL,
                message=f"Missing baseline tables: {missing_baseline}",
                failure_type=FailureType.STOP,
            ))
    except Exception as e:
        results.append(CheckResult(
            id=check_id,
            gate="PRE-5",
            name="Baseline tables exist",
            status=CheckStatus.FAIL,
            message=f"Error checking tables: {e}",
            failure_type=FailureType.STOP,
        ))

    return results, conn


# =============================================================================
# PRE-6: STATE ELIGIBILITY
# =============================================================================

def _check_PRE6_state_eligibility(
    exp_id: str,
    db_conn: Any
) -> list[CheckResult]:
    """
    PRE-6: State Eligibility checks (2 items).

    Verifies:
    24. Status is auditable
    25. If running, has completed cycles
    """
    results = []

    # Check 24: Status is auditable
    check_id = "PRE-6.24"
    try:
        with db_conn.cursor() as cur:
            cur.execute(
                "SELECT status FROM u2_experiments WHERE experiment_id = %s",
                (exp_id,)
            )
            row = cur.fetchone()
            status = row[0] if row else None
    except Exception as e:
        results.append(CheckResult(
            id=check_id,
            gate="PRE-6",
            name="Status is auditable",
            status=CheckStatus.FAIL,
            message=f"Database error: {e}",
            failure_type=FailureType.STOP,
        ))
        return results

    if status in AUDITABLE_STATUSES:
        results.append(CheckResult(
            id=check_id,
            gate="PRE-6",
            name="Status is auditable",
            status=CheckStatus.PASS,
            message=f"Status '{status}' is auditable",
        ))
    elif status == 'pending':
        results.append(CheckResult(
            id=check_id,
            gate="PRE-6",
            name="Status is auditable",
            status=CheckStatus.FAIL,
            message="Status is 'pending' - experiment not started",
            failure_type=FailureType.STOP,
        ))
    else:
        results.append(CheckResult(
            id=check_id,
            gate="PRE-6",
            name="Status is auditable",
            status=CheckStatus.FAIL,
            message=f"Status '{status}' is not auditable",
            failure_type=FailureType.STOP,
        ))

    # Check 25: If running, has completed cycles
    check_id = "PRE-6.25"
    if status == 'running':
        try:
            with db_conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(DISTINCT cycle_number) FROM u2_statements "
                    "WHERE experiment_id = %s",
                    (exp_id,)
                )
                row = cur.fetchone()
                cycle_count = row[0] if row else 0
        except Exception as e:
            results.append(CheckResult(
                id=check_id,
                gate="PRE-6",
                name="Has completed cycles",
                status=CheckStatus.FAIL,
                message=f"Database error: {e}",
                failure_type=FailureType.STOP,
            ))
            return results

        if cycle_count > 0:
            results.append(CheckResult(
                id=check_id,
                gate="PRE-6",
                name="Has completed cycles",
                status=CheckStatus.WARN,
                message=f"Running experiment with {cycle_count} completed cycle(s) - partial audit",
                failure_type=FailureType.WARN,
            ))
        else:
            results.append(CheckResult(
                id=check_id,
                gate="PRE-6",
                name="Has completed cycles",
                status=CheckStatus.FAIL,
                message="Running experiment with 0 completed cycles",
                failure_type=FailureType.STOP,
            ))
    else:
        results.append(CheckResult(
            id=check_id,
            gate="PRE-6",
            name="Has completed cycles",
            status=CheckStatus.PASS,
            message=f"Not running (status='{status}') - cycle check N/A",
        ))

    return results


# =============================================================================
# SNAPSHOT SCHEMA VERSION
# =============================================================================

PREFLIGHT_SNAPSHOT_SCHEMA_VERSION = "1.0.0"


# =============================================================================
# SNAPSHOT FUNCTIONS (Task 1)
# =============================================================================

def build_preflight_snapshot(report: PreFlightReport) -> dict[str, Any]:
    """
    Build a compact snapshot of pre-flight results for comparison.

    The snapshot captures:
    - Schema version for compatibility checking
    - Eligibility status enum value
    - Counts of checks by status (PASS/WARN/FAIL) and failure type
    - List of failed check IDs for delta tracking

    Args:
        report: PreFlightReport from run_preflight()

    Returns:
        Dictionary with snapshot data suitable for JSON serialization
    """
    # Count checks by status
    pass_count = sum(1 for c in report.checks if c.status == CheckStatus.PASS)
    warn_count = sum(1 for c in report.checks if c.status == CheckStatus.WARN)
    fail_count = sum(1 for c in report.checks if c.status == CheckStatus.FAIL)

    # Count by failure type
    fatal_count = sum(1 for c in report.checks if c.failure_type == FailureType.FATAL)
    stop_count = sum(1 for c in report.checks if c.failure_type == FailureType.STOP)
    warn_type_count = sum(1 for c in report.checks if c.failure_type == FailureType.WARN)

    # Collect failed check IDs (sorted for determinism)
    failed_check_ids = sorted([
        c.id for c in report.checks
        if c.status in (CheckStatus.FAIL, CheckStatus.WARN)
    ])

    return {
        "schema_version": PREFLIGHT_SNAPSHOT_SCHEMA_VERSION,
        "experiment_id": report.experiment_id,
        "preflight_timestamp": report.preflight_timestamp,
        "eligibility": report.eligibility_status.value,
        "counts": {
            "total": len(report.checks),
            "pass": pass_count,
            "warn": warn_count,
            "fail": fail_count,
        },
        "failure_types": {
            "fatal": fatal_count,
            "stop": stop_count,
            "warn": warn_type_count,
        },
        "failed_check_ids": failed_check_ids,
    }


def compare_preflight_snapshots(
    old: dict[str, Any],
    new: dict[str, Any]
) -> dict[str, Any]:
    """
    Compare two pre-flight snapshots to detect changes.

    Useful for tracking experiment remediation progress or detecting
    regressions between pre-flight runs.

    Args:
        old: Previous snapshot from build_preflight_snapshot()
        new: Current snapshot from build_preflight_snapshot()

    Returns:
        Dictionary with comparison results:
        - schema_compatible: bool
        - eligibility_change: tuple (old, new) or None if unchanged
        - new_failures: list of check IDs that failed in new but not old
        - resolved_failures: list of check IDs that failed in old but pass in new
        - count_deltas: dict with count changes
        - improved: bool - True if eligibility improved or failures decreased
        - regressed: bool - True if eligibility worsened or new failures appeared
    """
    # Check schema compatibility
    old_version = old.get("schema_version", "0.0.0")
    new_version = new.get("schema_version", "0.0.0")
    schema_compatible = old_version == new_version

    # Eligibility change detection
    old_eligibility = old.get("eligibility")
    new_eligibility = new.get("eligibility")
    eligibility_change = None
    if old_eligibility != new_eligibility:
        eligibility_change = (old_eligibility, new_eligibility)

    # Failed check ID deltas
    old_failed = set(old.get("failed_check_ids", []))
    new_failed = set(new.get("failed_check_ids", []))

    new_failures = sorted(new_failed - old_failed)
    resolved_failures = sorted(old_failed - new_failed)

    # Count deltas
    old_counts = old.get("counts", {})
    new_counts = new.get("counts", {})

    count_deltas = {
        "pass": new_counts.get("pass", 0) - old_counts.get("pass", 0),
        "warn": new_counts.get("warn", 0) - old_counts.get("warn", 0),
        "fail": new_counts.get("fail", 0) - old_counts.get("fail", 0),
    }

    # Failure type deltas
    old_types = old.get("failure_types", {})
    new_types = new.get("failure_types", {})

    type_deltas = {
        "fatal": new_types.get("fatal", 0) - old_types.get("fatal", 0),
        "stop": new_types.get("stop", 0) - old_types.get("stop", 0),
        "warn": new_types.get("warn", 0) - old_types.get("warn", 0),
    }

    # Determine improvement/regression
    # Eligibility ordering (best to worst)
    eligibility_order = [
        EligibilityStatus.ELIGIBLE.value,
        EligibilityStatus.ELIGIBLE_WARNED.value,
        EligibilityStatus.ELIGIBLE_PARTIAL.value,
        EligibilityStatus.BLOCKED_FIXABLE.value,
        EligibilityStatus.INADMISSIBLE.value,
    ]

    def eligibility_rank(e: str) -> int:
        try:
            return eligibility_order.index(e)
        except ValueError:
            return 999  # Unknown status ranks worst

    old_rank = eligibility_rank(old_eligibility) if old_eligibility else 999
    new_rank = eligibility_rank(new_eligibility) if new_eligibility else 999

    # Improved: eligibility got better OR failures decreased (no new failures)
    improved = (new_rank < old_rank) or (
        new_rank == old_rank and len(resolved_failures) > 0 and len(new_failures) == 0
    )

    # Regressed: eligibility got worse OR new failures appeared
    regressed = (new_rank > old_rank) or len(new_failures) > 0

    return {
        "schema_compatible": schema_compatible,
        "eligibility_change": eligibility_change,
        "new_failures": new_failures,
        "resolved_failures": resolved_failures,
        "count_deltas": count_deltas,
        "failure_type_deltas": type_deltas,
        "improved": improved,
        "regressed": regressed,
    }


# =============================================================================
# BUNDLE BRIDGE (Task 2)
# =============================================================================

def to_bundle_stage_result(report: PreFlightReport) -> dict[str, Any]:
    """
    Convert PreFlightReport to bundle-compatible stage result format.

    Produces a structure compatible with evidence bundle stage_results schema:
    - stage_name: "preflight"
    - status: "pass" | "warn" | "fail"
    - checks: list of check objects with id, status, message
    - summary: eligibility status and recommendation

    Args:
        report: PreFlightReport from run_preflight()

    Returns:
        Dictionary in bundle stage_result format
    """
    # Map eligibility to pass/warn/fail
    if report.eligibility_status in (EligibilityStatus.ELIGIBLE,):
        bundle_status = "pass"
    elif report.eligibility_status in (
        EligibilityStatus.ELIGIBLE_WARNED,
        EligibilityStatus.ELIGIBLE_PARTIAL
    ):
        bundle_status = "warn"
    else:
        bundle_status = "fail"

    # Convert checks to bundle format
    bundle_checks = []
    for check in sorted(report.checks, key=lambda c: c.id):
        bundle_checks.append({
            "id": check.id,
            "gate": check.gate,
            "name": check.name,
            "status": check.status.value.lower(),  # pass/fail/warn
            "message": check.message,
            "failure_type": check.failure_type.value.lower() if check.failure_type else None,
        })

    # Gate summaries
    gate_summaries = {}
    for gate_id, gate in sorted(report.gates.items()):
        gate_summaries[gate_id] = {
            "status": gate.status.value.lower(),
            "passed": gate.passed,
            "failed": gate.failed,
            "warned": gate.warned,
        }

    return {
        "stage_name": "preflight",
        "stage_version": PREFLIGHT_SNAPSHOT_SCHEMA_VERSION,
        "status": bundle_status,
        "experiment_id": report.experiment_id,
        "timestamp": report.preflight_timestamp,
        "eligibility": report.eligibility_status.value,
        "recommendation": report.recommendation,
        "gates": gate_summaries,
        "checks": bundle_checks,
        "fatal_reasons": report.fatal_reasons,
        "stop_reasons": report.stop_reasons,
        "warnings": report.warnings,
    }


# =============================================================================
# MARKDOWN RENDERER (Task 3)
# =============================================================================

def render_preflight_markdown(report: PreFlightReport) -> str:
    """
    Render PreFlightReport as GitHub-flavored markdown summary.

    Produces a human-readable summary suitable for:
    - GitHub Actions step summary ($GITHUB_STEP_SUMMARY)
    - Pull request comments
    - Audit documentation

    Args:
        report: PreFlightReport from run_preflight()

    Returns:
        Markdown-formatted string
    """
    lines = []

    # Header
    lines.append("# U2 Pre-Flight Audit Report")
    lines.append("")

    # Status badge (emoji)
    status_emoji = {
        EligibilityStatus.ELIGIBLE: "✅",
        EligibilityStatus.ELIGIBLE_WARNED: "⚠️",
        EligibilityStatus.ELIGIBLE_PARTIAL: "🔶",
        EligibilityStatus.BLOCKED_FIXABLE: "🛑",
        EligibilityStatus.INADMISSIBLE: "❌",
    }
    emoji = status_emoji.get(report.eligibility_status, "❓")

    lines.append(f"**Experiment:** `{report.experiment_id}`")
    lines.append(f"**Status:** {emoji} `{report.eligibility_status.value}`")
    lines.append(f"**Timestamp:** {report.preflight_timestamp}")
    lines.append("")

    # Recommendation
    lines.append("## Recommendation")
    lines.append("")
    lines.append(f"> {report.recommendation}")
    lines.append("")

    # Gate Summary Table
    lines.append("## Gate Summary")
    lines.append("")
    lines.append("| Gate | Status | Passed | Failed | Warned |")
    lines.append("|------|--------|--------|--------|--------|")

    for gate_id in sorted(report.gates.keys()):
        gate = report.gates[gate_id]
        gate_emoji = "✅" if gate.status == CheckStatus.PASS else (
            "⚠️" if gate.status == CheckStatus.WARN else "❌"
        )
        lines.append(
            f"| {gate_id} | {gate_emoji} {gate.status.value} | "
            f"{gate.passed} | {gate.failed} | {gate.warned} |"
        )

    lines.append("")

    # Fatal reasons (if any)
    if report.fatal_reasons:
        lines.append("## ❌ Fatal Conditions (INADMISSIBLE)")
        lines.append("")
        lines.append("These conditions **permanently** disqualify the experiment:")
        lines.append("")
        for reason in report.fatal_reasons:
            lines.append(f"- {reason}")
        lines.append("")

    # Stop reasons (if any)
    if report.stop_reasons:
        lines.append("## 🛑 Blocking Issues (FIXABLE)")
        lines.append("")
        lines.append("These issues must be resolved before audit:")
        lines.append("")
        for reason in report.stop_reasons:
            lines.append(f"- {reason}")
        lines.append("")

    # Warnings (if any)
    if report.warnings:
        lines.append("## ⚠️ Warnings")
        lines.append("")
        lines.append("These should be documented but do not block audit:")
        lines.append("")
        for warning in report.warnings:
            lines.append(f"- {warning}")
        lines.append("")

    # Detailed Check Results (collapsible)
    lines.append("## Detailed Check Results")
    lines.append("")
    lines.append("<details>")
    lines.append("<summary>Click to expand all checks</summary>")
    lines.append("")
    lines.append("| ID | Gate | Name | Status | Message |")
    lines.append("|-----|------|------|--------|---------|")

    for check in sorted(report.checks, key=lambda c: c.id):
        check_emoji = "✅" if check.status == CheckStatus.PASS else (
            "⚠️" if check.status == CheckStatus.WARN else "❌"
        )
        # Escape pipe characters in message
        safe_message = check.message.replace("|", "\\|")
        lines.append(
            f"| {check.id} | {check.gate} | {check.name} | "
            f"{check_emoji} {check.status.value} | {safe_message} |"
        )

    lines.append("")
    lines.append("</details>")
    lines.append("")

    # Footer
    passed = sum(1 for c in report.checks if c.status == CheckStatus.PASS)
    total = len(report.checks)
    lines.append("---")
    lines.append(f"*{passed}/{total} checks passed*")
    lines.append("")

    return "\n".join(lines)


# =============================================================================
# PHASE III: DRIFT TIMELINE (Task 1)
# =============================================================================

def build_preflight_drift_timeline(
    snapshots: list[dict[str, Any]]
) -> dict[str, Any]:
    """
    Analyze a sequence of preflight snapshots to detect drift patterns.

    This function takes a chronologically ordered list of snapshots (oldest first)
    and produces a timeline analysis including:
    - Eligibility shifts over time
    - Recurring failures (check IDs that fail repeatedly)
    - Run stability index (0.0 = unstable, 1.0 = perfectly stable)

    Args:
        snapshots: List of snapshots from build_preflight_snapshot(), ordered
                   chronologically (oldest first)

    Returns:
        Dictionary with drift timeline analysis:
        - eligibility_shifts: list of (timestamp, old_status, new_status) tuples
        - recurring_failures: dict of check_id -> failure_count (only if count > 1)
        - run_stability_index: float between 0.0 and 1.0
        - total_runs: number of snapshots analyzed
        - first_timestamp: earliest snapshot timestamp
        - last_timestamp: latest snapshot timestamp
    """
    if not snapshots:
        return {
            "eligibility_shifts": [],
            "recurring_failures": {},
            "run_stability_index": 1.0,
            "total_runs": 0,
            "first_timestamp": None,
            "last_timestamp": None,
        }

    if len(snapshots) == 1:
        snap = snapshots[0]
        return {
            "eligibility_shifts": [],
            "recurring_failures": {},
            "run_stability_index": 1.0,
            "total_runs": 1,
            "first_timestamp": snap.get("preflight_timestamp"),
            "last_timestamp": snap.get("preflight_timestamp"),
        }

    # Track eligibility shifts
    eligibility_shifts = []
    prev_eligibility = snapshots[0].get("eligibility")

    for snap in snapshots[1:]:
        curr_eligibility = snap.get("eligibility")
        if curr_eligibility != prev_eligibility:
            eligibility_shifts.append((
                snap.get("preflight_timestamp"),
                prev_eligibility,
                curr_eligibility,
            ))
        prev_eligibility = curr_eligibility

    # Count failure occurrences across all snapshots
    failure_counts: dict[str, int] = {}
    for snap in snapshots:
        for check_id in snap.get("failed_check_ids", []):
            failure_counts[check_id] = failure_counts.get(check_id, 0) + 1

    # Filter to recurring failures (> 1 occurrence)
    recurring_failures = {
        check_id: count
        for check_id, count in sorted(failure_counts.items())
        if count > 1
    }

    # Calculate run stability index
    # Based on: (1 - shifts/possible_shifts) * (1 - recurring_ratio)
    n = len(snapshots)
    max_shifts = n - 1  # Maximum possible eligibility changes

    shift_stability = 1.0 - (len(eligibility_shifts) / max_shifts) if max_shifts > 0 else 1.0

    # Recurring ratio: how many checks fail repeatedly vs total unique failures
    total_unique_failures = len(failure_counts)
    recurring_count = len(recurring_failures)
    recurring_ratio = recurring_count / total_unique_failures if total_unique_failures > 0 else 0.0
    failure_stability = 1.0 - recurring_ratio

    # Combined stability index (weighted average)
    run_stability_index = round(0.6 * shift_stability + 0.4 * failure_stability, 3)

    return {
        "eligibility_shifts": eligibility_shifts,
        "recurring_failures": recurring_failures,
        "run_stability_index": run_stability_index,
        "total_runs": n,
        "first_timestamp": snapshots[0].get("preflight_timestamp"),
        "last_timestamp": snapshots[-1].get("preflight_timestamp"),
    }


# =============================================================================
# PHASE III: MAAS BRIDGE (Task 2)
# =============================================================================

def summarize_preflight_for_maas(
    snapshot: dict[str, Any]
) -> dict[str, Any]:
    """
    Summarize preflight snapshot for MAAS (MathLedger Audit & Analysis System).

    Produces a compact summary suitable for MAAS ingestion, with:
    - admissible: bool indicating if experiment can proceed to audit
    - blocking_pf_ids: list of check IDs causing blockage (FATAL/STOP)
    - status: simplified status string for MAAS dashboard

    Args:
        snapshot: Snapshot from build_preflight_snapshot()

    Returns:
        Dictionary with MAAS-compatible summary:
        - admissible: True if ELIGIBLE/ELIGIBLE_WARNED/ELIGIBLE_PARTIAL
        - blocking_pf_ids: sorted list of check IDs with STOP/FATAL failures
        - status: "green" | "yellow" | "red"
        - eligibility: original eligibility status string
        - experiment_id: experiment identifier
        - timestamp: preflight timestamp
        - failure_summary: compact failure counts
    """
    eligibility = snapshot.get("eligibility", "UNKNOWN")

    # Determine admissibility
    admissible_statuses = {"ELIGIBLE", "ELIGIBLE_WARNED", "ELIGIBLE_PARTIAL"}
    admissible = eligibility in admissible_statuses

    # Determine blocking check IDs (those that are FAIL, not just WARN)
    # We need to identify STOP/FATAL failures from the snapshot
    failed_check_ids = snapshot.get("failed_check_ids", [])
    failure_types = snapshot.get("failure_types", {})

    # If there are FATAL or STOP failures, those are blocking
    fatal_count = failure_types.get("fatal", 0)
    stop_count = failure_types.get("stop", 0)

    # For blocking IDs, we consider all failed checks if there are STOP/FATAL
    # The snapshot doesn't distinguish which specific IDs are STOP vs WARN,
    # but if we have STOP or FATAL counts > 0, we're blocked
    if fatal_count > 0 or stop_count > 0:
        # All non-WARN failures are blocking
        # Since snapshot aggregates, we report all failed IDs when blocked
        blocking_pf_ids = sorted([
            cid for cid in failed_check_ids
            # Note: In full implementation, we'd track failure type per check
            # For now, if blocked, all failed checks are considered blocking
        ])
    else:
        blocking_pf_ids = []

    # Determine traffic light status
    if eligibility == "ELIGIBLE":
        status = "green"
    elif eligibility in {"ELIGIBLE_WARNED", "ELIGIBLE_PARTIAL"}:
        status = "yellow"
    else:
        status = "red"

    # Compact failure summary
    counts = snapshot.get("counts", {})
    failure_summary = {
        "total_checks": counts.get("total", 0),
        "passed": counts.get("pass", 0),
        "warned": counts.get("warn", 0),
        "failed": counts.get("fail", 0),
        "fatal": fatal_count,
        "stop": stop_count,
    }

    return {
        "admissible": admissible,
        "blocking_pf_ids": blocking_pf_ids,
        "status": status,
        "eligibility": eligibility,
        "experiment_id": snapshot.get("experiment_id"),
        "timestamp": snapshot.get("preflight_timestamp"),
        "failure_summary": failure_summary,
    }


# =============================================================================
# PHASE III: GLOBAL HEALTH SUMMARY (Task 3)
# =============================================================================

def summarize_preflight_for_global_health(
    snapshot: dict[str, Any],
    drift_timeline: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    Summarize preflight state for global health dashboard integration.

    Produces a health-oriented summary suitable for system-wide monitoring,
    including drift status from historical data if available.

    Args:
        snapshot: Current snapshot from build_preflight_snapshot()
        drift_timeline: Optional drift timeline from build_preflight_drift_timeline()

    Returns:
        Dictionary with global health summary:
        - preflight_ok: bool - True if audit can proceed
        - drift_status: "stable" | "improving" | "degrading" | "unknown"
        - failure_hotspots: list of frequently failing check IDs
        - health_score: float 0.0-1.0 combining current state and drift
        - experiment_id: experiment identifier
        - last_check: timestamp of this snapshot
    """
    eligibility = snapshot.get("eligibility", "UNKNOWN")

    # Determine if preflight is OK (can proceed to audit)
    preflight_ok = eligibility in {"ELIGIBLE", "ELIGIBLE_WARNED", "ELIGIBLE_PARTIAL"}

    # Calculate current health score from snapshot
    counts = snapshot.get("counts", {})
    total = counts.get("total", 0)
    passed = counts.get("pass", 0)
    current_pass_rate = passed / total if total > 0 else 0.0

    # Failure type penalties
    failure_types = snapshot.get("failure_types", {})
    fatal_penalty = 0.3 * min(failure_types.get("fatal", 0), 1)  # Cap at 1
    stop_penalty = 0.2 * min(failure_types.get("stop", 0) / 3, 1)  # Cap at 3

    current_health = max(0.0, current_pass_rate - fatal_penalty - stop_penalty)

    # Determine drift status and failure hotspots
    if drift_timeline is None:
        drift_status = "unknown"
        failure_hotspots = []
        stability_factor = 1.0
    else:
        stability = drift_timeline.get("run_stability_index", 1.0)
        shifts = drift_timeline.get("eligibility_shifts", [])
        recurring = drift_timeline.get("recurring_failures", {})

        # Failure hotspots are recurring failures sorted by count
        failure_hotspots = sorted(
            recurring.keys(),
            key=lambda k: recurring[k],
            reverse=True
        )[:5]  # Top 5 hotspots

        # Determine drift direction from recent shifts
        if len(shifts) == 0:
            drift_status = "stable"
        else:
            # Look at the last shift
            last_shift = shifts[-1]
            old_status, new_status = last_shift[1], last_shift[2]

            # Eligibility ordering (best to worst)
            eligibility_order = [
                "ELIGIBLE", "ELIGIBLE_WARNED", "ELIGIBLE_PARTIAL",
                "BLOCKED_FIXABLE", "INADMISSIBLE"
            ]

            def rank(s: str) -> int:
                try:
                    return eligibility_order.index(s)
                except ValueError:
                    return 999

            if rank(new_status) < rank(old_status):
                drift_status = "improving"
            elif rank(new_status) > rank(old_status):
                drift_status = "degrading"
            else:
                drift_status = "stable"

        stability_factor = stability

    # Combined health score (current state + stability)
    health_score = round(0.7 * current_health + 0.3 * stability_factor, 3)

    return {
        "preflight_ok": preflight_ok,
        "drift_status": drift_status,
        "failure_hotspots": failure_hotspots,
        "health_score": health_score,
        "experiment_id": snapshot.get("experiment_id"),
        "last_check": snapshot.get("preflight_timestamp"),
        "current_eligibility": eligibility,
        "pass_rate": round(current_pass_rate, 3),
    }


# =============================================================================
# PHASE IV: PREFLIGHT RELEASE EVALUATOR (Task 1)
# =============================================================================

def evaluate_preflight_for_release(
    global_summary: dict[str, Any],
    drift_timeline: dict[str, Any]
) -> dict[str, Any]:
    """
    Evaluate preflight state for release readiness.

    Determines whether an experiment's preflight status is sufficient
    for release/audit gate approval. Combines current state with drift
    analysis to produce a release decision.

    Args:
        global_summary: Output from summarize_preflight_for_global_health()
        drift_timeline: Output from build_preflight_drift_timeline()

    Returns:
        Dictionary with release evaluation:
        - release_ok: bool - True if release can proceed
        - status: "OK" | "WARN" | "BLOCK"
        - blocking_reasons: list of reasons preventing release
        - confidence: float 0.0-1.0 based on stability and health
    """
    blocking_reasons = []

    # Check current preflight status
    preflight_ok = global_summary.get("preflight_ok", False)
    eligibility = global_summary.get("current_eligibility", "UNKNOWN")
    health_score = global_summary.get("health_score", 0.0)
    drift_status = global_summary.get("drift_status", "unknown")

    # Check drift timeline
    stability_index = drift_timeline.get("run_stability_index", 1.0)
    recurring_failures = drift_timeline.get("recurring_failures", {})
    eligibility_shifts = drift_timeline.get("eligibility_shifts", [])

    # === BLOCKING CONDITIONS ===

    # Block 1: Preflight not OK (BLOCKED_FIXABLE or INADMISSIBLE)
    if not preflight_ok:
        if eligibility == "INADMISSIBLE":
            blocking_reasons.append(f"FATAL: Experiment is INADMISSIBLE")
        elif eligibility == "BLOCKED_FIXABLE":
            blocking_reasons.append(f"STOP: Experiment has blocking issues requiring fixes")
        else:
            blocking_reasons.append(f"STOP: Preflight status is {eligibility}")

    # Block 2: Active degradation
    if drift_status == "degrading":
        blocking_reasons.append("DRIFT: Eligibility is actively degrading")

    # Block 3: Very low stability (high churn)
    if stability_index < 0.3:
        blocking_reasons.append(f"STABILITY: Run stability index {stability_index:.2f} is critically low (<0.3)")

    # Block 4: Chronic recurring failures (same check fails >3 times)
    chronic_failures = [cid for cid, count in recurring_failures.items() if count > 3]
    if chronic_failures:
        blocking_reasons.append(f"CHRONIC: {len(chronic_failures)} check(s) failing repeatedly: {chronic_failures[:3]}")

    # === WARNING CONDITIONS (not blocking, but noted) ===
    warnings = []

    # Warn 1: Health score below threshold
    if health_score < 0.7 and not blocking_reasons:
        warnings.append(f"Health score {health_score:.2f} is below recommended (0.7)")

    # Warn 2: Recent eligibility shifts
    if len(eligibility_shifts) > 0 and drift_status != "degrading":
        warnings.append(f"Eligibility has shifted {len(eligibility_shifts)} time(s) recently")

    # Warn 3: Moderate instability
    if 0.3 <= stability_index < 0.6 and stability_index >= 0.3:
        warnings.append(f"Stability index {stability_index:.2f} is moderate")

    # === DETERMINE STATUS ===
    if blocking_reasons:
        status = "BLOCK"
        release_ok = False
    elif warnings or eligibility in ("ELIGIBLE_WARNED", "ELIGIBLE_PARTIAL"):
        status = "WARN"
        release_ok = True
    else:
        status = "OK"
        release_ok = True

    # Calculate confidence based on health and stability
    confidence = round(0.5 * health_score + 0.5 * stability_index, 3)

    return {
        "release_ok": release_ok,
        "status": status,
        "blocking_reasons": blocking_reasons,
        "warnings": warnings,
        "confidence": confidence,
        "eligibility": eligibility,
    }


# =============================================================================
# PHASE IV: MAAS AUDIT READINESS ADAPTER (Task 2)
# =============================================================================

def summarize_preflight_for_audit_readiness(
    global_summary: dict[str, Any],
    drift_timeline: dict[str, Any]
) -> dict[str, Any]:
    """
    Summarize preflight state for MAAS audit readiness dashboard.

    Provides a focused view for audit readiness assessment, combining
    current state with drift analysis in a format suitable for MAAS
    dashboards and alerting.

    Args:
        global_summary: Output from summarize_preflight_for_global_health()
        drift_timeline: Output from build_preflight_drift_timeline()

    Returns:
        Dictionary with audit readiness summary:
        - audit_ready: bool - True if experiment is ready for audit
        - drift_status: "stable" | "improving" | "degrading" | "unknown"
        - recurring_failures: dict of check_id -> count
        - status: "OK" | "ATTENTION" | "BLOCK"
        - experiment_id: experiment identifier
        - health_score: current health score
        - stability_index: run stability index
    """
    preflight_ok = global_summary.get("preflight_ok", False)
    drift_status = global_summary.get("drift_status", "unknown")
    health_score = global_summary.get("health_score", 0.0)
    eligibility = global_summary.get("current_eligibility", "UNKNOWN")

    stability_index = drift_timeline.get("run_stability_index", 1.0)
    recurring_failures = drift_timeline.get("recurring_failures", {})

    # Determine audit readiness
    # Audit ready if: preflight OK AND not actively degrading AND stability acceptable
    audit_ready = (
        preflight_ok and
        drift_status != "degrading" and
        stability_index >= 0.3
    )

    # Determine status
    if not preflight_ok or eligibility in ("INADMISSIBLE", "BLOCKED_FIXABLE"):
        status = "BLOCK"
    elif drift_status == "degrading" or stability_index < 0.5 or len(recurring_failures) > 2:
        status = "ATTENTION"
    else:
        status = "OK"

    # Build attention reasons for ATTENTION status
    attention_reasons = []
    if drift_status == "degrading":
        attention_reasons.append("Eligibility is degrading")
    if stability_index < 0.5:
        attention_reasons.append(f"Stability index {stability_index:.2f} is low")
    if len(recurring_failures) > 2:
        attention_reasons.append(f"{len(recurring_failures)} recurring failures detected")

    return {
        "audit_ready": audit_ready,
        "drift_status": drift_status,
        "recurring_failures": recurring_failures,
        "status": status,
        "experiment_id": global_summary.get("experiment_id"),
        "health_score": health_score,
        "stability_index": stability_index,
        "current_eligibility": eligibility,
        "attention_reasons": attention_reasons if status == "ATTENTION" else [],
    }


# =============================================================================
# PHASE IV: DIRECTOR PREFLIGHT PANEL (Task 3)
# =============================================================================

def build_preflight_director_panel(
    global_summary: dict[str, Any],
    release_eval: dict[str, Any]
) -> dict[str, Any]:
    """
    Build a Director-facing preflight panel for dashboard display.

    Creates a concise, executive-summary style panel suitable for
    Director dashboards, with a traffic light status, key metrics,
    and a neutral headline summarizing the preflight posture.

    Args:
        global_summary: Output from summarize_preflight_for_global_health()
        release_eval: Output from evaluate_preflight_for_release()

    Returns:
        Dictionary with Director panel data:
        - status_light: "green" | "yellow" | "red"
        - current_eligibility: eligibility status string
        - pass_rate: float 0.0-1.0
        - headline: neutral sentence summarizing preflight posture
        - experiment_id: experiment identifier
        - health_score: current health score
        - release_ok: whether release can proceed
    """
    eligibility = global_summary.get("current_eligibility", "UNKNOWN")
    pass_rate = global_summary.get("pass_rate", 0.0)
    health_score = global_summary.get("health_score", 0.0)
    drift_status = global_summary.get("drift_status", "unknown")
    experiment_id = global_summary.get("experiment_id", "unknown")

    release_status = release_eval.get("status", "BLOCK")
    release_ok = release_eval.get("release_ok", False)
    blocking_reasons = release_eval.get("blocking_reasons", [])
    confidence = release_eval.get("confidence", 0.0)

    # Determine status light
    if release_status == "OK":
        status_light = "green"
    elif release_status == "WARN":
        status_light = "yellow"
    else:
        status_light = "red"

    # Build neutral headline
    headline = _build_preflight_headline(
        eligibility=eligibility,
        pass_rate=pass_rate,
        drift_status=drift_status,
        release_ok=release_ok,
        blocking_reasons=blocking_reasons,
    )

    return {
        "status_light": status_light,
        "current_eligibility": eligibility,
        "pass_rate": pass_rate,
        "headline": headline,
        "experiment_id": experiment_id,
        "health_score": health_score,
        "release_ok": release_ok,
        "confidence": confidence,
        "drift_status": drift_status,
    }


def _build_preflight_headline(
    eligibility: str,
    pass_rate: float,
    drift_status: str,
    release_ok: bool,
    blocking_reasons: list[str],
) -> str:
    """
    Build a neutral headline summarizing preflight posture.

    The headline is designed to be informative without being alarmist,
    suitable for executive dashboards.
    """
    pass_pct = int(pass_rate * 100)

    if eligibility == "ELIGIBLE" and release_ok:
        if drift_status == "stable":
            return f"Preflight checks passing ({pass_pct}%). Experiment is audit-eligible and stable."
        elif drift_status == "improving":
            return f"Preflight checks passing ({pass_pct}%). Experiment is audit-eligible with improving trend."
        else:
            return f"Preflight checks passing ({pass_pct}%). Experiment is audit-eligible."

    if eligibility == "ELIGIBLE_WARNED" and release_ok:
        return f"Preflight passing with warnings ({pass_pct}%). Audit may proceed with documented advisories."

    if eligibility == "ELIGIBLE_PARTIAL" and release_ok:
        return f"Preflight passing partially ({pass_pct}%). Partial audit may proceed on completed cycles."

    if eligibility == "BLOCKED_FIXABLE":
        return f"Preflight blocked ({pass_pct}% passing). Fixable issues require attention before audit."

    if eligibility == "INADMISSIBLE":
        return f"Preflight failed ({pass_pct}% passing). Experiment is permanently inadmissible for audit."

    if not release_ok and blocking_reasons:
        reason_summary = blocking_reasons[0].split(":")[0] if blocking_reasons else "issues"
        return f"Release blocked due to {reason_summary.lower()}. Review required before proceeding."

    # Fallback
    return f"Preflight status: {eligibility}. {pass_pct}% of checks passing."


# =============================================================================
# PHASE V: BUNDLE-PREFLIGHT JOINT VIEW (Task 1)
# =============================================================================

def build_preflight_bundle_joint_view(
    preflight_global_summary: dict[str, Any],
    bundle_evolution: dict[str, Any]
) -> dict[str, Any]:
    """
    Build a joint view combining preflight and bundle evolution state.

    This function integrates preflight status with bundle evolution data
    to provide a unified view for release gating. A BLOCK in either
    preflight or bundle evolution results in integration BLOCK.

    Args:
        preflight_global_summary: Output from summarize_preflight_for_global_health()
        bundle_evolution: Bundle evolution status dict with at minimum:
            - status: "OK" | "WARN" | "BLOCK"
            - bundle_id: str (optional)
            - stage: str (optional)

    Returns:
        Dictionary with joint view:
        - integration_ready: bool - True only if both preflight and bundle OK/WARN
        - joint_status: "OK" | "WARN" | "BLOCK"
        - reasons: list of reasons for current status
        - preflight_status: preflight status summary
        - bundle_status: bundle status summary
        - experiment_id: experiment identifier
    """
    reasons = []

    # Extract preflight state
    preflight_ok = preflight_global_summary.get("preflight_ok", False)
    preflight_eligibility = preflight_global_summary.get("current_eligibility", "UNKNOWN")
    preflight_health = preflight_global_summary.get("health_score", 0.0)
    drift_status = preflight_global_summary.get("drift_status", "unknown")
    experiment_id = preflight_global_summary.get("experiment_id")

    # Extract bundle state
    bundle_status = bundle_evolution.get("status", "UNKNOWN")
    bundle_id = bundle_evolution.get("bundle_id")
    bundle_stage = bundle_evolution.get("stage", "unknown")

    # Map preflight eligibility to status
    if preflight_eligibility in ("INADMISSIBLE", "BLOCKED_FIXABLE"):
        preflight_status = "BLOCK"
    elif preflight_eligibility in ("ELIGIBLE_WARNED", "ELIGIBLE_PARTIAL"):
        preflight_status = "WARN"
    elif preflight_eligibility == "ELIGIBLE":
        preflight_status = "OK"
    else:
        preflight_status = "BLOCK"  # Unknown defaults to BLOCK

    # === DETERMINE JOINT STATUS ===

    # Rule 1: Any BLOCK => integration BLOCK
    if preflight_status == "BLOCK":
        reasons.append(f"PREFLIGHT_BLOCK: Preflight status is {preflight_eligibility}")

    if bundle_status == "BLOCK":
        reasons.append(f"BUNDLE_BLOCK: Bundle evolution is blocked at stage '{bundle_stage}'")

    # Rule 2: Degrading drift is a concern
    if drift_status == "degrading":
        reasons.append("DRIFT_DEGRADING: Preflight eligibility is actively degrading")

    # Rule 3: Low health is a warning
    if preflight_health < 0.5 and preflight_status != "BLOCK":
        reasons.append(f"LOW_HEALTH: Preflight health score {preflight_health:.2f} is low")

    # Determine final joint status
    if preflight_status == "BLOCK" or bundle_status == "BLOCK":
        joint_status = "BLOCK"
        integration_ready = False
    elif preflight_status == "WARN" or bundle_status == "WARN" or drift_status == "degrading":
        joint_status = "WARN"
        integration_ready = True  # Can proceed with caution
    else:
        joint_status = "OK"
        integration_ready = True

    # Build summary for each component
    preflight_summary = {
        "status": preflight_status,
        "eligibility": preflight_eligibility,
        "health_score": preflight_health,
        "drift_status": drift_status,
    }

    bundle_summary = {
        "status": bundle_status,
        "bundle_id": bundle_id,
        "stage": bundle_stage,
    }

    return {
        "integration_ready": integration_ready,
        "joint_status": joint_status,
        "reasons": reasons,
        "preflight_status": preflight_summary,
        "bundle_status": bundle_summary,
        "experiment_id": experiment_id,
    }


# =============================================================================
# PHASE V: GLOBAL CONSOLE ADAPTER (Task 2)
# =============================================================================

def summarize_preflight_for_global_console(
    global_summary: dict[str, Any],
    release_eval: dict[str, Any]
) -> dict[str, Any]:
    """
    Summarize preflight state for global console display.

    Produces a minimal, standardized summary for global console integration,
    focusing on the key indicators needed for system-wide visibility.

    Args:
        global_summary: Output from summarize_preflight_for_global_health()
        release_eval: Output from evaluate_preflight_for_release()

    Returns:
        Dictionary with global console summary:
        - preflight_ok: bool - True if preflight passes
        - status_light: "green" | "yellow" | "red"
        - pass_rate: float 0.0-1.0
        - headline: concise status headline
        - experiment_id: experiment identifier
        - timestamp: last check timestamp
    """
    preflight_ok = global_summary.get("preflight_ok", False)
    eligibility = global_summary.get("current_eligibility", "UNKNOWN")
    pass_rate = global_summary.get("pass_rate", 0.0)
    health_score = global_summary.get("health_score", 0.0)
    drift_status = global_summary.get("drift_status", "unknown")
    experiment_id = global_summary.get("experiment_id")
    last_check = global_summary.get("last_check")

    release_status = release_eval.get("status", "BLOCK")
    release_ok = release_eval.get("release_ok", False)
    blocking_reasons = release_eval.get("blocking_reasons", [])

    # Determine status light
    if release_status == "OK":
        status_light = "green"
    elif release_status == "WARN":
        status_light = "yellow"
    else:
        status_light = "red"

    # Build concise headline for console
    pass_pct = int(pass_rate * 100)
    if release_ok and eligibility == "ELIGIBLE":
        headline = f"Preflight OK ({pass_pct}%)"
    elif release_ok and eligibility in ("ELIGIBLE_WARNED", "ELIGIBLE_PARTIAL"):
        headline = f"Preflight OK with warnings ({pass_pct}%)"
    elif eligibility == "BLOCKED_FIXABLE":
        headline = f"Preflight BLOCKED - fixable ({pass_pct}%)"
    elif eligibility == "INADMISSIBLE":
        headline = f"Preflight FAILED - inadmissible ({pass_pct}%)"
    else:
        headline = f"Preflight: {eligibility} ({pass_pct}%)"

    return {
        "preflight_ok": preflight_ok,
        "status_light": status_light,
        "pass_rate": pass_rate,
        "headline": headline,
        "experiment_id": experiment_id,
        "timestamp": last_check,
        "health_score": health_score,
        "drift_status": drift_status,
    }


# =============================================================================
# PHASE V: GOVERNANCE SIGNAL ADAPTER (Task 3)
# =============================================================================

def to_governance_signal(
    global_summary: dict[str, Any],
    release_eval: dict[str, Any]
) -> dict[str, Any]:
    """
    Convert preflight state to normalized governance signal.

    Produces a signal conforming to CLAUDE I's canonical GovernanceSignal schema
    from backend.analytics.governance_verifier, enabling preflight to participate
    in the global governance alignment synthesizer.

    Canonical GovernanceSignal Schema (from governance_verifier.py):
    - layer_name: "preflight" (identifies this signal source)
    - status: "OK" | "WARN" | "BLOCK"
    - blocking_rules: List[str] - rule IDs causing issues
    - blocking_rate: float in [0, 1] - severity indicator
    - headline: str - one-line neutral summary

    PRIORITIZATION IN META-GOVERNANCE:
    =========================================================================
    Preflight is designated as a CRITICAL LAYER in governance_verifier.py:

        DEFAULT_CRITICAL_LAYERS = frozenset({
            LAYER_REPLAY,       # A
            LAYER_HT,           # L
            LAYER_PREFLIGHT,    # J  <-- THIS LAYER
            LAYER_ADMISSIBILITY # O
        })

    This means:
    1. If preflight status == BLOCK, global promotion is BLOCKED regardless
       of what bundle, topology, or replay report.
    2. Preflight BLOCK cannot be overridden by upstream layers.
    3. evaluate_global_promotion() will fail if any critical layer is BLOCK.

    Relative Priority Order (enforced by meta-governance):
        1. CRITICAL LAYERS (replay, ht, preflight, admissibility) - HARD VETO
        2. WARNING LAYERS (topology, bundle, metrics, budget) - soft gate
        3. ADVISORY LAYERS (conjecture, security) - informational

    When to block:
    - INADMISSIBLE eligibility → BLOCK (permanent, cannot be fixed)
    - BLOCKED_FIXABLE eligibility → BLOCK (can be fixed, then retried)
    - Pass rate < threshold → BLOCK
    - Missing critical checks → BLOCK

    When to warn:
    - ELIGIBLE_WARNED eligibility → WARN
    - ELIGIBLE_PARTIAL eligibility → WARN
    - Degrading drift status → WARN

    When OK:
    - ELIGIBLE eligibility → OK
    - All checks pass → OK
    =========================================================================

    Args:
        global_summary: Output from summarize_preflight_for_global_health()
        release_eval: Output from evaluate_preflight_for_release()

    Returns:
        Dictionary conforming to canonical GovernanceSignal schema
    """
    preflight_ok = global_summary.get("preflight_ok", False)
    eligibility = global_summary.get("current_eligibility", "UNKNOWN")
    health_score = global_summary.get("health_score", 0.0)
    drift_status = global_summary.get("drift_status", "unknown")
    pass_rate = global_summary.get("pass_rate", 0.0)
    experiment_id = global_summary.get("experiment_id")
    last_check = global_summary.get("last_check")

    release_status = release_eval.get("status", "BLOCK")
    release_ok = release_eval.get("release_ok", False)
    confidence = release_eval.get("confidence", 0.0)
    blocking_reasons = release_eval.get("blocking_reasons", [])
    warnings = release_eval.get("warnings", [])

    # Map release status to governance status
    if release_status == "BLOCK":
        gov_status = "BLOCK"
    elif release_status == "WARN":
        gov_status = "WARN"
    else:
        gov_status = "OK"

    # Build blocking_rules list (canonical format)
    blocking_rules: list[str] = []
    if blocking_reasons:
        blocking_rules.extend(blocking_reasons)
    if warnings and gov_status == "WARN":
        blocking_rules.extend(warnings)

    # Add drift warning if degrading
    if drift_status == "degrading" and "DRIFT" not in str(blocking_rules):
        blocking_rules.append("DRIFT: Eligibility trend is degrading")

    # Compute blocking_rate from pass_rate
    # blocking_rate = 1 - pass_rate (higher blocking_rate = more severe)
    blocking_rate = max(0.0, min(1.0, 1.0 - pass_rate))

    # Generate headline (neutral, single-sentence)
    headline = _generate_preflight_headline(
        gov_status, eligibility, pass_rate, drift_status, experiment_id
    )

    # Return canonical GovernanceSignal schema
    return {
        "layer_name": "preflight",
        "status": gov_status,
        "blocking_rules": blocking_rules,
        "blocking_rate": blocking_rate,
        "headline": headline,
    }


def _generate_preflight_headline(
    status: str,
    eligibility: str,
    pass_rate: float,
    drift_status: str,
    experiment_id: Optional[str]
) -> str:
    """
    Generate a neutral, single-sentence headline for preflight governance signal.

    Args:
        status: "OK" | "WARN" | "BLOCK"
        eligibility: Eligibility status string
        pass_rate: Check pass rate (0.0-1.0)
        drift_status: "stable" | "improving" | "degrading" | "unknown"
        experiment_id: Experiment identifier (if any)

    Returns:
        Single-sentence headline string
    """
    exp_suffix = f" for {experiment_id}" if experiment_id else ""

    if status == "BLOCK":
        if eligibility == "INADMISSIBLE":
            return f"Preflight BLOCKED{exp_suffix}: Experiment inadmissible (pass rate {pass_rate:.0%})."
        else:
            return f"Preflight BLOCKED{exp_suffix}: Eligibility {eligibility}, pass rate {pass_rate:.0%}."
    elif status == "WARN":
        drift_note = " (degrading)" if drift_status == "degrading" else ""
        return f"Preflight WARNING{exp_suffix}: Eligibility {eligibility}{drift_note}, pass rate {pass_rate:.0%}."
    else:
        return f"Preflight OK{exp_suffix}: All checks passed ({pass_rate:.0%})."


# =============================================================================
# ELIGIBILITY CLASSIFICATION
# =============================================================================

def _classify_eligibility(checks: list[CheckResult]) -> tuple[EligibilityStatus, str]:
    """
    Classify eligibility based on check results.

    Implements the decision tree from U2_PRE_FLIGHT_AUDIT_PLAYBOOK.md.

    Returns (status, recommendation).
    """
    # Collect failure types
    has_fatal = any(c.failure_type == FailureType.FATAL for c in checks)
    has_stop = any(c.failure_type == FailureType.STOP for c in checks)
    has_warn = any(c.failure_type == FailureType.WARN for c in checks)

    # Check for partial audit (running experiment)
    is_partial = any(
        c.id == "PRE-6.25" and c.status == CheckStatus.WARN
        for c in checks
    )

    # Decision tree (explicit, no shortcuts)
    if has_fatal:
        return (
            EligibilityStatus.INADMISSIBLE,
            "DO_NOT_PROCEED - FATAL condition(s) detected. Experiment is permanently inadmissible."
        )

    if has_stop:
        return (
            EligibilityStatus.BLOCKED_FIXABLE,
            "FIX_REQUIRED - STOP condition(s) detected. Fix issues before audit."
        )

    if is_partial:
        if has_warn:
            return (
                EligibilityStatus.ELIGIBLE_PARTIAL,
                "PROCEED_PARTIAL_WARNED - Partial audit with warnings. Document in report."
            )
        return (
            EligibilityStatus.ELIGIBLE_PARTIAL,
            "PROCEED_PARTIAL - Partial audit (running experiment). Document cycle count."
        )

    if has_warn:
        return (
            EligibilityStatus.ELIGIBLE_WARNED,
            "PROCEED_TO_AUDIT - Warnings documented. Audit may proceed."
        )

    return (
        EligibilityStatus.ELIGIBLE,
        "PROCEED_TO_AUDIT - All checks passed. Experiment is audit-eligible."
    )


def _aggregate_gates(checks: list[CheckResult]) -> dict[str, GateResult]:
    """Aggregate check results by gate."""
    gates: dict[str, GateResult] = {}

    for check in checks:
        gate = check.gate
        if gate not in gates:
            gates[gate] = GateResult(gate=gate, status=CheckStatus.PASS)

        gates[gate].checks.append(check)

        if check.status == CheckStatus.PASS:
            gates[gate].passed += 1
        elif check.status == CheckStatus.WARN:
            gates[gate].warned += 1
        else:
            gates[gate].failed += 1

    # Determine gate status
    for gate in gates.values():
        if gate.failed > 0:
            gate.status = CheckStatus.FAIL
        elif gate.warned > 0:
            gate.status = CheckStatus.WARN
        else:
            gate.status = CheckStatus.PASS

    return gates


# =============================================================================
# MAIN PREFLIGHT FUNCTION
# =============================================================================

def run_preflight(
    exp_id: str,
    run_dir: Path,
    db_url: str,
    prereg_path: Path,
) -> PreFlightReport:
    """
    Run all pre-flight checks for a U2 experiment.

    Args:
        exp_id: Experiment ID to check
        run_dir: Root directory of the project
        db_url: Database connection URL
        prereg_path: Path to PREREG_UPLIFT_U2.yaml

    Returns:
        PreFlightReport with all check results
    """
    all_checks: list[CheckResult] = []
    timestamp = datetime.utcnow().isoformat() + "Z"

    # PRE-5: Database connectivity (must run first to get connection)
    pre5_results, db_conn = _check_PRE5_database_connectivity(db_url)
    all_checks.extend(pre5_results)

    if db_conn is None:
        # Can't continue without DB connection
        report = PreFlightReport(
            experiment_id=exp_id,
            preflight_timestamp=timestamp,
            eligibility_status=EligibilityStatus.BLOCKED_FIXABLE,
            checks=all_checks,
            stop_reasons=["Database connection failed - cannot run remaining checks"],
            recommendation="FIX_REQUIRED - Establish database connection first",
        )
        report.gates = _aggregate_gates(all_checks)
        return report

    try:
        # PRE-1: Registration & Identity
        pre1_results = _check_PRE1_registration_identity(exp_id, db_conn)
        all_checks.extend(pre1_results)

        # PRE-2: Preregistration Integrity
        pre2_results = _check_PRE2_preregistration_integrity(exp_id, prereg_path, db_conn)
        all_checks.extend(pre2_results)

        # PRE-3: Baseline Snapshot
        pre3_results = _check_PRE3_baseline_snapshot(exp_id, db_conn)
        all_checks.extend(pre3_results)

        # PRE-4: Log Directory Integrity
        pre4_results = _check_PRE4_log_directory_integrity(exp_id, run_dir)
        all_checks.extend(pre4_results)

        # PRE-6: State Eligibility
        pre6_results = _check_PRE6_state_eligibility(exp_id, db_conn)
        all_checks.extend(pre6_results)

    finally:
        if db_conn:
            db_conn.close()

    # Classify eligibility
    eligibility, recommendation = _classify_eligibility(all_checks)

    # Collect reasons
    fatal_reasons = [
        f"{c.id}: {c.message}" for c in all_checks
        if c.failure_type == FailureType.FATAL
    ]
    stop_reasons = [
        f"{c.id}: {c.message}" for c in all_checks
        if c.failure_type == FailureType.STOP
    ]
    warnings = [
        f"{c.id}: {c.message}" for c in all_checks
        if c.failure_type == FailureType.WARN
    ]

    # Build report
    report = PreFlightReport(
        experiment_id=exp_id,
        preflight_timestamp=timestamp,
        eligibility_status=eligibility,
        checks=all_checks,
        fatal_reasons=fatal_reasons,
        stop_reasons=stop_reasons,
        warnings=warnings,
        recommendation=recommendation,
    )
    report.gates = _aggregate_gates(all_checks)

    # Generate notes
    passed = sum(1 for c in all_checks if c.status == CheckStatus.PASS)
    total = len(all_checks)
    report.notes = f"{passed}/{total} checks passed. Status: {eligibility.value}"

    return report


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="U2 Pre-Flight Audit Tool - PHASE II",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exit Codes:
  0 - ELIGIBLE or ELIGIBLE_WARNED (may proceed to audit)
  1 - BLOCKED_FIXABLE or INADMISSIBLE (cannot proceed)
  2 - Invalid arguments or runtime error

Output Formats:
  default  - Full JSON report
  --bundle-format - Bundle-compatible stage result JSON
  --markdown - GitHub-flavored markdown summary
  --snapshot - Compact snapshot JSON for comparison

Examples:
  python u2_preflight.py --exp-id u2_test_001
  python u2_preflight.py --exp-id u2_test_001 --output report.json
  python u2_preflight.py --exp-id u2_test_001 --status-only
  python u2_preflight.py --exp-id u2_test_001 --bundle-format > stage_result.json
  python u2_preflight.py --exp-id u2_test_001 --markdown >> $GITHUB_STEP_SUMMARY
        """,
    )

    parser.add_argument(
        "--exp-id",
        required=True,
        help="Experiment ID to check",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path.cwd(),
        help="Project root directory (default: current directory)",
    )
    parser.add_argument(
        "--db-url",
        default=os.environ.get("DATABASE_URL", "postgresql://ml:mlpass@localhost:5432/mathledger"),
        help="Database connection URL (default: $DATABASE_URL)",
    )
    parser.add_argument(
        "--prereg",
        type=Path,
        default=None,
        help="Path to PREREG_UPLIFT_U2.yaml (default: config/PREREG_UPLIFT_U2.yaml)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file path (default: stdout)",
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Print only eligibility status",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress all output except errors",
    )
    parser.add_argument(
        "--bundle-format",
        action="store_true",
        help="Output in bundle-compatible stage result format",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Output as GitHub-flavored markdown",
    )
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="Output compact snapshot for comparison",
    )

    args = parser.parse_args()

    # Default prereg path
    if args.prereg is None:
        args.prereg = args.run_dir / "config" / "PREREG_UPLIFT_U2.yaml"

    try:
        report = run_preflight(
            exp_id=args.exp_id,
            run_dir=args.run_dir,
            db_url=args.db_url,
            prereg_path=args.prereg,
        )
    except Exception as e:
        if not args.quiet:
            print(f"ERROR: Pre-flight failed with exception: {e}", file=sys.stderr)
        return 2

    # Determine output content based on format flags
    if args.bundle_format:
        output_content = json.dumps(to_bundle_stage_result(report), indent=2, sort_keys=False)
    elif args.markdown:
        output_content = render_preflight_markdown(report)
    elif args.snapshot:
        output_content = json.dumps(build_preflight_snapshot(report), indent=2, sort_keys=False)
    else:
        output_content = report.to_json()

    # Output
    if args.status_only:
        if not args.quiet:
            print(report.eligibility_status.value)
    elif args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output_content)
        if not args.quiet:
            print(f"Report written to {args.output}")
            print(f"Status: {report.eligibility_status.value}")
    else:
        if not args.quiet:
            print(output_content)

    # Exit code
    if report.eligibility_status in (EligibilityStatus.ELIGIBLE,
                                      EligibilityStatus.ELIGIBLE_WARNED,
                                      EligibilityStatus.ELIGIBLE_PARTIAL):
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
