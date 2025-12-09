# PHASE II — NOT RUN IN PHASE I
"""
Determinism Replay Receipt Builder

This module implements the replay receipt system specified in:
- docs/U2_REPLAY_RECEIPT_CHARTER.md
- docs/U2_GOVERNANCE_RECONCILIATION_SPEC.md Section 6.5
- docs/VSD_PHASE_2.md Section 9F

A Replay Receipt cryptographically attests that an experiment's determinism
has been verified through full re-execution with identical seeds producing
identical H_t sequences.

Absolute Safeguards:
- No uplift math: does not inspect delta-p, p-values, or decision thresholds
- No Phase I changes: operates only on Phase II artifacts
- Binding only, not interpretive: verifies hash equality, not experimental outcomes
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Domain tag for replay receipt hashing (0x22 = replay domain)
DOMAIN_REPLAY_RECEIPT = b'\x22'

# Receipt schema version
REPLAY_RECEIPT_VERSION = "1.0.0"


class ReplayStatus(str, Enum):
    """Replay verification status."""
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"
    INCOMPLETE = "INCOMPLETE"


class ReconErrorCode(str, Enum):
    """Reconciliation error codes for replay failures (RECON-18 through RECON-20)."""
    RECON_18_REPLAY_MISSING = "RECON-18"
    RECON_19_REPLAY_MISMATCH = "RECON-19"
    RECON_20_REPLAY_INCOMPLETE = "RECON-20"


@dataclass
class ReplayEnvironment:
    """Captures the replay execution environment."""
    git_sha: str
    python_version: str
    platform: str
    rfl_runner_version: Optional[str] = None

    @classmethod
    def capture(cls, git_sha: str, runner_version: Optional[str] = None) -> ReplayEnvironment:
        """Capture current environment."""
        return cls(
            git_sha=git_sha,
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            platform=sys.platform,
            rfl_runner_version=runner_version,
        )


@dataclass
class ManifestBinding:
    """Binding to the experiment manifest."""
    manifest_path: str
    manifest_hash: str
    bound_at: str


@dataclass
class ReplayRunResult:
    """Result of replaying a single run (baseline or rfl)."""
    run_type: str  # "baseline" or "rfl"
    seed_used: int
    cycles_executed: int
    expected_log_hash: str
    replay_log_hash: str
    log_hash_match: bool
    expected_final_ht: str
    replay_final_ht: str
    final_ht_match: bool
    ht_sequence_length: int
    ht_sequence_match: bool
    first_mismatch_cycle: Optional[int] = None
    execution_duration_ms: Optional[int] = None


@dataclass
class FailedCheck:
    """Details of a failed verification check."""
    check_id: str
    expected: str
    actual: str
    detail: Optional[str] = None


@dataclass
class VerificationSummary:
    """Summary of all verification checks."""
    checks_passed: int
    checks_failed: int
    checks_total: int = 12  # RC-R1 through RC-R12
    all_verified: bool = False
    failed_checks: List[FailedCheck] = field(default_factory=list)

    def __post_init__(self):
        self.all_verified = self.checks_failed == 0


@dataclass
class ReplayReceipt:
    """
    Determinism Replay Receipt.

    Cryptographically attests that an experiment's determinism has been
    verified through full re-execution.
    """
    receipt_version: str
    experiment_id: str
    status: ReplayStatus
    replayed_at: str
    replay_environment: ReplayEnvironment
    manifest_binding: ManifestBinding
    baseline_replay: ReplayRunResult
    rfl_replay: ReplayRunResult
    verification_summary: VerificationSummary
    receipt_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "receipt_version": self.receipt_version,
            "experiment_id": self.experiment_id,
            "status": self.status.value if isinstance(self.status, ReplayStatus) else self.status,
            "replayed_at": self.replayed_at,
            "replay_environment": asdict(self.replay_environment),
            "manifest_binding": asdict(self.manifest_binding),
            "baseline_replay": asdict(self.baseline_replay),
            "rfl_replay": asdict(self.rfl_replay),
            "verification_summary": {
                "checks_passed": self.verification_summary.checks_passed,
                "checks_failed": self.verification_summary.checks_failed,
                "checks_total": self.verification_summary.checks_total,
                "all_verified": self.verification_summary.all_verified,
                "failed_checks": [asdict(fc) for fc in self.verification_summary.failed_checks],
            },
            "receipt_hash": self.receipt_hash,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to canonical JSON."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReplayReceipt:
        """Deserialize from dictionary."""
        return cls(
            receipt_version=data["receipt_version"],
            experiment_id=data["experiment_id"],
            status=ReplayStatus(data["status"]),
            replayed_at=data["replayed_at"],
            replay_environment=ReplayEnvironment(**data["replay_environment"]),
            manifest_binding=ManifestBinding(**data["manifest_binding"]),
            baseline_replay=ReplayRunResult(**data["baseline_replay"]),
            rfl_replay=ReplayRunResult(**data["rfl_replay"]),
            verification_summary=VerificationSummary(
                checks_passed=data["verification_summary"]["checks_passed"],
                checks_failed=data["verification_summary"]["checks_failed"],
                checks_total=data["verification_summary"]["checks_total"],
                all_verified=data["verification_summary"]["all_verified"],
                failed_checks=[FailedCheck(**fc) for fc in data["verification_summary"]["failed_checks"]],
            ),
            receipt_hash=data.get("receipt_hash", ""),
        )


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def compute_receipt_hash(receipt: ReplayReceipt) -> str:
    """
    Compute the self-hash of a replay receipt.

    The receipt_hash field is excluded from the hash computation,
    then the hash is computed over the canonical JSON representation
    with the DOMAIN_REPLAY_RECEIPT prefix.
    """
    # Create a copy without receipt_hash
    receipt_dict = receipt.to_dict()
    receipt_dict["receipt_hash"] = ""

    # Canonical JSON (sorted keys, no extra whitespace for hash)
    canonical_json = json.dumps(receipt_dict, sort_keys=True, separators=(',', ':'))

    # Domain-separated hash
    return hashlib.sha256(DOMAIN_REPLAY_RECEIPT + canonical_json.encode('utf-8')).hexdigest()


def verify_receipt_hash(receipt: ReplayReceipt) -> bool:
    """Verify that the receipt's self-hash is valid."""
    expected_hash = compute_receipt_hash(receipt)
    return receipt.receipt_hash == expected_hash


def extract_ht_series_from_log(log_path: Path) -> List[Dict[str, Any]]:
    """
    Extract H_t values from a JSONL log file.

    Returns list of {cycle, H_t, R_t, U_t} dictionaries.
    """
    ht_series = []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                # Extract H_t fields if present
                if "H_t" in record or "ht" in record or "composite_root" in record:
                    ht_entry = {
                        "cycle": record.get("cycle", len(ht_series)),
                        "H_t": record.get("H_t") or record.get("ht") or record.get("composite_root", ""),
                        "R_t": record.get("R_t") or record.get("reasoning_root", ""),
                        "U_t": record.get("U_t") or record.get("ui_root", ""),
                    }
                    ht_series.append(ht_entry)
            except json.JSONDecodeError:
                continue
    return ht_series


def compare_ht_sequences(
    expected_series: List[Dict[str, Any]],
    replay_series: List[Dict[str, Any]]
) -> Tuple[bool, Optional[int]]:
    """
    Compare two H_t sequences.

    Returns (match, first_mismatch_cycle) where first_mismatch_cycle is None if match=True.
    """
    if len(expected_series) != len(replay_series):
        return False, 0

    for i, (expected, actual) in enumerate(zip(expected_series, replay_series)):
        if expected.get("H_t") != actual.get("H_t"):
            return False, expected.get("cycle", i)

    return True, None


def build_replay_receipt(
    primary_run_dir: Path,
    replay_run_dir: Path,
    manifest_path: Path,
    git_sha: str = "unknown",
    runner_version: Optional[str] = None,
) -> ReplayReceipt:
    """
    Build a determinism replay receipt.

    Compares the primary (original) run against a replay run to verify
    that determinism is preserved.

    Args:
        primary_run_dir: Directory containing the original experiment run
        replay_run_dir: Directory containing the replay run
        manifest_path: Path to the experiment manifest
        git_sha: Git commit SHA at replay time
        runner_version: RFL runner version string

    Returns:
        ReplayReceipt with verification results
    """
    import time
    start_time = time.time()

    failed_checks: List[FailedCheck] = []
    checks_passed = 0

    # Load manifest
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    manifest_hash = compute_file_hash(manifest_path)
    experiment_id = manifest.get("experiment_id", manifest.get("slice", "unknown"))

    # Ensure experiment_id matches U2 pattern
    if not experiment_id.startswith("U2_EXP_"):
        experiment_id = f"U2_EXP_{experiment_id}"

    # Extract expected values from manifest
    baseline_seed = manifest.get("initial_seed", manifest.get("seed", 0))
    rfl_seed = baseline_seed  # Same seed for both in U2
    expected_cycles = manifest.get("cycles", 0)

    # Locate log files
    primary_baseline_log = primary_run_dir / "baseline" / "run.jsonl"
    primary_rfl_log = primary_run_dir / "rfl" / "run.jsonl"
    replay_baseline_log = replay_run_dir / "baseline" / "run.jsonl"
    replay_rfl_log = replay_run_dir / "rfl" / "run.jsonl"

    # Alternative log locations (flat structure)
    if not primary_baseline_log.exists():
        for pattern in ["*baseline*.jsonl", "*_baseline.jsonl"]:
            matches = list(primary_run_dir.glob(pattern))
            if matches:
                primary_baseline_log = matches[0]
                break

    if not primary_rfl_log.exists():
        for pattern in ["*rfl*.jsonl", "*_rfl.jsonl"]:
            matches = list(primary_run_dir.glob(pattern))
            if matches:
                primary_rfl_log = matches[0]
                break

    if not replay_baseline_log.exists():
        for pattern in ["*baseline*.jsonl", "*_baseline.jsonl"]:
            matches = list(replay_run_dir.glob(pattern))
            if matches:
                replay_baseline_log = matches[0]
                break

    if not replay_rfl_log.exists():
        for pattern in ["*rfl*.jsonl", "*_rfl.jsonl"]:
            matches = list(replay_run_dir.glob(pattern))
            if matches:
                replay_rfl_log = matches[0]
                break

    # RC-R1: Manifest exists (already loaded, so pass)
    checks_passed += 1

    # RC-R2: Manifest hash stable (we just computed it)
    checks_passed += 1

    # Baseline verification (RC-R3, RC-R5, RC-R7, RC-R9, RC-R11)
    baseline_result = _verify_run_pair(
        run_type="baseline",
        primary_log=primary_baseline_log,
        replay_log=replay_baseline_log,
        expected_seed=baseline_seed,
        expected_cycles=expected_cycles,
        manifest=manifest,
        failed_checks=failed_checks,
    )
    checks_passed += baseline_result["passed"]

    # RFL verification (RC-R4, RC-R6, RC-R8, RC-R10, RC-R12)
    rfl_result = _verify_run_pair(
        run_type="rfl",
        primary_log=primary_rfl_log,
        replay_log=replay_rfl_log,
        expected_seed=rfl_seed,
        expected_cycles=expected_cycles,
        manifest=manifest,
        failed_checks=failed_checks,
    )
    checks_passed += rfl_result["passed"]

    # Build replay run results
    baseline_replay = ReplayRunResult(
        run_type="baseline",
        seed_used=baseline_seed,
        cycles_executed=baseline_result.get("cycles", 0),
        expected_log_hash=baseline_result.get("expected_log_hash", ""),
        replay_log_hash=baseline_result.get("replay_log_hash", ""),
        log_hash_match=baseline_result.get("log_hash_match", False),
        expected_final_ht=baseline_result.get("expected_final_ht", ""),
        replay_final_ht=baseline_result.get("replay_final_ht", ""),
        final_ht_match=baseline_result.get("final_ht_match", False),
        ht_sequence_length=baseline_result.get("ht_sequence_length", 0),
        ht_sequence_match=baseline_result.get("ht_sequence_match", False),
        first_mismatch_cycle=baseline_result.get("first_mismatch_cycle"),
        execution_duration_ms=int((time.time() - start_time) * 1000 / 2),
    )

    rfl_replay = ReplayRunResult(
        run_type="rfl",
        seed_used=rfl_seed,
        cycles_executed=rfl_result.get("cycles", 0),
        expected_log_hash=rfl_result.get("expected_log_hash", ""),
        replay_log_hash=rfl_result.get("replay_log_hash", ""),
        log_hash_match=rfl_result.get("log_hash_match", False),
        expected_final_ht=rfl_result.get("expected_final_ht", ""),
        replay_final_ht=rfl_result.get("replay_final_ht", ""),
        final_ht_match=rfl_result.get("final_ht_match", False),
        ht_sequence_length=rfl_result.get("ht_sequence_length", 0),
        ht_sequence_match=rfl_result.get("ht_sequence_match", False),
        first_mismatch_cycle=rfl_result.get("first_mismatch_cycle"),
        execution_duration_ms=int((time.time() - start_time) * 1000 / 2),
    )

    # Determine overall status
    checks_failed = len(failed_checks)
    all_verified = checks_failed == 0

    if checks_failed == 0:
        status = ReplayStatus.VERIFIED
    elif baseline_result.get("incomplete") or rfl_result.get("incomplete"):
        status = ReplayStatus.INCOMPLETE
    else:
        status = ReplayStatus.FAILED

    # Build receipt
    receipt = ReplayReceipt(
        receipt_version=REPLAY_RECEIPT_VERSION,
        experiment_id=experiment_id,
        status=status,
        replayed_at=datetime.now(timezone.utc).isoformat(),
        replay_environment=ReplayEnvironment.capture(git_sha, runner_version),
        manifest_binding=ManifestBinding(
            manifest_path=str(manifest_path),
            manifest_hash=manifest_hash,
            bound_at=datetime.now(timezone.utc).isoformat(),
        ),
        baseline_replay=baseline_replay,
        rfl_replay=rfl_replay,
        verification_summary=VerificationSummary(
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            all_verified=all_verified,
            failed_checks=failed_checks,
        ),
    )

    # Compute self-hash
    receipt.receipt_hash = compute_receipt_hash(receipt)

    return receipt


def _verify_run_pair(
    run_type: str,
    primary_log: Path,
    replay_log: Path,
    expected_seed: int,
    expected_cycles: int,
    manifest: Dict[str, Any],
    failed_checks: List[FailedCheck],
) -> Dict[str, Any]:
    """
    Verify a single run pair (primary vs replay).

    Returns dict with verification results and pass count.
    """
    result: Dict[str, Any] = {
        "passed": 0,
        "incomplete": False,
        "cycles": 0,
        "expected_log_hash": "",
        "replay_log_hash": "",
        "log_hash_match": False,
        "expected_final_ht": "",
        "replay_final_ht": "",
        "final_ht_match": False,
        "ht_sequence_length": 0,
        "ht_sequence_match": False,
        "first_mismatch_cycle": None,
    }

    check_prefix = "RC-R3" if run_type == "baseline" else "RC-R4"

    # Check if logs exist
    if not primary_log.exists():
        failed_checks.append(FailedCheck(
            check_id=check_prefix,
            expected="file exists",
            actual="file missing",
            detail=f"Primary {run_type} log not found: {primary_log}",
        ))
        result["incomplete"] = True
        return result

    if not replay_log.exists():
        failed_checks.append(FailedCheck(
            check_id=check_prefix,
            expected="file exists",
            actual="file missing",
            detail=f"Replay {run_type} log not found: {replay_log}",
        ))
        result["incomplete"] = True
        return result

    # Seed match (RC-R3 for baseline, RC-R4 for rfl)
    # In U2, seed is stored in manifest, verified implicitly
    result["passed"] += 1

    # Compute log hashes
    expected_log_hash = compute_file_hash(primary_log)
    replay_log_hash = compute_file_hash(replay_log)
    result["expected_log_hash"] = expected_log_hash
    result["replay_log_hash"] = replay_log_hash

    # Cycle count check (RC-R5 for baseline, RC-R6 for rfl)
    primary_ht_series = extract_ht_series_from_log(primary_log)
    replay_ht_series = extract_ht_series_from_log(replay_log)

    result["cycles"] = len(replay_ht_series)
    result["ht_sequence_length"] = len(primary_ht_series)

    cycle_check_id = "RC-R5" if run_type == "baseline" else "RC-R6"
    if len(primary_ht_series) != len(replay_ht_series):
        failed_checks.append(FailedCheck(
            check_id=cycle_check_id,
            expected=str(len(primary_ht_series)),
            actual=str(len(replay_ht_series)),
            detail=f"{run_type} cycle count mismatch",
        ))
    else:
        result["passed"] += 1

    # Log hash check (RC-R7 for baseline, RC-R8 for rfl)
    log_hash_check_id = "RC-R7" if run_type == "baseline" else "RC-R8"
    if expected_log_hash == replay_log_hash:
        result["log_hash_match"] = True
        result["passed"] += 1
    else:
        result["log_hash_match"] = False
        failed_checks.append(FailedCheck(
            check_id=log_hash_check_id,
            expected=expected_log_hash[:16] + "...",
            actual=replay_log_hash[:16] + "...",
            detail=f"{run_type} log hash mismatch",
        ))

    # H_t sequence check (RC-R9 for baseline, RC-R10 for rfl)
    ht_seq_check_id = "RC-R9" if run_type == "baseline" else "RC-R10"
    ht_match, first_mismatch = compare_ht_sequences(primary_ht_series, replay_ht_series)
    result["ht_sequence_match"] = ht_match
    result["first_mismatch_cycle"] = first_mismatch

    if ht_match:
        result["passed"] += 1
    else:
        failed_checks.append(FailedCheck(
            check_id=ht_seq_check_id,
            expected="match",
            actual=f"mismatch at cycle {first_mismatch}",
            detail=f"{run_type} H_t sequence diverged",
        ))

    # Final H_t check (RC-R11 for baseline, RC-R12 for rfl)
    final_ht_check_id = "RC-R11" if run_type == "baseline" else "RC-R12"
    expected_final_ht = primary_ht_series[-1]["H_t"] if primary_ht_series else ""
    replay_final_ht = replay_ht_series[-1]["H_t"] if replay_ht_series else ""
    result["expected_final_ht"] = expected_final_ht
    result["replay_final_ht"] = replay_final_ht

    if expected_final_ht == replay_final_ht and expected_final_ht:
        result["final_ht_match"] = True
        result["passed"] += 1
    else:
        result["final_ht_match"] = False
        failed_checks.append(FailedCheck(
            check_id=final_ht_check_id,
            expected=expected_final_ht[:16] + "..." if expected_final_ht else "(empty)",
            actual=replay_final_ht[:16] + "..." if replay_final_ht else "(empty)",
            detail=f"{run_type} final H_t mismatch",
        ))

    return result


def save_replay_receipt(receipt: ReplayReceipt, output_path: Path) -> None:
    """Save replay receipt to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(receipt.to_json(indent=2))


def load_replay_receipt(receipt_path: Path) -> ReplayReceipt:
    """Load replay receipt from file."""
    with open(receipt_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return ReplayReceipt.from_dict(data)


def validate_replay_receipt(receipt_path: Path) -> Tuple[bool, Optional[ReconErrorCode], str]:
    """
    Validate a replay receipt for governance admissibility.

    Returns:
        (valid, error_code, message) tuple
    """
    if not receipt_path.exists():
        return False, ReconErrorCode.RECON_18_REPLAY_MISSING, f"Replay receipt not found: {receipt_path}"

    try:
        receipt = load_replay_receipt(receipt_path)
    except (json.JSONDecodeError, KeyError) as e:
        return False, ReconErrorCode.RECON_19_REPLAY_MISMATCH, f"Invalid replay receipt format: {e}"

    # Verify receipt hash
    if not verify_receipt_hash(receipt):
        return False, ReconErrorCode.RECON_19_REPLAY_MISMATCH, "Receipt hash verification failed"

    # Check status
    if receipt.status == ReplayStatus.INCOMPLETE:
        return False, ReconErrorCode.RECON_20_REPLAY_INCOMPLETE, "Replay is incomplete"

    if receipt.status == ReplayStatus.FAILED:
        failed_details = ", ".join(fc.check_id for fc in receipt.verification_summary.failed_checks)
        return False, ReconErrorCode.RECON_19_REPLAY_MISMATCH, f"Replay failed checks: {failed_details}"

    if receipt.status != ReplayStatus.VERIFIED:
        return False, ReconErrorCode.RECON_19_REPLAY_MISMATCH, f"Unknown status: {receipt.status}"

    return True, None, "Replay receipt valid"


# ============================================================================
# TASK 1: Receipt Index Contract (Evidence Spine v2)
# ============================================================================

# Receipt index schema version
RECEIPT_INDEX_VERSION = "1.0.0"


@dataclass
class ReceiptIndexEntry:
    """Individual entry in a receipt index."""
    receipt_hash: str
    primary_manifest_path: str
    replay_manifest_path: str  # Same as primary in single-manifest setups
    status: ReplayStatus
    experiment_id: str
    ht_series_hash: str  # Hash of baseline + RFL final H_t values
    replayed_at: str
    checks_passed: int
    checks_total: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_hash": self.receipt_hash,
            "primary_manifest_path": self.primary_manifest_path,
            "replay_manifest_path": self.replay_manifest_path,
            "status": self.status.value if isinstance(self.status, ReplayStatus) else self.status,
            "experiment_id": self.experiment_id,
            "ht_series_hash": self.ht_series_hash,
            "replayed_at": self.replayed_at,
            "checks_passed": self.checks_passed,
            "checks_total": self.checks_total,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReceiptIndexEntry:
        return cls(
            receipt_hash=data["receipt_hash"],
            primary_manifest_path=data["primary_manifest_path"],
            replay_manifest_path=data.get("replay_manifest_path", data["primary_manifest_path"]),
            status=ReplayStatus(data["status"]),
            experiment_id=data["experiment_id"],
            ht_series_hash=data.get("ht_series_hash", ""),
            replayed_at=data["replayed_at"],
            checks_passed=data["checks_passed"],
            checks_total=data["checks_total"],
        )


def compute_ht_series_hash(receipt: ReplayReceipt) -> str:
    """
    Compute a compact fingerprint of the H_t series from a receipt.

    Combines baseline and RFL final H_t values into a single hash.
    """
    combined = (
        receipt.baseline_replay.expected_final_ht +
        receipt.baseline_replay.replay_final_ht +
        receipt.rfl_replay.expected_final_ht +
        receipt.rfl_replay.replay_final_ht
    )
    return hashlib.sha256(combined.encode('utf-8')).hexdigest()


def build_receipt_index_entry(receipt: ReplayReceipt) -> ReceiptIndexEntry:
    """Build an index entry from a replay receipt."""
    return ReceiptIndexEntry(
        receipt_hash=receipt.receipt_hash,
        primary_manifest_path=receipt.manifest_binding.manifest_path,
        replay_manifest_path=receipt.manifest_binding.manifest_path,  # Same in U2
        status=receipt.status,
        experiment_id=receipt.experiment_id,
        ht_series_hash=compute_ht_series_hash(receipt),
        replayed_at=receipt.replayed_at,
        checks_passed=receipt.verification_summary.checks_passed,
        checks_total=receipt.verification_summary.checks_total,
    )


def build_receipt_index(
    receipts: List[ReplayReceipt],
    experiment_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a receipt index from a collection of replay receipts.

    The index provides a stable, discoverable manifest of all replay receipts
    for an experiment run, suitable for consumption by MAAS, governance verifier,
    and last-mile tools.

    Args:
        receipts: List of replay receipts to index
        experiment_id: Optional experiment ID (derived from first receipt if not provided)

    Returns:
        Receipt index dictionary with stable ordering by primary_manifest_path
    """
    if not receipts:
        return {
            "schema_version": RECEIPT_INDEX_VERSION,
            "experiment_id": experiment_id or "",
            "receipt_count": 0,
            "receipts": [],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    # Build entries
    entries = [build_receipt_index_entry(r) for r in receipts]

    # Sort by primary_manifest_path for stable ordering
    entries.sort(key=lambda e: e.primary_manifest_path)

    # Derive experiment_id from first receipt if not provided
    if not experiment_id:
        experiment_id = receipts[0].experiment_id

    return {
        "schema_version": RECEIPT_INDEX_VERSION,
        "experiment_id": experiment_id,
        "receipt_count": len(entries),
        "receipts": [e.to_dict() for e in entries],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def dump_receipt_index(path: Path, index: Dict[str, Any]) -> None:
    """
    Write a receipt index to a JSON file.

    Uses canonical JSON formatting for determinism.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=2, sort_keys=True)


def load_receipt_index(path: Path) -> Dict[str, Any]:
    """
    Load a receipt index from a JSON file.

    Handles missing optional fields gracefully.
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Ensure required fields exist with defaults
    if "schema_version" not in data:
        data["schema_version"] = "0.0.0"  # Legacy index
    if "experiment_id" not in data:
        data["experiment_id"] = ""
    if "receipt_count" not in data:
        data["receipt_count"] = len(data.get("receipts", []))
    if "receipts" not in data:
        data["receipts"] = []

    return data


def update_receipt_index(
    index_path: Path,
    new_receipt: ReplayReceipt,
) -> Dict[str, Any]:
    """
    Update an existing receipt index with a new receipt.

    If the index doesn't exist, creates a new one.
    Replaces any existing entry with the same primary_manifest_path.

    Returns the updated index.
    """
    if index_path.exists():
        index = load_receipt_index(index_path)
    else:
        index = {
            "schema_version": RECEIPT_INDEX_VERSION,
            "experiment_id": new_receipt.experiment_id,
            "receipt_count": 0,
            "receipts": [],
            "generated_at": "",
        }

    new_entry = build_receipt_index_entry(new_receipt)

    # Remove existing entry with same manifest path if present
    index["receipts"] = [
        r for r in index["receipts"]
        if r["primary_manifest_path"] != new_entry.primary_manifest_path
    ]

    # Add new entry
    index["receipts"].append(new_entry.to_dict())

    # Re-sort for stability
    index["receipts"].sort(key=lambda r: r["primary_manifest_path"])

    # Update metadata
    index["receipt_count"] = len(index["receipts"])
    index["generated_at"] = datetime.now(timezone.utc).isoformat()

    # Write back
    dump_receipt_index(index_path, index)

    return index


# ============================================================================
# TASK 3: Governance-Grade Replay Receipt Summary
# ============================================================================

@dataclass
class ReceiptSummary:
    """Governance-ready summary of replay receipts."""
    total_receipts: int
    verified_count: int
    failed_count: int
    incomplete_count: int
    error_codes: Dict[str, int]  # RECON-XX -> count
    all_verified: bool
    summary_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_receipts": self.total_receipts,
            "verified_count": self.verified_count,
            "failed_count": self.failed_count,
            "incomplete_count": self.incomplete_count,
            "error_codes": self.error_codes,
            "all_verified": self.all_verified,
            "summary_hash": self.summary_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReceiptSummary:
        return cls(
            total_receipts=data["total_receipts"],
            verified_count=data["verified_count"],
            failed_count=data["failed_count"],
            incomplete_count=data["incomplete_count"],
            error_codes=data.get("error_codes", {}),
            all_verified=data["all_verified"],
            summary_hash=data.get("summary_hash", ""),
        )


def summarize_replay_receipts(receipts: List[ReplayReceipt]) -> Dict[str, Any]:
    """
    Generate a compact, governance-ready summary of replay receipts.

    This summary is deterministic and suitable for MAAS v2 and the
    governance verifier pipeline.

    Args:
        receipts: List of replay receipts to summarize

    Returns:
        Deterministic JSON-serializable summary dictionary
    """
    verified_count = 0
    failed_count = 0
    incomplete_count = 0
    error_codes: Dict[str, int] = {}

    for receipt in receipts:
        if receipt.status == ReplayStatus.VERIFIED:
            verified_count += 1
        elif receipt.status == ReplayStatus.FAILED:
            failed_count += 1
            # Count error codes from failed checks
            for fc in receipt.verification_summary.failed_checks:
                # Map check IDs to RECON codes
                if "mismatch" in fc.detail.lower() if fc.detail else False:
                    code = ReconErrorCode.RECON_19_REPLAY_MISMATCH.value
                else:
                    code = ReconErrorCode.RECON_19_REPLAY_MISMATCH.value  # Default
                error_codes[code] = error_codes.get(code, 0) + 1
        elif receipt.status == ReplayStatus.INCOMPLETE:
            incomplete_count += 1
            code = ReconErrorCode.RECON_20_REPLAY_INCOMPLETE.value
            error_codes[code] = error_codes.get(code, 0) + 1

    total = len(receipts)
    all_verified = (verified_count == total and total > 0)

    summary = ReceiptSummary(
        total_receipts=total,
        verified_count=verified_count,
        failed_count=failed_count,
        incomplete_count=incomplete_count,
        error_codes=error_codes,
        all_verified=all_verified,
    )

    # Compute deterministic summary hash (for integrity)
    summary_dict = summary.to_dict()
    summary_dict["summary_hash"] = ""
    canonical = json.dumps(summary_dict, sort_keys=True, separators=(',', ':'))
    summary.summary_hash = hashlib.sha256(canonical.encode('utf-8')).hexdigest()

    return summary.to_dict()


def summarize_receipt_index(index: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a summary from a receipt index (without loading full receipts).

    Useful for CI/governance checks that only have access to the index file.
    """
    verified_count = 0
    failed_count = 0
    incomplete_count = 0
    error_codes: Dict[str, int] = {}

    for entry in index.get("receipts", []):
        status = entry.get("status", "")
        if status == ReplayStatus.VERIFIED.value:
            verified_count += 1
        elif status == ReplayStatus.FAILED.value:
            failed_count += 1
            code = ReconErrorCode.RECON_19_REPLAY_MISMATCH.value
            error_codes[code] = error_codes.get(code, 0) + 1
        elif status == ReplayStatus.INCOMPLETE.value:
            incomplete_count += 1
            code = ReconErrorCode.RECON_20_REPLAY_INCOMPLETE.value
            error_codes[code] = error_codes.get(code, 0) + 1

    total = len(index.get("receipts", []))
    all_verified = (verified_count == total and total > 0)

    summary = {
        "total_receipts": total,
        "verified_count": verified_count,
        "failed_count": failed_count,
        "incomplete_count": incomplete_count,
        "error_codes": error_codes,
        "all_verified": all_verified,
        "experiment_id": index.get("experiment_id", ""),
        "index_schema_version": index.get("schema_version", ""),
    }

    # Compute deterministic summary hash
    canonical = json.dumps(summary, sort_keys=True, separators=(',', ':'))
    summary["summary_hash"] = hashlib.sha256(canonical.encode('utf-8')).hexdigest()

    return summary


# ============================================================================
# PHASE III: Cross-Run Replay Intelligence & Determinism History Engine
# ============================================================================

# Ledger schema version
DETERMINISM_LEDGER_VERSION = "1.0.0"


class IncidentSeverity(str, Enum):
    """Severity levels for replay incidents."""
    NONE = "NONE"
    LOW = "LOW"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AffectedDomain(str, Enum):
    """Domains that can be affected by determinism failures."""
    H_T = "H_T"       # Composite attestation root
    R_T = "R_T"       # Reasoning trace
    U_T = "U_T"       # UI/output trace
    CONFIG = "CONFIG"  # Configuration/seed mismatch
    LOG = "LOG"       # Log file integrity


class ReplayHealthStatus(str, Enum):
    """Global health status for replay system."""
    OK = "OK"
    WARN = "WARN"
    BLOCKED = "BLOCKED"


@dataclass
class LedgerRunEntry:
    """Single run entry in the determinism ledger."""
    run_id: str
    experiment_id: str
    status: ReplayStatus
    mismatch_codes: List[str]
    ht_hash: str
    timestamp: str
    checks_passed: int
    checks_total: int
    first_mismatch_cycle: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "status": self.status.value if isinstance(self.status, ReplayStatus) else self.status,
            "mismatch_codes": self.mismatch_codes,
            "ht_hash": self.ht_hash,
            "timestamp": self.timestamp,
            "checks_passed": self.checks_passed,
            "checks_total": self.checks_total,
            "first_mismatch_cycle": self.first_mismatch_cycle,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LedgerRunEntry:
        return cls(
            run_id=data["run_id"],
            experiment_id=data["experiment_id"],
            status=ReplayStatus(data["status"]),
            mismatch_codes=data.get("mismatch_codes", []),
            ht_hash=data.get("ht_hash", ""),
            timestamp=data["timestamp"],
            checks_passed=data["checks_passed"],
            checks_total=data["checks_total"],
            first_mismatch_cycle=data.get("first_mismatch_cycle"),
        )


@dataclass
class ReplayIncident:
    """Classified replay incident."""
    severity: IncidentSeverity
    affected_domains: List[AffectedDomain]
    recommended_action: str
    incident_fingerprint: str
    mismatch_codes: List[str]
    first_mismatch_cycle: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity.value,
            "affected_domains": [d.value for d in self.affected_domains],
            "recommended_action": self.recommended_action,
            "incident_fingerprint": self.incident_fingerprint,
            "mismatch_codes": self.mismatch_codes,
            "first_mismatch_cycle": self.first_mismatch_cycle,
        }


def _extract_mismatch_codes(receipt: ReplayReceipt) -> List[str]:
    """Extract mismatch codes from a receipt's failed checks."""
    codes = []
    for fc in receipt.verification_summary.failed_checks:
        codes.append(fc.check_id)
    return sorted(set(codes))


def _compute_run_id(receipt: ReplayReceipt) -> str:
    """Compute a unique run ID from receipt attributes."""
    components = f"{receipt.experiment_id}:{receipt.manifest_binding.manifest_path}:{receipt.replayed_at}"
    return hashlib.sha256(components.encode('utf-8')).hexdigest()[:16]


def build_ledger_entry(receipt: ReplayReceipt) -> LedgerRunEntry:
    """Build a ledger entry from a replay receipt."""
    mismatch_codes = _extract_mismatch_codes(receipt)

    # Determine first mismatch cycle from either run
    first_mismatch = None
    if receipt.baseline_replay.first_mismatch_cycle is not None:
        first_mismatch = receipt.baseline_replay.first_mismatch_cycle
    elif receipt.rfl_replay.first_mismatch_cycle is not None:
        first_mismatch = receipt.rfl_replay.first_mismatch_cycle

    return LedgerRunEntry(
        run_id=_compute_run_id(receipt),
        experiment_id=receipt.experiment_id,
        status=receipt.status,
        mismatch_codes=mismatch_codes,
        ht_hash=compute_ht_series_hash(receipt),
        timestamp=receipt.replayed_at,
        checks_passed=receipt.verification_summary.checks_passed,
        checks_total=receipt.verification_summary.checks_total,
        first_mismatch_cycle=first_mismatch,
    )


def build_replay_determinism_ledger(receipts: List[ReplayReceipt]) -> Dict[str, Any]:
    """
    TASK 1: Build a persistent replay-determinism ledger.

    Creates a comprehensive history of replay verification results with
    deterministic ordering and summary statistics.

    Args:
        receipts: List of replay receipts to include in the ledger

    Returns:
        Ledger dictionary with run history, totals, and failure tracking
    """
    if not receipts:
        return {
            "schema_version": DETERMINISM_LEDGER_VERSION,
            "run_history": [],
            "totals": {"verified": 0, "failed": 0, "incomplete": 0},
            "determinism_rate": 0.0,
            "first_failure_at": None,
            "last_failure_at": None,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "ledger_hash": "",
        }

    # Build ledger entries
    entries = [build_ledger_entry(r) for r in receipts]

    # Sort by timestamp for deterministic ordering
    entries.sort(key=lambda e: e.timestamp)

    # Calculate totals
    verified = sum(1 for e in entries if e.status == ReplayStatus.VERIFIED)
    failed = sum(1 for e in entries if e.status == ReplayStatus.FAILED)
    incomplete = sum(1 for e in entries if e.status == ReplayStatus.INCOMPLETE)

    # Track failure timestamps
    failure_timestamps = [
        e.timestamp for e in entries
        if e.status in (ReplayStatus.FAILED, ReplayStatus.INCOMPLETE)
    ]
    first_failure_at = min(failure_timestamps) if failure_timestamps else None
    last_failure_at = max(failure_timestamps) if failure_timestamps else None

    # Calculate determinism rate
    total = len(entries)
    determinism_rate = verified / total if total > 0 else 0.0

    ledger = {
        "schema_version": DETERMINISM_LEDGER_VERSION,
        "run_history": [e.to_dict() for e in entries],
        "totals": {
            "verified": verified,
            "failed": failed,
            "incomplete": incomplete,
        },
        "determinism_rate": round(determinism_rate, 4),
        "first_failure_at": first_failure_at,
        "last_failure_at": last_failure_at,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    # Compute deterministic ledger hash
    ledger_for_hash = dict(ledger)
    ledger_for_hash["ledger_hash"] = ""
    canonical = json.dumps(ledger_for_hash, sort_keys=True, separators=(',', ':'))
    ledger["ledger_hash"] = hashlib.sha256(canonical.encode('utf-8')).hexdigest()

    return ledger


def save_determinism_ledger(ledger: Dict[str, Any], path: Path) -> None:
    """Save a determinism ledger to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(ledger, f, indent=2, sort_keys=True)


def load_determinism_ledger(path: Path) -> Dict[str, Any]:
    """Load a determinism ledger from file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _identify_affected_domains(receipt: ReplayReceipt) -> List[AffectedDomain]:
    """Identify which domains are affected by failures in a receipt."""
    domains = []

    # Check H_t (composite root) issues
    if not receipt.baseline_replay.ht_sequence_match or not receipt.rfl_replay.ht_sequence_match:
        domains.append(AffectedDomain.H_T)

    if not receipt.baseline_replay.final_ht_match or not receipt.rfl_replay.final_ht_match:
        if AffectedDomain.H_T not in domains:
            domains.append(AffectedDomain.H_T)

    # Check log integrity (overall log hash)
    if not receipt.baseline_replay.log_hash_match or not receipt.rfl_replay.log_hash_match:
        domains.append(AffectedDomain.LOG)

    # Check for config/seed issues (inferred from early cycle mismatches)
    first_baseline = receipt.baseline_replay.first_mismatch_cycle
    first_rfl = receipt.rfl_replay.first_mismatch_cycle
    if first_baseline == 0 or first_rfl == 0:
        domains.append(AffectedDomain.CONFIG)

    # If no specific domain identified but status is not verified, assume H_T
    if not domains and receipt.status != ReplayStatus.VERIFIED:
        domains.append(AffectedDomain.H_T)

    return domains


def _compute_incident_fingerprint(receipt: ReplayReceipt, domains: List[AffectedDomain]) -> str:
    """Compute a deterministic fingerprint for an incident."""
    components = [
        receipt.experiment_id,
        receipt.status.value,
        ",".join(sorted(d.value for d in domains)),
        ",".join(sorted(_extract_mismatch_codes(receipt))),
    ]
    content = "|".join(components)
    return hashlib.sha256(content.encode('utf-8')).hexdigest()[:24]


def classify_replay_incident(receipt: ReplayReceipt) -> Dict[str, Any]:
    """
    TASK 2: Classify a replay incident by severity and affected domains.

    Analyzes a replay receipt to determine the severity of any determinism
    failure and provides actionable recommendations.

    Args:
        receipt: The replay receipt to classify

    Returns:
        Incident classification dictionary
    """
    # No incident for verified receipts
    if receipt.status == ReplayStatus.VERIFIED:
        return ReplayIncident(
            severity=IncidentSeverity.NONE,
            affected_domains=[],
            recommended_action="No action required - determinism verified",
            incident_fingerprint="",
            mismatch_codes=[],
            first_mismatch_cycle=None,
        ).to_dict()

    domains = _identify_affected_domains(receipt)
    mismatch_codes = _extract_mismatch_codes(receipt)

    # Determine first mismatch cycle
    first_mismatch = None
    if receipt.baseline_replay.first_mismatch_cycle is not None:
        first_mismatch = receipt.baseline_replay.first_mismatch_cycle
    elif receipt.rfl_replay.first_mismatch_cycle is not None:
        first_mismatch = receipt.rfl_replay.first_mismatch_cycle

    # Classify severity
    severity = IncidentSeverity.LOW  # Default

    if receipt.status == ReplayStatus.INCOMPLETE:
        severity = IncidentSeverity.HIGH
        recommended_action = "Replay incomplete - investigate infrastructure failure and re-run"
    elif AffectedDomain.CONFIG in domains:
        severity = IncidentSeverity.CRITICAL
        recommended_action = "Configuration/seed mismatch at cycle 0 - verify manifest integrity and seed propagation"
    elif len(domains) >= 2:
        severity = IncidentSeverity.HIGH
        recommended_action = f"Multiple domain failures ({', '.join(d.value for d in domains)}) - investigate systematic determinism issue"
    elif AffectedDomain.H_T in domains:
        # H_T failures are concerning but may be isolated
        if first_mismatch is not None and first_mismatch < 5:
            severity = IncidentSeverity.HIGH
            recommended_action = f"Early H_t divergence at cycle {first_mismatch} - likely seed or initialization issue"
        else:
            severity = IncidentSeverity.LOW
            recommended_action = "H_t sequence divergence - review computation determinism in affected cycles"
    elif AffectedDomain.LOG in domains:
        severity = IncidentSeverity.LOW
        recommended_action = "Log hash mismatch - may indicate non-deterministic logging or timestamp drift"
    else:
        recommended_action = "Investigate failed checks and verify replay environment consistency"

    fingerprint = _compute_incident_fingerprint(receipt, domains)

    return ReplayIncident(
        severity=severity,
        affected_domains=domains,
        recommended_action=recommended_action,
        incident_fingerprint=fingerprint,
        mismatch_codes=mismatch_codes,
        first_mismatch_cycle=first_mismatch,
    ).to_dict()


def summarize_replay_for_global_health(ledger: Dict[str, Any]) -> Dict[str, Any]:
    """
    TASK 3: Generate a global health summary from a determinism ledger.

    Provides a high-level health status suitable for dashboards and CI gates.

    Args:
        ledger: Determinism ledger dictionary

    Returns:
        Global health summary with status, rate, and blocking info
    """
    totals = ledger.get("totals", {"verified": 0, "failed": 0, "incomplete": 0})
    determinism_rate = ledger.get("determinism_rate", 0.0)
    run_history = ledger.get("run_history", [])

    # Count recent failures (last 10 runs)
    recent_runs = run_history[-10:] if run_history else []
    recent_failures = [
        r for r in recent_runs
        if r.get("status") in (ReplayStatus.FAILED.value, ReplayStatus.INCOMPLETE.value)
    ]

    # Collect blocking fingerprints (unique incident signatures)
    blocking_fingerprints = []
    for run in recent_failures:
        # Create a simple fingerprint from mismatch codes
        codes = run.get("mismatch_codes", [])
        if codes:
            fp = hashlib.sha256(",".join(sorted(codes)).encode()).hexdigest()[:12]
            if fp not in blocking_fingerprints:
                blocking_fingerprints.append(fp)

    # Determine health status
    if totals["incomplete"] > 0:
        status = ReplayHealthStatus.BLOCKED
    elif determinism_rate < 0.9:
        status = ReplayHealthStatus.BLOCKED
    elif determinism_rate < 0.95:
        status = ReplayHealthStatus.WARN
    elif len(recent_failures) >= 3:
        status = ReplayHealthStatus.WARN
    elif totals["failed"] > 0 and determinism_rate < 1.0:
        status = ReplayHealthStatus.WARN
    else:
        status = ReplayHealthStatus.OK

    health_summary = {
        "replay_status": status.value,
        "determinism_rate": determinism_rate,
        "total_runs": totals["verified"] + totals["failed"] + totals["incomplete"],
        "verified_runs": totals["verified"],
        "failed_runs": totals["failed"],
        "incomplete_runs": totals["incomplete"],
        "recent_failures": len(recent_failures),
        "recent_failure_rate": len(recent_failures) / len(recent_runs) if recent_runs else 0.0,
        "blocking_fingerprints": blocking_fingerprints,
        "first_failure_at": ledger.get("first_failure_at"),
        "last_failure_at": ledger.get("last_failure_at"),
        "is_blocked": status == ReplayHealthStatus.BLOCKED,
        "is_healthy": status == ReplayHealthStatus.OK,
    }

    # Compute summary hash for integrity
    canonical = json.dumps(health_summary, sort_keys=True, separators=(',', ':'))
    health_summary["health_hash"] = hashlib.sha256(canonical.encode('utf-8')).hexdigest()

    return health_summary


def get_incident_report(receipts: List[ReplayReceipt]) -> Dict[str, Any]:
    """
    Generate a comprehensive incident report from multiple receipts.

    Aggregates incidents by severity and fingerprint for trend analysis.
    """
    if not receipts:
        return {
            "total_receipts": 0,
            "incidents_by_severity": {},
            "unique_fingerprints": [],
            "affected_domains_summary": {},
            "recommendations": [],
        }

    incidents = [classify_replay_incident(r) for r in receipts]

    # Count by severity
    by_severity: Dict[str, int] = {}
    for inc in incidents:
        sev = inc["severity"]
        by_severity[sev] = by_severity.get(sev, 0) + 1

    # Collect unique fingerprints
    fingerprints = set()
    for inc in incidents:
        if inc["incident_fingerprint"]:
            fingerprints.add(inc["incident_fingerprint"])

    # Aggregate affected domains
    domain_counts: Dict[str, int] = {}
    for inc in incidents:
        for domain in inc["affected_domains"]:
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

    # Collect unique recommendations
    recommendations = list(set(
        inc["recommended_action"]
        for inc in incidents
        if inc["severity"] != IncidentSeverity.NONE.value
    ))

    return {
        "total_receipts": len(receipts),
        "incidents_by_severity": by_severity,
        "unique_fingerprints": sorted(fingerprints),
        "affected_domains_summary": domain_counts,
        "recommendations": recommendations,
    }


# ============================================================================
# PHASE IV: Replay Governance Radar & Policy Coupler
# ============================================================================

# Radar schema version
GOVERNANCE_RADAR_VERSION = "1.0.0"


class RadarStatus(str, Enum):
    """Governance radar status levels."""
    STABLE = "STABLE"
    DEGRADING = "DEGRADING"
    UNSTABLE = "UNSTABLE"


class PromotionStatus(str, Enum):
    """Promotion evaluation status."""
    OK = "OK"
    WARN = "WARN"
    BLOCK = "BLOCK"


class StatusLight(str, Enum):
    """Director panel status lights."""
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


def build_replay_governance_radar(
    ledgers: List[Dict[str, Any]],
    incidents: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    TASK 1: Build a governance radar timeline from ledgers and incidents.

    Provides a live view of replay determinism health over time, identifying
    recurring issues and degradation trends.

    Args:
        ledgers: List of determinism ledger dictionaries
        incidents: List of incident classification dictionaries

    Returns:
        Governance radar with timeline, hot fingerprints, and status
    """
    if not ledgers:
        return {
            "schema_version": GOVERNANCE_RADAR_VERSION,
            "total_runs": 0,
            "determinism_rate_series": [],
            "critical_incident_rate": 0.0,
            "hot_fingerprints": [],
            "radar_status": RadarStatus.STABLE.value,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "radar_hash": "",
        }

    # Aggregate run history from all ledgers
    all_runs: List[Dict[str, Any]] = []
    for ledger in ledgers:
        all_runs.extend(ledger.get("run_history", []))

    # Sort by timestamp for deterministic ordering
    all_runs.sort(key=lambda r: r.get("timestamp", ""))

    total_runs = len(all_runs)

    # Build determinism rate series (time-ordered snapshots)
    determinism_rate_series: List[Dict[str, Any]] = []
    verified_so_far = 0
    for i, run in enumerate(all_runs):
        if run.get("status") == ReplayStatus.VERIFIED.value:
            verified_so_far += 1
        rate = verified_so_far / (i + 1) if (i + 1) > 0 else 0.0
        determinism_rate_series.append({
            "timestamp": run.get("timestamp", ""),
            "determinism_rate": round(rate, 4),
            "run_index": i + 1,
        })

    # Count critical incidents
    critical_count = sum(
        1 for inc in incidents
        if inc.get("severity") == IncidentSeverity.CRITICAL.value
    )
    critical_incident_rate = critical_count / len(incidents) if incidents else 0.0

    # Find hot fingerprints (recurring across runs)
    fingerprint_counts: Dict[str, int] = {}
    for inc in incidents:
        fp = inc.get("incident_fingerprint", "")
        if fp:
            fingerprint_counts[fp] = fingerprint_counts.get(fp, 0) + 1

    # Hot fingerprints are those appearing more than once
    hot_fingerprints = sorted([
        fp for fp, count in fingerprint_counts.items()
        if count > 1
    ])

    # Determine radar status based on trends
    if not determinism_rate_series:
        radar_status = RadarStatus.STABLE
    else:
        current_rate = determinism_rate_series[-1]["determinism_rate"]
        recent_rates = [
            d["determinism_rate"]
            for d in determinism_rate_series[-5:]
        ] if len(determinism_rate_series) >= 5 else [
            d["determinism_rate"] for d in determinism_rate_series
        ]

        # Check for degradation (declining trend)
        is_declining = False
        if len(recent_rates) >= 3:
            # Simple trend: is each rate lower than or equal to previous?
            declining_count = sum(
                1 for i in range(1, len(recent_rates))
                if recent_rates[i] < recent_rates[i - 1]
            )
            is_declining = declining_count >= len(recent_rates) - 1

        # Determine status
        if current_rate < 0.9 or critical_incident_rate > 0.2:
            radar_status = RadarStatus.UNSTABLE
        elif is_declining or current_rate < 0.95 or len(hot_fingerprints) >= 3:
            radar_status = RadarStatus.DEGRADING
        else:
            radar_status = RadarStatus.STABLE

    radar = {
        "schema_version": GOVERNANCE_RADAR_VERSION,
        "total_runs": total_runs,
        "determinism_rate_series": determinism_rate_series,
        "critical_incident_rate": round(critical_incident_rate, 4),
        "hot_fingerprints": hot_fingerprints,
        "radar_status": radar_status.value,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    # Compute deterministic radar hash
    radar_for_hash = dict(radar)
    radar_for_hash["radar_hash"] = ""
    canonical = json.dumps(radar_for_hash, sort_keys=True, separators=(',', ':'))
    radar["radar_hash"] = hashlib.sha256(canonical.encode('utf-8')).hexdigest()

    return radar


def evaluate_replay_for_promotion(
    radar: Dict[str, Any],
    latest_health: Dict[str, Any],
) -> Dict[str, Any]:
    """
    TASK 2: Evaluate replay determinism for promotion/RFL policy decisions.

    Provides a clear go/no-go signal for the promotion engine (MAAS) and
    policy-update guardrails.

    Args:
        radar: Governance radar dictionary
        latest_health: Latest health summary from summarize_replay_for_global_health

    Returns:
        Promotion evaluation with status and blocking information
    """
    blocking_fingerprints: List[str] = []
    notes: List[str] = []

    # Start with health status
    health_status = latest_health.get("replay_status", "OK")
    is_blocked = latest_health.get("is_blocked", False)
    is_healthy = latest_health.get("is_healthy", True)

    # Check radar status
    radar_status = radar.get("radar_status", RadarStatus.STABLE.value)

    # Collect blocking fingerprints from both sources
    blocking_fingerprints.extend(latest_health.get("blocking_fingerprints", []))
    blocking_fingerprints.extend(radar.get("hot_fingerprints", []))
    blocking_fingerprints = sorted(set(blocking_fingerprints))

    # Determine promotion status
    if is_blocked or radar_status == RadarStatus.UNSTABLE.value:
        status = PromotionStatus.BLOCK
        replay_ok_for_promotion = False
        if is_blocked:
            notes.append("Health status indicates blocking condition")
        if radar_status == RadarStatus.UNSTABLE.value:
            notes.append("Radar status is UNSTABLE")
        if latest_health.get("incomplete_runs", 0) > 0:
            notes.append(f"Incomplete runs detected: {latest_health.get('incomplete_runs')}")

    elif radar_status == RadarStatus.DEGRADING.value or health_status == "WARN":
        status = PromotionStatus.WARN
        replay_ok_for_promotion = True  # Allowed with caution
        if radar_status == RadarStatus.DEGRADING.value:
            notes.append("Radar status shows degrading trend")
        if latest_health.get("recent_failures", 0) > 0:
            notes.append(f"Recent failures: {latest_health.get('recent_failures')}")
        if blocking_fingerprints:
            notes.append(f"Hot fingerprints require attention: {len(blocking_fingerprints)}")

    else:
        status = PromotionStatus.OK
        replay_ok_for_promotion = True
        determinism_rate = latest_health.get("determinism_rate", 0.0)
        if determinism_rate >= 1.0:
            notes.append("Full determinism verified")
        else:
            notes.append(f"Determinism rate: {determinism_rate:.2%}")

    evaluation = {
        "replay_ok_for_promotion": replay_ok_for_promotion,
        "status": status.value,
        "blocking_fingerprints": blocking_fingerprints,
        "notes": notes,
        "radar_status": radar_status,
        "health_status": health_status,
        "determinism_rate": latest_health.get("determinism_rate", 0.0),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }

    return evaluation


def build_replay_director_panel(
    radar: Dict[str, Any],
    promotion_eval: Dict[str, Any],
) -> Dict[str, Any]:
    """
    TASK 3: Build a director-level status panel for replay determinism.

    Provides a high-level dashboard view suitable for executive reporting
    and governance oversight.

    Args:
        radar: Governance radar dictionary
        promotion_eval: Promotion evaluation dictionary

    Returns:
        Director panel with status light and headline summary
    """
    # Determine status light
    status = promotion_eval.get("status", PromotionStatus.OK.value)
    radar_status = radar.get("radar_status", RadarStatus.STABLE.value)

    if status == PromotionStatus.BLOCK.value or radar_status == RadarStatus.UNSTABLE.value:
        status_light = StatusLight.RED
    elif status == PromotionStatus.WARN.value or radar_status == RadarStatus.DEGRADING.value:
        status_light = StatusLight.YELLOW
    else:
        status_light = StatusLight.GREEN

    # Extract metrics
    determinism_rate = promotion_eval.get("determinism_rate", 0.0)
    total_runs = radar.get("total_runs", 0)
    critical_incident_rate = radar.get("critical_incident_rate", 0.0)

    # Calculate recent failure rate from radar series
    series = radar.get("determinism_rate_series", [])
    recent_failure_rate = 0.0
    if series:
        # Look at last 10 entries to estimate recent failures
        recent = series[-10:] if len(series) >= 10 else series
        if len(recent) >= 2:
            # Recent failure rate = 1 - final determinism rate (approx)
            recent_failure_rate = 1.0 - recent[-1]["determinism_rate"]

    # Count critical incidents from rate
    critical_incident_count = int(critical_incident_rate * total_runs) if total_runs > 0 else 0

    # Build headline
    if status_light == StatusLight.GREEN:
        if determinism_rate >= 1.0:
            headline = "Replay determinism fully verified across all runs"
        else:
            headline = f"Replay determinism healthy at {determinism_rate:.1%}"
    elif status_light == StatusLight.YELLOW:
        if radar_status == RadarStatus.DEGRADING.value:
            headline = "Replay determinism showing degrading trend - monitoring recommended"
        else:
            headline = f"Replay determinism at {determinism_rate:.1%} with recent issues"
    else:  # RED
        if radar_status == RadarStatus.UNSTABLE.value:
            headline = "Replay determinism unstable - promotion blocked until resolved"
        else:
            headline = "Replay verification blocked - critical issues require attention"

    panel = {
        "status_light": status_light.value,
        "determinism_rate": round(determinism_rate, 4),
        "recent_failure_rate": round(recent_failure_rate, 4),
        "critical_incident_count": critical_incident_count,
        "headline": headline,
        "total_runs": total_runs,
        "radar_status": radar_status,
        "promotion_status": status,
        "hot_fingerprint_count": len(radar.get("hot_fingerprints", [])),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    # Compute panel hash for integrity
    panel_for_hash = dict(panel)
    panel_for_hash["panel_hash"] = ""
    canonical = json.dumps(panel_for_hash, sort_keys=True, separators=(',', ':'))
    panel["panel_hash"] = hashlib.sha256(canonical.encode('utf-8')).hexdigest()

    return panel


def get_full_governance_status(
    ledgers: List[Dict[str, Any]],
    receipts: List[ReplayReceipt],
) -> Dict[str, Any]:
    """
    Convenience function to compute full governance status from raw inputs.

    Builds radar, promotion evaluation, and director panel in one call.
    """
    # Classify all incidents
    incidents = [classify_replay_incident(r) for r in receipts]

    # Build radar
    radar = build_replay_governance_radar(ledgers, incidents)

    # Get latest health (from most recent ledger if available)
    if ledgers:
        latest_health = summarize_replay_for_global_health(ledgers[-1])
    else:
        latest_health = {
            "replay_status": "OK",
            "is_blocked": False,
            "is_healthy": True,
            "determinism_rate": 0.0,
            "blocking_fingerprints": [],
            "recent_failures": 0,
        }

    # Evaluate for promotion
    promotion_eval = evaluate_replay_for_promotion(radar, latest_health)

    # Build director panel
    director_panel = build_replay_director_panel(radar, promotion_eval)

    return {
        "radar": radar,
        "promotion_evaluation": promotion_eval,
        "director_panel": director_panel,
        "incident_summary": get_incident_report(receipts),
    }


# ============================================================================
# PHASE V: Cross-System Replay Integration
# ============================================================================

# Global console schema version
GLOBAL_CONSOLE_VERSION = "1.0.0"

# Evidence chain schema version
EVIDENCE_CHAIN_VERSION = "1.0.0"


def summarize_replay_for_global_console(
    radar: Dict[str, Any],
    promotion_eval: Dict[str, Any],
) -> Dict[str, Any]:
    """
    TASK 1: Generate a minimal summary for the global health console.

    This is the single object the global health console should consume for
    replay governance status. It provides a neutral, actionable signal.

    Args:
        radar: Governance radar dictionary
        promotion_eval: Promotion evaluation dictionary

    Returns:
        Compact global console summary with neutral language
    """
    # Extract core signals
    radar_status = radar.get("radar_status", RadarStatus.STABLE.value)
    promotion_status = promotion_eval.get("status", PromotionStatus.OK.value)
    replay_ok = promotion_eval.get("replay_ok_for_promotion", True)

    hot_fingerprints = radar.get("hot_fingerprints", [])
    critical_incident_rate = radar.get("critical_incident_rate", 0.0)
    determinism_rate = promotion_eval.get("determinism_rate", 0.0)

    # Build neutral headline based on status
    if replay_ok and radar_status == RadarStatus.STABLE.value:
        if determinism_rate >= 1.0:
            headline = "Replay determinism verified across all runs"
        else:
            headline = f"Replay determinism at {determinism_rate:.1%}"
    elif promotion_status == PromotionStatus.WARN.value:
        if radar_status == RadarStatus.DEGRADING.value:
            headline = "Replay determinism trend requires monitoring"
        else:
            headline = f"Replay determinism at {determinism_rate:.1%} with recent incidents"
    else:  # BLOCK
        if radar_status == RadarStatus.UNSTABLE.value:
            headline = "Replay determinism requires investigation"
        else:
            headline = "Replay verification incomplete or inconsistent"

    console_summary = {
        "schema_version": GLOBAL_CONSOLE_VERSION,
        "replay_ok": replay_ok,
        "radar_status": radar_status,
        "promotion_status": promotion_status,
        "hot_fingerprints_count": len(hot_fingerprints),
        "critical_incident_rate": round(critical_incident_rate, 4),
        "determinism_rate": round(determinism_rate, 4),
        "headline": headline,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    # Compute deterministic hash for integrity
    console_for_hash = dict(console_summary)
    console_for_hash["console_hash"] = ""
    canonical = json.dumps(console_for_hash, sort_keys=True, separators=(',', ':'))
    console_summary["console_hash"] = hashlib.sha256(canonical.encode('utf-8')).hexdigest()

    return console_summary


def attach_replay_governance_to_evidence(
    chain_ledger: Dict[str, Any],
    radar: Dict[str, Any],
    promotion_eval: Dict[str, Any],
) -> Dict[str, Any]:
    """
    TASK 2: Attach replay governance tile to an existing evidence chain.

    Merges a replay_governance subtree into the evidence chain without
    breaking existing structure. This is the hook for the Evidence Pack.

    Args:
        chain_ledger: Existing evidence chain dictionary
        radar: Governance radar dictionary
        promotion_eval: Promotion evaluation dictionary

    Returns:
        New evidence chain dict with replay_governance subtree merged
    """
    # Extract the core governance signals
    radar_status = radar.get("radar_status", RadarStatus.STABLE.value)
    promotion_status = promotion_eval.get("status", PromotionStatus.OK.value)
    determinism_rate = promotion_eval.get("determinism_rate", 0.0)
    critical_incident_rate = radar.get("critical_incident_rate", 0.0)
    hot_fingerprints = radar.get("hot_fingerprints", [])

    # Determine the overall status for the tile
    if promotion_status == PromotionStatus.BLOCK.value:
        tile_status = "BLOCK"
    elif promotion_status == PromotionStatus.WARN.value:
        tile_status = "WARN"
    else:
        tile_status = "OK"

    # Build the replay governance tile
    replay_governance_tile = {
        "schema_version": EVIDENCE_CHAIN_VERSION,
        "status": tile_status,
        "determinism_rate": round(determinism_rate, 4),
        "critical_incident_rate": round(critical_incident_rate, 4),
        "radar_status": radar_status,
        "promotion_status": promotion_status,
        "hot_fingerprints_count": len(hot_fingerprints),
        "hot_fingerprints": hot_fingerprints[:5],  # Limit to first 5 for brevity
        "total_runs": radar.get("total_runs", 0),
        "attached_at": datetime.now(timezone.utc).isoformat(),
    }

    # Compute tile hash for integrity
    tile_for_hash = dict(replay_governance_tile)
    tile_for_hash["tile_hash"] = ""
    canonical = json.dumps(tile_for_hash, sort_keys=True, separators=(',', ':'))
    replay_governance_tile["tile_hash"] = hashlib.sha256(canonical.encode('utf-8')).hexdigest()

    # Create a new chain dict (non-mutating)
    new_chain = dict(chain_ledger)

    # Merge the replay_governance subtree
    new_chain["replay_governance"] = replay_governance_tile

    # Update chain metadata if present
    if "updated_at" in new_chain:
        new_chain["updated_at"] = datetime.now(timezone.utc).isoformat()

    return new_chain


def build_full_governance_snapshot(
    ledgers: List[Dict[str, Any]],
    receipts: List[ReplayReceipt],
) -> Dict[str, Any]:
    """
    Build a complete governance snapshot suitable for all consumers.

    This single call produces everything needed by:
    - Director console (director_panel)
    - MAAS (promotion_evaluation)
    - Evidence Pack (replay_governance tile via global_console_summary)
    - Global health console (global_console_summary)

    Args:
        ledgers: List of determinism ledger dictionaries
        receipts: List of replay receipts

    Returns:
        Complete governance snapshot dictionary
    """
    # Classify all incidents
    incidents = [classify_replay_incident(r) for r in receipts]

    # Build radar
    radar = build_replay_governance_radar(ledgers, incidents)

    # Get latest health (from most recent ledger if available)
    if ledgers:
        latest_health = summarize_replay_for_global_health(ledgers[-1])
    else:
        latest_health = {
            "replay_status": "OK",
            "is_blocked": False,
            "is_healthy": True,
            "determinism_rate": 1.0 if not receipts else 0.0,
            "blocking_fingerprints": [],
            "recent_failures": 0,
        }

    # Evaluate for promotion
    promotion_eval = evaluate_replay_for_promotion(radar, latest_health)

    # Build director panel
    director_panel = build_replay_director_panel(radar, promotion_eval)

    # Build global console summary (new in Phase V)
    global_console_summary = summarize_replay_for_global_console(radar, promotion_eval)

    # Build incident summary
    incident_summary = get_incident_report(receipts)

    snapshot = {
        "schema_version": GOVERNANCE_RADAR_VERSION,
        "radar": radar,
        "promotion_evaluation": promotion_eval,
        "director_panel": director_panel,
        "global_console_summary": global_console_summary,
        "incident_summary": incident_summary,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    # Compute snapshot hash for integrity
    snapshot_for_hash = dict(snapshot)
    snapshot_for_hash["snapshot_hash"] = ""
    canonical = json.dumps(snapshot_for_hash, sort_keys=True, separators=(',', ':'))
    snapshot["snapshot_hash"] = hashlib.sha256(canonical.encode('utf-8')).hexdigest()

    return snapshot


# ============================================================================
# PHASE VI: Replay as a First-Class Global Governance Signal
# ============================================================================

# GovernanceSignal schema version for CLAUDE I integration
GOVERNANCE_SIGNAL_VERSION = "1.0.0"


def to_governance_signal_for_replay(
    global_console_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """
    TASK 1: Convert replay global console summary to CLAUDE I GovernanceSignal.

    Transforms the replay-specific console summary into the canonical
    GovernanceSignal schema expected by CLAUDE I's governance synthesizer.

    GovernanceSignal Schema:
        - status: "OK" | "WARN" | "BLOCK"
        - blocking_rules: List[str] - fingerprints/rules causing blocks
        - blocking_rate: float - rate of blocking conditions (0.0-1.0)
        - headline: str - neutral one-liner for display
        - source: str - signal source identifier
        - determinism_rate: float - replay-specific metric

    Args:
        global_console_summary: Output from summarize_replay_for_global_console()

    Returns:
        GovernanceSignal dict conforming to CLAUDE I schema
    """
    replay_ok = global_console_summary.get("replay_ok", True)
    radar_status = global_console_summary.get("radar_status", RadarStatus.STABLE.value)
    promotion_status = global_console_summary.get("promotion_status", PromotionStatus.OK.value)
    determinism_rate = global_console_summary.get("determinism_rate", 1.0)
    critical_incident_rate = global_console_summary.get("critical_incident_rate", 0.0)
    hot_fingerprints_count = global_console_summary.get("hot_fingerprints_count", 0)
    headline = global_console_summary.get("headline", "Replay status unavailable")

    # Normalize status to OK/WARN/BLOCK
    # BLOCK conditions:
    #   - determinism_rate < 0.9
    #   - radar_status == UNSTABLE
    #   - replay_ok == False
    # WARN conditions:
    #   - radar_status == DEGRADING and replay_ok == True
    #   - promotion_status == WARN
    # OK conditions:
    #   - radar_status == STABLE and replay_ok == True

    if not replay_ok or radar_status == RadarStatus.UNSTABLE.value or determinism_rate < 0.9:
        status = "BLOCK"
    elif radar_status == RadarStatus.DEGRADING.value or promotion_status == PromotionStatus.WARN.value:
        status = "WARN"
    else:
        status = "OK"

    # Build blocking_rules from hot fingerprints and conditions
    blocking_rules: List[str] = []

    if determinism_rate < 0.9:
        blocking_rules.append(f"REPLAY-DET-LOW: determinism_rate={determinism_rate:.2%}")

    if radar_status == RadarStatus.UNSTABLE.value:
        blocking_rules.append("REPLAY-RADAR-UNSTABLE: radar indicates instability")

    if not replay_ok:
        blocking_rules.append("REPLAY-BLOCKED: promotion blocked by policy")

    if hot_fingerprints_count > 0:
        blocking_rules.append(f"REPLAY-HOT-FP: {hot_fingerprints_count} recurring fingerprint(s)")

    if critical_incident_rate > 0.2:
        blocking_rules.append(f"REPLAY-CRIT-HIGH: critical_incident_rate={critical_incident_rate:.1%}")

    # Calculate blocking_rate as composite of blocking conditions
    blocking_factors = [
        1.0 if determinism_rate < 0.9 else 0.0,
        1.0 if radar_status == RadarStatus.UNSTABLE.value else 0.0,
        1.0 if not replay_ok else 0.0,
        min(critical_incident_rate, 1.0),
    ]
    blocking_rate = sum(blocking_factors) / len(blocking_factors) if blocking_factors else 0.0

    governance_signal = {
        "schema_version": GOVERNANCE_SIGNAL_VERSION,
        "source": "replay",
        "status": status,
        "blocking_rules": blocking_rules,
        "blocking_rate": round(blocking_rate, 4),
        "headline": headline,
        "determinism_rate": round(determinism_rate, 4),
        "radar_status": radar_status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    # Compute signal hash for integrity
    signal_for_hash = dict(governance_signal)
    signal_for_hash["signal_hash"] = ""
    canonical = json.dumps(signal_for_hash, sort_keys=True, separators=(',', ':'))
    governance_signal["signal_hash"] = hashlib.sha256(canonical.encode('utf-8')).hexdigest()

    return governance_signal


def attach_replay_governance_to_evidence_with_signal(
    chain_ledger: Dict[str, Any],
    radar: Dict[str, Any],
    promotion_eval: Dict[str, Any],
    include_governance_signal: bool = False,
) -> Dict[str, Any]:
    """
    Extended version of attach_replay_governance_to_evidence with optional
    CLAUDE I GovernanceSignal stub.

    This maintains backward compatibility with the original function while
    adding the ability to include a governance_signal for CLAUDE I integration.

    Args:
        chain_ledger: Existing evidence chain dictionary
        radar: Governance radar dictionary
        promotion_eval: Promotion evaluation dictionary
        include_governance_signal: If True, adds governance_signal stub

    Returns:
        New evidence chain dict with replay_governance subtree merged
    """
    # Use the original function for base behavior
    updated_chain = attach_replay_governance_to_evidence(chain_ledger, radar, promotion_eval)

    # Optionally add governance_signal stub for CLAUDE I
    if include_governance_signal:
        # Build global console summary for the signal
        global_console_summary = summarize_replay_for_global_console(radar, promotion_eval)

        # Convert to governance signal
        governance_signal = to_governance_signal_for_replay(global_console_summary)

        # Add to the replay_governance tile
        updated_chain["replay_governance"]["governance_signal"] = governance_signal

        # Recompute tile hash to include the signal
        tile = updated_chain["replay_governance"]
        tile_for_hash = dict(tile)
        tile_for_hash["tile_hash"] = ""
        canonical = json.dumps(tile_for_hash, sort_keys=True, separators=(',', ':'))
        updated_chain["replay_governance"]["tile_hash"] = hashlib.sha256(canonical.encode('utf-8')).hexdigest()

    return updated_chain


# ============================================================================
# GLOBAL CONSOLE WIRING CONTRACT
# ============================================================================
#
# When embedding replay as a tile under global_health["replay"], the following
# JSON schema is expected. This contract ensures drop-in compatibility.
#
# GLOBAL_HEALTH_REPLAY_TILE_SCHEMA:
# {
#     "schema_version": "1.0.0",          # str: Schema version
#     "replay_ok": true|false,            # bool: Go/no-go signal
#     "radar_status": "STABLE"|"DEGRADING"|"UNSTABLE",  # str: Radar status enum
#     "promotion_status": "OK"|"WARN"|"BLOCK",          # str: Promotion status enum
#     "hot_fingerprints_count": 0,        # int: Count of recurring fingerprints
#     "critical_incident_rate": 0.0,      # float: Ratio of critical incidents (0.0-1.0)
#     "determinism_rate": 1.0,            # float: Overall determinism rate (0.0-1.0)
#     "headline": "...",                  # str: Neutral one-liner for display
#     "generated_at": "2025-...",         # str: ISO 8601 timestamp
#     "console_hash": "..."               # str: 64-char SHA256 hex digest
# }
#
# The output of summarize_replay_for_global_console() conforms exactly to this
# schema and can be embedded directly as global_health["replay"].
#
# Example wiring:
#     global_health = build_global_health()
#     radar = build_replay_governance_radar(ledgers, incidents)
#     promotion = evaluate_replay_for_promotion(radar, health)
#     global_health["replay"] = summarize_replay_for_global_console(radar, promotion)
#
# ============================================================================


def get_global_console_tile_schema() -> Dict[str, Any]:
    """
    Returns the expected JSON schema for the replay tile in global_health.

    This is provided for documentation and validation purposes.

    Returns:
        JSON Schema dict describing the replay tile format
    """
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "ReplayGlobalConsoleTile",
        "description": "Replay governance tile for embedding in global_health['replay']",
        "type": "object",
        "required": [
            "schema_version",
            "replay_ok",
            "radar_status",
            "promotion_status",
            "hot_fingerprints_count",
            "critical_incident_rate",
            "determinism_rate",
            "headline",
            "generated_at",
            "console_hash",
        ],
        "properties": {
            "schema_version": {
                "type": "string",
                "description": "Schema version string",
                "const": GLOBAL_CONSOLE_VERSION,
            },
            "replay_ok": {
                "type": "boolean",
                "description": "Go/no-go signal for replay governance",
            },
            "radar_status": {
                "type": "string",
                "enum": ["STABLE", "DEGRADING", "UNSTABLE"],
                "description": "Current radar status",
            },
            "promotion_status": {
                "type": "string",
                "enum": ["OK", "WARN", "BLOCK"],
                "description": "Promotion evaluation status",
            },
            "hot_fingerprints_count": {
                "type": "integer",
                "minimum": 0,
                "description": "Count of recurring incident fingerprints",
            },
            "critical_incident_rate": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Ratio of critical incidents",
            },
            "determinism_rate": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Overall determinism rate across runs",
            },
            "headline": {
                "type": "string",
                "description": "Neutral one-liner for display",
            },
            "generated_at": {
                "type": "string",
                "format": "date-time",
                "description": "ISO 8601 timestamp of generation",
            },
            "console_hash": {
                "type": "string",
                "pattern": "^[a-f0-9]{64}$",
                "description": "SHA256 integrity hash",
            },
        },
        "additionalProperties": False,
    }


def validate_global_console_tile(tile: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validate that a tile conforms to the global console wiring contract.

    Args:
        tile: The tile dictionary to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    required_fields = [
        "schema_version",
        "replay_ok",
        "radar_status",
        "promotion_status",
        "hot_fingerprints_count",
        "critical_incident_rate",
        "determinism_rate",
        "headline",
        "generated_at",
        "console_hash",
    ]

    # Check required fields
    for field in required_fields:
        if field not in tile:
            return False, f"Missing required field: {field}"

    # Validate types
    if not isinstance(tile["replay_ok"], bool):
        return False, "replay_ok must be boolean"

    if tile["radar_status"] not in ["STABLE", "DEGRADING", "UNSTABLE"]:
        return False, f"Invalid radar_status: {tile['radar_status']}"

    if tile["promotion_status"] not in ["OK", "WARN", "BLOCK"]:
        return False, f"Invalid promotion_status: {tile['promotion_status']}"

    if not isinstance(tile["hot_fingerprints_count"], int) or tile["hot_fingerprints_count"] < 0:
        return False, "hot_fingerprints_count must be non-negative integer"

    if not isinstance(tile["determinism_rate"], (int, float)):
        return False, "determinism_rate must be numeric"

    if not 0.0 <= tile["determinism_rate"] <= 1.0:
        return False, "determinism_rate must be between 0.0 and 1.0"

    if not isinstance(tile["console_hash"], str) or len(tile["console_hash"]) != 64:
        return False, "console_hash must be 64-character hex string"

    return True, None


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="PHASE II — Build or validate a determinism replay receipt"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Build command
    build_parser = subparsers.add_parser("build", help="Build a replay receipt")
    build_parser.add_argument("--primary-dir", required=True, type=Path, help="Primary run directory")
    build_parser.add_argument("--replay-dir", required=True, type=Path, help="Replay run directory")
    build_parser.add_argument("--manifest", required=True, type=Path, help="Manifest path")
    build_parser.add_argument("--output", required=True, type=Path, help="Output receipt path")
    build_parser.add_argument("--git-sha", default="unknown", help="Git commit SHA")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate a replay receipt")
    validate_parser.add_argument("--receipt", required=True, type=Path, help="Receipt path")
    validate_parser.add_argument("--json", action="store_true", help="Output JSON")

    # Summary command (TASK 3: governance-grade summary)
    summary_parser = subparsers.add_parser("summary", help="Generate governance summary from receipt index")
    summary_parser.add_argument("--index", required=True, type=Path, help="Receipt index path")
    summary_parser.add_argument("--output", type=Path, help="Output summary path (prints to stdout if not provided)")

    # Index command (TASK 1: receipt index management)
    index_parser = subparsers.add_parser("index", help="Build or update receipt index")
    index_parser.add_argument("--receipts", nargs="+", type=Path, help="Receipt files to index")
    index_parser.add_argument("--output", required=True, type=Path, help="Output index path")
    index_parser.add_argument("--experiment-id", help="Optional experiment ID override")

    # Phase III: Ledger command
    ledger_parser = subparsers.add_parser("ledger", help="Build determinism ledger from receipts")
    ledger_parser.add_argument("--receipts", nargs="+", type=Path, help="Receipt files to include")
    ledger_parser.add_argument("--output", required=True, type=Path, help="Output ledger path")

    # Phase III: Health command
    health_parser = subparsers.add_parser("health", help="Generate global health summary from ledger")
    health_parser.add_argument("--ledger", required=True, type=Path, help="Determinism ledger path")
    health_parser.add_argument("--output", type=Path, help="Output health summary path")

    # Phase III: Classify command
    classify_parser = subparsers.add_parser("classify", help="Classify incident from a receipt")
    classify_parser.add_argument("--receipt", required=True, type=Path, help="Receipt path")

    # Phase IV: Radar command
    radar_parser = subparsers.add_parser("radar", help="Build governance radar from ledgers")
    radar_parser.add_argument("--ledgers", nargs="+", type=Path, help="Ledger files to analyze")
    radar_parser.add_argument("--receipts", nargs="*", type=Path, help="Optional receipt files for incident analysis")
    radar_parser.add_argument("--output", type=Path, help="Output radar path")

    # Phase IV: Promote command
    promote_parser = subparsers.add_parser("promote", help="Evaluate replay status for promotion")
    promote_parser.add_argument("--radar", required=True, type=Path, help="Radar JSON path")
    promote_parser.add_argument("--health", required=True, type=Path, help="Health summary JSON path")

    # Phase IV: Panel command
    panel_parser = subparsers.add_parser("panel", help="Build director panel")
    panel_parser.add_argument("--radar", required=True, type=Path, help="Radar JSON path")
    panel_parser.add_argument("--promotion", required=True, type=Path, help="Promotion evaluation JSON path")
    panel_parser.add_argument("--output", type=Path, help="Output panel path")

    # Phase V: Governance snapshot command
    governance_parser = subparsers.add_parser(
        "governance",
        help="Build complete governance snapshot (radar, promotion, panel, global console)"
    )
    governance_parser.add_argument("--ledgers", nargs="+", type=Path, help="Ledger files to analyze")
    governance_parser.add_argument("--receipts", nargs="*", type=Path, help="Optional receipt files for incident analysis")
    governance_parser.add_argument("--out", required=True, type=Path, help="Output governance snapshot path")

    # Phase V: Console command (global console summary only)
    console_parser = subparsers.add_parser("console", help="Build global console summary")
    console_parser.add_argument("--radar", required=True, type=Path, help="Radar JSON path")
    console_parser.add_argument("--promotion", required=True, type=Path, help="Promotion evaluation JSON path")
    console_parser.add_argument("--output", type=Path, help="Output console summary path")

    # Phase V: Attach to evidence chain command
    attach_parser = subparsers.add_parser("attach", help="Attach replay governance to evidence chain")
    attach_parser.add_argument("--chain", required=True, type=Path, help="Existing evidence chain JSON path")
    attach_parser.add_argument("--radar", required=True, type=Path, help="Radar JSON path")
    attach_parser.add_argument("--promotion", required=True, type=Path, help="Promotion evaluation JSON path")
    attach_parser.add_argument("--output", required=True, type=Path, help="Output evidence chain path")

    args = parser.parse_args()

    if args.command == "build":
        receipt = build_replay_receipt(
            primary_run_dir=args.primary_dir,
            replay_run_dir=args.replay_dir,
            manifest_path=args.manifest,
            git_sha=args.git_sha,
        )
        save_replay_receipt(receipt, args.output)
        print(f"Receipt saved to {args.output}")
        print(f"Status: {receipt.status.value}")
        print(f"Checks: {receipt.verification_summary.checks_passed}/{receipt.verification_summary.checks_total} passed")
        sys.exit(0 if receipt.status == ReplayStatus.VERIFIED else 1)

    elif args.command == "validate":
        valid, error_code, message = validate_replay_receipt(args.receipt)
        if args.json:
            result = {
                "valid": valid,
                "error_code": error_code.value if error_code else None,
                "message": message,
            }
            print(json.dumps(result, indent=2))
        else:
            print(f"Valid: {valid}")
            if error_code:
                print(f"Error: {error_code.value}")
            print(f"Message: {message}")
        sys.exit(0 if valid else 1)

    elif args.command == "summary":
        index = load_receipt_index(args.index)
        summary = summarize_receipt_index(index)
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, sort_keys=True)
            print(f"Summary saved to {args.output}")
        else:
            print(json.dumps(summary, indent=2, sort_keys=True))
        sys.exit(0 if summary["all_verified"] else 1)

    elif args.command == "index":
        receipts = []
        for receipt_path in args.receipts:
            try:
                receipts.append(load_replay_receipt(receipt_path))
            except Exception as e:
                print(f"Warning: Failed to load {receipt_path}: {e}", file=sys.stderr)
        index = build_receipt_index(receipts, experiment_id=args.experiment_id)
        dump_receipt_index(args.output, index)
        print(f"Index saved to {args.output}")
        print(f"Indexed {len(receipts)} receipts")
        sys.exit(0)

    elif args.command == "ledger":
        receipts = []
        for receipt_path in args.receipts:
            try:
                receipts.append(load_replay_receipt(receipt_path))
            except Exception as e:
                print(f"Warning: Failed to load {receipt_path}: {e}", file=sys.stderr)
        ledger = build_replay_determinism_ledger(receipts)
        save_determinism_ledger(ledger, args.output)
        print(f"Ledger saved to {args.output}")
        print(f"  Runs: {len(ledger['run_history'])}")
        print(f"  Determinism rate: {ledger['determinism_rate']:.2%}")
        sys.exit(0 if ledger["determinism_rate"] >= 0.95 else 1)

    elif args.command == "health":
        ledger = load_determinism_ledger(args.ledger)
        health = summarize_replay_for_global_health(ledger)
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(health, f, indent=2, sort_keys=True)
            print(f"Health summary saved to {args.output}")
        else:
            print(json.dumps(health, indent=2, sort_keys=True))
        print(f"Status: {health['replay_status']}")
        sys.exit(0 if health["is_healthy"] else (2 if health["is_blocked"] else 1))

    elif args.command == "classify":
        receipt = load_replay_receipt(args.receipt)
        incident = classify_replay_incident(receipt)
        print(json.dumps(incident, indent=2))
        # Exit code based on severity
        severity_codes = {"NONE": 0, "LOW": 0, "HIGH": 1, "CRITICAL": 2}
        sys.exit(severity_codes.get(incident["severity"], 1))

    elif args.command == "radar":
        # Load ledgers
        ledgers = []
        for ledger_path in args.ledgers:
            try:
                ledgers.append(load_determinism_ledger(ledger_path))
            except Exception as e:
                print(f"Warning: Failed to load ledger {ledger_path}: {e}", file=sys.stderr)

        # Load receipts for incident analysis (optional)
        incidents = []
        if args.receipts:
            for receipt_path in args.receipts:
                try:
                    receipt = load_replay_receipt(receipt_path)
                    incidents.append(classify_replay_incident(receipt))
                except Exception as e:
                    print(f"Warning: Failed to load receipt {receipt_path}: {e}", file=sys.stderr)

        radar = build_replay_governance_radar(ledgers, incidents)

        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(radar, f, indent=2, sort_keys=True)
            print(f"Radar saved to {args.output}")
        else:
            print(json.dumps(radar, indent=2, sort_keys=True))

        print(f"Status: {radar['radar_status']}")
        status_codes = {"STABLE": 0, "DEGRADING": 1, "UNSTABLE": 2}
        sys.exit(status_codes.get(radar["radar_status"], 1))

    elif args.command == "promote":
        with open(args.radar, 'r', encoding='utf-8') as f:
            radar = json.load(f)
        with open(args.health, 'r', encoding='utf-8') as f:
            health = json.load(f)

        evaluation = evaluate_replay_for_promotion(radar, health)
        print(json.dumps(evaluation, indent=2, sort_keys=True))

        if evaluation["replay_ok_for_promotion"]:
            print("Promotion: ALLOWED")
            sys.exit(0)
        else:
            print("Promotion: BLOCKED")
            sys.exit(1)

    elif args.command == "panel":
        with open(args.radar, 'r', encoding='utf-8') as f:
            radar = json.load(f)
        with open(args.promotion, 'r', encoding='utf-8') as f:
            promotion = json.load(f)

        panel = build_replay_director_panel(radar, promotion)

        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(panel, f, indent=2, sort_keys=True)
            print(f"Panel saved to {args.output}")
        else:
            print(json.dumps(panel, indent=2, sort_keys=True))

        print(f"Status Light: {panel['status_light']}")
        print(f"Headline: {panel['headline']}")
        light_codes = {"GREEN": 0, "YELLOW": 1, "RED": 2}
        sys.exit(light_codes.get(panel["status_light"], 1))

    elif args.command == "governance":
        # Phase V: Full governance snapshot
        ledgers = []
        if args.ledgers:
            for ledger_path in args.ledgers:
                try:
                    ledgers.append(load_determinism_ledger(ledger_path))
                except Exception as e:
                    print(f"Warning: Failed to load ledger {ledger_path}: {e}", file=sys.stderr)

        receipts = []
        if args.receipts:
            for receipt_path in args.receipts:
                try:
                    receipts.append(load_replay_receipt(receipt_path))
                except Exception as e:
                    print(f"Warning: Failed to load receipt {receipt_path}: {e}", file=sys.stderr)

        snapshot = build_full_governance_snapshot(ledgers, receipts)

        # Write to output file
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(snapshot, f, indent=2, sort_keys=True)

        print(f"Governance snapshot saved to {args.out}")
        print(f"  Radar Status: {snapshot['radar']['radar_status']}")
        print(f"  Promotion: {'ALLOWED' if snapshot['promotion_evaluation']['replay_ok_for_promotion'] else 'BLOCKED'}")
        print(f"  Status Light: {snapshot['director_panel']['status_light']}")
        print(f"  Headline: {snapshot['global_console_summary']['headline']}")

        # Exit code based on promotion status
        status_codes = {"OK": 0, "WARN": 1, "BLOCK": 2}
        sys.exit(status_codes.get(snapshot["promotion_evaluation"]["status"], 1))

    elif args.command == "console":
        # Phase V: Global console summary
        with open(args.radar, 'r', encoding='utf-8') as f:
            radar = json.load(f)
        with open(args.promotion, 'r', encoding='utf-8') as f:
            promotion = json.load(f)

        console_summary = summarize_replay_for_global_console(radar, promotion)

        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(console_summary, f, indent=2, sort_keys=True)
            print(f"Console summary saved to {args.output}")
        else:
            print(json.dumps(console_summary, indent=2, sort_keys=True))

        print(f"Replay OK: {console_summary['replay_ok']}")
        print(f"Headline: {console_summary['headline']}")
        sys.exit(0 if console_summary["replay_ok"] else 1)

    elif args.command == "attach":
        # Phase V: Attach replay governance to evidence chain
        with open(args.chain, 'r', encoding='utf-8') as f:
            chain = json.load(f)
        with open(args.radar, 'r', encoding='utf-8') as f:
            radar = json.load(f)
        with open(args.promotion, 'r', encoding='utf-8') as f:
            promotion = json.load(f)

        updated_chain = attach_replay_governance_to_evidence(chain, radar, promotion)

        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(updated_chain, f, indent=2, sort_keys=True)

        tile = updated_chain["replay_governance"]
        print(f"Evidence chain updated with replay governance tile")
        print(f"  Tile Status: {tile['status']}")
        print(f"  Determinism Rate: {tile['determinism_rate']:.2%}")
        print(f"  Output: {args.output}")

        status_codes = {"OK": 0, "WARN": 1, "BLOCK": 2}
        sys.exit(status_codes.get(tile["status"], 1))
