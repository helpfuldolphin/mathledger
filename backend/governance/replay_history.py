# PHASE III — CROSS-RUN HISTORY & INCIDENTS
"""
Implements the Replay History Ledger, Determinism Incident Extractor,
and the Global Health Hook for the replay evidence spine.
"""
import json
from typing import Dict, Any, List, Sequence, NamedTuple

# --- Data Structures (as defined by contracts) ---

class ReplayReceipt(NamedTuple):
    run_id: str
    manifest_path: str
    status: str  # "VERIFIED", "FAILED", "INCOMPLETE"
    recon_codes: List[str]
    expected_hash: str
    actual_hash: str
    timestamp: str = ""

class ReceiptIndexEntry(NamedTuple):
    receipts: Sequence[ReplayReceipt]
    # ... other index fields

class ReceiptSummary(NamedTuple):
    total_receipts: int
    num_verified: int
    num_failed: int
    num_incomplete: int
    recon_error_codes: List[str]
    # ... other summary fields

# --- TASK 1: Replay History Ledger ---

def build_replay_history(receipt_index: ReceiptIndexEntry) -> Dict[str, Any]:
    """Builds a deterministic, JSON-safe replay history ledger from a receipt index."""
    receipts = sorted(receipt_index.receipts, key=lambda r: r.run_id)
    
    verified_count = sum(1 for r in receipts if r.status == "VERIFIED")
    failed_count = sum(1 for r in receipts if r.status == "FAILED")
    incomplete_count = sum(1 for r in receipts if r.status == "INCOMPLETE")

    successful_timestamps = sorted([r.timestamp for r in receipts if r.status == "VERIFIED" and r.timestamp])
    failed_timestamps = sorted([r.timestamp for r in receipts if r.status == "FAILED" and r.timestamp])

    history = {
        "schema_version": "1.0.0",
        "total_receipts": len(receipts),
        "number_verified": verified_count,
        "number_failed": failed_count,
        "number_incomplete": incomplete_count,
        "first_successful_replay_at": successful_timestamps[0] if successful_timestamps else None,
        "last_failure_at": failed_timestamps[-1] if failed_timestamps else None,
    }
    return history

def save_replay_history(path: str, history: Dict[str, Any]):
    """Saves the replay history ledger to a file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, sort_keys=True)

def load_replay_history(path: str) -> Dict[str, Any]:
    """Loads a replay history ledger from a file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# --- TASK 2: Determinism Incident Extractor ---

def extract_replay_incidents(receipts: Sequence[ReplayReceipt]) -> List[Dict[str, Any]]:
    """Extracts a deterministic list of determinism incidents."""
    incidents = []
    # Sort for deterministic output
    for receipt in sorted(receipts, key=lambda r: r.run_id):
        if receipt.status in ("FAILED", "INCOMPLETE"):
            incident = {
                "run_id": receipt.run_id,
                "manifest_path": receipt.manifest_path,
                "status": receipt.status,
                "recon_codes_seen": sorted(list(set(receipt.recon_codes))),
                "ht_series_hash": receipt.actual_hash,
                "expected_hash": receipt.expected_hash,
            }
            incidents.append(incident)
    return incidents

# --- TASK 3: Global Health Hook ---

def summarize_replay_for_global_health(summary: ReceiptSummary) -> Dict[str, Any]:
    """Produces a small, stable summary for global_health.json."""
    all_verified = summary.total_receipts > 0 and summary.num_failed == 0 and summary.num_incomplete == 0
    failure_count = summary.num_failed + summary.num_incomplete
    
    status = "OK"
    if failure_count > 0:
        status = "WARN"
    # A BLOCKED status might be triggered by specific, critical RECON codes
    if "RECON-001" in summary.recon_error_codes: # Example of a critical code
        status = "BLOCKED"

    health_summary = {
        "all_verified": all_verified,
        "failure_count": failure_count,
        "recon_error_codes": sorted(list(set(summary.recon_error_codes))),
        "status": status,
    }
    return health_summary
