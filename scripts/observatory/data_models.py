# scripts/observatory/data_models.py
"""
Defines the data structures for the Hash Consistency Observatory.

These are inspired by the user's prompt and serve as a basis for the
newly requested functions.
"""
from typing import TypedDict, List, Literal, Sequence

# The status of an integrity check
IntegrityStatus = Literal["PASS", "WARN", "FAIL"]

class SliceIdentityCard(TypedDict):
    """
    Represents the state of a single slice at a specific point in time (an audit).
    This is a conceptual structure based on the prompt.
    """
    slice_name: str
    timestamp_utc: str
    integrity_status: IntegrityStatus
    has_prereg: bool
    has_manifest_entry: bool
    drift_events: List[str] # List of HASH-DRIFT codes detected

class ReconciliationResult(TypedDict):
    """
    Represents the detailed output of a full audit, which is then summarized.
    This is the data structure that the hash_reconciliation_auditor would produce internally.
    """
    slices: List[SliceIdentityCard]
    # This would contain more detailed check-by-check results in a real implementation
