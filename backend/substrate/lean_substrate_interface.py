"""
This module defines the JSON-based communication interface for interacting with
a Lean theorem-proving process. It specifies the precise, typed schemas for
request and response objects, and provides canonical serialization and hashing
functions to enforce the determinism contract.

Protocol Version: 1.0
"""
import json
import hashlib
from typing import TypedDict, Union, Literal, Optional, Dict

# ==============================================================================
# Protocol v1.0 Schemas
# ==============================================================================

class SubstrateBudget(TypedDict):
    """Defines the budget for a substrate execution."""
    cycle_budget_s: float
    taut_timeout_s: float

class SubstrateRequest(TypedDict):
    """Schema for a request sent TO the Lean substrate process."""
    protocol_version: Literal["1.0"]
    item_id: str
    cycle_seed: int
    formula: str
    budget: SubstrateBudget

class SubstrateSuccessResponse(TypedDict):
    """Schema for a successful response FROM the Lean substrate process."""
    status: Literal["ok"]
    request_hash: str
    result_payload: Dict
    cycles_consumed: int
    determinism_hash: str

class SubstrateErrorResponse(TypedDict):
    """Schema for a failed response FROM the Lean substrate process."""
    status: Literal["error"]
    request_hash: str
    error_code: str
    message: str

SubstrateResponse = Union[SubstrateSuccessResponse, SubstrateErrorResponse]


# ==============================================================================
# Canonical Serialization & Hashing
# ==============================================================================

def _canonical_json_dumps(data: Union[Dict, TypedDict]) -> bytes:
    """
    Serializes a dictionary or TypedDict to a canonical JSON byte string.
    Invariants: UTF-8, sorted keys, no whitespace, no ASCII escaping.
    """
    return json.dumps(
        data,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":")
    ).encode('utf-8')

def compute_request_hash(request: SubstrateRequest) -> str:
    """Computes the SHA256 hash of a canonicalized SubstrateRequest."""
    # The request_hash itself is not part of the hashed content.
    return hashlib.sha256(_canonical_json_dumps(request)).hexdigest()

def compute_determinism_hash(result_payload: Dict) -> str:
    """Computes the SHA256 hash of a canonicalized result payload."""
    return hashlib.sha256(_canonical_json_dumps(result_payload)).hexdigest()