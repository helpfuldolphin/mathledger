"""
Canonical Pydantic models for MathLedger's public API responses.

Every FastAPI route must return one of these schemas to guarantee
deterministic, contract-checked JSON payloads.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, constr


HexDigest = constr(pattern=r"^[0-9a-f]{64}$")


class ApiModel(BaseModel):
    """Base model enforcing strict field control and canonical serialization."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# ---------------------------------------------------------------------------
# Global / heartbeat responses
# ---------------------------------------------------------------------------


class HeartbeatProofs(ApiModel):
    success: int = Field(..., ge=0)


class HeartbeatLatest(ApiModel):
    merkle: Optional[str] = None


class HeartbeatBlocks(ApiModel):
    height: int = Field(..., ge=0)
    latest: HeartbeatLatest


class HeartbeatPolicy(ApiModel):
    hash: Optional[str] = None


class HeartbeatRedis(ApiModel):
    ml_jobs_len: int


class HeartbeatResponse(ApiModel):
    ok: bool
    ts: datetime
    proofs: HeartbeatProofs
    proofs_per_sec: float = Field(..., ge=0.0)
    blocks: HeartbeatBlocks
    policy: HeartbeatPolicy
    redis: HeartbeatRedis


class HealthResponse(ApiModel):
    status: str = Field(..., min_length=1)
    timestamp: datetime


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class ProofTotals(ApiModel):
    total: int = Field(..., ge=0)
    success: int = Field(..., ge=0)
    failure: int = Field(..., ge=0)
    success_rate: float = Field(..., ge=0.0, le=1.0)


class StatementTotals(ApiModel):
    total: int = Field(..., ge=0)
    max_depth: int = Field(..., ge=0)


class BlockTotals(ApiModel):
    count: int = Field(..., ge=0)
    height: int = Field(..., ge=0)


class FirstOrganismMetrics(ApiModel):
    runs_total: int = Field(..., ge=0)
    runs_completed: int = Field(default=0, ge=0)
    runs_failed: int = Field(default=0, ge=0)
    last_ht_hash: Optional[str] = None
    last_abstention_rate: float = Field(default=0.0, ge=0.0)
    latency_seconds: float = Field(..., ge=0.0)
    latency_buckets: Dict[str, int] = Field(default_factory=dict)
    abstention_buckets: Dict[str, int] = Field(default_factory=dict)


class MetricsResponse(ApiModel):
    generated_at: datetime
    proofs: ProofTotals
    statements: StatementTotals
    blocks: BlockTotals
    first_organism: Optional[FirstOrganismMetrics] = None
    queue_length: int = Field(..., ge=-1)


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------


class BlockLatestResponse(ApiModel):
    block_number: int = Field(..., ge=0)
    merkle_root: Optional[str] = None
    created_at: Optional[datetime] = None
    header: Dict[str, Any]


# ---------------------------------------------------------------------------
# Statements
# ---------------------------------------------------------------------------


class ProofSummary(ApiModel):
    method: Optional[str] = None
    status: Optional[str] = None
    success: Optional[bool] = None
    created_at: Optional[datetime] = None
    prover: Optional[str] = None
    duration_ms: Optional[int] = Field(default=None, ge=0)


class ParentSummary(ApiModel):
    hash: HexDigest
    display: Optional[str] = None


class StatementDetailResponse(ApiModel):
    hash: HexDigest
    text: Optional[str] = None
    normalized_text: Optional[str] = None
    display: str = Field(..., min_length=1)
    proofs: List[ProofSummary]
    parents: List[ParentSummary]


class RecentStatementItem(ApiModel):
    hash: HexDigest
    display: str = Field(..., min_length=1)
    text: Optional[str] = None
    normalized_text: Optional[str] = None


class RecentStatementsResponse(ApiModel):
    items: List[RecentStatementItem]


# ---------------------------------------------------------------------------
# Auxiliary read-only UI payloads
# ---------------------------------------------------------------------------


class ProofListResponse(ApiModel):
    proofs: List[ProofSummary]


class ParentListResponse(ApiModel):
    parents: List[ParentSummary]


# ---------------------------------------------------------------------------
# Attestation API
# ---------------------------------------------------------------------------


class UIEventRecord(ApiModel):
    event_id: str
    timestamp: float
    leaf_hash: str
    canonical_value: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class UIEventResponse(ApiModel):
    event_id: str
    timestamp: float
    leaf_hash: str


class UIEventListResponse(ApiModel):
    events: List[UIEventRecord]


class AttestationLatestResponse(ApiModel):
    block_number: Optional[int] = None
    reasoning_merkle_root: Optional[str] = None
    ui_merkle_root: Optional[str] = None
    composite_attestation_root: Optional[str] = None
    attestation_metadata: Dict[str, Any] = Field(default_factory=dict)
    block_hash: Optional[str] = None


class DerivationSimulationResponse(ApiModel):
    triggered: bool
    job_id: Optional[str] = None
    status: str
