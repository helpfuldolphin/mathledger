"""
AI Proof Ingestion Pipeline

Internal ingestion pipeline for AI-generated proofs.
Implements the core submission flow with mandatory shadow mode.

DESIGN PRINCIPLES (from spec):
1. Proof-or-Abstain: Same Lean verification as internal proofs
2. Provenance Chain: Full metadata recorded
3. Shadow Mode First: Mandatory for all AI submissions

This module provides the internal API. No public endpoint exposure.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from psycopg.cursor import Cursor
from psycopg.types.json import Json


# =============================================================================
# Data Structures
# =============================================================================


@dataclass(frozen=True)
class ProvenanceMetadata:
    """
    Provenance chain for an AI-submitted proof.

    All fields are required for traceability.
    """
    source_type: str  # Always "external_ai" for AI submissions
    source_id: str    # e.g., "gpt-4-turbo-2025-01"
    raw_output_hash: str  # SHA-256 of raw AI output
    prompt_hash: Optional[str] = None  # SHA-256 of prompt (optional)
    model_temperature: Optional[float] = None  # Temperature setting (optional)
    extra: Optional[Dict[str, Any]] = None  # Additional metadata

    def __post_init__(self):
        if self.source_type != "external_ai":
            raise ValueError(f"source_type must be 'external_ai', got: {self.source_type}")
        if not self.source_id:
            raise ValueError("source_id is required")
        if not self.raw_output_hash or len(self.raw_output_hash) != 64:
            raise ValueError("raw_output_hash must be a 64-char SHA-256 hex string")


@dataclass(frozen=True)
class AIProofSubmission:
    """
    An AI-generated proof submission.

    Contains the statement, proof term, and provenance metadata.
    """
    statement: str  # The mathematical statement
    proof_term: str  # The proof term (Lean syntax)
    provenance: ProvenanceMetadata
    submission_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    submitted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def compute_attestation_hash(self) -> str:
        """Compute deterministic hash for attestation."""
        payload = json.dumps({
            "statement": self.statement,
            "proof_term": self.proof_term,
            "source_id": self.provenance.source_id,
            "raw_output_hash": self.provenance.raw_output_hash,
            "submission_id": self.submission_id,
        }, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(b"MathLedger:AIProof:v1:" + payload.encode()).hexdigest()


@dataclass(frozen=True)
class IngestResult:
    """Result of ingesting an AI proof."""
    submission_id: str
    proof_id: Optional[str]  # None if verification failed before DB insert
    provenance_id: Optional[str]
    status: str  # "queued", "verified", "rejected"
    shadow_mode: bool  # Always True for AI submissions in Phase 1
    message: Optional[str] = None


# =============================================================================
# Provenance Recording
# =============================================================================


def record_provenance(
    cur: Cursor,
    proof_id: str,
    submission: AIProofSubmission,
) -> str:
    """
    Record provenance metadata for an AI-submitted proof.

    Args:
        cur: Database cursor
        proof_id: The proof ID in the proofs table
        submission: The AI proof submission

    Returns:
        The provenance record ID
    """
    metadata = {}
    if submission.provenance.prompt_hash:
        metadata["prompt_hash"] = submission.provenance.prompt_hash
    if submission.provenance.model_temperature is not None:
        metadata["model_temperature"] = submission.provenance.model_temperature
    if submission.provenance.extra:
        metadata.update(submission.provenance.extra)

    cur.execute(
        """
        INSERT INTO proof_provenance (
            proof_id,
            source_type,
            source_id,
            submission_id,
            raw_output_hash,
            submitter_attestation,
            metadata,
            created_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (
            proof_id,
            submission.provenance.source_type,
            submission.provenance.source_id,
            submission.submission_id,
            submission.provenance.raw_output_hash,
            submission.compute_attestation_hash(),
            Json(metadata) if metadata else None,
            submission.submitted_at,
        ),
    )
    return str(cur.fetchone()[0])


# =============================================================================
# Shadow Mode Enforcement
# =============================================================================


def ensure_shadow_mode(cur: Cursor, proof_id: str) -> None:
    """
    Ensure a proof is marked as shadow mode.

    This is a defensive check - the constraint should prevent
    external_ai proofs from having shadow_mode=false, but we
    enforce it at the application layer as well.

    Args:
        cur: Database cursor
        proof_id: The proof ID to enforce shadow mode on
    """
    cur.execute(
        """
        UPDATE proofs
        SET shadow_mode = true
        WHERE id = %s AND source_type = 'external_ai'
        """,
        (proof_id,),
    )


# =============================================================================
# Main Ingestion Function
# =============================================================================


def ingest_ai_proof(
    cur: Cursor,
    submission: AIProofSubmission,
    *,
    theory_name: str = "Propositional",
) -> IngestResult:
    """
    Ingest an AI-generated proof into the system.

    PHASE 1 BEHAVIOR:
    - All AI proofs are ingested in shadow mode
    - Verification is queued but not executed (returns "queued")
    - Provenance is recorded immediately
    - No slice progression credit

    This function does NOT:
    - Expose a public API
    - Execute Lean verification (that's handled by worker.py)
    - Apply rate limiting (Phase 2)
    - Handle graduation (Phase 2)

    Args:
        cur: Database cursor (caller manages transaction)
        submission: The AI proof submission
        theory_name: The theory/system name (default: Propositional)

    Returns:
        IngestResult with submission status
    """
    from normalization.canon import normalize
    from substrate.crypto.hashing import DOMAIN_STMT, hash_statement

    # Normalize the statement
    normalized = normalize(submission.statement)
    if not normalized:
        return IngestResult(
            submission_id=submission.submission_id,
            proof_id=None,
            provenance_id=None,
            status="rejected",
            shadow_mode=True,
            message="Statement is empty after normalization",
        )

    # Compute statement hash
    statement_hash = hash_statement(normalized)

    # Ensure theory exists
    cur.execute(
        """
        INSERT INTO theories (name)
        VALUES (%s)
        ON CONFLICT (name) DO NOTHING
        RETURNING id
        """,
        (theory_name,),
    )
    row = cur.fetchone()
    if row:
        system_id = str(row[0])
    else:
        cur.execute("SELECT id FROM theories WHERE name = %s", (theory_name,))
        system_id = str(cur.fetchone()[0])

    # Upsert statement (status=unknown until verified)
    cur.execute(
        """
        INSERT INTO statements (
            theory_id,
            system_id,
            hash,
            content,
            content_norm,
            normalized_text,
            status
        )
        VALUES (%s, %s, %s, %s, %s, %s, 'unknown')
        ON CONFLICT (hash) DO UPDATE
        SET content = COALESCE(statements.content, EXCLUDED.content),
            content_norm = COALESCE(statements.content_norm, EXCLUDED.content_norm)
        RETURNING id
        """,
        (
            system_id,
            system_id,
            statement_hash,
            submission.statement,
            normalized,
            submission.statement,
        ),
    )
    statement_id = str(cur.fetchone()[0])

    # Create proof record with shadow_mode=true, source_type=external_ai
    # Status is "queued" until worker verifies
    cur.execute(
        """
        INSERT INTO proofs (
            statement_id,
            system_id,
            prover,
            method,
            status,
            proof_text,
            success,
            source_type,
            shadow_mode
        )
        VALUES (%s, %s, 'external_ai', 'ai-submission', 'queued', %s, false, 'external_ai', true)
        RETURNING id
        """,
        (
            statement_id,
            system_id,
            submission.proof_term,
        ),
    )
    proof_id = str(cur.fetchone()[0])

    # Record provenance
    provenance_id = record_provenance(cur, proof_id, submission)

    # Defensive: ensure shadow mode (belt and suspenders)
    ensure_shadow_mode(cur, proof_id)

    return IngestResult(
        submission_id=submission.submission_id,
        proof_id=proof_id,
        provenance_id=provenance_id,
        status="queued",
        shadow_mode=True,
        message="AI proof queued for verification (shadow mode)",
    )


# =============================================================================
# Utility Functions
# =============================================================================


def compute_raw_output_hash(raw_output: str) -> str:
    """
    Compute SHA-256 hash of raw AI output.

    Use this to generate the raw_output_hash field for ProvenanceMetadata.

    Args:
        raw_output: The raw output string from the AI system

    Returns:
        64-character hex SHA-256 hash
    """
    return hashlib.sha256(raw_output.encode("utf-8")).hexdigest()


def compute_prompt_hash(prompt: str) -> str:
    """
    Compute SHA-256 hash of the prompt sent to the AI.

    Args:
        prompt: The prompt string

    Returns:
        64-character hex SHA-256 hash
    """
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
