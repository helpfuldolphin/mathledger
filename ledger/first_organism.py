"""
First Organism Ledger Helpers
=============================

This module exposes specific helpers for the First Organism test suite,
encapsulating the ingestion and sealing contract.
"""
from dataclasses import dataclass
from typing import Any, Dict, Sequence, Optional

from psycopg.cursor import Cursor

from ledger.ingest import LedgerIngestor


@dataclass(frozen=True)
class SealedBlock:
    """
    Represents a sealed block for the First Organism contract.
    Carries the dual roots and identity metadata.
    """
    reasoning_root: str  # R_t
    ui_root: str         # U_t
    composite_root: str  # H_t
    block_id: int
    sequence: int
    timestamp: str       # ISO formatted timestamp


def ingest_and_seal_for_first_organism(
    cur: Cursor,
    result: Dict[str, Any],
    ui_events: Sequence[str]
) -> SealedBlock:
    """
    Ingest a single proof result and seal it into a block with UI events.

    This helper wraps LedgerIngestor to enforce the First Organism contract:
    1. Ingest statement and proof.
    2. Seal block immediately with provided UI events.
    3. Return the dual roots and block identity.

    Args:
        cur: Database cursor (in transaction).
        result: Proof result dictionary (must contain 'statement', 'proof', 'prover', etc.).
        ui_events: List of UI event strings to be included in the block.

    Returns:
        SealedBlock containing R_t, U_t, H_t, and block metadata.
    """
    ingestor = LedgerIngestor()

    # Extract fields from the result dict
    theory_name = result.get("theory", "first_organism")
    statement = result.get("statement", "")
    proof_text = result.get("proof", "")
    prover = result.get("prover", "test_prover")
    status = result.get("status", "success")
    
    outcome = ingestor.ingest(
        cur=cur,
        theory_name=theory_name,
        ascii_statement=statement,
        proof_text=proof_text,
        prover=prover,
        status=status,
        module_name=result.get("module"),
        stdout=result.get("stdout"),
        stderr=result.get("stderr"),
        derivation_rule=result.get("derivation_rule"),
        derivation_depth=result.get("derivation_depth"),
        method=result.get("method"),
        duration_ms=result.get("duration_ms"),
        ui_events=ui_events,
        sealed_by="first_organism_test"
    )
    
    # Fetch the timestamp from the block record (not in BlockRecord by default, need to query or derive)
    # BlockRecord in ingest.py doesn't have timestamp. 
    # However, the DB has 'sealed_at'.
    # Let's query it to be precise, or rely on determinism if we can.
    # Querying is safer to match DB state.
    cur.execute("SELECT sealed_at FROM blocks WHERE id = %s", (outcome.block.id,))
    row = cur.fetchone()
    timestamp = row[0].isoformat() if row else ""

    return SealedBlock(
        reasoning_root=outcome.block.reasoning_root,
        ui_root=outcome.block.ui_root,
        composite_root=outcome.block.composite_root,
        block_id=outcome.block.id,
        sequence=outcome.block.number,
        timestamp=timestamp,
    )

