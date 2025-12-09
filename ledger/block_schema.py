"""
Canonical block schema construction and schema contract validation.

This module centralizes:
- Block header/payload canonicalization
- Payload hashing and header sealing
- Schema contract enforcement (monotone guard lives in ledger.monotone_guard)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from substrate.crypto.hashing import DOMAIN_ROOT, hash_block, sha256_hex
from substrate.repro.determinism import deterministic_timestamp_from_content


def _canonical_json(data: Any) -> str:
    """RFC8785-like canonical JSON serialization."""
    return json.dumps(data, separators=(",", ":"), sort_keys=True)


def _isoformat(dt: datetime) -> str:
    if dt.tzinfo is None:
        raise ValueError("sealed_at timestamp must be timezone-aware")
    return dt.isoformat(timespec="microseconds")


@dataclass(frozen=True)
class BlockSchemaContract:
    """Versioned schema contract metadata."""

    schema_version: str
    required_fields: Tuple[str, ...]
    optional_fields: Tuple[str, ...]
    monotone_fields: Tuple[str, ...]


BLOCK_SCHEMA_CONTRACT = BlockSchemaContract(
    schema_version="1.0.0",
    required_fields=(
        "block_number",
        "system_id",
        "reasoning_root",
        "ui_root",
        "composite_root",
        "statement_count",
        "proof_count",
        "sealed_at",
        "prev_hash",
        "version",
    ),
    optional_fields=(),
    monotone_fields=("block_number",),
)


@dataclass(frozen=True)
class SealedBlockSchema:
    """Container for deterministic block header/payload artifacts."""

    header: Dict[str, Any]
    statements: List[str]
    canonical_statements: List[Dict[str, Any]]
    canonical_proofs: List[Dict[str, Any]]
    block_hash: str
    payload_hash: str
    sealed_at: datetime

    def to_envelope(self) -> "BlockSchemaEnvelope":
        """Compress sealed footprint into invariant-checkable view."""
        return BlockSchemaEnvelope(
            header=self.header,
            statements=self.statements,
            canonical_proofs=self.canonical_proofs,
        )


@dataclass(frozen=True)
class BlockSchemaEnvelope:
    """Minimal view for running monotone invariant checks."""

    header: Mapping[str, Any]
    statements: Sequence[str]
    canonical_proofs: Sequence[Mapping[str, Any]]


def validate_block_schema(
    block: Mapping[str, Any],
    contract: BlockSchemaContract = BLOCK_SCHEMA_CONTRACT,
) -> Dict[str, Any]:
    """
    Validate a block header structure against the schema contract.

    Returns dict containing:
    - schema_version
    - missing_fields
    - unexpected_fields
    - status: "OK" | "WARN" | "BLOCK"
    """
    present_keys = set(block.keys())
    required = set(contract.required_fields)
    optional = set(contract.optional_fields)
    missing = set(field for field in contract.required_fields if field not in present_keys)
    missing.update(field for field in contract.monotone_fields if field not in present_keys)
    unexpected = {key for key in present_keys if key not in required and key not in optional}

    status = "OK"
    if missing:
        status = "BLOCK"
    elif unexpected:
        status = "WARN"

    return {
        "schema_version": contract.schema_version,
        "missing_fields": tuple(sorted(missing)),
        "unexpected_fields": tuple(sorted(unexpected)),
        "status": status,
    }


def seal_block_schema(
    *,
    system_id: str,
    block_number: int,
    reasoning_root: str,
    ui_root: str,
    composite_root: str,
    statements: Sequence[str],
    proofs: Sequence[Mapping[str, Any]],
    prev_hash: Optional[str],
    version: str = BLOCK_SCHEMA_CONTRACT.schema_version,
    sealed_at: Optional[datetime] = None,
    extra_header_fields: Optional[Mapping[str, Any]] = None,
) -> SealedBlockSchema:
    """
    Produce canonical block header/payload documents plus hashes.

    Args:
        system_id: Ledger namespace identifier.
        block_number: Deterministic block height (1-indexed).
        reasoning_root: R_t value for proofs.
        ui_root: U_t value for UI events.
        composite_root: H_t binding of R_t and U_t.
        statements: Statement hashes referenced by proofs.
        proofs: Iterable of proof descriptors with hash + statement_hash.
        prev_hash: Hash of the parent composite/block root.
        version: Header schema version tag.
        sealed_at: Optional deterministic timestamp override.
        extra_header_fields: Additional header metadata (deterministic).
    """
    if sealed_at is None:
        sealed_at = deterministic_timestamp_from_content(composite_root)

    sorted_statements = sorted(set(statements))
    statement_entries = [
        {"position": idx, "hash": value} for idx, value in enumerate(sorted_statements)
    ]

    sorted_proofs = _canonicalize_proofs(proofs)

    header_doc: Dict[str, Any] = {
        "block_number": block_number,
        "system_id": system_id,
        "reasoning_root": reasoning_root,
        "ui_root": ui_root,
        "composite_root": composite_root,
        "statement_count": len(sorted_statements),
        "proof_count": len(sorted_proofs),
        "sealed_at": _isoformat(sealed_at),
        "prev_hash": prev_hash,
        "version": version,
    }
    if extra_header_fields:
        header_doc.update(extra_header_fields)

    payload_doc = {
        "statements": sorted_statements,
        "proofs": [
            {
                "hash": proof["hash"],
                "statement_hash": proof["statement_hash"],
                "status": proof.get("status", ""),
            }
            for proof in sorted_proofs
        ],
    }

    payload_hash = sha256_hex(_canonical_json(payload_doc), domain=DOMAIN_ROOT)
    block_hash = hash_block(_canonical_json(header_doc))

    canonical_proofs = [
        {
            "position": idx,
            "hash": proof["hash"],
            "statement_hash": proof["statement_hash"],
            "status": proof.get("status", ""),
        }
        for idx, proof in enumerate(sorted_proofs)
    ]

    return SealedBlockSchema(
        header=header_doc,
        statements=sorted_statements,
        canonical_statements=statement_entries,
        canonical_proofs=canonical_proofs,
        block_hash=block_hash,
        payload_hash=payload_hash,
        sealed_at=sealed_at,
    )


def _canonicalize_proofs(proofs: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    def _proof_key(payload: Mapping[str, Any]) -> Tuple[str, str]:
        return payload["hash"], payload["statement_hash"]

    return sorted(proofs, key=_proof_key)
