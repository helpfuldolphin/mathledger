from __future__ import annotations

import copy

from attestation.dual_root import compute_composite_root

from ledger.block_schema import (
    BLOCK_SCHEMA_CONTRACT,
    BlockSchemaEnvelope,
    SealedBlockSchema,
    seal_block_schema,
    validate_block_schema,
)
from backend.ledger.monotone_guard import check_monotone_invariants
from substrate.crypto.hashing import sha256_hex


def _fake_root(seed: str) -> str:
    return sha256_hex(f"root::{seed}")


def _seal_block(
    block_number: int,
    prev_hash: str | None,
    *,
    system_id: str = "pl",
    statements: list[str] | None = None,
) -> SealedBlockSchema:
    stmt_hashes = statements or [f"STMT:{block_number}:a", f"STMT:{block_number}:b"]
    proofs = [
        {
            "hash": sha256_hex(f"proof::{block_number}::{idx}"),
            "statement_hash": stmt_hash,
            "status": "success",
        }
        for idx, stmt_hash in enumerate(stmt_hashes)
    ]
    r_t = _fake_root(f"R{block_number}")
    u_t = _fake_root(f"U{block_number}")
    h_t = compute_composite_root(r_t, u_t)
    return seal_block_schema(
        system_id=system_id,
        block_number=block_number,
        reasoning_root=r_t,
        ui_root=u_t,
        composite_root=h_t,
        statements=stmt_hashes,
        proofs=proofs,
        prev_hash=prev_hash,
    )


def test_seal_block_schema_canonicalizes_statements_and_proofs() -> None:
    statements = ["STMT:z", "STMT:a", "STMT:a"]
    proofs = [
        {"hash": "PROOF:z", "statement_hash": "STMT:z", "status": "success"},
        {"hash": "PROOF:a", "statement_hash": "STMT:a", "status": "success"},
    ]
    r_t = _fake_root("R_demo")
    u_t = _fake_root("U_demo")
    h_t = compute_composite_root(r_t, u_t)

    sealed = seal_block_schema(
        system_id="demo",
        block_number=9,
        reasoning_root=r_t,
        ui_root=u_t,
        composite_root=h_t,
        statements=statements,
        proofs=proofs,
        prev_hash=None,
    )

    assert sealed.statements == ["STMT:a", "STMT:z"]
    assert sealed.canonical_statements[0]["hash"] == "STMT:a"
    assert sealed.canonical_proofs[0]["hash"] == "PROOF:a"
    assert sealed.canonical_proofs[1]["hash"] == "PROOF:z"
    assert sealed.header["statement_count"] == 2
    assert sealed.header["proof_count"] == 2
    assert len(sealed.block_hash) == 64
    assert len(sealed.payload_hash) == 64


def test_check_monotone_invariants_accepts_valid_sequence() -> None:
    block1 = _seal_block(1, None)
    block2 = _seal_block(2, block1.header["composite_root"])
    envelopes = [block1.to_envelope(), block2.to_envelope()]

    violations = check_monotone_invariants(envelopes)

    assert violations == []


def test_check_monotone_invariants_detects_prev_hash_mismatch() -> None:
    block1 = _seal_block(1, None)
    block2 = _seal_block(2, block1.header["composite_root"])
    tampered_header = copy.deepcopy(block2.header)
    tampered_header["prev_hash"] = "BAD_HASH"

    tampered_envelope = BlockSchemaEnvelope(
        header=tampered_header,
        statements=block2.statements,
        canonical_proofs=block2.canonical_proofs,
    )

    violations = check_monotone_invariants(
        [block1.to_envelope(), tampered_envelope],
    )

    assert any(v.invariant_id == "INV-MONO-002" for v in violations)


def test_check_monotone_invariants_detects_unsorted_statements() -> None:
    block = _seal_block(1, None)
    envelope = block.to_envelope()
    tampered = BlockSchemaEnvelope(
        header=envelope.header,
        statements=list(reversed(envelope.statements)),
        canonical_proofs=envelope.canonical_proofs,
    )

    violations = check_monotone_invariants([tampered])

    assert any(v.invariant_id == "INV-MONO-003" for v in violations)


def test_validate_block_schema_blocks_on_missing_required_fields() -> None:
    report = validate_block_schema({"block_number": 10}, BLOCK_SCHEMA_CONTRACT)

    assert report["status"] == "BLOCK"
    assert "system_id" in report["missing_fields"]


def test_validate_block_schema_warns_on_unexpected_fields() -> None:
    block = _seal_block(1, None)
    header = block.header.copy()
    header["extra_field"] = "diagnostic"

    report = validate_block_schema(header, BLOCK_SCHEMA_CONTRACT)

    assert report["status"] == "WARN"
    assert "extra_field" in report["unexpected_fields"]
