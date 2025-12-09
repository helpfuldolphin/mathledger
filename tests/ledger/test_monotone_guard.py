from __future__ import annotations

from typing import Optional

from backend.ledger.monotone_guard import (
    check_monotone_ledger,
    summarize_monotone_ledger_for_global_health,
    to_governance_signal_for_ledger,
)
from attestation.dual_root import compute_composite_root
from ledger.block_schema import SealedBlockSchema, seal_block_schema
from substrate.crypto.hashing import sha256_hex


def _fake_root(seed: str) -> str:
    return sha256_hex(f"root::{seed}")


def _build_block(
    block_number: int,
    prev_header: Optional[dict] = None,
) -> SealedBlockSchema:
    prev_hash = prev_header["composite_root"] if prev_header else None
    statements = [f"STMT:{block_number}:a", f"STMT:{block_number}:b"]
    proofs = [
        {
            "hash": sha256_hex(f"proof::{block_number}::{idx}"),
            "statement_hash": stmt_hash,
            "status": "success",
        }
        for idx, stmt_hash in enumerate(statements)
    ]
    r_t = _fake_root(f"R{block_number}")
    u_t = _fake_root(f"U{block_number}")
    h_t = compute_composite_root(r_t, u_t)
    return seal_block_schema(
        system_id="pl",
        block_number=block_number,
        reasoning_root=r_t,
        ui_root=u_t,
        composite_root=h_t,
        statements=statements,
        proofs=proofs,
        prev_hash=prev_hash,
    )


def test_check_monotone_ledger_empty_sequence_ok() -> None:
    summary = check_monotone_ledger([])

    assert summary["status"] == "OK"
    assert summary["total_blocks"] == 0
    assert summary["warnings"] == []
    assert summary["violations"] == []


def test_check_monotone_ledger_strict_sequence_ok() -> None:
    block1 = _build_block(1)
    block2 = _build_block(2, block1.header)
    block3 = _build_block(3, block2.header)

    summary = check_monotone_ledger(
        [block1.header, block2.header, block3.header],
    )

    assert summary["status"] == "OK"
    assert summary["warnings"] == []
    assert summary["violations"] == []
    assert summary["total_blocks"] == 3


def test_check_monotone_ledger_detects_reordered_block() -> None:
    block1 = _build_block(1)
    block2 = _build_block(2, block1.header)
    block3 = _build_block(3, block2.header)

    summary = check_monotone_ledger(
        [block1.header, block3.header, block2.header],
    )

    assert summary["status"] == "BLOCK"
    assert any("expected block_number" in msg for msg in summary["violations"])


def test_check_monotone_ledger_detects_duplicate_block_number() -> None:
    block1 = _build_block(1)
    block2 = _build_block(2, block1.header)
    duplicate = dict(block2.header)
    duplicate["block_number"] = 1

    summary = check_monotone_ledger([block1.header, duplicate])

    assert summary["status"] == "BLOCK"
    assert any("duplicate block_number" in msg for msg in summary["violations"])


def test_check_monotone_ledger_warn_on_unexpected_field() -> None:
    block1 = _build_block(1)
    tampered_header = dict(block1.header)
    tampered_header["extra_field"] = "diagnostic"
    block2 = _build_block(2, block1.header)

    result = check_monotone_ledger([tampered_header, block2.header])

    assert result["status"] == "WARN"
    assert not result["violations"]
    assert result["warnings"]

    summary = summarize_monotone_ledger_for_global_health(result)
    assert summary["status"] == "WARN"
    assert summary["ledger_monotone"] is True
    assert summary["violation_count"] == 0

    signal = to_governance_signal_for_ledger(summary)
    assert signal["status"] == "WARN"
    assert signal["metadata"]["ledger_monotone"] is True
