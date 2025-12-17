"""Tests for Monotone Ledger Guard.

Tests the monotonicity checking for ledger operations.
"""

import json
from dataclasses import asdict

import pytest

from backend.ledger.monotone_guard import (
    MonotoneViolation,
    MonotoneCheckResult,
    check_monotone_ledger,
    verify_chain_integrity,
    summarize_blocks_file,
    summarize_monotone_ledger_for_global_health,
    summarize_and_write_tile,
    to_governance_signal_for_ledger,
)


class TestCheckMonotoneLedger:
    """Tests for check_monotone_ledger function."""

    def test_empty_blocks(self):
        """Empty block list should be valid."""
        result = check_monotone_ledger([])
        assert result.valid
        assert result.blocks_checked == 0
        assert len(result.violations) == 0

    def test_single_block(self):
        """Single block should be valid."""
        blocks = [{"block_id": 1, "height": 1, "hash": "abc123"}]
        result = check_monotone_ledger(blocks)
        assert result.valid
        assert result.blocks_checked == 1

    def test_valid_chain(self):
        """Valid increasing chain should pass."""
        blocks = [
            {"block_id": 1, "height": 1, "hash": "hash1", "timestamp": 100},
            {"block_id": 2, "height": 2, "hash": "hash2", "prev_hash": "hash1", "timestamp": 200},
            {"block_id": 3, "height": 3, "hash": "hash3", "prev_hash": "hash2", "timestamp": 300},
        ]
        result = check_monotone_ledger(blocks)
        assert result.valid
        assert result.blocks_checked == 3
        assert len(result.violations) == 0

    def test_height_violation(self):
        """Non-increasing height should fail."""
        blocks = [
            {"block_id": 1, "height": 5},
            {"block_id": 2, "height": 3},  # Violation: 3 < 5
        ]
        result = check_monotone_ledger(blocks)
        assert not result.valid
        assert len(result.violations) == 1
        assert result.violations[0].violation_type == "height"

    def test_hash_chain_violation(self):
        """Broken hash chain should fail."""
        blocks = [
            {"block_id": 1, "height": 1, "hash": "correct_hash"},
            {"block_id": 2, "height": 2, "prev_hash": "wrong_hash"},
        ]
        result = check_monotone_ledger(blocks)
        assert not result.valid
        assert any(v.violation_type == "hash_chain" for v in result.violations)

    def test_timestamp_violation(self):
        """Decreasing timestamp should fail."""
        blocks = [
            {"block_id": 1, "height": 1, "timestamp": 1000},
            {"block_id": 2, "height": 2, "timestamp": 500},  # Violation
        ]
        result = check_monotone_ledger(blocks)
        assert not result.valid
        assert any(v.violation_type == "timestamp" for v in result.violations)


class TestVerifyChainIntegrity:
    """Tests for verify_chain_integrity function."""

    def test_valid_chain_returns_true(self):
        """Valid chain should return True with empty errors."""
        blocks = [
            {"height": 1, "hash": "h1"},
            {"height": 2, "hash": "h2", "prev_hash": "h1"},
        ]
        valid, errors = verify_chain_integrity(blocks)
        assert valid
        assert errors == []

    def test_invalid_chain_returns_errors(self):
        """Invalid chain should return False with error messages."""
        blocks = [
            {"height": 2},
            {"height": 1},  # Violation
        ]
        valid, errors = verify_chain_integrity(blocks)
        assert not valid
        assert len(errors) > 0


class TestSummarizeLedger:
    """Tests for monotone ledger summaries."""

    def test_healthy_summary(self):
        """Valid chain should produce OK status with ledger_monotone True."""
        blocks = [
            {"block_id": 1, "height": 1, "hash": "h1", "timestamp": 10},
            {"block_id": 2, "height": 2, "hash": "h2", "prev_hash": "h1", "timestamp": 11},
        ]
        result = asdict(check_monotone_ledger(blocks))
        summary = summarize_monotone_ledger_for_global_health(result)
        assert summary["status"] == "OK"
        assert summary["ledger_monotone"] is True
        assert summary["violation_count"] == 0

    def test_warning_summary(self):
        """Missing hash linkage should produce WARN status without violations."""
        blocks = [
            {"block_id": 1, "height": 1, "hash": "h1"},
            {"block_id": 2, "height": 2},  # warning due to missing prev_hash
        ]
        result = asdict(check_monotone_ledger(blocks))
        summary = summarize_monotone_ledger_for_global_health(result)
        assert summary["status"] == "WARN"
        assert summary["ledger_monotone"] is True
        assert summary["violation_count"] == 0

    def test_violation_summary(self):
        """Invalid chain should produce BLOCK status."""
        blocks = [
            {"block_id": 1, "height": 5},
            {"block_id": 2, "height": 1},
        ]
        result = asdict(check_monotone_ledger(blocks))
        summary = summarize_monotone_ledger_for_global_health(result)
        assert summary["status"] == "BLOCK"
        assert summary["ledger_monotone"] is False
        assert summary["violation_count"] > 0

    def test_governance_signal_mapping(self):
        """Governance signal should encode severity and metadata."""
        blocks = [
            {"block_id": 1, "height": 1},
            {"block_id": 2, "height": 0},
        ]
        summary = summarize_monotone_ledger_for_global_health(
            asdict(check_monotone_ledger(blocks))
        )
        signal = to_governance_signal_for_ledger(summary)
        assert signal["status"] == "BLOCK"
        assert signal["severity"] == "blocking"
        assert signal["metadata"]["violation_count"] == summary["violation_count"]


class TestLedgerGovernanceSignals:
    """Tests for governance signal adapters and tile outputs."""

    def test_governance_signal_levels(self):
        """All statuses should map to severity/code consistently."""
        expectations = {
            "OK": ("non_blocking", "LEDGER-MONO-OK", True),
            "WARN": ("advisory", "LEDGER-MONO-WARN", True),
            "BLOCK": ("blocking", "LEDGER-MONO-BLOCK", False),
        }
        for status, (severity, code, monotone_ok) in expectations.items():
            summary = {
                "schema_version": "1.0.0",
                "status": status,
                "ledger_monotone": monotone_ok,
                "violation_count": 2 if status == "BLOCK" else 0,
                "headline": f"status {status}",
            }
            signal = to_governance_signal_for_ledger(summary)
            assert signal["status"] == status
            assert signal["severity"] == severity
            assert signal["code"] == code
            assert signal["metadata"]["ledger_monotone"] is monotone_ok
            assert signal["metadata"]["violation_count"] == summary["violation_count"]

    def test_summarize_and_write_tile_outputs_json(self, tmp_path):
        """Tile + governance signal should be written deterministically."""
        blocks = [
            {"block_id": 1, "height": 1, "hash": "h1"},
            {"block_id": 2, "height": 2, "prev_hash": "h1", "hash": "h2"},
        ]
        blocks_path = tmp_path / "blocks.jsonl"
        with blocks_path.open("w", encoding="utf-8") as handle:
            for block in blocks:
                handle.write(json.dumps(block) + "\n")

        tile_path = tmp_path / "tile.json"
        signal_path = tmp_path / "signal.json"
        summary = summarize_and_write_tile(
            blocks_path, tile_output=tile_path, signal_output=signal_path
        )

        assert tile_path.exists()
        assert signal_path.exists()

        tile_payload = json.loads(tile_path.read_text(encoding="utf-8"))
        signal_payload = json.loads(signal_path.read_text(encoding="utf-8"))

        assert tile_payload == summary
        assert signal_payload["status"] == summary["status"]
        assert signal_payload["signal"] == "ledger_monotone"
        assert "schema_version" in tile_payload
