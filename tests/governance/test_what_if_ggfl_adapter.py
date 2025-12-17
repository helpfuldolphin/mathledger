"""
Tests for What-If GGFL Adapter.

Tests what_if_for_alignment_view() function:
- Status is "warn" if hypothetical_block_rate > 0
- Drivers include top blocking gate and block rate
- conflict is always False
- Mode must be HYPOTHETICAL
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from backend.governance.fusion import (
    what_if_for_alignment_view,
    WhatIfAlignmentSignal,
    GovernanceAction,
    _extract_what_if_recommendations,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def status_signal_no_blocks() -> Dict[str, Any]:
    """Status signal with no hypothetical blocks."""
    return {
        "hypothetical_block_rate": 0.0,
        "blocking_gate_distribution": {},
        "first_block_cycle": None,
        "total_cycles": 100,
        "hypothetical_blocks": 0,
        "mode": "HYPOTHETICAL",
        "report_sha256": "abc123",
    }


@pytest.fixture
def status_signal_with_blocks() -> Dict[str, Any]:
    """Status signal with hypothetical blocks."""
    return {
        "hypothetical_block_rate": 0.15,
        "blocking_gate_distribution": {
            "G2_INVARIANT": 5,
            "G3_SAFE_REGION": 7,
            "G4_SOFT": 3,
        },
        "first_block_cycle": 12,
        "total_cycles": 100,
        "hypothetical_blocks": 15,
        "mode": "HYPOTHETICAL",
        "report_sha256": "def456",
    }


@pytest.fixture
def full_report_with_blocks() -> Dict[str, Any]:
    """Full What-If report format with blocks."""
    return {
        "schema_version": "1.0.0",
        "run_id": "test-run",
        "mode": "HYPOTHETICAL",
        "summary": {
            "total_cycles": 50,
            "hypothetical_allows": 40,
            "hypothetical_blocks": 10,
            "hypothetical_block_rate": 0.2,
            "blocking_gate_distribution": {
                "G4_SOFT": 8,
                "G2_INVARIANT": 2,
            },
            "first_hypothetical_block_cycle": 5,
        },
    }


@pytest.fixture
def status_signal_wrong_mode() -> Dict[str, Any]:
    """Status signal with wrong mode."""
    return {
        "hypothetical_block_rate": 0.1,
        "blocking_gate_distribution": {"G3_SAFE_REGION": 5},
        "mode": "SHADOW",  # Wrong - should be HYPOTHETICAL
    }


# =============================================================================
# STATUS CONVERSION TESTS
# =============================================================================

class TestStatusConversion:
    """Tests for status conversion (ok/warn)."""

    def test_status_ok_when_no_blocks(self, status_signal_no_blocks):
        """Should return status='ok' when no hypothetical blocks."""
        result = what_if_for_alignment_view(status_signal_no_blocks)

        assert result.status == "ok"
        assert result.hypothetical_block_rate == 0.0

    def test_status_warn_when_blocks_present(self, status_signal_with_blocks):
        """Should return status='warn' when hypothetical_block_rate > 0."""
        result = what_if_for_alignment_view(status_signal_with_blocks)

        assert result.status == "warn"
        assert result.hypothetical_block_rate == 0.15

    def test_status_warn_even_for_small_block_rate(self):
        """Should warn even for very small block rates."""
        signal = {
            "hypothetical_block_rate": 0.001,  # 0.1%
            "blocking_gate_distribution": {"G4_SOFT": 1},
            "mode": "HYPOTHETICAL",
        }
        result = what_if_for_alignment_view(signal)

        assert result.status == "warn"


# =============================================================================
# DRIVERS TESTS
# =============================================================================

class TestDrivers:
    """Tests for driver extraction."""

    def test_drivers_empty_when_no_blocks(self, status_signal_no_blocks):
        """Should have no drivers when no blocks."""
        result = what_if_for_alignment_view(status_signal_no_blocks)

        # No block rate driver, no top gate driver
        assert len(result.drivers) == 0

    def test_drivers_include_block_rate(self, status_signal_with_blocks):
        """Should include block rate in drivers."""
        result = what_if_for_alignment_view(status_signal_with_blocks)

        rate_drivers = [d for d in result.drivers if "hypothetical_block_rate" in d]
        assert len(rate_drivers) == 1
        assert "15.00%" in rate_drivers[0]

    def test_drivers_include_top_gate(self, status_signal_with_blocks):
        """Should include top blocking gate in drivers."""
        result = what_if_for_alignment_view(status_signal_with_blocks)

        gate_drivers = [d for d in result.drivers if "top_blocking_gate" in d]
        assert len(gate_drivers) == 1
        assert "G3_SAFE_REGION" in gate_drivers[0]  # 7 blocks is highest

    def test_drivers_include_mode_warning_when_wrong(self, status_signal_wrong_mode):
        """Should include mode warning driver when mode is not HYPOTHETICAL."""
        result = what_if_for_alignment_view(status_signal_wrong_mode)

        mode_drivers = [d for d in result.drivers if "mode=" in d]
        assert len(mode_drivers) == 1
        assert "SHADOW" in mode_drivers[0]
        assert "expected HYPOTHETICAL" in mode_drivers[0]


# =============================================================================
# TOP BLOCKING GATE TESTS
# =============================================================================

class TestTopBlockingGate:
    """Tests for top blocking gate extraction."""

    def test_top_gate_is_highest_count(self, status_signal_with_blocks):
        """Should identify gate with highest block count."""
        result = what_if_for_alignment_view(status_signal_with_blocks)

        # G3_SAFE_REGION has 7 blocks, highest
        assert result.top_blocking_gate == "G3_SAFE_REGION"

    def test_top_gate_none_when_no_blocks(self, status_signal_no_blocks):
        """Should be None when no blocks."""
        result = what_if_for_alignment_view(status_signal_no_blocks)

        assert result.top_blocking_gate is None

    def test_top_gate_from_full_report(self, full_report_with_blocks):
        """Should extract top gate from full report format."""
        result = what_if_for_alignment_view(full_report_with_blocks)

        # G4_SOFT has 8 blocks, highest
        assert result.top_blocking_gate == "G4_SOFT"


# =============================================================================
# CONFLICT TESTS
# =============================================================================

class TestConflict:
    """Tests for conflict field (always False)."""

    def test_conflict_always_false_no_blocks(self, status_signal_no_blocks):
        """Conflict should be False when no blocks."""
        result = what_if_for_alignment_view(status_signal_no_blocks)
        assert result.conflict is False

    def test_conflict_always_false_with_blocks(self, status_signal_with_blocks):
        """Conflict should be False even with blocks."""
        result = what_if_for_alignment_view(status_signal_with_blocks)
        assert result.conflict is False

    def test_conflict_always_false_wrong_mode(self, status_signal_wrong_mode):
        """Conflict should be False even with wrong mode."""
        result = what_if_for_alignment_view(status_signal_wrong_mode)
        assert result.conflict is False


# =============================================================================
# MODE TESTS
# =============================================================================

class TestMode:
    """Tests for mode field."""

    def test_mode_preserved_from_input(self, status_signal_with_blocks):
        """Should preserve mode from input."""
        result = what_if_for_alignment_view(status_signal_with_blocks)
        assert result.mode == "HYPOTHETICAL"

    def test_mode_wrong_value_preserved(self, status_signal_wrong_mode):
        """Should preserve wrong mode value (for driver warning)."""
        result = what_if_for_alignment_view(status_signal_wrong_mode)
        assert result.mode == "SHADOW"

    def test_mode_defaults_to_hypothetical(self):
        """Should default to HYPOTHETICAL if not provided."""
        signal = {"hypothetical_block_rate": 0.0}
        result = what_if_for_alignment_view(signal)
        assert result.mode == "HYPOTHETICAL"


# =============================================================================
# INPUT FORMAT TESTS
# =============================================================================

class TestInputFormats:
    """Tests for different input formats."""

    def test_accepts_status_signal_format(self, status_signal_with_blocks):
        """Should accept WhatIfStatusSignal.to_dict() format."""
        result = what_if_for_alignment_view(status_signal_with_blocks)

        assert result.hypothetical_block_rate == 0.15
        assert result.status == "warn"

    def test_accepts_full_report_format(self, full_report_with_blocks):
        """Should accept full WhatIfReport.to_dict() format."""
        result = what_if_for_alignment_view(full_report_with_blocks)

        assert result.hypothetical_block_rate == 0.2
        assert result.status == "warn"
        assert result.top_blocking_gate == "G4_SOFT"

    def test_accepts_minimal_dict(self):
        """Should accept minimal dict with just required fields."""
        signal = {"hypothetical_block_rate": 0.05}
        result = what_if_for_alignment_view(signal)

        assert result.hypothetical_block_rate == 0.05
        assert result.status == "warn"


# =============================================================================
# TO_DICT OUTPUT TESTS
# =============================================================================

class TestToDictOutput:
    """Tests for to_dict() output format."""

    def test_to_dict_has_signal_id(self, status_signal_with_blocks):
        """Output should have signal_id = SIG-WIF."""
        result = what_if_for_alignment_view(status_signal_with_blocks)
        output = result.to_dict()

        assert output["signal_id"] == "SIG-WIF"

    def test_to_dict_has_required_fields(self, status_signal_with_blocks):
        """Output should have all required fields."""
        result = what_if_for_alignment_view(status_signal_with_blocks)
        output = result.to_dict()

        assert "status" in output
        assert "hypothetical_block_rate" in output
        assert "top_blocking_gate" in output
        assert "drivers" in output
        assert "conflict" in output
        assert "mode" in output

    def test_to_dict_block_rate_rounded(self, status_signal_with_blocks):
        """Block rate should be rounded to 4 decimal places."""
        signal = dict(status_signal_with_blocks)
        signal["hypothetical_block_rate"] = 0.123456789

        result = what_if_for_alignment_view(signal)
        output = result.to_dict()

        assert output["hypothetical_block_rate"] == 0.1235


# =============================================================================
# RECOMMENDATIONS EXTRACTION TESTS
# =============================================================================

class TestRecommendationsExtraction:
    """Tests for _extract_what_if_recommendations()."""

    def test_allow_when_no_blocks(self):
        """Should produce ALLOW when no blocks."""
        signal = {
            "hypothetical_block_rate": 0.0,
            "mode": "HYPOTHETICAL",
        }
        recs = _extract_what_if_recommendations(signal)

        assert len(recs) == 1
        assert recs[0].action == GovernanceAction.ALLOW
        assert recs[0].signal_id == "what_if"

    def test_warning_when_blocks_present(self):
        """Should produce WARNING when blocks present."""
        signal = {
            "hypothetical_block_rate": 0.15,
            "top_blocking_gate": "G3_SAFE_REGION",
            "mode": "HYPOTHETICAL",
        }
        recs = _extract_what_if_recommendations(signal)

        warnings = [r for r in recs if r.action == GovernanceAction.WARNING]
        assert len(warnings) == 1
        assert "15.0%" in warnings[0].reason
        assert "G3_SAFE_REGION" in warnings[0].reason

    def test_warning_for_wrong_mode(self):
        """Should produce WARNING for wrong mode."""
        signal = {
            "hypothetical_block_rate": 0.0,
            "mode": "SHADOW",
        }
        recs = _extract_what_if_recommendations(signal)

        mode_warnings = [r for r in recs if "mode" in r.reason.lower()]
        assert len(mode_warnings) == 1
        assert "expected 'HYPOTHETICAL'" in mode_warnings[0].reason

    def test_never_produces_block(self):
        """Should never produce BLOCK or HARD_BLOCK."""
        signal = {
            "hypothetical_block_rate": 0.99,  # Very high
            "top_blocking_gate": "G2_INVARIANT",
            "mode": "SHADOW",  # Wrong mode too
        }
        recs = _extract_what_if_recommendations(signal)

        blocks = [r for r in recs if r.action in (GovernanceAction.BLOCK, GovernanceAction.HARD_BLOCK)]
        assert len(blocks) == 0

    def test_low_priority(self):
        """What-If recommendations should have low priority."""
        signal = {
            "hypothetical_block_rate": 0.5,
            "mode": "HYPOTHETICAL",
        }
        recs = _extract_what_if_recommendations(signal)

        for rec in recs:
            assert rec.priority <= 2  # Low priority (advisory only)
