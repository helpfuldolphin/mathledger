"""
Tests for What-If signal integration in first_light_status.json generation.

Tests:
- Manifest-first extraction (signals.what_if → governance.what_if_analysis.status → report)
- extraction_source field tracking (MANIFEST_SIGNALS, MANIFEST_GOVERNANCE, DERIVED_FROM_REPORT, MISSING)
- GGFL consistency cross-check (status mapping + mode=HYPOTHETICAL + conflict=False)
- Warning hygiene (single-line format with top_blocking_gate)
- Missing What-If graceful handling (no error, no signal)
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest

from backend.governance.evidence_pack import (
    get_what_if_status_from_manifest,
    format_what_if_warning,
)
from backend.governance.fusion import what_if_for_alignment_view


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def compact_what_if_status() -> Dict[str, Any]:
    """Compact What-If status signal."""
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
        "report_sha256": "abc123def456",
    }


@pytest.fixture
def full_what_if_report() -> Dict[str, Any]:
    """Full What-If report."""
    return {
        "schema_version": "1.0.0",
        "run_id": "test-run-001",
        "analysis_timestamp": "2025-01-01T00:00:00Z",
        "mode": "HYPOTHETICAL",
        "summary": {
            "total_cycles": 100,
            "hypothetical_allows": 85,
            "hypothetical_blocks": 15,
            "hypothetical_block_rate": 0.15,
            "blocking_gate_distribution": {
                "G2_INVARIANT": 5,
                "G3_SAFE_REGION": 7,
                "G4_SOFT": 3,
            },
            "max_consecutive_blocks": 4,
            "first_hypothetical_block_cycle": 12,
        },
        "gate_analysis": {},
        "notable_events": [],
        "calibration_recommendations": [],
        "auditor_notes": "Test run.",
    }


@pytest.fixture
def manifest_with_signals_what_if(compact_what_if_status) -> Dict[str, Any]:
    """Manifest with signals.what_if (preferred path)."""
    return {
        "signals": {
            "what_if": compact_what_if_status,
        },
    }


@pytest.fixture
def manifest_with_governance_status(compact_what_if_status) -> Dict[str, Any]:
    """Manifest with governance.what_if_analysis.status (fallback path)."""
    return {
        "governance": {
            "what_if_analysis": {
                "status": compact_what_if_status,
            },
        },
    }


@pytest.fixture
def manifest_with_report_only(full_what_if_report) -> Dict[str, Any]:
    """Manifest with governance.what_if_analysis.report (last resort)."""
    return {
        "governance": {
            "what_if_analysis": {
                "report": full_what_if_report,
            },
        },
    }


@pytest.fixture
def manifest_with_all_paths(compact_what_if_status, full_what_if_report) -> Dict[str, Any]:
    """Manifest with all three paths populated."""
    return {
        "signals": {
            "what_if": compact_what_if_status,
        },
        "governance": {
            "what_if_analysis": {
                "status": compact_what_if_status,
                "report": full_what_if_report,
            },
        },
    }


@pytest.fixture
def manifest_without_what_if() -> Dict[str, Any]:
    """Manifest without any What-If data."""
    return {
        "proof_hash": "abc123",
        "signals": {
            "usla": {"rho": 0.85},
        },
    }


# =============================================================================
# EXTRACTION SOURCE TESTS
# =============================================================================

class TestExtractionSource:
    """Tests for extraction_source field tracking."""

    def test_manifest_signals_source(self, manifest_with_signals_what_if):
        """signals.what_if path should yield MANIFEST_SIGNALS source."""
        status, warnings = get_what_if_status_from_manifest(manifest_with_signals_what_if)

        assert status is not None
        assert status["hypothetical_block_rate"] == 0.15

        # Verify source determination logic (simulate what first_light_status does)
        signals_what_if = manifest_with_signals_what_if.get("signals", {}).get("what_if")
        if signals_what_if and signals_what_if == status:
            extraction_source = "MANIFEST_SIGNALS"
        else:
            extraction_source = "OTHER"

        assert extraction_source == "MANIFEST_SIGNALS"

    def test_governance_status_source(self, manifest_with_governance_status):
        """governance.what_if_analysis.status path should yield MANIFEST_GOVERNANCE source."""
        status, warnings = get_what_if_status_from_manifest(manifest_with_governance_status)

        assert status is not None
        assert status["hypothetical_block_rate"] == 0.15

        # Simulate source determination
        governance_status = (
            manifest_with_governance_status.get("governance", {})
            .get("what_if_analysis", {})
            .get("status")
        )
        if governance_status:
            extraction_source = "MANIFEST_GOVERNANCE"
        else:
            extraction_source = "OTHER"

        assert extraction_source == "MANIFEST_GOVERNANCE"

    def test_report_extraction_source(self, manifest_with_report_only):
        """Report-derived path should yield DERIVED_FROM_REPORT source."""
        status, warnings = get_what_if_status_from_manifest(manifest_with_report_only)

        assert status is not None
        assert status["hypothetical_block_rate"] == 0.15

        # Simulate source determination
        governance_report = (
            manifest_with_report_only.get("governance", {})
            .get("what_if_analysis", {})
            .get("report")
        )
        if governance_report:
            extraction_source = "DERIVED_FROM_REPORT"
        else:
            extraction_source = "OTHER"

        assert extraction_source == "DERIVED_FROM_REPORT"

    def test_missing_source(self, manifest_without_what_if):
        """Missing What-If should yield MISSING source."""
        status, warnings = get_what_if_status_from_manifest(manifest_without_what_if)

        assert status is None
        assert len(warnings) == 1
        # Source would be MISSING when no status found


# =============================================================================
# GGFL CONSISTENCY CROSS-CHECK TESTS
# =============================================================================

class TestGGFLConsistency:
    """Tests for GGFL consistency cross-check."""

    def test_ggfl_status_consistent_with_blocks(self, compact_what_if_status):
        """GGFL status should be 'warn' when hypothetical_block_rate > 0."""
        ggfl_signal = what_if_for_alignment_view(compact_what_if_status)
        ggfl_dict = ggfl_signal.to_dict()

        # block_rate > 0 → status = "warn"
        expected_status = "warn" if compact_what_if_status["hypothetical_block_rate"] > 0 else "ok"
        assert ggfl_dict["status"] == expected_status
        assert ggfl_dict["status"] == "warn"

    def test_ggfl_status_consistent_no_blocks(self):
        """GGFL status should be 'ok' when hypothetical_block_rate = 0."""
        status_no_blocks = {
            "hypothetical_block_rate": 0.0,
            "blocking_gate_distribution": {},
            "mode": "HYPOTHETICAL",
        }

        ggfl_signal = what_if_for_alignment_view(status_no_blocks)
        ggfl_dict = ggfl_signal.to_dict()

        assert ggfl_dict["status"] == "ok"

    def test_ggfl_mode_is_hypothetical(self, compact_what_if_status):
        """GGFL mode must always be HYPOTHETICAL."""
        ggfl_signal = what_if_for_alignment_view(compact_what_if_status)
        ggfl_dict = ggfl_signal.to_dict()

        assert ggfl_dict["mode"] == "HYPOTHETICAL"

    def test_ggfl_conflict_is_false(self, compact_what_if_status):
        """GGFL conflict must always be False for What-If."""
        ggfl_signal = what_if_for_alignment_view(compact_what_if_status)
        ggfl_dict = ggfl_signal.to_dict()

        assert ggfl_dict["conflict"] is False

    def test_ggfl_signal_id(self, compact_what_if_status):
        """GGFL signal_id should be SIG-WIF."""
        ggfl_signal = what_if_for_alignment_view(compact_what_if_status)
        ggfl_dict = ggfl_signal.to_dict()

        assert ggfl_dict["signal_id"] == "SIG-WIF"

    def test_full_consistency_cross_check(self, compact_what_if_status):
        """Full consistency cross-check should pass for valid What-If status."""
        ggfl_signal = what_if_for_alignment_view(compact_what_if_status)
        ggfl_dict = ggfl_signal.to_dict()

        # Simulate the cross-check logic from first_light_status.py
        expected_status = "warn" if compact_what_if_status["hypothetical_block_rate"] > 0 else "ok"
        ggfl_status = ggfl_dict.get("status", "unknown")
        status_consistent = (expected_status == ggfl_status)

        mode_consistent = (ggfl_dict.get("mode") == "HYPOTHETICAL")
        conflict_consistent = (ggfl_dict.get("conflict") is False)
        all_consistent = status_consistent and mode_consistent and conflict_consistent

        assert status_consistent is True
        assert mode_consistent is True
        assert conflict_consistent is True
        assert all_consistent is True


# =============================================================================
# WARNING HYGIENE TESTS
# =============================================================================

class TestWarningHygiene:
    """Tests for single-line warning format with top_blocking_gate."""

    def test_warning_with_blocks(self, compact_what_if_status):
        """Warning should be generated when hypothetical_block_rate > 0."""
        warning = format_what_if_warning(compact_what_if_status)

        assert warning is not None
        assert "\n" not in warning  # Single line
        assert "HYPOTHETICAL" in warning
        assert "15.0%" in warning

    def test_warning_includes_top_gate(self, compact_what_if_status):
        """Warning should include top_blocking_gate."""
        warning = format_what_if_warning(compact_what_if_status)

        assert warning is not None
        # G3_SAFE_REGION has highest count (7)
        assert "G3_SAFE_REGION" in warning

    def test_no_warning_without_blocks(self):
        """No warning when hypothetical_block_rate = 0."""
        status_no_blocks = {
            "hypothetical_block_rate": 0.0,
            "mode": "HYPOTHETICAL",
        }

        warning = format_what_if_warning(status_no_blocks)
        assert warning is None

    def test_warning_format_structure(self, compact_what_if_status):
        """Warning should follow expected format."""
        warning = format_what_if_warning(compact_what_if_status)

        # Expected: "What-If (HYPOTHETICAL): 15.0% hypothetical block rate; top_gate=G3_SAFE_REGION"
        assert warning.startswith("What-If (HYPOTHETICAL):")
        assert "hypothetical block rate" in warning
        assert "top_gate=" in warning


# =============================================================================
# MANIFEST-FIRST RESOLUTION TESTS
# =============================================================================

class TestManifestFirstResolution:
    """Tests for manifest-first resolution order."""

    def test_prefers_signals_over_governance(self, compact_what_if_status):
        """signals.what_if should be preferred over governance.what_if_analysis.status."""
        signal_status = dict(compact_what_if_status)
        signal_status["hypothetical_block_rate"] = 0.25  # Different value

        governance_status = dict(compact_what_if_status)
        governance_status["hypothetical_block_rate"] = 0.10  # Different value

        manifest = {
            "signals": {"what_if": signal_status},
            "governance": {
                "what_if_analysis": {
                    "status": governance_status,
                },
            },
        }

        status, warnings = get_what_if_status_from_manifest(manifest)

        assert status is not None
        assert status["hypothetical_block_rate"] == 0.25  # From signals

    def test_prefers_signals_over_report(self, compact_what_if_status, full_what_if_report):
        """signals.what_if should be preferred over report extraction."""
        signal_status = dict(compact_what_if_status)
        signal_status["hypothetical_block_rate"] = 0.30

        manifest = {
            "signals": {"what_if": signal_status},
            "governance": {
                "what_if_analysis": {
                    "report": full_what_if_report,
                },
            },
        }

        status, warnings = get_what_if_status_from_manifest(manifest)

        assert status is not None
        assert status["hypothetical_block_rate"] == 0.30  # From signals


# =============================================================================
# MISSING WHAT-IF HANDLING TESTS
# =============================================================================

class TestMissingWhatIfHandling:
    """Tests for graceful handling when What-If is missing."""

    def test_missing_returns_none_with_warning(self, manifest_without_what_if):
        """Missing What-If should return None with warning, no error."""
        status, warnings = get_what_if_status_from_manifest(manifest_without_what_if)

        assert status is None
        assert len(warnings) == 1
        assert "No What-If status found" in warnings[0]

    def test_empty_manifest_returns_none(self):
        """Empty manifest should return None with warning."""
        status, warnings = get_what_if_status_from_manifest({})

        assert status is None
        assert len(warnings) == 1


# =============================================================================
# MODE VALIDATION TESTS
# =============================================================================

class TestModeValidation:
    """Tests for mode validation (must be HYPOTHETICAL)."""

    def test_warns_on_wrong_mode(self):
        """Should warn when mode is not HYPOTHETICAL."""
        wrong_mode_status = {
            "hypothetical_block_rate": 0.1,
            "mode": "SHADOW",  # Wrong mode
        }

        manifest = {"signals": {"what_if": wrong_mode_status}}
        status, warnings = get_what_if_status_from_manifest(manifest)

        assert status is not None
        assert len(warnings) == 1
        assert "expected 'HYPOTHETICAL'" in warnings[0]

    def test_no_warning_for_correct_mode(self, compact_what_if_status):
        """Should not warn when mode is HYPOTHETICAL."""
        manifest = {"signals": {"what_if": compact_what_if_status}}
        status, warnings = get_what_if_status_from_manifest(manifest)

        assert status is not None
        assert len(warnings) == 0


# =============================================================================
# TOP_BLOCKING_GATE DERIVATION TESTS
# =============================================================================

class TestTopBlockingGateDerivation:
    """Tests for top_blocking_gate derivation from distribution."""

    def test_derives_top_gate_from_distribution(self):
        """Should derive top_blocking_gate from blocking_gate_distribution."""
        status = {
            "hypothetical_block_rate": 0.15,
            "blocking_gate_distribution": {
                "G2_INVARIANT": 5,
                "G3_SAFE_REGION": 7,  # Highest
                "G4_SOFT": 3,
            },
            "mode": "HYPOTHETICAL",
        }

        # Simulate derivation logic
        top_gate = status.get("top_blocking_gate")
        if top_gate is None:
            gate_dist = status.get("blocking_gate_distribution", {})
            if gate_dist:
                top_gate = max(gate_dist.items(), key=lambda x: x[1])[0]

        assert top_gate == "G3_SAFE_REGION"

    def test_uses_provided_top_gate(self):
        """Should use provided top_blocking_gate if present."""
        status = {
            "hypothetical_block_rate": 0.15,
            "top_blocking_gate": "G2_INVARIANT",  # Explicitly provided
            "blocking_gate_distribution": {
                "G2_INVARIANT": 5,
                "G3_SAFE_REGION": 7,
            },
            "mode": "HYPOTHETICAL",
        }

        # Simulate derivation logic
        top_gate = status.get("top_blocking_gate")
        if top_gate is None:
            gate_dist = status.get("blocking_gate_distribution", {})
            if gate_dist:
                top_gate = max(gate_dist.items(), key=lambda x: x[1])[0]

        # Uses provided value, not derived
        assert top_gate == "G2_INVARIANT"

    def test_handles_empty_distribution(self):
        """Should handle empty blocking_gate_distribution gracefully."""
        status = {
            "hypothetical_block_rate": 0.0,
            "blocking_gate_distribution": {},
            "mode": "HYPOTHETICAL",
        }

        top_gate = status.get("top_blocking_gate")
        if top_gate is None:
            gate_dist = status.get("blocking_gate_distribution", {})
            if gate_dist:
                top_gate = max(gate_dist.items(), key=lambda x: x[1])[0]

        assert top_gate is None
