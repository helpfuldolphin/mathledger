"""
Tests for What-If Manifest Binding Consistency.

Tests:
- Manifest-first: prefers manifest.signals.what_if over full report
- Fallback chain: signals.what_if → governance.what_if_analysis.status → report
- Determinism: same input produces same output
- Warning hygiene: single-line format with top_blocking_gate
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict

import pytest

from backend.governance.evidence_pack import (
    get_what_if_status_from_manifest,
    format_what_if_warning,
    bind_what_if_to_manifest,
    _compute_report_hash,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def compact_status() -> Dict[str, Any]:
    """Compact status signal for manifest.signals.what_if."""
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
        "report_sha256": "abc123",
    }


@pytest.fixture
def full_report() -> Dict[str, Any]:
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
def manifest_with_signal_only(compact_status) -> Dict[str, Any]:
    """Manifest with only signals.what_if populated."""
    return {
        "signals": {
            "what_if": compact_status,
        },
    }


@pytest.fixture
def manifest_with_status_only(compact_status) -> Dict[str, Any]:
    """Manifest with only governance.what_if_analysis.status populated."""
    return {
        "governance": {
            "what_if_analysis": {
                "status": compact_status,
            },
        },
    }


@pytest.fixture
def manifest_with_report_only(full_report) -> Dict[str, Any]:
    """Manifest with only governance.what_if_analysis.report populated."""
    return {
        "governance": {
            "what_if_analysis": {
                "report": full_report,
            },
        },
    }


@pytest.fixture
def manifest_with_all(compact_status, full_report) -> Dict[str, Any]:
    """Manifest with all three populated (signal, status, report)."""
    return {
        "signals": {
            "what_if": compact_status,
        },
        "governance": {
            "what_if_analysis": {
                "status": compact_status,
                "report": full_report,
            },
        },
    }


# =============================================================================
# MANIFEST-FIRST TESTS
# =============================================================================

class TestManifestFirst:
    """Tests that manifest.signals.what_if is preferred."""

    def test_prefers_signals_what_if_over_status(self, compact_status):
        """Should prefer signals.what_if over governance.what_if_analysis.status."""
        signal_status = dict(compact_status)
        signal_status["hypothetical_block_rate"] = 0.20  # Different value

        other_status = dict(compact_status)
        other_status["hypothetical_block_rate"] = 0.10  # Different value

        manifest = {
            "signals": {"what_if": signal_status},
            "governance": {
                "what_if_analysis": {
                    "status": other_status,
                },
            },
        }

        status, warnings = get_what_if_status_from_manifest(manifest)

        assert status is not None
        assert status["hypothetical_block_rate"] == 0.20  # From signals.what_if
        assert len(warnings) == 0

    def test_prefers_signals_what_if_over_report(self, compact_status, full_report):
        """Should prefer signals.what_if over extracting from report."""
        signal_status = dict(compact_status)
        signal_status["hypothetical_block_rate"] = 0.25

        manifest = {
            "signals": {"what_if": signal_status},
            "governance": {
                "what_if_analysis": {
                    "report": full_report,
                },
            },
        }

        status, warnings = get_what_if_status_from_manifest(manifest)

        assert status is not None
        assert status["hypothetical_block_rate"] == 0.25  # From signals.what_if
        assert len(warnings) == 0

    def test_empty_signal_falls_back(self, compact_status):
        """Should fallback if signals.what_if is empty dict."""
        manifest = {
            "signals": {"what_if": {}},  # Empty
            "governance": {
                "what_if_analysis": {
                    "status": compact_status,
                },
            },
        }

        status, warnings = get_what_if_status_from_manifest(manifest)

        assert status is not None
        assert status["hypothetical_block_rate"] == 0.15  # From fallback status


# =============================================================================
# FALLBACK CHAIN TESTS
# =============================================================================

class TestFallbackChain:
    """Tests the fallback resolution chain."""

    def test_fallback_to_status_when_no_signal(self, manifest_with_status_only):
        """Should fallback to governance.what_if_analysis.status."""
        status, warnings = get_what_if_status_from_manifest(manifest_with_status_only)

        assert status is not None
        assert status["hypothetical_block_rate"] == 0.15
        assert len(warnings) == 0

    def test_fallback_to_report_extraction(self, manifest_with_report_only):
        """Should extract status from report as last resort."""
        status, warnings = get_what_if_status_from_manifest(manifest_with_report_only)

        assert status is not None
        assert status["hypothetical_block_rate"] == 0.15
        # No warnings for successful extraction

    def test_no_what_if_returns_none_with_warning(self):
        """Should return None with warning when nothing found."""
        manifest = {"other_data": "value"}

        status, warnings = get_what_if_status_from_manifest(manifest)

        assert status is None
        assert len(warnings) == 1
        assert "No What-If status found" in warnings[0]

    def test_empty_manifest_returns_none_with_warning(self):
        """Should return None with warning for empty manifest."""
        status, warnings = get_what_if_status_from_manifest({})

        assert status is None
        assert len(warnings) == 1

    def test_full_fallback_chain_signal_first(self, manifest_with_all):
        """Should use signal even when all three are present."""
        status, warnings = get_what_if_status_from_manifest(manifest_with_all)

        assert status is not None
        # Should match signals.what_if
        assert status["hypothetical_block_rate"] == 0.15


# =============================================================================
# WARNING HYGIENE TESTS
# =============================================================================

class TestWarningHygiene:
    """Tests for single-line warning format."""

    def test_no_warning_when_no_blocks(self):
        """Should return None when hypothetical_block_rate is 0."""
        status = {
            "hypothetical_block_rate": 0.0,
            "mode": "HYPOTHETICAL",
        }

        warning = format_what_if_warning(status)
        assert warning is None

    def test_single_line_warning_with_blocks(self, compact_status):
        """Should return single-line warning when blocks present."""
        warning = format_what_if_warning(compact_status)

        assert warning is not None
        assert "\n" not in warning  # Single line
        assert "15.0%" in warning
        assert "HYPOTHETICAL" in warning

    def test_warning_includes_top_blocking_gate(self, compact_status):
        """Warning should include top_blocking_gate."""
        warning = format_what_if_warning(compact_status)

        assert warning is not None
        assert "G3_SAFE_REGION" in warning  # 7 blocks is highest

    def test_warning_format_structure(self, compact_status):
        """Warning should follow expected format."""
        warning = format_what_if_warning(compact_status)

        # Expected: "What-If (HYPOTHETICAL): 15.0% hypothetical block rate; top_gate=G3_SAFE_REGION"
        assert warning.startswith("What-If (HYPOTHETICAL):")
        assert "hypothetical block rate" in warning
        assert "top_gate=" in warning

    def test_warning_without_top_gate(self):
        """Warning should work without top_blocking_gate."""
        status = {
            "hypothetical_block_rate": 0.1,
            "mode": "HYPOTHETICAL",
            # No top_blocking_gate
        }

        warning = format_what_if_warning(status)

        assert warning is not None
        assert "10.0%" in warning
        assert "top_gate" not in warning


# =============================================================================
# DETERMINISM TESTS
# =============================================================================

class TestDeterminism:
    """Tests for deterministic output."""

    def test_same_input_produces_same_status(self, compact_status):
        """Same manifest should produce same status."""
        manifest = {"signals": {"what_if": compact_status}}

        results = []
        for _ in range(3):
            status, _ = get_what_if_status_from_manifest(manifest)
            results.append(json.dumps(status, sort_keys=True))

        assert results[0] == results[1] == results[2]

    def test_same_report_produces_same_hash(self, full_report):
        """Same report should produce same hash."""
        hashes = []
        for _ in range(3):
            h = _compute_report_hash(full_report)
            hashes.append(h)

        assert hashes[0] == hashes[1] == hashes[2]

    def test_binding_produces_consistent_output(self, full_report):
        """Binding same report should produce consistent manifest structure."""
        manifest1 = {}
        manifest2 = {}

        bind_what_if_to_manifest(manifest1, report_dict=full_report)
        bind_what_if_to_manifest(manifest2, report_dict=full_report)

        # Compare status (excluding timestamps)
        status1 = manifest1["signals"]["what_if"]
        status2 = manifest2["signals"]["what_if"]

        assert status1["hypothetical_block_rate"] == status2["hypothetical_block_rate"]
        assert status1["blocking_gate_distribution"] == status2["blocking_gate_distribution"]
        assert status1["mode"] == status2["mode"]


# =============================================================================
# BIND TO MANIFEST TESTS
# =============================================================================

class TestBindToManifest:
    """Tests for bind_what_if_to_manifest()."""

    def test_binds_report_and_status(self, full_report):
        """Should bind both report and extracted status."""
        manifest = {}
        manifest, warnings = bind_what_if_to_manifest(manifest, report_dict=full_report)

        assert "governance" in manifest
        assert "what_if_analysis" in manifest["governance"]
        assert "report" in manifest["governance"]["what_if_analysis"]
        assert "status" in manifest["governance"]["what_if_analysis"]
        assert "signals" in manifest
        assert "what_if" in manifest["signals"]

    def test_signals_mirrors_status(self, full_report):
        """signals.what_if should mirror governance.what_if_analysis.status."""
        manifest = {}
        manifest, _ = bind_what_if_to_manifest(manifest, report_dict=full_report)

        status = manifest["governance"]["what_if_analysis"]["status"]
        signal = manifest["signals"]["what_if"]

        assert status == signal

    def test_report_hash_computed(self, full_report):
        """Should compute and store report hash."""
        manifest = {}
        manifest, _ = bind_what_if_to_manifest(manifest, report_dict=full_report)

        assert "report_sha256" in manifest["governance"]["what_if_analysis"]
        assert len(manifest["governance"]["what_if_analysis"]["report_sha256"]) == 64

    def test_mode_enforced_hypothetical(self):
        """Should enforce mode=HYPOTHETICAL in status."""
        wrong_mode_status = {
            "hypothetical_block_rate": 0.1,
            "mode": "SHADOW",  # Wrong
        }

        manifest = {}
        manifest, warnings = bind_what_if_to_manifest(
            manifest, status_dict=wrong_mode_status
        )

        assert manifest["signals"]["what_if"]["mode"] == "HYPOTHETICAL"
        assert any("changed to 'HYPOTHETICAL'" in w for w in warnings)

    def test_preserves_existing_manifest_data(self, full_report):
        """Should preserve existing manifest data."""
        manifest = {
            "proof_hash": "abc123",
            "signals": {
                "usla": {"rho": 0.85},
            },
        }

        manifest, _ = bind_what_if_to_manifest(manifest, report_dict=full_report)

        assert manifest["proof_hash"] == "abc123"
        assert manifest["signals"]["usla"]["rho"] == 0.85
        assert "what_if" in manifest["signals"]

    def test_no_input_returns_warning(self):
        """Should return warning when no report or status provided."""
        manifest = {}
        manifest, warnings = bind_what_if_to_manifest(manifest)

        assert len(warnings) == 1
        assert "No What-If report or status provided" in warnings[0]


# =============================================================================
# MODE VALIDATION TESTS
# =============================================================================

class TestModeValidation:
    """Tests for mode validation (must be HYPOTHETICAL)."""

    def test_warns_on_wrong_mode_in_signal(self):
        """Should warn when signals.what_if has wrong mode."""
        manifest = {
            "signals": {
                "what_if": {
                    "hypothetical_block_rate": 0.1,
                    "mode": "SHADOW",
                },
            },
        }

        status, warnings = get_what_if_status_from_manifest(manifest)

        assert status is not None
        assert len(warnings) == 1
        assert "expected 'HYPOTHETICAL'" in warnings[0]

    def test_warns_on_wrong_mode_in_status(self):
        """Should warn when governance.what_if_analysis.status has wrong mode."""
        manifest = {
            "governance": {
                "what_if_analysis": {
                    "status": {
                        "hypothetical_block_rate": 0.1,
                        "mode": "ACTIVE",
                    },
                },
            },
        }

        status, warnings = get_what_if_status_from_manifest(manifest)

        assert status is not None
        assert len(warnings) == 1
        assert "expected 'HYPOTHETICAL'" in warnings[0]

    def test_no_warning_for_correct_mode(self, compact_status):
        """Should not warn when mode is HYPOTHETICAL."""
        manifest = {"signals": {"what_if": compact_status}}

        status, warnings = get_what_if_status_from_manifest(manifest)

        assert status is not None
        assert len(warnings) == 0
