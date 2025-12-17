"""
Tests for chronicle risk register GGFL adapter.

SHADOW MODE: These tests verify that chronicle risk register signals are
correctly converted to GGFL alignment view format.
"""

import unittest
from typing import Any, Dict

from backend.health.chronicle_governance_adapter import (
    chronicle_risk_for_alignment_view,
    summarize_chronicle_risk_signal_consistency,
)


class TestChronicleRiskGGFLAdapter(unittest.TestCase):
    """Tests for chronicle risk register GGFL adapter."""

    def test_chronicle_risk_for_alignment_view_ok_status(self) -> None:
        """Verify GGFL adapter returns 'ok' status when no high-risk calibrations."""
        signal = {
            "total_calibrations": 5,
            "high_risk_count": 0,
            "high_risk_cal_ids_top3": [],
            "has_any_invariants_violated": False,
            "extraction_source": "MANIFEST",
        }

        view = chronicle_risk_for_alignment_view(signal)

        assert view["signal_type"] == "SIG-CHR"
        assert view["status"] == "ok"
        assert view["conflict"] is False
        assert view["weight_hint"] == "LOW"
        assert view["extraction_source"] == "MANIFEST"
        assert isinstance(view["drivers"], list)
        assert isinstance(view["summary"], str)

    def test_chronicle_risk_for_alignment_view_warn_status(self) -> None:
        """Verify GGFL adapter returns 'warn' status when high-risk calibrations present."""
        signal = {
            "total_calibrations": 5,
            "high_risk_count": 2,
            "high_risk_cal_ids_top3": ["cal_001", "cal_002"],
            "has_any_invariants_violated": True,
            "extraction_source": "EVIDENCE_JSON",
        }

        view = chronicle_risk_for_alignment_view(signal)

        assert view["signal_type"] == "SIG-CHR"
        assert view["status"] == "warn"
        assert view["conflict"] is False
        assert view["weight_hint"] == "LOW"
        assert view["extraction_source"] == "EVIDENCE_JSON"
        assert len(view["drivers"]) <= 3

    def test_chronicle_risk_for_alignment_view_drivers(self) -> None:
        """Verify drivers list is deterministic and includes expected information."""
        signal = {
            "total_calibrations": 5,
            "high_risk_count": 2,
            "high_risk_cal_ids_top3": ["cal_001", "cal_002"],
            "has_any_invariants_violated": True,
            "extraction_source": "MANIFEST",
        }

        view = chronicle_risk_for_alignment_view(signal)

        drivers = view["drivers"]
        assert len(drivers) <= 3
        assert any("high_risk_count" in d for d in drivers)
        assert any("HIGH recurrence + invariants violated" in d for d in drivers)
        assert any("top_risk_cal_ids" in d for d in drivers)

    def test_chronicle_risk_for_alignment_view_summary(self) -> None:
        """Verify summary is neutral and descriptive."""
        signal = {
            "total_calibrations": 3,
            "high_risk_count": 1,
            "high_risk_cal_ids_top3": ["cal_001"],
            "has_any_invariants_violated": True,
            "extraction_source": "MANIFEST",
        }

        view = chronicle_risk_for_alignment_view(signal)

        summary = view["summary"]
        assert isinstance(summary, str)
        assert "Chronicle risk register" in summary
        assert "1" in summary  # high_risk_count
        assert "3" in summary  # total_calibrations
        assert "invariants violated" in summary

    def test_chronicle_risk_for_alignment_view_no_calibrations(self) -> None:
        """Verify GGFL adapter handles zero calibrations gracefully."""
        signal = {
            "total_calibrations": 0,
            "high_risk_count": 0,
            "high_risk_cal_ids_top3": [],
            "has_any_invariants_violated": False,
            "extraction_source": "MISSING",
        }

        view = chronicle_risk_for_alignment_view(signal)

        assert view["status"] == "ok"
        assert "No calibration experiments" in view["summary"]
        assert view["extraction_source"] == "MISSING"

    def test_chronicle_risk_for_alignment_view_deterministic(self) -> None:
        """Verify GGFL adapter output is deterministic."""
        signal = {
            "total_calibrations": 5,
            "high_risk_count": 3,
            "high_risk_cal_ids_top3": ["cal_a", "cal_b", "cal_c"],
            "has_any_invariants_violated": True,
            "extraction_source": "MANIFEST",
        }

        view1 = chronicle_risk_for_alignment_view(signal)
        view2 = chronicle_risk_for_alignment_view(signal)

        assert view1 == view2
        assert view1["drivers"] == view2["drivers"]
        assert view1["summary"] == view2["summary"]

    def test_chronicle_risk_for_alignment_view_extraction_source(self) -> None:
        """Verify extraction_source is passed through correctly."""
        for source in ["MANIFEST", "EVIDENCE_JSON", "MISSING"]:
            signal = {
                "total_calibrations": 2,
                "high_risk_count": 0,
                "high_risk_cal_ids_top3": [],
                "has_any_invariants_violated": False,
                "extraction_source": source,
            }

            view = chronicle_risk_for_alignment_view(signal)

            assert view["extraction_source"] == source

    def test_chronicle_risk_for_alignment_view_warn_on_invariants_only(self) -> None:
        """Verify status is 'warn' when invariants violated even if high_risk_count is 0."""
        signal = {
            "total_calibrations": 2,
            "high_risk_count": 0,
            "high_risk_cal_ids_top3": [],
            "has_any_invariants_violated": True,  # Invariants violated but no high-risk
            "extraction_source": "MANIFEST",
        }

        view = chronicle_risk_for_alignment_view(signal)

        assert view["status"] == "warn"
        assert "invariants violated" in view["summary"]

    def test_chronicle_risk_for_alignment_view_drivers_limit(self) -> None:
        """Verify drivers list is limited to 3 items."""
        signal = {
            "total_calibrations": 10,
            "high_risk_count": 5,
            "high_risk_cal_ids_top3": ["cal_001", "cal_002", "cal_003"],
            "has_any_invariants_violated": True,
            "extraction_source": "MANIFEST",
        }

        view = chronicle_risk_for_alignment_view(signal)

        assert len(view["drivers"]) <= 3

    def test_chronicle_risk_for_alignment_view_driver_prefixes(self) -> None:
        """Verify drivers use deterministic prefixes CHR-DRV-001, CHR-DRV-002, CHR-DRV-003."""
        signal = {
            "total_calibrations": 10,
            "high_risk_count": 5,
            "high_risk_cal_ids_top3": ["cal_001", "cal_002", "cal_003"],
            "has_any_invariants_violated": True,
            "extraction_source": "MANIFEST",
        }

        view = chronicle_risk_for_alignment_view(signal)

        drivers = view["drivers"]
        assert len(drivers) <= 3
        
        # Check that all drivers use the correct prefix format
        for i, driver in enumerate(drivers):
            expected_prefix = f"CHR-DRV-{i+1:03d}"
            assert driver.startswith(expected_prefix), f"Driver {i+1} should start with {expected_prefix}, got {driver}"

    def test_chronicle_risk_for_alignment_view_status_enum(self) -> None:
        """Verify status is always 'ok' or 'warn' (frozen enum)."""
        for high_risk_count in [0, 1, 5]:
            for has_invariants in [False, True]:
                signal = {
                    "total_calibrations": 10,
                    "high_risk_count": high_risk_count,
                    "high_risk_cal_ids_top3": [],
                    "has_any_invariants_violated": has_invariants,
                    "extraction_source": "MANIFEST",
                }

                view = chronicle_risk_for_alignment_view(signal)

                assert view["status"] in ("ok", "warn"), f"Status must be 'ok' or 'warn', got {view['status']}"

    def test_chronicle_risk_for_alignment_view_extraction_source_enum(self) -> None:
        """Verify extraction_source is always MANIFEST, EVIDENCE_JSON, or MISSING (frozen enum)."""
        for source in ["MANIFEST", "EVIDENCE_JSON", "MISSING", "invalid", None]:
            signal = {
                "total_calibrations": 2,
                "high_risk_count": 0,
                "high_risk_cal_ids_top3": [],
                "has_any_invariants_violated": False,
                "extraction_source": source,
            }

            view = chronicle_risk_for_alignment_view(signal)

            assert view["extraction_source"] in ("MANIFEST", "EVIDENCE_JSON", "MISSING"), \
                f"extraction_source must be MANIFEST, EVIDENCE_JSON, or MISSING, got {view['extraction_source']}"

    def test_chronicle_risk_for_alignment_view_conflict_invariant(self) -> None:
        """Verify conflict is always False (invariant)."""
        signal = {
            "total_calibrations": 10,
            "high_risk_count": 5,
            "high_risk_cal_ids_top3": ["cal_001"],
            "has_any_invariants_violated": True,
            "extraction_source": "MANIFEST",
        }

        view = chronicle_risk_for_alignment_view(signal)

        assert view["conflict"] is False, "Conflict must always be False (invariant)"

    def test_summarize_chronicle_risk_signal_consistency_consistent(self) -> None:
        """Consistency check should return CONSISTENT when signals match."""
        status_signal = {
            "total_calibrations": 5,
            "high_risk_count": 0,
            "high_risk_cal_ids_top3": [],
            "has_any_invariants_violated": False,
            "extraction_source": "MANIFEST",
        }
        ggfl_signal = {
            "signal_type": "SIG-CHR",
            "status": "ok",
            "conflict": False,
            "weight_hint": "LOW",
            "drivers": [],
            "summary": "Chronicle risk register: 5 calibration experiment(s) show low to moderate recurrence risk patterns.",
            "extraction_source": "MANIFEST",
        }

        result = summarize_chronicle_risk_signal_consistency(status_signal, ggfl_signal)

        assert result["consistency"] == "CONSISTENT"
        assert result["conflict_invariant_violated"] is False
        assert any("consistent" in note.lower() for note in result["notes"])

    def test_summarize_chronicle_risk_signal_consistency_status_mismatch(self) -> None:
        """Consistency check should detect status mismatch."""
        status_signal = {
            "total_calibrations": 5,
            "high_risk_count": 2,
            "high_risk_cal_ids_top3": ["cal_001", "cal_002"],
            "has_any_invariants_violated": True,
            "extraction_source": "MANIFEST",
        }
        ggfl_signal = {
            "signal_type": "SIG-CHR",
            "status": "ok",  # Mismatch: status signal should derive "warn" but GGFL says "ok"
            "conflict": False,
            "weight_hint": "LOW",
            "drivers": ["CHR-DRV-001: high_risk_count=2"],
            "summary": "Chronicle risk register: 2 out of 5 calibration experiment(s) show HIGH recurrence likelihood with invariants violated.",
            "extraction_source": "MANIFEST",
        }

        result = summarize_chronicle_risk_signal_consistency(status_signal, ggfl_signal)

        assert result["consistency"] == "PARTIAL"
        assert result["conflict_invariant_violated"] is False
        assert any("status mismatch" in note.lower() for note in result["notes"])
        assert result["top_mismatch_type"] == "status_mismatch"

    def test_summarize_chronicle_risk_signal_consistency_conflict_invariant_violated(self) -> None:
        """Consistency check should detect conflict invariant violation."""
        status_signal = {
            "total_calibrations": 5,
            "high_risk_count": 2,
            "high_risk_cal_ids_top3": ["cal_001"],
            "has_any_invariants_violated": True,
            "extraction_source": "MANIFEST",
        }
        ggfl_signal = {
            "signal_type": "SIG-CHR",
            "status": "warn",
            "conflict": True,  # VIOLATION: conflict must always be False
            "weight_hint": "LOW",
            "drivers": ["CHR-DRV-001: high_risk_count=2"],
            "summary": "Chronicle risk register: 2 out of 5 calibration experiment(s) show HIGH recurrence likelihood with invariants violated.",
            "extraction_source": "MANIFEST",
        }

        result = summarize_chronicle_risk_signal_consistency(status_signal, ggfl_signal)

        assert result["consistency"] == "INCONSISTENT"
        assert result["conflict_invariant_violated"] is True
        assert any("conflict invariant violated" in note.lower() for note in result["notes"])
        assert result["top_mismatch_type"] == "conflict_invariant_violated"

    def test_summarize_chronicle_risk_signal_consistency_driver_count_violation(self) -> None:
        """Consistency check should detect driver count violation (> 3)."""
        status_signal = {
            "total_calibrations": 5,
            "high_risk_count": 2,
            "high_risk_cal_ids_top3": ["cal_001"],
            "has_any_invariants_violated": True,
            "extraction_source": "MANIFEST",
        }
        ggfl_signal = {
            "signal_type": "SIG-CHR",
            "status": "warn",
            "conflict": False,
            "weight_hint": "LOW",
            "drivers": ["CHR-DRV-001", "CHR-DRV-002", "CHR-DRV-003", "CHR-DRV-004"],  # VIOLATION: > 3 drivers
            "summary": "Chronicle risk register: 2 out of 5 calibration experiment(s) show HIGH recurrence likelihood with invariants violated.",
            "extraction_source": "MANIFEST",
        }

        result = summarize_chronicle_risk_signal_consistency(status_signal, ggfl_signal)

        assert result["consistency"] == "PARTIAL"
        assert result["conflict_invariant_violated"] is False
        assert any("driver count violation" in note.lower() for note in result["notes"])
        assert result["top_mismatch_type"] == "driver_count_violation"

    def test_summarize_chronicle_risk_signal_consistency_driver_prefix_violation(self) -> None:
        """Consistency check should detect driver prefix violation."""
        status_signal = {
            "total_calibrations": 5,
            "high_risk_count": 2,
            "high_risk_cal_ids_top3": ["cal_001"],
            "has_any_invariants_violated": True,
            "extraction_source": "MANIFEST",
        }
        ggfl_signal = {
            "signal_type": "SIG-CHR",
            "status": "warn",
            "conflict": False,
            "weight_hint": "LOW",
            "drivers": ["INVALID-PREFIX: high_risk_count=2"],  # VIOLATION: wrong prefix
            "summary": "Chronicle risk register: 2 out of 5 calibration experiment(s) show HIGH recurrence likelihood with invariants violated.",
            "extraction_source": "MANIFEST",
        }

        result = summarize_chronicle_risk_signal_consistency(status_signal, ggfl_signal)

        assert result["consistency"] == "PARTIAL"
        assert result["conflict_invariant_violated"] is False
        assert any("driver prefix violation" in note.lower() for note in result["notes"])
        assert result["top_mismatch_type"] == "driver_prefix_violation"

