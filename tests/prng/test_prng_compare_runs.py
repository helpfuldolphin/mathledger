"""
Tests for PRNG drift ledger comparison tool.

Covers:
- Deterministic ordering in delta reports
- Ledger comparison logic
- Drift status classification
- JSON serialization
"""

import json
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from rfl.prng.governance import DriftStatus
from scripts.prng_compare_runs import (
    load_ledger,
    compute_drift_delta,
    classify_drift_status,
    build_comparison_report,
    _compute_rules_delta,
)


class TestLedgerComparison:
    """Test ledger comparison logic."""

    def test_compute_drift_delta_basic(self):
        """Basic delta computation between two ledgers."""
        baseline = {
            "schema_version": "1.0.0",
            "total_runs": 10,
            "volatile_runs": 1,
            "drifting_runs": 2,
            "stable_runs": 7,
            "frequent_rules": {"R1": 3, "R2": 1},
        }
        
        comparison = {
            "schema_version": "1.0.0",
            "total_runs": 15,
            "volatile_runs": 2,
            "drifting_runs": 3,
            "stable_runs": 10,
            "frequent_rules": {"R1": 5, "R3": 2},
        }
        
        delta = compute_drift_delta(baseline, comparison)
        
        assert delta["total_runs_delta"] == 5
        assert delta["volatile_runs_delta"] == 1
        assert delta["drifting_runs_delta"] == 1
        assert delta["stable_runs_delta"] == 3
        assert delta["frequent_rules_delta"]["R1"] == 2
        assert delta["frequent_rules_delta"]["R2"] == -1  # R2 went from 1 to 0
        assert delta["frequent_rules_delta"]["R3"] == 2

    def test_compute_rules_delta_deterministic_ordering(self):
        """Rules delta should be deterministically sorted."""
        baseline = {"R3": 1, "R1": 2, "R2": 0}
        comparison = {"R2": 3, "R1": 5, "R4": 1}
        
        delta = _compute_rules_delta(baseline, comparison)
        
        # Should be sorted by rule_id
        rule_ids = list(delta.keys())
        assert rule_ids == sorted(rule_ids)
        
        # Verify values
        assert delta["R1"] == 3  # 5 - 2
        assert delta["R2"] == 3  # 3 - 0
        assert delta["R4"] == 1  # 1 - 0

    def test_compute_rules_delta_filters_zeros(self):
        """Rules delta should only include non-zero differences."""
        baseline = {"R1": 3, "R2": 2}
        comparison = {"R1": 3, "R2": 2}
        
        delta = _compute_rules_delta(baseline, comparison)
        
        # All deltas are zero, so dict should be empty
        assert delta == {}

    def test_classify_drift_status_stable(self):
        """Classify STABLE drift status."""
        runs = {
            "total_runs": 10,
            "volatile_runs": 0,
            "drifting_runs": 0,
            "stable_runs": 10,
        }
        
        status = classify_drift_status(runs)
        assert status == DriftStatus.STABLE.value

    def test_classify_drift_status_drifting(self):
        """Classify DRIFTING drift status."""
        runs = {
            "total_runs": 10,
            "volatile_runs": 0,
            "drifting_runs": 2,
            "stable_runs": 8,
        }
        
        status = classify_drift_status(runs)
        assert status == DriftStatus.DRIFTING.value

    def test_classify_drift_status_volatile(self):
        """Classify VOLATILE drift status (≥30% volatile runs)."""
        runs = {
            "total_runs": 10,
            "volatile_runs": 3,  # 30%
            "drifting_runs": 2,
            "stable_runs": 5,
        }
        
        status = classify_drift_status(runs)
        assert status == DriftStatus.VOLATILE.value

    def test_classify_drift_status_empty(self):
        """Empty runs should classify as STABLE."""
        runs = {
            "total_runs": 0,
            "volatile_runs": 0,
            "drifting_runs": 0,
            "stable_runs": 0,
        }
        
        status = classify_drift_status(runs)
        assert status == DriftStatus.STABLE.value


class TestComparisonReport:
    """Test build_comparison_report."""

    def test_build_report_p3_p4_p5(self):
        """Build report with all three phases."""
        p3_ledger = {
            "schema_version": "1.0.0",
            "total_runs": 10,
            "volatile_runs": 0,
            "drifting_runs": 1,
            "stable_runs": 9,
            "frequent_rules": {"R1": 1},
        }
        
        p4_ledger = {
            "schema_version": "1.0.0",
            "total_runs": 15,
            "volatile_runs": 1,
            "drifting_runs": 2,
            "stable_runs": 12,
            "frequent_rules": {"R1": 2, "R2": 1},
        }
        
        p5_ledger = {
            "schema_version": "1.0.0",
            "total_runs": 20,
            "volatile_runs": 2,
            "drifting_runs": 3,
            "stable_runs": 15,
            "frequent_rules": {"R1": 3, "R2": 2, "R3": 1},
        }
        
        report = build_comparison_report(p3_ledger, p4_ledger, p5_ledger)
        
        # Check structure
        assert report["schema_version"] == "1.0.0"
        assert "p3_mock" in report["phase_ledgers"]
        assert "p4_mock" in report["phase_ledgers"]
        assert "p5_real" in report["phase_ledgers"]
        
        # Check deltas
        assert "p3_to_p4" in report["deltas"]
        assert "p4_to_p5" in report["deltas"]
        assert "p3_to_p5" in report["deltas"]
        
        # Check transitions
        assert len(report["drift_status_transitions"]) == 2
        assert report["drift_status_transitions"][0]["from_phase"] == "p3_mock"
        assert report["drift_status_transitions"][0]["to_phase"] == "p4_mock"
        assert report["drift_status_transitions"][1]["from_phase"] == "p4_mock"
        assert report["drift_status_transitions"][1]["to_phase"] == "p5_real"
        
        # Check summary
        assert len(report["summary"]["phases_analyzed"]) == 3
        assert report["summary"]["current_status"] == DriftStatus.DRIFTING.value

    def test_build_report_deterministic_ordering(self):
        """Report should have deterministic JSON ordering."""
        p3_ledger = {
            "schema_version": "1.0.0",
            "total_runs": 5,
            "volatile_runs": 0,
            "drifting_runs": 0,
            "stable_runs": 5,
            "frequent_rules": {},
        }
        
        p4_ledger = {
            "schema_version": "1.0.0",
            "total_runs": 10,
            "volatile_runs": 1,
            "drifting_runs": 1,
            "stable_runs": 8,
            "frequent_rules": {"R2": 1, "R1": 1},  # Unsorted input
        }
        
        report = build_comparison_report(p3_ledger, p4_ledger, None)
        
        # Serialize and deserialize to check determinism
        json_str1 = json.dumps(report, sort_keys=True)
        json_str2 = json.dumps(report, sort_keys=True)
        
        assert json_str1 == json_str2
        
        # Check that frequent_rules_delta is sorted
        delta = report["deltas"]["p3_to_p4"]
        rule_ids = list(delta["frequent_rules_delta"].keys())
        assert rule_ids == sorted(rule_ids)

    def test_build_report_partial_phases(self):
        """Build report with only P4 and P5."""
        p4_ledger = {
            "schema_version": "1.0.0",
            "total_runs": 10,
            "volatile_runs": 0,
            "drifting_runs": 1,
            "stable_runs": 9,
            "frequent_rules": {},
        }
        
        p5_ledger = {
            "schema_version": "1.0.0",
            "total_runs": 15,
            "volatile_runs": 1,
            "drifting_runs": 2,
            "stable_runs": 12,
            "frequent_rules": {"R1": 1},
        }
        
        report = build_comparison_report(None, p4_ledger, p5_ledger)
        
        assert "p3_mock" not in report["phase_ledgers"]
        assert "p4_mock" in report["phase_ledgers"]
        assert "p5_real" in report["phase_ledgers"]
        
        assert "p3_to_p4" not in report["deltas"]
        assert "p4_to_p5" in report["deltas"]
        assert "p3_to_p5" not in report["deltas"]
        
        assert len(report["drift_status_transitions"]) == 1
        assert report["drift_status_transitions"][0]["from_phase"] == "p4_mock"
        assert report["drift_status_transitions"][0]["to_phase"] == "p5_real"

    def test_build_report_single_phase(self):
        """Build report with only P5."""
        p5_ledger = {
            "schema_version": "1.0.0",
            "total_runs": 20,
            "volatile_runs": 2,
            "drifting_runs": 3,
            "stable_runs": 15,
            "frequent_rules": {"R1": 2},
        }
        
        report = build_comparison_report(None, None, p5_ledger)
        
        assert len(report["phase_ledgers"]) == 1
        assert "p5_real" in report["phase_ledgers"]
        assert len(report["deltas"]) == 0
        assert len(report["drift_status_transitions"]) == 0
        assert report["summary"]["current_status"] == DriftStatus.DRIFTING.value


class TestLedgerIO:
    """Test ledger file I/O."""

    def test_load_ledger_valid(self):
        """Load a valid ledger from JSON file."""
        with TemporaryDirectory() as tmpdir:
            ledger_path = Path(tmpdir) / "ledger.json"
            ledger_data = {
                "schema_version": "1.0.0",
                "total_runs": 10,
                "volatile_runs": 1,
                "drifting_runs": 2,
                "stable_runs": 7,
                "frequent_rules": {"R1": 3},
            }
            
            with open(ledger_path, "w", encoding="utf-8") as f:
                json.dump(ledger_data, f)
            
            loaded = load_ledger(ledger_path)
            assert loaded == ledger_data

    def test_load_ledger_missing_field(self):
        """Loading ledger with missing field should raise ValueError."""
        with TemporaryDirectory() as tmpdir:
            ledger_path = Path(tmpdir) / "ledger.json"
            invalid_data = {
                "schema_version": "1.0.0",
                "total_runs": 10,
                # Missing volatile_runs, etc.
            }
            
            with open(ledger_path, "w", encoding="utf-8") as f:
                json.dump(invalid_data, f)
            
            with pytest.raises(ValueError, match="Missing required field"):
                load_ledger(ledger_path)

