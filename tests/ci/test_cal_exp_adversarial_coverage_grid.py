# tests/ci/test_cal_exp_adversarial_coverage_grid.py
"""
CI test for Cal-Exp adversarial coverage grid.

Verifies:
- Per-experiment snapshot emission (shape, determinism, JSON round-trip)
- Coverage grid aggregation (counts, missing failover detection)
- Evidence attachment (non-mutating, JSON-safe)
"""

import json
import pytest
import tempfile
from pathlib import Path
from typing import Dict, Any, List

from backend.health.adversarial_pressure_adapter import (
    emit_cal_exp_adversarial_coverage,
    build_adversarial_coverage_grid,
    attach_adversarial_coverage_grid_to_evidence,
    build_first_light_adversarial_coverage_annex,
    build_adversarial_priority_scenario_ledger,
    attach_priority_scenario_ledger_to_evidence,
    extract_adversarial_coverage_signal_for_status,
    adversarial_coverage_for_alignment_view,
    DRIVER_MISSING_FAILOVER_COUNT,
    DRIVER_REPEATED_PRIORITY_SCENARIOS,
    ADVERSARIAL_COVERAGE_ANNEX_SCHEMA_VERSION,
    ADVERSARIAL_COVERAGE_GRID_SCHEMA_VERSION,
    ADVERSARIAL_PRIORITY_SCENARIO_LEDGER_SCHEMA_VERSION,
)


@pytest.fixture
def sample_annex() -> Dict[str, Any]:
    """Sample coverage annex for testing."""
    return {
        "schema_version": ADVERSARIAL_COVERAGE_ANNEX_SCHEMA_VERSION,
        "p3_pressure_band": "LOW",
        "p4_pressure_band": "MEDIUM",
        "priority_scenarios": ["scenario1", "scenario2"],
        "has_failover": True,
    }


@pytest.fixture
def sample_snapshots() -> List[Dict[str, Any]]:
    """Sample coverage snapshots for testing."""
    return [
        {
            "schema_version": ADVERSARIAL_COVERAGE_ANNEX_SCHEMA_VERSION,
            "cal_id": "CAL-EXP-1",
            "p3_pressure_band": "LOW",
            "p4_pressure_band": "LOW",
            "priority_scenarios": ["scenario1"],
            "has_failover": True,
        },
        {
            "schema_version": ADVERSARIAL_COVERAGE_ANNEX_SCHEMA_VERSION,
            "cal_id": "CAL-EXP-2",
            "p3_pressure_band": "MEDIUM",
            "p4_pressure_band": "HIGH",
            "priority_scenarios": ["scenario2", "scenario3"],
            "has_failover": False,
        },
        {
            "schema_version": ADVERSARIAL_COVERAGE_ANNEX_SCHEMA_VERSION,
            "cal_id": "CAL-EXP-3",
            "p3_pressure_band": "HIGH",
            "p4_pressure_band": "HIGH",
            "priority_scenarios": ["scenario4"],
            "has_failover": False,
        },
    ]


@pytest.mark.hermetic
class TestCalExpAdversarialCoverageSnapshot:
    """Tests for emit_cal_exp_adversarial_coverage()."""

    def test_snapshot_has_required_fields(self, sample_annex):
        """Snapshot has all required fields."""
        snapshot = emit_cal_exp_adversarial_coverage("CAL-EXP-1", sample_annex)
        
        required_fields = [
            "schema_version",
            "cal_id",
            "p3_pressure_band",
            "p4_pressure_band",
            "priority_scenarios",
            "has_failover",
        ]
        
        for field in required_fields:
            assert field in snapshot, f"Missing required field: {field}"

    def test_snapshot_cal_id_matches(self, sample_annex):
        """Snapshot cal_id matches input."""
        snapshot = emit_cal_exp_adversarial_coverage("CAL-EXP-1", sample_annex)
        
        assert snapshot["cal_id"] == "CAL-EXP-1"

    def test_snapshot_priority_scenarios_limited_to_5(self, sample_annex):
        """Snapshot priority_scenarios limited to 5."""
        # Create annex with 10 scenarios
        large_annex = dict(sample_annex)
        large_annex["priority_scenarios"] = [f"scenario{i}" for i in range(10)]
        
        snapshot = emit_cal_exp_adversarial_coverage("CAL-EXP-1", large_annex)
        
        assert len(snapshot["priority_scenarios"]) <= 5

    def test_snapshot_is_deterministic(self, sample_annex):
        """Snapshot is deterministic."""
        s1 = emit_cal_exp_adversarial_coverage("CAL-EXP-1", sample_annex)
        s2 = emit_cal_exp_adversarial_coverage("CAL-EXP-1", sample_annex)
        
        json1 = json.dumps(s1, sort_keys=True)
        json2 = json.dumps(s2, sort_keys=True)
        assert json1 == json2

    def test_snapshot_is_json_serializable(self, sample_annex):
        """Snapshot is JSON-serializable."""
        snapshot = emit_cal_exp_adversarial_coverage("CAL-EXP-1", sample_annex)
        
        # Should not raise
        json_str = json.dumps(snapshot, sort_keys=True)
        assert isinstance(json_str, str)
        
        # Should be able to parse back
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert parsed["cal_id"] == "CAL-EXP-1"

    def test_snapshot_persists_to_file(self, sample_annex):
        """Snapshot persists to file when output_dir provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "calibration"
            snapshot = emit_cal_exp_adversarial_coverage(
                "CAL-EXP-1", sample_annex, output_dir=output_dir
            )
            
            # File should exist
            output_path = output_dir / "adversarial_coverage_CAL-EXP-1.json"
            assert output_path.exists()
            
            # File should be valid JSON
            with open(output_path, "r") as f:
                file_data = json.load(f)
            
            assert file_data["cal_id"] == "CAL-EXP-1"
            assert file_data == snapshot

    def test_snapshot_json_round_trip(self, sample_annex):
        """Snapshot survives JSON round-trip."""
        snapshot = emit_cal_exp_adversarial_coverage("CAL-EXP-1", sample_annex)
        
        # Round-trip through JSON
        json_str = json.dumps(snapshot, sort_keys=True)
        parsed = json.loads(json_str)
        
        # Should match original
        assert parsed["cal_id"] == snapshot["cal_id"]
        assert parsed["p3_pressure_band"] == snapshot["p3_pressure_band"]
        assert parsed["p4_pressure_band"] == snapshot["p4_pressure_band"]
        assert parsed["has_failover"] == snapshot["has_failover"]


@pytest.mark.hermetic
class TestAdversarialCoverageGrid:
    """Tests for build_adversarial_coverage_grid()."""

    def test_grid_has_required_fields(self, sample_snapshots):
        """Grid has all required fields."""
        grid = build_adversarial_coverage_grid(sample_snapshots)
        
        required_fields = [
            "schema_version",
            "total_experiments",
            "pressure_band_counts",
            "experiments_missing_failover",
            "experiments_by_pressure_band",
        ]
        
        for field in required_fields:
            assert field in grid, f"Missing required field: {field}"

    def test_grid_total_experiments_correct(self, sample_snapshots):
        """Grid total_experiments matches snapshot count."""
        grid = build_adversarial_coverage_grid(sample_snapshots)
        
        assert grid["total_experiments"] == len(sample_snapshots)

    def test_grid_pressure_band_counts_correct(self, sample_snapshots):
        """Grid pressure_band_counts are correct."""
        grid = build_adversarial_coverage_grid(sample_snapshots)
        
        # CAL-EXP-1: (LOW,LOW)
        # CAL-EXP-2: (MEDIUM,HIGH)
        # CAL-EXP-3: (HIGH,HIGH)
        
        assert grid["pressure_band_counts"]["(LOW,LOW)"] == 1
        assert grid["pressure_band_counts"]["(MEDIUM,HIGH)"] == 1
        assert grid["pressure_band_counts"]["(HIGH,HIGH)"] == 1

    def test_grid_detects_missing_failover(self, sample_snapshots):
        """Grid correctly identifies experiments missing failover."""
        grid = build_adversarial_coverage_grid(sample_snapshots)
        
        # CAL-EXP-2 and CAL-EXP-3 have has_failover == False
        assert "CAL-EXP-2" in grid["experiments_missing_failover"]
        assert "CAL-EXP-3" in grid["experiments_missing_failover"]
        assert "CAL-EXP-1" not in grid["experiments_missing_failover"]

    def test_grid_groups_experiments_by_pressure_band(self, sample_snapshots):
        """Grid groups experiments by pressure band combination."""
        grid = build_adversarial_coverage_grid(sample_snapshots)
        
        # Check grouping
        assert "CAL-EXP-1" in grid["experiments_by_pressure_band"]["(LOW,LOW)"]
        assert "CAL-EXP-2" in grid["experiments_by_pressure_band"]["(MEDIUM,HIGH)"]
        assert "CAL-EXP-3" in grid["experiments_by_pressure_band"]["(HIGH,HIGH)"]

    def test_grid_is_deterministic(self, sample_snapshots):
        """Grid is deterministic."""
        g1 = build_adversarial_coverage_grid(sample_snapshots)
        g2 = build_adversarial_coverage_grid(sample_snapshots)
        
        json1 = json.dumps(g1, sort_keys=True)
        json2 = json.dumps(g2, sort_keys=True)
        assert json1 == json2

    def test_grid_is_json_serializable(self, sample_snapshots):
        """Grid is JSON-serializable."""
        grid = build_adversarial_coverage_grid(sample_snapshots)
        
        # Should not raise
        json_str = json.dumps(grid, sort_keys=True)
        assert isinstance(json_str, str)
        
        # Should be able to parse back
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert parsed["total_experiments"] == len(sample_snapshots)

    def test_grid_handles_empty_snapshots(self):
        """Grid handles empty snapshot list."""
        grid = build_adversarial_coverage_grid([])
        
        assert grid["total_experiments"] == 0
        assert len(grid["pressure_band_counts"]) == 0
        assert len(grid["experiments_missing_failover"]) == 0


@pytest.mark.hermetic
class TestCoverageGridEvidenceAttachment:
    """Tests for attach_adversarial_coverage_grid_to_evidence()."""

    def test_evidence_attach_non_mutating(self, sample_snapshots):
        """Evidence attach does not mutate input."""
        evidence = {"timestamp": "2024-01-01", "data": {"test": "value"}}
        grid = build_adversarial_coverage_grid(sample_snapshots)
        
        original_evidence = evidence.copy()
        enriched = attach_adversarial_coverage_grid_to_evidence(evidence, grid)
        
        # Original should be unchanged
        assert evidence == original_evidence
        # Enriched should have governance key
        assert "governance" in enriched
        assert "adversarial_coverage_panel" in enriched["governance"]

    def test_evidence_attach_has_grid(self, sample_snapshots):
        """Attached evidence has coverage grid."""
        evidence = {"timestamp": "2024-01-01"}
        grid = build_adversarial_coverage_grid(sample_snapshots)
        
        enriched = attach_adversarial_coverage_grid_to_evidence(evidence, grid)
        
        attached_grid = enriched["governance"]["adversarial_coverage_panel"]
        assert attached_grid["total_experiments"] == len(sample_snapshots)
        assert "pressure_band_counts" in attached_grid

    def test_evidence_attach_is_json_serializable(self, sample_snapshots):
        """Attached evidence is JSON-serializable."""
        evidence = {"timestamp": "2024-01-01"}
        grid = build_adversarial_coverage_grid(sample_snapshots)
        
        enriched = attach_adversarial_coverage_grid_to_evidence(evidence, grid)
        
        # Should not raise
        json_str = json.dumps(enriched, sort_keys=True)
        assert isinstance(json_str, str)
        
        # Should be able to parse back
        parsed = json.loads(json_str)
        assert "governance" in parsed
        assert "adversarial_coverage_panel" in parsed["governance"]

    def test_evidence_attach_is_deterministic(self, sample_snapshots):
        """Evidence attach is deterministic."""
        evidence = {"timestamp": "2024-01-01"}
        grid = build_adversarial_coverage_grid(sample_snapshots)
        
        e1 = attach_adversarial_coverage_grid_to_evidence(evidence, grid)
        e2 = attach_adversarial_coverage_grid_to_evidence(evidence, grid)
        
        json1 = json.dumps(e1, sort_keys=True)
        json2 = json.dumps(e2, sort_keys=True)
        assert json1 == json2


@pytest.mark.hermetic
class TestAdversarialPriorityScenarioLedger:
    """Tests for build_adversarial_priority_scenario_ledger()."""

    def test_ledger_has_required_fields(self, sample_snapshots):
        """Ledger has all required fields."""
        ledger = build_adversarial_priority_scenario_ledger(sample_snapshots)
        
        required_fields = [
            "schema_version",
            "scenario_counts",
            "top_priority_scenarios",
            "experiments_missing_failover",
        ]
        
        for field in required_fields:
            assert field in ledger, f"Missing required field: {field}"

    def test_ledger_counts_scenarios_correctly(self, sample_snapshots):
        """Ledger counts scenario frequency correctly."""
        ledger = build_adversarial_priority_scenario_ledger(sample_snapshots)
        
        # CAL-EXP-1: ["scenario1"]
        # CAL-EXP-2: ["scenario2", "scenario3"]
        # CAL-EXP-3: ["scenario4"]
        
        assert ledger["scenario_counts"]["scenario1"] == 1
        assert ledger["scenario_counts"]["scenario2"] == 1
        assert ledger["scenario_counts"]["scenario3"] == 1
        assert ledger["scenario_counts"]["scenario4"] == 1

    def test_ledger_top_priority_scenarios_deterministic(self, sample_snapshots):
        """Ledger top_priority_scenarios is deterministic and limited to 10."""
        ledger1 = build_adversarial_priority_scenario_ledger(sample_snapshots)
        ledger2 = build_adversarial_priority_scenario_ledger(sample_snapshots)
        
        assert ledger1["top_priority_scenarios"] == ledger2["top_priority_scenarios"]
        assert len(ledger1["top_priority_scenarios"]) <= 10

    def test_ledger_scenario_counts_sorted(self, sample_snapshots):
        """Ledger scenario_counts are sorted deterministically."""
        ledger = build_adversarial_priority_scenario_ledger(sample_snapshots)
        
        # Check that scenario_counts dict keys are in sorted order
        scenarios = list(ledger["scenario_counts"].keys())
        assert scenarios == sorted(scenarios)

    def test_ledger_reuses_missing_failover(self, sample_snapshots):
        """Ledger reuses experiments_missing_failover from grid logic."""
        ledger = build_adversarial_priority_scenario_ledger(sample_snapshots)
        
        # CAL-EXP-2 and CAL-EXP-3 have has_failover == False
        assert "CAL-EXP-2" in ledger["experiments_missing_failover"]
        assert "CAL-EXP-3" in ledger["experiments_missing_failover"]
        assert "CAL-EXP-1" not in ledger["experiments_missing_failover"]

    def test_ledger_is_deterministic(self, sample_snapshots):
        """Ledger is deterministic."""
        l1 = build_adversarial_priority_scenario_ledger(sample_snapshots)
        l2 = build_adversarial_priority_scenario_ledger(sample_snapshots)
        
        json1 = json.dumps(l1, sort_keys=True)
        json2 = json.dumps(l2, sort_keys=True)
        assert json1 == json2

    def test_ledger_is_json_serializable(self, sample_snapshots):
        """Ledger is JSON-serializable."""
        ledger = build_adversarial_priority_scenario_ledger(sample_snapshots)
        
        # Should not raise
        json_str = json.dumps(ledger, sort_keys=True)
        assert isinstance(json_str, str)
        
        # Should be able to parse back
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert "scenario_counts" in parsed

    def test_ledger_handles_duplicate_scenarios(self):
        """Ledger correctly counts duplicate scenarios across experiments."""
        snapshots = [
            {
                "cal_id": "CAL-EXP-1",
                "priority_scenarios": ["s1", "s2"],
                "has_failover": True,
            },
            {
                "cal_id": "CAL-EXP-2",
                "priority_scenarios": ["s1", "s3"],
                "has_failover": True,
            },
            {
                "cal_id": "CAL-EXP-3",
                "priority_scenarios": ["s1", "s2"],
                "has_failover": True,
            },
        ]
        
        ledger = build_adversarial_priority_scenario_ledger(snapshots)
        
        # s1 appears 3 times, s2 appears 2 times, s3 appears 1 time
        assert ledger["scenario_counts"]["s1"] == 3
        assert ledger["scenario_counts"]["s2"] == 2
        assert ledger["scenario_counts"]["s3"] == 1
        
        # Top priority should be s1 (highest frequency)
        assert ledger["top_priority_scenarios"][0] == "s1"

    def test_ledger_handles_empty_snapshots(self):
        """Ledger handles empty snapshot list."""
        ledger = build_adversarial_priority_scenario_ledger([])
        
        assert len(ledger["scenario_counts"]) == 0
        assert len(ledger["top_priority_scenarios"]) == 0
        assert len(ledger["experiments_missing_failover"]) == 0


@pytest.mark.hermetic
class TestPriorityScenarioLedgerEvidenceAttachment:
    """Tests for attach_priority_scenario_ledger_to_evidence()."""

    def test_ledger_attach_non_mutating(self, sample_snapshots):
        """Ledger attach does not mutate input."""
        evidence = {"governance": {"adversarial_coverage_panel": {}}}
        ledger = build_adversarial_priority_scenario_ledger(sample_snapshots)
        
        original_evidence = json.loads(json.dumps(evidence))  # Deep copy via JSON
        enriched = attach_priority_scenario_ledger_to_evidence(evidence, ledger)
        
        # Original should be unchanged
        assert evidence["governance"]["adversarial_coverage_panel"] == {}
        # Enriched should have ledger
        assert "priority_scenario_ledger" in enriched["governance"]["adversarial_coverage_panel"]

    def test_ledger_attach_has_ledger(self, sample_snapshots):
        """Attached evidence has priority ledger."""
        evidence = {"governance": {"adversarial_coverage_panel": {}}}
        ledger = build_adversarial_priority_scenario_ledger(sample_snapshots)
        
        enriched = attach_priority_scenario_ledger_to_evidence(evidence, ledger)
        
        attached_ledger = enriched["governance"]["adversarial_coverage_panel"]["priority_scenario_ledger"]
        assert attached_ledger["scenario_counts"] == ledger["scenario_counts"]
        assert "top_priority_scenarios" in attached_ledger

    def test_ledger_attach_is_json_serializable(self, sample_snapshots):
        """Attached evidence is JSON-serializable."""
        evidence = {"governance": {"adversarial_coverage_panel": {}}}
        ledger = build_adversarial_priority_scenario_ledger(sample_snapshots)
        
        enriched = attach_priority_scenario_ledger_to_evidence(evidence, ledger)
        
        # Should not raise
        json_str = json.dumps(enriched, sort_keys=True)
        assert isinstance(json_str, str)
        
        # Should be able to parse back
        parsed = json.loads(json_str)
        assert "governance" in parsed
        assert "priority_scenario_ledger" in parsed["governance"]["adversarial_coverage_panel"]


@pytest.mark.hermetic
class TestAdversarialCoverageStatusSignal:
    """Tests for extract_adversarial_coverage_signal_for_status()."""

    def test_status_signal_extracts_correctly_from_manifest(self):
        """Status signal extracts correct fields from manifest (preferred)."""
        manifest = {
            "governance": {
                "adversarial_coverage_panel": {
                    "schema_version": "1.0.0",
                    "total_experiments": 3,
                    "experiments_missing_failover": ["CAL-EXP-2", "CAL-EXP-3"],
                    "priority_scenario_ledger": {
                        "top_priority_scenarios": ["s1", "s2", "s3", "s4", "s5", "s6"],
                    },
                },
            },
        }
        
        signal = extract_adversarial_coverage_signal_for_status(manifest=manifest)
        
        assert signal is not None
        assert signal["schema_version"] == "1.0.0"
        assert signal["mode"] == "SHADOW"
        assert signal["extraction_source"] == "MANIFEST"
        assert signal["total_experiments"] == 3
        assert signal["missing_failover_count"] == 2
        assert len(signal["top_priority_scenarios_top5"]) == 5
        assert signal["top_priority_scenarios_top5"] == ["s1", "s2", "s3", "s4", "s5"]
        assert signal["priority_scenario_ledger_present"] is True

    def test_status_signal_fallback_to_evidence(self):
        """Status signal falls back to evidence.json when manifest missing."""
        evidence = {
            "governance": {
                "adversarial_coverage_panel": {
                    "total_experiments": 2,
                    "experiments_missing_failover": ["CAL-EXP-1"],
                    "priority_scenario_ledger": {
                        "top_priority_scenarios": ["s1", "s2"],
                    },
                },
            },
        }
        
        signal = extract_adversarial_coverage_signal_for_status(evidence=evidence)
        
        assert signal is not None
        assert signal["extraction_source"] == "EVIDENCE_JSON"
        assert signal["total_experiments"] == 2
        assert signal["missing_failover_count"] == 1
        assert signal["priority_scenario_ledger_present"] is True

    def test_status_signal_manifest_preferred_over_evidence(self):
        """Status signal prefers manifest over evidence when both present."""
        manifest = {
            "governance": {
                "adversarial_coverage_panel": {
                    "total_experiments": 3,
                    "experiments_missing_failover": [],
                },
            },
        }
        evidence = {
            "governance": {
                "adversarial_coverage_panel": {
                    "total_experiments": 2,
                    "experiments_missing_failover": ["CAL-EXP-1"],
                },
            },
        }
        
        signal = extract_adversarial_coverage_signal_for_status(manifest=manifest, evidence=evidence)
        
        # Should use manifest (total_experiments = 3)
        assert signal is not None
        assert signal["extraction_source"] == "MANIFEST"
        assert signal["total_experiments"] == 3
        assert signal["missing_failover_count"] == 0

    def test_status_signal_returns_none_when_missing(self):
        """Status signal returns None when coverage panel not found."""
        signal = extract_adversarial_coverage_signal_for_status()
        
        assert signal is None

    def test_status_signal_handles_missing_ledger(self):
        """Status signal handles missing priority ledger gracefully with safe defaults."""
        manifest = {
            "governance": {
                "adversarial_coverage_panel": {
                    "total_experiments": 2,
                    "experiments_missing_failover": ["CAL-EXP-1"],
                    # priority_scenario_ledger missing
                },
            },
        }
        
        signal = extract_adversarial_coverage_signal_for_status(manifest=manifest)
        
        assert signal is not None
        assert signal["total_experiments"] == 2
        assert signal["missing_failover_count"] == 1
        assert signal["top_priority_scenarios_top5"] == []  # Safe default
        assert signal["priority_scenario_ledger_present"] is False

    def test_status_signal_is_deterministic(self):
        """Status signal extraction is deterministic."""
        manifest = {
            "governance": {
                "adversarial_coverage_panel": {
                    "total_experiments": 3,
                    "experiments_missing_failover": ["CAL-EXP-2"],
                    "priority_scenario_ledger": {
                        "top_priority_scenarios": ["s1", "s2", "s3"],
                    },
                },
            },
        }
        
        s1 = extract_adversarial_coverage_signal_for_status(manifest=manifest)
        s2 = extract_adversarial_coverage_signal_for_status(manifest=manifest)
        
        assert s1 == s2

    def test_status_signal_is_json_serializable(self):
        """Status signal is JSON-serializable."""
        manifest = {
            "governance": {
                "adversarial_coverage_panel": {
                    "total_experiments": 1,
                    "experiments_missing_failover": [],
                    "priority_scenario_ledger": {
                        "top_priority_scenarios": ["s1"],
                    },
                },
            },
        }
        
        signal = extract_adversarial_coverage_signal_for_status(manifest=manifest)
        
        # Should not raise
        json_str = json.dumps(signal, sort_keys=True)
        assert isinstance(json_str, str)
        
        # Should be able to parse back
        parsed = json.loads(json_str)
        assert parsed["total_experiments"] == 1
        assert parsed["mode"] == "SHADOW"


@pytest.mark.hermetic
class TestAdversarialCoverageGGFLAdapter:
    """Tests for adversarial_coverage_for_alignment_view()."""

    def test_ggfl_adapter_has_required_fields(self):
        """GGFL adapter has all required fields."""
        signal = {
            "total_experiments": 3,
            "missing_failover_count": 0,
            "top_priority_scenarios_top5": ["s1", "s2"],
        }
        
        view = adversarial_coverage_for_alignment_view(signal)
        
        required_fields = ["signal_type", "status", "conflict", "drivers", "summary"]
        for field in required_fields:
            assert field in view, f"Missing required field: {field}"

    def test_ggfl_adapter_signal_type(self):
        """GGFL adapter has correct signal_type."""
        signal = {"total_experiments": 1}
        view = adversarial_coverage_for_alignment_view(signal)
        
        assert view["signal_type"] == "SIG-ADV"

    def test_ggfl_adapter_conflict_always_false(self):
        """GGFL adapter conflict is always false (advisory only)."""
        signal = {"total_experiments": 1, "missing_failover_count": 10}
        view = adversarial_coverage_for_alignment_view(signal)
        
        assert view["conflict"] is False

    def test_ggfl_adapter_status_ok_when_no_issues(self):
        """GGFL adapter status is ok when no missing failover and no repeated scenarios."""
        signal = {
            "total_experiments": 3,
            "missing_failover_count": 0,
            "top_priority_scenarios_top5": ["s1", "s2", "s3", "s4", "s5"],
        }
        
        view = adversarial_coverage_for_alignment_view(signal)
        
        assert view["status"] == "ok"
        assert len(view["drivers"]) == 0

    def test_ggfl_adapter_status_warn_on_missing_failover(self):
        """GGFL adapter status is warn when missing_failover_count > 0."""
        signal = {
            "total_experiments": 3,
            "missing_failover_count": 2,
            "top_priority_scenarios_top5": ["s1", "s2"],
        }
        
        view = adversarial_coverage_for_alignment_view(signal)
        
        assert view["status"] == "warn"
        assert len(view["drivers"]) == 1
        assert view["drivers"][0] == DRIVER_MISSING_FAILOVER_COUNT

    def test_ggfl_adapter_status_warn_on_repeated_scenarios(self):
        """GGFL adapter status is warn when scenarios repeated >= 2 times."""
        signal = {
            "total_experiments": 3,
            "missing_failover_count": 0,
            "top_priority_scenarios_top5": ["s1", "s2", "s1", "s3", "s2"],
        }
        
        view = adversarial_coverage_for_alignment_view(signal)
        
        assert view["status"] == "warn"
        assert len(view["drivers"]) == 1
        assert view["drivers"][0] == DRIVER_REPEATED_PRIORITY_SCENARIOS

    def test_ggfl_adapter_drivers_ordered_deterministically(self):
        """GGFL adapter drivers are ordered deterministically (missing failover first)."""
        signal = {
            "total_experiments": 3,
            "missing_failover_count": 2,
            "top_priority_scenarios_top5": ["s1", "s2", "s1"],
        }
        
        view = adversarial_coverage_for_alignment_view(signal)
        
        # Missing failover should be first, then repeated scenarios
        assert len(view["drivers"]) == 2
        assert view["drivers"][0] == DRIVER_MISSING_FAILOVER_COUNT
        assert view["drivers"][1] == DRIVER_REPEATED_PRIORITY_SCENARIOS

    def test_ggfl_adapter_has_shadow_mode_invariants(self):
        """GGFL adapter includes shadow_mode_invariants with required keys and exact boolean values."""
        signal = {
            "total_experiments": 3,
            "missing_failover_count": 1,
            "top_priority_scenarios_top5": ["s1", "s2"],
        }
        
        view = adversarial_coverage_for_alignment_view(signal)
        
        assert "shadow_mode_invariants" in view
        invariants = view["shadow_mode_invariants"]
        
        # Assert required keys exist
        required_keys = {"advisory_only", "no_enforcement", "conflict_invariant"}
        assert set(invariants.keys()) == required_keys, f"Expected keys {required_keys}, got {set(invariants.keys())}"
        
        # Assert all values are exactly True (boolean, not truthy)
        assert invariants["advisory_only"] is True, "advisory_only must be exactly True (boolean)"
        assert invariants["no_enforcement"] is True, "no_enforcement must be exactly True (boolean)"
        assert invariants["conflict_invariant"] is True, "conflict_invariant must be exactly True (boolean)"
        
        # Assert conflict is always False
        assert view["conflict"] is False

    def test_ggfl_adapter_summary_one_sentence(self):
        """GGFL adapter summary is one neutral sentence."""
        signal = {
            "total_experiments": 3,
            "missing_failover_count": 1,
            "top_priority_scenarios_top5": [],
        }
        
        view = adversarial_coverage_for_alignment_view(signal)
        
        summary = view["summary"]
        # Should be one sentence (no periods except at end, or no periods at all)
        period_count = summary.count(".")
        assert period_count <= 1
        assert len(summary) > 0

    def test_ggfl_adapter_summary_includes_both_issues_when_present(self):
        """GGFL adapter summary mentions both missing failover and repeated scenarios in single sentence."""
        signal = {
            "total_experiments": 3,
            "missing_failover_count": 2,
            "top_priority_scenarios_top5": ["s1", "s2", "s1"],
        }
        
        view = adversarial_coverage_for_alignment_view(signal)
        
        summary = view["summary"]
        # Should mention both issues in one sentence
        assert "missing failover" in summary.lower()
        assert "repeated" in summary.lower()
        # Should be one sentence
        period_count = summary.count(".")
        assert period_count <= 1

    def test_ggfl_adapter_works_with_panel_format(self):
        """GGFL adapter works with panel format (not just signal format)."""
        panel = {
            "total_experiments": 2,
            "experiments_missing_failover": ["CAL-EXP-1"],
            "priority_scenario_ledger": {
                "top_priority_scenarios": ["s1", "s2", "s3"],
            },
        }
        
        view = adversarial_coverage_for_alignment_view(panel)
        
        assert view["signal_type"] == "SIG-ADV"
        assert view["status"] == "warn"  # missing failover
        assert view["drivers"][0] == DRIVER_MISSING_FAILOVER_COUNT

    def test_ggfl_adapter_is_deterministic(self):
        """GGFL adapter is deterministic."""
        signal = {
            "total_experiments": 3,
            "missing_failover_count": 1,
            "top_priority_scenarios_top5": ["s1", "s2", "s1"],
        }
        
        v1 = adversarial_coverage_for_alignment_view(signal)
        v2 = adversarial_coverage_for_alignment_view(signal)
        
        assert v1 == v2

    def test_ggfl_adapter_is_json_serializable(self):
        """GGFL adapter output is JSON-serializable."""
        signal = {"total_experiments": 1}
        view = adversarial_coverage_for_alignment_view(signal)
        
        # Should not raise
        json_str = json.dumps(view, sort_keys=True)
        assert isinstance(json_str, str)
        
        # Should be able to parse back
        parsed = json.loads(json_str)
        assert parsed["signal_type"] == "SIG-ADV"



