"""
Tests for Evidence Quality Attachment Helper (D3 Phase X).

Tests for attach_evidence_quality_to_evidence helper function.
"""

import json
import pytest
from typing import Dict, Any


class TestAttachEvidenceQualityToEvidence:
    """Tests for attach_evidence_quality_to_evidence helper."""
    
    @pytest.mark.unit
    def test_attach_evidence_quality_shape(self):
        """Test 203: Attach function returns correct shape."""
        from backend.health.evidence_quality_adapter import (
            attach_evidence_quality_to_evidence,
            build_evidence_governance_tile,
        )
        
        evidence = {
            "timestamp": "2024-01-01",
            "data": {"test": "value"},
        }
        
        tile = build_evidence_governance_tile(
            director_panel_v2={
                "status_light": "GREEN",
                "trajectory_class": "IMPROVING",
                "regression_status": "OK",
                "flags": [],
                "headline": "Evidence quality is good",
            },
            forecast={
                "predicted_band": "HIGH",
                "cycles_until_risk": 5,
            },
        )
        
        enriched = attach_evidence_quality_to_evidence(evidence, tile)
        
        assert "governance" in enriched
        assert "evidence_quality" in enriched["governance"]
        assert enriched["governance"]["evidence_quality"]["trajectory_class"] == "IMPROVING"
        assert enriched["governance"]["evidence_quality"]["predicted_band"] == "HIGH"
        assert enriched["governance"]["evidence_quality"]["cycles_until_risk"] == 5
        assert isinstance(enriched["governance"]["evidence_quality"]["flags"], list)
    
    @pytest.mark.unit
    def test_attach_evidence_quality_deterministic(self):
        """Test 204: Attach function is deterministic."""
        from backend.health.evidence_quality_adapter import (
            attach_evidence_quality_to_evidence,
            build_evidence_governance_tile,
        )
        
        evidence = {"timestamp": "2024-01-01"}
        tile = build_evidence_governance_tile(
            director_panel_v2={
                "trajectory_class": "STABLE",
                "flags": ["flag1", "flag2"],
            },
            forecast={"predicted_band": "MEDIUM", "cycles_until_risk": 3},
        )
        
        enriched1 = attach_evidence_quality_to_evidence(evidence, tile)
        enriched2 = attach_evidence_quality_to_evidence(evidence, tile)
        
        json1 = json.dumps(enriched1, sort_keys=True)
        json2 = json.dumps(enriched2, sort_keys=True)
        
        assert json1 == json2, "Attach function should be deterministic"
    
    @pytest.mark.unit
    def test_attach_evidence_quality_json_serializable(self):
        """Test 205: Attach function output is JSON serializable."""
        from backend.health.evidence_quality_adapter import (
            attach_evidence_quality_to_evidence,
            build_evidence_governance_tile,
        )
        
        evidence = {"timestamp": "2024-01-01", "nested": {"key": "value"}}
        tile = build_evidence_governance_tile(
            director_panel_v2={"trajectory_class": "OSCILLATING"},
            forecast={"predicted_band": "LOW", "cycles_until_risk": 1},
        )
        
        enriched = attach_evidence_quality_to_evidence(evidence, tile)
        
        # Should not raise
        json_str = json.dumps(enriched, sort_keys=True)
        assert len(json_str) > 0
        
        # Round-trip should work
        parsed = json.loads(json_str)
        assert parsed["governance"]["evidence_quality"]["trajectory_class"] == "OSCILLATING"
    
    @pytest.mark.unit
    def test_attach_evidence_quality_non_mutating(self):
        """Test 206: Attach function does not mutate input."""
        from backend.health.evidence_quality_adapter import (
            attach_evidence_quality_to_evidence,
            build_evidence_governance_tile,
        )
        
        evidence = {"timestamp": "2024-01-01", "data": {"key": "value"}}
        evidence_copy = json.loads(json.dumps(evidence))
        
        tile = build_evidence_governance_tile(
            director_panel_v2={"trajectory_class": "DEGRADING"},
        )
        
        enriched = attach_evidence_quality_to_evidence(evidence, tile)
        
        # Original evidence should be unchanged
        assert json.dumps(evidence, sort_keys=True) == json.dumps(evidence_copy, sort_keys=True)
        
        # Enriched should be a new dict
        assert enriched is not evidence
        assert "governance" not in evidence
        assert "governance" in enriched
    
    @pytest.mark.unit
    def test_attach_evidence_quality_handles_existing_governance(self):
        """Test 207: Attach function handles existing governance section."""
        from backend.health.evidence_quality_adapter import (
            attach_evidence_quality_to_evidence,
            build_evidence_governance_tile,
        )
        
        evidence = {
            "timestamp": "2024-01-01",
            "governance": {
                "other_tile": {"status": "OK"},
            },
        }
        
        tile = build_evidence_governance_tile(
            director_panel_v2={"trajectory_class": "IMPROVING"},
        )
        
        enriched = attach_evidence_quality_to_evidence(evidence, tile)
        
        # Should preserve existing governance
        assert "other_tile" in enriched["governance"]
        assert enriched["governance"]["other_tile"]["status"] == "OK"
        
        # Should add evidence_quality
        assert "evidence_quality" in enriched["governance"]
        assert enriched["governance"]["evidence_quality"]["trajectory_class"] == "IMPROVING"
    
    @pytest.mark.unit
    def test_extract_evidence_quality_summary_for_first_light(self):
        """Test 208: Extract summary function works correctly."""
        from backend.health.evidence_quality_adapter import (
            extract_evidence_quality_summary_for_first_light,
        )
        
        director_panel_v2 = {
            "trajectory_class": "IMPROVING",
            "flags": ["flag1", "flag2"],
        }
        forecast = {
            "predicted_band": "HIGH",
            "cycles_until_risk": 5,
        }
        
        summary = extract_evidence_quality_summary_for_first_light(
            director_panel_v2=director_panel_v2,
            forecast=forecast,
        )
        
        assert summary["trajectory_class"] == "IMPROVING"
        assert summary["predicted_band"] == "HIGH"
        assert summary["cycles_until_risk"] == 5
        assert summary["flags"] == ["flag1", "flag2"]
        
        # Test with None inputs
        summary_empty = extract_evidence_quality_summary_for_first_light()
        assert summary_empty["trajectory_class"] == "UNKNOWN"
        assert summary_empty["predicted_band"] == "UNKNOWN"
        assert summary_empty["cycles_until_risk"] is None
        assert summary_empty["flags"] == []


class TestFirstLightFailureShelf:
    """Tests for build_first_light_failure_shelf helper."""
    
    @pytest.mark.unit
    def test_failure_shelf_shape(self):
        """Test 209: Failure shelf has correct shape."""
        from backend.health.evidence_quality_adapter import (
            build_first_light_failure_shelf,
        )
        
        phase_portrait = {
            "phase_points": [[0, 2], [1, 1], [2, 1]],
            "trajectory_class": "DEGRADING",
            "neutral_notes": [],
        }
        
        forecast = {
            "predicted_band": "LOW",
            "confidence": 0.5,
            "cycles_until_risk": 1,
            "neutral_explanation": [
                "regression watchdog indicates blocking condition",
                "evidence quality is degrading",
            ],
        }
        
        shelf = build_first_light_failure_shelf(phase_portrait, forecast)
        
        assert "schema_version" in shelf
        assert shelf["schema_version"] == "1.0.0"
        assert shelf["trajectory_class"] == "DEGRADING"
        assert shelf["predicted_band"] == "LOW"
        assert shelf["cycles_until_risk"] == 1
        assert isinstance(shelf["flags"], list)
        assert len(shelf["flags"]) == 2
    
    @pytest.mark.unit
    def test_failure_shelf_json_serializable(self):
        """Test 210: Failure shelf is JSON serializable."""
        from backend.health.evidence_quality_adapter import (
            build_first_light_failure_shelf,
        )
        
        phase_portrait = {
            "trajectory_class": "OSCILLATING",
            "phase_points": [[0, 3], [1, 2], [2, 3]],
        }
        
        forecast = {
            "predicted_band": "MEDIUM",
            "cycles_until_risk": 2,
            "neutral_explanation": ["oscillating pattern detected"],
        }
        
        shelf = build_first_light_failure_shelf(phase_portrait, forecast)
        
        # Should not raise
        json_str = json.dumps(shelf, sort_keys=True)
        assert len(json_str) > 0
        
        # Round-trip should work
        parsed = json.loads(json_str)
        assert parsed["trajectory_class"] == "OSCILLATING"
        assert parsed["predicted_band"] == "MEDIUM"
    
    @pytest.mark.unit
    def test_failure_shelf_deterministic(self):
        """Test 211: Failure shelf is deterministic."""
        from backend.health.evidence_quality_adapter import (
            build_first_light_failure_shelf,
        )
        
        phase_portrait = {
            "trajectory_class": "IMPROVING",
            "phase_points": [[0, 1], [1, 2], [2, 3]],
        }
        
        forecast = {
            "predicted_band": "HIGH",
            "cycles_until_risk": 5,
            "neutral_explanation": ["evidence quality is improving"],
        }
        
        shelf1 = build_first_light_failure_shelf(phase_portrait, forecast)
        shelf2 = build_first_light_failure_shelf(phase_portrait, forecast)
        
        json1 = json.dumps(shelf1, sort_keys=True)
        json2 = json.dumps(shelf2, sort_keys=True)
        
        assert json1 == json2, "Failure shelf should be deterministic"
    
    @pytest.mark.unit
    def test_failure_shelf_flags_sorted(self):
        """Test 212: Failure shelf flags are sorted for determinism."""
        from backend.health.evidence_quality_adapter import (
            build_first_light_failure_shelf,
        )
        
        phase_portrait = {"trajectory_class": "STABLE"}
        
        forecast = {
            "predicted_band": "MEDIUM",
            "neutral_explanation": [
                "evidence quality shows attention points",
                "limited data points reduce forecast confidence",
            ],
        }
        
        shelf = build_first_light_failure_shelf(phase_portrait, forecast)
        
        # Flags should be sorted
        assert shelf["flags"] == sorted(shelf["flags"])
        assert shelf["flags"][0] == "evidence quality shows attention points"
    
    @pytest.mark.unit
    def test_attach_evidence_quality_includes_failure_shelf(self):
        """Test 213: attach_evidence_quality_to_evidence includes failure shelf when provided."""
        from backend.health.evidence_quality_adapter import (
            attach_evidence_quality_to_evidence,
            build_evidence_governance_tile,
        )
        
        evidence = {"timestamp": "2024-01-01"}
        
        phase_portrait = {
            "trajectory_class": "DEGRADING",
            "phase_points": [[0, 3], [1, 2]],
        }
        
        forecast = {
            "predicted_band": "LOW",
            "cycles_until_risk": 0,
            "neutral_explanation": ["regression detected"],
        }
        
        tile = build_evidence_governance_tile(
            phase_portrait=phase_portrait,
            forecast=forecast,
        )
        
        enriched = attach_evidence_quality_to_evidence(
            evidence, tile, phase_portrait=phase_portrait, forecast=forecast
        )
        
        assert "first_light_failure_shelf" in enriched["governance"]["evidence_quality"]
        shelf = enriched["governance"]["evidence_quality"]["first_light_failure_shelf"]
        assert shelf["trajectory_class"] == "DEGRADING"
        assert shelf["predicted_band"] == "LOW"
        assert shelf["cycles_until_risk"] == 0
    
    @pytest.mark.unit
    def test_attach_evidence_quality_no_failure_shelf_when_missing(self):
        """Test 214: attach_evidence_quality_to_evidence works without failure shelf inputs."""
        from backend.health.evidence_quality_adapter import (
            attach_evidence_quality_to_evidence,
            build_evidence_governance_tile,
        )
        
        evidence = {"timestamp": "2024-01-01"}
        tile = build_evidence_governance_tile()
        
        enriched = attach_evidence_quality_to_evidence(evidence, tile)
        
        # Should still work without failure shelf
        assert "evidence_quality" in enriched["governance"]
        assert "first_light_failure_shelf" not in enriched["governance"]["evidence_quality"]


class TestCalExpFailureShelf:
    """Tests for emit_cal_exp_failure_shelf helper."""
    
    @pytest.mark.unit
    def test_emit_cal_exp_failure_shelf_shape(self):
        """Test 215: Emit CAL-EXP shelf has correct shape."""
        from backend.health.evidence_quality_adapter import (
            build_first_light_failure_shelf,
            emit_cal_exp_failure_shelf,
        )
        import tempfile
        from pathlib import Path
        
        phase_portrait = {"trajectory_class": "DEGRADING", "phase_points": [[0, 3], [1, 2]]}
        forecast = {
            "predicted_band": "LOW",
            "cycles_until_risk": 0,
            "neutral_explanation": ["regression detected"],
        }
        
        shelf = build_first_light_failure_shelf(phase_portrait, forecast)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            enriched = emit_cal_exp_failure_shelf("cal_exp1", shelf, output_dir=Path(tmpdir))
            
            assert enriched["schema_version"] == "1.0.0"
            assert enriched["cal_id"] == "cal_exp1"
            assert enriched["episode_id"] == "cal_exp1_episode_1"
            assert enriched["rank"] == 1
            assert enriched["trajectory_class"] == "DEGRADING"
            assert enriched["predicted_band"] == "LOW"
            assert enriched["cycles_until_risk"] == 0
            assert len(enriched["flags"]) == 1
            
            # Verify file was written
            shelf_file = Path(tmpdir) / "evidence_failure_shelf_cal_exp1.json"
            assert shelf_file.exists()
    
    @pytest.mark.unit
    def test_emit_cal_exp_failure_shelf_no_persistence(self):
        """Test 216: Emit CAL-EXP shelf works without persistence."""
        from backend.health.evidence_quality_adapter import (
            build_first_light_failure_shelf,
            emit_cal_exp_failure_shelf,
        )
        
        shelf = build_first_light_failure_shelf(
            {"trajectory_class": "STABLE"},
            {"predicted_band": "MEDIUM", "cycles_until_risk": 3}
        )
        
        enriched = emit_cal_exp_failure_shelf("cal_exp2", shelf, output_dir=None)
        
        assert enriched["cal_id"] == "cal_exp2"
        assert enriched["trajectory_class"] == "STABLE"
    
    @pytest.mark.unit
    def test_emit_cal_exp_failure_shelf_json_serializable(self):
        """Test 217: Emit CAL-EXP shelf is JSON serializable."""
        from backend.health.evidence_quality_adapter import (
            build_first_light_failure_shelf,
            emit_cal_exp_failure_shelf,
        )
        
        shelf = build_first_light_failure_shelf(
            {"trajectory_class": "OSCILLATING"},
            {"predicted_band": "LOW", "neutral_explanation": ["flag1", "flag2"]}
        )
        
        enriched = emit_cal_exp_failure_shelf("cal_exp3", shelf)
        
        json_str = json.dumps(enriched, sort_keys=True)
        assert len(json_str) > 0
        
        parsed = json.loads(json_str)
        assert parsed["cal_id"] == "cal_exp3"
        assert parsed["flags"] == sorted(["flag1", "flag2"])


class TestGlobalFailureShortlist:
    """Tests for build_global_failure_shortlist helper."""
    
    @pytest.mark.unit
    def test_global_failure_shortlist_shape(self):
        """Test 218: Global shortlist has correct shape."""
        from backend.health.evidence_quality_adapter import (
            build_first_light_failure_shelf,
            emit_cal_exp_failure_shelf,
            build_global_failure_shortlist,
        )
        
        # Create multiple shelves
        shelf1 = build_first_light_failure_shelf(
            {"trajectory_class": "DEGRADING"},
            {"predicted_band": "LOW", "cycles_until_risk": 0, "neutral_explanation": ["critical"]}
        )
        shelf2 = build_first_light_failure_shelf(
            {"trajectory_class": "STABLE"},
            {"predicted_band": "HIGH", "cycles_until_risk": 5, "neutral_explanation": ["minor"]}
        )
        
        enriched1 = emit_cal_exp_failure_shelf("cal_exp1", shelf1)
        enriched2 = emit_cal_exp_failure_shelf("cal_exp2", shelf2)
        
        shortlist = build_global_failure_shortlist([enriched1, enriched2], max_items=10)
        
        assert shortlist["schema_version"] == "1.0.0"
        assert shortlist["total_shelves"] == 2
        assert len(shortlist["items"]) == 2
        assert shortlist["items"][0]["cal_id"] == "cal_exp1"  # LOW band should be first
        assert shortlist["items"][0]["rank"] == 1
        assert shortlist["items"][1]["rank"] == 2
    
    @pytest.mark.unit
    def test_global_failure_shortlist_severity_ordering(self):
        """Test 219: Global shortlist orders by severity correctly."""
        from backend.health.evidence_quality_adapter import (
            build_first_light_failure_shelf,
            emit_cal_exp_failure_shelf,
            build_global_failure_shortlist,
        )
        
        # Create shelves with different severity levels
        shelves = []
        for i, (traj, band, cycles) in enumerate([
            ("DEGRADING", "LOW", 0),
            ("OSCILLATING", "LOW", 1),
            ("STABLE", "MEDIUM", 2),
            ("IMPROVING", "HIGH", 5),
        ]):
            shelf = build_first_light_failure_shelf(
                {"trajectory_class": traj},
                {"predicted_band": band, "cycles_until_risk": cycles}
            )
            enriched = emit_cal_exp_failure_shelf(f"cal_exp{i+1}", shelf)
            shelves.append(enriched)
        
        shortlist = build_global_failure_shortlist(shelves, max_items=10)
        
        # Should be ordered: LOW/0/DEGRADING, LOW/1/OSCILLATING, MEDIUM/2/STABLE, HIGH/5/IMPROVING
        items = shortlist["items"]
        assert items[0]["predicted_band"] == "LOW"
        assert items[0]["cycles_until_risk"] == 0
        assert items[0]["trajectory_class"] == "DEGRADING"
        assert items[1]["predicted_band"] == "LOW"
        assert items[1]["cycles_until_risk"] == 1
        assert items[2]["predicted_band"] == "MEDIUM"
        assert items[3]["predicted_band"] == "HIGH"
    
    @pytest.mark.unit
    def test_global_failure_shortlist_truncation(self):
        """Test 220: Global shortlist truncates to max_items."""
        from backend.health.evidence_quality_adapter import (
            build_first_light_failure_shelf,
            emit_cal_exp_failure_shelf,
            build_global_failure_shortlist,
        )
        
        # Create 15 shelves
        shelves = []
        for i in range(15):
            shelf = build_first_light_failure_shelf(
                {"trajectory_class": "DEGRADING"},
                {"predicted_band": "LOW", "cycles_until_risk": i}
            )
            enriched = emit_cal_exp_failure_shelf(f"cal_exp{i+1}", shelf)
            shelves.append(enriched)
        
        shortlist = build_global_failure_shortlist(shelves, max_items=5)
        
        assert shortlist["total_shelves"] == 15
        assert len(shortlist["items"]) == 5
        assert shortlist["items"][0]["rank"] == 1
        assert shortlist["items"][4]["rank"] == 5
    
    @pytest.mark.unit
    def test_global_failure_shortlist_empty(self):
        """Test 221: Global shortlist handles empty input."""
        from backend.health.evidence_quality_adapter import build_global_failure_shortlist
        
        shortlist = build_global_failure_shortlist([], max_items=10)
        
        assert shortlist["schema_version"] == "1.0.0"
        assert shortlist["total_shelves"] == 0
        assert shortlist["items"] == []
    
    @pytest.mark.unit
    def test_global_failure_shortlist_deterministic(self):
        """Test 222: Global shortlist is deterministic."""
        from backend.health.evidence_quality_adapter import (
            build_first_light_failure_shelf,
            emit_cal_exp_failure_shelf,
            build_global_failure_shortlist,
        )
        
        shelves = []
        for i in range(5):
            shelf = build_first_light_failure_shelf(
                {"trajectory_class": "DEGRADING"},
                {"predicted_band": "LOW", "cycles_until_risk": i}
            )
            enriched = emit_cal_exp_failure_shelf(f"cal_exp{i+1}", shelf)
            shelves.append(enriched)
        
        shortlist1 = build_global_failure_shortlist(shelves, max_items=10)
        shortlist2 = build_global_failure_shortlist(shelves, max_items=10)
        
        json1 = json.dumps(shortlist1, sort_keys=True)
        json2 = json.dumps(shortlist2, sort_keys=True)
        
        assert json1 == json2, "Global shortlist should be deterministic"
    
    @pytest.mark.unit
    def test_global_failure_shortlist_json_serializable(self):
        """Test 223: Global shortlist is JSON serializable."""
        from backend.health.evidence_quality_adapter import (
            build_first_light_failure_shelf,
            emit_cal_exp_failure_shelf,
            build_global_failure_shortlist,
        )
        
        shelf = build_first_light_failure_shelf(
            {"trajectory_class": "STABLE"},
            {"predicted_band": "MEDIUM", "cycles_until_risk": 3}
        )
        enriched = emit_cal_exp_failure_shelf("cal_exp1", shelf)
        
        shortlist = build_global_failure_shortlist([enriched], max_items=10)
        
        json_str = json.dumps(shortlist, sort_keys=True)
        assert len(json_str) > 0
        
        parsed = json.loads(json_str)
        assert parsed["total_shelves"] == 1
        assert len(parsed["items"]) == 1


class TestAttachGlobalFailureShortlist:
    """Tests for attach_global_failure_shortlist_to_evidence helper."""
    
    @pytest.mark.unit
    def test_attach_global_failure_shortlist_shape(self):
        """Test 224: Attach global shortlist returns correct shape."""
        from backend.health.evidence_quality_adapter import (
            build_global_failure_shortlist,
            attach_global_failure_shortlist_to_evidence,
        )
        
        evidence = {"timestamp": "2024-01-01"}
        shortlist = build_global_failure_shortlist([], max_items=10)
        
        enriched = attach_global_failure_shortlist_to_evidence(evidence, shortlist)
        
        assert "governance" in enriched
        assert "evidence_failure_shortlist" in enriched["governance"]
        assert enriched["governance"]["evidence_failure_shortlist"]["total_shelves"] == 0
    
    @pytest.mark.unit
    def test_attach_global_failure_shortlist_non_mutating(self):
        """Test 225: Attach global shortlist does not mutate input."""
        from backend.health.evidence_quality_adapter import (
            build_global_failure_shortlist,
            attach_global_failure_shortlist_to_evidence,
        )
        
        evidence = {"timestamp": "2024-01-01", "data": {"key": "value"}}
        evidence_copy = json.loads(json.dumps(evidence))
        
        shortlist = build_global_failure_shortlist([], max_items=10)
        enriched = attach_global_failure_shortlist_to_evidence(evidence, shortlist)
        
        # Original evidence should be unchanged
        assert json.dumps(evidence, sort_keys=True) == json.dumps(evidence_copy, sort_keys=True)
        assert enriched is not evidence
        assert "governance" not in evidence
        assert "governance" in enriched


class TestFailureShortlistNavigationHints:
    """Tests for navigation hints in failure shortlist."""
    
    @pytest.mark.unit
    def test_shortlist_items_include_path_hints(self):
        """Test 226: Shortlist items include evidence_path_hint."""
        from backend.health.evidence_quality_adapter import (
            build_first_light_failure_shelf,
            emit_cal_exp_failure_shelf,
            build_global_failure_shortlist,
        )
        
        shelf1 = build_first_light_failure_shelf(
            {"trajectory_class": "DEGRADING"},
            {"predicted_band": "LOW", "cycles_until_risk": 0}
        )
        shelf2 = build_first_light_failure_shelf(
            {"trajectory_class": "STABLE"},
            {"predicted_band": "MEDIUM", "cycles_until_risk": 2}
        )
        
        enriched1 = emit_cal_exp_failure_shelf("cal_exp1", shelf1)
        enriched2 = emit_cal_exp_failure_shelf("cal_exp2", shelf2)
        
        shortlist = build_global_failure_shortlist([enriched1, enriched2], max_items=10)
        
        items = shortlist["items"]
        assert len(items) == 2
        assert items[0]["evidence_path_hint"] == "calibration/evidence_failure_shelf_cal_exp1.json"
        assert items[1]["evidence_path_hint"] == "calibration/evidence_failure_shelf_cal_exp2.json"
    
    @pytest.mark.unit
    def test_path_hints_deterministic(self):
        """Test 227: Path hints are deterministic."""
        from backend.health.evidence_quality_adapter import (
            build_first_light_failure_shelf,
            emit_cal_exp_failure_shelf,
            build_global_failure_shortlist,
        )
        
        shelf = build_first_light_failure_shelf(
            {"trajectory_class": "DEGRADING"},
            {"predicted_band": "LOW", "cycles_until_risk": 0}
        )
        enriched = emit_cal_exp_failure_shelf("cal_exp1", shelf)
        
        shortlist1 = build_global_failure_shortlist([enriched], max_items=10)
        shortlist2 = build_global_failure_shortlist([enriched], max_items=10)
        
        json1 = json.dumps(shortlist1, sort_keys=True)
        json2 = json.dumps(shortlist2, sort_keys=True)
        
        assert json1 == json2
        assert shortlist1["items"][0]["evidence_path_hint"] == shortlist2["items"][0]["evidence_path_hint"]


class TestFailureShortlistStatusSignal:
    """Tests for status signal extraction."""
    
    @pytest.mark.unit
    def test_extract_status_signal_shape(self):
        """Test 228: Status signal extraction returns correct shape."""
        from backend.health.evidence_quality_adapter import (
            build_first_light_failure_shelf,
            emit_cal_exp_failure_shelf,
            build_global_failure_shortlist,
            attach_global_failure_shortlist_to_evidence,
            extract_evidence_failure_shortlist_signal_for_status,
        )
        
        # Build shortlist
        shelves = []
        for i in range(3):
            shelf = build_first_light_failure_shelf(
                {"trajectory_class": "DEGRADING"},
                {"predicted_band": "LOW", "cycles_until_risk": i}
            )
            enriched = emit_cal_exp_failure_shelf(f"cal_exp{i+1}", shelf)
            shelves.append(enriched)
        
        shortlist = build_global_failure_shortlist(shelves, max_items=10)
        evidence = attach_global_failure_shortlist_to_evidence({}, shortlist)
        
        # Extract signal
        signal = extract_evidence_failure_shortlist_signal_for_status(
            pack_manifest=None,
            evidence_data=evidence,
        )
        
        assert signal is not None
        assert signal["extraction_source"] == "evidence.json"
        assert signal["total_items"] == 3
        assert len(signal["top5"]) == 3
        
        # Check top5 structure
        for item in signal["top5"]:
            assert "cal_id" in item
            assert "episode_id" in item
            assert "predicted_band" in item
            assert "cycles_until_risk" in item
    
    @pytest.mark.unit
    def test_extract_status_signal_manifest_first(self):
        """Test 228b: Status signal prefers manifest over evidence.json."""
        from backend.health.evidence_quality_adapter import (
            extract_evidence_failure_shortlist_signal_for_status,
        )
        
        pack_manifest = {
            "governance": {
                "evidence_failure_shortlist": {
                    "schema_version": "1.0.0",
                    "items": [{"cal_id": "cal_exp1", "predicted_band": "LOW"}],
                    "total_shelves": 1,
                },
            },
        }
        
        evidence_data = {
            "governance": {
                "evidence_failure_shortlist": {
                    "schema_version": "1.0.0",
                    "items": [{"cal_id": "cal_exp2", "predicted_band": "MEDIUM"}],
                    "total_shelves": 1,
                },
            },
        }
        
        signal = extract_evidence_failure_shortlist_signal_for_status(
            pack_manifest=pack_manifest,
            evidence_data=evidence_data,
        )
        
        assert signal["extraction_source"] == "manifest"
        assert signal["top5"][0]["cal_id"] == "cal_exp1"  # Should use manifest, not evidence.json
    
    @pytest.mark.unit
    def test_extract_status_signal_top5_truncation(self):
        """Test 229: Status signal truncates to top 5."""
        from backend.health.evidence_quality_adapter import (
            build_first_light_failure_shelf,
            emit_cal_exp_failure_shelf,
            build_global_failure_shortlist,
            attach_global_failure_shortlist_to_evidence,
            extract_evidence_failure_shortlist_signal_for_status,
        )
        
        # Build shortlist with 10 items
        shelves = []
        for i in range(10):
            shelf = build_first_light_failure_shelf(
                {"trajectory_class": "DEGRADING"},
                {"predicted_band": "LOW", "cycles_until_risk": i}
            )
            enriched = emit_cal_exp_failure_shelf(f"cal_exp{i+1}", shelf)
            shelves.append(enriched)
        
        shortlist = build_global_failure_shortlist(shelves, max_items=10)
        evidence = attach_global_failure_shortlist_to_evidence({}, shortlist)
        
        signal = extract_evidence_failure_shortlist_signal_for_status(
            pack_manifest=None,
            evidence_data=evidence,
        )
        
        assert signal["total_items"] == 10
        assert len(signal["top5"]) == 5
    
    @pytest.mark.unit
    def test_extract_status_signal_returns_none_when_missing(self):
        """Test 230: Status signal returns None when shortlist missing."""
        from backend.health.evidence_quality_adapter import (
            extract_evidence_failure_shortlist_signal_for_status,
        )
        
        signal = extract_evidence_failure_shortlist_signal_for_status(
            pack_manifest=None,
            evidence_data={},
        )
        
        assert signal is None
    
    @pytest.mark.unit
    def test_extract_status_signal_json_serializable(self):
        """Test 231: Status signal is JSON serializable."""
        from backend.health.evidence_quality_adapter import (
            build_first_light_failure_shelf,
            emit_cal_exp_failure_shelf,
            build_global_failure_shortlist,
            attach_global_failure_shortlist_to_evidence,
            extract_evidence_failure_shortlist_signal_for_status,
        )
        
        shelf = build_first_light_failure_shelf(
            {"trajectory_class": "STABLE"},
            {"predicted_band": "MEDIUM", "cycles_until_risk": 3}
        )
        enriched = emit_cal_exp_failure_shelf("cal_exp1", shelf)
        shortlist = build_global_failure_shortlist([enriched], max_items=10)
        evidence = attach_global_failure_shortlist_to_evidence({}, shortlist)
        
        signal = extract_evidence_failure_shortlist_signal_for_status(
            pack_manifest=None,
            evidence_data=evidence,
        )
        
        json_str = json.dumps(signal, sort_keys=True)
        assert len(json_str) > 0
        
        parsed = json.loads(json_str)
        assert parsed["total_items"] == 1
        assert len(parsed["top5"]) == 1


class TestFailureShortlistWarnings:
    """Tests for failure shortlist warnings."""
    
    @pytest.mark.unit
    def test_extract_warnings_high_band_detection(self):
        """Test 232: Warnings detect HIGH predicted_band in top5."""
        from backend.health.evidence_quality_adapter import (
            build_first_light_failure_shelf,
            emit_cal_exp_failure_shelf,
            build_global_failure_shortlist,
            attach_global_failure_shortlist_to_evidence,
            extract_evidence_failure_shortlist_warnings,
        )
        
        # Create shelves with HIGH band (unexpected in failure shortlist)
        shelf1 = build_first_light_failure_shelf(
            {"trajectory_class": "IMPROVING"},
            {"predicted_band": "HIGH", "cycles_until_risk": 5}
        )
        shelf2 = build_first_light_failure_shelf(
            {"trajectory_class": "STABLE"},
            {"predicted_band": "LOW", "cycles_until_risk": 0}
        )
        
        enriched1 = emit_cal_exp_failure_shelf("cal_exp1", shelf1)
        enriched2 = emit_cal_exp_failure_shelf("cal_exp2", shelf2)
        
        # HIGH should sort after LOW, so cal_exp2 should be first
        shortlist = build_global_failure_shortlist([enriched1, enriched2], max_items=10)
        evidence = attach_global_failure_shortlist_to_evidence({}, shortlist)
        
        warnings = extract_evidence_failure_shortlist_warnings(
            pack_manifest=None,
            evidence_data=evidence,
        )
        
        # Should warn about HIGH band item in top5
        assert len(warnings) == 1
        assert "HIGH" in warnings[0]
        assert "cal_exp1" in warnings[0]
        assert "high_band_count_in_top5" in warnings[0] or "1 item(s)" in warnings[0]
    
    @pytest.mark.unit
    def test_extract_warnings_no_warnings_when_all_low(self):
        """Test 233: No warnings when all items have LOW/MEDIUM bands."""
        from backend.health.evidence_quality_adapter import (
            build_first_light_failure_shelf,
            emit_cal_exp_failure_shelf,
            build_global_failure_shortlist,
            attach_global_failure_shortlist_to_evidence,
            extract_evidence_failure_shortlist_warnings,
        )
        
        shelves = []
        for i, band in enumerate(["LOW", "MEDIUM", "LOW"]):
            shelf = build_first_light_failure_shelf(
                {"trajectory_class": "DEGRADING"},
                {"predicted_band": band, "cycles_until_risk": i}
            )
            enriched = emit_cal_exp_failure_shelf(f"cal_exp{i+1}", shelf)
            shelves.append(enriched)
        
        shortlist = build_global_failure_shortlist(shelves, max_items=10)
        evidence = attach_global_failure_shortlist_to_evidence({}, shortlist)
        
        warnings = extract_evidence_failure_shortlist_warnings(
            pack_manifest=None,
            evidence_data=evidence,
        )
        
        assert len(warnings) == 0
    
    @pytest.mark.unit
    def test_extract_warnings_returns_empty_when_missing(self):
        """Test 234: Warnings return empty list when shortlist missing."""
        from backend.health.evidence_quality_adapter import (
            extract_evidence_failure_shortlist_warnings,
        )
        
        warnings = extract_evidence_failure_shortlist_warnings(
            pack_manifest=None,
            evidence_data={},
        )
        
        assert warnings == []
    
    @pytest.mark.unit
    def test_extract_warnings_multiple_high_bands(self):
        """Test 235: Warnings handle multiple HIGH band items (capped to 1)."""
        from backend.health.evidence_quality_adapter import (
            build_first_light_failure_shelf,
            emit_cal_exp_failure_shelf,
            build_global_failure_shortlist,
            attach_global_failure_shortlist_to_evidence,
            extract_evidence_failure_shortlist_warnings,
        )
        
        shelves = []
        for i in range(3):
            shelf = build_first_light_failure_shelf(
                {"trajectory_class": "IMPROVING"},
                {"predicted_band": "HIGH", "cycles_until_risk": 5 + i}
            )
            enriched = emit_cal_exp_failure_shelf(f"cal_exp{i+1}", shelf)
            shelves.append(enriched)
        
        shortlist = build_global_failure_shortlist(shelves, max_items=10)
        evidence = attach_global_failure_shortlist_to_evidence({}, shortlist)
        
        warnings = extract_evidence_failure_shortlist_warnings(
            pack_manifest=None,
            evidence_data=evidence,
        )
        
        # Should be capped to 1 warning
        assert len(warnings) == 1
        assert "3 item(s)" in warnings[0] or "high_band_count_in_top5" in warnings[0]
        assert "cal_exp1" in warnings[0]
        assert "cal_exp2" in warnings[0]
        assert "cal_exp3" in warnings[0]
        # Check that top_cal_ids are limited to 3
        assert "top_cal_ids" in warnings[0]


class TestEvidenceFailureShortlistGGFLAdapter:
    """Tests for evidence_failure_shortlist_for_alignment_view helper."""
    
    @pytest.mark.unit
    def test_ggfl_adapter_shape(self):
        """Test 236: GGFL adapter returns correct shape."""
        from backend.health.evidence_quality_adapter import (
            evidence_failure_shortlist_for_alignment_view,
        )
        
        signal = {
            "total_items": 3,
            "top5": [
                {"cal_id": "cal_exp1", "predicted_band": "LOW", "cycles_until_risk": 0},
                {"cal_id": "cal_exp2", "predicted_band": "MEDIUM", "cycles_until_risk": 2},
            ],
        }
        
        ggfl_signal = evidence_failure_shortlist_for_alignment_view(signal)
        
        assert ggfl_signal["signal_type"] == "SIG-EVID"
        assert ggfl_signal["status"] in ["ok", "warn"]
        assert ggfl_signal["conflict"] is False
        assert ggfl_signal["weight_hint"] == "LOW"
        assert isinstance(ggfl_signal["drivers"], list)
        assert isinstance(ggfl_signal["summary"], str)
    
    @pytest.mark.unit
    def test_ggfl_adapter_warn_status_high_band(self):
        """Test 237: GGFL adapter returns warn status when HIGH band present."""
        from backend.health.evidence_quality_adapter import (
            evidence_failure_shortlist_for_alignment_view,
        )
        
        signal = {
            "total_items": 2,
            "top5": [
                {"cal_id": "cal_exp1", "predicted_band": "HIGH", "cycles_until_risk": 5},
            ],
        }
        
        ggfl_signal = evidence_failure_shortlist_for_alignment_view(signal)
        
        assert ggfl_signal["status"] == "warn"
        assert "DRIVER_HIGH_BAND_PRESENT" in ggfl_signal["drivers"]
        assert "HIGH" in ggfl_signal["summary"]
    
    @pytest.mark.unit
    def test_ggfl_adapter_ok_status_no_high_band(self):
        """Test 238: GGFL adapter returns ok status when no HIGH band."""
        from backend.health.evidence_quality_adapter import (
            evidence_failure_shortlist_for_alignment_view,
        )
        
        signal = {
            "total_items": 2,
            "top5": [
                {"cal_id": "cal_exp1", "predicted_band": "LOW", "cycles_until_risk": 0},
            ],
        }
        
        ggfl_signal = evidence_failure_shortlist_for_alignment_view(signal)
        
        assert ggfl_signal["status"] == "ok"
        assert len(ggfl_signal["drivers"]) == 0
        # Summary should not mention HIGH band items (it says "no unexpected HIGH")
        assert "HIGH" in ggfl_signal["summary"]  # It says "no unexpected HIGH predicted_band values"
    
    @pytest.mark.unit
    def test_ggfl_adapter_handles_none_signal(self):
        """Test 239: GGFL adapter handles None signal."""
        from backend.health.evidence_quality_adapter import (
            evidence_failure_shortlist_for_alignment_view,
        )
        
        ggfl_signal = evidence_failure_shortlist_for_alignment_view(None)
        
        assert ggfl_signal["signal_type"] == "SIG-EVID"
        assert ggfl_signal["status"] == "ok"
        assert ggfl_signal["conflict"] is False
        assert ggfl_signal["weight_hint"] == "LOW"
    
    @pytest.mark.unit
    def test_ggfl_adapter_deterministic(self):
        """Test 240: GGFL adapter is deterministic."""
        from backend.health.evidence_quality_adapter import (
            evidence_failure_shortlist_for_alignment_view,
        )
        
        signal = {
            "total_items": 1,
            "top5": [
                {"cal_id": "cal_exp1", "predicted_band": "LOW", "cycles_until_risk": 0},
            ],
        }
        
        ggfl_signal1 = evidence_failure_shortlist_for_alignment_view(signal)
        ggfl_signal2 = evidence_failure_shortlist_for_alignment_view(signal)
        
        json1 = json.dumps(ggfl_signal1, sort_keys=True)
        json2 = json.dumps(ggfl_signal2, sort_keys=True)
        
        assert json1 == json2, "GGFL adapter should be deterministic"
    
    @pytest.mark.unit
    def test_ggfl_adapter_json_serializable(self):
        """Test 241: GGFL adapter output is JSON serializable."""
        from backend.health.evidence_quality_adapter import (
            evidence_failure_shortlist_for_alignment_view,
        )
        
        signal = {
            "total_items": 1,
            "top5": [
                {"cal_id": "cal_exp1", "predicted_band": "MEDIUM", "cycles_until_risk": 2},
            ],
        }
        
        ggfl_signal = evidence_failure_shortlist_for_alignment_view(signal)
        
        json_str = json.dumps(ggfl_signal, sort_keys=True)
        assert len(json_str) > 0
        
        parsed = json.loads(json_str)
        assert parsed["signal_type"] == "SIG-EVID"
        assert parsed["weight_hint"] == "LOW"
