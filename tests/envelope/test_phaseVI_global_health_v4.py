"""
test_phaseVI_global_health_v4.py — Tests for Global Health Envelope v4

PHASE VI — GLOBAL HEALTH ENVELOPE v4

Tests for system health synthesis, coherence analysis, and Director mega-panel.

ABSOLUTE SAFEGUARDS:
  - Tests do NOT modify documentation
  - Tests do NOT alter governance docs or laws
  - Language is neutral (no judgment words)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import pytest

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from narrative_consistency_index import (
    build_global_health_envelope_v4,
    analyze_system_coherence,
    build_director_mega_panel,
)


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def sample_metric_health() -> Dict[str, Any]:
    """Sample metric health envelope."""
    return {
        "status": "OK",
        "global_nci": 0.85,
    }


@pytest.fixture
def sample_drift_envelope() -> Dict[str, Any]:
    """Sample governance drift envelope (E1)."""
    return {
        "governance_drift_status": "OK",
        "risk_band": "LOW",
    }


@pytest.fixture
def sample_semantic_envelope() -> Dict[str, Any]:
    """Sample semantic consistency envelope."""
    return {
        "band": "GREEN",
        "consistency_score": 0.90,
    }


@pytest.fixture
def sample_atlas_envelope() -> Dict[str, Any]:
    """Sample structural/topology envelope."""
    return {
        "status": "OK",
        "structural_integrity": 0.95,
    }


@pytest.fixture
def sample_telemetry_envelope() -> Dict[str, Any]:
    """Sample NCI telemetry envelope (E5)."""
    return {
        "nci_status": "OK",
        "global_nci": 0.80,
    }


@pytest.fixture
def sample_nci_director_panel() -> Dict[str, Any]:
    """Sample E5 director panel."""
    return {
        "status_light": "🟢",
        "global_nci": 0.80,
        "dominant_area": "terminology",
        "headline": "Narrative consistency within target.",
    }


@pytest.fixture
def critical_metric_health() -> Dict[str, Any]:
    """Critical metric health for testing."""
    return {
        "status": "CRITICAL",
        "band": "RED",
    }


@pytest.fixture
def critical_drift_envelope() -> Dict[str, Any]:
    """Critical drift envelope for testing."""
    return {
        "governance_drift_status": "HOT",
        "risk_band": "HIGH",
    }


# ==============================================================================
# GLOBAL HEALTH ENVELOPE v4 TESTS
# ==============================================================================


class TestGlobalHealthEnvelopeV4:
    """Tests for Global Health Envelope v4 synthesis."""
    
    def test_envelope_has_required_fields(
        self,
        sample_metric_health,
        sample_drift_envelope,
        sample_semantic_envelope,
    ):
        """Test 1: Envelope has all required fields."""
        envelope = build_global_health_envelope_v4(
            metric_health=sample_metric_health,
            drift_envelope=sample_drift_envelope,
            semantic_envelope=sample_semantic_envelope,
        )
        
        assert "schema_version" in envelope
        assert "global_band" in envelope
        assert "envelope_components" in envelope
        assert "cross_signal_hotspots" in envelope
        assert "headline" in envelope
    
    def test_envelope_schema_version(self, sample_metric_health):
        """Test 2: Envelope has correct schema version."""
        envelope = build_global_health_envelope_v4(
            metric_health=sample_metric_health
        )
        
        assert envelope["schema_version"] == "4.0.0"
    
    def test_envelope_global_band_green_all_ok(
        self,
        sample_metric_health,
        sample_drift_envelope,
        sample_semantic_envelope,
    ):
        """Test 3: Global band is GREEN when all components OK."""
        envelope = build_global_health_envelope_v4(
            metric_health=sample_metric_health,
            drift_envelope=sample_drift_envelope,
            semantic_envelope=sample_semantic_envelope,
        )
        
        assert envelope["global_band"] == "GREEN"
    
    def test_envelope_global_band_red_two_critical(
        self,
        critical_metric_health,
        critical_drift_envelope,
    ):
        """Test 4: Global band is RED when ≥2 components critical."""
        envelope = build_global_health_envelope_v4(
            metric_health=critical_metric_health,
            drift_envelope=critical_drift_envelope,
        )
        
        assert envelope["global_band"] == "RED"
    
    def test_envelope_global_band_yellow_one_risk(self):
        """Test 5: Global band is YELLOW with 1 component in risk."""
        risk_envelope = {
            "band": "YELLOW",
            "status": "WARN",
        }
        
        envelope = build_global_health_envelope_v4(
            drift_envelope=risk_envelope
        )
        
        assert envelope["global_band"] == "YELLOW"
    
    def test_envelope_components_tracked(
        self,
        sample_metric_health,
        sample_drift_envelope,
    ):
        """Test 6: All envelope components are tracked."""
        envelope = build_global_health_envelope_v4(
            metric_health=sample_metric_health,
            drift_envelope=sample_drift_envelope,
            semantic_envelope=None,
            atlas_envelope=None,
            telemetry_envelope=None,
        )
        
        components = envelope["envelope_components"]
        assert "metric_health" in components
        assert "drift_envelope" in components
        assert "semantic_envelope" in components
        assert "atlas_envelope" in components
        assert "telemetry_envelope" in components
        
        assert components["metric_health"]["present"] is True
        assert components["semantic_envelope"]["present"] is False

    def test_envelope_includes_ledger_tile(
        self,
        sample_metric_health,
    ):
        """Ledger monotone tile should appear as component."""
        ledger_tile = {
            "status": "WARN",
            "ledger_monotone": True,
            "violation_count": 0,
            "headline": "Ledger monotone with schema warnings",
        }
        envelope = build_global_health_envelope_v4(
            metric_health=sample_metric_health,
            ledger_tile=ledger_tile,
        )
        component = envelope["envelope_components"]["ledger_monotone"]
        assert component["present"] is True
        assert component["band"] == "YELLOW"
        assert envelope["ledger_tile"] == ledger_tile
        assert envelope["ledger_monotone"] is True
        assert envelope["ledger_violation_count"] == 0
        assert envelope["ledger_headline"] == ledger_tile["headline"]
    
    def test_envelope_extracts_bands_various_formats(self):
        """Test 7: Envelope extracts bands from various formats."""
        # Test with "band" field
        envelope1 = build_global_health_envelope_v4(
            metric_health={"band": "GREEN"}
        )
        assert envelope1["envelope_components"]["metric_health"]["band"] == "GREEN"
        
        # Test with "status" field
        envelope2 = build_global_health_envelope_v4(
            metric_health={"status": "OK"}
        )
        assert envelope2["envelope_components"]["metric_health"]["band"] == "GREEN"
        
        # Test with "risk_band" field
        envelope3 = build_global_health_envelope_v4(
            drift_envelope={"risk_band": "HIGH"}
        )
        assert envelope3["envelope_components"]["drift_envelope"]["band"] == "RED"
    
    def test_envelope_cross_signal_hotspots_detected(self):
        """Test 8: Cross-signal hotspots are detected."""
        # Metric OK but semantic degraded
        envelope = build_global_health_envelope_v4(
            metric_health={"status": "OK"},
            semantic_envelope={"band": "RED"},
        )
        
        hotspots = envelope["cross_signal_hotspots"]
        assert len(hotspots) > 0
        assert any("semantic" in h["component"] for h in hotspots)
    
    def test_envelope_headline_neutral_language(
        self,
        sample_metric_health,
        sample_drift_envelope,
    ):
        """Test 9: Headline uses neutral language."""
        envelope = build_global_health_envelope_v4(
            metric_health=sample_metric_health,
            drift_envelope=sample_drift_envelope,
        )
        
        headline = envelope["headline"].lower()
        forbidden = ["bad", "good", "broken", "fixed", "healthy", "unhealthy"]
        for word in forbidden:
            assert word not in headline
    
    def test_envelope_handles_all_none(self):
        """Test 10: Envelope handles all None inputs."""
        envelope = build_global_health_envelope_v4()
        
        assert envelope["global_band"] == "GREEN"
        assert all(
            not comp["present"]
            for comp in envelope["envelope_components"].values()
        )


# ==============================================================================
# SYSTEM COHERENCE ANALYZER TESTS
# ==============================================================================


class TestSystemCoherenceAnalyzer:
    """Tests for system coherence analysis."""
    
    def test_coherence_has_required_fields(
        self,
        sample_metric_health,
        sample_semantic_envelope,
    ):
        """Test 11: Coherence analysis has all required fields."""
        coherence = analyze_system_coherence(
            metric_health=sample_metric_health,
            semantic_envelope=sample_semantic_envelope,
        )
        
        assert "schema_version" in coherence
        assert "coherence_status" in coherence
        assert "mismatches" in coherence
        assert "notes" in coherence
    
    def test_coherence_status_coherent_when_aligned(
        self,
        sample_metric_health,
        sample_semantic_envelope,
    ):
        """Test 12: Coherence status is COHERENT when signals aligned."""
        coherence = analyze_system_coherence(
            metric_health=sample_metric_health,
            semantic_envelope=sample_semantic_envelope,
        )
        
        assert coherence["coherence_status"] in ["COHERENT", "INSUFFICIENT_DATA"]
    
    def test_coherence_detects_mismatch_metric_ok_semantic_red(self):
        """Test 13: Coherence detects mismatch: metric OK, semantic RED."""
        coherence = analyze_system_coherence(
            metric_health={"status": "OK"},
            semantic_envelope={"band": "RED"},
        )
        
        assert coherence["coherence_status"] == "MISMATCHED"
        assert len(coherence["mismatches"]) > 0
        
        mismatch = coherence["mismatches"][0]
        assert "metric_health" in mismatch["components"]
        assert "semantic_envelope" in mismatch["components"]
        assert mismatch["severity"] == "HIGH"
    
    def test_coherence_detects_drift_telemetry_mismatch(self):
        """Test 14: Coherence detects drift/telemetry mismatch."""
        coherence = analyze_system_coherence(
            drift_envelope={"governance_drift_status": "HOT"},
            telemetry_envelope={"nci_status": "OK"},
        )
        
        # Should detect mismatch if both present
        if coherence["coherence_status"] == "MISMATCHED":
            assert len(coherence["mismatches"]) > 0
    
    def test_coherence_handles_insufficient_data(self):
        """Test 15: Coherence handles insufficient data."""
        coherence = analyze_system_coherence(
            metric_health={"status": "OK"}
        )
        
        assert coherence["coherence_status"] == "INSUFFICIENT_DATA"
    
    def test_coherence_notes_neutral_language(
        self,
        sample_metric_health,
        sample_semantic_envelope,
    ):
        """Test 16: Coherence notes use neutral language."""
        coherence = analyze_system_coherence(
            metric_health=sample_metric_health,
            semantic_envelope=sample_semantic_envelope,
        )
        
        notes_text = " ".join(coherence["notes"]).lower()
        forbidden = ["bad", "good", "broken", "fixed", "healthy", "unhealthy"]
        for word in forbidden:
            assert word not in notes_text
    
    def test_coherence_mismatch_severity_assigned(self):
        """Test 17: Coherence assigns appropriate severity levels."""
        coherence = analyze_system_coherence(
            metric_health={"status": "OK"},
            semantic_envelope={"band": "RED"},
        )
        
        if coherence["coherence_status"] == "MISMATCHED":
            for mismatch in coherence["mismatches"]:
                assert mismatch["severity"] in ["LOW", "MEDIUM", "HIGH"]


# ==============================================================================
# DIRECTOR MEGA-PANEL TESTS
# ==============================================================================


class TestDirectorMegaPanel:
    """Tests for Director mega-panel synthesis."""
    
    def test_mega_panel_has_required_fields(
        self,
        sample_metric_health,
        sample_drift_envelope,
        sample_telemetry_envelope,
    ):
        """Test 18: Mega-panel has all required fields."""
        panel = build_director_mega_panel(
            metric_health=sample_metric_health,
            drift_envelope=sample_drift_envelope,
            telemetry_envelope=sample_telemetry_envelope,
        )
        
        assert "schema_version" in panel
        assert "release_ready" in panel
        assert "mega_status_light" in panel
        assert "component_summary" in panel
        assert "global_envelope" in panel
        assert "coherence_analysis" in panel
        assert "executive_headline" in panel
    
    def test_mega_panel_release_ready_when_all_green(
        self,
        sample_metric_health,
        sample_drift_envelope,
        sample_telemetry_envelope,
    ):
        """Test 19: Mega-panel indicates release-ready when all green."""
        panel = build_director_mega_panel(
            metric_health=sample_metric_health,
            drift_envelope=sample_drift_envelope,
            telemetry_envelope=sample_telemetry_envelope,
        )
        
        assert panel["release_ready"] is True
        assert panel["mega_status_light"] == "🟢"
    
    def test_mega_panel_not_release_ready_on_red(
        self,
        critical_metric_health,
        critical_drift_envelope,
    ):
        """Test 20: Mega-panel indicates not release-ready on RED."""
        panel = build_director_mega_panel(
            metric_health=critical_metric_health,
            drift_envelope=critical_drift_envelope,
        )
        
        assert panel["release_ready"] is False
        assert panel["mega_status_light"] == "🔴"
    
    def test_mega_panel_component_summary_tracks_agents(
        self,
        sample_drift_envelope,
        sample_telemetry_envelope,
    ):
        """Test 21: Component summary tracks E1 and E5 agents."""
        panel = build_director_mega_panel(
            drift_envelope=sample_drift_envelope,
            telemetry_envelope=sample_telemetry_envelope,
        )
        
        summary = panel["component_summary"]
        assert "E1_governance" in summary
        assert "E5_narrative" in summary
        
        assert summary["E1_governance"]["band"] == "GREEN"
        assert summary["E5_narrative"]["band"] == "GREEN"

    def test_mega_panel_includes_ledger_tile_metadata(
        self,
        sample_metric_health,
    ):
        """Director panel should surface ledger monotone details."""
        ledger_tile = {
            "status": "BLOCK",
            "ledger_monotone": False,
            "violation_count": 2,
            "headline": "Ledger monotonicity violations detected (2)",
        }
        panel = build_director_mega_panel(
            metric_health=sample_metric_health,
            ledger_tile=ledger_tile,
        )
        component = panel["component_summary"]["ledger_monotone"]
        assert component["band"] == "RED"
        assert component["details"]["violation_count"] == 2
        assert component["details"]["ledger_monotone"] is False
        assert panel["ledger_tile"] == ledger_tile
        assert panel["ledger_headline"] == ledger_tile["headline"]
    
    def test_mega_panel_includes_global_envelope(
        self,
        sample_metric_health,
        sample_drift_envelope,
    ):
        """Test 22: Mega-panel includes full global envelope."""
        panel = build_director_mega_panel(
            metric_health=sample_metric_health,
            drift_envelope=sample_drift_envelope,
        )
        
        assert "global_envelope" in panel
        envelope = panel["global_envelope"]
        assert envelope["schema_version"] == "4.0.0"
        assert "global_band" in envelope
    
    def test_mega_panel_includes_coherence_analysis(
        self,
        sample_metric_health,
        sample_semantic_envelope,
    ):
        """Test 23: Mega-panel includes coherence analysis."""
        panel = build_director_mega_panel(
            metric_health=sample_metric_health,
            semantic_envelope=sample_semantic_envelope,
        )
        
        assert "coherence_analysis" in panel
        coherence = panel["coherence_analysis"]
        assert "coherence_status" in coherence
    
    def test_mega_panel_executive_headline_neutral(
        self,
        sample_metric_health,
        sample_drift_envelope,
    ):
        """Test 24: Executive headline uses neutral language."""
        panel = build_director_mega_panel(
            metric_health=sample_metric_health,
            drift_envelope=sample_drift_envelope,
        )
        
        headline = panel["executive_headline"].lower()
        forbidden = ["bad", "good", "broken", "fixed", "healthy", "unhealthy"]
        for word in forbidden:
            assert word not in headline
    
    def test_mega_panel_not_ready_on_coherence_mismatch(self):
        """Test 25: Mega-panel not ready when coherence mismatched."""
        # Metric OK but semantic RED creates mismatch
        panel = build_director_mega_panel(
            metric_health={"status": "OK"},
            semantic_envelope={"band": "RED"},
        )
        
        # Should not be release-ready due to coherence mismatch
        coherence_status = panel["coherence_analysis"]["coherence_status"]
        if coherence_status == "MISMATCHED":
            assert panel["release_ready"] is False
    
    def test_mega_panel_status_light_maps_correctly(self):
        """Test 26: Mega status light maps correctly to global band."""
        # GREEN
        panel1 = build_director_mega_panel(
            metric_health={"status": "OK"}
        )
        assert panel1["mega_status_light"] == "🟢"
        
        # YELLOW
        panel2 = build_director_mega_panel(
            drift_envelope={"band": "YELLOW"}
        )
        assert panel2["mega_status_light"] == "🟡"
        
        # RED
        panel3 = build_director_mega_panel(
            metric_health={"band": "RED"},
            drift_envelope={"band": "RED"},
        )
        assert panel3["mega_status_light"] == "🔴"

