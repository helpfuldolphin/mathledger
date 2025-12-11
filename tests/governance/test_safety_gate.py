"""
Tests for Safety Gate Module (Phase X Neural Link).

Validates:
- Gate tile is JSON-safe
- Evidence attachment is deterministic
- No mutation of inputs
- Status_light matches PASS/WARN/BLOCK mapping
"""

import json
import pytest
from backend.governance.safety_gate import (
    SafetyEnvelope,
    SafetyGateStatus,
    SafetyGateDecision,
    build_safety_gate_summary_for_first_light,
    build_safety_gate_tile_for_global_health,
    attach_safety_gate_to_evidence,
    build_global_health_surface,
)


class TestSafetyEnvelope:
    """Test SafetyEnvelope data structure."""
    
    def test_get_reasons_deterministic_ordering(self):
        """Reasons should be sorted alphabetically for determinism."""
        envelope = SafetyEnvelope(
            final_status=SafetyGateStatus.WARN,
            total_decisions=3,
            blocked_cycles=0,
            advisory_cycles=3,
            decisions=[
                SafetyGateDecision(cycle=1, status=SafetyGateStatus.WARN, reason="zebra"),
                SafetyGateDecision(cycle=2, status=SafetyGateStatus.WARN, reason="alpha"),
                SafetyGateDecision(cycle=3, status=SafetyGateStatus.WARN, reason="beta"),
            ]
        )
        
        reasons = envelope.get_reasons()
        assert reasons == ["alpha", "beta", "zebra"]
    
    def test_get_reasons_deduplication(self):
        """Duplicate reasons should be deduplicated."""
        envelope = SafetyEnvelope(
            final_status=SafetyGateStatus.WARN,
            total_decisions=3,
            blocked_cycles=0,
            advisory_cycles=3,
            decisions=[
                SafetyGateDecision(cycle=1, status=SafetyGateStatus.WARN, reason="duplicate"),
                SafetyGateDecision(cycle=2, status=SafetyGateStatus.WARN, reason="unique"),
                SafetyGateDecision(cycle=3, status=SafetyGateStatus.WARN, reason="duplicate"),
            ]
        )
        
        reasons = envelope.get_reasons()
        assert reasons == ["duplicate", "unique"]
    
    def test_get_reasons_with_limit(self):
        """Should respect limit parameter."""
        envelope = SafetyEnvelope(
            final_status=SafetyGateStatus.WARN,
            total_decisions=5,
            blocked_cycles=0,
            advisory_cycles=5,
            decisions=[
                SafetyGateDecision(cycle=i, status=SafetyGateStatus.WARN, reason=f"reason_{i}")
                for i in range(5)
            ]
        )
        
        reasons = envelope.get_reasons(limit=3)
        assert len(reasons) == 3
        assert reasons == ["reason_0", "reason_1", "reason_2"]
    
    def test_to_dict_structure(self):
        """to_dict should produce expected structure."""
        envelope = SafetyEnvelope(
            final_status=SafetyGateStatus.PASS,
            total_decisions=10,
            blocked_cycles=0,
            advisory_cycles=0,
        )
        
        result = envelope.to_dict()
        
        assert result["final_status"] == "PASS"
        assert result["total_decisions"] == 10
        assert result["blocked_cycles"] == 0
        assert result["advisory_cycles"] == 0
        assert "reasons" in result


class TestFirstLightSummary:
    """Test First Light summary generation."""
    
    def test_json_safe(self):
        """Summary should be JSON-serializable."""
        envelope = SafetyEnvelope(
            final_status=SafetyGateStatus.PASS,
            total_decisions=100,
            blocked_cycles=0,
            advisory_cycles=2,
            decisions=[
                SafetyGateDecision(cycle=1, status=SafetyGateStatus.WARN, reason="test"),
            ]
        )
        
        summary = build_safety_gate_summary_for_first_light(envelope)
        
        # Should not raise
        json_str = json.dumps(summary)
        assert json_str is not None
        
        # Should round-trip
        loaded = json.loads(json_str)
        assert loaded["final_status"] == "PASS"
    
    def test_contains_required_fields(self):
        """Summary should contain all required fields."""
        envelope = SafetyEnvelope(
            final_status=SafetyGateStatus.BLOCK,
            total_decisions=50,
            blocked_cycles=5,
            advisory_cycles=10,
        )
        
        summary = build_safety_gate_summary_for_first_light(envelope)
        
        assert "final_status" in summary
        assert "total_decisions" in summary
        assert "blocked_cycles" in summary
        assert "advisory_cycles" in summary
        assert "reasons" in summary
    
    def test_deterministic_reasons(self):
        """Reasons in summary should be deterministically ordered."""
        decisions = [
            SafetyGateDecision(cycle=1, status=SafetyGateStatus.WARN, reason="c"),
            SafetyGateDecision(cycle=2, status=SafetyGateStatus.WARN, reason="a"),
            SafetyGateDecision(cycle=3, status=SafetyGateStatus.WARN, reason="b"),
        ]
        
        envelope = SafetyEnvelope(
            final_status=SafetyGateStatus.WARN,
            total_decisions=3,
            blocked_cycles=0,
            advisory_cycles=3,
            decisions=decisions,
        )
        
        summary = build_safety_gate_summary_for_first_light(envelope)
        assert summary["reasons"] == ["a", "b", "c"]


class TestGlobalHealthTile:
    """Test global health tile generation."""
    
    def test_json_safe(self):
        """Tile should be JSON-serializable."""
        envelope = SafetyEnvelope(
            final_status=SafetyGateStatus.PASS,
            total_decisions=100,
            blocked_cycles=0,
            advisory_cycles=0,
        )
        
        tile = build_safety_gate_tile_for_global_health(envelope)
        
        # Should not raise
        json_str = json.dumps(tile)
        assert json_str is not None
    
    def test_status_light_mapping_pass(self):
        """PASS status should map to GREEN light."""
        envelope = SafetyEnvelope(
            final_status=SafetyGateStatus.PASS,
            total_decisions=100,
            blocked_cycles=0,
            advisory_cycles=0,
        )
        
        tile = build_safety_gate_tile_for_global_health(envelope)
        assert tile["status_light"] == "GREEN"
    
    def test_status_light_mapping_warn(self):
        """WARN status should map to YELLOW light."""
        envelope = SafetyEnvelope(
            final_status=SafetyGateStatus.WARN,
            total_decisions=100,
            blocked_cycles=0,
            advisory_cycles=10,
        )
        
        tile = build_safety_gate_tile_for_global_health(envelope)
        assert tile["status_light"] == "YELLOW"
    
    def test_status_light_mapping_block(self):
        """BLOCK status should map to RED light."""
        envelope = SafetyEnvelope(
            final_status=SafetyGateStatus.BLOCK,
            total_decisions=100,
            blocked_cycles=20,
            advisory_cycles=10,
        )
        
        tile = build_safety_gate_tile_for_global_health(envelope)
        assert tile["status_light"] == "RED"
    
    def test_blocked_fraction_calculation(self):
        """Blocked fraction should be correctly calculated."""
        envelope = SafetyEnvelope(
            final_status=SafetyGateStatus.BLOCK,
            total_decisions=100,
            blocked_cycles=25,
            advisory_cycles=0,
        )
        
        tile = build_safety_gate_tile_for_global_health(envelope)
        assert tile["blocked_fraction"] == 0.25
    
    def test_blocked_fraction_zero_decisions(self):
        """Blocked fraction should be 0 when no decisions."""
        envelope = SafetyEnvelope(
            final_status=SafetyGateStatus.PASS,
            total_decisions=0,
            blocked_cycles=0,
            advisory_cycles=0,
        )
        
        tile = build_safety_gate_tile_for_global_health(envelope)
        assert tile["blocked_fraction"] == 0.0
    
    def test_headline_generation(self):
        """Headline should be neutral and informative."""
        envelope = SafetyEnvelope(
            final_status=SafetyGateStatus.BLOCK,
            total_decisions=100,
            blocked_cycles=10,
            advisory_cycles=5,
        )
        
        tile = build_safety_gate_tile_for_global_health(envelope)
        assert "Safety gate" in tile["headline"]
        assert "BLOCK" in tile["headline"]
        assert "10 blocked" in tile["headline"]


class TestEvidenceAttachment:
    """Test evidence pack attachment."""
    
    def test_no_mutation_of_input(self):
        """Should not mutate input evidence dictionary."""
        original_evidence = {
            "version": "1.0.0",
            "governance": {
                "existing_field": "value"
            }
        }
        
        envelope = SafetyEnvelope(
            final_status=SafetyGateStatus.PASS,
            total_decisions=100,
            blocked_cycles=0,
            advisory_cycles=0,
        )
        
        # Make a copy to compare
        import copy
        evidence_before = copy.deepcopy(original_evidence)
        
        # Call function
        result = attach_safety_gate_to_evidence(original_evidence, envelope)
        
        # Original should be unchanged
        assert original_evidence == evidence_before
        
        # Result should have safety gate
        assert "safety_gate" in result["governance"]
    
    def test_creates_governance_section_if_missing(self):
        """Should create governance section if it doesn't exist."""
        evidence = {
            "version": "1.0.0",
        }
        
        envelope = SafetyEnvelope(
            final_status=SafetyGateStatus.PASS,
            total_decisions=100,
            blocked_cycles=0,
            advisory_cycles=0,
        )
        
        result = attach_safety_gate_to_evidence(evidence, envelope)
        
        assert "governance" in result
        assert "safety_gate" in result["governance"]
    
    def test_limits_reasons_to_top_3(self):
        """Should include only top 3 reasons in evidence."""
        decisions = [
            SafetyGateDecision(cycle=i, status=SafetyGateStatus.WARN, reason=f"reason_{i}")
            for i in range(10)
        ]
        
        envelope = SafetyEnvelope(
            final_status=SafetyGateStatus.WARN,
            total_decisions=10,
            blocked_cycles=0,
            advisory_cycles=10,
            decisions=decisions,
        )
        
        result = attach_safety_gate_to_evidence({}, envelope)
        
        reasons = result["governance"]["safety_gate"]["reasons"]
        assert len(reasons) == 3
    
    def test_deterministic_evidence_attachment(self):
        """Evidence attachment should be deterministic."""
        decisions = [
            SafetyGateDecision(cycle=1, status=SafetyGateStatus.WARN, reason="z"),
            SafetyGateDecision(cycle=2, status=SafetyGateStatus.WARN, reason="a"),
            SafetyGateDecision(cycle=3, status=SafetyGateStatus.WARN, reason="m"),
        ]
        
        envelope = SafetyEnvelope(
            final_status=SafetyGateStatus.WARN,
            total_decisions=3,
            blocked_cycles=0,
            advisory_cycles=3,
            decisions=decisions,
        )
        
        result1 = attach_safety_gate_to_evidence({}, envelope)
        result2 = attach_safety_gate_to_evidence({}, envelope)
        
        # Should produce identical results
        assert result1 == result2
        
        # Reasons should be sorted
        reasons = result1["governance"]["safety_gate"]["reasons"]
        assert reasons == ["a", "m", "z"]


class TestGlobalHealthSurface:
    """Test global health surface integration."""
    
    def test_includes_safety_gate_tile(self):
        """Should include safety gate tile when envelope provided."""
        existing_tiles = {
            "other_tile": {"status": "ok"}
        }
        
        envelope = SafetyEnvelope(
            final_status=SafetyGateStatus.PASS,
            total_decisions=100,
            blocked_cycles=0,
            advisory_cycles=0,
        )
        
        health = build_global_health_surface(existing_tiles, safety_envelope=envelope)
        
        assert "safety_gate" in health["tiles"]
        assert "other_tile" in health["tiles"]
    
    def test_omits_safety_gate_when_no_envelope(self):
        """Should not include safety gate tile when envelope not provided."""
        existing_tiles = {
            "other_tile": {"status": "ok"}
        }
        
        health = build_global_health_surface(existing_tiles)
        
        assert "safety_gate" not in health["tiles"]
        assert "other_tile" in health["tiles"]
    
    def test_shadow_mode_preserves_other_tiles(self):
        """Safety gate should not affect other tiles (shadow mode)."""
        existing_tiles = {
            "tile_a": {"value": 1},
            "tile_b": {"value": 2},
        }
        
        envelope = SafetyEnvelope(
            final_status=SafetyGateStatus.BLOCK,  # Even in BLOCK
            total_decisions=100,
            blocked_cycles=50,
            advisory_cycles=0,
        )
        
        health = build_global_health_surface(existing_tiles, safety_envelope=envelope)
        
        # Other tiles unchanged
        assert health["tiles"]["tile_a"] == {"value": 1}
        assert health["tiles"]["tile_b"] == {"value": 2}
        
        # Safety gate added
        assert "safety_gate" in health["tiles"]
