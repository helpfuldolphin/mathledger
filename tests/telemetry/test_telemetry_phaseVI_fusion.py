"""
PHASE II — Telemetry Phase VI Fusion Tests

Descriptive only — not admissible as uplift evidence.

This module provides tests for Phase VI telemetry fusion functionality:
- Telemetry–Topology–Semantic Fusion Tile
- Telemetry-Driven Uplift Phase Gate
- Director Tile v2

Author: metrics-engineer-4 (Agent D4)
"""

from __future__ import annotations

import unittest
from typing import Any, Dict

import pytest

from experiments.behavioral_telemetry_viz import (
    build_telemetry_director_tile_v2,
    build_telemetry_drift_envelope,
    build_telemetry_driven_uplift_phase_gate,
    build_telemetry_topology_semantic_fusion,
    summarize_telemetry_for_global_health,
    summarize_telemetry_for_uplift_safety,
)


# -----------------------------------------------------------------------------
# Test: Telemetry–Topology–Semantic Fusion Tile
# -----------------------------------------------------------------------------

class TestTelemetryTopologySemanticFusion(unittest.TestCase):
    """Tests for the telemetry–topology–semantic fusion tile."""

    def test_low_fusion_band_when_all_stable(self):
        """Fusion band should be LOW when all systems are stable."""
        telemetry_envelope = {
            'drift_band': 'LOW',
            'plots_with_repeated_drift': [],
            'neutral_notes': [],
        }
        topology_struct = {
            'events': [],
            'warnings': [],
            'stability_score': 1.0,
        }
        semantic_struct = {
            'changes': [],
            'coherence_score': 1.0,
            'drift_indicators': [],
        }
        
        fusion = build_telemetry_topology_semantic_fusion(
            telemetry_envelope, topology_struct, semantic_struct
        )
        
        self.assertEqual(fusion['fusion_band'], 'LOW')
        self.assertLess(fusion['fusion_risk_score'], 0.3)

    def test_medium_fusion_band_when_moderate_risk(self):
        """Fusion band should be MEDIUM when moderate risk detected."""
        telemetry_envelope = {
            'drift_band': 'MEDIUM',
            'plots_with_repeated_drift': [],
            'neutral_notes': [],
        }
        topology_struct = {
            'events': [],
            'warnings': ['warning1'],
            'stability_score': 0.6,
        }
        semantic_struct = {
            'changes': [],
            'coherence_score': 0.7,
            'drift_indicators': [],
        }
        
        fusion = build_telemetry_topology_semantic_fusion(
            telemetry_envelope, topology_struct, semantic_struct
        )
        
        self.assertEqual(fusion['fusion_band'], 'MEDIUM')
        self.assertGreaterEqual(fusion['fusion_risk_score'], 0.3)
        self.assertLess(fusion['fusion_risk_score'], 0.7)

    def test_high_fusion_band_when_high_risk(self):
        """Fusion band should be HIGH when high risk detected."""
        telemetry_envelope = {
            'drift_band': 'HIGH',
            'plots_with_repeated_drift': ['plot1'],
            'neutral_notes': [],
        }
        topology_struct = {
            'events': [],
            'warnings': ['warning1', 'warning2'],
            'stability_score': 0.3,
        }
        semantic_struct = {
            'changes': ['change1'],
            'coherence_score': 0.4,
            'drift_indicators': ['indicator1'],
        }
        
        fusion = build_telemetry_topology_semantic_fusion(
            telemetry_envelope, topology_struct, semantic_struct
        )
        
        self.assertEqual(fusion['fusion_band'], 'HIGH')
        self.assertGreaterEqual(fusion['fusion_risk_score'], 0.7)

    def test_fusion_risk_score_calculation(self):
        """Fusion risk score should be weighted average of component risks."""
        telemetry_envelope = {'drift_band': 'MEDIUM'}  # 0.5 risk
        topology_struct = {'stability_score': 0.5}  # 0.5 risk
        semantic_struct = {'coherence_score': 0.5}  # 0.5 risk
        
        fusion = build_telemetry_topology_semantic_fusion(
            telemetry_envelope, topology_struct, semantic_struct
        )
        
        # Expected: 0.4 * 0.5 + 0.3 * 0.5 + 0.3 * 0.5 = 0.5
        self.assertAlmostEqual(fusion['fusion_risk_score'], 0.5, places=2)

    def test_incoherence_vectors_detected(self):
        """Fusion should detect incoherence vectors."""
        telemetry_envelope = {
            'drift_band': 'HIGH',
            'plots_with_repeated_drift': [],
            'neutral_notes': [],
        }
        topology_struct = {
            'events': [],
            'warnings': ['warning1'],
            'stability_score': 0.4,
        }
        semantic_struct = {
            'changes': ['change1'],
            'coherence_score': 0.3,
            'drift_indicators': [],
        }
        
        fusion = build_telemetry_topology_semantic_fusion(
            telemetry_envelope, topology_struct, semantic_struct
        )
        
        self.assertGreater(len(fusion['incoherence_vectors']), 0)
        self.assertIn('telemetry:high_drift_band', fusion['incoherence_vectors'])

    def test_cross_domain_incoherence_detected(self):
        """Fusion should detect cross-domain incoherence."""
        telemetry_envelope = {
            'drift_band': 'MEDIUM',
            'plots_with_repeated_drift': [],
            'neutral_notes': [],
        }
        topology_struct = {
            'events': [],
            'warnings': [],
            'stability_score': 0.6,  # < 0.7
        }
        semantic_struct = {
            'changes': [],
            'coherence_score': 0.6,  # < 0.7
            'drift_indicators': [],
        }
        
        fusion = build_telemetry_topology_semantic_fusion(
            telemetry_envelope, topology_struct, semantic_struct
        )
        
        # Should detect cross-domain mismatches
        cross_domain_vectors = [
            v for v in fusion['incoherence_vectors']
            if v.startswith('cross_domain:')
        ]
        self.assertGreater(len(cross_domain_vectors), 0)

    def test_fusion_contains_neutral_notes(self):
        """Fusion should contain neutral notes."""
        telemetry_envelope = {
            'drift_band': 'MEDIUM',
            'plots_with_repeated_drift': [],
            'neutral_notes': [],
        }
        topology_struct = {
            'events': [],
            'warnings': [],
            'stability_score': 0.8,
        }
        semantic_struct = {
            'changes': [],
            'coherence_score': 0.9,
            'drift_indicators': [],
        }
        
        fusion = build_telemetry_topology_semantic_fusion(
            telemetry_envelope, topology_struct, semantic_struct
        )
        
        self.assertIn('neutral_notes', fusion)
        self.assertGreater(len(fusion['neutral_notes']), 0)
        self.assertIn('Fusion risk score', fusion['neutral_notes'][0])

    def test_fusion_risk_score_clamped(self):
        """Fusion risk score should be clamped to [0.0, 1.0]."""
        # Test with extreme values
        telemetry_envelope = {'drift_band': 'HIGH'}  # 0.9
        topology_struct = {'stability_score': -0.5}  # Would be 1.5 risk
        semantic_struct = {'coherence_score': 2.0}  # Would be -1.0 risk
        
        fusion = build_telemetry_topology_semantic_fusion(
            telemetry_envelope, topology_struct, semantic_struct
        )
        
        self.assertGreaterEqual(fusion['fusion_risk_score'], 0.0)
        self.assertLessEqual(fusion['fusion_risk_score'], 1.0)


# -----------------------------------------------------------------------------
# Test: Telemetry-Driven Uplift Phase Gate
# -----------------------------------------------------------------------------

class TestTelemetryDrivenUpliftPhaseGate(unittest.TestCase):
    """Tests for the telemetry-driven uplift phase gate."""

    def test_ok_gate_when_all_ok(self):
        """Gate status should be OK when all systems OK."""
        fusion_tile = {
            'fusion_band': 'LOW',
            'fusion_risk_score': 0.2,
            'incoherence_vectors': [],
            'neutral_notes': [],
        }
        uplift_safety = {
            'telemetry_ok_for_uplift': True,
            'status': 'OK',
            'blocking_reasons': [],
            'advisory_notes': [],
        }
        coupling_view = {
            'coupling_status': 'ALIGNED',
        }
        
        gate = build_telemetry_driven_uplift_phase_gate(
            fusion_tile, uplift_safety, coupling_view
        )
        
        self.assertEqual(gate['uplift_gate_status'], 'OK')
        self.assertEqual(len(gate['drivers']), 0)
        self.assertEqual(len(gate['recommended_hold_slices']), 0)

    def test_block_gate_when_high_fusion(self):
        """Gate status should be BLOCK when fusion band is HIGH."""
        fusion_tile = {
            'fusion_band': 'HIGH',
            'fusion_risk_score': 0.8,
            'incoherence_vectors': ['vector1', 'vector2'],
            'neutral_notes': [],
        }
        uplift_safety = {
            'telemetry_ok_for_uplift': True,
            'status': 'OK',
            'blocking_reasons': [],
            'advisory_notes': [],
        }
        coupling_view = {
            'coupling_status': 'ALIGNED',
        }
        
        gate = build_telemetry_driven_uplift_phase_gate(
            fusion_tile, uplift_safety, coupling_view
        )
        
        self.assertEqual(gate['uplift_gate_status'], 'BLOCK')
        self.assertGreater(len(gate['drivers']), 0)
        self.assertIn('HIGH', str(gate['drivers']))

    def test_block_gate_when_uplift_safety_block(self):
        """Gate status should be BLOCK when uplift safety is BLOCK."""
        fusion_tile = {
            'fusion_band': 'LOW',
            'fusion_risk_score': 0.1,
            'incoherence_vectors': [],
            'neutral_notes': [],
        }
        uplift_safety = {
            'telemetry_ok_for_uplift': False,
            'status': 'BLOCK',
            'blocking_reasons': ['Reason 1', 'Reason 2'],
            'advisory_notes': [],
        }
        coupling_view = {
            'coupling_status': 'ALIGNED',
        }
        
        gate = build_telemetry_driven_uplift_phase_gate(
            fusion_tile, uplift_safety, coupling_view
        )
        
        self.assertEqual(gate['uplift_gate_status'], 'BLOCK')
        self.assertIn('Reason 1', gate['drivers'])

    def test_block_gate_when_coupling_misaligned(self):
        """Gate status should be BLOCK when coupling is MISALIGNED."""
        fusion_tile = {
            'fusion_band': 'LOW',
            'fusion_risk_score': 0.1,
            'incoherence_vectors': [],
            'neutral_notes': [],
        }
        uplift_safety = {
            'telemetry_ok_for_uplift': True,
            'status': 'OK',
            'blocking_reasons': [],
            'advisory_notes': [],
        }
        coupling_view = {
            'coupling_status': 'MISALIGNED',
        }
        
        gate = build_telemetry_driven_uplift_phase_gate(
            fusion_tile, uplift_safety, coupling_view
        )
        
        self.assertEqual(gate['uplift_gate_status'], 'BLOCK')
        self.assertIn('MISALIGNED', str(gate['drivers']))

    def test_attention_gate_when_medium_fusion(self):
        """Gate status should be ATTENTION when fusion band is MEDIUM."""
        fusion_tile = {
            'fusion_band': 'MEDIUM',
            'fusion_risk_score': 0.5,
            'incoherence_vectors': ['vector1'],
            'neutral_notes': [],
        }
        uplift_safety = {
            'telemetry_ok_for_uplift': True,
            'status': 'OK',
            'blocking_reasons': [],
            'advisory_notes': [],
        }
        coupling_view = {
            'coupling_status': 'ALIGNED',
        }
        
        gate = build_telemetry_driven_uplift_phase_gate(
            fusion_tile, uplift_safety, coupling_view
        )
        
        self.assertEqual(gate['uplift_gate_status'], 'ATTENTION')
        self.assertIn('MEDIUM', str(gate['drivers']))

    def test_attention_gate_when_uplift_safety_attention(self):
        """Gate status should be ATTENTION when uplift safety is ATTENTION."""
        fusion_tile = {
            'fusion_band': 'LOW',
            'fusion_risk_score': 0.1,
            'incoherence_vectors': [],
            'neutral_notes': [],
        }
        uplift_safety = {
            'telemetry_ok_for_uplift': True,
            'status': 'ATTENTION',
            'blocking_reasons': [],
            'advisory_notes': ['Advisory note 1'],
        }
        coupling_view = {
            'coupling_status': 'ALIGNED',
        }
        
        gate = build_telemetry_driven_uplift_phase_gate(
            fusion_tile, uplift_safety, coupling_view
        )
        
        self.assertEqual(gate['uplift_gate_status'], 'ATTENTION')
        self.assertIn('Advisory note 1', gate['drivers'])

    def test_attention_gate_when_coupling_partial(self):
        """Gate status should be ATTENTION when coupling is PARTIAL."""
        fusion_tile = {
            'fusion_band': 'LOW',
            'fusion_risk_score': 0.1,
            'incoherence_vectors': [],
            'neutral_notes': [],
        }
        uplift_safety = {
            'telemetry_ok_for_uplift': True,
            'status': 'OK',
            'blocking_reasons': [],
            'advisory_notes': [],
        }
        coupling_view = {
            'coupling_status': 'PARTIAL',
        }
        
        gate = build_telemetry_driven_uplift_phase_gate(
            fusion_tile, uplift_safety, coupling_view
        )
        
        self.assertEqual(gate['uplift_gate_status'], 'ATTENTION')
        self.assertIn('PARTIAL', str(gate['drivers']))

    def test_recommended_hold_slices_when_high_fusion(self):
        """Gate should recommend holding slices when fusion is HIGH."""
        fusion_tile = {
            'fusion_band': 'HIGH',
            'fusion_risk_score': 0.8,
            'incoherence_vectors': ['telemetry:high_drift_band'],
            'neutral_notes': [],
        }
        uplift_safety = {
            'telemetry_ok_for_uplift': False,
            'status': 'BLOCK',
            'blocking_reasons': [],
            'advisory_notes': [],
        }
        coupling_view = {
            'coupling_status': 'ALIGNED',
        }
        
        gate = build_telemetry_driven_uplift_phase_gate(
            fusion_tile, uplift_safety, coupling_view
        )
        
        self.assertGreater(len(gate['recommended_hold_slices']), 0)

    def test_gate_contains_headline(self):
        """Gate should contain a neutral headline."""
        fusion_tile = {
            'fusion_band': 'LOW',
            'fusion_risk_score': 0.1,
            'incoherence_vectors': [],
            'neutral_notes': [],
        }
        uplift_safety = {
            'telemetry_ok_for_uplift': True,
            'status': 'OK',
            'blocking_reasons': [],
            'advisory_notes': [],
        }
        coupling_view = {
            'coupling_status': 'ALIGNED',
        }
        
        gate = build_telemetry_driven_uplift_phase_gate(
            fusion_tile, uplift_safety, coupling_view
        )
        
        self.assertIn('headline', gate)
        self.assertIsInstance(gate['headline'], str)
        self.assertGreater(len(gate['headline']), 0)

    def test_gate_without_coupling_view(self):
        """Gate should work without coupling view."""
        fusion_tile = {
            'fusion_band': 'LOW',
            'fusion_risk_score': 0.1,
            'incoherence_vectors': [],
            'neutral_notes': [],
        }
        uplift_safety = {
            'telemetry_ok_for_uplift': True,
            'status': 'OK',
            'blocking_reasons': [],
            'advisory_notes': [],
        }
        
        gate = build_telemetry_driven_uplift_phase_gate(
            fusion_tile, uplift_safety, coupling_view=None
        )
        
        self.assertEqual(gate['uplift_gate_status'], 'OK')


# -----------------------------------------------------------------------------
# Test: Director Tile v2
# -----------------------------------------------------------------------------

class TestTelemetryDirectorTileV2(unittest.TestCase):
    """Tests for the telemetry director tile v2."""

    def test_green_light_when_all_ok(self):
        """Status light should be GREEN when all systems OK."""
        fusion_tile = {
            'fusion_band': 'LOW',
            'fusion_risk_score': 0.1,
            'incoherence_vectors': [],
            'neutral_notes': [],
        }
        uplift_gate = {
            'uplift_gate_status': 'OK',
            'drivers': [],
            'recommended_hold_slices': [],
            'headline': 'OK',
        }
        coupling_view = {
            'coupling_status': 'ALIGNED',
        }
        telemetry_health = {
            'telemetry_ok': True,
            'status': 'OK',
        }
        
        tile = build_telemetry_director_tile_v2(
            fusion_tile, uplift_gate, coupling_view, telemetry_health
        )
        
        self.assertEqual(tile['status_light'], 'GREEN')
        self.assertTrue(tile['telemetry_ok'])

    def test_red_light_when_high_fusion(self):
        """Status light should be RED when fusion band is HIGH."""
        fusion_tile = {
            'fusion_band': 'HIGH',
            'fusion_risk_score': 0.8,
            'incoherence_vectors': [],
            'neutral_notes': [],
        }
        uplift_gate = {
            'uplift_gate_status': 'OK',
            'drivers': [],
            'recommended_hold_slices': [],
            'headline': 'OK',
        }
        coupling_view = {
            'coupling_status': 'ALIGNED',
        }
        telemetry_health = {
            'telemetry_ok': True,
            'status': 'OK',
        }
        
        tile = build_telemetry_director_tile_v2(
            fusion_tile, uplift_gate, coupling_view, telemetry_health
        )
        
        self.assertEqual(tile['status_light'], 'RED')

    def test_red_light_when_gate_blocked(self):
        """Status light should be RED when uplift gate is BLOCK."""
        fusion_tile = {
            'fusion_band': 'LOW',
            'fusion_risk_score': 0.1,
            'incoherence_vectors': [],
            'neutral_notes': [],
        }
        uplift_gate = {
            'uplift_gate_status': 'BLOCK',
            'drivers': ['Blocking reason'],
            'recommended_hold_slices': [],
            'headline': 'Blocked',
        }
        coupling_view = {
            'coupling_status': 'ALIGNED',
        }
        telemetry_health = {
            'telemetry_ok': True,
            'status': 'OK',
        }
        
        tile = build_telemetry_director_tile_v2(
            fusion_tile, uplift_gate, coupling_view, telemetry_health
        )
        
        self.assertEqual(tile['status_light'], 'RED')

    def test_yellow_light_when_medium_fusion(self):
        """Status light should be YELLOW when fusion band is MEDIUM."""
        fusion_tile = {
            'fusion_band': 'MEDIUM',
            'fusion_risk_score': 0.5,
            'incoherence_vectors': [],
            'neutral_notes': [],
        }
        uplift_gate = {
            'uplift_gate_status': 'OK',
            'drivers': [],
            'recommended_hold_slices': [],
            'headline': 'OK',
        }
        coupling_view = {
            'coupling_status': 'ALIGNED',
        }
        telemetry_health = {
            'telemetry_ok': True,
            'status': 'OK',
        }
        
        tile = build_telemetry_director_tile_v2(
            fusion_tile, uplift_gate, coupling_view, telemetry_health
        )
        
        self.assertEqual(tile['status_light'], 'YELLOW')

    def test_yellow_light_when_gate_attention(self):
        """Status light should be YELLOW when uplift gate is ATTENTION."""
        fusion_tile = {
            'fusion_band': 'LOW',
            'fusion_risk_score': 0.1,
            'incoherence_vectors': [],
            'neutral_notes': [],
        }
        uplift_gate = {
            'uplift_gate_status': 'ATTENTION',
            'drivers': ['Attention reason'],
            'recommended_hold_slices': [],
            'headline': 'Attention',
        }
        coupling_view = {
            'coupling_status': 'ALIGNED',
        }
        telemetry_health = {
            'telemetry_ok': True,
            'status': 'OK',
        }
        
        tile = build_telemetry_director_tile_v2(
            fusion_tile, uplift_gate, coupling_view, telemetry_health
        )
        
        self.assertEqual(tile['status_light'], 'YELLOW')

    def test_yellow_light_when_coupling_partial(self):
        """Status light should be YELLOW when coupling is PARTIAL."""
        fusion_tile = {
            'fusion_band': 'LOW',
            'fusion_risk_score': 0.1,
            'incoherence_vectors': [],
            'neutral_notes': [],
        }
        uplift_gate = {
            'uplift_gate_status': 'OK',
            'drivers': [],
            'recommended_hold_slices': [],
            'headline': 'OK',
        }
        coupling_view = {
            'coupling_status': 'PARTIAL',
        }
        telemetry_health = {
            'telemetry_ok': True,
            'status': 'OK',
        }
        
        tile = build_telemetry_director_tile_v2(
            fusion_tile, uplift_gate, coupling_view, telemetry_health
        )
        
        self.assertEqual(tile['status_light'], 'YELLOW')

    def test_tile_contains_all_fields(self):
        """Tile should contain all required fields."""
        fusion_tile = {
            'fusion_band': 'LOW',
            'fusion_risk_score': 0.1,
            'incoherence_vectors': [],
            'neutral_notes': [],
        }
        uplift_gate = {
            'uplift_gate_status': 'OK',
            'drivers': [],
            'recommended_hold_slices': [],
            'headline': 'OK',
        }
        coupling_view = {
            'coupling_status': 'ALIGNED',
        }
        telemetry_health = {
            'telemetry_ok': True,
            'status': 'OK',
        }
        
        tile = build_telemetry_director_tile_v2(
            fusion_tile, uplift_gate, coupling_view, telemetry_health
        )
        
        self.assertIn('status_light', tile)
        self.assertIn('fusion_band', tile)
        self.assertIn('uplift_gate_status', tile)
        self.assertIn('structural_coupling_state', tile)
        self.assertIn('telemetry_ok', tile)
        self.assertIn('headline', tile)

    def test_tile_headline_reflects_status(self):
        """Tile headline should reflect the current status."""
        fusion_tile = {
            'fusion_band': 'HIGH',
            'fusion_risk_score': 0.8,
            'incoherence_vectors': [],
            'neutral_notes': [],
        }
        uplift_gate = {
            'uplift_gate_status': 'BLOCK',
            'drivers': [],
            'recommended_hold_slices': [],
            'headline': 'Blocked',
        }
        coupling_view = {
            'coupling_status': 'ALIGNED',
        }
        telemetry_health = {
            'telemetry_ok': False,
            'status': 'BLOCK',
        }
        
        tile = build_telemetry_director_tile_v2(
            fusion_tile, uplift_gate, coupling_view, telemetry_health
        )
        
        self.assertIn('RED', tile['headline'])
        self.assertIn('blocked', tile['headline'].lower())


# -----------------------------------------------------------------------------
# pytest markers
# -----------------------------------------------------------------------------

pytestmark = pytest.mark.unit


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

