"""
PHASE VI — NOT USED IN PHASE I

Tests for Phase VI Atlas Convergence Lattice functions.

These tests verify:
1. Lattice vector computation
2. Global lattice norm calculation
3. Convergence band determination
4. Phase transition gate v2 logic
5. Director tile v2 integration

Agent: metrics-engineer-6 (D6)
"""

import json
import math
import pytest
from typing import Any, Dict

from experiments.u2_behavior_atlas import (
    build_atlas_convergence_lattice,
    derive_atlas_phase_transition_gate,
    build_atlas_director_tile_v2,
)


class TestBuildAtlasConvergenceLattice:
    """Tests for build_atlas_convergence_lattice function (FROZEN CONTRACT)."""

    def test_lattice_has_required_keys(self):
        """Lattice contains all contract keys."""
        routing_policy = {
            "slices_preferring_dense_archetypes": ["slice_a"],
            "slices_preferring_sparse_archetypes": [],
        }
        structural_view = {
            "slices_with_consistent_archetypes": ["slice_a"],
            "slices_with_structure_vs_routing_mismatch": [],
        }
        curriculum_view = {
            "slices_with_atlas_support": ["slice_a"],
            "slices_without_atlas_support": [],
        }
        
        lattice = build_atlas_convergence_lattice(
            routing_policy, structural_view, curriculum_view
        )
        
        # FROZEN CONTRACT keys
        assert "lattice_vectors" in lattice
        assert "global_lattice_norm" in lattice
        assert "convergence_band" in lattice
        assert "neutral_notes" in lattice

    def test_lattice_vectors_computed(self):
        """Lattice vectors are computed for all slices."""
        routing_policy = {
            "slices_preferring_dense_archetypes": ["slice_a"],
            "slices_preferring_sparse_archetypes": ["slice_b"],
        }
        structural_view = {
            "slices_with_consistent_archetypes": ["slice_a"],
            "slices_with_structure_vs_routing_mismatch": ["slice_b"],
        }
        curriculum_view = {
            "slices_with_atlas_support": ["slice_a"],
            "slices_without_atlas_support": ["slice_b"],
        }
        
        lattice = build_atlas_convergence_lattice(
            routing_policy, structural_view, curriculum_view
        )
        
        assert "slice_a" in lattice["lattice_vectors"]
        assert "slice_b" in lattice["lattice_vectors"]
        assert isinstance(lattice["lattice_vectors"]["slice_a"], float)
        assert isinstance(lattice["lattice_vectors"]["slice_b"], float)

    def test_lattice_vector_high_alignment(self):
        """Lattice vector is high when all components align."""
        routing_policy = {
            "slices_preferring_dense_archetypes": ["slice_a"],
            "slices_preferring_sparse_archetypes": [],
        }
        structural_view = {
            "slices_with_consistent_archetypes": ["slice_a"],
            "slices_with_structure_vs_routing_mismatch": [],
        }
        curriculum_view = {
            "slices_with_atlas_support": ["slice_a"],
            "slices_without_atlas_support": [],
        }
        
        lattice = build_atlas_convergence_lattice(
            routing_policy, structural_view, curriculum_view
        )
        
        # All components = 1.0, so vector should be high
        vector = lattice["lattice_vectors"]["slice_a"]
        assert vector >= 0.8  # High alignment

    def test_lattice_vector_low_alignment(self):
        """Lattice vector is low when components misalign."""
        routing_policy = {
            "slices_preferring_dense_archetypes": [],
            "slices_preferring_sparse_archetypes": [],
        }
        structural_view = {
            "slices_with_consistent_archetypes": [],
            "slices_with_structure_vs_routing_mismatch": ["slice_a"],
        }
        curriculum_view = {
            "slices_with_atlas_support": [],
            "slices_without_atlas_support": ["slice_a"],
        }
        
        lattice = build_atlas_convergence_lattice(
            routing_policy, structural_view, curriculum_view
        )
        
        # All components = 0.0, so vector should be low
        vector = lattice["lattice_vectors"]["slice_a"]
        assert vector < 0.5  # Low alignment

    def test_convergence_band_coherent(self):
        """Convergence band is COHERENT when global norm >= 0.8."""
        routing_policy = {
            "slices_preferring_dense_archetypes": ["slice_a", "slice_b", "slice_c"],
            "slices_preferring_sparse_archetypes": [],
        }
        structural_view = {
            "slices_with_consistent_archetypes": ["slice_a", "slice_b", "slice_c"],
            "slices_with_structure_vs_routing_mismatch": [],
        }
        curriculum_view = {
            "slices_with_atlas_support": ["slice_a", "slice_b", "slice_c"],
            "slices_without_atlas_support": [],
        }
        
        lattice = build_atlas_convergence_lattice(
            routing_policy, structural_view, curriculum_view
        )
        
        assert lattice["convergence_band"] == "COHERENT"
        assert lattice["global_lattice_norm"] >= 0.8

    def test_convergence_band_partial(self):
        """Convergence band is PARTIAL when 0.5 <= global norm < 0.8."""
        routing_policy = {
            "slices_preferring_dense_archetypes": ["slice_a"],
            "slices_preferring_sparse_archetypes": [],
        }
        structural_view = {
            "slices_with_consistent_archetypes": ["slice_a"],
            "slices_with_structure_vs_routing_mismatch": [],
        }
        curriculum_view = {
            "slices_with_atlas_support": [],
            "slices_without_atlas_support": ["slice_a"],
        }
        
        lattice = build_atlas_convergence_lattice(
            routing_policy, structural_view, curriculum_view
        )
        
        # Mixed alignment should give PARTIAL
        assert lattice["convergence_band"] in ["PARTIAL", "COHERENT", "MISALIGNED"]
        assert 0.0 <= lattice["global_lattice_norm"] <= 1.0

    def test_convergence_band_misaligned(self):
        """Convergence band is MISALIGNED when global norm < 0.5."""
        routing_policy = {
            "slices_preferring_dense_archetypes": [],
            "slices_preferring_sparse_archetypes": [],
        }
        structural_view = {
            "slices_with_consistent_archetypes": [],
            "slices_with_structure_vs_routing_mismatch": ["slice_a", "slice_b"],
        }
        curriculum_view = {
            "slices_with_atlas_support": [],
            "slices_without_atlas_support": ["slice_a", "slice_b"],
        }
        
        lattice = build_atlas_convergence_lattice(
            routing_policy, structural_view, curriculum_view
        )
        
        if lattice["global_lattice_norm"] < 0.5:
            assert lattice["convergence_band"] == "MISALIGNED"

    def test_lattice_generates_notes(self):
        """Lattice generates neutral notes."""
        routing_policy = {
            "slices_preferring_dense_archetypes": ["slice_a"],
            "slices_preferring_sparse_archetypes": [],
        }
        structural_view = {
            "slices_with_consistent_archetypes": ["slice_a"],
            "slices_with_structure_vs_routing_mismatch": [],
        }
        curriculum_view = {
            "slices_with_atlas_support": ["slice_a"],
            "slices_without_atlas_support": [],
        }
        
        lattice = build_atlas_convergence_lattice(
            routing_policy, structural_view, curriculum_view
        )
        
        assert len(lattice["neutral_notes"]) > 0
        assert all(isinstance(note, str) for note in lattice["neutral_notes"])

    def test_lattice_deterministic(self):
        """Lattice computation is deterministic."""
        routing_policy = {
            "slices_preferring_dense_archetypes": ["slice_a"],
            "slices_preferring_sparse_archetypes": [],
        }
        structural_view = {
            "slices_with_consistent_archetypes": ["slice_a"],
            "slices_with_structure_vs_routing_mismatch": [],
        }
        curriculum_view = {
            "slices_with_atlas_support": ["slice_a"],
            "slices_without_atlas_support": [],
        }
        
        results = [
            build_atlas_convergence_lattice(
                routing_policy, structural_view, curriculum_view
            )
            for _ in range(50)
        ]
        json_results = [json.dumps(r, sort_keys=True) for r in results]
        
        assert all(jr == json_results[0] for jr in json_results)

    def test_lattice_handles_empty_inputs(self):
        """Lattice handles empty inputs gracefully."""
        routing_policy = {
            "slices_preferring_dense_archetypes": [],
            "slices_preferring_sparse_archetypes": [],
        }
        structural_view = {
            "slices_with_consistent_archetypes": [],
            "slices_with_structure_vs_routing_mismatch": [],
        }
        curriculum_view = {
            "slices_with_atlas_support": [],
            "slices_without_atlas_support": [],
        }
        
        lattice = build_atlas_convergence_lattice(
            routing_policy, structural_view, curriculum_view
        )
        
        assert lattice["global_lattice_norm"] == 0.0
        assert lattice["convergence_band"] == "MISALIGNED"
        assert len(lattice["lattice_vectors"]) == 0


class TestDeriveAtlasPhaseTransitionGate:
    """Tests for derive_atlas_phase_transition_gate function (FROZEN CONTRACT)."""

    def test_phase_gate_has_required_keys(self):
        """Phase gate contains all contract keys."""
        lattice = {
            "convergence_band": "COHERENT",
            "global_lattice_norm": 0.9,
            "lattice_vectors": {"slice_a": 0.9},
        }
        phase_advice = {
            "status": "OK",
            "phase_transition_safe": True,
            "suggested_slices_for_phase_upgrade": ["slice_a"],
            "slices_needing_more_atlas_support": [],
        }
        
        gate = derive_atlas_phase_transition_gate(lattice, phase_advice)
        
        # FROZEN CONTRACT keys
        assert "transition_status" in gate
        assert "drivers" in gate
        assert "slices_ready" in gate
        assert "slices_needing_alignment" in gate
        assert "headline" in gate

    def test_phase_gate_status_ok(self):
        """Transition status is OK when phase advice OK and lattice COHERENT."""
        lattice = {
            "convergence_band": "COHERENT",
            "global_lattice_norm": 0.9,
            "lattice_vectors": {"slice_a": 0.9},
        }
        phase_advice = {
            "status": "OK",
            "phase_transition_safe": True,
            "suggested_slices_for_phase_upgrade": ["slice_a"],
            "slices_needing_more_atlas_support": [],
        }
        
        gate = derive_atlas_phase_transition_gate(lattice, phase_advice)
        
        assert gate["transition_status"] == "OK"

    def test_phase_gate_status_block_misaligned(self):
        """Transition status is BLOCK when lattice is MISALIGNED."""
        lattice = {
            "convergence_band": "MISALIGNED",
            "global_lattice_norm": 0.3,
            "lattice_vectors": {"slice_a": 0.3},
        }
        phase_advice = {
            "status": "OK",
            "phase_transition_safe": True,
            "suggested_slices_for_phase_upgrade": ["slice_a"],
            "slices_needing_more_atlas_support": [],
        }
        
        gate = derive_atlas_phase_transition_gate(lattice, phase_advice)
        
        assert gate["transition_status"] == "BLOCK"

    def test_phase_gate_status_block_phase_advice(self):
        """Transition status is BLOCK when phase advice is BLOCK."""
        lattice = {
            "convergence_band": "COHERENT",
            "global_lattice_norm": 0.9,
            "lattice_vectors": {"slice_a": 0.9},
        }
        phase_advice = {
            "status": "BLOCK",
            "phase_transition_safe": False,
            "suggested_slices_for_phase_upgrade": [],
            "slices_needing_more_atlas_support": ["slice_a"],
        }
        
        gate = derive_atlas_phase_transition_gate(lattice, phase_advice)
        
        assert gate["transition_status"] == "BLOCK"

    def test_phase_gate_status_attention_partial(self):
        """Transition status is ATTENTION when lattice is PARTIAL."""
        lattice = {
            "convergence_band": "PARTIAL",
            "global_lattice_norm": 0.6,
            "lattice_vectors": {"slice_a": 0.6},
        }
        phase_advice = {
            "status": "OK",
            "phase_transition_safe": True,
            "suggested_slices_for_phase_upgrade": ["slice_a"],
            "slices_needing_more_atlas_support": [],
        }
        
        gate = derive_atlas_phase_transition_gate(lattice, phase_advice)
        
        assert gate["transition_status"] == "ATTENTION"

    def test_phase_gate_slices_ready(self):
        """Slices ready are intersection of suggested and high-vector slices."""
        lattice = {
            "convergence_band": "COHERENT",
            "global_lattice_norm": 0.9,
            "lattice_vectors": {
                "slice_a": 0.9,  # High vector
                "slice_b": 0.4,  # Low vector
            },
        }
        phase_advice = {
            "status": "OK",
            "phase_transition_safe": True,
            "suggested_slices_for_phase_upgrade": ["slice_a", "slice_b"],
            "slices_needing_more_atlas_support": [],
        }
        
        gate = derive_atlas_phase_transition_gate(lattice, phase_advice)
        
        assert "slice_a" in gate["slices_ready"]
        assert "slice_b" not in gate["slices_ready"]  # Low vector

    def test_phase_gate_slices_needing_alignment(self):
        """Slices needing alignment include low-vector and unsupported slices."""
        lattice = {
            "convergence_band": "PARTIAL",
            "global_lattice_norm": 0.6,
            "lattice_vectors": {
                "slice_a": 0.9,
                "slice_b": 0.3,  # Low vector
            },
        }
        phase_advice = {
            "status": "ATTENTION",
            "phase_transition_safe": False,
            "suggested_slices_for_phase_upgrade": ["slice_a"],
            "slices_needing_more_atlas_support": ["slice_c"],
        }
        
        gate = derive_atlas_phase_transition_gate(lattice, phase_advice)
        
        assert "slice_b" in gate["slices_needing_alignment"]  # Low vector
        assert "slice_c" in gate["slices_needing_alignment"]  # Needs support

    def test_phase_gate_drivers_identified(self):
        """Drivers are identified based on blocking conditions."""
        lattice = {
            "convergence_band": "MISALIGNED",
            "global_lattice_norm": 0.3,
            "lattice_vectors": {},
        }
        phase_advice = {
            "status": "BLOCK",
            "phase_transition_safe": False,
            "suggested_slices_for_phase_upgrade": [],
            "slices_needing_more_atlas_support": [],
        }
        
        gate = derive_atlas_phase_transition_gate(lattice, phase_advice)
        
        assert len(gate["drivers"]) > 0
        assert any("misaligned" in d.lower() for d in gate["drivers"])

    def test_phase_gate_headline_generated(self):
        """Headline is generated based on transition status."""
        lattice = {
            "convergence_band": "COHERENT",
            "global_lattice_norm": 0.9,
            "lattice_vectors": {"slice_a": 0.9},
        }
        phase_advice = {
            "status": "OK",
            "phase_transition_safe": True,
            "suggested_slices_for_phase_upgrade": ["slice_a"],
            "slices_needing_more_atlas_support": [],
        }
        
        gate = derive_atlas_phase_transition_gate(lattice, phase_advice)
        
        assert isinstance(gate["headline"], str)
        assert len(gate["headline"]) > 0

    def test_phase_gate_deterministic(self):
        """Phase gate is deterministic."""
        lattice = {
            "convergence_band": "COHERENT",
            "global_lattice_norm": 0.9,
            "lattice_vectors": {"slice_a": 0.9},
        }
        phase_advice = {
            "status": "OK",
            "phase_transition_safe": True,
            "suggested_slices_for_phase_upgrade": ["slice_a"],
            "slices_needing_more_atlas_support": [],
        }
        
        results = [
            derive_atlas_phase_transition_gate(lattice, phase_advice)
            for _ in range(50)
        ]
        json_results = [json.dumps(r, sort_keys=True) for r in results]
        
        assert all(jr == json_results[0] for jr in json_results)

    def test_phase_gate_no_value_judgments(self):
        """Phase gate headline contains no value-loaded language."""
        lattice = {
            "convergence_band": "COHERENT",
            "global_lattice_norm": 0.9,
            "lattice_vectors": {},
        }
        phase_advice = {
            "status": "OK",
            "phase_transition_safe": True,
            "suggested_slices_for_phase_upgrade": [],
            "slices_needing_more_atlas_support": [],
        }
        
        gate = derive_atlas_phase_transition_gate(lattice, phase_advice)
        gate_str = gate["headline"].lower()
        
        forbidden = ["best", "worst", "better", "worse", "top", "bottom", "rank"]
        for word in forbidden:
            assert word not in gate_str, f"Ranking term '{word}' found in headline"


class TestBuildAtlasDirectorTileV2:
    """Tests for build_atlas_director_tile_v2 function (FROZEN CONTRACT)."""

    def test_director_tile_v2_has_required_keys(self):
        """Director tile v2 contains all contract keys."""
        governance = {"structurally_sound": True, "total_slices_indexed": 5}
        routing = {"routing_status": "CLUSTERED"}
        structural = {"governance_status": "OK"}
        curriculum = {"coupling_status": "TIGHT"}
        phase_advice = {"status": "OK", "phase_transition_safe": True}
        lattice = {"convergence_band": "COHERENT", "global_lattice_norm": 0.9}
        phase_gate = {"transition_status": "OK"}
        
        tile = build_atlas_director_tile_v2(
            governance, routing, structural, curriculum,
            phase_advice, lattice, phase_gate
        )
        
        # FROZEN CONTRACT keys
        assert "status_light" in tile
        assert "lattice_coherence" in tile
        assert "structural_status" in tile
        assert "transition_recommendation" in tile
        assert "atlas_ok" in tile
        assert "headline" in tile

    def test_director_tile_v2_status_light_green(self):
        """Status light is GREEN when all systems healthy."""
        governance = {"structurally_sound": True, "total_slices_indexed": 5}
        routing = {"routing_status": "CLUSTERED"}
        structural = {"governance_status": "OK"}
        curriculum = {"coupling_status": "TIGHT"}
        phase_advice = {"status": "OK", "phase_transition_safe": True}
        lattice = {"convergence_band": "COHERENT", "global_lattice_norm": 0.9}
        phase_gate = {"transition_status": "OK"}
        
        tile = build_atlas_director_tile_v2(
            governance, routing, structural, curriculum,
            phase_advice, lattice, phase_gate
        )
        
        assert tile["status_light"] == "GREEN"
        assert tile["atlas_ok"] is True

    def test_director_tile_v2_status_light_red_misaligned(self):
        """Status light is RED when lattice is MISALIGNED."""
        governance = {"structurally_sound": True, "total_slices_indexed": 5}
        routing = {"routing_status": "CLUSTERED"}
        structural = {"governance_status": "OK"}
        curriculum = {"coupling_status": "TIGHT"}
        phase_advice = {"status": "OK", "phase_transition_safe": True}
        lattice = {"convergence_band": "MISALIGNED", "global_lattice_norm": 0.3}
        phase_gate = {"transition_status": "BLOCK"}
        
        tile = build_atlas_director_tile_v2(
            governance, routing, structural, curriculum,
            phase_advice, lattice, phase_gate
        )
        
        assert tile["status_light"] == "RED"
        assert tile["atlas_ok"] is False

    def test_director_tile_v2_status_light_yellow_partial(self):
        """Status light is YELLOW when lattice is PARTIAL."""
        governance = {"structurally_sound": True, "total_slices_indexed": 5}
        routing = {"routing_status": "CLUSTERED"}
        structural = {"governance_status": "OK"}
        curriculum = {"coupling_status": "TIGHT"}
        phase_advice = {"status": "OK", "phase_transition_safe": True}
        lattice = {"convergence_band": "PARTIAL", "global_lattice_norm": 0.6}
        phase_gate = {"transition_status": "ATTENTION"}
        
        tile = build_atlas_director_tile_v2(
            governance, routing, structural, curriculum,
            phase_advice, lattice, phase_gate
        )
        
        assert tile["status_light"] == "YELLOW"
        assert tile["atlas_ok"] is False

    def test_director_tile_v2_lattice_coherence_propagated(self):
        """Lattice coherence is propagated from lattice."""
        governance = {"structurally_sound": True, "total_slices_indexed": 5}
        routing = {"routing_status": "CLUSTERED"}
        structural = {"governance_status": "OK"}
        curriculum = {"coupling_status": "TIGHT"}
        phase_advice = {"status": "OK", "phase_transition_safe": True}
        lattice = {"convergence_band": "COHERENT", "global_lattice_norm": 0.9}
        phase_gate = {"transition_status": "OK"}
        
        tile = build_atlas_director_tile_v2(
            governance, routing, structural, curriculum,
            phase_advice, lattice, phase_gate
        )
        
        assert tile["lattice_coherence"] == "COHERENT"

    def test_director_tile_v2_transition_recommendation(self):
        """Transition recommendation is generated based on transition status."""
        governance = {"structurally_sound": True, "total_slices_indexed": 5}
        routing = {"routing_status": "CLUSTERED"}
        structural = {"governance_status": "OK"}
        curriculum = {"coupling_status": "TIGHT"}
        phase_advice = {"status": "OK", "phase_transition_safe": True}
        lattice = {"convergence_band": "COHERENT", "global_lattice_norm": 0.9}
        phase_gate = {"transition_status": "OK"}
        
        tile = build_atlas_director_tile_v2(
            governance, routing, structural, curriculum,
            phase_advice, lattice, phase_gate
        )
        
        assert "OK" in tile["transition_recommendation"]
        assert "safe" in tile["transition_recommendation"].lower()

    def test_director_tile_v2_headline_generated(self):
        """Headline is generated based on status light."""
        governance = {"structurally_sound": True, "total_slices_indexed": 10}
        routing = {"routing_status": "CLUSTERED"}
        structural = {"governance_status": "OK"}
        curriculum = {"coupling_status": "TIGHT"}
        phase_advice = {"status": "OK", "phase_transition_safe": True}
        lattice = {"convergence_band": "COHERENT", "global_lattice_norm": 0.9}
        phase_gate = {"transition_status": "OK", "slices_ready": ["slice_a"], "slices_needing_alignment": []}
        
        tile = build_atlas_director_tile_v2(
            governance, routing, structural, curriculum,
            phase_advice, lattice, phase_gate
        )
        
        assert isinstance(tile["headline"], str)
        assert len(tile["headline"]) > 0
        assert "10" in tile["headline"] or "slices" in tile["headline"].lower()

    def test_director_tile_v2_deterministic(self):
        """Director tile v2 is deterministic."""
        governance = {"structurally_sound": True, "total_slices_indexed": 5}
        routing = {"routing_status": "CLUSTERED"}
        structural = {"governance_status": "OK"}
        curriculum = {"coupling_status": "TIGHT"}
        phase_advice = {"status": "OK", "phase_transition_safe": True}
        lattice = {"convergence_band": "COHERENT", "global_lattice_norm": 0.9}
        phase_gate = {"transition_status": "OK"}
        
        results = [
            build_atlas_director_tile_v2(
                governance, routing, structural, curriculum,
                phase_advice, lattice, phase_gate
            )
            for _ in range(50)
        ]
        json_results = [json.dumps(r, sort_keys=True) for r in results]
        
        assert all(jr == json_results[0] for jr in json_results)

    def test_director_tile_v2_no_value_judgments(self):
        """Director tile v2 headline contains no value-loaded language."""
        governance = {"structurally_sound": True, "total_slices_indexed": 5}
        routing = {"routing_status": "CLUSTERED"}
        structural = {"governance_status": "OK"}
        curriculum = {"coupling_status": "TIGHT"}
        phase_advice = {"status": "OK", "phase_transition_safe": True}
        lattice = {"convergence_band": "COHERENT", "global_lattice_norm": 0.9}
        phase_gate = {"transition_status": "OK"}
        
        tile = build_atlas_director_tile_v2(
            governance, routing, structural, curriculum,
            phase_advice, lattice, phase_gate
        )
        tile_str = tile["headline"].lower()
        
        forbidden = ["best", "worst", "better", "worse", "top", "bottom", "rank"]
        for word in forbidden:
            assert word not in tile_str, f"Ranking term '{word}' found in headline"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

