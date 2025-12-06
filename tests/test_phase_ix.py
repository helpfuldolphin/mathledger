"""
Tests for Harmony Protocol v1.1 and Celestial Dossier v2

Validates consensus, provenance tracking, and cryptographic integrity.
"""

import pytest
import json
from backend.ledger.consensus.harmony_v1_1 import HarmonyProtocol, NodeAttestation
from backend.ledger.consensus.celestial_dossier_v2 import CelestialDossier, ProvenanceNode
from backend.crypto.hashing import sha256_hex, DOMAIN_ROOT


class TestHarmonyProtocol:
    """Test suite for Harmony Protocol v1.1"""
    
    def test_node_registration(self):
        """Test node registration with initial weights."""
        harmony = HarmonyProtocol()
        
        harmony.register_node("node_1", initial_weight=1.0)
        harmony.register_node("node_2", initial_weight=0.8)
        
        assert "node_1" in harmony.node_trust_scores
        assert "node_2" in harmony.node_trust_scores
        assert harmony.node_trust_scores["node_1"] == 1.0
        assert harmony.node_trust_scores["node_2"] == 0.8
    
    def test_invalid_weight_registration(self):
        """Test that invalid weights are rejected."""
        harmony = HarmonyProtocol()
        
        with pytest.raises(ValueError):
            harmony.register_node("node_1", initial_weight=1.5)
        
        with pytest.raises(ValueError):
            harmony.register_node("node_2", initial_weight=-0.1)
    
    def test_attestation_submission(self):
        """Test submitting node attestations."""
        harmony = HarmonyProtocol()
        harmony.register_node("node_1")
        
        proposed_value = sha256_hex("test_state", domain=DOMAIN_ROOT)
        attestation = harmony.submit_attestation("node_1", proposed_value)
        
        assert attestation.node_id == "node_1"
        assert attestation.proposed_value == proposed_value
        assert attestation.round_number == 0
        assert len(attestation.signature) == 64
    
    def test_consensus_convergence(self):
        """Test consensus convergence with honest majority."""
        harmony = HarmonyProtocol()
        
        # Register 10 nodes
        for i in range(10):
            harmony.register_node(f"node_{i}")
        
        # 8 nodes propose same value (80% honest)
        canonical_value = sha256_hex("canonical", domain=DOMAIN_ROOT)
        byzantine_value = sha256_hex("byzantine", domain=DOMAIN_ROOT)
        
        attestations = []
        for i in range(10):
            if i < 8:
                attestations.append(harmony.submit_attestation(f"node_{i}", canonical_value))
            else:
                attestations.append(harmony.submit_attestation(f"node_{i}", byzantine_value))
        
        converged, value = harmony.evaluate_consensus(attestations)
        
        assert converged is True
        assert value == canonical_value
    
    def test_consensus_no_convergence(self):
        """Test that consensus fails without 2/3 majority."""
        harmony = HarmonyProtocol()
        
        # Register 9 nodes
        for i in range(9):
            harmony.register_node(f"node_{i}")
        
        # Split votes evenly (no 2/3 majority)
        value_a = sha256_hex("value_a", domain=DOMAIN_ROOT)
        value_b = sha256_hex("value_b", domain=DOMAIN_ROOT)
        value_c = sha256_hex("value_c", domain=DOMAIN_ROOT)
        
        attestations = []
        for i in range(9):
            if i < 3:
                attestations.append(harmony.submit_attestation(f"node_{i}", value_a))
            elif i < 6:
                attestations.append(harmony.submit_attestation(f"node_{i}", value_b))
            else:
                attestations.append(harmony.submit_attestation(f"node_{i}", value_c))
        
        converged, value = harmony.evaluate_consensus(attestations)
        
        assert converged is False
        assert value is None
    
    def test_trust_score_updates(self):
        """Test adaptive trust score updates."""
        harmony = HarmonyProtocol()
        
        # Start with non-max weight so we can see increase
        harmony.register_node("honest", initial_weight=0.9)
        harmony.register_node("byzantine", initial_weight=0.9)
        
        canonical_value = sha256_hex("canonical", domain=DOMAIN_ROOT)
        byzantine_value = sha256_hex("byzantine", domain=DOMAIN_ROOT)
        
        # Submit attestations
        attestations = [
            harmony.submit_attestation("honest", canonical_value),
            harmony.submit_attestation("byzantine", byzantine_value)
        ]
        
        initial_honest_weight = harmony.node_trust_scores["honest"]
        initial_byzantine_weight = harmony.node_trust_scores["byzantine"]
        
        # Run consensus (honest wins with 50% but we'll force it for test)
        harmony._update_trust_scores(attestations, canonical_value)
        
        # Honest weight should increase (0.9 * 1.05 = 0.945)
        assert harmony.node_trust_scores["honest"] > initial_honest_weight
        # Byzantine weight should decrease (0.9 * 0.95 = 0.855)
        assert harmony.node_trust_scores["byzantine"] < initial_byzantine_weight
    
    def test_harmony_root_computation(self):
        """Test deterministic Harmony root computation."""
        harmony = HarmonyProtocol()
        
        harmony.register_node("node_1")
        
        # Run a consensus round
        attestations = [
            harmony.submit_attestation("node_1", sha256_hex("state", domain=DOMAIN_ROOT))
        ]
        harmony.run_consensus_round(attestations)
        
        root1 = harmony.compute_harmony_root()
        root2 = harmony.compute_harmony_root()
        
        # Should be deterministic
        assert root1 == root2
        assert len(root1) == 64  # SHA-256 hex
    
    def test_safety_property(self):
        """Test safety property verification."""
        harmony = HarmonyProtocol()
        
        harmony.register_node("node_1")
        
        canonical_value = sha256_hex("canonical", domain=DOMAIN_ROOT)
        
        # Run multiple rounds with same value
        for _ in range(3):
            attestations = [harmony.submit_attestation("node_1", canonical_value)]
            harmony.run_consensus_round(attestations)
        
        # All rounds should converge to same value
        assert harmony.verify_safety_property() is True
    
    def test_liveness_property(self):
        """Test liveness property verification."""
        harmony = HarmonyProtocol()
        
        # Register 10 nodes
        for i in range(10):
            harmony.register_node(f"node_{i}")
        
        canonical_value = sha256_hex("canonical", domain=DOMAIN_ROOT)
        
        # Run round with high participation (all nodes)
        attestations = [
            harmony.submit_attestation(f"node_{i}", canonical_value)
            for i in range(10)
        ]
        harmony.run_consensus_round(attestations)
        
        # Should satisfy liveness with 100% participation
        assert harmony.verify_liveness_property(min_participation=0.67) is True


class TestCelestialDossier:
    """Test suite for Celestial Dossier v2"""
    
    def test_add_provenance_node(self):
        """Test adding provenance nodes."""
        dossier = CelestialDossier()
        
        state_hash = sha256_hex("state", domain=DOMAIN_ROOT)
        node = dossier.add_provenance_node(
            node_id="node_1",
            state_hash=state_hash,
            metadata={"key": "value"}
        )
        
        assert node.node_id == "node_1"
        assert node.state_hash == state_hash
        assert node.epoch == 0
        assert node.metadata["key"] == "value"
    
    def test_epoch_advancement(self):
        """Test epoch advancement."""
        dossier = CelestialDossier()
        
        assert dossier.current_epoch == 0
        
        new_epoch = dossier.advance_epoch()
        
        assert new_epoch == 1
        assert dossier.current_epoch == 1
    
    def test_cross_epoch_lineage(self):
        """Test cross-epoch provenance tracking."""
        dossier = CelestialDossier()
        
        # Add nodes in epoch 0
        node_e0 = dossier.add_provenance_node(
            node_id="node_e0",
            state_hash=sha256_hex("state_e0", domain=DOMAIN_ROOT)
        )
        
        # Advance to epoch 1
        dossier.advance_epoch()
        
        # Add node in epoch 1 with parent from epoch 0
        node_e1 = dossier.add_provenance_node(
            node_id="node_e1",
            state_hash=sha256_hex("state_e1", domain=DOMAIN_ROOT),
            parent_ids=["node_e0"]
        )
        
        # Get lineage
        lineage = dossier.get_lineage("node_e1")
        
        assert len(lineage) >= 2
        assert any(n.node_id == "node_e1" for n in lineage)
        assert any(n.node_id == "node_e0" for n in lineage)
    
    def test_dossier_root_computation(self):
        """Test deterministic dossier root computation."""
        dossier = CelestialDossier()
        
        dossier.add_provenance_node("node_1", sha256_hex("state_1", domain=DOMAIN_ROOT))
        dossier.add_provenance_node("node_2", sha256_hex("state_2", domain=DOMAIN_ROOT))
        
        root1 = dossier.compute_dossier_root()
        root2 = dossier.compute_dossier_root()
        
        # Should be deterministic
        assert root1 == root2
        assert len(root1) == 64  # SHA-256 hex
    
    def test_merkle_inclusion_proof(self):
        """Test Merkle inclusion proof generation and verification."""
        dossier = CelestialDossier()
        
        # Add multiple nodes in same epoch
        for i in range(5):
            dossier.add_provenance_node(
                node_id=f"node_{i}",
                state_hash=sha256_hex(f"state_{i}", domain=DOMAIN_ROOT)
            )
        
        # Generate proof for middle node
        proof_data = dossier.generate_merkle_inclusion_proof("node_2")
        
        assert proof_data is not None
        assert proof_data['node_id'] == "node_2"
        assert 'proof' in proof_data
        assert 'epoch_root' in proof_data
        
        # Verify proof
        is_valid = dossier.verify_merkle_inclusion_proof("node_2", proof_data)
        assert is_valid is True
    
    def test_cosmic_attestation_manifest(self):
        """Test CAM generation."""
        dossier = CelestialDossier()
        
        dossier.add_provenance_node("node_1", sha256_hex("state", domain=DOMAIN_ROOT))
        
        harmony_root = sha256_hex("harmony", domain=DOMAIN_ROOT)
        ledger_root = sha256_hex("ledger", domain=DOMAIN_ROOT)
        
        cam = dossier.generate_cosmic_attestation_manifest(
            harmony_root=harmony_root,
            ledger_root=ledger_root,
            federations=3,
            nodes=50
        )
        
        assert cam.harmony_root == harmony_root
        assert cam.ledger_root == ledger_root
        assert cam.federations == 3
        assert cam.nodes == 50
        assert len(cam.cosmic_root) == 64
        
        # Verify cosmic root computation
        combined = harmony_root + cam.dossier_root + ledger_root
        expected_cosmic = sha256_hex(combined, domain=DOMAIN_ROOT)
        assert cam.cosmic_root == expected_cosmic
    
    def test_provenance_graph_export(self):
        """Test provenance graph export."""
        dossier = CelestialDossier()
        
        # Create simple graph
        node_a = dossier.add_provenance_node("node_a", sha256_hex("a", domain=DOMAIN_ROOT))
        dossier.advance_epoch()
        node_b = dossier.add_provenance_node(
            "node_b",
            sha256_hex("b", domain=DOMAIN_ROOT),
            parent_ids=["node_a"]
        )
        
        graph = dossier.export_provenance_graph()
        
        assert 'nodes' in graph
        assert 'edges' in graph
        assert graph['total_nodes'] == 2
        assert graph['total_edges'] == 1
        assert graph['current_epoch'] == 1
    
    def test_statistics_computation(self):
        """Test provenance graph statistics."""
        dossier = CelestialDossier()
        
        # Add nodes across epochs
        for epoch in range(3):
            for i in range(2):
                parent_ids = [f"node_e{epoch-1}_n{i}"] if epoch > 0 else []
                dossier.add_provenance_node(
                    node_id=f"node_e{epoch}_n{i}",
                    state_hash=sha256_hex(f"state_{epoch}_{i}", domain=DOMAIN_ROOT),
                    parent_ids=parent_ids
                )
            if epoch < 2:
                dossier.advance_epoch()
        
        stats = dossier.compute_statistics()
        
        assert stats['total_nodes'] == 6
        assert stats['total_epochs'] == 3
        assert stats['cross_epoch_edges'] == 4  # 2 nodes × 2 epochs
        assert stats['avg_nodes_per_epoch'] == 2.0


class TestIntegration:
    """Integration tests for Phase IX components"""
    
    def test_end_to_end_consensus(self):
        """Test end-to-end consensus flow."""
        harmony = HarmonyProtocol()
        
        # Register nodes
        num_nodes = 20
        for i in range(num_nodes):
            harmony.register_node(f"validator_{i}")
        
        # Run consensus with 90% honest nodes
        canonical_value = sha256_hex("canonical", domain=DOMAIN_ROOT)
        byzantine_value = sha256_hex("byzantine", domain=DOMAIN_ROOT)
        
        attestations = []
        for i in range(num_nodes):
            if i < num_nodes * 0.9:
                attestations.append(harmony.submit_attestation(f"validator_{i}", canonical_value))
            else:
                attestations.append(harmony.submit_attestation(f"validator_{i}", byzantine_value))
        
        round_result = harmony.run_consensus_round(attestations)
        
        assert round_result.converged_value == canonical_value
        assert round_result.participation_rate == 1.0
        assert harmony.verify_safety_property() is True
        assert harmony.verify_liveness_property() is True
    
    def test_cosmic_unity_verification(self):
        """Test cosmic unity verification flow."""
        # Initialize systems
        harmony = HarmonyProtocol()
        dossier = CelestialDossier()
        
        # Run consensus
        harmony.register_node("node_1")
        attestations = [harmony.submit_attestation("node_1", sha256_hex("state", domain=DOMAIN_ROOT))]
        harmony.run_consensus_round(attestations)
        
        # Add provenance
        dossier.add_provenance_node("prov_1", sha256_hex("prov", domain=DOMAIN_ROOT))
        
        # Generate CAM
        cam = dossier.generate_cosmic_attestation_manifest(
            harmony_root=harmony.compute_harmony_root(),
            ledger_root=sha256_hex("ledger", domain=DOMAIN_ROOT),
            federations=1,
            nodes=1
        )
        
        # Verify all components present
        assert len(cam.harmony_root) == 64
        assert len(cam.dossier_root) == 64
        assert len(cam.ledger_root) == 64
        assert len(cam.cosmic_root) == 64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
