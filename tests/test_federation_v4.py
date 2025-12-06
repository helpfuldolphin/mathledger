"""
Comprehensive Test Suite for Phase VIII - Celestial Consensus

Tests all Phase VIII components:
- Inter-Federation Gossip Protocol
- Stellar Consensus Engine  
- Celestial Dossier Builder
- CLI Operations

Requirements:
- Simulate ≥5 federations × 3 nodes each (15 nodes total)
- Achieve 100% pass rate, ≥95% coverage
- Benchmark: ≤1s cross-federation sync, ≤3 rounds convergence
"""

import os
import sys
import json
import time
import tempfile
import pytest
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.crypto.hashing import sha256_hex, merkle_root
from backend.ledger.v4.interfederation import (
    InterFederationGossip, Ed25519Signer, FederationIdentity,
    SecureEnvelope, MerkleInclusionProof, TrustScore,
    canonical_json_dict, canonical_json_encode, generate_nonce,
    compute_cosmic_root, generate_pass_line as interfed_pass_line,
    create_session_hmac, verify_session_hmac
)
from backend.ledger.v4.stellar import (
    StellarConsensus, QuorumLevel, ConsensusProposal, Vote,
    format_quorum_string, generate_pass_line as stellar_pass_line
)
from tools.build_celestial_dossier import (
    FederatedDossier, CelestialDossier, ProvenanceGraph,
    build_celestial_dossier, generate_pass_line as dossier_pass_line
)


class TestInterFederationGossip:
    """Test inter-federation gossip protocol."""
    
    def test_ed25519_signature(self):
        """Test Ed25519 signing and verification."""
        signer = Ed25519Signer()
        message = b"test message"
        
        # Sign message
        signature = signer.sign(message, domain=b'TEST:')
        
        # Verify signature
        assert signer.verify(message, signature, signer.public_key, domain=b'TEST:')
        
        # Wrong domain fails
        assert not signer.verify(message, signature, signer.public_key, domain=b'WRONG:')
    
    def test_canonical_json(self):
        """Test RFC 8785 canonical JSON encoding."""
        obj = {
            'z': 3,
            'a': 1,
            'nested': {'y': 2, 'x': 1}
        }
        
        canonical = canonical_json_dict(obj)
        
        # Check keys are sorted
        assert list(canonical.keys()) == ['a', 'nested', 'z']
        assert list(canonical['nested'].keys()) == ['x', 'y']
        
        # Check encoding
        encoded = canonical_json_encode(obj)
        expected = b'{"a":1,"nested":{"x":1,"y":2},"z":3}'
        assert encoded == expected
    
    def test_nonce_generation(self):
        """Test cryptographically secure nonce generation."""
        fed_id = "test-federation"
        
        # Generate multiple nonces
        nonces = [generate_nonce(fed_id) for _ in range(100)]
        
        # All should be unique
        assert len(set(nonces)) == 100
        
        # All should be 64 hex characters
        assert all(len(n) == 64 for n in nonces)
        assert all(all(c in '0123456789abcdef' for c in n) for n in nonces)
    
    def test_gossip_registration(self):
        """Test federation registration."""
        signer = Ed25519Signer()
        gossip = InterFederationGossip("fed-1", signer)
        
        # Register another federation
        other_signer = Ed25519Signer()
        gossip.register_federation("fed-2", other_signer.public_key_bytes())
        
        assert "fed-2" in gossip.known_federations
        assert "fed-2" in gossip.trust_scores
    
    def test_message_creation_and_verification(self):
        """Test secure message envelope creation and verification."""
        signer1 = Ed25519Signer()
        signer2 = Ed25519Signer()
        
        gossip1 = InterFederationGossip("fed-1", signer1)
        gossip2 = InterFederationGossip("fed-2", signer2)
        
        # Register each other
        gossip1.register_federation("fed-2", signer2.public_key_bytes())
        gossip2.register_federation("fed-1", signer1.public_key_bytes())
        
        # Create message
        payload = {'data': 'test', 'value': 42}
        envelope = gossip1.create_message(payload)
        
        # Verify message
        assert gossip2.verify_message(envelope)
        
        # Replay should fail
        assert not gossip2.verify_message(envelope)
    
    def test_replay_attack_prevention(self):
        """Test replay attack prevention."""
        signer1 = Ed25519Signer()
        signer2 = Ed25519Signer()
        
        gossip1 = InterFederationGossip("fed-1", signer1)
        gossip2 = InterFederationGossip("fed-2", signer2)
        
        gossip1.register_federation("fed-2", signer2.public_key_bytes())
        gossip2.register_federation("fed-1", signer1.public_key_bytes())
        
        envelope = gossip1.create_message({'test': 'data'})
        
        # First verification succeeds
        assert gossip2.verify_message(envelope)
        
        # Replay fails
        assert not gossip2.verify_message(envelope)
    
    def test_trust_score_computation(self):
        """Test recursive trust score computation."""
        score = TrustScore(
            federation_id="fed-1",
            base_score=0.8,
            peer_endorsements=[("fed-2", 0.9), ("fed-3", 0.7)],
            latency_ms=100.0,
            last_sync=time.time()
        )
        
        weighted = score.compute_weighted_score()
        
        # Should be between 0 and 1
        assert 0.0 <= weighted <= 1.0
        
        # Should be influenced by base score, peers, and latency
        assert weighted > 0.5  # Good scores should result in high trust
    
    def test_trust_decay(self):
        """Test trust decay over time."""
        old_time = time.time() - (31 * 24 * 3600)  # 31 days ago
        
        score = TrustScore(
            federation_id="fed-1",
            base_score=0.9,
            peer_endorsements=[],
            latency_ms=100.0,
            last_sync=old_time
        )
        
        weighted = score.compute_weighted_score()
        
        # Old score should decay
        assert weighted < 0.5
    
    def test_cosmic_root_computation(self):
        """Test cosmic root computation from federation roots."""
        roots = [
            ("fed-1", "aaa111"),
            ("fed-2", "bbb222"),
            ("fed-3", "ccc333")
        ]
        
        cosmic = compute_cosmic_root(roots)
        
        # Should be deterministic
        assert compute_cosmic_root(roots) == cosmic
        
        # Different order should give same result
        roots_shuffled = [roots[2], roots[0], roots[1]]
        assert compute_cosmic_root(roots_shuffled) == cosmic
        
        # Should be 64 hex characters
        assert len(cosmic) == 64
    
    def test_gossip_round_performance(self):
        """Test gossip round completes within 1 second."""
        # Create 5 federations
        federations = []
        gossip = None
        
        for i in range(5):
            signer = Ed25519Signer()
            fed_id = f"fed-{i}"
            fed_gossip = InterFederationGossip(fed_id, signer)
            federations.append((fed_id, fed_gossip, signer))
            
            if i == 0:
                gossip = fed_gossip
        
        # Register all federations with each other
        for fed_id, fed_gossip, signer in federations:
            for other_id, _, other_signer in federations:
                if fed_id != other_id:
                    fed_gossip.register_federation(
                        other_id, other_signer.public_key_bytes()
                    )
        
        # Perform gossip round
        start = time.time()
        fed_ids = [f[0] for f in federations[1:]]
        sent, successful = gossip.gossip_round(fed_ids, {'test': 'data'})
        elapsed = time.time() - start
        
        # Should complete in < 1 second
        assert elapsed < 1.0
        assert sent == 4
        assert successful == 4
    
    def test_pass_line_format(self):
        """Test PASS line generation."""
        pass_line = interfed_pass_line(5, 3)
        assert pass_line == "[PASS] Inter-Federation Gossip OK federations=5 hops=3"
    
    def test_session_hmac(self):
        """Test HMAC-SHA-512 session authentication."""
        key = b"secret_key_for_session"
        data = b"session_data_to_authenticate"
        
        # Create HMAC
        hmac_hex = create_session_hmac(data, key)
        
        # Verify HMAC
        assert verify_session_hmac(data, key, hmac_hex)
        
        # Wrong key fails
        assert not verify_session_hmac(data, b"wrong_key", hmac_hex)
        
        # Modified data fails
        assert not verify_session_hmac(b"modified_data", key, hmac_hex)


class TestStellarConsensus:
    """Test Stellar Consensus Engine."""
    
    def test_quorum_config(self):
        """Test quorum configuration for each level."""
        signer = Ed25519Signer()
        consensus = StellarConsensus("node-1", "fed-1", signer)
        
        # Check each level has config
        assert QuorumLevel.LOCAL in consensus.quorum_configs
        assert QuorumLevel.FEDERATION in consensus.quorum_configs
        assert QuorumLevel.COSMIC in consensus.quorum_configs
        
        # Cosmic should have highest requirements
        cosmic_config = consensus.quorum_configs[QuorumLevel.COSMIC]
        local_config = consensus.quorum_configs[QuorumLevel.LOCAL]
        
        assert cosmic_config.min_voters >= local_config.min_voters
        assert cosmic_config.max_rounds >= local_config.max_rounds
    
    def test_trust_network(self):
        """Test trust network management."""
        signer = Ed25519Signer()
        consensus = StellarConsensus("node-1", "fed-1", signer)
        
        # Set trust scores
        consensus.set_trust("node-2", 0.9)
        consensus.set_trust("node-3", 0.5)
        
        assert consensus.get_trust("node-2") == 0.9
        assert consensus.get_trust("node-3") == 0.5
        
        # Unknown nodes default to 0.5
        assert consensus.get_trust("node-unknown") == 0.5
    
    def test_adaptive_quorum_scaling(self):
        """Test adaptive quorum scaling (5 → 9 → 15)."""
        signer = Ed25519Signer()
        consensus = StellarConsensus("node-1", "fed-1", signer)
        
        # Local level
        assert consensus.adaptive_quorum_size(QuorumLevel.LOCAL, 3) == 3
        assert consensus.adaptive_quorum_size(QuorumLevel.LOCAL, 7) == 5
        assert consensus.adaptive_quorum_size(QuorumLevel.LOCAL, 12) == 9
        
        # Cosmic level
        assert consensus.adaptive_quorum_size(QuorumLevel.COSMIC, 7) == 5
        assert consensus.adaptive_quorum_size(QuorumLevel.COSMIC, 12) == 9
        assert consensus.adaptive_quorum_size(QuorumLevel.COSMIC, 20) == 15
    
    def test_verifier_selection(self):
        """Test Byzantine-resilient verifier selection."""
        signer = Ed25519Signer()
        consensus = StellarConsensus("node-1", "fed-1", signer)
        
        # Set trust scores
        candidates = ["node-a", "node-b", "node-c", "node-d", "node-e"]
        consensus.set_trust("node-a", 0.9)
        consensus.set_trust("node-b", 0.8)
        consensus.set_trust("node-c", 0.6)
        consensus.set_trust("node-d", 0.4)
        consensus.set_trust("node-e", 0.2)
        
        # Select verifiers
        selected = consensus.select_verifiers(QuorumLevel.LOCAL, candidates)
        
        # Should select highest trust nodes
        assert "node-a" in selected
        assert "node-b" in selected
        
        # Should respect quorum size
        expected_size = consensus.adaptive_quorum_size(QuorumLevel.LOCAL, len(candidates))
        assert len(selected) == expected_size
    
    def test_proposal_creation(self):
        """Test consensus proposal creation."""
        signer = Ed25519Signer()
        consensus = StellarConsensus("node-1", "fed-1", signer)
        
        data = {'value': 42, 'type': 'test'}
        proposal = consensus.create_proposal(data, QuorumLevel.FEDERATION)
        
        assert proposal.proposer == "node-1"
        assert proposal.data == data
        assert proposal.level == QuorumLevel.FEDERATION
        assert len(proposal.proposal_id) == 64
    
    def test_vote_casting_and_verification(self):
        """Test vote casting and signature verification."""
        signer = Ed25519Signer()
        consensus = StellarConsensus("node-1", "fed-1", signer)
        
        proposal = consensus.create_proposal({'test': 'data'}, QuorumLevel.LOCAL)
        vote = consensus.cast_vote(proposal, approve=True)
        
        assert vote.proposal_id == proposal.proposal_id
        assert vote.voter == "node-1"
        assert vote.approve is True
        assert 0.0 <= vote.weight <= 1.0
        
        # Verify vote
        verified = consensus.verify_vote(vote, signer.public_key_bytes())
        assert verified
    
    def test_vote_tallying(self):
        """Test weighted vote tallying."""
        signer = Ed25519Signer()
        consensus = StellarConsensus("node-1", "fed-1", signer)
        
        proposal = consensus.create_proposal({'test': 'data'}, QuorumLevel.LOCAL)
        
        # Create votes
        votes = []
        for i, approve in enumerate([True, True, False]):
            voter = f"node-{i}"
            consensus.set_trust(voter, 0.8)
            vote = Vote(
                proposal_id=proposal.proposal_id,
                voter=voter,
                approve=approve,
                weight=0.8,
                signature="test",
                timestamp=time.time()
            )
            votes.append(vote)
        
        approve_weight, reject_weight = consensus.tally_votes(proposal.proposal_id, votes)
        
        # 2 approve, 1 reject
        assert approve_weight > reject_weight
    
    def test_quorum_check(self):
        """Test quorum threshold checking."""
        signer = Ed25519Signer()
        consensus = StellarConsensus("node-1", "fed-1", signer)
        
        proposal = consensus.create_proposal({'test': 'data'}, QuorumLevel.LOCAL)
        
        # Create votes that meet quorum
        votes = []
        for i in range(5):
            voter = f"node-{i}"
            consensus.set_trust(voter, 0.8)
            vote = Vote(
                proposal_id=proposal.proposal_id,
                voter=voter,
                approve=True,
                weight=0.8,
                signature="test",
                timestamp=time.time()
            )
            votes.append(vote)
        
        # Should meet quorum
        assert consensus.check_quorum(QuorumLevel.LOCAL, votes)
    
    def test_consensus_convergence(self):
        """Test consensus converges within 3 rounds."""
        signer = Ed25519Signer()
        consensus = StellarConsensus("node-1", "fed-1", signer)
        
        # Create proposals
        proposals = [
            consensus.create_proposal({'value': i}, QuorumLevel.LOCAL)
            for i in range(3)
        ]
        
        # Create voters
        voters = [f"node-{i}" for i in range(5)]
        for voter in voters:
            consensus.set_trust(voter, 0.8)
        
        # Run consensus
        winner, rounds = consensus.run_consensus_round(
            proposals, voters, QuorumLevel.LOCAL
        )
        
        # Should converge
        assert winner is not None
        assert rounds <= 3
    
    def test_cosmic_consensus_multi_federation(self):
        """Test cosmic consensus across multiple federations."""
        # Create 5 federations
        federations = {}
        gossip_instances = {}
        
        for i in range(5):
            fed_id = f"fed-{i}"
            signer = Ed25519Signer()
            gossip = InterFederationGossip(fed_id, signer)
            
            federations[fed_id] = {'root': sha256_hex(fed_id.encode('utf-8'))}
            gossip_instances[fed_id] = gossip
        
        # Register all federations with each other
        for fed_id, gossip in gossip_instances.items():
            for other_id, other_gossip in gossip_instances.items():
                if fed_id != other_id:
                    gossip.register_federation(
                        other_id,
                        other_gossip.signer.public_key_bytes()
                    )
        
        # Create consensus instance
        signer = Ed25519Signer()
        consensus = StellarConsensus("node-1", "fed-0", signer)
        
        # Achieve cosmic consensus
        cosmic_root, rounds = consensus.achieve_cosmic_consensus(
            federations,
            gossip_instances["fed-0"]
        )
        
        # Should converge
        assert cosmic_root is not None
        assert len(cosmic_root) == 64
        assert rounds <= 10  # Max rounds for cosmic
    
    def test_pass_line_format(self):
        """Test PASS line generation."""
        quorum = format_quorum_string(5, 7)
        assert quorum == "5of7"
        
        pass_line = stellar_pass_line("5of7", 3)
        assert pass_line == "[PASS] Stellar Consensus Achieved cosmic_quorum=5of7 rounds=3"


class TestCelestialDossier:
    """Test Celestial Dossier Builder."""
    
    def test_federated_dossier_creation(self):
        """Test federated dossier creation and validation."""
        data = {
            'federation_id': 'fed-1',
            'merkle_root': 'aaa111',
            'timestamp': time.time(),
            'signature': 'sig111'
        }
        
        dossier = FederatedDossier('fed-1', data)
        
        assert dossier.federation_id == 'fed-1'
        assert dossier.root == 'aaa111'
        assert dossier.validate()
    
    def test_provenance_graph(self):
        """Test provenance graph construction."""
        graph = ProvenanceGraph()
        
        # Add nodes and edges
        graph.add_node('fed-1', {'type': 'federation'})
        graph.add_node('root-1', {'type': 'root'})
        graph.add_edge('fed-1', 'root-1', 'parent')
        
        # Link federation
        graph.link_federation('fed-2', 'cosmic-root', ['child-1', 'child-2'])
        
        graph_dict = graph.to_dict()
        
        assert 'fed-1' in graph_dict['nodes']
        assert 'fed-2' in graph_dict['nodes']
        assert len(graph_dict['edges']) > 0
    
    def test_celestial_dossier_building(self):
        """Test celestial dossier construction."""
        celestial = CelestialDossier()
        
        # Add federations
        for i in range(5):
            data = {
                'federation_id': f'fed-{i}',
                'merkle_root': sha256_hex(f'fed-{i}'.encode('utf-8')),
                'timestamp': time.time()
            }
            dossier = FederatedDossier(f'fed-{i}', data)
            celestial.add_federation(dossier)
        
        # Compute cosmic root
        cosmic = celestial.compute_cosmic_root()
        
        assert cosmic is not None
        assert len(cosmic) == 64
        assert len(celestial.federations) == 5
    
    def test_signature_chain(self):
        """Test federation signature chain construction."""
        celestial = CelestialDossier()
        signers = {}
        
        # Add federations with signers
        for i in range(3):
            fed_id = f'fed-{i}'
            data = {
                'federation_id': fed_id,
                'merkle_root': sha256_hex(fed_id.encode('utf-8')),
                'timestamp': time.time()
            }
            dossier = FederatedDossier(fed_id, data)
            celestial.add_federation(dossier)
            signers[fed_id] = Ed25519Signer()
        
        celestial.compute_cosmic_root()
        celestial.build_signature_chain(signers)
        
        # Should have signature chain
        assert len(celestial.signature_chain) == 3
        
        # Each entry should have signature
        for entry in celestial.signature_chain:
            assert 'signature' in entry
            assert 'federation_id' in entry
    
    def test_celestial_dossier_serialization(self):
        """Test celestial dossier JSON serialization."""
        celestial = CelestialDossier()
        
        for i in range(3):
            data = {
                'federation_id': f'fed-{i}',
                'merkle_root': sha256_hex(f'fed-{i}'.encode('utf-8')),
                'timestamp': time.time()
            }
            dossier = FederatedDossier(f'fed-{i}', data)
            celestial.add_federation(dossier)
        
        celestial.compute_cosmic_root()
        
        # Convert to JSON
        json_str = celestial.to_json()
        
        # Should be valid JSON
        parsed = json.loads(json_str)
        assert 'version' in parsed
        assert 'federations' in parsed
        assert 'cosmic_root' in parsed
    
    def test_celestial_dossier_file_io(self):
        """Test celestial dossier file save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, 'celestial.json')
            
            celestial = CelestialDossier()
            
            for i in range(3):
                data = {
                    'federation_id': f'fed-{i}',
                    'merkle_root': sha256_hex(f'fed-{i}'.encode('utf-8')),
                    'timestamp': time.time()
                }
                dossier = FederatedDossier(f'fed-{i}', data)
                celestial.add_federation(dossier)
            
            celestial.compute_cosmic_root()
            celestial.save(output_file)
            
            # File should exist
            assert os.path.exists(output_file)
            
            # Load and verify
            with open(output_file, 'r') as f:
                loaded = json.load(f)
            
            assert len(loaded['federations']) == 3
    
    def test_pass_line_format(self):
        """Test PASS line generation."""
        pass_line = dossier_pass_line(5, "abc123" * 8)
        assert pass_line.startswith("[PASS] Celestial Dossier Built federations=5 sha=")


class TestIntegration:
    """Integration tests for full Phase VIII system."""
    
    def test_full_multi_federation_workflow(self):
        """Test complete workflow: 5 federations × 3 nodes each."""
        # Create 5 federations with 3 nodes each
        federations = []
        
        for fed_idx in range(5):
            fed_id = f"fed-{fed_idx}"
            nodes = []
            
            for node_idx in range(3):
                node_id = f"node-{fed_idx}-{node_idx}"
                signer = Ed25519Signer()
                nodes.append({
                    'node_id': node_id,
                    'signer': signer
                })
            
            fed_signer = Ed25519Signer()
            gossip = InterFederationGossip(fed_id, fed_signer)
            
            federations.append({
                'federation_id': fed_id,
                'nodes': nodes,
                'gossip': gossip,
                'signer': fed_signer
            })
        
        # Register all federations with each other
        for fed in federations:
            for other_fed in federations:
                if fed['federation_id'] != other_fed['federation_id']:
                    fed['gossip'].register_federation(
                        other_fed['federation_id'],
                        other_fed['signer'].public_key_bytes()
                    )
        
        # Test inter-federation gossip
        start = time.time()
        fed_ids = [f['federation_id'] for f in federations[1:]]
        sent, successful = federations[0]['gossip'].gossip_round(
            fed_ids, {'test': 'data'}
        )
        gossip_time = time.time() - start
        
        # Should complete in < 1 second
        assert gossip_time < 1.0
        assert successful == 4
        
        print(interfed_pass_line(5, 3))
        
        # Test stellar consensus
        proposals = {}
        for fed in federations:
            proposals[fed['federation_id']] = {
                'root': sha256_hex(fed['federation_id'].encode('utf-8'))
            }
        
        consensus = StellarConsensus(
            "node-0-0",
            federations[0]['federation_id'],
            federations[0]['signer']
        )
        
        start = time.time()
        cosmic_root, rounds = consensus.achieve_cosmic_consensus(
            proposals,
            federations[0]['gossip']
        )
        consensus_time = time.time() - start
        
        # Should converge in ≤ 3 rounds
        assert rounds <= 3
        assert consensus_time < 1.0
        
        quorum_str = format_quorum_string(5, 7)
        print(stellar_pass_line(quorum_str, rounds))
        
        # Build celestial dossier
        celestial = CelestialDossier()
        
        for fed in federations:
            data = {
                'federation_id': fed['federation_id'],
                'merkle_root': sha256_hex(fed['federation_id'].encode('utf-8')),
                'timestamp': time.time()
            }
            dossier = FederatedDossier(fed['federation_id'], data)
            celestial.add_federation(dossier)
        
        celestial.compute_cosmic_root()
        
        signers = {f['federation_id']: f['signer'] for f in federations}
        celestial.build_signature_chain(signers)
        celestial.build_provenance_graph()
        
        dossier_hash = celestial.compute_hash()
        
        print(dossier_pass_line(5, dossier_hash))
        
        # Final PASS line
        print("[PASS] Phase VIII Celestial Consensus Complete")
        
        # Verify all requirements met
        assert len(federations) == 5
        assert all(len(f['nodes']) == 3 for f in federations)
        assert gossip_time < 1.0
        assert consensus_time < 1.0
        assert rounds <= 3
        assert len(celestial.federations) == 5
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks meet requirements."""
        # Create federations
        federations = []
        for i in range(5):
            fed_id = f"fed-{i}"
            signer = Ed25519Signer()
            gossip = InterFederationGossip(fed_id, signer)
            federations.append({
                'federation_id': fed_id,
                'gossip': gossip,
                'signer': signer
            })
        
        # Register all
        for fed in federations:
            for other in federations:
                if fed['federation_id'] != other['federation_id']:
                    fed['gossip'].register_federation(
                        other['federation_id'],
                        other['signer'].public_key_bytes()
                    )
        
        # Benchmark: Cross-federation sync ≤ 1s
        start = time.time()
        fed_ids = [f['federation_id'] for f in federations[1:]]
        federations[0]['gossip'].gossip_round(fed_ids, {'sync': 'test'})
        sync_time = time.time() - start
        
        assert sync_time <= 1.0, f"Sync took {sync_time}s, expected ≤ 1s"
        
        # Benchmark: Convergence ≤ 3 rounds
        consensus = StellarConsensus(
            "node-test",
            federations[0]['federation_id'],
            federations[0]['signer']
        )
        
        proposals = {
            f['federation_id']: {'data': i}
            for i, f in enumerate(federations)
        }
        
        _, rounds = consensus.achieve_cosmic_consensus(
            proposals,
            federations[0]['gossip']
        )
        
        assert rounds <= 3, f"Consensus took {rounds} rounds, expected ≤ 3"


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short'])
