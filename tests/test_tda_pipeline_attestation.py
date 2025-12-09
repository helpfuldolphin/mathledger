"""
Tests for TDA Pipeline Attestation
===================================

Validates:
1. TDA pipeline hash computation
2. Configuration divergence detection
3. Attestation chain verification with TDA binding
4. Hard Gate decision cryptographic binding
5. Exit code 4 for TDA-Ledger divergence
"""

import pytest

from attestation.tda_pipeline import (
    TDAPipelineConfig,
    compute_tda_pipeline_hash,
    detect_tda_divergence,
)
from attestation.chain_verifier import (
    AttestationVerificationError,
    ExperimentBlock,
    AttestationChainVerifier,
    verify_experiment_attestation_chain,
)


class TestTDAPipelineHash:
    """Test TDA pipeline hash computation."""
    
    def test_compute_hash_deterministic(self):
        """Hash should be deterministic for same configuration."""
        config = {
            "max_breadth": 100,
            "max_depth": 50,
            "max_total": 1000,
            "verifier_tier": "tier1",
            "verifier_timeout": 10.0,
            "verifier_budget": {"breadth": 100, "depth": 50},
            "slice_id": "test_slice_a",
            "slice_config_hash": "abc123",
            "abstention_strategy": "conservative",
        }
        
        hash1 = compute_tda_pipeline_hash(config)
        hash2 = compute_tda_pipeline_hash(config)
        
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex
        int(hash1, 16)  # Verify valid hex
    
    def test_compute_hash_different_configs(self):
        """Different configurations should produce different hashes."""
        config1 = {
            "max_breadth": 100,
            "max_depth": 50,
            "max_total": 1000,
            "verifier_tier": "tier1",
            "verifier_timeout": 10.0,
            "verifier_budget": None,
            "slice_id": "slice_a",
            "slice_config_hash": "abc123",
            "abstention_strategy": "conservative",
        }
        
        config2 = config1.copy()
        config2["max_breadth"] = 200  # Different value
        
        hash1 = compute_tda_pipeline_hash(config1)
        hash2 = compute_tda_pipeline_hash(config2)
        
        assert hash1 != hash2
    
    def test_compute_hash_missing_field(self):
        """Should raise ValueError for missing required fields."""
        config = {
            "max_breadth": 100,
            "max_depth": 50,
            # Missing max_total
        }
        
        with pytest.raises(ValueError, match="Missing required field"):
            compute_tda_pipeline_hash(config)
    
    def test_tda_pipeline_config_class(self):
        """Test TDAPipelineConfig class."""
        config = TDAPipelineConfig(
            max_breadth=100,
            max_depth=50,
            max_total=1000,
            verifier_tier="tier1",
            verifier_timeout=10.0,
            verifier_budget=None,
            slice_id="slice_a",
            slice_config_hash="abc123",
            abstention_strategy="conservative",
        )
        
        hash_val = config.compute_hash()
        assert len(hash_val) == 64
        
        # Should be same as dict-based computation
        hash_from_dict = compute_tda_pipeline_hash(config.to_dict())
        assert hash_val == hash_from_dict


class TestTDADivergenceDetection:
    """Test TDA configuration divergence detection."""
    
    def test_no_divergence(self):
        """Identical configs should not show divergence."""
        config = {
            "max_breadth": 100,
            "max_depth": 50,
            "max_total": 1000,
            "verifier_tier": "tier1",
            "verifier_timeout": 10.0,
            "verifier_budget": None,
            "slice_id": "slice_a",
            "slice_config_hash": "abc123",
            "abstention_strategy": "conservative",
        }
        
        divergence = detect_tda_divergence(
            "run_1", config,
            "run_2", config.copy()
        )
        
        assert divergence is None
    
    def test_detect_divergence(self):
        """Should detect configuration changes."""
        config1 = {
            "max_breadth": 100,
            "max_depth": 50,
            "max_total": 1000,
            "verifier_tier": "tier1",
            "verifier_timeout": 10.0,
            "verifier_budget": None,
            "slice_id": "slice_a",
            "slice_config_hash": "abc123",
            "abstention_strategy": "conservative",
        }
        
        config2 = config1.copy()
        config2["max_breadth"] = 200
        config2["verifier_tier"] = "tier2"
        
        divergence = detect_tda_divergence(
            "run_1", config1,
            "run_2", config2
        )
        
        assert divergence is not None
        assert divergence.run_id_1 == "run_1"
        assert divergence.run_id_2 == "run_2"
        assert "max_breadth" in divergence.divergent_fields
        assert "verifier_tier" in divergence.divergent_fields
        assert divergence.divergent_fields["max_breadth"] == (100, 200)


class TestExperimentBlock:
    """Test experiment block integrity verification."""
    
    def test_block_integrity_valid(self):
        """Valid block should pass integrity check."""
        # Use real dual-root attestation
        from attestation.dual_root import (
            compute_reasoning_root,
            compute_ui_root,
            compute_composite_root,
        )
        
        r_t = compute_reasoning_root(["proof1", "proof2"])
        u_t = compute_ui_root(["event1", "event2"])
        h_t = compute_composite_root(r_t, u_t)
        
        tda_config = {
            "max_breadth": 100,
            "max_depth": 50,
            "max_total": 1000,
            "verifier_tier": "tier1",
            "verifier_timeout": 10.0,
            "verifier_budget": None,
            "slice_id": "slice_a",
            "slice_config_hash": "abc123",
            "abstention_strategy": "conservative",
        }
        
        block = ExperimentBlock(
            run_id="run_001",
            experiment_id="exp_001",
            reasoning_root=r_t,
            ui_root=u_t,
            composite_root=h_t,
            tda_pipeline_hash=compute_tda_pipeline_hash(tda_config),
            tda_config=tda_config,
            block_number=0,
        )
        
        is_valid, error = block.verify_integrity()
        assert is_valid
        assert error is None
    
    def test_block_integrity_bad_composite(self):
        """Block with wrong composite root should fail."""
        from attestation.dual_root import (
            compute_reasoning_root,
            compute_ui_root,
        )
        
        r_t = compute_reasoning_root(["proof1"])
        u_t = compute_ui_root(["event1"])
        
        tda_config = {
            "max_breadth": 100,
            "max_depth": 50,
            "max_total": 1000,
            "verifier_tier": "tier1",
            "verifier_timeout": 10.0,
            "verifier_budget": None,
            "slice_id": "slice_a",
            "slice_config_hash": "abc123",
            "abstention_strategy": "conservative",
        }
        
        block = ExperimentBlock(
            run_id="run_001",
            experiment_id="exp_001",
            reasoning_root=r_t,
            ui_root=u_t,
            composite_root="0" * 64,  # Wrong composite root
            tda_pipeline_hash=compute_tda_pipeline_hash(tda_config),
            tda_config=tda_config,
            block_number=0,
        )
        
        is_valid, error = block.verify_integrity()
        assert not is_valid
        assert "does not match" in error


class TestAttestationChainVerifier:
    """Test attestation chain verification."""
    
    def _make_block(
        self,
        run_id: str,
        block_number: int,
        tda_config: dict,
        prev_block_hash: str = None
    ) -> ExperimentBlock:
        """Helper to create a valid experiment block."""
        from attestation.dual_root import (
            compute_reasoning_root,
            compute_ui_root,
            compute_composite_root,
        )
        
        r_t = compute_reasoning_root([f"proof_{run_id}"])
        u_t = compute_ui_root([f"event_{run_id}"])
        h_t = compute_composite_root(r_t, u_t)
        
        return ExperimentBlock(
            run_id=run_id,
            experiment_id="exp_test",
            reasoning_root=r_t,
            ui_root=u_t,
            composite_root=h_t,
            tda_pipeline_hash=compute_tda_pipeline_hash(tda_config),
            tda_config=tda_config,
            prev_block_hash=prev_block_hash,
            block_number=block_number,
        )
    
    def test_verify_empty_chain(self):
        """Empty chain should be valid."""
        verifier = AttestationChainVerifier()
        result = verifier.verify_chain([])
        
        assert result.is_valid
        assert result.error_code == AttestationVerificationError.SUCCESS
    
    def test_verify_single_block(self):
        """Single block should be valid if internally consistent."""
        tda_config = {
            "max_breadth": 100,
            "max_depth": 50,
            "max_total": 1000,
            "verifier_tier": "tier1",
            "verifier_timeout": 10.0,
            "verifier_budget": None,
            "slice_id": "slice_a",
            "slice_config_hash": "abc123",
            "abstention_strategy": "conservative",
        }
        
        block = self._make_block("run_001", 0, tda_config)
        
        verifier = AttestationChainVerifier()
        result = verifier.verify_chain([block])
        
        assert result.is_valid
        assert result.error_code == AttestationVerificationError.SUCCESS
    
    def test_verify_chain_valid_linkage(self):
        """Chain with valid linkage should pass."""
        tda_config = {
            "max_breadth": 100,
            "max_depth": 50,
            "max_total": 1000,
            "verifier_tier": "tier1",
            "verifier_timeout": 10.0,
            "verifier_budget": None,
            "slice_id": "slice_a",
            "slice_config_hash": "abc123",
            "abstention_strategy": "conservative",
        }
        
        block0 = self._make_block("run_001", 0, tda_config)
        block1 = self._make_block(
            "run_002", 1, tda_config,
            prev_block_hash=block0.compute_block_hash()
        )
        
        verifier = AttestationChainVerifier()
        result = verifier.verify_chain([block0, block1])
        
        assert result.is_valid
        assert result.error_code == AttestationVerificationError.SUCCESS
    
    def test_verify_chain_broken_linkage(self):
        """Chain with broken linkage should fail."""
        tda_config = {
            "max_breadth": 100,
            "max_depth": 50,
            "max_total": 1000,
            "verifier_tier": "tier1",
            "verifier_timeout": 10.0,
            "verifier_budget": None,
            "slice_id": "slice_a",
            "slice_config_hash": "abc123",
            "abstention_strategy": "conservative",
        }
        
        block0 = self._make_block("run_001", 0, tda_config)
        block1 = self._make_block(
            "run_002", 1, tda_config,
            prev_block_hash="0" * 64  # Wrong hash
        )
        
        verifier = AttestationChainVerifier()
        result = verifier.verify_chain([block0, block1])
        
        assert not result.is_valid
        assert result.error_code == AttestationVerificationError.CHAIN_LINKAGE_BROKEN
    
    def test_verify_tda_divergence_strict(self):
        """TDA divergence should fail in strict mode."""
        tda_config1 = {
            "max_breadth": 100,
            "max_depth": 50,
            "max_total": 1000,
            "verifier_tier": "tier1",
            "verifier_timeout": 10.0,
            "verifier_budget": None,
            "slice_id": "slice_a",
            "slice_config_hash": "abc123",
            "abstention_strategy": "conservative",
        }
        
        tda_config2 = tda_config1.copy()
        tda_config2["max_breadth"] = 200  # Drift!
        
        block0 = self._make_block("run_001", 0, tda_config1)
        block1 = self._make_block(
            "run_002", 1, tda_config2,
            prev_block_hash=block0.compute_block_hash()
        )
        
        verifier = AttestationChainVerifier(strict_tda_consistency=True)
        result = verifier.verify_chain([block0, block1])
        
        assert not result.is_valid
        assert result.error_code == AttestationVerificationError.TDA_DIVERGENCE
        assert len(result.divergences) == 1
    
    def test_verify_tda_divergence_permissive(self):
        """TDA divergence should warn but pass in permissive mode."""
        tda_config1 = {
            "max_breadth": 100,
            "max_depth": 50,
            "max_total": 1000,
            "verifier_tier": "tier1",
            "verifier_timeout": 10.0,
            "verifier_budget": None,
            "slice_id": "slice_a",
            "slice_config_hash": "abc123",
            "abstention_strategy": "conservative",
        }
        
        tda_config2 = tda_config1.copy()
        tda_config2["max_breadth"] = 200
        
        block0 = self._make_block("run_001", 0, tda_config1)
        block1 = self._make_block(
            "run_002", 1, tda_config2,
            prev_block_hash=block0.compute_block_hash()
        )
        
        verifier = AttestationChainVerifier(strict_tda_consistency=False)
        result = verifier.verify_chain([block0, block1])
        
        assert result.is_valid  # Passes despite divergence
        assert result.error_code == AttestationVerificationError.SUCCESS
        assert len(result.divergences) == 1  # But divergence is recorded


class TestHardGateBinding:
    """Test Hard Gate decision cryptographic binding."""
    
    def test_gate_decisions_bound_in_hash(self):
        """Gate decisions should affect block hash."""
        from attestation.dual_root import (
            compute_reasoning_root,
            compute_ui_root,
            compute_composite_root,
        )
        
        r_t = compute_reasoning_root(["proof"])
        u_t = compute_ui_root(["event"])
        h_t = compute_composite_root(r_t, u_t)
        
        tda_config = {
            "max_breadth": 100,
            "max_depth": 50,
            "max_total": 1000,
            "verifier_tier": "tier1",
            "verifier_timeout": 10.0,
            "verifier_budget": None,
            "slice_id": "slice_a",
            "slice_config_hash": "abc123",
            "abstention_strategy": "conservative",
        }
        
        block1 = ExperimentBlock(
            run_id="run_001",
            experiment_id="exp",
            reasoning_root=r_t,
            ui_root=u_t,
            composite_root=h_t,
            tda_pipeline_hash=compute_tda_pipeline_hash(tda_config),
            tda_config=tda_config,
            gate_decisions={"G1": "PASS", "G2": "ABANDONED_TDA"},
            block_number=0,
        )
        
        block2 = ExperimentBlock(
            run_id="run_001",
            experiment_id="exp",
            reasoning_root=r_t,
            ui_root=u_t,
            composite_root=h_t,
            tda_pipeline_hash=compute_tda_pipeline_hash(tda_config),
            tda_config=tda_config,
            gate_decisions={"G1": "PASS", "G2": "PASS"},  # Different decision
            block_number=0,
        )
        
        hash1 = block1.compute_block_hash()
        hash2 = block2.compute_block_hash()
        
        assert hash1 != hash2  # Different gate decisions produce different hashes
    
    def test_verify_hard_gate_binding(self):
        """Should verify gate decisions match expected."""
        from attestation.dual_root import (
            compute_reasoning_root,
            compute_ui_root,
            compute_composite_root,
        )
        
        r_t = compute_reasoning_root(["proof"])
        u_t = compute_ui_root(["event"])
        h_t = compute_composite_root(r_t, u_t)
        
        tda_config = {
            "max_breadth": 100,
            "max_depth": 50,
            "max_total": 1000,
            "verifier_tier": "tier1",
            "verifier_timeout": 10.0,
            "verifier_budget": None,
            "slice_id": "slice_a",
            "slice_config_hash": "abc123",
            "abstention_strategy": "conservative",
        }
        
        block = ExperimentBlock(
            run_id="run_001",
            experiment_id="exp",
            reasoning_root=r_t,
            ui_root=u_t,
            composite_root=h_t,
            tda_pipeline_hash=compute_tda_pipeline_hash(tda_config),
            tda_config=tda_config,
            gate_decisions={"G1": "PASS", "G2": "ABANDONED_TDA"},
            block_number=0,
        )
        
        verifier = AttestationChainVerifier()
        
        # Matching decisions should pass
        is_valid, error = verifier.verify_hard_gate_binding(
            block,
            {"G1": "PASS", "G2": "ABANDONED_TDA"}
        )
        assert is_valid
        
        # Mismatched decisions should fail
        is_valid, error = verifier.verify_hard_gate_binding(
            block,
            {"G1": "PASS", "G2": "PASS"}  # Wrong decision
        )
        assert not is_valid
        assert "mismatch" in error.lower()
