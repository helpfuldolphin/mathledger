"""
Tests for Cross-Chain Attestation Verification

Validates cross-experiment attestation chain integrity, drift detection,
and CI guard functionality.
"""

import json
import pytest
import hashlib
from pathlib import Path
from datetime import datetime, timedelta

from attestation.cross_chain_verifier import (
    CrossChainVerifier,
    ChainVerificationResult,
    ChainDiscontinuity,
    DuplicateExperiment,
    HashDrift,
    SchemaDrift,
    DualRootMismatch,
    TimestampViolation,
)
from attestation.dual_root import (
    compute_reasoning_root,
    compute_ui_root,
    compute_composite_root,
)


class TestCrossChainVerifier:
    """Test cross-chain verifier basic functionality."""
    
    def test_empty_chain(self):
        """Empty chain should pass validation."""
        verifier = CrossChainVerifier()
        result = verifier.verify_chain([])
        
        assert result.total_experiments == 0
        assert result.valid_experiments == 0
        assert result.is_valid
        assert not result.has_warnings
    
    def test_single_valid_manifest(self):
        """Single valid manifest should pass."""
        r_t = 'a' * 64
        u_t = 'b' * 64
        h_t = compute_composite_root(r_t, u_t)
        
        manifest = {
            'experiment_id': 'EXP_001',
            'manifest_version': '1.0',
            'timestamp_utc': '2025-01-01T00:00:00Z',
            'reasoning_merkle_root': r_t,
            'ui_merkle_root': u_t,
            'composite_attestation_root': h_t,
        }
        
        verifier = CrossChainVerifier()
        result = verifier.verify_chain([manifest])
        
        assert result.total_experiments == 1
        assert result.valid_experiments == 1
        assert result.is_valid
        assert not result.has_warnings
    
    def test_chain_with_valid_prev_hash(self):
        """Chain with correct prev_hash links should pass."""
        manifest1 = {
            'experiment_id': 'EXP_001',
            'manifest_version': '1.0',
            'timestamp_utc': '2025-01-01T00:00:00Z',
        }
        
        # Compute hash of first manifest
        manifest1_hash = hashlib.sha256(
            json.dumps({k: v for k, v in manifest1.items() if k != 'prev_hash'}, sort_keys=True, separators=(',', ':')).encode('utf-8')
        ).hexdigest()
        
        manifest2 = {
            'experiment_id': 'EXP_002',
            'manifest_version': '1.0',
            'timestamp_utc': '2025-01-02T00:00:00Z',
            'prev_hash': manifest1_hash,
        }
        
        verifier = CrossChainVerifier()
        result = verifier.verify_chain([manifest1, manifest2])
        
        assert result.total_experiments == 2
        assert result.valid_experiments == 2
        assert len(result.chain_discontinuities) == 0
        assert result.is_valid
    
    def test_detect_chain_discontinuity(self):
        """Detect broken prev_hash link."""
        manifest1 = {
            'experiment_id': 'EXP_001',
            'manifest_version': '1.0',
            'timestamp_utc': '2025-01-01T00:00:00Z',
        }
        
        manifest2 = {
            'experiment_id': 'EXP_002',
            'manifest_version': '1.0',
            'timestamp_utc': '2025-01-02T00:00:00Z',
            'prev_hash': 'wrong_hash',
        }
        
        verifier = CrossChainVerifier()
        result = verifier.verify_chain([manifest1, manifest2])
        
        assert result.total_experiments == 2
        assert len(result.chain_discontinuities) == 1
        assert not result.is_valid
        
        discontinuity = result.chain_discontinuities[0]
        assert discontinuity.experiment_id == 'EXP_002'
        assert discontinuity.actual_prev_hash == 'wrong_hash'
        assert discontinuity.position == 1
    
    def test_detect_duplicate_experiment_ids(self):
        """Detect repeated experiment IDs."""
        manifest1 = {
            'experiment_id': 'EXP_001',
            'manifest_version': '1.0',
            'timestamp_utc': '2025-01-01T00:00:00Z',
        }
        
        manifest2 = {
            'experiment_id': 'EXP_001',  # Duplicate!
            'manifest_version': '1.0',
            'timestamp_utc': '2025-01-02T00:00:00Z',
        }
        
        verifier = CrossChainVerifier()
        result = verifier.verify_chain([manifest1, manifest2])
        
        assert result.total_experiments == 2
        assert len(result.duplicate_experiments) == 1
        assert not result.is_valid
        
        duplicate = result.duplicate_experiments[0]
        assert duplicate.experiment_id == 'EXP_001'
        assert duplicate.occurrences == [0, 1]


class TestDualRootVerification:
    """Test dual-root attestation verification."""
    
    def test_valid_dual_root(self):
        """Valid dual-root attestation should pass."""
        r_t = compute_reasoning_root(['proof1', 'proof2'])
        u_t = compute_ui_root(['event1'])
        h_t = compute_composite_root(r_t, u_t)
        
        manifest = {
            'experiment_id': 'EXP_001',
            'manifest_version': '1.0',
            'timestamp_utc': '2025-01-01T00:00:00Z',
            'reasoning_merkle_root': r_t,
            'ui_merkle_root': u_t,
            'composite_attestation_root': h_t,
        }
        
        verifier = CrossChainVerifier()
        result = verifier.verify_chain([manifest])
        
        assert result.total_experiments == 1
        assert len(result.dual_root_mismatches) == 0
        assert result.is_valid
    
    def test_detect_dual_root_mismatch(self):
        """Detect H_t != SHA256(R_t || U_t)."""
        r_t = compute_reasoning_root(['proof1'])
        u_t = compute_ui_root(['event1'])
        h_t_wrong = 'f' * 64  # Wrong hash
        
        manifest = {
            'experiment_id': 'EXP_001',
            'manifest_version': '1.0',
            'timestamp_utc': '2025-01-01T00:00:00Z',
            'reasoning_merkle_root': r_t,
            'ui_merkle_root': u_t,
            'composite_attestation_root': h_t_wrong,
        }
        
        verifier = CrossChainVerifier()
        result = verifier.verify_chain([manifest])
        
        assert result.total_experiments == 1
        assert len(result.dual_root_mismatches) == 1
        assert not result.is_valid
        
        mismatch = result.dual_root_mismatches[0]
        assert mismatch.experiment_id == 'EXP_001'
        assert mismatch.h_t == h_t_wrong
        assert mismatch.recomputed_h_t != h_t_wrong
    
    def test_partial_dual_root_fields(self):
        """Missing dual-root fields should not cause error."""
        manifest = {
            'experiment_id': 'EXP_001',
            'manifest_version': '1.0',
            'timestamp_utc': '2025-01-01T00:00:00Z',
            'reasoning_merkle_root': 'a' * 64,
            # Missing ui_merkle_root and composite_attestation_root
        }
        
        verifier = CrossChainVerifier()
        result = verifier.verify_chain([manifest])
        
        # Should not crash, just skip dual-root verification
        assert result.total_experiments == 1
        assert len(result.dual_root_mismatches) == 0


class TestAttestationDriftRadar:
    """Test attestation drift detection."""
    
    def test_detect_hash_drift(self):
        """Detect configuration hash drift across runs."""
        config1 = {'param': 'value1'}
        config2 = {'param': 'value2'}
        
        manifest1 = {
            'experiment_id': 'EXP_001',
            'manifest_version': '1.0',
            'timestamp_utc': '2025-01-01T00:00:00Z',
            'configuration': {
                'snapshot': config1,
            },
        }
        
        manifest2 = {
            'experiment_id': 'EXP_001',
            'manifest_version': '1.0',
            'timestamp_utc': '2025-01-02T00:00:00Z',
            'configuration': {
                'snapshot': config2,
            },
        }
        
        verifier = CrossChainVerifier()
        result = verifier.verify_chain([manifest1, manifest2])
        
        assert result.total_experiments == 2
        assert len(result.hash_drifts) == 1
        assert result.has_warnings
        
        drift = result.hash_drifts[0]
        assert drift.experiment_id == 'EXP_001'
        assert drift.hash_field == 'configuration.snapshot'
    
    def test_detect_schema_drift_missing_fields(self):
        """Detect missing required fields."""
        manifest = {
            'experiment_id': 'EXP_001',
            # Missing 'manifest_version' and 'timestamp_utc'
        }
        
        verifier = CrossChainVerifier()
        result = verifier.verify_chain([manifest])
        
        assert result.total_experiments == 1
        assert len(result.schema_drifts) == 1
        assert result.has_warnings
        
        drift = result.schema_drifts[0]
        assert 'manifest_version' in drift.missing_fields
        assert 'timestamp_utc' in drift.missing_fields
    
    def test_detect_schema_drift_extra_fields_strict(self):
        """Detect extra fields in strict mode."""
        manifest = {
            'experiment_id': 'EXP_001',
            'manifest_version': '1.0',
            'timestamp_utc': '2025-01-01T00:00:00Z',
            'unexpected_field': 'value',
        }
        
        verifier = CrossChainVerifier(strict_schema=True)
        result = verifier.verify_chain([manifest])
        
        assert result.total_experiments == 1
        assert len(result.schema_drifts) == 1
        assert result.has_warnings
        
        drift = result.schema_drifts[0]
        assert 'unexpected_field' in drift.extra_fields
    
    def test_no_schema_drift_extra_fields_lenient(self):
        """Extra fields allowed in lenient mode."""
        manifest = {
            'experiment_id': 'EXP_001',
            'manifest_version': '1.0',
            'timestamp_utc': '2025-01-01T00:00:00Z',
            'unexpected_field': 'value',
        }
        
        verifier = CrossChainVerifier(strict_schema=False)
        result = verifier.verify_chain([manifest])
        
        assert result.total_experiments == 1
        assert len(result.schema_drifts) == 0  # Extra fields OK in lenient mode
    
    def test_detect_timestamp_monotonicity_violation(self):
        """Detect timestamp going backwards."""
        manifest1 = {
            'experiment_id': 'EXP_001',
            'manifest_version': '1.0',
            'timestamp_utc': '2025-01-02T00:00:00Z',
        }
        
        manifest2 = {
            'experiment_id': 'EXP_002',
            'manifest_version': '1.0',
            'timestamp_utc': '2025-01-01T00:00:00Z',  # Earlier than manifest1!
        }
        
        verifier = CrossChainVerifier()
        result = verifier.verify_chain([manifest1, manifest2])
        
        assert result.total_experiments == 2
        assert len(result.timestamp_violations) == 1
        assert result.has_warnings
        
        violation = result.timestamp_violations[0]
        assert violation.experiment_id_1 == 'EXP_002'
        assert violation.experiment_id_2 == 'EXP_001'
    
    def test_skip_timestamp_check_when_disabled(self):
        """Timestamp checking can be disabled."""
        manifest1 = {
            'experiment_id': 'EXP_001',
            'manifest_version': '1.0',
            'timestamp_utc': '2025-01-02T00:00:00Z',
        }
        
        manifest2 = {
            'experiment_id': 'EXP_002',
            'manifest_version': '1.0',
            'timestamp_utc': '2025-01-01T00:00:00Z',
        }
        
        verifier = CrossChainVerifier()
        result = verifier.verify_chain([manifest1, manifest2], check_ordering=False)
        
        assert result.total_experiments == 2
        assert len(result.timestamp_violations) == 0


class TestChainVerificationResult:
    """Test verification result reporting."""
    
    def test_is_valid_with_no_issues(self):
        """Result with no issues is valid."""
        result = ChainVerificationResult()
        result.total_experiments = 5
        result.valid_experiments = 5
        
        assert result.is_valid
        assert not result.has_warnings
    
    def test_is_invalid_with_discontinuity(self):
        """Result with chain discontinuity is invalid."""
        result = ChainVerificationResult()
        result.total_experiments = 2
        result.valid_experiments = 1
        result.chain_discontinuities.append(
            ChainDiscontinuity(
                experiment_id='EXP_001',
                expected_prev_hash='abc',
                actual_prev_hash='def',
                position=1,
            )
        )
        
        assert not result.is_valid
        assert not result.has_warnings
    
    def test_has_warnings_with_drift(self):
        """Result with drift has warnings."""
        result = ChainVerificationResult()
        result.total_experiments = 2
        result.valid_experiments = 2
        result.hash_drifts.append(
            HashDrift(
                experiment_id='EXP_001',
                run_1=0,
                run_2=1,
                hash_field='config',
                hash_1='abc',
                hash_2='def',
            )
        )
        
        assert result.is_valid
        assert result.has_warnings
    
    def test_summary_generation(self):
        """Verify summary format."""
        result = ChainVerificationResult()
        result.total_experiments = 3
        result.valid_experiments = 2
        result.chain_discontinuities.append(
            ChainDiscontinuity(
                experiment_id='EXP_001',
                expected_prev_hash='abc',
                actual_prev_hash='def',
                position=1,
            )
        )
        
        summary = result.summary()
        
        assert 'Cross-Chain Verification Report' in summary
        assert 'Total experiments: 3' in summary
        assert 'Valid experiments: 2' in summary
        assert 'CRITICAL: 1 chain discontinuities' in summary


class TestArtifactsDirectoryVerification:
    """Test verification of artifacts directory."""
    
    def test_verify_empty_directory(self, tmp_path):
        """Empty directory should pass."""
        verifier = CrossChainVerifier()
        result = verifier.verify_artifacts_directory(tmp_path)
        
        assert result.total_experiments == 0
        assert result.is_valid
    
    def test_verify_directory_with_manifests(self, tmp_path):
        """Verify directory with multiple manifests."""
        # Create experiment directories
        exp1_dir = tmp_path / 'exp1'
        exp1_dir.mkdir()
        
        exp2_dir = tmp_path / 'exp2'
        exp2_dir.mkdir()
        
        # Create manifests
        manifest1 = {
            'experiment_id': 'EXP_001',
            'manifest_version': '1.0',
            'timestamp_utc': '2025-01-01T00:00:00Z',
        }
        
        with open(exp1_dir / 'attestation.json', 'w') as f:
            json.dump(manifest1, f)
        
        # Compute correct prev_hash
        manifest1_hash = hashlib.sha256(
            json.dumps(manifest1, sort_keys=True, separators=(',', ':')).encode('utf-8')
        ).hexdigest()
        
        manifest2 = {
            'experiment_id': 'EXP_002',
            'manifest_version': '1.0',
            'timestamp_utc': '2025-01-02T00:00:00Z',
            'prev_hash': manifest1_hash,
        }
        
        with open(exp2_dir / 'attestation.json', 'w') as f:
            json.dump(manifest2, f)
        
        # Verify
        verifier = CrossChainVerifier()
        result = verifier.verify_artifacts_directory(tmp_path)
        
        assert result.total_experiments == 2
        assert result.valid_experiments == 2
        assert result.is_valid
    
    def test_skip_invalid_json_files(self, tmp_path):
        """Invalid JSON files should be skipped."""
        exp_dir = tmp_path / 'exp1'
        exp_dir.mkdir()
        
        # Create invalid JSON
        with open(exp_dir / 'attestation.json', 'w') as f:
            f.write('invalid json {')
        
        verifier = CrossChainVerifier()
        result = verifier.verify_artifacts_directory(tmp_path)
        
        # Should not crash, just skip the file
        assert result.total_experiments == 0


class TestCompositeChainOrdering:
    """Test composite chain ordering verification."""
    
    def test_ordered_chain_by_timestamp(self):
        """Chain ordered by timestamp should pass (when prev_hash links are correct)."""
        base_time = datetime(2025, 1, 1)
        manifests = []
        prev_hash = None
        
        for i in range(5):
            timestamp = (base_time + timedelta(hours=i)).isoformat() + 'Z'
            manifest = {
                'experiment_id': f'EXP_{i:03d}',
                'manifest_version': '1.0',
                'timestamp_utc': timestamp,
            }
            
            if prev_hash is not None:
                manifest['prev_hash'] = prev_hash
            
            manifests.append(manifest)
            
            # Compute hash for next iteration
            prev_hash = hashlib.sha256(
                json.dumps({k: v for k, v in manifest.items() if k != 'prev_hash'}, sort_keys=True, separators=(',', ':')).encode('utf-8')
            ).hexdigest()
        
        verifier = CrossChainVerifier()
        result = verifier.verify_chain(manifests)
        
        assert result.total_experiments == 5
        assert len(result.timestamp_violations) == 0
        assert result.is_valid
    
    def test_detect_multiple_ordering_violations(self):
        """Detect multiple timestamp violations."""
        manifests = [
            {
                'experiment_id': 'EXP_001',
                'manifest_version': '1.0',
                'timestamp_utc': '2025-01-03T00:00:00Z',
            },
            {
                'experiment_id': 'EXP_002',
                'manifest_version': '1.0',
                'timestamp_utc': '2025-01-01T00:00:00Z',  # Violation 1
            },
            {
                'experiment_id': 'EXP_003',
                'manifest_version': '1.0',
                'timestamp_utc': '2025-01-02T00:00:00Z',  # OK (after EXP_002)
            },
        ]
        
        verifier = CrossChainVerifier()
        result = verifier.verify_chain(manifests)
        
        assert result.total_experiments == 3
        assert len(result.timestamp_violations) == 1  # Only first violation detected


class TestIntegrationScenarios:
    """Integration tests for real-world scenarios."""
    
    def test_complete_valid_chain(self):
        """Test complete valid chain with all features."""
        r_t1 = compute_reasoning_root(['proof1'])
        u_t1 = compute_ui_root(['event1'])
        h_t1 = compute_composite_root(r_t1, u_t1)
        
        manifest1 = {
            'experiment_id': 'U2_EXP_001',
            'manifest_version': '1.0',
            'timestamp_utc': '2025-01-01T00:00:00Z',
            'run_index': 1,
            'reasoning_merkle_root': r_t1,
            'ui_merkle_root': u_t1,
            'composite_attestation_root': h_t1,
            'provenance': {
                'git_commit': 'abc123',
                'user': 'test-user',
            },
            'configuration': {
                'snapshot': {'param': 'value1'},
            },
        }
        
        manifest1_hash = hashlib.sha256(
            json.dumps({k: v for k, v in manifest1.items() if k != 'prev_hash'}, sort_keys=True, separators=(',', ':')).encode('utf-8')
        ).hexdigest()
        
        r_t2 = compute_reasoning_root(['proof2'])
        u_t2 = compute_ui_root(['event2'])
        h_t2 = compute_composite_root(r_t2, u_t2)
        
        manifest2 = {
            'experiment_id': 'U2_EXP_002',
            'manifest_version': '1.0',
            'timestamp_utc': '2025-01-02T00:00:00Z',
            'run_index': 2,
            'prev_hash': manifest1_hash,
            'reasoning_merkle_root': r_t2,
            'ui_merkle_root': u_t2,
            'composite_attestation_root': h_t2,
            'provenance': {
                'git_commit': 'abc123',
                'user': 'test-user',
            },
            'configuration': {
                'snapshot': {'param': 'value1'},
            },
        }
        
        verifier = CrossChainVerifier()
        result = verifier.verify_chain([manifest1, manifest2])
        
        assert result.total_experiments == 2
        assert result.valid_experiments == 2
        assert result.is_valid
        assert not result.has_warnings
        
        summary = result.summary()
        assert '✓ All experiments verified successfully' in summary
    
    def test_multiple_issues_detected(self):
        """Test detection of multiple issue types."""
        manifest1 = {
            'experiment_id': 'EXP_001',
            'manifest_version': '1.0',
            'timestamp_utc': '2025-01-02T00:00:00Z',
            'reasoning_merkle_root': 'a' * 64,
            'ui_merkle_root': 'b' * 64,
            'composite_attestation_root': 'f' * 64,  # Wrong!
        }
        
        manifest2 = {
            'experiment_id': 'EXP_001',  # Duplicate!
            # Missing manifest_version and timestamp_utc
            'prev_hash': 'wrong_hash',  # Chain break!
            'configuration': {
                'snapshot': {'different': 'config'},
            },
        }
        
        verifier = CrossChainVerifier()
        result = verifier.verify_chain([manifest1, manifest2])
        
        assert result.total_experiments == 2
        assert not result.is_valid
        assert result.has_warnings
        
        # Check all issue types detected
        assert len(result.chain_discontinuities) == 1
        assert len(result.duplicate_experiments) == 1
        assert len(result.dual_root_mismatches) == 1
        assert len(result.schema_drifts) == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
