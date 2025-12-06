"""
Tests for Curriculum Fingerprinting and Drift Detection

Tests cover:
- Fingerprint computation and stability
- Hash determinism across runs
- Drift detection and reporting
- CLI functionality
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

from experiments.curriculum_loader_v2 import (
    CurriculumFingerprint,
    CurriculumLoaderV2,
    DriftReport,
    check_drift,
    compute_curriculum_fingerprint,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def simple_config() -> Dict[str, Any]:
    """Simple curriculum config for fingerprinting tests."""
    return {
        'schema_version': 'phase2-v1',
        'version': 2.1,
        'slices': {
            'slice_a': {
                'description': 'First slice',
                'parameters': {
                    'atoms': 4,
                    'depth_max': 5,
                },
                'success_metric': {
                    'kind': 'goal_hit',
                    'parameters': {
                        'min_goal_hits': 1,
                    },
                },
            },
            'slice_b': {
                'description': 'Second slice',
                'parameters': {
                    'atoms': 5,
                    'depth_max': 7,
                },
                'success_metric': {
                    'kind': 'sparse_success',
                    'parameters': {
                        'min_verified': 5,
                    },
                },
            },
        },
    }


@pytest.fixture
def loader_from_config(simple_config, tmp_path):
    """Create loader from simple config."""
    yaml_path = tmp_path / "test.yaml"
    import yaml
    with open(yaml_path, 'w') as f:
        yaml.dump(simple_config, f)
    
    return CurriculumLoaderV2.from_yaml(yaml_path)


# =============================================================================
# Fingerprint Computation Tests
# =============================================================================


def test_compute_fingerprint_basic(loader_from_config):
    """Test basic fingerprint computation."""
    fingerprint = compute_curriculum_fingerprint(loader_from_config)
    
    assert isinstance(fingerprint, CurriculumFingerprint)
    assert fingerprint.schema_version == 'phase2-v1'
    assert fingerprint.slice_count == 2
    assert set(fingerprint.metric_kinds) == {'goal_hit', 'sparse_success'}
    assert len(fingerprint.hash) == 64  # SHA-256 hex digest


def test_fingerprint_hash_is_deterministic(simple_config, tmp_path):
    """Test that same config produces same hash across runs."""
    # Load config multiple times
    fingerprints = []
    
    for i in range(3):
        yaml_path = tmp_path / f"test_{i}.yaml"
        import yaml
        with open(yaml_path, 'w') as f:
            yaml.dump(simple_config, f)
        
        loader = CurriculumLoaderV2.from_yaml(yaml_path)
        fp = compute_curriculum_fingerprint(loader)
        fingerprints.append(fp)
    
    # All hashes should be identical
    hashes = [fp.hash for fp in fingerprints]
    assert len(set(hashes)) == 1, "Hashes differ across runs"


def test_fingerprint_changes_on_parameter_modification(simple_config, tmp_path):
    """Test that modifying parameters changes the hash."""
    import yaml
    
    # Original config
    yaml_path_1 = tmp_path / "original.yaml"
    with open(yaml_path_1, 'w') as f:
        yaml.dump(simple_config, f)
    
    loader_1 = CurriculumLoaderV2.from_yaml(yaml_path_1)
    fp_1 = compute_curriculum_fingerprint(loader_1)
    
    # Modified config (change threshold)
    modified_config = json.loads(json.dumps(simple_config))  # Deep copy
    modified_config['slices']['slice_a']['success_metric']['parameters']['min_goal_hits'] = 2
    
    yaml_path_2 = tmp_path / "modified.yaml"
    with open(yaml_path_2, 'w') as f:
        yaml.dump(modified_config, f)
    
    loader_2 = CurriculumLoaderV2.from_yaml(yaml_path_2)
    fp_2 = compute_curriculum_fingerprint(loader_2)
    
    # Hashes should differ
    assert fp_1.hash != fp_2.hash


def test_fingerprint_changes_on_slice_addition(simple_config, tmp_path):
    """Test that adding a slice changes the hash."""
    import yaml
    
    # Original config
    yaml_path_1 = tmp_path / "original.yaml"
    with open(yaml_path_1, 'w') as f:
        yaml.dump(simple_config, f)
    
    loader_1 = CurriculumLoaderV2.from_yaml(yaml_path_1)
    fp_1 = compute_curriculum_fingerprint(loader_1)
    
    # Add a new slice
    modified_config = json.loads(json.dumps(simple_config))
    modified_config['slices']['slice_c'] = {
        'description': 'Third slice',
        'parameters': {'atoms': 6},
        'success_metric': {
            'kind': 'chain_success',
            'parameters': {'min_chain_length': 3},
        },
    }
    
    yaml_path_2 = tmp_path / "modified.yaml"
    with open(yaml_path_2, 'w') as f:
        yaml.dump(modified_config, f)
    
    loader_2 = CurriculumLoaderV2.from_yaml(yaml_path_2)
    fp_2 = compute_curriculum_fingerprint(loader_2)
    
    # Everything should differ
    assert fp_1.slice_count != fp_2.slice_count
    assert fp_1.hash != fp_2.hash


def test_fingerprint_metric_kinds_sorted(loader_from_config):
    """Test that metric_kinds list is sorted."""
    fingerprint = compute_curriculum_fingerprint(loader_from_config)
    
    assert fingerprint.metric_kinds == sorted(fingerprint.metric_kinds)


def test_fingerprint_serialization_roundtrip():
    """Test fingerprint serialization and deserialization."""
    original = CurriculumFingerprint(
        schema_version='phase2-v1',
        slice_count=3,
        metric_kinds=['goal_hit', 'sparse_success'],
        hash='a' * 64,
    )
    
    # Serialize
    data = original.to_dict()
    
    # Deserialize
    restored = CurriculumFingerprint.from_dict(data)
    
    assert restored.schema_version == original.schema_version
    assert restored.slice_count == original.slice_count
    assert restored.metric_kinds == original.metric_kinds
    assert restored.hash == original.hash


# =============================================================================
# Drift Detection Tests
# =============================================================================


def test_check_drift_no_differences():
    """Test drift check with matching fingerprints."""
    fp = CurriculumFingerprint(
        schema_version='phase2-v1',
        slice_count=2,
        metric_kinds=['goal_hit', 'sparse_success'],
        hash='abc123' * 10 + 'abcd',
    )
    
    report = check_drift(fp, fp)
    
    assert report.matches is True
    assert len(report.differences) == 0


def test_check_drift_schema_version_differs():
    """Test drift detection for schema version change."""
    fp1 = CurriculumFingerprint(
        schema_version='phase2-v1',
        slice_count=2,
        metric_kinds=['goal_hit'],
        hash='abc123' * 10 + 'abcd',
    )
    
    fp2 = CurriculumFingerprint(
        schema_version='phase2-v2',
        slice_count=2,
        metric_kinds=['goal_hit'],
        hash='abc123' * 10 + 'abcd',
    )
    
    report = check_drift(fp1, fp2)
    
    assert report.matches is False
    assert len(report.differences) > 0
    
    # Should mention schema_version
    diff_str = " ".join(report.differences)
    assert "schema_version" in diff_str
    assert "phase2-v1" in diff_str
    assert "phase2-v2" in diff_str


def test_check_drift_slice_count_differs():
    """Test drift detection for slice count change."""
    fp1 = CurriculumFingerprint(
        schema_version='phase2-v1',
        slice_count=2,
        metric_kinds=['goal_hit'],
        hash='abc123' * 10 + 'abcd',
    )
    
    fp2 = CurriculumFingerprint(
        schema_version='phase2-v1',
        slice_count=3,
        metric_kinds=['goal_hit'],
        hash='def456' * 10 + 'defg',
    )
    
    report = check_drift(fp1, fp2)
    
    assert report.matches is False
    
    # Should mention slice_count
    diff_str = " ".join(report.differences)
    assert "slice_count" in diff_str
    assert "2" in diff_str
    assert "3" in diff_str


def test_check_drift_metric_kinds_added():
    """Test drift detection when metric kinds are added."""
    # Expected has only goal_hit
    expected = CurriculumFingerprint(
        schema_version='phase2-v1',
        slice_count=2,
        metric_kinds=['goal_hit'],
        hash='abc123' * 10 + 'abcd',
    )
    
    # Current has goal_hit and sparse_success (added)
    current = CurriculumFingerprint(
        schema_version='phase2-v1',
        slice_count=2,
        metric_kinds=['goal_hit', 'sparse_success'],
        hash='def456' * 10 + 'defg',
    )
    
    report = check_drift(current, expected)
    
    assert report.matches is False
    
    # Should mention added metric kind
    diff_str = " ".join(report.differences)
    assert "metric_kinds added" in diff_str
    assert "sparse_success" in diff_str


def test_check_drift_metric_kinds_removed():
    """Test drift detection when metric kinds are removed."""
    # Expected has both
    expected = CurriculumFingerprint(
        schema_version='phase2-v1',
        slice_count=2,
        metric_kinds=['goal_hit', 'sparse_success'],
        hash='abc123' * 10 + 'abcd',
    )
    
    # Current only has goal_hit (sparse_success removed)
    current = CurriculumFingerprint(
        schema_version='phase2-v1',
        slice_count=2,
        metric_kinds=['goal_hit'],
        hash='def456' * 10 + 'defg',
    )
    
    report = check_drift(current, expected)
    
    assert report.matches is False
    
    # Should mention removed metric kind
    diff_str = " ".join(report.differences)
    assert "metric_kinds removed" in diff_str
    assert "sparse_success" in diff_str


def test_check_drift_only_hash_differs():
    """Test drift detection when only hash differs (threshold changes)."""
    fp1 = CurriculumFingerprint(
        schema_version='phase2-v1',
        slice_count=2,
        metric_kinds=['goal_hit', 'sparse_success'],
        hash='abc123' * 10 + 'abcd',
    )
    
    fp2 = CurriculumFingerprint(
        schema_version='phase2-v1',
        slice_count=2,
        metric_kinds=['goal_hit', 'sparse_success'],
        hash='def456' * 10 + 'defg',
    )
    
    report = check_drift(fp1, fp2)
    
    assert report.matches is False
    
    # Should indicate structural similarity but detail differences
    diff_str = " ".join(report.differences)
    assert "hash mismatch" in diff_str.lower()
    assert "threshold" in diff_str.lower() or "details" in diff_str.lower()


def test_check_drift_multiple_differences():
    """Test drift detection with multiple differences."""
    fp1 = CurriculumFingerprint(
        schema_version='phase2-v1',
        slice_count=2,
        metric_kinds=['goal_hit'],
        hash='abc123' * 10 + 'abcd',
    )
    
    fp2 = CurriculumFingerprint(
        schema_version='phase2-v2',
        slice_count=3,
        metric_kinds=['goal_hit', 'sparse_success'],
        hash='def456' * 10 + 'defg',
    )
    
    report = check_drift(fp1, fp2)
    
    assert report.matches is False
    assert len(report.differences) >= 3  # schema, count, metrics


def test_drift_report_str_format():
    """Test DriftReport string formatting."""
    # No drift
    report_match = DriftReport(matches=True, differences=[])
    assert "✓" in str(report_match) or "match" in str(report_match).lower()
    
    # With drift
    report_diff = DriftReport(
        matches=False,
        differences=["slice_count: expected 2, got 3", "hash mismatch"],
    )
    report_str = str(report_diff)
    assert "✗" in report_str or "differ" in report_str.lower()
    assert "slice_count" in report_str
    assert "hash mismatch" in report_str


# =============================================================================
# Integration Tests
# =============================================================================


def test_fingerprint_workflow_end_to_end(simple_config, tmp_path):
    """Test complete fingerprint workflow: compute, save, load, check."""
    import yaml
    
    # 1. Load curriculum
    yaml_path = tmp_path / "curriculum.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(simple_config, f)
    
    loader = CurriculumLoaderV2.from_yaml(yaml_path)
    
    # 2. Compute fingerprint
    fingerprint = compute_curriculum_fingerprint(loader)
    
    # 3. Save fingerprint
    fp_path = tmp_path / "fingerprint.json"
    with open(fp_path, 'w') as f:
        json.dump(fingerprint.to_dict(), f, indent=2)
    
    # 4. Load fingerprint
    with open(fp_path, 'r') as f:
        loaded_data = json.load(f)
    loaded_fp = CurriculumFingerprint.from_dict(loaded_data)
    
    # 5. Check drift (should match)
    report = check_drift(fingerprint, loaded_fp)
    
    assert report.matches is True
    
    # 6. Modify config and check again
    modified_config = json.loads(json.dumps(simple_config))
    modified_config['slices']['slice_a']['parameters']['atoms'] = 5
    
    yaml_path_2 = tmp_path / "curriculum_v2.yaml"
    with open(yaml_path_2, 'w') as f:
        yaml.dump(modified_config, f)
    
    loader_2 = CurriculumLoaderV2.from_yaml(yaml_path_2)
    fingerprint_2 = compute_curriculum_fingerprint(loader_2)
    
    report_2 = check_drift(fingerprint_2, loaded_fp)
    
    assert report_2.matches is False


def test_fingerprint_stable_across_slice_order(tmp_path):
    """Test that fingerprint is stable regardless of slice order in YAML."""
    import yaml
    
    # Config 1: slice_a before slice_b
    config_1 = {
        'schema_version': 'phase2-v1',
        'version': 2.1,
        'slices': {
            'slice_a': {
                'description': 'A',
                'parameters': {'atoms': 4},
                'success_metric': {'kind': 'goal_hit', 'parameters': {}},
            },
            'slice_b': {
                'description': 'B',
                'parameters': {'atoms': 5},
                'success_metric': {'kind': 'sparse_success', 'parameters': {}},
            },
        },
    }
    
    # Config 2: slice_b before slice_a (different YAML order)
    config_2 = {
        'schema_version': 'phase2-v1',
        'version': 2.1,
        'slices': {
            'slice_b': {
                'description': 'B',
                'parameters': {'atoms': 5},
                'success_metric': {'kind': 'sparse_success', 'parameters': {}},
            },
            'slice_a': {
                'description': 'A',
                'parameters': {'atoms': 4},
                'success_metric': {'kind': 'goal_hit', 'parameters': {}},
            },
        },
    }
    
    # Load both
    yaml_path_1 = tmp_path / "config1.yaml"
    with open(yaml_path_1, 'w') as f:
        yaml.dump(config_1, f)
    
    yaml_path_2 = tmp_path / "config2.yaml"
    with open(yaml_path_2, 'w') as f:
        yaml.dump(config_2, f)
    
    loader_1 = CurriculumLoaderV2.from_yaml(yaml_path_1)
    loader_2 = CurriculumLoaderV2.from_yaml(yaml_path_2)
    
    fp_1 = compute_curriculum_fingerprint(loader_1)
    fp_2 = compute_curriculum_fingerprint(loader_2)
    
    # Hashes should be identical (canonical ordering)
    assert fp_1.hash == fp_2.hash


def test_load_and_fingerprint_actual_phase2_config():
    """Test loading and fingerprinting actual Phase II config if it exists."""
    try:
        loader = CurriculumLoaderV2.from_default_phase2_config()
        fingerprint = compute_curriculum_fingerprint(loader)
        
        # Fingerprint should be valid
        assert fingerprint.schema_version in {'phase2-v1'}
        assert fingerprint.slice_count > 0
        assert len(fingerprint.metric_kinds) > 0
        assert len(fingerprint.hash) == 64
        
        print(f"✓ Phase II config fingerprint:")
        print(f"  Schema: {fingerprint.schema_version}")
        print(f"  Slices: {fingerprint.slice_count}")
        print(f"  Metrics: {fingerprint.metric_kinds}")
        print(f"  Hash: {fingerprint.hash[:16]}...")
        
    except FileNotFoundError:
        pytest.skip("Phase II config not found")
