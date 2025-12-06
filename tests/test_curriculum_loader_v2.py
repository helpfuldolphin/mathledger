"""
Tests for Phase II Curriculum Loader V2 with Drift Guard

Tests cover:
- Schema version validation
- Structural validation with clear error messages
- Curriculum loading and parsing
- Introspection methods
- Error handling
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

from experiments.curriculum_loader_v2 import (
    ALLOWED_SCHEMA_VERSIONS,
    CurriculumLoaderV2,
    CurriculumValidationError,
    SuccessMetricSpec,
    UpliftSlice,
    validate_curriculum_structure,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def valid_minimal_config() -> Dict[str, Any]:
    """Minimal valid curriculum configuration."""
    return {
        'schema_version': 'phase2-v1',
        'version': 2.1,
        'slices': {
            'test_slice': {
                'description': 'Test slice',
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
        },
    }


@pytest.fixture
def valid_full_config() -> Dict[str, Any]:
    """Full valid curriculum configuration with all fields."""
    return {
        'schema_version': 'phase2-v1',
        'version': 2.1,
        'slices': {
            'slice_a': {
                'description': 'First test slice',
                'parameters': {
                    'atoms': 4,
                    'depth_max': 5,
                    'breadth_max': 40,
                },
                'success_metric': {
                    'kind': 'goal_hit',
                    'parameters': {
                        'min_goal_hits': 1,
                        'min_total_verified': 3,
                    },
                },
                'uplift': {
                    'phase': 'II',
                    'experiment_family': 'U2',
                },
                'budget': {
                    'max_candidates_per_cycle': 40,
                    'max_cycles_per_run': 500,
                },
                'formula_pool_entries': ['p', 'q', 'p->q'],
            },
            'slice_b': {
                'description': 'Second test slice',
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


# =============================================================================
# Schema Version Tests
# =============================================================================


def test_validate_accepts_correct_schema_version(valid_minimal_config):
    """Test that validation accepts correct schema version."""
    # Should not raise
    validate_curriculum_structure(valid_minimal_config)


def test_validate_rejects_missing_schema_version(valid_minimal_config):
    """Test that validation rejects missing schema_version."""
    del valid_minimal_config['schema_version']
    
    with pytest.raises(CurriculumValidationError) as exc_info:
        validate_curriculum_structure(valid_minimal_config)
    
    assert "schema_version" in str(exc_info.value).lower()
    assert "missing" in str(exc_info.value).lower()


def test_validate_rejects_unknown_schema_version(valid_minimal_config):
    """Test that validation rejects unknown schema version."""
    valid_minimal_config['schema_version'] = 'unknown-version'
    
    with pytest.raises(CurriculumValidationError) as exc_info:
        validate_curriculum_structure(valid_minimal_config)
    
    error_msg = str(exc_info.value)
    assert "unknown-version" in error_msg
    assert "unsupported" in error_msg.lower()
    # Should list allowed versions
    for allowed in ALLOWED_SCHEMA_VERSIONS:
        assert allowed in error_msg


def test_validate_checks_version_field(valid_minimal_config):
    """Test that validation checks for version field."""
    del valid_minimal_config['version']
    
    with pytest.raises(CurriculumValidationError) as exc_info:
        validate_curriculum_structure(valid_minimal_config)
    
    assert "version" in str(exc_info.value).lower()


# =============================================================================
# Structural Validation Tests
# =============================================================================


def test_validate_rejects_missing_slices(valid_minimal_config):
    """Test that validation rejects missing slices dict."""
    del valid_minimal_config['slices']
    
    with pytest.raises(CurriculumValidationError) as exc_info:
        validate_curriculum_structure(valid_minimal_config)
    
    assert "slices" in str(exc_info.value).lower()


def test_validate_rejects_empty_slices(valid_minimal_config):
    """Test that validation rejects empty slices dict."""
    valid_minimal_config['slices'] = {}
    
    with pytest.raises(CurriculumValidationError) as exc_info:
        validate_curriculum_structure(valid_minimal_config)
    
    assert "no slices" in str(exc_info.value).lower()


def test_validate_rejects_slice_missing_description(valid_minimal_config):
    """Test that validation rejects slice without description."""
    del valid_minimal_config['slices']['test_slice']['description']
    
    with pytest.raises(CurriculumValidationError) as exc_info:
        validate_curriculum_structure(valid_minimal_config)
    
    error_msg = str(exc_info.value)
    assert "test_slice" in error_msg
    assert "description" in error_msg.lower()
    assert "missing" in error_msg.lower()


def test_validate_rejects_slice_missing_parameters(valid_minimal_config):
    """Test that validation rejects slice without parameters."""
    del valid_minimal_config['slices']['test_slice']['parameters']
    
    with pytest.raises(CurriculumValidationError) as exc_info:
        validate_curriculum_structure(valid_minimal_config)
    
    error_msg = str(exc_info.value)
    assert "test_slice" in error_msg
    assert "parameters" in error_msg.lower()


def test_validate_rejects_slice_missing_success_metric(valid_minimal_config):
    """Test that validation rejects slice without success_metric."""
    del valid_minimal_config['slices']['test_slice']['success_metric']
    
    with pytest.raises(CurriculumValidationError) as exc_info:
        validate_curriculum_structure(valid_minimal_config)
    
    error_msg = str(exc_info.value)
    assert "test_slice" in error_msg
    assert "success_metric" in error_msg.lower()


def test_validate_rejects_metric_missing_kind(valid_minimal_config):
    """Test that validation rejects success_metric without kind."""
    del valid_minimal_config['slices']['test_slice']['success_metric']['kind']
    
    with pytest.raises(CurriculumValidationError) as exc_info:
        validate_curriculum_structure(valid_minimal_config)
    
    error_msg = str(exc_info.value)
    assert "test_slice" in error_msg
    assert "kind" in error_msg.lower()


def test_validate_rejects_wrong_type_parameters(valid_minimal_config):
    """Test that validation rejects wrong types for parameters."""
    valid_minimal_config['slices']['test_slice']['parameters'] = "not a dict"
    
    with pytest.raises(CurriculumValidationError) as exc_info:
        validate_curriculum_structure(valid_minimal_config)
    
    error_msg = str(exc_info.value)
    assert "test_slice" in error_msg
    assert "parameters" in error_msg.lower()
    assert "dict" in error_msg.lower()


def test_validate_accepts_optional_fields(valid_full_config):
    """Test that validation accepts all optional fields when valid."""
    # Should not raise
    validate_curriculum_structure(valid_full_config)


def test_error_messages_are_clear_and_specific():
    """Test that validation errors are clear and actionable."""
    bad_config = {
        'schema_version': 'wrong-version',
        'slices': {
            'bad_slice': {
                'description': 'Test',
                'parameters': 'not a dict',
                'success_metric': {
                    # Missing 'kind'
                    'parameters': {},
                },
            },
        },
    }
    
    with pytest.raises(CurriculumValidationError) as exc_info:
        validate_curriculum_structure(bad_config)
    
    error_msg = str(exc_info.value)
    
    # Should mention all problems
    assert "schema_version" in error_msg or "wrong-version" in error_msg
    assert "version" in error_msg.lower()  # Missing version field
    assert "bad_slice" in error_msg
    assert "parameters" in error_msg.lower()
    assert "kind" in error_msg.lower()


# =============================================================================
# CurriculumLoaderV2 Tests
# =============================================================================


def test_loader_from_yaml_success(valid_full_config, tmp_path):
    """Test successful loading from YAML file."""
    yaml_path = tmp_path / "test_curriculum.yaml"
    
    # Write YAML
    import yaml
    with open(yaml_path, 'w') as f:
        yaml.dump(valid_full_config, f)
    
    # Load
    loader = CurriculumLoaderV2.from_yaml(yaml_path)
    
    assert loader.schema_version == 'phase2-v1'
    assert len(loader.slices) == 2
    assert 'slice_a' in loader.slices
    assert 'slice_b' in loader.slices


def test_loader_from_yaml_validates_structure(valid_minimal_config, tmp_path):
    """Test that loader validates structure on load."""
    # Make config invalid
    del valid_minimal_config['schema_version']
    
    yaml_path = tmp_path / "invalid.yaml"
    import yaml
    with open(yaml_path, 'w') as f:
        yaml.dump(valid_minimal_config, f)
    
    with pytest.raises(CurriculumValidationError):
        CurriculumLoaderV2.from_yaml(yaml_path)


def test_loader_from_yaml_raises_on_missing_file():
    """Test that loader raises FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError):
        CurriculumLoaderV2.from_yaml(Path("/nonexistent/file.yaml"))


def test_loader_get_slice_returns_uplift_slice(valid_minimal_config, tmp_path):
    """Test that get_slice returns UpliftSlice instance."""
    yaml_path = tmp_path / "test.yaml"
    import yaml
    with open(yaml_path, 'w') as f:
        yaml.dump(valid_minimal_config, f)
    
    loader = CurriculumLoaderV2.from_yaml(yaml_path)
    slice_obj = loader.get_slice('test_slice')
    
    assert isinstance(slice_obj, UpliftSlice)
    assert slice_obj.name == 'test_slice'
    assert slice_obj.description == 'Test slice'
    assert isinstance(slice_obj.success_metric, SuccessMetricSpec)


def test_loader_get_slice_returns_none_for_unknown():
    """Test that get_slice returns None for unknown slice."""
    loader = CurriculumLoaderV2(
        slices={},
        schema_version='phase2-v1',
        raw_config={},
    )
    
    assert loader.get_slice('unknown') is None


def test_loader_list_slice_names_sorted(valid_full_config, tmp_path):
    """Test that list_slice_names returns sorted list."""
    yaml_path = tmp_path / "test.yaml"
    import yaml
    with open(yaml_path, 'w') as f:
        yaml.dump(valid_full_config, f)
    
    loader = CurriculumLoaderV2.from_yaml(yaml_path)
    names = loader.list_slice_names()
    
    assert names == ['slice_a', 'slice_b']
    assert names == sorted(names)


def test_loader_get_metric_kinds(valid_full_config, tmp_path):
    """Test that get_metric_kinds returns unique metric kinds."""
    yaml_path = tmp_path / "test.yaml"
    import yaml
    with open(yaml_path, 'w') as f:
        yaml.dump(valid_full_config, f)
    
    loader = CurriculumLoaderV2.from_yaml(yaml_path)
    kinds = loader.get_metric_kinds()
    
    assert kinds == {'goal_hit', 'sparse_success'}


# =============================================================================
# UpliftSlice and SuccessMetricSpec Tests
# =============================================================================


def test_success_metric_spec_validation():
    """Test SuccessMetricSpec validation."""
    # Valid spec
    spec = SuccessMetricSpec(
        kind='goal_hit',
        parameters={'min_goal_hits': 1},
    )
    assert spec.kind == 'goal_hit'
    
    # Invalid kind (empty)
    with pytest.raises(ValueError) as exc_info:
        SuccessMetricSpec(kind='', parameters={})
    assert "kind" in str(exc_info.value).lower()
    
    # Invalid parameters (not dict)
    with pytest.raises(ValueError):
        SuccessMetricSpec(kind='goal_hit', parameters="not a dict")  # type: ignore


def test_success_metric_spec_to_dict():
    """Test SuccessMetricSpec serialization."""
    spec = SuccessMetricSpec(
        kind='goal_hit',
        parameters={'min_goal_hits': 1, 'min_total_verified': 3},
        target_hashes={'abc123', 'def456'},
    )
    
    data = spec.to_dict()
    
    assert data['kind'] == 'goal_hit'
    assert data['parameters'] == {'min_goal_hits': 1, 'min_total_verified': 3}
    assert set(data['target_hashes']) == {'abc123', 'def456'}
    # Target hashes should be sorted
    assert data['target_hashes'] == sorted(data['target_hashes'])


def test_success_metric_spec_from_dict():
    """Test SuccessMetricSpec deserialization."""
    data = {
        'kind': 'sparse_success',
        'parameters': {'min_verified': 5},
        'target_hashes': ['hash1', 'hash2'],
    }
    
    spec = SuccessMetricSpec.from_dict(data)
    
    assert spec.kind == 'sparse_success'
    assert spec.parameters == {'min_verified': 5}
    assert spec.target_hashes == {'hash1', 'hash2'}


def test_uplift_slice_validation():
    """Test UpliftSlice validation."""
    metric = SuccessMetricSpec(kind='goal_hit', parameters={})
    
    # Valid slice
    slice_obj = UpliftSlice(
        name='test',
        description='Test slice',
        parameters={'atoms': 4},
        success_metric=metric,
    )
    assert slice_obj.name == 'test'
    
    # Invalid name (empty)
    with pytest.raises(ValueError) as exc_info:
        UpliftSlice(
            name='',
            description='Test',
            parameters={},
            success_metric=metric,
        )
    assert "name" in str(exc_info.value).lower()
    
    # Invalid parameters (not dict)
    with pytest.raises(ValueError):
        UpliftSlice(
            name='test',
            description='Test',
            parameters="not a dict",  # type: ignore
            success_metric=metric,
        )
    
    # Invalid success_metric (not SuccessMetricSpec)
    with pytest.raises(ValueError):
        UpliftSlice(
            name='test',
            description='Test',
            parameters={},
            success_metric={'kind': 'goal_hit'},  # type: ignore
        )


def test_uplift_slice_from_dict():
    """Test UpliftSlice deserialization."""
    data = {
        'description': 'Test slice',
        'parameters': {'atoms': 4, 'depth_max': 5},
        'success_metric': {
            'kind': 'goal_hit',
            'parameters': {'min_goal_hits': 1},
        },
        'uplift': {'phase': 'II'},
        'budget': {'max_cycles_per_run': 500},
        'formula_pool_entries': ['p', 'q'],
    }
    
    slice_obj = UpliftSlice.from_dict('test_slice', data)
    
    assert slice_obj.name == 'test_slice'
    assert slice_obj.description == 'Test slice'
    assert slice_obj.parameters == {'atoms': 4, 'depth_max': 5}
    assert slice_obj.success_metric.kind == 'goal_hit'
    assert slice_obj.uplift == {'phase': 'II'}
    assert slice_obj.budget == {'max_cycles_per_run': 500}
    assert slice_obj.formula_pool_entries == ['p', 'q']


def test_uplift_slice_from_dict_raises_on_missing_metric():
    """Test that from_dict raises error for missing success_metric."""
    data = {
        'description': 'Test',
        'parameters': {},
    }
    
    with pytest.raises(ValueError) as exc_info:
        UpliftSlice.from_dict('test_slice', data)
    
    error_msg = str(exc_info.value)
    assert "test_slice" in error_msg
    assert "success_metric" in error_msg.lower()


# =============================================================================
# Integration Tests
# =============================================================================


def test_load_actual_phase2_config_if_exists():
    """Test loading actual Phase II config if it exists."""
    try:
        loader = CurriculumLoaderV2.from_default_phase2_config()
        
        # Should have loaded successfully
        assert loader.schema_version in ALLOWED_SCHEMA_VERSIONS
        assert len(loader.slices) > 0
        
        # Should have expected Phase II slices
        expected_slices = [
            'slice_uplift_goal',
            'slice_uplift_sparse',
            'slice_uplift_tree',
            'slice_uplift_dependency',
        ]
        
        for expected in expected_slices:
            assert expected in loader.slices, f"Expected slice '{expected}' not found"
        
        # All slices should have valid metric specs
        for slice_name, slice_obj in loader.slices.items():
            assert isinstance(slice_obj.success_metric, SuccessMetricSpec)
            assert slice_obj.success_metric.kind  # Non-empty
            
        print(f"✓ Loaded {len(loader.slices)} slices from Phase II config")
        
    except FileNotFoundError:
        # Config doesn't exist yet, skip test
        pytest.skip("Phase II config not found")


def test_loader_preserves_all_fields(valid_full_config, tmp_path):
    """Test that loader preserves all fields from YAML."""
    yaml_path = tmp_path / "test.yaml"
    import yaml
    with open(yaml_path, 'w') as f:
        yaml.dump(valid_full_config, f)
    
    loader = CurriculumLoaderV2.from_yaml(yaml_path)
    slice_a = loader.get_slice('slice_a')
    
    assert slice_a is not None
    assert slice_a.parameters == {
        'atoms': 4,
        'depth_max': 5,
        'breadth_max': 40,
    }
    assert slice_a.uplift == {
        'phase': 'II',
        'experiment_family': 'U2',
    }
    assert slice_a.budget == {
        'max_candidates_per_cycle': 40,
        'max_cycles_per_run': 500,
    }
    assert slice_a.formula_pool_entries == ['p', 'q', 'p->q']
