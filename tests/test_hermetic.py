"""
Tests for hermetic build framework.

Validates deterministic package installation, external API mocking,
and replay log comparison capabilities.
"""

import os
import json
import pytest
import tempfile
from pathlib import Path

os.environ['NO_NETWORK'] = 'true'

from backend.testing.hermetic import (
    PackageSpec,
    HermeticPackageManager,
    APIMockSpec,
    ExternalAPIMockRegistry,
    ReplayLogEntry,
    ReplayLogComparator,
    HermeticBuildValidator,
    validate_hermetic_build,
    generate_replay_manifest,
)


class TestPackageSpec:
    """Test package specification."""
    
    def test_package_spec_creation(self):
        """Test creating package spec."""
        pkg = PackageSpec('pytest', '8.0.0', 'abc123', 'pypi')
        assert pkg.name == 'pytest'
        assert pkg.version == '8.0.0'
        assert pkg.sha256 == 'abc123'
        
    def test_to_requirement(self):
        """Test converting to requirement string."""
        pkg = PackageSpec('pytest', '8.0.0')
        assert pkg.to_requirement() == 'pytest==8.0.0'
        
    def test_to_requirement_with_extras(self):
        """Test requirement with extras."""
        pkg = PackageSpec('psycopg', '3.2.0', extras=['binary'])
        assert pkg.to_requirement() == 'psycopg[binary]==3.2.0'


class TestHermeticPackageManager:
    """Test hermetic package manager."""
    
    def test_package_manager_initialization(self):
        """Test package manager initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = HermeticPackageManager(tmpdir)
            assert mgr.cache_dir.exists()
            assert mgr.manifest_path.exists() or not mgr.packages
            
    def test_add_package(self):
        """Test adding package to manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = HermeticPackageManager(tmpdir)
            mgr.add_package('pytest', '8.0.0', 'abc123')
            
            assert 'pytest' in mgr.packages
            assert mgr.packages['pytest'].version == '8.0.0'
            
    def test_generate_requirements(self):
        """Test generating requirements file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = HermeticPackageManager(tmpdir)
            mgr.add_package('pytest', '8.0.0')
            mgr.add_package('fastapi', '0.115.0')
            
            reqs = mgr.generate_requirements()
            assert 'pytest==8.0.0' in reqs
            assert 'fastapi==0.115.0' in reqs
            
    def test_generate_requirements_with_hashes(self):
        """Test generating requirements with hashes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = HermeticPackageManager(tmpdir)
            mgr.add_package('pytest', '8.0.0', 'abc123')
            
            reqs = mgr.generate_requirements()
            assert 'pytest==8.0.0 --hash=sha256:abc123' in reqs
            
    def test_manifest_persistence(self):
        """Test manifest persists across instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr1 = HermeticPackageManager(tmpdir)
            mgr1.add_package('pytest', '8.0.0')
            
            mgr2 = HermeticPackageManager(tmpdir)
            assert 'pytest' in mgr2.packages
            assert mgr2.packages['pytest'].version == '8.0.0'


class TestAPIMockSpec:
    """Test API mock specification."""
    
    def test_api_mock_spec_creation(self):
        """Test creating API mock spec."""
        endpoints = {
            'GET:/test': {'status': 200, 'body': 'test'}
        }
        mock = APIMockSpec('test_api', 'https://api.test.com', endpoints)
        
        assert mock.api_name == 'test_api'
        assert mock.base_url == 'https://api.test.com'
        assert 'GET:/test' in mock.endpoints


class TestExternalAPIMockRegistry:
    """Test external API mock registry."""
    
    def test_registry_initialization(self):
        """Test registry initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ExternalAPIMockRegistry(tmpdir)
            assert registry.registry_dir.exists()
            
    def test_register_api_mock(self):
        """Test registering API mock."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ExternalAPIMockRegistry(tmpdir)
            endpoints = {
                'GET:/test': {'status': 200, 'body': 'test'}
            }
            registry.register_api_mock('test_api', 'https://api.test.com', endpoints)
            
            assert 'test_api' in registry.mocks
            assert registry.mocks['test_api'].base_url == 'https://api.test.com'
            
    def test_get_mock_response(self):
        """Test getting mock response."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ExternalAPIMockRegistry(tmpdir)
            endpoints = {
                'GET:/test': {'status': 200, 'body': 'test response'}
            }
            registry.register_api_mock('test_api', 'https://api.test.com', endpoints)
            
            response = registry.get_mock_response('test_api', '/test', 'GET')
            assert response is not None
            assert response['status'] == 200
            assert response['body'] == 'test response'
            
    def test_get_mock_response_not_found(self):
        """Test getting non-existent mock response."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ExternalAPIMockRegistry(tmpdir)
            response = registry.get_mock_response('nonexistent', '/test')
            assert response is None
            
    def test_create_pypi_mock(self):
        """Test creating PyPI mock."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ExternalAPIMockRegistry(tmpdir)
            registry.create_pypi_mock()
            
            assert 'pypi' in registry.mocks
            assert registry.mocks['pypi'].base_url == 'https://pypi.org'
            
    def test_create_github_mock(self):
        """Test creating GitHub mock."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ExternalAPIMockRegistry(tmpdir)
            registry.create_github_mock()
            
            assert 'github' in registry.mocks
            assert registry.mocks['github'].base_url == 'https://api.github.com'
            assert registry.mocks['github'].auth_required is True
            
    def test_registry_persistence(self):
        """Test registry persists across instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry1 = ExternalAPIMockRegistry(tmpdir)
            endpoints = {'GET:/test': {'status': 200}}
            registry1.register_api_mock('test_api', 'https://api.test.com', endpoints)
            
            registry2 = ExternalAPIMockRegistry(tmpdir)
            assert 'test_api' in registry2.mocks


class TestReplayLogEntry:
    """Test replay log entry."""
    
    def test_replay_log_entry_creation(self):
        """Test creating replay log entry."""
        entry = ReplayLogEntry(
            timestamp='2025-01-01T00:00:00',
            operation='test_op',
            inputs={'x': 1},
            outputs={'y': 2},
            duration_ms=10.5
        )
        
        assert entry.operation == 'test_op'
        assert entry.inputs == {'x': 1}
        assert entry.outputs == {'y': 2}
        
    def test_to_dict(self):
        """Test converting to dictionary."""
        entry = ReplayLogEntry(
            timestamp='2025-01-01T00:00:00',
            operation='test_op',
            inputs={'x': 1},
            outputs={'y': 2},
            duration_ms=10.5
        )
        
        data = entry.to_dict()
        assert data['operation'] == 'test_op'
        assert data['inputs'] == {'x': 1}
        
    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            'timestamp': '2025-01-01T00:00:00',
            'operation': 'test_op',
            'inputs': {'x': 1},
            'outputs': {'y': 2},
            'duration_ms': 10.5
        }
        
        entry = ReplayLogEntry.from_dict(data)
        assert entry.operation == 'test_op'
        assert entry.inputs == {'x': 1}


class TestReplayLogComparator:
    """Test replay log comparator."""
    
    def test_comparator_initialization(self):
        """Test comparator initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            comparator = ReplayLogComparator(tmpdir)
            assert comparator.log_dir.exists()
            assert len(comparator.current_log) == 0
            
    def test_record_operation(self):
        """Test recording operation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            comparator = ReplayLogComparator(tmpdir)
            comparator.record_operation(
                'test_op',
                {'x': 1},
                {'y': 2},
                10.5
            )
            
            assert len(comparator.current_log) == 1
            assert comparator.current_log[0].operation == 'test_op'
            
    def test_save_and_load_log(self):
        """Test saving and loading log."""
        with tempfile.TemporaryDirectory() as tmpdir:
            comparator = ReplayLogComparator(tmpdir)
            comparator.record_operation('op1', {'x': 1}, {'y': 2}, 10.0)
            comparator.record_operation('op2', {'x': 2}, {'y': 4}, 15.0)
            
            comparator.save_log('test_log')
            
            loaded = comparator.load_log('test_log')
            assert len(loaded) == 2
            assert loaded[0].operation == 'op1'
            assert loaded[1].operation == 'op2'
            
    def test_compare_identical_logs(self):
        """Test comparing identical logs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            comp1 = ReplayLogComparator(tmpdir)
            comp1.record_operation('op1', {'x': 1}, {'y': 2}, 10.0)
            comp1.save_log('log1')
            
            comp2 = ReplayLogComparator(tmpdir)
            comp2.record_operation('op1', {'x': 1}, {'y': 2}, 10.0)
            comp2.save_log('log2')
            
            is_identical, diffs = comp1.compare_logs('log1', 'log2')
            assert is_identical is True
            assert len(diffs) == 0
            
    def test_compare_different_logs(self):
        """Test comparing different logs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            comp1 = ReplayLogComparator(tmpdir)
            comp1.record_operation('op1', {'x': 1}, {'y': 2}, 10.0)
            comp1.save_log('log1')
            
            comp2 = ReplayLogComparator(tmpdir)
            comp2.record_operation('op1', {'x': 1}, {'y': 3}, 10.0)  # Different output
            comp2.save_log('log2')
            
            is_identical, diffs = comp1.compare_logs('log1', 'log2')
            assert is_identical is False
            assert len(diffs) > 0
            assert any('Output mismatch' in diff for diff in diffs)
            
    def test_compare_different_length_logs(self):
        """Test comparing logs with different lengths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            comp1 = ReplayLogComparator(tmpdir)
            comp1.record_operation('op1', {'x': 1}, {'y': 2}, 10.0)
            comp1.save_log('log1')
            
            comp2 = ReplayLogComparator(tmpdir)
            comp2.record_operation('op1', {'x': 1}, {'y': 2}, 10.0)
            comp2.record_operation('op2', {'x': 2}, {'y': 4}, 15.0)
            comp2.save_log('log2')
            
            is_identical, diffs = comp1.compare_logs('log1', 'log2')
            assert is_identical is False
            assert any('Log length mismatch' in diff for diff in diffs)


class TestHermeticBuildValidator:
    """Test hermetic build validator."""
    
    def test_validator_initialization(self):
        """Test validator initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = HermeticBuildValidator(tmpdir)
            assert validator.artifacts_dir.exists()
            
    def test_validate_no_network(self):
        """Test NO_NETWORK validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = HermeticBuildValidator(tmpdir)
            assert validator.validate_no_network() is True
            
    def test_validate_package_determinism(self):
        """Test package determinism validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = HermeticBuildValidator(tmpdir)
            pkg_manager = HermeticPackageManager(tmpdir)
            
            assert validator.validate_package_determinism(pkg_manager) is False
            
            pkg_manager.add_package('pytest', '8.0.0')
            assert validator.validate_package_determinism(pkg_manager) is True
            
    def test_validate_api_isolation(self):
        """Test API isolation validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = HermeticBuildValidator(tmpdir)
            registry = ExternalAPIMockRegistry(tmpdir)
            
            assert validator.validate_api_isolation(registry) is False
            
            registry.create_pypi_mock()
            registry.create_github_mock()
            assert validator.validate_api_isolation(registry) is True
            
    def test_validate_replay_determinism(self):
        """Test replay determinism validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = HermeticBuildValidator(tmpdir)
            comparator = ReplayLogComparator(tmpdir)
            
            comparator.record_operation('op1', {'x': 1}, {'y': 2}, 10.0)
            comparator.save_log('log1')
            
            comparator.current_log = []
            comparator.record_operation('op1', {'x': 1}, {'y': 2}, 10.0)
            comparator.save_log('log2')
            
            assert validator.validate_replay_determinism(comparator, 'log1', 'log2') is True
            
    def test_run_full_validation(self):
        """Test running full validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = HermeticBuildValidator(tmpdir)
            results = validator.run_full_validation()
            
            assert 'checks' in results
            assert 'hermetic' in results
            assert 'status' in results
            assert results['status'] in ['PASS', 'FAIL']


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_validate_hermetic_build(self):
        """Test validate_hermetic_build function."""
        is_hermetic, results = validate_hermetic_build()
        
        assert isinstance(is_hermetic, bool)
        assert isinstance(results, dict)
        assert 'checks' in results
        
    def test_generate_replay_manifest(self):
        """Test generate_replay_manifest function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / 'replay_manifest.json'
            manifest = generate_replay_manifest(str(manifest_path))
            
            assert manifest_path.exists()
            assert manifest['hermetic'] is True
            assert 'components' in manifest
            assert 'validation' in manifest


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    def test_full_hermetic_workflow(self):
        """Test complete hermetic build workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pkg_manager = HermeticPackageManager(tmpdir)
            pkg_manager.add_package('pytest', '8.0.0', 'abc123')
            pkg_manager.add_package('fastapi', '0.115.0', 'def456')
            
            api_registry = ExternalAPIMockRegistry(tmpdir)
            api_registry.create_pypi_mock()
            api_registry.create_github_mock()
            
            comparator = ReplayLogComparator(tmpdir)
            comparator.record_operation('install_packages', {}, {'success': True}, 1000.0)
            comparator.record_operation('run_tests', {}, {'passed': 10}, 5000.0)
            comparator.save_log('build_log')
            
            validator = HermeticBuildValidator(tmpdir)
            results = validator.run_full_validation()
            
            assert results['checks']['no_network_enabled'] is True
            assert results['checks']['package_determinism'] is True
            assert results['checks']['api_isolation'] is True
            
    def test_deterministic_build_reproduction(self):
        """Test reproducing deterministic build."""
        with tempfile.TemporaryDirectory() as tmpdir:
            comparator = ReplayLogComparator(tmpdir)
            
            comparator.record_operation('compile', {'file': 'main.py'}, {'output': 'main.pyc'}, 100.0)
            comparator.record_operation('test', {'suite': 'unit'}, {'passed': 5}, 200.0)
            comparator.save_log('build1')
            
            comparator.current_log = []
            comparator.record_operation('compile', {'file': 'main.py'}, {'output': 'main.pyc'}, 100.0)
            comparator.record_operation('test', {'suite': 'unit'}, {'passed': 5}, 200.0)
            comparator.save_log('build2')
            
            is_identical, diffs = comparator.compare_logs('build1', 'build2')
            assert is_identical is True
            assert len(diffs) == 0
