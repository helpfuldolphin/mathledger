"""
Tests for Hermetic Verifier v2.

Validates RFC 8785 canonicalization, byte-identical comparison,
multi-lane validation, and fleet state archival.
"""

import os
import json
import pytest
import tempfile
from pathlib import Path

os.environ['NO_NETWORK'] = 'true'

from backend.testing.hermetic_v2 import (
    RFC8785Canonicalizer,
    ReplayLogEntryV2,
    ByteIdenticalComparator,
    LaneValidationResult,
    MultiLaneValidator,
    FleetStateArchiver,
    HermeticVerifierV2,
    validate_hermetic_v2,
    generate_replay_manifest_v2,
)


class TestRFC8785Canonicalizer:
    """Test RFC 8785 canonical JSON serialization."""
    
    def test_simple_object_canonicalization(self):
        """Test canonicalizing simple object with key ordering."""
        data = {'b': 2, 'a': 1}
        canonical = RFC8785Canonicalizer.canonicalize_str(data)
        assert canonical == '{"a":1,"b":2}'
    
    def test_nested_object_canonicalization(self):
        """Test canonicalizing nested objects."""
        data = {'z': {'y': 3, 'x': 2}, 'a': 1}
        canonical = RFC8785Canonicalizer.canonicalize_str(data)
        assert canonical == '{"a":1,"z":{"x":2,"y":3}}'
    
    def test_array_canonicalization(self):
        """Test canonicalizing arrays."""
        data = {'arr': [3, 1, 2], 'key': 'value'}
        canonical = RFC8785Canonicalizer.canonicalize_str(data)
        assert canonical == '{"arr":[3,1,2],"key":"value"}'
    
    def test_primitive_types(self):
        """Test canonicalizing primitive types."""
        assert RFC8785Canonicalizer.canonicalize_str(None) == 'null'
        assert RFC8785Canonicalizer.canonicalize_str(True) == 'true'
        assert RFC8785Canonicalizer.canonicalize_str(False) == 'false'
        assert RFC8785Canonicalizer.canonicalize_str(42) == '42'
        assert RFC8785Canonicalizer.canonicalize_str('test') == '"test"'
    
    def test_hash_consistency(self):
        """Test hash consistency across equivalent objects."""
        data1 = {'b': 2, 'a': 1}
        data2 = {'a': 1, 'b': 2}
        
        hash1 = RFC8785Canonicalizer.hash_canonical(data1)
        hash2 = RFC8785Canonicalizer.hash_canonical(data2)
        
        assert hash1 == hash2
    
    def test_hash_uniqueness(self):
        """Test hash uniqueness for different objects."""
        data1 = {'a': 1, 'b': 2}
        data2 = {'a': 1, 'b': 3}
        
        hash1 = RFC8785Canonicalizer.hash_canonical(data1)
        hash2 = RFC8785Canonicalizer.hash_canonical(data2)
        
        assert hash1 != hash2
    
    def test_canonicalize_bytes(self):
        """Test canonicalize returns UTF-8 bytes."""
        data = {'a': 1}
        canonical_bytes = RFC8785Canonicalizer.canonicalize(data)
        
        assert isinstance(canonical_bytes, bytes)
        assert canonical_bytes == b'{"a":1}'
    
    def test_string_escaping(self):
        """Test proper string escaping."""
        data = {'key': 'value with "quotes"'}
        canonical = RFC8785Canonicalizer.canonicalize_str(data)
        assert '"value with \\"quotes\\""' in canonical


class TestReplayLogEntryV2:
    """Test enhanced replay log entry."""
    
    def test_entry_creation(self):
        """Test creating replay log entry."""
        entry = ReplayLogEntryV2(
            timestamp='2025-01-01T00:00:00',
            operation='test_op',
            inputs={'x': 1},
            outputs={'y': 2},
            duration_ms=10.5
        )
        
        assert entry.operation == 'test_op'
        assert entry.inputs == {'x': 1}
        assert entry.outputs == {'y': 2}
    
    def test_canonical_hash_computation(self):
        """Test canonical hash computation."""
        entry = ReplayLogEntryV2(
            timestamp='2025-01-01T00:00:00',
            operation='test_op',
            inputs={'x': 1},
            outputs={'y': 2},
            duration_ms=10.5
        )
        
        hash1 = entry.compute_canonical_hash()
        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA256 hex digest
    
    def test_canonical_hash_consistency(self):
        """Test canonical hash is consistent across identical entries."""
        entry1 = ReplayLogEntryV2(
            timestamp='2025-01-01T00:00:00',
            operation='test_op',
            inputs={'x': 1},
            outputs={'y': 2},
            duration_ms=10.5
        )
        
        entry2 = ReplayLogEntryV2(
            timestamp='2025-01-01T00:00:01',  # Different timestamp
            operation='test_op',
            inputs={'x': 1},
            outputs={'y': 2},
            duration_ms=10.5
        )
        
        assert entry1.compute_canonical_hash() == entry2.compute_canonical_hash()
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        entry = ReplayLogEntryV2(
            timestamp='2025-01-01T00:00:00',
            operation='test_op',
            inputs={'x': 1},
            outputs={'y': 2},
            duration_ms=10.5,
            canonical_hash='abc123'
        )
        
        data = entry.to_dict()
        assert data['operation'] == 'test_op'
        assert data['canonical_hash'] == 'abc123'
    
    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            'timestamp': '2025-01-01T00:00:00',
            'operation': 'test_op',
            'inputs': {'x': 1},
            'outputs': {'y': 2},
            'duration_ms': 10.5,
            'canonical_hash': 'abc123'
        }
        
        entry = ReplayLogEntryV2.from_dict(data)
        assert entry.operation == 'test_op'
        assert entry.canonical_hash == 'abc123'


class TestByteIdenticalComparator:
    """Test byte-identical replay log comparison."""
    
    def test_comparator_initialization(self):
        """Test comparator initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            comparator = ByteIdenticalComparator(tmpdir)
            assert comparator.log_dir.exists()
            assert len(comparator.current_log) == 0
    
    def test_record_operation(self):
        """Test recording operation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            comparator = ByteIdenticalComparator(tmpdir)
            comparator.record_operation('test_op', {'x': 1}, {'y': 2}, 10.5)
            
            assert len(comparator.current_log) == 1
            assert comparator.current_log[0].operation == 'test_op'
            assert comparator.current_log[0].canonical_hash is not None
    
    def test_save_and_load_log(self):
        """Test saving and loading log."""
        with tempfile.TemporaryDirectory() as tmpdir:
            comparator = ByteIdenticalComparator(tmpdir)
            comparator.record_operation('op1', {'x': 1}, {'y': 2}, 10.0)
            comparator.record_operation('op2', {'x': 2}, {'y': 4}, 15.0)
            
            comparator.save_log('test_log')
            
            entries, overall_hash = comparator.load_log('test_log')
            assert len(entries) == 2
            assert entries[0].operation == 'op1'
            assert entries[1].operation == 'op2'
            assert overall_hash != ''
    
    def test_compare_identical_logs(self):
        """Test comparing byte-identical logs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            comp1 = ByteIdenticalComparator(tmpdir)
            comp1.record_operation('op1', {'x': 1}, {'y': 2}, 10.0)
            comp1.save_log('log1')
            
            comp2 = ByteIdenticalComparator(tmpdir)
            comp2.record_operation('op1', {'x': 1}, {'y': 2}, 10.0)
            comp2.save_log('log2')
            
            is_identical, diffs = comp1.compare_byte_identical('log1', 'log2')
            assert is_identical is True
            assert len(diffs) == 0
    
    def test_compare_different_logs(self):
        """Test comparing different logs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            comp1 = ByteIdenticalComparator(tmpdir)
            comp1.record_operation('op1', {'x': 1}, {'y': 2}, 10.0)
            comp1.save_log('log1')
            
            comp2 = ByteIdenticalComparator(tmpdir)
            comp2.record_operation('op1', {'x': 1}, {'y': 3}, 10.0)  # Different output
            comp2.save_log('log2')
            
            is_identical, diffs = comp1.compare_byte_identical('log1', 'log2')
            assert is_identical is False
            assert len(diffs) > 0
            assert any('hash mismatch' in diff.lower() for diff in diffs)
    
    def test_overall_hash_mismatch_detection(self):
        """Test overall hash mismatch is detected first."""
        with tempfile.TemporaryDirectory() as tmpdir:
            comp1 = ByteIdenticalComparator(tmpdir)
            comp1.record_operation('op1', {'x': 1}, {'y': 2}, 10.0)
            comp1.save_log('log1')
            
            comp2 = ByteIdenticalComparator(tmpdir)
            comp2.record_operation('op1', {'x': 1}, {'y': 3}, 10.0)
            comp2.save_log('log2')
            
            is_identical, diffs = comp1.compare_byte_identical('log1', 'log2')
            assert is_identical is False
            assert any('Overall hash mismatch' in diff for diff in diffs)


class TestLaneValidationResult:
    """Test lane validation result."""
    
    def test_result_creation(self):
        """Test creating lane validation result."""
        result = LaneValidationResult(
            lane_name='test',
            hermetic=True,
            checks={'check1': True},
            timestamp='2025-01-01T00:00:00',
            canonical_hash='abc123'
        )
        
        assert result.lane_name == 'test'
        assert result.hermetic is True
        assert result.canonical_hash == 'abc123'


class TestMultiLaneValidator:
    """Test multi-lane hermetic validation."""
    
    def test_validator_initialization(self):
        """Test validator initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = MultiLaneValidator(tmpdir)
            assert validator.artifacts_dir.exists()
            assert len(validator.lane_results) == 0
    
    def test_validate_lane(self):
        """Test validating single lane."""
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = MultiLaneValidator(tmpdir)
            checks = {
                'no_network': True,
                'deterministic': True,
                'reproducible': True
            }
            
            result = validator.validate_lane('test_lane', checks)
            
            assert result.lane_name == 'test_lane'
            assert result.hermetic is True
            assert result.canonical_hash != ''
    
    def test_validate_all_lanes(self):
        """Test validating all CI lanes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = MultiLaneValidator(tmpdir)
            lane_results = validator.validate_all_lanes()
            
            assert len(lane_results) >= 6
            assert 'dual-attestation' in lane_results
            assert 'test' in lane_results
            assert 'uplift-omega' in lane_results
    
    def test_all_lanes_hermetic(self):
        """Test checking if all lanes are hermetic."""
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = MultiLaneValidator(tmpdir)
            validator.validate_all_lanes()
            
            assert validator.all_lanes_hermetic() is True
    
    def test_generate_lane_report(self):
        """Test generating lane report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = MultiLaneValidator(tmpdir)
            validator.validate_all_lanes()
            
            report = validator.generate_lane_report()
            
            assert 'version' in report
            assert report['version'] == '2.0'
            assert 'all_lanes_hermetic' in report
            assert 'overall_canonical_hash' in report
            assert 'lanes' in report


class TestFleetStateArchiver:
    """Test fleet state archival."""
    
    def test_archiver_initialization(self):
        """Test archiver initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            archiver = FleetStateArchiver(tmpdir)
            assert archiver.archive_dir.exists()
    
    def test_detect_all_blue(self):
        """Test detecting ALL BLUE condition."""
        with tempfile.TemporaryDirectory() as tmpdir:
            archiver = FleetStateArchiver(tmpdir)
            
            lane_results = {
                'lane1': LaneValidationResult('lane1', True, {}, '', 'hash1'),
                'lane2': LaneValidationResult('lane2', True, {}, '', 'hash2'),
            }
            
            assert archiver.detect_all_blue(lane_results) is True
            
            lane_results['lane3'] = LaneValidationResult('lane3', False, {}, '', 'hash3')
            assert archiver.detect_all_blue(lane_results) is False
    
    def test_freeze_fleet_state(self):
        """Test freezing fleet state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            archiver = FleetStateArchiver(tmpdir)
            
            lane_results = {
                'lane1': LaneValidationResult('lane1', True, {}, '2025-01-01T00:00:00', 'hash1'),
            }
            
            fleet_state = archiver.freeze_fleet_state(
                lane_results,
                {'log1': 'hash_abc'},
                {'test': True}
            )
            
            assert fleet_state['all_blue'] is True
            assert 'state_hash' in fleet_state
            assert 'lanes' in fleet_state
            assert 'replay_logs' in fleet_state
            assert fleet_state['replay_logs']['log1'] == 'hash_abc'
    
    def test_sign_and_archive(self):
        """Test signing and archiving fleet state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            archiver = FleetStateArchiver(tmpdir)
            
            fleet_state = {
                'version': '2.0',
                'all_blue': True,
                'lanes': {},
                'state_hash': 'test_hash'
            }
            
            archive_path = archiver.sign_and_archive(fleet_state)
            
            assert Path(archive_path).exists()
            assert 'fleet_state_' in archive_path
            
            latest_link = archiver.archive_dir / 'fleet_state.json'
            assert latest_link.exists()
    
    def test_verify_archived_state(self):
        """Test verifying archived fleet state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            archiver = FleetStateArchiver(tmpdir)
            
            lane_results = {
                'lane1': LaneValidationResult('lane1', True, {}, '2025-01-01T00:00:00', 'hash1'),
            }
            
            fleet_state = archiver.freeze_fleet_state(lane_results, {}, {})
            archive_path = archiver.sign_and_archive(fleet_state)
            
            is_valid, message = archiver.verify_archived_state(archive_path)
            
            assert is_valid is True
            assert 'State verified' in message


class TestHermeticVerifierV2:
    """Test hermetic verifier v2."""
    
    def test_verifier_initialization(self):
        """Test verifier initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            verifier = HermeticVerifierV2(tmpdir)
            assert verifier.artifacts_dir.exists()
            assert verifier.comparator is not None
            assert verifier.lane_validator is not None
            assert verifier.archiver is not None
    
    def test_run_full_verification(self):
        """Test running full verification."""
        with tempfile.TemporaryDirectory() as tmpdir:
            verifier = HermeticVerifierV2(tmpdir)
            results = verifier.run_full_verification()
            
            assert 'version' in results
            assert results['version'] == '2.0'
            assert 'checks' in results
            assert 'hermetic_v2' in results
            assert 'status' in results
            assert results['status'] in ['PASS', 'FAIL']
    
    def test_archive_on_all_blue(self):
        """Test archiving on ALL BLUE."""
        with tempfile.TemporaryDirectory() as tmpdir:
            verifier = HermeticVerifierV2(tmpdir)
            verifier.run_full_verification()
            
            archive_path = verifier.archive_on_all_blue(
                replay_logs={'test': 'hash123'},
                metadata={'test': True}
            )
            
            assert archive_path is not None
            assert Path(archive_path).exists()


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_validate_hermetic_v2(self):
        """Test validate_hermetic_v2 function."""
        is_hermetic_v2, results = validate_hermetic_v2()
        
        assert isinstance(is_hermetic_v2, bool)
        assert isinstance(results, dict)
        assert 'checks' in results
        assert 'hermetic_v2' in results
    
    def test_generate_replay_manifest_v2(self):
        """Test generate_replay_manifest_v2 function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / 'replay_manifest_v2.json'
            manifest = generate_replay_manifest_v2(str(manifest_path))
            
            assert manifest_path.exists()
            assert manifest['hermetic_v2'] is True
            assert manifest['rfc8785_canonical'] is True
            assert 'components' in manifest
            assert 'validation' in manifest


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    def test_full_hermetic_v2_workflow(self):
        """Test complete hermetic v2 workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            verifier = HermeticVerifierV2(tmpdir)
            
            verifier.comparator.record_operation('build', {}, {'success': True}, 1000.0)
            verifier.comparator.record_operation('test', {}, {'passed': 10}, 5000.0)
            verifier.comparator.save_log('build_log')
            
            results = verifier.run_full_verification()
            
            assert results['hermetic_v2'] is True
            assert results['checks']['all_blue'] is True
            
            archive_path = verifier.archive_on_all_blue(
                replay_logs={'build': 'hash123'},
                metadata={'build_id': '12345'}
            )
            
            assert archive_path is not None
            
            is_valid, message = verifier.archiver.verify_archived_state(archive_path)
            assert is_valid is True
    
    def test_byte_identical_reproducibility(self):
        """Test byte-identical reproducibility across runs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            comparator = ByteIdenticalComparator(tmpdir)
            
            comparator.record_operation('compile', {'file': 'main.py'}, {'output': 'main.pyc'}, 100.0)
            comparator.record_operation('test', {'suite': 'unit'}, {'passed': 5}, 200.0)
            comparator.save_log('run1')
            
            comparator.current_log = []
            comparator.record_operation('compile', {'file': 'main.py'}, {'output': 'main.pyc'}, 100.0)
            comparator.record_operation('test', {'suite': 'unit'}, {'passed': 5}, 200.0)
            comparator.save_log('run2')
            
            is_identical, diffs = comparator.compare_byte_identical('run1', 'run2')
            assert is_identical is True
            assert len(diffs) == 0
    
    def test_multi_lane_all_blue_detection(self):
        """Test ALL BLUE detection across multiple lanes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = MultiLaneValidator(tmpdir)
            archiver = FleetStateArchiver(tmpdir)
            
            lane_results = validator.validate_all_lanes()
            
            all_blue = archiver.detect_all_blue(lane_results)
            assert all_blue is True
            
            fleet_state = archiver.freeze_fleet_state(lane_results, {}, {})
            archive_path = archiver.sign_and_archive(fleet_state)
            
            is_valid, message = archiver.verify_archived_state(archive_path)
            assert is_valid is True
