"""
Tests for Phase II curriculum loader and drift detection.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from curriculum.phase2_loader import (
    CurriculumLoaderV2,
    SuccessMetricSpec,
    UpliftSlice,
    CurriculumFingerprint,
    compute_curriculum_diff,
)


@pytest.fixture
def sample_slice_data():
    """Sample slice configuration."""
    return {
        "description": "Test goal-conditioned slice",
        "parameters": {
            "atoms": 4,
            "depth_max": 5,
            "breadth_max": 40
        },
        "success_metric": {
            "kind": "goal_hit",
            "parameters": {
                "min_goal_hits": 1,
                "min_total_verified": 3
            }
        },
        "uplift": {
            "phase": "II",
            "experiment_family": "U2"
        },
        "budget": {
            "max_candidates_per_cycle": 40
        }
    }


@pytest.fixture
def sample_curriculum_config():
    """Sample curriculum configuration."""
    return {
        "version": "2.1.0",
        "slices": {
            "slice_uplift_goal": {
                "description": "Goal-conditioned slice",
                "parameters": {
                    "atoms": 4,
                    "depth_max": 5
                },
                "success_metric": {
                    "kind": "goal_hit",
                    "parameters": {
                        "min_goal_hits": 1
                    }
                },
                "uplift": {
                    "phase": "II"
                }
            },
            "slice_uplift_sparse": {
                "description": "Sparse reward slice",
                "parameters": {
                    "atoms": 5,
                    "depth_max": 6
                },
                "success_metric": {
                    "kind": "sparse_reward",
                    "parameters": {
                        "min_verified": 5
                    }
                },
                "uplift": {
                    "phase": "II"
                }
            }
        }
    }


class TestSuccessMetricSpec:
    """Tests for SuccessMetricSpec."""
    
    def test_valid_metric_kinds(self):
        """Test all valid metric kinds."""
        valid_kinds = [
            "goal_hit", "multi_goal_success",
            "sparse_reward", "sparse_success",
            "tree_depth", "chain_depth", "chain_success",
            "dependency_coordination", "dependency_success"
        ]
        for kind in valid_kinds:
            spec = SuccessMetricSpec(kind=kind)
            assert spec.kind == kind
    
    def test_invalid_metric_kind(self):
        """Test invalid metric kind raises error."""
        with pytest.raises(ValueError, match="Invalid success metric kind"):
            SuccessMetricSpec(kind="invalid_kind")
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        spec = SuccessMetricSpec(
            kind="goal_hit",
            parameters={"min_goal_hits": 1}
        )
        result = spec.to_dict()
        assert result["kind"] == "goal_hit"
        assert result["parameters"]["min_goal_hits"] == 1


class TestUpliftSlice:
    """Tests for UpliftSlice."""
    
    def test_from_dict(self, sample_slice_data):
        """Test parsing slice from dictionary."""
        slice_obj = UpliftSlice.from_dict("test_slice", sample_slice_data)
        
        assert slice_obj.name == "test_slice"
        assert slice_obj.description == "Test goal-conditioned slice"
        assert slice_obj.parameters["atoms"] == 4
        assert slice_obj.success_metric.kind == "goal_hit"
        assert slice_obj.uplift["phase"] == "II"
    
    def test_missing_success_metric(self):
        """Test error when success_metric is missing."""
        data = {
            "description": "Test",
            "parameters": {"atoms": 4}
        }
        with pytest.raises(ValueError, match="missing required 'success_metric'"):
            UpliftSlice.from_dict("test", data)
    
    def test_missing_parameters(self):
        """Test error when parameters are missing."""
        data = {
            "description": "Test",
            "success_metric": {
                "kind": "goal_hit"
            }
        }
        with pytest.raises(ValueError, match="missing required 'parameters'"):
            UpliftSlice.from_dict("test", data)
    
    def test_to_dict(self, sample_slice_data):
        """Test conversion to dictionary."""
        slice_obj = UpliftSlice.from_dict("test_slice", sample_slice_data)
        result = slice_obj.to_dict()
        
        assert result["name"] == "test_slice"
        assert result["parameters"]["atoms"] == 4
        assert result["success_metric"]["kind"] == "goal_hit"


class TestCurriculumLoaderV2:
    """Tests for CurriculumLoaderV2."""
    
    def test_validate_config_valid(self, sample_curriculum_config):
        """Test validation passes for valid config."""
        errors = CurriculumLoaderV2.validate_config(sample_curriculum_config)
        assert errors == []
    
    def test_validate_config_missing_version(self):
        """Test validation catches missing version."""
        config = {"slices": {}}
        errors = CurriculumLoaderV2.validate_config(config)
        assert any("version" in e.lower() for e in errors)
    
    def test_validate_config_wrong_version(self):
        """Test validation catches wrong version."""
        config = {"version": "1.0.0", "slices": {}}
        errors = CurriculumLoaderV2.validate_config(config)
        assert any("version 2" in e.lower() for e in errors)
    
    def test_validate_config_missing_slices(self):
        """Test validation catches missing slices."""
        config = {"version": "2.1.0"}
        errors = CurriculumLoaderV2.validate_config(config)
        assert any("slices" in e.lower() for e in errors)
    
    def test_from_dict(self, sample_curriculum_config):
        """Test loading from dictionary."""
        curriculum = CurriculumLoaderV2.from_dict(sample_curriculum_config)
        
        assert curriculum.schema_version == "phase2-v2.1.0"
        assert len(curriculum.slices) == 2
        assert curriculum.slices[0].name in ["slice_uplift_goal", "slice_uplift_sparse"]
    
    def test_list_slices(self, sample_curriculum_config):
        """Test listing slice names."""
        curriculum = CurriculumLoaderV2.from_dict(sample_curriculum_config)
        names = curriculum.list_slices()
        
        assert len(names) == 2
        assert "slice_uplift_goal" in names
        assert "slice_uplift_sparse" in names
    
    def test_get_slice(self, sample_curriculum_config):
        """Test getting slice by name."""
        curriculum = CurriculumLoaderV2.from_dict(sample_curriculum_config)
        
        slice_obj = curriculum.get_slice("slice_uplift_goal")
        assert slice_obj is not None
        assert slice_obj.name == "slice_uplift_goal"
        
        missing = curriculum.get_slice("nonexistent")
        assert missing is None
    
    def test_show_slice(self, sample_curriculum_config):
        """Test showing slice details."""
        curriculum = CurriculumLoaderV2.from_dict(sample_curriculum_config)
        
        details = curriculum.show_slice("slice_uplift_goal")
        assert details["name"] == "slice_uplift_goal"
        assert "parameters" in details
        assert "success_metric" in details
    
    def test_show_slice_not_found(self, sample_curriculum_config):
        """Test error when showing nonexistent slice."""
        curriculum = CurriculumLoaderV2.from_dict(sample_curriculum_config)
        
        with pytest.raises(ValueError, match="not found"):
            curriculum.show_slice("nonexistent")
    
    def test_show_metrics(self, sample_curriculum_config):
        """Test showing all metrics."""
        curriculum = CurriculumLoaderV2.from_dict(sample_curriculum_config)
        
        metrics = curriculum.show_metrics()
        assert "slice_uplift_goal" in metrics
        assert metrics["slice_uplift_goal"]["kind"] == "goal_hit"


class TestCurriculumFingerprint:
    """Tests for CurriculumFingerprint."""
    
    def test_generate(self, sample_curriculum_config):
        """Test fingerprint generation."""
        curriculum = CurriculumLoaderV2.from_dict(sample_curriculum_config)
        fingerprint = CurriculumFingerprint.generate(curriculum)
        
        assert fingerprint.schema_version == "phase2-v2.1.0"
        assert fingerprint.slice_count == 2
        assert len(fingerprint.slice_names) == 2
        assert len(fingerprint.slice_fingerprints) == 2
        assert len(fingerprint.sha256) == 64  # SHA-256 hex
    
    def test_generate_with_run_id(self, sample_curriculum_config):
        """Test fingerprint with run ID."""
        curriculum = CurriculumLoaderV2.from_dict(sample_curriculum_config)
        fingerprint = CurriculumFingerprint.generate(curriculum, run_id="test-run-123")
        
        assert "@test-run-123" in fingerprint.timestamp
    
    def test_deterministic_hash(self, sample_curriculum_config):
        """Test that same config produces same hash."""
        curriculum = CurriculumLoaderV2.from_dict(sample_curriculum_config)
        fp1 = CurriculumFingerprint.generate(curriculum, run_id="fixed")
        fp2 = CurriculumFingerprint.generate(curriculum, run_id="fixed")
        
        # SHA256 should be same for identical config
        assert fp1.sha256 == fp2.sha256
    
    def test_save_and_load(self, sample_curriculum_config):
        """Test saving and loading fingerprint."""
        curriculum = CurriculumLoaderV2.from_dict(sample_curriculum_config)
        fingerprint = CurriculumFingerprint.generate(curriculum)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            fingerprint.save(temp_path)
            loaded = CurriculumFingerprint.load_from_file(temp_path)
            
            assert loaded.sha256 == fingerprint.sha256
            assert loaded.slice_count == fingerprint.slice_count
            assert loaded.schema_version == fingerprint.schema_version
        finally:
            os.unlink(temp_path)


class TestComputeCurriculumDiff:
    """Tests for curriculum diff computation."""
    
    def test_no_change(self, sample_curriculum_config):
        """Test diff when nothing changed."""
        curriculum = CurriculumLoaderV2.from_dict(sample_curriculum_config)
        fp1 = CurriculumFingerprint.generate(curriculum, run_id="run1")
        fp2 = CurriculumFingerprint.generate(curriculum, run_id="run2")
        
        diff = compute_curriculum_diff(fp1, fp2)
        
        assert not diff["changed"]
        assert not diff["schema_version_changed"]
        assert diff["slices_added"] == []
        assert diff["slices_removed"] == []
        assert diff["slices_modified"] == []
    
    def test_slice_added(self, sample_curriculum_config):
        """Test detecting added slice."""
        curriculum1 = CurriculumLoaderV2.from_dict(sample_curriculum_config)
        fp1 = CurriculumFingerprint.generate(curriculum1)
        
        # Add a slice
        config2 = sample_curriculum_config.copy()
        config2["slices"] = config2["slices"].copy()
        config2["slices"]["slice_uplift_new"] = {
            "description": "New slice",
            "parameters": {"atoms": 3},
            "success_metric": {"kind": "goal_hit"},
            "uplift": {"phase": "II"}
        }
        curriculum2 = CurriculumLoaderV2.from_dict(config2)
        fp2 = CurriculumFingerprint.generate(curriculum2)
        
        diff = compute_curriculum_diff(fp1, fp2)
        
        assert diff["changed"]
        assert "slice_uplift_new" in diff["slices_added"]
    
    def test_slice_removed(self, sample_curriculum_config):
        """Test detecting removed slice."""
        curriculum1 = CurriculumLoaderV2.from_dict(sample_curriculum_config)
        fp1 = CurriculumFingerprint.generate(curriculum1)
        
        # Remove a slice
        config2 = sample_curriculum_config.copy()
        config2["slices"] = {"slice_uplift_goal": config2["slices"]["slice_uplift_goal"]}
        curriculum2 = CurriculumLoaderV2.from_dict(config2)
        fp2 = CurriculumFingerprint.generate(curriculum2)
        
        diff = compute_curriculum_diff(fp1, fp2)
        
        assert diff["changed"]
        assert "slice_uplift_sparse" in diff["slices_removed"]
    
    def test_slice_modified(self, sample_curriculum_config):
        """Test detecting modified slice."""
        curriculum1 = CurriculumLoaderV2.from_dict(sample_curriculum_config)
        fp1 = CurriculumFingerprint.generate(curriculum1)
        
        # Modify a slice parameter
        config2 = sample_curriculum_config.copy()
        config2["slices"] = {k: v.copy() for k, v in config2["slices"].items()}
        config2["slices"]["slice_uplift_goal"]["parameters"] = {
            **config2["slices"]["slice_uplift_goal"]["parameters"],
            "atoms": 5  # Changed from 4
        }
        curriculum2 = CurriculumLoaderV2.from_dict(config2)
        fp2 = CurriculumFingerprint.generate(curriculum2)
        
        diff = compute_curriculum_diff(fp1, fp2)
        
        assert diff["changed"]
        assert "slice_uplift_goal" in diff["slices_modified"]
    
    def test_schema_version_changed(self, sample_curriculum_config):
        """Test detecting schema version change."""
        curriculum1 = CurriculumLoaderV2.from_dict(sample_curriculum_config)
        fp1 = CurriculumFingerprint.generate(curriculum1)
        
        # Change version
        config2 = sample_curriculum_config.copy()
        config2["version"] = "2.2.0"
        curriculum2 = CurriculumLoaderV2.from_dict(config2)
        fp2 = CurriculumFingerprint.generate(curriculum2)
        
        diff = compute_curriculum_diff(fp1, fp2)
        
        assert diff["schema_version_changed"]
        assert diff["old_schema_version"] == "phase2-v2.1.0"
        assert diff["new_schema_version"] == "phase2-v2.2.0"
