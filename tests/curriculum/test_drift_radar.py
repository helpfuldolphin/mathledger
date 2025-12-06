"""
Tests for curriculum drift radar and promotion guard.
"""

import json
import os
import tempfile

import pytest

from curriculum.phase2_loader import (
    CurriculumLoaderV2,
    CurriculumFingerprint,
    compute_curriculum_diff,
)
from curriculum.drift_radar import (
    build_curriculum_drift_history,
    classify_curriculum_drift_event,
    evaluate_curriculum_for_promotion,
    summarize_curriculum_for_global_health,
)


@pytest.fixture
def sample_curriculum_config():
    """Sample curriculum configuration."""
    return {
        "version": "2.1.0",
        "slices": {
            "slice_a": {
                "description": "Slice A",
                "parameters": {"atoms": 4},
                "success_metric": {"kind": "goal_hit"},
                "uplift": {"phase": "II"}
            }
        }
    }


@pytest.fixture
def create_fingerprint_files(sample_curriculum_config):
    """Helper to create temporary fingerprint files."""
    def _create(configs):
        """Create fingerprint files from list of configs."""
        paths = []
        for i, config in enumerate(configs):
            curriculum = CurriculumLoaderV2.from_dict(config)
            fingerprint = CurriculumFingerprint.generate(curriculum, run_id=f"run{i}")
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(fingerprint.to_dict(), f)
                paths.append(f.name)
        
        return paths
    
    yield _create
    
    # Cleanup (note: files need to be cleaned up by test if needed)


class TestBuildCurriculumDriftHistory:
    """Tests for build_curriculum_drift_history."""
    
    def test_empty_paths(self):
        """Test with no fingerprint paths."""
        history = build_curriculum_drift_history([])
        
        assert history["fingerprints"] == []
        assert history["schema_version"] is None
        assert history["drift_events_count"] == 0
    
    def test_single_fingerprint(self, sample_curriculum_config, create_fingerprint_files):
        """Test with single fingerprint."""
        paths = create_fingerprint_files([sample_curriculum_config])
        
        try:
            history = build_curriculum_drift_history(paths)
            
            assert len(history["fingerprints"]) == 1
            assert history["schema_version"] == "phase2-v2.1.0"
            assert history["drift_events_count"] == 0
            assert len(history["slice_count_series"]) == 1
        finally:
            for path in paths:
                os.unlink(path)
    
    def test_multiple_identical_fingerprints(self, sample_curriculum_config, create_fingerprint_files):
        """Test with multiple identical fingerprints."""
        paths = create_fingerprint_files([
            sample_curriculum_config,
            sample_curriculum_config,
            sample_curriculum_config
        ])
        
        try:
            history = build_curriculum_drift_history(paths)
            
            assert len(history["fingerprints"]) == 3
            assert history["drift_events_count"] == 0  # All identical
        finally:
            for path in paths:
                os.unlink(path)
    
    def test_drift_detection(self, sample_curriculum_config, create_fingerprint_files):
        """Test drift event detection."""
        config2 = sample_curriculum_config.copy()
        config2["slices"] = sample_curriculum_config["slices"].copy()
        config2["slices"]["slice_a"] = {
            **sample_curriculum_config["slices"]["slice_a"],
            "parameters": {"atoms": 5}  # Changed
        }
        
        paths = create_fingerprint_files([
            sample_curriculum_config,
            config2,
            config2
        ])
        
        try:
            history = build_curriculum_drift_history(paths)
            
            assert len(history["fingerprints"]) == 3
            assert history["drift_events_count"] == 1  # One drift between first and second
        finally:
            for path in paths:
                os.unlink(path)


class TestClassifyCurriculumDriftEvent:
    """Tests for classify_curriculum_drift_event."""
    
    def test_no_change(self):
        """Test classification when no change."""
        diff = {"changed": False}
        result = classify_curriculum_drift_event(diff)
        
        assert result["severity"] == "NONE"
        assert not result["blocking"]
        assert result["reasons"] == []
    
    def test_minor_change(self):
        """Test classification for minor parameter change."""
        diff = {
            "changed": True,
            "schema_version_changed": False,
            "slices_added": [],
            "slices_removed": [],
            "slices_modified": ["slice_a"],
            "old_schema_version": "phase2-v2.1.0",
            "new_schema_version": "phase2-v2.1.0"
        }
        result = classify_curriculum_drift_event(diff)
        
        assert result["severity"] == "MINOR"
        assert not result["blocking"]  # Minor changes not blocking
        assert len(result["reasons"]) > 0
    
    def test_major_schema_change(self):
        """Test classification for schema version change."""
        diff = {
            "changed": True,
            "schema_version_changed": True,
            "slices_added": [],
            "slices_removed": [],
            "slices_modified": [],
            "old_schema_version": "phase2-v2.1.0",
            "new_schema_version": "phase2-v2.2.0"
        }
        result = classify_curriculum_drift_event(diff)
        
        assert result["severity"] == "MAJOR"
        assert result["blocking"]
        assert any("schema version" in r.lower() for r in result["reasons"])
    
    def test_major_slice_added(self):
        """Test classification for added slice."""
        diff = {
            "changed": True,
            "schema_version_changed": False,
            "slices_added": ["slice_new"],
            "slices_removed": [],
            "slices_modified": [],
            "old_schema_version": "phase2-v2.1.0",
            "new_schema_version": "phase2-v2.1.0"
        }
        result = classify_curriculum_drift_event(diff)
        
        assert result["severity"] == "MAJOR"
        assert result["blocking"]
        assert any("added" in r.lower() for r in result["reasons"])
    
    def test_major_slice_removed(self):
        """Test classification for removed slice."""
        diff = {
            "changed": True,
            "schema_version_changed": False,
            "slices_added": [],
            "slices_removed": ["slice_old"],
            "slices_modified": [],
            "old_schema_version": "phase2-v2.1.0",
            "new_schema_version": "phase2-v2.1.0"
        }
        result = classify_curriculum_drift_event(diff)
        
        assert result["severity"] == "MAJOR"
        assert result["blocking"]
        assert any("removed" in r.lower() for r in result["reasons"])


class TestEvaluateCurriculumForPromotion:
    """Tests for evaluate_curriculum_for_promotion."""
    
    def test_no_fingerprints(self):
        """Test promotion evaluation with no fingerprints."""
        history = {"fingerprints": []}
        result = evaluate_curriculum_for_promotion(history)
        
        assert not result["promotion_ok"]
        assert len(result["blocking_reasons"]) > 0
    
    def test_promotion_ok_no_drift(self, sample_curriculum_config, create_fingerprint_files):
        """Test promotion OK when no drift."""
        paths = create_fingerprint_files([
            sample_curriculum_config,
            sample_curriculum_config
        ])
        
        try:
            history = build_curriculum_drift_history(paths)
            result = evaluate_curriculum_for_promotion(history)
            
            assert result["promotion_ok"]
            assert result["last_drift_severity"] == "NONE"
            assert result["blocking_reasons"] == []
        finally:
            for path in paths:
                os.unlink(path)
    
    def test_promotion_blocked_major_drift(self, sample_curriculum_config, create_fingerprint_files):
        """Test promotion blocked on major drift."""
        config2 = sample_curriculum_config.copy()
        config2["version"] = "2.2.0"  # Schema change
        
        paths = create_fingerprint_files([
            sample_curriculum_config,
            config2
        ])
        
        try:
            history = build_curriculum_drift_history(paths)
            result = evaluate_curriculum_for_promotion(history)
            
            assert not result["promotion_ok"]
            assert result["last_drift_severity"] == "MAJOR"
            assert len(result["blocking_reasons"]) > 0
        finally:
            for path in paths:
                os.unlink(path)
    
    def test_promotion_ok_minor_drift(self, sample_curriculum_config, create_fingerprint_files):
        """Test promotion OK with minor drift."""
        config2 = sample_curriculum_config.copy()
        config2["slices"] = {
            "slice_a": {
                **sample_curriculum_config["slices"]["slice_a"],
                "parameters": {"atoms": 5}  # Minor change
            }
        }
        
        paths = create_fingerprint_files([
            sample_curriculum_config,
            config2
        ])
        
        try:
            history = build_curriculum_drift_history(paths)
            result = evaluate_curriculum_for_promotion(history)
            
            # Minor changes should not block promotion
            assert result["promotion_ok"]
            assert result["last_drift_severity"] == "MINOR"
        finally:
            for path in paths:
                os.unlink(path)


class TestSummarizeCurriculumForGlobalHealth:
    """Tests for summarize_curriculum_for_global_health."""
    
    def test_no_fingerprints(self):
        """Test health summary with no fingerprints."""
        history = {"fingerprints": []}
        result = summarize_curriculum_for_global_health(history)
        
        assert not result["curriculum_ok"]
        assert result["status"] == "BLOCK"
        assert result["current_slice_count"] == 0
    
    def test_healthy_curriculum(self, sample_curriculum_config, create_fingerprint_files):
        """Test health summary for healthy curriculum."""
        paths = create_fingerprint_files([
            sample_curriculum_config,
            sample_curriculum_config
        ])
        
        try:
            history = build_curriculum_drift_history(paths)
            result = summarize_curriculum_for_global_health(history)
            
            assert result["curriculum_ok"]
            assert result["status"] == "OK"
            assert result["current_slice_count"] == 1
            assert result["recent_drift_events"] == 0
        finally:
            for path in paths:
                os.unlink(path)
    
    def test_blocked_curriculum(self, sample_curriculum_config, create_fingerprint_files):
        """Test health summary for blocked curriculum."""
        config2 = sample_curriculum_config.copy()
        config2["version"] = "2.2.0"  # Major change
        
        paths = create_fingerprint_files([
            sample_curriculum_config,
            config2
        ])
        
        try:
            history = build_curriculum_drift_history(paths)
            result = summarize_curriculum_for_global_health(history)
            
            assert not result["curriculum_ok"]
            assert result["status"] == "BLOCK"
            assert not result["details"]["promotion_ok"]
        finally:
            for path in paths:
                os.unlink(path)
    
    def test_warn_status(self, sample_curriculum_config, create_fingerprint_files):
        """Test WARN status with significant drift activity."""
        configs = [sample_curriculum_config]
        
        # Create 5 different configs to trigger WARN
        for i in range(1, 5):
            config = sample_curriculum_config.copy()
            config["slices"] = {
                "slice_a": {
                    **sample_curriculum_config["slices"]["slice_a"],
                    "parameters": {"atoms": 4 + i}
                }
            }
            configs.append(config)
        
        paths = create_fingerprint_files(configs)
        
        try:
            history = build_curriculum_drift_history(paths)
            result = summarize_curriculum_for_global_health(history)
            
            # Should warn due to high drift count (4 changes)
            assert result["curriculum_ok"]  # Still OK, just warning
            assert result["status"] == "WARN"
            assert result["details"]["total_drift_events"] > 3
        finally:
            for path in paths:
                os.unlink(path)
