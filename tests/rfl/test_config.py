"""
Tests for RFL Configuration

Validates configuration loading, validation, and serialization.
"""

import pytest
import tempfile
import os
from pathlib import Path

from backend.rfl.config import (
    RFLConfig,
    RFL_QUICK_CONFIG,
    RFL_PRODUCTION_CONFIG,
    CurriculumSlice
)


class TestRFLConfig:
    """Tests for RFLConfig dataclass."""

    def test_default_config(self):
        """Test default configuration creation."""
        config = RFLConfig()

        assert config.experiment_id == "rfl_001"
        assert config.num_runs == 40
        assert config.coverage_threshold == 0.92
        assert config.uplift_threshold == 1.0
        assert config.bootstrap_replicates == 10000
        assert len(config.curriculum) >= 1
        assert config.curriculum[0].start_run == 1
        assert config.curriculum[-1].end_run == config.num_runs

    def test_custom_config(self):
        """Test custom configuration."""
        config = RFLConfig(
            experiment_id="custom_exp",
            num_runs=20,
            coverage_threshold=0.95
        )

        assert config.experiment_id == "custom_exp"
        assert config.num_runs == 20
        assert config.coverage_threshold == 0.95
        assert config.curriculum[-1].end_run == 20

    def test_validate_success(self):
        """Test validation passes with valid config."""
        config = RFLConfig(
            num_runs=40,
            coverage_threshold=0.92,
            uplift_threshold=1.0,
            bootstrap_replicates=10000
        )

        config.validate()  # Should not raise

    def test_validate_num_runs_too_small(self):
        """Test validation fails with num_runs < 2."""
        config = RFLConfig(num_runs=1)

        with pytest.raises(ValueError, match="num_runs must be ≥2"):
            config.validate()

    def test_validate_coverage_threshold_invalid(self):
        """Test validation fails with invalid coverage threshold."""
        config = RFLConfig(coverage_threshold=1.5)

        with pytest.raises(ValueError, match="coverage_threshold must be in"):
            config.validate()

    def test_validate_bootstrap_replicates_too_small(self):
        """Test validation fails with too few bootstrap replicates."""
        config = RFLConfig(bootstrap_replicates=500)

        with pytest.raises(ValueError, match="bootstrap_replicates should be ≥1000"):
            config.validate()

    def test_validate_confidence_level_invalid(self):
        """Test validation fails with invalid confidence level."""
        config = RFLConfig(confidence_level=1.5)

        with pytest.raises(ValueError, match="confidence_level must be in"):
            config.validate()

    def test_validate_derive_steps_invalid(self):
        """Test validation fails with derive_steps <= 0."""
        config = RFLConfig(derive_steps=0)

        with pytest.raises(ValueError, match="derive_steps must be >0"):
            config.validate()

    def test_validate_dual_attestation_tolerance_invalid(self):
        """Test validation fails when dual attestation tolerance is non-positive."""
        config = RFLConfig(dual_attestation=True, dual_attestation_tolerance=0.0)

        with pytest.raises(ValueError, match="dual_attestation_tolerance"):
            config.validate()

    def test_validate_abstention_tolerance_invalid(self):
        """Test validation fails when abstention tolerance is outside [0, 1]."""
        config = RFLConfig(abstention_tolerance=1.5)

        with pytest.raises(ValueError, match="abstention_tolerance"):
            config.validate()

    def test_to_dict(self):
        """Test dictionary conversion."""
        config = RFLConfig(experiment_id="test")

        d = config.to_dict()

        assert d["experiment_id"] == "test"
        assert d["num_runs"] == 40
        assert isinstance(d, dict)
        assert "curriculum" in d
        assert isinstance(d["curriculum"], list)

    def test_to_json_from_json(self):
        """Test JSON serialization and deserialization."""
        config = RFLConfig(
            experiment_id="json_test",
            num_runs=25,
            coverage_threshold=0.95
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "config.json"

            # Save
            config.to_json(str(json_path))
            assert json_path.exists()

            # Load
            loaded_config = RFLConfig.from_json(str(json_path))

            assert loaded_config.experiment_id == "json_test"
            assert loaded_config.num_runs == 25
            assert loaded_config.coverage_threshold == 0.95
            assert len(loaded_config.curriculum) >= 1

    def test_from_env(self, monkeypatch):
        """Test loading configuration from environment variables."""
        monkeypatch.setenv("RFL_EXPERIMENT_ID", "env_test")
        monkeypatch.setenv("RFL_NUM_RUNS", "30")
        monkeypatch.setenv("RFL_COVERAGE_THRESHOLD", "0.95")
        monkeypatch.setenv("DERIVE_STEPS", "100")

        config = RFLConfig.from_env()

        assert config.experiment_id == "env_test"
        assert config.num_runs == 30
        assert config.coverage_threshold == 0.95
        assert config.derive_steps == 100
        assert config.curriculum[-1].end_run == 30

    def test_resolve_slice(self):
        """Test curriculum slice resolution."""
        config = RFLConfig(num_runs=12)

        first_slice = config.resolve_slice(1)
        mid_slice = config.resolve_slice(6)
        last_slice = config.resolve_slice(12)

        assert first_slice.start_run == 1
        assert last_slice.end_run == 12
        assert first_slice.name in {"warmup", "core"}
        assert mid_slice.contains(6)


class TestPresetConfigs:
    """Tests for preset configurations."""

    def test_quick_config(self):
        """Test quick configuration preset."""
        config = RFL_QUICK_CONFIG

        assert config.experiment_id == "rfl_quick"
        assert config.num_runs == 5
        assert config.derive_steps == 10
        assert config.bootstrap_replicates == 1000

    def test_production_config(self):
        """Test production configuration preset."""
        config = RFL_PRODUCTION_CONFIG

        assert config.experiment_id == "rfl_prod"
        assert config.num_runs == 40
        assert config.derive_steps == 100
        assert config.bootstrap_replicates == 10000

    def test_quick_config_validates(self):
        """Test quick config passes validation."""
        RFL_QUICK_CONFIG.validate()  # Should not raise

    def test_production_config_validates(self):
        """Test production config passes validation."""
        RFL_PRODUCTION_CONFIG.validate()  # Should not raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
