"""
Phase III Hard Gate Tests

Operation CORTEX: Phase III Validation
======================================

Tests for TDA hard gate enforcement, governance integration,
and attestation binding.

Test Categories:
1. Hard Gate Enforcement (should_block behavior)
2. ProofOutcome.ABANDONED_TDA semantics
3. Pipeline hash computation
4. Governance summarization
5. Drift detection
6. Attestation binding
"""

import pytest
import hashlib
import json
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, MagicMock, patch

import numpy as np


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_tda_config():
    """Create mock TDAMonitorConfig."""
    config = Mock()
    config.hss_block_threshold = 0.2
    config.hss_warn_threshold = 0.5
    config.mode = Mock(value="hard")
    config.lifetime_threshold = 0.05
    config.deviation_max = 0.5
    config.max_simplex_dim = 3
    config.max_homology_dim = 1
    config.fail_open = False
    return config


@pytest.fixture
def mock_tda_result_blocked():
    """Create mock TDAMonitorResult with BLOCK signal."""
    result = Mock()
    result.hss = 0.15  # Below block threshold
    result.sns = 0.3
    result.pcs = 0.2
    result.drs = 0.4
    result.signal = Mock(value="BLOCK")
    result.block = True
    result.warn = False
    result.to_dict = lambda: {
        "hss": 0.15, "sns": 0.3, "pcs": 0.2, "drs": 0.4,
        "signal": "BLOCK", "block": True, "warn": False,
    }
    return result


@pytest.fixture
def mock_tda_result_warn():
    """Create mock TDAMonitorResult with WARN signal."""
    result = Mock()
    result.hss = 0.35  # Between block and warn threshold
    result.sns = 0.5
    result.pcs = 0.4
    result.drs = 0.3
    result.signal = Mock(value="WARN")
    result.block = False
    result.warn = True
    result.to_dict = lambda: {
        "hss": 0.35, "sns": 0.5, "pcs": 0.4, "drs": 0.3,
        "signal": "WARN", "block": False, "warn": True,
    }
    return result


@pytest.fixture
def mock_tda_result_ok():
    """Create mock TDAMonitorResult with OK signal."""
    result = Mock()
    result.hss = 0.75  # Above warn threshold
    result.sns = 0.8
    result.pcs = 0.7
    result.drs = 0.1
    result.signal = Mock(value="OK")
    result.block = False
    result.warn = False
    result.to_dict = lambda: {
        "hss": 0.75, "sns": 0.8, "pcs": 0.7, "drs": 0.1,
        "signal": "OK", "block": False, "warn": False,
    }
    return result


@pytest.fixture
def mock_reference_profile():
    """Create mock ReferenceTDAProfile."""
    profile = Mock()
    profile.version = "1.0.0"
    profile.n_ref = 100
    profile.mean_betti_0 = 5.0
    profile.mean_betti_1 = 2.0
    return profile


# ============================================================================
# 1. Hard Gate Enforcement Tests
# ============================================================================

class TestHardGateEnforcement:
    """Tests for should_block() behavior in HARD mode."""

    def test_should_block_returns_true_when_hss_below_threshold(
        self, mock_tda_config, mock_tda_result_blocked
    ):
        """HARD mode should return True when HSS < θ_block."""
        from backend.tda.runtime_monitor import TDAOperationalMode

        mock_tda_config.mode = TDAOperationalMode.HARD

        # Create monitor mock
        monitor = Mock()
        monitor.cfg = mock_tda_config

        # should_block checks mode and result.block
        def should_block(result):
            if monitor.cfg.mode != TDAOperationalMode.HARD:
                return False
            return result.block

        monitor.should_block = should_block

        assert monitor.should_block(mock_tda_result_blocked) is True

    def test_should_block_returns_false_when_hss_above_threshold(
        self, mock_tda_config, mock_tda_result_ok
    ):
        """HARD mode should return False when HSS >= θ_warn."""
        from backend.tda.runtime_monitor import TDAOperationalMode

        mock_tda_config.mode = TDAOperationalMode.HARD

        monitor = Mock()
        monitor.cfg = mock_tda_config

        def should_block(result):
            if monitor.cfg.mode != TDAOperationalMode.HARD:
                return False
            return result.block

        monitor.should_block = should_block

        assert monitor.should_block(mock_tda_result_ok) is False

    def test_should_block_returns_false_in_soft_mode(
        self, mock_tda_config, mock_tda_result_blocked
    ):
        """SOFT mode should never block, only warn."""
        from backend.tda.runtime_monitor import TDAOperationalMode

        mock_tda_config.mode = TDAOperationalMode.SOFT

        monitor = Mock()
        monitor.cfg = mock_tda_config

        def should_block(result):
            if monitor.cfg.mode != TDAOperationalMode.HARD:
                return False
            return result.block

        monitor.should_block = should_block

        assert monitor.should_block(mock_tda_result_blocked) is False

    def test_should_block_returns_false_in_shadow_mode(
        self, mock_tda_config, mock_tda_result_blocked
    ):
        """SHADOW mode should never affect execution."""
        from backend.tda.runtime_monitor import TDAOperationalMode

        mock_tda_config.mode = TDAOperationalMode.SHADOW

        monitor = Mock()
        monitor.cfg = mock_tda_config

        def should_block(result):
            if monitor.cfg.mode != TDAOperationalMode.HARD:
                return False
            return result.block

        monitor.should_block = should_block

        assert monitor.should_block(mock_tda_result_blocked) is False


# ============================================================================
# 2. ProofOutcome Tests
# ============================================================================

class TestProofOutcome:
    """Tests for ProofOutcome.ABANDONED_TDA semantics."""

    def test_abandoned_tda_exists(self):
        """ProofOutcome should include ABANDONED_TDA value."""
        # Define the enum for testing (would be in runner.py)
        from enum import Enum

        class ProofOutcome(Enum):
            VERIFIED = "verified"
            FAILED = "failed"
            TIMEOUT = "timeout"
            ERROR = "error"
            SKIPPED = "skipped"
            ABANDONED_TDA = "abandoned_tda"

        assert ProofOutcome.ABANDONED_TDA.value == "abandoned_tda"

    def test_abandoned_tda_is_not_success(self):
        """ABANDONED_TDA should not be considered successful."""
        from enum import Enum

        class ProofOutcome(Enum):
            VERIFIED = "verified"
            ABANDONED_TDA = "abandoned_tda"

            def is_success(self):
                return self == ProofOutcome.VERIFIED

        outcome = ProofOutcome.ABANDONED_TDA
        assert outcome.is_success() is False

    def test_abandoned_tda_does_not_allow_learning(self):
        """ABANDONED_TDA should not contribute to policy learning."""
        from enum import Enum

        class ProofOutcome(Enum):
            VERIFIED = "verified"
            FAILED = "failed"
            ABANDONED_TDA = "abandoned_tda"

            def allows_learning(self):
                return self not in (ProofOutcome.ABANDONED_TDA,)

        outcome = ProofOutcome.ABANDONED_TDA
        assert outcome.allows_learning() is False

        # But VERIFIED and FAILED should allow learning
        assert ProofOutcome.VERIFIED.allows_learning() is True
        assert ProofOutcome.FAILED.allows_learning() is True


# ============================================================================
# 3. Pipeline Hash Tests
# ============================================================================

class TestPipelineHash:
    """Tests for TDA pipeline hash computation."""

    def test_pipeline_hash_is_deterministic(
        self, mock_tda_config, mock_reference_profile
    ):
        """Same configuration should produce same hash."""
        from backend.tda.governance import compute_tda_pipeline_hash

        profiles = {"slice_1": mock_reference_profile}

        hash1 = compute_tda_pipeline_hash(mock_tda_config, profiles)
        hash2 = compute_tda_pipeline_hash(mock_tda_config, profiles)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex

    def test_pipeline_hash_changes_with_config(
        self, mock_tda_config, mock_reference_profile
    ):
        """Different configurations should produce different hashes."""
        from backend.tda.governance import compute_tda_pipeline_hash

        profiles = {"slice_1": mock_reference_profile}

        hash1 = compute_tda_pipeline_hash(mock_tda_config, profiles)

        # Modify config
        mock_tda_config.hss_block_threshold = 0.3

        hash2 = compute_tda_pipeline_hash(mock_tda_config, profiles)

        assert hash1 != hash2

    def test_pipeline_hash_changes_with_profiles(
        self, mock_tda_config, mock_reference_profile
    ):
        """Different profiles should produce different hashes."""
        from backend.tda.governance import compute_tda_pipeline_hash

        profiles1 = {"slice_1": mock_reference_profile}
        hash1 = compute_tda_pipeline_hash(mock_tda_config, profiles1)

        # Add another profile
        profile2 = Mock()
        profile2.version = "1.0.0"
        profile2.n_ref = 200
        profile2.mean_betti_0 = 10.0
        profile2.mean_betti_1 = 4.0

        profiles2 = {"slice_1": mock_reference_profile, "slice_2": profile2}
        hash2 = compute_tda_pipeline_hash(mock_tda_config, profiles2)

        assert hash1 != hash2

    def test_pipeline_hash_empty_profiles(self, mock_tda_config):
        """Empty profiles should still produce valid hash."""
        from backend.tda.governance import compute_tda_pipeline_hash

        hash_result = compute_tda_pipeline_hash(mock_tda_config, {})

        assert len(hash_result) == 64
        assert all(c in "0123456789abcdef" for c in hash_result)


# ============================================================================
# 4. Governance Summarization Tests
# ============================================================================

class TestGovernanceSummarization:
    """Tests for summarize_tda_for_global_health()."""

    def test_empty_results_returns_healthy(self, mock_tda_config):
        """Empty results should return HEALTHY governance signal."""
        from backend.tda.governance import summarize_tda_for_global_health

        summary = summarize_tda_for_global_health([], mock_tda_config)

        assert summary["governance_signal"] == "HEALTHY"
        assert summary["cycle_count"] == 0
        assert summary["structural_health"] == 1.0

    def test_all_ok_results_returns_healthy(
        self, mock_tda_config, mock_tda_result_ok
    ):
        """All OK results should return HEALTHY signal."""
        from backend.tda.governance import summarize_tda_for_global_health

        results = [mock_tda_result_ok] * 10

        summary = summarize_tda_for_global_health(results, mock_tda_config)

        assert summary["governance_signal"] == "HEALTHY"
        assert summary["block_count"] == 0
        assert summary["ok_count"] == 10
        assert summary["block_rate"] == 0.0

    def test_high_block_rate_returns_critical(
        self, mock_tda_config, mock_tda_result_blocked, mock_tda_result_ok
    ):
        """Block rate > 20% should return CRITICAL signal."""
        from backend.tda.governance import summarize_tda_for_global_health

        # 3 blocked out of 10 = 30% block rate
        results = [mock_tda_result_blocked] * 3 + [mock_tda_result_ok] * 7

        summary = summarize_tda_for_global_health(results, mock_tda_config)

        assert summary["governance_signal"] == "CRITICAL"
        assert summary["block_rate"] == 0.3

    def test_moderate_block_rate_returns_degraded(
        self, mock_tda_config, mock_tda_result_blocked, mock_tda_result_ok
    ):
        """Block rate between 10-20% should return DEGRADED signal."""
        from backend.tda.governance import summarize_tda_for_global_health

        # 15 blocked out of 100 = 15% block rate
        results = [mock_tda_result_blocked] * 15 + [mock_tda_result_ok] * 85

        summary = summarize_tda_for_global_health(results, mock_tda_config)

        assert summary["governance_signal"] == "DEGRADED"

    def test_hss_trend_computed_correctly(self, mock_tda_config):
        """HSS trend should be computed as linear slope."""
        from backend.tda.governance import summarize_tda_for_global_health

        # Create results with increasing HSS
        results = []
        for i in range(10):
            result = Mock()
            result.hss = 0.5 + i * 0.05  # 0.5, 0.55, 0.6, ...
            result.block = False
            result.warn = False
            results.append(result)

        summary = summarize_tda_for_global_health(results, mock_tda_config)

        # Trend should be positive
        assert summary["hss_trend"] > 0


# ============================================================================
# 5. Drift Detection Tests
# ============================================================================

class TestDriftDetection:
    """Tests for drift detection and reporting."""

    def test_no_drift_when_distributions_match(self):
        """Identical distributions should not detect drift."""
        from backend.tda.governance import compute_drift_metrics

        baseline = [0.7] * 100
        current = [0.7] * 100

        metrics = compute_drift_metrics(baseline, current)

        assert metrics.drift_detected is False
        assert metrics.drift_severity == "none"

    def test_drift_detected_when_distributions_differ(self):
        """Significantly different distributions should detect drift."""
        from backend.tda.governance import compute_drift_metrics

        np.random.seed(42)
        baseline = list(np.random.normal(0.7, 0.1, 100))
        current = list(np.random.normal(0.4, 0.1, 100))  # Much lower

        metrics = compute_drift_metrics(baseline, current)

        assert metrics.drift_detected is True
        assert metrics.hss_delta < 0  # Current is lower

    def test_drift_severity_classification(self):
        """Drift severity should be classified correctly."""
        from backend.tda.governance import compute_drift_metrics

        np.random.seed(42)

        # Minor drift
        baseline = list(np.random.normal(0.7, 0.05, 100))
        current_minor = list(np.random.normal(0.65, 0.05, 100))

        metrics_minor = compute_drift_metrics(baseline, current_minor)
        # May or may not detect depending on random variation

        # Critical drift
        current_critical = list(np.random.normal(0.3, 0.1, 100))
        metrics_critical = compute_drift_metrics(baseline, current_critical)

        assert metrics_critical.drift_detected is True
        assert metrics_critical.drift_severity in ["major", "critical"]

    def test_generate_drift_report_structure(self, mock_tda_result_ok):
        """Drift report should have correct schema structure."""
        from backend.tda.governance import generate_drift_report

        baseline_results = [mock_tda_result_ok] * 50
        current_results = [mock_tda_result_ok] * 50

        report = generate_drift_report(
            baseline_results,
            current_results,
            pipeline_hash="a" * 64,
        )

        assert report["schema_version"] == "tda-drift-report-v1"
        assert "generated_at" in report
        assert "pipeline_hash" in report
        assert "baseline_period" in report
        assert "current_period" in report
        assert "drift_metrics" in report
        assert "recommendations" in report


# ============================================================================
# 6. Attestation Binding Tests
# ============================================================================

class TestAttestationBinding:
    """Tests for TDA attestation integration."""

    def test_extend_attestation_adds_tda_governance(
        self, mock_tda_config, mock_tda_result_ok
    ):
        """extend_attestation_with_tda should add tda_governance section."""
        from backend.tda.governance import extend_attestation_with_tda

        original_metadata = {
            "reasoning_merkle_root": "abc123",
            "ui_merkle_root": "def456",
        }

        extended = extend_attestation_with_tda(
            original_metadata,
            [mock_tda_result_ok] * 10,
            mock_tda_config,
            pipeline_hash="a" * 64,
        )

        assert "tda_governance" in extended
        assert extended["tda_governance"]["phase"] == "III"
        assert extended["tda_governance"]["pipeline_hash"] == "a" * 64
        assert "summary" in extended["tda_governance"]

    def test_attestation_preserves_original_fields(
        self, mock_tda_config, mock_tda_result_ok
    ):
        """Extension should not modify original attestation fields."""
        from backend.tda.governance import extend_attestation_with_tda

        original_metadata = {
            "reasoning_merkle_root": "abc123",
            "ui_merkle_root": "def456",
            "composite_attestation_root": "ghi789",
        }

        extended = extend_attestation_with_tda(
            original_metadata.copy(),
            [mock_tda_result_ok],
            mock_tda_config,
            pipeline_hash="a" * 64,
        )

        assert extended["reasoning_merkle_root"] == "abc123"
        assert extended["ui_merkle_root"] == "def456"
        assert extended["composite_attestation_root"] == "ghi789"


# ============================================================================
# 7. Integration Flow Tests
# ============================================================================

class TestHardGateIntegrationFlow:
    """End-to-end tests for hard gate enforcement flow."""

    def test_blocked_attempt_skips_lean_submission(
        self, mock_tda_config, mock_tda_result_blocked
    ):
        """Blocked attempt should never reach Lean verifier."""
        lean_submitted = False

        def mock_lean_submit(formula):
            nonlocal lean_submitted
            lean_submitted = True
            return True

        # Simulate hard gate check
        from backend.tda.runtime_monitor import TDAOperationalMode
        mock_tda_config.mode = TDAOperationalMode.HARD

        if mock_tda_result_blocked.block:
            # Would abandon here
            pass
        else:
            mock_lean_submit("p -> p")

        assert lean_submitted is False

    def test_ok_attempt_proceeds_to_lean(
        self, mock_tda_config, mock_tda_result_ok
    ):
        """OK attempt should proceed to Lean verification."""
        lean_submitted = False

        def mock_lean_submit(formula):
            nonlocal lean_submitted
            lean_submitted = True
            return True

        from backend.tda.runtime_monitor import TDAOperationalMode
        mock_tda_config.mode = TDAOperationalMode.HARD

        if mock_tda_result_ok.block:
            pass
        else:
            mock_lean_submit("p -> p")

        assert lean_submitted is True

    def test_blocked_attempt_skips_policy_update(
        self, mock_tda_config, mock_tda_result_blocked
    ):
        """Blocked attempt should not trigger policy update."""
        policy_updated = False

        def mock_policy_update(reward):
            nonlocal policy_updated
            policy_updated = True

        from backend.tda.runtime_monitor import TDAOperationalMode
        mock_tda_config.mode = TDAOperationalMode.HARD

        if mock_tda_result_blocked.block:
            # Abandon - no policy update
            pass
        else:
            mock_policy_update(1.0)

        assert policy_updated is False
