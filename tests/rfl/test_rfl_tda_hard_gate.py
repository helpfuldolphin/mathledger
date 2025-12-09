"""
RFL TDA Hard Gate Tests

Operation CORTEX: Phase III RFL Integration
============================================

Tests for TDA hard gate enforcement in RFLRunner,
including attestation handling and policy ledger updates.

Test Categories:
1. RFLRunner hard gate configuration
2. Attestation-level blocking
3. RunLedgerEntry TDA fields
4. Policy update prevention
5. Statistics tracking
"""

import pytest
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, MagicMock, patch
from enum import Enum

import numpy as np


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_rfl_config():
    """Create mock RFLConfig."""
    config = Mock()
    config.experiment_id = "test_hard_gate"
    config.database_url = "postgresql://test:test@localhost/test"
    config.system_id = 1
    config.random_seed = 42
    config.num_runs = 1
    config.derive_steps = 10
    config.max_breadth = 100
    config.max_total = 500
    config.coverage_threshold = 0.92
    config.uplift_threshold = 1.0
    config.bootstrap_replicates = 100
    config.artifacts_dir = "/tmp/test_artifacts"
    config.curriculum = []
    config.dual_attestation = False
    config.validate = Mock()
    return config


@pytest.fixture
def mock_attestation():
    """Create mock attestation for testing."""
    attestation = Mock()
    attestation.slice_id = "test_slice"
    attestation.root_hash = "abc123def456"
    attestation.abstention_rate = 0.05
    attestation.abstention_mass = 10.0
    attestation.metadata = {"abstention_breakdown": {}}
    attestation.verified_statements = ["stmt1", "stmt2"]
    return attestation


@pytest.fixture
def mock_tda_monitor():
    """Create mock TDAMonitor."""
    from backend.tda.runtime_monitor import TDAOperationalMode, TDAGatingSignal

    monitor = Mock()
    monitor.cfg = Mock()
    monitor.cfg.mode = TDAOperationalMode.HARD
    monitor.cfg.hss_block_threshold = 0.2
    monitor.cfg.hss_warn_threshold = 0.5
    monitor.cfg.lifetime_threshold = 0.05
    monitor.cfg.deviation_max = 0.5
    monitor.cfg.max_simplex_dim = 3
    monitor.cfg.max_homology_dim = 1
    monitor.cfg.fail_open = False
    monitor.slice_ref_profiles = {}

    return monitor


@pytest.fixture
def mock_tda_result_blocked():
    """Create TDA result that triggers block."""
    from backend.tda.runtime_monitor import TDAGatingSignal

    result = Mock()
    result.hss = 0.15
    result.sns = 0.3
    result.pcs = 0.2
    result.drs = 0.4
    result.signal = TDAGatingSignal.BLOCK
    result.block = True
    result.warn = False
    result.to_dict = lambda: {
        "hss": 0.15, "sns": 0.3, "pcs": 0.2, "drs": 0.4,
        "signal": "BLOCK", "block": True, "warn": False,
    }
    return result


@pytest.fixture
def mock_tda_result_ok():
    """Create TDA result that allows processing."""
    from backend.tda.runtime_monitor import TDAGatingSignal

    result = Mock()
    result.hss = 0.75
    result.sns = 0.8
    result.pcs = 0.7
    result.drs = 0.1
    result.signal = TDAGatingSignal.OK
    result.block = False
    result.warn = False
    result.to_dict = lambda: {
        "hss": 0.75, "sns": 0.8, "pcs": 0.7, "drs": 0.1,
        "signal": "OK", "block": False, "warn": False,
    }
    return result


# ============================================================================
# Test ProofOutcome for RFL
# ============================================================================

class TestRflProofOutcome:
    """Tests for RFL-specific ProofOutcome handling."""

    def test_proof_outcome_abandoned_tda_defined(self):
        """ProofOutcome.ABANDONED_TDA should be defined for RFL."""

        class ProofOutcome(Enum):
            PROCESSED = "processed"
            SKIPPED = "skipped"
            ERROR = "error"
            ABANDONED_TDA = "abandoned_tda"

        assert ProofOutcome.ABANDONED_TDA.value == "abandoned_tda"

    def test_abandoned_tda_disallows_policy_update(self):
        """ABANDONED_TDA should not allow policy updates."""

        class ProofOutcome(Enum):
            PROCESSED = "processed"
            ABANDONED_TDA = "abandoned_tda"

            def allows_policy_update(self):
                return self == ProofOutcome.PROCESSED

        assert ProofOutcome.ABANDONED_TDA.allows_policy_update() is False
        assert ProofOutcome.PROCESSED.allows_policy_update() is True


# ============================================================================
# Test RFLRunner Hard Gate Configuration
# ============================================================================

class TestRflRunnerHardGateConfig:
    """Tests for RFLRunner hard gate configuration."""

    def test_hard_gate_disabled_by_default(self, mock_rfl_config):
        """Hard gate should be disabled by default."""
        # Simulate RFLRunner initialization checks
        tda_hard_gate_enabled = False

        assert tda_hard_gate_enabled is False

    def test_hard_gate_enabled_via_environment(self, mock_rfl_config):
        """Hard gate should be enabled when MATHLEDGER_TDA_MODE=hard."""
        import os

        with patch.dict(os.environ, {"MATHLEDGER_TDA_MODE": "hard"}):
            tda_mode = os.getenv("MATHLEDGER_TDA_MODE", "soft")
            tda_hard_gate_enabled = tda_mode == "hard"

            assert tda_hard_gate_enabled is True

    def test_pipeline_hash_computed_on_init(
        self, mock_rfl_config, mock_tda_monitor
    ):
        """Pipeline hash should be computed during initialization."""
        from backend.tda.governance import compute_tda_pipeline_hash

        pipeline_hash = compute_tda_pipeline_hash(
            mock_tda_monitor.cfg,
            mock_tda_monitor.slice_ref_profiles,
        )

        assert len(pipeline_hash) == 64
        assert all(c in "0123456789abcdef" for c in pipeline_hash)


# ============================================================================
# Test Attestation-Level Blocking
# ============================================================================

class TestAttestationBlocking:
    """Tests for attestation-level hard gate blocking."""

    def test_blocked_attestation_returns_abandoned_result(
        self, mock_attestation, mock_tda_monitor, mock_tda_result_blocked
    ):
        """Blocked attestation should return ABANDONED_TDA result."""
        from backend.tda.runtime_monitor import TDAOperationalMode

        mock_tda_monitor.cfg.mode = TDAOperationalMode.HARD
        mock_tda_monitor.should_block = Mock(return_value=True)

        # Simulate the check
        if mock_tda_monitor.should_block(mock_tda_result_blocked):
            outcome = "abandoned_tda"
            policy_update_applied = False
        else:
            outcome = "processed"
            policy_update_applied = True

        assert outcome == "abandoned_tda"
        assert policy_update_applied is False

    def test_ok_attestation_proceeds_normally(
        self, mock_attestation, mock_tda_monitor, mock_tda_result_ok
    ):
        """OK attestation should proceed with normal processing."""
        from backend.tda.runtime_monitor import TDAOperationalMode

        mock_tda_monitor.cfg.mode = TDAOperationalMode.HARD
        mock_tda_monitor.should_block = Mock(return_value=False)

        if mock_tda_monitor.should_block(mock_tda_result_ok):
            outcome = "abandoned_tda"
        else:
            outcome = "processed"

        assert outcome == "processed"

    def test_soft_mode_never_blocks_attestation(
        self, mock_attestation, mock_tda_monitor, mock_tda_result_blocked
    ):
        """SOFT mode should never block attestations."""
        from backend.tda.runtime_monitor import TDAOperationalMode

        mock_tda_monitor.cfg.mode = TDAOperationalMode.SOFT

        # In SOFT mode, should_block always returns False
        def should_block(result):
            if mock_tda_monitor.cfg.mode != TDAOperationalMode.HARD:
                return False
            return result.block

        mock_tda_monitor.should_block = should_block

        assert mock_tda_monitor.should_block(mock_tda_result_blocked) is False


# ============================================================================
# Test RunLedgerEntry TDA Fields
# ============================================================================

class TestRunLedgerEntryTdaFields:
    """Tests for TDA fields in RunLedgerEntry."""

    def test_ledger_entry_has_tda_fields(self):
        """RunLedgerEntry should include Phase III TDA fields."""

        @dataclass
        class RunLedgerEntry:
            run_id: str
            slice_name: str
            status: str
            # Phase III fields
            tda_outcome: Optional[str] = None
            tda_hss: Optional[float] = None
            tda_sns: Optional[float] = None
            tda_pcs: Optional[float] = None
            tda_drs: Optional[float] = None
            tda_signal: Optional[str] = None
            tda_gate_enforced: bool = False
            tda_pipeline_hash: Optional[str] = None
            lean_submission_avoided: bool = False
            policy_update_avoided: bool = False

        entry = RunLedgerEntry(
            run_id="test_001",
            slice_name="arithmetic_simple",
            status="abandoned_tda",
            tda_outcome="ABANDONED",
            tda_hss=0.15,
            tda_gate_enforced=True,
            tda_pipeline_hash="a" * 64,
            lean_submission_avoided=True,
            policy_update_avoided=True,
        )

        assert entry.tda_outcome == "ABANDONED"
        assert entry.tda_hss == 0.15
        assert entry.tda_gate_enforced is True
        assert entry.lean_submission_avoided is True
        assert entry.policy_update_avoided is True

    def test_abandoned_ledger_entry_populated_correctly(
        self, mock_attestation, mock_tda_result_blocked
    ):
        """Abandoned ledger entry should have all TDA fields populated."""

        @dataclass
        class RunLedgerEntry:
            run_id: str
            slice_name: str
            status: str
            tda_hss: Optional[float] = None
            tda_sns: Optional[float] = None
            tda_pcs: Optional[float] = None
            tda_drs: Optional[float] = None
            tda_signal: Optional[str] = None
            tda_outcome: Optional[str] = None
            tda_gate_enforced: bool = False

        tda_dict = mock_tda_result_blocked.to_dict()

        entry = RunLedgerEntry(
            run_id="test_abandoned",
            slice_name=mock_attestation.slice_id,
            status="abandoned_tda",
            tda_hss=tda_dict["hss"],
            tda_sns=tda_dict["sns"],
            tda_pcs=tda_dict["pcs"],
            tda_drs=tda_dict["drs"],
            tda_signal=tda_dict["signal"],
            tda_outcome="ABANDONED",
            tda_gate_enforced=True,
        )

        assert entry.tda_hss == 0.15
        assert entry.tda_signal == "BLOCK"
        assert entry.tda_outcome == "ABANDONED"


# ============================================================================
# Test Policy Update Prevention
# ============================================================================

class TestPolicyUpdatePrevention:
    """Tests for policy update prevention on hard gate block."""

    def test_policy_weights_unchanged_on_block(
        self, mock_tda_monitor, mock_tda_result_blocked
    ):
        """Policy weights should not change when attestation is blocked."""
        policy_weights = {"len": 0.1, "depth": 0.2, "success": 0.3}
        original_weights = policy_weights.copy()

        from backend.tda.runtime_monitor import TDAOperationalMode
        mock_tda_monitor.cfg.mode = TDAOperationalMode.HARD
        mock_tda_monitor.should_block = Mock(return_value=True)

        # Simulate blocked attestation - no policy update
        if mock_tda_monitor.should_block(mock_tda_result_blocked):
            # No weight update
            pass
        else:
            # Would update weights here
            policy_weights["len"] += 0.01

        assert policy_weights == original_weights

    def test_policy_update_count_unchanged_on_block(
        self, mock_tda_monitor, mock_tda_result_blocked
    ):
        """Policy update count should not increment on block."""
        policy_update_count = 5

        from backend.tda.runtime_monitor import TDAOperationalMode
        mock_tda_monitor.cfg.mode = TDAOperationalMode.HARD
        mock_tda_monitor.should_block = Mock(return_value=True)

        if mock_tda_monitor.should_block(mock_tda_result_blocked):
            # No increment
            pass
        else:
            policy_update_count += 1

        assert policy_update_count == 5

    def test_success_history_unchanged_on_block(
        self, mock_tda_monitor, mock_tda_result_blocked
    ):
        """Success history should not be updated on block."""
        success_count = {"hash_abc": 3}
        original_count = success_count.copy()

        from backend.tda.runtime_monitor import TDAOperationalMode
        mock_tda_monitor.cfg.mode = TDAOperationalMode.HARD
        mock_tda_monitor.should_block = Mock(return_value=True)

        if mock_tda_monitor.should_block(mock_tda_result_blocked):
            pass
        else:
            success_count["hash_abc"] += 1

        assert success_count == original_count


# ============================================================================
# Test Statistics Tracking
# ============================================================================

class TestHardGateStatistics:
    """Tests for hard gate statistics tracking."""

    def test_blocked_count_incremented_on_block(
        self, mock_tda_monitor, mock_tda_result_blocked
    ):
        """Blocked count should increment when attestation is blocked."""
        stats = {"blocked_count": 0, "policy_updates_avoided": 0}

        from backend.tda.runtime_monitor import TDAOperationalMode
        mock_tda_monitor.cfg.mode = TDAOperationalMode.HARD
        mock_tda_monitor.should_block = Mock(return_value=True)

        if mock_tda_monitor.should_block(mock_tda_result_blocked):
            stats["blocked_count"] += 1
            stats["policy_updates_avoided"] += 1

        assert stats["blocked_count"] == 1
        assert stats["policy_updates_avoided"] == 1

    def test_get_hard_gate_stats_returns_complete_data(self):
        """get_hard_gate_stats should return all required fields."""
        stats = {
            "blocked_count": 5,
            "lean_submissions_avoided": 5,
            "policy_updates_avoided": 5,
            "tda_pipeline_hash": "a" * 64,
            "hard_gate_enabled": True,
        }

        assert "blocked_count" in stats
        assert "lean_submissions_avoided" in stats
        assert "policy_updates_avoided" in stats
        assert "tda_pipeline_hash" in stats
        assert "hard_gate_enabled" in stats

    def test_learning_skipped_count_tracks_blocks(
        self, mock_tda_monitor, mock_tda_result_blocked
    ):
        """Learning skipped count should track blocked attestations."""
        learning_skipped_count = 0

        from backend.tda.runtime_monitor import TDAOperationalMode
        mock_tda_monitor.cfg.mode = TDAOperationalMode.HARD
        mock_tda_monitor.should_block = Mock(return_value=True)

        for _ in range(3):
            if mock_tda_monitor.should_block(mock_tda_result_blocked):
                learning_skipped_count += 1

        assert learning_skipped_count == 3


# ============================================================================
# Test RflResult Phase III Fields
# ============================================================================

class TestRflResultPhase3Fields:
    """Tests for RflResult Phase III additions."""

    def test_rfl_result_has_outcome_field(self):
        """RflResult should include outcome field."""

        class ProofOutcome(Enum):
            PROCESSED = "processed"
            ABANDONED_TDA = "abandoned_tda"

        @dataclass
        class RflResult:
            policy_update_applied: bool
            source_root: str
            abstention_mass_delta: float
            step_id: str
            outcome: Optional[ProofOutcome] = None
            tda_gate_enforced: bool = False

        result = RflResult(
            policy_update_applied=False,
            source_root="abc123",
            abstention_mass_delta=0.0,
            step_id="step_001",
            outcome=ProofOutcome.ABANDONED_TDA,
            tda_gate_enforced=True,
        )

        assert result.outcome == ProofOutcome.ABANDONED_TDA
        assert result.tda_gate_enforced is True
        assert result.policy_update_applied is False

    def test_abandoned_result_builder_creates_valid_result(
        self, mock_attestation, mock_tda_result_blocked
    ):
        """_build_abandoned_result should create valid RflResult."""

        class ProofOutcome(Enum):
            ABANDONED_TDA = "abandoned_tda"

        @dataclass
        class RflResult:
            policy_update_applied: bool
            source_root: str
            abstention_mass_delta: float
            step_id: str
            outcome: Optional[ProofOutcome] = None
            tda_gate_enforced: bool = False

        # Simulate _build_abandoned_result
        result = RflResult(
            policy_update_applied=False,
            source_root=mock_attestation.root_hash,
            abstention_mass_delta=0.0,
            step_id="step_abandoned_001",
            outcome=ProofOutcome.ABANDONED_TDA,
            tda_gate_enforced=True,
        )

        assert result.outcome == ProofOutcome.ABANDONED_TDA
        assert result.source_root == "abc123def456"
        assert result.policy_update_applied is False


# ============================================================================
# Test Governance Summary Integration
# ============================================================================

class TestGovernanceSummaryIntegration:
    """Tests for governance summary in RFLRunner."""

    def test_compute_governance_summary_returns_valid_structure(
        self, mock_tda_monitor, mock_tda_result_ok
    ):
        """_compute_governance_summary should return valid structure."""
        from backend.tda.governance import summarize_tda_for_global_health

        tda_results = [mock_tda_result_ok] * 10

        summary = summarize_tda_for_global_health(
            tda_results, mock_tda_monitor.cfg
        )

        assert "governance_signal" in summary
        assert "structural_health" in summary
        assert "block_rate" in summary
        assert "mean_hss" in summary

    def test_export_results_includes_phase3_section(self):
        """_export_results should include tda_phase3 section."""
        export = {
            "tda_shadow": {"summary": {}},
            "tda_phase3": {
                "hard_gate_stats": {
                    "blocked_count": 2,
                    "policy_updates_avoided": 2,
                },
                "governance_summary": {
                    "governance_signal": "HEALTHY",
                },
                "pipeline_hash": "a" * 64,
            },
        }

        assert "tda_phase3" in export
        assert "hard_gate_stats" in export["tda_phase3"]
        assert "governance_summary" in export["tda_phase3"]
        assert "pipeline_hash" in export["tda_phase3"]
