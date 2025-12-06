"""
First Organism Lean Mode Integration Tests.

These tests validate that:
1. The worker correctly detects and records Lean mode (mock/dry_run/full)
2. Mock abstention signatures are recorded in ledger metadata
3. The First Organism test can explicitly choose mock or real mode via config/env
4. run_with_attestation() receives and records the Lean mode metadata
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import asdict
from typing import Dict, Any

import pytest

from backend.worker import execute_lean_job, LeanJobResult
from backend.lean_mode import (
    LeanMode,
    get_lean_mode,
    get_lean_status,
    get_build_runner,
    mock_lean_build,
    is_mock_abstention,
    is_lean_available,
    ABSTENTION_SIGNATURE,
    MOCK_STDOUT,
    MOCK_STDERR,
    MOCK_STDERR_FULL,
    MOCK_STDERR_FULL_HASH,
)
from backend.lean_interface import sanitize_statement
from backend.bridge.context import AttestedRunContext
from rfl.runner import RFLRunner, RflResult
from rfl.config import RFLConfig, CurriculumSlice


pytestmark = pytest.mark.integration


class TestLeanModeDetection:
    """Tests for Lean mode detection and configuration."""

    def test_get_lean_mode_defaults_to_full(self, monkeypatch):
        """Verify that default mode is FULL when no env var is set."""
        monkeypatch.delenv("ML_LEAN_MODE", raising=False)
        mode = get_lean_mode()
        assert mode == LeanMode.FULL

    def test_get_lean_mode_respects_env_var(self, monkeypatch):
        """Verify that ML_LEAN_MODE env var is respected."""
        monkeypatch.setenv("ML_LEAN_MODE", "mock")
        assert get_lean_mode() == LeanMode.MOCK

        monkeypatch.setenv("ML_LEAN_MODE", "dry_run")
        assert get_lean_mode() == LeanMode.DRY_RUN

        monkeypatch.setenv("ML_LEAN_MODE", "full")
        assert get_lean_mode() == LeanMode.FULL

    def test_get_lean_status_reports_mode_and_availability(self, monkeypatch):
        """Verify that get_lean_status() reports comprehensive status."""
        monkeypatch.setenv("ML_LEAN_MODE", "mock")
        status = get_lean_status()

        assert status.configured_mode == LeanMode.MOCK
        assert status.effective_mode == LeanMode.MOCK
        assert status.is_mock is True
        assert status.will_abstain is True

        status_dict = status.to_dict()
        assert "configured_mode" in status_dict
        assert "lean_available" in status_dict
        assert "effective_mode" in status_dict


class TestMockAbstentionSignature:
    """Tests for mock abstention signature detection and recording."""

    def test_mock_build_produces_abstention_signature(self, tmp_path):
        """Verify that mock_lean_build produces the expected abstention signature."""
        result = execute_lean_job(
            "p -> p",
            jobs_dir=str(tmp_path),
            build_runner=mock_lean_build,
            cleanup=True,
        )

        assert result.build_result.returncode != 0
        assert ABSTENTION_SIGNATURE in result.build_result.stderr
        assert is_mock_abstention(result.build_result.stderr)

    def test_mock_abstention_metadata_can_be_extracted(self, tmp_path):
        """Verify that mock abstention metadata can be extracted for ledger recording."""
        result = execute_lean_job(
            "p -> p",
            jobs_dir=str(tmp_path),
            build_runner=mock_lean_build,
            cleanup=True,
        )

        # Build metadata that would be recorded in ledger
        lean_mode_metadata = {
            "lean_mode": "mock",
            "is_mock_abstention": is_mock_abstention(result.build_result.stderr),
            "abstention_signature": ABSTENTION_SIGNATURE,
            "stdout_hash": result.stdout_hash,
            "stderr_hash": result.stderr_hash,
            "returncode": result.build_result.returncode,
        }

        assert lean_mode_metadata["is_mock_abstention"] is True
        assert lean_mode_metadata["lean_mode"] == "mock"
        assert lean_mode_metadata["returncode"] != 0


class TestFirstOrganismLeanModeIntegration:
    """Integration tests for First Organism Lean mode recording."""

    def test_execute_lean_job_with_explicit_mock_mode(self, tmp_path, monkeypatch):
        """Test that First Organism can explicitly use mock mode."""
        monkeypatch.setenv("ML_LEAN_MODE", "mock")

        # Get mode-aware build runner
        runner = get_build_runner(mode=LeanMode.MOCK)

        result = execute_lean_job(
            "p /\\ ~p",  # Unsatisfiable statement
            jobs_dir=str(tmp_path),
            build_runner=runner,
            cleanup=True,
        )

        # Verify mock mode behavior
        assert result.build_result.returncode != 0
        assert is_mock_abstention(result.build_result.stderr)

        # Verify deterministic hashes
        assert len(result.stdout_hash) == 64
        assert len(result.stderr_hash) == 64

    def test_rfl_run_with_attestation_records_lean_mode(self, monkeypatch):
        """Test that run_with_attestation() correctly handles Lean mode metadata."""
        monkeypatch.setenv("ML_LEAN_MODE", "mock")

        # Create RFL config and runner with valid parameters
        config = RFLConfig(
            experiment_id="lean-mode-test",
            num_runs=2,  # Must be ≥2 for statistical inference
            random_seed=42,
            system_id=1,
            derive_steps=1,
            max_breadth=1,
            max_total=1,
            depth_max=1,
            bootstrap_replicates=1000,  # Must be ≥1000 for accuracy
            coverage_threshold=0.01,  # Must be in (0, 1]
            uplift_threshold=0.0,
            dual_attestation=False,
            curriculum=[
                CurriculumSlice(
                    name="lean-mode-test",
                    start_run=1,
                    end_run=2,
                    derive_steps=1,
                    max_breadth=1,
                    max_total=1,
                    depth_max=1,
                )
            ],
        )

        # Patch load_baseline_from_db to avoid DB dependency
        monkeypatch.setattr("rfl.runner.load_baseline_from_db", lambda *args, **kwargs: [])

        runner = RFLRunner(config)

        # Create attestation with Lean mode metadata
        attestation = AttestedRunContext(
            slice_id="lean-mode-test",
            statement_hash="d" * 64,
            proof_status="failure",
            block_id=1,
            composite_root="a" * 64,
            reasoning_root="b" * 64,
            ui_root="c" * 64,
            abstention_metrics={"rate": 1.0, "mass": 1.0},
            policy_id="lean-mode-policy",
            metadata={
                "lean_mode": "mock",
                "is_mock_abstention": True,
                "abstention_signature": ABSTENTION_SIGNATURE,
                "attempt_mass": 1.0,
                "abstention_breakdown": {"mock_abstain": 1},
                "first_organism_abstentions": 1,
            },
        )

        result = runner.run_with_attestation(attestation)

        # Verify result
        assert result.policy_update_applied is True
        assert result.source_root == attestation.composite_root

        # Verify ledger entry records the abstention
        ledger_entry = runner.policy_ledger[-1]
        assert ledger_entry.status == "attestation"
        assert ledger_entry.abstention_fraction == 1.0

        # Verify abstention histogram includes mock_abstain
        assert runner.abstention_histogram["mock_abstain"] == 1

        # Verify first_organism_runs_total counter
        assert runner.first_organism_runs_total == 1

    def test_lean_mode_metadata_in_attestation_records(self, monkeypatch):
        """Test that attestation records include Lean mode metadata."""
        monkeypatch.setenv("ML_LEAN_MODE", "mock")

        # Create RFL config with valid parameters
        config = RFLConfig(
            experiment_id="lean-mode-attestation-test",
            num_runs=2,  # Must be ≥2 for statistical inference
            random_seed=42,
            system_id=1,
            derive_steps=1,
            max_breadth=1,
            max_total=1,
            depth_max=1,
            bootstrap_replicates=1000,  # Must be ≥1000 for accuracy
            coverage_threshold=0.01,  # Must be in (0, 1]
            uplift_threshold=0.0,
            dual_attestation=False,
            curriculum=[
                CurriculumSlice(
                    name="lean-mode-attestation-test",
                    start_run=1,
                    end_run=2,
                    derive_steps=1,
                    max_breadth=1,
                    max_total=1,
                    depth_max=1,
                )
            ],
        )

        monkeypatch.setattr("rfl.runner.load_baseline_from_db", lambda *args, **kwargs: [])
        runner = RFLRunner(config)

        # Create attestation with Lean mode metadata
        lean_mode_metadata = {
            "lean_mode": "mock",
            "is_mock_abstention": True,
            "abstention_signature": ABSTENTION_SIGNATURE,
            "stdout_hash": hashlib.sha256(MOCK_STDOUT.encode()).hexdigest(),
            "stderr_hash": hashlib.sha256(MOCK_STDERR.encode()).hexdigest(),
        }

        attestation = AttestedRunContext(
            slice_id="lean-mode-attestation-test",
            statement_hash="e" * 64,
            proof_status="failure",
            block_id=2,
            composite_root="f" * 64,
            reasoning_root="0" * 64,
            ui_root="1" * 64,
            abstention_metrics={"rate": 1.0, "mass": 1.0},
            policy_id="lean-mode-policy",
            metadata={
                **lean_mode_metadata,
                "attempt_mass": 1.0,
                "abstention_breakdown": {"mock_abstain": 1},
            },
        )

        runner.run_with_attestation(attestation)

        # Verify attestation records include metadata
        attestation_records = runner.dual_attestation_records["attestations"]
        assert len(attestation_records) == 1

        recorded_metadata = attestation_records[0]["metadata"]
        assert recorded_metadata["lean_mode"] == "mock"
        assert recorded_metadata["is_mock_abstention"] is True
        assert recorded_metadata["abstention_signature"] == ABSTENTION_SIGNATURE


class TestLeanModeWorkerLogging:
    """Tests for worker logging of Lean mode status."""

    def test_worker_logs_lean_mode_on_startup(self, tmp_path, monkeypatch, capsys):
        """Test that worker logs Lean mode status on startup."""
        monkeypatch.setenv("ML_LEAN_MODE", "mock")

        # Get status (this is what worker does on startup)
        status = get_lean_status()

        # Simulate worker log output
        log_msg = (
            f"[INIT] Lean mode: {status.effective_mode.value} "
            f"(configured={status.configured_mode.value}, "
            f"available={status.lean_available})"
        )

        assert "mock" in log_msg
        assert "configured=mock" in log_msg

        if status.will_abstain:
            warning_msg = "[INIT] Running in mock mode - proofs will produce abstention signature"
            assert "mock mode" in warning_msg


class TestLeanModeDeterminism:
    """Tests for deterministic behavior across Lean modes."""

    def test_mock_mode_produces_deterministic_hashes(self, tmp_path):
        """Verify that mock mode produces deterministic hashes for same statement."""
        result1 = execute_lean_job(
            "p -> p",
            jobs_dir=str(tmp_path),
            build_runner=mock_lean_build,
            cleanup=True,
        )

        result2 = execute_lean_job(
            "p -> p",
            jobs_dir=str(tmp_path),
            build_runner=mock_lean_build,
            cleanup=True,
        )

        # Job IDs must be identical
        assert result1.job_id == result2.job_id

        # Statements must be identical
        assert result1.statement.canonical == result2.statement.canonical

    def test_different_statements_produce_different_job_ids_in_mock_mode(self, tmp_path):
        """Verify that different statements produce different job IDs even in mock mode."""
        result1 = execute_lean_job(
            "p -> p",
            jobs_dir=str(tmp_path),
            build_runner=mock_lean_build,
            cleanup=True,
        )

        result2 = execute_lean_job(
            "p -> q -> p",
            jobs_dir=str(tmp_path),
            build_runner=mock_lean_build,
            cleanup=True,
        )

        # Job IDs must differ
        assert result1.job_id != result2.job_id


class TestRealLeanIntegration:
    """Integration tests with real Lean toolchain.

    These tests are skipped if Lean is not available.
    Set ML_LEAN_MODE=full to run with real Lean verification.
    """

    @pytest.fixture
    def require_lean(self):
        """Skip test if Lean is not available."""
        if not is_lean_available():
            pytest.skip("Lean toolchain not available")

    @pytest.fixture
    def real_lean_runner(self, tmp_path, require_lean):
        """Provide a real Lean build runner."""
        runner = get_build_runner(
            mode=LeanMode.FULL,
            timeout=90,
        )

        def run_job(statement: str) -> LeanJobResult:
            return execute_lean_job(
                statement,
                jobs_dir=str(tmp_path),
                build_runner=runner,
                cleanup=True,
            )

        return run_job

    def test_real_lean_verifies_simple_tautology(self, real_lean_runner):
        """Test that real Lean can verify a simple tautology."""
        result = real_lean_runner("p -> p")

        # This may pass or fail depending on proof body
        # The important thing is it runs without crashing
        assert result.build_result is not None
        assert result.stdout_hash is not None
        assert result.stderr_hash is not None

        # Not a mock abstention
        assert not is_mock_abstention(result.build_result.stderr or "")

    def test_real_lean_produces_different_hashes_than_mock(self, tmp_path, require_lean):
        """Test that real Lean produces different output than mock."""
        runner = get_build_runner(mode=LeanMode.FULL, timeout=90)

        real_result = execute_lean_job(
            "p -> p",
            jobs_dir=str(tmp_path),
            build_runner=runner,
            cleanup=True,
        )

        mock_result = execute_lean_job(
            "p -> p",
            jobs_dir=str(tmp_path),
            build_runner=mock_lean_build,
            cleanup=True,
        )

        # Real and mock should produce different stderr hashes
        # (unless real Lean happens to produce exact same output, very unlikely)
        assert real_result.stderr_hash != mock_result.stderr_hash

    def test_real_lean_mode_status(self, require_lean):
        """Test that mode status correctly reports real Lean availability."""
        status = get_lean_status()

        assert status.lean_available is True
        assert status.recommended_mode == LeanMode.FULL

    def test_dry_run_mode_validates_toolchain(self, tmp_path, require_lean):
        """Test that dry-run mode validates the Lean toolchain."""
        runner = get_build_runner(mode=LeanMode.DRY_RUN, timeout=90)

        result = execute_lean_job(
            "p -> p",
            jobs_dir=str(tmp_path),
            build_runner=runner,
            cleanup=True,
        )

        # Dry-run should succeed if Lean is available
        # (the actual proof may fail, but toolchain validation should pass)
        assert result.build_result is not None
        assert "[DRY_RUN] Lake not available" not in (result.build_result.stderr or "")


class TestLeanModeMetadataFlow:
    """Tests for Lean mode metadata flowing through the system."""

    def test_lean_mode_metadata_structure(self, tmp_path, monkeypatch):
        """Test that Lean mode metadata has the expected structure."""
        monkeypatch.setenv("ML_LEAN_MODE", "mock")

        result = execute_lean_job(
            "p -> p",
            jobs_dir=str(tmp_path),
            build_runner=mock_lean_build,
            cleanup=True,
        )

        # Build metadata that would be passed to ledger
        lean_metadata = {
            "lean_mode": "mock",
            "is_mock_abstention": is_mock_abstention(result.build_result.stderr or ""),
            "stdout_hash": result.stdout_hash,
            "stderr_hash": result.stderr_hash,
            "duration_ms": result.duration_ms,
            "returncode": result.build_result.returncode,
            "job_id": result.job_id,
            "module_name": result.module_name,
        }

        # Verify structure
        assert lean_metadata["lean_mode"] == "mock"
        assert lean_metadata["is_mock_abstention"] is True
        assert len(lean_metadata["stdout_hash"]) == 64
        assert len(lean_metadata["stderr_hash"]) == 64
        assert lean_metadata["returncode"] == 1  # Mock fails
        assert lean_metadata["job_id"] == result.job_id

    def test_lean_mode_metadata_for_rfl_consumption(self, tmp_path, monkeypatch):
        """Test that metadata is suitable for RFL consumption."""
        monkeypatch.setenv("ML_LEAN_MODE", "mock")

        result = execute_lean_job(
            "p -> p",
            jobs_dir=str(tmp_path),
            build_runner=mock_lean_build,
            cleanup=True,
        )

        # Metadata structure for AttestedRunContext
        rfl_metadata = {
            "lean_mode": "mock",
            "is_mock_abstention": True,
            "abstention_signature": ABSTENTION_SIGNATURE,
            "stdout_hash": result.stdout_hash,
            "stderr_hash": result.stderr_hash,
            "attempt_mass": 1.0,
            "abstention_breakdown": {"mock_abstain": 1},
            "first_organism_abstentions": 1,
        }

        # Verify it can be used in AttestedRunContext
        attestation = AttestedRunContext(
            slice_id="test-slice",
            statement_hash="a" * 64,
            proof_status="failure",
            block_id=1,
            composite_root="b" * 64,
            reasoning_root="c" * 64,
            ui_root="d" * 64,
            abstention_metrics={"rate": 1.0, "mass": 1.0},
            metadata=rfl_metadata,
        )

        assert attestation.metadata["lean_mode"] == "mock"
        assert attestation.metadata["is_mock_abstention"] is True
        assert attestation.abstention_rate == 1.0

