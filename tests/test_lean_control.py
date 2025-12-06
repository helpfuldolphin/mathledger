"""
Tests for Lean Control Module — Controlled Statements for Safe Failure Testing

These tests verify:
1. Controlled statement registry is populated correctly
2. Statement lookup and detection works
3. Mode switches (FIRST_ORGANISM_REAL_LEAN, FIRST_ORGANISM_CONTROLLED_ONLY) work
4. Controlled build runner applies correct strategy
5. MDAP compliance for controlled statements
"""

import os
import subprocess
from unittest.mock import patch

import pytest

from backend.lean_control import (
    ControlledStatementType,
    ControlledStatement,
    ControlledRunResult,
    ControlledModeStatus,
    CONTROLLED_STATEMENTS,
    CONTRADICTION_PATTERNS,
    SAFE_TAUTOLOGY,
    is_controlled_statement,
    get_controlled_statement,
    should_use_real_lean,
    should_verify_only_controlled,
    get_controlled_build_runner,
    execute_controlled_lean_job,
    get_controlled_mode_status,
)
from backend.lean_mode import ABSTENTION_SIGNATURE, is_mock_abstention


class TestControlledStatementRegistry:
    """Tests for controlled statement registry."""

    def test_registry_has_statements(self):
        """Registry should have controlled statements."""
        assert len(CONTROLLED_STATEMENTS) > 0
        assert len(CONTRADICTION_PATTERNS) > 0

    def test_safe_tautology_exists(self):
        """SAFE_TAUTOLOGY should be defined."""
        assert SAFE_TAUTOLOGY is not None
        assert SAFE_TAUTOLOGY.canonical == "p->p"
        assert SAFE_TAUTOLOGY.expected_success is True

    def test_all_tautologies_expect_success(self):
        """All tautologies should expect success."""
        for canonical, stmt in CONTROLLED_STATEMENTS.items():
            if stmt.statement_type == ControlledStatementType.TAUTOLOGY:
                assert stmt.expected_success is True, f"{canonical} should expect success"

    def test_all_contradictions_expect_failure(self):
        """All contradictions should expect failure."""
        for canonical, stmt in CONTRADICTION_PATTERNS.items():
            if stmt.statement_type == ControlledStatementType.CONTRADICTION:
                assert stmt.expected_success is False, f"{canonical} should expect failure"

    def test_statement_hash_is_deterministic(self):
        """Statement hashes should be deterministic."""
        hash1 = SAFE_TAUTOLOGY.hash
        hash2 = SAFE_TAUTOLOGY.hash
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex length


class TestControlledStatementLookup:
    """Tests for controlled statement lookup functions."""

    def test_is_controlled_statement_true(self):
        """Known statements should be detected as controlled."""
        assert is_controlled_statement("p->p") is True
        assert is_controlled_statement("p->q->p") is True
        assert is_controlled_statement("p/\\q->p") is True

    def test_is_controlled_statement_false(self):
        """Unknown statements should not be detected as controlled."""
        assert is_controlled_statement("x->y->z") is False
        assert is_controlled_statement("unknown") is False
        assert is_controlled_statement("") is False

    def test_get_controlled_statement_found(self):
        """get_controlled_statement returns statement for known canonicals."""
        stmt = get_controlled_statement("p->p")
        assert stmt is not None
        assert stmt.canonical == "p->p"
        assert stmt.expected_success is True

    def test_get_controlled_statement_contradiction(self):
        """get_controlled_statement works for contradictions."""
        stmt = get_controlled_statement("p/\\~p")
        assert stmt is not None
        assert stmt.statement_type == ControlledStatementType.CONTRADICTION
        assert stmt.expected_success is False

    def test_get_controlled_statement_not_found(self):
        """get_controlled_statement returns None for unknown canonicals."""
        stmt = get_controlled_statement("unknown")
        assert stmt is None


class TestControlledModeEnvironment:
    """Tests for environment-based mode switching."""

    def test_should_use_real_lean_default_false(self, monkeypatch):
        """Default should not use real Lean."""
        monkeypatch.delenv("FIRST_ORGANISM_REAL_LEAN", raising=False)
        assert should_use_real_lean() is False

    def test_should_use_real_lean_enabled(self, monkeypatch):
        """FIRST_ORGANISM_REAL_LEAN=true should enable real Lean (if available)."""
        monkeypatch.setenv("FIRST_ORGANISM_REAL_LEAN", "true")
        with patch("backend.lean_control.is_lean_available", return_value=True):
            assert should_use_real_lean() is True

    def test_should_use_real_lean_enabled_but_unavailable(self, monkeypatch):
        """FIRST_ORGANISM_REAL_LEAN=true but Lean unavailable returns False."""
        monkeypatch.setenv("FIRST_ORGANISM_REAL_LEAN", "true")
        with patch("backend.lean_control.is_lean_available", return_value=False):
            assert should_use_real_lean() is False

    def test_should_verify_only_controlled_default_false(self, monkeypatch):
        """Default should not verify only controlled."""
        monkeypatch.delenv("FIRST_ORGANISM_CONTROLLED_ONLY", raising=False)
        assert should_verify_only_controlled() is False

    def test_should_verify_only_controlled_enabled(self, monkeypatch):
        """FIRST_ORGANISM_CONTROLLED_ONLY=true should return True."""
        monkeypatch.setenv("FIRST_ORGANISM_CONTROLLED_ONLY", "true")
        assert should_verify_only_controlled() is True

    def test_various_truthy_values(self, monkeypatch):
        """Various truthy values should work."""
        for value in ["true", "1", "yes", "TRUE", "Yes"]:
            monkeypatch.setenv("FIRST_ORGANISM_CONTROLLED_ONLY", value)
            assert should_verify_only_controlled() is True


class TestControlledBuildRunner:
    """Tests for controlled build runner."""

    def test_runner_is_callable(self):
        """get_controlled_build_runner should return callable."""
        runner = get_controlled_build_runner()
        assert callable(runner)

    def test_runner_returns_completed_process(self):
        """Runner should return CompletedProcess."""
        runner = get_controlled_build_runner()
        result = runner("ML.Jobs.job_test")
        assert isinstance(result, subprocess.CompletedProcess)

    def test_runner_default_uses_mock(self, monkeypatch):
        """Default runner should use mock mode."""
        monkeypatch.delenv("FIRST_ORGANISM_REAL_LEAN", raising=False)
        runner = get_controlled_build_runner()
        result = runner("ML.Jobs.job_test")
        # Should have mock abstention signature
        assert is_mock_abstention(result.stderr)


class TestExecuteControlledLeanJob:
    """Tests for execute_controlled_lean_job function."""

    def test_controlled_statement_returns_result(self):
        """Controlled statement should return proper result."""
        proc, result = execute_controlled_lean_job(
            "p->p",
            "ML.Jobs.job_test",
            force_mock=True,
        )
        assert isinstance(proc, subprocess.CompletedProcess)
        assert isinstance(result, ControlledRunResult)
        assert result.is_controlled is True
        assert result.expected_success is True

    def test_non_controlled_statement(self):
        """Non-controlled statement should be detected."""
        proc, result = execute_controlled_lean_job(
            "unknown_formula",
            "ML.Jobs.job_test",
            force_mock=True,
        )
        assert result.is_controlled is False
        assert result.expected_success is None

    def test_mock_mode_produces_abstention(self):
        """Mock mode should produce abstention signature."""
        proc, result = execute_controlled_lean_job(
            "p->p",
            "ML.Jobs.job_test",
            force_mock=True,
        )
        assert result.used_real_lean is False
        assert result.abstention_signature == ABSTENTION_SIGNATURE

    def test_result_to_metadata(self):
        """ControlledRunResult should serialize to metadata."""
        proc, result = execute_controlled_lean_job(
            "p->p",
            "ML.Jobs.job_test",
            force_mock=True,
        )
        metadata = result.to_metadata()
        assert "is_controlled" in metadata
        assert "used_real_lean" in metadata
        assert "expected_success" in metadata
        assert "actual_success" in metadata
        assert "matches_expectation" in metadata
        assert "abstention_signature" in metadata


class TestControlledModeStatus:
    """Tests for controlled mode status reporting."""

    def test_status_returns_dataclass(self, monkeypatch):
        """get_controlled_mode_status should return status dataclass."""
        monkeypatch.delenv("FIRST_ORGANISM_REAL_LEAN", raising=False)
        monkeypatch.delenv("FIRST_ORGANISM_CONTROLLED_ONLY", raising=False)

        status = get_controlled_mode_status()
        assert isinstance(status, ControlledModeStatus)
        assert status.controlled_statement_count > 0

    def test_status_reflects_environment(self, monkeypatch):
        """Status should reflect environment settings."""
        monkeypatch.setenv("FIRST_ORGANISM_CONTROLLED_ONLY", "true")
        with patch("backend.lean_control.is_lean_available", return_value=False):
            status = get_controlled_mode_status()
            assert status.first_organism_controlled_only is True
            assert status.lean_available is False

    def test_status_to_dict(self, monkeypatch):
        """Status should serialize to dict."""
        monkeypatch.delenv("FIRST_ORGANISM_REAL_LEAN", raising=False)
        status = get_controlled_mode_status()
        d = status.to_dict()
        assert "first_organism_real_lean" in d
        assert "first_organism_controlled_only" in d
        assert "lean_available" in d
        assert "effective_mode" in d
        assert "controlled_statement_count" in d

    def test_effective_mode_full_mock(self, monkeypatch):
        """Effective mode should be full_mock when no flags set."""
        monkeypatch.delenv("FIRST_ORGANISM_REAL_LEAN", raising=False)
        monkeypatch.delenv("FIRST_ORGANISM_CONTROLLED_ONLY", raising=False)
        with patch("backend.lean_control.is_lean_available", return_value=False):
            status = get_controlled_mode_status()
            assert status.effective_mode == "full_mock"


class TestMDAPCompliance:
    """Tests for MDAP compliance in controlled mode."""

    def test_mock_output_is_deterministic(self):
        """Mock output should be deterministic for MDAP."""
        proc1, result1 = execute_controlled_lean_job(
            "p->p",
            "ML.Jobs.job_a",
            force_mock=True,
        )
        proc2, result2 = execute_controlled_lean_job(
            "p->p",
            "ML.Jobs.job_a",
            force_mock=True,
        )
        # Stdout should be identical
        assert proc1.stdout == proc2.stdout
        # Stderr should be identical (abstention signature)
        assert proc1.stderr == proc2.stderr

    def test_different_statements_same_mock_output(self):
        """Different statements should have same mock output for MDAP."""
        proc1, _ = execute_controlled_lean_job(
            "p->p",
            "ML.Jobs.job_a",
            force_mock=True,
        )
        proc2, _ = execute_controlled_lean_job(
            "p->q->p",
            "ML.Jobs.job_b",
            force_mock=True,
        )
        # Mock output should be identical for all statements
        assert proc1.stdout == proc2.stdout


class TestIntegrationWithWorker:
    """Integration tests with worker patterns."""

    def test_worker_can_detect_controlled_statements(self):
        """Worker pattern should detect controlled statements."""
        # Simulate what worker does
        canonical = "p->p"
        is_ctrl = is_controlled_statement(canonical)
        assert is_ctrl is True

        ctrl = get_controlled_statement(canonical)
        assert ctrl is not None
        assert ctrl.expected_success is True

    def test_worker_gets_proof_body_for_controlled(self):
        """Worker can get proof body for controlled statements."""
        ctrl = get_controlled_statement("p->p")
        assert ctrl is not None
        assert "intro" in ctrl.proof_body
        assert "exact" in ctrl.proof_body
