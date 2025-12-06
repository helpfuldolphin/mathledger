"""
Tests for Lean Mode Interface (backend.lean_mode)

These tests verify the three-mode Lean verification strategy:
1. MOCK: No Lean installed; deterministic failure/abstention signature
2. DRY_RUN: Lean installed; minimal test statement validation
3. FULL: Real Lean verification

MDAP Compliance:
- All mock outputs are deterministic (reproducible hashes)
- Abstention signatures are detectable
- Mode selection is explicit and auditable
"""

import hashlib
import os
import subprocess
from unittest.mock import patch, MagicMock

import pytest

from backend.lean_mode import (
    LeanMode,
    LeanModeConfig,
    mock_lean_build,
    dry_run_lean_build,
    full_lean_build,
    get_lean_mode,
    get_build_runner,
    is_mock_abstention,
    is_lean_available,
    recommended_mode,
    get_lean_status,
    MOCK_STDOUT,
    MOCK_STDERR,
    MOCK_STDOUT_HASH,
    MOCK_STDERR_HASH,
    ABSTENTION_SIGNATURE,
)


class TestLeanModeEnum:
    """Tests for LeanMode enumeration."""

    def test_mode_values(self):
        """Verify mode string values."""
        assert LeanMode.MOCK.value == "mock"
        assert LeanMode.DRY_RUN.value == "dry_run"
        assert LeanMode.FULL.value == "full"

    def test_mode_from_string(self):
        """Verify mode can be constructed from string."""
        assert LeanMode("mock") == LeanMode.MOCK
        assert LeanMode("dry_run") == LeanMode.DRY_RUN
        assert LeanMode("full") == LeanMode.FULL

    def test_invalid_mode_raises(self):
        """Invalid mode string should raise ValueError."""
        with pytest.raises(ValueError):
            LeanMode("invalid")


class TestMockLeanBuild:
    """Tests for mock_lean_build() - pure mock mode."""

    def test_returns_completed_process(self):
        """Mock build returns subprocess.CompletedProcess."""
        result = mock_lean_build("ML.Jobs.job_test123")
        assert isinstance(result, subprocess.CompletedProcess)

    def test_deterministic_returncode(self):
        """Mock always returns failure (returncode=1) for abstention."""
        result = mock_lean_build("ML.Jobs.job_abc")
        assert result.returncode == 1

    def test_deterministic_stdout(self):
        """Mock stdout is deterministic."""
        result1 = mock_lean_build("ML.Jobs.job_a")
        result2 = mock_lean_build("ML.Jobs.job_b")
        # Base stdout is the same (module name may vary in stderr)
        assert MOCK_STDOUT in result1.stdout
        assert MOCK_STDOUT in result2.stdout

    def test_contains_abstention_signature(self):
        """Mock stderr contains abstention signature."""
        result = mock_lean_build("ML.Jobs.job_test")
        assert ABSTENTION_SIGNATURE in result.stderr

    def test_module_name_in_args_not_stderr(self):
        """Module name is in args (not stderr) for MDAP determinism.

        IMPORTANT: stderr is deliberately deterministic (same for all statements)
        to ensure reproducible hashes for MDAP compliance. The module name is
        recorded in the args field only, not in output content.
        """
        result = mock_lean_build("ML.Jobs.job_foobar")
        # Module name should be in args, NOT in stderr
        assert "ML.Jobs.job_foobar" in result.args
        # stderr should be deterministic (same signature for all)
        assert "ML.Jobs.job_foobar" not in result.stderr

    def test_args_contain_lake_build(self):
        """Mock result args match lake build invocation."""
        result = mock_lean_build("ML.Jobs.job_xyz")
        assert result.args == ["lake", "build", "ML.Jobs.job_xyz"]

    def test_reproducible_runs(self):
        """Multiple calls produce identical base outputs."""
        results = [mock_lean_build("ML.Jobs.job_same") for _ in range(5)]
        # All should have same returncode
        assert all(r.returncode == 1 for r in results)
        # All should contain signature
        assert all(ABSTENTION_SIGNATURE in r.stderr for r in results)


class TestAbstentionSignature:
    """Tests for mock abstention detection."""

    def test_constant_hash_values(self):
        """Verify pre-computed hashes are correct."""
        expected_stdout_hash = hashlib.sha256(MOCK_STDOUT.encode("utf-8")).hexdigest()
        expected_stderr_hash = hashlib.sha256(MOCK_STDERR.encode("utf-8")).hexdigest()

        assert MOCK_STDOUT_HASH == expected_stdout_hash
        assert MOCK_STDERR_HASH == expected_stderr_hash

    def test_is_mock_abstention_true(self):
        """Detects mock abstention from stderr."""
        result = mock_lean_build("ML.Jobs.job_test")
        assert is_mock_abstention(result.stderr) is True

    def test_is_mock_abstention_false_real_stderr(self):
        """Real Lean errors are not detected as mock."""
        real_stderr = "error: unknown identifier 'xyz'"
        assert is_mock_abstention(real_stderr) is False

    def test_is_mock_abstention_false_empty(self):
        """Empty stderr is not mock abstention."""
        assert is_mock_abstention("") is False

    def test_abstention_signature_format(self):
        """Abstention signature follows expected format."""
        # Format: LEAN_MOCK_ABSTAIN::<hash_prefix>
        assert ABSTENTION_SIGNATURE.startswith("LEAN_MOCK_ABSTAIN::")
        parts = ABSTENTION_SIGNATURE.split("::")
        assert len(parts) == 2
        assert len(parts[1]) == 16  # Hash prefix length


class TestGetLeanMode:
    """Tests for get_lean_mode() environment detection."""

    def test_default_is_full(self):
        """Default mode is FULL when env var not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove ML_LEAN_MODE if present
            os.environ.pop("ML_LEAN_MODE", None)
            mode = get_lean_mode()
            assert mode == LeanMode.FULL

    def test_mock_from_env(self):
        """ML_LEAN_MODE=mock returns MOCK."""
        with patch.dict(os.environ, {"ML_LEAN_MODE": "mock"}):
            assert get_lean_mode() == LeanMode.MOCK

    def test_dry_run_from_env(self):
        """ML_LEAN_MODE=dry_run returns DRY_RUN."""
        with patch.dict(os.environ, {"ML_LEAN_MODE": "dry_run"}):
            assert get_lean_mode() == LeanMode.DRY_RUN

    def test_full_from_env(self):
        """ML_LEAN_MODE=full returns FULL."""
        with patch.dict(os.environ, {"ML_LEAN_MODE": "full"}):
            assert get_lean_mode() == LeanMode.FULL

    def test_case_insensitive(self):
        """Mode detection is case-insensitive."""
        with patch.dict(os.environ, {"ML_LEAN_MODE": "MOCK"}):
            assert get_lean_mode() == LeanMode.MOCK

        with patch.dict(os.environ, {"ML_LEAN_MODE": "Mock"}):
            assert get_lean_mode() == LeanMode.MOCK

    def test_invalid_defaults_to_full(self):
        """Invalid mode value defaults to FULL."""
        with patch.dict(os.environ, {"ML_LEAN_MODE": "invalid_mode"}):
            assert get_lean_mode() == LeanMode.FULL


class TestGetBuildRunner:
    """Tests for get_build_runner() factory."""

    def test_mock_mode_runner(self):
        """Mock mode returns mock_lean_build."""
        runner = get_build_runner(LeanMode.MOCK)
        result = runner("ML.Jobs.job_test")
        assert result.returncode == 1
        assert is_mock_abstention(result.stderr)

    def test_env_mode_detection(self):
        """Runner respects ML_LEAN_MODE environment."""
        with patch.dict(os.environ, {"ML_LEAN_MODE": "mock"}):
            runner = get_build_runner()  # No mode arg, uses env
            result = runner("ML.Jobs.job_env_test")
            assert is_mock_abstention(result.stderr)

    def test_full_mode_returns_callable(self):
        """Full mode returns a callable runner."""
        runner = get_build_runner(LeanMode.FULL, project_dir="/tmp/fake")
        assert callable(runner)

    def test_dry_run_mode_returns_callable(self):
        """Dry-run mode returns a callable runner."""
        runner = get_build_runner(LeanMode.DRY_RUN, project_dir="/tmp/fake")
        assert callable(runner)


class TestLeanModeConfig:
    """Tests for LeanModeConfig dataclass."""

    def test_from_env_defaults(self):
        """Config loads sensible defaults."""
        with patch.dict(os.environ, {}, clear=True):
            # Clear all relevant env vars
            for var in ["ML_LEAN_MODE", "ML_LEAN_DRY_RUN_STATEMENT",
                       "LEAN_PROJECT_DIR", "LEAN_BUILD_TIMEOUT"]:
                os.environ.pop(var, None)

            config = LeanModeConfig.from_env()
            assert config.mode == LeanMode.FULL
            assert config.dry_run_statement == "p -> p"
            assert config.build_timeout == 90

    def test_from_env_custom(self):
        """Config respects custom environment variables."""
        with patch.dict(os.environ, {
            "ML_LEAN_MODE": "mock",
            "ML_LEAN_DRY_RUN_STATEMENT": "q -> q",
            "LEAN_PROJECT_DIR": "/custom/path",
            "LEAN_BUILD_TIMEOUT": "120",
        }):
            config = LeanModeConfig.from_env()
            assert config.mode == LeanMode.MOCK
            assert config.dry_run_statement == "q -> q"
            assert config.lean_project_dir == "/custom/path"
            assert config.build_timeout == 120

    def test_frozen_dataclass(self):
        """Config is immutable (frozen dataclass)."""
        config = LeanModeConfig(
            mode=LeanMode.MOCK,
            dry_run_statement="p -> p",
            lean_project_dir="/tmp",
            build_timeout=60,
        )
        with pytest.raises(AttributeError):
            config.mode = LeanMode.FULL


class TestLeanAvailability:
    """Tests for is_lean_available() and recommended_mode()."""

    def test_is_lean_available_when_lake_works(self):
        """Returns True when lake --version succeeds."""
        mock_result = subprocess.CompletedProcess(
            args=["lake", "--version"],
            returncode=0,
            stdout="Lake version 4.0.0",
            stderr="",
        )
        with patch("subprocess.run", return_value=mock_result):
            assert is_lean_available() is True

    def test_is_lean_available_when_lake_fails(self):
        """Returns False when lake --version fails."""
        mock_result = subprocess.CompletedProcess(
            args=["lake", "--version"],
            returncode=1,
            stdout="",
            stderr="error",
        )
        with patch("subprocess.run", return_value=mock_result):
            assert is_lean_available() is False

    def test_is_lean_available_when_not_found(self):
        """Returns False when lake executable not found."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert is_lean_available() is False

    def test_is_lean_available_timeout(self):
        """Returns False when version check times out."""
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("lake", 10)):
            assert is_lean_available() is False

    def test_recommended_mode_with_lean(self):
        """Recommends FULL when Lean is available."""
        with patch("backend.lean_mode.is_lean_available", return_value=True):
            assert recommended_mode() == LeanMode.FULL

    def test_recommended_mode_without_lean(self):
        """Recommends MOCK when Lean is unavailable."""
        with patch("backend.lean_mode.is_lean_available", return_value=False):
            assert recommended_mode() == LeanMode.MOCK


class TestLeanModeStatus:
    """Tests for get_lean_status() comprehensive status."""

    def test_status_when_lean_available(self):
        """Status reflects Lean availability correctly."""
        with patch("backend.lean_mode.is_lean_available", return_value=True):
            with patch.dict(os.environ, {"ML_LEAN_MODE": "full"}):
                status = get_lean_status()
                assert status.configured_mode == LeanMode.FULL
                assert status.lean_available is True
                assert status.recommended_mode == LeanMode.FULL
                assert status.effective_mode == LeanMode.FULL
                assert status.is_mock is False
                assert status.will_abstain is False

    def test_status_when_lean_unavailable_but_full_configured(self):
        """Falls back to mock when Lean unavailable but configured for full."""
        with patch("backend.lean_mode.is_lean_available", return_value=False):
            with patch.dict(os.environ, {"ML_LEAN_MODE": "full"}):
                status = get_lean_status()
                assert status.configured_mode == LeanMode.FULL
                assert status.lean_available is False
                assert status.effective_mode == LeanMode.MOCK
                assert status.will_abstain is True

    def test_status_mock_mode(self):
        """Status correct for explicit mock mode."""
        with patch("backend.lean_mode.is_lean_available", return_value=True):
            with patch.dict(os.environ, {"ML_LEAN_MODE": "mock"}):
                status = get_lean_status()
                assert status.configured_mode == LeanMode.MOCK
                assert status.effective_mode == LeanMode.MOCK
                assert status.is_mock is True
                assert status.will_abstain is True

    def test_status_to_dict(self):
        """Status serializes to dictionary."""
        with patch("backend.lean_mode.is_lean_available", return_value=True):
            with patch.dict(os.environ, {"ML_LEAN_MODE": "dry_run"}):
                status = get_lean_status()
                d = status.to_dict()
                assert d["configured_mode"] == "dry_run"
                assert d["lean_available"] is True
                assert "effective_mode" in d
                assert "is_mock" in d
                assert "will_abstain" in d


class TestDryRunLeanBuild:
    """Tests for dry_run_lean_build() - validation mode."""

    def test_checks_lake_version_first(self):
        """Dry-run validates lake availability before building."""
        version_result = subprocess.CompletedProcess(
            args=["lake", "--version"],
            returncode=1,
            stdout="",
            stderr="not found",
        )
        with patch("subprocess.run", return_value=version_result):
            result = dry_run_lean_build(
                "ML.Jobs.job_test",
                project_dir="/tmp",
                timeout=10,
            )
            assert result.returncode == 127
            assert "[DRY_RUN]" in result.stderr
            assert "Lake not available" in result.stderr

    def test_lake_not_found(self):
        """Dry-run handles FileNotFoundError for lake."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = dry_run_lean_build(
                "ML.Jobs.job_test",
                project_dir="/tmp",
                timeout=10,
            )
            assert result.returncode == 127
            assert "not found in PATH" in result.stderr

    def test_lake_timeout(self):
        """Dry-run handles timeout during version check."""
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("lake", 10)):
            result = dry_run_lean_build(
                "ML.Jobs.job_test",
                project_dir="/tmp",
                timeout=10,
            )
            assert result.returncode == 124
            assert "timed out" in result.stderr


class TestFullLeanBuild:
    """Tests for full_lean_build() - real verification mode."""

    def test_calls_subprocess_with_correct_args(self):
        """Full build invokes lake build with correct arguments."""
        mock_result = subprocess.CompletedProcess(
            args=["lake", "build", "ML.Jobs.job_test"],
            returncode=0,
            stdout="Build completed",
            stderr="",
        )
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = full_lean_build(
                "ML.Jobs.job_test",
                project_dir="/test/project",
                timeout=60,
            )
            mock_run.assert_called_once_with(
                ["lake", "build", "ML.Jobs.job_test"],
                cwd="/test/project",
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
            )
            assert result.returncode == 0

    def test_handles_timeout(self):
        """Full build handles subprocess timeout."""
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(
            cmd=["lake", "build", "ML.Jobs.job_test"],
            timeout=60,
        )):
            result = full_lean_build(
                "ML.Jobs.job_test",
                project_dir="/test/project",
                timeout=60,
            )
            assert result.returncode == 124
            assert "[TIMEOUT]" in result.stderr


class TestIntegrationWithWorker:
    """Integration tests verifying lean_mode works with worker.py."""

    def test_mock_mode_produces_mock_abstain_prover(self):
        """Worker should detect mock abstention and set prover='mock_abstain'."""
        from backend.lean_mode import is_mock_abstention, ABSTENTION_SIGNATURE

        # Simulate what worker does
        result = mock_lean_build("ML.Jobs.job_integration")
        stderr = result.stderr or ""

        is_mock = is_mock_abstention(stderr)
        assert is_mock is True

        # Worker logic
        if is_mock:
            prover = "mock_abstain"
        else:
            prover = "lean4"

        assert prover == "mock_abstain"

    def test_mock_output_hashes_are_deterministic(self):
        """Mock hashes are stable for MDAP reproducibility."""
        result1 = mock_lean_build("ML.Jobs.job_a")
        result2 = mock_lean_build("ML.Jobs.job_a")

        # Base stdout should be identical
        assert result1.stdout == result2.stdout

        # Compute hashes (like worker does)
        hash1 = hashlib.sha256(result1.stdout.encode("utf-8")).hexdigest()
        hash2 = hashlib.sha256(result2.stdout.encode("utf-8")).hexdigest()
        assert hash1 == hash2
