"""
Tests for execute_lean_job() — the canonical Lean verification entry point.

These tests validate:
1. Stdout/stderr hashing is deterministic
2. Job IDs are deterministic (same statement → same ID)
3. Artifact cleanup works correctly
4. LeanMode integration (mock, dry_run, full)
5. Mock abstention signature detection
"""

import hashlib
import subprocess
from pathlib import Path

import pytest

from backend.worker import (
    LeanJobResult,
    TautologyRetryResult,
    execute_lean_job,
    lean_file_path,
    lean_module_name,
    remove_build_artifacts,
    retry_with_tautology_strategy,
    make_lean_source,
)
from backend.lean_interface import sanitize_statement
from backend.lean_mode import (
    LeanMode,
    ABSTENTION_SIGNATURE,
    MOCK_STDOUT,
    MOCK_STDERR,
    MOCK_STDERR_FULL,
    MOCK_STDOUT_HASH,
    MOCK_STDERR_HASH,
    MOCK_STDERR_FULL_HASH,
    mock_lean_build,
    is_mock_abstention,
)


class TestExecuteLeanJobHashing:
    """Tests for stdout/stderr hashing in execute_lean_job."""

    def test_stdout_stderr_hashing_is_deterministic(self, tmp_path):
        """Verify that stdout/stderr hashes are computed correctly."""
        called = []

        def fake_build(module_name: str) -> subprocess.CompletedProcess[str]:
            called.append(module_name)
            return subprocess.CompletedProcess(
                args=["lake", "build", module_name],
                returncode=0,
                stdout="deterministic stdout content",
                stderr="deterministic stderr content",
            )

        result = execute_lean_job(
            "p -> p",
            jobs_dir=str(tmp_path),
            build_runner=fake_build,
            cleanup=True,
        )

        # Verify hashes are computed correctly
        expected_stdout_hash = hashlib.sha256(b"deterministic stdout content").hexdigest()
        expected_stderr_hash = hashlib.sha256(b"deterministic stderr content").hexdigest()

        assert result.stdout_hash == expected_stdout_hash
        assert result.stderr_hash == expected_stderr_hash

    def test_empty_stdout_stderr_hashing(self, tmp_path):
        """Verify that empty stdout/stderr are hashed correctly."""

        def empty_build(module_name: str) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=["lake", "build", module_name],
                returncode=0,
                stdout="",
                stderr="",
            )

        result = execute_lean_job(
            "p -> p",
            jobs_dir=str(tmp_path),
            build_runner=empty_build,
            cleanup=True,
        )

        # Empty string hash
        empty_hash = hashlib.sha256(b"").hexdigest()
        assert result.stdout_hash == empty_hash
        assert result.stderr_hash == empty_hash

    def test_none_stdout_stderr_treated_as_empty(self, tmp_path):
        """Verify that None stdout/stderr are treated as empty strings."""

        def none_build(module_name: str) -> subprocess.CompletedProcess[str]:
            cp = subprocess.CompletedProcess(
                args=["lake", "build", module_name],
                returncode=0,
                stdout=None,
                stderr=None,
            )
            return cp

        result = execute_lean_job(
            "p -> p",
            jobs_dir=str(tmp_path),
            build_runner=none_build,
            cleanup=True,
        )

        # None should be treated as empty string
        empty_hash = hashlib.sha256(b"").hexdigest()
        assert result.stdout_hash == empty_hash
        assert result.stderr_hash == empty_hash


class TestExecuteLeanJobDeterminism:
    """Tests for deterministic job ID generation."""

    def test_same_statement_produces_same_job_id(self, tmp_path):
        """Verify that the same statement always produces the same job ID."""

        def noop_build(module_name: str) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=["lake", "build", module_name],
                returncode=0,
                stdout="",
                stderr="",
            )

        result1 = execute_lean_job(
            "p -> q -> p",
            jobs_dir=str(tmp_path),
            build_runner=noop_build,
            cleanup=True,
        )

        result2 = execute_lean_job(
            "p -> q -> p",
            jobs_dir=str(tmp_path),
            build_runner=noop_build,
            cleanup=True,
        )

        # Job IDs must be identical for identical statements
        assert result1.job_id == result2.job_id

    def test_different_statements_produce_different_job_ids(self, tmp_path):
        """Verify that different statements produce different job IDs."""

        def noop_build(module_name: str) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=["lake", "build", module_name],
                returncode=0,
                stdout="",
                stderr="",
            )

        result1 = execute_lean_job(
            "p -> p",
            jobs_dir=str(tmp_path),
            build_runner=noop_build,
            cleanup=True,
        )

        result2 = execute_lean_job(
            "p -> q -> p",
            jobs_dir=str(tmp_path),
            build_runner=noop_build,
            cleanup=True,
        )

        # Job IDs must differ for different statements
        assert result1.job_id != result2.job_id

    def test_normalized_statements_produce_same_job_id(self, tmp_path):
        """Verify that equivalent statements (after normalization) produce the same job ID."""

        def noop_build(module_name: str) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=["lake", "build", module_name],
                returncode=0,
                stdout="",
                stderr="",
            )

        # These should normalize to the same canonical form (whitespace variations)
        result1 = execute_lean_job(
            "p -> p",
            jobs_dir=str(tmp_path),
            build_runner=noop_build,
            cleanup=True,
        )

        result2 = execute_lean_job(
            "p  ->   p",  # Extra whitespace
            jobs_dir=str(tmp_path),
            build_runner=noop_build,
            cleanup=True,
        )

        # Normalized forms should produce the same job ID
        assert result1.job_id == result2.job_id
        assert result1.statement.canonical == result2.statement.canonical


class TestExecuteLeanJobArtifactCleanup:
    """Tests for artifact cleanup in execute_lean_job."""

    def test_cleanup_true_removes_lean_file(self, tmp_path):
        """Verify that cleanup=True removes the generated Lean file."""

        def noop_build(module_name: str) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=["lake", "build", module_name],
                returncode=0,
                stdout="",
                stderr="",
            )

        result = execute_lean_job(
            "p -> p",
            jobs_dir=str(tmp_path),
            build_runner=noop_build,
            cleanup=True,
        )

        # File should be cleaned up
        lean_path = Path(lean_file_path(str(tmp_path), result.job_id))
        assert not lean_path.exists()

    def test_cleanup_false_preserves_lean_file(self, tmp_path):
        """Verify that cleanup=False preserves the generated Lean file."""

        def noop_build(module_name: str) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=["lake", "build", module_name],
                returncode=0,
                stdout="",
                stderr="",
            )

        result = execute_lean_job(
            "p -> p",
            jobs_dir=str(tmp_path),
            build_runner=noop_build,
            cleanup=False,
        )

        # File should still exist
        lean_path = Path(lean_file_path(str(tmp_path), result.job_id))
        assert lean_path.exists()

        # Verify content
        content = lean_path.read_text(encoding="utf-8")
        assert "theorem job_" in content
        assert result.job_id in content

        # Manual cleanup
        lean_path.unlink()


class TestExecuteLeanJobMockMode:
    """Tests for LeanMode.MOCK integration."""

    def test_mock_build_produces_abstention_signature(self, tmp_path):
        """Verify that mock_lean_build produces the expected abstention signature."""
        result = execute_lean_job(
            "p -> p",
            jobs_dir=str(tmp_path),
            build_runner=mock_lean_build,
            cleanup=True,
        )

        # Mock build should fail (abstain)
        assert result.build_result.returncode != 0

        # Abstention signature should be in stderr
        assert ABSTENTION_SIGNATURE in result.build_result.stderr
        assert is_mock_abstention(result.build_result.stderr)

    def test_mock_build_hashes_are_deterministic(self, tmp_path):
        """Verify that mock build produces fully deterministic hashes."""
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

        # stdout hash must be identical (deterministic)
        assert result1.stdout_hash == result2.stdout_hash
        assert result1.stdout_hash == MOCK_STDOUT_HASH

        # stderr hash must be identical (fully deterministic - no module name)
        assert result1.stderr_hash == result2.stderr_hash
        assert result1.stderr_hash == MOCK_STDERR_FULL_HASH

    def test_mock_build_hashes_same_across_different_statements(self, tmp_path):
        """Verify that mock build produces identical hashes for different statements.

        This is critical for MDAP compliance: mock mode must produce reproducible
        hashes regardless of the input statement.
        """
        result1 = execute_lean_job(
            "p -> p",
            jobs_dir=str(tmp_path),
            build_runner=mock_lean_build,
            cleanup=True,
        )

        result2 = execute_lean_job(
            "p -> q -> p",  # Different statement
            jobs_dir=str(tmp_path),
            build_runner=mock_lean_build,
            cleanup=True,
        )

        # stdout/stderr hashes must be identical regardless of statement
        assert result1.stdout_hash == result2.stdout_hash
        assert result1.stderr_hash == result2.stderr_hash

        # Both must match the precomputed constants
        assert result1.stdout_hash == MOCK_STDOUT_HASH
        assert result1.stderr_hash == MOCK_STDERR_FULL_HASH

    def test_is_mock_abstention_detection(self):
        """Verify is_mock_abstention() correctly detects mock abstentions."""
        # Should detect mock abstention
        mock_stderr = f"Some error\n{ABSTENTION_SIGNATURE}\nMore error"
        assert is_mock_abstention(mock_stderr)

        # Should not detect real failures
        real_stderr = "type mismatch\nexpected: Prop\ngot: Nat"
        assert not is_mock_abstention(real_stderr)

        # Empty stderr
        assert not is_mock_abstention("")


class TestExecuteLeanJobResultStructure:
    """Tests for LeanJobResult structure and fields."""

    def test_result_contains_all_required_fields(self, tmp_path):
        """Verify that LeanJobResult contains all required fields."""

        def detailed_build(module_name: str) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=["lake", "build", module_name],
                returncode=0,
                stdout="Build successful",
                stderr="",
            )

        result = execute_lean_job(
            "p -> p",
            jobs_dir=str(tmp_path),
            build_runner=detailed_build,
            cleanup=True,
        )

        # Check all fields are present and have correct types
        assert isinstance(result, LeanJobResult)
        assert isinstance(result.job_id, str)
        assert len(result.job_id) == 12  # Truncated UUID hex

        assert result.statement is not None
        assert result.statement.canonical == "p->p"

        assert isinstance(result.lean_source, str)
        assert "theorem job_" in result.lean_source

        assert isinstance(result.build_result, subprocess.CompletedProcess)
        assert result.build_result.returncode == 0

        assert isinstance(result.stdout_hash, str)
        assert len(result.stdout_hash) == 64  # SHA256 hex

        assert isinstance(result.stderr_hash, str)
        assert len(result.stderr_hash) == 64

        assert isinstance(result.module_name, str)
        assert result.module_name == lean_module_name(result.job_id)

        assert isinstance(result.duration_ms, int)
        assert result.duration_ms >= 0

    def test_result_module_name_matches_job_id(self, tmp_path):
        """Verify that module_name is correctly derived from job_id."""

        def noop_build(module_name: str) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=["lake", "build", module_name],
                returncode=0,
                stdout="",
                stderr="",
            )

        result = execute_lean_job(
            "p -> p",
            jobs_dir=str(tmp_path),
            build_runner=noop_build,
            cleanup=True,
        )

        expected_module_name = f"ML.Jobs.job_{result.job_id}"
        assert result.module_name == expected_module_name


class TestExecuteLeanJobErrorHandling:
    """Tests for error handling in execute_lean_job."""

    def test_empty_statement_raises_value_error(self, tmp_path):
        """Verify that empty statements raise ValueError."""

        def noop_build(module_name: str) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=["lake", "build", module_name],
                returncode=0,
                stdout="",
                stderr="",
            )

        with pytest.raises(ValueError, match="empty"):
            execute_lean_job(
                "",
                jobs_dir=str(tmp_path),
                build_runner=noop_build,
                cleanup=True,
            )

    def test_whitespace_only_statement_raises_value_error(self, tmp_path):
        """Verify that whitespace-only statements raise ValueError."""

        def noop_build(module_name: str) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=["lake", "build", module_name],
                returncode=0,
                stdout="",
                stderr="",
            )

        with pytest.raises(ValueError, match="empty"):
            execute_lean_job(
                "   \n\t  ",
                jobs_dir=str(tmp_path),
                build_runner=noop_build,
                cleanup=True,
            )


class TestLeanModeConstants:
    """Tests for LeanMode constants used in First Organism."""

    def test_abstention_signature_format(self):
        """Verify ABSTENTION_SIGNATURE has expected format."""
        assert ABSTENTION_SIGNATURE.startswith("LEAN_MOCK_ABSTAIN::")
        # Should contain a hash prefix
        assert len(ABSTENTION_SIGNATURE) > len("LEAN_MOCK_ABSTAIN::")

    def test_mock_hashes_are_precomputed_correctly(self):
        """Verify that MOCK_STDOUT_HASH and MOCK_STDERR_HASH are correct."""
        computed_stdout_hash = hashlib.sha256(MOCK_STDOUT.encode("utf-8")).hexdigest()
        computed_stderr_hash = hashlib.sha256(MOCK_STDERR.encode("utf-8")).hexdigest()
        computed_stderr_full_hash = hashlib.sha256(MOCK_STDERR_FULL.encode("utf-8")).hexdigest()

        assert MOCK_STDOUT_HASH == computed_stdout_hash
        assert MOCK_STDERR_HASH == computed_stderr_hash
        assert MOCK_STDERR_FULL_HASH == computed_stderr_full_hash

        # Verify MOCK_STDERR_FULL contains the signature
        assert ABSTENTION_SIGNATURE in MOCK_STDERR_FULL


class TestRetryWithTautologyStrategy:
    """Tests for retry_with_tautology_strategy() — the Verification Ladder fallback."""

    def test_successful_tautology_retry(self, tmp_path):
        """Verify that tautology retry succeeds when build passes."""
        stmt = sanitize_statement("p -> p")

        def success_build(module_name: str) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=["lake", "build", module_name],
                returncode=0,
                stdout="Build successful with decide",
                stderr="",
            )

        result = retry_with_tautology_strategy(
            job_id="test123",
            stmt=stmt,
            jobs_dir=str(tmp_path),
            build_runner=success_build,
        )

        assert isinstance(result, TautologyRetryResult)
        assert result.success is True
        assert result.build_result.returncode == 0
        assert "classical" in result.lean_source
        assert "decide" in result.lean_source
        assert result.duration_ms >= 0

    def test_failed_tautology_retry(self, tmp_path):
        """Verify that tautology retry correctly reports failure."""
        stmt = sanitize_statement("p /\\ ~p")

        def fail_build(module_name: str) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=["lake", "build", module_name],
                returncode=1,
                stdout="",
                stderr="type mismatch",
            )

        result = retry_with_tautology_strategy(
            job_id="fail123",
            stmt=stmt,
            jobs_dir=str(tmp_path),
            build_runner=fail_build,
        )

        assert result.success is False
        assert result.build_result.returncode == 1
        assert "type mismatch" in result.build_result.stderr

    def test_tautology_retry_creates_lean_file(self, tmp_path):
        """Verify that retry creates the Lean file with decide tactic."""
        stmt = sanitize_statement("p -> p")

        def noop_build(module_name: str) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=["lake", "build", module_name],
                returncode=0,
                stdout="",
                stderr="",
            )

        result = retry_with_tautology_strategy(
            job_id="file123",
            stmt=stmt,
            jobs_dir=str(tmp_path),
            build_runner=noop_build,
        )

        # Check the file was created
        lean_path = Path(lean_file_path(str(tmp_path), "file123"))
        assert lean_path.exists()

        # Check content
        content = lean_path.read_text(encoding="utf-8")
        assert "classical" in content
        assert "decide" in content
        assert "theorem job_file123" in content

        # Cleanup
        lean_path.unlink()

    def test_tautology_retry_uses_mode_aware_runner(self, tmp_path):
        """Verify that retry uses the provided build runner (mode-aware)."""
        stmt = sanitize_statement("p -> p")
        calls = []

        def tracking_build(module_name: str) -> subprocess.CompletedProcess[str]:
            calls.append(module_name)
            return subprocess.CompletedProcess(
                args=["lake", "build", module_name],
                returncode=0,
                stdout="",
                stderr="",
            )

        retry_with_tautology_strategy(
            job_id="track123",
            stmt=stmt,
            jobs_dir=str(tmp_path),
            build_runner=tracking_build,
        )

        assert len(calls) == 1
        assert calls[0] == "ML.Jobs.job_track123"

    def test_tautology_retry_handles_write_error(self, tmp_path, monkeypatch):
        """Verify that retry handles file write errors gracefully."""
        stmt = sanitize_statement("p -> p")

        # Make the directory read-only to cause write failure
        # Use a non-existent path that can't be created
        bad_path = str(tmp_path / "nonexistent" / "deep" / "path")

        # Patch write_lean_file to raise an error
        from backend import worker

        def failing_write(path: str, content: str) -> None:
            raise PermissionError("Cannot write file")

        monkeypatch.setattr(worker, "write_lean_file", failing_write)

        def noop_build(module_name: str) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=["lake", "build", module_name],
                returncode=0,
                stdout="",
                stderr="",
            )

        result = retry_with_tautology_strategy(
            job_id="error123",
            stmt=stmt,
            jobs_dir=str(tmp_path),
            build_runner=noop_build,
        )

        # Should fail gracefully
        assert result.success is False
        assert result.build_result.returncode == 1
        assert "Cannot write file" in result.build_result.stderr
        assert result.duration_ms == 0

    def test_tautology_retry_measures_duration(self, tmp_path):
        """Verify that retry correctly measures build duration."""
        stmt = sanitize_statement("p -> p")

        def slow_build(module_name: str) -> subprocess.CompletedProcess[str]:
            import time
            time.sleep(0.05)  # 50ms delay
            return subprocess.CompletedProcess(
                args=["lake", "build", module_name],
                returncode=0,
                stdout="",
                stderr="",
            )

        result = retry_with_tautology_strategy(
            job_id="slow123",
            stmt=stmt,
            jobs_dir=str(tmp_path),
            build_runner=slow_build,
        )

        # Duration should be at least 50ms
        assert result.duration_ms >= 50

    def test_tautology_retry_lean_source_structure(self, tmp_path):
        """Verify the generated Lean source has correct structure."""
        stmt = sanitize_statement("p -> q -> p")

        def noop_build(module_name: str) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=["lake", "build", module_name],
                returncode=0,
                stdout="",
                stderr="",
            )

        result = retry_with_tautology_strategy(
            job_id="struct123",
            stmt=stmt,
            jobs_dir=str(tmp_path),
            build_runner=noop_build,
        )

        # Check structure
        assert "import Mathlib" in result.lean_source
        assert "namespace ML.Jobs" in result.lean_source
        assert "theorem job_struct123" in result.lean_source
        assert "classical" in result.lean_source
        assert "decide" in result.lean_source
        assert "end ML.Jobs" in result.lean_source
