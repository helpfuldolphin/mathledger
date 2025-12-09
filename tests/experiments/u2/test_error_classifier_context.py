"""
PHASE II — NOT USED IN PHASE I

Unit tests for error_classifier context features.

Tests the structured context and improved error messages added
for actionable error reporting.
"""

import json
import subprocess
import unittest

from experiments.u2.runtime.error_classifier import (
    RuntimeErrorKind,
    ErrorContext,
    classify_error,
    classify_error_with_context,
    build_error_result,
)


class TestErrorContextFields(unittest.TestCase):
    """Tests for ErrorContext experiment context fields."""

    def test_context_fields_present(self) -> None:
        """Test that context fields are stored correctly."""
        ctx = ErrorContext(
            kind=RuntimeErrorKind.VALIDATION,
            message="test error",
            slice_name="test_slice",
            cycle=5,
            mode="baseline",
            seed=12345,
        )
        self.assertEqual(ctx.slice_name, "test_slice")
        self.assertEqual(ctx.cycle, 5)
        self.assertEqual(ctx.mode, "baseline")
        self.assertEqual(ctx.seed, 12345)

    def test_context_fields_in_to_dict(self) -> None:
        """Test that context fields are included in to_dict()."""
        ctx = ErrorContext(
            kind=RuntimeErrorKind.FILE_NOT_FOUND,
            message="missing file",
            slice_name="my_slice",
            cycle=10,
            mode="rfl",
            seed=99999,
        )
        d = ctx.to_dict()
        self.assertEqual(d["slice_name"], "my_slice")
        self.assertEqual(d["cycle"], 10)
        self.assertEqual(d["mode"], "rfl")
        self.assertEqual(d["seed"], 99999)

    def test_context_fields_omitted_when_none(self) -> None:
        """Test that None context fields are not in to_dict()."""
        ctx = ErrorContext(
            kind=RuntimeErrorKind.UNKNOWN,
            message="test",
        )
        d = ctx.to_dict()
        self.assertNotIn("slice_name", d)
        self.assertNotIn("cycle", d)
        self.assertNotIn("mode", d)
        self.assertNotIn("seed", d)


class TestErrorContextFormatMessage(unittest.TestCase):
    """Tests for ErrorContext.format_message()."""

    def test_format_message_basic(self) -> None:
        """Test format_message with no context."""
        ctx = ErrorContext(
            kind=RuntimeErrorKind.VALIDATION,
            message="Invalid input",
        )
        self.assertEqual(ctx.format_message(), "Invalid input")

    def test_format_message_with_slice(self) -> None:
        """Test format_message includes slice."""
        ctx = ErrorContext(
            kind=RuntimeErrorKind.VALIDATION,
            message="Invalid input",
            slice_name="test_slice",
        )
        msg = ctx.format_message()
        self.assertIn("Invalid input", msg)
        self.assertIn("slice=test_slice", msg)

    def test_format_message_with_cycle(self) -> None:
        """Test format_message includes cycle."""
        ctx = ErrorContext(
            kind=RuntimeErrorKind.SUBPROCESS,
            message="Command failed",
            cycle=42,
        )
        msg = ctx.format_message()
        self.assertIn("cycle=42", msg)

    def test_format_message_full_context(self) -> None:
        """Test format_message with all context fields."""
        ctx = ErrorContext(
            kind=RuntimeErrorKind.FILE_NOT_FOUND,
            message="File not found: config.yaml",
            slice_name="arithmetic_simple",
            cycle=7,
            mode="baseline",
            seed=12345,
        )
        msg = ctx.format_message()
        self.assertIn("File not found: config.yaml", msg)
        self.assertIn("slice=arithmetic_simple", msg)
        self.assertIn("cycle=7", msg)
        self.assertIn("mode=baseline", msg)
        self.assertIn("seed=12345", msg)


class TestClassifyErrorWithContext(unittest.TestCase):
    """Tests for classify_error_with_context function."""

    def test_file_not_found_with_context(self) -> None:
        """Test FileNotFoundError includes context."""
        try:
            raise FileNotFoundError("config.yaml not found")
        except Exception as e:
            ctx = classify_error_with_context(
                e,
                slice_name="test_slice",
                cycle=5,
                mode="baseline",
                seed=42,
            )

        self.assertEqual(ctx.kind, RuntimeErrorKind.FILE_NOT_FOUND)
        self.assertEqual(ctx.slice_name, "test_slice")
        self.assertEqual(ctx.cycle, 5)
        self.assertEqual(ctx.mode, "baseline")
        self.assertEqual(ctx.seed, 42)

    def test_json_decode_with_context(self) -> None:
        """Test JSONDecodeError includes context."""
        try:
            json.loads("{invalid}")
        except json.JSONDecodeError as e:
            ctx = classify_error_with_context(
                e,
                slice_name="json_slice",
                cycle=10,
            )

        self.assertEqual(ctx.kind, RuntimeErrorKind.JSON_DECODE)
        self.assertEqual(ctx.slice_name, "json_slice")
        self.assertEqual(ctx.cycle, 10)

    def test_subprocess_with_context(self) -> None:
        """Test subprocess error includes context."""
        try:
            raise subprocess.CalledProcessError(
                returncode=127,
                cmd=["test", "command"],
                output="stdout",
                stderr="stderr",
            )
        except subprocess.CalledProcessError as e:
            ctx = classify_error_with_context(
                e,
                slice_name="subprocess_slice",
                cycle=3,
                mode="rfl",
            )

        self.assertEqual(ctx.kind, RuntimeErrorKind.SUBPROCESS)
        self.assertEqual(ctx.slice_name, "subprocess_slice")
        self.assertEqual(ctx.cycle, 3)
        self.assertEqual(ctx.mode, "rfl")

    def test_timeout_classification(self) -> None:
        """Test TimeoutExpired is classified as TIMEOUT."""
        try:
            raise subprocess.TimeoutExpired(cmd=["slow", "cmd"], timeout=30)
        except subprocess.TimeoutExpired as e:
            ctx = classify_error_with_context(e)

        self.assertEqual(ctx.kind, RuntimeErrorKind.TIMEOUT)
        self.assertIn("30", ctx.message)


class TestErrorMessageSpecificity(unittest.TestCase):
    """Tests ensuring error messages are short, specific, and actionable."""

    def test_subprocess_message_includes_exit_code(self) -> None:
        """Test subprocess message includes exit code."""
        try:
            raise subprocess.CalledProcessError(
                returncode=127,
                cmd=["python", "script.py"],
            )
        except subprocess.CalledProcessError as e:
            ctx = classify_error(e)

        self.assertIn("127", ctx.message)

    def test_subprocess_message_includes_command(self) -> None:
        """Test subprocess message includes command."""
        try:
            raise subprocess.CalledProcessError(
                returncode=1,
                cmd=["python", "run.py", "--flag"],
            )
        except subprocess.CalledProcessError as e:
            ctx = classify_error(e)

        self.assertIn("python", ctx.message)

    def test_json_message_includes_position(self) -> None:
        """Test JSON error message includes position."""
        try:
            json.loads("{invalid json at position 5}")
        except json.JSONDecodeError as e:
            ctx = classify_error(e)

        # Should include character position
        self.assertIn("char", ctx.message.lower())

    def test_messages_are_single_line(self) -> None:
        """Test that error messages don't contain newlines."""
        exceptions = [
            FileNotFoundError("missing file"),
            ValueError("bad value"),
            TypeError("wrong type"),
        ]
        for exc in exceptions:
            ctx = classify_error(exc)
            self.assertNotIn("\n", ctx.message)


class TestBuildErrorResultWithContext(unittest.TestCase):
    """Tests for build_error_result with context."""

    def test_build_result_includes_context(self) -> None:
        """Test build_error_result includes experiment context."""
        try:
            raise ValueError("invalid value")
        except Exception as e:
            result = build_error_result(
                e,
                slice_name="test_slice",
                cycle=5,
                mode="baseline",
                seed=42,
            )

        self.assertEqual(result["slice_name"], "test_slice")
        self.assertEqual(result["cycle"], 5)
        self.assertEqual(result["mode"], "baseline")
        self.assertEqual(result["seed"], 42)

    def test_build_result_json_serializable(self) -> None:
        """Test that result with context is JSON serializable."""
        try:
            raise RuntimeError("test error")
        except Exception as e:
            result = build_error_result(
                e,
                slice_name="test",
                cycle=10,
            )

        # Should not raise
        json_str = json.dumps(result)
        self.assertIsInstance(json_str, str)


class TestTimeoutErrorKind(unittest.TestCase):
    """Tests for the TIMEOUT error kind."""

    def test_timeout_kind_exists(self) -> None:
        """Test that TIMEOUT kind exists in enum."""
        self.assertEqual(RuntimeErrorKind.TIMEOUT.value, "timeout")

    def test_timeout_message_includes_duration(self) -> None:
        """Test timeout message includes timeout duration."""
        try:
            raise subprocess.TimeoutExpired(cmd=["slow"], timeout=60)
        except subprocess.TimeoutExpired as e:
            ctx = classify_error(e)

        self.assertEqual(ctx.kind, RuntimeErrorKind.TIMEOUT)
        self.assertIn("60", ctx.message)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)

