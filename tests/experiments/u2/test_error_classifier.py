"""
PHASE II — NOT USED IN PHASE I

Unit tests for experiments.u2.runtime.error_classifier module.

These tests verify:
    - RuntimeErrorKind enum values
    - ErrorContext dataclass
    - classify_error function mappings
    - build_error_result function
"""

import unittest
import json
import subprocess

from experiments.u2.runtime.error_classifier import (
    RuntimeErrorKind,
    ErrorContext,
    classify_error,
    build_error_result,
)


class TestRuntimeErrorKind(unittest.TestCase):
    """Tests for the RuntimeErrorKind enum."""

    def test_subprocess_kind(self) -> None:
        """Test SUBPROCESS error kind."""
        self.assertEqual(RuntimeErrorKind.SUBPROCESS.value, "subprocess")
        self.assertEqual(RuntimeErrorKind.SUBPROCESS.name, "SUBPROCESS")

    def test_json_decode_kind(self) -> None:
        """Test JSON_DECODE error kind."""
        self.assertEqual(RuntimeErrorKind.JSON_DECODE.value, "json_decode")

    def test_file_not_found_kind(self) -> None:
        """Test FILE_NOT_FOUND error kind."""
        self.assertEqual(RuntimeErrorKind.FILE_NOT_FOUND.value, "file_not_found")

    def test_validation_kind(self) -> None:
        """Test VALIDATION error kind."""
        self.assertEqual(RuntimeErrorKind.VALIDATION.value, "validation")

    def test_unknown_kind(self) -> None:
        """Test UNKNOWN error kind."""
        self.assertEqual(RuntimeErrorKind.UNKNOWN.value, "unknown")

    def test_all_kinds_exist(self) -> None:
        """Test that all expected error kinds exist."""
        expected = {"SUBPROCESS", "JSON_DECODE", "FILE_NOT_FOUND", "VALIDATION", "TIMEOUT", "UNKNOWN"}
        actual = {k.name for k in RuntimeErrorKind}
        self.assertEqual(actual, expected)


class TestErrorContext(unittest.TestCase):
    """Tests for the ErrorContext dataclass."""

    def test_create_basic_context(self) -> None:
        """Test creating a basic ErrorContext."""
        ctx = ErrorContext(
            kind=RuntimeErrorKind.FILE_NOT_FOUND,
            message="File not found: test.txt",
        )
        self.assertEqual(ctx.kind, RuntimeErrorKind.FILE_NOT_FOUND)
        self.assertEqual(ctx.message, "File not found: test.txt")
        self.assertIsNone(ctx.traceback_hash)
        self.assertFalse(ctx.recoverable)

    def test_create_full_context(self) -> None:
        """Test creating ErrorContext with all fields."""
        ctx = ErrorContext(
            kind=RuntimeErrorKind.SUBPROCESS,
            message="Command failed",
            traceback_hash="abc123",
            recoverable=False,
            original_exception_type="subprocess.CalledProcessError",
            subprocess_stdout="output",
            subprocess_stderr="error output",
        )
        self.assertEqual(ctx.subprocess_stdout, "output")
        self.assertEqual(ctx.subprocess_stderr, "error output")

    def test_immutability(self) -> None:
        """Test that ErrorContext is immutable."""
        ctx = ErrorContext(
            kind=RuntimeErrorKind.UNKNOWN,
            message="test",
        )
        with self.assertRaises(AttributeError):
            ctx.message = "modified"

    def test_to_dict_basic(self) -> None:
        """Test to_dict with basic context."""
        ctx = ErrorContext(
            kind=RuntimeErrorKind.VALIDATION,
            message="Invalid input",
        )
        result = ctx.to_dict()

        self.assertEqual(result["kind"], "validation")
        self.assertEqual(result["message"], "Invalid input")
        self.assertFalse(result["recoverable"])
        self.assertNotIn("traceback_hash", result)

    def test_to_dict_full(self) -> None:
        """Test to_dict with all fields populated."""
        ctx = ErrorContext(
            kind=RuntimeErrorKind.SUBPROCESS,
            message="Command failed",
            traceback_hash="hash123",
            recoverable=False,
            original_exception_type="subprocess.CalledProcessError",
            subprocess_stdout="stdout",
            subprocess_stderr="stderr",
        )
        result = ctx.to_dict()

        self.assertEqual(result["traceback_hash"], "hash123")
        self.assertEqual(result["exception_type"], "subprocess.CalledProcessError")
        self.assertEqual(result["stdout"], "stdout")
        self.assertEqual(result["stderr"], "stderr")


class TestClassifyError(unittest.TestCase):
    """Tests for the classify_error function."""

    def test_classify_file_not_found(self) -> None:
        """Test classifying FileNotFoundError."""
        try:
            raise FileNotFoundError("config.yaml not found")
        except Exception as e:
            ctx = classify_error(e)

        self.assertEqual(ctx.kind, RuntimeErrorKind.FILE_NOT_FOUND)
        self.assertIn("config.yaml", ctx.message)
        self.assertIsNotNone(ctx.traceback_hash)
        self.assertEqual(len(ctx.traceback_hash), 64)  # SHA256 hex length

    def test_classify_json_decode_error(self) -> None:
        """Test classifying json.JSONDecodeError."""
        try:
            json.loads("{invalid json}")
        except json.JSONDecodeError as e:
            ctx = classify_error(e)

        self.assertEqual(ctx.kind, RuntimeErrorKind.JSON_DECODE)
        self.assertIn("JSON error", ctx.message)

    def test_classify_value_error(self) -> None:
        """Test classifying ValueError."""
        try:
            raise ValueError("Invalid value provided")
        except Exception as e:
            ctx = classify_error(e)

        self.assertEqual(ctx.kind, RuntimeErrorKind.VALIDATION)
        self.assertIn("Invalid value provided", ctx.message)

    def test_classify_type_error(self) -> None:
        """Test classifying TypeError."""
        try:
            raise TypeError("Expected int, got str")
        except Exception as e:
            ctx = classify_error(e)

        self.assertEqual(ctx.kind, RuntimeErrorKind.VALIDATION)

    def test_classify_subprocess_error(self) -> None:
        """Test classifying subprocess.CalledProcessError."""
        try:
            raise subprocess.CalledProcessError(
                returncode=1,
                cmd=["test", "command"],
                output="test output",
                stderr="test stderr",
            )
        except subprocess.CalledProcessError as e:
            ctx = classify_error(e)

        self.assertEqual(ctx.kind, RuntimeErrorKind.SUBPROCESS)
        self.assertIn("exit 1", ctx.message)
        self.assertEqual(ctx.subprocess_stdout, "test output")
        self.assertEqual(ctx.subprocess_stderr, "test stderr")

    def test_classify_unknown_error(self) -> None:
        """Test classifying unknown exception types."""
        try:
            raise RuntimeError("Something unexpected happened")
        except Exception as e:
            ctx = classify_error(e)

        self.assertEqual(ctx.kind, RuntimeErrorKind.UNKNOWN)
        self.assertEqual(ctx.message, "Something unexpected happened")

    def test_classify_custom_exception(self) -> None:
        """Test classifying custom exception class."""
        class CustomError(Exception):
            pass

        try:
            raise CustomError("Custom error message")
        except Exception as e:
            ctx = classify_error(e)

        self.assertEqual(ctx.kind, RuntimeErrorKind.UNKNOWN)
        self.assertIn("CustomError", ctx.original_exception_type)

    def test_traceback_hash_determinism(self) -> None:
        """Test that same exception produces same traceback hash."""
        def raise_error():
            raise ValueError("test error")

        hashes = []
        for _ in range(10):
            try:
                raise_error()
            except Exception as e:
                ctx = classify_error(e)
                hashes.append(ctx.traceback_hash)

        # All hashes should be identical
        self.assertTrue(all(h == hashes[0] for h in hashes))

    def test_exception_type_name_builtin(self) -> None:
        """Test exception type name for builtin exceptions."""
        try:
            raise ValueError("test")
        except Exception as e:
            ctx = classify_error(e)

        self.assertEqual(ctx.original_exception_type, "ValueError")

    def test_exception_type_name_module(self) -> None:
        """Test exception type name for module exceptions."""
        try:
            raise subprocess.CalledProcessError(1, ["cmd"])
        except Exception as e:
            ctx = classify_error(e)

        self.assertEqual(ctx.original_exception_type, "subprocess.CalledProcessError")


class TestBuildErrorResult(unittest.TestCase):
    """Tests for the build_error_result function."""

    def test_basic_error_result(self) -> None:
        """Test building basic error result."""
        try:
            raise ValueError("test error")
        except Exception as e:
            result = build_error_result(e)

        self.assertIn("test error", result["error"])
        self.assertEqual(result["error_kind"], "validation")
        self.assertIn("traceback_hash", result)

    def test_error_result_with_context(self) -> None:
        """Test building error result with additional context."""
        try:
            raise FileNotFoundError("missing file")
        except Exception as e:
            result = build_error_result(e, context={"item": "test_item", "cycle": 5})

        self.assertEqual(result["context"]["item"], "test_item")
        self.assertEqual(result["context"]["cycle"], 5)

    def test_subprocess_error_result(self) -> None:
        """Test building error result for subprocess error."""
        try:
            raise subprocess.CalledProcessError(
                returncode=127,
                cmd=["test"],
                output="stdout content",
                stderr="stderr content",
            )
        except Exception as e:
            result = build_error_result(e)

        self.assertEqual(result["error_kind"], "subprocess")
        self.assertEqual(result["stdout"], "stdout content")
        self.assertEqual(result["stderr"], "stderr content")

    def test_error_result_json_serializable(self) -> None:
        """Test that error result is JSON serializable."""
        try:
            raise RuntimeError("serialization test")
        except Exception as e:
            result = build_error_result(e)

        # Should not raise
        json_str = json.dumps(result)
        self.assertIsInstance(json_str, str)


class TestBackwardCompatibility(unittest.TestCase):
    """
    Tests ensuring backward compatibility with original error handling.

    PHASE II — NOT USED IN PHASE I

    Original code pattern:
        except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError) as e:
            if isinstance(e, subprocess.CalledProcessError):
                mock_result = {"error": str(e), "stdout": e.stdout, "stderr": e.stderr}
            else:
                mock_result = {"error": str(e)}
    """

    def test_subprocess_error_format(self) -> None:
        """Test that subprocess errors produce compatible format."""
        try:
            raise subprocess.CalledProcessError(
                returncode=1,
                cmd=["test"],
                output="out",
                stderr="err",
            )
        except subprocess.CalledProcessError as e:
            ctx = classify_error(e)

            # Original format check
            if ctx.kind == RuntimeErrorKind.SUBPROCESS:
                mock_result = {
                    "error": str(e),
                    "stdout": ctx.subprocess_stdout,
                    "stderr": ctx.subprocess_stderr,
                }
            else:
                mock_result = {"error": str(e)}

        self.assertIn("error", mock_result)
        self.assertIn("stdout", mock_result)
        self.assertIn("stderr", mock_result)

    def test_other_error_format(self) -> None:
        """Test that other errors produce compatible format."""
        try:
            raise FileNotFoundError("missing.txt")
        except Exception as e:
            ctx = classify_error(e)

            if ctx.kind == RuntimeErrorKind.SUBPROCESS:
                mock_result = {
                    "error": str(e),
                    "stdout": ctx.subprocess_stdout,
                    "stderr": ctx.subprocess_stderr,
                }
            else:
                mock_result = {"error": str(e)}

        self.assertIn("error", mock_result)
        self.assertNotIn("stdout", mock_result)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)

