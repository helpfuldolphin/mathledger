"""
PHASE II — NOT USED IN PHASE I

Error Classifier Module
=======================

Provides runtime error classification utilities for U2 uplift experiments.
Extracts and centralizes the error handling logic from experiments/run_uplift_u2.py.

This module classifies exceptions into semantic categories without changing
which exceptions are caught or how they affect exit codes. It provides:
    - RuntimeErrorKind: Enum of error categories
    - ErrorContext: Structured error information with experiment context
    - classify_error: Exception to ErrorContext mapping
    - classify_error_with_context: Exception mapping with slice/cycle info

ERROR MESSAGE PHILOSOPHY
------------------------
Error messages in this module are:
- **Short**: One line summary, no speculation
- **Specific**: Includes concrete values (return codes, positions, paths)
- **Actionable**: Points to what failed, not why (unless we know for certain)

Example:
    >>> try:
    ...     raise FileNotFoundError("config.yaml not found")
    ... except Exception as e:
    ...     ctx = classify_error(e)
    ...     print(ctx.kind)
    RuntimeErrorKind.FILE_NOT_FOUND
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class RuntimeErrorKind(Enum):
    """
    Enumeration of runtime error categories for U2 experiments.

    These categories match the exception types handled in the original
    experiments/run_uplift_u2.py implementation:
        - SUBPROCESS: subprocess.CalledProcessError
        - JSON_DECODE: json.JSONDecodeError
        - FILE_NOT_FOUND: FileNotFoundError
        - VALIDATION: ValueError, TypeError (input validation)
        - TIMEOUT: subprocess.TimeoutExpired
        - UNKNOWN: Any other exception type

    Example:
        >>> RuntimeErrorKind.SUBPROCESS.value
        'subprocess'
        >>> RuntimeErrorKind.UNKNOWN.name
        'UNKNOWN'
    """

    SUBPROCESS = "subprocess"
    JSON_DECODE = "json_decode"
    FILE_NOT_FOUND = "file_not_found"
    VALIDATION = "validation"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ErrorContext:
    """
    Structured context for a classified runtime error.

    This dataclass provides a consistent representation of errors across
    the U2 runtime, enabling structured logging and error handling.

    Attributes:
        kind: The semantic category of the error.
        message: Human-readable error description (short, specific).
        traceback_hash: SHA-256 hash of the traceback string (if available).
        recoverable: Whether the error is potentially recoverable.
        original_exception_type: Fully qualified type name of the exception.
        subprocess_stdout: Captured stdout (for SUBPROCESS errors only).
        subprocess_stderr: Captured stderr (for SUBPROCESS errors only).
        slice_name: Name of the slice where error occurred (if available).
        cycle: Cycle index where error occurred (if available).
        mode: Execution mode when error occurred (if available).
        seed: Seed value when error occurred (if available).

    Example:
        >>> ctx = ErrorContext(
        ...     kind=RuntimeErrorKind.FILE_NOT_FOUND,
        ...     message="config.yaml not found",
        ...     slice_name="arithmetic_simple",
        ...     cycle=5,
        ... )
    """

    kind: RuntimeErrorKind
    message: str
    traceback_hash: Optional[str] = None
    recoverable: bool = False
    original_exception_type: Optional[str] = None
    subprocess_stdout: Optional[str] = None
    subprocess_stderr: Optional[str] = None
    # Experiment context fields
    slice_name: Optional[str] = None
    cycle: Optional[int] = None
    mode: Optional[str] = None
    seed: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation suitable for logging.
        """
        result: Dict[str, Any] = {
            "kind": self.kind.value,
            "message": self.message,
            "recoverable": self.recoverable,
        }
        if self.traceback_hash:
            result["traceback_hash"] = self.traceback_hash
        if self.original_exception_type:
            result["exception_type"] = self.original_exception_type
        if self.subprocess_stdout is not None:
            result["stdout"] = self.subprocess_stdout
        if self.subprocess_stderr is not None:
            result["stderr"] = self.subprocess_stderr
        # Include experiment context if available
        if self.slice_name is not None:
            result["slice_name"] = self.slice_name
        if self.cycle is not None:
            result["cycle"] = self.cycle
        if self.mode is not None:
            result["mode"] = self.mode
        if self.seed is not None:
            result["seed"] = self.seed
        return result

    def format_message(self) -> str:
        """
        Format a human-readable error message with context.

        Returns:
            Formatted message including slice/cycle context if available.
        """
        parts = [self.message]
        context_parts = []
        if self.slice_name:
            context_parts.append(f"slice={self.slice_name}")
        if self.cycle is not None:
            context_parts.append(f"cycle={self.cycle}")
        if self.mode:
            context_parts.append(f"mode={self.mode}")
        if self.seed is not None:
            context_parts.append(f"seed={self.seed}")
        
        if context_parts:
            parts.append(f"[{', '.join(context_parts)}]")
        
        return " ".join(parts)


def _compute_traceback_hash(tb_string: str) -> str:
    """
    Compute a stable SHA-256 hash of a traceback string.

    Args:
        tb_string: The formatted traceback string.

    Returns:
        64-character lowercase hexadecimal hash.
    """
    return hashlib.sha256(tb_string.encode("utf-8")).hexdigest()


def _get_exception_type_name(exc: Exception) -> str:
    """
    Get the fully qualified type name of an exception.

    Args:
        exc: The exception instance.

    Returns:
        String like "subprocess.CalledProcessError".
    """
    exc_type = type(exc)
    module = exc_type.__module__
    if module == "builtins":
        return exc_type.__name__
    return f"{module}.{exc_type.__name__}"


def _format_subprocess_message(exc: subprocess.CalledProcessError) -> str:
    """Format a short, specific message for subprocess failures."""
    cmd_str = " ".join(exc.cmd) if isinstance(exc.cmd, list) else str(exc.cmd)
    # Truncate command if too long
    if len(cmd_str) > 60:
        cmd_str = cmd_str[:57] + "..."
    return f"Subprocess exit {exc.returncode}: {cmd_str}"


def _format_json_message(exc: json.JSONDecodeError) -> str:
    """Format a short, specific message for JSON decode errors."""
    return f"JSON error at char {exc.pos}: {exc.msg}"


def _format_timeout_message(exc: subprocess.TimeoutExpired) -> str:
    """Format a short, specific message for timeout errors."""
    cmd_str = " ".join(exc.cmd) if isinstance(exc.cmd, list) else str(exc.cmd)
    if len(cmd_str) > 40:
        cmd_str = cmd_str[:37] + "..."
    return f"Timeout after {exc.timeout}s: {cmd_str}"


def classify_error(exc: Exception) -> ErrorContext:
    """
    Classify an exception into a structured ErrorContext.

    This function maps exceptions to semantic categories matching the
    original error handling in experiments/run_uplift_u2.py:

    - subprocess.CalledProcessError -> SUBPROCESS
    - subprocess.TimeoutExpired -> TIMEOUT
    - json.JSONDecodeError -> JSON_DECODE
    - FileNotFoundError -> FILE_NOT_FOUND
    - ValueError, TypeError -> VALIDATION
    - All others -> UNKNOWN

    Args:
        exc: The exception to classify.

    Returns:
        ErrorContext with structured error information.

    Example:
        >>> try:
        ...     subprocess.run(["false"], check=True)
        ... except subprocess.CalledProcessError as e:
        ...     ctx = classify_error(e)
        ...     ctx.kind
        RuntimeErrorKind.SUBPROCESS
    """
    return classify_error_with_context(exc)


def classify_error_with_context(
    exc: Exception,
    *,
    slice_name: Optional[str] = None,
    cycle: Optional[int] = None,
    mode: Optional[str] = None,
    seed: Optional[int] = None,
) -> ErrorContext:
    """
    Classify an exception with experiment context.

    This function is the preferred way to classify errors during experiment
    execution, as it captures the slice/cycle/mode/seed context for
    actionable error messages.

    Args:
        exc: The exception to classify.
        slice_name: Name of the slice where error occurred.
        cycle: Cycle index where error occurred.
        mode: Execution mode ("baseline" or "rfl").
        seed: Seed value for the cycle.

    Returns:
        ErrorContext with structured error and experiment context.

    Example:
        >>> try:
        ...     raise FileNotFoundError("missing.txt")
        ... except Exception as e:
        ...     ctx = classify_error_with_context(
        ...         e, slice_name="test", cycle=5, mode="baseline"
        ...     )
        ...     ctx.format_message()
        '[Errno 2] No such file or directory: ...' [slice=test, cycle=5, mode=baseline]
    """
    # Compute traceback hash
    tb_string = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    tb_hash = _compute_traceback_hash(tb_string) if tb_string else None

    # Get exception type name
    exc_type_name = _get_exception_type_name(exc)

    # Classify by exception type with specific messages
    if isinstance(exc, subprocess.TimeoutExpired):
        return ErrorContext(
            kind=RuntimeErrorKind.TIMEOUT,
            message=_format_timeout_message(exc),
            traceback_hash=tb_hash,
            recoverable=False,
            original_exception_type=exc_type_name,
            slice_name=slice_name,
            cycle=cycle,
            mode=mode,
            seed=seed,
        )

    if isinstance(exc, subprocess.CalledProcessError):
        return ErrorContext(
            kind=RuntimeErrorKind.SUBPROCESS,
            message=_format_subprocess_message(exc),
            traceback_hash=tb_hash,
            recoverable=False,
            original_exception_type=exc_type_name,
            subprocess_stdout=exc.stdout if hasattr(exc, "stdout") else None,
            subprocess_stderr=exc.stderr if hasattr(exc, "stderr") else None,
            slice_name=slice_name,
            cycle=cycle,
            mode=mode,
            seed=seed,
        )

    if isinstance(exc, json.JSONDecodeError):
        return ErrorContext(
            kind=RuntimeErrorKind.JSON_DECODE,
            message=_format_json_message(exc),
            traceback_hash=tb_hash,
            recoverable=False,
            original_exception_type=exc_type_name,
            slice_name=slice_name,
            cycle=cycle,
            mode=mode,
            seed=seed,
        )

    if isinstance(exc, FileNotFoundError):
        # Extract just the filename/path from the error
        msg = str(exc)
        return ErrorContext(
            kind=RuntimeErrorKind.FILE_NOT_FOUND,
            message=f"File not found: {msg}",
            traceback_hash=tb_hash,
            recoverable=False,
            original_exception_type=exc_type_name,
            slice_name=slice_name,
            cycle=cycle,
            mode=mode,
            seed=seed,
        )

    if isinstance(exc, (ValueError, TypeError)):
        return ErrorContext(
            kind=RuntimeErrorKind.VALIDATION,
            message=f"Validation: {exc}",
            traceback_hash=tb_hash,
            recoverable=False,
            original_exception_type=exc_type_name,
            slice_name=slice_name,
            cycle=cycle,
            mode=mode,
            seed=seed,
        )

    # Unknown/other exceptions
    return ErrorContext(
        kind=RuntimeErrorKind.UNKNOWN,
        message=str(exc) or f"Unknown error: {exc_type_name}",
        traceback_hash=tb_hash,
        recoverable=False,
        original_exception_type=exc_type_name,
        slice_name=slice_name,
        cycle=cycle,
        mode=mode,
        seed=seed,
    )


def build_error_result(
    exc: Exception,
    context: Optional[Dict[str, Any]] = None,
    *,
    slice_name: Optional[str] = None,
    cycle: Optional[int] = None,
    mode: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Build a structured error result dictionary.

    This function creates an error result compatible with the telemetry
    format used in experiments/run_uplift_u2.py.

    Args:
        exc: The exception that occurred.
        context: Optional additional context to include.
        slice_name: Name of the slice where error occurred.
        cycle: Cycle index where error occurred.
        mode: Execution mode.
        seed: Seed value.

    Returns:
        Dictionary suitable for inclusion in telemetry records.

    Example:
        >>> try:
        ...     raise ValueError("invalid input")
        ... except Exception as e:
        ...     result = build_error_result(
        ...         e, slice_name="test", cycle=5
        ...     )
        ...     result["error_kind"]
        'validation'
    """
    error_ctx = classify_error_with_context(
        exc,
        slice_name=slice_name,
        cycle=cycle,
        mode=mode,
        seed=seed,
    )

    result: Dict[str, Any] = {
        "error": error_ctx.message,
        "error_kind": error_ctx.kind.value,
    }

    if error_ctx.traceback_hash:
        result["traceback_hash"] = error_ctx.traceback_hash

    if error_ctx.subprocess_stdout is not None:
        result["stdout"] = error_ctx.subprocess_stdout
    if error_ctx.subprocess_stderr is not None:
        result["stderr"] = error_ctx.subprocess_stderr

    # Include experiment context
    if slice_name is not None:
        result["slice_name"] = slice_name
    if cycle is not None:
        result["cycle"] = cycle
    if mode is not None:
        result["mode"] = mode
    if seed is not None:
        result["seed"] = seed

    if context:
        result["context"] = context

    return result


__all__ = [
    "RuntimeErrorKind",
    "ErrorContext",
    "classify_error",
    "classify_error_with_context",
    "build_error_result",
]
