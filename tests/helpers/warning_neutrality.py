"""
Warning Neutrality Assertion Helpers for SHADOW MODE Artifacts.
==============================================================================

SINGLE SOURCE OF TRUTH for warning text neutrality validation across all
SHADOW MODE test suites (CAL-EXP-2, CAL-EXP-3, structural drill, CTRPK, etc.).

SCOPE:
- Single-line validation (no newlines in warnings)
- Banned alarm word detection
- Warning format stability checks

BANNED WORD LIST STATUS: FINAL (Phase X / CAL-EXP-2)
- Frozen as of Phase X completion
- See ALARM_SPECIFIC_BANNED_WORDS below for canonical list
- Combined with tools/lint_public_language.py BANNED_WORDS

DOCTRINE — NO RANKING LANGUAGE / NO ALARMISM:
- Advisory text must be observational, not evaluative
- Avoid words that imply judgment: "bad", "wrong", "error", "fail"
- Avoid words that imply alerting: "alert", "danger", "warning", "threat"
- Avoid words that imply detection: "detected", "anomaly", "violation"
- Use neutral phrasing: "recorded", "observed", "logged", "informational"

==============================================================================
PROCESS TO ADD NEW BANNED WORDS (post-freeze):
==============================================================================
1. Open a PR with justification in the commit message
2. Add the word to ALARM_SPECIFIC_BANNED_WORDS with a rationale comment
3. Run: uv run pytest tests/helpers/test_warning_neutrality.py -v
4. Verify no false positives in existing warning text
5. If the word is general (not alarm-specific), add to tools/lint_public_language.py
6. Update any production code that uses the newly-banned word

DO NOT add words without documented rationale.
DO NOT add words that appear in valid structured fields (e.g., "critical" as severity).
==============================================================================

USAGE:
    from tests.helpers.warning_neutrality import (
        pytest_assert_warning_neutral,
        pytest_assert_warnings_neutral,
        BANNED_ALARM_WORDS,
    )

    # Single warning
    pytest_assert_warning_neutral(warning_text, context="CTRPK warning")

    # List of warnings
    pytest_assert_warnings_neutral(warnings_list, context="status warnings")

SHADOW MODE: All functions are observational verification only.

==============================================================================
APPENDIX: WHY NEUTRAL LANGUAGE MATTERS
==============================================================================
SHADOW MODE artifacts are observational diagnostics, not enforcement mechanisms.
Warning text that uses alarm language ("error detected", "failure alert") creates
a false impression that the system is actively preventing or blocking behavior.

Neutral language principles:
1. OBSERVATIONAL: "Pattern recorded" not "Error detected"
2. INFORMATIONAL: "Condition logged" not "Violation observed"
3. NON-JUDGMENTAL: "Value outside band" not "Bad value"
4. NON-ALARMIST: "Informational" not "Warning" or "Alert"

This ensures:
- Users understand SHADOW MODE is advisory, not gating
- No false sense of system enforcement
- Clear separation between observation and action
- Consistent tone across all diagnostic output
==============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set

# Import canonical banned words from linter
from tools.lint_public_language import BANNED_WORDS as LINTER_BANNED_WORDS


@dataclass
class WarningNeutralityResult:
    """Result of a warning neutrality check."""

    passed: bool
    message: str
    violations: Optional[List[str]] = None


# =============================================================================
# ALARM-SPECIFIC BANNED WORDS — FINAL (Phase X / CAL-EXP-2)
# =============================================================================
# Extension to LINTER_BANNED_WORDS for warning text neutrality.
# STATUS: FROZEN. See module docstring for process to add new words.
#
# Rationale for alarm-specific extensions:
# - "detected/alert/danger/warning" imply active monitoring or alerting
# - "error/failed/failure" imply system malfunction vs. observed condition
# - "violation/anomaly/tension" imply deviation from expected behavior
# - "threat/risk" imply adversarial or safety-critical framing
# =============================================================================

ALARM_SPECIFIC_BANNED_WORDS: List[str] = [
    # Alarm/detection words (not in base linter, specific to warning text)
    "detected",      # Implies active detection vs. observation
    "alert",         # Implies alerting system
    "danger",        # Alarm framing
    "error",         # Implies system error vs. observed pattern
    "failed",        # Implies failure vs. condition observed
    "failure",       # Implies failure vs. condition observed
    "fail",          # Short form of failure
    "violation",     # Implies rule violation vs. pattern logged
    "violations",    # Plural form
    "anomaly",       # Implies deviation from norm
    "anomalies",     # Plural form
    "tension",       # Implies stress/conflict
    "threat",        # Adversarial framing
    "risk",          # Safety/adversarial framing
    "warning",       # Meta: the word "warning" should not appear in text
    # Evaluative/judgment words (from forbidden_terms consolidation)
    "bad",           # Evaluative judgment
    "wrong",         # Evaluative judgment
    "mistake",       # Evaluative judgment
    "fix",           # Implies something is broken
    "broken",        # Evaluative judgment
    "good",          # Positive evaluation (also avoid ranking)
    "urgent",        # Alarm word
    # NOTE: "critical" is NOT banned because it's a valid severity level enum
    # (e.g., "severity=CRITICAL" is acceptable in warning text)
]

# Combined canonical list: linter base + alarm-specific extensions
BANNED_ALARM_WORDS: List[str] = list(set(LINTER_BANNED_WORDS + ALARM_SPECIFIC_BANNED_WORDS))


def assert_single_line(warning: str) -> WarningNeutralityResult:
    """
    Assert that a warning string is single-line (no newlines).

    SHADOW MODE: Observational verification only.

    Args:
        warning: Warning string to check

    Returns:
        WarningNeutralityResult with pass/fail and message
    """
    if "\n" in warning:
        return WarningNeutralityResult(
            passed=False,
            message="Warning contains newline(s)",
            violations=[f"Found {warning.count(chr(10))} newline(s)"],
        )

    return WarningNeutralityResult(
        passed=True,
        message="Warning is single-line",
    )


def assert_no_banned_words(
    warning: str,
    banned_words: Optional[List[str]] = None,
) -> WarningNeutralityResult:
    """
    Assert that a warning string contains no banned alarm words.

    SHADOW MODE: Observational verification only.

    Args:
        warning: Warning string to check
        banned_words: Optional custom banned word list (defaults to BANNED_ALARM_WORDS)

    Returns:
        WarningNeutralityResult with pass/fail, message, and list of violations
    """
    if banned_words is None:
        banned_words = BANNED_ALARM_WORDS

    warning_lower = warning.lower()
    found_banned: List[str] = []

    for word in banned_words:
        if word.lower() in warning_lower:
            found_banned.append(word)

    if found_banned:
        return WarningNeutralityResult(
            passed=False,
            message=f"Warning contains {len(found_banned)} banned word(s)",
            violations=found_banned,
        )

    return WarningNeutralityResult(
        passed=True,
        message="Warning contains no banned words",
    )


def assert_warning_neutral(
    warning: str,
    banned_words: Optional[List[str]] = None,
) -> WarningNeutralityResult:
    """
    Assert that a warning string is neutral: single-line and no banned words.

    SHADOW MODE: Observational verification only.

    Args:
        warning: Warning string to check
        banned_words: Optional custom banned word list (defaults to BANNED_ALARM_WORDS)

    Returns:
        WarningNeutralityResult with pass/fail and combined violations
    """
    # Check single-line
    single_line_result = assert_single_line(warning)
    if not single_line_result.passed:
        return single_line_result

    # Check banned words
    banned_result = assert_no_banned_words(warning, banned_words)
    if not banned_result.passed:
        return banned_result

    return WarningNeutralityResult(
        passed=True,
        message="Warning is neutral (single-line, no banned words)",
    )


def validate_warning_list(
    warnings: List[str],
    banned_words: Optional[List[str]] = None,
) -> WarningNeutralityResult:
    """
    Validate a list of warning strings for neutrality.

    SHADOW MODE: Observational verification only.

    Args:
        warnings: List of warning strings to check
        banned_words: Optional custom banned word list (defaults to BANNED_ALARM_WORDS)

    Returns:
        WarningNeutralityResult with pass/fail and list of all violations
    """
    all_violations: List[str] = []

    for idx, warning in enumerate(warnings):
        result = assert_warning_neutral(warning, banned_words)
        if not result.passed:
            violation_str = f"[{idx}] {result.message}"
            if result.violations:
                violation_str += f": {result.violations}"
            all_violations.append(violation_str)

    if all_violations:
        return WarningNeutralityResult(
            passed=False,
            message=f"{len(all_violations)} warning(s) failed neutrality check",
            violations=all_violations,
        )

    return WarningNeutralityResult(
        passed=True,
        message=f"All {len(warnings)} warning(s) are neutral",
    )


# =============================================================================
# Pytest Assertion Helpers
# =============================================================================

def pytest_assert_warning_neutral(
    warning: str,
    banned_words: Optional[List[str]] = None,
    context: str = "",
) -> None:
    """
    Pytest assertion that a warning string is neutral.

    Raises AssertionError with detailed info on failure.

    Args:
        warning: Warning string to check
        banned_words: Optional custom banned word list
        context: Optional context string for error message
    """
    result = assert_warning_neutral(warning, banned_words)
    if not result.passed:
        ctx = f" ({context})" if context else ""
        raise AssertionError(
            f"Warning neutrality check failed{ctx}:\n"
            f"  {result.message}\n"
            f"  Violations: {result.violations}\n"
            f"  Warning: {warning!r}"
        )


def pytest_assert_warnings_neutral(
    warnings: List[str],
    banned_words: Optional[List[str]] = None,
    context: str = "",
) -> None:
    """
    Pytest assertion that all warning strings are neutral.

    Raises AssertionError with detailed info on failure.

    Args:
        warnings: List of warning strings to check
        banned_words: Optional custom banned word list
        context: Optional context string for error message
    """
    result = validate_warning_list(warnings, banned_words)
    if not result.passed:
        ctx = f" ({context})" if context else ""
        violations_str = "\n".join(f"  - {v}" for v in (result.violations or []))
        raise AssertionError(
            f"Warning list neutrality check failed{ctx}:\n"
            f"  {result.message}\n"
            f"Violations:\n{violations_str}"
        )
