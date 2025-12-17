"""
Tripwire tests for warning_neutrality helper.

These tests ensure the helper:
1. Only scans the warning string argument, not the context parameter
2. Correctly flags banned words with actionable error messages

SHADOW MODE: Observational verification only.
"""

import pytest

from tests.helpers.warning_neutrality import (
    assert_warning_neutral,
    assert_no_banned_words,
    pytest_assert_warning_neutral,
    BANNED_ALARM_WORDS,
)


class TestTripwireScopeGuard:
    """Tripwire: Ensure only warning string is scanned, not context."""

    def test_context_with_banned_word_does_not_trigger_failure(self):
        """Context parameter containing banned words should NOT cause failure.

        The helper should only scan the warning string, not the context.
        This proves structured fields (filenames, test names) in context
        are not falsely flagged.
        """
        # Clean warning text
        clean_warning = "Pattern recorded (informational)"

        # Context contains banned words - should NOT be scanned
        context_with_banned_words = "test_failure_handler.py::test_error_case"

        # This should pass - only warning string is checked
        result = assert_warning_neutral(clean_warning)
        assert result.passed, "Clean warning should pass"

        # Verify context is truly not scanned by calling pytest helper
        # If context were scanned, this would raise AssertionError
        pytest_assert_warning_neutral(
            clean_warning,
            context=context_with_banned_words,
        )

    def test_filename_in_context_not_scanned(self):
        """Filenames with words like 'fail' in context should not trigger."""
        clean_warning = "Structural drill: streak=3 [drill-001] (informational)"

        # Filename context that would fail if scanned
        filename_context = "test_fail_handler_error_detection.py"

        # Should pass - only warning is scanned
        pytest_assert_warning_neutral(clean_warning, context=filename_context)

    def test_structured_field_names_in_context_not_scanned(self):
        """Structured field names like 'error_count' in context should not trigger."""
        clean_warning = "CTRPK: value=2.5, status=WARN (informational)"

        # Structured field context that would fail if scanned
        field_context = "error_count validation in failure_metrics"

        # Should pass - only warning is scanned
        pytest_assert_warning_neutral(clean_warning, context=field_context)


class TestTripwireBannedWordDetection:
    """Tripwire: Ensure banned words ARE correctly flagged."""

    def test_flags_single_banned_word(self):
        """A single banned word in warning string MUST be flagged."""
        # Warning containing "detected" - a banned word
        bad_warning = "Anomaly detected in cycle 42"

        result = assert_warning_neutral(bad_warning)

        assert not result.passed, "Warning with banned word should fail"
        assert result.violations is not None
        assert len(result.violations) > 0

    def test_flags_evaluative_term(self):
        """Evaluative terms like 'bad' MUST be flagged."""
        bad_warning = "Bad configuration observed"

        result = assert_warning_neutral(bad_warning)

        assert not result.passed, "Warning with 'bad' should fail"
        assert "bad" in [v.lower() for v in result.violations]

    def test_error_message_is_actionable(self):
        """AssertionError message must include: violation, context, and warning text."""
        bad_warning = "Error detected in module"
        context = "CTRPK warning test"

        with pytest.raises(AssertionError) as exc_info:
            pytest_assert_warning_neutral(bad_warning, context=context)

        error_message = str(exc_info.value)

        # Must include the context
        assert "CTRPK warning test" in error_message, "Error should include context"

        # Must include at least one violation
        assert "Violations:" in error_message, "Error should list violations"

        # Must include the original warning for debugging
        assert "Error detected in module" in error_message, "Error should include warning text"

        # Must indicate what failed
        assert "neutrality check failed" in error_message, "Error should indicate failure type"

    def test_case_insensitive_detection(self):
        """Banned word detection must be case-insensitive."""
        # "FAILED" uppercase should still be caught
        bad_warning = "Operation FAILED during test"

        result = assert_warning_neutral(bad_warning)

        assert not result.passed, "Uppercase banned word should be caught"

    def test_banned_words_list_is_populated(self):
        """BANNED_ALARM_WORDS must contain expected terms."""
        # Verify key banned words are present
        expected_banned = ["detected", "alert", "error", "failed", "bad", "wrong"]

        for word in expected_banned:
            assert word in BANNED_ALARM_WORDS, f"'{word}' should be in BANNED_ALARM_WORDS"

    def test_clean_warning_passes(self):
        """A properly neutral warning should pass."""
        clean_warning = "Structural drill: STRUCTURAL_BREAK pattern recorded [drill-001] (severity=WARN, informational)"

        result = assert_warning_neutral(clean_warning)

        assert result.passed, f"Clean warning should pass, got: {result.message}"


class TestTripwireNewlineDetection:
    """Tripwire: Ensure multi-line warnings are flagged."""

    def test_multiline_warning_fails(self):
        """Warning with newline MUST be flagged."""
        multiline_warning = "Line 1\nLine 2"

        result = assert_warning_neutral(multiline_warning)

        assert not result.passed, "Multi-line warning should fail"
        assert "newline" in result.message.lower()

    def test_single_line_warning_passes(self):
        """Single-line warning should pass newline check."""
        single_line = "Pattern recorded (informational)"

        result = assert_warning_neutral(single_line)

        assert result.passed, "Single-line clean warning should pass"
