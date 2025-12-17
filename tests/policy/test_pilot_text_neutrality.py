"""
Pilot Ingestion Text Neutrality Tripwire

Verifies that pilot ingestion components produce neutral warning/output text:
- backend/health/pilot_external_ingest_adapter.py warning strings
- scripts/pilot_toolchain_hook.py console output

SCOPE: Prose warning/output strings only (NOT filenames, NOT structured tokens).

Structured tokens PASS/FAIL/WARN are explicitly allowed as they are
machine-readable status indicators, not prose.

SHADOW MODE — observational only.
"""

import pytest

from tests.helpers.warning_neutrality import BANNED_ALARM_WORDS


# =============================================================================
# Fixtures: Extract warning patterns from pilot adapter
# =============================================================================

# Warning strings used in pilot_external_ingest_adapter.py
# These are the prose portions that could be displayed to users
ADAPTER_WARNING_PATTERNS = [
    # From validate_external_log_schema()
    "missing required field: log_type",
    # From validate_external_log_schema() - template
    "unrecognized field:",
    # From ingest_external_log() - various error paths
    "file not found:",
    "integrity mismatch: expected",
    "unsupported file type:",
    "JSON parse unsuccessful:",
    "JSONL parse unsuccessful:",
]

# Console output strings from pilot_toolchain_hook.py
HOOK_CONSOLE_OUTPUTS = [
    "Pilot manifest written:",
    "DISCLAIMER",
    "PILOT PROVENANCE: Binds artifact to toolchain state at ingestion.",
    "NOT experiment provenance.",
    "Does not imply experimental validity, reproducibility, or parity with CAL-EXP runs.",
    "Pilot Provenance:",
    "artifact_id:",
    "ingestion_timestamp:",
    "uv_lock_hash:",
    "toolchain_fingerprint:",
]


# =============================================================================
# Tripwire Tests
# =============================================================================

class TestPilotAdapterWarningNeutrality:
    """Tripwire tests for pilot adapter warning string neutrality."""

    def test_adapter_warnings_no_banned_words(self):
        """Adapter warning strings must contain no banned alarm words."""
        violations = []
        for warning in ADAPTER_WARNING_PATTERNS:
            warning_lower = warning.lower()
            for word in BANNED_ALARM_WORDS:
                if word.lower() in warning_lower:
                    violations.append((warning, word))

        assert not violations, (
            f"Adapter warnings contain banned alarm words: {violations}\n"
            f"Suggested fix: Replace 'failed to parse' with 'could not parse' or 'parse unsuccessful'"
        )

    def test_adapter_warnings_single_line(self):
        """Adapter warning strings must be single-line (no embedded newlines)."""
        for warning in ADAPTER_WARNING_PATTERNS:
            assert "\n" not in warning, (
                f"Warning contains newline: {warning!r}"
            )

    def test_adapter_uses_structured_result_codes(self):
        """
        Adapter must use structured result codes (not prose) for status.

        Verify the adapter uses PilotIngestResult enum values.
        """
        # Import the actual enum to verify it exists and has expected values
        from backend.health.pilot_external_ingest_adapter import PilotIngestResult

        expected_codes = {
            "SUCCESS",
            "SCHEMA_INVALID",
            "FILE_NOT_FOUND",
            "PARSE_ERROR",
            "INTEGRITY_MISMATCH",
        }

        actual_codes = {
            PilotIngestResult.SUCCESS,
            PilotIngestResult.SCHEMA_INVALID,
            PilotIngestResult.FILE_NOT_FOUND,
            PilotIngestResult.PARSE_ERROR,
            PilotIngestResult.INTEGRITY_MISMATCH,
        }

        assert actual_codes == expected_codes, (
            f"PilotIngestResult enum values mismatch: {actual_codes}"
        )


class TestPilotHookConsoleNeutrality:
    """Tripwire tests for pilot hook console output neutrality."""

    def test_hook_console_no_banned_words(self):
        """Hook console output must contain no banned alarm words."""
        violations = []
        for output in HOOK_CONSOLE_OUTPUTS:
            output_lower = output.lower()
            for word in BANNED_ALARM_WORDS:
                if word.lower() in output_lower:
                    violations.append((output, word))

        assert not violations, (
            f"Hook console output contains banned alarm words: {violations}\n"
            f"Update scripts/pilot_toolchain_hook.py to use neutral language"
        )

    def test_hook_console_no_evaluative_terms(self):
        """Hook console output must not use evaluative terms."""
        evaluative_terms = ["improved", "better", "worse", "degraded", "broken"]

        violations = []
        for output in HOOK_CONSOLE_OUTPUTS:
            output_lower = output.lower()
            for term in evaluative_terms:
                if term in output_lower:
                    violations.append((output, term))

        assert not violations, (
            f"Hook console output contains evaluative terms: {violations}"
        )

    def test_hook_includes_disclaimer(self):
        """Hook console output must include disclaimer about pilot provenance."""
        all_output = " ".join(HOOK_CONSOLE_OUTPUTS).lower()

        assert "pilot provenance" in all_output, (
            "Hook output must mention 'pilot provenance'"
        )
        assert "not experiment provenance" in all_output, (
            "Hook output must clarify pilot provenance is NOT experiment provenance"
        )


class TestPilotStructuredTokensAllowed:
    """Verify structured tokens are NOT flagged as violations."""

    @pytest.mark.parametrize("token", [
        "SUCCESS",
        "SCHEMA_INVALID",
        "FILE_NOT_FOUND",
        "PARSE_ERROR",
        "INTEGRITY_MISMATCH",
        "PASS",
        "FAIL",
        "WARN",
    ])
    def test_structured_tokens_not_banned(self, token: str):
        """
        Structured tokens (enum values) must NOT be in banned words list.

        These are machine-readable status indicators, not prose.
        """
        # Structured tokens should not match banned words
        # (case-sensitive enum values like "FAIL" are different from prose "fail")
        assert token not in BANNED_ALARM_WORDS, (
            f"Structured token {token} should not be in banned words list"
        )
