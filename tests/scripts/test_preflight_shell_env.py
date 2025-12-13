"""
Tests for preflight_shell_env BOM detection.

Tests the detect_bom() function with string-based byte patterns.
Also tests advisory-only guarantees (no exit code changes, no file writes).
"""

import io
import sys
import pytest

from scripts.preflight_shell_env import (
    detect_bom,
    UTF16_LE_BOM,
    UTF8_BOM,
    print_preflight_advisory,
    check_bashrc_bom,
)


class TestDetectBom:
    """Tests for detect_bom() function."""

    def test_detects_utf16_le_bom(self):
        """UTF-16 LE BOM (0xFF 0xFE) is detected."""
        file_bytes = b"\xff\xfeexport VAR=value"
        result = detect_bom(file_bytes)
        assert result == "UTF-16-LE"

    def test_detects_utf8_bom(self):
        """UTF-8 BOM (0xEF 0xBB 0xBF) is detected."""
        file_bytes = b"\xef\xbb\xbfexport VAR=value"
        result = detect_bom(file_bytes)
        assert result == "UTF-8"

    def test_no_bom_returns_none(self):
        """Clean file without BOM returns None."""
        file_bytes = b"export VAR=value"
        result = detect_bom(file_bytes)
        assert result is None

    def test_empty_bytes_returns_none(self):
        """Empty bytes returns None."""
        result = detect_bom(b"")
        assert result is None

    def test_single_byte_returns_none(self):
        """Single byte (insufficient for BOM) returns None."""
        result = detect_bom(b"\xff")
        assert result is None

    def test_utf16_le_bom_constant(self):
        """UTF-16 LE BOM constant is correct."""
        assert UTF16_LE_BOM == b"\xff\xfe"

    def test_utf8_bom_constant(self):
        """UTF-8 BOM constant is correct."""
        assert UTF8_BOM == b"\xef\xbb\xbf"

    def test_partial_utf8_bom_not_detected(self):
        """Partial UTF-8 BOM (2 bytes) is not detected as UTF-8."""
        # Only 2 of 3 UTF-8 BOM bytes
        file_bytes = b"\xef\xbbexport"
        result = detect_bom(file_bytes)
        assert result is None

    def test_utf16_le_bom_exact_two_bytes(self):
        """Exactly 2 bytes matching UTF-16 LE BOM is detected."""
        result = detect_bom(b"\xff\xfe")
        assert result == "UTF-16-LE"


# =============================================================================
# Advisory-Only Guarantee Tests
# =============================================================================


class TestAdvisoryOnlyGuarantees:
    """
    Tests ensuring preflight is advisory-only:
    - Emits at most one line
    - Never changes exit codes
    - Does not write files
    """

    def test_print_preflight_advisory_emits_at_most_one_line(self, capsys, monkeypatch):
        """print_preflight_advisory() emits at most one line of output."""
        # Mock check_bashrc_bom to return a BOM detection
        monkeypatch.setattr(
            "scripts.preflight_shell_env.check_bashrc_bom",
            lambda: (True, "UTF-16-LE", "Test advisory message"),
        )

        print_preflight_advisory()
        captured = capsys.readouterr()

        # Count newlines in output
        lines = captured.out.strip().split("\n") if captured.out.strip() else []
        assert len(lines) <= 1, f"Expected at most 1 line, got {len(lines)}: {lines}"

    def test_print_preflight_advisory_emits_zero_lines_when_no_bom(self, capsys, monkeypatch):
        """print_preflight_advisory() emits nothing when no BOM detected."""
        # Mock check_bashrc_bom to return no BOM
        monkeypatch.setattr(
            "scripts.preflight_shell_env.check_bashrc_bom",
            lambda: (False, None, None),
        )

        print_preflight_advisory()
        captured = capsys.readouterr()

        assert captured.out == "", f"Expected no output, got: {captured.out!r}"

    def test_check_bashrc_bom_does_not_write_files(self, tmp_path, monkeypatch):
        """check_bashrc_bom() does not create or modify any files."""
        # Create a fake home with a .bashrc that has BOM
        fake_home = tmp_path / "fakehome"
        fake_home.mkdir()
        bashrc = fake_home / ".bashrc"
        bashrc.write_bytes(b"\xff\xfeexport TEST=1")

        # Record initial state
        initial_files = set(tmp_path.rglob("*"))
        initial_mtime = bashrc.stat().st_mtime

        # Mock environment to use fake home
        monkeypatch.setenv("USERPROFILE", str(fake_home))
        monkeypatch.setenv("HOME", str(fake_home))

        # Run the check
        has_bom, bom_type, advisory = check_bashrc_bom()

        # Verify BOM was detected (sanity check)
        assert has_bom is True
        assert bom_type == "UTF-16-LE"

        # Verify no files created
        final_files = set(tmp_path.rglob("*"))
        assert final_files == initial_files, f"New files created: {final_files - initial_files}"

        # Verify .bashrc not modified
        assert bashrc.stat().st_mtime == initial_mtime, ".bashrc was modified"

    def test_print_preflight_advisory_returns_none(self, monkeypatch):
        """print_preflight_advisory() returns None (no exit code influence)."""
        monkeypatch.setattr(
            "scripts.preflight_shell_env.check_bashrc_bom",
            lambda: (True, "UTF-16-LE", "Test message"),
        )

        result = print_preflight_advisory()
        assert result is None, f"Expected None return, got {result!r}"

    def test_preflight_module_has_no_file_write_operations(self):
        """Verify preflight module source has no file write operations."""
        import inspect
        import scripts.preflight_shell_env as module

        source = inspect.getsource(module)

        # Check for file write patterns
        write_patterns = [
            ".write(",
            ".write_bytes(",
            ".write_text(",
            "open(",  # We do use open() but only for reading
        ]

        # Find all open() calls and verify they're read-only
        import re
        open_calls = re.findall(r'open\([^)]+\)', source)
        for call in open_calls:
            # Verify mode is "rb" (read binary) or no mode (defaults to read)
            assert '"rb"' in call or "'rb'" in call or "mode" not in call.lower(), \
                f"Found non-read open() call: {call}"

        # Verify no direct write method calls on file handles
        assert ".write(" not in source, "Found .write() call in source"
        assert ".write_bytes(" not in source, "Found .write_bytes() call in source"
        assert ".write_text(" not in source, "Found .write_text() call in source"
