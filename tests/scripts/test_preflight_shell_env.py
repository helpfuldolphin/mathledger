"""
Tests for preflight_shell_env BOM detection.

Tests the detect_bom() function with string-based byte patterns.
"""

import pytest

from scripts.preflight_shell_env import detect_bom, UTF16_LE_BOM, UTF8_BOM


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
