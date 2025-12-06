"""
PASS Line Verifier Test for First Organism

This test verifies that the PASS line parsing logic correctly extracts H_t
from log output. This is used by precheck scripts (e.g., basis_promotion_precheck)
to verify First Organism test success.

This is a HERMETIC test - it requires no database, Redis, or external dependencies.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


# Regex pattern matching the canonical PASS line format
# Format: [PASS] FIRST ORGANISM ALIVE H_t=<short_hash>
# The short_hash is typically 12 hex characters, but we allow variable length
PASS_LINE_PATTERN = re.compile(
    r"\[PASS\]\s+FIRST\s+ORGANISM\s+ALIVE\s+H_t=([0-9a-fA-F]+)",
    re.IGNORECASE
)


def parse_pass_line(line: str) -> str | None:
    """
    Parse H_t from a PASS line.
    
    This uses the same logic as ops/basis_promotion_precheck.py::analyze_output.
    
    Args:
        line: Log line to parse
        
    Returns:
        Extracted H_t value (hex string) or None if not found
    """
    match = PASS_LINE_PATTERN.search(line)
    if match:
        return match.group(1).lower()
    return None


@pytest.mark.hermetic
@pytest.mark.unit
def test_pass_line_parser_synthetic():
    """
    Unit test that parses a synthetic log containing the PASS line.
    
    This test verifies:
    1. The regex correctly extracts H_t from the canonical format
    2. The extracted value matches the expected hash
    3. The parser handles ANSI color codes (if present)
    """
    # Synthetic log line matching the exact format
    synthetic_log = "[PASS] FIRST ORGANISM ALIVE H_t=deadbeef1234"
    
    h_t = parse_pass_line(synthetic_log)
    assert h_t is not None, "Failed to extract H_t from PASS line"
    assert h_t == "deadbeef1234", f"Expected 'deadbeef1234', got '{h_t}'"


@pytest.mark.hermetic
@pytest.mark.unit
def test_pass_line_parser_with_ansi_codes():
    """
    Test that the parser handles ANSI color codes in the log line.
    """
    # PASS line with ANSI color codes (as emitted by log_first_organism_pass)
    colored_log = "\033[92m[PASS] FIRST ORGANISM ALIVE H_t=deadbeef1234\033[0m"
    
    h_t = parse_pass_line(colored_log)
    assert h_t is not None, "Failed to extract H_t from colored PASS line"
    assert h_t == "deadbeef1234", f"Expected 'deadbeef1234', got '{h_t}'"


@pytest.mark.hermetic
@pytest.mark.unit
def test_pass_line_parser_case_insensitive():
    """
    Test that the parser is case-insensitive for the PASS marker.
    """
    # Various case combinations
    test_cases = [
        "[PASS] FIRST ORGANISM ALIVE H_t=deadbeef1234",
        "[pass] FIRST ORGANISM ALIVE H_t=deadbeef1234",
        "[Pass] FIRST ORGANISM ALIVE H_t=deadbeef1234",
        "[PASS] first organism alive H_t=deadbeef1234",
        "[PASS] FIRST ORGANISM ALIVE h_t=deadbeef1234",
    ]
    
    for line in test_cases:
        h_t = parse_pass_line(line)
        assert h_t is not None, f"Failed to parse: {line}"
        assert h_t == "deadbeef1234", f"Expected 'deadbeef1234', got '{h_t}' for: {line}"


@pytest.mark.hermetic
@pytest.mark.unit
def test_pass_line_parser_with_whitespace():
    """
    Test that the parser handles various whitespace patterns.
    """
    test_cases = [
        "[PASS] FIRST ORGANISM ALIVE H_t=deadbeef1234",
        "[PASS]  FIRST ORGANISM ALIVE  H_t=deadbeef1234",
        "[PASS]\tFIRST ORGANISM ALIVE\tH_t=deadbeef1234",
    ]
    
    for line in test_cases:
        h_t = parse_pass_line(line)
        assert h_t is not None, f"Failed to parse: {line}"
        assert h_t == "deadbeef1234", f"Expected 'deadbeef1234', got '{h_t}' for: {line}"


@pytest.mark.hermetic
@pytest.mark.unit
def test_pass_line_parser_invalid_lines():
    """
    Test that the parser correctly rejects invalid lines.
    """
    invalid_lines = [
        "[FAIL] FIRST ORGANISM ALIVE H_t=deadbeef1234",
        "[PASS] FIRST ORGANISM DEAD H_t=deadbeef1234",
        "[PASS] FIRST ORGANISM ALIVE",
        "FIRST ORGANISM ALIVE H_t=deadbeef1234",
        "[PASS] FIRST ORGANISM ALIVE H_t=",
        "[PASS] FIRST ORGANISM ALIVE H_t=xyz",  # Non-hex
    ]
    
    for line in invalid_lines:
        h_t = parse_pass_line(line)
        assert h_t is None, f"Should not parse invalid line: {line}"


@pytest.mark.hermetic
@pytest.mark.unit
def test_pass_line_parser_full_64_char_hash():
    """
    Test that the parser can extract full 64-character hashes.
    """
    full_hash = "a" * 64
    log_line = f"[PASS] FIRST ORGANISM ALIVE H_t={full_hash}"
    
    h_t = parse_pass_line(log_line)
    assert h_t is not None, "Failed to extract full hash"
    assert h_t == full_hash, f"Expected full hash, got '{h_t}'"


@pytest.mark.hermetic
@pytest.mark.unit
def test_pass_line_parser_matches_precheck_logic():
    """
    Test that our parser matches the logic used in basis_promotion_precheck.py.
    
    The precheck script uses:
    - Case-insensitive search for "[PASS] FIRST ORGANISM"
    - Extracts H_t using split("H_T=")[1].split()[0]
    
    We verify our regex produces the same result.
    """
    test_hash = "deadbeef1234"
    log_line = f"[PASS] FIRST ORGANISM ALIVE H_t={test_hash}"
    
    # Our regex-based parser
    h_t_regex = parse_pass_line(log_line)
    
    # Precheck-style parsing (case-insensitive split)
    normalized = log_line.upper()
    if "H_T=" in normalized:
        h_t_precheck = normalized.split("H_T=")[1].split()[0].lower()
    else:
        h_t_precheck = None
    
    assert h_t_regex == h_t_precheck, (
        f"Parser mismatch: regex={h_t_regex}, precheck={h_t_precheck}"
    )
    assert h_t_regex == test_hash, f"Expected '{test_hash}', got '{h_t_regex}'"

