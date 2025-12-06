"""
First Organism Contract Regression Tests
==========================================

Lightweight "tripwire" tests that guarantee no one accidentally breaks SPARK's
core assumptions while refactoring.

These tests are static contract checks - they only read source code and strings.
They do NOT require DB/Redis and should run fast in any environment.

Purpose:
    - Ensure test_first_organism_closed_loop_happy_path exists
    - Ensure [PASS] FIRST ORGANISM ALIVE string is present (so no one "cleans it up")
    - Ensure attestation file path hasn't been silently changed

These tests act as regression sentinels that fail loudly if someone refactors
away the SPARK test or the FO attestation behavior.
"""

import ast
from pathlib import Path

import pytest


# Path to the First Organism integration test file
FO_TEST_FILE = Path(__file__).parent.parent / "integration" / "test_first_organism.py"


def test_first_organism_test_exists():
    """
    Regression test: Assert that test_first_organism_closed_loop_happy_path
    exists in tests/integration/test_first_organism.py.
    
    This is a static contract check - it reads the source file and verifies
    the function definition exists.
    """
    assert FO_TEST_FILE.exists(), (
        f"First Organism test file not found: {FO_TEST_FILE}\n"
        "This is a critical regression - the SPARK test must exist."
    )
    
    source_code = FO_TEST_FILE.read_text(encoding="utf-8")
    
    # Parse the AST to find function definitions
    tree = ast.parse(source_code, filename=str(FO_TEST_FILE))
    
    function_names = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    ]
    
    assert "test_first_organism_closed_loop_happy_path" in function_names, (
        f"Critical test function 'test_first_organism_closed_loop_happy_path' not found in {FO_TEST_FILE}\n"
        f"Available functions: {sorted(function_names)}\n"
        "This is a regression - the SPARK closed loop test must exist."
    )


def test_first_organism_pass_string_present():
    """
    Regression test: Assert that '[PASS] FIRST ORGANISM ALIVE' string appears
    somewhere in the source file (so no one "cleans it up").
    
    This string is emitted by log_first_organism_pass() and is the canonical
    certification line that Cursor P relies on.
    """
    assert FO_TEST_FILE.exists(), (
        f"First Organism test file not found: {FO_TEST_FILE}"
    )
    
    source_code = FO_TEST_FILE.read_text(encoding="utf-8")
    
    # Check for the exact string pattern (case-sensitive)
    pass_string = "[PASS] FIRST ORGANISM ALIVE"
    
    assert pass_string in source_code, (
        f"Critical string '{pass_string}' not found in {FO_TEST_FILE}\n"
        "This string must be present - it's the canonical certification line.\n"
        "It may be in a docstring, comment, or emitted by log_first_organism_pass()."
    )
    
    # Also check the conftest where log_first_organism_pass is defined
    conftest_file = Path(__file__).parent.parent / "integration" / "conftest.py"
    if conftest_file.exists():
        conftest_code = conftest_file.read_text(encoding="utf-8")
        assert pass_string in conftest_code, (
            f"Critical string '{pass_string}' not found in {conftest_file}\n"
            "The log_first_organism_pass() function must emit this string."
        )


def test_attestation_file_path_contract():
    """
    Regression test: Assert the FO test references artifacts/first_organism/attestation.json.
    
    Ensures the attestation file path hasn't been silently changed.
    This path is critical for Cursor P's certification process.
    """
    assert FO_TEST_FILE.exists(), (
        f"First Organism test file not found: {FO_TEST_FILE}"
    )
    
    source_code = FO_TEST_FILE.read_text(encoding="utf-8")
    
    # The canonical attestation path
    expected_path = "artifacts/first_organism/attestation.json"
    
    # Check that the path appears in the source (could be in string literal, comment, etc.)
    assert expected_path in source_code, (
        f"Critical attestation path '{expected_path}' not found in {FO_TEST_FILE}\n"
        "This path must be present - it's where the FO attestation artifact is written.\n"
        "If the path was changed, this test will fail to catch the regression."
    )
    
    # Also verify the _write_attestation_artifact function uses this path
    # This function is defined in the test file and should contain the path
    assert "_write_attestation_artifact" in source_code, (
        f"Function '_write_attestation_artifact' not found in {FO_TEST_FILE}\n"
        "This function should write the attestation artifact."
    )
    
    # Verify the _write_attestation_artifact function contains the path
    # Parse AST to find the function and verify it references the path
    tree = ast.parse(source_code, filename=str(FO_TEST_FILE))
    
    function_found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_write_attestation_artifact":
            function_found = True
            # Extract the function's source lines to check for the path
            # We'll check the raw source around the function definition
            lines = source_code.splitlines()
            # Find the function's line range
            func_start = node.lineno - 1  # AST line numbers are 1-indexed
            func_end = (node.end_lineno if hasattr(node, 'end_lineno') else func_start + 50) - 1
            func_lines = '\n'.join(lines[func_start:min(func_end + 1, len(lines))])
            
            assert expected_path in func_lines, (
                f"Function '_write_attestation_artifact' does not contain path '{expected_path}'\n"
                "The attestation artifact path must be hardcoded in this function."
            )
            break
    
    # If function not found in AST, the earlier assertion would have caught it
    # But we've already verified the path is in the source, so we're good


def test_attestation_h_t_recomputes():
    """
    Regression test: Assert that sealed attestation H_t recomputes from R_t||U_t.
    
    This verifies the core H_t Invariant: H_t = SHA256(R_t || U_t)
    The test must verify that stored H_t matches recomputed H_t using the
    canonical compute_composite_root() function.
    
    This is a static contract check - it verifies the code structure ensures
    recomputability, not that it actually runs.
    """
    assert FO_TEST_FILE.exists(), (
        f"First Organism test file not found: {FO_TEST_FILE}"
    )
    
    source_code = FO_TEST_FILE.read_text(encoding="utf-8")
    
    # Verify canonical imports from attestation.dual_root
    assert "from attestation.dual_root import" in source_code, (
        f"Missing import from attestation.dual_root in {FO_TEST_FILE}\n"
        "The test must import compute_composite_root and verify_composite_integrity."
    )
    
    # Verify compute_composite_root is imported
    assert "compute_composite_root" in source_code, (
        f"compute_composite_root not found in {FO_TEST_FILE}\n"
        "This function is required for H_t recomputability verification."
    )
    
    # Verify verify_composite_integrity is imported
    assert "verify_composite_integrity" in source_code, (
        f"verify_composite_integrity not found in {FO_TEST_FILE}\n"
        "This function is required for H_t integrity verification."
    )
    
    # Verify the helper function exists
    assert "_assert_composite_root_recomputable" in source_code, (
        f"Helper function '_assert_composite_root_recomputable' not found in {FO_TEST_FILE}\n"
        "This function encapsulates the H_t recomputability check."
    )
    
    # Verify the helper function uses compute_composite_root
    tree = ast.parse(source_code, filename=str(FO_TEST_FILE))
    
    helper_function_found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_assert_composite_root_recomputable":
            helper_function_found = True
            # Extract function source to verify it uses compute_composite_root
            lines = source_code.splitlines()
            func_start = node.lineno - 1
            func_end = (node.end_lineno if hasattr(node, 'end_lineno') else func_start + 30) - 1
            func_lines = '\n'.join(lines[func_start:min(func_end + 1, len(lines))])
            
            assert "compute_composite_root" in func_lines, (
                f"Function '_assert_composite_root_recomputable' does not use compute_composite_root\n"
                "The H_t recomputability check must use the canonical function."
            )
            
            assert "verify_composite_integrity" in func_lines, (
                f"Function '_assert_composite_root_recomputable' does not use verify_composite_integrity\n"
                "The H_t integrity check must use the canonical verification function."
            )
            break
    
    assert helper_function_found, (
        f"Helper function '_assert_composite_root_recomputable' not found in AST\n"
        "This function must exist to verify H_t recomputability."
    )
    
    # Verify that recomputability is checked in the critical test path
    # Check that the string "recomput" appears (for "recomputable", "recomputed", etc.)
    assert "recomput" in source_code.lower(), (
        f"H_t recomputability check not found in {FO_TEST_FILE}\n"
        "The test must verify that H_t can be recomputed from R_t and U_t."
    )


def test_rfl_evidence_presence_consistency():
    """
    Regression test: Assert that RFL evidence files must either be empty (marked incomplete)
    or non-empty with verifiable JSONL schema.
    
    This is a static contract check - it verifies that validation code exists,
    not that files are actually validated. Does NOT assert uplift or cycle count.
    
    Phase-I Compliance: This test only verifies infrastructure exists. For Phase-I RFL
    evidence facts (cycle counts, schema, abstention rates), see docs/RFL_PHASE_I_TRUTH_SOURCE.md.
    """
    # Verify that JSONL schema validation function exists
    verifier_file = Path(__file__).parent.parent.parent / "tools" / "devin_e_toolbox" / "artifact_verifier.py"
    
    if verifier_file.exists():
        verifier_code = verifier_file.read_text(encoding="utf-8")
        assert "def verify_jsonl_schema" in verifier_code, (
            f"verify_jsonl_schema function not found in {verifier_file}\n"
            "RFL evidence files must be validated against JSONL schema."
        )
    
    # Verify that empty file handling exists (in validate_fo_logs.py or similar)
    validate_file = Path(__file__).parent.parent.parent / "experiments" / "validate_fo_logs.py"
    
    if validate_file.exists():
        validate_code = validate_file.read_text(encoding="utf-8")
        # Check for empty file handling (should return {"exists": True, "empty": True})
        assert '"empty"' in validate_code or "'empty'" in validate_code, (
            f"Empty file handling not found in {validate_file}\n"
            "RFL evidence files must be marked as incomplete if empty."
        )
    
    # Verify that the contract is documented (evidence files must be empty OR valid JSONL)
    # This is a documentation check, not a runtime check
    # The actual validation happens at runtime, but the contract must be clear

