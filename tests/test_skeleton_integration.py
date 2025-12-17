import subprocess
import sys
import unittest.mock

import pytest

# It's good practice to ensure the script path is discoverable
# For this test, we assume we run from the root of the project.
SCRIPT_PATH = "scripts/substrate_governance_check.py"
SCHEMA_PATH = "docs/governance/substrate/substrate_schema_draft07.json"

def test_import_is_non_interventional():
    """
    Asserts that importing the script module has no side effects like
    unexpected file or network access.
    """
    # Patch builtins.open to fail if called during import
    with unittest.mock.patch("builtins.open") as mock_open:
        # Patch socket to fail if any network access is attempted
        with unittest.mock.patch("socket.socket") as mock_socket:
            try:
                # We need to unload the module if it was already imported by another test
                if "scripts.substrate_governance_check" in sys.modules:
                    del sys.modules["scripts.substrate_governance_check"]
                
                import scripts.substrate_governance_check
                
                # We expect __main__ to not be run
                assert hasattr(scripts.substrate_governance_check, 'main')

            except Exception as e:
                pytest.fail(f"Importing the script failed unexpectedly: {e}")

            # Assert that no file was opened and no socket was created during import
            mock_open.assert_not_called()
            mock_socket.assert_not_called()

def test_shadow_mode_advisory_and_exit_code():
    """
    Runs the script in --shadow-only mode against RED and BLOCK fixtures
    and asserts the exit code is 0 and advisories are printed to stderr.
    """
    fixtures_to_test = {
        "RED": "tests/fixtures/health_red.json",
        "BLOCK": "tests/fixtures/health_block.json",
    }

    for status, fixture_path in fixtures_to_test.items():
        command = [
            sys.executable,
            SCRIPT_PATH,
            fixture_path,
            "--shadow-only",
            f"--schema-path={SCHEMA_PATH}",
        ]
        
        result = subprocess.run(command, capture_output=True, text=True)

        # Assert exit code is 0
        assert result.returncode == 0, f"Expected exit code 0 for {status} in shadow mode, but got {result.returncode}"

        # Assert advisory is printed to stderr
        assert "SHADOW-ONLY MODE ENABLED" in result.stderr, f"Missing shadow mode header for {status}"
        assert f"SUBSTRATE GOVERNANCE CHECK: FAILED ({status})" in result.stderr, f"Missing failure message for {status}"
        assert "EXITING 0 DUE TO SHADOW-ONLY MODE" in result.stderr, f"Missing shadow mode footer for {status}"

