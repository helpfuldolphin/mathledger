# tests/hash_observatory/test_reliability.py
import subprocess
from pathlib import Path
import sys

def test_file_system_reliability(tmp_path: Path):
    """
    A simple diagnostic test to confirm file write/read operations are reliable.
    """
    file_path = tmp_path / "reliability_check.txt"
    expected_content = "This is a test for file system reliability."
    
    # 1. Write the file
    try:
        file_path.write_text(expected_content, encoding='utf-8')
        print(f"\n[DIAGNOSTIC] Wrote to: {file_path.resolve()}")
    except Exception as e:
        pytest.fail(f"FATAL: write_text failed unexpectedly: {e}")

    # 2. Immediately read the file back
    try:
        actual_content = file_path.read_text(encoding='utf-8')
        print(f"[DIAGNOSTIC] Read back: '{actual_content}'")
    except Exception as e:
        pytest.fail(f"FATAL: read_text failed unexpectedly: {e}")
        
    # 3. Assert the content is what we expect
    assert actual_content == expected_content, "FATAL: File content mismatch. The test environment's file system is not reliable."

def test_script_execution_reliability(tmp_path: Path):
    """
    A simple diagnostic test to confirm a script can be written and then executed.
    """
    script_path = tmp_path / "simple_script.py"
    expected_output = "SIMPLE SCRIPT EXECUTED SUCCESSFULLY"
    
    script_content = f"print('{expected_output}')"
    
    # 1. Write the script file
    try:
        script_path.write_text(script_content, encoding='utf-8')
    except Exception as e:
        pytest.fail(f"FATAL: Could not write simple script for execution test: {e}")

    # 2. Execute the script
    result = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True, check=False)

    # 3. Assert the output is correct
    assert result.returncode == 0, f"Simple script failed to execute. Stderr: {result.stderr}"
    assert result.stdout.strip() == expected_output, "Simple script output was not as expected."

