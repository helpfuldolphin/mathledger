# tests/security/test_randomness_guards.py

import pytest
import subprocess
import sys
import os
from pathlib import Path

# --- Fixtures ---

@pytest.fixture
def temp_py_file(tmp_path: Path) -> Path:
    """A temporary python file for testing."""
    return tmp_path / "test_file.py"

# --- Static Linter Tests (Phase II) ---

def test_linter_finds_random_random_violation(temp_py_file: Path):
    """Test that the linter detects `random.random()`."""
    temp_py_file.write_text("import random\n\nx = random.random()")
    
    linter_path = "backend/security/randomness_static_linter.py"
    result = subprocess.run(
        [sys.executable, linter_path, str(temp_py_file)],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 1
    assert "Forbidden call to 'random.random'" in result.stdout

def test_linter_finds_np_random_violation(temp_py_file: Path):
    """Test that the linter detects `np.random.rand()`."""
    temp_py_file.write_text("import numpy as np\n\ny = np.random.rand(3)")
    
    linter_path = "backend/security/randomness_static_linter.py"
    result = subprocess.run(
        [sys.executable, linter_path, str(temp_py_file)],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 1
    assert "Forbidden call to 'numpy.random.rand'" in result.stdout

def test_linter_finds_os_urandom_violation(temp_py_file: Path):
    """Test that the linter detects `os.urandom()`."""
    temp_py_file.write_text("import os\n\nz = os.urandom(16)")
    
    linter_path = "backend/security/randomness_static_linter.py"
    result = subprocess.run(
        [sys.executable, linter_path, str(temp_py_file)],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 1
    assert "Forbidden call to 'os.urandom'" in result.stdout

def test_linter_finds_time_time_violation(temp_py_file: Path):
    """Test that the linter detects `time.time()`."""
    temp_py_file.write_text("import time\n\nt = time.time()")
    
    linter_path = "backend/security/randomness_static_linter.py"
    result = subprocess.run(
        [sys.executable, linter_path, str(temp_py_file)],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 1
    assert "Forbidden call to 'time.time'" in result.stdout

def test_linter_passes_clean_file(temp_py_file: Path):
    """Test that the linter passes a file with no violations."""
    temp_py_file.write_text("import math\n\nCLEAN = True\nx = math.pi")
    
    linter_path = "backend/security/randomness_static_linter.py"
    result = subprocess.run(
        [sys.executable, linter_path, str(temp_py_file)],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 0
    assert "Linter PASSED" in result.stdout

def test_linter_finds_from_import_violation(temp_py_file: Path):
    """Test that the linter detects `from random import random`."""
    temp_py_file.write_text("from random import random\n\nx = random()")
    
    linter_path = "backend/security/randomness_static_linter.py"
    result = subprocess.run(
        [sys.executable, linter_path, str(temp_py_file)],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 1
    assert "Direct call to forbidden import 'random.random'" in result.stdout

def test_linter_finds_unseeded_random_state(temp_py_file: Path):
    """Test that the linter detects unseeded `RandomState`."""
    temp_py_file.write_text("import numpy as np\n\nrs = np.random.RandomState()")
    
    linter_path = "backend/security/randomness_static_linter.py"
    result = subprocess.run(
        [sys.executable, linter_path, str(temp_py_file)],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 1
    assert "Unseeded numpy.random.RandomState instantiation" in result.stdout

def test_linter_passes_seeded_random_state(temp_py_file: Path):
    """Test that the linter allows `RandomState` when it is seeded."""
    temp_py_file.write_text("import numpy as np\n\nrs = np.random.RandomState(seed=42)")
    
    linter_path = "backend/security/randomness_static_linter.py"
    result = subprocess.run(
        [sys.executable, linter_path, str(temp_py_file)],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 0, f"Linter should pass but failed. stdout:\n{result.stdout}\nstderr:\n{result.stderr}"

@pytest.fixture
def waiver_whitelist_file(tmp_path: Path) -> Path:
    """Creates a temporary waiver whitelist file."""
    waiver_file = tmp_path / "determinism_waivers.yml"
    waiver_file.write_text("""
authorized_waivers:
  - TICKET-123
""")
    # Change CWD to the temp path so the linter can find the file.
    original_cwd = Path.cwd()
    os.chdir(tmp_path)
    yield waiver_file
    os.chdir(original_cwd)

def test_linter_honors_valid_waiver(temp_py_file: Path, waiver_whitelist_file: Path):
    """Test that a valid, whitelisted waiver is accepted."""
    temp_py_file.write_text("import random\nx = random.random() # DETERMINISM-WAIVER: JUSTIFIED TICKET-123")

    linter_path = Path.cwd().parent.parent / "backend/security/randomness_static_linter.py"
    result = subprocess.run(
        [sys.executable, str(linter_path), str(temp_py_file)],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 0
    assert "Linter Verdict: PASSED" in result.stdout
    assert "1 waived violation(s)" in result.stdout

def test_linter_rejects_unlisted_waiver(temp_py_file: Path, waiver_whitelist_file: Path):
    """Test that a waiver with a non-whitelisted ticket is a hard violation."""
    temp_py_file.write_text("import random\nx = random.random() # DETERMINISM-WAIVER: JUSTIFIED TICKET-999")

    linter_path = Path.cwd().parent.parent / "backend/security/randomness_static_linter.py"
    result = subprocess.run(
        [sys.executable, str(linter_path), str(temp_py_file)],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 1
    assert "Linter Verdict: FAILED" in result.stdout
    assert "Unapproved waiver ticket 'TICKET-999'" in result.stderr

def test_linter_rejects_malformed_waiver(temp_py_file: Path, waiver_whitelist_file: Path):
    """Test that a violation with a malformed waiver comment is a hard violation."""
    temp_py_file.write_text("import random\nx = random.random() # WAIVER: TICKET-123")

    linter_path = Path.cwd().parent.parent / "backend/security/randomness_static_linter.py"
    result = subprocess.run(
        [sys.executable, str(linter_path), str(temp_py_file)],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 1
    assert "Linter Verdict: FAILED" in result.stdout
    assert "Forbidden call to 'random.random'" in result.stderr



# --- Runtime Guard Tests (Phase III) ---

# Need to import these before the guard can patch them
import random
import numpy.random
import os
import time
from backend.security.u2_security import SecurityException
from backend.security.randomness_runtime_guard import activate_runtime_guard, allow_unrestricted_randomness

def test_runtime_guard_raises_on_random_random():
    """Test that the runtime guard blocks `random.random`."""
    activate_runtime_guard()
    with pytest.raises(SecurityException, match="Illegal call to 'random.random'"):
        random.random()

def test_runtime_guard_raises_on_np_random_rand():
    """Test that the runtime guard blocks `numpy.random.rand`."""
    activate_runtime_guard()
    with pytest.raises(SecurityException, match="Illegal call to 'numpy.random.rand'"):
        numpy.random.rand()

def test_runtime_guard_raises_on_os_urandom():
    """Test that the runtime guard blocks `os.urandom`."""
    activate_runtime_guard()
    with pytest.raises(SecurityException, match="Illegal call to 'os.urandom'"):
        os.urandom(1)

def test_runtime_guard_raises_on_time_time():
    """Test that the runtime guard blocks `time.time`."""
    activate_runtime_guard()
    with pytest.raises(SecurityException, match="Illegal call to 'time.time'"):
        time.time()

def test_allow_unrestricted_randomness_context_manager():
    """Test that the context manager correctly allows and re-blocks calls."""
    activate_runtime_guard()

    # It should fail before entering the context manager
    with pytest.raises(SecurityException):
        random.random()

    # It should succeed inside the context manager
    with allow_unrestricted_randomness():
        val1 = random.random()
        val2 = time.time()
        assert isinstance(val1, float)
        assert isinstance(val2, float)

    # It should fail again after exiting the context manager
    with pytest.raises(SecurityException):
        random.random()
