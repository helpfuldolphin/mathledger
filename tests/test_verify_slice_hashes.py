
import subprocess
import json
from pathlib import Path
import yaml
import pytest
import sys
import os
import hashlib

# Add project root to allow importing the script
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Helper to get path to the script
SCRIPT_PATH = project_root / "scripts" / "verify_slice_hashes.py"

# --- Test Data Generation Helpers ---
DOMAIN_STMT = b'\x02'

def normalize_for_test(s: str) -> str:
    return "".join(s.split())

def sha256_statement_for_test(s: str) -> str:
    normalized = normalize_for_test(s)
    return hashlib.sha256(DOMAIN_STMT + normalized.encode("ascii")).hexdigest()

# --- Pytest Fixtures ---

@pytest.fixture
def valid_hashed_config(tmp_path: Path) -> Path:
    f_p_p_hash = sha256_statement_for_test("p->q->p")
    data = {"systems": [{"name": "pl", "slices": [
        {"name": "slice-1", "formula_pool_entries": [{"formula": "p->q->p", "hash": f_p_p_hash}]}
    ]}]}
    file_path = tmp_path / "valid_hashed.yaml"
    with open(file_path, "w") as f: yaml.dump(data, f)
    return file_path

@pytest.fixture
def duplicate_name_config(tmp_path: Path) -> Path:
    data = {"systems": [{"name": "pl", "slices": [
        {"name": "dupe-slice"}, {"name": "dupe-slice"}
    ]}]}
    file_path = tmp_path / "dupe_name.yaml"
    with open(file_path, "w") as f: yaml.dump(data, f)
    return file_path

# --- Test Functions ---

def run_script(*args) -> subprocess.CompletedProcess:
    """Helper to run the script in a subprocess."""
    return subprocess.run([sys.executable, str(SCRIPT_PATH)] + list(args), capture_output=True, text=True)

def test_verify_success(valid_hashed_config: Path):
    """Test that verification passes for a valid hashed file."""
    result = run_script("--verify", str(valid_hashed_config))
    assert result.returncode == 0
    assert "Verification PASSED" in result.stdout

def test_verify_duplicate_name_error(duplicate_name_config: Path, tmp_path: Path):
    """Test HASH-DRIFT-11 for duplicate slice names."""
    report_path = tmp_path / "report.json"
    result = run_script("--verify", str(duplicate_name_config), "--report", str(report_path))
    assert result.returncode == 1
    assert "Verification FAILED" in result.stderr
    assert "[FILE ERROR] Slice name 'dupe-slice' is duplicated 2 times." in result.stdout
    
    with open(report_path, "r") as f:
        report = json.load(f)
    
    assert len(report["file_level_errors"]) == 1
    assert report["file_level_errors"][0]["error_code"] == "HASH-DRIFT-11"

