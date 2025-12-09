
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

from collections import OrderedDict

# --- Test Data Generation Helpers ---
DOMAIN_STMT = b'\x02'

def normalize_for_test(s: str) -> str:
    return "".join(s.split())

def sha256_statement_for_test(s: str) -> str:
    normalized = normalize_for_test(s)
    return hashlib.sha256(DOMAIN_STMT + normalized.encode("ascii")).hexdigest()

def get_canonical_slice_hash_for_test(slice_data: dict) -> str:
    canonical_data = slice_data.copy()
    canonical_data.pop("name", None)
    canonical_str = json.dumps(canonical_data, sort_keys=True, separators=( ",", ":"))
    return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

# Custom YAML loader to handle OrderedDict
def ordered_load(stream, Loader=yaml.SafeLoader, object_pairs_hook=OrderedDict):
    class OrderedLoader(Loader):
        pass
    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))
    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)
    return yaml.load(stream, OrderedLoader)

# --- Pytest Fixtures ---

@pytest.fixture
def source_config_valid(tmp_path: Path) -> Path:
    p = tmp_path / "source_valid.yaml"
    p.write_text("""
version: 2.1.0
slices:
  slice-a:
    description: \"Goal-conditioned uplift slice...\"
    uplift:
      phase: II
      experiment_family: U2
      not_allowed_in_phase_I: true
    parameters:
      atoms: 4
      depth_min: 2
      depth_max: 5
      breadth_max: 40
      total_max: 200
      formula_pool: 16
      axiom_instances: 24
      timeout_s: 1.0
      lean_timeout_s: 0.0
    success_metric:
      kind: goal_hit
      parameters:
        min_goal_hits: 1
        min_total_verified: 3
    budget:
      max_candidates_per_cycle: 40
      max_cycles_per_run: 500
    formula_pool_entries:
      - \"p->q->p\"
      - \"((p->q)->p)->p\"
      - \"(p->q)->((q->r)->(p->r))\"
""")
    return p

@pytest.fixture
def source_config_dupe_slice_name(tmp_path: Path) -> Path:
    p = tmp_path / "source_dupe_slice.yaml"
    p.write_text("""
version: 1.0
slices:
  slice-a: {{}}
  slice-a: {{}}
""")
    return p

@pytest.fixture
def source_config_dupe_formula(tmp_path: Path) -> Path:
    p = tmp_path / "source_dupe_formula.yaml"
    p.write_text("""
version: 1.0
slices:
  slice-a:
    formula_pool_entries:
      - \"p->q\"
      - \"p->q\"
""")
    return p

# --- Test Functions ---

def run_script(*args) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "scripts/compile_curriculum.py"] + list(args), capture_output=True, text=True, check=False)

def test_compile_success(source_config_valid: Path, tmp_path: Path):
    output_path = tmp_path / "output.yaml"
    result = run_script("--source", str(source_config_valid), "--output", str(output_path))
    
    assert result.returncode == 0
    assert f"Successfully compiled and hashed curriculum to '{output_path}'" in result.stdout
    
    with open(output_path, "r") as f:
        hashed_data = ordered_load(f)  # Use ordered_load here
    
    assert hashed_data["version"] == "2.1.0-hashed"
    slices = hashed_data["systems"][0]["slices"]
    assert len(slices) == 1
    assert slices[0]["name"] == "slice-a"
    assert slices[0]["formula_pool_entries"][0]["formula"] == "p->q->p"
    assert "hash" in slices[0]["formula_pool_entries"][0]
    assert "target_hashes" in slices[0]["success_metric"]

def test_fail_on_duplicate_slice_name(source_config_dupe_slice_name: Path, tmp_path: Path):
    output_path = tmp_path / "output.yaml"
    result = run_script("--source", str(source_config_dupe_slice_name), "--output", str(output_path))
    
    assert result.returncode == 1
    assert "FATAL: HASH-DRIFT-11" in result.stderr

def test_fail_on_duplicate_formula(source_config_dupe_formula: Path, tmp_path: Path):
    output_path = tmp_path / "output.yaml"
    result = run_script("--source", str(source_config_dupe_formula), "--output", str(output_path))
    
    assert result.returncode == 1
    assert "FATAL: HASH-DRIFT-10" in result.stderr
