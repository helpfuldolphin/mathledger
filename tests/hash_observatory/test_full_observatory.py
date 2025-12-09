# tests/hash_observatory/test_full_observatory.py
import subprocess
import json
from pathlib import Path
import yaml
import pytest
import sys
from collections import OrderedDict

# --- Inlined Hashing Helper ---
def get_canonical_slice_hash(slice_data: dict) -> str:
    import hashlib
    canonical_data = slice_data.copy()
    canonical_data.pop("name", None)
    canonical_str = json.dumps(canonical_data, sort_keys=True, separators=( "," , ":"))
    return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

# --- Fixtures ---
@pytest.fixture
def source_curriculum(tmp_path: Path) -> Path:
    p = tmp_path / "source.yaml"
    p.write_text("""
version: 1.0
slices:
  slice-a:
    parameters: {atoms: 4}
    formula_pool_entries: ["p->q"]
""")
    return p

@pytest.fixture
def compiled_curriculum(tmp_path: Path, source_curriculum: Path) -> Path:
    output_path = tmp_path / "compiled.yaml"
    res = subprocess.run([sys.executable, "scripts/compile_curriculum.py", "--source", str(source_curriculum), "--output", str(output_path)], check=True)
    return output_path

@pytest.fixture
def prereg_file(tmp_path: Path, compiled_curriculum: Path) -> Path:
    p = tmp_path / "prereg.yaml"
    with open(compiled_curriculum, "r") as f:
        curriculum = yaml.safe_load(f)
    slice_a_data = next(s for s in curriculum["systems"][0]["slices"] if s["name"] == "slice-a")
    slice_a_hash = get_canonical_slice_hash(slice_a_data)
    p.write_text(f"- experiment_id: EXP001\n  slice_name: slice-a\n  slice_config_hash: '{slice_a_hash}'")
    return p

@pytest.fixture
def manifest_file(tmp_path: Path, compiled_curriculum: Path, prereg_file: Path) -> Path:
    output_path = tmp_path / "manifest.json"
    subprocess.run([sys.executable, "scripts/generate_execution_manifest.py", "EXP001", "--config", str(compiled_curriculum), "--prereg", str(prereg_file), "--output", str(output_path)], check=True)
    return output_path

# --- Tests ---
def test_full_workflow_is_consistent(compiled_curriculum, prereg_file, manifest_file, tmp_path):
    ledger_path = tmp_path / "ledger.json"
    result = subprocess.run([
        sys.executable, "scripts/hash_reconciliation_auditor.py",
        "--config", str(compiled_curriculum), "--prereg", str(prereg_file),
        "--manifest", str(manifest_file), "--output", str(ledger_path)
    ], capture_output=True, text=True)

    assert result.returncode == 0, f"Auditor failed on valid data: {result.stderr}"
    # This is the corrected assertion
    assert "Status: CONSISTENT" in result.stdout

def test_drift_injection_breaks_consistency(tmp_path, source_curriculum, compiled_curriculum, prereg_file, manifest_file):
    # HASH-DRIFT-9: Manifest curriculum hash mismatches actual curriculum
    ledger_path = tmp_path / "ledger.json"
    # Modify the source curriculum *after* the manifest has been generated
    source_curriculum.write_text("version: 2.0\n" + source_curriculum.read_text())
    # Re-compile to create a new, different hashed file
    drifted_compiled_path = tmp_path / "drifted_compiled.yaml"
    subprocess.run([sys.executable, "scripts/compile_curriculum.py", "--source", str(source_curriculum), "--output", str(drifted_compiled_path)], check=True)
    
    result = subprocess.run([
        sys.executable, "scripts/hash_reconciliation_auditor.py",
        "--config", str(drifted_compiled_path), # Audit against the NEW curriculum
        "--prereg", str(prereg_file),
        "--manifest", str(manifest_file), # But use the OLD manifest
        "--output", str(ledger_path)
    ], capture_output=True, text=True)
    
    assert result.returncode == 1, "Auditor should have failed due to HASH-DRIFT-9"
    assert "Status: INCONSISTENT" in result.stdout
    with open(ledger_path, "r") as f:
        ledger = json.load(f)
    assert any(r['details'].get('error_code') == 'HASH-DRIFT-9' for r in ledger['audit_results'])