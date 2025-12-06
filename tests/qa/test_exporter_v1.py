import json, subprocess, tempfile, os, sys, pathlib, pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
EXPORTER = ROOT / "backend" / "tools" / "export_fol_ab.py"

def have_exporter():
    return EXPORTER.exists()

def run_exporter(args):
    env = os.environ.copy()
    env.setdefault("NO_NETWORK","true")
    p = subprocess.run([sys.executable, str(EXPORTER), *args], capture_output=True, text=True, env=env)
    return p.returncode, (p.stdout or "") + (p.stderr or "")

# ============================================================================
# PREFIX CONTRACT TESTS - MUST PASS SUBSET (FROZEN)
# These tests lock the message contract and must never regress
# ============================================================================

@pytest.mark.skipif(not have_exporter(), reason="exporter CLI not present yet")
def test_prefix_contract_dry_run_ok():
    """FROZEN: Test DRY-RUN ok: prefix contract - must never regress."""
    with tempfile.NamedTemporaryFile("w+", suffix=".jsonl", delete=False) as f:
        # Valid V1 statement record (matches exporter required_fields)
        rec = {"id": "stmt-1", "theory_id": "pl", "hash": "a"*64,
               "content_norm": "p -> p", "is_axiom": False}
        f.write(json.dumps(rec)+"\n"); f.flush()
        code, out = run_exporter(["--input", f.name, "--dry-run"])
        assert code == 0, f"Expected exit code 0, got {code}. Output: {out}"
        assert "DRY-RUN ok:" in out, f"Expected 'DRY-RUN ok:' prefix, got: {out}"

@pytest.mark.skipif(not have_exporter(), reason="exporter CLI not present yet")
def test_prefix_contract_mixed_schema():
    """FROZEN: Test mixed-schema: prefix contract - must never regress."""
    with tempfile.NamedTemporaryFile("w+", suffix=".jsonl", delete=False) as f:
        f.write('{"legacy": true}\n')
        f.write(json.dumps({"id": "stmt-1", "theory_id": "pl", "hash": "a"*64,
                            "content_norm": "p -> p", "is_axiom": False})+"\n")
        f.flush()
        code, out = run_exporter(["--input", f.name, "--dry-run"])
        assert code == 1, f"Expected exit code 1, got {code}. Output: {out}"
        assert "mixed-schema:" in out, f"Expected 'mixed-schema:' prefix, got: {out}"

@pytest.mark.skipif(not have_exporter(), reason="exporter CLI not present yet")
def test_prefix_contract_error_file_not_found():
    """FROZEN: Test error: prefix contract for file not found - must never regress."""
    nonexistent_file = "nonexistent_file_12345.jsonl"
    code, out = run_exporter(["--input", nonexistent_file, "--dry-run"])
    assert code == 1, f"Expected exit code 1, got {code}. Output: {out}"
    assert "error: File not found:" in out, f"Expected 'error: File not found:' prefix, got: {out}"

@pytest.mark.skipif(not have_exporter(), reason="exporter CLI not present yet")
def test_prefix_contract_error_empty_file():
    """FROZEN: Test error: prefix contract for empty file - must never regress."""
    with tempfile.NamedTemporaryFile("w+", suffix=".jsonl", delete=False) as f:
        f.write("")  # Empty file
        f.flush()
        code, out = run_exporter(["--input", f.name, "--dry-run"])
        assert code == 1, f"Expected exit code 1, got {code}. Output: {out}"
        assert "error: Empty file" in out, f"Expected 'error: Empty file' message, got: {out}"

# ============================================================================
# EDGE CASE TESTS - CRLF/LF TOLERANCE AND WINDOWS PATHS
# ============================================================================

@pytest.mark.skipif(not have_exporter(), reason="exporter CLI not present yet")
def test_crlf_lf_tolerance():
    """Test CRLF/LF line ending tolerance."""
    with tempfile.NamedTemporaryFile("w+", suffix=".jsonl", delete=False, newline='') as f:
        # Write with CRLF line endings
        rec = {"id": "stmt-1", "theory_id": "pl", "hash": "a"*64,
               "content_norm": "p -> p", "is_axiom": False}
        f.write(json.dumps(rec)+"\r\n")  # CRLF
        f.write(json.dumps(rec)+"\n")    # LF
        f.write(json.dumps(rec)+"\r\n")  # CRLF
        f.flush()
        code, out = run_exporter(["--input", f.name, "--dry-run"])
        assert code == 0, f"Expected exit code 0, got {code}. Output: {out}"
        assert "DRY-RUN ok:" in out, f"Expected 'DRY-RUN ok:' prefix, got: {out}"

@pytest.mark.skipif(not have_exporter(), reason="exporter CLI not present yet")
def test_windows_path_file_not_found():
    """Test Windows path handling for file not found errors."""
    # Test various Windows path formats
    windows_paths = [
        "C:\\nonexistent\\file.jsonl",
        "C:/nonexistent/file.jsonl",
        "\\\\server\\share\\nonexistent.jsonl",
        "relative\\path\\nonexistent.jsonl"
    ]

    for path in windows_paths:
        code, out = run_exporter(["--input", path, "--dry-run"])
        assert code == 1, f"Expected exit code 1 for path {path}, got {code}. Output: {out}"
        assert "error: File not found:" in out, f"Expected 'error: File not found:' for path {path}, got: {out}"
        # Check that the path appears in the error message (allowing for path normalization)
        path_normalized = path.replace("/", "\\")  # Normalize forward slashes to backslashes
        assert (path in out or path_normalized in out), f"Expected path {path} or {path_normalized} in error message, got: {out}"

@pytest.mark.skipif(not have_exporter(), reason="exporter CLI not present yet")
def test_mixed_line_endings():
    """Test file with mixed line endings (CRLF and LF)."""
    with tempfile.NamedTemporaryFile("w+", suffix=".jsonl", delete=False, newline='') as f:
        rec1 = {"id": "stmt-1", "theory_id": "pl", "hash": "a"*64, "content_norm": "p -> p", "is_axiom": False}
        rec2 = {"id": "stmt-2", "theory_id": "pl", "hash": "b"*64, "content_norm": "q -> q", "is_axiom": True}

        f.write(json.dumps(rec1)+"\r\n")  # CRLF
        f.write(json.dumps(rec2)+"\n")    # LF
        f.flush()
        code, out = run_exporter(["--input", f.name, "--dry-run"])
        assert code == 0, f"Expected exit code 0, got {code}. Output: {out}"
        assert "DRY-RUN ok:" in out, f"Expected 'DRY-RUN ok:' prefix, got: {out}"

# ============================================================================
# LEGACY TESTS (MAINTAINED FOR BACKWARD COMPATIBILITY)
# ============================================================================

@pytest.mark.skipif(not have_exporter(), reason="exporter CLI not present yet")
def test_dry_run_valid_v1_ok():
    """Legacy test - maintained for backward compatibility."""
    with tempfile.NamedTemporaryFile("w+", suffix=".jsonl", delete=False) as f:
        # Valid V1 statement record (matches exporter required_fields)
        rec = {"id": "stmt-1", "theory_id": "pl", "hash": "a"*64,
               "content_norm": "p -> p", "is_axiom": False}
        f.write(json.dumps(rec)+"\n"); f.flush()
        code, out = run_exporter(["--input", f.name, "--dry-run"])
        assert code == 0, out
        assert "DRY-RUN ok" in out, out

@pytest.mark.skipif(not have_exporter(), reason="exporter CLI not present yet")
def test_dry_run_mixed_schema_nonzero():
    """Legacy test - maintained for backward compatibility."""
    with tempfile.NamedTemporaryFile("w+", suffix=".jsonl", delete=False) as f:
        f.write('{"legacy": true}\n')
        f.write(json.dumps({"system":"fol","mode":"baseline","method":"fol-baseline","seed":"1",
                            "inserted_proofs":1,"wall_minutes":0.1,"block_no":1,"merkle":"0"*64})+"\n")
        f.flush()
        code, out = run_exporter(["--input", f.name, "--dry-run"])
        assert code != 0, out
        assert ("mixed-schema" in out.lower()) or ("mixed schema" in out.lower()) or ("error" in out.lower()), out
