"""
Integration tests for verify_baseline_ci.py

Tests baseline stability verification with Proof-or-Abstain discipline.
"""

import json
import subprocess
import tempfile
from pathlib import Path


def create_test_baseline(tmpdir: Path, checksums: dict, name: str = "baseline") -> Path:
    """Create a test baseline file."""
    baseline_data = {
        "format_version": "1.0",
        "baseline_type": "docs_delta_baseline",
        "checksums": checksums
    }
    baseline_path = tmpdir / f"{name}.json"
    with open(baseline_path, "w", encoding="ascii") as f:
        json.dump(baseline_data, f, sort_keys=True, separators=(',', ':'))
    return baseline_path


def create_test_delta_report(tmpdir: Path, checksums: dict, run_name: str) -> Path:
    """Create a test delta report."""
    delta_data = {
        "format_version": "1.0",
        "report_type": "docs_delta",
        "checksums": checksums,
        "delta": {
            "added": [],
            "removed": [],
            "modified": [],
            "unchanged": list(checksums.keys())
        },
        "failures": {
            "missing_artifacts": [],
            "broken_cross_links": [],
            "non_ascii_files": []
        }
    }
    delta_path = tmpdir / f"{run_name}_delta.json"
    with open(delta_path, "w", encoding="ascii") as f:
        json.dump(delta_data, f, sort_keys=True, separators=(',', ':'))
    return delta_path


def test_baseline_stable():
    """Test baseline stability detection (no drift)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        checksums = {
            "file1.md": "sha256:abc123",
            "file2.md": "sha256:def456",
            "file3.md": "sha256:ghi789"
        }
        
        baseline1 = create_test_baseline(tmpdir, checksums)
        baseline2 = create_test_baseline(tmpdir, checksums)
        
        result = subprocess.run(
            [
                "python", "tools/docs/verify_baseline_ci.py",
                "--baseline1", str(baseline1),
                "--baseline2", str(baseline2),
                "--baseline-only"
            ],
            cwd="/home/ubuntu/repos/mathledger",
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0, f"Verification failed: {result.stderr}"
        assert "[PASS] Baseline Stable Δ=0" in result.stdout
        assert "Baseline hashes identical" in result.stdout
        print("[OK] Baseline stable test passed")


def test_baseline_drift_added():
    """Test baseline drift detection (files added)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        checksums1 = {
            "file1.md": "sha256:abc123",
            "file2.md": "sha256:def456"
        }
        checksums2 = {
            "file1.md": "sha256:abc123",
            "file2.md": "sha256:def456",
            "file3.md": "sha256:ghi789"  # Added
        }
        
        baseline1 = create_test_baseline(tmpdir, checksums1, "baseline1")
        baseline2 = create_test_baseline(tmpdir, checksums2, "baseline2")
        
        result = subprocess.run(
            [
                "python", "tools/docs/verify_baseline_ci.py",
                "--baseline1", str(baseline1),
                "--baseline2", str(baseline2),
                "--baseline-only"
            ],
            cwd="/home/ubuntu/repos/mathledger",
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 1, f"Verification should fail on drift, got: {result.returncode}\nSTDOUT: {result.stdout}"
        assert "[FAIL] Baseline Drift Detected add=1 rm=0 mod=0" in result.stdout, f"Expected drift message not found in: {result.stdout}"
        assert "+ file3.md" in result.stdout, f"Expected added file not found in: {result.stdout}"
        print("[OK] Baseline drift (added) test passed")


def test_baseline_drift_removed():
    """Test baseline drift detection (files removed)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        checksums1 = {
            "file1.md": "sha256:abc123",
            "file2.md": "sha256:def456",
            "file3.md": "sha256:ghi789"
        }
        checksums2 = {
            "file1.md": "sha256:abc123",
            "file2.md": "sha256:def456"
        }
        
        baseline1 = create_test_baseline(tmpdir, checksums1, "baseline1")
        baseline2 = create_test_baseline(tmpdir, checksums2, "baseline2")
        
        result = subprocess.run(
            [
                "python", "tools/docs/verify_baseline_ci.py",
                "--baseline1", str(baseline1),
                "--baseline2", str(baseline2),
                "--baseline-only"
            ],
            cwd="/home/ubuntu/repos/mathledger",
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 1, "Verification should fail on drift"
        assert "[FAIL] Baseline Drift Detected add=0 rm=1 mod=0" in result.stdout
        assert "- file3.md" in result.stdout
        print("[OK] Baseline drift (removed) test passed")


def test_baseline_drift_modified():
    """Test baseline drift detection (files modified)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        checksums1 = {
            "file1.md": "sha256:abc123",
            "file2.md": "sha256:def456",
            "file3.md": "sha256:ghi789"
        }
        checksums2 = {
            "file1.md": "sha256:abc123",
            "file2.md": "sha256:def456",
            "file3.md": "sha256:xyz999"  # Modified
        }
        
        baseline1 = create_test_baseline(tmpdir, checksums1, "baseline1")
        baseline2 = create_test_baseline(tmpdir, checksums2, "baseline2")
        
        result = subprocess.run(
            [
                "python", "tools/docs/verify_baseline_ci.py",
                "--baseline1", str(baseline1),
                "--baseline2", str(baseline2),
                "--baseline-only"
            ],
            cwd="/home/ubuntu/repos/mathledger",
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 1, "Verification should fail on drift"
        assert "[FAIL] Baseline Drift Detected add=0 rm=0 mod=1" in result.stdout
        assert "~ file3.md" in result.stdout
        print("[OK] Baseline drift (modified) test passed")


def test_auto_detect_baselines():
    """Test auto-detection of baselines from delta reports."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        checksums = {
            "file1.md": "sha256:abc123",
            "file2.md": "sha256:def456"
        }
        
        run1 = create_test_delta_report(tmpdir, checksums, "run1")
        run2 = create_test_delta_report(tmpdir, checksums, "run2")
        
        result = subprocess.run(
            [
                "python", "tools/docs/verify_baseline_ci.py",
                "--run1", str(run1),
                "--run2", str(run2),
                "--auto-detect-baselines"
            ],
            cwd="/home/ubuntu/repos/mathledger",
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0, f"Verification failed: {result.stderr}"
        assert "[PASS] Baseline Stable Δ=0" in result.stdout
        print("[OK] Auto-detect baselines test passed")


def test_corrupt_baseline():
    """Test ABSTAIN on corrupt baseline."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        checksums = {"file1.md": "sha256:abc123"}
        baseline1 = create_test_baseline(tmpdir, checksums)
        
        baseline2 = tmpdir / "corrupt.json"
        with open(baseline2, "w") as f:
            f.write("{invalid json")
        
        result = subprocess.run(
            [
                "python", "tools/docs/verify_baseline_ci.py",
                "--baseline1", str(baseline1),
                "--baseline2", str(baseline2),
                "--baseline-only"
            ],
            cwd="/home/ubuntu/repos/mathledger",
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 1, "Should fail on corrupt baseline"
        assert "ABSTAIN: Invalid JSON" in result.stdout
        assert "Remediation:" in result.stdout
        print("[OK] Corrupt baseline test passed")


def test_missing_file():
    """Test ABSTAIN on missing file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        checksums = {"file1.md": "sha256:abc123"}
        baseline1 = create_test_baseline(tmpdir, checksums)
        
        baseline2 = tmpdir / "nonexistent.json"
        
        result = subprocess.run(
            [
                "python", "tools/docs/verify_baseline_ci.py",
                "--baseline1", str(baseline1),
                "--baseline2", str(baseline2),
                "--baseline-only"
            ],
            cwd="/home/ubuntu/repos/mathledger",
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 1, "Should fail on missing file"
        assert "ABSTAIN: File not found" in result.stdout
        assert "Remediation:" in result.stdout
        print("[OK] Missing file test passed")


def test_json_only_mode_stable():
    """Test JSON-only mode with stable baselines."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        checksums = {
            "file1.md": "sha256:abc123",
            "file2.md": "sha256:def456"
        }
        
        baseline1 = create_test_baseline(tmpdir, checksums, "baseline1")
        baseline2 = create_test_baseline(tmpdir, checksums, "baseline2")
        output_json = tmpdir / "verification.json"
        
        result = subprocess.run(
            [
                "python", "tools/docs/verify_baseline_ci.py",
                "--baseline1", str(baseline1),
                "--baseline2", str(baseline2),
                "--baseline-only",
                "--json-only", str(output_json)
            ],
            cwd="/home/ubuntu/repos/mathledger",
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0, f"Verification failed: {result.stderr}"
        assert result.stdout == "", f"Expected no stdout in JSON-only mode, got: {result.stdout}"
        assert output_json.exists(), "JSON output file not created"
        
        with open(output_json, "r") as f:
            output_data = json.load(f)
        
        assert output_data["result"] == "PASS", f"Expected PASS result, got: {output_data['result']}"
        assert output_data["message"] == "Baseline Stable Δ=0"
        assert output_data["drift"]["added"] == 0
        assert output_data["drift"]["removed"] == 0
        assert output_data["drift"]["modified"] == 0
        assert "signature" in output_data
        print("[OK] JSON-only mode (stable) test passed")


def test_json_only_mode_drift():
    """Test JSON-only mode with drift detected."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        checksums1 = {
            "file1.md": "sha256:abc123",
            "file2.md": "sha256:def456"
        }
        checksums2 = {
            "file1.md": "sha256:abc123",
            "file2.md": "sha256:def456",
            "file3.md": "sha256:ghi789"
        }
        
        baseline1 = create_test_baseline(tmpdir, checksums1, "baseline1")
        baseline2 = create_test_baseline(tmpdir, checksums2, "baseline2")
        output_json = tmpdir / "verification.json"
        
        result = subprocess.run(
            [
                "python", "tools/docs/verify_baseline_ci.py",
                "--baseline1", str(baseline1),
                "--baseline2", str(baseline2),
                "--baseline-only",
                "--json-only", str(output_json)
            ],
            cwd="/home/ubuntu/repos/mathledger",
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 1, "Verification should fail on drift"
        assert result.stdout == "", f"Expected no stdout in JSON-only mode, got: {result.stdout}"
        assert output_json.exists(), "JSON output file not created"
        
        with open(output_json, "r") as f:
            output_data = json.load(f)
        
        assert output_data["result"] == "FAIL", f"Expected FAIL result, got: {output_data['result']}"
        assert "add=1 rm=0 mod=0" in output_data["message"]
        assert output_data["drift"]["added"] == 1
        assert output_data["drift"]["removed"] == 0
        assert output_data["drift"]["modified"] == 0
        assert output_data["files"]["added"] == ["file3.md"]
        assert "signature" in output_data
        print("[OK] JSON-only mode (drift) test passed")


def test_signature_verification_valid():
    """Test signature verification with valid signature."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        checksums = {
            "file1.md": "sha256:abc123",
            "file2.md": "sha256:def456"
        }
        
        baseline1 = create_test_baseline(tmpdir, checksums, "baseline1")
        baseline2 = create_test_baseline(tmpdir, checksums, "baseline2")
        output_json = tmpdir / "verification.json"
        
        subprocess.run(
            [
                "python", "tools/docs/verify_baseline_ci.py",
                "--baseline1", str(baseline1),
                "--baseline2", str(baseline2),
                "--baseline-only",
                "--json-only", str(output_json)
            ],
            cwd="/home/ubuntu/repos/mathledger",
            capture_output=True,
            text=True,
            check=True
        )
        
        result = subprocess.run(
            [
                "python", "tools/docs/verify_baseline_ci.py",
                "--verify-signature", str(output_json)
            ],
            cwd="/home/ubuntu/repos/mathledger",
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0, "Signature verification should pass"
        assert "[PASS] Baseline Signature verified=true" in result.stdout
        print("[OK] Signature verification (valid) test passed")


def test_signature_verification_invalid():
    """Test signature verification with tampered signature."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        checksums = {
            "file1.md": "sha256:abc123",
            "file2.md": "sha256:def456"
        }
        
        baseline1 = create_test_baseline(tmpdir, checksums, "baseline1")
        baseline2 = create_test_baseline(tmpdir, checksums, "baseline2")
        output_json = tmpdir / "verification.json"
        
        subprocess.run(
            [
                "python", "tools/docs/verify_baseline_ci.py",
                "--baseline1", str(baseline1),
                "--baseline2", str(baseline2),
                "--baseline-only",
                "--json-only", str(output_json)
            ],
            cwd="/home/ubuntu/repos/mathledger",
            capture_output=True,
            text=True,
            check=True
        )
        
        with open(output_json, "r") as f:
            data = json.load(f)
        
        data["signature"] = "0" * 64
        
        with open(output_json, "w") as f:
            json.dump(data, f)
        
        result = subprocess.run(
            [
                "python", "tools/docs/verify_baseline_ci.py",
                "--verify-signature", str(output_json)
            ],
            cwd="/home/ubuntu/repos/mathledger",
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 1, "Signature verification should fail"
        assert "[FAIL] Baseline Signature mismatch" in result.stdout
        print("[OK] Signature verification (invalid) test passed")


def test_drift_visualization():
    """Test drift visualization HTML and JSONL generation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        checksums1 = {
            "file1.md": "sha256:abc123",
            "file2.md": "sha256:def456"
        }
        checksums2 = {
            "file1.md": "sha256:abc123",
            "file2.md": "sha256:def456",
            "file3.md": "sha256:ghi789"
        }
        
        baseline1 = create_test_baseline(tmpdir, checksums1, "baseline1")
        baseline2 = create_test_baseline(tmpdir, checksums2, "baseline2")
        output_json = tmpdir / "verification.json"
        
        result = subprocess.run(
            [
                "python", "tools/docs/verify_baseline_ci.py",
                "--baseline1", str(baseline1),
                "--baseline2", str(baseline2),
                "--baseline-only",
                "--json-only", str(output_json),
                "--emit-drift-report"
            ],
            cwd="/home/ubuntu/repos/mathledger",
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 1, "Verification should fail on drift"
        assert "[PASS] Drift Visualization generated files=1" in result.stdout
        
        html_report = tmpdir / "baseline_drift_report.html"
        jsonl_report = tmpdir / "baseline_drift_report.jsonl"
        
        assert html_report.exists(), "HTML report should be generated"
        assert jsonl_report.exists(), "JSONL report should be generated"
        
        with open(html_report, "r") as f:
            html_content = f.read()
        
        assert "<!DOCTYPE html>" in html_content
        assert "Baseline Drift Report" in html_content
        assert "file3.md" in html_content
        assert "Added" in html_content
        
        with open(jsonl_report, "r") as f:
            jsonl_lines = f.readlines()
        
        assert len(jsonl_lines) == 2, "JSONL should have header + 1 file change"
        
        header = json.loads(jsonl_lines[0])
        assert header["record_type"] == "header"
        assert header["drift_summary"]["added"] == 1
        
        file_change = json.loads(jsonl_lines[1])
        assert file_change["record_type"] == "file_change"
        assert file_change["file"] == "file3.md"
        assert file_change["status"] == "added"
        
        print("[OK] Drift visualization test passed")


def test_artifact_metadata():
    """Test artifact metadata generation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        checksums = {
            "file1.md": "sha256:abc123",
            "file2.md": "sha256:def456"
        }
        
        baseline1 = create_test_baseline(tmpdir, checksums, "baseline1")
        baseline2 = create_test_baseline(tmpdir, checksums, "baseline2")
        output_json = tmpdir / "verification.json"
        metadata_json = tmpdir / "artifact_metadata.json"
        
        result = subprocess.run(
            [
                "python", "tools/docs/verify_baseline_ci.py",
                "--baseline1", str(baseline1),
                "--baseline2", str(baseline2),
                "--baseline-only",
                "--json-only", str(output_json),
                "--emit-artifact-metadata", str(metadata_json)
            ],
            cwd="/home/ubuntu/repos/mathledger",
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0, "Verification should pass"
        assert metadata_json.exists(), "Metadata file should be generated"
        
        with open(metadata_json, "r") as f:
            metadata = json.load(f)
        
        assert metadata["format_version"] == "1.0"
        assert metadata["artifact_type"] == "baseline_verification_metadata"
        assert "run_id" in metadata
        assert "baseline1_sha256" in metadata
        assert "baseline2_sha256" in metadata
        assert "verification_sha256" in metadata
        assert metadata["stable"] == True
        
        print("[OK] Artifact metadata test passed")


if __name__ == "__main__":
    print("Running integration tests for verify_baseline_ci.py...")
    
    try:
        test_baseline_stable()
        test_baseline_drift_added()
        test_baseline_drift_removed()
        test_baseline_drift_modified()
        test_auto_detect_baselines()
        test_corrupt_baseline()
        test_missing_file()
        test_json_only_mode_stable()
        test_json_only_mode_drift()
        test_signature_verification_valid()
        test_signature_verification_invalid()
        test_drift_visualization()
        test_artifact_metadata()
        
        print("\n[PASS] All integration tests passed (13/13)")
    except AssertionError as e:
        print(f"\n[FAIL] Test failed: {e}")
        exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        exit(1)
