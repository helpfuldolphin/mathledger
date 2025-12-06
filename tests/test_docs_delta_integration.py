#!/usr/bin/env python3
"""
Integration tests for docs_delta.py baseline persistence.

Tests end-to-end baseline persistence workflow:
- First run: create baseline, all files "added"
- Second run: load baseline, all files "unchanged"
- Failure lens: missing artifact detection
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def create_test_docs_dir(tmpdir: Path) -> Path:
    """Create test docs directory with cross-ledger index."""
    docs_dir = tmpdir / "docs" / "methods"
    docs_dir.mkdir(parents=True)
    
    cross_ledger_index = {
        "format_version": "1.0",
        "sections": {}
    }
    (docs_dir / "cross_ledger_index.json").write_text(json.dumps(cross_ledger_index), encoding="ascii")
    
    return docs_dir


def test_baseline_persistence_first_run():
    """Test first run creates baseline with all files as 'added'."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        docs_dir = create_test_docs_dir(tmpdir)
        
        (docs_dir / "file1.md").write_text("# File 1", encoding="ascii")
        (docs_dir / "file2.md").write_text("# File 2", encoding="ascii")
        
        output_dir = tmpdir / "artifacts" / "docs"
        output_dir.mkdir(parents=True)
        
        baseline_path = output_dir / "baseline.json"
        delta_path = output_dir / "delta.json"
        
        result = subprocess.run(
            [
                sys.executable,
                "tools/docs/docs_delta.py",
                "--docs-dir", str(docs_dir),
                "--out", str(delta_path),
                "--write-baseline", str(baseline_path),
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        
        if result.returncode != 0:
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
        assert result.returncode == 0, f"First run failed: {result.stderr}"
        
        assert "No baseline found" not in result.stdout or "creating initial baseline" in result.stdout.lower()
        assert "[PASS] Docs Delta:" in result.stdout
        assert "Baseline SHA-256:" in result.stdout
        
        assert baseline_path.exists(), "Baseline file not created"
        
        with open(baseline_path, "r") as f:
            baseline = json.load(f)
        
        assert baseline["format_version"] == "1.0"
        assert baseline["baseline_type"] == "docs_delta_baseline"
        assert "checksums" in baseline
        assert len(baseline["checksums"]) == 3  # file1.md, file2.md, cross_ledger_index.json
        
        with open(delta_path, "r") as f:
            delta = json.load(f)
        
        assert delta["format_version"] == "1.0"
        assert delta["report_type"] == "docs_delta"
        assert len(delta["delta"]["added"]) == 3  # file1.md, file2.md, cross_ledger_index.json
        assert len(delta["delta"]["removed"]) == 0
        assert len(delta["delta"]["modified"]) == 0
        assert len(delta["delta"]["unchanged"]) == 0
        
        print("[OK] First run test passed")


def test_baseline_persistence_second_run():
    """Test second run loads baseline with all files as 'unchanged'."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        docs_dir = create_test_docs_dir(tmpdir)
        
        (docs_dir / "file1.md").write_text("# File 1", encoding="ascii")
        (docs_dir / "file2.md").write_text("# File 2", encoding="ascii")
        
        output_dir = tmpdir / "artifacts" / "docs"
        output_dir.mkdir(parents=True)
        
        baseline_path = output_dir / "baseline.json"
        delta_path = output_dir / "delta.json"
        
        result1 = subprocess.run(
            [
                sys.executable,
                "tools/docs/docs_delta.py",
                "--docs-dir", str(docs_dir),
                "--out", str(delta_path),
                "--write-baseline", str(baseline_path),
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        
        assert result1.returncode == 0, f"First run failed: {result1.stderr}"
        
        result2 = subprocess.run(
            [
                sys.executable,
                "tools/docs/docs_delta.py",
                "--docs-dir", str(docs_dir),
                "--baseline", str(baseline_path),
                "--out", str(delta_path),
                "--write-baseline", str(baseline_path),
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        
        assert result2.returncode == 0, f"Second run failed: {result2.stderr}"
        
        assert "Loading baseline" in result2.stdout
        assert "Loaded baseline with 3 checksums" in result2.stdout
        assert "[PASS] Docs Delta:" in result2.stdout
        
        with open(delta_path, "r") as f:
            delta = json.load(f)
        
        assert len(delta["delta"]["added"]) == 0
        assert len(delta["delta"]["removed"]) == 0
        assert len(delta["delta"]["modified"]) == 0
        assert len(delta["delta"]["unchanged"]) == 3  # file1.md, file2.md, cross_ledger_index.json
        
        print("[OK] Second run test passed")


def test_baseline_persistence_modified_files():
    """Test modified file detection."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        docs_dir = create_test_docs_dir(tmpdir)
        
        (docs_dir / "file1.md").write_text("# File 1", encoding="ascii")
        (docs_dir / "file2.md").write_text("# File 2", encoding="ascii")
        
        output_dir = tmpdir / "artifacts" / "docs"
        output_dir.mkdir(parents=True)
        
        baseline_path = output_dir / "baseline.json"
        delta_path = output_dir / "delta.json"
        
        result1 = subprocess.run(
            [
                sys.executable,
                "tools/docs/docs_delta.py",
                "--docs-dir", str(docs_dir),
                "--out", str(delta_path),
                "--write-baseline", str(baseline_path),
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        
        assert result1.returncode == 0
        
        (docs_dir / "file1.md").write_text("# File 1 Modified", encoding="ascii")
        
        result2 = subprocess.run(
            [
                sys.executable,
                "tools/docs/docs_delta.py",
                "--docs-dir", str(docs_dir),
                "--baseline", str(baseline_path),
                "--out", str(delta_path),
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        
        assert result2.returncode == 0
        
        with open(delta_path, "r") as f:
            delta = json.load(f)
        
        assert len(delta["delta"]["added"]) == 0
        assert len(delta["delta"]["removed"]) == 0
        assert len(delta["delta"]["modified"]) == 1
        assert len(delta["delta"]["unchanged"]) == 2  # file2.md, cross_ledger_index.json
        assert "methods/file1.md" in delta["delta"]["modified"]
        
        print("[OK] Modified files test passed")


def test_edge_case_corrupt_baseline():
    """Test graceful handling of corrupt baseline."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        docs_dir = create_test_docs_dir(tmpdir)
        (docs_dir / "file1.md").write_text("# File 1", encoding="ascii")
        
        output_dir = tmpdir / "artifacts" / "docs"
        output_dir.mkdir(parents=True)
        baseline_path = output_dir / "baseline.json"
        baseline_path.write_text("{invalid json", encoding="ascii")
        
        delta_path = output_dir / "delta.json"
        
        result = subprocess.run(
            [
                sys.executable,
                "tools/docs/docs_delta.py",
                "--docs-dir", str(docs_dir),
                "--baseline", str(baseline_path),
                "--out", str(delta_path),
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        
        assert result.returncode == 0
        
        assert "ABSTAIN: Baseline is corrupt" in result.stdout
        assert "Remediation:" in result.stdout
        assert "Continuing without baseline" in result.stdout
        
        with open(delta_path, "r") as f:
            delta = json.load(f)
        
        assert len(delta["delta"]["added"]) == 2  # file1.md, cross_ledger_index.json
        
        print("[OK] Corrupt baseline test passed")


def test_edge_case_wrong_version():
    """Test graceful handling of wrong version baseline."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        docs_dir = create_test_docs_dir(tmpdir)
        (docs_dir / "file1.md").write_text("# File 1", encoding="ascii")
        
        output_dir = tmpdir / "artifacts" / "docs"
        output_dir.mkdir(parents=True)
        baseline_path = output_dir / "baseline.json"
        
        wrong_version_baseline = {
            "format_version": "2.0",
            "baseline_type": "docs_delta_baseline",
            "checksums": {}
        }
        baseline_path.write_text(json.dumps(wrong_version_baseline), encoding="ascii")
        
        delta_path = output_dir / "delta.json"
        
        result = subprocess.run(
            [
                sys.executable,
                "tools/docs/docs_delta.py",
                "--docs-dir", str(docs_dir),
                "--baseline", str(baseline_path),
                "--out", str(delta_path),
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        
        assert result.returncode == 0
        
        assert "ABSTAIN: Baseline format version mismatch" in result.stdout
        assert "expected 1.0, got 2.0" in result.stdout
        assert "Remediation:" in result.stdout
        
        print("[OK] Wrong version test passed")


if __name__ == "__main__":
    print("Running integration tests for docs_delta.py baseline persistence...")
    
    test_baseline_persistence_first_run()
    test_baseline_persistence_second_run()
    test_baseline_persistence_modified_files()
    test_edge_case_corrupt_baseline()
    test_edge_case_wrong_version()
    
    print("\n[PASS] All integration tests passed (5/5)")
