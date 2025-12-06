#!/usr/bin/env python3
"""
CI Primer Simulator - Local simulation of GitHub Actions primer workflow

Simulates the three-cycle CI workflow:
1. First run (primer) - No previous artifacts, seeds current as previous
2. Second run (stable) - Compares against previous, expects stable
3. Third run (drift) - Introduces artificial file change, expects drift

This enables local validation of CI workflow logic without requiring actual CI execution.

Usage:
    python tools/docs/simulate_ci_primer.py
    python tools/docs/simulate_ci_primer.py --verbose
    python tools/docs/simulate_ci_primer.py --cleanup
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Tuple


def create_test_baseline(checksums: Dict[str, str], tmpdir: Path, name: str = "baseline") -> Path:
    """Create a test baseline file with given checksums."""
    baseline_data = {
        "format_version": "1.0",
        "baseline_type": "docs_delta_baseline",
        "checksums": checksums
    }
    
    baseline_path = tmpdir / f"{name}.json"
    canonical = json.dumps(baseline_data, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    
    with open(baseline_path, "w", encoding="ascii") as f:
        f.write(canonical)
    
    return baseline_path


def run_verification(baseline1: Path, baseline2: Path, output_dir: Path, verbose: bool = False) -> Tuple[int, str]:
    """Run baseline verification and return exit code and output."""
    output_json = output_dir / "baseline_verification.json"
    
    cmd = [
        sys.executable,
        "tools/docs/verify_baseline_ci.py",
        "--baseline1", str(baseline1),
        "--baseline2", str(baseline2),
        "--baseline-only",
        "--json-only", str(output_json),
        "--emit-drift-report",
        "--emit-artifact-metadata", str(output_dir / "artifact_metadata.json")
    ]
    
    result = subprocess.run(
        cmd,
        cwd="/home/ubuntu/repos/mathledger",
        capture_output=True,
        text=True
    )
    
    if verbose:
        print(f"Command: {' '.join(cmd)}")
        print(f"Exit code: {result.returncode}")
        print(f"Stdout: {result.stdout}")
        if result.stderr:
            print(f"Stderr: {result.stderr}")
    
    return result.returncode, result.stdout


def verify_signature(verification_json: Path, verbose: bool = False) -> Tuple[int, str]:
    """Verify signature of verification output."""
    cmd = [
        sys.executable,
        "tools/docs/verify_baseline_ci.py",
        "--verify-signature", str(verification_json)
    ]
    
    result = subprocess.run(
        cmd,
        cwd="/home/ubuntu/repos/mathledger",
        capture_output=True,
        text=True
    )
    
    if verbose:
        print(f"Command: {' '.join(cmd)}")
        print(f"Exit code: {result.returncode}")
        print(f"Stdout: {result.stdout}")
    
    return result.returncode, result.stdout


def main():
    parser = argparse.ArgumentParser(
        description="CI Primer Simulator - Local simulation of GitHub Actions workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--cleanup", action="store_true", help="Clean up simulation artifacts after run")
    
    args = parser.parse_args()
    
    tmpdir = Path(tempfile.mkdtemp(prefix="ci_sim_"))
    print(f"Simulation directory: {tmpdir}")
    
    try:
        print("\n=== CI Primer Simulation ===\n")
        
        run1_dir = tmpdir / "run1"
        run2_dir = tmpdir / "run2"
        run3_dir = tmpdir / "run3"
        
        run1_dir.mkdir()
        run2_dir.mkdir()
        run3_dir.mkdir()
        
        initial_checksums = {
            "file1.md": "sha256:abc123def456",
            "file2.md": "sha256:789012ghi345"
        }
        
        print("--- Run 1: Primer (First Run) ---")
        print("Simulating first CI run with no previous artifacts...")
        
        baseline1 = create_test_baseline(initial_checksums, run1_dir, "baseline_current")
        baseline_previous = create_test_baseline(initial_checksums, run1_dir, "baseline_previous")
        
        exit_code, output = run_verification(baseline_previous, baseline1, run1_dir, args.verbose)
        
        if exit_code == 0:
            print("[OK] Primer run succeeded (comparing current against itself)")
            print(f"Output: {output.strip()}")
        else:
            print(f"[FAIL] Primer run failed with exit code {exit_code}")
            return 1
        
        sig_exit, sig_output = verify_signature(run1_dir / "baseline_verification.json", args.verbose)
        if sig_exit == 0:
            print(f"[OK] Signature verification passed")
            print(f"Output: {sig_output.strip()}")
        else:
            print(f"[FAIL] Signature verification failed")
            return 1
        
        print("\n--- Run 2: Stable (Second Run) ---")
        print("Simulating second CI run with previous artifacts (no changes)...")
        
        baseline2 = create_test_baseline(initial_checksums, run2_dir, "baseline_current")
        
        exit_code, output = run_verification(baseline1, baseline2, run2_dir, args.verbose)
        
        if exit_code == 0:
            print("[OK] Second run succeeded (baselines stable)")
            print(f"Output: {output.strip()}")
        else:
            print(f"[FAIL] Second run failed with exit code {exit_code}")
            return 1
        
        sig_exit, sig_output = verify_signature(run2_dir / "baseline_verification.json", args.verbose)
        if sig_exit == 0:
            print(f"[OK] Signature verification passed")
        else:
            print(f"[FAIL] Signature verification failed")
            return 1
        
        print("\n--- Run 3: Drift (Third Run with Change) ---")
        print("Simulating third CI run with artificial file change...")
        
        modified_checksums = {
            "file1.md": "sha256:abc123def456",
            "file2.md": "sha256:789012ghi345",
            "file3.md": "sha256:newfile999"
        }
        
        baseline3 = create_test_baseline(modified_checksums, run3_dir, "baseline_current")
        
        exit_code, output = run_verification(baseline2, baseline3, run3_dir, args.verbose)
        
        if exit_code == 1:
            print("[OK] Third run detected drift (exit code 1)")
            print(f"Output: {output.strip()}")
            
            html_report = run3_dir / "baseline_drift_report.html"
            jsonl_report = run3_dir / "baseline_drift_report.jsonl"
            
            if html_report.exists():
                print(f"[OK] HTML drift report generated: {html_report}")
                print(f"     Size: {html_report.stat().st_size} bytes")
            else:
                print(f"[FAIL] HTML drift report not generated")
                return 1
            
            if jsonl_report.exists():
                print(f"[OK] JSONL drift report generated: {jsonl_report}")
                with open(jsonl_report, "r") as f:
                    lines = f.readlines()
                print(f"     Lines: {len(lines)}")
            else:
                print(f"[FAIL] JSONL drift report not generated")
                return 1
        else:
            print(f"[FAIL] Third run should have detected drift (exit code should be 1, got {exit_code})")
            return 1
        
        sig_exit, sig_output = verify_signature(run3_dir / "baseline_verification.json", args.verbose)
        if sig_exit == 0:
            print(f"[OK] Signature verification passed")
        else:
            print(f"[FAIL] Signature verification failed")
            return 1
        
        print("\n=== Simulation Complete ===")
        print(f"[PASS] All three cycles completed successfully")
        print(f"Artifacts preserved in: {tmpdir}")
        
        if args.cleanup:
            print(f"\nCleaning up simulation artifacts...")
            shutil.rmtree(tmpdir)
            print(f"[OK] Cleanup complete")
        
        return 0
    
    except Exception as e:
        print(f"\n[FAIL] Simulation error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
