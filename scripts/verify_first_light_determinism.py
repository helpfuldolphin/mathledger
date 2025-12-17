#!/usr/bin/env python3
"""
First-Light Determinism Verification

Runs P3 and P4 harnesses twice with identical configuration and verifies
that key artifacts are byte-identical.

SHADOW MODE CONTRACT:
- This script only verifies determinism
- No governance decisions are made or modified

Usage:
    python scripts/verify_first_light_determinism.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def run_p3_harness(output_dir: str) -> Path:
    """Run P3 harness and return output directory."""
    result = subprocess.run(
        [
            sys.executable,
            "scripts/usla_first_light_harness.py",
            "--cycles", "100",
            "--seed", "42",
            "--slice", "arithmetic_simple",
            "--runner-type", "u2",
            "--tau-0", "0.20",
            "--window-size", "20",
            "--output-dir", output_dir,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"P3 harness failed: {result.stderr}")
        sys.exit(1)

    # Find output directory
    dirs = list(Path(output_dir).glob("fl_*"))
    if not dirs:
        print(f"No fl_* directory found in {output_dir}")
        sys.exit(1)
    return dirs[0]


def run_p4_harness(output_dir: str) -> Path:
    """Run P4 harness and return output directory."""
    result = subprocess.run(
        [
            sys.executable,
            "scripts/usla_first_light_p4_harness.py",
            "--cycles", "100",
            "--seed", "42",
            "--slice", "arithmetic_simple",
            "--runner-type", "u2",
            "--tau-0", "0.20",
            "--output-dir", output_dir,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"P4 harness failed: {result.stderr}")
        sys.exit(1)

    # Find output directory
    dirs = list(Path(output_dir).glob("p4_*"))
    if not dirs:
        print(f"No p4_* directory found in {output_dir}")
        sys.exit(1)
    return dirs[0]


def compare_json_files(file1: Path, file2: Path, ignore_keys: List[str] = None) -> Tuple[bool, str]:
    """
    Compare two JSON files, optionally ignoring certain keys.

    Returns (is_identical, diff_description)
    """
    ignore_keys = ignore_keys or []

    with open(file1) as f:
        data1 = json.load(f)
    with open(file2) as f:
        data2 = json.load(f)

    def remove_keys(d, keys):
        if isinstance(d, dict):
            return {k: remove_keys(v, keys) for k, v in d.items() if k not in keys}
        elif isinstance(d, list):
            return [remove_keys(item, keys) for item in d]
        return d

    data1_filtered = remove_keys(data1, ignore_keys)
    data2_filtered = remove_keys(data2, ignore_keys)

    if data1_filtered == data2_filtered:
        return True, ""

    # Find differences
    diffs = []
    def find_diffs(d1, d2, path=""):
        if type(d1) != type(d2):
            diffs.append(f"{path}: type mismatch {type(d1).__name__} vs {type(d2).__name__}")
            return
        if isinstance(d1, dict):
            for k in set(d1.keys()) | set(d2.keys()):
                if k not in d1:
                    diffs.append(f"{path}.{k}: missing in run1")
                elif k not in d2:
                    diffs.append(f"{path}.{k}: missing in run2")
                else:
                    find_diffs(d1[k], d2[k], f"{path}.{k}")
        elif isinstance(d1, list):
            if len(d1) != len(d2):
                diffs.append(f"{path}: list length {len(d1)} vs {len(d2)}")
            else:
                for i, (v1, v2) in enumerate(zip(d1, d2)):
                    find_diffs(v1, v2, f"{path}[{i}]")
        elif d1 != d2:
            diffs.append(f"{path}: {d1} vs {d2}")

    find_diffs(data1_filtered, data2_filtered)
    return False, "; ".join(diffs[:5])


def main() -> int:
    print("=" * 60)
    print("First-Light Determinism Verification")
    print("=" * 60)
    print()

    base_dir = Path("results/determinism_test")
    if base_dir.exists():
        shutil.rmtree(base_dir)
    base_dir.mkdir(parents=True)

    # Run P3 twice
    print("Running P3 harness (run 1)...")
    p3_run1 = run_p3_harness(str(base_dir / "p3_run1"))
    print(f"  Output: {p3_run1}")

    print("Running P3 harness (run 2)...")
    p3_run2 = run_p3_harness(str(base_dir / "p3_run2"))
    print(f"  Output: {p3_run2}")

    # Run P4 twice
    print("Running P4 harness (run 1)...")
    p4_run1 = run_p4_harness(str(base_dir / "p4_run1"))
    print(f"  Output: {p4_run1}")

    print("Running P4 harness (run 2)...")
    p4_run2 = run_p4_harness(str(base_dir / "p4_run2"))
    print(f"  Output: {p4_run2}")

    print()
    print("Comparing artifacts...")
    print()

    all_passed = True

    # P3 comparisons (ignore timestamp-dependent fields)
    p3_files = [
        ("stability_report.json", ["timestamp", "start_time", "end_time", "timing"]),
        ("tda_metrics.json", ["timestamp"]),
        ("metrics_windows.json", []),
    ]

    print("P3 Artifacts:")
    for filename, ignore_keys in p3_files:
        file1 = p3_run1 / filename
        file2 = p3_run2 / filename

        if not file1.exists() or not file2.exists():
            print(f"  {filename}: MISSING")
            all_passed = False
            continue

        is_identical, diff = compare_json_files(file1, file2, ignore_keys)
        if is_identical:
            print(f"  {filename}: IDENTICAL")
        else:
            print(f"  {filename}: DIFFERS - {diff}")
            all_passed = False

    # P4 comparisons (ignore run_id which contains timestamp)
    p4_files = [
        ("p4_summary.json", ["start_time", "end_time", "execution", "run_id"]),
        ("twin_accuracy.json", []),
    ]

    print()
    print("P4 Artifacts:")
    for filename, ignore_keys in p4_files:
        file1 = p4_run1 / filename
        file2 = p4_run2 / filename

        if not file1.exists() or not file2.exists():
            print(f"  {filename}: MISSING")
            all_passed = False
            continue

        is_identical, diff = compare_json_files(file1, file2, ignore_keys)
        if is_identical:
            print(f"  {filename}: IDENTICAL")
        else:
            print(f"  {filename}: DIFFERS - {diff}")
            all_passed = False

    # Check raw data files - these contain timestamps so won't be byte-identical
    # but the numeric values should be identical
    print()
    print("Raw Data (content comparison, ignoring timestamps):")

    def compare_jsonl_numeric(file1: Path, file2: Path) -> Tuple[bool, str]:
        """Compare JSONL files ignoring timestamp fields."""
        with open(file1) as f:
            lines1 = [json.loads(line) for line in f]
        with open(file2) as f:
            lines2 = [json.loads(line) for line in f]

        if len(lines1) != len(lines2):
            return False, f"line count {len(lines1)} vs {len(lines2)}"

        # Remove timestamp fields for comparison
        def remove_timestamps(d):
            if isinstance(d, dict):
                return {k: remove_timestamps(v) for k, v in d.items()
                        if k not in ("timestamp", "start_time", "end_time")}
            elif isinstance(d, list):
                return [remove_timestamps(item) for item in d]
            return d

        for i, (r1, r2) in enumerate(zip(lines1, lines2)):
            r1_clean = remove_timestamps(r1)
            r2_clean = remove_timestamps(r2)
            if r1_clean != r2_clean:
                return False, f"record {i} differs"

        return True, ""

    p3_raw_ok, p3_raw_diff = compare_jsonl_numeric(
        p3_run1 / "synthetic_raw.jsonl",
        p3_run2 / "synthetic_raw.jsonl"
    )
    if p3_raw_ok:
        print(f"  P3 synthetic_raw.jsonl: IDENTICAL (ignoring timestamps)")
    else:
        print(f"  P3 synthetic_raw.jsonl: DIFFERS - {p3_raw_diff}")
        all_passed = False

    p4_real_ok, p4_real_diff = compare_jsonl_numeric(
        p4_run1 / "real_cycles.jsonl",
        p4_run2 / "real_cycles.jsonl"
    )
    if p4_real_ok:
        print(f"  P4 real_cycles.jsonl: IDENTICAL (ignoring timestamps)")
    else:
        print(f"  P4 real_cycles.jsonl: DIFFERS - {p4_real_diff}")
        all_passed = False

    p4_twin_ok, p4_twin_diff = compare_jsonl_numeric(
        p4_run1 / "twin_predictions.jsonl",
        p4_run2 / "twin_predictions.jsonl"
    )
    if p4_twin_ok:
        print(f"  P4 twin_predictions.jsonl: IDENTICAL (ignoring timestamps)")
    else:
        print(f"  P4 twin_predictions.jsonl: DIFFERS - {p4_twin_diff}")
        all_passed = False

    print()
    print("=" * 60)
    if all_passed:
        print("DETERMINISM CHECK: PASSED")
        print("All artifacts are identical between runs with same seed/config.")
    else:
        print("DETERMINISM CHECK: FAILED")
        print("Some artifacts differ between runs. Root cause must be identified.")
    print("=" * 60)

    # Cleanup
    shutil.rmtree(base_dir)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
