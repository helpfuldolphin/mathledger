#!/usr/bin/env python3
"""
Dyno Chart QA Cross-Check (Cursor L)

Performs comprehensive validation of:
1. Log file sanity (wide slice baseline + RFL)
2. Schema compatibility with analysis script
3. Dyno Chart integrity

Usage:
    python experiments/qa_dyno_chart.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd

# Expected file paths
BASELINE_WIDE = Path("results/fo_baseline_wide.jsonl")
RFL_WIDE = Path("results/fo_rfl_wide.jsonl")
DYNO_CHART = Path("artifacts/figures/rfl_dyno_chart.png")
ANALYSIS_SCRIPT = Path("experiments/analyze_abstention_curves.py")

# Required fields for analysis script compatibility
REQUIRED_FIELDS = ["cycle"]
OPTIONAL_FIELDS = ["status", "method", "verification_method", "abstention"]


class QAResult:
    """Container for QA check results."""
    def __init__(self, name: str):
        self.name = name
        self.passed = True
        self.issues: List[str] = []
        self.warnings: List[str] = []
    
    def fail(self, message: str):
        self.passed = False
        self.issues.append(message)
    
    def warn(self, message: str):
        self.warnings.append(message)
    
    def __str__(self) -> str:
        status = "✓ PASS" if self.passed else "✗ FAIL"
        result = f"{status}: {self.name}\n"
        if self.issues:
            result += "  Issues:\n"
            for issue in self.issues:
                result += f"    - {issue}\n"
        if self.warnings:
            result += "  Warnings:\n"
            for warning in self.warnings:
                result += f"    - {warning}\n"
        return result


def load_jsonl(filepath: Path) -> List[Dict[str, Any]]:
    """Load JSONL file and return list of entries."""
    if not filepath.exists():
        return []
    
    entries = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                entries.append(entry)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {i} in {filepath}: {e}")
    
    return entries


def check_log_sanity(filepath: Path, label: str) -> QAResult:
    """Check log file for sanity issues."""
    result = QAResult(f"Log Sanity: {label}")
    
    if not filepath.exists():
        result.fail(f"File does not exist: {filepath}")
        return result
    
    try:
        entries = load_jsonl(filepath)
    except Exception as e:
        result.fail(f"Failed to load file: {e}")
        return result
    
    if not entries:
        result.fail("File is empty")
        return result
    
    # Check for cycle indices
    cycles = [e.get("cycle") for e in entries if "cycle" in e]
    if not cycles:
        result.fail("No 'cycle' field found in entries")
        return result
    
    # Check for missing cycle indices (0..N-1)
    cycles_set = set(cycles)
    expected_cycles = set(range(len(entries)))
    missing = expected_cycles - cycles_set
    if missing:
        result.warn(f"Missing cycle indices: {sorted(list(missing))[:10]}{'...' if len(missing) > 10 else ''}")
    
    # Check for duplicate cycles
    if len(cycles) != len(cycles_set):
        duplicates = [c for c in cycles if cycles.count(c) > 1]
        result.fail(f"Duplicate cycle indices found: {set(duplicates)}")
    
    # Check cycle ordering
    if cycles != sorted(cycles):
        result.warn("Cycle indices are not in ascending order")
    
    # Check abstention distribution
    abstention_values = []
    for entry in entries:
        # Try multiple ways to determine abstention
        is_abstention = False
        if "abstention" in entry:
            is_abstention = bool(entry["abstention"])
        elif "status" in entry:
            is_abstention = str(entry["status"]).lower() == "abstain"
        elif "derivation" in entry and isinstance(entry["derivation"], dict):
            # Fallback: check derivation.abstained > 0
            abstained = entry["derivation"].get("abstained", 0)
            is_abstention = abstained > 0
        
        abstention_values.append(is_abstention)
    
    if abstention_values:
        abstention_rate = sum(abstention_values) / len(abstention_values)
        
        # Check for degenerate cases
        if abstention_rate == 0.0:
            result.warn("No abstentions found (0% abstention rate)")
        elif abstention_rate == 1.0:
            result.fail("100% abstention rate - degenerate case")
        elif abstention_rate > 0.95:
            result.warn(f"Very high abstention rate: {abstention_rate:.1%}")
        elif abstention_rate < 0.05:
            result.warn(f"Very low abstention rate: {abstention_rate:.1%}")
        else:
            result.warnings.append(f"Abstention rate: {abstention_rate:.1%} (reasonable)")
    
    # Check entry count
    result.warnings.append(f"Total entries: {len(entries)}")
    result.warnings.append(f"Cycle range: {min(cycles)} to {max(cycles)}")
    
    return result


def check_schema_compatibility(filepath: Path, label: str) -> QAResult:
    """Check schema compatibility with analysis script."""
    result = QAResult(f"Schema Compatibility: {label}")
    
    if not filepath.exists():
        result.fail(f"File does not exist: {filepath}")
        return result
    
    try:
        entries = load_jsonl(filepath)
    except Exception as e:
        result.fail(f"Failed to load file: {e}")
        return result
    
    if not entries:
        result.fail("File is empty")
        return result
    
    # Check for required fields
    sample = entries[0]
    if "cycle" not in sample:
        result.fail("Missing required field: 'cycle'")
    
    # Check for fields that analysis script expects
    has_status = "status" in sample
    has_method = "method" in sample
    has_verification_method = "verification_method" in sample
    has_abstention = "abstention" in sample
    
    # Analysis script can work with status or method, but prefers both
    if not has_status and not has_method and not has_verification_method:
        result.warn("Missing fields for abstention detection: 'status', 'method', or 'verification_method'")
        result.warn("Analysis script will fall back to derivation.abstained > 0")
    elif not has_status:
        result.warn("Missing 'status' field - analysis script will use method-based detection")
    elif not has_method and not has_verification_method:
        result.warn("Missing 'method'/'verification_method' field - analysis script will use status-based detection")
    
    # Check if abstention field exists (preferred)
    if has_abstention:
        result.warnings.append("Found 'abstention' field (preferred for analysis)")
    else:
        result.warn("Missing 'abstention' field - analysis script will compute from status/method")
    
    # Check data types
    if "cycle" in sample:
        if not isinstance(sample["cycle"], int):
            result.fail(f"'cycle' field must be int, got {type(sample['cycle'])}")
    
    if "abstention" in sample:
        abstention_type = type(sample["abstention"])
        if abstention_type not in (bool, int):
            result.warn(f"'abstention' field should be bool/int, got {abstention_type}")
    
    return result


def check_analysis_script_compatibility() -> QAResult:
    """Check if analysis script can load and process the logs."""
    result = QAResult("Analysis Script Compatibility")
    
    if not ANALYSIS_SCRIPT.exists():
        result.fail(f"Analysis script not found: {ANALYSIS_SCRIPT}")
        return result
    
    # Try to import and use the analysis script's load_logs function
    try:
        # We'll simulate what the analysis script does
        baseline_path = BASELINE_WIDE
        rfl_path = RFL_WIDE
        
        if not baseline_path.exists():
            result.warn(f"Baseline log not found: {baseline_path}")
            return result
        
        if not rfl_path.exists():
            result.warn(f"RFL log not found: {rfl_path}")
            return result
        
        # Try loading with pandas (what analysis script uses)
        try:
            baseline_entries = load_jsonl(baseline_path)
            rfl_entries = load_jsonl(rfl_path)
            
            if not baseline_entries:
                result.fail("Baseline log is empty")
                return result
            
            if not rfl_entries:
                result.fail("RFL log is empty")
                return result
            
            # Convert to DataFrame (simulating analysis script)
            df_baseline = pd.DataFrame(baseline_entries)
            df_rfl = pd.DataFrame(rfl_entries)
            
            # Check for cycle column
            if 'cycle' not in df_baseline.columns:
                result.fail("Baseline DataFrame missing 'cycle' column")
            if 'cycle' not in df_rfl.columns:
                result.fail("RFL DataFrame missing 'cycle' column")
            
            # Try to compute is_abstention (simulating analysis script logic)
            def is_abstention(row):
                status = str(row.get("status", "")).lower()
                method = row.get("method") or row.get("verification_method", "")
                if status == "abstain":
                    return True
                return method == "lean-disabled"
            
            try:
                df_baseline['is_abstention'] = df_baseline.apply(is_abstention, axis=1).astype(int)
                df_rfl['is_abstention'] = df_rfl.apply(is_abstention, axis=1).astype(int)
                result.warnings.append("Successfully computed is_abstention from status/method")
            except Exception as e:
                result.warn(f"Could not compute is_abstention: {e}")
                # Fallback: use abstention field if available
                if 'abstention' in df_baseline.columns:
                    df_baseline['is_abstention'] = df_baseline['abstention'].astype(int)
                if 'abstention' in df_rfl.columns:
                    df_rfl['is_abstention'] = df_rfl['abstention'].astype(int)
                    result.warnings.append("Using 'abstention' field directly")
            
            # Try computing rolling metrics
            window_size = 100
            df_baseline['abstention_rate_rolling'] = df_baseline['is_abstention'].rolling(
                window=window_size, min_periods=1
            ).mean()
            df_rfl['abstention_rate_rolling'] = df_rfl['is_abstention'].rolling(
                window=window_size, min_periods=1
            ).mean()
            
            result.warnings.append("Successfully computed rolling abstention rates")
            result.warnings.append(f"Baseline entries: {len(df_baseline)}, RFL entries: {len(df_rfl)}")
            
        except Exception as e:
            result.fail(f"Failed to process logs with analysis script logic: {e}")
            return result
        
    except Exception as e:
        result.fail(f"Failed to test analysis script compatibility: {e}")
        return result
    
    return result


def check_dyno_chart_integrity() -> QAResult:
    """Check if dyno chart exists and appears valid."""
    result = QAResult("Dyno Chart Integrity")
    
    if not DYNO_CHART.exists():
        result.fail(f"Dyno chart not found: {DYNO_CHART}")
        result.warn("Expected location: artifacts/figures/rfl_dyno_chart.png")
        return result
    
    # Check file size (should be non-zero)
    file_size = DYNO_CHART.stat().st_size
    if file_size == 0:
        result.fail("Dyno chart file is empty (0 bytes)")
        return result
    
    if file_size < 1000:  # Less than 1KB is suspicious for a PNG
        result.warn(f"Dyno chart file is very small: {file_size} bytes")
    
    # Try to verify it's a valid PNG (basic check)
    try:
        with open(DYNO_CHART, 'rb') as f:
            header = f.read(8)
            if header[:8] != b'\x89PNG\r\n\x1a\n':
                result.warn("File does not appear to be a valid PNG (header check failed)")
            else:
                result.warnings.append("PNG header validation passed")
    except Exception as e:
        result.warn(f"Could not verify PNG format: {e}")
    
    result.warnings.append(f"File size: {file_size:,} bytes")
    
    # Note: We can't easily verify the visual content without image processing
    # The user should manually verify:
    # - Both lines are drawn
    # - Axes are labeled
    # - No obvious bugs (swapped labels, empty line)
    result.warnings.append("Manual verification needed: Check that both lines are drawn, axes labeled, no bugs")
    
    return result


def main():
    """Run all QA checks."""
    print("=" * 70)
    print("Dyno Chart QA Cross-Check (Cursor L)")
    print("=" * 70)
    print()
    
    results: List[QAResult] = []
    
    # 1. Log sanity checks
    print("1. Log Sanity Checks")
    print("-" * 70)
    results.append(check_log_sanity(BASELINE_WIDE, "Baseline Wide Slice"))
    print(results[-1])
    print()
    
    results.append(check_log_sanity(RFL_WIDE, "RFL Wide Slice"))
    print(results[-1])
    print()
    
    # 2. Schema compatibility
    print("2. Schema Compatibility Checks")
    print("-" * 70)
    results.append(check_schema_compatibility(BASELINE_WIDE, "Baseline Wide Slice"))
    print(results[-1])
    print()
    
    results.append(check_schema_compatibility(RFL_WIDE, "RFL Wide Slice"))
    print(results[-1])
    print()
    
    # 3. Analysis script compatibility
    print("3. Analysis Script Compatibility")
    print("-" * 70)
    results.append(check_analysis_script_compatibility())
    print(results[-1])
    print()
    
    # 4. Dyno Chart integrity
    print("4. Dyno Chart Integrity")
    print("-" * 70)
    results.append(check_dyno_chart_integrity())
    print(results[-1])
    print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    
    print(f"Checks passed: {passed}/{total}")
    print()
    
    failed = [r for r in results if not r.passed]
    if failed:
        print("FAILED CHECKS:")
        for r in failed:
            print(f"  - {r.name}")
            for issue in r.issues:
                print(f"    • {issue}")
        print()
    
    all_passed = all(r.passed for r in results)
    
    if all_passed:
        print("✓ All QA checks passed!")
        return 0
    else:
        print("✗ Some QA checks failed. Please review issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

