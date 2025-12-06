#!/usr/bin/env python3
"""
U2 Uplift Artifact Auditor (QA Tool)
====================================

Audits U2 uplift experiment artifacts for consistency:
- baseline.jsonl
- rfl.jsonl
- experiment_manifest.json

This tool is READ-ONLY and makes NO interpretive claims about uplift.
It only reports structural integrity and raw statistics.

Usage:
    uv run python experiments/audit_uplift_u2.py <results_dir>

Example:
    uv run python experiments/audit_uplift_u2.py results/uplift_u2/slice_uplift_sparse
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Required fields in each JSONL record for schema validation
REQUIRED_FIELDS = [
    "cycle",
    "slice_name",
    "mode",
    "status",
    "abstention",
    "roots",
]

# Fields expected in roots object
REQUIRED_ROOT_FIELDS = ["h_t"]

# Fields expected in derivation object (if present)
DERIVATION_FIELDS = ["verified"]


def load_jsonl(path: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Load JSONL file and validate line-by-line.

    Returns:
        (records, errors): List of valid records and list of error messages
    """
    records = []
    errors = []

    if not path.exists():
        return records, [f"File does not exist: {path}"]

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: JSON parse error - {e}")

    return records, errors


def check_file_existence(path: Path) -> Dict[str, Any]:
    """Check if file exists and is non-empty."""
    result = {
        "exists": path.exists(),
        "path": str(path),
        "size_bytes": 0,
        "is_empty": True,
    }

    if path.exists():
        result["size_bytes"] = path.stat().st_size
        result["is_empty"] = result["size_bytes"] == 0

    return result


def check_cycle_continuity(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Check that cycles are contiguous from 0 to T-1.

    Returns:
        Dictionary with continuity check results
    """
    if not records:
        return {
            "total_cycles": 0,
            "is_contiguous": False,
            "missing_cycles": [],
            "duplicate_cycles": [],
            "error": "No records to check",
        }

    cycles = [r.get("cycle") for r in records if "cycle" in r]

    if not cycles:
        return {
            "total_cycles": 0,
            "is_contiguous": False,
            "missing_cycles": [],
            "duplicate_cycles": [],
            "error": "No 'cycle' field found in records",
        }

    cycle_set = set(cycles)
    total_cycles = len(records)
    expected_cycles = set(range(total_cycles))

    missing = sorted(list(expected_cycles - cycle_set))
    # Find duplicates: cycles that appear more than once
    seen = set()
    duplicates = []
    for c in cycles:
        if c in seen:
            duplicates.append(c)
        seen.add(c)

    is_contiguous = len(missing) == 0 and len(duplicates) == 0

    return {
        "total_cycles": total_cycles,
        "min_cycle": min(cycles),
        "max_cycle": max(cycles),
        "is_contiguous": is_contiguous,
        "missing_cycles": missing[:20],  # First 20 for brevity
        "missing_count": len(missing),
        "duplicate_cycles": list(set(duplicates))[:10],
        "duplicate_count": len(duplicates),
    }


def check_schema(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Check that required fields are present in all records.

    Returns:
        Dictionary with schema validation results
    """
    if not records:
        return {
            "is_valid": False,
            "missing_fields": [],
            "sample_fields": [],
            "error": "No records to validate",
        }

    missing_by_record = []

    for idx, record in enumerate(records):
        missing = []

        # Check top-level required fields
        for field in REQUIRED_FIELDS:
            if field not in record:
                missing.append(field)

        # Check nested roots fields
        roots = record.get("roots", {})
        if isinstance(roots, dict):
            for field in REQUIRED_ROOT_FIELDS:
                if field not in roots:
                    missing.append(f"roots.{field}")
        else:
            missing.append("roots (not a dict)")

        # Check derivation.verified if derivation exists
        derivation = record.get("derivation")
        if derivation is not None:
            if isinstance(derivation, dict):
                for field in DERIVATION_FIELDS:
                    if field not in derivation:
                        missing.append(f"derivation.{field}")
            else:
                missing.append("derivation (not a dict)")

        if missing:
            missing_by_record.append((idx, missing))

    # Sample fields from first record
    sample_fields = sorted(records[0].keys()) if records else []

    is_valid = len(missing_by_record) == 0

    return {
        "is_valid": is_valid,
        "total_records": len(records),
        "records_with_missing_fields": len(missing_by_record),
        "sample_missing": missing_by_record[:5],  # First 5 issues
        "sample_fields": sample_fields,
    }


def check_slice_alignment(
    records: List[Dict[str, Any]], expected_slice: Optional[str]
) -> Dict[str, Any]:
    """
    Check that slice_name in records matches the expected slice.

    Returns:
        Dictionary with slice alignment results
    """
    if not records:
        return {
            "is_aligned": False,
            "unique_slices": [],
            "error": "No records to check",
        }

    slices = [r.get("slice_name") for r in records if "slice_name" in r]
    unique_slices = list(set(slices))

    is_aligned = True
    mismatches = []

    if expected_slice:
        for idx, record in enumerate(records):
            record_slice = record.get("slice_name")
            if record_slice != expected_slice:
                is_aligned = False
                mismatches.append((idx, record_slice))

    # Also check for consistency within file
    if len(unique_slices) > 1:
        is_aligned = False

    return {
        "is_aligned": is_aligned,
        "unique_slices": unique_slices,
        "expected_slice": expected_slice,
        "mismatches": mismatches[:5],  # First 5 mismatches
        "mismatch_count": len(mismatches),
    }


def load_manifest(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Load experiment_manifest.json.

    Returns:
        (manifest_dict, error_message)
    """
    if not path.exists():
        return None, f"Manifest not found: {path}"

    try:
        with open(path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        return manifest, None
    except json.JSONDecodeError as e:
        return None, f"Manifest JSON parse error: {e}"


def check_manifest_consistency(
    manifest: Optional[Dict[str, Any]],
    baseline_cycles: int,
    rfl_cycles: int,
) -> Dict[str, Any]:
    """
    Check that manifest cycles value matches the actual cycle counts.

    Returns:
        Dictionary with manifest consistency results
    """
    if manifest is None:
        return {
            "is_consistent": False,
            "error": "No manifest to validate",
        }

    manifest_cycles = manifest.get("cycles")
    manifest_slice = manifest.get("slice")

    baseline_match = manifest_cycles == baseline_cycles if manifest_cycles else False
    rfl_match = manifest_cycles == rfl_cycles if manifest_cycles else False

    return {
        "is_consistent": baseline_match and rfl_match,
        "manifest_cycles": manifest_cycles,
        "baseline_cycles": baseline_cycles,
        "rfl_cycles": rfl_cycles,
        "baseline_matches": baseline_match,
        "rfl_matches": rfl_match,
        "manifest_slice": manifest_slice,
    }


def compute_raw_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute raw statistics per cycle (no interpretive claims).

    Returns:
        Dictionary with raw statistics labeled clearly
    """
    if not records:
        return {
            "total_records": 0,
            "raw_success_count": 0,
            "raw_abstention_count": 0,
            "note": "No data for statistics",
        }

    success_count = 0
    abstention_count = 0

    for record in records:
        # Check multiple possible success indicators
        if record.get("success") is True:
            success_count += 1
        elif record.get("status") == "success":
            success_count += 1
        elif record.get("proof_found") is True:
            success_count += 1

        # Check abstention
        if record.get("abstention") is True:
            abstention_count += 1
        elif record.get("status") == "abstain":
            abstention_count += 1

    total = len(records)

    return {
        "total_records": total,
        "raw_success_count": success_count,
        "raw_abstention_count": abstention_count,
        "raw_success_rate": success_count / total if total > 0 else 0.0,
        "raw_abstention_rate": abstention_count / total if total > 0 else 0.0,
        "note": "RAW STATS ONLY - no interpretive claims",
    }


def print_report(
    results_dir: Path,
    baseline_result: Dict[str, Any],
    rfl_result: Dict[str, Any],
    manifest_result: Dict[str, Any],
) -> bool:
    """
    Print markdown-style QA report to stdout.

    Returns:
        True if all checks pass, False otherwise
    """
    all_ok = True

    print("=" * 70)
    print("U2 UPLIFT ARTIFACT QA REPORT")
    print("=" * 70)
    print(f"Directory: {results_dir}")
    print()

    # --- Baseline Summary ---
    print("## Baseline Log (baseline.jsonl)")
    print("-" * 40)

    b_file = baseline_result["file_check"]
    if not b_file["exists"]:
        print(f"  ❌ File: MISSING")
        all_ok = False
    elif b_file["is_empty"]:
        print(f"  ❌ File: EXISTS but EMPTY (0 bytes)")
        all_ok = False
    else:
        print(f"  ✓ File: exists ({b_file['size_bytes']} bytes)")

    if baseline_result["jsonl_errors"]:
        print(f"  ❌ JSONL validation: {len(baseline_result['jsonl_errors'])} errors")
        for err in baseline_result["jsonl_errors"][:3]:
            print(f"      - {err}")
        all_ok = False
    elif b_file["exists"] and not b_file["is_empty"]:
        print(f"  ✓ JSONL validation: OK")

    b_cycles = baseline_result["cycle_check"]
    if b_cycles.get("error"):
        print(f"  ❌ Cycles: {b_cycles['error']}")
        all_ok = False
    elif not b_cycles["is_contiguous"]:
        print(f"  ❌ Cycles: NOT contiguous ({b_cycles['total_cycles']} records)")
        if b_cycles["missing_count"] > 0:
            print(f"      Missing: {b_cycles['missing_count']} cycles")
        if b_cycles["duplicate_count"] > 0:
            print(f"      Duplicates: {b_cycles['duplicate_count']} cycles")
        all_ok = False
    else:
        print(
            f"  ✓ Cycles: {b_cycles['total_cycles']} cycles, contiguous (0 to {b_cycles['max_cycle']})"
        )

    b_schema = baseline_result["schema_check"]
    if b_schema.get("error"):
        print(f"  ❌ Schema: {b_schema['error']}")
        all_ok = False
    elif not b_schema["is_valid"]:
        print(
            f"  ❌ Schema: {b_schema['records_with_missing_fields']} records missing fields"
        )
        all_ok = False
    else:
        print(f"  ✓ Schema: OK")

    b_stats = baseline_result["raw_stats"]
    if b_stats["total_records"] > 0:
        print(
            f"  [RAW STATS] {b_stats['total_records']} cycles, "
            f"success={b_stats['raw_success_count']}, abstention={b_stats['raw_abstention_count']}"
        )

    print()

    # --- RFL Summary ---
    print("## RFL Log (rfl.jsonl)")
    print("-" * 40)

    r_file = rfl_result["file_check"]
    if not r_file["exists"]:
        print(f"  ❌ File: MISSING")
        all_ok = False
    elif r_file["is_empty"]:
        print(f"  ❌ File: EXISTS but EMPTY (0 bytes)")
        all_ok = False
    else:
        print(f"  ✓ File: exists ({r_file['size_bytes']} bytes)")

    if rfl_result["jsonl_errors"]:
        print(f"  ❌ JSONL validation: {len(rfl_result['jsonl_errors'])} errors")
        for err in rfl_result["jsonl_errors"][:3]:
            print(f"      - {err}")
        all_ok = False
    elif r_file["exists"] and not r_file["is_empty"]:
        print(f"  ✓ JSONL validation: OK")

    r_cycles = rfl_result["cycle_check"]
    if r_cycles.get("error"):
        print(f"  ❌ Cycles: {r_cycles['error']}")
        all_ok = False
    elif not r_cycles["is_contiguous"]:
        print(f"  ❌ Cycles: NOT contiguous ({r_cycles['total_cycles']} records)")
        if r_cycles["missing_count"] > 0:
            print(f"      Missing: {r_cycles['missing_count']} cycles")
        if r_cycles["duplicate_count"] > 0:
            print(f"      Duplicates: {r_cycles['duplicate_count']} cycles")
        all_ok = False
    else:
        print(
            f"  ✓ Cycles: {r_cycles['total_cycles']} cycles, contiguous (0 to {r_cycles['max_cycle']})"
        )

    r_schema = rfl_result["schema_check"]
    if r_schema.get("error"):
        print(f"  ❌ Schema: {r_schema['error']}")
        all_ok = False
    elif not r_schema["is_valid"]:
        print(
            f"  ❌ Schema: {r_schema['records_with_missing_fields']} records missing fields"
        )
        all_ok = False
    else:
        print(f"  ✓ Schema: OK")

    r_stats = rfl_result["raw_stats"]
    if r_stats["total_records"] > 0:
        print(
            f"  [RAW STATS] {r_stats['total_records']} cycles, "
            f"success={r_stats['raw_success_count']}, abstention={r_stats['raw_abstention_count']}"
        )

    print()

    # --- Manifest Summary ---
    print("## Manifest (experiment_manifest.json)")
    print("-" * 40)

    m_check = manifest_result["manifest_check"]
    if manifest_result["manifest_error"]:
        print(f"  ❌ Manifest: {manifest_result['manifest_error']}")
        all_ok = False
    else:
        print(f"  ✓ Manifest: exists and valid JSON")

    if m_check.get("error"):
        print(f"  ❌ Consistency: {m_check['error']}")
    elif not m_check["is_consistent"]:
        print(f"  ❌ Consistency: cycle counts do not match manifest")
        print(f"      Manifest cycles: {m_check['manifest_cycles']}")
        print(f"      Baseline cycles: {m_check['baseline_cycles']}")
        print(f"      RFL cycles: {m_check['rfl_cycles']}")
        all_ok = False
    else:
        print(
            f"  ✓ Consistency: manifest cycles ({m_check['manifest_cycles']}) matches both logs"
        )

    if m_check.get("manifest_slice"):
        print(f"  Slice: {m_check['manifest_slice']}")

    print()

    # --- Slice Alignment ---
    print("## Slice Alignment")
    print("-" * 40)

    b_slice = baseline_result["slice_check"]
    r_slice = rfl_result["slice_check"]

    b_slices = b_slice.get("unique_slices", [])
    r_slices = r_slice.get("unique_slices", [])
    b_expected = b_slice.get("expected_slice")
    r_expected = r_slice.get("expected_slice")

    if b_slices and r_slices:
        if b_slices == r_slices and len(b_slices) == 1:
            actual_slice = b_slices[0]
            # Check if there's an expected slice from CLI or manifest
            expected = b_expected or r_expected
            if expected and expected != actual_slice:
                print(f"  ❌ Slice mismatch with expected: {expected}")
                print(f"      Actual slice in logs: {actual_slice}")
                all_ok = False
            else:
                print(f"  ✓ Both logs use same slice: {actual_slice}")
        else:
            print(f"  ⚠ Baseline slices: {b_slices}")
            print(f"  ⚠ RFL slices: {r_slices}")
            if b_slices != r_slices:
                print(f"  ❌ Slice mismatch between baseline and RFL")
                all_ok = False
    elif b_slice.get("error") or r_slice.get("error"):
        print(f"  ⚠ Could not determine slice alignment")

    print()

    # --- Final Verdict ---
    print("=" * 70)
    print("## VERDICT")
    print("=" * 70)

    if all_ok:
        print("  ✅ Ready for analysis: YES")
        print()
        print(
            "  All structural checks pass. Data integrity confirmed for QA purposes."
        )
    else:
        print("  ❌ Ready for analysis: NO")
        print()
        print("  Issues found - see details above.")

    print()
    print("NOTE: This report contains RAW STATS only. No claims about uplift,")
    print("significance, or experimental outcomes are made by this tool.")
    print("=" * 70)

    return all_ok


def audit_directory(results_dir: Path, expected_slice: Optional[str] = None) -> bool:
    """
    Main audit function for a U2 uplift results directory.

    Returns:
        True if all checks pass, False otherwise
    """
    baseline_path = results_dir / "baseline.jsonl"
    rfl_path = results_dir / "rfl.jsonl"
    manifest_path = results_dir / "experiment_manifest.json"

    # --- Audit Baseline ---
    baseline_file_check = check_file_existence(baseline_path)
    baseline_records, baseline_errors = load_jsonl(baseline_path)
    baseline_cycle_check = check_cycle_continuity(baseline_records)
    baseline_schema_check = check_schema(baseline_records)
    baseline_slice_check = check_slice_alignment(baseline_records, expected_slice)
    baseline_stats = compute_raw_stats(baseline_records)

    baseline_result = {
        "file_check": baseline_file_check,
        "jsonl_errors": baseline_errors,
        "cycle_check": baseline_cycle_check,
        "schema_check": baseline_schema_check,
        "slice_check": baseline_slice_check,
        "raw_stats": baseline_stats,
    }

    # --- Audit RFL ---
    rfl_file_check = check_file_existence(rfl_path)
    rfl_records, rfl_errors = load_jsonl(rfl_path)
    rfl_cycle_check = check_cycle_continuity(rfl_records)
    rfl_schema_check = check_schema(rfl_records)
    rfl_slice_check = check_slice_alignment(rfl_records, expected_slice)
    rfl_stats = compute_raw_stats(rfl_records)

    rfl_result = {
        "file_check": rfl_file_check,
        "jsonl_errors": rfl_errors,
        "cycle_check": rfl_cycle_check,
        "schema_check": rfl_schema_check,
        "slice_check": rfl_slice_check,
        "raw_stats": rfl_stats,
    }

    # --- Audit Manifest ---
    manifest, manifest_error = load_manifest(manifest_path)

    # If manifest exists, get expected slice from it
    if manifest and expected_slice is None:
        expected_slice = manifest.get("slice")
        # Re-check slice alignment with manifest slice
        baseline_slice_check = check_slice_alignment(baseline_records, expected_slice)
        rfl_slice_check = check_slice_alignment(rfl_records, expected_slice)
        baseline_result["slice_check"] = baseline_slice_check
        rfl_result["slice_check"] = rfl_slice_check

    manifest_consistency = check_manifest_consistency(
        manifest,
        baseline_cycle_check.get("total_cycles", 0),
        rfl_cycle_check.get("total_cycles", 0),
    )

    manifest_result = {
        "manifest": manifest,
        "manifest_error": manifest_error,
        "manifest_check": manifest_consistency,
    }

    # --- Print Report ---
    return print_report(results_dir, baseline_result, rfl_result, manifest_result)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="U2 Uplift Artifact Auditor - QA tool for consistency checks",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
This tool is READ-ONLY and makes NO interpretive claims about uplift.
It only reports structural integrity and raw statistics.

Example:
    uv run python experiments/audit_uplift_u2.py results/uplift_u2/slice_uplift_sparse
        """,
    )
    parser.add_argument(
        "results_dir",
        type=str,
        help="Path to U2 results directory containing baseline.jsonl, rfl.jsonl, and experiment_manifest.json",
    )
    parser.add_argument(
        "--slice",
        type=str,
        default=None,
        help="Expected slice name to verify alignment (optional, will be read from manifest if not provided)",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"ERROR: Results directory does not exist: {results_dir}", file=sys.stderr)
        sys.exit(1)

    if not results_dir.is_dir():
        print(f"ERROR: Path is not a directory: {results_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        success = audit_directory(results_dir, expected_slice=args.slice)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"ERROR: Audit failed: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
