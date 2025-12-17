#!/usr/bin/env python3
"""
Golden Manifest Comparison — CAL-EXP-3 Determinism Verifier

Compares a CAL-EXP-3 run against a golden/reference manifest to confirm:
    "This evidence pack matches a known-good CAL-EXP-3 run."

Usage:
    python scripts/verify_against_golden.py --run-dir results/cal_exp_3/<run_id>/
    python scripts/verify_against_golden.py --run-dir results/cal_exp_3/<run_id>/ --manifest results/golden/manifest_seed42.json

Exit codes:
    0 = MATCH (run matches golden reference)
    1 = MISMATCH (one or more invalidating differences)
    2 = ERROR (missing files, invalid manifest, etc.)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# Hash Functions (must match manifest generation)
# =============================================================================

def sha256_file(path: Path) -> str:
    """Hash file contents as raw bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def sha256_jsonl_canonical(path: Path) -> str:
    """Hash JSONL content with canonical JSON serialization per line."""
    h = hashlib.sha256()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                canonical = json.dumps(obj, separators=(",", ":"), sort_keys=True)
                h.update(canonical.encode("utf-8"))
                h.update(b"\n")
    return h.hexdigest()


def sha256_json_canonical(path: Path) -> str:
    """Hash JSON content with canonical serialization."""
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    canonical = json.dumps(obj, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


HASH_METHODS = {
    "sha256_file": sha256_file,
    "sha256_jsonl_canonical": sha256_jsonl_canonical,
    "sha256_json_canonical": sha256_json_canonical,
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ComparisonResult:
    """Result of comparing a single artifact."""
    artifact: str
    passed: bool
    expected_hash: str
    actual_hash: str
    mismatch_interpretation: str
    invalidates: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact": self.artifact,
            "status": "MATCH" if self.passed else "MISMATCH",
            "expected": self.expected_hash[:16] + "...",
            "actual": self.actual_hash[:16] + "..." if self.actual_hash else "MISSING",
            "invalidates": self.invalidates,
            "interpretation": self.mismatch_interpretation if not self.passed else None,
        }


@dataclass
class InvariantResult:
    """Result of checking a required invariant."""
    name: str
    passed: bool
    expected: Any
    actual: Any

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": "MATCH" if self.passed else "MISMATCH",
            "expected": self.expected,
            "actual": self.actual,
        }


@dataclass
class GoldenComparisonReport:
    """Full comparison report against golden manifest."""
    run_dir: str
    manifest_path: str
    artifact_results: List[ComparisonResult] = field(default_factory=list)
    invariant_results: List[InvariantResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True if all invalidating checks passed and no errors."""
        if self.errors:
            return False
        artifact_ok = all(r.passed or not r.invalidates for r in self.artifact_results)
        invariant_ok = all(r.passed for r in self.invariant_results)
        return artifact_ok and invariant_ok

    @property
    def mismatch_count(self) -> int:
        return sum(1 for r in self.artifact_results if not r.passed and r.invalidates)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "1.0.0",
            "verifier": "verify_against_golden.py",
            "run_dir": self.run_dir,
            "manifest_path": self.manifest_path,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "verdict": "MATCH" if self.passed else "MISMATCH",
            "summary": {
                "artifact_checks": len(self.artifact_results),
                "artifact_mismatches": self.mismatch_count,
                "invariant_checks": len(self.invariant_results),
                "invariant_mismatches": sum(1 for r in self.invariant_results if not r.passed),
                "errors": len(self.errors),
            },
            "artifact_results": [r.to_dict() for r in self.artifact_results],
            "invariant_results": [r.to_dict() for r in self.invariant_results],
            "errors": self.errors,
        }


# =============================================================================
# Comparison Logic
# =============================================================================

def load_manifest(manifest_path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Load and validate golden manifest."""
    if not manifest_path.exists():
        return None, f"Manifest not found: {manifest_path}"
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        # Basic validation
        if "deterministic_artifacts" not in manifest:
            return None, "Manifest missing 'deterministic_artifacts'"
        if "schema_version" not in manifest:
            return None, "Manifest missing 'schema_version'"
        return manifest, None
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON in manifest: {e}"
    except Exception as e:
        return None, f"Error loading manifest: {e}"


def compare_artifact(
    run_dir: Path,
    artifact_path: str,
    expected_hash: str,
    hash_method: str,
    mismatch_interpretation: str,
    must_match: bool,
) -> ComparisonResult:
    """Compare a single artifact against expected hash."""
    full_path = run_dir / artifact_path

    if not full_path.exists():
        return ComparisonResult(
            artifact=artifact_path,
            passed=False,
            expected_hash=expected_hash,
            actual_hash="",
            mismatch_interpretation=f"MISSING: {artifact_path}",
            invalidates=must_match,
        )

    hasher = HASH_METHODS.get(hash_method)
    if not hasher:
        return ComparisonResult(
            artifact=artifact_path,
            passed=False,
            expected_hash=expected_hash,
            actual_hash="",
            mismatch_interpretation=f"Unknown hash method: {hash_method}",
            invalidates=must_match,
        )

    try:
        actual_hash = hasher(full_path)
    except Exception as e:
        return ComparisonResult(
            artifact=artifact_path,
            passed=False,
            expected_hash=expected_hash,
            actual_hash="",
            mismatch_interpretation=f"Hash error: {e}",
            invalidates=must_match,
        )

    passed = actual_hash == expected_hash
    return ComparisonResult(
        artifact=artifact_path,
        passed=passed,
        expected_hash=expected_hash,
        actual_hash=actual_hash,
        mismatch_interpretation=mismatch_interpretation if not passed else "",
        invalidates=must_match,
    )


def check_invariant(
    run_dir: Path,
    name: str,
    expected_value: Any,
) -> InvariantResult:
    """Check a required invariant from run metadata."""
    # Load metadata
    metadata_path = run_dir / "RUN_METADATA.json"
    config_path = run_dir / "run_config.json"

    actual_value = None

    # Try metadata first
    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            if name in metadata:
                actual_value = metadata[name]
        except Exception:
            pass

    # Fall back to config
    if actual_value is None and config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            if name in config:
                actual_value = config[name]
        except Exception:
            pass

    # Special handling for corpus_hash
    if actual_value is None and name == "corpus_hash":
        corpus_path = run_dir / "validity" / "corpus_manifest.json"
        if corpus_path.exists():
            try:
                with open(corpus_path, "r", encoding="utf-8") as f:
                    corpus = json.load(f)
                actual_value = corpus.get("corpus_hash", corpus.get("hash"))
            except Exception:
                pass

    # Compare with tolerance for floats
    if isinstance(expected_value, float) and isinstance(actual_value, float):
        passed = abs(expected_value - actual_value) < 1e-10
    else:
        passed = actual_value == expected_value

    return InvariantResult(
        name=name,
        passed=passed,
        expected=expected_value,
        actual=actual_value,
    )


def compare_against_golden(
    run_dir: Path,
    manifest_path: Path,
) -> GoldenComparisonReport:
    """Compare a CAL-EXP-3 run against golden manifest."""
    report = GoldenComparisonReport(
        run_dir=str(run_dir),
        manifest_path=str(manifest_path),
    )

    # Load manifest
    manifest, err = load_manifest(manifest_path)
    if err:
        report.errors.append(err)
        return report

    # Check run directory exists
    if not run_dir.exists():
        report.errors.append(f"Run directory not found: {run_dir}")
        return report

    # Compare deterministic artifacts
    artifacts = manifest.get("deterministic_artifacts", {})
    for artifact_path, spec in artifacts.items():
        result = compare_artifact(
            run_dir=run_dir,
            artifact_path=artifact_path,
            expected_hash=spec["sha256_content"],
            hash_method=spec.get("hash_method", "sha256_file"),
            mismatch_interpretation=spec.get("mismatch_interpretation", "Hash mismatch"),
            must_match=spec.get("must_match", True),
        )
        report.artifact_results.append(result)

    # Check required invariants
    invariants = manifest.get("required_invariants", {})
    for name, expected_value in invariants.items():
        result = check_invariant(run_dir, name, expected_value)
        report.invariant_results.append(result)

    return report


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare CAL-EXP-3 run against golden manifest",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to CAL-EXP-3 run directory",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/golden/manifest_seed42.json"),
        help="Path to golden manifest (default: results/golden/manifest_seed42.json)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print MATCH/MISMATCH verdict",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output full JSON report",
    )

    args = parser.parse_args()

    report = compare_against_golden(args.run_dir, args.manifest)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    elif args.quiet:
        print("MATCH" if report.passed else "MISMATCH")
    else:
        # Human-readable output
        print("=" * 60)
        print("  CAL-EXP-3 GOLDEN MANIFEST COMPARISON")
        print("=" * 60)
        print(f"Run:      {report.run_dir}")
        print(f"Manifest: {report.manifest_path}")
        print()

        if report.errors:
            print("ERRORS:")
            for err in report.errors:
                print(f"  [ERROR] {err}")
            print()

        print("ARTIFACT COMPARISON:")
        for r in report.artifact_results:
            status = "MATCH" if r.passed else "MISMATCH"
            symbol = "OK" if r.passed else "X "
            print(f"  [{symbol}] {r.artifact}: {status}")
            if not r.passed:
                print(f"       Expected: {r.expected_hash[:32]}...")
                print(f"       Actual:   {r.actual_hash[:32] if r.actual_hash else 'MISSING'}...")
                print(f"       -> {r.mismatch_interpretation}")
        print()

        print("INVARIANT COMPARISON:")
        for r in report.invariant_results:
            status = "MATCH" if r.passed else "MISMATCH"
            symbol = "OK" if r.passed else "X "
            print(f"  [{symbol}] {r.name}: {status}")
            if not r.passed:
                print(f"       Expected: {r.expected}")
                print(f"       Actual:   {r.actual}")
        print()

        print("=" * 60)
        print(f"VERDICT: {'MATCH' if report.passed else 'MISMATCH'}")
        print("=" * 60)

        if not report.passed:
            print()
            print("MISMATCH INTERPRETATION TABLE:")
            print("-" * 60)
            for r in report.artifact_results:
                if not r.passed and r.invalidates:
                    print(f"  {r.artifact}:")
                    print(f"    -> {r.mismatch_interpretation}")
            for r in report.invariant_results:
                if not r.passed:
                    print(f"  {r.name}:")
                    print(f"    -> INVALIDATE: Required invariant mismatch")

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
