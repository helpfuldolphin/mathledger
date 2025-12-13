#!/usr/bin/env python3
"""
CAL-EXP-2 Run Verifier — Post-Run Invariant Check

Verifies that a CAL-EXP-2 run satisfies all SHADOW MODE invariants.
Advisory-only: does not gate CI unless explicitly configured.

Usage:
    python scripts/verify_cal_exp_2_run.py --run-dir results/cal_exp_2/<run_id>/

Exit codes:
    0 = PASS (all invariants satisfied)
    1 = FAIL (one or more invalidating conditions)

Source of Truth: docs/system_law/calibration/RUN_SHADOW_AUDIT_V0_1_CONTRACT.md
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Cross-shell preflight (advisory only, does not block)
try:
    from scripts.preflight_shell_env import print_preflight_advisory
    _PREFLIGHT_AVAILABLE = True
except ImportError:
    _PREFLIGHT_AVAILABLE = False


@dataclass
class CheckResult:
    """Result of a single invariant check."""
    name: str
    passed: bool
    expected: str
    actual: str
    invalidates: bool  # True = FAIL on violation, False = WARN only

    def __str__(self) -> str:
        status = "PASS" if self.passed else ("FAIL" if self.invalidates else "WARN")
        return f"[{status}] {self.name}: expected={self.expected}, actual={self.actual}"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        status = "PASS" if self.passed else ("FAIL" if self.invalidates else "WARN")
        return {
            "name": self.name,
            "status": status,
            "passed": self.passed,
            "expected": self.expected,
            "actual": self.actual,
            "invalidates": self.invalidates,
        }


@dataclass
class VerificationReport:
    """Aggregate verification report."""
    run_dir: str
    checks: List[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True if no invalidating checks failed."""
        return all(c.passed or not c.invalidates for c in self.checks)

    @property
    def fail_count(self) -> int:
        return sum(1 for c in self.checks if not c.passed and c.invalidates)

    @property
    def warn_count(self) -> int:
        return sum(1 for c in self.checks if not c.passed and not c.invalidates)

    @property
    def pass_count(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    def add(self, result: CheckResult) -> None:
        self.checks.append(result)

    def print_report(self) -> None:
        print(f"=== CAL-EXP-2 RUN VERIFICATION: {self.run_dir} ===")
        print()
        for check in self.checks:
            print(str(check))
        print()
        print(f"SUMMARY: {len(self.checks)} checks, {self.fail_count} FAIL, {self.warn_count} WARN")
        print(f"VERDICT: {'PASS' if self.passed else 'FAIL'}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize report to dictionary for JSON export."""
        return {
            "schema_version": "1.0.0",
            "verifier": "verify_cal_exp_2_run.py",
            "canonical_contract": "docs/system_law/calibration/RUN_SHADOW_AUDIT_V0_1_CONTRACT.md",
            "run_dir": self.run_dir,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "verdict": "PASS" if self.passed else "FAIL",
            "summary": {
                "total_checks": len(self.checks),
                "pass_count": self.pass_count,
                "fail_count": self.fail_count,
                "warn_count": self.warn_count,
            },
            "checks": [c.to_dict() for c in self.checks],
        }

    def write_json(self, path: Path) -> None:
        """Write report to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
            f.write("\n")


def load_json_safe(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Load JSON file, return (data, error)."""
    if not path.exists():
        return None, f"File not found: {path}"
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), None
    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {e}"
    except Exception as e:
        return None, f"Read error: {e}"


def validate_jsonl(path: Path) -> Tuple[int, Optional[str]]:
    """Validate JSONL file, return (line_count, first_error)."""
    if not path.exists():
        return 0, f"File not found: {path}"

    line_count = 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    json.loads(line)
                    line_count += 1
                except json.JSONDecodeError as e:
                    return line_count, f"Line {i}: {e}"
    except Exception as e:
        return line_count, f"Read error: {e}"

    return line_count, None


def check_divergence_actions(path: Path) -> Tuple[bool, str]:
    """Check all divergence actions are LOGGED_ONLY."""
    if not path.exists():
        return True, "no divergence log"  # Acceptable if no divergences

    violations = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    action = record.get("action", "LOGGED_ONLY")
                    if action != "LOGGED_ONLY":
                        violations.append(f"line {i}: action={action}")
                except json.JSONDecodeError:
                    pass  # Already caught by JSONL validation
    except Exception as e:
        return False, f"read error: {e}"

    if violations:
        return False, f"{len(violations)} violations: {violations[0]}"
    return True, "all LOGGED_ONLY"


def verify_run(run_dir: Path) -> VerificationReport:
    """Run all verification checks on a CAL-EXP-2 run directory."""
    report = VerificationReport(run_dir=str(run_dir))

    # =========================================================================
    # Required Files Check
    # =========================================================================
    required_files = [
        "run_config.json",
        "RUN_METADATA.json",
    ]

    for filename in required_files:
        path = run_dir / filename
        exists = path.exists()
        report.add(CheckResult(
            name=f"file_exists:{filename}",
            passed=exists,
            expected="exists",
            actual="exists" if exists else "missing",
            invalidates=True,
        ))

    # =========================================================================
    # Load run_config.json
    # =========================================================================
    config_path = run_dir / "run_config.json"
    config, config_err = load_json_safe(config_path)

    if config_err:
        report.add(CheckResult(
            name="run_config_parse",
            passed=False,
            expected="valid JSON",
            actual=config_err,
            invalidates=True,
        ))
        config = {}

    # =========================================================================
    # SHADOW Mode Marker (INVALIDATES)
    # =========================================================================
    mode = config.get("mode", "MISSING")
    report.add(CheckResult(
        name="mode",
        passed=(mode == "SHADOW"),
        expected="SHADOW",
        actual=mode,
        invalidates=True,
    ))

    # =========================================================================
    # Schema Version (INVALIDATES)
    # =========================================================================
    schema_version = config.get("schema_version", "MISSING")
    # Allow 1.x.x versions for forward compatibility
    valid_schema = schema_version.startswith("1.") if isinstance(schema_version, str) else False
    report.add(CheckResult(
        name="schema_version",
        passed=valid_schema,
        expected="1.x.x",
        actual=schema_version,
        invalidates=True,
    ))

    # =========================================================================
    # LR Bounds Check (INVALIDATES if outside [0, 1])
    # =========================================================================
    lr_overrides = config.get("twin_lr_overrides", {})
    for key in ["H", "rho", "tau", "beta"]:
        if key in lr_overrides:
            val = lr_overrides[key]
            in_bounds = isinstance(val, (int, float)) and 0 <= val <= 1
            report.add(CheckResult(
                name=f"lr_bounds:{key}",
                passed=in_bounds,
                expected="[0, 1]",
                actual=str(val),
                invalidates=True,
            ))

    # =========================================================================
    # Load RUN_METADATA.json
    # =========================================================================
    metadata_path = run_dir / "RUN_METADATA.json"
    metadata, metadata_err = load_json_safe(metadata_path)

    if metadata_err:
        report.add(CheckResult(
            name="metadata_parse",
            passed=False,
            expected="valid JSON",
            actual=metadata_err,
            invalidates=True,
        ))
        metadata = {}

    # =========================================================================
    # Enforcement Flag (INVALIDATES)
    # =========================================================================
    enforcement = metadata.get("enforcement", False)
    report.add(CheckResult(
        name="enforcement",
        passed=(enforcement is False),
        expected="false",
        actual=str(enforcement).lower(),
        invalidates=True,
    ))

    # =========================================================================
    # Status Check (INVALIDATES on blocking status)
    # =========================================================================
    status = metadata.get("status", "unknown")
    blocking_statuses = {"blocked", "failed", "enforced", "rejected", "aborted", "stopped"}
    is_blocking = status.lower() in blocking_statuses
    report.add(CheckResult(
        name="status",
        passed=(not is_blocking),
        expected="non-blocking",
        actual=status,
        invalidates=True,
    ))

    # =========================================================================
    # Cycles Completed (INVALIDATES if < 90% of requested)
    # =========================================================================
    cycles_completed = metadata.get("cycles_completed", 0)
    cycles_requested = metadata.get("total_cycles_requested", 0)

    if cycles_requested > 0:
        completion_ratio = cycles_completed / cycles_requested
        threshold_met = completion_ratio >= 0.9
        report.add(CheckResult(
            name="cycles_completed",
            passed=threshold_met,
            expected=f">= {int(cycles_requested * 0.9)} (90%)",
            actual=f"{cycles_completed}/{cycles_requested} ({completion_ratio:.1%})",
            invalidates=True,
        ))

    # =========================================================================
    # JSONL Validation (INVALIDATES on parse error)
    # =========================================================================
    jsonl_files = [
        "divergence_log.jsonl",
        "real_cycles.jsonl",
        "twin_predictions.jsonl",
    ]

    for filename in jsonl_files:
        path = run_dir / filename
        if path.exists():
            count, err = validate_jsonl(path)
            report.add(CheckResult(
                name=f"jsonl_valid:{filename}",
                passed=(err is None),
                expected="valid JSONL",
                actual=f"{count} lines" if err is None else err,
                invalidates=True,
            ))

    # =========================================================================
    # Divergence Actions (INVALIDATES if not LOGGED_ONLY)
    # =========================================================================
    div_log = run_dir / "divergence_log.jsonl"
    if div_log.exists():
        actions_ok, actions_msg = check_divergence_actions(div_log)
        report.add(CheckResult(
            name="divergence_actions",
            passed=actions_ok,
            expected="LOGGED_ONLY",
            actual=actions_msg,
            invalidates=True,
        ))

    # =========================================================================
    # Advisory Checks (WARN only, do not invalidate)
    # =========================================================================

    # Seed check (WARN if not canonical)
    params = config.get("parameters", {})
    seed = params.get("seed", config.get("seed"))
    if seed is not None:
        report.add(CheckResult(
            name="seed_canonical",
            passed=(seed == 42),
            expected="42 (CAL-EXP-2 canonical)",
            actual=str(seed),
            invalidates=False,  # WARN only
        ))

    # LR drift check (WARN if different from canonical)
    canonical_lrs = {"H": 0.20, "rho": 0.15, "tau": 0.02, "beta": 0.12}
    for key, expected_val in canonical_lrs.items():
        if key in lr_overrides:
            actual_val = lr_overrides[key]
            report.add(CheckResult(
                name=f"lr_canonical:{key}",
                passed=(abs(actual_val - expected_val) < 0.001),
                expected=str(expected_val),
                actual=str(actual_val),
                invalidates=False,  # WARN only
            ))

    return report


def main() -> int:
    # Cross-shell preflight advisory (non-blocking)
    if _PREFLIGHT_AVAILABLE:
        print_preflight_advisory()

    parser = argparse.ArgumentParser(
        description="Verify CAL-EXP-2 run satisfies SHADOW MODE invariants",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to run directory (e.g., results/cal_exp_2/p4_20251212_103832/)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print PASS/FAIL verdict",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=None,
        help="Path to write JSON verification report (default: <run_dir>/cal_exp_2_verification_report.json)",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Disable JSON report output",
    )

    args = parser.parse_args()

    if not args.run_dir.exists():
        print(f"ERROR: Run directory not found: {args.run_dir}")
        return 1

    report = verify_run(args.run_dir)

    # Determine report output path
    if not args.no_report:
        report_path = args.output_report
        if report_path is None:
            report_path = args.run_dir / "cal_exp_2_verification_report.json"
        try:
            report.write_json(report_path)
            if not args.quiet:
                print(f"Report written to: {report_path}")
        except Exception as e:
            # Report writing failure is non-fatal (advisory only)
            print(f"WARNING: Could not write report: {e}")

    if args.quiet:
        print("PASS" if report.passed else "FAIL")
    else:
        report.print_report()

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
