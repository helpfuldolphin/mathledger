#!/usr/bin/env python3
"""
CAL-EXP-4 Run Verifier — Variance Stress Test Comparability Check

Verifies that a CAL-EXP-4 run satisfies temporal and variance comparability
requirements per the authoritative schemas. Fail-close on incompatible profiles.

SHADOW MODE: Advisory-only. Does not gate CI merges.

Usage:
    python scripts/verify_cal_exp_4_run.py --run-dir results/cal_exp_4/<run_id>/

Exit codes:
    0 = PASS (all comparability checks satisfied)
    1 = FAIL (one or more fail-close conditions triggered)

Authoritative Sources:
    - schemas/cal_exp_4/temporal_structure_audit.schema.json
    - schemas/cal_exp_4/variance_profile_audit.schema.json
    - docs/system_law/calibration/CAL_EXP_4_VERIFIER_PLAN.md

================================================================================
F5.x FAILURE CODES (Variance Stress Test Failures)
================================================================================

F5.1: Temporal structure incompatible (temporal_structure_pass=false)
F5.2: Variance ratio out of bounds (variance_ratio_acceptable=false)
F5.3: Windowed drift excessive (windowed_drift_acceptable=false)
F5.4: Missing audit artifact
F5.5: Schema validation failure (malformed JSON, missing required fields)
F5.6: Pathological data (NaN/Inf detected)
F5.7: IQR ratio out of bounds (iqr_ratio_acceptable=false)

================================================================================
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# =============================================================================
# Constants
# =============================================================================

SCHEMA_VERSION = "1.0.0"
EXPERIMENT_ID = "CAL-EXP-4"

# Required audit artifacts
TEMPORAL_AUDIT_PATH = "validity/temporal_structure_audit.json"
VARIANCE_AUDIT_PATH = "validity/variance_profile_audit.json"

# Schema field requirements
TEMPORAL_REQUIRED_FIELDS = {
    "schema_version",
    "experiment_id",
    "baseline_arm",
    "treatment_arm",
    "comparability",
    "thresholds",
    "generated_at",
}

VARIANCE_REQUIRED_FIELDS = {
    "schema_version",
    "experiment_id",
    "baseline_arm",
    "treatment_arm",
    "comparability",
    "thresholds",
    "generated_at",
}

TEMPORAL_COMPARABILITY_FIELDS = {
    "cycle_count_match",
    "cycle_indices_identical",
    "coverage_ratio_match",
    "gap_structure_compatible",
    "temporal_structure_compatible",
    "temporal_structure_pass",
}

VARIANCE_COMPARABILITY_FIELDS = {
    "variance_ratio",
    "variance_ratio_acceptable",
    "windowed_variance_drift",
    "windowed_drift_acceptable",
    "iqr_ratio",
    "iqr_ratio_acceptable",
    "profile_compatible",
    "variance_profile_pass",
    "claim_cap_applied",
}

ARM_TEMPORAL_FIELDS = {
    "cycle_count",
    "cycle_min",
    "cycle_max",
    "cycle_gap_max",
    "cycle_gap_mean",
    "monotonic_cycle_indices",
    "timestamp_monotonic",
    "temporal_coverage_ratio",
}

ARM_VARIANCE_FIELDS = {
    "delta_p_count",
    "delta_p_mean",
    "delta_p_variance",
    "delta_p_std",
    "delta_p_iqr",
    "delta_p_range",
    "delta_p_min",
    "delta_p_max",
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CheckResult:
    """Result of a single invariant check."""
    name: str
    passed: bool
    expected: str
    actual: str
    invalidates: bool  # True = FAIL on violation, False = WARN only
    f5_code: Optional[str] = None  # F5.x failure code if applicable

    def __str__(self) -> str:
        status = "PASS" if self.passed else ("FAIL" if self.invalidates else "WARN")
        code_suffix = f" [{self.f5_code}]" if self.f5_code and not self.passed else ""
        return f"{status}: {self.name}: expected={self.expected}, actual={self.actual}{code_suffix}"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        status = "PASS" if self.passed else ("FAIL" if self.invalidates else "WARN")
        result = {
            "name": self.name,
            "status": status,
            "passed": self.passed,
            "expected": self.expected,
            "actual": self.actual,
            "invalidates": self.invalidates,
        }
        if self.f5_code:
            result["f5_code"] = self.f5_code
        return result


@dataclass
class VerificationReport:
    """Aggregate verification report for CAL-EXP-4."""
    run_dir: str
    checks: List[CheckResult] = field(default_factory=list)
    temporal_comparability: bool = True
    variance_comparability: bool = True
    claim_cap_applied: bool = False
    claim_cap_level: Optional[str] = None

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

    @property
    def f5_failure_codes(self) -> List[str]:
        """List of F5.x codes for failed checks."""
        codes = []
        for c in self.checks:
            if not c.passed and c.f5_code and c.f5_code not in codes:
                codes.append(c.f5_code)
        return sorted(codes)

    def add(self, result: CheckResult) -> None:
        self.checks.append(result)

    def print_report(self) -> None:
        print(f"=== CAL-EXP-4 RUN VERIFICATION ===")
        print(f"Run Directory: {self.run_dir}")
        print()
        for check in self.checks:
            print(str(check))
        print()
        print(f"SUMMARY: {len(self.checks)} checks, {self.fail_count} FAIL, {self.warn_count} WARN")
        print(f"temporal_comparability: {self.temporal_comparability}")
        print(f"variance_comparability: {self.variance_comparability}")
        print(f"claim_cap_applied: {self.claim_cap_applied}")
        print(f"claim_cap_level: {self.claim_cap_level}")
        if self.f5_failure_codes:
            print(f"f5_failure_codes: {self.f5_failure_codes}")
        print(f"VERDICT: {'PASS' if self.passed else 'FAIL'}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize report to dictionary for JSON export (deterministic)."""
        return {
            "schema_version": "1.0.0",
            "verifier": "verify_cal_exp_4_run.py",
            "canonical_sources": [
                "schemas/cal_exp_4/temporal_structure_audit.schema.json",
                "schemas/cal_exp_4/variance_profile_audit.schema.json",
                "docs/system_law/calibration/CAL_EXP_4_VERIFIER_PLAN.md",
            ],
            "run_dir": self.run_dir,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "verdict": "PASS" if self.passed else "FAIL",
            "summary": {
                "total_checks": len(self.checks),
                "pass_count": self.pass_count,
                "fail_count": self.fail_count,
                "warn_count": self.warn_count,
            },
            "temporal_comparability": self.temporal_comparability,
            "variance_comparability": self.variance_comparability,
            "f5_failure_codes": self.f5_failure_codes,
            "claim_cap_applied": self.claim_cap_applied,
            "claim_cap_level": self.claim_cap_level,
            "checks": [c.to_dict() for c in self.checks],
        }

    def write_json(self, path: Path) -> None:
        """Write report to JSON file with deterministic ordering."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)


# =============================================================================
# File Loading Utilities
# =============================================================================

def load_json_safe(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Load JSON file with error handling. Returns (data, error_msg)."""
    if not path.exists():
        return None, f"file not found: {path}"
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None, f"expected object, got {type(data).__name__}"
        return data, None
    except json.JSONDecodeError as e:
        return None, f"invalid JSON: {e}"
    except Exception as e:
        return None, f"read error: {e}"


# =============================================================================
# Schema Validation
# =============================================================================

def validate_schema_version(data: Dict[str, Any], expected: str) -> Tuple[bool, str]:
    """Validate schema_version field."""
    actual = data.get("schema_version")
    if actual != expected:
        return False, f"schema_version={actual}, expected={expected}"
    return True, f"schema_version={actual}"


def validate_experiment_id(data: Dict[str, Any], expected: str) -> Tuple[bool, str]:
    """Validate experiment_id field."""
    actual = data.get("experiment_id")
    if actual != expected:
        return False, f"experiment_id={actual}, expected={expected}"
    return True, f"experiment_id={actual}"


def validate_required_fields(
    data: Dict[str, Any],
    required: Set[str],
    context: str
) -> Tuple[bool, List[str]]:
    """Validate that all required fields are present. Returns (passed, missing)."""
    missing = required - set(data.keys())
    return len(missing) == 0, sorted(missing)


def validate_arm_fields(
    arm_data: Optional[Dict[str, Any]],
    required: Set[str],
    arm_name: str
) -> Tuple[bool, List[str]]:
    """Validate arm profile has required fields."""
    if arm_data is None:
        return False, [f"{arm_name} is null"]
    if not isinstance(arm_data, dict):
        return False, [f"{arm_name} is not an object"]
    missing = required - set(arm_data.keys())
    return len(missing) == 0, sorted(missing)


def check_for_nan_inf(value: Any, path: str) -> List[str]:
    """Recursively check for NaN/Inf in numeric values. Returns list of paths with issues."""
    issues = []
    if isinstance(value, float):
        if math.isnan(value):
            issues.append(f"{path}: NaN")
        elif math.isinf(value):
            issues.append(f"{path}: Inf")
    elif isinstance(value, dict):
        for k, v in value.items():
            issues.extend(check_for_nan_inf(v, f"{path}.{k}"))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            issues.extend(check_for_nan_inf(v, f"{path}[{i}]"))
    return issues


# =============================================================================
# Temporal Structure Checks
# =============================================================================

def check_temporal_audit_present(run_dir: Path, report: VerificationReport) -> Optional[Dict[str, Any]]:
    """Check for temporal_structure_audit.json presence and validity."""
    path = run_dir / TEMPORAL_AUDIT_PATH
    data, error = load_json_safe(path)

    if error:
        report.add(CheckResult(
            name="artifact:temporal_structure_audit",
            passed=False,
            expected="file present and valid JSON",
            actual=error,
            invalidates=True,
            f5_code="F5.4",
        ))
        report.temporal_comparability = False
        return None

    report.add(CheckResult(
        name="artifact:temporal_structure_audit",
        passed=True,
        expected="file present and valid JSON",
        actual="present",
        invalidates=True,
    ))
    return data


def validate_temporal_schema(data: Dict[str, Any], report: VerificationReport) -> bool:
    """Validate temporal audit against schema requirements."""
    # Check schema_version
    passed, msg = validate_schema_version(data, SCHEMA_VERSION)
    report.add(CheckResult(
        name="schema:temporal_version",
        passed=passed,
        expected=f"schema_version={SCHEMA_VERSION}",
        actual=msg,
        invalidates=True,
        f5_code="F5.5" if not passed else None,
    ))
    if not passed:
        report.temporal_comparability = False
        return False

    # Check experiment_id
    passed, msg = validate_experiment_id(data, EXPERIMENT_ID)
    report.add(CheckResult(
        name="schema:temporal_experiment_id",
        passed=passed,
        expected=f"experiment_id={EXPERIMENT_ID}",
        actual=msg,
        invalidates=True,
        f5_code="F5.5" if not passed else None,
    ))
    if not passed:
        report.temporal_comparability = False
        return False

    # Check required top-level fields
    passed, missing = validate_required_fields(data, TEMPORAL_REQUIRED_FIELDS, "temporal_audit")
    report.add(CheckResult(
        name="schema:temporal_required_fields",
        passed=passed,
        expected="all required fields present",
        actual="present" if passed else f"missing: {missing}",
        invalidates=True,
        f5_code="F5.5" if not passed else None,
    ))
    if not passed:
        report.temporal_comparability = False
        return False

    # Check arm fields
    for arm_name in ["baseline_arm", "treatment_arm"]:
        arm_data = data.get(arm_name)
        passed, missing = validate_arm_fields(arm_data, ARM_TEMPORAL_FIELDS, arm_name)
        report.add(CheckResult(
            name=f"schema:temporal_{arm_name}_fields",
            passed=passed,
            expected="all required fields present",
            actual="present" if passed else f"missing: {missing}",
            invalidates=True,
            f5_code="F5.5" if not passed else None,
        ))
        if not passed:
            report.temporal_comparability = False
            return False

    # Check comparability fields
    comp = data.get("comparability", {})
    passed, missing = validate_required_fields(comp, TEMPORAL_COMPARABILITY_FIELDS, "temporal.comparability")
    report.add(CheckResult(
        name="schema:temporal_comparability_fields",
        passed=passed,
        expected="all comparability fields present",
        actual="present" if passed else f"missing: {missing}",
        invalidates=True,
        f5_code="F5.5" if not passed else None,
    ))
    if not passed:
        report.temporal_comparability = False
        return False

    return True


def check_temporal_nan_inf(data: Dict[str, Any], report: VerificationReport) -> bool:
    """Check for NaN/Inf in temporal audit data."""
    issues = check_for_nan_inf(data, "temporal_audit")
    passed = len(issues) == 0
    report.add(CheckResult(
        name="pathology:temporal_nan_inf",
        passed=passed,
        expected="no NaN/Inf values",
        actual="clean" if passed else f"{len(issues)} issues: {issues[:3]}",
        invalidates=True,
        f5_code="F5.6" if not passed else None,
    ))
    if not passed:
        report.temporal_comparability = False
    return passed


def check_temporal_comparability_predicates(data: Dict[str, Any], report: VerificationReport) -> None:
    """Check all 7 temporal comparability predicates."""
    comp = data.get("comparability", {})
    baseline = data.get("baseline_arm", {})
    treatment = data.get("treatment_arm", {})

    # 1. cycle_count_match
    value = comp.get("cycle_count_match", False)
    report.add(CheckResult(
        name="temporal:cycle_count_match",
        passed=value is True,
        expected="true",
        actual=str(value),
        invalidates=True,
        f5_code="F5.1" if value is not True else None,
    ))
    if value is not True:
        report.temporal_comparability = False

    # 2. cycle_indices_identical
    value = comp.get("cycle_indices_identical", False)
    report.add(CheckResult(
        name="temporal:cycle_indices_identical",
        passed=value is True,
        expected="true",
        actual=str(value),
        invalidates=True,
        f5_code="F5.1" if value is not True else None,
    ))
    if value is not True:
        report.temporal_comparability = False

    # 3. coverage_ratio_match
    value = comp.get("coverage_ratio_match", False)
    report.add(CheckResult(
        name="temporal:coverage_ratio_match",
        passed=value is True,
        expected="true",
        actual=str(value),
        invalidates=True,
        f5_code="F5.1" if value is not True else None,
    ))
    if value is not True:
        report.temporal_comparability = False

    # 4. gap_structure_compatible
    value = comp.get("gap_structure_compatible", False)
    report.add(CheckResult(
        name="temporal:gap_structure_compatible",
        passed=value is True,
        expected="true",
        actual=str(value),
        invalidates=True,
        f5_code="F5.1" if value is not True else None,
    ))
    if value is not True:
        report.temporal_comparability = False

    # 5. temporal_structure_compatible
    value = comp.get("temporal_structure_compatible", False)
    report.add(CheckResult(
        name="temporal:structure_compatible",
        passed=value is True,
        expected="true",
        actual=str(value),
        invalidates=True,
        f5_code="F5.1" if value is not True else None,
    ))
    if value is not True:
        report.temporal_comparability = False

    # 6. baseline monotonic
    value = baseline.get("monotonic_cycle_indices", False)
    report.add(CheckResult(
        name="temporal:baseline_monotonic",
        passed=value is True,
        expected="true",
        actual=str(value),
        invalidates=True,
        f5_code="F5.1" if value is not True else None,
    ))
    if value is not True:
        report.temporal_comparability = False

    # 7. treatment monotonic
    value = treatment.get("monotonic_cycle_indices", False)
    report.add(CheckResult(
        name="temporal:treatment_monotonic",
        passed=value is True,
        expected="true",
        actual=str(value),
        invalidates=True,
        f5_code="F5.1" if value is not True else None,
    ))
    if value is not True:
        report.temporal_comparability = False

    # Composite: temporal_structure_pass
    value = comp.get("temporal_structure_pass", False)
    report.add(CheckResult(
        name="temporal:structure_pass",
        passed=value is True,
        expected="true",
        actual=str(value),
        invalidates=True,
        f5_code="F5.1" if value is not True else None,
    ))
    if value is not True:
        report.temporal_comparability = False


# =============================================================================
# Variance Profile Checks
# =============================================================================

def check_variance_audit_present(run_dir: Path, report: VerificationReport) -> Optional[Dict[str, Any]]:
    """Check for variance_profile_audit.json presence and validity."""
    path = run_dir / VARIANCE_AUDIT_PATH
    data, error = load_json_safe(path)

    if error:
        report.add(CheckResult(
            name="artifact:variance_profile_audit",
            passed=False,
            expected="file present and valid JSON",
            actual=error,
            invalidates=True,
            f5_code="F5.4",
        ))
        report.variance_comparability = False
        # Missing audit triggers claim cap to L3
        report.claim_cap_applied = True
        report.claim_cap_level = "L3"
        return None

    report.add(CheckResult(
        name="artifact:variance_profile_audit",
        passed=True,
        expected="file present and valid JSON",
        actual="present",
        invalidates=True,
    ))
    return data


def validate_variance_schema(data: Dict[str, Any], report: VerificationReport) -> bool:
    """Validate variance audit against schema requirements."""
    # Check schema_version
    passed, msg = validate_schema_version(data, SCHEMA_VERSION)
    report.add(CheckResult(
        name="schema:variance_version",
        passed=passed,
        expected=f"schema_version={SCHEMA_VERSION}",
        actual=msg,
        invalidates=True,
        f5_code="F5.5" if not passed else None,
    ))
    if not passed:
        report.variance_comparability = False
        return False

    # Check experiment_id
    passed, msg = validate_experiment_id(data, EXPERIMENT_ID)
    report.add(CheckResult(
        name="schema:variance_experiment_id",
        passed=passed,
        expected=f"experiment_id={EXPERIMENT_ID}",
        actual=msg,
        invalidates=True,
        f5_code="F5.5" if not passed else None,
    ))
    if not passed:
        report.variance_comparability = False
        return False

    # Check required top-level fields
    passed, missing = validate_required_fields(data, VARIANCE_REQUIRED_FIELDS, "variance_audit")
    report.add(CheckResult(
        name="schema:variance_required_fields",
        passed=passed,
        expected="all required fields present",
        actual="present" if passed else f"missing: {missing}",
        invalidates=True,
        f5_code="F5.5" if not passed else None,
    ))
    if not passed:
        report.variance_comparability = False
        return False

    # Check arm fields
    for arm_name in ["baseline_arm", "treatment_arm"]:
        arm_data = data.get(arm_name)
        passed, missing = validate_arm_fields(arm_data, ARM_VARIANCE_FIELDS, arm_name)
        report.add(CheckResult(
            name=f"schema:variance_{arm_name}_fields",
            passed=passed,
            expected="all required fields present",
            actual="present" if passed else f"missing: {missing}",
            invalidates=True,
            f5_code="F5.5" if not passed else None,
        ))
        if not passed:
            report.variance_comparability = False
            return False

    # Check comparability fields
    comp = data.get("comparability", {})
    passed, missing = validate_required_fields(comp, VARIANCE_COMPARABILITY_FIELDS, "variance.comparability")
    report.add(CheckResult(
        name="schema:variance_comparability_fields",
        passed=passed,
        expected="all comparability fields present",
        actual="present" if passed else f"missing: {missing}",
        invalidates=True,
        f5_code="F5.5" if not passed else None,
    ))
    if not passed:
        report.variance_comparability = False
        return False

    return True


def check_variance_nan_inf(data: Dict[str, Any], report: VerificationReport) -> bool:
    """Check for NaN/Inf in variance audit data."""
    issues = check_for_nan_inf(data, "variance_audit")
    passed = len(issues) == 0
    report.add(CheckResult(
        name="pathology:variance_nan_inf",
        passed=passed,
        expected="no NaN/Inf values",
        actual="clean" if passed else f"{len(issues)} issues: {issues[:3]}",
        invalidates=True,
        f5_code="F5.6" if not passed else None,
    ))
    if not passed:
        report.variance_comparability = False
    return passed


def check_arm_pathology(data: Dict[str, Any], report: VerificationReport) -> bool:
    """Check for has_nan/has_inf flags in arm data."""
    all_clean = True

    for arm_name in ["baseline_arm", "treatment_arm"]:
        arm = data.get(arm_name, {})
        has_nan = arm.get("has_nan", False)
        has_inf = arm.get("has_inf", False)

        if has_nan:
            report.add(CheckResult(
                name=f"pathology:{arm_name}_has_nan",
                passed=False,
                expected="false",
                actual="true",
                invalidates=True,
                f5_code="F5.6",
            ))
            report.variance_comparability = False
            all_clean = False

        if has_inf:
            report.add(CheckResult(
                name=f"pathology:{arm_name}_has_inf",
                passed=False,
                expected="false",
                actual="true",
                invalidates=True,
                f5_code="F5.6",
            ))
            report.variance_comparability = False
            all_clean = False

    if all_clean:
        report.add(CheckResult(
            name="pathology:arm_data_clean",
            passed=True,
            expected="no has_nan/has_inf flags",
            actual="clean",
            invalidates=True,
        ))

    return all_clean


def check_variance_comparability_predicates(data: Dict[str, Any], report: VerificationReport) -> None:
    """Check variance comparability predicates and apply claim capping."""
    comp = data.get("comparability", {})

    # F5.2: variance_ratio_acceptable
    value = comp.get("variance_ratio_acceptable", False)
    ratio = comp.get("variance_ratio", "N/A")
    report.add(CheckResult(
        name="variance:ratio_acceptable",
        passed=value is True,
        expected="true",
        actual=f"{value} (ratio={ratio})",
        invalidates=False,  # May cap claim instead of fail-close
        f5_code="F5.2" if value is not True else None,
    ))
    if value is not True:
        report.variance_comparability = False

    # F5.3: windowed_drift_acceptable
    value = comp.get("windowed_drift_acceptable", False)
    drift = comp.get("windowed_variance_drift", "N/A")
    report.add(CheckResult(
        name="variance:windowed_drift_acceptable",
        passed=value is True,
        expected="true",
        actual=f"{value} (drift={drift})",
        invalidates=False,  # May cap claim instead of fail-close
        f5_code="F5.3" if value is not True else None,
    ))
    if value is not True:
        report.variance_comparability = False

    # F5.7: iqr_ratio_acceptable
    value = comp.get("iqr_ratio_acceptable", False)
    iqr_ratio = comp.get("iqr_ratio", "N/A")
    report.add(CheckResult(
        name="variance:iqr_ratio_acceptable",
        passed=value is True,
        expected="true",
        actual=f"{value} (iqr_ratio={iqr_ratio})",
        invalidates=False,  # May cap claim instead of fail-close
        f5_code="F5.7" if value is not True else None,
    ))
    if value is not True:
        report.variance_comparability = False

    # profile_compatible (composite)
    value = comp.get("profile_compatible", False)
    report.add(CheckResult(
        name="variance:profile_compatible",
        passed=value is True,
        expected="true",
        actual=str(value),
        invalidates=True,  # Fail-close on incompatible profiles
        f5_code="F5.2" if value is not True else None,  # Generic variance failure
    ))
    if value is not True:
        report.variance_comparability = False

    # variance_profile_pass (final verdict from harness)
    value = comp.get("variance_profile_pass", False)
    report.add(CheckResult(
        name="variance:profile_pass",
        passed=value is True,
        expected="true",
        actual=str(value),
        invalidates=True,
        f5_code="F5.2" if value is not True else None,
    ))
    if value is not True:
        report.variance_comparability = False

    # Apply claim capping from audit artifact
    claim_cap_applied = comp.get("claim_cap_applied", False)
    claim_cap_level = comp.get("claim_cap_level")

    if claim_cap_applied:
        report.claim_cap_applied = True
        report.claim_cap_level = claim_cap_level
        report.add(CheckResult(
            name="claim:cap_applied",
            passed=True,  # Not a failure, just informational
            expected="claim cap from audit",
            actual=f"capped to {claim_cap_level}",
            invalidates=False,
        ))


# =============================================================================
# Main Verification
# =============================================================================

def verify_run(run_dir: Path) -> VerificationReport:
    """Run all CAL-EXP-4 verification checks."""
    report = VerificationReport(run_dir=str(run_dir))

    # Check run directory exists
    if not run_dir.exists():
        report.add(CheckResult(
            name="run_dir:exists",
            passed=False,
            expected="directory exists",
            actual="not found",
            invalidates=True,
            f5_code="F5.4",
        ))
        report.temporal_comparability = False
        report.variance_comparability = False
        return report

    report.add(CheckResult(
        name="run_dir:exists",
        passed=True,
        expected="directory exists",
        actual="found",
        invalidates=True,
    ))

    # =========================================================================
    # Temporal Structure Audit
    # =========================================================================
    temporal_data = check_temporal_audit_present(run_dir, report)

    if temporal_data is not None:
        # Validate schema
        if validate_temporal_schema(temporal_data, report):
            # Check for NaN/Inf
            if check_temporal_nan_inf(temporal_data, report):
                # Check all 7 temporal comparability predicates
                check_temporal_comparability_predicates(temporal_data, report)

    # =========================================================================
    # Variance Profile Audit
    # =========================================================================
    variance_data = check_variance_audit_present(run_dir, report)

    if variance_data is not None:
        # Validate schema
        if validate_variance_schema(variance_data, report):
            # Check for NaN/Inf
            if check_variance_nan_inf(variance_data, report):
                # Check arm pathology flags
                if check_arm_pathology(variance_data, report):
                    # Check variance comparability predicates
                    check_variance_comparability_predicates(variance_data, report)

    return report


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="CAL-EXP-4 Run Verifier — Variance Stress Test Comparability Check"
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to CAL-EXP-4 run directory",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=None,
        help="Path to write JSON verification report",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output",
    )

    args = parser.parse_args()

    report = verify_run(args.run_dir)

    if not args.quiet:
        report.print_report()

    if args.output_report:
        report.write_json(args.output_report)
        if not args.quiet:
            print(f"\nReport written to: {args.output_report}")

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
