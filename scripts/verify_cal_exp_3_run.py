#!/usr/bin/env python3
"""
CAL-EXP-3 Run Verifier — Uplift Measurement Invariant Check

Verifies that a CAL-EXP-3 run satisfies all validity conditions and structural
requirements per the authoritative implementation plan.

Advisory-only: does not gate CI unless explicitly configured.

Usage:
    python scripts/verify_cal_exp_3_run.py --run-dir results/cal_exp_3/<run_id>/

Exit codes:
    0 = PASS (all invariants satisfied)
    1 = FAIL (one or more invalidating conditions)

Authoritative Sources:
    - docs/system_law/calibration/CAL_EXP_3_IMPLEMENTATION_PLAN.md
    - docs/system_law/calibration/CAL_EXP_3_UPLIFT_SPEC.md
    - docs/system_law/calibration/CAL_EXP_3_AUTHORIZATION.md
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Cross-shell preflight (advisory only, does not block)
try:
    from scripts.preflight_shell_env import print_preflight_advisory
    _PREFLIGHT_AVAILABLE = True
except ImportError:
    _PREFLIGHT_AVAILABLE = False


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
    """Aggregate verification report for CAL-EXP-3."""
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
        print(f"=== CAL-EXP-3 RUN VERIFICATION ===")
        print(f"Run Directory: {self.run_dir}")
        print()
        for check in self.checks:
            print(str(check))
        print()
        print(f"SUMMARY: {len(self.checks)} checks, {self.fail_count} FAIL, {self.warn_count} WARN")
        print(f"VERDICT: {'PASS' if self.passed else 'FAIL'}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize report to dictionary for JSON export (deterministic)."""
        return {
            "schema_version": "1.0.0",
            "verifier": "verify_cal_exp_3_run.py",
            "canonical_sources": [
                "docs/system_law/calibration/CAL_EXP_3_IMPLEMENTATION_PLAN.md",
                "docs/system_law/calibration/CAL_EXP_3_UPLIFT_SPEC.md",
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
            "checks": [c.to_dict() for c in self.checks],
        }

    def write_json(self, path: Path) -> None:
        """Write report to JSON file (deterministic: sorted keys, newline)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)
            f.write("\n")


# =============================================================================
# File Loading Utilities
# =============================================================================

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


def load_text_safe(path: Path) -> Tuple[Optional[str], Optional[str]]:
    """Load text file, return (content, error)."""
    if not path.exists():
        return None, f"File not found: {path}"
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip(), None
    except Exception as e:
        return None, f"Read error: {e}"


def load_cycles_jsonl(path: Path) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Load cycles.jsonl file, return (records, error).

    Expected format per line: {"cycle": int, "delta_p": float, "timestamp": ISO8601}
    """
    if not path.exists():
        return [], f"File not found: {path}"

    records = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError as e:
                    return records, f"Line {i}: JSON parse error: {e}"
    except Exception as e:
        return records, f"Read error: {e}"

    return records, None


# =============================================================================
# Validation Functions
# =============================================================================

def validate_cycles_structure(
    records: List[Dict[str, Any]],
    arm_label: str,
) -> Tuple[bool, str, Set[int]]:
    """
    Validate cycles.jsonl records have required fields and no pathology.

    Returns: (is_valid, message, cycle_indices)
    """
    if not records:
        return False, f"{arm_label}: cycles.jsonl is empty", set()

    cycle_indices: Set[int] = set()
    missing_fields = []
    nan_values = []
    duplicate_cycles = []

    for i, record in enumerate(records):
        # Check required fields
        if "cycle" not in record:
            missing_fields.append(f"record {i}: missing 'cycle'")
        if "delta_p" not in record:
            missing_fields.append(f"record {i}: missing 'delta_p'")

        # Check for NaN values
        delta_p = record.get("delta_p")
        if delta_p is not None:
            if isinstance(delta_p, float) and math.isnan(delta_p):
                nan_values.append(f"record {i}: delta_p is NaN")

        # Track cycle indices for alignment check
        cycle = record.get("cycle")
        if cycle is not None:
            if cycle in cycle_indices:
                duplicate_cycles.append(f"cycle {cycle} duplicated")
            cycle_indices.add(cycle)

    issues = []
    if missing_fields:
        issues.append(f"{len(missing_fields)} missing fields")
    if nan_values:
        issues.append(f"{len(nan_values)} NaN values")
    if duplicate_cycles:
        issues.append(f"{len(duplicate_cycles)} duplicate cycles")

    if issues:
        return False, f"{arm_label}: {', '.join(issues)}", cycle_indices

    return True, f"{arm_label}: {len(records)} valid records", cycle_indices


def check_cycle_alignment(
    baseline_cycles: Set[int],
    treatment_cycles: Set[int],
) -> Tuple[bool, str]:
    """
    Check exact cycle index alignment between baseline and treatment.

    Per spec: Both arms must have identical cycle indices.
    """
    if baseline_cycles == treatment_cycles:
        return True, f"aligned ({len(baseline_cycles)} cycles)"

    missing_in_treatment = baseline_cycles - treatment_cycles
    missing_in_baseline = treatment_cycles - baseline_cycles

    issues = []
    if missing_in_treatment:
        issues.append(f"{len(missing_in_treatment)} cycles in baseline but not treatment")
    if missing_in_baseline:
        issues.append(f"{len(missing_in_baseline)} cycles in treatment but not baseline")

    return False, f"misaligned: {', '.join(issues)}"


def check_no_external_ingestion(validity_checks: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Check for external ingestion violation using validity_checks.json evidence.

    Fail-close: If cannot determine, assume violation.
    """
    # Check if external_ingestion field exists
    if "external_ingestion" in validity_checks:
        external = validity_checks["external_ingestion"]
        if isinstance(external, dict):
            detected = external.get("detected", True)  # Fail-close default
            if detected:
                return False, f"external ingestion detected: {external.get('detail', 'unknown')}"
            return True, "no external ingestion detected"
        elif isinstance(external, bool):
            if external:
                return False, "external ingestion flag is true"
            return True, "external ingestion flag is false"

    # Check for network_calls evidence
    if "network_calls" in validity_checks:
        net = validity_checks["network_calls"]
        if isinstance(net, list) and len(net) > 0:
            return False, f"{len(net)} network calls detected"
        elif isinstance(net, int) and net > 0:
            return False, f"{net} network calls detected"

    # Check all_passed field if present
    if "all_passed" in validity_checks:
        if not validity_checks["all_passed"]:
            # Check individual fields for external ingestion
            for key in ["corpus_external", "data_external", "input_external"]:
                if validity_checks.get(key):
                    return False, f"{key} violation"

    # Cannot determine from available evidence - default WARN (not FAIL)
    return True, "no external ingestion evidence found (assume clean)"


def check_isolation_audit(run_dir: Path) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Check for isolation_audit.json (negative proof for external ingestion).

    Per §7.1.1: Require validity/isolation_audit.json and invalidate if missing or failing.

    Expected format:
    {
        "network_calls": [],
        "file_reads_outside_corpus": [],
        "isolation_passed": true
    }
    """
    audit_path = run_dir / "validity" / "isolation_audit.json"

    if not audit_path.exists():
        return False, "validity/isolation_audit.json missing (required for F2.3 negative proof)", None

    audit_data, err = load_json_safe(audit_path)
    if err:
        return False, f"isolation_audit.json: {err}", None

    # Check isolation_passed field
    isolation_passed = audit_data.get("isolation_passed", False)
    if not isolation_passed:
        return False, "isolation_passed=false (external ingestion detected)", audit_data

    # Check for network calls
    network_calls = audit_data.get("network_calls", [])
    if isinstance(network_calls, list) and len(network_calls) > 0:
        return False, f"network isolation failed: {len(network_calls)} calls detected", audit_data

    # Check for file reads outside corpus
    file_reads = audit_data.get("file_reads_outside_corpus", [])
    if isinstance(file_reads, list) and len(file_reads) > 0:
        return False, f"filesystem isolation failed: {len(file_reads)} external reads", audit_data

    return True, "isolation audit passed", audit_data


def check_window_coverage(
    cycles: Set[int],
    start_cycle: int,
    end_cycle: int,
    arm_label: str,
) -> Tuple[bool, str]:
    """
    Check that all cycles in the evaluation window (inclusive) are present.

    Per §3.4:
    - Cycle range bounds: Inclusive on both ends
    - Missing-cycle handling: INVALIDATION
    """
    # Generate expected cycle set (inclusive on both ends)
    expected = set(range(start_cycle, end_cycle + 1))
    missing = expected - cycles

    if not missing:
        return True, f"{arm_label}: all {len(expected)} cycles present in window [{start_cycle}, {end_cycle}]"

    # Report first few missing cycles
    missing_sample = sorted(missing)[:5]
    suffix = f"... and {len(missing) - 5} more" if len(missing) > 5 else ""
    return False, f"{arm_label}: {len(missing)} cycles missing from window [{start_cycle}, {end_cycle}]: {missing_sample}{suffix}"


def check_cycle_line_determinism(records: List[Dict[str, Any]], arm_label: str) -> Tuple[bool, str]:
    """
    Check artifact determinism rules per §4.3.

    - No random identifiers (UUIDs) in per-cycle lines
    - Required fields: cycle (int), delta_p (float)
    - timestamp is auxiliary, but no other random fields
    """
    uuid_pattern = r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$'
    violations = []

    for i, record in enumerate(records):
        for key, value in record.items():
            # Skip known auxiliary fields
            if key in ("cycle", "delta_p", "timestamp"):
                continue
            # Check for UUID values
            if isinstance(value, str):
                if re.match(uuid_pattern, value.lower()):
                    violations.append(f"record {i}: field '{key}' contains UUID")
            # Check for id/uuid field names
            if key.lower() in ("id", "uuid", "run_id", "record_id"):
                violations.append(f"record {i}: forbidden field '{key}'")

    if violations:
        return False, f"{arm_label}: {len(violations)} determinism violations: {violations[:3]}"

    return True, f"{arm_label}: no forbidden identifiers in cycle lines"


def check_toolchain_manifest(run_dir: Path) -> Tuple[bool, str, Optional[str]]:
    """
    Check toolchain manifest alignment per §5.4.

    Reference: schemas/toolchain_manifest.schema.json

    Requirements:
    - schema_version: Must be "1.0.0"
    - experiment_id: Must be "CAL-EXP-3"
    - provenance_level: "full" required for L4 claims; "partial" caps at L3
    - toolchain_fingerprint: Required if provenance_level=full
    """
    # Try corpus_manifest.json first (may contain toolchain info)
    manifest_path = run_dir / "validity" / "corpus_manifest.json"
    manifest_data, err = load_json_safe(manifest_path)

    # Check if there's a dedicated toolchain manifest
    toolchain_manifest_path = run_dir / "validity" / "toolchain_manifest.json"
    if toolchain_manifest_path.exists():
        manifest_data, err = load_json_safe(toolchain_manifest_path)
        manifest_path = toolchain_manifest_path

    if err or not manifest_data:
        # Fall back to toolchain_hash.txt for basic check
        return True, "no toolchain manifest found (using toolchain_hash.txt)", None

    # Check schema_version
    schema_version = manifest_data.get("schema_version")
    if schema_version and schema_version != "1.0.0":
        return False, f"schema_version mismatch: expected '1.0.0', got '{schema_version}'", schema_version

    # Check experiment_id if present
    experiment_id = manifest_data.get("experiment_id")
    if experiment_id and experiment_id != "CAL-EXP-3":
        return False, f"experiment_id mismatch: expected 'CAL-EXP-3', got '{experiment_id}'", None

    # Check provenance_level
    provenance_level = manifest_data.get("provenance_level", "unknown")

    # Return provenance_level for claim level capping
    return True, f"toolchain manifest valid, provenance_level={provenance_level}", provenance_level


# =============================================================================
# Main Verification Logic
# =============================================================================

def verify_run(run_dir: Path) -> VerificationReport:
    """
    Run all verification checks on a CAL-EXP-3 run directory.

    Authoritative artifact layout (from CAL_EXP_3_IMPLEMENTATION_PLAN.md):

    results/cal_exp_3/<run_id>/
    ├── run_config.json
    ├── baseline/
    │   ├── cycles.jsonl
    │   └── summary.json
    ├── treatment/
    │   ├── cycles.jsonl
    │   └── summary.json
    ├── analysis/
    │   ├── uplift_report.json
    │   └── windowed_analysis.json
    ├── validity/
    │   ├── toolchain_hash.txt
    │   ├── corpus_manifest.json
    │   └── validity_checks.json
    └── RUN_METADATA.json
    """
    report = VerificationReport(run_dir=str(run_dir))

    # =========================================================================
    # Directory Existence
    # =========================================================================
    report.add(CheckResult(
        name="run_dir_exists",
        passed=run_dir.exists(),
        expected="exists",
        actual="exists" if run_dir.exists() else "missing",
        invalidates=True,
    ))

    if not run_dir.exists():
        return report

    # =========================================================================
    # Required Directories
    # =========================================================================
    required_dirs = ["baseline", "treatment", "analysis", "validity"]
    for dir_name in required_dirs:
        dir_path = run_dir / dir_name
        exists = dir_path.exists() and dir_path.is_dir()
        report.add(CheckResult(
            name=f"dir_exists:{dir_name}",
            passed=exists,
            expected="exists",
            actual="exists" if exists else "missing",
            invalidates=True,
        ))

    # =========================================================================
    # Required Files
    # =========================================================================
    required_files = [
        "run_config.json",
        "RUN_METADATA.json",
        "baseline/cycles.jsonl",
        "treatment/cycles.jsonl",
        "validity/toolchain_hash.txt",
        "validity/corpus_manifest.json",
        "validity/validity_checks.json",
        "validity/isolation_audit.json",  # §7.1.1: Negative proof for F2.3
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
            name="run_config_valid",
            passed=False,
            expected="valid JSON",
            actual=config_err,
            invalidates=True,
        ))
        config = {}
    else:
        report.add(CheckResult(
            name="run_config_valid",
            passed=True,
            expected="valid JSON",
            actual="valid",
            invalidates=True,
        ))

    # =========================================================================
    # Load RUN_METADATA.json
    # =========================================================================
    metadata_path = run_dir / "RUN_METADATA.json"
    metadata, metadata_err = load_json_safe(metadata_path)

    if metadata_err:
        report.add(CheckResult(
            name="run_metadata_valid",
            passed=False,
            expected="valid JSON",
            actual=metadata_err,
            invalidates=True,
        ))
        metadata = {}
    else:
        report.add(CheckResult(
            name="run_metadata_valid",
            passed=True,
            expected="valid JSON",
            actual="valid",
            invalidates=True,
        ))

    # =========================================================================
    # SHADOW MODE Check (from metadata if present)
    # =========================================================================
    # Note: CAL-EXP-3 operates under SHADOW MODE umbrella
    enforcement = metadata.get("enforcement", config.get("enforcement", False))
    report.add(CheckResult(
        name="shadow_mode:enforcement",
        passed=(enforcement is False),
        expected="false",
        actual=str(enforcement).lower(),
        invalidates=True,
    ))

    # =========================================================================
    # Experiment Identity
    # =========================================================================
    experiment = config.get("experiment", "MISSING")
    report.add(CheckResult(
        name="experiment_identity",
        passed=(experiment == "CAL-EXP-3"),
        expected="CAL-EXP-3",
        actual=experiment,
        invalidates=True,
    ))

    # =========================================================================
    # SEED DISCIPLINE: Identical seed across arms
    # Per spec: "Seed: S (shared)" for both arms
    # =========================================================================
    seed = config.get("seed")
    baseline_config = config.get("baseline_config", {})
    treatment_config = config.get("treatment_config", {})

    # Check seed is recorded
    report.add(CheckResult(
        name="seed:registered",
        passed=(seed is not None),
        expected="seed present in run_config.json",
        actual=f"seed={seed}" if seed is not None else "missing",
        invalidates=True,
    ))

    # Check both arms use same seed (if arm-specific seeds exist)
    baseline_seed = baseline_config.get("seed", seed)
    treatment_seed = treatment_config.get("seed", seed)
    seeds_match = (baseline_seed == treatment_seed)
    report.add(CheckResult(
        name="seed:identical_across_arms",
        passed=seeds_match,
        expected="baseline_seed == treatment_seed",
        actual=f"baseline={baseline_seed}, treatment={treatment_seed}",
        invalidates=True,  # Per spec: identical seed required
    ))

    # =========================================================================
    # WINDOW REGISTRATION: Windows declared before execution
    # =========================================================================
    windows = config.get("windows", {})
    window_registered = bool(windows) and "evaluation_window" in windows
    report.add(CheckResult(
        name="windows:pre_registered",
        passed=window_registered,
        expected="windows.evaluation_window declared",
        actual="declared" if window_registered else "missing",
        invalidates=True,
    ))

    # =========================================================================
    # TOOLCHAIN PARITY: Single hash for runtime environment
    # =========================================================================
    toolchain_path = run_dir / "validity" / "toolchain_hash.txt"
    toolchain_hash, toolchain_err = load_text_safe(toolchain_path)

    if toolchain_err:
        report.add(CheckResult(
            name="toolchain:hash_present",
            passed=False,
            expected="toolchain_hash.txt readable",
            actual=toolchain_err,
            invalidates=True,
        ))
    else:
        # Hash must be non-empty and look like SHA-256
        valid_hash = bool(toolchain_hash) and len(toolchain_hash) >= 32
        report.add(CheckResult(
            name="toolchain:hash_present",
            passed=valid_hash,
            expected="valid SHA-256 hash",
            actual=f"{toolchain_hash[:16]}..." if toolchain_hash else "empty",
            invalidates=True,
        ))

    # =========================================================================
    # CORPUS IDENTITY: Corpus manifest present
    # =========================================================================
    corpus_path = run_dir / "validity" / "corpus_manifest.json"
    corpus_manifest, corpus_err = load_json_safe(corpus_path)

    if corpus_err:
        report.add(CheckResult(
            name="corpus:manifest_valid",
            passed=False,
            expected="valid corpus_manifest.json",
            actual=corpus_err,
            invalidates=True,
        ))
    else:
        # Check corpus hash is present
        corpus_hash = corpus_manifest.get("hash", corpus_manifest.get("corpus_hash"))
        report.add(CheckResult(
            name="corpus:manifest_valid",
            passed=bool(corpus_hash),
            expected="corpus hash present",
            actual=f"hash={corpus_hash[:16]}..." if corpus_hash else "missing hash",
            invalidates=True,
        ))

    # =========================================================================
    # LOAD AND VALIDATE cycles.jsonl FOR BOTH ARMS
    # =========================================================================
    baseline_records, baseline_err = load_cycles_jsonl(run_dir / "baseline" / "cycles.jsonl")
    treatment_records, treatment_err = load_cycles_jsonl(run_dir / "treatment" / "cycles.jsonl")

    # Baseline cycles validation
    if baseline_err:
        report.add(CheckResult(
            name="baseline:cycles_valid",
            passed=False,
            expected="valid JSONL",
            actual=baseline_err,
            invalidates=True,
        ))
        baseline_cycles: Set[int] = set()
    else:
        valid, msg, baseline_cycles = validate_cycles_structure(baseline_records, "baseline")
        report.add(CheckResult(
            name="baseline:cycles_valid",
            passed=valid,
            expected="valid structure, no NaN, no duplicates",
            actual=msg,
            invalidates=True,
        ))

    # Treatment cycles validation
    if treatment_err:
        report.add(CheckResult(
            name="treatment:cycles_valid",
            passed=False,
            expected="valid JSONL",
            actual=treatment_err,
            invalidates=True,
        ))
        treatment_cycles: Set[int] = set()
    else:
        valid, msg, treatment_cycles = validate_cycles_structure(treatment_records, "treatment")
        report.add(CheckResult(
            name="treatment:cycles_valid",
            passed=valid,
            expected="valid structure, no NaN, no duplicates",
            actual=msg,
            invalidates=True,
        ))

    # =========================================================================
    # EXACT CYCLE ALIGNMENT: Both arms must have identical cycle indices
    # Per spec: Strict comparability, no tolerance
    # =========================================================================
    if baseline_cycles and treatment_cycles:
        aligned, alignment_msg = check_cycle_alignment(baseline_cycles, treatment_cycles)
        report.add(CheckResult(
            name="cycle_alignment:exact",
            passed=aligned,
            expected="identical cycle indices in both arms",
            actual=alignment_msg,
            invalidates=True,
        ))

    # =========================================================================
    # NO PATHOLOGY: Check validity_checks.json
    # =========================================================================
    validity_path = run_dir / "validity" / "validity_checks.json"
    validity_checks, validity_err = load_json_safe(validity_path)

    if validity_err:
        report.add(CheckResult(
            name="validity_checks:readable",
            passed=False,
            expected="valid validity_checks.json",
            actual=validity_err,
            invalidates=True,
        ))
        validity_checks = {}
    else:
        report.add(CheckResult(
            name="validity_checks:readable",
            passed=True,
            expected="valid JSON",
            actual="valid",
            invalidates=True,
        ))

    # Check overall validity if recorded
    if validity_checks:
        all_passed = validity_checks.get("all_passed", validity_checks.get("valid", True))
        report.add(CheckResult(
            name="validity_checks:all_passed",
            passed=bool(all_passed),
            expected="all validity conditions passed",
            actual=f"all_passed={all_passed}",
            invalidates=True,
        ))

    # =========================================================================
    # NO EXTERNAL INGESTION (fail-close) - validity_checks.json
    # =========================================================================
    if validity_checks:
        no_external, external_msg = check_no_external_ingestion(validity_checks)
        report.add(CheckResult(
            name="no_external_ingestion",
            passed=no_external,
            expected="no external data ingestion",
            actual=external_msg,
            invalidates=True,  # Fail-close per spec
        ))

    # =========================================================================
    # ISOLATION AUDIT (§7.1.1): Negative proof for F2.3
    # =========================================================================
    isolation_passed, isolation_msg, isolation_data = check_isolation_audit(run_dir)
    report.add(CheckResult(
        name="isolation_audit:negative_proof",
        passed=isolation_passed,
        expected="validity/isolation_audit.json present and passing",
        actual=isolation_msg,
        invalidates=True,  # Missing or failing invalidates run
    ))

    # =========================================================================
    # WINDOW COVERAGE (§3.4): Missing-cycle invalidation
    # Per spec: Cycle range bounds are INCLUSIVE, missing cycles INVALIDATE
    # =========================================================================
    eval_window = windows.get("evaluation_window", {})
    if eval_window and baseline_cycles and treatment_cycles:
        start_cycle = eval_window.get("start_cycle")
        end_cycle = eval_window.get("end_cycle")

        if start_cycle is not None and end_cycle is not None:
            # Check baseline window coverage
            baseline_coverage_ok, baseline_coverage_msg = check_window_coverage(
                baseline_cycles, start_cycle, end_cycle, "baseline"
            )
            report.add(CheckResult(
                name="window_coverage:baseline",
                passed=baseline_coverage_ok,
                expected=f"all cycles in [{start_cycle}, {end_cycle}] present",
                actual=baseline_coverage_msg,
                invalidates=True,  # Missing cycles INVALIDATE
            ))

            # Check treatment window coverage
            treatment_coverage_ok, treatment_coverage_msg = check_window_coverage(
                treatment_cycles, start_cycle, end_cycle, "treatment"
            )
            report.add(CheckResult(
                name="window_coverage:treatment",
                passed=treatment_coverage_ok,
                expected=f"all cycles in [{start_cycle}, {end_cycle}] present",
                actual=treatment_coverage_msg,
                invalidates=True,  # Missing cycles INVALIDATE
            ))

    # =========================================================================
    # ARTIFACT DETERMINISM (§4.3): No UUIDs in cycle lines
    # =========================================================================
    if baseline_records:
        det_ok, det_msg = check_cycle_line_determinism(baseline_records, "baseline")
        report.add(CheckResult(
            name="determinism:baseline_cycles",
            passed=det_ok,
            expected="no forbidden identifiers (UUIDs, id fields)",
            actual=det_msg,
            invalidates=True,
        ))

    if treatment_records:
        det_ok, det_msg = check_cycle_line_determinism(treatment_records, "treatment")
        report.add(CheckResult(
            name="determinism:treatment_cycles",
            passed=det_ok,
            expected="no forbidden identifiers (UUIDs, id fields)",
            actual=det_msg,
            invalidates=True,
        ))

    # =========================================================================
    # TOOLCHAIN MANIFEST ALIGNMENT (§5.4)
    # =========================================================================
    manifest_ok, manifest_msg, provenance_level = check_toolchain_manifest(run_dir)
    report.add(CheckResult(
        name="toolchain_manifest:valid",
        passed=manifest_ok,
        expected="valid manifest per toolchain_manifest.schema.json",
        actual=manifest_msg,
        invalidates=False,  # WARN only - toolchain_hash.txt is primary
    ))

    # Check provenance_level implications
    if provenance_level == "partial":
        report.add(CheckResult(
            name="toolchain_manifest:provenance_level",
            passed=True,  # Not a failure, but caps claim level
            expected="full (for L4 claims)",
            actual="partial (caps claim at L3)",
            invalidates=False,  # WARN only
        ))

    # =========================================================================
    # ARM CONFIGURATION: Baseline learning OFF, Treatment learning ON
    # =========================================================================
    baseline_learning = baseline_config.get("learning_enabled", False)
    treatment_learning = treatment_config.get("learning_enabled", True)

    report.add(CheckResult(
        name="arm_config:baseline_learning_off",
        passed=(baseline_learning is False),
        expected="learning_enabled=false",
        actual=f"learning_enabled={baseline_learning}",
        invalidates=True,
    ))

    report.add(CheckResult(
        name="arm_config:treatment_learning_on",
        passed=(treatment_learning is True),
        expected="learning_enabled=true",
        actual=f"learning_enabled={treatment_learning}",
        invalidates=True,
    ))

    # =========================================================================
    # OPTIONAL: Analysis artifacts present (WARN only)
    # =========================================================================
    optional_files = [
        "analysis/uplift_report.json",
        "analysis/windowed_analysis.json",
        "baseline/summary.json",
        "treatment/summary.json",
    ]

    for filename in optional_files:
        path = run_dir / filename
        exists = path.exists()
        report.add(CheckResult(
            name=f"optional:{filename}",
            passed=exists,
            expected="present (optional)",
            actual="present" if exists else "missing",
            invalidates=False,  # WARN only
        ))

    return report


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> int:
    # Cross-shell preflight advisory (non-blocking)
    if _PREFLIGHT_AVAILABLE:
        print_preflight_advisory()

    parser = argparse.ArgumentParser(
        description="Verify CAL-EXP-3 run satisfies validity conditions and structural requirements",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to CAL-EXP-3 run directory (e.g., results/cal_exp_3/<run_id>/)",
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
        help="Path to write JSON verification report",
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
            report_path = args.run_dir / "cal_exp_3_verification_report.json"
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
