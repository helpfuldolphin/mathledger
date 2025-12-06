#!/usr/bin/env python3
"""
VCP-2.2 promotion precheck harness for the First Organism.

Runs the canonical integration test, verifies determinism/security,
and writes an RFC 8785-compatible JSON report consumed by the future
Wave 1 promotion machinery.

HARD GATES (must pass for promotion):
- SPARK checks: closed_loop_standalone + determinism tests
- Attestation file: artifacts/first_organism/attestation.json with R_t, U_t, H_t

INFORMATIONAL CHECKS (plumbing, not success metrics):
- RFL logs (fo_rfl.jsonl, fo_rfl_50.jsonl, etc.): Existence only, no parsing
- Dyno charts (rfl_dyno_chart.png): Existence only, no validation

SPARK Gate Clarification:
The precheck script does NOT fail if RFL logs show 100% abstention or
any other abstention pattern. RFL log content (abstention rates, uplift
metrics, cycle counts) are NOT evaluated for SPARK pass/fail. Only FO
hermetic tests and attestation integrity are hard gates.
- Wide Slice data: Phase II expectations, advisory only

CRITICAL: RFL logs and Dyno charts are plumbing checks; Phase I does not
require abstention reduction to promote. The precheck does NOT parse RFL
logs to demand uplift or validate abstention rates. 100% abstention in
RFL logs is acceptable and will not block promotion.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

TEST_CASE = "tests/integration/test_first_organism.py"
MARKER = "first_organism"
KEY_EXPR = "closed_loop_standalone or determinism"
PYTEST_CMD = [
    "uv",
    "run",
    "pytest",
    TEST_CASE,
    "-m",
    MARKER,
    "-k",
    KEY_EXPR,
    "-v",
    "-s",
]
SCRIPT_ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = SCRIPT_ROOT / "artifacts" / "first_organism" / "basis_precheck_report.json"
ATTESTATION_PATH = SCRIPT_ROOT / "artifacts" / "first_organism" / "attestation.json"
BASELINE_WIDE_PATH = SCRIPT_ROOT / "results" / "fo_baseline_wide.jsonl"
RFL_WIDE_PATH = SCRIPT_ROOT / "results" / "fo_rfl_wide.jsonl"
DYNO_CHART_PATH = SCRIPT_ROOT / "artifacts" / "figures" / "rfl_dyno_chart.png"


class PrecheckFailure(Exception):
    """Raised when a precheck guard fails."""


def run_test() -> tuple[int, str]:
    """Execute the First Organism integration test and capture stdout."""
    process = subprocess.run(
        PYTEST_CMD,
        cwd=SCRIPT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
    )
    return process.returncode, process.stdout


def analyze_output(output: str) -> tuple[bool, bool, str]:
    """
    Parse pytest output for motif clues and attestation lines.
    
    Expected PASS line format (from tests/integration/conftest.py:log_first_organism_pass):
    "[PASS] FIRST ORGANISM ALIVE H_t=<12-char-hex>"
    """
    normalized = output.upper()
    if "[ABSTAIN]" in normalized:
        raise PrecheckFailure("Security harness reported [ABSTAIN]; aborting promotion precheck.")
    pass_line = ""
    for line in output.splitlines():
        # Match the exact format: [PASS] FIRST ORGANISM ALIVE H_t=<hash>
        if "[PASS] FIRST ORGANISM ALIVE" in line.upper():
            pass_line = line
            break
    if not pass_line:
        raise PrecheckFailure("First Organism PASS marker missing from logs.")
    if "H_T=" not in pass_line.upper():
        raise PrecheckFailure("Dual-attestation roots (H_t) not logged on PASS line.")
    closed_loop_pass = any(
        "test_first_organism_closed_loop_standalone" in line.lower() and "passed" in line.lower()
        for line in output.splitlines()
    )
    determinism_pass = any(
        "test_first_organism_determinism" in line.lower() and "passed" in line.lower()
        for line in output.splitlines()
    )
    # Extract H_t from PASS line: format is "H_t=<12-char-hex>" (may have ANSI color codes)
    ht_marker = ""
    if "H_T=" in pass_line.upper():
        # Handle ANSI escape codes that may be present
        clean_line = pass_line
        # Remove ANSI escape sequences if present
        import re
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        clean_line = ansi_escape.sub('', clean_line)
        parts = clean_line.upper().split("H_T=")
        if len(parts) > 1:
            ht_marker = parts[1].split()[0] if parts[1].split() else ""
    return closed_loop_pass, determinism_pass, ht_marker


def load_attestation() -> tuple[str, str, str, bytes]:
    """Load the attestation artifact and extract the roots."""
    if not ATTESTATION_PATH.exists():
        raise PrecheckFailure(f"Missing attestation file: {ATTESTATION_PATH}")
    raw = ATTESTATION_PATH.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise PrecheckFailure(f"Attestation JSON corrupted: {exc}") from exc

    def _get(key_names: list[str], label: str) -> str:
        for key in key_names:
            if key in payload:
                return str(payload[key])
        raise PrecheckFailure(f"{label} missing in attestation JSON.")

    # Attestation file uses R_t, U_t, H_t (with underscores, uppercase) per artifacts/first_organism/attestation.json
    rt = _get(["R_t", "reasoningMerkleRoot", "reasoning_root", "reasoningRoot", "reasoning"], "R_t")
    ut = _get(["U_t", "uiMerkleRoot", "ui_root", "uiRoot", "ui"], "U_t")
    ht = _get(["H_t", "compositeAttestationRoot", "composite_root", "compositeRoot", "H"], "H_t")
    return rt, ut, ht, raw


def check_wide_slice_data() -> dict[str, object]:
    """
    Check for Wide Slice Dyno Chart data files (existence only, no parsing).
    
    NOTE: These are Phase II expectations. For Phase I, we check for:
    - results/fo_baseline.jsonl (first-organism-slice baseline)
    - results/fo_rfl_50.jsonl (50-cycle RFL sanity run)
    
    Wide Slice (slice_medium) runs are Phase II and may not exist yet.
    
    IMPORTANT: This function only checks file existence. It does NOT:
    - Parse RFL logs to demand uplift
    - Validate abstention rates (100% abstention is acceptable)
    - Check for success metrics or performance improvements
    
    RFL logs and Dyno charts are plumbing checks; Phase I does not require
    abstention reduction to promote.
    
    Returns status dict with file presence only (no schema validation).
    """
    baseline_present = BASELINE_WIDE_PATH.exists()
    rfl_present = RFL_WIDE_PATH.exists()
    
    # Existence check only - no parsing, no validation, no uplift checks
    # This ensures 100% abstention in RFL logs won't break the precheck
    
    return {
        "baseline_log_present": baseline_present,
        "rfl_log_present": rfl_present,
        "schema_check_passed": False,  # Not checked - existence only
    }


def check_dyno_chart() -> bool:
    """
    Check if Dyno Chart visualization exists (existence only, no validation).
    
    IMPORTANT: This is a plumbing check, not a success metric.
    Phase I does not require Dyno chart validation to promote.
    The chart may show 100% abstention and that is acceptable.
    """
    return DYNO_CHART_PATH.exists()


def write_report(report: dict[str, object]) -> bytes:
    """Serialize the report with canonical JSON settings."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        report,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    REPORT_PATH.write_bytes(payload)
    return payload


def main() -> None:
    report: dict[str, object] = {
        "status": "fail",
        "reason": "Precheck not executed.",
        "ht": None,
        "rt": None,
        "ut": None,
        "timestamp_iso": None,
        "test_case": TEST_CASE,
        "security_harness_passed": False,
        "determinism_verified": False,
        "spark_status": {
            "closed_loop_standalone": False,
            "determinism": False,
        },
        "wide_slice_data_status": {
            "baseline_log_present": False,
            "rfl_log_present": False,
            "schema_check_passed": False,
        },
        "dyno_chart_present": False,
    }

    try:
        # --- HARD GATE: SPARK Checks (closed_loop_standalone + determinism) ---
        retcode, stdout = run_test()
        if retcode != 0:
            raise PrecheckFailure(f"First Organism test failed (exit code {retcode}).")
        closed_loop_pass, determinism_pass, ht_from_log = analyze_output(stdout)
        if not closed_loop_pass or not determinism_pass:
            raise PrecheckFailure("Hermetic First Organism tests did not both pass.")
        rt, ut, ht, first_bytes = load_attestation()
        candidate_ht = hashlib.sha256((rt + ut).encode("ascii")).hexdigest()
        if candidate_ht != ht:
            raise PrecheckFailure("Computed H_t disagrees with attestation JSON.")
        if ht_from_log and ht_from_log != ht.upper():
            raise PrecheckFailure("H_t logged on PASS line does not match attestation H_t.")

        # SPARK checks passed - update report
        report["spark_status"] = {
            "closed_loop_standalone": closed_loop_pass,
            "determinism": determinism_pass,
        }

        # --- INFORMATIONAL: RFL Logs and Dyno Chart Checks (plumbing, not success metrics) ---
        # RFL logs and Dyno charts are plumbing checks; Phase I does not require
        # abstention reduction to promote. These checks are existence-only and
        # do NOT parse logs to demand uplift or validate abstention rates.
        wide_slice_status = check_wide_slice_data()
        dyno_chart_present = check_dyno_chart()
        
        report["wide_slice_data_status"] = wide_slice_status
        report["dyno_chart_present"] = dyno_chart_present

        # Determine overall status and reason
        # NOTE: RFL/Dyno presence does NOT affect promotion status - SPARK is the only gate
        all_wide_present = (
            wide_slice_status["baseline_log_present"]
            and wide_slice_status["rfl_log_present"]
        )
        
        # Status is always "pass" if SPARK checks passed (RFL/Dyno are informational only)
        status = "pass"
        if all_wide_present and dyno_chart_present:
            reason = "Hermetic First Organism (closed loop + determinism) is alive. RFL logs and Dyno chart present (informational)."
        elif all_wide_present:
            reason = "Hermetic First Organism (closed loop + determinism) is alive. RFL logs present, Dyno chart missing (informational)."
        else:
            reason = "Hermetic First Organism (closed loop + determinism) is alive. RFL logs/Dyno chart missing (informational, does not block promotion)."

        report.update(
            status=status,
            reason=reason,
            ht=ht,
            rt=rt,
            ut=ut,
            timestamp_iso=datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            security_harness_passed=True,
            determinism_verified=True,
        )
        payload = write_report(report)
        print(f"Precheck report written to {REPORT_PATH} (SHA256={hashlib.sha256(payload).hexdigest()}).")
        
        # Print informational notes if data is missing (not warnings - these don't block)
        if not all_wide_present:
            print("\n[INFO] RFL logs missing (Phase II expectation, informational only):")
            if not wide_slice_status["baseline_log_present"]:
                print(f"  - Missing: {BASELINE_WIDE_PATH}")
                print(f"    Note: Phase I uses results/fo_baseline.jsonl (first-organism-slice)")
            if not wide_slice_status["rfl_log_present"]:
                print(f"  - Missing: {RFL_WIDE_PATH}")
                print(f"    Note: Phase I uses results/fo_rfl_50.jsonl (50-cycle RFL sanity run)")
            print("  Note: RFL logs are plumbing checks; Phase I does not require abstention reduction to promote.")
        if not dyno_chart_present:
            print(f"\n[INFO] Dyno Chart visualization missing (informational only): {DYNO_CHART_PATH}")
            print("  Note: Check artifacts/figures/rfl_abstention_rate.png for Phase I visualization")
            print("  Note: Dyno charts are plumbing checks; Phase I does not require validation to promote.")
        
        sys.exit(0)
    except PrecheckFailure as exc:
        report["reason"] = str(exc)
        report["timestamp_iso"] = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        # Still check wide slice data even on failure for completeness
        report["wide_slice_data_status"] = check_wide_slice_data()
        report["dyno_chart_present"] = check_dyno_chart()
        payload = write_report(report)
        print(f"Precheck failed: {exc}")
        print(f"Report recorded at {REPORT_PATH}")
        sys.exit(1)


if __name__ == "__main__":
    main()

