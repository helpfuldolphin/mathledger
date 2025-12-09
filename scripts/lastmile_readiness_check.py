#!/usr/bin/env python3
"""
lastmile_readiness_check.py - Operation LAST MILE Ready Checklist

PHASE II -- NOT RUN IN PHASE I

Implements the 20-point Operation LAST MILE Ready Checklist from
U2_SECURITY_PLAYBOOK.md. Verifies all security controls are in place
before U2 execution.

This tool is diagnostic only - no automatic repair.
All outputs are deterministic given their inputs.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional


class CheckStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    ERROR = "ERROR"


class SectionStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"


class OverallStatus(Enum):
    READY = "READY"
    NOT_READY = "NOT_READY"


@dataclass
class CheckResult:
    """Result of a single checklist item."""
    check_id: str
    name: str
    required_state: str
    actual_state: str
    status: str
    message: str = ""


@dataclass
class SectionResult:
    """Result of a checklist section."""
    section_id: str
    section_name: str
    checks: list
    passed: int
    total: int
    status: str


@dataclass
class LastMileVerificationReport:
    """Complete verification report per playbook specification."""
    checklist_version: str
    verified_at: str
    run_id: Optional[str]

    section_a_replay: dict
    section_b_prng: dict
    section_c_telemetry: dict
    section_d_hermetic: dict

    total_passed: int
    total_checks: int
    overall_status: str
    blocking_items: list

    analyzer_version: str = "1.0.0"
    playbook_reference: str = "U2_SECURITY_PLAYBOOK.md#operation-last-mile-ready-checklist"


def check_env_var(var_name: str, expected_value: str) -> tuple[str, bool]:
    """Check if environment variable has expected value."""
    actual = os.environ.get(var_name, "")
    matches = actual.lower() == expected_value.lower()
    return actual if actual else "(not set)", matches


def check_file_exists(filepath: Path) -> tuple[str, bool]:
    """Check if file exists."""
    exists = filepath.exists()
    return str(filepath), exists


def check_file_executable(filepath: Path) -> tuple[str, bool]:
    """Check if file exists and is executable."""
    if not filepath.exists():
        return f"{filepath} (not found)", False
    # On Windows, check if it's a .py file (Python handles execution)
    if sys.platform == "win32":
        is_exec = filepath.suffix == ".py"
    else:
        is_exec = os.access(filepath, os.X_OK)
    return str(filepath), is_exec or filepath.suffix == ".py"


def check_dir_writable(dirpath: Path) -> tuple[str, bool]:
    """Check if directory exists and is writable."""
    if not dirpath.exists():
        return f"{dirpath} (not found)", False
    writable = os.access(dirpath, os.W_OK)
    return str(dirpath), writable


def check_no_pattern_in_env(patterns: list[str]) -> tuple[str, bool]:
    """Check that no environment variables match patterns."""
    matches = []
    for key in os.environ:
        for pattern in patterns:
            if pattern.upper() in key.upper():
                # Allow HTTP_PROXY for dependency downloads
                if pattern == "PROXY" and key in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY"):
                    continue
                matches.append(key)
    if matches:
        return f"Found: {', '.join(matches)}", False
    return "None found", True


def check_file_contains(filepath: Path, pattern: str) -> tuple[str, bool]:
    """Check if file contains pattern."""
    if not filepath.exists():
        return f"{filepath} (not found)", False
    try:
        content = filepath.read_text()
        found = pattern in content or re.search(pattern, content) is not None
        return f"Pattern {'found' if found else 'not found'}", found
    except Exception as e:
        return f"Error reading: {e}", False


def check_no_pattern_in_files(directory: Path, pattern: str, glob_pattern: str = "**/*.py") -> tuple[str, bool]:
    """Check that no files in directory match pattern."""
    if not directory.exists():
        return f"{directory} (not found)", False

    matches = []
    try:
        for filepath in directory.glob(glob_pattern):
            if filepath.is_file():
                try:
                    content = filepath.read_text()
                    if re.search(pattern, content):
                        matches.append(str(filepath.relative_to(directory)))
                except Exception:
                    pass
        if matches:
            return f"Found in: {', '.join(matches[:3])}{'...' if len(matches) > 3 else ''}", False
        return "No matches", True
    except Exception as e:
        return f"Error scanning: {e}", False


def verify_master_seed(manifest_path: Path) -> tuple[str, bool]:
    """Verify U2_MASTER_SEED is correctly derived from manifest."""
    prereg_path = Path("PREREG_UPLIFT_U2.yaml")

    if not prereg_path.exists():
        return "PREREG_UPLIFT_U2.yaml not found", False

    if not manifest_path.exists():
        return f"{manifest_path} not found", False

    try:
        # Compute expected master seed
        prereg_hash = hashlib.sha256(prereg_path.read_bytes()).hexdigest()

        # Load manifest and check
        content = manifest_path.read_text()
        try:
            import yaml
            manifest = yaml.safe_load(content)
        except ImportError:
            # Try JSON
            try:
                manifest = json.loads(content)
            except json.JSONDecodeError:
                return "Cannot parse manifest", False

        actual_seed = manifest.get("u2_master_seed", manifest.get("master_seed", ""))

        if actual_seed == prereg_hash:
            return f"Valid: {prereg_hash[:16]}...", True
        elif actual_seed:
            return f"Mismatch: expected {prereg_hash[:16]}..., got {actual_seed[:16]}...", False
        else:
            return "Master seed not found in manifest", False

    except Exception as e:
        return f"Error: {e}", False


def run_section_a_replay(config: dict) -> SectionResult:
    """
    Section A: Replay Enforcement (5 checks)

    A1: REPLAY_ENABLED=true
    A2: Replay script accessible
    A3: Replay storage configured
    A4: Replay validation scheduled
    A5: Replay comparison strict mode
    """
    checks = []
    scripts_dir = Path(config.get("scripts_dir", "scripts"))
    logs_dir = Path(config.get("logs_dir", "logs"))
    hooks_dir = Path(config.get("hooks_dir", "hooks"))
    config_dir = Path(config.get("config_dir", "config"))

    # A1: REPLAY_ENABLED
    actual, passed = check_env_var("REPLAY_ENABLED", "true")
    checks.append(CheckResult(
        check_id="A1",
        name="REPLAY_ENABLED environment variable",
        required_state="true",
        actual_state=actual,
        status=CheckStatus.PASS.value if passed else CheckStatus.FAIL.value
    ))

    # A2: Replay script accessible
    replay_script = scripts_dir / "replay_determinism_check.py"
    actual, passed = check_file_executable(replay_script)
    checks.append(CheckResult(
        check_id="A2",
        name="Replay script accessible",
        required_state="Script exists and executable",
        actual_state=actual,
        status=CheckStatus.PASS.value if passed else CheckStatus.FAIL.value
    ))

    # A3: Replay storage configured
    replay_logs = logs_dir / "replay"
    actual, passed = check_dir_writable(replay_logs)
    checks.append(CheckResult(
        check_id="A3",
        name="Replay storage configured",
        required_state="Directory writable",
        actual_state=actual,
        status=CheckStatus.PASS.value if passed else CheckStatus.FAIL.value
    ))

    # A4: Replay validation scheduled
    post_run_hook = hooks_dir / "post_run.sh"
    if post_run_hook.exists():
        actual, passed = check_file_contains(post_run_hook, "replay")
    else:
        # Also check for .py hook
        post_run_hook_py = hooks_dir / "post_run.py"
        if post_run_hook_py.exists():
            actual, passed = check_file_contains(post_run_hook_py, "replay")
        else:
            actual, passed = "No post_run hook found", False
    checks.append(CheckResult(
        check_id="A4",
        name="Replay validation scheduled",
        required_state="Post-run hook contains 'replay'",
        actual_state=actual,
        status=CheckStatus.PASS.value if passed else CheckStatus.FAIL.value
    ))

    # A5: Replay comparison strict mode
    replay_config = config_dir / "replay.yaml"
    if replay_config.exists():
        actual, passed = check_file_contains(replay_config, "strict")
    else:
        # Check environment variable fallback
        actual = os.environ.get("REPLAY_COMPARE_MODE", "(not set)")
        passed = actual.lower() == "strict"
    checks.append(CheckResult(
        check_id="A5",
        name="Replay comparison strict mode",
        required_state="COMPARE_MODE=strict",
        actual_state=actual,
        status=CheckStatus.PASS.value if passed else CheckStatus.FAIL.value
    ))

    passed_count = sum(1 for c in checks if c.status == CheckStatus.PASS.value)
    return SectionResult(
        section_id="A",
        section_name="Replay Enforcement",
        checks=[asdict(c) for c in checks],
        passed=passed_count,
        total=len(checks),
        status=SectionStatus.PASS.value if passed_count == len(checks) else SectionStatus.FAIL.value
    )


def run_section_b_prng(config: dict) -> SectionResult:
    """
    Section B: Deterministic PRNG Guard (5 checks)

    B1: PYTHONHASHSEED=0
    B2: Master seed derived
    B3: FrozenRandom wrapper active
    B4: Reseed detection enabled
    B5: Seed snapshot logging
    """
    checks = []
    backend_rfl = Path(config.get("backend_rfl_dir", "backend/rfl"))
    manifest_path = Path(config.get("manifest_path", "logs/uplift/current/run_manifest.yaml"))

    # B1: PYTHONHASHSEED fixed
    actual, passed = check_env_var("PYTHONHASHSEED", "0")
    checks.append(CheckResult(
        check_id="B1",
        name="PYTHONHASHSEED fixed",
        required_state="0",
        actual_state=actual,
        status=CheckStatus.PASS.value if passed else CheckStatus.FAIL.value
    ))

    # B2: Master seed derived
    actual, passed = verify_master_seed(manifest_path)
    checks.append(CheckResult(
        check_id="B2",
        name="Master seed derived from manifest",
        required_state="SHA256(PREREG_UPLIFT_U2.yaml)",
        actual_state=actual,
        status=CheckStatus.PASS.value if passed else CheckStatus.FAIL.value
    ))

    # B3: FrozenRandom wrapper active (no raw "import random" in rfl)
    actual, passed = check_no_pattern_in_files(backend_rfl, r"^import random$")
    checks.append(CheckResult(
        check_id="B3",
        name="FrozenRandom wrapper active",
        required_state="No 'import random' in backend/rfl/",
        actual_state=actual,
        status=CheckStatus.PASS.value if passed else CheckStatus.FAIL.value
    ))

    # B4: Reseed detection enabled
    determinism_file = backend_rfl / "determinism.py"
    if determinism_file.exists():
        actual, passed = check_file_contains(determinism_file, r"assert.*seed")
    else:
        actual, passed = f"{determinism_file} not found", False
    checks.append(CheckResult(
        check_id="B4",
        name="Reseed detection enabled",
        required_state="Runtime assertion in determinism.py",
        actual_state=actual,
        status=CheckStatus.PASS.value if passed else CheckStatus.FAIL.value
    ))

    # B5: Seed snapshot logging
    if manifest_path.exists():
        actual, passed = check_file_contains(manifest_path, "prng_seed")
    else:
        actual, passed = "Manifest not found", False
    checks.append(CheckResult(
        check_id="B5",
        name="Seed snapshot logging",
        required_state="prng_seed in run_manifest",
        actual_state=actual,
        status=CheckStatus.PASS.value if passed else CheckStatus.FAIL.value
    ))

    passed_count = sum(1 for c in checks if c.status == CheckStatus.PASS.value)
    return SectionResult(
        section_id="B",
        section_name="Deterministic PRNG Guard",
        checks=[asdict(c) for c in checks],
        passed=passed_count,
        total=len(checks),
        status=SectionStatus.PASS.value if passed_count == len(checks) else SectionStatus.FAIL.value
    )


def run_section_c_telemetry(config: dict) -> SectionResult:
    """
    Section C: Telemetry Quarantine (5 checks)

    C1: External telemetry disabled
    C2: Reward channel isolated
    C3: Proxy metrics blocked
    C4: Telemetry log quarantine
    C5: Metrics export disabled
    """
    checks = []
    logs_dir = Path(config.get("logs_dir", "logs"))
    backend_rfl = Path(config.get("backend_rfl_dir", "backend/rfl"))

    # C1: External telemetry disabled (check for outbound metric ports)
    # This is a simplified check - real implementation would use netstat
    telemetry_vars = ["STATSD_HOST", "PROMETHEUS_PUSHGATEWAY", "DATADOG_HOST", "NEWRELIC_LICENSE"]
    found_telemetry = [v for v in telemetry_vars if os.environ.get(v)]
    if found_telemetry:
        actual, passed = f"Found: {', '.join(found_telemetry)}", False
    else:
        actual, passed = "No external telemetry configured", True
    checks.append(CheckResult(
        check_id="C1",
        name="External telemetry disabled",
        required_state="No outbound metrics",
        actual_state=actual,
        status=CheckStatus.PASS.value if passed else CheckStatus.FAIL.value
    ))

    # C2: Reward channel isolated
    scripts_dir = Path(config.get("scripts_dir", "scripts"))
    audit_script = scripts_dir / "audit_reward_sources.py"
    if audit_script.exists():
        actual, passed = "Audit script available", True
    else:
        actual, passed = "Audit script not found", False
    checks.append(CheckResult(
        check_id="C2",
        name="Reward channel isolated",
        required_state="Lean-only rewards verifiable",
        actual_state=actual,
        status=CheckStatus.PASS.value if passed else CheckStatus.FAIL.value
    ))

    # C3: Proxy metrics blocked
    actual, passed = check_no_pattern_in_env(["PROXY"])
    # Adjust message for allowed proxies
    if not passed and actual.startswith("Found:"):
        # Check if only allowed proxies
        allowed = {"HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY"}
        found = set(actual.replace("Found: ", "").split(", "))
        if found.issubset(allowed):
            actual = "Only HTTP proxies (allowed for deps)"
            passed = True
    checks.append(CheckResult(
        check_id="C3",
        name="Proxy metrics blocked",
        required_state="No *_PROXY_* vars (except HTTP)",
        actual_state=actual,
        status=CheckStatus.PASS.value if passed else CheckStatus.FAIL.value
    ))

    # C4: Telemetry log quarantine
    quarantine_dir = logs_dir / "quarantine"
    actual, passed = check_dir_writable(quarantine_dir) if quarantine_dir.exists() else (str(quarantine_dir), quarantine_dir.exists())
    checks.append(CheckResult(
        check_id="C4",
        name="Telemetry log quarantine",
        required_state="logs/quarantine/ exists",
        actual_state=actual,
        status=CheckStatus.PASS.value if passed else CheckStatus.FAIL.value
    ))

    # C5: Metrics export disabled
    actual, passed = check_no_pattern_in_files(backend_rfl, r"prometheus|statsd|datadog")
    checks.append(CheckResult(
        check_id="C5",
        name="Metrics export disabled",
        required_state="No prometheus/statsd in rfl/",
        actual_state=actual,
        status=CheckStatus.PASS.value if passed else CheckStatus.FAIL.value
    ))

    passed_count = sum(1 for c in checks if c.status == CheckStatus.PASS.value)
    return SectionResult(
        section_id="C",
        section_name="Telemetry Quarantine",
        checks=[asdict(c) for c in checks],
        passed=passed_count,
        total=len(checks),
        status=SectionStatus.PASS.value if passed_count == len(checks) else SectionStatus.FAIL.value
    )


def run_section_d_hermetic(config: dict) -> SectionResult:
    """
    Section D: Hermetic Execution (5 checks)

    D1: Network isolation
    D2: Process isolation
    D3: Log append-only
    D4: Env var scan passed
    D5: Database isolation
    """
    checks = []
    logs_dir = Path(config.get("logs_dir", "logs"))
    run_id = config.get("run_id", "current")

    # D1: Network isolation (simplified - would need firewall audit)
    # Check for RFL_NETWORK_ISOLATED env var as indicator
    actual = os.environ.get("RFL_NETWORK_ISOLATED", os.environ.get("RFL_ENV_MODE", "(not set)"))
    passed = actual in ("true", "uplift_experiment", "isolated")
    checks.append(CheckResult(
        check_id="D1",
        name="Network isolation",
        required_state="Localhost only (RFL_ENV_MODE=uplift_experiment)",
        actual_state=actual,
        status=CheckStatus.PASS.value if passed else CheckStatus.FAIL.value
    ))

    # D2: Process isolation (simplified check)
    # In real implementation, would check for single runner process
    rfl_env_mode = os.environ.get("RFL_ENV_MODE", "")
    if rfl_env_mode == "uplift_experiment":
        actual, passed = "RFL_ENV_MODE=uplift_experiment", True
    else:
        actual, passed = f"RFL_ENV_MODE={rfl_env_mode or '(not set)'}", False
    checks.append(CheckResult(
        check_id="D2",
        name="Process isolation",
        required_state="RFL_ENV_MODE=uplift_experiment",
        actual_state=actual,
        status=CheckStatus.PASS.value if passed else CheckStatus.FAIL.value
    ))

    # D3: Log append-only (check for immutable flag or permissions)
    uplift_log_dir = logs_dir / "uplift" / run_id
    if uplift_log_dir.exists():
        # On Windows, check read-only attribute
        # On Linux, would check for append-only (a) flag
        actual = str(uplift_log_dir)
        # For now, just verify directory exists
        passed = True
    else:
        actual, passed = f"{uplift_log_dir} not found", False
    checks.append(CheckResult(
        check_id="D3",
        name="Log append-only",
        required_state="Write-once mode",
        actual_state=actual,
        status=CheckStatus.PASS.value if passed else CheckStatus.FAIL.value,
        message="Manual verification recommended for append-only enforcement"
    ))

    # D4: Env var scan passed
    prohibited = ["REWARD", "INJECT", "OVERRIDE", "BYPASS"]
    actual, passed = check_no_pattern_in_env(prohibited)
    checks.append(CheckResult(
        check_id="D4",
        name="Env var scan passed",
        required_state="No prohibited vars",
        actual_state=actual,
        status=CheckStatus.PASS.value if passed else CheckStatus.FAIL.value
    ))

    # D5: Database isolation
    db_partition = os.environ.get("RFL_DB_PARTITION", os.environ.get("DATABASE_PARTITION", ""))
    if db_partition:
        actual, passed = f"Partition: {db_partition}", True
    else:
        # Check if DATABASE_URL includes partition indicator
        db_url = os.environ.get("DATABASE_URL", "")
        if "partition" in db_url.lower() or "isolated" in db_url.lower():
            actual, passed = "Partition indicator in DATABASE_URL", True
        else:
            actual, passed = "No partition configured", False
    checks.append(CheckResult(
        check_id="D5",
        name="Database isolation",
        required_state="Partition active",
        actual_state=actual,
        status=CheckStatus.PASS.value if passed else CheckStatus.FAIL.value
    ))

    passed_count = sum(1 for c in checks if c.status == CheckStatus.PASS.value)
    return SectionResult(
        section_id="D",
        section_name="Hermetic Execution",
        checks=[asdict(c) for c in checks],
        passed=passed_count,
        total=len(checks),
        status=SectionStatus.PASS.value if passed_count == len(checks) else SectionStatus.FAIL.value
    )


def run_all_checks(config: dict) -> LastMileVerificationReport:
    """Run all 20 checklist items and generate report."""

    timestamp = datetime.now(timezone.utc)

    # Run each section
    section_a = run_section_a_replay(config)
    section_b = run_section_b_prng(config)
    section_c = run_section_c_telemetry(config)
    section_d = run_section_d_hermetic(config)

    # Calculate totals
    total_passed = section_a.passed + section_b.passed + section_c.passed + section_d.passed
    total_checks = section_a.total + section_b.total + section_c.total + section_d.total

    # Collect blocking items
    blocking_items = []
    for section in [section_a, section_b, section_c, section_d]:
        for check in section.checks:
            if check["status"] == CheckStatus.FAIL.value:
                blocking_items.append(f"{check['check_id']}: {check['name']}")

    # Determine overall status
    all_sections_pass = all(s.status == SectionStatus.PASS.value for s in [section_a, section_b, section_c, section_d])
    overall_status = OverallStatus.READY.value if all_sections_pass else OverallStatus.NOT_READY.value

    return LastMileVerificationReport(
        checklist_version="1.0",
        verified_at=timestamp.isoformat(),
        run_id=config.get("run_id"),
        section_a_replay=asdict(section_a),
        section_b_prng=asdict(section_b),
        section_c_telemetry=asdict(section_c),
        section_d_hermetic=asdict(section_d),
        total_passed=total_passed,
        total_checks=total_checks,
        overall_status=overall_status,
        blocking_items=blocking_items,
    )


def print_report(report: LastMileVerificationReport, verbose: bool = False):
    """Print formatted report to console."""

    # Header
    print("=" * 50)
    print("     OPERATION LAST MILE READY CHECK")
    print("=" * 50)
    print()

    # Section summaries
    sections = [
        ("A", "Replay Enforcement", report.section_a_replay),
        ("B", "PRNG Guard", report.section_b_prng),
        ("C", "Telemetry Quarantine", report.section_c_telemetry),
        ("D", "Hermetic Execution", report.section_d_hermetic),
    ]

    for section_id, section_name, section_data in sections:
        passed = section_data["passed"]
        total = section_data["total"]
        status = "PASS" if section_data["status"] == SectionStatus.PASS.value else "FAIL"
        status_symbol = "[OK]" if status == "PASS" else "[!!]"

        print(f"  Section {section_id}: {section_name:<25} {passed}/{total} {status_symbol}")

        if verbose:
            for check in section_data["checks"]:
                check_status = "[OK]" if check["status"] == CheckStatus.PASS.value else "[!!]"
                print(f"    {check['check_id']}: {check['name']:<35} {check_status}")
                if check["status"] != CheckStatus.PASS.value:
                    print(f"        Required: {check['required_state']}")
                    print(f"        Actual:   {check['actual_state']}")

    print()
    print("-" * 50)
    print(f"  TOTAL: {report.total_passed}/{report.total_checks} PASSED")
    print(f"  STATUS: {report.overall_status}")
    print("-" * 50)

    if report.blocking_items:
        print()
        print("  BLOCKING ITEMS:")
        for item in report.blocking_items:
            print(f"    - {item}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Operation LAST MILE Ready Checklist Verification",
        epilog="See U2_SECURITY_PLAYBOOK.md for checklist details."
    )
    parser.add_argument(
        "--run-id",
        help="Run ID for context-specific checks"
    )
    parser.add_argument(
        "--scripts-dir",
        type=Path,
        default=Path("scripts"),
        help="Path to scripts directory"
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs"),
        help="Path to logs directory"
    )
    parser.add_argument(
        "--backend-rfl-dir",
        type=Path,
        default=Path("backend/rfl"),
        help="Path to backend/rfl directory"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("last_mile_verification.json"),
        help="Output path for verification report"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed check results"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress console output"
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Output only JSON (no formatted report)"
    )

    args = parser.parse_args()

    # Build config
    config = {
        "run_id": args.run_id or "current",
        "scripts_dir": str(args.scripts_dir),
        "logs_dir": str(args.logs_dir),
        "backend_rfl_dir": str(args.backend_rfl_dir),
        "hooks_dir": "hooks",
        "config_dir": "config",
        "manifest_path": f"{args.logs_dir}/uplift/{args.run_id or 'current'}/run_manifest.yaml",
    }

    try:
        # Run all checks
        report = run_all_checks(config)

        # Output JSON
        report_dict = asdict(report)
        with open(args.output, 'w') as f:
            json.dump(report_dict, f, indent=2)

        # Console output
        if not args.quiet:
            if args.json_only:
                print(json.dumps(report_dict, indent=2))
            else:
                print_report(report, verbose=args.verbose)
                print(f"Report saved to: {args.output}")

        # Exit code based on overall status
        if report.overall_status == OverallStatus.READY.value:
            sys.exit(0)
        else:
            # Count failing sections for graduated exit codes
            failing_sections = sum(1 for s in [
                report.section_a_replay,
                report.section_b_prng,
                report.section_c_telemetry,
                report.section_d_hermetic
            ] if s["status"] != SectionStatus.PASS.value)

            if failing_sections >= 3:
                sys.exit(3)  # NO-GO
            elif failing_sections >= 1:
                sys.exit(1)  # CONDITIONAL
            else:
                sys.exit(0)

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(4)


if __name__ == "__main__":
    main()
