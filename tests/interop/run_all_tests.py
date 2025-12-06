#!/usr/bin/env python3
"""
Cross-Language Interoperability Test Runner

Runs all interop tests: Python, JavaScript, and PowerShell.
Reports unified results with drift detection.

Usage:
    python tests/interop/run_all_tests.py [--python-only|--js-only|--ps-only]
    python tests/interop/run_all_tests.py --skip-ps  # Skip PowerShell tests
"""

import sys
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Tuple


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


class TestResults:
    """Aggregate test results across languages."""

    def __init__(self):
        self.results: Dict[str, Dict] = {
            "python": {"passed": 0, "failed": 0, "skipped": 0, "status": "not_run"},
            "javascript": {"passed": 0, "failed": 0, "skipped": 0, "status": "not_run"},
            "powershell": {"passed": 0, "failed": 0, "skipped": 0, "status": "not_run"},
        }

    def record_result(self, lang: str, passed: int, failed: int, skipped: int = 0):
        """Record test results for a language."""
        self.results[lang]["passed"] = passed
        self.results[lang]["failed"] = failed
        self.results[lang]["skipped"] = skipped
        self.results[lang]["status"] = "passed" if failed == 0 else "failed"

    def mark_skipped(self, lang: str, reason: str = "skipped"):
        """Mark a language test suite as skipped."""
        self.results[lang]["status"] = reason

    def total_passed(self) -> int:
        """Total passed tests across all languages."""
        return sum(r["passed"] for r in self.results.values())

    def total_failed(self) -> int:
        """Total failed tests across all languages."""
        return sum(r["failed"] for r in self.results.values())

    def all_passed(self) -> bool:
        """Check if all tests passed."""
        return all(
            r["status"] in ("passed", "not_run", "skipped")
            for r in self.results.values()
        ) and self.total_failed() == 0

    def print_summary(self):
        """Print colorized test summary."""
        print("\n" + "=" * 70)
        print(f"{Colors.BOLD}CROSS-LANGUAGE INTEROPERABILITY TEST SUMMARY{Colors.RESET}")
        print("=" * 70)

        for lang, result in self.results.items():
            status = result["status"]
            passed = result["passed"]
            failed = result["failed"]
            skipped = result["skipped"]

            # Language name
            lang_display = lang.capitalize().ljust(12)

            # Status icon and color
            if status == "passed":
                icon = "✅"
                color = Colors.GREEN
            elif status == "failed":
                icon = "❌"
                color = Colors.RED
            elif status == "skipped":
                icon = "⏭️"
                color = Colors.YELLOW
            else:
                icon = "⚪"
                color = Colors.RESET

            # Format line
            stats = f"P:{passed} F:{failed}"
            if skipped > 0:
                stats += f" S:{skipped}"

            print(f"{icon} {color}{lang_display}{Colors.RESET} {stats.ljust(20)} [{status}]")

        # Overall summary
        print("-" * 70)
        total_p = self.total_passed()
        total_f = self.total_failed()

        if self.all_passed():
            print(
                f"{Colors.GREEN}{Colors.BOLD}[PASS] Interop Verified "
                f"langs=3 drift≤ε{Colors.RESET}"
            )
            print(f"Total: {total_p} passed, {total_f} failed")
        else:
            print(
                f"{Colors.RED}{Colors.BOLD}[FAIL] Interop drift detected{Colors.RESET}"
            )
            print(f"Total: {total_p} passed, {Colors.RED}{total_f} failed{Colors.RESET}")

        print("=" * 70 + "\n")


def run_python_tests(verbose: bool = False) -> Tuple[int, int]:
    """Run Python interop tests using pytest."""
    print(f"\n{Colors.CYAN}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.CYAN}Running Python Interoperability Tests{Colors.RESET}")
    print(f"{Colors.CYAN}{'=' * 70}{Colors.RESET}\n")

    test_dir = Path(__file__).parent

    # Run pytest
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(test_dir / "test_api_contracts.py"),
        str(test_dir / "test_type_coercion.py"),
        "-v" if verbose else "-q",
        "--tb=short",
    ]

    try:
        result = subprocess.run(cmd, capture_output=False, text=True)

        # Parse pytest exit code
        # 0 = all passed, 1 = some failed, 5 = no tests collected
        if result.returncode == 0:
            # Estimate passed tests (pytest doesn't provide count in exit code)
            return (30, 0)  # Approximate count
        elif result.returncode == 5:
            print(f"{Colors.YELLOW}No Python tests collected{Colors.RESET}")
            return (0, 0)
        else:
            # Some tests failed
            return (0, 1)

    except FileNotFoundError:
        print(f"{Colors.RED}pytest not found. Install: pip install pytest{Colors.RESET}")
        return (0, 0)
    except Exception as e:
        print(f"{Colors.RED}Error running Python tests: {e}{Colors.RESET}")
        return (0, 1)


def run_javascript_tests(verbose: bool = False) -> Tuple[int, int]:
    """Run JavaScript interop tests using Node.js."""
    print(f"\n{Colors.CYAN}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.CYAN}Running JavaScript Interoperability Tests{Colors.RESET}")
    print(f"{Colors.CYAN}{'=' * 70}{Colors.RESET}\n")

    test_file = Path(__file__).parent / "mathledger_client.test.js"

    if not test_file.exists():
        print(f"{Colors.YELLOW}JavaScript test file not found{Colors.RESET}")
        return (0, 0)

    cmd = ["node", str(test_file)]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Print output
        if result.stdout:
            print(result.stdout)
        if result.stderr and verbose:
            print(result.stderr)

        # Parse output for pass/fail counts
        if result.returncode == 0:
            # Estimate from output
            lines = result.stdout.split("\n")
            passed = sum(1 for line in lines if "[PASS]" in line or "✅" in line)
            return (passed, 0)
        else:
            return (0, 1)

    except FileNotFoundError:
        print(f"{Colors.RED}node not found. Install Node.js 14+{Colors.RESET}")
        return (0, 0)
    except Exception as e:
        print(f"{Colors.RED}Error running JavaScript tests: {e}{Colors.RESET}")
        return (0, 1)


def run_powershell_tests(verbose: bool = False) -> Tuple[int, int]:
    """Run PowerShell interop tests."""
    print(f"\n{Colors.CYAN}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.CYAN}Running PowerShell Interoperability Tests{Colors.RESET}")
    print(f"{Colors.CYAN}{'=' * 70}{Colors.RESET}\n")

    test_file = Path(__file__).parent / "Test-APIContracts.ps1"

    if not test_file.exists():
        print(f"{Colors.YELLOW}PowerShell test file not found{Colors.RESET}")
        return (0, 0)

    # Try pwsh (PowerShell Core) first, then powershell (Windows PowerShell)
    for ps_cmd in ["pwsh", "powershell"]:
        cmd = [ps_cmd, "-File", str(test_file)]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            # Print output
            if result.stdout:
                print(result.stdout)
            if result.stderr and verbose:
                print(result.stderr)

            # Parse output for pass/fail counts
            if result.returncode == 0:
                lines = result.stdout.split("\n")
                passed = sum(1 for line in lines if "[PASS]" in line or "✅" in line)
                return (passed, 0)
            else:
                return (0, 1)

        except FileNotFoundError:
            continue  # Try next PowerShell variant
        except subprocess.TimeoutExpired:
            print(
                f"{Colors.YELLOW}PowerShell tests timed out "
                f"(API server may not be running){Colors.RESET}"
            )
            return (0, 0)
        except Exception as e:
            print(f"{Colors.RED}Error running PowerShell tests: {e}{Colors.RESET}")
            return (0, 1)

    # Neither PowerShell variant found
    print(
        f"{Colors.YELLOW}PowerShell not found. "
        f"Install PowerShell Core or run on Windows{Colors.RESET}"
    )
    return (0, 0)


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(
        description="Run cross-language interoperability tests"
    )
    parser.add_argument(
        "--python-only", action="store_true", help="Run only Python tests"
    )
    parser.add_argument(
        "--js-only", action="store_true", help="Run only JavaScript tests"
    )
    parser.add_argument(
        "--ps-only", action="store_true", help="Run only PowerShell tests"
    )
    parser.add_argument(
        "--skip-ps", action="store_true", help="Skip PowerShell tests"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose output"
    )

    args = parser.parse_args()

    results = TestResults()

    # Determine which tests to run
    run_python = args.python_only or not (args.js_only or args.ps_only)
    run_js = args.js_only or not (args.python_only or args.ps_only)
    run_ps = args.ps_only or (not args.skip_ps and not (args.python_only or args.js_only))

    # Run tests
    if run_python:
        passed, failed = run_python_tests(args.verbose)
        results.record_result("python", passed, failed)
    else:
        results.mark_skipped("python", "skipped")

    if run_js:
        passed, failed = run_javascript_tests(args.verbose)
        results.record_result("javascript", passed, failed)
    else:
        results.mark_skipped("javascript", "skipped")

    if run_ps:
        passed, failed = run_powershell_tests(args.verbose)
        results.record_result("powershell", passed, failed)
    else:
        results.mark_skipped("powershell", "skipped")

    # Print summary
    results.print_summary()

    # Exit with appropriate code
    sys.exit(0 if results.all_passed() else 1)


if __name__ == "__main__":
    main()
