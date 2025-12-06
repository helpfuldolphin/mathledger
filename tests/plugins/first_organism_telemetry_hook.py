"""
First Organism Test Telemetry Plugin.

A pytest plugin that captures First Organism test results and emits
telemetry metrics to Redis for consumption by the Metrics Oracle (Cursor K).

This plugin:
1. Hooks into pytest's test collection to identify first_organism marked tests
2. Records test timing, success/failure, and attestation data
3. Emits telemetry via FirstOrganismTelemetry on test completion
4. Extracts H_t from test stdout/fixtures when available

Usage:
    # Auto-registered in conftest.py or via pytest_plugins
    pytest -m first_organism

    # Enable even without Redis (will log but not emit)
    FIRST_ORGANISM_TELEMETRY_DRY_RUN=1 pytest -m first_organism

Metrics emitted:
    - duration_seconds: Test wall-clock duration
    - ht_hash: Composite root H_t (extracted from test output or fixtures)
    - abstention_count: Number of abstentions observed
    - success: Test pass/fail status
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

try:
    from backend.metrics.first_organism_telemetry import (
        FirstOrganismTelemetry,
        FirstOrganismRunResult,
        emit_first_organism_metrics,
    )
    TELEMETRY_AVAILABLE = True
except ImportError:
    TELEMETRY_AVAILABLE = False


# Pattern to extract H_t from test output
# Matches: "[PASS] FIRST ORGANISM ALIVE H_t=abc123..."
HT_PATTERN = re.compile(r"H_t=([a-fA-F0-9]{12,64})")

# Pattern to extract abstention count from test output
ABSTENTION_PATTERN = re.compile(r"[Aa]bsten(?:tion|ed)[^0-9]*(\d+)")


@dataclass
class FirstOrganismTestResult:
    """Captured result from a First Organism test."""

    nodeid: str
    duration_seconds: float
    passed: bool
    ht_hash: str = ""
    abstention_count: int = 0
    stdout: str = ""
    stderr: str = ""
    fixture_data: Dict[str, Any] = field(default_factory=dict)


class FirstOrganismTelemetryHook:
    """
    Pytest plugin for First Organism telemetry collection.

    This plugin captures test results for tests marked with @pytest.mark.first_organism
    and emits metrics to Redis for trend tracking.
    """

    def __init__(self):
        self._results: List[FirstOrganismTestResult] = []
        self._current_test_start: Optional[float] = None
        self._current_test_nodeid: Optional[str] = None
        self._telemetry: Optional[FirstOrganismTelemetry] = None
        self._dry_run = os.getenv("FIRST_ORGANISM_TELEMETRY_DRY_RUN", "0") == "1"

        if TELEMETRY_AVAILABLE and not self._dry_run:
            self._telemetry = FirstOrganismTelemetry()

    def _is_first_organism_test(self, item: pytest.Item) -> bool:
        """Check if test is marked as first_organism."""
        return item.get_closest_marker("first_organism") is not None

    def _extract_ht_from_output(self, output: str) -> str:
        """Extract H_t hash from test output."""
        match = HT_PATTERN.search(output)
        return match.group(1) if match else ""

    def _extract_abstention_count(self, output: str) -> int:
        """Extract abstention count from test output."""
        match = ABSTENTION_PATTERN.search(output)
        return int(match.group(1)) if match else 0

    def pytest_runtest_setup(self, item: pytest.Item) -> None:
        """Called before each test setup."""
        if self._is_first_organism_test(item):
            self._current_test_start = time.time()
            self._current_test_nodeid = item.nodeid

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item: pytest.Item, call: pytest.CallInfo):
        """Called after each test phase (setup, call, teardown)."""
        outcome = yield
        report = outcome.get_result()

        if call.when != "call":
            return

        if not self._is_first_organism_test(item):
            return

        if self._current_test_start is None:
            return

        duration = time.time() - self._current_test_start
        passed = report.passed

        # Try to get captured output from the report
        stdout = ""
        stderr = ""
        if hasattr(report, "capstdout"):
            stdout = report.capstdout or ""
        if hasattr(report, "capstderr"):
            stderr = report.capstderr or ""

        # Also try sections
        if hasattr(report, "sections"):
            for name, content in report.sections:
                if "stdout" in name.lower():
                    stdout += content
                if "stderr" in name.lower():
                    stderr += content

        combined_output = f"{stdout}\n{stderr}"

        # Extract metrics from output
        ht_hash = self._extract_ht_from_output(combined_output)
        abstention_count = self._extract_abstention_count(combined_output)

        # Try to get fixture data if available
        fixture_data = {}
        if hasattr(item, "funcargs"):
            ctx = item.funcargs.get("first_organism_attestation_context")
            if ctx and isinstance(ctx, dict):
                attestation = ctx.get("attestation")
                if attestation:
                    ht_hash = ht_hash or getattr(attestation, "composite_root", "")[:16]
                    fixture_data["attestation"] = {
                        "composite_root": getattr(attestation, "composite_root", ""),
                        "reasoning_root": getattr(attestation, "reasoning_root", ""),
                        "ui_root": getattr(attestation, "ui_root", ""),
                    }
                # Extract abstention count from derivation outcome
                outcome_data = ctx.get("derivation_outcome")
                if outcome_data and hasattr(outcome_data, "abstained"):
                    abstention_count = max(abstention_count, len(getattr(outcome_data, "abstained", [])))

        result = FirstOrganismTestResult(
            nodeid=item.nodeid,
            duration_seconds=duration,
            passed=passed,
            ht_hash=ht_hash,
            abstention_count=abstention_count,
            stdout=stdout,
            stderr=stderr,
            fixture_data=fixture_data,
        )

        self._results.append(result)
        self._current_test_start = None
        self._current_test_nodeid = None

    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        """Called after all tests complete. Emit aggregated telemetry."""
        fo_results = [r for r in self._results if r.nodeid]

        if not fo_results:
            return

        # Aggregate metrics across all First Organism tests
        total_duration = sum(r.duration_seconds for r in fo_results)
        total_abstentions = sum(r.abstention_count for r in fo_results)
        all_passed = all(r.passed for r in fo_results)
        test_count = len(fo_results)

        # Use the most recent H_t or generate a summary
        ht_hash = ""
        for result in reversed(fo_results):
            if result.ht_hash:
                ht_hash = result.ht_hash
                break

        # Log summary
        status_str = "SUCCESS" if all_passed else "FAILURE"
        print(f"\n[TELEMETRY] First Organism Summary:")
        print(f"  Tests: {test_count}")
        print(f"  Duration: {total_duration:.2f}s")
        print(f"  Abstentions: {total_abstentions}")
        print(f"  H_t: {ht_hash[:16] if ht_hash else 'N/A'}")
        print(f"  Status: {status_str}")

        # Emit telemetry
        if self._telemetry and self._telemetry.available:
            emitted = emit_first_organism_metrics(
                duration_seconds=total_duration,
                ht_hash=ht_hash,
                abstention_count=total_abstentions,
                success=all_passed,
                metadata={
                    "test_count": test_count,
                    "test_results": [
                        {
                            "nodeid": r.nodeid,
                            "passed": r.passed,
                            "duration": r.duration_seconds,
                        }
                        for r in fo_results
                    ],
                },
            )
            if emitted:
                print("  [OK] Telemetry emitted to Redis")
            else:
                print("  [WARN] Telemetry emission failed")
        elif self._dry_run:
            print("  [DRY-RUN] Telemetry not emitted (dry run mode)")
        else:
            print("  [SKIP] Telemetry not available")


# Register the plugin
def pytest_configure(config: pytest.Config) -> None:
    """Register the First Organism telemetry plugin."""
    config.pluginmanager.register(FirstOrganismTelemetryHook(), "first_organism_telemetry")


def pytest_unconfigure(config: pytest.Config) -> None:
    """Unregister the plugin."""
    plugin = config.pluginmanager.get_plugin("first_organism_telemetry")
    if plugin:
        config.pluginmanager.unregister(plugin)
