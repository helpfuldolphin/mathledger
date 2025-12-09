"""
Metric Conformance Snapshot — Live Drift Detection & Promotion Guard

This module provides utilities for:
1. Building conformance snapshots from test results
2. Saving/loading snapshots to/from disk
3. Comparing snapshots to detect drift
4. Enforcing promotion guards based on conformance levels

Version: 1.0.0
Status: ENFORCED
Phase: II U2
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


__all__ = [
    # Enums
    "ConformanceLevel",
    "DriftClass",
    # Data classes
    "MetricConformanceResult",
    "ConformanceSnapshot",
    "SnapshotComparison",
    "DriftEvent",
    "DriftLedger",
    "PromotionReadiness",
    "GlobalHealthStatus",
    # Snapshot building
    "build_conformance_snapshot",
    "save_conformance_snapshot",
    "load_conformance_snapshot",
    # Comparison & promotion
    "compare_conformance_snapshots",
    "can_promote_metric",
    # Phase III: Drift Memory & Promotion Oracle
    "build_metric_drift_ledger",
    "compute_metric_promotion_readiness",
    "summarize_metric_conformance_for_global_health",
    # Phase IV: Conformance Drift Compass & Promotion Dashboard
    "build_metric_drift_compass",
    "summarize_conformance_for_promotion_dashboard",
    "build_conformance_director_panel",
    # Phase V: Metrics × Budget × Policy Triangulation
    "build_metric_budget_joint_view",
    "summarize_conformance_for_global_console",
    "attach_policy_telemetry_hint",
    # Phase VI: Metrics + Budget as One Governance Surface
    "summarize_conformance_and_budget_for_global_console",
    "to_governance_signal_for_metrics",
    # Report rendering
    "render_conformance_report",
    # Helpers
    "get_git_sha",
]


class ConformanceLevel(IntEnum):
    """Conformance test levels per METRIC_CONFORMANCE_SUITE_SPEC.md"""
    L0 = 0  # Minimal: type + determinism smoke tests
    L1 = 1  # Standard: all invariants
    L2 = 2  # Full: invariants + boundary + monotonicity
    L3 = 3  # Exhaustive: full + stress tests


class DriftClass(IntEnum):
    """Metric drift classification per METRIC_CORRECTNESS_CONTRACT.md"""
    D0_COSMETIC = 0           # Documentation, naming, logging changes
    D1_ADDITIVE = 1           # New optional parameters with defaults
    D2_BEHAVIORAL_COMPAT = 2  # Internal algorithm change, same outputs
    D3_BEHAVIORAL_BREAK = 3   # Output changes for existing inputs
    D4_SCHEMA_BREAK = 4       # Parameter or return type changes
    D5_SEMANTIC_BREAK = 5     # Fundamental metric meaning changes


# Mapping: drift class → minimum required conformance level
DRIFT_TO_CONFORMANCE: Dict[DriftClass, ConformanceLevel] = {
    DriftClass.D0_COSMETIC: ConformanceLevel.L0,
    DriftClass.D1_ADDITIVE: ConformanceLevel.L1,
    DriftClass.D2_BEHAVIORAL_COMPAT: ConformanceLevel.L2,
    DriftClass.D3_BEHAVIORAL_BREAK: ConformanceLevel.L3,
    DriftClass.D4_SCHEMA_BREAK: ConformanceLevel.L3,
    # D5 is special: no conformance level allows shipping
}


@dataclass
class MetricConformanceResult:
    """Result of conformance tests for a single metric."""
    metric_name: str
    tests_passed: Dict[str, bool]  # test_id -> passed
    tests_by_level: Dict[str, List[str]] = field(default_factory=dict)  # L0/L1/L2/L3 -> [test_ids]

    def passed_at_level(self, level: ConformanceLevel) -> int:
        """Count tests passed at the given level."""
        level_key = f"L{level.value}"
        test_ids = self.tests_by_level.get(level_key, [])
        return sum(1 for tid in test_ids if self.tests_passed.get(tid, False))

    def total_at_level(self, level: ConformanceLevel) -> int:
        """Count total tests at the given level."""
        level_key = f"L{level.value}"
        return len(self.tests_by_level.get(level_key, []))

    def level_pass_rate(self, level: ConformanceLevel) -> float:
        """Get pass rate (0.0-1.0) for the given level."""
        total = self.total_at_level(level)
        if total == 0:
            return 1.0  # No tests = vacuously passing
        return self.passed_at_level(level) / total


@dataclass
class ConformanceSnapshot:
    """
    Point-in-time snapshot of metric conformance test results.

    Used for drift detection and promotion gating.
    """
    # Identity
    snapshot_id: str
    timestamp: str  # ISO 8601 format
    git_sha: Optional[str]

    # Results per metric
    metrics: Dict[str, Dict[str, Any]]  # metric_name -> {tests_passed_by_level, ...}

    # Summary statistics
    total_tests: int
    total_passed: int
    conformance_levels: Dict[str, int]  # L0/L1/L2/L3 -> passed count

    # Metadata
    version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "snapshot_id": self.snapshot_id,
            "timestamp": self.timestamp,
            "git_sha": self.git_sha,
            "metrics": self.metrics,
            "total_tests": self.total_tests,
            "total_passed": self.total_passed,
            "conformance_levels": self.conformance_levels,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConformanceSnapshot":
        """Create from dictionary."""
        return cls(
            snapshot_id=data["snapshot_id"],
            timestamp=data["timestamp"],
            git_sha=data.get("git_sha"),
            metrics=data["metrics"],
            total_tests=data["total_tests"],
            total_passed=data["total_passed"],
            conformance_levels=data["conformance_levels"],
            version=data.get("version", "1.0.0"),
        )


@dataclass
class SnapshotComparison:
    """Result of comparing two conformance snapshots."""
    old_snapshot_id: str
    new_snapshot_id: str

    # Level changes
    regressed_levels: Dict[str, Tuple[int, int]]  # level -> (old_passed, new_passed)
    improved_levels: Dict[str, Tuple[int, int]]   # level -> (old_passed, new_passed)
    unchanged_levels: Dict[str, int]              # level -> passed_count

    # Test changes
    tests_lost: List[str]    # test_ids that were passing but now fail
    tests_gained: List[str]  # test_ids that were failing but now pass

    # Summary
    is_regression: bool
    regression_severity: Optional[str]  # "minor", "major", "critical"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "old_snapshot_id": self.old_snapshot_id,
            "new_snapshot_id": self.new_snapshot_id,
            "regressed_levels": {k: list(v) for k, v in self.regressed_levels.items()},
            "improved_levels": {k: list(v) for k, v in self.improved_levels.items()},
            "unchanged_levels": self.unchanged_levels,
            "tests_lost": self.tests_lost,
            "tests_gained": self.tests_gained,
            "is_regression": self.is_regression,
            "regression_severity": self.regression_severity,
        }


def get_git_sha() -> Optional[str]:
    """
    Get current git SHA from environment or git command.

    Checks in order:
    1. GITHUB_SHA environment variable (GitHub Actions)
    2. CI_COMMIT_SHA environment variable (GitLab CI)
    3. GIT_SHA environment variable (generic)
    4. Falls back to None if unavailable
    """
    # Check common CI environment variables
    for env_var in ["GITHUB_SHA", "CI_COMMIT_SHA", "GIT_SHA", "GIT_COMMIT"]:
        sha = os.environ.get(env_var)
        if sha:
            return sha[:12]  # Short SHA

    # Try running git command (optional, don't fail if unavailable)
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass

    return None


def _compute_snapshot_id(
    metrics: Dict[str, Dict[str, Any]],
    timestamp: str,
    git_sha: Optional[str],
) -> str:
    """Compute deterministic snapshot ID from contents."""
    # Create canonical representation for hashing
    canonical = json.dumps(
        {
            "metrics": metrics,
            "timestamp": timestamp,
            "git_sha": git_sha,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def build_conformance_snapshot(
    results: List[MetricConformanceResult],
    timestamp: Optional[str] = None,
    git_sha: Optional[str] = None,
) -> ConformanceSnapshot:
    """
    Build a conformance snapshot from test results.

    Args:
        results: List of MetricConformanceResult objects
        timestamp: ISO 8601 timestamp (defaults to now)
        git_sha: Git commit SHA (defaults to auto-detect)

    Returns:
        ConformanceSnapshot capturing the current state
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    if git_sha is None:
        git_sha = get_git_sha()

    # Build metrics dictionary
    metrics: Dict[str, Dict[str, Any]] = {}
    total_tests = 0
    total_passed = 0
    level_counts: Dict[str, int] = {"L0": 0, "L1": 0, "L2": 0, "L3": 0}

    for result in results:
        metric_data: Dict[str, Any] = {
            "tests_passed": result.tests_passed,
            "tests_by_level": result.tests_by_level,
            "passed_by_level": {},
            "total_by_level": {},
        }

        for level in ConformanceLevel:
            level_key = f"L{level.value}"
            passed = result.passed_at_level(level)
            total = result.total_at_level(level)

            metric_data["passed_by_level"][level_key] = passed
            metric_data["total_by_level"][level_key] = total

            level_counts[level_key] += passed
            total_tests += total
            total_passed += passed

        metrics[result.metric_name] = metric_data

    # Compute deterministic snapshot ID
    snapshot_id = _compute_snapshot_id(metrics, timestamp, git_sha)

    return ConformanceSnapshot(
        snapshot_id=snapshot_id,
        timestamp=timestamp,
        git_sha=git_sha,
        metrics=metrics,
        total_tests=total_tests,
        total_passed=total_passed,
        conformance_levels=level_counts,
        version="1.0.0",
    )


def save_conformance_snapshot(path: str | Path, snapshot: ConformanceSnapshot) -> None:
    """
    Save a conformance snapshot to disk as JSON.

    Args:
        path: File path to write to
        snapshot: Snapshot to save
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(snapshot.to_dict(), f, indent=2, sort_keys=True)


def load_conformance_snapshot(path: str | Path) -> ConformanceSnapshot:
    """
    Load a conformance snapshot from disk.

    Args:
        path: File path to read from

    Returns:
        ConformanceSnapshot loaded from file

    Raises:
        FileNotFoundError: If file does not exist
        json.JSONDecodeError: If file is not valid JSON
        KeyError: If required fields are missing
    """
    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return ConformanceSnapshot.from_dict(data)


def compare_conformance_snapshots(
    old: ConformanceSnapshot,
    new: ConformanceSnapshot,
) -> SnapshotComparison:
    """
    Compare two conformance snapshots to detect drift.

    Args:
        old: Previous snapshot (baseline)
        new: Current snapshot (candidate)

    Returns:
        SnapshotComparison with drift analysis
    """
    regressed_levels: Dict[str, Tuple[int, int]] = {}
    improved_levels: Dict[str, Tuple[int, int]] = {}
    unchanged_levels: Dict[str, int] = {}

    # Compare level counts
    for level_key in ["L0", "L1", "L2", "L3"]:
        old_count = old.conformance_levels.get(level_key, 0)
        new_count = new.conformance_levels.get(level_key, 0)

        if new_count < old_count:
            regressed_levels[level_key] = (old_count, new_count)
        elif new_count > old_count:
            improved_levels[level_key] = (old_count, new_count)
        else:
            unchanged_levels[level_key] = old_count

    # Track individual test changes
    tests_lost: List[str] = []
    tests_gained: List[str] = []

    # Collect all test results from both snapshots
    old_tests: Dict[str, bool] = {}
    new_tests: Dict[str, bool] = {}

    for metric_name, metric_data in old.metrics.items():
        for test_id, passed in metric_data.get("tests_passed", {}).items():
            old_tests[f"{metric_name}::{test_id}"] = passed

    for metric_name, metric_data in new.metrics.items():
        for test_id, passed in metric_data.get("tests_passed", {}).items():
            new_tests[f"{metric_name}::{test_id}"] = passed

    # Find lost and gained tests
    all_tests = set(old_tests.keys()) | set(new_tests.keys())
    for test_id in all_tests:
        old_passed = old_tests.get(test_id, False)
        new_passed = new_tests.get(test_id, False)

        if old_passed and not new_passed:
            tests_lost.append(test_id)
        elif not old_passed and new_passed:
            tests_gained.append(test_id)

    # Determine regression severity
    is_regression = len(regressed_levels) > 0 or len(tests_lost) > 0
    regression_severity: Optional[str] = None

    if is_regression:
        # Critical: L0 or L1 regression
        if "L0" in regressed_levels or "L1" in regressed_levels:
            regression_severity = "critical"
        # Major: L2 regression
        elif "L2" in regressed_levels:
            regression_severity = "major"
        # Minor: L3 only or just test count changes
        else:
            regression_severity = "minor"

    return SnapshotComparison(
        old_snapshot_id=old.snapshot_id,
        new_snapshot_id=new.snapshot_id,
        regressed_levels=regressed_levels,
        improved_levels=improved_levels,
        unchanged_levels=unchanged_levels,
        tests_lost=sorted(tests_lost),
        tests_gained=sorted(tests_gained),
        is_regression=is_regression,
        regression_severity=regression_severity,
    )


def can_promote_metric(
    old_snapshot: ConformanceSnapshot,
    new_snapshot: ConformanceSnapshot,
    required_level: ConformanceLevel = ConformanceLevel.L1,
    allow_minor_regression: bool = False,
) -> Tuple[bool, str]:
    """
    Determine if a metric change can be promoted based on conformance.

    This is the core promotion guard used in CI pipelines.

    Args:
        old_snapshot: Baseline snapshot (e.g., main branch)
        new_snapshot: Candidate snapshot (e.g., PR branch)
        required_level: Minimum conformance level required (default L1)
        allow_minor_regression: If True, allow L3-only regressions

    Returns:
        Tuple of (can_promote: bool, reason: str)

    Examples:
        >>> can_promote_metric(baseline, candidate)
        (True, 'No conformance regression detected')

        >>> can_promote_metric(baseline, candidate_with_l2_regression)
        (False, 'Conformance regression: L2 (28 -> 26 tests)')
    """
    comparison = compare_conformance_snapshots(old_snapshot, new_snapshot)

    # Check for any regression
    if not comparison.is_regression:
        return True, "No conformance regression detected"

    # Check regression severity
    if comparison.regression_severity == "critical":
        regressed = ", ".join(
            f"{k} ({v[0]} -> {v[1]} tests)"
            for k, v in comparison.regressed_levels.items()
            if k in ("L0", "L1")
        )
        return False, f"Critical conformance regression: {regressed}"

    if comparison.regression_severity == "major":
        regressed = ", ".join(
            f"{k} ({v[0]} -> {v[1]} tests)"
            for k, v in comparison.regressed_levels.items()
            if k == "L2"
        )
        return False, f"Major conformance regression: {regressed}"

    # Minor regression (L3 only or test count changes)
    if comparison.regression_severity == "minor":
        if allow_minor_regression:
            return True, "Minor conformance regression allowed (L3 only)"

        regressed = ", ".join(
            f"{k} ({v[0]} -> {v[1]} tests)"
            for k, v in comparison.regressed_levels.items()
        )
        tests_lost = len(comparison.tests_lost)
        return False, f"Minor conformance regression: {regressed}, {tests_lost} tests lost"

    return True, "Conformance check passed"


def render_conformance_report(
    snapshots: Sequence[ConformanceSnapshot],
    title: str = "Metric Conformance Report",
    highlight_regressions: bool = True,
) -> str:
    """
    Render a human-readable Markdown report from conformance snapshots.

    Args:
        snapshots: Sequence of snapshots to include (most recent last)
        title: Report title
        highlight_regressions: If True, mark regressions with warning emoji

    Returns:
        Markdown-formatted report string
    """
    lines: List[str] = []

    # Header
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")

    if not snapshots:
        lines.append("*No snapshots provided.*")
        return "\n".join(lines)

    # Summary table header
    lines.append("## Conformance Summary")
    lines.append("")
    lines.append("| Snapshot | Git SHA | L0 | L1 | L2 | L3 | Total | Status |")
    lines.append("|----------|---------|----|----|----|----|-------|--------|")

    # Build rows with regression detection
    prev_snapshot: Optional[ConformanceSnapshot] = None
    for snapshot in snapshots:
        sha = snapshot.git_sha or "N/A"
        l0 = snapshot.conformance_levels.get("L0", 0)
        l1 = snapshot.conformance_levels.get("L1", 0)
        l2 = snapshot.conformance_levels.get("L2", 0)
        l3 = snapshot.conformance_levels.get("L3", 0)
        total = f"{snapshot.total_passed}/{snapshot.total_tests}"

        # Check for regression
        status = "OK"
        if prev_snapshot and highlight_regressions:
            comparison = compare_conformance_snapshots(prev_snapshot, snapshot)
            if comparison.is_regression:
                if comparison.regression_severity == "critical":
                    status = "CRITICAL"
                elif comparison.regression_severity == "major":
                    status = "MAJOR"
                else:
                    status = "MINOR"

        lines.append(f"| {snapshot.snapshot_id[:8]} | {sha} | {l0} | {l1} | {l2} | {l3} | {total} | {status} |")
        prev_snapshot = snapshot

    lines.append("")

    # Detailed metrics section (for latest snapshot)
    latest = snapshots[-1]
    lines.append("## Metric Details (Latest)")
    lines.append("")
    lines.append("| Metric | L0 | L1 | L2 | L3 |")
    lines.append("|--------|----|----|----|----|")

    for metric_name, metric_data in sorted(latest.metrics.items()):
        passed_by_level = metric_data.get("passed_by_level", {})
        total_by_level = metric_data.get("total_by_level", {})

        l0 = f"{passed_by_level.get('L0', 0)}/{total_by_level.get('L0', 0)}"
        l1 = f"{passed_by_level.get('L1', 0)}/{total_by_level.get('L1', 0)}"
        l2 = f"{passed_by_level.get('L2', 0)}/{total_by_level.get('L2', 0)}"
        l3 = f"{passed_by_level.get('L3', 0)}/{total_by_level.get('L3', 0)}"

        lines.append(f"| {metric_name} | {l0} | {l1} | {l2} | {l3} |")

    lines.append("")

    # Drift comparison section (if multiple snapshots)
    if len(snapshots) >= 2:
        lines.append("## Drift Analysis")
        lines.append("")

        old_snap = snapshots[-2]
        new_snap = snapshots[-1]
        comparison = compare_conformance_snapshots(old_snap, new_snap)

        lines.append(f"Comparing: `{old_snap.snapshot_id[:8]}` -> `{new_snap.snapshot_id[:8]}`")
        lines.append("")

        if comparison.is_regression:
            lines.append(f"**Regression Detected**: {comparison.regression_severity}")
            lines.append("")

            if comparison.regressed_levels:
                lines.append("### Regressed Levels")
                for level, (old_c, new_c) in comparison.regressed_levels.items():
                    lines.append(f"- {level}: {old_c} -> {new_c} (-{old_c - new_c})")
                lines.append("")

            if comparison.tests_lost:
                lines.append("### Tests Lost")
                for test_id in comparison.tests_lost[:10]:  # Limit to 10
                    lines.append(f"- `{test_id}`")
                if len(comparison.tests_lost) > 10:
                    lines.append(f"- ... and {len(comparison.tests_lost) - 10} more")
                lines.append("")
        else:
            lines.append("No regression detected.")
            lines.append("")

        if comparison.improved_levels:
            lines.append("### Improved Levels")
            for level, (old_c, new_c) in comparison.improved_levels.items():
                lines.append(f"- {level}: {old_c} -> {new_c} (+{new_c - old_c})")
            lines.append("")

        if comparison.tests_gained:
            lines.append("### Tests Gained")
            for test_id in comparison.tests_gained[:10]:
                lines.append(f"- `{test_id}`")
            if len(comparison.tests_gained) > 10:
                lines.append(f"- ... and {len(comparison.tests_gained) - 10} more")
            lines.append("")

    # Footer
    lines.append("---")
    lines.append("*Generated by MathLedger Metric Conformance System*")

    return "\n".join(lines)


# =============================================================================
# PHASE III: Drift Memory & Promotion Oracle
# =============================================================================

@dataclass
class DriftEvent:
    """A single drift event in the conformance history."""
    metric: str
    from_level: str  # e.g., "L2"
    to_level: str    # e.g., "L1"
    severity: str    # "critical", "major", "minor"
    snapshot_id: str
    timestamp: str
    tests_lost: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "metric": self.metric,
            "from_level": self.from_level,
            "to_level": self.to_level,
            "severity": self.severity,
            "snapshot_id": self.snapshot_id,
            "timestamp": self.timestamp,
            "tests_lost": self.tests_lost,
        }


@dataclass
class DriftLedger:
    """
    Accumulated drift history across multiple snapshots.

    Tracks all drift events and identifies metrics with repeated regressions.
    """
    schema_version: str
    drift_events: List[DriftEvent]
    metrics_with_repeated_regressions: Dict[str, int]  # metric -> regression count
    total_regressions: int
    total_improvements: int
    first_snapshot_id: Optional[str]
    last_snapshot_id: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "schema_version": self.schema_version,
            "drift_events": [e.to_dict() for e in self.drift_events],
            "metrics_with_repeated_regressions": self.metrics_with_repeated_regressions,
            "total_regressions": self.total_regressions,
            "total_improvements": self.total_improvements,
            "first_snapshot_id": self.first_snapshot_id,
            "last_snapshot_id": self.last_snapshot_id,
        }


@dataclass
class PromotionReadiness:
    """Result of promotion readiness evaluation."""
    ready_for_promotion: bool
    blocking_regressions: List[Dict[str, Any]]  # List of blocking regression details
    justification: str
    comparison_details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "ready_for_promotion": self.ready_for_promotion,
            "blocking_regressions": self.blocking_regressions,
            "justification": self.justification,
            "comparison_details": self.comparison_details,
        }


@dataclass
class GlobalHealthStatus:
    """Global health status derived from conformance snapshot."""
    weakest_metric: Optional[str]
    weakest_metric_pass_rate: float
    any_blockers: bool
    blocker_count: int
    overall_conformance_status: str  # "healthy", "degraded", "critical", "failing"
    metrics_by_status: Dict[str, List[str]]  # status -> [metric_names]
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "weakest_metric": self.weakest_metric,
            "weakest_metric_pass_rate": self.weakest_metric_pass_rate,
            "any_blockers": self.any_blockers,
            "blocker_count": self.blocker_count,
            "overall_conformance_status": self.overall_conformance_status,
            "metrics_by_status": self.metrics_by_status,
            "summary": self.summary,
        }


def _get_metric_level_from_snapshot(
    snapshot: ConformanceSnapshot,
    metric_name: str,
) -> Tuple[str, float]:
    """
    Get the effective conformance level and pass rate for a metric.

    Returns the highest level with 100% pass rate, or the lowest level
    if none have 100%.
    """
    metric_data = snapshot.metrics.get(metric_name, {})
    passed_by_level = metric_data.get("passed_by_level", {})
    total_by_level = metric_data.get("total_by_level", {})

    best_level = "L0"
    best_rate = 0.0

    for level in ["L0", "L1", "L2", "L3"]:
        passed = passed_by_level.get(level, 0)
        total = total_by_level.get(level, 0)

        if total == 0:
            continue

        rate = passed / total
        if rate == 1.0:
            best_level = level
            best_rate = rate
        elif rate > best_rate:
            best_rate = rate
            # Don't update best_level unless 100%

    return best_level, best_rate


def _compute_metric_pass_rate(
    snapshot: ConformanceSnapshot,
    metric_name: str,
) -> float:
    """Compute overall pass rate for a single metric."""
    metric_data = snapshot.metrics.get(metric_name, {})
    passed_by_level = metric_data.get("passed_by_level", {})
    total_by_level = metric_data.get("total_by_level", {})

    total_passed = sum(passed_by_level.values())
    total_tests = sum(total_by_level.values())

    if total_tests == 0:
        return 1.0  # Vacuously passing

    return total_passed / total_tests


def build_metric_drift_ledger(
    history_snapshots: Sequence[ConformanceSnapshot],
) -> DriftLedger:
    """
    Build a drift ledger from a sequence of historical snapshots.

    Analyzes the history to detect all drift events and identify
    metrics with repeated regressions.

    Args:
        history_snapshots: Sequence of snapshots in chronological order

    Returns:
        DriftLedger with accumulated drift history
    """
    drift_events: List[DriftEvent] = []
    regression_counts: Dict[str, int] = {}
    total_regressions = 0
    total_improvements = 0

    if not history_snapshots:
        return DriftLedger(
            schema_version="1.0.0",
            drift_events=[],
            metrics_with_repeated_regressions={},
            total_regressions=0,
            total_improvements=0,
            first_snapshot_id=None,
            last_snapshot_id=None,
        )

    # Compare consecutive snapshots
    for i in range(1, len(history_snapshots)):
        old_snap = history_snapshots[i - 1]
        new_snap = history_snapshots[i]

        # Get all metrics across both snapshots
        all_metrics = set(old_snap.metrics.keys()) | set(new_snap.metrics.keys())

        for metric_name in all_metrics:
            old_level, _ = _get_metric_level_from_snapshot(old_snap, metric_name)
            new_level, _ = _get_metric_level_from_snapshot(new_snap, metric_name)

            # Compare old vs new metric data
            old_data = old_snap.metrics.get(metric_name, {})
            new_data = new_snap.metrics.get(metric_name, {})

            old_passed = old_data.get("passed_by_level", {})
            new_passed = new_data.get("passed_by_level", {})

            # Calculate tests lost and gained for this metric
            tests_lost = 0
            tests_gained = 0
            for level in ["L0", "L1", "L2", "L3"]:
                old_count = old_passed.get(level, 0)
                new_count = new_passed.get(level, 0)
                if new_count < old_count:
                    tests_lost += old_count - new_count
                elif new_count > old_count:
                    tests_gained += new_count - old_count

            # Detect regression
            level_order = {"L0": 0, "L1": 1, "L2": 2, "L3": 3}
            if tests_lost > 0:
                # Determine severity
                severity = "minor"
                if level_order.get(new_level, 0) < level_order.get(old_level, 0):
                    if new_level in ("L0", "L1"):
                        severity = "critical"
                    elif new_level == "L2":
                        severity = "major"

                drift_events.append(DriftEvent(
                    metric=metric_name,
                    from_level=old_level,
                    to_level=new_level,
                    severity=severity,
                    snapshot_id=new_snap.snapshot_id,
                    timestamp=new_snap.timestamp,
                    tests_lost=tests_lost,
                ))

                regression_counts[metric_name] = regression_counts.get(metric_name, 0) + 1
                total_regressions += 1

            # Track improvements (tests gained)
            if tests_gained > 0:
                total_improvements += 1

    # Filter to metrics with repeated regressions (2+)
    repeated_regressions = {
        m: c for m, c in regression_counts.items() if c >= 2
    }

    return DriftLedger(
        schema_version="1.0.0",
        drift_events=drift_events,
        metrics_with_repeated_regressions=repeated_regressions,
        total_regressions=total_regressions,
        total_improvements=total_improvements,
        first_snapshot_id=history_snapshots[0].snapshot_id if history_snapshots else None,
        last_snapshot_id=history_snapshots[-1].snapshot_id if history_snapshots else None,
    )


def compute_metric_promotion_readiness(
    old_snapshot: ConformanceSnapshot,
    new_snapshot: ConformanceSnapshot,
    require_l1_pass: bool = True,
    allow_minor_regression: bool = False,
) -> PromotionReadiness:
    """
    Compute promotion readiness for a metric change.

    This is the Promotion Oracle that determines if a change is ready
    for promotion based on conformance comparison.

    Args:
        old_snapshot: Baseline snapshot (e.g., main branch)
        new_snapshot: Candidate snapshot (e.g., PR branch)
        require_l1_pass: Require all L1 tests to pass (default True)
        allow_minor_regression: Allow L3-only regressions (default False)

    Returns:
        PromotionReadiness with detailed analysis
    """
    comparison = compare_conformance_snapshots(old_snapshot, new_snapshot)
    blocking_regressions: List[Dict[str, Any]] = []

    # Check for L0/L1 regressions (always blocking)
    for level in ["L0", "L1"]:
        if level in comparison.regressed_levels:
            old_count, new_count = comparison.regressed_levels[level]
            blocking_regressions.append({
                "level": level,
                "old_count": old_count,
                "new_count": new_count,
                "severity": "critical",
                "reason": f"{level} regression: {old_count} -> {new_count} tests",
            })

    # Check for L2 regressions (blocking)
    if "L2" in comparison.regressed_levels:
        old_count, new_count = comparison.regressed_levels["L2"]
        blocking_regressions.append({
            "level": "L2",
            "old_count": old_count,
            "new_count": new_count,
            "severity": "major",
            "reason": f"L2 regression: {old_count} -> {new_count} tests",
        })

    # Check for L3 regressions (blocking unless allow_minor_regression)
    if "L3" in comparison.regressed_levels and not allow_minor_regression:
        old_count, new_count = comparison.regressed_levels["L3"]
        blocking_regressions.append({
            "level": "L3",
            "old_count": old_count,
            "new_count": new_count,
            "severity": "minor",
            "reason": f"L3 regression: {old_count} -> {new_count} tests",
        })

    # Check L1 pass requirement
    if require_l1_pass:
        l1_passed = new_snapshot.conformance_levels.get("L1", 0)
        # Count total L1 tests across all metrics
        l1_total = 0
        for metric_data in new_snapshot.metrics.values():
            l1_total += metric_data.get("total_by_level", {}).get("L1", 0)

        if l1_total > 0 and l1_passed < l1_total:
            blocking_regressions.append({
                "level": "L1",
                "old_count": l1_total,
                "new_count": l1_passed,
                "severity": "critical",
                "reason": f"L1 not fully passing: {l1_passed}/{l1_total} tests",
            })

    ready = len(blocking_regressions) == 0

    # Build justification
    if ready:
        if comparison.is_regression and allow_minor_regression:
            justification = "Ready for promotion with minor regression waiver"
        else:
            justification = "All conformance checks passed"
    else:
        severities = [b["severity"] for b in blocking_regressions]
        if "critical" in severities:
            justification = f"Blocked by {len(blocking_regressions)} critical regression(s)"
        elif "major" in severities:
            justification = f"Blocked by {len(blocking_regressions)} major regression(s)"
        else:
            justification = f"Blocked by {len(blocking_regressions)} minor regression(s)"

    return PromotionReadiness(
        ready_for_promotion=ready,
        blocking_regressions=blocking_regressions,
        justification=justification,
        comparison_details=comparison.to_dict(),
    )


def summarize_metric_conformance_for_global_health(
    snapshot: ConformanceSnapshot,
) -> GlobalHealthStatus:
    """
    Summarize metric conformance status for global health monitoring.

    Identifies the weakest metric, any blockers, and overall health status.

    Args:
        snapshot: Current conformance snapshot

    Returns:
        GlobalHealthStatus with health summary
    """
    if not snapshot.metrics:
        return GlobalHealthStatus(
            weakest_metric=None,
            weakest_metric_pass_rate=1.0,
            any_blockers=False,
            blocker_count=0,
            overall_conformance_status="healthy",
            metrics_by_status={"healthy": [], "degraded": [], "critical": [], "failing": []},
            summary="No metrics to evaluate",
        )

    metrics_by_status: Dict[str, List[str]] = {
        "healthy": [],
        "degraded": [],
        "critical": [],
        "failing": [],
    }

    weakest_metric: Optional[str] = None
    weakest_rate = 1.0
    blocker_count = 0

    for metric_name in snapshot.metrics:
        pass_rate = _compute_metric_pass_rate(snapshot, metric_name)

        # Update weakest metric
        if pass_rate < weakest_rate:
            weakest_rate = pass_rate
            weakest_metric = metric_name

        # Categorize metric
        if pass_rate == 1.0:
            metrics_by_status["healthy"].append(metric_name)
        elif pass_rate >= 0.9:
            metrics_by_status["degraded"].append(metric_name)
        elif pass_rate >= 0.5:
            metrics_by_status["critical"].append(metric_name)
            blocker_count += 1
        else:
            metrics_by_status["failing"].append(metric_name)
            blocker_count += 1

    # Determine overall status
    if metrics_by_status["failing"]:
        overall_status = "failing"
    elif metrics_by_status["critical"]:
        overall_status = "critical"
    elif metrics_by_status["degraded"]:
        overall_status = "degraded"
    else:
        overall_status = "healthy"

    any_blockers = blocker_count > 0

    # Build summary
    total_metrics = len(snapshot.metrics)
    healthy_count = len(metrics_by_status["healthy"])

    if overall_status == "healthy":
        summary = f"All {total_metrics} metrics passing at 100%"
    elif overall_status == "degraded":
        degraded_count = len(metrics_by_status["degraded"])
        summary = f"{healthy_count}/{total_metrics} healthy, {degraded_count} degraded (>90% passing)"
    elif overall_status == "critical":
        critical_count = len(metrics_by_status["critical"])
        summary = f"{blocker_count} blocker(s): {critical_count} critical (50-90% passing)"
    else:
        failing_count = len(metrics_by_status["failing"])
        summary = f"{blocker_count} blocker(s): {failing_count} failing (<50% passing)"

    return GlobalHealthStatus(
        weakest_metric=weakest_metric,
        weakest_metric_pass_rate=weakest_rate,
        any_blockers=any_blockers,
        blocker_count=blocker_count,
        overall_conformance_status=overall_status,
        metrics_by_status=metrics_by_status,
        summary=summary,
    )


# =============================================================================
# PHASE IV: Conformance Drift Compass & Promotion Dashboard
# =============================================================================

def build_metric_drift_compass(ledger: DriftLedger) -> Dict[str, Any]:
    """
    Build a drift compass from a drift ledger.

    The drift compass provides a high-level view of metric stability,
    identifying chronically regressing metrics and consistently improving ones.

    Args:
        ledger: DriftLedger from build_metric_drift_ledger()

    Returns:
        Dict with:
        - schema_version: Version string
        - metrics_with_chronic_regressions: List of metrics with 2+ regressions
        - metrics_consistently_improving: List of metrics with only improvements
        - compass_status: "STABLE" | "CAUTION" | "CRITICAL"
    """
    # Identify chronically regressing metrics (2+ regressions)
    chronic_regressions = list(ledger.metrics_with_repeated_regressions.keys())

    # Identify metrics that only improved (no regressions)
    # Build set of metrics that regressed
    regressed_metrics = {e.metric for e in ledger.drift_events}

    # Find metrics that had improvements but no regressions
    # We track this by looking at the ledger stats
    improving_metrics: List[str] = []

    # A metric is "consistently improving" if it never appears in drift_events
    # but the total_improvements > 0 suggests some metrics improved
    # Since we don't track improvement events by metric, we'll use a heuristic:
    # if total_improvements > 0 and total_regressions == 0, status is improving
    # Otherwise, we look at individual metric behavior

    # Determine compass status
    if len(chronic_regressions) > 0:
        # Any chronic regressions = CRITICAL
        compass_status = "CRITICAL"
    elif ledger.total_regressions > 0:
        # Any regressions (but not chronic) = CAUTION
        compass_status = "CAUTION"
    else:
        # No regressions = STABLE
        compass_status = "STABLE"

    # Check for critical severity events
    critical_events = [e for e in ledger.drift_events if e.severity == "critical"]
    if critical_events:
        compass_status = "CRITICAL"

    return {
        "schema_version": "1.0.0",
        "metrics_with_chronic_regressions": chronic_regressions,
        "metrics_consistently_improving": improving_metrics,
        "compass_status": compass_status,
        "total_regressions": ledger.total_regressions,
        "total_improvements": ledger.total_improvements,
        "drift_event_count": len(ledger.drift_events),
    }


def summarize_conformance_for_promotion_dashboard(
    promotion_readiness: Dict[str, Any],
    drift_compass: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Summarize conformance status for the promotion dashboard.

    Combines promotion readiness and drift compass data into a unified view
    suitable for display in a CI/CD dashboard.

    Args:
        promotion_readiness: Result from compute_metric_promotion_readiness().to_dict()
        drift_compass: Result from build_metric_drift_compass()

    Returns:
        Dict with:
        - promotion_ok: bool
        - blocking_metrics: List of metric names blocking promotion
        - status: "OK" | "WARN" | "BLOCK"
        - headline: Neutral explanation string
    """
    # Extract promotion readiness data
    ready = promotion_readiness.get("ready_for_promotion", False)
    blocking_regressions = promotion_readiness.get("blocking_regressions", [])

    # Extract blocking metric names
    blocking_metrics: List[str] = []
    for blocker in blocking_regressions:
        # Blockers contain level info, not metric names directly
        # Include the level as identifier
        level = blocker.get("level", "unknown")
        blocking_metrics.append(f"{level}_conformance")

    # Determine status
    compass_status = drift_compass.get("compass_status", "STABLE")

    if not ready:
        status = "BLOCK"
    elif compass_status == "CRITICAL":
        status = "WARN"
    elif compass_status == "CAUTION":
        status = "WARN"
    else:
        status = "OK"

    # Build neutral headline
    if status == "OK":
        headline = "All conformance checks passed. No blocking regressions detected."
    elif status == "WARN":
        chronic = drift_compass.get("metrics_with_chronic_regressions", [])
        if chronic:
            headline = f"Conformance passed but {len(chronic)} metric(s) show repeated regressions."
        else:
            total_reg = drift_compass.get("total_regressions", 0)
            headline = f"Conformance passed. {total_reg} regression(s) detected in history."
    else:  # BLOCK
        blocker_count = len(blocking_regressions)
        headline = f"Promotion blocked by {blocker_count} conformance regression(s)."

    return {
        "promotion_ok": ready,
        "blocking_metrics": blocking_metrics,
        "status": status,
        "headline": headline,
    }


def build_conformance_director_panel(
    global_health_status: Dict[str, Any],
    drift_compass: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a Director-facing conformance panel.

    Provides a high-level summary suitable for executive dashboards,
    combining health status and drift compass into a single view.

    Args:
        global_health_status: Result from summarize_metric_conformance_for_global_health().to_dict()
        drift_compass: Result from build_metric_drift_compass()

    Returns:
        Dict with:
        - status_light: "GREEN" | "YELLOW" | "RED"
        - overall_conformance_status: "healthy" | "degraded" | "critical" | "failing"
        - weakest_metric: Name of weakest metric or None
        - headline: Neutral one-liner summarizing conformance posture
    """
    # Extract health status
    overall_status = global_health_status.get("overall_conformance_status", "healthy")
    weakest_metric = global_health_status.get("weakest_metric")
    weakest_rate = global_health_status.get("weakest_metric_pass_rate", 1.0)
    any_blockers = global_health_status.get("any_blockers", False)

    # Extract compass status
    compass_status = drift_compass.get("compass_status", "STABLE")
    chronic_regressions = drift_compass.get("metrics_with_chronic_regressions", [])

    # Determine status light
    if overall_status == "failing" or compass_status == "CRITICAL":
        status_light = "RED"
    elif overall_status in ("critical", "degraded") or compass_status == "CAUTION":
        status_light = "YELLOW"
    else:
        status_light = "GREEN"

    # Build neutral headline
    if status_light == "GREEN":
        headline = "Metric conformance is stable. All metrics within acceptable thresholds."
    elif status_light == "YELLOW":
        if chronic_regressions:
            headline = f"Elevated attention: {len(chronic_regressions)} metric(s) with repeated regressions."
        elif weakest_metric:
            headline = f"Elevated attention: {weakest_metric} at {weakest_rate:.0%} pass rate."
        else:
            headline = "Elevated attention: Some metrics below optimal thresholds."
    else:  # RED
        if any_blockers:
            blocker_count = global_health_status.get("blocker_count", 0)
            headline = f"Action required: {blocker_count} metric(s) blocking promotion."
        elif chronic_regressions:
            headline = f"Action required: {len(chronic_regressions)} metric(s) chronically regressing."
        else:
            headline = "Action required: Critical conformance issues detected."

    return {
        "status_light": status_light,
        "overall_conformance_status": overall_status,
        "weakest_metric": weakest_metric,
        "headline": headline,
        "compass_status": compass_status,
    }


# =============================================================================
# PHASE V: Metrics × Budget × Policy Triangulation
# =============================================================================

METRIC_BUDGET_JOINT_SCHEMA_VERSION = "1.0.0"
GLOBAL_CONSOLE_SCHEMA_VERSION = "1.0.0"
POLICY_TELEMETRY_HINT_SCHEMA_VERSION = "1.0.0"


def build_metric_budget_joint_view(
    drift_compass: Dict[str, Any],
    budget_uplift_view: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a joint metric-budget view for unified uplift decisions.

    Combines metric drift compass with budget uplift view to produce
    a single unified gate signal for promotion decisions.

    Args:
        drift_compass: Result from build_metric_drift_compass()
        budget_uplift_view: Result from summarize_budget_and_metrics_for_uplift()

    Returns:
        Dict with:
        - schema_version: Version string
        - joint_status: "OK" | "WARN" | "BLOCK"
        - blocking_metrics: List of metrics blocking promotion
        - blocking_slices: List of slices blocking promotion
        - notes: List of neutral rationale strings
    """
    notes: List[str] = []
    blocking_metrics: List[str] = []
    blocking_slices: List[str] = []

    # Extract compass status
    compass_status = drift_compass.get("compass_status", "STABLE")
    chronic_regressions = drift_compass.get("metrics_with_chronic_regressions", [])

    # Extract budget uplift view
    budget_status = budget_uplift_view.get("status", "OK")
    budget_blocking_slices = budget_uplift_view.get("blocking_slices", [])
    budget_notes = budget_uplift_view.get("notes", [])
    uplift_ready = budget_uplift_view.get("uplift_ready", True)

    # Collect blocking slices from budget
    blocking_slices = list(budget_blocking_slices)

    # Collect blocking metrics from compass
    if chronic_regressions:
        blocking_metrics.extend(chronic_regressions)

    # Determine joint status
    # BLOCK conditions:
    # 1. Budget status is BLOCK
    # 2. Compass status is CRITICAL (chronic or critical events)
    # 3. Budget uplift_ready is False

    if budget_status == "BLOCK":
        joint_status = "BLOCK"
        notes.append("Budget risk blocks promotion")
        notes.extend(budget_notes)
    elif compass_status == "CRITICAL":
        joint_status = "BLOCK"
        if chronic_regressions:
            notes.append(f"Chronic metric regressions detected: {', '.join(chronic_regressions)}")
        else:
            notes.append("Critical metric drift events detected")
    elif not uplift_ready:
        joint_status = "BLOCK"
        notes.append("Budget uplift readiness check failed")
        notes.extend(budget_notes)
    # WARN conditions:
    # 1. Budget status is WARN
    # 2. Compass status is CAUTION
    elif budget_status == "WARN" or compass_status == "CAUTION":
        joint_status = "WARN"
        if budget_status == "WARN":
            notes.append("Budget risk is elevated")
            notes.extend(budget_notes)
        if compass_status == "CAUTION":
            total_regressions = drift_compass.get("total_regressions", 0)
            notes.append(f"Metric drift detected: {total_regressions} regression(s) in history")
    else:
        joint_status = "OK"
        notes.append("Metrics and budget are within acceptable bounds")

    return {
        "schema_version": METRIC_BUDGET_JOINT_SCHEMA_VERSION,
        "joint_status": joint_status,
        "blocking_metrics": blocking_metrics,
        "blocking_slices": blocking_slices,
        "notes": notes,
    }


def summarize_conformance_for_global_console(
    global_health_status: Dict[str, Any],
    drift_compass: Dict[str, Any],
    joint_view: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Summarize conformance status for the global health console.

    Provides a unified view suitable for consumption by the single
    global health console, combining health status, drift compass,
    and joint metric-budget view.

    Args:
        global_health_status: Result from summarize_metric_conformance_for_global_health().to_dict()
        drift_compass: Result from build_metric_drift_compass()
        joint_view: Result from build_metric_budget_joint_view()

    Returns:
        Dict with:
        - schema_version: Version string
        - metrics_ok: bool - True if metrics are within acceptable bounds
        - status_light: "GREEN" | "YELLOW" | "RED"
        - weakest_metric: Name of weakest metric or None
        - headline: Neutral one-liner summarizing overall posture
    """
    # Extract health status
    overall_conformance = global_health_status.get("overall_conformance_status", "healthy")
    weakest_metric = global_health_status.get("weakest_metric")
    weakest_rate = global_health_status.get("weakest_metric_pass_rate", 1.0)
    any_blockers = global_health_status.get("any_blockers", False)

    # Extract compass status
    compass_status = drift_compass.get("compass_status", "STABLE")
    chronic_regressions = drift_compass.get("metrics_with_chronic_regressions", [])

    # Extract joint view status
    joint_status = joint_view.get("joint_status", "OK")
    blocking_metrics = joint_view.get("blocking_metrics", [])
    blocking_slices = joint_view.get("blocking_slices", [])

    # Determine metrics_ok
    # Metrics are OK if:
    # - Joint status is not BLOCK
    # - Conformance status is healthy or degraded (not critical/failing)
    # - Compass status is not CRITICAL
    metrics_ok = (
        joint_status != "BLOCK"
        and overall_conformance in ("healthy", "degraded")
        and compass_status != "CRITICAL"
    )

    # Determine status light
    if joint_status == "BLOCK" or overall_conformance == "failing" or compass_status == "CRITICAL":
        status_light = "RED"
    elif joint_status == "WARN" or overall_conformance in ("critical", "degraded") or compass_status == "CAUTION":
        status_light = "YELLOW"
    else:
        status_light = "GREEN"

    # Build neutral headline
    if status_light == "GREEN":
        headline = "System healthy. Metrics and budget within acceptable bounds."
    elif status_light == "YELLOW":
        if compass_status == "CAUTION":
            total_reg = drift_compass.get("total_regressions", 0)
            headline = f"Elevated attention: {total_reg} metric regression(s) detected."
        elif overall_conformance == "degraded":
            headline = f"Elevated attention: {weakest_metric} at {weakest_rate:.0%} pass rate."
        elif blocking_slices:
            headline = f"Elevated attention: {len(blocking_slices)} slice(s) require review."
        else:
            headline = "Elevated attention: Some subsystems below optimal thresholds."
    else:  # RED
        blockers = []
        if blocking_metrics:
            blockers.append(f"{len(blocking_metrics)} metric(s)")
        if blocking_slices:
            blockers.append(f"{len(blocking_slices)} slice(s)")
        if any_blockers:
            blockers.append("conformance failures")

        if blockers:
            headline = f"Action required: {', '.join(blockers)} blocking promotion."
        else:
            headline = "Action required: Critical issues detected across subsystems."

    return {
        "schema_version": GLOBAL_CONSOLE_SCHEMA_VERSION,
        "metrics_ok": metrics_ok,
        "status_light": status_light,
        "weakest_metric": weakest_metric,
        "headline": headline,
    }


def attach_policy_telemetry_hint(
    conformance_panel: Dict[str, Any],
    policy_drift_radar: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Attach policy telemetry hints to a conformance panel.

    Enriches a conformance panel with policy drift information for
    dashboard display. This is a compatibility stub that adds
    policy-related fields without gating logic.

    Args:
        conformance_panel: Any conformance panel dict (e.g., director panel)
        policy_drift_radar: Optional policy drift radar dict with fields:
            - volatility_status: "STABLE" | "VOLATILE" | "CRITICAL"
            - max_l2_drift: float (maximum L2 distance observed)
            If None, defaults are used.

    Returns:
        Dict containing all fields from conformance_panel plus:
        - policy_volatility_status: "STABLE" | "VOLATILE" | "CRITICAL" | "UNKNOWN"
        - max_policy_l2_drift: float (0.0 if unknown)
        - policy_hint_schema_version: Version string
    """
    # Start with a copy of the conformance panel
    enriched = dict(conformance_panel)

    # Extract policy drift info if available
    if policy_drift_radar is not None:
        volatility_status = policy_drift_radar.get("volatility_status", "UNKNOWN")
        max_l2_drift = policy_drift_radar.get("max_l2_drift", 0.0)
    else:
        volatility_status = "UNKNOWN"
        max_l2_drift = 0.0

    # Add policy telemetry hints
    enriched["policy_volatility_status"] = volatility_status
    enriched["max_policy_l2_drift"] = max_l2_drift
    enriched["policy_hint_schema_version"] = POLICY_TELEMETRY_HINT_SCHEMA_VERSION

    return enriched


# =============================================================================
# PHASE VI: Metrics + Budget as One Governance Surface
# =============================================================================

GOVERNANCE_CONSOLE_TILE_SCHEMA_VERSION = "1.0.0"
GOVERNANCE_SIGNAL_SCHEMA_VERSION = "1.0.0"


def summarize_conformance_and_budget_for_global_console(
    global_health_status: Dict[str, Any],
    joint_view: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Produce a global console tile combining conformance and budget status.

    This is a simplified adapter that produces a single tile suitable for
    display in the global health console, treating metrics + budget as
    one unified governance surface.

    Args:
        global_health_status: Result from summarize_metric_conformance_for_global_health().to_dict()
        joint_view: Result from build_metric_budget_joint_view()

    Returns:
        Dict with:
        - schema_version: Version string
        - metrics_ok: bool - True if metrics layer is healthy
        - status_light: "GREEN" | "YELLOW" | "RED"
        - headline: Neutral one-liner summarizing metrics layer posture
        - weakest_metric: Name of weakest metric or None
    """
    # Extract health status
    overall_conformance = global_health_status.get("overall_conformance_status", "healthy")
    weakest_metric = global_health_status.get("weakest_metric")
    weakest_rate = global_health_status.get("weakest_metric_pass_rate", 1.0)
    any_blockers = global_health_status.get("any_blockers", False)

    # Extract joint view status
    joint_status = joint_view.get("joint_status", "OK")
    blocking_metrics = joint_view.get("blocking_metrics", [])
    blocking_slices = joint_view.get("blocking_slices", [])
    joint_notes = joint_view.get("notes", [])

    # Determine metrics_ok
    # Metrics layer is OK if:
    # - Joint status is not BLOCK
    # - Conformance status is healthy or degraded (not critical/failing)
    metrics_ok = (
        joint_status != "BLOCK"
        and overall_conformance in ("healthy", "degraded")
    )

    # Determine status light
    if joint_status == "BLOCK" or overall_conformance == "failing":
        status_light = "RED"
    elif joint_status == "WARN" or overall_conformance in ("critical", "degraded"):
        status_light = "YELLOW"
    else:
        status_light = "GREEN"

    # Build neutral headline
    if status_light == "GREEN":
        headline = "Metrics layer healthy. Conformance and budget within bounds."
    elif status_light == "YELLOW":
        if overall_conformance == "degraded" and weakest_metric:
            headline = f"Metrics layer elevated: {weakest_metric} at {weakest_rate:.0%}."
        elif blocking_slices:
            headline = f"Metrics layer elevated: {len(blocking_slices)} slice(s) under review."
        elif joint_notes:
            headline = f"Metrics layer elevated: {joint_notes[0]}"
        else:
            headline = "Metrics layer elevated: Some thresholds below optimal."
    else:  # RED
        blockers = []
        if blocking_metrics:
            blockers.append(f"{len(blocking_metrics)} metric(s)")
        if blocking_slices:
            blockers.append(f"{len(blocking_slices)} slice(s)")
        if any_blockers:
            blockers.append("conformance failures")

        if blockers:
            headline = f"Metrics layer blocked: {', '.join(blockers)} require attention."
        else:
            headline = "Metrics layer blocked: Critical issues detected."

    return {
        "schema_version": GOVERNANCE_CONSOLE_TILE_SCHEMA_VERSION,
        "metrics_ok": metrics_ok,
        "status_light": status_light,
        "headline": headline,
        "weakest_metric": weakest_metric,
    }


def to_governance_signal_for_metrics(
    global_console_tile: Dict[str, Any],
    joint_view: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generate a governance signal for CLAUDE I integration.

    Maps the combined metric + budget status into a structured signal
    that CLAUDE I (Governance) can consume for promotion decisions.

    Args:
        global_console_tile: Result from summarize_conformance_and_budget_for_global_console()
        joint_view: Result from build_metric_budget_joint_view()

    Returns:
        Dict with:
        - schema_version: Version string
        - signal_type: "METRICS_LAYER_GATE"
        - status: "PROCEED" | "PROCEED_WITH_CAUTION" | "BLOCK"
        - blocking_rules: List of rule codes that triggered blocking
        - blocking_rate: float (0.0 = no blocking, 1.0 = full block)
        - metrics_ok: bool
        - notes: List of neutral explanation strings
    """
    # Extract tile status
    metrics_ok = global_console_tile.get("metrics_ok", False)
    status_light = global_console_tile.get("status_light", "RED")

    # Extract joint view details
    joint_status = joint_view.get("joint_status", "BLOCK")
    blocking_metrics = joint_view.get("blocking_metrics", [])
    blocking_slices = joint_view.get("blocking_slices", [])
    joint_notes = joint_view.get("notes", [])

    # Build blocking rules
    blocking_rules: List[str] = []

    # Check for metric drift critical
    if blocking_metrics:
        blocking_rules.append("METRIC_DRIFT_CRITICAL")

    # Check for budget high risk
    if blocking_slices:
        blocking_rules.append("BUDGET_HIGH_RISK")

    # Check for uplift readiness failure
    if joint_status == "BLOCK" and not blocking_metrics and not blocking_slices:
        blocking_rules.append("UPLIFT_READINESS_FAILED")

    # Determine status
    if status_light == "GREEN":
        status = "PROCEED"
        blocking_rate = 0.0
    elif status_light == "YELLOW":
        status = "PROCEED_WITH_CAUTION"
        # Partial blocking rate based on number of warnings
        warning_count = len(blocking_slices) + (1 if blocking_metrics else 0)
        blocking_rate = min(0.5, warning_count * 0.1)
    else:  # RED
        status = "BLOCK"
        # Full or partial blocking based on severity
        total_blockers = len(blocking_rules)
        blocking_rate = min(1.0, 0.5 + (total_blockers * 0.25))

    # Build notes
    notes: List[str] = []
    if metrics_ok:
        notes.append("Metrics layer within acceptable bounds")
    else:
        notes.append("Metrics layer requires attention")

    if blocking_rules:
        notes.append(f"Active blocking rules: {', '.join(blocking_rules)}")

    notes.extend(joint_notes)

    return {
        "schema_version": GOVERNANCE_SIGNAL_SCHEMA_VERSION,
        "signal_type": "METRICS_LAYER_GATE",
        "status": status,
        "blocking_rules": blocking_rules,
        "blocking_rate": blocking_rate,
        "metrics_ok": metrics_ok,
        "notes": notes,
    }
