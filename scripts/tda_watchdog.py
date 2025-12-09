#!/usr/bin/env python3
"""
TDA Autonomous Watchdog Daemon — Phase VI Auto-Watchdog

Operation CORTEX: Phase VI Auto-Watchdog & Global Health Coupler
=================================================================

Automated monitor that can be run via cron/CI to detect TDA health issues.
Turns the operator console into an unattended watchdog.

Usage:
    python scripts/tda_watchdog.py \
        --governance-log artifacts/tda/governance_runs/*.json \
        --config config/tda_watchdog.yaml \
        --output artifacts/tda/watchdog_report.json

Exit Codes:
    0 - tda_status == "OK"
    1 - tda_status == "ATTENTION"
    2 - tda_status == "ALERT" or parsing/IO error

This enables CI/cron to treat CORTEX as a true watchdog.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

# Ensure parent packages are importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.health.tda_adapter import (
    summarize_tda_for_global_health,
    TDA_STATUS_OK,
    TDA_STATUS_ATTENTION,
    TDA_STATUS_ALERT,
)
from backend.tda.governance_console import (
    build_governance_console_snapshot,
    GOVERNANCE_CONSOLE_SCHEMA_VERSION,
)

# Schema version for watchdog report
WATCHDOG_REPORT_SCHEMA_VERSION = "1.0.0"

# Exit codes
EXIT_OK = 0
EXIT_ATTENTION = 1
EXIT_ALERT = 2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Alert Data Structures
# ============================================================================

@dataclass
class Alert:
    """Single alert from watchdog evaluation."""
    code: str
    severity: str  # "ATTENTION" | "ALERT"
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WatchdogReport:
    """Complete watchdog report."""
    schema_version: str
    generated_at: str
    tda_status: str
    block_rate: float
    mean_hss: Optional[float]
    hss_trend: str
    governance_signal: str
    recent_runs: int
    signal_strength: str  # "strong" | "weak"
    alerts: List[Alert] = field(default_factory=list)
    metrics: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "tda_status": self.tda_status,
            "block_rate": round(self.block_rate, 4),
            "mean_hss": round(self.mean_hss, 4) if self.mean_hss is not None else None,
            "hss_trend": self.hss_trend,
            "governance_signal": self.governance_signal,
            "recent_runs": self.recent_runs,
            "signal_strength": self.signal_strength,
            "alerts": [a.to_dict() for a in self.alerts],
        }
        if self.metrics is not None:
            d["metrics"] = self.metrics
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ============================================================================
# Configuration Loading
# ============================================================================

@dataclass
class WatchdogConfig:
    """Parsed watchdog configuration."""
    # Block rate thresholds
    block_rate_max_ok: float = 0.05
    block_rate_max_attention: float = 0.15
    block_rate_code_elevated: str = "TDA_BLOCK_RATE_ELEVATED"
    block_rate_code_high: str = "TDA_BLOCK_RATE_HIGH"

    # Mean HSS thresholds
    mean_hss_min_ok: float = 0.6
    mean_hss_min_attention: float = 0.4
    mean_hss_code_low: str = "TDA_MEAN_HSS_LOW"
    mean_hss_code_critical: str = "TDA_MEAN_HSS_CRITICAL"

    # HSS trend
    alert_on_degrading_trend: bool = True
    hss_trend_code: str = "TDA_HSS_TREND_DEGRADING"

    # Golden alignment
    alert_on_drifting: bool = True
    alert_on_broken: bool = True
    golden_code_drifting: str = "TDA_GOLDEN_ALIGNMENT_DRIFTING"
    golden_code_broken: str = "TDA_GOLDEN_ALIGNMENT_BROKEN"

    # Exception windows
    exception_max_active_ok: int = 0
    exception_code: str = "TDA_EXCEPTION_WINDOW_ACTIVE"

    # Signal strength
    min_runs_for_strong_signal: int = 10
    weak_signal_code: str = "TDA_WEAK_SIGNAL"

    # Combined rules
    combined_block_degrading_enabled: bool = True
    combined_block_degrading_threshold: float = 0.2
    combined_block_degrading_code: str = "TDA_COMBINED_BLOCK_AND_DEGRADING"
    combined_exception_drift_enabled: bool = True
    combined_exception_drift_code: str = "TDA_EXCEPTION_WITH_ALIGNMENT_DRIFT"

    # Output
    include_metrics: bool = True


def load_config(path: Optional[Path]) -> WatchdogConfig:
    """
    Load watchdog configuration from YAML file.

    Args:
        path: Path to config file, or None for defaults.

    Returns:
        WatchdogConfig with parsed values.
    """
    if path is None or not path.exists():
        logger.info("Using default watchdog configuration")
        return WatchdogConfig()

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    config = WatchdogConfig()

    # Parse block_rate section
    if "block_rate" in raw:
        br = raw["block_rate"]
        config.block_rate_max_ok = float(br.get("max_ok", config.block_rate_max_ok))
        config.block_rate_max_attention = float(br.get("max_attention", config.block_rate_max_attention))
        if "alert_codes" in br:
            config.block_rate_code_elevated = br["alert_codes"].get("elevated", config.block_rate_code_elevated)
            config.block_rate_code_high = br["alert_codes"].get("high", config.block_rate_code_high)

    # Parse mean_hss section
    if "mean_hss" in raw:
        hss = raw["mean_hss"]
        config.mean_hss_min_ok = float(hss.get("min_ok", config.mean_hss_min_ok))
        config.mean_hss_min_attention = float(hss.get("min_attention", config.mean_hss_min_attention))
        if "alert_codes" in hss:
            config.mean_hss_code_low = hss["alert_codes"].get("low", config.mean_hss_code_low)
            config.mean_hss_code_critical = hss["alert_codes"].get("critical", config.mean_hss_code_critical)

    # Parse hss_trend section
    if "hss_trend" in raw:
        trend = raw["hss_trend"]
        config.alert_on_degrading_trend = trend.get("alert_on_degrading", config.alert_on_degrading_trend)
        config.hss_trend_code = trend.get("alert_code", config.hss_trend_code)

    # Parse golden_alignment section
    if "golden_alignment" in raw:
        golden = raw["golden_alignment"]
        config.alert_on_drifting = golden.get("alert_on_drifting", config.alert_on_drifting)
        config.alert_on_broken = golden.get("alert_on_broken", config.alert_on_broken)
        if "alert_codes" in golden:
            config.golden_code_drifting = golden["alert_codes"].get("drifting", config.golden_code_drifting)
            config.golden_code_broken = golden["alert_codes"].get("broken", config.golden_code_broken)

    # Parse exception_windows section
    if "exception_windows" in raw:
        exc = raw["exception_windows"]
        config.exception_max_active_ok = int(exc.get("max_active_ok", config.exception_max_active_ok))
        config.exception_code = exc.get("alert_code", config.exception_code)

    # Parse signal_strength section
    if "signal_strength" in raw:
        sig = raw["signal_strength"]
        config.min_runs_for_strong_signal = int(sig.get("min_runs_for_strong_signal", config.min_runs_for_strong_signal))
        config.weak_signal_code = sig.get("alert_code", config.weak_signal_code)

    # Parse combined_rules section
    if "combined_rules" in raw:
        combined = raw["combined_rules"]
        if "block_rate_with_degrading_trend" in combined:
            rule = combined["block_rate_with_degrading_trend"]
            config.combined_block_degrading_enabled = rule.get("enabled", config.combined_block_degrading_enabled)
            config.combined_block_degrading_threshold = float(rule.get("block_rate_threshold", config.combined_block_degrading_threshold))
            config.combined_block_degrading_code = rule.get("alert_code", config.combined_block_degrading_code)
        if "exception_with_alignment_drift" in combined:
            rule = combined["exception_with_alignment_drift"]
            config.combined_exception_drift_enabled = rule.get("enabled", config.combined_exception_drift_enabled)
            config.combined_exception_drift_code = rule.get("alert_code", config.combined_exception_drift_code)

    # Parse output section
    if "output" in raw:
        output = raw["output"]
        config.include_metrics = output.get("include_metrics", config.include_metrics)

    logger.info(f"Loaded watchdog configuration from {path}")
    return config


# ============================================================================
# Snapshot Loading
# ============================================================================

def load_snapshot(path: Path) -> Optional[Dict[str, Any]]:
    """
    Load a single governance snapshot from JSON file.

    Args:
        path: Path to snapshot JSON file.

    Returns:
        Parsed snapshot dictionary, or None if loading fails.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle wrapped formats
        if "governance_snapshot" in data:
            return data["governance_snapshot"]
        if "tda_governance" in data:
            return data["tda_governance"]

        return data

    except json.JSONDecodeError as e:
        logger.warning(f"Invalid JSON in {path}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Error loading {path}: {e}")
        return None


def load_snapshots_from_glob(pattern: str) -> List[Dict[str, Any]]:
    """
    Load governance snapshots from glob pattern.

    Args:
        pattern: Glob pattern for snapshot files.

    Returns:
        List of loaded snapshots.
    """
    paths = sorted(glob.glob(pattern))
    logger.info(f"Found {len(paths)} files matching pattern: {pattern}")

    snapshots = []
    for path in paths:
        snapshot = load_snapshot(Path(path))
        if snapshot:
            snapshots.append(snapshot)

    logger.info(f"Successfully loaded {len(snapshots)} snapshots")
    return snapshots


def aggregate_snapshots(snapshots: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate multiple snapshots into a single summary.

    Uses the most recent snapshot's values with aggregated cycle counts.

    Args:
        snapshots: Sequence of governance snapshots.

    Returns:
        Aggregated snapshot dictionary.
    """
    if not snapshots:
        return {
            "schema_version": GOVERNANCE_CONSOLE_SCHEMA_VERSION,
            "cycle_count": 0,
            "block_rate": 0.0,
            "warn_rate": 0.0,
            "mean_hss": 0.0,
            "hss_trend": "unknown",
            "golden_alignment": "UNKNOWN",
            "exception_windows_active": 0,
            "recent_exceptions": [],
            "governance_signal": "HEALTHY",
        }

    if len(snapshots) == 1:
        return snapshots[0]

    # Aggregate metrics
    total_cycles = sum(s.get("cycle_count", 0) for s in snapshots)
    total_blocks = sum(
        int(s.get("block_rate", 0) * s.get("cycle_count", 0))
        for s in snapshots
    )
    total_warns = sum(
        int(s.get("warn_rate", 0) * s.get("cycle_count", 0))
        for s in snapshots
    )

    hss_values = [s.get("mean_hss", 0) for s in snapshots if s.get("mean_hss") is not None]
    mean_hss = sum(hss_values) / len(hss_values) if hss_values else 0.0

    # Use most recent snapshot for trend/alignment
    latest = snapshots[-1]

    # Count active exception windows across all snapshots
    exception_windows_active = sum(
        s.get("exception_windows_active", 0) for s in snapshots
    )

    return {
        "schema_version": GOVERNANCE_CONSOLE_SCHEMA_VERSION,
        "cycle_count": total_cycles,
        "block_rate": total_blocks / total_cycles if total_cycles > 0 else 0.0,
        "warn_rate": total_warns / total_cycles if total_cycles > 0 else 0.0,
        "mean_hss": mean_hss,
        "hss_trend": latest.get("hss_trend", "unknown"),
        "golden_alignment": latest.get("golden_alignment", "UNKNOWN"),
        "exception_windows_active": exception_windows_active,
        "recent_exceptions": latest.get("recent_exceptions", []),
        "governance_signal": latest.get("governance_signal", "HEALTHY"),
    }


# ============================================================================
# Alert Evaluation
# ============================================================================

def evaluate_alerts(
    snapshot: Dict[str, Any],
    config: WatchdogConfig,
) -> List[Alert]:
    """
    Evaluate snapshot against config thresholds to generate alerts.

    Args:
        snapshot: Aggregated governance snapshot.
        config: Watchdog configuration.

    Returns:
        List of Alert objects.
    """
    alerts = []
    block_rate = snapshot.get("block_rate", 0.0)
    mean_hss = snapshot.get("mean_hss")
    hss_trend = snapshot.get("hss_trend", "").upper()
    golden_alignment = snapshot.get("golden_alignment", "").upper()
    cycle_count = snapshot.get("cycle_count", 0)
    exception_windows_active = snapshot.get("exception_windows_active", 0)

    # Block rate alerts
    if block_rate > config.block_rate_max_attention:
        alerts.append(Alert(
            code=config.block_rate_code_high,
            severity="ALERT",
            message=f"block_rate={block_rate:.2%} exceeds max_attention={config.block_rate_max_attention:.2%}",
        ))
    elif block_rate > config.block_rate_max_ok:
        alerts.append(Alert(
            code=config.block_rate_code_elevated,
            severity="ATTENTION",
            message=f"block_rate={block_rate:.2%} exceeds max_ok={config.block_rate_max_ok:.2%}",
        ))

    # Mean HSS alerts
    if mean_hss is not None:
        if mean_hss < config.mean_hss_min_attention:
            alerts.append(Alert(
                code=config.mean_hss_code_critical,
                severity="ALERT",
                message=f"mean_hss={mean_hss:.4f} below min_attention={config.mean_hss_min_attention}",
            ))
        elif mean_hss < config.mean_hss_min_ok:
            alerts.append(Alert(
                code=config.mean_hss_code_low,
                severity="ATTENTION",
                message=f"mean_hss={mean_hss:.4f} below min_ok={config.mean_hss_min_ok}",
            ))

    # HSS trend alerts
    if config.alert_on_degrading_trend and hss_trend == "DEGRADING":
        alerts.append(Alert(
            code=config.hss_trend_code,
            severity="ATTENTION",
            message=f"hss_trend={hss_trend} over {cycle_count} cycles",
        ))

    # Golden alignment alerts
    if config.alert_on_broken and golden_alignment == "BROKEN":
        alerts.append(Alert(
            code=config.golden_code_broken,
            severity="ALERT",
            message="golden_alignment is BROKEN",
        ))
    elif config.alert_on_drifting and golden_alignment == "DRIFTING":
        alerts.append(Alert(
            code=config.golden_code_drifting,
            severity="ATTENTION",
            message="golden_alignment is DRIFTING",
        ))

    # Exception window alerts
    if exception_windows_active > config.exception_max_active_ok:
        alerts.append(Alert(
            code=config.exception_code,
            severity="ATTENTION",
            message=f"{exception_windows_active} exception window(s) active",
        ))

    # Combined rule: block_rate + degrading trend
    if config.combined_block_degrading_enabled:
        if block_rate >= config.combined_block_degrading_threshold and hss_trend == "DEGRADING":
            alerts.append(Alert(
                code=config.combined_block_degrading_code,
                severity="ALERT",
                message=f"block_rate={block_rate:.2%} with degrading HSS trend",
            ))

    # Combined rule: exception window + alignment drift
    if config.combined_exception_drift_enabled:
        if exception_windows_active > 0 and golden_alignment in ("DRIFTING", "BROKEN"):
            alerts.append(Alert(
                code=config.combined_exception_drift_code,
                severity="ALERT",
                message=f"exception window active with golden_alignment={golden_alignment}",
            ))

    # Weak signal alert
    if cycle_count < config.min_runs_for_strong_signal:
        alerts.append(Alert(
            code=config.weak_signal_code,
            severity="ATTENTION",
            message=f"only {cycle_count} cycles analyzed (min={config.min_runs_for_strong_signal} for strong signal)",
        ))

    return alerts


def determine_status(alerts: List[Alert]) -> str:
    """
    Determine overall TDA status from alerts.

    Args:
        alerts: List of generated alerts.

    Returns:
        Status string: "OK" | "ATTENTION" | "ALERT"
    """
    if any(a.severity == "ALERT" for a in alerts):
        return TDA_STATUS_ALERT
    if any(a.severity == "ATTENTION" for a in alerts):
        return TDA_STATUS_ATTENTION
    return TDA_STATUS_OK


# ============================================================================
# Report Generation
# ============================================================================

def generate_watchdog_report(
    snapshots: Sequence[Dict[str, Any]],
    config: WatchdogConfig,
) -> WatchdogReport:
    """
    Generate watchdog report from governance snapshots.

    Args:
        snapshots: Sequence of governance snapshots.
        config: Watchdog configuration.

    Returns:
        WatchdogReport with status and alerts.
    """
    # Aggregate snapshots
    aggregated = aggregate_snapshots(snapshots)

    # Get health tile
    health_tile = summarize_tda_for_global_health(aggregated)

    # Evaluate alerts
    alerts = evaluate_alerts(aggregated, config)

    # Determine status
    tda_status = determine_status(alerts)

    # Determine signal strength
    cycle_count = aggregated.get("cycle_count", 0)
    signal_strength = "strong" if cycle_count >= config.min_runs_for_strong_signal else "weak"

    # Build metrics if requested
    metrics = None
    if config.include_metrics:
        metrics = {
            "cycle_count": cycle_count,
            "warn_rate": round(aggregated.get("warn_rate", 0.0), 4),
            "golden_alignment": aggregated.get("golden_alignment", "UNKNOWN"),
            "exception_windows_active": aggregated.get("exception_windows_active", 0),
            "runs_aggregated": len(snapshots),
        }

    return WatchdogReport(
        schema_version=WATCHDOG_REPORT_SCHEMA_VERSION,
        generated_at=datetime.utcnow().isoformat() + "Z",
        tda_status=tda_status,
        block_rate=aggregated.get("block_rate", 0.0),
        mean_hss=aggregated.get("mean_hss"),
        hss_trend=health_tile.get("hss_trend", "UNKNOWN"),
        governance_signal=health_tile.get("governance_signal", "OK"),
        recent_runs=len(snapshots),
        signal_strength=signal_strength,
        alerts=alerts,
        metrics=metrics,
    )


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    """
    Main entry point for TDA watchdog daemon.

    Returns:
        Exit code based on TDA status.
    """
    parser = argparse.ArgumentParser(
        description="TDA Autonomous Watchdog Daemon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exit Codes:
    0 - tda_status == "OK"
    1 - tda_status == "ATTENTION"
    2 - tda_status == "ALERT" or error

Examples:
    python scripts/tda_watchdog.py \\
        --governance-log "artifacts/tda/*.json" \\
        --config config/tda_watchdog.yaml \\
        --output watchdog_report.json
        """,
    )

    parser.add_argument(
        "--governance-log",
        type=str,
        required=True,
        help="Glob pattern for governance snapshot files",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to watchdog configuration YAML file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output file path for JSON report",
    )
    parser.add_argument(
        "--json-stdout",
        action="store_true",
        help="Print JSON report to stdout",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress summary output",
    )

    args = parser.parse_args()

    try:
        # Load configuration
        config = load_config(args.config)

        # Load snapshots
        snapshots = load_snapshots_from_glob(args.governance_log)

        if not snapshots:
            logger.error("No governance snapshots found")
            return EXIT_ALERT

        # Generate report
        report = generate_watchdog_report(snapshots, config)

        # Write output
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(report.to_json())
            logger.info(f"Report written to: {args.output}")

        if args.json_stdout:
            print(report.to_json())

        # Print summary
        if not args.quiet:
            print(f"\n{'='*60}")
            print("TDA WATCHDOG REPORT")
            print(f"{'='*60}")
            print(f"Status: {report.tda_status}")
            print(f"Signal Strength: {report.signal_strength}")
            print(f"Recent Runs: {report.recent_runs}")
            print(f"Block Rate: {report.block_rate:.2%}")
            print(f"Mean HSS: {report.mean_hss:.4f}" if report.mean_hss else "Mean HSS: N/A")
            print(f"HSS Trend: {report.hss_trend}")
            print(f"Governance Signal: {report.governance_signal}")

            if report.alerts:
                print(f"\nAlerts ({len(report.alerts)}):")
                for alert in report.alerts:
                    print(f"  [{alert.severity}] {alert.code}: {alert.message}")
            else:
                print("\nNo alerts generated.")

            print(f"{'='*60}\n")

        # Return exit code based on status
        if report.tda_status == TDA_STATUS_OK:
            return EXIT_OK
        elif report.tda_status == TDA_STATUS_ATTENTION:
            return EXIT_ATTENTION
        else:
            return EXIT_ALERT

    except Exception as e:
        logger.error(f"Watchdog error: {e}")
        return EXIT_ALERT


if __name__ == "__main__":
    sys.exit(main())
