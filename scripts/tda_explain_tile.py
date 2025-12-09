#!/usr/bin/env python3
"""
TDA Explain Tile — Phase VI Global Health Hook

Operation CORTEX: Phase VI Auto-Watchdog & Global Health Coupler
=================================================================

Tile-level explanation tool that provides human-readable analysis
of TDA health tiles from global_health.json.

This closes the loop between operator console and global health surface:
same truth, two views.

Usage:
    # Explain TDA tile from global_health.json
    python scripts/tda_explain_tile.py --global-health global_health.json

    # Explain standalone TDA tile
    python scripts/tda_explain_tile.py --tile tda_tile.json

    # Output to file
    python scripts/tda_explain_tile.py --global-health global_health.json --output explanation.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure parent packages are importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.health.tda_adapter import (
    TDA_STATUS_OK,
    TDA_STATUS_ATTENTION,
    TDA_STATUS_ALERT,
    HSS_TREND_IMPROVING,
    HSS_TREND_STABLE,
    HSS_TREND_DEGRADING,
    HSS_TREND_UNKNOWN,
)

# Schema version for tile explanation
TILE_EXPLANATION_SCHEMA_VERSION = "1.0.0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Reason Codes for Tile Explanation
# ============================================================================

# Status reason codes
REASON_STATUS_OK = "STATUS_OK"
REASON_STATUS_ATTENTION = "STATUS_ATTENTION"
REASON_STATUS_ALERT = "STATUS_ALERT"

# Block rate reason codes
REASON_BLOCK_RATE_ZERO = "BLOCK_RATE_ZERO"
REASON_BLOCK_RATE_LOW = "BLOCK_RATE_LOW"
REASON_BLOCK_RATE_ELEVATED = "BLOCK_RATE_ELEVATED"
REASON_BLOCK_RATE_HIGH = "BLOCK_RATE_HIGH"

# HSS reason codes
REASON_HSS_HEALTHY = "HSS_HEALTHY"
REASON_HSS_LOW = "HSS_LOW"
REASON_HSS_CRITICAL = "HSS_CRITICAL"

# HSS trend reason codes
REASON_HSS_TREND_IMPROVING = "HSS_TREND_IMPROVING"
REASON_HSS_TREND_STABLE = "HSS_TREND_STABLE"
REASON_HSS_TREND_DEGRADING = "HSS_TREND_DEGRADING"
REASON_HSS_TREND_UNKNOWN = "HSS_TREND_UNKNOWN"

# Governance signal reason codes
REASON_GOVERNANCE_OK = "GOVERNANCE_SIGNAL_OK"
REASON_GOVERNANCE_WARN = "GOVERNANCE_SIGNAL_WARN"
REASON_GOVERNANCE_BLOCK = "GOVERNANCE_SIGNAL_BLOCK"


# ============================================================================
# Tile Explanation Data Structures
# ============================================================================

@dataclass
class TileExplanation:
    """Structured explanation of a TDA health tile."""
    schema_version: str
    tda_status: str
    reason_codes: List[str]
    metrics: Dict[str, Any]
    human_summary: str
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "tda_status": self.tda_status,
            "reason_codes": self.reason_codes,
            "metrics": self.metrics,
            "human_summary": self.human_summary,
            "notes": self.notes,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ============================================================================
# Explanation Logic
# ============================================================================

def classify_block_rate(block_rate: float) -> str:
    """Classify block rate into reason code."""
    if block_rate == 0.0:
        return REASON_BLOCK_RATE_ZERO
    elif block_rate < 0.05:
        return REASON_BLOCK_RATE_LOW
    elif block_rate < 0.15:
        return REASON_BLOCK_RATE_ELEVATED
    else:
        return REASON_BLOCK_RATE_HIGH


def classify_mean_hss(mean_hss: Optional[float]) -> str:
    """Classify mean HSS into reason code."""
    if mean_hss is None:
        return "HSS_UNAVAILABLE"
    elif mean_hss >= 0.6:
        return REASON_HSS_HEALTHY
    elif mean_hss >= 0.4:
        return REASON_HSS_LOW
    else:
        return REASON_HSS_CRITICAL


def classify_hss_trend(hss_trend: str) -> str:
    """Classify HSS trend into reason code."""
    trend_upper = hss_trend.upper() if hss_trend else "UNKNOWN"
    return {
        HSS_TREND_IMPROVING: REASON_HSS_TREND_IMPROVING,
        HSS_TREND_STABLE: REASON_HSS_TREND_STABLE,
        HSS_TREND_DEGRADING: REASON_HSS_TREND_DEGRADING,
    }.get(trend_upper, REASON_HSS_TREND_UNKNOWN)


def classify_governance_signal(signal: str) -> str:
    """Classify governance signal into reason code."""
    signal_upper = signal.upper() if signal else "OK"
    return {
        "OK": REASON_GOVERNANCE_OK,
        "WARN": REASON_GOVERNANCE_WARN,
        "BLOCK": REASON_GOVERNANCE_BLOCK,
    }.get(signal_upper, REASON_GOVERNANCE_OK)


def classify_status(tda_status: str) -> str:
    """Classify TDA status into reason code."""
    status_upper = tda_status.upper() if tda_status else "OK"
    return {
        TDA_STATUS_OK: REASON_STATUS_OK,
        TDA_STATUS_ATTENTION: REASON_STATUS_ATTENTION,
        TDA_STATUS_ALERT: REASON_STATUS_ALERT,
    }.get(status_upper, REASON_STATUS_OK)


def build_human_summary(
    tda_status: str,
    block_rate: float,
    mean_hss: Optional[float],
    hss_trend: str,
    governance_signal: str,
) -> str:
    """
    Build human-readable summary of tile status.

    Uses neutral, structural language.
    """
    parts = []

    # Status summary
    status_upper = tda_status.upper() if tda_status else "OK"
    if status_upper == TDA_STATUS_ALERT:
        parts.append("TDA health tile indicates ALERT condition.")
    elif status_upper == TDA_STATUS_ATTENTION:
        parts.append("TDA health tile indicates ATTENTION condition.")
    else:
        parts.append("TDA health tile indicates OK condition.")

    # Block rate summary
    if block_rate == 0.0:
        parts.append("No cycles blocked.")
    elif block_rate < 0.05:
        parts.append(f"Block rate is low ({block_rate:.2%}).")
    elif block_rate < 0.15:
        parts.append(f"Block rate is elevated ({block_rate:.2%}).")
    else:
        parts.append(f"Block rate is high ({block_rate:.2%}).")

    # Mean HSS summary
    if mean_hss is not None:
        if mean_hss >= 0.6:
            parts.append(f"Mean HSS ({mean_hss:.4f}) is within healthy range.")
        elif mean_hss >= 0.4:
            parts.append(f"Mean HSS ({mean_hss:.4f}) is below recommended threshold.")
        else:
            parts.append(f"Mean HSS ({mean_hss:.4f}) is at critical level.")

    # HSS trend summary
    trend_upper = hss_trend.upper() if hss_trend else "UNKNOWN"
    if trend_upper == HSS_TREND_DEGRADING:
        parts.append("HSS trend is degrading over time.")
    elif trend_upper == HSS_TREND_IMPROVING:
        parts.append("HSS trend is improving over time.")
    elif trend_upper == HSS_TREND_STABLE:
        parts.append("HSS trend is stable.")

    # Governance signal summary
    signal_upper = governance_signal.upper() if governance_signal else "OK"
    if signal_upper == "BLOCK":
        parts.append("Governance signal indicates blocking condition.")
    elif signal_upper == "WARN":
        parts.append("Governance signal indicates warning condition.")

    return " ".join(parts)


def explain_tile(tile: Dict[str, Any]) -> TileExplanation:
    """
    Generate structured explanation of a TDA health tile.

    Args:
        tile: TDA health tile dictionary.
            Expected keys: tda_status, block_rate, mean_hss, hss_trend,
                          governance_signal, notes

    Returns:
        TileExplanation with reason codes and human summary.
    """
    # Extract values with defaults
    tda_status = tile.get("tda_status", "OK")
    block_rate = float(tile.get("block_rate", 0.0))
    mean_hss = tile.get("mean_hss")
    if mean_hss is not None:
        mean_hss = float(mean_hss)
    hss_trend = tile.get("hss_trend", "UNKNOWN")
    governance_signal = tile.get("governance_signal", "OK")
    notes = tile.get("notes", [])

    # Build reason codes
    reason_codes = [
        classify_status(tda_status),
        classify_block_rate(block_rate),
        classify_mean_hss(mean_hss),
        classify_hss_trend(hss_trend),
        classify_governance_signal(governance_signal),
    ]

    # Build metrics
    metrics = {
        "block_rate": round(block_rate, 4),
        "mean_hss": round(mean_hss, 4) if mean_hss is not None else None,
        "hss_trend": hss_trend,
        "governance_signal": governance_signal,
    }

    # Build human summary
    human_summary = build_human_summary(
        tda_status=tda_status,
        block_rate=block_rate,
        mean_hss=mean_hss,
        hss_trend=hss_trend,
        governance_signal=governance_signal,
    )

    return TileExplanation(
        schema_version=TILE_EXPLANATION_SCHEMA_VERSION,
        tda_status=tda_status,
        reason_codes=reason_codes,
        metrics=metrics,
        human_summary=human_summary,
        notes=list(notes),
    )


def load_tile_from_global_health(path: Path) -> Dict[str, Any]:
    """
    Load TDA tile from global_health.json.

    Args:
        path: Path to global_health.json file.

    Returns:
        TDA tile dictionary.

    Raises:
        ValueError: If TDA tile not found.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "tda" not in data:
        raise ValueError("No 'tda' section found in global_health.json")

    return data["tda"]


def load_tile_direct(path: Path) -> Dict[str, Any]:
    """
    Load TDA tile directly from JSON file.

    Args:
        path: Path to tile JSON file.

    Returns:
        TDA tile dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# CLI
# ============================================================================

def print_explanation(explanation: TileExplanation) -> None:
    """Print human-readable explanation to console."""
    print(f"\n{'='*60}")
    print("TDA TILE EXPLANATION")
    print(f"{'='*60}")

    print(f"\nStatus: {explanation.tda_status}")

    print("\nReason Codes:")
    for code in explanation.reason_codes:
        print(f"  - {code}")

    print("\nMetrics:")
    for key, value in explanation.metrics.items():
        if value is not None:
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    print(f"\nSummary:\n  {explanation.human_summary}")

    if explanation.notes:
        print("\nNotes:")
        for note in explanation.notes:
            print(f"  - {note}")

    print(f"\n{'='*60}\n")


def main() -> int:
    """
    Main entry point for TDA tile explanation tool.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    parser = argparse.ArgumentParser(
        description="Explain TDA health tile from global_health.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/tda_explain_tile.py --global-health global_health.json
    python scripts/tda_explain_tile.py --tile tda_tile.json
    python scripts/tda_explain_tile.py --global-health global_health.json --output explanation.json
        """,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--global-health",
        type=Path,
        help="Path to global_health.json file",
    )
    input_group.add_argument(
        "--tile",
        type=Path,
        help="Path to standalone TDA tile JSON file",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output file path for JSON explanation",
    )
    parser.add_argument(
        "--json-stdout",
        action="store_true",
        help="Print JSON explanation to stdout",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress human-readable output",
    )

    args = parser.parse_args()

    try:
        # Load tile
        if args.global_health:
            if not args.global_health.exists():
                logger.error(f"File not found: {args.global_health}")
                return 1
            tile = load_tile_from_global_health(args.global_health)
            logger.info(f"Loaded TDA tile from {args.global_health}")
        else:
            if not args.tile.exists():
                logger.error(f"File not found: {args.tile}")
                return 1
            tile = load_tile_direct(args.tile)
            logger.info(f"Loaded TDA tile from {args.tile}")

        # Generate explanation
        explanation = explain_tile(tile)

        # Output
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(explanation.to_json())
            logger.info(f"Explanation written to: {args.output}")

        if args.json_stdout:
            print(explanation.to_json())

        if not args.quiet:
            print_explanation(explanation)

        return 0

    except ValueError as e:
        logger.error(str(e))
        return 1
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
