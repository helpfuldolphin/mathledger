#!/usr/bin/env python3
"""
TDA Explain Block CLI — Phase V Operator Tool

Operation CORTEX: Phase V Operator Console & Self-Audit Harness
================================================================

One-shot, hand-operable tool to explain why the TDA hard gate
fired for a specific cycle.

Usage:
    python -m scripts.tda_explain_block \
        --run-ledger artifacts/runs/run_123.json \
        --cycle-id 42

    python -m scripts.tda_explain_block \
        --run-ledger artifacts/runs/run_123.json \
        --cycle-id 42 \
        --output explanation.json

Output:
    JSON object with structured explanation of gate decision.

Design:
    - No free-text judgment — only structured reasons and flags
    - Uses same decision logic as runtime (no fork)
    - Robust to missing TDA data (emits status: "UNKNOWN")
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure parent packages are importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.tda.governance import TDAHardGateMode
from backend.tda.governance_console import (
    build_block_explanation_from_ledger_entry,
    build_block_explanation,
    BLOCK_EXPLANATION_SCHEMA_VERSION,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_run_ledger(path: Path) -> Dict[str, Any]:
    """
    Load run ledger from JSON file.

    Args:
        path: Path to run ledger JSON file.

    Returns:
        Parsed run ledger dictionary.

    Raises:
        FileNotFoundError: If file doesn't exist.
        json.JSONDecodeError: If file is not valid JSON.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_cycle_entry(
    ledger: Dict[str, Any],
    cycle_id: int,
) -> Optional[Dict[str, Any]]:
    """
    Extract cycle entry from run ledger.

    Supports multiple ledger formats:
    - entries: List of cycle entries
    - cycles: Dict mapping cycle_id to entry
    - results: List of result entries with cycle_id field

    Args:
        ledger: Run ledger dictionary.
        cycle_id: Cycle index or unique ID.

    Returns:
        Cycle entry dict, or None if not found.
    """
    # Format 1: List of entries indexed by position
    if "entries" in ledger and isinstance(ledger["entries"], list):
        entries = ledger["entries"]
        if 0 <= cycle_id < len(entries):
            return entries[cycle_id]

    # Format 2: Dict mapping cycle_id to entry
    if "cycles" in ledger and isinstance(ledger["cycles"], dict):
        return ledger["cycles"].get(str(cycle_id))

    # Format 3: List of results with cycle_id field
    if "results" in ledger and isinstance(ledger["results"], list):
        for entry in ledger["results"]:
            if entry.get("cycle_id") == cycle_id:
                return entry
            if entry.get("cycle") == cycle_id:
                return entry

    # Format 4: TDA-specific tiles list
    if "tda_tiles" in ledger and isinstance(ledger["tda_tiles"], list):
        tiles = ledger["tda_tiles"]
        if 0 <= cycle_id < len(tiles):
            return tiles[cycle_id]

    # Format 5: Direct list at root
    if isinstance(ledger, list):
        if 0 <= cycle_id < len(ledger):
            return ledger[cycle_id]

    return None


def extract_run_id(ledger: Dict[str, Any], path: Path) -> str:
    """
    Extract run ID from ledger or path.

    Args:
        ledger: Run ledger dictionary.
        path: Path to ledger file.

    Returns:
        Run ID string.
    """
    if "run_id" in ledger:
        return str(ledger["run_id"])
    if "id" in ledger:
        return str(ledger["id"])
    # Fall back to filename
    return path.stem


def extract_tda_mode(ledger: Dict[str, Any]) -> TDAHardGateMode:
    """
    Extract TDA mode from ledger metadata.

    Args:
        ledger: Run ledger dictionary.

    Returns:
        TDAHardGateMode, defaulting to HARD if not specified.
    """
    mode_str = None

    # Check various locations for mode
    if "tda_mode" in ledger:
        mode_str = ledger["tda_mode"]
    elif "config" in ledger and "tda_mode" in ledger["config"]:
        mode_str = ledger["config"]["tda_mode"]
    elif "metadata" in ledger and "tda_mode" in ledger["metadata"]:
        mode_str = ledger["metadata"]["tda_mode"]

    if mode_str:
        try:
            return TDAHardGateMode(mode_str.lower())
        except ValueError:
            pass

    return TDAHardGateMode.HARD


def extract_golden_alignment(ledger: Dict[str, Any]) -> str:
    """
    Extract golden alignment status from ledger.

    Args:
        ledger: Run ledger dictionary.

    Returns:
        Alignment status string.
    """
    if "golden_alignment" in ledger:
        return ledger["golden_alignment"]
    if "calibration" in ledger and "status" in ledger["calibration"]:
        status = ledger["calibration"]["status"]
        if status == "OK":
            return "ALIGNED"
        return status
    return "ALIGNED"


def build_explanation(
    ledger: Dict[str, Any],
    cycle_id: int,
    path: Path,
) -> Dict[str, Any]:
    """
    Build block explanation for a cycle.

    Args:
        ledger: Run ledger dictionary.
        cycle_id: Cycle index.
        path: Path to ledger file.

    Returns:
        Block explanation dictionary.
    """
    run_id = extract_run_id(ledger, path)
    tda_mode = extract_tda_mode(ledger)
    golden_alignment = extract_golden_alignment(ledger)

    # Try to find cycle entry
    entry = extract_cycle_entry(ledger, cycle_id)

    if entry is None:
        # Cycle not found — return UNKNOWN
        logger.warning(f"Cycle {cycle_id} not found in ledger")
        return {
            "schema_version": BLOCK_EXPLANATION_SCHEMA_VERSION,
            "run_id": run_id,
            "cycle_id": cycle_id,
            "tda_mode": tda_mode.value,
            "hss": 0.0,
            "scores": {"sns": 0.0, "pcs": 0.0, "drs": 0.0},
            "gate_decision": {
                "status": "UNKNOWN",
                "reason_codes": ["CYCLE_NOT_FOUND"],
                "exception_window_applied": False,
            },
            "effects": {
                "lean_submission_avoided": False,
                "policy_update_avoided": False,
            },
            "status": "UNKNOWN",
        }

    # Build explanation from entry
    explanation = build_block_explanation_from_ledger_entry(
        run_id=run_id,
        cycle_id=cycle_id,
        ledger_entry=entry,
        tda_mode=tda_mode,
        golden_alignment=golden_alignment,
    )

    return explanation.to_dict()


def main() -> int:
    """
    Main entry point for TDA explain block CLI.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    parser = argparse.ArgumentParser(
        description="Explain why TDA hard gate fired for a specific cycle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m scripts.tda_explain_block --run-ledger run.json --cycle-id 42
    python -m scripts.tda_explain_block --run-ledger run.json --cycle-id 42 --output explain.json
        """,
    )

    parser.add_argument(
        "--run-ledger",
        type=Path,
        required=True,
        help="Path to run ledger JSON file",
    )
    parser.add_argument(
        "--cycle-id",
        type=int,
        required=True,
        help="Cycle index or unique ID to explain",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        default=True,
        help="Pretty-print JSON output (default: true)",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Compact JSON output (no indentation)",
    )

    args = parser.parse_args()

    # Validate input file
    if not args.run_ledger.exists():
        logger.error(f"Run ledger not found: {args.run_ledger}")
        return 1

    try:
        # Load ledger
        logger.info(f"Loading run ledger: {args.run_ledger}")
        ledger = load_run_ledger(args.run_ledger)

        # Build explanation
        logger.info(f"Building explanation for cycle {args.cycle_id}")
        explanation = build_explanation(ledger, args.cycle_id, args.run_ledger)

        # Format output
        indent = None if args.compact else 2
        output = json.dumps(explanation, indent=indent)

        # Write output
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output)
            logger.info(f"Explanation written to: {args.output}")
        else:
            print(output)

        return 0

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in run ledger: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error building explanation: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
