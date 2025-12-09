#!/usr/bin/env python3
"""
PHASE II — NOT USED IN PHASE I

U2 Runtime Introspection CLI
============================

Developer tool for inspecting how a U2 runtime run is configured,
without executing any cycles or writing any output files.

This tool performs a dry-run of configuration loading and displays:
- Selected slice name and parameters
- Seed schedule configuration
- Ordering strategy class name
- Error classifier and trace logger classes used

USAGE
-----
    # Inspect a specific configuration
    uv run python experiments/u2_runtime_inspect.py \\
        --slice slice_uplift_goal \\
        --mode baseline \\
        --cycles 10 \\
        --seed 42

    # Show runtime API contract
    uv run python experiments/u2_runtime_inspect.py --show-contract

    # Show all error kinds with descriptions
    uv run python experiments/u2_runtime_inspect.py --show-error-kinds

    # Show all feature flags with defaults and stability
    uv run python experiments/u2_runtime_inspect.py --show-feature-flags

    # Dry-run config validation
    uv run python experiments/u2_runtime_inspect.py --dry-run-config config/my_config.yaml

    # Health snapshot (machine-readable runtime state)
    uv run python experiments/u2_runtime_inspect.py --health-snapshot --config config/test.yaml

    # Check flag policy against environment
    uv run python experiments/u2_runtime_inspect.py --check-flag-policy --env prod

OUTPUT
------
Structured summary to stdout. No files written, no database access.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load and return the configuration file."""
    if not config_path.exists():
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def get_slice_config(
    config: Dict[str, Any],
    slice_name: str,
) -> Dict[str, Any]:
    """Extract slice configuration from the loaded config."""
    # Try dict-style slices
    slice_config = config.get("slices", {}).get(slice_name, {})
    if slice_config:
        return slice_config
    
    # Try list-style slices
    for item in config.get("slices", []):
        if isinstance(item, dict) and item.get("name") == slice_name:
            return item
    
    return {}


def format_seed_preview(seeds: list, max_show: int = 3) -> str:
    """Format seed schedule with first/last preview."""
    n = len(seeds)
    if n == 0:
        return "[]"
    if n <= max_show * 2:
        return str(seeds)
    
    first = seeds[:max_show]
    last = seeds[-max_show:]
    return f"[{', '.join(map(str, first))}, ..., {', '.join(map(str, last))}]"


def inspect_runtime(
    slice_name: str,
    mode: str,
    cycles: int,
    seed: int,
    config_path: Optional[Path] = None,
    output_format: str = "text",
) -> Dict[str, Any]:
    """
    Perform dry-run inspection of runtime configuration.
    
    Returns structured inspection results without executing cycles.
    """
    # Import runtime modules (no side effects on import)
    from experiments.u2.runtime import (
        generate_seed_schedule,
        BaselineOrderingStrategy,
        RflOrderingStrategy,
        RuntimeErrorKind,
        __version__ as runtime_version,
    )
    from experiments.u2.runtime.error_classifier import classify_error
    from experiments.u2.runtime.trace_logger import TraceWriter, TelemetryRecord
    
    # Load config
    config: Dict[str, Any] = {}
    if config_path and config_path.exists():
        config = load_config(config_path)
    
    slice_config = get_slice_config(config, slice_name)
    
    # Generate seed schedule (pure, no side effects)
    seed_schedule = generate_seed_schedule(seed, cycles)
    
    # Determine strategy class
    if mode == "baseline":
        strategy_class = BaselineOrderingStrategy.__name__
    else:
        strategy_class = RflOrderingStrategy.__name__
    
    # Build inspection result
    result = {
        "runtime_version": runtime_version,
        "slice": {
            "name": slice_name,
            "found_in_config": bool(slice_config),
            "parameters": {
                "items_count": len(slice_config.get("items", [])),
                "atoms": slice_config.get("atoms"),
                "depth_max": slice_config.get("depth_max"),
                "max_breadth": slice_config.get("max_breadth"),
                "max_total": slice_config.get("max_total"),
                "prereg_hash": slice_config.get("prereg_hash"),
            },
        },
        "seed_schedule": {
            "initial_seed": seed,
            "num_cycles": cycles,
            "algorithm": seed_schedule.algorithm,
            "first_3_seeds": seed_schedule.cycle_seeds[:3] if cycles >= 3 else seed_schedule.cycle_seeds,
            "last_3_seeds": seed_schedule.cycle_seeds[-3:] if cycles >= 3 else seed_schedule.cycle_seeds,
        },
        "ordering": {
            "mode": mode,
            "strategy_class": strategy_class,
        },
        "components": {
            "error_classifier": "experiments.u2.runtime.error_classifier",
            "error_kinds": [k.name for k in RuntimeErrorKind],
            "trace_logger": "experiments.u2.runtime.trace_logger",
            "trace_writer_class": TraceWriter.__name__,
            "telemetry_record_class": TelemetryRecord.__name__,
        },
        "config_path": str(config_path) if config_path else None,
    }
    
    return result


def print_text_output(result: Dict[str, Any]) -> None:
    """Print human-readable text output."""
    print("=" * 60)
    print("U2 Runtime Inspection Report")
    print("=" * 60)
    print(f"Runtime Version: {result['runtime_version']}")
    print()
    
    # Slice info
    print("SLICE CONFIGURATION")
    print("-" * 40)
    slice_info = result["slice"]
    print(f"  Name: {slice_info['name']}")
    print(f"  Found in config: {slice_info['found_in_config']}")
    params = slice_info["parameters"]
    print(f"  Items count: {params['items_count']}")
    if params.get("atoms"):
        print(f"  Atoms: {params['atoms']}")
    if params.get("depth_max"):
        print(f"  Depth max: {params['depth_max']}")
    if params.get("max_breadth"):
        print(f"  Max breadth: {params['max_breadth']}")
    if params.get("max_total"):
        print(f"  Max total: {params['max_total']}")
    if params.get("prereg_hash"):
        print(f"  Prereg hash: {params['prereg_hash']}")
    print()
    
    # Seed schedule
    print("SEED SCHEDULE")
    print("-" * 40)
    sched = result["seed_schedule"]
    print(f"  Initial seed: {sched['initial_seed']}")
    print(f"  Num cycles: {sched['num_cycles']}")
    print(f"  Algorithm: {sched['algorithm']}")
    print(f"  First 3 seeds: {sched['first_3_seeds']}")
    print(f"  Last 3 seeds: {sched['last_3_seeds']}")
    print()
    
    # Ordering
    print("ORDERING STRATEGY")
    print("-" * 40)
    ordering = result["ordering"]
    print(f"  Mode: {ordering['mode']}")
    print(f"  Strategy class: {ordering['strategy_class']}")
    print()
    
    # Components
    print("RUNTIME COMPONENTS")
    print("-" * 40)
    comp = result["components"]
    print(f"  Error classifier: {comp['error_classifier']}")
    print(f"  Error kinds: {', '.join(comp['error_kinds'])}")
    print(f"  Trace logger: {comp['trace_logger']}")
    print(f"  TraceWriter class: {comp['trace_writer_class']}")
    print(f"  TelemetryRecord class: {comp['telemetry_record_class']}")
    print()
    
    if result.get("config_path"):
        print(f"Config file: {result['config_path']}")
    
    print("=" * 60)
    print("NOTE: This is a dry-run. No cycles executed, no files written.")
    print("=" * 60)


# ============================================================================
# Runtime Contract Introspection
# ============================================================================

def get_contract_info() -> Dict[str, Any]:
    """
    Get runtime API contract information.
    
    Returns structured info about:
    - Runtime version
    - Exported symbols
    - Module invariants
    - Feature flags
    """
    from experiments.u2 import runtime
    from experiments.u2.runtime import __version__, FEATURE_FLAGS
    
    return {
        "version": __version__,
        "symbols": sorted(runtime.__all__),
        "symbol_count": len(runtime.__all__),
        "modules": {
            "seed_manager": [
                "SeedSchedule",
                "generate_seed_schedule",
                "hash_string",
            ],
            "cycle_orchestrator": [
                "CycleState",
                "CycleResult",
                "CycleExecutionError",
                "OrderingStrategy",
                "BaselineOrderingStrategy",
                "RflOrderingStrategy",
                "execute_cycle",
                "get_ordering_strategy",
            ],
            "error_classifier": [
                "RuntimeErrorKind",
                "ErrorContext",
                "classify_error",
                "classify_error_with_context",
                "build_error_result",
            ],
            "trace_logger": [
                "PHASE_II_LABEL",
                "TelemetryRecord",
                "TraceWriter",
                "TraceReader",
                "build_telemetry_record",
            ],
            "feature_flags": [
                "FeatureFlagStability",
                "RuntimeFeatureFlag",
                "FEATURE_FLAGS",
                "get_feature_flag",
                "set_feature_flag",
                "reset_feature_flags",
                "list_feature_flags",
            ],
        },
        "feature_flags": sorted(FEATURE_FLAGS.keys()),
        "invariants": {
            "INV-RUN-1": "No duplication of ordering logic outside cycle_orchestrator",
            "INV-RUN-2": "All runtime modules emit short, actionable error messages",
            "INV-RUN-3": "Runtime is hermetic: no direct filesystem writes except designated paths",
            "INV-RUN-4": "API surface does not grow without explicit justification",
        },
        "guarantees": {
            "determinism": "Same inputs always produce same outputs",
            "no_side_effects": "No I/O or state changes on import",
            "input_validation": "Invalid inputs raise early with clear messages",
        },
    }


def print_contract_text(contract: Dict[str, Any]) -> None:
    """Print contract info in human-readable format."""
    print("=" * 60)
    print("U2 Runtime API Contract")
    print("=" * 60)
    print(f"Version: {contract['version']}")
    print(f"Total exported symbols: {contract['symbol_count']}")
    print()
    
    print("EXPORTED SYMBOLS BY MODULE")
    print("-" * 40)
    for module, symbols in contract["modules"].items():
        print(f"\n  {module}:")
        for sym in symbols:
            print(f"    - {sym}")
    print()
    
    print("RUNTIME INVARIANTS")
    print("-" * 40)
    for code, desc in contract["invariants"].items():
        print(f"  {code}: {desc}")
    print()
    
    print("API GUARANTEES")
    print("-" * 40)
    for key, desc in contract["guarantees"].items():
        print(f"  {key}: {desc}")
    print()
    
    print("=" * 60)
    print("NOTE: Only symbols in __all__ are part of the stable public API.")
    print("=" * 60)


# ============================================================================
# Error Kinds Introspection
# ============================================================================

# Error kind descriptions (short, specific)
ERROR_KIND_DESCRIPTIONS: Dict[str, str] = {
    "SUBPROCESS": "External process returned non-zero exit code",
    "JSON_DECODE": "Failed to parse JSON data at specific position",
    "FILE_NOT_FOUND": "Required file or path does not exist",
    "VALIDATION": "Input failed type or value validation",
    "TIMEOUT": "Operation exceeded time limit",
    "UNKNOWN": "Unclassified exception type",
}


def get_error_kinds_info() -> Dict[str, Any]:
    """
    Get information about all RuntimeErrorKind values.
    
    Returns structured info with descriptions.
    """
    from experiments.u2.runtime import RuntimeErrorKind
    
    kinds = []
    for kind in RuntimeErrorKind:
        kinds.append({
            "name": kind.name,
            "value": kind.value,
            "description": ERROR_KIND_DESCRIPTIONS.get(kind.name, "No description"),
        })
    
    return {
        "error_kinds": kinds,
        "total_count": len(kinds),
        "module": "experiments.u2.runtime.error_classifier",
    }


def print_error_kinds_text(info: Dict[str, Any]) -> None:
    """Print error kinds in human-readable format."""
    print("=" * 60)
    print("U2 Runtime Error Kinds")
    print("=" * 60)
    print(f"Module: {info['module']}")
    print(f"Total kinds: {info['total_count']}")
    print()
    
    print("ERROR KIND CATALOG")
    print("-" * 40)
    for kind in info["error_kinds"]:
        print(f"\n  {kind['name']} ({kind['value']})")
        print(f"    {kind['description']}")
    print()
    
    print("=" * 60)
    print("NOTE: Use classify_error() to map exceptions to these kinds.")
    print("=" * 60)


# ============================================================================
# Feature Flags Introspection
# ============================================================================

def get_feature_flags_info() -> Dict[str, Any]:
    """
    Get information about all runtime feature flags.
    
    Returns structured info about flags, defaults, and stability.
    """
    from experiments.u2.runtime import FEATURE_FLAGS, __version__
    
    flags = []
    for name in sorted(FEATURE_FLAGS.keys()):
        flag = FEATURE_FLAGS[name]
        flags.append(flag.to_dict())
    
    return {
        "runtime_version": __version__,
        "feature_flags": flags,
        "total_count": len(flags),
        "module": "experiments.u2.runtime",
    }


def print_feature_flags_text(info: Dict[str, Any]) -> None:
    """Print feature flags in human-readable format."""
    print("=" * 60)
    print("U2 Runtime Feature Flags")
    print("=" * 60)
    print(f"Runtime Version: {info['runtime_version']}")
    print(f"Total flags: {info['total_count']}")
    print()
    
    print("FEATURE FLAG REGISTRY")
    print("-" * 40)
    for flag in info["feature_flags"]:
        stability_marker = {
            "stable": "[STABLE]",
            "beta": "[BETA]",
            "experimental": "[EXPERIMENTAL]",
        }.get(flag["stability"], "[?]")
        
        default_str = "ON" if flag["default"] else "OFF"
        print(f"\n  {flag['name']} {stability_marker}")
        print(f"    Default: {default_str}")
        print(f"    {flag['description']}")
    print()
    
    print("=" * 60)
    print("NOTE: Use get_feature_flag() to query flags at runtime.")
    print("      Use set_feature_flag() to override for testing.")
    print("=" * 60)


# ============================================================================
# Dry-Run Configuration Validation
# ============================================================================

def validate_config(config_path: Path) -> Dict[str, Any]:
    """
    Validate a configuration file without running an experiment.
    
    Checks:
    - File exists and is valid YAML
    - Required fields are present
    - Mode maps to valid ordering strategy
    - No missing runtime components
    
    Returns structured validation result.
    """
    from experiments.u2.runtime import (
        get_ordering_strategy,
        __version__ as runtime_version,
    )
    
    errors: list = []
    warnings: list = []
    
    # Check file exists
    if not config_path.exists():
        return {
            "status": "ERROR",
            "runtime_version": runtime_version,
            "config_path": str(config_path),
            "errors": [{"code": "CONFIG_NOT_FOUND", "message": f"Config file not found: {config_path}"}],
            "warnings": [],
        }
    
    # Load config
    try:
        config = load_config(config_path)
    except Exception as e:
        return {
            "status": "ERROR",
            "runtime_version": runtime_version,
            "config_path": str(config_path),
            "errors": [{"code": "CONFIG_PARSE_ERROR", "message": f"Failed to parse config: {e}"}],
            "warnings": [],
        }
    
    # Check for required top-level fields
    if "slices" not in config:
        warnings.append({
            "code": "NO_SLICES",
            "message": "Config does not define any slices.",
        })
    
    # Validate each slice
    slices = config.get("slices", {})
    if isinstance(slices, dict):
        slice_items = slices.items()
    elif isinstance(slices, list):
        slice_items = [(s.get("name", f"slice_{i}"), s) for i, s in enumerate(slices)]
    else:
        slice_items = []
        errors.append({
            "code": "INVALID_SLICES_FORMAT",
            "message": "slices must be a dict or list",
        })
    
    for slice_name, slice_config in slice_items:
        # Check mode if specified
        mode = slice_config.get("mode")
        if mode:
            if mode not in ("baseline", "rfl"):
                errors.append({
                    "code": "INVALID_MODE",
                    "message": f"Slice '{slice_name}' has invalid mode: {mode}. Must be 'baseline' or 'rfl'.",
                })
        
        # Check cycles if specified
        cycles = slice_config.get("cycles")
        if cycles is not None and (not isinstance(cycles, int) or cycles < 0):
            errors.append({
                "code": "INVALID_CYCLES",
                "message": f"Slice '{slice_name}' has invalid cycles: {cycles}. Must be non-negative integer.",
            })
        
        # Check seed if specified
        seed = slice_config.get("seed")
        if seed is not None and not isinstance(seed, int):
            errors.append({
                "code": "INVALID_SEED",
                "message": f"Slice '{slice_name}' has invalid seed: {seed}. Must be integer.",
            })
    
    # Determine overall status
    status = "OK" if not errors else "ERROR"
    
    return {
        "status": status,
        "runtime_version": runtime_version,
        "config_path": str(config_path),
        "slices_found": len(list(slice_items)) if slice_items else 0,
        "errors": errors,
        "warnings": warnings,
    }


# ============================================================================
# Health Snapshot
# ============================================================================

def get_health_snapshot(config_path: Optional[str]) -> Dict[str, Any]:
    """
    Build a runtime health snapshot.
    
    This includes config validation if a path is provided.
    """
    from experiments.u2.runtime import (
        build_runtime_health_snapshot,
        validate_flag_policy,
        summarize_runtime_for_global_health,
    )
    
    # First validate config if provided
    config_validation = None
    if config_path:
        path = Path(config_path)
        config_validation = validate_config(path)
    
    # Build health snapshot
    snapshot = build_runtime_health_snapshot(
        config_path=config_path,
        config_validation_result=config_validation,
    )
    
    # Add global health summary (default to dev environment for snapshot)
    policy_result = validate_flag_policy("dev")
    global_summary = summarize_runtime_for_global_health(snapshot, policy_result)
    
    return {
        **snapshot,
        "global_health_summary": global_summary,
    }


def print_health_snapshot_text(result: Dict[str, Any]) -> None:
    """Print health snapshot in human-readable format."""
    status = result.get("global_health_summary", {}).get("status", "UNKNOWN")
    status_symbol = {"OK": "[OK]", "WARN": "[WARN]", "BLOCK": "[BLOCK]"}.get(status, "[?]")
    
    print("=" * 60)
    print(f"U2 Runtime Health Snapshot {status_symbol}")
    print("=" * 60)
    print(f"Schema Version: {result.get('schema_version')}")
    print(f"Runtime Version: {result.get('runtime_version')}")
    print(f"Config Valid: {result.get('config_valid')}")
    print()
    
    print("ACTIVE FLAGS")
    print("-" * 40)
    active_flags = result.get("active_flags", {})
    flag_stabilities = result.get("flag_stabilities", {})
    for name, value in sorted(active_flags.items()):
        stability = flag_stabilities.get(name, "?")
        value_str = "ON" if value else "OFF"
        print(f"  {name}: {value_str} [{stability}]")
    print()
    
    if result.get("config_errors"):
        print("CONFIG ERRORS")
        print("-" * 40)
        for err in result["config_errors"]:
            print(f"  [{err.get('code', 'ERROR')}] {err.get('message', 'Unknown error')}")
        print()
    
    summary = result.get("global_health_summary", {})
    print("GLOBAL HEALTH SUMMARY")
    print("-" * 40)
    print(f"  Status: {summary.get('status', 'UNKNOWN')}")
    print(f"  Runtime OK: {summary.get('runtime_ok')}")
    print(f"  Flag Policy OK: {summary.get('flag_policy_ok')}")
    if summary.get("beta_flags_active"):
        print(f"  Beta Flags Active: {', '.join(summary['beta_flags_active'])}")
    if summary.get("experimental_flags_active"):
        print(f"  Experimental Flags Active: {', '.join(summary['experimental_flags_active'])}")
    print()
    
    print("=" * 60)
    print("NOTE: This snapshot is machine-readable. Use --json for structured output.")
    print("=" * 60)


# ============================================================================
# Flag Policy Validation
# ============================================================================

def get_flag_policy_result(env_context: str) -> Dict[str, Any]:
    """
    Validate flag policy for the given environment.
    """
    from experiments.u2.runtime import validate_flag_policy
    return validate_flag_policy(env_context)


def print_flag_policy_text(result: Dict[str, Any]) -> None:
    """Print flag policy result in human-readable format."""
    status_symbol = "[OK]" if result["policy_ok"] else "[VIOLATION]"
    
    print("=" * 60)
    print(f"U2 Flag Policy Check {status_symbol}")
    print("=" * 60)
    print(f"Environment: {result.get('env_context')}")
    print(f"Flags Checked: {result.get('flags_checked')}")
    print(f"Policy OK: {result.get('policy_ok')}")
    print()
    
    violations = result.get("violations", [])
    if violations:
        print("POLICY VIOLATIONS")
        print("-" * 40)
        for v in violations:
            print(f"  Flag: {v.get('flag_name')}")
            print(f"    Stability: {v.get('stability')}")
            print(f"    Current Value: {v.get('current_value')}")
            print(f"    Reason: {v.get('reason')}")
            print()
    else:
        print("No policy violations.")
        print()
    
    print("=" * 60)
    print("POLICY RULES:")
    print("  - STABLE flags: May be freely toggled in any environment")
    print("  - BETA flags: Only allowed ON in 'dev' environment")
    print("  - EXPERIMENTAL flags: Must be OFF in 'ci' and 'prod'")
    print("=" * 60)


def print_validation_text(result: Dict[str, Any]) -> None:
    """Print validation result in human-readable format."""
    status_symbol = "[OK]" if result["status"] == "OK" else "[ERROR]"
    
    print("=" * 60)
    print(f"U2 Runtime Config Validation {status_symbol}")
    print("=" * 60)
    print(f"Runtime Version: {result['runtime_version']}")
    print(f"Config Path: {result['config_path']}")
    print(f"Slices Found: {result.get('slices_found', 'N/A')}")
    print(f"Status: {result['status']}")
    print()
    
    if result["errors"]:
        print("ERRORS")
        print("-" * 40)
        for err in result["errors"]:
            print(f"  [{err['code']}] {err['message']}")
        print()
    
    if result["warnings"]:
        print("WARNINGS")
        print("-" * 40)
        for warn in result["warnings"]:
            print(f"  [{warn['code']}] {warn['message']}")
        print()
    
    if not result["errors"] and not result["warnings"]:
        print("No errors or warnings.")
        print()
    
    print("=" * 60)
    print("NOTE: This is a dry-run validation. No experiment was executed.")
    print("=" * 60)


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="U2 Runtime Introspection Tool (PHASE II)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Inspect baseline mode with 10 cycles
    python experiments/u2_runtime_inspect.py --slice test_slice --mode baseline --cycles 10 --seed 42

    # Output as JSON for programmatic use
    python experiments/u2_runtime_inspect.py --slice test --mode rfl --cycles 100 --seed 12345 --json

    # Show runtime API contract
    python experiments/u2_runtime_inspect.py --show-contract

    # Show all error kinds with descriptions
    python experiments/u2_runtime_inspect.py --show-error-kinds

    # Show all feature flags
    python experiments/u2_runtime_inspect.py --show-feature-flags

    # Validate a config file (dry-run)
    python experiments/u2_runtime_inspect.py --dry-run-config config/my_config.yaml

    # Generate health snapshot
    python experiments/u2_runtime_inspect.py --health-snapshot --config config/test.yaml

    # Check flag policy for environment
    python experiments/u2_runtime_inspect.py --check-flag-policy --env prod

This tool is read-only: no logs, no manifests, no database access.
""",
    )
    
    # Inspection mode arguments (mutually exclusive groups)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--show-contract",
        action="store_true",
        help="Show runtime API contract (version, symbols, invariants).",
    )
    mode_group.add_argument(
        "--show-error-kinds",
        action="store_true",
        help="Show all RuntimeErrorKind values with descriptions.",
    )
    mode_group.add_argument(
        "--show-feature-flags",
        action="store_true",
        help="Show all feature flags with defaults and stability.",
    )
    mode_group.add_argument(
        "--dry-run-config",
        type=str,
        metavar="PATH",
        help="Validate a config file without running an experiment.",
    )
    mode_group.add_argument(
        "--health-snapshot",
        action="store_true",
        help="Generate machine-readable runtime health snapshot.",
    )
    mode_group.add_argument(
        "--check-flag-policy",
        action="store_true",
        help="Check flag policy against environment (requires --env).",
    )
    
    # Configuration inspection arguments
    parser.add_argument(
        "--slice",
        type=str,
        help="Slice name to inspect.",
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "rfl"],
        help="Execution mode to inspect.",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        help="Number of cycles to inspect.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master seed (default: 42).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/curriculum_uplift_phase2.yaml",
        help="Path to config file (default: config/curriculum_uplift_phase2.yaml).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of text.",
    )
    parser.add_argument(
        "--output",
        type=str,
        metavar="PATH",
        help="Write output to file instead of stdout (for --health-snapshot).",
    )
    parser.add_argument(
        "--env",
        type=str,
        choices=["dev", "ci", "prod"],
        help="Environment context for policy validation (dev, ci, prod).",
    )
    
    args = parser.parse_args()
    
    try:
        # Handle --show-contract
        if args.show_contract:
            contract = get_contract_info()
            if args.json:
                print(json.dumps(contract, indent=2))
            else:
                print_contract_text(contract)
            return 0
        
        # Handle --show-error-kinds
        if args.show_error_kinds:
            error_info = get_error_kinds_info()
            if args.json:
                print(json.dumps(error_info, indent=2))
            else:
                print_error_kinds_text(error_info)
            return 0
        
        # Handle --show-feature-flags
        if args.show_feature_flags:
            flags_info = get_feature_flags_info()
            if args.json:
                print(json.dumps(flags_info, indent=2))
            else:
                print_feature_flags_text(flags_info)
            return 0
        
        # Handle --dry-run-config
        if args.dry_run_config:
            config_path = Path(args.dry_run_config)
            validation_result = validate_config(config_path)
            if args.json:
                print(json.dumps(validation_result, indent=2))
            else:
                print_validation_text(validation_result)
            # Return non-zero exit code on validation errors
            return 0 if validation_result["status"] == "OK" else 1
        
        # Handle --health-snapshot
        if args.health_snapshot:
            health_result = get_health_snapshot(args.config)
            output_json = json.dumps(health_result, indent=2)
            
            if args.output:
                # Write to file
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(output_json)
                print(f"Health snapshot written to: {output_path}")
            elif args.json:
                print(output_json)
            else:
                print_health_snapshot_text(health_result)
            return 0
        
        # Handle --check-flag-policy
        if args.check_flag_policy:
            if not args.env:
                print(
                    "ERROR: --check-flag-policy requires --env (dev, ci, or prod).",
                    file=sys.stderr,
                )
                return 1
            
            policy_result = get_flag_policy_result(args.env)
            if args.json:
                print(json.dumps(policy_result, indent=2))
            else:
                print_flag_policy_text(policy_result)
            
            # Return non-zero if policy violations exist
            return 0 if policy_result["policy_ok"] else 1
        
        # For config inspection, require slice/mode/cycles
        if not args.slice or not args.mode or args.cycles is None:
            print(
                "ERROR: --slice, --mode, and --cycles are required for config inspection.\n"
                "       Use --show-contract, --show-error-kinds, --show-feature-flags,\n"
                "       or --dry-run-config for other introspection modes.",
                file=sys.stderr,
            )
            return 1
        
        # Validate inputs
        if args.cycles < 0:
            print("ERROR: cycles must be non-negative", file=sys.stderr)
            return 1
        
        config_path = Path(args.config) if args.config else None
        
        result = inspect_runtime(
            slice_name=args.slice,
            mode=args.mode,
            cycles=args.cycles,
            seed=args.seed,
            config_path=config_path,
        )
        
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print_text_output(result)
        
        return 0
        
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

