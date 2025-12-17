#!/usr/bin/env python3
"""
Phase X P4: First-Light Shadow Harness CLI

Command-line harness for running P4 shadow experiments with real runner coupling.
See docs/system_law/Phase_X_P4_Spec.md for full specification.

SHADOW MODE CONTRACT:
- This harness runs in SHADOW mode only
- All outputs are observational
- No governance decisions are modified
- Uses MockTelemetryProvider for testing

Usage:
    python scripts/usla_first_light_p4_harness.py --cycles 100 --seed 42 --output-dir results/p4

Artifacts Generated:
    1. real_cycles.jsonl     - Real runner observations
    2. twin_predictions.jsonl - Twin runner predictions
    3. divergence_log.jsonl  - Divergence analysis results
    4. p4_summary.json       - Final summary report
    5. twin_accuracy.json    - Twin prediction accuracy metrics
    6. run_config.json       - Run configuration

Status: P4 IMPLEMENTATION
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.logging.jsonl_writer import JsonlWriter
from backend.health.u2_dynamics_tile import (
    attach_u2_dynamics_to_p4_summary,
    build_u2_dynamics_window_metrics,
    DEFAULT_U2_DYNAMICS_WINDOW_SIZE,
)
from backend.health.identity_alignment_checker import (
    CheckResult,
    check_p5_identity_alignment,
)
from backend.telemetry.rtts_window_validator import (
    RTTSWindowValidator,
    rtts_validate_window,
    RTTS_VALIDATION_SCHEMA_VERSION,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase X P4 First-Light Shadow Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=100,
        help="Number of cycles to observe (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/first_light_p4",
        help="Output directory for artifacts (default: results/first_light_p4)",
    )
    parser.add_argument(
        "--slice",
        type=str,
        default="arithmetic_simple",
        help="Slice name (default: arithmetic_simple)",
    )
    parser.add_argument(
        "--runner-type",
        type=str,
        choices=["u2", "rfl"],
        default="u2",
        help="Runner type (default: u2)",
    )
    parser.add_argument(
        "--tau-0",
        type=float,
        default=0.20,
        help="Initial tau threshold (default: 0.20)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration without running",
    )
    # P5: Telemetry adapter selection
    parser.add_argument(
        "--telemetry-adapter",
        type=str,
        choices=["mock", "real"],
        default="mock",
        help="Telemetry adapter type (default: mock). Use 'real' for P5 POC.",
    )
    parser.add_argument(
        "--adapter-config",
        type=str,
        default=None,
        help="Path to adapter configuration JSON (optional, for 'real' adapter)",
    )
    # P5 Identity Pre-flight arguments
    parser.add_argument(
        "--synthetic-config",
        type=str,
        default=None,
        help="Path to synthetic slice config JSON/YAML for P5 identity pre-flight check",
    )
    parser.add_argument(
        "--prod-config",
        type=str,
        default=None,
        help="Path to production slice config JSON/YAML for P5 identity pre-flight check",
    )
    parser.add_argument(
        "--p4-evidence-pack",
        type=str,
        default=None,
        help="Path to P4 evidence pack JSON for identity baseline comparison",
    )
    # P5 Diagnostic emission (SHADOW MODE)
    parser.add_argument(
        "--emit-p5-diagnostic",
        action="store_true",
        help="Emit P5 divergence diagnostic at end of run (SHADOW MODE only)",
    )
    # P5.2 RTTS Validation emission (SHADOW MODE)
    parser.add_argument(
        "--emit-rtts-validation",
        action="store_true",
        help="Emit rtts_validation.json with RTTS mock detection and validation (SHADOW MODE only)",
    )
    # P5 UPGRADE-1: Per-component learning rates for Twin
    parser.add_argument(
        "--twin-lr-H",
        type=float,
        default=None,
        help="Learning rate for Twin H component (default: 0.1)",
    )
    parser.add_argument(
        "--twin-lr-rho",
        type=float,
        default=None,
        help="Learning rate for Twin rho component (default: 0.1)",
    )
    parser.add_argument(
        "--twin-lr-tau",
        type=float,
        default=None,
        help="Learning rate for Twin tau component (default: 0.05)",
    )
    parser.add_argument(
        "--twin-lr-beta",
        type=float,
        default=None,
        help="Learning rate for Twin beta component (default: 0.1)",
    )
    return parser.parse_args()


JSONL_WRITE_KWARGS = {"ensure_ascii": True, "separators": (",", ": ")}


def run_identity_preflight(
    synthetic_config_path: Optional[str],
    prod_config_path: Optional[str],
    p4_evidence_pack_path: Optional[str],
) -> Optional[Dict[str, Any]]:
    """
    Run P5 identity pre-flight check.

    SHADOW MODE CONTRACT:
    - This function is advisory only
    - It does NOT block harness execution regardless of result
    - Results are recorded in run_config.json for observational purposes

    Args:
        synthetic_config_path: Path to synthetic slice config
        prod_config_path: Path to production slice config
        p4_evidence_pack_path: Optional path to P4 evidence pack

    Returns:
        Dict with identity_preflight results, or None if skipped
    """
    if not synthetic_config_path or not prod_config_path:
        return None

    try:
        # Load configs
        syn_path = Path(synthetic_config_path)
        prod_path = Path(prod_config_path)

        if not syn_path.exists():
            return {
                "status": "ERROR",
                "error": f"Synthetic config not found: {synthetic_config_path}",
                "skipped": False,
            }

        if not prod_path.exists():
            return {
                "status": "ERROR",
                "error": f"Production config not found: {prod_config_path}",
                "skipped": False,
            }

        # Load JSON/YAML configs
        def load_config(path: Path) -> Dict[str, Any]:
            content = path.read_text(encoding="utf-8")
            if path.suffix in (".yaml", ".yml"):
                try:
                    import yaml
                    return yaml.safe_load(content)
                except ImportError:
                    return json.loads(content)
            return json.loads(content)

        synthetic_config = load_config(syn_path)
        production_config = load_config(prod_path)

        # Load P4 evidence pack if provided
        evidence_pack: Optional[Dict[str, Any]] = None
        if p4_evidence_pack_path:
            evidence_path = Path(p4_evidence_pack_path)
            if evidence_path.exists():
                evidence_pack = load_config(evidence_path)

        # Run identity alignment check
        report = check_p5_identity_alignment(
            synthetic_config,
            production_config,
            evidence_pack,
        )

        # Extract top reasons for non-OK status
        top_reasons: List[str] = []
        if report.overall_status != CheckResult.OK:
            # Collect blocking issues first, then investigation items
            top_reasons = report.blocking_issues[:3] + report.investigation_items[:2]
            top_reasons = top_reasons[:5]  # Limit to 5 total

        return {
            "status": report.overall_status.value,
            "exit_code": report.get_exit_code(),
            "synthetic_fingerprint": report.synthetic_fingerprint,
            "production_fingerprint": report.production_fingerprint,
            "fingerprint_match": report.synthetic_fingerprint == report.production_fingerprint,
            "invariant_summary": {
                check.invariant: check.status.value
                for check in report.checks
                if check.invariant
            },
            "blocking_issues": report.blocking_issues,
            "investigation_items": report.investigation_items,
            "top_reasons": top_reasons,
            "timestamp": report.timestamp,
            "skipped": False,
            # Full report for detailed analysis
            "full_report": report.to_dict(),
        }

    except Exception as e:
        return {
            "status": "ERROR",
            "error": str(e),
            "skipped": False,
        }


# Schema version for identity preflight artifact
IDENTITY_PREFLIGHT_SCHEMA_VERSION = "1.0.0"


def save_identity_preflight_artifact(
    output_dir: Path,
    identity_preflight_result: Dict[str, Any],
) -> Path:
    """
    Save identity preflight results as a dedicated artifact file.

    SHADOW MODE CONTRACT:
    - This artifact is for reproducible audits only
    - It does NOT gate any operations
    - The file is advisory and observational

    Args:
        output_dir: Output directory for the artifact
        identity_preflight_result: Identity preflight check result

    Returns:
        Path to the created artifact file
    """
    # Build standardized artifact structure
    artifact: Dict[str, Any] = {
        "schema_version": IDENTITY_PREFLIGHT_SCHEMA_VERSION,
        "timestamp": identity_preflight_result.get("timestamp", datetime.now(timezone.utc).isoformat()),
        "status": identity_preflight_result.get("status"),
        "fingerprint_match": identity_preflight_result.get("fingerprint_match", False),
        "synthetic_fingerprint": identity_preflight_result.get("synthetic_fingerprint"),
        "production_fingerprint": identity_preflight_result.get("production_fingerprint"),
        "top_reasons": identity_preflight_result.get("top_reasons", []),
        "checks": [],
        "invariant_summary": identity_preflight_result.get("invariant_summary", {}),
        "mode": "SHADOW",
        "shadow_mode_contract": {
            "observational_only": True,
            "no_control_flow_influence": True,
            "advisory_status": identity_preflight_result.get("status", "UNKNOWN"),
        },
    }

    # Extract checks from full_report if available
    full_report = identity_preflight_result.get("full_report", {})
    if full_report and "checks" in full_report:
        artifact["checks"] = full_report["checks"]
    else:
        # Build checks from invariant_summary
        for inv, status in artifact["invariant_summary"].items():
            artifact["checks"].append({
                "invariant": inv,
                "status": status,
            })

    # Add error info if present
    if "error" in identity_preflight_result:
        artifact["error"] = identity_preflight_result["error"]

    # Write artifact file
    artifact_path = output_dir / "p5_identity_preflight.json"
    with open(artifact_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    return artifact_path


def run_harness(args: argparse.Namespace) -> int:
    """
    Run the P4 shadow harness.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Import here to avoid import errors if dependencies missing
    from backend.topology.first_light.config_p4 import FirstLightConfigP4
    from backend.topology.first_light.runner_p4 import FirstLightShadowRunnerP4
    from backend.topology.first_light.telemetry_adapter import MockTelemetryProvider
    from backend.topology.first_light.real_telemetry_adapter import (
        RealTelemetryAdapter,
        RealTelemetryAdapterConfig,
        AdapterMode,
    )

    # Create output directory with timestamp
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"p4_{timestamp}"
    output_dir = Path(args.output_dir) / run_id

    # Get telemetry adapter type
    adapter_type = getattr(args, "telemetry_adapter", "mock")
    adapter_config_path = getattr(args, "adapter_config", None)

    # Load adapter config if provided
    adapter_config: Optional[RealTelemetryAdapterConfig] = None
    adapter_mode = "synthetic"  # Default mode for real adapter
    if adapter_config_path:
        adapter_config = RealTelemetryAdapterConfig.from_json_file(adapter_config_path)
        adapter_mode = adapter_config.mode

    # Dry run mode
    if args.dry_run:
        print("=" * 60)
        print("DRY RUN - Configuration Preview")
        print("=" * 60)
        print(f"  Cycles:      {args.cycles}")
        print(f"  Seed:        {args.seed}")
        print(f"  Output dir:  {output_dir}")
        print(f"  Slice:       {args.slice}")
        print(f"  Runner type: {args.runner_type}")
        print(f"  tau_0:       {args.tau_0}")
        print(f"  Adapter:     {adapter_type}")
        if adapter_config:
            print(f"  Adapter cfg: {adapter_config_path}")
            print(f"  Adapter mode:{adapter_mode}")
            if adapter_config.trace_path:
                print(f"  Trace path:  {adapter_config.trace_path}")
        print("=" * 60)
        print("No output files generated (dry run)")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase X P4: First-Light Shadow Harness")
    print("=" * 60)
    print(f"  Run ID:      {run_id}")
    print(f"  Cycles:      {args.cycles}")
    print(f"  Seed:        {args.seed}")
    print(f"  Output dir:  {output_dir}")
    print(f"  Slice:       {args.slice}")
    print(f"  Runner type: {args.runner_type}")
    print(f"  tau_0:       {args.tau_0}")
    print(f"  Adapter:     {adapter_type}")
    if adapter_config:
        print(f"  Adapter cfg: {adapter_config_path}")
        print(f"  Adapter mode:{adapter_mode}")
        if adapter_config.trace_path:
            print(f"  Trace path:  {adapter_config.trace_path}")
    print("=" * 60)
    print()

    # =========================================================================
    # P5 IDENTITY PRE-FLIGHT CHECK (SHADOW MODE)
    # =========================================================================
    # Run identity alignment check when real adapter is selected and configs provided.
    # SHADOW MODE CONTRACT:
    # - This check is ADVISORY ONLY
    # - It does NOT block harness execution regardless of result
    # - Results are recorded in run_config.json under identity_preflight
    identity_preflight_result: Optional[Dict[str, Any]] = None
    synthetic_config_path = getattr(args, "synthetic_config", None)
    prod_config_path = getattr(args, "prod_config", None)
    p4_evidence_pack_path = getattr(args, "p4_evidence_pack", None)

    if adapter_type == "real" and (synthetic_config_path or prod_config_path):
        print("P5 Identity Pre-Flight Check (SHADOW MODE - advisory only)")
        print("-" * 60)

        if synthetic_config_path and prod_config_path:
            identity_preflight_result = run_identity_preflight(
                synthetic_config_path,
                prod_config_path,
                p4_evidence_pack_path,
            )

            if identity_preflight_result:
                status = identity_preflight_result.get("status", "UNKNOWN")
                fp_match = identity_preflight_result.get("fingerprint_match", False)

                print(f"  Status:           {status}")
                print(f"  Fingerprint Match: {'YES' if fp_match else 'NO'}")

                if status == "OK":
                    print("  [OK] Safe to proceed with real telemetry")
                elif status == "INVESTIGATE":
                    print("  [INVESTIGATE] Review recommended before production")
                    for reason in identity_preflight_result.get("top_reasons", [])[:3]:
                        print(f"    - {reason}")
                elif status == "BLOCK":
                    print("  [BLOCK] Issues detected (advisory only, not blocking)")
                    for reason in identity_preflight_result.get("top_reasons", [])[:3]:
                        print(f"    - {reason}")
                elif status == "ERROR":
                    print(f"  [ERROR] {identity_preflight_result.get('error', 'Unknown error')}")

                print()
                print("  NOTE: SHADOW MODE - harness will continue regardless of status")
        else:
            print("  Skipped: Both --synthetic-config and --prod-config required")
            identity_preflight_result = {"status": "SKIPPED", "skipped": True}

        print("-" * 60)
        print()

    try:
        # Create telemetry provider based on adapter type
        if adapter_type == "real":
            if adapter_config:
                # P5 BASELINE: Use config-driven adapter
                if adapter_config.mode == AdapterMode.TRACE:
                    print(f"Using RealTelemetryAdapter (TRACE mode: {adapter_config.trace_path})")
                else:
                    print("Using RealTelemetryAdapter (SYNTHETIC mode)")

                # Override config with CLI args where appropriate
                if args.seed is not None:
                    adapter_config.seed = args.seed
                if args.slice:
                    adapter_config.slice_name = args.slice
                if args.runner_type:
                    adapter_config.runner_type = args.runner_type

                provider = RealTelemetryAdapter.from_config(adapter_config)
            else:
                # P5 POC: Use RealTelemetryAdapter in synthetic mode
                print("Using RealTelemetryAdapter (P5 POC synthetic mode)")
                provider = RealTelemetryAdapter(
                    runner_type=args.runner_type,
                    slice_name=args.slice,
                    seed=args.seed,
                    source_label="P5_ADAPTER_STUB",
                    mode=AdapterMode.SYNTHETIC,
                )
        else:
            # Default: Use MockTelemetryProvider
            provider = MockTelemetryProvider(
                runner_type=args.runner_type,
                slice_name=args.slice,
                seed=args.seed,
            )

        # Build twin_lr_overrides from CLI args (P5 UPGRADE-1)
        twin_lr_overrides: Optional[Dict[str, float]] = None
        lr_H = getattr(args, "twin_lr_H", None)
        lr_rho = getattr(args, "twin_lr_rho", None)
        lr_tau = getattr(args, "twin_lr_tau", None)
        lr_beta = getattr(args, "twin_lr_beta", None)
        if any(lr is not None for lr in [lr_H, lr_rho, lr_tau, lr_beta]):
            twin_lr_overrides = {}
            if lr_H is not None:
                twin_lr_overrides["H"] = lr_H
            if lr_rho is not None:
                twin_lr_overrides["rho"] = lr_rho
            if lr_tau is not None:
                twin_lr_overrides["tau"] = lr_tau
            if lr_beta is not None:
                twin_lr_overrides["beta"] = lr_beta
            print(f"  Twin LR overrides: {twin_lr_overrides}")

        # Create P4 configuration
        config = FirstLightConfigP4(
            slice_name=args.slice,
            runner_type=args.runner_type,
            total_cycles=args.cycles,
            tau_0=args.tau_0,
            telemetry_adapter=provider,
            log_dir=str(output_dir),
            run_id=run_id,
            twin_lr_overrides=twin_lr_overrides,
        )

        # Validate configuration
        config.validate_or_raise()

        # Save run configuration
        save_run_config(output_dir, args, run_id, identity_preflight_result)

        # Save identity preflight artifact file (if preflight was run)
        # This creates a dedicated p5_identity_preflight.json for reproducible audits
        if identity_preflight_result and not identity_preflight_result.get("skipped"):
            preflight_artifact_path = save_identity_preflight_artifact(
                output_dir, identity_preflight_result
            )
            print(f"  Identity preflight artifact: {preflight_artifact_path.name}")

        # Create and run P4 runner
        print("Starting P4 shadow observation...")
        runner = FirstLightShadowRunnerP4(config, seed=args.seed)

        # Track progress
        cycle_count = 0
        real_cycles_path = output_dir / "real_cycles.jsonl"
        twin_predictions_path = output_dir / "twin_predictions.jsonl"
        divergence_log_path = output_dir / "divergence_log.jsonl"

        with JsonlWriter(str(real_cycles_path), json_kwargs=JSONL_WRITE_KWARGS) as real_writer, \
             JsonlWriter(str(twin_predictions_path), json_kwargs=JSONL_WRITE_KWARGS) as twin_writer, \
             JsonlWriter(str(divergence_log_path), json_kwargs=JSONL_WRITE_KWARGS) as div_writer:

            for observation in runner.run_cycles(args.cycles):
                cycle_count += 1

                # Write real observation
                real_writer.write(observation.to_dict())

                # Write twin prediction (most recent)
                twin_obs = runner.get_twin_observations()[-1]
                twin_writer.write(twin_obs.to_dict())

                # Write divergence snapshot (most recent)
                div_snap = runner.get_divergence_snapshots()[-1]
                div_writer.write(div_snap.to_dict())

                # Progress indicator
                if cycle_count % 10 == 0 or cycle_count == args.cycles:
                    print(f"  Observed {cycle_count}/{args.cycles} cycles...")

        print()
        print(f"Completed {cycle_count} cycles")

        # Finalize and get result
        result = runner.finalize()
        u2_dynamics_tile = None
        try:
            u2_dynamics_tile = build_u2_dynamics_window_metrics(
                real_cycles_path, window_size=DEFAULT_U2_DYNAMICS_WINDOW_SIZE
            )
        except Exception:
            # SHADOW MODE: Advisory-only; ignore any computation failures
            u2_dynamics_tile = None

        # Save summary
        save_summary(output_dir, result, u2_dynamics_tile=u2_dynamics_tile)

        # Save twin accuracy metrics
        save_twin_accuracy(output_dir, runner)

        # P5 Diagnostic emission (SHADOW MODE)
        emit_p5_diagnostic = getattr(args, "emit_p5_diagnostic", False)
        if emit_p5_diagnostic:
            try:
                save_p5_diagnostic(
                    output_dir=output_dir,
                    divergence_log_path=divergence_log_path,
                    result=result,
                    run_id=run_id,
                    cycle_count=cycle_count,
                )
                print("  P5 diagnostic emitted (SHADOW MODE)")
            except Exception as p5_err:
                # SHADOW MODE: Never fail on diagnostic emission
                print(f"  P5 diagnostic emission skipped: {p5_err}")

        # P5.2 RTTS Validation emission (SHADOW MODE)
        emit_rtts_validation = getattr(args, "emit_rtts_validation", False)
        if emit_rtts_validation:
            try:
                rtts_path = save_rtts_validation(
                    output_dir=output_dir,
                    real_cycles_path=real_cycles_path,
                    run_id=run_id,
                    cycle_count=cycle_count,
                )
                if rtts_path:
                    print(f"  RTTS validation emitted: {rtts_path.name} (SHADOW MODE)")
                else:
                    print("  RTTS validation skipped: no telemetry data")
            except Exception as rtts_err:
                # SHADOW MODE: Never fail on RTTS validation emission
                print(f"  RTTS validation emission skipped: {rtts_err}")

        # Print summary
        print()
        print("=" * 60)
        print("P4 Shadow Observation Complete")
        print("=" * 60)
        print(f"  Cycles completed:    {result.cycles_completed}")
        print(f"  Success rate:        {result.u2_success_rate_final:.2%}")
        print(f"  Divergence rate:     {result.divergence_rate:.2%}")
        print(f"  Total divergences:   {result.total_divergences}")
        print(f"  Max div streak:      {result.max_divergence_streak}")
        print()
        print("Twin Prediction Accuracy:")
        print(f"  Success:             {result.twin_success_prediction_accuracy:.2%}")
        print(f"  Blocked:             {result.twin_blocked_prediction_accuracy:.2%}")
        print(f"  Omega:               {result.twin_omega_prediction_accuracy:.2%}")
        print(f"  Hard OK:             {result.twin_hard_ok_prediction_accuracy:.2%}")
        print()
        print(f"Artifacts written to: {output_dir}")
        print("=" * 60)

        # Check success criteria
        criteria = result.meets_success_criteria()
        all_passed = all(criteria.values())

        print()
        print("Success Criteria (SHADOW MODE - observational only):")
        for criterion, passed in criteria.items():
            status = "PASS" if passed else "OBSERVE"
            print(f"  {criterion}: {status}")
        print()

        return 0 if all_passed else 0  # Always return 0 in shadow mode

    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()

        # Save error report
        error_report = {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
        }

        error_path = output_dir / "error_report.json"
        with open(error_path, "w") as f:
            json.dump(error_report, f, indent=2)

        return 1


def save_run_config(
    output_dir: Path,
    args: argparse.Namespace,
    run_id: str,
    identity_preflight: Optional[Dict[str, Any]] = None,
) -> None:
    """Save run configuration to JSON file.

    Args:
        output_dir: Output directory path
        args: Parsed command-line arguments
        run_id: Run identifier
        identity_preflight: Optional P5 identity pre-flight check result
    """
    adapter_type = getattr(args, "telemetry_adapter", "mock")
    adapter_config_path = getattr(args, "adapter_config", None)

    # Determine telemetry source for P5 status reporting
    if adapter_type == "mock":
        telemetry_source = "mock"
    elif adapter_config_path:
        # Load config to determine mode
        from backend.topology.first_light.real_telemetry_adapter import (
            RealTelemetryAdapterConfig,
            AdapterMode,
        )
        try:
            adapter_cfg = RealTelemetryAdapterConfig.from_json_file(adapter_config_path)
            telemetry_source = f"real_{adapter_cfg.mode}"
        except Exception:
            telemetry_source = "real_synthetic"
    else:
        telemetry_source = "real_synthetic"

    # Build twin_lr_overrides from CLI args for config logging
    lr_H = getattr(args, "twin_lr_H", None)
    lr_rho = getattr(args, "twin_lr_rho", None)
    lr_tau = getattr(args, "twin_lr_tau", None)
    lr_beta = getattr(args, "twin_lr_beta", None)
    twin_lr_overrides: Optional[Dict[str, float]] = None
    if any(lr is not None for lr in [lr_H, lr_rho, lr_tau, lr_beta]):
        twin_lr_overrides = {}
        if lr_H is not None:
            twin_lr_overrides["H"] = lr_H
        if lr_rho is not None:
            twin_lr_overrides["rho"] = lr_rho
        if lr_tau is not None:
            twin_lr_overrides["tau"] = lr_tau
        if lr_beta is not None:
            twin_lr_overrides["beta"] = lr_beta

    config: Dict[str, Any] = {
        "schema_version": "1.4.0",  # Bumped for P5 UPGRADE-1 twin_lr_overrides
        "mode": "SHADOW",
        "phase": "P4",
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "cycles": args.cycles,
            "seed": args.seed,
            "slice": args.slice,
            "runner_type": args.runner_type,
            "tau_0": args.tau_0,
        },
        "twin_lr_overrides": twin_lr_overrides,  # P5 UPGRADE-1
        "telemetry_adapter": adapter_type,
        "telemetry_source": telemetry_source,  # P5: "mock" | "real_synthetic" | "real_trace"
        "adapter_config_path": adapter_config_path,
        "output_dir": str(output_dir),
    }

    # P5: Add identity pre-flight results if available
    # SHADOW MODE CONTRACT:
    # - identity_preflight is recorded for observational purposes only
    # - It does NOT affect harness execution or gating decisions
    if identity_preflight is not None:
        # Remove full_report from config (too verbose), keep summary fields
        preflight_summary = {k: v for k, v in identity_preflight.items() if k != "full_report"}
        config["identity_preflight"] = preflight_summary

    config_path = output_dir / "run_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def save_summary(output_dir: Path, result: Any, u2_dynamics_tile: Optional[Dict[str, Any]] = None) -> None:
    """Save P4 summary report."""
    summary = result.to_dict()
    if u2_dynamics_tile is not None:
        try:
            summary = attach_u2_dynamics_to_p4_summary(summary, u2_dynamics_tile)
        except Exception:
            # SHADOW MODE: Never fail on advisory U2 dynamics wiring
            pass
    summary_path = output_dir / "p4_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


def save_twin_accuracy(output_dir: Path, runner: Any) -> None:
    """Save twin accuracy metrics."""
    summary = runner._divergence_analyzer.get_summary()

    accuracy = {
        "schema_version": "1.0.0",
        "mode": "SHADOW",
        "total_comparisons": summary.total_comparisons,
        "accuracy_metrics": {
            "success_prediction": round(summary.success_accuracy, 4),
            "blocked_prediction": round(summary.blocked_accuracy, 4),
            "omega_prediction": round(summary.omega_accuracy, 4),
            "hard_ok_prediction": round(summary.hard_ok_accuracy, 4),
        },
        "divergence_metrics": {
            "total_divergences": summary.total_divergences,
            "divergence_rate": round(summary.divergence_rate, 4),
            "by_type": {
                "state": summary.state_divergences,
                "outcome": summary.outcome_divergences,
                "combined": summary.combined_divergences,
            },
            "by_severity": {
                "minor": summary.minor_divergences,
                "moderate": summary.moderate_divergences,
                "severe": summary.severe_divergences,
            },
        },
        "streak_metrics": {
            "max_divergence_streak": summary.max_divergence_streak,
            "current_streak": summary.current_streak,
        },
    }

    accuracy_path = output_dir / "twin_accuracy.json"
    with open(accuracy_path, "w") as f:
        json.dump(accuracy, f, indent=2)


def save_rtts_validation(
    output_dir: Path,
    real_cycles_path: Path,
    run_id: str,
    cycle_count: int,
) -> Optional[Path]:
    """
    Save RTTS validation results as rtts_validation.json.

    P5.2 VALIDATE STAGE (NO ENFORCEMENT):
    - Runs all RTTS validators on telemetry window
    - Emits schema-versioned validation block
    - All results are LOGGED_ONLY, no gating

    SHADOW MODE CONTRACT:
    - This artifact is for observational analysis only
    - It does NOT gate any operations
    - Mock detection flags are advisory
    - All fields are optional for downstream consumers

    Args:
        output_dir: Output directory for the artifact
        real_cycles_path: Path to real_cycles.jsonl
        run_id: Run identifier
        cycle_count: Number of cycles completed

    Returns:
        Path to created artifact, or None if emission failed
    """
    from backend.topology.first_light.data_structures_p4 import TelemetrySnapshot

    if not real_cycles_path.exists():
        return None

    # Load telemetry snapshots from real_cycles.jsonl
    snapshots: List[Any] = []
    try:
        with open(real_cycles_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    # Convert to TelemetrySnapshot
                    # Map real_cycles.jsonl fields to TelemetrySnapshot fields
                    snapshot = TelemetrySnapshot(
                        cycle=entry.get("cycle", len(snapshots)),
                        H=entry.get("H", 0.5),
                        rho=entry.get("rho", 0.5),
                        tau=entry.get("tau", 0.2),
                        beta=entry.get("beta", 0.5),
                        in_omega=entry.get("in_omega", entry.get("omega", 0.5) > 0.5),
                        success=entry.get("success", True),
                        real_blocked=entry.get("real_blocked", entry.get("blocked", False)),
                        hard_ok=entry.get("hard_ok", True),
                        timestamp=entry.get("timestamp", ""),
                    )
                    snapshots.append(snapshot)
    except (json.JSONDecodeError, OSError, KeyError) as e:
        # SHADOW MODE: Never fail on validation emission
        return None

    if not snapshots:
        return None

    # Run RTTS validation
    validation_block = rtts_validate_window(snapshots)

    # Build artifact structure with stable MOCK-NNN codes
    mock_flags = []
    if validation_block.mock_detection:
        md = validation_block.mock_detection
        if md.mock_001_var_H_low:
            mock_flags.append("MOCK-001")
        if md.mock_002_var_rho_low:
            mock_flags.append("MOCK-002")
        if md.mock_003_cor_low:
            mock_flags.append("MOCK-003")
        if md.mock_004_cor_high:
            mock_flags.append("MOCK-004")
        if md.mock_005_acf_low:
            mock_flags.append("MOCK-005")
        if md.mock_006_acf_high:
            mock_flags.append("MOCK-006")
        if md.mock_007_kurtosis_low:
            mock_flags.append("MOCK-007")
        if md.mock_008_kurtosis_high:
            mock_flags.append("MOCK-008")
        if md.mock_009_jump_H:
            mock_flags.append("MOCK-009")
        if md.mock_010_discrete_rho:
            mock_flags.append("MOCK-010")

    artifact: Dict[str, Any] = {
        "schema_version": RTTS_VALIDATION_SCHEMA_VERSION,
        "run_id": run_id,
        "timestamp": validation_block.timestamp,
        "block_id": validation_block.block_id,
        "mode": "SHADOW",
        "action": "LOGGED_ONLY",
        # Overall status
        "overall_status": validation_block.overall_status,
        "validation_passed": validation_block.validation_passed,
        # Window metadata
        "window": {
            "size": validation_block.window_size,
            "start_cycle": validation_block.window_start_cycle,
            "end_cycle": validation_block.window_end_cycle,
            "total_cycles": cycle_count,
        },
        # Warnings aggregated from all components
        "all_warnings": validation_block.all_warnings,
        "warning_count": validation_block.warning_count,
        # Standardized MOCK-NNN flags
        "mock_detection_flags": mock_flags,
        # Component results
        "statistical": validation_block.statistical.to_dict() if validation_block.statistical else None,
        "correlation": validation_block.correlation.to_dict() if validation_block.correlation else None,
        "continuity": validation_block.continuity.to_dict() if validation_block.continuity else None,
        "mock_detection": validation_block.mock_detection.to_dict() if validation_block.mock_detection else None,
    }

    # Write artifact
    artifact_path = output_dir / "rtts_validation.json"
    with open(artifact_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    return artifact_path


# =============================================================================
# Canonical Source Taxonomy + Reconciliation Mode + "Never Lie" Audit
# =============================================================================
#
# P5 DIAGNOSTIC CONTRACT:
# - P5 Diagnostic is a RECONCILIATION VIEW; NOT a metric authority.
# - It reports exactly which metrics were used and which were stub/none.
# - All lists are deterministically sorted (alphabetical by signal name).
# - All lists are capped to declared maximums.
#
# =============================================================================

# Frozen allowed source labels - NEVER emit other strings
CANONICAL_SOURCE_LABELS = frozenset({"real", "stub", "none"})

# Metric definitions version - references canonical calibration doc
METRIC_DEFINITIONS_VERSION = "docs/system_law/calibration/METRIC_DEFINITIONS.md@v1.0.0"

# =============================================================================
# Explicit Caps (NEVER LIE CONTRACT)
# =============================================================================
MAX_MISSING_ARTIFACTS = 6       # Cap on missing_required_artifacts list
MAX_PATHS_TRIED_PER_SIGNAL = 3  # Cap on expected_paths_tried per signal
MAX_REASON_CODES = 3            # Cap on reason_codes_top3

# =============================================================================
# Reason Codes (Machine-Readable)
# =============================================================================
REASON_CODE_MISSING_REQUIRED = "MISSING_REQUIRED"
REASON_CODE_MALFORMED_JSON = "MALFORMED_JSON"
REASON_CODE_MISSING_KEYS = "MISSING_KEYS"
REASON_CODE_UNKNOWN_SOURCE_COERCED = "UNKNOWN_SOURCE_COERCED"
REASON_CODE_FILE_NOT_FOUND = "FILE_NOT_FOUND"

# Mapping from human-readable reasons to machine-readable codes
REASON_TO_CODE = {
    "file not found": REASON_CODE_FILE_NOT_FOUND,
    "file not found (optional)": REASON_CODE_FILE_NOT_FOUND,
    "malformed JSON": REASON_CODE_MALFORMED_JSON,
    "missing required keys": REASON_CODE_MISSING_KEYS,
}


def _validate_source_label(source: str, signal_name: str) -> tuple:
    """
    Validate and canonicalize source label.

    Args:
        source: The source label to validate
        signal_name: Name of signal for advisory note

    Returns:
        Tuple of (canonical_source, advisory_note or None)
        Unknown sources are coerced to "stub" with advisory note.
    """
    if source in CANONICAL_SOURCE_LABELS:
        return source, None
    # Unknown source - coerce to "stub" with advisory
    advisory = f"[Advisory] Unknown source '{source}' for {signal_name} coerced to 'stub'"
    return "stub", advisory


def _build_signal_inputs(
    source_summary: Dict[str, str],
    missing_reasons: Dict[str, str],
    paths_tried: Optional[Dict[str, List[str]]] = None,
    selected_paths: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Build signal_inputs block with canonical validation and explainability.

    Args:
        source_summary: Dict of signal_name -> source label
        missing_reasons: Dict of signal_name -> reason for stub/none
        paths_tried: Dict of signal_name -> list of paths tried (max 3 per signal)
        selected_paths: Dict of signal_name -> selected path when real

    Returns:
        Complete signal_inputs dict with source_summary, counts, missing_required_artifacts,
        and diagnostic_integrity block.
    """
    paths_tried = paths_tried or {}
    selected_paths = selected_paths or {}

    validated_summary: Dict[str, str] = {}
    advisories: List[str] = []

    # Validate all source labels
    for signal_name, source in source_summary.items():
        canonical, advisory = _validate_source_label(source, signal_name)
        validated_summary[signal_name] = canonical
        if advisory:
            advisories.append(advisory)

    coerced_count = len(advisories)

    # Build missing_required_artifacts (capped to MAX_MISSING_ARTIFACTS, deterministic order)
    missing_artifacts: List[Dict[str, Any]] = []
    reason_code_counts: Dict[str, int] = {}

    # Deterministic ordering: alphabetical by signal name
    for signal_name in sorted(missing_reasons.keys()):
        source = validated_summary.get(signal_name, "none")
        if source in ("stub", "none"):
            reason = missing_reasons[signal_name]
            entry: Dict[str, Any] = {
                "signal": signal_name,
                "source": source,
                "reason": reason,
            }
            # Add expected_paths_tried (capped to MAX_PATHS_TRIED_PER_SIGNAL)
            if signal_name in paths_tried:
                entry["expected_paths_tried"] = paths_tried[signal_name][:MAX_PATHS_TRIED_PER_SIGNAL]
            missing_artifacts.append(entry)

            # Track reason codes for histogram
            reason_code = REASON_TO_CODE.get(reason, REASON_CODE_MISSING_REQUIRED)
            reason_code_counts[reason_code] = reason_code_counts.get(reason_code, 0) + 1
        elif source == "real" and signal_name in selected_paths:
            # For real signals, we track selected_path separately (not in missing)
            pass

    # Apply explicit cap (NEVER LIE CONTRACT)
    missing_artifacts = missing_artifacts[:MAX_MISSING_ARTIFACTS]

    # Track coerced sources in reason codes
    if coerced_count > 0:
        reason_code_counts[REASON_CODE_UNKNOWN_SOURCE_COERCED] = coerced_count

    # Add selected_path for real signals in missing_reasons (edge case: file found but tracked)
    # Actually, for real signals we add selected_path to a separate structure
    real_signal_paths: Dict[str, str] = {}
    for signal_name in sorted(selected_paths.keys()):
        if validated_summary.get(signal_name) == "real":
            real_signal_paths[signal_name] = selected_paths[signal_name]

    real_count = sum(1 for s in validated_summary.values() if s == "real")
    stub_count = sum(1 for s in validated_summary.values() if s == "stub")
    none_count = sum(1 for s in validated_summary.values() if s == "none")

    # Build reason_codes_top3: sorted by count desc, then code asc (deterministic)
    sorted_codes = sorted(
        reason_code_counts.items(),
        key=lambda x: (-x[1], x[0])  # desc count, asc code name
    )
    reason_codes_top3 = [code for code, _ in sorted_codes[:MAX_REASON_CODES]]

    result: Dict[str, Any] = {
        "metric_definitions_version": METRIC_DEFINITIONS_VERSION,
        "source_summary": validated_summary,
        "real_count": real_count,
        "stub_count": stub_count,
        "none_count": none_count,
        "missing_required_artifacts": missing_artifacts,
        "diagnostic_integrity": {
            "uses_only_canonical_sources": coerced_count == 0,
            "coerced_sources_count": coerced_count,
            "missing_required_count": len(missing_artifacts),
            "reason_codes_top3": reason_codes_top3,
        },
    }

    # Add selected_paths for real signals (deterministic order)
    if real_signal_paths:
        result["selected_paths"] = real_signal_paths

    # Add advisories if any unknown sources were coerced
    if advisories:
        result["source_coercion_advisories"] = advisories

    return result


def _load_replay_signal(output_dir: Path, result: Any) -> tuple:
    """Load replay signal from real artifact or build stub.

    Returns:
        Tuple of (signal_dict, source_label, reason_if_missing, paths_tried, selected_path)
    """
    replay_paths = [
        output_dir / "replay_safety_governance_signal.json",
        output_dir / "replay_governance_radar.json",
        output_dir / "p4_shadow" / "replay_safety_governance_signal.json",
    ]
    paths_tried = [str(p.relative_to(output_dir)) if p.is_relative_to(output_dir) else p.name for p in replay_paths[:3]]
    malformed_path = None
    for path in replay_paths:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                selected = str(path.relative_to(output_dir)) if path.is_relative_to(output_dir) else path.name
                return {
                    "status": data.get("status", data.get("governance_status", "OK")),
                    "governance_alignment": data.get("governance_alignment", data.get("alignment", "aligned")),
                    "conflict": data.get("conflict", False),
                    "reasons": data.get("reasons", []),
                    "source": "real",
                }, "real", None, paths_tried, selected
            except json.JSONDecodeError:
                malformed_path = path
            except OSError:
                pass
    div_rate = getattr(result, "divergence_rate", 0.0)
    replay_status = "BLOCK" if div_rate > 0.2 else ("WARN" if div_rate > 0.1 else "OK")
    reason = "malformed JSON" if malformed_path else "file not found"
    return {
        "status": replay_status,
        "governance_alignment": "aligned" if replay_status == "OK" else "tension",
        "conflict": replay_status == "BLOCK",
        "reasons": [f"[Safety] Divergence rate: {div_rate:.2%}"],
        "source": "stub",
    }, "stub", reason, paths_tried, None


def _load_topology_signal(output_dir: Path) -> tuple:
    """Load topology signal from real artifact or build stub.

    Returns:
        Tuple of (signal_dict, source_label, reason_if_missing, paths_tried, selected_path)
    """
    topology_paths = [
        output_dir / "p5_topology_auditor_report.json",
        output_dir / "p4_shadow" / "p5_topology_auditor_report.json",
        output_dir / "topology_bundle.json",
    ]
    paths_tried = [str(p.relative_to(output_dir)) if p.is_relative_to(output_dir) else p.name for p in topology_paths[:3]]
    malformed_path = None
    for path in topology_paths:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                mode = "STABLE"
                if "p5_summary" in data:
                    joint_status = data["p5_summary"].get("joint_status", "ALIGNED")
                    mode = {"ALIGNED": "STABLE", "TENSION": "DRIFT", "DIVERGENT": "TURBULENT"}.get(joint_status, "STABLE")
                elif "mode" in data:
                    mode = data.get("mode", "STABLE")
                selected = str(path.relative_to(output_dir)) if path.is_relative_to(output_dir) else path.name
                return {
                    "mode": mode,
                    "persistence_drift": data.get("persistence_drift", 0.0),
                    "within_omega": data.get("within_omega", True),
                    "source": "real",
                }, "real", None, paths_tried, selected
            except json.JSONDecodeError:
                malformed_path = path
            except OSError:
                pass
    reason = "malformed JSON" if malformed_path else "file not found"
    return {"mode": "STABLE", "persistence_drift": 0.02, "within_omega": True, "source": "stub"}, "stub", reason, paths_tried, None


def _load_budget_signal(output_dir: Path) -> tuple:
    """Load budget signal from real artifact or build stub.

    Returns:
        Tuple of (signal_dict, source_label, reason_if_missing, paths_tried, selected_path)
    """
    budget_paths = [
        output_dir / "budget_calibration_summary.json",
        output_dir / "budget_risk_signal.json",
        output_dir / "p4_shadow" / "budget_calibration_summary.json",
    ]
    paths_tried = [str(p.relative_to(output_dir)) if p.is_relative_to(output_dir) else p.name for p in budget_paths[:3]]
    malformed_path = None
    for path in budget_paths:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                stability_class = data.get("stability_class", data.get("budget_stability", "STABLE"))
                if hasattr(stability_class, "value"):
                    stability_class = stability_class.value
                selected = str(path.relative_to(output_dir)) if path.is_relative_to(output_dir) else path.name
                return {
                    "stability_class": stability_class,
                    "health_score": data.get("health_score", 90),
                    "stability_index": data.get("stability_index", 0.95),
                    "source": "real",
                }, "real", None, paths_tried, selected
            except json.JSONDecodeError:
                malformed_path = path
            except OSError:
                pass
    reason = "malformed JSON" if malformed_path else "file not found"
    return {"stability_class": "STABLE", "health_score": 90, "stability_index": 0.95, "source": "stub"}, "stub", reason, paths_tried, None


def _load_identity_signal(output_dir: Path) -> tuple:
    """Load identity signal from real artifact or return None.

    Returns:
        Tuple of (signal_dict or None, source_label, reason_if_missing, paths_tried, selected_path)
    """
    identity_paths = [
        output_dir / "identity_check.json",
        output_dir / "p4_shadow" / "identity_check.json",
        output_dir / "run_config.json",
    ]
    paths_tried = [str(p.relative_to(output_dir)) if p.is_relative_to(output_dir) else p.name for p in identity_paths[:3]]
    malformed_path = None
    missing_keys_path = None
    for path in identity_paths:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "identity_preflight" in data:
                    preflight = data["identity_preflight"]
                    passed = preflight.get("status") == "PASSED"
                    selected = str(path.relative_to(output_dir)) if path.is_relative_to(output_dir) else path.name
                    return {
                        "block_hash_valid": passed, "merkle_root_valid": passed,
                        "signature_valid": passed, "chain_continuous": passed,
                        "pq_attestation_valid": passed, "dual_root_consistent": passed,
                        "source": "real",
                    }, "real", None, paths_tried, selected
                if "block_hash_valid" in data:
                    data["source"] = "real"
                    selected = str(path.relative_to(output_dir)) if path.is_relative_to(output_dir) else path.name
                    return data, "real", None, paths_tried, selected
                # File exists but missing required keys
                missing_keys_path = path
            except json.JSONDecodeError:
                malformed_path = path
            except OSError:
                pass
    # Determine reason
    if malformed_path:
        reason = "malformed JSON"
    elif missing_keys_path:
        reason = "missing required keys"
    else:
        reason = "file not found (optional)"
    return None, "none", reason, paths_tried, None


def _load_structure_signal(output_dir: Path) -> tuple:
    """Load structure signal from real artifact or return None.

    Returns:
        Tuple of (signal_dict or None, source_label, reason_if_missing, paths_tried, selected_path)
    """
    structure_paths = [
        output_dir / "dag_coherence_check.json",
        output_dir / "p4_shadow" / "dag_coherence_check.json",
        output_dir / "structure_check.json",
    ]
    paths_tried = [str(p.relative_to(output_dir)) if p.is_relative_to(output_dir) else p.name for p in structure_paths[:3]]
    malformed_path = None
    missing_keys_path = None
    for path in structure_paths:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "dag_coherent" in data:
                    data["source"] = "real"
                    selected = str(path.relative_to(output_dir)) if path.is_relative_to(output_dir) else path.name
                    return data, "real", None, paths_tried, selected
                # File exists but missing required keys
                missing_keys_path = path
            except json.JSONDecodeError:
                malformed_path = path
            except OSError:
                pass
    # Determine reason
    if malformed_path:
        reason = "malformed JSON"
    elif missing_keys_path:
        reason = "missing required keys"
    else:
        reason = "file not found (optional)"
    return None, "none", reason, paths_tried, None


def save_p5_diagnostic(
    output_dir: Path,
    divergence_log_path: Path,
    result: Any,
    run_id: str,
    cycle_count: int,
) -> None:
    """
    Save P5 divergence diagnostic at end of run.

    SHADOW MODE CONTRACT:
    - This is observational output only
    - Does not influence any control flow
    - Loads real signals when available, falls back to stubs when missing
    - Tracks source of each signal input (real vs stub)
    - Source labels are canonicalized to {"real", "stub", "none"}
    - Logs advisory line summarizing source counts

    Args:
        output_dir: Output directory for artifacts
        divergence_log_path: Path to divergence_log.jsonl
        result: P4 runner result object
        run_id: Run identifier
        cycle_count: Number of cycles completed
    """
    from backend.health.p5_divergence_interpreter import interpret_p5_divergence

    source_summary: Dict[str, str] = {}
    missing_reasons: Dict[str, str] = {}
    paths_tried: Dict[str, List[str]] = {}
    selected_paths: Dict[str, str] = {}

    # Read last divergence snapshot for P4 state
    divergence_snapshot = {"severity": "NONE", "type": "NONE", "divergence_pct": 0.0}
    divergence_paths = ["divergence_log.jsonl"]
    paths_tried["divergence"] = divergence_paths
    if divergence_log_path.exists():
        try:
            with open(divergence_log_path, "r") as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1].strip()
                    if last_line:
                        divergence_snapshot = json.loads(last_line)
            source_summary["divergence"] = "real"
            selected_paths["divergence"] = "divergence_log.jsonl"
        except (json.JSONDecodeError, OSError):
            source_summary["divergence"] = "stub"
            missing_reasons["divergence"] = "malformed JSON"
    else:
        source_summary["divergence"] = "stub"
        missing_reasons["divergence"] = "file not found"

    # Load signals - preferring real artifacts over stubs
    replay_signal, replay_source, replay_reason, replay_paths, replay_selected = _load_replay_signal(output_dir, result)
    source_summary["replay"] = replay_source
    paths_tried["replay"] = replay_paths
    if replay_reason:
        missing_reasons["replay"] = replay_reason
    if replay_selected:
        selected_paths["replay"] = replay_selected

    topology_signal, topology_source, topology_reason, topology_paths, topology_selected = _load_topology_signal(output_dir)
    source_summary["topology"] = topology_source
    paths_tried["topology"] = topology_paths
    if topology_reason:
        missing_reasons["topology"] = topology_reason
    if topology_selected:
        selected_paths["topology"] = topology_selected

    budget_signal, budget_source, budget_reason, budget_paths, budget_selected = _load_budget_signal(output_dir)
    source_summary["budget"] = budget_source
    paths_tried["budget"] = budget_paths
    if budget_reason:
        missing_reasons["budget"] = budget_reason
    if budget_selected:
        selected_paths["budget"] = budget_selected

    identity_signal, identity_source, identity_reason, identity_paths, identity_selected = _load_identity_signal(output_dir)
    source_summary["identity"] = identity_source
    paths_tried["identity"] = identity_paths
    if identity_reason:
        missing_reasons["identity"] = identity_reason
    if identity_selected:
        selected_paths["identity"] = identity_selected

    structure_signal, structure_source, structure_reason, structure_paths, structure_selected = _load_structure_signal(output_dir)
    source_summary["structure"] = structure_source
    paths_tried["structure"] = structure_paths
    if structure_reason:
        missing_reasons["structure"] = structure_reason
    if structure_selected:
        selected_paths["structure"] = structure_selected

    # Generate P5 diagnostic
    diagnostic = interpret_p5_divergence(
        divergence_snapshot=divergence_snapshot,
        replay_signal=replay_signal,
        topology_signal=topology_signal,
        budget_signal=budget_signal,
        identity_signal=identity_signal,
        structure_signal=structure_signal,
        cycle=cycle_count,
        run_id=run_id,
    )

    # Add signal source summary with canonical validation, paths, and explainability
    diagnostic["signal_inputs"] = _build_signal_inputs(
        source_summary, missing_reasons, paths_tried, selected_paths
    )

    # Write diagnostic
    diagnostic_path = output_dir / "p5_divergence_diagnostic.json"
    with open(diagnostic_path, "w") as f:
        json.dump(diagnostic, f, indent=2)

    # Log advisory line summarizing source counts (SHADOW MODE - observational only)
    signal_inputs = diagnostic["signal_inputs"]
    real_count = signal_inputs["real_count"]
    stub_count = signal_inputs["stub_count"]
    none_count = signal_inputs["none_count"]
    print(f"[P5 Diagnostic] Source summary: real={real_count}, stub={stub_count}, none={none_count}")


def main() -> int:
    """Main entry point."""
    args = parse_args()
    return run_harness(args)


if __name__ == "__main__":
    sys.exit(main())
