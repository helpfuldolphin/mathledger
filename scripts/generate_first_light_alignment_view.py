#!/usr/bin/env python3
"""
First-Light Global Alignment View Generator

Generates the unified governance signal fusion view (first_light_alignment.json)
from per-layer signal artifacts in a First Light evidence pack.

This script:
1. Reads per-layer signals from existing JSON artifacts
2. Calls build_global_alignment_view() from GGFL
3. Emits first_light_alignment.json alongside first_light_status.json

SHADOW MODE CONTRACT:
- This script is purely observational
- It reads existing artifacts and produces alignment analysis
- It does not run harnesses, tests, or modify any governance decisions
- It does not execute governance enforcement

Signal Sources:
- topology: stability_report.json → USLA state metrics
- replay: (synthetic stub in P3; real in P4)
- metrics: metrics_windows.json → success/block rates
- budget: (synthetic stub; no budget enforcement in First Light)
- structure: (DAG coherence from TDA metrics)
- telemetry: p4_summary.json → twin accuracy
- identity: (synthetic stub; hash verification)
- narrative: run_config.json → curriculum context

Usage:
    python scripts/generate_first_light_alignment_view.py \
        --p3-dir results/first_light/golden_run/p3 \
        --p4-dir results/first_light/golden_run/p4 \
        --evidence-pack-dir results/first_light/evidence_pack_first_light

Output:
    first_light_alignment.json in the evidence pack directory
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.governance.fusion import build_global_alignment_view


def extract_p5_replay_signal_for_ggfl(
    replay_logs_path: Path,
    expected_hashes: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Extract P5 replay signal and transform to GGFL format.

    SHADOW MODE CONTRACT:
    - This function is purely observational
    - It extracts signal for GGFL alignment, not for gating
    - The result is advisory only

    ROBUSTNESS (v1.1.0):
    - Supports rotated JSONL segments (*.jsonl in directory)
    - Skips .jsonl.gz files with advisory warning (gzip not required)
    - Schema guard: preserves extraction on schema mismatch, surfaces schema_ok=false
    - Malformed lines are skipped with counter, pipeline completes
    - Deterministic ordering via sorted file/line processing

    Args:
        replay_logs_path: Path to P5 replay logs (JSONL file or directory).
        expected_hashes: Optional dict mapping cycle_id -> expected_hash.

    Returns:
        Dict in GGFL unified format or None if extraction fails.
    """
    try:
        from backend.health.replay_governance_adapter import (
            extract_p5_replay_safety_from_logs,
            replay_for_alignment_view_p5,
        )
    except ImportError:
        return None

    if not replay_logs_path.exists():
        return None

    # Robustness tracking
    advisory_warnings: List[str] = []
    skipped_gz_count = 0
    malformed_line_count = 0

    # Load replay logs with robustness
    replay_logs: List[Dict[str, Any]] = []

    if replay_logs_path.is_file():
        # Single JSONL file
        try:
            with open(replay_logs_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            replay_logs.append(json.loads(line))
                        except json.JSONDecodeError:
                            malformed_line_count += 1
                            advisory_warnings.append(
                                f"Malformed JSON at {replay_logs_path.name}:{line_num}"
                            )
        except OSError as e:
            advisory_warnings.append(f"Failed to read {replay_logs_path.name}: {e}")
            return None
    elif replay_logs_path.is_dir():
        # Directory with logs (supports rotation: *.json, *.jsonl)
        log_files: List[Path] = []
        for pattern in ["*.json", "*.jsonl"]:
            log_files.extend(replay_logs_path.glob(pattern))

        # Check for .gz files and skip with warning
        gz_files = list(replay_logs_path.glob("*.jsonl.gz"))
        if gz_files:
            skipped_gz_count = len(gz_files)
            advisory_warnings.append(
                f"Skipped {skipped_gz_count} .jsonl.gz file(s): gzip decompression not supported"
            )

        # Sort for deterministic ordering
        log_files = sorted(set(log_files), key=lambda p: p.name)

        for log_file in log_files:
            if log_file.suffix == ".jsonl":
                # JSONL file (rotated segment)
                try:
                    with open(log_file, "r", encoding="utf-8") as f:
                        for line_num, line in enumerate(f, 1):
                            line = line.strip()
                            if line:
                                try:
                                    replay_logs.append(json.loads(line))
                                except json.JSONDecodeError:
                                    malformed_line_count += 1
                                    advisory_warnings.append(
                                        f"Malformed JSON at {log_file.name}:{line_num}"
                                    )
                except OSError as e:
                    advisory_warnings.append(f"Failed to read {log_file.name}: {e}")
            else:
                # Single JSON file
                try:
                    with open(log_file, "r", encoding="utf-8") as f:
                        replay_logs.append(json.load(f))
                except json.JSONDecodeError:
                    malformed_line_count += 1
                    advisory_warnings.append(f"Malformed JSON in {log_file.name}")
                except OSError as e:
                    advisory_warnings.append(f"Failed to read {log_file.name}: {e}")
    else:
        return None

    if not replay_logs:
        # No valid logs - return warning-only GGFL signal if we have warnings
        if advisory_warnings:
            return {
                "status": "warn",
                "alignment": "unknown",
                "conflict": False,
                "top_reasons": sorted(advisory_warnings),
                "p5_grade": False,
                "determinism_band": None,
                "telemetry_source": None,
                "schema_ok": False,
                "advisory_only": True,
                "skipped_gz_count": skipped_gz_count,
                "malformed_line_count": malformed_line_count,
            }
        return None

    # Schema/version guard: check for required P5 fields
    schema_ok = True
    required_p5_fields = {"cycle_id", "trace_hash", "timestamp"}
    for log in replay_logs:
        if not isinstance(log, dict):
            schema_ok = False
            advisory_warnings.append("Log entry is not a dict")
            break
        missing = required_p5_fields - set(log.keys())
        if missing:
            schema_ok = False
            advisory_warnings.append(f"Missing P5 fields: {sorted(missing)}")
            break

    # Extract production_run_id from logs
    production_run_id = "unknown"
    if replay_logs and isinstance(replay_logs[0], dict):
        production_run_id = replay_logs[0].get(
            "run_id", replay_logs[0].get("production_run_id", "unknown")
        )

    # Extract P5 signal and transform to GGFL format
    try:
        signal = extract_p5_replay_safety_from_logs(
            replay_logs=replay_logs,
            production_run_id=production_run_id,
            expected_hashes=expected_hashes,
            telemetry_source="real",
        )
        ggfl_signal = replay_for_alignment_view_p5(signal)
        # Add robustness fields
        ggfl_signal["schema_ok"] = schema_ok
        ggfl_signal["advisory_only"] = True  # SHADOW MODE
        ggfl_signal["skipped_gz_count"] = skipped_gz_count
        ggfl_signal["malformed_line_count"] = malformed_line_count
        if advisory_warnings:
            # Merge advisory warnings into top_reasons
            existing_reasons = ggfl_signal.get("top_reasons", [])
            ggfl_signal["top_reasons"] = sorted(
                set(existing_reasons) | set(advisory_warnings)
            )
        return ggfl_signal
    except Exception as e:
        # Signal extraction failed - return partial GGFL signal with warnings
        advisory_warnings.append(f"Signal extraction failed: {e}")
        return {
            "status": "warn",
            "alignment": "unknown",
            "conflict": False,
            "top_reasons": sorted(advisory_warnings),
            "p5_grade": False,
            "determinism_band": None,
            "telemetry_source": None,
            "schema_ok": False,
            "advisory_only": True,
            "skipped_gz_count": skipped_gz_count,
            "malformed_line_count": malformed_line_count,
        }


def load_json_safe(file_path: Path) -> Optional[Dict[str, Any]]:
    """Load JSON file, returning None on error."""
    try:
        with open(file_path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"  WARNING: Could not load {file_path}: {e}")
        return None


def find_run_dir(base_dir: Path, prefix: str) -> Optional[Path]:
    """Find the most recent run directory with given prefix."""
    if not base_dir.exists():
        return None
    dirs = list(base_dir.glob(f"{prefix}*"))
    if not dirs:
        return None
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs[0]


def extract_topology_signal(
    stability_report: Optional[Dict[str, Any]],
    tda_metrics: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Extract topology signal from P3 stability report and TDA metrics.

    Maps:
    - H: success_rate (proxy for HSS)
    - rho: rsi.mean
    - within_omega: omega.occupancy_rate >= 0.90
    - C: inferred from metrics
    """
    signal: Dict[str, Any] = {}

    if stability_report:
        metrics = stability_report.get("metrics", {})

        # HSS proxy from success rate
        signal["H"] = metrics.get("success_rate", 0.0)

        # RSI from rsi.mean
        rsi = metrics.get("rsi", {})
        signal["rho"] = rsi.get("mean", 0.0)

        # Tau from config or default
        config = stability_report.get("config", {})
        signal["tau"] = config.get("tau_0", 0.20)

        # Omega occupancy check
        omega = metrics.get("omega", {})
        omega_rate = omega.get("occupancy_rate", 0.0)
        signal["within_omega"] = omega_rate >= 0.90

        # Block rate
        hard_mode = metrics.get("hard_mode", {})
        signal["beta"] = 1.0 - hard_mode.get("ok_rate", 1.0)

        # Depth and branch factor (defaults)
        signal["D"] = 5
        signal["D_dot"] = 0.0
        signal["B"] = 2.0

        # Shear (default)
        signal["S"] = 0.1

        # Convergence class (infer from RSI stability)
        rsi_std = rsi.get("std", 0.0)
        if rsi_std < 0.05:
            signal["C"] = 0  # CONVERGING
        elif rsi_std < 0.15:
            signal["C"] = 1  # OSCILLATING
        else:
            signal["C"] = 2  # DIVERGING

        # Jacobian (default)
        signal["J"] = 2.5

        # CDI and invariant status
        red_flags = stability_report.get("red_flag_summary", {})
        signal["active_cdis"] = []
        signal["invariant_violations"] = []

        if red_flags.get("hypothetical_abort", False):
            signal["active_cdis"] = ["CDI-007"]  # Exception exhaustion

    # Enhance with TDA metrics if available
    if tda_metrics:
        tda_summary = tda_metrics.get("summary", {})
        if "hss" in tda_summary:
            signal["H"] = tda_summary["hss"]

    return signal


def extract_replay_signal(
    p4_summary: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Extract replay signal from P4 summary.

    In P3 synthetic mode, this is a stub with nominal values.
    In P4 shadow mode, uses twin accuracy as proxy.
    """
    signal: Dict[str, Any] = {
        "replay_verified": True,
        "replay_divergence": 0.0,
        "replay_latency_ms": 50,
        "replay_hash_match": True,
        "replay_depth_valid": True,
    }

    if p4_summary:
        twin_acc = p4_summary.get("twin_accuracy", {})
        divergence = p4_summary.get("divergence_analysis", {})

        # Use twin accuracy as proxy for replay verification
        success_acc = twin_acc.get("success_prediction_accuracy", 1.0)
        signal["replay_verified"] = success_acc >= 0.70

        # Divergence rate as replay divergence proxy
        div_rate = divergence.get("divergence_rate", 0.0)
        signal["replay_divergence"] = div_rate

        # Hash match based on twin accuracy
        signal["replay_hash_match"] = success_acc >= 0.50

    return signal


def extract_metrics_signal(
    stability_report: Optional[Dict[str, Any]],
    metrics_windows: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Extract metrics signal from stability report and metrics windows.
    """
    signal: Dict[str, Any] = {
        "success_rate": 0.0,
        "abstention_rate": 0.0,
        "block_rate": 0.0,
        "throughput": 1.0,
        "latency_p50_ms": 100,
        "latency_p99_ms": 500,
        "queue_depth": 0,
    }

    if stability_report:
        metrics = stability_report.get("metrics", {})
        signal["success_rate"] = metrics.get("success_rate", 0.0)

        delta_p = metrics.get("delta_p", {})
        signal["abstention_rate"] = delta_p.get("abstention_final", 0.0)

        hard_mode = metrics.get("hard_mode", {})
        signal["block_rate"] = 1.0 - hard_mode.get("ok_rate", 1.0)

    if metrics_windows:
        # Use latest window metrics if available
        windows = metrics_windows.get("windows", [])
        if windows:
            latest = windows[-1]
            if "success_metrics" in latest:
                signal["success_rate"] = latest["success_metrics"].get(
                    "cumulative_success_rate", signal["success_rate"]
                )

    return signal


def extract_budget_signal() -> Dict[str, Any]:
    """
    Extract budget signal (synthetic stub for First Light).

    First Light experiments do not enforce budget constraints.
    """
    return {
        "compute_budget_remaining": 1.0,
        "memory_utilization": 0.3,
        "storage_headroom_gb": 100.0,
        "verification_quota_remaining": 10000,
        "budget_exhaustion_eta_cycles": 10000,
    }


def extract_structure_signal(
    tda_metrics: Optional[Dict[str, Any]],
    stability_report: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Extract structure signal from TDA metrics.

    DAG coherence is assumed in First Light synthetic mode.
    """
    signal: Dict[str, Any] = {
        "dag_coherent": True,
        "orphan_count": 0,
        "max_fanout": 10,
        "depth_distribution": {},
        "cycle_detected": False,
        "min_cut_capacity": 0.5,
    }

    if tda_metrics:
        summary = tda_metrics.get("summary", {})
        # Use connectivity metrics if available
        if "min_cut" in summary:
            signal["min_cut_capacity"] = summary["min_cut"]

    if stability_report:
        # Check for structural anomalies via red flags
        red_flags = stability_report.get("red_flag_summary", {})
        by_type = red_flags.get("by_severity", {})
        if by_type.get("CRITICAL", 0) > 0:
            signal["dag_coherent"] = False

    return signal


def extract_telemetry_signal(
    p4_summary: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Extract telemetry signal from P4 summary.

    In First Light, telemetry is always healthy (synthetic).
    """
    signal: Dict[str, Any] = {
        "lean_healthy": True,
        "db_healthy": True,
        "redis_healthy": True,
        "worker_count": 4,
        "error_rate": 0.0,
        "last_error": None,
        "uptime_seconds": 86400,
    }

    if p4_summary:
        # Mode check
        mode = p4_summary.get("mode", "SHADOW")
        if mode != "SHADOW":
            signal["error_rate"] = 0.1

    return signal


def extract_identity_signal(
    manifest: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Extract identity signal from evidence pack manifest.

    Hash verification is based on manifest integrity.
    """
    signal: Dict[str, Any] = {
        "block_hash_valid": True,
        "merkle_root_valid": True,
        "signature_valid": True,
        "chain_continuous": True,
        "pq_attestation_valid": True,
        "dual_root_consistent": True,
    }

    if manifest:
        # Check SHADOW mode compliance
        compliance = manifest.get("shadow_mode_compliance", {})
        signal["chain_continuous"] = compliance.get("no_governance_modification", True)
        signal["dual_root_consistent"] = compliance.get("all_divergence_logged_only", True)

    return signal


def extract_narrative_signal(
    run_config: Optional[Dict[str, Any]],
    stability_report: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Extract narrative signal from run config.
    """
    signal: Dict[str, Any] = {
        "current_slice": "unknown",
        "slice_progress": 0.0,
        "epoch": 0,
        "curriculum_health": "HEALTHY",
        "drift_detected": False,
        "narrative_coherence": 1.0,
    }

    if run_config:
        signal["current_slice"] = run_config.get("slice_name", "unknown")

    if stability_report:
        config = stability_report.get("config", {})
        signal["current_slice"] = config.get("slice_name", signal["current_slice"])

        # Progress based on cycles completed
        timing = stability_report.get("timing", {})
        completed = timing.get("cycles_completed", 0)
        total = config.get("total_cycles", 1000)
        signal["slice_progress"] = completed / max(total, 1)

        # Check for criteria pass as curriculum health
        criteria = stability_report.get("criteria_evaluation", {})
        if not criteria.get("all_passed", True):
            signal["curriculum_health"] = "DEGRADED"

        # Detect drift from delta-p
        metrics = stability_report.get("metrics", {})
        delta_p = metrics.get("delta_p", {})
        slope = delta_p.get("success_slope")
        if slope is not None and slope < -0.01:
            signal["drift_detected"] = True

    return signal


def generate_alignment_view(
    p3_dir: Path,
    p4_dir: Path,
    evidence_pack_dir: Path,
    cycle: int = 0,
    p5_replay_logs_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Generate the global alignment view from First Light artifacts.

    Args:
        p3_dir: Directory containing P3 artifacts
        p4_dir: Directory containing P4 artifacts
        evidence_pack_dir: Evidence pack directory
        cycle: Cycle number for alignment view
        p5_replay_logs_path: Optional path to P5 replay logs for GGFL integration

    Returns:
        Global alignment view dict
    """
    print("Loading artifacts...")

    # Find run directories
    p3_run_dir = find_run_dir(p3_dir, "fl_")
    p4_run_dir = find_run_dir(p4_dir, "p4_")

    # Load P3 artifacts
    stability_report = None
    tda_metrics = None
    metrics_windows = None
    run_config = None

    if p3_run_dir:
        print(f"  P3 run: {p3_run_dir.name}")
        stability_report = load_json_safe(p3_run_dir / "stability_report.json")
        tda_metrics = load_json_safe(p3_run_dir / "tda_metrics.json")
        metrics_windows = load_json_safe(p3_run_dir / "metrics_windows.json")
        run_config = load_json_safe(p3_run_dir / "run_config.json")
    else:
        print("  WARNING: No P3 run directory found")

    # Load P4 artifacts
    p4_summary = None
    if p4_run_dir:
        print(f"  P4 run: {p4_run_dir.name}")
        p4_summary = load_json_safe(p4_run_dir / "p4_summary.json")
    else:
        print("  WARNING: No P4 run directory found")

    # Load evidence pack manifest
    manifest = load_json_safe(evidence_pack_dir / "manifest.json")

    print()
    print("Extracting per-layer signals...")

    # Extract all signals
    topology = extract_topology_signal(stability_report, tda_metrics)
    print(f"  topology: H={topology.get('H', 0):.3f}, rho={topology.get('rho', 0):.3f}, within_omega={topology.get('within_omega')}")

    replay = extract_replay_signal(p4_summary)
    print(f"  replay: verified={replay.get('replay_verified')}, divergence={replay.get('replay_divergence', 0):.3f}")

    # Extract P5 replay signal for GGFL if provided (SHADOW MODE - advisory only)
    p5_replay_ggfl: Optional[Dict[str, Any]] = None
    if p5_replay_logs_path:
        p5_replay_ggfl = extract_p5_replay_signal_for_ggfl(p5_replay_logs_path)
        if p5_replay_ggfl:
            print(f"  replay_p5: status={p5_replay_ggfl.get('status')}, alignment={p5_replay_ggfl.get('alignment')}, p5_grade={p5_replay_ggfl.get('p5_grade')}")

    metrics = extract_metrics_signal(stability_report, metrics_windows)
    print(f"  metrics: success_rate={metrics.get('success_rate', 0):.3f}, block_rate={metrics.get('block_rate', 0):.3f}")

    budget = extract_budget_signal()
    print(f"  budget: remaining={budget.get('compute_budget_remaining', 0):.1%}")

    structure = extract_structure_signal(tda_metrics, stability_report)
    print(f"  structure: dag_coherent={structure.get('dag_coherent')}, min_cut={structure.get('min_cut_capacity', 0):.2f}")

    telemetry = extract_telemetry_signal(p4_summary)
    print(f"  telemetry: lean_healthy={telemetry.get('lean_healthy')}, error_rate={telemetry.get('error_rate', 0):.3f}")

    identity = extract_identity_signal(manifest)
    print(f"  identity: chain_continuous={identity.get('chain_continuous')}, dual_root_consistent={identity.get('dual_root_consistent')}")

    narrative = extract_narrative_signal(run_config, stability_report)
    print(f"  narrative: slice={narrative.get('current_slice')}, health={narrative.get('curriculum_health')}")

    print()
    print("Building global alignment view...")

    # Build alignment view using GGFL
    alignment_view = build_global_alignment_view(
        topology=topology,
        replay=replay,
        metrics=metrics,
        budget=budget,
        structure=structure,
        telemetry=telemetry,
        identity=identity,
        narrative=narrative,
        cycle=cycle,
    )

    # Add First Light specific metadata
    alignment_view["first_light"] = {
        "p3_run_id": p3_run_dir.name if p3_run_dir else None,
        "p4_run_id": p4_run_dir.name if p4_run_dir else None,
        "evidence_pack": str(evidence_pack_dir),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    # Add P5 replay GGFL signal if present (SHADOW MODE - advisory only)
    # Reference: docs/system_law/Replay_Safety_P5_Engineering_Plan.md
    if p5_replay_ggfl:
        alignment_view["signals"] = alignment_view.get("signals", {})
        alignment_view["signals"]["replay_p5"] = p5_replay_ggfl
        # Mark mode as SHADOW
        alignment_view["signals"]["replay_p5"]["mode"] = "SHADOW"
        alignment_view["signals"]["replay_p5"]["shadow_mode_contract"] = {
            "observational_only": True,
            "no_control_flow_influence": True,
            "no_governance_modification": True,
        }

    return alignment_view


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate First-Light Global Alignment View"
    )
    parser.add_argument(
        "--p3-dir",
        type=str,
        default="results/first_light/golden_run/p3",
        help="Path to P3 run directory (containing fl_* subdirectory)",
    )
    parser.add_argument(
        "--p4-dir",
        type=str,
        default="results/first_light/golden_run/p4",
        help="Path to P4 run directory (containing p4_* subdirectory)",
    )
    parser.add_argument(
        "--evidence-pack-dir",
        type=str,
        default="results/first_light/evidence_pack_first_light",
        help="Path to evidence pack directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for alignment JSON (default: <evidence-pack-dir>/first_light_alignment.json)",
    )
    parser.add_argument(
        "--cycle",
        type=int,
        default=0,
        help="Cycle number for alignment view (default: 0)",
    )
    parser.add_argument(
        "--p5-replay-logs",
        type=str,
        help="Optional path to P5 replay logs (JSONL file or directory) for GGFL integration (SHADOW MODE).",
    )

    args = parser.parse_args()

    p3_dir = Path(args.p3_dir)
    p4_dir = Path(args.p4_dir)
    evidence_pack_dir = Path(args.evidence_pack_dir)

    # Parse optional P5 replay logs (SHADOW MODE - advisory only)
    p5_replay_logs_path = None
    if args.p5_replay_logs:
        p5_replay_logs_path = Path(args.p5_replay_logs)
        if not p5_replay_logs_path.exists():
            print(f"WARNING: P5 replay logs not found: {p5_replay_logs_path}")
            p5_replay_logs_path = None

    # Generate alignment view
    print("=" * 60)
    print("First-Light Global Alignment View Generator")
    print("=" * 60)
    print()
    print(f"P3 Directory: {p3_dir}")
    print(f"P4 Directory: {p4_dir}")
    print(f"Evidence Pack: {evidence_pack_dir}")
    if p5_replay_logs_path:
        print(f"P5 Replay Logs: {p5_replay_logs_path}")
    print()

    alignment_view = generate_alignment_view(
        p3_dir=p3_dir,
        p4_dir=p4_dir,
        evidence_pack_dir=evidence_pack_dir,
        cycle=args.cycle,
        p5_replay_logs_path=p5_replay_logs_path,
    )

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = evidence_pack_dir / "first_light_alignment.json"

    # Write alignment view
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(alignment_view, f, indent=2)

    # Print summary
    print()
    print("=" * 60)
    print("Alignment View Summary")
    print("=" * 60)
    print()

    fusion = alignment_view.get("fusion_result", {})
    escalation = alignment_view.get("escalation", {})
    conflicts = alignment_view.get("conflict_detections", [])

    print(f"  Decision: {fusion.get('decision')}")
    print(f"  Is Hard: {fusion.get('is_hard')}")
    print(f"  Primary Reason: {fusion.get('primary_reason')}")
    print()
    print(f"  Escalation Level: {escalation.get('level_name')}")
    print(f"  Trigger: {escalation.get('trigger_reason')}")
    print()
    print(f"  Conflicts Detected: {len(conflicts)}")
    for conflict in conflicts:
        print(f"    - {conflict.get('rule_id')}: {conflict.get('description')}")
    print()
    print(f"  Headline: {alignment_view.get('headline')}")
    print()
    print("=" * 60)
    print(f"Alignment view written to: {output_path}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
