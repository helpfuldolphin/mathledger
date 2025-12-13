#!/usr/bin/env python3
"""
First-Light Evidence Pack Builder

Builds the canonical evidence pack from P3/P4 golden run artifacts.
See Evidence_Pack_Spec_PhaseX.md for structure specification.

SHADOW MODE CONTRACT:
- This script only packages existing artifacts
- No governance decisions are made or modified
- All artifacts retain mode="SHADOW" markers

Usage:
    python scripts/build_first_light_evidence_pack.py \
        --p3-dir results/first_light/golden_run/p3 \
        --p4-dir results/first_light/golden_run/p4 \
        --output-dir results/first_light/evidence_pack_first_light

Runtime Profile Integration (Future):
    To include runtime profile governance signals in the evidence pack:
    
    1. Run chaos harness to generate summary:
       python experiments/u2_runtime_chaos.py \
           --profile prod-hardened \
           --env-context prod \
           --runs 100 \
           --seed 42 \
           --output artifacts/runtime_profile_health/chaos_summary.json
    
    2. Load chaos summary and build snapshot:
       from experiments.u2.runtime import build_runtime_profile_snapshot_for_first_light
       import json
       
       with open("artifacts/runtime_profile_health/chaos_summary.json") as f:
           chaos_summary = json.load(f)
       
       runtime_profile_snapshot = build_runtime_profile_snapshot_for_first_light(
           profile_name="prod-hardened",
           chaos_summary=chaos_summary,
       )
    
    3. Attach to evidence pack manifest under governance section:
       # In build_evidence_pack() function, after manifest creation:
       # if runtime_profile_snapshot is not None:
       #     if "governance" not in manifest:
       #         manifest["governance"] = {}
       #     manifest["governance"]["runtime_profile"] = runtime_profile_snapshot
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

# ====================================================================
# Evidence Pack Attachment Provenance Enums (PROVENANCE LAW v1)
# ====================================================================
# Canonical enum sets for provenance tracking when attaching artifacts.
EXTRACTION_SOURCE_VALUES = Literal["MANIFEST", "DIRECT_FLAG", "FALLBACK", "MISSING"]
MIRRORED_FROM_VALUES = Literal["GOVERNANCE_PATH", "LEGACY_PATH"]


def coerce_extraction_source(value: Any) -> str:
    """
    Coerce extraction_source to canonical enum value.

    Args:
        value: Any value (string, None, etc.)

    Returns:
        Canonical extraction_source value: "MANIFEST" | "DIRECT_FLAG" | "FALLBACK" | "MISSING"
    """
    if isinstance(value, str):
        value_upper = value.upper()
        if value_upper in ("MANIFEST", "DIRECT_FLAG", "FALLBACK", "MISSING"):
            return value_upper
    return "MISSING"


def coerce_mirrored_from(value: Any) -> Optional[str]:
    """
    Coerce mirrored_from to canonical enum value.
    
    Args:
        value: Any value (string, None, etc.)
    
    Returns:
        Canonical mirrored_from value: "GOVERNANCE_PATH" | "LEGACY_PATH" | None
        Returns None if value is invalid (should not be set)
    """
    if isinstance(value, str):
        value_upper = value.upper()
        if value_upper in ("GOVERNANCE_PATH", "LEGACY_PATH"):
            return value_upper
    # Unknown value → None (omit field)
    return None

from scripts.first_light_proof_hash_snapshot import generate_snapshot

from tools.evidence_plots import (
    plot_delta_p,
    plot_omega_occupancy,
    plot_rsi,
)
from backend.topology.first_light.calibration_annex import load_cal_exp1_annex


def manifest_path(file_path: Path, pack_root: Path) -> str:
    """Return a manifest-friendly POSIX path rooted at the evidence pack."""
    file_path = Path(file_path)
    pack_root = Path(pack_root)
    try:
        relative = file_path.relative_to(pack_root)
    except ValueError:
        relative = file_path
    return relative.as_posix()


PROOF_SNAPSHOT_ENV_FLAG = "FIRST_LIGHT_INCLUDE_PROOF_SNAPSHOT"
PROOF_LOG_ENV_VAR = "FIRST_LIGHT_PROOF_LOG"
DEFAULT_PROOF_SNAPSHOT_PATH = Path("compliance") / "proof_log_snapshot.json"
DEFAULT_PROOF_LOG_CANDIDATES = [
    Path("proofs.jsonl"),
    Path("proof_log.jsonl"),
    Path("p3_synthetic") / "proofs.jsonl",
    Path("p3_synthetic") / "proof_log.jsonl",
    Path("p4_shadow") / "proofs.jsonl",
    Path("p4_shadow") / "proof_log.jsonl",
]


def _truthy_env(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def find_run_dir(base_dir: Path, prefix: str) -> Path:
    """Find the run directory with given prefix."""
    dirs = list(base_dir.glob(f"{prefix}*"))
    if not dirs:
        raise FileNotFoundError(f"No {prefix}* directory found in {base_dir}")
    if len(dirs) > 1:
        # Take the most recent
        dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs[0]


def discover_cal_exp1_report(run_dir: Path) -> Optional[Path]:
    """
    Discover CAL-EXP-1 report path under a run directory.

    Search order:
    1) run_dir/calibration/cal_exp1_report.json
    2) run_dir/cal_exp1_report.json
    """
    candidates = [
        run_dir / "calibration" / "cal_exp1_report.json",
        run_dir / "cal_exp1_report.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def discover_cal_exp2_report(run_dir: Path) -> Optional[Path]:
    """
    Discover CAL-EXP-2 report path under a run directory.

    Search order:
    1) run_dir/calibration/cal_exp2_report.json
    2) run_dir/cal_exp2_report.json
    """
    candidates = [
        run_dir / "calibration" / "cal_exp2_report.json",
        run_dir / "cal_exp2_report.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def discover_cal_exp3_report(run_dir: Path) -> Optional[Path]:
    """
    Discover CAL-EXP-3 report path under a run directory.

    Search order:
    1) run_dir/calibration/cal_exp3_report.json
    2) run_dir/cal_exp3_report.json
    """
    candidates = [
        run_dir / "calibration" / "cal_exp3_report.json",
        run_dir / "cal_exp3_report.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def copy_artifacts(
    src_dir: Path,
    dst_dir: Path,
    artifacts: List[str],
    *,
    pack_root: Path,
) -> List[Dict[str, Any]]:
    """Copy artifacts and return manifest entries."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    entries = []

    for artifact in artifacts:
        src_path = src_dir / artifact
        if src_path.exists():
            dst_path = dst_dir / artifact
            shutil.copy2(src_path, dst_path)
            entries.append({
                "path": manifest_path(dst_path, pack_root),
                "sha256": compute_sha256(dst_path),
                "size_bytes": dst_path.stat().st_size,
            })
        else:
            print(f"  WARNING: Missing artifact {artifact}")

    return entries


def resolve_proof_log_path(
    explicit: Optional[Path],
    p3_run_dir: Path,
    p4_run_dir: Path,
) -> Tuple[Optional[Path], List[str]]:
    """Resolve proof log path by checking explicit, env, and default candidates."""
    search: List[Path] = []
    checked: List[str] = []
    seen: set[str] = set()

    def register(candidate: Path) -> None:
        key = str(candidate)
        if key not in seen:
            search.append(candidate)
            seen.add(key)

    if explicit:
        register(explicit)
        if not explicit.is_absolute():
            register(p3_run_dir / explicit)
            register(p4_run_dir / explicit)

    env_value = os.environ.get(PROOF_LOG_ENV_VAR)
    if env_value:
        env_path = Path(env_value)
        register(env_path)
        if not env_path.is_absolute():
            register(p3_run_dir / env_path)
            register(p4_run_dir / env_path)

    for relative in DEFAULT_PROOF_LOG_CANDIDATES:
        register(p3_run_dir / relative)
        register(p4_run_dir / relative)

    for candidate in search:
        checked.append(str(candidate))
        if candidate.exists():
            return candidate, checked

    return None, checked


def maybe_generate_proof_snapshot(
    *,
    include_snapshot: bool,
    proof_log_path: Optional[Path],
    p3_run_dir: Path,
    p4_run_dir: Path,
    compliance_dir: Path,
    pack_root: Path,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Generate proof_log_snapshot.json when requested."""
    if not include_snapshot:
        return None, None

    resolved, checked = resolve_proof_log_path(proof_log_path, p3_run_dir, p4_run_dir)
    if not resolved:
        print("  WARNING: Proof snapshot requested but proof log not found.")
        if checked:
            print("           Checked paths:")
            for entry in checked:
                print(f"             - {entry}")
        return None, None

    snapshot_path = compliance_dir / DEFAULT_PROOF_SNAPSHOT_PATH.name
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        snapshot_payload = generate_snapshot(str(resolved), str(snapshot_path))
    except Exception as exc:  # pragma: no cover - logging only
        print(f"  WARNING: Failed to generate proof snapshot: {exc}")
        return None, None

    file_hash = compute_sha256(snapshot_path)
    rel_path = manifest_path(snapshot_path, pack_root)

    manifest_entry = {
        "path": rel_path,
        "sha256": file_hash,
        "size_bytes": snapshot_path.stat().st_size,
    }
    manifest_meta = {
        "path": rel_path,
        "sha256": file_hash,
        "schema_version": snapshot_payload.get("schema_version"),
        "canonical_hash_algorithm": snapshot_payload.get("canonical_hash_algorithm"),
        "canonicalization_version": snapshot_payload.get("canonicalization_version"),
        "canonical_hash": snapshot_payload.get("canonical_hash"),
        "entry_count": snapshot_payload.get("entry_count"),
        "source": snapshot_payload.get("source"),
    }

    print(f"  Proof snapshot generated at {snapshot_path} (hash={file_hash})")
    return manifest_entry, manifest_meta


def generate_plot_artifacts(
    viz_dir: Path,
    p3_dir: Path,
    p4_dir: Path,
    pack_root: Path,
) -> List[Dict[str, Any]]:
    """Render SVG figures from the copied artifacts."""
    entries: List[Dict[str, Any]] = []
    viz_dir.mkdir(parents=True, exist_ok=True)
    print("Generating visualization artifacts...")
    tasks = [
        (
            "Delta p vs cycles",
            plot_delta_p,
            p4_dir / "divergence_log.jsonl",
            viz_dir / "delta_p_vs_cycles.svg",
        ),
        (
            "RSI vs cycles",
            plot_rsi,
            p3_dir / "metrics_windows.json",
            viz_dir / "rsi_vs_cycles.svg",
        ),
        (
            "Omega occupancy vs cycles",
            plot_omega_occupancy,
            p3_dir / "metrics_windows.json",
            viz_dir / "omega_occupancy_vs_cycles.svg",
        ),
    ]
    for label, func, source, destination in tasks:
        if not source.exists():
            print(f"  WARNING: {label} source missing at {source}")
            continue
        try:
            func(str(source), str(destination))
        except Exception as exc:
            print(f"  WARNING: Failed to generate {label}: {exc}")
            continue
        entries.append({
            "path": manifest_path(destination, pack_root),
            "sha256": compute_sha256(destination),
            "size_bytes": destination.stat().st_size,
        })
        print(f"  Generated {label} -> {destination}")
    return entries


def build_compliance_narrative(out_dir: Path) -> str:
    """Create compliance narrative stub."""
    narrative = """# First-Light Evidence Pack Compliance Narrative

## SHADOW MODE Declaration

This evidence pack was generated from Phase X P3/P4 First-Light shadow experiments.

**All artifacts in this pack are OBSERVATION-ONLY.**

- No governance decisions were modified during data collection
- No abort conditions were enforced
- All divergence records have `action="LOGGED_ONLY"`
- All cycle records have `mode="SHADOW"`

## Evidence Traceability

- P3 Synthetic artifacts: Generated by `usla_first_light_harness.py`
- P4 Shadow artifacts: Generated by `usla_first_light_p4_harness.py`
- Configuration: Frozen in `first_light_golden_run_config.json`

## Reference Documents

- Phase_X_P3_Spec.md: P3 First-Light specification
- Phase_X_P4_Spec.md: P4 Shadow coupling specification
- Phase_X_Prelaunch_Review.md: Pre-launch review criteria

## Compliance Statement

This run is SHADOW MODE only; no governance decisions were modified or enforced.

---
Generated: {timestamp}
"""
    return narrative.format(timestamp=datetime.now(timezone.utc).isoformat())


def build_evidence_pack(
    p3_dir: Path,
    p4_dir: Path,
    out_dir: Path,
    *,
    plots_dir: Path | None = None,
    include_proof_snapshot: bool = False,
    proof_log: Optional[Path] = None,
    cal_exp1_report: Optional[Path] = None,
    cal_exp2_report: Optional[Path] = None,
    cal_exp3_report: Optional[Path] = None,
    ledger_guard_summary: Optional[Path] = None,
) -> None:
    """
    Build First-Light evidence pack from P3/P4 golden run artifacts.

    Args:
        p3_dir: Directory containing P3 artifacts (with fl_* subdirectory)
        p4_dir: Directory containing P4 artifacts (with p4_* subdirectory)
        out_dir: Output directory for evidence pack
        plots_dir: Target directory for SVGs (relative to pack root if not absolute)
    """
    print("=" * 60)
    print("Building First-Light Evidence Pack")
    print("=" * 60)

    # Find actual run directories
    p3_run_dir = find_run_dir(p3_dir, "fl_")
    p4_run_dir = find_run_dir(p4_dir, "p4_")

    print(f"P3 source: {p3_run_dir}")
    print(f"P4 source: {p4_run_dir}")
    print(f"Output:    {out_dir}")
    print()

    # Create output structure
    out_dir.mkdir(parents=True, exist_ok=True)

    # P3 artifacts
    p3_artifacts = [
        "synthetic_raw.jsonl",
        "stability_report.json",
        "red_flag_matrix.json",
        "metrics_windows.json",
        "tda_metrics.json",
        "run_config.json",
    ]

    # P4 artifacts
    p4_artifacts = [
        "real_cycles.jsonl",
        "twin_predictions.jsonl",
        "divergence_log.jsonl",
        "p4_summary.json",
        "twin_accuracy.json",
        "run_config.json",
    ]

    manifest_entries = []
    cal_annex = None
    ledger_guard_summary_payload: Optional[Dict[str, Any]] = None
    ledger_guard_summary_entry: Optional[Dict[str, Any]] = None

    # Copy P3 artifacts
    print("Copying P3 synthetic artifacts...")
    p3_out = out_dir / "p3_synthetic"
    p3_entries = copy_artifacts(p3_run_dir, p3_out, p3_artifacts, pack_root=out_dir)
    manifest_entries.extend(p3_entries)
    print(f"  Copied {len(p3_entries)} P3 artifacts")

    # Copy P4 artifacts
    print("Copying P4 shadow artifacts...")
    p4_out = out_dir / "p4_shadow"
    p4_entries = copy_artifacts(p4_run_dir, p4_out, p4_artifacts, pack_root=out_dir)
    manifest_entries.extend(p4_entries)
    print(f"  Copied {len(p4_entries)} P4 artifacts")

    cal_exp_report_refs: Dict[str, Dict[str, Any]] = {}

    # Attach CAL-EXP-1 report if available
    cal_report_source: Optional[Path] = None
    cal_report_path: Optional[str] = None
    candidate_paths: List[Path] = []
    if cal_exp1_report is not None:
        candidate_paths.append(cal_exp1_report)
    for run_dir in (p4_run_dir, p3_run_dir):
        discovered = discover_cal_exp1_report(run_dir)
        if discovered is not None:
            candidate_paths.append(discovered)
    for candidate in candidate_paths:
        if candidate and candidate.exists():
            cal_report_source = candidate
            break

    if cal_report_source:
        cal_report_target = out_dir / "calibration" / "cal_exp1_report.json"
        cal_report_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cal_report_source, cal_report_target)
        cal_report_path = manifest_path(cal_report_target, out_dir)
        manifest_entries.append({
            "path": manifest_path(cal_report_target, out_dir),
            "sha256": compute_sha256(cal_report_target),
            "size_bytes": cal_report_target.stat().st_size,
        })
        cal_annex = load_cal_exp1_annex(cal_report_target)
        print(f"Attached CAL-EXP-1 report from {cal_report_source}")
    else:
        print("No CAL-EXP-1 report found; skipping calibration annex.")

    def attach_cal_exp_report(
        label: str,
        *,
        key: str,
        explicit: Optional[Path],
        target_name: str,
    ) -> None:
        target = out_dir / "calibration" / target_name
        legacy_target = out_dir / target_name

        source: Optional[Path] = None
        extraction_source_raw: Optional[str] = None

        if explicit is not None:
            if explicit.exists():
                source = explicit
                extraction_source_raw = "DIRECT_FLAG"
            else:
                print(f"  WARNING: Explicit {label} report not found at {explicit}; falling back.")

        if source is None:
            if target.exists():
                source = target
                extraction_source_raw = "MANIFEST"
            elif legacy_target.exists():
                source = legacy_target
                extraction_source_raw = "MANIFEST"

        if source is None:
            for run_dir in (p4_run_dir, p3_run_dir):
                for candidate in (
                    run_dir / "calibration" / target_name,
                    run_dir / target_name,
                ):
                    if candidate.exists():
                        source = candidate
                        extraction_source_raw = "FALLBACK"
                        break
                if source is not None:
                    break

        if source is None:
            print(f"No {label} report found; skipping.")
            return

        mirrored_from_raw = "GOVERNANCE_PATH" if source.parent.name == "calibration" else "LEGACY_PATH"
        extraction_source = coerce_extraction_source(extraction_source_raw)
        mirrored_from = coerce_mirrored_from(mirrored_from_raw) or "LEGACY_PATH"

        target.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)

        entry = {
            "path": manifest_path(target, out_dir),
            "sha256": compute_sha256(target),
            "size_bytes": target.stat().st_size,
        }
        manifest_entries.append(entry)

        schema_version = None
        mode = "SHADOW"
        try:
            report_payload = json.loads(target.read_text(encoding="utf-8"))
            if isinstance(report_payload, dict):
                schema_version = report_payload.get("schema_version")
                mode = report_payload.get("mode", mode)
        except (OSError, json.JSONDecodeError):
            pass

        cal_exp_report_refs[key] = {
            "schema_version": schema_version,
            "mode": mode,
            "path": entry["path"],
            "sha256": entry["sha256"],
            "size_bytes": entry["size_bytes"],
            "extraction_source": extraction_source,
            "mirrored_from": mirrored_from,
        }
        print(f"Attached {label} report from {source} (extraction_source={extraction_source}, mirrored_from={mirrored_from})")

    attach_cal_exp_report(
        "CAL-EXP-2",
        key="cal_exp2",
        explicit=cal_exp2_report,
        target_name="cal_exp2_report.json",
    )
    attach_cal_exp_report(
        "CAL-EXP-3",
        key="cal_exp3",
        explicit=cal_exp3_report,
        target_name="cal_exp3_report.json",
    )

    # Create visualizations placeholder
    print("Creating visualization placeholders...")
    viz_dir = plots_dir if plots_dir is not None else Path("visualizations")
    if not viz_dir.is_absolute():
        viz_dir = out_dir / viz_dir
    viz_dir.mkdir(parents=True, exist_ok=True)

    viz_placeholder = viz_dir / "README.md"
    viz_placeholder.write_text(
        "# Visualizations\n\n"
        "Standard plot outputs generated alongside the evidence pack.\n\n"
        "Included figures:\n"
        "- `delta_p_vs_cycles.svg`: Twin vs real objective delta\n"
        "- `rsi_vs_cycles.svg`: RSI trend across synthetic windows\n"
        "- `omega_occupancy_vs_cycles.svg`: Omega occupancy trend\n"
    )
    manifest_entries.append({
        "path": manifest_path(viz_placeholder, out_dir),
        "sha256": compute_sha256(viz_placeholder),
        "size_bytes": viz_placeholder.stat().st_size,
    })
    viz_dir_display = manifest_path(viz_dir, out_dir) or str(viz_dir)
    viz_dir_display = viz_dir_display.replace("\\", "/")
    if viz_dir_display and not viz_dir_display.endswith("/"):
        viz_dir_display = f"{viz_dir_display}/"
    viz_section_note = "(SVGs generated)"
    manifest_entries.extend(
        generate_plot_artifacts(
            viz_dir=viz_dir,
            p3_dir=p3_out,
            p4_dir=p4_out,
            pack_root=out_dir,
        )
    )

    # Create compliance directory
    print("Creating compliance narrative...")
    compliance_dir = out_dir / "compliance"
    compliance_dir.mkdir(exist_ok=True)

    narrative_path = compliance_dir / "compliance_narrative.md"
    narrative_path.write_text(build_compliance_narrative(out_dir))
    manifest_entries.append({
        "path": "compliance/compliance_narrative.md",
        "sha256": compute_sha256(narrative_path),
        "size_bytes": narrative_path.stat().st_size,
    })

    if ledger_guard_summary is None:
        for candidate in (
            p4_run_dir / "ledger_guard_summary.json",
            p4_run_dir / "governance" / "ledger_guard_summary.json",
            p4_run_dir / "compliance" / "ledger_guard_summary.json",
            p3_run_dir / "ledger_guard_summary.json",
            p3_run_dir / "governance" / "ledger_guard_summary.json",
            p3_run_dir / "compliance" / "ledger_guard_summary.json",
            p4_dir / "ledger_guard_summary.json",
            p4_dir / "governance" / "ledger_guard_summary.json",
            p4_dir / "compliance" / "ledger_guard_summary.json",
            p3_dir / "ledger_guard_summary.json",
            p3_dir / "governance" / "ledger_guard_summary.json",
            p3_dir / "compliance" / "ledger_guard_summary.json",
            out_dir / "governance" / "ledger_guard_summary.json",
            out_dir / "compliance" / "ledger_guard_summary.json",
        ):
            if candidate.exists():
                ledger_guard_summary = candidate
                print(f"  Auto-detected ledger guard summary: {candidate}")
                break

    if ledger_guard_summary is not None:
        summary_path = Path(ledger_guard_summary)
        if summary_path.exists():
            governance_dir = out_dir / "governance"
            governance_dir.mkdir(exist_ok=True)
            target_path = governance_dir / "ledger_guard_summary.json"
            if summary_path.resolve() != target_path.resolve():
                shutil.copy2(summary_path, target_path)
            try:
                ledger_guard_summary_payload = json.loads(
                    target_path.read_text(encoding="utf-8")
                )
                if (
                    isinstance(ledger_guard_summary_payload, dict)
                    and "schema_version" not in ledger_guard_summary_payload
                ):
                    ledger_guard_summary_payload["schema_version"] = "1.0.0"
                    target_path.write_text(
                        json.dumps(ledger_guard_summary_payload, indent=2),
                        encoding="utf-8",
                    )
            except json.JSONDecodeError:
                print("  WARNING: Failed to parse ledger guard summary; manifest entry only.")
                ledger_guard_summary_payload = None

            ledger_guard_summary_entry = {
                "path": manifest_path(target_path, out_dir),
                "sha256": compute_sha256(target_path),
                "size_bytes": target_path.stat().st_size,
            }
            manifest_entries.append(ledger_guard_summary_entry)
            print(f"  Attached ledger guard summary: {target_path}")
        else:
            print(f"  WARNING: Ledger guard summary not found at {summary_path}; skipping attachment.")

    include_snapshot_flag = include_proof_snapshot or _truthy_env(
        os.environ.get(PROOF_SNAPSHOT_ENV_FLAG)
    )
    snapshot_entry = None
    snapshot_manifest_meta = None
    if include_snapshot_flag:
        snapshot_entry, snapshot_manifest_meta = maybe_generate_proof_snapshot(
            include_snapshot=True,
            proof_log_path=proof_log,
            p3_run_dir=p3_run_dir,
            p4_run_dir=p4_run_dir,
            compliance_dir=compliance_dir,
            pack_root=out_dir,
        )
        if snapshot_entry:
            manifest_entries.append(snapshot_entry)

    # Load P3 and P4 summaries for manifest metadata
    with open(p3_run_dir / "stability_report.json") as f:
        p3_summary = json.load(f)

    with open(p4_run_dir / "p4_summary.json") as f:
        p4_summary = json.load(f)

    # Extract run IDs
    p3_run_id = p3_run_dir.name
    p4_run_id = p4_run_dir.name

    # Build manifest
    print("Building manifest...")
    manifest = {
        "schema_version": "1.0.0",
        "pack_type": "first_light_evidence",
        "mode": "SHADOW",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_ids": {
            "p3_synthetic": p3_run_id,
            "p4_shadow": p4_run_id,
        },
        "configuration": {
            "slice": "arithmetic_simple",
            "runner_type": "u2",
            "cycles": 1000,
            "seed": 42,
            "tau_0": 0.20,
            "window_size": 50,
        },
        "summary": {
            "p3": {
                "success_rate": p3_summary.get("metrics", {}).get("success_rate", 0),
                "mean_rsi": p3_summary.get("metrics", {}).get("rsi", {}).get("mean", 0),
                "omega_occupancy": p3_summary.get("metrics", {}).get("omega_occupancy", 0),
                "total_red_flags": p3_summary.get("red_flag_summary", {}).get("total", 0),
            },
            "p4": {
                "success_rate": p4_summary.get("uplift_metrics", {}).get("u2_success_rate_final", 0),
                "divergence_rate": p4_summary.get("divergence_analysis", {}).get("divergence_rate", 0),
                "twin_success_accuracy": p4_summary.get("twin_accuracy", {}).get("success_prediction_accuracy", 0),
                "max_divergence_streak": p4_summary.get("divergence_analysis", {}).get("max_divergence_streak", 0),
            },
        },
        "shadow_mode_compliance": {
            "all_divergence_logged_only": True,
            "no_governance_modification": True,
            "no_abort_enforcement": True,
        },
        # PHASE II — Snapshot Planner Runbook Summary
        # TODO(PHASE-II-SNAPSHOT): Integrate snapshot runbook summary when snapshot_root is available
        # Example integration:
        #   from experiments.u2.snapshot_history import (
        #       build_multi_run_snapshot_history,
        #       plan_future_runs,
        #       build_snapshot_runbook_summary,
        #   )
        #   if snapshot_root:
        #       multi_history = build_multi_run_snapshot_history([...run_dirs...])
        #       plan = plan_future_runs(multi_history)
        #       snapshot_runbook = build_snapshot_runbook_summary(multi_history, plan)
        #       manifest["operations"] = {
        #           "auto_resume": snapshot_runbook,
        #       }
        # ====================================================================
        # Runtime Profile Governance Integration (Future - Spec Only)
        # ====================================================================
        # To include runtime profile governance signals in evidence pack:
        #
        # 1. Add CLI arguments to main():
        #    parser.add_argument(
        #        "--runtime-profile-name",
        #        type=str,
        #        help="Runtime profile name (e.g., prod-hardened)",
        #    )
        #    parser.add_argument(
        #        "--chaos-summary-path",
        #        type=str,
        #        help="Path to chaos harness summary JSON from experiments/u2_runtime_chaos.py",
        #    )
        #
        # 2. In build_evidence_pack(), after manifest creation (before writing manifest.json):
        #    if runtime_profile_name and chaos_summary_path:
        #        try:
        #            from experiments.u2.runtime import (
        #                build_runtime_profile_snapshot_for_first_light,
        #            )
        #            import json
        #
        #            chaos_summary_file = Path(chaos_summary_path)
        #            if chaos_summary_file.exists():
        #                with open(chaos_summary_file, "r", encoding="utf-8") as f:
        #                    chaos_summary = json.load(f)
        #
        #                runtime_profile_snapshot = (
        #                    build_runtime_profile_snapshot_for_first_light(
        #                        profile_name=runtime_profile_name,
        #                        chaos_summary=chaos_summary,
        #                    )
        #                )
        #
        #                # Attach to manifest under governance section
        #                if "governance" not in manifest:
        #                    manifest["governance"] = {}
        #                manifest["governance"]["runtime_profile"] = runtime_profile_snapshot
        #
        #                print(f"  Attached runtime profile snapshot: {runtime_profile_name}")
        #        except Exception as e:
        #            # SHADOW MODE: Never fail build due to runtime profile issues
        #            print(f"  WARNING: Failed to attach runtime profile snapshot: {e}")
        #
        # Expected structure in manifest:
        # {
        #     "governance": {
        #         "runtime_profile": {
        #             "schema_version": "1.0.0",
        #             "profile": "prod-hardened",
        #             "status_light": "GREEN" | "YELLOW" | "RED",
        #             "profile_stability": 0.95,
        #             "no_run_rate": 0.05,
        #         }
        #     }
        # }
        #
        # Required inputs:
        #   - profile_name: Runtime profile name (e.g., "prod-hardened", "ci-strict", "dev-default")
        #   - chaos_summary_path: Path to JSON file from experiments/u2_runtime_chaos.py
        #
        "files": manifest_entries,
        "file_count": len(manifest_entries),
    }

    if ledger_guard_summary_entry is not None:
        manifest["governance"] = manifest.get("governance", {})
        manifest["governance"].setdefault("schema_versioned", {})
        violation_counts = None
        if ledger_guard_summary_payload is not None:
            violation_counts = ledger_guard_summary_payload.get("violation_counts")
            if violation_counts is None:
                violation_counts = ledger_guard_summary_payload.get("violation_count")

        manifest["governance"]["schema_versioned"]["ledger_guard_summary"] = {
            "schema_version": (
                ledger_guard_summary_payload.get("schema_version")
                if ledger_guard_summary_payload is not None
                else None
            ),
            "mode": "SHADOW",
            "path": ledger_guard_summary_entry["path"],
            "sha256": ledger_guard_summary_entry["sha256"],
            "size_bytes": ledger_guard_summary_entry["size_bytes"],
            "status_light": (
                ledger_guard_summary_payload.get("status_light")
                if ledger_guard_summary_payload is not None
                else None
            ),
            "violation_counts": violation_counts,
            "headline": (
                ledger_guard_summary_payload.get("headline")
                if ledger_guard_summary_payload is not None
                else None
            ),
        }

    if cal_exp_report_refs:
        manifest["governance"] = manifest.get("governance", {})
        manifest["governance"].setdefault("schema_versioned", {})
        manifest["governance"]["schema_versioned"]["cal_exp_reports"] = cal_exp_report_refs

    if cal_annex:
        manifest["governance"] = manifest.get("governance", {})
        manifest["governance"].setdefault("p5_calibration", {})
        manifest["governance"]["p5_calibration"]["cal_exp1"] = cal_annex
        if cal_report_path is not None:
            manifest["governance"]["p5_calibration"]["cal_exp1_report_path"] = cal_report_path

        # Attach structural calibration panel if present in evidence
        # This mirrors evidence["governance"]["structure"]["calibration_panel"] into manifest
        # The calibration panel is attached to evidence via attach_lean_shadow_to_evidence()
        # and should be mirrored in the manifest for status extraction
        # If evidence_pack.json exists, read calibration panel from it and attach to manifest
        evidence_pack_json_path = out_dir / "evidence_pack.json"
        if evidence_pack_json_path.exists():
            try:
                with open(evidence_pack_json_path, "r", encoding="utf-8") as f:
                    evidence = json.load(f)
                governance = evidence.get("governance", {})
                structure = governance.get("structure", {})
                calibration_panel = structure.get("calibration_panel")
                if calibration_panel:
                    manifest["governance"] = manifest.get("governance", {})
                    if "structure" not in manifest["governance"]:
                        manifest["governance"]["structure"] = {}
                    manifest["governance"]["structure"]["calibration_panel"] = calibration_panel
                    print("  Attached structural calibration panel to manifest")
            except (json.JSONDecodeError, OSError, KeyError):
                # Evidence pack not available or invalid - not an error (SHADOW MODE)
                pass

    # ====================================================================
    # PRNG Regime Timeseries Mirroring (SHADOW MODE)
    # ====================================================================
    # Mirror PRNG regime timeseries from CAL-EXP reports into manifest
    # Canonical location: manifest["governance"]["prng_regime_timeseries"]
    # This ensures the timeseries is available for status extraction
    # SHADOW MODE CONTRACT:
    # - Mirroring is purely observational (no gating)
    # - Prefer governance path from CAL-EXP report (canonical)
    # - Fallback to legacy path if governance path not present
    # - Mark mirrored_from if legacy path was used (PROVENANCE LAW v1)
    # - Include source_paths_checked for integrity invariant
    prng_timeseries_mirrored = False
    mirrored_from_raw: Optional[str] = None
    source_paths_checked: List[str] = []
    # Check CAL-EXP report paths (same as status generator)
    cal_exp_report_paths = [
        out_dir / "calibration" / "cal_exp1_report.json",
        out_dir / "calibration" / "cal_exp2_report.json",
        out_dir / "calibration" / "cal_exp3_report.json",
        out_dir / "cal_exp_report.json",
        out_dir / "cal_exp1_report.json",
        out_dir / "cal_exp2_report.json",
        out_dir / "cal_exp3_report.json",
    ]
    
    for report_path in cal_exp_report_paths:
        if report_path.exists():
            try:
                with open(report_path, "r", encoding="utf-8") as f:
                    cal_exp_report = json.load(f)
                    # Track which paths we checked (for integrity invariant)
                    try:
                        rel_path = report_path.relative_to(out_dir)
                        source_paths_checked.append(str(rel_path))
                    except ValueError:
                        # Absolute path or outside out_dir - use full path
                        source_paths_checked.append(str(report_path))
                    
                    # Prefer governance path (canonical location)
                    governance = cal_exp_report.get("governance", {})
                    prng_timeseries = governance.get("prng_regime_timeseries")
                    if prng_timeseries:
                        manifest["governance"] = manifest.get("governance", {})
                        # Create a copy and add integrity invariant fields
                        prng_timeseries_copy = prng_timeseries.copy()
                        prng_timeseries_copy["source_paths_checked"] = sorted(source_paths_checked)  # Deterministic ordering
                        manifest["governance"]["prng_regime_timeseries"] = prng_timeseries_copy
                        prng_timeseries_mirrored = True
                        mirrored_from_raw = "GOVERNANCE_PATH"
                        print("  Mirrored PRNG regime timeseries to manifest (governance path)")
                        break
                    # Fallback to legacy path
                    prng_timeseries = cal_exp_report.get("prng_regime_timeseries")
                    if prng_timeseries:
                        manifest["governance"] = manifest.get("governance", {})
                        # Create a copy and mark as mirrored from legacy
                        prng_timeseries_copy = prng_timeseries.copy()
                        mirrored_from_raw = "LEGACY_PATH"
                        # Coerce mirrored_from to canonical enum value
                        mirrored_from_coerced = coerce_mirrored_from(mirrored_from_raw)
                        if mirrored_from_coerced:
                            prng_timeseries_copy["mirrored_from"] = mirrored_from_coerced
                        prng_timeseries_copy["source_paths_checked"] = sorted(source_paths_checked)  # Deterministic ordering
                        manifest["governance"]["prng_regime_timeseries"] = prng_timeseries_copy
                        prng_timeseries_mirrored = True
                        print("  Mirrored PRNG regime timeseries to manifest (legacy path, marked)")
                        break
            except (json.JSONDecodeError, IOError):
                # Invalid JSON or read error - skip silently (advisory only)
                pass

        # Build runtime profile calibration annex if runtime profile snapshot is available
        # Note: runtime_profile_snapshot may be attached later via CLI args, so we check
        # both the current manifest and attempt to load from chaos summary if provided
        runtime_profile_snapshot = manifest.get("governance", {}).get("runtime_profile")
        if runtime_profile_snapshot and cal_report_source:
            try:
                from experiments.u2.runtime import build_runtime_profile_calibration_annex
                import copy

                # Deep copy manifest to avoid mutation
                manifest_copy = copy.deepcopy(manifest)

                # Load windows from the original cal_exp1_report.json file
                # (cal_annex is a compact summary that doesn't include windows)
                cal_report_data = json.loads(cal_report_source.read_text(encoding="utf-8"))
                cal_windows = cal_report_data.get("windows") or []

                if cal_windows:
                    calibration_annex = build_runtime_profile_calibration_annex(
                        runtime_profile_snapshot=runtime_profile_snapshot,
                        cal_exp1_windows=cal_windows,
                    )
                    if calibration_annex:
                        # Ensure governance section exists
                        if "governance" not in manifest_copy:
                            manifest_copy["governance"] = {}
                        manifest_copy["governance"]["runtime_profile_calibration"] = calibration_annex
                        # Update manifest reference (structured rebuild)
                        manifest["governance"] = manifest_copy["governance"]
                        print("  Attached runtime profile calibration annex")
            except Exception as e:
                # SHADOW MODE: Never fail build due to calibration annex issues
                # Attach error object instead
                if "governance" not in manifest:
                    manifest["governance"] = {}
                manifest["governance"]["runtime_profile_calibration"] = {
                    "error": str(e),
                    "mode": "SHADOW",
                    "schema_version": "1.0.0",
                }
                print(f"  WARNING: Failed to build runtime profile calibration annex: {e}")
                print("  Attached error object to manifest (SHADOW MODE)")

    # ====================================================================
    # RTTS Validation Reference (SHADOW MODE)
    # ====================================================================
    # If rtts_validation.json exists in P4 run directory, record reference
    # under manifest.governance.rtts_validation_reference with path+sha256.
    # SHADOW MODE CONTRACT:
    # - Reference is purely observational
    # - Status generator uses this reference for integrity-checked loading
    # - Missing file is not an error
    rtts_validation_path = None
    if p4_run_dir:
        rtts_candidates = [
            p4_run_dir / "rtts_validation.json",
            p4_run_dir / "p4_shadow" / "rtts_validation.json",
        ]
        for candidate in rtts_candidates:
            if candidate.exists():
                rtts_validation_path = candidate
                break

    if rtts_validation_path:
        try:
            rtts_sha256 = compute_sha256(rtts_validation_path)
            # Copy to governance directory
            governance_dir = out_dir / "governance"
            governance_dir.mkdir(exist_ok=True)
            rtts_dest = governance_dir / "rtts_validation.json"
            shutil.copy2(rtts_validation_path, rtts_dest)

            # Record reference in manifest
            manifest["governance"] = manifest.get("governance", {})
            manifest["governance"]["rtts_validation_reference"] = {
                "path": f"governance/rtts_validation.json",
                "sha256": rtts_sha256,
                "source_path": str(rtts_validation_path),
                "mode": "SHADOW",
                "action": "LOGGED_ONLY",
            }
            print(f"  Attached RTTS validation reference (sha256: {rtts_sha256[:12]}...)")
        except (OSError, IOError) as e:
            # SHADOW MODE: Never fail build due to RTTS issues
            print(f"  WARNING: Failed to attach RTTS validation reference: {e}")

    if snapshot_manifest_meta:
        manifest["proof_log_snapshot"] = snapshot_manifest_meta

    manifest_file = out_dir / "manifest.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    print()
    print("=" * 60)
    print("Evidence Pack Complete")
    print("=" * 60)
    print(f"  Total files: {len(manifest_entries)}")
    print(f"  Output: {out_dir}")
    print()
    print("Structure:")
    print("  evidence_pack_first_light/")
    print("    +-- p3_synthetic/        (6 artifacts)")
    print("    +-- p4_shadow/           (6 artifacts)")
    print(f"    +-- {viz_dir_display:23} {viz_section_note}")
    print("    +-- compliance/")
    print("    |   +-- compliance_narrative.md")
    print("    +-- manifest.json")
    print()
    print("SHADOW MODE: All artifacts are observation-only.")
    print("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build First-Light Evidence Pack"
    )
    parser.add_argument(
        "--p3-dir",
        type=str,
        required=True,
        help="P3 artifacts directory",
    )
    parser.add_argument(
        "--p4-dir",
        type=str,
        required=True,
        help="P4 artifacts directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/first_light/evidence_pack_first_light",
        help="Output directory for evidence pack",
    )
    parser.add_argument(
        "--cal-exp1-report",
        type=str,
        help="Optional path to cal_exp1_report.json for CAL-EXP-1 annex",
    )
    parser.add_argument(
        "--cal-exp2-report",
        type=str,
        help="Optional path to cal_exp2_report.json for CAL-EXP-2 annex",
    )
    parser.add_argument(
        "--cal-exp3-report",
        type=str,
        help="Optional path to cal_exp3_report.json for CAL-EXP-3 annex",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default="visualizations",
        help="Directory (relative to pack root) for generated SVG outputs",
    )
    parser.add_argument(
        "--include-proof-snapshot",
        action="store_true",
        help="Generate compliance/proof_log_snapshot.json (also enabled via FIRST_LIGHT_INCLUDE_PROOF_SNAPSHOT env var).",
    )
    parser.add_argument(
        "--proof-log",
        type=str,
        help="Explicit proof log JSONL path (defaults to searching under P3/P4 runs).",
    )
    parser.add_argument(
        "--ledger-guard-summary",
        type=str,
        help="Optional path to ledger_guard_summary.json for inclusion under governance/.",
    )
    args = parser.parse_args()

    try:
        build_evidence_pack(
            p3_dir=Path(args.p3_dir),
            p4_dir=Path(args.p4_dir),
            out_dir=Path(args.output_dir),
            plots_dir=Path(args.plots_dir) if args.plots_dir else None,
            include_proof_snapshot=args.include_proof_snapshot,
            proof_log=Path(args.proof_log) if args.proof_log else None,
            cal_exp1_report=Path(args.cal_exp1_report) if args.cal_exp1_report else None,
            cal_exp2_report=Path(args.cal_exp2_report) if args.cal_exp2_report else None,
            cal_exp3_report=Path(args.cal_exp3_report) if args.cal_exp3_report else None,
            ledger_guard_summary=Path(args.ledger_guard_summary)
            if args.ledger_guard_summary
            else None,
        )
        return 0
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
