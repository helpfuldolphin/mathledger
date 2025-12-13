"""
CTRPK (Curriculum Transition Requests Per 1K Cycles) Detection Module.

Provides detection and attachment functions for CTRPK artifacts in evidence packs.

SHADOW MODE CONTRACT:
- All functions are purely observational
- Detection does not affect pack generation success/failure
- Provides curriculum churn metrics for calibration era monitoring

See docs/system_law/Curriculum_PhaseX_Invariants.md Section 12.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


# CTRPK artifact filename
CTRPK_COMPACT_ARTIFACT = "ctrpk_compact.json"


@dataclass
class CTRPKReference:
    """
    Reference to CTRPK (Curriculum Transition Requests Per 1K Cycles) compact block.

    SHADOW MODE CONTRACT:
    - CTRPK is purely observational
    - Detection does not affect pack generation success/failure
    - Provides curriculum churn metrics for calibration era monitoring
    """
    path: str
    sha256: str
    value: float
    status: str  # "OK" | "WARN" | "BLOCK"
    trend: str  # "IMPROVING" | "STABLE" | "DEGRADING"
    window_cycles: int
    transition_requests: int
    mode: str = "SHADOW"


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def detect_ctrpk_artifact(run_dir: Path) -> Optional[CTRPKReference]:
    """
    Detect and extract reference to CTRPK compact block if present.

    SHADOW MODE CONTRACT:
    - This function is purely observational
    - Detection does not affect pack generation success/failure
    - Provides curriculum churn metrics for calibration era monitoring

    Expected locations:
    - ctrpk_compact.json (root)
    - curriculum/ctrpk_compact.json

    Args:
        run_dir: Path to the run directory or evidence pack directory.

    Returns:
        CTRPKReference if ctrpk_compact.json exists, None otherwise.
    """
    # Check expected locations
    ctrpk_path = run_dir / CTRPK_COMPACT_ARTIFACT
    if not ctrpk_path.exists():
        ctrpk_path = run_dir / "curriculum" / CTRPK_COMPACT_ARTIFACT
        if not ctrpk_path.exists():
            return None

    try:
        # Compute hash
        file_hash = compute_file_hash(ctrpk_path)

        # Extract key fields
        with open(ctrpk_path, "r", encoding="utf-8") as f:
            ctrpk_data = json.load(f)

        # Determine relative path for reference
        rel_path = (
            f"curriculum/{CTRPK_COMPACT_ARTIFACT}"
            if (run_dir / "curriculum" / CTRPK_COMPACT_ARTIFACT).exists()
            else CTRPK_COMPACT_ARTIFACT
        )

        return CTRPKReference(
            path=rel_path,
            sha256=file_hash,
            value=ctrpk_data.get("value", 0.0),
            status=ctrpk_data.get("status", "OK"),
            trend=ctrpk_data.get("trend", "STABLE"),
            window_cycles=ctrpk_data.get("window_cycles", 0),
            transition_requests=ctrpk_data.get("transition_requests", 0),
            mode="SHADOW",
        )
    except (json.JSONDecodeError, OSError):
        # File exists but couldn't be parsed - still reference it with defaults
        try:
            file_hash = compute_file_hash(ctrpk_path)
            rel_path = (
                f"curriculum/{CTRPK_COMPACT_ARTIFACT}"
                if (run_dir / "curriculum" / CTRPK_COMPACT_ARTIFACT).exists()
                else CTRPK_COMPACT_ARTIFACT
            )
            return CTRPKReference(
                path=rel_path,
                sha256=file_hash,
                value=0.0,
                status="UNKNOWN",
                trend="STABLE",
                window_cycles=0,
                transition_requests=0,
                mode="SHADOW",
            )
        except OSError:
            return None


def attach_ctrpk_to_manifest(
    manifest: Dict[str, Any],
    ctrpk_ref: CTRPKReference,
) -> Dict[str, Any]:
    """
    Attach CTRPK reference to evidence pack manifest.

    SHADOW MODE CONTRACT:
    - Non-mutating: creates new dict with CTRPK attached
    - CTRPK is advisory only, does not gate manifest validity

    Args:
        manifest: Evidence pack manifest dict
        ctrpk_ref: CTRPK reference from detect_ctrpk_artifact()

    Returns:
        Updated manifest with CTRPK under governance.curriculum.ctrpk
    """
    import copy
    result = copy.deepcopy(manifest)

    if "governance" not in result:
        result["governance"] = {}

    if "curriculum" not in result["governance"]:
        result["governance"]["curriculum"] = {}

    result["governance"]["curriculum"]["ctrpk"] = {
        "path": ctrpk_ref.path,
        "sha256": ctrpk_ref.sha256,
        "value": ctrpk_ref.value,
        "status": ctrpk_ref.status,
        "trend": ctrpk_ref.trend,
        "window_cycles": ctrpk_ref.window_cycles,
        "transition_requests": ctrpk_ref.transition_requests,
        # SHADOW MODE CONTRACT marker
        "mode": "SHADOW",
        "shadow_mode_contract": {
            "observational_only": True,
            "no_control_flow_influence": True,
            "no_governance_modification": True,
        },
    }

    return result


def load_ctrpk_from_manifest(manifest: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extract CTRPK data from evidence pack manifest.

    Args:
        manifest: Evidence pack manifest dict

    Returns:
        CTRPK compact dict if present, None otherwise
    """
    governance = manifest.get("governance", {})
    curriculum = governance.get("curriculum", {})
    return curriculum.get("ctrpk")


def ctrpk_for_alignment_view(
    signal_or_ref: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert CTRPK signal/reference to GGFL alignment view format.

    PHASE X — GGFL ADAPTER FOR CTRPK (SIG-CTRPK)

    Normalizes the CTRPK signal into the Global Governance Fusion Layer (GGFL)
    unified format for cross-subsystem alignment views.

    SHADOW MODE CONTRACT:
    - This function is purely observational
    - It does not gate or block any operations
    - Never claims "good/bad", only descriptive
    - CTRPK never triggers conflict directly (conflict: false always)
    - Deterministic output for identical inputs (byte-identical for identical inputs)

    Args:
        signal_or_ref: CTRPK signal dict (from first_light_status.json) or
            manifest CTRPK dict (from governance.curriculum.ctrpk).
            Expected keys: value, status, trend
            Optional keys: window_cycles, transition_requests

    Returns:
        GGFL-normalized dict with:
        - signal_type: "SIG-CTRPK" (identifies this as a curriculum transition signal)
        - status: "ok" | "warn" (warn if status is WARN/BLOCK or trend is DEGRADING)
        - conflict: false (CTRPK never triggers conflict directly, invariant)
        - drivers: List[str] (value+trend drivers, deterministic ordering)
        - summary: str (one neutral sentence)
        - weight_hint: "LOW" (ensures CTRPK doesn't overpower fusion)
        - shadow_mode_invariants: Dict with advisory_only, no_enforcement, conflict_invariant
    """
    # Extract CTRPK fields
    value = signal_or_ref.get("value", 0.0)
    ctrpk_status = signal_or_ref.get("status", "OK").upper()
    trend = signal_or_ref.get("trend", "STABLE").upper()
    window_cycles = signal_or_ref.get("window_cycles", 0)
    transition_requests = signal_or_ref.get("transition_requests", 0)

    # Determine GGFL status: warn if status is WARN/BLOCK or trend is DEGRADING
    warn_conditions = (
        ctrpk_status in ("WARN", "BLOCK")
        or trend == "DEGRADING"
    )
    status = "warn" if warn_conditions else "ok"

    # Build drivers using reason codes only (max 3)
    # Canonical reason codes: DRIVER_STATUS_BLOCK, DRIVER_STATUS_WARN, DRIVER_TREND_DEGRADING
    drivers: list[str] = []

    # Priority order: BLOCK > WARN > DEGRADING (deterministic)
    if ctrpk_status == "BLOCK":
        drivers.append("DRIVER_STATUS_BLOCK")
    if ctrpk_status == "WARN":
        drivers.append("DRIVER_STATUS_WARN")
    if trend == "DEGRADING":
        drivers.append("DRIVER_TREND_DEGRADING")

    # Limit to 3 drivers (already enforced by logic above, max is 3 unique codes)
    drivers = drivers[:3]

    # Build neutral summary sentence
    if warn_conditions:
        # Describe specific condition
        if ctrpk_status in ("WARN", "BLOCK"):
            summary = (
                f"CTRPK: curriculum churn metric at {value:.2f} "
                f"(status={ctrpk_status}, trend={trend}). "
                f"Monitoring curriculum transition rate over {window_cycles} cycles."
            )
        else:
            # trend == DEGRADING
            summary = (
                f"CTRPK: curriculum churn metric at {value:.2f} "
                f"with {trend} trend (status={ctrpk_status}). "
                f"Monitoring curriculum transition rate over {window_cycles} cycles."
            )
    else:
        summary = (
            f"CTRPK: curriculum churn metric at {value:.2f} "
            f"(status={ctrpk_status}, trend={trend}). "
            f"Curriculum transition rate nominal over {window_cycles} cycles."
        )

    return {
        "signal_type": "SIG-CTRPK",
        "status": status,
        "conflict": False,  # CTRPK never triggers conflict directly (invariant)
        "drivers": drivers,
        "summary": summary,
        "weight_hint": "LOW",  # Ensures CTRPK doesn't overpower fusion semantics
        "shadow_mode_invariants": {
            "advisory_only": True,
            "no_enforcement": True,
            "conflict_invariant": True,
        },
    }
