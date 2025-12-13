"""
P5 Identity Alignment Checker

Compares synthetic vs production configurations for P5 alignment before
enabling RealTelemetryAdapter. Produces OK / INVESTIGATE / BLOCK status.

See: docs/system_law/P5_Identity_Flight_Check_Runbook.md

Status: PHASE X P5 PRE-FLIGHT
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

__all__ = [
    "CheckResult",
    "CheckReport",
    "CheckItem",
    "check_p5_identity_alignment",
    "diagnose_config_divergence",
    "identity_preflight_for_alignment_view",
    "summarize_identity_preflight_signal_consistency",
    "EXTRACTION_SOURCE_CLI",
    "EXTRACTION_SOURCE_MANIFEST",
    "EXTRACTION_SOURCE_LEGACY_FILE",
    "EXTRACTION_SOURCE_RUN_CONFIG",
    "EXTRACTION_SOURCE_MISSING",
]

# Canonical extraction source constants
EXTRACTION_SOURCE_CLI = "CLI"
EXTRACTION_SOURCE_MANIFEST = "MANIFEST"
EXTRACTION_SOURCE_LEGACY_FILE = "LEGACY_FILE"
EXTRACTION_SOURCE_RUN_CONFIG = "RUN_CONFIG"
EXTRACTION_SOURCE_MISSING = "MISSING"


class CheckResult(Enum):
    """Result status for P5 identity checks."""

    OK = "OK"
    INVESTIGATE = "INVESTIGATE"
    BLOCK = "BLOCK"


@dataclass
class CheckItem:
    """Individual check result."""

    name: str
    status: CheckResult
    details: str
    invariant: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "name": self.name,
            "status": self.status.value,
            "details": self.details,
            "invariant": self.invariant,
        }


@dataclass
class CheckReport:
    """
    Report from P5 identity alignment check.

    Exit codes:
        0 = OK (safe to enable RealTelemetryAdapter)
        1 = INVESTIGATE (review items, proceed with caution)
        2 = BLOCK (do not enable, resolve issues first)
    """

    overall_status: CheckResult = CheckResult.OK
    checks: List[CheckItem] = field(default_factory=list)
    blocking_issues: List[str] = field(default_factory=list)
    investigation_items: List[str] = field(default_factory=list)
    timestamp: str = ""
    synthetic_fingerprint: str = ""
    production_fingerprint: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def add_check(
        self,
        name: str,
        status: CheckResult,
        details: str,
        invariant: Optional[str] = None,
    ) -> None:
        """Add a check result."""
        item = CheckItem(name=name, status=status, details=details, invariant=invariant)
        self.checks.append(item)

        if status == CheckResult.BLOCK:
            self.blocking_issues.append(f"{name}: {details}")
            self.overall_status = CheckResult.BLOCK
        elif status == CheckResult.INVESTIGATE and self.overall_status != CheckResult.BLOCK:
            self.investigation_items.append(f"{name}: {details}")
            self.overall_status = CheckResult.INVESTIGATE

    def get_exit_code(self) -> int:
        """Get CLI exit code."""
        if self.overall_status == CheckResult.BLOCK:
            return 2
        elif self.overall_status == CheckResult.INVESTIGATE:
            return 1
        return 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "schema": "p5-identity-alignment-report/1.0.0",
            "timestamp": self.timestamp,
            "overall_status": self.overall_status.value,
            "exit_code": self.get_exit_code(),
            "fingerprints": {
                "synthetic": self.synthetic_fingerprint,
                "production": self.production_fingerprint,
                "match": self.synthetic_fingerprint == self.production_fingerprint,
            },
            "invariant_summary": {
                "SI-001": self._get_invariant_status("SI-001"),
                "SI-002": self._get_invariant_status("SI-002"),
                "SI-003": self._get_invariant_status("SI-003"),
                "SI-004": self._get_invariant_status("SI-004"),
                "SI-005": self._get_invariant_status("SI-005"),
                "SI-006": self._get_invariant_status("SI-006"),
            },
            "checks": [c.to_dict() for c in self.checks],
            "blocking_issues": self.blocking_issues,
            "investigation_items": self.investigation_items,
        }

    def _get_invariant_status(self, invariant_id: str) -> str:
        """Get status for a specific invariant."""
        for check in self.checks:
            if check.invariant == invariant_id:
                return check.status.value
        return "UNCHECKED"

    def to_text_report(self) -> str:
        """Generate human-readable report."""
        lines = [
            "=" * 60,
            "P5 IDENTITY FLIGHT CHECK REPORT",
            "=" * 60,
            "",
            f"Timestamp: {self.timestamp}",
            f"Overall Status: {self.overall_status.value}",
            f"Exit Code: {self.get_exit_code()}",
            "",
            "FINGERPRINTS:",
            f"  Synthetic:  {self.synthetic_fingerprint[:32]}...",
            f"  Production: {self.production_fingerprint[:32]}...",
            f"  Match: {'YES' if self.synthetic_fingerprint == self.production_fingerprint else 'NO'}",
            "",
        ]

        if self.blocking_issues:
            lines.append("BLOCKING ISSUES:")
            for issue in self.blocking_issues:
                lines.append(f"  [X] {issue}")
            lines.append("")

        if self.investigation_items:
            lines.append("INVESTIGATION ITEMS:")
            for item in self.investigation_items:
                lines.append(f"  [?] {item}")
            lines.append("")

        lines.append("CHECK DETAILS:")
        for check in self.checks:
            icon = {"OK": "[OK]", "INVESTIGATE": "[??]", "BLOCK": "[XX]"}[check.status.value]
            inv = f"[{check.invariant}]" if check.invariant else "[---]"
            lines.append(f"  {icon} {inv} {check.name}")
            lines.append(f"        {check.details}")

        lines.extend(["", "=" * 60])

        return "\n".join(lines)


def _compute_fingerprint(config: Dict[str, Any]) -> str:
    """
    Compute slice fingerprint from config.

    Uses same algorithm as slice_identity.compute_slice_fingerprint().
    """
    relevant: Dict[str, Any] = {}

    if "params" in config:
        relevant["params"] = config["params"]

    if "gates" in config:
        relevant["gates"] = config["gates"]

    # If neither params nor gates, use entire config minus metadata
    if not relevant:
        relevant = {
            k: v
            for k, v in config.items()
            if not k.startswith("_") and k not in ("name", "version", "description", "created_at", "updated_at")
        }

    canonical = json.dumps(
        relevant,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )

    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten nested dict with dot-notation keys."""
    result: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            result.update(_flatten_dict(v, key))
        else:
            result[key] = v
    return result


def diagnose_config_divergence(
    synthetic_config: Dict[str, Any],
    production_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Diagnose configuration divergence between synthetic and production.

    Args:
        synthetic_config: Config used in P3/P4 synthetic runs
        production_config: Config from production environment

    Returns:
        Diagnosis report with differing parameters
    """
    syn_fp = _compute_fingerprint(synthetic_config)
    prod_fp = _compute_fingerprint(production_config)

    if syn_fp == prod_fp:
        return {
            "match": True,
            "diagnosis": "No config divergence",
            "synthetic_fingerprint": syn_fp,
            "production_fingerprint": prod_fp,
        }

    # Find differing params
    syn_params = synthetic_config.get("params", {})
    prod_params = production_config.get("params", {})

    differing_params = []
    all_keys = set(syn_params.keys()) | set(prod_params.keys())
    for key in sorted(all_keys):
        syn_val = syn_params.get(key)
        prod_val = prod_params.get(key)
        if syn_val != prod_val:
            differing_params.append({
                "param": key,
                "synthetic": syn_val,
                "production": prod_val,
            })

    # Find differing gates
    syn_gates = _flatten_dict(synthetic_config.get("gates", {}))
    prod_gates = _flatten_dict(production_config.get("gates", {}))

    differing_gates = []
    all_gate_keys = set(syn_gates.keys()) | set(prod_gates.keys())
    for key in sorted(all_gate_keys):
        syn_val = syn_gates.get(key)
        prod_val = prod_gates.get(key)
        if syn_val != prod_val:
            differing_gates.append({
                "gate": key,
                "synthetic": syn_val,
                "production": prod_val,
            })

    return {
        "match": False,
        "synthetic_fingerprint": syn_fp,
        "production_fingerprint": prod_fp,
        "differing_params": differing_params,
        "differing_gates": differing_gates,
        "diagnosis": "Config divergence detected",
        "recommended_action": "Align production config with synthetic baseline",
    }


def check_p5_identity_alignment(
    synthetic_config: Dict[str, Any],
    production_config: Dict[str, Any],
    p4_evidence_pack: Optional[Dict[str, Any]] = None,
) -> CheckReport:
    """
    Compare synthetic vs production configs for P5 alignment.

    This is the main entry point for P5 identity pre-flight checks.

    Args:
        synthetic_config: Config used in P3/P4 synthetic runs
        production_config: Config from production environment
        p4_evidence_pack: Optional P4 evidence pack for baseline comparison

    Returns:
        CheckReport with OK / INVESTIGATE / BLOCK status

    Exit Codes:
        0 = OK (safe to enable RealTelemetryAdapter)
        1 = INVESTIGATE (review items, proceed with caution)
        2 = BLOCK (do not enable, resolve issues first)

    Example:
        >>> report = check_p5_identity_alignment(syn_config, prod_config)
        >>> print(report.overall_status)
        CheckResult.OK
        >>> sys.exit(report.get_exit_code())
    """
    report = CheckReport()

    # Compute fingerprints
    syn_fp = _compute_fingerprint(synthetic_config)
    prod_fp = _compute_fingerprint(production_config)
    report.synthetic_fingerprint = syn_fp
    report.production_fingerprint = prod_fp

    # =========================================================================
    # CHECK 1: SI-001 - Fingerprint Match
    # =========================================================================
    if syn_fp == prod_fp:
        report.add_check(
            name="Fingerprint Match",
            status=CheckResult.OK,
            details=f"Fingerprints match: {syn_fp[:16]}...",
            invariant="SI-001",
        )
    else:
        report.add_check(
            name="Fingerprint Match",
            status=CheckResult.BLOCK,
            details=f"MISMATCH: syn={syn_fp[:16]}... prod={prod_fp[:16]}...",
            invariant="SI-001",
        )

    # =========================================================================
    # CHECK 2: SI-002 - Config Immutability Controls
    # =========================================================================
    hot_reload = production_config.get("_meta", {}).get("hot_reload_enabled", False)
    auto_scaling = production_config.get("_meta", {}).get("auto_scaling_enabled", False)

    if not hot_reload and not auto_scaling:
        report.add_check(
            name="Config Immutability",
            status=CheckResult.OK,
            details="Hot-reload and auto-scaling disabled",
            invariant="SI-002",
        )
    elif hot_reload:
        report.add_check(
            name="Config Immutability",
            status=CheckResult.INVESTIGATE,
            details="Hot-reload enabled - may cause mid-run drift",
            invariant="SI-002",
        )
    elif auto_scaling:
        report.add_check(
            name="Config Immutability",
            status=CheckResult.INVESTIGATE,
            details="Auto-scaling enabled - may affect config stability",
            invariant="SI-002",
        )

    # =========================================================================
    # CHECK 3: SI-003 - Drift Detection Wiring
    # =========================================================================
    drift_guard_enabled = production_config.get("_meta", {}).get("drift_guard_enabled", True)

    if drift_guard_enabled:
        report.add_check(
            name="Drift Detection",
            status=CheckResult.OK,
            details="Drift guard enabled",
            invariant="SI-003",
        )
    else:
        report.add_check(
            name="Drift Detection",
            status=CheckResult.BLOCK,
            details="Drift guard DISABLED - cannot detect runtime drift",
            invariant="SI-003",
        )

    # =========================================================================
    # CHECK 4: SI-004 - Provenance Chain
    # =========================================================================
    syn_curriculum_fp = synthetic_config.get("_curriculum_fingerprint")
    prod_curriculum_fp = production_config.get("_curriculum_fingerprint")

    if syn_curriculum_fp and prod_curriculum_fp:
        if syn_curriculum_fp == prod_curriculum_fp:
            report.add_check(
                name="Curriculum Fingerprint",
                status=CheckResult.OK,
                details=f"Curriculum match: {syn_curriculum_fp[:16]}...",
                invariant="SI-004",
            )
        else:
            report.add_check(
                name="Curriculum Fingerprint",
                status=CheckResult.INVESTIGATE,
                details=f"Curriculum diverge: syn={syn_curriculum_fp[:16]}... prod={prod_curriculum_fp[:16]}...",
                invariant="SI-004",
            )
    elif syn_curriculum_fp or prod_curriculum_fp:
        report.add_check(
            name="Curriculum Fingerprint",
            status=CheckResult.INVESTIGATE,
            details="Curriculum fingerprint missing in one config",
            invariant="SI-004",
        )
    else:
        report.add_check(
            name="Curriculum Fingerprint",
            status=CheckResult.INVESTIGATE,
            details="Curriculum fingerprint not available in either config",
            invariant="SI-004",
        )

    # =========================================================================
    # CHECK 5: SI-005 - P4 Evidence Baseline
    # =========================================================================
    if p4_evidence_pack:
        p4_slice_identity = p4_evidence_pack.get("governance", {}).get("slice_identity", {})
        p4_baseline_fp = p4_slice_identity.get("baseline_fingerprint") or p4_slice_identity.get("computed_fingerprint")

        if p4_baseline_fp:
            if prod_fp == p4_baseline_fp:
                report.add_check(
                    name="P4 Evidence Baseline",
                    status=CheckResult.OK,
                    details="Production matches P4 evidence baseline",
                    invariant="SI-005",
                )
            else:
                report.add_check(
                    name="P4 Evidence Baseline",
                    status=CheckResult.BLOCK,
                    details=f"Production differs from P4: prod={prod_fp[:16]}... p4={p4_baseline_fp[:16]}...",
                    invariant="SI-005",
                )
        else:
            report.add_check(
                name="P4 Evidence Baseline",
                status=CheckResult.INVESTIGATE,
                details="P4 evidence pack missing baseline fingerprint",
                invariant="SI-005",
            )
    else:
        report.add_check(
            name="P4 Evidence Baseline",
            status=CheckResult.INVESTIGATE,
            details="No P4 evidence pack provided for comparison",
            invariant="SI-005",
        )

    # =========================================================================
    # CHECK 6: SI-006 - Version Compatibility
    # =========================================================================
    syn_version = synthetic_config.get("version", "0.0.0")
    prod_version = production_config.get("version", "0.0.0")

    try:
        syn_parts = [int(x) for x in str(syn_version).split(".")[:3]]
        prod_parts = [int(x) for x in str(prod_version).split(".")[:3]]
        while len(syn_parts) < 3:
            syn_parts.append(0)
        while len(prod_parts) < 3:
            prod_parts.append(0)
        syn_major, syn_minor, syn_patch = syn_parts[:3]
        prod_major, prod_minor, prod_patch = prod_parts[:3]
    except (ValueError, IndexError):
        syn_major = syn_minor = syn_patch = 0
        prod_major = prod_minor = prod_patch = 0

    if syn_version == prod_version:
        report.add_check(
            name="Version Compatibility",
            status=CheckResult.OK,
            details=f"Versions match: {syn_version}",
            invariant="SI-006",
        )
    elif syn_major == prod_major:
        report.add_check(
            name="Version Compatibility",
            status=CheckResult.INVESTIGATE,
            details=f"Minor/patch version diff: syn={syn_version} prod={prod_version}",
            invariant="SI-006",
        )
    else:
        report.add_check(
            name="Version Compatibility",
            status=CheckResult.BLOCK,
            details=f"MAJOR version diff: syn={syn_version} prod={prod_version} - treat as new baseline",
            invariant="SI-006",
        )

    # =========================================================================
    # CHECK 7: Parameter-by-Parameter Diff (diagnostic)
    # =========================================================================
    syn_params = synthetic_config.get("params", {})
    prod_params = production_config.get("params", {})

    differing_params = []
    all_param_keys = set(syn_params.keys()) | set(prod_params.keys())
    for key in sorted(all_param_keys):
        syn_val = syn_params.get(key)
        prod_val = prod_params.get(key)
        if syn_val != prod_val:
            differing_params.append(f"{key}: syn={syn_val} prod={prod_val}")

    if not differing_params:
        report.add_check(
            name="Parameter Alignment",
            status=CheckResult.OK,
            details="All parameters match",
            invariant=None,
        )
    elif len(differing_params) <= 2:
        report.add_check(
            name="Parameter Alignment",
            status=CheckResult.INVESTIGATE,
            details=f"Differing params ({len(differing_params)}): {'; '.join(differing_params)}",
            invariant=None,
        )
    else:
        summary = "; ".join(differing_params[:3])
        if len(differing_params) > 3:
            summary += f"... (+{len(differing_params) - 3} more)"
        report.add_check(
            name="Parameter Alignment",
            status=CheckResult.BLOCK,
            details=f"Many differing params ({len(differing_params)}): {summary}",
            invariant=None,
        )

    # =========================================================================
    # CHECK 8: Gate Alignment (diagnostic)
    # =========================================================================
    syn_gates = _flatten_dict(synthetic_config.get("gates", {}))
    prod_gates = _flatten_dict(production_config.get("gates", {}))

    differing_gates = []
    all_gate_keys = set(syn_gates.keys()) | set(prod_gates.keys())
    for key in sorted(all_gate_keys):
        syn_val = syn_gates.get(key)
        prod_val = prod_gates.get(key)
        if syn_val != prod_val:
            differing_gates.append(f"{key}: syn={syn_val} prod={prod_val}")

    if not syn_gates and not prod_gates:
        report.add_check(
            name="Gate Alignment",
            status=CheckResult.OK,
            details="No gates configured",
            invariant=None,
        )
    elif not differing_gates:
        report.add_check(
            name="Gate Alignment",
            status=CheckResult.OK,
            details="All gates match",
            invariant=None,
        )
    else:
        summary = "; ".join(differing_gates[:3])
        if len(differing_gates) > 3:
            summary += f"... (+{len(differing_gates) - 3} more)"
        report.add_check(
            name="Gate Alignment",
            status=CheckResult.INVESTIGATE,
            details=f"Differing gates ({len(differing_gates)}): {summary}",
            invariant=None,
        )

    return report


# =============================================================================
# GGFL ADAPTER: identity_preflight_for_alignment_view
# =============================================================================


def _extract_identity_drivers(identity_report: Dict[str, Any]) -> List[str]:
    """
    Extract deterministic driver list from identity preflight report.

    Returns up to 3 non-OK invariant IDs (sorted ascending) for determinism.
    Invariant IDs are returned directly (e.g., "SI-001", "SI-003", "SI-005").

    Args:
        identity_report: Identity preflight report dict

    Returns:
        List of non-OK invariant IDs (max 3, sorted ascending)
    """
    # Extract invariant-based drivers (deterministic order by invariant ID ascending)
    invariant_summary = identity_report.get("invariant_summary", {})

    # Collect non-OK invariant IDs
    non_ok_invariants: List[str] = []
    for inv_id in sorted(invariant_summary.keys()):
        inv_status = invariant_summary.get(inv_id, "OK")
        if inv_status != "OK":
            non_ok_invariants.append(inv_id)

    # Return up to 3 invariant IDs (sorted, deterministic)
    return non_ok_invariants[:3]


def identity_preflight_for_alignment_view(
    identity_report: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Map identity preflight signal to GGFL alignment view.

    SHADOW MODE CONTRACT:
    - Purely observational, no side effects
    - Advisory classification only, no gating
    - This function NEVER influences control flow

    Mapping:
        identity_report absent -> status="ok", neutral summary
        status == "OK" -> status="ok"
        status == "BLOCK" or "INVESTIGATE" -> status="warn"

    Args:
        identity_report: Identity preflight report dict from p5_identity_preflight.json,
                         or None if not available.

    Returns:
        GGFL-normalized dict with:
        - signal_type: "SIG-ID" (constant)
        - status: "ok" | "warn"
        - conflict: False (constant, identity preflight is advisory)
        - drivers: List[str] (max 3, deterministic ordering)
        - summary: str (one neutral sentence)
        - mode: "SHADOW" (constant)
    """
    # Handle missing report
    if not identity_report:
        return {
            "signal_type": "SIG-ID",
            "status": "ok",
            "conflict": False,
            "drivers": [],
            "summary": "Identity preflight report not available.",
            "mode": "SHADOW",
        }

    # Extract key fields
    status = identity_report.get("status", "UNKNOWN")
    fp_match = identity_report.get("fingerprint_match", True)
    blocking_issues = identity_report.get("blocking_issues", [])
    investigation_items = identity_report.get("investigation_items", [])

    # Map status to GGFL status (ok/warn)
    if status in ("BLOCK", "INVESTIGATE"):
        ggfl_status = "warn"
    else:
        ggfl_status = "ok"

    # Extract deterministic drivers
    drivers = _extract_identity_drivers(identity_report)

    # Build neutral summary (no alarm language, factual only)
    if status == "OK" and fp_match:
        summary = "Identity preflight completed with no issues detected."
    elif status == "OK" and not fp_match:
        summary = "Identity preflight completed. Fingerprint difference noted."
    elif status == "INVESTIGATE":
        item_count = len(investigation_items)
        summary = f"Identity preflight flagged {item_count} item(s) for review."
    elif status == "BLOCK":
        issue_count = len(blocking_issues)
        summary = f"Identity preflight identified {issue_count} blocking issue(s)."
    else:
        summary = f"Identity preflight completed with status: {status}."

    return {
        "signal_type": "SIG-ID",
        "status": ggfl_status,
        "conflict": False,  # Identity preflight is advisory only
        "drivers": drivers,
        "summary": summary,
        "mode": "SHADOW",
    }


def summarize_identity_preflight_signal_consistency(
    status_signal: Dict[str, Any],
    ggfl_signal: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Cross-check consistency between status signal and GGFL SIG-ID signal.

    SIG-ID CONTRACT v1: Status↔GGFL Consistency Checker

    This function validates that the status signal and GGFL signal are consistent
    with each other and that the conflict invariant is maintained (conflict must
    always be False for identity preflight).

    SHADOW MODE CONTRACT:
    - This function is purely observational
    - It does not gate or block any operations
    - Detects inconsistencies for advisory purposes only
    - No gating, just neutral advisory notes

    Consistency levels:
    - CONSISTENT: Status mapping matches, conflict=False
    - PARTIAL: Minor differences (e.g., drivers count differs)
    - INCONSISTENT: Status mapping mismatch or conflict invariant violated

    Args:
        status_signal: Identity preflight signal from status JSON
            Must contain: status, fingerprint_match
        ggfl_signal: Identity preflight signal from GGFL adapter
            Must contain: signal_type, status, conflict, drivers

    Returns:
        Dictionary with:
        - schema_version: "1.0.0"
        - mode: "SHADOW"
        - consistency: "CONSISTENT" | "PARTIAL" | "INCONSISTENT"
        - notes: List of neutral descriptive notes about inconsistencies
        - conflict_invariant_violated: bool (True if conflict is ever True)
        - top_mismatch_type: Optional[str] (Top mismatch type for non-CONSISTENT)
    """
    notes: List[str] = []
    consistency = "CONSISTENT"
    conflict_invariant_violated = False
    top_mismatch_type: Optional[str] = None

    # Check conflict invariant (must always be False for identity preflight)
    ggfl_conflict = ggfl_signal.get("conflict", False)
    if ggfl_conflict is True:
        conflict_invariant_violated = True
        consistency = "INCONSISTENT"
        top_mismatch_type = "CONFLICT_INVARIANT"
        notes.append("Conflict invariant violated: GGFL conflict=True (expected False).")

    # Extract status values
    status_status = status_signal.get("status", "UNKNOWN")
    ggfl_status = ggfl_signal.get("status", "unknown")

    # Expected mapping: OK -> "ok", BLOCK/INVESTIGATE -> "warn"
    expected_ggfl_status = "ok" if status_status == "OK" else "warn"

    if ggfl_status != expected_ggfl_status:
        if consistency == "CONSISTENT":
            consistency = "INCONSISTENT"
            top_mismatch_type = "STATUS_MAPPING"
        notes.append(
            f"Status mapping mismatch: status={status_status} expected GGFL status="
            f"{expected_ggfl_status}, got {ggfl_status}."
        )

    # Check signal type invariant
    ggfl_signal_type = ggfl_signal.get("signal_type", "")
    if ggfl_signal_type != "SIG-ID":
        if consistency == "CONSISTENT":
            consistency = "PARTIAL"
            top_mismatch_type = "SIGNAL_TYPE"
        notes.append(f"Signal type mismatch: expected SIG-ID, got {ggfl_signal_type}.")

    # Check drivers consistency (max 3 invariant IDs, sorted)
    ggfl_drivers = ggfl_signal.get("drivers", [])
    if len(ggfl_drivers) > 3:
        if consistency == "CONSISTENT":
            consistency = "PARTIAL"
            top_mismatch_type = "DRIVERS_CAP"
        notes.append(f"Drivers cap exceeded: {len(ggfl_drivers)} drivers (max 3).")

    # Check drivers are sorted (determinism invariant)
    if ggfl_drivers != sorted(ggfl_drivers):
        if consistency == "CONSISTENT":
            consistency = "PARTIAL"
            top_mismatch_type = "DRIVERS_ORDER"
        notes.append("Drivers not sorted: violates determinism invariant.")

    # Check mode invariant
    ggfl_mode = ggfl_signal.get("mode", "")
    if ggfl_mode != "SHADOW":
        if consistency == "CONSISTENT":
            consistency = "PARTIAL"
            top_mismatch_type = "MODE_INVARIANT"
        notes.append(f"Mode invariant: expected SHADOW, got {ggfl_mode}.")

    return {
        "schema_version": "1.0.0",
        "mode": "SHADOW",
        "consistency": consistency,
        "notes": notes,
        "conflict_invariant_violated": conflict_invariant_violated,
        "top_mismatch_type": top_mismatch_type,
    }
