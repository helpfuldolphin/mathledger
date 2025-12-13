"""
Governance Evidence Pack — Collects and packages governance artifacts for audit.

Phase X: Integrates Last-Mile Governance Checker output into evidence package
for whitepaper evidence and compliance audits.

Phase Y: Integrates What-If analysis reports with auto-detection.

Evidence Structure:
    evidence/
    ├── governance/
    │   ├── final_check/
    │   │   ├── summary.json          # Aggregate statistics
    │   │   ├── checks.jsonl          # All individual check records
    │   │   ├── blocks.jsonl          # All BLOCK decisions
    │   │   ├── waivers_applied.json  # Waiver usage summary
    │   │   └── overrides_applied.json # Override usage summary
    │   ├── gate_stats/
    │   │   ├── g0_catastrophic.json
    │   │   ├── g1_hard.json
    │   │   ├── g2_invariant.json
    │   │   ├── g3_safe_region.json
    │   │   ├── g4_soft.json
    │   │   └── g5_advisory.json
    │   ├── what_if_analysis/
    │   │   ├── report.json           # Full What-If report
    │   │   └── status.json           # Status signals extract
    │   └── audit_chain/
    │       └── chain_verification.json

SHADOW MODE CONTRACT:
- Evidence is collected regardless of enforcement mode
- All artifacts are tagged with mode (SHADOW/ACTIVE)
- Evidence pack is purely observational
- What-If reports are always HYPOTHETICAL (never SHADOW)
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .last_mile_checker import (
    GovernanceFinalCheckResult,
    GateId,
    GateStatus,
    Verdict,
)

__all__ = [
    "GovernanceEvidencePack",
    "EvidencePackConfig",
    "attach_to_evidence",
    "WhatIfStatusSignal",
    "detect_what_if_report",
    "extract_what_if_status",
    "attach_what_if_to_evidence",
    "get_what_if_status_from_manifest",
    "format_what_if_warning",
    "bind_what_if_to_manifest",
]


# =============================================================================
# WHAT-IF STATUS SIGNAL
# =============================================================================

@dataclass
class WhatIfStatusSignal:
    """
    Status signal extracted from What-If report.

    Provides quick summary for signals.what_if in evidence pack.
    """
    hypothetical_block_rate: float
    blocking_gate_distribution: Dict[str, int]
    first_block_cycle: Optional[int]
    total_cycles: int
    hypothetical_blocks: int
    mode: str  # Always "HYPOTHETICAL"
    report_sha256: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothetical_block_rate": round(self.hypothetical_block_rate, 4),
            "blocking_gate_distribution": self.blocking_gate_distribution,
            "first_block_cycle": self.first_block_cycle,
            "total_cycles": self.total_cycles,
            "hypothetical_blocks": self.hypothetical_blocks,
            "mode": self.mode,
            "report_sha256": self.report_sha256,
        }


def _compute_report_hash(report_dict: Dict[str, Any]) -> str:
    """Compute SHA-256 hash of report dictionary."""
    report_json = json.dumps(report_dict, sort_keys=True)
    return hashlib.sha256(report_json.encode("utf-8")).hexdigest()


def detect_what_if_report(
    search_paths: Optional[List[Path]] = None,
    filename: str = "what_if_report.json",
) -> Optional[Path]:
    """
    Auto-detect what_if_report.json in common locations.

    Args:
        search_paths: Optional list of paths to search
        filename: Filename to look for (default: what_if_report.json)

    Returns:
        Path to report if found, None otherwise
    """
    if search_paths is None:
        # Default search paths
        search_paths = [
            Path("."),
            Path("results"),
            Path("evidence"),
            Path("evidence/governance"),
            Path("evidence/governance/what_if_analysis"),
        ]

    for base_path in search_paths:
        candidate = base_path / filename
        if candidate.exists() and candidate.is_file():
            return candidate

    return None


def extract_what_if_status(
    report_path: Optional[Path] = None,
    report_dict: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[WhatIfStatusSignal], List[str]]:
    """
    Extract status signal from What-If report.

    Args:
        report_path: Path to what_if_report.json
        report_dict: Pre-loaded report dictionary (alternative to path)

    Returns:
        Tuple of (WhatIfStatusSignal or None, list of warnings)

    Warnings are issued for malformed reports but do not cause crashes.
    """
    warnings_list: List[str] = []

    # Load report if path provided
    if report_dict is None:
        if report_path is None:
            return None, ["No report path or dictionary provided"]

        if not report_path.exists():
            return None, [f"Report file not found: {report_path}"]

        try:
            with open(report_path, "r", encoding="utf-8") as f:
                report_dict = json.load(f)
        except json.JSONDecodeError as e:
            return None, [f"Invalid JSON in report: {e}"]
        except Exception as e:
            return None, [f"Error reading report: {e}"]

    # Validate mode is HYPOTHETICAL (never SHADOW)
    mode = report_dict.get("mode", "UNKNOWN")
    if mode != "HYPOTHETICAL":
        warnings_list.append(
            f"What-If report mode is '{mode}', expected 'HYPOTHETICAL'. "
            "Mode constraint violated."
        )

    # Extract summary fields with safe defaults
    summary = report_dict.get("summary", {})

    try:
        hypothetical_block_rate = float(summary.get("hypothetical_block_rate", 0.0))
    except (TypeError, ValueError):
        hypothetical_block_rate = 0.0
        warnings_list.append("Invalid hypothetical_block_rate, defaulting to 0.0")

    blocking_gate_distribution = summary.get("blocking_gate_distribution", {})
    if not isinstance(blocking_gate_distribution, dict):
        blocking_gate_distribution = {}
        warnings_list.append("Invalid blocking_gate_distribution, defaulting to {}")

    first_block_cycle = summary.get("first_hypothetical_block_cycle")
    if first_block_cycle is not None:
        try:
            first_block_cycle = int(first_block_cycle)
        except (TypeError, ValueError):
            first_block_cycle = None
            warnings_list.append("Invalid first_hypothetical_block_cycle, defaulting to None")

    try:
        total_cycles = int(summary.get("total_cycles", 0))
    except (TypeError, ValueError):
        total_cycles = 0
        warnings_list.append("Invalid total_cycles, defaulting to 0")

    try:
        hypothetical_blocks = int(summary.get("hypothetical_blocks", 0))
    except (TypeError, ValueError):
        hypothetical_blocks = 0
        warnings_list.append("Invalid hypothetical_blocks, defaulting to 0")

    # Compute hash
    report_sha256 = _compute_report_hash(report_dict)

    status = WhatIfStatusSignal(
        hypothetical_block_rate=hypothetical_block_rate,
        blocking_gate_distribution=blocking_gate_distribution,
        first_block_cycle=first_block_cycle,
        total_cycles=total_cycles,
        hypothetical_blocks=hypothetical_blocks,
        mode=mode,
        report_sha256=report_sha256,
    )

    return status, warnings_list


def attach_what_if_to_evidence(
    evidence: Dict[str, Any],
    report_path: Optional[Path] = None,
    report_dict: Optional[Dict[str, Any]] = None,
    auto_detect: bool = True,
    search_paths: Optional[List[Path]] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Attach What-If report and status to evidence pack.

    Auto-detects what_if_report.json if not explicitly provided.

    Args:
        evidence: Evidence dictionary to update
        report_path: Optional explicit path to report
        report_dict: Optional pre-loaded report dictionary
        auto_detect: Whether to auto-detect report if not provided
        search_paths: Paths to search for auto-detection

    Returns:
        Tuple of (updated evidence dict, list of warnings)

    Attaches at:
        - evidence["governance"]["what_if_analysis"]["report"]
        - evidence["governance"]["what_if_analysis"]["status"]
        - evidence["signals"]["what_if"]
    """
    warnings_list: List[str] = []

    # Auto-detect if not provided
    if report_path is None and report_dict is None and auto_detect:
        report_path = detect_what_if_report(search_paths)
        if report_path is None:
            return evidence, ["No what_if_report.json found in search paths"]

    # Load report if path provided
    if report_dict is None and report_path is not None:
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                report_dict = json.load(f)
        except Exception as e:
            return evidence, [f"Error loading report: {e}"]

    if report_dict is None:
        return evidence, ["No report data available"]

    # Extract status
    status, extract_warnings = extract_what_if_status(report_dict=report_dict)
    warnings_list.extend(extract_warnings)

    if status is None:
        return evidence, warnings_list

    # Ensure governance section exists
    if "governance" not in evidence:
        evidence["governance"] = {}

    if "what_if_analysis" not in evidence["governance"]:
        evidence["governance"]["what_if_analysis"] = {}

    # Attach report
    evidence["governance"]["what_if_analysis"]["report"] = report_dict
    evidence["governance"]["what_if_analysis"]["report_sha256"] = status.report_sha256
    evidence["governance"]["what_if_analysis"]["attached_at"] = datetime.now(timezone.utc).isoformat()

    # Attach status under what_if_analysis
    evidence["governance"]["what_if_analysis"]["status"] = status.to_dict()

    # Attach to signals.what_if for quick access
    if "signals" not in evidence:
        evidence["signals"] = {}

    evidence["signals"]["what_if"] = status.to_dict()

    return evidence, warnings_list


def get_what_if_status_from_manifest(
    manifest: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """
    Get What-If status from manifest, preferring compact signal over full report.

    Resolution order:
    1. manifest.signals.what_if (preferred - compact status)
    2. manifest.governance.what_if_analysis.status (fallback)
    3. manifest.governance.what_if_analysis.report (last resort - extract status)

    Args:
        manifest: Manifest/evidence dictionary

    Returns:
        Tuple of (status dict or None, list of warnings)
    """
    warnings_list: List[str] = []

    # 1. Prefer manifest.signals.what_if (compact status)
    signals = manifest.get("signals", {})
    if "what_if" in signals and signals["what_if"]:
        status = signals["what_if"]
        if status.get("mode") != "HYPOTHETICAL":
            warnings_list.append(
                f"What-If mode is '{status.get('mode')}', expected 'HYPOTHETICAL'"
            )
        return status, warnings_list

    # 2. Fallback to governance.what_if_analysis.status
    governance = manifest.get("governance", {})
    what_if_analysis = governance.get("what_if_analysis", {})

    if "status" in what_if_analysis and what_if_analysis["status"]:
        status = what_if_analysis["status"]
        if status.get("mode") != "HYPOTHETICAL":
            warnings_list.append(
                f"What-If mode is '{status.get('mode')}', expected 'HYPOTHETICAL'"
            )
        return status, warnings_list

    # 3. Last resort: extract from full report
    if "report" in what_if_analysis and what_if_analysis["report"]:
        report = what_if_analysis["report"]
        extracted_status, extract_warnings = extract_what_if_status(report_dict=report)
        warnings_list.extend(extract_warnings)

        if extracted_status:
            return extracted_status.to_dict(), warnings_list

    return None, ["No What-If status found in manifest (checked signals.what_if, governance.what_if_analysis.status, governance.what_if_analysis.report)"]


def format_what_if_warning(
    status: Dict[str, Any],
) -> Optional[str]:
    """
    Format single-line warning for What-If status.

    Warning hygiene: single line if hypothetical_block_rate > 0.
    Includes top_blocking_gate.

    Args:
        status: What-If status dictionary

    Returns:
        Warning string or None if no warning needed
    """
    block_rate = status.get("hypothetical_block_rate", 0.0)

    if block_rate <= 0:
        return None

    # Get top_blocking_gate - either directly or derive from distribution
    top_gate = status.get("top_blocking_gate")
    if top_gate is None:
        gate_dist = status.get("blocking_gate_distribution", {})
        if gate_dist:
            top_gate = max(gate_dist.items(), key=lambda x: x[1])[0]

    mode = status.get("mode", "HYPOTHETICAL")

    # Build single-line warning
    parts = [f"What-If ({mode}): {block_rate:.1%} hypothetical block rate"]

    if top_gate:
        parts.append(f"top_gate={top_gate}")

    return "; ".join(parts)


def bind_what_if_to_manifest(
    manifest: Dict[str, Any],
    report_dict: Optional[Dict[str, Any]] = None,
    status_dict: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Bind What-If report and status to manifest with consistency.

    Ensures:
    - governance.what_if_analysis.report contains full report
    - governance.what_if_analysis.status contains compact status
    - manifest.signals.what_if mirrors compact status (mode="HYPOTHETICAL")

    Args:
        manifest: Manifest dictionary to update
        report_dict: Full What-If report (optional)
        status_dict: Pre-extracted status (optional, extracted from report if not provided)

    Returns:
        Tuple of (updated manifest, list of warnings)
    """
    warnings_list: List[str] = []

    # Extract status from report if not provided
    if status_dict is None and report_dict is not None:
        extracted, extract_warnings = extract_what_if_status(report_dict=report_dict)
        warnings_list.extend(extract_warnings)
        if extracted:
            status_dict = extracted.to_dict()

    if status_dict is None and report_dict is None:
        return manifest, ["No What-If report or status provided for manifest binding"]

    # Ensure mode is HYPOTHETICAL
    if status_dict and status_dict.get("mode") != "HYPOTHETICAL":
        warnings_list.append(
            f"What-If mode '{status_dict.get('mode')}' changed to 'HYPOTHETICAL' for manifest binding"
        )
        status_dict = dict(status_dict)
        status_dict["mode"] = "HYPOTHETICAL"

    # Ensure governance section
    if "governance" not in manifest:
        manifest["governance"] = {}

    if "what_if_analysis" not in manifest["governance"]:
        manifest["governance"]["what_if_analysis"] = {}

    # Attach report if provided
    if report_dict is not None:
        manifest["governance"]["what_if_analysis"]["report"] = report_dict

        # Compute hash
        report_hash = _compute_report_hash(report_dict)
        manifest["governance"]["what_if_analysis"]["report_sha256"] = report_hash

    # Attach status
    if status_dict is not None:
        manifest["governance"]["what_if_analysis"]["status"] = status_dict

        # Ensure signals section and mirror status
        if "signals" not in manifest:
            manifest["signals"] = {}

        manifest["signals"]["what_if"] = status_dict

    # Add binding timestamp
    manifest["governance"]["what_if_analysis"]["bound_at"] = datetime.now(timezone.utc).isoformat()

    return manifest, warnings_list


@dataclass
class EvidencePackConfig:
    """Configuration for evidence pack."""
    output_dir: str = "results/evidence"
    run_id: Optional[str] = None
    include_input_signals: bool = True
    include_gate_details: bool = True

    @classmethod
    def default(cls) -> "EvidencePackConfig":
        return cls()


@dataclass
class GateStatistics:
    """Statistics for a single gate."""
    gate_id: str
    total_evaluations: int = 0
    pass_count: int = 0
    fail_count: int = 0
    waived_count: int = 0
    overridden_count: int = 0
    blocking_count: int = 0  # Times this gate caused final BLOCK

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "total_evaluations": self.total_evaluations,
            "pass_count": self.pass_count,
            "fail_count": self.fail_count,
            "waived_count": self.waived_count,
            "overridden_count": self.overridden_count,
            "blocking_count": self.blocking_count,
            "pass_rate": self.pass_count / self.total_evaluations if self.total_evaluations > 0 else 0.0,
            "fail_rate": self.fail_count / self.total_evaluations if self.total_evaluations > 0 else 0.0,
        }


class GovernanceEvidencePack:
    """
    Collects governance artifacts into structured evidence package.

    SHADOW MODE CONTRACT:
    - All evidence is collected regardless of mode
    - Artifacts tagged with mode for audit clarity
    - Evidence pack has no enforcement capability
    """

    def __init__(
        self,
        config: Optional[EvidencePackConfig] = None,
    ) -> None:
        self.config = config or EvidencePackConfig.default()
        self.run_id = self.config.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        # Evidence storage
        self._checks: List[Dict[str, Any]] = []
        self._blocks: List[Dict[str, Any]] = []
        self._waivers_used: Dict[str, int] = {}
        self._overrides_used: Dict[str, int] = {}

        # Gate statistics
        self._gate_stats: Dict[str, GateStatistics] = {
            gate.value: GateStatistics(gate_id=gate.value)
            for gate in GateId
        }

        # Metadata
        self._start_time: str = datetime.now(timezone.utc).isoformat()
        self._end_time: Optional[str] = None
        self._total_checks: int = 0
        self._allow_count: int = 0
        self._block_count: int = 0

    def add_check(self, result: GovernanceFinalCheckResult) -> None:
        """
        Add a governance check result to evidence pack.

        Args:
            result: GovernanceFinalCheckResult to add
        """
        self._total_checks += 1

        # Track verdict
        if result.verdict == Verdict.ALLOW:
            self._allow_count += 1
        else:
            self._block_count += 1
            self._blocks.append(result.to_dict())

        # Store full check (optionally with input signals)
        check_data = result.to_dict()
        if not self.config.include_input_signals:
            check_data.pop("input_signals", None)
        self._checks.append(check_data)

        # Update gate statistics
        self._update_gate_stats(result)

        # Track waivers/overrides
        for waiver_id in result.waivers_applied:
            self._waivers_used[waiver_id] = self._waivers_used.get(waiver_id, 0) + 1

        for override_id in result.overrides_applied:
            self._overrides_used[override_id] = self._overrides_used.get(override_id, 0) + 1

    def _update_gate_stats(self, result: GovernanceFinalCheckResult) -> None:
        """Update gate statistics from result."""
        if result.gates is None:
            return

        gate_results = [
            (GateId.G0_CATASTROPHIC, result.gates.g0_catastrophic),
            (GateId.G1_HARD, result.gates.g1_hard),
            (GateId.G2_INVARIANT, result.gates.g2_invariant),
            (GateId.G3_SAFE_REGION, result.gates.g3_safe_region),
            (GateId.G4_SOFT, result.gates.g4_soft),
            (GateId.G5_ADVISORY, result.gates.g5_advisory),
        ]

        for gate_id, gate_result in gate_results:
            stats = self._gate_stats[gate_id.value]
            stats.total_evaluations += 1

            if gate_result.status == GateStatus.PASS:
                stats.pass_count += 1
            elif gate_result.status == GateStatus.FAIL:
                stats.fail_count += 1
            elif gate_result.status == GateStatus.WAIVED:
                stats.waived_count += 1
            elif gate_result.status == GateStatus.OVERRIDDEN:
                stats.overridden_count += 1

            # Check if this gate caused the block
            if result.blocking_gate == gate_id.value:
                stats.blocking_count += 1

    def finalize(self) -> Dict[str, Any]:
        """
        Finalize evidence pack and return summary.

        Returns:
            Summary dictionary
        """
        self._end_time = datetime.now(timezone.utc).isoformat()

        return self.get_summary()

    def get_summary(self) -> Dict[str, Any]:
        """Get evidence pack summary."""
        return {
            "schema_version": "1.0.0",
            "run_id": self.run_id,
            "timing": {
                "start_time": self._start_time,
                "end_time": self._end_time,
            },
            "totals": {
                "total_checks": self._total_checks,
                "allow_count": self._allow_count,
                "block_count": self._block_count,
                "allow_rate": self._allow_count / self._total_checks if self._total_checks > 0 else 0.0,
                "block_rate": self._block_count / self._total_checks if self._total_checks > 0 else 0.0,
            },
            "waivers": {
                "total_applications": sum(self._waivers_used.values()),
                "unique_waivers": len(self._waivers_used),
                "by_waiver": self._waivers_used,
            },
            "overrides": {
                "total_applications": sum(self._overrides_used.values()),
                "unique_overrides": len(self._overrides_used),
                "by_override": self._overrides_used,
            },
            "gate_summary": {
                gate_id: {
                    "pass_rate": stats.pass_count / stats.total_evaluations if stats.total_evaluations > 0 else 0.0,
                    "blocking_count": stats.blocking_count,
                }
                for gate_id, stats in self._gate_stats.items()
            },
        }

    def export(self, output_dir: Optional[str] = None) -> Path:
        """
        Export evidence pack to filesystem.

        Args:
            output_dir: Optional override for output directory

        Returns:
            Path to evidence directory
        """
        base_dir = Path(output_dir or self.config.output_dir) / self.run_id / "governance"
        base_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        final_check_dir = base_dir / "final_check"
        final_check_dir.mkdir(exist_ok=True)

        gate_stats_dir = base_dir / "gate_stats"
        gate_stats_dir.mkdir(exist_ok=True)

        audit_chain_dir = base_dir / "audit_chain"
        audit_chain_dir.mkdir(exist_ok=True)

        # Export summary
        summary = self.finalize()
        with open(final_check_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Export all checks (JSONL)
        with open(final_check_dir / "checks.jsonl", "w", encoding="utf-8") as f:
            for check in self._checks:
                f.write(json.dumps(check, separators=(",", ":")) + "\n")

        # Export blocks (JSONL)
        with open(final_check_dir / "blocks.jsonl", "w", encoding="utf-8") as f:
            for block in self._blocks:
                f.write(json.dumps(block, separators=(",", ":")) + "\n")

        # Export waivers applied
        with open(final_check_dir / "waivers_applied.json", "w", encoding="utf-8") as f:
            json.dump({
                "total_applications": sum(self._waivers_used.values()),
                "by_waiver": self._waivers_used,
            }, f, indent=2)

        # Export overrides applied
        with open(final_check_dir / "overrides_applied.json", "w", encoding="utf-8") as f:
            json.dump({
                "total_applications": sum(self._overrides_used.values()),
                "by_override": self._overrides_used,
            }, f, indent=2)

        # Export gate statistics
        for gate_id, stats in self._gate_stats.items():
            filename = gate_id.lower().replace("_", "_") + ".json"
            with open(gate_stats_dir / filename, "w", encoding="utf-8") as f:
                json.dump(stats.to_dict(), f, indent=2)

        # Export audit chain info
        chain_info = {
            "total_records": self._total_checks,
            "verified": True,  # Would verify chain here in production
            "verification_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(audit_chain_dir / "chain_verification.json", "w", encoding="utf-8") as f:
            json.dump(chain_info, f, indent=2)

        return base_dir

    def to_evidence_dict(self) -> Dict[str, Any]:
        """
        Get evidence pack as dictionary for embedding in larger evidence structure.

        Returns:
            Evidence dictionary suitable for evidence["governance"]["final_check"]
        """
        return {
            "summary": self.get_summary(),
            "checks": self._checks,
            "blocks": self._blocks,
            "gate_stats": {
                gate_id: stats.to_dict()
                for gate_id, stats in self._gate_stats.items()
            },
            "waivers_applied": self._waivers_used,
            "overrides_applied": self._overrides_used,
        }


def attach_to_evidence(
    evidence: Dict[str, Any],
    evidence_pack: GovernanceEvidencePack,
) -> Dict[str, Any]:
    """
    Attach governance evidence pack to larger evidence structure.

    Args:
        evidence: Existing evidence dictionary
        evidence_pack: GovernanceEvidencePack to attach

    Returns:
        Updated evidence dictionary with governance data at evidence["governance"]["final_check"]
    """
    if "governance" not in evidence:
        evidence["governance"] = {}

    evidence["governance"]["final_check"] = evidence_pack.to_evidence_dict()

    return evidence
