#!/usr/bin/env python3
"""
Phase Migration Check — Unified Control Tower

Orchestrates the complete phase migration validation pipeline:
1. Generate Phase Impact Report (pr_migration_linter signals)
2. Run Migration Dry-Run Simulation (phase_migration_simulator)
3. Validate Migration Intent (if present)
4. Compare Intent against Impact Report (advisor mode)

Modes:
- Default: Full check with detailed output
- --summary: Reviewer-facing summary (red/yellow/green)
- --author-check: Pre-flight checklist for PR authors
- STRICT_PHASE_MIGRATION=1: CI hard-fail mode

JSON Contracts:
- --summary --json → SummaryResult (overall_signal, transitions, migration_intent, advisor_alignment)
- --author-check --json → AuthorCheckResult (preconditions_required, documented, missing, recommendations)

Author: Agent E4 (doc-ops-4) — Phase Migration Architect
Date: 2025-12-06

ABSOLUTE SAFEGUARDS:
- Read-only analysis — no mutations to production state
- Advisory only — no auto-approval or auto-rejection
- Deterministic output — same diff → same report

================================================================================
UX RECIPES — Who Uses What and When
================================================================================

FOR AUTHORS (before pushing):
    python tools/phase_migration_check.py --author-check
    
    This shows:
    - Required preconditions (PRE-001, PRE-002, etc.) for your changes
    - Suggested migration_intent.yaml sections to fill in
    - Verification commands to run before committing
    
    If signals detected, generate intent:
        python tools/pr_migration_linter.py --generate-template > migration_intent.yaml

FOR REVIEWERS (during PR review):
    python tools/phase_migration_check.py --summary
    
    This shows:
    - Traffic light: 🟢 GREEN / 🟡 YELLOW / 🔴 RED
    - Phase transitions detected (Phase I → II, etc.)
    - Migration intent status (PRESENT / MISSING / INVALID)
    - Advisor alignment (ALIGNED / MISALIGNED / etc.)
    
    For machine-readable output:
        python tools/phase_migration_check.py --summary --json

FOR CI (when to enable --strict):
    # Advisory mode (default) — logs warnings but doesn't fail
    python tools/phase_migration_check.py
    
    # Strict mode — fails on WARN or FAIL
    STRICT_PHASE_MIGRATION=1 python tools/phase_migration_check.py
    # OR
    python tools/phase_migration_check.py --strict
    
    Enable strict mode when:
    - The branch is protected (main, mvdp-*)
    - The team has adopted migration governance
    - You want to prevent accidental phase boundary crossings

STRICT MODE POLICY:
    When --strict or STRICT_PHASE_MIGRATION=1 is set:
    - ANY WARN or FAIL impact severity → exit 1
    - Missing migration_intent.yaml when signals present → exit 1
    - Advisor status MISALIGNED → exit 1
    
    This tool never auto-generates or mutates migration_intent.yaml.
    This tool never merges or rejects PRs — it advises; CI enforces via exit codes.
================================================================================
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# JSON Contract Dataclasses
# =============================================================================

@dataclass
class SummaryResult:
    """
    Formalized contract for --summary --json output.
    
    Contract:
    {
      "overall_signal": "GREEN|YELLOW|RED",
      "transitions": [{"phase": "...", "severity": "..."}],
      "migration_intent": "PRESENT|MISSING|INVALID",
      "advisor_alignment": "ALIGNED|MISALIGNED|INCOMPLETE|NO_IMPACT|N/A"
    }
    """
    overall_signal: str  # "GREEN" | "YELLOW" | "RED"
    transitions: list[dict[str, str]]
    migration_intent: str  # "PRESENT" | "MISSING" | "INVALID"
    advisor_alignment: str  # "ALIGNED" | "MISALIGNED" | "INCOMPLETE" | "NO_IMPACT" | "N/A"
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_signal": self.overall_signal,
            "transitions": self.transitions,
            "migration_intent": self.migration_intent,
            "advisor_alignment": self.advisor_alignment,
        }


@dataclass
class AuthorCheckResult:
    """
    Formalized contract for --author-check --json output.
    
    Contract:
    {
      "preconditions_required": [...],
      "preconditions_documented": [...],
      "preconditions_missing": [...],
      "signals_detected": [...],
      "recommendations": [...]
    }
    """
    preconditions_required: list[dict[str, str]]
    preconditions_documented: list[str]
    preconditions_missing: list[str]
    signals_detected: list[str]
    recommendations: list[str]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "preconditions_required": self.preconditions_required,
            "preconditions_documented": self.preconditions_documented,
            "preconditions_missing": self.preconditions_missing,
            "signals_detected": self.signals_detected,
            "recommendations": self.recommendations,
        }


@dataclass
class StrictModePolicy:
    """
    Codified strict mode CI enforcement policy.
    
    Policy (when strict=True):
    - ANY WARN or FAIL in phase impact → exit 1
    - Missing migration_intent.yaml when signals present → exit 1
    - MISALIGNED advisor status → exit 1
    """
    strict_enabled: bool
    has_critical_or_warn_impact: bool
    migration_intent_required: bool
    migration_intent_present: bool
    advisor_misaligned: bool
    
    @property
    def should_fail(self) -> bool:
        """Determine if CI should fail based on policy."""
        if not self.strict_enabled:
            return False
        
        # Policy: ANY WARN or FAIL in phase impact → exit 1
        if self.has_critical_or_warn_impact:
            return True
        
        # Policy: Missing migration_intent.yaml when signals present → exit 1
        if self.migration_intent_required and not self.migration_intent_present:
            return True
        
        # Policy: MISALIGNED advisor → exit 1
        if self.advisor_misaligned:
            return True
        
        return False
    
    @property
    def failure_reason(self) -> str | None:
        """Get the reason for CI failure."""
        if not self.should_fail:
            return None
        
        reasons = []
        if self.has_critical_or_warn_impact:
            reasons.append("WARN/CRITICAL phase impacts detected")
        if self.migration_intent_required and not self.migration_intent_present:
            reasons.append("migration_intent.yaml missing but required")
        if self.advisor_misaligned:
            reasons.append("migration intent misaligned with impact")
        
        return "; ".join(reasons)


@dataclass
class PhaseImpactMap:
    """
    Structured view of how each phase is affected by current changes.
    
    Contract:
    {
      "phases": {
        "PHASE_I": {"signal": "GREEN|YELLOW|RED", "notes": [...]},
        "PHASE_II": {...},
        "PHASE_IIB": {...},
        "PHASE_III": {...}
      }
    }
    """
    phases: dict[str, dict[str, Any]]
    
    def to_dict(self) -> dict[str, Any]:
        return {"phases": self.phases}


@dataclass
class MigrationPosture:
    """
    Migration posture snapshot — summary of current migration state.
    
    Contract:
    {
      "schema_version": "1.0.0",
      "overall_signal": "GREEN|YELLOW|RED",
      "strict_mode_recommended": true|false,
      "preconditions_missing_count": int,
      "phases_with_activity": ["PHASE_II", ...]
    }
    """
    schema_version: str
    overall_signal: str
    strict_mode_recommended: bool
    preconditions_missing_count: int
    phases_with_activity: list[str]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "overall_signal": self.overall_signal,
            "strict_mode_recommended": self.strict_mode_recommended,
            "preconditions_missing_count": self.preconditions_missing_count,
            "phases_with_activity": self.phases_with_activity,
        }


# =============================================================================
# Phase Impact Map Builder
# =============================================================================

# Canonical phase names for the impact map
CANONICAL_PHASES = ["PHASE_I", "PHASE_II", "PHASE_IIB", "PHASE_III"]

# Mapping from transition descriptions to affected phases
TRANSITION_TO_PHASES: dict[str, list[str]] = {
    "Phase I → Phase II": ["PHASE_I", "PHASE_II"],
    "Phase II → Phase IIb": ["PHASE_II", "PHASE_IIB"],
    "Phase II → Phase III": ["PHASE_II", "PHASE_III"],
    "Phase IIb → Phase III": ["PHASE_IIB", "PHASE_III"],
    "Determinism contract change": ["PHASE_I", "PHASE_II", "PHASE_IIB", "PHASE_III"],
    "Cross-phase": ["PHASE_I", "PHASE_II", "PHASE_IIB", "PHASE_III"],
}


def build_phase_impact_map(summary: SummaryResult) -> PhaseImpactMap:
    """
    Build a structured view of how each phase is affected by current changes.
    
    Derives signal from transitions + advisor alignment.
    
    Args:
        summary: The SummaryResult containing transitions and alignment info
        
    Returns:
        PhaseImpactMap with per-phase signals and notes
    """
    # Initialize all phases with GREEN and empty notes
    phases: dict[str, dict[str, Any]] = {
        phase: {"signal": "GREEN", "notes": []}
        for phase in CANONICAL_PHASES
    }
    
    # Process each transition
    for trans in summary.transitions:
        phase_name = trans.get("phase", "")
        severity = trans.get("severity", "INFO")
        
        # Determine which phases are affected
        affected_phases = TRANSITION_TO_PHASES.get(phase_name, [])
        
        # If no direct mapping, try to infer from phase name
        if not affected_phases:
            # Check for partial matches
            for key, phase_list in TRANSITION_TO_PHASES.items():
                if key.lower() in phase_name.lower() or phase_name.lower() in key.lower():
                    affected_phases = phase_list
                    break
            
            # Fallback: affect all phases for unknown transitions
            if not affected_phases:
                affected_phases = CANONICAL_PHASES
        
        # Map severity to signal
        severity_to_signal = {
            "CRITICAL": "RED",
            "WARN": "YELLOW",
            "INFO": "GREEN",
        }
        signal = severity_to_signal.get(severity, "GREEN")
        
        # Update affected phases
        for phase in affected_phases:
            if phase in phases:
                # Elevate signal (RED > YELLOW > GREEN)
                current_signal = phases[phase]["signal"]
                signal_priority = {"RED": 3, "YELLOW": 2, "GREEN": 1}
                if signal_priority.get(signal, 0) > signal_priority.get(current_signal, 0):
                    phases[phase]["signal"] = signal
                
                # Add note
                note = f"{phase_name}: {severity}"
                if note not in phases[phase]["notes"]:
                    phases[phase]["notes"].append(note)
    
    # Apply advisor alignment adjustment
    if summary.advisor_alignment == "MISALIGNED":
        # MISALIGNED affects all phases with transitions
        for phase, data in phases.items():
            if data["notes"]:  # Has activity
                if data["signal"] != "RED":
                    data["signal"] = "YELLOW"
                data["notes"].append("Advisor: MISALIGNED")
    elif summary.advisor_alignment == "INCOMPLETE":
        # INCOMPLETE is a warning on active phases
        for phase, data in phases.items():
            if data["notes"] and data["signal"] == "GREEN":
                data["signal"] = "YELLOW"
                data["notes"].append("Advisor: INCOMPLETE")
    
    # Apply migration intent status
    if summary.migration_intent == "MISSING":
        # Missing intent is a concern for phases with activity
        for phase, data in phases.items():
            if data["notes"]:
                if data["signal"] == "GREEN":
                    data["signal"] = "YELLOW"
                data["notes"].append("Intent: MISSING")
    elif summary.migration_intent == "INVALID":
        # Invalid intent is a bigger concern
        for phase, data in phases.items():
            if data["notes"]:
                if data["signal"] != "RED":
                    data["signal"] = "YELLOW"
                data["notes"].append("Intent: INVALID")
    
    return PhaseImpactMap(phases=phases)


# =============================================================================
# Migration Posture Builder
# =============================================================================

POSTURE_SCHEMA_VERSION = "1.0.0"


def build_migration_posture(
    summary: SummaryResult,
    author: AuthorCheckResult,
) -> MigrationPosture:
    """
    Build a migration posture snapshot summarizing the current state.
    
    Args:
        summary: The SummaryResult with overall signal and transitions
        author: The AuthorCheckResult with preconditions info
        
    Returns:
        MigrationPosture with posture snapshot data
    """
    # Determine phases with activity from transitions
    phases_with_activity: set[str] = set()
    for trans in summary.transitions:
        phase_name = trans.get("phase", "")
        affected = TRANSITION_TO_PHASES.get(phase_name, [])
        if not affected:
            # Try partial match
            for key, phase_list in TRANSITION_TO_PHASES.items():
                if key.lower() in phase_name.lower() or phase_name.lower() in key.lower():
                    affected = phase_list
                    break
        phases_with_activity.update(affected)
    
    # Determine if strict mode should be recommended
    # Recommend strict when:
    # - Any CRITICAL/WARN impacts
    # - Migration intent is missing or invalid
    # - Missing preconditions
    strict_recommended = (
        summary.overall_signal in ("RED", "YELLOW") or
        summary.migration_intent in ("MISSING", "INVALID") or
        len(author.preconditions_missing) > 0
    )
    
    return MigrationPosture(
        schema_version=POSTURE_SCHEMA_VERSION,
        overall_signal=summary.overall_signal,
        strict_mode_recommended=strict_recommended,
        preconditions_missing_count=len(author.preconditions_missing),
        phases_with_activity=sorted(list(phases_with_activity)),
    )


# =============================================================================
# Migration Governance Snapshot
# =============================================================================

GOVERNANCE_SNAPSHOT_SCHEMA_VERSION = "1.0.0"


def build_migration_governance_snapshot(
    impact_map: PhaseImpactMap,
    posture: MigrationPosture,
) -> dict[str, Any]:
    """
    Build a migration governance snapshot combining impact map and posture.
    
    Provides a consolidated view for governance decision-making.
    
    Args:
        impact_map: The PhaseImpactMap with per-phase signals
        posture: The MigrationPosture with overall state
        
    Returns:
        Dictionary with governance snapshot data
    """
    # Extract phases with RED signal
    phases_with_red = [
        phase
        for phase, data in impact_map.phases.items()
        if data.get("signal") == "RED"
    ]
    
    return {
        "schema_version": GOVERNANCE_SNAPSHOT_SCHEMA_VERSION,
        "overall_signal": posture.overall_signal,
        "phases_with_activity": posture.phases_with_activity,
        "phases_with_red_signal": sorted(phases_with_red),
        "preconditions_missing_count": posture.preconditions_missing_count,
        "strict_mode_recommended": posture.strict_mode_recommended,
    }


# =============================================================================
# Director Migration Status Mapping
# =============================================================================

def map_migration_to_director_status(
    governance_snapshot: dict[str, Any],
) -> dict[str, Any]:
    """
    Map migration governance snapshot to director status view.
    
    Provides a high-level status light and rationale for executive review.
    Uses neutral, non-prescriptive language.
    
    Args:
        governance_snapshot: The governance snapshot from build_migration_governance_snapshot
        
    Returns:
        Dictionary with status_light and rationale
    """
    overall_signal = governance_snapshot.get("overall_signal", "GREEN")
    phases_with_red = governance_snapshot.get("phases_with_red_signal", [])
    preconditions_missing = governance_snapshot.get("preconditions_missing_count", 0)
    phases_with_activity = governance_snapshot.get("phases_with_activity", [])
    
    # Status light matches overall signal
    status_light = overall_signal
    
    # Build neutral rationale
    rationale_parts: list[str] = []
    
    if phases_with_red:
        phase_list = ", ".join(phases_with_red)
        rationale_parts.append(f"{phase_list} show RED signals")
    
    if preconditions_missing > 0:
        rationale_parts.append(f"{preconditions_missing} preconditions missing")
    
    if phases_with_activity and not phases_with_red:
        phase_list = ", ".join(phases_with_activity)
        rationale_parts.append(f"Activity detected in {phase_list}")
    
    if not rationale_parts:
        rationale_parts.append("No migration activity detected")
    
    rationale = ". ".join(rationale_parts) + "."
    
    return {
        "status_light": status_light,
        "rationale": rationale,
    }


# =============================================================================
# Global Health Migration Summary
# =============================================================================

def summarize_migration_for_global_health(
    governance_snapshot: dict[str, Any],
) -> dict[str, Any]:
    """
    Summarize migration state for global health dashboard.
    
    Provides a compact signal suitable for system-wide health monitoring.
    
    Args:
        governance_snapshot: The governance snapshot from build_migration_governance_snapshot
        
    Returns:
        Dictionary with migration_ok, overall_signal, phases_with_red, status
    """
    overall_signal = governance_snapshot.get("overall_signal", "GREEN")
    phases_with_red = governance_snapshot.get("phases_with_red_signal", [])
    strict_recommended = governance_snapshot.get("strict_mode_recommended", False)
    
    # Determine status
    # BLOCK: RED signal AND strict mode recommended
    # WARN: YELLOW signal OR RED without strict recommendation
    # OK: GREEN signal
    if overall_signal == "RED" and strict_recommended:
        status = "BLOCK"
    elif overall_signal in ("RED", "YELLOW"):
        status = "WARN"
    else:
        status = "OK"
    
    # migration_ok: True when status is OK
    migration_ok = status == "OK"
    
    return {
        "migration_ok": migration_ok,
        "overall_signal": overall_signal,
        "phases_with_red": phases_with_red,
        "status": status,
    }


# =============================================================================
# Phase Migration Playbook
# =============================================================================

# Phase descriptions for playbook
PHASE_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "PHASE_I": {
        "name": "Phase I",
        "description": "Baseline RFL loop with in-memory state. No database persistence.",
        "typical_changes": [
            "Core RFL loop implementation",
            "Attestation and H_t computation",
            "Curriculum slice definitions",
            "Determinism envelope validation",
        ],
    },
    "PHASE_II": {
        "name": "Phase II",
        "description": "Uplift-enabled with database persistence. Evidence sealed.",
        "typical_changes": [
            "Database schema and migrations",
            "RFL database integration",
            "Evidence sealing mechanisms",
            "Preregistration documents",
        ],
    },
    "PHASE_IIB": {
        "name": "Phase IIb",
        "description": "Lean-enabled verification. Timeout calibration complete.",
        "typical_changes": [
            "Lean toolchain integration",
            "Proof generation and verification",
            "Timeout calibration",
            "U1 experiment analysis",
        ],
    },
    "PHASE_III": {
        "name": "Phase III",
        "description": "Generalized asymmetry with basis/ package. Crypto core active.",
        "typical_changes": [
            "basis/ package imports",
            "Ed25519 and RFC 8785 activation",
            "Proof middleware",
            "Security model elevation",
        ],
    },
}


def render_phase_migration_playbook(
    governance_snapshot: dict[str, Any],
    strict_policy: dict[str, Any] | None = None,
) -> str:
    """
    Render a phase migration playbook as markdown.
    
    Explains which phases are active, their signals, missing preconditions,
    and what kinds of changes typically belong to each phase.
    
    Args:
        governance_snapshot: The governance snapshot from build_migration_governance_snapshot
        strict_policy: Optional strict mode policy information
        
    Returns:
        Markdown string with playbook content
    """
    lines: list[str] = []
    
    # Header
    lines.append("# Phase Migration Playbook")
    lines.append("")
    lines.append("This playbook outlines the current phase migration state and guidance.")
    lines.append("")
    
    # Overall Status
    overall_signal = governance_snapshot.get("overall_signal", "GREEN")
    signal_emoji = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}.get(overall_signal, "⚪")
    lines.append(f"## {signal_emoji} Overall Status: {overall_signal}")
    lines.append("")
    
    # Active Phases
    phases_with_activity = governance_snapshot.get("phases_with_activity", [])
    phases_with_red = governance_snapshot.get("phases_with_red_signal", [])
    
    if phases_with_activity:
        lines.append("### Active Phases")
        lines.append("")
        lines.append("The following phases show migration activity:")
        lines.append("")
        for phase in phases_with_activity:
            signal = "RED" if phase in phases_with_red else "YELLOW"
            signal_icon = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}.get(signal, "⚪")
            phase_info = PHASE_DESCRIPTIONS.get(phase, {"name": phase, "description": "Unknown phase"})
            lines.append(f"- {signal_icon} **{phase_info['name']}** ({signal})")
            lines.append(f"  - {phase_info['description']}")
        lines.append("")
    else:
        lines.append("### Active Phases")
        lines.append("")
        lines.append("No active phase migration detected.")
        lines.append("")
    
    # Missing Preconditions
    preconditions_missing = governance_snapshot.get("preconditions_missing_count", 0)
    if preconditions_missing > 0:
        lines.append("### Missing Preconditions")
        lines.append("")
        lines.append(f"**{preconditions_missing} preconditions** are currently missing.")
        lines.append("")
        lines.append("Review the `--author-check` output for specific precondition requirements.")
        lines.append("")
    
    # Phase Descriptions
    lines.append("### Phase Characteristics")
    lines.append("")
    lines.append("Each phase has distinct characteristics and typical change patterns:")
    lines.append("")
    
    for phase_key in ["PHASE_I", "PHASE_II", "PHASE_IIB", "PHASE_III"]:
        phase_info = PHASE_DESCRIPTIONS.get(phase_key, {})
        if not phase_info:
            continue
        
        lines.append(f"#### {phase_info['name']}")
        lines.append("")
        lines.append(f"{phase_info['description']}")
        lines.append("")
        lines.append("**Typical changes:**")
        for change in phase_info.get("typical_changes", []):
            lines.append(f"- {change}")
        lines.append("")
    
    # Strict Mode
    strict_recommended = governance_snapshot.get("strict_mode_recommended", False)
    if strict_recommended:
        lines.append("### Strict Mode Recommendation")
        lines.append("")
        lines.append("⚠️ **Strict mode is recommended** for this migration.")
        lines.append("")
        lines.append("This means CI will fail on warnings, not just failures.")
        lines.append("")
    
    # Footer
    lines.append("---")
    lines.append("")
    lines.append("*Generated by `phase_migration_check.py --playbook`*")
    lines.append("")
    
    return "\n".join(lines)


# =============================================================================
# Cross-Agent Migration Contract
# =============================================================================

PHASE_CONTRACT_VERSION = "1.0.0"

# Expected downstream checks for each phase transition
DOWNSTREAM_CHECKS_BY_PHASE: dict[str, list[str]] = {
    "PHASE_I": ["curriculum_drift_guard", "determinism_audit"],
    "PHASE_II": [
        "curriculum_drift_guard",
        "determinism_audit",
        "metrics_conformance",
        "evidence_sealing_verification",
    ],
    "PHASE_IIB": [
        "curriculum_drift_guard",
        "determinism_audit",
        "metrics_conformance",
        "lean_proof_verification",
        "timeout_calibration_check",
    ],
    "PHASE_III": [
        "curriculum_drift_guard",
        "determinism_audit",
        "metrics_conformance",
        "telemetry_governance",
        "basis_purity_audit",
        "crypto_core_verification",
    ],
}


def build_phase_migration_contract(
    governance_snapshot: dict[str, Any],
) -> dict[str, Any]:
    """
    Build a cross-agent migration contract that all agents should respect.
    
    This contract defines what phases are involved, whether strict mode is required,
    and what downstream checks are expected.
    
    Args:
        governance_snapshot: The governance snapshot from build_migration_governance_snapshot
        
    Returns:
        Dictionary with phase migration contract data
    """
    phases_with_activity = governance_snapshot.get("phases_with_activity", [])
    strict_recommended = governance_snapshot.get("strict_mode_recommended", False)
    
    # Collect all downstream checks for active phases
    expected_checks: set[str] = set()
    for phase in phases_with_activity:
        checks = DOWNSTREAM_CHECKS_BY_PHASE.get(phase, [])
        expected_checks.update(checks)
    
    # If no specific phases, include base checks
    if not phases_with_activity:
        expected_checks.update(DOWNSTREAM_CHECKS_BY_PHASE.get("PHASE_I", []))
    
    return {
        "phase_contract_version": PHASE_CONTRACT_VERSION,
        "phases_involved": sorted(phases_with_activity),
        "strict_mode_required": strict_recommended,
        "expected_downstream_checks": sorted(list(expected_checks)),
    }


# =============================================================================
# Director Migration Story Block
# =============================================================================

def build_migration_director_panel(
    governance_snapshot: dict[str, Any],
    migration_contract: dict[str, Any],
) -> dict[str, Any]:
    """
    Build a director panel with status light, overall signal, and headline.
    
    Provides a high-level narrative about where in the phase ladder the current changes sit.
    
    Args:
        governance_snapshot: The governance snapshot from build_migration_governance_snapshot
        migration_contract: The migration contract from build_phase_migration_contract
        
    Returns:
        Dictionary with director panel data
    """
    overall_signal = governance_snapshot.get("overall_signal", "GREEN")
    phases_with_red = governance_snapshot.get("phases_with_red_signal", [])
    phases_involved = migration_contract.get("phases_involved", [])
    
    # Status light matches overall signal
    status_light = overall_signal
    
    # Build headline based on phase activity
    headline_parts: list[str] = []
    
    if phases_with_red:
        if len(phases_with_red) == 1:
            phase_name = PHASE_DESCRIPTIONS.get(phases_with_red[0], {}).get("name", phases_with_red[0])
            headline_parts.append(f"Critical activity in {phase_name}")
        else:
            phase_names = [
                PHASE_DESCRIPTIONS.get(p, {}).get("name", p) for p in phases_with_red[:2]
            ]
            headline_parts.append(f"Critical activity across {', '.join(phase_names)}")
            if len(phases_with_red) > 2:
                headline_parts.append(f"and {len(phases_with_red) - 2} other phase(s)")
    elif phases_involved:
        if len(phases_involved) == 1:
            phase_name = PHASE_DESCRIPTIONS.get(phases_involved[0], {}).get("name", phases_involved[0])
            headline_parts.append(f"Migration activity in {phase_name}")
        else:
            phase_names = [
                PHASE_DESCRIPTIONS.get(p, {}).get("name", p) for p in phases_involved[:2]
            ]
            headline_parts.append(f"Migration activity across {', '.join(phase_names)}")
            if len(phases_involved) > 2:
                headline_parts.append(f"and {len(phases_involved) - 2} other phase(s)")
    else:
        headline_parts.append("No active phase migration detected")
    
    # Add context about phase ladder position
    if phases_involved:
        # Determine highest phase involved
        phase_order = ["PHASE_I", "PHASE_II", "PHASE_IIB", "PHASE_III"]
        highest_phase_idx = max(
            (phase_order.index(p) for p in phases_involved if p in phase_order),
            default=-1
        )
        if highest_phase_idx >= 0:
            highest_phase = phase_order[highest_phase_idx]
            phase_name = PHASE_DESCRIPTIONS.get(highest_phase, {}).get("name", highest_phase)
            headline_parts.append(f"with changes reaching {phase_name}")
    
    headline = ". ".join(headline_parts) + "."
    
    return {
        "status_light": status_light,
        "overall_signal": overall_signal,
        "phases_with_red": phases_with_red,
        "headline": headline,
    }


# =============================================================================
# Evidence Pack Story Tile
# =============================================================================

EVIDENCE_TILE_SCHEMA_VERSION = "1.0.0"


def build_phase_migration_evidence_tile(
    migration_contract: dict[str, Any],
    director_panel: dict[str, Any],
) -> dict[str, Any]:
    """
    Build an evidence pack story tile for phase migration.
    
    This tile is designed to drop straight into an evidence pack or global health JSON.
    It consolidates migration contract and director panel data into a single,
    evidence-friendly structure.
    
    Args:
        migration_contract: The contract from build_phase_migration_contract
        director_panel: The panel from build_migration_director_panel
        
    Returns:
        Dictionary with evidence tile data
    """
    phases_involved = migration_contract.get("phases_involved", [])
    checks_required = migration_contract.get("expected_downstream_checks", [])
    status_light = director_panel.get("status_light", "GREEN")
    headline = director_panel.get("headline", "")
    
    # Build neutral notes
    neutral_notes: list[str] = []
    
    if phases_involved:
        phase_list = ", ".join(phases_involved)
        neutral_notes.append(f"Phases involved: {phase_list}")
    
    if checks_required:
        neutral_notes.append(f"{len(checks_required)} downstream checks required")
    
    if migration_contract.get("strict_mode_required", False):
        neutral_notes.append("Strict mode enforcement recommended")
    
    if not neutral_notes:
        neutral_notes.append("No active phase migration detected")
    
    return {
        "schema_version": EVIDENCE_TILE_SCHEMA_VERSION,
        "phases_involved": phases_involved,
        "status_light": status_light,
        "headline": headline,
        "checks_required": checks_required,
        "neutral_notes": neutral_notes,
    }


# =============================================================================
# Reviewer Guidance Builder (Markdown)
# =============================================================================

def build_reviewer_guidance(
    summary: SummaryResult,
    author: AuthorCheckResult,
    impact_map: PhaseImpactMap,
) -> str:
    """
    Build structured markdown blocks for reviewer PR comments.
    
    Uses advisory language ("consider", "check") not prescriptive ("must", "required").
    
    Args:
        summary: The SummaryResult with overall signal and transitions
        author: The AuthorCheckResult with preconditions info
        impact_map: The PhaseImpactMap with per-phase signals
        
    Returns:
        Markdown string suitable for PR comments
    """
    lines: list[str] = []
    
    # Header with traffic light
    signal_emoji = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}.get(
        summary.overall_signal, "⚪"
    )
    lines.append(f"## {signal_emoji} Phase Migration Review")
    lines.append("")
    
    # Section 1: Signals Observed
    lines.append("### Signals Observed")
    lines.append("")
    if author.signals_detected:
        for sig in author.signals_detected:
            lines.append(f"- `{sig}`")
    else:
        lines.append("- No migration signals detected")
    lines.append("")
    
    # Section 2: Phases Touched
    lines.append("### Phases Touched")
    lines.append("")
    lines.append("| Phase | Signal | Notes |")
    lines.append("|-------|--------|-------|")
    for phase, data in impact_map.phases.items():
        signal = data.get("signal", "GREEN")
        signal_icon = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}.get(signal, "⚪")
        notes = ", ".join(data.get("notes", [])) or "—"
        lines.append(f"| {phase} | {signal_icon} {signal} | {notes} |")
    lines.append("")
    
    # Section 3: Intent Status
    lines.append("### Migration Intent")
    lines.append("")
    intent_status = summary.migration_intent
    advisor_status = summary.advisor_alignment
    
    if intent_status == "PRESENT":
        lines.append("✅ `migration_intent.yaml` is present")
        if advisor_status == "ALIGNED":
            lines.append("✅ Intent aligns with detected impacts")
        elif advisor_status == "MISALIGNED":
            lines.append("⚠️ Intent may not fully match detected impacts — consider reviewing")
        elif advisor_status == "INCOMPLETE":
            lines.append("⚠️ Some signals may not be acknowledged — consider updating intent")
    elif intent_status == "MISSING":
        lines.append("❌ `migration_intent.yaml` not found")
        if author.signals_detected:
            lines.append("")
            lines.append("Consider whether this PR crosses a phase boundary.")
    elif intent_status == "INVALID":
        lines.append("⚠️ `migration_intent.yaml` has validation issues")
    lines.append("")
    
    # Section 4: Questions to Consider
    lines.append("### Questions to Consider")
    lines.append("")
    
    questions: list[str] = []
    
    # Add questions based on detected state
    if summary.overall_signal == "RED":
        questions.append("Is this a deliberate phase migration?")
        questions.append("Have all preconditions been verified?")
    
    if author.preconditions_missing:
        questions.append(
            f"Have the {len(author.preconditions_missing)} missing preconditions "
            f"({', '.join(author.preconditions_missing[:3])}...) been addressed?"
        )
    
    if summary.migration_intent == "MISSING" and author.signals_detected:
        questions.append("Should this PR include a `migration_intent.yaml`?")
    
    if advisor_status == "MISALIGNED":
        questions.append("Does the declared intent match what this PR actually changes?")
    
    # Add phase-specific questions
    for phase, data in impact_map.phases.items():
        if data.get("signal") == "RED":
            questions.append(f"Check: Is the {phase} boundary crossing intentional?")
        elif data.get("signal") == "YELLOW" and data.get("notes"):
            questions.append(f"Review: {phase} has warnings — any concerns?")
    
    # Default questions if none generated
    if not questions:
        questions.append("No specific concerns flagged — standard review applies.")
    
    for q in questions:
        lines.append(f"- {q}")
    lines.append("")
    
    # Footer with command hint
    lines.append("---")
    lines.append("")
    lines.append("*Generated by `phase_migration_check.py --reviewer-guidance`*")
    lines.append("")
    
    return "\n".join(lines)


# =============================================================================
# Precondition Registry — Maps signals to required preconditions
# =============================================================================

PRECONDITION_REGISTRY: dict[str, dict[str, Any]] = {
    # Phase I → Phase II preconditions
    "PRE-001": {
        "id": "PRE-001",
        "name": "Evidence Sealed",
        "description": "Phase I evidence (attestation.json) exists with valid H_t",
        "phase_transition": "Phase I → Phase II",
        "signals": ["db_write_introduction", "rfl_db_enable"],
        "verification": "python tools/verify_ht_recomputable.py",
    },
    "PRE-002": {
        "id": "PRE-002",
        "name": "Determinism Verified",
        "description": "Determinism tests pass with identical H_t across runs",
        "phase_transition": "Phase I → Phase II",
        "signals": ["db_write_introduction", "rfl_db_enable", "determinism_envelope_change"],
        "verification": "pytest tests/integration/test_first_organism_determinism.py",
    },
    "PRE-003": {
        "id": "PRE-003",
        "name": "Baseline Cycles Complete",
        "description": "1000+ cycle baseline exists in results/",
        "phase_transition": "Phase I → Phase II",
        "signals": ["db_write_introduction", "rfl_db_enable"],
        "verification": "wc -l results/fo_baseline.jsonl",
    },
    "PRE-004": {
        "id": "PRE-004",
        "name": "Preregistration Document",
        "description": "PREREG_UPLIFT_U2.yaml committed before experiments",
        "phase_transition": "Phase I → Phase II",
        "signals": ["prereg_addition", "uplift_slice_addition"],
        "verification": "test -f experiments/prereg/PREREG_UPLIFT_U2.yaml",
    },
    
    # Phase II → Phase IIb preconditions
    "PRE-005": {
        "id": "PRE-005",
        "name": "U1 Experiment Complete",
        "description": "U1 experiment results analyzed (INVALID/NULL/POSITIVE)",
        "phase_transition": "Phase II → Phase IIb",
        "signals": ["lean_enable", "lean_timeout_config"],
        "verification": "test -d results/uplift_u1/",
    },
    "PRE-006": {
        "id": "PRE-006",
        "name": "Lean Toolchain Built",
        "description": "backend/lean_proj/ builds successfully",
        "phase_transition": "Phase II → Phase IIb",
        "signals": ["lean_enable", "lean_toolchain_update"],
        "verification": "cd backend/lean_proj && lake build",
    },
    "PRE-007": {
        "id": "PRE-007",
        "name": "Timeout Calibrated",
        "description": "Lean timeout calibration performed and documented",
        "phase_transition": "Phase II → Phase IIb",
        "signals": ["lean_timeout_config"],
        "verification": "python tools/calibrate_lean_timeout.py",
    },
    
    # Phase II → Phase III preconditions
    "PRE-008": {
        "id": "PRE-008",
        "name": "Uplift Demonstrated",
        "description": "At least one U2 slice shows Δp ≥ 0.05",
        "phase_transition": "Phase II → Phase III",
        "signals": ["basis_import_activation", "ed25519_introduction"],
        "verification": "python experiments/analyze_uplift.py",
    },
    "PRE-009": {
        "id": "PRE-009",
        "name": "Crypto Core Audited",
        "description": "Security audit passed for backend/crypto/core.py",
        "phase_transition": "Phase II → Phase III",
        "signals": ["ed25519_introduction", "rfc8785_activation"],
        "verification": "python scripts/phase3_validation.py",
    },
    "PRE-010": {
        "id": "PRE-010",
        "name": "basis/ Import Purity",
        "description": "No forbidden imports in basis/ package",
        "phase_transition": "Phase II → Phase III",
        "signals": ["basis_import_activation"],
        "verification": "python tools/verify_basis_purity.py",
    },
    "PRE-011": {
        "id": "PRE-011",
        "name": "Proof Middleware Ready",
        "description": "Proof middleware tested with ≥80% coverage",
        "phase_transition": "Phase II → Phase III",
        "signals": ["proof_middleware_addition"],
        "verification": "pytest tests/test_proof_middleware.py --cov",
    },
    
    # Cross-phase preconditions
    "PRE-012": {
        "id": "PRE-012",
        "name": "Curriculum Monotonicity",
        "description": "Slice changes preserve atoms/depth_max monotonicity",
        "phase_transition": "Cross-phase",
        "signals": ["curriculum_active_change"],
        "verification": "python scripts/phase_migration_simulator.py --verbose",
    },
    "PRE-013": {
        "id": "PRE-013",
        "name": "Security Model Documented",
        "description": "Security elevation documented in PR description",
        "phase_transition": "Cross-phase",
        "signals": ["security_model_change"],
        "verification": "Manual review",
    },
}


@dataclass
class MigrationCheckResult:
    """Complete result of phase migration check."""
    check_id: str
    timestamp: str
    base_ref: str
    head_ref: str
    
    # Phase Impact Report results
    impact_report: dict[str, Any] | None
    impact_severity: str  # "CRITICAL" | "WARN" | "INFO" | "NONE"
    
    # Simulation results
    simulation_result: dict[str, Any] | None
    simulation_status: str  # "READY" | "READY_WITH_WARNINGS" | "BLOCKED"
    current_phase: str
    
    # Intent validation results
    intent_found: bool
    intent_valid: bool
    intent_validation: dict[str, Any] | None
    
    # Advisor results (if intent exists)
    advisor_result: dict[str, Any] | None
    advisor_status: str | None  # "ALIGNED" | "MISALIGNED" | "INCOMPLETE" | None
    
    # Overall verdict
    verdict: str  # "PASS" | "WARN" | "FAIL"
    summary: str
    recommendations: list[str]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "timestamp": self.timestamp,
            "base_ref": self.base_ref,
            "head_ref": self.head_ref,
            "impact_report": self.impact_report,
            "impact_severity": self.impact_severity,
            "simulation_result": self.simulation_result,
            "simulation_status": self.simulation_status,
            "current_phase": self.current_phase,
            "intent_found": self.intent_found,
            "intent_valid": self.intent_valid,
            "intent_validation": self.intent_validation,
            "advisor_result": self.advisor_result,
            "advisor_status": self.advisor_status,
            "verdict": self.verdict,
            "summary": self.summary,
            "recommendations": self.recommendations,
        }


def run_migration_check(
    base_ref: str = "main",
    head_ref: str = "HEAD",
    project_root: Path | None = None,
    output_dir: Path | None = None,
    verbose: bool = False,
) -> MigrationCheckResult:
    """
    Run the complete phase migration check pipeline.
    
    Args:
        base_ref: Base git ref (default: "main")
        head_ref: Head git ref (default: "HEAD")
        project_root: Project root directory (default: auto-detect)
        output_dir: Directory for output files (default: project_root)
        verbose: Print verbose output
    
    Returns:
        MigrationCheckResult with complete analysis
    """
    import hashlib
    
    if project_root is None:
        project_root = PROJECT_ROOT
    else:
        project_root = Path(project_root)
    
    if output_dir is None:
        output_dir = project_root
    else:
        output_dir = Path(output_dir)
    
    # Generate check ID
    timestamp = datetime.now(timezone.utc).isoformat()
    check_id = hashlib.sha256(
        f"migration_check_{base_ref}_{head_ref}_{timestamp}".encode()
    ).hexdigest()[:16]
    
    recommendations = []
    
    # -------------------------------------------------------------------------
    # Step 1: Generate Phase Impact Report
    # -------------------------------------------------------------------------
    if verbose:
        print("📊 Step 1: Generating Phase Impact Report...")
    
    try:
        from scripts.phase_migration_simulator import generate_phase_impact_report
        
        impact_report_path = output_dir / "phase_impact_report.json"
        impact_report = generate_phase_impact_report(
            base_ref=base_ref,
            head_ref=head_ref,
            out_path=impact_report_path,
            project_root=project_root,
        )
        impact_severity = impact_report.get("overall_severity", "NONE")
        
        if verbose:
            print(f"   Severity: {impact_severity}")
            print(f"   Impacts: {len(impact_report.get('impacts', []))}")
    except Exception as e:
        impact_report = None
        impact_severity = "ERROR"
        recommendations.append(f"Impact report generation failed: {e}")
        if verbose:
            print(f"   ❌ Error: {e}")
    
    # -------------------------------------------------------------------------
    # Step 2: Run Migration Simulation
    # -------------------------------------------------------------------------
    if verbose:
        print("\n🔍 Step 2: Running Migration Simulation...")
    
    try:
        from scripts.phase_migration_simulator import PhaseMigrationSimulator
        
        simulator = PhaseMigrationSimulator(project_root)
        sim_result = simulator.run_simulation()
        simulation_result = sim_result.to_dict()
        simulation_status = sim_result.overall_status
        current_phase = sim_result.current_phase.value
        
        # Save simulation result
        sim_result_path = output_dir / "migration_sim_result.json"
        simulator.save_result(sim_result, sim_result_path)
        
        if verbose:
            print(f"   Status: {simulation_status}")
            print(f"   Current Phase: {current_phase}")
    except Exception as e:
        simulation_result = None
        simulation_status = "ERROR"
        current_phase = "unknown"
        recommendations.append(f"Simulation failed: {e}")
        if verbose:
            print(f"   ❌ Error: {e}")
    
    # -------------------------------------------------------------------------
    # Step 3: Validate Migration Intent (if present)
    # -------------------------------------------------------------------------
    if verbose:
        print("\n📋 Step 3: Checking Migration Intent...")
    
    intent_path = project_root / "migration_intent.yaml"
    intent_found = intent_path.exists()
    intent_valid = False
    intent_validation = None
    
    if intent_found:
        try:
            from tools.validate_migration_intent import MigrationIntentValidator
            
            validator = MigrationIntentValidator()
            validation_result = validator.validate_file(intent_path)
            intent_validation = validation_result.to_dict()
            intent_valid = validation_result.valid
            
            if verbose:
                print(f"   Found: Yes")
                print(f"   Valid: {'Yes' if intent_valid else 'No'}")
        except Exception as e:
            intent_validation = {"error": str(e)}
            recommendations.append(f"Intent validation failed: {e}")
            if verbose:
                print(f"   ❌ Error: {e}")
    else:
        if verbose:
            print("   Found: No")
        if impact_severity == "CRITICAL":
            recommendations.append(
                "Critical migration signals detected but no migration_intent.yaml found"
            )
    
    # -------------------------------------------------------------------------
    # Step 4: Run Advisor (if intent and impact report exist)
    # -------------------------------------------------------------------------
    if verbose:
        print("\n🎯 Step 4: Running Migration Advisor...")
    
    advisor_result = None
    advisor_status = None
    
    if intent_found and impact_report:
        try:
            from tools.validate_migration_intent import MigrationIntentValidator
            import yaml
            
            validator = MigrationIntentValidator()
            with open(intent_path) as f:
                intent_content = yaml.safe_load(f) or {}
            
            advisor = validator.advise(intent_content, impact_report)
            advisor_result = advisor.to_dict()
            advisor_status = advisor.status
            
            if verbose:
                print(f"   Status: {advisor_status}")
                if advisor.missing_acknowledgments:
                    print(f"   Missing acknowledgments: {len(advisor.missing_acknowledgments)}")
            
            # Add advisor recommendations
            recommendations.extend(advisor.recommendations)
            
        except Exception as e:
            advisor_result = {"error": str(e)}
            recommendations.append(f"Advisor failed: {e}")
            if verbose:
                print(f"   ❌ Error: {e}")
    elif verbose:
        print("   Skipped (no intent or impact report)")
    
    # -------------------------------------------------------------------------
    # Determine Overall Verdict
    # -------------------------------------------------------------------------
    if verbose:
        print("\n⚖️  Determining Verdict...")
    
    # Verdict logic:
    # - FAIL: Critical impacts without valid intent, or simulation blocked
    # - WARN: Warning impacts, or advisor misaligned/incomplete
    # - PASS: No critical issues
    
    if impact_severity == "CRITICAL" and not intent_found:
        verdict = "FAIL"
        summary = "Critical migration signals detected without migration_intent.yaml"
    elif impact_severity == "CRITICAL" and not intent_valid:
        verdict = "FAIL"
        summary = "Critical migration signals detected with invalid migration_intent.yaml"
    elif advisor_status == "MISALIGNED":
        verdict = "FAIL"
        summary = "Migration intent does not match detected phase impacts"
    elif simulation_status == "BLOCKED":
        verdict = "WARN"
        summary = "Migration simulation blocked — preconditions not met"
    elif advisor_status == "INCOMPLETE":
        verdict = "WARN"
        summary = "Migration intent missing acknowledgments for critical signals"
    elif impact_severity == "WARN":
        verdict = "WARN"
        summary = "Warning-level migration signals detected"
    elif impact_severity == "CRITICAL" and advisor_status == "ALIGNED":
        verdict = "PASS"
        summary = "Critical migration signals properly declared and aligned"
    else:
        verdict = "PASS"
        summary = "No blocking migration issues detected"
    
    if verbose:
        verdict_icons = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}
        print(f"   {verdict_icons.get(verdict, '?')} {verdict}: {summary}")
    
    # Add detected transitions to recommendations for author guidance
    if impact_report:
        detected_impacts = impact_report.get("impacts", [])
        for impact in detected_impacts:
            if impact.get("severity") == "CRITICAL":
                recommendations.append(
                    f"Document preconditions for: {impact.get('phase', 'unknown')}"
                )
    
    return MigrationCheckResult(
        check_id=check_id,
        timestamp=timestamp,
        base_ref=base_ref,
        head_ref=head_ref,
        impact_report=impact_report,
        impact_severity=impact_severity,
        simulation_result=simulation_result,
        simulation_status=simulation_status,
        current_phase=current_phase,
        intent_found=intent_found,
        intent_valid=intent_valid,
        intent_validation=intent_validation,
        advisor_result=advisor_result,
        advisor_status=advisor_status,
        verdict=verdict,
        summary=summary,
        recommendations=recommendations,
    )


# =============================================================================
# Summary Mode — Reviewer-Facing Output
# =============================================================================

def build_summary_result(result: MigrationCheckResult) -> SummaryResult:
    """
    Build the formalized SummaryResult contract from MigrationCheckResult.
    
    This ensures --summary and --summary --json always return consistent data.
    """
    # Map verdict to traffic light signal
    verdict_to_signal = {
        "PASS": "GREEN",
        "WARN": "YELLOW",
        "FAIL": "RED",
    }
    overall_signal = verdict_to_signal.get(result.verdict, "RED")
    
    # Extract transitions from impact report
    transitions = []
    if result.impact_report:
        for impact in result.impact_report.get("impacts", []):
            transitions.append({
                "phase": impact.get("phase", "Unknown"),
                "severity": impact.get("severity", "UNKNOWN"),
            })
    
    # Determine migration intent status
    if not result.intent_found:
        migration_intent = "MISSING"
    elif not result.intent_valid:
        migration_intent = "INVALID"
    else:
        migration_intent = "PRESENT"
    
    # Determine advisor alignment
    if result.advisor_status:
        advisor_alignment = result.advisor_status
    else:
        advisor_alignment = "N/A"
    
    return SummaryResult(
        overall_signal=overall_signal,
        transitions=transitions,
        migration_intent=migration_intent,
        advisor_alignment=advisor_alignment,
    )


def format_reviewer_summary(result: MigrationCheckResult) -> str:
    """
    Format a short, human-friendly summary for reviewers.
    
    Shows red/yellow/green light with key status indicators.
    Uses build_summary_result() to ensure text and JSON views are consistent.
    """
    # Build the structured result for consistency
    summary_data = build_summary_result(result)
    
    lines = []
    lines.append("")
    lines.append("╔══════════════════════════════════════════════════════════════╗")
    lines.append("║           PHASE MIGRATION SUMMARY (Reviewer View)            ║")
    lines.append("╚══════════════════════════════════════════════════════════════╝")
    lines.append("")
    
    # Traffic light verdict (from structured data)
    lights = {"GREEN": "🟢 GREEN", "YELLOW": "🟡 YELLOW", "RED": "🔴 RED"}
    light = lights.get(summary_data.overall_signal, "⚪ UNKNOWN")
    lines.append(f"  Signal: {light}")
    lines.append("")
    
    # Phase transitions detected (from structured data)
    lines.append("  Transitions:")
    if summary_data.transitions:
        for trans in summary_data.transitions:
            severity_icon = {"CRITICAL": "🔴", "WARN": "🟡", "INFO": "🔵"}.get(
                trans.get("severity", ""), "⚪"
            )
            phase = trans.get("phase", "Unknown")
            severity = trans.get("severity", "UNKNOWN")
            lines.append(f"    {severity_icon} {phase} ({severity})")
    else:
        lines.append("    None detected")
    lines.append("")
    
    # Migration intent status (from structured data)
    lines.append("  MIGRATION_INTENT:")
    intent_icons = {"PRESENT": "✅", "MISSING": "❌", "INVALID": "⚠️"}
    icon = intent_icons.get(summary_data.migration_intent, "?")
    lines.append(f"    {icon} {summary_data.migration_intent}")
    lines.append("")
    
    # Advisor alignment (from structured data)
    lines.append("  ADVISOR:")
    alignment_icons = {
        "ALIGNED": "✅",
        "MISALIGNED": "❌",
        "INCOMPLETE": "⚠️",
        "NO_IMPACT": "ℹ️",
        "N/A": "—",
    }
    icon = alignment_icons.get(summary_data.advisor_alignment, "?")
    lines.append(f"    Alignment: {icon} {summary_data.advisor_alignment}")
    lines.append("")
    
    # Quick summary
    lines.append(f"  Summary: {result.summary}")
    lines.append("")
    
    return "\n".join(lines)


# =============================================================================
# Author Pre-Flight Checklist
# =============================================================================

def build_author_check_result(result: MigrationCheckResult) -> AuthorCheckResult:
    """
    Build the formalized AuthorCheckResult contract from MigrationCheckResult.
    
    This ensures --author-check and --author-check --json always return consistent data.
    """
    # Get required preconditions
    preconditions = get_required_preconditions(result)
    
    # Build preconditions_required list (full details)
    preconditions_required = [
        {
            "id": p.get("id", ""),
            "name": p.get("name", ""),
            "description": p.get("description", ""),
            "phase_transition": p.get("phase_transition", ""),
            "verification": p.get("verification", ""),
        }
        for p in preconditions
    ]
    
    # Detect documented preconditions from intent (if present)
    preconditions_documented = []
    if result.intent_validation and result.intent_validation.get("valid"):
        # Check if intent has preconditions_verified field
        intent_data = result.intent_validation.get("parsed_intent", {})
        if isinstance(intent_data, dict):
            verified = intent_data.get("preconditions_verified", [])
            for v in verified:
                # Extract PRE-XXX ID from string
                if isinstance(v, str) and "PRE-" in v:
                    import re
                    match = re.search(r"PRE-\d+", v)
                    if match:
                        preconditions_documented.append(match.group())
    
    # Calculate missing preconditions
    required_ids = {p.get("id", "") for p in preconditions}
    documented_ids = set(preconditions_documented)
    preconditions_missing = sorted(list(required_ids - documented_ids))
    
    # Collect detected signals
    signals_detected = []
    if result.impact_report:
        all_signals = set()
        for impact in result.impact_report.get("impacts", []):
            for sig in impact.get("signals", []):
                signal_name = sig.get("signal", "")
                if signal_name:
                    all_signals.add(signal_name)
        signals_detected = sorted(list(all_signals))
    
    # Build recommendations
    recommendations = []
    if preconditions_missing:
        recommendations.append(
            f"Document {len(preconditions_missing)} missing preconditions in migration_intent.yaml"
        )
        for pre_id in preconditions_missing[:5]:
            # Find precondition details
            precond = PRECONDITION_REGISTRY.get(pre_id)
            if precond:
                recommendations.append(
                    f"  → {pre_id}: {precond.get('name', 'Unknown')} - {precond.get('verification', '')}"
                )
    
    if not result.intent_found and preconditions:
        recommendations.append(
            "Create migration_intent.yaml: python tools/pr_migration_linter.py --generate-template"
        )
    
    if result.intent_found and not result.intent_valid:
        recommendations.append(
            "Fix validation errors in migration_intent.yaml"
        )
    
    if result.advisor_status == "MISALIGNED":
        recommendations.append(
            "Update migration_intent.yaml to match detected phase impacts"
        )
    
    return AuthorCheckResult(
        preconditions_required=preconditions_required,
        preconditions_documented=preconditions_documented,
        preconditions_missing=preconditions_missing,
        signals_detected=signals_detected,
        recommendations=recommendations,
    )


def get_required_preconditions(result: MigrationCheckResult) -> list[dict[str, Any]]:
    """
    Get the list of preconditions required for detected migration signals.
    
    Returns preconditions from PRECONDITION_REGISTRY that match detected signals.
    """
    required = []
    seen_ids = set()
    
    if not result.impact_report:
        return required
    
    # Collect all detected signals
    detected_signals = set()
    for impact in result.impact_report.get("impacts", []):
        for sig in impact.get("signals", []):
            signal_name = sig.get("signal", "")
            if signal_name:
                detected_signals.add(signal_name)
    
    # Match signals to preconditions
    for pre_id, precond in PRECONDITION_REGISTRY.items():
        for signal in precond.get("signals", []):
            if signal in detected_signals and pre_id not in seen_ids:
                required.append(precond)
                seen_ids.add(pre_id)
                break
    
    # Sort by ID
    required.sort(key=lambda x: x.get("id", ""))
    
    return required


def format_author_checklist(result: MigrationCheckResult) -> str:
    """
    Format a pre-flight checklist for PR authors.
    
    Suggests sections to fill in migration_intent.yaml based on detected signals.
    """
    lines = []
    lines.append("")
    lines.append("╔══════════════════════════════════════════════════════════════╗")
    lines.append("║         AUTHOR PRE-FLIGHT CHECKLIST                          ║")
    lines.append("╚══════════════════════════════════════════════════════════════╝")
    lines.append("")
    
    # Get required preconditions
    preconditions = get_required_preconditions(result)
    
    if not preconditions:
        lines.append("  ✅ No migration-specific preconditions required.")
        lines.append("")
        lines.append("  Your PR does not appear to cross phase boundaries.")
        lines.append("")
        return "\n".join(lines)
    
    # Header
    lines.append("  Your PR triggers migration signals. Please document the following")
    lines.append("  preconditions in your migration_intent.yaml:")
    lines.append("")
    
    # Group by phase transition
    by_phase: dict[str, list[dict]] = {}
    for precond in preconditions:
        phase = precond.get("phase_transition", "Unknown")
        if phase not in by_phase:
            by_phase[phase] = []
        by_phase[phase].append(precond)
    
    # Print grouped preconditions
    for phase, precond_list in by_phase.items():
        lines.append(f"  ── {phase} ──")
        lines.append("")
        for precond in precond_list:
            pre_id = precond.get("id", "???")
            name = precond.get("name", "Unknown")
            desc = precond.get("description", "")
            verification = precond.get("verification", "Manual")
            
            lines.append(f"  [{pre_id}] {name}")
            lines.append(f"       {desc}")
            lines.append(f"       Verify: {verification}")
            lines.append("")
    
    # Suggested migration_intent.yaml content
    lines.append("  ═══════════════════════════════════════════════════════════════")
    lines.append("  Suggested migration_intent.yaml sections:")
    lines.append("  ═══════════════════════════════════════════════════════════════")
    lines.append("")
    lines.append("  preconditions_verified:")
    for precond in preconditions:
        pre_id = precond.get("id", "???")
        name = precond.get("name", "Unknown")
        lines.append(f"    - \"{pre_id}: {name} - [VERIFIED/PENDING]\"")
    lines.append("")
    
    # Detected signals to acknowledge
    if result.impact_report:
        all_signals = set()
        for impact in result.impact_report.get("impacts", []):
            for sig in impact.get("signals", []):
                all_signals.add(sig.get("signal", ""))
        
        if all_signals:
            lines.append("  signals_acknowledged:")
            for sig in sorted(all_signals):
                if sig:
                    lines.append(f"    - {sig}")
            lines.append("")
    
    # Generate template command
    lines.append("  ═══════════════════════════════════════════════════════════════")
    lines.append("  To generate a template:")
    lines.append("  ═══════════════════════════════════════════════════════════════")
    lines.append("")
    lines.append("    python tools/pr_migration_linter.py --generate-template > migration_intent.yaml")
    lines.append("")
    
    return "\n".join(lines)


# =============================================================================
# Strict Mode (CI Enforcement)
# =============================================================================

def is_strict_mode() -> bool:
    """
    Check if strict mode is enabled via environment variable.
    
    STRICT_PHASE_MIGRATION=1 enables hard-fail CI mode.
    """
    return os.environ.get("STRICT_PHASE_MIGRATION", "").lower() in ("1", "true", "yes")


def main():
    """Main entry point for phase migration check."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Phase Migration Check — Unified Control Tower",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
╔════════════════════════════════════════════════════════════════════════════╗
║                          UX RECIPES                                         ║
╚════════════════════════════════════════════════════════════════════════════╝

FOR AUTHORS (before pushing):
  python phase_migration_check.py --author-check
  python phase_migration_check.py --author-check --json  # JSON contract

FOR REVIEWERS (during PR review):
  python phase_migration_check.py --summary
  python phase_migration_check.py --reviewer-guidance  # Markdown for PR comments

FOR CI (strict enforcement):
  STRICT_PHASE_MIGRATION=1 python phase_migration_check.py
  python phase_migration_check.py --strict

FOR DASHBOARDS/AUTOMATION:
  python phase_migration_check.py --posture        # Migration posture snapshot
  python phase_migration_check.py --impact-map     # Per-phase impact map
  python phase_migration_check.py --governance-snapshot  # Governance snapshot
  python phase_migration_check.py --director-status      # Director status view
  python phase_migration_check.py --global-health        # Global health summary
  python phase_migration_check.py --playbook            # Phase migration playbook (Markdown)
  python phase_migration_check.py --migration-contract  # Cross-agent contract (JSON)
  python phase_migration_check.py --director-panel      # Director panel (JSON)
  python phase_migration_check.py --evidence-tile      # Evidence pack story tile (JSON)

╔════════════════════════════════════════════════════════════════════════════╗
║                          EXAMPLES                                           ║
╚════════════════════════════════════════════════════════════════════════════╝

  # Full migration check
  python phase_migration_check.py
  
  # Reviewer summary with JSON output
  python phase_migration_check.py --summary --json
  
  # Author checklist with JSON output
  python phase_migration_check.py --author-check --json
  
  # Reviewer guidance (markdown for PR comments)
  python phase_migration_check.py --reviewer-guidance
  
  # Migration posture snapshot
  python phase_migration_check.py --posture
  
  # Per-phase impact map
  python phase_migration_check.py --impact-map

╔════════════════════════════════════════════════════════════════════════════╗
║                      JSON CONTRACTS                                         ║
╚════════════════════════════════════════════════════════════════════════════╝

--summary --json returns:
  {
    "overall_signal": "GREEN|YELLOW|RED",
    "transitions": [{"phase": "...", "severity": "..."}],
    "migration_intent": "PRESENT|MISSING|INVALID",
    "advisor_alignment": "ALIGNED|MISALIGNED|INCOMPLETE|NO_IMPACT|N/A"
  }

--author-check --json returns:
  {
    "preconditions_required": [...],
    "preconditions_documented": [...],
    "preconditions_missing": [...],
    "signals_detected": [...],
    "recommendations": [...]
  }

--posture returns:
  {
    "schema_version": "1.0.0",
    "overall_signal": "GREEN|YELLOW|RED",
    "strict_mode_recommended": true|false,
    "preconditions_missing_count": int,
    "phases_with_activity": ["PHASE_II", ...]
  }

--impact-map returns:
  {
    "phases": {
      "PHASE_I": {"signal": "GREEN|YELLOW|RED", "notes": [...]},
      "PHASE_II": {...}, ...
    }
  }

--governance-snapshot returns:
  {
    "schema_version": "1.0.0",
    "overall_signal": "GREEN|YELLOW|RED",
    "phases_with_activity": ["PHASE_II", ...],
    "phases_with_red_signal": ["PHASE_I", ...],
    "preconditions_missing_count": int,
    "strict_mode_recommended": true|false
  }

--director-status returns:
  {
    "status_light": "GREEN|YELLOW|RED",
    "rationale": "neutral description"
  }

--global-health returns:
  {
    "migration_ok": true|false,
    "overall_signal": "GREEN|YELLOW|RED",
    "phases_with_red": ["PHASE_I", ...],
    "status": "OK|WARN|BLOCK"
  }

--migration-contract returns:
  {
    "phase_contract_version": "1.0.0",
    "phases_involved": ["PHASE_II", ...],
    "strict_mode_required": true|false,
    "expected_downstream_checks": ["curriculum_drift_guard", ...]
  }

--director-panel returns:
  {
    "status_light": "GREEN|YELLOW|RED",
    "overall_signal": "GREEN|YELLOW|RED",
    "phases_with_red": ["PHASE_I", ...],
    "headline": "neutral narrative about phase ladder position"
  }

--evidence-tile returns:
  {
    "schema_version": "1.0.0",
    "phases_involved": ["PHASE_II", ...],
    "status_light": "GREEN|YELLOW|RED",
    "headline": "neutral narrative",
    "checks_required": ["curriculum_drift_guard", ...],
    "neutral_notes": ["Phases involved: ...", ...]
  }

╔════════════════════════════════════════════════════════════════════════════╗
║                      STRICT MODE POLICY                                     ║
╚════════════════════════════════════════════════════════════════════════════╝

When --strict or STRICT_PHASE_MIGRATION=1:
  - ANY WARN or FAIL impact → exit 1
  - Missing migration_intent.yaml when signals present → exit 1
  - MISALIGNED advisor status → exit 1

Environment Variables:
  STRICT_PHASE_MIGRATION=1  Hard-fail CI on WARN or FAIL
        """
    )
    parser.add_argument(
        "--base", "-b",
        default="main",
        help="Base git ref (default: main)",
    )
    parser.add_argument(
        "--head", "-H",
        default="HEAD",
        help="Head git ref (default: HEAD)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=None,
        help="Output directory for report files",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print verbose output",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output complete result as JSON",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error on warnings (not just failures)",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root directory (default: auto-detect)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Output reviewer-facing summary only (red/yellow/green)",
    )
    parser.add_argument(
        "--author-check",
        action="store_true",
        help="Output author pre-flight checklist",
    )
    parser.add_argument(
        "--reviewer-guidance",
        action="store_true",
        help="Output markdown guidance blocks for PR comments",
    )
    parser.add_argument(
        "--posture",
        action="store_true",
        help="Output migration posture snapshot (JSON)",
    )
    parser.add_argument(
        "--impact-map",
        action="store_true",
        help="Output per-phase impact map (JSON)",
    )
    parser.add_argument(
        "--governance-snapshot",
        action="store_true",
        help="Output migration governance snapshot (JSON)",
    )
    parser.add_argument(
        "--director-status",
        action="store_true",
        help="Output director migration status (JSON)",
    )
    parser.add_argument(
        "--global-health",
        action="store_true",
        help="Output global health migration summary (JSON)",
    )
    parser.add_argument(
        "--playbook",
        action="store_true",
        help="Output phase migration playbook (Markdown)",
    )
    parser.add_argument(
        "--migration-contract",
        action="store_true",
        help="Output cross-agent migration contract (JSON)",
    )
    parser.add_argument(
        "--director-panel",
        action="store_true",
        help="Output director migration panel (JSON)",
    )
    parser.add_argument(
        "--evidence-tile",
        action="store_true",
        help="Output evidence pack story tile (JSON)",
    )
    
    args = parser.parse_args()
    
    # Check for env-based strict mode
    strict_mode = args.strict or is_strict_mode()
    if is_strict_mode() and not args.strict:
        print("ℹ️  STRICT_PHASE_MIGRATION=1 detected — enabling strict mode")
        print()
    
    # Run the check (silently for specialized modes)
    specialized_modes = (
        args.summary or args.author_check or args.reviewer_guidance or
        args.posture or args.impact_map or args.governance_snapshot or
        args.director_status or args.global_health or args.playbook or
        args.migration_contract or args.director_panel or args.evidence_tile
    )
    verbose = args.verbose and not specialized_modes
    
    result = run_migration_check(
        base_ref=args.base,
        head_ref=args.head,
        project_root=args.project_root,
        output_dir=args.output_dir,
        verbose=verbose,
    )
    
    # -------------------------------------------------------------------------
    # Summary Mode (Reviewer-Facing)
    # -------------------------------------------------------------------------
    if args.summary:
        summary_result = build_summary_result(result)
        if args.json:
            # Output formalized JSON contract
            print(json.dumps(summary_result.to_dict(), indent=2))
        else:
            # Output human-readable text
            print(format_reviewer_summary(result))
        # No exit code change for summary mode — advisory only
        sys.exit(0)
    
    # -------------------------------------------------------------------------
    # Author Pre-Flight Checklist Mode
    # -------------------------------------------------------------------------
    if args.author_check:
        author_result = build_author_check_result(result)
        if args.json:
            # Output formalized JSON contract
            print(json.dumps(author_result.to_dict(), indent=2))
        else:
            # Output human-readable text
            print(format_author_checklist(result))
        # No exit code change for author-check mode — advisory only
        sys.exit(0)
    
    # -------------------------------------------------------------------------
    # Reviewer Guidance Mode (Markdown)
    # -------------------------------------------------------------------------
    if args.reviewer_guidance:
        summary_result = build_summary_result(result)
        author_result = build_author_check_result(result)
        impact_map = build_phase_impact_map(summary_result)
        
        if args.json:
            # Output structured data for programmatic use
            guidance_data = {
                "summary": summary_result.to_dict(),
                "author_check": author_result.to_dict(),
                "impact_map": impact_map.to_dict(),
            }
            print(json.dumps(guidance_data, indent=2))
        else:
            # Output markdown for PR comments
            print(build_reviewer_guidance(summary_result, author_result, impact_map))
        sys.exit(0)
    
    # -------------------------------------------------------------------------
    # Migration Posture Snapshot Mode
    # -------------------------------------------------------------------------
    if args.posture:
        summary_result = build_summary_result(result)
        author_result = build_author_check_result(result)
        posture = build_migration_posture(summary_result, author_result)
        
        # Posture is always JSON
        print(json.dumps(posture.to_dict(), indent=2))
        sys.exit(0)
    
    # -------------------------------------------------------------------------
    # Phase Impact Map Mode
    # -------------------------------------------------------------------------
    if args.impact_map:
        summary_result = build_summary_result(result)
        impact_map = build_phase_impact_map(summary_result)
        
        # Impact map is always JSON
        print(json.dumps(impact_map.to_dict(), indent=2))
        sys.exit(0)
    
    # -------------------------------------------------------------------------
    # Governance Snapshot Mode
    # -------------------------------------------------------------------------
    if args.governance_snapshot:
        summary_result = build_summary_result(result)
        author_result = build_author_check_result(result)
        impact_map = build_phase_impact_map(summary_result)
        posture = build_migration_posture(summary_result, author_result)
        governance = build_migration_governance_snapshot(impact_map, posture)
        
        # Governance snapshot is always JSON
        print(json.dumps(governance, indent=2))
        sys.exit(0)
    
    # -------------------------------------------------------------------------
    # Director Status Mode
    # -------------------------------------------------------------------------
    if args.director_status:
        summary_result = build_summary_result(result)
        author_result = build_author_check_result(result)
        impact_map = build_phase_impact_map(summary_result)
        posture = build_migration_posture(summary_result, author_result)
        governance = build_migration_governance_snapshot(impact_map, posture)
        director_status = map_migration_to_director_status(governance)
        
        # Director status is always JSON
        print(json.dumps(director_status, indent=2))
        sys.exit(0)
    
    # -------------------------------------------------------------------------
    # Global Health Mode
    # -------------------------------------------------------------------------
    if args.global_health:
        summary_result = build_summary_result(result)
        author_result = build_author_check_result(result)
        impact_map = build_phase_impact_map(summary_result)
        posture = build_migration_posture(summary_result, author_result)
        governance = build_migration_governance_snapshot(impact_map, posture)
        global_health = summarize_migration_for_global_health(governance)
        
        # Global health is always JSON
        print(json.dumps(global_health, indent=2))
        sys.exit(0)
    
    # -------------------------------------------------------------------------
    # Phase Migration Playbook Mode
    # -------------------------------------------------------------------------
    if args.playbook:
        summary_result = build_summary_result(result)
        author_result = build_author_check_result(result)
        impact_map = build_phase_impact_map(summary_result)
        posture = build_migration_posture(summary_result, author_result)
        governance = build_migration_governance_snapshot(impact_map, posture)
        
        # Build strict policy info if available
        strict_policy = None
        if strict_mode:
            strict_policy = {
                "enabled": True,
                "will_fail_on_warn": True,
            }
        
        # Playbook is always markdown
        playbook = render_phase_migration_playbook(governance, strict_policy)
        print(playbook)
        sys.exit(0)
    
    # -------------------------------------------------------------------------
    # Migration Contract Mode
    # -------------------------------------------------------------------------
    if args.migration_contract:
        summary_result = build_summary_result(result)
        author_result = build_author_check_result(result)
        impact_map = build_phase_impact_map(summary_result)
        posture = build_migration_posture(summary_result, author_result)
        governance = build_migration_governance_snapshot(impact_map, posture)
        contract = build_phase_migration_contract(governance)
        
        # Migration contract is always JSON
        print(json.dumps(contract, indent=2))
        sys.exit(0)
    
    # -------------------------------------------------------------------------
    # Director Panel Mode
    # -------------------------------------------------------------------------
    if args.director_panel:
        summary_result = build_summary_result(result)
        author_result = build_author_check_result(result)
        impact_map = build_phase_impact_map(summary_result)
        posture = build_migration_posture(summary_result, author_result)
        governance = build_migration_governance_snapshot(impact_map, posture)
        contract = build_phase_migration_contract(governance)
        panel = build_migration_director_panel(governance, contract)
        
        # Director panel is always JSON
        print(json.dumps(panel, indent=2))
        sys.exit(0)
    
    # -------------------------------------------------------------------------
    # Evidence Tile Mode
    # -------------------------------------------------------------------------
    if args.evidence_tile:
        summary_result = build_summary_result(result)
        author_result = build_author_check_result(result)
        impact_map = build_phase_impact_map(summary_result)
        posture = build_migration_posture(summary_result, author_result)
        governance = build_migration_governance_snapshot(impact_map, posture)
        contract = build_phase_migration_contract(governance)
        panel = build_migration_director_panel(governance, contract)
        evidence_tile = build_phase_migration_evidence_tile(contract, panel)
        
        # Evidence tile is always JSON
        print(json.dumps(evidence_tile, indent=2))
        sys.exit(0)
    
    # -------------------------------------------------------------------------
    # Default Full Output Mode
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("    PHASE MIGRATION CHECK — Control Tower")
    print("=" * 60)
    print()
    print(f"  Base: {args.base}")
    print(f"  Head: {args.head}")
    if strict_mode:
        print(f"  Mode: STRICT (failures and warnings will exit non-zero)")
    print()
    
    print("=" * 60)
    print("    RESULTS")
    print("=" * 60)
    print()
    
    # Print summary
    verdict_icons = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}
    print(f"  {verdict_icons.get(result.verdict, '?')} Verdict: {result.verdict}")
    print(f"  Summary: {result.summary}")
    print()
    
    print("  📊 Impact Analysis:")
    print(f"     Severity: {result.impact_severity}")
    if result.impact_report:
        impacts = result.impact_report.get("impacts", [])
        for impact in impacts[:3]:
            print(f"     • {impact['phase']} ({impact['signal_count']} signals)")
    print()
    
    print("  🔍 Simulation:")
    print(f"     Status: {result.simulation_status}")
    print(f"     Current Phase: {result.current_phase}")
    print()
    
    print("  📋 Migration Intent:")
    print(f"     Found: {'Yes' if result.intent_found else 'No'}")
    if result.intent_found:
        print(f"     Valid: {'Yes' if result.intent_valid else 'No'}")
        if result.advisor_status:
            print(f"     Advisor: {result.advisor_status}")
    print()
    
    if result.recommendations:
        print("  💡 Recommendations:")
        for rec in result.recommendations[:5]:
            print(f"     • {rec}")
        if len(result.recommendations) > 5:
            print(f"     ... and {len(result.recommendations) - 5} more")
        print()
    
    # Save full result
    output_dir = args.output_dir or PROJECT_ROOT
    result_path = output_dir / "migration_check_result.json"
    with open(result_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"  💾 Full result saved to: {result_path}")
    
    if args.json:
        print()
        print("=" * 60)
        print("    JSON OUTPUT")
        print("=" * 60)
        print(json.dumps(result.to_dict(), indent=2))
    
    # -------------------------------------------------------------------------
    # Exit Code Determination (Strict Mode Policy as Code)
    # -------------------------------------------------------------------------
    # Build strict mode policy
    policy = StrictModePolicy(
        strict_enabled=strict_mode,
        has_critical_or_warn_impact=result.impact_severity in ("CRITICAL", "WARN"),
        migration_intent_required=(
            result.impact_severity in ("CRITICAL", "WARN") and 
            result.impact_report is not None and
            len(result.impact_report.get("impacts", [])) > 0
        ),
        migration_intent_present=result.intent_found and result.intent_valid,
        advisor_misaligned=result.advisor_status == "MISALIGNED",
    )
    
    # Non-strict mode: only FAIL verdict causes exit 1
    if not strict_mode:
        if result.verdict == "FAIL":
            sys.exit(1)
        else:
            sys.exit(0)
    
    # Strict mode: apply policy
    if policy.should_fail:
        print()
        print("⚠️  STRICT MODE: CI Enforcement Triggered")
        print(f"   Reason: {policy.failure_reason}")
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()

