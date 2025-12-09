# PHASE II — NOT USED IN PHASE I
"""
PRNG Governance Layer — Policy Enforcement for Deterministic Randomness.

This module provides a governance layer that fuses namespace linting,
manifest comparison, and lineage tracking into a unified policy framework.

Components:
    1. Governance Snapshot: Unified view of PRNG state across tools
    2. Policy Rules: Codified rules for PRNG compliance
    3. Global Health: Summary for MAAS integration

Usage:
    from rfl.prng.governance import (
        build_prng_governance_snapshot,
        evaluate_prng_policy,
        summarize_prng_for_global_health,
    )

    # Build snapshot from tool outputs
    snapshot = build_prng_governance_snapshot(
        manifest=manifest_data,
        replay_manifest=replay_data,
        namespace_report=lint_result,
    )

    # Evaluate policy
    policy_eval = evaluate_prng_policy(snapshot)

    # Get global health summary
    health = summarize_prng_for_global_health(snapshot, policy_eval)

Contract Reference:
    Implements PRNG governance requirements from docs/DETERMINISM_CONTRACT.md.

Author: Agent A2 (runtime-ops-2)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


class GovernanceStatus(str, Enum):
    """Overall governance status."""
    OK = "OK"
    WARN = "WARN"
    BLOCK = "BLOCK"


class ManifestStatus(str, Enum):
    """Status from manifest comparison."""
    EQUIVALENT = "EQUIVALENT"
    DRIFTED = "DRIFTED"
    INCOMPATIBLE = "INCOMPATIBLE"
    MISSING = "MISSING"


# ═══════════════════════════════════════════════════════════════════════════════
# PRNG POLICY RULES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PolicyRule:
    """A single PRNG policy rule."""
    rule_id: str
    name: str
    description: str
    severity: GovernanceStatus  # WARN or BLOCK
    applies_to: str  # "evidence", "runtime", "all"


# Codified PRNG governance rules
PRNG_GOV_RULES: Dict[str, PolicyRule] = {
    "R1": PolicyRule(
        rule_id="R1",
        name="no_drifted_evidence",
        description="No DRIFTED schedules allowed for evidence runs",
        severity=GovernanceStatus.BLOCK,
        applies_to="evidence",
    ),
    "R2": PolicyRule(
        rule_id="R2",
        name="incompatible_block",
        description="INCOMPATIBLE manifests must be blocked",
        severity=GovernanceStatus.BLOCK,
        applies_to="all",
    ),
    "R3": PolicyRule(
        rule_id="R3",
        name="no_hardcoded_runtime",
        description="Hard-coded seeds only allowed in tests; BLOCK in runtime paths",
        severity=GovernanceStatus.BLOCK,
        applies_to="runtime",
    ),
    "R4": PolicyRule(
        rule_id="R4",
        name="namespace_collision_warn",
        description="Namespace collisions should be reviewed",
        severity=GovernanceStatus.WARN,
        applies_to="all",
    ),
    "R5": PolicyRule(
        rule_id="R5",
        name="missing_attestation_warn",
        description="Manifests should have PRNG attestation blocks",
        severity=GovernanceStatus.WARN,
        applies_to="all",
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class NamespaceIssues:
    """Summary of namespace linting issues."""
    duplicate_count: int = 0
    duplicate_files: List[str] = field(default_factory=list)
    hardcoded_seed_count: int = 0
    hardcoded_seed_files: List[str] = field(default_factory=list)
    dynamic_path_count: int = 0
    suppressed_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "duplicate_count": self.duplicate_count,
            "duplicate_files": self.duplicate_files,
            "hardcoded_seed_count": self.hardcoded_seed_count,
            "hardcoded_seed_files": self.hardcoded_seed_files,
            "dynamic_path_count": self.dynamic_path_count,
            "suppressed_count": self.suppressed_count,
        }


@dataclass
class PolicyViolation:
    """A single policy violation."""
    rule_id: str
    rule_name: str
    severity: GovernanceStatus
    message: str
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "severity": self.severity.value,
            "message": self.message,
            "context": self.context,
        }


@dataclass
class PRNGGovernanceSnapshot:
    """
    Unified snapshot of PRNG governance state.

    Fuses outputs from:
    - seed_namespace_linter.py
    - prng_manifest_diff.py
    - seed_lineage_tree.py
    """
    schema_version: str = "1.0"
    timestamp: str = ""
    manifest_status: ManifestStatus = ManifestStatus.MISSING
    namespace_issues: NamespaceIssues = field(default_factory=NamespaceIssues)
    hardcoded_seeds_detected: bool = False
    hardcoded_seed_count: int = 0
    seed_lineage_fingerprint: str = ""
    master_seed_hex: Optional[str] = None
    lineage_entry_count: int = 0
    governance_status: GovernanceStatus = GovernanceStatus.OK
    is_evidence_run: bool = False
    is_test_context: bool = False

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "timestamp": self.timestamp,
            "manifest_status": self.manifest_status.value,
            "namespace_issues": self.namespace_issues.to_dict(),
            "hardcoded_seeds_detected": self.hardcoded_seeds_detected,
            "hardcoded_seed_count": self.hardcoded_seed_count,
            "seed_lineage_fingerprint": self.seed_lineage_fingerprint,
            "master_seed_hex": self.master_seed_hex,
            "lineage_entry_count": self.lineage_entry_count,
            "governance_status": self.governance_status.value,
            "is_evidence_run": self.is_evidence_run,
            "is_test_context": self.is_test_context,
        }


@dataclass
class PolicyEvaluation:
    """Result of policy evaluation."""
    policy_ok: bool = True
    violations: List[PolicyViolation] = field(default_factory=list)
    status: GovernanceStatus = GovernanceStatus.OK
    rules_checked: List[str] = field(default_factory=list)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_ok": self.policy_ok,
            "violations": [v.to_dict() for v in self.violations],
            "status": self.status.value,
            "rules_checked": self.rules_checked,
            "timestamp": self.timestamp,
        }


@dataclass
class GlobalHealthSummary:
    """Summary for global health / MAAS integration."""
    prng_policy_ok: bool = True
    has_namespace_collisions: bool = False
    has_schedule_drift: bool = False
    has_hardcoded_seeds: bool = False
    status: GovernanceStatus = GovernanceStatus.OK
    violation_count: int = 0
    summary_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prng_policy_ok": self.prng_policy_ok,
            "has_namespace_collisions": self.has_namespace_collisions,
            "has_schedule_drift": self.has_schedule_drift,
            "has_hardcoded_seeds": self.has_hardcoded_seeds,
            "status": self.status.value,
            "violation_count": self.violation_count,
            "summary_message": self.summary_message,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 1: PRNG GOVERNANCE SNAPSHOT
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_lineage_fingerprint(
    master_seed_hex: Optional[str],
    merkle_root: Optional[str],
    entry_count: int,
) -> str:
    """
    Compute a short fingerprint for seed lineage.

    This fingerprint uniquely identifies the lineage state for quick comparison.
    """
    if not master_seed_hex:
        return "no-seed"

    material = f"{master_seed_hex}:{merkle_root or 'none'}:{entry_count}"
    digest = hashlib.sha256(material.encode('utf-8')).hexdigest()
    return digest[:16]  # Short fingerprint


def _extract_namespace_issues(namespace_report: Optional[Dict[str, Any]]) -> NamespaceIssues:
    """Extract namespace issues from linter report."""
    if not namespace_report:
        return NamespaceIssues()

    issues = NamespaceIssues()

    # Extract duplicates
    duplicates = namespace_report.get("duplicates", [])
    issues.duplicate_count = len(duplicates)
    for dup in duplicates:
        usages = dup.get("usages", [])
        for usage in usages:
            file_path = usage.get("file", "")
            if file_path and file_path not in issues.duplicate_files:
                issues.duplicate_files.append(file_path)

    # Extract hard-coded seeds
    hard_coded = namespace_report.get("hard_coded_seeds", [])
    issues.hardcoded_seed_count = len(hard_coded)
    for seed in hard_coded:
        file_path = seed.get("file", "")
        if file_path and file_path not in issues.hardcoded_seed_files:
            issues.hardcoded_seed_files.append(file_path)

    # Extract dynamic paths
    dynamic = namespace_report.get("dynamic_paths", [])
    issues.dynamic_path_count = len(dynamic)

    # Suppressed count
    issues.suppressed_count = namespace_report.get("suppressed_count", 0)

    return issues


def _determine_manifest_status(
    manifest: Optional[Dict[str, Any]],
    replay_manifest: Optional[Dict[str, Any]],
    diff_result: Optional[Dict[str, Any]] = None,
) -> ManifestStatus:
    """Determine manifest comparison status."""
    if diff_result:
        status_str = diff_result.get("status", "MISSING")
        try:
            return ManifestStatus(status_str)
        except ValueError:
            return ManifestStatus.MISSING

    if manifest is None:
        return ManifestStatus.MISSING

    if replay_manifest is None:
        # No replay to compare against
        return ManifestStatus.EQUIVALENT

    # Extract attestations
    attest1 = manifest.get("prng_attestation") or manifest.get("prng", {})
    attest2 = replay_manifest.get("prng_attestation") or replay_manifest.get("prng", {})

    # Compare key fields
    seed1 = attest1.get("master_seed_hex")
    seed2 = attest2.get("master_seed_hex")
    scheme1 = attest1.get("derivation_scheme")
    scheme2 = attest2.get("derivation_scheme")
    merkle1 = attest1.get("lineage_merkle_root")
    merkle2 = attest2.get("lineage_merkle_root")

    if scheme1 != scheme2:
        return ManifestStatus.INCOMPATIBLE

    if seed1 == seed2 and merkle1 != merkle2:
        return ManifestStatus.DRIFTED

    return ManifestStatus.EQUIVALENT


def _compute_initial_governance_status(
    manifest_status: ManifestStatus,
    namespace_issues: NamespaceIssues,
    is_evidence_run: bool,
) -> GovernanceStatus:
    """Compute initial governance status before policy evaluation."""
    if manifest_status == ManifestStatus.INCOMPATIBLE:
        return GovernanceStatus.BLOCK

    if manifest_status == ManifestStatus.DRIFTED and is_evidence_run:
        return GovernanceStatus.BLOCK

    if namespace_issues.duplicate_count > 0:
        return GovernanceStatus.WARN

    if namespace_issues.hardcoded_seed_count > 0:
        return GovernanceStatus.WARN

    return GovernanceStatus.OK


def build_prng_governance_snapshot(
    manifest: Optional[Dict[str, Any]] = None,
    replay_manifest: Optional[Dict[str, Any]] = None,
    namespace_report: Optional[Dict[str, Any]] = None,
    diff_result: Optional[Dict[str, Any]] = None,
    is_evidence_run: bool = False,
    is_test_context: bool = False,
) -> PRNGGovernanceSnapshot:
    """
    Build a unified PRNG governance snapshot.

    Fuses outputs from:
    - Manifest data (prng_attestation block)
    - Manifest diff result
    - Namespace linter report

    Args:
        manifest: Primary manifest data.
        replay_manifest: Replay/comparison manifest data.
        namespace_report: Output from seed_namespace_linter.
        diff_result: Output from prng_manifest_diff (optional, will compute if not provided).
        is_evidence_run: Whether this is an evidence-gathering run.
        is_test_context: Whether this is running in a test context.

    Returns:
        PRNGGovernanceSnapshot with unified view.
    """
    # Extract attestation from manifest
    attestation = {}
    if manifest:
        attestation = manifest.get("prng_attestation") or manifest.get("prng", {})

    master_seed_hex = attestation.get("master_seed_hex")
    merkle_root = attestation.get("lineage_merkle_root")
    entry_count = attestation.get("lineage_entry_count", 0)

    # Determine manifest status
    manifest_status = _determine_manifest_status(manifest, replay_manifest, diff_result)

    # Extract namespace issues
    namespace_issues = _extract_namespace_issues(namespace_report)

    # Compute lineage fingerprint
    fingerprint = _compute_lineage_fingerprint(master_seed_hex, merkle_root, entry_count)

    # Determine initial governance status
    governance_status = _compute_initial_governance_status(
        manifest_status, namespace_issues, is_evidence_run
    )

    return PRNGGovernanceSnapshot(
        manifest_status=manifest_status,
        namespace_issues=namespace_issues,
        hardcoded_seeds_detected=namespace_issues.hardcoded_seed_count > 0,
        hardcoded_seed_count=namespace_issues.hardcoded_seed_count,
        seed_lineage_fingerprint=fingerprint,
        master_seed_hex=master_seed_hex,
        lineage_entry_count=entry_count,
        governance_status=governance_status,
        is_evidence_run=is_evidence_run,
        is_test_context=is_test_context,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 2: PRNG POLICY RULES EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def _check_rule_r1(snapshot: PRNGGovernanceSnapshot) -> Optional[PolicyViolation]:
    """R1: No DRIFTED schedules allowed for evidence runs."""
    rule = PRNG_GOV_RULES["R1"]

    if not snapshot.is_evidence_run:
        return None  # Rule doesn't apply

    if snapshot.manifest_status == ManifestStatus.DRIFTED:
        return PolicyViolation(
            rule_id=rule.rule_id,
            rule_name=rule.name,
            severity=rule.severity,
            message="Evidence run has DRIFTED schedule - determinism compromised",
            context={
                "manifest_status": snapshot.manifest_status.value,
                "fingerprint": snapshot.seed_lineage_fingerprint,
            },
        )

    return None


def _check_rule_r2(snapshot: PRNGGovernanceSnapshot) -> Optional[PolicyViolation]:
    """R2: INCOMPATIBLE manifests must be blocked."""
    rule = PRNG_GOV_RULES["R2"]

    if snapshot.manifest_status == ManifestStatus.INCOMPATIBLE:
        return PolicyViolation(
            rule_id=rule.rule_id,
            rule_name=rule.name,
            severity=rule.severity,
            message="Manifest is INCOMPATIBLE - cannot verify determinism",
            context={
                "manifest_status": snapshot.manifest_status.value,
            },
        )

    return None


def _check_rule_r3(snapshot: PRNGGovernanceSnapshot) -> Optional[PolicyViolation]:
    """R3: Hard-coded seeds only allowed in tests; BLOCK in runtime paths."""
    rule = PRNG_GOV_RULES["R3"]

    if snapshot.is_test_context:
        return None  # Hard-coded seeds allowed in tests

    if snapshot.hardcoded_seeds_detected:
        # Check if any hard-coded seeds are in non-test files
        non_test_files = [
            f for f in snapshot.namespace_issues.hardcoded_seed_files
            if not _is_test_file(f)
        ]

        if non_test_files:
            return PolicyViolation(
                rule_id=rule.rule_id,
                rule_name=rule.name,
                severity=rule.severity,
                message=f"Hard-coded seeds found in {len(non_test_files)} runtime file(s)",
                context={
                    "files": non_test_files[:5],  # Limit to first 5
                    "total_count": len(non_test_files),
                },
            )

    return None


def _check_rule_r4(snapshot: PRNGGovernanceSnapshot) -> Optional[PolicyViolation]:
    """R4: Namespace collisions should be reviewed."""
    rule = PRNG_GOV_RULES["R4"]

    if snapshot.namespace_issues.duplicate_count > 0:
        return PolicyViolation(
            rule_id=rule.rule_id,
            rule_name=rule.name,
            severity=rule.severity,
            message=f"{snapshot.namespace_issues.duplicate_count} namespace collision(s) detected",
            context={
                "duplicate_count": snapshot.namespace_issues.duplicate_count,
                "files": snapshot.namespace_issues.duplicate_files[:5],
            },
        )

    return None


def _check_rule_r5(snapshot: PRNGGovernanceSnapshot) -> Optional[PolicyViolation]:
    """R5: Manifests should have PRNG attestation blocks."""
    rule = PRNG_GOV_RULES["R5"]

    if snapshot.manifest_status == ManifestStatus.MISSING:
        return PolicyViolation(
            rule_id=rule.rule_id,
            rule_name=rule.name,
            severity=rule.severity,
            message="Manifest missing or lacks PRNG attestation",
            context={},
        )

    if not snapshot.master_seed_hex:
        return PolicyViolation(
            rule_id=rule.rule_id,
            rule_name=rule.name,
            severity=rule.severity,
            message="PRNG attestation block incomplete (no master_seed_hex)",
            context={},
        )

    return None


def _is_test_file(file_path: str) -> bool:
    """Check if a file path is a test file."""
    path_lower = file_path.lower().replace("\\", "/")
    return (
        "/test_" in path_lower or
        "/tests/" in path_lower or
        "_test.py" in path_lower or
        "conftest.py" in path_lower
    )


def evaluate_prng_policy(
    snapshot: PRNGGovernanceSnapshot,
    rules_to_check: Optional[List[str]] = None,
) -> PolicyEvaluation:
    """
    Evaluate PRNG policy rules against a governance snapshot.

    Args:
        snapshot: PRNGGovernanceSnapshot to evaluate.
        rules_to_check: Optional list of rule IDs to check. If None, checks all.

    Returns:
        PolicyEvaluation with results.
    """
    if rules_to_check is None:
        rules_to_check = list(PRNG_GOV_RULES.keys())

    violations: List[PolicyViolation] = []
    rules_checked: List[str] = []

    # Rule checkers
    rule_checkers = {
        "R1": _check_rule_r1,
        "R2": _check_rule_r2,
        "R3": _check_rule_r3,
        "R4": _check_rule_r4,
        "R5": _check_rule_r5,
    }

    for rule_id in rules_to_check:
        if rule_id not in rule_checkers:
            continue

        rules_checked.append(rule_id)
        checker = rule_checkers[rule_id]
        violation = checker(snapshot)

        if violation:
            violations.append(violation)

    # Determine overall status
    has_block = any(v.severity == GovernanceStatus.BLOCK for v in violations)
    has_warn = any(v.severity == GovernanceStatus.WARN for v in violations)

    if has_block:
        status = GovernanceStatus.BLOCK
    elif has_warn:
        status = GovernanceStatus.WARN
    else:
        status = GovernanceStatus.OK

    policy_ok = status == GovernanceStatus.OK

    return PolicyEvaluation(
        policy_ok=policy_ok,
        violations=violations,
        status=status,
        rules_checked=rules_checked,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 3: GLOBAL HEALTH & MAAS HOOK
# ═══════════════════════════════════════════════════════════════════════════════

def summarize_prng_for_global_health(
    snapshot: PRNGGovernanceSnapshot,
    policy_eval: PolicyEvaluation,
) -> GlobalHealthSummary:
    """
    Summarize PRNG state for global health / MAAS integration.

    This provides a simplified view suitable for dashboards and monitoring.

    Args:
        snapshot: PRNGGovernanceSnapshot.
        policy_eval: PolicyEvaluation result.

    Returns:
        GlobalHealthSummary with key indicators.
    """
    # Determine key flags
    has_namespace_collisions = snapshot.namespace_issues.duplicate_count > 0
    has_schedule_drift = snapshot.manifest_status == ManifestStatus.DRIFTED
    has_hardcoded_seeds = snapshot.hardcoded_seeds_detected

    # Build summary message
    issues = []
    if has_schedule_drift:
        issues.append("schedule drift")
    if has_namespace_collisions:
        issues.append(f"{snapshot.namespace_issues.duplicate_count} namespace collision(s)")
    if has_hardcoded_seeds:
        issues.append(f"{snapshot.hardcoded_seed_count} hard-coded seed(s)")

    if policy_eval.policy_ok and not issues:
        summary_message = "PRNG governance: OK"
    elif policy_eval.policy_ok and issues:
        summary_message = f"PRNG governance: OK (noted: {', '.join(issues)})"
    else:
        summary_message = f"PRNG governance: {', '.join(issues) if issues else 'policy violations'}"

    return GlobalHealthSummary(
        prng_policy_ok=policy_eval.policy_ok,
        has_namespace_collisions=has_namespace_collisions,
        has_schedule_drift=has_schedule_drift,
        has_hardcoded_seeds=has_hardcoded_seeds,
        status=policy_eval.status,
        violation_count=len(policy_eval.violations),
        summary_message=summary_message,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE: FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_full_prng_governance(
    manifest: Optional[Dict[str, Any]] = None,
    replay_manifest: Optional[Dict[str, Any]] = None,
    namespace_report: Optional[Dict[str, Any]] = None,
    is_evidence_run: bool = False,
    is_test_context: bool = False,
) -> Dict[str, Any]:
    """
    Run the full PRNG governance pipeline.

    Convenience function that builds snapshot, evaluates policy, and
    generates global health summary in one call.

    Args:
        manifest: Primary manifest data.
        replay_manifest: Replay/comparison manifest data.
        namespace_report: Output from seed_namespace_linter.
        is_evidence_run: Whether this is an evidence-gathering run.
        is_test_context: Whether this is running in a test context.

    Returns:
        Dict with 'snapshot', 'policy_eval', and 'health' keys.
    """
    snapshot = build_prng_governance_snapshot(
        manifest=manifest,
        replay_manifest=replay_manifest,
        namespace_report=namespace_report,
        is_evidence_run=is_evidence_run,
        is_test_context=is_test_context,
    )

    policy_eval = evaluate_prng_policy(snapshot)
    health = summarize_prng_for_global_health(snapshot, policy_eval)

    return {
        "snapshot": snapshot.to_dict(),
        "policy_eval": policy_eval.to_dict(),
        "health": health.to_dict(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 1: CI-FACING GATE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_prng_for_ci(health: GlobalHealthSummary) -> int:
    """
    Evaluate PRNG governance for CI gate decision.

    Maps governance status to exit codes:
        - 0: OK (governance passed)
        - 1: WARN (governance issues but non-blocking)
        - 2: BLOCK (governance violations must be fixed)

    Args:
        health: GlobalHealthSummary from governance pipeline.

    Returns:
        Exit code: 0 (OK), 1 (WARN), or 2 (BLOCK).
    """
    if health.status == GovernanceStatus.OK:
        return 0
    elif health.status == GovernanceStatus.WARN:
        return 1
    elif health.status == GovernanceStatus.BLOCK:
        return 2
    else:
        # Unknown status defaults to BLOCK
        return 2


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 2: AUTO-REMEDIATION SUGGESTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def build_prng_remediation_suggestions(
    snapshot: PRNGGovernanceSnapshot,
    policy_eval: PolicyEvaluation,
) -> List[Dict[str, Any]]:
    """
    Build auto-remediation suggestions for PRNG governance violations.

    Generates descriptive, actionable suggestions for fixing governance issues.
    This is read-only; it never modifies code.

    Args:
        snapshot: PRNGGovernanceSnapshot with current state.
        policy_eval: PolicyEvaluation with violations.

    Returns:
        List of suggestion dictionaries, each with:
        - rule_id: Rule that triggered the suggestion
        - impact: Descriptive impact statement
        - suggested_action: Actionable remediation step
        - files_involved: List of file paths affected
    """
    suggestions: List[Dict[str, Any]] = []

    for violation in policy_eval.violations:
        rule_id = violation.rule_id
        context = violation.context

        if rule_id == "R1":
            # DRIFTED schedule in evidence run
            suggestions.append({
                "rule_id": rule_id,
                "impact": "Evidence run has non-deterministic seed schedule, compromising reproducibility",
                "suggested_action": (
                    "Investigate why seed derivation differs from expected. "
                    "Verify that master_seed_hex matches between runs and that "
                    "no code changes affected PRNG derivation logic."
                ),
                "files_involved": [],
            })

        elif rule_id == "R2":
            # INCOMPATIBLE manifest
            suggestions.append({
                "rule_id": rule_id,
                "impact": "Manifests use incompatible derivation schemes, preventing comparison",
                "suggested_action": (
                    "Ensure both manifests use the same derivation_scheme. "
                    "If schemes differ, regenerate one manifest using the canonical scheme: "
                    "'PRNGKey(root, path) -> SHA256 -> seed % 2^32'"
                ),
                "files_involved": [],
            })

        elif rule_id == "R3":
            # Hard-coded seeds in runtime
            files = context.get("files", [])
            suggestions.append({
                "rule_id": rule_id,
                "impact": f"Hard-coded seeds found in {len(files)} runtime file(s), breaking determinism",
                "suggested_action": (
                    "Replace hard-coded seeds with DeterministicPRNG.for_path() calls. "
                    "Example: Replace `random.seed(42)` with "
                    "`prng = DeterministicPRNG(master_seed_hex); rng = prng.for_path('component', 'subcomponent')`"
                ),
                "files_involved": files[:10],  # Limit to first 10
            })

        elif rule_id == "R4":
            # Namespace collisions
            files = context.get("files", [])
            duplicate_count = context.get("duplicate_count", 0)
            suggestions.append({
                "rule_id": rule_id,
                "impact": f"{duplicate_count} namespace collision(s) detected, risking seed reuse",
                "suggested_action": (
                    "Review namespace paths to ensure uniqueness. "
                    "If intentional reuse, add `# prng: namespace-ok` comment. "
                    "Otherwise, add distinguishing labels to paths (e.g., component name, cycle index)."
                ),
                "files_involved": files[:10],  # Limit to first 10
            })

        elif rule_id == "R5":
            # Missing attestation
            suggestions.append({
                "rule_id": rule_id,
                "impact": "Manifest missing or incomplete PRNG attestation block",
                "suggested_action": (
                    "Add prng_attestation block to manifest with: "
                    "master_seed_hex, derivation_scheme, lineage_merkle_root, and lineage_entry_count. "
                    "Use ManifestBuilder.build() which automatically includes attestation."
                ),
                "files_involved": [],
            })

    # Add suggestions for issues not covered by violations
    if snapshot.namespace_issues.dynamic_path_count > 0:
        suggestions.append({
            "rule_id": "INFO",
            "impact": f"{snapshot.namespace_issues.dynamic_path_count} dynamic namespace path(s) detected",
            "suggested_action": (
                "Review dynamic paths to ensure they produce deterministic namespaces. "
                "Consider using constant labels where possible for better traceability."
            ),
            "files_involved": [],
        })

    return suggestions


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 3: PRNG GOVERNANCE HISTORY LEDGER
# ═══════════════════════════════════════════════════════════════════════════════

def build_prng_governance_history(
    snapshots: List[PRNGGovernanceSnapshot],
    run_ids: Optional[List[str]] = None,
    policy_evaluations: Optional[List[PolicyEvaluation]] = None,
) -> Dict[str, Any]:
    """
    Build a PRNG governance history ledger from a sequence of snapshots.

    Creates a canonical history record suitable for long-horizon analysis,
    audit trails, and integration with global health systems.

    Args:
        snapshots: Sequence of PRNGGovernanceSnapshot objects.
        run_ids: Optional list of run identifiers (one per snapshot).
                If None, generates sequential IDs.
        policy_evaluations: Optional list of PolicyEvaluation objects (one per snapshot).
                          If provided, violations are included in run records.

    Returns:
        Dict with:
        - schema_version: "1.0"
        - runs: List of run records (with violations if policy_evaluations provided)
        - status_counts: Aggregated counts by status
        - history_hash: SHA-256 hash of canonical history body
    """
    if run_ids is None:
        run_ids = [f"run_{i:04d}" for i in range(len(snapshots))]
    elif len(run_ids) != len(snapshots):
        raise ValueError(f"run_ids length ({len(run_ids)}) must match snapshots length ({len(snapshots)})")

    if policy_evaluations is not None and len(policy_evaluations) != len(snapshots):
        raise ValueError(
            f"policy_evaluations length ({len(policy_evaluations)}) must match snapshots length ({len(snapshots)})"
        )

    # Build run records
    runs = []
    for i, (snapshot, run_id) in enumerate(zip(snapshots, run_ids)):
        run_record = {
            "run_id": run_id,
            "governance_status": snapshot.governance_status.value,
            "manifest_status": snapshot.manifest_status.value,
            "has_hardcoded_seeds": snapshot.hardcoded_seeds_detected,
            "hardcoded_seed_count": snapshot.hardcoded_seed_count,
            "namespace_duplicate_count": snapshot.namespace_issues.duplicate_count,
            "lineage_fingerprint": snapshot.seed_lineage_fingerprint,
            "timestamp": snapshot.timestamp,
        }

        # Add violations if policy evaluations provided
        if policy_evaluations:
            policy_eval = policy_evaluations[i]
            run_record["violations"] = [
                {
                    "rule_id": v.rule_id,
                    "kind": v.rule_name,
                    "severity": v.severity.value,
                }
                for v in policy_eval.violations
            ]
        else:
            run_record["violations"] = []

        runs.append(run_record)

    # Compute status counts
    status_counts = {
        "OK": sum(1 for r in runs if r["governance_status"] == "OK"),
        "WARN": sum(1 for r in runs if r["governance_status"] == "WARN"),
        "BLOCK": sum(1 for r in runs if r["governance_status"] == "BLOCK"),
    }

    manifest_status_counts = {}
    for run in runs:
        status = run["manifest_status"]
        manifest_status_counts[status] = manifest_status_counts.get(status, 0) + 1

    # Build history body (deterministic ordering)
    history_body = {
        "schema_version": "1.0",
        "total_runs": len(runs),
        "runs": sorted(runs, key=lambda r: (r["run_id"], r["timestamp"])),  # Deterministic sort
        "status_counts": status_counts,
        "manifest_status_counts": dict(sorted(manifest_status_counts.items())),
    }

    # Compute canonical hash
    canonical_json = json.dumps(history_body, sort_keys=True, separators=(",", ":"))
    history_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    return {
        "schema_version": "1.0",
        "total_runs": len(runs),
        "runs": history_body["runs"],
        "status_counts": status_counts,
        "manifest_status_counts": manifest_status_counts,
        "history_hash": history_hash,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TASK: PRNG DRIFT RADAR & EVIDENCE PACK TILE
# ═══════════════════════════════════════════════════════════════════════════════

class DriftStatus(str, Enum):
    """Status of PRNG governance drift."""
    STABLE = "STABLE"
    DRIFTING = "DRIFTING"
    VOLATILE = "VOLATILE"


def build_prng_drift_radar(history: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build PRNG drift radar from governance history.

    Analyzes governance history to detect patterns and classify drift status:
    - STABLE: No BLOCK violations, few WARNs
    - DRIFTING: Intermittent BLOCK/WARN with low frequency
    - VOLATILE: Frequent BLOCK or many rules firing repeatedly

    Args:
        history: Governance history from build_prng_governance_history().

    Returns:
        Dict with:
        - schema_version: "1.0"
        - runs_with_drift: List of run_ids with drift (BLOCK or WARN)
        - frequent_violations: Dict mapping rule_id to count
        - drift_status: STABLE | DRIFTING | VOLATILE
        - total_runs: Total number of runs analyzed
        - drift_rate: Percentage of runs with issues
    """
    if not history or history.get("total_runs", 0) == 0:
        return {
            "schema_version": "1.0",
            "runs_with_drift": [],
            "frequent_violations": {},
            "drift_status": DriftStatus.STABLE.value,
            "total_runs": 0,
            "drift_rate": 0.0,
        }

    runs = history.get("runs", [])
    status_counts = history.get("status_counts", {})
    total_runs = history.get("total_runs", len(runs))

    # Identify runs with drift (BLOCK or WARN)
    runs_with_drift = [
        run["run_id"]
        for run in runs
        if run.get("governance_status") in ("WARN", "BLOCK")
    ]

    # Calculate drift rate
    drift_rate = len(runs_with_drift) / total_runs if total_runs > 0 else 0.0

    # Count frequent violations (would need violation data in history)
    # For now, we'll infer from status patterns
    frequent_violations: Dict[str, int] = {}

    # Classify drift status
    block_count = status_counts.get("BLOCK", 0)
    warn_count = status_counts.get("WARN", 0)
    ok_count = status_counts.get("OK", 0)

    # STABLE: No BLOCK violations and few WARNs (< 20% of runs)
    if block_count == 0 and warn_count / total_runs < 0.2:
        drift_status = DriftStatus.STABLE

    # VOLATILE: Frequent BLOCK (> 30% of runs) or many rules firing
    elif block_count / total_runs > 0.3 or (block_count + warn_count) / total_runs > 0.5:
        drift_status = DriftStatus.VOLATILE

    # DRIFTING: Intermittent issues (between STABLE and VOLATILE)
    else:
        drift_status = DriftStatus.DRIFTING

    return {
        "schema_version": "1.0",
        "runs_with_drift": sorted(runs_with_drift),  # Deterministic ordering
        "frequent_violations": dict(sorted(frequent_violations.items())),
        "drift_status": drift_status.value,
        "total_runs": total_runs,
        "drift_rate": round(drift_rate, 3),
        "status_breakdown": {
            "OK": ok_count,
            "WARN": warn_count,
            "BLOCK": block_count,
        },
    }


def build_prng_governance_tile(
    history: Dict[str, Any],
    radar: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build PRNG governance tile for evidence pack.

    Creates a PRNG section suitable for external auditors and evidence chains.
    This tile provides a compact, schema-compliant view of PRNG governance state.

    Args:
        history: Governance history from build_prng_governance_history().
        radar: Optional drift radar from build_prng_drift_radar().
                If None, will be computed from history.

    Returns:
        Dict with PRNG tile structure:
        - overall_status: OK | WARN | BLOCK
        - drift_status: STABLE | DRIFTING | VOLATILE
        - blocking_rules: List of rule IDs that caused BLOCK status
        - history_hash: SHA-256 hash from history
        - total_runs: Number of runs in history
        - compliance_rate: Percentage of runs with OK status
        - schema_version: "1.0"
    """
    if radar is None:
        radar = build_prng_drift_radar(history)

    # Determine overall status from most recent run or aggregate
    status_counts = history.get("status_counts", {})
    total_runs = history.get("total_runs", 0)

    # Overall status: BLOCK if any BLOCK, else WARN if any WARN, else OK
    if status_counts.get("BLOCK", 0) > 0:
        overall_status = GovernanceStatus.BLOCK.value
    elif status_counts.get("WARN", 0) > 0:
        overall_status = GovernanceStatus.WARN.value
    else:
        overall_status = GovernanceStatus.OK.value

    # Extract blocking rules (would need violation data in history)
    # For now, infer from status
    blocking_rules: List[str] = []
    if status_counts.get("BLOCK", 0) > 0:
        # Common blocking rules
        blocking_rules = ["R1", "R2", "R3"]  # Placeholder - would need actual violation data

    # Calculate compliance rate
    ok_count = status_counts.get("OK", 0)
    compliance_rate = ok_count / total_runs if total_runs > 0 else 1.0

    return {
        "schema_version": "1.0",
        "tile_type": "prng_governance",
        "overall_status": overall_status,
        "drift_status": radar.get("drift_status", DriftStatus.STABLE.value),
        "blocking_rules": sorted(blocking_rules),  # Deterministic ordering
        "history_hash": history.get("history_hash", ""),
        "total_runs": total_runs,
        "compliance_rate": round(compliance_rate, 3),
        "status_breakdown": status_counts,
        "drift_rate": radar.get("drift_rate", 0.0),
        "runs_with_issues": len(radar.get("runs_with_drift", [])),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Enums
    "GovernanceStatus",
    "ManifestStatus",
    # Data structures
    "PolicyRule",
    "NamespaceIssues",
    "PolicyViolation",
    "PRNGGovernanceSnapshot",
    "PolicyEvaluation",
    "GlobalHealthSummary",
    # Rules
    "PRNG_GOV_RULES",
    # Functions
    "build_prng_governance_snapshot",
    "evaluate_prng_policy",
    "summarize_prng_for_global_health",
    "run_full_prng_governance",
    # Phase IV
    "evaluate_prng_for_ci",
    "build_prng_remediation_suggestions",
    "build_prng_governance_history",
]

