#!/usr/bin/env python3
"""
PR Migration Linter

Detects when a PR implicitly constitutes a phase migration and enforces
the requirement for a migration_intent.yaml declaration file.

Author: Agent E4 (doc-ops-4) — Phase Migration Architect
Date: 2025-12-06

ABSOLUTE SAFEGUARDS:
- Read-only analysis
- No mutations to production state
- No uplift claims
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class MigrationSignal(Enum):
    """Types of implicit migration signals detected in PRs."""
    
    # Phase I → Phase II signals
    DB_WRITE_INTRODUCTION = "db_write_introduction"
    RFL_DB_ENABLE = "rfl_db_enable"
    PREREG_ADDITION = "prereg_addition"
    UPLIFT_SLICE_ADDITION = "uplift_slice_addition"
    
    # Phase II → Phase IIb signals
    LEAN_ENABLE = "lean_enable"
    LEAN_TIMEOUT_CONFIG = "lean_timeout_config"
    LEAN_TOOLCHAIN_UPDATE = "lean_toolchain_update"
    
    # Phase II → Phase III signals
    BASIS_IMPORT_ACTIVATION = "basis_import_activation"
    ED25519_INTRODUCTION = "ed25519_introduction"
    RFC8785_ACTIVATION = "rfc8785_activation"
    PROOF_MIDDLEWARE_ADDITION = "proof_middleware_addition"
    
    # General signals
    CURRICULUM_ACTIVE_CHANGE = "curriculum_active_change"
    SECURITY_MODEL_CHANGE = "security_model_change"
    DETERMINISM_ENVELOPE_CHANGE = "determinism_envelope_change"


@dataclass
class MigrationSignalMatch:
    """A detected migration signal in the PR diff."""
    signal: MigrationSignal
    file_path: str
    line_number: int | None
    context: str
    severity: str  # "critical" | "warning" | "info"
    implied_migration: str  # e.g., "Phase I → Phase II"


@dataclass
class MigrationIntent:
    """Parsed migration intent declaration."""
    declared: bool
    source_phase: str | None
    target_phase: str | None
    justification: str | None
    preconditions_verified: list[str]
    rollback_plan: str | None
    approvers: list[str]
    
    @classmethod
    def from_yaml(cls, yaml_content: dict) -> "MigrationIntent":
        """Parse from YAML content."""
        return cls(
            declared=True,
            source_phase=yaml_content.get("source_phase"),
            target_phase=yaml_content.get("target_phase"),
            justification=yaml_content.get("justification"),
            preconditions_verified=yaml_content.get("preconditions_verified", []),
            rollback_plan=yaml_content.get("rollback_plan"),
            approvers=yaml_content.get("approvers", []),
        )
    
    @classmethod
    def not_declared(cls) -> "MigrationIntent":
        """Return an empty intent indicating no declaration."""
        return cls(
            declared=False,
            source_phase=None,
            target_phase=None,
            justification=None,
            preconditions_verified=[],
            rollback_plan=None,
            approvers=[],
        )


@dataclass
class PRMigrationLintResult:
    """Result of PR migration linting."""
    has_migration_signals: bool
    signals: list[MigrationSignalMatch]
    intent: MigrationIntent
    verdict: str  # "PASS" | "FAIL" | "WARN"
    errors: list[str]
    warnings: list[str]
    recommendations: list[str]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "has_migration_signals": self.has_migration_signals,
            "signals": [
                {
                    "signal": s.signal.value,
                    "file_path": s.file_path,
                    "line_number": s.line_number,
                    "context": s.context[:200],
                    "severity": s.severity,
                    "implied_migration": s.implied_migration,
                }
                for s in self.signals
            ],
            "intent": {
                "declared": self.intent.declared,
                "source_phase": self.intent.source_phase,
                "target_phase": self.intent.target_phase,
                "justification": self.intent.justification,
                "preconditions_verified": self.intent.preconditions_verified,
                "rollback_plan": self.intent.rollback_plan,
                "approvers": self.intent.approvers,
            },
            "verdict": self.verdict,
            "errors": self.errors,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
        }


class PRMigrationLinter:
    """
    Linter for detecting implicit phase migrations in PRs.
    
    Analyzes git diff output to detect changes that constitute
    phase boundary crossings and validates migration declarations.
    """
    
    # Patterns for detecting migration signals
    SIGNAL_PATTERNS: dict[MigrationSignal, list[dict[str, Any]]] = {
        # Phase I → Phase II signals
        MigrationSignal.DB_WRITE_INTRODUCTION: [
            {"pattern": r"\+.*session\.commit\(\)", "file_glob": "*.py"},
            {"pattern": r"\+.*session\.add\(", "file_glob": "*.py"},
            {"pattern": r"\+.*db\.add\(", "file_glob": "*.py"},
        ],
        MigrationSignal.RFL_DB_ENABLE: [
            {"pattern": r"\+.*RFL_DB_ENABLED\s*=\s*True", "file_glob": "*.py"},
            {"pattern": r"\+.*rfl_db_enabled:\s*true", "file_glob": "*.yaml"},
        ],
        MigrationSignal.PREREG_ADDITION: [
            {"pattern": r".*", "file_glob": "**/prereg/*.yaml"},
            {"pattern": r".*", "file_glob": "**/PREREG_*.yaml"},
        ],
        MigrationSignal.UPLIFT_SLICE_ADDITION: [
            {"pattern": r"\+.*slice_uplift", "file_glob": "curriculum*.yaml"},
            {"pattern": r"\+.*name:\s*slice_uplift", "file_glob": "*.yaml"},
        ],
        
        # Phase II → Phase IIb signals
        MigrationSignal.LEAN_ENABLE: [
            {"pattern": r"\+.*lean_enabled:\s*true", "file_glob": "*.yaml"},
            {"pattern": r"\+.*LEAN_ENABLED\s*=\s*True", "file_glob": "*.py"},
        ],
        MigrationSignal.LEAN_TIMEOUT_CONFIG: [
            {"pattern": r"\+.*lean_timeout_ms:", "file_glob": "*.yaml"},
            {"pattern": r"\+.*lean_timeout_s:", "file_glob": "*.yaml"},
        ],
        MigrationSignal.LEAN_TOOLCHAIN_UPDATE: [
            {"pattern": r".*", "file_glob": "**/lean_proj/**"},
            {"pattern": r".*", "file_glob": "**/lakefile.lean"},
        ],
        
        # Phase II → Phase III signals
        MigrationSignal.BASIS_IMPORT_ACTIVATION: [
            {"pattern": r"\+.*from basis\.", "file_glob": "backend/**/*.py"},
            {"pattern": r"\+.*import basis", "file_glob": "backend/**/*.py"},
        ],
        MigrationSignal.ED25519_INTRODUCTION: [
            {"pattern": r"\+.*ed25519", "file_glob": "*.py"},
            {"pattern": r"\+.*Ed25519", "file_glob": "*.py"},
        ],
        MigrationSignal.RFC8785_ACTIVATION: [
            {"pattern": r"\+.*rfc8785", "file_glob": "*.py"},
            {"pattern": r"\+.*RFC8785", "file_glob": "*.py"},
            {"pattern": r"\+.*canonicalize", "file_glob": "**/crypto/*.py"},
        ],
        MigrationSignal.PROOF_MIDDLEWARE_ADDITION: [
            {"pattern": r"\+.*ProofMiddleware", "file_glob": "*.py"},
            {"pattern": r"\+.*proof_middleware", "file_glob": "*.py"},
        ],
        
        # General signals
        MigrationSignal.CURRICULUM_ACTIVE_CHANGE: [
            {"pattern": r"\+\s*active:", "file_glob": "curriculum*.yaml"},
            {"pattern": r"-\s*active:", "file_glob": "curriculum*.yaml"},
        ],
        MigrationSignal.SECURITY_MODEL_CHANGE: [
            {"pattern": r"\+.*require_signatures:", "file_glob": "*.yaml"},
            {"pattern": r"\+.*FIRST_ORGANISM_STRICT", "file_glob": "*.py"},
        ],
        MigrationSignal.DETERMINISM_ENVELOPE_CHANGE: [
            {"pattern": r".*", "file_glob": "**/repro/determinism.py"},
            {"pattern": r".*", "file_glob": "normalization/canon.py"},
            {"pattern": r".*", "file_glob": "attestation/dual_root.py"},
        ],
    }
    
    # Mapping signals to implied migrations
    SIGNAL_MIGRATION_MAP: dict[MigrationSignal, str] = {
        MigrationSignal.DB_WRITE_INTRODUCTION: "Phase I → Phase II",
        MigrationSignal.RFL_DB_ENABLE: "Phase I → Phase II",
        MigrationSignal.PREREG_ADDITION: "Phase I → Phase II",
        MigrationSignal.UPLIFT_SLICE_ADDITION: "Phase I → Phase II",
        MigrationSignal.LEAN_ENABLE: "Phase II → Phase IIb",
        MigrationSignal.LEAN_TIMEOUT_CONFIG: "Phase II → Phase IIb",
        MigrationSignal.LEAN_TOOLCHAIN_UPDATE: "Phase II → Phase IIb",
        MigrationSignal.BASIS_IMPORT_ACTIVATION: "Phase II → Phase III",
        MigrationSignal.ED25519_INTRODUCTION: "Phase II → Phase III",
        MigrationSignal.RFC8785_ACTIVATION: "Phase II → Phase III",
        MigrationSignal.PROOF_MIDDLEWARE_ADDITION: "Phase II → Phase III",
        MigrationSignal.CURRICULUM_ACTIVE_CHANGE: "Cross-phase (requires analysis)",
        MigrationSignal.SECURITY_MODEL_CHANGE: "Security elevation",
        MigrationSignal.DETERMINISM_ENVELOPE_CHANGE: "Determinism contract change",
    }
    
    # Signal severities
    SIGNAL_SEVERITY: dict[MigrationSignal, str] = {
        MigrationSignal.DB_WRITE_INTRODUCTION: "critical",
        MigrationSignal.RFL_DB_ENABLE: "critical",
        MigrationSignal.LEAN_ENABLE: "critical",
        MigrationSignal.BASIS_IMPORT_ACTIVATION: "critical",
        MigrationSignal.CURRICULUM_ACTIVE_CHANGE: "critical",
        MigrationSignal.DETERMINISM_ENVELOPE_CHANGE: "critical",
        MigrationSignal.PREREG_ADDITION: "warning",
        MigrationSignal.UPLIFT_SLICE_ADDITION: "warning",
        MigrationSignal.LEAN_TIMEOUT_CONFIG: "warning",
        MigrationSignal.LEAN_TOOLCHAIN_UPDATE: "warning",
        MigrationSignal.ED25519_INTRODUCTION: "warning",
        MigrationSignal.RFC8785_ACTIVATION: "warning",
        MigrationSignal.PROOF_MIDDLEWARE_ADDITION: "warning",
        MigrationSignal.SECURITY_MODEL_CHANGE: "info",
    }
    
    def __init__(self, project_root: Path | None = None):
        self.project_root = project_root or PROJECT_ROOT
    
    def get_diff(self, base_ref: str = "main", head_ref: str = "HEAD") -> str:
        """Get git diff between two refs."""
        try:
            result = subprocess.run(
                ["git", "diff", f"{base_ref}...{head_ref}", "--unified=3"],
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )
            return result.stdout
        except Exception as e:
            return ""
    
    def get_changed_files(self, base_ref: str = "main", head_ref: str = "HEAD") -> list[str]:
        """Get list of changed files between two refs."""
        try:
            result = subprocess.run(
                ["git", "diff", f"{base_ref}...{head_ref}", "--name-only"],
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )
            return [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
        except Exception:
            return []
    
    def _match_glob(self, file_path: str, glob_pattern: str) -> bool:
        """Check if a file path matches a glob pattern (simplified)."""
        import fnmatch
        
        # Handle ** patterns
        if "**" in glob_pattern:
            # Convert ** to regex-compatible pattern
            regex_pattern = glob_pattern.replace("**", ".*").replace("*", "[^/]*")
            return bool(re.match(regex_pattern, file_path))
        
        return fnmatch.fnmatch(file_path, glob_pattern)
    
    def detect_signals_from_diff(self, diff_content: str) -> list[MigrationSignalMatch]:
        """Detect migration signals from diff content."""
        signals = []
        
        current_file = None
        current_line = 0
        
        for line in diff_content.split("\n"):
            # Track current file
            if line.startswith("diff --git"):
                match = re.search(r"b/(.+)$", line)
                if match:
                    current_file = match.group(1)
                    current_line = 0
            
            # Track line numbers
            if line.startswith("@@"):
                match = re.search(r"\+(\d+)", line)
                if match:
                    current_line = int(match.group(1))
            elif current_line > 0:
                current_line += 1
            
            # Check for signals
            for signal, patterns in self.SIGNAL_PATTERNS.items():
                for pattern_def in patterns:
                    pattern = pattern_def["pattern"]
                    file_glob = pattern_def["file_glob"]
                    
                    # Check file match
                    if current_file and not self._match_glob(current_file, file_glob):
                        continue
                    
                    # Check pattern match
                    if re.search(pattern, line):
                        signals.append(MigrationSignalMatch(
                            signal=signal,
                            file_path=current_file or "unknown",
                            line_number=current_line if current_line > 0 else None,
                            context=line[:200],
                            severity=self.SIGNAL_SEVERITY.get(signal, "info"),
                            implied_migration=self.SIGNAL_MIGRATION_MAP.get(signal, "Unknown"),
                        ))
        
        # Deduplicate by (signal, file_path)
        seen = set()
        unique_signals = []
        for sig in signals:
            key = (sig.signal, sig.file_path)
            if key not in seen:
                seen.add(key)
                unique_signals.append(sig)
        
        return unique_signals
    
    def detect_signals_from_files(self, changed_files: list[str]) -> list[MigrationSignalMatch]:
        """Detect migration signals from file paths alone."""
        signals = []
        
        for file_path in changed_files:
            # Check for prereg addition
            if "prereg" in file_path.lower() and file_path.endswith(".yaml"):
                signals.append(MigrationSignalMatch(
                    signal=MigrationSignal.PREREG_ADDITION,
                    file_path=file_path,
                    line_number=None,
                    context=f"Preregistration file: {file_path}",
                    severity="warning",
                    implied_migration="Phase I → Phase II",
                ))
            
            # Check for lean toolchain changes
            if "lean_proj" in file_path or file_path.endswith("lakefile.lean"):
                signals.append(MigrationSignalMatch(
                    signal=MigrationSignal.LEAN_TOOLCHAIN_UPDATE,
                    file_path=file_path,
                    line_number=None,
                    context=f"Lean toolchain file: {file_path}",
                    severity="warning",
                    implied_migration="Phase II → Phase IIb",
                ))
            
            # Check for determinism envelope changes
            envelope_files = [
                "normalization/canon.py",
                "attestation/dual_root.py",
                "repro/determinism.py",
                "basis/logic/normalizer.py",
                "basis/crypto/hash.py",
            ]
            for ef in envelope_files:
                if file_path.endswith(ef) or ef in file_path:
                    signals.append(MigrationSignalMatch(
                        signal=MigrationSignal.DETERMINISM_ENVELOPE_CHANGE,
                        file_path=file_path,
                        line_number=None,
                        context=f"Determinism envelope file: {file_path}",
                        severity="critical",
                        implied_migration="Determinism contract change",
                    ))
                    break
        
        return signals
    
    def load_migration_intent(self) -> MigrationIntent:
        """Load migration intent declaration from migration_intent.yaml."""
        intent_paths = [
            self.project_root / "migration_intent.yaml",
            self.project_root / ".migration_intent.yaml",
        ]
        
        for path in intent_paths:
            if path.exists():
                try:
                    import yaml
                    with open(path) as f:
                        content = yaml.safe_load(f)
                    return MigrationIntent.from_yaml(content or {})
                except Exception:
                    pass
        
        return MigrationIntent.not_declared()
    
    def lint(
        self,
        base_ref: str = "main",
        head_ref: str = "HEAD",
        diff_content: str | None = None,
    ) -> PRMigrationLintResult:
        """
        Run the PR migration linter.
        
        Args:
            base_ref: Git ref for base branch
            head_ref: Git ref for PR head
            diff_content: Pre-computed diff content (optional)
        
        Returns:
            PRMigrationLintResult with verdict and details
        """
        errors = []
        warnings = []
        recommendations = []
        
        # Get diff if not provided
        if diff_content is None:
            diff_content = self.get_diff(base_ref, head_ref)
        
        # Get changed files
        changed_files = self.get_changed_files(base_ref, head_ref)
        
        # Detect signals
        signals_from_diff = self.detect_signals_from_diff(diff_content)
        signals_from_files = self.detect_signals_from_files(changed_files)
        
        # Merge and deduplicate signals
        all_signals = signals_from_diff + signals_from_files
        seen = set()
        unique_signals = []
        for sig in all_signals:
            key = (sig.signal, sig.file_path)
            if key not in seen:
                seen.add(key)
                unique_signals.append(sig)
        
        has_migration_signals = len(unique_signals) > 0
        has_critical_signals = any(s.severity == "critical" for s in unique_signals)
        
        # Load migration intent
        intent = self.load_migration_intent()
        
        # Determine verdict
        if has_critical_signals and not intent.declared:
            verdict = "FAIL"
            errors.append(
                "Critical migration signals detected but no migration_intent.yaml found. "
                "Create a migration_intent.yaml file declaring this phase migration."
            )
        elif has_migration_signals and not intent.declared:
            verdict = "WARN"
            warnings.append(
                "Migration signals detected. Consider adding a migration_intent.yaml "
                "if this PR constitutes a phase migration."
            )
        elif intent.declared:
            # Validate intent
            if not intent.source_phase or not intent.target_phase:
                verdict = "WARN"
                warnings.append(
                    "migration_intent.yaml found but missing source_phase or target_phase."
                )
            elif not intent.justification:
                verdict = "WARN"
                warnings.append(
                    "migration_intent.yaml found but missing justification."
                )
            elif not intent.rollback_plan:
                verdict = "WARN"
                warnings.append(
                    "migration_intent.yaml found but missing rollback_plan."
                )
            else:
                verdict = "PASS"
        else:
            verdict = "PASS"
        
        # Add recommendations
        if has_migration_signals:
            implied_migrations = set(s.implied_migration for s in unique_signals)
            for migration in implied_migrations:
                recommendations.append(
                    f"Run migration simulator for: {migration}"
                )
        
        if MigrationSignal.DETERMINISM_ENVELOPE_CHANGE in [s.signal for s in unique_signals]:
            recommendations.append(
                "Run determinism tests: pytest tests/integration/test_first_organism_determinism.py"
            )
        
        return PRMigrationLintResult(
            has_migration_signals=has_migration_signals,
            signals=unique_signals,
            intent=intent,
            verdict=verdict,
            errors=errors,
            warnings=warnings,
            recommendations=recommendations,
        )


def generate_migration_intent_template() -> str:
    """Generate a migration_intent.yaml template."""
    return """# Migration Intent Declaration
# Required when a PR constitutes a phase migration

# Source and target phases
source_phase: phase_i  # phase_i | phase_ii | phase_iib | phase_iii
target_phase: phase_ii  # phase_i | phase_ii | phase_iib | phase_iii

# Justification for this migration
justification: |
  Describe why this migration is necessary and what it accomplishes.

# Preconditions that have been verified
preconditions_verified:
  - "Evidence sealed: attestation.json exists with valid H_t"
  - "Determinism test passes"
  - "1000+ cycle baseline exists"

# Rollback plan if migration causes issues
rollback_plan: |
  Describe how to revert this migration if problems occur.

# Required approvers (GitHub usernames)
approvers:
  - maintainer1
  - maintainer2

# Additional metadata
metadata:
  migration_simulator_run: false  # Set to true after running simulator
  simulation_id: null  # ID from migration_sim_result.json
  pr_number: null  # Will be filled by CI
"""


def main():
    """Main entry point for PR migration linter."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="PR Migration Linter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pr_migration_linter.py
  python pr_migration_linter.py --base main --head feature-branch
  python pr_migration_linter.py --generate-template
  python pr_migration_linter.py --output lint_result.json
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
        "--output", "-o",
        type=Path,
        default=None,
        help="Output path for lint result JSON",
    )
    parser.add_argument(
        "--generate-template",
        action="store_true",
        help="Generate migration_intent.yaml template",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on warnings (not just errors)",
    )
    
    args = parser.parse_args()
    
    if args.generate_template:
        print(generate_migration_intent_template())
        return
    
    # Run linter
    linter = PRMigrationLinter()
    print(f"🔍 Linting PR for migration signals...")
    print(f"   Base: {args.base}")
    print(f"   Head: {args.head}")
    print()
    
    result = linter.lint(args.base, args.head)
    
    # Print results
    verdict_icons = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}
    print(f"{verdict_icons.get(result.verdict, '?')} Verdict: {result.verdict}")
    print()
    
    if result.signals:
        print(f"📡 Migration Signals Detected: {len(result.signals)}")
        for sig in result.signals:
            severity_icons = {"critical": "🔴", "warning": "🟡", "info": "🔵"}
            icon = severity_icons.get(sig.severity, "⚪")
            print(f"   {icon} [{sig.signal.value}] {sig.file_path}")
            print(f"      Implied: {sig.implied_migration}")
        print()
    
    if result.intent.declared:
        print(f"📋 Migration Intent Declared:")
        print(f"   {result.intent.source_phase} → {result.intent.target_phase}")
        print()
    
    if result.errors:
        print("❌ Errors:")
        for error in result.errors:
            print(f"   • {error}")
        print()
    
    if result.warnings:
        print("⚠️ Warnings:")
        for warning in result.warnings:
            print(f"   • {warning}")
        print()
    
    if result.recommendations:
        print("💡 Recommendations:")
        for rec in result.recommendations:
            print(f"   • {rec}")
        print()
    
    # Save output if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"💾 Results saved to: {args.output}")
    
    # Exit with appropriate code
    if result.verdict == "FAIL":
        sys.exit(1)
    elif result.verdict == "WARN" and args.strict:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()

