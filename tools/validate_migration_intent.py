#!/usr/bin/env python3
"""
Migration Intent Validator

Validates migration_intent.yaml files against the schema and custom rules.

Author: Agent E4 (doc-ops-4) — Phase Migration Architect
Date: 2025-12-06
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class ValidationError:
    """A validation error."""
    field: str
    message: str
    severity: str  # "error" | "warning"


@dataclass
class ValidationResult:
    """Result of migration intent validation."""
    valid: bool
    errors: list[ValidationError]
    warnings: list[ValidationError]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": [{"field": e.field, "message": e.message} for e in self.errors],
            "warnings": [{"field": w.field, "message": w.message} for w in self.warnings],
        }


@dataclass
class AdvisorResult:
    """Result of migration intent advisor comparison."""
    status: str  # "ALIGNED" | "MISALIGNED" | "INCOMPLETE"
    phase_match: bool
    transition_declared: str | None  # e.g., "phase_i → phase_ii"
    transition_detected: list[str]   # e.g., ["Phase I → Phase II"]
    missing_acknowledgments: list[str]  # Critical signals not acknowledged
    extra_acknowledgments: list[str]    # Acknowledged but not detected
    unacknowledged_critical: list[dict[str, Any]]  # Full signal details
    recommendations: list[str]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "phase_match": self.phase_match,
            "transition_declared": self.transition_declared,
            "transition_detected": self.transition_detected,
            "missing_acknowledgments": self.missing_acknowledgments,
            "extra_acknowledgments": self.extra_acknowledgments,
            "unacknowledged_critical": self.unacknowledged_critical,
            "recommendations": self.recommendations,
        }


class MigrationIntentValidator:
    """Validator for migration_intent.yaml files."""
    
    # Valid phase transitions
    VALID_TRANSITIONS = {
        "phase_i": ["phase_ii"],
        "phase_ii": ["phase_iib", "phase_iii"],
        "phase_iib": ["phase_iii"],
        "phase_iii": [],
    }
    
    # Phase order for regression check
    PHASE_ORDER = ["phase_i", "phase_ii", "phase_iib", "phase_iii"]
    
    # Required fields
    REQUIRED_FIELDS = [
        "source_phase",
        "target_phase",
        "justification",
        "preconditions_verified",
        "rollback_plan",
    ]
    
    # Valid phases
    VALID_PHASES = ["phase_i", "phase_ii", "phase_iib", "phase_iii"]
    
    # Valid signals
    VALID_SIGNALS = [
        "db_write_introduction",
        "rfl_db_enable",
        "prereg_addition",
        "uplift_slice_addition",
        "lean_enable",
        "lean_timeout_config",
        "lean_toolchain_update",
        "basis_import_activation",
        "ed25519_introduction",
        "rfc8785_activation",
        "proof_middleware_addition",
        "curriculum_active_change",
        "security_model_change",
        "determinism_envelope_change",
    ]
    
    # Mapping from phase transitions to signal groups
    TRANSITION_TO_SIGNALS = {
        "Phase I → Phase II": [
            "db_write_introduction",
            "rfl_db_enable",
            "prereg_addition",
            "uplift_slice_addition",
        ],
        "Phase II → Phase IIb": [
            "lean_enable",
            "lean_timeout_config",
            "lean_toolchain_update",
        ],
        "Phase II → Phase III": [
            "basis_import_activation",
            "ed25519_introduction",
            "rfc8785_activation",
            "proof_middleware_addition",
        ],
    }
    
    def validate(self, intent: dict[str, Any]) -> ValidationResult:
        """Validate a migration intent dictionary."""
        errors = []
        warnings = []
        
        # Check required fields
        for field in self.REQUIRED_FIELDS:
            if field not in intent:
                errors.append(ValidationError(
                    field=field,
                    message=f"Required field '{field}' is missing",
                    severity="error",
                ))
        
        # Validate source_phase
        source = intent.get("source_phase")
        if source and source not in self.VALID_PHASES:
            errors.append(ValidationError(
                field="source_phase",
                message=f"Invalid source_phase '{source}'. Must be one of: {self.VALID_PHASES}",
                severity="error",
            ))
        
        # Validate target_phase
        target = intent.get("target_phase")
        if target and target not in self.VALID_PHASES:
            errors.append(ValidationError(
                field="target_phase",
                message=f"Invalid target_phase '{target}'. Must be one of: {self.VALID_PHASES}",
                severity="error",
            ))
        
        # Validate transition is valid
        if source and target and source in self.VALID_TRANSITIONS:
            valid_targets = self.VALID_TRANSITIONS[source]
            if target not in valid_targets:
                errors.append(ValidationError(
                    field="target_phase",
                    message=f"Invalid transition: {source} → {target}. Valid targets from {source}: {valid_targets}",
                    severity="error",
                ))
        
        # Check for regression
        if source and target:
            source_idx = self.PHASE_ORDER.index(source) if source in self.PHASE_ORDER else -1
            target_idx = self.PHASE_ORDER.index(target) if target in self.PHASE_ORDER else -1
            if source_idx >= 0 and target_idx >= 0 and target_idx < source_idx:
                errors.append(ValidationError(
                    field="target_phase",
                    message=f"Phase regression not allowed: {source} → {target}",
                    severity="error",
                ))
        
        # Validate justification length
        justification = intent.get("justification", "")
        if justification and len(justification) < 50:
            warnings.append(ValidationError(
                field="justification",
                message=f"Justification is too short ({len(justification)} chars). Should be at least 50 characters.",
                severity="warning",
            ))
        
        # Validate preconditions
        preconditions = intent.get("preconditions_verified", [])
        if not isinstance(preconditions, list):
            errors.append(ValidationError(
                field="preconditions_verified",
                message="preconditions_verified must be a list",
                severity="error",
            ))
        elif len(preconditions) < 1:
            errors.append(ValidationError(
                field="preconditions_verified",
                message="At least one precondition must be verified",
                severity="error",
            ))
        
        # Validate rollback plan length
        rollback = intent.get("rollback_plan", "")
        if rollback and len(rollback) < 30:
            warnings.append(ValidationError(
                field="rollback_plan",
                message=f"Rollback plan is too short ({len(rollback)} chars). Should be at least 30 characters.",
                severity="warning",
            ))
        
        # Validate signals_acknowledged
        signals = intent.get("signals_acknowledged", [])
        if signals:
            for signal in signals:
                if signal not in self.VALID_SIGNALS:
                    warnings.append(ValidationError(
                        field="signals_acknowledged",
                        message=f"Unknown signal: {signal}",
                        severity="warning",
                    ))
        
        # Check metadata recommendations
        metadata = intent.get("metadata", {})
        if not metadata.get("migration_simulator_run"):
            warnings.append(ValidationError(
                field="metadata.migration_simulator_run",
                message="Migration simulator should be run before migration",
                severity="warning",
            ))
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )
    
    def validate_file(self, file_path: Path) -> ValidationResult:
        """Validate a migration_intent.yaml file."""
        try:
            import yaml
            with open(file_path) as f:
                intent = yaml.safe_load(f)
            
            if intent is None:
                return ValidationResult(
                    valid=False,
                    errors=[ValidationError(
                        field="file",
                        message="File is empty or invalid YAML",
                        severity="error",
                    )],
                    warnings=[],
                )
            
            return self.validate(intent)
            
        except ImportError:
            return ValidationResult(
                valid=False,
                errors=[ValidationError(
                    field="dependency",
                    message="PyYAML not installed",
                    severity="error",
                )],
                warnings=[],
            )
        except Exception as e:
            return ValidationResult(
                valid=False,
                errors=[ValidationError(
                    field="file",
                    message=f"Error reading file: {e}",
                    severity="error",
                )],
                warnings=[],
            )
    
    def advise(
        self,
        intent: dict[str, Any],
        impact_report: dict[str, Any],
    ) -> AdvisorResult:
        """
        Compare migration intent against phase impact report.
        
        Verifies that:
        - source_phase/target_phase match detected impacts
        - All CRITICAL signals are acknowledged in preconditions_verified or justification
        
        Args:
            intent: Parsed migration_intent.yaml content
            impact_report: Parsed phase_impact_report.json content
        
        Returns:
            AdvisorResult with comparison status and details
        
        NOTE: This is advisory only — no auto-approval or auto-rejection.
        """
        recommendations = []
        
        # Extract declared transition
        source = intent.get("source_phase")
        target = intent.get("target_phase")
        transition_declared = f"{source} → {target}" if source and target else None
        
        # Extract detected transitions from impact report
        impacts = impact_report.get("impacts", [])
        transition_detected = [i.get("phase") for i in impacts if i.get("phase")]
        
        # Normalize detected transitions for comparison
        def normalize_transition(t: str) -> str:
            """Normalize transition string for comparison."""
            t = t.lower().replace("→", "->").replace(" ", "")
            t = t.replace("phasei", "phase_i").replace("phaseii", "phase_ii")
            t = t.replace("phaseiib", "phase_iib").replace("phaseiii", "phase_iii")
            return t
        
        # Check phase match
        declared_normalized = normalize_transition(transition_declared) if transition_declared else ""
        detected_normalized = [normalize_transition(t) for t in transition_detected]
        
        # Build expected transition string from declared phases
        expected_transition = f"{source}->{target}" if source and target else ""
        
        # Check if the declared transition matches any detected transition
        phase_match = False
        for detected in transition_detected:
            # Handle various formats: "Phase I → Phase II", "phase_i → phase_ii", etc.
            detected_clean = detected.lower().replace(" ", "").replace("→", "->")
            detected_clean = detected_clean.replace("phasei", "phase_i").replace("phaseii", "phase_ii")
            detected_clean = detected_clean.replace("phaseiib", "phase_iib").replace("phaseiii", "phase_iii")
            
            if expected_transition.lower().replace(" ", "") == detected_clean:
                phase_match = True
                break
            
            # Also check partial matches (e.g., declared covers detected)
            if source and target:
                if source.lower() in detected.lower() and target.lower() in detected.lower():
                    phase_match = True
                    break
        
        # Extract all critical signals from impact report
        critical_signals = []
        for impact in impacts:
            if impact.get("severity") == "CRITICAL":
                for sig in impact.get("signals", []):
                    critical_signals.append(sig)
        
        # Get acknowledged signals from intent
        acknowledged = set(intent.get("signals_acknowledged", []))
        
        # Also check if signals are mentioned in justification or preconditions
        justification = intent.get("justification", "").lower()
        preconditions = " ".join(intent.get("preconditions_verified", [])).lower()
        combined_text = justification + " " + preconditions
        
        # Identify missing acknowledgments (critical signals not acknowledged)
        missing_acknowledgments = []
        unacknowledged_critical = []
        
        for sig in critical_signals:
            signal_name = sig.get("signal", "")
            # Check if acknowledged explicitly or mentioned in text
            explicitly_acked = signal_name in acknowledged
            mentioned_in_text = (
                signal_name.replace("_", " ") in combined_text or
                signal_name.replace("_", "-") in combined_text or
                signal_name in combined_text
            )
            
            if not explicitly_acked and not mentioned_in_text:
                missing_acknowledgments.append(signal_name)
                unacknowledged_critical.append(sig)
        
        # Identify extra acknowledgments (acknowledged but not detected)
        detected_signals = set()
        for impact in impacts:
            for sig in impact.get("signals", []):
                detected_signals.add(sig.get("signal", ""))
        
        extra_acknowledgments = [s for s in acknowledged if s not in detected_signals]
        
        # Determine status
        if not transition_detected:
            status = "NO_IMPACT"
            recommendations.append("No phase transition impacts detected — migration intent may not be needed")
        elif not phase_match:
            status = "MISALIGNED"
            recommendations.append(f"Declared transition '{transition_declared}' does not match detected: {transition_detected}")
            recommendations.append("Update source_phase and target_phase in migration_intent.yaml")
        elif missing_acknowledgments:
            status = "INCOMPLETE"
            recommendations.append(f"Missing acknowledgment for {len(missing_acknowledgments)} critical signal(s)")
            for sig in missing_acknowledgments[:3]:
                recommendations.append(f"  - Add '{sig}' to signals_acknowledged or address in justification")
        else:
            status = "ALIGNED"
            if extra_acknowledgments:
                recommendations.append(f"Note: {len(extra_acknowledgments)} acknowledged signal(s) not detected in PR")
        
        return AdvisorResult(
            status=status,
            phase_match=phase_match,
            transition_declared=transition_declared,
            transition_detected=transition_detected,
            missing_acknowledgments=list(set(missing_acknowledgments)),
            extra_acknowledgments=extra_acknowledgments,
            unacknowledged_critical=unacknowledged_critical[:10],  # Limit to 10
            recommendations=recommendations,
        )
    
    def advise_from_files(
        self,
        intent_path: Path,
        report_path: Path,
    ) -> AdvisorResult:
        """
        Compare migration intent file against phase impact report file.
        
        Args:
            intent_path: Path to migration_intent.yaml
            report_path: Path to phase_impact_report.json
        
        Returns:
            AdvisorResult with comparison status
        """
        try:
            import yaml
            with open(intent_path) as f:
                intent = yaml.safe_load(f) or {}
        except Exception as e:
            return AdvisorResult(
                status="ERROR",
                phase_match=False,
                transition_declared=None,
                transition_detected=[],
                missing_acknowledgments=[],
                extra_acknowledgments=[],
                unacknowledged_critical=[],
                recommendations=[f"Error reading intent file: {e}"],
            )
        
        try:
            with open(report_path) as f:
                impact_report = json.load(f)
        except Exception as e:
            return AdvisorResult(
                status="ERROR",
                phase_match=False,
                transition_declared=None,
                transition_detected=[],
                missing_acknowledgments=[],
                extra_acknowledgments=[],
                unacknowledged_critical=[],
                recommendations=[f"Error reading impact report: {e}"],
            )
        
        return self.advise(intent, impact_report)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate migration_intent.yaml files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate migration intent schema
  python validate_migration_intent.py migration_intent.yaml
  
  # Compare intent against impact report (advisor mode)
  python validate_migration_intent.py --advise --intent migration_intent.yaml --report phase_impact_report.json
  
  # Output as JSON
  python validate_migration_intent.py --json migration_intent.yaml
        """
    )
    parser.add_argument(
        "file",
        type=Path,
        nargs="?",
        default=PROJECT_ROOT / "migration_intent.yaml",
        help="Path to migration_intent.yaml (for validation mode)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "--advise",
        action="store_true",
        help="Run in advisor mode: compare intent against impact report",
    )
    parser.add_argument(
        "--intent", "-i",
        type=Path,
        help="Path to migration_intent.yaml (for advisor mode)",
    )
    parser.add_argument(
        "--report", "-r",
        type=Path,
        help="Path to phase_impact_report.json (for advisor mode)",
    )
    
    args = parser.parse_args()
    
    validator = MigrationIntentValidator()
    
    # Advisor mode
    if args.advise:
        intent_path = args.intent or args.file or (PROJECT_ROOT / "migration_intent.yaml")
        report_path = args.report or (PROJECT_ROOT / "phase_impact_report.json")
        
        if not intent_path.exists():
            print(f"❌ Intent file not found: {intent_path}")
            sys.exit(1)
        if not report_path.exists():
            print(f"❌ Impact report not found: {report_path}")
            print("   Run: python scripts/phase_migration_simulator.py --impact-report")
            sys.exit(1)
        
        result = validator.advise_from_files(intent_path, report_path)
        
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            status_icons = {
                "ALIGNED": "✅",
                "MISALIGNED": "❌",
                "INCOMPLETE": "⚠️",
                "NO_IMPACT": "ℹ️",
                "ERROR": "💥",
            }
            icon = status_icons.get(result.status, "?")
            
            print(f"{icon} Migration Intent Advisor")
            print(f"   Status: {result.status}")
            print(f"   Phase Match: {'Yes' if result.phase_match else 'No'}")
            print()
            print(f"   Declared: {result.transition_declared or 'None'}")
            print(f"   Detected: {', '.join(result.transition_detected) or 'None'}")
            print()
            
            if result.missing_acknowledgments:
                print(f"⚠️  Missing Acknowledgments ({len(result.missing_acknowledgments)}):")
                for sig in result.missing_acknowledgments[:5]:
                    print(f"   • {sig}")
                if len(result.missing_acknowledgments) > 5:
                    print(f"   ... and {len(result.missing_acknowledgments) - 5} more")
                print()
            
            if result.extra_acknowledgments:
                print(f"ℹ️  Extra Acknowledgments ({len(result.extra_acknowledgments)}):")
                for sig in result.extra_acknowledgments[:5]:
                    print(f"   • {sig}")
                print()
            
            if result.recommendations:
                print("💡 Recommendations:")
                for rec in result.recommendations:
                    print(f"   {rec}")
        
        # Exit codes: 0=aligned, 1=misaligned/incomplete, 2=error
        if result.status == "ALIGNED" or result.status == "NO_IMPACT":
            sys.exit(0)
        elif result.status == "ERROR":
            sys.exit(2)
        else:
            sys.exit(1)
    
    # Validation mode (default)
    result = validator.validate_file(args.file)
    
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        if result.valid:
            print("✅ Migration intent is valid")
        else:
            print("❌ Migration intent is invalid")
        
        if result.errors:
            print("\nErrors:")
            for error in result.errors:
                print(f"  • [{error.field}] {error.message}")
        
        if result.warnings:
            print("\nWarnings:")
            for warning in result.warnings:
                print(f"  • [{warning.field}] {warning.message}")
    
    sys.exit(0 if result.valid else 1)


if __name__ == "__main__":
    main()

