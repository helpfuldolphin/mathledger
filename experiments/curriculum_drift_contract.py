# PHASE II — NOT USED IN PHASE I
# File: experiments/curriculum_drift_contract.py
"""
Curriculum Drift Contract Validation.

Implements institutional-grade change control rules for curriculum drift:

    SEMANTIC drift     → BLOCK (critical experiment definition change)
    STRUCTURAL drift   → BLOCK (slice topology change)
    PARAMETRIC_MAJOR   → BLOCK (significant parameter deviation)
    PARAMETRIC_MINOR   → WARN  (minor tuning, proceed with caution)
    COSMETIC drift     → PASS  (no meaningful change)
    NONE               → PASS  (identical)

Usage:
    from experiments.curriculum_drift_contract import DriftContract, validate_drift
    
    # Validate a diff result
    result = validate_drift(diff)
    if result.verdict == ContractVerdict.BLOCK:
        sys.exit(1)
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from experiments.curriculum_hash_ledger import (
    DriftType,
    RiskLevel,
    DRIFT_RISK_MAP,
)


class ContractVerdict(str, Enum):
    """Contract validation verdict."""
    PASS = "PASS"    # Proceed without concern
    WARN = "WARN"    # Proceed but flag for review
    BLOCK = "BLOCK"  # Do not proceed


class ContractRule(str, Enum):
    """Identifiers for contract rules."""
    STRUCTURAL_BLOCK = "STRUCTURAL_BLOCK"
    SEMANTIC_BLOCK = "SEMANTIC_BLOCK"
    PARAMETRIC_MAJOR_BLOCK = "PARAMETRIC_MAJOR_BLOCK"
    PARAMETRIC_MINOR_WARN = "PARAMETRIC_MINOR_WARN"
    COSMETIC_PASS = "COSMETIC_PASS"
    NONE_PASS = "NONE_PASS"


@dataclass
class RuleViolation:
    """A single contract rule violation."""
    rule: ContractRule
    drift_type: DriftType
    affected_slices: List[str]
    message: str
    severity: ContractVerdict


@dataclass
class ContractValidationResult:
    """Result of drift contract validation."""
    verdict: ContractVerdict
    violations: List[RuleViolation] = field(default_factory=list)
    warnings: List[RuleViolation] = field(default_factory=list)
    passes: List[RuleViolation] = field(default_factory=list)
    summary: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "verdict": self.verdict.value,
            "violations": [
                {
                    "rule": v.rule.value,
                    "drift_type": v.drift_type.value,
                    "affected_slices": v.affected_slices,
                    "message": v.message,
                    "severity": v.severity.value
                }
                for v in self.violations
            ],
            "warnings": [
                {
                    "rule": w.rule.value,
                    "drift_type": w.drift_type.value,
                    "affected_slices": w.affected_slices,
                    "message": w.message,
                    "severity": w.severity.value
                }
                for w in self.warnings
            ],
            "passes": [
                {
                    "rule": p.rule.value,
                    "drift_type": p.drift_type.value,
                    "affected_slices": p.affected_slices,
                    "message": p.message,
                    "severity": p.severity.value
                }
                for p in self.passes
            ],
            "summary": self.summary,
            "timestamp": self.timestamp,
            "has_violations": len(self.violations) > 0,
            "has_warnings": len(self.warnings) > 0
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class DriftContract:
    """
    Drift Contract enforcement engine.
    
    Validates drift reports against institutional change control rules.
    """
    
    # Rule definitions mapping drift types to verdicts and rules
    RULES: Dict[DriftType, tuple] = {
        DriftType.STRUCTURAL: (ContractVerdict.BLOCK, ContractRule.STRUCTURAL_BLOCK),
        DriftType.SEMANTIC: (ContractVerdict.BLOCK, ContractRule.SEMANTIC_BLOCK),
        DriftType.PARAMETRIC_MAJOR: (ContractVerdict.BLOCK, ContractRule.PARAMETRIC_MAJOR_BLOCK),
        DriftType.PARAMETRIC_MINOR: (ContractVerdict.WARN, ContractRule.PARAMETRIC_MINOR_WARN),
        DriftType.COSMETIC: (ContractVerdict.PASS, ContractRule.COSMETIC_PASS),
        DriftType.NONE: (ContractVerdict.PASS, ContractRule.NONE_PASS),
    }
    
    # Human-readable rule descriptions
    RULE_DESCRIPTIONS: Dict[ContractRule, str] = {
        ContractRule.STRUCTURAL_BLOCK: "Slice added or removed — requires explicit approval",
        ContractRule.SEMANTIC_BLOCK: "Experiment definition changed (formula pool, targets, metrics) — requires explicit approval",
        ContractRule.PARAMETRIC_MAJOR_BLOCK: "Major parameter deviation (>50% or removed) — requires explicit approval",
        ContractRule.PARAMETRIC_MINOR_WARN: "Minor parameter tuning (<10%) — proceed with caution",
        ContractRule.COSMETIC_PASS: "Cosmetic changes only — safe to proceed",
        ContractRule.NONE_PASS: "No changes detected — safe to proceed",
    }
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize the contract validator.
        
        Args:
            strict_mode: If True, WARN-level violations become BLOCK.
        """
        self.strict_mode = strict_mode
    
    def validate(self, diff: Dict[str, Any]) -> ContractValidationResult:
        """
        Validate a drift report against the contract rules.
        
        Args:
            diff: A drift classification dict from CurriculumHashLedger.classify_drift()
        
        Returns:
            ContractValidationResult with verdict, violations, and warnings.
        """
        violations: List[RuleViolation] = []
        warnings: List[RuleViolation] = []
        passes: List[RuleViolation] = []
        
        # Check overall drift type
        drift_type_str = diff.get("drift_type", "NONE")
        try:
            overall_drift = DriftType(drift_type_str)
        except ValueError:
            overall_drift = DriftType.NONE
        
        # Get affected slices
        affected_slices = diff.get("affected_slices", {})
        
        # Process each affected slice
        if affected_slices:
            for slice_name, slice_info in affected_slices.items():
                slice_drift_str = slice_info.get("drift_type", "NONE")
                try:
                    slice_drift = DriftType(slice_drift_str)
                except ValueError:
                    slice_drift = DriftType.SEMANTIC  # Default to SEMANTIC for unknown
                
                verdict, rule = self.RULES.get(slice_drift, (ContractVerdict.BLOCK, ContractRule.SEMANTIC_BLOCK))
                description = self.RULE_DESCRIPTIONS.get(rule, "Unknown rule")
                
                violation = RuleViolation(
                    rule=rule,
                    drift_type=slice_drift,
                    affected_slices=[slice_name],
                    message=f"[{slice_name}] {description}",
                    severity=verdict
                )
                
                if verdict == ContractVerdict.BLOCK:
                    violations.append(violation)
                elif verdict == ContractVerdict.WARN:
                    if self.strict_mode:
                        violation.severity = ContractVerdict.BLOCK
                        violations.append(violation)
                    else:
                        warnings.append(violation)
                else:
                    passes.append(violation)
        else:
            # No per-slice info, use overall drift type
            verdict, rule = self.RULES.get(overall_drift, (ContractVerdict.PASS, ContractRule.NONE_PASS))
            description = self.RULE_DESCRIPTIONS.get(rule, "No changes")
            
            all_slices = (
                diff.get("added_slices", []) +
                diff.get("removed_slices", []) +
                diff.get("changed_slices", [])
            )
            
            violation = RuleViolation(
                rule=rule,
                drift_type=overall_drift,
                affected_slices=all_slices,
                message=description,
                severity=verdict
            )
            
            if verdict == ContractVerdict.BLOCK:
                violations.append(violation)
            elif verdict == ContractVerdict.WARN:
                if self.strict_mode:
                    violation.severity = ContractVerdict.BLOCK
                    violations.append(violation)
                else:
                    warnings.append(violation)
            else:
                passes.append(violation)
        
        # Determine overall verdict
        if violations:
            overall_verdict = ContractVerdict.BLOCK
        elif warnings:
            overall_verdict = ContractVerdict.WARN
        else:
            overall_verdict = ContractVerdict.PASS
        
        # Generate summary
        summary = self._generate_summary(overall_verdict, violations, warnings, passes, diff)
        
        return ContractValidationResult(
            verdict=overall_verdict,
            violations=violations,
            warnings=warnings,
            passes=passes,
            summary=summary
        )
    
    def _generate_summary(
        self,
        verdict: ContractVerdict,
        violations: List[RuleViolation],
        warnings: List[RuleViolation],
        passes: List[RuleViolation],
        diff: Dict[str, Any]
    ) -> str:
        """Generate a human-readable summary."""
        lines = []
        
        if verdict == ContractVerdict.BLOCK:
            lines.append(f"🚫 CONTRACT VIOLATION: {len(violations)} blocking issue(s)")
            for v in violations:
                lines.append(f"   • {v.message}")
        elif verdict == ContractVerdict.WARN:
            lines.append(f"⚠️  CONTRACT WARNING: {len(warnings)} issue(s) require review")
            for w in warnings:
                lines.append(f"   • {w.message}")
        else:
            lines.append("✅ CONTRACT PASSED: No blocking issues")
        
        # Add hash comparison
        old_hash = diff.get("old_curriculum_hash", "unknown")[:12]
        new_hash = diff.get("new_curriculum_hash", "unknown")[:12]
        if old_hash != new_hash:
            lines.append(f"   Hash: {old_hash}... → {new_hash}...")
        
        return "\n".join(lines)


def validate_drift(
    diff: Dict[str, Any],
    strict_mode: bool = False
) -> ContractValidationResult:
    """
    Convenience function to validate drift against contract rules.
    
    Args:
        diff: A drift classification dict from CurriculumHashLedger.classify_drift()
        strict_mode: If True, WARN-level violations become BLOCK.
    
    Returns:
        ContractValidationResult with verdict, violations, and warnings.
    """
    contract = DriftContract(strict_mode=strict_mode)
    return contract.validate(diff)


def main():
    """CLI entry point for drift contract validation."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description="Validate curriculum drift against contract rules."
    )
    parser.add_argument(
        "diff_file",
        type=str,
        nargs="?",
        help="Path to JSON diff file (or - for stdin)."
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Strict mode: WARN-level issues become BLOCK."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON."
    )
    
    args = parser.parse_args()
    
    # Read diff from file or stdin
    if args.diff_file == "-" or args.diff_file is None:
        import sys
        diff_data = sys.stdin.read()
    else:
        with open(args.diff_file, 'r') as f:
            diff_data = f.read()
    
    diff = json.loads(diff_data)
    
    # Validate
    result = validate_drift(diff, strict_mode=args.strict)
    
    # Output
    if args.json:
        print(result.to_json())
    else:
        print(result.summary)
        print()
        if result.violations:
            print("Violations:")
            for v in result.violations:
                print(f"  [{v.rule.value}] {v.message}")
        if result.warnings:
            print("Warnings:")
            for w in result.warnings:
                print(f"  [{w.rule.value}] {w.message}")
    
    # Exit code based on verdict
    if result.verdict == ContractVerdict.BLOCK:
        sys.exit(1)
    elif result.verdict == ContractVerdict.WARN:
        sys.exit(0)  # Warnings don't fail by default
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()

