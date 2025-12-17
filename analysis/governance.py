"""Analysis Governance module.

Provides governance analysis and reporting utilities.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class GovernanceAnalysis:
    """Governance analysis result."""
    compliant: bool = True
    score: float = 1.0
    violations: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


def analyze_governance(
    data: Dict[str, Any],
    rules: Optional[List[Dict[str, Any]]] = None,
) -> GovernanceAnalysis:
    """Analyze governance compliance."""
    rules = rules or []
    violations = []

    for rule in rules:
        field_name = rule.get("field")
        if field_name and field_name not in data:
            violations.append(f"Missing required field: {field_name}")

    score = 1.0 - (len(violations) * 0.1)

    return GovernanceAnalysis(
        compliant=len(violations) == 0,
        score=max(0.0, score),
        violations=violations,
        recommendations=[],
    )


def summarize_governance_for_report(
    analysis: GovernanceAnalysis,
) -> Dict[str, Any]:
    """Summarize governance analysis for reporting."""
    return {
        "compliant": analysis.compliant,
        "score": analysis.score,
        "violation_count": len(analysis.violations),
        "violations": analysis.violations,
    }


__all__ = [
    "GovernanceAnalysis",
    "analyze_governance",
    "summarize_governance_for_report",
]
