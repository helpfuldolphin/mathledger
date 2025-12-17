"""Metric governance module.

Provides governance rules and validation for metrics.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class GovernanceRule:
    """Governance rule definition."""
    name: str
    metric_name: str
    threshold: float
    operator: str = "gte"  # gte, lte, eq
    severity: str = "warn"


@dataclass
class GovernanceViolation:
    """Governance violation record."""
    rule_name: str
    metric_name: str
    actual_value: float
    threshold: float
    severity: str


@dataclass
class GovernanceReport:
    """Governance validation report."""
    valid: bool = True
    violations: List[GovernanceViolation] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def create_governance_rule(
    name: str,
    metric_name: str,
    threshold: float,
    operator: str = "gte",
    severity: str = "warn",
) -> GovernanceRule:
    """Create a governance rule."""
    return GovernanceRule(
        name=name,
        metric_name=metric_name,
        threshold=threshold,
        operator=operator,
        severity=severity,
    )


def validate_metrics(
    metrics: Dict[str, float],
    rules: List[GovernanceRule],
) -> GovernanceReport:
    """Validate metrics against governance rules."""
    violations = []
    warnings = []

    for rule in rules:
        value = metrics.get(rule.metric_name)
        if value is None:
            warnings.append(f"Metric {rule.metric_name} not found")
            continue

        violated = False
        if rule.operator == "gte" and value < rule.threshold:
            violated = True
        elif rule.operator == "lte" and value > rule.threshold:
            violated = True
        elif rule.operator == "eq" and value != rule.threshold:
            violated = True

        if violated:
            violations.append(GovernanceViolation(
                rule_name=rule.name,
                metric_name=rule.metric_name,
                actual_value=value,
                threshold=rule.threshold,
                severity=rule.severity,
            ))

    # Check for blocking violations
    blocking = any(v.severity == "block" for v in violations)

    return GovernanceReport(
        valid=not blocking,
        violations=violations,
        warnings=warnings,
        metadata={"rules_checked": len(rules)},
    )


def load_governance_rules(path: str) -> List[GovernanceRule]:
    """Load governance rules from file (stub)."""
    return []


def load_promotion_policy(path: Optional[str] = None) -> Dict[str, Any]:
    """Load promotion policy from file."""
    return {
        "default": {
            "max_flapping_events": 2,
            "max_long_term_drift": 0.05,
            "block_on_regression_outlier": True,
        },
        "lenient": {
            "max_flapping_events": 5,
            "max_long_term_drift": 0.1,
            "block_on_regression_outlier": False,
        },
    }


def can_promote_metric(
    baseline: Dict[str, Any],
    candidate: Dict[str, Any],
    timeline_data: Optional[Dict[str, Any]] = None,
    policy: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    """Check if a metric can be promoted based on governance rules."""
    policy = policy or load_promotion_policy()
    default_policy = policy.get("default", {})
    timeline_data = timeline_data or {}

    analytics = timeline_data.get("advanced_analytics", {})

    # Check flapping
    flapping_count = analytics.get("flapping_events_count", 0)
    max_flapping = default_policy.get("max_flapping_events", 2)
    if flapping_count > max_flapping:
        return False, f"flapping detected: {flapping_count} events exceeds max {max_flapping}"

    # Check drift
    drift = analytics.get("long_term_drift", 0.0)
    max_drift = default_policy.get("max_long_term_drift", 0.05)
    if drift > max_drift:
        return False, f"drift detected: {drift} exceeds max {max_drift}"

    # Check regression outlier
    if analytics.get("is_regression_outlier", False):
        if default_policy.get("block_on_regression_outlier", True):
            return False, "regression outlier detected"

    return True, "promotion allowed"


def get_promotion_status(
    metrics: Dict[str, float],
    rules: List[GovernanceRule],
) -> Dict[str, Any]:
    """Get promotion status for all metrics."""
    status = {}
    for metric_name, value in metrics.items():
        can_promote, reason = can_promote_metric(
            {"metric_name": metric_name},
            {"metric_name": metric_name, "value": value},
        )
        status[metric_name] = {
            "value": value,
            "can_promote": can_promote,
            "reason": reason,
        }
    return status


def build_metric_conformance_timeline(
    history: List[Dict[str, Any]],
    rules: Optional[List[GovernanceRule]] = None,
) -> Dict[str, Any]:
    """Build metric conformance timeline from history."""
    rules = rules or []

    timeline = []
    for entry in history:
        timestamp = entry.get("timestamp")
        metrics = entry.get("metrics", {})
        report = validate_metrics(metrics, rules)
        timeline.append({
            "timestamp": timestamp,
            "valid": report.valid,
            "violations": len(report.violations),
        })

    return {
        "timeline": timeline,
        "total_entries": len(history),
        "conforming_entries": sum(1 for t in timeline if t["valid"]),
    }


def summarize_metrics_for_global_console(
    metrics: Dict[str, float],
    rules: Optional[List[GovernanceRule]] = None,
) -> Dict[str, Any]:
    """Summarize metrics for global console display."""
    rules = rules or []
    report = validate_metrics(metrics, rules)
    return {
        "metric_count": len(metrics),
        "valid": report.valid,
        "violations": len(report.violations),
        "warnings": len(report.warnings),
    }


__all__ = [
    "GovernanceRule",
    "GovernanceViolation",
    "GovernanceReport",
    "create_governance_rule",
    "validate_metrics",
    "load_governance_rules",
    "load_promotion_policy",
    "can_promote_metric",
    "get_promotion_status",
    "build_metric_conformance_timeline",
    "summarize_metrics_for_global_console",
]
