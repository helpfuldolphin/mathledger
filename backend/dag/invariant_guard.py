# backend/dag/invariant_guard.py
"""
Invariant Guard for the Statement DAG.

Provides a configurable ruleset that can be used to enforce structural
constraints on a Proof DAG as it evolves through curriculum slices.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

# Announce compliance on import
print(
    "PHASE II — NOT USED IN PHASE I: Loading DAG Invariant Guard.",
    file=__import__("sys").stderr,
)


@dataclass
class SliceProfile:
    """Summary statistics for a single slice within the DAG."""

    slice_id: str
    max_depth: int
    max_branching_factor: float
    node_kind_counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class ProofDag:
    """
    Aggregated representation of a proof DAG.

    Attributes:
        slices: Mapping of slice identifier to its profile.
        metric_ledger: Ordered list of evolution metrics (cycle indexed).
    """

    slices: Dict[str, SliceProfile] = field(default_factory=dict)
    metric_ledger: List[Dict[str, Any]] = field(default_factory=list)

    def global_max_depth(self) -> int:
        """Return the maximum depth observed across the metric ledger."""
        return max((m.get("MaxDepth(t)", 0) for m in self.metric_ledger), default=0)

    def global_branching_factor(self) -> float:
        """Return the highest observed branching factor."""
        return max(
            (m.get("GlobalBranchingFactor(t)", 0.0) for m in self.metric_ledger),
            default=0.0,
        )


def _resolve_per_slice_value(
    per_slice_config: Any, slice_id: str
) -> Optional[Any]:
    """
    Helper to resolve a per-slice rule value.

    Supports wildcard/default entries via the "*" key.
    """
    if per_slice_config is None:
        return None
    if not isinstance(per_slice_config, dict):
        return per_slice_config
    if slice_id in per_slice_config:
        return per_slice_config[slice_id]
    return per_slice_config.get("*")


def evaluate_dag_invariants(dag: ProofDag, rules: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate the configured invariant rules against the provided DAG snapshot.

    Returns:
        dict with:
            - violated_invariants: List[str] describing each failure.
            - status: "OK" | "WARN" | "BLOCK"
            - details: Structured data for observability (optional for callers).
    """
    violations: List[str] = []
    details: List[Dict[str, Any]] = []

    depth_rules = rules.get("max_depth_per_slice", {})
    allowed_kinds_rules = rules.get("allowed_node_kinds", {})
    branching_rules = rules.get("max_branching_factor_per_slice", {})

    for slice_id, profile in dag.slices.items():
        max_depth_limit = _resolve_per_slice_value(depth_rules, slice_id)
        if max_depth_limit is not None and profile.max_depth > max_depth_limit:
            violation_name = f"{slice_id}.max_depth>{max_depth_limit}"
            violations.append(violation_name)
            details.append(
                {
                    "slice": slice_id,
                    "invariant": "max_depth_per_slice",
                    "observed": profile.max_depth,
                    "limit": max_depth_limit,
                }
            )

        allowed_kinds = _resolve_per_slice_value(allowed_kinds_rules, slice_id)
        if allowed_kinds:
            allowed_set = set(allowed_kinds)
            unknown_kinds = [
                kind for kind in profile.node_kind_counts.keys() if kind not in allowed_set
            ]
            if unknown_kinds:
                violation_name = f"{slice_id}.node_kinds"
                violations.append(violation_name)
                details.append(
                    {
                        "slice": slice_id,
                        "invariant": "allowed_node_kinds",
                        "disallowed_kinds": sorted(unknown_kinds),
                        "allowed": sorted(allowed_set),
                    }
                )

        slice_branch_limit = _resolve_per_slice_value(branching_rules, slice_id)
        if (
            slice_branch_limit is not None
            and profile.max_branching_factor > slice_branch_limit
        ):
            violation_name = f"{slice_id}.branching>{slice_branch_limit}"
            violations.append(violation_name)
            details.append(
                {
                    "slice": slice_id,
                    "invariant": "max_branching_factor_per_slice",
                    "observed": profile.max_branching_factor,
                    "limit": slice_branch_limit,
                }
            )

    global_branching_limit = rules.get("max_branching_factor")
    if global_branching_limit is not None:
        observed_branching = dag.global_branching_factor()
        if observed_branching > global_branching_limit:
            violations.append("global.max_branching_factor")
            details.append(
                {
                    "invariant": "max_branching_factor",
                    "observed": observed_branching,
                    "limit": global_branching_limit,
                }
            )

    if not violations:
        status = "OK"
    elif len(violations) == 1:
        status = "WARN"
    else:
        status = "BLOCK"

    return {
        "violated_invariants": violations,
        "status": status,
        "details": details,
    }


def load_invariant_rules(path: Path) -> Dict[str, Any]:
    """
    Load invariant constraints from a YAML or JSON file.

    Supported schema:
        globals:
            max_branching_factor: float
        slices:
            SLICE_NAME:
                max_depth: int
                max_branching_factor: float
                allowed_node_kinds: [..]

    Returns a dictionary compatible with evaluate_dag_invariants.
    """
    if not path.exists():
        raise FileNotFoundError(f"Invariant rules file not found: {path}")

    raw_text = path.read_text(encoding="utf-8")

    data: Any
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        if yaml is None:
            raise ValueError(
                "Invariant rules file is not valid JSON and PyYAML is unavailable."
            )
        data = yaml.safe_load(raw_text)

    if not isinstance(data, dict):
        raise ValueError("Invariant rules must be defined as a mapping/dictionary.")

    rules: Dict[str, Any] = {}

    globals_cfg = data.get("globals", {})
    if isinstance(globals_cfg, dict):
        if "max_branching_factor" in globals_cfg:
            rules["max_branching_factor"] = globals_cfg["max_branching_factor"]

    # Allow direct specification of rule maps
    depth_rules: Dict[str, Any] = {}
    allowed_rules: Dict[str, Any] = {}
    branching_per_slice: Dict[str, Any] = {}

    if isinstance(data.get("max_depth_per_slice"), dict):
        depth_rules.update(data["max_depth_per_slice"])
    def _coerce_kind_set(value: Any, slice_name: str) -> Any:
        if isinstance(value, (list, set, tuple)):
            return set(value)
        raise ValueError(
            f"allowed_node_kinds for slice '{slice_name}' must be a list or set."
        )

    if isinstance(data.get("allowed_node_kinds"), dict):
        for k, v in data["allowed_node_kinds"].items():
            allowed_rules[k] = _coerce_kind_set(v, k)
    if isinstance(data.get("max_branching_factor_per_slice"), dict):
        branching_per_slice.update(data["max_branching_factor_per_slice"])

    slices_cfg = data.get("slices", {})
    if slices_cfg and not isinstance(slices_cfg, dict):
        raise ValueError("'slices' section must be a mapping of slice names.")

    if isinstance(slices_cfg, dict):
        for slice_name, slice_cfg in slices_cfg.items():
            if not isinstance(slice_name, str):
                raise ValueError("Slice identifiers must be strings.")
            if not isinstance(slice_cfg, dict):
                continue
            if "max_depth" in slice_cfg:
                depth_rules[slice_name] = slice_cfg["max_depth"]
            if "max_branching_factor" in slice_cfg:
                branching_per_slice[slice_name] = slice_cfg["max_branching_factor"]
            if "allowed_node_kinds" in slice_cfg:
                kinds = slice_cfg["allowed_node_kinds"]
                allowed_rules[slice_name] = _coerce_kind_set(kinds, slice_name)

    if depth_rules:
        rules["max_depth_per_slice"] = depth_rules
    if allowed_rules:
        rules["allowed_node_kinds"] = allowed_rules
    if branching_per_slice:
        rules["max_branching_factor_per_slice"] = branching_per_slice

    return rules


def summarize_dag_invariants_for_global_health(
    report: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Summarize an invariant evaluation report for global health dashboards.
    """
    violations = report.get("violated_invariants", []) or []
    status = report.get("status", "OK")
    violation_count = len(violations)

    if status == "OK":
        headline = "All invariants satisfied."
    elif status == "WARN":
        headline = (
            f"Single invariant breached: {violations[0]}"
            if violations
            else "Single invariant breach detected."
        )
    else:
        headline = (
            f"{violation_count} invariants breached."
            if violation_count
            else "Invariant guard reported BLOCK status."
        )

    return {
        "status": status,
        "violation_count": violation_count,
        "headline": headline,
    }
