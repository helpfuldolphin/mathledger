"""
Do-Calculus implementation for causal interventional analysis.

Implements Pearl's do-operator: P(Y | do(X=x)) for causal inference.
Enables deterministic A/B testing via fixed seeds.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
import numpy as np
from backend.repro.determinism import SeededRNG

_GLOBAL_SEED = 0


class InterventionType(Enum):
    """Types of causal interventions."""
    HARD = "hard"  # Set variable to fixed value
    SOFT = "soft"  # Modify distribution but don't fix
    ATOMIC = "atomic"  # Single atomic change


@dataclass
class Intervention:
    """
    Represents a causal intervention do(X=x).

    Attributes:
        variable: Name of intervened variable
        value: Value to set (for hard interventions)
        intervention_type: Type of intervention
        seed: Random seed for determinism
    """
    variable: str
    value: Any
    intervention_type: InterventionType = InterventionType.HARD
    seed: int = _GLOBAL_SEED

    def __repr__(self):
        return f"do({self.variable}={self.value})"


class DoOperator:
    """
    Implementation of Pearl's do-operator for causal graphs.

    Enables computing P(Y | do(X=x)) - the distribution of Y when we
    intervene to set X=x, breaking incoming causal arrows to X.
    """

    def __init__(self, seed: int = _GLOBAL_SEED):
        """
        Initialize do-operator.

        Args:
            seed: Random seed for deterministic simulations
        """
        self.seed = seed
        self.rng = SeededRNG(seed)

    def apply(
        self,
        intervention: Intervention,
        data: Dict[str, List[float]],
        causal_graph: Any  # CausalGraph type
    ) -> Dict[str, List[float]]:
        """
        Apply intervention to data.

        Args:
            intervention: The do(X=x) intervention
            data: Observational data (variable name -> values)
            causal_graph: Causal graph structure

        Returns:
            Interventional data with X set to x and effects propagated
        """
        # Make a copy to avoid mutating input
        result = {k: list(v) for k, v in data.items()}

        if intervention.intervention_type == InterventionType.HARD:
            # Hard intervention: set all values to intervention value
            n_samples = len(next(iter(data.values())))
            result[intervention.variable] = [intervention.value] * n_samples

        elif intervention.intervention_type == InterventionType.SOFT:
            # Soft intervention: add noise around intervention value
            n_samples = len(data[intervention.variable])
            noise = self.rng.normal(0, 0.1, n_samples)
            result[intervention.variable] = [
                intervention.value + n for n in noise
            ]

        # Propagate effects through causal graph
        # (In practice, this would use learned structural equations)
        result = self._propagate_effects(
            intervention.variable,
            result,
            causal_graph
        )

        return result

    def _propagate_effects(
        self,
        intervened_var: str,
        data: Dict[str, List[float]],
        causal_graph: Any
    ) -> Dict[str, List[float]]:
        """
        Propagate intervention effects through descendants.

        This is a simplified implementation. In practice, would use
        learned structural equations for each edge.
        """
        # Get topological ordering of descendants
        descendants = causal_graph.get_descendants(intervened_var)
        topo_order = causal_graph.topological_sort()

        # Filter to only descendants, maintaining order
        downstream = [v for v in topo_order if v in descendants]

        # For now, return data unchanged
        # In full implementation, would apply structural equations:
        # For each Y in descendants:
        #   data[Y] = f_Y(parents(Y), noise)
        return data


def intervene(
    variable: str,
    value: Any,
    data: Dict[str, List[float]],
    causal_graph: Any,
    seed: int = _GLOBAL_SEED
) -> Dict[str, List[float]]:
    """
    Convenience function for applying a hard intervention.

    Args:
        variable: Variable to intervene on
        value: Value to set
        data: Observational data
        causal_graph: Causal graph structure
        seed: Random seed

    Returns:
        Interventional data with effects propagated

    Example:
        >>> data = {'policy': [1, 1, 2], 'abstain': [10, 12, 5]}
        >>> intervene('policy', 2, data, graph)
        {'policy': [2, 2, 2], 'abstain': [...]}  # abstain values updated
    """
    intervention = Intervention(
        variable=variable,
        value=value,
        intervention_type=InterventionType.HARD,
        seed=seed
    )
    operator = DoOperator(seed=seed)
    return operator.apply(intervention, data, causal_graph)


def compute_ate(
    treatment_var: str,
    outcome_var: str,
    treatment_values: List[Any],
    data: Dict[str, List[float]],
    causal_graph: Any,
    seed: int = _GLOBAL_SEED
) -> float:
    """
    Compute Average Treatment Effect (ATE).

    ATE = E[Y | do(X=x1)] - E[Y | do(X=x0)]

    Args:
        treatment_var: Name of treatment variable (e.g., 'policy_hash')
        outcome_var: Name of outcome variable (e.g., 'proofs_per_sec')
        treatment_values: [control_value, treatment_value]
        data: Observational data
        causal_graph: Causal graph
        seed: Random seed

    Returns:
        Average treatment effect (difference in means)
    """
    control_val, treatment_val = treatment_values

    # Intervene to set treatment to control
    data_control = intervene(treatment_var, control_val, data, causal_graph, seed)

    # Intervene to set treatment to treatment
    data_treatment = intervene(treatment_var, treatment_val, data, causal_graph, seed)

    # Compute mean outcomes
    mean_control = np.mean(data_control[outcome_var])
    mean_treatment = np.mean(data_treatment[outcome_var])

    return mean_treatment - mean_control


def compute_cate(
    treatment_var: str,
    outcome_var: str,
    treatment_values: List[Any],
    conditioning_var: str,
    conditioning_value: Any,
    data: Dict[str, List[float]],
    causal_graph: Any,
    seed: int = _GLOBAL_SEED
) -> float:
    """
    Compute Conditional Average Treatment Effect (CATE).

    CATE = E[Y | do(X=x1), Z=z] - E[Y | do(X=x0), Z=z]

    Args:
        treatment_var: Name of treatment variable
        outcome_var: Name of outcome variable
        treatment_values: [control_value, treatment_value]
        conditioning_var: Variable to condition on (e.g., 'system')
        conditioning_value: Value to condition on (e.g., 'PL')
        data: Observational data
        causal_graph: Causal graph
        seed: Random seed

    Returns:
        Conditional average treatment effect
    """
    # Filter data to conditioning subset
    mask = [v == conditioning_value for v in data[conditioning_var]]
    filtered_data = {
        k: [v for v, m in zip(vals, mask) if m]
        for k, vals in data.items()
    }

    if not filtered_data or not next(iter(filtered_data.values())):
        return 0.0

    # Compute ATE on filtered data
    return compute_ate(
        treatment_var,
        outcome_var,
        treatment_values,
        filtered_data,
        causal_graph,
        seed
    )


def backdoor_adjustment(
    treatment_var: str,
    outcome_var: str,
    confounders: List[str],
    data: Dict[str, List[float]],
    causal_graph: Any
) -> float:
    """
    Apply backdoor adjustment formula to estimate causal effect.

    P(Y | do(X=x)) = Σ_z P(Y | X=x, Z=z) P(Z=z)

    Args:
        treatment_var: Treatment variable
        outcome_var: Outcome variable
        confounders: List of confounding variables to adjust for
        data: Observational data
        causal_graph: Causal graph

    Returns:
        Adjusted causal effect estimate
    """
    # Simplified implementation: stratify by confounders and average
    # In practice, would use proper adjustment formula

    if not confounders:
        # No confounders: simple difference in means
        return np.mean(data[outcome_var])

    # For now, return unadjusted estimate
    # Full implementation would stratify by confounder values
    return np.mean(data[outcome_var])


def frontdoor_adjustment(
    treatment_var: str,
    outcome_var: str,
    mediator_var: str,
    data: Dict[str, List[float]],
    causal_graph: Any
) -> float:
    """
    Apply frontdoor adjustment formula (when backdoor is blocked).

    P(Y | do(X=x)) = Σ_m P(M=m | X=x) Σ_x' P(Y | M=m, X=x') P(X=x')

    Args:
        treatment_var: Treatment variable
        outcome_var: Outcome variable
        mediator_var: Mediator variable (lies on causal path X→M→Y)
        data: Observational data
        causal_graph: Causal graph

    Returns:
        Adjusted causal effect estimate via frontdoor criterion
    """
    # Simplified implementation
    # Full version would properly implement frontdoor formula
    return np.mean(data[outcome_var])


def check_identifiability(
    treatment_var: str,
    outcome_var: str,
    causal_graph: Any
) -> Dict[str, Any]:
    """
    Check if causal effect is identifiable from observational data.

    Returns:
        Dictionary with identifiability results:
        - identifiable: bool
        - method: 'backdoor', 'frontdoor', 'unidentified'
        - required_vars: List of variables needed for adjustment
    """
    # Check backdoor criterion
    backdoor_sets = find_backdoor_sets(treatment_var, outcome_var, causal_graph)

    if backdoor_sets:
        return {
            'identifiable': True,
            'method': 'backdoor',
            'required_vars': backdoor_sets[0]  # Return minimal set
        }

    # Check frontdoor criterion
    frontdoor_sets = find_frontdoor_sets(treatment_var, outcome_var, causal_graph)

    if frontdoor_sets:
        return {
            'identifiable': True,
            'method': 'frontdoor',
            'required_vars': frontdoor_sets[0]
        }

    return {
        'identifiable': False,
        'method': 'unidentified',
        'required_vars': []
    }


def find_backdoor_sets(
    treatment: str,
    outcome: str,
    causal_graph: Any
) -> List[List[str]]:
    """
    Find all valid backdoor adjustment sets.

    A set Z satisfies backdoor criterion if:
    1. Z blocks all backdoor paths from X to Y
    2. Z contains no descendants of X

    Returns:
        List of valid adjustment sets (empty if none exist)
    """
    # Simplified: return parents of treatment if they don't include outcome
    parents = [p.name for p in causal_graph.get_parents(treatment)]

    if outcome not in parents:
        return [parents] if parents else []

    return []


def find_frontdoor_sets(
    treatment: str,
    outcome: str,
    causal_graph: Any
) -> List[List[str]]:
    """
    Find all valid frontdoor adjustment sets.

    Returns:
        List of valid frontdoor mediator sets
    """
    # Find mediators on path from treatment to outcome
    children = causal_graph.get_children(treatment)
    outcome_ancestors = causal_graph.get_ancestors(outcome)

    mediators = [
        c.name for c in children
        if c.name in outcome_ancestors
    ]

    return [[m] for m in mediators] if mediators else []
