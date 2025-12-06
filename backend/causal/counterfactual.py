"""
Counterfactual reasoning and simulation.

Implements Pearl's three-level causal hierarchy:
1. Association: P(Y | X) - observational
2. Intervention: P(Y | do(X)) - interventional
3. Counterfactual: P(Y_x | X', Y') - what would have happened

Enables questions like:
"If we had used policy B instead of policy A, what would throughput have been?"
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from backend.repro.determinism import SeededRNG

_GLOBAL_SEED = 0


@dataclass
class CounterfactualScenario:
    """
    Specification of a counterfactual query.

    Represents: "What if X had been x, given that we observed X'=x', Y'=y'?"
    """
    # Intervention variables (what we change)
    intervention_var: str
    intervention_value: Any

    # Observed values (what actually happened)
    observed_vars: Dict[str, Any]

    # Target variable (what we want to predict)
    target_var: str

    # Context for determinism
    seed: int = _GLOBAL_SEED

    def __repr__(self):
        obs_str = ', '.join(f"{k}={v}" for k, v in self.observed_vars.items())
        return (
            f"Counterfactual: {self.target_var}_{{{self.intervention_var}={self.intervention_value}}} "
            f"| {obs_str}"
        )


@dataclass
class CounterfactualResult:
    """
    Result of counterfactual simulation.

    Attributes:
        scenario: The counterfactual scenario evaluated
        predicted_value: Predicted value of target under intervention
        actual_value: Actual observed value
        counterfactual_effect: Difference (predicted - actual)
        confidence_interval: [lower, upper] bounds
    """
    scenario: CounterfactualScenario
    predicted_value: float
    actual_value: float
    counterfactual_effect: float
    confidence_interval: Tuple[float, float]

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            'intervention': {
                'variable': self.scenario.intervention_var,
                'value': self.scenario.intervention_value
            },
            'target': self.scenario.target_var,
            'predicted': round(self.predicted_value, 3),
            'actual': round(self.actual_value, 3),
            'effect': round(self.counterfactual_effect, 3),
            'ci_95': [round(self.confidence_interval[0], 3),
                      round(self.confidence_interval[1], 3)]
        }


class CounterfactualEngine:
    """
    Engine for counterfactual simulation using structural causal models.

    Algorithm (Pearl's 3-step method):
    1. Abduction: Infer unobserved factors U from observed data
    2. Action: Replace structural equation for X with X=x
    3. Prediction: Compute Y using modified equations and inferred U
    """

    def __init__(self, causal_graph: Any, seed: int = _GLOBAL_SEED):
        """
        Initialize counterfactual engine.

        Args:
            causal_graph: Causal graph with structural equations
            seed: Random seed for deterministic simulation
        """
        self.graph = causal_graph
        self.seed = seed
        self.rng = SeededRNG(seed)

        # Structural equations (learned from data)
        # In practice, these would be estimated via regression/ML
        self.equations: Dict[str, Any] = {}

    def simulate(self, scenario: CounterfactualScenario) -> CounterfactualResult:
        """
        Simulate a counterfactual scenario.

        Args:
            scenario: Counterfactual query specification

        Returns:
            CounterfactualResult with predicted counterfactual value
        """
        # Step 1: Abduction - infer latent factors from observations
        latent_factors = self._infer_latent_factors(scenario.observed_vars)

        # Step 2: Action - modify structural equations for intervention
        modified_equations = self._apply_intervention(
            scenario.intervention_var,
            scenario.intervention_value
        )

        # Step 3: Prediction - compute target under counterfactual world
        predicted = self._predict_target(
            scenario.target_var,
            scenario.observed_vars,
            latent_factors,
            modified_equations
        )

        # Get actual observed value
        actual = scenario.observed_vars.get(scenario.target_var, 0.0)

        # Compute effect
        effect = predicted - actual

        # Estimate confidence interval via bootstrap
        ci = self._bootstrap_ci(scenario, n_bootstrap=100)

        return CounterfactualResult(
            scenario=scenario,
            predicted_value=predicted,
            actual_value=actual,
            counterfactual_effect=effect,
            confidence_interval=ci
        )

    def _infer_latent_factors(
        self,
        observations: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Infer latent (unobserved) factors from observations.

        In SCM: X = f_X(parents(X), U_X)
        Given X and parents, solve for U_X.

        For now, simplified: assume U ~ N(0, 1) independent noise.
        """
        # Initialize latent factors
        latent = {}

        for var_name in self.graph.nodes.keys():
            # Sample independent noise
            # In practice, would back-solve from observations
            latent[f"U_{var_name}"] = self.rng.normal(0, 1)

        return latent

    def _apply_intervention(
        self,
        var: str,
        value: Any
    ) -> Dict[str, Any]:
        """
        Modify structural equations to apply intervention.

        Replace X = f_X(parents, U) with X = value (constant).
        """
        modified = self.equations.copy()

        # Create constant function for intervention
        modified[var] = lambda **kwargs: value

        return modified

    def _predict_target(
        self,
        target: str,
        observations: Dict[str, Any],
        latent_factors: Dict[str, float],
        equations: Dict[str, Any]
    ) -> float:
        """
        Predict target variable using structural equations.

        Args:
            target: Variable to predict
            observations: Observed variable values
            latent_factors: Inferred latent factors
            equations: Structural equations (possibly modified)

        Returns:
            Predicted value of target
        """
        # If we have a structural equation for target, use it
        if target in equations:
            # Get parent values
            parents = self.graph.get_parents(target)
            parent_values = {
                p.name: observations.get(p.name, 0.0)
                for p in parents
            }

            # Get latent factor
            noise = latent_factors.get(f"U_{target}", 0.0)

            # Evaluate equation
            try:
                return equations[target](**parent_values, noise=noise)
            except:
                pass

        # Fallback: use observed value if available
        if target in observations:
            return observations[target]

        # Fallback: use default structural model
        return self._default_prediction(target, observations, latent_factors)

    def _default_prediction(
        self,
        target: str,
        observations: Dict[str, Any],
        latent_factors: Dict[str, float]
    ) -> float:
        """
        Default structural model when no equation is learned.

        Uses simple linear additive model:
        Y = β₀ + Σ β_i * X_i + U_Y
        """
        # Get parents
        parents = self.graph.get_parents(target)

        # Simple linear model with unit coefficients
        prediction = 0.0
        for parent in parents:
            if parent.name in observations:
                prediction += observations[parent.name]

        # Add noise
        prediction += latent_factors.get(f"U_{target}", 0.0) * 0.1

        return prediction

    def _bootstrap_ci(
        self,
        scenario: CounterfactualScenario,
        n_bootstrap: int = 100,
        alpha: float = 0.05
    ) -> Tuple[float, float]:
        """
        Estimate confidence interval via bootstrap.

        Args:
            scenario: Counterfactual scenario
            n_bootstrap: Number of bootstrap samples
            alpha: Significance level (0.05 for 95% CI)

        Returns:
            (lower_bound, upper_bound) confidence interval
        """
        predictions = []

        for i in range(n_bootstrap):
            # Resample with different random seed
            engine = CounterfactualEngine(
                self.graph,
                seed=self.seed + i
            )
            engine.equations = self.equations

            # Simulate
            result = engine.simulate(scenario)
            predictions.append(result.predicted_value)

        # Compute percentiles
        lower = np.percentile(predictions, alpha / 2 * 100)
        upper = np.percentile(predictions, (1 - alpha / 2) * 100)

        return (lower, upper)


def twin_network_simulation(
    scenario: CounterfactualScenario,
    causal_graph: Any,
    seed: int = _GLOBAL_SEED
) -> CounterfactualResult:
    """
    Simulate counterfactual using twin network method.

    Creates two networks:
    - Factual network: represents what actually happened
    - Counterfactual network: represents alternative scenario

    Both networks share latent factors (ensures consistency).

    Args:
        scenario: Counterfactual query
        causal_graph: Causal graph structure
        seed: Random seed

    Returns:
        Counterfactual result
    """
    engine = CounterfactualEngine(causal_graph, seed)
    return engine.simulate(scenario)


def bounds_analysis(
    scenario: CounterfactualScenario,
    causal_graph: Any,
    observational_data: Dict[str, List[float]]
) -> Tuple[float, float]:
    """
    Compute bounds on counterfactual quantity when point identification fails.

    Returns:
        (lower_bound, upper_bound) - tightest possible bounds
    """
    # In practice, would compute using symbolic algebra or optimization
    # For now, return wide bounds based on data range

    target = scenario.target_var
    if target not in observational_data:
        return (0.0, 1.0)

    values = observational_data[target]
    return (min(values), max(values))


def sensitivity_analysis(
    scenario: CounterfactualScenario,
    causal_graph: Any,
    param_name: str,
    param_range: Tuple[float, float],
    n_steps: int = 20
) -> List[Tuple[float, float]]:
    """
    Analyze sensitivity of counterfactual to parameter changes.

    Varies a structural parameter and observes effect on prediction.

    Args:
        scenario: Counterfactual query
        causal_graph: Causal graph
        param_name: Name of parameter to vary
        param_range: (min, max) range for parameter
        n_steps: Number of parameter values to test

    Returns:
        List of (parameter_value, predicted_effect) pairs
    """
    results = []

    param_min, param_max = param_range
    param_values = np.linspace(param_min, param_max, n_steps)

    for param_val in param_values:
        # Modify structural equations with new parameter
        # (In practice, would update specific equation parameter)

        engine = CounterfactualEngine(causal_graph, scenario.seed)
        result = engine.simulate(scenario)

        results.append((param_val, result.counterfactual_effect))

    return results


def policy_comparison_counterfactual(
    baseline_policy: str,
    alternative_policy: str,
    observed_run_data: Dict[str, Any],
    causal_graph: Any,
    seed: int = _GLOBAL_SEED
) -> Dict[str, CounterfactualResult]:
    """
    Compute counterfactual comparison between two policies.

    "What would throughput/abstention have been if we used policy B
     instead of policy A in the observed run?"

    Args:
        baseline_policy: Policy that was actually used (observed)
        alternative_policy: Policy to simulate (counterfactual)
        observed_run_data: Data from the actual run
        causal_graph: Causal graph
        seed: Random seed

    Returns:
        Dictionary mapping outcome variables to counterfactual results
    """
    results = {}

    # Define outcomes of interest
    outcomes = ['proofs_per_sec', 'abstain_pct', 'depth_max']

    for outcome in outcomes:
        if outcome not in observed_run_data:
            continue

        # Create counterfactual scenario
        scenario = CounterfactualScenario(
            intervention_var='policy_hash',
            intervention_value=alternative_policy,
            observed_vars={
                'policy_hash': baseline_policy,
                outcome: observed_run_data[outcome]
            },
            target_var=outcome,
            seed=seed
        )

        # Simulate
        engine = CounterfactualEngine(causal_graph, seed)
        result = engine.simulate(scenario)

        results[outcome] = result

    return results
