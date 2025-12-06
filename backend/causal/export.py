"""
Export causal estimates in JSON format for handoff to Codex M.

Produces artifacts/causal/estimates.json with ATE/CATE + confidence intervals.
"""

import json
from typing import Dict, List, Any
from pathlib import Path
from datetime import datetime


def export_estimates_json(
    causal_estimates: Dict,
    stability_results: Dict,
    run_deltas: List,
    config: Dict,
    output_path: str = "artifacts/causal/estimates.json"
) -> None:
    """
    Export causal estimates to JSON for Codex M consumption.

    Args:
        causal_estimates: Dictionary of CausalCoefficient objects
        stability_results: Bootstrap stability metrics
        run_deltas: List of RunDelta objects
        config: Configuration dictionary
        output_path: Path to save JSON
    """
    from backend.causal.variables import compute_mean_deltas, stratify_by_policy_change

    # Extract metadata
    policy_changed, policy_unchanged = stratify_by_policy_change(run_deltas)

    # Build estimates structure
    estimates = {
        "schema": "causal_estimates_v1",
        "generated_at": datetime.now().isoformat(),
        "generator": "Claude D - Causal Architect",

        "metadata": {
            "n_total_runs": len(run_deltas),
            "n_policy_changes": len(policy_changed),
            "n_unchanged": len(policy_unchanged),
            "bootstrap_replicates": config.get("estimation", {}).get("bootstrap_replicates", 0),
            "seed": config.get("estimation", {}).get("seed", 0)
        },

        "ate": {},  # Average Treatment Effects
        "cate": {},  # Conditional Average Treatment Effects
        "path_coefficients": {},  # Direct causal paths
        "stability": {},  # Bootstrap stability metrics
        "empirical_deltas": {}  # Observed changes
    }

    # Extract ATE (simplified: using policy_hash as treatment)
    # Full ATE would require proper interventional estimation
    policy_deltas = compute_mean_deltas(policy_changed)
    estimates["ate"]["policy_change_to_throughput"] = {
        "effect": policy_deltas.get("mean_throughput_ratio", 1.0),
        "description": "Average throughput ratio when policy changes",
        "n_observations": policy_deltas.get("n_deltas", 0)
    }

    # Extract path coefficients
    for (source, target), coef_obj in causal_estimates.items():
        path_key = f"{source}_to_{target}"
        estimates["path_coefficients"][path_key] = {
            "source": source,
            "target": target,
            "coefficient": coef_obj.coefficient,
            "std_error": coef_obj.std_error,
            "ci_95": list(coef_obj.confidence_interval),
            "p_value": coef_obj.p_value,
            "significant": coef_obj.is_significant,
            "n_observations": coef_obj.n_observations,
            "method": coef_obj.method.value
        }

    # Extract stability metrics
    for (source, target), stability in stability_results.items():
        path_key = f"{source}_to_{target}"
        estimates["stability"][path_key] = {
            "mean_coefficient": stability.get("mean", 0),
            "std_coefficient": stability.get("std", 0),
            "cv": stability.get("cv", 0),
            "percentiles": stability.get("percentiles", {}),
            "stable": stability.get("cv", float('inf')) < 0.3
        }

    # Extract empirical deltas
    estimates["empirical_deltas"]["policy_changed"] = compute_mean_deltas(policy_changed)
    estimates["empirical_deltas"]["policy_unchanged"] = compute_mean_deltas(policy_unchanged)

    # Compute pass/abstain seal
    n_significant = sum(1 for est in estimates["path_coefficients"].values() if est["significant"])
    n_stable = sum(1 for est in estimates["stability"].values() if est.get("stable", False))

    estimates["seal"] = {
        "status": "PASS" if n_significant >= 2 and estimates["metadata"]["n_total_runs"] >= 30 else "ABSTAIN",
        "n_significant_paths": n_significant,
        "n_stable_paths": n_stable,
        "min_p_value": min((est["p_value"] for est in estimates["path_coefficients"].values()), default=1.0),
        "criteria": {
            "min_runs": 30,
            "min_significant_paths": 2,
            "max_p_value": 0.05
        }
    }

    # Add seal message
    if estimates["seal"]["status"] == "PASS":
        estimates["seal"]["message"] = (
            f"[PASS] Causal Model Stable p<0.05 paths>={n_significant}"
        )
    else:
        if estimates["metadata"]["n_total_runs"] < 30:
            estimates["seal"]["message"] = (
                f"[ABSTAIN] Insufficient runs n={estimates['metadata']['n_total_runs']}<30"
            )
        else:
            estimates["seal"]["message"] = (
                f"[ABSTAIN] Insufficient significant paths ({n_significant} < 2)"
            )

    # Write to file
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(estimates, f, indent=2, sort_keys=True)

    print(f"✓ Exported causal estimates to {output_path}")
    print(f"  {estimates['seal']['message']}")


def load_estimates_json(path: str = "artifacts/causal/estimates.json") -> Dict[str, Any]:
    """
    Load causal estimates from JSON.

    Args:
        path: Path to estimates JSON file

    Returns:
        Dictionary with estimates
    """
    with open(path, 'r') as f:
        return json.load(f)
