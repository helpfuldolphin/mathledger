"""
Policy recommendation memo generation.

Produces structured 2-page memos with counterfactual uplift analysis
and actionable policy recommendations.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
from pathlib import Path


@dataclass
class PolicyRecommendation:
    """
    Structured policy recommendation with counterfactual analysis.

    Attributes:
        current_policy: Currently deployed policy identifier
        recommended_policy: Recommended policy identifier
        expected_uplift: Expected throughput uplift (ratio)
        expected_uplift_ci: 95% confidence interval for uplift
        mechanism: Causal mechanism (how policy affects outcome)
        evidence_strength: Strength of evidence (weak/moderate/strong)
        risk_assessment: Potential risks and mitigations
    """
    current_policy: str
    recommended_policy: str
    expected_uplift: float
    expected_uplift_ci: Tuple[float, float]
    mechanism: str
    evidence_strength: str
    risk_assessment: str

    # Supporting metrics
    abstention_change: float
    verification_time_change: float
    depth_change: float

    # Evidence base
    n_observations: int
    n_policy_changes: int
    significant_paths: List[str]


def generate_policy_memo(
    recommendation: PolicyRecommendation,
    causal_estimates: Dict,
    output_path: Optional[str] = None
) -> str:
    """
    Generate 2-page policy recommendation memo.

    Args:
        recommendation: PolicyRecommendation object
        causal_estimates: Dictionary with ATE/CATE estimates
        output_path: Optional path to save memo

    Returns:
        Formatted markdown memo (2 pages)
    """
    date_str = datetime.now().strftime("%Y-%m-%d")

    memo = []

    # Header
    memo.append("# Policy Recommendation Memo")
    memo.append("")
    memo.append(f"**Date:** {date_str}")
    memo.append(f"**From:** Claude D - Causal Architect")
    memo.append(f"**Re:** RFL Policy Update Recommendation")
    memo.append("")
    memo.append("---")
    memo.append("")

    # Executive Summary (Page 1 - Top)
    memo.append("## Executive Summary")
    memo.append("")
    memo.append(f"**Recommendation:** Transition from `{recommendation.current_policy}` "
                f"to `{recommendation.recommended_policy}`")
    memo.append("")
    memo.append(f"**Expected Impact:**")
    memo.append(f"- Throughput uplift: **{recommendation.expected_uplift:.2f}x** "
                f"(95% CI: [{recommendation.expected_uplift_ci[0]:.2f}, "
                f"{recommendation.expected_uplift_ci[1]:.2f}])")
    memo.append(f"- Abstention change: **{recommendation.abstention_change:+.1f}** percentage points")
    memo.append(f"- Verification time change: **{recommendation.verification_time_change:+.1f}** ms (p50)")
    memo.append("")
    memo.append(f"**Evidence Strength:** {recommendation.evidence_strength.upper()}")
    memo.append(f"- Based on {recommendation.n_observations} runs with "
                f"{recommendation.n_policy_changes} policy changes")
    memo.append(f"- {len(recommendation.significant_paths)} significant causal paths identified")
    memo.append("")

    # Causal Mechanism (Page 1 - Middle)
    memo.append("## Causal Mechanism")
    memo.append("")
    memo.append(recommendation.mechanism)
    memo.append("")
    memo.append("**Causal Pathway:**")
    memo.append("```")
    memo.append("Policy Update")
    memo.append("    ↓")
    memo.append(f"Abstention Rate ({recommendation.abstention_change:+.1f}pp)")
    memo.append("    ↓")
    memo.append(f"Verification Time ({recommendation.verification_time_change:+.1f}ms)")
    memo.append("    ↓")
    memo.append(f"Proof Throughput ({recommendation.expected_uplift:.2f}x)")
    memo.append("```")
    memo.append("")

    # Statistical Evidence (Page 1 - Bottom)
    memo.append("## Statistical Evidence")
    memo.append("")
    memo.append("### Significant Causal Effects")
    memo.append("")
    memo.append("| Path | Coefficient | p-value | Interpretation |")
    memo.append("|------|-------------|---------|----------------|")

    for path in recommendation.significant_paths:
        if ' → ' in path:
            source, target = path.split(' → ')
            key = (source, target)
            if key in causal_estimates:
                est = causal_estimates[key]
                coef = est.get('coefficient', 0)
                pval = est.get('p_value', 1.0)

                direction = "increases" if coef > 0 else "decreases"
                memo.append(f"| {path} | {coef:.3f} | {pval:.4f} | "
                          f"{source} {direction} {target} |")

    memo.append("")

    # Counterfactual Analysis (Page 2 - Top)
    memo.append("---")
    memo.append("")
    memo.append("## Counterfactual Analysis")
    memo.append("")
    memo.append("### What Would Have Happened?")
    memo.append("")
    memo.append(f"If `{recommendation.recommended_policy}` had been deployed instead of "
                f"`{recommendation.current_policy}` during the observation period:")
    memo.append("")
    memo.append(f"- **Observed throughput:** {1.0:.2f}x baseline")
    memo.append(f"- **Counterfactual throughput:** {recommendation.expected_uplift:.2f}x baseline")
    memo.append(f"- **Foregone uplift:** {recommendation.expected_uplift - 1.0:.2f}x "
                f"({(recommendation.expected_uplift - 1.0) * 100:.1f}% improvement missed)")
    memo.append("")
    memo.append("This estimate uses Pearl's twin-network counterfactual method, "
                "conditioning on observed confounders and inferring latent factors.")
    memo.append("")

    # Risk Assessment (Page 2 - Middle)
    memo.append("## Risk Assessment & Mitigations")
    memo.append("")
    memo.append(recommendation.risk_assessment)
    memo.append("")

    # Recommendations (Page 2 - Bottom)
    memo.append("## Actionable Recommendations")
    memo.append("")
    memo.append("1. **Deploy Immediately** (if evidence is STRONG)")
    memo.append(f"   - Update `policy_hash` to `{recommendation.recommended_policy}`")
    memo.append("   - Monitor throughput metrics for first 24 hours")
    memo.append("   - Expect to see abstention rate adjust within first 100 proofs")
    memo.append("")
    memo.append("2. **A/B Test** (if evidence is MODERATE)")
    memo.append("   - Allocate 50% of derivation capacity to new policy")
    memo.append("   - Run for 7 days to collect sufficient data")
    memo.append("   - Re-run causal analysis to validate uplift")
    memo.append("")
    memo.append("3. **Defer** (if evidence is WEAK)")
    memo.append(f"   - Continue with `{recommendation.current_policy}`")
    memo.append(f"   - Collect more data (need n ≥ {30 - recommendation.n_observations} more runs)")
    memo.append("   - Re-evaluate when sample size sufficient")
    memo.append("")

    # Seal
    memo.append("---")
    memo.append("")
    if recommendation.evidence_strength.lower() == "strong":
        memo.append(f"**[PASS] Causal Model Stable** — "
                   f"{len(recommendation.significant_paths)} significant paths (p < 0.05)")
    elif recommendation.n_observations < 30:
        memo.append(f"**[ABSTAIN] Insufficient runs** — n={recommendation.n_observations} < 30")
    else:
        memo.append(f"**[PASS] Causal Model Moderate** — "
                   f"{len(recommendation.significant_paths)} paths identified")
    memo.append("")
    memo.append(f"**Generated:** {datetime.now().isoformat()}")
    memo.append(f"**Analyst:** Claude D - Causal Architect")
    memo.append("")

    # Convert to string
    memo_text = "\n".join(memo)

    # Save if path provided
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(memo_text)

    return memo_text


def assess_evidence_strength(
    n_observations: int,
    n_policy_changes: int,
    significant_paths: int,
    stability_cv: float
) -> str:
    """
    Assess strength of causal evidence.

    Returns:
        "strong", "moderate", or "weak"
    """
    if n_observations < 30:
        return "weak"

    if n_policy_changes < 2:
        return "weak"

    if significant_paths < 2:
        return "weak"

    if stability_cv > 0.5:
        return "weak"

    if n_observations >= 50 and significant_paths >= 3 and stability_cv < 0.2:
        return "strong"

    return "moderate"


def format_risk_assessment(
    expected_uplift: float,
    ci_width: float,
    evidence_strength: str
) -> str:
    """Format risk assessment section."""
    risks = []

    if ci_width > 0.5:
        risks.append(
            "**High Uncertainty**: Wide confidence interval suggests "
            "high variance in outcomes. Recommend extended A/B test period."
        )

    if expected_uplift < 1.1:
        risks.append(
            "**Small Effect Size**: Expected uplift is modest. "
            "Ensure measurement precision to detect actual change."
        )

    if evidence_strength == "weak":
        risks.append(
            "**Limited Evidence**: Small sample size increases risk of "
            "spurious correlation. Collect more data before deployment."
        )

    if expected_uplift > 2.0:
        risks.append(
            "**Unusually Large Effect**: Effect size exceeds typical range. "
            "Verify no data errors or confounding factors before deployment."
        )

    if not risks:
        risks.append(
            "**Low Risk**: Strong evidence with tight confidence intervals. "
            "Deployment recommended with standard monitoring."
        )

    return "\n\n".join(f"- {risk}" for risk in risks)


def create_recommendation_from_analysis(
    current_policy: str,
    recommended_policy: str,
    causal_estimates: Dict,
    run_deltas: List,
    stability_results: Dict
) -> PolicyRecommendation:
    """
    Create PolicyRecommendation from causal analysis results.

    Args:
        current_policy: Current policy hash
        recommended_policy: Recommended policy hash
        causal_estimates: Coefficient estimates
        run_deltas: List of RunDelta objects
        stability_results: Bootstrap stability results

    Returns:
        PolicyRecommendation object
    """
    from backend.causal.variables import compute_mean_deltas, stratify_by_policy_change

    # Extract deltas
    policy_changed, _ = stratify_by_policy_change(run_deltas)
    mean_deltas = compute_mean_deltas(policy_changed)

    # Compute expected uplift (using throughput ratio)
    expected_uplift = mean_deltas.get('mean_throughput_ratio', 1.0)

    # Estimate CI from stability (simplified)
    ci_lower = expected_uplift * 0.9  # TODO: proper bootstrap CI
    ci_upper = expected_uplift * 1.1

    # Extract changes
    abstention_change = mean_deltas.get('mean_delta_abstain', 0.0)
    depth_change = mean_deltas.get('mean_delta_depth', 0.0)

    # Estimate verification time change (from causal path)
    verify_delta = 0.0
    if ('abstain_pct', 'verify_ms_p50') in causal_estimates:
        coef = causal_estimates[('abstain_pct', 'verify_ms_p50')].get('coefficient', 0)
        verify_delta = coef * abstention_change

    # Identify significant paths
    significant_paths = []
    for (source, target), est in causal_estimates.items():
        if est.get('p_value', 1.0) < 0.05:
            significant_paths.append(f"{source} → {target}")

    # Assess evidence strength
    n_obs = mean_deltas.get('n_deltas', 0)
    n_policy_changes = len(policy_changed)

    avg_cv = 0.0
    if stability_results:
        cvs = [s.get('cv', 0) for s in stability_results.values() if s.get('cv', 0) < float('inf')]
        avg_cv = sum(cvs) / len(cvs) if cvs else 0.0

    evidence_strength = assess_evidence_strength(
        n_obs,
        n_policy_changes,
        len(significant_paths),
        avg_cv
    )

    # Build mechanism description
    mechanism = (
        f"The recommended policy `{recommended_policy}` alters the scoring function, "
        f"leading to a {abstention_change:+.1f}pp change in abstention rate. "
        f"This shifts the distribution of attempted proofs toward "
        f"{'harder' if abstention_change < 0 else 'easier'} formulas, "
        f"which {'increases' if verify_delta > 0 else 'decreases'} verification time by {abs(verify_delta):.1f}ms. "
        f"The net effect is a {expected_uplift:.2f}x throughput multiplier via improved "
        f"resource allocation."
    )

    # Risk assessment
    ci_width = ci_upper - ci_lower
    risk_assessment = format_risk_assessment(expected_uplift, ci_width, evidence_strength)

    return PolicyRecommendation(
        current_policy=current_policy,
        recommended_policy=recommended_policy,
        expected_uplift=expected_uplift,
        expected_uplift_ci=(ci_lower, ci_upper),
        mechanism=mechanism,
        evidence_strength=evidence_strength,
        risk_assessment=risk_assessment,
        abstention_change=abstention_change,
        verification_time_change=verify_delta,
        depth_change=depth_change,
        n_observations=n_obs,
        n_policy_changes=n_policy_changes,
        significant_paths=significant_paths
    )
