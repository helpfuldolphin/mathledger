# Copyright 2025 MathLedger
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Dynamics-Theory Unification Engine (conjecture_engine.py)
=========================================================

This module implements Algorithm 1 from U2_DYNAMICS_SEMANTIC_MODEL.md.
It serves as the main entry point for running a full analysis on
experimental results and testing them against the theoretical conjectures.

Author: Gemini M, Dynamics-Theory Unification Engineer
"""

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Import the analysis functions from our dynamics module
from analysis import u2_dynamics as u2

DEFAULT_THRESHOLDS_PATH = Path("config/dynamics_thresholds.json")


# --- Behavior Classification Engine (for governance summaries) ---

class BehaviorClassification(Enum):
    """Classification of experimental behavior based on governance summary."""
    STRONG_UPLIFT = "strong_uplift"
    WEAK_UPLIFT = "weak_uplift"
    INCONCLUSIVE = "inconclusive"
    HARMFUL_REGRESSION = "harmful_regression"
    REGRESSION_IN_SUCCESS = "regression_in_success"


def run_conjecture_engine(summary_path: str) -> Dict[str, Any]:
    """
    Analyze a governance summary file and derive a behavior classification
    and conjecture statement.

    Args:
        summary_path: Path to the JSON summary file.

    Returns:
        A dictionary containing the behavior classification and derived conjecture.

    Raises:
        ValueError: If the summary file cannot be loaded or parsed.
    """
    try:
        with open(summary_path, "r") as f:
            summary = json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        raise ValueError(f"Failed to load or parse summary file: {e}")

    # Extract key metrics
    governance = summary.get("governance", {})
    metrics = summary.get("metrics", {})
    throughput = metrics.get("throughput", {})
    success_rate = metrics.get("success_rate", {})

    governance_passed = governance.get("passed", False)
    throughput_significant = throughput.get("significant", False)
    throughput_delta_pct = throughput.get("delta_pct", 0.0)
    success_delta = success_rate.get("delta", 0.0)

    # Classification logic
    if throughput_significant and throughput_delta_pct < 0:
        classification = BehaviorClassification.HARMFUL_REGRESSION
        conjecture = "The RFL policy is strictly worse than baseline in throughput."
    elif not governance_passed and success_delta < 0:
        classification = BehaviorClassification.REGRESSION_IN_SUCCESS
        conjecture = "The RFL policy may be over-optimized for speed at the cost of correctness."
    elif not throughput_significant:
        classification = BehaviorClassification.INCONCLUSIVE
        conjecture = "The evidence is insufficient to either support or refute the uplift hypothesis."
    elif governance_passed and throughput_significant and throughput_delta_pct > 0:
        classification = BehaviorClassification.STRONG_UPLIFT
        conjecture = "The RFL policy is more efficient and effective than baseline."
    else:
        classification = BehaviorClassification.WEAK_UPLIFT
        conjecture = "The RFL policy is marginally better than baseline but does not meet governance thresholds."

    return {
        "behavior_classification": classification.name,
        "derived_conjecture": conjecture,
        "governance_passed": governance_passed,
        "metrics_summary": {
            "throughput_significant": throughput_significant,
            "throughput_delta_pct": throughput_delta_pct,
            "success_delta": success_delta,
        }
    }


def _load_thresholds(path: Path = DEFAULT_THRESHOLDS_PATH) -> Dict[str, float]:
    """Loads thresholds from disk with a deterministic fallback."""
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "stagnation_std_thresh": 0.01,
            "trend_tau_thresh": -0.2,
            "oscillation_omega_thresh": 0.3,
            "step_size_thresh": 0.1,
        }


def _emit_dynamics_debug(
    snapshots: List[u2.DynamicsDebugSnapshot],
    output_path: Optional[str],
) -> Optional[str]:
    if not snapshots:
        return None

    if output_path:
        target = Path(output_path)
    else:
        digest_source = "|".join(f"{snap.run_id}:{snap.mode}:{snap.slice_name}" for snap in snapshots)
        digest = hashlib.sha256(digest_source.encode("utf-8")).hexdigest()[:12]
        target = Path("artifacts/dynamics") / f"dynamics_debug_{digest}.jsonl"

    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for snapshot in snapshots:
            payload = asdict(snapshot)
            payload["uplift_ci"] = list(snapshot.uplift_ci)
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")
    return str(target)


def run_conjecture_analysis(baseline_records: list,
                            rfl_records: list,
                            slice_uplift_threshold: float,
                            conjectures_to_test: list,
                            thresholds: Optional[dict] = None,
                            debug_dynamics: bool = False,
                            debug_output_path: Optional[str] = None,
                            run_metadata: Optional[Dict[str, Any]] = None) -> dict:
    """
    Implements Algorithm 1: Automated Conjecture Test.

    Args:
        baseline_records: A list of per-cycle records from a baseline run.
        rfl_records: A list of per-cycle records from an RFL run.
        slice_uplift_threshold: The preregistered uplift threshold (tau_i).
        conjectures_to_test: A list of strings with conjecture IDs to test.
        thresholds: A dictionary of thresholds for pattern detection.

    Returns:
        A dictionary report mapping each conjecture to its evidential status.
    """
    if thresholds is None:
        thresholds = _load_thresholds()

    thresholds_struct = u2.DynamicsThresholds.from_mapping(thresholds)
    report = {}
    debug_snapshots: List[u2.DynamicsDebugSnapshot] = []
    metadata = run_metadata or {}
    slice_name = str(metadata.get("slice_name", "synthetic"))
    run_prefix = str(metadata.get("run_id", "mock-run"))
    baseline_run_id = str(metadata.get("baseline_run_id", f"{run_prefix}-baseline"))
    rfl_run_id = str(metadata.get("rfl_run_id", f"{run_prefix}-rfl"))

    def _capture_snapshot(run_id: str, mode: str, records: list, pattern: Optional[str] = None):
        run_summary = {
            "run_id": run_id,
            "slice_name": slice_name,
            "mode": mode,
            "records": records,
            "comparison": {
                "baseline_records": baseline_records,
                "rfl_records": rfl_records,
            },
            "pattern": pattern,
        }
        debug_snapshots.append(u2.build_dynamics_debug_snapshot(run_summary, thresholds_struct))

    # 1. Data Extraction
    a_t_baseline = u2.estimate_A_t(baseline_records)
    a_t_rfl = u2.estimate_A_t(rfl_records)

    # 2. Experiment Validity Check
    baseline_pattern = u2.detect_pattern(a_t_baseline, baseline_records, thresholds)
    if baseline_pattern == "Stagnation" and a_t_baseline.mean() > 0.95:
        report["experiment_outcome"] = "DEGENERATE"
        report["experiment_summary"] = "Baseline run was degenerate (stagnant at >95% abstention)."
        for conj in conjectures_to_test:
            if conj not in report:
                report[conj] = {"status": "INCONCLUSIVE", "rationale": "Baseline was degenerate."}
        if debug_dynamics:
            _capture_snapshot(baseline_run_id, "baseline", baseline_records, baseline_pattern)
            _capture_snapshot(rfl_run_id, "rfl", rfl_records)
            _emit_dynamics_debug(debug_snapshots, debug_output_path)
        return report
    
    report["experiment_outcome"] = "VALID"

    # 3. Primary Metric Calculation
    uplift_results = u2.estimate_G_i(baseline_records, rfl_records)
    gain = uplift_results['delta']
    ci = (uplift_results['ci_95_lower'], uplift_results['ci_95_upper'])

    # 4. Behavioral Classification & Dynamics Metrics
    rfl_pattern = u2.detect_pattern(a_t_rfl, rfl_records, thresholds)
    psi = u2.estimate_policy_stability(rfl_records)
    omega = u2.estimate_oscillation_index(rfl_records)
    tau, _ = u2.kendalltau(a_t_rfl.index, a_t_rfl.values) if not a_t_rfl.empty else (np.nan, np.nan)


    report["experiment_summary"] = {
        "rfl_pattern_detected": rfl_pattern,
        "uplift_gain": gain,
        "uplift_gain_ci_95": list(ci),
        "dynamics_metrics": {
            "policy_stability_psi": psi,
            "policy_oscillation_omega": omega,
            "abstention_trend_tau": tau
        },
        "thresholds_used": thresholds
    }


    # 5. Conjecture Evaluation Loop
    for conj in conjectures_to_test:
        status = "NOT_TESTED"
        rationale = ""
        evidence_ref = report["experiment_summary"]["dynamics_metrics"]

        if conj == "Conjecture 3.1": # Supermartingale / Negative Drift
            if rfl_pattern in ["Negative Drift", "Logistic-like Decay"]:
                status = "SUPPORTS"
                rationale = f"RFL run exhibited '{rfl_pattern}' (tau={evidence_ref['abstention_trend_tau']:.3f}), a clear sign of negative drift."
            elif rfl_pattern in ["Stagnation", "Oscillation", "Irregular"]:
                status = "CONTRADICTS"
                rationale = f"RFL run showed '{rfl_pattern}', which lacks the required negative drift."
            else:
                status = "INCONCLUSIVE"
                rationale = "Data was insufficient to determine a clear pattern."

        elif conj == "Conjecture 4.1": # Logistic Decay
            if rfl_pattern == "Logistic-like Decay":
                status = "SUPPORTS"
                rationale = "The abstention rate curve fits the logistic decay pattern."
            elif rfl_pattern == "Negative Drift":
                status = "CONSISTENT"
                rationale = "Learning occurred (Negative Drift), but the curve was not definitively logistic."
            elif rfl_pattern in ["Stagnation", "Oscillation", "Irregular"]:
                status = "CONTRADICTS"
                rationale = f"Observed pattern '{rfl_pattern}' does not match logistic decay."
            else:
                status = "INCONCLUSIVE"

        elif conj == "Conjecture 6.1": # Convergence
            final_abstention = a_t_rfl.iloc[-1] if not a_t_rfl.empty else np.nan
            mean_abstention = float(a_t_rfl.mean()) if not a_t_rfl.empty else np.nan
            if rfl_pattern == 'Stagnation' and mean_abstention > 0.1:
                status = "CONTRADICTS"
                rationale = f"System plateaued at a high abstention rate of {mean_abstention:.2f}."
            elif rfl_pattern in ['Logistic-like Decay', 'Negative Drift'] and (mean_abstention < 0.5 or final_abstention < 0.3):
                status = "CONSISTENT"
                rationale = f"System shows a clear downward trend with mean abstention {mean_abstention:.2f}."
            elif final_abstention < 0.1:
                status = "CONSISTENT"
                rationale = f"Abstention terminated at {final_abstention:.2f}, indicating convergence."
            else:
                status = "INCONCLUSIVE"
                rationale = "Experiment may not have run long enough to test convergence."

        elif conj == "Phase II Uplift":
            if rfl_pattern == "Oscillation":
                status = "CONTRADICTS"
                rationale = "Oscillatory dynamics violate the stability requirement for uplift claims."
            elif gain > slice_uplift_threshold and ci[0] > 0:
                status = "SUPPORTS"
                rationale = f"Uplift gain of {gain:.3f} exceeded threshold {slice_uplift_threshold} with 95% CI {ci}."
            elif gain > 0 and ci[0] > 0:
                status = "CONSISTENT"
                rationale = f"Positive uplift gain of {gain:.3f} was observed, but did not meet threshold {slice_uplift_threshold}."
            else:
                status = "CONTRADICTS"
                rationale = f"Uplift gain of {gain:.3f} was not statistically significant or was negative."

        report[conj] = {
            "status": status,
            "rationale": rationale
        }

    if debug_dynamics:
        _capture_snapshot(baseline_run_id, "baseline", baseline_records, baseline_pattern)
        _capture_snapshot(rfl_run_id, "rfl", rfl_records, rfl_pattern)
        _emit_dynamics_debug(debug_snapshots, debug_output_path)

    return report


def generate_mock_data(scenario: str) -> tuple:
    """Generates mock data for different test scenarios."""
    scenario_seeds = {
        "degenerate": 11,
        "positive_logistic": 13,
        "null": 17,
        "instability": 19,
    }
    rng = np.random.default_rng(scenario_seeds.get(scenario, 23))
    cycles = np.arange(1, 201)
    # Baseline is always stagnant for these tests
    baseline_records = [
        {'cycle': int(c), 'metrics': {'abstention_rate': 0.8 + rng.normal(0, 0.005)}, 'policy': {'theta': [0.0, 0.0]}}
        for c in cycles
    ]
    rfl_abstention = np.zeros_like(cycles, dtype=float)

    if scenario == "degenerate":
        rfl_abstention = np.full_like(cycles, 0.98, dtype=float) + rng.normal(0, 0.005, 200)
        baseline_records = [
            {'cycle': int(c), 'metrics': {'abstention_rate': 0.98 + rng.normal(0, 0.005)}, 'policy': {'theta': [0.0, 0.0]}}
            for c in cycles
        ]
    elif scenario == "positive_logistic":
        # Explicit plateau + clean decay
        plateau = np.full(40, 0.8) + rng.normal(0, 0.005, 40)
        decay_cycles = np.arange(1, 161)
        decay = 0.2 + (0.8 - 0.2) / (1 + np.exp(0.05 * (decay_cycles - 80)))
        rfl_abstention = np.concatenate([plateau, decay])
    elif scenario == "null":
        rfl_abstention = np.full_like(cycles, 0.8, dtype=float) + rng.normal(0, 0.005, 200)
    elif scenario == "instability":
        # Clean sine wave for abstention data to avoid step detection
        rfl_abstention = 0.6 + 0.2 * np.sin(cycles / 10)

    # Mock policy data
    thetas = np.zeros((200, 2))
    if scenario == "instability":
        # Policy updates that clearly reverse direction to trigger high Omega
        for i in range(1, 200):
            direction = 1 if (i // 5) % 2 == 0 else -1
            thetas[i] = thetas[i-1] + direction * np.array([0.5, -0.5])
    else: 
        # Smoothly converging policy for non-instability cases
        thetas[0] = np.array([2.0, 2.0])
        for i in range(1, 200):
            thetas[i] = thetas[i-1] * 0.95 # Smooth decay to zero
            
    rfl_records = [
        {'cycle': int(c), 'metrics': {'abstention_rate': float(a)}, 'policy': {'theta': [float(th[0]), float(th[1])]}}
        for c, a, th in zip(cycles, rfl_abstention, thetas)
    ]
    
    return baseline_records, rfl_records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dynamics-Theory Unification Engine harness.")
    parser.add_argument(
        "--debug-dynamics",
        action="store_true",
        help="Emit per-run dynamics debug snapshots.",
    )
    parser.add_argument(
        "--debug-output",
        type=str,
        default=None,
        help="Optional path for the dynamics debug JSONL artifact.",
    )
    args = parser.parse_args()

    print("--- Running Dynamics-Theory Unification Engine ---")

    SLICE_THRESHOLD = 0.10
    CONJECTURES = [
        "Conjecture 3.1",
        "Conjecture 4.1",
        "Conjecture 6.1",
        "Phase II Uplift"
    ]

    print("\nAnalyzing 'Positive Uplift' scenario...")
    baseline_data, rfl_data = generate_mock_data("positive_logistic")
    thresholds = _load_thresholds()

    final_report = run_conjecture_analysis(
        baseline_records=baseline_data,
        rfl_records=rfl_data,
        slice_uplift_threshold=SLICE_THRESHOLD,
        conjectures_to_test=CONJECTURES,
        thresholds=thresholds,
        debug_dynamics=args.debug_dynamics,
        debug_output_path=args.debug_output,
        run_metadata={"run_id": "mock-positive-uplift", "slice_name": "synthetic"},
    )

    output_dir = "artifacts/dynamics"
    os.makedirs(output_dir, exist_ok=True)

    report_path = os.path.join(output_dir, "conjecture_report.json")
    with open(report_path, 'w', encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, sort_keys=True)

    print(f"\nSuccessfully generated conjecture report at: {report_path}")
    print("\n--- Report Summary ---")
    print(json.dumps(final_report, indent=2, sort_keys=True))
    print("\n--- Engine Finished ---")
