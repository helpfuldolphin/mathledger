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
U2 Dynamics Calibration Module (calibration.py)
================================================

This module provides functions for calibrating the analysis thresholds
based on the statistical properties of a given experimental run.

Author: Gemini M, Dynamics-Theory Unification Analyst
"""

import numpy as np
import pandas as pd
from . import u2_dynamics as u2

def generate_calibration_notes(records: list, thresholds: dict) -> dict:
    """
    Analyzes a given dataset to recommend adjustments to analysis thresholds.

    Args:
        records: A list of per-cycle records from an experimental run.
        thresholds: The currently configured thresholds.

    Returns:
        A dictionary containing calibration notes and recommendations.
    """
    notes = {}
    time_series = u2.estimate_A_t(records)
    
    if time_series.empty or len(time_series) < 20:
        return {"error": "Not enough data for calibration."}

    # --- Stagnation Threshold Analysis ---
    rolling_std = time_series.rolling(window=20).std().dropna()
    noise_floor_est = np.percentile(rolling_std, 10) if not rolling_std.empty else 0.0
    notes['stagnation_threshold'] = {
        "current": thresholds.get('stagnation_std_thresh'),
        "observed_10th_percentile_rolling_std": f"{noise_floor_est:.4f}",
        "recommendation": f"Consider setting stagnation_std_thresh slightly above the observed noise floor, e.g., {noise_floor_est * 1.5:.4f}."
    }

    # --- Trend Threshold Analysis ---
    tau, _ = u2.kendalltau(time_series.index, time_series.values)
    notes['trend_threshold'] = {
        "current": thresholds.get('trend_tau_thresh'),
        "observed_kendall_tau": f"{tau:.4f}",
        "recommendation": "A strong negative trend will have tau < -0.5. A value of -0.2 is sensitive to weak trends. Adjust based on desired sensitivity."
    }

    # --- Oscillation Threshold Analysis ---
    if 'policy' in records[0]:
        omega = u2.estimate_oscillation_index(records)
        thetas = np.array([r['policy']['theta'] for r in records if 'policy' in r and 'theta' in r['policy']])
        if len(thetas) > 2:
            deltas = thetas[1:] - thetas[:-1]
            dot_products = np.sum(deltas[1:] * deltas[:-1], axis=1)
            notes['oscillation_threshold'] = {
                "current": thresholds.get('oscillation_omega_thresh'),
                "observed_omega": f"{omega:.4f}",
                "dot_product_25th_percentile": f"{np.percentile(dot_products, 25):.4f}",
                "recommendation": "A value of 0.3 implies 30% of updates are reversals. If the dot product percentile is close to zero, even a lower Omega might indicate instability. Adjust based on how strictly 'reversal' is defined."
            }

    # --- Step Size Threshold Analysis ---
    diffs = time_series.diff().abs().dropna()
    if not diffs.empty:
        notes['step_size_threshold'] = {
            "current": thresholds.get('step_size_thresh'),
            "observed_99th_percentile_jump": f"{np.percentile(diffs, 99):.4f}",
            "observed_max_jump": f"{diffs.max():.4f}",
            "recommendation": "Set step_size_thresh below the max jump but above the 99th percentile to detect only true outlier events."
        }
        
    return notes
