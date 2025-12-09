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
Tests for the Dynamics-Theory Unification Engine.
"""
import pytest
import json
from analysis.conjecture_engine import run_conjecture_analysis, generate_mock_data

# --- Test Fixtures ---

@pytest.fixture(scope="module")
def thresholds():
    """Loads the dynamics thresholds from the config file."""
    with open('config/dynamics_thresholds.json', 'r') as f:
        return json.load(f)

@pytest.fixture(scope="module")
def conjectures():
    """Provides the standard list of conjectures to test."""
    return ["Conjecture 3.1", "Conjecture 4.1", "Conjecture 6.1", "Phase II Uplift"]

@pytest.fixture
def degenerate_data():
    return generate_mock_data("degenerate")

@pytest.fixture
def positive_logistic_data():
    return generate_mock_data("positive_logistic")

@pytest.fixture
def null_data():
    return generate_mock_data("null")

@pytest.fixture
def instability_data():
    return generate_mock_data("instability")


# --- Test Cases ---

def test_degenerate_scenario(degenerate_data, conjectures, thresholds):
    """
    Tests that the engine correctly identifies a degenerate experiment.
    """
    baseline_records, rfl_records = degenerate_data
    report = run_conjecture_analysis(baseline_records, rfl_records, 0.1, conjectures, thresholds)

    assert report["experiment_outcome"] == "DEGENERATE"
    assert report["Conjecture 3.1"]["status"] == "INCONCLUSIVE"

def test_positive_logistic_scenario(positive_logistic_data, conjectures, thresholds):
    """
    Tests a positive uplift scenario with logistic decay.
    """
    baseline_records, rfl_records = positive_logistic_data
    report = run_conjecture_analysis(baseline_records, rfl_records, 0.1, conjectures, thresholds)

    assert report["experiment_outcome"] == "VALID"
    
    summary = report["experiment_summary"]
    assert summary["rfl_pattern_detected"] == "Logistic-like Decay"
    assert summary["uplift_gain"] > 0.1
    assert summary["dynamics_metrics"]["policy_oscillation_omega"] < thresholds["oscillation_omega_thresh"]

    assert report["Phase II Uplift"]["status"] == "SUPPORTS"
    assert report["Conjecture 3.1"]["status"] == "SUPPORTS"
    assert report["Conjecture 4.1"]["status"] == "SUPPORTS"
    assert report["Conjecture 6.1"]["status"] == "CONSISTENT"


def test_null_scenario(null_data, conjectures, thresholds):
    """
    Tests a null result where RFL provides no benefit.
    """
    baseline_records, rfl_records = null_data
    report = run_conjecture_analysis(baseline_records, rfl_records, 0.1, conjectures, thresholds)

    assert report["experiment_outcome"] == "VALID"
    
    summary = report["experiment_summary"]
    assert summary["rfl_pattern_detected"] == "Stagnation"
    assert summary["uplift_gain"] < 0.1

    assert report["Phase II Uplift"]["status"] == "CONTRADICTS"
    assert report["Conjecture 3.1"]["status"] == "CONTRADICTS"

def test_instability_scenario(instability_data, conjectures, thresholds):
    """
    Tests a scenario with an unstable, oscillating policy.
    """
    baseline_records, rfl_records = instability_data
    report = run_conjecture_analysis(baseline_records, rfl_records, 0.1, conjectures, thresholds)

    assert report["experiment_outcome"] == "VALID"
    
    summary = report["experiment_summary"]
    assert summary["rfl_pattern_detected"] == "Oscillation"
    assert summary["dynamics_metrics"]["policy_oscillation_omega"] > thresholds["oscillation_omega_thresh"]
    
    assert report["Conjecture 3.1"]["status"] == "CONTRADICTS"
    assert report["Phase II Uplift"]["status"] == "CONTRADICTS"