# Copyright 2025 MathLedger
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may not use this file except in compliance with the License.
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
Tests for the Uplift Governance and Cross-Run Analysis Module.
"""
import pytest
from analysis.governance import (
    summarize_conjecture_status_for_global_health,
    combine_conjectures_with_governance,
    build_conjecture_timeline
)

# --- Mock Data Fixtures ---

@pytest.fixture
def healthy_snapshot():
    """A snapshot representing a healthy learning state."""
    return {
        "Phase II Uplift": {"status": "SUPPORTS"},
        "Conjecture 3.1": {"status": "SUPPORTS"},
        "Conjecture 4.1": {"status": "SUPPORTS"},
    }

@pytest.fixture
def unhealthy_snapshot():
    """A snapshot where a key conjecture is contradicted."""
    return {
        "Phase II Uplift": {"status": "CONTRADICTS"},
        "Conjecture 3.1": {"status": "SUPPORTS"},
        "Conjecture 4.1": {"status": "CONSISTENT"},
    }

@pytest.fixture
def mixed_snapshot():
    """A snapshot with mixed but not critical results."""
    return {
        "Phase II Uplift": {"status": "SUPPORTS"},
        "Conjecture 3.1": {"status": "CONSISTENT"},
        "Conjecture 4.1": {"status": "CONTRADICTS"}, # Non-key conjecture
    }

@pytest.fixture
def inconclusive_snapshot():
    """A snapshot where results are mostly inconclusive."""
    return {
        "Phase II Uplift": {"status": "INCONCLUSIVE"},
        "Conjecture 3.1": {"status": "INCONCLUSIVE"},
        "Conjecture 4.1": {"status": "CONSISTENT"},
    }

@pytest.fixture
def snapshot_timeline_data():
    """A sequence of snapshots representing a project's history."""
    return [
        # Run 1: Early, inconclusive results
        {
            "Phase II Uplift": {"status": "INCONCLUSIVE"},
            "Conjecture 3.1": {"status": "CONSISTENT"},
        },
        # Run 2: Healthy progress
        {
            "Phase II Uplift": {"status": "SUPPORTS"},
            "Conjecture 3.1": {"status": "SUPPORTS"},
        },
        # Run 3: A regression occurs
        {
            "Phase II Uplift": {"status": "SUPPORTS"},
            "Conjecture 3.1": {"status": "CONTRADICTS"},
        },
        # Run 4: The regression is fixed
        {
            "Phase II Uplift": {"status": "SUPPORTS"},
            "Conjecture 3.1": {"status": "SUPPORTS"},
        },
        # Run 5: Another regression
        {
            "Phase II Uplift": {"status": "CONTRADICTS"},
            "Conjecture 3.1": {"status": "CONTRADICTS"},
        }
    ]

# --- Tests for Task 3: Global Health Signal ---

def test_global_health_healthy(healthy_snapshot):
    summary = summarize_conjecture_status_for_global_health(healthy_snapshot)
    assert summary["learning_health"] == "HEALTHY"
    assert summary["any_key_conjecture_contradicted"] is False
    assert summary["supports_vs_contradicts_ratio"] == float('inf')

def test_global_health_unhealthy(unhealthy_snapshot):
    summary = summarize_conjecture_status_for_global_health(unhealthy_snapshot)
    assert summary["learning_health"] == "UNHEALTHY"
    assert summary["any_key_conjecture_contradicted"] is True
    assert summary["supports_vs_contradicts_ratio"] == 1.0

def test_global_health_mixed(mixed_snapshot):
    summary = summarize_conjecture_status_for_global_health(mixed_snapshot)
    assert summary["learning_health"] == "MIXED"
    assert summary["any_key_conjecture_contradicted"] is False
    assert summary["supports_vs_contradicts_ratio"] == 1.0

def test_global_health_inconclusive(inconclusive_snapshot):
    summary = summarize_conjecture_status_for_global_health(inconclusive_snapshot)
    assert summary["learning_health"] == "INCONCLUSIVE"
    assert summary["any_key_conjecture_contradicted"] is False

# --- Tests for Task 1: Governance Integration ---

def test_governance_integration_ok(healthy_snapshot):
    gov_posture = {"status": "OK", "gates": {"U1_complete": "passed"}}
    result = combine_conjectures_with_governance(gov_posture, healthy_snapshot)
    assert result["governance_blocking"] is False
    assert result["dynamics_status"] == "OK"
    assert result["uplift_readiness_flag"] is True

def test_governance_integration_dynamics_warn(mixed_snapshot):
    gov_posture = {"status": "OK", "gates": {"U1_complete": "passed"}}
    result = combine_conjectures_with_governance(gov_posture, mixed_snapshot)
    assert result["governance_blocking"] is False
    assert result["dynamics_status"] == "WARN"
    assert result["uplift_readiness_flag"] is False

def test_governance_integration_dynamics_attention(unhealthy_snapshot):
    gov_posture = {"status": "OK", "gates": {"U1_complete": "passed"}}
    result = combine_conjectures_with_governance(gov_posture, unhealthy_snapshot)
    assert result["governance_blocking"] is False
    assert result["dynamics_status"] == "ATTENTION"
    assert result["uplift_readiness_flag"] is False

def test_governance_integration_gov_blocking(healthy_snapshot):
    gov_posture = {"status": "OK", "gates": {"U1_complete": "failed"}}
    result = combine_conjectures_with_governance(gov_posture, healthy_snapshot)
    assert result["governance_blocking"] is True
    assert result["dynamics_status"] == "OK"
    assert result["uplift_readiness_flag"] is False

def test_governance_integration_both_blocking(unhealthy_snapshot):
    gov_posture = {"status": "ATTENTION", "gates": {"U1_complete": "failed"}}
    result = combine_conjectures_with_governance(gov_posture, unhealthy_snapshot)
    assert result["governance_blocking"] is True
    # Dynamics status is "ATTENTION" because it's the higher priority failure
    assert result["dynamics_status"] == "ATTENTION"
    assert result["uplift_readiness_flag"] is False

# --- Tests for Task 2: Conjecture History Timeline ---

def test_conjecture_timeline(snapshot_timeline_data):
    timeline = build_conjecture_timeline(snapshot_timeline_data)
    
    # Test evolution tracking
    assert timeline["conjecture_evolution"]["Phase II Uplift"] == [
        "INCONCLUSIVE", "SUPPORTS", "SUPPORTS", "SUPPORTS", "CONTRADICTS"
    ]
    assert timeline["conjecture_evolution"]["Conjecture 3.1"] == [
        "CONSISTENT", "SUPPORTS", "CONTRADICTS", "SUPPORTS", "CONTRADICTS"
    ]

    # Test transition counts
    assert timeline["transition_metrics"]["Phase II Uplift"]["to_CONTRADICTS"] == 1
    assert timeline["transition_metrics"]["Conjecture 3.1"]["to_CONTRADICTS"] == 2
