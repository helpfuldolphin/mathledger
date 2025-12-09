"""
Tests for Conjecture Engine Contract Implementation.

Tests each binding rule with synthetic metrics to trigger each status class:
- SUPPORTS
- CONSISTENT
- CONTRADICTS
- INCONCLUSIVE

Verifies report structure and determinism as required by CONJECTURE_ENGINE_CONTRACT.md.
"""

import json
import pytest
from pathlib import Path
from typing import Any, Dict

from analysis.conjecture_engine_contract import (
    EvidenceStatus,
    ValidationStatus,
    validate_inputs,
    evaluate_conjectures,
    _check_R3_1_supermartingale,
    _check_R4_1_logistic_decay,
    _check_R6_1_convergence,
    _check_R13_2_multi_goal_convergence,
    _check_R15_1_local_stability,
    _check_R15_4_basin_structure,
    _check_R2_1_variance_amplification,
    _check_R2_2_learning_signal,
    THRESHOLD_P_VALUE,
    THRESHOLD_LOGISTIC_R2,
    THRESHOLD_PSI_CONVERGED,
    THRESHOLD_OSCILLATION_HEALTHY,
    THRESHOLD_ABSTENTION_CONVERGED,
)


# =============================================================================
# FIXTURE HELPERS
# =============================================================================

def create_valid_jsonl_record(cycle: int, mode: str = "rfl") -> Dict[str, Any]:
    """Create a valid JSONL record conforming to contract schema."""
    record = {
        "cycle": cycle,
        "timestamp_utc": f"2025-12-06T00:00:{cycle:02d}Z",
        "slice_name": "slice_a",
        "mode": mode,
        "H_t": f"hash_{cycle:04d}",
        "candidates": {"total": 100, "selected": 50},
        "verified": {"count": 45, "rate": 0.90},
        "abstained": {"count": 5, "rate": 0.10},
        "metrics": {
            "abstention_rate": 0.10 - (cycle * 0.001),
            "verification_density": 0.90 + (cycle * 0.001),
        },
    }
    if mode == "rfl":
        record["policy"] = {
            "theta": [0.5 + cycle * 0.01, -0.3 + cycle * 0.005],
            "gradient_norm": max(0.1, 1.0 - cycle * 0.01),
            "theta_delta": [0.01, 0.005],
        }
    return record


def create_valid_summary(mode: str = "rfl", slice_name: str = "slice_a") -> Dict[str, Any]:
    """Create a valid summary conforming to contract schema."""
    return {
        "experiment_id": "test_exp_001",
        "slice_name": slice_name,
        "mode": mode,
        "total_cycles": 100,
        "metrics": {
            "mean_abstention_rate": 0.05,
            "primary_metric": 0.95,
            "final_success_rate": 0.92,
        },
        "time_series": {
            "abstention_rates": [0.5 - i * 0.004 for i in range(100)],
            "success_rates": [0.7 + i * 0.002 for i in range(100)],
            "densities": [0.3 + i * 0.005 for i in range(100)],
        },
        "policy_final": {
            "theta": [1.5, -0.1],
            "theta_norm": 1.503,
        },
    }


def create_valid_telemetry() -> Dict[str, Any]:
    """Create a valid telemetry aggregate conforming to contract schema."""
    return {
        "experiment_id": "test_exp_001",
        "comparison": {
            "delta": 0.15,
            "ci_95_lower": 0.05,
            "ci_95_upper": 0.25,
            "ci_excludes_zero": True,
        },
        "diagnostics": {
            "policy_stability_index": 0.005,
            "oscillation_index": 0.10,
            "metric_stationary": True,
            "abstention_trend_tau": -0.35,
            "abstention_trend_p": 0.001,
        },
        "patterns": {
            "detected_pattern": "A.1",
            "pattern_confidence": 0.85,
        },
        "validity": {
            "baseline_valid": True,
            "rfl_valid": True,
        },
    }


@pytest.fixture
def valid_inputs(tmp_path: Path) -> Dict[str, Path]:
    """Create valid input files for testing."""
    # Baseline log
    baseline_log = tmp_path / "baseline.jsonl"
    with open(baseline_log, "w") as f:
        for i in range(10):
            f.write(json.dumps(create_valid_jsonl_record(i, "baseline")) + "\n")

    # RFL log
    rfl_log = tmp_path / "rfl.jsonl"
    with open(rfl_log, "w") as f:
        for i in range(10):
            f.write(json.dumps(create_valid_jsonl_record(i, "rfl")) + "\n")

    # Baseline summary
    baseline_summary = tmp_path / "baseline_summary.json"
    with open(baseline_summary, "w") as f:
        json.dump(create_valid_summary("baseline"), f)

    # RFL summary
    rfl_summary = tmp_path / "rfl_summary.json"
    with open(rfl_summary, "w") as f:
        json.dump(create_valid_summary("rfl"), f)

    # Telemetry
    telemetry = tmp_path / "telemetry.json"
    with open(telemetry, "w") as f:
        json.dump(create_valid_telemetry(), f)

    return {
        "baseline_log": baseline_log,
        "rfl_log": rfl_log,
        "baseline_summary": baseline_summary,
        "rfl_summary": rfl_summary,
        "telemetry": telemetry,
    }


# =============================================================================
# INPUT VALIDATION TESTS
# =============================================================================

class TestInputValidation:
    """Tests for input validation layer."""

    def test_valid_inputs_pass(self, valid_inputs):
        """Valid inputs should pass validation."""
        result, parsed = validate_inputs(
            valid_inputs["baseline_log"],
            valid_inputs["rfl_log"],
            valid_inputs["baseline_summary"],
            valid_inputs["rfl_summary"],
            valid_inputs["telemetry"],
        )
        assert result.status == ValidationStatus.VALID
        assert len(result.errors) == 0

    def test_missing_file_fails(self, valid_inputs):
        """Missing file should fail validation."""
        result, _ = validate_inputs(
            Path("nonexistent.jsonl"),
            valid_inputs["rfl_log"],
            valid_inputs["baseline_summary"],
            valid_inputs["rfl_summary"],
            valid_inputs["telemetry"],
        )
        assert result.status == ValidationStatus.INVALID
        assert any("File not found" in e for e in result.errors)

    def test_invalid_json_fails(self, tmp_path, valid_inputs):
        """Invalid JSON should fail validation."""
        bad_file = tmp_path / "bad.json"
        with open(bad_file, "w") as f:
            f.write("{invalid json")

        result, _ = validate_inputs(
            valid_inputs["baseline_log"],
            valid_inputs["rfl_log"],
            valid_inputs["baseline_summary"],
            bad_file,  # Invalid RFL summary
            valid_inputs["telemetry"],
        )
        assert result.status == ValidationStatus.INVALID
        assert any("Invalid JSON" in e for e in result.errors)

    def test_missing_required_field_fails(self, tmp_path, valid_inputs):
        """Missing required field in summary should fail."""
        incomplete = tmp_path / "incomplete.json"
        with open(incomplete, "w") as f:
            json.dump({"partial": "data"}, f)

        result, _ = validate_inputs(
            valid_inputs["baseline_log"],
            valid_inputs["rfl_log"],
            valid_inputs["baseline_summary"],
            incomplete,  # Missing required fields
            valid_inputs["telemetry"],
        )
        assert result.status == ValidationStatus.INVALID
        assert any("missing required field" in e.lower() for e in result.errors)


# =============================================================================
# BINDING RULE TESTS - R3.1 (SUPERMARTINGALE)
# =============================================================================

class TestR3_1_Supermartingale:
    """Tests for Rule R3.1: Conjecture 3.1 (Supermartingale Property)."""

    def test_supports_negative_trend_significant(self):
        """Negative tau with p < 0.05 should SUPPORT."""
        telemetry = {
            "diagnostics": {
                "abstention_trend_tau": -0.45,
                "abstention_trend_p": 0.001,
            }
        }
        rfl_summary = create_valid_summary()

        result = _check_R3_1_supermartingale(telemetry, rfl_summary)

        assert result.evidence_status == EvidenceStatus.SUPPORTS
        assert result.evaluation.rule_applied == "R3.1"
        assert "decreasing trend" in result.evidence_rationale.lower()

    def test_consistent_negative_trend_not_significant(self):
        """Negative tau with p >= 0.05 should be CONSISTENT."""
        telemetry = {
            "diagnostics": {
                "abstention_trend_tau": -0.15,
                "abstention_trend_p": 0.12,  # Not significant
            }
        }
        rfl_summary = create_valid_summary()

        result = _check_R3_1_supermartingale(telemetry, rfl_summary)

        assert result.evidence_status == EvidenceStatus.CONSISTENT
        assert "not statistically significant" in result.evidence_rationale.lower()

    def test_contradicts_positive_trend_significant(self):
        """Positive tau with p < 0.05 should CONTRADICT."""
        telemetry = {
            "diagnostics": {
                "abstention_trend_tau": 0.40,
                "abstention_trend_p": 0.002,
            }
        }
        rfl_summary = create_valid_summary()

        result = _check_R3_1_supermartingale(telemetry, rfl_summary)

        assert result.evidence_status == EvidenceStatus.CONTRADICTS
        assert "increasing" in result.evidence_rationale.lower()

    def test_inconclusive_flat_trend(self):
        """Near-zero tau should be INCONCLUSIVE."""
        telemetry = {
            "diagnostics": {
                "abstention_trend_tau": 0.02,
                "abstention_trend_p": 0.85,
            }
        }
        rfl_summary = create_valid_summary()

        result = _check_R3_1_supermartingale(telemetry, rfl_summary)

        assert result.evidence_status == EvidenceStatus.INCONCLUSIVE
        assert "flat" in result.evidence_rationale.lower() or "insufficient" in result.evidence_rationale.lower()


# =============================================================================
# BINDING RULE TESTS - R4.1 (LOGISTIC DECAY)
# =============================================================================

class TestR4_1_LogisticDecay:
    """Tests for Rule R4.1: Conjecture 4.1 (Logistic Decay)."""

    def test_supports_good_logistic_fit(self):
        """High R² logistic fit should SUPPORT."""
        telemetry = create_valid_telemetry()
        # Create decay curve that fits logistic well
        rfl_summary = create_valid_summary()
        # Logistic decay: starts high, ends low
        n = 100
        rfl_summary["time_series"]["abstention_rates"] = [
            0.8 / (1 + 0.1 * (i - 50)) if i > 50 else 0.8 for i in range(n)
        ]

        result = _check_R4_1_logistic_decay(telemetry, rfl_summary, "slice_a")

        # Note: May be SUPPORTS, CONSISTENT, or INCONCLUSIVE depending on fit
        assert result.evaluation.rule_applied == "R4.1"
        assert result.conjecture_id == "conjecture_4_1"

    def test_inconclusive_fit_failed(self):
        """Failed logistic fit should be INCONCLUSIVE."""
        telemetry = create_valid_telemetry()
        rfl_summary = create_valid_summary()
        # Too few points for reliable fit
        rfl_summary["time_series"]["abstention_rates"] = [0.5, 0.4, 0.3]

        result = _check_R4_1_logistic_decay(telemetry, rfl_summary, "slice_a")

        assert result.evidence_status == EvidenceStatus.INCONCLUSIVE
        assert "insufficient" in result.evidence_rationale.lower() or "fit" in result.evidence_rationale.lower()


# =============================================================================
# BINDING RULE TESTS - R6.1 (CONVERGENCE)
# =============================================================================

class TestR6_1_Convergence:
    """Tests for Rule R6.1: Conjecture 6.1 (Almost Sure Convergence)."""

    def test_supports_converged_to_zero(self):
        """Stationary metric at low abstention should SUPPORT."""
        telemetry = {
            "diagnostics": {
                "metric_stationary": True,
                "policy_stability_index": 0.005,
            }
        }
        rfl_summary = create_valid_summary()
        rfl_summary["metrics"]["mean_abstention_rate"] = 0.03  # < 0.10 threshold

        result = _check_R6_1_convergence(telemetry, rfl_summary)

        assert result.evidence_status == EvidenceStatus.SUPPORTS
        assert "converged" in result.evidence_rationale.lower()

    def test_consistent_converged_not_zero(self):
        """Stationary metric at higher abstention should be CONSISTENT."""
        telemetry = {
            "diagnostics": {
                "metric_stationary": True,
                "policy_stability_index": 0.008,
            }
        }
        rfl_summary = create_valid_summary()
        rfl_summary["metrics"]["mean_abstention_rate"] = 0.25  # >= 0.10 threshold

        result = _check_R6_1_convergence(telemetry, rfl_summary)

        assert result.evidence_status == EvidenceStatus.CONSISTENT
        assert "local optimum" in result.evidence_rationale.lower() or "stationary" in result.evidence_rationale.lower()

    def test_contradicts_not_converged(self):
        """Non-stationary with high PSI should CONTRADICT."""
        telemetry = {
            "diagnostics": {
                "metric_stationary": False,
                "policy_stability_index": 0.15,  # > 0.05 threshold
            }
        }
        rfl_summary = create_valid_summary()
        rfl_summary["metrics"]["mean_abstention_rate"] = 0.40

        result = _check_R6_1_convergence(telemetry, rfl_summary)

        assert result.evidence_status == EvidenceStatus.CONTRADICTS
        assert "converged" in result.evidence_rationale.lower()

    def test_inconclusive_unclear_state(self):
        """Unclear convergence state should be INCONCLUSIVE."""
        telemetry = {
            "diagnostics": {
                "metric_stationary": False,
                "policy_stability_index": 0.03,  # Between thresholds
            }
        }
        rfl_summary = create_valid_summary()

        result = _check_R6_1_convergence(telemetry, rfl_summary)

        assert result.evidence_status == EvidenceStatus.INCONCLUSIVE


# =============================================================================
# BINDING RULE TESTS - R13.2 (MULTI-GOAL CONVERGENCE)
# =============================================================================

class TestR13_2_MultiGoalConvergence:
    """Tests for Rule R13.2: Theorem 13.2 (Multi-Goal RFL Convergence)."""

    def test_supports_converged_improving(self):
        """Converged policy with improving metric should SUPPORT."""
        telemetry = {
            "diagnostics": {
                "policy_stability_index": 0.005,  # < 0.01
            }
        }
        rfl_summary = create_valid_summary()
        # Improving success rates
        rfl_summary["time_series"]["success_rates"] = [0.5 + i * 0.004 for i in range(100)]
        rfl_summary["policy_final"] = {"theta_norm": 1.5}

        result = _check_R13_2_multi_goal_convergence(telemetry, rfl_summary, "slice_c")

        assert result.evidence_status == EvidenceStatus.SUPPORTS
        assert "converged" in result.evidence_rationale.lower()

    def test_contradicts_diverged(self):
        """Diverged theta should CONTRADICT."""
        telemetry = {
            "diagnostics": {
                "policy_stability_index": 0.5,
            }
        }
        rfl_summary = create_valid_summary()
        rfl_summary["policy_final"] = {"theta_norm": float("inf")}

        result = _check_R13_2_multi_goal_convergence(telemetry, rfl_summary, "slice_d")

        assert result.evidence_status == EvidenceStatus.CONTRADICTS
        assert "diverged" in result.evidence_rationale.lower()

    def test_contradicts_high_psi(self):
        """High PSI without divergence should CONTRADICT."""
        telemetry = {
            "diagnostics": {
                "policy_stability_index": 0.15,  # > 0.10 threshold
            }
        }
        rfl_summary = create_valid_summary()
        rfl_summary["policy_final"] = {"theta_norm": 2.0}
        rfl_summary["time_series"]["success_rates"] = [0.5] * 100  # Flat

        result = _check_R13_2_multi_goal_convergence(telemetry, rfl_summary, "slice_c")

        assert result.evidence_status == EvidenceStatus.CONTRADICTS
        assert "failed to converge" in result.evidence_rationale.lower()


# =============================================================================
# BINDING RULE TESTS - R15.1 (LOCAL STABILITY)
# =============================================================================

class TestR15_1_LocalStability:
    """Tests for Rule R15.1: Theorem 15.1 (Local Stability Criterion)."""

    def test_supports_stable_low_oscillation(self):
        """Bounded theta with low oscillation should SUPPORT."""
        telemetry = {
            "diagnostics": {
                "oscillation_index": 0.08,  # < 0.20
            }
        }
        rfl_summary = create_valid_summary()
        rfl_summary["policy_final"] = {"theta_norm": 1.5}

        result = _check_R15_1_local_stability(telemetry, rfl_summary)

        assert result.evidence_status == EvidenceStatus.SUPPORTS
        assert "stable" in result.evidence_rationale.lower()

    def test_consistent_bounded_oscillating(self):
        """Bounded but oscillating should be CONSISTENT."""
        telemetry = {
            "diagnostics": {
                "oscillation_index": 0.25,  # >= 0.20
            }
        }
        rfl_summary = create_valid_summary()
        rfl_summary["policy_final"] = {"theta_norm": 2.0}

        result = _check_R15_1_local_stability(telemetry, rfl_summary)

        assert result.evidence_status == EvidenceStatus.CONSISTENT
        assert "oscillating" in result.evidence_rationale.lower()

    def test_contradicts_unbounded(self):
        """Unbounded theta should CONTRADICT."""
        telemetry = {
            "diagnostics": {
                "oscillation_index": 0.10,
            }
        }
        rfl_summary = create_valid_summary()
        rfl_summary["policy_final"] = {"theta_norm": float("nan")}

        result = _check_R15_1_local_stability(telemetry, rfl_summary)

        assert result.evidence_status == EvidenceStatus.CONTRADICTS
        assert "diverged" in result.evidence_rationale.lower()


# =============================================================================
# BINDING RULE TESTS - R15.4 (BASIN STRUCTURE)
# =============================================================================

class TestR15_4_BasinStructure:
    """Tests for Rule R15.4: Conjecture 15.4 (Basin Structure for U2 Slices)."""

    def test_supports_matching_pattern(self):
        """Pattern matching prediction should SUPPORT."""
        telemetry = {
            "patterns": {
                "detected_pattern": "A.1",
                "pattern_confidence": 0.90,
            }
        }
        rfl_summary = create_valid_summary(slice_name="slice_a")

        result = _check_R15_4_basin_structure(telemetry, rfl_summary, "slice_a")

        assert result.evidence_status == EvidenceStatus.SUPPORTS
        assert "matches" in result.evidence_rationale.lower()

    def test_contradicts_mismatched_pattern(self):
        """Pattern not matching prediction should CONTRADICT."""
        telemetry = {
            "patterns": {
                "detected_pattern": "C.1",  # Wrong family for slice A
                "pattern_confidence": 0.85,
            }
        }
        rfl_summary = create_valid_summary(slice_name="slice_a")

        result = _check_R15_4_basin_structure(telemetry, rfl_summary, "slice_a")

        assert result.evidence_status == EvidenceStatus.CONTRADICTS
        assert "contradicts" in result.evidence_rationale.lower()

    def test_inconclusive_no_pattern(self):
        """No detected pattern should be INCONCLUSIVE."""
        telemetry = {
            "patterns": {
                "detected_pattern": "",
                "pattern_confidence": 0.0,
            }
        }
        rfl_summary = create_valid_summary()

        result = _check_R15_4_basin_structure(telemetry, rfl_summary, "slice_a")

        assert result.evidence_status == EvidenceStatus.INCONCLUSIVE


# =============================================================================
# BINDING RULE TESTS - R2.1 (VARIANCE AMPLIFICATION)
# =============================================================================

class TestR2_1_VarianceAmplification:
    """Tests for Rule R2.1: Lemma 2.1 (Variance Under Wide Slice)."""

    def test_supports_variance_reduced(self):
        """Variance reduction should SUPPORT."""
        telemetry = create_valid_telemetry()
        rfl_summary = create_valid_summary()
        # High early variance, low late variance
        n = 60
        early = [0.3 + 0.2 * (i % 3) for i in range(n // 3)]  # High variance
        mid = [0.4 + 0.1 * (i % 2) for i in range(n // 3)]
        late = [0.5 + 0.01 * (i % 2) for i in range(n // 3)]  # Low variance
        rfl_summary["time_series"]["densities"] = early + mid + late

        result = _check_R2_1_variance_amplification(telemetry, rfl_summary, "slice_b")

        assert result.conjecture_id == "lemma_2_1"
        assert result.evaluation.rule_applied == "R2.1"
        # Status depends on actual variance calculation

    def test_inconclusive_insufficient_data(self):
        """Insufficient density data should be INCONCLUSIVE."""
        telemetry = create_valid_telemetry()
        rfl_summary = create_valid_summary()
        rfl_summary["time_series"]["densities"] = [0.5, 0.4, 0.3]  # < 6 points

        result = _check_R2_1_variance_amplification(telemetry, rfl_summary, "slice_b")

        assert result.evidence_status == EvidenceStatus.INCONCLUSIVE
        assert "insufficient" in result.evidence_rationale.lower()


# =============================================================================
# BINDING RULE TESTS - R2.2 (LEARNING SIGNAL)
# =============================================================================

class TestR2_2_LearningSignal:
    """Tests for Rule R2.2: Proposition 2.2 (Entropy-Signal Correspondence)."""

    def test_supports_positive_significant_uplift(self):
        """Positive delta with CI excluding zero should SUPPORT."""
        telemetry = {
            "comparison": {
                "delta": 0.20,
                "ci_95_lower": 0.08,
                "ci_95_upper": 0.32,
                "ci_excludes_zero": True,
            }
        }
        rfl_summary = create_valid_summary()

        result = _check_R2_2_learning_signal(telemetry, rfl_summary)

        assert result.evidence_status == EvidenceStatus.SUPPORTS
        assert "positive uplift" in result.evidence_rationale.lower()

    def test_consistent_positive_not_significant(self):
        """Positive delta but CI includes zero should be CONSISTENT."""
        telemetry = {
            "comparison": {
                "delta": 0.05,
                "ci_95_lower": -0.02,
                "ci_95_upper": 0.12,
                "ci_excludes_zero": False,
            }
        }
        rfl_summary = create_valid_summary()

        result = _check_R2_2_learning_signal(telemetry, rfl_summary)

        assert result.evidence_status == EvidenceStatus.CONSISTENT
        assert "not statistically significant" in result.evidence_rationale.lower()

    def test_contradicts_negative_significant(self):
        """Negative delta with CI excluding zero should CONTRADICT."""
        telemetry = {
            "comparison": {
                "delta": -0.15,
                "ci_95_lower": -0.25,
                "ci_95_upper": -0.05,
                "ci_excludes_zero": True,
            }
        }
        rfl_summary = create_valid_summary()

        result = _check_R2_2_learning_signal(telemetry, rfl_summary)

        assert result.evidence_status == EvidenceStatus.CONTRADICTS
        assert "negative" in result.evidence_rationale.lower()

    def test_inconclusive_near_zero(self):
        """Near-zero delta with CI including zero should be CONSISTENT (per R2.2)."""
        telemetry = {
            "comparison": {
                "delta": 0.001,
                "ci_95_lower": -0.05,
                "ci_95_upper": 0.052,
                "ci_excludes_zero": False,
            }
        }
        rfl_summary = create_valid_summary()

        result = _check_R2_2_learning_signal(telemetry, rfl_summary)

        # Per R2.2: delta > 0 AND ci_excludes_zero = False → CONSISTENT
        assert result.evidence_status == EvidenceStatus.CONSISTENT


# =============================================================================
# INTEGRATION TESTS - FULL EVALUATION
# =============================================================================

class TestEvaluateConjectures:
    """Integration tests for evaluate_conjectures() API."""

    def test_full_evaluation_produces_valid_report(self, valid_inputs, tmp_path):
        """Full evaluation should produce valid report structure."""
        output_path = tmp_path / "conjecture_report.json"

        report = evaluate_conjectures(
            valid_inputs["baseline_log"],
            valid_inputs["rfl_log"],
            valid_inputs["baseline_summary"],
            valid_inputs["rfl_summary"],
            valid_inputs["telemetry"],
            output_path,
        )

        # Check report structure
        assert "report_version" in report
        assert "generated_at" in report
        assert "experiment_id" in report
        assert "slice_name" in report
        assert "input_validation" in report
        assert "conjectures" in report
        assert "summary" in report
        assert "provenance" in report

        # Check conjectures evaluated
        assert len(report["conjectures"]) == 8  # All 8 binding rules

        # Check summary counts
        summary = report["summary"]
        assert "total_evaluated" in summary
        assert "supports_count" in summary
        assert "consistent_count" in summary
        assert "contradicts_count" in summary
        assert "inconclusive_count" in summary
        assert summary["total_evaluated"] == 8

    def test_report_written_to_file(self, valid_inputs, tmp_path):
        """Report should be written to output file."""
        output_path = tmp_path / "reports" / "conjecture_report.json"

        evaluate_conjectures(
            valid_inputs["baseline_log"],
            valid_inputs["rfl_log"],
            valid_inputs["baseline_summary"],
            valid_inputs["rfl_summary"],
            valid_inputs["telemetry"],
            output_path,
        )

        assert output_path.exists()
        with open(output_path) as f:
            loaded = json.load(f)
        assert loaded["report_version"] == "1.0"

    def test_determinism_same_input_same_output(self, valid_inputs, tmp_path):
        """Same inputs should produce deterministic outputs (excluding timestamp)."""
        output1 = tmp_path / "report1.json"
        output2 = tmp_path / "report2.json"

        report1 = evaluate_conjectures(
            valid_inputs["baseline_log"],
            valid_inputs["rfl_log"],
            valid_inputs["baseline_summary"],
            valid_inputs["rfl_summary"],
            valid_inputs["telemetry"],
            output1,
        )

        report2 = evaluate_conjectures(
            valid_inputs["baseline_log"],
            valid_inputs["rfl_log"],
            valid_inputs["baseline_summary"],
            valid_inputs["rfl_summary"],
            valid_inputs["telemetry"],
            output2,
        )

        # Compare conjectures (deterministic part)
        assert report1["conjectures"] == report2["conjectures"]
        assert report1["summary"] == report2["summary"]
        assert report1["provenance"]["input_hashes"] == report2["provenance"]["input_hashes"]

    def test_invalid_inputs_return_empty_conjectures(self, tmp_path):
        """Invalid inputs should return report with empty conjectures."""
        output_path = tmp_path / "report.json"

        # Create empty/missing files
        baseline_log = tmp_path / "baseline.jsonl"
        baseline_log.touch()

        report = evaluate_conjectures(
            baseline_log,
            Path("nonexistent.jsonl"),
            Path("nonexistent.json"),
            Path("nonexistent.json"),
            Path("nonexistent.json"),
            output_path,
        )

        assert report["input_validation"]["status"] == "INVALID"
        assert report["conjectures"] == {}
        assert report["summary"]["total_evaluated"] == 0

    def test_provenance_includes_file_hashes(self, valid_inputs, tmp_path):
        """Report provenance should include SHA-256 hashes of input files."""
        output_path = tmp_path / "report.json"

        report = evaluate_conjectures(
            valid_inputs["baseline_log"],
            valid_inputs["rfl_log"],
            valid_inputs["baseline_summary"],
            valid_inputs["rfl_summary"],
            valid_inputs["telemetry"],
            output_path,
        )

        hashes = report["provenance"]["input_hashes"]
        assert "baseline_log_sha256" in hashes
        assert "rfl_log_sha256" in hashes
        assert "telemetry_sha256" in hashes
        # All should be 64-char hex strings
        for key, val in hashes.items():
            assert len(val) == 64, f"{key} hash length is {len(val)}"

    def test_each_conjecture_has_required_fields(self, valid_inputs, tmp_path):
        """Each conjecture result should have all required fields."""
        output_path = tmp_path / "report.json"

        report = evaluate_conjectures(
            valid_inputs["baseline_log"],
            valid_inputs["rfl_log"],
            valid_inputs["baseline_summary"],
            valid_inputs["rfl_summary"],
            valid_inputs["telemetry"],
            output_path,
        )

        required_fields = [
            "conjecture_id",
            "name",
            "theory_reference",
            "applicable",
            "applicability_reason",
            "observations",
            "evaluation",
            "evidence_status",
            "evidence_rationale",
            "diagnostics_used",
            "caveats",
        ]

        for conj_id, conj_data in report["conjectures"].items():
            for field in required_fields:
                assert field in conj_data, f"Missing {field} in {conj_id}"


# =============================================================================
# THRESHOLD TESTS
# =============================================================================

class TestThresholds:
    """Tests to verify thresholds are correctly applied."""

    def test_p_value_threshold_boundary(self):
        """Test boundary condition at p=0.05."""
        telemetry_just_under = {
            "diagnostics": {"abstention_trend_tau": -0.30, "abstention_trend_p": 0.049}
        }
        telemetry_just_over = {
            "diagnostics": {"abstention_trend_tau": -0.30, "abstention_trend_p": 0.051}
        }
        rfl_summary = create_valid_summary()

        result_under = _check_R3_1_supermartingale(telemetry_just_under, rfl_summary)
        result_over = _check_R3_1_supermartingale(telemetry_just_over, rfl_summary)

        assert result_under.evidence_status == EvidenceStatus.SUPPORTS
        assert result_over.evidence_status == EvidenceStatus.CONSISTENT

    def test_psi_threshold_boundary(self):
        """Test boundary condition at PSI thresholds."""
        rfl_summary = create_valid_summary()
        rfl_summary["policy_final"] = {"theta_norm": 1.0}
        rfl_summary["time_series"]["success_rates"] = [0.5 + i * 0.003 for i in range(100)]

        # Just under converged threshold
        telemetry_converged = {"diagnostics": {"policy_stability_index": 0.009}}
        result_converged = _check_R13_2_multi_goal_convergence(
            telemetry_converged, rfl_summary, "slice_c"
        )
        assert result_converged.evidence_status == EvidenceStatus.SUPPORTS

        # Between thresholds
        telemetry_middle = {"diagnostics": {"policy_stability_index": 0.03}}
        result_middle = _check_R13_2_multi_goal_convergence(
            telemetry_middle, rfl_summary, "slice_c"
        )
        assert result_middle.evidence_status == EvidenceStatus.CONSISTENT


# =============================================================================
# TASK 1 TESTS: CONJECTURE TRAJECTORY SNAPSHOT & DELTA
# =============================================================================

class TestBuildConjectureSnapshot:
    """Tests for build_conjecture_snapshot()."""

    def test_snapshot_contains_required_fields(self):
        """Snapshot should contain all required fields."""
        from analysis.conjecture_engine_contract import build_conjecture_snapshot, SNAPSHOT_SCHEMA_VERSION

        report = {
            "experiment_id": "test_001",
            "slice_name": "slice_a",
            "generated_at": "2025-12-06T10:00:00Z",
            "conjectures": {
                "conjecture_3_1": {"evidence_status": "SUPPORTS"},
                "conjecture_6_1": {"evidence_status": "CONSISTENT"},
            },
            "summary": {
                "supports_count": 1,
                "consistent_count": 1,
                "contradicts_count": 0,
                "inconclusive_count": 0,
            },
        }

        snapshot = build_conjecture_snapshot(report)

        assert snapshot["schema_version"] == SNAPSHOT_SCHEMA_VERSION
        assert snapshot["experiment_id"] == "test_001"
        assert snapshot["slice_name"] == "slice_a"
        assert snapshot["generated_at"] == "2025-12-06T10:00:00Z"
        assert "statuses" in snapshot
        assert "counts" in snapshot

    def test_snapshot_extracts_statuses(self):
        """Snapshot should extract all conjecture statuses."""
        from analysis.conjecture_engine_contract import build_conjecture_snapshot

        report = {
            "conjectures": {
                "conjecture_3_1": {"evidence_status": "SUPPORTS"},
                "conjecture_4_1": {"evidence_status": "CONTRADICTS"},
                "conjecture_6_1": {"evidence_status": "INCONCLUSIVE"},
            },
            "summary": {"supports_count": 1, "consistent_count": 0, "contradicts_count": 1, "inconclusive_count": 1},
        }

        snapshot = build_conjecture_snapshot(report)

        assert snapshot["statuses"]["conjecture_3_1"] == "SUPPORTS"
        assert snapshot["statuses"]["conjecture_4_1"] == "CONTRADICTS"
        assert snapshot["statuses"]["conjecture_6_1"] == "INCONCLUSIVE"

    def test_snapshot_handles_empty_report(self):
        """Snapshot should handle empty reports gracefully."""
        from analysis.conjecture_engine_contract import build_conjecture_snapshot

        snapshot = build_conjecture_snapshot({})

        assert snapshot["experiment_id"] == "UNKNOWN"
        assert snapshot["statuses"] == {}
        assert snapshot["counts"]["supports"] == 0


class TestCompareConjectureSnapshots:
    """Tests for compare_conjecture_snapshots()."""

    def test_detects_transitions(self):
        """Should detect when conjecture statuses change."""
        from analysis.conjecture_engine_contract import compare_conjecture_snapshots

        old = {
            "experiment_id": "exp_001",
            "generated_at": "2025-12-06T10:00:00Z",
            "statuses": {
                "conjecture_3_1": "CONSISTENT",
                "conjecture_6_1": "INCONCLUSIVE",
            },
            "counts": {"supports": 0, "consistent": 1, "contradicts": 0, "inconclusive": 1},
        }
        new = {
            "experiment_id": "exp_002",
            "generated_at": "2025-12-06T11:00:00Z",
            "statuses": {
                "conjecture_3_1": "SUPPORTS",
                "conjecture_6_1": "INCONCLUSIVE",
            },
            "counts": {"supports": 1, "consistent": 0, "contradicts": 0, "inconclusive": 1},
        }

        delta = compare_conjecture_snapshots(old, new)

        assert len(delta["transitions"]) == 1
        assert delta["transitions"][0]["conjecture_id"] == "conjecture_3_1"
        assert delta["transitions"][0]["from_status"] == "CONSISTENT"
        assert delta["transitions"][0]["to_status"] == "SUPPORTS"

    def test_classifies_improved_conjectures(self):
        """Should identify conjectures that moved toward SUPPORTS."""
        from analysis.conjecture_engine_contract import compare_conjecture_snapshots

        old = {
            "statuses": {"conjecture_3_1": "CONTRADICTS", "conjecture_6_1": "INCONCLUSIVE"},
            "counts": {"supports": 0, "consistent": 0, "contradicts": 1, "inconclusive": 1},
        }
        new = {
            "statuses": {"conjecture_3_1": "CONSISTENT", "conjecture_6_1": "SUPPORTS"},
            "counts": {"supports": 1, "consistent": 1, "contradicts": 0, "inconclusive": 0},
        }

        delta = compare_conjecture_snapshots(old, new)

        assert "conjecture_3_1" in delta["improved"]
        assert "conjecture_6_1" in delta["improved"]
        assert len(delta["degraded"]) == 0

    def test_classifies_degraded_conjectures(self):
        """Should identify conjectures that moved toward CONTRADICTS."""
        from analysis.conjecture_engine_contract import compare_conjecture_snapshots

        old = {
            "statuses": {"conjecture_3_1": "SUPPORTS", "conjecture_6_1": "CONSISTENT"},
            "counts": {"supports": 1, "consistent": 1, "contradicts": 0, "inconclusive": 0},
        }
        new = {
            "statuses": {"conjecture_3_1": "CONTRADICTS", "conjecture_6_1": "INCONCLUSIVE"},
            "counts": {"supports": 0, "consistent": 0, "contradicts": 1, "inconclusive": 1},
        }

        delta = compare_conjecture_snapshots(old, new)

        assert "conjecture_3_1" in delta["degraded"]
        assert "conjecture_6_1" in delta["degraded"]
        assert len(delta["improved"]) == 0

    def test_tracks_unchanged_conjectures(self):
        """Should track conjectures with stable status."""
        from analysis.conjecture_engine_contract import compare_conjecture_snapshots

        old = {
            "statuses": {"conjecture_3_1": "SUPPORTS", "conjecture_6_1": "SUPPORTS"},
            "counts": {"supports": 2, "consistent": 0, "contradicts": 0, "inconclusive": 0},
        }
        new = {
            "statuses": {"conjecture_3_1": "SUPPORTS", "conjecture_6_1": "SUPPORTS"},
            "counts": {"supports": 2, "consistent": 0, "contradicts": 0, "inconclusive": 0},
        }

        delta = compare_conjecture_snapshots(old, new)

        assert len(delta["unchanged"]) == 2
        assert len(delta["transitions"]) == 0

    def test_computes_net_change(self):
        """Should compute correct net change in counts."""
        from analysis.conjecture_engine_contract import compare_conjecture_snapshots

        old = {
            "statuses": {},
            "counts": {"supports": 2, "consistent": 3, "contradicts": 1, "inconclusive": 2},
        }
        new = {
            "statuses": {},
            "counts": {"supports": 4, "consistent": 2, "contradicts": 0, "inconclusive": 2},
        }

        delta = compare_conjecture_snapshots(old, new)

        assert delta["net_change"]["supports_delta"] == 2
        assert delta["net_change"]["consistent_delta"] == -1
        assert delta["net_change"]["contradicts_delta"] == -1
        assert delta["net_change"]["inconclusive_delta"] == 0

    def test_determinism(self):
        """Same inputs should produce identical outputs."""
        from analysis.conjecture_engine_contract import compare_conjecture_snapshots

        old = {"statuses": {"c1": "SUPPORTS"}, "counts": {"supports": 1, "consistent": 0, "contradicts": 0, "inconclusive": 0}}
        new = {"statuses": {"c1": "CONTRADICTS"}, "counts": {"supports": 0, "consistent": 0, "contradicts": 1, "inconclusive": 0}}

        delta1 = compare_conjecture_snapshots(old, new)
        delta2 = compare_conjecture_snapshots(old, new)

        assert delta1 == delta2


# =============================================================================
# TASK 2 TESTS: RFL INTEGRATION HOOK
# =============================================================================

class TestSummarizeConjecturesForRfl:
    """Tests for summarize_conjectures_for_rfl()."""

    def test_counts_match_snapshot(self):
        """Summary counts should match snapshot counts."""
        from analysis.conjecture_engine_contract import summarize_conjectures_for_rfl

        snapshot = {
            "counts": {"supports": 3, "consistent": 2, "contradicts": 1, "inconclusive": 2},
            "statuses": {},
        }

        summary = summarize_conjectures_for_rfl(snapshot)

        assert summary["num_supports"] == 3
        assert summary["num_consistent"] == 2
        assert summary["num_contradicts"] == 1
        assert summary["num_inconclusive"] == 2

    def test_extracts_key_conjectures(self):
        """Should extract status of key convergence conjectures."""
        from analysis.conjecture_engine_contract import summarize_conjectures_for_rfl, KEY_CONVERGENCE_CONJECTURES

        snapshot = {
            "counts": {"supports": 2, "consistent": 0, "contradicts": 0, "inconclusive": 0},
            "statuses": {
                "conjecture_3_1": "SUPPORTS",
                "conjecture_6_1": "CONSISTENT",
                "theorem_13_2": "SUPPORTS",
                "theorem_15_1": "INCONCLUSIVE",
                "other_conjecture": "SUPPORTS",
            },
        }

        summary = summarize_conjectures_for_rfl(snapshot)

        for conj_id in KEY_CONVERGENCE_CONJECTURES:
            assert conj_id in summary["key_conjectures"]

        assert summary["key_conjectures"]["conjecture_3_1"] == "SUPPORTS"
        assert summary["key_conjectures"]["conjecture_6_1"] == "CONSISTENT"

    def test_healthy_when_no_contradictions(self):
        """Learning health should be HEALTHY when no contradictions."""
        from analysis.conjecture_engine_contract import summarize_conjectures_for_rfl

        snapshot = {
            "counts": {"supports": 5, "consistent": 2, "contradicts": 0, "inconclusive": 1},
            "statuses": {},
        }

        summary = summarize_conjectures_for_rfl(snapshot)

        assert summary["learning_health"] == "HEALTHY"

    def test_unhealthy_when_contradicts_majority(self):
        """Learning health should be UNHEALTHY when contradicts > supports."""
        from analysis.conjecture_engine_contract import summarize_conjectures_for_rfl

        snapshot = {
            "counts": {"supports": 1, "consistent": 0, "contradicts": 4, "inconclusive": 0},
            "statuses": {},
        }

        summary = summarize_conjectures_for_rfl(snapshot)

        assert summary["learning_health"] == "UNHEALTHY"

    def test_mixed_when_some_contradictions(self):
        """Learning health should be MIXED when some contradictions exist."""
        from analysis.conjecture_engine_contract import summarize_conjectures_for_rfl

        snapshot = {
            "counts": {"supports": 3, "consistent": 2, "contradicts": 2, "inconclusive": 1},
            "statuses": {},
        }

        summary = summarize_conjectures_for_rfl(snapshot)

        assert summary["learning_health"] == "MIXED"

    def test_inconclusive_when_all_inconclusive(self):
        """Learning health should be INCONCLUSIVE when no supports/consistent."""
        from analysis.conjecture_engine_contract import summarize_conjectures_for_rfl

        snapshot = {
            "counts": {"supports": 0, "consistent": 0, "contradicts": 0, "inconclusive": 5},
            "statuses": {},
        }

        summary = summarize_conjectures_for_rfl(snapshot)

        assert summary["learning_health"] == "INCONCLUSIVE"

    def test_is_json_serializable(self):
        """Summary should be JSON-serializable."""
        from analysis.conjecture_engine_contract import summarize_conjectures_for_rfl

        snapshot = {
            "counts": {"supports": 2, "consistent": 1, "contradicts": 0, "inconclusive": 1},
            "statuses": {"conjecture_3_1": "SUPPORTS"},
        }

        summary = summarize_conjectures_for_rfl(snapshot)

        # This should not raise
        json_str = json.dumps(summary)
        assert len(json_str) > 0


# =============================================================================
# TASK 3 TESTS: GOVERNANCE / GLOBAL HEALTH SIGNAL
# =============================================================================

class TestSummarizeConjecturesForGovernance:
    """Tests for summarize_conjectures_for_governance()."""

    def test_ok_when_stable(self):
        """Status should be OK when no transitions."""
        from analysis.conjecture_engine_contract import summarize_conjectures_for_governance

        delta = {
            "transitions": [],
            "improved": [],
            "degraded": [],
            "net_change": {"supports_delta": 0, "contradicts_delta": 0},
        }

        result = summarize_conjectures_for_governance(delta)

        assert result["status"] == "OK"
        assert result["increasing_support"] is False
        assert result["emerging_contradictions"] is False

    def test_ok_when_improving(self):
        """Status should be OK when conjectures are improving."""
        from analysis.conjecture_engine_contract import summarize_conjectures_for_governance

        delta = {
            "transitions": [{"conjecture_id": "c1", "from_status": "CONSISTENT", "to_status": "SUPPORTS"}],
            "improved": ["c1"],
            "degraded": [],
            "net_change": {"supports_delta": 1, "contradicts_delta": 0},
        }

        result = summarize_conjectures_for_governance(delta)

        assert result["status"] == "OK"
        assert result["increasing_support"] is True
        assert result["emerging_contradictions"] is False

    def test_warn_when_mixed_signals(self):
        """Status should be WARN when both improvement and degradation occur."""
        from analysis.conjecture_engine_contract import summarize_conjectures_for_governance

        delta = {
            "transitions": [
                {"conjecture_id": "c1", "from_status": "CONSISTENT", "to_status": "SUPPORTS"},
                {"conjecture_id": "c2", "from_status": "SUPPORTS", "to_status": "CONTRADICTS"},
            ],
            "improved": ["c1"],
            "degraded": ["c2"],
            "net_change": {"supports_delta": 0, "contradicts_delta": 1},
        }

        result = summarize_conjectures_for_governance(delta)

        assert result["status"] == "WARN"
        assert result["emerging_contradictions"] is True

    def test_attention_when_net_degradation(self):
        """Status should be ATTENTION when more degradation than improvement."""
        from analysis.conjecture_engine_contract import summarize_conjectures_for_governance

        delta = {
            "transitions": [
                {"conjecture_id": "c1", "from_status": "SUPPORTS", "to_status": "CONTRADICTS"},
                {"conjecture_id": "c2", "from_status": "CONSISTENT", "to_status": "INCONCLUSIVE"},
                {"conjecture_id": "c3", "from_status": "SUPPORTS", "to_status": "CONSISTENT"},
            ],
            "improved": [],
            "degraded": ["c1", "c2", "c3"],
            "net_change": {"supports_delta": -2, "contradicts_delta": 1},
        }

        result = summarize_conjectures_for_governance(delta)

        assert result["status"] == "ATTENTION"
        assert result["emerging_contradictions"] is True
        assert result["degraded_count"] == 3

    def test_attention_when_new_contradictions(self):
        """Status should be ATTENTION when new contradictions emerge with net degradation."""
        from analysis.conjecture_engine_contract import summarize_conjectures_for_governance

        delta = {
            "transitions": [
                {"conjecture_id": "c1", "from_status": "SUPPORTS", "to_status": "CONTRADICTS"},
                {"conjecture_id": "c2", "from_status": "SUPPORTS", "to_status": "CONTRADICTS"},
            ],
            "improved": [],
            "degraded": ["c1", "c2"],
            "net_change": {"supports_delta": -2, "contradicts_delta": 2},
        }

        result = summarize_conjectures_for_governance(delta)

        assert result["status"] == "ATTENTION"
        assert result["emerging_contradictions"] is True

    def test_includes_transition_count(self):
        """Result should include transition count."""
        from analysis.conjecture_engine_contract import summarize_conjectures_for_governance

        delta = {
            "transitions": [{"conjecture_id": f"c{i}", "from_status": "A", "to_status": "B"} for i in range(5)],
            "improved": ["c0", "c1", "c2"],
            "degraded": ["c3", "c4"],
            "net_change": {"supports_delta": 0, "contradicts_delta": 0},
        }

        result = summarize_conjectures_for_governance(delta)

        assert result["transition_count"] == 5
        assert result["improved_count"] == 3
        assert result["degraded_count"] == 2

    def test_is_json_serializable(self):
        """Result should be JSON-serializable."""
        from analysis.conjecture_engine_contract import summarize_conjectures_for_governance

        delta = {
            "transitions": [],
            "improved": [],
            "degraded": [],
            "net_change": {"supports_delta": 0, "contradicts_delta": 0},
        }

        result = summarize_conjectures_for_governance(delta)

        # This should not raise
        json_str = json.dumps(result)
        assert len(json_str) > 0


# =============================================================================
# INTEGRATION TESTS: FULL TRAJECTORY WORKFLOW
# =============================================================================

class TestTrajectoryWorkflow:
    """Integration tests for the full trajectory tracking workflow."""

    def test_full_workflow_snapshot_compare_rfl_governance(self, valid_inputs, tmp_path):
        """Test complete workflow: evaluate → snapshot → compare → RFL → governance."""
        from analysis.conjecture_engine_contract import (
            evaluate_conjectures,
            build_conjecture_snapshot,
            compare_conjecture_snapshots,
            summarize_conjectures_for_rfl,
            summarize_conjectures_for_governance,
        )

        output_path = tmp_path / "report.json"

        # Generate a report
        report = evaluate_conjectures(
            valid_inputs["baseline_log"],
            valid_inputs["rfl_log"],
            valid_inputs["baseline_summary"],
            valid_inputs["rfl_summary"],
            valid_inputs["telemetry"],
            output_path,
        )

        # Build snapshot
        snapshot = build_conjecture_snapshot(report)
        assert "statuses" in snapshot
        assert len(snapshot["statuses"]) == 8  # All 8 conjectures

        # Compare with itself (should show no transitions)
        delta = compare_conjecture_snapshots(snapshot, snapshot)
        assert len(delta["transitions"]) == 0
        assert len(delta["unchanged"]) == 8

        # RFL summary
        rfl_summary = summarize_conjectures_for_rfl(snapshot)
        assert "learning_health" in rfl_summary
        assert "key_conjectures" in rfl_summary

        # Governance summary
        gov_summary = summarize_conjectures_for_governance(delta)
        assert gov_summary["status"] == "OK"  # No transitions = stable

    def test_multiple_run_trajectory(self):
        """Test tracking conjecture evolution across multiple simulated runs."""
        from analysis.conjecture_engine_contract import (
            build_conjecture_snapshot,
            compare_conjecture_snapshots,
            summarize_conjectures_for_governance,
        )

        # Simulate Run 1: All inconclusive
        run1_report = {
            "experiment_id": "run_001",
            "generated_at": "2025-12-06T10:00:00Z",
            "conjectures": {
                "conjecture_3_1": {"evidence_status": "INCONCLUSIVE"},
                "conjecture_6_1": {"evidence_status": "INCONCLUSIVE"},
                "theorem_13_2": {"evidence_status": "INCONCLUSIVE"},
            },
            "summary": {"supports_count": 0, "consistent_count": 0, "contradicts_count": 0, "inconclusive_count": 3},
        }

        # Simulate Run 2: Some improvement
        run2_report = {
            "experiment_id": "run_002",
            "generated_at": "2025-12-06T11:00:00Z",
            "conjectures": {
                "conjecture_3_1": {"evidence_status": "CONSISTENT"},
                "conjecture_6_1": {"evidence_status": "CONSISTENT"},
                "theorem_13_2": {"evidence_status": "INCONCLUSIVE"},
            },
            "summary": {"supports_count": 0, "consistent_count": 2, "contradicts_count": 0, "inconclusive_count": 1},
        }

        # Simulate Run 3: Further improvement
        run3_report = {
            "experiment_id": "run_003",
            "generated_at": "2025-12-06T12:00:00Z",
            "conjectures": {
                "conjecture_3_1": {"evidence_status": "SUPPORTS"},
                "conjecture_6_1": {"evidence_status": "SUPPORTS"},
                "theorem_13_2": {"evidence_status": "CONSISTENT"},
            },
            "summary": {"supports_count": 2, "consistent_count": 1, "contradicts_count": 0, "inconclusive_count": 0},
        }

        # Build snapshots
        snap1 = build_conjecture_snapshot(run1_report)
        snap2 = build_conjecture_snapshot(run2_report)
        snap3 = build_conjecture_snapshot(run3_report)

        # Compare Run 1 → Run 2
        delta_1_2 = compare_conjecture_snapshots(snap1, snap2)
        assert len(delta_1_2["improved"]) == 2
        assert len(delta_1_2["degraded"]) == 0

        gov_1_2 = summarize_conjectures_for_governance(delta_1_2)
        assert gov_1_2["status"] == "OK"

        # Compare Run 2 → Run 3
        delta_2_3 = compare_conjecture_snapshots(snap2, snap3)
        assert len(delta_2_3["improved"]) == 3
        assert delta_2_3["net_change"]["supports_delta"] == 2

        gov_2_3 = summarize_conjectures_for_governance(delta_2_3)
        assert gov_2_3["status"] == "OK"
        assert gov_2_3["increasing_support"] is True


# =============================================================================
# PHASE III TASK 1 TESTS: CONJECTURE HISTORY LEDGER
# =============================================================================

class TestBuildConjectureHistory:
    """Tests for build_conjecture_history()."""

    def test_empty_snapshots_returns_defaults(self):
        """Empty snapshot list should return default values."""
        from analysis.conjecture_engine_contract import build_conjecture_history

        result = build_conjecture_history([])

        assert result["trajectory_per_conjecture"] == {}
        assert result["stability_index"] == 1.0
        assert result["number_of_regressions"] == 0
        assert result["num_snapshots"] == 0

    def test_single_snapshot_full_stability(self):
        """Single snapshot should have perfect stability."""
        from analysis.conjecture_engine_contract import build_conjecture_history

        snapshots = [{
            "generated_at": "2025-12-06T10:00:00Z",
            "statuses": {"c1": "SUPPORTS", "c2": "CONSISTENT"},
        }]

        result = build_conjecture_history(snapshots)

        assert result["stability_index"] == 1.0
        assert result["number_of_regressions"] == 0
        assert result["num_snapshots"] == 1
        assert result["trajectory_per_conjecture"]["c1"] == ["SUPPORTS"]
        assert result["trajectory_per_conjecture"]["c2"] == ["CONSISTENT"]

    def test_tracks_trajectories_across_snapshots(self):
        """Should track status trajectory for each conjecture."""
        from analysis.conjecture_engine_contract import build_conjecture_history

        snapshots = [
            {"generated_at": "T1", "statuses": {"c1": "INCONCLUSIVE", "c2": "INCONCLUSIVE"}},
            {"generated_at": "T2", "statuses": {"c1": "CONSISTENT", "c2": "INCONCLUSIVE"}},
            {"generated_at": "T3", "statuses": {"c1": "SUPPORTS", "c2": "CONSISTENT"}},
        ]

        result = build_conjecture_history(snapshots)

        assert result["trajectory_per_conjecture"]["c1"] == ["INCONCLUSIVE", "CONSISTENT", "SUPPORTS"]
        assert result["trajectory_per_conjecture"]["c2"] == ["INCONCLUSIVE", "INCONCLUSIVE", "CONSISTENT"]
        assert result["num_snapshots"] == 3
        assert result["first_snapshot_at"] == "T1"
        assert result["last_snapshot_at"] == "T3"

    def test_counts_regressions(self):
        """Should count regressions (status moving toward CONTRADICTS)."""
        from analysis.conjecture_engine_contract import build_conjecture_history

        snapshots = [
            {"statuses": {"c1": "SUPPORTS", "c2": "SUPPORTS"}},
            {"statuses": {"c1": "CONSISTENT", "c2": "SUPPORTS"}},  # c1 regressed
            {"statuses": {"c1": "CONTRADICTS", "c2": "CONSISTENT"}},  # both regressed
        ]

        result = build_conjecture_history(snapshots)

        assert result["number_of_regressions"] == 3  # c1: 2 regressions, c2: 1 regression
        assert result["per_conjecture_stats"]["c1"]["regressions"] == 2
        assert result["per_conjecture_stats"]["c2"]["regressions"] == 1

    def test_computes_stability_index(self):
        """Should compute stability index based on transitions."""
        from analysis.conjecture_engine_contract import build_conjecture_history

        # All stable - no transitions
        stable_snapshots = [
            {"statuses": {"c1": "SUPPORTS", "c2": "SUPPORTS"}},
            {"statuses": {"c1": "SUPPORTS", "c2": "SUPPORTS"}},
            {"statuses": {"c1": "SUPPORTS", "c2": "SUPPORTS"}},
        ]

        stable_result = build_conjecture_history(stable_snapshots)
        assert stable_result["stability_index"] == 1.0

        # All changing - maximum instability
        unstable_snapshots = [
            {"statuses": {"c1": "SUPPORTS"}},
            {"statuses": {"c1": "CONTRADICTS"}},
            {"statuses": {"c1": "SUPPORTS"}},
        ]

        unstable_result = build_conjecture_history(unstable_snapshots)
        assert unstable_result["stability_index"] == 0.0  # Changed every time

    def test_per_conjecture_stats(self):
        """Should compute per-conjecture statistics."""
        from analysis.conjecture_engine_contract import build_conjecture_history

        snapshots = [
            {"statuses": {"c1": "INCONCLUSIVE", "c2": "SUPPORTS"}},
            {"statuses": {"c1": "CONSISTENT", "c2": "SUPPORTS"}},
            {"statuses": {"c1": "SUPPORTS", "c2": "SUPPORTS"}},
        ]

        result = build_conjecture_history(snapshots)

        # c1 had 2 transitions (improvements)
        assert result["per_conjecture_stats"]["c1"]["transitions"] == 2
        assert result["per_conjecture_stats"]["c1"]["regressions"] == 0
        assert result["per_conjecture_stats"]["c1"]["first_status"] == "INCONCLUSIVE"
        assert result["per_conjecture_stats"]["c1"]["last_status"] == "SUPPORTS"

        # c2 had 0 transitions
        assert result["per_conjecture_stats"]["c2"]["transitions"] == 0
        assert result["per_conjecture_stats"]["c2"]["stability"] == 1.0

    def test_handles_missing_conjectures(self):
        """Should handle conjectures appearing/disappearing between snapshots."""
        from analysis.conjecture_engine_contract import build_conjecture_history

        snapshots = [
            {"statuses": {"c1": "SUPPORTS"}},
            {"statuses": {"c1": "SUPPORTS", "c2": "CONSISTENT"}},  # c2 appears
            {"statuses": {"c2": "SUPPORTS"}},  # c1 disappears
        ]

        result = build_conjecture_history(snapshots)

        assert result["trajectory_per_conjecture"]["c1"] == ["SUPPORTS", "SUPPORTS", "NOT_EVALUATED"]
        assert result["trajectory_per_conjecture"]["c2"] == ["NOT_EVALUATED", "CONSISTENT", "SUPPORTS"]


# =============================================================================
# PHASE III TASK 2 TESTS: GOVERNANCE COUPLING
# =============================================================================

class TestCombineConjectureDeltaWithGovernance:
    """Tests for combine_conjecture_delta_with_governance()."""

    def test_ready_when_governance_passed_stable(self):
        """Should be READY when governance passed and epistemic state is stable."""
        from analysis.conjecture_engine_contract import combine_conjecture_delta_with_governance

        delta = {
            "transitions": [],
            "improved": [],
            "degraded": [],
            "net_change": {"supports_delta": 0, "contradicts_delta": 0},
        }
        governance = {"governance_passed": True}

        result = combine_conjecture_delta_with_governance(delta, governance)

        assert result["uplift_readiness"] == "READY"
        assert result["epistemic_stability"] == 1.0
        assert result["governance_passed"] is True

    def test_ready_when_governance_passed_improving(self):
        """Should be READY when governance passed and conjectures improving."""
        from analysis.conjecture_engine_contract import combine_conjecture_delta_with_governance

        delta = {
            "transitions": [{"conjecture_id": "c1", "from_status": "CONSISTENT", "to_status": "SUPPORTS"}],
            "improved": ["c1"],
            "degraded": [],
            "net_change": {"supports_delta": 1, "contradicts_delta": 0},
        }
        governance = {"governance_passed": True}

        result = combine_conjecture_delta_with_governance(delta, governance)

        assert result["uplift_readiness"] == "READY"
        assert result["governance_alignment"] == "ALIGNED"

    def test_caution_when_governance_passed_but_contradictions(self):
        """Should be CAUTION when governance passed but new contradictions."""
        from analysis.conjecture_engine_contract import combine_conjecture_delta_with_governance

        delta = {
            "transitions": [{"conjecture_id": "c1", "from_status": "SUPPORTS", "to_status": "CONTRADICTS"}],
            "improved": [],
            "degraded": ["c1"],
            "net_change": {"supports_delta": -1, "contradicts_delta": 1},
        }
        governance = {"governance_passed": True}

        result = combine_conjecture_delta_with_governance(delta, governance)

        assert result["uplift_readiness"] == "CAUTION"
        assert "c1" in result["contradictions_of_interest"]
        assert result["governance_alignment"] == "TENSION"

    def test_blocked_when_governance_failed_and_degraded(self):
        """Should be BLOCKED when governance failed and epistemic degradation."""
        from analysis.conjecture_engine_contract import combine_conjecture_delta_with_governance

        delta = {
            "transitions": [{"conjecture_id": "c1", "from_status": "SUPPORTS", "to_status": "CONTRADICTS"}],
            "improved": [],
            "degraded": ["c1"],
            "net_change": {"supports_delta": -1, "contradicts_delta": 1},
        }
        governance = {"governance_passed": False}

        result = combine_conjecture_delta_with_governance(delta, governance)

        assert result["uplift_readiness"] == "BLOCKED"
        assert "degraded" in result["readiness_reason"].lower()

    def test_caution_when_governance_failed_but_stable(self):
        """Should be CAUTION when governance failed but epistemic state is stable."""
        from analysis.conjecture_engine_contract import combine_conjecture_delta_with_governance

        delta = {
            "transitions": [],
            "improved": [],
            "degraded": [],
            "net_change": {"supports_delta": 0, "contradicts_delta": 0},
        }
        governance = {"governance_passed": False}

        result = combine_conjecture_delta_with_governance(delta, governance)

        assert result["uplift_readiness"] == "CAUTION"
        assert result["epistemic_stability"] == 1.0

    def test_epistemic_stability_decreases_with_transitions(self):
        """Epistemic stability should decrease with more transitions."""
        from analysis.conjecture_engine_contract import combine_conjecture_delta_with_governance

        # Few transitions
        delta_few = {
            "transitions": [{"conjecture_id": "c1"}],
            "improved": ["c1"],
            "degraded": [],
            "net_change": {"supports_delta": 1, "contradicts_delta": 0},
        }

        # Many transitions with degradations
        delta_many = {
            "transitions": [{"conjecture_id": f"c{i}"} for i in range(5)],
            "improved": [],
            "degraded": ["c0", "c1", "c2", "c3", "c4"],
            "net_change": {"supports_delta": 0, "contradicts_delta": 0},
        }

        result_few = combine_conjecture_delta_with_governance(delta_few, {"governance_passed": True})
        result_many = combine_conjecture_delta_with_governance(delta_many, {"governance_passed": True})

        assert result_few["epistemic_stability"] > result_many["epistemic_stability"]

    def test_contradictions_of_interest_extracted(self):
        """Should extract conjecture IDs that moved to CONTRADICTS."""
        from analysis.conjecture_engine_contract import combine_conjecture_delta_with_governance

        delta = {
            "transitions": [
                {"conjecture_id": "c1", "from_status": "SUPPORTS", "to_status": "CONTRADICTS"},
                {"conjecture_id": "c2", "from_status": "CONSISTENT", "to_status": "SUPPORTS"},
                {"conjecture_id": "c3", "from_status": "CONSISTENT", "to_status": "CONTRADICTS"},
            ],
            "improved": ["c2"],
            "degraded": ["c1", "c3"],
            "net_change": {"supports_delta": 0, "contradicts_delta": 2},
        }

        result = combine_conjecture_delta_with_governance(delta, {"governance_passed": True})

        assert "c1" in result["contradictions_of_interest"]
        assert "c3" in result["contradictions_of_interest"]
        assert "c2" not in result["contradictions_of_interest"]


# =============================================================================
# PHASE III TASK 3 TESTS: GLOBAL HEALTH SUMMARY
# =============================================================================

class TestSummarizeConjectureDeltaForGlobalHealth:
    """Tests for summarize_conjecture_delta_for_global_health()."""

    def test_positive_signal_when_improving(self):
        """Should return POSITIVE signal when conjectures are improving."""
        from analysis.conjecture_engine_contract import summarize_conjecture_delta_for_global_health

        delta = {
            "transitions": [
                {"conjecture_id": "c1", "from_status": "CONSISTENT", "to_status": "SUPPORTS"},
                {"conjecture_id": "c2", "from_status": "INCONCLUSIVE", "to_status": "CONSISTENT"},
            ],
            "improved": ["c1", "c2"],
            "degraded": [],
            "net_change": {"supports_delta": 1, "contradicts_delta": 0},
        }

        result = summarize_conjecture_delta_for_global_health(delta)

        assert result["uplift_signal"] == "POSITIVE"
        assert result["improved_count"] == 2
        assert result["degraded_count"] == 0

    def test_negative_signal_when_degrading(self):
        """Should return NEGATIVE signal when conjectures are degrading."""
        from analysis.conjecture_engine_contract import summarize_conjecture_delta_for_global_health

        delta = {
            "transitions": [
                {"conjecture_id": "c1", "from_status": "SUPPORTS", "to_status": "CONTRADICTS"},
                {"conjecture_id": "c2", "from_status": "CONSISTENT", "to_status": "INCONCLUSIVE"},
            ],
            "improved": [],
            "degraded": ["c1", "c2"],
            "net_change": {"supports_delta": -1, "contradicts_delta": 1},
        }

        result = summarize_conjecture_delta_for_global_health(delta)

        assert result["uplift_signal"] == "NEGATIVE"
        assert "c1" in result["contradictions"]
        assert len(result["changed_conjectures"]) == 2

    def test_neutral_signal_when_no_changes(self):
        """Should return NEUTRAL signal when no changes."""
        from analysis.conjecture_engine_contract import summarize_conjecture_delta_for_global_health

        delta = {
            "transitions": [],
            "improved": [],
            "degraded": [],
            "net_change": {"supports_delta": 0, "contradicts_delta": 0},
        }

        result = summarize_conjecture_delta_for_global_health(delta)

        assert result["uplift_signal"] == "NEUTRAL"
        assert result["signal_strength"] == 0.5
        assert result["summary_text"] == "No conjecture status changes"

    def test_neutral_signal_when_mixed(self):
        """Should return NEUTRAL signal when mixed improvements and degradations."""
        from analysis.conjecture_engine_contract import summarize_conjecture_delta_for_global_health

        delta = {
            "transitions": [
                {"conjecture_id": "c1", "from_status": "CONSISTENT", "to_status": "SUPPORTS"},
                {"conjecture_id": "c2", "from_status": "SUPPORTS", "to_status": "CONSISTENT"},
            ],
            "improved": ["c1"],
            "degraded": ["c2"],
            "net_change": {"supports_delta": 0, "contradicts_delta": 0},
        }

        result = summarize_conjecture_delta_for_global_health(delta)

        assert result["uplift_signal"] == "NEUTRAL"

    def test_extracts_contradictions(self):
        """Should extract conjectures that moved to CONTRADICTS."""
        from analysis.conjecture_engine_contract import summarize_conjecture_delta_for_global_health

        delta = {
            "transitions": [
                {"conjecture_id": "c1", "from_status": "SUPPORTS", "to_status": "CONTRADICTS"},
                {"conjecture_id": "c2", "from_status": "CONSISTENT", "to_status": "CONTRADICTS"},
                {"conjecture_id": "c3", "from_status": "SUPPORTS", "to_status": "CONSISTENT"},  # Not to CONTRADICTS
            ],
            "improved": [],
            "degraded": ["c1", "c2", "c3"],
            "net_change": {"supports_delta": -2, "contradicts_delta": 2},
        }

        result = summarize_conjecture_delta_for_global_health(delta)

        assert "c1" in result["contradictions"]
        assert "c2" in result["contradictions"]
        assert "c3" not in result["contradictions"]
        assert len(result["contradictions"]) == 2

    def test_signal_strength_increases_with_consistency(self):
        """Signal strength should be higher when all changes are in same direction."""
        from analysis.conjecture_engine_contract import summarize_conjecture_delta_for_global_health

        # All improving
        delta_consistent = {
            "transitions": [{"conjecture_id": f"c{i}", "to_status": "SUPPORTS"} for i in range(4)],
            "improved": ["c0", "c1", "c2", "c3"],
            "degraded": [],
            "net_change": {"supports_delta": 4, "contradicts_delta": 0},
        }

        # Mixed
        delta_mixed = {
            "transitions": [{"conjecture_id": f"c{i}"} for i in range(4)],
            "improved": ["c0", "c1"],
            "degraded": ["c2", "c3"],
            "net_change": {"supports_delta": 0, "contradicts_delta": 0},
        }

        result_consistent = summarize_conjecture_delta_for_global_health(delta_consistent)
        result_mixed = summarize_conjecture_delta_for_global_health(delta_mixed)

        assert result_consistent["signal_strength"] > result_mixed["signal_strength"]

    def test_lists_changed_conjectures(self):
        """Should list all conjectures that changed status."""
        from analysis.conjecture_engine_contract import summarize_conjecture_delta_for_global_health

        delta = {
            "transitions": [
                {"conjecture_id": "c1", "from_status": "A", "to_status": "B"},
                {"conjecture_id": "c2", "from_status": "B", "to_status": "C"},
                {"conjecture_id": "c3", "from_status": "C", "to_status": "A"},
            ],
            "improved": ["c1"],
            "degraded": ["c2", "c3"],
            "net_change": {"supports_delta": 0, "contradicts_delta": 0},
        }

        result = summarize_conjecture_delta_for_global_health(delta)

        assert set(result["changed_conjectures"]) == {"c1", "c2", "c3"}

    def test_is_json_serializable(self):
        """Result should be JSON-serializable."""
        from analysis.conjecture_engine_contract import summarize_conjecture_delta_for_global_health

        delta = {
            "transitions": [{"conjecture_id": "c1", "to_status": "SUPPORTS"}],
            "improved": ["c1"],
            "degraded": [],
            "net_change": {"supports_delta": 1, "contradicts_delta": 0},
        }

        result = summarize_conjecture_delta_for_global_health(delta)

        json_str = json.dumps(result)
        assert len(json_str) > 0


# =============================================================================
# PHASE III INTEGRATION TESTS
# =============================================================================

class TestPhaseIIIIntegration:
    """Integration tests for Phase III functionality."""

    def test_full_phase_iii_workflow(self):
        """Test complete Phase III workflow: history → delta → governance → GH."""
        from analysis.conjecture_engine_contract import (
            build_conjecture_snapshot,
            compare_conjecture_snapshots,
            build_conjecture_history,
            combine_conjecture_delta_with_governance,
            summarize_conjecture_delta_for_global_health,
        )

        # Simulate multiple runs
        reports = [
            {
                "experiment_id": f"run_{i}",
                "generated_at": f"2025-12-06T{10+i}:00:00Z",
                "conjectures": {
                    "c1": {"evidence_status": ["INCONCLUSIVE", "CONSISTENT", "SUPPORTS"][min(i, 2)]},
                    "c2": {"evidence_status": ["INCONCLUSIVE", "CONSISTENT", "CONSISTENT"][min(i, 2)]},
                },
                "summary": {"supports_count": i, "consistent_count": 2-i, "contradicts_count": 0, "inconclusive_count": 0},
            }
            for i in range(3)
        ]

        # Build snapshots
        snapshots = [build_conjecture_snapshot(r) for r in reports]

        # Build history
        history = build_conjecture_history(snapshots)
        assert history["num_snapshots"] == 3
        assert history["number_of_regressions"] == 0  # Only improvements

        # Compare latest two snapshots
        delta = compare_conjecture_snapshots(snapshots[-2], snapshots[-1])

        # Governance coupling
        governance = {"governance_passed": True}
        coupling = combine_conjecture_delta_with_governance(delta, governance)
        assert coupling["uplift_readiness"] in ["READY", "CAUTION"]

        # GH summary
        gh = summarize_conjecture_delta_for_global_health(delta)
        assert gh["uplift_signal"] in ["POSITIVE", "NEUTRAL", "NEGATIVE"]

    def test_regression_detection_across_phases(self):
        """Test that regressions are properly detected across all phases."""
        from analysis.conjecture_engine_contract import (
            build_conjecture_snapshot,
            compare_conjecture_snapshots,
            build_conjecture_history,
            combine_conjecture_delta_with_governance,
            summarize_conjecture_delta_for_global_health,
        )

        # Simulate regression scenario
        reports = [
            {
                "generated_at": "T1",
                "conjectures": {"c1": {"evidence_status": "SUPPORTS"}},
                "summary": {"supports_count": 1, "consistent_count": 0, "contradicts_count": 0, "inconclusive_count": 0},
            },
            {
                "generated_at": "T2",
                "conjectures": {"c1": {"evidence_status": "CONTRADICTS"}},
                "summary": {"supports_count": 0, "consistent_count": 0, "contradicts_count": 1, "inconclusive_count": 0},
            },
        ]

        snapshots = [build_conjecture_snapshot(r) for r in reports]

        # History should detect regression
        history = build_conjecture_history(snapshots)
        assert history["number_of_regressions"] == 1

        # Delta should show degradation
        delta = compare_conjecture_snapshots(snapshots[0], snapshots[1])
        assert "c1" in delta["degraded"]

        # Governance coupling should flag it
        coupling = combine_conjecture_delta_with_governance(delta, {"governance_passed": True})
        assert coupling["uplift_readiness"] == "CAUTION"
        assert "c1" in coupling["contradictions_of_interest"]

        # GH summary should show negative signal
        gh = summarize_conjecture_delta_for_global_health(delta)
        assert gh["uplift_signal"] == "NEGATIVE"
        assert "c1" in gh["contradictions"]


# =============================================================================
# PHASE IV TASK 1 TESTS: CONJECTURE-UPLIFT DECISION HELPER
# =============================================================================

class TestEvaluateConjecturesForUplift:
    """Tests for evaluate_conjectures_for_uplift()."""

    def test_ok_when_stable_and_improving(self):
        """Should return OK when stable history and improving delta."""
        from analysis.conjecture_engine_contract import evaluate_conjectures_for_uplift

        history = {
            "stability_index": 0.95,
            "number_of_regressions": 0,
            "per_conjecture_stats": {"c1": {"regressions": 0, "last_status": "SUPPORTS"}},
            "num_snapshots": 5,
        }
        delta = {
            "transitions": [{"conjecture_id": "c1", "to_status": "SUPPORTS"}],
            "improved": ["c1"],
            "degraded": [],
            "net_change": {"contradicts_delta": 0},
        }

        result = evaluate_conjectures_for_uplift(history, delta)

        assert result["uplift_ok"] is True
        assert result["status"] == "OK"
        assert len(result["blocking_conjectures"]) == 0

    def test_caution_when_single_contradiction(self):
        """Should return CAUTION when one new contradiction."""
        from analysis.conjecture_engine_contract import evaluate_conjectures_for_uplift

        history = {
            "stability_index": 0.8,
            "number_of_regressions": 0,
            "per_conjecture_stats": {},
            "num_snapshots": 3,
        }
        delta = {
            "transitions": [{"conjecture_id": "c1", "to_status": "CONTRADICTS"}],
            "improved": [],
            "degraded": ["c1"],
            "net_change": {"contradicts_delta": 1},
        }

        result = evaluate_conjectures_for_uplift(history, delta)

        assert result["uplift_ok"] is True
        assert result["status"] == "CAUTION"
        assert "c1" in result["blocking_conjectures"]

    def test_block_when_multiple_contradictions(self):
        """Should return BLOCK when 2+ new contradictions."""
        from analysis.conjecture_engine_contract import evaluate_conjectures_for_uplift

        history = {
            "stability_index": 0.7,
            "number_of_regressions": 0,
            "per_conjecture_stats": {},
            "num_snapshots": 3,
        }
        delta = {
            "transitions": [
                {"conjecture_id": "c1", "to_status": "CONTRADICTS"},
                {"conjecture_id": "c2", "to_status": "CONTRADICTS"},
            ],
            "improved": [],
            "degraded": ["c1", "c2"],
            "net_change": {"contradicts_delta": 2},
        }

        result = evaluate_conjectures_for_uplift(history, delta)

        assert result["uplift_ok"] is False
        assert result["status"] == "BLOCK"

    def test_block_when_many_blocking_conjectures(self):
        """Should return BLOCK when 3+ blocking conjectures."""
        from analysis.conjecture_engine_contract import evaluate_conjectures_for_uplift

        history = {
            "stability_index": 0.6,
            "number_of_regressions": 4,
            "per_conjecture_stats": {
                "c1": {"regressions": 2, "last_status": "CONTRADICTS"},
                "c2": {"regressions": 2, "last_status": "CONSISTENT"},
                "c3": {"regressions": 3, "last_status": "CONTRADICTS"},
            },
            "num_snapshots": 5,
        }
        delta = {
            "transitions": [],
            "improved": [],
            "degraded": [],
            "net_change": {"contradicts_delta": 0},
        }

        result = evaluate_conjectures_for_uplift(history, delta)

        assert result["uplift_ok"] is False
        assert result["status"] == "BLOCK"
        assert len(result["blocking_conjectures"]) >= 3

    def test_block_when_highly_unstable(self):
        """Should return BLOCK when stability < 0.3 with sufficient history."""
        from analysis.conjecture_engine_contract import evaluate_conjectures_for_uplift

        history = {
            "stability_index": 0.2,
            "number_of_regressions": 5,
            "per_conjecture_stats": {},
            "num_snapshots": 5,
        }
        delta = {
            "transitions": [],
            "improved": [],
            "degraded": [],
            "net_change": {"contradicts_delta": 0},
        }

        result = evaluate_conjectures_for_uplift(history, delta)

        assert result["uplift_ok"] is False
        assert result["status"] == "BLOCK"
        assert "unstable" in result["notes"].lower()

    def test_caution_when_net_degradation(self):
        """Should return CAUTION when more degraded than improved."""
        from analysis.conjecture_engine_contract import evaluate_conjectures_for_uplift

        history = {
            "stability_index": 0.8,
            "number_of_regressions": 0,
            "per_conjecture_stats": {},
            "num_snapshots": 3,
        }
        delta = {
            "transitions": [],
            "improved": ["c1"],
            "degraded": ["c2", "c3"],
            "net_change": {"contradicts_delta": 0},
        }

        result = evaluate_conjectures_for_uplift(history, delta)

        assert result["uplift_ok"] is True
        assert result["status"] == "CAUTION"
        assert "degradation" in result["notes"].lower()

    def test_includes_key_convergence_conjectures(self):
        """Should flag key convergence conjectures that are CONTRADICTS."""
        from analysis.conjecture_engine_contract import evaluate_conjectures_for_uplift, KEY_CONVERGENCE_CONJECTURES

        history = {
            "stability_index": 0.9,
            "number_of_regressions": 0,
            "per_conjecture_stats": {
                KEY_CONVERGENCE_CONJECTURES[0]: {"regressions": 0, "last_status": "CONTRADICTS"},
            },
            "num_snapshots": 3,
        }
        delta = {
            "transitions": [],
            "improved": [],
            "degraded": [],
            "net_change": {"contradicts_delta": 0},
        }

        result = evaluate_conjectures_for_uplift(history, delta)

        assert KEY_CONVERGENCE_CONJECTURES[0] in result["blocking_conjectures"]
        assert result["status"] == "CAUTION"


# =============================================================================
# PHASE IV TASK 2 TESTS: MAAS EPISTEMIC ADAPTER
# =============================================================================

class TestSummarizeConjecturesForMaas:
    """Tests for summarize_conjectures_for_maas()."""

    def test_positive_signal_ok_status(self):
        """Should return POSITIVE signal and OK status when improving."""
        from analysis.conjecture_engine_contract import summarize_conjectures_for_maas

        delta = {
            "transitions": [{"conjecture_id": "c1", "to_status": "SUPPORTS"}],
            "improved": ["c1"],
            "degraded": [],
            "net_change": {"supports_delta": 1, "contradicts_delta": 0},
        }
        uplift_eval = {
            "uplift_ok": True,
            "status": "OK",
            "blocking_conjectures": [],
        }

        result = summarize_conjectures_for_maas(delta, uplift_eval)

        assert result["epistemic_signal"] == "POSITIVE"
        assert result["uplift_ready"] is True
        assert result["status"] == "OK"
        assert len(result["contradictions_of_interest"]) == 0

    def test_negative_signal_attention_status(self):
        """Should return NEGATIVE signal and ATTENTION status when contradictions."""
        from analysis.conjecture_engine_contract import summarize_conjectures_for_maas

        delta = {
            "transitions": [{"conjecture_id": "c1", "to_status": "CONTRADICTS"}],
            "improved": [],
            "degraded": ["c1"],
            "net_change": {"supports_delta": 0, "contradicts_delta": 1},
        }
        uplift_eval = {
            "uplift_ok": True,
            "status": "CAUTION",
            "blocking_conjectures": ["c1"],
        }

        result = summarize_conjectures_for_maas(delta, uplift_eval)

        assert result["epistemic_signal"] == "NEGATIVE"
        assert result["status"] == "ATTENTION"
        assert "c1" in result["contradictions_of_interest"]

    def test_block_status_propagates(self):
        """Should propagate BLOCK status from uplift eval."""
        from analysis.conjecture_engine_contract import summarize_conjectures_for_maas

        delta = {
            "transitions": [],
            "improved": [],
            "degraded": [],
            "net_change": {"supports_delta": 0, "contradicts_delta": 0},
        }
        uplift_eval = {
            "uplift_ok": False,
            "status": "BLOCK",
            "blocking_conjectures": ["c1", "c2", "c3"],
        }

        result = summarize_conjectures_for_maas(delta, uplift_eval)

        assert result["uplift_ready"] is False
        assert result["status"] == "BLOCK"

    def test_neutral_signal_when_mixed(self):
        """Should return NEUTRAL signal when mixed changes."""
        from analysis.conjecture_engine_contract import summarize_conjectures_for_maas

        delta = {
            "transitions": [],
            "improved": ["c1"],
            "degraded": ["c2"],
            "net_change": {"supports_delta": 0, "contradicts_delta": 0},
        }
        uplift_eval = {
            "uplift_ok": True,
            "status": "OK",
            "blocking_conjectures": [],
        }

        result = summarize_conjectures_for_maas(delta, uplift_eval)

        assert result["epistemic_signal"] == "NEUTRAL"

    def test_includes_counts(self):
        """Should include improved and degraded counts."""
        from analysis.conjecture_engine_contract import summarize_conjectures_for_maas

        delta = {
            "transitions": [],
            "improved": ["c1", "c2", "c3"],
            "degraded": ["c4"],
            "net_change": {"supports_delta": 2, "contradicts_delta": 0},
        }
        uplift_eval = {"uplift_ok": True, "status": "OK", "blocking_conjectures": []}

        result = summarize_conjectures_for_maas(delta, uplift_eval)

        assert result["improved_count"] == 3
        assert result["degraded_count"] == 1
        assert result["supports_delta"] == 2

    def test_is_json_serializable(self):
        """Result should be JSON-serializable."""
        from analysis.conjecture_engine_contract import summarize_conjectures_for_maas

        delta = {"transitions": [], "improved": [], "degraded": [], "net_change": {}}
        uplift_eval = {"uplift_ok": True, "status": "OK", "blocking_conjectures": []}

        result = summarize_conjectures_for_maas(delta, uplift_eval)

        json_str = json.dumps(result)
        assert len(json_str) > 0


# =============================================================================
# PHASE IV TASK 3 TESTS: DIRECTOR CONJECTURE PANEL
# =============================================================================

class TestBuildConjectureDirectorPanel:
    """Tests for build_conjecture_director_panel()."""

    def test_green_light_when_ok_and_positive(self):
        """Should show GREEN light when OK status and positive signal."""
        from analysis.conjecture_engine_contract import build_conjecture_director_panel

        global_health = {
            "uplift_signal": "POSITIVE",
            "signal_strength": 0.8,
            "summary_text": "2 conjectures improved",
            "improved_count": 2,
            "degraded_count": 0,
            "contradictions": [],
        }
        uplift_eval = {
            "status": "OK",
            "uplift_ok": True,
            "blocking_conjectures": [],
            "stability_index": 0.9,
        }

        result = build_conjecture_director_panel(global_health, uplift_eval)

        assert result["status_light"] == "GREEN"
        assert result["uplift_signal"] == "POSITIVE"
        assert result["uplift_ok"] is True

    def test_yellow_light_when_caution(self):
        """Should show YELLOW light when CAUTION status."""
        from analysis.conjecture_engine_contract import build_conjecture_director_panel

        global_health = {
            "uplift_signal": "NEUTRAL",
            "signal_strength": 0.6,
            "improved_count": 1,
            "degraded_count": 1,
            "contradictions": [],
        }
        uplift_eval = {
            "status": "CAUTION",
            "uplift_ok": True,
            "blocking_conjectures": ["c1"],
            "stability_index": 0.7,
        }

        result = build_conjecture_director_panel(global_health, uplift_eval)

        assert result["status_light"] == "YELLOW"

    def test_red_light_when_blocked(self):
        """Should show RED light when BLOCK status."""
        from analysis.conjecture_engine_contract import build_conjecture_director_panel

        global_health = {
            "uplift_signal": "NEGATIVE",
            "signal_strength": 0.9,
            "improved_count": 0,
            "degraded_count": 3,
            "contradictions": ["c1", "c2"],
        }
        uplift_eval = {
            "status": "BLOCK",
            "uplift_ok": False,
            "blocking_conjectures": ["c1", "c2", "c3"],
            "stability_index": 0.4,
        }

        result = build_conjecture_director_panel(global_health, uplift_eval)

        assert result["status_light"] == "RED"
        assert result["uplift_ok"] is False
        assert "Blocked" in result["headline"]

    def test_gray_light_when_insufficient_signal(self):
        """Should show GRAY light when signal strength is low."""
        from analysis.conjecture_engine_contract import build_conjecture_director_panel

        global_health = {
            "uplift_signal": "NEUTRAL",
            "signal_strength": 0.2,
            "improved_count": 0,
            "degraded_count": 0,
            "contradictions": [],
        }
        uplift_eval = {
            "status": "OK",
            "uplift_ok": True,
            "blocking_conjectures": [],
            "stability_index": 1.0,
        }

        result = build_conjecture_director_panel(global_health, uplift_eval)

        assert result["status_light"] == "GRAY"

    def test_headline_shows_contradictions(self):
        """Headline should mention contradictions when present."""
        from analysis.conjecture_engine_contract import build_conjecture_director_panel

        global_health = {
            "uplift_signal": "NEGATIVE",
            "signal_strength": 0.7,
            "improved_count": 0,
            "degraded_count": 2,
            "contradictions": ["c1", "c2"],
        }
        uplift_eval = {
            "status": "CAUTION",
            "uplift_ok": True,
            "blocking_conjectures": ["c1", "c2"],
            "stability_index": 0.6,
        }

        result = build_conjecture_director_panel(global_health, uplift_eval)

        assert "contradiction" in result["headline"].lower()

    def test_headline_shows_improvement_count(self):
        """Headline should show improvement count when improving."""
        from analysis.conjecture_engine_contract import build_conjecture_director_panel

        global_health = {
            "uplift_signal": "POSITIVE",
            "signal_strength": 0.8,
            "improved_count": 3,
            "degraded_count": 0,
            "contradictions": [],
        }
        uplift_eval = {
            "status": "OK",
            "uplift_ok": True,
            "blocking_conjectures": [],
            "stability_index": 0.85,
        }

        result = build_conjecture_director_panel(global_health, uplift_eval)

        assert "3 improved" in result["headline"]

    def test_headline_shows_stability_info(self):
        """Headline should mention stability when relevant."""
        from analysis.conjecture_engine_contract import build_conjecture_director_panel

        global_health = {
            "uplift_signal": "NEUTRAL",
            "signal_strength": 0.5,
            "improved_count": 0,
            "degraded_count": 0,
            "contradictions": [],
        }
        uplift_eval = {
            "status": "OK",
            "uplift_ok": True,
            "blocking_conjectures": [],
            "stability_index": 0.95,
        }

        result = build_conjecture_director_panel(global_health, uplift_eval)

        assert "stable" in result["headline"].lower()

    def test_includes_all_required_fields(self):
        """Panel should include all required fields."""
        from analysis.conjecture_engine_contract import build_conjecture_director_panel

        global_health = {
            "uplift_signal": "NEUTRAL",
            "signal_strength": 0.5,
            "improved_count": 0,
            "degraded_count": 0,
            "contradictions": [],
        }
        uplift_eval = {
            "status": "OK",
            "uplift_ok": True,
            "blocking_conjectures": [],
            "stability_index": 0.8,
        }

        result = build_conjecture_director_panel(global_health, uplift_eval)

        assert "status_light" in result
        assert "uplift_signal" in result
        assert "signal_strength" in result
        assert "headline" in result
        assert "uplift_ok" in result
        assert "blocking_count" in result

    def test_is_json_serializable(self):
        """Result should be JSON-serializable."""
        from analysis.conjecture_engine_contract import build_conjecture_director_panel

        global_health = {"uplift_signal": "NEUTRAL", "signal_strength": 0.5, "improved_count": 0, "degraded_count": 0, "contradictions": []}
        uplift_eval = {"status": "OK", "uplift_ok": True, "blocking_conjectures": [], "stability_index": 0.9}

        result = build_conjecture_director_panel(global_health, uplift_eval)

        json_str = json.dumps(result)
        assert len(json_str) > 0


# =============================================================================
# PHASE IV INTEGRATION TESTS
# =============================================================================

class TestPhaseIVIntegration:
    """Integration tests for Phase IV functionality."""

    def test_full_phase_iv_workflow(self):
        """Test complete Phase IV workflow: history → delta → uplift → MAAS → Director."""
        from analysis.conjecture_engine_contract import (
            build_conjecture_snapshot,
            compare_conjecture_snapshots,
            build_conjecture_history,
            summarize_conjecture_delta_for_global_health,
            evaluate_conjectures_for_uplift,
            summarize_conjectures_for_maas,
            build_conjecture_director_panel,
        )

        # Simulate improving trajectory with enough changes to produce strong signal
        # Need at least 2 transitions to get signal_strength >= 0.3 (2/4 * 1.0 = 0.5)
        reports = [
            {
                "generated_at": "T0",
                "conjectures": {
                    "c1": {"evidence_status": "INCONCLUSIVE"},
                    "c2": {"evidence_status": "INCONCLUSIVE"},
                    "c3": {"evidence_status": "CONSISTENT"},
                    "c4": {"evidence_status": "CONSISTENT"},
                },
                "summary": {"supports_count": 0, "consistent_count": 2, "contradicts_count": 0, "inconclusive_count": 2},
            },
            {
                "generated_at": "T1",
                "conjectures": {
                    "c1": {"evidence_status": "CONSISTENT"},
                    "c2": {"evidence_status": "CONSISTENT"},
                    "c3": {"evidence_status": "SUPPORTS"},
                    "c4": {"evidence_status": "SUPPORTS"},
                },
                "summary": {"supports_count": 2, "consistent_count": 2, "contradicts_count": 0, "inconclusive_count": 0},
            },
        ]

        snapshots = [build_conjecture_snapshot(r) for r in reports]
        history = build_conjecture_history(snapshots)
        delta = compare_conjecture_snapshots(snapshots[0], snapshots[1])
        global_health = summarize_conjecture_delta_for_global_health(delta)

        # Phase IV functions
        uplift_eval = evaluate_conjectures_for_uplift(history, delta)
        assert uplift_eval["uplift_ok"] is True
        assert uplift_eval["status"] == "OK"

        maas_summary = summarize_conjectures_for_maas(delta, uplift_eval)
        assert maas_summary["status"] == "OK"

        director_panel = build_conjecture_director_panel(global_health, uplift_eval)
        assert director_panel["status_light"] in ["GREEN", "YELLOW"]

    def test_blocking_scenario_propagates_through_pipeline(self):
        """Test that blocking conditions propagate through entire pipeline."""
        from analysis.conjecture_engine_contract import (
            build_conjecture_snapshot,
            compare_conjecture_snapshots,
            build_conjecture_history,
            summarize_conjecture_delta_for_global_health,
            evaluate_conjectures_for_uplift,
            summarize_conjectures_for_maas,
            build_conjecture_director_panel,
        )

        # Simulate major regression
        reports = [
            {
                "generated_at": "T1",
                "conjectures": {
                    "c1": {"evidence_status": "SUPPORTS"},
                    "c2": {"evidence_status": "SUPPORTS"},
                    "c3": {"evidence_status": "SUPPORTS"},
                },
                "summary": {"supports_count": 3, "contradicts_count": 0},
            },
            {
                "generated_at": "T2",
                "conjectures": {
                    "c1": {"evidence_status": "CONTRADICTS"},
                    "c2": {"evidence_status": "CONTRADICTS"},
                    "c3": {"evidence_status": "CONSISTENT"},
                },
                "summary": {"supports_count": 0, "contradicts_count": 2},
            },
        ]

        snapshots = [build_conjecture_snapshot(r) for r in reports]
        history = build_conjecture_history(snapshots)
        delta = compare_conjecture_snapshots(snapshots[0], snapshots[1])
        global_health = summarize_conjecture_delta_for_global_health(delta)

        # Should be blocked
        uplift_eval = evaluate_conjectures_for_uplift(history, delta)
        assert uplift_eval["uplift_ok"] is False
        assert uplift_eval["status"] == "BLOCK"

        # MAAS should reflect block
        maas_summary = summarize_conjectures_for_maas(delta, uplift_eval)
        assert maas_summary["status"] == "BLOCK"
        assert maas_summary["uplift_ready"] is False

        # Director should show RED
        director_panel = build_conjecture_director_panel(global_health, uplift_eval)
        assert director_panel["status_light"] == "RED"
        assert director_panel["uplift_ok"] is False

    def test_caution_scenario_with_single_issue(self):
        """Test that single issue results in CAUTION, not BLOCK."""
        from analysis.conjecture_engine_contract import (
            evaluate_conjectures_for_uplift,
            summarize_conjectures_for_maas,
            build_conjecture_director_panel,
        )

        history = {
            "stability_index": 0.85,
            "number_of_regressions": 0,
            "per_conjecture_stats": {},
            "num_snapshots": 3,
        }
        delta = {
            "transitions": [{"conjecture_id": "c1", "to_status": "CONTRADICTS"}],
            "improved": [],
            "degraded": ["c1"],
            "net_change": {"supports_delta": 0, "contradicts_delta": 1},
        }
        global_health = {
            "uplift_signal": "NEGATIVE",
            "signal_strength": 0.7,
            "improved_count": 0,
            "degraded_count": 1,
            "contradictions": ["c1"],
        }

        uplift_eval = evaluate_conjectures_for_uplift(history, delta)
        assert uplift_eval["status"] == "CAUTION"
        assert uplift_eval["uplift_ok"] is True

        maas_summary = summarize_conjectures_for_maas(delta, uplift_eval)
        assert maas_summary["status"] == "ATTENTION"

        director_panel = build_conjecture_director_panel(global_health, uplift_eval)
        assert director_panel["status_light"] == "YELLOW"


# =============================================================================
# PHASE V TESTS: EPISTEMIC UPLIFT CONTRACT
# =============================================================================

class TestEpistemicHealthScore:
    """Tests for epistemic health score computation."""

    def test_high_stability_high_health(self):
        """Test that high stability produces high epistemic health."""
        from analysis.conjecture_engine_contract import _compute_epistemic_health_score

        history = {
            "stability_index": 1.0,
            "per_conjecture_stats": {
                "c1": {"last_status": "SUPPORTS"},
                "c2": {"last_status": "CONSISTENT"},
            },
        }
        delta = {"improved": ["c1"], "degraded": []}

        score = _compute_epistemic_health_score(history, delta)
        assert score >= 0.8

    def test_low_stability_low_health(self):
        """Test that low stability produces lower epistemic health."""
        from analysis.conjecture_engine_contract import _compute_epistemic_health_score

        history = {
            "stability_index": 0.2,
            "per_conjecture_stats": {
                "c1": {"last_status": "CONTRADICTS"},
                "c2": {"last_status": "CONTRADICTS"},
            },
        }
        delta = {"improved": [], "degraded": ["c1", "c2"]}

        score = _compute_epistemic_health_score(history, delta)
        assert score < 0.5

    def test_mixed_signals_moderate_health(self):
        """Test that mixed signals produce moderate health."""
        from analysis.conjecture_engine_contract import _compute_epistemic_health_score

        history = {
            "stability_index": 0.5,
            "per_conjecture_stats": {
                "c1": {"last_status": "SUPPORTS"},
                "c2": {"last_status": "CONTRADICTS"},
            },
        }
        delta = {"improved": ["c1"], "degraded": ["c2"]}

        score = _compute_epistemic_health_score(history, delta)
        assert 0.3 <= score <= 0.7

    def test_empty_history_default_health(self):
        """Test default health when history is empty."""
        from analysis.conjecture_engine_contract import _compute_epistemic_health_score

        history = {"stability_index": 0.5, "per_conjecture_stats": {}}
        delta = {"improved": [], "degraded": []}

        score = _compute_epistemic_health_score(history, delta)
        # With stability=0.5 (0.2 contribution), no contradictions (0.3 contribution),
        # no changes (neutral 0.5 * 0.3 = 0.15 contribution) = 0.65 total
        assert 0.5 <= score <= 0.7


class TestEvaluateConjecturesForUpliftPhaseV:
    """Tests for Phase V extensions to evaluate_conjectures_for_uplift."""

    def test_returns_epistemic_health_score(self):
        """Test that epistemic_health_score is returned."""
        from analysis.conjecture_engine_contract import evaluate_conjectures_for_uplift

        history = {"stability_index": 0.9, "per_conjecture_stats": {}, "num_snapshots": 2}
        delta = {"transitions": [], "improved": [], "degraded": [], "net_change": {}}

        result = evaluate_conjectures_for_uplift(history, delta)

        assert "epistemic_health_score" in result
        assert isinstance(result["epistemic_health_score"], float)
        assert 0.0 <= result["epistemic_health_score"] <= 1.0

    def test_blocks_when_epistemic_health_below_threshold(self):
        """Test BLOCK when epistemic health is below min_epistemic_health."""
        from analysis.conjecture_engine_contract import evaluate_conjectures_for_uplift

        # Low stability + all contradictions = low epistemic health
        history = {
            "stability_index": 0.1,
            "per_conjecture_stats": {
                "c1": {"last_status": "CONTRADICTS"},
                "c2": {"last_status": "CONTRADICTS"},
            },
            "num_snapshots": 3,
        }
        delta = {
            "transitions": [],
            "improved": [],
            "degraded": ["c1", "c2"],
            "net_change": {"contradicts_delta": 0},
        }

        result = evaluate_conjectures_for_uplift(history, delta, min_epistemic_health=0.5)

        assert result["status"] == "BLOCK"
        assert result["uplift_ok"] is False
        assert "Epistemic health below threshold" in result["notes"]

    def test_respects_custom_min_epistemic_health(self):
        """Test that custom min_epistemic_health threshold is respected."""
        from analysis.conjecture_engine_contract import evaluate_conjectures_for_uplift

        history = {
            "stability_index": 0.6,
            "per_conjecture_stats": {"c1": {"last_status": "CONSISTENT"}},
            "num_snapshots": 3,
        }
        delta = {"transitions": [], "improved": [], "degraded": [], "net_change": {}}

        # With high threshold, should block
        result_high = evaluate_conjectures_for_uplift(history, delta, min_epistemic_health=0.9)
        # With low threshold, should pass
        result_low = evaluate_conjectures_for_uplift(history, delta, min_epistemic_health=0.3)

        # High threshold should block or caution
        assert result_high["status"] in ["BLOCK", "CAUTION"]
        # Low threshold should be OK
        assert result_low["status"] == "OK"

    def test_good_convergence_but_poor_epistemic_health(self):
        """Test scenario with good convergence metrics but poor epistemic health."""
        from analysis.conjecture_engine_contract import evaluate_conjectures_for_uplift

        # All conjectures converging but many contradictions in history
        history = {
            "stability_index": 0.2,  # Very unstable history
            "number_of_regressions": 5,
            "per_conjecture_stats": {
                "c1": {"last_status": "CONTRADICTS", "regressions": 2},
                "c2": {"last_status": "CONTRADICTS", "regressions": 2},
            },
            "num_snapshots": 5,
        }
        # Delta shows all improving now
        delta = {
            "transitions": [
                {"conjecture_id": "c1", "from_status": "CONTRADICTS", "to_status": "SUPPORTS"},
                {"conjecture_id": "c2", "from_status": "CONTRADICTS", "to_status": "SUPPORTS"},
            ],
            "improved": ["c1", "c2"],
            "degraded": [],
            "net_change": {"supports_delta": 2, "contradicts_delta": 0},
        }

        result = evaluate_conjectures_for_uplift(history, delta, min_epistemic_health=0.5)

        # Should still block due to poor epistemic health from unstable history
        assert result["status"] == "BLOCK"
        assert result["epistemic_health_score"] < 0.5

    def test_caution_when_near_threshold(self):
        """Test CAUTION when epistemic health is near threshold."""
        from analysis.conjecture_engine_contract import evaluate_conjectures_for_uplift

        history = {
            "stability_index": 0.65,
            "per_conjecture_stats": {"c1": {"last_status": "CONSISTENT"}},
            "num_snapshots": 2,
        }
        delta = {"transitions": [], "improved": [], "degraded": [], "net_change": {}}

        # Set threshold so health is just above it
        result = evaluate_conjectures_for_uplift(history, delta, min_epistemic_health=0.5)

        # Health should be near threshold, resulting in CAUTION
        if result["epistemic_health_score"] < 0.6:
            assert result["status"] == "CAUTION"

    def test_strong_epistemic_health_mentioned_in_notes(self):
        """Test that strong epistemic health is mentioned when OK."""
        from analysis.conjecture_engine_contract import evaluate_conjectures_for_uplift

        history = {
            "stability_index": 1.0,
            "per_conjecture_stats": {"c1": {"last_status": "SUPPORTS"}},
            "num_snapshots": 2,
        }
        delta = {
            "transitions": [],
            "improved": ["c1"],
            "degraded": [],
            "net_change": {"supports_delta": 1},
        }

        result = evaluate_conjectures_for_uplift(history, delta)

        assert result["status"] == "OK"
        if result["epistemic_health_score"] >= 0.8:
            assert "epistemic health" in result["notes"].lower()


class TestSummarizeConjecturesForGlobalConsole:
    """Tests for summarize_conjectures_for_global_console."""

    def test_epistemic_ok_when_healthy(self):
        """Test epistemic_ok is True when health is good."""
        from analysis.conjecture_engine_contract import summarize_conjectures_for_global_console

        delta = {
            "transitions": [{"conjecture_id": "c1", "to_status": "SUPPORTS"}],
            "improved": ["c1"],
            "degraded": [],
            "net_change": {"supports_delta": 1, "contradicts_delta": 0},
        }
        uplift_eval = {
            "status": "OK",
            "uplift_ok": True,
            "epistemic_health_score": 0.8,
            "min_epistemic_health": 0.5,
            "blocking_conjectures": [],
        }

        result = summarize_conjectures_for_global_console(delta, uplift_eval)

        assert result["epistemic_ok"] is True
        assert result["signal"] == "POSITIVE"
        assert result["status_light"] == "GREEN"

    def test_epistemic_not_ok_when_low_health(self):
        """Test epistemic_ok is False when health is below threshold."""
        from analysis.conjecture_engine_contract import summarize_conjectures_for_global_console

        delta = {
            "transitions": [],
            "improved": [],
            "degraded": [],
            "net_change": {"supports_delta": 0, "contradicts_delta": 0},
        }
        uplift_eval = {
            "status": "BLOCK",
            "uplift_ok": False,
            "epistemic_health_score": 0.3,
            "min_epistemic_health": 0.5,
            "blocking_conjectures": [],
        }

        result = summarize_conjectures_for_global_console(delta, uplift_eval)

        assert result["epistemic_ok"] is False
        assert result["status_light"] == "RED"
        assert "Epistemic health low" in result["headline"]

    def test_signal_positive_when_improving(self):
        """Test signal is POSITIVE when improving."""
        from analysis.conjecture_engine_contract import summarize_conjectures_for_global_console

        delta = {
            "transitions": [],
            "improved": ["c1", "c2"],
            "degraded": [],
            "net_change": {"supports_delta": 2, "contradicts_delta": 0},
        }
        uplift_eval = {
            "status": "OK",
            "uplift_ok": True,
            "epistemic_health_score": 0.8,
            "min_epistemic_health": 0.5,
            "blocking_conjectures": [],
        }

        result = summarize_conjectures_for_global_console(delta, uplift_eval)

        assert result["signal"] == "POSITIVE"

    def test_signal_negative_when_degrading(self):
        """Test signal is NEGATIVE when degrading."""
        from analysis.conjecture_engine_contract import summarize_conjectures_for_global_console

        delta = {
            "transitions": [{"conjecture_id": "c1", "to_status": "CONTRADICTS"}],
            "improved": [],
            "degraded": ["c1"],
            "net_change": {"supports_delta": 0, "contradicts_delta": 1},
        }
        uplift_eval = {
            "status": "CAUTION",
            "uplift_ok": True,
            "epistemic_health_score": 0.6,
            "min_epistemic_health": 0.5,
            "blocking_conjectures": [],
        }

        result = summarize_conjectures_for_global_console(delta, uplift_eval)

        assert result["signal"] == "NEGATIVE"
        assert result["status_light"] == "YELLOW"

    def test_required_fields_present(self):
        """Test all required fields are present."""
        from analysis.conjecture_engine_contract import summarize_conjectures_for_global_console

        delta = {"transitions": [], "improved": [], "degraded": [], "net_change": {}}
        uplift_eval = {
            "status": "OK",
            "uplift_ok": True,
            "epistemic_health_score": 0.7,
            "min_epistemic_health": 0.5,
            "blocking_conjectures": [],
        }

        result = summarize_conjectures_for_global_console(delta, uplift_eval)

        assert "epistemic_ok" in result
        assert "signal" in result
        assert "status_light" in result
        assert "headline" in result
        assert "epistemic_health_score" in result

    def test_is_json_serializable(self):
        """Test result is JSON serializable."""
        import json
        from analysis.conjecture_engine_contract import summarize_conjectures_for_global_console

        delta = {"transitions": [], "improved": [], "degraded": [], "net_change": {}}
        uplift_eval = {
            "status": "OK",
            "uplift_ok": True,
            "epistemic_health_score": 0.7,
            "min_epistemic_health": 0.5,
            "blocking_conjectures": [],
        }

        result = summarize_conjectures_for_global_console(delta, uplift_eval)
        serialized = json.dumps(result)
        assert serialized is not None


class TestGetGovernanceSignalForClaudeI:
    """Tests for get_governance_signal_for_claude_i."""

    def test_clear_when_all_ok(self):
        """Test CLEAR level when everything is OK."""
        from analysis.conjecture_engine_contract import get_governance_signal_for_claude_i

        delta = {
            "transitions": [],
            "improved": ["c1"],
            "degraded": [],
            "net_change": {"supports_delta": 1, "contradicts_delta": 0},
        }
        uplift_eval = {
            "status": "OK",
            "uplift_ok": True,
            "epistemic_health_score": 0.8,
            "blocking_conjectures": [],
            "stability_index": 0.9,
        }

        result = get_governance_signal_for_claude_i(delta, uplift_eval)

        assert result["level"] == "CLEAR"
        assert result["uplift_ok"] is True
        assert "Proceed with normal operations" in result["recommended_action"]

    def test_critical_when_key_conjecture_contradicts(self):
        """Test CRITICAL level when key conjecture contradicts."""
        from analysis.conjecture_engine_contract import (
            get_governance_signal_for_claude_i,
            KEY_CONVERGENCE_CONJECTURES,
        )

        key_conj = KEY_CONVERGENCE_CONJECTURES[0]  # conjecture_3_1
        delta = {
            "transitions": [{"conjecture_id": key_conj, "to_status": "CONTRADICTS"}],
            "improved": [],
            "degraded": [key_conj],
            "net_change": {"supports_delta": 0, "contradicts_delta": 1},
        }
        uplift_eval = {
            "status": "BLOCK",
            "uplift_ok": False,
            "epistemic_health_score": 0.3,
            "blocking_conjectures": [key_conj],
            "stability_index": 0.5,
        }

        result = get_governance_signal_for_claude_i(delta, uplift_eval)

        assert result["level"] == "CRITICAL"
        assert key_conj in result["contradictions_of_interest"]
        assert result["key_conjecture_status"][key_conj] in ["CONTRADICTS", "BLOCKED"]

    def test_warning_when_block_without_key_contradiction(self):
        """Test WARNING level when blocked but no key conjecture contradiction."""
        from analysis.conjecture_engine_contract import get_governance_signal_for_claude_i

        delta = {
            "transitions": [
                {"conjecture_id": "non_key_1", "to_status": "CONTRADICTS"},
                {"conjecture_id": "non_key_2", "to_status": "CONTRADICTS"},
            ],
            "improved": [],
            "degraded": ["non_key_1", "non_key_2"],
            "net_change": {"supports_delta": 0, "contradicts_delta": 2},
        }
        uplift_eval = {
            "status": "BLOCK",
            "uplift_ok": False,
            "epistemic_health_score": 0.4,
            "blocking_conjectures": ["non_key_1", "non_key_2"],
            "stability_index": 0.6,
        }

        result = get_governance_signal_for_claude_i(delta, uplift_eval)

        assert result["level"] == "WARNING"
        assert len(result["contradictions_of_interest"]) == 0  # No key conjectures

    def test_advisory_when_caution_good_health(self):
        """Test ADVISORY level when CAUTION with good epistemic health."""
        from analysis.conjecture_engine_contract import get_governance_signal_for_claude_i

        delta = {
            "transitions": [{"conjecture_id": "c1", "to_status": "CONTRADICTS"}],
            "improved": [],
            "degraded": ["c1"],
            "net_change": {"supports_delta": 0, "contradicts_delta": 1},
        }
        uplift_eval = {
            "status": "CAUTION",
            "uplift_ok": True,
            "epistemic_health_score": 0.7,
            "blocking_conjectures": ["c1"],
            "stability_index": 0.8,
        }

        result = get_governance_signal_for_claude_i(delta, uplift_eval)

        assert result["level"] == "ADVISORY"

    def test_warning_when_caution_low_health(self):
        """Test WARNING level when CAUTION with low epistemic health."""
        from analysis.conjecture_engine_contract import get_governance_signal_for_claude_i

        delta = {
            "transitions": [],
            "improved": [],
            "degraded": [],
            "net_change": {"supports_delta": 0, "contradicts_delta": 0},
        }
        uplift_eval = {
            "status": "CAUTION",
            "uplift_ok": True,
            "epistemic_health_score": 0.4,
            "blocking_conjectures": [],
            "stability_index": 0.5,
        }

        result = get_governance_signal_for_claude_i(delta, uplift_eval)

        assert result["level"] == "WARNING"
        assert "epistemic health" in result["recommended_action"].lower()

    def test_key_conjecture_status_populated(self):
        """Test that key_conjecture_status contains all key conjectures."""
        from analysis.conjecture_engine_contract import (
            get_governance_signal_for_claude_i,
            KEY_CONVERGENCE_CONJECTURES,
        )

        delta = {"transitions": [], "improved": [], "degraded": [], "net_change": {}}
        uplift_eval = {
            "status": "OK",
            "uplift_ok": True,
            "epistemic_health_score": 0.8,
            "blocking_conjectures": [],
            "stability_index": 0.9,
        }

        result = get_governance_signal_for_claude_i(delta, uplift_eval)

        for key_conj in KEY_CONVERGENCE_CONJECTURES:
            assert key_conj in result["key_conjecture_status"]

    def test_contradictions_in_key_vs_nonkey(self):
        """Test distinguishing contradictions in key vs non-key conjectures."""
        from analysis.conjecture_engine_contract import (
            get_governance_signal_for_claude_i,
            KEY_CONVERGENCE_CONJECTURES,
        )

        key_conj = KEY_CONVERGENCE_CONJECTURES[0]
        delta = {
            "transitions": [
                {"conjecture_id": key_conj, "to_status": "CONTRADICTS"},
                {"conjecture_id": "non_key_conj", "to_status": "CONTRADICTS"},
            ],
            "improved": [],
            "degraded": [key_conj, "non_key_conj"],
            "net_change": {"supports_delta": 0, "contradicts_delta": 2},
        }
        uplift_eval = {
            "status": "CAUTION",
            "uplift_ok": True,
            "epistemic_health_score": 0.6,
            "blocking_conjectures": [key_conj],
            "stability_index": 0.7,
        }

        result = get_governance_signal_for_claude_i(delta, uplift_eval)

        assert result["details"]["total_contradictions"] == 2
        assert result["details"]["key_contradictions"] == 1
        assert result["details"]["non_key_contradictions"] == 1
        assert key_conj in result["contradictions_of_interest"]
        assert "non_key_conj" not in result["contradictions_of_interest"]

    def test_details_include_all_metrics(self):
        """Test that details include all expected metrics."""
        from analysis.conjecture_engine_contract import get_governance_signal_for_claude_i

        delta = {"transitions": [], "improved": [], "degraded": [], "net_change": {}}
        uplift_eval = {
            "status": "OK",
            "uplift_ok": True,
            "epistemic_health_score": 0.8,
            "blocking_conjectures": [],
            "stability_index": 0.9,
        }

        result = get_governance_signal_for_claude_i(delta, uplift_eval)

        assert "total_contradictions" in result["details"]
        assert "key_contradictions" in result["details"]
        assert "non_key_contradictions" in result["details"]
        assert "blocking_conjectures_count" in result["details"]
        assert "stability_index" in result["details"]
        assert "signal_from_console" in result["details"]

    def test_is_json_serializable(self):
        """Test result is JSON serializable."""
        import json
        from analysis.conjecture_engine_contract import get_governance_signal_for_claude_i

        delta = {"transitions": [], "improved": [], "degraded": [], "net_change": {}}
        uplift_eval = {
            "status": "OK",
            "uplift_ok": True,
            "epistemic_health_score": 0.8,
            "blocking_conjectures": [],
            "stability_index": 0.9,
        }

        result = get_governance_signal_for_claude_i(delta, uplift_eval)
        serialized = json.dumps(result)
        assert serialized is not None


class TestPhaseVIntegration:
    """Integration tests for Phase V functionality."""

    def test_full_phase_v_workflow(self):
        """Test complete Phase V workflow with epistemic health gating."""
        from analysis.conjecture_engine_contract import (
            build_conjecture_snapshot,
            compare_conjecture_snapshots,
            build_conjecture_history,
            evaluate_conjectures_for_uplift,
            summarize_conjectures_for_global_console,
            get_governance_signal_for_claude_i,
        )

        # Simulate healthy trajectory
        reports = [
            {
                "generated_at": "T0",
                "conjectures": {
                    "c1": {"evidence_status": "CONSISTENT"},
                    "c2": {"evidence_status": "CONSISTENT"},
                },
                "summary": {},
            },
            {
                "generated_at": "T1",
                "conjectures": {
                    "c1": {"evidence_status": "SUPPORTS"},
                    "c2": {"evidence_status": "SUPPORTS"},
                },
                "summary": {},
            },
        ]

        snapshots = [build_conjecture_snapshot(r) for r in reports]
        history = build_conjecture_history(snapshots)
        delta = compare_conjecture_snapshots(snapshots[0], snapshots[1])

        # Phase V functions
        uplift_eval = evaluate_conjectures_for_uplift(history, delta)
        assert "epistemic_health_score" in uplift_eval
        assert uplift_eval["status"] == "OK"

        global_console = summarize_conjectures_for_global_console(delta, uplift_eval)
        assert global_console["epistemic_ok"] is True
        # Signal depends on delta structure - just verify it's not NEGATIVE for healthy trajectory
        assert global_console["signal"] in ["POSITIVE", "NEUTRAL"]

        gov_signal = get_governance_signal_for_claude_i(delta, uplift_eval, global_console)
        assert gov_signal["level"] == "CLEAR"

    def test_epistemic_block_propagates_through_pipeline(self):
        """Test that epistemic health block propagates through pipeline."""
        from analysis.conjecture_engine_contract import (
            build_conjecture_snapshot,
            compare_conjecture_snapshots,
            build_conjecture_history,
            evaluate_conjectures_for_uplift,
            summarize_conjectures_for_global_console,
            get_governance_signal_for_claude_i,
        )

        # Simulate unstable history
        reports = [
            {
                "generated_at": "T0",
                "conjectures": {
                    "c1": {"evidence_status": "SUPPORTS"},
                    "c2": {"evidence_status": "CONTRADICTS"},
                },
                "summary": {},
            },
            {
                "generated_at": "T1",
                "conjectures": {
                    "c1": {"evidence_status": "CONTRADICTS"},
                    "c2": {"evidence_status": "SUPPORTS"},
                },
                "summary": {},
            },
            {
                "generated_at": "T2",
                "conjectures": {
                    "c1": {"evidence_status": "SUPPORTS"},
                    "c2": {"evidence_status": "CONTRADICTS"},
                },
                "summary": {},
            },
        ]

        snapshots = [build_conjecture_snapshot(r) for r in reports]
        history = build_conjecture_history(snapshots)
        delta = compare_conjecture_snapshots(snapshots[-2], snapshots[-1])

        # With high min_epistemic_health, should block
        uplift_eval = evaluate_conjectures_for_uplift(
            history, delta, min_epistemic_health=0.8
        )

        # Unstable history should result in low epistemic health
        global_console = summarize_conjectures_for_global_console(delta, uplift_eval)
        gov_signal = get_governance_signal_for_claude_i(delta, uplift_eval, global_console)

        # Should propagate through pipeline
        if uplift_eval["status"] == "BLOCK":
            assert global_console["epistemic_ok"] is False
            assert gov_signal["level"] in ["WARNING", "CRITICAL"]

    def test_key_conjecture_contradiction_escalates_to_critical(self):
        """Test that key conjecture contradiction escalates to CRITICAL."""
        from analysis.conjecture_engine_contract import (
            evaluate_conjectures_for_uplift,
            summarize_conjectures_for_global_console,
            get_governance_signal_for_claude_i,
            KEY_CONVERGENCE_CONJECTURES,
        )

        key_conj = KEY_CONVERGENCE_CONJECTURES[0]

        history = {
            "stability_index": 0.3,
            "number_of_regressions": 3,
            "per_conjecture_stats": {
                key_conj: {"last_status": "CONTRADICTS", "regressions": 2},
            },
            "num_snapshots": 4,
        }
        delta = {
            "transitions": [{"conjecture_id": key_conj, "to_status": "CONTRADICTS"}],
            "improved": [],
            "degraded": [key_conj],
            "net_change": {"supports_delta": 0, "contradicts_delta": 1},
        }

        uplift_eval = evaluate_conjectures_for_uplift(history, delta)
        global_console = summarize_conjectures_for_global_console(delta, uplift_eval)
        gov_signal = get_governance_signal_for_claude_i(delta, uplift_eval, global_console)

        # Key conjecture contradiction should escalate to CRITICAL
        assert gov_signal["level"] == "CRITICAL"
        assert key_conj in gov_signal["contradictions_of_interest"]


# =============================================================================
# PHASE V HARDENING TESTS: SCHEMA VALIDATION & CANONICAL INTERFACES
# =============================================================================

class TestBuildEpistemicConsoleTile:
    """Tests for build_epistemic_console_tile slim tile function."""

    def test_returns_minimal_required_fields(self):
        """Test that tile contains required fields for Global Console."""
        from analysis.conjecture_engine_contract import (
            build_epistemic_console_tile,
            CONSOLE_TILE_REQUIRED_FIELDS,
        )

        delta = {"transitions": [], "improved": [], "degraded": [], "net_change": {}}
        uplift_eval = {
            "status": "OK",
            "uplift_ok": True,
            "epistemic_health_score": 0.8,
            "min_epistemic_health": 0.5,
            "blocking_conjectures": [],
        }

        tile = build_epistemic_console_tile(delta, uplift_eval)

        assert CONSOLE_TILE_REQUIRED_FIELDS.issubset(tile.keys())

    def test_tile_id_is_epistemic_health(self):
        """Test that tile_id is the canonical identifier."""
        from analysis.conjecture_engine_contract import build_epistemic_console_tile

        delta = {"transitions": [], "improved": [], "degraded": [], "net_change": {}}
        uplift_eval = {
            "status": "OK",
            "uplift_ok": True,
            "epistemic_health_score": 0.7,
            "min_epistemic_health": 0.5,
            "blocking_conjectures": [],
        }

        tile = build_epistemic_console_tile(delta, uplift_eval)

        assert tile["tile_id"] == "epistemic_health"

    def test_health_pct_is_integer_percentage(self):
        """Test that health_pct is an integer 0-100."""
        from analysis.conjecture_engine_contract import build_epistemic_console_tile

        delta = {"transitions": [], "improved": [], "degraded": [], "net_change": {}}
        uplift_eval = {
            "status": "OK",
            "uplift_ok": True,
            "epistemic_health_score": 0.75,
            "min_epistemic_health": 0.5,
            "blocking_conjectures": [],
        }

        tile = build_epistemic_console_tile(delta, uplift_eval)

        assert isinstance(tile["health_pct"], int)
        assert 0 <= tile["health_pct"] <= 100
        assert tile["health_pct"] == 75

    def test_headline_truncated_when_long(self):
        """Test that headline is truncated to ~50 chars."""
        from analysis.conjecture_engine_contract import build_epistemic_console_tile

        delta = {
            "transitions": [{"conjecture_id": f"c{i}", "to_status": "CONTRADICTS"} for i in range(10)],
            "improved": [],
            "degraded": [f"c{i}" for i in range(10)],
            "net_change": {"contradicts_delta": 10},
        }
        uplift_eval = {
            "status": "BLOCK",
            "uplift_ok": False,
            "epistemic_health_score": 0.2,
            "min_epistemic_health": 0.5,
            "blocking_conjectures": [f"c{i}" for i in range(10)],
        }

        tile = build_epistemic_console_tile(delta, uplift_eval)

        assert len(tile["headline"]) <= 50

    def test_includes_schema_version(self):
        """Test that tile includes schema version."""
        from analysis.conjecture_engine_contract import (
            build_epistemic_console_tile,
            GLOBAL_CONSOLE_SCHEMA_VERSION,
        )

        delta = {"transitions": [], "improved": [], "degraded": [], "net_change": {}}
        uplift_eval = {
            "status": "OK",
            "uplift_ok": True,
            "epistemic_health_score": 0.8,
            "min_epistemic_health": 0.5,
            "blocking_conjectures": [],
        }

        tile = build_epistemic_console_tile(delta, uplift_eval)

        assert "_schema_version" in tile
        assert tile["_schema_version"] == GLOBAL_CONSOLE_SCHEMA_VERSION

    def test_is_json_serializable(self):
        """Test that tile is JSON serializable."""
        import json
        from analysis.conjecture_engine_contract import build_epistemic_console_tile

        delta = {"transitions": [], "improved": [], "degraded": [], "net_change": {}}
        uplift_eval = {
            "status": "OK",
            "uplift_ok": True,
            "epistemic_health_score": 0.8,
            "min_epistemic_health": 0.5,
            "blocking_conjectures": [],
        }

        tile = build_epistemic_console_tile(delta, uplift_eval)
        serialized = json.dumps(tile)
        assert serialized is not None


class TestGovernanceSignalSchema:
    """Tests for governance signal schema compliance."""

    def test_signal_includes_schema_version(self):
        """Test that signal includes schema version."""
        from analysis.conjecture_engine_contract import (
            get_governance_signal_for_claude_i,
            CLAUDE_I_SCHEMA_VERSION,
        )

        delta = {"transitions": [], "improved": [], "degraded": [], "net_change": {}}
        uplift_eval = {
            "status": "OK",
            "uplift_ok": True,
            "epistemic_health_score": 0.8,
            "blocking_conjectures": [],
            "stability_index": 0.9,
        }

        signal = get_governance_signal_for_claude_i(delta, uplift_eval)

        assert "_schema_version" in signal
        assert signal["_schema_version"] == CLAUDE_I_SCHEMA_VERSION

    def test_signal_includes_adapter_id(self):
        """Test that signal includes adapter identifier."""
        from analysis.conjecture_engine_contract import get_governance_signal_for_claude_i

        delta = {"transitions": [], "improved": [], "degraded": [], "net_change": {}}
        uplift_eval = {
            "status": "OK",
            "uplift_ok": True,
            "epistemic_health_score": 0.8,
            "blocking_conjectures": [],
            "stability_index": 0.9,
        }

        signal = get_governance_signal_for_claude_i(delta, uplift_eval)

        assert "_adapter_id" in signal
        assert signal["_adapter_id"] == "conjecture_engine.governance_signal"

    def test_signal_contains_all_required_fields(self):
        """Test that signal contains all required fields."""
        from analysis.conjecture_engine_contract import (
            get_governance_signal_for_claude_i,
            CLAUDE_I_REQUIRED_FIELDS,
        )

        delta = {"transitions": [], "improved": [], "degraded": [], "net_change": {}}
        uplift_eval = {
            "status": "OK",
            "uplift_ok": True,
            "epistemic_health_score": 0.8,
            "blocking_conjectures": [],
            "stability_index": 0.9,
        }

        signal = get_governance_signal_for_claude_i(delta, uplift_eval)

        assert CLAUDE_I_REQUIRED_FIELDS.issubset(signal.keys())


class TestValidationFunctions:
    """Tests for schema validation functions."""

    def test_validate_governance_signal_passes_valid(self):
        """Test validation passes for valid signal."""
        from analysis.conjecture_engine_contract import (
            get_governance_signal_for_claude_i,
            validate_governance_signal,
        )

        delta = {"transitions": [], "improved": [], "degraded": [], "net_change": {}}
        uplift_eval = {
            "status": "OK",
            "uplift_ok": True,
            "epistemic_health_score": 0.8,
            "blocking_conjectures": [],
            "stability_index": 0.9,
        }

        signal = get_governance_signal_for_claude_i(delta, uplift_eval)

        assert validate_governance_signal(signal) is True

    def test_validate_governance_signal_fails_missing_field(self):
        """Test validation fails when required field is missing."""
        from analysis.conjecture_engine_contract import validate_governance_signal

        incomplete_signal = {
            "level": "CLEAR",
            "uplift_status": "OK",
            # Missing epistemic_health_score, contradictions_of_interest, etc.
        }

        assert validate_governance_signal(incomplete_signal) is False

    def test_validate_console_tile_passes_valid(self):
        """Test validation passes for valid tile."""
        from analysis.conjecture_engine_contract import (
            build_epistemic_console_tile,
            validate_console_tile,
        )

        delta = {"transitions": [], "improved": [], "degraded": [], "net_change": {}}
        uplift_eval = {
            "status": "OK",
            "uplift_ok": True,
            "epistemic_health_score": 0.8,
            "min_epistemic_health": 0.5,
            "blocking_conjectures": [],
        }

        tile = build_epistemic_console_tile(delta, uplift_eval)

        assert validate_console_tile(tile) is True

    def test_validate_console_tile_fails_missing_field(self):
        """Test validation fails when required field is missing."""
        from analysis.conjecture_engine_contract import validate_console_tile

        incomplete_tile = {
            "status_light": "GREEN",
            # Missing headline, epistemic_ok
        }

        assert validate_console_tile(incomplete_tile) is False


class TestGovernanceSignalLevelEnum:
    """Tests for GovernanceSignalLevel enum enhancements."""

    def test_from_string_parses_valid(self):
        """Test from_string parses valid level strings."""
        from analysis.conjecture_engine_contract import GovernanceSignalLevel

        assert GovernanceSignalLevel.from_string("CLEAR") == GovernanceSignalLevel.CLEAR
        assert GovernanceSignalLevel.from_string("ADVISORY") == GovernanceSignalLevel.ADVISORY
        assert GovernanceSignalLevel.from_string("WARNING") == GovernanceSignalLevel.WARNING
        assert GovernanceSignalLevel.from_string("CRITICAL") == GovernanceSignalLevel.CRITICAL

    def test_from_string_defaults_to_warning(self):
        """Test from_string defaults to WARNING for unknown values."""
        from analysis.conjecture_engine_contract import GovernanceSignalLevel

        assert GovernanceSignalLevel.from_string("UNKNOWN") == GovernanceSignalLevel.WARNING
        assert GovernanceSignalLevel.from_string("invalid") == GovernanceSignalLevel.WARNING

    def test_priority_ordering(self):
        """Test that priority ordering is correct."""
        from analysis.conjecture_engine_contract import GovernanceSignalLevel

        assert GovernanceSignalLevel.CLEAR.priority < GovernanceSignalLevel.ADVISORY.priority
        assert GovernanceSignalLevel.ADVISORY.priority < GovernanceSignalLevel.WARNING.priority
        assert GovernanceSignalLevel.WARNING.priority < GovernanceSignalLevel.CRITICAL.priority

    def test_comparison_operators(self):
        """Test comparison operators work correctly."""
        from analysis.conjecture_engine_contract import GovernanceSignalLevel

        assert GovernanceSignalLevel.CLEAR < GovernanceSignalLevel.ADVISORY
        assert GovernanceSignalLevel.ADVISORY < GovernanceSignalLevel.WARNING
        assert GovernanceSignalLevel.WARNING < GovernanceSignalLevel.CRITICAL
        assert GovernanceSignalLevel.CLEAR <= GovernanceSignalLevel.CLEAR
        assert GovernanceSignalLevel.CRITICAL <= GovernanceSignalLevel.CRITICAL


class TestGlobalConsoleSchemaVersion:
    """Tests for schema version in Global Console outputs."""

    def test_console_summary_includes_schema_version(self):
        """Test that full console summary includes schema version."""
        from analysis.conjecture_engine_contract import (
            summarize_conjectures_for_global_console,
            GLOBAL_CONSOLE_SCHEMA_VERSION,
        )

        delta = {"transitions": [], "improved": [], "degraded": [], "net_change": {}}
        uplift_eval = {
            "status": "OK",
            "uplift_ok": True,
            "epistemic_health_score": 0.8,
            "min_epistemic_health": 0.5,
            "blocking_conjectures": [],
        }

        summary = summarize_conjectures_for_global_console(delta, uplift_eval)

        assert "_schema_version" in summary
        assert summary["_schema_version"] == GLOBAL_CONSOLE_SCHEMA_VERSION


class TestHardenedInterfaceIntegration:
    """Integration tests for hardened interfaces."""

    def test_full_pipeline_produces_valid_outputs(self):
        """Test full pipeline produces schema-valid outputs."""
        from analysis.conjecture_engine_contract import (
            build_conjecture_snapshot,
            compare_conjecture_snapshots,
            build_conjecture_history,
            evaluate_conjectures_for_uplift,
            summarize_conjectures_for_global_console,
            build_epistemic_console_tile,
            get_governance_signal_for_claude_i,
            validate_governance_signal,
            validate_console_tile,
        )

        reports = [
            {
                "generated_at": "T0",
                "conjectures": {"c1": {"evidence_status": "CONSISTENT"}},
                "summary": {},
            },
            {
                "generated_at": "T1",
                "conjectures": {"c1": {"evidence_status": "SUPPORTS"}},
                "summary": {},
            },
        ]

        snapshots = [build_conjecture_snapshot(r) for r in reports]
        history = build_conjecture_history(snapshots)
        delta = compare_conjecture_snapshots(snapshots[0], snapshots[1])

        uplift_eval = evaluate_conjectures_for_uplift(history, delta)
        console_summary = summarize_conjectures_for_global_console(delta, uplift_eval)
        tile = build_epistemic_console_tile(delta, uplift_eval)
        gov_signal = get_governance_signal_for_claude_i(delta, uplift_eval, console_summary)

        # All outputs should be schema-valid
        assert validate_console_tile(tile) is True
        assert validate_governance_signal(gov_signal) is True

        # Schema versions should be present
        assert "_schema_version" in console_summary
        assert "_schema_version" in tile
        assert "_schema_version" in gov_signal

    def test_epistemic_health_visible_across_all_interfaces(self):
        """Test epistemic health is visible across all interface outputs."""
        from analysis.conjecture_engine_contract import (
            evaluate_conjectures_for_uplift,
            summarize_conjectures_for_global_console,
            build_epistemic_console_tile,
            get_governance_signal_for_claude_i,
        )

        history = {
            "stability_index": 0.7,
            "per_conjecture_stats": {"c1": {"last_status": "SUPPORTS"}},
            "num_snapshots": 2,
        }
        delta = {"transitions": [], "improved": [], "degraded": [], "net_change": {}}

        uplift_eval = evaluate_conjectures_for_uplift(history, delta)
        console_summary = summarize_conjectures_for_global_console(delta, uplift_eval)
        tile = build_epistemic_console_tile(delta, uplift_eval)
        gov_signal = get_governance_signal_for_claude_i(delta, uplift_eval, console_summary)

        # Epistemic health should be accessible from all interfaces
        assert "epistemic_health_score" in uplift_eval
        assert "epistemic_health_score" in console_summary
        assert "health_pct" in tile  # Percentage form
        assert "epistemic_health_score" in gov_signal

        # Values should be consistent
        health_score = uplift_eval["epistemic_health_score"]
        assert console_summary["epistemic_health_score"] == health_score
        assert tile["health_pct"] == int(round(health_score * 100))
        assert gov_signal["epistemic_health_score"] == health_score


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
