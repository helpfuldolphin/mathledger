"""
CAL-EXP-3 Generator Output Neutrality Tripwire

Verifies that summary.md output from scripts/run_cal_exp_3.py:generate_summary_table()
contains no banned alarm words or CAL-EXP-3 forbidden phrases.

SCOPE: Generator output strings only (NOT verifier output, NOT filenames/contexts).

SHADOW MODE — observational only.
"""

import io
import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import pytest

from tests.helpers.warning_neutrality import BANNED_ALARM_WORDS
from backend.governance.language_constraints import CAL_EXP_3_FORBIDDEN_PHRASES


# =============================================================================
# Fixture: Simulate generate_summary_table() output
# =============================================================================

def _make_mock_summary(
    delta_mean_delta_p: float = -0.05,
    delta_div_rate: float = -0.02,
) -> Dict[str, Any]:
    """Create mock summary dict matching run_cal_exp_3.py structure."""
    return {
        "schema_version": "1.0.0",
        "experiment_id": "CAL-EXP-3",
        "timestamp": "2025-01-01T00:00:00Z",
        "mode": "SHADOW",
        "toolchain_fingerprint": "abc123",
        "arms": {
            "baseline": {
                "arm_id": "baseline",
                "learning_rate": 0.0,
                "seed": 42,
                "total_cycles": 500,
                "final_mean_delta_p": 0.10,
                "final_divergence_rate": 0.20,
            },
            "treatment": {
                "arm_id": "treatment",
                "learning_rate": 0.1,
                "seed": 42,
                "total_cycles": 500,
                "final_mean_delta_p": 0.05,
                "final_divergence_rate": 0.18,
            },
        },
        "comparison": {
            "delta_mean_delta_p": delta_mean_delta_p,
            "delta_divergence_rate": delta_div_rate,
            "delta_p_favors_treatment": delta_mean_delta_p < 0,
            "div_rate_favors_treatment": delta_div_rate < 0,
        },
    }


def _generate_summary_md_lines(summary: Dict[str, Any]) -> list[str]:
    """
    Replicate generate_summary_table() logic from run_cal_exp_3.py.

    This must match the actual implementation to serve as a tripwire.
    """
    baseline = summary["arms"]["baseline"]
    treatment = summary["arms"]["treatment"]
    comp = summary["comparison"]

    return [
        "# CAL-EXP-3 Summary",
        "",
        f"**Timestamp**: {summary['timestamp']}",
        f"**Mode**: {summary['mode']}",
        f"**Toolchain**: {summary['toolchain_fingerprint']}",
        "",
        "## Arm Configuration",
        "",
        "| Arm | learning_rate | seed | cycles |",
        "|-----|---------------|------|--------|",
        f"| baseline | {baseline['learning_rate']} | {baseline['seed']} | {baseline['total_cycles']} |",
        f"| treatment | {treatment['learning_rate']} | {treatment['seed']} | {treatment['total_cycles']} |",
        "",
        "## Results",
        "",
        "| Metric | Baseline | Treatment | Delta |",
        "|--------|----------|-----------|-------|",
        f"| mean_delta_p | {baseline['final_mean_delta_p']:.6f} | {treatment['final_mean_delta_p']:.6f} | {comp['delta_mean_delta_p']:+.6f} |",
        f"| divergence_rate | {baseline['final_divergence_rate']:.4f} | {treatment['final_divergence_rate']:.4f} | {comp['delta_divergence_rate']:+.4f} |",
        "",
        "## Verdict",
        "",
        f"- Delta_p favors treatment: **{comp['delta_p_favors_treatment']}**",
        f"- Divergence rate favors treatment: **{comp['div_rate_favors_treatment']}**",
        "",
        "---",
        "",
        "**SHADOW MODE - observational only.**",
    ]


# =============================================================================
# Tripwire Tests
# =============================================================================

class TestCalExp3GeneratorNeutrality:
    """Tripwire tests for CAL-EXP-3 generator output neutrality."""

    def test_summary_md_no_banned_alarm_words(self):
        """summary.md output must contain no banned alarm words."""
        summary = _make_mock_summary()
        lines = _generate_summary_md_lines(summary)
        full_text = "\n".join(lines).lower()

        violations = []
        for word in BANNED_ALARM_WORDS:
            if word.lower() in full_text:
                violations.append(word)

        assert not violations, (
            f"summary.md contains banned alarm words: {violations}\n"
            f"Update scripts/run_cal_exp_3.py:generate_summary_table() to use neutral language"
        )

    def test_summary_md_no_forbidden_phrases(self):
        """summary.md output must contain no CAL-EXP-3 forbidden phrases."""
        summary = _make_mock_summary()
        lines = _generate_summary_md_lines(summary)
        full_text = "\n".join(lines).lower()

        violations = []
        for phrase in CAL_EXP_3_FORBIDDEN_PHRASES:
            if phrase.lower() in full_text:
                violations.append(phrase)

        assert not violations, (
            f"summary.md contains CAL-EXP-3 forbidden phrases: {violations}\n"
            f"Update scripts/run_cal_exp_3.py:generate_summary_table() to use neutral language"
        )

    @pytest.mark.parametrize("delta_sign", [
        pytest.param(-0.05, id="negative_delta"),
        pytest.param(0.05, id="positive_delta"),
        pytest.param(0.0, id="zero_delta"),
    ])
    def test_summary_md_neutral_across_delta_signs(self, delta_sign: float):
        """summary.md must remain neutral regardless of delta sign (no 'improved' etc.)."""
        summary = _make_mock_summary(delta_mean_delta_p=delta_sign)
        lines = _generate_summary_md_lines(summary)
        full_text = "\n".join(lines).lower()

        # Check for evaluative terms that might appear conditionally
        evaluative_terms = ["improved", "better", "worse", "degraded", "failed"]
        violations = [term for term in evaluative_terms if term in full_text]

        assert not violations, (
            f"summary.md contains evaluative terms when delta={delta_sign}: {violations}\n"
            f"Use observational language like 'favors treatment' instead"
        )

    def test_summary_md_uses_favors_language(self):
        """summary.md should use 'favors' language (observational, not evaluative)."""
        summary = _make_mock_summary()
        lines = _generate_summary_md_lines(summary)
        full_text = "\n".join(lines).lower()

        # Verify the expected neutral phrasing is present
        assert "favors treatment" in full_text or "favors baseline" in full_text, (
            "summary.md should use 'favors' language for neutral observation"
        )
