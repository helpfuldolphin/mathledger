"""
CAL-EXP-2 Verifier CTRPK Invariance Tests.

Proves that CTRPK presence does NOT affect verify_cal_exp_2_run.py pass/fail outcome.

SHADOW MODE CONTRACT:
- CTRPK is purely observational
- CTRPK presence/absence must not change verifier verdict
- Only CTRPK signal and warnings may differ

Similar to identity/NVR invariance tests.
"""

import inspect
import json
import re
from pathlib import Path
from typing import Any, Dict

import pytest

import scripts.verify_cal_exp_2_run as verifier_module
from scripts.verify_cal_exp_2_run import verify_run, VerificationReport

# Import warning neutrality helper
from tests.helpers.warning_neutrality import (
    pytest_assert_warning_neutral,
    pytest_assert_warnings_neutral,
    BANNED_ALARM_WORDS,
)

# Custom banned words for CTRPK warnings - allows "block" as a status enum value
# (similar to how "critical" is allowed for severity enums)
CTRPK_BANNED_WORDS = [w for w in BANNED_ALARM_WORDS if w not in ("block", "blocks", "blocking")]


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def synthetic_cal_exp_2_run_dir(tmp_path: Path) -> Path:
    """
    Create a synthetic CAL-EXP-2 run directory with all required files.

    This represents a minimal PASSING run with:
    - run_config.json: mode=SHADOW, schema_version=1.0.0
    - RUN_METADATA.json: enforcement=false, status=completed
    - divergence_log.jsonl: all LOGGED_ONLY actions
    """
    run_dir = tmp_path / "cal_exp_2_run"
    run_dir.mkdir()

    # run_config.json
    run_config = {
        "schema_version": "1.0.0",
        "mode": "SHADOW",
        "twin_lr_overrides": {
            "H": 0.20,
            "rho": 0.15,
            "tau": 0.02,
            "beta": 0.12,
        },
        "parameters": {
            "seed": 42,
        },
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    # RUN_METADATA.json
    run_metadata = {
        "status": "completed",
        "enforcement": False,
        "cycles_completed": 1000,
        "total_cycles_requested": 1000,
    }
    (run_dir / "RUN_METADATA.json").write_text(
        json.dumps(run_metadata, indent=2), encoding="utf-8"
    )

    # divergence_log.jsonl (all LOGGED_ONLY)
    divergences = [
        {"cycle": 100, "divergence_pct": 0.05, "action": "LOGGED_ONLY"},
        {"cycle": 200, "divergence_pct": 0.08, "action": "LOGGED_ONLY"},
        {"cycle": 300, "divergence_pct": 0.03, "action": "LOGGED_ONLY"},
    ]
    (run_dir / "divergence_log.jsonl").write_text(
        "\n".join(json.dumps(d) for d in divergences) + "\n", encoding="utf-8"
    )

    # real_cycles.jsonl
    real_cycles = [
        {"cycle": i, "value": 0.5 + (i % 10) * 0.01}
        for i in range(1, 101)
    ]
    (run_dir / "real_cycles.jsonl").write_text(
        "\n".join(json.dumps(c) for c in real_cycles) + "\n", encoding="utf-8"
    )

    # twin_predictions.jsonl
    predictions = [
        {"cycle": i, "predicted": 0.5 + (i % 10) * 0.01}
        for i in range(1, 101)
    ]
    (run_dir / "twin_predictions.jsonl").write_text(
        "\n".join(json.dumps(p) for p in predictions) + "\n", encoding="utf-8"
    )

    return run_dir


def add_ctrpk_to_run_dir(
    run_dir: Path,
    ctrpk_value: float = 2.5,
    ctrpk_status: str = "WARN",
    ctrpk_trend: str = "STABLE",
) -> None:
    """Add CTRPK compact artifact to run directory."""
    ctrpk_data = {
        "value": ctrpk_value,
        "status": ctrpk_status,
        "trend": ctrpk_trend,
        "window_cycles": 10000,
        "transition_requests": int(ctrpk_value * 10),
    }
    (run_dir / "ctrpk_compact.json").write_text(
        json.dumps(ctrpk_data, indent=2), encoding="utf-8"
    )


def add_manifest_with_ctrpk(
    run_dir: Path,
    ctrpk_value: float = 2.5,
    ctrpk_status: str = "WARN",
    ctrpk_trend: str = "STABLE",
) -> None:
    """Add evidence pack manifest with CTRPK to run directory."""
    manifest = {
        "schema_version": "1.0.0",
        "mode": "SHADOW",
        "file_count": 5,
        "governance": {
            "curriculum": {
                "ctrpk": {
                    "value": ctrpk_value,
                    "status": ctrpk_status,
                    "trend": ctrpk_trend,
                    "window_cycles": 10000,
                    "transition_requests": int(ctrpk_value * 10),
                    "path": "ctrpk_compact.json",
                }
            }
        },
        "shadow_mode_compliance": {
            "all_divergence_logged_only": True,
            "no_governance_modification": True,
            "no_abort_enforcement": True,
        },
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


# -----------------------------------------------------------------------------
# CAL-EXP-2 Verifier CTRPK Invariance Tests
# -----------------------------------------------------------------------------

class TestCALEXP2VerifierCTRPKInvariance:
    """
    CAL-EXP-2 Verifier CTRPK Invariance Tests.

    Proves that CTRPK presence does NOT affect verify_cal_exp_2_run.py
    pass/fail outcome.

    SHADOW MODE CONTRACT:
    - CTRPK is purely observational
    - CTRPK must not influence verifier verdict
    - Verifier checks are independent of CTRPK
    """

    def test_verifier_verdict_same_with_and_without_ctrpk_compact(
        self, synthetic_cal_exp_2_run_dir: Path
    ) -> None:
        """
        PROOF: Verifier verdict is identical with and without ctrpk_compact.json.

        Run verify_run() twice:
        1. Without CTRPK artifact
        2. With CTRPK artifact (WARN status)

        Both must produce the same verdict.
        """
        run_dir = synthetic_cal_exp_2_run_dir

        # --- Run WITHOUT CTRPK ---
        report_no_ctrpk = verify_run(run_dir)
        verdict_no_ctrpk = report_no_ctrpk.passed

        # --- Add CTRPK and run again ---
        add_ctrpk_to_run_dir(run_dir, ctrpk_value=3.0, ctrpk_status="WARN")

        report_with_ctrpk = verify_run(run_dir)
        verdict_with_ctrpk = report_with_ctrpk.passed

        # --- PROOF: Same verdict ---
        assert verdict_no_ctrpk == verdict_with_ctrpk, (
            f"CTRPK INVARIANCE VIOLATION: verdict changed\n"
            f"Without CTRPK: {verdict_no_ctrpk}\n"
            f"With CTRPK: {verdict_with_ctrpk}"
        )

        # Both should PASS (synthetic run is valid)
        assert verdict_no_ctrpk is True
        assert verdict_with_ctrpk is True

    def test_verifier_verdict_same_with_manifest_ctrpk(
        self, synthetic_cal_exp_2_run_dir: Path
    ) -> None:
        """
        PROOF: Verifier verdict is identical with and without manifest CTRPK.

        Tests that CTRPK in manifest.json doesn't affect verifier.
        """
        run_dir = synthetic_cal_exp_2_run_dir

        # --- Run WITHOUT manifest ---
        report_no_manifest = verify_run(run_dir)
        verdict_no_manifest = report_no_manifest.passed

        # --- Add manifest with CTRPK ---
        add_manifest_with_ctrpk(
            run_dir,
            ctrpk_value=5.0,
            ctrpk_status="BLOCK",
            ctrpk_trend="DEGRADING",
        )

        report_with_manifest = verify_run(run_dir)
        verdict_with_manifest = report_with_manifest.passed

        # --- PROOF: Same verdict ---
        assert verdict_no_manifest == verdict_with_manifest, (
            f"CTRPK INVARIANCE VIOLATION: manifest CTRPK changed verdict\n"
            f"Without manifest: {verdict_no_manifest}\n"
            f"With manifest CTRPK: {verdict_with_manifest}"
        )

    def test_verifier_check_count_same_with_and_without_ctrpk(
        self, synthetic_cal_exp_2_run_dir: Path
    ) -> None:
        """
        PROOF: Verifier runs the same checks with and without CTRPK.

        The number and names of checks should be identical.
        CTRPK presence must not add or remove any checks.
        """
        run_dir = synthetic_cal_exp_2_run_dir

        # --- Run WITHOUT CTRPK ---
        report_no_ctrpk = verify_run(run_dir)
        check_names_no_ctrpk = [c.name for c in report_no_ctrpk.checks]

        # --- Add CTRPK ---
        add_ctrpk_to_run_dir(run_dir)
        add_manifest_with_ctrpk(run_dir)

        report_with_ctrpk = verify_run(run_dir)
        check_names_with_ctrpk = [c.name for c in report_with_ctrpk.checks]

        # --- PROOF: Same checks ---
        assert check_names_no_ctrpk == check_names_with_ctrpk, (
            f"CTRPK INVARIANCE VIOLATION: check list differs\n"
            f"Without CTRPK: {check_names_no_ctrpk}\n"
            f"With CTRPK: {check_names_with_ctrpk}"
        )

    def test_verifier_pass_fail_counts_same_with_and_without_ctrpk(
        self, synthetic_cal_exp_2_run_dir: Path
    ) -> None:
        """
        PROOF: Verifier pass/fail/warn counts are identical with and without CTRPK.
        """
        run_dir = synthetic_cal_exp_2_run_dir

        # --- Run WITHOUT CTRPK ---
        report_no_ctrpk = verify_run(run_dir)

        # --- Add CTRPK ---
        add_ctrpk_to_run_dir(
            run_dir,
            ctrpk_value=8.0,
            ctrpk_status="BLOCK",
            ctrpk_trend="DEGRADING",
        )

        report_with_ctrpk = verify_run(run_dir)

        # --- PROOF: Same counts ---
        assert report_no_ctrpk.pass_count == report_with_ctrpk.pass_count, (
            f"CTRPK INVARIANCE VIOLATION: pass_count differs\n"
            f"Without CTRPK: {report_no_ctrpk.pass_count}\n"
            f"With CTRPK: {report_with_ctrpk.pass_count}"
        )
        assert report_no_ctrpk.fail_count == report_with_ctrpk.fail_count, (
            f"CTRPK INVARIANCE VIOLATION: fail_count differs\n"
            f"Without CTRPK: {report_no_ctrpk.fail_count}\n"
            f"With CTRPK: {report_with_ctrpk.fail_count}"
        )
        assert report_no_ctrpk.warn_count == report_with_ctrpk.warn_count, (
            f"CTRPK INVARIANCE VIOLATION: warn_count differs\n"
            f"Without CTRPK: {report_no_ctrpk.warn_count}\n"
            f"With CTRPK: {report_with_ctrpk.warn_count}"
        )

    def test_verifier_fail_verdict_unaffected_by_ctrpk(
        self, synthetic_cal_exp_2_run_dir: Path
    ) -> None:
        """
        PROOF: A FAILING run remains FAILING regardless of CTRPK presence.

        CTRPK cannot "fix" a failing run or make a passing run fail.
        """
        run_dir = synthetic_cal_exp_2_run_dir

        # Make the run FAIL by setting enforcement=true
        run_metadata = {
            "status": "completed",
            "enforcement": True,  # This violates SHADOW MODE
            "cycles_completed": 1000,
            "total_cycles_requested": 1000,
        }
        (run_dir / "RUN_METADATA.json").write_text(
            json.dumps(run_metadata, indent=2), encoding="utf-8"
        )

        # --- Run WITHOUT CTRPK ---
        report_no_ctrpk = verify_run(run_dir)

        # --- Add CTRPK (even BLOCK status) ---
        add_ctrpk_to_run_dir(
            run_dir,
            ctrpk_value=10.0,
            ctrpk_status="BLOCK",
            ctrpk_trend="DEGRADING",
        )

        report_with_ctrpk = verify_run(run_dir)

        # --- PROOF: Both FAIL (CTRPK doesn't affect verdict) ---
        assert report_no_ctrpk.passed is False, "Expected FAIL without CTRPK"
        assert report_with_ctrpk.passed is False, "Expected FAIL with CTRPK"

        # Fail count should be the same
        assert report_no_ctrpk.fail_count == report_with_ctrpk.fail_count

    def test_verifier_ctrpk_block_does_not_cause_fail(
        self, synthetic_cal_exp_2_run_dir: Path
    ) -> None:
        """
        PROOF: CTRPK status=BLOCK does NOT cause verifier to FAIL.

        Even with the most severe CTRPK status, the verifier should PASS
        if all other invariants are satisfied.

        SHADOW MODE CONTRACT: CTRPK is advisory only, never gating.
        """
        run_dir = synthetic_cal_exp_2_run_dir

        # Add most severe CTRPK
        add_ctrpk_to_run_dir(
            run_dir,
            ctrpk_value=100.0,  # Extreme value
            ctrpk_status="BLOCK",
            ctrpk_trend="DEGRADING",
        )
        add_manifest_with_ctrpk(
            run_dir,
            ctrpk_value=100.0,
            ctrpk_status="BLOCK",
            ctrpk_trend="DEGRADING",
        )

        report = verify_run(run_dir)

        # --- PROOF: Still PASS ---
        assert report.passed is True, (
            f"CTRPK NON-GATING VIOLATION: CTRPK BLOCK caused verifier to FAIL\n"
            f"This violates SHADOW MODE CONTRACT.\n"
            f"Failed checks: {[c.name for c in report.checks if not c.passed]}"
        )

    def test_verifier_report_json_structure_same_with_and_without_ctrpk(
        self, synthetic_cal_exp_2_run_dir: Path
    ) -> None:
        """
        PROOF: Verifier JSON report structure is identical with and without CTRPK.

        The report schema should not change based on CTRPK presence.
        """
        run_dir = synthetic_cal_exp_2_run_dir

        # --- Run WITHOUT CTRPK ---
        report_no_ctrpk = verify_run(run_dir)
        report_dict_no_ctrpk = report_no_ctrpk.to_dict()

        # --- Add CTRPK ---
        add_ctrpk_to_run_dir(run_dir)

        report_with_ctrpk = verify_run(run_dir)
        report_dict_with_ctrpk = report_with_ctrpk.to_dict()

        # --- PROOF: Same top-level keys ---
        assert set(report_dict_no_ctrpk.keys()) == set(report_dict_with_ctrpk.keys()), (
            f"REPORT STRUCTURE VIOLATION: top-level keys differ\n"
            f"Without CTRPK: {set(report_dict_no_ctrpk.keys())}\n"
            f"With CTRPK: {set(report_dict_with_ctrpk.keys())}"
        )

        # Same verdict
        assert report_dict_no_ctrpk["verdict"] == report_dict_with_ctrpk["verdict"]

        # Same summary structure
        assert set(report_dict_no_ctrpk["summary"].keys()) == set(report_dict_with_ctrpk["summary"].keys())


class TestCALEXP2VerifierCTRPKCrossMatrix:
    """
    Cross-matrix tests for CTRPK invariance across all status/trend combinations.

    Ensures verifier verdict is stable across the full CTRPK state space.
    """

    CTRPK_MATRIX = [
        # (status, trend)
        ("OK", "STABLE"),
        ("OK", "IMPROVING"),
        ("OK", "DEGRADING"),
        ("WARN", "STABLE"),
        ("WARN", "IMPROVING"),
        ("WARN", "DEGRADING"),
        ("BLOCK", "STABLE"),
        ("BLOCK", "IMPROVING"),
        ("BLOCK", "DEGRADING"),
    ]

    def test_verifier_verdict_stable_across_ctrpk_matrix(
        self, synthetic_cal_exp_2_run_dir: Path
    ) -> None:
        """
        PROOF: Verifier verdict is PASS for all CTRPK status/trend combinations.

        None of the CTRPK states should cause the verifier to fail.
        """
        run_dir = synthetic_cal_exp_2_run_dir

        # Get baseline verdict (no CTRPK)
        baseline_report = verify_run(run_dir)
        baseline_verdict = baseline_report.passed

        # Test all CTRPK combinations
        for status, trend in self.CTRPK_MATRIX:
            # Add CTRPK with this combination
            add_ctrpk_to_run_dir(
                run_dir,
                ctrpk_value=5.0,
                ctrpk_status=status,
                ctrpk_trend=trend,
            )

            report = verify_run(run_dir)

            assert report.passed == baseline_verdict, (
                f"CTRPK INVARIANCE VIOLATION: verdict differs for "
                f"status={status}, trend={trend}\n"
                f"Baseline: {baseline_verdict}, Got: {report.passed}"
            )

            # Clean up for next iteration
            (run_dir / "ctrpk_compact.json").unlink()

    def test_verifier_check_results_identical_across_ctrpk_matrix(
        self, synthetic_cal_exp_2_run_dir: Path
    ) -> None:
        """
        PROOF: Individual check results are identical across all CTRPK states.
        """
        run_dir = synthetic_cal_exp_2_run_dir

        # Get baseline check results
        baseline_report = verify_run(run_dir)
        baseline_results = {c.name: c.passed for c in baseline_report.checks}

        for status, trend in self.CTRPK_MATRIX:
            add_ctrpk_to_run_dir(
                run_dir,
                ctrpk_value=5.0,
                ctrpk_status=status,
                ctrpk_trend=trend,
            )

            report = verify_run(run_dir)
            results = {c.name: c.passed for c in report.checks}

            assert results == baseline_results, (
                f"CTRPK INVARIANCE VIOLATION: check results differ for "
                f"status={status}, trend={trend}\n"
                f"Baseline: {baseline_results}\n"
                f"Got: {results}"
            )

            (run_dir / "ctrpk_compact.json").unlink()


# -----------------------------------------------------------------------------
# FINAL NON-INTERFERENCE AUDIT: Verifier Source Grep Test
# -----------------------------------------------------------------------------

class TestCALEXP2VerifierCTRPKSourceAudit:
    """
    FINAL NON-INTERFERENCE AUDIT: Verifier has zero CTRPK references.

    Greps the verifier source code to ensure no accidental coupling to CTRPK.
    """

    def test_verifier_source_has_zero_ctrpk_references(self) -> None:
        """
        AUDIT: verify_cal_exp_2_run.py source contains zero 'ctrpk' references.

        This test greps the verifier module source for any CTRPK-related strings.
        Any match indicates accidental coupling that violates non-interference.
        """
        # Get the source file path
        source_path = Path(inspect.getfile(verifier_module))

        # Read the source code
        source_code = source_path.read_text(encoding="utf-8")

        # Search for ctrpk (case-insensitive)
        ctrpk_pattern = re.compile(r"ctrpk", re.IGNORECASE)
        matches = ctrpk_pattern.findall(source_code)

        assert len(matches) == 0, (
            f"CTRPK COUPLING VIOLATION: verify_cal_exp_2_run.py contains "
            f"{len(matches)} 'ctrpk' reference(s).\n"
            f"The verifier must not read or reference CTRPK data.\n"
            f"Source file: {source_path}"
        )

    def test_verifier_source_has_zero_curriculum_churn_references(self) -> None:
        """
        AUDIT: verify_cal_exp_2_run.py source contains zero curriculum churn references.

        Also checks for synonyms/related terms that might indicate coupling.
        """
        source_path = Path(inspect.getfile(verifier_module))
        source_code = source_path.read_text(encoding="utf-8")

        # Patterns that would indicate CTRPK coupling
        coupling_patterns = [
            r"ctrpk",
            r"curriculum.?churn",
            r"transition.?request",  # CTRPK-specific
            r"ctrpk_compact",
        ]

        violations = []
        for pattern in coupling_patterns:
            regex = re.compile(pattern, re.IGNORECASE)
            matches = regex.findall(source_code)
            if matches:
                violations.append(f"'{pattern}': {len(matches)} match(es)")

        assert len(violations) == 0, (
            f"CTRPK COUPLING VIOLATION: verifier contains CTRPK-related references:\n"
            + "\n".join(f"  - {v}" for v in violations)
        )

    def test_verifier_module_has_no_ctrpk_imports(self) -> None:
        """
        AUDIT: verify_cal_exp_2_run.py has no CTRPK-related imports.
        """
        source_path = Path(inspect.getfile(verifier_module))
        source_code = source_path.read_text(encoding="utf-8")

        # Check imports section for CTRPK-related modules
        import_pattern = re.compile(r"^(?:from|import)\s+.*ctrpk", re.IGNORECASE | re.MULTILINE)
        matches = import_pattern.findall(source_code)

        assert len(matches) == 0, (
            f"CTRPK IMPORT VIOLATION: verifier imports CTRPK-related modules:\n"
            + "\n".join(f"  - {m}" for m in matches)
        )


# -----------------------------------------------------------------------------
# CTRPK Warning Neutrality Tests
# -----------------------------------------------------------------------------

class TestCTRPKWarningNeutrality:
    """
    Tests that CTRPK warning strings pass warning neutrality helper.

    SHADOW MODE CONTRACT: Warnings must be neutral, factual, and non-alarming.
    """

    # All CTRPK warning templates
    CTRPK_WARNING_TEMPLATES = [
        # BLOCK status
        "CTRPK status is BLOCK (value=5.00, trend=STABLE). "
        "Curriculum churn elevated; review curriculum transition rate.",
        # WARN + DEGRADING
        "CTRPK status is WARN with DEGRADING trend (value=3.50). "
        "Curriculum churn elevated; monitor transition rate.",
        # WARN only
        "CTRPK status is WARN (value=2.50, trend=STABLE). "
        "Curriculum churn elevated.",
        # DEGRADING only
        "CTRPK trend is DEGRADING (value=1.50, status=OK). "
        "Curriculum transition rate increasing.",
        # Invalid structure
        "CTRPK invalid structure: missing required 'status' field. "
        "Extraction source: MANIFEST.",
        # Mismatch warning
        "CTRPK mismatch: CLI value=1.0, manifest value=2.0. "
        "Using manifest (precedence).",
    ]

    def test_all_ctrpk_warning_templates_are_neutral(self) -> None:
        """
        PROOF: All CTRPK warning templates pass warning neutrality check.

        Uses the reusable warning_neutrality helper to verify:
        - No banned alarm words (except BLOCK as status enum)
        - Single-line format
        - Neutral, descriptive language

        NOTE: Uses CTRPK_BANNED_WORDS which allows "block" as a status enum value,
        similar to how "critical" is allowed for severity enums in the base helper.
        """
        for warning in self.CTRPK_WARNING_TEMPLATES:
            pytest_assert_warning_neutral(
                warning,
                context="CTRPK warning template",
                banned_words=CTRPK_BANNED_WORDS,
            )

    def test_ctrpk_warnings_are_single_line(self) -> None:
        """
        PROOF: All CTRPK warning templates are single-line.
        """
        for warning in self.CTRPK_WARNING_TEMPLATES:
            assert "\n" not in warning, (
                f"MULTI-LINE WARNING VIOLATION: CTRPK warning contains newline:\n"
                f"'{warning}'"
            )

    def test_ctrpk_warnings_use_neutral_language(self) -> None:
        """
        PROOF: CTRPK warnings use neutral language (no alarming words).

        Additional check beyond the helper for CTRPK-specific terms.
        """
        alarming_words = [
            "critical", "danger", "alert", "emergency", "urgent",
            "fail", "error", "broken", "crisis", "severe",
        ]

        for warning in self.CTRPK_WARNING_TEMPLATES:
            warning_lower = warning.lower()
            for word in alarming_words:
                assert word not in warning_lower, (
                    f"ALARMING LANGUAGE VIOLATION: CTRPK warning contains '{word}':\n"
                    f"'{warning}'"
                )

    def test_ctrpk_warnings_are_descriptive_not_prescriptive(self) -> None:
        """
        PROOF: CTRPK warnings describe state, don't prescribe action.

        Warnings should say "X is Y" not "You must do Z".
        """
        prescriptive_patterns = [
            r"\byou must\b",
            r"\byou should\b",
            r"\bdo not\b",
            r"\bnever\b",
            r"\balways\b",
            r"\brequired to\b",
        ]

        for warning in self.CTRPK_WARNING_TEMPLATES:
            warning_lower = warning.lower()
            for pattern in prescriptive_patterns:
                match = re.search(pattern, warning_lower)
                assert match is None, (
                    f"PRESCRIPTIVE LANGUAGE VIOLATION: CTRPK warning contains '{pattern}':\n"
                    f"'{warning}'"
                )
