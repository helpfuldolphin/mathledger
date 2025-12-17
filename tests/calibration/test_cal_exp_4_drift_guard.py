"""
CAL-EXP-4 Drift Guard Tripwire Tests.

Prevents CAL-EXP-4 from drifting into:
- A second CAL-EXP-3 (by reusing/renaming uplift metrics)
- A pilot-blended experiment (by referencing pilot paths)
- Capability/intelligence claims (forbidden per LANGUAGE_CONSTRAINTS)
- New slope/trend-based metrics (temporal structure is validity, not metric)

Reference: docs/system_law/calibration/CAL_EXP_4_INDEX.md
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Set

import pytest

# =============================================================================
# Forbidden Terms (metric names and claim language)
# =============================================================================

FORBIDDEN_METRIC_TERMS = {
    "uplift_score",      # CAL-EXP-3 already defines uplift; no renaming
    "intelligence",      # Unoperationalized; forbidden per LANGUAGE_CONSTRAINTS
    "generalization",    # Requires OOD evidence; out of scope
    "capability",        # Overreach; CAL-EXP-4 is measurement stress only
}

# NEW: Forbidden slope/trend metric patterns
# CAL-EXP-4 inherits delta_delta_p from CAL-EXP-3; it does not redefine it
# Temporal structure is a validity condition, not a metric
FORBIDDEN_SLOPE_METRIC_PATTERNS = [
    r"\bdelta_delta_p\s*=",          # Redefining delta_delta_p
    r"\bslope_metric\b",             # Slope as metric name
    r"\btrend_metric\b",             # Trend as metric name
    r"\bvariance_slope\b",           # Slope-based variance metric
    r"\bautocorr_trend\b",           # Trend-based autocorrelation metric
    r"def\s+compute_slope\s*\(",     # Function computing slope as metric
    r"def\s+compute_trend\s*\(",     # Function computing trend as metric
    r'"slope":\s*{',                 # JSON key for slope object
    r'"trend":\s*{',                 # JSON key for trend object
]

FORBIDDEN_CLAIM_PATTERNS = [
    r"\bimproved\b",           # Causal claim without mechanism evidence
    r"\bimprovement\b",        # Causal claim without mechanism evidence
    r"\bintelligence\b",       # Unoperationalized
    r"\bgeneralization\b",     # Requires OOD evidence
    r"\bcapability\b",         # Overreach
    r"\blearning works\b",     # Mechanism attribution (F3.2)
    r"\bsystem improved\b",    # Absolute progress claim
]

# =============================================================================
# Forbidden Paths (pilot involvement)
# =============================================================================

FORBIDDEN_PATH_PATTERNS = [
    r"\bpilot/",               # No pilot involvement
    r"\bexternal/",            # No external ingestion
    r"experiments/pilot",      # No pilot experiments
    r"pilot_config",           # No pilot configuration
    r"pilot_data",             # No pilot data
]

# =============================================================================
# CAL-EXP-4 Document Paths (updated for actual filenames)
# =============================================================================

CAL_EXP_4_SPEC_PATH = "docs/system_law/calibration/CAL_EXP_4_VARIANCE_STRESS_SPEC.md"
CAL_EXP_4_PLAN_PATH = "docs/system_law/calibration/CAL_EXP_4_IMPLEMENTATION_PLAN.md"
CAL_EXP_4_VERIFIER_PLAN_PATH = "docs/system_law/calibration/CAL_EXP_4_VERIFIER_PLAN.md"
CAL_EXP_4_VERIFIER_PATH = "scripts/verify_cal_exp_4_run.py"


# =============================================================================
# Helper Functions
# =============================================================================

def get_project_root() -> Path:
    """Get project root directory."""
    # Navigate up from tests/calibration/ to project root
    return Path(__file__).resolve().parent.parent.parent


def scan_file_for_patterns(
    file_path: Path,
    patterns: List[str],
    case_insensitive: bool = True,
) -> List[str]:
    """Scan a file for forbidden patterns, return matches."""
    if not file_path.exists():
        return []

    content = file_path.read_text(encoding="utf-8", errors="replace")
    flags = re.IGNORECASE if case_insensitive else 0

    matches = []
    for pattern in patterns:
        found = re.findall(pattern, content, flags)
        if found:
            matches.extend(found)

    return matches


def scan_file_for_metric_names(
    file_path: Path,
    forbidden_terms: Set[str],
) -> List[str]:
    """
    Scan a file for forbidden metric names.

    Looks for patterns like:
    - "metric_name": ...
    - metric_name = ...
    - def metric_name(...)
    """
    if not file_path.exists():
        return []

    content = file_path.read_text(encoding="utf-8", errors="replace")
    violations = []

    for term in forbidden_terms:
        # Check for JSON key pattern: "term":
        if f'"{term}"' in content.lower():
            violations.append(f'"{term}" as JSON key')

        # Check for Python assignment: term =
        pattern = rf'\b{term}\s*='
        if re.search(pattern, content, re.IGNORECASE):
            violations.append(f'{term} as variable')

        # Check for function definition: def term(
        pattern = rf'def\s+{term}\s*\('
        if re.search(pattern, content, re.IGNORECASE):
            violations.append(f'{term} as function')

    return violations


def scan_file_for_slope_metrics(file_path: Path) -> List[str]:
    """
    Scan a file for forbidden slope/trend metric definitions.

    CAL-EXP-4 inherits delta_delta_p from CAL-EXP-3; it does not redefine it.
    Temporal structure (variance, autocorrelation) is validity, not metric.
    """
    if not file_path.exists():
        return []

    content = file_path.read_text(encoding="utf-8", errors="replace")
    violations = []

    for pattern in FORBIDDEN_SLOPE_METRIC_PATTERNS:
        found = re.findall(pattern, content, re.IGNORECASE)
        if found:
            violations.extend([f"'{match}' (slope/trend metric)" for match in found])

    return violations


# =============================================================================
# Tripwire Tests
# =============================================================================

class TestCALEXP4DriftGuard:
    """Tripwire tests to prevent CAL-EXP-4 drift."""

    @pytest.fixture
    def project_root(self) -> Path:
        return get_project_root()

    def test_no_forbidden_metric_terms_in_spec(self, project_root: Path) -> None:
        """
        CAL-EXP-4 spec MUST NOT introduce forbidden metric names.

        Forbidden: uplift_score, intelligence, generalization, capability
        """
        spec_path = project_root / CAL_EXP_4_SPEC_PATH

        if not spec_path.exists():
            pytest.skip("CAL_EXP_4_VARIANCE_STRESS_SPEC.md not yet created")

        violations = scan_file_for_metric_names(spec_path, FORBIDDEN_METRIC_TERMS)

        assert not violations, (
            f"CAL-EXP-4 spec contains forbidden metric terms: {violations}. "
            "CAL-EXP-4 is measurement integrity stress only; "
            "it must not introduce new capability metrics."
        )

    def test_no_forbidden_metric_terms_in_plan(self, project_root: Path) -> None:
        """
        CAL-EXP-4 implementation plan MUST NOT introduce forbidden metric names.
        """
        plan_path = project_root / CAL_EXP_4_PLAN_PATH

        if not plan_path.exists():
            pytest.skip("CAL_EXP_4_IMPLEMENTATION_PLAN.md not yet created")

        violations = scan_file_for_metric_names(plan_path, FORBIDDEN_METRIC_TERMS)

        assert not violations, (
            f"CAL-EXP-4 plan contains forbidden metric terms: {violations}. "
            "CAL-EXP-4 is measurement integrity stress only."
        )

    def test_no_forbidden_claim_language_in_spec(self, project_root: Path) -> None:
        """
        CAL-EXP-4 spec MUST NOT use forbidden claim language in CLAIMS.

        Forbidden: improved, improvement, intelligence, generalization, capability

        EXCEPTION: Terms appearing in documentation of what is forbidden or
        in "IS NOT" sections (scope fences) are allowed.
        """
        spec_path = project_root / CAL_EXP_4_SPEC_PATH

        if not spec_path.exists():
            pytest.skip("CAL_EXP_4_VARIANCE_STRESS_SPEC.md not yet created")

        content = spec_path.read_text(encoding="utf-8", errors="replace")

        # Check for forbidden terms that appear as actual claims
        # (not in documentation sections explaining what's forbidden)
        violations = []

        for pattern in FORBIDDEN_CLAIM_PATTERNS:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                # Get surrounding context (100 chars before and after)
                start = max(0, match.start() - 100)
                end = min(len(content), match.end() + 100)
                context = content[start:end].lower()

                # Allow if in scope fence ("IS NOT" table) or forbidden documentation
                safe_contexts = [
                    "is not",           # Scope fence table
                    "forbidden",        # Documenting forbidden terms
                    "not permitted",    # Documenting prohibitions
                    "claims forbidden", # Explicit prohibition
                    "invalid claim",    # Invalid claims section
                    "why invalid",      # Invalid claims explanation
                ]
                if any(ctx in context for ctx in safe_contexts):
                    continue  # Allowed in documentation context

                violations.append(match.group())

        assert not violations, (
            f"CAL-EXP-4 spec contains forbidden claim language: {set(violations)}. "
            "Use neutral language: measured, observed, favors."
        )

    def test_no_pilot_paths_in_spec(self, project_root: Path) -> None:
        """
        CAL-EXP-4 spec MUST NOT reference pilot paths.

        Pilot involvement is forbidden per CAL_EXP_4_INDEX.md.
        """
        spec_path = project_root / CAL_EXP_4_SPEC_PATH

        if not spec_path.exists():
            pytest.skip("CAL_EXP_4_VARIANCE_STRESS_SPEC.md not yet created")

        matches = scan_file_for_patterns(spec_path, FORBIDDEN_PATH_PATTERNS)

        assert not matches, (
            f"CAL-EXP-4 spec references pilot paths: {matches}. "
            "Pilot involvement is FORBIDDEN per CAL_EXP_4_INDEX.md."
        )

    def test_no_pilot_paths_in_plan(self, project_root: Path) -> None:
        """
        CAL-EXP-4 implementation plan MUST NOT reference pilot paths.
        """
        plan_path = project_root / CAL_EXP_4_PLAN_PATH

        if not plan_path.exists():
            pytest.skip("CAL_EXP_4_IMPLEMENTATION_PLAN.md not yet created")

        matches = scan_file_for_patterns(plan_path, FORBIDDEN_PATH_PATTERNS)

        assert not matches, (
            f"CAL-EXP-4 plan references pilot paths: {matches}. "
            "Pilot involvement is FORBIDDEN."
        )

    def test_no_pilot_paths_in_verifier(self, project_root: Path) -> None:
        """
        CAL-EXP-4 verifier MUST NOT reference pilot paths.
        """
        verifier_path = project_root / CAL_EXP_4_VERIFIER_PATH

        if not verifier_path.exists():
            pytest.skip("verify_cal_exp_4_run.py not yet created")

        matches = scan_file_for_patterns(verifier_path, FORBIDDEN_PATH_PATTERNS)

        assert not matches, (
            f"CAL-EXP-4 verifier references pilot paths: {matches}. "
            "Pilot involvement is FORBIDDEN."
        )

    def test_index_exists_and_has_forbidden_sections(self, project_root: Path) -> None:
        """
        CAL_EXP_4_INDEX.md MUST exist and document drift prevention.
        """
        index_path = project_root / "docs/system_law/calibration/CAL_EXP_4_INDEX.md"

        assert index_path.exists(), (
            "CAL_EXP_4_INDEX.md must exist. "
            "This is the authoritative source for CAL-EXP-4 governance."
        )

        content = index_path.read_text(encoding="utf-8", errors="replace")

        # Check for required governance sections
        assert "Forbidden Terms" in content or "forbidden" in content.lower(), (
            "CAL_EXP_4_INDEX.md must document forbidden terms."
        )

        assert "Pilot" in content, (
            "CAL_EXP_4_INDEX.md must document pilot involvement rules."
        )

    def test_cal_exp_4_not_duplicate_of_cal_exp_3(self, project_root: Path) -> None:
        """
        CAL-EXP-4 MUST NOT be a renamed copy of CAL-EXP-3.

        Checks that spec does not simply rename uplift to something else.
        """
        spec_path = project_root / CAL_EXP_4_SPEC_PATH

        if not spec_path.exists():
            pytest.skip("CAL_EXP_4_VARIANCE_STRESS_SPEC.md not yet created")

        content = spec_path.read_text(encoding="utf-8", errors="replace").lower()

        # Check for suspicious patterns that suggest CAL-EXP-3 cloning
        clone_indicators = [
            "learning uplift measurement",  # CAL-EXP-3 title
        ]

        found_indicators = [ind for ind in clone_indicators if ind in content]

        # Allow if explicitly referencing CAL-EXP-3 for comparison
        if found_indicators:
            if "cal-exp-3" in content or "predecessor" in content or "inherit" in content:
                pass  # OK - referencing predecessor
            else:
                assert False, (
                    f"CAL-EXP-4 spec appears to clone CAL-EXP-3: {found_indicators}. "
                    "CAL-EXP-4 must be measurement integrity stress, not uplift measurement."
                )

    def test_no_slope_metrics_in_spec(self, project_root: Path) -> None:
        """
        CAL-EXP-4 spec MUST NOT define new slope/trend-based metrics.

        Temporal structure (variance, autocorrelation) is a VALIDITY condition,
        not a new metric. CAL-EXP-4 inherits delta_delta_p from CAL-EXP-3.
        """
        spec_path = project_root / CAL_EXP_4_SPEC_PATH

        if not spec_path.exists():
            pytest.skip("CAL_EXP_4_VARIANCE_STRESS_SPEC.md not yet created")

        violations = scan_file_for_slope_metrics(spec_path)

        assert not violations, (
            f"CAL-EXP-4 spec defines forbidden slope/trend metrics: {violations}. "
            "Temporal structure is a validity condition, not a metric. "
            "CAL-EXP-4 inherits delta_delta_p from CAL-EXP-3; do not redefine it."
        )

    def test_no_slope_metrics_in_plan(self, project_root: Path) -> None:
        """
        CAL-EXP-4 implementation plan MUST NOT define new slope/trend metrics.
        """
        plan_path = project_root / CAL_EXP_4_PLAN_PATH

        if not plan_path.exists():
            pytest.skip("CAL_EXP_4_IMPLEMENTATION_PLAN.md not yet created")

        violations = scan_file_for_slope_metrics(plan_path)

        assert not violations, (
            f"CAL-EXP-4 plan defines forbidden slope/trend metrics: {violations}. "
            "Temporal structure is a validity condition, not a metric."
        )

    def test_no_slope_metrics_in_verifier(self, project_root: Path) -> None:
        """
        CAL-EXP-4 verifier MUST NOT define new slope/trend metrics.
        """
        verifier_path = project_root / CAL_EXP_4_VERIFIER_PATH

        if not verifier_path.exists():
            pytest.skip("verify_cal_exp_4_run.py not yet created")

        violations = scan_file_for_slope_metrics(verifier_path)

        assert not violations, (
            f"CAL-EXP-4 verifier defines forbidden slope/trend metrics: {violations}. "
            "Temporal structure is a validity condition, not a metric."
        )

    def test_spec_declares_measurement_stress_scope(self, project_root: Path) -> None:
        """
        CAL-EXP-4 spec MUST explicitly declare it is measurement integrity stress.

        This prevents scope creep into capability claims.
        """
        spec_path = project_root / CAL_EXP_4_SPEC_PATH

        if not spec_path.exists():
            pytest.skip("CAL_EXP_4_VARIANCE_STRESS_SPEC.md not yet created")

        content = spec_path.read_text(encoding="utf-8", errors="replace").lower()

        # Check for measurement stress scope declaration
        stress_indicators = [
            "measurement integrity stress",
            "stress test",
            "variance stress",
            "verifier soundness",
            "comparability integrity",
        ]

        found = any(ind in content for ind in stress_indicators)

        assert found, (
            "CAL-EXP-4 spec must explicitly declare measurement integrity stress scope. "
            "Expected one of: measurement integrity stress, stress test, variance stress"
        )

    def test_index_documents_f5_capping(self, project_root: Path) -> None:
        """
        CAL_EXP_4_INDEX.md MUST document F5.x claim capping rules.
        """
        index_path = project_root / "docs/system_law/calibration/CAL_EXP_4_INDEX.md"

        if not index_path.exists():
            pytest.skip("CAL_EXP_4_INDEX.md not yet created")

        content = index_path.read_text(encoding="utf-8", errors="replace")

        # Check for F5.x capping documentation
        assert "F5" in content, (
            "CAL_EXP_4_INDEX.md must document F5.x failure taxonomy."
        )

        assert "CAP" in content or "cap" in content.lower(), (
            "CAL_EXP_4_INDEX.md must document claim capping rules."
        )
