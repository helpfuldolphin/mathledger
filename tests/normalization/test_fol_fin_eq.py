"""
Tests for FOL_FIN_EQ_v1 exhaustive enumeration verifier.

Phase 2 RED: These tests will fail until normalization/fol_fin_eq.py is implemented.

Tests the verifier core including:
- Verification with witnesses (VERIFIED)
- Refutation with counterexamples (REFUTED)
- Resource limit abstention (ABSTAINED)
- Deterministic enumeration order
- Variable binding order semantics
"""

import pytest
from pathlib import Path

# Import modules that exist (Phase 1)
from normalization.domain_spec import DomainSpec, parse_domain_spec
from normalization.fol_ast import parse_fol_formula, extract_quantifier_report

# Import module that doesn't exist yet (will fail with ModuleNotFoundError)
from normalization.fol_fin_eq import verify_fol_fin_eq, estimate_assignment_upper_bound

FIXTURES = Path(__file__).parent.parent / "fixtures" / "fol"


class TestZ2Verification:
    """Z2 group axiom verification tests."""

    @pytest.fixture
    def z2_domain(self):
        return parse_domain_spec(FIXTURES / "z2_domain.json")

    def test_z2_identity_verified(self, z2_domain):
        """Z2 satisfies identity axiom."""
        formula = parse_fol_formula((FIXTURES / "identity_formula.json"))
        result = verify_fol_fin_eq(z2_domain, formula)
        assert result.status == "VERIFIED"
        assert result.quantifier_report["forall_vars"] == ["x"]

    def test_z2_inverse_verified(self, z2_domain):
        """Z2 satisfies inverse axiom with witnesses."""
        formula = parse_fol_formula((FIXTURES / "inverse_formula.json"))
        result = verify_fol_fin_eq(z2_domain, formula)
        assert result.status == "VERIFIED"
        assert result.witnesses is not None
        assert len(result.witnesses) > 0

    def test_z2_associativity_verified(self, z2_domain):
        """Z2 satisfies associativity axiom."""
        formula = parse_fol_formula((FIXTURES / "associativity_formula.json"))
        result = verify_fol_fin_eq(z2_domain, formula)
        assert result.status == "VERIFIED"
        assert result.quantifier_report["forall_vars"] == ["x", "y", "z"]
        assert result.enumeration_stats.assignments_checked == 8  # 2^3


class TestRefutation:
    """Refutation tests."""

    def test_broken_assoc_refuted(self):
        """Broken table produces REFUTED with counterexample."""
        domain = parse_domain_spec(FIXTURES / "z2_broken.json")
        formula = parse_fol_formula((FIXTURES / "associativity_formula.json"))
        result = verify_fol_fin_eq(domain, formula)
        assert result.status == "REFUTED"
        assert result.counterexample is not None
        assert "assignment" in result.counterexample


class TestDeterminism:
    """Determinism contract tests."""

    def test_determinism_identical_output(self):
        """Same input produces byte-identical result."""
        domain = parse_domain_spec(FIXTURES / "z2_domain.json")
        formula = parse_fol_formula((FIXTURES / "identity_formula.json"))
        result1 = verify_fol_fin_eq(domain, formula)
        result2 = verify_fol_fin_eq(domain, formula)
        assert result1 == result2


class TestFormulaDerivdedABSTAIN:
    """Tests for formula-derived assignment bound ABSTAIN."""

    def test_4var_domain20_abstains(self):
        """4 universally-bound vars on 20-element domain ABSTAINS (20^4 = 160,000 > 125,000)."""
        domain = parse_domain_spec(FIXTURES / "d20_domain.json")
        formula = parse_fol_formula((FIXTURES / "four_var_formula.json"))
        result = verify_fol_fin_eq(domain, formula)
        assert result.status == "ABSTAINED"
        assert result.resource_limit_reason == "ASSIGNMENT_COUNT_EXCEEDS_LIMIT"
        assert result.resource_limits is not None
        assert result.resource_limits.computed_estimate == 160000
        assert result.resource_limits.max_assignments == 125000


class TestDeterministicSelection:
    """Tests for deterministic witness/counterexample selection (Section 0.2)."""

    def test_counterexample_is_first_falsifying(self):
        """Counterexample for ∀x. x≠0 on Z₃ MUST be {x: "0"} (first in lex order).

        Fixture: z3_domain.json with elements ["0","1","2"]
        Formula: neq_zero_formula.json = ∀x. x≠0
        Expected: REFUTED, counterexample {x:"0"}, assignments_checked=1 (early exit)
        """
        domain = parse_domain_spec(FIXTURES / "z3_domain.json")
        formula = parse_fol_formula((FIXTURES / "neq_zero_formula.json"))
        result = verify_fol_fin_eq(domain, formula)
        assert result.status == "REFUTED"
        assert result.counterexample is not None
        assert result.counterexample["assignment"] == {"x": "0"}
        # Early exit: only checked first assignment before finding counterexample
        assert result.enumeration_stats.assignments_checked == 1

    def test_witness_unique_satisfying(self):
        """Witness for ∃y. y=2 on Z₃ MUST be {y: "2"} (unique satisfying element).

        Fixture: z3_domain.json with elements ["0","1","2"]
        Formula: exists_eq_two_formula.json = ∃y. y=two
        Expected: VERIFIED, witness {y:"2"}
        Note: Only "2" satisfies, so this tests unique witness identification.
        """
        domain = parse_domain_spec(FIXTURES / "z3_domain.json")
        formula = parse_fol_formula((FIXTURES / "exists_eq_two_formula.json"))
        result = verify_fol_fin_eq(domain, formula)
        assert result.status == "VERIFIED"
        assert result.witnesses is not None
        assert result.witnesses["y"]["witness_value"] == "2"

    def test_multiple_satisfying_uses_first(self):
        """When MULTIPLE elements satisfy ∃, the FIRST in array order is recorded.

        Fixture: idempotent_domain.json with elements ["a","b","c"], all idempotent
        Formula: multiple_witness_formula.json = ∃y. y*y=y
        Expected: All of a,b,c satisfy (a*a=a, b*b=b, c*c=c), witness MUST be "a" (first)
        """
        domain = parse_domain_spec(FIXTURES / "idempotent_domain.json")
        formula = parse_fol_formula((FIXTURES / "multiple_witness_formula.json"))
        result = verify_fol_fin_eq(domain, formula)
        assert result.status == "VERIFIED"
        assert result.witnesses is not None
        assert result.witnesses["y"]["witness_value"] == "a"  # First of three satisfying


class TestEnumerationSemantics:
    """Tests for enumeration order and semantics (Section 0.2 + Semantics Lock)."""

    def test_forall_true_checks_all_assignments(self):
        """When ∀ formula is always true, ALL assignments must be checked (no early exit).

        Fixture: z2_domain.json with elements ["0","1"]
        Formula: forall_always_true_formula.json = ∀x. x=x (reflexivity)
        Expected: VERIFIED, assignments_checked=2 (all elements checked)
        """
        domain = parse_domain_spec(FIXTURES / "z2_domain.json")
        formula = parse_fol_formula((FIXTURES / "forall_always_true_formula.json"))
        result = verify_fol_fin_eq(domain, formula)
        assert result.status == "VERIFIED"
        assert result.enumeration_stats.assignments_checked == 2  # |D| = 2, all checked

    def test_forall_false_exits_on_first_falsifying(self):
        """When ∀ formula falsifies on first element, early exit with assignments_checked=1.

        Fixture: z3_domain.json with elements ["0","1","2"]
        Formula: neq_zero_formula.json = ∀x. x≠0
        Expected: REFUTED, counterexample {x:"0"}, assignments_checked=1
        """
        domain = parse_domain_spec(FIXTURES / "z3_domain.json")
        formula = parse_fol_formula((FIXTURES / "neq_zero_formula.json"))
        result = verify_fol_fin_eq(domain, formula)
        assert result.status == "REFUTED"
        assert result.counterexample["assignment"] == {"x": "0"}
        assert result.enumeration_stats.assignments_checked == 1  # Early exit

    def test_nested_quantifiers_count_same(self):
        """Nested quantifiers count same as prefix for assignment bound."""
        domain = parse_domain_spec(FIXTURES / "z2_domain.json")
        formula = parse_fol_formula((FIXTURES / "inverse_formula.json"))  # ∀x.∃y.P
        report = extract_quantifier_report(formula)
        # Total vars = 2
        total_vars = len(report)
        assert total_vars == 2
        # Bound = 2² = 4
        bound = estimate_assignment_upper_bound(report, 2)
        assert bound == 4


class TestVariableOrderCanary:
    """Canary tests proving variable binding order is semantic (Section R12)."""

    def test_variable_order_changes_witness(self):
        """Different variable binding order → different FIRST witness.

        Formula XY: ∃x.∃y. check(x,y)=T  — x binds first, witness is (a,b)
        Formula YX: ∃y.∃x. check(x,y)=T  — y binds first, witness is (b,a)

        Domain: variable_order_domain.json with check(a,b)=T, check(b,a)=T, others=F

        This proves that variable order is semantically meaningful.
        """
        domain = parse_domain_spec(FIXTURES / "variable_order_domain.json")

        # Formula XY: ∃x.∃y. check(x,y)=T
        formula_xy = parse_fol_formula((FIXTURES / "variable_order_witness_formula_xy.json"))
        result_xy = verify_fol_fin_eq(domain, formula_xy)
        assert result_xy.status == "VERIFIED"

        # Formula YX: ∃y.∃x. check(x,y)=T
        formula_yx = parse_fol_formula((FIXTURES / "variable_order_witness_formula_yx.json"))
        result_yx = verify_fol_fin_eq(domain, formula_yx)
        assert result_yx.status == "VERIFIED"

        # Witnesses MUST be DIFFERENT
        witness_xy = (result_xy.witnesses["x"]["witness_value"],
                      result_xy.witnesses["y"]["witness_value"])
        witness_yx = (result_yx.witnesses["x"]["witness_value"],
                      result_yx.witnesses["y"]["witness_value"])

        assert witness_xy == ("a", "b"), f"Expected (a,b), got {witness_xy}"
        assert witness_yx == ("b", "a"), f"Expected (b,a), got {witness_yx}"
        assert witness_xy != witness_yx, "Witnesses must differ to prove order is semantic"
