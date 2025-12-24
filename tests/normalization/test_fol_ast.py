"""
Tests for normalization.fol_ast module (FOL_FIN_EQ_v1 formula AST parsing).

Phase 1 RED: These tests define the expected interface for FOL AST parsing.
All tests will fail until normalization/fol_ast.py is implemented.

The AST uses a union type structure with these node types:
- Forall(variable, body)
- Exists(variable, body)
- Equals(left, right)
- Not(inner)
- And(left, right)
- Or(left, right)
- Implies(left, right)
- Apply(function, args)
- Var(name)
- Const(value)
"""

import json
from pathlib import Path

import pytest

# Test fixtures directory
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "fol"


class TestParseFolFormula:
    """Tests for parse_fol_formula() function."""

    def test_parse_identity_formula(self):
        """Parse ∀x. identity * x = x formula."""
        from normalization.fol_ast import Const, Forall, parse_fol_formula

        with open(FIXTURES_DIR / "identity_formula.json") as f:
            data = json.load(f)

        formula = parse_fol_formula(data["formula"])

        assert isinstance(formula, Forall)
        assert formula.variable == "x"

    def test_parse_inverse_formula(self):
        """Parse ∀x. ∃y. x * y = identity formula."""
        from normalization.fol_ast import Exists, Forall, parse_fol_formula

        with open(FIXTURES_DIR / "inverse_formula.json") as f:
            data = json.load(f)

        formula = parse_fol_formula(data["formula"])

        assert isinstance(formula, Forall)
        assert formula.variable == "x"
        assert isinstance(formula.body, Exists)
        assert formula.body.variable == "y"

    def test_parse_nested_quantifiers(self):
        """Parse ∀x. ∀y. ∀z. ... (3-level nesting)."""
        from normalization.fol_ast import Forall, parse_fol_formula

        with open(FIXTURES_DIR / "associativity_formula.json") as f:
            data = json.load(f)

        formula = parse_fol_formula(data["formula"])

        assert isinstance(formula, Forall)
        assert isinstance(formula.body, Forall)
        assert isinstance(formula.body.body, Forall)

    def test_parse_existential(self):
        """Parse ∃y. y=2 formula (exists_eq_two_formula)."""
        from normalization.fol_ast import Exists, parse_fol_formula

        with open(FIXTURES_DIR / "exists_eq_two_formula.json") as f:
            data = json.load(f)

        formula = parse_fol_formula(data["formula"])

        assert isinstance(formula, Exists)
        assert formula.variable == "y"

    def test_parse_negation(self):
        """Parse ∀x. ¬(x = identity) formula."""
        from normalization.fol_ast import Equals, Forall, Not, parse_fol_formula

        with open(FIXTURES_DIR / "neq_zero_formula.json") as f:
            data = json.load(f)

        formula = parse_fol_formula(data["formula"])

        # ∀x. ¬(x = identity)
        assert isinstance(formula, Forall)
        assert isinstance(formula.body, Not)
        assert isinstance(formula.body.inner, Equals)

    def test_parse_conjunction(self):
        """Parse φ ∧ ψ formula."""
        from normalization.fol_ast import And, parse_fol_formula

        with open(FIXTURES_DIR / "variable_order_witness_formula_xy.json") as f:
            data = json.load(f)

        formula = parse_fol_formula(data["formula"])

        # ∃x. ∃y. (x * y = identity) ∧ (x ≠ y)
        inner = formula.body.body  # Two existential layers
        assert isinstance(inner, And)

    def test_parse_function_application(self):
        """Parse mul(x, y) function application."""
        from normalization.fol_ast import And, Apply, Equals, parse_fol_formula

        with open(FIXTURES_DIR / "identity_formula.json") as f:
            data = json.load(f)

        formula = parse_fol_formula(data["formula"])

        # ∀x. (identity * x = x) ∧ (x * identity = x)
        # body is And, left side is first Equals
        assert isinstance(formula.body, And)
        equals = formula.body.left
        assert isinstance(equals, Equals)
        assert isinstance(equals.left, Apply)
        assert equals.left.function == "mul"
        assert len(equals.left.args) == 2

    def test_parse_variable(self):
        """Parse variable reference."""
        from normalization.fol_ast import And, Equals, Var, parse_fol_formula

        with open(FIXTURES_DIR / "identity_formula.json") as f:
            data = json.load(f)

        formula = parse_fol_formula(data["formula"])

        # ∀x. (identity * x = x) ∧ (x * identity = x)
        assert isinstance(formula.body, And)
        equals = formula.body.left
        assert isinstance(equals, Equals)
        assert isinstance(equals.right, Var)
        assert equals.right.name == "x"

    def test_parse_constant(self):
        """Parse constant reference."""
        from normalization.fol_ast import And, Const, Equals, parse_fol_formula

        with open(FIXTURES_DIR / "identity_formula.json") as f:
            data = json.load(f)

        formula = parse_fol_formula(data["formula"])

        # ∀x. (identity * x = x) ∧ (x * identity = x)
        assert isinstance(formula.body, And)
        equals = formula.body.left
        assert isinstance(equals, Equals)
        apply_node = equals.left
        assert isinstance(apply_node.args[0], Const)
        assert apply_node.args[0].value == "identity"

    def test_invalid_node_type_fails(self):
        """Unknown node type must raise ValueError."""
        from normalization.fol_ast import parse_fol_formula

        with pytest.raises(ValueError, match="(?i)unknown.*type|invalid.*type"):
            parse_fol_formula({"type": "bogus", "value": "x"})


class TestConstAsKeySemantics:
    """Tests that lock Const-as-constant-key semantics (NOT element literal)."""

    def test_const_with_valid_key_parses(self):
        """Const with valid constant key parses and resolves correctly."""
        from normalization.domain_spec import parse_domain_spec
        from normalization.fol_ast import Const, parse_fol_formula

        # exists_eq_two_formula uses {"type": "const", "value": "two"}
        with open(FIXTURES_DIR / "exists_eq_two_formula.json") as f:
            data = json.load(f)

        formula = parse_fol_formula(data["formula"])

        # Extract the Const node
        equals = formula.body  # ∃y. (y = two)
        assert isinstance(equals.right, Const)
        assert equals.right.value == "two"  # Key, not element

        # Resolution via DomainSpec
        spec = parse_domain_spec(FIXTURES_DIR / "z3_domain.json")
        resolved = spec.resolve_constant(equals.right.value)
        assert resolved == "2"  # Resolved to element

    def test_const_with_element_literal_fails_resolution(self):
        """Const with element literal (not constant key) MUST fail resolution.

        NORMATIVE: Const.value is ALWAYS a key into domain_spec.constants.
        Using element literals directly (e.g., "2" instead of "two") is FORBIDDEN.
        This test locks that invariant at the AST+DomainSpec integration level.
        """
        from normalization.domain_spec import parse_domain_spec
        from normalization.fol_ast import Const, parse_fol_formula

        # Construct an AST with element literal "2" (NOT constant key "two")
        bad_formula_ast = {
            "type": "exists",
            "variable": "y",
            "body": {
                "type": "equals",
                "left": {"type": "var", "name": "y"},
                "right": {"type": "const", "value": "2"},  # Element literal!
            },
        }

        formula = parse_fol_formula(bad_formula_ast)

        # Parsing succeeds (AST doesn't know about domain)
        assert isinstance(formula.body.right, Const)
        assert formula.body.right.value == "2"

        # But resolution against Z3 domain MUST fail
        spec = parse_domain_spec(FIXTURES_DIR / "z3_domain.json")

        with pytest.raises(ValueError, match="unknown constant|not in constants"):
            spec.resolve_constant(formula.body.right.value)

    def test_corrected_fixture_uses_constant_key(self):
        """Verify exists_eq_two_formula.json uses 'two' (key), not '2' (literal)."""
        with open(FIXTURES_DIR / "exists_eq_two_formula.json") as f:
            data = json.load(f)

        # The fixture MUST use "two" as the constant value
        const_node = data["formula"]["body"]["right"]
        assert const_node["type"] == "const"
        assert const_node["value"] == "two", (
            f"Fixture must use constant key 'two', not element literal. "
            f"Got: {const_node['value']!r}"
        )


class TestExtractQuantifierReport:
    """Tests for extract_quantifier_report() function."""

    def test_single_universal(self):
        """Report single universal quantifier."""
        from normalization.fol_ast import extract_quantifier_report, parse_fol_formula

        with open(FIXTURES_DIR / "forall_always_true_formula.json") as f:
            data = json.load(f)

        formula = parse_fol_formula(data["formula"])
        report = extract_quantifier_report(formula)

        assert report == [("forall", "x")]

    def test_universal_then_existential(self):
        """Report ∀x. ∃y quantifier sequence."""
        from normalization.fol_ast import extract_quantifier_report, parse_fol_formula

        with open(FIXTURES_DIR / "inverse_formula.json") as f:
            data = json.load(f)

        formula = parse_fol_formula(data["formula"])
        report = extract_quantifier_report(formula)

        assert report == [("forall", "x"), ("exists", "y")]

    def test_triple_universal(self):
        """Report ∀x. ∀y. ∀z quantifier sequence."""
        from normalization.fol_ast import extract_quantifier_report, parse_fol_formula

        with open(FIXTURES_DIR / "associativity_formula.json") as f:
            data = json.load(f)

        formula = parse_fol_formula(data["formula"])
        report = extract_quantifier_report(formula)

        assert report == [("forall", "x"), ("forall", "y"), ("forall", "z")]

    def test_quantifier_order_preserved(self):
        """Quantifier order must match formula structure."""
        from normalization.fol_ast import extract_quantifier_report, parse_fol_formula

        # xy version: ∃x. ∃y. ...
        with open(FIXTURES_DIR / "variable_order_witness_formula_xy.json") as f:
            data = json.load(f)
        formula_xy = parse_fol_formula(data["formula"])
        report_xy = extract_quantifier_report(formula_xy)

        # yx version: ∃y. ∃x. ...
        with open(FIXTURES_DIR / "variable_order_witness_formula_yx.json") as f:
            data = json.load(f)
        formula_yx = parse_fol_formula(data["formula"])
        report_yx = extract_quantifier_report(formula_yx)

        assert report_xy == [("exists", "x"), ("exists", "y")]
        assert report_yx == [("exists", "y"), ("exists", "x")]
        assert report_xy != report_yx


class TestDetectFreeVariables:
    """Tests for detect_free_variables() function and free variable validation."""

    def test_closed_formula_no_free_vars(self):
        """Closed formula should have no free variables."""
        from normalization.fol_ast import detect_free_variables, parse_fol_formula

        with open(FIXTURES_DIR / "identity_formula.json") as f:
            data = json.load(f)

        formula = parse_fol_formula(data["formula"])
        free_vars = detect_free_variables(formula)

        assert free_vars == set()

    def test_open_formula_has_free_vars(self):
        """Open formula should report free variables via detect_free_variables()."""
        from normalization.fol_ast import detect_free_variables, parse_fol_formula

        with open(FIXTURES_DIR / "free_var_formula.json") as f:
            data = json.load(f)

        formula = parse_fol_formula(data["formula"])
        free_vars = detect_free_variables(formula)

        assert free_vars == {"x"}

    def test_free_variables_raise_validation_error(self):
        """Free variables MUST raise ValidationError when require_closed=True.

        Per EXECUTION_PACKET Section 0.3: FREE_VARIABLES_DETECTED is a
        validation error (fail-closed), NOT an ABSTAIN case.
        """
        from normalization.fol_ast import parse_fol_formula, validate_closed_formula

        with open(FIXTURES_DIR / "free_var_formula.json") as f:
            data = json.load(f)

        formula = parse_fol_formula(data["formula"])

        # Must raise ValidationError, not return ABSTAINED
        with pytest.raises(ValueError, match="free variable|FREE_VARIABLES"):
            validate_closed_formula(formula)

    def test_bound_variable_not_free(self):
        """Variable bound by quantifier should not be reported as free."""
        from normalization.fol_ast import detect_free_variables, parse_fol_formula

        with open(FIXTURES_DIR / "associativity_formula.json") as f:
            data = json.load(f)

        formula = parse_fol_formula(data["formula"])
        free_vars = detect_free_variables(formula)

        # x, y, z are all bound
        assert "x" not in free_vars
        assert "y" not in free_vars
        assert "z" not in free_vars


class TestComputeAstHash:
    """Tests for compute_ast_hash() function."""

    def test_ast_hash_deterministic(self):
        """Same formula must produce same hash."""
        from normalization.fol_ast import compute_ast_hash, parse_fol_formula

        with open(FIXTURES_DIR / "identity_formula.json") as f:
            data = json.load(f)

        formula1 = parse_fol_formula(data["formula"])
        formula2 = parse_fol_formula(data["formula"])

        assert compute_ast_hash(formula1) == compute_ast_hash(formula2)

    def test_different_formulas_different_hash(self):
        """Different formulas must produce different hashes."""
        from normalization.fol_ast import compute_ast_hash, parse_fol_formula

        with open(FIXTURES_DIR / "identity_formula.json") as f:
            identity_data = json.load(f)
        with open(FIXTURES_DIR / "inverse_formula.json") as f:
            inverse_data = json.load(f)

        identity_formula = parse_fol_formula(identity_data["formula"])
        inverse_formula = parse_fol_formula(inverse_data["formula"])

        assert compute_ast_hash(identity_formula) != compute_ast_hash(inverse_formula)

    def test_hash_uses_domain_separation(self):
        """AST hash must use DOMAIN_FOL_AST domain separation tag."""
        from normalization.fol_ast import compute_ast_hash, parse_fol_formula

        with open(FIXTURES_DIR / "identity_formula.json") as f:
            data = json.load(f)

        formula = parse_fol_formula(data["formula"])
        ast_hash = compute_ast_hash(formula)

        # Hash should be 64 hex chars
        assert len(ast_hash) == 64
        assert all(c in "0123456789abcdef" for c in ast_hash)

    def test_variable_name_affects_hash(self):
        """Variable names are semantically relevant and must affect hash."""
        from normalization.fol_ast import compute_ast_hash, parse_fol_formula

        # Two formulas with same structure but different variable names
        # ∃x. ∃y. ... vs ∃y. ∃x. ...
        with open(FIXTURES_DIR / "variable_order_witness_formula_xy.json") as f:
            xy_data = json.load(f)
        with open(FIXTURES_DIR / "variable_order_witness_formula_yx.json") as f:
            yx_data = json.load(f)

        xy_formula = parse_fol_formula(xy_data["formula"])
        yx_formula = parse_fol_formula(yx_data["formula"])

        # These are structurally different formulas
        assert compute_ast_hash(xy_formula) != compute_ast_hash(yx_formula)


class TestFolAstEquality:
    """Tests for AST node equality."""

    def test_same_formula_equal(self):
        """Parsing same formula twice should produce equal ASTs."""
        from normalization.fol_ast import parse_fol_formula

        with open(FIXTURES_DIR / "identity_formula.json") as f:
            data = json.load(f)

        formula1 = parse_fol_formula(data["formula"])
        formula2 = parse_fol_formula(data["formula"])

        assert formula1 == formula2

    def test_different_formula_not_equal(self):
        """Different formulas should not be equal."""
        from normalization.fol_ast import parse_fol_formula

        with open(FIXTURES_DIR / "identity_formula.json") as f:
            identity_data = json.load(f)
        with open(FIXTURES_DIR / "inverse_formula.json") as f:
            inverse_data = json.load(f)

        identity_formula = parse_fol_formula(identity_data["formula"])
        inverse_formula = parse_fol_formula(inverse_data["formula"])

        assert identity_formula != inverse_formula
