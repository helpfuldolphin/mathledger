# PHASE II — NOT USED IN PHASE I
#
# Unit tests for the u2_safe_eval module.

import pytest

from experiments.u2_safe_eval import safe_eval_arithmetic, _eval_ast_node


class TestSafeEvalArithmetic:
    """Tests for safe_eval_arithmetic function."""

    def test_addition(self) -> None:
        """Test simple addition."""
        assert safe_eval_arithmetic("1 + 2") == 3
        assert safe_eval_arithmetic("10 + 20") == 30

    def test_subtraction(self) -> None:
        """Test simple subtraction."""
        assert safe_eval_arithmetic("5 - 3") == 2
        assert safe_eval_arithmetic("10 - 20") == -10

    def test_multiplication(self) -> None:
        """Test simple multiplication."""
        assert safe_eval_arithmetic("3 * 4") == 12
        assert safe_eval_arithmetic("7 * 8") == 56

    def test_division(self) -> None:
        """Test simple division."""
        assert safe_eval_arithmetic("10 / 2") == 5.0
        assert safe_eval_arithmetic("7 / 2") == 3.5

    def test_division_by_zero(self) -> None:
        """Test that division by zero returns None."""
        assert safe_eval_arithmetic("1 / 0") is None
        assert safe_eval_arithmetic("100 / 0") is None

    def test_unary_operators(self) -> None:
        """Test unary plus and minus."""
        assert safe_eval_arithmetic("-5") == -5
        assert safe_eval_arithmetic("+5") == 5
        assert safe_eval_arithmetic("--5") == 5

    def test_float_literals(self) -> None:
        """Test float literals."""
        assert safe_eval_arithmetic("3.14") == 3.14
        assert safe_eval_arithmetic("1.5 + 2.5") == 4.0

    def test_complex_expression(self) -> None:
        """Test complex arithmetic expressions."""
        assert safe_eval_arithmetic("3 * 4 + 5") == 17
        assert safe_eval_arithmetic("10 - 2 * 3") == 4
        assert safe_eval_arithmetic("(1 + 2) * 3") == 9

    def test_negative_numbers(self) -> None:
        """Test negative number handling."""
        assert safe_eval_arithmetic("-3 + 5") == 2
        assert safe_eval_arithmetic("5 + -3") == 2

    def test_rejects_variable_names(self) -> None:
        """Test that variable names are rejected."""
        with pytest.raises(ValueError, match="Unsupported AST node type"):
            safe_eval_arithmetic("x + 1")

    def test_rejects_function_calls(self) -> None:
        """Test that function calls are rejected."""
        with pytest.raises(ValueError, match="Unsupported AST node type"):
            safe_eval_arithmetic("abs(-5)")

    def test_rejects_attribute_access(self) -> None:
        """Test that attribute access is rejected."""
        with pytest.raises(ValueError, match="Unsupported AST node type"):
            safe_eval_arithmetic("foo.bar")

    def test_rejects_import_attempts(self) -> None:
        """Test that import attempts are rejected."""
        with pytest.raises(ValueError, match="Unsupported AST node type"):
            safe_eval_arithmetic('__import__("os")')

    def test_rejects_string_constants(self) -> None:
        """Test that string constants are rejected."""
        with pytest.raises(ValueError, match="Unsupported constant type"):
            safe_eval_arithmetic('"hello"')

    def test_rejects_list_literals(self) -> None:
        """Test that list literals are rejected."""
        with pytest.raises(ValueError, match="Unsupported AST node type"):
            safe_eval_arithmetic("[1, 2, 3]")

    def test_syntax_error(self) -> None:
        """Test that syntax errors propagate."""
        with pytest.raises(SyntaxError):
            safe_eval_arithmetic("1 +")

    def test_empty_string(self) -> None:
        """Test that empty string raises syntax error."""
        with pytest.raises(SyntaxError):
            safe_eval_arithmetic("")

    def test_whitespace_handling(self) -> None:
        """Test that whitespace is handled correctly."""
        assert safe_eval_arithmetic("1 + 2") == 3
        assert safe_eval_arithmetic("1+2") == 3
        # Note: Leading whitespace causes IndentationError in ast.parse
        # This is expected Python behavior, not a bug in our code
