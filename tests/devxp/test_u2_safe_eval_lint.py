"""
Tests for U2 Safe Eval Lint Mode

This module tests the safe_eval_arithmetic function with lint mode enabled,
ensuring that diagnostic information is provided without weakening security.

PHASE II — NOT USED IN PHASE I
"""

import pytest
from experiments.u2_safe_eval import (
    safe_eval_arithmetic,
    safe_eval_lint,
    SafeEvalLintResult,
)


class TestSafeEvalArithmeticBasic:
    """Test basic arithmetic evaluation (non-lint mode)."""
    
    def test_simple_addition(self):
        result = safe_eval_arithmetic("2 + 3")
        assert result == 5.0
    
    def test_multiplication_and_subtraction(self):
        result = safe_eval_arithmetic("10 - 2 * 3")
        assert result == 4.0
    
    def test_parentheses(self):
        result = safe_eval_arithmetic("(10 - 2) * 3")
        assert result == 24.0
    
    def test_division(self):
        result = safe_eval_arithmetic("15 / 3")
        assert result == 5.0
    
    def test_floor_division(self):
        result = safe_eval_arithmetic("17 // 3")
        assert result == 5.0
    
    def test_modulo(self):
        result = safe_eval_arithmetic("17 % 3")
        assert result == 2.0
    
    def test_power(self):
        result = safe_eval_arithmetic("2 ** 3")
        assert result == 8.0
    
    def test_unary_minus(self):
        result = safe_eval_arithmetic("-(5 + 3)")
        assert result == -8.0
    
    def test_unary_plus(self):
        result = safe_eval_arithmetic("+5")
        assert result == 5.0
    
    def test_float_literals(self):
        result = safe_eval_arithmetic("2.5 + 3.7")
        assert abs(result - 6.2) < 1e-10
    
    def test_complex_expression(self):
        result = safe_eval_arithmetic("(2 + 3) * 4 - 10 / 2")
        assert result == 15.0


class TestSafeEvalArithmeticRejections:
    """Test that forbidden constructs are rejected (non-lint mode)."""
    
    def test_variable_name_rejected(self):
        result = safe_eval_arithmetic("x + 1")
        assert result is None
    
    def test_function_call_rejected(self):
        result = safe_eval_arithmetic("abs(-5)")
        assert result is None
    
    def test_attribute_access_rejected(self):
        result = safe_eval_arithmetic("math.pi")
        assert result is None
    
    def test_list_literal_rejected(self):
        result = safe_eval_arithmetic("[1, 2, 3]")
        assert result is None
    
    def test_string_literal_rejected(self):
        result = safe_eval_arithmetic("'hello'")
        assert result is None
    
    def test_import_rejected(self):
        result = safe_eval_arithmetic("import os")
        assert result is None
    
    def test_lambda_rejected(self):
        result = safe_eval_arithmetic("lambda x: x + 1")
        assert result is None
    
    def test_empty_expression(self):
        result = safe_eval_arithmetic("")
        assert result is None
    
    def test_whitespace_only(self):
        result = safe_eval_arithmetic("   ")
        assert result is None


class TestSafeEvalLintModeValid:
    """Test lint mode with valid expressions."""
    
    def test_simple_addition_lint(self):
        result = safe_eval_arithmetic("2 + 3", lint=True)
        assert isinstance(result, SafeEvalLintResult)
        assert result.is_valid is True
        assert result.reason is None
        assert result.first_illegal_node_type is None
    
    def test_complex_expression_lint(self):
        result = safe_eval_arithmetic("(2 + 3) * 4 - 10 / 2", lint=True)
        assert isinstance(result, SafeEvalLintResult)
        assert result.is_valid is True
    
    def test_float_literals_lint(self):
        result = safe_eval_arithmetic("2.5 + 3.7", lint=True)
        assert isinstance(result, SafeEvalLintResult)
        assert result.is_valid is True
    
    def test_unary_operators_lint(self):
        result = safe_eval_arithmetic("-(+5)", lint=True)
        assert isinstance(result, SafeEvalLintResult)
        assert result.is_valid is True
    
    def test_all_operators_lint(self):
        result = safe_eval_arithmetic("2 + 3 - 4 * 5 / 2 // 1 % 3 ** 2", lint=True)
        assert isinstance(result, SafeEvalLintResult)
        assert result.is_valid is True


class TestSafeEvalLintModeInvalid:
    """Test lint mode with invalid expressions - should provide diagnostics."""
    
    def test_variable_name_lint(self):
        result = safe_eval_arithmetic("x + 1", lint=True)
        assert isinstance(result, SafeEvalLintResult)
        assert result.is_valid is False
        assert result.reason is not None
        assert "Variable names not allowed" in result.reason
        assert "x" in result.reason
        assert result.first_illegal_node_type == "Name"
    
    def test_function_call_lint(self):
        result = safe_eval_arithmetic("abs(-5)", lint=True)
        assert isinstance(result, SafeEvalLintResult)
        assert result.is_valid is False
        assert result.reason is not None
        assert "Function calls not allowed" in result.reason
        assert result.first_illegal_node_type == "Call"
    
    def test_attribute_access_lint(self):
        result = safe_eval_arithmetic("math.pi", lint=True)
        assert isinstance(result, SafeEvalLintResult)
        assert result.is_valid is False
        assert result.reason is not None
        assert "not allowed" in result.reason.lower()
        # Could be either Name (math) or Attribute depending on parse order
        assert result.first_illegal_node_type in ["Name", "Attribute"]
    
    def test_string_literal_lint(self):
        result = safe_eval_arithmetic("'hello'", lint=True)
        assert isinstance(result, SafeEvalLintResult)
        assert result.is_valid is False
        assert result.reason is not None
        assert "numeric" in result.reason.lower()
    
    def test_list_literal_lint(self):
        result = safe_eval_arithmetic("[1, 2, 3]", lint=True)
        assert isinstance(result, SafeEvalLintResult)
        assert result.is_valid is False
        assert result.reason is not None
        assert result.first_illegal_node_type == "List"
    
    def test_lambda_lint(self):
        result = safe_eval_arithmetic("lambda x: x + 1", lint=True)
        assert isinstance(result, SafeEvalLintResult)
        assert result.is_valid is False
        assert result.reason is not None
        assert "Lambda" in result.reason
        assert result.first_illegal_node_type == "Lambda"
    
    def test_comprehension_lint(self):
        result = safe_eval_arithmetic("[x for x in range(10)]", lint=True)
        assert isinstance(result, SafeEvalLintResult)
        assert result.is_valid is False
        assert result.reason is not None
        assert "Comprehension" in result.reason
        assert result.first_illegal_node_type == "ListComp"
    
    def test_subscript_lint(self):
        result = safe_eval_arithmetic("x[0]", lint=True)
        assert isinstance(result, SafeEvalLintResult)
        assert result.is_valid is False
        assert result.reason is not None
        assert result.first_illegal_node_type in ["Name", "Subscript"]
    
    def test_empty_expression_lint(self):
        result = safe_eval_arithmetic("", lint=True)
        assert isinstance(result, SafeEvalLintResult)
        assert result.is_valid is False
        assert result.reason is not None
        assert "Empty" in result.reason
    
    def test_syntax_error_lint(self):
        result = safe_eval_arithmetic("2 +", lint=True)
        assert isinstance(result, SafeEvalLintResult)
        assert result.is_valid is False
        assert result.reason is not None
        assert "Syntax error" in result.reason


class TestSafeEvalLintFunction:
    """Test the convenience safe_eval_lint function."""
    
    def test_lint_function_valid(self):
        result = safe_eval_lint("2 + 3")
        assert isinstance(result, SafeEvalLintResult)
        assert result.is_valid is True
    
    def test_lint_function_invalid(self):
        result = safe_eval_lint("x + 1")
        assert isinstance(result, SafeEvalLintResult)
        assert result.is_valid is False
        assert result.reason is not None
        assert "Variable" in result.reason


class TestSafeEvalEdgeCases:
    """Test edge cases and error handling."""
    
    def test_division_by_zero_non_lint(self):
        result = safe_eval_arithmetic("1 / 0")
        assert result is None
    
    def test_division_by_zero_lint(self):
        # Note: Division by zero is a runtime error, not a parse-time error
        # Lint mode only checks AST structure, not runtime semantics
        result = safe_eval_arithmetic("1 / 0", lint=True)
        # The expression is syntactically valid
        assert isinstance(result, SafeEvalLintResult)
        assert result.is_valid is True
    
    def test_very_large_number(self):
        result = safe_eval_arithmetic("10 ** 100")
        assert result == 1e100
    
    def test_non_string_input_non_lint(self):
        result = safe_eval_arithmetic(123)  # type: ignore
        assert result is None
    
    def test_non_string_input_lint(self):
        result = safe_eval_arithmetic(123, lint=True)  # type: ignore
        assert isinstance(result, SafeEvalLintResult)
        assert result.is_valid is False
        assert "must be string" in result.reason


class TestSafeEvalNoSideEffects:
    """Test that evaluation has no side effects and is sandboxed."""
    
    def test_no_builtins_available(self):
        # Attempting to use builtins should fail
        result = safe_eval_arithmetic("__import__('os')")
        assert result is None
    
    def test_no_file_access(self):
        result = safe_eval_arithmetic("open('/etc/passwd')")
        assert result is None
    
    def test_no_exec(self):
        result = safe_eval_arithmetic("exec('print(1)')")
        assert result is None


class TestSafeEvalConsistency:
    """Test that lint mode and eval mode are consistent."""
    
    def test_valid_expression_consistency(self):
        expr = "2 + 3 * 4"
        lint_result = safe_eval_arithmetic(expr, lint=True)
        eval_result = safe_eval_arithmetic(expr, lint=False)
        
        assert isinstance(lint_result, SafeEvalLintResult)
        assert lint_result.is_valid is True
        assert eval_result is not None
        assert eval_result == 14.0
    
    def test_invalid_expression_consistency(self):
        expr = "x + 1"
        lint_result = safe_eval_arithmetic(expr, lint=True)
        eval_result = safe_eval_arithmetic(expr, lint=False)
        
        assert isinstance(lint_result, SafeEvalLintResult)
        assert lint_result.is_valid is False
        assert eval_result is None
    
    def test_multiple_invalid_expressions(self):
        invalid_exprs = [
            "import os",
            "lambda x: x",
            "x + y",
            "[1, 2, 3]",
            "{'a': 1}",
            "abs(-5)",
        ]
        
        for expr in invalid_exprs:
            lint_result = safe_eval_arithmetic(expr, lint=True)
            eval_result = safe_eval_arithmetic(expr, lint=False)
            
            assert isinstance(lint_result, SafeEvalLintResult), f"Failed for {expr}"
            assert lint_result.is_valid is False, f"Should be invalid: {expr}"
            assert eval_result is None, f"Should return None: {expr}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
