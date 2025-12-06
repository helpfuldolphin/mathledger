"""
Tests for U2 Safe Evaluation Module

Validates safe evaluation with lint mode and security constraints.
"""

import pytest
from experiments.u2.u2_safe_eval import (
    lint_expression,
    safe_eval,
    SafeEvalLintResult,
    batch_lint_expressions,
)


class TestLintExpression:
    """Tests for expression linting."""
    
    def test_lint_safe_arithmetic(self):
        """Test that safe arithmetic passes linting."""
        result = lint_expression("2 + 2")
        assert result.is_safe
        assert len(result.issues) == 0
        assert len(result.dangerous_nodes) == 0
    
    def test_lint_safe_comparison(self):
        """Test that safe comparisons pass linting."""
        result = lint_expression("10 > 5")
        assert result.is_safe
        assert len(result.issues) == 0
    
    def test_lint_dangerous_import(self):
        """Test that import statements are flagged."""
        result = lint_expression("import os")
        assert not result.is_safe
        assert "Import" in result.dangerous_nodes
        assert any("Dangerous" in issue for issue in result.issues)
    
    def test_lint_dangerous_call(self):
        """Test that function calls are flagged."""
        result = lint_expression("print('hello')")
        assert not result.is_safe
        assert "Call" in result.dangerous_nodes
    
    def test_lint_dangerous_attribute(self):
        """Test that attribute access is flagged."""
        result = lint_expression("os.system")
        assert not result.is_safe
        assert "Attribute" in result.dangerous_nodes
    
    def test_lint_empty_expression(self):
        """Test that empty expressions are rejected."""
        result = lint_expression("")
        assert not result.is_safe
        assert "Empty expression" in result.issues
    
    def test_lint_syntax_error(self):
        """Test that syntax errors are caught."""
        result = lint_expression("2 +")
        assert not result.is_safe
        assert any("Syntax error" in issue for issue in result.issues)
    
    def test_lint_with_allowed_names(self):
        """Test linting with allowed variable names."""
        result = lint_expression("x + y", allowed_names={"x", "y"})
        assert result.is_safe
        
        result = lint_expression("x + z", allowed_names={"x", "y"})
        assert not result.is_safe
        assert any("Disallowed variable name: z" in issue for issue in result.issues)


class TestSafeEval:
    """Tests for safe evaluation."""
    
    def test_safe_eval_arithmetic(self):
        """Test safe evaluation of arithmetic."""
        result = safe_eval("2 + 2")
        assert result == 4
        
        result = safe_eval("10 * 5")
        assert result == 50
    
    def test_safe_eval_comparison(self):
        """Test safe evaluation of comparisons."""
        result = safe_eval("10 > 5")
        assert result is True
        
        result = safe_eval("3 == 3")
        assert result is True
    
    def test_safe_eval_with_allowed_names(self):
        """Test evaluation with allowed variables."""
        result = safe_eval("x + y", allowed_names={"x": 10, "y": 5})
        assert result == 15
    
    def test_safe_eval_dangerous_import_raises(self):
        """Test that dangerous imports raise errors."""
        with pytest.raises(ValueError, match="Unsafe expression"):
            safe_eval("import os")
    
    def test_safe_eval_dangerous_call_raises(self):
        """Test that function calls raise errors."""
        with pytest.raises(ValueError, match="Unsafe expression"):
            safe_eval("print('test')")
    
    def test_safe_eval_lint_only_mode(self):
        """Test lint-only mode returns SafeEvalLintResult."""
        result = safe_eval("2 + 2", lint_only=True)
        assert isinstance(result, SafeEvalLintResult)
        assert result.is_safe
    
    def test_safe_eval_safe_builtins(self):
        """Test that safe builtins are available."""
        result = safe_eval("abs(-5)")
        assert result == 5
        
        result = safe_eval("max(1, 2, 3)")
        assert result == 3
        
        result = safe_eval("min(1, 2, 3)")
        assert result == 1
    
    def test_safe_eval_no_unsafe_builtins(self):
        """Test that unsafe builtins are not available."""
        with pytest.raises(Exception):
            safe_eval("open('/etc/passwd')")
        
        with pytest.raises(Exception):
            safe_eval("exec('print(1)')")


class TestBatchLintExpressions:
    """Tests for batch expression linting."""
    
    def test_batch_lint_all_safe(self):
        """Test batch linting with all safe expressions."""
        expressions = ["1 + 1", "2 * 3", "10 > 5"]
        results = batch_lint_expressions(expressions)
        
        assert len(results) == 3
        assert all(r.is_safe for r in results)
    
    def test_batch_lint_mixed_safety(self):
        """Test batch linting with mixed safety."""
        expressions = ["1 + 1", "import os", "2 * 3"]
        results = batch_lint_expressions(expressions)
        
        assert len(results) == 3
        assert results[0].is_safe
        assert not results[1].is_safe
        assert results[2].is_safe
    
    def test_batch_lint_empty_list(self):
        """Test batch linting with empty list."""
        results = batch_lint_expressions([])
        assert len(results) == 0


@pytest.mark.unit
class TestSafeEvalIntegration:
    """Integration tests for safe eval."""
    
    def test_complex_arithmetic(self):
        """Test complex arithmetic expressions."""
        result = safe_eval("(2 + 3) * 4 - 1")
        assert result == 19
        
        result = safe_eval("2 ** 3")
        assert result == 8
    
    def test_boolean_logic(self):
        """Test boolean logic expressions."""
        result = safe_eval("True and False")
        assert result is False
        
        result = safe_eval("True or False")
        assert result is True
        
        result = safe_eval("not False")
        assert result is True
