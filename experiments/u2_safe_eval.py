"""
Safe Evaluation Module for U2 Uplift Experiments

This module provides safe arithmetic expression evaluation with optional lint mode
for diagnostics. It enforces strict AST constraints to prevent code injection.

PHASE II — NOT USED IN PHASE I

Security Principles:
- Only allow arithmetic operations (+, -, *, /, //, %, **)
- Only allow numeric literals (int, float)
- No variable names, function calls, or other constructs
- Provide diagnostic feedback without weakening security
"""

import ast
from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class SafeEvalLintResult:
    """
    Result from safe_eval_arithmetic lint mode.
    
    Attributes:
        is_valid: Whether the expression passed all security checks
        reason: Human-readable explanation if invalid
        first_illegal_node_type: AST node type of first violation (if any)
    """
    is_valid: bool
    reason: Optional[str] = None
    first_illegal_node_type: Optional[str] = None


# Allowed AST node types for safe arithmetic evaluation
ALLOWED_NODE_TYPES = {
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,  # For numeric literals
    ast.Num,  # Legacy Python < 3.8 compatibility
    # Operators
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.UAdd,
    ast.USub,
}


def _check_ast_safety(node: ast.AST, lint_mode: bool = False) -> Optional[SafeEvalLintResult]:
    """
    Recursively check if AST contains only allowed node types.
    
    Args:
        node: AST node to check
        lint_mode: If True, return detailed lint result instead of None
        
    Returns:
        None if valid (lint_mode=False)
        SafeEvalLintResult if lint_mode=True
        
    Raises:
        ValueError: If node is invalid and lint_mode=False
    """
    # Check current node type
    if type(node) not in ALLOWED_NODE_TYPES:
        node_type_name = type(node).__name__
        reason = f"Forbidden AST node type: {node_type_name}"
        
        # Add more specific reasons for common violations
        if isinstance(node, ast.Name):
            reason = f"Variable names not allowed: '{node.id}'"
        elif isinstance(node, ast.Call):
            reason = "Function calls not allowed"
        elif isinstance(node, ast.Attribute):
            reason = "Attribute access not allowed"
        elif isinstance(node, ast.Subscript):
            reason = "Subscript/indexing not allowed"
        elif isinstance(node, ast.Lambda):
            reason = "Lambda functions not allowed"
        elif isinstance(node, ast.ListComp) or isinstance(node, ast.DictComp):
            reason = "Comprehensions not allowed"
        
        if lint_mode:
            return SafeEvalLintResult(
                is_valid=False,
                reason=reason,
                first_illegal_node_type=node_type_name
            )
        else:
            raise ValueError(reason)
    
    # Special check for Constant nodes - only allow numbers
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            reason = f"Only numeric constants allowed, got: {type(node.value).__name__}"
            if lint_mode:
                return SafeEvalLintResult(
                    is_valid=False,
                    reason=reason,
                    first_illegal_node_type="Constant (non-numeric)"
                )
            else:
                raise ValueError(reason)
    
    # Recursively check child nodes
    for child in ast.iter_child_nodes(node):
        result = _check_ast_safety(child, lint_mode)
        if result is not None and not result.is_valid:
            return result
    
    if lint_mode:
        return SafeEvalLintResult(is_valid=True)
    return None


def safe_eval_arithmetic(expr: str, *, lint: bool = False) -> Union[float, None, SafeEvalLintResult]:
    """
    Safely evaluate an arithmetic expression.
    
    This function parses and evaluates arithmetic expressions containing only:
    - Numeric literals (integers and floats)
    - Basic arithmetic operators: +, -, *, /, //, %, **
    - Parentheses for grouping
    
    No variables, function calls, or other Python constructs are allowed.
    
    Args:
        expr: String expression to evaluate (e.g., "2 + 3 * 4")
        lint: If True, return SafeEvalLintResult with diagnostics instead of evaluating
        
    Returns:
        If lint=False:
            - float: Result of evaluation
            - None: If expression is invalid or evaluation fails
        If lint=True:
            - SafeEvalLintResult: Diagnostic information about the expression
            
    Examples:
        >>> safe_eval_arithmetic("2 + 3")
        5.0
        >>> safe_eval_arithmetic("(10 - 2) * 3")
        24.0
        >>> safe_eval_arithmetic("x + 1")  # Variable not allowed
        None
        >>> result = safe_eval_arithmetic("x + 1", lint=True)
        >>> result.is_valid
        False
        >>> result.reason
        "Variable names not allowed: 'x'"
    """
    if not isinstance(expr, str):
        if lint:
            return SafeEvalLintResult(
                is_valid=False,
                reason=f"Expression must be string, got {type(expr).__name__}",
                first_illegal_node_type=None
            )
        return None
    
    expr = expr.strip()
    if not expr:
        if lint:
            return SafeEvalLintResult(
                is_valid=False,
                reason="Empty expression",
                first_illegal_node_type=None
            )
        return None
    
    try:
        # Parse expression into AST
        tree = ast.parse(expr, mode='eval')
        
        # Check AST safety
        lint_result = _check_ast_safety(tree, lint_mode=lint)
        
        if lint:
            # In lint mode, return the diagnostic result
            return lint_result
        
        # If not in lint mode and we got here, AST is safe
        # Compile and evaluate
        code = compile(tree, '<string>', 'eval')
        result = eval(code, {"__builtins__": {}})
        
        # Return as float for consistency
        return float(result)
        
    except SyntaxError as e:
        if lint:
            return SafeEvalLintResult(
                is_valid=False,
                reason=f"Syntax error: {str(e)}",
                first_illegal_node_type=None
            )
        return None
    except (ValueError, TypeError, ZeroDivisionError, OverflowError) as e:
        if lint:
            return SafeEvalLintResult(
                is_valid=False,
                reason=f"Evaluation error: {str(e)}",
                first_illegal_node_type=None
            )
        return None
    except Exception as e:
        # Catch-all for unexpected errors
        if lint:
            return SafeEvalLintResult(
                is_valid=False,
                reason=f"Unexpected error: {type(e).__name__}: {str(e)}",
                first_illegal_node_type=None
            )
        return None


def safe_eval_lint(expr: str) -> SafeEvalLintResult:
    """
    Lint an arithmetic expression without evaluating it.
    
    This is a convenience wrapper around safe_eval_arithmetic(expr, lint=True).
    
    Args:
        expr: String expression to lint
        
    Returns:
        SafeEvalLintResult: Diagnostic information about the expression
        
    Examples:
        >>> result = safe_eval_lint("2 + 3")
        >>> result.is_valid
        True
        >>> result = safe_eval_lint("import os")
        >>> result.is_valid
        False
        >>> result.reason
        'Syntax error: invalid syntax...'
    """
    result = safe_eval_arithmetic(expr, lint=True)
    assert isinstance(result, SafeEvalLintResult)
    return result
