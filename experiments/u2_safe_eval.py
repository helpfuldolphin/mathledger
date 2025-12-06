# PHASE II — NOT USED IN PHASE I
#
# Safe arithmetic AST evaluator for Phase II U2 experiments.
# This module provides a secure alternative to eval() for arithmetic expressions.
#
# Absolute Safeguards:
# - Only allows int/float literals, unary +/- and binary +, -, *, /
# - No names, attributes, calls, lambdas, comprehensions, etc.
# - Division by zero returns None (matches existing behavior)
# - Rejects any malformed or unsafe expressions

import ast
from typing import Optional, Union

# Supported types for safe evaluation
NumericType = Union[int, float]


def _eval_ast_node(node: ast.AST) -> Optional[NumericType]:
    """
    Recursively evaluate a safe subset of Python AST nodes.

    Args:
        node: An AST node representing part of an arithmetic expression.

    Returns:
        The numeric result, or None if evaluation fails (e.g., division by zero).

    Raises:
        ValueError: If the node type is not in the allowed safe subset.
    """
    if isinstance(node, ast.Expression):
        return _eval_ast_node(node.body)

    if isinstance(node, ast.Constant):
        # Only allow numeric constants
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")

    if isinstance(node, ast.UnaryOp):
        operand = _eval_ast_node(node.operand)
        if operand is None:
            return None
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise ValueError(f"Unsupported unary operator: {type(node.op)}")

    if isinstance(node, ast.BinOp):
        left = _eval_ast_node(node.left)
        right = _eval_ast_node(node.right)
        if left is None or right is None:
            return None

        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                return None  # Division by zero returns None
            return left / right
        raise ValueError(f"Unsupported binary operator: {type(node.op)}")

    raise ValueError(f"Unsupported AST node type: {type(node)}")


def safe_eval_arithmetic(expr: str) -> Optional[NumericType]:
    """
    Safely evaluate a simple arithmetic expression.

    This function parses and evaluates arithmetic expressions without using
    Python's eval(), making it safe against code injection attacks.

    Supported operations:
        - Integer and float literals (e.g., 42, 3.14, -17)
        - Unary operators: +, -
        - Binary operators: +, -, *, /

    Not supported (will raise ValueError):
        - Variable names or identifiers
        - Function calls
        - Attribute access
        - List/dict comprehensions
        - Lambda expressions
        - Any other Python constructs

    Args:
        expr: A string containing a simple arithmetic expression.

    Returns:
        The numeric result of the expression, or None if:
        - Division by zero occurs
        - The expression cannot be evaluated

    Raises:
        ValueError: If the expression contains unsupported constructs.
        SyntaxError: If the expression is not valid Python syntax.

    Examples:
        >>> safe_eval_arithmetic("1 + 2")
        3
        >>> safe_eval_arithmetic("10 / 2")
        5.0
        >>> safe_eval_arithmetic("1 / 0")
        None
        >>> safe_eval_arithmetic("3 * 4 + 5")
        17
    """
    try:
        tree = ast.parse(expr, mode='eval')
        return _eval_ast_node(tree)
    except SyntaxError:
        raise
    except ValueError:
        raise
    except (TypeError, AttributeError, RecursionError):
        # Handle unexpected but recoverable errors gracefully
        return None
