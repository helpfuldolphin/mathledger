"""
PHASE-II — NOT USED IN PHASE I

Success Metrics for U2 Uplift Experiments
=========================================

This module provides success metric functions for different experiment slices.
Each metric function determines whether an experiment trial was successful
based on the input item and its computed result.

**Determinism Notes:**
    - All metric functions are pure (no side effects, deterministic output).
    - Metrics do not use any random operations.
    - Same inputs always produce the same boolean result.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional


# Type alias for metric functions
MetricFunction = Callable[[str, Any], bool]


def metric_arithmetic_simple(item: str, result: Any) -> bool:
    """Evaluate success for arithmetic expressions.

    Success is determined by whether the Python eval of the item
    matches the expected result. This metric is used for simple
    arithmetic expression experiments.

    Args:
        item: A string representing an arithmetic expression (e.g., "2 + 3").
        result: The expected result of evaluating the expression.

    Returns:
        True if eval(item) equals result, False otherwise.

    Raises:
        No exceptions are raised; evaluation failures return False.

    Example:
        >>> metric_arithmetic_simple("2 + 3", 5)
        True
        >>> metric_arithmetic_simple("2 + 3", 6)
        False
        >>> metric_arithmetic_simple("invalid", 0)
        False

    **Determinism Notes:**
        - Pure function with no side effects.
        - Uses ast.literal_eval for safe evaluation (no code execution).
    """
    import ast
    try:
        # Use ast.literal_eval for safe evaluation of literals only
        # This prevents code injection while allowing arithmetic constants
        evaluated = ast.literal_eval(item)
        return evaluated == result
    except (ValueError, SyntaxError):
        # ast.literal_eval can't handle arithmetic expressions like "1+1"
        # Fall back to manual parsing for simple arithmetic
        return _safe_eval_arithmetic(item) == result
    except Exception:
        return False


def _safe_eval_arithmetic(expr: str) -> Any:
    """Safely evaluate simple arithmetic expressions.

    This function evaluates basic arithmetic expressions without using eval(),
    supporting only +, -, *, / operations on integers and floats.

    Args:
        expr: A string arithmetic expression (e.g., "1+1", "2*3").

    Returns:
        The numeric result, or None if evaluation fails.

    **Determinism Notes:**
        - Pure function with no side effects.
        - No code execution, only number parsing and arithmetic.
    """
    import ast
    import operator

    # Supported operators (safe subset)
    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.USub: operator.neg,
    }

    def _eval_node(node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):  # Python 3.8+
            return node.value
        elif isinstance(node, ast.Num):  # Python 3.7 compatibility
            return node.n
        elif isinstance(node, ast.BinOp):
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            op = ops.get(type(node.op))
            if op is None:
                return None
            return op(left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = _eval_node(node.operand)
            op = ops.get(type(node.op))
            if op is None:
                return None
            return op(operand)
        else:
            return None

    try:
        tree = ast.parse(expr, mode='eval')
        return _eval_node(tree.body)
    except Exception:
        return None


def metric_algebra_expansion(item: str, result: Any) -> bool:
    """Evaluate success for algebra expansion experiments.

    This is a mock metric that determines success based on whether
    the result string is longer than the input item. In a real scenario,
    this would involve actual algebraic expansion verification.

    Args:
        item: A string representing an algebraic expression.
        result: The result of attempting to expand the expression.

    Returns:
        True if len(str(result)) > len(item), False otherwise.

    Example:
        >>> metric_algebra_expansion("x", "Expanded(x)")
        True
        >>> metric_algebra_expansion("very_long_expression", "short")
        False

    **Determinism Notes:**
        - Pure function with no side effects.
        - Uses only string length comparison.
    """
    return len(str(result)) > len(item)


# Registry of metric functions by slice name
METRIC_DISPATCHER: Dict[str, MetricFunction] = {
    "arithmetic_simple": metric_arithmetic_simple,
    "algebra_expansion": metric_algebra_expansion,
}


def get_metric_function(slice_name: str) -> Optional[MetricFunction]:
    """Retrieve the metric function for a given slice name.

    Args:
        slice_name: The name of the experiment slice.

    Returns:
        The metric function for the slice, or None if not found.

    Raises:
        No exceptions; returns None for unknown slices.

    Example:
        >>> func = get_metric_function("arithmetic_simple")
        >>> func("1 + 1", 2)
        True
        >>> get_metric_function("unknown_slice") is None
        True
    """
    return METRIC_DISPATCHER.get(slice_name)


def register_metric(slice_name: str, metric_func: MetricFunction) -> None:
    """Register a new metric function for a slice.

    This allows extending the metric dispatcher with custom metrics
    for new experiment slices.

    Args:
        slice_name: The name of the experiment slice.
        metric_func: The metric function to register.

    Raises:
        ValueError: If slice_name is empty or metric_func is not callable.

    Example:
        >>> def custom_metric(item, result):
        ...     return True
        >>> register_metric("custom_slice", custom_metric)
        >>> get_metric_function("custom_slice") is not None
        True
    """
    if not slice_name:
        raise ValueError("slice_name cannot be empty")
    if not callable(metric_func):
        raise ValueError("metric_func must be callable")

    METRIC_DISPATCHER[slice_name] = metric_func
