"""
U2 Safe Evaluation Module - Phase III Safety Layer

Provides safe evaluation of expressions with lint mode for static analysis.
Enforces strict constraints on allowed operations and prevents unsafe code execution.
"""

import ast
from typing import Any, List, Optional, Set
from dataclasses import dataclass


# Allowed AST node types for safe evaluation
SAFE_NODE_TYPES = {
    ast.Module,
    ast.Expr,
    ast.Expression,
    ast.Constant,  # Literals (numbers, strings, etc.)
    ast.Num,  # For older Python versions
    ast.Str,  # For older Python versions
    ast.Load,
    ast.UnaryOp,
    ast.UAdd,
    ast.USub,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.Not,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Name,  # Variable references (must be in allowed_names)
    ast.Call,  # Function calls (handled specially - only safe builtins allowed)
}

# Dangerous node types that should never be allowed
DANGEROUS_NODE_TYPES = {
    ast.Import,
    ast.ImportFrom,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Lambda,
    ast.Attribute,  # Attribute access (can be dangerous)
    ast.Delete,
    ast.Assign,
    ast.AugAssign,
    ast.AnnAssign,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.If,
    ast.With,
    ast.AsyncWith,
    ast.Raise,
    ast.Try,
    ast.Assert,
    ast.Global,
    ast.Nonlocal,
    ast.Yield,
    ast.YieldFrom,
    ast.Await,
}

# Safe builtins that are allowed in function calls
SAFE_BUILTINS = {
    'abs', 'min', 'max', 'round', 'sum', 'len',
    'int', 'float', 'str', 'bool',
    'True', 'False', 'None',
}


@dataclass
class SafeEvalLintResult:
    """
    Result of linting an expression for safe evaluation.
    
    Attributes:
        is_safe: Whether the expression passes safety checks
        issues: List of safety issues found
        dangerous_nodes: Node types that were flagged as dangerous
        expression: Original expression that was linted
    """
    is_safe: bool
    issues: List[str]
    dangerous_nodes: List[str]
    expression: str
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "is_safe": self.is_safe,
            "issues": self.issues,
            "dangerous_nodes": self.dangerous_nodes,
            "expression": self.expression,
        }


def lint_expression(expression: str, allowed_names: Optional[Set[str]] = None) -> SafeEvalLintResult:
    """
    Lint an expression for safe evaluation without executing it.
    
    This performs static analysis on the AST to identify potentially unsafe
    operations before execution.
    
    Args:
        expression: String expression to lint
        allowed_names: Set of variable names that are allowed (None = allow all Name nodes)
        
    Returns:
        SafeEvalLintResult with safety status and any issues found
    """
    issues: List[str] = []
    dangerous_nodes: List[str] = []
    
    if not expression or not expression.strip():
        issues.append("Empty expression")
        return SafeEvalLintResult(
            is_safe=False,
            issues=issues,
            dangerous_nodes=dangerous_nodes,
            expression=expression,
        )
    
    try:
        tree = ast.parse(expression, mode='eval')
    except SyntaxError as e:
        # Try parsing as a statement to detect imports and other statement-level constructs
        try:
            stmt_tree = ast.parse(expression, mode='exec')
            # If it parses as a statement, check for dangerous statement-level nodes
            for node in ast.walk(stmt_tree):
                node_type = type(node)
                if node_type in DANGEROUS_NODE_TYPES:
                    dangerous_nodes.append(node_type.__name__)
                    issues.append(f"Dangerous operation: {node_type.__name__}")
            
            if not issues:
                issues.append(f"Statement not allowed in expression context: {e}")
        except:
            issues.append(f"Syntax error: {e}")
        
        return SafeEvalLintResult(
            is_safe=False,
            issues=issues,
            dangerous_nodes=dangerous_nodes,
            expression=expression,
        )
    
    # Walk the AST and check for unsafe nodes
    for node in ast.walk(tree):
        node_type = type(node)
        node_name = node_type.__name__
        
        # Check for dangerous nodes
        if node_type in DANGEROUS_NODE_TYPES:
            dangerous_nodes.append(node_name)
            issues.append(f"Dangerous operation: {node_name}")
        
        # Special handling for Call nodes - allow only safe builtins
        elif node_type == ast.Call:
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                if func_name not in SAFE_BUILTINS:
                    dangerous_nodes.append("Call")
                    issues.append(f"Dangerous function call: {func_name}")
                # Safe builtin calls are allowed
            else:
                # Complex calls (attributes, subscripts) are not allowed
                dangerous_nodes.append("Call")
                issues.append(f"Complex function call not allowed")
        
        # Check for disallowed safe nodes (excluding Call which we handle specially)
        elif node_type not in SAFE_NODE_TYPES and node_type != ast.Call:
            dangerous_nodes.append(node_name)
            issues.append(f"Disallowed operation: {node_name}")
        
        # Check variable names if restrictions are provided
        if isinstance(node, ast.Name) and allowed_names is not None:
            # Skip checking builtin names
            if node.id not in SAFE_BUILTINS and node.id not in allowed_names:
                issues.append(f"Disallowed variable name: {node.id}")
    
    is_safe = len(issues) == 0
    
    return SafeEvalLintResult(
        is_safe=is_safe,
        issues=issues,
        dangerous_nodes=dangerous_nodes,
        expression=expression,
    )


def safe_eval(
    expression: str,
    allowed_names: Optional[dict] = None,
    lint_only: bool = False,
) -> Any:
    """
    Safely evaluate an expression with restricted operations.
    
    This function performs static analysis before evaluation and only allows
    basic arithmetic and comparison operations.
    
    Args:
        expression: String expression to evaluate
        allowed_names: Dictionary of allowed variable names and their values
        lint_only: If True, only lint without evaluating (returns SafeEvalLintResult)
        
    Returns:
        Evaluated result, or SafeEvalLintResult if lint_only=True
        
    Raises:
        ValueError: If expression fails safety checks
        Exception: Any exception from evaluation (if not lint_only)
    """
    if allowed_names is None:
        allowed_names = {}
    
    # Lint the expression first
    lint_result = lint_expression(expression, set(allowed_names.keys()) if allowed_names else None)
    
    if lint_only:
        return lint_result
    
    if not lint_result.is_safe:
        raise ValueError(f"Unsafe expression: {', '.join(lint_result.issues)}")
    
    # Use restricted eval with only the allowed names
    try:
        # Create restricted globals/locals with only safe builtins
        safe_builtins = {
            'abs': abs,
            'min': min,
            'max': max,
            'round': round,
            'sum': sum,
            'len': len,
            'int': int,
            'float': float,
            'str': str,
            'bool': bool,
            'True': True,
            'False': False,
            'None': None,
        }
        
        # Merge with allowed names
        safe_globals = {**safe_builtins, **allowed_names}
        
        # Evaluate in restricted environment
        result = eval(expression, {"__builtins__": {}}, safe_globals)
        return result
        
    except Exception as e:
        raise Exception(f"Evaluation error: {e}") from e


def batch_lint_expressions(expressions: List[str], allowed_names: Optional[Set[str]] = None) -> List[SafeEvalLintResult]:
    """
    Lint multiple expressions for safe evaluation.
    
    Args:
        expressions: List of expressions to lint
        allowed_names: Set of allowed variable names
        
    Returns:
        List of SafeEvalLintResult, one per expression
    """
    return [lint_expression(expr, allowed_names) for expr in expressions]
