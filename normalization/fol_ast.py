"""
FOL formula AST parsing for FOL_FIN_EQ_v1.

This module provides dataclasses for FOL formula AST nodes and functions
for parsing, validation, and hashing of formulas.

NORMATIVE INVARIANTS:
- Const.value is ALWAYS a key into domain_spec.constants (NOT element literal)
- Free variables trigger ValidationError (fail-closed)
- AST hashing uses DOMAIN_FOL_AST domain separation tag
- Quantifier variable names use 'variable' field (not 'var')

JSON Schema (fixtures):
- Forall: {"type": "forall", "variable": str, "body": AST}
- Exists: {"type": "exists", "variable": str, "body": AST}
- Not: {"type": "not", "inner": AST}
- And: {"type": "and", "left": AST, "right": AST}
- Or: {"type": "or", "left": AST, "right": AST}
- Equals: {"type": "equals", "left": AST, "right": AST}
- Apply: {"type": "apply", "function": str, "args": [AST, ...]}
- Var: {"type": "var", "name": str}
- Const: {"type": "const", "value": str}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from governance.registry_hash import canonicalize_json
from substrate.crypto.hashing import DOMAIN_FOL_AST, sha256_hex


# =============================================================================
# AST Node Dataclasses (frozen for immutability and hashing)
# =============================================================================


@dataclass(frozen=True)
class Var:
    """Variable reference node."""

    name: str


@dataclass(frozen=True)
class Const:
    """Constant reference node.

    NORMATIVE: value is a KEY into domain_spec.constants, NOT an element literal.
    Resolution happens via domain_spec.resolve_constant(value).
    """

    value: str


@dataclass(frozen=True)
class Apply:
    """Function application node."""

    function: str
    args: tuple["FolAst", ...]


@dataclass(frozen=True)
class Equals:
    """Equality node (left = right)."""

    left: "FolAst"
    right: "FolAst"


@dataclass(frozen=True)
class Not:
    """Negation node."""

    inner: "FolAst"


@dataclass(frozen=True)
class And:
    """Conjunction node (left ∧ right)."""

    left: "FolAst"
    right: "FolAst"


@dataclass(frozen=True)
class Or:
    """Disjunction node (left ∨ right)."""

    left: "FolAst"
    right: "FolAst"


@dataclass(frozen=True)
class Forall:
    """Universal quantifier node (∀variable. body)."""

    variable: str
    body: "FolAst"


@dataclass(frozen=True)
class Exists:
    """Existential quantifier node (∃variable. body)."""

    variable: str
    body: "FolAst"


# Type alias for any AST node
FolAst = Union[Var, Const, Apply, Equals, Not, And, Or, Forall, Exists]


# =============================================================================
# Parsing
# =============================================================================


def parse_fol_formula(data: dict) -> FolAst:
    """Parse a FOL formula from JSON dict representation.

    FAIL-CLOSED: Raises ValueError on unknown node types.

    Args:
        data: JSON-like dict representing the formula AST

    Returns:
        Parsed FolAst node

    Raises:
        ValueError: If node type is unknown or structure is invalid
    """
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict, got {type(data).__name__}")

    node_type = data.get("type")
    if node_type is None:
        raise ValueError("Missing 'type' field in AST node")

    if node_type == "var":
        name = data.get("name")
        if name is None:
            raise ValueError("Var node missing 'name' field")
        return Var(name=name)

    elif node_type == "const":
        value = data.get("value")
        if value is None:
            raise ValueError("Const node missing 'value' field")
        return Const(value=value)

    elif node_type == "apply":
        function = data.get("function")
        if function is None:
            raise ValueError("Apply node missing 'function' field")
        args_data = data.get("args")
        if args_data is None:
            raise ValueError("Apply node missing 'args' field")
        if not isinstance(args_data, list):
            raise ValueError("Apply node 'args' must be a list")
        args = tuple(parse_fol_formula(arg) for arg in args_data)
        return Apply(function=function, args=args)

    elif node_type == "equals":
        left_data = data.get("left")
        right_data = data.get("right")
        if left_data is None or right_data is None:
            raise ValueError("Equals node missing 'left' or 'right' field")
        return Equals(left=parse_fol_formula(left_data), right=parse_fol_formula(right_data))

    elif node_type == "not":
        inner_data = data.get("inner")
        if inner_data is None:
            raise ValueError("Not node missing 'inner' field")
        return Not(inner=parse_fol_formula(inner_data))

    elif node_type == "and":
        left_data = data.get("left")
        right_data = data.get("right")
        if left_data is None or right_data is None:
            raise ValueError("And node missing 'left' or 'right' field")
        return And(left=parse_fol_formula(left_data), right=parse_fol_formula(right_data))

    elif node_type == "or":
        left_data = data.get("left")
        right_data = data.get("right")
        if left_data is None or right_data is None:
            raise ValueError("Or node missing 'left' or 'right' field")
        return Or(left=parse_fol_formula(left_data), right=parse_fol_formula(right_data))

    elif node_type == "forall":
        variable = data.get("variable")
        body_data = data.get("body")
        if variable is None:
            raise ValueError("Forall node missing 'variable' field")
        if body_data is None:
            raise ValueError("Forall node missing 'body' field")
        return Forall(variable=variable, body=parse_fol_formula(body_data))

    elif node_type == "exists":
        variable = data.get("variable")
        body_data = data.get("body")
        if variable is None:
            raise ValueError("Exists node missing 'variable' field")
        if body_data is None:
            raise ValueError("Exists node missing 'body' field")
        return Exists(variable=variable, body=parse_fol_formula(body_data))

    else:
        raise ValueError(f"Unknown AST node type: '{node_type}'")


# =============================================================================
# Quantifier Analysis
# =============================================================================


def extract_quantifier_report(formula: FolAst) -> list[tuple[str, str]]:
    """Extract quantifier report in binding order.

    Returns list of (quantifier_type, variable_name) tuples in the order
    quantifiers appear when traversing AST from root to leaves.

    Args:
        formula: Parsed FOL formula

    Returns:
        List of ("forall", var) or ("exists", var) tuples in binding order
    """
    result: list[tuple[str, str]] = []
    _collect_quantifiers(formula, result)
    return result


def _collect_quantifiers(node: FolAst, result: list[tuple[str, str]]) -> None:
    """Helper to collect quantifiers recursively."""
    if isinstance(node, Forall):
        result.append(("forall", node.variable))
        _collect_quantifiers(node.body, result)
    elif isinstance(node, Exists):
        result.append(("exists", node.variable))
        _collect_quantifiers(node.body, result)
    elif isinstance(node, Not):
        _collect_quantifiers(node.inner, result)
    elif isinstance(node, (And, Or, Equals)):
        _collect_quantifiers(node.left, result)
        _collect_quantifiers(node.right, result)
    elif isinstance(node, Apply):
        for arg in node.args:
            _collect_quantifiers(arg, result)
    # Var and Const have no children


def compute_quantifier_depth(formula: FolAst) -> int:
    """Compute maximum quantifier nesting depth.

    depth(∀x.φ) = 1 + depth(φ)
    depth(∃x.φ) = 1 + depth(φ)
    depth(φ ∧ ψ) = max(depth(φ), depth(ψ))
    etc.

    Args:
        formula: Parsed FOL formula

    Returns:
        Maximum quantifier nesting depth (0 if no quantifiers)
    """
    if isinstance(node := formula, (Forall, Exists)):
        return 1 + compute_quantifier_depth(node.body)
    elif isinstance(node, Not):
        return compute_quantifier_depth(node.inner)
    elif isinstance(node, (And, Or, Equals)):
        return max(compute_quantifier_depth(node.left), compute_quantifier_depth(node.right))
    elif isinstance(node, Apply):
        if not node.args:
            return 0
        return max(compute_quantifier_depth(arg) for arg in node.args)
    else:
        # Var, Const
        return 0


# =============================================================================
# Free Variable Detection
# =============================================================================


def detect_free_variables(formula: FolAst) -> set[str]:
    """Detect free variables in a formula.

    A variable is free if it appears in the formula but is not bound
    by any enclosing quantifier.

    Args:
        formula: Parsed FOL formula

    Returns:
        Set of free variable names
    """
    return _find_free_vars(formula, bound_vars=frozenset())


def _find_free_vars(node: FolAst, bound_vars: frozenset[str]) -> set[str]:
    """Helper to find free variables recursively."""
    if isinstance(node, Var):
        if node.name in bound_vars:
            return set()
        else:
            return {node.name}
    elif isinstance(node, Const):
        return set()
    elif isinstance(node, Apply):
        result: set[str] = set()
        for arg in node.args:
            result |= _find_free_vars(arg, bound_vars)
        return result
    elif isinstance(node, Equals):
        return _find_free_vars(node.left, bound_vars) | _find_free_vars(node.right, bound_vars)
    elif isinstance(node, Not):
        return _find_free_vars(node.inner, bound_vars)
    elif isinstance(node, (And, Or)):
        return _find_free_vars(node.left, bound_vars) | _find_free_vars(node.right, bound_vars)
    elif isinstance(node, Forall):
        return _find_free_vars(node.body, bound_vars | {node.variable})
    elif isinstance(node, Exists):
        return _find_free_vars(node.body, bound_vars | {node.variable})
    else:
        # Exhaustive match - should not reach here
        return set()


def validate_closed_formula(formula: FolAst) -> None:
    """Validate that formula has no free variables.

    FAIL-CLOSED: Raises ValueError if free variables are detected.

    Args:
        formula: Parsed FOL formula

    Raises:
        ValueError: If formula contains free variables
    """
    free_vars = detect_free_variables(formula)
    if free_vars:
        vars_list = ", ".join(sorted(free_vars))
        raise ValueError(f"Formula contains free variables: {vars_list}")


# =============================================================================
# AST Canonicalization and Hashing
# =============================================================================


def canonicalize_ast(formula: FolAst) -> str:
    """Convert AST to canonical JSON string for hashing.

    Uses governance.registry_hash.canonicalize_json for deterministic output.

    Args:
        formula: Parsed FOL formula

    Returns:
        Canonical JSON string representation
    """
    ast_dict = _ast_to_dict(formula)
    return canonicalize_json(ast_dict)


def _ast_to_dict(node: FolAst) -> dict:
    """Convert AST node to dict for JSON serialization."""
    if isinstance(node, Var):
        return {"type": "var", "name": node.name}
    elif isinstance(node, Const):
        return {"type": "const", "value": node.value}
    elif isinstance(node, Apply):
        return {
            "type": "apply",
            "function": node.function,
            "args": [_ast_to_dict(arg) for arg in node.args],
        }
    elif isinstance(node, Equals):
        return {
            "type": "equals",
            "left": _ast_to_dict(node.left),
            "right": _ast_to_dict(node.right),
        }
    elif isinstance(node, Not):
        return {"type": "not", "inner": _ast_to_dict(node.inner)}
    elif isinstance(node, And):
        return {
            "type": "and",
            "left": _ast_to_dict(node.left),
            "right": _ast_to_dict(node.right),
        }
    elif isinstance(node, Or):
        return {
            "type": "or",
            "left": _ast_to_dict(node.left),
            "right": _ast_to_dict(node.right),
        }
    elif isinstance(node, Forall):
        return {
            "type": "forall",
            "variable": node.variable,
            "body": _ast_to_dict(node.body),
        }
    elif isinstance(node, Exists):
        return {
            "type": "exists",
            "variable": node.variable,
            "body": _ast_to_dict(node.body),
        }
    else:
        raise ValueError(f"Unknown AST node type: {type(node).__name__}")


def compute_ast_hash(formula: FolAst) -> str:
    """Compute canonical hash of formula AST.

    Uses:
    - canonicalize_json from governance.registry_hash
    - sha256_hex with DOMAIN_FOL_AST domain separation

    Args:
        formula: Parsed FOL formula

    Returns:
        64-character hex hash string
    """
    canonical = canonicalize_ast(formula)
    payload = canonical.encode("utf-8")
    return sha256_hex(payload, domain=DOMAIN_FOL_AST)
