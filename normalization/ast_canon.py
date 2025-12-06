"""
AST-based Canonical Normalization for Propositional Logic.

This module provides a robust, structure-aware normalization pipeline that:
1. Parses statements into an Abstract Syntax Tree (AST)
2. Normalizes the AST structure (commutative sorting, idempotency, etc.)
3. Serializes the normalized AST to canonical bytes

The canonical identity is preserved:
    hash(s) = SHA256(DOMAIN_STMT || canonical_bytes(s))

This AST-based approach is more robust than string manipulation for:
- Complex nested formulas
- Future FOL extension (quantifiers, binders)
- Structural equivalence checking
- Alpha-renaming (when variables are introduced)

Usage:
    from normalization.ast_canon import parse_ast, normalize_ast, serialize_ast, canonical_bytes_ast

    ast = parse_ast("(p ∧ q) → r")
    normalized = normalize_ast(ast)
    payload = serialize_ast(normalized)
    # payload is ASCII bytes ready for hashing
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from functools import total_ordering
from typing import FrozenSet, List, Optional, Sequence, Tuple, Union


# ---------------------------------------------------------------------------
# Unicode → ASCII Symbol Mapping (canonical alphabet)
# ---------------------------------------------------------------------------

_SYMBOL_MAP = {
    # implications / equivalences
    "→": "->", "⇒": "->", "⟹": "->",
    "↔": "<->", "⇔": "<->",
    # conjunction / disjunction
    "∧": "/\\", "⋀": "/\\",
    "∨": "\\/", "⋁": "\\/",
    # negation
    "¬": "~", "￢": "~",
    # parentheses styles (normalize exotic to ASCII)
    "（": "(", "）": ")",
    "⟨": "(", "⟩": ")",
    # whitespace-like chars
    "\u00A0": " ", "\u2002": " ", "\u2003": " ", "\u2009": " ",
    "\u202F": " ", "\u3000": " ",
}


def _to_ascii(s: str) -> str:
    """Map Unicode logic symbols to ASCII alphabet."""
    for unicode_char, ascii_equiv in _SYMBOL_MAP.items():
        if unicode_char in s:
            s = s.replace(unicode_char, ascii_equiv)
    return s


# ---------------------------------------------------------------------------
# AST Node Types
# ---------------------------------------------------------------------------

class OpKind(Enum):
    """Operator kinds for propositional logic."""
    ATOM = auto()      # Atomic proposition (p, q, r, ...)
    NOT = auto()       # Negation (~)
    AND = auto()       # Conjunction (/\)
    OR = auto()        # Disjunction (\/)
    IMPLIES = auto()   # Implication (->)
    IFF = auto()       # Biconditional (<->)


@total_ordering
@dataclass(frozen=True, slots=True)
class Expr(ABC):
    """Base class for AST expressions."""

    @abstractmethod
    def to_canonical(self) -> str:
        """Serialize to canonical ASCII string."""
        ...

    @abstractmethod
    def atoms(self) -> FrozenSet[str]:
        """Return set of atomic propositions."""
        ...

    @abstractmethod
    def depth(self) -> int:
        """Return formula depth."""
        ...

    def __lt__(self, other: "Expr") -> bool:
        """Lexicographic ordering for deterministic sorting."""
        return self.to_canonical() < other.to_canonical()


@dataclass(frozen=True, slots=True)
class Atom(Expr):
    """Atomic proposition."""
    name: str

    def to_canonical(self) -> str:
        return self.name

    def atoms(self) -> FrozenSet[str]:
        return frozenset({self.name})

    def depth(self) -> int:
        return 0


@dataclass(frozen=True, slots=True)
class Not(Expr):
    """Negation."""
    operand: Expr

    def to_canonical(self) -> str:
        inner = self.operand.to_canonical()
        if isinstance(self.operand, (And, Or, Implies, Iff)):
            return f"~({inner})"
        return f"~{inner}"

    def atoms(self) -> FrozenSet[str]:
        return self.operand.atoms()

    def depth(self) -> int:
        return 1 + self.operand.depth()


@dataclass(frozen=True, slots=True)
class And(Expr):
    """Conjunction (n-ary, flattened, sorted)."""
    operands: Tuple[Expr, ...]

    def __post_init__(self):
        if len(self.operands) < 2:
            raise ValueError("And requires at least 2 operands")

    def to_canonical(self) -> str:
        parts = []
        for op in self.operands:
            s = op.to_canonical()
            if isinstance(op, (Or, Implies, Iff)):
                s = f"({s})"
            parts.append(s)
        return "/\\".join(parts)

    def atoms(self) -> FrozenSet[str]:
        result: FrozenSet[str] = frozenset()
        for op in self.operands:
            result = result | op.atoms()
        return result

    def depth(self) -> int:
        return 1 + max(op.depth() for op in self.operands)


@dataclass(frozen=True, slots=True)
class Or(Expr):
    """Disjunction (n-ary, flattened, sorted)."""
    operands: Tuple[Expr, ...]

    def __post_init__(self):
        if len(self.operands) < 2:
            raise ValueError("Or requires at least 2 operands")

    def to_canonical(self) -> str:
        parts = []
        for op in self.operands:
            s = op.to_canonical()
            if isinstance(op, (And, Implies, Iff)):
                s = f"({s})"
            parts.append(s)
        return "\\/".join(parts)

    def atoms(self) -> FrozenSet[str]:
        result: FrozenSet[str] = frozenset()
        for op in self.operands:
            result = result | op.atoms()
        return result

    def depth(self) -> int:
        return 1 + max(op.depth() for op in self.operands)


@dataclass(frozen=True, slots=True)
class Implies(Expr):
    """Implication (binary, right-associative)."""
    antecedent: Expr
    consequent: Expr

    def to_canonical(self) -> str:
        left = self.antecedent.to_canonical()
        right = self.consequent.to_canonical()
        # Wrap left if it contains operators
        if isinstance(self.antecedent, (And, Or, Implies, Iff)):
            left = f"({left})"
        return f"{left}->{right}"

    def atoms(self) -> FrozenSet[str]:
        return self.antecedent.atoms() | self.consequent.atoms()

    def depth(self) -> int:
        return 1 + max(self.antecedent.depth(), self.consequent.depth())


@dataclass(frozen=True, slots=True)
class Iff(Expr):
    """Biconditional (binary)."""
    left: Expr
    right: Expr

    def to_canonical(self) -> str:
        l = self.left.to_canonical()
        r = self.right.to_canonical()
        if isinstance(self.left, (And, Or, Implies, Iff)):
            l = f"({l})"
        if isinstance(self.right, (And, Or, Implies, Iff)):
            r = f"({r})"
        return f"{l}<->{r}"

    def atoms(self) -> FrozenSet[str]:
        return self.left.atoms() | self.right.atoms()

    def depth(self) -> int:
        return 1 + max(self.left.depth(), self.right.depth())


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class TokenKind(Enum):
    ATOM = auto()
    NOT = auto()
    AND = auto()
    OR = auto()
    IMPLIES = auto()
    IFF = auto()
    LPAREN = auto()
    RPAREN = auto()
    EOF = auto()


@dataclass(frozen=True, slots=True)
class Token:
    kind: TokenKind
    value: str
    pos: int


_TOKEN_PATTERNS = [
    (r"[A-Za-z][A-Za-z0-9_]*", TokenKind.ATOM),
    (r"~", TokenKind.NOT),
    (r"/\\", TokenKind.AND),
    (r"\\/", TokenKind.OR),
    (r"<->", TokenKind.IFF),
    (r"->", TokenKind.IMPLIES),
    (r"\(", TokenKind.LPAREN),
    (r"\)", TokenKind.RPAREN),
    (r"\s+", None),  # Skip whitespace
]

_COMPILED_PATTERNS = [(re.compile(p), k) for p, k in _TOKEN_PATTERNS]


def tokenize(s: str) -> List[Token]:
    """Tokenize an ASCII propositional formula."""
    tokens: List[Token] = []
    pos = 0
    while pos < len(s):
        matched = False
        for pattern, kind in _COMPILED_PATTERNS:
            m = pattern.match(s, pos)
            if m:
                if kind is not None:
                    tokens.append(Token(kind, m.group(), pos))
                pos = m.end()
                matched = True
                break
        if not matched:
            raise ValueError(f"Unexpected character at position {pos}: {s[pos]!r}")
    tokens.append(Token(TokenKind.EOF, "", pos))
    return tokens


# ---------------------------------------------------------------------------
# Recursive Descent Parser
# ---------------------------------------------------------------------------

class Parser:
    """Recursive descent parser for propositional logic."""

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def current(self) -> Token:
        return self.tokens[self.pos]

    def consume(self, kind: TokenKind) -> Token:
        tok = self.current()
        if tok.kind != kind:
            raise ValueError(f"Expected {kind}, got {tok.kind} at position {tok.pos}")
        self.pos += 1
        return tok

    def parse(self) -> Expr:
        expr = self.parse_iff()
        if self.current().kind != TokenKind.EOF:
            raise ValueError(f"Unexpected token at position {self.current().pos}")
        return expr

    def parse_iff(self) -> Expr:
        """Parse biconditional (lowest precedence)."""
        left = self.parse_implies()
        while self.current().kind == TokenKind.IFF:
            self.consume(TokenKind.IFF)
            right = self.parse_implies()
            left = Iff(left, right)
        return left

    def parse_implies(self) -> Expr:
        """Parse implication (right-associative)."""
        left = self.parse_or()
        if self.current().kind == TokenKind.IMPLIES:
            self.consume(TokenKind.IMPLIES)
            right = self.parse_implies()  # Right-associative
            return Implies(left, right)
        return left

    def parse_or(self) -> Expr:
        """Parse disjunction."""
        left = self.parse_and()
        operands = [left]
        while self.current().kind == TokenKind.OR:
            self.consume(TokenKind.OR)
            operands.append(self.parse_and())
        if len(operands) == 1:
            return operands[0]
        return Or(tuple(operands))

    def parse_and(self) -> Expr:
        """Parse conjunction."""
        left = self.parse_unary()
        operands = [left]
        while self.current().kind == TokenKind.AND:
            self.consume(TokenKind.AND)
            operands.append(self.parse_unary())
        if len(operands) == 1:
            return operands[0]
        return And(tuple(operands))

    def parse_unary(self) -> Expr:
        """Parse negation and atoms."""
        if self.current().kind == TokenKind.NOT:
            self.consume(TokenKind.NOT)
            operand = self.parse_unary()
            return Not(operand)
        return self.parse_primary()

    def parse_primary(self) -> Expr:
        """Parse atoms and parenthesized expressions."""
        tok = self.current()
        if tok.kind == TokenKind.ATOM:
            self.consume(TokenKind.ATOM)
            return Atom(tok.value)
        if tok.kind == TokenKind.LPAREN:
            self.consume(TokenKind.LPAREN)
            expr = self.parse_iff()
            self.consume(TokenKind.RPAREN)
            return expr
        raise ValueError(f"Unexpected token {tok.kind} at position {tok.pos}")


def parse_ast(s: str) -> Expr:
    """Parse a propositional formula into an AST."""
    ascii_s = _to_ascii(s)
    tokens = tokenize(ascii_s)
    parser = Parser(tokens)
    return parser.parse()


# ---------------------------------------------------------------------------
# AST Normalization
# ---------------------------------------------------------------------------

def normalize_ast(expr: Expr) -> Expr:
    """
    Normalize an AST to canonical form.

    Transformations:
    1. Flatten nested And/Or
    2. Sort And/Or operands lexicographically
    3. Remove duplicate operands (idempotency)
    4. Simplify single-operand And/Or to the operand
    5. Double negation elimination: ~~p → p
    """
    if isinstance(expr, Atom):
        return expr

    if isinstance(expr, Not):
        inner = normalize_ast(expr.operand)
        # Double negation elimination
        if isinstance(inner, Not):
            return inner.operand
        return Not(inner)

    if isinstance(expr, And):
        # Flatten and normalize operands
        flattened: List[Expr] = []
        for op in expr.operands:
            normalized_op = normalize_ast(op)
            if isinstance(normalized_op, And):
                flattened.extend(normalized_op.operands)
            else:
                flattened.append(normalized_op)
        # Sort and deduplicate
        unique = tuple(sorted(set(flattened)))
        if len(unique) == 1:
            return unique[0]
        return And(unique)

    if isinstance(expr, Or):
        # Flatten and normalize operands
        flattened: List[Expr] = []
        for op in expr.operands:
            normalized_op = normalize_ast(op)
            if isinstance(normalized_op, Or):
                flattened.extend(normalized_op.operands)
            else:
                flattened.append(normalized_op)
        # Sort and deduplicate
        unique = tuple(sorted(set(flattened)))
        if len(unique) == 1:
            return unique[0]
        return Or(unique)

    if isinstance(expr, Implies):
        left = normalize_ast(expr.antecedent)
        right = normalize_ast(expr.consequent)
        return Implies(left, right)

    if isinstance(expr, Iff):
        left = normalize_ast(expr.left)
        right = normalize_ast(expr.right)
        # Canonicalize order for biconditional
        if right < left:
            left, right = right, left
        return Iff(left, right)

    raise TypeError(f"Unknown expression type: {type(expr)}")


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def serialize_ast(expr: Expr) -> bytes:
    """Serialize normalized AST to canonical ASCII bytes."""
    canonical_str = expr.to_canonical()
    return canonical_str.encode("ascii")


def canonical_bytes_ast(s: str) -> bytes:
    """
    Full pipeline: parse → normalize → serialize.

    This is the AST-based equivalent of canonical_bytes from normalization.canon.
    """
    ast = parse_ast(s)
    normalized = normalize_ast(ast)
    return serialize_ast(normalized)


# ---------------------------------------------------------------------------
# Compatibility helpers
# ---------------------------------------------------------------------------

def normalize_via_ast(s: str) -> str:
    """Normalize using AST pipeline, return canonical string."""
    ast = parse_ast(s)
    normalized = normalize_ast(ast)
    return normalized.to_canonical()


def are_equivalent_ast(a: str, b: str) -> bool:
    """Check if two formulas are equivalent via AST normalization."""
    return normalize_via_ast(a) == normalize_via_ast(b)


def get_atomic_propositions_ast(s: str) -> FrozenSet[str]:
    """Extract atomic propositions via AST parsing."""
    ast = parse_ast(s)
    return ast.atoms()


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    # AST types
    "Expr",
    "Atom",
    "Not",
    "And",
    "Or",
    "Implies",
    "Iff",
    "OpKind",
    # Parsing
    "parse_ast",
    "tokenize",
    "Token",
    "TokenKind",
    # Normalization
    "normalize_ast",
    # Serialization
    "serialize_ast",
    "canonical_bytes_ast",
    # Compatibility
    "normalize_via_ast",
    "are_equivalent_ast",
    "get_atomic_propositions_ast",
]

