"""
Tests for normalization/ast_canon.py AST-based canonicalization.
"""

import pytest

from normalization.ast_canon import (
    Atom,
    Not,
    And,
    Or,
    Implies,
    Iff,
    parse_ast,
    normalize_ast,
    serialize_ast,
    canonical_bytes_ast,
    normalize_via_ast,
    are_equivalent_ast,
    get_atomic_propositions_ast,
    tokenize,
    TokenKind,
)


class TestTokenizer:
    """Test the tokenizer."""

    def test_tokenize_simple(self):
        tokens = tokenize("p -> q")
        kinds = [t.kind for t in tokens]
        assert kinds == [TokenKind.ATOM, TokenKind.IMPLIES, TokenKind.ATOM, TokenKind.EOF]

    def test_tokenize_conjunction(self):
        tokens = tokenize("p /\\ q")
        kinds = [t.kind for t in tokens]
        assert kinds == [TokenKind.ATOM, TokenKind.AND, TokenKind.ATOM, TokenKind.EOF]

    def test_tokenize_disjunction(self):
        tokens = tokenize("p \\/ q")
        kinds = [t.kind for t in tokens]
        assert kinds == [TokenKind.ATOM, TokenKind.OR, TokenKind.ATOM, TokenKind.EOF]

    def test_tokenize_negation(self):
        tokens = tokenize("~p")
        kinds = [t.kind for t in tokens]
        assert kinds == [TokenKind.NOT, TokenKind.ATOM, TokenKind.EOF]

    def test_tokenize_biconditional(self):
        tokens = tokenize("p <-> q")
        kinds = [t.kind for t in tokens]
        assert kinds == [TokenKind.ATOM, TokenKind.IFF, TokenKind.ATOM, TokenKind.EOF]

    def test_tokenize_parentheses(self):
        tokens = tokenize("(p)")
        kinds = [t.kind for t in tokens]
        assert kinds == [TokenKind.LPAREN, TokenKind.ATOM, TokenKind.RPAREN, TokenKind.EOF]

    def test_tokenize_complex(self):
        tokens = tokenize("(p /\\ q) -> (r \\/ s)")
        kinds = [t.kind for t in tokens]
        expected = [
            TokenKind.LPAREN, TokenKind.ATOM, TokenKind.AND, TokenKind.ATOM, TokenKind.RPAREN,
            TokenKind.IMPLIES,
            TokenKind.LPAREN, TokenKind.ATOM, TokenKind.OR, TokenKind.ATOM, TokenKind.RPAREN,
            TokenKind.EOF,
        ]
        assert kinds == expected


class TestParser:
    """Test the parser."""

    def test_parse_atom(self):
        ast = parse_ast("p")
        assert isinstance(ast, Atom)
        assert ast.name == "p"

    def test_parse_negation(self):
        ast = parse_ast("~p")
        assert isinstance(ast, Not)
        assert isinstance(ast.operand, Atom)

    def test_parse_conjunction(self):
        ast = parse_ast("p /\\ q")
        assert isinstance(ast, And)
        assert len(ast.operands) == 2

    def test_parse_disjunction(self):
        ast = parse_ast("p \\/ q")
        assert isinstance(ast, Or)
        assert len(ast.operands) == 2

    def test_parse_implication(self):
        ast = parse_ast("p -> q")
        assert isinstance(ast, Implies)
        assert isinstance(ast.antecedent, Atom)
        assert isinstance(ast.consequent, Atom)

    def test_parse_biconditional(self):
        ast = parse_ast("p <-> q")
        assert isinstance(ast, Iff)

    def test_parse_nested(self):
        ast = parse_ast("(p /\\ q) -> r")
        assert isinstance(ast, Implies)
        assert isinstance(ast.antecedent, And)
        assert isinstance(ast.consequent, Atom)

    def test_parse_unicode(self):
        ast = parse_ast("p ∧ q")
        assert isinstance(ast, And)

    def test_parse_implication_right_associative(self):
        ast = parse_ast("p -> q -> r")
        assert isinstance(ast, Implies)
        assert isinstance(ast.antecedent, Atom)
        assert isinstance(ast.consequent, Implies)


class TestNormalization:
    """Test AST normalization."""

    def test_normalize_atom(self):
        ast = parse_ast("p")
        normalized = normalize_ast(ast)
        assert isinstance(normalized, Atom)
        assert normalized.name == "p"

    def test_normalize_double_negation(self):
        ast = parse_ast("~~p")
        normalized = normalize_ast(ast)
        assert isinstance(normalized, Atom)
        assert normalized.name == "p"

    def test_normalize_conjunction_sort(self):
        ast = parse_ast("q /\\ p")
        normalized = normalize_ast(ast)
        assert isinstance(normalized, And)
        # Should be sorted: p before q
        assert normalized.operands[0].to_canonical() < normalized.operands[1].to_canonical()

    def test_normalize_conjunction_dedupe(self):
        ast = parse_ast("p /\\ p")
        normalized = normalize_ast(ast)
        # Should collapse to single p
        assert isinstance(normalized, Atom)
        assert normalized.name == "p"

    def test_normalize_conjunction_flatten(self):
        ast = parse_ast("(p /\\ q) /\\ r")
        normalized = normalize_ast(ast)
        assert isinstance(normalized, And)
        assert len(normalized.operands) == 3

    def test_normalize_disjunction_sort(self):
        ast = parse_ast("q \\/ p")
        normalized = normalize_ast(ast)
        assert isinstance(normalized, Or)
        assert normalized.operands[0].to_canonical() < normalized.operands[1].to_canonical()

    def test_normalize_disjunction_dedupe(self):
        ast = parse_ast("p \\/ p")
        normalized = normalize_ast(ast)
        assert isinstance(normalized, Atom)

    def test_normalize_disjunction_flatten(self):
        ast = parse_ast("(p \\/ q) \\/ r")
        normalized = normalize_ast(ast)
        assert isinstance(normalized, Or)
        assert len(normalized.operands) == 3

    def test_normalize_biconditional_order(self):
        ast1 = parse_ast("q <-> p")
        ast2 = parse_ast("p <-> q")
        norm1 = normalize_ast(ast1)
        norm2 = normalize_ast(ast2)
        # Should produce same canonical form
        assert norm1.to_canonical() == norm2.to_canonical()


class TestSerialization:
    """Test AST serialization."""

    def test_serialize_atom(self):
        ast = Atom("p")
        assert serialize_ast(ast) == b"p"

    def test_serialize_negation(self):
        ast = Not(Atom("p"))
        assert serialize_ast(ast) == b"~p"

    def test_serialize_conjunction(self):
        ast = And((Atom("p"), Atom("q")))
        assert serialize_ast(ast) == b"p/\\q"

    def test_serialize_disjunction(self):
        ast = Or((Atom("p"), Atom("q")))
        assert serialize_ast(ast) == b"p\\/q"

    def test_serialize_implication(self):
        ast = Implies(Atom("p"), Atom("q"))
        assert serialize_ast(ast) == b"p->q"

    def test_serialize_biconditional(self):
        ast = Iff(Atom("p"), Atom("q"))
        assert serialize_ast(ast) == b"p<->q"

    def test_serialize_complex(self):
        ast = Implies(And((Atom("p"), Atom("q"))), Atom("r"))
        assert serialize_ast(ast) == b"(p/\\q)->r"


class TestCanonicalBytes:
    """Test the full canonical_bytes_ast pipeline."""

    def test_canonical_bytes_simple(self):
        payload = canonical_bytes_ast("p -> q")
        assert payload == b"p->q"

    def test_canonical_bytes_unicode(self):
        payload = canonical_bytes_ast("p ∧ q")
        assert payload == b"p/\\q"

    def test_canonical_bytes_sorted(self):
        payload = canonical_bytes_ast("q /\\ p")
        assert payload == b"p/\\q"

    def test_canonical_bytes_deduped(self):
        payload = canonical_bytes_ast("p /\\ p")
        assert payload == b"p"

    def test_canonical_bytes_ascii_only(self):
        payload = canonical_bytes_ast("p ∧ q → r")
        assert all(b < 128 for b in payload)


class TestEquivalence:
    """Test equivalence checking via AST."""

    def test_equivalent_commutative(self):
        assert are_equivalent_ast("p /\\ q", "q /\\ p")
        assert are_equivalent_ast("p \\/ q", "q \\/ p")

    def test_equivalent_idempotent(self):
        assert are_equivalent_ast("p /\\ p", "p")
        assert are_equivalent_ast("p \\/ p", "p")

    def test_equivalent_unicode_ascii(self):
        assert are_equivalent_ast("p ∧ q", "p /\\ q")
        assert are_equivalent_ast("p → q", "p -> q")

    def test_not_equivalent_different(self):
        assert not are_equivalent_ast("p", "q")
        assert not are_equivalent_ast("p -> q", "q -> p")


class TestAtomExtraction:
    """Test atomic proposition extraction."""

    def test_atoms_simple(self):
        atoms = get_atomic_propositions_ast("p")
        assert atoms == frozenset({"p"})

    def test_atoms_conjunction(self):
        atoms = get_atomic_propositions_ast("p /\\ q /\\ r")
        assert atoms == frozenset({"p", "q", "r"})

    def test_atoms_complex(self):
        atoms = get_atomic_propositions_ast("(p -> q) /\\ (r \\/ s)")
        assert atoms == frozenset({"p", "q", "r", "s"})

    def test_atoms_with_negation(self):
        atoms = get_atomic_propositions_ast("~p /\\ q")
        assert atoms == frozenset({"p", "q"})


class TestDepth:
    """Test formula depth calculation."""

    def test_depth_atom(self):
        ast = parse_ast("p")
        assert ast.depth() == 0

    def test_depth_negation(self):
        ast = parse_ast("~p")
        assert ast.depth() == 1

    def test_depth_binary(self):
        ast = parse_ast("p /\\ q")
        assert ast.depth() == 1

    def test_depth_nested(self):
        ast = parse_ast("(p /\\ q) -> r")
        assert ast.depth() == 2

    def test_depth_deep(self):
        ast = parse_ast("((p /\\ q) -> r) -> s")
        assert ast.depth() == 3

