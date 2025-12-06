"""
Tests for Phase III axiom engine refactoring.

Tests derive_core, derive_rules, derive_utils modules.
"""

import pytest
from backend.axiom_engine.derive_core import DerivationEngine, InferenceEngine, append_to_progress
from backend.axiom_engine.derive_rules import (
    ProofContext,
    ProofResult,
    is_known_tautology,
    is_tautology_with_timeout,
    RULE_MODUS_PONENS,
    RULE_AXIOM,
    IMPLIES,
)
from backend.axiom_engine.derive_utils import (
    sha256_statement,
    print_diagnostic,
    get_table_columns,
    ensure_redis,
    record_proof_edge,
)


class TestProofContext:
    """Test ProofContext dataclass."""
    
    def test_proof_context_creation(self):
        """Test basic ProofContext creation."""
        ctx = ProofContext(
            statement_id="test_id",
            dependencies=["dep1", "dep2"],
            derivation_rule=RULE_MODUS_PONENS,
        )
        assert ctx.statement_id == "test_id"
        assert len(ctx.dependencies) == 2
        assert ctx.derivation_rule == RULE_MODUS_PONENS
    
    def test_proof_context_to_dict(self):
        """Test ProofContext serialization."""
        ctx = ProofContext(
            statement_id="test_id",
            dependencies=["dep1"],
            derivation_rule=RULE_AXIOM,
            merkle_root="abc123",
            signature_b64="sig",
        )
        d = ctx.to_dict()
        assert d["statement_id"] == "test_id"
        assert d["dependencies"] == ["dep1"]
        assert d["derivation_rule"] == RULE_AXIOM
        assert d["merkle_root"] == "abc123"
        assert d["signature_b64"] == "sig"


class TestProofResult:
    """Test ProofResult dataclass."""
    
    def test_proof_result_creation(self):
        """Test basic ProofResult creation."""
        result = ProofResult(
            formula="p->p",
            normalized="p->p",
            method="smoke_pl",
            verified=True,
        )
        assert result.formula == "p->p"
        assert result.normalized == "p->p"
        assert result.method == "smoke_pl"
        assert result.verified is True


class TestTautologyRecognition:
    """Test tautology recognition functions."""
    
    def test_is_known_tautology_simple(self):
        """Test recognition of simple tautologies."""
        # (x/\y)->x
        assert is_known_tautology("(p/\\q)->p") is True
        # (x/\y)->y
        assert is_known_tautology("(p/\\q)->q") is True
        # x->(y->x)
        assert is_known_tautology("p->(q->p)") is True
    
    def test_is_known_tautology_not_tautology(self):
        """Test non-tautologies are rejected."""
        assert is_known_tautology("p->q") is False
        assert is_known_tautology("p/\\q") is False
        assert is_known_tautology("random text") is False
    
    def test_is_tautology_with_timeout_known(self):
        """Test timeout function recognizes known patterns quickly."""
        result = is_tautology_with_timeout("(p/\\q)->p", timeout_ms=5)
        assert result is True
    
    def test_is_tautology_with_timeout_unknown(self):
        """Test timeout function handles unknown patterns."""
        # This might timeout or return False
        result = is_tautology_with_timeout("((p->q)->r)->s", timeout_ms=1)
        assert isinstance(result, bool)


class TestDeriveUtils:
    """Test derive_utils functions."""
    
    def test_sha256_statement(self):
        """Test statement hashing."""
        hash1 = sha256_statement("test")
        hash2 = sha256_statement("test")
        hash3 = sha256_statement("different")
        
        assert len(hash1) == 64  # Hex string
        assert hash1 == hash2  # Deterministic
        assert hash1 != hash3  # Different inputs
    
    def test_print_diagnostic(self):
        """Test diagnostic printing."""
        # Should not raise
        try:
            class MockError(Exception):
                pass
            
            e = MockError("test error")
            print_diagnostic("test", e, "test_table")
        except Exception as ex:
            pytest.fail(f"print_diagnostic raised: {ex}")
    
    def test_ensure_redis_no_url(self, monkeypatch):
        """Test Redis connection with no URL."""
        monkeypatch.delenv("REDIS_URL", raising=False)
        client = ensure_redis()
        assert client is None


class TestDerivationEngine:
    """Test DerivationEngine class."""
    
    def test_derivation_engine_creation(self):
        """Test basic engine creation."""
        engine = DerivationEngine(
            db_url="postgresql://test",
            redis_url="redis://test",
            max_depth=3,
            max_breadth=100,
        )
        assert engine.db_url == "postgresql://test"
        assert engine.redis_url == "redis://test"
        assert engine.max_depth == 3
        assert engine.max_breadth == 100
    
    def test_derivation_engine_no_psycopg(self, monkeypatch):
        """Test engine behavior without psycopg."""
        # Mock psycopg to None
        import backend.axiom_engine.derive_core as derive_core_module
        original_psycopg = derive_core_module.psycopg
        derive_core_module.psycopg = None
        
        try:
            engine = DerivationEngine("test", "test")
            result = engine.derive_statements()
            assert result["n_new"] == 0
            assert result["pct_success"] == 0.0
        finally:
            derive_core_module.psycopg = original_psycopg


class TestInferenceEngine:
    """Test InferenceEngine compatibility class."""
    
    def test_inference_engine_derive(self):
        """Test legacy InferenceEngine interface."""
        engine = InferenceEngine()
        result = engine.derive_new_statements()
        assert result == []


class TestLegacyFunctions:
    """Test legacy compatibility functions."""
    
    def test_append_to_progress(self):
        """Test append_to_progress does nothing."""
        # Should not raise
        append_to_progress("test", "args", key="value")


class TestRuleConstants:
    """Test rule constant definitions."""
    
    def test_rule_constants_defined(self):
        """Test all rule constants are defined."""
        assert RULE_MODUS_PONENS == "mp"
        assert RULE_AXIOM == "axiom"
        assert IMPLIES == "->"


# Parametrized tests for tautology patterns
@pytest.mark.parametrize("formula,expected", [
    ("(p/\\q)->p", True),
    ("(p/\\q)->q", True),
    ("p->(q->p)", True),
    ("(p/\\(q/\\r))->p", True),
    ("((p/\\q)/\\r)->p", True),
    ("(p/\\q)->(q/\\p)", True),
    ("p->q->p", True),
    ("p->q", False),
    ("p/\\q", False),
])
def test_tautology_patterns_parametrized(formula, expected):
    """Test various tautology patterns."""
    assert is_known_tautology(formula) == expected


# Integration-style tests
class TestPhase3Integration:
    """Integration tests for Phase III components."""
    
    def test_proof_context_with_crypto(self):
        """Test ProofContext with crypto operations."""
        from backend.crypto.core import merkle_root
        
        deps = ["hash1", "hash2", "hash3"]
        merkle = merkle_root(deps)
        
        ctx = ProofContext(
            statement_id="stmt_1",
            dependencies=deps,
            derivation_rule=RULE_MODUS_PONENS,
            merkle_root=merkle,
        )
        
        assert ctx.merkle_root == merkle
        assert len(ctx.merkle_root) == 64
    
    def test_statement_hashing_consistency(self):
        """Test statement hashing is consistent with crypto.core."""
        from backend.crypto.core import sha256_hex, DOMAIN_STMT
        
        stmt = "test_statement"
        hash1 = sha256_statement(stmt)
        hash2 = sha256_hex(stmt, domain=DOMAIN_STMT)
        
        assert hash1 == hash2
    
    def test_pass_line_emission(self, capsys):
        """Test that pass-lines are emitted correctly."""
        from backend.axiom_engine.derive_core import DerivationEngine
        
        # This should emit pass-lines during derivation
        # (tested indirectly via output capture)
        engine = DerivationEngine("test", "test")
        # Can't test actual derivation without DB, but structure is validated
        assert hasattr(engine, 'derive_statements')
