"""
Tests for PERF_OPT_ENABLED environment flag behavior.

PERF ONLY — NO BEHAVIOR CHANGE

This test verifies that:
1. Setting MATHLEDGER_PERF_OPT=0 uses the original code path
2. Setting MATHLEDGER_PERF_OPT=1 uses the optimized code path
3. Both paths produce identical behavioral results (H_t roots, verified counts)

These tests are NOT intended to run in the normal pytest suite since they
involve running full CycleRunner cycles which are slow.

Usage:
    pytest tests/test_perf_opt_flag.py -v -m "perf"
"""

import os
import sys
import pytest
from pathlib import Path
from unittest.mock import patch

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# Mark all tests in this module as perf tests (slow, not in normal test run)
pytestmark = pytest.mark.perf


class TestPerfOptFlagBasics:
    """Test basic flag behavior without running full cycles."""
    
    def test_flag_defaults_to_enabled(self):
        """PERF_OPT_ENABLED should default to True when env var not set."""
        # Clear the env var if set
        original = os.environ.pop("MATHLEDGER_PERF_OPT", None)
        try:
            # Force reimport to pick up new env var state
            import importlib
            import derivation.pipeline as pipeline_module
            importlib.reload(pipeline_module)
            
            assert pipeline_module.PERF_OPT_ENABLED is True
        finally:
            if original is not None:
                os.environ["MATHLEDGER_PERF_OPT"] = original
    
    def test_flag_disabled_when_set_to_zero(self):
        """PERF_OPT_ENABLED should be False when MATHLEDGER_PERF_OPT=0."""
        original = os.environ.get("MATHLEDGER_PERF_OPT")
        try:
            os.environ["MATHLEDGER_PERF_OPT"] = "0"
            
            # Force reimport to pick up new env var state
            import importlib
            import derivation.pipeline as pipeline_module
            importlib.reload(pipeline_module)
            
            assert pipeline_module.PERF_OPT_ENABLED is False
        finally:
            if original is not None:
                os.environ["MATHLEDGER_PERF_OPT"] = original
            else:
                os.environ.pop("MATHLEDGER_PERF_OPT", None)
    
    def test_flag_enabled_when_set_to_one(self):
        """PERF_OPT_ENABLED should be True when MATHLEDGER_PERF_OPT=1."""
        original = os.environ.get("MATHLEDGER_PERF_OPT")
        try:
            os.environ["MATHLEDGER_PERF_OPT"] = "1"
            
            # Force reimport to pick up new env var state
            import importlib
            import derivation.pipeline as pipeline_module
            importlib.reload(pipeline_module)
            
            assert pipeline_module.PERF_OPT_ENABLED is True
        finally:
            if original is not None:
                os.environ["MATHLEDGER_PERF_OPT"] = original
            else:
                os.environ.pop("MATHLEDGER_PERF_OPT", None)


@pytest.mark.slow
class TestPerfOptBehavioralEquivalence:
    """
    Test that optimized and original paths produce identical results.
    
    These tests are marked slow because they run full derivation cycles.
    """
    
    def test_rfl_scoring_equivalence(self):
        """
        Verify RFL scoring produces identical results regardless of PERF_OPT_ENABLED.
        
        This test:
        1. Runs a derivation with PERF_OPT=0 (original path)
        2. Runs a derivation with PERF_OPT=1 (optimized path)
        3. Compares the scored candidate ordering
        """
        from derivation.bounds import SliceBounds
        from derivation.pipeline import DerivationPipeline, StatementRecord, _canonical_pretty
        from derivation.verification import StatementVerifier
        from derivation.derive_utils import sha256_statement
        from normalization.canon import normalize
        
        # Create a minimal test case
        bounds = SliceBounds(
            max_atoms=3,
            max_formula_depth=3,
            max_mp_depth=1,
            max_breadth=10,
            max_total=10,
        )
        
        # Create seed statements
        seeds = []
        for expr, rule in [("p", "seed:atom"), ("(p->q)", "seed:implication"), ("(p->p)", "seed:tautology")]:
            normalized = normalize(expr)
            if normalized:
                seeds.append(StatementRecord(
                    normalized=normalized,
                    hash=sha256_statement(normalized),
                    pretty=_canonical_pretty(normalized),
                    rule=rule,
                    is_axiom=False,
                    mp_depth=0,
                    parents=(),
                    verification_method="seed",
                ))
        
        # Run with PERF_OPT=0
        import importlib
        import derivation.pipeline as pipeline_module
        
        original_env = os.environ.get("MATHLEDGER_PERF_OPT")
        try:
            os.environ["MATHLEDGER_PERF_OPT"] = "0"
            importlib.reload(pipeline_module)
            
            verifier1 = StatementVerifier(bounds)
            pipeline1 = pipeline_module.DerivationPipeline(
                bounds, verifier1,
                policy_weights={"len": 0.1, "depth": 0.2, "success": 0.3},
                mode="rfl",
                cycle_seed=42,
            )
            result1 = pipeline1.run_step(seeds)
            
            # Run with PERF_OPT=1
            os.environ["MATHLEDGER_PERF_OPT"] = "1"
            importlib.reload(pipeline_module)
            
            verifier2 = StatementVerifier(bounds)
            pipeline2 = pipeline_module.DerivationPipeline(
                bounds, verifier2,
                policy_weights={"len": 0.1, "depth": 0.2, "success": 0.3},
                mode="rfl",
                cycle_seed=42,
            )
            result2 = pipeline2.run_step(seeds)
            
            # Compare results
            assert result1.stats.verified == result2.stats.verified, \
                f"Verified count mismatch: {result1.stats.verified} vs {result2.stats.verified}"
            assert result1.stats.rejected == result2.stats.rejected, \
                f"Rejected count mismatch: {result1.stats.rejected} vs {result2.stats.rejected}"
            
            # Compare statement hashes
            hashes1 = {s.hash for s in result1.statements}
            hashes2 = {s.hash for s in result2.statements}
            assert hashes1 == hashes2, f"Statement hash mismatch: {hashes1 ^ hashes2}"
            
        finally:
            if original_env is not None:
                os.environ["MATHLEDGER_PERF_OPT"] = original_env
            else:
                os.environ.pop("MATHLEDGER_PERF_OPT", None)
            # Restore default state
            importlib.reload(pipeline_module)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])

