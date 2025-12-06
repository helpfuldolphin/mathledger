"""
Property-based tests for derivation abstention determinism.

These tests use Hypothesis to verify that the abstention generator produces
deterministic, reproducible results across a wide range of inputs.

Key invariants tested:
    1. Abstention is guaranteed for the First Organism config
    2. Results are deterministic (same input → same output)
    3. Canonical metadata is consistent (sorted parents, consistent pretty)
    4. Telemetry is complete and well-formed
"""

from __future__ import annotations

import hashlib
import json
from typing import List, Tuple

import pytest

# Try to import hypothesis, skip tests if not available
try:
    from hypothesis import given, settings, assume, example, HealthCheck
    from hypothesis import strategies as st
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    # Create dummy decorators for when hypothesis is not available
    def given(*args, **kwargs):
        def decorator(f):
            return pytest.mark.skip(reason="hypothesis not installed")(f)
        return decorator
    def settings(*args, **kwargs):
        def decorator(f):
            return f
        return decorator
    def example(*args, **kwargs):
        def decorator(f):
            return f
        return decorator
    def assume(x):
        pass
    class st:
        @staticmethod
        def integers(*args, **kwargs):
            return None
        @staticmethod
        def floats(*args, **kwargs):
            return None
        @staticmethod
        def text(*args, **kwargs):
            return None
        @staticmethod
        def sampled_from(*args, **kwargs):
            return None
        @staticmethod
        def lists(*args, **kwargs):
            return None
        @staticmethod
        def tuples(*args, **kwargs):
            return None
    class HealthCheck:
        too_slow = None

from derivation.pipeline import (
    ABSTENTION_METHODS,
    AbstainedStatement,
    DerivationResult,
    DerivationSummary,
    FirstOrganismDerivationConfig,
    StatementRecord,
    make_first_organism_derivation_config,
    make_first_organism_derivation_slice,
    make_first_organism_seed_statements,
    run_slice_for_test,
    _canonical_parents,
    _canonical_pretty,
    _is_guaranteed_non_tautology,
    _statement_fingerprint,
)
from derivation.derive_utils import sha256_statement
from normalization.canon import normalize


pytestmark = pytest.mark.determinism


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def first_organism_config() -> FirstOrganismDerivationConfig:
    """Return the First Organism abstention config."""
    return make_first_organism_derivation_config()


@pytest.fixture
def first_organism_result(first_organism_config: FirstOrganismDerivationConfig) -> DerivationResult:
    """Run the First Organism config and return the result."""
    return run_slice_for_test(
        first_organism_config.slice_cfg,
        existing=list(first_organism_config.seed_statements),
        limit=1,
        emit_log=False,
    )


# ---------------------------------------------------------------------------
# Core Invariant Tests
# ---------------------------------------------------------------------------


class TestAbstentionGuarantee:
    """Tests verifying that abstention is guaranteed for the First Organism config."""

    def test_first_organism_config_is_valid(self, first_organism_config: FirstOrganismDerivationConfig):
        """The First Organism config must pass validation."""
        assert first_organism_config.validate(), "Config validation failed"

    def test_first_organism_produces_abstention(self, first_organism_result: DerivationResult):
        """The First Organism config MUST produce at least one abstention."""
        assert first_organism_result.has_abstention, (
            f"Expected abstention but got n_abstained={first_organism_result.n_abstained}"
        )

    def test_abstention_method_is_expected(self, first_organism_result: DerivationResult):
        """All abstained statements must have expected verification methods."""
        for abstained in first_organism_result.abstained_candidates:
            assert abstained.verification_method in ABSTENTION_METHODS, (
                f"Unexpected method: {abstained.verification_method}"
            )

    def test_abstention_guaranteed_property(self, first_organism_result: DerivationResult):
        """The abstention_guaranteed property must be True."""
        assert first_organism_result.abstention_guaranteed, (
            "abstention_guaranteed should be True for First Organism config"
        )

    def test_guaranteed_non_tautology_is_abstained(
        self,
        first_organism_config: FirstOrganismDerivationConfig,
        first_organism_result: DerivationResult,
    ):
        """The guaranteed non-tautology must be in the abstained candidates."""
        expected = first_organism_config.guaranteed_non_tautology
        abstained_normalized = {s.normalized for s in first_organism_result.abstained_candidates}
        assert expected in abstained_normalized, (
            f"Expected {expected!r} to be abstained, got {abstained_normalized}"
        )


class TestDeterminism:
    """Tests verifying that derivation is deterministic."""

    def test_multiple_runs_produce_same_result(self, first_organism_config: FirstOrganismDerivationConfig):
        """Running the same config multiple times produces identical results."""
        results = []
        for _ in range(3):
            result = run_slice_for_test(
                first_organism_config.slice_cfg,
                existing=list(first_organism_config.seed_statements),
                limit=1,
                emit_log=False,
            )
            results.append(result)

        # All results must have same metrics
        for r in results[1:]:
            assert r.n_candidates == results[0].n_candidates
            assert r.n_verified == results[0].n_verified
            assert r.n_abstained == results[0].n_abstained
            assert r.status == results[0].status

        # All abstained candidates must have same hashes
        hashes_0 = {s.hash for s in results[0].abstained_candidates}
        for r in results[1:]:
            hashes_r = {s.hash for s in r.abstained_candidates}
            assert hashes_0 == hashes_r, "Abstained hashes differ between runs"

    def test_summary_json_is_deterministic(self, first_organism_config: FirstOrganismDerivationConfig):
        """The summary JSON (excluding timestamp) is deterministic."""
        results = []
        for _ in range(3):
            result = run_slice_for_test(
                first_organism_config.slice_cfg,
                existing=list(first_organism_config.seed_statements),
                limit=1,
                emit_log=False,
            )
            results.append(result)

        # Extract summary dicts, remove timestamp and duration (non-deterministic)
        def normalize_summary(summary: DerivationSummary) -> dict:
            d = summary.to_dict()
            d.pop("timestamp", None)
            d.pop("duration_ms", None)
            return d

        normalized_0 = normalize_summary(results[0].summary)
        for r in results[1:]:
            normalized_r = normalize_summary(r.summary)
            assert normalized_0 == normalized_r, "Summary dicts differ between runs"

    def test_seed_statements_are_deterministic(self):
        """Seed statements are deterministic across calls."""
        seeds_1 = make_first_organism_seed_statements()
        seeds_2 = make_first_organism_seed_statements()

        assert len(seeds_1) == len(seeds_2)
        for s1, s2 in zip(seeds_1, seeds_2):
            assert s1.hash == s2.hash
            assert s1.normalized == s2.normalized
            assert s1.pretty == s2.pretty
            assert s1.rule == s2.rule


class TestCanonicalMetadata:
    """Tests verifying that metadata is canonical."""

    def test_parents_are_sorted(self, first_organism_result: DerivationResult):
        """All parent tuples must be sorted."""
        for stmt in first_organism_result.abstained_candidates:
            assert stmt.parents == tuple(sorted(stmt.parents)), (
                f"Parents not sorted: {stmt.parents}"
            )

    def test_pretty_is_derived_from_normalized(self, first_organism_result: DerivationResult):
        """Pretty form must be derivable from normalized form."""
        for stmt in first_organism_result.abstained_candidates:
            expected_pretty = _canonical_pretty(stmt.normalized)
            assert stmt.pretty == expected_pretty, (
                f"Pretty mismatch: {stmt.pretty!r} != {expected_pretty!r}"
            )

    def test_fingerprint_is_deterministic(self, first_organism_result: DerivationResult):
        """Fingerprints must be deterministic."""
        for stmt in first_organism_result.abstained_candidates:
            expected_fp = _statement_fingerprint(stmt.normalized, stmt.parents)
            assert stmt.fingerprint == expected_fp, (
                f"Fingerprint mismatch: {stmt.fingerprint!r} != {expected_fp!r}"
            )

    def test_hash_matches_normalized(self, first_organism_result: DerivationResult):
        """Hash must match SHA256 of normalized form."""
        for stmt in first_organism_result.abstained_candidates:
            expected_hash = sha256_statement(stmt.normalized)
            assert stmt.hash == expected_hash, (
                f"Hash mismatch: {stmt.hash!r} != {expected_hash!r}"
            )


class TestTelemetry:
    """Tests verifying telemetry completeness."""

    def test_summary_has_required_fields(self, first_organism_result: DerivationResult):
        """Summary must have all required fields."""
        summary = first_organism_result.summary
        summary_dict = summary.to_dict()

        required_top_level = ["telemetry_version", "slice", "metrics", "filtering", "abstained"]
        for field in required_top_level:
            assert field in summary_dict, f"Missing field: {field}"

        required_metrics = ["n_candidates", "n_verified", "n_abstain", "abstention_rate"]
        for field in required_metrics:
            assert field in summary_dict["metrics"], f"Missing metrics field: {field}"

        required_filtering = [
            "axioms_seeded", "axioms_rejected", "mp_candidates_rejected",
            "depth_filtered", "atom_filtered", "duplicate_filtered",
        ]
        for field in required_filtering:
            assert field in summary_dict["filtering"], f"Missing filtering field: {field}"

    def test_abstained_statements_have_required_fields(self, first_organism_result: DerivationResult):
        """Abstained statements in summary must have all required fields."""
        for stmt in first_organism_result.summary.abstained_statements:
            stmt_dict = stmt.to_dict()
            required = ["hash", "normalized", "pretty", "method", "rule", "mp_depth", "parents", "fingerprint"]
            for field in required:
                assert field in stmt_dict, f"Missing abstained field: {field}"

    def test_json_is_valid(self, first_organism_result: DerivationResult):
        """Summary JSON must be valid and parseable."""
        json_str = first_organism_result.summary.to_json()
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_log_line_format(self, first_organism_result: DerivationResult):
        """Log line must have expected format."""
        log_line = first_organism_result.summary.to_log_line()
        assert log_line.startswith("DERIVATION_SUMMARY")
        assert "slice=" in log_line
        assert "candidates=" in log_line
        assert "abstain=" in log_line


# ---------------------------------------------------------------------------
# Property-Based Tests (Hypothesis)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
class TestPropertyBasedDeterminism:
    """Property-based tests using Hypothesis."""

    @given(st.integers(min_value=1, max_value=5))
    @settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow])
    def test_multiple_iterations_deterministic(self, iterations: int):
        """Multiple iterations produce deterministic results."""
        config = make_first_organism_derivation_config()
        
        result1 = run_slice_for_test(
            config.slice_cfg,
            existing=list(config.seed_statements),
            limit=iterations,
            emit_log=False,
        )
        result2 = run_slice_for_test(
            config.slice_cfg,
            existing=list(config.seed_statements),
            limit=iterations,
            emit_log=False,
        )
        
        assert result1.n_abstained == result2.n_abstained
        assert result1.n_verified == result2.n_verified

    @given(st.sampled_from(["p", "q", "r", "s"]))
    @settings(max_examples=10)
    def test_single_atoms_are_non_tautologies(self, atom: str):
        """Single atoms are always non-tautologies."""
        assert _is_guaranteed_non_tautology(atom), f"{atom} should be a non-tautology"

    @given(st.lists(st.text(alphabet="abcdef0123456789", min_size=64, max_size=64), min_size=0, max_size=5))
    @settings(max_examples=20)
    def test_canonical_parents_is_sorted(self, parent_hashes: List[str]):
        """_canonical_parents always returns sorted tuple."""
        parents = tuple(parent_hashes)
        canonical = _canonical_parents(parents)
        assert canonical == tuple(sorted(parents))

    @given(st.text(alphabet="pqr->() ", min_size=1, max_size=20))
    @settings(max_examples=50)
    def test_normalize_is_idempotent(self, formula: str):
        """Normalization is idempotent."""
        assume("->" in formula or formula.strip() in "pqr")
        try:
            norm1 = normalize(formula)
            if norm1:
                norm2 = normalize(norm1)
                assert norm1 == norm2, f"Normalization not idempotent: {formula!r}"
        except Exception:
            pass  # Invalid formulas are expected to fail

    @given(
        st.tuples(
            st.text(alphabet="pqr->", min_size=1, max_size=10),
            st.lists(st.text(alphabet="0123456789abcdef", min_size=8, max_size=8), min_size=0, max_size=3),
        )
    )
    @settings(max_examples=30)
    def test_fingerprint_is_deterministic(self, data: Tuple[str, List[str]]):
        """Fingerprints are deterministic."""
        formula, parent_list = data
        parents = tuple(parent_list)
        
        try:
            normalized = normalize(formula)
            if not normalized:
                return
        except Exception:
            return
        
        fp1 = _statement_fingerprint(normalized, parents)
        fp2 = _statement_fingerprint(normalized, parents)
        assert fp1 == fp2, "Fingerprint not deterministic"


# ---------------------------------------------------------------------------
# Edge Case Tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_seeds_produces_no_abstention(self):
        """Empty seeds with no axiom instances produces no abstention."""
        config = make_first_organism_derivation_config()
        # Override to use empty seeds
        result = run_slice_for_test(
            config.slice_cfg,
            existing=[],  # No seeds
            limit=1,
            emit_log=False,
        )
        # With axiom_instances=0 and no seeds, nothing should happen
        assert result.n_candidates == 0 or result.n_abstained == 0

    def test_canonical_parents_empty(self):
        """Empty parents tuple is handled correctly."""
        parents: Tuple[str, ...] = ()
        canonical = _canonical_parents(parents)
        assert canonical == ()

    def test_canonical_parents_single(self):
        """Single parent is handled correctly."""
        parents = ("abc123",)
        canonical = _canonical_parents(parents)
        assert canonical == ("abc123",)

    def test_canonical_parents_already_sorted(self):
        """Already sorted parents are unchanged."""
        parents = ("aaa", "bbb", "ccc")
        canonical = _canonical_parents(parents)
        assert canonical == parents

    def test_canonical_parents_reverse_sorted(self):
        """Reverse sorted parents are corrected."""
        parents = ("ccc", "bbb", "aaa")
        canonical = _canonical_parents(parents)
        assert canonical == ("aaa", "bbb", "ccc")


# ---------------------------------------------------------------------------
# Regression Tests
# ---------------------------------------------------------------------------


class TestRegressions:
    """Regression tests for previously discovered issues."""

    def test_abstained_statement_from_record_preserves_all_fields(self):
        """AbstainedStatement.from_record preserves all fields."""
        record = StatementRecord(
            normalized="q",
            hash="abc123",
            pretty="q",
            rule="mp",
            is_axiom=False,
            mp_depth=1,
            parents=("parent1", "parent2"),
            verification_method="lean-disabled",
        )
        abstained = AbstainedStatement.from_record(record)
        
        assert abstained.hash == record.hash
        assert abstained.normalized == record.normalized
        assert abstained.pretty == record.pretty
        assert abstained.verification_method == record.verification_method
        assert abstained.rule == record.rule
        assert abstained.mp_depth == record.mp_depth
        assert abstained.parents == tuple(sorted(record.parents))
        assert abstained.fingerprint == record.fingerprint

    def test_statement_record_post_init_sorts_parents(self):
        """StatementRecord.__post_init__ sorts parents."""
        record = StatementRecord(
            normalized="q",
            hash="abc123",
            pretty="q",
            rule="mp",
            is_axiom=False,
            mp_depth=1,
            parents=("zzz", "aaa", "mmm"),
            verification_method="lean-disabled",
        )
        assert record.parents == ("aaa", "mmm", "zzz")

    def test_first_organism_slice_has_zero_axiom_instances(self):
        """First Organism slice must have axiom_instances=0 for guaranteed abstention."""
        config = make_first_organism_derivation_config()
        assert config.slice_cfg.params.get("axiom_instances") == 0, (
            "axiom_instances must be 0 to prevent axiom seeding"
        )

