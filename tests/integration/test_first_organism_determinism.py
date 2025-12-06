#!/usr/bin/env python3
"""
First Organism Determinism Test Suite
======================================

This test suite verifies that the First Organism closed loop produces
byte-for-byte identical outputs when run with the same seed.

The test exercises the full path:
    UI Event → Curriculum Gate → Derivation → Lean Verify (abstention) →
    Dual-Attest seal H_t → RFL runner metabolism.

Cryptographic Invariants Verified:
    1. H_t (composite root) is identical across runs with same seed
    2. All intermediate artifacts (logs, IDs, timestamps) are identical
    3. RFC 8785 canonical JSON is used for all serialization
    4. No wall-clock time, random sources, or UUIDs leak into outputs

Usage:
    pytest tests/integration/test_first_organism_determinism.py -v
    pytest tests/integration/test_first_organism_determinism.py -k bitwise

CI Integration:
    This test is designed to run in CI without external dependencies.
    It uses in-memory fixtures and deterministic helpers only.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Tuple

import pytest

from backend.repro.first_organism_harness import (
    DeterministicDerivationResult,
    DeterministicGateVerdict,
    DeterministicRflStep,
    DeterministicSealResult,
    DeterministicUIEvent,
    FirstOrganismResult,
    deterministic_derivation_result,
    deterministic_gate_verdict,
    deterministic_rfl_step,
    deterministic_seal,
    deterministic_ui_event,
    rfc8785_canonicalize,
    rfc8785_hash,
    run_first_organism_deterministic,
    verify_determinism,
)


pytestmark = [pytest.mark.integration, pytest.mark.determinism]


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def deterministic_seed() -> int:
    """Fixed seed for reproducible tests."""
    return 42


@pytest.fixture
def alternate_seed() -> int:
    """Alternate seed to verify different outputs."""
    return 12345


@pytest.fixture
def sample_ui_payload() -> Dict[str, Any]:
    """Sample UI event payload."""
    return {
        "action": "toggle_abstain",
        "statement_hash": "a" * 64,
        "source": "test_harness",
    }


@pytest.fixture
def sample_metrics() -> Dict[str, Any]:
    """Sample curriculum metrics."""
    return {
        "metrics": {
            "coverage": {"ci_lower": 0.95, "sample_size": 24},
            "proofs": {"abstention_rate": 12.0, "attempt_mass": 3200},
            "curriculum": {
                "active_slice": {
                    "wallclock_minutes": 45.0,
                    "proof_velocity_cv": 0.05,
                }
            },
            "throughput": {
                "proofs_per_hour": 240.0,
                "coefficient_of_variation": 0.04,
                "window_minutes": 60,
            },
            "queue": {"backlog_fraction": 0.12},
        },
        "provenance": {"attestation_hash": "test-attestation-hash"},
    }


# ---------------------------------------------------------------------------
# Unit Tests: Individual Component Determinism
# ---------------------------------------------------------------------------


class TestDeterministicUIEvent:
    """Tests for deterministic_ui_event."""

    def test_same_seed_produces_identical_output(
        self, deterministic_seed: int, sample_ui_payload: Dict[str, Any]
    ):
        """Verify same seed produces identical UI event."""
        event1 = deterministic_ui_event(deterministic_seed, sample_ui_payload)
        event2 = deterministic_ui_event(deterministic_seed, sample_ui_payload)

        assert event1.event_id == event2.event_id
        assert event1.timestamp == event2.timestamp
        assert event1.canonical_json == event2.canonical_json
        assert event1.leaf_hash == event2.leaf_hash

    def test_different_seed_produces_different_output(
        self,
        deterministic_seed: int,
        alternate_seed: int,
        sample_ui_payload: Dict[str, Any],
    ):
        """Verify different seeds produce different UI events."""
        event1 = deterministic_ui_event(deterministic_seed, sample_ui_payload)
        event2 = deterministic_ui_event(alternate_seed, sample_ui_payload)

        assert event1.event_id != event2.event_id
        assert event1.timestamp != event2.timestamp
        # leaf_hash should be the same since payload is the same
        assert event1.leaf_hash == event2.leaf_hash

    def test_canonical_json_is_valid(
        self, deterministic_seed: int, sample_ui_payload: Dict[str, Any]
    ):
        """Verify canonical JSON is valid and parseable."""
        event = deterministic_ui_event(deterministic_seed, sample_ui_payload)

        # Should be valid JSON
        parsed = json.loads(event.canonical_json)
        assert "event_id" in parsed
        assert "timestamp" in parsed

        # Should be canonical (sorted keys, no whitespace)
        assert "," in event.canonical_json
        assert ": " not in event.canonical_json  # No space after colon

    def test_leaf_hash_is_deterministic(
        self, deterministic_seed: int, sample_ui_payload: Dict[str, Any]
    ):
        """Verify leaf hash is SHA-256 of canonical payload."""
        event = deterministic_ui_event(deterministic_seed, sample_ui_payload)

        # Manually compute expected hash
        payload_with_type = dict(sample_ui_payload)
        payload_with_type["event_type"] = "select_statement"
        canonical = rfc8785_canonicalize(payload_with_type)
        expected_hash = hashlib.sha256(canonical.encode("ascii")).hexdigest()

        assert event.leaf_hash == expected_hash


class TestDeterministicGateVerdict:
    """Tests for deterministic_gate_verdict."""

    def test_same_inputs_produce_identical_verdict(
        self, deterministic_seed: int, sample_metrics: Dict[str, Any]
    ):
        """Verify same inputs produce identical verdict."""
        statuses = [
            {"gate": "coverage", "passed": True, "message": "OK"},
            {"gate": "abstention", "passed": True, "message": "OK"},
        ]

        verdict1 = deterministic_gate_verdict(
            deterministic_seed,
            "test-slice",
            sample_metrics,
            statuses,
            advance=True,
            reason="all passed",
        )
        verdict2 = deterministic_gate_verdict(
            deterministic_seed,
            "test-slice",
            sample_metrics,
            statuses,
            advance=True,
            reason="all passed",
        )

        assert verdict1.audit_json == verdict2.audit_json
        assert verdict1.audit_hash == verdict2.audit_hash
        assert verdict1.timestamp == verdict2.timestamp


class TestDeterministicDerivationResult:
    """Tests for deterministic_derivation_result."""

    def test_same_inputs_produce_identical_result(self, deterministic_seed: int):
        """Verify same inputs produce identical derivation result."""
        hashes = ["hash1", "hash2", "hash3"]

        result1 = deterministic_derivation_result(
            deterministic_seed,
            "test-slice",
            status="abstain",
            n_candidates=10,
            n_abstained=3,
            abstained_hashes=hashes,
        )
        result2 = deterministic_derivation_result(
            deterministic_seed,
            "test-slice",
            status="abstain",
            n_candidates=10,
            n_abstained=3,
            abstained_hashes=hashes,
        )

        assert result1.run_id == result2.run_id
        assert result1.canonical_json == result2.canonical_json
        assert result1.result_hash == result2.result_hash

    def test_hash_order_is_normalized(self, deterministic_seed: int):
        """Verify abstained hashes are sorted for determinism."""
        hashes_unordered = ["z_hash", "a_hash", "m_hash"]
        hashes_ordered = ["a_hash", "m_hash", "z_hash"]

        result1 = deterministic_derivation_result(
            deterministic_seed,
            "test-slice",
            status="abstain",
            n_candidates=10,
            n_abstained=3,
            abstained_hashes=hashes_unordered,
        )
        result2 = deterministic_derivation_result(
            deterministic_seed,
            "test-slice",
            status="abstain",
            n_candidates=10,
            n_abstained=3,
            abstained_hashes=hashes_ordered,
        )

        # Should be identical because hashes are sorted internally
        assert result1.abstained_hashes == result2.abstained_hashes
        assert result1.result_hash == result2.result_hash


class TestDeterministicSeal:
    """Tests for deterministic_seal (dual-root attestation)."""

    def test_same_inputs_produce_identical_seal(self, deterministic_seed: int):
        """Verify same inputs produce identical seal."""
        ui_event = deterministic_ui_event(
            deterministic_seed, {"action": "test", "hash": "abc"}
        )
        derivation = deterministic_derivation_result(
            deterministic_seed,
            "test-slice",
            status="abstain",
            n_candidates=5,
            n_abstained=2,
            abstained_hashes=["h1", "h2"],
        )

        seal1 = deterministic_seal(deterministic_seed, derivation, [ui_event])
        seal2 = deterministic_seal(deterministic_seed, derivation, [ui_event])

        assert seal1.reasoning_root == seal2.reasoning_root
        assert seal1.ui_root == seal2.ui_root
        assert seal1.composite_root == seal2.composite_root
        assert seal1.attestation_json == seal2.attestation_json
        assert seal1.attestation_hash == seal2.attestation_hash

    def test_composite_root_formula(self, deterministic_seed: int):
        """Verify H_t = SHA256(R_t || U_t)."""
        ui_event = deterministic_ui_event(
            deterministic_seed, {"action": "test", "hash": "abc"}
        )
        derivation = deterministic_derivation_result(
            deterministic_seed,
            "test-slice",
            status="abstain",
            n_candidates=5,
            n_abstained=2,
            abstained_hashes=["h1", "h2"],
        )

        seal = deterministic_seal(deterministic_seed, derivation, [ui_event])

        # Manually verify H_t = SHA256(R_t || U_t)
        composite_data = f"{seal.reasoning_root}{seal.ui_root}".encode("ascii")
        expected_h_t = hashlib.sha256(composite_data).hexdigest()

        assert seal.composite_root == expected_h_t

    def test_empty_ui_events_produces_sentinel(self, deterministic_seed: int):
        """Verify empty UI events use sentinel hash."""
        derivation = deterministic_derivation_result(
            deterministic_seed,
            "test-slice",
            status="abstain",
            n_candidates=5,
            n_abstained=2,
            abstained_hashes=["h1", "h2"],
        )

        seal = deterministic_seal(deterministic_seed, derivation, [])

        expected_ui_root = hashlib.sha256(b"UI:EMPTY").hexdigest()
        assert seal.ui_root == expected_ui_root


class TestDeterministicRflStep:
    """Tests for deterministic_rfl_step."""

    def test_same_inputs_produce_identical_step(self, deterministic_seed: int):
        """Verify same inputs produce identical RFL step."""
        ui_event = deterministic_ui_event(
            deterministic_seed, {"action": "test", "hash": "abc"}
        )
        derivation = deterministic_derivation_result(
            deterministic_seed,
            "test-slice",
            status="abstain",
            n_candidates=10,
            n_abstained=3,
            abstained_hashes=["h1", "h2", "h3"],
        )
        seal = deterministic_seal(deterministic_seed, derivation, [ui_event])

        step1 = deterministic_rfl_step(
            deterministic_seed, seal, derivation, slice_name="test-slice"
        )
        step2 = deterministic_rfl_step(
            deterministic_seed, seal, derivation, slice_name="test-slice"
        )

        assert step1.step_id == step2.step_id
        assert step1.ledger_entry_json == step2.ledger_entry_json
        assert step1.ledger_entry_hash == step2.ledger_entry_hash
        assert step1.symbolic_descent == step2.symbolic_descent


# ---------------------------------------------------------------------------
# Integration Tests: Full Pipeline Determinism
# ---------------------------------------------------------------------------


class TestFirstOrganismFullPipeline:
    """Tests for run_first_organism_deterministic."""

    def test_full_pipeline_is_deterministic(self, deterministic_seed: int):
        """Verify full pipeline produces identical output across runs."""
        result1 = run_first_organism_deterministic(deterministic_seed)
        result2 = run_first_organism_deterministic(deterministic_seed)

        # All intermediate artifacts must be identical
        assert result1.ui_event.canonical_json == result2.ui_event.canonical_json
        assert result1.gate_verdict.audit_hash == result2.gate_verdict.audit_hash
        assert result1.derivation_result.result_hash == result2.derivation_result.result_hash
        assert result1.seal_result.attestation_hash == result2.seal_result.attestation_hash
        assert result1.rfl_step.ledger_entry_hash == result2.rfl_step.ledger_entry_hash

        # Final composite root must be identical
        assert result1.composite_root == result2.composite_root

        # Run hash must be identical
        assert result1.run_hash == result2.run_hash

    def test_different_seeds_produce_different_outputs(
        self, deterministic_seed: int, alternate_seed: int
    ):
        """Verify different seeds produce different outputs."""
        result1 = run_first_organism_deterministic(deterministic_seed)
        result2 = run_first_organism_deterministic(alternate_seed)

        assert result1.composite_root != result2.composite_root
        assert result1.run_hash != result2.run_hash

    def test_canonical_json_is_reproducible(self, deterministic_seed: int):
        """Verify canonical JSON serialization is reproducible."""
        result1 = run_first_organism_deterministic(deterministic_seed)
        result2 = run_first_organism_deterministic(deterministic_seed)

        json1 = result1.to_canonical_json()
        json2 = result2.to_canonical_json()

        assert json1 == json2

        # Verify it's valid JSON
        parsed = json.loads(json1)
        assert parsed["seed"] == deterministic_seed
        assert parsed["composite_root"] == result1.composite_root

    def test_hard_slice_reports_abstention_failure(self, deterministic_seed: int):
        """Verify the hard slice triggers the abstention path deterministically."""
        result1 = run_first_organism_deterministic(
            deterministic_seed, slice_name="first_organism_pl2_hard"
        )
        result2 = run_first_organism_deterministic(
            deterministic_seed, slice_name="first_organism_pl2_hard"
        )

        assert result1.run_hash == result2.run_hash
        assert result1.gate_verdict.advance is False
        assert result1.gate_verdict.reason.startswith("abstention gate")
        assert any(
            gate["gate"] == "abstention" and not gate["passed"]
            for gate in result1.gate_verdict.gate_statuses
        )

    def test_verify_determinism_helper(self, deterministic_seed: int):
        """Verify the verify_determinism helper function."""
        assert verify_determinism(deterministic_seed, runs=5)


# ---------------------------------------------------------------------------
# Bitwise Reproducibility Tests
# ---------------------------------------------------------------------------


class TestBitwiseReproducibility:
    """Tests for byte-for-byte reproducibility."""

    @pytest.mark.parametrize("seed", [0, 1, 42, 12345, 999999])
    def test_bitwise_identical_across_seeds(self, seed: int):
        """Verify bitwise reproducibility for various seeds."""
        result1 = run_first_organism_deterministic(seed)
        result2 = run_first_organism_deterministic(seed)

        # Serialize to bytes
        bytes1 = result1.to_canonical_json().encode("ascii")
        bytes2 = result2.to_canonical_json().encode("ascii")

        assert bytes1 == bytes2

    def test_bitwise_identical_with_custom_inputs(self, deterministic_seed: int):
        """Verify bitwise reproducibility with custom inputs."""
        custom_payload = {
            "action": "prove",
            "statement_hash": "custom_hash_" + "x" * 50,
            "metadata": {"key": "value"},
        }
        custom_hashes = [f"abstain_hash_{i}" for i in range(5)]

        result1 = run_first_organism_deterministic(
            deterministic_seed,
            ui_payload=custom_payload,
            n_candidates=20,
            n_abstained=5,
            abstained_hashes=custom_hashes,
        )
        result2 = run_first_organism_deterministic(
            deterministic_seed,
            ui_payload=custom_payload,
            n_candidates=20,
            n_abstained=5,
            abstained_hashes=custom_hashes,
        )

        bytes1 = result1.to_canonical_json().encode("ascii")
        bytes2 = result2.to_canonical_json().encode("ascii")

        assert bytes1 == bytes2

    def test_hash_stability_across_multiple_runs(self, deterministic_seed: int):
        """Verify hash stability across 10 runs."""
        hashes: List[str] = []

        for _ in range(10):
            result = run_first_organism_deterministic(deterministic_seed)
            hashes.append(result.run_hash)

        # All hashes must be identical
        assert len(set(hashes)) == 1


# ---------------------------------------------------------------------------
# RFC 8785 Canonical JSON Tests
# ---------------------------------------------------------------------------


class TestRFC8785Canonicalization:
    """Tests for RFC 8785 canonical JSON serialization."""

    def test_keys_are_sorted(self):
        """Verify keys are sorted lexicographically."""
        obj = {"z": 1, "a": 2, "m": 3}
        canonical = rfc8785_canonicalize(obj)
        parsed = json.loads(canonical)

        # Keys should be in sorted order
        assert list(parsed.keys()) == ["a", "m", "z"]

    def test_no_whitespace(self):
        """Verify no unnecessary whitespace."""
        obj = {"key": "value", "nested": {"inner": 123}}
        canonical = rfc8785_canonicalize(obj)

        # No spaces after colons or commas
        assert ": " not in canonical
        assert ", " not in canonical

    def test_ascii_only(self):
        """Verify output is ASCII-only."""
        obj = {"key": "value with émoji 🎉"}
        canonical = rfc8785_canonicalize(obj)

        # Should be ASCII-encodable
        canonical.encode("ascii")  # Should not raise

        # Non-ASCII should be escaped
        assert "\\u" in canonical

    def test_hash_is_deterministic(self):
        """Verify rfc8785_hash is deterministic."""
        obj = {"key": "value", "number": 42}

        hash1 = rfc8785_hash(obj)
        hash2 = rfc8785_hash(obj)

        assert hash1 == hash2

    def test_nested_objects_are_sorted(self):
        """Verify nested objects have sorted keys."""
        obj = {
            "outer": {"z": 1, "a": 2},
            "array": [{"b": 1, "a": 2}],
        }
        canonical = rfc8785_canonicalize(obj)

        # Should contain sorted keys
        assert '"a":2' in canonical
        assert canonical.index('"a":2') < canonical.index('"z":1')


# ---------------------------------------------------------------------------
# Regression Tests
# ---------------------------------------------------------------------------


class TestDeterminismRegression:
    """Regression tests for known determinism issues."""

    def test_no_datetime_now_in_output(self, deterministic_seed: int):
        """Verify no datetime.now() leaks into output."""
        result = run_first_organism_deterministic(deterministic_seed)
        canonical = result.to_canonical_json()

        # Should not contain current date patterns
        # (This is a heuristic check)
        import datetime
        today = datetime.date.today().isoformat()
        assert today not in canonical or "2025-01-01" in canonical

    def test_no_uuid_in_output(self, deterministic_seed: int):
        """Verify no uuid.uuid4() leaks into output."""
        result = run_first_organism_deterministic(deterministic_seed)
        canonical = result.to_canonical_json()

        # UUIDs have a specific format with dashes
        # Our deterministic IDs use prefixes like "ui-event-", "block-", etc.
        import re
        uuid_pattern = r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
        matches = re.findall(uuid_pattern, canonical, re.IGNORECASE)

        # Should have no UUID v4 matches
        assert len(matches) == 0

    def test_timestamps_are_content_derived(self, deterministic_seed: int):
        """Verify timestamps are derived from content, not wall-clock."""
        result1 = run_first_organism_deterministic(deterministic_seed)

        # Wait a bit (simulated by running again)
        result2 = run_first_organism_deterministic(deterministic_seed)

        # Timestamps should be identical
        assert result1.ui_event.timestamp == result2.ui_event.timestamp
        assert result1.seal_result.sealed_at == result2.seal_result.sealed_at


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_abstentions(self, deterministic_seed: int):
        """Verify handling of zero abstentions."""
        result = run_first_organism_deterministic(
            deterministic_seed,
            n_candidates=10,
            n_abstained=0,
            abstained_hashes=[],
        )

        assert result.derivation_result.status == "success"
        assert result.derivation_result.n_abstained == 0

    def test_all_abstentions(self, deterministic_seed: int):
        """Verify handling of 100% abstention rate."""
        result = run_first_organism_deterministic(
            deterministic_seed,
            n_candidates=10,
            n_abstained=10,
            abstained_hashes=[f"h{i}" for i in range(10)],
        )

        assert result.derivation_result.status == "abstain"
        assert result.rfl_step.symbolic_descent < 0  # Negative descent

    def test_empty_ui_payload(self, deterministic_seed: int):
        """Verify handling of empty UI payload."""
        result = run_first_organism_deterministic(
            deterministic_seed,
            ui_payload={},
        )

        assert result.ui_event.event_id
        assert result.ui_event.leaf_hash

    def test_large_number_of_abstentions(self, deterministic_seed: int):
        """Verify handling of large number of abstentions."""
        n = 1000
        hashes = [f"hash_{i:04d}" for i in range(n)]

        result = run_first_organism_deterministic(
            deterministic_seed,
            n_candidates=n,
            n_abstained=n,
            abstained_hashes=hashes,
        )

        assert len(result.derivation_result.abstained_hashes) == n
        assert result.run_hash  # Should complete without error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

