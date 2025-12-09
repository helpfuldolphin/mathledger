"""
Tests for HT-Series Replay Invariant Verifier

This module tests the INV-REPLAY-HT-* invariants as defined in
H_T_SERIES_GOVERNANCE_CHARTER.md v1.1.0 Section 10.

Test categories:
- Golden case: All invariants pass with matching data
- Failure cases: Each invariant fails appropriately with mismatched data

STATUS: PHASE II - NOT RUN IN PHASE I
"""

import hashlib
import json
import pytest
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any

from backend.ht.ht_replay_verifier import (
    check_INV_REPLAY_HT_1,
    check_INV_REPLAY_HT_2,
    check_INV_REPLAY_HT_3,
    check_INV_REPLAY_HT_4,
    check_INV_REPLAY_HT_5,
    verify_mdap_ht_replay_triangle,
    verify_all_replay_invariants,
    generate_verification_report,
    InvariantStatus,
    FailureSeverity,
    DOMAIN_PRIMARY_REPLAY_BINDING,
    DOMAIN_HT_MDAP_BIND,
    canonical_json,
    sha256_hex,
)

# Mark all tests in this file
pytestmark = [pytest.mark.ht_spec, pytest.mark.replay]


# ==============================================================================
# Test Fixtures
# ==============================================================================

@pytest.fixture
def mdap_seed() -> int:
    """MDAP seed constant from spec."""
    return 0x4D444150


@pytest.fixture
def experiment_id() -> str:
    """Sample experiment ID."""
    return "uplift_u2_goal_001"


@pytest.fixture
def sample_chain_final() -> str:
    """Sample chain final hash."""
    return "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"


@pytest.fixture
def sample_mdap_attestation_hash() -> str:
    """Sample MDAP attestation hash."""
    return "f1e2d3c4b5a6f1e2d3c4b5a6f1e2d3c4b5a6f1e2d3c4b5a6f1e2d3c4b5a6f1e2"


@pytest.fixture
def sample_ht_mdap_binding(sample_chain_final, sample_mdap_attestation_hash) -> str:
    """Compute correct MDAP binding from chain final and attestation hash."""
    binding = hashlib.sha256(
        DOMAIN_HT_MDAP_BIND +
        bytes.fromhex(sample_chain_final) +
        bytes.fromhex(sample_mdap_attestation_hash)
    ).hexdigest()
    return binding


@pytest.fixture
def sample_primary_replay_binding(sample_chain_final) -> str:
    """Compute correct primary-replay binding (identical chains)."""
    binding = hashlib.sha256(
        DOMAIN_PRIMARY_REPLAY_BINDING +
        bytes.fromhex(sample_chain_final) +
        bytes.fromhex(sample_chain_final)
    ).hexdigest()
    return binding


@pytest.fixture
def sample_series_entries() -> list:
    """Sample Ht series entries for 5 cycles."""
    entries = []
    for i in range(5):
        entries.append({
            "cycle": i,
            "H_t": hashlib.sha256(f"ht_{i}".encode()).hexdigest(),
            "R_t": hashlib.sha256(f"rt_{i}".encode()).hexdigest(),
            "U_t": None,
            "cycle_seed": hashlib.sha256(f"seed_{i}".encode()).hexdigest(),
            "chain": hashlib.sha256(f"chain_{i}".encode()).hexdigest(),
            "success": i % 2 == 0,
            "verified_count": i + 1
        })
    return entries


@pytest.fixture
def primary_ht_series(
    experiment_id,
    sample_chain_final,
    sample_ht_mdap_binding,
    sample_series_entries
) -> Dict[str, Any]:
    """Create a valid primary Ht series."""
    return {
        "meta": {
            "version": "2.0.0",
            "format": "ht_series",
            "experiment_id": experiment_id,
            "slice": "slice_uplift_goal",
            "mode": "rfl",
            "total_cycles": 5,
            "generated_utc": "2025-12-06T00:00:00Z"
        },
        "mdap_binding": {
            "mdap_seed": "0x4D444150",
            "schedule_hash": "abcd1234" * 8,
            "attestation_hash": "f1e2d3c4b5a6f1e2d3c4b5a6f1e2d3c4b5a6f1e2d3c4b5a6f1e2d3c4b5a6f1e2"
        },
        "series": sample_series_entries,
        "summary": {
            "ht_first": sample_series_entries[0]["H_t"],
            "ht_last": sample_series_entries[-1]["H_t"],
            "chain_final": sample_chain_final,
            "ht_mdap_binding": sample_ht_mdap_binding,
            "success_count": 3,
            "success_rate": 0.6,
            "total_verified": 15
        },
        "checkpoints": [
            {"cycle": 0, "H_t": sample_series_entries[0]["H_t"], "chain": sample_series_entries[0]["chain"]},
            {"cycle": 4, "H_t": sample_series_entries[4]["H_t"], "chain": sample_series_entries[4]["chain"]}
        ]
    }


@pytest.fixture
def replay_ht_series(primary_ht_series) -> Dict[str, Any]:
    """Create a matching replay Ht series (deep copy of primary)."""
    return deepcopy(primary_ht_series)


@pytest.fixture
def sample_manifest(sample_mdap_attestation_hash, experiment_id) -> Dict[str, Any]:
    """Create a sample manifest with MDAP attestation."""
    return {
        "meta": {
            "version": "2.0.0",
            "phase": "II"
        },
        "experiment": {
            "id": experiment_id,
            "family": "slice_uplift_goal",
            "status": "complete"
        },
        "mdap_attestation": {
            "mdap_seed": "0x4D444150",
            "experiment_id": experiment_id,
            "total_cycles": 5,
            "schedule_hash": "abcd1234" * 8,
            "attestation_timestamp_utc": "2025-12-06T00:00:00Z",
            "attestation_hash": sample_mdap_attestation_hash
        }
    }


@pytest.fixture
def replay_receipt(
    experiment_id,
    primary_ht_series,
    sample_chain_final,
    sample_ht_mdap_binding,
    sample_primary_replay_binding
) -> Dict[str, Any]:
    """Create a valid replay receipt."""
    # Compute series hash
    series_hash = sha256_hex(canonical_json(primary_ht_series))

    receipt_without_hash = {
        "meta": {
            "version": "2.0.0",
            "type": "replay_receipt",
            "generated_utc": "2025-12-06T01:00:00Z"
        },
        "primary_run": {
            "experiment_id": experiment_id,
            "manifest_hash": "deadbeef" * 8,
            "ht_series_hash": series_hash,
            "chain_final": sample_chain_final,
            "ht_mdap_binding": sample_ht_mdap_binding
        },
        "replay_run": {
            "replay_id": f"{experiment_id}_replay_001",
            "replay_timestamp_utc": "2025-12-06T01:00:00Z",
            "ht_series_hash": series_hash,
            "chain_final": sample_chain_final,
            "ht_mdap_binding": sample_ht_mdap_binding
        },
        "verification": {
            "series_match": True,
            "chain_match": True,
            "binding_match": True,
            "cycle_divergence": None,
            "divergence_details": None
        },
        "binding": {
            "primary_replay_binding": sample_primary_replay_binding
        }
    }

    # Compute receipt hash
    receipt_hash = sha256_hex(canonical_json(receipt_without_hash))
    receipt_without_hash["binding"]["receipt_hash"] = receipt_hash

    return receipt_without_hash


# ==============================================================================
# Golden Case Tests - All Invariants Pass
# ==============================================================================

class TestGoldenCase:
    """Tests where all invariants should pass with matching data."""

    def test_all_replay_invariants_pass(
        self,
        primary_ht_series,
        replay_ht_series,
        replay_receipt,
        sample_manifest
    ):
        """All INV-REPLAY-HT-* invariants pass with identical series."""
        results = verify_all_replay_invariants(
            primary_ht_series,
            replay_ht_series,
            replay_receipt,
            sample_manifest
        )

        assert len(results) == 5, "Should check all 5 replay invariants"

        for result in results:
            assert result.status == InvariantStatus.PASS, (
                f"{result.invariant_id} should PASS: {result.message}"
            )

    def test_triangle_verification_passes(
        self,
        primary_ht_series,
        replay_ht_series,
        replay_receipt,
        sample_manifest
    ):
        """MDAP-Ht-Replay triangle verification passes with valid data."""
        result = verify_mdap_ht_replay_triangle(
            sample_manifest,
            primary_ht_series,
            replay_ht_series,
            replay_receipt
        )

        assert result.triangle_valid is True
        assert result.mdap_vertex_valid is True
        assert result.primary_ht_vertex_valid is True
        assert result.replay_ht_vertex_valid is True
        assert len(result.invariants_checked) == 5

    def test_verification_report_all_passed(
        self,
        primary_ht_series,
        replay_ht_series,
        replay_receipt,
        sample_manifest,
        experiment_id
    ):
        """Verification report shows all passed when data matches."""
        results = verify_all_replay_invariants(
            primary_ht_series,
            replay_ht_series,
            replay_receipt,
            sample_manifest
        )

        report = generate_verification_report(
            results,
            experiment_id,
            f"{experiment_id}_replay_001"
        )

        assert report["summary"]["all_passed"] is True
        assert report["summary"]["passed"] == 5
        assert report["summary"]["failed"] == 0
        assert "failure_summary" not in report


# ==============================================================================
# INV-REPLAY-HT-1 Tests - Series Identity
# ==============================================================================

class TestINV_REPLAY_HT_1:
    """Tests for INV-REPLAY-HT-1: Ht series identity."""

    def test_passes_with_identical_series(self, primary_ht_series, replay_ht_series):
        """INV-REPLAY-HT-1 passes when series are byte-identical."""
        result = check_INV_REPLAY_HT_1(primary_ht_series, replay_ht_series)

        assert result.invariant_id == "INV-REPLAY-HT-1"
        assert result.status == InvariantStatus.PASS
        assert "byte-identical" in result.message.lower()

    def test_fails_with_single_cycle_mismatch(self, primary_ht_series, replay_ht_series):
        """INV-REPLAY-HT-1 fails when a single Ht value differs."""
        # Modify one Ht value in replay
        replay_ht_series["series"][2]["H_t"] = "modified" + "0" * 56

        result = check_INV_REPLAY_HT_1(primary_ht_series, replay_ht_series)

        assert result.status == InvariantStatus.FAIL
        assert result.severity == FailureSeverity.CRITICAL
        assert result.expected != result.actual
        assert "differ" in result.message.lower()

    def test_fails_with_metadata_change(self, primary_ht_series, replay_ht_series):
        """INV-REPLAY-HT-1 fails when metadata differs."""
        replay_ht_series["meta"]["generated_utc"] = "2025-12-07T00:00:00Z"

        result = check_INV_REPLAY_HT_1(primary_ht_series, replay_ht_series)

        assert result.status == InvariantStatus.FAIL
        assert result.severity == FailureSeverity.CRITICAL

    def test_fails_with_extra_cycle(self, primary_ht_series, replay_ht_series):
        """INV-REPLAY-HT-1 fails when replay has extra cycles."""
        replay_ht_series["series"].append({
            "cycle": 5,
            "H_t": "extra" + "0" * 59,
            "R_t": "extra" + "0" * 59,
            "U_t": None,
            "cycle_seed": "extra" + "0" * 59,
            "chain": "extra" + "0" * 59,
            "success": True,
            "verified_count": 1
        })

        result = check_INV_REPLAY_HT_1(primary_ht_series, replay_ht_series)

        assert result.status == InvariantStatus.FAIL


# ==============================================================================
# INV-REPLAY-HT-2 Tests - Chain Final Equivalence
# ==============================================================================

class TestINV_REPLAY_HT_2:
    """Tests for INV-REPLAY-HT-2: Chain final equivalence."""

    def test_passes_with_matching_chain_finals(self, primary_ht_series, replay_ht_series):
        """INV-REPLAY-HT-2 passes when chain finals match."""
        result = check_INV_REPLAY_HT_2(primary_ht_series, replay_ht_series)

        assert result.invariant_id == "INV-REPLAY-HT-2"
        assert result.status == InvariantStatus.PASS
        assert "identical" in result.message.lower()

    def test_fails_with_different_chain_finals(self, primary_ht_series, replay_ht_series):
        """INV-REPLAY-HT-2 fails when chain finals differ."""
        replay_ht_series["summary"]["chain_final"] = "different" + "0" * 55

        result = check_INV_REPLAY_HT_2(primary_ht_series, replay_ht_series)

        assert result.status == InvariantStatus.FAIL
        assert result.severity == FailureSeverity.CRITICAL
        assert result.expected == primary_ht_series["summary"]["chain_final"]
        assert result.actual == "different" + "0" * 55

    def test_reports_divergence_cycle(self, primary_ht_series, replay_ht_series):
        """INV-REPLAY-HT-2 reports first divergence cycle when series differ."""
        # Modify series at cycle 2
        replay_ht_series["series"][2]["H_t"] = "diverged" + "0" * 56
        replay_ht_series["summary"]["chain_final"] = "different" + "0" * 55

        result = check_INV_REPLAY_HT_2(primary_ht_series, replay_ht_series)

        assert result.status == InvariantStatus.FAIL
        assert result.cycle == 2
        assert "first_divergence_cycle" in result.details

    def test_fails_with_missing_primary_chain(self, primary_ht_series, replay_ht_series):
        """INV-REPLAY-HT-2 fails when primary chain_final is missing."""
        del primary_ht_series["summary"]["chain_final"]

        result = check_INV_REPLAY_HT_2(primary_ht_series, replay_ht_series)

        assert result.status == InvariantStatus.FAIL
        assert "missing" in result.message.lower()

    def test_fails_with_missing_replay_chain(self, primary_ht_series, replay_ht_series):
        """INV-REPLAY-HT-2 fails when replay chain_final is missing."""
        del replay_ht_series["summary"]["chain_final"]

        result = check_INV_REPLAY_HT_2(primary_ht_series, replay_ht_series)

        assert result.status == InvariantStatus.FAIL
        assert "missing" in result.message.lower()


# ==============================================================================
# INV-REPLAY-HT-3 Tests - MDAP Binding Preservation
# ==============================================================================

class TestINV_REPLAY_HT_3:
    """Tests for INV-REPLAY-HT-3: MDAP binding preservation."""

    def test_passes_with_matching_bindings(
        self,
        primary_ht_series,
        replay_ht_series,
        sample_manifest
    ):
        """INV-REPLAY-HT-3 passes when MDAP bindings match."""
        result = check_INV_REPLAY_HT_3(
            primary_ht_series,
            replay_ht_series,
            sample_manifest
        )

        assert result.invariant_id == "INV-REPLAY-HT-3"
        assert result.status == InvariantStatus.PASS

    def test_fails_with_different_bindings(self, primary_ht_series, replay_ht_series):
        """INV-REPLAY-HT-3 fails when MDAP bindings differ."""
        replay_ht_series["summary"]["ht_mdap_binding"] = "different" + "0" * 55

        result = check_INV_REPLAY_HT_3(primary_ht_series, replay_ht_series)

        assert result.status == InvariantStatus.FAIL
        assert result.severity == FailureSeverity.CRITICAL
        assert "BINDING_MISMATCH" in result.details.get("reason", "")

    def test_fails_with_missing_primary_binding(self, primary_ht_series, replay_ht_series):
        """INV-REPLAY-HT-3 fails when primary ht_mdap_binding is missing."""
        del primary_ht_series["summary"]["ht_mdap_binding"]

        result = check_INV_REPLAY_HT_3(primary_ht_series, replay_ht_series)

        assert result.status == InvariantStatus.FAIL
        assert "MISSING_PRIMARY_BINDING" in result.details.get("error", "")

    def test_fails_with_incorrect_binding_computation(
        self,
        primary_ht_series,
        replay_ht_series,
        sample_manifest
    ):
        """INV-REPLAY-HT-3 fails when binding doesn't match expected computation."""
        # Set both to same wrong value
        wrong_binding = "wrongbinding" + "0" * 52
        primary_ht_series["summary"]["ht_mdap_binding"] = wrong_binding
        replay_ht_series["summary"]["ht_mdap_binding"] = wrong_binding

        result = check_INV_REPLAY_HT_3(
            primary_ht_series,
            replay_ht_series,
            sample_manifest
        )

        assert result.status == InvariantStatus.FAIL
        assert "BINDING_COMPUTATION_ERROR" in result.details.get("reason", "")


# ==============================================================================
# INV-REPLAY-HT-4 Tests - Replay Receipt Integrity
# ==============================================================================

class TestINV_REPLAY_HT_4:
    """Tests for INV-REPLAY-HT-4: Replay receipt integrity."""

    def test_passes_with_valid_receipt(
        self,
        replay_receipt,
        primary_ht_series,
        replay_ht_series
    ):
        """INV-REPLAY-HT-4 passes with correctly constructed receipt."""
        result = check_INV_REPLAY_HT_4(
            replay_receipt,
            primary_ht_series=primary_ht_series,
            replay_ht_series=replay_ht_series
        )

        assert result.invariant_id == "INV-REPLAY-HT-4"
        assert result.status == InvariantStatus.PASS

    def test_fails_with_wrong_receipt_hash(
        self,
        replay_receipt,
        primary_ht_series,
        replay_ht_series
    ):
        """INV-REPLAY-HT-4 fails when receipt_hash is incorrect."""
        replay_receipt["binding"]["receipt_hash"] = "wronghash" + "0" * 55

        result = check_INV_REPLAY_HT_4(
            replay_receipt,
            primary_ht_series=primary_ht_series,
            replay_ht_series=replay_ht_series
        )

        assert result.status == InvariantStatus.FAIL
        assert result.severity == FailureSeverity.HIGH
        assert any(
            c["check"] == "RECEIPT_SELF_HASH"
            for c in result.details.get("checks_failed", [])
        )

    def test_fails_with_wrong_chain_final_in_receipt(
        self,
        replay_receipt,
        primary_ht_series,
        replay_ht_series
    ):
        """INV-REPLAY-HT-4 fails when receipt chain_final doesn't match series."""
        replay_receipt["primary_run"]["chain_final"] = "wrongchain" + "0" * 54

        result = check_INV_REPLAY_HT_4(
            replay_receipt,
            primary_ht_series=primary_ht_series,
            replay_ht_series=replay_ht_series
        )

        assert result.status == InvariantStatus.FAIL
        assert any(
            c["check"] == "PRIMARY_CHAIN_FINAL"
            for c in result.details.get("checks_failed", [])
        )


# ==============================================================================
# INV-REPLAY-HT-5 Tests - Primary-Replay Binding Hash
# ==============================================================================

class TestINV_REPLAY_HT_5:
    """Tests for INV-REPLAY-HT-5: Primary-replay binding hash."""

    def test_passes_with_correct_binding(self, replay_receipt):
        """INV-REPLAY-HT-5 passes with correctly computed binding."""
        result = check_INV_REPLAY_HT_5(replay_receipt)

        assert result.invariant_id == "INV-REPLAY-HT-5"
        assert result.status == InvariantStatus.PASS

    def test_fails_with_wrong_binding(self, replay_receipt):
        """INV-REPLAY-HT-5 fails when primary_replay_binding is incorrect."""
        replay_receipt["binding"]["primary_replay_binding"] = "wrongbinding" + "0" * 52

        result = check_INV_REPLAY_HT_5(replay_receipt)

        assert result.status == InvariantStatus.FAIL
        assert result.severity == FailureSeverity.HIGH
        assert result.expected != result.actual

    def test_fails_with_missing_primary_chain(self, replay_receipt):
        """INV-REPLAY-HT-5 fails when primary chain_final is missing."""
        del replay_receipt["primary_run"]["chain_final"]

        result = check_INV_REPLAY_HT_5(replay_receipt)

        assert result.status == InvariantStatus.FAIL
        assert "MISSING_PRIMARY_CHAIN" in result.details.get("error", "")

    def test_fails_with_missing_replay_chain(self, replay_receipt):
        """INV-REPLAY-HT-5 fails when replay chain_final is missing."""
        del replay_receipt["replay_run"]["chain_final"]

        result = check_INV_REPLAY_HT_5(replay_receipt)

        assert result.status == InvariantStatus.FAIL
        assert "MISSING_REPLAY_CHAIN" in result.details.get("error", "")

    def test_fails_with_missing_binding(self, replay_receipt):
        """INV-REPLAY-HT-5 fails when primary_replay_binding is missing."""
        del replay_receipt["binding"]["primary_replay_binding"]

        result = check_INV_REPLAY_HT_5(replay_receipt)

        assert result.status == InvariantStatus.FAIL
        assert "MISSING_BINDING" in result.details.get("error", "")


# ==============================================================================
# Triangle Verification Tests
# ==============================================================================

class TestTriangleVerification:
    """Tests for MDAP-Ht-Replay triangle verification."""

    def test_triangle_invalid_with_series_mismatch(
        self,
        primary_ht_series,
        replay_ht_series,
        replay_receipt,
        sample_manifest
    ):
        """Triangle verification fails when series don't match."""
        replay_ht_series["series"][0]["H_t"] = "mismatch" + "0" * 56

        result = verify_mdap_ht_replay_triangle(
            sample_manifest,
            primary_ht_series,
            replay_ht_series,
            replay_receipt
        )

        assert result.triangle_valid is False
        assert any(
            r.invariant_id == "INV-REPLAY-HT-1" and r.status == InvariantStatus.FAIL
            for r in result.results
        )

    def test_triangle_invalid_with_missing_mdap(
        self,
        primary_ht_series,
        replay_ht_series,
        replay_receipt,
        sample_manifest
    ):
        """Triangle verification fails when MDAP attestation is missing."""
        del sample_manifest["mdap_attestation"]["attestation_hash"]

        result = verify_mdap_ht_replay_triangle(
            sample_manifest,
            primary_ht_series,
            replay_ht_series,
            replay_receipt
        )

        assert result.mdap_vertex_valid is False

    def test_triangle_reports_all_invariants(
        self,
        primary_ht_series,
        replay_ht_series,
        replay_receipt,
        sample_manifest
    ):
        """Triangle verification checks all 5 replay invariants."""
        result = verify_mdap_ht_replay_triangle(
            sample_manifest,
            primary_ht_series,
            replay_ht_series,
            replay_receipt
        )

        expected_invariants = [
            "INV-REPLAY-HT-1",
            "INV-REPLAY-HT-2",
            "INV-REPLAY-HT-3",
            "INV-REPLAY-HT-4",
            "INV-REPLAY-HT-5"
        ]

        for inv in expected_invariants:
            assert inv in result.invariants_checked


# ==============================================================================
# Verification Report Tests
# ==============================================================================

class TestVerificationReport:
    """Tests for verification report generation."""

    def test_report_includes_failure_summary_on_failure(
        self,
        primary_ht_series,
        replay_ht_series,
        replay_receipt,
        sample_manifest,
        experiment_id
    ):
        """Report includes failure_summary when invariants fail."""
        # Cause a failure
        replay_ht_series["series"][0]["H_t"] = "mismatch" + "0" * 56

        results = verify_all_replay_invariants(
            primary_ht_series,
            replay_ht_series,
            replay_receipt,
            sample_manifest
        )

        report = generate_verification_report(
            results,
            experiment_id,
            f"{experiment_id}_replay_001"
        )

        assert report["summary"]["all_passed"] is False
        assert "failure_summary" in report
        assert report["failure_summary"]["critical_count"] >= 1
        assert report["failure_summary"]["impact"] == "INVALID"

    def test_report_structure_conforms_to_spec(
        self,
        primary_ht_series,
        replay_ht_series,
        replay_receipt,
        sample_manifest,
        experiment_id
    ):
        """Report structure matches Charter §10.5.4 requirements."""
        results = verify_all_replay_invariants(
            primary_ht_series,
            replay_ht_series,
            replay_receipt,
            sample_manifest
        )

        report = generate_verification_report(
            results,
            experiment_id,
            f"{experiment_id}_replay_001"
        )

        # Check required fields
        assert "meta" in report
        assert report["meta"]["report_version"] == "2.0.0"
        assert "generated_utc" in report["meta"]

        assert "experiment" in report
        assert report["experiment"]["experiment_id"] == experiment_id

        assert "summary" in report
        assert "all_passed" in report["summary"]
        assert "total_checks" in report["summary"]
        assert "passed" in report["summary"]
        assert "failed" in report["summary"]

        assert "results" in report
        assert len(report["results"]) == 5


# ==============================================================================
# Triangle Contract v1.0.0 Tests
# ==============================================================================

class TestTriangleContract:
    """Tests for Triangle Contract v1.0.0 compliance."""

    def test_contract_version_present_in_report(
        self,
        primary_ht_series,
        replay_ht_series,
        replay_receipt,
        sample_manifest,
        experiment_id
    ):
        """Report includes contract_version in meta section."""
        results = verify_all_replay_invariants(
            primary_ht_series,
            replay_ht_series,
            replay_receipt,
            sample_manifest
        )

        report = generate_verification_report(
            results,
            experiment_id,
            f"{experiment_id}_replay_001",
            primary_ht_series=primary_ht_series,
            replay_ht_series=replay_ht_series
        )

        assert "contract_version" in report["meta"]
        assert report["meta"]["contract_version"] == "1.0.0"

    def test_contract_section_present(
        self,
        primary_ht_series,
        replay_ht_series,
        replay_receipt,
        sample_manifest,
        experiment_id
    ):
        """Report includes contract section with required fields."""
        results = verify_all_replay_invariants(
            primary_ht_series,
            replay_ht_series,
            replay_receipt,
            sample_manifest
        )

        report = generate_verification_report(
            results,
            experiment_id,
            f"{experiment_id}_replay_001",
            primary_ht_series=primary_ht_series,
            replay_ht_series=replay_ht_series
        )

        assert "contract" in report
        contract = report["contract"]

        assert contract["version"] == "1.0.0"
        assert "primary_ht_series_hash" in contract
        assert "replay_ht_series_hash" in contract
        assert "invariant_status_map" in contract
        assert "mdap_binding_status" in contract

    def test_invariant_status_map_complete(
        self,
        primary_ht_series,
        replay_ht_series,
        replay_receipt,
        sample_manifest,
        experiment_id
    ):
        """Invariant status map contains all 5 INV-REPLAY-HT-* entries."""
        results = verify_all_replay_invariants(
            primary_ht_series,
            replay_ht_series,
            replay_receipt,
            sample_manifest
        )

        report = generate_verification_report(
            results,
            experiment_id,
            f"{experiment_id}_replay_001",
            primary_ht_series=primary_ht_series,
            replay_ht_series=replay_ht_series
        )

        status_map = report["contract"]["invariant_status_map"]

        expected_invariants = [
            "INV-REPLAY-HT-1",
            "INV-REPLAY-HT-2",
            "INV-REPLAY-HT-3",
            "INV-REPLAY-HT-4",
            "INV-REPLAY-HT-5"
        ]

        for inv in expected_invariants:
            assert inv in status_map
            assert status_map[inv] in ["PASS", "FAIL"]

    def test_invariant_status_map_reflects_failures(
        self,
        primary_ht_series,
        replay_ht_series,
        replay_receipt,
        sample_manifest,
        experiment_id
    ):
        """Invariant status map correctly shows FAIL for failed invariants."""
        # Cause INV-REPLAY-HT-1 to fail
        replay_ht_series["series"][0]["H_t"] = "mismatch" + "0" * 56

        results = verify_all_replay_invariants(
            primary_ht_series,
            replay_ht_series,
            replay_receipt,
            sample_manifest
        )

        report = generate_verification_report(
            results,
            experiment_id,
            f"{experiment_id}_replay_001",
            primary_ht_series=primary_ht_series,
            replay_ht_series=replay_ht_series
        )

        status_map = report["contract"]["invariant_status_map"]
        assert status_map["INV-REPLAY-HT-1"] == "FAIL"

    def test_series_hashes_computed(
        self,
        primary_ht_series,
        replay_ht_series,
        replay_receipt,
        sample_manifest,
        experiment_id
    ):
        """Primary and replay series hashes are computed when data provided."""
        results = verify_all_replay_invariants(
            primary_ht_series,
            replay_ht_series,
            replay_receipt,
            sample_manifest
        )

        report = generate_verification_report(
            results,
            experiment_id,
            f"{experiment_id}_replay_001",
            primary_ht_series=primary_ht_series,
            replay_ht_series=replay_ht_series
        )

        contract = report["contract"]
        assert contract["primary_ht_series_hash"] is not None
        assert contract["replay_ht_series_hash"] is not None
        assert len(contract["primary_ht_series_hash"]) == 64  # SHA256 hex
        assert len(contract["replay_ht_series_hash"]) == 64

    def test_mdap_binding_status_reflects_inv3(
        self,
        primary_ht_series,
        replay_ht_series,
        replay_receipt,
        sample_manifest,
        experiment_id
    ):
        """MDAP binding status reflects INV-REPLAY-HT-3 result."""
        results = verify_all_replay_invariants(
            primary_ht_series,
            replay_ht_series,
            replay_receipt,
            sample_manifest
        )

        report = generate_verification_report(
            results,
            experiment_id,
            f"{experiment_id}_replay_001",
            primary_ht_series=primary_ht_series,
            replay_ht_series=replay_ht_series
        )

        assert report["contract"]["mdap_binding_status"] == "PASS"

    def test_backward_compatibility_without_series_data(
        self,
        primary_ht_series,
        replay_ht_series,
        replay_receipt,
        sample_manifest,
        experiment_id
    ):
        """Report generation works without series data (backward compatible)."""
        results = verify_all_replay_invariants(
            primary_ht_series,
            replay_ht_series,
            replay_receipt,
            sample_manifest
        )

        # Call without series data (old API)
        report = generate_verification_report(
            results,
            experiment_id,
            f"{experiment_id}_replay_001"
        )

        # Should still have contract section but hashes are None
        assert "contract" in report
        assert report["contract"]["primary_ht_series_hash"] is None
        assert report["contract"]["replay_ht_series_hash"] is None
        # Status map should still be populated
        assert len(report["contract"]["invariant_status_map"]) == 5


# ==============================================================================
# Governance Summary Tests
# ==============================================================================

class TestGovernanceSummary:
    """Tests for summarize_ht_replay_for_governance helper."""

    def test_all_pass_returns_ok(
        self,
        primary_ht_series,
        replay_ht_series,
        replay_receipt,
        sample_manifest,
        experiment_id
    ):
        """When all invariants pass, ht_replay_status is OK."""
        from backend.ht.ht_replay_verifier import summarize_ht_replay_for_governance

        results = verify_all_replay_invariants(
            primary_ht_series,
            replay_ht_series,
            replay_receipt,
            sample_manifest
        )

        report = generate_verification_report(
            results,
            experiment_id,
            f"{experiment_id}_replay_001",
            primary_ht_series=primary_ht_series,
            replay_ht_series=replay_ht_series
        )

        summary = summarize_ht_replay_for_governance(report)

        assert summary["all_critical_invariants_pass"] is True
        assert summary["high_invariants_pass"] is True
        assert summary["ht_replay_status"] == "OK"

    def test_critical_fail_returns_fail(
        self,
        primary_ht_series,
        replay_ht_series,
        replay_receipt,
        sample_manifest,
        experiment_id
    ):
        """When critical invariant fails, ht_replay_status is FAIL."""
        from backend.ht.ht_replay_verifier import summarize_ht_replay_for_governance

        # Cause INV-REPLAY-HT-1 (critical) to fail
        replay_ht_series["series"][0]["H_t"] = "mismatch" + "0" * 56

        results = verify_all_replay_invariants(
            primary_ht_series,
            replay_ht_series,
            replay_receipt,
            sample_manifest
        )

        report = generate_verification_report(
            results,
            experiment_id,
            f"{experiment_id}_replay_001",
            primary_ht_series=primary_ht_series,
            replay_ht_series=replay_ht_series
        )

        summary = summarize_ht_replay_for_governance(report)

        assert summary["all_critical_invariants_pass"] is False
        assert summary["ht_replay_status"] == "FAIL"

    def test_high_only_fail_returns_warn(
        self,
        primary_ht_series,
        replay_ht_series,
        replay_receipt,
        sample_manifest,
        experiment_id
    ):
        """When only high severity invariant fails, ht_replay_status is WARN."""
        from backend.ht.ht_replay_verifier import summarize_ht_replay_for_governance

        # Cause INV-REPLAY-HT-5 (high) to fail by corrupting binding
        replay_receipt["binding"]["primary_replay_binding"] = "wrongbinding" + "0" * 52

        results = verify_all_replay_invariants(
            primary_ht_series,
            replay_ht_series,
            replay_receipt,
            sample_manifest
        )

        report = generate_verification_report(
            results,
            experiment_id,
            f"{experiment_id}_replay_001",
            primary_ht_series=primary_ht_series,
            replay_ht_series=replay_ht_series
        )

        summary = summarize_ht_replay_for_governance(report)

        assert summary["all_critical_invariants_pass"] is True
        assert summary["high_invariants_pass"] is False
        assert summary["ht_replay_status"] == "WARN"

    def test_summary_contains_required_fields(
        self,
        primary_ht_series,
        replay_ht_series,
        replay_receipt,
        sample_manifest,
        experiment_id
    ):
        """Summary contains all required fields for MAAS/dashboard."""
        from backend.ht.ht_replay_verifier import summarize_ht_replay_for_governance

        results = verify_all_replay_invariants(
            primary_ht_series,
            replay_ht_series,
            replay_receipt,
            sample_manifest
        )

        report = generate_verification_report(
            results,
            experiment_id,
            f"{experiment_id}_replay_001",
            primary_ht_series=primary_ht_series,
            replay_ht_series=replay_ht_series
        )

        summary = summarize_ht_replay_for_governance(report)

        assert "all_critical_invariants_pass" in summary
        assert "high_invariants_pass" in summary
        assert "ht_replay_status" in summary
        assert "contract_version" in summary
        assert "experiment_id" in summary
        assert "replay_id" in summary

    def test_summary_contract_version_matches(
        self,
        primary_ht_series,
        replay_ht_series,
        replay_receipt,
        sample_manifest,
        experiment_id
    ):
        """Summary contract_version matches report contract version."""
        from backend.ht.ht_replay_verifier import (
            summarize_ht_replay_for_governance,
            TRIANGLE_CONTRACT_VERSION
        )

        results = verify_all_replay_invariants(
            primary_ht_series,
            replay_ht_series,
            replay_receipt,
            sample_manifest
        )

        report = generate_verification_report(
            results,
            experiment_id,
            f"{experiment_id}_replay_001",
            primary_ht_series=primary_ht_series,
            replay_ht_series=replay_ht_series
        )

        summary = summarize_ht_replay_for_governance(report)

        assert summary["contract_version"] == TRIANGLE_CONTRACT_VERSION
        assert summary["contract_version"] == "1.0.0"

    def test_inv3_fail_is_critical(
        self,
        primary_ht_series,
        replay_ht_series,
        replay_receipt,
        sample_manifest,
        experiment_id
    ):
        """INV-REPLAY-HT-3 failure is classified as critical (FAIL status)."""
        from backend.ht.ht_replay_verifier import summarize_ht_replay_for_governance

        # Cause INV-REPLAY-HT-3 to fail
        replay_ht_series["summary"]["ht_mdap_binding"] = "different" + "0" * 55

        results = verify_all_replay_invariants(
            primary_ht_series,
            replay_ht_series,
            replay_receipt,
            sample_manifest
        )

        report = generate_verification_report(
            results,
            experiment_id,
            f"{experiment_id}_replay_001",
            primary_ht_series=primary_ht_series,
            replay_ht_series=replay_ht_series
        )

        summary = summarize_ht_replay_for_governance(report)

        assert summary["all_critical_invariants_pass"] is False
        assert summary["ht_replay_status"] == "FAIL"

    def test_inv4_fail_is_high(
        self,
        primary_ht_series,
        replay_ht_series,
        replay_receipt,
        sample_manifest,
        experiment_id
    ):
        """INV-REPLAY-HT-4 failure is classified as high (WARN status)."""
        from backend.ht.ht_replay_verifier import summarize_ht_replay_for_governance

        # Cause INV-REPLAY-HT-4 to fail by corrupting receipt chain_final
        # Use valid hex string that differs from the series chain_final
        replay_receipt["primary_run"]["chain_final"] = "abcd" * 16  # 64 hex chars

        results = verify_all_replay_invariants(
            primary_ht_series,
            replay_ht_series,
            replay_receipt,
            sample_manifest
        )

        report = generate_verification_report(
            results,
            experiment_id,
            f"{experiment_id}_replay_001",
            primary_ht_series=primary_ht_series,
            replay_ht_series=replay_ht_series
        )

        summary = summarize_ht_replay_for_governance(report)

        assert summary["all_critical_invariants_pass"] is True
        assert summary["high_invariants_pass"] is False
        assert summary["ht_replay_status"] == "WARN"


# ==============================================================================
# Phase III: HT Replay History Ledger Tests
# ==============================================================================

class TestHTReplayHistoryLedger:
    """Tests for build_ht_replay_history function."""

    def test_empty_reports_returns_empty_history(self):
        """Empty reports list returns empty history."""
        from backend.ht.ht_replay_verifier import build_ht_replay_history

        history = build_ht_replay_history([])

        assert history["total_runs"] == 0
        assert history["ok_count"] == 0
        assert history["warn_count"] == 0
        assert history["fail_count"] == 0
        assert history["run_entries"] == []
        assert history["recurrent_failures"] == {}

    def test_single_ok_report(
        self,
        primary_ht_series,
        replay_ht_series,
        replay_receipt,
        sample_manifest,
        experiment_id
    ):
        """Single OK report creates correct history."""
        from backend.ht.ht_replay_verifier import build_ht_replay_history

        results = verify_all_replay_invariants(
            primary_ht_series,
            replay_ht_series,
            replay_receipt,
            sample_manifest
        )

        report = generate_verification_report(
            results,
            experiment_id,
            f"{experiment_id}_replay_001",
            primary_ht_series=primary_ht_series,
            replay_ht_series=replay_ht_series
        )

        history = build_ht_replay_history([report])

        assert history["total_runs"] == 1
        assert history["ok_count"] == 1
        assert history["warn_count"] == 0
        assert history["fail_count"] == 0
        assert len(history["run_entries"]) == 1
        assert history["run_entries"][0]["ht_replay_status"] == "OK"

    def test_multiple_reports_counts_correctly(
        self,
        primary_ht_series,
        replay_ht_series,
        replay_receipt,
        sample_manifest,
        experiment_id
    ):
        """Multiple reports with different statuses count correctly."""
        from backend.ht.ht_replay_verifier import build_ht_replay_history
        from copy import deepcopy

        reports = []

        # Report 1: OK
        results1 = verify_all_replay_invariants(
            primary_ht_series,
            replay_ht_series,
            replay_receipt,
            sample_manifest
        )
        report1 = generate_verification_report(
            results1, experiment_id, f"{experiment_id}_replay_001",
            primary_ht_series=primary_ht_series,
            replay_ht_series=replay_ht_series
        )
        report1["meta"]["generated_utc"] = "2025-12-01T00:00:00Z"
        reports.append(report1)

        # Report 2: FAIL (corrupt series)
        replay_ht_series_bad = deepcopy(replay_ht_series)
        replay_ht_series_bad["series"][0]["H_t"] = "mismatch" + "0" * 56
        results2 = verify_all_replay_invariants(
            primary_ht_series,
            replay_ht_series_bad,
            replay_receipt,
            sample_manifest
        )
        report2 = generate_verification_report(
            results2, experiment_id, f"{experiment_id}_replay_002",
            primary_ht_series=primary_ht_series,
            replay_ht_series=replay_ht_series_bad
        )
        report2["meta"]["generated_utc"] = "2025-12-02T00:00:00Z"
        reports.append(report2)

        # Report 3: WARN (corrupt receipt binding)
        replay_receipt_bad = deepcopy(replay_receipt)
        replay_receipt_bad["binding"]["primary_replay_binding"] = "abcd" * 16
        results3 = verify_all_replay_invariants(
            primary_ht_series,
            replay_ht_series,
            replay_receipt_bad,
            sample_manifest
        )
        report3 = generate_verification_report(
            results3, experiment_id, f"{experiment_id}_replay_003",
            primary_ht_series=primary_ht_series,
            replay_ht_series=replay_ht_series
        )
        report3["meta"]["generated_utc"] = "2025-12-03T00:00:00Z"
        reports.append(report3)

        history = build_ht_replay_history(reports)

        assert history["total_runs"] == 3
        assert history["ok_count"] == 1
        assert history["fail_count"] == 1
        assert history["warn_count"] == 1

    def test_recurrent_failures_tracked(
        self,
        primary_ht_series,
        replay_ht_series,
        replay_receipt,
        sample_manifest,
        experiment_id
    ):
        """Recurrent failures are tracked when same invariant fails multiple times."""
        from backend.ht.ht_replay_verifier import build_ht_replay_history
        from copy import deepcopy

        reports = []

        # Two reports with same failure (INV-REPLAY-HT-1)
        for i in range(2):
            replay_ht_series_bad = deepcopy(replay_ht_series)
            replay_ht_series_bad["series"][0]["H_t"] = f"mismatch{i}" + "0" * 55
            results = verify_all_replay_invariants(
                primary_ht_series,
                replay_ht_series_bad,
                replay_receipt,
                sample_manifest
            )
            report = generate_verification_report(
                results, experiment_id, f"{experiment_id}_replay_{i:03d}",
                primary_ht_series=primary_ht_series,
                replay_ht_series=replay_ht_series_bad
            )
            report["meta"]["generated_utc"] = f"2025-12-0{i+1}T00:00:00Z"
            reports.append(report)

        history = build_ht_replay_history(reports)

        assert "INV-REPLAY-HT-1" in history["recurrent_failures"]
        assert history["recurrent_failures"]["INV-REPLAY-HT-1"] == 2

    def test_entries_sorted_by_timestamp(
        self,
        primary_ht_series,
        replay_ht_series,
        replay_receipt,
        sample_manifest,
        experiment_id
    ):
        """History entries are sorted by timestamp."""
        from backend.ht.ht_replay_verifier import build_ht_replay_history

        reports = []
        timestamps = ["2025-12-03T00:00:00Z", "2025-12-01T00:00:00Z", "2025-12-02T00:00:00Z"]

        for i, ts in enumerate(timestamps):
            results = verify_all_replay_invariants(
                primary_ht_series,
                replay_ht_series,
                replay_receipt,
                sample_manifest
            )
            report = generate_verification_report(
                results, experiment_id, f"{experiment_id}_replay_{i:03d}",
                primary_ht_series=primary_ht_series,
                replay_ht_series=replay_ht_series
            )
            report["meta"]["generated_utc"] = ts
            reports.append(report)

        history = build_ht_replay_history(reports)

        entry_timestamps = [e["timestamp_utc"] for e in history["run_entries"]]
        assert entry_timestamps == sorted(entry_timestamps)


# ==============================================================================
# Phase III: Drift Classifier Tests
# ==============================================================================

class TestDriftClassifier:
    """Tests for compare_ht_history function."""

    def test_stable_when_same_status(self):
        """STABLE when old and new have same status and health score."""
        from backend.ht.ht_replay_verifier import compare_ht_history

        old_history = {
            "run_entries": [{"ht_replay_status": "OK"}],
            "ok_count": 1,
            "warn_count": 0,
            "fail_count": 0,
            "total_runs": 1,
            "recurrent_failures": {}
        }

        new_history = {
            "run_entries": [{"ht_replay_status": "OK"}],
            "ok_count": 1,
            "warn_count": 0,
            "fail_count": 0,
            "total_runs": 1,
            "recurrent_failures": {}
        }

        result = compare_ht_history(old_history, new_history)

        assert result["status"] == "STABLE"
        assert result["old_latest_status"] == "OK"
        assert result["new_latest_status"] == "OK"

    def test_improved_fail_to_ok(self):
        """IMPROVED when status changes from FAIL to OK."""
        from backend.ht.ht_replay_verifier import compare_ht_history

        old_history = {
            "run_entries": [{"ht_replay_status": "FAIL"}],
            "ok_count": 0,
            "warn_count": 0,
            "fail_count": 1,
            "total_runs": 1,
            "recurrent_failures": {}
        }

        new_history = {
            "run_entries": [{"ht_replay_status": "OK"}],
            "ok_count": 1,
            "warn_count": 0,
            "fail_count": 0,
            "total_runs": 1,
            "recurrent_failures": {}
        }

        result = compare_ht_history(old_history, new_history)

        assert result["status"] == "IMPROVED"

    def test_improved_fail_to_warn(self):
        """IMPROVED when status changes from FAIL to WARN."""
        from backend.ht.ht_replay_verifier import compare_ht_history

        old_history = {
            "run_entries": [{"ht_replay_status": "FAIL"}],
            "ok_count": 0,
            "warn_count": 0,
            "fail_count": 1,
            "total_runs": 1,
            "recurrent_failures": {}
        }

        new_history = {
            "run_entries": [{"ht_replay_status": "WARN"}],
            "ok_count": 0,
            "warn_count": 1,
            "fail_count": 0,
            "total_runs": 1,
            "recurrent_failures": {}
        }

        result = compare_ht_history(old_history, new_history)

        assert result["status"] == "IMPROVED"

    def test_regressed_ok_to_fail(self):
        """REGRESSED when status changes from OK to FAIL."""
        from backend.ht.ht_replay_verifier import compare_ht_history

        old_history = {
            "run_entries": [{"ht_replay_status": "OK"}],
            "ok_count": 1,
            "warn_count": 0,
            "fail_count": 0,
            "total_runs": 1,
            "recurrent_failures": {}
        }

        new_history = {
            "run_entries": [{"ht_replay_status": "FAIL"}],
            "ok_count": 0,
            "warn_count": 0,
            "fail_count": 1,
            "total_runs": 1,
            "recurrent_failures": {}
        }

        result = compare_ht_history(old_history, new_history)

        assert result["status"] == "REGRESSED"

    def test_regressed_ok_to_warn(self):
        """REGRESSED when status changes from OK to WARN."""
        from backend.ht.ht_replay_verifier import compare_ht_history

        old_history = {
            "run_entries": [{"ht_replay_status": "OK"}],
            "ok_count": 1,
            "warn_count": 0,
            "fail_count": 0,
            "total_runs": 1,
            "recurrent_failures": {}
        }

        new_history = {
            "run_entries": [{"ht_replay_status": "WARN"}],
            "ok_count": 0,
            "warn_count": 1,
            "fail_count": 0,
            "total_runs": 1,
            "recurrent_failures": {}
        }

        result = compare_ht_history(old_history, new_history)

        assert result["status"] == "REGRESSED"

    def test_improved_by_health_score_tiebreaker(self):
        """IMPROVED when same status but better health score."""
        from backend.ht.ht_replay_verifier import compare_ht_history

        old_history = {
            "run_entries": [{"ht_replay_status": "OK"}],
            "ok_count": 1,
            "warn_count": 0,
            "fail_count": 1,  # health_score = 0
            "total_runs": 2,
            "recurrent_failures": {}
        }

        new_history = {
            "run_entries": [{"ht_replay_status": "OK"}],
            "ok_count": 2,
            "warn_count": 0,
            "fail_count": 0,  # health_score = 2
            "total_runs": 2,
            "recurrent_failures": {}
        }

        result = compare_ht_history(old_history, new_history)

        assert result["status"] == "IMPROVED"
        assert result["new_health_score"] > result["old_health_score"]

    def test_comparison_details_included(self):
        """Comparison result includes detailed metrics."""
        from backend.ht.ht_replay_verifier import compare_ht_history

        old_history = {
            "run_entries": [{"ht_replay_status": "OK"}],
            "ok_count": 1,
            "warn_count": 0,
            "fail_count": 0,
            "total_runs": 1,
            "recurrent_failures": {}
        }

        new_history = {
            "run_entries": [{"ht_replay_status": "FAIL"}],
            "ok_count": 0,
            "warn_count": 0,
            "fail_count": 1,
            "total_runs": 1,
            "recurrent_failures": {"INV-REPLAY-HT-1": 2}
        }

        result = compare_ht_history(old_history, new_history)

        assert "details" in result
        assert result["details"]["latest_status_changed"] is True
        assert result["details"]["old_total_runs"] == 1
        assert result["details"]["new_total_runs"] == 1
        assert "INV-REPLAY-HT-1" in result["details"]["new_recurrent_failures"]


# ==============================================================================
# Phase III: Global Health Summary Tests
# ==============================================================================

class TestGlobalHealthSummary:
    """Tests for summarize_ht_for_global_health function."""

    def test_ht_ok_when_last_ok_no_recurrent(self):
        """ht_ok is True when last run OK and no recurrent critical failures."""
        from backend.ht.ht_replay_verifier import summarize_ht_for_global_health

        history = {
            "run_entries": [
                {"ht_replay_status": "OK", "replay_id": "r1", "all_critical_pass": True}
            ],
            "ok_count": 1,
            "warn_count": 0,
            "fail_count": 0,
            "total_runs": 1,
            "recurrent_failures": {}
        }

        summary = summarize_ht_for_global_health(history)

        assert summary["ht_ok"] is True
        assert summary["last_status"] == "OK"
        assert summary["health_score"] == 1

    def test_ht_not_ok_when_last_fail(self):
        """ht_ok is False when last run is FAIL."""
        from backend.ht.ht_replay_verifier import summarize_ht_for_global_health

        history = {
            "run_entries": [
                {"ht_replay_status": "FAIL", "replay_id": "r1", "all_critical_pass": False}
            ],
            "ok_count": 0,
            "warn_count": 0,
            "fail_count": 1,
            "total_runs": 1,
            "recurrent_failures": {}
        }

        summary = summarize_ht_for_global_health(history)

        assert summary["ht_ok"] is False
        assert summary["last_status"] == "FAIL"

    def test_ht_not_ok_when_recurrent_critical(self):
        """ht_ok is False when recurrent critical failures exist."""
        from backend.ht.ht_replay_verifier import summarize_ht_for_global_health

        history = {
            "run_entries": [
                {"ht_replay_status": "OK", "replay_id": "r1", "all_critical_pass": True}
            ],
            "ok_count": 1,
            "warn_count": 0,
            "fail_count": 0,
            "total_runs": 1,
            "recurrent_failures": {"INV-REPLAY-HT-1": 2}  # Critical invariant
        }

        summary = summarize_ht_for_global_health(history)

        assert summary["ht_ok"] is False
        assert "INV-REPLAY-HT-1" in summary["recurrent_critical_failures"]

    def test_ht_ok_when_recurrent_high_only(self):
        """ht_ok is True when only high severity invariants fail recurrently."""
        from backend.ht.ht_replay_verifier import summarize_ht_for_global_health

        history = {
            "run_entries": [
                {"ht_replay_status": "OK", "replay_id": "r1", "all_critical_pass": True}
            ],
            "ok_count": 1,
            "warn_count": 0,
            "fail_count": 0,
            "total_runs": 1,
            "recurrent_failures": {"INV-REPLAY-HT-4": 2}  # High severity only
        }

        summary = summarize_ht_for_global_health(history)

        assert summary["ht_ok"] is True  # Still OK because not critical
        assert summary["recurrent_critical_failures"] == []

    def test_critical_fail_runs_identified(self):
        """Critical fail runs are correctly identified."""
        from backend.ht.ht_replay_verifier import summarize_ht_for_global_health

        history = {
            "run_entries": [
                {"ht_replay_status": "OK", "replay_id": "r1", "all_critical_pass": True},
                {"ht_replay_status": "FAIL", "replay_id": "r2", "all_critical_pass": False},
                {"ht_replay_status": "OK", "replay_id": "r3", "all_critical_pass": True},
                {"ht_replay_status": "FAIL", "replay_id": "r4", "all_critical_pass": False},
            ],
            "ok_count": 2,
            "warn_count": 0,
            "fail_count": 2,
            "total_runs": 4,
            "recurrent_failures": {}
        }

        summary = summarize_ht_for_global_health(history)

        assert summary["critical_fail_runs"] == ["r2", "r4"]
        assert summary["total_runs"] == 4

    def test_empty_history_returns_unknown(self):
        """Empty history returns UNKNOWN status."""
        from backend.ht.ht_replay_verifier import summarize_ht_for_global_health

        history = {
            "run_entries": [],
            "ok_count": 0,
            "warn_count": 0,
            "fail_count": 0,
            "total_runs": 0,
            "recurrent_failures": {}
        }

        summary = summarize_ht_for_global_health(history)

        assert summary["last_status"] == "UNKNOWN"
        assert summary["ht_ok"] is False  # Can't be OK with unknown status
        assert summary["health_score"] == 0

    def test_summary_contains_all_required_fields(self):
        """Summary contains all fields required for global health monitoring."""
        from backend.ht.ht_replay_verifier import summarize_ht_for_global_health

        history = {
            "run_entries": [
                {"ht_replay_status": "OK", "replay_id": "r1", "all_critical_pass": True}
            ],
            "ok_count": 1,
            "warn_count": 0,
            "fail_count": 0,
            "total_runs": 1,
            "recurrent_failures": {}
        }

        summary = summarize_ht_for_global_health(history)

        required_fields = [
            "ht_ok",
            "last_status",
            "critical_fail_runs",
            "health_score",
            "recurrent_critical_failures",
            "total_runs",
            "ok_count",
            "warn_count",
            "fail_count"
        ]

        for field in required_fields:
            assert field in summary, f"Missing required field: {field}"


# ==============================================================================
# Phase IV: Hₜ Replay Release Evaluator Tests
# ==============================================================================

class TestReleaseEvaluator:
    """Tests for evaluate_ht_replay_for_release function."""

    def test_release_ok_when_all_healthy(self):
        """Release is OK when all runs pass and drift is stable."""
        from backend.ht.ht_replay_verifier import evaluate_ht_replay_for_release

        history = {
            "run_entries": [{"ht_replay_status": "OK"}],
            "ok_count": 1,
            "warn_count": 0,
            "fail_count": 0,
            "total_runs": 1,
            "recurrent_failures": {}
        }

        drift = {
            "status": "STABLE",
            "details": {"latest_status_changed": False}
        }

        result = evaluate_ht_replay_for_release(history, drift)

        assert result["release_ok"] is True
        assert result["status"] == "OK"
        assert result["blocking_reasons"] == []

    def test_release_blocked_when_last_fail(self):
        """Release is blocked when last run is FAIL."""
        from backend.ht.ht_replay_verifier import evaluate_ht_replay_for_release

        history = {
            "run_entries": [{"ht_replay_status": "FAIL"}],
            "ok_count": 0,
            "warn_count": 0,
            "fail_count": 1,
            "total_runs": 1,
            "recurrent_failures": {}
        }

        drift = {
            "status": "STABLE",
            "details": {"latest_status_changed": False}
        }

        result = evaluate_ht_replay_for_release(history, drift)

        assert result["release_ok"] is False
        assert result["status"] == "BLOCK"
        assert any("FAIL" in r for r in result["blocking_reasons"])

    def test_release_blocked_when_regressed(self):
        """Release is blocked when drift is REGRESSED."""
        from backend.ht.ht_replay_verifier import evaluate_ht_replay_for_release

        history = {
            "run_entries": [{"ht_replay_status": "OK"}],
            "ok_count": 1,
            "warn_count": 0,
            "fail_count": 0,
            "total_runs": 1,
            "recurrent_failures": {}
        }

        drift = {
            "status": "REGRESSED",
            "details": {"latest_status_changed": True}
        }

        result = evaluate_ht_replay_for_release(history, drift)

        assert result["release_ok"] is False
        assert result["status"] == "BLOCK"
        assert any("REGRESSED" in r for r in result["blocking_reasons"])

    def test_release_blocked_when_recurrent_critical(self):
        """Release is blocked when recurrent critical failures exist."""
        from backend.ht.ht_replay_verifier import evaluate_ht_replay_for_release

        history = {
            "run_entries": [{"ht_replay_status": "OK"}],
            "ok_count": 1,
            "warn_count": 0,
            "fail_count": 0,
            "total_runs": 1,
            "recurrent_failures": {"INV-REPLAY-HT-1": 2}  # Critical
        }

        drift = {
            "status": "STABLE",
            "details": {"latest_status_changed": False}
        }

        result = evaluate_ht_replay_for_release(history, drift)

        assert result["release_ok"] is False
        assert result["status"] == "BLOCK"
        assert any("Recurrent critical" in r for r in result["blocking_reasons"])

    def test_release_blocked_when_high_fail_rate(self):
        """Release is blocked when fail rate exceeds 50%."""
        from backend.ht.ht_replay_verifier import evaluate_ht_replay_for_release

        history = {
            "run_entries": [
                {"ht_replay_status": "FAIL"},
                {"ht_replay_status": "FAIL"},
                {"ht_replay_status": "OK"}
            ],
            "ok_count": 1,
            "warn_count": 0,
            "fail_count": 2,
            "total_runs": 3,
            "recurrent_failures": {}
        }

        drift = {
            "status": "STABLE",
            "details": {"latest_status_changed": False}
        }

        result = evaluate_ht_replay_for_release(history, drift)

        assert result["release_ok"] is False
        assert result["status"] == "BLOCK"
        assert any("fail rate" in r.lower() for r in result["blocking_reasons"])

    def test_release_warn_when_last_warn(self):
        """Release has WARN status when last run is WARN."""
        from backend.ht.ht_replay_verifier import evaluate_ht_replay_for_release

        history = {
            "run_entries": [{"ht_replay_status": "WARN"}],
            "ok_count": 0,
            "warn_count": 1,
            "fail_count": 0,
            "total_runs": 1,
            "recurrent_failures": {}
        }

        drift = {
            "status": "STABLE",
            "details": {"latest_status_changed": False}
        }

        result = evaluate_ht_replay_for_release(history, drift)

        assert result["release_ok"] is True  # Warnings don't block
        assert result["status"] == "WARN"
        assert any("WARN" in r for r in result["warning_reasons"])

    def test_release_warn_when_recurrent_high(self):
        """Release has WARN status when recurrent high-severity failures."""
        from backend.ht.ht_replay_verifier import evaluate_ht_replay_for_release

        history = {
            "run_entries": [{"ht_replay_status": "OK"}],
            "ok_count": 1,
            "warn_count": 0,
            "fail_count": 0,
            "total_runs": 1,
            "recurrent_failures": {"INV-REPLAY-HT-4": 2}  # High severity
        }

        drift = {
            "status": "STABLE",
            "details": {"latest_status_changed": False}
        }

        result = evaluate_ht_replay_for_release(history, drift)

        assert result["release_ok"] is True
        assert result["status"] == "WARN"
        assert any("high-severity" in r.lower() for r in result["warning_reasons"])


# ==============================================================================
# Phase IV: MAAS Hₜ Summary Tests
# ==============================================================================

class TestMaasSummary:
    """Tests for summarize_ht_replay_for_maas function."""

    def test_maas_ok_when_healthy(self):
        """MAAS status is OK when all healthy."""
        from backend.ht.ht_replay_verifier import summarize_ht_replay_for_maas

        history = {
            "run_entries": [{"ht_replay_status": "OK"}],
            "ok_count": 1,
            "warn_count": 0,
            "fail_count": 0,
            "total_runs": 1,
            "recurrent_failures": {}
        }

        drift = {"status": "STABLE"}

        result = summarize_ht_replay_for_maas(history, drift)

        assert result["ht_replay_admissible"] is True
        assert result["status"] == "OK"
        assert result["recurrent_critical_failures"] == []

    def test_maas_block_when_fail(self):
        """MAAS status is BLOCK when last run is FAIL."""
        from backend.ht.ht_replay_verifier import summarize_ht_replay_for_maas

        history = {
            "run_entries": [{"ht_replay_status": "FAIL"}],
            "ok_count": 0,
            "warn_count": 0,
            "fail_count": 1,
            "total_runs": 1,
            "recurrent_failures": {}
        }

        drift = {"status": "STABLE"}

        result = summarize_ht_replay_for_maas(history, drift)

        assert result["ht_replay_admissible"] is False
        assert result["status"] == "BLOCK"

    def test_maas_block_when_regressed(self):
        """MAAS status is BLOCK when drift is REGRESSED."""
        from backend.ht.ht_replay_verifier import summarize_ht_replay_for_maas

        history = {
            "run_entries": [{"ht_replay_status": "OK"}],
            "ok_count": 1,
            "warn_count": 0,
            "fail_count": 0,
            "total_runs": 1,
            "recurrent_failures": {}
        }

        drift = {"status": "REGRESSED"}

        result = summarize_ht_replay_for_maas(history, drift)

        assert result["ht_replay_admissible"] is False
        assert result["status"] == "BLOCK"

    def test_maas_block_when_recurrent_critical(self):
        """MAAS status is BLOCK when recurrent critical failures."""
        from backend.ht.ht_replay_verifier import summarize_ht_replay_for_maas

        history = {
            "run_entries": [{"ht_replay_status": "OK"}],
            "ok_count": 1,
            "warn_count": 0,
            "fail_count": 0,
            "total_runs": 1,
            "recurrent_failures": {"INV-REPLAY-HT-2": 3}
        }

        drift = {"status": "STABLE"}

        result = summarize_ht_replay_for_maas(history, drift)

        assert result["ht_replay_admissible"] is False
        assert result["status"] == "BLOCK"
        assert "INV-REPLAY-HT-2" in result["recurrent_critical_failures"]

    def test_maas_attention_when_warn(self):
        """MAAS status is ATTENTION when last run is WARN."""
        from backend.ht.ht_replay_verifier import summarize_ht_replay_for_maas

        history = {
            "run_entries": [{"ht_replay_status": "WARN"}],
            "ok_count": 0,
            "warn_count": 1,
            "fail_count": 0,
            "total_runs": 1,
            "recurrent_failures": {}
        }

        drift = {"status": "STABLE"}

        result = summarize_ht_replay_for_maas(history, drift)

        assert result["ht_replay_admissible"] is True
        assert result["status"] == "ATTENTION"

    def test_maas_attention_when_recurrent_high(self):
        """MAAS status is ATTENTION when recurrent high-severity failures."""
        from backend.ht.ht_replay_verifier import summarize_ht_replay_for_maas

        history = {
            "run_entries": [{"ht_replay_status": "OK"}],
            "ok_count": 1,
            "warn_count": 0,
            "fail_count": 0,
            "total_runs": 1,
            "recurrent_failures": {"INV-REPLAY-HT-5": 2}  # High severity
        }

        drift = {"status": "STABLE"}

        result = summarize_ht_replay_for_maas(history, drift)

        assert result["ht_replay_admissible"] is True
        assert result["status"] == "ATTENTION"

    def test_maas_contains_required_fields(self):
        """MAAS summary contains all required fields."""
        from backend.ht.ht_replay_verifier import summarize_ht_replay_for_maas

        history = {
            "run_entries": [{"ht_replay_status": "OK"}],
            "ok_count": 1,
            "warn_count": 0,
            "fail_count": 0,
            "total_runs": 1,
            "recurrent_failures": {}
        }

        drift = {"status": "STABLE"}

        result = summarize_ht_replay_for_maas(history, drift)

        required_fields = [
            "ht_replay_admissible",
            "recurrent_critical_failures",
            "status",
            "drift_status",
            "last_status",
            "health_score"
        ]

        for field in required_fields:
            assert field in result, f"Missing required field: {field}"


# ==============================================================================
# Phase IV: Director Hₜ Panel Tests
# ==============================================================================

class TestDirectorPanel:
    """Tests for build_ht_replay_director_panel function."""

    def test_green_light_when_healthy(self):
        """Status light is GREEN when all healthy."""
        from backend.ht.ht_replay_verifier import build_ht_replay_director_panel

        global_health = {
            "ht_ok": True,
            "last_status": "OK",
            "critical_fail_runs": [],
            "health_score": 5,
            "recurrent_critical_failures": [],
            "total_runs": 5,
            "ok_count": 5,
            "fail_count": 0
        }

        drift = {
            "status": "STABLE",
            "details": {"latest_status_changed": False}
        }

        result = build_ht_replay_director_panel(global_health, drift)

        assert result["status_light"] == "GREEN"
        assert "healthy" in result["headline"].lower()

    def test_red_light_when_not_ok(self):
        """Status light is RED when ht_ok is False."""
        from backend.ht.ht_replay_verifier import build_ht_replay_director_panel

        global_health = {
            "ht_ok": False,
            "last_status": "FAIL",
            "critical_fail_runs": ["r1", "r2"],
            "health_score": -2,
            "recurrent_critical_failures": [],
            "total_runs": 3,
            "ok_count": 1,
            "fail_count": 2
        }

        drift = {
            "status": "STABLE",
            "details": {"latest_status_changed": False}
        }

        result = build_ht_replay_director_panel(global_health, drift)

        assert result["status_light"] == "RED"
        assert result["critical_fail_runs_count"] == 2

    def test_red_light_when_regressed(self):
        """Status light is RED when drift is REGRESSED."""
        from backend.ht.ht_replay_verifier import build_ht_replay_director_panel

        global_health = {
            "ht_ok": True,
            "last_status": "OK",
            "critical_fail_runs": [],
            "health_score": 1,
            "recurrent_critical_failures": [],
            "total_runs": 1,
            "ok_count": 1,
            "fail_count": 0
        }

        drift = {
            "status": "REGRESSED",
            "details": {"latest_status_changed": True}
        }

        result = build_ht_replay_director_panel(global_health, drift)

        assert result["status_light"] == "RED"
        assert "regressed" in result["headline"].lower()

    def test_yellow_light_when_warn(self):
        """Status light is YELLOW when last status is WARN."""
        from backend.ht.ht_replay_verifier import build_ht_replay_director_panel

        global_health = {
            "ht_ok": True,
            "last_status": "WARN",
            "critical_fail_runs": [],
            "health_score": 0,
            "recurrent_critical_failures": [],
            "total_runs": 1,
            "ok_count": 0,
            "fail_count": 0
        }

        drift = {
            "status": "STABLE",
            "details": {"latest_status_changed": False}
        }

        result = build_ht_replay_director_panel(global_health, drift)

        assert result["status_light"] == "YELLOW"
        assert "attention" in result["headline"].lower()

    def test_yellow_light_when_status_changed(self):
        """Status light is YELLOW when status changed."""
        from backend.ht.ht_replay_verifier import build_ht_replay_director_panel

        global_health = {
            "ht_ok": True,
            "last_status": "OK",
            "critical_fail_runs": [],
            "health_score": 1,
            "recurrent_critical_failures": [],
            "total_runs": 1,
            "ok_count": 1,
            "fail_count": 0
        }

        drift = {
            "status": "IMPROVED",
            "details": {"latest_status_changed": True}
        }

        result = build_ht_replay_director_panel(global_health, drift)

        assert result["status_light"] == "YELLOW"
        assert "changed" in result["headline"].lower() or "monitoring" in result["headline"].lower()

    def test_gray_light_when_no_data(self):
        """Status light is GRAY when no runs."""
        from backend.ht.ht_replay_verifier import build_ht_replay_director_panel

        global_health = {
            "ht_ok": False,
            "last_status": "UNKNOWN",
            "critical_fail_runs": [],
            "health_score": 0,
            "recurrent_critical_failures": [],
            "total_runs": 0,
            "ok_count": 0,
            "fail_count": 0
        }

        drift = {
            "status": "UNKNOWN",
            "details": {}
        }

        result = build_ht_replay_director_panel(global_health, drift)

        assert result["status_light"] == "GRAY"
        assert "no" in result["headline"].lower() and "data" in result["headline"].lower()

    def test_red_light_with_recurrent_critical(self):
        """Status light is RED when recurrent critical failures."""
        from backend.ht.ht_replay_verifier import build_ht_replay_director_panel

        global_health = {
            "ht_ok": False,
            "last_status": "OK",
            "critical_fail_runs": [],
            "health_score": 1,
            "recurrent_critical_failures": ["INV-REPLAY-HT-1"],
            "total_runs": 3,
            "ok_count": 3,
            "fail_count": 0
        }

        drift = {
            "status": "STABLE",
            "details": {"latest_status_changed": False}
        }

        result = build_ht_replay_director_panel(global_health, drift)

        assert result["status_light"] == "RED"
        assert "critical" in result["headline"].lower()

    def test_panel_contains_required_fields(self):
        """Director panel contains all required fields."""
        from backend.ht.ht_replay_verifier import build_ht_replay_director_panel

        global_health = {
            "ht_ok": True,
            "last_status": "OK",
            "critical_fail_runs": [],
            "health_score": 1,
            "recurrent_critical_failures": [],
            "total_runs": 1,
            "ok_count": 1,
            "fail_count": 0
        }

        drift = {
            "status": "STABLE",
            "details": {}
        }

        result = build_ht_replay_director_panel(global_health, drift)

        required_fields = [
            "status_light",
            "last_status",
            "critical_fail_runs_count",
            "headline",
            "health_score",
            "drift_status"
        ]

        for field in required_fields:
            assert field in result, f"Missing required field: {field}"


# ==============================================================================
# Phase V: Hₜ Replay Governance View Tests
# ==============================================================================

class TestGovernanceView:
    """Tests for build_ht_replay_governance_view function."""

    def test_aligned_when_both_ok(self):
        """ALIGNED when both HT and radar report OK."""
        from backend.ht.ht_replay_verifier import build_ht_replay_governance_view

        history = {
            "run_entries": [{"ht_replay_status": "OK"}],
            "ok_count": 1,
            "warn_count": 0,
            "fail_count": 0,
            "total_runs": 1,
            "recurrent_failures": {}
        }

        drift = {"status": "STABLE"}
        replay_radar = {"status": "OK"}

        result = build_ht_replay_governance_view(history, drift, replay_radar)

        assert result["alignment_status"] == "ALIGNED"
        assert result["ht_ok"] is True
        assert result["blocking_fingerprints"] == []

    def test_aligned_when_both_block(self):
        """ALIGNED when both HT and radar report BLOCK/FAIL."""
        from backend.ht.ht_replay_verifier import build_ht_replay_governance_view

        history = {
            "run_entries": [{"ht_replay_status": "FAIL"}],
            "ok_count": 0,
            "warn_count": 0,
            "fail_count": 1,
            "total_runs": 1,
            "recurrent_failures": {}
        }

        drift = {"status": "STABLE"}
        replay_radar = {"status": "BLOCK"}

        result = build_ht_replay_governance_view(history, drift, replay_radar)

        assert result["alignment_status"] == "ALIGNED"

    def test_aligned_when_both_warn(self):
        """ALIGNED when both HT and radar report WARN."""
        from backend.ht.ht_replay_verifier import build_ht_replay_governance_view

        history = {
            "run_entries": [{"ht_replay_status": "WARN"}],
            "ok_count": 0,
            "warn_count": 1,
            "fail_count": 0,
            "total_runs": 1,
            "recurrent_failures": {}
        }

        drift = {"status": "STABLE"}
        replay_radar = {"status": "WARN"}

        result = build_ht_replay_governance_view(history, drift, replay_radar)

        assert result["alignment_status"] == "ALIGNED"

    def test_divergent_ht_block_vs_radar_ok(self):
        """DIVERGENT when HT reports BLOCK and radar reports OK."""
        from backend.ht.ht_replay_verifier import build_ht_replay_governance_view

        history = {
            "run_entries": [{"ht_replay_status": "FAIL"}],
            "ok_count": 0,
            "warn_count": 0,
            "fail_count": 1,
            "total_runs": 1,
            "recurrent_failures": {}
        }

        drift = {"status": "STABLE"}
        replay_radar = {"status": "OK"}

        result = build_ht_replay_governance_view(history, drift, replay_radar)

        assert result["alignment_status"] == "DIVERGENT"
        assert result["ht_ok"] is False
        assert any("BLOCK" in fp and "OK" in fp for fp in result["blocking_fingerprints"])

    def test_divergent_ht_ok_vs_radar_block(self):
        """DIVERGENT when HT reports OK and radar reports BLOCK."""
        from backend.ht.ht_replay_verifier import build_ht_replay_governance_view

        history = {
            "run_entries": [{"ht_replay_status": "OK"}],
            "ok_count": 1,
            "warn_count": 0,
            "fail_count": 0,
            "total_runs": 1,
            "recurrent_failures": {}
        }

        drift = {"status": "STABLE"}
        replay_radar = {"status": "BLOCK"}

        result = build_ht_replay_governance_view(history, drift, replay_radar)

        assert result["alignment_status"] == "DIVERGENT"
        assert any("Radar reports BLOCK" in fp for fp in result["blocking_fingerprints"])

    def test_divergent_ht_ok_vs_radar_fail(self):
        """DIVERGENT when HT reports OK and radar reports FAIL (normalized to BLOCK)."""
        from backend.ht.ht_replay_verifier import build_ht_replay_governance_view

        history = {
            "run_entries": [{"ht_replay_status": "OK"}],
            "ok_count": 1,
            "warn_count": 0,
            "fail_count": 0,
            "total_runs": 1,
            "recurrent_failures": {}
        }

        drift = {"status": "STABLE"}
        replay_radar = {"status": "FAIL"}  # Normalized to BLOCK

        result = build_ht_replay_governance_view(history, drift, replay_radar)

        assert result["alignment_status"] == "DIVERGENT"

    def test_tension_ht_warn_vs_radar_ok(self):
        """TENSION when HT reports WARN and radar reports OK."""
        from backend.ht.ht_replay_verifier import build_ht_replay_governance_view

        history = {
            "run_entries": [{"ht_replay_status": "WARN"}],
            "ok_count": 0,
            "warn_count": 1,
            "fail_count": 0,
            "total_runs": 1,
            "recurrent_failures": {}
        }

        drift = {"status": "STABLE"}
        replay_radar = {"status": "OK"}

        result = build_ht_replay_governance_view(history, drift, replay_radar)

        assert result["alignment_status"] == "TENSION"

    def test_tension_ht_ok_vs_radar_warn(self):
        """TENSION when HT reports OK and radar reports WARN."""
        from backend.ht.ht_replay_verifier import build_ht_replay_governance_view

        history = {
            "run_entries": [{"ht_replay_status": "OK"}],
            "ok_count": 1,
            "warn_count": 0,
            "fail_count": 0,
            "total_runs": 1,
            "recurrent_failures": {}
        }

        drift = {"status": "STABLE"}
        replay_radar = {"status": "WARN"}

        result = build_ht_replay_governance_view(history, drift, replay_radar)

        assert result["alignment_status"] == "TENSION"

    def test_tension_ht_attention_vs_radar_ok(self):
        """TENSION when HT reports ATTENTION (normalized to WARN) and radar reports OK."""
        from backend.ht.ht_replay_verifier import build_ht_replay_governance_view

        history = {
            "run_entries": [{"ht_replay_status": "OK"}],
            "ok_count": 1,
            "warn_count": 0,
            "fail_count": 0,
            "total_runs": 1,
            "recurrent_failures": {"INV-REPLAY-HT-4": 2}  # High severity causes ATTENTION
        }

        drift = {"status": "STABLE"}
        replay_radar = {"status": "OK"}

        result = build_ht_replay_governance_view(history, drift, replay_radar)

        assert result["alignment_status"] == "TENSION"

    def test_divergent_includes_critical_failure_details(self):
        """DIVERGENT includes critical failure details in blocking_fingerprints."""
        from backend.ht.ht_replay_verifier import build_ht_replay_governance_view

        history = {
            "run_entries": [{"ht_replay_status": "FAIL"}],
            "ok_count": 0,
            "warn_count": 0,
            "fail_count": 1,
            "total_runs": 1,
            "recurrent_failures": {"INV-REPLAY-HT-1": 2}  # Critical failure
        }

        drift = {"status": "STABLE"}
        replay_radar = {"status": "OK"}

        result = build_ht_replay_governance_view(history, drift, replay_radar)

        assert result["alignment_status"] == "DIVERGENT"
        # Should include critical failure details
        assert any("INV-REPLAY-HT-1" in fp for fp in result["blocking_fingerprints"])

    def test_divergent_includes_regressed_drift(self):
        """DIVERGENT includes REGRESSED drift in blocking_fingerprints."""
        from backend.ht.ht_replay_verifier import build_ht_replay_governance_view

        history = {
            "run_entries": [{"ht_replay_status": "FAIL"}],
            "ok_count": 0,
            "warn_count": 0,
            "fail_count": 1,
            "total_runs": 1,
            "recurrent_failures": {}
        }

        drift = {"status": "REGRESSED"}  # This should trigger BLOCK in HT
        replay_radar = {"status": "OK"}

        result = build_ht_replay_governance_view(history, drift, replay_radar)

        assert result["alignment_status"] == "DIVERGENT"
        assert any("REGRESSED" in fp for fp in result["blocking_fingerprints"])

    def test_view_contains_required_fields(self):
        """Governance view contains all required fields."""
        from backend.ht.ht_replay_verifier import build_ht_replay_governance_view

        history = {
            "run_entries": [{"ht_replay_status": "OK"}],
            "ok_count": 1,
            "warn_count": 0,
            "fail_count": 0,
            "total_runs": 1,
            "recurrent_failures": {}
        }

        drift = {"status": "STABLE"}
        replay_radar = {"status": "OK"}

        result = build_ht_replay_governance_view(history, drift, replay_radar)

        required_fields = [
            "ht_ok",
            "replay_radar_status",
            "alignment_status",
            "blocking_fingerprints",
            "ht_status",
            "drift_status"
        ]

        for field in required_fields:
            assert field in result, f"Missing required field: {field}"


# ==============================================================================
# Phase V: Global Console Adapter Tests
# ==============================================================================

class TestGlobalConsoleAdapter:
    """Tests for summarize_ht_for_global_console function."""

    def test_green_light_when_aligned_ok(self):
        """GREEN light when aligned and HT OK."""
        from backend.ht.ht_replay_verifier import summarize_ht_for_global_console

        global_health = {
            "ht_ok": True,
            "last_status": "OK",
            "critical_fail_runs": [],
            "health_score": 5,
            "recurrent_critical_failures": [],
            "total_runs": 5,
            "ok_count": 5,
            "fail_count": 0
        }

        drift = {"status": "STABLE", "details": {"latest_status_changed": False}}

        governance_view = {
            "ht_ok": True,
            "replay_radar_status": "OK",
            "alignment_status": "ALIGNED",
            "blocking_fingerprints": [],
            "ht_status": "OK",
            "drift_status": "STABLE"
        }

        result = summarize_ht_for_global_console(global_health, drift, governance_view)

        assert result["status_light"] == "GREEN"
        assert result["ht_ok"] is True
        assert result["alignment_status"] == "ALIGNED"

    def test_red_light_when_divergent(self):
        """RED light when alignment is DIVERGENT."""
        from backend.ht.ht_replay_verifier import summarize_ht_for_global_console

        global_health = {
            "ht_ok": False,
            "last_status": "FAIL",
            "critical_fail_runs": ["r1"],
            "health_score": -1,
            "recurrent_critical_failures": [],
            "total_runs": 1,
            "ok_count": 0,
            "fail_count": 1
        }

        drift = {"status": "STABLE", "details": {"latest_status_changed": False}}

        governance_view = {
            "ht_ok": False,
            "replay_radar_status": "OK",
            "alignment_status": "DIVERGENT",
            "blocking_fingerprints": ["HT reports BLOCK while radar reports OK"],
            "ht_status": "BLOCK",
            "drift_status": "STABLE"
        }

        result = summarize_ht_for_global_console(global_health, drift, governance_view)

        assert result["status_light"] == "RED"
        assert result["alignment_status"] == "DIVERGENT"
        assert "divergence" in result["headline"].lower()

    def test_headline_updated_for_tension(self):
        """Headline includes tension note when TENSION."""
        from backend.ht.ht_replay_verifier import summarize_ht_for_global_console

        global_health = {
            "ht_ok": True,
            "last_status": "OK",
            "critical_fail_runs": [],
            "health_score": 1,
            "recurrent_critical_failures": [],
            "total_runs": 1,
            "ok_count": 1,
            "fail_count": 0
        }

        drift = {"status": "STABLE", "details": {"latest_status_changed": False}}

        governance_view = {
            "ht_ok": True,
            "replay_radar_status": "WARN",
            "alignment_status": "TENSION",
            "blocking_fingerprints": ["Radar reports WARN while HT reports OK"],
            "ht_status": "OK",
            "drift_status": "STABLE"
        }

        result = summarize_ht_for_global_console(global_health, drift, governance_view)

        assert "tension" in result["headline"].lower()

    def test_divergent_overrides_green_to_red(self):
        """DIVERGENT alignment overrides GREEN status to RED."""
        from backend.ht.ht_replay_verifier import summarize_ht_for_global_console

        # HT itself is OK (would be GREEN)
        global_health = {
            "ht_ok": True,
            "last_status": "OK",
            "critical_fail_runs": [],
            "health_score": 5,
            "recurrent_critical_failures": [],
            "total_runs": 5,
            "ok_count": 5,
            "fail_count": 0
        }

        drift = {"status": "STABLE", "details": {"latest_status_changed": False}}

        # But radar reports BLOCK, causing DIVERGENT
        governance_view = {
            "ht_ok": True,
            "replay_radar_status": "BLOCK",
            "alignment_status": "DIVERGENT",
            "blocking_fingerprints": ["Radar reports BLOCK while HT reports OK"],
            "ht_status": "OK",
            "drift_status": "STABLE"
        }

        result = summarize_ht_for_global_console(global_health, drift, governance_view)

        assert result["status_light"] == "RED"  # Overridden due to divergence

    def test_console_contains_required_fields(self):
        """Console summary contains all required fields."""
        from backend.ht.ht_replay_verifier import summarize_ht_for_global_console

        global_health = {
            "ht_ok": True,
            "last_status": "OK",
            "critical_fail_runs": [],
            "health_score": 1,
            "recurrent_critical_failures": [],
            "total_runs": 1,
            "ok_count": 1,
            "fail_count": 0
        }

        drift = {"status": "STABLE", "details": {}}

        governance_view = {
            "ht_ok": True,
            "replay_radar_status": "OK",
            "alignment_status": "ALIGNED",
            "blocking_fingerprints": [],
            "ht_status": "OK",
            "drift_status": "STABLE"
        }

        result = summarize_ht_for_global_console(global_health, drift, governance_view)

        required_fields = [
            "status_light",
            "ht_ok",
            "drift_status",
            "headline",
            "alignment_status",
            "replay_radar_status",
            "blocking_fingerprints"
        ]

        for field in required_fields:
            assert field in result, f"Missing required field: {field}"


# ==============================================================================
# Phase V: Evidence Pack HT Tile Tests
# ==============================================================================

class TestEvidencePackTile:
    """Tests for build_ht_evidence_pack_tile function."""

    def test_ok_tile_when_healthy(self):
        """ht_replay_ok is True when history is healthy."""
        from backend.ht.ht_replay_verifier import build_ht_evidence_pack_tile

        history = {
            "run_entries": [
                {"ht_replay_status": "OK", "replay_id": "r1", "all_critical_pass": True}
            ],
            "ok_count": 1,
            "warn_count": 0,
            "fail_count": 0,
            "total_runs": 1,
            "recurrent_failures": {}
        }

        result = build_ht_evidence_pack_tile(history)

        assert result["ht_replay_ok"] is True
        assert result["critical_fail_runs"] == []
        assert result["recurrent_critical_invariants"] == []
        assert result["last_status"] == "OK"

    def test_not_ok_when_fail(self):
        """ht_replay_ok is False when last run is FAIL."""
        from backend.ht.ht_replay_verifier import build_ht_evidence_pack_tile

        history = {
            "run_entries": [
                {"ht_replay_status": "FAIL", "replay_id": "r1", "all_critical_pass": False}
            ],
            "ok_count": 0,
            "warn_count": 0,
            "fail_count": 1,
            "total_runs": 1,
            "recurrent_failures": {}
        }

        result = build_ht_evidence_pack_tile(history)

        assert result["ht_replay_ok"] is False
        assert result["last_status"] == "FAIL"

    def test_critical_fail_runs_tracked(self):
        """Critical fail runs are tracked in tile."""
        from backend.ht.ht_replay_verifier import build_ht_evidence_pack_tile

        history = {
            "run_entries": [
                {"ht_replay_status": "OK", "replay_id": "r1", "all_critical_pass": True},
                {"ht_replay_status": "FAIL", "replay_id": "r2", "all_critical_pass": False},
                {"ht_replay_status": "FAIL", "replay_id": "r3", "all_critical_pass": False},
            ],
            "ok_count": 1,
            "warn_count": 0,
            "fail_count": 2,
            "total_runs": 3,
            "recurrent_failures": {}
        }

        result = build_ht_evidence_pack_tile(history)

        assert "r2" in result["critical_fail_runs"]
        assert "r3" in result["critical_fail_runs"]
        assert "r1" not in result["critical_fail_runs"]

    def test_recurrent_critical_invariants_tracked(self):
        """Recurrent critical invariants are tracked in tile."""
        from backend.ht.ht_replay_verifier import build_ht_evidence_pack_tile

        history = {
            "run_entries": [
                {"ht_replay_status": "OK", "replay_id": "r1", "all_critical_pass": True}
            ],
            "ok_count": 1,
            "warn_count": 0,
            "fail_count": 0,
            "total_runs": 1,
            "recurrent_failures": {
                "INV-REPLAY-HT-1": 3,  # Critical
                "INV-REPLAY-HT-4": 2   # High severity (not critical)
            }
        }

        result = build_ht_evidence_pack_tile(history)

        # Only critical invariants should be in recurrent_critical_invariants
        assert "INV-REPLAY-HT-1" in result["recurrent_critical_invariants"]
        assert "INV-REPLAY-HT-4" not in result["recurrent_critical_invariants"]

    def test_not_ok_when_recurrent_critical(self):
        """ht_replay_ok is False when recurrent critical failures exist."""
        from backend.ht.ht_replay_verifier import build_ht_evidence_pack_tile

        history = {
            "run_entries": [
                {"ht_replay_status": "OK", "replay_id": "r1", "all_critical_pass": True}
            ],
            "ok_count": 1,
            "warn_count": 0,
            "fail_count": 0,
            "total_runs": 1,
            "recurrent_failures": {"INV-REPLAY-HT-2": 2}  # Critical
        }

        result = build_ht_evidence_pack_tile(history)

        assert result["ht_replay_ok"] is False

    def test_drift_status_included_when_provided(self):
        """Drift status is included when drift is provided."""
        from backend.ht.ht_replay_verifier import build_ht_evidence_pack_tile

        history = {
            "run_entries": [
                {"ht_replay_status": "OK", "replay_id": "r1", "all_critical_pass": True}
            ],
            "ok_count": 1,
            "warn_count": 0,
            "fail_count": 0,
            "total_runs": 1,
            "recurrent_failures": {}
        }

        drift = {"status": "REGRESSED"}

        result = build_ht_evidence_pack_tile(history, drift)

        assert result["drift_status"] == "REGRESSED"

    def test_drift_status_not_evaluated_when_omitted(self):
        """Drift status is NOT_EVALUATED when drift is None."""
        from backend.ht.ht_replay_verifier import build_ht_evidence_pack_tile

        history = {
            "run_entries": [
                {"ht_replay_status": "OK", "replay_id": "r1", "all_critical_pass": True}
            ],
            "ok_count": 1,
            "warn_count": 0,
            "fail_count": 0,
            "total_runs": 1,
            "recurrent_failures": {}
        }

        result = build_ht_evidence_pack_tile(history, None)

        assert result["drift_status"] == "NOT_EVALUATED"

    def test_tile_is_json_serializable(self):
        """Tile is JSON serializable."""
        from backend.ht.ht_replay_verifier import build_ht_evidence_pack_tile
        import json

        history = {
            "run_entries": [
                {"ht_replay_status": "OK", "replay_id": "r1", "all_critical_pass": True}
            ],
            "ok_count": 1,
            "warn_count": 0,
            "fail_count": 0,
            "total_runs": 1,
            "recurrent_failures": {}
        }

        result = build_ht_evidence_pack_tile(history)

        # Should not raise
        json_str = json.dumps(result)
        assert isinstance(json_str, str)

    def test_tile_contains_required_fields(self):
        """Evidence pack tile contains all required fields."""
        from backend.ht.ht_replay_verifier import build_ht_evidence_pack_tile

        history = {
            "run_entries": [
                {"ht_replay_status": "OK", "replay_id": "r1", "all_critical_pass": True}
            ],
            "ok_count": 1,
            "warn_count": 0,
            "fail_count": 0,
            "total_runs": 1,
            "recurrent_failures": {}
        }

        result = build_ht_evidence_pack_tile(history)

        required_fields = [
            "ht_replay_ok",
            "critical_fail_runs",
            "recurrent_critical_invariants",
            "last_status",
            "health_score",
            "total_runs",
            "drift_status",
            "generated_utc"
        ]

        for field in required_fields:
            assert field in result, f"Missing required field: {field}"

    def test_generated_utc_is_iso_format(self):
        """generated_utc is in ISO format."""
        from backend.ht.ht_replay_verifier import build_ht_evidence_pack_tile

        history = {
            "run_entries": [],
            "ok_count": 0,
            "warn_count": 0,
            "fail_count": 0,
            "total_runs": 0,
            "recurrent_failures": {}
        }

        result = build_ht_evidence_pack_tile(history)

        # Should end with Z and be parseable
        assert result["generated_utc"].endswith("Z")
        from datetime import datetime
        # Should not raise
        datetime.fromisoformat(result["generated_utc"].replace("Z", "+00:00"))


# ==============================================================================
# Phase V: Governance Signal Layer Tests
# ==============================================================================

class TestGovernanceSignal:
    """Tests for to_governance_signal_for_ht function."""

    def test_aligned_maps_to_ok(self):
        """ALIGNED alignment_status maps to OK signal."""
        from backend.ht.ht_replay_verifier import to_governance_signal_for_ht

        governance_view = {
            "ht_ok": True,
            "replay_radar_status": "OK",
            "alignment_status": "ALIGNED",
            "blocking_fingerprints": [],
            "ht_status": "OK",
            "drift_status": "STABLE"
        }

        result = to_governance_signal_for_ht(governance_view)

        assert result["signal"] == "OK"
        assert result["source"] == "ht_replay"
        assert result["alignment_status"] == "ALIGNED"

    def test_tension_maps_to_warn(self):
        """TENSION alignment_status maps to WARN signal."""
        from backend.ht.ht_replay_verifier import to_governance_signal_for_ht

        governance_view = {
            "ht_ok": True,
            "replay_radar_status": "WARN",
            "alignment_status": "TENSION",
            "blocking_fingerprints": ["Radar reports WARN while HT reports OK"],
            "ht_status": "OK",
            "drift_status": "STABLE"
        }

        result = to_governance_signal_for_ht(governance_view)

        assert result["signal"] == "WARN"
        assert result["alignment_status"] == "TENSION"

    def test_divergent_maps_to_block(self):
        """DIVERGENT alignment_status maps to BLOCK signal."""
        from backend.ht.ht_replay_verifier import to_governance_signal_for_ht

        governance_view = {
            "ht_ok": False,
            "replay_radar_status": "OK",
            "alignment_status": "DIVERGENT",
            "blocking_fingerprints": ["HT reports BLOCK while radar reports OK"],
            "ht_status": "BLOCK",
            "drift_status": "STABLE"
        }

        result = to_governance_signal_for_ht(governance_view)

        assert result["signal"] == "BLOCK"
        assert result["alignment_status"] == "DIVERGENT"

    def test_unknown_alignment_defaults_to_ok(self):
        """Unknown alignment_status defaults to OK signal."""
        from backend.ht.ht_replay_verifier import to_governance_signal_for_ht

        governance_view = {
            "ht_ok": True,
            "replay_radar_status": "OK",
            "alignment_status": "UNKNOWN_VALUE",
            "blocking_fingerprints": [],
            "ht_status": "OK",
            "drift_status": "STABLE"
        }

        result = to_governance_signal_for_ht(governance_view)

        assert result["signal"] == "OK"

    def test_missing_alignment_defaults_to_ok(self):
        """Missing alignment_status defaults to OK signal."""
        from backend.ht.ht_replay_verifier import to_governance_signal_for_ht

        governance_view = {
            "ht_ok": True,
            "replay_radar_status": "OK",
            # alignment_status is missing
            "blocking_fingerprints": [],
            "ht_status": "OK",
            "drift_status": "STABLE"
        }

        result = to_governance_signal_for_ht(governance_view)

        assert result["signal"] == "OK"
        assert result["alignment_status"] == "ALIGNED"  # Default

    def test_source_is_ht_replay(self):
        """Source field is always 'ht_replay'."""
        from backend.ht.ht_replay_verifier import to_governance_signal_for_ht

        governance_view = {
            "ht_ok": True,
            "alignment_status": "ALIGNED",
            "blocking_fingerprints": []
        }

        result = to_governance_signal_for_ht(governance_view)

        assert result["source"] == "ht_replay"

    def test_blocking_fingerprints_preserved(self):
        """Blocking fingerprints are preserved in signal."""
        from backend.ht.ht_replay_verifier import to_governance_signal_for_ht

        fingerprints = [
            "HT reports BLOCK while radar reports OK",
            "HT critical failures: INV-REPLAY-HT-1"
        ]

        governance_view = {
            "ht_ok": False,
            "alignment_status": "DIVERGENT",
            "blocking_fingerprints": fingerprints,
            "ht_status": "BLOCK"
        }

        result = to_governance_signal_for_ht(governance_view)

        assert result["blocking_fingerprints"] == fingerprints

    def test_ht_ok_preserved(self):
        """ht_ok status is preserved in signal."""
        from backend.ht.ht_replay_verifier import to_governance_signal_for_ht

        governance_view = {
            "ht_ok": False,
            "alignment_status": "DIVERGENT",
            "blocking_fingerprints": []
        }

        result = to_governance_signal_for_ht(governance_view)

        assert result["ht_ok"] is False

    def test_details_contains_context(self):
        """Details dict contains additional context."""
        from backend.ht.ht_replay_verifier import to_governance_signal_for_ht

        governance_view = {
            "ht_ok": True,
            "replay_radar_status": "OK",
            "alignment_status": "ALIGNED",
            "blocking_fingerprints": [],
            "ht_status": "OK",
            "drift_status": "STABLE",
            "health_score": 5,
            "recurrent_critical_failures": ["INV-REPLAY-HT-1"]
        }

        result = to_governance_signal_for_ht(governance_view)

        assert "details" in result
        assert result["details"]["ht_status"] == "OK"
        assert result["details"]["radar_status"] == "OK"
        assert result["details"]["drift_status"] == "STABLE"
        assert result["details"]["health_score"] == 5
        assert "INV-REPLAY-HT-1" in result["details"]["recurrent_critical_failures"]

    def test_signal_contains_required_fields(self):
        """Governance signal contains all required fields."""
        from backend.ht.ht_replay_verifier import to_governance_signal_for_ht

        governance_view = {
            "ht_ok": True,
            "alignment_status": "ALIGNED",
            "blocking_fingerprints": []
        }

        result = to_governance_signal_for_ht(governance_view)

        required_fields = [
            "signal",
            "source",
            "alignment_status",
            "blocking_fingerprints",
            "ht_ok",
            "details"
        ]

        for field in required_fields:
            assert field in result, f"Missing required field: {field}"


class TestGovernanceSignalFromHistory:
    """Tests for build_ht_governance_signal_from_history convenience function."""

    def test_builds_signal_from_ok_history(self):
        """Builds OK signal from healthy history."""
        from backend.ht.ht_replay_verifier import build_ht_governance_signal_from_history

        history = {
            "run_entries": [{"ht_replay_status": "OK"}],
            "ok_count": 1,
            "warn_count": 0,
            "fail_count": 0,
            "total_runs": 1,
            "recurrent_failures": {}
        }

        drift = {"status": "STABLE"}
        replay_radar = {"status": "OK"}

        result = build_ht_governance_signal_from_history(history, drift, replay_radar)

        assert result["signal"] == "OK"
        assert result["source"] == "ht_replay"

    def test_builds_block_signal_from_divergent_state(self):
        """Builds BLOCK signal when HT and radar diverge."""
        from backend.ht.ht_replay_verifier import build_ht_governance_signal_from_history

        history = {
            "run_entries": [{"ht_replay_status": "FAIL"}],
            "ok_count": 0,
            "warn_count": 0,
            "fail_count": 1,
            "total_runs": 1,
            "recurrent_failures": {}
        }

        drift = {"status": "STABLE"}
        replay_radar = {"status": "OK"}

        result = build_ht_governance_signal_from_history(history, drift, replay_radar)

        assert result["signal"] == "BLOCK"
        assert result["alignment_status"] == "DIVERGENT"

    def test_builds_warn_signal_from_tension_state(self):
        """Builds WARN signal when HT and radar have tension."""
        from backend.ht.ht_replay_verifier import build_ht_governance_signal_from_history

        history = {
            "run_entries": [{"ht_replay_status": "WARN"}],
            "ok_count": 0,
            "warn_count": 1,
            "fail_count": 0,
            "total_runs": 1,
            "recurrent_failures": {}
        }

        drift = {"status": "STABLE"}
        replay_radar = {"status": "OK"}

        result = build_ht_governance_signal_from_history(history, drift, replay_radar)

        assert result["signal"] == "WARN"
        assert result["alignment_status"] == "TENSION"

    def test_end_to_end_integration(self):
        """End-to-end test: history -> governance view -> signal."""
        from backend.ht.ht_replay_verifier import (
            build_ht_governance_signal_from_history,
            build_ht_replay_governance_view,
            to_governance_signal_for_ht
        )

        history = {
            "run_entries": [{"ht_replay_status": "OK"}],
            "ok_count": 1,
            "warn_count": 0,
            "fail_count": 0,
            "total_runs": 1,
            "recurrent_failures": {}
        }

        drift = {"status": "STABLE"}
        replay_radar = {"status": "OK"}

        # Build via convenience function
        result_convenience = build_ht_governance_signal_from_history(
            history, drift, replay_radar
        )

        # Build via two-step process
        governance_view = build_ht_replay_governance_view(history, drift, replay_radar)
        result_two_step = to_governance_signal_for_ht(governance_view)

        # Should be equivalent
        assert result_convenience["signal"] == result_two_step["signal"]
        assert result_convenience["alignment_status"] == result_two_step["alignment_status"]
        assert result_convenience["source"] == result_two_step["source"]
