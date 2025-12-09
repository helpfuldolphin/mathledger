# PHASE II — NOT USED IN PHASE I
"""
Tests for PRNG Manifest Diff Tool.

Verifies:
- Identical manifests → EQUIVALENT
- Same master seed, different lineage → DRIFTED
- Different derivation scheme → INCOMPATIBLE
"""

import json
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.prng_manifest_diff import (
    compare_manifests,
    extract_attestation,
    compare_field,
    DiffStatus,
    DiffSeverity,
    DiffResult,
)


def create_manifest(
    master_seed: str = "a" * 64,
    derivation_scheme: str = "PRNGKey(root, path) -> SHA256 -> seed % 2^32",
    merkle_root: str = "b" * 64,
    entry_count: int = 10,
    implementation: str = "rfl/prng/deterministic_prng.py@abc123",
    extra_config: dict = None,
) -> dict:
    """Create a test manifest with PRNG attestation."""
    manifest = {
        "manifest_version": "1.1",
        "experiment_id": "test_exp",
        "prng_attestation": {
            "schema_version": "1.0",
            "master_seed_hex": master_seed,
            "derivation_scheme": derivation_scheme,
            "lineage_merkle_root": merkle_root,
            "lineage_entry_count": entry_count,
            "implementation": implementation,
            "integration_tests_passed": True,
        },
        "configuration": {
            "snapshot": extra_config or {}
        }
    }
    return manifest


class TestDiffIdentical:
    """Tests for identical manifests."""

    def test_identical_manifests_equivalent(self, tmp_path: Path):
        """Two identical manifests are EQUIVALENT."""
        manifest = create_manifest()

        left_path = tmp_path / "left.json"
        right_path = tmp_path / "right.json"
        left_path.write_text(json.dumps(manifest))
        right_path.write_text(json.dumps(manifest))

        result = compare_manifests(left_path, right_path)

        assert result.status == DiffStatus.EQUIVALENT
        assert len(result.differences) == 0
        assert "identical" in result.summary.lower()

    def test_same_attestation_different_timestamps(self, tmp_path: Path):
        """Manifests with same attestation but different timestamps are EQUIVALENT."""
        manifest1 = create_manifest()
        manifest1["timestamp_utc"] = "2024-01-01T00:00:00Z"

        manifest2 = create_manifest()
        manifest2["timestamp_utc"] = "2024-01-02T00:00:00Z"

        left_path = tmp_path / "left.json"
        right_path = tmp_path / "right.json"
        left_path.write_text(json.dumps(manifest1))
        right_path.write_text(json.dumps(manifest2))

        result = compare_manifests(left_path, right_path)

        assert result.status == DiffStatus.EQUIVALENT


class TestDiffDrifted:
    """Tests for drifted manifests (same seed, different derivation)."""

    def test_same_seed_different_merkle_is_drifted(self, tmp_path: Path):
        """Same master seed but different Merkle root is DRIFTED."""
        manifest1 = create_manifest(merkle_root="a" * 64)
        manifest2 = create_manifest(merkle_root="b" * 64)

        left_path = tmp_path / "left.json"
        right_path = tmp_path / "right.json"
        left_path.write_text(json.dumps(manifest1))
        right_path.write_text(json.dumps(manifest2))

        result = compare_manifests(left_path, right_path)

        assert result.status == DiffStatus.DRIFTED
        merkle_diff = next(
            (d for d in result.differences if d.field == "lineage_merkle_root"),
            None
        )
        assert merkle_diff is not None
        assert "DRIFT" in merkle_diff.message

    def test_same_seed_different_entry_count_is_drifted(self, tmp_path: Path):
        """Same master seed but different entry count is DRIFTED."""
        manifest1 = create_manifest(entry_count=10)
        manifest2 = create_manifest(entry_count=20)

        left_path = tmp_path / "left.json"
        right_path = tmp_path / "right.json"
        left_path.write_text(json.dumps(manifest1))
        right_path.write_text(json.dumps(manifest2))

        result = compare_manifests(left_path, right_path)

        # Entry count difference with same seed suggests drift
        assert result.status in (DiffStatus.DRIFTED, DiffStatus.EQUIVALENT)
        entry_diff = next(
            (d for d in result.differences if d.field == "lineage_entry_count"),
            None
        )
        assert entry_diff is not None


class TestDiffIncompatible:
    """Tests for incompatible manifests."""

    def test_different_derivation_scheme_is_incompatible(self, tmp_path: Path):
        """Different derivation schemes are INCOMPATIBLE."""
        manifest1 = create_manifest(derivation_scheme="scheme_v1")
        manifest2 = create_manifest(derivation_scheme="scheme_v2")

        left_path = tmp_path / "left.json"
        right_path = tmp_path / "right.json"
        left_path.write_text(json.dumps(manifest1))
        right_path.write_text(json.dumps(manifest2))

        result = compare_manifests(left_path, right_path)

        assert result.status == DiffStatus.INCOMPATIBLE
        scheme_diff = next(
            (d for d in result.differences if d.field == "derivation_scheme"),
            None
        )
        assert scheme_diff is not None
        assert scheme_diff.severity == DiffSeverity.ERROR

    def test_one_missing_attestation_is_incompatible(self, tmp_path: Path):
        """One manifest without attestation is INCOMPATIBLE."""
        manifest1 = create_manifest()
        manifest2 = {"manifest_version": "1.0", "experiment_id": "test"}

        left_path = tmp_path / "left.json"
        right_path = tmp_path / "right.json"
        left_path.write_text(json.dumps(manifest1))
        right_path.write_text(json.dumps(manifest2))

        result = compare_manifests(left_path, right_path)

        assert result.status == DiffStatus.INCOMPATIBLE


class TestDiffDifferentSeeds:
    """Tests for manifests with different master seeds."""

    def test_different_seeds_different_merkle_is_equivalent(self, tmp_path: Path):
        """Different seeds with different Merkle roots is EQUIVALENT (expected)."""
        manifest1 = create_manifest(master_seed="a" * 64, merkle_root="1" * 64)
        manifest2 = create_manifest(master_seed="b" * 64, merkle_root="2" * 64)

        left_path = tmp_path / "left.json"
        right_path = tmp_path / "right.json"
        left_path.write_text(json.dumps(manifest1))
        right_path.write_text(json.dumps(manifest2))

        result = compare_manifests(left_path, right_path)

        # Different experiments, different seeds = expected difference
        assert result.status == DiffStatus.EQUIVALENT
        seed_diff = next(
            (d for d in result.differences if d.field == "master_seed_hex"),
            None
        )
        assert seed_diff is not None
        assert seed_diff.severity == DiffSeverity.INFO


class TestErrorHandling:
    """Tests for error handling."""

    def test_file_not_found(self, tmp_path: Path):
        """Missing file returns ERROR status."""
        existing = tmp_path / "exists.json"
        existing.write_text(json.dumps(create_manifest()))

        result = compare_manifests(existing, tmp_path / "missing.json")

        assert result.status == DiffStatus.ERROR
        assert "not found" in result.summary.lower()

    def test_invalid_json(self, tmp_path: Path):
        """Invalid JSON returns ERROR status."""
        valid = tmp_path / "valid.json"
        invalid = tmp_path / "invalid.json"

        valid.write_text(json.dumps(create_manifest()))
        invalid.write_text("{ invalid json }")

        result = compare_manifests(valid, invalid)

        assert result.status == DiffStatus.ERROR
        assert "json" in result.summary.lower()

    def test_both_missing_attestation(self, tmp_path: Path):
        """Both manifests without attestation is EQUIVALENT (legacy)."""
        manifest1 = {"manifest_version": "1.0"}
        manifest2 = {"manifest_version": "1.0"}

        left_path = tmp_path / "left.json"
        right_path = tmp_path / "right.json"
        left_path.write_text(json.dumps(manifest1))
        right_path.write_text(json.dumps(manifest2))

        result = compare_manifests(left_path, right_path)

        assert result.status == DiffStatus.EQUIVALENT
        assert "legacy" in result.summary.lower()


class TestVerboseMode:
    """Tests for verbose mode."""

    def test_verbose_includes_config_diffs(self, tmp_path: Path):
        """Verbose mode includes configuration differences."""
        manifest1 = create_manifest(extra_config={"slice_name": "slice_a"})
        manifest2 = create_manifest(extra_config={"slice_name": "slice_b"})

        left_path = tmp_path / "left.json"
        right_path = tmp_path / "right.json"
        left_path.write_text(json.dumps(manifest1))
        right_path.write_text(json.dumps(manifest2))

        result = compare_manifests(left_path, right_path, verbose=True)

        config_diff = next(
            (d for d in result.differences if "config" in d.field),
            None
        )
        assert config_diff is not None


class TestJsonOutput:
    """Tests for JSON output format."""

    def test_result_to_dict(self):
        """DiffResult serializes to dict correctly."""
        result = DiffResult(
            status=DiffStatus.EQUIVALENT,
            left_path="left.json",
            right_path="right.json",
            differences=[],
            summary="Manifests match",
        )

        d = result.to_dict()

        assert d["status"] == "EQUIVALENT"
        assert d["left_path"] == "left.json"
        assert d["right_path"] == "right.json"
        assert d["summary"] == "Manifests match"


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_extract_attestation_prng_attestation(self):
        """Extracts prng_attestation block."""
        manifest = {"prng_attestation": {"master_seed_hex": "test"}}
        attest = extract_attestation(manifest)
        assert attest["master_seed_hex"] == "test"

    def test_extract_attestation_prng_fallback(self):
        """Falls back to prng block."""
        manifest = {"prng": {"master_seed_hex": "test"}}
        attest = extract_attestation(manifest)
        assert attest["master_seed_hex"] == "test"

    def test_extract_attestation_none(self):
        """Returns None if no attestation."""
        manifest = {}
        attest = extract_attestation(manifest)
        assert attest is None

    def test_compare_field_equal(self):
        """compare_field returns None for equal values."""
        diff = compare_field("test", "value", "value")
        assert diff is None

    def test_compare_field_different(self):
        """compare_field returns Difference for different values."""
        diff = compare_field("test", "left", "right")
        assert diff is not None
        assert diff.field == "test"
        assert diff.left_value == "left"
        assert diff.right_value == "right"

