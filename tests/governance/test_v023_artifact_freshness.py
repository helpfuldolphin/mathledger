"""
Tests for v0.2.3 Artifact Freshness (Version Drift Prevention)

Verifies:
1. v0.2.3 evidence pack examples exist and have correct version pins
2. v0.2.3 verifier vectors exist and have correct version
3. NO stale version strings (/v0.2.1/, pack_version: v0.2.1) in v0.2.3 artifacts
4. URLs point to /v0.2.3/evidence-pack/verify/

Run with:
    uv run pytest tests/governance/test_v023_artifact_freshness.py -v
"""

import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).parent.parent.parent
V023_EXAMPLES = REPO_ROOT / "releases" / "evidence_pack_examples.v0.2.3.json"
V023_VECTORS = REPO_ROOT / "releases" / "evidence_pack_verifier_vectors.v0.2.3.json"

# Stale version patterns that MUST NOT appear in v0.2.3 artifacts
STALE_PATTERNS = [
    "/v0.2.1/",
    "/v0.2.2/",
    "pack_version\": \"v0.2.1",
    "pack_version\": \"v0.2.2",
    "\"version\": \"v0.2.1",
    "\"version\": \"v0.2.2",
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def examples_v023():
    """Load v0.2.3 evidence pack examples."""
    if not V023_EXAMPLES.exists():
        pytest.skip(f"v0.2.3 examples not found: {V023_EXAMPLES}")
    with open(V023_EXAMPLES, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def vectors_v023():
    """Load v0.2.3 verifier vectors."""
    if not V023_VECTORS.exists():
        pytest.skip(f"v0.2.3 vectors not found: {V023_VECTORS}")
    with open(V023_VECTORS, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def examples_raw():
    """Load v0.2.3 examples as raw text for pattern matching."""
    if not V023_EXAMPLES.exists():
        pytest.skip(f"v0.2.3 examples not found: {V023_EXAMPLES}")
    return V023_EXAMPLES.read_text(encoding="utf-8")


@pytest.fixture
def vectors_raw():
    """Load v0.2.3 vectors as raw text for pattern matching."""
    if not V023_VECTORS.exists():
        pytest.skip(f"v0.2.3 vectors not found: {V023_VECTORS}")
    return V023_VECTORS.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Tests: File Existence
# ---------------------------------------------------------------------------

class TestV023ArtifactsExist:
    """Verify v0.2.3 artifacts exist."""

    def test_v023_examples_file_exists(self):
        """v0.2.3 evidence pack examples must exist."""
        assert V023_EXAMPLES.exists(), f"Missing: {V023_EXAMPLES}"

    def test_v023_vectors_file_exists(self):
        """v0.2.3 verifier vectors must exist."""
        assert V023_VECTORS.exists(), f"Missing: {V023_VECTORS}"


# ---------------------------------------------------------------------------
# Tests: No Stale Version Strings
# ---------------------------------------------------------------------------

class TestNoStaleVersionStrings:
    """Verify no stale version strings in v0.2.3 artifacts."""

    @pytest.mark.parametrize("pattern", STALE_PATTERNS)
    def test_examples_no_stale_pattern(self, examples_raw, pattern):
        """v0.2.3 examples must not contain stale version patterns."""
        assert pattern not in examples_raw, (
            f"STALE VERSION DETECTED in evidence_pack_examples.v0.2.3.json: '{pattern}'\n"
            f"All version references must be v0.2.3"
        )

    @pytest.mark.parametrize("pattern", STALE_PATTERNS)
    def test_vectors_no_stale_pattern(self, vectors_raw, pattern):
        """v0.2.3 vectors must not contain stale version patterns."""
        assert pattern not in vectors_raw, (
            f"STALE VERSION DETECTED in evidence_pack_verifier_vectors.v0.2.3.json: '{pattern}'\n"
            f"All version references must be v0.2.3"
        )


# ---------------------------------------------------------------------------
# Tests: Correct Version Pins in Examples
# ---------------------------------------------------------------------------

class TestExamplesVersionPins:
    """Verify all packs in v0.2.3 examples have correct version pins."""

    def test_all_packs_have_v023_pack_version(self, examples_v023):
        """Each pack must have pack_version: v0.2.3."""
        examples = examples_v023.get("examples", {})
        for name, example in examples.items():
            pack = example.get("pack", {})
            pack_version = pack.get("pack_version")
            assert pack_version == "v0.2.3", (
                f"Pack '{name}' has pack_version='{pack_version}', expected 'v0.2.3'"
            )

    def test_description_references_v023(self, examples_v023):
        """Top-level description must reference /v0.2.3/."""
        description = examples_v023.get("description", "")
        assert "/v0.2.3/" in description, (
            f"Description does not reference /v0.2.3/: {description}"
        )

    def test_usage_step2_references_v023(self, examples_v023):
        """Usage step_2 must reference /v0.2.3/evidence-pack/verify/."""
        instructions = examples_v023.get("usage_instructions", {})
        step_2 = instructions.get("step_2", "")
        assert "/v0.2.3/evidence-pack/verify/" in step_2, (
            f"step_2 does not reference /v0.2.3/evidence-pack/verify/: {step_2}"
        )


# ---------------------------------------------------------------------------
# Tests: Correct Version in Vectors
# ---------------------------------------------------------------------------

class TestVectorsVersion:
    """Verify v0.2.3 vectors have correct metadata version."""

    def test_metadata_version_is_v023(self, vectors_v023):
        """Metadata version must be v0.2.3."""
        metadata = vectors_v023.get("metadata", {})
        version = metadata.get("version")
        assert version == "v0.2.3", (
            f"Metadata version='{version}', expected 'v0.2.3'"
        )

    def test_metadata_has_content_hash(self, vectors_v023):
        """Metadata must include content_hash for integrity."""
        metadata = vectors_v023.get("metadata", {})
        assert "content_hash" in metadata, "Missing content_hash in metadata"
        assert len(metadata["content_hash"]) == 64, "content_hash must be 64-char hex"


# ---------------------------------------------------------------------------
# Tests: Pack Structure Integrity
# ---------------------------------------------------------------------------

class TestPackStructure:
    """Verify pack structure is correct in v0.2.3 examples."""

    def test_valid_pack_has_required_fields(self, examples_v023):
        """Valid pack must have all required fields."""
        pack = examples_v023["examples"]["valid_boundary_demo"]["pack"]
        required = ["schema_version", "pack_version", "uvil_events",
                    "reasoning_artifacts", "u_t", "r_t", "h_t"]
        for field in required:
            assert field in pack, f"Missing required field: {field}"

    def test_hash_fields_are_64_char_hex(self, examples_v023):
        """Hash fields must be 64-character hex strings."""
        pack = examples_v023["examples"]["valid_boundary_demo"]["pack"]
        for field in ["u_t", "r_t", "h_t"]:
            value = pack[field]
            assert len(value) == 64, f"{field} must be 64 chars"
            assert all(c in "0123456789abcdef" for c in value), f"{field} must be hex"


# ---------------------------------------------------------------------------
# Tests: Hash Contract Unchanged
# ---------------------------------------------------------------------------

class TestHashContractUnchanged:
    """Verify hash values are unchanged from v0.2.2 (same input = same output)."""

    def test_valid_pack_hashes_match_canonical(self, examples_v023):
        """
        Hash values must match what canonical functions produce.
        This test imports the same functions used by replay_verify.
        """
        from attestation.dual_root import (
            compute_ui_root,
            compute_reasoning_root,
            compute_composite_root,
        )

        pack = examples_v023["examples"]["valid_boundary_demo"]["pack"]

        recomputed_ut = compute_ui_root(pack["uvil_events"])
        recomputed_rt = compute_reasoning_root(pack["reasoning_artifacts"])
        recomputed_ht = compute_composite_root(recomputed_rt, recomputed_ut)

        assert pack["u_t"] == recomputed_ut, "U_t hash mismatch"
        assert pack["r_t"] == recomputed_rt, "R_t hash mismatch"
        assert pack["h_t"] == recomputed_ht, "H_t hash mismatch"
