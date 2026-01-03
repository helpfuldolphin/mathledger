"""
Release Metadata Guard Test: Canonical v0.2.0 Values.

This test FAILS if releases/releases.json contains incorrect v0.2.0 metadata.
It exists because Claude B previously reported wrong tag/commit values.

CANONICAL VALUES (immutable):
- current_version: "v0.2.0"
- versions["v0.2.0"].tag: "v0.2.0-demo-lock"
- versions["v0.2.0"].commit: "27a94c8a58139cb10349f6418336c618f528cbab"

If this test fails, the fix is to RESTORE the correct values in releases.json,
NOT to "update" the test to match wrong values.

See: docs/RELEASE_METADATA_DIAGNOSTIC.md

Run with:
    uv run pytest tests/governance/test_release_metadata_guard.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# Paths
REPO_ROOT = Path(__file__).parent.parent.parent
RELEASES_JSON = REPO_ROOT / "releases" / "releases.json"

# ---------------------------------------------------------------------------
# CANONICAL VALUES - DO NOT CHANGE
# ---------------------------------------------------------------------------

CANONICAL_CURRENT_VERSION = "v0.2.0"
CANONICAL_V020_TAG = "v0.2.0-demo-lock"
CANONICAL_V020_COMMIT = "27a94c8a58139cb10349f6418336c618f528cbab"
CANONICAL_V020_DATE = "2026-01-02"

# v0 reference values
CANONICAL_V0_TAG = "v0-demo-lock"
CANONICAL_V0_COMMIT = "ab8f51ab389aed7b3412cb987fc70d0d4f2bbe0b"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def releases_data() -> dict:
    """Load releases.json data."""
    assert RELEASES_JSON.exists(), f"releases.json not found at {RELEASES_JSON}"
    return json.loads(RELEASES_JSON.read_text())


# ---------------------------------------------------------------------------
# Guard Tests - FAIL if values are wrong
# ---------------------------------------------------------------------------


class TestCanonicalV020Guard:
    """
    Guard tests for v0.2.0 canonical values.

    If ANY of these tests fail, it means releases.json has been corrupted.
    The fix is to restore correct values, NOT to update these tests.
    """

    def test_current_version_is_v020(self, releases_data: dict):
        """GUARD: current_version must be 'v0.2.0'."""
        actual = releases_data.get("current_version")
        assert actual == CANONICAL_CURRENT_VERSION, (
            f"METADATA CORRUPTION DETECTED!\n"
            f"current_version is '{actual}' but should be '{CANONICAL_CURRENT_VERSION}'.\n"
            f"See docs/RELEASE_METADATA_DIAGNOSTIC.md for fix instructions."
        )

    def test_v020_tag_is_demo_lock(self, releases_data: dict):
        """GUARD: v0.2.0 tag must be 'v0.2.0-demo-lock'."""
        versions = releases_data.get("versions", {})
        v020 = versions.get("v0.2.0", {})
        actual_tag = v020.get("tag")

        assert actual_tag == CANONICAL_V020_TAG, (
            f"METADATA CORRUPTION DETECTED!\n"
            f"v0.2.0 tag is '{actual_tag}' but should be '{CANONICAL_V020_TAG}'.\n"
            f"If a tool reported a different tag, that tool is reading the wrong source.\n"
            f"See docs/RELEASE_METADATA_DIAGNOSTIC.md for fix instructions."
        )

    def test_v020_commit_is_canonical(self, releases_data: dict):
        """GUARD: v0.2.0 commit must be '27a94c8a...'."""
        versions = releases_data.get("versions", {})
        v020 = versions.get("v0.2.0", {})
        actual_commit = v020.get("commit")

        assert actual_commit == CANONICAL_V020_COMMIT, (
            f"METADATA CORRUPTION DETECTED!\n"
            f"v0.2.0 commit is '{actual_commit}'\n"
            f"but should be '{CANONICAL_V020_COMMIT}'.\n"
            f"If a tool reported a different commit, that tool is reading the wrong source.\n"
            f"See docs/RELEASE_METADATA_DIAGNOSTIC.md for fix instructions."
        )

    def test_v020_date_is_canonical(self, releases_data: dict):
        """GUARD: v0.2.0 date_locked must be '2026-01-02'."""
        versions = releases_data.get("versions", {})
        v020 = versions.get("v0.2.0", {})
        actual_date = v020.get("date_locked")

        assert actual_date == CANONICAL_V020_DATE, (
            f"Date mismatch: '{actual_date}' vs expected '{CANONICAL_V020_DATE}'"
        )


class TestCanonicalV0Guard:
    """Guard tests for v0 canonical values."""

    def test_v0_tag_is_demo_lock(self, releases_data: dict):
        """GUARD: v0 tag must be 'v0-demo-lock'."""
        versions = releases_data.get("versions", {})
        v0 = versions.get("v0", {})
        actual_tag = v0.get("tag")

        assert actual_tag == CANONICAL_V0_TAG, (
            f"v0 tag is '{actual_tag}' but should be '{CANONICAL_V0_TAG}'"
        )

    def test_v0_commit_is_canonical(self, releases_data: dict):
        """GUARD: v0 commit must be 'ab8f51ab...'."""
        versions = releases_data.get("versions", {})
        v0 = versions.get("v0", {})
        actual_commit = v0.get("commit")

        assert actual_commit == CANONICAL_V0_COMMIT, (
            f"v0 commit is '{actual_commit}' but should be '{CANONICAL_V0_COMMIT}'"
        )


class TestNoWrongTags:
    """Ensure wrong tags from Claude B are NOT present."""

    def test_no_pilot_audit_hardened_tag(self, releases_data: dict):
        """GUARD: 'v0.9.4-pilot-audit-hardened' must NOT appear anywhere."""
        json_str = json.dumps(releases_data)
        wrong_tag = "v0.9.4-pilot-audit-hardened"

        assert wrong_tag not in json_str, (
            f"WRONG TAG DETECTED: '{wrong_tag}' appears in releases.json.\n"
            f"This tag is from Claude B and is incorrect.\n"
            f"The correct v0.2.0 tag is '{CANONICAL_V020_TAG}'."
        )

    def test_no_wrong_commit(self, releases_data: dict):
        """GUARD: '07ea0edf...' (wrong commit) must NOT appear."""
        json_str = json.dumps(releases_data)
        wrong_commit = "07ea0edf02ff4173e81cef8ecfedf50195bb8673"

        assert wrong_commit not in json_str, (
            f"WRONG COMMIT DETECTED: '{wrong_commit}' appears in releases.json.\n"
            f"This commit is from Claude B and is incorrect.\n"
            f"The correct v0.2.0 commit is '{CANONICAL_V020_COMMIT}'."
        )
