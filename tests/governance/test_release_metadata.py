"""
Release Metadata Validation Tests.

These tests ensure that releases/releases.json is consistent with:
1. Git tags in the repository
2. docs/V0_LOCK.md release notes
3. Internal consistency (current_version matches a release)

Contract: docs/RELEASE_METADATA_CONTRACT.md

Run with:
    uv run pytest tests/governance/test_release_metadata.py -v
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Paths
REPO_ROOT = Path(__file__).parent.parent.parent
RELEASES_JSON = REPO_ROOT / "releases" / "releases.json"
V0_LOCK_MD = REPO_ROOT / "docs" / "V0_LOCK.md"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def releases_data() -> Dict[str, Any]:
    """Load releases.json data."""
    assert RELEASES_JSON.exists(), f"releases.json not found at {RELEASES_JSON}"
    return json.loads(RELEASES_JSON.read_text())


@pytest.fixture
def v0_lock_content() -> str:
    """Load V0_LOCK.md content."""
    assert V0_LOCK_MD.exists(), f"V0_LOCK.md not found at {V0_LOCK_MD}"
    return V0_LOCK_MD.read_text()


# ---------------------------------------------------------------------------
# Schema Validation
# ---------------------------------------------------------------------------


class TestReleasesJsonSchema:
    """Test that releases.json has required structure."""

    def test_file_exists(self):
        """Verify releases.json exists."""
        assert RELEASES_JSON.exists()

    def test_valid_json(self, releases_data: Dict[str, Any]):
        """Verify file is valid JSON."""
        assert isinstance(releases_data, dict)

    def test_has_current_version(self, releases_data: Dict[str, Any]):
        """Verify current_version field exists."""
        assert "current_version" in releases_data
        assert isinstance(releases_data["current_version"], str)

    def test_has_releases_array(self, releases_data: Dict[str, Any]):
        """Verify releases array exists."""
        assert "releases" in releases_data
        assert isinstance(releases_data["releases"], list)
        assert len(releases_data["releases"]) > 0

    def test_releases_have_required_fields(self, releases_data: Dict[str, Any]):
        """Verify each release has required fields."""
        required_fields = [
            "version",
            "git_tag",
            "commit_hash",
            "date_locked",
            "status",
            "tier_counts",
        ]

        for release in releases_data["releases"]:
            for field in required_fields:
                assert field in release, f"Release missing {field}: {release.get('version', 'unknown')}"

    def test_commit_hash_format(self, releases_data: Dict[str, Any]):
        """Verify commit hashes are 40 hex characters."""
        for release in releases_data["releases"]:
            commit = release["commit_hash"]
            assert len(commit) == 40, f"Invalid commit hash length: {commit}"
            assert all(c in "0123456789abcdef" for c in commit.lower()), f"Invalid commit hash: {commit}"

    def test_date_format(self, releases_data: Dict[str, Any]):
        """Verify dates are YYYY-MM-DD format."""
        date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        for release in releases_data["releases"]:
            date = release["date_locked"]
            assert date_pattern.match(date), f"Invalid date format: {date}"

    def test_valid_status(self, releases_data: Dict[str, Any]):
        """Verify status is valid enum value."""
        valid_statuses = {"current", "superseded", "internal"}
        for release in releases_data["releases"]:
            status = release["status"]
            assert status in valid_statuses, f"Invalid status: {status}"

    def test_tier_counts_structure(self, releases_data: Dict[str, Any]):
        """Verify tier_counts has required fields."""
        for release in releases_data["releases"]:
            tier_counts = release["tier_counts"]
            assert "tier_a" in tier_counts
            assert "tier_b" in tier_counts
            assert "tier_c" in tier_counts
            assert isinstance(tier_counts["tier_a"], int)
            assert isinstance(tier_counts["tier_b"], int)
            assert isinstance(tier_counts["tier_c"], int)


# ---------------------------------------------------------------------------
# Internal Consistency
# ---------------------------------------------------------------------------


class TestInternalConsistency:
    """Test internal consistency of releases.json."""

    def test_current_version_exists_in_releases(self, releases_data: Dict[str, Any]):
        """Verify current_version matches a release."""
        current = releases_data["current_version"]
        versions = [r["version"] for r in releases_data["releases"]]
        assert current in versions, f"current_version {current} not in releases: {versions}"

    def test_exactly_one_current_release(self, releases_data: Dict[str, Any]):
        """Verify exactly one release has status=current."""
        current_releases = [r for r in releases_data["releases"] if r["status"] == "current"]
        assert len(current_releases) == 1, f"Expected 1 current release, got {len(current_releases)}"

    def test_current_version_matches_current_release(self, releases_data: Dict[str, Any]):
        """Verify current_version matches the release with status=current."""
        current_version = releases_data["current_version"]
        current_release = next(r for r in releases_data["releases"] if r["status"] == "current")
        assert current_release["version"] == current_version, (
            f"current_version ({current_version}) != current release version ({current_release['version']})"
        )

    def test_unique_versions(self, releases_data: Dict[str, Any]):
        """Verify no duplicate versions."""
        versions = [r["version"] for r in releases_data["releases"]]
        assert len(versions) == len(set(versions)), f"Duplicate versions: {versions}"

    def test_unique_tags(self, releases_data: Dict[str, Any]):
        """Verify no duplicate git tags."""
        tags = [r["git_tag"] for r in releases_data["releases"]]
        assert len(tags) == len(set(tags)), f"Duplicate tags: {tags}"

    def test_unique_commits(self, releases_data: Dict[str, Any]):
        """Verify no duplicate commit hashes."""
        commits = [r["commit_hash"] for r in releases_data["releases"]]
        assert len(commits) == len(set(commits)), f"Duplicate commits: {commits}"


# ---------------------------------------------------------------------------
# Git Validation
# ---------------------------------------------------------------------------


class TestGitConsistency:
    """Test that releases.json matches git repository."""

    def _git_tag_exists(self, tag: str) -> bool:
        """Check if a git tag exists."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", tag],
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _git_tag_commit(self, tag: str) -> str:
        """Get commit hash for a git tag."""
        result = subprocess.run(
            ["git", "rev-parse", f"{tag}^{{commit}}"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        return result.stdout.strip()

    def test_tags_exist(self, releases_data: Dict[str, Any]):
        """Verify all git tags exist in repository."""
        for release in releases_data["releases"]:
            tag = release["git_tag"]
            assert self._git_tag_exists(tag), f"Git tag not found: {tag}"

    def test_commits_match_tags(self, releases_data: Dict[str, Any]):
        """Verify commit hashes match git tag commits."""
        for release in releases_data["releases"]:
            tag = release["git_tag"]
            expected_commit = release["commit_hash"]

            if not self._git_tag_exists(tag):
                pytest.skip(f"Tag {tag} not found")

            actual_commit = self._git_tag_commit(tag)
            assert actual_commit == expected_commit, (
                f"Commit mismatch for {tag}: "
                f"releases.json has {expected_commit}, git has {actual_commit}"
            )


# ---------------------------------------------------------------------------
# Cross-Reference with V0_LOCK.md
# ---------------------------------------------------------------------------


class TestV0LockConsistency:
    """Test that releases.json matches V0_LOCK.md."""

    def test_v0_2_0_commit_in_v0_lock(
        self,
        releases_data: Dict[str, Any],
        v0_lock_content: str,
    ):
        """Verify v0.2.0 commit hash appears in V0_LOCK.md."""
        v020_release = next(
            (r for r in releases_data["releases"] if r["version"] == "0.2.0"),
            None,
        )
        if v020_release is None:
            pytest.skip("v0.2.0 release not found")

        commit = v020_release["commit_hash"]
        assert commit in v0_lock_content, (
            f"v0.2.0 commit {commit} not found in V0_LOCK.md"
        )

    def test_v0_2_0_tag_in_v0_lock(
        self,
        releases_data: Dict[str, Any],
        v0_lock_content: str,
    ):
        """Verify v0.2.0 tag appears in V0_LOCK.md."""
        v020_release = next(
            (r for r in releases_data["releases"] if r["version"] == "0.2.0"),
            None,
        )
        if v020_release is None:
            pytest.skip("v0.2.0 release not found")

        tag = v020_release["git_tag"]
        assert tag in v0_lock_content, (
            f"v0.2.0 tag {tag} not found in V0_LOCK.md"
        )

    def test_tier_counts_in_v0_lock(
        self,
        releases_data: Dict[str, Any],
        v0_lock_content: str,
    ):
        """Verify tier counts appear in V0_LOCK.md."""
        v020_release = next(
            (r for r in releases_data["releases"] if r["version"] == "0.2.0"),
            None,
        )
        if v020_release is None:
            pytest.skip("v0.2.0 release not found")

        tier_a = v020_release["tier_counts"]["tier_a"]
        # Check that the tier A count appears somewhere in V0_LOCK.md
        # This is a loose check - we just verify the number is mentioned
        assert str(tier_a) in v0_lock_content, (
            f"Tier A count {tier_a} not found in V0_LOCK.md"
        )


# ---------------------------------------------------------------------------
# Deployment Readiness
# ---------------------------------------------------------------------------


class TestDeploymentReadiness:
    """Test that metadata is deployment-ready."""

    def test_current_has_verification_commands(self, releases_data: Dict[str, Any]):
        """Verify current release has verification commands."""
        current_release = next(r for r in releases_data["releases"] if r["status"] == "current")
        assert "verification_commands" in current_release, "Current release missing verification_commands"
        assert len(current_release["verification_commands"]) > 0, "Current release has empty verification_commands"

    def test_metadata_is_authoritative(self, releases_data: Dict[str, Any]):
        """Verify metadata declares itself authoritative."""
        assert "metadata" in releases_data
        assert releases_data["metadata"].get("authoritative") is True

    def test_contract_reference(self, releases_data: Dict[str, Any]):
        """Verify contract document is referenced."""
        assert "metadata" in releases_data
        contract = releases_data["metadata"].get("contract", "")
        assert "RELEASE_METADATA_CONTRACT" in contract
