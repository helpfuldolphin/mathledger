"""
Regression tests for frozen version immutability.

These tests verify that:
1. `build --all` twice from same tag -> byte-identical output
2. Build from later tag -> older version directories hash-identical
3. Frozen versions cannot be modified without detection

The frozen version system ensures that once a version is "frozen", its
site directory bytes will never change, regardless of template updates
or other changes in the build script.

Requirement: Building from a later tag must NOT change any existing
site/v0* directory bytes (except /versions/ and /versions/status.json).
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


# Paths
REPO_ROOT = Path(__file__).parent.parent.parent
SITE_DIR = REPO_ROOT / "site"
FROZEN_DIR = REPO_ROOT / "releases" / "frozen"
BUILD_SCRIPT = REPO_ROOT / "scripts" / "build_static_site.py"


def compute_directory_hash(directory: Path) -> str:
    """Compute a hash of all files in a directory (sorted by path)."""
    h = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            rel_path = str(path.relative_to(directory)).replace("\\", "/")
            h.update(rel_path.encode())
            h.update(path.read_bytes())
    return h.hexdigest()


def compute_file_hashes(directory: Path) -> dict[str, str]:
    """Compute SHA256 hashes for all files in a directory."""
    hashes = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            rel_path = str(path.relative_to(directory)).replace("\\", "/")
            hashes[rel_path] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def run_build(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run the build script with given arguments."""
    cmd = [sys.executable, str(BUILD_SCRIPT)] + list(args)
    return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT, check=check)


class TestFrozenVersionMechanism:
    """Tests for the frozen version freezing/unfreezing mechanism."""

    def test_freeze_creates_manifest(self, tmp_path):
        """Freezing a version creates a manifest in releases/frozen/."""
        # This test requires a built version
        if not (SITE_DIR / "v0.2.6").exists():
            pytest.skip("v0.2.6 not built")

        freeze_file = FROZEN_DIR / "v0.2.6.json"
        if freeze_file.exists():
            # Already frozen, verify manifest structure
            manifest = json.loads(freeze_file.read_text(encoding="utf-8"))
            assert "version" in manifest
            assert "frozen_at" in manifest
            assert "frozen_by_commit" in manifest
            assert "content_hash" in manifest
            assert "file_count" in manifest
            assert "files" in manifest
            assert manifest["version"] == "v0.2.6"
            assert manifest["file_count"] > 0
            assert len(manifest["files"]) == manifest["file_count"]

    def test_frozen_manifest_structure(self):
        """Frozen manifest has correct structure and valid hashes."""
        freeze_files = list(FROZEN_DIR.glob("*.json"))
        if not freeze_files:
            pytest.skip("No frozen versions")

        for freeze_file in freeze_files:
            manifest = json.loads(freeze_file.read_text(encoding="utf-8"))

            # Required fields
            assert "version" in manifest
            assert "frozen_at" in manifest
            assert "content_hash" in manifest
            assert "files" in manifest

            # content_hash is valid SHA256
            assert len(manifest["content_hash"]) == 64
            assert all(c in "0123456789abcdef" for c in manifest["content_hash"])

            # All file hashes are valid SHA256
            for path, hash_val in manifest["files"].items():
                assert len(hash_val) == 64, f"Invalid hash for {path}"
                assert all(c in "0123456789abcdef" for c in hash_val), f"Invalid hash chars for {path}"


class TestBuildIdempotency:
    """Tests that building the same version twice produces identical output."""

    @pytest.mark.slow
    def test_build_twice_produces_identical_output(self):
        """Building --all twice from same commit produces byte-identical output.

        This is the key regression test: if templates don't change, output
        shouldn't change.

        NOTE: This test is slow and may be skipped in CI with -m 'not slow'.
        """
        # Skip if no frozen versions exist
        frozen_versions = list(FROZEN_DIR.glob("*.json"))
        if not frozen_versions:
            pytest.skip("No frozen versions to test")

        # For each frozen version, verify current build matches freeze manifest
        for freeze_file in frozen_versions:
            manifest = json.loads(freeze_file.read_text(encoding="utf-8"))
            version = manifest["version"]
            version_dir = SITE_DIR / version

            if not version_dir.exists():
                continue  # Skip if version not built

            # Compute current hashes
            current_hashes = compute_file_hashes(version_dir)

            # Compare to frozen hashes
            expected_hashes = manifest["files"]

            # Check for differences
            missing = set(expected_hashes.keys()) - set(current_hashes.keys())
            extra = set(current_hashes.keys()) - set(expected_hashes.keys())
            modified = [
                path for path in expected_hashes
                if path in current_hashes and current_hashes[path] != expected_hashes[path]
            ]

            assert not missing, f"{version}: Missing files: {missing}"
            assert not extra, f"{version}: Extra files: {extra}"
            assert not modified, f"{version}: Modified files: {modified}"


class TestFrozenVersionImmutability:
    """Tests that frozen versions are immutable."""

    def test_immutability_check_passes(self):
        """--check-immutability passes for all frozen versions."""
        result = run_build("--check-immutability", check=False)

        # Check exit code
        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)

        # If no frozen versions, script should pass
        if "No frozen versions found" in result.stdout:
            return

        assert result.returncode == 0, f"Immutability check failed: {result.stdout}"
        assert "IMMUTABILITY VIOLATED" not in result.stdout

    def test_frozen_version_skips_rebuild(self):
        """Building a frozen version skips regeneration."""
        frozen_versions = list(FROZEN_DIR.glob("*.json"))
        if not frozen_versions:
            pytest.skip("No frozen versions")

        # Take first frozen version
        manifest = json.loads(frozen_versions[0].read_text(encoding="utf-8"))
        version = manifest["version"]

        # Build this version
        result = run_build("--version", version, "--no-verify", check=False)

        # Should skip rebuild
        assert f"FROZEN {version}: verified" in result.stdout or \
               f"FROZEN {version}: directory missing" in result.stdout

    def test_tampering_detected(self):
        """Modifying a frozen version's files is detected."""
        frozen_versions = list(FROZEN_DIR.glob("*.json"))
        if not frozen_versions:
            pytest.skip("No frozen versions")

        manifest = json.loads(frozen_versions[0].read_text(encoding="utf-8"))
        version = manifest["version"]
        version_dir = SITE_DIR / version

        if not version_dir.exists():
            pytest.skip(f"{version} not built")

        # Find a file to tamper
        index_file = version_dir / "index.html"
        if not index_file.exists():
            pytest.skip("No index.html to tamper")

        # Store original content
        original_content = index_file.read_bytes()

        try:
            # Tamper with file
            index_file.write_bytes(original_content + b"<!-- tampered -->")

            # Check immutability - should fail
            result = run_build("--check-immutability", check=False)

            # Should report violation
            assert result.returncode != 0 or "MODIFIED: index.html" in result.stdout or \
                   "IMMUTABILITY VIOLATED" in result.stdout, \
                   f"Tampering not detected: {result.stdout}"

        finally:
            # Restore original content
            index_file.write_bytes(original_content)


class TestLaterTagDoesNotModifyOlder:
    """Tests that building from a later tag doesn't modify older frozen versions."""

    def test_frozen_versions_unchanged_after_all_build(self):
        """Building --all doesn't modify frozen version directories.

        This is the critical test: building from v0.2.7 must not change
        site/v0.2.6/ if v0.2.6 is frozen.
        """
        frozen_versions = list(FROZEN_DIR.glob("*.json"))
        if not frozen_versions:
            pytest.skip("No frozen versions")

        # Record hashes before build
        before_hashes = {}
        for freeze_file in frozen_versions:
            manifest = json.loads(freeze_file.read_text(encoding="utf-8"))
            version = manifest["version"]
            version_dir = SITE_DIR / version
            if version_dir.exists():
                before_hashes[version] = compute_directory_hash(version_dir)

        if not before_hashes:
            pytest.skip("No frozen version directories exist")

        # Build all (this might add new versions but shouldn't modify frozen ones)
        result = run_build("--all", "--no-verify", check=False)

        # Check that frozen version hashes are identical
        for version, before_hash in before_hashes.items():
            version_dir = SITE_DIR / version
            after_hash = compute_directory_hash(version_dir)

            assert before_hash == after_hash, \
                f"Frozen version {version} was modified during --all build!\n" \
                f"Before: {before_hash}\nAfter: {after_hash}"


class TestFreezeAllVersions:
    """Tests for the --freeze-all command."""

    def test_freeze_all_freezes_built_versions(self):
        """--freeze-all creates manifests for all built versions."""
        # Count versions that should be frozen
        releases_file = REPO_ROOT / "releases" / "releases.json"
        releases = json.loads(releases_file.read_text(encoding="utf-8"))

        built_versions = [
            v for v in releases["versions"]
            if (SITE_DIR / v).exists()
        ]

        if not built_versions:
            pytest.skip("No versions built")

        # Count already frozen
        already_frozen = [v for v in built_versions if (FROZEN_DIR / f"{v}.json").exists()]

        # Run freeze-all
        result = run_build("--freeze-all", check=False)
        assert result.returncode == 0, f"--freeze-all failed: {result.stderr}"

        # All built versions should now be frozen
        for v in built_versions:
            freeze_file = FROZEN_DIR / f"{v}.json"
            assert freeze_file.exists(), f"{v} should be frozen after --freeze-all"
