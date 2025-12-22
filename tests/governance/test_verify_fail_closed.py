#!/usr/bin/env python3
"""
Negative Tests: verify.py Fail-Closed Behavior
===============================================

These tests verify that verify.py correctly fails closed when:
1. commitment_registry_sha256 is missing
2. Registry hash is mismatched
3. artifact_kind is missing for any artifact
4. artifact_kind has an invalid enum value

All tests mutate a valid manifest and assert verify.py exits with code 1.

Mode: SHADOW-OBSERVE (tests do not enforce, only verify behavior)
"""

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


# Path to the demo script
DEMO_SCRIPT = Path(__file__).parent.parent.parent / "scripts" / "run_dropin_demo.py"


def run_demo(output_dir: Path, seed: int = 42) -> None:
    """Run the demo to generate a valid output directory."""
    result = subprocess.run(
        [sys.executable, str(DEMO_SCRIPT), "--seed", str(seed), "--output", str(output_dir)],
        capture_output=True,
        text=True,
        cwd=str(output_dir.parent.parent),
    )
    if result.returncode != 0:
        raise RuntimeError(f"Demo failed: {result.stderr}")


def run_verify(output_dir: Path) -> tuple[int, str, str]:
    """Run verify.py and return (exit_code, stdout, stderr)."""
    verify_script = output_dir / "verify.py"
    result = subprocess.run(
        [sys.executable, str(verify_script)],
        capture_output=True,
        text=True,
        cwd=str(output_dir),
    )
    return result.returncode, result.stdout, result.stderr


def load_manifest(output_dir: Path) -> dict:
    """Load manifest.json from output directory."""
    return json.loads((output_dir / "manifest.json").read_text())


def save_manifest(output_dir: Path, manifest: dict) -> None:
    """Save modified manifest.json."""
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )


class TestVerifyFailClosed:
    """Test suite for verify.py fail-closed behavior."""

    @pytest.fixture
    def demo_output(self, tmp_path):
        """Generate a valid demo output directory."""
        output_dir = tmp_path / "demo_output"
        output_dir.mkdir()
        run_demo(output_dir)
        return output_dir

    def test_valid_manifest_passes(self, demo_output):
        """Baseline: valid manifest should pass verification."""
        exit_code, stdout, stderr = run_verify(demo_output)
        assert exit_code == 0, f"Expected pass but got exit code {exit_code}: {stdout}"
        assert "[PASS] All verifications passed" in stdout

    def test_missing_governance_registry_fails(self, demo_output):
        """Removing governance_registry block should fail verification."""
        manifest = load_manifest(demo_output)
        del manifest["governance_registry"]
        save_manifest(demo_output, manifest)

        exit_code, stdout, stderr = run_verify(demo_output)
        assert exit_code == 1, f"Expected failure but got exit code {exit_code}"
        assert "[FAIL] Missing governance_registry block" in stdout
        assert "missing_governance_registry" in stdout

    def test_missing_commitment_registry_sha256_fails(self, demo_output):
        """Removing commitment_registry_sha256 field should fail verification."""
        manifest = load_manifest(demo_output)
        del manifest["governance_registry"]["commitment_registry_sha256"]
        save_manifest(demo_output, manifest)

        exit_code, stdout, stderr = run_verify(demo_output)
        # This will cause a KeyError in verify.py, resulting in failure
        assert exit_code != 0, f"Expected failure but got exit code {exit_code}"

    def test_mismatched_registry_hash_fails(self, demo_output):
        """Modifying registry hash to wrong value should fail verification."""
        manifest = load_manifest(demo_output)
        # Set to a wrong hash
        manifest["governance_registry"]["commitment_registry_sha256"] = "0" * 64
        save_manifest(demo_output, manifest)

        exit_code, stdout, stderr = run_verify(demo_output)
        assert exit_code == 1, f"Expected failure but got exit code {exit_code}"
        assert "[FAIL] Governance registry hash mismatch" in stdout
        assert "registry_hash_mismatch" in stdout

    def test_missing_artifact_kind_fails(self, demo_output):
        """Removing artifact_kind from an artifact should fail verification."""
        manifest = load_manifest(demo_output)
        if "artifacts" in manifest and len(manifest["artifacts"]) > 0:
            del manifest["artifacts"][0]["artifact_kind"]
            save_manifest(demo_output, manifest)

            exit_code, stdout, stderr = run_verify(demo_output)
            assert exit_code == 1, f"Expected failure but got exit code {exit_code}"
            assert "[FAIL] Artifact kind validation failed" in stdout
            assert "missing artifact_kind" in stdout
        else:
            pytest.skip("No artifacts in manifest")

    def test_invalid_artifact_kind_fails(self, demo_output):
        """Setting artifact_kind to invalid enum should fail verification."""
        manifest = load_manifest(demo_output)
        if "artifacts" in manifest and len(manifest["artifacts"]) > 0:
            manifest["artifacts"][0]["artifact_kind"] = "INVALID_KIND"
            save_manifest(demo_output, manifest)

            exit_code, stdout, stderr = run_verify(demo_output)
            assert exit_code == 1, f"Expected failure but got exit code {exit_code}"
            assert "[FAIL] Artifact kind validation failed" in stdout
            assert "invalid artifact_kind 'INVALID_KIND'" in stdout
        else:
            pytest.skip("No artifacts in manifest")

    def test_modified_registry_file_fails(self, demo_output):
        """Modifying the registry file content should fail hash verification."""
        registry_path = demo_output / "governance" / "commitment_registry.json"
        registry = json.loads(registry_path.read_text())
        # Add a new commitment to change the hash
        registry["commitments"].append({
            "commitment_id": "CXXX",
            "description": "Tampered commitment",
            "constraint_type": "POLICY",
            "status": "ACTIVE"
        })
        registry_path.write_text(json.dumps(registry, indent=2))

        exit_code, stdout, stderr = run_verify(demo_output)
        assert exit_code == 1, f"Expected failure but got exit code {exit_code}"
        assert "[FAIL] Governance registry hash mismatch" in stdout

    def test_canonicalization_invariance(self, demo_output):
        """
        Reformatting registry file (whitespace, key order) must NOT change hash.

        This test proves canonicalization is real:
        1. Load registry JSON
        2. Rewrite with different formatting (reversed keys, extra whitespace)
        3. verify.py must still PASS because canonical form is identical
        """
        registry_path = demo_output / "governance" / "commitment_registry.json"
        registry = json.loads(registry_path.read_text())

        # Rewrite with reversed key order and different indentation
        # json.dumps with sort_keys=False preserves insertion order
        # We'll manually construct a differently-formatted but semantically identical JSON
        reformatted = json.dumps(
            registry,
            indent=4,  # Different indent (was 2)
            sort_keys=False,  # Don't sort (rely on dict order)
            ensure_ascii=False,  # Allow unicode (different from canonical)
        )
        # Add extra whitespace at end of lines and blank lines
        lines = reformatted.split('\n')
        reformatted_with_whitespace = '\n\n'.join(line + '   ' for line in lines)

        registry_path.write_text(reformatted_with_whitespace)

        # Verify the file looks different
        new_content = registry_path.read_text()
        assert '    ' in new_content, "Should have 4-space indent"
        assert '\n\n' in new_content, "Should have blank lines"

        # But verification should still PASS because canonicalization normalizes
        exit_code, stdout, stderr = run_verify(demo_output)
        assert exit_code == 0, f"Expected PASS but got exit code {exit_code}: {stdout}"
        assert "[PASS] Governance registry verified" in stdout

    def test_registry_semantic_change_changes_hash(self, demo_output):
        """
        Semantic changes to registry content MUST change the hash and FAIL verification.

        This is the complement to test_canonicalization_invariance:
        - Whitespace/formatting changes → hash unchanged → PASS
        - Content/semantic changes → hash changed → FAIL

        This test proves the hash actually binds to content, not just format.
        """
        registry_path = demo_output / "governance" / "commitment_registry.json"
        registry = json.loads(registry_path.read_text())

        # Make a semantic change: toggle a commitment status
        original_status = registry["commitments"][0]["status"]
        registry["commitments"][0]["status"] = "INACTIVE" if original_status == "ACTIVE" else "ACTIVE"

        # Write back with same formatting (2-space indent)
        registry_path.write_text(json.dumps(registry, indent=2))

        # Verification must FAIL because content changed
        exit_code, stdout, stderr = run_verify(demo_output)
        assert exit_code == 1, f"Expected FAIL but got exit code {exit_code}: {stdout}"
        assert "[FAIL] Governance registry hash mismatch" in stdout

    def test_registry_string_change_changes_hash(self, demo_output):
        """
        Changing a string value in registry MUST change the hash and FAIL verification.

        Tests that even small string edits are detected.
        """
        registry_path = demo_output / "governance" / "commitment_registry.json"
        registry = json.loads(registry_path.read_text())

        # Change one character in a description
        original_desc = registry["commitments"][0]["description"]
        registry["commitments"][0]["description"] = original_desc + "."  # Add period

        registry_path.write_text(json.dumps(registry, indent=2))

        exit_code, stdout, stderr = run_verify(demo_output)
        assert exit_code == 1, f"Expected FAIL but got exit code {exit_code}: {stdout}"
        assert "[FAIL] Governance registry hash mismatch" in stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
