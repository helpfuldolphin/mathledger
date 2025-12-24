"""
Fail-closed governance tests for FOL certificates.

Phase 2 RED: These tests will fail until scripts/run_fol_fin_eq_demo.py is implemented.

Tests fail-closed behavior including:
- Missing logic_fragment causes failure
- Wrong logic_fragment causes failure
- Certificate hash mismatch causes failure
- Strategy/fragment mismatch causes failure
"""

import pytest
import subprocess
import json
from pathlib import Path

# Import module that doesn't exist yet (will fail with ModuleNotFoundError)
# This is the demo script that produces output we can tamper with
from scripts.run_fol_fin_eq_demo import run_demo


class TestFailClosedBehavior:
    """Fail-closed behavior tests."""

    @pytest.fixture
    def demo_output(self, tmp_path):
        """Run demo and return output path."""
        subprocess.run([
            "uv", "run", "python", "scripts/run_fol_fin_eq_demo.py",
            "--domain", "z2", "--output", str(tmp_path)
        ], check=True)
        return tmp_path

    def test_missing_logic_fragment_fails(self, demo_output):
        """Manifest without logic_fragment causes verify.py to fail."""
        manifest = demo_output / "manifest.json"
        data = json.loads(manifest.read_text())
        del data["logic_fragment"]
        manifest.write_text(json.dumps(data))
        result = subprocess.run(["python", demo_output / "verify.py"], capture_output=True)
        assert result.returncode != 0

    def test_wrong_logic_fragment_fails(self, demo_output):
        """Wrong logic_fragment causes verify.py to fail."""
        manifest = demo_output / "manifest.json"
        data = json.loads(manifest.read_text())
        data["logic_fragment"] = "PL"
        manifest.write_text(json.dumps(data))
        result = subprocess.run(["python", demo_output / "verify.py"], capture_output=True)
        assert result.returncode != 0

    def test_certificate_hash_mismatch_fails(self, demo_output):
        """Modified certificate with stale hash causes verify.py to fail."""
        certs = list((demo_output / "certificates").glob("*.json"))
        if certs:
            cert = json.loads(certs[0].read_text())
            cert["status"] = "TAMPERED"
            certs[0].write_text(json.dumps(cert))
        result = subprocess.run(["python", demo_output / "verify.py"], capture_output=True)
        assert result.returncode != 0

    def test_strategy_fragment_mismatch_fails(self, demo_output):
        """FOL_FIN_EQ_v1 with non-exhaustive strategy fails schema validation."""
        certs = list((demo_output / "certificates").glob("*.json"))
        if certs:
            cert = json.loads(certs[0].read_text())
            cert["verification_strategy"] = "smt_z3"
            certs[0].write_text(json.dumps(cert))
        # Schema validation happens before hash check
        result = subprocess.run(["python", demo_output / "verify.py"], capture_output=True)
        assert result.returncode != 0
