"""
Tests for mock oracle CLI.

Verifies that the CLI produces correct exit codes, machine-parsable
JSON output, and handles various input modes correctly.

ABSOLUTE SAFEGUARD: These tests exercise the mock oracle only — never production.
"""

from __future__ import annotations

import json
import os
import tempfile
from io import StringIO
from unittest import mock

import pytest

os.environ["MATHLEDGER_ALLOW_MOCK_ORACLE"] = "1"

from backend.verification.mock_oracle_cli import main, create_parser
from backend.verification import SLICE_PROFILES


@pytest.mark.unit
class TestCLIBasics:
    """Basic tests for CLI functionality."""
    
    def test_parser_creation(self):
        """Parser is created successfully."""
        parser = create_parser()
        assert parser is not None
    
    def test_help_exits_zero(self):
        """--help exits with code 0."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0
    
    def test_no_formula_exits_one(self):
        """Missing formula exits with code 1."""
        with mock.patch("sys.stderr", new_callable=StringIO):
            result = main([])
        assert result == 1
    
    def test_single_formula_exits_zero(self):
        """Single formula verification exits with code 0."""
        with mock.patch("sys.stdout", new_callable=StringIO):
            result = main(["--formula", "p -> p"])
        assert result == 0


@pytest.mark.unit
class TestCLICoverage:
    """Tests for --coverage and --all-profiles flags."""
    
    def test_coverage_flag_exits_zero(self):
        """--coverage exits with code 0."""
        with mock.patch("sys.stdout", new_callable=StringIO):
            result = main(["--coverage"])
        assert result == 0
    
    def test_coverage_outputs_profile_info(self):
        """--coverage outputs profile coverage information."""
        stdout = StringIO()
        with mock.patch("sys.stdout", stdout):
            main(["--coverage", "--profile", "default"])
        
        output = stdout.getvalue()
        assert "default" in output
        assert "Verified" in output or "verified" in output.lower()
    
    def test_coverage_json_output(self):
        """--coverage --json produces valid JSON."""
        stdout = StringIO()
        with mock.patch("sys.stdout", stdout):
            result = main(["--coverage", "--json"])
        
        assert result == 0
        data = json.loads(stdout.getvalue())
        assert "profile" in data
        assert "coverage" in data
    
    def test_all_profiles_flag(self):
        """--all-profiles shows all profiles."""
        stdout = StringIO()
        with mock.patch("sys.stdout", stdout):
            result = main(["--all-profiles"])
        
        assert result == 0
        output = stdout.getvalue()
        
        for profile in SLICE_PROFILES:
            assert profile in output
    
    def test_all_profiles_json(self):
        """--all-profiles --json produces valid JSON with all profiles."""
        stdout = StringIO()
        with mock.patch("sys.stdout", stdout):
            result = main(["--all-profiles", "--json"])
        
        assert result == 0
        data = json.loads(stdout.getvalue())
        
        assert "profiles" in data
        for profile in SLICE_PROFILES:
            assert profile in data["profiles"]


@pytest.mark.unit
class TestCLIVerification:
    """Tests for formula verification via CLI."""
    
    def test_single_formula_human_output(self):
        """Single formula produces human-readable output."""
        stdout = StringIO()
        with mock.patch("sys.stdout", stdout):
            result = main(["--formula", "p -> p"])
        
        assert result == 0
        output = stdout.getvalue()
        assert "Formula:" in output
        assert "Bucket:" in output
    
    def test_single_formula_json_output(self):
        """--json produces valid JSON for single formula."""
        stdout = StringIO()
        with mock.patch("sys.stdout", stdout):
            result = main(["--formula", "p -> q", "--json"])
        
        assert result == 0
        data = json.loads(stdout.getvalue())
        
        assert "config" in data
        assert "results" in data
        assert len(data["results"]) == 1
        assert data["results"][0]["formula"] == "p -> q"
    
    def test_count_multiple_verifications(self):
        """--count N verifies formula N times."""
        stdout = StringIO()
        with mock.patch("sys.stdout", stdout):
            result = main(["--formula", "p -> p", "--count", "5", "--json"])
        
        assert result == 0
        data = json.loads(stdout.getvalue())
        assert len(data["results"]) == 5
        
        # All should be for same formula
        for r in data["results"]:
            assert r["formula"] == "p -> p"
    
    def test_formulas_file(self):
        """--formulas-file reads formulas from file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("p -> p\n")
            f.write("q -> q\n")
            f.write("# comment line\n")
            f.write("r -> r\n")
            f.name
        
        try:
            stdout = StringIO()
            with mock.patch("sys.stdout", stdout):
                result = main(["--formulas-file", f.name, "--json"])
            
            assert result == 0
            data = json.loads(stdout.getvalue())
            
            # Should have 3 formulas (comment line ignored)
            assert len(data["results"]) == 3
            formulas = [r["formula"] for r in data["results"]]
            assert "p -> p" in formulas
            assert "q -> q" in formulas
            assert "r -> r" in formulas
        finally:
            os.unlink(f.name)
    
    def test_missing_file_exits_one(self):
        """--formulas-file with missing file exits with code 1."""
        stderr = StringIO()
        with mock.patch("sys.stderr", stderr):
            result = main(["--formulas-file", "/nonexistent/file.txt"])
        
        assert result == 1
        assert "not found" in stderr.getvalue().lower()


@pytest.mark.unit
class TestCLIProfiles:
    """Tests for profile selection via CLI."""
    
    def test_default_profile(self):
        """Default profile is 'default'."""
        stdout = StringIO()
        with mock.patch("sys.stdout", stdout):
            result = main(["--formula", "p -> p", "--json"])
        
        assert result == 0
        data = json.loads(stdout.getvalue())
        assert data["config"]["profile"] == "default"
    
    def test_custom_profile(self):
        """--profile selects custom profile."""
        stdout = StringIO()
        with mock.patch("sys.stdout", stdout):
            result = main(["--formula", "p -> p", "--profile", "goal_hit", "--json"])
        
        assert result == 0
        data = json.loads(stdout.getvalue())
        assert data["config"]["profile"] == "goal_hit"


@pytest.mark.unit
class TestCLINegativeControl:
    """Tests for --negative-control flag."""
    
    def test_negative_control_flag(self):
        """--negative-control sets negative_control in config."""
        stdout = StringIO()
        with mock.patch("sys.stdout", stdout):
            result = main(["--formula", "p -> p", "--negative-control", "--json"])
        
        assert result == 0
        data = json.loads(stdout.getvalue())
        
        assert data["config"]["negative_control"] is True
        assert data["results"][0]["reason"] == "negative_control"
        assert data["results"][0]["verified"] is False
    
    def test_negative_control_all_abstain(self):
        """--negative-control makes all results negative_control."""
        stdout = StringIO()
        with mock.patch("sys.stdout", stdout):
            result = main([
                "--formula", "p -> p",
                "--count", "10",
                "--negative-control",
                "--json"
            ])
        
        assert result == 0
        data = json.loads(stdout.getvalue())
        
        for r in data["results"]:
            assert r["verified"] is False
            assert r["reason"] == "negative_control"


@pytest.mark.unit
class TestCLISummary:
    """Tests for --summary flag."""
    
    def test_summary_flag_human(self):
        """--summary shows summary in human output."""
        stdout = StringIO()
        with mock.patch("sys.stdout", stdout):
            result = main(["--formula", "p -> p", "--count", "5", "--summary"])
        
        assert result == 0
        output = stdout.getvalue()
        assert "SUMMARY" in output
        assert "Total" in output
    
    def test_summary_flag_json(self):
        """--summary includes summary in JSON output."""
        stdout = StringIO()
        with mock.patch("sys.stdout", stdout):
            result = main([
                "--formula", "p -> p",
                "--count", "10",
                "--summary",
                "--json"
            ])
        
        assert result == 0
        data = json.loads(stdout.getvalue())
        
        assert "summary" in data
        assert "total" in data["summary"]
        assert data["summary"]["total"] == 10


@pytest.mark.unit
class TestCLIExitCodes:
    """Tests for CLI exit codes."""
    
    def test_success_exits_zero(self):
        """Successful verification exits with code 0."""
        with mock.patch("sys.stdout", new_callable=StringIO):
            result = main(["--formula", "p -> p"])
        assert result == 0
    
    def test_coverage_exits_zero(self):
        """--coverage exits with code 0."""
        with mock.patch("sys.stdout", new_callable=StringIO):
            result = main(["--coverage"])
        assert result == 0
    
    def test_missing_formula_exits_one(self):
        """Missing required formula exits with code 1."""
        with mock.patch("sys.stderr", new_callable=StringIO):
            result = main([])
        assert result == 1
    
    def test_invalid_file_exits_one(self):
        """Invalid file path exits with code 1."""
        with mock.patch("sys.stderr", new_callable=StringIO):
            result = main(["--formulas-file", "/no/such/file.txt"])
        assert result == 1


@pytest.mark.unit
class TestCLIOptions:
    """Tests for various CLI options."""
    
    def test_timeout_ms_option(self):
        """--timeout-ms sets timeout configuration."""
        stdout = StringIO()
        with mock.patch("sys.stdout", stdout):
            result = main([
                "--formula", "p -> p",
                "--timeout-ms", "200",
                "--json"
            ])
        
        assert result == 0
        data = json.loads(stdout.getvalue())
        assert data["config"]["timeout_ms"] == 200
    
    def test_seed_option(self):
        """--seed sets seed configuration."""
        stdout = StringIO()
        with mock.patch("sys.stdout", stdout):
            result = main([
                "--formula", "p -> p",
                "--seed", "42",
                "--json"
            ])
        
        assert result == 0
        data = json.loads(stdout.getvalue())
        assert data["config"]["seed"] == 42
    
    def test_enable_crashes_option(self):
        """--enable-crashes sets enable_crashes configuration."""
        stdout = StringIO()
        with mock.patch("sys.stdout", stdout):
            result = main([
                "--formula", "p -> p",
                "--enable-crashes",
                "--json"
            ])
        
        assert result == 0
        data = json.loads(stdout.getvalue())
        assert data["config"]["enable_crashes"] is True


@pytest.mark.unit
class TestCLIDeterminism:
    """Tests for CLI output determinism."""
    
    def test_same_formula_same_output(self):
        """Same formula produces same JSON output."""
        def run():
            stdout = StringIO()
            with mock.patch("sys.stdout", stdout):
                main(["--formula", "p -> q", "--seed", "42", "--json"])
            return json.loads(stdout.getvalue())
        
        data1 = run()
        data2 = run()
        
        assert data1["results"] == data2["results"]
    
    def test_deterministic_across_runs(self):
        """Multiple invocations produce identical results."""
        results = []
        
        for _ in range(3):
            stdout = StringIO()
            with mock.patch("sys.stdout", stdout):
                main([
                    "--formula", "(p -> q) -> (~q -> ~p)",
                    "--seed", "123",
                    "--json"
                ])
            results.append(json.loads(stdout.getvalue())["results"][0])
        
        assert results[0] == results[1] == results[2]

