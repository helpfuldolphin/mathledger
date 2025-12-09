#!/usr/bin/env python3
"""
==============================================================================
PHASE II — SYNTHETIC TEST DATA ONLY
==============================================================================

Test Suite for Scenario Registry and CI Sweep
----------------------------------------------

Tests for:
    - Registry loading and validation
    - Registry enforcement (15 scenarios, categories sorted)
    - Sweep behavior with mocked registry
    - CI sweep modes (--schema-only, --ci-only)
    - Contract export format and stability
    - Malformed distribution failure handling

NOT derived from real derivations; NOT part of Evidence Pack.

==============================================================================
"""

import json
import tempfile
from pathlib import Path
from typing import Dict
from unittest.mock import patch, MagicMock

import pytest

import sys
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.synthetic_uplift.noise_models import SAFETY_LABEL


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory(prefix="sweep_test_") as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_registry():
    """Create a minimal mock registry for testing."""
    return {
        "label": SAFETY_LABEL,
        "version": "1.0",
        "categories": {
            "uplift": {"description": "Test uplift"},
            "drift": {"description": "Test drift"},
        },
        "scenarios": {
            "synthetic_test_null": {
                "version": "1.0",
                "description": "Test null scenario",
                "category": "uplift",
                "ci_sweep_included": True,
                "parameters": {
                    "seed": 42,
                    "num_cycles": 100,
                    "probabilities": {
                        "baseline": {"class_a": 0.5},
                        "rfl": {"class_a": 0.5}
                    },
                    "drift": {"mode": "none"},
                    "correlation": {"rho": 0.0},
                    "rare_events": []
                }
            },
            "synthetic_test_drift": {
                "version": "1.0",
                "description": "Test drift scenario",
                "category": "drift",
                "ci_sweep_included": False,
                "parameters": {
                    "seed": 42,
                    "num_cycles": 100,
                    "probabilities": {"baseline": {"class_a": 0.6}},
                    "drift": {"mode": "cyclical", "amplitude": 0.1, "period": 50},
                    "correlation": {"rho": 0.0},
                    "rare_events": []
                }
            }
        },
        "ci_sweep_scenarios": ["synthetic_test_null"]
    }


# ==============================================================================
# REGISTRY TESTS
# ==============================================================================

class TestScenarioRegistry:
    """Tests for scenario registry."""
    
    def test_registry_file_exists(self):
        """Registry file should exist."""
        registry_path = Path(__file__).parents[1] / "scenario_registry.json"
        assert registry_path.exists(), f"Registry not found at {registry_path}"
    
    def test_registry_valid_json(self):
        """Registry should be valid JSON."""
        registry_path = Path(__file__).parents[1] / "scenario_registry.json"
        with open(registry_path, "r", encoding="utf-8") as f:
            registry = json.load(f)
        
        assert "label" in registry
        assert registry["label"] == SAFETY_LABEL
    
    def test_registry_has_required_fields(self):
        """Registry should have all required fields."""
        registry_path = Path(__file__).parents[1] / "scenario_registry.json"
        with open(registry_path, "r", encoding="utf-8") as f:
            registry = json.load(f)
        
        assert "registry_version" in registry
        assert "categories" in registry
        assert "scenarios" in registry
        assert "ci_sweep_scenarios" in registry
    
    def test_registry_scenarios_have_required_fields(self):
        """Each scenario should have required fields."""
        registry_path = Path(__file__).parents[1] / "scenario_registry.json"
        with open(registry_path, "r", encoding="utf-8") as f:
            registry = json.load(f)
        
        required_fields = ["version", "description", "category", "ci_sweep_included", "parameters"]
        
        for name, scenario in registry.get("scenarios", {}).items():
            for field in required_fields:
                assert field in scenario, f"Scenario {name} missing field: {field}"
    
    def test_registry_ci_sweep_scenarios_exist(self):
        """CI sweep scenarios should exist in scenarios."""
        registry_path = Path(__file__).parents[1] / "scenario_registry.json"
        with open(registry_path, "r", encoding="utf-8") as f:
            registry = json.load(f)
        
        scenarios = set(registry.get("scenarios", {}).keys())
        ci_sweep = registry.get("ci_sweep_scenarios", [])
        
        for name in ci_sweep:
            assert name in scenarios, f"CI sweep scenario not in registry: {name}"
    
    def test_registry_all_scenarios_start_with_synthetic(self):
        """All scenario names should start with 'synthetic_'."""
        registry_path = Path(__file__).parents[1] / "scenario_registry.json"
        with open(registry_path, "r", encoding="utf-8") as f:
            registry = json.load(f)
        
        for name in registry.get("scenarios", {}).keys():
            assert name.startswith("synthetic_"), f"Scenario name must start with 'synthetic_': {name}"


# ==============================================================================
# SWEEP TESTS
# ==============================================================================

class TestScenarioSweep:
    """Tests for scenario sweep harness."""
    
    def test_sweep_dry_run(self, temp_output_dir):
        """Dry run should validate without generating."""
        from experiments.synthetic_uplift.run_scenario_sweep import run_sweep
        
        result = run_sweep(
            scenarios=["synthetic_null_uplift"],
            out_dir=temp_output_dir,
            verbose=False,
            dry_run=True,
        )
        
        assert result.scenarios_run == 1
        assert result.scenarios_passed == 1
        assert result.scenarios_failed == 0
    
    def test_sweep_detects_unknown_scenario(self, temp_output_dir):
        """Sweep should fail on unknown scenarios."""
        from experiments.synthetic_uplift.run_scenario_sweep import run_sweep
        
        result = run_sweep(
            scenarios=["synthetic_nonexistent_xyz"],
            out_dir=temp_output_dir,
            verbose=False,
            dry_run=True,
        )
        
        assert result.scenarios_failed == 1
        assert any("Unknown scenario" in e for e in result.errors)
    
    def test_sweep_generates_logs(self, temp_output_dir):
        """Full sweep should generate log files."""
        from experiments.synthetic_uplift.run_scenario_sweep import run_sweep
        
        result = run_sweep(
            scenarios=["synthetic_null_uplift"],
            out_dir=temp_output_dir,
            verbose=False,
            dry_run=False,
        )
        
        # Check files were created (primary validation)
        scenario_dir = temp_output_dir / "synthetic_null_uplift"
        assert scenario_dir.exists()
        
        baseline_path = scenario_dir / "synthetic_null_uplift_baseline.jsonl"
        rfl_path = scenario_dir / "synthetic_null_uplift_rfl.jsonl"
        
        assert baseline_path.exists()
        assert rfl_path.exists()
        
        # Check that generation results (not analysis) succeeded
        generation_results = [r for r in result.results if r.mode in ("baseline", "rfl")]
        assert all(r.success for r in generation_results), "Generation should succeed"
    
    def test_sweep_report_written(self, temp_output_dir):
        """Sweep should write a report file."""
        from experiments.synthetic_uplift.run_scenario_sweep import (
            run_sweep,
            write_sweep_report,
        )
        
        result = run_sweep(
            scenarios=["synthetic_null_uplift"],
            out_dir=temp_output_dir,
            verbose=False,
            dry_run=True,
        )
        
        report_path = temp_output_dir / "test_report.json"
        write_sweep_report(result, report_path)
        
        assert report_path.exists()
        
        with open(report_path) as f:
            report = json.load(f)
        
        assert report["label"] == SAFETY_LABEL
        assert "summary" in report
        assert report["summary"]["scenarios_run"] == 1


# ==============================================================================
# CONTRACT EXPORT TESTS
# ==============================================================================

class TestContractExport:
    """Tests for contract export functionality."""
    
    def test_export_contracts_creates_file(self, temp_output_dir):
        """Export contracts should create output file."""
        from experiments.synthetic_uplift.universe_browser import cmd_export_contracts
        
        out_path = temp_output_dir / "contracts.json"
        
        # Create mock args
        args = MagicMock()
        args.out = str(out_path)
        args.category = None
        args.ci_only = False
        
        result = cmd_export_contracts(args)
        
        assert result == 0
        assert out_path.exists()
    
    def test_export_contracts_has_required_fields(self, temp_output_dir):
        """Exported contracts should have required fields."""
        from experiments.synthetic_uplift.universe_browser import cmd_export_contracts
        
        out_path = temp_output_dir / "contracts.json"
        
        args = MagicMock()
        args.out = str(out_path)
        args.category = None
        args.ci_only = False
        
        cmd_export_contracts(args)
        
        with open(out_path) as f:
            contracts = json.load(f)
        
        assert contracts["label"] == SAFETY_LABEL
        assert "scenarios" in contracts
        
        # Check first scenario has required contract fields
        if contracts["scenarios"]:
            first = list(contracts["scenarios"].values())[0]
            assert "probability_ranges" in first
            assert "drift_characteristics" in first
            assert "correlation_settings" in first
            assert "rare_event_definitions" in first
    
    def test_export_contracts_ci_only_filter(self, temp_output_dir):
        """CI-only filter should reduce output."""
        from experiments.synthetic_uplift.universe_browser import cmd_export_contracts
        from experiments.synthetic_uplift.run_scenario_sweep import load_registry
        
        # Get CI sweep count
        registry = load_registry()
        ci_count = len(registry.get("ci_sweep_scenarios", []))
        
        # Export all
        all_path = temp_output_dir / "all.json"
        args_all = MagicMock()
        args_all.out = str(all_path)
        args_all.category = None
        args_all.ci_only = False
        cmd_export_contracts(args_all)
        
        # Export CI only
        ci_path = temp_output_dir / "ci.json"
        args_ci = MagicMock()
        args_ci.out = str(ci_path)
        args_ci.category = None
        args_ci.ci_only = True
        cmd_export_contracts(args_ci)
        
        with open(all_path) as f:
            all_contracts = json.load(f)
        with open(ci_path) as f:
            ci_contracts = json.load(f)
        
        assert len(ci_contracts["scenarios"]) == ci_count
        assert len(ci_contracts["scenarios"]) <= len(all_contracts["scenarios"])


# ==============================================================================
# REGISTRY CLI TESTS
# ==============================================================================

class TestRegistryCLI:
    """Tests for registry CLI command."""
    
    def test_registry_text_format(self, capsys):
        """Registry text format should print table."""
        from experiments.synthetic_uplift.universe_browser import cmd_registry
        
        args = MagicMock()
        args.format = "text"
        
        result = cmd_registry(args)
        
        assert result == 0
        
        captured = capsys.readouterr()
        assert "SCENARIO REGISTRY" in captured.out
        assert "synthetic_" in captured.out
    
    def test_registry_json_format(self, capsys):
        """Registry JSON format should be valid JSON."""
        from experiments.synthetic_uplift.universe_browser import cmd_registry
        
        args = MagicMock()
        args.format = "json"
        
        result = cmd_registry(args)
        
        assert result == 0
        
        captured = capsys.readouterr()
        # Should be valid JSON
        parsed = json.loads(captured.out)
        assert "scenarios" in parsed


# ==============================================================================
# REGISTRY ENFORCEMENT TESTS
# ==============================================================================

class TestRegistryEnforcement:
    """Tests for canonical registry enforcement."""
    
    def test_registry_has_exactly_15_scenarios(self):
        """Registry must contain exactly 15 scenarios."""
        from experiments.synthetic_uplift.run_scenario_sweep import load_registry
        
        registry = load_registry()
        scenarios = registry.get("scenarios", {})
        expected = registry.get("expected_scenario_count", 15)
        
        assert len(scenarios) == expected, (
            f"Registry must have exactly {expected} scenarios, got {len(scenarios)}"
        )
    
    def test_registry_categories_sorted(self):
        """Registry categories must be in sorted order."""
        from experiments.synthetic_uplift.run_scenario_sweep import load_registry
        
        registry = load_registry()
        categories = list(registry.get("categories", {}).keys())
        
        assert categories == sorted(categories), (
            f"Categories must be sorted. Got: {categories}"
        )
    
    def test_registry_has_version_field(self):
        """Registry must have registry_version field."""
        from experiments.synthetic_uplift.run_scenario_sweep import load_registry
        
        registry = load_registry()
        
        assert "registry_version" in registry, "Registry must have registry_version"
        assert registry["registry_version"], "registry_version cannot be empty"
    
    def test_all_scenarios_have_category(self):
        """All scenarios must have a valid category."""
        from experiments.synthetic_uplift.run_scenario_sweep import load_registry
        
        registry = load_registry()
        categories = set(registry.get("categories", {}).keys())
        scenarios = registry.get("scenarios", {})
        
        for name, scenario in scenarios.items():
            assert "category" in scenario, f"Scenario '{name}' missing category"
            assert scenario["category"] in categories, (
                f"Scenario '{name}' has invalid category: {scenario['category']}"
            )
    
    def test_ci_sweep_scenarios_exist(self):
        """All CI sweep scenarios must exist in registry."""
        from experiments.synthetic_uplift.run_scenario_sweep import load_registry
        
        registry = load_registry()
        ci_sweep = registry.get("ci_sweep_scenarios", [])
        scenarios = registry.get("scenarios", {})
        
        for name in ci_sweep:
            assert name in scenarios, f"CI sweep scenario not in registry: {name}"
    
    def test_all_categories_have_scenarios(self):
        """Each category should have at least one scenario."""
        from experiments.synthetic_uplift.run_scenario_sweep import load_registry
        
        registry = load_registry()
        categories = set(registry.get("categories", {}).keys())
        scenarios = registry.get("scenarios", {})
        
        used_categories = {s.get("category") for s in scenarios.values()}
        
        for cat in categories:
            assert cat in used_categories, f"Category '{cat}' has no scenarios"
    
    def test_registry_enforcement_module_loads(self):
        """Registry enforcement module should load without error."""
        from experiments.synthetic_uplift.registry_enforcement import (
            enforce_registry,
            validate_registry_structure,
            load_registry,
        )
        
        valid, result = enforce_registry()
        assert valid, f"Registry validation failed: {result.errors}"
    
    def test_registry_validation_detects_errors(self):
        """Registry validation should detect structural errors."""
        from experiments.synthetic_uplift.registry_enforcement import (
            validate_registry_structure,
        )
        
        # Create invalid registry
        invalid_registry = {
            "label": "WRONG LABEL",
            "scenarios": {},
        }
        
        result = validate_registry_structure(invalid_registry)
        
        assert not result.valid
        assert len(result.errors) > 0


# ==============================================================================
# SCHEMA-ONLY MODE TESTS
# ==============================================================================

class TestSchemaOnlyMode:
    """Tests for --schema-only sweep mode."""
    
    def test_schema_validation_success(self, temp_output_dir):
        """Schema-only validation should pass for valid scenarios."""
        from experiments.synthetic_uplift.run_scenario_sweep import (
            run_schema_validation,
            get_ci_sweep_scenarios,
        )
        
        scenarios = get_ci_sweep_scenarios()
        result = run_schema_validation(scenarios=scenarios, verbose=False)
        
        assert result.scenarios_passed == len(scenarios)
        assert result.scenarios_failed == 0
    
    def test_schema_validation_detects_invalid(self, temp_output_dir):
        """Schema-only should detect invalid scenarios."""
        from experiments.synthetic_uplift.run_scenario_sweep import (
            run_schema_validation,
        )
        
        # Use a non-existent scenario
        result = run_schema_validation(
            scenarios=["synthetic_nonexistent"],
            verbose=False,
        )
        
        assert result.scenarios_failed == 1
    
    def test_validate_scenario_schema_function(self):
        """validate_scenario_schema should check required fields."""
        from experiments.synthetic_uplift.run_scenario_sweep import (
            validate_scenario_schema,
        )
        
        # Valid scenario
        valid, errors = validate_scenario_schema("synthetic_null_uplift")
        assert valid, f"Expected valid, got errors: {errors}"
        
        # Non-existent
        valid, errors = validate_scenario_schema("synthetic_fake")
        assert not valid


# ==============================================================================
# CONTRACT STABILITY TESTS
# ==============================================================================

class TestContractStability:
    """Tests for contract re-export stability."""
    
    def test_reexport_produces_identical_contracts(self, temp_output_dir):
        """Re-exporting contracts should produce identical files (except timestamp)."""
        from experiments.synthetic_uplift.universe_browser import cmd_export_contracts
        from experiments.synthetic_uplift.contract_schema import contracts_are_identical
        
        # First export
        path1 = temp_output_dir / "contracts1.json"
        args1 = MagicMock()
        args1.out = str(path1)
        args1.category = None
        args1.ci_only = True
        cmd_export_contracts(args1)
        
        # Second export
        path2 = temp_output_dir / "contracts2.json"
        args2 = MagicMock()
        args2.out = str(path2)
        args2.category = None
        args2.ci_only = True
        cmd_export_contracts(args2)
        
        # Compare
        with open(path1) as f:
            contract1 = json.load(f)
        with open(path2) as f:
            contract2 = json.load(f)
        
        identical, differences = contracts_are_identical(
            contract1, contract2, ignore_timestamp=True
        )
        
        assert identical, f"Contracts differ: {differences}"
    
    def test_contract_schema_validation(self, temp_output_dir):
        """Exported contracts should pass schema validation."""
        from experiments.synthetic_uplift.universe_browser import cmd_export_contracts
        from experiments.synthetic_uplift.contract_schema import (
            validate_contract_file,
        )
        
        out_path = temp_output_dir / "contracts.json"
        args = MagicMock()
        args.out = str(out_path)
        args.category = None
        args.ci_only = False
        cmd_export_contracts(args)
        
        result = validate_contract_file(out_path)
        
        assert result.valid, f"Contract validation failed: {result.errors}"
    
    def test_ci_only_export_produces_6_scenarios(self, temp_output_dir):
        """CI-only export should produce exactly 6 scenarios."""
        from experiments.synthetic_uplift.universe_browser import cmd_export_contracts
        
        out_path = temp_output_dir / "contracts.json"
        args = MagicMock()
        args.out = str(out_path)
        args.category = None
        args.ci_only = True
        cmd_export_contracts(args)
        
        with open(out_path) as f:
            contracts = json.load(f)
        
        assert len(contracts["scenarios"]) == 6, (
            f"Expected 6 CI scenarios, got {len(contracts['scenarios'])}"
        )
    
    def test_contract_fingerprint_deterministic(self, temp_output_dir):
        """Contract fingerprint should be deterministic."""
        from experiments.synthetic_uplift.universe_browser import cmd_export_contracts
        from experiments.synthetic_uplift.contract_schema import get_contract_fingerprint
        
        # Export twice
        path1 = temp_output_dir / "c1.json"
        path2 = temp_output_dir / "c2.json"
        
        for path in [path1, path2]:
            args = MagicMock()
            args.out = str(path)
            args.category = None
            args.ci_only = True
            cmd_export_contracts(args)
        
        with open(path1) as f:
            c1 = json.load(f)
        with open(path2) as f:
            c2 = json.load(f)
        
        fp1 = get_contract_fingerprint(c1)
        fp2 = get_contract_fingerprint(c2)
        
        assert fp1 == fp2, "Fingerprints should match"
    
    def test_contract_has_registry_version(self, temp_output_dir):
        """Exported contracts should include registry_version."""
        from experiments.synthetic_uplift.universe_browser import cmd_export_contracts
        
        out_path = temp_output_dir / "contracts.json"
        args = MagicMock()
        args.out = str(out_path)
        args.category = None
        args.ci_only = False
        cmd_export_contracts(args)
        
        with open(out_path) as f:
            contracts = json.load(f)
        
        assert "registry_version" in contracts
        assert contracts["registry_version"], "registry_version should not be empty"


# ==============================================================================
# SWEEP ERROR HANDLING TESTS
# ==============================================================================

class TestSweepErrorHandling:
    """Tests for clean error handling in sweep."""
    
    def test_malformed_scenario_clean_error(self, temp_output_dir):
        """Malformed scenario should produce clean error, not stack trace."""
        from experiments.synthetic_uplift.run_scenario_sweep import (
            run_schema_validation,
        )
        
        # Non-existent scenario
        result = run_schema_validation(
            scenarios=["synthetic_malformed_test"],
            verbose=False,
        )
        
        assert result.scenarios_failed == 1
        assert len(result.errors) > 0
        # Error should be clean message, not traceback
        assert "Traceback" not in str(result.errors)
    
    def test_repeated_sweeps_deterministic(self, temp_output_dir):
        """Repeated sweeps should produce consistent results."""
        from experiments.synthetic_uplift.run_scenario_sweep import (
            run_sweep,
        )
        
        results = []
        for _ in range(3):
            result = run_sweep(
                scenarios=["synthetic_null_uplift"],
                out_dir=temp_output_dir,
                verbose=False,
                dry_run=True,
            )
            results.append((result.scenarios_passed, result.scenarios_failed))
        
        # All results should be identical
        assert len(set(results)) == 1, "Sweep results should be deterministic"


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

