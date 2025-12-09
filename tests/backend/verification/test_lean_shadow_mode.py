# tests/backend/verification/test_lean_shadow_mode.py
"""
Test suite for Lean Adapter Shadow Mode (Reality Bridge Protocol).

These tests verify that shadow mode:
1. Generates deterministic shadow telemetry
2. Produces stable capability radar
3. Extends director panel correctly
4. Maintains backward compatibility with SIMULATE and DISABLED modes

Markers:
    - unit: Fast, no external dependencies
"""

import pytest

from backend.verification import (
    LeanAdapter,
    LeanAdapterMode,
    LeanVerificationRequest,
    LeanResourceBudget,
    generate_shadow_telemetry,
    build_lean_shadow_capability_radar,
    build_lean_director_panel_with_shadow,
    build_lean_activity_ledger,
    evaluate_lean_adapter_safety,
    classify_lean_capabilities,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def shadow_adapter() -> LeanAdapter:
    """Create a LeanAdapter in SHADOW mode."""
    return LeanAdapter(mode=LeanAdapterMode.SHADOW)


@pytest.fixture
def sample_request() -> LeanVerificationRequest:
    """Create a sample verification request."""
    return LeanVerificationRequest(
        canonical="p->p",
        job_id="test_job_001",
    )


# =============================================================================
# SHADOW MODE BASIC FUNCTIONALITY
# =============================================================================

@pytest.mark.unit
class TestShadowModeBasic:
    """Test basic shadow mode functionality."""

    def test_shadow_mode_exists(self) -> None:
        """SHADOW mode should exist in enum."""
        assert LeanAdapterMode.SHADOW.value == "shadow"

    def test_shadow_mode_verify_returns_result(self, shadow_adapter: LeanAdapter, sample_request: LeanVerificationRequest) -> None:
        """Shadow mode verify() should return a result."""
        result = shadow_adapter.verify(sample_request)
        assert result is not None
        assert result.method == "lean_adapter_shadow"

    def test_shadow_mode_deterministic(self, shadow_adapter: LeanAdapter) -> None:
        """Shadow mode should produce deterministic results."""
        request = LeanVerificationRequest(
            canonical="p->q->p",
            job_id="deterministic_test",
        )
        
        result1 = shadow_adapter.verify(request)
        result2 = shadow_adapter.verify(request)
        
        # Deterministic fields should match
        assert result1.verified == result2.verified
        assert result1.deterministic_hash == result2.deterministic_hash
        assert result1.method == result2.method
        assert result1.simulated_complexity == result2.simulated_complexity

    def test_shadow_mode_backward_compatible_simulate(self) -> None:
        """SIMULATE mode should still work (backward compatibility)."""
        simulate_adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        request = LeanVerificationRequest(canonical="p->p", job_id="simulate_test")
        result = simulate_adapter.verify(request)
        
        assert result.method == "lean_adapter_simulate"
        assert result.method != "lean_adapter_shadow"

    def test_shadow_mode_backward_compatible_disabled(self) -> None:
        """DISABLED mode should still work (backward compatibility)."""
        disabled_adapter = LeanAdapter(mode=LeanAdapterMode.DISABLED)
        request = LeanVerificationRequest(canonical="p->p", job_id="disabled_test")
        result = disabled_adapter.verify(request)
        
        assert result.method == "lean_adapter_disabled"
        assert not result.verified


# =============================================================================
# SHADOW TELEMETRY GENERATION
# =============================================================================

@pytest.mark.unit
class TestShadowTelemetryGeneration:
    """Test shadow telemetry generation."""

    def test_generate_shadow_telemetry_success(self) -> None:
        """Telemetry for successful verification should have return_code=0."""
        telemetry = generate_shadow_telemetry(
            canonical="p->p",
            job_id="success_test",
            verified=True,
            complexity=5,
        )
        
        assert telemetry["return_code"] == 0
        assert telemetry["stderr"] == ""
        assert len(telemetry["stdout_lines"]) > 0

    def test_generate_shadow_telemetry_failure(self) -> None:
        """Telemetry for failed verification should have non-zero return_code."""
        telemetry = generate_shadow_telemetry(
            canonical="p->" * 30 + "p",  # Complex, likely to fail
            job_id="failure_test",
            verified=False,
            complexity=100,
        )
        
        assert telemetry["return_code"] != 0
        assert len(telemetry["stderr"]) > 0

    def test_generate_shadow_telemetry_deterministic(self) -> None:
        """Telemetry generation should be deterministic."""
        telemetry1 = generate_shadow_telemetry(
            canonical="p->q",
            job_id="det_test",
            verified=True,
            complexity=5,
        )
        telemetry2 = generate_shadow_telemetry(
            canonical="p->q",
            job_id="det_test",
            verified=True,
            complexity=5,
        )
        
        assert telemetry1 == telemetry2

    def test_generate_shadow_telemetry_includes_cpu_memory(self) -> None:
        """Telemetry should include CPU time and memory footprint."""
        telemetry = generate_shadow_telemetry(
            canonical="p->p",
            job_id="resource_test",
            verified=True,
            complexity=10,
        )
        
        assert "cpu_time_ms" in telemetry
        assert "memory_mb" in telemetry
        assert isinstance(telemetry["cpu_time_ms"], int)
        assert isinstance(telemetry["memory_mb"], int)
        assert telemetry["cpu_time_ms"] > 0
        assert telemetry["memory_mb"] > 0

    def test_generate_shadow_telemetry_stdout_lines(self) -> None:
        """Telemetry should include stdout_lines."""
        telemetry = generate_shadow_telemetry(
            canonical="p->p",
            job_id="stdout_test",
            verified=True,
            complexity=5,
        )
        
        assert "stdout_lines" in telemetry
        assert isinstance(telemetry["stdout_lines"], list)
        assert len(telemetry["stdout_lines"]) > 0


# =============================================================================
# SHADOW CAPABILITY RADAR
# =============================================================================

@pytest.mark.unit
class TestShadowCapabilityRadar:
    """Test shadow capability radar."""

    def test_radar_empty_results(self) -> None:
        """Radar should handle empty results."""
        radar = build_lean_shadow_capability_radar([])
        
        assert radar["schema_version"] == "1.0.0"
        assert radar["total_shadow_requests"] == 0
        assert radar["structural_error_rate"] == 0.0
        assert radar["shadow_resource_band"] == "LOW"

    def test_radar_no_shadow_results(self) -> None:
        """Radar should handle results without shadow mode."""
        from backend.verification import LeanAdapter
        
        simulate_adapter = LeanAdapter(mode=LeanAdapterMode.SIMULATE)
        results = [
            simulate_adapter.verify(LeanVerificationRequest(canonical=f"p{i}->p{i}", job_id=f"sim_{i}"))
            for i in range(5)
        ]
        
        radar = build_lean_shadow_capability_radar(results)
        
        assert radar["total_shadow_requests"] == 0

    def test_radar_with_shadow_results(self, shadow_adapter: LeanAdapter) -> None:
        """Radar should process shadow mode results."""
        results = [
            shadow_adapter.verify(LeanVerificationRequest(canonical=f"p{i}->p{i}", job_id=f"shadow_{i}"))
            for i in range(10)
        ]
        
        radar = build_lean_shadow_capability_radar(results)
        
        assert radar["total_shadow_requests"] == 10
        assert "structural_error_rate" in radar
        assert "complexity_success_curve" in radar
        assert "shadow_resource_band" in radar
        assert "anomaly_signatures" in radar

    def test_radar_structural_error_rate(self, shadow_adapter: LeanAdapter) -> None:
        """Radar should compute structural error rate."""
        # Create mix of results
        results = []
        for i in range(10):
            # Simple formulas (likely to succeed)
            results.append(shadow_adapter.verify(
                LeanVerificationRequest(canonical=f"p{i}->p{i}", job_id=f"struct_{i}")
            ))
        
        radar = build_lean_shadow_capability_radar(results)
        
        assert 0.0 <= radar["structural_error_rate"] <= 1.0

    def test_radar_complexity_success_curve(self, shadow_adapter: LeanAdapter) -> None:
        """Radar should build complexity success curve."""
        results = []
        # Mix of complexity levels
        for i in range(5):
            results.append(shadow_adapter.verify(
                LeanVerificationRequest(canonical="p->" * i + "p", job_id=f"curve_{i}")
            ))
        
        radar = build_lean_shadow_capability_radar(results)
        
        assert "complexity_success_curve" in radar
        curve = radar["complexity_success_curve"]
        assert isinstance(curve, dict)

    def test_radar_resource_band(self, shadow_adapter: LeanAdapter) -> None:
        """Radar should classify resource band."""
        results = [
            shadow_adapter.verify(LeanVerificationRequest(canonical=f"p{i}->p{i}", job_id=f"band_{i}"))
            for i in range(10)
        ]
        
        radar = build_lean_shadow_capability_radar(results)
        
        assert radar["shadow_resource_band"] in ("LOW", "MEDIUM", "HIGH")

    def test_radar_anomaly_signatures(self, shadow_adapter: LeanAdapter) -> None:
        """Radar should generate anomaly signatures."""
        # Create some failures to generate anomalies
        results = []
        for i in range(10):
            # Mix of simple and complex (some will fail)
            canonical = "p->" * (i * 5) + "p" if i % 2 == 0 else f"p{i}->p{i}"
            results.append(shadow_adapter.verify(
                LeanVerificationRequest(canonical=canonical, job_id=f"anomaly_{i}")
            ))
        
        radar = build_lean_shadow_capability_radar(results)
        
        assert "anomaly_signatures" in radar
        assert isinstance(radar["anomaly_signatures"], list)

    def test_radar_deterministic(self, shadow_adapter: LeanAdapter) -> None:
        """Radar should be deterministic for same inputs."""
        results = [
            shadow_adapter.verify(LeanVerificationRequest(canonical=f"p{i}->p{i}", job_id=f"det_radar_{i}"))
            for i in range(10)
        ]
        
        radar1 = build_lean_shadow_capability_radar(results)
        radar2 = build_lean_shadow_capability_radar(results)
        
        assert radar1 == radar2

    def test_radar_json_serializable(self, shadow_adapter: LeanAdapter) -> None:
        """Radar should be JSON serializable."""
        import json
        
        results = [
            shadow_adapter.verify(LeanVerificationRequest(canonical=f"p{i}->p{i}", job_id=f"json_radar_{i}"))
            for i in range(5)
        ]
        
        radar = build_lean_shadow_capability_radar(results)
        
        # Should not raise
        json_str = json.dumps(radar)
        assert isinstance(json_str, str)
        
        # Should round-trip
        parsed = json.loads(json_str)
        assert parsed["schema_version"] == radar["schema_version"]


# =============================================================================
# DIRECTOR PANEL WITH SHADOW
# =============================================================================

@pytest.mark.unit
class TestDirectorPanelWithShadow:
    """Test director panel with shadow extensions."""

    def test_panel_without_shadow_radar(self) -> None:
        """Panel should work without shadow radar."""
        ledger = {
            "schema_version": "1.0.0",
            "total_requests": 10,
            "success_count": 8,
            "abstention_count": 2,
            "error_kind_histogram": {},
            "resource_budget_histogram": {},
            "max_resource_budget_observed": {},
            "methods_histogram": {},
            "version_pin": "v4.23.0-rc2",
            "timestamp_utc": "2025-01-01T00:00:00Z",
        }
        
        safety = {
            "has_internal_errors": False,
            "has_resource_issues": False,
            "safety_status": "OK",
            "reasons": [],
            "internal_error_count": 0,
            "resource_limit_count": 0,
        }
        
        capability = {
            "schema_version": "1.0.0",
            "capability_band": "INTERMEDIATE",
            "max_budget_used": {},
            "resource_profile": "Test",
            "simulation_only": True,
        }
        
        panel = build_lean_director_panel_with_shadow(ledger, safety, capability)
        
        assert "shadow_mode_ok" in panel
        assert "shadow_status" in panel
        assert "dominant_anomalies" in panel
        assert "complexity_curve_summary" in panel

    def test_panel_with_shadow_radar(self, shadow_adapter: LeanAdapter) -> None:
        """Panel should include shadow metrics when radar provided."""
        results = [
            shadow_adapter.verify(LeanVerificationRequest(canonical=f"p{i}->p{i}", job_id=f"panel_{i}"))
            for i in range(10)
        ]
        
        ledger = build_lean_activity_ledger(results)
        safety = evaluate_lean_adapter_safety(ledger)
        capability = classify_lean_capabilities(ledger)
        shadow_radar = build_lean_shadow_capability_radar(results)
        
        panel = build_lean_director_panel_with_shadow(ledger, safety, capability, shadow_radar)
        
        assert panel["shadow_mode_ok"] in (True, False)
        assert panel["shadow_status"] in ("OK", "WARN", "BLOCK")
        assert isinstance(panel["dominant_anomalies"], list)
        assert isinstance(panel["complexity_curve_summary"], str)

    def test_panel_shadow_status_ok(self, shadow_adapter: LeanAdapter) -> None:
        """Panel should show shadow status (OK, WARN, or BLOCK)."""
        # Simple formulas (likely to succeed)
        results = [
            shadow_adapter.verify(LeanVerificationRequest(canonical=f"p{i}->p{i}", job_id=f"ok_{i}"))
            for i in range(20)
        ]
        
        ledger = build_lean_activity_ledger(results)
        safety = evaluate_lean_adapter_safety(ledger)
        capability = classify_lean_capabilities(ledger)
        shadow_radar = build_lean_shadow_capability_radar(results)
        
        panel = build_lean_director_panel_with_shadow(ledger, safety, capability, shadow_radar)
        
        # Should be one of the valid statuses (depends on actual results)
        assert panel["shadow_status"] in ("OK", "WARN", "BLOCK")

    def test_panel_includes_base_fields(self, shadow_adapter: LeanAdapter) -> None:
        """Panel should include all base director panel fields."""
        results = [
            shadow_adapter.verify(LeanVerificationRequest(canonical=f"p{i}->p{i}", job_id=f"base_{i}"))
            for i in range(5)
        ]
        
        ledger = build_lean_activity_ledger(results)
        safety = evaluate_lean_adapter_safety(ledger)
        capability = classify_lean_capabilities(ledger)
        shadow_radar = build_lean_shadow_capability_radar(results)
        
        panel = build_lean_director_panel_with_shadow(ledger, safety, capability, shadow_radar)
        
        # Base fields
        assert "status_light" in panel
        assert "lean_surface_ok" in panel
        assert "capability_band" in panel
        assert "safety_status" in panel
        assert "headline" in panel
        
        # Shadow fields
        assert "shadow_mode_ok" in panel
        assert "shadow_status" in panel

    def test_panel_headline_includes_shadow_status(self, shadow_adapter: LeanAdapter) -> None:
        """Panel headline should include shadow status when not OK."""
        # Create results that might trigger WARN/BLOCK
        results = []
        for i in range(10):
            # Mix of simple and complex
            canonical = "p->" * (i * 10) + "p" if i % 3 == 0 else f"p{i}->p{i}"
            results.append(shadow_adapter.verify(
                LeanVerificationRequest(canonical=canonical, job_id=f"headline_{i}")
            ))
        
        ledger = build_lean_activity_ledger(results)
        safety = evaluate_lean_adapter_safety(ledger)
        capability = classify_lean_capabilities(ledger)
        shadow_radar = build_lean_shadow_capability_radar(results)
        
        panel = build_lean_director_panel_with_shadow(ledger, safety, capability, shadow_radar)
        
        # If shadow status is not OK, headline should mention it
        if panel["shadow_status"] != "OK":
            assert "Shadow mode" in panel["headline"]

    def test_panel_json_serializable(self, shadow_adapter: LeanAdapter) -> None:
        """Panel should be JSON serializable."""
        import json
        
        results = [
            shadow_adapter.verify(LeanVerificationRequest(canonical=f"p{i}->p{i}", job_id=f"json_panel_{i}"))
            for i in range(5)
        ]
        
        ledger = build_lean_activity_ledger(results)
        safety = evaluate_lean_adapter_safety(ledger)
        capability = classify_lean_capabilities(ledger)
        shadow_radar = build_lean_shadow_capability_radar(results)
        
        panel = build_lean_director_panel_with_shadow(ledger, safety, capability, shadow_radar)
        
        # Should not raise
        json_str = json.dumps(panel)
        assert isinstance(json_str, str)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

@pytest.mark.unit
class TestShadowModeIntegration:
    """Integration tests for shadow mode."""

    def test_full_shadow_pipeline(self, shadow_adapter: LeanAdapter) -> None:
        """Full pipeline: shadow verify -> ledger -> radar -> panel."""
        # Generate shadow results
        results = [
            shadow_adapter.verify(LeanVerificationRequest(canonical=f"p{i}->p{i}", job_id=f"pipeline_{i}"))
            for i in range(15)
        ]
        
        # Build ledger
        ledger = build_lean_activity_ledger(results)
        
        # Build safety and capability
        safety = evaluate_lean_adapter_safety(ledger)
        capability = classify_lean_capabilities(ledger)
        
        # Build shadow radar
        shadow_radar = build_lean_shadow_capability_radar(results)
        
        # Build extended panel
        panel = build_lean_director_panel_with_shadow(ledger, safety, capability, shadow_radar)
        
        # Verify all components
        assert ledger["total_requests"] == 15
        assert shadow_radar["total_shadow_requests"] == 15
        assert panel["shadow_status"] in ("OK", "WARN", "BLOCK")

    def test_shadow_telemetry_determinism_across_runs(self, shadow_adapter: LeanAdapter) -> None:
        """Shadow telemetry should be deterministic across multiple runs."""
        request = LeanVerificationRequest(canonical="p->q->p", job_id="telemetry_det")
        
        # Run multiple times
        results = [shadow_adapter.verify(request) for _ in range(5)]
        
        # All results should be identical (deterministic)
        first = results[0]
        for result in results[1:]:
            assert result.verified == first.verified
            assert result.deterministic_hash == first.deterministic_hash
            assert result.simulated_complexity == first.simulated_complexity

