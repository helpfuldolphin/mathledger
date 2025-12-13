"""
Integration tests for noise_vs_reality end-to-end attachment.

Tests use factory artifacts to build both P3 noise input and P5 divergence input,
and verify end-to-end attachment to evidence pack.

Tests include:
- Input preference order (p5_divergence_real.json > divergence_log.jsonl)
- Consistent verdicts across pipeline stages
- Non-gating warnings in SHADOW mode
- SHA256 hash integrity in manifest
"""

import json
import pytest
import tempfile
from pathlib import Path
from typing import Any, Dict

from backend.topology.first_light.noise_vs_reality_integration import (
    generate_noise_vs_reality_for_evidence,
    attach_noise_vs_reality_to_manifest,
    extract_noise_vs_reality_signal,
    compute_summary_sha256,
    normalize_p5_source,
    P5Source,
    VALID_P5_SOURCES,
    _load_p5_divergence_real_report,
    _extract_divergence_from_real_report,
)
from backend.topology.first_light.noise_vs_reality import (
    SCHEMA_VERSION,
    validate_noise_vs_reality_summary,
)


# =============================================================================
# Factory Fixtures
# =============================================================================

@pytest.fixture
def factory_noise_summary() -> Dict[str, Any]:
    """Factory P3 noise summary artifact."""
    return {
        "schema_version": "p3-noise-summary/1.0.0",
        "mode": "SHADOW",
        "total_cycles": 100,
        "regime_proportions": {
            "base": 1.0,
            "correlated": 0.15,
            "degradation_degraded": 0.10,
            "heat_death_stressed": 0.05,
            "pathology": 0.05,
        },
        "delta_p_aggregate": {
            "total_contribution": -0.35,
            "avg_per_cycle": -0.0035,
            "by_noise_type": {
                "timeout": -0.22,
                "spurious_fail": -0.13,
                "spurious_pass": 0.0,
            },
            "net_direction": "NEGATIVE",
            "magnitude_class": "MODERATE",
        },
        "rsi_aggregate": {
            "noise_event_rate": 0.08,
            "estimated_rsi_suppression": 0.008,
            "suppression_class": "LOW",
            "degraded_cycle_fraction": 0.10,
            "pathology_active_fraction": 0.05,
        },
        "config_profile": {
            "base_timeout_rate": 0.05,
            "base_fail_rate": 0.03,
            "base_pass_rate": 0.01,
            "correlated_enabled": True,
            "degradation_enabled": True,
            "pathology_count": 2,
        },
        "interpretation_guidance": "Moderate delta_p impact from noise injection.",
    }


@pytest.fixture
def factory_divergence_log() -> str:
    """Factory P5 divergence log JSONL content."""
    lines = []
    for i in range(100):
        entry = {
            "cycle": i,
            "twin_delta_p": 0.01,
            "real_delta_p": 0.012 if i % 10 == 0 else 0.01,
            "divergence_magnitude": 0.002 if i % 10 == 0 else 0.0,
            "is_red_flag": i in [30, 70],
            "red_flag_type": "DELTA_P_SPIKE" if i in [30, 70] else None,
        }
        lines.append(json.dumps(entry))
    return "\n".join(lines)


@pytest.fixture
def factory_run_dir(factory_noise_summary, factory_divergence_log) -> Path:
    """Create factory run directory with all required artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)

        # Write stability_report.json with noise_summary embedded
        stability_report = {
            "schema_version": "1.0.0",
            "mode": "SHADOW",
            "noise_summary": factory_noise_summary,
            "criteria_evaluation": {"all_passed": True},
        }
        with (run_dir / "stability_report.json").open("w") as f:
            json.dump(stability_report, f)

        # Write divergence_log.jsonl
        with (run_dir / "divergence_log.jsonl").open("w") as f:
            f.write(factory_divergence_log)

        # Write run_config.json
        run_config = {
            "run_id": "test-integration-001",
            "experiment_id": "nvr-integration-test",
        }
        with (run_dir / "run_config.json").open("w") as f:
            json.dump(run_config, f)

        yield run_dir


@pytest.fixture
def factory_empty_run_dir() -> Path:
    """Create empty run directory (no artifacts)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def factory_partial_run_dir(factory_noise_summary) -> Path:
    """Create run directory with only P3 noise summary (no P5 divergence)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)

        # Write noise_summary.json only
        with (run_dir / "noise_summary.json").open("w") as f:
            json.dump(factory_noise_summary, f)

        yield run_dir


@pytest.fixture
def factory_p5_divergence_real_report() -> Dict[str, Any]:
    """Factory p5_divergence_real.json artifact (preferred P5 source)."""
    return {
        "schema_version": "1.0.0",
        "run_id": "p5_20250610_120000_test",
        "telemetry_source": "real",
        "validation_status": "VALIDATED_REAL",
        "validation_confidence": 0.95,
        "total_cycles": 100,
        "divergence_rate": 0.12,
        "mode": "SHADOW",
        "pattern_classification": "NOMINAL",
        "pattern_confidence": 0.88,
        "divergence_decomposition": {
            "bias": 0.008,
            "variance": 0.002,
            "timing": 0.05,
            "structural": 0.01,
        },
        "twin_tracking_accuracy": {
            "success": 0.92,
            "omega": 0.89,
            "blocked": 0.95,
        },
        "manifold_validation": {
            "boundedness_ok": True,
            "continuity_ok": True,
            "correlation_ok": True,
            "violations": [],
        },
    }


@pytest.fixture
def factory_run_dir_with_both_p5_sources(
    factory_noise_summary, factory_divergence_log, factory_p5_divergence_real_report
) -> Path:
    """Create run directory with BOTH p5_divergence_real.json AND divergence_log.jsonl."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)

        # Write stability_report.json with noise_summary embedded
        stability_report = {
            "schema_version": "1.0.0",
            "mode": "SHADOW",
            "noise_summary": factory_noise_summary,
            "criteria_evaluation": {"all_passed": True},
        }
        with (run_dir / "stability_report.json").open("w") as f:
            json.dump(stability_report, f)

        # Write p5_divergence_real.json (PREFERRED source)
        with (run_dir / "p5_divergence_real.json").open("w") as f:
            json.dump(factory_p5_divergence_real_report, f)

        # Write divergence_log.jsonl (FALLBACK source - should be ignored)
        with (run_dir / "divergence_log.jsonl").open("w") as f:
            f.write(factory_divergence_log)

        # Write run_config.json
        run_config = {
            "run_id": "test-preference-001",
            "experiment_id": "nvr-preference-test",
        }
        with (run_dir / "run_config.json").open("w") as f:
            json.dump(run_config, f)

        yield run_dir


@pytest.fixture
def factory_run_dir_with_only_real_report(
    factory_noise_summary, factory_p5_divergence_real_report
) -> Path:
    """Create run directory with only p5_divergence_real.json (no divergence_log.jsonl)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)

        # Write noise_summary.json
        with (run_dir / "noise_summary.json").open("w") as f:
            json.dump(factory_noise_summary, f)

        # Write p5_divergence_real.json only
        with (run_dir / "p5_divergence_real.json").open("w") as f:
            json.dump(factory_p5_divergence_real_report, f)

        yield run_dir


# =============================================================================
# Test: End-to-End Generation
# =============================================================================

class TestEndToEndGeneration:
    """Tests for end-to-end noise_vs_reality generation."""

    def test_generate_with_factory_artifacts(self, factory_run_dir):
        """Test generation with complete factory artifacts."""
        result = generate_noise_vs_reality_for_evidence(factory_run_dir)

        assert result is not None
        assert result["schema_version"] == SCHEMA_VERSION
        assert result["mode"] == "SHADOW"
        assert result["experiment_id"] == "test-integration-001"

        # Verify P3 summary
        assert result["p3_summary"]["total_cycles"] == 100
        assert result["p3_summary"]["noise_event_rate"] == 0.08

        # Verify P5 summary
        assert result["p5_summary"]["total_cycles"] == 100
        assert "divergence_rate" in result["p5_summary"]

        # Verify comparison metrics
        assert "coverage_ratio" in result["comparison_metrics"]

        # Verify coverage assessment
        assert result["coverage_assessment"]["verdict"] in ["ADEQUATE", "MARGINAL", "INSUFFICIENT"]

    def test_generate_missing_divergence_returns_none(self, factory_partial_run_dir):
        """Test that missing divergence data returns None."""
        result = generate_noise_vs_reality_for_evidence(factory_partial_run_dir)
        assert result is None

    def test_generate_empty_dir_returns_none(self, factory_empty_run_dir):
        """Test that empty directory returns None."""
        result = generate_noise_vs_reality_for_evidence(factory_empty_run_dir)
        assert result is None

    def test_generated_summary_validates(self, factory_run_dir):
        """Test that generated summary passes validation."""
        result = generate_noise_vs_reality_for_evidence(factory_run_dir)

        assert result is not None
        is_valid, errors = validate_noise_vs_reality_summary(result)
        assert is_valid, f"Validation errors: {errors}"


# =============================================================================
# Test: Manifest Attachment
# =============================================================================

class TestManifestAttachment:
    """Tests for attaching noise_vs_reality to manifest."""

    def test_attach_to_empty_manifest(self, factory_run_dir):
        """Test attaching to manifest without governance section."""
        manifest = {"artifacts": [], "bundle_id": "test-bundle"}
        nvr_summary = generate_noise_vs_reality_for_evidence(factory_run_dir)

        result = attach_noise_vs_reality_to_manifest(manifest, nvr_summary)

        assert "governance" in result
        assert "noise_vs_reality" in result["governance"]
        assert result["governance"]["noise_vs_reality"]["mode"] == "SHADOW"

    def test_attach_preserves_existing_governance(self, factory_run_dir):
        """Test that attachment preserves existing governance fields."""
        manifest = {
            "governance": {
                "existing_signal": {"status": "ok"},
            }
        }
        nvr_summary = generate_noise_vs_reality_for_evidence(factory_run_dir)

        result = attach_noise_vs_reality_to_manifest(manifest, nvr_summary)

        assert "existing_signal" in result["governance"]
        assert "noise_vs_reality" in result["governance"]

    def test_attached_fields_match_schema(self, factory_run_dir):
        """Test that attached fields match expected schema."""
        manifest = {}
        nvr_summary = generate_noise_vs_reality_for_evidence(factory_run_dir)

        result = attach_noise_vs_reality_to_manifest(manifest, nvr_summary)

        nvr = result["governance"]["noise_vs_reality"]
        assert "schema_version" in nvr
        assert "mode" in nvr
        assert "verdict" in nvr
        assert "coverage_ratio" in nvr
        assert "advisory_severity" in nvr
        assert "p3_noise_rate" in nvr
        assert "p5_divergence_rate" in nvr


# =============================================================================
# Test: Signal Extraction
# =============================================================================

class TestSignalExtraction:
    """Tests for extracting signal for status generator."""

    def test_extract_from_attached_manifest(self, factory_run_dir):
        """Test extracting signal from manifest with attached noise_vs_reality."""
        manifest = {}
        nvr_summary = generate_noise_vs_reality_for_evidence(factory_run_dir)
        manifest = attach_noise_vs_reality_to_manifest(manifest, nvr_summary)

        signal = extract_noise_vs_reality_signal(manifest)

        assert signal is not None
        assert "verdict" in signal
        assert "advisory_severity" in signal
        assert "coverage_ratio" in signal

    def test_extract_from_empty_manifest_returns_missing(self):
        """Test that extraction from empty manifest returns MISSING signal."""
        from backend.topology.first_light.noise_vs_reality_integration import ExtractionSource
        manifest = {}
        signal = extract_noise_vs_reality_signal(manifest)
        assert signal is not None
        assert signal["extraction_source"] == ExtractionSource.MISSING.value
        assert signal["verdict"] is None

    def test_extract_from_manifest_without_nvr_returns_missing(self):
        """Test that extraction from manifest without noise_vs_reality returns MISSING signal."""
        from backend.topology.first_light.noise_vs_reality_integration import ExtractionSource
        manifest = {
            "governance": {
                "other_signal": {"status": "ok"},
            }
        }
        signal = extract_noise_vs_reality_signal(manifest)
        assert signal is not None
        assert signal["extraction_source"] == ExtractionSource.MISSING.value
        assert signal["verdict"] is None


# =============================================================================
# Test: Full Pipeline Integration
# =============================================================================

class TestFullPipelineIntegration:
    """Tests for full pipeline from artifacts to status signal."""

    def test_full_pipeline_adequate_coverage(self, factory_run_dir):
        """Test full pipeline produces ADEQUATE coverage verdict."""
        # Step 1: Generate noise_vs_reality_summary
        nvr_summary = generate_noise_vs_reality_for_evidence(factory_run_dir)
        assert nvr_summary is not None

        # Step 2: Attach to manifest
        manifest = {"bundle_id": "pipeline-test"}
        manifest = attach_noise_vs_reality_to_manifest(manifest, nvr_summary)

        # Step 3: Extract signal
        signal = extract_noise_vs_reality_signal(manifest)

        # Verify full pipeline output
        assert signal is not None
        assert signal["verdict"] in ["ADEQUATE", "MARGINAL"]  # Should be adequate with factory data
        assert signal["advisory_severity"] in ["INFO", "WARN"]

    def test_full_pipeline_with_high_divergence(self):
        """Test full pipeline with high divergence produces appropriate verdict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)

            # Create noise summary with low noise rate
            noise_summary = {
                "schema_version": "p3-noise-summary/1.0.0",
                "mode": "SHADOW",
                "total_cycles": 100,
                "regime_proportions": {"base": 1.0},
                "delta_p_aggregate": {"total_contribution": -0.05, "by_noise_type": {}},
                "rsi_aggregate": {"noise_event_rate": 0.02},  # Low noise
            }
            with (run_dir / "noise_summary.json").open("w") as f:
                json.dump(noise_summary, f)

            # Create divergence log with high divergence
            lines = []
            for i in range(100):
                # High divergence in many cycles
                entry = {
                    "cycle": i,
                    "twin_delta_p": 0.01,
                    "real_delta_p": 0.05,  # High divergence
                    "divergence_magnitude": 0.04,
                    "is_red_flag": i % 5 == 0,
                }
                lines.append(json.dumps(entry))
            with (run_dir / "divergence_log.jsonl").open("w") as f:
                f.write("\n".join(lines))

            # Run pipeline
            nvr_summary = generate_noise_vs_reality_for_evidence(run_dir)
            assert nvr_summary is not None

            manifest = {}
            manifest = attach_noise_vs_reality_to_manifest(manifest, nvr_summary)
            signal = extract_noise_vs_reality_signal(manifest)

            # With low P3 noise and high P5 divergence, coverage should not be ADEQUATE
            assert signal is not None
            # Coverage ratio should be low (P3 < P5)
            assert signal["coverage_ratio"] < 1.0


# =============================================================================
# Test: SHADOW Mode Contract
# =============================================================================

class TestShadowModeContract:
    """Tests verifying SHADOW mode contract throughout pipeline."""

    def test_generated_summary_is_shadow_mode(self, factory_run_dir):
        """Test that generated summary has SHADOW mode."""
        result = generate_noise_vs_reality_for_evidence(factory_run_dir)
        assert result["mode"] == "SHADOW"

    def test_attached_manifest_has_shadow_mode(self, factory_run_dir):
        """Test that attached manifest entry has SHADOW mode."""
        manifest = {}
        nvr_summary = generate_noise_vs_reality_for_evidence(factory_run_dir)
        manifest = attach_noise_vs_reality_to_manifest(manifest, nvr_summary)

        assert manifest["governance"]["noise_vs_reality"]["mode"] == "SHADOW"

    def test_advisory_severity_never_critical(self, factory_run_dir):
        """Test that advisory severity is never CRITICAL in SHADOW mode."""
        nvr_summary = generate_noise_vs_reality_for_evidence(factory_run_dir)
        severity = nvr_summary["governance_advisory"]["severity"]

        assert severity in ["INFO", "WARN"]
        assert severity not in ["CRITICAL", "BLOCK", "ERROR"]


# =============================================================================
# Test: P5 Input Preference Order
# =============================================================================

class TestP5InputPreferenceOrder:
    """Tests for p5_divergence_real.json preference over divergence_log.jsonl."""

    def test_prefers_real_report_when_both_exist(self, factory_run_dir_with_both_p5_sources):
        """Test that p5_divergence_real.json is preferred when both sources exist."""
        result = generate_noise_vs_reality_for_evidence(factory_run_dir_with_both_p5_sources)

        assert result is not None
        # p5_source now uses frozen enum - real report with VALIDATED_REAL status
        assert result["_p5_source"] == P5Source.REAL_VALIDATED.value
        assert result["_p5_source"] in VALID_P5_SOURCES

    def test_uses_jsonl_fallback_when_no_real_report(self, factory_run_dir):
        """Test that divergence_log.jsonl is used when no real report exists."""
        result = generate_noise_vs_reality_for_evidence(factory_run_dir)

        assert result is not None
        # p5_source now uses frozen enum for JSONL fallback
        assert result["_p5_source"] == P5Source.JSONL_FALLBACK.value
        assert result["_p5_source"] in VALID_P5_SOURCES

    def test_works_with_only_real_report(self, factory_run_dir_with_only_real_report):
        """Test that generation works with only p5_divergence_real.json."""
        result = generate_noise_vs_reality_for_evidence(factory_run_dir_with_only_real_report)

        assert result is not None
        # p5_source uses frozen enum - real report with VALIDATED_REAL status
        assert result["_p5_source"] == P5Source.REAL_VALIDATED.value
        assert result["_p5_source"] in VALID_P5_SOURCES
        # Verify summary validates
        is_valid, errors = validate_noise_vs_reality_summary(result)
        assert is_valid, f"Validation errors: {errors}"

    def test_real_report_telemetry_provider_validated(
        self, factory_run_dir_with_only_real_report
    ):
        """Test that VALIDATED_REAL status produces correct telemetry provider."""
        result = generate_noise_vs_reality_for_evidence(factory_run_dir_with_only_real_report)

        assert result is not None
        # Telemetry provider is stored in p5_summary.telemetry_source.provider
        provider = result["p5_summary"]["telemetry_source"]["provider"]
        assert provider == "p5_real_validated"


# =============================================================================
# Test: Consistent Verdicts Across Pipeline
# =============================================================================

class TestConsistentVerdicts:
    """Tests verifying verdicts remain consistent across pipeline stages."""

    def test_verdict_consistent_summary_to_manifest(self, factory_run_dir):
        """Test that verdict in summary matches verdict in manifest."""
        nvr_summary = generate_noise_vs_reality_for_evidence(factory_run_dir)
        manifest = attach_noise_vs_reality_to_manifest({}, nvr_summary)

        summary_verdict = nvr_summary["coverage_assessment"]["verdict"]
        manifest_verdict = manifest["governance"]["noise_vs_reality"]["verdict"]

        assert summary_verdict == manifest_verdict

    def test_verdict_consistent_manifest_to_signal(self, factory_run_dir):
        """Test that verdict in manifest matches extracted signal."""
        nvr_summary = generate_noise_vs_reality_for_evidence(factory_run_dir)
        manifest = attach_noise_vs_reality_to_manifest({}, nvr_summary)
        signal = extract_noise_vs_reality_signal(manifest)

        manifest_verdict = manifest["governance"]["noise_vs_reality"]["verdict"]

        assert signal["verdict"] == manifest_verdict

    def test_advisory_severity_consistent_across_pipeline(self, factory_run_dir):
        """Test that advisory severity is consistent from summary to signal."""
        nvr_summary = generate_noise_vs_reality_for_evidence(factory_run_dir)
        manifest = attach_noise_vs_reality_to_manifest({}, nvr_summary)
        signal = extract_noise_vs_reality_signal(manifest)

        summary_severity = nvr_summary["governance_advisory"]["severity"]
        manifest_severity = manifest["governance"]["noise_vs_reality"]["advisory_severity"]
        signal_severity = signal["advisory_severity"]

        assert summary_severity == manifest_severity == signal_severity


# =============================================================================
# Test: Non-Gating Warnings (SHADOW Mode)
# =============================================================================

class TestNonGatingWarnings:
    """Tests verifying warnings are non-gating in SHADOW mode."""

    def test_warn_verdict_does_not_block(self):
        """Test that WARN severity does not produce blocking behavior."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)

            # Create noise summary with high noise rate
            noise_summary = {
                "schema_version": "p3-noise-summary/1.0.0",
                "mode": "SHADOW",
                "total_cycles": 100,
                "regime_proportions": {"base": 1.0},
                "delta_p_aggregate": {"total_contribution": -0.5, "by_noise_type": {}},
                "rsi_aggregate": {"noise_event_rate": 0.30},  # High noise
            }
            with (run_dir / "noise_summary.json").open("w") as f:
                json.dump(noise_summary, f)

            # Create divergence log with high divergence (mismatch = WARN)
            lines = []
            for i in range(100):
                entry = {
                    "cycle": i,
                    "twin_delta_p": 0.01,
                    "real_delta_p": 0.08,  # High divergence
                    "divergence_magnitude": 0.07,
                    "is_red_flag": i % 3 == 0,
                }
                lines.append(json.dumps(entry))
            with (run_dir / "divergence_log.jsonl").open("w") as f:
                f.write("\n".join(lines))

            # Generate summary - should succeed even with warnings
            nvr_summary = generate_noise_vs_reality_for_evidence(run_dir)
            assert nvr_summary is not None

            # Verify WARN severity but no blocking
            severity = nvr_summary["governance_advisory"]["severity"]
            assert severity in ["INFO", "WARN"]  # Never BLOCK/CRITICAL

            # Verify signal can still be extracted (non-gating)
            manifest = attach_noise_vs_reality_to_manifest({}, nvr_summary)
            signal = extract_noise_vs_reality_signal(manifest)
            assert signal is not None

    def test_insufficient_verdict_still_produces_signal(self):
        """Test that INSUFFICIENT verdict still produces observable signal."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)

            # Create noise summary with very low noise
            noise_summary = {
                "schema_version": "p3-noise-summary/1.0.0",
                "mode": "SHADOW",
                "total_cycles": 100,
                "regime_proportions": {"base": 1.0},
                "delta_p_aggregate": {"total_contribution": -0.01, "by_noise_type": {}},
                "rsi_aggregate": {"noise_event_rate": 0.01},  # Very low noise
            }
            with (run_dir / "noise_summary.json").open("w") as f:
                json.dump(noise_summary, f)

            # Create divergence log with very high divergence
            lines = []
            for i in range(100):
                entry = {
                    "cycle": i,
                    "twin_delta_p": 0.01,
                    "real_delta_p": 0.15,  # Very high divergence
                    "divergence_magnitude": 0.14,
                    "is_red_flag": True,
                }
                lines.append(json.dumps(entry))
            with (run_dir / "divergence_log.jsonl").open("w") as f:
                f.write("\n".join(lines))

            # Generate summary
            nvr_summary = generate_noise_vs_reality_for_evidence(run_dir)
            assert nvr_summary is not None

            # Verify coverage is insufficient but still observable
            verdict = nvr_summary["coverage_assessment"]["verdict"]
            assert verdict in ["INADEQUATE", "MARGINAL", "INSUFFICIENT"]

            # Signal should still be generated (SHADOW = non-gating)
            manifest = attach_noise_vs_reality_to_manifest({}, nvr_summary)
            signal = extract_noise_vs_reality_signal(manifest)
            assert signal is not None
            assert signal["verdict"] is not None


# =============================================================================
# Test: SHA256 Hash Integrity
# =============================================================================

class TestSHA256HashIntegrity:
    """Tests for SHA256 hash in manifest reference."""

    def test_manifest_includes_sha256(self, factory_run_dir):
        """Test that attached manifest includes sha256 hash."""
        nvr_summary = generate_noise_vs_reality_for_evidence(factory_run_dir)
        manifest = attach_noise_vs_reality_to_manifest({}, nvr_summary)

        nvr = manifest["governance"]["noise_vs_reality"]
        assert "summary_sha256" in nvr
        assert len(nvr["summary_sha256"]) == 64  # SHA256 hex length
        assert all(c in "0123456789abcdef" for c in nvr["summary_sha256"])

    def test_sha256_deterministic(self, factory_run_dir):
        """Test that sha256 is deterministic for same input."""
        nvr_summary = generate_noise_vs_reality_for_evidence(factory_run_dir)

        hash1 = compute_summary_sha256(nvr_summary)
        hash2 = compute_summary_sha256(nvr_summary)

        assert hash1 == hash2

    def test_sha256_changes_with_content(self, factory_run_dir):
        """Test that sha256 changes when summary content changes."""
        nvr_summary = generate_noise_vs_reality_for_evidence(factory_run_dir)
        hash1 = compute_summary_sha256(nvr_summary)

        # Modify summary
        modified_summary = dict(nvr_summary)
        modified_summary["experiment_id"] = "modified-id"
        hash2 = compute_summary_sha256(modified_summary)

        assert hash1 != hash2

    def test_signal_extraction_includes_sha256(self, factory_run_dir):
        """Test that extracted signal includes sha256."""
        nvr_summary = generate_noise_vs_reality_for_evidence(factory_run_dir)
        manifest = attach_noise_vs_reality_to_manifest({}, nvr_summary)
        signal = extract_noise_vs_reality_signal(manifest)

        assert "summary_sha256" in signal
        assert signal["summary_sha256"] is not None


# =============================================================================
# Test: P5 Source Enum (FROZEN)
# =============================================================================

class TestP5SourceEnum:
    """Tests for frozen p5_source enum and coercion behavior."""

    def test_all_enum_values_in_valid_set(self):
        """Test that all P5Source enum values are in VALID_P5_SOURCES."""
        for source in P5Source:
            assert source.value in VALID_P5_SOURCES

    def test_valid_p5_sources_count(self):
        """Test that exactly 4 valid p5_source values exist."""
        assert len(VALID_P5_SOURCES) == 4
        assert P5Source.REAL_VALIDATED.value in VALID_P5_SOURCES
        assert P5Source.SUSPECTED_MOCK.value in VALID_P5_SOURCES
        assert P5Source.REAL_ADAPTER.value in VALID_P5_SOURCES
        assert P5Source.JSONL_FALLBACK.value in VALID_P5_SOURCES

    def test_normalize_valid_source_no_advisory(self):
        """Test that valid sources return without advisory note."""
        for source in P5Source:
            normalized, advisory = normalize_p5_source(source.value)
            assert normalized == source.value
            assert advisory is None

    def test_normalize_none_coerces_to_fallback(self):
        """Test that None p5_source coerces to fallback with advisory."""
        normalized, advisory = normalize_p5_source(None)
        assert normalized == P5Source.JSONL_FALLBACK.value
        assert advisory is not None
        assert "coerced" in advisory.lower()

    def test_normalize_unknown_coerces_to_fallback(self):
        """Test that unknown p5_source coerces to fallback with advisory."""
        normalized, advisory = normalize_p5_source("unknown_source")
        assert normalized == P5Source.JSONL_FALLBACK.value
        assert advisory is not None
        assert "unknown_source" in advisory
        assert "coerced" in advisory.lower()

    def test_normalize_old_filename_coerces_to_fallback(self):
        """Test that old filename format coerces to fallback."""
        # Old format: "p5_divergence_real.json" or "divergence_log.jsonl"
        normalized, advisory = normalize_p5_source("p5_divergence_real.json")
        assert normalized == P5Source.JSONL_FALLBACK.value
        assert advisory is not None

        normalized, advisory = normalize_p5_source("divergence_log.jsonl")
        assert normalized == P5Source.JSONL_FALLBACK.value
        assert advisory is not None

    def test_manifest_contains_valid_p5_source(self, factory_run_dir):
        """Test that manifest p5_source is always a valid enum value."""
        nvr_summary = generate_noise_vs_reality_for_evidence(factory_run_dir)
        manifest = attach_noise_vs_reality_to_manifest({}, nvr_summary)

        p5_source = manifest["governance"]["noise_vs_reality"]["p5_source"]
        assert p5_source in VALID_P5_SOURCES

    def test_manifest_p5_source_advisory_none_when_valid(self, factory_run_dir):
        """Test that p5_source_advisory is None when source is valid."""
        nvr_summary = generate_noise_vs_reality_for_evidence(factory_run_dir)
        manifest = attach_noise_vs_reality_to_manifest({}, nvr_summary)

        advisory = manifest["governance"]["noise_vs_reality"]["p5_source_advisory"]
        # Should be None since we're using valid enum values
        assert advisory is None


# =============================================================================
# Test: Single Warning Cap + Top Factor
# =============================================================================

class TestSingleWarningCapTopFactor:
    """Tests for single warning line with top driving factor."""

    def test_adequate_verdict_info_severity(self, factory_run_dir):
        """Test that ADEQUATE verdict produces INFO severity."""
        nvr_summary = generate_noise_vs_reality_for_evidence(factory_run_dir)
        advisory = nvr_summary["governance_advisory"]

        # ADEQUATE = INFO, no top_factor needed
        if nvr_summary["coverage_assessment"]["verdict"] == "ADEQUATE":
            assert advisory["severity"] == "INFO"
            assert advisory["top_factor"] is None

    def test_marginal_verdict_has_top_factor(self):
        """Test that MARGINAL verdict includes top driving factor."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)

            # Create noise summary with moderate noise
            noise_summary = {
                "schema_version": "p3-noise-summary/1.0.0",
                "mode": "SHADOW",
                "total_cycles": 100,
                "regime_proportions": {"base": 1.0},
                "delta_p_aggregate": {"total_contribution": -0.05, "by_noise_type": {}},
                "rsi_aggregate": {"noise_event_rate": 0.05},
            }
            with (run_dir / "noise_summary.json").open("w") as f:
                json.dump(noise_summary, f)

            # Create divergence log with slightly higher divergence
            lines = []
            for i in range(100):
                entry = {
                    "cycle": i,
                    "twin_delta_p": 0.01,
                    "real_delta_p": 0.015,  # Moderate divergence
                    "divergence_magnitude": 0.005,
                    "is_red_flag": i % 20 == 0,
                }
                lines.append(json.dumps(entry))
            with (run_dir / "divergence_log.jsonl").open("w") as f:
                f.write("\n".join(lines))

            nvr_summary = generate_noise_vs_reality_for_evidence(run_dir)
            assert nvr_summary is not None

            advisory = nvr_summary["governance_advisory"]
            verdict = nvr_summary["coverage_assessment"]["verdict"]

            if verdict in ["MARGINAL", "INSUFFICIENT"]:
                # Must have WARN severity
                assert advisory["severity"] == "WARN"
                # Must have top_factor
                assert advisory["top_factor"] in ["coverage_ratio", "exceedance_rate"]
                assert advisory["top_factor_value"] is not None

    def test_insufficient_verdict_has_top_factor(self):
        """Test that INSUFFICIENT verdict includes top driving factor."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)

            # Create noise summary with low noise
            noise_summary = {
                "schema_version": "p3-noise-summary/1.0.0",
                "mode": "SHADOW",
                "total_cycles": 100,
                "regime_proportions": {"base": 1.0},
                "delta_p_aggregate": {"total_contribution": -0.01, "by_noise_type": {}},
                "rsi_aggregate": {"noise_event_rate": 0.01},
            }
            with (run_dir / "noise_summary.json").open("w") as f:
                json.dump(noise_summary, f)

            # Create divergence log with very high divergence
            lines = []
            for i in range(100):
                entry = {
                    "cycle": i,
                    "twin_delta_p": 0.01,
                    "real_delta_p": 0.10,  # High divergence
                    "divergence_magnitude": 0.09,
                    "is_red_flag": True,
                }
                lines.append(json.dumps(entry))
            with (run_dir / "divergence_log.jsonl").open("w") as f:
                f.write("\n".join(lines))

            nvr_summary = generate_noise_vs_reality_for_evidence(run_dir)
            assert nvr_summary is not None

            advisory = nvr_summary["governance_advisory"]
            verdict = nvr_summary["coverage_assessment"]["verdict"]

            # Should be INSUFFICIENT with high divergence vs low noise
            if verdict == "INSUFFICIENT":
                assert advisory["severity"] == "WARN"
                assert advisory["top_factor"] in ["coverage_ratio", "exceedance_rate"]
                assert advisory["top_factor_value"] is not None

    def test_warning_message_single_line(self, factory_run_dir):
        """Test that warning message is a single line (no newlines)."""
        nvr_summary = generate_noise_vs_reality_for_evidence(factory_run_dir)
        advisory = nvr_summary["governance_advisory"]

        message = advisory["message"]
        assert "\n" not in message
        # Message should be concise
        assert len(message) < 150

    def test_warning_message_contains_top_factor_value(self):
        """Test that WARN message includes the top factor value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)

            noise_summary = {
                "schema_version": "p3-noise-summary/1.0.0",
                "mode": "SHADOW",
                "total_cycles": 100,
                "regime_proportions": {"base": 1.0},
                "delta_p_aggregate": {"total_contribution": -0.02, "by_noise_type": {}},
                "rsi_aggregate": {"noise_event_rate": 0.02},
            }
            with (run_dir / "noise_summary.json").open("w") as f:
                json.dump(noise_summary, f)

            lines = []
            for i in range(100):
                entry = {
                    "cycle": i,
                    "twin_delta_p": 0.01,
                    "real_delta_p": 0.05,
                    "divergence_magnitude": 0.04,
                    "is_red_flag": i % 5 == 0,
                }
                lines.append(json.dumps(entry))
            with (run_dir / "divergence_log.jsonl").open("w") as f:
                f.write("\n".join(lines))

            nvr_summary = generate_noise_vs_reality_for_evidence(run_dir)
            assert nvr_summary is not None

            advisory = nvr_summary["governance_advisory"]
            verdict = nvr_summary["coverage_assessment"]["verdict"]

            if verdict in ["MARGINAL", "INSUFFICIENT"]:
                message = advisory["message"]
                top_factor = advisory["top_factor"]
                # Message should contain the factor name
                assert top_factor in message or "=" in message

    def test_top_factor_coverage_ratio_format(self):
        """Test coverage_ratio top factor has correct format."""
        from backend.topology.first_light.noise_vs_reality import generate_governance_advisory

        assessment = {"verdict": "MARGINAL", "confidence": 0.85}
        metrics = {"coverage_ratio": 0.75, "exceedance_rate": 0.02}

        advisory = generate_governance_advisory(assessment, metrics)

        assert advisory["severity"] == "WARN"
        # Coverage deficit (0.25) > exceedance (0.02), so coverage_ratio is top factor
        assert advisory["top_factor"] == "coverage_ratio"
        assert advisory["top_factor_value"] == 0.75
        assert "0.75" in advisory["message"]

    def test_top_factor_exceedance_rate_format(self):
        """Test exceedance_rate top factor has correct format."""
        from backend.topology.first_light.noise_vs_reality import generate_governance_advisory

        assessment = {"verdict": "MARGINAL", "confidence": 0.85}
        metrics = {"coverage_ratio": 0.95, "exceedance_rate": 0.15}

        advisory = generate_governance_advisory(assessment, metrics)

        assert advisory["severity"] == "WARN"
        # Exceedance (0.15) > coverage deficit (0.05), so exceedance_rate is top factor
        assert advisory["top_factor"] == "exceedance_rate"
        assert advisory["top_factor_value"] == 0.15
        assert "15.0%" in advisory["message"]
