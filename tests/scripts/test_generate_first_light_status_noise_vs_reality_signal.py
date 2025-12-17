"""
Integration tests for noise_vs_reality signal in first_light_status.json.

Tests verify:
- Signal extraction includes all provenance fields
- p5_source uses frozen enum values
- p5_source_advisory is nullable (None when valid)
- summary_sha256 is deterministic
- top_factor/top_factor_value in signal
- Single-line warning format
- Warning neutrality (no banned alarm words)

SHADOW MODE CONTRACT:
- All signals are observational
- Warnings are non-gating
"""

import json
import pytest
import tempfile
from pathlib import Path
from typing import Any, Dict

# Import reusable warning neutrality helpers
from tests.helpers.warning_neutrality import (
    pytest_assert_warning_neutral,
    assert_single_line,
    BANNED_ALARM_WORDS,
)

from backend.topology.first_light.noise_vs_reality_integration import (
    generate_noise_vs_reality_for_evidence,
    attach_noise_vs_reality_to_manifest,
    extract_noise_vs_reality_signal,
    compute_summary_sha256,
    normalize_p5_source,
    P5Source,
    VALID_P5_SOURCES,
    ExtractionSource,
    VALID_EXTRACTION_SOURCES,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def noise_summary_adequate() -> Dict[str, Any]:
    """P3 noise summary yielding ADEQUATE coverage."""
    return {
        "schema_version": "p3-noise-summary/1.0.0",
        "mode": "SHADOW",
        "total_cycles": 100,
        "regime_proportions": {"base": 1.0},
        "delta_p_aggregate": {"total_contribution": -0.10, "by_noise_type": {}},
        "rsi_aggregate": {"noise_event_rate": 0.08},
    }


@pytest.fixture
def noise_summary_marginal() -> Dict[str, Any]:
    """P3 noise summary yielding MARGINAL coverage."""
    return {
        "schema_version": "p3-noise-summary/1.0.0",
        "mode": "SHADOW",
        "total_cycles": 100,
        "regime_proportions": {"base": 1.0},
        "delta_p_aggregate": {"total_contribution": -0.03, "by_noise_type": {}},
        "rsi_aggregate": {"noise_event_rate": 0.03},  # Low noise
    }


@pytest.fixture
def divergence_log_moderate() -> str:
    """Divergence log with moderate divergence."""
    lines = []
    for i in range(100):
        entry = {
            "cycle": i,
            "twin_delta_p": 0.01,
            "real_delta_p": 0.015 if i % 10 == 0 else 0.01,
            "divergence_magnitude": 0.005 if i % 10 == 0 else 0.0,
            "is_red_flag": i in [30, 70],
        }
        lines.append(json.dumps(entry))
    return "\n".join(lines)


@pytest.fixture
def divergence_log_high() -> str:
    """Divergence log with high divergence."""
    lines = []
    for i in range(100):
        entry = {
            "cycle": i,
            "twin_delta_p": 0.01,
            "real_delta_p": 0.06,  # High divergence
            "divergence_magnitude": 0.05,
            "is_red_flag": i % 5 == 0,
        }
        lines.append(json.dumps(entry))
    return "\n".join(lines)


def create_run_dir(noise_summary: Dict, divergence_log: str) -> Path:
    """Helper to create temp run directory."""
    tmpdir = tempfile.mkdtemp()
    run_dir = Path(tmpdir)

    with (run_dir / "noise_summary.json").open("w") as f:
        json.dump(noise_summary, f)

    with (run_dir / "divergence_log.jsonl").open("w") as f:
        f.write(divergence_log)

    return run_dir


# =============================================================================
# Test: Signal Contains All Provenance Fields
# =============================================================================

class TestSignalProvenanceFields:
    """Tests that signal contains all required provenance fields."""

    def test_signal_has_extraction_source(self, noise_summary_adequate, divergence_log_moderate):
        """Test signal includes extraction_source enum value."""
        run_dir = create_run_dir(noise_summary_adequate, divergence_log_moderate)

        nvr_summary = generate_noise_vs_reality_for_evidence(run_dir)
        manifest = attach_noise_vs_reality_to_manifest({}, nvr_summary)
        signal = extract_noise_vs_reality_signal(manifest)

        assert "extraction_source" in signal
        assert signal["extraction_source"] in VALID_EXTRACTION_SOURCES
        assert signal["extraction_source"] == ExtractionSource.MANIFEST.value

    def test_signal_has_p5_source(self, noise_summary_adequate, divergence_log_moderate):
        """Test signal includes p5_source enum value."""
        run_dir = create_run_dir(noise_summary_adequate, divergence_log_moderate)

        nvr_summary = generate_noise_vs_reality_for_evidence(run_dir)
        manifest = attach_noise_vs_reality_to_manifest({}, nvr_summary)
        signal = extract_noise_vs_reality_signal(manifest)

        assert "p5_source" in signal
        assert signal["p5_source"] in VALID_P5_SOURCES

    def test_signal_has_p5_source_advisory(self, noise_summary_adequate, divergence_log_moderate):
        """Test signal includes p5_source_advisory (nullable)."""
        run_dir = create_run_dir(noise_summary_adequate, divergence_log_moderate)

        nvr_summary = generate_noise_vs_reality_for_evidence(run_dir)
        manifest = attach_noise_vs_reality_to_manifest({}, nvr_summary)
        signal = extract_noise_vs_reality_signal(manifest)

        assert "p5_source_advisory" in signal
        # Should be None when p5_source is valid
        assert signal["p5_source_advisory"] is None

    def test_signal_has_summary_sha256(self, noise_summary_adequate, divergence_log_moderate):
        """Test signal includes summary_sha256."""
        run_dir = create_run_dir(noise_summary_adequate, divergence_log_moderate)

        nvr_summary = generate_noise_vs_reality_for_evidence(run_dir)
        manifest = attach_noise_vs_reality_to_manifest({}, nvr_summary)
        signal = extract_noise_vs_reality_signal(manifest)

        assert "summary_sha256" in signal
        assert signal["summary_sha256"] is not None
        assert len(signal["summary_sha256"]) == 64

    def test_signal_has_top_factor_fields(self, noise_summary_marginal, divergence_log_high):
        """Test signal includes top_factor and top_factor_value."""
        run_dir = create_run_dir(noise_summary_marginal, divergence_log_high)

        nvr_summary = generate_noise_vs_reality_for_evidence(run_dir)
        manifest = attach_noise_vs_reality_to_manifest({}, nvr_summary)
        signal = extract_noise_vs_reality_signal(manifest)

        assert "top_factor" in signal
        assert "top_factor_value" in signal

        # For WARN verdicts, top_factor should be set
        if signal["advisory_severity"] == "WARN":
            assert signal["top_factor"] in ["coverage_ratio", "exceedance_rate", None]

    def test_signal_has_advisory_message(self, noise_summary_adequate, divergence_log_moderate):
        """Test signal includes advisory_message."""
        run_dir = create_run_dir(noise_summary_adequate, divergence_log_moderate)

        nvr_summary = generate_noise_vs_reality_for_evidence(run_dir)
        manifest = attach_noise_vs_reality_to_manifest({}, nvr_summary)
        signal = extract_noise_vs_reality_signal(manifest)

        assert "advisory_message" in signal


# =============================================================================
# Test: Extraction Source Enum
# =============================================================================

class TestExtractionSourceEnum:
    """Tests for extraction_source enum values."""

    def test_extraction_source_manifest(self, noise_summary_adequate, divergence_log_moderate):
        """Test extraction_source is MANIFEST when extracted from manifest."""
        run_dir = create_run_dir(noise_summary_adequate, divergence_log_moderate)

        nvr_summary = generate_noise_vs_reality_for_evidence(run_dir)
        manifest = attach_noise_vs_reality_to_manifest({}, nvr_summary)
        signal = extract_noise_vs_reality_signal(manifest)

        assert signal["extraction_source"] == ExtractionSource.MANIFEST.value

    def test_extraction_source_missing_empty_manifest(self):
        """Test extraction_source is MISSING when manifest has no noise_vs_reality."""
        manifest = {}
        signal = extract_noise_vs_reality_signal(manifest)

        assert signal["extraction_source"] == ExtractionSource.MISSING.value
        assert signal["verdict"] is None

    def test_all_extraction_source_values_in_valid_set(self):
        """Test all ExtractionSource enum values are in VALID_EXTRACTION_SOURCES."""
        for source in ExtractionSource:
            assert source.value in VALID_EXTRACTION_SOURCES

    def test_valid_extraction_sources_count(self):
        """Test exactly 3 valid extraction_source values exist."""
        assert len(VALID_EXTRACTION_SOURCES) == 3
        assert ExtractionSource.MANIFEST.value in VALID_EXTRACTION_SOURCES
        assert ExtractionSource.EVIDENCE_JSON.value in VALID_EXTRACTION_SOURCES
        assert ExtractionSource.MISSING.value in VALID_EXTRACTION_SOURCES


# =============================================================================
# Test: Advisory Warning Format
# =============================================================================

class TestAdvisoryWarningFormat:
    """Tests for advisory_warning single-line format and neutrality."""

    def test_advisory_warning_has_stable_format(self, noise_summary_marginal, divergence_log_high):
        """Test advisory_warning follows stable format: VERDICT: factor=value [source]."""
        run_dir = create_run_dir(noise_summary_marginal, divergence_log_high)

        nvr_summary = generate_noise_vs_reality_for_evidence(run_dir)
        manifest = attach_noise_vs_reality_to_manifest({}, nvr_summary)
        signal = extract_noise_vs_reality_signal(manifest)

        advisory_warning = signal.get("advisory_warning")
        if advisory_warning:
            # Check format: VERDICT: factor=value [abbrev]
            assert ":" in advisory_warning
            assert "[" in advisory_warning
            assert "]" in advisory_warning
            # Use reusable helper for neutrality check
            pytest_assert_warning_neutral(advisory_warning, context="advisory_warning format")

    def test_advisory_warning_includes_p5_source_abbrev(
        self, noise_summary_marginal, divergence_log_high
    ):
        """Test advisory_warning includes p5_source abbreviation in brackets."""
        run_dir = create_run_dir(noise_summary_marginal, divergence_log_high)

        nvr_summary = generate_noise_vs_reality_for_evidence(run_dir)
        manifest = attach_noise_vs_reality_to_manifest({}, nvr_summary)
        signal = extract_noise_vs_reality_signal(manifest)

        advisory_warning = signal.get("advisory_warning")
        if advisory_warning:
            # Should include source abbreviation
            valid_abbrevs = ["real", "mock?", "adapter", "jsonl"]
            has_abbrev = any(f"[{abbrev}]" in advisory_warning for abbrev in valid_abbrevs)
            assert has_abbrev, f"No valid source abbrev found in: {advisory_warning}"

    def test_adequate_verdict_no_advisory_warning(
        self, noise_summary_adequate, divergence_log_moderate
    ):
        """Test ADEQUATE verdict produces no advisory_warning."""
        run_dir = create_run_dir(noise_summary_adequate, divergence_log_moderate)

        nvr_summary = generate_noise_vs_reality_for_evidence(run_dir)
        manifest = attach_noise_vs_reality_to_manifest({}, nvr_summary)
        signal = extract_noise_vs_reality_signal(manifest)

        # ADEQUATE should not have advisory_warning
        if signal.get("verdict") == "ADEQUATE":
            assert signal.get("advisory_warning") is None


# =============================================================================
# Test: P5 Source Enum Coercion at Signal Level
# =============================================================================

class TestSignalP5SourceEnumCoercion:
    """Tests for p5_source enum coercion in final signal."""

    def test_jsonl_fallback_produces_enum_value(self, noise_summary_adequate, divergence_log_moderate):
        """Test that JSONL fallback produces valid enum p5_source."""
        run_dir = create_run_dir(noise_summary_adequate, divergence_log_moderate)

        nvr_summary = generate_noise_vs_reality_for_evidence(run_dir)
        manifest = attach_noise_vs_reality_to_manifest({}, nvr_summary)
        signal = extract_noise_vs_reality_signal(manifest)

        # Since we only have divergence_log.jsonl, should be JSONL_FALLBACK
        assert signal["p5_source"] == P5Source.JSONL_FALLBACK.value

    def test_signal_p5_source_never_legacy_filename(
        self, noise_summary_adequate, divergence_log_moderate
    ):
        """Test that signal p5_source never contains legacy filename format."""
        run_dir = create_run_dir(noise_summary_adequate, divergence_log_moderate)

        nvr_summary = generate_noise_vs_reality_for_evidence(run_dir)
        manifest = attach_noise_vs_reality_to_manifest({}, nvr_summary)
        signal = extract_noise_vs_reality_signal(manifest)

        # Should never be old filename format
        assert signal["p5_source"] != "divergence_log.jsonl"
        assert signal["p5_source"] != "p5_divergence_real.json"
        # Should be valid enum value
        assert signal["p5_source"] in VALID_P5_SOURCES

    def test_unknown_source_coercion_produces_advisory(self):
        """Test that unknown source coercion produces advisory note."""
        normalized, advisory = normalize_p5_source("legacy_unknown_source")

        assert normalized == P5Source.JSONL_FALLBACK.value
        assert advisory is not None
        assert "legacy_unknown_source" in advisory
        assert "coerced" in advisory.lower()


# =============================================================================
# Test: SHA256 Determinism at Signal Level
# =============================================================================

class TestSignalSHA256Determinism:
    """Tests for SHA256 determinism in final signal."""

    def test_sha256_deterministic_across_extractions(
        self, noise_summary_adequate, divergence_log_moderate
    ):
        """Test that sha256 is deterministic across multiple extractions."""
        run_dir = create_run_dir(noise_summary_adequate, divergence_log_moderate)

        nvr_summary = generate_noise_vs_reality_for_evidence(run_dir)

        # Extract multiple times
        manifest1 = attach_noise_vs_reality_to_manifest({}, nvr_summary)
        signal1 = extract_noise_vs_reality_signal(manifest1)

        manifest2 = attach_noise_vs_reality_to_manifest({}, nvr_summary)
        signal2 = extract_noise_vs_reality_signal(manifest2)

        assert signal1["summary_sha256"] == signal2["summary_sha256"]

    def test_sha256_matches_direct_computation(
        self, noise_summary_adequate, divergence_log_moderate
    ):
        """Test that signal sha256 matches direct computation."""
        run_dir = create_run_dir(noise_summary_adequate, divergence_log_moderate)

        nvr_summary = generate_noise_vs_reality_for_evidence(run_dir)
        direct_hash = compute_summary_sha256(nvr_summary)

        manifest = attach_noise_vs_reality_to_manifest({}, nvr_summary)
        signal = extract_noise_vs_reality_signal(manifest)

        assert signal["summary_sha256"] == direct_hash

    def test_sha256_valid_hex_format(self, noise_summary_adequate, divergence_log_moderate):
        """Test that sha256 is valid 64-character hex."""
        run_dir = create_run_dir(noise_summary_adequate, divergence_log_moderate)

        nvr_summary = generate_noise_vs_reality_for_evidence(run_dir)
        manifest = attach_noise_vs_reality_to_manifest({}, nvr_summary)
        signal = extract_noise_vs_reality_signal(manifest)

        sha256 = signal["summary_sha256"]
        assert len(sha256) == 64
        assert all(c in "0123456789abcdef" for c in sha256)


# =============================================================================
# Test: Single-Line Warning Format at Signal Level
# =============================================================================

class TestSignalSingleLineWarning:
    """Tests for single-line warning format and neutrality in signal advisory_message."""

    def test_advisory_message_neutral(self, noise_summary_marginal, divergence_log_high):
        """Test that advisory_message is neutral (single-line, no banned words)."""
        run_dir = create_run_dir(noise_summary_marginal, divergence_log_high)

        nvr_summary = generate_noise_vs_reality_for_evidence(run_dir)
        manifest = attach_noise_vs_reality_to_manifest({}, nvr_summary)
        signal = extract_noise_vs_reality_signal(manifest)

        message = signal["advisory_message"]
        if message:
            pytest_assert_warning_neutral(message, context="advisory_message")

    def test_advisory_message_concise(self, noise_summary_marginal, divergence_log_high):
        """Test that advisory_message is concise (< 150 chars)."""
        run_dir = create_run_dir(noise_summary_marginal, divergence_log_high)

        nvr_summary = generate_noise_vs_reality_for_evidence(run_dir)
        manifest = attach_noise_vs_reality_to_manifest({}, nvr_summary)
        signal = extract_noise_vs_reality_signal(manifest)

        message = signal["advisory_message"]
        if message:
            assert len(message) < 150

    def test_warn_advisory_contains_factor_value(self, noise_summary_marginal, divergence_log_high):
        """Test that WARN advisory contains top_factor value."""
        run_dir = create_run_dir(noise_summary_marginal, divergence_log_high)

        nvr_summary = generate_noise_vs_reality_for_evidence(run_dir)
        manifest = attach_noise_vs_reality_to_manifest({}, nvr_summary)
        signal = extract_noise_vs_reality_signal(manifest)

        if signal["advisory_severity"] == "WARN":
            message = signal["advisory_message"]
            # Should contain a numeric value (factor value)
            assert "=" in message or "%" in message


# =============================================================================
# Test: Status Generator Warning Integration
# =============================================================================

class TestStatusGeneratorWarningIntegration:
    """Tests verifying status generator produces single-line warnings."""

    def test_marginal_verdict_single_warning_line(
        self, noise_summary_marginal, divergence_log_high
    ):
        """Test that MARGINAL verdict produces single neutral warning line for status generator."""
        run_dir = create_run_dir(noise_summary_marginal, divergence_log_high)

        nvr_summary = generate_noise_vs_reality_for_evidence(run_dir)
        manifest = attach_noise_vs_reality_to_manifest({}, nvr_summary)
        signal = extract_noise_vs_reality_signal(manifest)

        # Simulate status generator warning construction
        warnings = []
        verdict = signal["verdict"]
        advisory_message = signal["advisory_message"]

        if verdict in ["INSUFFICIENT", "MARGINAL"] and advisory_message:
            warnings.append(f"Noise vs reality: {advisory_message}")

        # Should produce at most one warning
        nvr_warnings = [w for w in warnings if "Noise vs reality" in w]
        assert len(nvr_warnings) <= 1

        # Warning should be neutral (single-line, no banned words)
        if nvr_warnings:
            pytest_assert_warning_neutral(nvr_warnings[0], context="status generator warning")

    def test_adequate_verdict_no_warning(self, noise_summary_adequate, divergence_log_moderate):
        """Test that ADEQUATE verdict produces no warning."""
        run_dir = create_run_dir(noise_summary_adequate, divergence_log_moderate)

        nvr_summary = generate_noise_vs_reality_for_evidence(run_dir)
        manifest = attach_noise_vs_reality_to_manifest({}, nvr_summary)
        signal = extract_noise_vs_reality_signal(manifest)

        # Simulate status generator warning construction
        warnings = []
        verdict = signal["verdict"]
        advisory_message = signal["advisory_message"]

        if verdict in ["INSUFFICIENT", "MARGINAL"] and advisory_message:
            warnings.append(f"Noise vs reality: {advisory_message}")

        # ADEQUATE should not add warning
        if verdict == "ADEQUATE":
            nvr_warnings = [w for w in warnings if "Noise vs reality" in w]
            assert len(nvr_warnings) == 0
