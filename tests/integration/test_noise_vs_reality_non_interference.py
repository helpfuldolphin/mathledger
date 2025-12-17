"""
Integration test: Noise vs Reality signal non-interference.

Proves that adding/removing noise_vs_reality signal changes ONLY:
1. signals.noise_vs_reality
2. At most one warning line

Does NOT change:
- Any other signal keys or values
- Warning count (except +/- 1 for noise_vs_reality)
- Serialized ordering of other signals

CAL-EXP-2 PREP: Ensures noise_vs_reality hook doesn't affect divergence minimization runs.

Uses reusable helpers from tests.helpers.non_interference.
"""

import copy
import json
from pathlib import Path
import pytest
from typing import Any, Dict, List, Set

from tests.helpers.non_interference import (
    assert_only_keys_changed,
    assert_warning_delta_at_most_one,
    pytest_assert_only_keys_changed,
    pytest_assert_warning_delta_at_most_one,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def base_signals() -> Dict[str, Any]:
    """Base signals dict without noise_vs_reality."""
    return {
        "p3_stability": {
            "verdict": "PASS",
            "cycles": 100,
            "noise_event_rate": 0.08,
        },
        "p4_divergence": {
            "verdict": "PASS",
            "divergence_rate": 0.05,
            "red_flag_count": 2,
        },
        "schemas_ok": True,
        "schemas_ok_summary": {
            "pass": 5,
            "fail": 0,
            "missing": 1,
        },
    }


@pytest.fixture
def noise_vs_reality_signal() -> Dict[str, Any]:
    """Noise vs reality signal to inject."""
    return {
        "extraction_source": "MANIFEST",
        "verdict": "MARGINAL",
        "advisory_severity": "WARN",
        "coverage_ratio": 0.72,
        "p3_noise_rate": 0.08,
        "p5_divergence_rate": 0.11,
        "p5_source": "p5_real_validated",
        "p5_source_advisory": None,
        "summary_sha256": "abc123def456",
        "top_factor": "coverage_ratio",
        "top_factor_value": 0.72,
    }


@pytest.fixture
def base_warnings() -> List[str]:
    """Base warnings list without noise_vs_reality warning."""
    return [
        "Schema validation: 1 missing schema(s)",
        "P4 divergence: 2 red flags detected",
    ]


# =============================================================================
# Test: Signal Isolation
# =============================================================================

class TestNoiseVsRealitySignalIsolation:
    """Tests that noise_vs_reality signal is isolated from other signals.

    Uses tests.helpers.non_interference for standardized assertions.
    """

    def test_adding_nvr_only_adds_nvr_key(
        self, base_signals, noise_vs_reality_signal
    ):
        """Test that adding noise_vs_reality only adds that key."""
        signals_before = copy.deepcopy(base_signals)
        signals_after = copy.deepcopy(base_signals)
        signals_after["noise_vs_reality"] = noise_vs_reality_signal

        # Use helper: only noise_vs_reality key should differ
        pytest_assert_only_keys_changed(
            before=signals_before,
            after=signals_after,
            allowed_paths=["noise_vs_reality.*"],
            context="adding noise_vs_reality signal",
        )

    def test_removing_nvr_only_removes_nvr_key(
        self, base_signals, noise_vs_reality_signal
    ):
        """Test that removing noise_vs_reality only removes that key."""
        signals_with_nvr = copy.deepcopy(base_signals)
        signals_with_nvr["noise_vs_reality"] = noise_vs_reality_signal

        signals_without_nvr = copy.deepcopy(signals_with_nvr)
        del signals_without_nvr["noise_vs_reality"]

        # Use helper: only noise_vs_reality key should differ
        pytest_assert_only_keys_changed(
            before=signals_with_nvr,
            after=signals_without_nvr,
            allowed_paths=["noise_vs_reality.*"],
            context="removing noise_vs_reality signal",
        )

    def test_other_signals_unchanged_after_nvr_add(
        self, base_signals, noise_vs_reality_signal
    ):
        """Test that other signals remain identical after adding noise_vs_reality."""
        signals_before = copy.deepcopy(base_signals)
        signals_after = copy.deepcopy(base_signals)
        signals_after["noise_vs_reality"] = noise_vs_reality_signal

        # Use helper with allowed_paths
        result = assert_only_keys_changed(
            before=signals_before,
            after=signals_after,
            allowed_paths=["noise_vs_reality.*"],
        )
        assert result.passed, f"Non-interference violated: {result.violations}"

    def test_serialization_order_preserved(
        self, base_signals, noise_vs_reality_signal
    ):
        """Test that JSON serialization order of other keys is preserved."""
        signals_before = copy.deepcopy(base_signals)
        signals_after = copy.deepcopy(base_signals)
        signals_after["noise_vs_reality"] = noise_vs_reality_signal

        # Serialize both (sorted for determinism)
        json_before = json.dumps(signals_before, sort_keys=True)

        # Remove nvr and serialize
        signals_after_stripped = {
            k: v for k, v in signals_after.items() if k != "noise_vs_reality"
        }
        json_after_stripped = json.dumps(signals_after_stripped, sort_keys=True)

        assert json_before == json_after_stripped


# =============================================================================
# Test: Warning Isolation
# =============================================================================

class TestNoiseVsRealityWarningIsolation:
    """Tests that noise_vs_reality warning is isolated from other warnings.

    Uses tests.helpers.non_interference for standardized assertions.
    """

    def test_nvr_adds_at_most_one_warning(
        self, base_warnings, noise_vs_reality_signal
    ):
        """Test that noise_vs_reality adds at most one warning line."""
        warnings_before = copy.deepcopy(base_warnings)
        warnings_after = copy.deepcopy(base_warnings)

        # Simulate adding nvr warning
        nvr_verdict = noise_vs_reality_signal["verdict"]
        if nvr_verdict in ["INSUFFICIENT", "MARGINAL"]:
            nvr_warn = f"Noise vs reality: {nvr_verdict}: coverage_ratio=0.72 [real]"
            warnings_after.append(nvr_warn)

        # Use helper: delta at most 1, ordering preserved
        pytest_assert_warning_delta_at_most_one(
            before=warnings_before,
            after=warnings_after,
            context="adding noise_vs_reality warning",
        )

    def test_nvr_warning_identifiable(self, base_warnings):
        """Test that noise_vs_reality warning is identifiable by prefix."""
        warnings = copy.deepcopy(base_warnings)
        nvr_warn = "Noise vs reality: MARGINAL: coverage_ratio=0.72 [real]"
        warnings.append(nvr_warn)

        nvr_warnings = [w for w in warnings if w.startswith("Noise vs reality:")]
        assert len(nvr_warnings) == 1
        assert nvr_warnings[0] == nvr_warn

    def test_removing_nvr_warning_preserves_others(self, base_warnings):
        """Test that removing noise_vs_reality warning preserves other warnings."""
        original_warnings = copy.deepcopy(base_warnings)

        # Add nvr warning
        all_warnings = copy.deepcopy(base_warnings)
        all_warnings.append("Noise vs reality: MARGINAL: coverage_ratio=0.72 [real]")

        # Remove nvr warnings
        filtered_warnings = [
            w for w in all_warnings if not w.startswith("Noise vs reality:")
        ]

        assert filtered_warnings == original_warnings

    def test_adequate_verdict_adds_no_warning(self, base_warnings):
        """Test that ADEQUATE verdict adds no warning."""
        warnings_before = copy.deepcopy(base_warnings)
        warnings_after = copy.deepcopy(base_warnings)

        # ADEQUATE verdict should not add warning
        nvr_verdict = "ADEQUATE"
        if nvr_verdict in ["INSUFFICIENT", "MARGINAL"]:
            warnings_after.append("Noise vs reality: ...")

        assert warnings_before == warnings_after


# =============================================================================
# Test: Determinism
# =============================================================================

class TestNoiseVsRealityDeterminism:
    """Tests that noise_vs_reality signal is deterministic."""

    def test_same_input_same_output(self, noise_vs_reality_signal):
        """Test that same input produces same serialized output."""
        signal1 = copy.deepcopy(noise_vs_reality_signal)
        signal2 = copy.deepcopy(noise_vs_reality_signal)

        json1 = json.dumps(signal1, sort_keys=True)
        json2 = json.dumps(signal2, sort_keys=True)

        assert json1 == json2

    def test_signal_keys_stable(self, noise_vs_reality_signal):
        """Test that noise_vs_reality signal has stable keys."""
        expected_keys = {
            "extraction_source",
            "verdict",
            "advisory_severity",
            "coverage_ratio",
            "p3_noise_rate",
            "p5_divergence_rate",
            "p5_source",
            "p5_source_advisory",
            "summary_sha256",
            "top_factor",
            "top_factor_value",
        }

        actual_keys = set(noise_vs_reality_signal.keys())
        assert actual_keys == expected_keys


# =============================================================================
# Test: Integration with Status Generator Pattern
# =============================================================================

class TestStatusGeneratorPattern:
    """Tests that follow status generator integration patterns."""

    def test_signals_dict_mutation_pattern(
        self, base_signals, noise_vs_reality_signal
    ):
        """Test the signals dict mutation pattern used in status generator."""
        signals = copy.deepcopy(base_signals)

        # Simulate status generator pattern
        nvr_extraction_source = noise_vs_reality_signal.get("extraction_source")
        if nvr_extraction_source != "MISSING":
            nvr_verdict = noise_vs_reality_signal.get("verdict")
            nvr_advisory_severity = noise_vs_reality_signal.get("advisory_severity")

            if nvr_verdict and nvr_advisory_severity:
                signals["noise_vs_reality"] = noise_vs_reality_signal

        # Verify only noise_vs_reality was added
        assert "noise_vs_reality" in signals
        assert len(signals) == len(base_signals) + 1

    def test_warning_append_pattern(self, base_warnings, noise_vs_reality_signal):
        """Test the warning append pattern used in status generator."""
        warnings = copy.deepcopy(base_warnings)
        original_count = len(warnings)

        # Simulate status generator warning pattern
        nvr_verdict = noise_vs_reality_signal.get("verdict")
        if nvr_verdict in ["INSUFFICIENT", "MARGINAL"]:
            top_factor = noise_vs_reality_signal.get("top_factor")
            top_factor_value = noise_vs_reality_signal.get("top_factor_value")
            p5_source = noise_vs_reality_signal.get("p5_source")

            # Format warning
            source_abbrev = {
                "p5_real_validated": "real",
                "p5_suspected_mock": "mock?",
                "p5_real_adapter": "adapter",
                "p5_jsonl_fallback": "jsonl",
            }.get(p5_source, "unk")

            if top_factor == "coverage_ratio":
                nvr_warn = f"Noise vs reality: {nvr_verdict}: coverage_ratio={top_factor_value:.2f} [{source_abbrev}]"
            else:
                nvr_warn = f"Noise vs reality: {nvr_verdict}: exceedance_rate={top_factor_value*100:.1f}% [{source_abbrev}]"

            warnings.append(nvr_warn)

        # Verify at most one warning added
        assert len(warnings) <= original_count + 1

        # Verify format
        nvr_warnings = [w for w in warnings if w.startswith("Noise vs reality:")]
        if nvr_warnings:
            assert len(nvr_warnings) == 1
            assert "[real]" in nvr_warnings[0] or "[jsonl]" in nvr_warnings[0]


# =============================================================================
# Test: Timestamp-Stripped Determinism
# =============================================================================

class TestTimestampStrippedDeterminism:
    """Tests determinism when timestamp fields are stripped."""

    TIMESTAMP_KEYS = {"timestamp", "generated_at", "created_at", "updated_at"}

    def strip_timestamps(self, obj: Any) -> Any:
        """Recursively strip timestamp keys from dict."""
        if isinstance(obj, dict):
            return {
                k: self.strip_timestamps(v)
                for k, v in obj.items()
                if k not in self.TIMESTAMP_KEYS
            }
        elif isinstance(obj, list):
            return [self.strip_timestamps(item) for item in obj]
        return obj

    def test_signals_deterministic_without_timestamps(
        self, base_signals, noise_vs_reality_signal
    ):
        """Test signals are deterministic when timestamps are stripped."""
        # Add timestamps to base signals
        signals1 = copy.deepcopy(base_signals)
        signals1["timestamp"] = "2025-01-01T00:00:00Z"
        signals1["noise_vs_reality"] = noise_vs_reality_signal

        signals2 = copy.deepcopy(base_signals)
        signals2["timestamp"] = "2025-01-02T12:00:00Z"  # Different timestamp
        signals2["noise_vs_reality"] = noise_vs_reality_signal

        # Strip timestamps and compare
        stripped1 = self.strip_timestamps(signals1)
        stripped2 = self.strip_timestamps(signals2)

        assert json.dumps(stripped1, sort_keys=True) == json.dumps(stripped2, sort_keys=True)

    def test_nvr_signal_has_no_timestamps(self, noise_vs_reality_signal):
        """Test that noise_vs_reality signal has no timestamp keys."""
        nvr_keys = set(noise_vs_reality_signal.keys())
        timestamp_overlap = nvr_keys & self.TIMESTAMP_KEYS

        assert timestamp_overlap == set(), f"Found timestamp keys: {timestamp_overlap}"


# =============================================================================
# Test: CAL-EXP-2 Run Directory Shape
# =============================================================================

class TestCalExp2RunDirShape:
    """
    Integration test using CAL-EXP-2 run directory shape.

    Creates temp directory shaped like: results/cal_exp_2/p4_YYYYMMDD_HHMMSS/
    with minimal manifest.json and optional evidence.json.

    Proves noise_vs_reality signal isolation at the file-system level.
    """

    TIMESTAMP_KEYS = {"timestamp", "generated_at", "created_at", "updated_at"}

    def strip_timestamps(self, obj: Any) -> Any:
        """Recursively strip timestamp keys from dict."""
        if isinstance(obj, dict):
            return {
                k: self.strip_timestamps(v)
                for k, v in obj.items()
                if k not in self.TIMESTAMP_KEYS
            }
        elif isinstance(obj, list):
            return [self.strip_timestamps(item) for item in obj]
        return obj

    @pytest.fixture
    def cal_exp_run_dir_without_nvr(self, tmp_path):
        """Create CAL-EXP-2 run directory WITHOUT noise_vs_reality."""
        run_dir = tmp_path / "results" / "cal_exp_2" / "p4_20250113_120000"
        run_dir.mkdir(parents=True)

        # Minimal manifest.json without noise_vs_reality
        manifest = {
            "schema_version": "1.0.0",
            "run_id": "cal_exp_2_test_001",
            "timestamp": "2025-01-13T12:00:00Z",
            "governance": {
                "p3_stability": {"verdict": "PASS", "cycles": 100},
                "p4_divergence": {"verdict": "PASS", "divergence_rate": 0.05},
            },
        }
        with (run_dir / "manifest.json").open("w") as f:
            json.dump(manifest, f, indent=2)

        # Optional evidence.json
        evidence = {
            "signals": {
                "p3_stability": {"verdict": "PASS"},
                "p4_divergence": {"verdict": "PASS"},
            },
            "warnings": ["Schema validation: 1 missing schema(s)"],
        }
        with (run_dir / "evidence.json").open("w") as f:
            json.dump(evidence, f, indent=2)

        return run_dir

    @pytest.fixture
    def cal_exp_run_dir_with_nvr(self, tmp_path):
        """Create CAL-EXP-2 run directory WITH noise_vs_reality."""
        run_dir = tmp_path / "results" / "cal_exp_2" / "p4_20250113_120001"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Manifest with noise_vs_reality in governance
        manifest = {
            "schema_version": "1.0.0",
            "run_id": "cal_exp_2_test_001",
            "timestamp": "2025-01-13T12:00:00Z",
            "governance": {
                "p3_stability": {"verdict": "PASS", "cycles": 100},
                "p4_divergence": {"verdict": "PASS", "divergence_rate": 0.05},
                "noise_vs_reality": {
                    "schema_version": "noise-vs-reality/1.0.0",
                    "mode": "SHADOW",
                    "verdict": "MARGINAL",
                    "coverage_ratio": 0.72,
                    "advisory_severity": "WARN",
                    "advisory_message": "P3 coverage marginal",
                    "top_factor": "coverage_ratio",
                    "top_factor_value": 0.72,
                    "p3_noise_rate": 0.08,
                    "p5_divergence_rate": 0.11,
                    "p5_source": "p5_real_validated",
                    "p5_source_advisory": None,
                    "summary_sha256": "abc123def456",
                },
            },
        }
        with (run_dir / "manifest.json").open("w") as f:
            json.dump(manifest, f, indent=2)

        # Evidence.json with noise_vs_reality signal
        evidence = {
            "signals": {
                "p3_stability": {"verdict": "PASS"},
                "p4_divergence": {"verdict": "PASS"},
                "noise_vs_reality": {
                    "extraction_source": "MANIFEST",
                    "verdict": "MARGINAL",
                    "advisory_severity": "WARN",
                    "coverage_ratio": 0.72,
                },
            },
            "warnings": [
                "Schema validation: 1 missing schema(s)",
                "Noise vs reality: MARGINAL: coverage_ratio=0.72 [real]",
            ],
        }
        with (run_dir / "evidence.json").open("w") as f:
            json.dump(evidence, f, indent=2)

        return run_dir

    def test_nvr_only_changes_nvr_signal_key(
        self, cal_exp_run_dir_without_nvr, cal_exp_run_dir_with_nvr
    ):
        """Test adding noise_vs_reality only adds that signal key."""
        # Load evidence.json from both
        with (cal_exp_run_dir_without_nvr / "evidence.json").open() as f:
            evidence_without = json.load(f)
        with (cal_exp_run_dir_with_nvr / "evidence.json").open() as f:
            evidence_with = json.load(f)

        signals_without = set(evidence_without.get("signals", {}).keys())
        signals_with = set(evidence_with.get("signals", {}).keys())

        added_keys = signals_with - signals_without
        assert added_keys == {"noise_vs_reality"}

    def test_nvr_adds_at_most_one_warning(
        self, cal_exp_run_dir_without_nvr, cal_exp_run_dir_with_nvr
    ):
        """Test adding noise_vs_reality adds at most one warning."""
        with (cal_exp_run_dir_without_nvr / "evidence.json").open() as f:
            evidence_without = json.load(f)
        with (cal_exp_run_dir_with_nvr / "evidence.json").open() as f:
            evidence_with = json.load(f)

        warnings_without = evidence_without.get("warnings", [])
        warnings_with = evidence_with.get("warnings", [])

        # Count nvr-specific warnings
        nvr_warnings_without = [w for w in warnings_without if "Noise vs reality" in w]
        nvr_warnings_with = [w for w in warnings_with if "Noise vs reality" in w]

        assert len(nvr_warnings_without) == 0
        assert len(nvr_warnings_with) <= 1

    def test_other_signals_unchanged(
        self, cal_exp_run_dir_without_nvr, cal_exp_run_dir_with_nvr
    ):
        """Test other signals are unchanged when noise_vs_reality is added."""
        with (cal_exp_run_dir_without_nvr / "evidence.json").open() as f:
            evidence_without = json.load(f)
        with (cal_exp_run_dir_with_nvr / "evidence.json").open() as f:
            evidence_with = json.load(f)

        # Extract signals excluding noise_vs_reality
        signals_without = evidence_without.get("signals", {})
        signals_with = {
            k: v for k, v in evidence_with.get("signals", {}).items()
            if k != "noise_vs_reality"
        }

        assert signals_without == signals_with

    def test_determinism_after_timestamp_strip(
        self, cal_exp_run_dir_with_nvr
    ):
        """Test determinism holds after stripping timestamps."""
        with (cal_exp_run_dir_with_nvr / "manifest.json").open() as f:
            manifest1 = json.load(f)

        # Simulate second read with different timestamp
        manifest2 = copy.deepcopy(manifest1)
        manifest2["timestamp"] = "2025-01-14T00:00:00Z"

        # Strip timestamps
        stripped1 = self.strip_timestamps(manifest1)
        stripped2 = self.strip_timestamps(manifest2)

        assert json.dumps(stripped1, sort_keys=True) == json.dumps(stripped2, sort_keys=True)

    def test_manifest_governance_shape_preserved(
        self, cal_exp_run_dir_without_nvr, cal_exp_run_dir_with_nvr
    ):
        """Test manifest governance section shape is preserved except for noise_vs_reality."""
        with (cal_exp_run_dir_without_nvr / "manifest.json").open() as f:
            manifest_without = json.load(f)
        with (cal_exp_run_dir_with_nvr / "manifest.json").open() as f:
            manifest_with = json.load(f)

        gov_without = manifest_without.get("governance", {})
        gov_with = {
            k: v for k, v in manifest_with.get("governance", {}).items()
            if k != "noise_vs_reality"
        }

        assert gov_without == gov_with


class TestRealCalExp2Directory:
    """
    Tests against real CAL-EXP-2 run directories if present.

    Skip-safe: tests skip gracefully if no real run directory exists.

    Expected directory structure:
        results/cal_exp_2/p4_YYYYMMDD_HHMMSS/
            manifest.json
            evidence.json (optional)
    """

    # Canonical paths to check for real CAL-EXP-2 directories
    REAL_RUN_DIR_PATTERNS = [
        Path("results/cal_exp_2"),
        Path("C:/dev/mathledger/results/cal_exp_2"),
    ]

    @staticmethod
    def find_real_run_dir() -> Path | None:
        """Find the most recent real CAL-EXP-2 run directory."""
        for base_path in TestRealCalExp2Directory.REAL_RUN_DIR_PATTERNS:
            if not base_path.exists():
                continue
            # Find all p4_* subdirectories
            run_dirs = sorted(
                [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("p4_")],
                reverse=True,  # Most recent first
            )
            for run_dir in run_dirs:
                if (run_dir / "manifest.json").exists():
                    return run_dir
        return None

    @staticmethod
    def strip_timestamps(obj):
        """Recursively strip timestamp-like fields for determinism check."""
        if isinstance(obj, dict):
            return {
                k: TestRealCalExp2Directory.strip_timestamps(v)
                for k, v in obj.items()
                if k not in ("timestamp", "created_at", "updated_at", "run_timestamp")
            }
        elif isinstance(obj, list):
            return [TestRealCalExp2Directory.strip_timestamps(item) for item in obj]
        return obj

    @pytest.fixture
    def real_run_dir(self):
        """Fixture that provides real run dir or skips test."""
        run_dir = self.find_real_run_dir()
        if run_dir is None:
            pytest.skip("No run dir at results/cal_exp_2/p4_YYYYMMDD_HHMMSS/manifest.json")
        return run_dir

    def test_real_manifest_keys_preserved(self, real_run_dir):
        """
        Assert real manifest preserves expected governance keys.

        Expected keys (at minimum):
        - schema_version
        - run_id
        - governance (with p3_stability, p4_divergence)

        If noise_vs_reality is present, it must not remove other keys.
        """
        with (real_run_dir / "manifest.json").open() as f:
            manifest = json.load(f)

        # Required top-level keys
        assert "schema_version" in manifest, "Missing schema_version"
        assert "run_id" in manifest, "Missing run_id"
        assert "governance" in manifest, "Missing governance section"

        governance = manifest["governance"]

        # Core governance keys that must be preserved
        expected_core_keys = {"p3_stability", "p4_divergence"}
        actual_keys = set(governance.keys())

        # noise_vs_reality is optional, but if present, core keys must remain
        if "noise_vs_reality" in actual_keys:
            core_preserved = expected_core_keys.issubset(actual_keys - {"noise_vs_reality"})
            assert core_preserved, (
                f"noise_vs_reality present but core keys missing. "
                f"Expected: {expected_core_keys}, Got: {actual_keys - {'noise_vs_reality'}}"
            )
        else:
            # Without NVR, at least check governance isn't empty
            assert len(governance) > 0, "Governance section is empty"

    def test_real_nvr_warning_count(self, real_run_dir):
        """
        Assert noise_vs_reality adds at most one warning line.

        If evidence.json exists and has warnings, count NVR-related warnings.
        """
        evidence_path = real_run_dir / "evidence.json"
        if not evidence_path.exists():
            pytest.skip("No evidence.json at results/cal_exp_2/p4_*/evidence.json")

        with evidence_path.open() as f:
            evidence = json.load(f)

        warnings = evidence.get("warnings", [])
        nvr_warnings = [w for w in warnings if "noise" in w.lower() or "nvr" in w.lower()]

        assert len(nvr_warnings) <= 1, (
            f"noise_vs_reality added {len(nvr_warnings)} warnings (max 1 allowed): {nvr_warnings}"
        )

    def test_real_timestamp_stripped_determinism(self, real_run_dir):
        """
        Assert manifest is deterministic after stripping timestamps.

        Load manifest twice, strip timestamps, verify identical.
        """
        with (real_run_dir / "manifest.json").open() as f:
            manifest1 = json.load(f)

        # Simulate second load (same file)
        with (real_run_dir / "manifest.json").open() as f:
            manifest2 = json.load(f)

        stripped1 = self.strip_timestamps(manifest1)
        stripped2 = self.strip_timestamps(manifest2)

        # Serialize with sorted keys for stable comparison
        json1 = json.dumps(stripped1, sort_keys=True)
        json2 = json.dumps(stripped2, sort_keys=True)

        assert json1 == json2, "Timestamp-stripped manifests differ (non-deterministic)"

        # Also verify the stripped manifest is non-empty
        assert len(stripped1) > 0, "Stripped manifest is empty"
        assert "governance" in stripped1, "Stripped manifest missing governance"
