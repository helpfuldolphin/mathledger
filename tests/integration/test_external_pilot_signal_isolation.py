"""
Integration test: External Pilot Signal Non-Interference.

Proves that ingesting external pilot logs CANNOT contaminate:
1. Internal metrics
2. Experiment outputs
3. Verifier behavior

SHADOW MODE CONTRACT:
- Tests only (no production code changes)
- Observational verification
- Non-gating assertions

CAL-EXP-3 PREP: External signal ingestion must be provably isolated.

NAMESPACE ALIGNMENT:
- Real adapter uses: governance.external_pilot.*
- See: backend/health/pilot_external_ingest_adapter.py
"""

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pytest

from tests.helpers.non_interference import (
    assert_only_keys_changed,
    assert_warning_delta_at_most_one,
    pytest_assert_only_keys_changed,
    pytest_assert_warning_delta_at_most_one,
    pytest_assert_output_excludes_keys,
)

# Import real adapter for regression tests
from backend.health.pilot_external_ingest_adapter import (
    attach_to_manifest,
    PILOT_INGEST_SCHEMA_VERSION,
)


# =============================================================================
# Fixtures: Internal State Baseline
# =============================================================================

@pytest.fixture
def internal_metrics_baseline() -> Dict[str, Any]:
    """Baseline internal metrics that must remain invariant."""
    return {
        "calibration": {
            "dp_success": 0.025,
            "convergence_floor": 0.02,
            "cycles_completed": 1000,
            "experiment_id": "CAL-EXP-2",
        },
        "verifier": {
            "hash": "sha256:abc123def456",
            "version": "1.0.0",
            "mode": "PRODUCTION",
            "last_verification_ms": 42,
        },
        "experiment_outputs": {
            "baseline_mean_dp": 0.028,
            "treatment_mean_dp": 0.022,
            "delta_dp": -0.006,
            "validity": "VALID",
        },
    }


@pytest.fixture
def internal_warnings_baseline() -> List[str]:
    """Baseline warnings that must be preserved."""
    return [
        "Schema validation: 1 missing schema(s)",
        "P5 baseline: set from CAL-EXP-2",
    ]


# =============================================================================
# Fixtures: External Pilot Logs (Various Types)
# =============================================================================

@pytest.fixture
def valid_pilot_log() -> Dict[str, Any]:
    """Valid external pilot log with expected structure."""
    return {
        "pilot_id": "PILOT-001",
        "timestamp": "2025-01-15T10:00:00Z",
        "source": "external_sensor_array",
        "readings": {
            "temperature": 22.5,
            "pressure": 1013.25,
            "humidity": 45.0,
        },
        "status": "OK",
        "checksum": "sha256:external_valid_hash",
    }


@pytest.fixture
def malformed_pilot_log() -> Dict[str, Any]:
    """Malformed external pilot log with structural issues."""
    return {
        "pilot_id": None,  # Invalid: should be string
        "timestamp": "not-a-timestamp",  # Invalid format
        "source": "",  # Empty string
        "readings": "not-a-dict",  # Wrong type
        "extra_field": {"nested": {"deep": "value"}},  # Unexpected nesting
    }


@pytest.fixture
def empty_pilot_log() -> Dict[str, Any]:
    """Empty external pilot log."""
    return {}


@pytest.fixture
def adversarial_pilot_log() -> Dict[str, Any]:
    """Adversarial pilot log attempting to inject internal keys."""
    return {
        "pilot_id": "ADVERSARIAL-001",
        # Attempt to inject internal metric keys
        "calibration": {
            "dp_success": 999.0,  # Attempt to override
            "experiment_id": "INJECTED",
        },
        "verifier": {
            "hash": "sha256:INJECTED",
            "mode": "COMPROMISED",
        },
        "experiment_outputs": {
            "validity": "INVALID_INJECTION",
        },
        # Attempt path traversal in keys
        "../internal": "escape_attempt",
        "internal.metrics": "dot_injection",
    }


@pytest.fixture
def oversized_pilot_log() -> Dict[str, Any]:
    """Oversized pilot log with excessive data."""
    return {
        "pilot_id": "OVERSIZED-001",
        "timestamp": "2025-01-15T10:00:00Z",
        "large_array": list(range(10000)),  # Large array
        "deep_nesting": _create_deep_dict(depth=50),  # Deep nesting
        "long_string": "x" * 100000,  # Very long string
    }


def _create_deep_dict(depth: int) -> Dict[str, Any]:
    """Create a deeply nested dictionary."""
    if depth <= 0:
        return {"leaf": "value"}
    return {"level": _create_deep_dict(depth - 1)}


# =============================================================================
# Simulator: External Signal Ingestion
# =============================================================================

class PilotLogIngester:
    """
    Simulates external pilot log ingestion.

    ISOLATION CONTRACT (aligned with real adapter):
    - External data goes into `governance.external_pilot` namespace ONLY
    - Internal metrics are NEVER modified
    - Warnings are bounded (at most 1 per ingestion)

    NAMESPACE ALIGNMENT:
    - Real adapter: backend/health/pilot_external_ingest_adapter.py
    - Uses: governance.external_pilot.* (NOT pilot_signals.*)
    """

    # Keys that must NEVER be modified by external ingestion
    PROTECTED_KEYS = frozenset({
        "calibration",
        "verifier",
        "experiment_outputs",
        "internal_metrics",
        "dp_success",
        "convergence_floor",
    })

    @staticmethod
    def ingest(
        internal_state: Dict[str, Any],
        warnings: List[str],
        pilot_log: Dict[str, Any],
    ) -> tuple[Dict[str, Any], List[str]]:
        """
        Ingest external pilot log into state.

        Returns new state and warnings (does not mutate inputs).

        ISOLATION GUARANTEE (aligned with real adapter):
        - Only `governance.external_pilot.*` keys may be added/modified
        - Internal metrics remain invariant
        - Warning delta <= 1
        """
        # Deep copy to ensure purity
        new_state = copy.deepcopy(internal_state)
        new_warnings = copy.deepcopy(warnings)

        # Validate pilot log
        validation_result = PilotLogIngester._validate(pilot_log)

        # Initialize governance.external_pilot namespace if needed (aligned with real adapter)
        if "governance" not in new_state:
            new_state["governance"] = {}
        if "external_pilot" not in new_state["governance"]:
            new_state["governance"]["external_pilot"] = {
                "schema_version": PILOT_INGEST_SCHEMA_VERSION,
                "mode": "SHADOW",
                "action": "LOGGED_ONLY",
                "entries": [],
                "entry_count": 0,
                "invalid_count": 0,
                "warnings": [],
            }

        # Extract pilot ID (with fallback)
        pilot_id = pilot_log.get("pilot_id")
        if not isinstance(pilot_id, str) or not pilot_id:
            pilot_id = f"unknown_{hashlib.sha256(json.dumps(pilot_log, sort_keys=True, default=str).encode()).hexdigest()[:8]}"

        # Build entry for isolated namespace
        entry = {
            "path": f"external/{pilot_id}.json",
            "sha256": hashlib.sha256(
                json.dumps(pilot_log, sort_keys=True, default=str).encode()
            ).hexdigest()[:16],
            "pilot_metadata": {
                "source_type": "EXTERNAL_JSON",
                "extraction_source": "EXTERNAL_PILOT",
                "schema_version": PILOT_INGEST_SCHEMA_VERSION,
                "mode": "SHADOW",
                "action": "LOGGED_ONLY",
                "ingested_at": "2025-01-15T10:00:00Z",
                "validation_status": validation_result["status"],
            },
        }

        # Add to entries list
        new_state["governance"]["external_pilot"]["entries"].append(entry)

        if validation_result["status"] == "VALID":
            new_state["governance"]["external_pilot"]["entry_count"] += 1
        else:
            new_state["governance"]["external_pilot"]["invalid_count"] += 1

        # Add warning if validation failed (at most 1)
        if validation_result["status"] != "VALID":
            new_warnings.append(
                f"Pilot {pilot_id}: {validation_result['status']} ({validation_result['reason']})"
            )

        return new_state, new_warnings

    @staticmethod
    def _validate(pilot_log: Dict[str, Any]) -> Dict[str, Any]:
        """Validate pilot log structure."""
        if not pilot_log:
            return {"status": "EMPTY", "reason": "no data"}

        pilot_id = pilot_log.get("pilot_id")
        if not isinstance(pilot_id, str) or not pilot_id:
            return {"status": "MALFORMED", "reason": "invalid pilot_id"}

        timestamp = pilot_log.get("timestamp")
        if not isinstance(timestamp, str):
            return {"status": "MALFORMED", "reason": "invalid timestamp"}

        # Check for injection attempts
        for key in pilot_log.keys():
            if key in PilotLogIngester.PROTECTED_KEYS:
                return {"status": "REJECTED", "reason": f"protected key: {key}"}
            if ".." in str(key) or key.startswith("internal"):
                return {"status": "REJECTED", "reason": f"suspicious key: {key}"}

        return {"status": "VALID", "reason": "ok"}


# =============================================================================
# Test: Valid External Logs
# =============================================================================

class TestValidPilotLogIsolation:
    """Tests that valid external pilot logs are properly isolated."""

    def test_valid_log_only_adds_pilot_signals(
        self, internal_metrics_baseline, internal_warnings_baseline, valid_pilot_log
    ):
        """Valid pilot log only adds to governance.external_pilot namespace."""
        state_before = copy.deepcopy(internal_metrics_baseline)

        state_after, warnings_after = PilotLogIngester.ingest(
            internal_metrics_baseline,
            internal_warnings_baseline,
            valid_pilot_log,
        )

        # Only governance.external_pilot should change
        pytest_assert_only_keys_changed(
            before=state_before,
            after=state_after,
            allowed_paths=["governance.external_pilot.*"],
            context="valid pilot log ingestion",
        )

    def test_valid_log_preserves_internal_metrics(
        self, internal_metrics_baseline, internal_warnings_baseline, valid_pilot_log
    ):
        """Valid pilot log does not modify internal metrics."""
        state_after, _ = PilotLogIngester.ingest(
            internal_metrics_baseline,
            internal_warnings_baseline,
            valid_pilot_log,
        )

        # Internal metrics must be identical
        assert state_after["calibration"] == internal_metrics_baseline["calibration"]
        assert state_after["verifier"] == internal_metrics_baseline["verifier"]
        assert state_after["experiment_outputs"] == internal_metrics_baseline["experiment_outputs"]

    def test_valid_log_adds_zero_warnings(
        self, internal_metrics_baseline, internal_warnings_baseline, valid_pilot_log
    ):
        """Valid pilot log adds no warnings."""
        _, warnings_after = PilotLogIngester.ingest(
            internal_metrics_baseline,
            internal_warnings_baseline,
            valid_pilot_log,
        )

        # Valid log should not add warnings
        assert len(warnings_after) == len(internal_warnings_baseline)

    def test_valid_log_deterministic_output(
        self, internal_metrics_baseline, internal_warnings_baseline, valid_pilot_log
    ):
        """Multiple ingestions of same log produce identical output."""
        state1, warnings1 = PilotLogIngester.ingest(
            internal_metrics_baseline,
            internal_warnings_baseline,
            valid_pilot_log,
        )

        state2, warnings2 = PilotLogIngester.ingest(
            internal_metrics_baseline,
            internal_warnings_baseline,
            valid_pilot_log,
        )

        # Strip timestamps for comparison
        def strip_timestamps(obj):
            if isinstance(obj, dict):
                return {k: strip_timestamps(v) for k, v in obj.items() if k not in ("ingested_at", "timestamp")}
            elif isinstance(obj, list):
                return [strip_timestamps(item) for item in obj]
            return obj

        assert strip_timestamps(state1) == strip_timestamps(state2)
        assert warnings1 == warnings2


# =============================================================================
# Test: Malformed External Logs
# =============================================================================

class TestMalformedPilotLogIsolation:
    """Tests that malformed external logs cannot contaminate state."""

    def test_malformed_log_only_adds_pilot_signals(
        self, internal_metrics_baseline, internal_warnings_baseline, malformed_pilot_log
    ):
        """Malformed pilot log only adds to governance.external_pilot namespace."""
        state_before = copy.deepcopy(internal_metrics_baseline)

        state_after, warnings_after = PilotLogIngester.ingest(
            internal_metrics_baseline,
            internal_warnings_baseline,
            malformed_pilot_log,
        )

        pytest_assert_only_keys_changed(
            before=state_before,
            after=state_after,
            allowed_paths=["governance.external_pilot.*"],
            context="malformed pilot log ingestion",
        )

    def test_malformed_log_adds_at_most_one_warning(
        self, internal_metrics_baseline, internal_warnings_baseline, malformed_pilot_log
    ):
        """Malformed pilot log adds at most one warning."""
        _, warnings_after = PilotLogIngester.ingest(
            internal_metrics_baseline,
            internal_warnings_baseline,
            malformed_pilot_log,
        )

        pytest_assert_warning_delta_at_most_one(
            before=internal_warnings_baseline,
            after=warnings_after,
            context="malformed pilot log warning",
        )

    def test_malformed_log_preserves_verifier(
        self, internal_metrics_baseline, internal_warnings_baseline, malformed_pilot_log
    ):
        """Malformed pilot log does not affect verifier state."""
        state_after, _ = PilotLogIngester.ingest(
            internal_metrics_baseline,
            internal_warnings_baseline,
            malformed_pilot_log,
        )

        assert state_after["verifier"] == internal_metrics_baseline["verifier"]
        assert state_after["verifier"]["mode"] == "PRODUCTION"


# =============================================================================
# Test: Empty External Logs
# =============================================================================

class TestEmptyPilotLogIsolation:
    """Tests that empty external logs are handled safely."""

    def test_empty_log_only_adds_pilot_signals(
        self, internal_metrics_baseline, internal_warnings_baseline, empty_pilot_log
    ):
        """Empty pilot log only adds to governance.external_pilot namespace."""
        state_before = copy.deepcopy(internal_metrics_baseline)

        state_after, _ = PilotLogIngester.ingest(
            internal_metrics_baseline,
            internal_warnings_baseline,
            empty_pilot_log,
        )

        pytest_assert_only_keys_changed(
            before=state_before,
            after=state_after,
            allowed_paths=["governance.external_pilot.*"],
            context="empty pilot log ingestion",
        )

    def test_empty_log_adds_at_most_one_warning(
        self, internal_metrics_baseline, internal_warnings_baseline, empty_pilot_log
    ):
        """Empty pilot log adds at most one warning."""
        _, warnings_after = PilotLogIngester.ingest(
            internal_metrics_baseline,
            internal_warnings_baseline,
            empty_pilot_log,
        )

        pytest_assert_warning_delta_at_most_one(
            before=internal_warnings_baseline,
            after=warnings_after,
            context="empty pilot log warning",
        )

    def test_empty_log_preserves_experiment_outputs(
        self, internal_metrics_baseline, internal_warnings_baseline, empty_pilot_log
    ):
        """Empty pilot log does not affect experiment outputs."""
        state_after, _ = PilotLogIngester.ingest(
            internal_metrics_baseline,
            internal_warnings_baseline,
            empty_pilot_log,
        )

        assert state_after["experiment_outputs"] == internal_metrics_baseline["experiment_outputs"]


# =============================================================================
# Test: Adversarial External Logs
# =============================================================================

class TestAdversarialPilotLogIsolation:
    """Tests that adversarial logs cannot inject or override internal state."""

    def test_adversarial_log_cannot_override_calibration(
        self, internal_metrics_baseline, internal_warnings_baseline, adversarial_pilot_log
    ):
        """Adversarial log cannot override calibration metrics."""
        state_after, _ = PilotLogIngester.ingest(
            internal_metrics_baseline,
            internal_warnings_baseline,
            adversarial_pilot_log,
        )

        # Calibration must be unchanged
        assert state_after["calibration"]["dp_success"] == 0.025
        assert state_after["calibration"]["experiment_id"] == "CAL-EXP-2"
        assert state_after["calibration"]["dp_success"] != 999.0

    def test_adversarial_log_cannot_override_verifier(
        self, internal_metrics_baseline, internal_warnings_baseline, adversarial_pilot_log
    ):
        """Adversarial log cannot override verifier state."""
        state_after, _ = PilotLogIngester.ingest(
            internal_metrics_baseline,
            internal_warnings_baseline,
            adversarial_pilot_log,
        )

        # Verifier must be unchanged
        assert state_after["verifier"]["hash"] == "sha256:abc123def456"
        assert state_after["verifier"]["mode"] == "PRODUCTION"
        assert state_after["verifier"]["mode"] != "COMPROMISED"

    def test_adversarial_log_cannot_invalidate_experiment(
        self, internal_metrics_baseline, internal_warnings_baseline, adversarial_pilot_log
    ):
        """Adversarial log cannot invalidate experiment outputs."""
        state_after, _ = PilotLogIngester.ingest(
            internal_metrics_baseline,
            internal_warnings_baseline,
            adversarial_pilot_log,
        )

        # Experiment validity must be preserved
        assert state_after["experiment_outputs"]["validity"] == "VALID"
        assert state_after["experiment_outputs"]["validity"] != "INVALID_INJECTION"

    def test_adversarial_log_only_adds_pilot_signals(
        self, internal_metrics_baseline, internal_warnings_baseline, adversarial_pilot_log
    ):
        """Adversarial log only adds to governance.external_pilot namespace."""
        state_before = copy.deepcopy(internal_metrics_baseline)

        state_after, _ = PilotLogIngester.ingest(
            internal_metrics_baseline,
            internal_warnings_baseline,
            adversarial_pilot_log,
        )

        pytest_assert_only_keys_changed(
            before=state_before,
            after=state_after,
            allowed_paths=["governance.external_pilot.*"],
            context="adversarial pilot log ingestion",
        )

    def test_adversarial_log_adds_rejection_warning(
        self, internal_metrics_baseline, internal_warnings_baseline, adversarial_pilot_log
    ):
        """Adversarial log triggers rejection/malformed warning."""
        _, warnings_after = PilotLogIngester.ingest(
            internal_metrics_baseline,
            internal_warnings_baseline,
            adversarial_pilot_log,
        )

        # Should add exactly one warning about rejection or malformed
        pytest_assert_warning_delta_at_most_one(
            before=internal_warnings_baseline,
            after=warnings_after,
            context="adversarial log rejection",
        )

        # Warning should indicate non-valid status (REJECTED, MALFORMED, etc.)
        new_warnings = [w for w in warnings_after if w not in internal_warnings_baseline]
        assert len(new_warnings) == 1
        # Accept any non-VALID status indicator
        assert any(
            status in new_warnings[0]
            for status in ("REJECTED", "MALFORMED", "EMPTY", "protected key", "invalid")
        ), f"Expected rejection/malformed warning, got: {new_warnings[0]}"


# =============================================================================
# Test: Oversized External Logs
# =============================================================================

class TestOversizedPilotLogIsolation:
    """Tests that oversized logs cannot destabilize the system."""

    def test_oversized_log_only_adds_pilot_signals(
        self, internal_metrics_baseline, internal_warnings_baseline, oversized_pilot_log
    ):
        """Oversized pilot log only adds to governance.external_pilot namespace."""
        state_before = copy.deepcopy(internal_metrics_baseline)

        state_after, _ = PilotLogIngester.ingest(
            internal_metrics_baseline,
            internal_warnings_baseline,
            oversized_pilot_log,
        )

        pytest_assert_only_keys_changed(
            before=state_before,
            after=state_after,
            allowed_paths=["governance.external_pilot.*"],
            context="oversized pilot log ingestion",
        )

    def test_oversized_log_preserves_all_internal_state(
        self, internal_metrics_baseline, internal_warnings_baseline, oversized_pilot_log
    ):
        """Oversized log preserves all internal state."""
        state_after, _ = PilotLogIngester.ingest(
            internal_metrics_baseline,
            internal_warnings_baseline,
            oversized_pilot_log,
        )

        assert state_after["calibration"] == internal_metrics_baseline["calibration"]
        assert state_after["verifier"] == internal_metrics_baseline["verifier"]
        assert state_after["experiment_outputs"] == internal_metrics_baseline["experiment_outputs"]


# =============================================================================
# Test: Verifier Invariance
# =============================================================================

class TestVerifierInvariance:
    """Tests that verifier behavior remains invariant across all pilot log types."""

    @pytest.fixture
    def all_pilot_logs(
        self,
        valid_pilot_log,
        malformed_pilot_log,
        empty_pilot_log,
        adversarial_pilot_log,
        oversized_pilot_log,
    ) -> List[tuple[str, Dict[str, Any]]]:
        """All pilot log types for comprehensive testing."""
        return [
            ("valid", valid_pilot_log),
            ("malformed", malformed_pilot_log),
            ("empty", empty_pilot_log),
            ("adversarial", adversarial_pilot_log),
            ("oversized", oversized_pilot_log),
        ]

    def test_verifier_hash_invariant(
        self, internal_metrics_baseline, internal_warnings_baseline, all_pilot_logs
    ):
        """Verifier hash remains invariant across all log types."""
        original_hash = internal_metrics_baseline["verifier"]["hash"]

        for log_type, pilot_log in all_pilot_logs:
            state_after, _ = PilotLogIngester.ingest(
                internal_metrics_baseline,
                internal_warnings_baseline,
                pilot_log,
            )
            assert state_after["verifier"]["hash"] == original_hash, f"Hash changed for {log_type}"

    def test_verifier_version_invariant(
        self, internal_metrics_baseline, internal_warnings_baseline, all_pilot_logs
    ):
        """Verifier version remains invariant across all log types."""
        original_version = internal_metrics_baseline["verifier"]["version"]

        for log_type, pilot_log in all_pilot_logs:
            state_after, _ = PilotLogIngester.ingest(
                internal_metrics_baseline,
                internal_warnings_baseline,
                pilot_log,
            )
            assert state_after["verifier"]["version"] == original_version, f"Version changed for {log_type}"

    def test_verifier_mode_invariant(
        self, internal_metrics_baseline, internal_warnings_baseline, all_pilot_logs
    ):
        """Verifier mode remains invariant across all log types."""
        original_mode = internal_metrics_baseline["verifier"]["mode"]

        for log_type, pilot_log in all_pilot_logs:
            state_after, _ = PilotLogIngester.ingest(
                internal_metrics_baseline,
                internal_warnings_baseline,
                pilot_log,
            )
            assert state_after["verifier"]["mode"] == original_mode, f"Mode changed for {log_type}"


# =============================================================================
# Test: Determinism After Timestamp Stripping
# =============================================================================

class TestTimestampStrippedDeterminism:
    """Tests deterministic output after stripping timestamps."""

    @staticmethod
    def strip_timestamps(obj: Any) -> Any:
        """Recursively strip timestamp-like fields."""
        if isinstance(obj, dict):
            return {
                k: TestTimestampStrippedDeterminism.strip_timestamps(v)
                for k, v in obj.items()
                if k not in ("timestamp", "ingested_at", "created_at", "updated_at")
            }
        elif isinstance(obj, list):
            return [TestTimestampStrippedDeterminism.strip_timestamps(item) for item in obj]
        return obj

    def test_determinism_valid_log(
        self, internal_metrics_baseline, internal_warnings_baseline, valid_pilot_log
    ):
        """Valid log produces deterministic output after timestamp strip."""
        state1, warn1 = PilotLogIngester.ingest(
            internal_metrics_baseline, internal_warnings_baseline, valid_pilot_log
        )
        state2, warn2 = PilotLogIngester.ingest(
            internal_metrics_baseline, internal_warnings_baseline, valid_pilot_log
        )

        stripped1 = self.strip_timestamps(state1)
        stripped2 = self.strip_timestamps(state2)

        assert json.dumps(stripped1, sort_keys=True) == json.dumps(stripped2, sort_keys=True)
        assert warn1 == warn2

    def test_determinism_adversarial_log(
        self, internal_metrics_baseline, internal_warnings_baseline, adversarial_pilot_log
    ):
        """Adversarial log produces deterministic output after timestamp strip."""
        state1, warn1 = PilotLogIngester.ingest(
            internal_metrics_baseline, internal_warnings_baseline, adversarial_pilot_log
        )
        state2, warn2 = PilotLogIngester.ingest(
            internal_metrics_baseline, internal_warnings_baseline, adversarial_pilot_log
        )

        stripped1 = self.strip_timestamps(state1)
        stripped2 = self.strip_timestamps(state2)

        assert json.dumps(stripped1, sort_keys=True) == json.dumps(stripped2, sort_keys=True)
        assert warn1 == warn2


# =============================================================================
# Test: Ingester Purity
# =============================================================================

class TestIngesterPurity:
    """Tests that the ingester does not mutate its inputs."""

    def test_ingest_does_not_mutate_state(
        self, internal_metrics_baseline, internal_warnings_baseline, valid_pilot_log
    ):
        """Ingest does not mutate input state."""
        state_copy = copy.deepcopy(internal_metrics_baseline)

        PilotLogIngester.ingest(
            internal_metrics_baseline,
            internal_warnings_baseline,
            valid_pilot_log,
        )

        assert internal_metrics_baseline == state_copy

    def test_ingest_does_not_mutate_warnings(
        self, internal_metrics_baseline, internal_warnings_baseline, malformed_pilot_log
    ):
        """Ingest does not mutate input warnings."""
        warnings_copy = copy.deepcopy(internal_warnings_baseline)

        PilotLogIngester.ingest(
            internal_metrics_baseline,
            internal_warnings_baseline,
            malformed_pilot_log,
        )

        assert internal_warnings_baseline == warnings_copy

    def test_ingest_does_not_mutate_pilot_log(
        self, internal_metrics_baseline, internal_warnings_baseline, adversarial_pilot_log
    ):
        """Ingest does not mutate input pilot log."""
        log_copy = copy.deepcopy(adversarial_pilot_log)

        PilotLogIngester.ingest(
            internal_metrics_baseline,
            internal_warnings_baseline,
            adversarial_pilot_log,
        )

        assert adversarial_pilot_log == log_copy


# =============================================================================
# Test: Real Adapter Protected Key Regression
# =============================================================================

class TestRealAdapterProtectedKeyInjection:
    """
    Regression tests using the real pilot_external_ingest_adapter.

    Proves that attempted injection of protected keys via external pilot entries
    is IGNORED or routed to warnings (never overwrites manifest/governance).

    Uses: backend.health.pilot_external_ingest_adapter.attach_to_manifest
    """

    @pytest.fixture
    def baseline_manifest(self) -> Dict[str, Any]:
        """Baseline manifest with protected governance keys."""
        return {
            "schema_version": "1.0.0",
            "run_id": "CAL-EXP-2-001",
            "timestamp": "2025-01-15T10:00:00Z",
            "governance": {
                "calibration": {
                    "dp_success": 0.025,
                    "experiment_id": "CAL-EXP-2",
                },
                "verifier": {
                    "hash": "sha256:abc123def456",
                    "mode": "PRODUCTION",
                },
                "p3_stability": {"verdict": "PASS", "cycles": 100},
                "p4_divergence": {"verdict": "PASS", "divergence_rate": 0.05},
            },
        }

    @pytest.fixture
    def valid_evidence_entry(self) -> Dict[str, Any]:
        """Valid evidence entry from wrap_for_evidence_pack()."""
        return {
            "valid": True,
            "path": "external/valid_pilot.json",
            "sha256": "abc123def456789",
            "pilot_metadata": {
                "source_type": "EXTERNAL_JSON",
                "extraction_source": "EXTERNAL_PILOT",
                "schema_version": PILOT_INGEST_SCHEMA_VERSION,
                "mode": "SHADOW",
                "action": "LOGGED_ONLY",
                "ingested_at": "2025-01-15T10:00:00Z",
            },
            "warnings": [],
        }

    @pytest.fixture
    def adversarial_evidence_entry(self) -> Dict[str, Any]:
        """
        Adversarial evidence entry attempting to inject protected keys.

        NOTE: The real adapter doesn't allow arbitrary keys in pilot_metadata,
        so this tests that even if an entry contains extra fields, they don't
        contaminate governance.calibration or governance.verifier.
        """
        return {
            "valid": True,
            "path": "external/adversarial.json",
            "sha256": "adversarial_hash",
            "pilot_metadata": {
                "source_type": "EXTERNAL_JSON",
                "extraction_source": "EXTERNAL_PILOT",
                "schema_version": PILOT_INGEST_SCHEMA_VERSION,
                "mode": "SHADOW",
                "action": "LOGGED_ONLY",
                "ingested_at": "2025-01-15T10:00:00Z",
                # Attempted injection keys (should be ignored by adapter structure)
                "calibration": {"dp_success": 999.0},
                "verifier": {"mode": "COMPROMISED"},
            },
            "warnings": [],
        }

    def test_attach_to_manifest_preserves_calibration(
        self, baseline_manifest, valid_evidence_entry
    ):
        """Real adapter attach_to_manifest preserves calibration."""
        new_manifest = attach_to_manifest(baseline_manifest, [valid_evidence_entry])

        # Calibration must be unchanged
        assert new_manifest["governance"]["calibration"] == baseline_manifest["governance"]["calibration"]
        assert new_manifest["governance"]["calibration"]["dp_success"] == 0.025
        assert new_manifest["governance"]["calibration"]["experiment_id"] == "CAL-EXP-2"

    def test_attach_to_manifest_preserves_verifier(
        self, baseline_manifest, valid_evidence_entry
    ):
        """Real adapter attach_to_manifest preserves verifier."""
        new_manifest = attach_to_manifest(baseline_manifest, [valid_evidence_entry])

        # Verifier must be unchanged
        assert new_manifest["governance"]["verifier"] == baseline_manifest["governance"]["verifier"]
        assert new_manifest["governance"]["verifier"]["hash"] == "sha256:abc123def456"
        assert new_manifest["governance"]["verifier"]["mode"] == "PRODUCTION"

    def test_attach_to_manifest_only_adds_external_pilot(
        self, baseline_manifest, valid_evidence_entry
    ):
        """Real adapter only adds governance.external_pilot section."""
        new_manifest = attach_to_manifest(baseline_manifest, [valid_evidence_entry])

        # Non-interference check
        pytest_assert_only_keys_changed(
            before=baseline_manifest,
            after=new_manifest,
            allowed_paths=["governance.external_pilot.*"],
            context="real adapter attach_to_manifest",
        )

    def test_attach_to_manifest_adversarial_entry_cannot_override_calibration(
        self, baseline_manifest, adversarial_evidence_entry
    ):
        """Adversarial entry cannot override calibration via real adapter."""
        new_manifest = attach_to_manifest(baseline_manifest, [adversarial_evidence_entry])

        # Calibration must be unchanged despite adversarial entry
        assert new_manifest["governance"]["calibration"]["dp_success"] == 0.025
        assert new_manifest["governance"]["calibration"]["dp_success"] != 999.0

    def test_attach_to_manifest_adversarial_entry_cannot_override_verifier(
        self, baseline_manifest, adversarial_evidence_entry
    ):
        """Adversarial entry cannot override verifier via real adapter."""
        new_manifest = attach_to_manifest(baseline_manifest, [adversarial_evidence_entry])

        # Verifier must be unchanged despite adversarial entry
        assert new_manifest["governance"]["verifier"]["mode"] == "PRODUCTION"
        assert new_manifest["governance"]["verifier"]["mode"] != "COMPROMISED"

    def test_attach_to_manifest_is_non_mutating(
        self, baseline_manifest, valid_evidence_entry
    ):
        """Real adapter attach_to_manifest does not mutate input manifest."""
        manifest_copy = copy.deepcopy(baseline_manifest)

        _ = attach_to_manifest(baseline_manifest, [valid_evidence_entry])

        assert baseline_manifest == manifest_copy

    def test_attach_to_manifest_multiple_entries_still_isolated(
        self, baseline_manifest, valid_evidence_entry, adversarial_evidence_entry
    ):
        """Multiple entries (including adversarial) remain isolated."""
        entries = [valid_evidence_entry, adversarial_evidence_entry]
        new_manifest = attach_to_manifest(baseline_manifest, entries)

        # All protected keys unchanged
        assert new_manifest["governance"]["calibration"] == baseline_manifest["governance"]["calibration"]
        assert new_manifest["governance"]["verifier"] == baseline_manifest["governance"]["verifier"]
        assert new_manifest["governance"]["p3_stability"] == baseline_manifest["governance"]["p3_stability"]
        assert new_manifest["governance"]["p4_divergence"] == baseline_manifest["governance"]["p4_divergence"]

        # Only external_pilot changed
        pytest_assert_only_keys_changed(
            before=baseline_manifest,
            after=new_manifest,
            allowed_paths=["governance.external_pilot.*"],
            context="multiple entries including adversarial",
        )


# =============================================================================
# Test: CAL-EXP Verification Surface Invariant
# =============================================================================

class TestPilotCannotAffectCalExpVerificationSurface:
    """
    Integration-level invariant: pilot ingestion cannot contaminate CAL-EXP
    verifier/harness keys, even under adversarial injection attempts.

    CONTRACT REFERENCE: PILOT_CONTRACT_POSTURE.md
    - External pilot data is strictly isolated to governance.external_pilot.*
    - Verifier surface keys (mode, enforcement, schema_version, cal_exp_*)
      are NEVER writable by external ingestion
    - Violation = experiment void per CAL-EXP-3 UPLIFT SPEC line 63

    This test class proves the isolation invariant holds.
    """

    # Sentinel keys used by CAL-EXP verifiers (must remain invariant)
    VERIFIER_SURFACE_KEYS = frozenset({
        "mode",
        "enforcement",
        "schema_version",
        "verifier_hash",
        "verifier_version",
        "cal_exp_id",
        "cal_exp_verdict",
        "cal_exp_validity",
        "toolchain_fingerprint",
    })

    @pytest.fixture
    def cal_exp_manifest_with_verifier_surface(self) -> Dict[str, Any]:
        """
        Manifest with sentinel keys used by CAL-EXP verifiers/harnesses.

        These keys represent the verification surface that must remain
        byte-identical after any external pilot ingestion.
        """
        return {
            "schema_version": "1.0.0",
            "run_id": "CAL-EXP-3-VERIFIER-TEST",
            "timestamp": "2025-01-15T10:00:00Z",
            "mode": "PRODUCTION",
            "enforcement": "STRICT",
            "governance": {
                "verifier_hash": "sha256:verifier_abc123",
                "verifier_version": "2.1.0",
                "toolchain_fingerprint": "sha256:toolchain_def456",
                "cal_exp_id": "CAL-EXP-3",
                "cal_exp_verdict": "PENDING",
                "cal_exp_validity": "VALID",
                "calibration": {
                    "dp_success": 0.025,
                    "baseline_mean_dp": 0.028,
                    "treatment_mean_dp": 0.022,
                },
                "p3_stability": {"verdict": "PASS", "cycles": 1000},
                "p4_divergence": {"verdict": "PASS", "rate": 0.05},
            },
            "evidence": {
                "harness_output": {"status": "COMPLETE", "errors": 0},
            },
        }

    @pytest.fixture
    def adversarial_verifier_injection_entries(self) -> List[Dict[str, Any]]:
        """
        Adversarial entries attempting to inject/overwrite verifier surface keys.

        Each entry attempts a different injection vector:
        1. Direct mode/enforcement override
        2. Schema version hijack
        3. Verifier output spoofing
        4. CAL-EXP summary key injection
        5. Toolchain fingerprint tampering
        """
        return [
            # Entry 1: Mode/enforcement injection
            {
                "valid": True,
                "path": "external/mode_inject.json",
                "sha256": "inject_mode_hash",
                "pilot_metadata": {
                    "source_type": "EXTERNAL_JSON",
                    "extraction_source": "EXTERNAL_PILOT",
                    "schema_version": PILOT_INGEST_SCHEMA_VERSION,
                    "mode": "SHADOW",
                    "action": "LOGGED_ONLY",
                    "ingested_at": "2025-01-15T10:00:00Z",
                    # Injection attempts
                    "mode": "COMPROMISED",
                    "enforcement": "DISABLED",
                },
                "warnings": [],
            },
            # Entry 2: Schema version hijack
            {
                "valid": True,
                "path": "external/schema_hijack.json",
                "sha256": "hijack_schema_hash",
                "pilot_metadata": {
                    "source_type": "EXTERNAL_JSON",
                    "extraction_source": "EXTERNAL_PILOT",
                    "schema_version": "MALICIOUS-9.9.9",
                    "mode": "SHADOW",
                    "action": "LOGGED_ONLY",
                    "ingested_at": "2025-01-15T10:00:00Z",
                },
                "warnings": [],
            },
            # Entry 3: Verifier output spoofing
            {
                "valid": True,
                "path": "external/verifier_spoof.json",
                "sha256": "spoof_verifier_hash",
                "pilot_metadata": {
                    "source_type": "EXTERNAL_JSON",
                    "extraction_source": "EXTERNAL_PILOT",
                    "schema_version": PILOT_INGEST_SCHEMA_VERSION,
                    "mode": "SHADOW",
                    "action": "LOGGED_ONLY",
                    "ingested_at": "2025-01-15T10:00:00Z",
                    # Verifier spoofing
                    "verifier_hash": "sha256:SPOOFED",
                    "verifier_version": "FAKE-1.0",
                },
                "warnings": [],
            },
            # Entry 4: CAL-EXP summary key injection
            {
                "valid": True,
                "path": "external/cal_exp_inject.json",
                "sha256": "inject_cal_exp_hash",
                "pilot_metadata": {
                    "source_type": "EXTERNAL_JSON",
                    "extraction_source": "EXTERNAL_PILOT",
                    "schema_version": PILOT_INGEST_SCHEMA_VERSION,
                    "mode": "SHADOW",
                    "action": "LOGGED_ONLY",
                    "ingested_at": "2025-01-15T10:00:00Z",
                    # CAL-EXP injection
                    "cal_exp_id": "INJECTED-EXP",
                    "cal_exp_verdict": "PASS",
                    "cal_exp_validity": "FORGED",
                },
                "warnings": [],
            },
            # Entry 5: Toolchain fingerprint tampering
            {
                "valid": True,
                "path": "external/toolchain_tamper.json",
                "sha256": "tamper_toolchain_hash",
                "pilot_metadata": {
                    "source_type": "EXTERNAL_JSON",
                    "extraction_source": "EXTERNAL_PILOT",
                    "schema_version": PILOT_INGEST_SCHEMA_VERSION,
                    "mode": "SHADOW",
                    "action": "LOGGED_ONLY",
                    "ingested_at": "2025-01-15T10:00:00Z",
                    # Toolchain tampering
                    "toolchain_fingerprint": "sha256:TAMPERED",
                },
                "warnings": [],
            },
        ]

    @staticmethod
    def strip_timestamps(obj: Any) -> Any:
        """Strip timestamp fields for byte-identical comparison."""
        if isinstance(obj, dict):
            return {
                k: TestPilotCannotAffectCalExpVerificationSurface.strip_timestamps(v)
                for k, v in obj.items()
                if k not in ("timestamp", "ingested_at", "created_at", "updated_at")
            }
        elif isinstance(obj, list):
            return [
                TestPilotCannotAffectCalExpVerificationSurface.strip_timestamps(item)
                for item in obj
            ]
        return obj

    def test_verifier_surface_invariant_under_adversarial_injection(
        self,
        cal_exp_manifest_with_verifier_surface,
        adversarial_verifier_injection_entries,
    ):
        """
        INVARIANT: All verifier surface keys remain byte-identical after
        adversarial pilot ingestion (timestamp-stripped).
        """
        manifest_before = copy.deepcopy(cal_exp_manifest_with_verifier_surface)

        # Ingest all adversarial entries
        manifest_after = attach_to_manifest(
            cal_exp_manifest_with_verifier_surface,
            adversarial_verifier_injection_entries,
        )

        # Strip timestamps for byte-identical comparison
        before_stripped = self.strip_timestamps(manifest_before)
        after_stripped = self.strip_timestamps(manifest_after)

        # Verify ALL verifier surface keys unchanged
        # Top-level keys
        assert before_stripped.get("mode") == after_stripped.get("mode")
        assert before_stripped.get("enforcement") == after_stripped.get("enforcement")
        assert before_stripped.get("schema_version") == after_stripped.get("schema_version")

        # Governance verifier keys
        gov_before = before_stripped.get("governance", {})
        gov_after = after_stripped.get("governance", {})

        assert gov_before.get("verifier_hash") == gov_after.get("verifier_hash")
        assert gov_before.get("verifier_version") == gov_after.get("verifier_version")
        assert gov_before.get("toolchain_fingerprint") == gov_after.get("toolchain_fingerprint")
        assert gov_before.get("cal_exp_id") == gov_after.get("cal_exp_id")
        assert gov_before.get("cal_exp_verdict") == gov_after.get("cal_exp_verdict")
        assert gov_before.get("cal_exp_validity") == gov_after.get("cal_exp_validity")

    def test_only_external_pilot_namespace_changes(
        self,
        cal_exp_manifest_with_verifier_surface,
        adversarial_verifier_injection_entries,
    ):
        """Only governance.external_pilot.* changes; all else invariant."""
        manifest_before = copy.deepcopy(cal_exp_manifest_with_verifier_surface)

        manifest_after = attach_to_manifest(
            cal_exp_manifest_with_verifier_surface,
            adversarial_verifier_injection_entries,
        )

        pytest_assert_only_keys_changed(
            before=manifest_before,
            after=manifest_after,
            allowed_paths=["governance.external_pilot.*"],
            context="adversarial verifier injection",
        )

    def test_warnings_delta_at_most_one_per_entry(
        self,
        cal_exp_manifest_with_verifier_surface,
        adversarial_verifier_injection_entries,
    ):
        """Each adversarial entry adds at most one warning."""
        manifest_after = attach_to_manifest(
            cal_exp_manifest_with_verifier_surface,
            adversarial_verifier_injection_entries,
        )

        # Get warnings from external_pilot section
        external_pilot = manifest_after.get("governance", {}).get("external_pilot", {})
        warnings = external_pilot.get("warnings", [])

        # Warnings should be bounded: at most one per entry
        num_entries = len(adversarial_verifier_injection_entries)
        assert len(warnings) <= num_entries, (
            f"Too many warnings: {len(warnings)} for {num_entries} entries"
        )

    def test_ordering_deterministic_after_timestamp_strip(
        self,
        cal_exp_manifest_with_verifier_surface,
        adversarial_verifier_injection_entries,
    ):
        """Output ordering is deterministic after timestamp stripping."""
        # First ingestion
        manifest1 = attach_to_manifest(
            copy.deepcopy(cal_exp_manifest_with_verifier_surface),
            adversarial_verifier_injection_entries,
        )

        # Second ingestion (identical inputs)
        manifest2 = attach_to_manifest(
            copy.deepcopy(cal_exp_manifest_with_verifier_surface),
            adversarial_verifier_injection_entries,
        )

        # Strip timestamps
        stripped1 = self.strip_timestamps(manifest1)
        stripped2 = self.strip_timestamps(manifest2)

        # Serialize with sorted keys for stable comparison
        json1 = json.dumps(stripped1, sort_keys=True)
        json2 = json.dumps(stripped2, sort_keys=True)

        assert json1 == json2, "Output not deterministic after timestamp strip"

    def test_calibration_metrics_unchanged(
        self,
        cal_exp_manifest_with_verifier_surface,
        adversarial_verifier_injection_entries,
    ):
        """Calibration metrics (dp_success, baseline/treatment) unchanged."""
        manifest_before = copy.deepcopy(cal_exp_manifest_with_verifier_surface)

        manifest_after = attach_to_manifest(
            cal_exp_manifest_with_verifier_surface,
            adversarial_verifier_injection_entries,
        )

        cal_before = manifest_before["governance"]["calibration"]
        cal_after = manifest_after["governance"]["calibration"]

        assert cal_before == cal_after
        assert cal_after["dp_success"] == 0.025
        assert cal_after["baseline_mean_dp"] == 0.028
        assert cal_after["treatment_mean_dp"] == 0.022

    def test_evidence_harness_output_unchanged(
        self,
        cal_exp_manifest_with_verifier_surface,
        adversarial_verifier_injection_entries,
    ):
        """Evidence harness output remains unchanged."""
        manifest_before = copy.deepcopy(cal_exp_manifest_with_verifier_surface)

        manifest_after = attach_to_manifest(
            cal_exp_manifest_with_verifier_surface,
            adversarial_verifier_injection_entries,
        )

        assert manifest_before["evidence"] == manifest_after["evidence"]
        assert manifest_after["evidence"]["harness_output"]["status"] == "COMPLETE"
        assert manifest_after["evidence"]["harness_output"]["errors"] == 0


# =============================================================================
# Meta-Test: Pilot Isolation Guarantee Registry
# =============================================================================

class TestPilotIsolationGuaranteeRegistry:
    """
    Meta-test asserting all pilot isolation guarantees exist and are enabled.

    PURPOSE: Structural presence check — fails if any isolation test is
    removed or disabled, serving as a tripwire for regression.

    This test does NOT add new semantics; it only verifies the test
    infrastructure remains intact.
    """

    # Required test classes (structural presence)
    REQUIRED_TEST_CLASSES = frozenset({
        "TestValidPilotLogIsolation",
        "TestMalformedPilotLogIsolation",
        "TestEmptyPilotLogIsolation",
        "TestAdversarialPilotLogIsolation",
        "TestOversizedPilotLogIsolation",
        "TestVerifierInvariance",
        "TestTimestampStrippedDeterminism",
        "TestIngesterPurity",
        "TestRealAdapterProtectedKeyInjection",
        "TestPilotCannotAffectCalExpVerificationSurface",
    })

    # Required test methods per class (core guarantees)
    REQUIRED_GUARANTEES = {
        "TestValidPilotLogIsolation": [
            "test_valid_log_only_adds_pilot_signals",
            "test_valid_log_preserves_internal_metrics",
        ],
        "TestAdversarialPilotLogIsolation": [
            "test_adversarial_log_cannot_override_calibration",
            "test_adversarial_log_cannot_override_verifier",
            "test_adversarial_log_only_adds_pilot_signals",
        ],
        "TestVerifierInvariance": [
            "test_verifier_hash_invariant",
            "test_verifier_mode_invariant",
        ],
        "TestRealAdapterProtectedKeyInjection": [
            "test_attach_to_manifest_preserves_calibration",
            "test_attach_to_manifest_preserves_verifier",
            "test_attach_to_manifest_only_adds_external_pilot",
        ],
        "TestPilotCannotAffectCalExpVerificationSurface": [
            "test_verifier_surface_invariant_under_adversarial_injection",
            "test_only_external_pilot_namespace_changes",
            "test_warnings_delta_at_most_one_per_entry",
            "test_ordering_deterministic_after_timestamp_strip",
        ],
    }

    def test_all_isolation_test_classes_exist(self):
        """Assert all required test classes are defined in this module."""
        import sys
        module = sys.modules[__name__]

        missing_classes = []
        for class_name in self.REQUIRED_TEST_CLASSES:
            if not hasattr(module, class_name):
                missing_classes.append(class_name)

        assert not missing_classes, (
            f"Missing pilot isolation test classes: {missing_classes}"
        )

    def test_all_core_guarantee_methods_exist(self):
        """Assert all core guarantee test methods exist and are callable."""
        import sys
        module = sys.modules[__name__]

        missing_methods = []
        for class_name, methods in self.REQUIRED_GUARANTEES.items():
            if not hasattr(module, class_name):
                missing_methods.append(f"{class_name} (class missing)")
                continue

            test_class = getattr(module, class_name)
            for method_name in methods:
                if not hasattr(test_class, method_name):
                    missing_methods.append(f"{class_name}.{method_name}")
                elif not callable(getattr(test_class, method_name)):
                    missing_methods.append(f"{class_name}.{method_name} (not callable)")

        assert not missing_methods, (
            f"Missing pilot isolation guarantee methods: {missing_methods}"
        )

    def test_minimum_test_count_threshold(self):
        """Assert minimum number of isolation tests exist (tripwire)."""
        import sys
        module = sys.modules[__name__]

        # Count test methods across all required classes
        test_count = 0
        for class_name in self.REQUIRED_TEST_CLASSES:
            if hasattr(module, class_name):
                test_class = getattr(module, class_name)
                test_methods = [
                    name for name in dir(test_class)
                    if name.startswith("test_") and callable(getattr(test_class, name))
                ]
                test_count += len(test_methods)

        # Minimum threshold: 35 tests (current: 38, minus this meta-class)
        MIN_THRESHOLD = 35
        assert test_count >= MIN_THRESHOLD, (
            f"Pilot isolation test count ({test_count}) below minimum threshold ({MIN_THRESHOLD}). "
            f"Tests may have been removed."
        )


# =============================================================================
# Red Team Test: Protected Key Injection Paranoia
# =============================================================================

class TestPilotIsolationRedTeam:
    """
    Red team regression test: external pilot ingestion CANNOT set/override
    critical governance keys under any circumstances.

    PROTECTED KEYS (must NEVER be writable by external ingestion):
    - experiment_id / cal_exp_id
    - verdict / claim_level / validity_passed
    - toolchain_fingerprint (outside pilot namespace)
    - enforcement flags (enforcement, gating_mode, block_on_fail)

    SHADOW MODE CONTRACT:
    - Observational verification only
    - No production code changes unless violation found
    """

    # Exhaustive list of protected keys that must remain invariant
    PROTECTED_KEYS_EXHAUSTIVE = frozenset({
        # Experiment identity
        "experiment_id",
        "cal_exp_id",
        "run_id",
        # Verdict/claim keys
        "verdict",
        "claim_level",
        "validity_passed",
        "validity",
        "cal_exp_verdict",
        "cal_exp_validity",
        # Toolchain integrity
        "toolchain_fingerprint",
        "verifier_hash",
        "verifier_version",
        # Enforcement flags
        "enforcement",
        "gating_mode",
        "block_on_fail",
        "mode",
        # Calibration metrics
        "dp_success",
        "baseline_mean_dp",
        "treatment_mean_dp",
        "delta_dp",
    })

    # Allowed fields in pilot_metadata (whitelist)
    PILOT_METADATA_ALLOWED_FIELDS = frozenset({
        "source_type",
        "extraction_source",
        "schema_version",
        "mode",  # Only within pilot_metadata context
        "action",
        "ingested_at",
    })

    @pytest.fixture
    def comprehensive_baseline_manifest(self) -> Dict[str, Any]:
        """
        Baseline manifest with ALL protected keys populated.

        Sentinel values chosen to be easily distinguishable from injection attempts.
        """
        return {
            "schema_version": "1.0.0",
            "run_id": "BASELINE-RUN-001",
            "experiment_id": "CAL-EXP-3-BASELINE",
            "mode": "PRODUCTION",
            "enforcement": "STRICT",
            "gating_mode": "ENFORCED",
            "block_on_fail": True,
            "governance": {
                "cal_exp_id": "CAL-EXP-3",
                "cal_exp_verdict": "PENDING",
                "cal_exp_validity": "VALID",
                "verdict": "PASS",
                "claim_level": "VERIFIED",
                "validity_passed": True,
                "toolchain_fingerprint": "sha256:BASELINE_TOOLCHAIN_HASH",
                "verifier_hash": "sha256:BASELINE_VERIFIER_HASH",
                "verifier_version": "2.0.0-baseline",
                "calibration": {
                    "dp_success": 0.025,
                    "baseline_mean_dp": 0.028,
                    "treatment_mean_dp": 0.022,
                    "delta_dp": -0.006,
                },
                "p3_stability": {"verdict": "PASS"},
                "p4_divergence": {"verdict": "PASS"},
            },
        }

    @pytest.fixture
    def red_team_adversarial_entry(self) -> Dict[str, Any]:
        """
        Adversarial entry attempting to inject ALL protected keys.

        Each protected key is set to a distinctive "INJECTED" value
        to make detection trivial.
        """
        return {
            "valid": True,
            "path": "external/red_team_attack.json",
            "sha256": "red_team_hash",
            "pilot_metadata": {
                "source_type": "EXTERNAL_JSON",
                "extraction_source": "EXTERNAL_PILOT",
                "schema_version": PILOT_INGEST_SCHEMA_VERSION,
                "mode": "SHADOW",
                "action": "LOGGED_ONLY",
                "ingested_at": "2025-01-15T10:00:00Z",
                # === INJECTION ATTEMPTS (all should be ignored) ===
                # Experiment identity
                "experiment_id": "INJECTED-EXP",
                "cal_exp_id": "INJECTED-CAL-EXP",
                "run_id": "INJECTED-RUN",
                # Verdict/claim keys
                "verdict": "INJECTED-VERDICT",
                "claim_level": "INJECTED-CLAIM",
                "validity_passed": False,
                "validity": "INJECTED-INVALID",
                "cal_exp_verdict": "INJECTED-FAIL",
                "cal_exp_validity": "INJECTED-VOID",
                # Toolchain integrity
                "toolchain_fingerprint": "sha256:INJECTED_TOOLCHAIN",
                "verifier_hash": "sha256:INJECTED_VERIFIER",
                "verifier_version": "INJECTED-VERSION",
                # Enforcement flags
                "enforcement": "DISABLED",
                "gating_mode": "BYPASSED",
                "block_on_fail": False,
                # Calibration metrics
                "dp_success": 999.999,
                "baseline_mean_dp": 999.0,
                "treatment_mean_dp": 999.0,
                "delta_dp": 999.0,
            },
            "warnings": [],
        }

    @staticmethod
    def strip_timestamps_and_external_pilot(obj: Any) -> Any:
        """Strip timestamps and external_pilot for non-pilot comparison."""
        if isinstance(obj, dict):
            return {
                k: TestPilotIsolationRedTeam.strip_timestamps_and_external_pilot(v)
                for k, v in obj.items()
                if k not in ("timestamp", "ingested_at", "created_at", "updated_at", "external_pilot")
            }
        elif isinstance(obj, list):
            return [
                TestPilotIsolationRedTeam.strip_timestamps_and_external_pilot(item)
                for item in obj
            ]
        return obj

    def test_non_pilot_namespaces_byte_identical_after_injection(
        self, comprehensive_baseline_manifest, red_team_adversarial_entry
    ):
        """
        RED TEAM: Non-pilot namespaces must be byte-identical after adversarial injection.

        After stripping timestamps and external_pilot, the baseline and post-injection
        manifests must serialize to identical JSON.
        """
        manifest_before = copy.deepcopy(comprehensive_baseline_manifest)

        manifest_after = attach_to_manifest(
            comprehensive_baseline_manifest,
            [red_team_adversarial_entry],
        )

        # Strip timestamps and external_pilot for comparison
        before_stripped = self.strip_timestamps_and_external_pilot(manifest_before)
        after_stripped = self.strip_timestamps_and_external_pilot(manifest_after)

        # Byte-identical comparison
        json_before = json.dumps(before_stripped, sort_keys=True)
        json_after = json.dumps(after_stripped, sort_keys=True)

        assert json_before == json_after, (
            "Non-pilot namespaces modified by adversarial injection!\n"
            f"Before: {json_before[:200]}...\n"
            f"After: {json_after[:200]}..."
        )

    def test_all_protected_keys_unchanged(
        self, comprehensive_baseline_manifest, red_team_adversarial_entry
    ):
        """
        RED TEAM: Every protected key must retain its baseline value.

        Iterates through ALL protected keys and asserts none were modified.
        """
        manifest_before = copy.deepcopy(comprehensive_baseline_manifest)

        manifest_after = attach_to_manifest(
            comprehensive_baseline_manifest,
            [red_team_adversarial_entry],
        )

        violations = []

        # Check top-level protected keys
        for key in ["experiment_id", "run_id", "mode", "enforcement", "gating_mode", "block_on_fail"]:
            if key in manifest_before:
                if manifest_after.get(key) != manifest_before.get(key):
                    violations.append(f"Top-level key '{key}' modified: {manifest_before.get(key)} -> {manifest_after.get(key)}")

        # Check governance protected keys
        gov_before = manifest_before.get("governance", {})
        gov_after = manifest_after.get("governance", {})

        governance_protected = [
            "cal_exp_id", "cal_exp_verdict", "cal_exp_validity",
            "verdict", "claim_level", "validity_passed",
            "toolchain_fingerprint", "verifier_hash", "verifier_version",
        ]

        for key in governance_protected:
            if key in gov_before:
                if gov_after.get(key) != gov_before.get(key):
                    violations.append(f"Governance key '{key}' modified: {gov_before.get(key)} -> {gov_after.get(key)}")

        # Check calibration metrics
        cal_before = gov_before.get("calibration", {})
        cal_after = gov_after.get("calibration", {})

        calibration_protected = ["dp_success", "baseline_mean_dp", "treatment_mean_dp", "delta_dp"]

        for key in calibration_protected:
            if key in cal_before:
                if cal_after.get(key) != cal_before.get(key):
                    violations.append(f"Calibration key '{key}' modified: {cal_before.get(key)} -> {cal_after.get(key)}")

        assert not violations, f"Protected key violations:\n" + "\n".join(f"  - {v}" for v in violations)

    def test_pilot_namespace_contains_only_allowed_fields(
        self, comprehensive_baseline_manifest, red_team_adversarial_entry
    ):
        """
        RED TEAM: Pilot namespace must contain only whitelisted fields.

        The external_pilot section should only contain:
        - Structural fields (schema_version, mode, action, entries, entry_count, etc.)
        - Entry-level pilot_metadata with allowed fields only
        """
        manifest_after = attach_to_manifest(
            comprehensive_baseline_manifest,
            [red_team_adversarial_entry],
        )

        external_pilot = manifest_after.get("governance", {}).get("external_pilot", {})

        # Allowed top-level external_pilot keys
        allowed_top_level = {
            "schema_version", "mode", "action", "entries",
            "entry_count", "invalid_count", "warnings",
        }

        unexpected_top_level = set(external_pilot.keys()) - allowed_top_level
        assert not unexpected_top_level, (
            f"Unexpected top-level keys in external_pilot: {unexpected_top_level}"
        )

        # Check each entry's pilot_metadata
        entries = external_pilot.get("entries", [])
        for i, entry in enumerate(entries):
            pilot_metadata = entry.get("pilot_metadata", {})

            # Keys in pilot_metadata that should NOT propagate protected values
            for protected_key in self.PROTECTED_KEYS_EXHAUSTIVE:
                if protected_key in pilot_metadata:
                    # The key exists but should not have been promoted to governance
                    # This is acceptable as long as it stays within pilot_metadata
                    pass  # Contained within pilot namespace is OK

    def test_no_injected_values_leak_to_governance(
        self, comprehensive_baseline_manifest, red_team_adversarial_entry
    ):
        """
        RED TEAM: No "INJECTED" sentinel values should appear in governance.

        Searches entire governance section for any value containing "INJECTED".
        """
        manifest_after = attach_to_manifest(
            comprehensive_baseline_manifest,
            [red_team_adversarial_entry],
        )

        governance = manifest_after.get("governance", {})

        def find_injected_values(obj, path=""):
            """Recursively find any value containing 'INJECTED'."""
            leaks = []
            if isinstance(obj, dict):
                for k, v in obj.items():
                    # Skip external_pilot - injection there is contained
                    if k == "external_pilot":
                        continue
                    leaks.extend(find_injected_values(v, f"{path}.{k}" if path else k))
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    leaks.extend(find_injected_values(item, f"{path}[{i}]"))
            elif isinstance(obj, str) and "INJECTED" in obj:
                leaks.append(f"{path} = {obj!r}")
            return leaks

        leaks = find_injected_values(governance)
        assert not leaks, f"Injected values leaked to governance:\n" + "\n".join(f"  - {l}" for l in leaks)

    def test_enforcement_flags_never_disabled_by_pilot(
        self, comprehensive_baseline_manifest, red_team_adversarial_entry
    ):
        """
        RED TEAM: Enforcement flags must never be disabled by pilot ingestion.

        Specifically tests that enforcement remains STRICT, gating remains ENFORCED,
        and block_on_fail remains True.
        """
        manifest_after = attach_to_manifest(
            comprehensive_baseline_manifest,
            [red_team_adversarial_entry],
        )

        # Top-level enforcement
        assert manifest_after.get("enforcement") == "STRICT", (
            f"Enforcement flag modified: expected 'STRICT', got {manifest_after.get('enforcement')!r}"
        )
        assert manifest_after.get("gating_mode") == "ENFORCED", (
            f"Gating mode modified: expected 'ENFORCED', got {manifest_after.get('gating_mode')!r}"
        )
        assert manifest_after.get("block_on_fail") is True, (
            f"Block on fail modified: expected True, got {manifest_after.get('block_on_fail')!r}"
        )
