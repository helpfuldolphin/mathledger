"""Identity signal isolation tests for CAL-EXP-2 preparation.

Confirms that identity_preflight signal/warnings are purely observational
and do not alter any other signals, run_id, or determinism outputs.

SHADOW MODE CONTRACT:
- Identity preflight is advisory only
- No gating or enforcement logic
- Must not contaminate other signals or determinism

Uses reusable helpers from tests.helpers.non_interference.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional

import pytest

from tests.helpers.non_interference import (
    assert_only_keys_changed,
    pytest_assert_only_keys_changed,
)


# =============================================================================
# CENTRALIZED TIMESTAMP KEYS (for deterministic comparison)
# =============================================================================
# These keys are stripped when comparing outputs for determinism.
# Centralized here to avoid duplication and ensure consistency.
TIMESTAMP_KEYS_TO_STRIP: FrozenSet[str] = frozenset({
    "timestamp",
    "created_at",
    "updated_at",
    "generated_at",
    "run_timestamp",
    "start_time",
    "end_time",
    "duration_ms",
    "elapsed_ms",
    "wall_clock_ms",
})


def strip_timestamp_keys(
    obj: Any,
    keys_to_strip: Optional[FrozenSet[str]] = None,
) -> Any:
    """Recursively strip timestamp and non-deterministic keys from dict/list.

    Args:
        obj: Object to strip keys from
        keys_to_strip: Set of key names to remove (default: TIMESTAMP_KEYS_TO_STRIP)

    Returns:
        Object with specified keys removed
    """
    if keys_to_strip is None:
        keys_to_strip = TIMESTAMP_KEYS_TO_STRIP

    if isinstance(obj, dict):
        return {
            k: strip_timestamp_keys(v, keys_to_strip)
            for k, v in obj.items()
            if k not in keys_to_strip
        }
    elif isinstance(obj, list):
        return [strip_timestamp_keys(item, keys_to_strip) for item in obj]
    else:
        return obj


class TestIdentitySignalDeterminism:
    """Test that generate_status produces identical output for identical inputs."""

    def test_identical_inputs_produce_identical_outputs(self, tmp_path: Path) -> None:
        """Running generate_status twice with identical inputs produces identical output."""
        from scripts.generate_first_light_status import generate_status

        # Create test fixtures
        manifest = {
            "governance": {
                "slice_identity": {
                    "p5_preflight_reference": {
                        "status": "OK",
                        "sha256": "abc123def456",
                        "fingerprint_match": True,
                    }
                },
                "policy_drift": {
                    "status": "OK",
                    "hash": "policy123",
                }
            },
            "signals": {
                "nci_p5": {
                    "slo_status": "OK",
                    "global_nci": 0.85,
                }
            }
        }

        evidence = {
            "governance": {
                "nci": {
                    "health_contribution": {
                        "status": "OK",
                        "global_nci": 0.85,
                    }
                }
            }
        }

        # Create results directory with run_config
        config_path = tmp_path / "run_config.json"
        config_path.write_text(json.dumps({
            "telemetry_adapter": "mock",
            "run_id": "test-run-123",
        }))

        # Run generate_status twice
        status_1 = generate_status(
            results_dir=tmp_path,
            manifest=manifest,
            evidence=evidence,
        )

        status_2 = generate_status(
            results_dir=tmp_path,
            manifest=manifest,
            evidence=evidence,
        )

        # Strip timestamps for comparison
        status_1_stripped = strip_timestamp_keys(status_1)
        status_2_stripped = strip_timestamp_keys(status_2)

        # Outputs must be identical
        assert status_1_stripped == status_2_stripped, (
            "generate_status produced different outputs for identical inputs"
        )

    def test_identity_signal_does_not_affect_other_signals(self, tmp_path: Path) -> None:
        """Identity preflight signal does not alter other signals in output."""
        from scripts.generate_first_light_status import generate_status

        # Base manifest without identity
        base_manifest = {
            "governance": {
                "policy_drift": {
                    "status": "OK",
                    "hash": "policy123",
                }
            },
            "signals": {
                "nci_p5": {
                    "slo_status": "OK",
                    "global_nci": 0.85,
                }
            }
        }

        evidence = {
            "governance": {
                "nci": {
                    "health_contribution": {
                        "status": "OK",
                        "global_nci": 0.85,
                    }
                }
            }
        }

        # Manifest with identity OK
        manifest_with_identity_ok = copy.deepcopy(base_manifest)
        manifest_with_identity_ok["governance"]["slice_identity"] = {
            "p5_preflight_reference": {
                "status": "OK",
                "sha256": "abc123",
            }
        }

        # Manifest with identity BLOCK
        manifest_with_identity_block = copy.deepcopy(base_manifest)
        manifest_with_identity_block["governance"]["slice_identity"] = {
            "p5_preflight_reference": {
                "status": "BLOCK",
                "sha256": "def456",
            }
        }

        # Generate status with different identity states
        status_no_identity = generate_status(
            manifest=base_manifest,
            evidence=evidence,
        )

        status_identity_ok = generate_status(
            manifest=manifest_with_identity_ok,
            evidence=evidence,
        )

        status_identity_block = generate_status(
            manifest=manifest_with_identity_block,
            evidence=evidence,
        )

        # Extract non-identity signals
        def get_non_identity_signals(status: Dict[str, Any]) -> Dict[str, Any]:
            signals = status.get("signals", {})
            return {k: v for k, v in signals.items() if k != "p5_identity_preflight"}

        non_identity_no_id = get_non_identity_signals(status_no_identity)
        non_identity_ok = get_non_identity_signals(status_identity_ok)
        non_identity_block = get_non_identity_signals(status_identity_block)

        # Non-identity signals must be identical regardless of identity state
        assert non_identity_no_id == non_identity_ok, (
            "Identity OK altered non-identity signals"
        )
        assert non_identity_no_id == non_identity_block, (
            "Identity BLOCK altered non-identity signals"
        )

        # Core status fields must be identical
        for field in ["schema_version", "telemetry_source", "shadow_mode_ok", "mode"]:
            assert status_no_identity.get(field) == status_identity_ok.get(field), (
                f"Identity OK altered {field}"
            )
            assert status_no_identity.get(field) == status_identity_block.get(field), (
                f"Identity BLOCK altered {field}"
            )


class TestIdentityWarningIsolation:
    """Test that identity warnings don't affect other warning ordering/counts."""

    def test_identity_warnings_appended_not_inserted(self, tmp_path: Path) -> None:
        """Identity warnings don't change ordering of other warnings."""
        from scripts.generate_first_light_status import generate_status

        # Create manifest that will generate schema warnings
        # (no p3/p4 dirs means schema validation will produce warnings)
        manifest_no_identity = {
            "governance": {
                "policy_drift": {"status": "OK"}
            },
            "signals": {
                "nci_p5": {"slo_status": "OK", "global_nci": 0.9}
            }
        }

        manifest_with_identity_block = copy.deepcopy(manifest_no_identity)
        manifest_with_identity_block["governance"]["slice_identity"] = {
            "p5_preflight_reference": {"status": "BLOCK"}
        }

        evidence = {
            "governance": {
                "nci": {"health_contribution": {"status": "OK"}}
            }
        }

        # Generate without identity
        status_no_id = generate_status(
            manifest=manifest_no_identity,
            evidence=evidence,
        )

        # Generate with identity BLOCK (should add warning)
        status_with_id = generate_status(
            manifest=manifest_with_identity_block,
            evidence=evidence,
        )

        warnings_no_id = status_no_id.get("warnings", [])
        warnings_with_id = status_with_id.get("warnings", [])

        # Filter out identity-specific warnings
        non_identity_warnings_no_id = [
            w for w in warnings_no_id if "identity" not in w.lower()
        ]
        non_identity_warnings_with_id = [
            w for w in warnings_with_id if "identity" not in w.lower()
        ]

        # Non-identity warnings must be identical in content and order
        assert non_identity_warnings_no_id == non_identity_warnings_with_id, (
            "Identity warning presence altered other warnings"
        )

        # Identity warnings should be present when status is BLOCK
        identity_warnings = [w for w in warnings_with_id if "identity" in w.lower()]
        assert len(identity_warnings) == 1, (
            f"Expected exactly 1 identity warning, got {len(identity_warnings)}"
        )
        assert "BLOCK" in identity_warnings[0], "Identity warning should mention BLOCK"
        assert "source=" in identity_warnings[0], "Identity warning should include source"

    def test_identity_ok_produces_no_warning(self, tmp_path: Path) -> None:
        """Identity status OK should not produce any identity warning."""
        from scripts.generate_first_light_status import generate_status

        manifest = {
            "governance": {
                "slice_identity": {
                    "p5_preflight_reference": {"status": "OK", "sha256": "abc"}
                }
            }
        }

        status = generate_status(manifest=manifest)

        warnings = status.get("warnings", [])
        identity_warnings = [w for w in warnings if "identity" in w.lower()]

        assert len(identity_warnings) == 0, (
            f"OK identity status should not produce warnings, got: {identity_warnings}"
        )

    def test_warning_counts_predictable(self, tmp_path: Path) -> None:
        """Warning counts should be predictable based on input state."""
        from scripts.generate_first_light_status import generate_status

        # Test with INVESTIGATE status
        manifest_investigate = {
            "governance": {
                "slice_identity": {
                    "p5_preflight_reference": {"status": "INVESTIGATE"}
                }
            }
        }

        # Test with BLOCK status
        manifest_block = {
            "governance": {
                "slice_identity": {
                    "p5_preflight_reference": {"status": "BLOCK"}
                }
            }
        }

        status_investigate = generate_status(manifest=manifest_investigate)
        status_block = generate_status(manifest=manifest_block)

        # Both INVESTIGATE and BLOCK should produce exactly 1 identity warning
        inv_warnings = [
            w for w in status_investigate.get("warnings", [])
            if "identity" in w.lower()
        ]
        block_warnings = [
            w for w in status_block.get("warnings", [])
            if "identity" in w.lower()
        ]

        assert len(inv_warnings) == 1, "INVESTIGATE should produce 1 identity warning"
        assert len(block_warnings) == 1, "BLOCK should produce 1 identity warning"
        assert "INVESTIGATE" in inv_warnings[0]
        assert "BLOCK" in block_warnings[0]


class TestIdentitySignalIsolationFromRunId:
    """Test that identity signal doesn't affect run_id or determinism metrics."""

    def test_identity_does_not_affect_telemetry_source(self, tmp_path: Path) -> None:
        """Identity preflight state doesn't alter telemetry_source detection."""
        from scripts.generate_first_light_status import generate_status

        # Create run_config with specific telemetry settings
        config_path = tmp_path / "run_config.json"
        config_path.write_text(json.dumps({
            "telemetry_source": "real_synthetic",
            "run_id": "test-run-456",
        }))

        # Test with different identity states
        cli_identity_ok = {"status": "OK"}
        cli_identity_block = {"status": "BLOCK"}

        status_ok = generate_status(
            results_dir=tmp_path,
            cli_identity=cli_identity_ok,
        )

        status_block = generate_status(
            results_dir=tmp_path,
            cli_identity=cli_identity_block,
        )

        # Telemetry source must be identical regardless of identity state
        assert status_ok["telemetry_source"] == status_block["telemetry_source"], (
            "Identity state altered telemetry_source"
        )
        assert status_ok["telemetry_source"] == "real_synthetic"

    def test_identity_does_not_affect_divergence_signals(self, tmp_path: Path) -> None:
        """Identity preflight state doesn't alter divergence baseline signals."""
        from scripts.generate_first_light_status import generate_status

        # Create p4_summary with divergence data
        p4_dir = tmp_path / "p4_run"
        p4_dir.mkdir()

        summary_path = p4_dir / "p4_summary.json"
        summary_path.write_text(json.dumps({
            "divergence_analysis": {
                "divergence_rate": 0.05,
                "max_divergence_streak": 2,
            },
            "twin_accuracy": {
                "success_prediction_accuracy": 0.95,
            }
        }))

        config_path = p4_dir / "run_config.json"
        config_path.write_text(json.dumps({
            "telemetry_source": "real_trace",
        }))

        # Test with different identity states via CLI
        status_no_id = generate_status(p4_dir=p4_dir)
        status_ok = generate_status(p4_dir=p4_dir, cli_identity={"status": "OK"})
        status_block = generate_status(p4_dir=p4_dir, cli_identity={"status": "BLOCK"})

        # Extract divergence signals
        div_no_id = status_no_id["signals"].get("p5_divergence_baseline")
        div_ok = status_ok["signals"].get("p5_divergence_baseline")
        div_block = status_block["signals"].get("p5_divergence_baseline")

        # Divergence signals must be identical
        assert div_no_id == div_ok, "Identity OK altered divergence baseline"
        assert div_no_id == div_block, "Identity BLOCK altered divergence baseline"

        # Verify divergence data is correct
        assert div_no_id is not None, "Divergence baseline should be present"
        assert div_no_id["divergence_rate"] == 0.05
        assert div_no_id["max_divergence_streak"] == 2


class TestCalExp2ArtifactLevelIsolation:
    """Artifact-level CAL-EXP-2 integration test for identity signal non-interference.

    Simulates a minimal CAL-EXP-2 output pack and proves that changing
    identity_preflight input (OK vs BLOCK vs missing) changes ONLY:
    - signals.p5_identity_preflight
    - at most 1 warning line (identity)

    It must NOT change:
    - any P4 divergence summary fields
    - telemetry_source
    - ordering of non-identity warnings

    SHADOW MODE CONTRACT:
    - All outputs observational only
    - No gating or enforcement
    - Deterministic ordering
    """

    def _create_cal_exp2_artifact_pack(self, base_dir: Path) -> Dict[str, Path]:
        """Create a minimal CAL-EXP-2 output pack structure.

        Returns dict with paths to key artifacts.
        """
        # Create directory structure
        p3_dir = base_dir / "p3_synthetic"
        p4_dir = base_dir / "p4_divergence"
        evidence_pack_dir = base_dir / "evidence_pack"

        p3_dir.mkdir(parents=True)
        p4_dir.mkdir(parents=True)
        evidence_pack_dir.mkdir(parents=True)

        # P3 synthetic results (minimal)
        p3_summary = {
            "schema_version": "1.0.0",
            "mode": "SHADOW",
            "total_cycles": 100,
            "noise_detected": 5,
            "noise_rate": 0.05,
        }
        (p3_dir / "p3_summary.json").write_text(json.dumps(p3_summary, indent=2))

        # P4 divergence summary (CAL-EXP-2 core artifact)
        p4_summary = {
            "schema_version": "1.0.0",
            "mode": "SHADOW",
            "divergence_analysis": {
                "divergence_rate": 0.03,
                "max_divergence_streak": 1,
                "total_divergences": 3,
                "total_comparisons": 100,
            },
            "twin_accuracy": {
                "success_prediction_accuracy": 0.97,
                "failure_prediction_accuracy": 0.92,
            },
            "calibration_window": {
                "start_cycle": 0,
                "end_cycle": 99,
                "window_size": 100,
            },
        }
        (p4_dir / "p4_summary.json").write_text(json.dumps(p4_summary, indent=2))

        # Run config (real telemetry for divergence baseline)
        run_config = {
            "telemetry_source": "real_trace",
            "telemetry_adapter": "real",
            "run_id": "cal-exp-2-test-run-001",
            "mode": "SHADOW",
        }
        (p4_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

        # Evidence pack manifest (base, no identity)
        manifest_base = {
            "schema_version": "1.0.0",
            "mode": "SHADOW",
            "governance": {
                "policy_drift": {
                    "status": "OK",
                    "hash": "policy_abc123",
                }
            },
            "signals": {
                "nci_p5": {
                    "slo_status": "OK",
                    "global_nci": 0.92,
                }
            },
            "artifacts": {
                "p3_summary": str(p3_dir / "p3_summary.json"),
                "p4_summary": str(p4_dir / "p4_summary.json"),
            },
        }
        (evidence_pack_dir / "manifest.json").write_text(
            json.dumps(manifest_base, indent=2)
        )

        # Evidence JSON
        evidence = {
            "governance": {
                "nci": {
                    "health_contribution": {
                        "status": "OK",
                        "global_nci": 0.92,
                    }
                }
            }
        }
        (evidence_pack_dir / "evidence.json").write_text(
            json.dumps(evidence, indent=2)
        )

        return {
            "p3_dir": p3_dir,
            "p4_dir": p4_dir,
            "evidence_pack_dir": evidence_pack_dir,
            "manifest_path": evidence_pack_dir / "manifest.json",
        }

    def _add_identity_to_manifest(
        self,
        manifest_path: Path,
        status: str,
    ) -> None:
        """Add identity preflight to manifest with given status."""
        manifest = json.loads(manifest_path.read_text())
        manifest["governance"]["slice_identity"] = {
            "p5_preflight_reference": {
                "status": status,
                "sha256": f"identity_hash_{status.lower()}",
                "fingerprint_match": status == "OK",
            }
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))

    def _remove_identity_from_manifest(self, manifest_path: Path) -> None:
        """Remove identity preflight from manifest."""
        manifest = json.loads(manifest_path.read_text())
        if "slice_identity" in manifest.get("governance", {}):
            del manifest["governance"]["slice_identity"]
        manifest_path.write_text(json.dumps(manifest, indent=2))

    def test_cal_exp2_artifact_identity_isolation(self, tmp_path: Path) -> None:
        """CAL-EXP-2 artifact-level test: identity changes only identity signal/warning.

        This test simulates a complete CAL-EXP-2 output pack and verifies that
        changing identity_preflight (OK vs BLOCK vs missing) affects ONLY:
        - signals.p5_identity_preflight
        - at most 1 identity warning

        It must NOT change:
        - P4 divergence summary fields (divergence_rate, max_divergence_streak, etc.)
        - telemetry_source
        - ordering of non-identity warnings
        - any other signals
        """
        from scripts.generate_first_light_status import generate_status

        # Create CAL-EXP-2 artifact pack
        artifacts = self._create_cal_exp2_artifact_pack(tmp_path)

        # ===================================================================
        # Generate status with NO identity (baseline)
        # ===================================================================
        self._remove_identity_from_manifest(artifacts["manifest_path"])
        status_no_identity = generate_status(
            p3_dir=artifacts["p3_dir"],
            p4_dir=artifacts["p4_dir"],
            evidence_pack_dir=artifacts["evidence_pack_dir"],
        )

        # ===================================================================
        # Generate status with identity OK
        # ===================================================================
        self._add_identity_to_manifest(artifacts["manifest_path"], "OK")
        status_identity_ok = generate_status(
            p3_dir=artifacts["p3_dir"],
            p4_dir=artifacts["p4_dir"],
            evidence_pack_dir=artifacts["evidence_pack_dir"],
        )

        # ===================================================================
        # Generate status with identity BLOCK
        # ===================================================================
        self._add_identity_to_manifest(artifacts["manifest_path"], "BLOCK")
        status_identity_block = generate_status(
            p3_dir=artifacts["p3_dir"],
            p4_dir=artifacts["p4_dir"],
            evidence_pack_dir=artifacts["evidence_pack_dir"],
        )

        # ===================================================================
        # VERIFICATION: P4 divergence summary fields unchanged
        # ===================================================================
        div_no_id = status_no_identity["signals"].get("p5_divergence_baseline")
        div_ok = status_identity_ok["signals"].get("p5_divergence_baseline")
        div_block = status_identity_block["signals"].get("p5_divergence_baseline")

        assert div_no_id is not None, "Divergence baseline missing in no-identity case"
        assert div_no_id == div_ok, (
            f"Identity OK altered P4 divergence: {div_no_id} vs {div_ok}"
        )
        assert div_no_id == div_block, (
            f"Identity BLOCK altered P4 divergence: {div_no_id} vs {div_block}"
        )

        # Verify specific divergence fields
        assert div_no_id["divergence_rate"] == 0.03
        assert div_no_id["max_divergence_streak"] == 1
        assert div_no_id["twin_success_accuracy"] == 0.97

        # ===================================================================
        # VERIFICATION: telemetry_source unchanged
        # ===================================================================
        assert status_no_identity["telemetry_source"] == "real_trace"
        assert status_identity_ok["telemetry_source"] == "real_trace"
        assert status_identity_block["telemetry_source"] == "real_trace"

        # ===================================================================
        # VERIFICATION: Non-identity warnings unchanged (ordering preserved)
        # ===================================================================
        def extract_non_identity_warnings(status: Dict[str, Any]) -> List[str]:
            return [w for w in status.get("warnings", []) if "identity" not in w.lower()]

        non_id_warnings_base = extract_non_identity_warnings(status_no_identity)
        non_id_warnings_ok = extract_non_identity_warnings(status_identity_ok)
        non_id_warnings_block = extract_non_identity_warnings(status_identity_block)

        assert non_id_warnings_base == non_id_warnings_ok, (
            "Identity OK altered non-identity warnings"
        )
        assert non_id_warnings_base == non_id_warnings_block, (
            "Identity BLOCK altered non-identity warnings"
        )

        # ===================================================================
        # VERIFICATION: Identity warnings follow single-cap rule
        # ===================================================================
        def extract_identity_warnings(status: Dict[str, Any]) -> List[str]:
            return [w for w in status.get("warnings", []) if "identity" in w.lower()]

        id_warnings_base = extract_identity_warnings(status_no_identity)
        id_warnings_ok = extract_identity_warnings(status_identity_ok)
        id_warnings_block = extract_identity_warnings(status_identity_block)

        # No identity -> no identity warning
        assert len(id_warnings_base) == 0, (
            f"Missing identity should produce 0 warnings, got: {id_warnings_base}"
        )

        # OK identity -> no identity warning
        assert len(id_warnings_ok) == 0, (
            f"OK identity should produce 0 warnings, got: {id_warnings_ok}"
        )

        # BLOCK identity -> exactly 1 identity warning
        assert len(id_warnings_block) == 1, (
            f"BLOCK identity should produce exactly 1 warning, got: {id_warnings_block}"
        )
        assert "BLOCK" in id_warnings_block[0]
        assert "source=" in id_warnings_block[0]
        assert "(advisory)" in id_warnings_block[0]

        # ===================================================================
        # VERIFICATION: Only identity signal differs
        # ===================================================================
        def extract_non_identity_signals(status: Dict[str, Any]) -> Dict[str, Any]:
            signals = status.get("signals", {})
            return {k: v for k, v in signals.items() if k != "p5_identity_preflight"}

        non_id_signals_base = extract_non_identity_signals(status_no_identity)
        non_id_signals_ok = extract_non_identity_signals(status_identity_ok)
        non_id_signals_block = extract_non_identity_signals(status_identity_block)

        assert non_id_signals_base == non_id_signals_ok, (
            "Identity OK altered non-identity signals"
        )
        assert non_id_signals_base == non_id_signals_block, (
            "Identity BLOCK altered non-identity signals"
        )

        # ===================================================================
        # VERIFICATION: Identity signal correctly reflects input
        # ===================================================================
        id_signal_base = status_no_identity["signals"].get("p5_identity_preflight", {})
        id_signal_ok = status_identity_ok["signals"].get("p5_identity_preflight", {})
        id_signal_block = status_identity_block["signals"].get("p5_identity_preflight", {})

        # Missing identity -> extraction_source=MISSING, status=OK (default)
        assert id_signal_base.get("extraction_source") == "MISSING"
        assert id_signal_base.get("status") == "OK"

        # OK identity -> extraction_source=MANIFEST, status=OK
        assert id_signal_ok.get("extraction_source") == "MANIFEST"
        assert id_signal_ok.get("status") == "OK"

        # BLOCK identity -> extraction_source=MANIFEST, status=BLOCK
        assert id_signal_block.get("extraction_source") == "MANIFEST"
        assert id_signal_block.get("status") == "BLOCK"

        # ===================================================================
        # VERIFICATION: Core status fields unchanged
        # ===================================================================
        for field in ["schema_version", "mode", "shadow_mode_ok"]:
            assert status_no_identity.get(field) == status_identity_ok.get(field), (
                f"Identity OK altered {field}"
            )
            assert status_no_identity.get(field) == status_identity_block.get(field), (
                f"Identity BLOCK altered {field}"
            )


class TestCalExp2VerifierNonInterference:
    """Test that CAL-EXP-2 verifier PASS/FAIL is invariant to identity content.

    The verifier (scripts/verify_cal_exp_2_run.py) operates on run artifacts
    (run_config.json, RUN_METADATA.json, JSONL files) which do not contain
    identity preflight data. This test explicitly confirms the non-interference.

    SHADOW MODE CONTRACT:
    - Verifier decision must not depend on identity content
    - Identity is purely observational (advisory only)
    """

    def _create_valid_cal_exp2_run_dir(self, base_dir: Path) -> Path:
        """Create a valid CAL-EXP-2 run directory that passes verification."""
        run_dir = base_dir / "cal_exp_2_run"
        run_dir.mkdir(parents=True)

        # run_config.json (valid SHADOW mode)
        run_config = {
            "schema_version": "1.0.0",
            "mode": "SHADOW",
            "seed": 42,
            "twin_lr_overrides": {
                "H": 0.20,
                "rho": 0.15,
                "tau": 0.02,
                "beta": 0.12,
            },
            "parameters": {
                "seed": 42,
            }
        }
        (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

        # RUN_METADATA.json (valid non-blocking status)
        run_metadata = {
            "enforcement": False,
            "status": "completed",
            "cycles_completed": 100,
            "total_cycles_requested": 100,
        }
        (run_dir / "RUN_METADATA.json").write_text(json.dumps(run_metadata, indent=2))

        # divergence_log.jsonl (all LOGGED_ONLY)
        divergence_entries = [
            {"cycle": 1, "diverged": False, "action": "LOGGED_ONLY"},
            {"cycle": 5, "diverged": True, "action": "LOGGED_ONLY"},
            {"cycle": 10, "diverged": False, "action": "LOGGED_ONLY"},
        ]
        with open(run_dir / "divergence_log.jsonl", "w") as f:
            for entry in divergence_entries:
                f.write(json.dumps(entry) + "\n")

        return run_dir

    def _add_identity_to_run_dir(
        self,
        run_dir: Path,
        identity_status: str,
    ) -> None:
        """Add identity preflight file to run directory."""
        identity_data = {
            "schema_version": "1.0.0",
            "status": identity_status,
            "fingerprint_match": identity_status == "OK",
            "sha256": f"identity_hash_{identity_status.lower()}",
            "mode": "SHADOW",
        }
        (run_dir / "p5_identity_preflight.json").write_text(
            json.dumps(identity_data, indent=2)
        )

    def _remove_identity_from_run_dir(self, run_dir: Path) -> None:
        """Remove identity preflight file from run directory."""
        identity_path = run_dir / "p5_identity_preflight.json"
        if identity_path.exists():
            identity_path.unlink()

    def test_verifier_pass_fail_invariant_to_identity(self, tmp_path: Path) -> None:
        """Verifier PASS/FAIL decision is invariant to identity content.

        Tests that adding/changing/removing identity preflight files does not
        affect the verifier's decision on a valid CAL-EXP-2 run.
        """
        from scripts.verify_cal_exp_2_run import verify_run

        # Create valid run directory
        run_dir = self._create_valid_cal_exp2_run_dir(tmp_path)

        # ===================================================================
        # Verify with NO identity file
        # ===================================================================
        self._remove_identity_from_run_dir(run_dir)
        report_no_identity = verify_run(run_dir)

        # ===================================================================
        # Verify with identity OK
        # ===================================================================
        self._add_identity_to_run_dir(run_dir, "OK")
        report_identity_ok = verify_run(run_dir)

        # ===================================================================
        # Verify with identity BLOCK
        # ===================================================================
        self._add_identity_to_run_dir(run_dir, "BLOCK")
        report_identity_block = verify_run(run_dir)

        # ===================================================================
        # Verify with identity INVESTIGATE
        # ===================================================================
        self._add_identity_to_run_dir(run_dir, "INVESTIGATE")
        report_identity_investigate = verify_run(run_dir)

        # ===================================================================
        # VERIFICATION: PASS/FAIL verdict is identical
        # ===================================================================
        assert report_no_identity.passed == report_identity_ok.passed, (
            "Identity OK changed verifier PASS/FAIL verdict"
        )
        assert report_no_identity.passed == report_identity_block.passed, (
            "Identity BLOCK changed verifier PASS/FAIL verdict"
        )
        assert report_no_identity.passed == report_identity_investigate.passed, (
            "Identity INVESTIGATE changed verifier PASS/FAIL verdict"
        )

        # All should PASS (valid run)
        assert report_no_identity.passed is True, "Valid run should PASS"

        # ===================================================================
        # VERIFICATION: Check counts are identical
        # ===================================================================
        assert len(report_no_identity.checks) == len(report_identity_ok.checks), (
            "Identity OK changed check count"
        )
        assert len(report_no_identity.checks) == len(report_identity_block.checks), (
            "Identity BLOCK changed check count"
        )
        assert len(report_no_identity.checks) == len(report_identity_investigate.checks), (
            "Identity INVESTIGATE changed check count"
        )

        # ===================================================================
        # VERIFICATION: Individual check results are identical
        # ===================================================================
        for i, check_no_id in enumerate(report_no_identity.checks):
            check_ok = report_identity_ok.checks[i]
            check_block = report_identity_block.checks[i]
            check_investigate = report_identity_investigate.checks[i]

            # Name should match
            assert check_no_id.name == check_ok.name == check_block.name == check_investigate.name

            # Pass/fail status should match
            assert check_no_id.passed == check_ok.passed, (
                f"Identity OK changed check '{check_no_id.name}' result"
            )
            assert check_no_id.passed == check_block.passed, (
                f"Identity BLOCK changed check '{check_no_id.name}' result"
            )
            assert check_no_id.passed == check_investigate.passed, (
                f"Identity INVESTIGATE changed check '{check_no_id.name}' result"
            )

    def test_verifier_does_not_read_identity_files(self, tmp_path: Path) -> None:
        """Verifier ignores identity files even when present.

        Explicitly tests that even corrupted identity files don't affect
        verifier results, proving complete isolation.
        """
        from scripts.verify_cal_exp_2_run import verify_run

        # Create valid run directory
        run_dir = self._create_valid_cal_exp2_run_dir(tmp_path)

        # ===================================================================
        # Add corrupted/invalid identity file
        # ===================================================================
        (run_dir / "p5_identity_preflight.json").write_text("INVALID JSON {{{")

        # ===================================================================
        # Verify - should still PASS (verifier ignores identity files)
        # ===================================================================
        report = verify_run(run_dir)

        assert report.passed is True, (
            "Verifier should ignore corrupted identity file"
        )

        # No checks should reference identity
        for check in report.checks:
            assert "identity" not in check.name.lower(), (
                f"Verifier should not have identity checks: {check.name}"
            )
            assert "preflight" not in check.name.lower(), (
                f"Verifier should not have preflight checks: {check.name}"
            )

    def test_verifier_fail_is_also_invariant_to_identity(self, tmp_path: Path) -> None:
        """Verifier FAIL verdict is also invariant to identity content.

        Tests that a failing run (invalid mode) fails regardless of identity.
        """
        from scripts.verify_cal_exp_2_run import verify_run

        # Create run directory with INVALID mode (will FAIL)
        run_dir = tmp_path / "failing_run"
        run_dir.mkdir(parents=True)

        run_config = {
            "schema_version": "1.0.0",
            "mode": "PRODUCTION",  # Invalid - should be SHADOW
        }
        (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

        run_metadata = {
            "enforcement": False,
            "status": "completed",
            "cycles_completed": 100,
            "total_cycles_requested": 100,
        }
        (run_dir / "RUN_METADATA.json").write_text(json.dumps(run_metadata, indent=2))

        # ===================================================================
        # Test FAIL is consistent across identity states
        # ===================================================================
        self._remove_identity_from_run_dir(run_dir)
        report_no_identity = verify_run(run_dir)

        self._add_identity_to_run_dir(run_dir, "OK")
        report_identity_ok = verify_run(run_dir)

        self._add_identity_to_run_dir(run_dir, "BLOCK")
        report_identity_block = verify_run(run_dir)

        # All should FAIL (invalid mode)
        assert report_no_identity.passed is False, "Invalid mode should FAIL"
        assert report_identity_ok.passed is False, "FAIL should persist with identity OK"
        assert report_identity_block.passed is False, "FAIL should persist with identity BLOCK"

        # Fail count should be identical
        assert report_no_identity.fail_count == report_identity_ok.fail_count
        assert report_no_identity.fail_count == report_identity_block.fail_count


class TestJsonSerializationOrderingStability:
    """Test that JSON serialization ordering remains stable when identity changes.

    This test ensures that toggling identity_preflight changes ONLY:
    - signals.p5_identity_preflight
    - at most one warning line

    And that JSON serialization ordering of all other signals remains
    byte-for-byte identical after timestamp stripping.

    SHADOW MODE CONTRACT:
    - Identity is purely observational
    - No side effects on other signals or their serialization order
    """

    def _strip_identity_from_status(self, status: Dict[str, Any]) -> Dict[str, Any]:
        """Remove identity-related fields for comparison."""
        result = strip_timestamp_keys(status)

        # Remove identity signal
        if "signals" in result and "p5_identity_preflight" in result["signals"]:
            result = copy.deepcopy(result)
            del result["signals"]["p5_identity_preflight"]

        # Remove identity warnings
        if "warnings" in result:
            result = copy.deepcopy(result) if result is status else result
            result["warnings"] = [
                w for w in result["warnings"]
                if "identity" not in w.lower()
            ]

        return result

    def test_json_serialization_ordering_stable_across_identity_states(
        self,
        tmp_path: Path,
    ) -> None:
        """JSON serialization of non-identity fields is byte-identical across identity states.

        Uses json.dumps with sort_keys=True to ensure deterministic ordering,
        then compares the serialized strings after stripping identity fields.
        """
        from scripts.generate_first_light_status import generate_status

        # Create rich manifest with multiple signals
        base_manifest = {
            "governance": {
                "policy_drift": {
                    "status": "WARN",
                    "hash": "policy_xyz789",
                    "drift_detected": True,
                }
            },
            "signals": {
                "nci_p5": {
                    "slo_status": "WARN",
                    "global_nci": 0.72,
                }
            }
        }

        evidence = {
            "governance": {
                "nci": {
                    "health_contribution": {
                        "status": "WARN",
                        "global_nci": 0.72,
                    }
                }
            }
        }

        # Create results dir with run config
        config_path = tmp_path / "run_config.json"
        config_path.write_text(json.dumps({
            "telemetry_source": "real_synthetic",
            "run_id": "json-order-test-001",
        }))

        # Generate status with NO identity
        manifest_no_id = copy.deepcopy(base_manifest)
        status_no_id = generate_status(
            results_dir=tmp_path,
            manifest=manifest_no_id,
            evidence=evidence,
        )

        # Generate status with identity OK
        manifest_ok = copy.deepcopy(base_manifest)
        manifest_ok["governance"]["slice_identity"] = {
            "p5_preflight_reference": {"status": "OK", "sha256": "abc"}
        }
        status_ok = generate_status(
            results_dir=tmp_path,
            manifest=manifest_ok,
            evidence=evidence,
        )

        # Generate status with identity BLOCK
        manifest_block = copy.deepcopy(base_manifest)
        manifest_block["governance"]["slice_identity"] = {
            "p5_preflight_reference": {"status": "BLOCK", "sha256": "def"}
        }
        status_block = generate_status(
            results_dir=tmp_path,
            manifest=manifest_block,
            evidence=evidence,
        )

        # Generate status with identity INVESTIGATE
        manifest_investigate = copy.deepcopy(base_manifest)
        manifest_investigate["governance"]["slice_identity"] = {
            "p5_preflight_reference": {"status": "INVESTIGATE", "sha256": "ghi"}
        }
        status_investigate = generate_status(
            results_dir=tmp_path,
            manifest=manifest_investigate,
            evidence=evidence,
        )

        # Strip identity fields from all statuses
        stripped_no_id = self._strip_identity_from_status(status_no_id)
        stripped_ok = self._strip_identity_from_status(status_ok)
        stripped_block = self._strip_identity_from_status(status_block)
        stripped_investigate = self._strip_identity_from_status(status_investigate)

        # Serialize to JSON with deterministic ordering
        json_no_id = json.dumps(stripped_no_id, sort_keys=True, indent=2)
        json_ok = json.dumps(stripped_ok, sort_keys=True, indent=2)
        json_block = json.dumps(stripped_block, sort_keys=True, indent=2)
        json_investigate = json.dumps(stripped_investigate, sort_keys=True, indent=2)

        # All serializations must be byte-identical
        assert json_no_id == json_ok, (
            "Identity OK altered JSON serialization of non-identity fields"
        )
        assert json_no_id == json_block, (
            "Identity BLOCK altered JSON serialization of non-identity fields"
        )
        assert json_no_id == json_investigate, (
            "Identity INVESTIGATE altered JSON serialization of non-identity fields"
        )

    def test_identity_changes_only_expected_fields(self, tmp_path: Path) -> None:
        """Exhaustively verify that identity changes affect only identity-related fields.

        Compares full status dicts field-by-field to ensure no unexpected changes.
        """
        from scripts.generate_first_light_status import generate_status

        manifest_no_id = {
            "governance": {
                "policy_drift": {"status": "OK", "hash": "p123"}
            },
            "signals": {
                "nci_p5": {"slo_status": "OK", "global_nci": 0.9}
            }
        }

        manifest_block = copy.deepcopy(manifest_no_id)
        manifest_block["governance"]["slice_identity"] = {
            "p5_preflight_reference": {"status": "BLOCK"}
        }

        evidence = {
            "governance": {
                "nci": {"health_contribution": {"status": "OK"}}
            }
        }

        status_no_id = generate_status(manifest=manifest_no_id, evidence=evidence)
        status_block = generate_status(manifest=manifest_block, evidence=evidence)

        # Strip timestamps for comparison
        status_no_id = strip_timestamp_keys(status_no_id)
        status_block = strip_timestamp_keys(status_block)

        # ===================================================================
        # Compare all top-level fields except warnings
        # ===================================================================
        for key in set(status_no_id.keys()) | set(status_block.keys()):
            if key == "warnings":
                continue  # Handled separately
            if key == "signals":
                continue  # Handled separately

            assert status_no_id.get(key) == status_block.get(key), (
                f"Top-level field '{key}' changed unexpectedly"
            )

        # ===================================================================
        # Compare signals (excluding p5_identity_preflight)
        # ===================================================================
        signals_no_id = {
            k: v for k, v in status_no_id.get("signals", {}).items()
            if k != "p5_identity_preflight"
        }
        signals_block = {
            k: v for k, v in status_block.get("signals", {}).items()
            if k != "p5_identity_preflight"
        }

        assert signals_no_id == signals_block, (
            "Non-identity signals changed unexpectedly"
        )

        # ===================================================================
        # Compare warnings (excluding identity warnings)
        # ===================================================================
        warnings_no_id = [
            w for w in status_no_id.get("warnings", [])
            if "identity" not in w.lower()
        ]
        warnings_block = [
            w for w in status_block.get("warnings", [])
            if "identity" not in w.lower()
        ]

        assert warnings_no_id == warnings_block, (
            "Non-identity warnings changed unexpectedly"
        )

        # ===================================================================
        # Verify identity warning count is at most 1
        # ===================================================================
        identity_warnings = [
            w for w in status_block.get("warnings", [])
            if "identity" in w.lower()
        ]
        assert len(identity_warnings) <= 1, (
            f"Expected at most 1 identity warning, got {len(identity_warnings)}"
        )

        # ===================================================================
        # Verify identity signal is present and correct
        # ===================================================================
        id_signal = status_block.get("signals", {}).get("p5_identity_preflight", {})
        assert id_signal.get("status") == "BLOCK"
        assert id_signal.get("extraction_source") == "MANIFEST"

    def test_warning_ordering_deterministic(self, tmp_path: Path) -> None:
        """Verify warning list ordering is deterministic across runs.

        Runs generate_status multiple times and confirms warnings appear
        in identical order (after stripping identity warnings).
        """
        from scripts.generate_first_light_status import generate_status

        manifest = {
            "governance": {
                "policy_drift": {"status": "WARN"},
                "slice_identity": {
                    "p5_preflight_reference": {"status": "BLOCK"}
                }
            },
            "signals": {
                "nci_p5": {"slo_status": "WARN", "global_nci": 0.65}
            }
        }

        evidence = {
            "governance": {
                "nci": {"health_contribution": {"status": "WARN"}}
            }
        }

        # Run multiple times
        statuses = [
            generate_status(manifest=manifest, evidence=evidence)
            for _ in range(5)
        ]

        # Extract non-identity warnings from each run
        all_warnings = [
            [w for w in s.get("warnings", []) if "identity" not in w.lower()]
            for s in statuses
        ]

        # All runs must produce identical warning lists
        for i, warnings in enumerate(all_warnings[1:], start=2):
            assert warnings == all_warnings[0], (
                f"Warning ordering changed between run 1 and run {i}"
            )
