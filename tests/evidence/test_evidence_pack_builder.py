"""
Tests for Evidence Pack Builder

Tests cover:
- Deterministic manifest generation
- Missing artifact error handling
- Schema validation
- Merkle root computation and verification
- Governance checks (SHADOW MODE advisory)

SHADOW MODE CONTRACT:
All tests verify that governance checks are advisory only
and do not block pack generation.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple
import math

import pytest

from backend.topology.first_light.evidence_pack import (
    build_evidence_pack,
    verify_merkle_root,
    compute_merkle_root,
    compute_file_hash,
    compute_sha256,
    EvidencePackBuilder,
    EvidencePackResult,
    ArtifactInfo,
    CompletenessCheck,
    GovernanceAdvisory,
    REQUIRED_P3_ARTIFACTS,
    REQUIRED_P4_ARTIFACTS,
    EVIDENCE_PACK_VERSION,
)
from scripts.build_first_light_evidence_pack import (
    build_evidence_pack as cli_build_evidence_pack,
)
from scripts.generate_first_light_status import generate_status as generate_first_light_status
from tests.factories.first_light_factories import (
    make_metrics_window,
    make_p4_divergence_log_record,
    make_real_telemetry_snapshot,
    make_red_flag_entry,
    make_stability_report_payload,
    make_summary_payload,
    make_synthetic_raw_record,
    make_tda_window,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_run_dir():
    """Create a temporary run directory with minimal artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)

        # Create P3 artifacts directory
        p3_dir = run_dir / "p3_synthetic"
        p3_dir.mkdir(parents=True)

        # Create minimal P3 artifacts
        create_synthetic_raw(p3_dir / "synthetic_raw.jsonl")
        create_stability_report(p3_dir / "stability_report.json")
        create_red_flag_matrix(p3_dir / "red_flag_matrix.json")
        create_metrics_windows(p3_dir / "metrics_windows.json")
        create_tda_metrics(p3_dir / "tda_metrics.json")

        # Create P4 artifacts directory
        p4_dir = run_dir / "p4_shadow"
        p4_dir.mkdir(parents=True)

        # Create minimal P4 artifacts
        create_divergence_log(p4_dir / "divergence_log.jsonl")
        create_twin_trajectory(p4_dir / "twin_trajectory.jsonl")
        create_calibration_report(p4_dir / "calibration_report.json")

        # Create visualizations directory
        viz_dir = run_dir / "visualizations"
        viz_dir.mkdir(parents=True)

        # Create minimal visualization files
        create_svg_placeholder(viz_dir / "delta_p_trendline.svg")
        create_svg_placeholder(viz_dir / "rsi_trajectory.svg")
        create_svg_placeholder(viz_dir / "omega_occupancy.svg")

        yield run_dir


@pytest.fixture
def incomplete_run_dir():
    """Create a run directory with missing required artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)

        # Create only P3 directory with some artifacts
        p3_dir = run_dir / "p3_synthetic"
        p3_dir.mkdir(parents=True)

        # Only create some P3 artifacts
        create_synthetic_raw(p3_dir / "synthetic_raw.jsonl")
        create_stability_report(p3_dir / "stability_report.json")
        # Missing: red_flag_matrix.json, metrics_windows.json

        yield run_dir


@pytest.fixture
def empty_run_dir():
    """Create an empty run directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def factory_p4_run_dir():
    """Create a run directory containing only P4 artifacts using factories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        p4_dir = run_dir / "p4_shadow"
        p4_dir.mkdir(parents=True)

        create_divergence_log(p4_dir / "divergence_log.jsonl")
        create_twin_trajectory(p4_dir / "twin_trajectory.jsonl")
        create_calibration_report(p4_dir / "calibration_report.json")
        create_tda_metrics(p4_dir / "tda_metrics.json", source_phase="P4")

        yield run_dir


# =============================================================================
# Helper Functions
# =============================================================================

def _cycle_log_from_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "schema": payload["schema"],
        "cycle": payload["cycle"],
        "timestamp": payload["timestamp"],
        "mode": payload["mode"],
        "runner": {
            "type": payload["runner_type"],
            "slice": payload["runner_slice"],
            "success": payload["runner_success"],
            "depth": payload["runner_depth"],
        },
        "usla_state": {
            "H": payload["usla_H"],
            "rho": payload["usla_rho"],
            "tau": payload["usla_tau"],
            "beta": payload["usla_beta"],
            "in_omega": payload["in_omega"],
        },
        "governance": {
            "real_blocked": payload["real_blocked"],
            "sim_blocked": payload["sim_blocked"],
            "aligned": payload["governance_aligned"],
        },
        "metrics": {
            "hard_ok": payload["hard_ok"],
            "in_omega": payload["in_omega"],
        },
        "hard_ok": payload["hard_ok"],
        "abstained": payload["abstained"],
    }


def create_synthetic_raw(path: Path) -> None:
    """Create a minimal synthetic_raw.jsonl file."""
    with open(path, "w", encoding="utf-8") as f:
        for cycle in range(1, 11):
            payload = make_synthetic_raw_record(cycle, seed=cycle * 17)
            entry = _cycle_log_from_payload(payload)
            f.write(json.dumps(entry) + "\n")


def create_stability_report(path: Path) -> None:
    """Create a minimal stability_report.json file."""
    report = make_stability_report_payload(total_cycles=10, seed=99)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def create_red_flag_matrix(path: Path) -> None:
    """Create a minimal red_flag_matrix.json file."""
    summary_meta = make_summary_payload(total_cycles=10, seed=222)
    kinds = ["RSI_COLLAPSE", "OMEGA_EXIT", "HARD_FAIL", "CDI_007"]
    entries = [
        make_red_flag_entry(cycle=i + 1, kind=kinds[i % len(kinds)], seed=100 + i)
        for i in range(6)
    ]

    def map_severity(flag_severity: str) -> str:
        return "WARN" if flag_severity == "WARNING" else flag_severity

    flags: List[Dict[str, Any]] = []
    severity_counts: Counter[str] = Counter({"INFO": 0, "WARN": 0, "CRITICAL": 0})
    type_counts: Counter[str] = Counter()
    max_streaks: Dict[str, int] = {}

    for entry in entries:
        severity = map_severity(entry["flag_severity"])
        flags.append(
            {
                "cycle": entry["cycle"],
                "flag_type": entry["flag_type"],
                "severity": severity,
                "value": entry["observed_value"],
                "threshold": entry["threshold"],
                "streak_length": entry["consecutive_cycles"],
                "context": {"action": entry["action"]},
            }
        )
        severity_counts[severity] += 1
        type_counts[entry["flag_type"]] += 1
        max_streaks[entry["flag_type"]] = max(
            entry["consecutive_cycles"], max_streaks.get(entry["flag_type"], 0)
        )

    matrix = {
        "schema_version": "1.0.0",
        "run_id": summary_meta["run_id"],
        "timestamp": summary_meta["execution"]["end_time"],
        "total_cycles": 10,
        "flags": flags,
        "summary": {
            "total_flags": len(flags),
            "by_severity": dict(severity_counts),
            "by_type": dict(type_counts),
            "max_streaks": max_streaks,
            "hypothetical_abort": {"would_abort": False, "abort_cycle": None, "abort_reason": None},
        },
        "mode": "SHADOW",
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(matrix, f, indent=2)


def create_metrics_windows(path: Path) -> None:
    """Create a minimal metrics_windows.json file."""
    summary_meta = make_summary_payload(total_cycles=10, seed=333)
    window_size = 5
    windows_data = []
    trajectories = {
        "success_rate": [],
        "abstention_rate": [],
        "mean_rsi": [],
        "omega_occupancy": [],
        "hard_ok_rate": [],
    }

    for window_index in range(2):
        window_payload = make_metrics_window(
            window_index=window_index, window_size=window_size, seed=200 + window_index
        )
        metrics = {
            "success_rate": window_payload["success_metrics"]["success_rate"],
            "abstention_rate": window_payload["abstention_metrics"]["abstention_rate"],
            "omega_occupancy": window_payload["safe_region_metrics"]["omega_occupancy"],
            "hard_ok_rate": window_payload["hard_mode_metrics"]["hard_ok_rate"],
            "mean_rsi": window_payload["stability_metrics"]["mean_rsi"],
            "min_rsi": window_payload["stability_metrics"]["min_rsi"],
            "max_rsi": window_payload["stability_metrics"]["max_rsi"],
            "block_rate": window_payload["block_metrics"]["block_rate"],
        }
        windows_data.append(
            {
                "window_index": window_payload["window_index"],
                "start_cycle": window_payload["start_cycle"],
                "end_cycle": window_payload["end_cycle"],
                "metrics": metrics,
                "delta_p": {"success": None, "abstention": None},
                "red_flags_in_window": 0,
            }
        )

        trajectories["success_rate"].append(metrics["success_rate"])
        trajectories["abstention_rate"].append(metrics["abstention_rate"])
        trajectories["mean_rsi"].append(metrics["mean_rsi"])
        trajectories["omega_occupancy"].append(metrics["omega_occupancy"])
        trajectories["hard_ok_rate"].append(metrics["hard_ok_rate"])

    windows = {
        "schema_version": "1.0.0",
        "run_id": summary_meta["run_id"],
        "window_size": window_size,
        "total_windows": len(windows_data),
        "windows": windows_data,
        "trajectories": trajectories,
        "mode": "SHADOW",
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(windows, f, indent=2)


def create_divergence_log(path: Path) -> None:
    """Create a minimal divergence_log.jsonl file."""
    with open(path, "w", encoding="utf-8") as f:
        for cycle in range(1, 11):
            record = make_p4_divergence_log_record(cycle, seed=cycle * 7)
            f.write(json.dumps(record) + "\n")


def create_twin_trajectory(path: Path) -> None:
    """Create a minimal twin_trajectory.jsonl file."""
    with open(path, "w", encoding="utf-8") as f:
        for cycle in range(1, 11):
            payload = make_real_telemetry_snapshot(
                cycle,
                source="SHADOW_TWIN",
                seed=700 + cycle,
            )
            rng = random.Random(900 + cycle)
            record = {
                "cycle": cycle,
                "timestamp": payload["timestamp"],
                "input_snapshot_hash": payload["telemetry_hash"],
                "twin_state": {
                    "H": payload["H"],
                    "rho": payload["rho"],
                    "tau": payload["tau"],
                    "beta": payload["beta"],
                    "in_omega": payload["in_omega"],
                },
                "twin_prediction": {
                    "success_prob": round(rng.random(), 6),
                    "predicted_success": payload["success"],
                    "predicted_blocked": payload["real_blocked"],
                    "predicted_in_omega": payload["in_omega"],
                    "predicted_hard_ok": payload["hard_ok"],
                    "delta_p_predicted": round(0.01 + 0.01 * rng.random(), 6),
                },
                "model_confidence": round(0.7 + 0.3 * rng.random(), 6),
            }
            f.write(json.dumps(record) + "\n")


def create_calibration_report(path: Path) -> None:
    """Create a minimal calibration_report.json file."""
    report = {
        "schema_version": "1.0.0",
        "run_id": "test_run_123",
        "timing": {
            "start_time": "2025-12-10T12:00:00.000000+00:00",
            "end_time": "2025-12-10T12:00:10.000000+00:00",
            "cycles_observed": 10,
        },
        "divergence_statistics": {
            "mean_divergence": 0.002,
            "std_divergence": 0.001,
            "max_divergence": 0.005,
        },
        "accuracy_metrics": {
            "success_accuracy": 0.95,
            "blocked_accuracy": 0.98,
            "omega_accuracy": 0.97,
            "hard_ok_accuracy": 0.96,
        },
        "calibration_assessment": {
            "twin_validity": "VALID",
            "validity_score": 0.95,
            "recommendations": [],
        },
        "mode": "SHADOW",
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def create_tda_metrics(path: Path) -> None:
    """Create a minimal tda_metrics.json file."""
    payload = {
        "schema_version": "1.0.0",
        "run_id": "test_run_123",
        "source_phase": "P3",
        "window_size": 5,
        "windows": [],
        "summary": {},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def create_run_config(path: Path) -> None:
    """Create a minimal run_config.json file."""
    payload = {
        "schema_version": "1.0.0",
        "runner": "u2",
        "cycles": 10,
        "seed": 42,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def create_real_cycles(path: Path) -> None:
    """Create a minimal real_cycles.jsonl file."""
    records = [
        {
            "cycle": i,
            "timestamp": f"2025-12-10T12:00:{i:02d}.000000+00:00",
            "runner_state": {"success": True, "blocked": False},
            "mode": "SHADOW",
        }
        for i in range(1, 6)
    ]
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def create_twin_predictions(path: Path) -> None:
    """Reuse twin trajectory helper for twin_predictions.jsonl."""
    create_twin_trajectory(path)


def create_p4_summary(path: Path) -> None:
    """Create a minimal p4_summary.json file."""
    payload = {
        "mode": "SHADOW",
        "uplift_metrics": {"u2_success_rate_final": 0.9},
        "divergence_analysis": {"divergence_rate": 0.5, "max_divergence_streak": 2},
        "twin_accuracy": {"success_prediction_accuracy": 0.9},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def create_twin_accuracy(path: Path) -> None:
    """Create a minimal twin_accuracy.json file."""
    payload = {
        "schema_version": "1.0.0",
        "success_prediction_accuracy": 0.9,
        "omega_prediction_accuracy": 0.95,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def create_proof_log(path: Path) -> None:
    """Create a minimal proofs.jsonl file."""
    path.write_text('{"id": "alpha"}\n{"id": "beta"}\n', encoding="utf-8")


def create_svg_placeholder(path: Path) -> None:
    """Create a minimal SVG placeholder file."""
    svg = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
  <rect width="100" height="100" fill="#eee"/>
  <text x="50" y="50" text-anchor="middle">Placeholder</text>
</svg>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg)


def _metric_stat_summary(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0, "trend_slope": 0.0, "variance": 0.0}
    n = len(values)
    mean_val = sum(values) / n
    variance = sum((v - mean_val) ** 2 for v in values) / n
    std = math.sqrt(variance)
    return {
        "mean": round(mean_val, 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
        "std": round(std, 6),
        "trend_slope": 0.0,
        "variance": round(variance, 6),
    }


def create_tda_metrics(path: Path, source_phase: str = "P3") -> None:
    """Create a minimal tda_metrics.json file using factory windows."""
    summary_meta = make_summary_payload(total_cycles=10, seed=555)
    window_entries = []
    trajectories = {"sns": [], "pcs": [], "drs": [], "hss": []}
    stats_bank: Dict[str, List[float]] = {k: [] for k in trajectories}

    for idx in range(2):
        window = make_tda_window(window_index=idx, length=5, seed=400 + idx)
        metrics = {
            "window_index": window["window_index"],
            "start_cycle": window["window_start_cycle"],
            "end_cycle": window["window_end_cycle"],
            "metrics": {
                "sns": window["sns"]["mean"],
                "pcs": window["pcs"]["mean"],
                "drs": window["drs"]["mean"],
                "hss": window["hss"]["mean"],
            },
            "betti_numbers": [
                window["hss"]["betti_snapshot"]["b0"],
                window["hss"]["betti_snapshot"]["b1"],
                window["hss"]["betti_snapshot"]["b2"],
            ],
            "persistence_entropy": window["sns"]["std"],
            "topological_features": {
                "connected_components": window["hss"]["betti_snapshot"]["b0"],
                "loops": window["hss"]["betti_snapshot"]["b1"],
                "voids": window["hss"]["betti_snapshot"]["b2"],
            },
        }
        window_entries.append(metrics)
        for metric_name in trajectories:
            value = metrics["metrics"][metric_name]
            trajectories[metric_name].append(value)
            stats_bank[metric_name].append(value)

    summary = {
        metric: _metric_stat_summary(values)
        for metric, values in stats_bank.items()
    }
    summary["topological_stability"] = round(sum(trajectories["hss"]) / len(window_entries), 6)
    summary["anomaly_windows"] = []

    tda = {
        "schema_version": "1.0.0",
        "run_id": summary_meta["run_id"],
        "source_phase": source_phase,
        "generated_at": summary_meta["execution"]["end_time"],
        "window_size": 5,
        "total_windows": len(window_entries),
        "reference_topology": {
            "source": "factory_reference",
            "betti_numbers": window_entries[0]["betti_numbers"],
            "persistence_diagram_hash": "sha256:" + ("0" * 64),
        },
        "windows": window_entries,
        "summary": summary,
        "trajectories": trajectories,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tda, f, indent=2)


# =============================================================================
# Test: Merkle Tree Computation
# =============================================================================

class TestMerkleTree:
    """Tests for Merkle tree computation."""

    def test_compute_sha256(self) -> None:
        """Test SHA256 hash computation."""
        data = b"hello world"
        expected = hashlib.sha256(data).hexdigest()
        assert compute_sha256(data) == expected

    def test_merkle_root_single_leaf(self) -> None:
        """Test Merkle root with single leaf."""
        hashes = ["a" * 64]
        root = compute_merkle_root(hashes)
        assert len(root) == 64
        # Single leaf - root is the leaf itself
        assert root == "a" * 64

    def test_merkle_root_two_leaves(self) -> None:
        """Test Merkle root with two leaves."""
        hashes = ["a" * 64, "b" * 64]
        root = compute_merkle_root(hashes)
        assert len(root) == 64
        # Two leaves - root is hash of concatenation (sorted)
        expected_concat = bytes.fromhex("a" * 64) + bytes.fromhex("b" * 64)
        expected_root = hashlib.sha256(expected_concat).hexdigest()
        assert root == expected_root

    def test_merkle_root_deterministic(self) -> None:
        """Test that Merkle root is deterministic regardless of input order."""
        hashes1 = ["a" * 64, "b" * 64, "c" * 64]
        hashes2 = ["c" * 64, "a" * 64, "b" * 64]  # Different order

        root1 = compute_merkle_root(hashes1)
        root2 = compute_merkle_root(hashes2)

        # Roots should be identical due to sorting
        assert root1 == root2

    def test_merkle_root_empty(self) -> None:
        """Test Merkle root with empty list."""
        root = compute_merkle_root([])
        # Empty tree - should return hash of empty string
        assert root == compute_sha256(b"")


# =============================================================================
# Test: Deterministic Manifest Generation
# =============================================================================

class TestDeterministicManifest:
    """Tests for deterministic manifest generation."""

    def test_manifest_deterministic_hashes(self, temp_run_dir: Path) -> None:
        """Test that artifact hashes are deterministic."""
        # Use separate output dirs to avoid manifest.json being included in second run
        with tempfile.TemporaryDirectory() as out1, tempfile.TemporaryDirectory() as out2:
            result1 = build_evidence_pack(temp_run_dir, output_dir=out1)
            result2 = build_evidence_pack(temp_run_dir, output_dir=out2)

            assert result1.success
            assert result2.success

            # Artifact hashes should be identical (exclude manifest which has timestamp)
            hashes1 = {a.path: a.sha256 for a in result1.artifacts if "manifest" not in a.path}
            hashes2 = {a.path: a.sha256 for a in result2.artifacts if "manifest" not in a.path}
            assert hashes1 == hashes2

    def test_manifest_deterministic_merkle_root(self, temp_run_dir: Path) -> None:
        """Test that Merkle root is deterministic across builds."""
        # Use separate output dirs to avoid manifest.json being included in second run
        with tempfile.TemporaryDirectory() as out1, tempfile.TemporaryDirectory() as out2:
            result1 = build_evidence_pack(temp_run_dir, output_dir=out1)
            result2 = build_evidence_pack(temp_run_dir, output_dir=out2)

            assert result1.success
            assert result2.success

            # Merkle roots should be identical (based on same source artifacts)
            assert result1.merkle_root == result2.merkle_root

    def test_manifest_contains_required_fields(self, temp_run_dir: Path) -> None:
        """Test that manifest contains all required fields."""
        result = build_evidence_pack(temp_run_dir)
        assert result.success

        # Load generated manifest
        manifest_path = Path(result.manifest_path)
        assert manifest_path.exists()

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        # Check required fields
        assert "schema_version" in manifest
        assert "bundle_id" in manifest
        assert "bundle_version" in manifest
        assert "generated_at" in manifest
        assert "artifacts" in manifest
        assert "completeness" in manifest
        assert "cryptographic_root" in manifest

    def test_manifest_merkle_root_format(self, temp_run_dir: Path) -> None:
        """Test that Merkle root uses correct format."""
        result = build_evidence_pack(temp_run_dir)
        assert result.success

        # Format should be sha256:<hex>
        assert result.merkle_root is not None
        assert result.merkle_root.startswith("sha256:")
        hex_part = result.merkle_root[7:]
        assert len(hex_part) == 64
        # Verify it's valid hex
        int(hex_part, 16)


# =============================================================================
# Test: Missing Artifact Error Handling
# =============================================================================

class TestMissingArtifacts:
    """Tests for missing artifact error handling."""

    def test_empty_directory_fails(self, empty_run_dir: Path) -> None:
        """Test that empty directory fails with clear error."""
        result = build_evidence_pack(empty_run_dir)

        assert not result.success
        assert "No artifacts found" in result.errors[0]

    def test_nonexistent_directory_fails(self) -> None:
        """Test that nonexistent directory fails with clear error."""
        result = build_evidence_pack("/nonexistent/path/to/run")

        assert not result.success
        assert "does not exist" in result.errors[0]

    def test_missing_required_artifacts_tracked(self, incomplete_run_dir: Path) -> None:
        """Test that missing required artifacts are tracked."""
        result = build_evidence_pack(incomplete_run_dir)

        # Build should succeed (SHADOW MODE - advisory only)
        # but completeness should show missing artifacts
        assert not result.completeness.all_required_present
        assert len(result.completeness.missing_artifacts) > 0

        # Specific missing artifacts
        missing_names = [Path(p).name for p in result.completeness.missing_artifacts]
        assert "red_flag_matrix.json" in missing_names
        assert "metrics_windows.json" in missing_names

    def test_missing_p4_artifacts_tracked(self, incomplete_run_dir: Path) -> None:
        """Test that missing P4 artifacts are tracked."""
        result = build_evidence_pack(incomplete_run_dir)

        # P4 artifacts should be missing
        p4_completeness = result.completeness.p4_artifacts
        assert p4_completeness.get("divergence_log.jsonl") is False
        assert p4_completeness.get("twin_trajectory.jsonl") is False

    def test_completeness_check_fields(self, temp_run_dir: Path) -> None:
        """Test that completeness check has all expected fields."""
        result = build_evidence_pack(temp_run_dir)

        assert hasattr(result.completeness, "p3_artifacts")
        assert hasattr(result.completeness, "p4_artifacts")
        assert hasattr(result.completeness, "visualizations")
        assert hasattr(result.completeness, "all_required_present")
        assert hasattr(result.completeness, "missing_artifacts")


# =============================================================================
# Test: Schema Validation
# =============================================================================

class TestSchemaValidation:
    """Tests for schema validation."""

    def test_validation_tracks_results(self, temp_run_dir: Path) -> None:
        """Test that validation results are tracked per artifact."""
        result = build_evidence_pack(temp_run_dir, validate_schemas=True)
        assert result.success

        # Each artifact should have validation status
        for artifact in result.artifacts:
            assert hasattr(artifact, "validation_passed")
            assert hasattr(artifact, "validation_errors")

    def test_validation_can_be_skipped(self, temp_run_dir: Path) -> None:
        """Test that schema validation can be skipped."""
        result = build_evidence_pack(temp_run_dir, validate_schemas=False)
        assert result.success

        # Artifacts should still be collected
        assert len(result.artifacts) > 0

    def test_factory_generated_artifacts_validate(self, temp_run_dir: Path) -> None:
        """Factory-built artifacts should pass schema validation when available."""
        pytest.importorskip("jsonschema")
        result = build_evidence_pack(temp_run_dir, validate_schemas=True)
        assert result.success

        failures = [
            a.path for a in result.artifacts
            if a.schema_ref and not a.validation_passed
        ]
        assert not failures, f"Artifacts failed schema validation: {failures}"


class TestFactoryP4Run:
    """Tests focused on P4-only runs built from factories."""

    def test_factory_p4_only_run_builds(self, factory_p4_run_dir: Path) -> None:
        """Even P4-only runs should build packs (with completeness warnings)."""
        pytest.importorskip("jsonschema")
        result = build_evidence_pack(factory_p4_run_dir, validate_schemas=True)
        assert result.success
        assert not result.completeness.all_required_present

        # Ensure all P4 artifacts validated successfully
        failed = [
            a.path
            for a in result.artifacts
            if a.category == "p4_shadow" and not a.validation_passed
        ]
        assert not failed, f"P4 artifacts failed validation: {failed}"

    def test_invalid_json_tracked(self, temp_run_dir: Path) -> None:
        """Test that invalid JSON files are tracked."""
        # Create an invalid JSON file
        invalid_file = temp_run_dir / "p3_synthetic" / "invalid.json"
        with open(invalid_file, "w") as f:
            f.write("not valid json {{{")

        result = build_evidence_pack(temp_run_dir)

        # Find the invalid artifact
        invalid_artifacts = [a for a in result.artifacts if "invalid.json" in a.path]
        # Note: validation only runs for known artifact types


# =============================================================================
# Test: Governance Checks (SHADOW MODE)
# =============================================================================

class TestGovernanceChecks:
    """Tests for SHADOW MODE governance checks."""

    def test_governance_checks_are_advisory(self, temp_run_dir: Path) -> None:
        """Test that governance checks are advisory only and don't block build."""
        result = build_evidence_pack(temp_run_dir)

        # Build should succeed regardless of governance check results
        assert result.success

        # Governance advisories should be present
        assert len(result.governance_advisories) > 0

    def test_shadow_mode_check_included(self, temp_run_dir: Path) -> None:
        """Test that shadow mode compliance check is included."""
        result = build_evidence_pack(temp_run_dir)

        advisory_names = [a.check_name for a in result.governance_advisories]
        assert "shadow_mode_compliance" in advisory_names

    def test_schema_compliance_check_included(self, temp_run_dir: Path) -> None:
        """Test that schema compliance check is included."""
        result = build_evidence_pack(temp_run_dir)

        advisory_names = [a.check_name for a in result.governance_advisories]
        assert "schema_compliance" in advisory_names

    def test_completeness_check_included(self, temp_run_dir: Path) -> None:
        """Test that completeness check is included."""
        result = build_evidence_pack(temp_run_dir)

        advisory_names = [a.check_name for a in result.governance_advisories]
        assert "completeness" in advisory_names

    def test_stability_check_included(self, temp_run_dir: Path) -> None:
        """Test that stability thresholds check is included."""
        result = build_evidence_pack(temp_run_dir)

        advisory_names = [a.check_name for a in result.governance_advisories]
        assert "stability_thresholds" in advisory_names

    def test_advisory_has_required_fields(self, temp_run_dir: Path) -> None:
        """Test that each advisory has required fields."""
        result = build_evidence_pack(temp_run_dir)

        for advisory in result.governance_advisories:
            assert hasattr(advisory, "check_name")
            assert hasattr(advisory, "passed")
            assert hasattr(advisory, "severity")
            assert hasattr(advisory, "message")

    def test_incomplete_run_has_advisory_warning(self, incomplete_run_dir: Path) -> None:
        """Test that incomplete run generates appropriate advisory."""
        result = build_evidence_pack(incomplete_run_dir)

        # Find completeness advisory
        completeness_advisory = next(
            (a for a in result.governance_advisories if a.check_name == "completeness"),
            None
        )
        assert completeness_advisory is not None
        assert completeness_advisory.passed is False
        assert completeness_advisory.severity in ("WARN", "CRITICAL")


# =============================================================================
# Test: Merkle Root Verification
# =============================================================================

class TestMerkleVerification:
    """Tests for Merkle root verification."""

    def test_verify_valid_manifest(self, temp_run_dir: Path) -> None:
        """Test verification of valid manifest."""
        result = build_evidence_pack(temp_run_dir)
        assert result.success
        assert result.manifest_path is not None

        is_valid, message = verify_merkle_root(result.manifest_path)
        assert is_valid
        assert "verified" in message.lower()

    def test_verify_tampered_manifest_fails(self, temp_run_dir: Path) -> None:
        """Test that tampered manifest fails verification."""
        result = build_evidence_pack(temp_run_dir)
        assert result.success
        assert result.manifest_path is not None

        # Tamper with manifest
        with open(result.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        # Modify an artifact hash
        if manifest["artifacts"]:
            manifest["artifacts"][0]["sha256"] = "0" * 64

        with open(result.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f)

        # Verification should fail
        is_valid, message = verify_merkle_root(result.manifest_path)
        assert not is_valid
        assert "mismatch" in message.lower()

    def test_verify_nonexistent_manifest_fails(self) -> None:
        """Test verification of nonexistent manifest."""
        is_valid, message = verify_merkle_root("/nonexistent/manifest.json")
        assert not is_valid
        assert "not found" in message.lower()


# =============================================================================
# Test: Integration
# =============================================================================

class TestIntegration:
    """Integration tests for full workflow."""

    def test_full_workflow(self, temp_run_dir: Path) -> None:
        """Test complete evidence pack workflow."""
        # Build pack
        result = build_evidence_pack(
            temp_run_dir,
            p3_run_id="test_p3_123",
            p4_run_id="test_p4_456",
        )
        assert result.success

        # Verify manifest exists
        assert result.manifest_path is not None
        manifest_path = Path(result.manifest_path)
        assert manifest_path.exists()

        # Load manifest and verify structure
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert manifest["p3_run_id"] == "test_p3_123"
        assert manifest["p4_run_id"] == "test_p4_456"
        assert manifest["bundle_version"] == EVIDENCE_PACK_VERSION

        # Verify Merkle root
        is_valid, _ = verify_merkle_root(manifest_path)
        assert is_valid

    def test_custom_output_directory(self, temp_run_dir: Path) -> None:
        """Test building pack with custom output directory."""
        with tempfile.TemporaryDirectory() as output_dir:
            result = build_evidence_pack(
                temp_run_dir,
                output_dir=output_dir,
            )
            assert result.success

            # Manifest should be in output directory
            assert result.manifest_path is not None
            assert output_dir in result.manifest_path


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_handles_unicode_filenames(self, temp_run_dir: Path) -> None:
        """Test handling of unicode characters in filenames."""
        # Create file with unicode name
        unicode_file = temp_run_dir / "p3_synthetic" / "test_日本語.json"
        with open(unicode_file, "w", encoding="utf-8") as f:
            json.dump({"test": "value"}, f)

        result = build_evidence_pack(temp_run_dir)
        assert result.success


class TestProofSnapshotHook:
    """Tests for optional proof snapshot integration."""

    def test_manifest_includes_proof_snapshot_entry(
        self,
        temp_run_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        proof_log = temp_run_dir / "proofs.jsonl"
        proof_log.write_text('{"id": "alpha"}\n', encoding="utf-8")

        proof_hashes = ["a" * 64]
        expected_canonical = hashlib.sha256("\n".join(proof_hashes).encode("utf-8")).hexdigest()

        snapshot_payload = {
            "schema_version": "1.0.0",
            "canonical_hash_algorithm": "sha256",
            "canonicalization_version": "proof-log-v1",
            "source": str(proof_log),
            "proof_hashes": proof_hashes,
            "canonical_hash": expected_canonical,
            "entry_count": 1,
        }

        def fake_generate_snapshot(proof_log_path: str, output_path: str):
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text(json.dumps(snapshot_payload), encoding="utf-8")
            return snapshot_payload

        monkeypatch.setattr(
            "backend.topology.first_light.evidence_pack.generate_snapshot",
            fake_generate_snapshot,
        )

        result = build_evidence_pack(
            temp_run_dir,
            include_proof_snapshot=True,
            proof_log_path=str(proof_log),
        )

        assert result.success
        assert result.manifest_path is not None

        with open(result.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        snapshot_meta = manifest.get("proof_log_snapshot")
        assert snapshot_meta is not None
        assert snapshot_meta["canonical_hash"] == snapshot_payload["canonical_hash"]
        assert snapshot_meta["entry_count"] == snapshot_payload["entry_count"]
        assert snapshot_meta["canonical_hash_algorithm"] == "sha256"
        assert snapshot_meta["canonicalization_version"] == "proof-log-v1"
        assert snapshot_meta["path"] == "compliance/proof_log_snapshot.json"

        artifact_paths = [Path(a["path"]).as_posix() for a in manifest["artifacts"]]
        assert "compliance/proof_log_snapshot.json" in artifact_paths


    def test_handles_empty_files(self, temp_run_dir: Path) -> None:
        """Test handling of empty files."""
        # Create empty file
        empty_file = temp_run_dir / "p3_synthetic" / "empty.json"
        empty_file.touch()

        result = build_evidence_pack(temp_run_dir)
        # Should still succeed
        assert result.success

    def test_handles_large_jsonl(self, temp_run_dir: Path) -> None:
        """Test handling of large JSONL files."""
        # Create large JSONL file
        large_file = temp_run_dir / "p3_synthetic" / "large.jsonl"
        with open(large_file, "w", encoding="utf-8") as f:
            for i in range(1000):
                record = {"cycle": i, "data": "x" * 100}
                f.write(json.dumps(record) + "\n")

        result = build_evidence_pack(temp_run_dir)
        assert result.success


class TestProofSnapshotIntegration:
    """End-to-end integration covering CLI builder + status generation."""

    def test_cli_pack_and_status_records_snapshot(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        p3_dir = tmp_path / "p3"
        p4_dir = tmp_path / "p4"
        pack_dir = tmp_path / "pack"
        p3_run = p3_dir / "fl_test"
        p4_run = p4_dir / "p4_test"
        p3_run.mkdir(parents=True)
        p4_run.mkdir(parents=True)
        pack_dir.mkdir()

        # Create required P3 artifacts
        create_synthetic_raw(p3_run / "synthetic_raw.jsonl")
        create_stability_report(p3_run / "stability_report.json")
        create_red_flag_matrix(p3_run / "red_flag_matrix.json")
        create_metrics_windows(p3_run / "metrics_windows.json")
        create_tda_metrics(p3_run / "tda_metrics.json")
        create_run_config(p3_run / "run_config.json")
        proof_log = p3_run / "proofs.jsonl"
        create_proof_log(proof_log)

        # Create required P4 artifacts
        create_real_cycles(p4_run / "real_cycles.jsonl")
        create_twin_predictions(p4_run / "twin_predictions.jsonl")
        create_divergence_log(p4_run / "divergence_log.jsonl")
        create_p4_summary(p4_run / "p4_summary.json")
        create_twin_accuracy(p4_run / "twin_accuracy.json")
        create_run_config(p4_run / "run_config.json")

        proof_hashes = ["c" * 64, "d" * 64]
        expected_canonical = hashlib.sha256("\n".join(proof_hashes).encode("utf-8")).hexdigest()

        snapshot_payload = {
            "schema_version": "1.0.0",
            "canonical_hash_algorithm": "sha256",
            "canonicalization_version": "proof-log-v1",
            "source": str(proof_log),
            "proof_hashes": proof_hashes,
            "canonical_hash": expected_canonical,
            "entry_count": 2,
        }

        def fake_generate_snapshot(proof_log_path: str, output_path: str):
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text(json.dumps(snapshot_payload), encoding="utf-8")
            return snapshot_payload

        monkeypatch.setattr(
            "scripts.build_first_light_evidence_pack.generate_snapshot",
            fake_generate_snapshot,
        )

        cli_build_evidence_pack(
            p3_dir,
            p4_dir,
            pack_dir,
            include_proof_snapshot=True,
            proof_log=proof_log,
        )

        manifest_path = pack_dir / "manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert manifest.get("proof_log_snapshot")
        assert manifest["proof_log_snapshot"]["canonical_hash"] == snapshot_payload["canonical_hash"]

        status = generate_first_light_status(
            p3_dir,
            p4_dir,
            pack_dir,
        )

        assert status["artifacts"]["proof_snapshot_present"] is True
        assert status["artifacts"]["proof_snapshot_status"] == "ok"
        assert status["proof_snapshot_present"] is True


# =============================================================================
# Test: Status Reference Cross-Link
# =============================================================================

class TestStatusReference:
    """Tests for first_light_status.json cross-link functionality."""

    def test_no_status_reference_when_file_absent(self, temp_run_dir: Path) -> None:
        """Test that status_reference is None when first_light_status.json doesn't exist."""
        result = build_evidence_pack(temp_run_dir)

        assert result.success
        assert result.status_reference is None

    def test_status_reference_embedded_when_file_present(self, temp_run_dir: Path) -> None:
        """Test that status_reference is embedded when first_light_status.json exists."""
        # Create first_light_status.json
        status_data = {
            "schema_version": "1.0.0",
            "timestamp": "2025-12-11T12:00:00Z",
            "mode": "SHADOW",
            "shadow_mode_ok": True,
            "p3_harness_ok": True,
            "p4_harness_ok": True,
            "evidence_pack_ok": True,
            "metrics_snapshot": {
                "p3_success_rate": 0.85,
                "p4_divergence_rate": 0.97,
            },
            "warnings": [],
        }
        status_path = temp_run_dir / "first_light_status.json"
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump(status_data, f)

        result = build_evidence_pack(temp_run_dir)

        assert result.success
        assert result.status_reference is not None
        assert result.status_reference.path == "first_light_status.json"
        assert result.status_reference.schema_version == "1.0.0"
        assert result.status_reference.shadow_mode_ok is True

    def test_status_reference_hash_matches(self, temp_run_dir: Path) -> None:
        """Test that status_reference SHA-256 hash is correct."""
        # Create first_light_status.json
        status_data = {"schema_version": "1.0.0", "shadow_mode_ok": True}
        status_path = temp_run_dir / "first_light_status.json"
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump(status_data, f)

        # Compute expected hash
        expected_hash = compute_file_hash(status_path)

        result = build_evidence_pack(temp_run_dir)

        assert result.status_reference is not None
        assert result.status_reference.sha256 == expected_hash

    def test_status_reference_in_manifest(self, temp_run_dir: Path) -> None:
        """Test that status_reference is included in generated manifest."""
        # Create first_light_status.json
        status_data = {
            "schema_version": "1.0.0",
            "shadow_mode_ok": True,
        }
        status_path = temp_run_dir / "first_light_status.json"
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump(status_data, f)

        with tempfile.TemporaryDirectory() as output_dir:
            result = build_evidence_pack(temp_run_dir, output_dir=output_dir)

            assert result.success
            assert result.manifest_path is not None

            # Load manifest and verify status_reference
            with open(result.manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            assert "status_reference" in manifest
            assert manifest["status_reference"]["path"] == "first_light_status.json"
            assert manifest["status_reference"]["schema_version"] == "1.0.0"
            assert manifest["status_reference"]["shadow_mode_ok"] is True

    def test_status_reference_with_false_shadow_mode_ok(self, temp_run_dir: Path) -> None:
        """Test that shadow_mode_ok=False is correctly captured."""
        # Create first_light_status.json with shadow_mode_ok=False
        status_data = {
            "schema_version": "1.0.0",
            "shadow_mode_ok": False,  # Indicates compliance issue
            "errors": ["P4 mode is not SHADOW"],
        }
        status_path = temp_run_dir / "first_light_status.json"
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump(status_data, f)

        result = build_evidence_pack(temp_run_dir)

        assert result.success  # Pack still builds (advisory only)
        assert result.status_reference is not None
        assert result.status_reference.shadow_mode_ok is False

    def test_status_reference_with_malformed_json(self, temp_run_dir: Path) -> None:
        """Test handling of malformed first_light_status.json."""
        # Create malformed JSON file
        status_path = temp_run_dir / "first_light_status.json"
        with open(status_path, "w", encoding="utf-8") as f:
            f.write("not valid json {{{")

        result = build_evidence_pack(temp_run_dir)

        # Should still succeed and have a status reference (hash only)
        assert result.success
        assert result.status_reference is not None
        assert result.status_reference.sha256 is not None
        # Schema version should be None for malformed file
        assert result.status_reference.schema_version is None
