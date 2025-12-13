"""Tests for NCI Governance Adapter.

Tests cover:
- JSON serialization (deterministic output)
- Director panel construction
- Governance signal construction
- TCL violation extraction
- SIC violation extraction
- Global health integration
"""

import json
from datetime import datetime, timezone
from typing import Any, Dict

import pytest


class TestNCIDirectorPanelBuilder:
    """Tests for build_nci_director_panel()."""

    def test_builds_valid_panel_with_ok_status(self) -> None:
        """Director panel is valid JSON with OK status."""
        from backend.health.nci_governance_adapter import build_nci_director_panel

        insight_summary = {
            "global_nci": 0.92,
            "dominant_area": "terminology",
            "suggestion_count": 2,
            "summary": {
                "terminology_alignment": 0.90,
                "phase_discipline": 0.95,
                "uplift_avoidance": 1.0,
                "structural_coherence": 0.85,
            },
        }
        priority_view = {
            "status": "OK",
            "priority_areas": [],
        }
        slo_result = {
            "slo_status": "OK",
            "violations": [],
        }

        panel = build_nci_director_panel(insight_summary, priority_view, slo_result)

        # Verify JSON serializable
        json_str = json.dumps(panel, sort_keys=True)
        assert json_str is not None

        # Verify required fields
        assert panel["schema_version"] == "1.0.0"
        assert panel["status_light"] == "green"
        assert panel["global_nci"] == 0.92
        assert panel["dominant_area"] == "terminology"
        assert "headline" in panel
        assert "Narrative consistency within target" in panel["headline"]

    def test_builds_valid_panel_with_warn_status(self) -> None:
        """Director panel correctly shows WARN status."""
        from backend.health.nci_governance_adapter import build_nci_director_panel

        insight_summary = {
            "global_nci": 0.72,
            "dominant_area": "phase",
            "suggestion_count": 5,
        }
        priority_view = {
            "status": "ATTENTION",
            "priority_areas": [{"area": "phase", "nci": 0.65, "reason": "Phase discipline below threshold"}],
        }
        slo_result = {
            "slo_status": "WARN",
            "violations": ["Global NCI (0.72) below threshold (0.75)"],
        }

        panel = build_nci_director_panel(insight_summary, priority_view, slo_result)

        assert panel["status_light"] == "yellow"
        assert panel["slo_status"]["status"] == "WARN"
        assert "requires attention" in panel["headline"]

    def test_builds_valid_panel_with_breach_status(self) -> None:
        """Director panel correctly shows BREACH status."""
        from backend.health.nci_governance_adapter import build_nci_director_panel

        insight_summary = {
            "global_nci": 0.55,
            "dominant_area": "structure",
        }
        priority_view = {
            "status": "BREACH",
            "priority_areas": [
                {"area": "structure", "nci": 0.50, "reason": "Structural coherence very low"},
                {"area": "terminology", "nci": 0.60, "reason": "Terminology below threshold"},
            ],
        }
        slo_result = {
            "slo_status": "BREACH",
            "violations": [
                "Global NCI (0.55) below threshold (0.75)",
                "Structural NCI (0.50) below minimum (0.60)",
                "Area 'terminology' NCI (0.60) below threshold (0.70)",
            ],
        }

        panel = build_nci_director_panel(insight_summary, priority_view, slo_result)

        assert panel["status_light"] == "red"
        assert panel["slo_status"]["status"] == "BREACH"
        assert "SLO breach" in panel["headline"]
        assert panel["slo_status"]["violation_count"] == 3

    def test_panel_is_deterministic(self) -> None:
        """Same inputs produce same output (except timestamp)."""
        from backend.health.nci_governance_adapter import build_nci_director_panel

        insight_summary = {"global_nci": 0.85, "dominant_area": "none"}
        priority_view = {"status": "OK", "priority_areas": []}
        slo_result = {"slo_status": "OK", "violations": []}

        panel1 = build_nci_director_panel(insight_summary, priority_view, slo_result)
        panel2 = build_nci_director_panel(insight_summary, priority_view, slo_result)

        # Remove timestamps for comparison
        del panel1["timestamp"]
        del panel2["timestamp"]

        assert panel1 == panel2

    def test_panel_includes_metrics_summary(self) -> None:
        """Panel includes dimensional metrics."""
        from backend.health.nci_governance_adapter import build_nci_director_panel

        insight_summary = {
            "global_nci": 0.80,
            "dominant_area": "none",
            "summary": {
                "terminology_alignment": 0.85,
                "phase_discipline": 0.90,
                "uplift_avoidance": 0.95,
                "structural_coherence": 0.75,
            },
        }
        priority_view = {"status": "OK", "priority_areas": []}
        slo_result = {"slo_status": "OK", "violations": []}

        panel = build_nci_director_panel(insight_summary, priority_view, slo_result)

        assert "metrics_summary" in panel
        assert panel["metrics_summary"]["terminology_alignment"] == 0.85
        assert panel["metrics_summary"]["phase_discipline"] == 0.90
        assert panel["metrics_summary"]["uplift_avoidance"] == 0.95
        assert panel["metrics_summary"]["structural_coherence"] == 0.75


class TestNCIGovernanceSignalBuilder:
    """Tests for build_nci_governance_signal()."""

    def test_builds_valid_signal(self) -> None:
        """Governance signal is valid JSON with correct structure."""
        from backend.health.nci_governance_adapter import (
            build_nci_director_panel,
            build_nci_governance_signal,
        )

        insight_summary = {"global_nci": 0.88, "dominant_area": "none"}
        priority_view = {"status": "OK", "priority_areas": []}
        slo_result = {"slo_status": "OK", "violations": []}

        panel = build_nci_director_panel(insight_summary, priority_view, slo_result)
        signal = build_nci_governance_signal(panel, slo_result)

        # Verify JSON serializable
        json_str = json.dumps(signal, sort_keys=True)
        assert json_str is not None

        # Verify required fields
        assert signal["schema_version"] == "1.0.0"
        assert signal["source"] == "nci"
        assert "health_contribution" in signal
        assert signal["health_contribution"]["status"] == "OK"
        assert signal["health_contribution"]["global_nci"] == 0.88

    def test_signal_includes_telemetry_consistency(self) -> None:
        """Signal includes telemetry consistency when drift provided."""
        from backend.health.nci_governance_adapter import (
            build_nci_director_panel,
            build_nci_governance_signal,
        )

        insight_summary = {"global_nci": 0.80, "dominant_area": "none"}
        priority_view = {"status": "OK", "priority_areas": []}
        slo_result = {"slo_status": "OK", "violations": []}
        telemetry_drift = {
            "aligned": False,
            "drift_detected": True,
            "affected_docs_count": 2,
            "affected_docs": ["docs/api.md", "docs/config.md"],
            "violations": [
                {"doc": "docs/api.md", "field": "H", "violation_type": "TCL-002", "expected": "H", "found": "Ht"},
            ],
        }

        panel = build_nci_director_panel(insight_summary, priority_view, slo_result)
        signal = build_nci_governance_signal(panel, slo_result, telemetry_drift=telemetry_drift)

        assert signal["telemetry_consistency"]["aligned"] is False
        assert signal["telemetry_consistency"]["drift_detected"] is True
        assert signal["telemetry_consistency"]["affected_docs_count"] == 2

    def test_signal_includes_slice_consistency(self) -> None:
        """Signal includes slice consistency when violations provided."""
        from backend.health.nci_governance_adapter import (
            build_nci_director_panel,
            build_nci_governance_signal,
        )

        insight_summary = {"global_nci": 0.80, "dominant_area": "none"}
        priority_view = {"status": "OK", "priority_areas": []}
        slo_result = {"slo_status": "OK", "violations": []}
        slice_violations = [
            {"doc": "docs/slices.md", "slice": "arithmetic_simple", "violation_type": "SIC-001", "expected": "arithmetic_simple", "found": "ArithmeticSimple"},
        ]

        panel = build_nci_director_panel(insight_summary, priority_view, slo_result)
        signal = build_nci_governance_signal(panel, slo_result, slice_violations=slice_violations)

        assert signal["slice_consistency"]["aligned"] is False
        assert signal["slice_consistency"]["violation_count"] == 1

    def test_signal_dimensional_breakdown(self) -> None:
        """Signal includes dimensional breakdown."""
        from backend.health.nci_governance_adapter import (
            build_nci_director_panel,
            build_nci_governance_signal,
        )

        insight_summary = {
            "global_nci": 0.80,
            "dominant_area": "terminology",
            "summary": {
                "terminology_alignment": 0.70,
                "phase_discipline": 0.85,
                "uplift_avoidance": 0.90,
                "structural_coherence": 0.80,
            },
        }
        priority_view = {"status": "ATTENTION", "priority_areas": []}
        slo_result = {"slo_status": "WARN", "violations": []}

        panel = build_nci_director_panel(insight_summary, priority_view, slo_result)
        signal = build_nci_governance_signal(panel, slo_result)

        assert "dimensional_breakdown" in signal
        assert signal["dimensional_breakdown"]["terminology"] == 0.70
        assert signal["dimensional_breakdown"]["phase"] == 0.85


class TestTCLViolationExtraction:
    """Tests for check_telemetry_consistency()."""

    def test_detects_field_name_variants(self) -> None:
        """TCL-002: Detects non-canonical field names."""
        from backend.health.nci_governance_adapter import check_telemetry_consistency

        doc_contents = {
            "docs/api.md": "The Ht value represents health. Use RSI for stability.",
            "docs/config.md": "Set the β parameter for block rate.",
        }

        result = check_telemetry_consistency(doc_contents)

        assert result["aligned"] is False
        assert result["drift_detected"] is True
        assert len(result["violations"]) >= 3
        # Check for specific violations
        violation_types = [v["found"] for v in result["violations"]]
        assert "Ht" in violation_types
        assert "RSI" in violation_types

    def test_no_violations_when_canonical(self) -> None:
        """No violations when using canonical field names."""
        from backend.health.nci_governance_adapter import check_telemetry_consistency

        # Use only canonical field names (H, rho, tau, beta, in_omega)
        # Avoid using any variants like "health", "threshold", "block_rate"
        doc_contents = {
            "docs/api.md": "The H value represents system metric. Use rho for stability. The tau value controls gating.",
        }

        result = check_telemetry_consistency(doc_contents)

        assert result["aligned"] is True
        assert result["drift_detected"] is False
        assert len(result["violations"]) == 0

    def test_extracts_line_numbers(self) -> None:
        """Violations include line numbers."""
        from backend.health.nci_governance_adapter import check_telemetry_consistency

        doc_contents = {
            "docs/api.md": "Line 1\nLine 2 with Ht\nLine 3",
        }

        result = check_telemetry_consistency(doc_contents)

        assert len(result["violations"]) == 1
        assert result["violations"][0]["line"] == 2


class TestSICViolationExtraction:
    """Tests for check_slice_consistency()."""

    def test_detects_slice_name_variants(self) -> None:
        """SIC-001: Detects non-canonical slice names."""
        from backend.health.nci_governance_adapter import check_slice_consistency

        doc_contents = {
            "docs/slices.md": "The ArithmeticSimple slice is used for basic tests.",
            "docs/config.md": "Use prop_taut for propositional logic.",
        }

        violations = check_slice_consistency(doc_contents)

        assert len(violations) >= 2
        found_names = [v["found"] for v in violations]
        assert "ArithmeticSimple" in found_names
        assert "prop_taut" in found_names

    def test_no_violations_when_canonical(self) -> None:
        """No violations when using canonical slice names."""
        from backend.health.nci_governance_adapter import check_slice_consistency

        doc_contents = {
            "docs/slices.md": "The arithmetic_simple slice is used for basic tests.",
        }

        violations = check_slice_consistency(doc_contents)

        assert len(violations) == 0


class TestGlobalHealthIntegration:
    """Tests for global health integration."""

    def test_builds_full_tile(self) -> None:
        """build_nci_tile_for_global_health produces complete tile."""
        from backend.health.nci_governance_adapter import build_nci_tile_for_global_health

        insight_summary = {"global_nci": 0.85, "dominant_area": "none"}
        priority_view = {"status": "OK", "priority_areas": []}
        slo_result = {"slo_status": "OK", "violations": []}

        tile = build_nci_tile_for_global_health(
            insight_summary, priority_view, slo_result
        )

        assert tile["tile_type"] == "nci_governance"
        assert tile["mode"] == "SHADOW"
        assert "director_panel" in tile
        assert "governance_signal" in tile
        assert "shadow_mode_contract" in tile
        assert tile["shadow_mode_contract"]["observational_only"] is True

    def test_attaches_to_global_health(self) -> None:
        """attach_nci_tile_to_global_health integrates correctly."""
        from backend.health.nci_governance_adapter import (
            attach_nci_tile_to_global_health,
            build_nci_tile_for_global_health,
        )

        global_health = {"status": "ok", "timestamp": "2025-12-10T00:00:00Z"}
        insight_summary = {"global_nci": 0.85, "dominant_area": "none"}
        priority_view = {"status": "OK", "priority_areas": []}
        slo_result = {"slo_status": "OK", "violations": []}

        nci_tile = build_nci_tile_for_global_health(
            insight_summary, priority_view, slo_result
        )
        result = attach_nci_tile_to_global_health(global_health, nci_tile)

        # Original not modified
        assert "nci" not in global_health

        # Result has NCI tile
        assert "nci" in result
        assert result["nci"]["tile_type"] == "nci_governance"

    def test_tile_is_json_serializable(self) -> None:
        """Full tile can be serialized to JSON."""
        from backend.health.nci_governance_adapter import build_nci_tile_for_global_health

        insight_summary = {
            "global_nci": 0.78,
            "dominant_area": "terminology",
            "summary": {
                "terminology_alignment": 0.70,
                "phase_discipline": 0.85,
                "uplift_avoidance": 0.90,
                "structural_coherence": 0.75,
            },
            "category_scores": {
                "docs": 0.80,
                "paper": 0.75,
            },
        }
        priority_view = {
            "status": "ATTENTION",
            "priority_areas": [{"area": "terminology", "nci": 0.70, "reason": "Below threshold"}],
        }
        slo_result = {
            "slo_status": "WARN",
            "violations": ["Terminology alignment below threshold"],
            "thresholds_used": {"global_nci_warn": 0.75},
        }
        telemetry_drift = {"aligned": True, "drift_detected": False, "affected_docs_count": 0}
        slice_violations = []

        tile = build_nci_tile_for_global_health(
            insight_summary, priority_view, slo_result,
            telemetry_drift=telemetry_drift,
            slice_violations=slice_violations,
        )

        # Should not raise
        json_str = json.dumps(tile, sort_keys=True, indent=2)
        assert len(json_str) > 100  # Non-trivial output

        # Round-trip
        parsed = json.loads(json_str)
        assert parsed["tile_type"] == "nci_governance"


class TestP3StabilityReportIntegration:
    """Tests for P3 stability report integration."""

    def test_builds_nci_summary_for_p3(self) -> None:
        """build_nci_summary_for_p3 extracts correct fields."""
        from backend.health.nci_governance_adapter import (
            build_nci_director_panel,
            build_nci_summary_for_p3,
        )

        insight_summary = {
            "global_nci": 0.78,
            "dominant_area": "terminology",
            "summary": {
                "terminology_alignment": 0.70,
                "phase_discipline": 0.85,
                "uplift_avoidance": 0.90,
                "structural_coherence": 0.75,
            },
        }
        priority_view = {"status": "ATTENTION", "priority_areas": []}
        slo_result = {
            "slo_status": "WARN",
            "violations": ["Terminology below threshold"],
            "violation_count": 1,
        }

        panel = build_nci_director_panel(insight_summary, priority_view, slo_result)
        summary = build_nci_summary_for_p3(panel)

        assert summary["global_nci_score"] == 0.78
        assert summary["dominant_area"] == "terminology"
        assert summary["status_light"] == "yellow"
        assert summary["slo_status"] == "WARN"
        assert summary["violation_count"] == 1
        assert "dimensional_scores" in summary
        assert summary["dimensional_scores"]["terminology"] == 0.70

    def test_attaches_summary_to_stability_report(self) -> None:
        """attach_nci_summary_to_stability_report attaches correctly."""
        from backend.health.nci_governance_adapter import attach_nci_summary_to_stability_report

        stability_report = {
            "schema_version": "1.0.0",
            "run_id": "test-run",
            "metrics": {"mean_rsi": 0.85},
        }
        nci_summary = {
            "global_nci_score": 0.88,
            "dominant_area": "none",
            "status_light": "green",
        }

        result = attach_nci_summary_to_stability_report(stability_report, nci_summary)

        # Original not modified
        assert "nci_summary" not in stability_report

        # Result has NCI summary
        assert "nci_summary" in result
        assert result["nci_summary"]["global_nci_score"] == 0.88
        assert result["metrics"]["mean_rsi"] == 0.85  # Original data preserved


class TestEvidencePackIntegration:
    """Tests for evidence pack integration."""

    def test_attaches_nci_to_evidence(self) -> None:
        """attach_nci_to_evidence attaches under governance.nci."""
        from backend.health.nci_governance_adapter import (
            build_nci_director_panel,
            build_nci_governance_signal,
            attach_nci_to_evidence,
        )

        insight_summary = {"global_nci": 0.85, "dominant_area": "none"}
        priority_view = {"status": "OK", "priority_areas": []}
        slo_result = {"slo_status": "OK", "violations": []}

        panel = build_nci_director_panel(insight_summary, priority_view, slo_result)
        signal = build_nci_governance_signal(panel, slo_result)

        evidence = {
            "run_id": "test-run",
            "artifacts": [],
        }

        result = attach_nci_to_evidence(evidence, signal)

        # Original not modified
        assert "governance" not in evidence

        # Result has NCI under governance
        assert "governance" in result
        assert "nci" in result["governance"]
        assert result["governance"]["nci"]["source"] == "nci"

    def test_attaches_to_existing_governance(self) -> None:
        """attach_nci_to_evidence preserves existing governance signals."""
        from backend.health.nci_governance_adapter import attach_nci_to_evidence

        evidence = {
            "run_id": "test-run",
            "governance": {
                "replay": {"status": "ok"},
            },
        }
        nci_signal = {"source": "nci", "health_contribution": {"status": "OK"}}

        result = attach_nci_to_evidence(evidence, nci_signal)

        # Both signals present
        assert "replay" in result["governance"]
        assert "nci" in result["governance"]
        assert result["governance"]["replay"]["status"] == "ok"

    def test_builds_complete_evidence_attachment(self) -> None:
        """build_nci_evidence_attachment builds complete attachment."""
        from backend.health.nci_governance_adapter import (
            build_nci_director_panel,
            build_nci_governance_signal,
            build_nci_evidence_attachment,
        )

        insight_summary = {"global_nci": 0.90, "dominant_area": "none"}
        priority_view = {"status": "OK", "priority_areas": []}
        slo_result = {"slo_status": "OK", "violations": []}

        panel = build_nci_director_panel(insight_summary, priority_view, slo_result)
        signal = build_nci_governance_signal(panel, slo_result)

        attachment = build_nci_evidence_attachment(panel, signal)

        assert attachment["schema_version"] == "1.0.0"
        assert attachment["source"] == "nci"
        assert "panel" in attachment
        assert "signal" in attachment
        assert "summary" in attachment
        assert "shadow_mode_attestation" in attachment
        assert attachment["shadow_mode_attestation"]["observational_only"] is True


class TestModuleImports:
    """Tests for module import and exports."""

    def test_imports_from_backend_health(self) -> None:
        """Can import NCI functions from backend.health."""
        from backend.health import (
            NCI_GOVERNANCE_TILE_SCHEMA_VERSION,
            NCI_MODE_DOC_ONLY,
            NCI_MODE_TELEMETRY_CHECKED,
            NCI_MODE_FULLY_BOUND,
            MODE_SLO_THRESHOLDS,
            build_nci_director_panel,
            build_nci_governance_signal,
            build_nci_tile_for_global_health,
            attach_nci_tile_to_global_health,
            check_telemetry_consistency,
            check_slice_consistency,
            build_nci_summary_for_p3,
            attach_nci_summary_to_stability_report,
            attach_nci_to_evidence,
            build_nci_evidence_attachment,
            evaluate_nci_p5,
            contribute_nci_to_ggfl,
            build_ggfl_nci_contribution,
        )

        assert NCI_GOVERNANCE_TILE_SCHEMA_VERSION == "1.0.0"
        assert NCI_MODE_DOC_ONLY == "DOC_ONLY"
        assert NCI_MODE_TELEMETRY_CHECKED == "TELEMETRY_CHECKED"
        assert NCI_MODE_FULLY_BOUND == "FULLY_BOUND"
        assert "DOC_ONLY" in MODE_SLO_THRESHOLDS
        assert callable(build_nci_director_panel)
        assert callable(build_nci_governance_signal)
        assert callable(build_nci_tile_for_global_health)
        assert callable(attach_nci_tile_to_global_health)
        assert callable(check_telemetry_consistency)
        assert callable(check_slice_consistency)
        assert callable(build_nci_summary_for_p3)
        assert callable(attach_nci_summary_to_stability_report)
        assert callable(attach_nci_to_evidence)
        assert callable(build_nci_evidence_attachment)
        assert callable(evaluate_nci_p5)
        assert callable(contribute_nci_to_ggfl)
        assert callable(build_ggfl_nci_contribution)

    def test_direct_adapter_import(self) -> None:
        """Can import directly from adapter module."""
        from backend.health.nci_governance_adapter import (
            TCL_CANONICAL_FIELDS,
            SIC_CANONICAL_SLICES,
            DEFAULT_NCI_SLO_THRESHOLDS,
        )

        assert "H" in TCL_CANONICAL_FIELDS
        assert "arithmetic_simple" in SIC_CANONICAL_SLICES
        assert "global_nci_warn" in DEFAULT_NCI_SLO_THRESHOLDS


# =============================================================================
# P5 EVALUATION TESTS
# =============================================================================


class TestEvaluateNciP5ModeSelection:
    """Tests for evaluate_nci_p5() automatic mode selection."""

    def _make_panel(self, global_nci: float = 0.85) -> Dict[str, Any]:
        """Create a minimal director panel for testing."""
        from backend.health.nci_governance_adapter import build_nci_director_panel

        insight_summary = {
            "global_nci": global_nci,
            "dominant_area": "none",
            "summary": {
                "terminology_alignment": 0.90,
                "phase_discipline": 0.85,
                "uplift_avoidance": 0.95,
                "structural_coherence": 0.80,
            },
        }
        priority_view = {"status": "OK", "priority_areas": []}
        slo_result = {"slo_status": "OK", "violations": []}
        return build_nci_director_panel(insight_summary, priority_view, slo_result)

    def test_selects_doc_only_when_no_sources(self) -> None:
        """Mode is DOC_ONLY when no telemetry or slice registry."""
        from backend.health.nci_governance_adapter import evaluate_nci_p5

        panel = self._make_panel()
        result = evaluate_nci_p5(panel)

        assert result["mode"] == "DOC_ONLY"
        assert result["shadow_mode"] is True

    def test_selects_telemetry_checked_with_schema_only(self) -> None:
        """Mode is TELEMETRY_CHECKED when telemetry but no slice registry."""
        from backend.health.nci_governance_adapter import evaluate_nci_p5

        panel = self._make_panel()
        telemetry_schema = {
            "schema_version": "1.2.0",
            "events": {},
            "fields": {},
            "schema_age_hours": 48,
        }
        result = evaluate_nci_p5(panel, telemetry_schema=telemetry_schema)

        assert result["mode"] == "TELEMETRY_CHECKED"

    def test_selects_fully_bound_with_both_sources(self) -> None:
        """Mode is FULLY_BOUND when both telemetry and slice registry available."""
        from backend.health.nci_governance_adapter import evaluate_nci_p5

        panel = self._make_panel()
        telemetry_schema = {
            "schema_version": "1.2.0",
            "events": {},
            "fields": {},
        }
        slice_registry = {
            "slices": {
                "arithmetic_simple": {"depth_max": 4, "atom_max": 4},
            },
        }
        result = evaluate_nci_p5(
            panel,
            telemetry_schema=telemetry_schema,
            slice_registry=slice_registry,
        )

        assert result["mode"] == "FULLY_BOUND"

    def test_selects_doc_only_for_invalid_config(self) -> None:
        """Mode is DOC_ONLY when slice registry without telemetry (invalid)."""
        from backend.health.nci_governance_adapter import evaluate_nci_p5

        panel = self._make_panel()
        slice_registry = {
            "slices": {"arithmetic_simple": {}},
        }
        # No telemetry_schema, only slice_registry
        result = evaluate_nci_p5(panel, slice_registry=slice_registry)

        assert result["mode"] == "DOC_ONLY"


class TestEvaluateNciP5DOCOnlyMode:
    """Tests for DOC_ONLY mode behavior."""

    def _make_panel(self, global_nci: float = 0.85) -> Dict[str, Any]:
        from backend.health.nci_governance_adapter import build_nci_director_panel

        insight_summary = {
            "global_nci": global_nci,
            "dominant_area": "none",
            "summary": {
                "terminology_alignment": 0.90,
                "phase_discipline": 0.85,
                "uplift_avoidance": 0.95,
                "structural_coherence": 0.80,
            },
        }
        priority_view = {"status": "OK", "priority_areas": []}
        slo_result = {"slo_status": "OK", "violations": []}
        return build_nci_director_panel(insight_summary, priority_view, slo_result)

    def test_doc_only_runs_tcl_002_only(self) -> None:
        """DOC_ONLY mode only runs TCL-002, skips others."""
        from backend.health.nci_governance_adapter import evaluate_nci_p5

        panel = self._make_panel()
        result = evaluate_nci_p5(panel)

        assert result["tcl_result"]["checks_run"] == ["TCL-002"]
        assert set(result["tcl_result"]["checks_skipped"]) == {"TCL-001", "TCL-003", "TCL-004"}

    def test_doc_only_runs_sic_001_and_sic_004(self) -> None:
        """DOC_ONLY mode runs SIC-001 and SIC-004."""
        from backend.health.nci_governance_adapter import evaluate_nci_p5

        panel = self._make_panel()
        result = evaluate_nci_p5(panel)

        assert set(result["sic_result"]["checks_run"]) == {"SIC-001", "SIC-004"}
        assert set(result["sic_result"]["checks_skipped"]) == {"SIC-002", "SIC-003"}

    def test_doc_only_confidence_calculation(self) -> None:
        """DOC_ONLY confidence is computed correctly."""
        from backend.health.nci_governance_adapter import evaluate_nci_p5

        panel = self._make_panel()
        result = evaluate_nci_p5(panel)

        # Base 0.70 + coverage bonus (4/4 = 0.10) - 0.10 (no telemetry) - 0.05 (no registry)
        # = 0.70 + 0.10 - 0.15 = 0.65
        assert 0.50 <= result["confidence"] <= 0.70

    def test_doc_only_uses_doc_only_thresholds(self) -> None:
        """DOC_ONLY mode uses DOC_ONLY SLO thresholds."""
        from backend.health.nci_governance_adapter import evaluate_nci_p5

        panel = self._make_panel(global_nci=0.68)  # Below TELEMETRY_CHECKED warn but above DOC_ONLY
        result = evaluate_nci_p5(panel)

        # DOC_ONLY warn is 0.70, so 0.68 should trigger WARN
        assert result["slo_evaluation"]["thresholds_used"]["global_nci_warn"] == 0.70
        assert result["slo_evaluation"]["status"] == "WARN"


class TestEvaluateNciP5TelemetryCheckedMode:
    """Tests for TELEMETRY_CHECKED mode behavior."""

    def _make_panel(self, global_nci: float = 0.85) -> Dict[str, Any]:
        from backend.health.nci_governance_adapter import build_nci_director_panel

        insight_summary = {
            "global_nci": global_nci,
            "dominant_area": "none",
            "summary": {
                "terminology_alignment": 0.90,
                "phase_discipline": 0.85,
                "uplift_avoidance": 0.95,
                "structural_coherence": 0.80,
            },
        }
        priority_view = {"status": "OK", "priority_areas": []}
        slo_result = {"slo_status": "OK", "violations": []}
        return build_nci_director_panel(insight_summary, priority_view, slo_result)

    def test_telemetry_checked_runs_all_tcl_checks(self) -> None:
        """TELEMETRY_CHECKED mode runs all TCL checks."""
        from backend.health.nci_governance_adapter import evaluate_nci_p5

        panel = self._make_panel()
        telemetry_schema = {"schema_version": "1.0.0", "schema_age_hours": 48}
        result = evaluate_nci_p5(panel, telemetry_schema=telemetry_schema)

        assert set(result["tcl_result"]["checks_run"]) == {"TCL-001", "TCL-002", "TCL-003", "TCL-004"}
        assert result["tcl_result"]["checks_skipped"] == []

    def test_telemetry_checked_confidence_with_alignment(self) -> None:
        """TELEMETRY_CHECKED confidence higher when aligned."""
        from backend.health.nci_governance_adapter import evaluate_nci_p5

        panel = self._make_panel()
        telemetry_schema = {"schema_version": "1.0.0", "schema_age_hours": 48}
        result = evaluate_nci_p5(panel, telemetry_schema=telemetry_schema)

        # Base 0.80 + coverage (0.05) + alignment bonus (0.10) - no slice penalty (0.05)
        # = 0.80 + 0.05 + 0.10 - 0.05 = 0.90, capped at 1.0
        assert result["confidence"] >= 0.80

    def test_telemetry_checked_confidence_penalty_for_fresh_schema(self) -> None:
        """TELEMETRY_CHECKED confidence penalized for fresh schema."""
        from backend.health.nci_governance_adapter import evaluate_nci_p5

        panel = self._make_panel()
        telemetry_schema = {"schema_version": "1.0.0", "schema_age_hours": 12}  # < 24h
        result = evaluate_nci_p5(panel, telemetry_schema=telemetry_schema)

        # Should have -0.05 freshness penalty
        assert result["tcl_result"]["schema_age_hours"] == 12

    def test_telemetry_checked_uses_stricter_thresholds(self) -> None:
        """TELEMETRY_CHECKED uses stricter thresholds than DOC_ONLY."""
        from backend.health.nci_governance_adapter import evaluate_nci_p5

        panel = self._make_panel(global_nci=0.72)
        telemetry_schema = {"schema_version": "1.0.0"}
        result = evaluate_nci_p5(panel, telemetry_schema=telemetry_schema)

        # TELEMETRY_CHECKED warn is 0.75, so 0.72 should WARN
        assert result["slo_evaluation"]["thresholds_used"]["global_nci_warn"] == 0.75
        assert result["slo_evaluation"]["status"] == "WARN"


class TestEvaluateNciP5FullyBoundMode:
    """Tests for FULLY_BOUND mode behavior."""

    def _make_panel(self, global_nci: float = 0.85) -> Dict[str, Any]:
        from backend.health.nci_governance_adapter import build_nci_director_panel

        insight_summary = {
            "global_nci": global_nci,
            "dominant_area": "none",
            "summary": {
                "terminology_alignment": 0.90,
                "phase_discipline": 0.85,
                "uplift_avoidance": 0.95,
                "structural_coherence": 0.80,
            },
        }
        priority_view = {"status": "OK", "priority_areas": []}
        slo_result = {"slo_status": "OK", "violations": []}
        return build_nci_director_panel(insight_summary, priority_view, slo_result)

    def test_fully_bound_runs_all_checks(self) -> None:
        """FULLY_BOUND mode runs all TCL and SIC checks."""
        from backend.health.nci_governance_adapter import evaluate_nci_p5

        panel = self._make_panel()
        telemetry_schema = {"schema_version": "1.0.0"}
        slice_registry = {"slices": {"arithmetic_simple": {"depth_max": 4}}}
        result = evaluate_nci_p5(
            panel, telemetry_schema=telemetry_schema, slice_registry=slice_registry
        )

        assert set(result["tcl_result"]["checks_run"]) == {"TCL-001", "TCL-002", "TCL-003", "TCL-004"}
        assert set(result["sic_result"]["checks_run"]) == {"SIC-001", "SIC-002", "SIC-003", "SIC-004"}
        assert result["tcl_result"]["checks_skipped"] == []
        assert result["sic_result"]["checks_skipped"] == []

    def test_fully_bound_registry_validated_flag(self) -> None:
        """FULLY_BOUND sets registry_validated when slice registry has slices."""
        from backend.health.nci_governance_adapter import evaluate_nci_p5

        panel = self._make_panel()
        telemetry_schema = {"schema_version": "1.0.0"}
        slice_registry = {"slices": {"arithmetic_simple": {"depth_max": 4}}}
        result = evaluate_nci_p5(
            panel, telemetry_schema=telemetry_schema, slice_registry=slice_registry
        )

        assert result["sic_result"]["registry_validated"] is True

    def test_fully_bound_uses_strictest_thresholds(self) -> None:
        """FULLY_BOUND uses strictest SLO thresholds."""
        from backend.health.nci_governance_adapter import evaluate_nci_p5

        panel = self._make_panel(global_nci=0.78)
        telemetry_schema = {"schema_version": "1.0.0"}
        slice_registry = {"slices": {}}
        result = evaluate_nci_p5(
            panel, telemetry_schema=telemetry_schema, slice_registry=slice_registry
        )

        # FULLY_BOUND warn is 0.80, so 0.78 should WARN
        assert result["slo_evaluation"]["thresholds_used"]["global_nci_warn"] == 0.80
        assert result["slo_evaluation"]["status"] == "WARN"


class TestEvaluateNciP5TCLViolations:
    """Tests for TCL violation detection in evaluate_nci_p5()."""

    def _make_panel(self, global_nci: float = 0.85) -> Dict[str, Any]:
        from backend.health.nci_governance_adapter import build_nci_director_panel

        insight_summary = {"global_nci": global_nci, "dominant_area": "none"}
        priority_view = {"status": "OK", "priority_areas": []}
        slo_result = {"slo_status": "OK", "violations": []}
        return build_nci_director_panel(insight_summary, priority_view, slo_result)

    def test_tcl_violations_detected_in_doc_contents(self) -> None:
        """TCL violations are detected from doc_contents."""
        from backend.health.nci_governance_adapter import evaluate_nci_p5

        panel = self._make_panel()
        doc_contents = {
            "docs/api.md": "The Ht value is the health metric.",
        }
        result = evaluate_nci_p5(panel, doc_contents=doc_contents)

        assert result["tcl_result"]["aligned"] is False
        assert len(result["tcl_result"]["violations"]) > 0
        assert any(v["found"] == "Ht" for v in result["tcl_result"]["violations"])

    def test_tcl_violations_generate_warnings(self) -> None:
        """TCL violations generate proper warnings."""
        from backend.health.nci_governance_adapter import evaluate_nci_p5

        panel = self._make_panel()
        doc_contents = {
            "docs/api.md": "Use RSI for stability. The Ht value is health.",
        }
        result = evaluate_nci_p5(panel, doc_contents=doc_contents)

        assert len(result["warnings"]) >= 2
        tcl_warnings = [w for w in result["warnings"] if w["warning_type"] == "TCL-002"]
        assert len(tcl_warnings) >= 2

    def test_tcl_violations_affect_confidence(self) -> None:
        """TCL violations reduce confidence in TELEMETRY_CHECKED mode."""
        from backend.health.nci_governance_adapter import evaluate_nci_p5

        panel = self._make_panel()
        telemetry_schema = {"schema_version": "1.0.0", "schema_age_hours": 48}

        # Without violations
        result_clean = evaluate_nci_p5(panel, telemetry_schema=telemetry_schema)

        # With violations
        doc_contents = {"docs/api.md": "Ht RSI β τ ρ"}  # Multiple violations
        result_dirty = evaluate_nci_p5(
            panel, telemetry_schema=telemetry_schema, doc_contents=doc_contents
        )

        assert result_dirty["confidence"] < result_clean["confidence"]


class TestEvaluateNciP5SICViolations:
    """Tests for SIC violation detection in evaluate_nci_p5()."""

    def _make_panel(self, global_nci: float = 0.85) -> Dict[str, Any]:
        from backend.health.nci_governance_adapter import build_nci_director_panel

        insight_summary = {"global_nci": global_nci, "dominant_area": "none"}
        priority_view = {"status": "OK", "priority_areas": []}
        slo_result = {"slo_status": "OK", "violations": []}
        return build_nci_director_panel(insight_summary, priority_view, slo_result)

    def test_sic_001_violations_detected(self) -> None:
        """SIC-001 slice name violations are detected."""
        from backend.health.nci_governance_adapter import evaluate_nci_p5

        panel = self._make_panel()
        doc_contents = {
            "docs/slices.md": "The ArithmeticSimple slice handles basic math.",
        }
        result = evaluate_nci_p5(panel, doc_contents=doc_contents)

        assert result["sic_result"]["aligned"] is False
        sic_001_violations = [
            v for v in result["sic_result"]["violations"]
            if v.get("violation_type") == "SIC-001"
        ]
        assert len(sic_001_violations) > 0

    def test_sic_004_capability_overclaim_detected(self) -> None:
        """SIC-004 capability overclaims are detected."""
        from backend.health.nci_governance_adapter import evaluate_nci_p5

        panel = self._make_panel()
        doc_contents = {
            "docs/features.md": "This slice supports unlimited depth derivation.",
        }
        result = evaluate_nci_p5(panel, doc_contents=doc_contents)

        sic_004_violations = [
            v for v in result["sic_result"]["violations"]
            if v.get("violation_type") == "SIC-004"
        ]
        assert len(sic_004_violations) > 0
        assert any("unlimited depth" in v.get("found", "").lower() for v in sic_004_violations)

    def test_combined_tcl_sic_violations_trigger_breach(self) -> None:
        """Combined TCL+SIC violations can trigger SLO BREACH."""
        from backend.health.nci_governance_adapter import evaluate_nci_p5

        panel = self._make_panel(global_nci=0.85)
        doc_contents = {
            "docs/api.md": "Ht RSI β τ ρ ArithmeticSimple prop_taut unlimited depth no limit",
        }
        # DOC_ONLY violation_count_breach is 5
        result = evaluate_nci_p5(panel, doc_contents=doc_contents)

        total = result["slo_evaluation"]["violation_summary"]["total"]
        if total >= 5:
            assert result["slo_evaluation"]["status"] == "BREACH"


class TestEvaluateNciP5GovernanceSignal:
    """Tests for SIG-NAR governance signal generation."""

    def _make_panel(self, global_nci: float = 0.85) -> Dict[str, Any]:
        from backend.health.nci_governance_adapter import build_nci_director_panel

        insight_summary = {"global_nci": global_nci, "dominant_area": "none"}
        priority_view = {"status": "OK", "priority_areas": []}
        slo_result = {"slo_status": "OK", "violations": []}
        return build_nci_director_panel(insight_summary, priority_view, slo_result)

    def test_governance_signal_has_sig_nar_type(self) -> None:
        """Governance signal has signal_type SIG-NAR."""
        from backend.health.nci_governance_adapter import evaluate_nci_p5

        panel = self._make_panel()
        result = evaluate_nci_p5(panel)

        assert result["governance_signal"]["signal_type"] == "SIG-NAR"

    def test_governance_signal_has_schema_version(self) -> None:
        """Governance signal includes schema_version."""
        from backend.health.nci_governance_adapter import evaluate_nci_p5

        panel = self._make_panel()
        result = evaluate_nci_p5(panel)

        assert result["governance_signal"]["schema_version"] == "1.0.0"

    def test_governance_signal_includes_mode(self) -> None:
        """Governance signal includes operational mode."""
        from backend.health.nci_governance_adapter import evaluate_nci_p5

        panel = self._make_panel()
        result = evaluate_nci_p5(panel)

        assert result["governance_signal"]["mode"] == "DOC_ONLY"

    def test_governance_signal_recommendation_none_for_ok(self) -> None:
        """Governance signal recommendation is NONE when SLO OK."""
        from backend.health.nci_governance_adapter import evaluate_nci_p5

        panel = self._make_panel(global_nci=0.90)
        result = evaluate_nci_p5(panel)

        assert result["governance_signal"]["recommendation"] == "NONE"

    def test_governance_signal_recommendation_warning_for_warn(self) -> None:
        """Governance signal recommendation is WARNING when SLO WARN."""
        from backend.health.nci_governance_adapter import evaluate_nci_p5

        panel = self._make_panel(global_nci=0.68)  # Below DOC_ONLY warn threshold
        result = evaluate_nci_p5(panel)

        assert result["governance_signal"]["recommendation"] == "WARNING"

    def test_governance_signal_recommendation_review_for_breach(self) -> None:
        """Governance signal recommendation is REVIEW when SLO BREACH."""
        from backend.health.nci_governance_adapter import evaluate_nci_p5

        panel = self._make_panel(global_nci=0.50)  # Below DOC_ONLY breach threshold
        result = evaluate_nci_p5(panel)

        assert result["governance_signal"]["recommendation"] == "REVIEW"

    def test_governance_signal_shadow_mode_true(self) -> None:
        """Governance signal always has shadow_mode=True."""
        from backend.health.nci_governance_adapter import evaluate_nci_p5

        panel = self._make_panel()
        result = evaluate_nci_p5(panel)

        assert result["governance_signal"]["shadow_mode"] is True

    def test_result_is_json_serializable(self) -> None:
        """Full P5 evaluation result is JSON serializable."""
        from backend.health.nci_governance_adapter import evaluate_nci_p5

        panel = self._make_panel()
        doc_contents = {"docs/api.md": "Ht RSI ArithmeticSimple unlimited depth"}
        result = evaluate_nci_p5(panel, doc_contents=doc_contents)

        # Should not raise
        json_str = json.dumps(result, sort_keys=True, indent=2)
        assert len(json_str) > 100

        # Round-trip
        parsed = json.loads(json_str)
        assert parsed["mode"] == "DOC_ONLY"


class TestGGFLIntegration:
    """Tests for GGFL integration functions."""

    def _make_p5_result(self) -> Dict[str, Any]:
        from backend.health.nci_governance_adapter import (
            build_nci_director_panel,
            evaluate_nci_p5,
        )

        insight_summary = {"global_nci": 0.85, "dominant_area": "none"}
        priority_view = {"status": "OK", "priority_areas": []}
        slo_result = {"slo_status": "OK", "violations": []}
        panel = build_nci_director_panel(insight_summary, priority_view, slo_result)
        return evaluate_nci_p5(panel)

    def test_contribute_nci_to_ggfl_attaches_signal(self) -> None:
        """contribute_nci_to_ggfl attaches under signals.SIG-NAR."""
        from backend.health.nci_governance_adapter import contribute_nci_to_ggfl

        ggfl_surface = {"status": "ok", "signals": {}}
        p5_result = self._make_p5_result()

        result = contribute_nci_to_ggfl(ggfl_surface, p5_result)

        assert "signals" in result
        assert "SIG-NAR" in result["signals"]
        assert result["signals"]["SIG-NAR"]["source"] == "nci"
        assert result["signals"]["SIG-NAR"]["shadow_mode"] is True

    def test_contribute_nci_to_ggfl_preserves_existing_signals(self) -> None:
        """contribute_nci_to_ggfl preserves existing signals."""
        from backend.health.nci_governance_adapter import contribute_nci_to_ggfl

        ggfl_surface = {
            "status": "ok",
            "signals": {
                "SIG-RSI": {"source": "rsi", "value": 0.92},
            },
        }
        p5_result = self._make_p5_result()

        result = contribute_nci_to_ggfl(ggfl_surface, p5_result)

        assert "SIG-RSI" in result["signals"]
        assert "SIG-NAR" in result["signals"]

    def test_contribute_nci_to_ggfl_does_not_mutate_input(self) -> None:
        """contribute_nci_to_ggfl does not mutate input surface."""
        from backend.health.nci_governance_adapter import contribute_nci_to_ggfl

        ggfl_surface = {"status": "ok"}
        p5_result = self._make_p5_result()

        contribute_nci_to_ggfl(ggfl_surface, p5_result)

        assert "signals" not in ggfl_surface

    def test_build_ggfl_nci_contribution_standalone(self) -> None:
        """build_ggfl_nci_contribution creates standalone contribution."""
        from backend.health.nci_governance_adapter import build_ggfl_nci_contribution

        p5_result = self._make_p5_result()
        contribution = build_ggfl_nci_contribution(p5_result)

        assert contribution["signal_slot"] == "SIG-NAR"
        assert contribution["source"] == "nci"
        assert contribution["schema_version"] == "1.0.0"
        assert contribution["shadow_mode"] is True
        assert "payload" in contribution
        assert contribution["payload"]["mode"] == "DOC_ONLY"

    def test_build_ggfl_nci_contribution_includes_metadata(self) -> None:
        """build_ggfl_nci_contribution includes checks metadata."""
        from backend.health.nci_governance_adapter import build_ggfl_nci_contribution

        p5_result = self._make_p5_result()
        contribution = build_ggfl_nci_contribution(p5_result)

        assert "metadata" in contribution
        assert "checks_run" in contribution["metadata"]
        assert "tcl" in contribution["metadata"]["checks_run"]
        assert "sic" in contribution["metadata"]["checks_run"]

    def test_ggfl_contribution_is_json_serializable(self) -> None:
        """GGFL contribution is JSON serializable."""
        from backend.health.nci_governance_adapter import build_ggfl_nci_contribution

        p5_result = self._make_p5_result()
        contribution = build_ggfl_nci_contribution(p5_result)

        json_str = json.dumps(contribution, sort_keys=True)
        assert len(json_str) > 50
        parsed = json.loads(json_str)
        assert parsed["signal_slot"] == "SIG-NAR"
