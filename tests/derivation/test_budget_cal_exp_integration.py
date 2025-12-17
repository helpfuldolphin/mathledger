"""
Tests for budget storyline integration into CAL-EXP reports.
"""

import json
from typing import Any, Dict

import pytest

from derivation.budget_cal_exp_integration import (
    annotate_cal_exp_windows_with_budget_storyline,
    attach_budget_storyline_to_cal_exp_report,
    build_budget_confounding_truth_table,
    build_cal_exp_budget_storyline_from_snapshots,
    extract_budget_storyline_from_cal_exp_report,
    validate_budget_confounding_defaults,
)
from derivation.budget_invariants import (
    build_budget_invariant_snapshot,
    build_budget_invariant_timeline,
    build_budget_storyline,
    project_budget_stability_horizon,
)


# Mock PipelineStats-like objects
class MockStats:
    def __init__(
        self,
        budget_exhausted: bool = False,
        max_candidates_hit: bool = False,
        timeout_abstentions: int = 0,
        statements_skipped: int = 0,
        candidates_considered: int = 0,
        budget_remaining_s: float | None = None,
        post_exhaustion_candidates: int = 0,
    ):
        self.budget_exhausted = budget_exhausted
        self.max_candidates_hit = max_candidates_hit
        self.timeout_abstentions = timeout_abstentions
        self.statements_skipped = statements_skipped
        self.candidates_considered = candidates_considered
        self.budget_remaining_s = budget_remaining_s
        self.post_exhaustion_candidates = post_exhaustion_candidates


def test_attach_budget_storyline_to_cal_exp_report_cal_exp1():
    """Test attaching budget storyline to CAL-EXP-1 report."""
    # Build synthetic timeline, storyline, projection
    snapshots = [build_budget_invariant_snapshot(MockStats()) for _ in range(5)]
    timeline = build_budget_invariant_timeline(snapshots)
    health_history = [{"health_score": 90.0, "trend_status": "STABLE"} for _ in range(5)]
    storyline = build_budget_storyline(timeline, health_history)
    projection = project_budget_stability_horizon(health_history)
    
    # Create CAL-EXP-1 report
    report = {
        "schema_version": "1.0.0",
        "params": {
            "adapter": "real",
            "cycles": 200,
            "learning_rate": 0.1,
            "seed": 42,
        },
        "windows": [],
        "summary": {
            "final_divergence_rate": 0.05,
        },
    }
    
    # Attach budget storyline
    enriched = attach_budget_storyline_to_cal_exp_report(
        report=report,
        timeline=timeline,
        storyline=storyline,
        projection=projection,
        experiment_id="CAL-EXP-1",
        run_id="cal_exp1_20250101_120000",
    )
    
    # Verify structure
    assert "budget_storyline_summary" in enriched
    summary = enriched["budget_storyline_summary"]
    assert summary["schema_version"] == "1.0.0"
    assert summary["experiment_id"] == "CAL-EXP-1"
    assert summary["run_id"] == "cal_exp1_20250101_120000"
    assert "combined_status" in summary
    assert "stability_index" in summary
    assert "episodes_count" in summary
    assert "projection_class" in summary
    assert "key_structural_events" in summary
    
    # Verify original report unchanged (non-mutating)
    assert "budget_storyline_summary" not in report
    
    # Verify JSON serializable
    json_str = json.dumps(enriched)
    assert isinstance(json_str, str)
    decoded = json.loads(json_str)
    assert decoded == enriched


def test_attach_budget_storyline_to_cal_exp_report_cal_exp2_with_window():
    """Test attaching budget storyline to CAL-EXP-2 report with window bounds."""
    snapshots = [build_budget_invariant_snapshot(MockStats()) for _ in range(10)]
    timeline = build_budget_invariant_timeline(snapshots)
    health_history = [{"health_score": 85.0, "trend_status": "STABLE"} for _ in range(10)]
    storyline = build_budget_storyline(timeline, health_history)
    projection = project_budget_stability_horizon(health_history)
    
    report = {
        "schema_version": "1.0.0",
        "params": {"cycles": 1000, "seed": 123},
        "windows": [],
    }
    
    enriched = attach_budget_storyline_to_cal_exp_report(
        report=report,
        timeline=timeline,
        storyline=storyline,
        projection=projection,
        experiment_id="CAL-EXP-2",
        run_id="cal_exp2_20250101_120000",
        window_start=0,
        window_end=100,
    )
    
    summary = enriched["budget_storyline_summary"]
    assert summary["experiment_id"] == "CAL-EXP-2"
    assert summary["window_start"] == 0
    assert summary["window_end"] == 100


def test_attach_budget_storyline_to_cal_exp_report_cal_exp3():
    """Test attaching budget storyline to CAL-EXP-3 report."""
    snapshots = [build_budget_invariant_snapshot(MockStats(timeout_abstentions=5)) for _ in range(3)]
    timeline = build_budget_invariant_timeline(snapshots)
    health_history = [{"health_score": 75.0, "trend_status": "DEGRADING"} for _ in range(3)]
    storyline = build_budget_storyline(timeline, health_history)
    projection = project_budget_stability_horizon(health_history)
    
    report = {
        "schema_version": "1.0.0",
        "params": {"cycles": 500},
        "regime_change_probes": [],
    }
    
    enriched = attach_budget_storyline_to_cal_exp_report(
        report=report,
        timeline=timeline,
        storyline=storyline,
        projection=projection,
        experiment_id="CAL-EXP-3",
        run_id="cal_exp3_20250101_120000",
        window_start=200,
        window_end=300,
    )
    
    summary = enriched["budget_storyline_summary"]
    assert summary["experiment_id"] == "CAL-EXP-3"
    assert summary["window_start"] == 200
    assert summary["window_end"] == 300


def test_extract_budget_storyline_from_cal_exp_report():
    """Test extracting budget storyline from CAL-EXP report."""
    # Report with storyline
    report_with = {
        "schema_version": "1.0.0",
        "budget_storyline_summary": {
            "schema_version": "1.0.0",
            "experiment_id": "CAL-EXP-1",
            "combined_status": "OK",
        },
    }
    
    extracted = extract_budget_storyline_from_cal_exp_report(report_with)
    assert extracted is not None
    assert extracted["experiment_id"] == "CAL-EXP-1"
    
    # Report without storyline
    report_without = {"schema_version": "1.0.0", "windows": []}
    
    extracted = extract_budget_storyline_from_cal_exp_report(report_without)
    assert extracted is None


def test_build_cal_exp_budget_storyline_from_snapshots():
    """Test building budget storyline summary from snapshots."""
    snapshots = [build_budget_invariant_snapshot(MockStats()) for _ in range(5)]
    health_history = [{"health_score": 90.0, "trend_status": "STABLE"} for _ in range(5)]
    
    summary = build_cal_exp_budget_storyline_from_snapshots(
        snapshots=snapshots,
        budget_health_history=health_history,
        experiment_id="CAL-EXP-1",
        run_id="cal_exp1_20250101_120000",
        window_start=0,
        window_end=200,
    )
    
    assert summary["schema_version"] == "1.0.0"
    assert summary["experiment_id"] == "CAL-EXP-1"
    assert summary["run_id"] == "cal_exp1_20250101_120000"
    assert summary["window_start"] == 0
    assert summary["window_end"] == 200
    assert "combined_status" in summary
    assert "stability_index" in summary
    assert "episodes_count" in summary
    assert "projection_class" in summary
    assert "key_structural_events" in summary
    
    # Verify JSON serializable
    json_str = json.dumps(summary)
    assert isinstance(json_str, str)


def test_cal_exp_budget_storyline_deterministic():
    """Test CAL-EXP budget storyline is deterministic."""
    snapshots = [build_budget_invariant_snapshot(MockStats()) for _ in range(5)]
    health_history = [{"health_score": 90.0, "trend_status": "STABLE"} for _ in range(5)]
    
    report = {"schema_version": "1.0.0", "windows": []}
    snapshots = [build_budget_invariant_snapshot(MockStats()) for _ in range(5)]
    timeline = build_budget_invariant_timeline(snapshots)
    storyline = build_budget_storyline(timeline, health_history)
    projection = project_budget_stability_horizon(health_history)
    
    enriched1 = attach_budget_storyline_to_cal_exp_report(
        report, timeline, storyline, projection, "CAL-EXP-1", "run1"
    )
    enriched2 = attach_budget_storyline_to_cal_exp_report(
        report, timeline, storyline, projection, "CAL-EXP-1", "run1"
    )
    
    assert enriched1 == enriched2


def test_annotate_cal_exp_windows_with_budget_storyline():
    """Test annotating CAL-EXP windows with budget storyline (strict mode, default)."""
    report = {
        "schema_version": "1.0.0",
        "windows": [
            {"divergence_rate": 0.05, "delta_bias": 0.01},
            {"divergence_rate": 0.06, "delta_bias": 0.02},
            {"divergence_rate": 0.04, "delta_bias": 0.01},
        ],
    }
    
    budget_storyline = {
        "schema_version": "1.0.0",
        "combined_status": "OK",
        "stability_index": 0.98,
        "projection_class": "STABLE",
    }
    
    annotated = annotate_cal_exp_windows_with_budget_storyline(report, budget_storyline)
    
    # Verify structure
    assert "windows" in annotated
    assert len(annotated["windows"]) == 3
    
    # Verify each window is annotated
    for window in annotated["windows"]:
        assert "budget_combined_status" in window
        assert "budget_stability_index" in window
        assert "budget_projection_class" in window
        assert "budget_confounded" in window
        assert "budget_confound_reason" in window
        assert "confound_stability_threshold" in window
        assert window["budget_combined_status"] == "OK"
        assert window["budget_stability_index"] == 0.98
        assert window["budget_projection_class"] == "STABLE"
        assert window["budget_confounded"] is False  # OK status AND stability_index >= 0.95
        assert window["budget_confound_reason"] is None
        assert window["confound_stability_threshold"] == 0.95
    
    # Verify original report unchanged (non-mutating)
    assert "budget_combined_status" not in report["windows"][0]
    
    # Verify original window data preserved
    assert annotated["windows"][0]["divergence_rate"] == 0.05
    assert annotated["windows"][1]["delta_bias"] == 0.02


def test_annotate_windows_strict_confounding_warn_but_high_stability():
    """Test strict mode: WARN status with high stability_index is NOT confounded."""
    report = {"schema_version": "1.0.0", "windows": [{"divergence_rate": 0.05}]}
    
    budget_storyline = {
        "combined_status": "WARN",
        "stability_index": 0.98,  # Above threshold
        "projection_class": "DRIFTING",
    }
    
    annotated = annotate_cal_exp_windows_with_budget_storyline(report, budget_storyline, strict_confounding=True)
    
    # Strict mode: WARN alone is NOT confounded (needs both WARN AND low stability)
    assert annotated["windows"][0]["budget_confounded"] is False
    assert annotated["windows"][0]["budget_combined_status"] == "WARN"
    assert annotated["windows"][0]["budget_confound_reason"] == "STATUS_WARN"


def test_annotate_windows_strict_confounding_ok_but_low_stability():
    """Test strict mode: OK status with low stability_index is NOT confounded."""
    report = {"schema_version": "1.0.0", "windows": [{"divergence_rate": 0.05}]}
    
    budget_storyline = {
        "combined_status": "OK",
        "stability_index": 0.90,  # Below 0.95 threshold
        "projection_class": "STABLE",
    }
    
    annotated = annotate_cal_exp_windows_with_budget_storyline(report, budget_storyline, strict_confounding=True)
    
    # Strict mode: Low stability alone is NOT confounded (needs both WARN/BLOCK AND low stability)
    assert annotated["windows"][0]["budget_confounded"] is False
    assert annotated["windows"][0]["budget_stability_index"] == 0.90
    assert annotated["windows"][0]["budget_confound_reason"] == "LOW_STABILITY_INDEX"


def test_annotate_windows_strict_confounding_both_conditions():
    """Test strict mode: WARN/BLOCK AND low stability_index IS confounded."""
    report = {"schema_version": "1.0.0", "windows": [{"divergence_rate": 0.05}]}
    
    budget_storyline = {
        "combined_status": "WARN",
        "stability_index": 0.90,  # Below threshold
        "projection_class": "DRIFTING",
    }
    
    annotated = annotate_cal_exp_windows_with_budget_storyline(report, budget_storyline, strict_confounding=True)
    
    # Strict mode: Both conditions met → confounded
    assert annotated["windows"][0]["budget_confounded"] is True
    assert annotated["windows"][0]["budget_confound_reason"] == "STATUS_AND_LOW_STABILITY"


def test_annotate_windows_strict_confounding_block_and_low_stability():
    """Test strict mode: BLOCK status with low stability_index IS confounded."""
    report = {"schema_version": "1.0.0", "windows": [{"divergence_rate": 0.05}]}
    
    budget_storyline = {
        "combined_status": "BLOCK",
        "stability_index": 0.80,  # Below threshold
        "projection_class": "VOLATILE",
    }
    
    annotated = annotate_cal_exp_windows_with_budget_storyline(report, budget_storyline, strict_confounding=True)
    
    # Strict mode: Both conditions met → confounded
    assert annotated["windows"][0]["budget_confounded"] is True
    assert annotated["windows"][0]["budget_combined_status"] == "BLOCK"
    assert annotated["windows"][0]["budget_projection_class"] == "VOLATILE"
    assert annotated["windows"][0]["budget_confound_reason"] == "STATUS_AND_LOW_STABILITY"


def test_annotate_windows_legacy_confounding_warn_status():
    """Test legacy mode: WARN status sets confounded to True (OR-rule)."""
    report = {"schema_version": "1.0.0", "windows": [{"divergence_rate": 0.05}]}
    
    budget_storyline = {
        "combined_status": "WARN",
        "stability_index": 0.98,  # Above threshold
        "projection_class": "DRIFTING",
    }
    
    annotated = annotate_cal_exp_windows_with_budget_storyline(report, budget_storyline, strict_confounding=False)
    
    # Legacy mode: WARN alone sets confounded to True (OR-rule)
    assert annotated["windows"][0]["budget_confounded"] is True
    assert annotated["windows"][0]["budget_confound_reason"] == "STATUS_WARN"


def test_annotate_windows_legacy_confounding_low_stability():
    """Test legacy mode: Low stability_index sets confounded to True (OR-rule)."""
    report = {"schema_version": "1.0.0", "windows": [{"divergence_rate": 0.05}]}
    
    budget_storyline = {
        "combined_status": "OK",
        "stability_index": 0.90,  # Below threshold
        "projection_class": "STABLE",
    }
    
    annotated = annotate_cal_exp_windows_with_budget_storyline(report, budget_storyline, strict_confounding=False)
    
    # Legacy mode: Low stability alone sets confounded to True (OR-rule)
    assert annotated["windows"][0]["budget_confounded"] is True
    assert annotated["windows"][0]["budget_confound_reason"] == "LOW_STABILITY_INDEX"


def test_annotate_windows_custom_threshold():
    """Test custom confound_stability_threshold parameter."""
    report = {"schema_version": "1.0.0", "windows": [{"divergence_rate": 0.05}]}
    
    budget_storyline = {
        "combined_status": "WARN",
        "stability_index": 0.92,  # Above default 0.95, below custom 0.93
        "projection_class": "DRIFTING",
    }
    
    annotated = annotate_cal_exp_windows_with_budget_storyline(
        report, budget_storyline, confound_stability_threshold=0.93, strict_confounding=True
    )
    
    # With custom threshold 0.93, stability_index 0.92 is below threshold
    assert annotated["windows"][0]["budget_confounded"] is True
    assert annotated["windows"][0]["confound_stability_threshold"] == 0.93
    assert annotated["windows"][0]["budget_confound_reason"] == "STATUS_AND_LOW_STABILITY"


def test_annotate_windows_missing_budget_fields():
    """Test handling of missing budget fields (should default safely)."""
    report = {"schema_version": "1.0.0", "windows": [{"divergence_rate": 0.05}]}
    
    budget_storyline = {}  # Missing fields
    
    annotated = annotate_cal_exp_windows_with_budget_storyline(report, budget_storyline, strict_confounding=True)
    
    # Missing fields should default safely
    assert annotated["windows"][0]["budget_combined_status"] == "OK"  # Default
    assert annotated["windows"][0]["budget_stability_index"] == 1.0  # Default
    assert annotated["windows"][0]["budget_confounded"] is False  # OK AND stability >= threshold
    assert annotated["windows"][0]["budget_confound_reason"] is None


def test_annotate_windows_deterministic_ordering():
    """Test windows are sorted deterministically by index."""
    report = {
        "schema_version": "1.0.0",
        "windows": [
            {"divergence_rate": 0.03},  # Window 2
            {"divergence_rate": 0.01},  # Window 0
            {"divergence_rate": 0.02},  # Window 1
        ],
    }
    
    budget_storyline = {
        "combined_status": "OK",
        "stability_index": 0.98,
        "projection_class": "STABLE",
    }
    
    annotated = annotate_cal_exp_windows_with_budget_storyline(report, budget_storyline)
    
    # Windows should be in original order (by enumerate index)
    assert len(annotated["windows"]) == 3
    assert annotated["windows"][0]["divergence_rate"] == 0.03
    assert annotated["windows"][1]["divergence_rate"] == 0.01
    assert annotated["windows"][2]["divergence_rate"] == 0.02
    
    # Multiple calls should produce identical results
    annotated2 = annotate_cal_exp_windows_with_budget_storyline(report, budget_storyline)
    assert annotated == annotated2


def test_annotate_windows_deterministic_json_serialization():
    """Test deterministic JSON serialization of annotated windows."""
    import json
    
    report = {
        "schema_version": "1.0.0",
        "windows": [
            {"divergence_rate": 0.05},
            {"divergence_rate": 0.06},
        ],
    }
    
    budget_storyline = {
        "combined_status": "WARN",
        "stability_index": 0.92,
        "projection_class": "DRIFTING",
    }
    
    annotated1 = annotate_cal_exp_windows_with_budget_storyline(report, budget_storyline, strict_confounding=True)
    annotated2 = annotate_cal_exp_windows_with_budget_storyline(report, budget_storyline, strict_confounding=True)
    
    # JSON serialization should be deterministic
    json1 = json.dumps(annotated1, sort_keys=True)
    json2 = json.dumps(annotated2, sort_keys=True)
    
    assert json1 == json2
    assert "budget_confound_reason" in json1
    assert "confound_stability_threshold" in json1


def test_truth_table_contains_all_rows_and_is_deterministic():
    """Test truth table contains all 6 combinations and is deterministic."""
    table1 = build_budget_confounding_truth_table()
    table2 = build_budget_confounding_truth_table()
    
    # Verify schema
    assert table1["schema_version"] == "1.0.0"
    assert table1["mode"] == "SHADOW"
    assert "confound_stability_threshold" in table1
    assert table1["confound_stability_threshold"] == 0.95
    
    # Verify 6 rows (3 statuses × 2 stability relations)
    assert len(table1["truth_table"]) == 6
    
    # Verify all combinations are present
    statuses = set()
    stability_relations = set()
    for row in table1["truth_table"]:
        assert "status" in row
        assert "stability_index" in row
        assert "stability_relation" in row
        assert "strict_confounding_result" in row
        assert "legacy_confounding_result" in row
        assert "strict_confound_reason" in row
        assert "legacy_confound_reason" in row
        
        statuses.add(row["status"])
        stability_relations.add(row["stability_relation"])
    
    assert statuses == {"OK", "WARN", "BLOCK"}
    assert stability_relations == {"above_threshold", "below_threshold"}
    
    # Verify determinism (byte-identical JSON)
    import json
    json1 = json.dumps(table1, sort_keys=True)
    json2 = json.dumps(table2, sort_keys=True)
    assert json1 == json2


def test_truth_table_strict_vs_legacy_logic():
    """Test truth table reflects correct strict vs legacy logic."""
    table = build_budget_confounding_truth_table()
    
    # Find OK + below_threshold row
    ok_below = next(
        r for r in table["truth_table"]
        if r["status"] == "OK" and r["stability_relation"] == "below_threshold"
    )
    # Strict: OK alone is NOT confounded
    assert ok_below["strict_confounding_result"] is False
    # Legacy: Low stability alone IS confounded
    assert ok_below["legacy_confounding_result"] is True
    
    # Find WARN + above_threshold row
    warn_above = next(
        r for r in table["truth_table"]
        if r["status"] == "WARN" and r["stability_relation"] == "above_threshold"
    )
    # Strict: WARN alone is NOT confounded
    assert warn_above["strict_confounding_result"] is False
    # Legacy: WARN alone IS confounded
    assert warn_above["legacy_confounding_result"] is True
    
    # Find WARN + below_threshold row
    warn_below = next(
        r for r in table["truth_table"]
        if r["status"] == "WARN" and r["stability_relation"] == "below_threshold"
    )
    # Both strict and legacy: WARN + low stability IS confounded
    assert warn_below["strict_confounding_result"] is True
    assert warn_below["legacy_confounding_result"] is True


def test_validate_defaults_warns_on_missing_reason():
    """Test validator warns when budget_confounded is present but reason is missing."""
    report = {
        "windows": [
            {
                "budget_confounded": True,
                # Missing budget_confound_reason and confound_stability_threshold
            },
            {
                "budget_confounded": False,
                # Missing budget_confound_reason and confound_stability_threshold
            },
        ],
    }
    
    warnings = validate_budget_confounding_defaults(report)
    
    # Each window triggers 2 warnings: missing reason and missing threshold
    assert len(warnings) == 4
    
    # Verify structured format with codes
    assert all("code" in w for w in warnings)
    assert all("message" in w for w in warnings)
    assert all("window_index" in w for w in warnings)
    
    reason_warnings = [w for w in warnings if w["code"] == "BUDGET-DEF-001"]
    threshold_warnings = [w for w in warnings if w["code"] == "BUDGET-DEF-002"]
    assert len(reason_warnings) == 2
    assert len(threshold_warnings) == 2
    
    # Verify messages contain window reference
    assert "window[0]" in reason_warnings[0]["message"]
    assert "window[1]" in reason_warnings[1]["message"]


def test_validate_defaults_warns_on_missing_threshold():
    """Test validator warns when budget fields are present but threshold is missing."""
    report = {
        "windows": [
            {
                "budget_combined_status": "OK",
                "budget_stability_index": 0.98,
                # Missing confound_stability_threshold
            },
        ],
    }
    
    warnings = validate_budget_confounding_defaults(report)
    
    assert len(warnings) == 1
    assert warnings[0]["code"] == "BUDGET-DEF-002"
    assert "window[0]" in warnings[0]["message"]
    assert "confound_stability_threshold missing" in warnings[0]["message"]
    assert warnings[0]["window_index"] == 0


def test_validate_defaults_no_warnings_for_complete_annotation():
    """Test validator produces no warnings for complete annotations."""
    report = {
        "windows": [
            {
                "budget_confounded": True,
                "budget_confound_reason": "STATUS_AND_LOW_STABILITY",
                "confound_stability_threshold": 0.95,
            },
            {
                "budget_confounded": False,
                "budget_confound_reason": None,
                "confound_stability_threshold": 0.95,
            },
        ],
    }
    
    warnings = validate_budget_confounding_defaults(report)
    
    assert len(warnings) == 0


def test_validate_defaults_handles_empty_windows():
    """Test validator handles empty windows list."""
    report = {"windows": []}
    
    warnings = validate_budget_confounding_defaults(report)
    
    assert len(warnings) == 0


def test_validate_defaults_warning_codes_stable_and_deterministic():
    """Test warning codes are stable and list order is deterministic."""
    report = {
        "windows": [
            {
                "budget_confounded": True,  # Missing reason
                "budget_combined_status": "OK",  # Has budget fields, missing threshold
            },
            {
                "budget_confounded": False,  # Missing reason
            },
        ],
    }
    
    # Run multiple times to verify determinism
    warnings1 = validate_budget_confounding_defaults(report)
    warnings2 = validate_budget_confounding_defaults(report)
    warnings3 = validate_budget_confounding_defaults(report)
    
    # All runs should produce identical results
    assert warnings1 == warnings2 == warnings3
    
    # Verify codes are stable
    codes = [w["code"] for w in warnings1]
    assert "BUDGET-DEF-001" in codes
    assert "BUDGET-DEF-002" in codes
    
    # Verify deterministic ordering by window_index
    window_indices = [w["window_index"] for w in warnings1]
    assert window_indices == sorted(window_indices)
    
    # Verify JSON serialization is deterministic
    import json
    json1 = json.dumps(warnings1, sort_keys=True)
    json2 = json.dumps(warnings2, sort_keys=True)
    assert json1 == json2



