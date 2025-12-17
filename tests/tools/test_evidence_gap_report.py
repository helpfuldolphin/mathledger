import json

from tools.evidence_gap_report import REQUIRED_FILES, generate_evidence_gap_report


def test_evidence_gap_report_identifies_present_and_missing(tmp_path):
    # Create a partial evidence pack with a subset of required files.
    (tmp_path / "manifest.json").write_text("{}\n")
    (tmp_path / "p3_synthetic").mkdir()
    (tmp_path / "p4_shadow").mkdir()
    (tmp_path / "compliance").mkdir()

    (tmp_path / "p3_synthetic" / "synthetic_raw.jsonl").write_text("stub\n")
    (tmp_path / "p3_synthetic" / "stability_report.json").write_text("{}\n")
    (tmp_path / "p4_shadow" / "divergence_log.jsonl").write_text("stub\n")
    (tmp_path / "p4_shadow" / "twin_predictions.jsonl").write_text("stub\n")
    (tmp_path / "compliance" / "compliance_narrative.md").write_text("# narrative\n")

    report = generate_evidence_gap_report(tmp_path)

    assert report["present"] == [
        "manifest.json",
        "p3_synthetic/synthetic_raw.jsonl",
        "p3_synthetic/stability_report.json",
        "p4_shadow/twin_predictions.jsonl",
        "p4_shadow/divergence_log.jsonl",
        "compliance/compliance_narrative.md",
    ]
    assert report["missing"] == [
        "p3_synthetic/red_flag_matrix.json",
        "p3_synthetic/metrics_windows.json",
        "p3_synthetic/tda_metrics.json",
        "p4_shadow/real_cycles.jsonl",
        "p4_shadow/p4_summary.json",
        "p4_shadow/twin_accuracy.json",
        "p4_shadow/tda_metrics.json",
        "visualizations/delta_p_vs_cycles.svg",
        "visualizations/rsi_vs_cycles.svg",
        "visualizations/omega_occupancy_vs_cycles.svg",
        "compliance/proof_log_snapshot.json",
    ]

    # Ensure the required list stays in sync with the report keys.
    assert sorted(report["present"] + report["missing"]) == sorted(REQUIRED_FILES)


def test_write_gap_status(tmp_path):
    report = {
        "present": ["manifest.json", "p3_synthetic/synthetic_raw.jsonl"],
        "missing": ["p4_shadow/divergence_log.jsonl", "compliance/compliance_narrative.md"],
    }
    status_path = tmp_path / "gap_status.json"

    # Write status summary and verify contents.
    from tools.evidence_gap_report import write_gap_status

    write_gap_status(report, status_path)

    payload = status_path.read_text()
    assert '"present_count": 2' in payload
    assert '"missing_count": 2' in payload
    assert '"missing": [\n    "compliance/compliance_narrative.md",\n    "p4_shadow/divergence_log.jsonl"\n  ]' in payload


def test_cli_status_json(tmp_path):
    status_path = tmp_path / "gap_status.json"

    import sys
    from tools import evidence_gap_report

    original_argv = sys.argv[:]
    sys.argv = [
        "evidence_gap_report",
        "--root",
        str(tmp_path),
        "--status-json",
        str(status_path),
    ]
    try:
        evidence_gap_report.main()
    finally:
        sys.argv = original_argv

    assert status_path.exists(), "CLI should write the status JSON"
    data = json.loads(status_path.read_text())
    assert data["present_count"] == 0
    assert data["missing_count"] == len(REQUIRED_FILES)
    # Missing list should be sorted.
    assert data["missing"] == sorted(REQUIRED_FILES)


def test_plan_output(tmp_path):
    # Create a partial pack: only P3 files exist.
    (tmp_path / "p3_synthetic").mkdir()
    (tmp_path / "p3_synthetic" / "synthetic_raw.jsonl").write_text("stub\n")
    (tmp_path / "p3_synthetic" / "stability_report.json").write_text("{}\n")
    (tmp_path / "p3_synthetic" / "red_flag_matrix.json").write_text("{}\n")
    (tmp_path / "p3_synthetic" / "metrics_windows.json").write_text("{}\n")
    (tmp_path / "p3_synthetic" / "tda_metrics.json").write_text("{}\n")

    import sys
    from tools import evidence_gap_report

    plan_path = tmp_path / "plan.json"
    original_argv = sys.argv[:]
    sys.argv = [
        "evidence_gap_report",
        "--root",
        str(tmp_path),
        "--plan-output",
        str(plan_path),
    ]
    try:
        evidence_gap_report.main()
    finally:
        sys.argv = original_argv

    assert plan_path.exists(), "Plan output should be written"
    plan = json.loads(plan_path.read_text())
    jobs = plan["jobs"]
    # With P3 complete, we expect P4 + manifest + compliance jobs.
    expected_job_ids = [
        "job_p4_harness",
        "job_proof_snapshot",
        "job_first_light_plots",
        "job_build_manifest",
        "job_compliance_narrative",
    ]
    assert [job["job_id"] for job in jobs] == expected_job_ids
    assert jobs[0]["action"] == "run_p4_harness"
    assert "p4_shadow/p4_summary.json" in jobs[0]["expected_outputs"]


def test_generate_job_plan_orders_proof_snapshot_before_plots() -> None:
    from tools.evidence_gap_report import (
        PROOF_SNAPSHOT_FILE,
        VISUALIZATION_FILES,
        generate_job_plan,
    )

    report = {"present": [], "missing": [VISUALIZATION_FILES[0], PROOF_SNAPSHOT_FILE]}
    plan = generate_job_plan(report)
    jobs = plan["jobs"]

    assert [job["job_id"] for job in jobs] == [
        "job_proof_snapshot",
        "job_first_light_plots",
    ]
    assert [job["action"] for job in jobs] == [
        "run_proof_log_snapshot",
        "run_first_light_plots",
    ]
    assert jobs[1]["expected_outputs"] == VISUALIZATION_FILES


def test_cli_markdown_output_is_stable(tmp_path):
    from tools.evidence_gap_report import OPTIONAL_ARTIFACTS, REQUIRED_FILES

    # Create a pack with all required files except the manifest,
    # leaving optional artifacts missing so optional jobs are planned.
    for rel_path in REQUIRED_FILES:
        if rel_path in OPTIONAL_ARTIFACTS or rel_path == "manifest.json":
            continue
        path = tmp_path / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stub\n", encoding="utf-8")

    markdown_path = tmp_path / "digest.md"

    import sys
    from tools import evidence_gap_report

    original_argv = sys.argv[:]
    sys.argv = [
        "evidence_gap_report",
        "--root",
        str(tmp_path),
        "--markdown-output",
        str(markdown_path),
        "--deterministic",
    ]
    try:
        evidence_gap_report.main()
    finally:
        sys.argv = original_argv

    expected = (
        "```json\n"
        "{\n"
        '  "schema_version": "evidence-gap-digest/1.0",\n'
        '  "mode": "SHADOW",\n'
        '  "generated_at": "1970-01-01T00:00:00Z"\n'
        "}\n"
        "```\n"
        "\n"
        "# Evidence Gap Digest\n"
        "\n"
        "## Missing required\n"
        "- `manifest.json`\n"
        "\n"
        "## Optional jobs planned (ordered)\n"
        "- `job_proof_snapshot` (`run_proof_log_snapshot`) -> `compliance/proof_log_snapshot.json`\n"
        "- `job_first_light_plots` (`run_first_light_plots`) -> `visualizations/delta_p_vs_cycles.svg`, `visualizations/rsi_vs_cycles.svg`, `visualizations/omega_occupancy_vs_cycles.svg`\n"
    )
    assert markdown_path.read_bytes() == expected.encode("utf-8")
    assert b"\r" not in markdown_path.read_bytes()


def test_render_gap_digest_markdown_ascii_only_toggle() -> None:
    from tools.evidence_gap_report import render_gap_digest_markdown

    report = {"present": [], "missing": ["p3_synthetic/\N{GREEK CAPITAL LETTER DELTA}.json"]}
    plan = {"jobs": []}

    ascii_digest = render_gap_digest_markdown(
        report,
        plan,
        ascii_only=True,
        deterministic=True,
    )
    ascii_digest.encode("ascii")
    assert "\\u0394" in ascii_digest

    utf8_digest = render_gap_digest_markdown(
        report,
        plan,
        ascii_only=False,
        deterministic=True,
    )
    assert "\N{GREEK CAPITAL LETTER DELTA}" in utf8_digest


def test_render_gap_digest_markdown_default_ascii_only_windows() -> None:
    import sys

    from tools.evidence_gap_report import render_gap_digest_markdown

    report = {"present": [], "missing": ["p3_synthetic/\N{GREEK CAPITAL LETTER DELTA}.json"]}
    plan = {"jobs": []}

    digest = render_gap_digest_markdown(report, plan, ascii_only=None, deterministic=True)
    if sys.platform == "win32":
        digest.encode("ascii")
        assert "\\u0394" in digest
    else:
        assert "\N{GREEK CAPITAL LETTER DELTA}" in digest


def test_write_markdown_digest_normalizes_newlines(tmp_path) -> None:
    from tools.evidence_gap_report import write_markdown_digest

    out_path = tmp_path / "digest.md"
    write_markdown_digest("line1\r\nline2\r\n", out_path)
    assert out_path.read_bytes() == b"line1\nline2\n"
