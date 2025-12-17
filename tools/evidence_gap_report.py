import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Mapping

# Design note (backfill stub):
# A future backfill orchestrator can consume generate_evidence_gap_report(root) to
# decide which generators to run. If P3 artifacts are missing, it would schedule
# the P3 synthetic harness; if P4 artifacts are missing, it would run the P4
# shadow harness; if visualizations/proof snapshots are missing, it would trigger
# the plotter or proof hash snapshot job. The report remains read-only and
# advisory; orchestration should be a separate, explicit step.

# Visualization (plots) artifacts for the First Light evidence pack.
# These are optional for gating but should be backfilled when missing.
VISUALIZATION_FILES: List[str] = [
    "visualizations/delta_p_vs_cycles.svg",
    "visualizations/rsi_vs_cycles.svg",
    "visualizations/omega_occupancy_vs_cycles.svg",
]

# Optional compliance artifacts for the First Light evidence pack.
PROOF_SNAPSHOT_FILE = "compliance/proof_log_snapshot.json"

OPTIONAL_ARTIFACTS = set(VISUALIZATION_FILES) | {PROOF_SNAPSHOT_FILE}

DIGEST_SCHEMA_VERSION = "evidence-gap-digest/1.0"

# Required artifacts for the First Light evidence pack.
# Note: This list includes optional-but-desired artifacts to enable backfill planning.
REQUIRED_FILES: List[str] = [
    # Pack-level
    "manifest.json",
    # P3 (synthetic) artifacts
    "p3_synthetic/synthetic_raw.jsonl",
    "p3_synthetic/stability_report.json",
    "p3_synthetic/red_flag_matrix.json",
    "p3_synthetic/metrics_windows.json",
    "p3_synthetic/tda_metrics.json",
    # P4 (shadow) artifacts
    "p4_shadow/real_cycles.jsonl",
    "p4_shadow/twin_predictions.jsonl",
    "p4_shadow/divergence_log.jsonl",
    "p4_shadow/p4_summary.json",
    "p4_shadow/twin_accuracy.json",
    "p4_shadow/tda_metrics.json",
    # Visualizations (plots)
    *VISUALIZATION_FILES,
    # Compliance
    "compliance/compliance_narrative.md",
    PROOF_SNAPSHOT_FILE,
]


def generate_evidence_gap_report(root: Path | str, required: Iterable[str] | None = None) -> Mapping[str, List[str]]:
    """
    Scan the given root directory and report which required evidence files are present or missing.
    """
    root_path = Path(root)
    required_list = list(required) if required is not None else REQUIRED_FILES

    present: List[str] = []
    missing: List[str] = []

    for filename in required_list:
        target = root_path / filename
        if target.is_file():
            present.append(filename)
        else:
            missing.append(filename)

    return {"present": present, "missing": missing}


def write_gap_status(report: Mapping[str, List[str]], output_path: Path | str) -> None:
    """
    Persist a small status summary derived from the gap report.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "present_count": len(report.get("present", [])),
        "missing_count": len(report.get("missing", [])),
        "missing": sorted(report.get("missing", [])),
    }
    output.write_text(json.dumps(summary, indent=2))


def _default_ascii_only() -> bool:
    return sys.platform == "win32"


def _ascii_safe_markdown(text: str) -> str:
    return text.encode("ascii", errors="backslashreplace").decode("ascii")


def _generated_at(*, deterministic: bool) -> str:
    if deterministic:
        return "1970-01-01T00:00:00Z"
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def render_gap_digest_markdown(
    report: Mapping[str, List[str]],
    plan: Mapping[str, List[Mapping[str, object]]] | None = None,
    *,
    ascii_only: bool | None = None,
    deterministic: bool = False,
) -> str:
    if ascii_only is None:
        ascii_only = _default_ascii_only()

    missing_required = [
        path
        for path in report.get("missing", [])
        if path not in OPTIONAL_ARTIFACTS
    ]

    if plan is None:
        plan = generate_job_plan(report)

    jobs = list(plan.get("jobs", []))
    optional_jobs = [
        job
        for job in jobs
        if set(job.get("expected_outputs", [])) <= OPTIONAL_ARTIFACTS
    ]

    header = {
        "schema_version": DIGEST_SCHEMA_VERSION,
        "mode": "SHADOW",
        "generated_at": _generated_at(deterministic=deterministic),
    }
    header_json = json.dumps(
        header,
        indent=2,
        ensure_ascii=ascii_only,
        sort_keys=False,
    )

    lines = ["```json", header_json, "```", "", "# Evidence Gap Digest", "", "## Missing required"]
    if missing_required:
        lines.extend([f"- `{path}`" for path in missing_required])
    else:
        lines.append("- (none)")

    lines.extend(["", "## Optional jobs planned (ordered)"])
    if optional_jobs:
        for job in optional_jobs:
            outputs = job.get("expected_outputs", [])
            outputs_str = ", ".join(f"`{path}`" for path in outputs)
            lines.append(
                f"- `{job.get('job_id')}` (`{job.get('action')}`) -> {outputs_str}"
            )
    else:
        lines.append("- (none)")

    digest = "\n".join(lines) + "\n"
    return _ascii_safe_markdown(digest) if ascii_only else digest


def write_markdown_digest(markdown: str, output_path: Path | str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    normalized = markdown.replace("\r\n", "\n").replace("\r", "\n")
    output.write_text(normalized, encoding="utf-8", newline="\n")


def generate_job_plan(report: Mapping[str, List[str]]) -> Mapping[str, List[Mapping[str, object]]]:
    """
    Build a deterministic job plan from the gap report.

    Rules:
    - Any missing P3 artifacts => schedule a P3 harness job.
    - Any missing P4 artifacts => schedule a P4 harness job.
    - Missing proof snapshot => schedule proof log snapshot generation.
    - Missing visualizations (.svg) => schedule plots generation.
    - Missing manifest => schedule manifest rebuild.
    - Missing compliance narrative => flag as manual draft.

    Priority order (lower is earlier):
    10 P3 harness
    20 P4 harness
    25 proof snapshot (preferred over plots)
    26 plots
    30 manifest rebuild
    40 compliance narrative draft
    """
    missing = set(report.get("missing", []))

    p3_outputs = [
        "p3_synthetic/synthetic_raw.jsonl",
        "p3_synthetic/stability_report.json",
        "p3_synthetic/red_flag_matrix.json",
        "p3_synthetic/metrics_windows.json",
        "p3_synthetic/tda_metrics.json",
    ]
    p4_outputs = [
        "p4_shadow/real_cycles.jsonl",
        "p4_shadow/twin_predictions.jsonl",
        "p4_shadow/divergence_log.jsonl",
        "p4_shadow/p4_summary.json",
        "p4_shadow/twin_accuracy.json",
        "p4_shadow/tda_metrics.json",
    ]
    viz_outputs = VISUALIZATION_FILES

    jobs = []

    def add_job(job_id: str, action: str, expected_outputs: List[str], priority: int, inputs: List[str] | None = None) -> None:
        jobs.append(
            {
                "job_id": job_id,
                "action": action,
                "inputs": inputs or [],
                "expected_outputs": expected_outputs,
                "priority": priority,
            }
        )

    if missing.intersection(p3_outputs):
        add_job(
            job_id="job_p3_harness",
            action="run_p3_harness",
            inputs=["p3_run_config.json"],
            expected_outputs=p3_outputs,
            priority=10,
        )

    if missing.intersection(p4_outputs):
        add_job(
            job_id="job_p4_harness",
            action="run_p4_harness",
            inputs=["p4_run_config.json"],
            expected_outputs=p4_outputs,
            priority=20,
        )

    if PROOF_SNAPSHOT_FILE in missing:
        add_job(
            job_id="job_proof_snapshot",
            action="run_proof_log_snapshot",
            inputs=["proof_log.jsonl"],
            expected_outputs=[PROOF_SNAPSHOT_FILE],
            priority=25,
        )

    if missing.intersection(viz_outputs):
        add_job(
            job_id="job_first_light_plots",
            action="run_first_light_plots",
            inputs=["p3_synthetic/", "p4_shadow/"],
            expected_outputs=viz_outputs,
            priority=26,
        )

    if "manifest.json" in missing:
        add_job(
            job_id="job_build_manifest",
            action="run_build_first_light_evidence_pack",
            inputs=["p3_synthetic/", "p4_shadow/"],
            expected_outputs=["manifest.json"],
            priority=30,
        )

    if "compliance/compliance_narrative.md" in missing:
        add_job(
            job_id="job_compliance_narrative",
            action="draft_compliance_narrative",
            inputs=["compliance_requirements.md"],
            expected_outputs=["compliance/compliance_narrative.md"],
            priority=40,
        )

    # Deterministic ordering by priority then job_id.
    jobs.sort(key=lambda j: (j["priority"], j["job_id"]))
    return {"jobs": jobs}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report evidence gaps for the Pre-Launch Review manifest."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=None,
        help="Root directory to scan (default: current directory).",
    )
    parser.add_argument(
        "--root",
        dest="root_option",
        default=None,
        help="Root directory to scan (overrides positional argument).",
    )
    parser.add_argument(
        "--status-json",
        dest="status_json",
        help="Optional path to write a gap_status.json summary.",
    )
    parser.add_argument(
        "--plan-output",
        dest="plan_output",
        help="Path to write a backfill job plan JSON derived from gaps (no execution).",
    )
    parser.add_argument(
        "--markdown-output",
        dest="markdown_output",
        help="Path to write a short markdown digest (missing required + optional jobs).",
    )
    ascii_group = parser.add_mutually_exclusive_group()
    ascii_group.add_argument(
        "--ascii-only",
        dest="ascii_only",
        action="store_true",
        help="Escape non-ASCII in markdown output (default: enabled on Windows).",
    )
    ascii_group.add_argument(
        "--no-ascii-only",
        dest="ascii_only",
        action="store_false",
        help="Allow non-ASCII characters in markdown output (UTF-8).",
    )
    parser.set_defaults(ascii_only=None)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Emit deterministic markdown digest header (stable generated_at).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = args.root_option or args.root or "."
    report = generate_evidence_gap_report(root)
    print(json.dumps(report, indent=2))
    plan = None
    if args.status_json:
        write_gap_status(report, args.status_json)
    if args.plan_output:
        plan = plan or generate_job_plan(report)
        Path(args.plan_output).write_text(json.dumps(plan, indent=2))
    if args.markdown_output:
        plan = plan or generate_job_plan(report)
        digest = render_gap_digest_markdown(
            report,
            plan,
            ascii_only=args.ascii_only,
            deterministic=args.deterministic,
        )
        write_markdown_digest(digest, args.markdown_output)


if __name__ == "__main__":
    main()
