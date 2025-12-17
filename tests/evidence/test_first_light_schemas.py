from pathlib import Path

from tools.evidence_schema_check import validate_file_against_schema


REPO_ROOT = Path(__file__).resolve().parents[2]

# Short harness runs keep the JSON/JSONL payloads small for local validation.
P3_SAMPLE_RUN = REPO_ROOT / "results" / "test_harness" / "fl_20251211_042514_seed42"
P4_SAMPLE_RUN = REPO_ROOT / "results" / "test_p4" / "p4_20251211_043607"

SYNTHETIC_SCHEMA = REPO_ROOT / "schemas" / "evidence" / "first_light_synthetic_raw.schema.json"
RED_FLAG_SCHEMA = REPO_ROOT / "schemas" / "evidence" / "first_light_red_flag_matrix.schema.json"
DIVERGENCE_SCHEMA = REPO_ROOT / "schemas" / "evidence" / "p4_divergence_log.schema.json"


def test_first_light_synthetic_raw_matches_schema():
    payload = P3_SAMPLE_RUN / "synthetic_raw.jsonl"
    assert payload.exists()

    assert validate_file_against_schema(payload, SYNTHETIC_SCHEMA) is True


def test_first_light_red_flag_matrix_matches_schema():
    payload = P3_SAMPLE_RUN / "red_flag_matrix.json"
    assert payload.exists()

    assert validate_file_against_schema(payload, RED_FLAG_SCHEMA) is True


def test_p4_divergence_log_matches_schema():
    payload = P4_SAMPLE_RUN / "divergence_log.jsonl"
    assert payload.exists()

    assert validate_file_against_schema(payload, DIVERGENCE_SCHEMA) is True


def test_schema_validation_helper_sets_flag_true():
    from scripts.generate_first_light_status import (
        DEFAULT_SCHEMA_ROOT,
        validate_schema_artifacts,
    )

    schemas_ok, warnings, report = validate_schema_artifacts(
        P3_SAMPLE_RUN.parent, P4_SAMPLE_RUN.parent, schema_root=DEFAULT_SCHEMA_ROOT
    )

    assert schemas_ok is True
    assert warnings == []
    labels = [result["label"] for result in report["results"]]
    assert labels == [
        "P3 synthetic_raw.jsonl",
        "P3 red_flag_matrix.json",
        "P4 divergence_log.jsonl",
    ]


def test_schema_validation_warns_on_missing_schema(tmp_path):
    from scripts.generate_first_light_status import validate_schema_artifacts
    import shutil

    schema_root = tmp_path / "schemas"
    schema_root.mkdir(parents=True)
    shutil.copy(SYNTHETIC_SCHEMA, schema_root / SYNTHETIC_SCHEMA.name)
    shutil.copy(RED_FLAG_SCHEMA, schema_root / RED_FLAG_SCHEMA.name)

    schemas_ok, warnings, report = validate_schema_artifacts(
        P3_SAMPLE_RUN.parent, P4_SAMPLE_RUN.parent, schema_root=schema_root
    )

    assert schemas_ok is False
    fail_count = sum(result["status"] == "fail" for result in report["results"])
    missing_count = sum(
        result["status"] in {"missing_payload", "missing_schema"} for result in report["results"]
    )
    assert warnings == [f"Schema validation issues: fail={fail_count} missing={missing_count}"]
    statuses = {result["label"]: result["status"] for result in report["results"]}
    assert statuses["P4 divergence_log.jsonl"] == "missing_schema"


def test_build_schemas_ok_summary_orders_and_counts_failures():
    from scripts.generate_first_light_status import build_schemas_ok_summary

    report = {
        "schema_root": "schemas/evidence",
        "runs": {"p3": "p3_run", "p4": "p4_run"},
        "schemas_ok": False,
        "results": [
            {"label": "Zeta artifact", "schema": "schemas/evidence/zeta.schema.json", "status": "fail", "errors": ["Bad zeta"]},
            {"label": "Alpha artifact", "schema": "schemas/evidence/alpha.schema.json", "status": "missing_schema", "errors": ["Missing schema alpha"]},
            {"label": "Beta artifact", "schema": "schemas/evidence/beta.schema.json", "status": "missing_payload", "errors": ["Missing payload beta"]},
            {"label": "Epsilon artifact", "schema": "schemas/evidence/epsilon.schema.json", "status": "fail", "errors": ["Bad epsilon"]},
            {"label": "Delta artifact", "schema": "schemas/evidence/delta.schema.json", "status": "missing_schema", "reason_code": "SCHEMA_ROOT_NOT_FOUND", "errors": ["Missing schema delta"]},
            {"label": "Gamma artifact", "schema": "schemas/evidence/gamma.schema.json", "status": "fail", "errors": ["Bad gamma"]},
            {"label": "Pass artifact", "schema": "schemas/evidence/pass.schema.json", "status": "pass", "errors": []},
        ],
    }

    summary = build_schemas_ok_summary(report, extraction_source="REPORT_FILE")

    assert summary == {
        "extraction_source": "REPORT_FILE",
        "pass": 1,
        "fail": 3,
        "missing": 3,
        "top_reason_code": "SCHEMA_VALIDATION_FAILED",
        "top_failures": [
            {
                "artifact": "Alpha artifact",
                "reason_code": "MISSING_SCHEMA",
                "schema_path": "schemas/evidence/alpha.schema.json",
                "note": "Schema file missing; sync schemas or set --schema-root.",
            },
            {
                "artifact": "Beta artifact",
                "reason_code": "MISSING_PAYLOAD",
                "schema_path": "schemas/evidence/beta.schema.json",
                "note": "Payload missing; re-run harness or check run directory.",
            },
            {
                "artifact": "Delta artifact",
                "reason_code": "SCHEMA_ROOT_NOT_FOUND",
                "schema_path": "schemas/evidence/delta.schema.json",
                "note": "Schema root missing; provide correct --schema-root or restore schemas.",
            },
            {
                "artifact": "Epsilon artifact",
                "reason_code": "SCHEMA_VALIDATION_FAILED",
                "schema_path": "schemas/evidence/epsilon.schema.json",
                "note": "Payload violates schema; inspect report errors for drift.",
            },
            {
                "artifact": "Gamma artifact",
                "reason_code": "SCHEMA_VALIDATION_FAILED",
                "schema_path": "schemas/evidence/gamma.schema.json",
                "note": "Payload violates schema; inspect report errors for drift.",
            },
        ],
    }


def test_schema_validation_reports_schema_root_not_found(tmp_path):
    from scripts.generate_first_light_status import validate_schema_artifacts

    schema_root = tmp_path / "missing_schemas_root"
    schemas_ok, warnings, report = validate_schema_artifacts(
        P3_SAMPLE_RUN.parent, P4_SAMPLE_RUN.parent, schema_root=schema_root
    )

    assert schemas_ok is False
    assert warnings == ["Schema validation issues: fail=0 missing=3"]
    reason_codes = {result["label"]: result.get("reason_code") for result in report["results"]}
    assert set(reason_codes.values()) == {"SCHEMA_ROOT_NOT_FOUND"}


def test_build_schemas_ok_summary_top_reason_code_tie_breaks_by_code():
    from scripts.generate_first_light_status import build_schemas_ok_summary

    report = {
        "schema_root": "schemas/evidence",
        "runs": {"p3": "p3_run", "p4": "p4_run"},
        "schemas_ok": False,
        "results": [
            {"label": "A", "schema": "schemas/evidence/a.schema.json", "status": "missing_schema", "errors": ["missing schema"]},
            {"label": "B", "schema": "schemas/evidence/b.schema.json", "status": "missing_schema", "errors": ["missing schema"]},
            {"label": "C", "schema": "schemas/evidence/c.schema.json", "status": "missing_payload", "errors": ["missing payload"]},
            {"label": "D", "schema": "schemas/evidence/d.schema.json", "status": "missing_payload", "errors": ["missing payload"]},
        ],
    }

    summary = build_schemas_ok_summary(report, extraction_source="REPORT_FILE")
    assert summary["top_reason_code"] == "MISSING_PAYLOAD"


def test_build_schemas_ok_summary_extraction_source_missing_when_no_report():
    from scripts.generate_first_light_status import build_schemas_ok_summary

    summary = build_schemas_ok_summary(None, extraction_source="MISSING")
    assert summary == {
        "extraction_source": "MISSING",
        "pass": 0,
        "fail": 0,
        "missing": 0,
        "top_reason_code": None,
        "top_failures": [],
    }
