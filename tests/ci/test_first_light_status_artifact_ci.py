import json
import os
from pathlib import Path

import pytest

from tools.ci.first_light_status_artifact_contract import (
    build_first_light_status_artifact_contract_report,
    validate_first_light_status_artifact_file,
)


def _fixture_path() -> Path:
    return Path(__file__).resolve().parents[2] / "testsfixtures" / "first_light_status_sample.json"


def _load_fixture() -> dict:
    return json.loads(_fixture_path().read_text(encoding="utf-8"))


def test_ci_first_light_status_artifact_matches_contract() -> None:
    artifact_path = os.environ.get("FIRST_LIGHT_STATUS_ARTIFACT_PATH")
    if not artifact_path:
        pytest.skip("CI-only: FIRST_LIGHT_STATUS_ARTIFACT_PATH not set")

    status_path = Path(artifact_path)
    result = validate_first_light_status_artifact_file(
        status_path=status_path,
        fixture_path=_fixture_path(),
    )
    assert result.ok, "Status artifact contract violations:\n" + "\n".join(
        [f"reason_codes={result.reason_codes}", *result.errors]
    )


def test_old_schema_fails_with_reason_code(tmp_path: Path) -> None:
    status = _load_fixture()
    status["schema_version"] = "1.1.9"

    status_path = tmp_path / "first_light_status.json"
    status_path.write_text(json.dumps(status), encoding="utf-8")

    result = validate_first_light_status_artifact_file(
        status_path=status_path,
        fixture_path=_fixture_path(),
        allowed_schema_versions=[">=1.2.0"],
    )
    assert not result.ok
    assert "SCHEMA_VERSION_TOO_OLD" in result.reason_codes


def test_unknown_telemetry_source_fails_with_reason_code(tmp_path: Path) -> None:
    status = _load_fixture()
    status["telemetry_source"] = "unknown_source"

    status_path = tmp_path / "first_light_status.json"
    status_path.write_text(json.dumps(status), encoding="utf-8")

    result = validate_first_light_status_artifact_file(
        status_path=status_path,
        fixture_path=_fixture_path(),
        allowed_schema_versions=[">=1.2.0"],
    )
    assert not result.ok
    assert "TELEMETRY_SOURCE_INVALID" in result.reason_codes


def test_success_passes_contract(tmp_path: Path) -> None:
    status = _load_fixture()
    status["schema_version"] = "1.5.0"
    status["telemetry_source"] = "real_synthetic"

    status_path = tmp_path / "first_light_status.json"
    status_path.write_text(json.dumps(status), encoding="utf-8")

    result = validate_first_light_status_artifact_file(
        status_path=status_path,
        fixture_path=_fixture_path(),
        allowed_schema_versions=[">=1.2.0"],
    )
    assert result.ok
    assert result.reason_codes == []


def test_reason_codes_report_is_stable_and_truncated(tmp_path: Path) -> None:
    status = _load_fixture()
    status.pop("shadow_mode_ok", None)
    status["schema_version"] = "1.1.9"
    status["telemetry_source"] = "unknown_source"
    status["proof_snapshot_present"] = "nope"
    status["p5_divergence_baseline"] = {"telemetry_source": "unknown_source"}

    status_path = tmp_path / "first_light_status.json"
    status_path.write_text(json.dumps(status), encoding="utf-8")

    result_a = validate_first_light_status_artifact_file(
        status_path=status_path,
        fixture_path=_fixture_path(),
        allowed_schema_versions=[">=1.2.0"],
    )
    result_b = validate_first_light_status_artifact_file(
        status_path=status_path,
        fixture_path=_fixture_path(),
        allowed_schema_versions=[">=1.2.0"],
    )

    report_a = build_first_light_status_artifact_contract_report(result_a, top_n=3)
    report_b = build_first_light_status_artifact_contract_report(result_b, top_n=3)

    assert report_a == report_b
    assert list(report_a.keys()) == ["passed", "reason_codes_topN", "details"]
    assert report_a["passed"] is False

    expected_full_codes = [
        "KEY_TYPE_MISMATCH",
        "MISSING_REQUIRED_KEY",
        "P5_BASELINE_TELEMETRY_SOURCE_INVALID",
        "SCHEMA_VERSION_TOO_OLD",
        "TELEMETRY_SOURCE_INVALID",
    ]
    assert report_a["details"]["reason_codes"] == expected_full_codes
    assert report_a["reason_codes_topN"] == expected_full_codes[:3]
