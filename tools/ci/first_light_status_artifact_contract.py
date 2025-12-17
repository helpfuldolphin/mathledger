from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

ALLOWED_TELEMETRY_SOURCES = {"mock", "real_synthetic", "real_trace"}
DEFAULT_REASON_CODES_TOP_N = 5

_SEMVER_RE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
_VERSION_CONSTRAINT_RE = re.compile(
    r"^\s*(?P<op>(==|>=|<=|>|<))?\s*(?P<ver>(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*))\s*$"
)


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: list[str]
    reason_codes: list[str]


@dataclass(frozen=True)
class _VersionConstraint:
    op: str
    version: tuple[int, int, int]


def _parse_semver(version: str) -> tuple[int, int, int]:
    match = _SEMVER_RE.match(version.strip())
    if not match:
        raise ValueError(f"not a semver: {version!r}")
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object, got {type(value).__name__}")
    return value


def _parse_version_constraint(token: str) -> _VersionConstraint:
    match = _VERSION_CONSTRAINT_RE.match(token)
    if not match:
        raise ValueError(f"invalid version constraint: {token!r}")
    op = match.group("op") or "=="
    version = _parse_semver(match.group("ver"))
    return _VersionConstraint(op=op, version=version)


def _parse_allowed_schema_versions(allowed: list[str]) -> list[list[_VersionConstraint]]:
    rules: list[list[_VersionConstraint]] = []
    for raw_rule in allowed:
        parts = [p.strip() for p in raw_rule.split(",") if p.strip()]
        if not parts:
            raise ValueError("empty schema version rule")
        rules.append([_parse_version_constraint(part) for part in parts])
    return rules


def _version_satisfies_constraint(
    version: tuple[int, int, int], constraint: _VersionConstraint
) -> bool:
    if constraint.op == "==":
        return version == constraint.version
    if constraint.op == ">=":
        return version >= constraint.version
    if constraint.op == "<=":
        return version <= constraint.version
    if constraint.op == ">":
        return version > constraint.version
    if constraint.op == "<":
        return version < constraint.version
    raise ValueError(f"unsupported operator: {constraint.op!r}")


def _version_allowed(version: tuple[int, int, int], allowed: list[str]) -> bool:
    rules = _parse_allowed_schema_versions(allowed)
    return any(
        all(_version_satisfies_constraint(version, constraint) for constraint in rule)
        for rule in rules
    )


def _min_allowed_version(allowed: list[str]) -> tuple[int, int, int] | None:
    rules = _parse_allowed_schema_versions(allowed)
    rule_mins: list[tuple[int, int, int]] = []
    for rule in rules:
        lower_bounds = [
            c.version for c in rule if c.op in {"==", ">=", ">"}
        ]
        if lower_bounds:
            rule_mins.append(max(lower_bounds))
    return min(rule_mins) if rule_mins else None


def _default_allowed_schema_versions(fixture: Mapping[str, Any]) -> list[str]:
    fixture_schema_version = fixture.get("schema_version")
    if isinstance(fixture_schema_version, str):
        try:
            _parse_semver(fixture_schema_version)
        except ValueError:
            return [">=1.2.0"]
        return [f">={fixture_schema_version}"]
    return [">=1.2.0"]


def build_first_light_status_artifact_contract_report(
    result: ValidationResult, *, top_n: int = DEFAULT_REASON_CODES_TOP_N
) -> dict[str, Any]:
    canonical_reason_codes = sorted(set(result.reason_codes))
    report: dict[str, Any] = {
        "passed": result.ok,
        "reason_codes_topN": canonical_reason_codes[: max(0, top_n)],
        "details": {
            "reason_codes": canonical_reason_codes,
            "errors": result.errors,
        },
    }
    return report


def validate_first_light_status_against_fixture(
    *,
    status: Mapping[str, Any],
    fixture: Mapping[str, Any],
    allowed_schema_versions: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    reason_codes: list[str] = []

    def add_error(code: str, message: str) -> None:
        reason_codes.append(code)
        errors.append(message)

    required_keys = set(fixture.keys())
    missing_keys = sorted(required_keys - set(status.keys()))
    if missing_keys:
        add_error("MISSING_REQUIRED_KEY", f"Missing required top-level keys: {missing_keys}")

    for key in sorted(required_keys & set(status.keys())):
        fixture_value = fixture.get(key)
        status_value = status.get(key)
        if isinstance(fixture_value, bool) and not isinstance(status_value, bool):
            add_error(
                "KEY_TYPE_MISMATCH",
                f"Key '{key}' expected bool, got {type(status_value).__name__}",
            )
        if isinstance(fixture_value, str) and not isinstance(status_value, str):
            add_error(
                "KEY_TYPE_MISMATCH",
                f"Key '{key}' expected str, got {type(status_value).__name__}",
            )

    allowed_schema_versions = (
        allowed_schema_versions
        if allowed_schema_versions is not None
        else _default_allowed_schema_versions(fixture)
    )
    status_schema_version = status.get("schema_version")
    status_schema_parsed: tuple[int, int, int] | None = None
    if status_schema_version is None:
        add_error("SCHEMA_VERSION_MISSING", "schema_version is missing or null")
    elif not isinstance(status_schema_version, str):
        add_error(
            "SCHEMA_VERSION_TYPE",
            f"schema_version expected str, got {type(status_schema_version).__name__}",
        )
    else:
        try:
            status_schema_parsed = _parse_semver(status_schema_version)
        except ValueError as exc:
            add_error("SCHEMA_VERSION_INVALID", f"Invalid schema_version: {exc}")

    if status_schema_parsed is not None:
        try:
            if not _version_allowed(status_schema_parsed, allowed_schema_versions):
                min_allowed = _min_allowed_version(allowed_schema_versions)
                if min_allowed is not None and status_schema_parsed < min_allowed:
                    add_error(
                        "SCHEMA_VERSION_TOO_OLD",
                        f"schema_version {status_schema_version!r} is older than allowed "
                        f"minimum {_format_semver(min_allowed)!r}",
                    )
                else:
                    add_error(
                        "SCHEMA_VERSION_NOT_ALLOWED",
                        f"schema_version {status_schema_version!r} does not satisfy allowed rules "
                        f"{allowed_schema_versions}",
                    )
        except ValueError as exc:
            add_error(
                "SCHEMA_VERSION_RULES_INVALID",
                f"Invalid allowed schema version rules {allowed_schema_versions}: {exc}",
            )

    telemetry_source = status.get("telemetry_source")
    if telemetry_source is None:
        add_error("TELEMETRY_SOURCE_MISSING", "telemetry_source is missing or null")
    elif not isinstance(telemetry_source, str):
        add_error(
            "TELEMETRY_SOURCE_TYPE",
            f"telemetry_source expected str, got {type(telemetry_source).__name__}",
        )
    elif telemetry_source not in ALLOWED_TELEMETRY_SOURCES:
        add_error(
            "TELEMETRY_SOURCE_INVALID",
            f"telemetry_source {telemetry_source!r} not in allowed taxonomy "
            f"{sorted(ALLOWED_TELEMETRY_SOURCES)}",
        )

    baseline = status.get("p5_divergence_baseline")
    if isinstance(baseline, dict) and "telemetry_source" in baseline:
        baseline_source = baseline.get("telemetry_source")
        if isinstance(baseline_source, str) and baseline_source not in ALLOWED_TELEMETRY_SOURCES:
            add_error(
                "P5_BASELINE_TELEMETRY_SOURCE_INVALID",
                f"p5_divergence_baseline.telemetry_source {baseline_source!r} not in "
                f"allowed taxonomy {sorted(ALLOWED_TELEMETRY_SOURCES)}",
            )

    deduped_codes = list(dict.fromkeys(reason_codes))
    return errors, deduped_codes


def _format_semver(version: tuple[int, int, int]) -> str:
    major, minor, patch = version
    return f"{major}.{minor}.{patch}"


def validate_first_light_status_artifact_file(
    *,
    status_path: Path,
    fixture_path: Path,
    allowed_schema_versions: list[str] | None = None,
) -> ValidationResult:
    errors: list[str] = []
    reason_codes: list[str] = []

    def add_error(code: str, message: str) -> None:
        reason_codes.append(code)
        errors.append(message)

    if not fixture_path.exists():
        add_error("FIXTURE_NOT_FOUND", f"Fixture JSON not found: {fixture_path}")
        return ValidationResult(ok=False, errors=errors, reason_codes=list(dict.fromkeys(reason_codes)))
    if not status_path.exists():
        add_error("STATUS_ARTIFACT_NOT_FOUND", f"Status artifact JSON not found: {status_path}")
        return ValidationResult(ok=False, errors=errors, reason_codes=list(dict.fromkeys(reason_codes)))

    try:
        status_raw = _load_json(status_path)
    except json.JSONDecodeError as exc:
        add_error("STATUS_JSON_INVALID", f"Invalid JSON in status artifact {status_path}: {exc}")
        return ValidationResult(ok=False, errors=errors, reason_codes=list(dict.fromkeys(reason_codes)))

    try:
        fixture_raw = _load_json(fixture_path)
    except json.JSONDecodeError as exc:
        add_error("FIXTURE_JSON_INVALID", f"Invalid JSON in fixture {fixture_path}: {exc}")
        return ValidationResult(ok=False, errors=errors, reason_codes=list(dict.fromkeys(reason_codes)))

    try:
        status = _require_mapping(status_raw, "status artifact")
        fixture = _require_mapping(fixture_raw, "fixture")
    except TypeError as exc:
        add_error("JSON_NOT_OBJECT", str(exc))
        return ValidationResult(ok=False, errors=errors, reason_codes=list(dict.fromkeys(reason_codes)))

    contract_errors, contract_codes = validate_first_light_status_against_fixture(
        status=status,
        fixture=fixture,
        allowed_schema_versions=allowed_schema_versions,
    )
    errors.extend(contract_errors)
    reason_codes.extend(contract_codes)
    deduped_codes = list(dict.fromkeys(reason_codes))
    return ValidationResult(ok=not errors, errors=errors, reason_codes=deduped_codes)
