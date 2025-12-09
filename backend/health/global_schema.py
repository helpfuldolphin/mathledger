"""
Schema definition and validation for ``global_health.json`` payloads.

The schema is intentionally small but strict; the surface is a contract that
summarizes global FM health for downstream dashboards.  Invariants are enforced
so that consumers can rely on the ``status`` and ``fm_ok`` flags matching the
derived semantics described in ``scripts/fm_canonicalize.py``.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

from .tda_adapter import validate_tda_health_tile

GLOBAL_HEALTH_SCHEMA_VERSION = "1.0.0"

ALLOWED_ALIGNMENT_STATUSES = {"WELL_DISTRIBUTED", "CONCENTRATED", "SPARSE"}
ALLOWED_STATUS_VALUES = {"OK", "WARN", "BLOCK"}


class SchemaValidationError(ValueError):
    """Raised when an input payload violates the global health schema."""


def _ensure_bool(field: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    raise SchemaValidationError(f"{field} must be a boolean, got {type(value)!r}")


def _ensure_float(field: str, value: Any) -> float:
    if isinstance(value, bool):
        raise SchemaValidationError(f"{field} must be a number, got bool")
    if isinstance(value, (int, float)):
        return float(value)
    raise SchemaValidationError(f"{field} must be numeric, got {type(value)!r}")


def _ensure_int(field: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaValidationError(f"{field} must be an integer, got {type(value)!r}")
    return value


def _ensure_str(field: str, value: Any) -> str:
    if not isinstance(value, str):
        raise SchemaValidationError(f"{field} must be a string, got {type(value)!r}")
    return value


def validate_global_health(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize a ``global_health`` payload.

    The returned dictionary contains the canonicalized fields.  No mutation is
    applied to the caller's mapping.  The schema enforces the invariants used by
    :func:`scripts.fm_canonicalize.summarize_fm_for_global_health`.
    """
    normalized: Dict[str, Any] = dict(payload)

    schema_version = normalized.get("schema_version", GLOBAL_HEALTH_SCHEMA_VERSION)
    schema_version = _ensure_str("schema_version", schema_version)
    if schema_version != GLOBAL_HEALTH_SCHEMA_VERSION:
        raise SchemaValidationError(
            f"schema_version must be {GLOBAL_HEALTH_SCHEMA_VERSION}, "
            f"got {schema_version}"
        )
    normalized["schema_version"] = schema_version

    fm_ok = _ensure_bool("fm_ok", normalized.get("fm_ok"))
    coverage_pct = _ensure_float("coverage_pct", normalized.get("coverage_pct"))
    external_only = _ensure_int(
        "external_only_labels", normalized.get("external_only_labels")
    )
    if external_only < 0:
        raise SchemaValidationError("external_only_labels cannot be negative")

    alignment_status = _ensure_str(
        "alignment_status", normalized.get("alignment_status")
    )
    if alignment_status not in ALLOWED_ALIGNMENT_STATUSES:
        raise SchemaValidationError(
            f"alignment_status must be one of {sorted(ALLOWED_ALIGNMENT_STATUSES)}, "
            f"got {alignment_status}"
        )

    status = _ensure_str("status", normalized.get("status"))
    if status not in ALLOWED_STATUS_VALUES:
        raise SchemaValidationError(
            f"status must be one of {sorted(ALLOWED_STATUS_VALUES)}, got {status}"
        )

    if status == "OK":
        if not fm_ok:
            raise SchemaValidationError("status OK requires fm_ok=True")
        if external_only != 0:
            raise SchemaValidationError(
                "status OK requires external_only_labels == 0"
            )
    else:
        if fm_ok:
            raise SchemaValidationError("fm_ok must be False when status is not OK")

    if external_only > 0 and status != "BLOCK":
        raise SchemaValidationError(
            "external_only_labels > 0 requires status == 'BLOCK'"
        )
    if not 0.0 <= coverage_pct <= 100.0:
        raise SchemaValidationError("coverage_pct must be between 0 and 100")

    tda_section = normalized.get("tda")
    if tda_section is not None:
        if not isinstance(tda_section, Mapping):
            raise SchemaValidationError("tda section must be a JSON object")
        try:
            normalized["tda"] = validate_tda_health_tile(tda_section)
        except ValueError as exc:
            raise SchemaValidationError(f"TDA tile invalid: {exc}") from exc
    else:
        normalized.pop("tda", None)

    normalized.update(
        {
            "fm_ok": fm_ok,
            "coverage_pct": coverage_pct,
            "external_only_labels": external_only,
            "alignment_status": alignment_status,
            "status": status,
        }
    )

    return normalized
