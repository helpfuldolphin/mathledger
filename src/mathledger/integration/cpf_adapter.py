"""Wave A6 CPF adapter: deterministic normalization for experiment contexts."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping, Sequence


CPF_SCHEMA_VERSION = "v1"
INDICATOR_ID_PATTERN = re.compile(r"^\d+\.\d+$")
MATURITY_LEVELS = frozenset({"low", "medium", "high", "critical"})


class CPFAdapterViolation(ValueError):
    """Raised when CPF assessment payloads violate normalization contract."""

    ERROR_CODE = "CPF_ADAPTER_VIOLATION"

    def __init__(
        self,
        message: str,
        *,
        field: str | None = None,
        index: int | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.field = field
        self.index = index
        self.details = dict(details or {})

    def to_error_response(self) -> dict[str, Any]:
        return {
            "error_code": self.ERROR_CODE,
            "message": str(self),
            "field": self.field,
            "index": self.index,
            "details": self.details,
        }


def _canonicalize_json(data: Mapping[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _content_hash(data: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonicalize_json(data).encode("utf-8")).hexdigest()


def _risk_band(score: float) -> str:
    if score >= 0.85:
        return "critical"
    if score >= 0.66:
        return "high"
    if score >= 0.33:
        return "medium"
    return "low"


def _normalize_assessment(assessment: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    if not isinstance(assessment, Mapping):
        raise CPFAdapterViolation("Assessment must be a mapping.", field="assessments", index=index)

    indicator_id = str(assessment.get("indicator_id", "")).strip()
    if not indicator_id:
        raise CPFAdapterViolation(
            "Assessment requires non-empty indicator_id.",
            field="indicator_id",
            index=index,
        )
    if not INDICATOR_ID_PATTERN.match(indicator_id):
        raise CPFAdapterViolation(
            "indicator_id must match '<domain>.<indicator>' numeric format (e.g., '7.2').",
            field="indicator_id",
            index=index,
        )

    try:
        bayesian_score = float(assessment.get("bayesian_score"))
    except (TypeError, ValueError) as exc:
        raise CPFAdapterViolation(
            "bayesian_score must be numeric in [0.0, 1.0].",
            field="bayesian_score",
            index=index,
        ) from exc
    if bayesian_score < 0.0 or bayesian_score > 1.0:
        raise CPFAdapterViolation(
            "bayesian_score must be in [0.0, 1.0].",
            field="bayesian_score",
            index=index,
        )

    confidence_raw = assessment.get("confidence", 0.0)
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError) as exc:
        raise CPFAdapterViolation(
            "confidence must be numeric in [0.0, 1.0].",
            field="confidence",
            index=index,
        ) from exc
    if confidence < 0.0 or confidence > 1.0:
        raise CPFAdapterViolation(
            "confidence must be in [0.0, 1.0].",
            field="confidence",
            index=index,
        )

    maturity_raw = assessment.get("maturity_level")
    if maturity_raw is None:
        maturity_level = _risk_band(bayesian_score)
    else:
        maturity_level = str(maturity_raw).strip().lower()
        if maturity_level not in MATURITY_LEVELS:
            raise CPFAdapterViolation(
                "maturity_level must be one of low|medium|high|critical.",
                field="maturity_level",
                index=index,
            )

    category = assessment.get("category")
    category_str = "" if category is None else str(category).strip()

    assessor = str(assessment.get("assessor", "cpf")).strip() or "cpf"
    assessment_date = str(assessment.get("assessment_date", "")).strip()

    source = str(assessment.get("source", "cpf")).strip() or "cpf"
    event_count_raw = assessment.get("event_count", 0)
    try:
        event_count = int(event_count_raw)
    except (TypeError, ValueError) as exc:
        raise CPFAdapterViolation(
            "event_count must be an integer >= 0.",
            field="event_count",
            index=index,
        ) from exc
    if event_count < 0:
        raise CPFAdapterViolation(
            "event_count must be >= 0.",
            field="event_count",
            index=index,
        )

    raw_data = assessment.get("raw_data")
    if isinstance(raw_data, Mapping):
        source = str(raw_data.get("source", source)).strip() or source
        if "event_count" in raw_data:
            try:
                event_count = int(raw_data.get("event_count", 0))
            except (TypeError, ValueError) as exc:
                raise CPFAdapterViolation(
                    "raw_data.event_count must be an integer >= 0.",
                    field="raw_data.event_count",
                    index=index,
                ) from exc
            if event_count < 0:
                raise CPFAdapterViolation(
                    "raw_data.event_count must be >= 0.",
                    field="raw_data.event_count",
                    index=index,
                )

    return {
        "indicator_id": indicator_id,
        "category": category_str,
        "bayesian_score": round(bayesian_score, 6),
        "confidence": round(confidence, 6),
        "maturity_level": maturity_level,
        "assessor": assessor,
        "assessment_date": assessment_date,
        "source": source,
        "event_count": event_count,
    }


def normalize_cpf_assessments(
    assessments: Sequence[Mapping[str, Any]],
    *,
    snapshot_epoch: int,
    schema_version: str = CPF_SCHEMA_VERSION,
) -> dict[str, Any]:
    """
    Normalize CPF assessment outputs into deterministic snapshot artifact.
    """
    if schema_version != CPF_SCHEMA_VERSION:
        raise CPFAdapterViolation(
            f"Unsupported schema_version: {schema_version!r}.",
            field="schema_version",
        )
    if isinstance(assessments, (str, bytes)):
        raise CPFAdapterViolation("assessments must be a sequence.", field="assessments")
    if snapshot_epoch < 0:
        raise CPFAdapterViolation("snapshot_epoch must be non-negative.", field="snapshot_epoch")

    normalized = [
        _normalize_assessment(assessment, index=idx)
        for idx, assessment in enumerate(assessments)
    ]
    normalized.sort(key=lambda item: item["indicator_id"])

    indicator_count = len(normalized)
    if indicator_count:
        mean_score = round(sum(item["bayesian_score"] for item in normalized) / indicator_count, 6)
        max_score = max(item["bayesian_score"] for item in normalized)
    else:
        mean_score = 0.0
        max_score = 0.0

    top_indicators = sorted(
        (
            {"indicator_id": item["indicator_id"], "bayesian_score": item["bayesian_score"]}
            for item in normalized
        ),
        key=lambda item: (-item["bayesian_score"], item["indicator_id"]),
    )[:5]

    payload_for_hash = {
        "schema_version": schema_version,
        "snapshot_epoch": snapshot_epoch,
        "indicators": normalized,
    }
    snapshot_id = _content_hash(payload_for_hash)

    return {
        "schema_version": schema_version,
        "snapshot_id": snapshot_id,
        "snapshot_epoch": snapshot_epoch,
        "indicator_count": indicator_count,
        "aggregate": {
            "mean_bayesian_score": mean_score,
            "max_bayesian_score": round(max_score, 6),
            "risk_band": _risk_band(max_score),
        },
        "top_indicators": top_indicators,
        "indicators": normalized,
    }


def cpf_snapshot_json(snapshot: Mapping[str, Any]) -> str:
    """Canonical compact serialization for deterministic CPF snapshots."""
    return _canonicalize_json(snapshot)


__all__ = [
    "CPF_SCHEMA_VERSION",
    "CPFAdapterViolation",
    "normalize_cpf_assessments",
    "cpf_snapshot_json",
]
