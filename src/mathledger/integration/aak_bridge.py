"""Wave A5 bridge: Lane-B AAK references to Lane-A MathLedger roots."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from mathledger.evidence import assert_evidence_pack_passes


BRIDGE_SCHEMA_VERSION = "v1"
ALLOWED_SOURCES = frozenset({"captured", "synthetic"})
ALLOWED_CAPTURE_MODES = frozenset({"hash_ref", "inline_minimal"})
ALLOWED_PSYCH_CONTEXT_KEYS = frozenset(
    {
        "source",
        "capture_mode",
        "snapshot_id",
        "convergence_score",
        "elevated_categories",
        "elevated_indicators",
        "artifact_path",
        "artifact_hash",
        "psych_root_hash",
        "reasoning_root_ref",
        "ui_root_ref",
    }
)
FORBIDDEN_AUTHORITY_KEYS = frozenset(
    {
        "trust_class",
        "validation_outcome",
        "claim_id",
        "committed_partition_id",
        "authority_verdict",
        "verification_route",
    }
)


class AAKBridgeViolation(ValueError):
    """Raised when AAK bridge payloads violate Lane-A/Lane-B separation."""

    ERROR_CODE = "AAK_BRIDGE_VIOLATION"

    def __init__(
        self,
        message: str,
        *,
        field: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.field = field
        self.details = dict(details or {})

    def to_error_response(self) -> dict[str, Any]:
        return {
            "error_code": self.ERROR_CODE,
            "message": str(self),
            "field": self.field,
            "details": self.details,
        }


def _canonicalize_json(data: Mapping[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _content_hash(data: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonicalize_json(data).encode("utf-8")).hexdigest()


def _require_hex_digest(value: str | None, *, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise AAKBridgeViolation(f"{field} must be a 64-char lowercase hex digest.", field=field)
    return value


def _normalize_categories(categories: Sequence[str] | None) -> list[str]:
    if categories is None:
        return []
    normalized = sorted({str(category).strip() for category in categories if str(category).strip()})
    return normalized


def _normalize_indicators(indicators: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    if indicators is None:
        return []

    normalized: list[dict[str, Any]] = []
    for idx, indicator in enumerate(indicators):
        if not isinstance(indicator, Mapping):
            raise AAKBridgeViolation(
                "elevated_indicators entries must be mappings.",
                field=f"elevated_indicators[{idx}]",
            )

        indicator_id = str(indicator.get("indicator_id", "")).strip()
        if not indicator_id:
            raise AAKBridgeViolation(
                "elevated_indicators entries require indicator_id.",
                field=f"elevated_indicators[{idx}].indicator_id",
            )

        try:
            activation_level = float(indicator.get("activation_level", 0.0))
        except (TypeError, ValueError) as exc:
            raise AAKBridgeViolation(
                "activation_level must be numeric in [0, 100].",
                field=f"elevated_indicators[{idx}].activation_level",
            ) from exc
        if activation_level < 0.0 or activation_level > 100.0:
            raise AAKBridgeViolation(
                "activation_level must be within [0, 100].",
                field=f"elevated_indicators[{idx}].activation_level",
            )

        category = indicator.get("category")
        normalized.append(
            {
                "indicator_id": indicator_id,
                "activation_level": round(activation_level, 6),
                "category": None if category is None else str(category),
            }
        )

    normalized.sort(key=lambda item: (item["indicator_id"], item["activation_level"]))
    return normalized


def _assert_safe_artifact_path(path: str) -> None:
    parsed = Path(path)
    if parsed.is_absolute() or ".." in parsed.parts or parsed.parts[:1] != ("psych",):
        raise AAKBridgeViolation(
            "artifact_path must be a safe relative path under psych/.",
            field="artifact_path",
        )


def _assert_no_authority_escalation(payload: Any) -> None:
    stack: list[Any] = [payload]
    while stack:
        current = stack.pop()
        if isinstance(current, Mapping):
            for key, value in current.items():
                if key in FORBIDDEN_AUTHORITY_KEYS:
                    raise AAKBridgeViolation(
                        "Lane-B payload contains forbidden authority-bearing key.",
                        field=str(key),
                    )
                stack.append(value)
        elif isinstance(current, (list, tuple)):
            stack.extend(current)


def build_lane_b_psych_context(
    *,
    evidence_pack: Mapping[str, Any],
    source: str = "captured",
    capture_mode: str = "hash_ref",
    snapshot_id: str | None = None,
    convergence_score: float | None = None,
    elevated_categories: Sequence[str] | None = None,
    elevated_indicators: Sequence[Mapping[str, Any]] | None = None,
    artifact_path: str | None = None,
    artifact_hash: str | None = None,
    psych_root_hash: str | None = None,
) -> dict[str, Any]:
    """
    Build AAK-compatible psych_context that references Lane-A roots without authority transfer.
    """
    verification = assert_evidence_pack_passes(evidence_pack)
    lane_a_roots = verification["recorded"]

    source_norm = str(source).strip().lower()
    capture_mode_norm = str(capture_mode).strip().lower()
    if source_norm not in ALLOWED_SOURCES:
        raise AAKBridgeViolation("source must be one of captured|synthetic.", field="source")
    if capture_mode_norm not in ALLOWED_CAPTURE_MODES:
        raise AAKBridgeViolation(
            "capture_mode must be one of hash_ref|inline_minimal.",
            field="capture_mode",
        )

    convergence_norm: float | None = None
    if convergence_score is not None:
        try:
            convergence_norm = float(convergence_score)
        except (TypeError, ValueError) as exc:
            raise AAKBridgeViolation(
                "convergence_score must be numeric in [0.0, 1.0].",
                field="convergence_score",
            ) from exc
        if convergence_norm < 0.0 or convergence_norm > 1.0:
            raise AAKBridgeViolation(
                "convergence_score must be in [0.0, 1.0].",
                field="convergence_score",
            )

    if (artifact_path is None) != (artifact_hash is None):
        raise AAKBridgeViolation(
            "artifact_path and artifact_hash must be provided together.",
            field="artifact_path",
        )
    if artifact_path is not None:
        _assert_safe_artifact_path(artifact_path)

    psych_hash = _require_hex_digest(psych_root_hash, field="psych_root_hash")
    art_hash = _require_hex_digest(artifact_hash, field="artifact_hash")
    categories = _normalize_categories(elevated_categories)
    indicators = _normalize_indicators(elevated_indicators)

    payload: dict[str, Any] = {
        "source": source_norm,
        "capture_mode": capture_mode_norm,
        "reasoning_root_ref": lane_a_roots["r_t"],
        "ui_root_ref": lane_a_roots["u_t"],
    }
    if snapshot_id is not None:
        snapshot_norm = str(snapshot_id).strip()
        if not snapshot_norm:
            raise AAKBridgeViolation("snapshot_id cannot be empty.", field="snapshot_id")
        payload["snapshot_id"] = snapshot_norm
    if convergence_norm is not None:
        payload["convergence_score"] = round(convergence_norm, 6)
    if categories:
        payload["elevated_categories"] = categories
    if indicators:
        payload["elevated_indicators"] = indicators
    if artifact_path is not None:
        payload["artifact_path"] = artifact_path
        payload["artifact_hash"] = art_hash
    if psych_hash is not None:
        payload["psych_root_hash"] = psych_hash

    _assert_no_authority_escalation(payload)
    return payload


def validate_lane_b_reference(
    psych_context: Mapping[str, Any],
    *,
    evidence_pack: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Validate a Lane-B psych_context payload against Lane-A evidence roots.
    """
    if not isinstance(psych_context, Mapping):
        raise AAKBridgeViolation("psych_context must be a mapping.", field="psych_context")

    unknown = set(psych_context.keys()) - set(ALLOWED_PSYCH_CONTEXT_KEYS)
    if unknown:
        raise AAKBridgeViolation(
            f"psych_context has unexpected keys: {sorted(unknown)}",
            field="psych_context",
        )

    if "reasoning_root_ref" not in psych_context or "ui_root_ref" not in psych_context:
        raise AAKBridgeViolation(
            "psych_context must include reasoning_root_ref and ui_root_ref.",
            field="psych_context",
        )

    normalized = build_lane_b_psych_context(
        evidence_pack=evidence_pack,
        source=str(psych_context.get("source", "captured")),
        capture_mode=str(psych_context.get("capture_mode", "hash_ref")),
        snapshot_id=psych_context.get("snapshot_id"),
        convergence_score=psych_context.get("convergence_score"),
        elevated_categories=psych_context.get("elevated_categories"),
        elevated_indicators=psych_context.get("elevated_indicators"),
        artifact_path=psych_context.get("artifact_path"),
        artifact_hash=psych_context.get("artifact_hash"),
        psych_root_hash=psych_context.get("psych_root_hash"),
    )

    expected = assert_evidence_pack_passes(evidence_pack)["recorded"]
    provided_reasoning_ref = str(psych_context.get("reasoning_root_ref"))
    provided_ui_ref = str(psych_context.get("ui_root_ref"))
    if provided_reasoning_ref != expected["r_t"]:
        raise AAKBridgeViolation(
            "reasoning_root_ref mismatch with Lane-A evidence root.",
            field="reasoning_root_ref",
        )
    if provided_ui_ref != expected["u_t"]:
        raise AAKBridgeViolation(
            "ui_root_ref mismatch with Lane-A evidence root.",
            field="ui_root_ref",
        )
    if normalized["reasoning_root_ref"] != expected["r_t"]:
        raise AAKBridgeViolation(
            "reasoning_root_ref mismatch with Lane-A evidence root.",
            field="reasoning_root_ref",
        )
    if normalized["ui_root_ref"] != expected["u_t"]:
        raise AAKBridgeViolation(
            "ui_root_ref mismatch with Lane-A evidence root.",
            field="ui_root_ref",
        )

    _assert_no_authority_escalation(normalized)
    return normalized


def build_aak_bridge_packet(
    *,
    evidence_pack: Mapping[str, Any],
    psych_context: Mapping[str, Any] | None = None,
    **psych_context_kwargs: Any,
) -> dict[str, Any]:
    """
    Build deterministic Lane-B bridge packet with explicit Lane-A root references.
    """
    verification = assert_evidence_pack_passes(evidence_pack)
    roots = verification["recorded"]

    if psych_context is None:
        context = build_lane_b_psych_context(
            evidence_pack=evidence_pack,
            **psych_context_kwargs,
        )
    else:
        context = validate_lane_b_reference(psych_context, evidence_pack=evidence_pack)

    packet_payload = {
        "bridge_schema_version": BRIDGE_SCHEMA_VERSION,
        "lane": "B",
        "mathledger_refs": {
            "u_t": roots["u_t"],
            "r_t": roots["r_t"],
            "h_t": roots["h_t"],
        },
        "psych_context": context,
    }
    packet_payload["lane_b_reference_id"] = _content_hash(
        {
            "mathledger_refs": packet_payload["mathledger_refs"],
            "psych_context": packet_payload["psych_context"],
        }
    )
    return packet_payload


__all__ = [
    "AAKBridgeViolation",
    "build_lane_b_psych_context",
    "validate_lane_b_reference",
    "build_aak_bridge_packet",
]
