"""Psychological governance gates for scenario-level decisioning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


RISK_GREEN_MAX = 0.50
RISK_RED_MIN = 0.70
INDICATOR_ELEVATION_THRESHOLD = 0.60
CONVERGENCE_RED_MIN = 0.25
CONVERGENCE_RED_MIN_CATEGORIES = 3
CONVERGENCE_YELLOW_MIN = 0.20
CONVERGENCE_YELLOW_MIN_CATEGORIES = 2
TOTAL_CPF_CATEGORIES = 10
GOVERNANCE_DECISIONS = frozenset({"GREEN", "YELLOW", "RED"})


class PsychologicalGateViolation(ValueError):
    """Raised when psychological gate evaluation inputs are malformed."""

    ERROR_CODE = "PSYCHOLOGICAL_GATE_VIOLATION"

    def __init__(
        self,
        message: str,
        *,
        field: str | None = None,
        index: int | None = None,
    ) -> None:
        super().__init__(message)
        self.field = field
        self.index = index

    def to_error_response(self) -> dict[str, object]:
        return {
            "error_code": self.ERROR_CODE,
            "message": str(self),
            "field": self.field,
            "index": self.index,
        }


@dataclass(frozen=True)
class GovernanceGateEvaluation:
    """Result of applying risk + convergence gates to CPF snapshot indicators."""

    risk_score: float
    risk_gate: str
    convergence_score: float
    convergence_gate: str
    categories_elevated: tuple[str, ...]
    indicators_elevated: tuple[str, ...]
    overall_decision: str

    def to_dict(self) -> dict[str, object]:
        return {
            "risk_score": self.risk_score,
            "risk_gate": self.risk_gate,
            "convergence_score": self.convergence_score,
            "convergence_gate": self.convergence_gate,
            "categories_elevated": list(self.categories_elevated),
            "indicators_elevated": list(self.indicators_elevated),
            "overall_decision": self.overall_decision,
        }


def _normalize_indicator_category(indicator: Mapping[str, object], *, index: int) -> str:
    indicator_id = str(indicator.get("indicator_id", "")).strip()
    if not indicator_id:
        raise PsychologicalGateViolation(
            "Indicator missing indicator_id.",
            field="indicator_id",
            index=index,
        )
    if "." in indicator_id:
        return indicator_id.split(".", 1)[0]
    category = str(indicator.get("category", "")).strip()
    if not category:
        raise PsychologicalGateViolation(
            "Indicator missing category fallback for non-numeric indicator_id.",
            field="category",
            index=index,
        )
    return category.lower()


def _normalize_indicators(
    indicators: Sequence[Mapping[str, object]],
) -> tuple[tuple[str, float, str], ...]:
    normalized: list[tuple[str, float, str]] = []
    for idx, indicator in enumerate(indicators):
        if not isinstance(indicator, Mapping):
            raise PsychologicalGateViolation(
                "Indicator entries must be mappings.",
                field="indicators",
                index=idx,
            )
        indicator_id = str(indicator.get("indicator_id", "")).strip()
        if not indicator_id:
            raise PsychologicalGateViolation(
                "Indicator missing indicator_id.",
                field="indicator_id",
                index=idx,
            )
        try:
            score = float(indicator.get("bayesian_score"))
        except (TypeError, ValueError) as exc:
            raise PsychologicalGateViolation(
                "Indicator bayesian_score must be numeric.",
                field="bayesian_score",
                index=idx,
            ) from exc
        if score < 0.0 or score > 1.0:
            raise PsychologicalGateViolation(
                "Indicator bayesian_score must be in [0.0, 1.0].",
                field="bayesian_score",
                index=idx,
            )
        category = _normalize_indicator_category(indicator, index=idx)
        normalized.append((indicator_id, score, category))
    normalized.sort(key=lambda item: item[0])
    return tuple(normalized)


def _risk_gate(score: float) -> str:
    if score >= RISK_RED_MIN:
        return "RED"
    if score >= RISK_GREEN_MAX:
        return "YELLOW"
    return "GREEN"


def _convergence_gate(convergence_score: float, categories_count: int) -> str:
    if (
        convergence_score >= CONVERGENCE_RED_MIN
        and categories_count >= CONVERGENCE_RED_MIN_CATEGORIES
    ):
        return "RED"
    if (
        convergence_score >= CONVERGENCE_YELLOW_MIN
        and categories_count >= CONVERGENCE_YELLOW_MIN_CATEGORIES
    ):
        return "YELLOW"
    return "GREEN"


def evaluate_psychological_governance(
    indicators: Sequence[Mapping[str, object]],
) -> GovernanceGateEvaluation:
    """
    Evaluate risk and convergence governance gates from CPF indicator set.

    Rules mirror the architecture described in `cpf_mathledger_integration_v2.tex`:
    - Risk gate: GREEN (<0.50), YELLOW (0.50-<0.70), RED (>=0.70)
    - Convergence gate: elevated categories at threshold 0.60
      with RED escalation at >=3 categories (CS1 >= 0.25)
    """
    normalized = _normalize_indicators(indicators)
    if normalized:
        risk_score = max(score for _, score, _ in normalized)
    else:
        risk_score = 0.0
    risk_gate = _risk_gate(risk_score)

    elevated = [(indicator_id, category) for indicator_id, score, category in normalized if score >= INDICATOR_ELEVATION_THRESHOLD]
    categories_elevated = tuple(sorted({category for _, category in elevated}))
    indicators_elevated = tuple(sorted(indicator_id for indicator_id, _ in elevated))

    convergence_score = round(len(categories_elevated) / float(TOTAL_CPF_CATEGORIES), 6)
    convergence_gate = _convergence_gate(convergence_score, len(categories_elevated))

    if "RED" in {risk_gate, convergence_gate}:
        overall = "RED"
    elif "YELLOW" in {risk_gate, convergence_gate}:
        overall = "YELLOW"
    else:
        overall = "GREEN"

    return GovernanceGateEvaluation(
        risk_score=round(risk_score, 6),
        risk_gate=risk_gate,
        convergence_score=convergence_score,
        convergence_gate=convergence_gate,
        categories_elevated=categories_elevated,
        indicators_elevated=indicators_elevated,
        overall_decision=overall,
    )


__all__ = [
    "RISK_GREEN_MAX",
    "RISK_RED_MIN",
    "INDICATOR_ELEVATION_THRESHOLD",
    "CONVERGENCE_RED_MIN",
    "CONVERGENCE_RED_MIN_CATEGORIES",
    "CONVERGENCE_YELLOW_MIN",
    "CONVERGENCE_YELLOW_MIN_CATEGORIES",
    "TOTAL_CPF_CATEGORIES",
    "GOVERNANCE_DECISIONS",
    "PsychologicalGateViolation",
    "GovernanceGateEvaluation",
    "evaluate_psychological_governance",
]

