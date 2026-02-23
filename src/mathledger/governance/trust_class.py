"""Trust class and outcome definitions for Wave A2 governance kernel."""

from __future__ import annotations

from enum import Enum
from typing import Union


class TrustClass(str, Enum):
    """Trust classes for routing authority-bearing vs advisory claims."""

    FV = "FV"
    MV = "MV"
    PA = "PA"
    ADV = "ADV"


class Outcome(str, Enum):
    """Verification outcomes recorded in authority artifacts."""

    VERIFIED = "VERIFIED"
    REFUTED = "REFUTED"
    ABSTAINED = "ABSTAINED"


AUTHORITY_BEARING_TRUST_CLASSES = frozenset({TrustClass.FV, TrustClass.MV, TrustClass.PA})


def parse_trust_class(value: Union[str, TrustClass]) -> TrustClass:
    """Parse and normalize trust class value."""
    if isinstance(value, TrustClass):
        return value
    if not isinstance(value, str):
        raise ValueError(f"Trust class must be a string or TrustClass, got: {type(value)!r}")
    normalized = value.strip().upper()
    try:
        return TrustClass(normalized)
    except ValueError as exc:
        raise ValueError(
            f"Invalid trust class: {value!r}. Must be one of FV, MV, PA, ADV."
        ) from exc


def is_authority_bearing(value: Union[str, TrustClass]) -> bool:
    """Return True for FV/MV/PA and False for ADV."""
    return parse_trust_class(value) in AUTHORITY_BEARING_TRUST_CLASSES

