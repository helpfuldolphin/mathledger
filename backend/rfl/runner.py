"""Deprecated RFL runner shim.

This module is deprecated. Import from rfl.runner directly instead.
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from rfl.runner import *  # noqa: F401,F403

warnings.warn(
    "backend.rfl.runner is deprecated; import rfl.runner instead.",
    DeprecationWarning,
    stacklevel=2,
)


@dataclass
class AttestationInput:
    """Attestation input for RFL runs.

    Stub class for backwards compatibility.
    """
    slice_id: str
    run_id: Optional[str] = None
    expected_hash: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


__all__ = [
    "RFLRunner",
    "RflResult",
    "RunLedgerEntry",
    "AttestationInput",
]
