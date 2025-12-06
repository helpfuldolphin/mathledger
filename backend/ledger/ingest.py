"""
Deprecated ledger ingest shim.

All functionality has been moved to ledger.ingest.
This module re-exports for backwards compatibility.
"""

import warnings

from ledger.ingest import (  # noqa: F401
    BlockRecord,
    IngestOutcome,
    LedgerIngestor,
    ProofRecord,
    StatementRecord,
)

warnings.warn(
    "backend.ledger.ingest is deprecated; import ledger.ingest instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "BlockRecord",
    "IngestOutcome",
    "LedgerIngestor",
    "ProofRecord",
    "StatementRecord",
]
