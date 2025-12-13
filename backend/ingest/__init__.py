"""
AI Proof Ingestion Adapter

Internal pipeline for ingesting externally-generated proofs (AI or other sources).
This module implements Phase 1 of the AI Proof Ingestion Adapter specification.

See: docs/architecture/AI_PROOF_INGESTION_ADAPTER.md

PHASE 1 SCOPE:
- Database provenance recording
- Shadow mode enforcement
- Source type tagging
- Internal pipeline only (no public API)

OUT OF SCOPE (Phase 1):
- Public API endpoint
- Rate limiting
- Graduation logic
- UX/UI
"""

from backend.ingest.pipeline import (
    AIProofSubmission,
    ProvenanceMetadata,
    ingest_ai_proof,
    IngestResult,
)

__all__ = [
    "AIProofSubmission",
    "ProvenanceMetadata",
    "ingest_ai_proof",
    "IngestResult",
]
