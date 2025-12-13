-- Migration: 018_ai_proof_ingestion
-- Author: Claude Code
-- Date: 2025-12-13
-- Purpose: Add schema support for AI Proof Ingestion Adapter (Phase 1)
-- Spec: docs/architecture/AI_PROOF_INGESTION_ADAPTER.md

BEGIN;

-- =============================================================================
-- 1. Add source_type to proofs table
-- =============================================================================
-- Distinguishes internal proofs from external AI submissions
-- Default 'internal' preserves existing proof semantics

ALTER TABLE proofs
ADD COLUMN IF NOT EXISTS source_type TEXT NOT NULL DEFAULT 'internal';

COMMENT ON COLUMN proofs.source_type IS
  'Origin of proof: internal (axiom engine), external_ai (AI submission)';

-- =============================================================================
-- 2. Add shadow_mode flag to proofs table
-- =============================================================================
-- Shadow mode proofs are recorded but do not advance slice progression
-- or trigger governance enforcement

ALTER TABLE proofs
ADD COLUMN IF NOT EXISTS shadow_mode BOOLEAN NOT NULL DEFAULT false;

COMMENT ON COLUMN proofs.shadow_mode IS
  'Shadow mode proofs are observational only - no governance enforcement';

-- =============================================================================
-- 3. Create proof_provenance table for external proof metadata
-- =============================================================================
-- Tracks full provenance chain for AI-submitted proofs
-- Required fields: source_id, raw_output_hash
-- Optional fields: submitter_attestation, metadata

CREATE TABLE IF NOT EXISTS proof_provenance (
  id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  proof_id              UUID NOT NULL REFERENCES proofs(id) ON DELETE CASCADE,
  source_type           TEXT NOT NULL,
  source_id             TEXT NOT NULL,
  submission_id         UUID NOT NULL UNIQUE,
  raw_output_hash       TEXT NOT NULL,
  submitter_attestation TEXT,
  metadata              JSONB,
  created_at            TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE proof_provenance IS
  'Provenance chain for externally-submitted proofs (AI or other sources)';

COMMENT ON COLUMN proof_provenance.source_id IS
  'Identifier for the proof source (e.g., gpt-4-turbo-2025-01)';

COMMENT ON COLUMN proof_provenance.raw_output_hash IS
  'SHA-256 hash of the raw AI output before any processing';

COMMENT ON COLUMN proof_provenance.submission_id IS
  'Unique identifier for this submission (for idempotency)';

-- =============================================================================
-- 4. Indexes for provenance queries
-- =============================================================================

CREATE INDEX IF NOT EXISTS idx_provenance_source
  ON proof_provenance(source_id);

CREATE INDEX IF NOT EXISTS idx_provenance_submission
  ON proof_provenance(submission_id);

CREATE INDEX IF NOT EXISTS idx_proofs_source_type
  ON proofs(source_type);

CREATE INDEX IF NOT EXISTS idx_proofs_shadow_mode
  ON proofs(shadow_mode) WHERE shadow_mode = true;

-- =============================================================================
-- 5. Constraint: shadow_mode required for external_ai
-- =============================================================================
-- This constraint enforces the mandatory shadow mode policy for Phase 1.
-- Graduation will be handled by application logic, not database constraint.
-- This ensures we never accidentally bypass shadow mode.

ALTER TABLE proofs
ADD CONSTRAINT chk_external_ai_requires_shadow
CHECK (source_type != 'external_ai' OR shadow_mode = true);

COMMIT;
