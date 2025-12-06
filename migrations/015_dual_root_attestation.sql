-- migrations/015_dual_root_attestation.sql
-- Mirror Auditor: Dual-Root Attestation Symmetry
-- Generated: 2025-11-04 by Claude N (Mirror Auditor)
--
-- Purpose: Extend blocks table to support dual-root attestation (R_t ↔ U_t)
-- ensuring cryptographic binding of reasoning events and human events.
--
-- Design Principles:
-- 1. Idempotent: Safe to run multiple times
-- 2. Backward Compatible: Nullable columns for existing blocks
-- 3. Dual-Root Symmetry: R_t (reasoning) + U_t (UI) → H_t (composite)
-- 4. Epistemic Integrity: Every block can attest to both proof and human lineage

-- ============================================================================
-- DUAL-ROOT ATTESTATION COLUMNS
-- ============================================================================

-- Add reasoning_merkle_root (R_t): Merkle root of proof/reasoning events
ALTER TABLE blocks
ADD COLUMN IF NOT EXISTS reasoning_merkle_root TEXT;

-- Add ui_merkle_root (U_t): Merkle root of UI/human interaction events
ALTER TABLE blocks
ADD COLUMN IF NOT EXISTS ui_merkle_root TEXT;

-- Add composite_attestation_root (H_t): SHA256(R_t || U_t)
-- This binds both roots cryptographically for dual attestation
ALTER TABLE blocks
ADD COLUMN IF NOT EXISTS composite_attestation_root TEXT;

-- Add attestation metadata for audit trails
ALTER TABLE blocks
ADD COLUMN IF NOT EXISTS attestation_metadata JSONB DEFAULT '{}'::jsonb;

-- ============================================================================
-- INDEXES FOR MIRROR AUDITOR QUERIES
-- ============================================================================

-- Index for reasoning root lookups (proof lineage)
CREATE INDEX IF NOT EXISTS blocks_reasoning_merkle_root_idx
ON blocks(reasoning_merkle_root)
WHERE reasoning_merkle_root IS NOT NULL;

-- Index for UI root lookups (human event lineage)
CREATE INDEX IF NOT EXISTS blocks_ui_merkle_root_idx
ON blocks(ui_merkle_root)
WHERE ui_merkle_root IS NOT NULL;

-- Index for composite attestation verification
CREATE INDEX IF NOT EXISTS blocks_composite_attestation_root_idx
ON blocks(composite_attestation_root)
WHERE composite_attestation_root IS NOT NULL;

-- ============================================================================
-- CONSTRAINTS
-- ============================================================================

-- Constraint: If composite_attestation_root is set, both R_t and U_t must exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'blocks_composite_requires_dual_roots'
    ) THEN
        ALTER TABLE blocks ADD CONSTRAINT blocks_composite_requires_dual_roots
        CHECK (
            (composite_attestation_root IS NULL) OR
            (reasoning_merkle_root IS NOT NULL AND ui_merkle_root IS NOT NULL)
        );
    END IF;
END $$;

-- ============================================================================
-- MIGRATION TRACKING
-- ============================================================================

INSERT INTO schema_migrations (version, description)
VALUES ('015_dual_root_attestation', 'Mirror Auditor: Add dual-root attestation columns (R_t, U_t, H_t)')
ON CONFLICT (version) DO NOTHING;

-- ============================================================================
-- COMMENTS FOR DOCUMENTATION
-- ============================================================================

COMMENT ON COLUMN blocks.reasoning_merkle_root IS
'R_t: Merkle root of reasoning/proof events for this block';

COMMENT ON COLUMN blocks.ui_merkle_root IS
'U_t: Merkle root of UI/human interaction events for this block';

COMMENT ON COLUMN blocks.composite_attestation_root IS
'H_t: SHA256(R_t || U_t) - Composite dual attestation binding both event streams';

COMMENT ON COLUMN blocks.attestation_metadata IS
'Mirror Auditor metadata: timestamps, verification status, cross-epoch hashes';
