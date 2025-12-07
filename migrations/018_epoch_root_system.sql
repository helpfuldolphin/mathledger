-- Migration: 018_epoch_root_system.sql
-- Author: Manus-B (Ledger Replay Architect & Attestation Runtime Engineer)
-- Date: 2025-12-06
-- Description: Implement epoch-level aggregation of block attestation roots
--
-- Purpose:
--   - Enable efficient verification of large ledger segments
--   - Provide natural checkpoints for archival and pruning
--   - Support hierarchical Merkle proofs (blocks → epochs → super-epochs)
--
-- Epoch Structure:
--   - Epoch N contains blocks [N*100, (N+1)*100)
--   - Epoch root E_t = MerkleRoot([H_0, H_1, ..., H_99])
--   - Each H_i is the composite attestation root of block i in the epoch
--
-- Dependencies:
--   - Requires migration 016 (monotone_ledger.sql) for blocks table
--   - Requires migration 015 (dual_root_attestation.sql) for attestation roots
--
-- Compatibility:
--   - PQ-safe: epoch_root column can store SHA-256 or SHA-3 hashes
--   - Hash algorithm version tracked in epoch_metadata
--   - Supports dual-commitment during PQ migration

-- ============================================================================
-- EPOCHS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS epochs (
    -- Primary key
    id BIGSERIAL PRIMARY KEY,
    
    -- Epoch identification
    epoch_number BIGINT NOT NULL,
    
    -- Block range (half-open interval: [start, end))
    start_block_number BIGINT NOT NULL,
    end_block_number BIGINT NOT NULL,
    block_count INT NOT NULL,
    
    -- Epoch root: Merkle root of all composite attestation roots (H_t) in epoch
    -- E_t = MerkleRoot([H_0, H_1, ..., H_{block_count-1}])
    -- Length: 64 chars for SHA-256, 128 chars for SHA-3-512
    epoch_root TEXT NOT NULL,
    
    -- Aggregate statistics
    total_proofs INT NOT NULL DEFAULT 0,
    total_ui_events INT NOT NULL DEFAULT 0,
    
    -- Sealing metadata
    sealed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    sealed_by TEXT DEFAULT 'epoch_sealer',
    
    -- Epoch attestation metadata (JSONB for flexibility)
    -- Contains:
    --   - composite_roots: List[str] - All H_t values in epoch
    --   - block_ids: List[int] - All block IDs in epoch
    --   - epoch_size: int - Expected blocks per epoch (usually 100)
    --   - hash_version: str - Hash algorithm version ("sha256-v1", "sha3-v1", etc.)
    --   - hash_algorithm: str - Hash algorithm name ("SHA-256", "SHA-3-512", etc.)
    epoch_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    
    -- System reference (which theory/system this epoch belongs to)
    system_id UUID REFERENCES theories(id) ON DELETE CASCADE,
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================================
-- INDEXES
-- ============================================================================

-- Primary lookup: epoch number (most common query)
CREATE INDEX epochs_epoch_number_idx ON epochs(epoch_number DESC);

-- System-scoped queries
CREATE INDEX epochs_system_id_idx ON epochs(system_id);

-- Epoch root lookup (for verification)
CREATE INDEX epochs_epoch_root_idx ON epochs(epoch_root);

-- Block range queries (find epoch containing block N)
CREATE INDEX epochs_block_range_idx ON epochs(start_block_number, end_block_number);

-- Composite index for system-scoped epoch queries
CREATE INDEX epochs_system_epoch_idx ON epochs(system_id, epoch_number DESC);

-- ============================================================================
-- CONSTRAINTS
-- ============================================================================

-- Unique epoch number per system
CREATE UNIQUE INDEX epochs_system_epoch_unique ON epochs(system_id, epoch_number);

-- Block range validity
ALTER TABLE epochs ADD CONSTRAINT epochs_block_range_valid
CHECK (end_block_number > start_block_number);

-- Block count consistency
ALTER TABLE epochs ADD CONSTRAINT epochs_block_count_valid
CHECK (block_count = end_block_number - start_block_number);

-- Block count positive
ALTER TABLE epochs ADD CONSTRAINT epochs_block_count_positive
CHECK (block_count > 0);

-- Epoch number non-negative
ALTER TABLE epochs ADD CONSTRAINT epochs_epoch_number_non_negative
CHECK (epoch_number >= 0);

-- Epoch root non-empty
ALTER TABLE epochs ADD CONSTRAINT epochs_epoch_root_non_empty
CHECK (LENGTH(epoch_root) > 0);

-- Epoch root hex format (64 chars for SHA-256, 128 for SHA-3-512)
ALTER TABLE epochs ADD CONSTRAINT epochs_epoch_root_hex_format
CHECK (epoch_root ~ '^[0-9a-f]{64}$' OR epoch_root ~ '^[0-9a-f]{128}$');

-- ============================================================================
-- BLOCKS TABLE EXTENSION
-- ============================================================================

-- Add epoch reference to blocks table
ALTER TABLE blocks ADD COLUMN IF NOT EXISTS epoch_id BIGINT REFERENCES epochs(id) ON DELETE SET NULL;

-- Index for epoch-to-blocks queries
CREATE INDEX IF NOT EXISTS blocks_epoch_id_idx ON epochs(epoch_id);

-- ============================================================================
-- COMMENTS (Documentation)
-- ============================================================================

COMMENT ON TABLE epochs IS 
'Epoch-level aggregation of block attestation roots for efficient verification.

An epoch is a fixed-size sequence of blocks (default: 100 blocks) with a 
deterministic boundary. The epoch root is the Merkle root of all composite 
attestation roots (H_t) in the epoch, enabling O(log n) verification instead 
of O(n) for large ledger segments.

Epoch Structure:
  - Epoch N: Blocks [N*100, (N+1)*100)
  - E_t = MerkleRoot([H_0, H_1, ..., H_99])
  - Each H_i is the composite attestation root of block i

Benefits:
  - Efficient verification of large ledger segments
  - Natural boundaries for archival and pruning
  - Hierarchical Merkle proofs (blocks → epochs → super-epochs)
  - Scalability: O(log n) verification cost vs O(n)';

COMMENT ON COLUMN epochs.epoch_number IS 
'Epoch number (0-indexed). Epoch N contains blocks [N*100, (N+1)*100).';

COMMENT ON COLUMN epochs.epoch_root IS 
'E_t: Merkle root of all composite attestation roots (H_t) in epoch.
Length: 64 chars (SHA-256) or 128 chars (SHA-3-512).
Computed as: E_t = MerkleRoot([H_0, H_1, ..., H_{block_count-1}])';

COMMENT ON COLUMN epochs.epoch_metadata IS 
'Epoch attestation metadata (JSONB).
Contains:
  - composite_roots: List[str] - All H_t values in epoch
  - block_ids: List[int] - All block IDs in epoch
  - epoch_size: int - Expected blocks per epoch
  - hash_version: str - Hash algorithm version
  - hash_algorithm: str - Hash algorithm name';

COMMENT ON COLUMN epochs.sealed_at IS 
'Timestamp when epoch was sealed (finalized).';

COMMENT ON COLUMN epochs.sealed_by IS 
'Entity that sealed the epoch (e.g., "epoch_sealer", "admin", "backfill_script").';

COMMENT ON COLUMN blocks.epoch_id IS 
'Reference to the epoch this block belongs to.
NULL if block not yet assigned to an epoch (e.g., recent blocks before epoch sealing).';

-- ============================================================================
-- FUNCTIONS
-- ============================================================================

-- Function: Get epoch number for a given block number
CREATE OR REPLACE FUNCTION get_epoch_number(block_num BIGINT, epoch_size INT DEFAULT 100)
RETURNS BIGINT AS $$
BEGIN
    RETURN block_num / epoch_size;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

COMMENT ON FUNCTION get_epoch_number IS 
'Get epoch number for a given block number.
Example: get_epoch_number(0) = 0, get_epoch_number(99) = 0, get_epoch_number(100) = 1';

-- Function: Get epoch block range
CREATE OR REPLACE FUNCTION get_epoch_range(epoch_num BIGINT, epoch_size INT DEFAULT 100)
RETURNS TABLE(start_block BIGINT, end_block BIGINT) AS $$
BEGIN
    RETURN QUERY SELECT 
        epoch_num * epoch_size AS start_block,
        (epoch_num + 1) * epoch_size AS end_block;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

COMMENT ON FUNCTION get_epoch_range IS 
'Get block range for an epoch (half-open interval: [start, end)).
Example: get_epoch_range(0) = (0, 100), get_epoch_range(1) = (100, 200)';

-- Function: Check if epoch should be sealed after this block
CREATE OR REPLACE FUNCTION should_seal_epoch(block_num BIGINT, epoch_size INT DEFAULT 100)
RETURNS BOOLEAN AS $$
BEGIN
    RETURN (block_num + 1) % epoch_size = 0;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

COMMENT ON FUNCTION should_seal_epoch IS 
'Check if an epoch should be sealed after this block.
Returns TRUE if block_num is the last block in an epoch.
Example: should_seal_epoch(99) = TRUE, should_seal_epoch(100) = FALSE';

-- ============================================================================
-- VIEWS
-- ============================================================================

-- View: Epoch summary with block count verification
CREATE OR REPLACE VIEW epoch_summary AS
SELECT 
    e.id,
    e.epoch_number,
    e.start_block_number,
    e.end_block_number,
    e.block_count AS declared_block_count,
    COUNT(b.id) AS actual_block_count,
    e.block_count = COUNT(b.id) AS block_count_valid,
    e.total_proofs,
    e.total_ui_events,
    e.epoch_root,
    e.sealed_at,
    e.sealed_by,
    e.system_id
FROM epochs e
LEFT JOIN blocks b ON b.epoch_id = e.id
GROUP BY e.id;

COMMENT ON VIEW epoch_summary IS 
'Epoch summary with block count verification.
Compares declared_block_count (from epochs table) with actual_block_count (from blocks table).
block_count_valid = TRUE if counts match.';

-- View: Unsealed blocks (blocks not yet assigned to an epoch)
CREATE OR REPLACE VIEW unsealed_blocks AS
SELECT 
    b.id,
    b.block_number,
    b.system_id,
    get_epoch_number(b.block_number) AS expected_epoch_number,
    b.created_at
FROM blocks b
WHERE b.epoch_id IS NULL
ORDER BY b.block_number;

COMMENT ON VIEW unsealed_blocks IS 
'Blocks not yet assigned to an epoch (epoch_id IS NULL).
Shows expected_epoch_number for each unsealed block.';

-- ============================================================================
-- MIGRATION SAFETY
-- ============================================================================

-- This migration is SAFE to run on existing databases:
--   - Creates new table (epochs) without modifying existing tables
--   - Adds nullable column (epoch_id) to blocks table
--   - No data loss risk
--   - Backfill can be run separately after migration

-- Rollback strategy:
--   - DROP TABLE epochs CASCADE;
--   - ALTER TABLE blocks DROP COLUMN epoch_id;

-- ============================================================================
-- BACKFILL NOTES
-- ============================================================================

-- After running this migration, run backfill script to:
--   1. Group existing blocks into epochs
--   2. Compute epoch roots for historical epochs
--   3. Update blocks.epoch_id for all blocks
--
-- See: scripts/backfill_epochs.py

-- Example backfill query (DO NOT RUN IN MIGRATION):
-- 
-- WITH epoch_blocks AS (
--     SELECT 
--         system_id,
--         get_epoch_number(block_number) AS epoch_number,
--         ARRAY_AGG(composite_attestation_root ORDER BY block_number) AS composite_roots,
--         ARRAY_AGG(id ORDER BY block_number) AS block_ids,
--         MIN(block_number) AS start_block,
--         MAX(block_number) + 1 AS end_block,
--         COUNT(*) AS block_count
--     FROM blocks
--     GROUP BY system_id, get_epoch_number(block_number)
-- )
-- INSERT INTO epochs (
--     system_id, epoch_number, start_block_number, end_block_number, 
--     block_count, epoch_root, epoch_metadata
-- )
-- SELECT 
--     system_id,
--     epoch_number,
--     start_block,
--     end_block,
--     block_count,
--     compute_merkle_root(composite_roots) AS epoch_root,  -- Requires Python
--     jsonb_build_object(
--         'composite_roots', composite_roots,
--         'block_ids', block_ids,
--         'epoch_size', 100,
--         'backfilled', true
--     ) AS epoch_metadata
-- FROM epoch_blocks;

-- ============================================================================
-- END MIGRATION
-- ============================================================================
