-- Migration: 017_enforce_app_timestamps
-- Author: Gemini C
-- Date: 2025-11-27
-- Purpose: Remove DEFAULT NOW() from critical tables to enforce application-layer deterministic timestamps.

BEGIN;

-- Statements: strictly content-derived timestamps
ALTER TABLE statements ALTER COLUMN created_at DROP DEFAULT;
-- Note: updated_at might still be useful as operational metadata, but for strictness we drop it too.
ALTER TABLE statements ALTER COLUMN updated_at DROP DEFAULT;

-- Proofs: strictly content-derived timestamps
ALTER TABLE proofs ALTER COLUMN created_at DROP DEFAULT;
ALTER TABLE proofs ALTER COLUMN updated_at DROP DEFAULT;

-- Blocks: strictly content-derived timestamps (from Merkle root)
ALTER TABLE blocks ALTER COLUMN created_at DROP DEFAULT;
ALTER TABLE blocks ALTER COLUMN updated_at DROP DEFAULT;
ALTER TABLE blocks ALTER COLUMN sealed_at DROP DEFAULT;

-- Runs: strictly seeded timestamps
ALTER TABLE runs ALTER COLUMN started_at DROP DEFAULT;

COMMIT;

