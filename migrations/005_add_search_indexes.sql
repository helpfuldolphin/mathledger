-- Migration 005: Add search performance indexes
-- This migration adds database indexes to improve search performance
-- NOTE: CONCURRENTLY removed for migration compatibility (can't run in transaction)
-- In production, consider using CONCURRENTLY to avoid table locks

-- Enable trigram extension first (required for GIN index)
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Add GIN index for full-text search on content_norm
-- This will significantly improve ILIKE queries on the content_norm column
CREATE INDEX IF NOT EXISTS idx_statements_content_norm_gin
ON statements USING gin(content_norm gin_trgm_ops);

-- Add B-tree index on derivation_depth for range queries
-- This will improve queries with depth_gt and depth_lt filters
CREATE INDEX IF NOT EXISTS idx_statements_derivation_depth
ON statements (derivation_depth);

-- Add B-tree index on status for status filtering
-- This will improve queries filtering by status
CREATE INDEX IF NOT EXISTS idx_statements_status
ON statements (status);

-- Add B-tree index on system_id for theory filtering
-- This will improve joins with the theories table
CREATE INDEX IF NOT EXISTS idx_statements_system_id
ON statements (system_id);

-- Add composite index for common search patterns
-- This will improve queries that filter by both status and derivation_depth
CREATE INDEX IF NOT EXISTS idx_statements_status_depth
ON statements (status, derivation_depth);

-- Add index on created_at for ordering
-- This will improve ORDER BY created_at DESC queries
CREATE INDEX IF NOT EXISTS idx_statements_created_at
ON statements (created_at DESC);

-- Enable trigram extension for GIN index (if not already enabled)
-- This is required for the gin_trgm_ops operator class
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Add comment explaining the indexes
COMMENT ON INDEX idx_statements_content_norm_gin IS 'GIN index for full-text search on statement content';
COMMENT ON INDEX idx_statements_derivation_depth IS 'B-tree index for derivation depth range queries';
COMMENT ON INDEX idx_statements_status IS 'B-tree index for status filtering';
COMMENT ON INDEX idx_statements_system_id IS 'B-tree index for theory filtering';
COMMENT ON INDEX idx_statements_status_depth IS 'Composite index for status and depth filtering';
COMMENT ON INDEX idx_statements_created_at IS 'B-tree index for chronological ordering';
