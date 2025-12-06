-- Migration 000: Schema Version Tracking
-- This migration creates the schema_migrations table for tracking applied migrations.
-- MUST run before all other migrations.

-- Create schema_migrations tracking table
CREATE TABLE IF NOT EXISTS schema_migrations (
    version TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    checksum TEXT,  -- SHA-256 of migration file contents (optional)
    duration_ms INTEGER,  -- Migration execution time (optional)
    status TEXT DEFAULT 'success' CHECK (status IN ('success', 'failed', 'skipped'))
);

-- Create index for querying migration history
CREATE INDEX IF NOT EXISTS idx_schema_migrations_applied_at ON schema_migrations(applied_at DESC);

-- Record this migration
INSERT INTO schema_migrations (version, checksum, status)
VALUES ('000_schema_version', 'init', 'success')
ON CONFLICT (version) DO NOTHING;

-- Add comment for documentation
COMMENT ON TABLE schema_migrations IS 'Tracks applied database migrations for MathLedger schema evolution';
COMMENT ON COLUMN schema_migrations.version IS 'Migration file name (e.g., 001_init, 002_blocks_lemmas)';
COMMENT ON COLUMN schema_migrations.applied_at IS 'Timestamp when migration was applied';
COMMENT ON COLUMN schema_migrations.checksum IS 'SHA-256 hash of migration file to detect modifications';
COMMENT ON COLUMN schema_migrations.duration_ms IS 'Migration execution time in milliseconds';
COMMENT ON COLUMN schema_migrations.status IS 'Migration status: success, failed, or skipped';
