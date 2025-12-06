-- Fix proofs schema and statements hash alignment
-- Additive migration - no destructive DDL

-- Add status column to proofs table
ALTER TABLE proofs ADD COLUMN IF NOT EXISTS status TEXT DEFAULT 'success';

-- Add constraint with Postgres 15 compatible syntax
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'proofs_status_check') THEN
        ALTER TABLE proofs ADD CONSTRAINT proofs_status_check CHECK (status IN ('success', 'failure'));
    END IF;
END $$;

-- Backfill existing rows as success
UPDATE proofs SET status = 'success' WHERE status IS NULL;

-- Add index on statements hash
CREATE INDEX IF NOT EXISTS idx_statements_hash ON statements(hash);

-- Recompute statements hash from normalized content if needed
-- This ensures hash matches sha256(normalized_text)
UPDATE statements
SET hash = encode(sha256(content_norm::bytea), 'hex')
WHERE hash IS NULL OR hash = '';

-- Ensure we have the Propositional theory
INSERT INTO theories (name, slug, version, logic)
VALUES ('Propositional', 'pl', 'v0', 'classical')
ON CONFLICT (name) DO NOTHING;
