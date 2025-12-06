-- Fix statements hash column and add index
-- Additive migration - no destructive DDL

-- Add hash column if missing
ALTER TABLE statements ADD COLUMN IF NOT EXISTS hash TEXT;

-- Backfill hash from normalized content
UPDATE statements
SET hash = encode(sha256(content_norm::bytea), 'hex')
WHERE hash IS NULL OR hash = '';

-- Create index on hash column
CREATE INDEX IF NOT EXISTS idx_statements_hash ON statements(hash);

-- Ensure proofs schema has proper defaults
ALTER TABLE proofs ADD COLUMN IF NOT EXISTS status TEXT DEFAULT 'success';

-- Add constraint with Postgres 15 compatible syntax
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'proofs_status_check') THEN
        ALTER TABLE proofs ADD CONSTRAINT proofs_status_check CHECK (status IN ('success', 'failure'));
    END IF;
END $$;

-- Backfill existing proofs as success
UPDATE proofs SET status = 'success' WHERE status IS NULL;

-- Add created_at if missing
ALTER TABLE proofs ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT NOW();
