-- Normalize statements text and ensure hash consistency
-- Additive migration - no destructive DDL

-- Add normalized_text column if missing
ALTER TABLE statements ADD COLUMN IF NOT EXISTS normalized_text TEXT;

-- Backfill normalized_text with ML normalization
-- Replace unicode operators with ASCII equivalents and remove spaces
UPDATE statements SET normalized_text =
  REPLACE(
    REPLACE(
      REPLACE(
        REPLACE(text, '→', '->'),
        '∧', '/\\'
      ),
      '∨', '\\/'
    ),
    ' ', ''
  )
WHERE normalized_text IS NULL OR normalized_text = '';

-- Ensure hash column exists
ALTER TABLE statements ADD COLUMN IF NOT EXISTS hash TEXT;

-- Backfill hash from normalized_text
UPDATE statements
SET hash = LOWER(encode(sha256(normalized_text::bytea), 'hex'))
WHERE hash IS NULL OR hash = '';

-- Create index on hash column
CREATE INDEX IF NOT EXISTS idx_statements_hash ON statements(hash);

-- Ensure proofs schema has proper defaults for inserts
ALTER TABLE proofs ADD COLUMN IF NOT EXISTS method TEXT DEFAULT 'axiom';
ALTER TABLE proofs ADD COLUMN IF NOT EXISTS status TEXT DEFAULT 'success';
ALTER TABLE proofs ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT NOW();

-- Add constraint if not exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'proofs_status_check'
    ) THEN
        ALTER TABLE proofs ADD CONSTRAINT proofs_status_check
        CHECK (status IN ('success', 'failure'));
    END IF;
END $$;
