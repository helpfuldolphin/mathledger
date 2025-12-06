-- Ensure schema parity for statements and proofs
-- Idempotent migration with proper constraint handling

-- Statements: Ensure normalized_text and hash columns exist
ALTER TABLE statements ADD COLUMN IF NOT EXISTS normalized_text TEXT;
ALTER TABLE statements ADD COLUMN IF NOT EXISTS hash TEXT;

-- Backfill normalized_text with proper ASCII normalization
UPDATE statements SET normalized_text =
  REPLACE(
    REPLACE(
      REPLACE(
        REPLACE(
          REPLACE(text, '→', '->'),
          '∧', '/\\'
        ),
        '∨', '\\/'
      ),
      ' ', ''
    ),
    '  ', ''  -- Remove double spaces
  )
WHERE normalized_text IS NULL OR normalized_text = '';

-- Backfill hash from normalized_text (ensure it's lowercase hex)
UPDATE statements
SET hash = LOWER(encode(sha256(normalized_text::bytea), 'hex'))
WHERE hash IS NULL OR hash = '';

-- Create index on hash if not exists
CREATE INDEX IF NOT EXISTS idx_statements_hash ON statements(hash);

-- Proofs: Ensure all required columns exist with proper defaults
ALTER TABLE proofs ADD COLUMN IF NOT EXISTS method TEXT DEFAULT 'axiom';
ALTER TABLE proofs ADD COLUMN IF NOT EXISTS status TEXT DEFAULT 'success';
ALTER TABLE proofs ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT NOW();

-- Add status constraint using proper PostgreSQL syntax
DO $$
BEGIN
    -- Try to add the constraint
    BEGIN
        ALTER TABLE proofs ADD CONSTRAINT proofs_status_check
        CHECK (status IN ('success', 'failure'));
    EXCEPTION
        WHEN duplicate_object THEN
            NULL; -- Constraint already exists, ignore
    END;
END $$;
