-- Idempotent migration: ensure normalized_text, hash, and index exist
-- Safe to run multiple times - no destructive DDL

-- Add normalized_text column if missing
ALTER TABLE statements ADD COLUMN IF NOT EXISTS normalized_text TEXT;

-- Add hash column if missing
ALTER TABLE statements ADD COLUMN IF NOT EXISTS hash TEXT;

-- Backfill normalized_text with ASCII normalization (idempotent)
-- Replace unicode operators with ASCII equivalents and remove spaces
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
WHERE normalized_text IS NULL OR normalized_text = '' OR normalized_text !=
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
    '  ', ''
  );

-- Backfill hash from normalized_text (idempotent)
UPDATE statements
SET hash = LOWER(encode(sha256(normalized_text::bytea), 'hex'))
WHERE hash IS NULL OR hash = '' OR hash != LOWER(encode(sha256(normalized_text::bytea), 'hex'));

-- Create index if not exists (idempotent)
CREATE INDEX IF NOT EXISTS idx_statements_hash ON statements(hash);
