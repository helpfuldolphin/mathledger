
-- Ensure slug column exists and is properly populated
-- Idempotent migration to fix CI schema issues

BEGIN;

ALTER TABLE theories ADD COLUMN IF NOT EXISTS slug text;

CREATE UNIQUE INDEX IF NOT EXISTS theories_slug_idx ON theories(slug);

UPDATE theories SET slug = 'pl' WHERE name = 'Propositional' AND (slug IS NULL OR slug = '');
UPDATE theories SET slug = 'fol' WHERE name = 'First Order' AND (slug IS NULL OR slug = '');
UPDATE theories SET slug = 'migration' WHERE name = 'Migration' AND (slug IS NULL OR slug = '');

INSERT INTO theories (name, slug, version, logic) VALUES
  ('Propositional', 'pl', 'v0', 'classical'),
  ('First Order', 'fol', 'v0', 'classical')
ON CONFLICT (name) DO UPDATE SET 
  slug = EXCLUDED.slug,
  version = EXCLUDED.version,
  logic = EXCLUDED.logic;

COMMIT;
