import os, psycopg
url = os.environ["DATABASE_URL"]

DDL = """
-- policy_settings must support key/value upsert with updated_at
CREATE TABLE IF NOT EXISTS policy_settings(
  id SERIAL PRIMARY KEY,
  key TEXT,
  value TEXT,
  policy_hash VARCHAR(64),
  created_at TIMESTAMPTZ DEFAULT now()
);
ALTER TABLE policy_settings
  ADD COLUMN IF NOT EXISTS key TEXT,
  ADD COLUMN IF NOT EXISTS value TEXT,
  ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT now();

-- ensure key is unique for ON CONFLICT(key) DO UPDATE ...
CREATE UNIQUE INDEX IF NOT EXISTS uq_policy_settings_key ON policy_settings(key);

-- statements needs updated_at because code updates it on upsert
ALTER TABLE statements
  ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT now();

-- proofs often rely on created_at for /metrics proofs_per_sec
ALTER TABLE proofs
  ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT now();

-- sanity: blocks already has created_at; keep as-is
"""
with psycopg.connect(url) as c, c.cursor() as cur:
    cur.execute(DDL)
    c.commit()
print("Schema patch applied.")
