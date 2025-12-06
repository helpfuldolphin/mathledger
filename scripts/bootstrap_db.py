import os, psycopg
url=os.environ["DATABASE_URL"]
ddl = """
CREATE TABLE IF NOT EXISTS systems(
  id SERIAL PRIMARY KEY,
  name TEXT UNIQUE NOT NULL
);
INSERT INTO systems(name) VALUES ('pl') ON CONFLICT (name) DO NOTHING;

CREATE TABLE IF NOT EXISTS statements(
  id SERIAL PRIMARY KEY,
  system_id INTEGER REFERENCES systems(id) ON DELETE SET NULL,
  text TEXT,
  normalized_text TEXT,
  hash VARCHAR(64) UNIQUE,
  created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_statements_hash ON statements(hash);

CREATE TABLE IF NOT EXISTS proofs(
  id SERIAL PRIMARY KEY,
  statement_id INTEGER REFERENCES statements(id) ON DELETE CASCADE,
  method TEXT,
  status TEXT,
  success BOOLEAN,
  created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_proofs_created_at ON proofs(created_at);

CREATE TABLE IF NOT EXISTS blocks(
  id SERIAL PRIMARY KEY,
  block_number INTEGER UNIQUE,
  merkle_root TEXT,
  created_at TIMESTAMPTZ DEFAULT now(),
  header JSONB,
  statements JSONB
);

CREATE TABLE IF NOT EXISTS policy_settings(
  id SERIAL PRIMARY KEY,
  key TEXT,
  value TEXT,
  policy_hash VARCHAR(64),
  created_at TIMESTAMPTZ DEFAULT now()
);
"""
with psycopg.connect(url) as c, c.cursor() as cur:
    cur.execute(ddl); c.commit()
print("DB schema OK")
