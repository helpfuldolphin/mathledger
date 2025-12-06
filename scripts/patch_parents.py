import os, psycopg
ddl = """
CREATE TABLE IF NOT EXISTS proof_parents(
  child_hash  VARCHAR(64) NOT NULL,
  parent_hash VARCHAR(64) NOT NULL,
  created_at  TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_pp_child  ON proof_parents(child_hash);
CREATE INDEX IF NOT EXISTS idx_pp_parent ON proof_parents(parent_hash);
"""
with psycopg.connect(os.environ["DATABASE_URL"]) as c, c.cursor() as cur:
    cur.execute(ddl); c.commit()
print("proof_parents ready")
