import os, psycopg
url=os.environ["DATABASE_URL"]
DDL = """
ALTER TABLE statements
  ADD COLUMN IF NOT EXISTS system TEXT,
  ADD COLUMN IF NOT EXISTS normalized TEXT,
  ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT now();  -- harmless if already present
"""
with psycopg.connect(url) as c, c.cursor() as cur:
    cur.execute(DDL); c.commit()
print("Statements columns OK")
