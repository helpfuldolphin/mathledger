import os, psycopg
url=os.environ["DATABASE_URL"]
ddl = """
ALTER TABLE policy_settings
  ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT now();
CREATE UNIQUE INDEX IF NOT EXISTS uq_policy_settings_key ON policy_settings(key);

ALTER TABLE statements ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT now();
ALTER TABLE proofs     ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT now();
ALTER TABLE blocks     ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT now();
"""
with psycopg.connect(url) as c, c.cursor() as cur:
    cur.execute(ddl); c.commit()
print("DB patch OK")
