import os, psycopg
u=os.environ["DATABASE_URL"]
ddl = """
CREATE TABLE IF NOT EXISTS proof_parents(
  proof_id INTEGER REFERENCES proofs(id) ON DELETE CASCADE,
  parent_statement_id INTEGER REFERENCES statements(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_proof_parents_proof ON proof_parents(proof_id);
"""
with psycopg.connect(u) as c, c.cursor() as cur:
    cur.execute(ddl); c.commit()
print("proof_parents ready")
