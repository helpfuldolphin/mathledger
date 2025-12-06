import os, psycopg
with psycopg.connect(os.environ["DATABASE_URL"]) as c, c.cursor() as cur:
    # delete duplicates first (keep earliest)
    cur.execute("""
        DELETE FROM proof_parents a
        USING proof_parents b
        WHERE a.ctid > b.ctid
          AND a.child_hash  = b.child_hash
          AND a.parent_hash = b.parent_hash;
    """)
    c.commit()
    cur.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS uq_proof_parents
        ON proof_parents(child_hash, parent_hash);
    """)
    c.commit()
print("proof_parents deduped and unique-indexed")
