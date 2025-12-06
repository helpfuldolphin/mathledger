import os, psycopg
u=os.environ["DATABASE_URL"]
with psycopg.connect(u) as c, c.cursor() as cur:
    cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='statements'")
    sc={r[0].lower() for r in cur.fetchall()}
    hcol='hash' if 'hash' in sc else ('canonical_hash' if 'canonical_hash' in sc else None)
    order='created_at' if 'created_at' in sc else 'id'
    cur.execute(f"SELECT {hcol} FROM statements ORDER BY COALESCE({order}, now()) DESC LIMIT 1")
    print(cur.fetchone()[0])
