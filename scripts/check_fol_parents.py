import os, psycopg, hashlib
u=os.environ["DATABASE_URL"]
H1=hashlib.sha256(b"a=c").hexdigest()
H2=hashlib.sha256(b"f(a)=f(c)").hexdigest()
with psycopg.connect(u) as c, c.cursor() as cur:
    cur.execute("SELECT COUNT(*) FROM statements WHERE hash=%s",(H1,)); print("has a=c:",cur.fetchone()[0])
    cur.execute("SELECT COUNT(*) FROM statements WHERE hash=%s",(H2,)); print("has f(a)=f(c):",cur.fetchone()[0])
    cur.execute("""SELECT COUNT(*) FROM proofs p JOIN statements s ON p.statement_id=s.id WHERE s.hash=%s""",(H2,))
    print("proofs for f(a)=f(c):", cur.fetchone()[0])
    cur.execute("SELECT parent_hash FROM proof_parents WHERE child_hash=%s ORDER BY created_at",(H1,))
    print("parents of a=c:", [r[0] for r in cur.fetchall()])
    cur.execute("SELECT parent_hash FROM proof_parents WHERE child_hash=%s ORDER BY created_at",(H2,))
    print("parents of f(a)=f(c):", [r[0] for r in cur.fetchall()])
