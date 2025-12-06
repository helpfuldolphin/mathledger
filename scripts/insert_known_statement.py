import os, psycopg, hashlib
url=os.environ["DATABASE_URL"]
h=hashlib.sha256(b"p->(q->p)").hexdigest()
with psycopg.connect(url) as c, c.cursor() as cur:
    cur.execute("""
      INSERT INTO statements(system_id,text,normalized_text,hash,created_at)
      VALUES ((SELECT id FROM systems WHERE name='pl'), %s,%s,%s, now())
      ON CONFLICT (hash) DO NOTHING
    """, ('p->(q->p)','p->(q->p)',h))
    c.commit()
print(h)
