import os, psycopg
url=os.environ["DATABASE_URL"]
with psycopg.connect(url) as c, c.cursor() as cur:
    cur.execute("""
        INSERT INTO policy_settings(key, value, updated_at)
        VALUES ('active_policy_hash','bootstrap', now())
        ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value, updated_at=now();
    """)
    c.commit()
print("Policy row OK")
