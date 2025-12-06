import os, psycopg
url=os.environ["DATABASE_URL"]

CANDIDATE_COLS = [
  # common alternates engines use
  ("pretty","text"),
  ("normalized","text"),
  ("statement","text"),
  ("formula","text"),
  ("normalized_hash","varchar(64)"),
  ("norm_hash","varchar(64)"),
  ("canonical_hash","varchar(64)"),
  ("policy_hash","varchar(64)"),
  ("source","text"),
  ("meta","jsonb"),
  ("hash_short","varchar(16)"),
  ("run_id","integer"),
]

DDL = []
for name, typ in CANDIDATE_COLS:
    if name in ("meta",):
        DDL.append(f"ALTER TABLE statements ADD COLUMN IF NOT EXISTS {name} {typ} DEFAULT '{{}}'::jsonb;")
    else:
        DDL.append(f"ALTER TABLE statements ADD COLUMN IF NOT EXISTS {name} {typ};")

DDL.append("CREATE UNIQUE INDEX IF NOT EXISTS uq_statements_hash ON statements(hash);")

with psycopg.connect(url) as c, c.cursor() as cur:
    cur.execute(";\n".join(DDL)); c.commit()
print("Statements widened OK")
