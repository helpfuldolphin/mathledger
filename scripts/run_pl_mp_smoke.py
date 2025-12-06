# scripts/run_pl_mp_smoke.py
from __future__ import annotations
import os, json, hashlib, time
from datetime import datetime
from typing import Tuple, Any
import psycopg

def _sha(s:str)->str: return hashlib.sha256(s.encode()).hexdigest()
def _db(): return os.environ["DATABASE_URL"]

def _ensure_system(cur, name="pl")->int:
    cur.execute("SELECT id FROM systems WHERE name=%s LIMIT 1",(name,))
    r=cur.fetchone()
    if r: return int(r[0])
    cols=set()
    cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='systems'")
    cols={c[0].lower() for c in cur.fetchall()}
    coln=[]; vals=[]
    for k,v in (("name",name),("slug",name),("logic","PL"),("version","v1"),
                ("created_at",datetime.utcnow()),("updated_at",datetime.utcnow())):
        if k in cols: coln.append(k); vals.append(v)
    if not coln:
        cur.execute("INSERT INTO systems(name) VALUES (%s) RETURNING id",(name,))
        return int(cur.fetchone()[0])
    ph=",".join(["%s"]*len(coln))
    cur.execute(f"INSERT INTO systems({','.join(coln)}) VALUES({ph}) RETURNING id", vals)
    return int(cur.fetchone()[0])

def _upsert_statement(cur, system_id:int, pretty:str, norm:str)->Tuple[int,str]:
    h=_sha(norm)
    cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='statements'")
    sc={r[0].lower() for r in cur.fetchall()}
    key = "hash" if "hash" in sc else ("canonical_hash" if "canonical_hash" in sc else None)
    if key:
        cur.execute(f"SELECT id FROM statements WHERE {key}=%s LIMIT 1",(h,))
        r=cur.fetchone()
        if r: return int(r[0]), h
    coln=[]; vals=[]
    for k,v in (("system_id",system_id),("text",pretty),("statement",pretty),
                ("normalized_text",norm),("normalized",norm),
                ("hash",h),("canonical_hash",h),
                ("created_at",datetime.utcnow()),("updated_at",datetime.utcnow())):
        if k in sc: coln.append(k); vals.append(v)
    if not coln:
        cur.execute("INSERT INTO statements(normalized_text) VALUES (%s) RETURNING id",(norm,))
        return int(cur.fetchone()[0]), h
    ph=",".join(["%s"]*len(coln))
    cur.execute(f"INSERT INTO statements({','.join(coln)}) VALUES({ph}) RETURNING id", vals)
    return int(cur.fetchone()[0]), h

def _insert_proof(cur, sid:int, method="mp", status="success")->int:
    cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='proofs'")
    pc={r[0].lower() for r in cur.fetchall()}
    coln=[]; vals=[]
    for k,v in (("statement_id",sid),("method",method),("status",status),
                ("success",True),("created_at",datetime.utcnow()),("updated_at",datetime.utcnow())):
        if k in pc: coln.append(k); vals.append(v)
    if not coln or "statement_id" not in coln:
        cur.execute("INSERT INTO proofs(statement_id) VALUES (%s) RETURNING id",(sid,))
        return int(cur.fetchone()[0])
    ph=",".join(["%s"]*len(coln))
    cur.execute(f"INSERT INTO proofs({','.join(coln)}) VALUES({ph}) RETURNING id", vals)
    r=cur.fetchone(); return int(r[0]) if r else 0

def _edge(cur, child_h:str, parent_h:str)->None:
    cur.execute("""
        CREATE TABLE IF NOT EXISTS proof_parents(
          child_hash  VARCHAR(64) NOT NULL,
          parent_hash VARCHAR(64) NOT NULL,
          created_at  TIMESTAMPTZ DEFAULT now()
        );
        CREATE INDEX IF NOT EXISTS idx_pp_child  ON proof_parents(child_hash);
        CREATE INDEX IF NOT EXISTS idx_pp_parent ON proof_parents(parent_hash);
    """)
    cur.execute("""
        INSERT INTO proof_parents(child_hash,parent_hash)
        VALUES (%s,%s) ON CONFLICT DO NOTHING
    """, (child_h, parent_h))

def main()->int:
    db=_db()
    with psycopg.connect(db, connect_timeout=5) as conn, conn.cursor() as cur:
        sys_id = _ensure_system(cur,"pl")
        # Premises for MP: p, and p->(q->p)
        sid_p,  hp  = _upsert_statement(cur, sys_id, "p", "p")
        imp     = "p->(q->p)"
        sid_imp,himp= _upsert_statement(cur, sys_id, "p -> (q -> p)", imp)
        # Conclusion: q->p
        concl   = "q->p"
        sid_c,hc= _upsert_statement(cur, sys_id, "q -> p", concl)
        _insert_proof(cur, sid_c, method="mp", status="success")
        # Parent edges for MP
        _edge(cur, hc, hp)
        _edge(cur, hc, himp)
        conn.commit()
        print("MP_INSERTED=1")
        print("CHILD_HASH="+hc)
    return 0

if __name__=="__main__":
    import sys
    try: sys.exit(main())
    except SystemExit as e: pass
