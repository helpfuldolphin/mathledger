# scripts/run_fol_smoke.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, time, hashlib
from datetime import datetime
from typing import List, Dict, Tuple
import psycopg

try:
    import redis
except Exception:
    redis = None

from backend.fol_eq.cc import CC, const as C, fun as F
from backend.ledger.blocking import seal_block

def _sha(s: str) -> str: return hashlib.sha256(s.encode("utf-8")).hexdigest()
def _canon_eq(lhs: str, rhs: str) -> str:
    a, b = lhs.replace(" ",""), rhs.replace(" ","")
    return f"{a}={b}" if a <= b else f"{b}={a}"

def _table_cols(cur, table:str)->set[str]:
    cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name=%s", (table,))
    return {r[0].lower() for r in cur.fetchall()}

def _ensure_system(cur, name="fol_eq")->int:
    cur.execute("SELECT id FROM systems WHERE name=%s LIMIT 1", (name,))
    row = cur.fetchone()
    if row: return int(row[0])
    cols=_table_cols(cur,"systems")
    coln=[]; vals=[]
    for nm,val in (("name",name),("slug",name),("logic","EUF"),("version","v1"),
                   ("created_at",datetime.utcnow()),("updated_at",datetime.utcnow())):
        if nm in cols: coln.append(nm); vals.append(val)
    if not coln:
        cur.execute("INSERT INTO systems(name) VALUES (%s) RETURNING id",(name,)); return int(cur.fetchone()[0])
    ph=",".join(["%s"]*len(coln))
    cur.execute(f"INSERT INTO systems({','.join(coln)}) VALUES({ph}) RETURNING id", vals)
    return int(cur.fetchone()[0])

def _upsert_statement(cur, system_id:int, pretty:str, norm:str)->Tuple[int,str]:
    h=_sha(norm)
    sc=_table_cols(cur,"statements")
    if "hash" in sc:
        cur.execute("SELECT id FROM statements WHERE hash=%s LIMIT 1",(h,)); r=cur.fetchone()
        if r: return int(r[0]), h
    elif "canonical_hash" in sc:
        cur.execute("SELECT id FROM statements WHERE canonical_hash=%s LIMIT 1",(h,)); r=cur.fetchone()
        if r: return int(r[0]), h
    coln=[]; vals=[]
    for nm,val in (("system_id",system_id),("text",pretty),("statement",pretty),
                   ("normalized_text",norm),("normalized",norm),
                   ("hash",h),("canonical_hash",h),
                   ("created_at",datetime.utcnow()),("updated_at",datetime.utcnow())):
        if nm in sc: coln.append(nm); vals.append(val)
    if not coln:
        cur.execute("INSERT INTO statements(normalized_text) VALUES (%s) RETURNING id",(norm,))
        return int(cur.fetchone()[0]), h
    ph=",".join(["%s"]*len(coln))
    cur.execute(f"INSERT INTO statements({','.join(coln)}) VALUES({ph}) RETURNING id", vals)
    return int(cur.fetchone()[0]), h

def _insert_proof(cur, statement_id:int, method="cc", status="success")->int:
    pc=_table_cols(cur,"proofs")
    coln=[]; vals=[]
    for nm,val in (("statement_id",statement_id),("method",method),("status",status),
                   ("success",True),("created_at",datetime.utcnow()),("updated_at",datetime.utcnow())):
        if nm in pc: coln.append(nm); vals.append(val)
    if not coln or "statement_id" not in coln:
        cur.execute("INSERT INTO proofs(statement_id) VALUES (%s) RETURNING id",(statement_id,)); return int(cur.fetchone()[0])
    ph=",".join(["%s"]*len(coln))
    cur.execute(f"INSERT INTO proofs({','.join(coln)}) VALUES({ph}) RETURNING id", vals)
    r=cur.fetchone(); return int(r[0]) if r else 0

def _persist_block(cur, merkle:str, leafs:List[Dict[str,object]])->int:
    bc=_table_cols(cur,"blocks")
    bn=1
    if "block_number" in bc:
        cur.execute("SELECT COALESCE(MAX(block_number),0)+1 FROM blocks"); bn=int(cur.fetchone()[0])
    header={"version":"v1","timestamp":time.time(),"merkle_root":merkle,"block_number":bn}
    coln=[]; vals=[]
    for nm,val in (("block_number",bn),("merkle_root",merkle),
                   ("header",json.dumps(header)),("statements",json.dumps(leafs)),
                   ("created_at",datetime.utcnow()),("updated_at",datetime.utcnow())):
        if nm in bc: coln.append(nm); vals.append(val)
    if not coln:
        cur.execute("INSERT INTO blocks(merkle_root) VALUES (%s) RETURNING id",(merkle,))
        r=cur.fetchone(); return int(r[0]) if r else 1
    ph=",".join(["%s"]*len(coln)); ret="block_number" if "block_number" in bc else "id"
    cur.execute(f"INSERT INTO blocks({','.join(coln)}) VALUES({ph}) RETURNING {ret}", vals)
    r=cur.fetchone(); return int(r[0]) if r else bn

def _ensure_redis():
    url=os.getenv("REDIS_URL")
    if not url or not redis: return None
    try:
        return redis.from_url(url, socket_timeout=0.2, socket_connect_timeout=0.2, retry_on_timeout=False)
    except Exception:
        return None

def main()->int:
    db=os.environ.get("DATABASE_URL")
    if not db:
        print("PROOFS_INSERTED=0"); print("MERKLE="); print("BLOCK=-1"); print("ENQUEUED=0"); print("ERROR=MissingDB"); return 1

    # Build EUF instance
    a,b,c=C("a"),C("b"),C("c"); f=lambda x:F("f",x)
    axioms=[(a,b),(b,c)]; goal1=(a,c); goal2=(f(a),f(c))

    cc=CC()
    # Pre-register parents so congruence can propagate
    cc.add_term(a); cc.add_term(b); cc.add_term(c)
    cc.add_term(f(a)); cc.add_term(f(c))
    cc.assert_eqs(axioms)
    if not (cc.equal(*goal1) and cc.equal(*goal2)):
        print("PROOFS_INSERTED=0"); print("MERKLE="); print("BLOCK=-1"); print("ENQUEUED=0"); print("ERROR=CCNotDerived"); return 1

    # Canonicals
    ax1_norm=_canon_eq("a","b"); ax2_norm=_canon_eq("b","c")
    s1_norm=_canon_eq("a","c");  s2_norm=_canon_eq("f(a)","f(c)")

    proofs_inserted=0; merkle_root=""; block_number=-1; enqueued=0

    try:
        with psycopg.connect(db, connect_timeout=5) as conn, conn.cursor() as cur:
            # ensure parent-edge table
            cur.execute("""
              CREATE TABLE IF NOT EXISTS proof_parents(
                child_hash  VARCHAR(64) NOT NULL,
                parent_hash VARCHAR(64) NOT NULL,
                created_at  TIMESTAMPTZ DEFAULT now()
              );
              CREATE INDEX IF NOT EXISTS idx_pp_child  ON proof_parents(child_hash);
              CREATE INDEX IF NOT EXISTS idx_pp_parent ON proof_parents(parent_hash);
            """)

            system_id=_ensure_system(cur,"fol_eq")

            # --- Axioms as statements + proofs ---
            ax1_id, ax1_h = _upsert_statement(cur, system_id, "a = b", ax1_norm)
            ax2_id, ax2_h = _upsert_statement(cur, system_id, "b = c", ax2_norm)
            _insert_proof(cur, ax1_id, method="axiom", status="success")
            _insert_proof(cur, ax2_id, method="axiom", status="success")

            # --- Derived equalities ---
            sid1, h1 = _upsert_statement(cur, system_id, "a = c", s1_norm)
            sid2, h2 = _upsert_statement(cur, system_id, "f(a) = f(c)", s2_norm)
            _insert_proof(cur, sid1, method="cc", status="success")
            _insert_proof(cur, sid2, method="cc", status="success")
            proofs_inserted = 4  # 2 axioms + 2 derived

            # --- Parent edges: a=c <= {a=b, b=c};  f(a)=f(c) <= {a=c} ---
            cur.execute("INSERT INTO proof_parents(child_hash,parent_hash) VALUES (%s,%s) ON CONFLICT DO NOTHING", (h1, ax1_h))
            cur.execute("INSERT INTO proof_parents(child_hash,parent_hash) VALUES (%s,%s) ON CONFLICT DO NOTHING", (h1, ax2_h))
            cur.execute("INSERT INTO proof_parents(child_hash,parent_hash) VALUES (%s,%s) ON CONFLICT DO NOTHING", (h2, h1))

            # Seal & persist block
            leafs=[{"statement_hash":h1,"method":"cc","status":"success"},
                   {"statement_hash":h2,"method":"cc","status":"success"}]
            blk=seal_block("fol_eq", leafs); merkle_root=blk.get("merkle_root","") or ""
            block_number=_persist_block(cur, merkle_root, leafs)

            # Enqueue
            rc=_ensure_redis()
            if rc:
                try:
                    rc.rpush("ml:jobs", json.dumps({"text":"a = c","theory":"FOL="}))
                    rc.rpush("ml:jobs", json.dumps({"text":"f(a) = f(c)","theory":"FOL="}))
                    enqueued=2
                except Exception: enqueued=0

            conn.commit()

    except Exception as e:
        print(f"PROOFS_INSERTED={proofs_inserted}"); print(f"MERKLE={merkle_root}"); print(f"BLOCK={block_number}")
        print(f"ENQUEUED={enqueued}"); print(f"ERROR={type(e).__name__}"); return 1

    print(f"PROOFS_INSERTED={proofs_inserted}"); print(f"MERKLE={merkle_root}")
    print(f"BLOCK={block_number}"); print(f"ENQUEUED={enqueued}")
    return 0

if __name__=="__main__":
    try: raise SystemExit(main())
    except SystemExit as e:
        if e.code not in (0,1): print("ERROR=Unknown")
